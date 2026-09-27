// Linear-elastic voxel finite-element solver.
//
// The part is voxelized into a regular grid of cubic 8-node elements. Each element has
// a density in (0, 1] (its filled volume fraction) that scales its stiffness, which
// smooths out the staircase along curved or inclined surfaces.
//
// K u = f is solved matrix-free with conjugate gradients preconditioned by a geometric
// multigrid V-cycle with degree-2 Chebyshev smoothing. Coarse operators are exact Galerkin
// products (P^T K P) computed element-by-element, which keeps the iteration count low
// (typically 10-50) even for long slender parts in bending, where plain Jacobi-CG would need
// thousands. Matrix products gather each node's rows from its 8 voxels; inside a uniform region
// (the 8 voxels the same s * Kb, the level's uniform element) they use the assembled 27-point
// stencil, 243 multiply-adds instead of 576 (the native app's scheme).
//
// Units: the solver works in a normalized system (E = 1, voxel size = 1). With
// physical E [Pa] and voxel size h [m], u_physical = u_normalized / (E * h) for forces in N.

import { HEX_NODES, hexElement } from './hex8.js';

const COARSEST_MAX_DOF = 1100;

// corner index of the element node at unit offset (x, y, z)
const corner = (x, y, z) => (y ? (x ? 2 : 3) : (x ? 1 : 0)) + 4 * z;

/** The smoother estimate's Lanczos start vector on a level: pseudo-random, unit length, 0 where held. */
export function lanczosStart(L) {
  const n = L.nDof, v = new Float64Array(n);
  let seed = 12345, nv = 0;
  for (let i = 0; i < n; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    v[i] = L.fixed[i] ? 0 : seed / 0x7fffffff - 0.5;
    nv += v[i] * v[i];
  }
  nv = Math.sqrt(nv) || 1;
  for (let i = 0; i < n; i++) v[i] /= nv;
  return v;
}

/** Largest eigenvalue of a symmetric tridiagonal matrix (diagonal a, off-diagonal b), by bisection. */
export function tridiagonalMax(a, b) {
  const k = a.length;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < k; i++) {
    const r = (i > 0 ? Math.abs(b[i - 1]) : 0) + (i + 1 < k ? Math.abs(b[i]) : 0);
    lo = Math.min(lo, a[i] - r);
    hi = Math.max(hi, a[i] + r);
  }
  const above = (x) => {
    let count = 0, q = 1;
    for (let i = 0; i < k; i++) {
      q = a[i] - x - (i > 0 ? (b[i - 1] * b[i - 1]) / q : 0);
      if (q === 0) q = 1e-300;
      if (q > 0) count++;
    }
    return count;
  };
  for (let it = 0; it < 100 && hi - lo > 1e-12 * Math.abs(hi); it++) {
    const mid = 0.5 * (lo + hi);
    if (above(mid) > 0) lo = mid;
    else hi = mid;
  }
  return hi;
}

/**
 * A level's uniform element Kb (K0 on the finest level, then the exact Galerkin product of 8 uniform
 * children - not 2 K0, the element is not a plain trilinear one) and its assembled stencil: for each
 * neighbour row (oy, oz) of three nodes (ox = 0..2, 9 contiguous DOFs), the 3 x 9 block on them.
 */
function setBase(L, K) {
  L.Kb = Float64Array.from(K);
  L.S = new Float64Array(243);
  for (let v = 0; v < 8; v++) {
    const di = v & 1, dj = (v >> 1) & 1, dk = v >> 2, a = corner(1 - di, 1 - dj, 1 - dk);
    for (let w = 0; w < 8; w++) {
      const bx = w & 1, by = (w >> 1) & 1, bz = w >> 2, b = corner(bx, by, bz);
      const row = dj + by + 3 * (dk + bz), col = 3 * (di + bx);
      for (let d = 0; d < 3; d++) for (let c = 0; c < 3; c++) L.S[(row * 3 + d) * 9 + col + c] += K[(3 * a + d) * 24 + 3 * b + c];
    }
  }
}

/** Degree-2 Chebyshev smoother over [0.1, 1.15] * lmax of D^-1 K (Adams et al. 2003). */
function chebyshevCoefficients(lmax) {
  const b = 1.15 * lmax, a = 0.1 * lmax;
  const theta = (b + a) / 2, delta = (b - a) / 2, sigma = theta / delta;
  const rho0 = 1 / sigma, rho1 = 1 / (2 * sigma - rho0);
  return { first: 1 / theta, c1: rho1 * rho0, c2: (2 * rho1) / delta };
}

// Trilinear prolongation from a coarse element to each of its 8 children.
// CHILD_P[c][a] = [b0, w0, b1, w1, ...]: fine child-node a interpolates coarse nodes b with weight w.
const CHILD_P = (() => {
  const all = [];
  for (let c = 0; c < 8; c++) {
    const ci = c & 1, cj = (c >> 1) & 1, ck = (c >> 2) & 1;
    const rows = [];
    for (let a = 0; a < 8; a++) {
      const sx = (ci + HEX_NODES[a][0]) / 2;
      const sy = (cj + HEX_NODES[a][1]) / 2;
      const sz = (ck + HEX_NODES[a][2]) / 2;
      const list = [];
      for (let b = 0; b < 8; b++) {
        const [bx, by, bz] = HEX_NODES[b];
        const w = (bx ? sx : 1 - sx) * (by ? sy : 1 - sy) * (bz ? sz : 1 - sz);
        if (w > 1e-12) list.push(b, w);
      }
      rows.push(list);
    }
    all.push(rows);
  }
  return all;
})();

// out[outOff..] += scale * P^T * K[kOff..] * P for one child.
function galerkinAdd(out, outOff, K, kOff, scale, Prow, T) {
  T.fill(0);
  for (let a = 0; a < 8; a++) {
    const list = Prow[a];
    for (let t = 0; t < list.length; t += 2) {
      const b3 = 3 * list[t], w = list[t + 1], a3 = 3 * a;
      for (let r = 0; r < 24; r++) {
        const kr = kOff + r * 24 + a3, tr = r * 24 + b3;
        T[tr] += K[kr] * w;
        T[tr + 1] += K[kr + 1] * w;
        T[tr + 2] += K[kr + 2] * w;
      }
    }
  }
  for (let a = 0; a < 8; a++) {
    const list = Prow[a];
    for (let t = 0; t < list.length; t += 2) {
      const b3 = 3 * list[t], ws = list[t + 1] * scale, a3 = 3 * a;
      for (let d = 0; d < 3; d++) {
        const o = outOff + (b3 + d) * 24, tr = (a3 + d) * 24;
        for (let c = 0; c < 24; c++) out[o + c] += ws * T[tr + c];
      }
    }
  }
}

function makeLevel(nx, ny, nz) {
  const NX = nx + 1, NY = ny + 1, NZ = nz + 1;
  const NXY = NX * NY;
  const nNodes = NXY * NZ;
  return {
    nx, ny, nz, NX, NY, NZ, nNodes, nDof: 3 * nNodes,
    off: Int32Array.of(0, 1, 1 + NX, NX, NXY, 1 + NXY, 1 + NX + NXY, NX + NXY),
    elems: null, base: null, rho: null, K: null,
    fixed: null, bc: null, invDiag: null, omega: 0.5,
    r: null, z: null, t: null,
  };
}

function transferMap(nFineElems) {
  const n = nFineElems + 1;
  const c0 = new Int32Array(n), c1 = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) { c0[i] = i / 2; c1[i] = -1; }
    else { c0[i] = (i - 1) / 2; c1[i] = (i + 1) / 2; }
  }
  return { c0, c1 };
}

export class VoxelFEA {
  /**
   * @param {object} o
   * @param {number[]} o.dims       [nx, ny, nz] element counts
   * @param {Float32Array} o.density per element (index i + nx*(j + ny*k)); 0 = empty
   * @param {number} o.nu           Poisson's ratio
   * @param {Uint8Array} o.bc       per DOF (3 per node, node index i + (nx+1)*(j + (ny+1)*k)); 1 = held at zero
   * @param {number} [o.coarsestMaxDof] coarsen until the last level has at most this many free DOFs
   * @param {Float64Array} [o.diagAdd] per DOF, added to the diagonal of K (a lumped mass shift
   *        K + s*M, or ground springs). Coarse levels get its lumped restriction.
   */
  constructor({ dims, density, nu, bc, coarsestMaxDof = COARSEST_MAX_DOF, diagAdd = null }) {
    if (!dims || dims.length !== 3 || dims.some((n) => !Number.isInteger(n) || n < 1)) {
      throw new Error('The structural grid must have three positive integer dimensions.');
    }
    if (density.length !== dims[0] * dims[1] * dims[2] ||
        bc.length !== 3 * (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1)) {
      throw new Error('Structural grid, density and fixture sizes do not match.');
    }
    for (const rho of density) {
      if (!Number.isFinite(rho) || rho < 0 || rho > 1) throw new Error('Voxel density must be between zero and one.');
    }
    this.options = { dims, density, nu, bc, coarsestMaxDof, diagAdd }; // lets a worker pool rebuild the same model
    this.nu = nu;
    const element = hexElement(nu);
    this.K0 = element.K;
    this.cornerStress = element.cornerStress;
    const T = new Float64Array(576);
    this.ue = new Float64Array(24);
    this.T = T;


    const L0 = makeLevel(dims[0], dims[1], dims[2]);
    const { nx, ny, NX, NY } = L0;
    let count = 0;
    for (let e = 0; e < density.length; e++) if (density[e] > 0) count++;
    L0.elems = new Int32Array(count);
    L0.base = new Int32Array(count);
    L0.rho = new Float64Array(count);
    L0.scale = L0.rho;
    setBase(L0, this.K0);
    let q = 0;
    for (let k = 0; k < dims[2]; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const e = i + nx * (j + ny * k);
          if (density[e] > 0) {
            L0.elems[q] = e;
            L0.base[q] = i + NX * (j + NY * k);
            L0.rho[q] = density[e];
            q++;
          }
        }
      }
    }
    L0.bc = bc;
    if (diagAdd) {
      if (diagAdd.length !== L0.nDof || diagAdd.some((v) => !(v >= 0))) throw new Error('Diagonal shift must be a nonnegative value per DOF.');
      L0.diagAdd = diagAdd;
    }
    this.finishLevel(L0);
    this.levels = [L0];

    while (true) {
      const L = this.levels[this.levels.length - 1];
      if (L.freeDof <= coarsestMaxDof || this.levels.length >= 12) break;
      if (L.nx <= 1 && L.ny <= 1 && L.nz <= 1) break;
      this.levels.push(this.coarsen(L));
    }
    // the smoothers' eigenvalue estimates come later: prepareSmoothers() before a CPU V-cycle, or the
    // GPU solver, which makes them on the GPU (they are most of the setup time for large models)
    this.buildCoarseSolver(this.levels[this.levels.length - 1]);
  }

  get nNodes() { return this.levels[0].nNodes; }

  finishLevel(L) {
    const active = new Uint8Array(L.nNodes);
    const diag = new Float64Array(L.nDof);
    const nE = L.elems.length;
    for (let e = 0; e < nE; e++) {
      const n0 = L.base[e];
      const K = L.K ? L.K : this.K0;
      const kb = L.K ? e * 576 : 0;
      const s = L.K ? 1 : L.rho[e];
      for (let a = 0; a < 8; a++) {
        const n = n0 + L.off[a];
        active[n] = 1;
        for (let d = 0; d < 3; d++) {
          const r = 3 * a + d;
          diag[3 * n + d] += s * K[kb + r * 25];
        }
      }
    }
    if (L.diagAdd) for (let i = 0; i < L.nDof; i++) diag[i] += L.diagAdd[i];
    L.fixed = new Uint8Array(L.nDof);
    L.invDiag = new Float64Array(L.nDof);
    let free = 0;
    for (let n = 0; n < L.nNodes; n++) {
      for (let d = 0; d < 3; d++) {
        const i = 3 * n + d;
        if (!active[n] || L.bc[i] || !(diag[i] > 0)) L.fixed[i] = 1;
        else { L.invDiag[i] = 1 / diag[i]; free++; }
      }
    }
    L.activeNode = active;
    L.freeDof = free;
    L.r = new Float64Array(L.nDof);
    L.z = new Float64Array(L.nDof);
    L.t = new Float64Array(L.nDof);
    L.d = new Float64Array(L.nDof);
    // voxel -> element; node -> s when its 8 voxels are all the same s * K0 (27-point stencil), else 0
    const { nx, ny, nz, NX, NY } = L;
    L.emap = new Int32Array(nx * ny * nz).fill(-1);
    const vs = new Float64Array(nx * ny * nz);
    for (let e = 0; e < nE; e++) {
      L.emap[L.elems[e]] = e;
      vs[L.elems[e]] = L.scale[e] > 0 ? L.scale[e] : -1;
    }
    L.nodeScale = new Float64Array(L.nNodes);
    for (let k = 1; k < nz; k++) {
      for (let j = 1; j < ny; j++) {
        for (let i = 1; i < nx; i++) {
          const s0 = vs[i - 1 + nx * (j - 1 + ny * (k - 1))];
          let u = s0 > 0;
          for (let v = 1; v < 8 && u; v++) u = vs[i - 1 + (v & 1) + nx * (j - 1 + ((v >> 1) & 1) + ny * (k - 1 + (v >> 2)))] === s0;
          if (u) L.nodeScale[i + NX * (j + NY * k)] = s0;
        }
      }
    }
  }

  coarsen(F) {
    const C = makeLevel((F.nx + 1) >> 1, (F.ny + 1) >> 1, (F.nz + 1) >> 1);
    const cmap = new Int32Array(C.nx * C.ny * C.nz).fill(-1);
    const fineCoarse = new Int32Array(F.elems.length);
    const fineChild = new Uint8Array(F.elems.length);
    const order = [];
    for (let q = 0; q < F.elems.length; q++) {
      const e = F.elems[q];
      const i = e % F.nx, j = ((e / F.nx) | 0) % F.ny, k = (e / (F.nx * F.ny)) | 0;
      const ce = (i >> 1) + C.nx * ((j >> 1) + C.ny * (k >> 1));
      if (cmap[ce] < 0) { cmap[ce] = order.length; order.push(ce); }
      fineCoarse[q] = cmap[ce];
      fineChild[q] = (i & 1) | ((j & 1) << 1) | ((k & 1) << 2);
    }
    const nE = order.length;
    C.elems = Int32Array.from(order);
    C.base = new Int32Array(nE);
    for (let q = 0; q < nE; q++) {
      const ce = C.elems[q];
      const I = ce % C.nx, J = ((ce / C.nx) | 0) % C.ny, K = (ce / (C.nx * C.ny)) | 0;
      C.base[q] = I + C.NX * (J + C.NY * K);
    }
    C.K = new Float64Array(nE * 576);
    // Galerkin products of the fine level's uniform element per child position; a coarse element
    // whose 8 children are s * Kb is s times their sum, the coarse level's own uniform element
    const M = [], Kc = new Float64Array(576);
    for (let c = 0; c < 8; c++) {
      const Mc = new Float64Array(576);
      galerkinAdd(Mc, 0, F.Kb, 0, 1, CHILD_P[c], this.T);
      for (let t = 0; t < 576; t++) Kc[t] += Mc[t];
      M.push(Mc);
    }
    setBase(C, Kc);
    const kids = new Int32Array(nE), first = new Float64Array(nE).fill(NaN);
    C.scale = new Float64Array(nE);
    for (let q = 0; q < F.elems.length; q++) {
      const cq = fineCoarse[q], out = cq * 576, c = fineChild[q];
      kids[cq]++;
      const sq = F.scale[q] > 0 ? F.scale[q] : -1;
      if (Number.isNaN(first[cq])) first[cq] = sq;
      else if (first[cq] !== sq) first[cq] = -1;
      if (F.scale[q] > 0) {
        const Mc = M[c], s = F.scale[q];
        for (let t = 0; t < 576; t++) C.K[out + t] += s * Mc[t];
      } else {
        galerkinAdd(C.K, out, F.K, q * 576, 1, CHILD_P[c], this.T);
      }
    }
    for (let q = 0; q < nE; q++) C.scale[q] = kids[q] === 8 && first[q] > 0 ? first[q] : -1;
    // A coarse DOF is held if it interpolates onto any held fine DOF.
    const mx = transferMap(F.nx), my = transferMap(F.ny), mz = transferMap(F.nz);
    C.bc = new Uint8Array(C.nDof);
    for (let k = 0; k < F.NZ; k++) {
      for (let j = 0; j < F.NY; j++) {
        for (let i = 0; i < F.NX; i++) {
          const fn = 3 * (i + F.NX * (j + F.NY * k));
          if (!(F.bc[fn] | F.bc[fn + 1] | F.bc[fn + 2])) continue;
          for (const K of [mz.c0[k], mz.c1[k]]) {
            if (K < 0) continue;
            for (const J of [my.c0[j], my.c1[j]]) {
              if (J < 0) continue;
              for (const I of [mx.c0[i], mx.c1[i]]) {
                if (I < 0) continue;
                const cn = 3 * (I + C.NX * (J + C.NY * K));
                C.bc[cn] |= F.bc[fn];
                C.bc[cn + 1] |= F.bc[fn + 1];
                C.bc[cn + 2] |= F.bc[fn + 2];
              }
            }
          }
        }
      }
    }
    F.maps = { mx, my, mz };
    if (F.diagAdd) {
      // lumped Galerkin product of a diagonal: row sums of P^T D P = P^T (D 1)
      C.diagAdd = new Float64Array(C.nDof);
      this.restrict(F, C, F.diagAdd, C.diagAdd, false);
    }
    this.finishLevel(C);
    return C;
  }

  /** y = K x on free DOFs (held/inactive DOFs of y are zeroed unless raw). */
  apply(L, x, y, raw = false) {
    const { NX, NY, NZ, nx, ny, nz, off, emap, nodeScale, activeNode, fixed } = L;
    const NXY = NX * NY, S0 = L.S, D = L.diagAdd;
    const Kmat = L.K || this.K0, own = !!L.K;
    for (let k = 0; k < NZ; k++) {
      for (let j = 0; j < NY; j++) {
        for (let i = 0; i < NX; i++) {
          const n = i + NX * (j + NY * k), o = 3 * n;
          let a0 = 0, a1 = 0, a2 = 0;
          if (activeNode[n]) {
            const s0 = nodeScale[n];
            if (s0 > 0) {
              // 27-point stencil: nine rows of three neighbours, 9 contiguous DOFs each
              const b = o - 3 * (1 + NX + NXY);
              for (let r = 0; r < 9; r++) {
                const xr = b + 3 * ((r % 3) * NX + ((r / 3) | 0) * NXY), sr = r * 27;
                for (let c = 0; c < 9; c++) {
                  const u = x[xr + c];
                  a0 += S0[sr + c] * u;
                  a1 += S0[sr + 9 + c] * u;
                  a2 += S0[sr + 18 + c] * u;
                }
              }
              a0 *= s0; a1 *= s0; a2 *= s0;
            } else {
              // the rows of this node's corner in each of its voxels
              for (let v = 0; v < 8; v++) {
                const di = v & 1, dj = (v >> 1) & 1, dk = v >> 2;
                const ei = i + di - 1, ej = j + dj - 1, ek = k + dk - 1;
                if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= nz) continue;
                const e = emap[ei + nx * (ej + ny * ek)];
                if (e < 0) continue;
                const nb = n - (1 - di) - NX * (1 - dj) - NXY * (1 - dk);
                const kb = (own ? e * 576 : 0) + 72 * corner(1 - di, 1 - dj, 1 - dk);
                let s0_ = 0, s1 = 0, s2 = 0;
                for (let c = 0; c < 8; c++) {
                  const p = 3 * (nb + off[c]), kc = kb + 3 * c;
                  const u0 = x[p], u1 = x[p + 1], u2 = x[p + 2];
                  s0_ += Kmat[kc] * u0 + Kmat[kc + 1] * u1 + Kmat[kc + 2] * u2;
                  s1 += Kmat[kc + 24] * u0 + Kmat[kc + 25] * u1 + Kmat[kc + 26] * u2;
                  s2 += Kmat[kc + 48] * u0 + Kmat[kc + 49] * u1 + Kmat[kc + 50] * u2;
                }
                const s = own ? 1 : L.rho[e];
                a0 += s * s0_; a1 += s * s1; a2 += s * s2;
              }
            }
            if (D) { a0 += D[o] * x[o]; a1 += D[o + 1] * x[o + 1]; a2 += D[o + 2] * x[o + 2]; }
          }
          if (raw) { y[o] = a0; y[o + 1] = a1; y[o + 2] = a2; }
          else {
            y[o] = fixed[o] ? 0 : a0;
            y[o + 1] = fixed[o + 1] ? 0 : a1;
            y[o + 2] = fixed[o + 2] ? 0 : a2;
          }
        }
      }
    }
  }

  /** Estimates the smoothers' eigenvalue ranges where they are not known yet. */
  prepareSmoothers() {
    for (let l = 0; l < this.levels.length - 1; l++) if (!(this.levels[l].lmax > 0)) this.estimateOmega(this.levels[l]);
  }

  /** Largest eigenvalue of D^-1 K on a level: 10 Lanczos steps on D^-1/2 K D^-1/2. */
  estimateOmega(L) {
    const n = L.nDof, sq = new Float64Array(n);
    for (let i = 0; i < n; i++) sq[i] = Math.sqrt(L.invDiag[i]);
    let v = lanczosStart(L), vPrev = new Float64Array(n);
    const w = new Float64Array(n), x = new Float64Array(n);
    const alpha = [], beta = [];
    let b = 0;
    for (let j = 0; j < 10; j++) {
      for (let i = 0; i < n; i++) x[i] = sq[i] * v[i];
      this.apply(L, x, w);
      for (let i = 0; i < n; i++) w[i] = sq[i] * w[i] - b * vPrev[i];
      const a = dot(w, v);
      for (let i = 0; i < n; i++) w[i] -= a * v[i];
      alpha.push(a);
      b = Math.sqrt(dot(w, w));
      if (!(b > 1e-12 * Math.abs(a))) break;
      beta.push(b);
      [vPrev, v] = [v, vPrev];
      for (let i = 0; i < n; i++) v[i] = w[i] / b;
    }
    beta.length = alpha.length - 1;
    L.lmax = tridiagonalMax(alpha, beta);
    L.omega = 1.2 / (L.lmax * 1.05); // damped Jacobi (coarsest-level fallback, GPU solver)
  }

  buildCoarseSolver(L) {
    const map = new Int32Array(L.nDof).fill(-1);
    let m = 0;
    for (let i = 0; i < L.nDof; i++) if (!L.fixed[i]) map[i] = m++;
    this.coarse = { map, m, A: null };
    if (m === 0 || m > 3000) return; // falls back to Jacobi sweeps
    const A = new Float64Array(m * m);
    const dofs = new Int32Array(24);
    for (let e = 0; e < L.elems.length; e++) {
      const n0 = L.base[e];
      for (let a = 0; a < 8; a++) for (let d = 0; d < 3; d++) dofs[3 * a + d] = map[3 * (n0 + L.off[a]) + d];
      const K = L.K ? L.K : this.K0;
      const kb = L.K ? e * 576 : 0;
      const s = L.K ? 1 : L.rho[e];
      for (let r = 0; r < 24; r++) {
        const gr = dofs[r];
        if (gr < 0) continue;
        for (let c = 0; c < 24; c++) {
          const gc = dofs[c];
          if (gc < 0) continue;
          A[gr * m + gc] += s * K[kb + r * 24 + c];
        }
      }
    }
    if (L.diagAdd) for (let i = 0; i < L.nDof; i++) if (map[i] >= 0) A[map[i] * m + map[i]] += L.diagAdd[i];
    let maxDiag = 0;
    for (let i = 0; i < m; i++) maxDiag = Math.max(maxDiag, A[i * m + i]);
    // In-place Cholesky (lower triangle, row-major).
    for (let j = 0; j < m; j++) {
      const rj = j * m;
      let s = A[rj + j];
      for (let k = 0; k < j; k++) s -= A[rj + k] * A[rj + k];
      if (!(s > 1e-10 * maxDiag)) s = maxDiag; // regularize near-mechanisms at the coarsest level
      const ljj = Math.sqrt(s);
      A[rj + j] = ljj;
      for (let i = j + 1; i < m; i++) {
        const ri = i * m;
        let t = A[ri + j];
        for (let k = 0; k < j; k++) t -= A[ri + k] * A[rj + k];
        A[ri + j] = t / ljj;
      }
    }
    this.coarse.A = A;
    this.coarse.y = new Float64Array(m);
  }

  coarseSolve(L) {
    const { map, m, A, y } = this.coarse;
    const r = L.r, z = L.z;
    z.fill(0);
    if (!A) {
      for (let s = 0; s < 40; s++) this.jacobiSweep(L, z, s === 0);
      return;
    }
    for (let i = 0; i < L.nDof; i++) if (map[i] >= 0) y[map[i]] = r[i];
    for (let i = 0; i < m; i++) {
      const ri = i * m;
      let s = y[i];
      for (let k = 0; k < i; k++) s -= A[ri + k] * y[k];
      y[i] = s / A[ri + i];
    }
    // backward by column sweeps: each reads a row of the factor
    for (let i = m - 1; i >= 0; i--) {
      const ri = i * m, xi = y[i] / A[ri + i];
      y[i] = xi;
      for (let k = 0; k < i; k++) y[k] -= A[ri + k] * xi;
    }
    for (let i = 0; i < L.nDof; i++) if (map[i] >= 0) z[i] = y[map[i]];
  }

  /** Degree-2 Chebyshev smoothing of L.z (from zero, or from its current value). */
  chebyshev(L, fromZero) {
    const { r, z, t, d, invDiag: Di } = L;
    const c = chebyshevCoefficients(L.lmax), n = z.length;
    if (fromZero) {
      for (let i = 0; i < n; i++) { d[i] = c.first * Di[i] * r[i]; z[i] = d[i]; }
    } else {
      this.apply(L, z, t);
      for (let i = 0; i < n; i++) { d[i] = c.first * Di[i] * (r[i] - t[i]); z[i] += d[i]; }
    }
    this.apply(L, z, t);
    for (let i = 0; i < n; i++) { d[i] = c.c1 * d[i] + c.c2 * Di[i] * (r[i] - t[i]); z[i] += d[i]; }
  }

  jacobiSweep(L, z, fromZero) {
    const { r, t, invDiag, omega } = L;
    if (fromZero) {
      for (let i = 0; i < z.length; i++) z[i] = omega * invDiag[i] * r[i];
      return;
    }
    this.apply(L, z, t);
    for (let i = 0; i < z.length; i++) z[i] += omega * invDiag[i] * (r[i] - t[i]);
  }

  restrict(F, C, rf, rc, zeroFixed = true) {
    rc.fill(0);
    const { mx, my, mz } = F.maps;
    for (let k = 0; k < F.NZ; k++) {
      const k0 = mz.c0[k], k1 = mz.c1[k], wk = k1 < 0 ? 1 : 0.5;
      for (let j = 0; j < F.NY; j++) {
        const j0 = my.c0[j], j1 = my.c1[j], wj = j1 < 0 ? 1 : 0.5;
        for (let i = 0; i < F.NX; i++) {
          const fn = 3 * (i + F.NX * (j + F.NY * k));
          const vx = rf[fn], vy = rf[fn + 1], vz = rf[fn + 2];
          if (vx === 0 && vy === 0 && vz === 0) continue;
          const i0 = mx.c0[i], i1 = mx.c1[i], wi = i1 < 0 ? 1 : 0.5;
          for (let kk = 0; kk < 2; kk++) {
            const K = kk ? k1 : k0;
            if (K < 0) continue;
            for (let jj = 0; jj < 2; jj++) {
              const J = jj ? j1 : j0;
              if (J < 0) continue;
              for (let ii = 0; ii < 2; ii++) {
                const I = ii ? i1 : i0;
                if (I < 0) continue;
                const w = wi * wj * wk;
                const cn = 3 * (I + C.NX * (J + C.NY * K));
                rc[cn] += w * vx;
                rc[cn + 1] += w * vy;
                rc[cn + 2] += w * vz;
              }
            }
          }
        }
      }
    }
    if (zeroFixed) for (let i = 0; i < rc.length; i++) if (C.fixed[i]) rc[i] = 0;
  }

  prolongAdd(F, C, zc, zf) {
    const { mx, my, mz } = F.maps;
    for (let k = 0; k < F.NZ; k++) {
      const k0 = mz.c0[k], k1 = mz.c1[k], wk = k1 < 0 ? 1 : 0.5;
      for (let j = 0; j < F.NY; j++) {
        const j0 = my.c0[j], j1 = my.c1[j], wj = j1 < 0 ? 1 : 0.5;
        for (let i = 0; i < F.NX; i++) {
          const fn = 3 * (i + F.NX * (j + F.NY * k));
          if (F.fixed[fn] && F.fixed[fn + 1] && F.fixed[fn + 2]) continue;
          const i0 = mx.c0[i], i1 = mx.c1[i], wi = i1 < 0 ? 1 : 0.5;
          let sx = 0, sy = 0, sz = 0;
          for (let kk = 0; kk < 2; kk++) {
            const K = kk ? k1 : k0;
            if (K < 0) continue;
            for (let jj = 0; jj < 2; jj++) {
              const J = jj ? j1 : j0;
              if (J < 0) continue;
              for (let ii = 0; ii < 2; ii++) {
                const I = ii ? i1 : i0;
                if (I < 0) continue;
                const w = wi * wj * wk;
                const cn = 3 * (I + C.NX * (J + C.NY * K));
                sx += w * zc[cn];
                sy += w * zc[cn + 1];
                sz += w * zc[cn + 2];
              }
            }
          }
          if (!F.fixed[fn]) zf[fn] += sx;
          if (!F.fixed[fn + 1]) zf[fn + 1] += sy;
          if (!F.fixed[fn + 2]) zf[fn + 2] += sz;
        }
      }
    }
  }

  /** z = one multigrid V-cycle applied to the full-length vector r (its held DOFs are ignored). */
  precondition(r, z = new Float64Array(r.length)) {
    const L = this.levels[0];
    this.prepareSmoothers();
    L.r.set(r);
    for (let i = 0; i < r.length; i++) if (L.fixed[i]) L.r[i] = 0;
    this.vcycle(0);
    z.set(L.z);
    return z;
  }

  vcycle(l) {
    const L = this.levels[l];
    if (l === this.levels.length - 1) { this.coarseSolve(L); return; }
    const C = this.levels[l + 1];
    const { r, z, t } = L;
    this.chebyshev(L, true);
    this.apply(L, z, t);
    for (let i = 0; i < t.length; i++) t[i] = r[i] - t[i];
    this.restrict(L, C, t, C.r);
    this.vcycle(l + 1);
    this.prolongAdd(L, C, C.z, z);
    this.chebyshev(L, false);
  }

  /**
   * Solve K u = f in normalized units. Returns the normalized displacement.
   * @param {Float64Array} f
   * @param {{tol?: number, maxIter?: number, x0?: Float64Array, onProgress?: (it: number, res: number) => boolean|void}} opts
   *        onProgress may return true to cancel.
   */
  solve(f, { tol = 1e-6, maxIter = 500, x0 = null, onProgress = null } = {}) {
    const L = this.levels[0];
    const n = L.nDof, fixed = L.fixed;
    if (f.length !== n || (x0 && x0.length !== n)) throw new Error('Force or displacement vector has the wrong size.');
    if (!(tol > 0) || !Number.isFinite(tol) || !Number.isInteger(maxIter) || maxIter < 1) throw new Error('Invalid solver tolerance or iteration limit.');
    for (let i = 0; i < n; i++) {
      if (!Number.isFinite(f[i]) || (x0 && !Number.isFinite(x0[i]))) throw new Error('Forces and displacements must be finite numbers.');
    }
    const x = x0 ? Float64Array.from(x0) : new Float64Array(n);
    const r = new Float64Array(n), p = new Float64Array(n), q = new Float64Array(n);
    for (let i = 0; i < n; i++) if (fixed[i]) x[i] = 0;
    let bnorm = 0;
    for (let i = 0; i < n; i++) if (!fixed[i]) bnorm += f[i] * f[i];
    bnorm = Math.sqrt(bnorm);
    if (bnorm === 0) return { u: x.fill(0), iterations: 0, residual: 0, converged: true };
    this.apply(L, x, q);
    for (let i = 0; i < n; i++) r[i] = fixed[i] ? 0 : f[i] - q[i];
    this.prepareSmoothers();
    const precond = () => {
      L.r.set(r);
      this.vcycle(0);
      return L.z;
    };
    let z = precond();
    p.set(z);
    let rz = dot(r, z);
    let res = Math.sqrt(dot(r, r)) / bnorm;
    let it = 0;
    let cancelled = false;
    for (; it < maxIter && res > tol; it++) {
      this.apply(L, p, q);
      const pq = dot(p, q);
      if (!(pq > 0)) break; // loss of positive-definiteness (mechanism)
      const alpha = rz / pq;
      for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
      res = Math.sqrt(dot(r, r)) / bnorm;
      if (onProgress && onProgress(it + 1, res) === true) { cancelled = true; it++; break; }
      if (res <= tol) { it++; break; }
      z = precond();
      const rzNew = dot(r, z);
      const beta = rzNew / rz;
      rz = rzNew;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    }
    // The recursively updated residual can drift, especially near a mechanism.
    // Report convergence against the actual equilibrium equations.
    this.apply(L, x, q);
    for (let i = 0; i < n; i++) r[i] = fixed[i] ? 0 : f[i] - q[i];
    res = Math.sqrt(dot(r, r)) / bnorm;
    return { u: x, iterations: it, residual: res, converged: !cancelled && Number.isFinite(res) && res <= tol * 10, cancelled };
  }

  /** Reaction forces (normalized system, equals Newtons when f was in Newtons) summed over held DOFs. */
  reactions(u, f = null) {
    const L = this.levels[0];
    const y = new Float64Array(L.nDof);
    this.apply(L, u, y, true);
    const R = [0, 0, 0];
    for (let n = 0; n < L.nNodes; n++) {
      for (let d = 0; d < 3; d++) if (L.bc[3 * n + d] && L.activeNode[n]) {
        const i = 3 * n + d;
        R[d] += y[i] - (f ? f[i] : 0);
      }
    }
    return R;
  }

  /**
   * Stresses from physical displacements u [m], modulus E [Pa] and voxel size h [m].
   * Nodal values average the corner stresses of the surrounding voxels (weighted by density),
   * which recovers surface stresses much better than centroid values.
   * Returns Pa. elemVM is each voxel's centroid von Mises stress.
   */
  stresses(u, E, h) {
    const L = this.levels[0];
    const S = this.cornerStress.map((m) => m.map((v) => (v * E) / h));
    const nE = L.elems.length;
    const elemVM = new Float32Array(nE);
    const nodeVM = new Float32Array(L.nNodes), nodeP1 = new Float32Array(L.nNodes), nodeP3 = new Float32Array(L.nNodes);
    const nodeW = new Float32Array(L.nNodes);
    const ue = this.ue, pr = [0, 0, 0];
    const sig = new Float64Array(6);
    for (let e = 0; e < nE; e++) {
      const n0 = L.base[e];
      for (let a = 0; a < 8; a++) {
        const n = 3 * (n0 + L.off[a]);
        ue[3 * a] = u[n];
        ue[3 * a + 1] = u[n + 1];
        ue[3 * a + 2] = u[n + 2];
      }
      const w = L.rho[e];
      const c = [0, 0, 0, 0, 0, 0];
      for (let a = 0; a < 8; a++) {
        const Sa = S[a];
        for (let i = 0; i < 6; i++) {
          let s = 0;
          for (let q = 0; q < 24; q++) s += Sa[i * 24 + q] * ue[q];
          sig[i] = s;
          c[i] += s / 8;
        }
        const vm = vonMises(sig);
        principalStresses(sig[0], sig[1], sig[2], sig[3], sig[4], sig[5], pr);
        const n = n0 + L.off[a];
        nodeVM[n] += w * vm;
        nodeP1[n] += w * pr[0];
        nodeP3[n] += w * pr[2];
        nodeW[n] += w;
      }
      elemVM[e] = vonMises(c);
    }
    for (let n = 0; n < L.nNodes; n++) {
      if (nodeW[n] > 0) {
        nodeVM[n] /= nodeW[n];
        nodeP1[n] /= nodeW[n];
        nodeP3[n] /= nodeW[n];
      }
    }
    return { elemVM, nodeVM, nodeP1, nodeP3, elems: L.elems };
  }

  /**
   * Nodal stress tensors (6 per node: xx, yy, zz, xy, yz, zx; Pa) from physical displacements,
   * averaged over the corners of the surrounding voxels like stresses(). Linear in u, so modal
   * and load-case results can be superposed tensor by tensor.
   */
  stressTensors(u, E, h) {
    const L = this.levels[0];
    const S = this.cornerStress.map((m) => m.map((v) => (v * E) / h));
    const out = new Float32Array(6 * L.nNodes);
    const W = new Float32Array(L.nNodes);
    const ue = this.ue;
    for (let e = 0; e < L.elems.length; e++) {
      const n0 = L.base[e];
      for (let a = 0; a < 8; a++) {
        const n = 3 * (n0 + L.off[a]);
        ue[3 * a] = u[n];
        ue[3 * a + 1] = u[n + 1];
        ue[3 * a + 2] = u[n + 2];
      }
      const w = L.rho[e];
      for (let a = 0; a < 8; a++) {
        const Sa = S[a], n = n0 + L.off[a];
        for (let i = 0; i < 6; i++) {
          let s = 0;
          for (let q = 0; q < 24; q++) s += Sa[i * 24 + q] * ue[q];
          out[6 * n + i] += w * s;
        }
        W[n] += w;
      }
    }
    for (let n = 0; n < L.nNodes; n++) if (W[n] > 0) for (let i = 0; i < 6; i++) out[6 * n + i] /= W[n];
    return out;
  }
}

/** von Mises stress of a Voigt tensor (xx, yy, zz, xy, yz, zx) stored at s[o..o+5]. */
export function vonMisesAt(s, o = 0) {
  const sx = s[o], sy = s[o + 1], sz = s[o + 2], txy = s[o + 3], tyz = s[o + 4], tzx = s[o + 5];
  return Math.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy * txy + tyz * tyz + tzx * tzx));
}

function vonMises(s) {
  const [sx, sy, sz, txy, tyz, tzx] = s;
  return Math.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (txy * txy + tyz * tyz + tzx * tzx));
}

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Principal stresses of a symmetric 3x3 tensor, sorted descending into out[0..2]. */
export function principalStresses(sx, sy, sz, txy, tyz, tzx, out) {
  const p1 = txy * txy + tyz * tyz + tzx * tzx;
  const q = (sx + sy + sz) / 3;
  const scale = Math.abs(sx) + Math.abs(sy) + Math.abs(sz) + Math.sqrt(p1);
  if (p1 <= 1e-24 * scale * scale) {
    const v = [sx, sy, sz].sort((a, b) => b - a);
    out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
    return out;
  }
  const p2 = (sx - q) ** 2 + (sy - q) ** 2 + (sz - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  const b11 = (sx - q) / p, b22 = (sy - q) / p, b33 = (sz - q) / p;
  const b12 = txy / p, b23 = tyz / p, b13 = tzx / p;
  const det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13);
  const r = Math.min(1, Math.max(-1, det / 2));
  const phi = Math.acos(r) / 3;
  out[0] = q + 2 * p * Math.cos(phi);
  out[2] = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  out[1] = 3 * q - out[0] - out[2];
  return out;
}

/**
 * Drop voxels that are not face-connected to any voxel touching a held node
 * (they would float freely and make the system singular).
 * @returns {{density: Float32Array, removed: number}}
 */
export function pruneFloating(dims, density, heldNode) {
  const [nx, ny, nz] = dims;
  const NX = nx + 1, NY = ny + 1;
  const nE = nx * ny * nz;
  const seen = new Uint8Array(nE);
  const queue = new Int32Array(nE);
  let qh = 0, qt = 0;
  const off = [0, 1, 1 + NX, NX, NX * NY, 1 + NX * NY, 1 + NX + NX * NY, NX + NX * NY];
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const e = i + nx * (j + ny * k);
        if (!(density[e] > 0)) continue;
        const n0 = i + NX * (j + NY * k);
        for (let a = 0; a < 8; a++) {
          if (heldNode[n0 + off[a]]) { seen[e] = 1; queue[qt++] = e; break; }
        }
      }
    }
  }
  while (qh < qt) {
    const e = queue[qh++];
    const i = e % nx, j = ((e / nx) | 0) % ny, k = (e / (nx * ny)) | 0;
    const nb = [
      i > 0 ? e - 1 : -1, i < nx - 1 ? e + 1 : -1,
      j > 0 ? e - nx : -1, j < ny - 1 ? e + nx : -1,
      k > 0 ? e - nx * ny : -1, k < nz - 1 ? e + nx * ny : -1,
    ];
    for (const m of nb) {
      if (m >= 0 && !seen[m] && density[m] > 0) { seen[m] = 1; queue[qt++] = m; }
    }
  }
  const out = new Float32Array(nE);
  let removed = 0;
  for (let e = 0; e < nE; e++) {
    if (density[e] > 0) {
      if (seen[e]) out[e] = density[e];
      else removed++;
    }
  }
  return { density: out, removed };
}
