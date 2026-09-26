// Heat transfer on the voxel model: steady-state and transient conduction with fixed
// temperatures, heat sources and surface convection.
//
// One temperature per grid node, 8-node brick conduction elements (conductivity scaled by each
// voxel's fill fraction), lumped heat capacity, backward-Euler time steps. The linear systems are
// solved with conjugate gradients preconditioned by a scalar geometric multigrid V-cycle with
// Galerkin coarse operators, the same scheme as the structural solver.
//
// Normalized units: the system is divided by k*h (conductivity x voxel size), so temperatures
// come out directly in degrees and a node's convection term is h_c * A / (k h).
import { GEO_T } from './eigen.js';
import { HEX_NODES } from './hex8.js';

const SMOOTH_SWEEPS = 2;
const COARSEST_MAX = 1500;

// conduction matrix of the unit cube with k = 1: integral of grad N_a . grad N_b
export const CONDUCTION_K = GEO_T[0].map((v, i) => v + GEO_T[1][i] + GEO_T[2][i]);

const CHILD_P = (() => {
  const all = [];
  for (let c = 0; c < 8; c++) {
    const ci = c & 1, cj = (c >> 1) & 1, ck = (c >> 2) & 1;
    const rows = [];
    for (let a = 0; a < 8; a++) {
      const sx = (ci + HEX_NODES[a][0]) / 2, sy = (cj + HEX_NODES[a][1]) / 2, sz = (ck + HEX_NODES[a][2]) / 2;
      const w = new Float64Array(8);
      for (let b = 0; b < 8; b++) {
        const [bx, by, bz] = HEX_NODES[b];
        w[b] = (bx ? sx : 1 - sx) * (by ? sy : 1 - sy) * (bz ? sz : 1 - sz);
      }
      rows.push(w);
    }
    all.push(rows);
  }
  return all;
})();

function makeLevel(nx, ny, nz) {
  const NX = nx + 1, NY = ny + 1, NZ = nz + 1, NXY = NX * NY;
  return {
    nx, ny, nz, NX, NY, NZ, nNodes: NXY * NZ,
    off: Int32Array.of(0, 1, 1 + NX, NX, NXY, 1 + NXY, 1 + NX + NXY, NX + NXY),
    elems: null, base: null, rho: null, K: null, fixed: null, diagAdd: null, invDiag: null, omega: 0.6,
  };
}

function transferMap(nFine) {
  const n = nFine + 1, c0 = new Int32Array(n), c1 = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (i % 2 === 0) { c0[i] = i / 2; c1[i] = -1; } else { c0[i] = (i - 1) / 2; c1[i] = (i + 1) / 2; }
  }
  return { c0, c1 };
}

export class ScalarVoxelSolver {
  /**
   * @param {object} o
   * @param {number[]} o.dims          voxel counts
   * @param {Float32Array} o.density   conductivity scale per voxel (fill fraction), 0 = empty
   * @param {Uint8Array} o.fixed       per node, 1 = prescribed temperature
   * @param {Float64Array} [o.diagAdd] per node, added to the diagonal (convection, heat capacity / dt)
   */
  constructor({ dims, density, fixed, diagAdd = null, coarsestMax = COARSEST_MAX }) {
    const [nx, ny, nz] = dims;
    const L0 = makeLevel(nx, ny, nz);
    if (density.length !== nx * ny * nz || fixed.length !== L0.nNodes) throw new Error('Thermal grid sizes do not match.');
    let count = 0;
    for (const d of density) if (d > 0) count++;
    L0.elems = new Int32Array(count);
    L0.base = new Int32Array(count);
    L0.rho = new Float64Array(count);
    let q = 0;
    for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const e = i + nx * (j + ny * k);
      if (density[e] > 0) {
        L0.elems[q] = e;
        L0.base[q] = i + L0.NX * (j + L0.NY * k);
        L0.rho[q] = density[e];
        q++;
      }
    }
    L0.bc = fixed;
    L0.diagAdd = diagAdd;
    this.finishLevel(L0);
    this.levels = [L0];
    while (true) {
      const L = this.levels[this.levels.length - 1];
      if (L.freeDof <= coarsestMax || this.levels.length >= 12 || (L.nx <= 1 && L.ny <= 1 && L.nz <= 1)) break;
      this.levels.push(this.coarsen(L));
    }
    for (let l = 0; l < this.levels.length - 1; l++) this.estimateOmega(this.levels[l]);
    this.buildCoarse(this.levels[this.levels.length - 1]);
  }

  get nNodes() { return this.levels[0].nNodes; }

  finishLevel(L) {
    const active = new Uint8Array(L.nNodes), diag = new Float64Array(L.nNodes);
    for (let e = 0; e < L.elems.length; e++) {
      const n0 = L.base[e];
      for (let a = 0; a < 8; a++) {
        const n = n0 + L.off[a];
        active[n] = 1;
        diag[n] += L.K ? L.K[64 * e + 9 * a] : L.rho[e] * CONDUCTION_K[9 * a];
      }
    }
    if (L.diagAdd) for (let n = 0; n < L.nNodes; n++) diag[n] += L.diagAdd[n];
    L.fixed = new Uint8Array(L.nNodes);
    L.invDiag = new Float64Array(L.nNodes);
    let free = 0;
    for (let n = 0; n < L.nNodes; n++) {
      if (!active[n] || L.bc[n] || !(diag[n] > 0)) L.fixed[n] = 1;
      else { L.invDiag[n] = 1 / diag[n]; free++; }
    }
    L.active = active;
    L.freeDof = free;
    L.r = new Float64Array(L.nNodes);
    L.z = new Float64Array(L.nNodes);
    L.t = new Float64Array(L.nNodes);
  }

  coarsen(F) {
    const C = makeLevel((F.nx + 1) >> 1, (F.ny + 1) >> 1, (F.nz + 1) >> 1);
    const cmap = new Int32Array(C.nx * C.ny * C.nz).fill(-1);
    const order = [], fc = new Int32Array(F.elems.length), child = new Uint8Array(F.elems.length);
    for (let q = 0; q < F.elems.length; q++) {
      const e = F.elems[q];
      const i = e % F.nx, j = ((e / F.nx) | 0) % F.ny, k = (e / (F.nx * F.ny)) | 0;
      const ce = (i >> 1) + C.nx * ((j >> 1) + C.ny * (k >> 1));
      if (cmap[ce] < 0) { cmap[ce] = order.length; order.push(ce); }
      fc[q] = cmap[ce];
      child[q] = (i & 1) | ((j & 1) << 1) | ((k & 1) << 2);
    }
    C.elems = Int32Array.from(order);
    C.base = new Int32Array(order.length);
    for (let q = 0; q < order.length; q++) {
      const ce = order[q];
      C.base[q] = (ce % C.nx) + C.NX * ((((ce / C.nx) | 0) % C.ny) + C.NY * ((ce / (C.nx * C.ny)) | 0));
    }
    C.K = new Float64Array(64 * order.length);
    const T = new Float64Array(64);
    for (let q = 0; q < F.elems.length; q++) {
      const P = CHILD_P[child[q]], o = 64 * fc[q];
      const Kf = F.K ? F.K.subarray(64 * q, 64 * q + 64) : CONDUCTION_K;
      const s = F.K ? 1 : F.rho[q];
      // T = Kf P, then C += s P^T T
      T.fill(0);
      for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) {
        const kab = Kf[a * 8 + b];
        if (kab === 0) continue;
        const pb = P[b];
        for (let c = 0; c < 8; c++) T[a * 8 + c] += kab * pb[c];
      }
      for (let a = 0; a < 8; a++) {
        const pa = P[a];
        for (let d = 0; d < 8; d++) {
          const w = pa[d] * s;
          if (w === 0) continue;
          for (let c = 0; c < 8; c++) C.K[o + d * 8 + c] += w * T[a * 8 + c];
        }
      }
    }
    const mx = transferMap(F.nx), my = transferMap(F.ny), mz = transferMap(F.nz);
    F.maps = { mx, my, mz };
    C.bc = new Uint8Array(C.nNodes);
    for (let k = 0; k < F.NZ; k++) for (let j = 0; j < F.NY; j++) for (let i = 0; i < F.NX; i++) {
      if (!F.bc[i + F.NX * (j + F.NY * k)]) continue;
      for (const K of [mz.c0[k], mz.c1[k]]) if (K >= 0) for (const J of [my.c0[j], my.c1[j]]) if (J >= 0) for (const I of [mx.c0[i], mx.c1[i]]) if (I >= 0) C.bc[I + C.NX * (J + C.NY * K)] = 1;
    }
    if (F.diagAdd) {
      C.diagAdd = new Float64Array(C.nNodes);
      this.restrict(F, C, F.diagAdd, C.diagAdd, false);
    }
    this.finishLevel(C);
    return C;
  }

  apply(L, x, y, raw = false) {
    y.fill(0);
    const off = L.off, K = L.K, xe = new Float64Array(8);
    for (let e = 0; e < L.elems.length; e++) {
      const n0 = L.base[e];
      for (let a = 0; a < 8; a++) xe[a] = x[n0 + off[a]];
      const Ke = K ? K : CONDUCTION_K, o = K ? 64 * e : 0, s = K ? 1 : L.rho[e];
      for (let a = 0; a < 8; a++) {
        let sum = 0;
        for (let b = 0; b < 8; b++) sum += Ke[o + a * 8 + b] * xe[b];
        y[n0 + off[a]] += s * sum;
      }
    }
    if (L.diagAdd) for (let n = 0; n < y.length; n++) y[n] += L.diagAdd[n] * x[n];
    if (!raw) for (let n = 0; n < y.length; n++) if (L.fixed[n]) y[n] = 0;
  }

  estimateOmega(L) {
    let v = new Float64Array(L.nNodes), w = new Float64Array(L.nNodes);
    let seed = 7;
    for (let i = 0; i < v.length; i++) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; v[i] = L.fixed[i] ? 0 : seed / 0x7fffffff - 0.5; }
    let lambda = 1;
    for (let it = 0; it < 12; it++) {
      let nv = 0;
      for (const x of v) nv += x * x;
      nv = Math.sqrt(nv) || 1;
      for (let i = 0; i < v.length; i++) v[i] /= nv;
      this.apply(L, v, w);
      let nw = 0;
      for (let i = 0; i < w.length; i++) { w[i] *= L.invDiag[i]; nw += w[i] * w[i]; }
      lambda = Math.sqrt(nw);
      [v, w] = [w, v];
    }
    L.omega = 1.2 / (lambda * 1.05);
  }

  buildCoarse(L) {
    const map = new Int32Array(L.nNodes).fill(-1);
    let m = 0;
    for (let n = 0; n < L.nNodes; n++) if (!L.fixed[n]) map[n] = m++;
    this.coarse = { map, m, A: null };
    if (m === 0 || m > 4000) return;
    const A = new Float64Array(m * m);
    for (let e = 0; e < L.elems.length; e++) {
      const n0 = L.base[e];
      const Ke = L.K ? L.K : CONDUCTION_K, o = L.K ? 64 * e : 0, s = L.K ? 1 : L.rho[e];
      for (let a = 0; a < 8; a++) {
        const ga = map[n0 + L.off[a]];
        if (ga < 0) continue;
        for (let b = 0; b < 8; b++) {
          const gb = map[n0 + L.off[b]];
          if (gb >= 0) A[ga * m + gb] += s * Ke[o + a * 8 + b];
        }
      }
    }
    if (L.diagAdd) for (let n = 0; n < L.nNodes; n++) if (map[n] >= 0) A[map[n] * m + map[n]] += L.diagAdd[n];
    let maxDiag = 0;
    for (let i = 0; i < m; i++) maxDiag = Math.max(maxDiag, A[i * m + i]);
    for (let j = 0; j < m; j++) {
      const rj = j * m;
      let s = A[rj + j];
      for (let k = 0; k < j; k++) s -= A[rj + k] * A[rj + k];
      if (!(s > 1e-12 * maxDiag)) s = maxDiag;
      const d = Math.sqrt(s);
      A[rj + j] = d;
      for (let i = j + 1; i < m; i++) {
        const ri = i * m;
        let t = A[ri + j];
        for (let k = 0; k < j; k++) t -= A[ri + k] * A[rj + k];
        A[ri + j] = t / d;
      }
    }
    this.coarse.A = A;
    this.coarse.y = new Float64Array(m);
  }

  coarseSolve(L) {
    const { map, m, A, y } = this.coarse;
    L.z.fill(0);
    if (!A) { for (let s = 0; s < 40; s++) this.jacobi(L, s === 0); return; }
    for (let n = 0; n < L.nNodes; n++) if (map[n] >= 0) y[map[n]] = L.r[n];
    for (let i = 0; i < m; i++) { let s = y[i]; for (let k = 0; k < i; k++) s -= A[i * m + k] * y[k]; y[i] = s / A[i * m + i]; }
    for (let i = m - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < m; k++) s -= A[k * m + i] * y[k]; y[i] = s / A[i * m + i]; }
    for (let n = 0; n < L.nNodes; n++) if (map[n] >= 0) L.z[n] = y[map[n]];
  }

  jacobi(L, first) {
    const { r, z, t, invDiag, omega } = L;
    if (first) { for (let i = 0; i < z.length; i++) z[i] = omega * invDiag[i] * r[i]; return; }
    this.apply(L, z, t);
    for (let i = 0; i < z.length; i++) z[i] += omega * invDiag[i] * (r[i] - t[i]);
  }

  restrict(F, C, rf, rc, zeroFixed = true) {
    rc.fill(0);
    const { mx, my, mz } = F.maps;
    for (let k = 0; k < F.NZ; k++) for (let j = 0; j < F.NY; j++) for (let i = 0; i < F.NX; i++) {
      const v = rf[i + F.NX * (j + F.NY * k)];
      if (v === 0) continue;
      for (let kk = 0; kk < 2; kk++) {
        const K = kk ? mz.c1[k] : mz.c0[k];
        if (K < 0) continue;
        const wk = mz.c1[k] < 0 ? 1 : 0.5;
        for (let jj = 0; jj < 2; jj++) {
          const J = jj ? my.c1[j] : my.c0[j];
          if (J < 0) continue;
          const wj = my.c1[j] < 0 ? 1 : 0.5;
          for (let ii = 0; ii < 2; ii++) {
            const I = ii ? mx.c1[i] : mx.c0[i];
            if (I < 0) continue;
            const wi = mx.c1[i] < 0 ? 1 : 0.5;
            rc[I + C.NX * (J + C.NY * K)] += wi * wj * wk * v;
          }
        }
      }
    }
    if (zeroFixed) for (let n = 0; n < rc.length; n++) if (C.fixed[n]) rc[n] = 0;
  }

  prolongAdd(F, C, zc, zf) {
    const { mx, my, mz } = F.maps;
    for (let k = 0; k < F.NZ; k++) for (let j = 0; j < F.NY; j++) for (let i = 0; i < F.NX; i++) {
      const fn = i + F.NX * (j + F.NY * k);
      if (F.fixed[fn]) continue;
      let s = 0;
      for (let kk = 0; kk < 2; kk++) {
        const K = kk ? mz.c1[k] : mz.c0[k];
        if (K < 0) continue;
        const wk = mz.c1[k] < 0 ? 1 : 0.5;
        for (let jj = 0; jj < 2; jj++) {
          const J = jj ? my.c1[j] : my.c0[j];
          if (J < 0) continue;
          const wj = my.c1[j] < 0 ? 1 : 0.5;
          for (let ii = 0; ii < 2; ii++) {
            const I = ii ? mx.c1[i] : mx.c0[i];
            if (I < 0) continue;
            const wi = mx.c1[i] < 0 ? 1 : 0.5;
            s += wi * wj * wk * zc[I + C.NX * (J + C.NY * K)];
          }
        }
      }
      zf[fn] += s;
    }
  }

  vcycle(l) {
    const L = this.levels[l];
    if (l === this.levels.length - 1) { this.coarseSolve(L); return; }
    const C = this.levels[l + 1];
    this.jacobi(L, true);
    for (let s = 1; s < SMOOTH_SWEEPS; s++) this.jacobi(L, false);
    this.apply(L, L.z, L.t);
    for (let i = 0; i < L.t.length; i++) L.t[i] = L.r[i] - L.t[i];
    this.restrict(L, C, L.t, C.r);
    this.vcycle(l + 1);
    this.prolongAdd(L, C, C.z, L.z);
    for (let s = 0; s < SMOOTH_SWEEPS; s++) this.jacobi(L, false);
  }

  /**
   * Solve A T = f with T fixed at x0 on prescribed nodes. x0 also warm-starts the free nodes.
   * @returns {{x: Float64Array, iterations: number, residual: number, converged: boolean}}
   */
  solve(f, { x0, tol = 1e-8, maxIter = 400 } = {}) {
    const L = this.levels[0], n = L.nNodes, fixed = L.fixed;
    const x = x0 ? Float64Array.from(x0) : new Float64Array(n);
    for (let i = 0; i < n; i++) if (fixed[i] && !L.active[i]) x[i] = 0;
    const r = new Float64Array(n), p = new Float64Array(n), q = new Float64Array(n);
    this.apply(L, x, q, true);
    let bn = 0;
    for (let i = 0; i < n; i++) {
      r[i] = fixed[i] ? 0 : f[i] - q[i];
      if (!fixed[i]) bn += f[i] * f[i] + q[i] * q[i];
    }
    bn = Math.sqrt(bn) || 1;
    const dot = (a, b) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * b[i]; return s; };
    let res = Math.sqrt(dot(r, r)) / bn, it = 0;
    if (res <= tol) return { x, iterations: 0, residual: res, converged: true };
    const pre = () => { L.r.set(r); this.vcycle(0); return L.z; };
    let z = pre();
    p.set(z);
    let rz = dot(r, z);
    for (; it < maxIter && res > tol; it++) {
      this.apply(L, p, q);
      const pq = dot(p, q);
      if (!(pq > 0)) break;
      const a = rz / pq;
      for (let i = 0; i < n; i++) { x[i] += a * p[i]; r[i] -= a * q[i]; }
      res = Math.sqrt(dot(r, r)) / bn;
      if (res <= tol) { it++; break; }
      z = pre();
      const rzn = dot(r, z);
      const beta = rzn / rz;
      rz = rzn;
      for (let i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
    }
    return { x, iterations: it, residual: res, converged: res <= tol * 10 };
  }
}

/** Lumped heat-capacity share per node (normalized: voxel volume 1, rho*cp 1). */
export function nodeCapacity(dims, density) {
  const [nx, ny, nz] = dims, NX = nx + 1, NY = ny + 1;
  const c = new Float64Array(NX * NY * (nz + 1));
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    const d = density[i + nx * (j + ny * k)];
    if (!(d > 0)) continue;
    for (let a = 0; a < 8; a++) c[i + (a & 1) + NX * (j + ((a >> 1) & 1) + NY * (k + ((a >> 2) & 1)))] += d / 8;
  }
  return c;
}

/**
 * Heat flux magnitude [W/m^2] per node (voxel-centroid gradients averaged to nodes).
 * T in degrees, k [W/m K], h voxel size [m].
 */
export function heatFlux(solver, T, k, h) {
  const L = solver.levels[0];
  const out = new Float32Array(L.nNodes), w = new Float32Array(L.nNodes);
  for (let e = 0; e < L.elems.length; e++) {
    const n0 = L.base[e];
    let gx = 0, gy = 0, gz = 0;
    for (let a = 0; a < 8; a++) {
      const t = T[n0 + L.off[a]] / 4; // centroid gradient of the trilinear field: +-1/4 per node
      const [x, y, z] = HEX_NODES[a];
      gx += x ? t : -t; gy += y ? t : -t; gz += z ? t : -t;
    }
    const q = (k * Math.hypot(gx, gy, gz)) / h;
    const rho = L.rho[e];
    for (let a = 0; a < 8; a++) { out[n0 + L.off[a]] += rho * q; w[n0 + L.off[a]] += rho; }
  }
  for (let n = 0; n < out.length; n++) if (w[n] > 0) out[n] /= w[n];
  return out;
}

/**
 * Steady or transient heat conduction.
 * @param {object} o
 * @param {number[]} o.dims
 * @param {Float32Array} o.density
 * @param {Uint8Array} o.fixedNode         1 where the temperature is prescribed
 * @param {Float64Array} o.fixedValue      prescribed temperatures [deg]
 * @param {Float64Array} o.source          heat input per node [W]
 * @param {Float64Array} o.convH           h_c * area per node [W/K]
 * @param {Float64Array} o.convT           ambient temperature per node (weighted by convH) [deg]
 * @param {number} o.k                     conductivity [W/m K]
 * @param {number} o.h                     voxel size [m]
 * @param {number} [o.rhoCp]               volumetric heat capacity [J/m^3 K] (transient)
 * @param {number} [o.duration]            [s]; 0 or missing = steady state
 * @param {number} [o.steps]
 * @param {number} [o.initial]             initial temperature [deg] (transient)
 * @param {(i: number, t: number, T: Float64Array) => boolean|void} [o.onStep] return true to cancel
 */
export function solveHeat({ dims, density, fixedNode, fixedValue, source, convH, convT, k, h, rhoCp = 0, duration = 0, steps = 30, initial = 20, onStep = null, onBuilt = null }) {
  if (!(k > 0) || !(h > 0)) throw new Error('Thermal conductivity and voxel size must be positive.');
  const kh = k * h;
  const nNodes = (dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1);
  const conv = new Float64Array(nNodes);
  for (let n = 0; n < nNodes; n++) conv[n] = convH[n] / kh;
  const rhsBase = new Float64Array(nNodes);
  for (let n = 0; n < nNodes; n++) rhsBase[n] = (source[n] + convH[n] * convT[n]) / kh;
  const transient = duration > 0 && rhoCp > 0;
  let hasSink = fixedNode.some((v) => v) || conv.some((v) => v > 0);
  if (!transient && !hasSink) throw new Error('Steady state needs somewhere for the heat to go: add a fixed temperature or convection.');
  const x0 = new Float64Array(nNodes).fill(transient ? initial : 20);
  for (let n = 0; n < nNodes; n++) if (fixedNode[n]) x0[n] = fixedValue[n];
  if (!transient) {
    const solver = new ScalarVoxelSolver({ dims, density, fixed: fixedNode, diagAdd: conv });
    onBuilt?.(solver);
    const sol = solver.solve(rhsBase, { x0 });
    onStep?.(0, 0, sol.x);
    return { solver, T: sol.x, frames: [{ t: 0, T: sol.x }], converged: sol.converged, iterations: sol.iterations };
  }
  const dt = duration / steps;
  // heat capacity per node / (dt k h): rho cp h^3 fill/8 / (dt k h) = rho cp h^2 / (k dt) * fill/8
  const cap = nodeCapacity(dims, density);
  const cs = (rhoCp * h * h) / (k * dt);
  const diag = new Float64Array(nNodes);
  for (let n = 0; n < nNodes; n++) diag[n] = conv[n] + cs * cap[n];
  const solver = new ScalarVoxelSolver({ dims, density, fixed: fixedNode, diagAdd: diag });
  onBuilt?.(solver);
  let T = x0;
  onStep?.(0, 0, T);
  const frames = [{ t: 0, T: Float64Array.from(T) }];
  const rhs = new Float64Array(nNodes);
  let converged = true, iterations = 0;
  for (let s = 1; s <= steps; s++) {
    for (let n = 0; n < nNodes; n++) rhs[n] = rhsBase[n] + cs * cap[n] * T[n];
    const sol = solver.solve(rhs, { x0: T });
    converged &&= sol.converged;
    iterations += sol.iterations;
    T = sol.x;
    frames.push({ t: s * dt, T: Float64Array.from(T) });
    if (onStep && onStep(s, s * dt, T) === true) throw Object.assign(new Error('Cancelled'), { cancelled: true });
  }
  return { solver, T, frames, converged, iterations };
}
