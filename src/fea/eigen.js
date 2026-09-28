// Eigenvalue problems on the voxel model: natural frequencies (K x = w^2 M x) and linear
// buckling (K x = -lambda K_G x).
//
// Both are solved with LOBPCG (Knyazev 2001): a block conjugate-gradient method that only needs
// matrix-vector products and a preconditioner. The multigrid V-cycle (or the GPU multigrid-CG)
// that already solves K u = f makes a very good preconditioner here, so a handful of modes
// converge in a few dozen iterations even on large, slender models.
import { elasticityMatrix, HEX_NODES } from './hex8.js';
import { principalStresses, corner } from './solver.js';

/** Lumped mass per DOF in normalized units (voxel side 1, density 1): fill fraction / 8 per node. */
export function lumpedMass(fea) {
  const L = fea.levels[0];
  const m = new Float64Array(L.nDof);
  for (let e = 0; e < L.elems.length; e++) {
    const w = L.rho[e] / 8, n0 = L.base[e];
    for (let a = 0; a < 8; a++) {
      const n = 3 * (n0 + L.off[a]);
      m[n] += w;
      m[n + 1] += w;
      m[n + 2] += w;
    }
  }
  return m;
}

// Shape-function gradients of the unit cube at natural point (xi, eta, zeta): 8 x 3.
function shapeGradients(xi, eta, zeta) {
  const g = new Float64Array(24);
  for (let a = 0; a < 8; a++) {
    const sx = 2 * HEX_NODES[a][0] - 1, sy = 2 * HEX_NODES[a][1] - 1, sz = 2 * HEX_NODES[a][2] - 1;
    g[3 * a] = (2 * sx * (1 + sy * eta) * (1 + sz * zeta)) / 8;
    g[3 * a + 1] = (2 * sy * (1 + sx * xi) * (1 + sz * zeta)) / 8;
    g[3 * a + 2] = (2 * sz * (1 + sx * xi) * (1 + sy * eta)) / 8;
  }
  return g;
}

// T[c] (8 x 8) with sum_c s_c T[c] = integral of grad(N_a) . S . grad(N_b) over the unit cube,
// for a constant stress S in Voigt order (xx, yy, zz, xy, yz, zx).
export const GEO_T = (() => {
  const H = Array.from({ length: 9 }, () => new Float64Array(64));
  const g = 1 / Math.sqrt(3);
  for (const xi of [-g, g]) for (const eta of [-g, g]) for (const zeta of [-g, g]) {
    const G = shapeGradients(xi, eta, zeta);
    for (let k = 0; k < 3; k++) for (let l = 0; l < 3; l++) {
      const M = H[3 * k + l];
      for (let a = 0; a < 8; a++) for (let b = 0; b < 8; b++) M[a * 8 + b] += (G[3 * a + k] * G[3 * b + l]) / 8;
    }
  }
  const sum = (p, q) => H[p].map((v, i) => v + H[q][i]);
  return [H[0], H[4], H[8], sum(1, 3), sum(5, 7), sum(6, 2)];
})();

// Centroid strain-displacement matrix (6 x 24) of the unit cube; the incompatible modes have
// zero gradient at the centroid, so this is also the element's mean strain.
const B0 = (() => {
  const G = shapeGradients(0, 0, 0);
  const B = new Float64Array(144);
  for (let a = 0; a < 8; a++) {
    const c = 3 * a, dx = G[c], dy = G[c + 1], dz = G[c + 2];
    B[c] = dx; B[24 + c + 1] = dy; B[48 + c + 2] = dz;
    B[72 + c] = dy; B[72 + c + 1] = dx;
    B[96 + c + 1] = dz; B[96 + c + 2] = dy;
    B[120 + c] = dz; B[120 + c + 2] = dx;
  }
  return B;
})();
export { B0 as CENTROID_B };

/**
 * Mean stress of every voxel, in units of E (so dimensionless), from physical displacements
 * u [m] and voxel size h [m]. Returns 6 values per solved voxel (fea.levels[0].elems order).
 */
export function elementStresses(fea, u, h) {
  const L = fea.levels[0];
  const D = elasticityMatrix(fea.nu);
  const out = new Float64Array(6 * L.elems.length);
  const eps = new Float64Array(6);
  for (let e = 0; e < L.elems.length; e++) {
    const n0 = L.base[e];
    eps.fill(0);
    for (let a = 0; a < 8; a++) {
      const n = 3 * (n0 + L.off[a]);
      for (let d = 0; d < 3; d++) {
        const v = u[n + d] / h;
        if (v === 0) continue;
        const c = 3 * a + d;
        for (let i = 0; i < 6; i++) eps[i] += B0[i * 24 + c] * v;
      }
    }
    for (let i = 0; i < 6; i++) {
      let s = 0;
      for (let j = 0; j < 6; j++) s += D[i * 6 + j] * eps[j];
      out[6 * e + i] = s;
    }
  }
  return out;
}

/**
 * Geometric (stress) stiffness K_G of a pre-stressed model, in the solver's normalized units, so
 * that K + lambda * K_G is singular at the buckling load factor lambda. `sigma` is the
 * dimensionless voxel stress from elementStresses (smeared by each voxel's fill fraction here).
 */
export class GeometricStiffness {
  constructor(fea, sigma) {
    const L = fea.levels[0];
    this.L = L;
    this.fea = fea;
    this.threads = null;
    // in shared memory when fea has helper threads (share())
    const len = 64 * L.elems.length;
    this.G = fea.threads ? new Float64Array(new SharedArrayBuffer(8 * len)) : new Float64Array(len);
    for (let e = 0; e < L.elems.length; e++) {
      const o = 64 * e, rho = L.rho[e];
      for (let c = 0; c < 6; c++) {
        const s = rho * sigma[6 * e + c];
        if (s === 0) continue;
        const T = GEO_T[c];
        for (let i = 0; i < 64; i++) this.G[o + i] += s * T[i];
      }
    }
  }

  /** Hands the stress stiffness to fea's helper threads, which then share its products. */
  async share() {
    const t = this.fea.threads;
    if (!t) return;
    await t.share({ kg: this.G });
    this.threads = t;
  }

  /** y = K_G x; held DOFs of y are zeroed. */
  apply(x, y) {
    if (this.threads) this.threads.geometric(x, y);
    else geometricNodes(this.L, this.G, x, y, 0, this.L.nNodes);
  }
}

/**
 * y = K_G x on the nodes n0 <= n < n1 (0 on held DOFs), with G the 8 x 8 stress stiffness of each
 * voxel: each node adds its voxels' rows in element order, as a scatter over the voxels would.
 */
export function geometricNodes(L, G, x, y, n0, n1) {
  const { NX, NY, nx, ny, nz, emap, fixed, off } = L;
  const NXY = NX * NY;
  let i = n0 % NX, j = ((n0 / NX) | 0) % NY, k = (n0 / NXY) | 0;
  for (let n = n0; n < n1; n++) {
    let ax = 0, ay = 0, az = 0;
    for (let v = 0; v < 8; v++) {
      const di = v & 1, dj = (v >> 1) & 1, dk = v >> 2;
      const ei = i + di - 1, ej = j + dj - 1, ek = k + dk - 1;
      if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= nz) continue;
      const e = emap[ei + nx * (ej + ny * ek)];
      if (e < 0) continue;
      const nb = n - (1 - di) - NX * (1 - dj) - NXY * (1 - dk), o = 64 * e + 8 * corner(1 - di, 1 - dj, 1 - dk);
      let sx = 0, sy = 0, sz = 0;
      for (let b = 0; b < 8; b++) {
        const g = G[o + b], p = 3 * (nb + off[b]);
        sx += g * x[p]; sy += g * x[p + 1]; sz += g * x[p + 2];
      }
      ax += sx; ay += sy; az += sz;
    }
    const q = 3 * n;
    y[q] = fixed[q] ? 0 : ax;
    y[q + 1] = fixed[q + 1] ? 0 : ay;
    y[q + 2] = fixed[q + 2] ? 0 : az;
    if (++i === NX) { i = 0; if (++j === NY) { j = 0; k++; } }
  }
}

// ---------- small dense linear algebra ----------

/** In-place Cholesky of an SPD k x k matrix (row-major, lower triangle). Returns false if not SPD. */
function cholesky(A, k) {
  for (let j = 0; j < k; j++) {
    let s = A[j * k + j];
    for (let q = 0; q < j; q++) s -= A[j * k + q] ** 2;
    if (!(s > 0)) return false;
    const d = Math.sqrt(s);
    A[j * k + j] = d;
    for (let i = j + 1; i < k; i++) {
      let t = A[i * k + j];
      for (let q = 0; q < j; q++) t -= A[i * k + q] * A[j * k + q];
      A[i * k + j] = t / d;
    }
    for (let i = 0; i < j; i++) A[i * k + j] = 0;
  }
  return true;
}

/** Eigen-decomposition of a symmetric k x k matrix by cyclic Jacobi; returns ascending values and column vectors. */
export function symmetricEigen(Ain, k) {
  const A = Float64Array.from(Ain);
  const V = new Float64Array(k * k);
  for (let i = 0; i < k; i++) V[i * k + i] = 1;
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0, total = 0;
    for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) {
      const a = A[i * k + j] ** 2;
      total += a;
      if (i !== j) off += a;
    }
    if (off <= 1e-30 * total || off === 0) break;
    for (let p = 0; p < k - 1; p++) {
      for (let q = p + 1; q < k; q++) {
        const apq = A[p * k + q];
        if (Math.abs(apq) < 1e-300) continue;
        const theta = (A[q * k + q] - A[p * k + p]) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), s = t * c;
        for (let r = 0; r < k; r++) {
          const arp = A[r * k + p], arq = A[r * k + q];
          A[r * k + p] = c * arp - s * arq;
          A[r * k + q] = s * arp + c * arq;
        }
        for (let r = 0; r < k; r++) {
          const apr = A[p * k + r], aqr = A[q * k + r];
          A[p * k + r] = c * apr - s * aqr;
          A[q * k + r] = s * apr + c * aqr;
        }
        for (let r = 0; r < k; r++) {
          const vrp = V[r * k + p], vrq = V[r * k + q];
          V[r * k + p] = c * vrp - s * vrq;
          V[r * k + q] = s * vrp + c * vrq;
        }
      }
    }
  }
  const order = Array.from({ length: k }, (_, i) => i).sort((a, b) => A[a * k + a] - A[b * k + b]);
  const values = order.map((i) => A[i * k + i]);
  const vectors = new Float64Array(k * k);
  order.forEach((src, dst) => { for (let r = 0; r < k; r++) vectors[r * k + dst] = V[r * k + src]; });
  return { values, vectors };
}

/** Generalized symmetric eigenproblem gA c = theta gB c (gB SPD). Returns null if gB is not SPD. */
function generalizedEigen(gA, gB, k) {
  const Lc = Float64Array.from(gB);
  if (!cholesky(Lc, k)) return null;
  // C = L^-1 gA L^-T
  const Y = new Float64Array(k * k); // Y = L^-1 gA
  for (let col = 0; col < k; col++) {
    for (let i = 0; i < k; i++) {
      let s = gA[i * k + col];
      for (let q = 0; q < i; q++) s -= Lc[i * k + q] * Y[q * k + col];
      Y[i * k + col] = s / Lc[i * k + i];
    }
  }
  const C = new Float64Array(k * k); // C = Y L^-T = (L^-1 Y^T)^T
  for (let row = 0; row < k; row++) {
    for (let i = 0; i < k; i++) {
      let s = Y[row * k + i];
      for (let q = 0; q < i; q++) s -= Lc[i * k + q] * C[row * k + q];
      C[row * k + i] = s / Lc[i * k + i];
    }
  }
  for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
    const m = 0.5 * (C[i * k + j] + C[j * k + i]);
    C[i * k + j] = C[j * k + i] = m;
  }
  const { values, vectors } = symmetricEigen(C, k);
  // back-substitute: c = L^-T y
  const out = new Float64Array(k * k);
  for (let col = 0; col < k; col++) {
    for (let i = k - 1; i >= 0; i--) {
      let s = vectors[i * k + col];
      for (let q = i + 1; q < k; q++) s -= Lc[q * k + i] * out[q * k + col];
      out[i * k + col] = s / Lc[i * k + i];
    }
  }
  return { values, vectors: out };
}

// ---------- LOBPCG ----------

// The block operations walk the vectors in cache-sized chunks, so a block of a few dozen long
// vectors is read from memory once per operation rather than once per pair of vectors. Each works
// on a range of chunks, so that helper threads (threads.js) can share it, and sums over the vectors
// are taken chunk by chunk and added up in chunk order: the same on any number of threads.
export const CHUNK = 2048;

/** Dense block kernels on the chunks c0 <= c < c1 of vectors of length n: kernel(V, coef, ints, part, n, c0, c1). */
export const DENSE_OP = { gram: 0, combine: 1, subtract: 2, triangular: 3, residual: 4, diagonal: 5 };
export const DENSE = [];
// part[c nU nV + i nV + j] = U[i] . V[j] on chunk c (only j <= i when ints[0] = 1)
DENSE[DENSE_OP.gram] = ([U, V], coef, ints, part, n, c0, c1) => {
  const lower = ints[0] === 1, nU = U.length, nV = V.length;
  for (let c = c0; c < c1; c++) {
    const t0 = c * CHUNK, t1 = Math.min(n, t0 + CHUNK), o = c * nU * nV;
    for (let i = 0; i < nU; i++) {
      const u = U[i];
      for (let j = 0; j < (lower ? i + 1 : nV); j++) {
        const v = V[j];
        let s0 = 0, s1 = 0, t = t0;
        for (; t + 1 < t1; t += 2) { s0 += u[t] * v[t]; s1 += u[t + 1] * v[t + 1]; }
        if (t < t1) s0 += u[t] * v[t];
        part[o + i * nV + j] = s0 + s1;
      }
    }
  }
};
// out[j] = plus[j] (or 0) + sum over the parts V_q of sum_i V_q[i] C[(row_q + i) k + j], with
// V = [out, plus (may be empty), V_0, V_1, ...] and ints = [k, row_0, row_1, ...]
DENSE[DENSE_OP.combine] = (V, C, ints, part, n, c0, c1) => {
  const out = V[0], plus = V[1].length ? V[1] : null, k = ints[0];
  for (let c = c0; c < c1; c++) {
    const t0 = c * CHUNK, t1 = Math.min(n, t0 + CHUNK);
    for (let j = 0; j < out.length; j++) {
      const o = out[j];
      if (plus) { const p = plus[j]; for (let t = t0; t < t1; t++) o[t] = p[t]; }
      else for (let t = t0; t < t1; t++) o[t] = 0;
      for (let q = 2; q < V.length; q++) {
        const Vq = V[q], row = ints[q - 1];
        for (let i = 0; i < Vq.length; i++) {
          const cc = C[(row + i) * k + j];
          if (cc === 0) continue;
          const v = Vq[i];
          for (let t = t0; t < t1; t++) o[t] += cc * v[t];
        }
      }
    }
  }
};
// W[j] -= sum_i X[i] C[i ld + j], in place (ints = [ld])
DENSE[DENSE_OP.subtract] = ([W, X], C, ints, part, n, c0, c1) => {
  const ld = ints[0];
  for (let c = c0; c < c1; c++) {
    const t0 = c * CHUNK, t1 = Math.min(n, t0 + CHUNK);
    for (let j = 0; j < W.length; j++) {
      const w = W[j];
      for (let i = 0; i < X.length; i++) {
        const cc = C[i * ld + j];
        if (cc === 0) continue;
        const x = X[i];
        for (let t = t0; t < t1; t++) w[t] -= cc * x[t];
      }
    }
  }
};
// M <- M L^-T for each block M, with the lower Cholesky factor L in G (ints = [k]): column j only
// needs the finished columns before it
DENSE[DENSE_OP.triangular] = (Ms, G, ints, part, n, c0, c1) => {
  const k = ints[0];
  for (const M of Ms) {
    for (let c = c0; c < c1; c++) {
      const t0 = c * CHUNK, t1 = Math.min(n, t0 + CHUNK);
      for (let j = 0; j < k; j++) {
        const v = M[j];
        for (let q = 0; q < j; q++) {
          const g = G[j * k + q], w = M[q];
          for (let t = t0; t < t1; t++) v[t] -= g * w[t];
        }
        const inv = 1 / G[j * k + j];
        for (let t = t0; t < t1; t++) v[t] *= inv;
      }
    }
  }
};
// R[j] = AX[j] - theta_j BX[j], with each chunk's r.r, ax.ax and bx.bx in part[3 (c m + j) + 0..2]
DENSE[DENSE_OP.residual] = ([AX, BX, R], theta, ints, part, n, c0, c1) => {
  const m = R.length;
  for (let c = c0; c < c1; c++) {
    const t0 = c * CHUNK, t1 = Math.min(n, t0 + CHUNK);
    for (let j = 0; j < m; j++) {
      const ax = AX[j], bx = BX[j], r = R[j], th = theta[j];
      let rr = 0, na = 0, nb = 0;
      for (let t = t0; t < t1; t++) {
        const v = ax[t] - th * bx[t];
        r[t] = v;
        rr += v * v; na += ax[t] * ax[t]; nb += bx[t] * bx[t];
      }
      const o = 3 * (c * m + j);
      part[o] = rr; part[o + 1] = na; part[o + 2] = nb;
    }
  }
};
// Y[j] = D[0] * X[j] entry by entry (a diagonal B)
DENSE[DENSE_OP.diagonal] = ([Y, X, D], coef, ints, part, n, c0, c1) => {
  const d = D[0];
  for (let c = c0; c < c1; c++) {
    const t0 = c * CHUNK, t1 = Math.min(n, t0 + CHUNK);
    for (let j = 0; j < Y.length; j++) {
      const y = Y[j], x = X[j];
      for (let t = t0; t < t1; t++) y[t] = d[t] * x[t];
    }
  }
};

const NONE = [];

/**
 * LOBPCG's vectors and block operations. With helper threads the vectors come from an arena of
 * shared memory (released explicitly: release()), and operations whose vectors all live there run
 * on all threads; any others run on this thread with the same results.
 */
class Dense {
  static async create(n, m, threads) {
    const d = new Dense(n, m);
    // the most vectors alive at once: X, W, P and the new X and P, with their A and B images
    const slots = Math.min(16 * m + 2, Math.floor(1.5e9 / (8 * Math.max(1, n))));
    if (threads && d.chunks > 1 && slots >= 4 * m) {
      try {
        const shared = (Type, len) => new Type(new SharedArrayBuffer(len * Type.BYTES_PER_ELEMENT));
        const arena = shared(Float64Array, slots * n);
        const extra = {
          arena, arenaN: n,
          dlist: shared(Int32Array, 8 + 6 * m), dcoef: shared(Float64Array, 9 * m * m + 3 * m),
          dint: shared(Int32Array, 8), dpart: shared(Float64Array, d.part.length),
        };
        await threads.share(extra);
        d.threads = threads;
        d.buf = arena.buffer;
        d.views = Array.from({ length: slots }, (_, i) => arena.subarray(i * n, (i + 1) * n));
        d.live = new Uint8Array(slots);
        d.free = Array.from({ length: slots }, (_, i) => slots - 1 - i);
      } catch {
        d.threads = null; // not enough memory to share: this thread alone
      }
    }
    return d;
  }

  constructor(n, m) {
    this.n = n;
    this.chunks = Math.ceil(n / CHUNK);
    this.part = new Float64Array(this.chunks * Math.max(m * m, 3 * m));
    this.threads = null;
    this.buf = null;
    this.free = [];
  }

  /** A vector of length n (from the arena while it lasts; contents undefined). */
  vec() {
    if (!this.free.length) return new Float64Array(this.n);
    const i = this.free.pop();
    this.live[i] = 1;
    return this.views[i];
  }

  slot(v) {
    return v.buffer === this.buf ? v.byteOffset / (8 * this.n) : -1;
  }

  /** Returns the vectors of the given blocks to the arena, except those in `keep`. */
  release(blocks, keep = null) {
    if (!this.buf) return;
    for (const V of blocks) {
      if (!V) continue;
      for (const v of V) {
        const i = this.slot(v);
        if (i < 0 || !this.live[i] || keep?.includes(v)) continue;
        this.live[i] = 0;
        this.free.push(i);
      }
    }
  }

  run(op, lists, coef, ints) {
    if (this.threads && lists.every((V) => V.every((v) => this.slot(v) >= 0))) {
      return this.threads.dense(op, lists.map((V) => V.map((v) => this.slot(v))), coef, ints, this.chunks);
    }
    DENSE[op](lists, coef, ints, this.part, this.n, 0, this.chunks);
    return this.part;
  }

  /** G[(r0 + i) * ld + c0 + j] = U[i] . V[j] for every pair (only j <= i when `lower`). */
  gram(U, V, G, ld, r0, c0, lower = false) {
    const nU = U.length, nV = V.length;
    const part = this.run(DENSE_OP.gram, [U, V], NONE, [lower ? 1 : 0]);
    for (let i = 0; i < nU; i++) {
      for (let j = 0; j < (lower ? i + 1 : nV); j++) {
        let s = 0;
        for (let c = 0, o = i * nV + j; c < this.chunks; c++, o += nU * nV) s += part[o];
        G[(r0 + i) * ld + c0 + j] = s;
      }
    }
  }

  /**
   * New vectors out[j] = sum over the parts [V, row] of sum_i V[i] * C[(row + i) * k + j], j < cols.
   * With `plus`, out[j] also gets plus[j] added.
   */
  combine(parts, C, k, cols, plus = null) {
    const out = Array.from({ length: cols }, () => this.vec());
    this.run(DENSE_OP.combine, [out, plus || NONE, ...parts.map(([V]) => V)], C, [k, ...parts.map(([, row]) => row)]);
    return out;
  }

  /** W[j] -= sum_i X[i] * C[i * ld + j], in place. */
  subtract(W, X, C, ld) {
    this.run(DENSE_OP.subtract, [W, X], C, [ld]);
  }

  /** New vectors d * X[j] (d a vector: a diagonal matrix). */
  diagonal(X, d) {
    const out = X.map(() => this.vec());
    this.run(DENSE_OP.diagonal, [out, X, [d]], NONE, NONE);
    return out;
  }

  /** R[j] = AX[j] - theta[j] BX[j] (R given); returns r.r, ax.ax and bx.bx of each (3 per vector). */
  residual(AX, BX, R, theta) {
    const m = R.length, part = this.run(DENSE_OP.residual, [AX, BX, R], theta, NONE), sums = new Float64Array(3 * m);
    for (let c = 0; c < this.chunks; c++) for (let q = 0; q < 3 * m; q++) sums[q] += part[3 * c * m + q];
    return sums;
  }

  /** Drops the arena (the helpers hold it until then). */
  async close() {
    if (this.threads) await this.threads.share({ arena: null, dlist: null, dcoef: null, dint: null, dpart: null });
    this.threads = null;
    this.buf = null;
    this.views = null;
  }
}

/**
 * B-orthonormalize a block in place (with its A and B images) by Cholesky of its Gram matrix.
 * Returns false when the block is numerically rank-deficient.
 */
function bOrthonormalize(dense, V, AV, BV) {
  const k = V.length;
  const G = new Float64Array(k * k);
  dense.gram(V, BV, G, k, 0, 0, true);
  for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) G[j * k + i] = G[i * k + j];
  let scale = 0;
  for (let i = 0; i < k; i++) scale = Math.max(scale, G[i * k + i]);
  if (!(scale > 0)) return false;
  if (!cholesky(G, k)) return false;
  for (let i = 0; i < k; i++) if (!(G[i * k + i] > 1e-7 * Math.sqrt(scale))) return false;
  dense.run(DENSE_OP.triangular, [V, AV, BV].filter(Boolean), G, [k]);
  return true;
}

/**
 * Smallest eigenpairs of A x = theta B x (A symmetric, B symmetric positive definite).
 * @param {object} o
 * @param {number} o.n                vector length
 * @param {(x: Float64Array, y: Float64Array) => void} o.applyA
 * @param {(x: Float64Array, y: Float64Array) => void} o.applyB
 * @param {(V: Float64Array[], vec: () => Float64Array) => Promise<Float64Array[]>} [o.blockA]  A times
 *                                    a block, e.g. on several threads (instead of applyA per vector),
 *                                    into new vectors from vec()
 * @param {(V: Float64Array[], vec: () => Float64Array) => Promise<Float64Array[]>} [o.blockB]  likewise for B
 * @param {Float64Array} [o.diagB]    B is this diagonal (instead of applyB)
 * @param {boolean} [o.cheapB]        B is cheap to apply (a diagonal): its images are recomputed
 *                                    rather than carried along as combinations
 * @param {(r: Float64Array[], vec: () => Float64Array) => Promise<Float64Array[]>|Float64Array[]} o.precond
 *                                    ~ A^-1 (or a shifted inverse), into new vectors (from vec())
 * @param {number} o.nev              eigenpairs wanted
 * @param {number} [o.block]          block size (extra "guard" vectors speed up convergence)
 * @param {Float64Array} [o.mask]     1 on DOFs that take part (0 = held)
 * @param {import('./threads.js').Threads} [o.threads]  helper threads for the block operations
 * @param {(it: number, res: number, conv: number) => boolean|void} [o.onProgress] return true to cancel
 */
export async function lobpcg({ n, applyA, applyB, blockA = null, blockB = null, diagB = null, cheapB = false, precond, nev, block = nev + Math.min(4, Math.max(2, nev)), tol = 1e-5, maxIter = 300, mask = null, threads = null, onProgress = null, seed = 12345, settleAbove = Infinity }) {
  const m = Math.max(nev, block);
  const dense = await Dense.create(n, m, threads);
  try {
    return await iterate();
  } finally {
    await dense.close();
  }

  async function iterate() {
  const vec = () => dense.vec();
  const D = diagB ? vec() : null;
  if (D) D.set(diagB);
  const imageA = blockA ? (V) => blockA(V, vec) : (V) => V.map((v) => { const y = vec(); applyA(v, y); return y; });
  const imageB = D ? (V) => dense.diagonal(V, D) : blockB ? (V) => blockB(V, vec) : (V) => V.map((v) => { const y = vec(); applyB(v, y); return y; });
  let s = seed;
  const rand = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  // start from preconditioned random vectors (smooth, low-energy shapes)
  let X = [];
  for (let j = 0; j < m; j++) {
    const v = vec();
    for (let t = 0; t < n; t++) v[t] = mask && !mask[t] ? 0 : rand();
    X.push(v);
  }
  const X0 = X;
  X = await precond(X0, vec);
  dense.release([X0], X);
  let BX = await imageB(X);
  if (!bOrthonormalize(dense, X, null, BX)) throw new Error('Could not start the eigenvalue solver (model has too few free nodes).');
  let AX = await imageA(X);
  let theta;
  // initial Rayleigh-Ritz on X
  {
    const gA = new Float64Array(m * m), gB = new Float64Array(m * m);
    dense.gram(X, AX, gA, m, 0, 0, true);
    dense.gram(X, BX, gB, m, 0, 0, true);
    for (let i = 0; i < m; i++) for (let j = 0; j < i; j++) { gA[j * m + i] = gA[i * m + j]; gB[j * m + i] = gB[i * m + j]; }
    const ev = generalizedEigen(gA, gB, m);
    if (!ev) throw new Error('Eigenvalue solver failed to start.');
    const X1 = dense.combine([[X, 0]], ev.vectors, m, m);
    const AX1 = dense.combine([[AX, 0]], ev.vectors, m, m);
    const BX1 = cheapB ? null : dense.combine([[BX, 0]], ev.vectors, m, m);
    dense.release([X, AX, BX]);
    X = X1;
    AX = AX1;
    BX = cheapB ? await imageB(X) : BX1;
    theta = ev.values.slice(0, m);
  }
  let P = null, AP = null, BP = null;
  let res = new Array(m).fill(1);
  const settled = new Array(m).fill(false);
  const history = Array.from({ length: 10 }, () => new Array(m).fill(0)); // Ritz values of the last 10 iterations
  let it = 0;
  for (; it < maxIter; it++) {
    // residuals
    const Rall = X.map(() => vec());
    const sums = dense.residual(AX, BX, Rall, theta);
    const R = [], active = [];
    for (let j = 0; j < m; j++) {
      const th = theta[j];
      res[j] = Math.sqrt(sums[3 * j]) / (Math.sqrt(sums[3 * j + 1]) + Math.abs(th) * Math.sqrt(sums[3 * j + 2]) || 1);
      // settled above the bound: its eigenvalue lies outside the searched range (Ritz values only
      // decrease towards the eigenvalues), so it needs no more accuracy
      settled[j] = it >= 10 && th > settleAbove && Math.abs(th - history[it % 10][j]) <= 0.01 * Math.abs(settleAbove);
      if (res[j] > tol && !settled[j]) { active.push(j); R.push(Rall[j]); }
    }
    dense.release([Rall], R);
    history[it % 10] = theta.slice();
    const conv = res.slice(0, nev).filter((r, j) => r <= tol || settled[j]).length;
    const worst = Math.max(0, ...res.slice(0, nev).filter((_, j) => !settled[j]));
    if (onProgress && onProgress(it, worst, conv) === true) throw Object.assign(new Error('Cancelled'), { cancelled: true });
    if (conv === nev) break;
    // preconditioned residuals, B-orthogonal to X
    let W = await precond(R, vec);
    dense.release([R], W);
    if (mask) for (const w of W) for (let t = 0; t < n; t++) if (!mask[t]) w[t] = 0;
    const nW = W.length;
    const XBW = new Float64Array(m * nW);
    dense.gram(BX, W, XBW, nW, 0, 0);
    dense.subtract(W, X, XBW, nW);
    let BW = await imageB(W);
    if (!bOrthonormalize(dense, W, null, BW)) {
      // dependent search directions: restart the momentum and try once more with a jitter
      dense.release([P, AP, BP]);
      P = AP = BP = null;
      for (const w of W) for (let t = 0; t < n; t++) if (!mask || mask[t]) w[t] += 1e-8 * rand();
      dense.release([BW]);
      BW = await imageB(W);
      if (!bOrthonormalize(dense, W, null, BW)) break;
    }
    const AW = await imageA(W);
    let usedP = false;
    if (P) {
      // P columns follow the active set
      const P1 = active.map((j) => P[j]).filter(Boolean), AP1 = active.map((j) => AP[j]).filter(Boolean), BP1 = active.map((j) => BP[j]).filter(Boolean);
      dense.release([P], P1);
      dense.release([AP], AP1);
      dense.release([BP], BP1);
      P = P1; AP = AP1; BP = BP1;
      usedP = P.length > 0 && bOrthonormalize(dense, P, AP, BP);
    }
    // Rayleigh-Ritz on [X W P]. X, W and P are each B-orthonormal and X^T A X = diag(theta), so
    // only the other blocks of the projected matrices need products.
    const nP = usedP ? P.length : 0;
    const k = m + nW + nP;
    const gA = new Float64Array(k * k), gB = new Float64Array(k * k);
    for (let i = 0; i < m; i++) gA[i * k + i] = theta[i];
    for (let i = 0; i < k; i++) gB[i * k + i] = 1;
    dense.gram(X, AW, gA, k, 0, m);
    dense.gram(W, AW, gA, k, m, m, true);
    dense.gram(X, BW, gB, k, 0, m);
    if (usedP) {
      dense.gram(X, AP, gA, k, 0, m + nW);
      dense.gram(W, AP, gA, k, m, m + nW);
      dense.gram(P, AP, gA, k, m + nW, m + nW, true);
      dense.gram(X, BP, gB, k, 0, m + nW);
      dense.gram(W, BP, gB, k, m, m + nW);
    }
    // mirror: blocks above the diagonal were filled, and the lower triangles of the diagonal blocks
    const blockOf = (i) => (i < m ? 0 : i < m + nW ? 1 : 2);
    for (let i = 0; i < k; i++) for (let j = i + 1; j < k; j++) {
      if (blockOf(i) === blockOf(j)) { gA[i * k + j] = gA[j * k + i]; gB[i * k + j] = gB[j * k + i]; }
      else { gA[j * k + i] = gA[i * k + j]; gB[j * k + i] = gB[i * k + j]; }
    }
    let ev = generalizedEigen(gA, gB, k);
    let kk = k;
    if (!ev && usedP) {
      // ill-conditioned with the momentum block: drop it
      kk = m + nW;
      const sub = (G) => { const o = new Float64Array(kk * kk); for (let i = 0; i < kk; i++) for (let j = 0; j < kk; j++) o[i * kk + j] = G[i * k + j]; return o; };
      ev = generalizedEigen(sub(gA), sub(gB), kk);
      usedP = false;
    }
    if (!ev) break;
    theta = ev.values.slice(0, m);
    const C = ev.vectors;
    // new momentum: the W and P parts of the Ritz vectors; new X adds its X part
    const step = (Xb, Wb, Pb) => {
      const p = dense.combine(usedP ? [[Wb, m], [Pb, m + nW]] : [[Wb, m]], C, kk, m);
      return [dense.combine([[Xb, 0]], C, kk, m, p), p];
    };
    const [X1, P1] = step(X, W, P);
    const [AX1, AP1] = step(AX, AW, AP);
    const [BX1, BP1] = cheapB ? [null, null] : step(BX, BW, BP);
    dense.release([X, W, P, AX, AW, AP, BX, BW, BP]);
    [X, P, AX, AP] = [X1, P1, AX1, AP1];
    if (cheapB) { BX = await imageB(X); BP = await imageB(P); }
    else [BX, BP] = [BX1, BP1];
  }
  return {
    values: theta.slice(0, nev),
    // copied out of the arena, which goes with the solver
    vectors: X.slice(0, nev).map((v) => Float64Array.from(v)),
    residuals: res.slice(0, nev),
    iterations: it,
    converged: res.slice(0, nev).every((r, j) => r <= tol * 10 || settled[j]),
  };
  }
}

// ---------- studies ----------

/** Maps between full DOF vectors and vectors of the free (active, unheld) DOFs only. */
export function freeDofs(fea) {
  const L = fea.levels[0];
  let n = 0;
  for (let i = 0; i < L.nDof; i++) if (!L.fixed[i]) n++;
  const idx = new Int32Array(n);
  n = 0;
  for (let i = 0; i < L.nDof; i++) if (!L.fixed[i]) idx[n++] = i;
  return {
    n,
    idx,
    scatter(x, full) { full.fill(0); for (let t = 0; t < idx.length; t++) full[idx[t]] = x[t]; return full; },
    gather(full, x) { for (let t = 0; t < idx.length; t++) x[t] = full[idx[t]]; return x; },
  };
}

/** Default preconditioner: one multigrid V-cycle per vector (full-length vectors). */
export function cpuPreconditioner(fea) {
  return (R) => R.map((r) => fea.precondition(r));
}

function compactOps(fea, precondFull, applyFull) {
  const L = fea.levels[0];
  const map = freeDofs(fea);
  const full = new Float64Array(L.nDof), out = new Float64Array(L.nDof);
  const pre = precondFull || cpuPreconditioner(fea);
  const onFull = async (op, V, vec) => (await op(V.map((v) => map.scatter(v, new Float64Array(L.nDof))))).map((y) => map.gather(y, vec ? vec() : new Float64Array(map.n)));
  return {
    map,
    applyK(x, y) { fea.apply(L, map.scatter(x, full), out); map.gather(out, y); },
    /** K times a block on the given (e.g. multi-threaded) full-length product, if any. */
    blockK: applyFull ? (V, vec) => onFull(applyFull, V, vec) : null,
    precond: (R, vec) => onFull(pre, R, vec),
    expand(x) { return map.scatter(x, new Float64Array(L.nDof)); },
  };
}

/**
 * Lowest natural frequencies. `fea` may carry a mass shift (diagAdd = shift * M) when the part is
 * free-floating, which keeps K + shift*M invertible for the preconditioner.
 * @returns {{freqs: number[], lambdas: number[], modes: Float64Array[], mass: Float64Array, converged: boolean, iterations: number}}
 *          modes are full-length, normalized so that mode^T M mode = 1 with the normalized mass.
 */
export async function naturalFrequencies(fea, { nev, E, density, h, shift = 0, precondFull = null, applyFull = null, onProgress = null, tol = 1e-5 }) {
  const mass = lumpedMass(fea);
  const ops = compactOps(fea, precondFull, applyFull);
  const mc = ops.map.gather(mass, new Float64Array(ops.map.n));
  const r = await lobpcg({
    n: ops.map.n, nev, tol, onProgress, threads: fea.threads,
    applyA: ops.applyK,
    blockA: ops.blockK,
    diagB: mc,
    cheapB: true,
    precond: ops.precond,
  });
  // normalized eigenvalue lambda = w^2 rho h^2 / E
  const lambdas = r.values.map((v) => Math.max(0, v - shift));
  const freqs = lambdas.map((l) => Math.sqrt((l * E) / (density * h * h)) / (2 * Math.PI));
  return { freqs, lambdas, modes: r.vectors.map((v) => ops.expand(v)), mass, converged: r.converged, iterations: r.iterations, residuals: r.residuals };
}

/**
 * Linear buckling load factors: K x = -lambda K_G x, smallest positive lambda first.
 * `sigma` is the dimensionless voxel stress (elementStresses) of the applied loads.
 */
export async function bucklingFactors(fea, sigma, { nev, precondFull = null, applyFull = null, onProgress = null, tol = 1e-5, maxFactor = Infinity }) {
  // K_G is positive semi-definite when no voxel is in compression: nothing can buckle
  let smax = 0, compression = 0;
  const pr = [0, 0, 0];
  for (let e = 0; 6 * e + 5 < sigma.length; e++) {
    const o = 6 * e;
    principalStresses(sigma[o], sigma[o + 1], sigma[o + 2], sigma[o + 3], sigma[o + 4], sigma[o + 5], pr);
    smax = Math.max(smax, Math.abs(pr[0]), Math.abs(pr[2]));
    compression = Math.max(compression, -pr[2]);
  }
  if (!(compression > 1e-12 * smax)) {
    const L = fea.levels[0];
    return { factors: new Array(nev).fill(Infinity), modes: Array.from({ length: nev }, () => new Float64Array(L.nDof)), converged: true, iterations: 0, residuals: [], maxFactor };
  }
  // factors above maxFactor are reported as Infinity: a part that yields long before never reaches
  // them, and the search there is slow (a cluster of eigenvalues near zero) and ill-conditioned
  const floor = Number.isFinite(maxFactor) && maxFactor > 0 ? 1 / maxFactor : 0;
  const KG = new GeometricStiffness(fea, sigma);
  await KG.share();
  const ops = compactOps(fea, precondFull, applyFull);
  const L = fea.levels[0];
  const full = new Float64Array(L.nDof), out = new Float64Array(L.nDof);
  // smallest (most negative) theta of K_G x = theta K x  <=>  buckling factor -1/theta
  const r = await lobpcg({
    n: ops.map.n, nev, tol, onProgress, settleAbove: floor > 0 ? -floor : Infinity, threads: fea.threads,
    applyA: (x, y) => { KG.apply(ops.map.scatter(x, full), out); ops.map.gather(out, y); },
    applyB: ops.applyK,
    blockB: ops.blockK,
    precond: ops.precond,
  });
  const factors = r.values.map((t) => (t < -floor ? -1 / t : Infinity));
  return { factors, modes: r.vectors.map((v) => ops.expand(v)), converged: r.converged, iterations: r.iterations, residuals: r.residuals, maxFactor };
}
