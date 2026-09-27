// Eigenvalue problems on the voxel model: natural frequencies (K x = w^2 M x) and linear
// buckling (K x = -lambda K_G x).
//
// Both are solved with LOBPCG (Knyazev 2001): a block conjugate-gradient method that only needs
// matrix-vector products and a preconditioner. The multigrid V-cycle (or the GPU multigrid-CG)
// that already solves K u = f makes a very good preconditioner here, so a handful of modes
// converge in a few dozen iterations even on large, slender models.
import { elasticityMatrix, HEX_NODES } from './hex8.js';
import { principalStresses } from './solver.js';

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
    this.G = new Float64Array(64 * L.elems.length);
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

  /** y = K_G x; held DOFs of y are zeroed. */
  apply(x, y) {
    const L = this.L, G = this.G, off = L.off;
    y.fill(0);
    const ue = new Float64Array(24);
    for (let e = 0; e < L.elems.length; e++) {
      const n0 = L.base[e], o = 64 * e;
      for (let a = 0; a < 8; a++) {
        const n = 3 * (n0 + off[a]);
        ue[3 * a] = x[n]; ue[3 * a + 1] = x[n + 1]; ue[3 * a + 2] = x[n + 2];
      }
      for (let a = 0; a < 8; a++) {
        let sx = 0, sy = 0, sz = 0;
        for (let b = 0; b < 8; b++) {
          const g = G[o + a * 8 + b];
          sx += g * ue[3 * b]; sy += g * ue[3 * b + 1]; sz += g * ue[3 * b + 2];
        }
        const n = 3 * (n0 + off[a]);
        y[n] += sx; y[n + 1] += sy; y[n + 2] += sz;
      }
    }
    for (let i = 0; i < y.length; i++) if (L.fixed[i]) y[i] = 0;
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

// The block operations below walk the vectors in cache-sized chunks, so a block of a few dozen
// long vectors is read from memory once per operation rather than once per pair of vectors.
const CHUNK = 2048;

/** G[(r0 + i) * ld + c0 + j] = U[i] . V[j] for every pair (only j <= i when `lower`). */
function gram(U, V, G, ld, r0, c0, lower = false) {
  const n = U.length ? U[0].length : 0;
  for (let i = 0; i < U.length; i++) for (let j = 0; j < (lower ? i + 1 : V.length); j++) G[(r0 + i) * ld + c0 + j] = 0;
  for (let t0 = 0; t0 < n; t0 += CHUNK) {
    const t1 = Math.min(n, t0 + CHUNK);
    for (let i = 0; i < U.length; i++) {
      const u = U[i];
      for (let j = 0; j < (lower ? i + 1 : V.length); j++) {
        const v = V[j];
        let s0 = 0, s1 = 0, t = t0;
        for (; t + 1 < t1; t += 2) { s0 += u[t] * v[t]; s1 += u[t + 1] * v[t + 1]; }
        if (t < t1) s0 += u[t] * v[t];
        G[(r0 + i) * ld + c0 + j] += s0 + s1;
      }
    }
  }
}

/**
 * New vectors out[j] = sum over the parts [V, row] of sum_i V[i] * C[(row + i) * k + j], j < cols.
 * With `plus`, out[j] also gets plus[j] added.
 */
function combine(parts, C, k, cols, n, plus = null) {
  const out = Array.from({ length: cols }, () => new Float64Array(n));
  for (let t0 = 0; t0 < n; t0 += CHUNK) {
    const t1 = Math.min(n, t0 + CHUNK);
    for (let j = 0; j < cols; j++) {
      const o = out[j];
      if (plus) { const p = plus[j]; for (let t = t0; t < t1; t++) o[t] = p[t]; }
      for (const [V, row] of parts) {
        for (let i = 0; i < V.length; i++) {
          const c = C[(row + i) * k + j];
          if (c === 0) continue;
          const v = V[i];
          for (let t = t0; t < t1; t++) o[t] += c * v[t];
        }
      }
    }
  }
  return out;
}

/** W[j] -= sum_i X[i] * C[i * ld + j], in place. */
function subtractCombination(W, X, C, ld) {
  const n = W.length ? W[0].length : 0;
  for (let t0 = 0; t0 < n; t0 += CHUNK) {
    const t1 = Math.min(n, t0 + CHUNK);
    for (let j = 0; j < W.length; j++) {
      const w = W[j];
      for (let i = 0; i < X.length; i++) {
        const c = C[i * ld + j];
        if (c === 0) continue;
        const x = X[i];
        for (let t = t0; t < t1; t++) w[t] -= c * x[t];
      }
    }
  }
}

/**
 * B-orthonormalize a block in place (with its A and B images) by Cholesky of its Gram matrix.
 * Returns false when the block is numerically rank-deficient.
 */
function bOrthonormalize(V, AV, BV) {
  const k = V.length;
  const G = new Float64Array(k * k);
  gram(V, BV, G, k, 0, 0, true);
  for (let i = 0; i < k; i++) for (let j = 0; j < i; j++) G[j * k + i] = G[i * k + j];
  let scale = 0;
  for (let i = 0; i < k; i++) scale = Math.max(scale, G[i * k + i]);
  if (!(scale > 0)) return false;
  if (!cholesky(G, k)) return false;
  for (let i = 0; i < k; i++) if (!(G[i * k + i] > 1e-7 * Math.sqrt(scale))) return false;
  // V <- V L^-T: column j only needs the finished columns before it, chunk by chunk
  const n = V[0].length;
  for (const M of [V, AV, BV]) {
    if (!M) continue;
    for (let t0 = 0; t0 < n; t0 += CHUNK) {
      const t1 = Math.min(n, t0 + CHUNK);
      for (let j = 0; j < k; j++) {
        const v = M[j];
        for (let q = 0; q < j; q++) {
          const c = G[j * k + q], w = M[q];
          for (let t = t0; t < t1; t++) v[t] -= c * w[t];
        }
        const inv = 1 / G[j * k + j];
        for (let t = t0; t < t1; t++) v[t] *= inv;
      }
    }
  }
  return true;
}

/**
 * Smallest eigenpairs of A x = theta B x (A symmetric, B symmetric positive definite).
 * @param {object} o
 * @param {number} o.n                vector length
 * @param {(x: Float64Array, y: Float64Array) => void} o.applyA
 * @param {(x: Float64Array, y: Float64Array) => void} o.applyB
 * @param {(V: Float64Array[]) => Promise<Float64Array[]>} [o.blockA]  A times a block, e.g. on
 *                                    several threads (instead of applyA per vector)
 * @param {(V: Float64Array[]) => Promise<Float64Array[]>} [o.blockB]  likewise for B
 * @param {boolean} [o.cheapB]        B is cheap to apply (a diagonal): its images are recomputed
 *                                    rather than carried along as combinations
 * @param {(r: Float64Array[]) => Promise<Float64Array[]>|Float64Array[]} o.precond  ~ A^-1 (or a shifted inverse)
 * @param {number} o.nev              eigenpairs wanted
 * @param {number} [o.block]          block size (extra "guard" vectors speed up convergence)
 * @param {Float64Array} [o.mask]     1 on DOFs that take part (0 = held)
 * @param {(it: number, res: number, conv: number) => boolean|void} [o.onProgress] return true to cancel
 */
export async function lobpcg({ n, applyA, applyB, blockA = null, blockB = null, cheapB = false, precond, nev, block = nev + Math.min(4, Math.max(2, nev)), tol = 1e-5, maxIter = 300, mask = null, onProgress = null, seed = 12345, settleAbove = Infinity }) {
  const m = Math.max(nev, block);
  const vec = () => new Float64Array(n);
  const imageA = blockA || ((V) => V.map((v) => { const y = vec(); applyA(v, y); return y; }));
  const imageB = blockB || ((V) => V.map((v) => { const y = vec(); applyB(v, y); return y; }));
  let s = seed;
  const rand = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff - 0.5; };
  // start from preconditioned random vectors (smooth, low-energy shapes)
  let X = [];
  for (let j = 0; j < m; j++) {
    const v = vec();
    for (let t = 0; t < n; t++) v[t] = mask && !mask[t] ? 0 : rand();
    X.push(v);
  }
  X = await precond(X);
  let BX = await imageB(X);
  if (!bOrthonormalize(X, null, BX)) throw new Error('Could not start the eigenvalue solver (model has too few free nodes).');
  let AX = await imageA(X);
  let theta;
  // initial Rayleigh-Ritz on X
  {
    const gA = new Float64Array(m * m), gB = new Float64Array(m * m);
    gram(X, AX, gA, m, 0, 0, true);
    gram(X, BX, gB, m, 0, 0, true);
    for (let i = 0; i < m; i++) for (let j = 0; j < i; j++) { gA[j * m + i] = gA[i * m + j]; gB[j * m + i] = gB[i * m + j]; }
    const ev = generalizedEigen(gA, gB, m);
    if (!ev) throw new Error('Eigenvalue solver failed to start.');
    X = combine([[X, 0]], ev.vectors, m, m, n);
    AX = combine([[AX, 0]], ev.vectors, m, m, n);
    BX = cheapB ? await imageB(X) : combine([[BX, 0]], ev.vectors, m, m, n);
    theta = ev.values.slice(0, m);
  }
  let P = null, AP = null, BP = null;
  let res = new Array(m).fill(1);
  const settled = new Array(m).fill(false);
  const history = Array.from({ length: 10 }, () => new Array(m).fill(0)); // Ritz values of the last 10 iterations
  let it = 0;
  for (; it < maxIter; it++) {
    // residuals
    const R = [], active = [];
    for (let j = 0; j < m; j++) {
      const r = vec(), ax = AX[j], bx = BX[j], th = theta[j];
      let rr = 0, na = 0, nb = 0;
      for (let t = 0; t < n; t++) {
        const v = ax[t] - th * bx[t];
        r[t] = v;
        rr += v * v; na += ax[t] * ax[t]; nb += bx[t] * bx[t];
      }
      res[j] = Math.sqrt(rr) / (Math.sqrt(na) + Math.abs(th) * Math.sqrt(nb) || 1);
      // settled above the bound: its eigenvalue lies outside the searched range (Ritz values only
      // decrease towards the eigenvalues), so it needs no more accuracy
      settled[j] = it >= 10 && th > settleAbove && Math.abs(th - history[it % 10][j]) <= 0.01 * Math.abs(settleAbove);
      if (res[j] > tol && !settled[j]) { active.push(j); R.push(r); }
    }
    history[it % 10] = theta.slice();
    const conv = res.slice(0, nev).filter((r, j) => r <= tol || settled[j]).length;
    const worst = Math.max(0, ...res.slice(0, nev).filter((_, j) => !settled[j]));
    if (onProgress && onProgress(it, worst, conv) === true) throw Object.assign(new Error('Cancelled'), { cancelled: true });
    if (conv === nev) break;
    // preconditioned residuals, B-orthogonal to X
    let W = await precond(R);
    if (mask) for (const w of W) for (let t = 0; t < n; t++) if (!mask[t]) w[t] = 0;
    const nW = W.length;
    const XBW = new Float64Array(m * nW);
    gram(BX, W, XBW, nW, 0, 0);
    subtractCombination(W, X, XBW, nW);
    let BW = await imageB(W);
    if (!bOrthonormalize(W, null, BW)) {
      // dependent search directions: restart the momentum and try once more with a jitter
      P = AP = BP = null;
      for (const w of W) for (let t = 0; t < n; t++) if (!mask || mask[t]) w[t] += 1e-8 * rand();
      BW = await imageB(W);
      if (!bOrthonormalize(W, null, BW)) break;
    }
    const AW = await imageA(W);
    let usedP = false;
    if (P) {
      const keep = active.map((j) => j); // P columns follow the active set
      P = keep.map((j) => P[j]).filter(Boolean);
      AP = keep.map((j) => AP[j]).filter(Boolean);
      BP = keep.map((j) => BP[j]).filter(Boolean);
      usedP = P.length > 0 && bOrthonormalize(P, AP, BP);
    }
    // Rayleigh-Ritz on [X W P]. X, W and P are each B-orthonormal and X^T A X = diag(theta), so
    // only the other blocks of the projected matrices need products.
    const nP = usedP ? P.length : 0;
    const k = m + nW + nP;
    const gA = new Float64Array(k * k), gB = new Float64Array(k * k);
    for (let i = 0; i < m; i++) gA[i * k + i] = theta[i];
    for (let i = 0; i < k; i++) gB[i * k + i] = 1;
    gram(X, AW, gA, k, 0, m);
    gram(W, AW, gA, k, m, m, true);
    gram(X, BW, gB, k, 0, m);
    if (usedP) {
      gram(X, AP, gA, k, 0, m + nW);
      gram(W, AP, gA, k, m, m + nW);
      gram(P, AP, gA, k, m + nW, m + nW, true);
      gram(X, BP, gB, k, 0, m + nW);
      gram(W, BP, gB, k, m, m + nW);
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
      const p = combine(usedP ? [[Wb, m], [Pb, m + nW]] : [[Wb, m]], C, kk, m, n);
      return [combine([[Xb, 0]], C, kk, m, n, p), p];
    };
    [X, P] = step(X, W, P);
    [AX, AP] = step(AX, AW, AP);
    if (cheapB) { BX = await imageB(X); BP = await imageB(P); }
    else [BX, BP] = step(BX, BW, BP);
  }
  return {
    values: theta.slice(0, nev),
    vectors: X.slice(0, nev),
    residuals: res.slice(0, nev),
    iterations: it,
    converged: res.slice(0, nev).every((r, j) => r <= tol * 10 || settled[j]),
  };
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
  const onFull = async (op, V) => (await op(V.map((v) => map.scatter(v, new Float64Array(L.nDof))))).map((y) => map.gather(y, new Float64Array(map.n)));
  return {
    map,
    applyK(x, y) { fea.apply(L, map.scatter(x, full), out); map.gather(out, y); },
    /** K times a block on the given (e.g. multi-threaded) full-length product, if any. */
    blockK: applyFull ? (V) => onFull(applyFull, V) : null,
    precond: (R) => onFull(pre, R),
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
    n: ops.map.n, nev, tol, onProgress,
    applyA: ops.applyK,
    blockA: ops.blockK,
    applyB: (x, y) => { for (let t = 0; t < x.length; t++) y[t] = mc[t] * x[t]; },
    cheapB: true,
    precond: (R) => ops.precond(R),
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
  const ops = compactOps(fea, precondFull, applyFull);
  const L = fea.levels[0];
  const full = new Float64Array(L.nDof), out = new Float64Array(L.nDof);
  // smallest (most negative) theta of K_G x = theta K x  <=>  buckling factor -1/theta
  const r = await lobpcg({
    n: ops.map.n, nev, tol, onProgress, settleAbove: floor > 0 ? -floor : Infinity,
    applyA: (x, y) => { KG.apply(ops.map.scatter(x, full), out); ops.map.gather(out, y); },
    applyB: ops.applyK,
    blockB: ops.blockK,
    precond: (R) => ops.precond(R),
  });
  const factors = r.values.map((t) => (t < -floor ? -1 / t : Infinity));
  return { factors, modes: r.vectors.map((v) => ops.expand(v)), converged: r.converged, iterations: r.iterations, residuals: r.residuals, maxFactor };
}
