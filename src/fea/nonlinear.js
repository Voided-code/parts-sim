// Nonlinear static analysis: large displacements and metal plasticity.
//
// Geometry: element-independent co-rotational formulation. Each voxel's rigid rotation is taken
// from the polar decomposition of its mean deformation gradient; the (small) strain left after
// removing it is resisted by the same incompatible-mode element used in the linear study. That
// keeps the linear study's accuracy for bending and adds large rotations and stress stiffening.
//
// Material: von Mises (J2) plasticity with linear isotropic hardening from the yield strength to
// the tensile strength over the elongation at break, evaluated from each voxel's mean strain
// (radial return). Brittle materials stay elastic and fail at their tensile strength.
//
// Equilibrium: Newton iterations under load control with adaptive increments. Each Newton step
// solves with a symmetric tangent (rotated elastic stiffness minus the plastic stiffness loss)
// by flexible conjugate gradients preconditioned with the elastic multigrid (CPU or GPU).
//
// Normalized units (E = 1, voxel = 1): displacement u~ = u / h, force f~ = f / (E h^2),
// stress s~ = s / E. These make K u~ = f~ exactly the linear solver's system.
import { elasticityMatrix, hexElement, HEX_NODES } from './hex8.js';
import { CENTROID_B as B0, GEO_T } from './eigen.js';
import { corner } from './solver.js';

const REF = HEX_NODES.map(([x, y, z]) => [x - 0.5, y - 0.5, z - 0.5]);

/** Rotation part of a 3x3 matrix (row-major) by Higham's scaled Newton iteration. */
export function polarRotation(F, out = new Float64Array(9)) {
  let R = Float64Array.from(F);
  const inv = new Float64Array(9);
  for (let it = 0; it < 30; it++) {
    const a = R[0], b = R[1], c = R[2], d = R[3], e = R[4], f = R[5], g = R[6], h = R[7], i = R[8];
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    if (!(Math.abs(det) > 1e-300)) break;
    // inverse transpose = cofactor / det
    inv[0] = A / det; inv[1] = B / det; inv[2] = C / det;
    inv[3] = -(b * i - c * h) / det; inv[4] = (a * i - c * g) / det; inv[5] = -(a * h - b * g) / det;
    inv[6] = (b * f - c * e) / det; inv[7] = -(a * f - c * d) / det; inv[8] = (a * e - b * d) / det;
    // scaling (Frobenius) speeds up convergence far from orthogonal
    let nR = 0, nI = 0;
    for (let k = 0; k < 9; k++) { nR += R[k] * R[k]; nI += inv[k] * inv[k]; }
    const g2 = Math.sqrt(Math.sqrt(nI / nR)) || 1;
    let diff = 0;
    for (let k = 0; k < 9; k++) {
      const v = 0.5 * (g2 * R[k] + inv[k] / g2);
      diff = Math.max(diff, Math.abs(v - R[k]));
      R[k] = v;
    }
    if (diff < 1e-13) break;
  }
  out.set(R);
  return out;
}

/** Flexible preconditioned conjugate gradients: op(x, y), precond(r) -> z (may be async). */
export async function fcg(op, precond, b, { tol = 1e-3, maxIter = 200, x0 = null, onIter = null } = {}) {
  const n = b.length;
  const x = x0 ? Float64Array.from(x0) : new Float64Array(n);
  const r = new Float64Array(n), q = new Float64Array(n), p = new Float64Array(n), rOld = new Float64Array(n);
  const dot = (a, c) => { let s = 0; for (let i = 0; i < n; i++) s += a[i] * c[i]; return s; };
  const bn = Math.sqrt(dot(b, b));
  if (!(bn > 0)) return { x: x.fill(0), iterations: 0, residual: 0 };
  if (x0) { op(x, q); for (let i = 0; i < n; i++) r[i] = b[i] - q[i]; } else r.set(b);
  let rel = Math.sqrt(dot(r, r)) / bn;
  if (rel <= tol) return { x, iterations: 0, residual: rel };
  let z = await precond(r);
  p.set(z);
  let rz = dot(r, z);
  let it = 0;
  for (; it < maxIter; it++) {
    op(p, q);
    const pq = dot(p, q);
    if (!(pq > 0)) break;
    const alpha = rz / pq;
    rOld.set(r);
    for (let i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
    rel = Math.sqrt(dot(r, r)) / bn;
    onIter?.(it + 1, rel);
    if (rel <= tol) { it++; break; }
    const zn = await precond(r);
    let num = 0;
    for (let i = 0; i < n; i++) num += zn[i] * (r[i] - rOld[i]);
    const beta = Math.max(0, num / rz);
    rz = dot(r, zn);
    for (let i = 0; i < n; i++) p[i] = zn[i] + beta * p[i];
  }
  return { x, iterations: it, residual: rel };
}

/**
 * Hardening modulus (normalized by E) for linear hardening from yield to UTS at the elongation.
 * Returns null for brittle materials (no plasticity).
 */
export function hardeningFor(material) {
  if (material.brittle) return null;
  const E = material.E * 1e3; // GPa -> MPa
  const eu = material.elongation > 0 ? material.elongation : 0.15;
  const epsPlastic = Math.max(1e-3, eu - material.uts / E);
  const H = Math.max(0, (material.uts - material.yield) / epsPlastic); // MPa
  return { yield: material.yield / E, H: Math.max(H / E, 1e-5), eu, uts: material.uts / E };
}

// One thread's scratch vectors for the element loops below (each helper thread has its own module).
const W = {
  ue: new Float64Array(24), ul: new Float64Array(24), fl: new Float64Array(24),
  eps: new Float64Array(6), sig: new Float64Array(6), mean: new Float64Array(6),
};

/**
 * Local (rotation-free) displacements ul of element e from global u~; stores the element rotation.
 * M is a NonlinearModel or a helper thread's shared copy of its state (threads.js), L its level.
 */
function localDisplacement(M, L, e, u, ue, ul) {
  const n0 = L.base[e];
  for (let a = 0; a < 8; a++) {
    const n = 3 * (n0 + L.off[a]);
    ue[3 * a] = u[n]; ue[3 * a + 1] = u[n + 1]; ue[3 * a + 2] = u[n + 2];
  }
  if (!M.large) { ul.set(ue); return; }
  // mean deformation gradient F = I + sum_a u_a (x) grad N_a(centre)
  const F = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let a = 0; a < 8; a++) {
    const gx = B0[3 * a], gy = B0[24 + 3 * a + 1], gz = B0[48 + 3 * a + 2];
    for (let i = 0; i < 3; i++) {
      const v = ue[3 * a + i];
      F[3 * i] += v * gx; F[3 * i + 1] += v * gy; F[3 * i + 2] += v * gz;
    }
  }
  const R = M.R.subarray(9 * e, 9 * e + 9);
  polarRotation(F, R);
  let cx = 0, cy = 0, cz = 0;
  for (let a = 0; a < 8; a++) { cx += ue[3 * a]; cy += ue[3 * a + 1]; cz += ue[3 * a + 2]; }
  cx /= 8; cy /= 8; cz /= 8;
  for (let a = 0; a < 8; a++) {
    const X = REF[a];
    const dx = X[0] + ue[3 * a] - cx, dy = X[1] + ue[3 * a + 1] - cy, dz = X[2] + ue[3 * a + 2] - cz;
    // R^T d - X
    ul[3 * a] = R[0] * dx + R[3] * dy + R[6] * dz - X[0];
    ul[3 * a + 1] = R[1] * dx + R[4] * dy + R[7] * dz - X[1];
    ul[3 * a + 2] = R[2] * dx + R[5] * dy + R[8] * dz - X[2];
  }
}

/** Local element forces fl rotated to global and times rho, into element e's 24 entries of fe. */
function globalForces(M, e, fl, rho, fe) {
  const o = 24 * e;
  if (!M.large) {
    for (let c = 0; c < 24; c++) fe[o + c] = rho * fl[c];
    return;
  }
  const R = M.R, r = 9 * e;
  for (let a = 0; a < 8; a++) {
    const x = fl[3 * a], y = fl[3 * a + 1], z = fl[3 * a + 2], f = o + 3 * a;
    fe[f] = rho * (R[r] * x + R[r + 1] * y + R[r + 2] * z);
    fe[f + 1] = rho * (R[r + 3] * x + R[r + 4] * y + R[r + 5] * z);
    fe[f + 2] = rho * (R[r + 6] * x + R[r + 7] * y + R[r + 8] * z);
  }
}

/**
 * Internal forces of the elements e0 <= e < e1 at u~ into fe (24 per element, global), updating
 * each Gauss point's trial plastic state from the committed one. Returns their largest equivalent
 * plastic strain.
 */
export function forceElements(M, L, u, fe, e0, e1) {
  const K = M.K0, D = M.D, G = M.G, pl = M.plastic;
  const { fl, ul, eps, sig, mean } = W;
  let maxAlpha = 0;
  for (let e = e0; e < e1; e++) {
    localDisplacement(M, L, e, u, W.ue, ul);
    mean.fill(0);
    if (!pl) {
      for (let r = 0; r < 24; r++) {
        let s = 0;
        const row = r * 24;
        for (let c = 0; c < 24; c++) s += K[row + c] * ul[c];
        fl[r] = s;
      }
      if (M.geo) {
        // mean stress D B0 u for the stress stiffness
        eps.fill(0);
        for (let c = 0; c < 24; c++) { const v = ul[c]; if (v !== 0) for (let i = 0; i < 6; i++) eps[i] += B0[i * 24 + c] * v; }
        for (let i = 0; i < 6; i++) { let s = 0; for (let j = 0; j < 6; j++) s += D[i * 6 + j] * eps[j]; mean[i] = s; }
      }
    } else {
      fl.fill(0);
      let bits = 0;
      for (let g = 0; g < 8; g++) {
        const B = M.gaussB[g], o = 48 * e + 6 * g, oa = 8 * e + g;
        eps.fill(0);
        for (let c = 0; c < 24; c++) { const v = ul[c]; if (v !== 0) for (let i = 0; i < 6; i++) eps[i] += B[i * 24 + c] * v; }
        for (let i = 0; i < 6; i++) eps[i] -= M.ep[o + i];
        for (let i = 0; i < 6; i++) { let s = 0; for (let j = 0; j < 6; j++) s += D[i * 6 + j] * eps[j]; sig[i] = s; }
        let alpha = M.alpha[oa];
        for (let i = 0; i < 6; i++) M.epTrial[o + i] = M.ep[o + i];
        const p = (sig[0] + sig[1] + sig[2]) / 3;
        const s0 = sig[0] - p, s1 = sig[1] - p, s2 = sig[2] - p;
        const q = Math.sqrt(1.5 * (s0 * s0 + s1 * s1 + s2 * s2 + 2 * (sig[3] ** 2 + sig[4] ** 2 + sig[5] ** 2)));
        const fy = q - (pl.yield + pl.H * alpha);
        if (fy > 0 && q > 0) {
          // radial return
          const dg = fy / (3 * G + pl.H);
          const k = (1.5 / q) * dg;
          // consistent (algorithmic) tangent of the radial return:
          // D_ep = D - 2G(1 - th) I_dev - 2G thb n (x) n, n = s / |s|
          const th = 1 - (3 * G * dg) / q, thb = 1 / (1 + pl.H / (3 * G)) - (1 - th);
          const f = M.flow, of = 64 * e + 8 * g, ns = 1 / (Math.sqrt(2 / 3) * q);
          f[of] = s0 * ns; f[of + 1] = s1 * ns; f[of + 2] = s2 * ns; f[of + 3] = sig[3] * ns; f[of + 4] = sig[4] * ns; f[of + 5] = sig[5] * ns;
          f[of + 6] = 2 * G * (1 - th); f[of + 7] = 2 * G * thb;
          M.epTrial[o] += k * s0; M.epTrial[o + 1] += k * s1; M.epTrial[o + 2] += k * s2;
          M.epTrial[o + 3] += 2 * k * sig[3]; M.epTrial[o + 4] += 2 * k * sig[4]; M.epTrial[o + 5] += 2 * k * sig[5];
          const shrink = 1 - (3 * G * dg) / q;
          sig[0] = p + s0 * shrink; sig[1] = p + s1 * shrink; sig[2] = p + s2 * shrink;
          sig[3] *= shrink; sig[4] *= shrink; sig[5] *= shrink;
          alpha += dg;
          bits |= 1 << g;
        }
        M.alphaTrial[oa] = alpha;
        if (alpha > maxAlpha) maxAlpha = alpha;
        for (let i = 0; i < 6; i++) { M.gpStress[o + i] = sig[i]; mean[i] += sig[i] / 8; }
        // f += Bbar^T sigma / 8
        for (let c = 0; c < 24; c++) {
          let v = 0;
          for (let i = 0; i < 6; i++) v += B[i * 24 + c] * sig[i];
          fl[c] += v / 8;
        }
      }
      M.yielding[e] = bits;
    }
    if (M.geo) {
      const gm = M.geo.subarray(64 * e, 64 * e + 64);
      gm.fill(0);
      for (let c = 0; c < 6; c++) {
        const sc = mean[c];
        if (sc === 0) continue;
        const T = GEO_T[c];
        for (let i = 0; i < 64; i++) gm[i] += sc * T[i];
      }
    }
    globalForces(M, e, fl, L.rho[e], fe);
  }
  return maxAlpha;
}

/**
 * Tangent forces J_e x of the elements e0 <= e < e1 into fe, with the rotations, stresses and
 * plastic state of the last forceElements call: rotated elastic stiffness + stress stiffness -
 * plastic stiffness loss.
 */
export function tangentElements(M, L, x, fe, e0, e1) {
  const K = M.K0, R = M.R;
  const { ue, ul, fl, eps } = W;
  for (let e = e0; e < e1; e++) {
    const n0 = L.base[e], o = 9 * e;
    for (let a = 0; a < 8; a++) {
      const n = 3 * (n0 + L.off[a]);
      ue[3 * a] = x[n]; ue[3 * a + 1] = x[n + 1]; ue[3 * a + 2] = x[n + 2];
    }
    if (M.large) {
      for (let a = 0; a < 8; a++) {
        const vx = ue[3 * a], vy = ue[3 * a + 1], vz = ue[3 * a + 2];
        ul[3 * a] = R[o] * vx + R[o + 3] * vy + R[o + 6] * vz;
        ul[3 * a + 1] = R[o + 1] * vx + R[o + 4] * vy + R[o + 7] * vz;
        ul[3 * a + 2] = R[o + 2] * vx + R[o + 5] * vy + R[o + 8] * vz;
      }
    } else ul.set(ue);
    for (let r = 0; r < 24; r++) {
      let s = 0;
      const row = r * 24;
      for (let c = 0; c < 24; c++) s += K[row + c] * ul[c];
      fl[r] = s;
    }
    if (M.geo) {
      const g = M.geo, o64 = 64 * e;
      for (let a = 0; a < 8; a++) {
        let sx = 0, sy = 0, sz = 0;
        for (let b = 0; b < 8; b++) {
          const v = g[o64 + a * 8 + b];
          sx += v * ul[3 * b]; sy += v * ul[3 * b + 1]; sz += v * ul[3 * b + 2];
        }
        fl[3 * a] += sx; fl[3 * a + 1] += sy; fl[3 * a + 2] += sz;
      }
    }
    const bits = M.plastic ? M.yielding[e] : 0;
    if (bits) {
      for (let gp = 0; gp < 8; gp++) {
        if (!(bits & (1 << gp))) continue;
        const B = M.gaussB[gp], f = M.flow, of = 64 * e + 8 * gp;
        eps.fill(0);
        for (let c = 0; c < 24; c++) { const v = ul[c]; if (v !== 0) for (let i = 0; i < 6; i++) eps[i] += B[i * 24 + c] * v; }
        const em = (eps[0] + eps[1] + eps[2]) / 3;
        const ne = f[of] * eps[0] + f[of + 1] * eps[1] + f[of + 2] * eps[2] + f[of + 3] * eps[3] + f[of + 4] * eps[4] + f[of + 5] * eps[5];
        const c1 = f[of + 6], c2 = f[of + 7] * ne;
        // stress-like loss t = c1 dev(eps) + c2 n (tensor shear = engineering / 2)
        const t = W.sig;
        t[0] = c1 * (eps[0] - em) + c2 * f[of]; t[1] = c1 * (eps[1] - em) + c2 * f[of + 1]; t[2] = c1 * (eps[2] - em) + c2 * f[of + 2];
        t[3] = c1 * 0.5 * eps[3] + c2 * f[of + 3]; t[4] = c1 * 0.5 * eps[4] + c2 * f[of + 4]; t[5] = c1 * 0.5 * eps[5] + c2 * f[of + 5];
        for (let c = 0; c < 24; c++) {
          let v = 0;
          for (let i = 0; i < 6; i++) v += B[i * 24 + c] * t[i];
          fl[c] -= v / 8;
        }
      }
    }
    globalForces(M, e, fl, L.rho[e], fe);
  }
}

/**
 * out = the element vectors fe summed at the nodes n0 <= n < n1 (0 on held DOFs). Each node adds
 * its voxels in element order, as a scatter over the elements would.
 */
export function gatherNodes(L, fe, out, n0, n1) {
  const { NX, NY, nx, ny, nz, emap, fixed } = L;
  const NXY = NX * NY;
  let i = n0 % NX, j = ((n0 / NX) | 0) % NY, k = (n0 / NXY) | 0;
  for (let n = n0; n < n1; n++) {
    let a0 = 0, a1 = 0, a2 = 0;
    for (let v = 0; v < 8; v++) {
      const di = v & 1, dj = (v >> 1) & 1, dk = v >> 2;
      const ei = i + di - 1, ej = j + dj - 1, ek = k + dk - 1;
      if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= nz) continue;
      const e = emap[ei + nx * (ej + ny * ek)];
      if (e < 0) continue;
      const f = 24 * e + 3 * corner(1 - di, 1 - dj, 1 - dk);
      a0 += fe[f]; a1 += fe[f + 1]; a2 += fe[f + 2];
    }
    const o = 3 * n;
    out[o] = fixed[o] ? 0 : a0;
    out[o + 1] = fixed[o + 1] ? 0 : a1;
    out[o + 2] = fixed[o + 2] ? 0 : a2;
    if (++i === NX) { i = 0; if (++j === NY) { j = 0; k++; } }
  }
}

export class NonlinearModel {
  /**
   * @param {import('./solver.js').VoxelFEA} fea  elastic model (its levels[0] carries the mesh and held DOFs)
   * @param {object} o
   * @param {boolean} o.largeDisplacement
   * @param {{yield: number, H: number}|null} o.plastic   normalized by E (hardeningFor)
   */
  constructor(fea, { largeDisplacement = true, plastic = null }) {
    this.fea = fea;
    this.L = fea.levels[0];
    const nE = this.nE = this.L.elems.length;
    this.large = largeDisplacement;
    this.plastic = plastic;
    const nu = fea.nu;
    const el = hexElement(nu);
    this.D = elasticityMatrix(nu);
    this.G = 1 / (2 * (1 + nu));
    this.K0 = fea.K0;
    this.gaussB = el.gaussB;
    this.cornerS = el.cornerStress;
    // the element state lives in shared memory when fea has helper threads (share())
    const shared = !!fea.threads;
    const arr = (Type, n) => new Type(shared ? new SharedArrayBuffer(n * Type.BYTES_PER_ELEMENT) : n);
    // plastic state at the 8 Gauss points of each voxel: committed (converged) and trial
    if (plastic) {
      this.ep = arr(Float64Array, 48 * nE);
      this.alpha = arr(Float64Array, 8 * nE);
      this.epTrial = arr(Float64Array, 48 * nE);
      this.alphaTrial = arr(Float64Array, 8 * nE);
      this.gpStress = arr(Float64Array, 48 * nE);
      this.flow = arr(Float64Array, 64 * nE); // consistent tangent data per yielding point: n (6), c1, c2
      this.yielding = arr(Uint8Array, nE); // bit g set = Gauss point g is yielding in this iteration
      this.everPlastic = new Uint8Array(nE);
    }
    this.R = arr(Float64Array, 9 * nE);
    for (let e = 0; e < nE; e++) { this.R[9 * e] = this.R[9 * e + 4] = this.R[9 * e + 8] = 1; }
    // stress stiffness of each voxel (8 x 8, from its current mean stress) for the Newton tangent
    this.geo = largeDisplacement ? arr(Float64Array, 64 * nE) : null;
    // each element's forces (global, 24 per voxel) before they are summed at the nodes
    this.fe = arr(Float64Array, 24 * nE);
    this.ul = new Float64Array(24);
    this.sig = new Float64Array(6);
    this.threads = null;
  }

  /** Hands the element state to fea's helper threads, which then share the element loops. */
  async share() {
    const t = this.fea.threads;
    if (!t) return;
    const { K0, D, G, gaussB, large, plastic, R, geo, ep, alpha, epTrial, alphaTrial, gpStress, flow, yielding, fe } = this;
    await t.share({ nl: { K0, D, G, gaussB, large, plastic, R, geo, ep, alpha, epTrial, alphaTrial, gpStress, flow, yielding }, fe });
    this.threads = t;
  }

  /** Local (rotation-free) element displacements from global u~ into this.ul; stores the element rotation. */
  localDisplacement(e, u) {
    localDisplacement(this, this.L, e, u, W.ue, this.ul);
  }

  /**
   * Internal forces F_int(u~) into `out`, updating each Gauss point's trial plastic state from the
   * committed one. Returns the largest equivalent plastic strain.
   */
  internalForce(u, out) {
    if (this.threads) return this.threads.internalForce(u, out);
    const maxAlpha = forceElements(this, this.L, u, this.fe, 0, this.nE);
    gatherNodes(this.L, this.fe, out, 0, this.L.nNodes);
    return maxAlpha;
  }

  /**
   * Symmetric tangent product y = J x with the rotations, stresses and plastic state of the last
   * internalForce call: rotated elastic stiffness + stress stiffness - plastic stiffness loss.
   */
  applyTangent(x, y) {
    if (this.threads) { this.threads.applyTangent(x, y); return; }
    tangentElements(this, this.L, x, this.fe, 0, this.nE);
    gatherNodes(this.L, this.fe, y, 0, this.L.nNodes);
  }

  commit() {
    if (!this.plastic) return;
    this.ep.set(this.epTrial);
    this.alpha.set(this.alphaTrial);
    for (let e = 0; e < this.nE; e++) {
      if (this.everPlastic[e]) continue;
      for (let g = 0; g < 8; g++) if (this.alpha[8 * e + g] > 0) { this.everPlastic[e] = 1; break; }
    }
  }

  /**
   * Nodal fields at displacement u~ (after internalForce at the same u): von Mises and max/min
   * principal stress (units of E) and equivalent plastic strain. Elastic voxels use the same
   * corner stress recovery as the linear study; voxels that have yielded use their Gauss-point
   * stresses (each corner takes the nearest point), which stay on the yield surface.
   */
  nodalFields(u) {
    const L = this.L;
    const vm = new Float32Array(L.nNodes), p1 = new Float32Array(L.nNodes), p3 = new Float32Array(L.nNodes);
    const pe = new Float32Array(L.nNodes), w = new Float32Array(L.nNodes);
    const sig = this.sig, pr = [0, 0, 0];
    let maxP1 = 0;
    for (let e = 0; e < this.nE; e++) {
      const rho = L.rho[e];
      // voxels that have yielded or are close to it report their Gauss-point stresses, which never
      // leave the yield surface (corner extrapolation next to a plastic zone would overshoot it)
      let near = false;
      if (this.plastic && !this.everPlastic[e]) {
        for (let g = 0; g < 8 && !near; g++) {
          const o = 48 * e + 6 * g, s = this.gpStress;
          const q = Math.sqrt(0.5 * ((s[o] - s[o + 1]) ** 2 + (s[o + 1] - s[o + 2]) ** 2 + (s[o + 2] - s[o]) ** 2) + 3 * (s[o + 3] ** 2 + s[o + 4] ** 2 + s[o + 5] ** 2));
          near = q > 0.85 * this.plastic.yield;
        }
      }
      const yielded = this.plastic && (this.everPlastic[e] || near);
      if (!yielded) this.localDisplacement(e, u);
      let aMean = 0;
      if (this.plastic) for (let g = 0; g < 8; g++) aMean += this.alpha[8 * e + g] / 8;
      for (let a = 0; a < 8; a++) {
        if (yielded) {
          const o = 48 * e + 6 * a;
          for (let i = 0; i < 6; i++) sig[i] = this.gpStress[o + i];
        } else {
          const S = this.cornerS[a];
          for (let i = 0; i < 6; i++) {
            let s = 0;
            for (let c = 0; c < 24; c++) s += S[i * 24 + c] * this.ul[c];
            sig[i] = s;
          }
        }
        const v = Math.sqrt(0.5 * ((sig[0] - sig[1]) ** 2 + (sig[1] - sig[2]) ** 2 + (sig[2] - sig[0]) ** 2) + 3 * (sig[3] ** 2 + sig[4] ** 2 + sig[5] ** 2));
        principal(sig[0], sig[1], sig[2], sig[3], sig[4], sig[5], pr);
        if (pr[0] > maxP1) maxP1 = pr[0];
        const n = L.base[e] + L.off[a];
        vm[n] += rho * v; p1[n] += rho * pr[0]; p3[n] += rho * pr[2];
        pe[n] += rho * aMean;
        w[n] += rho;
      }
    }
    for (let n = 0; n < L.nNodes; n++) if (w[n] > 0) { vm[n] /= w[n]; p1[n] /= w[n]; p3[n] /= w[n]; pe[n] /= w[n]; }
    return { vm, p1, p3, pe, maxP1 };
  }
}

/**
 * Newton solve of F_int(u) = lam * f at one load level, starting from u (updated in place).
 * @returns {{converged: boolean, iterations: number, residual: number, maxAlpha: number}}
 */
export async function equilibrate(model, u, fLam, precond, { tol = 1e-4, maxNewton = 25, alphaLimit = Infinity, onIter = null } = {}) {
  const n = u.length;
  const fint = new Float64Array(n), r = new Float64Array(n);
  const fixed = model.L.fixed;
  let fn = 0;
  for (let i = 0; i < n; i++) if (!fixed[i]) fn += fLam[i] * fLam[i];
  fn = Math.sqrt(fn) || 1;
  let rel = Infinity, maxAlpha = 0, it = 0, prev = Infinity, grow = 0;
  for (; it < maxNewton; it++) {
    maxAlpha = model.internalForce(u, fint);
    let rr = 0;
    for (let i = 0; i < n; i++) { r[i] = fixed[i] ? 0 : fLam[i] - fint[i]; rr += r[i] * r[i]; }
    rel = Math.sqrt(rr) / fn;
    if (onIter && onIter(it, rel) === true) throw Object.assign(new Error('Cancelled'), { cancelled: true });
    if (!Number.isFinite(rel)) break;
    if (rel <= tol) return { converged: true, iterations: it, residual: rel, maxAlpha };
    // diverging: let the caller cut the increment (the first corrector may overshoot, so only
    // persistent growth, a residual that stays above the load, or absurd plastic strains count)
    if (it >= 2 && rel > 0.9 * prev) { if (++grow >= 3) break; } else grow = 0;
    if ((it >= 6 && rel > 1) || maxAlpha > alphaLimit) break;
    prev = rel;
    const sol = await fcg((x, y) => model.applyTangent(x, y), precond, r, { tol: 0.05, maxIter: 80 });
    for (let i = 0; i < n; i++) u[i] += sol.x[i];
  }
  return { converged: false, iterations: it, residual: rel, maxAlpha };
}

/**
 * Ramp the load factor from 0 with adaptive increments.
 * - target: load factor to reach (1 = the applied loads), or Infinity to continue until the part
 *   collapses (no equilibrium any more), tears (plastic strain at the elongation), cracks (brittle:
 *   stress at the tensile strength) or deforms grossly.
 * onStep({lam, u, ...}) is called after each converged increment.
 */
export async function loadRamp(model, f, precond, { target = 1, steps = 10, maxDisp = Infinity, rupture = Infinity, crackStress = Infinity, maxSteps = 80, stepAlpha = 0.01, stepDisp = Infinity, maxTime = Infinity, onStep = null, onIter = null, onAttempt = null }) {
  const started = performance.now();
  // generalized displacement along the loads, to watch the structure's stiffness
  let fAbs = 0;
  for (let i = 0; i < f.length; i++) if (!model.L.fixed[i]) fAbs += Math.abs(f[i]);
  const along = (u) => { let w = 0; for (let i = 0; i < f.length; i++) if (!model.L.fixed[i]) w += f[i] * u[i]; return w / (fAbs || 1); };
  let k0 = 0, Dprev = 0, lamPrev = 0;
  const n = f.length;
  let u = new Float64Array(n), lam = 0;
  let dlam = Number.isFinite(target) ? target / steps : 1 / steps;
  const minStep = 1e-3 * (Number.isFinite(target) ? target : 1);
  const fl = new Float64Array(n);
  let reason = 'reached', count = 0, lastAlpha = 0, lastDisp = 0;
  while (lam < target - 1e-12 && count < maxSteps) {
    const lamTry = Math.min(target, lam + dlam);
    // start from the last equilibrium (the first Newton step is the tangent predictor; a secant
    // extrapolation would stretch rotating voxels and start far from equilibrium)
    const uTry = Float64Array.from(u);
    for (let i = 0; i < n; i++) fl[i] = lamTry * f[i];
    const alphaLimit = Math.min(Number.isFinite(rupture) ? 2 * rupture : Infinity, Number.isFinite(stepAlpha) ? lastAlpha + 10 * stepAlpha : Infinity);
    const res = await equilibrate(model, uTry, fl, precond, { alphaLimit, onIter: (it, rel) => onIter?.(lamTry, it, rel) });
    // keep increments small enough to trace the load-displacement curve (and find the collapse load)
    let dmaxTry = 0;
    if (res.converged) {
      for (let i = 0; i < n; i += 3) dmaxTry = Math.max(dmaxTry, uTry[i] * uTry[i] + uTry[i + 1] * uTry[i + 1] + uTry[i + 2] * uTry[i + 2]);
      dmaxTry = Math.sqrt(dmaxTry);
    }
    // a converged increment is kept even when it yielded or moved more than wanted (re-solving it
    // in smaller pieces costs far more than it gains); the next increment is just made smaller
    const tooBig = res.converged && (res.maxAlpha - lastAlpha > stepAlpha || dmaxTry - lastDisp > stepDisp);
    onAttempt?.({ lam: lamTry, converged: res.converged, tooBig, iterations: res.iterations, residual: res.residual, alpha: res.maxAlpha });
    if (!res.converged) {
      dlam *= 0.35;
      if (dlam < minStep) { reason = 'collapse'; break; }
      continue;
    }
    model.commit();
    u = uTry; lam = lamTry;
    const prevAlpha = lastAlpha, prevDisp = lastDisp;
    lastAlpha = res.maxAlpha; lastDisp = dmaxTry;
    count++;
    let dmax = 0;
    for (let i = 0; i < n; i += 3) dmax = Math.max(dmax, u[i] * u[i] + u[i + 1] * u[i + 1] + u[i + 2] * u[i + 2]);
    dmax = Math.sqrt(dmax);
    const fields = model.nodalFields(u);
    let smax = 0;
    for (const v of fields.vm) if (v > smax) smax = v;
    const p1max = fields.maxP1;
    await onStep?.({ lam, u, fields, iterations: res.iterations, maxAlpha: res.maxAlpha, maxDisp: dmax, maxStress: smax });
    if (res.maxAlpha >= rupture) { reason = 'rupture'; break; }
    if (p1max >= crackStress) { reason = 'crack'; break; }
    if (dmax > maxDisp) { reason = 'large deformation'; break; }
    if (performance.now() - started > maxTime * 1000) { reason = 'time limit'; break; }
    // collapse: the load hardly rises any more while the part keeps deflecting (a mechanism has
    // formed; with strain hardening the load creeps up a little but the part has failed)
    const D = along(u);
    if (count === 1 && D > 0) k0 = lam / D;
    else if (k0 > 0 && D > Dprev) {
      const k = (lam - lamPrev) / (D - Dprev);
      if (k < 0.03 * k0) { reason = 'collapse'; break; }
    }
    Dprev = D; lamPrev = lam;
    if (tooBig) {
      const over = Math.max((lastAlpha - prevAlpha) / stepAlpha, (lastDisp - prevDisp) / stepDisp);
      dlam = Math.max(minStep, dlam / Math.max(1.5, over));
    } else if (res.iterations <= 4) dlam *= 1.5;
    else if (res.iterations > 10) dlam *= 0.6;
  }
  if (count >= maxSteps && lam < target) reason = 'step limit';
  return { u, lam, reason };
}

function principal(sx, sy, sz, txy, tyz, tzx, out) {
  const q = (sx + sy + sz) / 3;
  const p1 = txy * txy + tyz * tyz + tzx * tzx;
  const p2 = (sx - q) ** 2 + (sy - q) ** 2 + (sz - q) ** 2 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  if (!(p > 0)) { out[0] = out[1] = out[2] = q; return out; }
  const b11 = (sx - q) / p, b22 = (sy - q) / p, b33 = (sz - q) / p, b12 = txy / p, b23 = tyz / p, b13 = tzx / p;
  const det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13);
  const phi = Math.acos(Math.min(1, Math.max(-1, det / 2))) / 3;
  out[0] = q + 2 * p * Math.cos(phi);
  out[2] = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  out[1] = 3 * q - out[0] - out[2];
  return out;
}
