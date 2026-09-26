// Linear dynamic response by modal superposition (main thread, from a modal study).
//
// Each mode i (natural frequency w_i, mass-normalized shape phi_i) obeys
//   q_i'' + 2 z w_i q_i' + w_i^2 q_i = G_i g(t),   G_i = phi_i^T f
// for a load pattern f scaled in time by g(t). The response uses the mode-acceleration method:
//   u = u_static g(t) + sum_i phi_i (q_i - G_i g / w_i^2)
// so the modes that were not computed still contribute their static share exactly (u_static is
// K^-1 f). Stresses superpose the same way. For base shaking the pattern is the inertia load
// f = -M r of a unit ground acceleration along r, and u is the motion relative to the base.
import { vonMisesAt } from './solver.js';

/**
 * @typedef {object} ModalBasis
 * @property {number[]} omegas         rad/s
 * @property {number[]} gamma          G_i for the load pattern
 * @property {Float32Array[]} modeU    per mode, vertex displacement per unit modal coordinate (3 per vertex) [model units]
 * @property {Float32Array[]} modeS    per mode, vertex stress tensor per unit modal coordinate (6 per vertex) [Pa]
 * @property {Float32Array} staticU    vertex displacement under the pattern (g = 1)
 * @property {Float32Array} staticS    vertex stress tensor under the pattern (g = 1)
 */

/** Complex modal coefficients (re, im) at forcing frequency W [rad/s], including the static correction. */
export function harmonicCoefficients(basis, W, zeta) {
  const re = [], im = [];
  basis.omegas.forEach((w, i) => {
    const a = w * w - W * W, b = 2 * zeta * w * W, d = a * a + b * b;
    const G = basis.gamma[i];
    re.push(G * (a / d - 1 / (w * w)));
    im.push(G * (-b / d));
  });
  return { re, im };
}

/** Peak (over the cycle) displacement magnitude and von Mises at every vertex for forcing frequency W. */
export function harmonicField(basis, W, zeta, { phases = 16, vertices = null } = {}) {
  const { re, im } = harmonicCoefficients(basis, W, zeta);
  const nV = basis.staticU.length / 3;
  const list = vertices || null;
  const count = list ? list.length : nV;
  const disp = new Float32Array(count), vm = new Float32Array(count);
  const ur = [0, 0, 0], ui = [0, 0, 0], sr = new Float64Array(6), si = new Float64Array(6), s = new Float64Array(6);
  const cos = [], sin = [];
  for (let p = 0; p < phases; p++) { cos.push(Math.cos((Math.PI * p) / phases)); sin.push(Math.sin((Math.PI * p) / phases)); }
  const m = re.length;
  for (let q = 0; q < count; q++) {
    const v = list ? list[q] : q;
    for (let d = 0; d < 3; d++) { ur[d] = basis.staticU[3 * v + d]; ui[d] = 0; }
    for (let c = 0; c < 6; c++) { sr[c] = basis.staticS[6 * v + c]; si[c] = 0; }
    for (let i = 0; i < m; i++) {
      const U = basis.modeU[i], S = basis.modeS[i];
      for (let d = 0; d < 3; d++) { ur[d] += re[i] * U[3 * v + d]; ui[d] += im[i] * U[3 * v + d]; }
      for (let c = 0; c < 6; c++) { sr[c] += re[i] * S[6 * v + c]; si[c] += im[i] * S[6 * v + c]; }
    }
    if (Number.isNaN(ur[0])) { disp[q] = vm[q] = NaN; continue; }
    // |u| amplitude: the largest length over the cycle of Re(u e^{i t})
    let best = 0, bestS = 0;
    for (let p = 0; p < phases; p++) {
      const x = ur[0] * cos[p] - ui[0] * sin[p], y = ur[1] * cos[p] - ui[1] * sin[p], z = ur[2] * cos[p] - ui[2] * sin[p];
      best = Math.max(best, x * x + y * y + z * z);
      for (let c = 0; c < 6; c++) s[c] = sr[c] * cos[p] - si[c] * sin[p];
      bestS = Math.max(bestS, vonMisesAt(s));
    }
    disp[q] = Math.sqrt(best);
    vm[q] = bestS;
  }
  return { disp, vm };
}

/** Real displacement field at one phase angle (for animating the steady vibration). */
export function harmonicShape(basis, W, zeta, phase) {
  const { re, im } = harmonicCoefficients(basis, W, zeta);
  const c = Math.cos(phase), s = Math.sin(phase);
  const u = Float32Array.from(basis.staticU, (x) => x * c);
  re.forEach((r, i) => {
    const k = r * c - im[i] * s, U = basis.modeU[i];
    for (let t = 0; t < u.length; t++) u[t] += k * U[t];
  });
  return u;
}

/** Excitation time function g(t) (dimensionless multiplier of the load pattern). */
export function excitation({ kind, amplitude = 1, freq = 10, duration = 0.01, total = 0.05 }) {
  switch (kind) {
    case 'sine': return (t) => (t <= total ? amplitude * Math.sin(2 * Math.PI * freq * t) : 0);
    case 'step': return (t) => (t >= 0 ? amplitude : 0);
    case 'quake': {
      // synthetic ground motion: filtered noise (1-10 Hz band) under a build-up / decay envelope
      let seed = 2024;
      const rand = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff - 0.5; };
      const comps = Array.from({ length: 40 }, (_, k) => ({ f: 1 + (9 * k) / 39, ph: 2 * Math.PI * rand(), a: 0.5 + rand() }));
      let peak = 0;
      const raw = (t) => {
        const env = t < 0.15 * total ? (t / (0.15 * total)) ** 2 : Math.exp((-3 * (t - 0.15 * total)) / total);
        let s = 0;
        for (const c of comps) s += c.a * Math.sin(2 * Math.PI * c.f * t + c.ph);
        return env * s;
      };
      for (let t = 0; t < total; t += total / 2000) peak = Math.max(peak, Math.abs(raw(t)));
      return (t) => (t >= 0 && t <= total ? (amplitude * raw(t)) / (peak || 1) : 0);
    }
    default: // half-sine shock pulse
      return (t) => (t >= 0 && t <= duration ? amplitude * Math.sin((Math.PI * t) / duration) : 0);
  }
}

/**
 * Modal coordinates q_i(t) for excitation g(t), integrated per mode with the average-acceleration
 * Newmark method on a step of at most 1/24 of the shortest period (period error < 0.2%).
 * Returns times and, per mode, q_i(t) - G_i g(t) / w_i^2 (the dynamic part of the response).
 */
export function timeHistory(basis, g, { zeta, total, maxSteps = 20000 }) {
  const wMax = Math.max(...basis.omegas);
  const dt = Math.max(total / maxSteps, Math.min((2 * Math.PI) / wMax / 24, total / 400));
  const steps = Math.ceil(total / dt);
  const times = new Float64Array(steps + 1), gv = new Float64Array(steps + 1);
  for (let k = 0; k <= steps; k++) { times[k] = k * dt; gv[k] = g(k * dt); }
  const dyn = basis.omegas.map((w, i) => {
    const G = basis.gamma[i], c = 2 * zeta * w, k = w * w;
    const kh = k + (2 * c) / dt + 4 / (dt * dt), A = 4 / dt + 2 * c;
    const out = new Float32Array(steps + 1);
    let x = 0, v = 0, acc = G * gv[0];
    out[0] = -(G * gv[0]) / k;
    for (let n = 0; n < steps; n++) {
      const dp = G * (gv[n + 1] - gv[n]) + A * v + 2 * acc;
      const dx = dp / kh;
      const dv = (2 * dx) / dt - 2 * v;
      const da = (4 * dx) / (dt * dt) - (4 * v) / dt - 2 * acc;
      x += dx; v += dv; acc += da;
      out[n + 1] = x - (G * gv[n + 1]) / k;
    }
    return out;
  });
  return { times, g: gv, dyn, dt };
}

/** Vertex displacement (3 per vertex) at time index k of a time history. */
export function shapeAt(basis, hist, k) {
  const gk = hist.g[k];
  const u = Float32Array.from(basis.staticU, (x) => x * gk);
  hist.dyn.forEach((q, i) => {
    const c = q[k], U = basis.modeU[i];
    if (c !== 0) for (let t = 0; t < u.length; t++) u[t] += c * U[t];
  });
  return u;
}

/** von Mises at every vertex (or a subset) at time index k. */
export function stressAt(basis, hist, k, vertices = null) {
  const gk = hist.g[k];
  const nV = basis.staticS.length / 6;
  const list = vertices;
  const count = list ? list.length : nV;
  const vm = new Float32Array(count), s = new Float64Array(6);
  const coef = hist.dyn.map((q) => q[k]);
  for (let q = 0; q < count; q++) {
    const v = list ? list[q] : q;
    for (let c = 0; c < 6; c++) s[c] = basis.staticS[6 * v + c] * gk;
    for (let i = 0; i < coef.length; i++) {
      const a = coef[i];
      if (a === 0) continue;
      const S = basis.modeS[i];
      for (let c = 0; c < 6; c++) s[c] += a * S[6 * v + c];
    }
    vm[q] = Number.isNaN(s[0]) ? NaN : vonMisesAt(s);
  }
  return vm;
}
