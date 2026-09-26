// Fatigue (stress-life) evaluation of a linear static result under repeated loading.
//
// S-N curve: a straight line in log-log from the tensile strength at one cycle, through
// f * UTS at 1000 cycles, to the fatigue strength Se at Ne cycles (Basquin), with Se reduced
// by a surface-finish factor for metals (Marin, Shigley's Mechanical Engineering Design).
// Below Se, materials with an endurance limit (steel, titanium) last indefinitely; others keep
// following the line. Mean stress: Goodman correction for tensile means; compressive means are
// not credited. The equivalent stress is von Mises signed by the dominant principal stress.

const SURFACE = {
  // Marin surface factor ka = a * UTS[MPa]^b
  polished: null,
  ground: [1.58, -0.085],
  machined: [4.51, -0.265],
  'hot-rolled': [57.7, -0.718],
  forged: [272, -0.995],
};

export const FINISHES = [
  ['polished', 'Polished'],
  ['ground', 'Ground'],
  ['machined', 'Machined / cold drawn'],
  ['hot-rolled', 'Hot rolled'],
  ['forged', 'As forged'],
];

/**
 * @param {object} mat  material with uts [MPa] and fatigue: {Se [MPa], Ne, endurance, metal}
 * @param {string} finish
 */
export function snCurve(mat, finish = 'machined') {
  const uts = mat.uts;
  const fat = mat.fatigue || { Se: 0.4 * uts, Ne: 1e7, endurance: false, metal: false };
  let ka = 1;
  const s = SURFACE[finish];
  if (fat.metal && s) ka = Math.min(1, s[0] * Math.pow(uts, s[1]));
  const Se = Math.max(1e-3, fat.Se * ka);
  const Ne = fat.Ne;
  const S3 = Math.max(Se * 1.001, Math.min(0.9 * uts, uts)); // strength at 1000 cycles
  const b = Math.log10(Se / S3) / Math.log10(Ne / 1e3); // negative slope
  return { uts, Se, Ne, S3, b, ka, endurance: !!fat.endurance };
}

/** Strength [MPa] at N cycles. */
export function strengthAt(curve, N) {
  if (N <= 1) return curve.uts;
  if (N <= 1e3) return curve.uts * Math.pow(curve.S3 / curve.uts, Math.log10(N) / 3);
  if (N >= curve.Ne && curve.endurance) return curve.Se;
  return curve.S3 * Math.pow(N / 1e3, curve.b);
}

/** Cycles to failure at a fully reversed equivalent amplitude S [MPa]. */
export function cyclesToFailure(curve, S) {
  if (!(S > 0)) return Infinity;
  if (S >= curve.uts) return 1;
  if (S >= curve.S3) return Math.pow(10, (3 * Math.log10(S / curve.uts)) / Math.log10(curve.S3 / curve.uts));
  if (curve.endurance && S <= curve.Se) return Infinity;
  return 1e3 * Math.pow(S / curve.S3, 1 / curve.b);
}

/** Alternating and mean stress of a cycle between R*smax and smax. */
export function cycleStress(smax, R) {
  return { sa: (Math.abs(smax) * (1 - R)) / 2, sm: (smax * (1 + R)) / 2 };
}

/** Goodman equivalent fully reversed amplitude; Infinity when the mean alone reaches UTS. */
export function goodman(sa, sm, uts) {
  if (sm <= 0) return sa;
  if (sm >= uts) return Infinity;
  return sa / (1 - sm / uts);
}

/**
 * Per-vertex fatigue results from the stress at the peak of the load cycle.
 * @param {Float32Array} vm   von Mises [Pa] at the peak load
 * @param {Float32Array} p1   max principal [Pa]
 * @param {Float32Array} p3   min principal [Pa]
 * @param {object} o
 * @param {object} o.material
 * @param {string} o.finish
 * @param {number} o.R        load ratio (min / max): -1 fully reversed, 0 zero-based
 * @param {number} o.cycles   design life
 * @param {number} [o.scale]  multiplies the stresses (load factor)
 */
export function fatigueField(vm, p1, p3, { material, finish, R, cycles, scale = 1 }) {
  const curve = snCurve(material, finish);
  const n = vm.length;
  const life = new Float32Array(n), damage = new Float32Array(n), fos = new Float32Array(n);
  const Sn = strengthAt(curve, cycles);
  let minLife = Infinity, worst = -1, minFos = Infinity;
  for (let v = 0; v < n; v++) {
    if (Number.isNaN(vm[v])) { life[v] = damage[v] = fos[v] = NaN; continue; }
    const sign = Math.abs(p1[v]) >= Math.abs(p3[v]) ? Math.sign(p1[v]) || 1 : Math.sign(p3[v]) || 1;
    const smax = (sign * vm[v] * scale) / 1e6; // MPa
    const { sa, sm } = cycleStress(smax, R);
    const sar = goodman(sa, sm, curve.uts);
    const N = cyclesToFailure(curve, sar);
    life[v] = N;
    damage[v] = Number.isFinite(N) ? cycles / N : 0;
    const denom = sa / Sn + Math.max(0, sm) / curve.uts;
    fos[v] = denom > 0 ? 1 / denom : Infinity;
    if (N < minLife) { minLife = N; worst = v; }
    if (fos[v] < minFos) minFos = fos[v];
  }
  return { curve, life, damage, fos, minLife, worst, minFos, strengthAtLife: Sn };
}
