// Drop test: explicit dynamics of the part hitting a rigid floor.
//
// Central-difference time integration with a lumped mass matrix, the same linear-elastic voxel
// elements as the static study, a small stiffness-proportional damping that removes numerical
// ringing at the highest frequencies, and frictionless penalty contact with a rigid floor under
// the part (gravity along -Y). The part starts touching the floor, moving at the impact speed
// sqrt(2 g H) of a drop from height H. The time step is set from the highest natural frequency
// (power iteration), so the run is stable without user tuning.
//
// Physical units throughout: u [m], v [m/s], t [s], forces [N], stresses [Pa].
import { lumpedMass } from './eigen.js';

/** Highest eigenvalue of M^-1 K (normalized units) by power iteration. */
export function maxEigenvalue(fea, mass, iterations = 40) {
  const L = fea.levels[0], n = L.nDof;
  let v = new Float64Array(n), w = new Float64Array(n);
  let s = 99;
  for (let i = 0; i < n; i++) { s = (s * 1103515245 + 12345) & 0x7fffffff; v[i] = mass[i] > 0 && !L.fixed[i] ? s / 0x7fffffff - 0.5 : 0; }
  let lam = 0;
  for (let it = 0; it < iterations; it++) {
    let nv = 0;
    for (let i = 0; i < n; i++) nv += v[i] * v[i];
    nv = Math.sqrt(nv) || 1;
    for (let i = 0; i < n; i++) v[i] /= nv;
    fea.apply(L, v, w);
    let nw = 0;
    for (let i = 0; i < n; i++) { w[i] = mass[i] > 0 ? w[i] / mass[i] : 0; nw += w[i] * w[i]; }
    lam = Math.sqrt(nw);
    [v, w] = [w, v];
  }
  return lam;
}

/**
 * Explicit drop simulation on the CPU.
 * @param {import('./solver.js').VoxelFEA} fea   free (unheld) voxel model
 * @param {object} o
 * @param {number} o.E, o.rho, o.h            modulus [Pa], density [kg/m^3], voxel size [m]
 * @param {number} o.speed                    impact speed [m/s]
 * @param {Float64Array} o.nodeY              height of every node above the floor [m] (0 for the lowest)
 * @param {Uint8Array} o.surface              1 for nodes that can touch the floor
 * @param {number} [o.frames]                 animation frames to keep
 * @param {number} [o.maxSteps]
 * @param {(step: object) => void} [o.onFrame]
 * @param {(frac: number) => boolean|void} [o.onProgress] return true to cancel
 */
export function dropTestCPU(fea, { E, rho, h, speed, nodeY, surface, g = 9.81, frames = 48, maxSteps = 20000, maxTime = Infinity, onFrame = null, onProgress = null }) {
  const L = fea.levels[0], n = L.nDof, nN = L.nNodes;
  const Mn = lumpedMass(fea);
  const lamMax = maxEigenvalue(fea, Mn) * 1.05;
  const wMax = Math.sqrt((lamMax * E) / (rho * h * h));
  const xiHigh = 0.1; // damping ratio at the highest frequency
  const beta = (2 * xiHigh) / wMax;
  const dt = 0.9 * (2 / wMax) * (Math.sqrt(1 + xiHigh * xiHigh) - xiHigh);
  const m = new Float64Array(n);
  for (let i = 0; i < n; i++) m[i] = rho * h * h * h * Mn[i];
  // penalty contact: each node's contact frequency is half the highest element frequency
  const kc = new Float64Array(nN), cc = new Float64Array(nN);
  for (let q = 0; q < nN; q++) {
    if (!surface[q] || !(m[3 * q + 1] > 0)) continue;
    kc[q] = 0.25 * wMax * wMax * m[3 * q + 1];
    cc[q] = 2 * 0.05 * Math.sqrt(kc[q] * m[3 * q + 1]);
  }
  const u = new Float64Array(n), v = new Float64Array(n), w = new Float64Array(n), q = new Float64Array(n), a = new Float64Array(n);
  for (let i = 1; i < n; i += 3) if (m[i] > 0) v[i] = -speed;
  const Eh = E * h;
  let contactStarted = false, contactEnd = -1, contactStart = 0, peakForce = 0, t = 0, step = 0;
  const history = [];
  const vmMax = new Float32Array(nN);
  const tPeak = new Float32Array(nN);
  // enough samples of the stress envelope to catch the peak (stress changes slowly per step)
  const sampleEvery = 4;
  let lastFrame = -Infinity;
  const forces = () => {
    // internal + damping forces, contact, gravity -> acceleration
    for (let i = 0; i < n; i++) w[i] = u[i] + beta * v[i];
    fea.apply(L, w, q, true);
    let fc = 0;
    for (let i = 0; i < n; i++) a[i] = m[i] > 0 ? -Eh * q[i] / m[i] : 0;
    for (let k = 0; k < nN; k++) {
      const i = 3 * k + 1;
      if (!(m[i] > 0)) continue;
      a[i] -= g;
      if (kc[k] > 0) {
        const pen = -(nodeY[k] + u[i]);
        if (pen > 0) {
          const f = Math.max(0, kc[k] * pen - cc[k] * v[i]);
          a[i] += f / m[i];
          fc += f;
        }
      }
    }
    return fc;
  };
  let fc = forces();
  for (let i = 0; i < n; i++) v[i] += 0.5 * dt * a[i];
  const sampleStress = () => {
    const st = fea.stresses(u, E, h);
    for (let k = 0; k < nN; k++) if (st.nodeVM[k] > vmMax[k]) { vmMax[k] = st.nodeVM[k]; tPeak[k] = t; }
    return st;
  };
  let estEnd = maxTime;
  for (step = 1; step <= maxSteps; step++) {
    for (let i = 0; i < n; i++) u[i] += dt * v[i];
    t += dt;
    fc = forces();
    for (let i = 0; i < n; i++) v[i] += dt * a[i];
    if (fc > 0 && !contactStarted) { contactStarted = true; contactStart = t; }
    if (fc > peakForce) peakForce = fc;
    if (contactStarted && fc > 0) contactEnd = -1;
    else if (contactStarted && fc === 0 && contactEnd < 0) contactEnd = t;
    // stop a while after the part has left the floor (or bounced off it)
    if (contactEnd > 0 && estEnd === maxTime) estEnd = Math.min(maxTime, contactEnd + Math.max(0.3 * (contactEnd - contactStart), 50 * dt));
    if (contactEnd < 0 && estEnd !== maxTime) estEnd = maxTime; // touched down again
    const done = t >= estEnd || step === maxSteps;
    if (step % sampleEvery === 0 || done) {
      const st = sampleStress();
      history.push({ t, force: fc });
      const frameGap = Math.max(dt, (Number.isFinite(estEnd) ? estEnd : Math.max(t, 1e-6) * 2) / frames);
      if (onFrame && (t - lastFrame >= frameGap || done)) {
        lastFrame = t;
        onFrame({ t, u: Float32Array.from(u), vm: st.nodeVM, force: fc });
      }
      if (onProgress && onProgress(Math.min(0.99, Number.isFinite(estEnd) ? t / estEnd : step / maxSteps)) === true) {
        throw Object.assign(new Error('Cancelled'), { cancelled: true });
      }
    }
    if (done) break;
  }
  return {
    vmMax, tPeak, history, dt, steps: step, duration: t, peakForce,
    contactTime: contactStarted ? (contactEnd > 0 ? contactEnd : t) - contactStart : 0,
    wMax, rebounded: contactEnd > 0,
  };
}
