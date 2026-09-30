// CPU solver of the v1 flow engine (flow.js), and the reference for its GPU shader (flow.wgsl).
//
// One step is three passes over disjoint cells: the bulk fluid (branch-free), the wall cells (from
// their records) and the tunnel's faces. Populations stream in place ("esoteric pull", Lehmann 2022):
// one array, where each step reads and writes the same slots of every cell, so passes and threads
// never touch each other's data. Layout F[d * N + cell].
//
// Collision: recursive regularised BGK ("rr": Malaspinas 2015; Coreixas et al. 2017; the D3Q19
// equilibrium and regularised non-equilibrium to third order) or plain BGK ("bgk", as in v0.6),
// both with a Smagorinsky subgrid model. Wall cells may add a wall model: the eddy viscosity that
// makes the wall shear match Spalding's law of the wall at the cell's distance from the wall.
//
// Walls: bounce-back, with the wall at its true position on each link (Bouzidi et al. 2001). The
// ground plane is a moving belt at the wind speed. Forces on the part come from momentum exchange
// on its wall links (pressure and friction), summed per wall cell; the pressure integral over the
// wetted voxel faces is kept as a check.
import { BULK, WALL, SOLID, FACE, CX, CY, CZ, W, OPP, P3, flowParams, inletVelocity, spaldingUtau, reichardt, reichardtUtau } from './flow.js';
import * as K from './flow-kernel.js';

const CS2 = 1 / 3;

// scratch for the third-order moments (each worker has its own copy of this module)
const A0 = new Float64Array(6), A1 = new Float64Array(6);

/** Third-order equilibrium moments over rho, as the P3 combinations: P+ sums, then P- differences. */
function moments3(a, ux, uy, uz) {
  const xxy = ux * ux * uy, yzz = uy * uz * uz, xzz = ux * uz * uz, xyy = ux * uy * uy, yyz = uy * uy * uz, xxz = ux * ux * uz;
  a[0] = xxy + yzz; a[1] = xzz + xyy; a[2] = yyz + xxz; a[3] = xxy - yzz; a[4] = xzz - xyy; a[5] = yyz - xxz;
  return a;
}

/**
 * Post-collision populations g from incoming f (both length 19). Returns rho and u in out[0..3].
 * tauMin: a lower bound on the relaxation time (the wall model); 0 for none.
 */
export function collide(f, g, out, tau0, smag, rr, tauMin) {
  let rho = 0, jx = 0, jy = 0, jz = 0, pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
  for (let i = 0; i < 19; i++) {
    const v = f[i], cx = CX[i], cy = CY[i], cz = CZ[i];
    rho += v;
    jx += cx * v; jy += cy * v; jz += cz * v;
    pxx += cx * cx * v; pyy += cy * cy * v; pzz += cz * cz * v;
    pxy += cx * cy * v; pxz += cx * cz * v; pyz += cy * cz * v;
  }
  const ux = jx / rho, uy = jy / rho, uz = jz / rho;
  // non-equilibrium stress (the same for any equilibrium order on D3Q19)
  const nxx = pxx - rho * (ux * ux + CS2), nyy = pyy - rho * (uy * uy + CS2), nzz = pzz - rho * (uz * uz + CS2);
  const nxy = pxy - rho * ux * uy, nxz = pxz - rho * ux * uz, nyz = pyz - rho * uy * uz;
  const q = Math.sqrt(nxx * nxx + nyy * nyy + nzz * nzz + 2 * (nxy * nxy + nxz * nxz + nyz * nyz));
  let tau = 0.5 * (tau0 + Math.sqrt(tau0 * tau0 + (smag * q) / rho));
  if (tau < tauMin) tau = tauMin;
  const om = 1 / tau;
  const usq = 1.5 * (ux * ux + uy * uy + uz * uz);
  out[0] = rho; out[1] = ux; out[2] = uy; out[3] = uz;
  if (!rr) {
    for (let i = 0; i < 19; i++) {
      const cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
      const e = W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - usq);
      g[i] = f[i] - om * (f[i] - e);
    }
    return;
  }
  // third-order equilibrium moments (over rho) and the recursive non-equilibrium ones
  const a0 = moments3(A0, ux, uy, uz);
  const xxy = 2 * ux * nxy + uy * nxx, yzz = 2 * uz * nyz + uy * nzz, xzz = 2 * uz * nxz + ux * nzz;
  const xyy = 2 * uy * nxy + ux * nyy, yyz = 2 * uy * nyz + uz * nyy, xxz = 2 * ux * nxz + uz * nxx;
  const a1 = A1;
  a1[0] = xxy + yzz; a1[1] = xzz + xyy; a1[2] = yyz + xxz; a1[3] = xxy - yzz; a1[4] = xzz - xyy; a1[5] = yyz - xxz;
  const tr = CS2 * (nxx + nyy + nzz), k = 1 - om;
  for (let i = 0; i < 19; i++) {
    const cx = CX[i], cy = CY[i], cz = CZ[i], p = P3[i];
    const cu = cx * ux + cy * uy + cz * uz;
    const t3e = 13.5 * (p[0] * a0[0] + p[1] * a0[1] + p[2] * a0[2]) + 4.5 * (p[3] * a0[3] + p[4] * a0[4] + p[5] * a0[5]);
    const e = W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - usq + t3e);
    const h2 = cx * cx * nxx + cy * cy * nyy + cz * cz * nzz + 2 * (cx * cy * nxy + cx * cz * nxz + cy * cz * nyz) - tr;
    const t3n = 13.5 * (p[0] * a1[0] + p[1] * a1[1] + p[2] * a1[2]) + 4.5 * (p[3] * a1[3] + p[4] * a1[4] + p[5] * a1[5]);
    g[i] = e + k * W[i] * (4.5 * h2 + t3n);
  }
}

/** Equilibrium populations (the collision model's order) at rho, u. */
export function equilibrium(g, rho, ux, uy, uz, rr) {
  const usq = 1.5 * (ux * ux + uy * uy + uz * uz);
  const a0 = rr ? moments3(A0, ux, uy, uz) : null;
  for (let i = 0; i < 19; i++) {
    const cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
    let t3 = 0;
    if (rr) {
      const p = P3[i];
      t3 = 13.5 * (p[0] * a0[0] + p[1] * a0[1] + p[2] * a0[2]) + 4.5 * (p[3] * a0[3] + p[4] * a0[4] + p[5] * a0[5]);
    }
    g[i] = W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - usq + t3);
  }
}

// ---------- streaming: where each population of a cell lives ----------
//
// In-place ("esoteric pull"): with nb = cell + c_i for odd i, a step of parity `odd` loads
//   even: f[0] = F0[n], f[i] = F[i+1][n], f[i+1] = F[i][nb]
//   odd:  f[0] = F0[n], f[i] = F[i][n],   f[i+1] = F[i+1][nb]
// and stores its post-collision values in the same slots with the roles swapped
//   even: F[i][nb] = g[i], F[i+1][n] = g[i+1];   odd: F[i+1][nb] = g[i], F[i][n] = g[i+1].
// A population that would come from a solid cell is its own outgoing opposite from the last step
// (bounce-back): the same load with the other parity.
//
// Two-buffer ("ab", tests only): f[i] = A[i][n - c_i], stored g[i] -> B[i][n]; bounce-back A[opp i][n].

/** Linear index of cell n + c_i for each direction (z wraps when periodic). */
function neighbours(s, x, y, z, nb) {
  const { nx, ny, nz, periodicZ } = s;
  for (let i = 0; i < 19; i++) {
    let zz = z + CZ[i];
    if (periodicZ) zz = zz < 0 ? nz - 1 : zz >= nz ? 0 : zz;
    nb[i] = x + CX[i] + nx * (y + CY[i] + ny * zz);
  }
}

function load(s, n, nb, f, odd) {
  const { F, N } = s;
  if (s.ab) {
    const A = s.A;
    f[0] = A[n];
    // pull: population i arrives from n - c_i, the neighbour of the opposite direction
    for (let i = 1; i < 19; i++) f[i] = A[i * N + nb[OPP[i]]];
    return;
  }
  f[0] = F[n];
  for (let i = 1; i < 19; i += 2) {
    const m = nb[i];
    if (odd) { f[i] = F[i * N + n]; f[i + 1] = F[(i + 1) * N + m]; }
    else { f[i] = F[(i + 1) * N + n]; f[i + 1] = F[i * N + m]; }
  }
}

/** Bounce-back: the cell's own outgoing opp(k) from the last step. */
function bounced(s, n, nb, k, odd) {
  const { F, N } = s;
  if (s.ab) return s.A[OPP[k] * N + n];
  if (k & 1) return odd ? F[(k + 1) * N + n] : F[k * N + n];
  const m = nb[k - 1];
  return odd ? F[(k - 1) * N + m] : F[k * N + m];
}

function store(s, n, nb, g, odd) {
  const { F, N } = s;
  if (s.ab) {
    const B = s.B;
    for (let i = 0; i < 19; i++) B[i * N + n] = g[i];
    return;
  }
  F[n] = g[0];
  for (let i = 1; i < 19; i += 2) {
    const m = nb[i];
    if (odd) { F[(i + 1) * N + m] = g[i]; F[i * N + n] = g[i + 1]; }
    else { F[i * N + m] = g[i]; F[(i + 1) * N + n] = g[i + 1]; }
  }
}

/** Where the post-collision g[i] of cell n (this step) now lives, or -1 outside the tunnel. */
function storedIndex(s, n, nb, i, odd) {
  const { N } = s;
  if (s.ab) return i * N + n;
  if (i === 0) return n;
  if (i & 1) return nb[i] < 0 ? -1 : odd ? (i + 1) * N + nb[i] : i * N + nb[i];
  return odd ? (i - 1) * N + n : i * N + n;
}

// ---------- the three passes (run on the calling thread or shared with helpers) ----------

/** Scratch of one thread. */
export function scratch() {
  return { f: new Float64Array(19), g: new Float64Array(19), out: new Float64Array(4), nb: new Int32Array(19) };
}

const RUNS = { rr: [K.bulkRunRREven, K.bulkRunRROdd], bgk: [K.bulkRunBGKEven, K.bulkRunBGKOdd] };

/** Per row (y + ny z): runs [x0, x1) of bulk cells, for the unrolled kernels. */
export function bulkRuns(kind, dims) {
  const [nx, ny, nz] = dims;
  const runs = [], start = new Int32Array(ny * nz + 1);
  for (let row = 0; row < ny * nz; row++) {
    let x0 = -1;
    for (let x = 1; x < nx; x++) {
      const bulk = x < nx - 1 && kind[x + nx * row] === BULK;
      if (bulk && x0 < 0) x0 = x;
      if (!bulk && x0 >= 0) { runs.push(x0, x); x0 = -1; }
    }
    start[row + 1] = runs.length / 2;
  }
  return { runs: Int32Array.from(runs), start };
}

/** Offsets of the neighbours n + c_i in a layer z (the span wraps around when periodic). */
function layerOffsets(s, z) {
  const { nx, ny, nz, periodicZ } = s;
  const o = new Int32Array(19);
  for (let i = 0; i < 19; i++) {
    let dz = CZ[i];
    if (periodicZ && z + dz < 0) dz += nz;
    if (periodicZ && z + dz >= nz) dz -= nz;
    o[i] = CX[i] + nx * (CY[i] + ny * dz);
  }
  return o;
}

/** Bulk cells of rows r0 <= row < r1 (row = y + ny z). */
export function bulkRows(s, t, r0, r1) {
  const { nx, ny, nz, kind, tau0, smag, rr, odd } = s;
  if (!s.ab) {
    // the unrolled kernels, run by run
    const run = (rr ? RUNS.rr : RUNS.bgk)[odd];
    const { runs, start } = s.bulk;
    if (!s.offsets) s.offsets = [layerOffsets(s, 0), layerOffsets(s, Math.min(1, nz - 1)), layerOffsets(s, nz - 1)];
    for (let row = r0; row < r1; row++) {
      const z = (row / ny) | 0;
      const o = s.offsets[z === 0 ? 0 : z === nz - 1 ? 2 : 1];
      const base = nx * row;
      for (let r = start[row]; r < start[row + 1]; r++) run(s.F, s.N, base + runs[2 * r], base + runs[2 * r + 1], o, tau0, smag);
    }
    return;
  }
  const { f, g, out, nb } = t;
  for (let row = r0; row < r1; row++) {
    const y = row % ny, z = (row / ny) | 0;
    const base = nx * row;
    for (let x = 1; x < nx - 1; x++) {
      const n = base + x;
      if (kind[n] !== BULK) continue;
      neighbours(s, x, y, z, nb);
      load(s, n, nb, f, odd);
      collide(f, g, out, tau0, smag, rr, 0);
      store(s, n, nb, g, odd);
    }
  }
}

/** The Smagorinsky relaxation time of populations f at rho, u (collide's, without the collision). */
export function smagorinskyTau(f, rho, ux, uy, uz, tau0, smag) {
  let pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
  for (let i = 0; i < 19; i++) {
    const v = f[i], cx = CX[i], cy = CY[i], cz = CZ[i];
    pxx += cx * cx * v; pyy += cy * cy * v; pzz += cz * cz * v;
    pxy += cx * cy * v; pxz += cx * cz * v; pyz += cy * cz * v;
  }
  const nxx = pxx - rho * (ux * ux + CS2), nyy = pyy - rho * (uy * uy + CS2), nzz = pzz - rho * (uz * uz + CS2);
  const nxy = pxy - rho * ux * uy, nxz = pxz - rho * ux * uz, nyz = pyz - rho * uy * uz;
  const q = Math.sqrt(nxx * nxx + nyy * nyy + nzz * nzz + 2 * (nxy * nxy + nxz * nxz + nyz * nyz));
  return 0.5 * (tau0 + Math.sqrt(tau0 * tau0 + (smag * q) / rho));
}

const us_ = (x, y, z) => Math.hypot(x, y, z);

/**
 * Post-collision populations of the regularised state at rho, u with non-equilibrium stress Pi
 * (xx, yy, zz, xy, xz, yz): feq + k W (H2 : Pi / (2 cs^4) + the recursive third order terms).
 */
export function regularized(g, rho, ux, uy, uz, pi, k, rr) {
  equilibrium(g, rho, ux, uy, uz, rr);
  const [nxx, nyy, nzz, nxy, nxz, nyz] = pi;
  const tr = CS2 * (nxx + nyy + nzz);
  let a1 = null;
  if (rr) {
    const xxy = 2 * ux * nxy + uy * nxx, yzz = 2 * uz * nyz + uy * nzz, xzz = 2 * uz * nxz + ux * nzz;
    const xyy = 2 * uy * nxy + ux * nyy, yyz = 2 * uy * nyz + uz * nyy, xxz = 2 * ux * nxz + uz * nxx;
    a1 = A1;
    a1[0] = xxy + yzz; a1[1] = xzz + xyy; a1[2] = yyz + xxz; a1[3] = xxy - yzz; a1[4] = xzz - xyy; a1[5] = yyz - xxz;
  }
  for (let i = 0; i < 19; i++) {
    const cx = CX[i], cy = CY[i], cz = CZ[i];
    const h2 = cx * cx * nxx + cy * cy * nyy + cz * cz * nzz + 2 * (cx * cy * nxy + cx * cz * nxz + cy * cz * nyz) - tr;
    let t3 = 0;
    if (rr) {
      const p = P3[i];
      t3 = 13.5 * (p[0] * a1[0] + p[1] * a1[1] + p[2] * a1[2]) + 4.5 * (p[3] * a1[3] + p[4] * a1[4] + p[5] * a1[5]);
    }
    g[i] += k * W[i] * (4.5 * h2 + t3);
  }
}

/**
 * Wall model (Malaspinas & Sagaut 2014, with Reichardt's law of the wall): the flow is sampled at
 * the bulk cell m = n + c_j nearest the wall normal, at distance y2 from the wall; its tangential
 * speed there gives the friction velocity u_tau. The wall cell's populations are then rebuilt, as
 * the regularised state with the law's velocity at its own distance y1, the density at m, and the
 * shear stress rho u_tau^2 carried by the eddy viscosity nu / (du+/dy+) the law implies. Its force
 * on the part is the pressure on the surface next to it plus that shear. Returns false when the
 * sample is unusable (the cell keeps plain bounce-back).
 */
function wallModelCell(s, t, r, n, x, y, z) {
  const { rec, nx, ny, nz, nu0, odd, rr, uBelt, acc } = s;
  const j = rec.samp[r];
  const xm = x + CX[j], ym = y + CY[j], zm = s.periodicZ ? (z + CZ[j] + nz) % nz : z + CZ[j];
  const m = xm + nx * (ym + ny * zm);
  // the sample's velocity and density, from where its post-collision populations now live
  const nbm = t.nbm || (t.nbm = new Int32Array(19));
  neighbours(s, xm, ym, zm, nbm);
  const src = s.ab ? s.B : s.F;
  let rho2 = 0, jx = 0, jy = 0, jz = 0;
  for (let i = 0; i < 19; i++) {
    const v = src[storedIndex(s, m, nbm, i, odd)];
    rho2 += v; jx += CX[i] * v; jy += CY[i] * v; jz += CZ[i] * v;
  }
  if (!(rho2 > 0)) return false;
  const wx = rec.onBelt[r] ? uBelt : 0;
  const nxv = rec.normal[3 * r], nyv = rec.normal[3 * r + 1], nzv = rec.normal[3 * r + 2];
  const ux = jx / rho2 - wx, uy = jy / rho2, uz = jz / rho2;
  // the sample's velocity along the wall (not filtered in time: a delayed wall law lets the two walls
  // of a narrow gap drive each other into growing oscillations)
  const un2 = ux * nxv + uy * nyv + uz * nzv;
  const tx = ux - un2 * nxv, ty = uy - un2 * nyv, tz = uz - un2 * nzv;
  const ut2 = Math.hypot(tx, ty, tz);
  const y1 = rec.dist[r], y2 = rec.y2[r];
  let utau = 0, ut1 = 0, dudn = 0, tauN = 0.5 + 3 * nu0;
  if (ut2 > 1e-12) {
    utau = reichardtUtau(ut2, y2, nu0);
    const [up1, dup1] = reichardt((y1 * utau) / nu0);
    ut1 = utau * up1;
    dudn = ((utau * utau) / nu0) * dup1;
    tauN = 0.5 + (3 * nu0) / dup1;
  }
  const ex = ut2 > 1e-12 ? tx / ut2 : 0, ey = ut2 > 1e-12 ? ty / ut2 : 0, ez = ut2 > 1e-12 ? tz / ut2 : 0;
  // velocity at the wall cell: the law's, along the wall only (a velocity toward the wall pumps
  // pressure waves in narrow gaps)
  const u1x = wx + ut1 * ex, u1y = ut1 * ey, u1z = ut1 * ez;
  // non-equilibrium stress of the shear du_t/dn: Pi = -rho tau / 3 * du/dn (t n + n t)
  const c = (-rho2 * tauN * dudn) / 3;
  const pi = t.pi || (t.pi = new Float64Array(6));
  pi[0] = 2 * c * ex * nxv; pi[1] = 2 * c * ey * nyv; pi[2] = 2 * c * ez * nzv;
  pi[3] = c * (ex * nyv + ey * nxv); pi[4] = c * (ex * nzv + ez * nxv); pi[5] = c * (ey * nzv + ez * nyv);
  const { g, nb } = t;
  regularized(g, rho2, u1x, u1y, u1z, pi, 1 - 1 / tauN, rr);
  neighbours(s, x, y, z, nb);
  store(s, n, nb, g, odd);
  const mask = rec.mask[r];
  for (let k = 1; k < 19; k++) {
    if (!(mask & (1 << k))) continue;
    s.aux[19 * r + k] = g[k];
    s.bb[19 * r + k] = g[OPP[k]];
  }
  // force on the part: pressure on the surface next to the cell, and the wall shear along the flow
  const ax = rec.area[3 * r], ay = rec.area[3 * r + 1], az = rec.area[3 * r + 2];
  const area = Math.hypot(ax, ay, az), p = (rho2 - 1) * CS2, tw = rho2 * utau * utau * area;
  acc[4 * r] += -p * ax + tw * ex;
  acc[4 * r + 1] += -p * ay + tw * ey;
  acc[4 * r + 2] += -p * az + tw * ez;
  acc[4 * r + 3] += rho2 - 1;
  s.tauWall[r] = utau;
  return true;
}
// weight of the newest value in the wall model's time filter (a time constant of about 1 / a steps)
export const SLIP_FILTER = 0.05;

/** Wall records r0 <= r < r1: bounce-back with the wall's position, wall model, forces. */
export function wallCells(s, t, r0, r1) {
  const { nx, ny, tau0, smag, rr, odd, rec, aux, acc, nu0, wallModel, uBelt } = s;
  const { f, g, out, nb } = t;
  const fb = t.fb || (t.fb = new Float64Array(19));
  for (let r = r0; r < r1; r++) {
    const n = rec.cell[r], mask = rec.mask[r], gm = rec.groundMask[r];
    const x = n % nx, y = ((n / nx) | 0) % ny, z = (n / (nx * ny)) | 0;
    neighbours(s, x, y, z, nb);
    // the wall model on the part when it is on, and always on the moving ground: plain bounce-back
    // from a belt moving with the air leaves grid-scale waves undamped at low viscosity
    if (rec.samp[r] && ((wallModel && s.wallMode === 'model') || rec.onBelt[r]) && wallModelCell(s, t, r, n, x, y, z)) continue;
    load(s, n, nb, f, odd);
    // incoming populations from solid cells, and the momentum they exchange with the part
    let mx = 0, my = 0, mz = 0;
    // bounce-back: the cell's own outgoing populations toward its walls from the last step, kept in
    // its record (a thin wall's other side is fluid, so the slots across it are not free to read)
    for (let k = 1; k < 19; k++) if (mask & (1 << k)) fb[k] = s.bb[19 * r + k];
    const fq = t.fq || (t.fq = new Float64Array(19));
    for (let k = 1; k < 19; k++) {
      if (!(mask & (1 << k))) continue;
      const v = fb[k];
      let fk;
      if (gm & (1 << k)) {
        // moving ground belt, half-way: f_k = f*_opp + 6 w_k rho_w (c_k . u_w), with rho_w = 1
        fk = v + 6 * W[k] * CX[k] * uBelt;
        fq[k] = 6;
      } else {
        const qb = rec.q[18 * r + k - 1];
        fk = v;
        fq[k] = 6;
        if (qb) {
          const qq = (qb - 1) / 254;
          if (qq < 0.5) fk = 2 * qq * v + (1 - 2 * qq) * f[OPP[k]];
          else { fk = (0.5 / qq) * v + (1 - 0.5 / qq) * aux[19 * r + k]; fq[k] = 3 / qq; }
        }
      }
      f[k] = fk;
    }
    // slip wall model: the wall moves along the near-wall flow at the speed that makes the wall
    // shear stress the log law's rho u_tau^2 at this cell's distance from the wall, given the cell's
    // own (molecular + subgrid) viscosity. Bounce-back off a moving wall adds 6 w_k rho (c_k . u_w)
    // (3 / q times at links with q >= 1/2).
    let wx = 0, wy = 0, wz = 0;
    if (wallModel && s.wallMode === 'slip') {
      let rho = 0, jx = 0, jy = 0, jz = 0;
      for (let i = 0; i < 19; i++) { rho += f[i]; jx += CX[i] * f[i]; jy += CY[i] * f[i]; jz += CZ[i] * f[i]; }
      const nrm = rec.normal, d = rec.dist[r];
      const nx_ = nrm[3 * r], ny_ = nrm[3 * r + 1], nz_ = nrm[3 * r + 2];
      let ux = jx / rho, uy = jy / rho, uz = jz / rho;
      if (rec.onBelt[r]) ux -= uBelt;
      const un = ux * nx_ + uy * ny_ + uz * nz_;
      const tx = ux - un * nx_, ty = uy - un * ny_, tz = uz - un * nz_;
      const ut = Math.hypot(tx, ty, tz);
      const utau = spaldingUtau(ut, d, nu0);
      if (ut > 1e-9 && utau > 0) {
        const nuEff = (smagorinskyTau(f, rho, ux + (rec.onBelt[r] ? uBelt : 0), uy, uz, tau0, smag) - 0.5) / 3;
        const us = Math.min(ut, Math.max(0, ut - (utau * utau * d) / nuEff));
        // low-pass in time: a wall that followed each step's velocity would feed the odd-even
        // oscillation of a nearly inviscid cell
        const sl = s.slip, a = SLIP_FILTER;
        sl[3 * r] += a * ((us * tx) / ut - sl[3 * r]);
        sl[3 * r + 1] += a * ((us * ty) / ut - sl[3 * r + 1]);
        sl[3 * r + 2] += a * ((us * tz) / ut - sl[3 * r + 2]);
        // along the current flow only, and never faster than it
        const along = Math.min(ut, Math.max(0, (sl[3 * r] * tx + sl[3 * r + 1] * ty + sl[3 * r + 2] * tz) / ut));
        wx = (along * tx) / ut; wy = (along * ty) / ut; wz = (along * tz) / ut;
        for (let k = 1; k < 19; k++) {
          if (mask & (1 << k)) f[k] += fq[k] * W[k] * rho * (CX[k] * wx + CY[k] * wy + CZ[k] * wz);
        }
      }
      s.tauWall[r] = us_(wx, wy, wz);
    }
    // momentum to the part (Galilean invariant for a moving wall, Wen et al. 2014): what left toward
    // it minus what came back, relative to the wall, with c_opp = -c_k; less the air at rest's (the
    // reference pressure, as the wall model's surface integral has it: a part with both kinds of wall
    // cell must not feel the ambient pressure on one side only)
    for (let k = 1; k < 19; k++) {
      if (!(mask & (1 << k)) || gm & (1 << k)) continue;
      const v = fb[k], fk = f[k], e = v + fk - 2 * W[k];
      mx -= CX[k] * e + wx * (v - fk);
      my -= CY[k] * e + wy * (v - fk);
      mz -= CZ[k] * e + wz * (v - fk);
    }
    // older variants, kept to compare: the log-law eddy viscosity sets the wall cell's relaxation
    // ('replace', unstable at high Reynolds numbers), or bounds it from below ('max'); with 'model' the
    // cells it leaves out (narrow gaps) keep plain bounce-back
    let tauMin = 0, tauWall = 0;
    if (wallModel && (s.wallMode === 'replace' || s.wallMode === 'max')) {
      let rho = 0, jx = 0, jy = 0, jz = 0;
      for (let i = 0; i < 19; i++) { rho += f[i]; jx += CX[i] * f[i]; jy += CY[i] * f[i]; jz += CZ[i] * f[i]; }
      const nrm = rec.normal, d = rec.dist[r];
      const nx_ = nrm[3 * r], ny_ = nrm[3 * r + 1], nz_ = nrm[3 * r + 2];
      let ux = jx / rho, uy = jy / rho, uz = jz / rho;
      if (rec.onBelt[r]) ux -= uBelt;
      const un = ux * nx_ + uy * ny_ + uz * nz_;
      const ut = Math.hypot(ux - un * nx_, uy - un * ny_, uz - un * nz_);
      const utau = spaldingUtau(ut, d, nu0);
      if (ut > 0 && utau > 0) {
        const nuT = Math.max(0, (utau * utau * d) / ut - nu0);
        tauWall = 3 * (nu0 + nuT) + 0.5;
        if (s.wallMode === 'max') { tauMin = tauWall; tauWall = 0; }
      }
      s.tauWall[r] = tauWall || tauMin;
    }
    if (tauWall > 0) collide(f, g, out, tauWall, 0, rr, 0);
    else collide(f, g, out, tau0, smag, rr, tauMin);
    store(s, n, nb, g, odd);
    // for the next step: the outgoing populations toward the walls (bounce-back) and away from them
    // (interpolation at q >= 1/2)
    for (let k = 1; k < 19; k++) {
      if (!(mask & (1 << k))) continue;
      aux[19 * r + k] = g[k];
      s.bb[19 * r + k] = g[OPP[k]];
    }
    acc[4 * r] += mx; acc[4 * r + 1] += my; acc[4 * r + 2] += mz; acc[4 * r + 3] += out[0] - 1;
  }
}

/**
 * Per face cell: its nearest interior cell n0 (-1 at the inlet, -2 when n0 is solid), which
 * directions leave the tunnel, and the neighbour-offset layers of the cell and of n0.
 */
export function faceTable(kind, dims, faces, periodicZ) {
  const [nx, ny, nz] = dims;
  const n0 = new Int32Array(faces.length), out = new Uint32Array(faces.length);
  const layer = new Uint8Array(faces.length), layer0 = new Uint8Array(faces.length);
  const layerOf = (z) => (z === 0 ? 0 : z === nz - 1 ? 2 : 1);
  faces.forEach((n, j) => {
    const x = n % nx, y = ((n / nx) | 0) % ny, z = (n / (nx * ny)) | 0;
    let m = 0;
    for (let i = 1; i < 19; i++) {
      const xx = x + CX[i], yy = y + CY[i], zz = z + CZ[i];
      if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || (!periodicZ && (zz < 0 || zz >= nz))) m |= 1 << i;
    }
    out[j] = m;
    layer[j] = layerOf(z);
    if (x === 0) { n0[j] = -1; return; }
    const z0 = periodicZ ? z : Math.min(Math.max(z, 1), nz - 2);
    const c0 = Math.min(x, nx - 2) + nx * (Math.min(Math.max(y, 1), ny - 2) + ny * z0);
    n0[j] = kind[c0] === BULK || kind[c0] === WALL ? c0 : -2;
    layer0[j] = layerOf(z0);
  });
  return { n0, out, layer, layer0 };
}

/** Face cells i0 <= i < i1: the inlet's free stream, or the open faces' ambient pressure. */
export function faceCells(s, t, i0, i1) {
  if (!s.ab) return faceCellsInPlace(s, t, i0, i1);
  const { nx, ny, nz, faces, odd, rr, uin, kind, N } = s;
  const { f, g, nb } = t;
  const nb0 = t.nb0 || (t.nb0 = new Int32Array(19));
  for (let j = i0; j < i1; j++) {
    const n = faces[j];
    const x = n % nx, y = ((n / nx) | 0) % ny, z = (n / (nx * ny)) | 0;
    let ux = uin, uy = 0, uz = 0;
    if (x > 0) {
      // velocity of the nearest interior cell, from its populations after this step's collision
      const x0 = Math.min(x, nx - 2), y0 = Math.min(Math.max(y, 1), ny - 2), z0 = s.periodicZ ? z : Math.min(Math.max(z, 1), nz - 2);
      const n0 = x0 + nx * (y0 + ny * z0);
      if (kind[n0] === BULK || kind[n0] === WALL) {
        neighbours(s, x0, y0, z0, nb0);
        let rho = 0, jx = 0, jy = 0, jz = 0;
        const src = s.ab ? s.B : s.F;
        for (let i = 0; i < 19; i++) {
          const v = src[storedIndex(s, n0, nb0, i, odd)];
          rho += v; jx += CX[i] * v; jy += CY[i] * v; jz += CZ[i] * v;
        }
        ux = jx / rho; uy = jy / rho; uz = jz / rho;
      }
    }
    equilibrium(g, 1, ux, uy, uz, rr);
    // store, skipping slots outside the tunnel
    neighbours(s, x, y, z, nb);
    for (let i = 1; i < 19; i++) {
      const xx = x + CX[i], yy = y + CY[i], zz = z + CZ[i];
      if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || (!s.periodicZ && (zz < 0 || zz >= nz))) nb[i] = -1;
    }
    for (let i = 0; i < 19; i++) s.B[i * N + n] = g[i];
  }
}

function faceCellsInPlace(s, t, i0, i1) {
  const { faces, odd, rr, uin, N, F, face } = s;
  const { g } = t;
  if (!s.offsets) s.offsets = [layerOffsets(s, 0), layerOffsets(s, Math.min(1, s.nz - 1)), layerOffsets(s, s.nz - 1)];
  // the inlet's populations are the same everywhere
  if (!t.gin) t.gin = new Float64Array(19);
  equilibrium(t.gin, 1, uin, 0, 0, rr);
  for (let j = i0; j < i1; j++) {
    const n = faces[j], m0 = face.n0[j];
    let src = t.gin;
    if (m0 !== -1) {
      let ux = uin, uy = 0, uz = 0;
      if (m0 >= 0) {
        // velocity of the nearest interior cell, from where its post-collision populations now live
        const o = s.offsets[face.layer0[j]];
        let rho = F[m0], jx = 0, jy = 0, jz = 0;
        for (let i = 1; i < 19; i += 2) {
          const a = odd ? F[(i + 1) * N + m0 + o[i]] : F[i * N + m0 + o[i]];
          const b = odd ? F[i * N + m0] : F[(i + 1) * N + m0];
          rho += a + b;
          jx += CX[i] * (a - b); jy += CY[i] * (a - b); jz += CZ[i] * (a - b);
        }
        ux = jx / rho; uy = jy / rho; uz = jz / rho;
      }
      equilibrium(g, 1, ux, uy, uz, rr);
      src = g;
    }
    // store, skipping slots outside the tunnel
    const o = s.offsets[face.layer[j]], outMask = face.out[j];
    F[n] = src[0];
    for (let i = 1; i < 19; i += 2) {
      if (!(outMask & (1 << i))) F[(odd ? i + 1 : i) * N + n + o[i]] = src[i];
      F[(odd ? i : i + 1) * N + n] = src[i + 1];
    }
  }
}

// ---------- the solver ----------

export class FlowCPU {
  /**
   * @param {object} grid  flow.js buildFlowGrid()
   * @param {{uLat: number, nuLat: number, smagorinsky?: number, collision?: 'rr'|'bgk', wallModel?: boolean, layout?: 'esoteric'|'ab', belt?: boolean}} o
   */
  constructor(grid, o) {
    const [nx, ny, nz] = grid.dims;
    const N = nx * ny * nz;
    const p = flowParams(o);
    const rec = grid.rec;
    // wall cells whose nearest wall is the (moving) ground
    if (!rec.onBelt) rec.onBelt = Uint8Array.from(rec.dist, (d, r) => (rec.groundMask[r] && d === 0.5 ? 1 : 0));
    const faces = [];
    for (let c = 0; c < N; c++) if (grid.kind[c] === FACE) faces.push(c);
    this.s = {
      nx, ny, nz, N, kind: grid.kind, periodicZ: grid.periodicZ, rec,
      faces: Uint32Array.from(faces),
      ab: o.layout === 'ab',
      rr: (o.collision || 'rr') === 'rr',
      wallModel: o.wallModel !== false && (o.collision || 'rr') === 'rr',
      wallMode: o.wallMode || 'model',
      tau0: p.tau0, smag: p.smag, nu0: p.nuLat,
      uLat: p.uLat, belt: o.belt !== false, uBelt: 0,
      aux: new Float32Array(19 * rec.count), // float32 like the populations
      bb: new Float32Array(19 * rec.count),
      acc: new Float64Array(4 * rec.count),
      tauWall: new Float32Array(rec.count),
      slip: new Float32Array(3 * rec.count), // the wall model's filtered wall velocity
      rhoSum: new Float64Array(rec.count),
      odd: 0, uin: 0,
    };
    if (this.s.ab) { this.s.A = new Float32Array(19 * N); this.s.B = new Float32Array(19 * N); }
    else {
      this.s.F = new Float32Array(19 * N);
      this.s.bulk = bulkRuns(grid.kind, grid.dims);
      this.s.face = faceTable(grid.kind, grid.dims, this.s.faces, grid.periodicZ);
    }
    this.dims = grid.dims;
    this.N = N;
    this.grid = grid;
    this.t = scratch();
    this.reset();
  }

  reset() {
    const s = this.s;
    for (const buf of s.ab ? [s.A, s.B] : [s.F]) for (let i = 0; i < 19; i++) buf.fill(W[i], i * s.N, (i + 1) * s.N);
    for (let r = 0; r < s.rec.count; r++) for (let k = 0; k < 19; k++) { s.aux[19 * r + k] = W[k]; s.bb[19 * r + k] = W[k]; }
    s.acc.fill(0);
    s.slip.fill(0);
    s.rhoSum.fill(0);
    this.steps = 0;
    this.accSteps = 0;
    this.rhoSteps = 0;
  }

  /** Set every cell to the equilibrium at rho, u (tests; the next step's streaming pattern). */
  fillUniform(rho, ux, uy, uz) {
    const s = this.s, g = this.t.g;
    equilibrium(g, rho, ux, uy, uz, s.rr);
    for (const buf of s.ab ? [s.A, s.B] : [s.F]) {
      for (let i = 0; i < 19; i++) {
        // in place, a step of parity p loads f_i from slot i + 1 (even) or i (odd), for odd i
        let slot = i;
        if (!s.ab && i > 0 && (this.steps & 1) === 0) slot = i & 1 ? i + 1 : i - 1;
        buf.fill(g[i], slot * s.N, (slot + 1) * s.N);
      }
    }
  }

  /** Advance `count` steps. */
  step(count = 1) {
    const s = this.s, t = this.t;
    for (let k = 0; k < count; k++) {
      s.odd = this.steps & 1;
      s.uin = inletVelocity(s.uLat, this.steps);
      s.uBelt = s.belt ? s.uin : 0; // the ground belt ramps up with the wind
      if (this.threads) this.threads.step(this);
      else {
        bulkRows(s, t, 0, s.ny * s.nz);
        wallCells(s, t, 0, s.rec.count);
        faceCells(s, t, 0, s.faces.length);
      }
      if (s.ab) [s.A, s.B] = [s.B, s.A];
      this.steps++;
      this.accSteps++;
    }
  }

  /**
   * Forces on the part since the last call, summed over its steps (lattice units): momentum exchange
   * and the pressure integral over wetted voxel faces. Also adds to the wall cells' long-time
   * density sums (surfacePressure).
   */
  takeForces() {
    const { acc, rec, rhoSum } = this.s;
    const me = [0, 0, 0], pr = [0, 0, 0], fr = [0, 0, 0];
    for (let r = 0; r < rec.count; r++) {
      const fx = acc[4 * r], fy = acc[4 * r + 1], fz = acc[4 * r + 2];
      me[0] += fx; me[1] += fy; me[2] += fz;
      // friction: the part of the wall cell's force along the wall
      const nx_ = rec.normal[3 * r], ny_ = rec.normal[3 * r + 1], nz_ = rec.normal[3 * r + 2];
      const fn = fx * nx_ + fy * ny_ + fz * nz_;
      fr[0] += fx - fn * nx_; fr[1] += fy - fn * ny_; fr[2] += fz - fn * nz_;
      const drho = acc[4 * r + 3];
      rhoSum[r] += drho;
      // pressure (rho - 1) c_s^2 on the voxel faces toward the part: axis links 1-6
      const m = rec.mask[r] & ~rec.groundMask[r];
      for (let k = 1; k <= 6; k++) {
        if (!(m & (1 << k))) continue;
        pr[0] -= CX[k] * drho * CS2; pr[1] -= CY[k] * drho * CS2; pr[2] -= CZ[k] * drho * CS2;
      }
    }
    acc.fill(0);
    const steps = this.accSteps;
    this.rhoSteps += steps;
    this.accSteps = 0;
    return { steps, me, pressure: pr, friction: fr };
  }

  /** Start the long-time surface averages again (after the flow has developed). */
  resetAverages() {
    this.s.rhoSum.fill(0);
    this.rhoSteps = 0;
  }

  /** Time-averaged density minus one at each wall cell (for the surface pressure). */
  surfaceRho() {
    const { rhoSum, rec } = this.s;
    const out = new Float32Array(rec.count);
    if (this.rhoSteps) for (let r = 0; r < rec.count; r++) out[r] = rhoSum[r] / this.rhoSteps;
    return out;
  }

  /** [rho, ux, uy, uz] per cell now (solid cells: 1, 0, 0, 0). */
  macro() {
    const s = this.s, t = this.t;
    const out = new Float32Array(4 * s.N);
    const odd = this.steps & 1;
    for (let z = 0; z < s.nz; z++) {
      for (let y = 0; y < s.ny; y++) {
        for (let x = 0; x < s.nx; x++) {
          const n = x + s.nx * (y + s.ny * z);
          out[4 * n] = 1;
          if (s.kind[n] === SOLID || s.kind[n] === FACE) continue;
          neighbours(s, x, y, z, t.nb);
          load(s, n, t.nb, t.f, odd);
          let rho = 0, jx = 0, jy = 0, jz = 0;
          for (let i = 0; i < 19; i++) {
            let v = t.f[i];
            // links from solid cells: the bounced value (the wall position does not matter here)
            if (s.kind[n] === WALL) {
              const r = this.recordOf(n);
              if (s.rec.mask[r] & (1 << i)) v = s.bb[19 * r + i];
            }
            rho += v; jx += CX[i] * v; jy += CY[i] * v; jz += CZ[i] * v;
          }
          out[4 * n] = rho; out[4 * n + 1] = jx / rho; out[4 * n + 2] = jy / rho; out[4 * n + 3] = jz / rho;
        }
      }
    }
    // faces hold the equilibrium they were given: the free stream, or ambient pressure with the
    // velocity of the nearest interior cell
    const uin = inletVelocity(s.uLat, Math.max(0, this.steps - 1));
    for (const n of s.faces) {
      const x = n % s.nx, y = ((n / s.nx) | 0) % s.ny, z = (n / (s.nx * s.ny)) | 0;
      if (x === 0) { out.set([1, uin, 0, 0], 4 * n); continue; }
      const z0 = s.periodicZ ? z : Math.min(Math.max(z, 1), s.nz - 2);
      const n0 = Math.min(x, s.nx - 2) + s.nx * (Math.min(Math.max(y, 1), s.ny - 2) + s.ny * z0);
      out.set([1, out[4 * n0 + 1], out[4 * n0 + 2], out[4 * n0 + 3]], 4 * n);
    }
    return out;
  }

  /**
   * The reduced view fields now (flow-gpu.js readFields): [rho - 1, ux, uy, uz] averaged over the bulk
   * cells of each coarse cell of cf^3 cells (rho - 1 = -2 where there are none).
   */
  coarseFields(cf = 1) {
    const s = this.s, t = this.t, odd = this.steps & 1;
    const dims = [s.nx, s.ny, s.nz].map((d) => Math.ceil(d / cf));
    const nc = dims[0] * dims[1] * dims[2];
    const out = new Float32Array(4 * nc), count = new Uint32Array(nc);
    for (let z = 0; z < s.nz; z++) {
      for (let y = 0; y < s.ny; y++) {
        const row = s.nx * (y + s.ny * z);
        for (let x = 0; x < s.nx; x++) {
          const n = row + x;
          if (s.kind[n] !== BULK) continue;
          neighbours(s, x, y, z, t.nb);
          load(s, n, t.nb, t.f, odd);
          let rho = 0, jx = 0, jy = 0, jz = 0;
          for (let i = 0; i < 19; i++) { const v = t.f[i]; rho += v; jx += CX[i] * v; jy += CY[i] * v; jz += CZ[i] * v; }
          const c = Math.floor(x / cf) + dims[0] * (Math.floor(y / cf) + dims[1] * Math.floor(z / cf));
          out[4 * c] += rho - 1; out[4 * c + 1] += jx / rho; out[4 * c + 2] += jy / rho; out[4 * c + 3] += jz / rho;
          count[c]++;
        }
      }
    }
    for (let c = 0; c < nc; c++) {
      if (!count[c]) { out[4 * c] = -2; continue; }
      for (let k = 0; k < 4; k++) out[4 * c + k] /= count[c];
    }
    return { dims, factor: cf, inst: out };
  }

  recordOf(n) {
    if (!this.recIndex) {
      this.recIndex = new Map();
      for (let r = 0; r < this.s.rec.count; r++) this.recIndex.set(this.s.rec.cell[r], r);
    }
    return this.recIndex.get(n);
  }
}
