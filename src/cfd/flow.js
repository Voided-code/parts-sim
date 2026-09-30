// The v1 flow engine's lattice, tunnel description and physics shared by its CPU and GPU solvers
// (flow-cpu.js, flow-gpu.js / flow.wgsl, and the native app's copies).
//
// D3Q19 lattice. Directions come in opposite pairs (i, i + 1) for odd i, which the in-place
// ("esoteric pull", Lehmann 2022) streaming relies on. Wind along +x, lattice y up.
//
// Cells have a kind:
//   BULK  interior fluid with no solid neighbour (the fast path);
//   WALL  interior fluid next to a solid cell: it has a wall record (links, wall distance and normal);
//   SOLID the part, or the ground layer under a ground plane;
//   FACE  a cell on the tunnel's boundary (not solid): x = 0 is the inlet, the other faces are open
//         (ambient pressure, the velocity of the nearest interior cell). With a periodic span the
//         z faces wrap around instead.
export const BULK = 0, WALL = 1, SOLID = 2, FACE = 3;

export const CX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
export const CY = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1];
export const CZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1];
export const W = [1 / 3, ...Array(6).fill(1 / 18), ...Array(12).fill(1 / 36)];
export const OPP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17];
export const RAMP_STEPS = 400;

/** Inlet speed at a step: ramped up over the first RAMP_STEPS steps. */
export const inletVelocity = (uLat, step) => uLat * Math.min(1, (step + 1) / RAMP_STEPS);

// Third-order Hermite combinations that D3Q19 supports (Malaspinas 2015; Coreixas et al. 2017):
// P+ = H_xxy + H_yzz, H_xzz + H_xyy, H_yyz + H_xxz and P- the differences, with H_abc = c_a c_b c_c - cs^2 (...).
// Their lattice norms make 1/(2 cs^6) = 13.5 the coefficient of the P+ terms and 1/(6 cs^6) = 4.5 of the P- terms.
const h3 = (a, b) => a * (b * b - 1 / 3); // H_bba for a velocity with components a (odd) and b
export const P3 = CX.map((_, i) => {
  const x = CX[i], y = CY[i], z = CZ[i];
  const xxy = h3(y, x), yzz = h3(y, z), xzz = h3(x, z), xyy = h3(x, y), yyz = h3(z, y), xxz = h3(z, x);
  return [xxy + yzz, xzz + xyy, yyz + xxz, xxy - yzz, xzz - xyy, yyz - xxz];
});

/** Relaxation parameters: tau0 from the lattice viscosity, and the Smagorinsky factor 18 sqrt(2) C^2. */
export function flowParams({ uLat, nuLat, smagorinsky = 0.16 }) {
  if (!Number.isFinite(uLat) || uLat < 0 || uLat >= 0.3) throw new Error('Lattice speed must be finite and between 0 and 0.3.');
  if (!Number.isFinite(nuLat) || nuLat <= 0) throw new Error('Lattice viscosity must be finite and positive.');
  if (!Number.isFinite(smagorinsky) || smagorinsky < 0) throw new Error('Smagorinsky constant must be finite and non-negative.');
  return { uLat, nuLat, tau0: 3 * nuLat + 0.5, smag: 18 * Math.SQRT2 * smagorinsky * smagorinsky };
}

// ---------- wall model ----------

export const KAPPA = 0.41, B_LOG = 5.2;
const EKB = Math.exp(-KAPPA * B_LOG);

/**
 * Friction velocity from Spalding's law of the wall, given the tangential speed ut at distance y
 * from the wall and the kinematic viscosity nu (any consistent units). Newton on u+.
 */
export function spaldingUtau(ut, y, nu) {
  if (!(ut > 0) || !(y > 0)) return 0;
  // start from the viscous sublayer or the log law, whichever fits
  let utau = Math.sqrt((nu * ut) / y);
  if ((utau * y) / nu > 11) utau = ut / (Math.log((y * ut) / nu) / KAPPA + B_LOG - 1.5);
  for (let it = 0; it < 8; it++) {
    const up = ut / utau, k = KAPPA * up;
    const e = Math.exp(k);
    // y+ = u+ + e^{-kB} (e^{k u+} - 1 - k u+ - (k u+)^2/2 - (k u+)^3/6)
    const g = up + EKB * (e - 1 - k - (k * k) / 2 - (k * k * k) / 6) - (y * utau) / nu;
    const dgdup = 1 + EKB * KAPPA * (e - 1 - k - (k * k) / 2);
    // d/dutau of g: dg/du+ * (-u+/utau) - y/nu
    const d = dgdup * (-up / utau) - y / nu;
    const next = utau - g / d;
    utau = next > 0 ? next : 0.5 * utau;
    if (Math.abs(g) < 1e-6 * (1 + (y * utau) / nu)) break;
  }
  return utau;
}

const C_REICHARDT = 7.8;

/**
 * Reichardt's law of the wall (one formula from the viscous sublayer through the log layer):
 * [u+, du+/dy+] at y+.
 */
export function reichardt(yp) {
  const e11 = Math.exp(-yp / 11), e3 = Math.exp(-yp / 3);
  return [
    Math.log(1 + KAPPA * yp) / KAPPA + C_REICHARDT * (1 - e11 - (yp / 11) * e3),
    1 / (1 + KAPPA * yp) + C_REICHARDT * (e11 / 11 - e3 / 11 + (yp / 33) * e3),
  ];
}

/** Friction velocity u_tau with u_tau u+(y u_tau / nu) = ut (Reichardt), by safeguarded Newton. */
export function reichardtUtau(ut, y, nu) {
  if (!(ut > 0) || !(y > 0) || !(nu > 0)) return 0;
  let utau = Math.max(Math.sqrt((nu * ut) / y), ut / 30);
  for (let it = 0; it < 16; it++) {
    const yp = (y * utau) / nu;
    const [up, dup] = reichardt(yp);
    const F = utau * up - ut;
    const next = utau - F / (up + yp * dup);
    utau = next > 0 ? next : 0.5 * utau;
    if (Math.abs(F) < 1e-7 * ut) break;
  }
  return utau;
}

// ---------- tunnel description ----------

/**
 * Cell kinds and wall records for a tunnel plan (airflow.js planTunnel) and the part's triangles in
 * the wind frame (plan.q). The part is voxelized only within its bounding box.
 * @returns {{dims: number[], N: number, kind: Uint8Array, rec: object, ground: boolean, periodicZ: boolean}}
 */
export function buildFlowGrid(plan, tris, { voxelize, wallRayCaster }) {
  const { dims, h, origin } = plan;
  const [nx, ny, nz] = dims;
  const N = nx * ny * nz;
  const ground = plan.ground !== null && plan.ground !== undefined;
  const periodicZ = !!plan.periodicSpan;
  const kind = new Uint8Array(N);
  // the tunnel's faces, and the ground layer
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      const row = nx * (y + ny * z);
      const edge = y === 0 || y === ny - 1 || (!periodicZ && (z === 0 || z === nz - 1));
      if (ground && y === 0) kind.fill(SOLID, row, row + nx);
      else if (edge) kind.fill(FACE, row, row + nx);
      else { kind[row] = FACE; kind[row + nx - 1] = FACE; }
    }
  }
  // the part, voxelized within its bounding box (one cell of margin)
  const lo = [0, 1, 2].map((a) => Math.max(0, Math.floor((plan.min[a] - origin[a]) / h) - 1));
  const hi = [0, 1, 2].map((a) => Math.min(dims[a], Math.ceil((plan.max[a] - origin[a]) / h) + 1));
  if (periodicZ) { lo[2] = 0; hi[2] = nz; }
  const sub = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  let solidCount = 0;
  if (sub.every((s) => s > 0)) {
    const frac = voxelize(plan.q, tris, { origin: [0, 1, 2].map((a) => origin[a] + lo[a] * h), h, dims: sub }, 2);
    for (let z = 0; z < sub[2]; z++) {
      for (let y = 0; y < sub[1]; y++) {
        for (let x = 0; x < sub[0]; x++) {
          if (frac[x + sub[0] * (y + sub[1] * z)] < 0.5) continue;
          const c = lo[0] + x + nx * (lo[1] + y + ny * (lo[2] + z));
          if (kind[c] !== FACE) { kind[c] = SOLID; solidCount++; }
        }
      }
    }
  }
  // Wall cells: interior fluid cells with a link blocked by the part (toward a solid cell, or across
  // the surface to a fluid cell on its other side: thin walls, such as a wing's trailing edge, that no
  // cell centre lies inside), or by the ground. Candidates for thin walls lie within a cell of a
  // triangle; their links are cast against the part.
  const zOff = (z, dz) => (periodicZ ? (z + dz + nz) % nz : z + dz);
  const box = [hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]];
  const near = new Uint8Array(Math.max(1, box[0] * box[1] * box[2]));
  if (tris && plan.q) {
    const P = plan.q;
    for (let t = 0; t < tris.length; t += 3) {
      const a = [], b = [];
      for (let ax = 0; ax < 3; ax++) {
        let mn = Infinity, mx = -Infinity;
        for (let v = 0; v < 3; v++) {
          const p = (P[3 * tris[t + v] + ax] - origin[ax]) / h - 0.5;
          if (p < mn) mn = p;
          if (p > mx) mx = p;
        }
        a.push(Math.max(lo[ax], Math.floor(mn) - 1) - lo[ax]);
        b.push(Math.min(hi[ax] - 1, Math.ceil(mx) + 1) - lo[ax]);
      }
      for (let z = a[2]; z <= b[2]; z++) for (let y = a[1]; y <= b[1]; y++) near.fill(1, a[0] + box[0] * (y + box[1] * z), b[0] + 1 + box[0] * (y + box[1] * z));
    }
  }
  const cast = wallRayCaster ? wallRayCaster(plan) : null;
  const R = { cell: [], mask: [], ground: [], q: [], normal: [], dist: [] };
  const qb = new Uint8Array(18);
  const scanCell = (x, y, z, candidate) => {
    const c = x + nx * (y + ny * z);
    if (kind[c] !== BULK) return;
    let mask = 0, gmask = 0, best = Infinity, bn = null;
    qb.fill(0);
    for (let i = 1; i < 19; i++) {
      const s = x - CX[i] + nx * (y - CY[i] + ny * zOff(z, -CZ[i]));
      const solid = kind[s] === SOLID;
      if (solid && ground && y - CY[i] === 0) {
        mask |= 1 << i;
        gmask |= 1 << i;
        // the ground plane sits half-way between layers 0 and 1
        if (0.5 < best) { best = 0.5; bn = [0, 1, 0]; }
        continue;
      }
      if (!solid && !candidate) continue;
      const hit = cast ? cast(x, y, z, i) : null;
      if (!hit) { if (solid) mask |= 1 << i; continue; }
      mask |= 1 << i;
      qb[i - 1] = 1 + Math.round(254 * Math.min(1, Math.max(0.01, hit.q)));
      // perpendicular distance from the cell centre to the hit triangle's plane
      const cn = -CX[i] * hit.normal[0] - CY[i] * hit.normal[1] - CZ[i] * hit.normal[2];
      const d = hit.q * Math.abs(cn);
      if (d < best) {
        best = d;
        // the normal points into the fluid: against the link, which runs toward the wall
        const sg = cn > 0 ? -1 : 1;
        bn = [sg * hit.normal[0], sg * hit.normal[1], sg * hit.normal[2]];
      }
    }
    if (!mask) return;
    // interpolation (q < 1/2) needs the population from the next fluid cell away from the wall
    for (let i = 1; i < 19; i++) {
      if (!qb[i - 1] || (qb[i - 1] - 1) / 254 >= 0.5) continue;
      const n2 = x + CX[i] + nx * (y + CY[i] + ny * zOff(z, CZ[i]));
      if ((kind[n2] !== BULK && kind[n2] !== WALL) || mask & (1 << OPP[i])) qb[i - 1] = 128; // q = 1/2
    }
    if (!bn) {
      // no surface found (a staircase corner): half a cell toward the solid neighbours
      let sx = 0, sy = 0, sz = 0;
      for (let i = 1; i < 19; i++) if (mask & (1 << i)) { sx += CX[i]; sy += CY[i]; sz += CZ[i]; }
      const l = Math.hypot(sx, sy, sz) || 1;
      best = 0.5;
      bn = [sx / l, sy / l, sz / l];
    }
    kind[c] = WALL;
    R.cell.push(c); R.mask.push(mask); R.ground.push(gmask); R.q.push(...qb); R.normal.push(...bn); R.dist.push(Math.max(0.05, best));
  };
  const x0 = Math.max(1, lo[0]), x1 = Math.min(nx - 1, hi[0]);
  const y0 = Math.max(1, lo[1]), y1 = Math.min(ny - 1, hi[1]);
  const z0 = periodicZ ? 0 : Math.max(1, lo[2]), z1 = periodicZ ? nz : Math.min(nz - 1, hi[2]);
  for (let z = z0; z < z1; z++) {
    for (let y = y0; y < y1; y++) {
      for (let x = x0; x < x1; x++) scanCell(x, y, z, near[x - lo[0] + box[0] * (y - lo[1] + box[1] * (z - lo[2]))] === 1);
    }
  }
  // the cells above the ground layer, outside the part's box
  if (ground) {
    for (let z = periodicZ ? 0 : 1; z < (periodicZ ? nz : nz - 1); z++) {
      for (let x = 1; x < nx - 1; x++) {
        if (!(x >= x0 && x < x1 && z >= z0 && z < z1 && y0 <= 1 && y1 > 1)) scanCell(x, 1, z, false);
      }
    }
  }
  const rec = {
    count: R.cell.length, cell: Uint32Array.from(R.cell), mask: Uint32Array.from(R.mask), q: Uint8Array.from(R.q),
    normal: Float32Array.from(R.normal), dist: Float32Array.from(R.dist), groundMask: Uint32Array.from(R.ground),
  };
  wallModelData(plan, tris, rec, kind, { lo, hi, periodicZ });
  return { dims, N, kind, rec, ground, periodicZ, solidCount, h, origin };
}

/**
 * For the wall model (flow-cpu.js wallCells), per wall record:
 * - samp: the direction c_j nearest the wall normal whose cell n + c_j is bulk fluid, where the model
 *   samples the flow (0: none, the record keeps plain bounce-back), and y2, that cell's distance
 *   from the wall;
 * - area: the part's surface next to the record, as a vector into the fluid (cells^2), from its
 *   triangles (each goes to the nearest record on its outer side). With a periodic span only the
 *   part of each triangle inside the tunnel's span counts.
 */
function wallModelData(plan, tris, rec, kind, { lo, hi, periodicZ }) {
  const [nx, ny, nz] = plan.dims;
  const n = rec.count;
  rec.samp = new Uint8Array(n);
  rec.y2 = new Float32Array(n);
  rec.area = new Float32Array(3 * n);
  const zOff = (z, dz) => (periodicZ ? (z + dz + nz) % nz : z + dz);
  for (let r = 0; r < n; r++) {
    const c = rec.cell[r], x = c % nx, y = Math.floor(c / nx) % ny, z = Math.floor(c / (nx * ny));
    const nv = [rec.normal[3 * r], rec.normal[3 * r + 1], rec.normal[3 * r + 2]];
    let best = 0, bj = 0;
    for (let j = 1; j < 19; j++) {
      const dot = (CX[j] * nv[0] + CY[j] * nv[1] + CZ[j] * nv[2]) / Math.hypot(CX[j], CY[j], CZ[j]);
      if (dot <= best + 1e-6 || dot < 0.5) continue;
      const xx = x + CX[j], yy = y + CY[j], zz = zOff(z, CZ[j]);
      if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
      if (kind[xx + nx * (yy + ny * zz)] !== BULK) continue;
      best = dot; bj = j;
    }
    rec.samp[r] = bj;
    if (bj) rec.y2[r] = rec.dist[r] + CX[bj] * nv[0] + CY[bj] * nv[1] + CZ[bj] * nv[2];
  }
  if (!tris || !plan.q || !n) return;
  // records by cell within the part's box
  const box = [hi[0] - lo[0] + 2, hi[1] - lo[1] + 2, hi[2] - lo[2] + 2];
  const at = new Int32Array(box[0] * box[1] * box[2]).fill(-1);
  const boxIndex = (x, y, z) => {
    const bx = x - lo[0] + 1, by = y - lo[1] + 1, bz = z - lo[2] + 1;
    return bx < 0 || by < 0 || bz < 0 || bx >= box[0] || by >= box[1] || bz >= box[2] ? -1 : bx + box[0] * (by + box[1] * bz);
  };
  for (let r = 0; r < n; r++) {
    const c = rec.cell[r], i = boxIndex(c % nx, Math.floor(c / nx) % ny, Math.floor(c / (nx * ny)));
    if (i >= 0 && !(rec.groundMask[r] && rec.dist[r] === 0.5)) at[i] = r;
  }
  const { origin, h } = plan, P = plan.q;
  const v = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let t = 0; t < tris.length; t += 3) {
    for (let k = 0; k < 3; k++) for (let a = 0; a < 3; a++) v[k][a] = (P[3 * tris[t + k] + a] - origin[a]) / h;
    const e1 = [v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]], e2 = [v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]];
    // area vector (outward: the part's triangles face out)
    const A = [(e1[1] * e2[2] - e1[2] * e2[1]) / 2, (e1[2] * e2[0] - e1[0] * e2[2]) / 2, (e1[0] * e2[1] - e1[1] * e2[0]) / 2];
    const len = Math.hypot(A[0], A[1], A[2]);
    if (!(len > 0)) continue;
    let frac = 1;
    const cz = (v[0][2] + v[1][2] + v[2][2]) / 3;
    let zc = cz;
    if (periodicZ) {
      const z0 = Math.min(v[0][2], v[1][2], v[2][2]), z1 = Math.max(v[0][2], v[1][2], v[2][2]);
      frac = z1 > z0 ? Math.max(0, Math.min(z1, nz) - Math.max(z0, 0)) / (z1 - z0) : z0 >= 0 && z0 < nz ? 1 : 0;
      if (!frac) continue;
      zc = Math.min(nz - 0.5, Math.max(0.5, cz));
    }
    // a point just outside the surface, and the nearest record around it
    const p = [(v[0][0] + v[1][0] + v[2][0]) / 3 + (0.6 * A[0]) / len, (v[0][1] + v[1][1] + v[2][1]) / 3 + (0.6 * A[1]) / len, zc + (0.6 * A[2]) / len];
    const cx = Math.floor(p[0]), cy = Math.floor(p[1]), czz = Math.floor(p[2]);
    let bestR = -1, bestD = Infinity;
    for (let dz = -1; dz <= 1; dz++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const zz = periodicZ ? (czz + dz + nz) % nz : czz + dz;
          const i = boxIndex(cx + dx, cy + dy, zz);
          if (i < 0 || at[i] < 0) continue;
          const d = (cx + dx + 0.5 - p[0]) ** 2 + (cy + dy + 0.5 - p[1]) ** 2 + (czz + dz + 0.5 - p[2]) ** 2;
          if (d < bestD) { bestD = d; bestR = at[i]; }
        }
      }
    }
    if (bestR < 0) continue;
    for (let a = 0; a < 3; a++) rec.area[3 * bestR + a] += A[a] * frac;
  }
}
