// Topology optimization: remove material where it carries little load, keeping the part as stiff
// as possible for a target mass.
//
// SIMP (solid isotropic material with penalization): every voxel gets a design density x in
// [xmin, 1]; its stiffness scales with x^p (p = 3), so intermediate densities are inefficient and
// the optimizer is driven to solid/void. Compliance (the work of the loads, the inverse of
// stiffness) is minimized for a volume budget with the optimality-criteria update and a density
// filter of radius rmin voxels (no checkerboards, mesh-independent members).
// Voxels at fixtures and loads are kept solid so the supports and load paths stay attached.
// Each iteration is one linear static solve, warm-started from the previous one.

export const XMIN = 1e-3;

/** Neighbour lists of the density filter: weights max(0, rmin - distance) over solid voxels. */
export function densityFilter(dims, solid, rmin) {
  const [nx, ny, nz] = dims;
  const r = Math.ceil(rmin) - 1;
  const offs = [];
  for (let k = -r - 1; k <= r + 1; k++) for (let j = -r - 1; j <= r + 1; j++) for (let i = -r - 1; i <= r + 1; i++) {
    const w = rmin - Math.hypot(i, j, k);
    if (w > 0) offs.push([i, j, k, w]);
  }
  const ids = solid; // Int32Array of voxel indices taking part
  const pos = new Int32Array(nx * ny * nz).fill(-1);
  ids.forEach((e, q) => { pos[e] = q; });
  const start = new Int32Array(ids.length + 1);
  const nbr = [], wts = [];
  for (let q = 0; q < ids.length; q++) {
    const e = ids[q];
    const i = e % nx, j = ((e / nx) | 0) % ny, k = (e / (nx * ny)) | 0;
    for (const [di, dj, dk, w] of offs) {
      const a = i + di, b = j + dj, c = k + dk;
      if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
      const p = pos[a + nx * (b + ny * c)];
      if (p >= 0) { nbr.push(p); wts.push(w); }
    }
    start[q + 1] = nbr.length;
  }
  const W = new Float64Array(ids.length);
  for (let q = 0; q < ids.length; q++) for (let t = start[q]; t < start[q + 1]; t++) W[q] += wts[t];
  return { start, nbr: Int32Array.from(nbr), wts: Float64Array.from(wts), W };
}

function applyFilter(F, x, out) {
  for (let q = 0; q < x.length; q++) {
    let s = 0;
    for (let t = F.start[q]; t < F.start[q + 1]; t++) s += F.wts[t] * x[F.nbr[t]];
    out[q] = s / F.W[q];
  }
  return out;
}

// sensitivities of a filtered quantity back to the design variables (the filter's transpose)
function applyFilterT(F, g, out) {
  out.fill(0);
  for (let q = 0; q < g.length; q++) {
    const v = g[q] / F.W[q];
    for (let t = F.start[q]; t < F.start[q + 1]; t++) out[F.nbr[t]] += F.wts[t] * v;
  }
  return out;
}

/** u_e^T K0 u_e for every solved voxel of a VoxelFEA (unit stiffness, no density scale). */
export function elementEnergies(fea, u) {
  const L = fea.levels[0], K = fea.K0;
  const out = new Float64Array(L.elems.length), ue = new Float64Array(24);
  for (let e = 0; e < L.elems.length; e++) {
    const n0 = L.base[e];
    for (let a = 0; a < 8; a++) {
      const n = 3 * (n0 + L.off[a]);
      ue[3 * a] = u[n]; ue[3 * a + 1] = u[n + 1]; ue[3 * a + 2] = u[n + 2];
    }
    let s = 0;
    for (let r = 0; r < 24; r++) {
      let t = 0;
      for (let c = 0; c < 24; c++) t += K[r * 24 + c] * ue[c];
      s += ue[r] * t;
    }
    out[e] = s;
  }
  return out;
}

/**
 * @param {object} o
 * @param {number[]} o.dims
 * @param {Float32Array} o.fill        solid fraction per voxel from voxelization (0 = outside)
 * @param {Uint8Array} o.keep          per voxel, 1 = must stay solid
 * @param {number} o.volFrac           fraction of the (filled) volume to keep
 * @param {(density: Float32Array, x0: Float64Array|null) => Promise<{u: Float64Array, fea: object, f: Float64Array}>} o.solve
 * @param {(it: object) => boolean|void} [o.onIter]  return true to stop
 */
export async function optimizeTopology({ dims, fill, keep, volFrac, solve, penal = 3, rmin = 1.5, maxIter = 40, move = 0.2, tol = 0.01, onIter = null }) {
  const ids = [];
  for (let e = 0; e < fill.length; e++) if (fill[e] > 0) ids.push(e);
  const solid = Int32Array.from(ids);
  const m = solid.length;
  const F = densityFilter(dims, solid, rmin);
  const vol = new Float64Array(m);
  let totalVol = 0, keptVol = 0;
  const free = new Uint8Array(m);
  for (let q = 0; q < m; q++) {
    vol[q] = fill[solid[q]];
    totalVol += vol[q];
    if (keep[solid[q]]) keptVol += vol[q];
    else free[q] = 1;
  }
  const target = Math.max(volFrac * totalVol, keptVol * 1.02);
  const x = new Float64Array(m);
  // start uniform at the budget left after the kept voxels
  const start = Math.min(1, Math.max(XMIN, (target - keptVol) / Math.max(1e-12, totalVol - keptVol)));
  for (let q = 0; q < m; q++) x[q] = free[q] ? start : 1;
  const xPhys = new Float64Array(m), dc = new Float64Array(m), dcx = new Float64Array(m), dv = new Float64Array(m), dvx = new Float64Array(m);
  const density = new Float32Array(fill.length);
  let u0 = null, history = [], change = 1;
  let lastEnergy = null, last = null;
  for (let it = 0; it < maxIter; it++) {
    applyFilter(F, x, xPhys);
    for (let q = 0; q < m; q++) if (!free[q]) xPhys[q] = 1;
    for (let q = 0; q < m; q++) density[solid[q]] = Math.min(1, fill[solid[q]] * (XMIN + (1 - XMIN) * xPhys[q] ** penal));
    const sol = await solve(density, u0);
    u0 = sol.u;
    const energy = elementEnergies(sol.fea, sol.u);
    // solved voxels come back in fea.levels[0].elems order (== ascending voxel index == solid order
    // unless voxels were pruned); map by voxel index
    const L = sol.fea.levels[0];
    const byVoxel = new Map();
    for (let t = 0; t < L.elems.length; t++) byVoxel.set(L.elems[t], energy[t]);
    let c = 0;
    for (let i = 0; i < sol.f.length; i++) c += sol.f[i] * sol.u[i];
    let used = 0;
    for (let q = 0; q < m; q++) {
      const eng = byVoxel.get(solid[q]) || 0;
      dc[q] = -penal * (1 - XMIN) * fill[solid[q]] * xPhys[q] ** (penal - 1) * eng;
      dv[q] = vol[q];
      used += vol[q] * xPhys[q];
    }
    history.push({ it, compliance: c, volume: used / totalVol, change });
    lastEnergy = energy; last = sol;
    if (onIter && onIter({ it, compliance: c, volume: used / totalVol, change, xPhys, solid }) === true) break;
    if (it > 4 && change < tol) break;
    applyFilterT(F, dc, dcx);
    applyFilterT(F, dv, dvx);
    // optimality criteria: bisection on the volume multiplier
    let l1 = 0, l2 = 1e9;
    const xNew = new Float64Array(m);
    const phys = new Float64Array(m);
    while ((l2 - l1) / (l1 + l2 + 1e-30) > 1e-4) {
      const lm = 0.5 * (l1 + l2);
      for (let q = 0; q < m; q++) {
        if (!free[q]) { xNew[q] = 1; continue; }
        const B = Math.sqrt(Math.max(0, -dcx[q]) / (lm * dvx[q] + 1e-300));
        xNew[q] = Math.max(0, Math.max(x[q] - move, Math.min(1, Math.min(x[q] + move, x[q] * B))));
      }
      applyFilter(F, xNew, phys);
      let v = 0;
      for (let q = 0; q < m; q++) v += vol[q] * (free[q] ? phys[q] : 1);
      if (v > target) l1 = lm; else l2 = lm;
    }
    change = 0;
    for (let q = 0; q < m; q++) { change = Math.max(change, Math.abs(xNew[q] - x[q])); x[q] = xNew[q]; }
  }
  applyFilter(F, x, xPhys);
  for (let q = 0; q < m; q++) if (!free[q]) xPhys[q] = 1;
  const result = new Float32Array(fill.length);
  for (let q = 0; q < m; q++) result[solid[q]] = xPhys[q];
  return { density: result, history, keptFraction: keptVol / totalVol, last, lastEnergy };
}
