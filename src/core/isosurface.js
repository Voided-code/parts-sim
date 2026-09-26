// Smooth closed surface of a voxel density field (for the topology-optimization result).
//
// Naive surface nets: one vertex per grid cell that the iso-level crosses, placed at the mean
// of the edge crossings, and one quad per crossed grid edge joining the four cells around it.
// A light Laplacian smoothing pass rounds off the voxel steps. The field is sampled at voxel
// centres and padded with empty space so the surface is always closed.

/**
 * @param {Float32Array} density  per voxel (index i + nx*(j + ny*k))
 * @param {number[]} dims
 * @param {number[]} origin       world position of voxel (0,0,0)'s min corner
 * @param {number} h              voxel size
 * @param {number} level          iso-level (0.5 keeps voxels that are more than half solid)
 * @returns {{positions: Float32Array, index: Uint32Array}}
 */
export function surfaceNets(density, dims, origin, h, level = 0.5, smooth = 2) {
  const [nx, ny, nz] = dims;
  // padded sample grid: sample (a,b,c) = voxel (a-1,b-1,c-1) centre
  const SX = nx + 2, SY = ny + 2, SZ = nz + 2;
  const f = new Float32Array(SX * SY * SZ);
  for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
    f[i + 1 + SX * (j + 1 + SY * (k + 1))] = density[i + nx * (j + ny * k)];
  }
  const at = (a, b, c) => f[a + SX * (b + SY * c)];
  // cells between samples: (SX-1)(SY-1)(SZ-1)
  const CX = SX - 1, CY = SY - 1, CZ = SZ - 1;
  const cellVert = new Int32Array(CX * CY * CZ).fill(-1);
  const pos = [];
  const corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
  const edges = [[0, 1], [2, 3], [4, 5], [6, 7], [0, 2], [1, 3], [4, 6], [5, 7], [0, 4], [1, 5], [2, 6], [3, 7]];
  const v = new Float32Array(8);
  for (let c = 0; c < CZ; c++) for (let b = 0; b < CY; b++) for (let a = 0; a < CX; a++) {
    let inside = 0;
    for (let q = 0; q < 8; q++) {
      v[q] = at(a + corners[q][0], b + corners[q][1], c + corners[q][2]);
      if (v[q] > level) inside++;
    }
    if (inside === 0 || inside === 8) continue;
    let sx = 0, sy = 0, sz = 0, cnt = 0;
    for (const [p, q] of edges) {
      if ((v[p] > level) === (v[q] > level)) continue;
      const t = (level - v[p]) / (v[q] - v[p]);
      sx += corners[p][0] + t * (corners[q][0] - corners[p][0]);
      sy += corners[p][1] + t * (corners[q][1] - corners[p][1]);
      sz += corners[p][2] + t * (corners[q][2] - corners[p][2]);
      cnt++;
    }
    cellVert[a + CX * (b + CY * c)] = pos.length / 3;
    // sample (a,b,c) sits at voxel (a-1,b-1,c-1)'s centre: world = origin + (a - 0.5) h
    pos.push(a + sx / cnt, b + sy / cnt, c + sz / cnt);
  }
  const idx = [];
  const cell = (a, b, c) => (a < 0 || b < 0 || c < 0 || a >= CX || b >= CY || c >= CZ ? -1 : cellVert[a + CX * (b + CY * c)]);
  // each grid edge between two samples with a sign change gets a quad of the 4 cells around it
  for (let c = 0; c < SZ; c++) for (let b = 0; b < SY; b++) for (let a = 0; a < SX; a++) {
    const inA = at(a, b, c) > level;
    for (let axis = 0; axis < 3; axis++) {
      const a2 = a + (axis === 0), b2 = b + (axis === 1), c2 = c + (axis === 2);
      if (a2 >= SX || b2 >= SY || c2 >= SZ) continue;
      if (inA === at(a2, b2, c2) > level) continue;
      // the four cells sharing this edge
      let q;
      if (axis === 0) q = [cell(a, b - 1, c - 1), cell(a, b, c - 1), cell(a, b, c), cell(a, b - 1, c)];
      else if (axis === 1) q = [cell(a - 1, b, c - 1), cell(a - 1, b, c), cell(a, b, c), cell(a, b, c - 1)];
      else q = [cell(a - 1, b - 1, c), cell(a, b - 1, c), cell(a, b, c), cell(a - 1, b, c)];
      if (q.some((x) => x < 0)) continue;
      // orient outward: normals point from inside to outside
      if (inA) idx.push(q[0], q[1], q[2], q[0], q[2], q[3]);
      else idx.push(q[0], q[2], q[1], q[0], q[3], q[2]);
    }
  }
  let P = Float32Array.from(pos);
  const index = Uint32Array.from(idx);
  // Laplacian smoothing over the mesh graph
  if (smooth > 0 && index.length) {
    const nV = P.length / 3;
    const nb = Array.from({ length: nV }, () => new Set());
    for (let t = 0; t < index.length; t += 3) {
      for (let e = 0; e < 3; e++) { nb[index[t + e]].add(index[t + (e + 1) % 3]); nb[index[t + (e + 1) % 3]].add(index[t + e]); }
    }
    for (let s = 0; s < smooth; s++) {
      const Q = new Float32Array(P.length);
      for (let i = 0; i < nV; i++) {
        let x = 0, y = 0, z = 0;
        for (const j of nb[i]) { x += P[3 * j]; y += P[3 * j + 1]; z += P[3 * j + 2]; }
        const c = nb[i].size || 1;
        Q[3 * i] = 0.5 * P[3 * i] + 0.5 * (x / c);
        Q[3 * i + 1] = 0.5 * P[3 * i + 1] + 0.5 * (y / c);
        Q[3 * i + 2] = 0.5 * P[3 * i + 2] + 0.5 * (z / c);
      }
      P = Q;
    }
  }
  for (let i = 0; i < P.length; i += 3) {
    P[i] = origin[0] + (P[i] - 0.5) * h;
    P[i + 1] = origin[1] + (P[i + 1] - 0.5) * h;
    P[i + 2] = origin[2] + (P[i + 2] - 0.5) * h;
  }
  return { positions: P, index };
}

/** Binary STL of an indexed triangle mesh. */
export function toSTL(positions, index, name = 'part') {
  const nT = index.length / 3;
  const buf = new ArrayBuffer(84 + 50 * nT);
  const dv = new DataView(buf);
  const head = `${name} - Parts Sim`.slice(0, 79);
  for (let i = 0; i < head.length; i++) dv.setUint8(i, head.charCodeAt(i) & 0x7f);
  dv.setUint32(80, nT, true);
  let o = 84;
  for (let t = 0; t < nT; t++) {
    const a = 3 * index[3 * t], b = 3 * index[3 * t + 1], c = 3 * index[3 * t + 2];
    const ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
    const vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    for (const val of [nx, ny, nz]) { dv.setFloat32(o, val, true); o += 4; }
    for (const p of [a, b, c]) for (let d = 0; d < 3; d++) { dv.setFloat32(o, positions[p + d], true); o += 4; }
    dv.setUint16(o, 0, true);
    o += 2;
  }
  return buf;
}
