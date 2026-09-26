// Mapping between the voxel grid's nodes and the part's surface vertices (shared by the main
// thread and the solver worker).

/**
 * Trilinear interpolation weights from grid nodes to every vertex, skipping inactive nodes.
 * A vertex just outside the solved voxels borrows the nearest active node (within 1.5 voxels);
 * anything farther has no result (drawn grey, undeformed) rather than someone else's value.
 * @param {Float32Array} vertices  xyz per vertex (model units)
 * @param {{origin: number[], h: number, dims: number[]}} grid
 * @param {Uint8Array} activeNode
 */
export function vertexWeights(vertices, { origin, h, dims }, activeNode) {
  const nV = vertices.length / 3;
  const NX = dims[0] + 1, NY = dims[1] + 1, NZ = dims[2] + 1;
  const node = (i, j, k) => i + NX * (j + NY * k);
  const idx = new Int32Array(8 * nV).fill(-1);
  const wts = new Float32Array(8 * nV);
  const g = [0, 0, 0], c = [0, 0, 0], t = [0, 0, 0];
  for (let v = 0; v < nV; v++) {
    for (let d = 0; d < 3; d++) {
      g[d] = (vertices[3 * v + d] - origin[d]) / h;
      c[d] = Math.min(dims[d] - 1, Math.max(0, Math.floor(g[d])));
      t[d] = Math.min(1, Math.max(0, g[d] - c[d]));
    }
    let sum = 0;
    for (let q = 0; q < 8; q++) {
      const n = node(c[0] + (q & 1), c[1] + ((q >> 1) & 1), c[2] + ((q >> 2) & 1));
      if (!activeNode[n]) continue;
      const w = (q & 1 ? t[0] : 1 - t[0]) * ((q >> 1) & 1 ? t[1] : 1 - t[1]) * ((q >> 2) & 1 ? t[2] : 1 - t[2]) + 1e-6;
      idx[8 * v + q] = n;
      wts[8 * v + q] = w;
      sum += w;
    }
    if (sum > 0) {
      for (let q = 0; q < 8; q++) wts[8 * v + q] /= sum;
      continue;
    }
    let best = -1, bestD = 2.25;
    const ci = Math.round(g[0]), cj = Math.round(g[1]), ck = Math.round(g[2]);
    for (let k = Math.max(0, ck - 2); k <= Math.min(NZ - 1, ck + 2); k++) {
      for (let j = Math.max(0, cj - 2); j <= Math.min(NY - 1, cj + 2); j++) {
        for (let i = Math.max(0, ci - 2); i <= Math.min(NX - 1, ci + 2); i++) {
          const n = node(i, j, k);
          if (!activeNode[n]) continue;
          const d = (i - g[0]) ** 2 + (j - g[1]) ** 2 + (k - g[2]) ** 2;
          if (d < bestD) { bestD = d; best = n; }
        }
      }
    }
    if (best >= 0) { idx[8 * v] = best; wts[8 * v] = 1; }
  }
  return { idx, wts, nV };
}

/** Interpolate a nodal field with `comps` components onto the vertices (NaN where no data). */
export function interpolate(W, field, comps = 1, scale = 1) {
  const nV = W.idx.length / 8;
  const out = new Float32Array(comps * nV);
  for (let v = 0; v < nV; v++) {
    let any = false;
    for (let q = 0; q < 8; q++) {
      const n = W.idx[8 * v + q];
      if (n < 0) continue;
      any = true;
      const w = W.wts[8 * v + q] * scale;
      for (let c = 0; c < comps; c++) out[comps * v + c] += w * field[comps * n + c];
    }
    if (!any) for (let c = 0; c < comps; c++) out[comps * v + c] = NaN;
  }
  return out;
}
