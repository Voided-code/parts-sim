// Robust mesh voxelization by ray casting.
//
// Each voxel is supersampled (sub^3 points). For every sample we cast rays along X, Y and Z
// and count surface crossings (even-odd rule); a sample is solid when at least two of the
// three rays agree. The majority vote tolerates small holes and flipped triangles that are
// common in STL/OBJ exports. The result is a volume fraction in [0, 1] per voxel.

/**
 * @param {ArrayLike<number>} pos   vertex positions (xyz)
 * @param {ArrayLike<number>} index triangle vertex indices (3 per triangle)
 * @param {{origin: number[], h: number, dims: number[]}} grid voxel i spans origin + [i, i+1) * h
 * @param {number} sub samples per voxel edge
 * @returns {Float32Array} filled fraction per voxel, index i + nx*(j + ny*k)
 */
export function voxelize(pos, index, grid, sub = 2) {
  const { origin, h, dims } = grid;
  const S = [dims[0] * sub, dims[1] * sub, dims[2] * sub];
  const ds = h / sub;
  const votes = new Uint8Array(S[0] * S[1] * S[2]);
  const strides = [1, S[0], S[0] * S[1]];
  const nTri = index.length / 3;
  // Tiny irrational offsets keep rays off shared edges/vertices of axis-aligned CAD geometry.
  const jitter = [0.000137 * ds, 0.000291 * ds, 0.000213 * ds];

  for (let a = 0; a < 3; a++) {
    const u = (a + 1) % 3, v = (a + 2) % 3;
    const Nu = S[u], Nv = S[v], Na = S[a];
    const ou = origin[u] + jitter[u], ov = origin[v] + jitter[v];
    const counts = new Uint32Array(Nu * Nv + 1);

    // Pass 1 counts hits per ray, pass 2 stores them.
    let hits = null;
    for (let pass = 0; pass < 2; pass++) {
      const fill = pass === 1 ? new Uint32Array(counts) : null;
      for (let t = 0; t < nTri; t++) {
        const i0 = 3 * index[3 * t], i1 = 3 * index[3 * t + 1], i2 = 3 * index[3 * t + 2];
        const au = pos[i0 + u], av = pos[i0 + v], aa = pos[i0 + a];
        const bu = pos[i1 + u], bv = pos[i1 + v], ba = pos[i1 + a];
        const cu = pos[i2 + u], cv = pos[i2 + v], ca = pos[i2 + a];
        const det = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);
        if (det === 0) continue;
        const minU = Math.min(au, bu, cu), maxU = Math.max(au, bu, cu);
        const minV = Math.min(av, bv, cv), maxV = Math.max(av, bv, cv);
        const su0 = Math.max(0, Math.ceil((minU - ou) / ds - 0.5));
        const su1 = Math.min(Nu - 1, Math.floor((maxU - ou) / ds - 0.5));
        const sv0 = Math.max(0, Math.ceil((minV - ov) / ds - 0.5));
        const sv1 = Math.min(Nv - 1, Math.floor((maxV - ov) / ds - 0.5));
        if (su0 > su1 || sv0 > sv1) continue;
        const inv = 1 / det;
        for (let sv = sv0; sv <= sv1; sv++) {
          const pv = ov + (sv + 0.5) * ds;
          for (let su = su0; su <= su1; su++) {
            const pu = ou + (su + 0.5) * ds;
            const l1 = ((bv - cv) * (pu - cu) + (cu - bu) * (pv - cv)) * inv;
            if (l1 < 0) continue;
            const l2 = ((cv - av) * (pu - cu) + (au - cu) * (pv - cv)) * inv;
            if (l2 < 0) continue;
            const l3 = 1 - l1 - l2;
            if (l3 < 0) continue;
            const ray = su + Nu * sv;
            if (pass === 0) counts[ray]++;
            else hits[fill[ray]++] = l1 * aa + l2 * ba + l3 * ca;
          }
        }
      }
      if (pass === 0) {
        // exclusive prefix sum
        let acc = 0;
        for (let r = 0; r < Nu * Nv; r++) {
          const c = counts[r];
          counts[r] = acc;
          acc += c;
        }
        counts[Nu * Nv] = acc;
        hits = new Float32Array(acc);
      }
    }

    const oa = origin[a];
    for (let sv = 0; sv < Nv; sv++) {
      for (let su = 0; su < Nu; su++) {
        const ray = su + Nu * sv;
        const start = counts[ray], end = counts[ray + 1];
        if (end - start < 2) continue;
        const list = hits.subarray(start, end).sort();
        const base = su * strides[u] + sv * strides[v];
        for (let q = 0; q + 1 < list.length; q += 2) {
          const s0 = Math.max(0, Math.ceil((list[q] - oa) / ds - 0.5));
          const s1 = Math.min(Na - 1, Math.floor((list[q + 1] - oa) / ds - 0.5));
          for (let s = s0; s <= s1; s++) votes[base + s * strides[a]]++;
        }
      }
    }
  }

  const [nx, ny, nz] = dims;
  const frac = new Float32Array(nx * ny * nz);
  const inv = 1 / (sub * sub * sub);
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        let c = 0;
        for (let dk = 0; dk < sub; dk++) {
          for (let dj = 0; dj < sub; dj++) {
            const row = (i * sub) + S[0] * (j * sub + dj + S[1] * (k * sub + dk));
            for (let di = 0; di < sub; di++) if (votes[row + di] >= 2) c++;
          }
        }
        frac[i + nx * (j + ny * k)] = c * inv;
      }
    }
  }
  return frac;
}

/** A voxel grid that encloses a bounding box with `n` voxels along its longest side. */
export function gridForBox(min, max, n, pad = 0) {
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  const h = Math.max(size[0], size[1], size[2]) / n;
  const dims = size.map((s) => Math.max(1, Math.ceil(s / h - 1e-6)) + 2 * pad);
  const origin = min.map((m, i) => m - (dims[i] * h - size[i]) / 2);
  return { origin, h, dims };
}
