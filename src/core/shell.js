// Thin walls in the voxel model.
//
// Volume sampling keeps a voxel only when it is mostly filled, so a wall thinner than a voxel
// either vanishes or, when it runs diagonally through the grid, breaks into voxels that touch
// only along edges; those carry no load and the part falls apart in the solver. Here every
// voxel that a thin wall's surface passes through is kept (the voxels a plane crosses always
// form a face-connected layer), with a density that holds exactly the wall's volume. In-plane
// (membrane) stiffness and stress then match the real wall; bending of the wall itself is only
// approximated, which is why the automatic resolution still aims for about two voxels through
// every wall.
import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';

function bvhOf(part) {
  if (!part._bvh) {
    // indirect: the BVH must not reorder the index, so hit.faceIndex is the part's triangle
    // (and the part's own arrays stay untouched)
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(part.vertices, 3));
    g.setIndex(new THREE.BufferAttribute(Uint32Array.from(part.tris), 1));
    part._bvh = new MeshBVH(g, { indirect: true });
  }
  return part._bvh;
}

const ray = new THREE.Ray();

/** Distance through the material behind triangle t (to the opposite face), or Infinity beyond `far`. */
function wallThicknessAt(part, bvh, t, far) {
  const V = part.vertices, T = part.tris, N = part.triNormal;
  const eps = part.bbox.diag * 1e-6;
  let cx = 0, cy = 0, cz = 0;
  for (let k = 0; k < 3; k++) {
    cx += V[3 * T[3 * t + k]] / 3;
    cy += V[3 * T[3 * t + k] + 1] / 3;
    cz += V[3 * T[3 * t + k] + 2] / 3;
  }
  const nx = N[3 * t], ny = N[3 * t + 1], nz = N[3 * t + 2];
  ray.origin.set(cx - nx * eps, cy - ny * eps, cz - nz * eps);
  ray.direction.set(-nx, -ny, -nz);
  const hit = bvh.raycastFirst(ray, THREE.DoubleSide, 0, far);
  if (!hit || hit.faceIndex === t) return Infinity;
  const o = hit.faceIndex;
  // the far side of a wall faces the other way
  if (N[3 * o] * nx + N[3 * o + 1] * ny + N[3 * o + 2] * nz > -0.3) return Infinity;
  return hit.distance + eps;
}

/**
 * Typical thickness of the part's thin walls, from an area-weighted sample of the surface:
 * the 20th percentile of measured wall thickness, or Infinity for chunky parts.
 */
export function typicalWallThickness(part, samples = 1500) {
  if (part._wallThickness !== undefined) return part._wallThickness;
  const bvh = bvhOf(part);
  const far = part.bbox.diag * 0.15;
  const cdf = new Float64Array(part.nTri);
  let acc = 0;
  for (let t = 0; t < part.nTri; t++) cdf[t] = acc += part.triArea[t];
  const thick = [];
  let seed = 7;
  for (let s = 0; s < samples; s++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const r = (seed / 0x7fffffff) * acc;
    let lo = 0, hi = part.nTri - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < r) lo = mid + 1; else hi = mid;
    }
    thick.push(wallThicknessAt(part, bvh, lo, far));
  }
  thick.sort((a, b) => a - b);
  const finite = thick.filter(Number.isFinite);
  // walls must cover a meaningful share of the surface to count
  part._wallThickness = finite.length > samples * 0.15 ? thick[Math.floor(samples * 0.2)] : Infinity;
  return part._wallThickness;
}

// Akenine-Moller triangle / axis-aligned box overlap test (box given by centre and half size).
function triBoxOverlap(c, hs, a, b, d) {
  const v0 = [a[0] - c[0], a[1] - c[1], a[2] - c[2]];
  const v1 = [b[0] - c[0], b[1] - c[1], b[2] - c[2]];
  const v2 = [d[0] - c[0], d[1] - c[1], d[2] - c[2]];
  const e = [
    [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]],
    [v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]],
    [v0[0] - v2[0], v0[1] - v2[1], v0[2] - v2[2]],
  ];
  for (const ed of e) {
    for (let k = 0; k < 3; k++) {
      // axis = unit_k x ed
      const ax = k === 0 ? [0, -ed[2], ed[1]] : k === 1 ? [ed[2], 0, -ed[0]] : [-ed[1], ed[0], 0];
      const p0 = ax[0] * v0[0] + ax[1] * v0[1] + ax[2] * v0[2];
      const p1 = ax[0] * v1[0] + ax[1] * v1[1] + ax[2] * v1[2];
      const p2 = ax[0] * v2[0] + ax[1] * v2[1] + ax[2] * v2[2];
      const r = hs * (Math.abs(ax[0]) + Math.abs(ax[1]) + Math.abs(ax[2]));
      if (Math.min(p0, p1, p2) > r || Math.max(p0, p1, p2) < -r) return false;
    }
  }
  for (let k = 0; k < 3; k++) {
    if (Math.min(v0[k], v1[k], v2[k]) > hs || Math.max(v0[k], v1[k], v2[k]) < -hs) return false;
  }
  const n = [e[0][1] * e[1][2] - e[0][2] * e[1][1], e[0][2] * e[1][0] - e[0][0] * e[1][2], e[0][0] * e[1][1] - e[0][1] * e[1][0]];
  const dist = n[0] * v0[0] + n[1] * v0[1] + n[2] * v0[2];
  return Math.abs(dist) <= hs * (Math.abs(n[0]) + Math.abs(n[1]) + Math.abs(n[2]));
}

/**
 * Density that thin walls (thinner than `maxThickness`) contribute to each voxel of a grid,
 * 0 where there is no thin wall. Each wall-face triangle carries half the wall volume behind
 * it (area x thickness / 2; the opposite face carries the other half), shared among the voxels
 * it passes through. Averaging over each voxel's marked neighbours then spreads that volume
 * evenly along the layer, so stair-step corners of a diagonal wall are not weak links, while
 * the total volume - and so the wall's in-plane stiffness - is conserved.
 */
export function thinWallDensity(part, { origin, h, dims }, maxThickness) {
  const [nx, ny, nz] = dims;
  const vol = new Float32Array(nx * ny * nz);
  const bvh = bvhOf(part);
  const V = part.vertices, T = part.tris;
  const a = [0, 0, 0], b = [0, 0, 0], d = [0, 0, 0], c = [0, 0, 0];
  const hit = [];
  for (let t = 0; t < part.nTri; t++) {
    const thick = wallThicknessAt(part, bvh, t, maxThickness);
    if (!Number.isFinite(thick)) continue;
    for (let k = 0; k < 3; k++) {
      a[k] = V[3 * T[3 * t] + k];
      b[k] = V[3 * T[3 * t + 1] + k];
      d[k] = V[3 * T[3 * t + 2] + k];
    }
    const lo = [0, 1, 2].map((k) => Math.max(0, Math.floor((Math.min(a[k], b[k], d[k]) - origin[k]) / h)));
    const hi = [0, 1, 2].map((k) => Math.min(dims[k] - 1, Math.floor((Math.max(a[k], b[k], d[k]) - origin[k]) / h)));
    hit.length = 0;
    for (let k = lo[2]; k <= hi[2]; k++) {
      for (let j = lo[1]; j <= hi[1]; j++) {
        for (let i = lo[0]; i <= hi[0]; i++) {
          c[0] = origin[0] + (i + 0.5) * h;
          c[1] = origin[1] + (j + 0.5) * h;
          c[2] = origin[2] + (k + 0.5) * h;
          if (triBoxOverlap(c, h / 2, a, b, d)) hit.push(i + nx * (j + ny * k));
        }
      }
    }
    const share = (part.triArea[t] * thick) / 2 / Math.max(1, hit.length);
    for (const e of hit) vol[e] += share;
  }
  const out = new Float32Array(nx * ny * nz);
  const h3 = h * h * h;
  let count = 0;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const e = i + nx * (j + ny * k);
        if (!vol[e]) continue;
        let sum = 0, n = 0;
        for (let dk = -1; dk <= 1; dk++) {
          for (let dj = -1; dj <= 1; dj++) {
            for (let di = -1; di <= 1; di++) {
              const ii = i + di, jj = j + dj, kk = k + dk;
              if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue;
              const v = vol[ii + nx * (jj + ny * kk)];
              if (v) { sum += v; n++; }
            }
          }
        }
        out[e] = Math.min(1, sum / (n * h3));
        count++;
      }
    }
  }
  return { density: out, count };
}
