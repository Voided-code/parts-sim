// Exact wall distances for interpolated (Bouzidi) bounce-back.
//
// Plain bounce-back puts every wall half-way between lattice nodes, so a curved surface
// behaves like a staircase of tiny sharp edges; on rounded noses that kills the suction
// peak and badly over-predicts drag. Casting each fluid->solid lattice link against the
// real triangle mesh gives the fraction q of the link that lies in the fluid, which the
// solvers use to place the no-slip wall where the surface actually is.
import * as THREE from 'three';
import { MeshBVH } from 'three-mesh-bvh';
import { CX, CY, CZ } from './lbm-cpu.js';

/**
 * @param {Float32Array} positions  part vertices in the lattice (wind) frame, model units
 * @param {Uint32Array} tris
 * @param {{origin: number[], h: number, dims: number[]}} grid
 * @param {Uint8Array} solid
 * @returns {{links: Uint8Array, count: number}} links[i * N + c] = 1 + round(254 q) for the link
 *          from fluid cell c toward its solid neighbour c - c_i (q = fluid fraction of the link);
 *          0 means no link (or no surface found, which falls back to half-way bounce-back).
 */
export function wallLinks(positions, tris, { origin, h, dims }, solid) {
  const [nx, ny, nz] = dims;
  const N = nx * ny * nz;
  const links = new Uint8Array(19 * N);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  // copy: building a BVH reorders its index in place, and `tris` is the part's own array
  geometry.setIndex(new THREE.BufferAttribute(Uint32Array.from(tris), 1));
  const bvh = new MeshBVH(geometry, { indirect: true });
  const ray = new THREE.Ray();
  const dirs = CX.map((_, i) => new THREE.Vector3(-CX[i], -CY[i], -CZ[i]));
  const lens = dirs.map((d) => d.length() * h);
  for (const d of dirs) d.normalize();
  const off = CX.map((_, i) => CX[i] + nx * (CY[i] + ny * CZ[i]));
  let count = 0;
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const c = x + nx * (y + ny * z);
        if (solid[c]) continue;
        for (let i = 1; i < 19; i++) {
          if (!solid[c - off[i]]) continue;
          ray.origin.set(origin[0] + (x + 0.5) * h, origin[1] + (y + 0.5) * h, origin[2] + (z + 0.5) * h);
          ray.direction.copy(dirs[i]);
          const hit = bvh.raycastFirst(ray, THREE.DoubleSide, 0, lens[i]);
          if (!hit) continue;
          const q = Math.min(1, Math.max(0.01, hit.distance / lens[i]));
          links[i * N + c] = 1 + Math.round(254 * q);
          count++;
        }
      }
    }
  }
  geometry.dispose();
  return { links, count };
}
