// Walls thinner than a voxel: they must stay connected at any orientation, and in-plane
// bending (the wall's membrane stiffness) must match beam theory.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { buildPart } from '../src/core/mesh.js';
import { StructuralModel } from '../src/fea/structural.js';
import { VoxelFEA, pruneFloating } from '../src/fea/solver.js';

const E = 200e9, nu = 0.3, F = 20;

/** Strip L x H (in plane) x t, rotated `deg` about its length axis (x). */
function strip({ L = 200, H = 20, t = 1, deg = 0 }) {
  const g = new THREE.BoxGeometry(L, H, t).rotateX((deg * Math.PI) / 180).toNonIndexed();
  return buildPart({ positions: Float32Array.from(g.attributes.position.array), name: 'strip' }, { maxTris: 60000 });
}

function tris(part, test) {
  const out = [];
  for (let t = 0; t < part.nTri; t++) {
    const c = [0, 1, 2].map((d) => (part.vertices[3 * part.tris[3 * t] + d] + part.vertices[3 * part.tris[3 * t + 1] + d] + part.vertices[3 * part.tris[3 * t + 2] + d]) / 3);
    const n = [0, 1, 2].map((d) => part.triNormal[3 * t + d]);
    if (test(c, n, part.bbox)) out.push(t);
  }
  return Int32Array.from(out);
}

/** Cantilever: clamp the x-min end, load the x-max end along the strip's in-plane height. */
function cantilever(part, deg, resolution) {
  const model = new StructuralModel(part, resolution);
  const a = (deg * Math.PI) / 180;
  const dir = [0, -Math.cos(a), -Math.sin(a)]; // in-plane, rotated with the strip
  const asm = model.assemble({
    fixtures: [{ name: 'root', patches: [{ tris: tris(part, (c, n, b) => n[0] < -0.9 && c[0] < b.min[0] + 0.01), clip: null }] }],
    loads: [{ name: 'tip', type: 'force', magnitude: F, dir, patches: [{ tris: tris(part, (c, n, b) => n[0] > 0.9 && c[0] > b.max[0] - 0.01), clip: null }] }],
    gravity: false, material: { E: 200, nu: 0.3, yield: 351, uts: 420, density: 7850 }, toMeters: 0.001,
  });
  const held = new Uint8Array(asm.bc.length / 3);
  for (let n = 0; n < held.length; n++) held[n] = asm.bc[3 * n];
  const pruned = pruneFloating(model.dims, model.density, held);
  const fea = new VoxelFEA({ dims: model.dims, density: pruned.density, nu, bc: asm.bc });
  const sol = fea.solve(asm.f, { tol: 1e-7, maxIter: 800 });
  const hm = model.h * 0.001;
  // tip deflection along the load: average over the loaded end's nodes
  let s = 0, c = 0;
  for (let n = 0; n < model.nNodes; n++) {
    if (!(asm.f[3 * n] || asm.f[3 * n + 1] || asm.f[3 * n + 2])) continue;
    s += (sol.u[3 * n] * dir[0] + sol.u[3 * n + 1] * dir[1] + sol.u[3 * n + 2] * dir[2]) / (E * hm);
    c++;
  }
  // largest von Mises stress in the voxels at mid-span (the strip's edges carry M*c/I)
  const u = sol.u.map((v) => v / (E * hm));
  const st = fea.stresses(u, E, hm);
  let midVM = 0;
  const xMid = Math.round(model.dims[0] / 2);
  for (let n = 0; n < model.nNodes; n++) {
    if (n % model.NX === xMid && st.nodeVM[n] > midVM) midVM = st.nodeVM[n];
  }
  return { tip: s / c, midVM, sol, removed: pruned.removed, voxels: model.voxelCount, thin: model.thinVoxels, h: model.h };
}

test('thin wall stays connected and bends like beam theory at any orientation', () => {
  const L = 0.2, H = 0.02, t = 0.001;
  const theory = (F * L ** 3) / (3 * E * ((t * H ** 3) / 12)) + (F * L) / ((5 / 6) * (E / (2 * (1 + nu))) * H * t);
  const midStress = (F * (L / 2) * (H / 2)) / ((t * H ** 3) / 12); // M*c/I at mid-span = 30 MPa
  for (const deg of [0, 30, 45]) {
    const r = cantilever(strip({ deg }), deg, 128);
    const ratio = r.tip / theory;
    const sRatio = r.midVM / midStress;
    console.log(`  ${deg}°: voxel ${r.h.toFixed(2)} mm (wall 1 mm), ${r.voxels} voxels (${r.thin} thin-wall), removed ${r.removed}, ${r.sol.iterations} its, tip ${(r.tip * 1e3).toFixed(3)} mm vs ${(theory * 1e3).toFixed(3)} mm (x${ratio.toFixed(2)}), mid-span stress ${(r.midVM / 1e6).toFixed(1)} vs ${(midStress / 1e6).toFixed(1)} MPa (x${sRatio.toFixed(2)})`);
    assert.ok(sRatio > 0.75 && sRatio < 1.35, `${deg}°: mid-span stress ratio ${sRatio}`);
    assert.ok(r.sol.converged, `${deg}°: solver converged`);
    assert.ok(r.removed < 0.02 * r.voxels, `${deg}°: wall stays connected (${r.removed} removed)`);
    assert.ok(ratio > 0.75 && ratio < 1.3, `${deg}°: deflection ratio ${ratio}`);
  }
});

test('wall-thickness and airflow ray casts leave the part triangles untouched', async () => {
  const { typicalWallThickness } = await import('../src/core/shell.js');
  const { wallLinks } = await import('../src/cfd/links.js');
  const part = strip({ deg: 30 });
  const before = Uint32Array.from(part.tris);
  typicalWallThickness(part);
  new StructuralModel(part, 64);
  const q = Float32Array.from(part.vertices);
  const grid = { origin: [-110, -5, -15], h: 4, dims: [55, 10, 10] };
  wallLinks(q, part.tris, grid, new Uint8Array(5500));
  assert.deepEqual(part.tris, before);
  assert.ok(Math.abs(typicalWallThickness(part) - 1) < 0.05, `measured wall ${typicalWallThickness(part)}`);
});
