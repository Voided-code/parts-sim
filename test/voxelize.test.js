import test from 'node:test';
import assert from 'node:assert/strict';
import { voxelize, gridForBox } from '../src/core/voxelize.js';

function uvSphere(r, seg = 48) {
  const pos = [], idx = [];
  for (let i = 0; i <= seg; i++) {
    const th = (i / seg) * Math.PI;
    for (let j = 0; j < 2 * seg; j++) {
      const ph = (j / (2 * seg)) * 2 * Math.PI;
      pos.push(r * Math.sin(th) * Math.cos(ph), r * Math.cos(th), r * Math.sin(th) * Math.sin(ph));
    }
  }
  const W = 2 * seg;
  for (let i = 0; i < seg; i++) {
    for (let j = 0; j < W; j++) {
      const a = i * W + j, b = i * W + ((j + 1) % W), c = (i + 1) * W + j, d = (i + 1) * W + ((j + 1) % W);
      if (i > 0) idx.push(a, b, c);
      if (i < seg - 1) idx.push(b, d, c);
    }
  }
  return { pos: Float32Array.from(pos), idx: Uint32Array.from(idx) };
}

test('sphere volume from voxel fractions', () => {
  const r = 10;
  const { pos, idx } = uvSphere(r);
  const grid = gridForBox([-r, -r, -r], [r, r, r], 24);
  const frac = voxelize(pos, idx, grid, 3);
  let vol = 0;
  for (const f of frac) vol += f;
  vol *= grid.h ** 3;
  const exact = (4 / 3) * Math.PI * r ** 3;
  console.log(`  voxel volume ${vol.toFixed(1)} vs exact ${exact.toFixed(1)} (${((vol / exact - 1) * 100).toFixed(2)}%)`);
  assert.ok(Math.abs(vol / exact - 1) < 0.02);
});

test('box is fully solid and grid-aligned', () => {
  // unit cube 0..10, 12 triangles
  const p = [0, 0, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0, 0, 0, 10, 10, 0, 10, 10, 10, 10, 0, 10, 10];
  const f = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
  const grid = gridForBox([0, 0, 0], [10, 10, 10], 5);
  const frac = voxelize(Float32Array.from(p), Uint32Array.from(f), grid, 2);
  assert.equal(frac.length, 125);
  for (const v of frac) assert.equal(v, 1);
});
