import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import occtimportjs from 'occt-import-js';
import { importFile } from '../src/core/importers.js';
import { readCADMesh } from '../src/core/cad.js';
import { buildPart } from '../src/core/mesh.js';

const cubePositions = [0, 0, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0, 0, 0, 10, 10, 0, 10, 10, 10, 10, 0, 10, 10];
const cubeIndex = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3];
const partOf = (src, maxTris = 12) => buildPart(src, { maxTris });

function assertCube(part, size = 10) {
  assert.deepEqual(part.bbox.size, [size, size, size]);
  assert.ok(Math.abs(part.volume - size ** 3) < 1e-4);
  assert.equal(part.faceCount, 6);
  assert.ok(part.neighbors.every((n) => n >= 0));
}

test('OBJ and ASCII/binary STL import a watertight cube with identical dimensions', async () => {
  const obj = cubePositions.reduce((text, _, i, p) => i % 3 ? text : `${text}v ${p.slice(i, i + 3).join(' ')}\n`, '')
    + cubeIndex.reduce((text, _, i, p) => i % 3 ? text : `${text}f ${p.slice(i, i + 3).map((v) => v + 1).join(' ')}\n`, '');
  let stl = 'solid cube\n';
  const binary = new ArrayBuffer(84 + 12 * 50), view = new DataView(binary);
  view.setUint32(80, 12, true);
  for (let t = 0; t < 12; t++) {
    stl += 'facet normal 0 0 0\nouter loop\n';
    for (let c = 0; c < 3; c++) {
      const v = cubeIndex[3 * t + c];
      stl += `vertex ${cubePositions.slice(3 * v, 3 * v + 3).join(' ')}\n`;
      for (let d = 0; d < 3; d++) view.setFloat32(84 + 50 * t + 12 + 12 * c + 4 * d, cubePositions[3 * v + d], true);
    }
    stl += 'endloop\nendfacet\n';
  }
  stl += 'endsolid cube\n';
  for (const file of [new File([obj], 'cube.OBJ'), new File([stl], 'cube.stl'), new File([binary], 'binary.stl')]) {
    const src = await importFile(file);
    assert.equal(src.units, null);
    assertCube(partOf(src));
  }
});

test('STEP reader tessellates real CAD faces with millimeter dimensions', async () => {
  const occt = await occtimportjs();
  const buffer = await readFile(new URL('../node_modules/occt-import-js/test/testfiles/simple-basic-cube/cube.stp', import.meta.url));
  const src = readCADMesh(occt, 'step', buffer);
  assert.ok(src.bodies > 0);
  const part = partOf(src);
  assertCube(part, part.bbox.size[0]);
  assert.equal(part.brepFaces, true);
  assert.equal(part.faceOf.length, part.nTri);
  assert.throws(() => readCADMesh({ ReadFile: () => ({ success: true, meshes: [] }) }, 'step', new ArrayBuffer(0)), /no tessellated surfaces/);
});

test('mesh validation rejects corrupt indices and non-finite coordinates', () => {
  assert.throws(() => partOf({ positions: cubePositions, index: [0, 1, 99] }), /invalid triangle index/);
  assert.throws(() => partOf({ positions: [0, 0, 0, NaN, 1, 0, 1, 0, 0] }), /non-finite/);
  assert.throws(() => partOf({ positions: cubePositions, index: [0, 1] }), /complete triangles/);
  assert.throws(() => partOf({ positions: cubePositions, index: cubeIndex, faceIds: [0] }), /face data/);
});

test('unused source vertices do not change the part bounds or welding tolerance', () => {
  assertCube(partOf({ positions: [...cubePositions, 1e9, 1e9, 1e9], index: cubeIndex }));
});

test('mixed triangle winding and disconnected reversed shells face outward', () => {
  const positions = [...cubePositions, ...cubePositions.map((p, i) => p + (i % 3 === 0 ? 20 : 0))];
  const index = [...cubeIndex, ...cubeIndex.map((v) => v + 8)];
  for (let t = 0; t < index.length / 3; t++) {
    if (t % 3 || t >= 12) [index[3 * t + 1], index[3 * t + 2]] = [index[3 * t + 2], index[3 * t + 1]];
  }
  const part = partOf({ positions, index });
  assert.equal(part.volume, 2000);
  assert.equal(part.faceCount, 12);
  assert.ok(part.neighbors.every((n) => n >= 0));
  for (let t = 0; t < part.nTri; t++) {
    const p = [0, 0, 0];
    for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++) p[d] += part.vertices[3 * part.tris[3 * t + c] + d] / 3;
    const center = [p[0] < 0 ? -10 : 10, 5, 0];
    assert.ok(p.reduce((dot, v, d) => dot + (v - center[d]) * part.triNormal[3 * t + d], 0) > 0);
  }
});

test('enclosed cavity shells retain inward normals and subtract their volume', () => {
  const positions = [...cubePositions, ...cubePositions.map((v) => 2 + v * 0.6)];
  const index = [...cubeIndex, ...cubeIndex.map((v) => v + 8)];
  const part = partOf({ positions, index });
  assert.equal(part.volume, 1000 - 216);
});

test('refinement respects its triangle budget and retains a closed surface', () => {
  const part = partOf({ positions: cubePositions, index: cubeIndex }, 150);
  assert.ok(part.nTri > 12 && part.nTri <= 150);
  assertCube(part);
});

test('external glTF resources are rejected before network access', async () => {
  let requests = 0;
  const fetchOriginal = globalThis.fetch;
  globalThis.fetch = async () => { requests++; throw new Error('Unexpected request'); };
  try {
    await assert.rejects(importFile(new File([JSON.stringify({ asset: { version: '2.0' }, buffers: [{ uri: 'https://example.com/model.bin', byteLength: 32 }] })], 'remote.gltf')), /self-contained GLB/);
    assert.equal(requests, 0);
  } finally {
    globalThis.fetch = fetchOriginal;
  }
});

test('empty imports and OBJ files without surfaces report useful errors', async () => {
  await assert.rejects(importFile(new File([], 'empty.stl')), /empty/);
  await assert.rejects(importFile(new File(['v 0 0 0\nv 1 0 0\nl 1 2'], 'lines.obj')), /No meshes/);
});
