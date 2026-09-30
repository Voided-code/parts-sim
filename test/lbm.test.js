import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { LBMCPU, W, CX, CY, CZ, inletVelocity, RAMP_STEPS } from '../src/cfd/lbm-cpu.js';
import { LBMGPU } from '../src/cfd/lbm-gpu.js';
import { AirflowStudy } from '../src/cfd/airflow.js';
import { pressureForceCoefficients } from '../src/cfd/forces.js';

test('uniform free stream stays uniform across all open faces', () => {
  const dims = [9, 7, 6], N = dims.reduce((a, b) => a * b);
  const sim = new LBMCPU({ dims, solid: new Uint8Array(N), uLat: 0.08, nuLat: 0.002 });
  for (let i = 0; i < 19; i++) {
    const cu = CX[i] * sim.uLat;
    sim.f.fill(W[i] * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * sim.uLat ** 2), i * N, (i + 1) * N);
  }
  sim.steps = RAMP_STEPS;
  for (let step = 0; step < 30; step++) sim.step(true);
  for (let c = 0; c < N; c++) {
    assert.ok(Math.abs(sim.macro[4 * c] - 1) < 2e-6);
    assert.ok(Math.abs(sim.macro[4 * c + 1] - sim.uLat) < 2e-6);
    assert.ok(Math.abs(sim.macro[4 * c + 2]) < 2e-6);
    assert.ok(Math.abs(sim.macro[4 * c + 3]) < 2e-6);
  }
});

test('reset reproduces the same inlet ramp and flow field', () => {
  const sim = new LBMCPU({ dims: [9, 7, 6], solid: new Uint8Array(378), uLat: 0.08, nuLat: 0.002 });
  for (let i = 0; i < 15; i++) sim.step(true);
  const first = sim.macro.slice();
  sim.reset();
  assert.equal(sim.steps, 0);
  for (let i = 0; i < 15; i++) sim.step(true);
  assert.deepEqual(sim.macro, first);
  assert.ok(Math.abs(sim.macro[1] - inletVelocity(0.08, 14)) < 1e-8);
});

test('invalid solver inputs fail before allocating distributions', () => {
  const params = { dims: [3, 3, 3], solid: new Uint8Array(27), uLat: 0.08, nuLat: 0.002 };
  for (const changed of [{ dims: [2, 3, 3] }, { solid: new Uint8Array(1) }, { nuLat: NaN }, { uLat: Infinity }]) {
    assert.throws(() => new LBMCPU({ ...params, ...changed }));
  }
});

test('GPU batches apply each CPU inlet ramp value, including the ramp endpoint', async () => {
  const offsets = [];
  let written;
  const sim = Object.assign(Object.create(LBMGPU.prototype), {
    dims: [3, 3, 3], N: 27, wgX: 1, wgY: 1, paramStride: 256,
    steps: RAMP_STEPS - 2, parity: 0, tau0: 0.506, smag: 0.1, uLat: 0.08,
    bindGroups: [{}, {}],
    device: {
      queue: { writeBuffer: (_buffer, _offset, data) => { written = data; }, submit: () => {}, onSubmittedWorkDone: () => Promise.resolve() },
      createCommandEncoder: () => ({
        beginComputePass: () => ({ setPipeline() {}, setBindGroup: (_index, _group, dynamic) => offsets.push(dynamic[0]), dispatchWorkgroups() {}, end() {} }),
        finish() {},
      }),
    },
  });
  await sim.step(4);
  assert.deepEqual(offsets, [0, 256, 512, 768]);
  for (let i = 0; i < 4; i++) {
    const uin = new Float32Array(written, i * 256 + 32, 4)[1];
    assert.ok(Math.abs(uin - inletVelocity(0.08, RAMP_STEPS - 2 + i)) < 1e-8);
  }
  assert.equal(sim.steps, RAMP_STEPS + 2);
});

test('pressure integration cancels uniform pressure and resolves a known face load', () => {
  const dims = [5, 5, 5], center = 2 + 5 * (2 + 5 * 2);
  const solid = new Uint8Array(125), rho = new Float32Array(125).fill(1.01);
  solid[center] = 1;
  const uniform = pressureForceCoefficients(rho, solid, dims, 0.08);
  assert.deepEqual(uniform.C, [0, 0, 0]);
  assert.equal(uniform.frontal, 1);
  rho.fill(1);
  rho[center - 1] = 1 + 3 * 0.5 * 0.08 ** 2 * 2;
  const loaded = pressureForceCoefficients(rho, solid, dims, 0.08);
  assert.ok(Math.abs(loaded.C[0] - 2) < 3e-5);
  assert.deepEqual(loaded.C.slice(1), [0, 0]);
});

function studyForTest() {
  return new AirflowStudy({ flowGroup: new THREE.Group(), onFrame: () => () => {} });
}

test('disposing during setup cancels before meshing or creating a worker', async () => {
  const study = studyForTest();
  const setup = study.setup({ vertices: Float32Array.of(0, 0, 0, 1, 1, 1), tris: new Uint32Array() }, {
    dir: new THREE.Vector3(1, 0, 0), speed: 20, airDensity: 1.225, toMeters: 1, engine: 'cpu', cells: 90e3,
  });
  study.dispose(false);
  await assert.rejects(setup, { name: 'AbortError' });
  assert.equal(study.ready, false);
  assert.equal(study.sim, undefined);
});

test('disposing an initializing worker terminates it and settles initialization', async () => {
  const original = globalThis.Worker;
  let worker;
  globalThis.Worker = class {
    constructor() { worker = this; }
    postMessage() {}
    terminate() { this.terminated = true; }
  };
  const study = studyForTest();
  Object.assign(study, { engine: 'CPU', dims: [3, 3, 3], solid: new Uint8Array(27), nuLat: 0.002 });
  try {
    const pending = study.createSolver();
    assert.equal(study.ready, false);
    study.dispose(false);
    await assert.rejects(pending, { name: 'AbortError' });
    assert.equal(worker.terminated, true);
    assert.equal(study.ready, false);
  } finally {
    study.dispose();
    if (original === undefined) delete globalThis.Worker;
    else globalThis.Worker = original;
  }
});

test('an unstable flow stops the study and says so, without changing the physics', () => {
  let status = '';
  const study = new AirflowStudy({ flowGroup: new THREE.Group(), onFrame: () => () => {} }, { onStatus: (m) => { status = m; } });
  Object.assign(study, { N: 8, dims: [8, 1, 1], running: true, engine: 'CPU', sim: { worker: { terminate() {}, postMessage() {} } }, nuLat: 1e-5 });
  study.onForces({ steps: 10, me: [NaN, 0, 0], pressure: [0, 0, 0], friction: [0, 0, 0] }, 20);
  assert.equal(study.running, false);
  assert.equal(study.results, null);
  assert.match(status, /unstable/);
  assert.equal(study.nuLat, 1e-5);
  study.dispose();
});

test('flow past a cube stays stable and gives a bluff-body drag coefficient', () => {
  const dims = [72, 36, 36];
  const [nx, ny, nz] = dims;
  const solid = new Uint8Array(nx * ny * nz);
  const s = 8, x0 = 18, y0 = 14, z0 = 14;
  for (let z = z0; z < z0 + s; z++) for (let y = y0; y < y0 + s; y++) for (let x = x0; x < x0 + s; x++) solid[x + nx * (y + ny * z)] = 1;
  const uLat = 0.08;
  const sim = new LBMCPU({ dims, solid, uLat, nuLat: 0.002 });
  const t0 = performance.now();
  const steps = 1400;
  const rhoAvg = new Float32Array(nx * ny * nz);
  let samples = 0;
  for (let i = 0; i < steps; i++) {
    const sample = i >= 900 && i % 10 === 0;
    sim.step(sample);
    if (sample) {
      for (let c = 0; c < rhoAvg.length; c++) rhoAvg[c] += sim.macro[4 * c];
      samples++;
    }
  }
  const ms = performance.now() - t0;
  for (let c = 0; c < rhoAvg.length; c++) rhoAvg[c] /= samples;
  let maxU = 0;
  for (let c = 0; c < nx * ny * nz; c++) {
    const u = Math.hypot(sim.macro[4 * c + 1], sim.macro[4 * c + 2], sim.macro[4 * c + 3]);
    assert.ok(Number.isFinite(u));
    maxU = Math.max(maxU, u);
  }
  const { C, frontal } = pressureForceCoefficients(rhoAvg, solid, dims, uLat);
  const cd = C[0] / frontal;
  const mlups = (nx * ny * nz * steps) / (ms * 1000);
  console.log(`  ${steps} steps in ${ms.toFixed(0)} ms (${mlups.toFixed(1)} MLUPS), max |u| ${(maxU / uLat).toFixed(2)} U, Cd ${cd.toFixed(2)}, lift ${(C[1] / frontal).toFixed(3)}`);
  assert.equal(frontal, s * s);
  assert.ok(maxU < 3 * uLat);
  // Broad regression bounds for this coarse pressure-only model, not an accuracy certification.
  assert.ok(cd > 0.7 && cd < 2.0, `Cd ${cd}`);
  assert.ok(Math.abs(C[1] / frontal) < 0.1);
});

test('wall links measure the true wall position along lattice links', async () => {
  const { wallLinks } = await import('../src/cfd/links.js');
  const { voxelize } = await import('../src/core/voxelize.js');
  // box occupying x in [10.3, 14.3], y and z in [3, 7] on a unit grid
  const g = new THREE.BoxGeometry(4, 4, 4).translate(12.3, 5, 5).toNonIndexed();
  const positions = Float32Array.from(g.attributes.position.array);
  const tris = Uint32Array.from({ length: positions.length / 3 }, (_, i) => i);
  const grid = { origin: [0, 0, 0], h: 1, dims: [20, 10, 10] };
  const N = 2000;
  const solid = Uint8Array.from(voxelize(positions, tris, grid, 2), (f) => (f >= 0.5 ? 1 : 0));
  const { links, count } = wallLinks(positions, tris, grid, solid);
  assert.ok(count > 0);
  // fluid cell (9, 5, 5) sees the face x = 10.3 through its +x link, stored under direction 2 (c = -x)
  const c = 9 + 20 * (5 + 10 * 5);
  assert.equal(solid[c], 0);
  assert.equal(solid[c + 1], 1);
  const q = (links[2 * N + c] - 1) / 254;
  assert.ok(Math.abs(q - 0.8) < 0.01, `q ${q}`);
  // diagonal link (+x, +y) from the same cell also crosses the face at x = 10.3
  const qd = (links[8 * N + c] - 1) / 254;
  assert.ok(Math.abs(qd - 0.8) < 0.01, `q diagonal ${qd}`);
  g.dispose();
});
