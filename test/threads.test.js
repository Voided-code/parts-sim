// The solvers give exactly the same results on helper threads (core/threads.js) as on one thread.
// (Each test stops its helpers: under the test runner they would keep the process alive.)
import './helpers/webworker.mjs';
import test from 'node:test';
import assert from 'node:assert/strict';
import { VoxelFEA } from '../src/fea/solver.js';
import { Threads, useThreads } from '../src/fea/threads.js';
import { naturalFrequencies, bucklingFactors, elementStresses } from '../src/fea/eigen.js';
import { NonlinearModel, loadRamp, hardeningFor } from '../src/fea/nonlinear.js';
import { cpuPreconditioner } from '../src/fea/eigen.js';
import { LBMCPU } from '../src/cfd/lbm-cpu.js';
import { LBMThreads } from '../src/cfd/lbm-threads.js';

const nu = 0.3;

/** A cantilever bar along x, clamped at x = 0, with a tip load (fx, fy) and a notch. */
function bar(nx = 64, n = 12, fx = 0, fy = -1) {
  const dims = [nx, n, n], NX = nx + 1, NY = n + 1, NZ = n + 1;
  const density = new Float32Array(nx * n * n).fill(1);
  for (let k = 0; k < n; k++) for (let i = 20; i < 23; i++) density[i + nx * (n - 1 + n * k)] = 0.5; // partly filled voxels
  const bc = new Uint8Array(3 * NX * NY * NZ), f = new Float64Array(bc.length);
  for (let k = 0; k < NZ; k++) for (let j = 0; j < NY; j++) {
    bc.fill(1, 3 * NX * (j + NY * k), 3 * NX * (j + NY * k) + 3);
    const tip = 3 * (nx + NX * (j + NY * k));
    f[tip] = fx / (NY * NZ);
    f[tip + 1] = fy / (NY * NZ);
  }
  return { dims, density, bc, f };
}

async function withThreads(model) {
  const fea = new VoxelFEA({ dims: model.dims, density: model.density, nu, bc: model.bc });
  const threads = await useThreads(fea, { minDof: 0, helpers: 2 });
  assert.ok(threads, 'helper threads started');
  return fea;
}

test('static solve, stresses and reactions: the same on helper threads', async () => {
  const m = bar();
  const one = new VoxelFEA({ dims: m.dims, density: m.density, nu, bc: m.bc });
  const many = await withThreads(m);
  const a = one.solve(m.f), b = many.solve(m.f);
  assert.equal(b.iterations, a.iterations);
  assert.deepStrictEqual(b.u, a.u);
  assert.deepStrictEqual(many.stresses(b.u, 2e11, 1e-3).nodeVM, one.stresses(a.u, 2e11, 1e-3).nodeVM);
  assert.deepStrictEqual(many.reactions(b.u, m.f), one.reactions(a.u, m.f));
  // the next model takes the helpers over
  many.threads.release();
  const m2 = bar(48, 12);
  const next = await withThreads(m2);
  const c = next.solve(m2.f), d = new VoxelFEA({ dims: m2.dims, density: m2.density, nu, bc: m2.bc }).solve(m2.f);
  assert.deepStrictEqual(c.u, d.u);
  next.threads.stop();
});

test('frequencies and buckling (LOBPCG in a shared arena): the same on helper threads', async () => {
  const m = bar(64, 12, -1, 0);
  const one = new VoxelFEA({ dims: m.dims, density: m.density, nu, bc: m.bc });
  const many = await withThreads(m);
  const opts = { nev: 3, E: 2e11, density: 7850, h: 1e-3 };
  const fa = await naturalFrequencies(one, opts), fb = await naturalFrequencies(many, opts);
  assert.equal(fb.iterations, fa.iterations);
  assert.deepStrictEqual(fb.freqs, fa.freqs);
  assert.deepStrictEqual(fb.modes, fa.modes);
  const u = one.solve(m.f).u.map((v) => v / (2e11 * 1e-3));
  const sigma = elementStresses(one, u, 1e-3);
  const ba = await bucklingFactors(one, sigma, { nev: 1 }), bb = await bucklingFactors(many, sigma, { nev: 1 });
  assert.deepStrictEqual(bb.factors, ba.factors);
  assert.deepStrictEqual(bb.modes, ba.modes);
  many.threads.stop();
});

test('nonlinear static with plasticity: the same on helper threads', async () => {
  const m = bar(48, 10, 0, -40);
  const plastic = hardeningFor({ E: 200, yield: 250, uts: 400, elongation: 0.2 });
  const f = m.f.map((v) => v / (2e11 * 1e-6));
  const run = async (fea) => {
    const model = new NonlinearModel(fea, { largeDisplacement: true, plastic });
    await model.share();
    const pre = cpuPreconditioner(fea);
    return loadRamp(model, f, async (r) => pre([r])[0], { target: 1, steps: 3, maxSteps: 6 });
  };
  const a = await run(new VoxelFEA({ dims: m.dims, density: m.density, nu, bc: m.bc }));
  const many = await withThreads(m);
  const b = await run(many);
  assert.equal(b.lam, a.lam);
  assert.deepStrictEqual(b.u, a.u);
  many.threads.stop();
});

test('flow solver: the same on helper threads', async () => {
  const dims = [48, 24, 24], N = dims[0] * dims[1] * dims[2];
  const solid = new Uint8Array(N);
  for (let z = 9; z < 15; z++) for (let y = 9; y < 15; y++) for (let x = 14; x < 20; x++) solid[x + dims[0] * (y + dims[1] * z)] = 1;
  const one = new LBMCPU({ dims, solid, uLat: 0.08, nuLat: 0.002 }), many = new LBMCPU({ dims, solid, uLat: 0.08, nuLat: 0.002 });
  assert.ok(await LBMThreads.start(many, { minCells: 0, helpers: 2 }), 'helper threads started');
  for (let s = 0; s < 20; s++) { one.step(s === 19); many.step(s === 19); }
  assert.deepStrictEqual(Float32Array.from(many.macro), Float32Array.from(one.macro));
  many.threads.stop();
});

test('threads need shared memory: none without cross-origin isolation', async () => {
  const isolated = globalThis.crossOriginIsolated;
  globalThis.crossOriginIsolated = false;
  try {
    const m = bar(16, 4);
    assert.equal(await Threads.start(new VoxelFEA({ dims: m.dims, density: m.density, nu, bc: m.bc }), { minDof: 0, helpers: 2 }), null);
  } finally {
    globalThis.crossOriginIsolated = isolated;
  }
});
