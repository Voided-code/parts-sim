import test from 'node:test';
import assert from 'node:assert/strict';
import { hexStiffness } from '../src/fea/hex8.js';
import { VoxelFEA, principalStresses, pruneFloating } from '../src/fea/solver.js';
import { StructuralModel, FEAJob, validateMaterial } from '../src/fea/structural.js';
import { buildPart } from '../src/core/mesh.js';
import { MATERIALS } from '../src/core/materials.js';

test('element stiffness has rigid-body null space and is symmetric', () => {
  const K = hexStiffness(0.3);
  for (let d = 0; d < 3; d++) {
    for (let r = 0; r < 24; r++) {
      let s = 0;
      for (let a = 0; a < 8; a++) s += K[r * 24 + 3 * a + d];
      assert.ok(Math.abs(s) < 1e-12, `translation ${d} row ${r}: ${s}`);
    }
  }
  for (let i = 0; i < 24; i++) for (let j = 0; j < 24; j++) assert.equal(K[i * 24 + j], K[j * 24 + i]);
});

test('principal stresses of a known tensor', () => {
  const out = [0, 0, 0];
  principalStresses(50, -20, 10, 30, 0, 0, out);
  // xy block eigenvalues: 15 +- sqrt(35^2 + 30^2)
  const r = Math.hypot(35, 30);
  assert.ok(Math.abs(out[0] - (15 + r)) < 1e-9);
  assert.ok(Math.abs(out[1] - 10) < 1e-9);
  assert.ok(Math.abs(out[2] - (15 - r)) < 1e-9);
});

/** Cantilever: length Lm along x, square section b; clamped at x = 0, tip load F in -y. */
function cantilever(n, { Lm = 0.1, b = 0.01, F = 100, E = 200e9, nu = 0.3 } = {}) {
  const h = b / n;
  const nx = Math.round(Lm / h), ny = n, nz = n;
  const density = new Float32Array(nx * ny * nz).fill(1);
  const NX = nx + 1, NY = ny + 1, NZ = nz + 1;
  const bc = new Uint8Array(3 * NX * NY * NZ);
  const f = new Float64Array(3 * NX * NY * NZ);
  for (let k = 0; k < NZ; k++) {
    for (let j = 0; j < NY; j++) {
      const n0 = NX * (j + NY * k);
      bc[3 * n0] = bc[3 * n0 + 1] = bc[3 * n0 + 2] = 1;
      // consistent nodal weights for a uniform traction on the tip face
      const wj = j === 0 || j === ny ? 0.5 : 1, wk = k === 0 || k === nz ? 0.5 : 1;
      f[3 * (nx + n0) + 1] = (-F * wj * wk) / (ny * nz);
    }
  }
  const t0 = performance.now();
  const fea = new VoxelFEA({ dims: [nx, ny, nz], density, nu, bc });
  const sol = fea.solve(f, { tol: 1e-7 });
  const ms = performance.now() - t0;
  const u = sol.u.map((v) => v / (E * h));
  let tip = 0;
  for (let k = 0; k < NZ; k++) for (let j = 0; j < NY; j++) tip += u[3 * (nx + NX * (j + NY * k)) + 1];
  tip /= NY * NZ;
  const I = (b * b ** 3) / 12;
  const G = E / (2 * (1 + nu));
  const expected = -(F * Lm ** 3) / (3 * E * I) - (F * Lm) / ((5 / 6) * G * b * b);
  const st = fea.stresses(u, E, h);
  const R = fea.reactions(sol.u);
  return { tip, expected, sol, ms, st, R, levels: fea.levels.length, nElem: nx * ny * nz };
}

test('cantilever tip deflection matches Timoshenko beam theory', () => {
  for (const n of [2, 4, 8]) {
    const r = cantilever(n);
    const err = Math.abs(r.tip / r.expected - 1);
    console.log(
      `  n=${n} (${r.nElem} voxels, ${r.levels} MG levels): tip ${(r.tip * 1e3).toFixed(4)} mm, theory ${(r.expected * 1e3).toFixed(4)} mm, ` +
        `err ${(err * 100).toFixed(2)}%, ${r.sol.iterations} PCG its, ${r.ms.toFixed(0)} ms`,
    );
    assert.ok(r.sol.converged, 'solver converged');
    assert.ok(err < 0.06, `error ${err}`);
    // Reaction balances the applied load.
    assert.ok(Math.abs(r.R[1] - 100) < 1e-3 * 100, `reaction ${r.R}`);
  }
});

test('cantilever surface bending stress matches M*c/I', () => {
  const n = 8;
  const r = cantilever(n);
  const nx = 80, NX = nx + 1, NY = n + 1;
  // Top surface node at mid-span, mid-width: M = F * L/2.
  const mid = nx / 2 + NX * (n + NY * (n / 2));
  const theoryMid = (100 * 0.05 * 0.005) / ((0.01 * 0.01 ** 3) / 12); // 30 MPa
  const vmMid = r.st.nodeVM[mid];
  console.log(`  mid-span surface von Mises ${(vmMid / 1e6).toFixed(2)} MPa, beam theory ${(theoryMid / 1e6).toFixed(2)} MPa`);
  assert.ok(Math.abs(vmMid / theoryMid - 1) < 0.05);
  let max = 0;
  for (const v of r.st.nodeVM) max = Math.max(max, v);
  console.log(`  max von Mises ${(max / 1e6).toFixed(1)} MPa at the clamped root (theory 60 MPa + clamp concentration)`);
  assert.ok(max > 0.95 * 60e6 && max < 1.4 * 60e6);
});

test('pruneFloating removes voxels not connected to supports', () => {
  const dims = [4, 1, 1];
  const density = Float32Array.of(1, 1, 0, 1);
  const held = new Uint8Array(5 * 2 * 2);
  held[0] = 1; // node (0,0,0)
  const { density: d, removed } = pruneFloating(dims, density, held);
  assert.deepEqual(Array.from(d), [1, 1, 0, 0]);
  assert.equal(removed, 1);
});

function cubeStudy() {
  const part = buildPart({
    positions: [0, 0, 0, 10, 0, 0, 10, 10, 0, 0, 10, 0, 0, 0, 10, 10, 0, 10, 10, 10, 10, 0, 10, 10],
    index: [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3],
  }, { maxTris: 12 });
  const model = new StructuralModel(part, 2);
  const patch = { tris: Int32Array.from([0, 1]), clip: null };
  return { model, patch, setup: { fixtures: [], gravity: false, material: MATERIALS[0], toMeters: 0.001 } };
}

test('overlapping selection patches do not double pressure and force directions normalize', () => {
  const { model, patch, setup } = cubeStudy();
  const pressure = { name: 'Pressure', type: 'pressure', magnitude: 1, patches: [patch] };
  const one = model.assemble({ ...setup, loads: [pressure] });
  const overlap = model.assemble({ ...setup, loads: [{ ...pressure, patches: [patch, patch] }] });
  assert.deepEqual(overlap.total, one.total);
  assert.ok(Math.abs(Math.hypot(...one.total) - 100) < 1e-4, '1 MPa on 100 mm² gives 100 N');
  const applied = model.assemble({ ...setup, loads: [{ name: 'Force', type: 'force', magnitude: 20, dir: [0, -4, 0], patches: [patch] }] });
  assert.ok(Math.abs(applied.total[1] + 20) < 1e-6);
  assert.throws(() => model.assemble({ ...setup, loads: [{ name: 'Force', type: 'force', magnitude: 20, dir: [0, 0, 0], patches: [patch] }] }), /nonzero force direction/);
});

test('invalid material constants and nonfinite loads are rejected', () => {
  for (const nu of [-1, 0.5, NaN, Infinity]) assert.throws(() => hexStiffness(nu), /Poisson/);
  for (const key of ['E', 'yield', 'uts', 'density']) assert.throws(() => validateMaterial({ ...MATERIALS[0], [key]: 0 }), /positive/);
  const fea = new VoxelFEA({ dims: [1, 1, 1], density: Float32Array.of(1), nu: 0.3, bc: new Uint8Array(24) });
  assert.throws(() => fea.solve(new Float64Array(24).fill(NaN)), /finite/);
});

test('support reaction includes loads applied directly to fixed nodes', () => {
  const bc = new Uint8Array(24).fill(1);
  const fea = new VoxelFEA({ dims: [1, 1, 1], density: Float32Array.of(1), nu: 0.3, bc });
  const f = new Float64Array(24);
  f[1] = -100;
  const sol = fea.solve(f);
  assert.deepEqual(fea.reactions(sol.u, f), [0, 100, 0]);
});

test('zero load clears an earlier displacement warm start', () => {
  const bc = new Uint8Array(24);
  for (let n = 0; n < 8; n += 2) bc.fill(1, 3 * n, 3 * n + 3);
  const fea = new VoxelFEA({ dims: [1, 1, 1], density: Float32Array.of(1), nu: 0.3, bc });
  const sol = fea.solve(new Float64Array(24), { x0: new Float64Array(24).fill(10) });
  assert.ok(sol.converged);
  assert.ok(sol.u.every((v) => v === 0));
});

test('worker cancellation ignores late messages and callback errors terminate the job', async (t) => {
  class MockWorker {
    postMessage() {}
    terminate() { this.terminated = true; }
  }
  const originalWorker = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  globalThis.Worker = MockWorker;
  t.after(() => {
    if (originalWorker) Object.defineProperty(globalThis, 'Worker', originalWorker);
    else delete globalThis.Worker;
  });
  let progress = 0;
  const job = new FEAJob({ f: new Float64Array(1) }, { onProgress: () => progress++ });
  job.cancel();
  await assert.rejects(job.promise, { cancelled: true });
  job.worker.onmessage({ data: { type: 'progress' } });
  assert.equal(progress, 0);
  assert.equal(job.worker.terminated, true);
  const badCallback = new FEAJob({ f: new Float64Array(1) }, { onStep: () => { throw new Error('Display failed'); } });
  badCallback.worker.onmessage({ data: { type: 'breakStep' } });
  await assert.rejects(badCallback.promise, /Display failed/);
  assert.equal(badCallback.worker.terminated, true);
});

test('the GPU solver runs the native app\'s shader, with coarse matrices rounded to halves', async () => {
  const { readFile } = await import('node:fs/promises');
  const { toHalf } = await import('../src/fea/gpu-solver.js');
  const js = await readFile(new URL('../src/fea/gpu-solver.js', import.meta.url), 'utf8');
  const start = js.indexOf('const SHADER = /* wgsl */ `') + 'const SHADER = /* wgsl */ `'.length;
  const native = await readFile(new URL('../native/shaders/fea.wgsl', import.meta.url), 'utf8');
  assert.equal(js.slice(start, js.indexOf('`;', start)).trim(), native.trim());
  // half bits -> value, to check the rounding
  const value = (h) => {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    return e === 0 ? s * m * 2 ** -24 : s * (1 + m / 1024) * 2 ** (e - 15);
  };
  for (const [v, h] of [[0, 0], [1, 0x3c00], [-2, 0xc000], [0.5, 0x3800], [65504, 0x7bff], [2 ** -24, 1], [1 + 2 ** -11, 0x3c00], [1 + 3 * 2 ** -11, 0x3c02]]) {
    assert.equal(toHalf(v), h, `${v}`);
  }
  let seed = 1;
  for (let i = 0; i < 10000; i++) {
    seed = (seed * 16807) % 2147483647;
    const v = (seed / 2147483647 - 0.5) * 2 ** ((i % 26) - 10); // up to 2^15, inside the half range
    const back = value(toHalf(v));
    assert.ok(Math.abs(back - v) <= Math.max(Math.abs(v) * 2 ** -11, 2 ** -25), `${v} -> ${back}`);
  }
});
