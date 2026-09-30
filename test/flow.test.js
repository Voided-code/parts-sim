// The v1 flow engine (src/cfd/flow*.js) on the CPU: streaming, walls, wall model, statistics.
import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { buildFlowGrid, reichardt, reichardtUtau, spaldingUtau, CX, CY, CZ, W, BULK, WALL, SOLID, FACE } from '../src/cfd/flow.js';
import { FlowCPU } from '../src/cfd/flow-cpu.js';
import { buildTunnel, syntheticFlowGrid, runCase } from '../src/cfd/flowcase.js';
import { caseById } from '../src/cfd/validation.js';
import { batchMeans } from '../src/cfd/stats.js';

test('in-place streaming matches two-buffer streaming exactly (walls, links, faces)', () => {
  const t = buildTunnel(caseById('sphere'), { across: 8, legacy: false });
  for (const collision of ['bgk', 'rr']) {
    const params = { uLat: 0.08, nuLat: 0.002, collision, wallModel: false };
    const a = new FlowCPU(t.grid, { ...params, layout: 'esoteric' });
    const b = new FlowCPU(t.grid, { ...params, layout: 'ab' });
    a.step(60);
    b.step(60);
    const ma = a.macro(), mb = b.macro();
    let d = 0;
    for (let i = 0; i < ma.length; i++) d = Math.max(d, Math.abs(ma[i] - mb[i]));
    // the same arithmetic up to rounding (the unrolled kernels fold the third-order coefficients)
    assert.ok(d < 1e-6, `${collision}: largest difference ${d}`);
    const fa = a.takeForces(), fb = b.takeForces();
    assert.ok(Math.abs(fa.me[0] - fb.me[0]) <= 1e-5 * Math.abs(fa.me[0]), `${collision} drag ${fa.me[0]} vs ${fb.me[0]}`);
  }
});

test('a free stream through an empty tunnel stays uniform', () => {
  const grid = syntheticFlowGrid(20000);
  grid.kind.fill(BULK);
  // no part: faces only
  const [nx, ny, nz] = grid.dims;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (x === 0 || y === 0 || z === 0 || x === nx - 1 || y === ny - 1 || z === nz - 1) grid.kind[x + nx * (y + ny * z)] = FACE;
  }
  grid.rec = { count: 0, cell: new Uint32Array(), mask: new Uint32Array(), q: new Uint8Array(), normal: new Float32Array(), dist: new Float32Array(), groundMask: new Uint32Array(), samp: new Uint8Array(), y2: new Float32Array(), area: new Float32Array() };
  const sim = new FlowCPU(grid, { uLat: 0.08, nuLat: 0.002 });
  // start from the free stream, past the inlet ramp
  sim.fillUniform(1, 0.08, 0, 0);
  sim.steps = 400;
  sim.step(30);
  const m = sim.macro();
  for (let c = 0; c < grid.N; c++) {
    assert.ok(Math.abs(m[4 * c] - 1) < 2e-6, `rho ${m[4 * c]}`);
    assert.ok(Math.abs(m[4 * c + 1] - 0.08) < 2e-6, `ux ${m[4 * c + 1]}`);
    assert.ok(Math.abs(m[4 * c + 2]) < 2e-6 && Math.abs(m[4 * c + 3]) < 2e-6);
  }
});

test('a plate with no cell inside it still stops the flow (thin-wall links)', () => {
  // a 0.2 mm thick plate across the flow in a 1 mm grid: no cell centre is inside it
  const make = () => {
    const g = new THREE.BoxGeometry(0.2, 12, 12).toNonIndexed();
    return Float32Array.from(g.attributes.position.array);
  };
  const c = { ...caseById('plate'), make, lref: 0.012 };
  const t = buildTunnel(c, { across: 12, legacy: false });
  const { kind, rec } = t.grid;
  let solid = 0;
  for (const k of kind) if (k === SOLID) solid++;
  assert.equal(solid, 0);
  assert.ok(rec.count > 200, `${rec.count} wall cells`);
  const sim = new FlowCPU(t.grid, { uLat: 0.08, nuLat: 0.01, collision: 'bgk' });
  sim.step(500);
  sim.takeForces();
  sim.step(200);
  const f = sim.takeForces();
  const cd = f.me[0] / f.steps / (0.5 * 0.08 ** 2 * 12 * 12);
  assert.ok(cd > 0.5 && cd < 2.5, `plate Cd ${cd}`);
});

test('the laws of the wall: Reichardt from the sublayer to the log layer, and the friction velocity', () => {
  const [u1] = reichardt(1), [u100, d100] = reichardt(100), [u1000] = reichardt(1000);
  assert.ok(Math.abs(u1 - 1) < 0.02, `u+(1) = ${u1}`);
  // Reichardt's formula tends to the log law with an intercept near 5.6
  assert.ok(Math.abs(u100 - (Math.log(100) / 0.41 + 5.6)) < 0.3, `u+(100) = ${u100}`);
  assert.ok(Math.abs(d100 - 1 / (0.41 * 100)) < 0.003);
  assert.ok(u1000 > u100);
  for (const [ut, y, nu] of [[0.05, 0.5, 1e-5], [0.08, 1.5, 3e-6], [0.001, 0.5, 1e-3]]) {
    const utau = reichardtUtau(ut, y, nu);
    const [up] = reichardt((y * utau) / nu);
    assert.ok(Math.abs(utau * up - ut) < 1e-6 * ut, `ut ${ut}: ${utau * up}`);
    const s = spaldingUtau(ut, y, nu);
    assert.ok(Math.abs(s - utau) < 0.07 * utau, `Spalding ${s} vs Reichardt ${utau}`);
  }
});

test('wall records carry the part\'s whole surface, once', () => {
  for (const [id, area] of [['cube', 6 * 100 * 100], ['sphere', Math.PI * 100 * 100]]) {
    const t = buildTunnel(caseById(id), { across: 16, legacy: false });
    const { rec } = t.grid, h = t.plan.h;
    let mag = 0;
    const sum = [0, 0, 0];
    for (let r = 0; r < rec.count; r++) {
      const a = [rec.area[3 * r], rec.area[3 * r + 1], rec.area[3 * r + 2]];
      mag += Math.hypot(...a);
      for (let k = 0; k < 3; k++) sum[k] += a[k];
    }
    assert.ok(Math.abs((mag * h * h) / area - 1) < 0.01, `${id}: ${mag * h * h} of ${area} mm^2`);
    // a closed surface's area vectors cancel
    assert.ok(Math.hypot(...sum) < 1e-6 * mag, `${id}: net area ${sum}`);
    for (let r = 0; r < rec.count; r++) assert.equal(t.grid.kind[rec.cell[r]], WALL);
  }
});

test('the 95% interval of a correlated series covers its mean about 95% of the time', () => {
  // mulberry32
  let seed = 7;
  const rand = () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let x = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    x = (x + Math.imul(x ^ (x >>> 7), 61 | x)) ^ x;
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
  const gauss = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  let covered = 0;
  const trials = 300;
  for (let t = 0; t < trials; t++) {
    // AR(1) noise around 3 with correlation 0.9
    const xs = [];
    let e = 0;
    for (let i = 0; i < 2000; i++) { e = 0.9 * e + gauss(); xs.push(3 + e); }
    const s = batchMeans(xs);
    if (Math.abs(s.mean - 3) <= s.ci) covered++;
  }
  assert.ok(covered / trials > 0.85 && covered / trials < 0.99, `coverage ${covered / trials}`);
});

test('flow past a coarse cube: a bluff-body drag coefficient from momentum exchange', async () => {
  const r = await runCase(caseById('cube'), { engine: 'cpu', solver: 'v1', across: 10, tol: 0.05, maxFlowThroughs: 5, wallModel: false });
  // broad regression bounds on this coarse grid, not an accuracy certification
  assert.ok(r.cd > 0.6 && r.cd < 1.4, `Cd ${r.cd}`);
  assert.ok(Math.abs(r.cl) < 0.1, `Cl ${r.cl}`);
  assert.equal(r.viscosityDoublings, 0);
});

void CX; void CY; void CZ; void W;
