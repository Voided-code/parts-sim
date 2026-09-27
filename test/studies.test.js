// Validation of the studies beyond linear static against textbook results.
import test from 'node:test';
import assert from 'node:assert/strict';
import { VoxelFEA } from '../src/fea/solver.js';
import { naturalFrequencies, bucklingFactors, elementStresses, lumpedMass } from '../src/fea/eigen.js';
import { NonlinearModel, loadRamp, hardeningFor, equilibrate } from '../src/fea/nonlinear.js';
import { cpuPreconditioner } from '../src/fea/eigen.js';
import { maxEigenvalue, dropTestCPU } from '../src/fea/explicit.js';
import { solveHeat, heatFlux } from '../src/fea/thermal.js';
import { harmonicField, timeHistory, excitation, shapeAt } from '../src/fea/response.js';
import { snCurve, cyclesToFailure, strengthAt, goodman, fatigueField } from '../src/fea/fatigue.js';
import { optimizeTopology } from '../src/fea/topology.js';
import { surfaceNets, toSTL } from '../src/core/isosurface.js';

const E = 200e9, rho = 7850, nu = 0.3;

/** Prismatic bar along x (length L, square side b, n voxels across), optionally clamped at x = 0. */
function bar({ n = 2, b = 0.01, L = 0.1, clamp = true, axis = 0 } = {}) {
  const h = b / n, len = Math.round(L / h);
  const dims = [n, n, n];
  dims[axis] = len;
  const [nx, ny, nz] = dims, NX = nx + 1, NY = ny + 1, NZ = nz + 1;
  const bc = new Uint8Array(3 * NX * NY * NZ);
  if (clamp) {
    for (let k = 0; k < NZ; k++) for (let j = 0; j < NY; j++) for (let i = 0; i < NX; i++) {
      const c = [i, j, k][axis];
      if (c === 0) bc.fill(1, 3 * (i + NX * (j + NY * k)), 3 * (i + NX * (j + NY * k)) + 3);
    }
  }
  return { dims, h, NX, NY, NZ, len, bc, density: new Float32Array(nx * ny * nz).fill(1), node: (i, j, k) => i + NX * (j + NY * k) };
}

/** Uniform traction on the x = L face along `dir` (total F) with consistent nodal weights. */
function tipForce(g, F, dir) {
  const f = new Float64Array(g.bc.length);
  const [, ny, nz] = g.dims;
  for (let k = 0; k <= nz; k++) for (let j = 0; j <= ny; j++) {
    const w = (j === 0 || j === ny ? 0.5 : 1) * (k === 0 || k === nz ? 0.5 : 1);
    f[3 * g.node(g.len, j, k) + dir] = (F * w) / (ny * nz);
  }
  return f;
}

test('natural frequencies of a cantilever match Euler-Bernoulli beam theory', async () => {
  const g = bar({ n: 2 });
  const fea = new VoxelFEA({ dims: g.dims, density: g.density, nu, bc: g.bc });
  const r = await naturalFrequencies(fea, { nev: 2, E, density: rho, h: g.h });
  const I = 0.01 ** 4 / 12, A = 1e-4;
  const f1 = (1.8751 ** 2 / (2 * Math.PI)) * Math.sqrt((E * I) / (rho * A * 0.1 ** 4));
  console.log(`  modes ${r.freqs.map((x) => x.toFixed(1)).join(', ')} Hz vs ${f1.toFixed(1)} Hz (twice: square section)`);
  assert.ok(r.converged);
  for (const f of r.freqs) assert.ok(Math.abs(f / f1 - 1) < 0.02, `${f} vs ${f1}`);
});

test('free-floating part: rigid-body modes are separated from the first flexible mode', async () => {
  const g = bar({ n: 2, clamp: false });
  const probe = new VoxelFEA({ dims: g.dims, density: g.density, nu, bc: g.bc, coarsestMaxDof: Infinity });
  const M = lumpedMass(probe);
  const shift = 1e-6 * maxEigenvalue(probe, M, 20);
  const fea = new VoxelFEA({ dims: g.dims, density: g.density, nu, bc: g.bc, diagAdd: M.map((v) => v * shift) });
  const r = await naturalFrequencies(fea, { nev: 8, E, density: rho, h: g.h, shift });
  const rigid = r.lambdas.filter((l) => l < 1e-3 * shift).length;
  const I = 0.01 ** 4 / 12, A = 1e-4;
  const f1 = (4.7300 ** 2 / (2 * Math.PI)) * Math.sqrt((E * I) / (rho * A * 0.1 ** 4));
  const first = r.freqs[6];
  console.log(`  ${rigid} rigid-body modes; first flexible ${first.toFixed(0)} Hz vs free-free theory ${f1.toFixed(0)} Hz`);
  assert.equal(rigid, 6);
  assert.ok(Math.abs(first / f1 - 1) < 0.05);
});

test('buckling load of a clamped column matches Euler', async () => {
  const g = bar({ n: 2, L: 0.2 });
  const fea = new VoxelFEA({ dims: g.dims, density: g.density, nu, bc: g.bc });
  const F = 1000;
  const sol = fea.solve(tipForce(g, -F, 0), { tol: 1e-9 });
  const u = sol.u.map((v) => v / (E * g.h));
  const r = await bucklingFactors(fea, elementStresses(fea, u, g.h), { nev: 1 });
  const Pcr = (Math.PI ** 2 * E * (0.01 ** 4 / 12)) / (4 * 0.2 ** 2);
  console.log(`  load factor ${r.factors[0].toFixed(3)} vs ${(Pcr / F).toFixed(3)}`);
  assert.ok(Math.abs(r.factors[0] / (Pcr / F) - 1) < 0.03);
  // the same search range as the app still finds it
  const rr = await bucklingFactors(fea, elementStresses(fea, u, g.h), { nev: 1, maxFactor: 1e4 });
  assert.ok(Math.abs(rr.factors[0] / r.factors[0] - 1) < 1e-3);
  // pulling the column never buckles it: the clamp's small sideways compression only gives
  // meaningless factors far beyond the range, and the search stops once that is clear
  const t = fea.solve(tipForce(g, F, 0), { tol: 1e-9 });
  const r2 = await bucklingFactors(fea, elementStresses(fea, t.u.map((v) => v / (E * g.h)), g.h), { nev: 1, maxFactor: 1e4 });
  console.log(`  pulled: no buckling in range after ${r2.iterations} iterations`);
  assert.equal(r2.factors[0], Infinity);
  assert.ok(r2.iterations < 100, `${r2.iterations} iterations`);
});

test('nonlinear: large deflection of a cantilever follows the elastica', async () => {
  const g = bar({ n: 2, L: 0.2 });
  const fea = new VoxelFEA({ dims: g.dims, density: g.density, nu, bc: g.bc });
  const pre = cpuPreconditioner(fea);
  const P = (E * (0.01 ** 4 / 12)) / 0.2 ** 2; // PL^2/EI = 1
  const f = tipForce(g, -P, 1).map((v) => v / (E * g.h * g.h));
  const model = new NonlinearModel(fea, { largeDisplacement: true, plastic: null });
  const r = await loadRamp(model, f, async (x) => pre([x])[0], { target: 1, steps: 5 });
  let tip = 0;
  for (let k = 0; k <= 2; k++) for (let j = 0; j <= 2; j++) tip += r.u[3 * g.node(g.len, j, k) + 1];
  tip = (-tip / 9) * g.h / 0.2;
  console.log(`  tip deflection ${tip.toFixed(4)} L (elastica 0.3017 L, linear theory 0.333 L)`);
  assert.equal(r.reason, 'reached');
  assert.ok(Math.abs(tip - 0.3017) < 0.01);
});

test('nonlinear: plastic collapse near the limit load and a permanent bend after unloading', async () => {
  const g = bar({ n: 4, L: 0.2 });
  const fea = new VoxelFEA({ dims: g.dims, density: g.density, nu, bc: g.bc });
  const pre = cpuPreconditioner(fea);
  const mat = { E: 200, yield: 250, uts: 300, elongation: 0.3 };
  const Plim = (250e6 * 0.01 ** 3) / 4 / 0.2;
  const f = tipForce(g, -Plim, 1).map((v) => v / (E * g.h * g.h));
  const model = new NonlinearModel(fea, { largeDisplacement: false, plastic: hardeningFor(mat) });
  const r = await loadRamp(model, f, async (x) => pre([x])[0], { target: Infinity, steps: 8, stepAlpha: 0.01, rupture: 0.3, maxDisp: 0.15 * g.len });
  console.log(`  stopped at ${r.lam.toFixed(3)} x the plastic limit load (${r.reason}); first yield at 0.667`);
  assert.ok(['collapse', 'large deformation'].includes(r.reason), r.reason);
  assert.ok(r.lam > 0.95 && r.lam < 1.3, `collapse factor ${r.lam}`);
  const u = Float64Array.from(r.u);
  const un = await equilibrate(model, u, new Float64Array(u.length), async (x) => pre([x])[0]);
  let tip = 0;
  for (let k = 0; k <= 4; k++) for (let j = 0; j <= 4; j++) tip += u[3 * g.node(g.len, j, k) + 1];
  assert.ok(un.converged);
  assert.ok(tip < 0, 'the unloaded beam stays bent down');
});

test('heat conduction: linear bar, cooling fin and transient energy balance', () => {
  const n = 4, b = 0.01, h = b / n, nx = 40, N = 41 * 25;
  const idx = (i, j, k) => i + 41 * (j + 5 * k);
  const density = new Float32Array(nx * n * n).fill(1), z = new Float64Array(N);
  const fixedNode = new Uint8Array(N), fixedValue = new Float64Array(N);
  for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) {
    fixedNode[idx(0, j, k)] = fixedNode[idx(nx, j, k)] = 1;
    fixedValue[idx(0, j, k)] = 100;
  }
  const r = solveHeat({ dims: [nx, n, n], density, fixedNode, fixedValue, source: z, convH: z, convT: z, k: 50, h });
  assert.ok(Math.abs(r.T[idx(20, 2, 2)] - 50) < 1e-6);
  assert.ok(Math.abs(heatFlux(r.solver, r.T, 50, h)[idx(20, 2, 2)] - 50000) < 1);
  // fin: base 100 C, h = 25 W/m2K all round, ambient 20 C
  const base = new Uint8Array(N), convH = new Float64Array(N), convT = new Float64Array(N).fill(20);
  for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) base[idx(0, j, k)] = 1;
  const face = (list) => { for (const m of list) convH[m] += (25 * h * h) / 4; };
  for (let i = 0; i < nx; i++) for (let t = 0; t < n; t++) {
    face([idx(i, 0, t), idx(i + 1, 0, t), idx(i, 0, t + 1), idx(i + 1, 0, t + 1)]);
    face([idx(i, n, t), idx(i + 1, n, t), idx(i, n, t + 1), idx(i + 1, n, t + 1)]);
    face([idx(i, t, 0), idx(i + 1, t, 0), idx(i, t + 1, 0), idx(i + 1, t + 1, 0)]);
    face([idx(i, t, n), idx(i + 1, t, n), idx(i, t + 1, n), idx(i + 1, t + 1, n)]);
  }
  for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) face([idx(nx, j, k), idx(nx, j + 1, k), idx(nx, j, k + 1), idx(nx, j + 1, k + 1)]);
  const fin = solveHeat({ dims: [nx, n, n], density, fixedNode: base, fixedValue: new Float64Array(N).fill(100), source: z, convH, convT, k: 200, h });
  const m = Math.sqrt((25 * 4 * b) / (200 * b * b)), hmk = 25 / (m * 200);
  const tipTheory = 20 + 80 / (Math.cosh(m * 0.1) + hmk * Math.sinh(m * 0.1));
  let tip = 0;
  for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) tip += fin.T[idx(nx, j, k)] / 25;
  console.log(`  fin tip ${tip.toFixed(3)} C vs ${tipTheory.toFixed(3)} C`);
  assert.ok(Math.abs(tip - tipTheory) < 0.05);
  // insulated bar heated at one end: mean temperature rise = Q t / (rho cp V)
  const src = new Float64Array(N);
  for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) src[idx(0, j, k)] = 10 / 25;
  const rhoCp = 2700 * 896;
  const tr = solveHeat({ dims: [nx, n, n], density, fixedNode: new Uint8Array(N), fixedValue: z, source: src, convH: z, convT: z, k: 167, h, rhoCp, duration: 60, steps: 20, initial: 20 });
  let mean = 0, w = 0;
  for (let k = 0; k <= n; k++) for (let j = 0; j <= n; j++) for (let i = 0; i <= nx; i++) {
    const c = (i === 0 || i === nx ? 0.5 : 1) * (j === 0 || j === n ? 0.5 : 1) * (k === 0 || k === n ? 0.5 : 1);
    mean += c * tr.T[idx(i, j, k)]; w += c;
  }
  assert.ok(Math.abs(mean / w - (20 + 600 / (rhoCp * b * b * 0.1))) < 1e-6);
});

test('drop test: bar hitting the floor end-on matches 1D impact theory', () => {
  const n = 4, b = 0.01, h = b / n, ny = 40, NX = n + 1, NY = ny + 1, N = NX * NY * NX;
  const fea = new VoxelFEA({ dims: [n, ny, n], density: new Float32Array(n * ny * n).fill(1), nu, bc: new Uint8Array(3 * N), coarsestMaxDof: Infinity });
  const nodeY = new Float64Array(N), surface = new Uint8Array(N);
  for (let k = 0; k <= n; k++) for (let j = 0; j <= ny; j++) for (let i = 0; i <= n; i++) {
    const q = i + NX * (j + NY * k);
    nodeY[q] = j * h;
    surface[q] = i === 0 || j === 0 || k === 0 || i === n || j === ny || k === n ? 1 : 0;
  }
  const v = Math.sqrt(2 * 9.81);
  const r = dropTestCPU(fea, { E, rho, h, speed: v, nodeY, surface });
  const c = Math.sqrt(E / rho);
  console.log(`  contact ${(r.contactTime * 1e6).toFixed(1)} us vs 2L/c ${(0.2 / c * 1e6).toFixed(1)} us; peak force ${r.peakForce.toFixed(0)} N vs rho c v A ${(rho * c * v * b * b).toFixed(0)} N`);
  assert.ok(r.rebounded);
  assert.ok(Math.abs(r.contactTime / (0.2 / c) - 1) < 0.1);
  assert.ok(Math.abs(r.peakForce / (rho * c * v * b * b) - 1) < 0.12);
});

test('modal superposition: harmonic and step responses of one mode are exact', () => {
  const w = 2 * Math.PI * 100;
  const basis = { omegas: [w], gamma: [1], modeU: [Float32Array.of(1, 0, 0)], modeS: [Float32Array.of(1, 0, 0, 0, 0, 0)], staticU: Float32Array.of(1 / w / w, 0, 0), staticS: Float32Array.of(1 / w / w, 0, 0, 0, 0, 0) };
  const z = 0.02, st = 1 / w / w;
  for (const f of [0, 50, 100, 200]) {
    const exact = st / Math.hypot(1 - (f / 100) ** 2, (2 * z * f) / 100);
    assert.ok(Math.abs(harmonicField(basis, 2 * Math.PI * f, z).disp[0] / exact - 1) < 1e-3, `${f} Hz`);
  }
  const hist = timeHistory(basis, excitation({ kind: 'step' }), { zeta: z, total: 0.05 });
  let peak = 0;
  for (let k = 0; k < hist.times.length; k++) peak = Math.max(peak, shapeAt(basis, hist, k)[0]);
  assert.ok(Math.abs(peak / st - (1 + Math.exp((-Math.PI * z) / Math.sqrt(1 - z * z)))) < 2e-3);
});

test('fatigue: S-N curve, Goodman correction and life field', () => {
  const steel = { uts: 420, fatigue: { Se: 210, Ne: 1e6, endurance: true, metal: true } };
  const c = snCurve(steel, 'polished');
  assert.equal(strengthAt(c, 1e3), 378);
  assert.equal(strengthAt(c, 1e7), 210);
  assert.equal(cyclesToFailure(c, 200), Infinity);
  assert.ok(Math.abs(Math.log10(cyclesToFailure(c, strengthAt(c, 3e4))) - Math.log10(3e4)) < 1e-9);
  assert.ok(Math.abs(goodman(100, 100, 420) - 100 / (1 - 100 / 420)) < 1e-9);
  assert.ok(snCurve(steel, 'machined').Se < 210);
  const f = fatigueField(Float32Array.of(300e6, 100e6), Float32Array.of(300e6, 100e6), Float32Array.of(0, 0), { material: steel, finish: 'polished', R: -1, cycles: 1e6 });
  assert.ok(f.life[0] < 1e6 && f.life[1] === Infinity);
  assert.equal(f.worst, 0);
  assert.ok(Math.abs(f.fos[1] - 2.1) < 1e-6);
});

test('topology optimization keeps the volume budget, stiffens the design and meshes a closed surface', async () => {
  const nx = 30, ny = 10, nz = 2, NX = 31, NY = 11, N = NX * NY * 3;
  const fill = new Float32Array(nx * ny * nz).fill(1);
  const bc = new Uint8Array(3 * N);
  for (let k = 0; k <= nz; k++) for (let j = 0; j <= ny; j++) bc.fill(1, 3 * (NX * (j + NY * k)), 3 * (NX * (j + NY * k)) + 3);
  const f = new Float64Array(3 * N);
  for (let k = 0; k <= nz; k++) f[3 * (nx + NX * NY * k) + 1] = -1;
  const keep = new Uint8Array(fill.length);
  for (let k = 0; k < nz; k++) { keep[nx - 1 + nx * ny * k] = 1; for (let j = 0; j < ny; j++) keep[nx * (j + ny * k)] = 1; }
  const r = await optimizeTopology({
    dims: [nx, ny, nz], fill, keep, volFrac: 0.5, maxIter: 25,
    solve: async (density, x0) => {
      const fea = new VoxelFEA({ dims: [nx, ny, nz], density, nu, bc });
      return { u: fea.solve(f, { x0, tol: 1e-6, maxIter: 800 }).u, fea, f };
    },
  });
  const first = r.history[0], last = r.history[r.history.length - 1];
  console.log(`  compliance ${first.compliance.toFixed(1)} -> ${last.compliance.toFixed(1)}, volume ${last.volume.toFixed(3)}`);
  assert.ok(Math.abs(last.volume - 0.5) < 0.01);
  assert.ok(last.compliance < 0.6 * first.compliance);
  const s = surfaceNets(r.density, [nx, ny, nz], [0, 0, 0], 1, 0.5);
  // closed, consistently oriented surface: every directed edge appears once and its reverse once
  const edges = new Map();
  for (let t = 0; t < s.index.length; t += 3) for (let e = 0; e < 3; e++) {
    const a = s.index[t + e], b = s.index[t + ((e + 1) % 3)];
    edges.set(`${a},${b}`, (edges.get(`${a},${b}`) || 0) + 1);
  }
  let bad = 0;
  for (const [k, cnt] of edges) { const [a, b] = k.split(','); if (cnt !== 1 || edges.get(`${b},${a}`) !== 1) bad++; }
  assert.ok(s.index.length > 0);
  assert.ok(bad / edges.size < 0.02, `${bad} of ${edges.size} edges not manifold`);
  assert.equal(new DataView(toSTL(s.positions, s.index)).getUint32(80, true), s.index.length / 3);
});
