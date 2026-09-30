// Benchmark and validation run (bench.html). Runs without clicks: in the desktop app with --bench,
// or in a browser. Writes one JSON report, saved after every item so a crash still leaves data.
//
// URL parameters: suite = full | quick | kit | speed | check | tune | engine | backend | sustained; cases = comma-separated
// case ids (optional); solver = v0.6 | v1 (and its options, below).
import * as THREE from 'three';
import { CASES, caseById } from '../cfd/validation.js';
import { runCase, slope } from '../cfd/flowcase.js';
import { LBMGPU, gpuMaxCells } from '../cfd/lbm-gpu.js';
import { AirflowStudy, windDirection } from '../cfd/airflow.js';
import { SAMPLES } from '../core/samples.js';
import { buildPart } from '../core/mesh.js';

/* global __APP_VERSION__, __GIT_COMMIT__ */
const params = new URLSearchParams(location.search);
const suite = params.get('suite') || 'full';
const only = params.get('cases')?.split(',').filter(Boolean) || null;
// the solver and its options (experiments): solver = v1 (the app's) | v0.6 (the baseline), collision =
// rr | bgk, wm = 0 | 1, nufloor
const solver = params.get('solver') || 'v1';
const engineOptions = {
  solver,
  collision: params.get('collision') || 'rr',
  // wm = 1 | 0 forces the wall model on or off; without it, the app's automatic choice
  ...(params.get('wm') ? { wallModel: params.get('wm') !== '0' } : {}),
  wallMode: params.get('wmmode') || 'model',
  ...(params.get('nufloor') ? { nuFloor: Number(params.get('nufloor')) } : {}),
  ...(params.get('smag') ? { smagorinsky: Number(params.get('smag')) } : {}),
  ...(params.get('margin') ? { marginScale: Number(params.get('margin')) } : {}),
};
const acrossScale = Number(params.get('across') || 1);
const bridge = window.partsSimBench || null;

// peak memory bandwidth of GPUs we compare against (GB/s)
const PEAK_BANDWIDTH = [[/7800 ?XT/i, 624], [/Apple M5/i, 153]];

const report = {
  format: 'parts-sim-bench/1',
  app: { version: typeof __APP_VERSION__ !== 'undefined' ? __APP_VERSION__ : 'dev', commit: typeof __GIT_COMMIT__ !== 'undefined' ? __GIT_COMMIT__ : 'dev', shell: bridge ? 'electron' : 'browser' },
  suite,
  solver: engineOptions,
  started: new Date().toISOString(),
  units: { air: 'ISA sea level, 15 °C: density 1.225 kg/m³, dynamic viscosity 1.789e-5 Pa·s', force: 'N', coefficients: 'Cd = F / (0.5 ρ U² A_ref), A_ref and L_ref per case' },
  machine: { userAgent: navigator.userAgent, threads: navigator.hardwareConcurrency, crossOriginIsolated: self.crossOriginIsolated },
  adapter: null,
  validation: [],
  ladder: [],
  appLoop: [],
  sustained: null,
  errors: [],
  events: [],
};

const logEl = document.getElementById('log');
const stateEl = document.getElementById('state');
const barEl = document.querySelector('#bar > div');
function log(msg) {
  const line = `${new Date().toISOString().slice(11, 19)} ${msg}`;
  logEl.textContent += `${line}\n`;
  console.log(line);
}
const state = (msg) => { stateEl.textContent = msg; };
const progress = (f) => { barEl.style.width = `${Math.round(100 * Math.min(1, f))}%`; };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function save() {
  report.updated = new Date().toISOString();
  window.__benchReport = report;
  if (bridge) await bridge.write(JSON.stringify(report, null, 1));
}

function error(where, err) {
  const message = err?.message || String(err);
  report.errors.push({ where, message, at: new Date().toISOString() });
  log(`ERROR in ${where}: ${message}`);
}

const plain = (o) => {
  const out = {};
  for (const k in o) {
    const v = o[k];
    if (typeof v !== 'function') out[k] = v && typeof v === 'object' && !Array.isArray(v) ? plain(v) : v;
  }
  return out;
};

async function adapterInfo() {
  if (!navigator.gpu) return { available: false };
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) return { available: false };
  const info = plain(adapter.info || {});
  const peak = PEAK_BANDWIDTH.find(([re]) => re.test(`${info.description} ${info.device} ${info.architecture}`))?.[1] ?? null;
  // with --enable-webgpu-developer-features the adapter reports its memory heaps
  let vram = null;
  for (const hp of adapter.info?.memoryHeaps || []) if (hp.properties & 1) vram = Math.max(vram || 0, hp.size);
  return {
    available: true,
    info,
    isFallback: !!(adapter.info?.isFallbackAdapter || adapter.isFallbackAdapter),
    features: [...adapter.features].sort(),
    limits: plain(adapter.limits),
    wgslFeatures: [...(navigator.gpu.wgslLanguageFeatures || [])].sort(),
    peakBandwidthGBs: peak,
    deviceLocalBytes: vram,
  };
}

// ---------- validation ----------

function variants(c) {
  if (suite === 'quick' || suite === 'kit') return [{ variant: 'base' }];
  return [{ variant: 'base' }, { variant: 'viscosity x2', nuScale: 2 }, { variant: 'finer grid', across: Math.round(c.across * 1.26) }];
}

async function validation() {
  // the benchmark kit: every case but the 2D wing section (its three angles take long)
  const cases = CASES.filter((c) => (only ? only.includes(c.id) : suite === 'quick' ? ['plate', 'cube', 'sphere', 'wing', 'ahmed25'].includes(c.id) : suite === 'kit' ? c.id !== 'naca0012' : true));
  const jobs = [];
  for (const c of cases) for (const v of variants(c)) jobs.push([c, v]);
  let done = 0;
  for (const [c, v] of jobs) {
    const alphas = c.alphas || [undefined];
    const runs = [];
    for (const alpha of alphas) {
      const label = `${c.id}${alpha !== undefined ? ` α=${alpha}°` : ''} (${v.variant})`;
      state(`Validation: ${label}`);
      log(`${label}: meshing`);
      const t0 = performance.now();
      try {
        const r = await runCase(c, {
          ...engineOptions,
          engine: 'webgpu', across: Math.round((v.across || c.across) * acrossScale), nuScale: v.nuScale, alpha, tol: 0.02, maxFlowThroughs: 8,
          onProgress: (p) => state(`Validation: ${label} · ${p.phase} · step ${p.steps}${p.of ? ` of ${p.of}` : ''}${p.ci ? ` · ±${(100 * p.ci).toFixed(1)}%` : ''}`),
        });
        r.variant = v.variant;
        r.wallSeconds = (performance.now() - t0) / 1000;
        runs.push(r);
        if (r.skipped) log(`${label}: skipped (${r.skipped})`);
        else if (r.error) log(`${label}: ${r.error}`);
        else log(`${label}: ${r.dims.join('×')} cells, ${r.steps} steps, drag ${r.drag.toFixed(3)} ± ${r.dragCI.toFixed(3)} N, lift ${r.lift.toFixed(3)} N, Cd ${r.cd.toFixed(3)}, Cl ${r.cl.toFixed(3)}, ${r.wallSeconds.toFixed(0)} s${r.viscosityDoublings ? `, viscosity doubled ${r.viscosityDoublings}×` : ''}`);
      } catch (err) {
        error(`validation ${label}`, err);
        runs.push({ id: c.id, alpha: alpha ?? null, variant: v.variant, error: err.message });
      }
      report.validation.push(runs[runs.length - 1]);
      await save();
    }
    if (c.alphas && runs.every((r) => Number.isFinite(r.cl))) {
      report.validation.push({ id: c.id, variant: v.variant, clSlopePerDeg: slope(c.alphas, runs.map((r) => r.cl)), reference: c.ref.clSlope });
      await save();
    }
    progress(++done / jobs.length);
  }
}

// ---------- speed ---------------

/** A tunnel with a cube in it (like the native bench) of about n cells. */
function syntheticTunnel(n) {
  const ny = Math.max(16, Math.round(Math.cbrt(n / 2.5))), nz = ny, nx = Math.round(n / (ny * nz));
  const solid = new Uint8Array(nx * ny * nz);
  const s = Math.floor(ny / 4), x0 = Math.floor(nx / 4), y0 = (ny - s) >> 1, z0 = (nz - s) >> 1;
  for (let z = z0; z < z0 + s; z++) for (let y = y0; y < y0 + s; y++) solid.fill(1, x0 + nx * (y + ny * z), x0 + s + nx * (y + ny * z));
  return { dims: [nx, ny, nz], solid };
}

const LEGACY_BYTES = { half: 2 * 19 * 2 + 4, full: 2 * 19 * 4 + 4 }; // populations in and out, flags
const LEGACY_GPU_BYTES_PER_CELL = 2 * 38 + 4 + 16 + 19 + 4 + 16;
// v1: 19 populations read and written in place, one kind byte (wall and face cells are a few %)
const V1_BYTES = { half: 2 * 19 * 2 + 1, full: 2 * 19 * 4 + 1 };
const V1_GPU_BYTES_PER_CELL = { half: 19 * 2 + 1, full: 19 * 4 + 1 };

/** A benchmark solver of about n cells: the chosen solver on a tunnel with a cube in it. */
async function speedSim(n, o = {}) {
  if (solver === 'v1') {
    const { syntheticFlowGrid } = await import('../cfd/flowcase.js');
    const { FlowGPU } = await import('../cfd/flow-gpu.js');
    const grid = syntheticFlowGrid(n);
    const collision = o.collision || engineOptions.collision;
    const sim = await FlowGPU.create(grid, { uLat: 0.08, nuLat: collision === 'rr' ? 1e-5 : 0.002, collision, timing: true, ...o });
    return { sim, cells: grid.N, dims: grid.dims, storage: sim.half ? 'FP16S' : 'FP32', bytes: sim.half ? V1_BYTES.half : V1_BYTES.full };
  }
  const t = syntheticTunnel(n);
  const sim = await LBMGPU.create({ ...t, links: null, uLat: 0.08, nuLat: 0.002, timing: true, ...o });
  return { sim, cells: t.dims[0] * t.dims[1] * t.dims[2], dims: t.dims, storage: sim.half ? 'FP16S' : 'FP32', bytes: sim.half ? LEGACY_BYTES.half : LEGACY_BYTES.full };
}

async function measure(sim, cells, seconds) {
  // warm up, then run for about `seconds`, timing the GPU with timestamp queries
  // submissions of about 40 ms: never long enough for the OS to reset the GPU
  let batch = Math.max(1, Math.min(400, Math.round(2e7 / cells)));
  await sim.step(Math.min(20, batch));
  const g0 = sim.gpuSeconds, s0 = sim.steps, t0 = performance.now();
  while (performance.now() - t0 < seconds * 1000) {
    const tb = performance.now();
    await sim.step(batch);
    const dt = performance.now() - tb;
    batch = Math.max(1, Math.min(400, Math.round((batch * 40) / Math.max(1, dt))));
  }
  const wall = (performance.now() - t0) / 1000, steps = sim.steps - s0, gpu = sim.gpuSeconds - g0;
  if (sim.error) throw new Error(`GPU error: ${sim.error}`);
  // each batch here waits for the last, so the GPU runs nearly all the time: timestamps that add up to
  // much less than the wall-clock time are not trustworthy on this GPU (reported, not used)
  const suspect = gpu > 0 && gpu < 0.8 * wall;
  return {
    steps, wallSeconds: wall, gpuSeconds: gpu, timestampSuspect: suspect,
    kernelMLUPS: gpu > 0 && !suspect ? (cells * steps) / gpu / 1e6 : null, wallMLUPS: (cells * steps) / wall / 1e6,
  };
}

async function ladder(largest = Infinity) {
  const peak = report.adapter?.peakBandwidthGBs;
  let cap = 0;
  try { cap = solver === 'v1' ? Infinity : await gpuMaxCells(); } catch (err) { return error('ladder', err); }
  // leave room for the desktop: at most 70% of the GPU's memory (or 4 GB when it is not reported)
  const budget = 0.7 * (report.adapter?.deviceLocalBytes || 4e9);
  const perCell = solver === 'v1' ? V1_GPU_BYTES_PER_CELL.half : LEGACY_GPU_BYTES_PER_CELL;
  cap = Math.min(0.95 * cap, budget / perCell, largest);
  const sizes = [];
  for (let n = 1e6; n <= cap; n *= 2) sizes.push(n);
  report.ladderCap = { cells: Math.floor(cap), reason: 'min(storage binding limit, 70% of GPU memory)' };
  for (const n of sizes) {
    state(`Speed ladder: ${(n / 1e6).toFixed(0)}M cells`);
    let sim = null;
    try {
      const t = await speedSim(n);
      sim = t.sim;
      const { cells, bytes } = t;
      const m = await measure(sim, cells, 4);
      const row = {
        cells, dims: t.dims, solver, storage: t.storage, bytesPerCellStep: bytes, ...m,
        gbs: m.kernelMLUPS ? (m.kernelMLUPS * bytes) / 1e3 : null,
        busyPercent: m.gpuSeconds > 0 ? (100 * m.gpuSeconds) / m.wallSeconds : null,
      };
      row.percentOfPeak = peak && row.gbs ? (100 * row.gbs) / peak : null;
      report.ladder.push(row);
      log(`ladder ${(cells / 1e6).toFixed(1)}M: kernel ${row.kernelMLUPS?.toFixed(0)} MLUPS (${row.gbs?.toFixed(0)} GB/s${row.percentOfPeak ? `, ${row.percentOfPeak.toFixed(0)}% of peak` : ''}), wall ${row.wallMLUPS.toFixed(0)} MLUPS, GPU busy ${row.busyPercent?.toFixed(0)}%`);
    } catch (err) {
      error(`ladder ${n}`, err);
      report.ladder.push({ cells: n, error: err.message });
      await save();
      break;
    } finally {
      sim?.destroy();
    }
    await save();
    await sleep(200);
  }
}

/**
 * Kernel variants on one large grid (v1 engine): workgroup shapes along x and in 2D tiles, 16- and
 * 32-bit storage, both collision models. Every variant stays selectable, so the fastest on each GPU
 * can be picked from this table.
 */
async function tune(n = 32e6) {
  if (solver !== 'v1') return;
  const budget = 0.7 * (report.adapter?.deviceLocalBytes || 4e9);
  n = Math.min(n, budget / V1_GPU_BYTES_PER_CELL.full);
  report.tune = [];
  const variants = [];
  for (const [wgx, wgy] of [[64, 1], [128, 1], [256, 1], [32, 2], [32, 4], [64, 2], [64, 4], [16, 4], [16, 8]]) variants.push({ wgx, wgy, half: true, collision: 'rr' });
  variants.push({ wgx: 64, wgy: 1, half: false, collision: 'rr' }, { wgx: 64, wgy: 1, half: true, collision: 'bgk' });
  for (const v of variants) {
    const label = `${v.wgx}x${v.wgy} ${v.half ? '16-bit' : '32-bit'} ${v.collision}`;
    state(`Tuning: ${label}`);
    let sim = null;
    try {
      const t = await speedSim(n, v);
      sim = t.sim;
      if (v.half && !sim.half) continue; // no 16-bit storage here
      const m = await measure(sim, t.cells, 3);
      const row = { ...v, cells: t.cells, bytesPerCellStep: t.bytes, ...m, gbs: m.kernelMLUPS ? (m.kernelMLUPS * t.bytes) / 1e3 : null };
      report.tune.push(row);
      log(`tune ${label}: kernel ${row.kernelMLUPS?.toFixed(0)} MLUPS (${row.gbs?.toFixed(0)} GB/s), wall ${row.wallMLUPS.toFixed(0)}`);
    } catch (err) {
      error(`tune ${label}`, err);
    } finally {
      sim?.destroy();
    }
    await save();
    await sleep(200);
  }
}

/** The app's own airflow loop (particles, averaging, forces) on a sample, for whole-app speed. */
async function appLoop(sampleId, cells, seconds) {
  const sample = SAMPLES.find((s) => s.id === sampleId);
  const src = sample.make();
  const part = buildPart({ ...src, name: sample.name });
  const callbacks = new Set();
  const viewer = { flowGroup: new THREE.Group(), onFrame: (cb) => { callbacks.add(cb); return () => callbacks.delete(cb); } };
  let raf = 0, last = performance.now();
  const tick = () => {
    const t = performance.now();
    for (const cb of callbacks) cb(Math.min(0.1, (t - last) / 1000));
    last = t;
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  // the solver being measured: the v1 study, or v0.6's (the baseline)
  const Study = solver === 'v1' ? AirflowStudy : (await import('../cfd/legacy-study.js')).LegacyAirflowStudy;
  const study = new Study(viewer, { onStatus: () => {} });
  study.autoStop = false;
  LBMGPU.timing = true;
  const { FlowGPU } = await import('../cfd/flow-gpu.js');
  FlowGPU.timing = true;
  try {
    const air = sample.airflow;
    await study.setup(part, { dir: windDirection(air.yaw, air.pitch), speed: air.speed, airDensity: 1.225, cells, engine: 'gpu', toMeters: 0.001, ground: air.ground ?? null });
    study.start();
    // skip the first seconds (shader warm-up), then measure
    await sleep(3000);
    const s0 = study.sim.steps, g0 = study.sim.gpuSeconds, t0 = performance.now();
    await sleep(seconds * 1000);
    const steps = study.sim.steps - s0, gpu = study.sim.gpuSeconds - g0, wall = (performance.now() - t0) / 1000;
    const row = {
      sample: sampleId, cells: study.N, dims: study.dims, engine: study.engine, seconds: wall, steps,
      wallMLUPS: (study.N * steps) / wall / 1e6, kernelMLUPS: gpu > 0 ? (study.N * steps) / gpu / 1e6 : null,
      busyPercent: gpu > 0 ? (100 * gpu) / wall : null, recoveries: study.recoveries || 0,
    };
    report.appLoop.push(row);
    log(`app loop ${sampleId} ${(study.N / 1e6).toFixed(1)}M: ${row.wallMLUPS.toFixed(0)} MLUPS whole app, kernel ${row.kernelMLUPS?.toFixed(0)}, GPU busy ${row.busyPercent?.toFixed(0)}%`);
  } catch (err) {
    error(`app loop ${sampleId}`, err);
  } finally {
    study.dispose();
    cancelAnimationFrame(raf);
    LBMGPU.timing = false;
    FlowGPU.timing = false;
    await save();
  }
}

async function sustained(minutes, size = 0) {
  const big = report.ladder.filter((r) => !r.error).map((r) => r.cells);
  const n = size || (big.length ? big[Math.max(0, big.length - 2)] : 8e6);
  let sim = null, cells = 0;
  const samples = [];
  try {
    const t = await speedSim(n);
    sim = t.sim;
    cells = t.cells;
    const t0 = performance.now();
    let mark = t0, g = sim.gpuSeconds, s = sim.steps;
    let batch = Math.max(1, Math.min(400, Math.round(2e7 / cells)));
    while (performance.now() - t0 < minutes * 60e3) {
      const tb = performance.now();
      await sim.step(batch);
      batch = Math.max(1, Math.min(400, Math.round((batch * 40) / Math.max(1, performance.now() - tb))));
      if (performance.now() - mark >= 10e3) {
        const wall = (performance.now() - mark) / 1000, gpu = sim.gpuSeconds - g, steps = sim.steps - s;
        samples.push({ at: (performance.now() - t0) / 1000, steps, kernelMLUPS: gpu > 0 ? (cells * steps) / gpu / 1e6 : null, wallMLUPS: (cells * steps) / wall / 1e6 });
        if (params.get('debug')) log(`sustained sample: ${steps} steps in ${wall.toFixed(1)} s, GPU ${gpu.toFixed(1)} s, cells ${cells}, error ${sim.error}, lost ${sim.lost}`);
        mark = performance.now(); g = sim.gpuSeconds; s = sim.steps;
        state(`Sustained run: ${Math.round((mark - t0) / 1000)} of ${minutes * 60} s, ${samples.at(-1).wallMLUPS.toFixed(0)} MLUPS`);
        report.sustained = { cells, minutes, samples };
        await save();
      }
    }
    const first = samples.filter((x) => x.at <= 60), rest = samples.filter((x) => x.at > 60);
    const avg = (a, k) => a.reduce((acc, x) => acc + (x[k] || 0), 0) / (a.length || 1);
    report.sustained = { cells, minutes, samples, firstMinuteMLUPS: avg(first, 'wallMLUPS'), restMLUPS: avg(rest, 'wallMLUPS') };
    report.sustained.dropPercent = 100 * (1 - report.sustained.restMLUPS / report.sustained.firstMinuteMLUPS);
    log(`sustained ${(cells / 1e6).toFixed(0)}M: first minute ${report.sustained.firstMinuteMLUPS.toFixed(0)} MLUPS, after ${report.sustained.restMLUPS.toFixed(0)} (${report.sustained.dropPercent.toFixed(1)}% lower)`);
  } catch (err) {
    error('sustained', err);
  } finally {
    sim?.destroy();
    await save();
  }
}

/**
 * The GPU engine against the CPU reference on small grids: the same steps from rest must give the
 * same forces (to float rounding with 32-bit storage).
 */
async function check() {
  const { buildTunnel } = await import('../cfd/flowcase.js');
  const { FlowCPU } = await import('../cfd/flow-cpu.js');
  const { FlowGPU } = await import('../cfd/flow-gpu.js');
  report.check = [];
  for (const id of ['cube', 'sphere', 'ahmed25', 'naca0012']) {
    const c = caseById(id);
    const t = buildTunnel(c, { across: id === 'naca0012' ? 24 : id.startsWith('ahmed') ? 30 : 10, alpha: c.alphas ? 8 : undefined, legacy: false });
    for (const collision of ['bgk', 'rr']) {
      const params = { uLat: 0.08, nuLat: collision === 'rr' ? 1e-5 : 0.002, collision, wallModel: engineOptions.wallModel ?? true, wallMode: engineOptions.wallMode };
      const cpu = new FlowCPU(t.grid, params);
      let gpu32 = null, gpu16 = null;
      try {
        gpu32 = await FlowGPU.create(t.grid, { ...params, half: false });
        gpu16 = await FlowGPU.create(t.grid, { ...params, half: true });
        const steps = 300;
        cpu.step(steps);
        await gpu32.step(steps);
        await gpu16.step(steps);
        const a = cpu.takeForces(), b = await gpu32.takeForces(), h = await gpu16.takeForces();
        const rel = (x, y) => Math.abs(x - y) / Math.max(1e-12, Math.abs(x));
        const row = {
          id, collision, dims: t.grid.dims, records: t.grid.rec.count, steps,
          cpu: a.me, gpu32: b.me, gpu16: h.me,
          dragDiff32: rel(a.me[0], b.me[0]), dragDiff16: rel(a.me[0], h.me[0]), pressureDiff32: rel(a.pressure[0], b.pressure[0]),
        };
        report.check.push(row);
        log(`check ${id} ${collision} ${t.grid.dims.join('×')}: drag CPU ${a.me[0].toFixed(5)}, GPU 32-bit ${b.me[0].toFixed(5)} (${(100 * row.dragDiff32).toFixed(4)}%), 16-bit ${h.me[0].toFixed(5)} (${(100 * row.dragDiff16).toFixed(3)}%)`);
      } catch (err) {
        error(`check ${id} ${collision}`, err);
      } finally {
        gpu32?.destroy();
        gpu16?.destroy();
      }
      await save();
    }
  }
}

async function main() {
  window.addEventListener('error', (e) => error('page', e.error || e.message));
  window.addEventListener('unhandledrejection', (e) => error('promise', e.reason));
  log(`Parts Sim ${report.app.version} (${report.app.commit}), suite "${suite}"`);
  report.adapter = await adapterInfo();
  log(report.adapter.available ? `GPU: ${report.adapter.info.description || report.adapter.info.device || report.adapter.info.vendor} (${report.adapter.info.backend || 'backend not reported'})` : 'No WebGPU adapter: nothing to measure.');
  await save();
  if (report.adapter.available && !report.adapter.isFallback) {
    if (suite === 'backend') {
      // which backend a set of switches gives, and its speed
      await ladder(32e6);
    } else if (suite === 'check') {
      await check();
    } else if (suite === 'sustained') {
      await sustained(Number(params.get('minutes') || 1), Number(params.get('n') || 0));
    } else if (suite === 'tune') {
      await tune();
    } else if (suite === 'engine') {
      // the v1 engine on this GPU: agreement with the CPU reference, speed, kernel variants, the app
      await check();
      await ladder();
      await tune();
      await appLoop('ahmed', 1e6, 15);
      await appLoop('ahmed', 16e6, 20);
      await appLoop('ahmed', 64e6, 20);
    } else if (suite === 'kit') {
      await validation();
      await sustained(5);
    } else {
      if (suite !== 'speed') await validation();
      if (suite !== 'quick') {
        await ladder();
        await tune();
        await appLoop('ahmed', 1e6, 15);
        await appLoop('ahmed', 16e6, 20);
        await sustained(suite === 'speed' ? 2 : 5);
      }
    }
  }
  report.finished = new Date().toISOString();
  if (bridge?.readNative) {
    try { report.native = JSON.parse(await bridge.readNative()); } catch { /* no native results in this kit */ }
  }
  await save();
  state(`Done. ${report.errors.length ? `${report.errors.length} error(s). ` : ''}${bridge ? 'Report saved.' : ''}`);
  progress(1);
  log('Done.');
  bridge?.done();
}

main();
