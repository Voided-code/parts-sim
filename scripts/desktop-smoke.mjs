// End-to-end check of the desktop app (Electron) with Playwright.
//
//   npm run build && npm run test:desktop
//   PARTS_SIM_APP="release/mac-arm64/Parts Sim.app/Contents/MacOS/Parts Sim" npm run test:desktop   # packaged app
//   ELECTRON_ARGS=--disable-gpu npm run test:desktop   # software rendering, like CI machines without a GPU
//
// Opens a synthetic SolidWorks assembly from the command line (so its part is read from the
// same folder through the shell), then drives menu commands, a STEP import (WebAssembly under
// the app's content security policy), a bend test and the airflow solver.
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, copyFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { _electron as electron } from 'playwright';
import { cubeAssemblyFiles } from '../test/fixtures/solidworks.js';

const root = path.resolve(import.meta.dirname, '..');
const dir = await mkdtemp(path.join(tmpdir(), 'parts-sim-'));
const { part, assembly } = cubeAssemblyFiles();
await writeFile(path.join(dir, 'Cube.SLDPRT'), part);
await writeFile(path.join(dir, 'robot.SLDASM'), assembly);
const step = path.join(dir, 'cube.stp');
await copyFile(path.join(root, 'node_modules/occt-import-js/test/testfiles/simple-basic-cube/cube.stp'), step);
await mkdir(path.join(root, 'test-artifacts'), { recursive: true });

const packaged = process.env.PARTS_SIM_APP;
const env = { ...process.env };
delete env.ELECTRON_RUN_AS_NODE; // leaked by editor terminals; would start Electron as plain Node
env.PARTS_SIM_USER_DATA = path.join(dir, 'profile'); // run beside (not inside) a copy the user has open
const extra = process.env.ELECTRON_ARGS?.split(' ').filter(Boolean) ?? [];
const app = await electron.launch({
  env,
  ...(packaged ? { executablePath: path.resolve(root, packaged), args: [...extra, path.join(dir, 'robot.SLDASM')] } : { args: [...extra, root, path.join(dir, 'robot.SLDASM')] }),
  cwd: root,
});
const errors = [];
const output = []; // Electron's own log, printed when a step fails
for (const stream of [app.process().stdout, app.process().stderr]) stream?.on('data', (d) => output.push(...String(d).split('\n').filter(Boolean)));
let page = null;
try {
  page = await app.firstWindow();
  page.on('pageerror', (e) => errors.push(e.stack || e.message));
  page.on('console', (m) => m.type() === 'error' && errors.push(m.text()));
  const idle = () => page.waitForFunction(() => document.querySelector('#busy').hidden, null, { timeout: 120000 });
  const command = (c) => app.evaluate(({ BrowserWindow }, c) => BrowserWindow.getAllWindows()[0].webContents.send('command', c), c);
  const open = (p) => app.evaluate(({ BrowserWindow }, p) => {
    const fs = process.getBuiltinModule('node:fs');
    BrowserWindow.getAllWindows()[0].webContents.send('open-files', [{ name: p.split(/[\\/]/).pop(), path: p, data: fs.readFileSync(p) }]);
  }, p);

  // 1. SolidWorks assembly from argv, part resolved from the same folder
  await page.waitForFunction(() => window.partsSim?.part, null, { timeout: 60000 });
  await idle();
  const asm = await page.evaluate(() => ({ info: window.partsSim.partInfo, size: window.partsSim.part.bbox.size.map(Math.round), desktop: !!window.partsSim.desktop, title: document.title, isolated: crossOriginIsolated }));
  assert.equal(asm.desktop, true);
  assert.equal(asm.isolated, true, 'cross-origin isolation (the solvers\' shared-memory threads)');
  assert.match(asm.info, /3 of 3 components/, asm.info);
  assert.deepEqual(asm.size, [140, 120, 20]);
  assert.match(asm.title, /robot — Parts Sim/);
  console.log('PASS: SolidWorks assembly opened from the command line, parts read from its folder');

  // 2. STEP (OpenCascade WebAssembly in a worker) under the app's CSP
  await open(step);
  await page.waitForFunction(() => /STEP/.test(window.partsSim.partInfo), null, { timeout: 60000 });
  await idle();
  assert.equal(await page.evaluate(() => window.partsSim.part.faceCount), 6);
  console.log('PASS: STEP import through the shell');

  // 3. Menu commands: sample, view, bend test
  await command('sample:bracket');
  await page.waitForFunction(() => window.partsSim.part?.name === 'Bolted mounting bracket', null, { timeout: 60000 });
  await idle();
  await command('view:top');
  await command('run:bend');
  await page.waitForFunction(() => window.partsSim.structural.result, null, { timeout: 120000 });
  await idle();
  const kpis = (await page.textContent('#kpis')).replace(/\s+/g, ' ');
  assert.match(kpis, /Max von Mises\s*\d/);
  await shot('desktop-bend');
  console.log(`PASS: menu commands and bend test (${kpis.slice(0, 60)}…)`);

  // 3b. The GPU structural solver (when an adapter exists) must agree with the CPU solver
  const first = await page.evaluate(() => {
    const r = window.partsSim.structural.result;
    return { vm: r.maxVM, d: r.maxDisp, gpu: /on the GPU/.test(document.querySelector('#status').textContent) };
  });
  await page.selectOption('#fea-engine', 'cpu');
  await command('run:bend');
  await page.waitForFunction(() => /on the CPU/.test(document.querySelector('#status').textContent), null, { timeout: 120000 });
  await idle();
  const cpu = await page.evaluate(() => ({ vm: window.partsSim.structural.result.maxVM, d: window.partsSim.structural.result.maxDisp }));
  assert.ok(Math.abs(first.vm / cpu.vm - 1) < 1e-4 && Math.abs(first.d / cpu.d - 1) < 1e-4, `GPU ${JSON.stringify(first)} vs CPU ${JSON.stringify(cpu)}`);
  await page.selectOption('#fea-engine', 'auto');
  console.log(`PASS: bend test ${first.gpu ? 'on the GPU matches the CPU solver' : 'fell back to the CPU (no WebGPU adapter)'}`);

  // 3c. Other studies through the Simulate menu: frequency, drop test (explicit GPU dynamics), thermal
  await command('study:modal');
  await command('run:study');
  await page.waitForFunction(() => window.partsSim.structural.studies.modal.result, null, { timeout: 120000 });
  await idle();
  const f1 = await page.evaluate(() => window.partsSim.structural.studies.modal.result.modes[0].freq);
  assert.ok(f1 > 0, `frequency ${f1}`);
  await command('study:drop');
  await command('run:study');
  await page.waitForFunction(() => window.partsSim.structural.studies.drop.result?.done, null, { timeout: 120000 });
  await idle();
  const drop = await page.evaluate(() => ({ peak: window.partsSim.structural.studies.drop.result.peak, engine: window.partsSim.structural.studies.drop.result.engine }));
  assert.ok(drop.peak > 0, JSON.stringify(drop));
  await page.evaluate(() => {
    const a = window.partsSim;
    a.setTab('thermal');
    a.thermal.items = [{ id: 1, type: 'temp', name: 'Hot bolts', patches: a.structural.fixtures[0].patches, value: 90 }];
    a.thermal.renderList();
  });
  await command('run:thermal');
  await page.waitForFunction(() => window.partsSim.thermal.result, null, { timeout: 120000 });
  await idle();
  const th = await page.evaluate(() => window.partsSim.thermal.result.max);
  assert.ok(Math.abs(th - 90) < 1e-3, `hottest ${th}`);
  await shot('desktop-thermal');
  await command('study:static');
  console.log(`PASS: frequency (${f1.toFixed(0)} Hz), drop test on the ${drop.engine} and heat transfer`);

  // 4. Airflow on the GPU (or the CPU fallback when no WebGPU adapter exists)
  await command('sample:ahmed');
  await page.waitForFunction(() => window.partsSim.part?.name === 'Ahmed body (car)', null, { timeout: 60000 });
  await idle();
  // grid-size slider: on the GPU, go past the old "High" preset (2.2 M cells)
  await page.waitForFunction(() => window.partsSim.airflow.gpu !== undefined, null, { timeout: 30000 });
  const target = await page.evaluate(() => {
    const a = window.partsSim.airflow, cap = a.capacity();
    const cells = a.usesGPU() ? Math.min(3e6, cap.max) : cap.default;
    return { cells, value: Math.round((1000 * Math.log(cells / cap.min)) / Math.log(cap.max / cap.min)) };
  });
  await page.evaluate((v) => {
    const el = document.querySelector('#flow-cells');
    el.value = String(v);
    el.dispatchEvent(new Event('input'));
  }, target.value);
  await command('run:airflow');
  const flowState = () => page.evaluate(() => {
    const s = window.partsSim.airflow.study;
    return { status: document.querySelector('#status')?.textContent, ready: s.ready, running: s.running, engine: s.engine, N: s.N, steps: s.steps, dims: s.dims, cd: s.results?.cd, building: window.partsSim.airflow.building };
  });
  await page.waitForFunction(() => window.partsSim.airflow.study.results, null, { timeout: 180000 }).catch(async (err) => {
    console.log('airflow state:', JSON.stringify(await flowState()));
    throw err;
  });
  await page.waitForTimeout(3000);
  // the drag comes from time-averaged pressures, which take longer to build on a slow CPU
  await page.waitForFunction(() => window.partsSim.airflow.study.results?.cd > 0, null, { timeout: 180000 }).catch(async (err) => {
    console.log('airflow state:', JSON.stringify(await flowState()));
    throw err;
  });
  const flow = await page.evaluate(() => ({ engine: window.partsSim.airflow.study.engine, cd: window.partsSim.airflow.study.results.cd, cells: window.partsSim.airflow.study.N }));
  assert.ok(flow.cells <= target.cells * 1.001 && flow.cells > 0.7 * target.cells, `grid ${flow.cells} for ${target.cells}`);
  assert.ok(Number.isFinite(flow.cd) && flow.cd > 0, JSON.stringify(flow));
  await shot('desktop-airflow');
  await command('run:airflow');
  console.log(`PASS: airflow on ${flow.engine}, ${(flow.cells / 1e6).toFixed(2)} M cells from the grid-size slider (Cd ${flow.cd.toFixed(2)}, smoke check only)`);

  assert.deepEqual(errors, []);
  console.log('PASS: no page errors');
} catch (err) {
  await report();
  throw err;
} finally {
  const proc = app.process();
  const closed = await Promise.race([app.close().then(() => true, () => false), new Promise((r) => setTimeout(r, 10000, false))]);
  if (!closed) proc.kill();
}

/**
 * Saves a screenshot for looking at later. It is not a check: on the 2-core CI runners Electron's
 * software rendering sometimes takes over half a minute for a frame, so a slow one only warns.
 */
async function shot(name) {
  try {
    await page.screenshot({ path: path.join(root, `test-artifacts/${name}.png`), timeout: 90000 });
  } catch (err) {
    console.log(`WARN: no ${name} screenshot (${err.message.split('\n')[0]})`);
  }
}

/** Prints what the app was doing when a step failed, and saves a screenshot next to the others. */
async function report() {
  const within = (p) => Promise.race([p, new Promise((_, no) => setTimeout(() => no(new Error('no answer within 10 s')), 10000))]);
  console.log(`--- page errors: ${errors.length ? `\n${errors.join('\n')}` : 'none'}`);
  console.log(`--- Electron output (last 40 lines):\n${output.slice(-40).join('\n')}`);
  if (!page) return;
  const state = await within(page.evaluate(() => {
    const a = window.partsSim, text = (s) => document.querySelector(s)?.textContent.trim();
    return { app: !!a, part: a?.part?.name, info: a?.partInfo, tab: a?.tab, status: text('#status'), busy: document.querySelector('#busy')?.hidden === false ? text('#busy-text') : null };
  })).catch((e) => e.message);
  console.log(`--- app state: ${JSON.stringify(state)}`);
  await within(page.screenshot({ path: path.join(root, 'test-artifacts/desktop-failure.png') })).catch(() => {});
}
