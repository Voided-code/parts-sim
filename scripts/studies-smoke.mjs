// Runs every study once in a real browser (Chromium with WebGPU) on the built app and saves a
// screenshot of each result: npm run build && node scripts/studies-smoke.mjs
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { preview } from 'vite';

const server = process.env.PARTS_SIM_URL ? null : await preview({ preview: { host: '127.0.0.1', port: 4174, strictPort: false } });
const base = process.env.PARTS_SIM_URL || server.resolvedUrls.local[0];
const only = process.argv.slice(2);
await mkdir('test-artifacts', { recursive: true });
let browser, page, shot = 0;
const errors = [];
const idle = () => page.waitForFunction(() => document.querySelector('#busy').hidden, null, { timeout: 600000 });
const status = () => page.textContent('#status');
const statusClass = () => page.locator('#status').getAttribute('class');

async function study(id, setup = null, check = null) {
  if (only.length && !only.includes(id)) return;
  const t0 = performance.now();
  await page.evaluate((s) => { window.partsSim.setTab('structural'); window.partsSim.structural.setStudy(s); }, id);
  if (setup) await page.evaluate(setup);
  await page.click('#btn-run');
  await page.waitForTimeout(100);
  await idle();
  const st = await status();
  const cls = await statusClass();
  console.log(`${id.padEnd(10)} ${((performance.now() - t0) / 1000).toFixed(1).padStart(5)} s  ${cls ? `[${cls}] ` : ''}${st}`);
  assert.ok(!cls?.includes('error') || id === 'drop' || id === 'buckling', `${id}: ${st}`);
  if (id !== 'static') assert.equal(await page.locator('#study-card').isHidden(), false, `${id}: results card`);
  if (check) assert.ok(await page.evaluate(check), `${id}: result check`);
  await page.waitForTimeout(400);
  await page.evaluate(() => document.querySelector('#study-card:not([hidden]), #results-card:not([hidden])')?.scrollIntoView({ block: 'start' }));
  await page.screenshot({ path: `test-artifacts/study-${++shot}-${id}.png` });
}

try {
  browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}),
    args: process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu'],
  });
  page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.on('pageerror', (err) => { errors.push(err.stack || err.message); console.error('Browser error:', err.stack || err.message); });
  page.on('console', (m) => { if (m.type() === 'error') console.error('console:', m.text()); });
  await page.goto(`${base}?sample=beam`);
  await page.waitForFunction(() => !!window.partsSim?.part, null, { timeout: 60000 });
  await idle();

  await study('static');
  await study('nonlinear', null, () => window.partsSim.structural.studies.nonlinear.result.steps.length > 0);
  await study('nonlinear', () => {
    const s = window.partsSim.structural.studies.nonlinear;
    s.opts.mode = 'failure';
    window.partsSim.structural.renderStudyOptions();
  }, () => ['collapse', 'rupture', 'large deformation', 'crack', 'step limit'].includes(window.partsSim.structural.studies.nonlinear.result.reason));
  await study('modal', null, () => {
    const r = window.partsSim.structural.studies.modal.result;
    // 200 x 20 x 10 mm steel cantilever: first bending ~ 203 Hz (weak axis)
    return r.modes.length === 5 && r.modes[0].freq > 150 && r.modes[0].freq < 260;
  });
  await study('buckling', null, () => window.partsSim.structural.studies.buckling.result.modes.length > 0);
  await study('fatigue', null, () => window.partsSim.structural.studies.fatigue.result.fat.minLife > 0);
  await study('drop', null, () => window.partsSim.structural.studies.drop.result.peak > 0);
  await study('dynamic', null, () => window.partsSim.structural.studies.dynamic.result.curve.length > 100);
  await study('dynamic', () => {
    const s = window.partsSim.structural.studies.dynamic;
    s.opts.type = 'shock';
    s.opts.source = 'base';
    window.partsSim.structural.renderStudyOptions();
  }, () => window.partsSim.structural.studies.dynamic.result.tcurve.length > 10);
  await study('optimize', () => { window.partsSim.structural.studies.optimize.opts.iters = 12; }, () => !!window.partsSim.structural.studies.optimize.result.shape);
  await study('optimize', () => {
    const s = window.partsSim.structural.studies.optimize;
    s.opts.goal = 'sizing';
    window.partsSim.structural.renderStudyOptions();
  }, () => window.partsSim.structural.studies.optimize.result.rows.some((r) => r.feasible));

  if (!only.length || only.includes('thermal')) {
    const t0 = performance.now();
    await page.evaluate(() => {
      const a = window.partsSim;
      a.setTab('thermal');
      const s = a.structural;
      const fix = s.fixtures[0].patches, tip = s.loads[0].patches;
      a.thermal.items = [
        { id: 901, type: 'temp', name: 'Clamp at 20 °C', patches: fix.map((p) => ({ ...p })), value: 20 },
        { id: 902, type: 'heat', name: 'Heater', patches: tip.map((p) => ({ ...p })), value: 5 },
      ];
      a.thermal.renderList();
    });
    await page.click('#btn-th-run');
    await page.waitForTimeout(100);
    await idle();
    console.log(`thermal    ${((performance.now() - t0) / 1000).toFixed(1).padStart(5)} s  ${await status()}`);
    assert.equal(await page.locator('#th-results').isHidden(), false);
    await page.screenshot({ path: 'test-artifacts/study-thermal.png' });
    await page.click('#th-mode [data-mode="transient"]');
    await page.fill('#th-duration', '300');
    await page.click('#btn-th-run');
    await page.waitForTimeout(100);
    await idle();
    console.log(`thermal-t  ${await status()}`);
    await page.screenshot({ path: 'test-artifacts/study-thermal-b.png' });
  }
  assert.deepEqual(errors, []);
  console.log('All studies ran.');
} finally {
  await browser?.close();
  await server?.close();
}
