// The app path for the other results: a frequency study and an airflow run are saved with
// app.psim.save(), a fresh page opens the bytes, and the cards, the legend and the picture agree.
//   npm run build && CHROME=... node scripts/psim-app-smoke.mjs
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import { preview } from 'vite';

const server = process.env.PARTS_SIM_URL ? null : await preview({ preview: { host: '127.0.0.1', port: 4176, strictPort: false } });
const base = process.env.PARTS_SIM_URL || server.resolvedUrls.local[0];
const out = 'test-artifacts/web';
await mkdir(out, { recursive: true });
let browser;
const errors = [];
const idle = (p) => p.waitForFunction(() => document.querySelector('#busy').hidden, null, { timeout: 600000 });
const text = (p, sel) => p.locator(sel).innerText();

async function open(url) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', (e) => { errors.push(e.stack || e.message); console.error('Browser error:', e.stack || e.message); });
  await page.goto(url);
  await page.waitForFunction(() => !!window.partsSim, null, { timeout: 60000 });
  return page;
}
async function pixelDiff(page, a, b) {
  return page.evaluate(async ([x, y]) => {
    const load = async (b64) => { const i = new Image(); i.src = `data:image/png;base64,${b64}`; await i.decode(); const c = document.createElement('canvas'); c.width = i.width; c.height = i.height; const g = c.getContext('2d'); g.drawImage(i, 0, 0); return g.getImageData(0, 0, i.width, i.height).data; };
    const [p, q] = [await load(x), await load(y)];
    let sum = 0;
    for (let k = 0; k < p.length; k += 4) sum += Math.abs(p[k] - q[k]) + Math.abs(p[k + 1] - q[k + 1]) + Math.abs(p[k + 2] - q[k + 2]);
    return sum / (p.length / 4) / 3;
  }, [a.toString('base64'), b.toString('base64')]);
}
const bytesOf = (page) => page.evaluate(async () => Array.from(await window.partsSim.psim.save()));

try {
  browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}),
    args: process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu'],
  });

  // ---- frequency study on the beam
  let a = await open(`${base}?sample=beam`);
  await a.waitForFunction(() => !!window.partsSim?.part, null, { timeout: 60000 });
  await idle(a);
  await a.evaluate(() => { const r = document.querySelector('#res-range'); r.value = 24; r.dispatchEvent(new Event('input')); r.dispatchEvent(new Event('change')); window.partsSim.structural.setStudy('modal'); });
  await a.click('#btn-run');
  await a.waitForTimeout(300);
  await idle(a);
  await a.waitForTimeout(600);
  const card = await text(a, '#study-card');
  const legend = await text(a, '#legend');
  assert.match(card, /Hz/);
  const shotLive = await a.locator('#viewport').screenshot();
  const modal = Buffer.from(await bytesOf(a));
  await writeFile(`${out}/app-modal.psim`, modal);
  console.log(`frequency file: ${modal.length} bytes`);
  await a.close();
  let b = await open(base);
  await b.evaluate(async (arr) => { await window.partsSim.psim.open(Uint8Array.from(arr), 'modal.psim'); }, [...modal]);
  await b.waitForFunction(() => !document.querySelector('#file-banner').hidden, null, { timeout: 30000 });
  await b.waitForTimeout(600);
  assert.equal(await b.evaluate(() => window.partsSim.structural.study), 'modal');
  assert.equal(await text(b, '#study-card'), card, 'frequency card');
  assert.equal(await text(b, '#legend'), legend, 'frequency legend');
  const d1 = await pixelDiff(b, shotLive, await b.locator('#viewport').screenshot());
  console.log(`frequency picture difference ${d1.toFixed(3)}/255`);
  assert.ok(d1 < 6);
  await b.close();

  // ---- the series studies (steps and frames are predicted from the one before): break test and drop test
  for (const [id, sample, study, run, card, kpis] of [['break', 'lbracket', 'static', '#btn-break', '#break-card', '#break-kpis'], ['drop', 'beam', 'drop', '#btn-run', '#study-card', '#study-card']]) {
    const x = await open(`${base}?sample=${sample}`);
    await x.waitForFunction(() => !!window.partsSim?.part, null, { timeout: 60000 });
    await idle(x);
    await x.evaluate(([n, st]) => { const r = document.querySelector('#res-range'); r.value = n; r.dispatchEvent(new Event('input')); r.dispatchEvent(new Event('change')); window.partsSim.structural.setStudy(st); }, [id === 'break' ? 28 : 20, study]);
    await x.click(run);
    await x.waitForTimeout(300);
    await idle(x);
    await x.waitForTimeout(700);
    const text0 = await text(x, kpis);
    const leg0 = await text(x, '#legend');
    const shot0 = await x.locator('#viewport').screenshot();
    const bytesX = Buffer.from(await x.evaluate(async (all) => Array.from(await window.partsSim.psim.save(all ? {} : {})), true));
    await writeFile(`${out}/app-${id}.psim`, bytesX);
    const steps = await x.evaluate((i) => (i === 'break' ? window.partsSim.structural.brk.steps.length : window.partsSim.structural.studies.drop.result.frames.length), id);
    console.log(`${id} file: ${bytesX.length} bytes (${steps} ${id === 'break' ? 'steps' : 'frames'})`);
    await x.close();
    const y = await open(base);
    await y.evaluate(async (arr) => { await window.partsSim.psim.open(Uint8Array.from(arr), `${id}.psim`); }, [...bytesX]);
    await y.waitForFunction(() => !document.querySelector('#file-banner').hidden, null, { timeout: 60000 });
    await y.waitForTimeout(900);
    assert.equal(await text(y, kpis), text0, `${id} card`);
    const d = await pixelDiff(y, shot0, await y.locator('#viewport').screenshot());
    console.log(`${id} picture difference ${d.toFixed(3)}/255 (legend ${leg0 === (await text(y, '#legend')) ? 'identical' : 'differs'})`);
    assert.ok(d < 8, `${id} picture`);
    await y.close();
  }

  // ---- airflow on the Ahmed body, CPU engine, 50k cells
  a = await open(`${base}?sample=ahmed`);
  await a.waitForFunction(() => !!window.partsSim?.part, null, { timeout: 60000 });
  await idle(a);
  await a.click('[data-tab="airflow"]');
  await a.selectOption('#engine-select', 'cpu');
  await a.evaluate(() => { const f = window.partsSim.airflow; f.cells = 50000; f.syncCells(); f.markDirty(); });
  await a.click('#btn-flow-run');
  await a.waitForFunction(() => window.partsSim.airflow.study.samples >= 12, null, { timeout: 280000 });
  await a.check('#chk-slice');
  await a.uncheck('#chk-particles');
  await a.evaluate(async () => { window.partsSim.airflow.study.pause(); await new Promise((r) => setTimeout(r, 800)); });
  const kpis = await text(a, '#flow-kpis');
  const notes = await text(a, '#flow-notes');
  const flowLegend = await text(a, '#legend');
  const shotFlow = await a.locator('#viewport').screenshot();
  const air = Buffer.from(await bytesOf(a));
  await writeFile(`${out}/app-airflow.psim`, air);
  console.log(`airflow file: ${air.length} bytes`);
  await a.close();
  b = await open(base);
  const t0 = Date.now();
  await b.evaluate(async (arr) => { await window.partsSim.psim.open(Uint8Array.from(arr), 'airflow.psim'); }, [...air]);
  await b.waitForFunction(() => !document.querySelector('#file-banner').hidden, null, { timeout: 60000 });
  console.log(`airflow open: ${Date.now() - t0} ms (includes copying the file in)`);
  await b.waitForTimeout(800);
  assert.equal(await b.evaluate(() => window.partsSim.tab), 'airflow');
  assert.equal(await b.evaluate(() => window.partsSim.airflow.study.frozen), true);
  assert.equal(await text(b, '#flow-kpis'), kpis, 'airflow results card');
  assert.equal(await text(b, '#flow-notes'), notes, 'airflow notes');
  assert.equal(await text(b, '#legend'), flowLegend, 'airflow legend');
  const d2 = await pixelDiff(b, shotFlow, await b.locator('#viewport').screenshot());
  console.log(`airflow picture difference ${d2.toFixed(3)}/255`);
  assert.ok(d2 < 8);
  await b.screenshot({ path: `${out}/psim-airflow-loaded.png` });
  assert.deepEqual(errors, []);
  console.log('psim app smoke passed');
} finally {
  await browser?.close();
  await server?.close();
}
