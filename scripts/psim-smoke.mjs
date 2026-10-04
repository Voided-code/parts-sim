// .psim in a real browser: run a bend test and a heat-transfer study, save the file, open it in a
// fresh page and check that the numbers and the picture match without solving; then the Save
// dialog, the file input, damaged files and Re-run.
//   npm run build && CHROME=... node scripts/psim-smoke.mjs
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import { preview } from 'vite';

const server = process.env.PARTS_SIM_URL ? null : await preview({ preview: { host: '127.0.0.1', port: 4175, strictPort: false } });
const base = process.env.PARTS_SIM_URL || server.resolvedUrls.local[0];
const out = 'test-artifacts/web';
await mkdir(out, { recursive: true });
let browser;
const errors = [];

async function newPage(url) {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on('pageerror', (e) => { errors.push(e.stack || e.message); console.error('Browser error:', e.stack || e.message); });
  await page.goto(url);
  await page.waitForFunction(() => !!window.partsSim, null, { timeout: 60000 });
  return page;
}
const idle = (page) => page.waitForFunction(() => document.querySelector('#busy').hidden, null, { timeout: 600000 });

/** Mean absolute difference (0-255) of the two PNGs, computed in the browser. */
async function pixelDiff(page, a, b) {
  return page.evaluate(async ([x, y]) => {
    const load = async (b64) => { const i = new Image(); i.src = `data:image/png;base64,${b64}`; await i.decode(); const c = document.createElement('canvas'); c.width = i.width; c.height = i.height; const g = c.getContext('2d'); g.drawImage(i, 0, 0); return g.getImageData(0, 0, i.width, i.height).data; };
    const [p, q] = [await load(x), await load(y)];
    let sum = 0;
    for (let k = 0; k < p.length; k += 4) sum += Math.abs(p[k] - q[k]) + Math.abs(p[k + 1] - q[k + 1]) + Math.abs(p[k + 2] - q[k + 2]);
    return sum / (p.length / 4) / 3;
  }, [a.toString('base64'), b.toString('base64')]);
}
const shotOf = (page) => page.locator('#viewport').screenshot();
const text = (page, sel) => page.locator(sel).innerText();

try {
  browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}),
    args: process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu'],
  });

  // ---- 1. a bend test and a thermal run
  const a = await newPage(`${base}?sample=beam`);
  await a.waitForFunction(() => !!window.partsSim?.part, null, { timeout: 60000 });
  await idle(a);
  await a.evaluate(() => { const r = document.querySelector('#res-range'); r.value = 32; r.dispatchEvent(new Event('input')); r.dispatchEvent(new Event('change')); });
  await a.click('#btn-run');
  await a.waitForTimeout(200);
  await idle(a);
  const kpis = await text(a, '#kpis');
  const legend = await text(a, '#legend');
  assert.match(kpis, /von Mises/);
  await a.waitForTimeout(500);
  const shotStatic = await shotOf(a);
  await writeFile(`${out}/psim-static-live.png`, shotStatic);

  // heat: hold the clamped end at 100 C and let the rest cool in air
  await a.evaluate(() => {
    const app = window.partsSim;
    app.setTab('thermal');
    app.thermal.items.push({ id: 900, type: 'temp', name: 'Hot end', value: 100, ambient: 20, patches: [{ tris: Int32Array.from(app.structural.fixtures[0].patches[0].tris), clip: null }] });
    app.thermal.renderList();
  });
  await a.click('#btn-th-run');
  await a.waitForTimeout(200);
  await idle(a);
  const thKpis = await text(a, '#th-kpis');
  assert.match(thKpis, /Hottest/);
  await a.evaluate(() => window.partsSim.setTab('structural'));
  await a.waitForTimeout(300);

  const t0 = Date.now();
  const bytesArr = await a.evaluate(async () => Array.from(await window.partsSim.psim.save({ results: ['static', 'thermal'] })));
  const saveMs = Date.now() - t0;
  const bytes = Buffer.from(bytesArr);
  await writeFile(`${out}/smoke.psim`, bytes);
  console.log(`saved ${bytes.length} bytes in ${saveMs} ms`, JSON.stringify(await a.evaluate(() => window.partsSim.psimTimings?.write)));
  console.log(execFileSync(process.execPath, ['scripts/psim.mjs', 'verify', `${out}/smoke.psim`]).toString().trim());
  await a.close();

  // ---- 2. a fresh page opens the file: same numbers, no solving
  const b = await newPage(base);
  const t1 = Date.now();
  await b.evaluate(async (arr) => { await window.partsSim.psim.open(Uint8Array.from(arr), 'smoke.psim'); }, bytesArr);
  await b.waitForFunction(() => !document.querySelector('#file-banner').hidden, null, { timeout: 30000 });
  const openMs = Date.now() - t1;
  console.log(`opened in ${openMs} ms`, JSON.stringify(await b.evaluate(() => window.partsSim.psimTimings?.open)));
  assert.match(await text(b, '#file-banner'), /Loaded from file, not computed here/);
  assert.equal(await b.evaluate(() => window.partsSim.part.name), 'Cantilever beam');
  assert.equal(await text(b, '#kpis'), kpis, 'the results card shows the saved numbers');
  assert.equal(await text(b, '#legend'), legend, 'the legend matches');
  assert.equal(await b.evaluate(() => window.partsSim.structural.fixtures.length), 1);
  assert.equal(await b.evaluate(() => window.partsSim.structural.loads.length), 1);
  await b.waitForTimeout(500);
  const shotLoaded = await shotOf(b);
  await writeFile(`${out}/psim-static-loaded.png`, shotLoaded);
  const diff = await pixelDiff(b, shotStatic, shotLoaded);
  console.log(`picture difference (mean abs, 0-255): ${diff.toFixed(2)}`);
  assert.ok(diff < 6, `the loaded picture differs from the live one (${diff})`);

  // thermal result on the thermal tab
  await b.evaluate(() => window.partsSim.setTab('thermal'));
  await b.waitForTimeout(300);
  assert.equal(await text(b, '#th-kpis'), thKpis, 'the thermal numbers match');

  // the info panel
  await b.click('#file-banner button:has-text("File info")');
  const info = await text(b, '#info-dialog');
  assert.match(info, /Sections/);
  assert.match(info, /Largest error/);
  await b.screenshot({ path: `${out}/psim-info.png` });
  await b.keyboard.press('Escape');

  // ---- 3. the Save dialog and a download
  await b.evaluate(() => window.partsSim.setTab('structural'));
  await b.click('#btn-save');
  await b.waitForSelector('#save-body input[type=checkbox]');
  const dlgText = await text(b, '#save-dialog');
  assert.match(dlgText, /Geometry, compact/);
  assert.match(dlgText, /About [\d.]+ (KB|MB)/);
  await b.screenshot({ path: `${out}/psim-save-dialog.png` });
  await b.evaluate(() => { delete window.showSaveFilePicker; }); // headless Chrome would wait on the picker
  const [download] = await Promise.all([b.waitForEvent('download'), b.click('#save-body button:has-text("Save")')]);
  const saved = await readFile(await download.path());
  assert.equal(download.suggestedFilename(), 'Cantilever beam.psim');
  assert.ok(saved.length > 1000);
  console.log(`dialog download: ${saved.length} bytes`);

  // ---- 4. the file input, a damaged file, and Re-run
  const c = await newPage(base);
  await c.setInputFiles('#file-input', { name: 'from-input.psim', mimeType: 'application/x-parts-sim', buffer: saved });
  await c.waitForFunction(() => !document.querySelector('#file-banner').hidden, null, { timeout: 30000 });
  assert.equal(await text(c, '#kpis'), kpis);
  const cut = saved.subarray(0, Math.floor(saved.length / 2));
  await c.setInputFiles('#file-input', { name: 'cut.psim', mimeType: 'application/x-parts-sim', buffer: cut });
  await c.waitForFunction(() => /cut\.psim/.test(document.querySelector('#status').textContent), null, { timeout: 30000 });
  assert.match(await text(c, '#status'), /damaged|cut short/);
  assert.equal(await c.evaluate(() => window.partsSim.part.name), 'Cantilever beam', 'a bad file leaves the open part alone');
  await c.setInputFiles('#file-input', { name: 'from-input.psim', mimeType: 'application/x-parts-sim', buffer: saved });
  await c.waitForFunction(() => !document.querySelector('#file-banner').hidden);
  await c.click('#file-banner button:has-text("Re-run")');
  await c.waitForTimeout(300);
  await idle(c);
  await c.waitForFunction(() => /Re-run finished/.test(document.querySelector('#status').textContent), null, { timeout: 60000 });
  console.log(`re-run: ${await text(c, '#status')}`);
  assert.equal(await c.locator('#file-banner').isHidden(), true);
  assert.deepEqual(errors, []);
  console.log('psim smoke passed');
} finally {
  await browser?.close();
  await server?.close();
}
