// Browser smoke for saving and loading the airflow result (.psim RAIR section).
// Runs a small airflow job, exports the result, pushes it through the .psim arrays codec, resets the study,
// imports it as a frozen study and checks that the results card, legend, slice, streamlines and picture agree.
//
//   CHROME="$HOME/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing" \
//   node scripts/psim-airflow-smoke.mjs
// Environment: AIRFLOW_SAMPLE (ahmed|wing), AIRFLOW_ENGINE (cpu|auto), AIRFLOW_CELLS (default 50000), AIRFLOW_SAMPLES (default 12).
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const sampleId = process.env.AIRFLOW_SAMPLE || 'ahmed';
const engine = process.env.AIRFLOW_ENGINE || 'cpu';
const cells = Number(process.env.AIRFLOW_CELLS || 50000);
const minSamples = Number(process.env.AIRFLOW_SAMPLES || 12);

const server = await createServer({ server: { host: '127.0.0.1', port: 5199, strictPort: false, watch: null, headers: { 'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp' } }, logLevel: 'error' });
await server.listen(); 
const base = server.resolvedUrls.local[0];
await mkdir('test-artifacts', { recursive: true });
let browser, page;
const errors = [];
const wait = (fn, arg) => page.waitForFunction(fn, arg, { timeout: 280000 });

try {
  browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}),
    args: process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu'],
  });

  page = await browser.newPage({ viewport: { width: 1280, height: 860 } });
  page.on('pageerror', (err) => { errors.push(err.stack || err.message); console.error('Browser error:', err.stack || err.message); });
  console.log('browser started');
  await page.goto(`${base}?sample=${sampleId}`);
  await wait(() => !!window.partsSim?.part);
  await wait(() => document.querySelector('#busy').hidden);
  console.log('page loaded');
  await page.click('[data-tab="airflow"]');
  await page.selectOption('#engine-select', engine);
  await page.evaluate((n) => { const a = window.partsSim.airflow; a.cells = n; a.syncCells(); a.markDirty(); }, cells);
  await page.click('#btn-flow-run');
  console.log('run started');
  await wait(() => window.partsSim.airflow.study.steps > 0);
  console.log('stepping');
  await page.check('#chk-streamlines');
  await page.check('#chk-slice');
  await page.uncheck('#chk-particles'); // particles are random; compared separately
  await wait(() => window.partsSim.airflow.study.samples >= 1);
  const t0 = Date.now();
  const progress = setInterval(async () => {
    const p = await page.evaluate(() => { const s = window.partsSim.airflow.study; return `${s.engine} step ${s.steps} samples ${s.samples} ${s.developing ? 'developing' : 'averaging'} ${(s.mlups || 0).toFixed(1)} MLUPS`; }).catch(() => '');
    console.log(`  ${((Date.now() - t0) / 1000).toFixed(0)} s: ${p}`);
  }, 15000);
  await wait((n) => window.partsSim.airflow.study.samples >= n, minSamples);
  clearInterval(progress);
  const dims = await page.evaluate(() => window.partsSim.airflow.study.dims);
  console.log(`run: ${(Date.now() - t0) / 1000} s to ${minSamples} samples, grid ${dims.join('x')}`);

  // freeze the live state and take what a viewer would see
  await page.evaluate(async () => {
    const a = window.partsSim.airflow, s = a.study;
    s.pause();
    await new Promise((r) => setTimeout(r, 800));
    s.updateSlice();
    s.buildStreamlines();
    a.update();
    a.applyColoring(true);
    a.renderLegends();
  });
  const snapshot = () => page.evaluate(() => {
    const a = window.partsSim.airflow, s = a.study;
    const slice = s.group.getObjectByName('slice');
    const lines = s.group.getObjectByName('streamlines');
    return {
      kpis: document.querySelector('#flow-kpis').textContent,
      notes: document.querySelector('#flow-notes').textContent,
      state: document.querySelector('#flow-state').textContent,
      legend: document.querySelector('#legend').textContent,
      cpRange: a.cpRange ? [...a.cpRange] : null,
      sliceTexels: slice ? Array.from(slice.material.map.image.data) : null,
      streamVerts: lines ? lines.geometry.attributes.position.count : -1,
      results: JSON.parse(JSON.stringify({ ...s.results, force: s.results.force.toArray() })),
      cpRaw: Array.from(s.surfaceCp()),
      reynolds: [s.reynolds, s.reynoldsLength, s.simReynolds],
      frozen: s.frozen,
    };
  });
  const shot = async (path) => {
    const box = await page.locator('#viewport > canvas').boundingBox();
    return page.screenshot({ path, clip: box });
  };
  const live = await snapshot();
  const liveShot = await shot('test-artifacts/airflow-live.png');
  const liveState = await page.evaluate(() => window.partsSim.airflow.exportState());

  // export, push through the container codec, and measure
  const packed = await page.evaluate(async () => {
    const { encodeArrays, decodeArrays } = await import('/src/core/psim.js');
    const s = window.partsSim.airflow.study;
    const ex = s.exportResults();
    if (!ex) throw new Error('exportResults returned null after a run');
    const enc = encodeArrays(ex.meta, ex.arrays);
    const view = new DataView(enc.bytes.buffer, enc.bytes.byteOffset);
    const head = JSON.parse(new TextDecoder().decode(enc.bytes.subarray(4, 4 + view.getUint32(0, true))));
    const deflate = async (u8) => {
      const cs = new Blob([u8]).stream().pipeThrough(new CompressionStream('deflate-raw'));
      return (await new Response(cs).arrayBuffer()).byteLength;
    };
    const per = head.arrays.map((e) => ({ name: e.name, n: e.n, bytes: e.bytes, err: e.err }));
    const dec = decodeArrays(enc.bytes);
    window.__dec = dec;
    window.__raw = ex.arrays.map((a) => ({ name: a.name, data: a.data }));
    return { total: enc.bytes.length, deflated: await deflate(enc.bytes), per, headerBytes: 4 + view.getUint32(0, true), metaKeys: Object.keys(ex.meta) };
  });
  console.log('stored arrays:');
  for (const p of packed.per) console.log(`  ${p.name}: ${p.n} values, ${p.bytes} bytes, error bound ${p.err}`);
  console.log(`  header ${packed.headerBytes} bytes; section ${packed.total} bytes; deflated ${packed.deflated} bytes`);

  // reset, then import as a frozen study
  const loadMs = await page.evaluate(async () => {
    const a = window.partsSim.airflow;
    a.reset();
    if (a.study.results || a.study.fields) throw new Error('reset left results behind');
    const t = performance.now();
    await a.loadResults(window.__dec.meta, window.__dec.arrays, null);
    const ms = performance.now() - t;
    await new Promise((r) => setTimeout(r, 300));
    return ms;
  });
  console.log(`import (plan + solid cells + fields): ${loadMs.toFixed(0)} ms`);
  assert.equal(await page.evaluate(() => window.partsSim.airflow.study.frozen), true, 'study must be frozen');
  assert.equal(await page.evaluate(() => window.partsSim.airflow.study.ready), false, 'a frozen study has no solver');
  await page.evaluate(() => { const s = window.partsSim.airflow.study; s.updateSlice(); s.buildStreamlines(); window.partsSim.airflow.applyColoring(true); });
  const loaded = await snapshot();
  const loadedShot = await shot('test-artifacts/airflow-loaded.png');

  // 1. the results card
  assert.equal(loaded.kpis, live.kpis, 'results card numbers');
  assert.equal(loaded.notes, live.notes, 'results card notes (Reynolds numbers, wall model)');
  assert.equal(loaded.state, live.state, 'state line (engine, grid, steps, MLUPS, samples)');
  assert.deepEqual(loaded.results, live.results, 'results object');
  assert.deepEqual(loaded.reynolds, live.reynolds, 'Reynolds numbers');
  // 2. Cp legend within the stated q16 bound
  const cpErr = packed.per.find((p) => p.name === 'airflow.cp').err;
  assert.ok(live.cpRange && loaded.cpRange, 'Cp range present');
  for (let i = 0; i < 2; i++) assert.ok(Math.abs(loaded.cpRange[i] - live.cpRange[i]) <= cpErr * 1.0001 + 1e-6, `Cp legend ${i}: ${loaded.cpRange[i]} vs ${live.cpRange[i]} (bound ${cpErr})`);
  let worst = 0, nanSame = true;
  for (let i = 0; i < live.cpRaw.length; i++) {
    const a = live.cpRaw[i], b = loaded.cpRaw[i];
    if (Number.isNaN(a) !== Number.isNaN(b)) nanSame = false;
    else if (!Number.isNaN(a)) worst = Math.max(worst, Math.abs(a - b));
  }
  assert.ok(nanSame, 'NaN pattern of Cp');
  assert.ok(worst <= cpErr * 1.0001 + 1e-6, `Cp per vertex error ${worst} > ${cpErr}`);
  console.log(`Cp: legend ${live.cpRange.map((v) => v.toFixed(3))} -> ${loaded.cpRange.map((v) => v.toFixed(3))}, worst vertex error ${worst.toExponential(2)} (bound ${cpErr.toExponential(2)})`);
  // 3. slice and streamlines
  assert.ok(live.sliceTexels && loaded.sliceTexels && live.sliceTexels.length === loaded.sliceTexels.length, 'slice size');
  let sd = 0;
  for (let i = 0; i < live.sliceTexels.length; i++) sd += Math.abs(live.sliceTexels[i] - loaded.sliceTexels[i]);
  const sliceMad = sd / live.sliceTexels.length;
  console.log(`slice texture: ${live.sliceTexels.length / 4} texels, mean abs difference ${sliceMad.toFixed(3)} / 255`);
  console.log(`streamline vertices: live ${live.streamVerts}, loaded ${loaded.streamVerts}`);
  assert.ok(live.streamVerts > 0, 'live streamlines exist');
  assert.equal(loaded.streamVerts, live.streamVerts, 'streamline vertex count');
  // 4. picture
  const diff = await page.evaluate(async ([a, b]) => {
    const load = async (b64) => {
      const bmp = await createImageBitmap(await (await fetch(`data:image/png;base64,${b64}`)).blob());
      const c = new OffscreenCanvas(bmp.width, bmp.height);
      const ctx = c.getContext('2d');
      ctx.drawImage(bmp, 0, 0);
      return ctx.getImageData(0, 0, bmp.width, bmp.height).data;
    };
    const x = await load(a), y = await load(b);
    let sum = 0, n = 0, big = 0;
    for (let i = 0; i < x.length; i += 4) {
      const d = (Math.abs(x[i] - y[i]) + Math.abs(x[i + 1] - y[i + 1]) + Math.abs(x[i + 2] - y[i + 2])) / 3;
      sum += d; n++;
      if (d > 32) big++;
    }
    return { mad: sum / n, bigFraction: big / n };
  }, [liveShot.toString('base64'), loadedShot.toString('base64')]);
  console.log(`viewport picture: mean absolute pixel difference ${diff.mad.toFixed(3)} / 255, ${(100 * diff.bigFraction).toFixed(2)} % of pixels differ by more than 32`);
  assert.ok(diff.mad < 3, `pictures differ too much (${diff.mad})`);

  // 5. particles move in the loaded field; the wind load works from a frozen study
  await page.check('#chk-particles');
  await page.evaluate(() => { const s = window.partsSim.airflow.study; for (let i = 0; i < 20; i++) s.frame(0.03); });
  const moved = await page.evaluate(() => {
    const s = window.partsSim.airflow.study;
    return s.particles.visible && s.particles.geometry.attributes.position.array.some((v) => v !== 0);
  });
  assert.ok(moved, 'particles drawn from the loaded field');
  await page.evaluate(() => window.partsSim.airflow.useAsLoad());
  assert.ok(await page.evaluate(() => window.partsSim.structural.loads.some((l) => l.type === 'wind')), 'wind load from a frozen study');
  await page.click('[data-tab="airflow"]');

  // 6. state round trip, and Run airflow on a frozen study sets it up fresh
  const state2 = await page.evaluate((st) => { const a = window.partsSim.airflow; a.importState(st); return a.exportState(); }, liveState);
  assert.deepEqual(state2, liveState, 'exportState/importState round trip');
  await page.evaluate(() => window.partsSim.airflow.importState({}));
  await page.click('#btn-flow-run');
  await wait(() => window.partsSim.airflow.study.ready && window.partsSim.airflow.study.steps > 0);
  assert.equal(await page.evaluate(() => window.partsSim.airflow.study.frozen), false, 'Run airflow clears the frozen flag');
  await page.click('#btn-flow-run'); // pause
  console.log('PASS: airflow result save/restore');
  await writeFile('test-artifacts/psim-airflow-sizes.json', JSON.stringify({ sampleId, engine, dims, ...packed, sliceMad, diff }, null, 2));
  assert.deepEqual(errors, [], 'No uncaught browser errors');
} catch (err) {
  if (page) await page.screenshot({ path: 'test-artifacts/failure-airflow-psim.png' }).catch(() => {});
  throw err;
} finally {
  await browser?.close();
  await server.close();
}
