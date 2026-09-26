import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { chromium } from 'playwright';
import { preview } from 'vite';
import { BoxGeometry, SphereGeometry } from 'three';

// Run against the production build by default; PARTS_SIM_URL can target a dev server.
const server = process.env.PARTS_SIM_URL ? null : await preview({ preview: { host: '127.0.0.1', port: 4173, strictPort: false } });
const base = process.env.PARTS_SIM_URL || server.resolvedUrls.local[0];
await mkdir('test-artifacts', { recursive: true });
let browser;
let page;
const errors = [];
const idle = () => page.waitForFunction(() => document.querySelector('#busy').hidden, null, { timeout: 180000 });
const wait = (fn) => page.waitForFunction(fn, null, { timeout: 180000 });
const upload = async (name, buffer) => {
  await page.setInputFiles('#file-input', { name, mimeType: 'application/octet-stream', buffer });
  await idle();
  assert.ok(!(await page.locator('#status').getAttribute('class')).includes('error'), await page.textContent('#status'));
};
const sample = async (id) => {
  await page.goto(`${base}?sample=${id}`);
  await wait(() => !!window.partsSim?.part);
  await idle();
};
const resolution = async (n) => page.locator('#res-range').evaluate((el, n) => {
  el.value = n;
  el.dispatchEvent(new Event('input'));
  el.dispatchEvent(new Event('change'));
}, n);
function triangles(geometry) {
  const flat = geometry.toNonIndexed();
  const positions = Float32Array.from(flat.attributes.position.array);
  flat.dispose();
  geometry.dispose();
  return positions;
}
function stl(positions) {
  const n = positions.length / 9;
  const data = Buffer.alloc(84 + n * 50);
  data.writeUInt32LE(n, 80);
  for (let t = 0; t < n; t++) for (let i = 0; i < 9; i++) data.writeFloatLE(positions[9 * t + i], 84 + 50 * t + 12 + 4 * i);
  return data;
}

try {
  browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}),
    args: process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu'],
  });
  page = await browser.newPage({ viewport: { width: 1440, height: 960 } });
  page.on('pageerror', (err) => { errors.push(err.stack || err.message); console.error('Browser error:', err.stack || err.message); });
  await page.goto(base);
  await wait(() => !!window.partsSim);
  await page.click('#btn-help');
  assert.equal(await page.locator('#help').evaluate((el) => el.open), true);
  await page.getByRole('button', { name: 'Got it' }).click();

  const box = triangles(new BoxGeometry(100, 10, 10));
  await upload('beam.stl', stl(box));
  assert.equal(await page.evaluate(() => window.partsSim.part.faceCount), 6);
  const obj = [];
  for (let i = 0; i < box.length; i += 3) obj.push(`v ${box[i]} ${box[i + 1]} ${box[i + 2]}`);
  for (let i = 0; i < box.length / 3; i += 3) obj.push(`f ${i + 1} ${i + 2} ${i + 3}`);
  await upload('beam.obj', Buffer.from(obj.join('\n')));
  await page.click('[data-tab="structural"]');
  // Select opposite end faces with actual pointer clicks and verify rest-space hits.
  await page.evaluate(() => window.partsSim.viewer.setView('left'));
  await page.click('#btn-add-fixture');
  const canvas = page.locator('#viewport > canvas');
  const bounds = await canvas.boundingBox();
  await page.mouse.click(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.click('#picker-done');
  assert.ok(await page.evaluate(() => {
    const a = window.partsSim;
    return a.structural.fixtures[0].patches[0].tris.every((t) => a.part.triNormal[3 * t] < -0.99);
  }), 'Support click must select the left end face');
  await page.click('[data-view="right"]');
  await page.click('#btn-add-force');
  await page.mouse.click(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.click('#picker-done');
  assert.ok(await page.evaluate(() => {
    const a = window.partsSim;
    return a.structural.loads[0].patches[0].tris.every((t) => a.part.triNormal[3 * t] > 0.99);
  }), 'Load click must select the right end face');
  await page.locator('#load-editor button').filter({ hasText: /^−Y$/ }).click();
  await page.locator('#load-editor .mag input').fill('100');
  await page.locator('#load-editor .mag input').dispatchEvent('change');
  await page.click('[data-view="iso"]');
  await resolution(32);
  await page.click('#btn-run');
  await wait(() => !!window.partsSim.structural.result);
  await idle();
  assert.ok(await page.evaluate(() => {
    const r = window.partsSim.structural.result;
    return r.maxVM > 0 && Number.isFinite(r.maxVM) && r.maxDisp > 0 && Number.isFinite(r.maxDisp);
  }));
  for (const plot of ['disp', 'fos', 'p1', 'p3', 'vm']) await page.selectOption('#plot-select', plot);
  await page.check('#chk-animate');
  await page.uncheck('#chk-animate');
  await page.screenshot({ path: 'test-artifacts/bend.png' });
  const download = page.waitForEvent('download');
  await page.click('#btn-shot');
  assert.ok((await download).suggestedFilename().endsWith('.png'));
  console.log('PASS: STL/OBJ import, face selection, force direction, bend plots and screenshot');

  await sample('beam');
  await resolution(24);
  await page.click('#btn-break');
  await wait(() => window.partsSim.structural.brk?.done);
  await idle();
  assert.ok(await page.evaluate(() => window.partsSim.structural.brk.steps.length > 0));
  await page.click('#btn-break-play');
  await page.click('#btn-break-play');
  await page.screenshot({ path: 'test-artifacts/break.png' });
  console.log('PASS: progressive break illustration and playback');

  await page.setInputFiles('#file-input', 'node_modules/occt-import-js/test/testfiles/simple-basic-cube/cube.stp');
  await idle();
  assert.ok(await page.evaluate(() => window.partsSim.part.name === 'cube' && window.partsSim.part.brepFaces));
  await page.setInputFiles('#file-input', { name: 'invalid.obj', mimeType: 'text/plain', buffer: Buffer.from('v NaN 0 0\nf 1 1 1') });
  await idle();
  assert.equal(await page.evaluate(() => window.partsSim.part.name), 'cube');
  assert.ok((await page.locator('#status').getAttribute('class')).includes('error'));
  console.log('PASS: STEP/WASM import and failed-import recovery');

  await upload('sphere.stl', stl(triangles(new SphereGeometry(25, 24, 16))));
  await page.click('[data-tab="airflow"]');
  await page.selectOption('#engine-select', 'cpu');
  await page.fill('#flow-cells', '0'); // smallest grid
  assert.equal(await page.evaluate(() => window.partsSim.airflow.cells), 50e3);
  await page.click('#btn-flow-run');
  await wait(() => window.partsSim.airflow.study.steps > 10);
  await idle();
  assert.equal(await page.evaluate(() => window.partsSim.airflow.study.engine), 'CPU');
  await page.click('#btn-flow-run');
  assert.equal(await page.evaluate(() => window.partsSim.airflow.study.running), false);
  await page.click('#btn-flow-run');
  assert.equal(await page.evaluate(() => window.partsSim.airflow.study.running), true);
  await page.click('#btn-flow-reset');
  await wait(() => window.partsSim.airflow.study.ready && !window.partsSim.airflow.study.initializing);
  assert.equal(await page.evaluate(() => window.partsSim.airflow.study.samples), 0);
  console.log('PASS: CPU airflow, pause/resume and reset');

  const gpu = await page.evaluate(async () => !!(navigator.gpu && await navigator.gpu.requestAdapter()));
  if (gpu) {
    await page.selectOption('#engine-select', 'auto');
    // the GPU's grid-size range goes past the old "High" (2.2 M cells)
    assert.ok(await page.evaluate(() => window.partsSim.airflow.capacity().max > 2.2e6));
    await page.click('#btn-flow-run');
    await wait(() => window.partsSim.airflow.study.samples > 2);
    await idle();
    assert.equal(await page.evaluate(() => window.partsSim.airflow.study.engine), 'WebGPU');
    const aero = await page.evaluate(() => window.partsSim.airflow.study.results);
    assert.ok(Number.isFinite(aero.cd) && aero.cd > 0, JSON.stringify(aero));
    await page.check('#chk-streamlines');
    await page.check('#chk-slice');
    await page.screenshot({ path: 'test-artifacts/airflow.png' });
    await page.click('#btn-wind-load');
    assert.ok(await page.evaluate(() => window.partsSim.structural.loads.some((l) => l.type === 'wind')));
    console.log(`PASS: WebGPU airflow, display layers and pressure transfer (Cd ${aero.cd.toFixed(3)}, smoke check only)`);
  } else console.log('SKIP: WebGPU unavailable; CPU airflow verified');

  await sample('bracket');
  await page.setViewportSize({ width: 390, height: 844 });
  await page.click('[data-view="iso"]');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
  await page.screenshot({ path: 'test-artifacts/mobile.png', fullPage: true });
  assert.deepEqual(errors, [], 'No uncaught browser errors');
  console.log('PASS: narrow layout and no uncaught browser errors');
} catch (err) {
  if (page) {
    console.error('Status:', await page.locator('#status').textContent().catch(() => 'unavailable'));
    await page.screenshot({ path: 'test-artifacts/failure.png' }).catch(() => {});
  }
  throw err;
} finally {
  await browser?.close();
  await new Promise((resolve) => server ? server.httpServer.close(resolve) : resolve());
}
