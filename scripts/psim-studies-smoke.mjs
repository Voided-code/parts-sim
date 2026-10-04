// Saves and restores the structural study results the way a .psim file does (src/core/psim.js arrays
// container): run a study, exportResult() -> encodeArrays -> decodeArrays -> clear() -> importResult() ->
// show(), and compare the KPIs, the legend and the viewport with the original run.
//   CHROME="$HOME/Library/Caches/ms-playwright/chromium-1223/chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing" \
//   lockf -k test-artifacts/cpu.lock node scripts/psim-studies-smoke.mjs [study ids]
import assert from 'node:assert/strict';
import { chromium } from 'playwright';
import { spawn } from 'node:child_process';

// The dev server (not the build) so the page can import /src/core/psim.js. It runs in its own process:
// started in this one, vite's file watcher stalls the browser launch.
let server = null;
let base = process.env.PARTS_SIM_URL;
if (!base) {
  server = spawn(process.execPath, ['node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '4179'], { stdio: ['ignore', 'pipe', 'inherit'] });
  base = await new Promise((resolve, reject) => {
    let out = '';
    server.stdout.on('data', (d) => { out += String(d).replace(/\u001b\[[0-9;]*m/g, ''); const m = out.match(/http:\/\/127\.0\.0\.1:\d+\//); if (m) resolve(m[0]); });
    server.on('exit', (code) => reject(new Error(`vite exited with ${code}`)));
  });
}
const only = process.argv.slice(2);
let browser, page;
const errors = [];
const idle = () => page.waitForFunction(() => document.querySelector('#busy').hidden, null, { timeout: 300000 });

/** Mean absolute difference of two PNGs in 0..255 per colour channel, computed in the page. */
async function pixelDiff(a, b) {
  return page.evaluate(async ([x, y]) => {
    const load = (b64) => new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = `data:image/png;base64,${b64}`; });
    const px = async (b64) => {
      const i = await load(b64), c = document.createElement('canvas');
      c.width = i.width; c.height = i.height;
      const g = c.getContext('2d');
      g.drawImage(i, 0, 0);
      return g.getImageData(0, 0, c.width, c.height).data;
    };
    const p = await px(x), q = await px(y);
    let s = 0, n = 0;
    for (let k = 0; k < p.length; k += 4) for (let c = 0; c < 3; c++) { s += Math.abs(p[k + c] - q[k + c]); n++; }
    return s / n;
  }, [a.toString('base64'), b.toString('base64')]);
}

/** The text the UI shows for a study: KPIs, legend, list rows, table rows. */
const snapshot = () => page.evaluate(() => {
  const t = (sel) => document.querySelector(sel)?.innerText ?? '';
  return {
    kpis: t('#study-kpis'), legend: t('#legend'), title: t('#study-title'),
    rows: document.querySelectorAll('#study-body li, #study-body tbody tr').length,
    status: t('#status'),
  };
});

const numbers = (s) => (s.match(/-?\d+(?:\.\d+)?(?:e[+-]?\d+)?/gi) || []).map(Number);

/** Same words and the same numbers within `tol` (relative to the larger, or 1e-9 absolute). */
function sameText(a, b, tol) {
  if (a.replace(/[-\d.e+]+/gi, '#') !== b.replace(/[-\d.e+]+/gi, '#')) return false;
  const x = numbers(a), y = numbers(b);
  return x.length === y.length && x.every((v, i) => Math.abs(v - y[i]) <= tol * Math.max(Math.abs(v), Math.abs(y[i])) + 1e-9);
}

/**
 * One check. `setup` runs in the page before the run (options), `quant` is the relative tolerance for the numbers
 * the UI shows (0 = must be identical text).
 */
async function check(id, label, { setup = null, quant = 0, results = null, frozen = null } = {}) {
  if (only.length && !only.includes(id) && !only.includes(label)) return;
  const t0 = performance.now();
  await page.evaluate(([sid, fn]) => {
    const a = window.partsSim, s = a.structural;
    a.setTab('structural');
    s.setStudy(sid);
    const res = document.querySelector('#res-range');
    res.value = 40;
    res.dispatchEvent(new Event('input'));
    res.dispatchEvent(new Event('change'));
    if (fn) new Function('st', 's', fn)(s.studies[sid], s);
    s.renderStudyOptions();
  }, [id, setup]);
  await page.click('#btn-run');
  await page.waitForTimeout(100);
  await idle();
  const runStatus = await page.textContent('#status');
  assert.ok(!(await page.locator('#status').getAttribute('class'))?.includes('error') || id === 'buckling' || id === 'drop', `${label}: ${runStatus}`);
  if (frozen) await page.evaluate(frozen); // stop animations so both pictures show the same instant
  await page.waitForTimeout(500);
  const before = await snapshot();
  const shotBefore = await page.locator('#viewport').screenshot();
  const resultBefore = results ? await page.evaluate(results) : null;

  // save: export -> container arrays -> decoded, as a file would hold it
  const stored = await page.evaluate(async ([sid]) => {
    const { encodeArrays, decodeArrays } = await import('/src/core/psim.js');
    const st = window.partsSim.structural.studies[sid];
    const ex = st.exportResult();
    if (!ex) return { none: true };
    const { bytes, bounds } = encodeArrays(ex.meta, ex.arrays);
    const dec = decodeArrays(bytes);
    window.__stored = { meta: dec.meta, arrays: dec.arrays, opts: JSON.parse(JSON.stringify(st.exportOptions())), note: st.fileNote(st.result) };
    const per = {};
    for (const a of ex.arrays) per[a.name.replace(/\.\d+/g, '.#')] = (per[a.name.replace(/\.\d+/g, '.#')] || 0) + a.data.length;
    return { bytes: bytes.length, arrays: ex.arrays.length, bounds: bounds.length, note: window.__stored.note, per, enc: [...new Set(ex.arrays.map((a) => a.enc))].join('+') };
  }, [id]);
  assert.ok(!stored.none, `${label}: exportResult() is null after a run`);

  // restore into a cleared study
  await page.evaluate(([sid]) => {
    const a = window.partsSim, s = a.structural, st = s.studies[sid];
    st.clear();
    assert_(st.result === null, 'clear() leaves a result');
    document.querySelector('#study-kpis').textContent = '';
    document.querySelector('#legend').hidden = true;
    a.viewer.setScalars(null);
    a.viewer.setDeformation(null);
    const { meta, arrays, opts } = window.__stored;
    st.importResult(meta, arrays);
    st.importOptions(opts);
    s.display = 'study';
    st.show();
    function assert_(c, m) { if (!c) throw new Error(m); }
  }, [id]);
  if (frozen) await page.evaluate(frozen);
  await page.waitForTimeout(500);
  const after = await snapshot();
  const shotAfter = await page.locator('#viewport').screenshot();
  const resultAfter = results ? await page.evaluate(results) : null;

  const exact = before.kpis === after.kpis && before.legend === after.legend;
  assert.ok(sameText(before.kpis, after.kpis, quant), `${label}: KPIs differ\n--- run\n${before.kpis}\n--- file\n${after.kpis}`);
  assert.ok(sameText(before.legend, after.legend, quant), `${label}: legend differs\n--- run\n${before.legend}\n--- file\n${after.legend}`);
  assert.equal(after.rows, before.rows, `${label}: list/table rows`);
  assert.equal(after.title, before.title);
  if (results) assert.deepEqual(resultAfter, resultBefore, `${label}: result counts`);
  const diff = await pixelDiff(shotBefore, shotAfter);
  assert.ok(diff < 6, `${label}: viewport differs by ${diff.toFixed(2)}`);
  const secs = (performance.now() - t0) / 1000;
  console.log(`${label.padEnd(18)} ${secs.toFixed(1).padStart(5)} s  ${String(stored.bytes).padStart(8)} B  ${stored.arrays} arrays (${stored.enc || 'none'})  text ${exact ? 'identical' : `within ${quant}`}  pixel diff ${diff.toFixed(3)}  note: ${stored.note}`);
  if (secs > 60) console.log(`  (slow: ${secs.toFixed(0)} s)`);
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
  await page.goto(`${base}?sample=beam`, { timeout: 120000 });
  await page.waitForFunction(() => !!window.partsSim?.part, null, { timeout: 120000 });
  await idle();
  // the solvers run on the CPU: this test is about the files, not about the engines (PSIM_ENGINE=auto lets the GPU in)
  await page.evaluate((e) => { const el = document.querySelector('#fea-engine'); if (el && [...el.options].some((o) => o.value === e)) el.value = e; }, process.env.PSIM_ENGINE || 'cpu');
  // the animations run off real time: freeze them
  const still = `(() => { const st = window.partsSim.structural.studies; for (const k of ['modal', 'buckling']) { st[k].view.animate = false; st[k].draw(); } })()`;

  // nonlinear numbers come from the file's JSON: exact; the pictures use 16-bit fields
  await check('nonlinear', 'nonlinear applied', { results: () => { const r = window.partsSim.structural.studies.nonlinear.result; return { steps: r.steps.length, unloaded: !!r.unloaded, reason: r.reason }; } });
  await check('nonlinear', 'nonlinear failure', {
    setup: 'st.opts.mode = "failure"; st.opts.steps = 6;',
    results: () => { const r = window.partsSim.structural.studies.nonlinear.result; return { steps: r.steps.length, unloaded: !!r.unloaded, reason: r.reason }; },
  });
  await check('modal', 'modal', { frozen: still, quant: 1e-6, setup: 'st.view.animate = false;', results: () => window.partsSim.structural.studies.modal.result.modes.map((m) => [m.freq, ...m.eff]) });
  await check('buckling', 'buckling', { frozen: still, quant: 1e-6, setup: 'st.view.animate = false;', results: () => window.partsSim.structural.studies.buckling.result.modes.map((m) => m.factor) });
  await check('fatigue', 'fatigue', { quant: 2e-3 });
  await check('drop', 'drop', { setup: 'st.opts.height = 0.5;', quant: 2e-5, results: () => { const r = window.partsSim.structural.studies.drop.result; return { frames: r.frames.length, times: r.times.length, peak: r.peak }; } });
  await check('optimize', 'topology', { setup: 'st.opts.goal = "topology"; st.opts.iters = 10;', quant: 6e-3 });
  await check('optimize', 'sizing', { setup: 'st.opts.goal = "sizing";', results: () => window.partsSim.structural.studies.optimize.result.rows.length });

  // linear dynamic is not stored in version 1
  const dyn = await page.evaluate(() => { const d = window.partsSim.structural.studies.dynamic; return [d.exportResult(), typeof d.exportOptions()]; });
  assert.deepEqual(dyn, [null, 'object']);
  console.log('dynamic            exportResult() is null (not stored in version 1)');

  // option import tolerates junk
  await page.evaluate(() => {
    const st = window.partsSim.structural.studies;
    for (const k of Object.keys(st)) { st[k].importOptions({ opts: { nev: 'x', steps: -3, plot: 4, height: null }, view: { plot: {}, mode: 'a' } }); st[k].importOptions(null); st[k].importOptions(5); }
  });
  // wrong part size is reported
  const err = await page.evaluate(() => {
    const st = window.partsSim.structural.studies.modal;
    const { meta, arrays } = window.__stored ?? {};
    try { st.importResult({ ...(meta || {}), nVert: 3 }, new Map()); return null; } catch (e) { return e.message; }
  });
  assert.match(err, /vertices/);
  assert.deepEqual(errors, []);
  console.log('All study result checks passed.');
} finally {
  await browser?.close();
  server?.kill();
}
