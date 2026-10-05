// Opens .psim files written by another app (the native app) in the web app and prints what the
// panels show, so the numbers can be compared with the other app's own.
//   npm run build && CHROME=... node scripts/psim-cross-smoke.mjs file.psim [file.psim ...]
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { basename } from 'node:path';
import { chromium } from 'playwright';
import { preview } from 'vite';

const files = process.argv.slice(2);
assert.ok(files.length, 'give at least one .psim file');
const server = process.env.PARTS_SIM_URL ? null : await preview({ preview: { host: '127.0.0.1', port: 4177, strictPort: false } });
const base = process.env.PARTS_SIM_URL || server.resolvedUrls.local[0];
let browser;
try {
  browser = await chromium.launch({
    headless: true,
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}),
    args: process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu'],
  });
  let failed = 0;
  for (const file of files) {
    const bytes = new Uint8Array(await readFile(file));
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(base);
    await page.waitForFunction(() => !!window.partsSim, null, { timeout: 60000 });
    await page.evaluate(async ([arr, name]) => { await window.partsSim.psim.open(Uint8Array.from(arr), name); }, [[...bytes], basename(file)]);
    await page.waitForTimeout(800);
    const r = await page.evaluate(() => ({
      status: document.querySelector('#status').textContent,
      cls: document.querySelector('#status').className,
      banner: document.querySelector('#file-banner').hidden ? '' : document.querySelector('#file-banner').textContent,
      tab: window.partsSim.tab,
      kpis: document.querySelector('#kpis')?.innerText,
      thermal: document.querySelector('#th-kpis')?.innerText,
      flow: document.querySelector('#flow-card')?.hidden ? '' : document.querySelector('#flow-kpis')?.innerText,
      frozen: window.partsSim.airflow?.study?.frozen,
      study: window.partsSim.structural.study,
      card: document.querySelector('#study-card')?.hidden ? '' : document.querySelector('#study-card')?.innerText.slice(0, 160),
      breakKpis: document.querySelector('#break-card')?.hidden ? '' : document.querySelector('#break-kpis')?.innerText,
      part: window.partsSim.part?.name,
    }));
    const ok = !errors.length && !/error/.test(r.cls) && /Loaded from file/.test(r.banner);
    console.log(`${ok ? 'ok  ' : 'FAIL'} ${basename(file)} -> ${r.part}, tab ${r.tab}\n  ${r.status}\n  ${(r.kpis || '').replace(/\n+/g, ' | ')}\n  ${(r.thermal || '').replace(/\n+/g, ' | ')}${r.card ? `\n  ${r.study}: ${r.card.replace(/\n+/g, ' | ')}` : ''}${r.breakKpis ? `\n  break: ${r.breakKpis.replace(/\n+/g, ' | ')}` : ''}${r.flow ? `\n  airflow (frozen ${r.frozen}): ${r.flow.replace(/\n+/g, ' | ')}` : ''}${errors.length ? `\n  errors: ${errors.join('; ')}` : ''}`);
    if (!ok) failed++;
    await page.screenshot({ path: `test-artifacts/web/cross-${basename(file, '.psim')}.png` });
    await page.close();
  }
  assert.equal(failed, 0, `${failed} file(s) did not open cleanly`);
} finally {
  await browser?.close();
  await server?.close();
}
