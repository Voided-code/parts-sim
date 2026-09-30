// Runs the benchmark page (bench.html) in Chromium with WebGPU and saves its JSON report.
//   node scripts/bench-web.mjs [suite] [out.json] [cases] [more URL parameters, e.g. solver=v1&collision=bgk]
// suite: full | quick | speed (default quick); cases: comma-separated case ids. Needs `npm run build`.
// CHROME=<path> picks the browser binary (see scripts/browser-smoke.mjs).
import { writeFile } from 'node:fs/promises';
import { chromium } from 'playwright';
import { preview } from 'vite';

const [suite = 'quick', out = `test-artifacts/bench-${suite}.json`, cases = '', extra = ''] = process.argv.slice(2);
const server = await preview({ preview: { host: '127.0.0.1', port: 4174, strictPort: false } });
const base = server.resolvedUrls.local[0];
const browser = await chromium.launch({
  headless: true,
  ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}),
  args: ['--enable-unsafe-webgpu', '--enable-webgpu-developer-features', ...(process.platform === 'darwin' ? ['--use-angle=metal'] : [])],
});
try {
  const page = await browser.newPage();
  page.on('console', (m) => { if (m.type() === 'log') console.log(m.text()); });
  page.on('pageerror', (e) => console.error('page error:', e.message));
  await page.goto(`${base}bench.html?suite=${suite}${cases ? `&cases=${cases}` : ''}${extra ? `&${extra}` : ''}`);
  let last = 0;
  for (;;) {
    const r = await page.evaluate(() => window.__benchReport && { finished: window.__benchReport.finished, json: JSON.stringify(window.__benchReport, null, 1) });
    if (r && Date.now() - last > 30e3) { await writeFile(out, r.json); last = Date.now(); }
    if (r?.finished) { await writeFile(out, r.json); break; }
    await new Promise((res) => setTimeout(res, 2000));
  }
  console.log(`Report: ${out}`);
} finally {
  await browser.close();
  server.httpServer.close();
}
