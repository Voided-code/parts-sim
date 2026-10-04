// Checks on the built site (run `node scripts/build-site.mjs` first):
//   links      every internal link and file reference in _site resolves; nothing loads from another site
//   downloads  the download page in three states (all built, some missing, none), screenshots in
//              test-artifacts/web/, button rules asserted
//   isolation  /app/ gets cross-origin isolation from its service worker, or says it runs on one thread
//   smoke      scripts/browser-smoke.mjs against the served /app/
// With no argument all four run. Browser parts need CHROME=<path to Chrome for Testing>.
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { serveSite } from './serve-site.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const site = join(root, '_site');
const shots = join(root, 'test-artifacts/web');
const want = process.argv.slice(2);
const run = (name) => want.length === 0 || want.includes(name);
mkdirSync(shots, { recursive: true });

function walk(dir, found = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    statSync(p).isDirectory() ? walk(p, found) : found.push(p);
  }
  return found;
}

// ---- links --------------------------------------------------------------------------------------
function checkLinks() {
  const pages = walk(site).filter((f) => f.endsWith('.html'));
  const ids = new Map();
  const idsOf = (file) => {
    if (!ids.has(file)) ids.set(file, new Set([...readFileSync(file, 'utf8').matchAll(/\sid="([^"]+)"/g)].map((m) => m[1])));
    return ids.get(file);
  };
  const problems = [];
  let count = 0;
  for (const page of pages) {
    const html = readFileSync(page, 'utf8');
    for (const tag of html.matchAll(/<(\w+)\s([^>]*)>/g)) {
      for (const m of tag[2].matchAll(/(?:^|\s)(href|src|action)="([^"]*)"/g)) {
        const [, , value] = m;
        if (!value || value.startsWith('data:') || value.startsWith('mailto:')) continue;
        if (/^https?:/.test(value) || value.startsWith('//')) {
          // an outside address is fine as a link the visitor clicks, never as something the page loads
          if (tag[1] !== 'a') problems.push(`${page}: <${tag[1]}> loads from another site: ${value}`);
          continue;
        }
        count++;
        const [pathPart, hash] = value.split('#');
        let target = pathPart ? resolve(dirname(page), pathPart.split('?')[0]) : page;
        if (!target.startsWith(site)) { problems.push(`${page}: leaves the site: ${value}`); continue; }
        if (existsSync(target) && statSync(target).isDirectory()) target = join(target, 'index.html');
        if (!existsSync(target)) { problems.push(`${page}: dead link ${value}`); continue; }
        if (hash && target.endsWith('.html') && !idsOf(target).has(hash)) problems.push(`${page}: no #${hash} in ${target.slice(site.length)}`);
      }
    }
  }
  // JS and CSS that name other origins
  for (const f of walk(site).filter((p) => /\.(css|js|mjs)$/.test(p) && !p.includes('/app/'))) {
    for (const m of readFileSync(f, 'utf8').matchAll(/(?:url\(|import\s+[^'"]*from\s*|import\()\s*['"]?(https?:\/\/[^'")\s]+)/g)) problems.push(`${f}: loads ${m[1]}`);
  }
  console.log(`links: ${pages.length} pages, ${count} internal references checked`);
  assert.deepEqual(problems, [], problems.join('\n'));
}

// ---- browser parts ------------------------------------------------------------------------------
async function launch() {
  const { chromium } = await import('playwright');
  return chromium.launch({
    headless: true,
    ...(process.env.CHROME ? { executablePath: process.env.CHROME } : {}),
    args: process.platform === 'darwin' ? ['--enable-unsafe-webgpu', '--use-angle=metal'] : ['--enable-unsafe-webgpu'],
  });
}

const seed = JSON.parse(readFileSync(join(root, 'site/downloads.seed.json'), 'utf8'));
const STATES = {
  'all-built': seed,
  // the native Windows and Linux packages and one Electron file are missing, as before a CI run
  'some-missing': { ...seed, assets: seed.assets.filter((a) => !/native-(Windows|Linux)|\.AppImage$|arm64-win\.zip$/.test(a.name)) },
  none: { ...seed, assets: [] },
};

async function checkDownloads() {
  const browser = await launch();
  try {
    for (const [state, data] of Object.entries(STATES)) {
      const dir = join(root, `_site-${state}`);
      const file = join(root, `_site-${state}.json`);
      writeFileSync(file, JSON.stringify(data));
      execFileSync(process.execPath, [join(root, 'scripts/build-site.mjs'), '--skip-app', '--out', dir, '--downloads', file], { stdio: 'pipe' });
      const { server, url } = await serveSite(dir);
      for (const scheme of ['light', 'dark']) {
        const context = await browser.newContext({ viewport: { width: 1180, height: 900 }, colorScheme: scheme });
        const page = await context.newPage();
        // the live GitHub check must not decide the outcome: fail it, so the built-in data shows
        await page.route('https://api.github.com/**', (route) => route.abort());
        const errors = [];
        page.on('pageerror', (e) => errors.push(e.message));
        await page.goto(`${url}download/`);
        await page.waitForSelector('#dl-list');
        const cards = await page.$$eval('#dl-list .card', (els) => els.map((el) => ({
          pkg: el.dataset.package,
          missing: el.classList.contains('missing'),
          link: el.querySelector('a.btn')?.getAttribute('href') ?? null,
          disabled: el.querySelector('.btn.disabled')?.getAttribute('aria-disabled') ?? null,
          label: el.querySelector('.btn')?.textContent.trim(),
          why: el.querySelector('.btn.disabled')?.getAttribute('title') ?? null,
        })));
        const names = new Set(data.assets.map((a) => a.name));
        for (const c of cards) {
          if (c.missing) {
            assert.equal(c.link, null, `${state}/${c.pkg}: a missing package has a link`);
            assert.equal(c.disabled, 'true', `${state}/${c.pkg}: missing button is not aria-disabled`);
            assert.equal(c.label, 'Not built yet');
            assert.ok(c.why, `${state}/${c.pkg}: no reason on hover`);
          } else {
            assert.ok(c.link && names.has(decodeURIComponent(c.link.split('/').pop())), `${state}/${c.pkg}: link ${c.link} is not a release asset`);
          }
        }
        const ready = cards.filter((c) => !c.missing && c.pkg).length;
        assert.equal(ready, data.assets.filter((a) => cards.some((c) => c.link?.endsWith('/' + a.name))).length);
        if (state === 'none') assert.equal(ready, 0);
        if (state === 'all-built') assert.equal(cards.filter((c) => c.missing && c.pkg).length, 0);
        assert.deepEqual(errors, []);
        if (scheme === 'light') console.log(`downloads/${state}: ${ready} of ${cards.filter((c) => c.pkg).length} packages live`);
        await page.screenshot({ path: join(shots, `download-${state}-${scheme}.png`), fullPage: true });
        await context.close();
      }
      // a phone, and the keyboard reaching a greyed button
      const phone = await browser.newContext({ viewport: { width: 390, height: 800 }, isMobile: true });
      const p = await phone.newPage();
      await p.route('https://api.github.com/**', (route) => route.abort());
      await p.goto(`${url}download/`);
      assert.ok(await p.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), `${state}: sideways scroll on a phone`);
      if (state === 'all-built') await p.screenshot({ path: join(shots, 'download-phone.png'), fullPage: true });
      await phone.close();
      await new Promise((r) => server.close(r));
    }
    // OS detection: the visitor's system comes first
    for (const [ua, os] of [['Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/130', 'windows'], ['Mozilla/5.0 (X11; Linux x86_64) Chrome/130', 'linux'], ['Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Chrome/130', 'mac']]) {
      const { server, url } = await serveSite(join(root, '_site-all-built'));
      const context = await browser.newContext({ userAgent: ua });
      const page = await context.newPage();
      await page.route('https://api.github.com/**', (route) => route.abort());
      await page.goto(`${url}download/`);
      await page.waitForSelector('#dl-list');
      await page.waitForFunction(() => document.querySelector('.os[style*="order"]'));
      const first = await page.$$eval('#dl-list .os', (els) => els.sort((a, b) => a.style.order - b.style.order)[0].dataset.os);
      assert.equal(first, os, `detected system for ${ua}`);
      console.log(`downloads/detect: ${os} first`);
      await context.close();
      await new Promise((r) => server.close(r));
    }
  } finally {
    await browser.close();
  }
}

async function checkIsolation() {
  const { server, url } = await serveSite(site);
  const browser = await launch();
  try {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    const errors = [];
    page.on('pageerror', (e) => errors.push(e.message));
    await page.goto(`${url}app/`);
    await page.waitForFunction(() => self.crossOriginIsolated === true && !!window.partsSim, null, { timeout: 60000 });
    assert.equal(await page.evaluate(() => typeof SharedArrayBuffer), 'function');
    await page.waitForTimeout(2200);
    assert.equal(await page.locator('[role=status]').count(), 0, 'one-thread note shown although isolated');
    await page.screenshot({ path: join(shots, 'app-isolated.png') });
    console.log('isolation: service worker gave /app/ cross-origin isolation');
    await context.close();

    const blocked = await browser.newContext({ serviceWorkers: 'block', viewport: { width: 1440, height: 900 } });
    const p2 = await blocked.newPage();
    await p2.goto(`${url}app/`);
    await p2.waitForFunction(() => !!window.partsSim);
    await p2.waitForSelector('[role=status]', { timeout: 10000 });
    assert.match(await p2.textContent('[role=status]'), /one thread/);
    assert.equal(await p2.evaluate(() => self.crossOriginIsolated), false);
    await p2.screenshot({ path: join(shots, 'app-one-thread.png') });
    console.log('isolation: without the service worker the app opens and says it runs on one thread');
    await blocked.close();
    assert.deepEqual(errors, []);
  } finally {
    await browser.close();
    await new Promise((r) => server.close(r));
  }
}

async function checkSmoke() {
  const { server, url } = await serveSite(site);
  try {
    const r = await new Promise((done) => {
      const child = spawnSync(process.execPath, [join(root, 'scripts/browser-smoke.mjs')], { stdio: 'inherit', env: { ...process.env, PARTS_SIM_URL: `${url}app/` } });
      done(child);
    });
    assert.equal(r.status, 0, 'browser-smoke failed against the built /app/');
  } finally {
    server.close();
  }
}

if (run('links')) checkLinks();
if (run('downloads')) await checkDownloads();
if (run('isolation')) await checkIsolation();
if (run('smoke')) await checkSmoke();
console.log('site checks passed');
