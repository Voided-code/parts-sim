// Builds the website into _site/: landing page (/), the web app (/app/), docs (/docs/), downloads
// (/download/). Every link is relative, so it works under a sub-path such as /parts-sim/.
//
//   node scripts/build-site.mjs [--out _site] [--downloads file.json] [--skip-app]
//
// Downloads data comes from, in order: --downloads, the latest GitHub release (through `gh`),
// site/downloads.seed.json.
import { execFileSync } from 'node:child_process';
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, posix, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { fromRelease, downloadsHtml, sourceHtml, esc, REPO } from '../site/downloads-lib.mjs';
import { renderMarkdown, section, escapeHtml } from './lib/markdown.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const arg = (name) => { const i = process.argv.indexOf(name); return i < 0 ? null : process.argv[i + 1]; };
const out = resolve(root, arg('--out') ?? '_site');
const skipApp = process.argv.includes('--skip-app');
const pkg = JSON.parse(readFileSync(join(root, 'package.json'), 'utf8'));
const read = (p) => readFileSync(join(root, p), 'utf8');
const write = (p, text) => { mkdirSync(dirname(join(out, p)), { recursive: true }); writeFileSync(join(out, p), text); };

// ---- downloads data -----------------------------------------------------------------------------
function loadDownloads() {
  const file = arg('--downloads');
  if (file) return { data: JSON.parse(readFileSync(resolve(file), 'utf8')), from: file };
  try {
    const json = execFileSync('gh', ['release', 'view', '--repo', REPO, '--json', 'tagName,url,assets,publishedAt'], { stdio: ['ignore', 'pipe', 'pipe'], timeout: 30000 }).toString();
    const data = fromRelease(JSON.parse(json));
    if (data.assets.length) return { data, from: `latest release ${data.tag}` };
  } catch { /* offline, no gh, or no release yet */ }
  return { data: JSON.parse(read('site/downloads.seed.json')), from: 'site/downloads.seed.json' };
}

// ---- pages --------------------------------------------------------------------------------------
const NAV = [
  ['', 'Home'], ['app/', 'Web app'], ['download/', 'Download'], ['docs/airflow-accuracy.html', 'Airflow accuracy'], ['docs/', 'Docs'],
];
const CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self' https://api.github.com; base-uri 'none'; form-action 'none'; object-src 'none'";

function layout({ title, description, body, depth, current }) {
  const r = depth ? '../'.repeat(depth) : './';
  const nav = NAV.map(([href, label]) => `<li><a href="${r}${href}"${href === current ? ' aria-current="page"' : ''}>${label}</a></li>`).join('');
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="${CSP}">
<title>${escapeHtml(title)}</title>
<meta name="description" content="${escapeHtml(description)}">
<link rel="icon" href="${r}favicon.svg" type="image/svg+xml">
<link rel="stylesheet" href="${r}style.css">
</head>
<body>
<a class="skip" href="#main">Skip to the content</a>
<header class="site"><div class="wrap">
<a class="brand" href="${r}"><img src="${r}favicon.svg" alt="" width="26" height="26">Parts Sim</a>
<nav class="main" aria-label="Main"><ul>${nav}<li><a href="https://github.com/${REPO}">GitHub</a></li></ul></nav>
</div></header>
<main id="main"><div class="wrap">
${body}
</div></main>
<footer class="site"><div class="wrap">
<ul><li><a href="https://github.com/${REPO}">Source code</a></li><li><a href="https://github.com/${REPO}/blob/main/LICENSE">MIT licence</a></li><li><a href="https://github.com/${REPO}/blob/main/THIRD_PARTY_NOTICES.md">Third-party notices</a></li><li><a href="https://github.com/${REPO}/issues">Report a problem</a></li></ul>
<p>Parts Sim is free and open source. This site has no analytics, no cookies and no third-party scripts, fonts or images.</p>
</div></footer>
<script src="${r}site.js" type="module"></script>
</body>
</html>
`;
}

const fill = (tpl, values) => tpl.replace(/\{\{(\w+)\}\}/g, (_, k) => values[k] ?? '');

// docs: rendered from the repository's Markdown files; a missing one is simply left out
const DOCS = [
  { src: 'docs/airflow-accuracy.md', out: 'docs/airflow-accuracy.html', blurb: 'How accurate the airflow study is, against published wind-tunnel data, and what limits it.' },
  { src: 'docs/psim-format.md', out: 'docs/psim-format.html', blurb: 'The .psim file format: one small file with a part, its setup and its results.' },
];

function rewriteFor(srcFile, depth) {
  const r = '../'.repeat(depth);
  const known = new Map(DOCS.map((d) => [d.src, d.out]));
  const resolveRepo = (href) => posix.normalize(posix.join(posix.dirname(srcFile), href.split('#')[0]));
  return {
    rewriteHref(href) {
      if (/^([a-z][a-z0-9+.-]*:|#|\/\/)/i.test(href)) return href;
      const hash = href.includes('#') ? '#' + href.split('#')[1] : '';
      const target = resolveRepo(href);
      if (known.has(target) && existsSync(join(root, target))) return r + known.get(target) + hash;
      return `https://github.com/${REPO}/blob/main/${target}${hash}`;
    },
    rewriteSrc(src) {
      if (/^([a-z][a-z0-9+.-]*:|\/\/)/i.test(src)) return src;
      const target = resolveRepo(src);
      return target.startsWith('docs/images/') ? (r || './') + target.slice('docs/'.length) : src;
    },
  };
}

function docPage(d) {
  const depth = d.out.split('/').length - 1;
  const { html, headings, title } = renderMarkdown(read(d.src), rewriteFor(d.src, depth));
  const toc = headings.filter((h) => h.level === 2);
  const tocHtml = toc.length > 3
    ? `<nav class="toc" aria-label="On this page"><strong>On this page</strong><ul>${toc.map((h) => `<li><a href="#${h.id}">${escapeHtml(h.text)}</a></li>`).join('')}</ul></nav>` : '';
  const edit = `<p><a href="https://github.com/${REPO}/blob/main/${d.src}">This page on GitHub</a></p>`;
  const body = `<article class="doc">${html.replace(/(<\/h1>)/, `$1${tocHtml}`)}${edit}</article>`;
  return { title, page: layout({ title: `${title} – Parts Sim`, description: d.blurb, body, depth, current: d.out }) };
}

// ---- build --------------------------------------------------------------------------------------
rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

// the web app: the normal build, served from /app/
if (!skipApp) {
  execFileSync(process.execPath, [join(root, 'node_modules/vite/bin/vite.js'), 'build', '--outDir', join(out, 'app'), '--emptyOutDir'], { cwd: root, stdio: 'inherit' });
  rmSync(join(out, 'app/bench.html'), { force: true });
  cpSync(join(root, 'site/vendor/coi-serviceworker.js'), join(out, 'app/coi-serviceworker.js'));
  cpSync(join(root, 'site/app-isolation.js'), join(out, 'app/app-isolation.js'));
  // GitHub Pages cannot send COOP/COEP headers; the service worker adds them, and the second script
  // says so when threads still are not available
  const file = join(out, 'app/index.html');
  const index = readFileSync(file, 'utf8');
  if (!index.includes('<head>')) throw new Error('app/index.html has no <head>');
  writeFileSync(file, index.replace('<head>', '<head>\n    <script src="./coi-serviceworker.js"></script>\n    <script src="./app-isolation.js" defer></script>'));
}

for (const f of ['style.css', 'site.js', 'download.js', 'downloads-lib.mjs']) cpSync(join(root, 'site', f), join(out, f));
cpSync(join(root, 'build/icon.svg'), join(out, 'favicon.svg'));
cpSync(join(root, 'docs/images'), join(out, 'images'), { recursive: true });
write('.nojekyll', '');

const { data, from } = loadDownloads();
write('downloads.json', JSON.stringify(data, null, 2) + '\n');

const docs = DOCS.filter((d) => existsSync(join(root, d.src)));
const pages = [];
for (const d of docs) { const { title, page } = docPage(d); write(d.out, page); pages.push({ ...d, title }); }

const accuracySource = existsSync(join(root, DOCS[0].src)) ? read(DOCS[0].src) : '';
const short = section(accuracySource, 'The short version');
const accuracy = short ? renderMarkdown(short, rewriteFor(DOCS[0].src, 0)).html : '<p>The accuracy page lists every validation case with its published reference.</p>';

write('index.html', layout({
  title: 'Parts Sim – structural, thermal and airflow simulation',
  description: 'Open-source structural, thermal and airflow simulation for SolidWorks, STEP, IGES, STL and OBJ parts. Runs in your browser or as a desktop app.',
  body: fill(read('site/pages/index.body.html'), { root: './', accuracy }), depth: 0, current: '',
}));

const sourceAvailable = existsSync(join(root, 'install.sh'));
write('download/index.html', layout({
  title: 'Download – Parts Sim',
  description: 'Parts Sim for macOS, Windows and Linux: native and Electron packages, with checksums.',
  body: fill(read('site/pages/download.body.html'), { root: '../', version: esc(data.version), downloads: downloadsHtml(data), source: sourceHtml(sourceAvailable) }),
  depth: 1, current: 'download/',
}));

write('docs/index.html', layout({
  title: 'Docs – Parts Sim',
  description: 'Documentation for Parts Sim.',
  body: `<h1>Docs</h1><ul>${pages.map((p) => `<li><a href="../${p.out}">${escapeHtml(p.title)}</a>: ${escapeHtml(p.blurb)}</li>`).join('')}
<li><a href="https://github.com/${REPO}#readme">README</a>: install, quick start, how the solvers work, validation and limits.</li></ul>`,
  depth: 1, current: 'docs/',
}));

write('404.html', layout({
  title: 'Page not found – Parts Sim', description: 'Page not found.',
  body: '<h1>Page not found</h1><p>That page does not exist. Try the <a href="./">home page</a> or the <a href="./download/">downloads</a>.</p>', depth: 0, current: '',
}));

console.log(`site built in ${out} (version ${pkg.version}; downloads from ${from}; build from source ${sourceAvailable ? 'live' : 'coming soon'}; docs: ${pages.map((p) => p.out).join(', ')})`);
