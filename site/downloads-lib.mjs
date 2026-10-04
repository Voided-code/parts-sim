// The download cards: which release file belongs on which card, and how a card is drawn. The page
// (in the browser) and scripts/build-site.mjs (in Node, for the no-script version) share this file,
// so both draw the same HTML from the same downloads data.
//
// downloads data: { version, tag, releaseUrl, checksums: {name, url}|null, assets: [{ name, size, digest, url }] }
// Asset names are GitHub's: spaces in the file names become dots.

export const REPO = 'Voided-code/parts-sim';

/** One entry per package we ship. `file` is the asset name; {v} is the version. */
export const PACKAGES = [
  { id: 'mac-native-arm64', os: 'mac', app: 'native', title: 'Apple silicon', kind: 'Disk image (.dmg)', file: 'Parts-Sim-{v}-native-Darwin-arm64.dmg' },
  { id: 'mac-electron-arm64-dmg', os: 'mac', app: 'electron', title: 'Apple silicon', kind: 'Disk image (.dmg)', file: 'Parts.Sim-{v}-arm64.dmg' },
  { id: 'mac-electron-arm64-zip', os: 'mac', app: 'electron', title: 'Apple silicon', kind: 'Zip archive (.zip)', file: 'Parts.Sim-{v}-arm64-mac.zip' },
  { id: 'mac-electron-x64-dmg', os: 'mac', app: 'electron', title: 'Intel', kind: 'Disk image (.dmg)', file: 'Parts.Sim-{v}.dmg' },
  { id: 'mac-electron-x64-zip', os: 'mac', app: 'electron', title: 'Intel', kind: 'Zip archive (.zip)', file: 'Parts.Sim-{v}-mac.zip' },
  { id: 'win-native-x64-exe', os: 'windows', app: 'native', title: 'Windows 10/11, x64', kind: 'Installer (.exe)', file: 'Parts-Sim-{v}-native-Windows-AMD64.exe' },
  { id: 'win-native-x64-zip', os: 'windows', app: 'native', title: 'Windows 10/11, x64', kind: 'Portable zip (.zip)', file: 'Parts-Sim-{v}-native-Windows-AMD64.zip' },
  { id: 'win-electron-setup', os: 'windows', app: 'electron', title: 'Windows 10/11, x64 and ARM', kind: 'Installer (.exe)', file: 'Parts.Sim.Setup.{v}.exe' },
  { id: 'win-electron-x64-zip', os: 'windows', app: 'electron', title: 'Windows 10/11, x64', kind: 'Portable zip (.zip)', file: 'Parts.Sim-{v}-win.zip' },
  { id: 'win-electron-arm64-zip', os: 'windows', app: 'electron', title: 'Windows 10/11, ARM', kind: 'Portable zip (.zip)', file: 'Parts.Sim-{v}-arm64-win.zip' },
  { id: 'linux-native-x64', os: 'linux', app: 'native', title: 'Linux, x86-64', kind: 'AppImage', file: 'Parts-Sim-{v}-native-Linux-x86_64.AppImage' },
  { id: 'linux-electron-appimage', os: 'linux', app: 'electron', title: 'Linux, x86-64', kind: 'AppImage', file: 'Parts.Sim-{v}.AppImage' },
  { id: 'linux-electron-targz', os: 'linux', app: 'electron', title: 'Linux, x86-64', kind: 'Tarball (.tar.gz)', file: 'parts-sim-{v}.tar.gz' },
];

export const OS_NAMES = { mac: 'macOS', windows: 'Windows', linux: 'Linux' };
export const APP_NAMES = { native: 'Native app', electron: 'Desktop app (Electron)' };
export const APP_NOTES = {
  native: 'C++ and Qt. Fastest CPU solvers (about 10 times the Electron app’s), and airflow on the GPU through Metal, Direct3D 12 or Vulkan.',
  electron: 'The same app as the web page, in a desktop shell: Open With, menus and file associations. Its GPU backend is chosen by Chromium.',
};
export const CHECKSUMS_NAME = 'Parts-Sim-{v}-SHA256SUMS.txt';

const VERIFY = {
  windows: (f) => `certutil -hashfile "${f}" SHA256`,
  mac: (f) => `shasum -a 256 "${f}"`,
  linux: (f) => `sha256sum "${f}"`,
};

export const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

export function formatSize(bytes) {
  if (!Number.isFinite(bytes)) return '';
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${(bytes / 1e6).toFixed(bytes >= 1e8 ? 0 : 1)} MB`;
}

/** Turns a GitHub release (as `gh release view --json` or the REST API gives it) into downloads data. */
export function fromRelease(release) {
  const tag = release.tagName ?? release.tag_name;
  const assets = (release.assets ?? []).map((a) => ({
    name: a.name,
    size: a.size,
    digest: a.digest ?? null,
    url: a.url && a.url.includes('/releases/download/') ? a.url : (a.browser_download_url ?? `https://github.com/${REPO}/releases/download/${tag}/${a.name}`),
  }));
  return {
    version: tag.replace(/^v/, ''),
    tag,
    releaseUrl: release.url ?? release.html_url ?? `https://github.com/${REPO}/releases/tag/${tag}`,
    publishedAt: release.publishedAt ?? release.published_at ?? null,
    assets,
  };
}

/** The cards for downloads data: each package with its asset, or null when the release has none. */
export function cards(data) {
  const byName = new Map((data?.assets ?? []).map((a) => [a.name, a]));
  return PACKAGES.map((p) => {
    const file = p.file.replace('{v}', data?.version ?? '');
    return { ...p, fileName: file, asset: data ? byName.get(file) ?? null : null };
  });
}

function cardHtml(c, data) {
  const a = c.asset;
  const id = `dl-${c.id}`;
  const head = `<h4>${esc(c.title)}</h4><p class="kind">${esc(c.kind)}</p>`;
  if (!a) {
    const why = data
      ? `Not attached to release ${data.tag} (${c.fileName}). It appears here once it is built and uploaded.`
      : 'No release information is available yet.';
    return `<article class="card missing" data-package="${c.id}">${head}
      <p class="file"><code>${esc(c.fileName)}</code></p>
      <span class="btn disabled" role="link" aria-disabled="true" tabindex="0" aria-describedby="${id}-why" title="${esc(why)}">Not built yet</span>
      <p class="why" id="${id}-why">${esc(why)}</p></article>`;
  }
  const sha = a.digest ? a.digest.replace(/^sha256:/, '') : '';
  return `<article class="card ready" data-package="${c.id}">${head}
      <p class="file"><code>${esc(a.name)}</code></p>
      <p class="meta">Version ${esc(data.version)} · ${esc(formatSize(a.size))}</p>
      <a class="btn primary" href="${esc(a.url)}" download>Download</a>
      ${sha ? `<details><summary>SHA-256 and how to check it</summary><p class="sha"><code>${esc(sha)}</code></p>
      <p>Check on ${esc(OS_NAMES[c.os])}:</p><pre><code>${esc(VERIFY[c.os](a.name))}</code></pre></details>` : ''}
    </article>`;
}

/** The download cards grouped by operating system, then by app. */
export function downloadsHtml(data) {
  const list = cards(data);
  const built = list.filter((c) => c.asset).length;
  const summary = data
    ? `${built} of ${list.length} packages are attached to release ${esc(data.tag)}.`
    : 'No release information is available.';
  const groups = Object.keys(OS_NAMES).map((os) => {
    const apps = Object.keys(APP_NAMES).map((app) => {
      const items = list.filter((c) => c.os === os && c.app === app);
      if (!items.length) return '';
      return `<div class="app"><h3>${esc(APP_NAMES[app])}</h3><p class="note">${esc(APP_NOTES[app])}</p>
        <div class="cards">${items.map((c) => cardHtml(c, data)).join('\n')}</div></div>`;
    }).join('\n');
    return `<section class="os" data-os="${os}" aria-labelledby="os-${os}"><h2 id="os-${os}">${esc(OS_NAMES[os])} <span class="yours" hidden>your system</span></h2>${apps}</section>`;
  }).join('\n');
  const sums = data?.assets?.find((a) => a.name === CHECKSUMS_NAME.replace('{v}', data.version));
  return `<p class="summary" id="dl-summary">${summary}${sums ? ` Checksums for every file: <a href="${esc(sums.url)}">${esc(sums.name)}</a>.` : ''}</p>
    <div class="os-list" id="dl-list">${groups}</div>`;
}

/** The "Build from source" card. `available` is true once install.sh is on main. */
export function sourceHtml(available) {
  const cmds = `git clone https://github.com/${REPO}.git\ncd parts-sim\n./install.sh`;
  const win = `git clone https://github.com/${REPO}.git\ncd parts-sim\npowershell -ExecutionPolicy Bypass -File install.ps1`;
  if (!available) {
    return `<article class="card missing source" id="source-card"><h4>Build from source</h4>
      <p class="kind">Coming soon</p>
      <p>A one-step installer (<code>install.sh</code> for macOS and Linux, <code>install.ps1</code> for Windows) will build the app from a git clone. It is not on the main branch yet.</p>
      <span class="btn disabled" role="link" aria-disabled="true" tabindex="0">Coming soon</span></article>`;
  }
  return `<article class="card ready source" id="source-card"><h4>Build from source</h4>
      <p class="kind">After <code>git clone</code>; needs Node.js 20.19 or newer</p>
      <p>macOS and Linux:</p><pre><code>${esc(cmds)}</code></pre>
      <p>Windows (PowerShell):</p><pre><code>${esc(win)}</code></pre>
      <p>The first run takes a few minutes. The app goes into your user folder; no administrator rights.</p></article>`;
}
