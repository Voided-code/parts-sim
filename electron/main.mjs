// Parts Sim desktop app (macOS, Windows, Linux).
//
// The simulation app itself is the Vite build in dist/, served from a private app:// scheme
// (a secure context, so WebGPU, module workers and WebAssembly all work). This process adds
// what a browser cannot: native menus, open/save dialogs, opening files from Finder/Explorer
// and the command line, and reading an assembly's part files from the folder it lives in.
import { app, BrowserWindow, Menu, dialog, ipcMain, nativeTheme, net, protocol, shell } from 'electron';
import { readFile, readdir, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const ROOT = path.join(import.meta.dirname, '..', 'dist');
const ORIGIN = 'app://parts-sim';
const OPENABLE = ['step', 'stp', 'iges', 'igs', 'brep', 'brp', 'sldprt', 'sldasm', 'slddrw', 'stl', 'obj', '3mf', 'ply', 'glb', 'gltf'];
const SAMPLES = [
  ['beam', 'Cantilever beam'], ['lbracket', 'L-bracket (PLA print)'], ['bracket', 'Bolted mounting bracket'],
  ['wrench', 'Open-end wrench'], ['hook', 'Crane hook'], ['wing', 'Wing (NACA 2412)'], ['ahmed', 'Ahmed body (car)'],
];
const CSP = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "worker-src 'self' blob:",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' data: blob:",
  "connect-src 'self' data: blob:",
  "object-src 'none'",
  "base-uri 'none'",
].join('; ');
// The OpenCascade (STEP/IGES) glue code generates functions at runtime. Allow that only in
// its own worker, which receives nothing but the file being imported.
const CAD_WORKER_CSP = CSP.replace("script-src 'self' 'wasm-unsafe-eval'", "script-src 'self' 'wasm-unsafe-eval' 'unsafe-eval'");

protocol.registerSchemesAsPrivileged([
  { scheme: 'app', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true, codeCache: true } },
]);

// WebGPU is on by default on macOS and Windows; Linux still needs the flag (the app falls
// back to the CPU airflow engine when no adapter is available).
if (process.platform === 'linux') {
  app.commandLine.appendSwitch('enable-unsafe-webgpu');
  app.commandLine.appendSwitch('enable-features', 'Vulkan');
}
// Without a usable GPU (virtual machines, remote desktops, blocklisted drivers) Chromium no longer
// falls back to software WebGL on its own, and the 3D view could not start. The fallback is only
// "unsafe" for untrusted web content; this window runs nothing but the bundled app.
app.commandLine.appendSwitch('enable-unsafe-swiftshader');

let win = null;
let rendererReady = false;
const pendingPaths = [];

// Benchmark mode (the Windows benchmark kit): --bench[=full|quick|speed] runs bench.html without
// clicks and writes its JSON report to --bench-out (default: next to the executable). --bench-native
// names a folder whose native-app reports (native*.json) are included in it. --bench-cases limits the validation cases, and
// --bench-solver picks the flow solver (v0.6 or v1).
const argValue = (name) => process.argv.find((a) => a === `--${name}` || a.startsWith(`--${name}=`))?.split('=').slice(1).join('=');
const benchSuite = argValue('bench');
const bench = benchSuite === undefined ? null : {
  suite: benchSuite || 'full',
  cases: argValue('bench-cases') || '',
  solver: argValue('bench-solver') || '',
  out: path.resolve(argValue('bench-out') || path.join(path.dirname(process.execPath), 'parts-sim-bench-report.json')),
  native: argValue('bench-native') ? path.resolve(argValue('bench-native')) : null,
};

// full-precision GPU timestamps and the adapter's memory heaps and backend in the report
if (bench) app.commandLine.appendSwitch('enable-webgpu-developer-features');

// Tests run with their own profile so they neither collide with nor open files in a running copy
// (the single-instance lock is per profile).
if (process.env.PARTS_SIM_USER_DATA) app.setPath('userData', process.env.PARTS_SIM_USER_DATA);
else if (bench) app.setPath('userData', path.join(app.getPath('temp'), 'parts-sim-bench'));
if (!app.requestSingleInstanceLock()) app.quit();

app.on('second-instance', (_e, argv) => {
  queueOpen(filesFromArgv(argv));
  if (win) {
    if (win.isMinimized()) win.restore();
    win.focus();
  }
});

// macOS: double-click in Finder, drop on the Dock icon, Open With. May fire before 'ready'.
app.on('open-file', (e, file) => {
  e.preventDefault();
  queueOpen([file]);
});

function filesFromArgv(argv) {
  return argv.slice(1).filter((a) => !a.startsWith('-') && OPENABLE.includes(path.extname(a).slice(1).toLowerCase()));
}

function queueOpen(paths) {
  if (!paths.length) return;
  pendingPaths.push(...paths.map((p) => path.resolve(p)));
  flushOpen();
}

async function flushOpen() {
  if (!win || !rendererReady || !pendingPaths.length) return;
  const paths = pendingPaths.splice(0);
  const files = [];
  for (const p of paths) {
    try {
      files.push({ name: path.basename(p), path: p, data: await readFile(p) });
      app.addRecentDocument(p);
    } catch (err) {
      dialog.showErrorBox('Could not open file', `${p}\n\n${err.message}`);
    }
  }
  if (files.length) win.webContents.send('open-files', files);
}

async function showOpenDialog() {
  const res = await dialog.showOpenDialog(win, {
    title: 'Open part or assembly',
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'CAD parts', extensions: OPENABLE },
      { name: 'SolidWorks', extensions: ['sldprt', 'sldasm'] },
      { name: 'STEP / IGES', extensions: ['step', 'stp', 'iges', 'igs'] },
      { name: 'Meshes', extensions: ['stl', 'obj', '3mf', 'ply', 'glb'] },
    ],
  });
  if (!res.canceled) queueOpen(res.filePaths);
}

function send(command) {
  win?.webContents.send('command', command);
}

function buildMenu() {
  const mac = process.platform === 'darwin';
  const cmd = (label, command, accelerator) => ({ label, accelerator, click: () => send(command) });
  const template = [
    ...(mac ? [{ role: 'appMenu' }] : []),
    {
      label: 'File',
      submenu: [
        { label: 'Open…', accelerator: 'CmdOrCtrl+O', click: showOpenDialog },
        ...(mac ? [{ role: 'recentDocuments', submenu: [{ role: 'clearRecentDocuments' }] }] : []),
        { label: 'Open Sample', submenu: SAMPLES.map(([id, label]) => cmd(label, `sample:${id}`)) },
        { type: 'separator' },
        cmd('Save Screenshot…', 'screenshot', 'CmdOrCtrl+Shift+S'),
        { type: 'separator' },
        mac ? { role: 'close' } : { role: 'quit' },
      ],
    },
    { role: 'editMenu' },
    {
      label: 'View',
      submenu: [
        cmd('Isometric', 'view:iso', 'CmdOrCtrl+1'),
        cmd('Front', 'view:front', 'CmdOrCtrl+2'),
        cmd('Top', 'view:top', 'CmdOrCtrl+3'),
        cmd('Right', 'view:right', 'CmdOrCtrl+4'),
        { type: 'separator' },
        cmd('Toggle Edges', 'toggle:edges', 'CmdOrCtrl+E'),
        cmd('Toggle X-ray', 'toggle:xray', 'CmdOrCtrl+Shift+X'),
        { type: 'separator' },
        { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' },
        ...(app.isPackaged ? [] : [{ role: 'toggleDevTools' }]),
      ],
    },
    {
      label: 'Simulate',
      submenu: [
        cmd('Run Current Study', 'run:study', 'CmdOrCtrl+R'),
        cmd('Run Break Test', 'run:break', 'CmdOrCtrl+Shift+R'),
        { type: 'separator' },
        {
          label: 'Structural Study',
          submenu: [
            cmd('Linear Static', 'study:static'),
            cmd('Nonlinear Static', 'study:nonlinear'),
            cmd('Frequency', 'study:modal'),
            cmd('Buckling', 'study:buckling'),
            cmd('Fatigue', 'study:fatigue'),
            cmd('Drop Test', 'study:drop'),
            cmd('Linear Dynamic', 'study:dynamic'),
            cmd('Optimization', 'study:optimize'),
          ],
        },
        cmd('Run Heat Transfer', 'run:thermal', 'CmdOrCtrl+T'),
        cmd('Run / Pause Airflow', 'run:airflow', 'CmdOrCtrl+Alt+R'),
      ],
    },
    { role: 'windowMenu' },
    {
      role: 'help',
      submenu: [
        cmd('How Parts Sim Works', 'help'),
        {
          label: 'Open-Source Licences',
          click: () => dialog.showMessageBox(win, {
            type: 'info',
            title: 'Open-source licences',
            message: `Parts Sim ${app.getVersion()} is MIT-licensed.`,
            detail: 'It includes three.js and three-mesh-bvh (MIT), fflate (MIT), occt-import-js and Open CASCADE Technology (LGPL-2.1, shipped as a separate WebAssembly module), and Electron/Chromium (MIT and BSD-style licences). SolidWorks reading follows the format research in sldprt-export (MIT) and the cadmpeg specification (CC BY 4.0). See THIRD_PARTY_NOTICES.md for details.',
          }),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function createWindow() {
  const mac = process.platform === 'darwin';
  win = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 960,
    minHeight: 620,
    title: 'Parts Sim',
    show: false,
    backgroundColor: nativeTheme.shouldUseDarkColors ? '#111317' : '#e9edf2',
    ...(mac ? { titleBarStyle: 'hiddenInset', trafficLightPosition: { x: 14, y: 16 } } : {}),
    icon: path.join(import.meta.dirname, '..', 'build', 'icon.png'),
    webPreferences: {
      preload: path.join(import.meta.dirname, 'preload.cjs'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      spellcheck: false,
      ...(bench ? { additionalArguments: ['--parts-sim-bench'], backgroundThrottling: false } : {}),
    },
  });
  win.once('ready-to-show', () => win.show());
  win.on('closed', () => {
    win = null;
    rendererReady = false;
  });
  // Only the bundled app may run in the window; links open in the system browser.
  win.webContents.on('will-navigate', (e, url) => {
    if (!url.startsWith(`${ORIGIN}/`)) {
      e.preventDefault();
      if (/^https?:/i.test(url)) shell.openExternal(url);
    }
  });
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/i.test(url)) shell.openExternal(url);
    return { action: 'deny' };
  });
  if (bench) {
    const query = new URLSearchParams({ suite: bench.suite, ...(bench.cases ? { cases: bench.cases } : {}), ...(bench.solver ? { solver: bench.solver } : {}) });
    win.loadURL(`${ORIGIN}/bench.html?${query}`);
  } else win.loadURL(`${ORIGIN}/index.html`);
}

if (bench) {
  // the report is written whole after every item, through a temporary file so it is never half-written
  ipcMain.handle('bench-write', async (e, json) => {
    if (e.sender !== win?.webContents || typeof json !== 'string') return false;
    await writeFile(`${bench.out}.tmp`, json);
    await rename(`${bench.out}.tmp`, bench.out);
    return true;
  });
  // the native app's reports (native*.json) in the --bench-native folder, as one JSON array
  ipcMain.handle('bench-native', async (e) => {
    if (e.sender !== win?.webContents || !bench.native) return null;
    const names = (await readdir(bench.native).catch(() => [])).filter((f) => /^native.*\.json$/i.test(f)).sort();
    const reports = [];
    for (const f of names) {
      try { reports.push(JSON.parse(await readFile(path.join(bench.native, f), 'utf8'))); } catch { /* an unfinished report */ }
    }
    return reports.length ? JSON.stringify(reports) : null;
  });
  ipcMain.on('bench-done', (e) => {
    if (e.sender === win?.webContents) setTimeout(() => app.quit(), 500);
  });
}

ipcMain.on('renderer-ready', (e) => {
  if (e.sender !== win?.webContents) return;
  rendererReady = true;
  flushOpen();
});
ipcMain.on('open-dialog', (e) => {
  if (e.sender === win?.webContents) showOpenDialog();
});

// An assembly asks for one of its part files, by name, from the folder of a file the user
// opened. Only plain .SLDPRT file names are accepted: no paths, no other file types.
ipcMain.handle('read-sibling', async (e, openedPath, name) => {
  if (e.sender !== win?.webContents || typeof openedPath !== 'string' || typeof name !== 'string') return null;
  if (!path.isAbsolute(openedPath) || name !== path.basename(name) || !/\.sldprt$/i.test(name)) return null;
  const dir = path.dirname(openedPath);
  try {
    return await readFile(path.join(dir, name));
  } catch {
    // Windows paths inside the assembly may differ in case from the files on disk.
    const match = (await readdir(dir).catch(() => [])).find((f) => f.toLowerCase() === name.toLowerCase());
    return match ? readFile(path.join(dir, match)).catch(() => null) : null;
  }
});

ipcMain.handle('save-image', async (e, dataUrl, suggested) => {
  if (e.sender !== win?.webContents || typeof dataUrl !== 'string' || !dataUrl.startsWith('data:image/png;base64,')) return false;
  const res = await dialog.showSaveDialog(win, {
    title: 'Save screenshot',
    defaultPath: path.join(app.getPath('pictures'), `${String(suggested || 'parts-sim').replace(/[\\/:*?"<>|]/g, '_')}.png`),
    filters: [{ name: 'PNG image', extensions: ['png'] }],
  });
  if (res.canceled || !res.filePath) return false;
  await writeFile(res.filePath, Buffer.from(dataUrl.slice(dataUrl.indexOf(',') + 1), 'base64'));
  return true;
});

// Save a generated mesh (the topology-optimization result) as STL.
ipcMain.handle('save-file', async (e, data, suggested) => {
  if (e.sender !== win?.webContents || !(data instanceof Uint8Array) || typeof suggested !== 'string' || !/\.stl$/i.test(suggested)) return false;
  const res = await dialog.showSaveDialog(win, {
    title: 'Export STL',
    defaultPath: path.join(app.getPath('documents'), suggested.replace(/[\\/:*?"<>|]/g, '_')),
    filters: [{ name: 'STL mesh', extensions: ['stl'] }],
  });
  if (res.canceled || !res.filePath) return false;
  await writeFile(res.filePath, data);
  return true;
});

app.whenReady().then(() => {
  protocol.handle('app', (req) => {
    const { pathname } = new URL(req.url);
    const file = path.normalize(path.join(ROOT, decodeURIComponent(pathname)));
    if (!file.startsWith(ROOT + path.sep)) return new Response('Not found', { status: 404 });
    return net.fetch(pathToFileURL(file).toString()).then((res) => {
      const headers = new Headers(res.headers);
      headers.set('Content-Security-Policy', /[\\/]occt\.worker-[\w-]+\.js$/.test(file) ? CAD_WORKER_CSP : CSP);
      // cross-origin isolation: lets the solvers share memory between threads (SharedArrayBuffer)
      headers.set('Cross-Origin-Opener-Policy', 'same-origin');
      headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
      if (file.endsWith('.wasm')) headers.set('Content-Type', 'application/wasm');
      return new Response(res.body, { status: res.status, headers });
    });
  });
  if (!bench) buildMenu();
  createWindow();
  if (!bench) queueOpen(filesFromArgv(process.argv));
  app.on('activate', () => {
    if (!BrowserWindow.getAllWindows().length) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});
