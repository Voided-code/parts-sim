// Builds the Windows benchmark kit: the Electron app for Windows x64 (unpacked), the native
// benchmark and app when a Windows build is given, a run-benchmark.cmd and a README.
//   node scripts/make-kit.mjs [--native <folder with bench.exe and the native app>] [--skip-build]
// Output: release/kit/parts-sim-kit-<version>-<commit>.zip
import { execFileSync, execSync } from 'node:child_process';
import { cp, mkdir, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const args = process.argv.slice(2);
const nativeDir = args.includes('--native') ? path.resolve(args[args.indexOf('--native') + 1]) : null;
const version = JSON.parse(await readFile(path.join(root, 'package.json'), 'utf8')).version;
let commit = execSync('git rev-parse --short HEAD', { cwd: root }).toString().trim();
if (execSync('git status --porcelain --untracked-files=no', { cwd: root }).toString().trim()) commit += '-modified';
const name = `parts-sim-kit-${version}-${commit}`;
const out = path.join(root, 'release', 'kit', name);

const env = { ...process.env, CSC_IDENTITY_AUTO_DISCOVERY: 'false' };
delete env.ELECTRON_RUN_AS_NODE;
if (!args.includes('--skip-build')) {
  execSync('npx vite build', { cwd: root, stdio: 'inherit', env });
  // the unpacked app only: no installer, x64
  execSync('npx electron-builder --win dir --x64 --publish never', { cwd: root, stdio: 'inherit', env });
}
await rm(out, { recursive: true, force: true });
await mkdir(out, { recursive: true });
await cp(path.join(root, 'release', 'win-unpacked'), path.join(out, 'electron'), { recursive: true });
if (nativeDir) {
  // CI's artifacts: the portable package (Parts-Sim-*.zip) and bench.exe; bench.exe goes next to the
  // app's DLLs
  const files = await readdir(nativeDir, { recursive: true });
  const zipFile = files.find((f) => /Parts-Sim-.*\.zip$/i.test(f));
  const benchFile = files.find((f) => /(^|[\\/])bench\.exe$/i.test(f));
  if (!zipFile || !benchFile) throw new Error(`${nativeDir} needs the native package (.zip) and bench.exe`);
  const tmp = path.join(root, 'release', 'kit', 'native-unzip');
  await rm(tmp, { recursive: true, force: true });
  execFileSync('ditto', ['-x', '-k', path.join(nativeDir, zipFile), tmp]);
  const exe = (await readdir(tmp, { recursive: true })).find((f) => /(^|[\\/])Parts Sim\.exe$/.test(f));
  if (!exe) throw new Error('The native package has no Parts Sim.exe');
  await cp(path.join(tmp, path.dirname(exe)), path.join(out, 'native'), { recursive: true });
  await cp(path.join(nativeDir, benchFile), path.join(out, 'native', 'bench.exe'));
  await rm(tmp, { recursive: true, force: true });
}

// the validation cases' meshes for the native benchmark (scripts/export-cases.mjs)
execFileSync(process.execPath, [path.join(root, 'scripts', 'export-cases.mjs'), path.join(out, 'cases')], { stdio: 'inherit' });

const cmd = String.raw`@echo off
rem Parts Sim benchmark kit ${version} (${commit}). Runs for about 30 minutes without clicks.
setlocal
cd /d "%~dp0"
set "OUT=%~dp0report"
if exist "%OUT%" rmdir /s /q "%OUT%"
mkdir "%OUT%"
echo Parts Sim benchmark kit ${version} (${commit})
echo Leave the PC alone until it says Done (about 30 minutes). Closing the window stops it.
echo.
powershell -NoProfile -Command "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate,VideoProcessor | ConvertTo-Json | Out-File -Encoding utf8 '%OUT%\gpu.json'; Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json | Out-File -Encoding utf8 '%OUT%\cpu.json'; Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,TotalVisibleMemorySize | ConvertTo-Json | Out-File -Encoding utf8 '%OUT%\os.json'"
if exist "native\bench.exe" (
  echo [1/6] Native app: flow engine speed with Vulkan, then Direct3D 12...
  set PARTS_SIM_GPU_BACKEND=vulkan
  "native\bench.exe" --json "%OUT%\native-vulkan.json" kit > "%OUT%\native-vulkan.log" 2>&1
  set PARTS_SIM_GPU_BACKEND=dx12
  "native\bench.exe" --json "%OUT%\native-dx12.json" kit > "%OUT%\native-dx12.log" 2>&1
  set PARTS_SIM_GPU_BACKEND=
  echo [2/6] Native app: the airflow study itself on the Ahmed body...
  "native\bench.exe" --json "%OUT%\native-app.json" app cases 4000000 16000000 64000000 128000000 > "%OUT%\native-app.log" 2>&1
  echo [3/6] Native app: validation cases...
  "native\bench.exe" --json "%OUT%\native-cases.json" cases cases plate,cube,sphere,ahmed25,ahmed35,wing,plate-turbulent,plate-laminar > "%OUT%\native-cases.log" 2>&1
) else (
  echo [1/6] No native build in this kit: steps 1 to 3 skipped.
)
echo [4/6] Desktop app: flow engine checks, speed and the airflow study (WebGPU)...
"electron\Parts Sim.exe" --bench=engine --bench-out="%OUT%\report-engine.json"
echo [5/6] Desktop app: validation cases and five minutes of sustained speed...
"electron\Parts Sim.exe" --bench=kit --bench-out="%OUT%\report.json" --bench-native="%OUT%"
echo [6/6] Desktop app speed with Chromium's Vulkan switches (to see which backend it picks)...
"electron\Parts Sim.exe" --bench=backend --bench-out="%OUT%\report-vulkan.json" --use-vulkan=native --enable-features=Vulkan,SkiaGraphite
powershell -NoProfile -Command "Compress-Archive -Force -Path '%OUT%\*' -DestinationPath '%~dp0parts-sim-report.zip'"
echo.
echo Done. Send back parts-sim-report.zip (in this folder).
pause
`;
await writeFile(path.join(out, 'run-benchmark.cmd'), cmd.replace(/\n/g, '\r\n'));

const readme = `Parts Sim benchmark kit ${version} (${commit})
1. Unzip this folder anywhere, close games and other heavy apps, and double-click run-benchmark.cmd.
2. If Windows SmartScreen warns about an unknown app, choose More info, then Run anyway (the builds are not code-signed).
3. It runs for about 30 minutes without clicks in six steps; windows open and close by themselves. Leave the PC alone until the window says Done.
4. The results land in this folder as parts-sim-report.zip (and the report folder beside it).
5. Send parts-sim-report.zip back. It holds speeds, forces and the names and drivers of the GPU, CPU and Windows version, nothing else.
`;
await writeFile(path.join(out, 'README.txt'), readme.replace(/\n/g, '\r\n'));

const zip = `${out}.zip`;
await rm(zip, { force: true });
execFileSync('ditto', ['-c', '-k', '--keepParent', out, zip]);
console.log(`Kit: ${zip}`);
if (!existsSync(path.join(out, 'native', 'bench.exe'))) console.log('(no native Windows build included)');
