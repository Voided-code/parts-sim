// Before/after table of two benchmark reports (report.json from the kit, or bench-web.mjs output).
//   node scripts/kit-compare.mjs before.json after.json
// A report folder or its parts-sim-report.zip also works (report.json is read from it).
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';

// a kit's reports: report.json, with the ladder and app loop of report-engine.json (kit 2 on) merged in
function load(file) {
  let read;
  if (statSync(file).isDirectory()) read = (name) => (existsSync(path.join(file, name)) ? readFileSync(path.join(file, name), 'utf8') : null);
  else if (file.endsWith('.zip')) read = (name) => { try { return execFileSync('unzip', ['-p', file, name], { stdio: ['ignore', 'pipe', 'ignore'] }).toString() || null; } catch { return null; } };
  else return JSON.parse(readFileSync(file, 'utf8'));
  const report = JSON.parse(read('report.json'));
  const engine = read('report-engine.json');
  if (engine) {
    const e = JSON.parse(engine);
    for (const k of ['ladder', 'appLoop', 'tune']) if (!report[k]?.length && e[k]?.length) report[k] = e[k];
    report.errors = [...(report.errors || []), ...(e.errors || [])];
  }
  return report;
}

const [a, b] = process.argv.slice(2).map((f) => {
  if (!f || !existsSync(f)) throw new Error('Usage: node scripts/kit-compare.mjs before.json after.json');
  return load(f);
});

const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? '–' : typeof v === 'number' ? v.toFixed(d) : String(v));
const pct = (x, y) => (Number.isFinite(x) && Number.isFinite(y) && x !== 0 ? `${y >= x ? '+' : ''}${(100 * (y / x - 1)).toFixed(1)}%` : '');
const row = (cells) => `| ${cells.join(' | ')} |`;

console.log(`Before: ${a.app.version} (${a.app.commit}) on ${a.adapter?.info?.description || '?'} ${a.adapter?.info?.backend || ''}`);
console.log(`After:  ${b.app.version} (${b.app.commit}) on ${b.adapter?.info?.description || '?'} ${b.adapter?.info?.backend || ''}\n`);

// ---- validation
console.log('## Validation (forces and coefficients; ± is the 95% interval)\n');
console.log(row(['Case', 'Variant', 'Reference', 'Before Cd / Cl', 'After Cd / Cl', 'Before drag N', 'After drag N', 'After steps', 'After s']));
console.log(row(Array(9).fill('---')));
const key = (r) => `${r.id}|${r.alpha ?? ''}|${r.variant}`;
const before = new Map(a.validation.map((r) => [key(r), r]));
for (const r of b.validation) {
  const o = before.get(key(r)) || {};
  const label = `${r.id}${r.alpha !== null && r.alpha !== undefined ? ` α=${r.alpha}` : ''}`;
  if (r.clSlopePerDeg !== undefined) {
    console.log(row([label, r.variant, `slope ${fmt(r.reference)}/°`, `slope ${fmt(o.clSlopePerDeg, 4)}`, `slope ${fmt(r.clSlopePerDeg, 4)}`, '', '', '', '']));
    continue;
  }
  const rf = r.reference || {};
  const ref = r.cfReference ? `Cf ${fmt(r.cfReference, 5)}` : rf.cd ? `Cd ${fmt(rf.cd, 3)}` : rf.cl ? `Cl ${fmt(rf.cl, 2)}` : '';
  const coef = (x) => (x.skipped ? 'n/a' : x.error ? 'failed' : x.cf !== undefined && x.cf !== null ? `Cf ${fmt(x.cf, 5)}` : `${fmt(x.cd)} ± ${fmt(x.cdCI, 3)} / ${fmt(x.cl)}`);
  console.log(row([label, r.variant, ref, o.id ? coef(o) : '–', coef(r), fmt(o.drag), fmt(r.drag), fmt(r.steps, 0), fmt(r.seconds, 0)]));
}

// ---- speed
console.log('\n## Speed ladder (kernel = GPU time from timestamp queries)\n');
console.log(row(['Cells', 'Before kernel MLUPS', 'After kernel MLUPS', 'Change', 'After GB/s', 'After % of peak', 'After bytes/cell step', 'After GPU busy']));
console.log(row(Array(8).fill('---')));
const ladderB = new Map(a.ladder.filter((r) => !r.error).map((r) => [Math.round(r.cells / 1e6), r]));
for (const r of b.ladder.filter((x) => !x.error)) {
  const o = ladderB.get(Math.round(r.cells / 1e6)) || {};
  console.log(row([`${(r.cells / 1e6).toFixed(1)}M`, fmt(o.kernelMLUPS, 0), fmt(r.kernelMLUPS, 0), pct(o.kernelMLUPS, r.kernelMLUPS), fmt(r.gbs, 0), fmt(r.percentOfPeak, 0), fmt(r.bytesPerCellStep, 0), fmt(r.busyPercent, 0)]));
}
const largest = (x) => Math.max(0, ...x.ladder.filter((r) => !r.error).map((r) => r.cells));
console.log(`\nLargest grid run: before ${(largest(a) / 1e6).toFixed(0)}M cells, after ${(largest(b) / 1e6).toFixed(0)}M cells.`);

console.log('\n## Whole app (the airflow loop with particles, averaging and forces)\n');
console.log(row(['Sample, cells', 'Before MLUPS', 'After MLUPS', 'Before GPU busy', 'After GPU busy', 'After kernel MLUPS']));
console.log(row(Array(6).fill('---')));
for (const r of b.appLoop || []) {
  const o = (a.appLoop || []).find((x) => x.sample === r.sample && Math.abs(Math.log(x.cells / r.cells)) < 0.3) || {};
  console.log(row([`${r.sample} ${(r.cells / 1e6).toFixed(1)}M`, fmt(o.wallMLUPS, 0), fmt(r.wallMLUPS, 0), `${fmt(o.busyPercent, 0)}%`, `${fmt(r.busyPercent, 0)}%`, fmt(r.kernelMLUPS, 0)]));
}

const sus = (x) => (x.sustained ? `${fmt(x.sustained.firstMinuteMLUPS, 0)} → ${fmt(x.sustained.restMLUPS, 0)} MLUPS (${fmt(x.sustained.dropPercent, 1)}% lower) at ${(x.sustained.cells / 1e6).toFixed(0)}M` : '–');
console.log(`\nSustained: before ${sus(a)}; after ${sus(b)}.`);
console.log(`Errors: before ${a.errors.length}, after ${b.errors.length}.${b.errors.length ? `\n${b.errors.map((e) => `  - ${e.where}: ${e.message}`).join('\n')}` : ''}`);
const doublings = (x) => x.validation.reduce((s, r) => s + (r.viscosityDoublings || 0), 0);
console.log(`Viscosity doublings: before ${doublings(a)}, after ${doublings(b)}.`);
