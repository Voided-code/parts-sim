// Inspect, check and convert .psim files.
//
//   node scripts/psim.mjs inspect <file>          sections, sizes, info and error bounds
//   node scripts/psim.mjs verify <file>           reads every section; exit status 1 when the file is damaged
//   node scripts/psim.mjs to-stl <file> [out.stl] the part as binary STL
//   node scripts/psim.mjs forces <file> [out.csv] forces and results as CSV
//   node scripts/psim.mjs fixtures                writes test/fixtures/psim/*.psim (one per format version)
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PsimError, inspectPsim, readPsim, writePsim } from '../src/core/psim.js';
import { CREATED, sampleContent } from '../test/helpers/psim-content.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const [command, file, out] = process.argv.slice(2);
const load = (f) => new Uint8Array(readFileSync(f));
const kb = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)} MB` : `${(n / 1e3).toFixed(1)} KB`);

function csvCell(v) {
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

async function main() {
  if (command === 'inspect') {
    const bytes = load(file);
    const i = await inspectPsim(bytes);
    console.log(`${file}: ${kb(bytes.length)}, format ${i.version} (reader needs ${i.minReader})`);
    console.log('section  codec  stored      raw         ratio');
    for (const e of i.table) {
      console.log(`${e.id}     ${e.codec === 0 ? 'stored' : e.codec === 1 ? 'deflate' : e.codec}  ${kb(e.stored).padEnd(10)}  ${kb(e.raw).padEnd(10)}  ${(e.raw / Math.max(1, e.stored)).toFixed(1)}${e.flags & 1 ? '  lossy' : ''}`);
    }
    console.log(JSON.stringify(i.info, null, 2));
    if (i.thumb) console.log(`thumbnail: ${i.thumb.length} bytes`);
  } else if (command === 'verify') {
    const bytes = load(file);
    const f = await readPsim(bytes);
    const parts = [];
    if (f.geometry) parts.push(`geometry ${f.geometry.vertices.length / 3} vertices, ${f.geometry.tris.length / 3} triangles`);
    if (f.setup) parts.push('setup');
    for (const k of ['rfea', 'rair']) if (f[k]) parts.push(`${k === 'rfea' ? 'structural/thermal' : 'airflow'} results (${f[k].arrays.size} arrays)`);
    console.log(`${file}: OK, ${parts.join(', ') || 'info only'}${f.skipped.length ? `; skipped unknown sections ${f.skipped.join(', ')}` : ''}`);
  } else if (command === 'to-stl') {
    const f = await readPsim(load(file), { want: ['INFO', 'GEOM'] });
    if (!f.geometry) throw new PsimError('The file has no geometry.');
    const { vertices: V, tris: T } = f.geometry;
    const n = T.length / 3;
    const buf = Buffer.alloc(84 + 50 * n);
    buf.write(`Parts Sim ${f.info.name ?? ''}`.slice(0, 79), 0, 'latin1');
    buf.writeUInt32LE(n, 80);
    for (let t = 0; t < n; t++) {
      const o = 84 + 50 * t;
      const p = [0, 1, 2].map((k) => T[3 * t + k] * 3);
      const e1 = [0, 1, 2].map((d) => V[p[1] + d] - V[p[0] + d]);
      const e2 = [0, 1, 2].map((d) => V[p[2] + d] - V[p[0] + d]);
      const nx = e1[1] * e2[2] - e1[2] * e2[1], ny = e1[2] * e2[0] - e1[0] * e2[2], nz = e1[0] * e2[1] - e1[1] * e2[0];
      const l = Math.hypot(nx, ny, nz) || 1;
      [nx / l, ny / l, nz / l].forEach((v, k) => buf.writeFloatLE(v, o + 4 * k));
      for (let k = 0; k < 3; k++) for (let d = 0; d < 3; d++) buf.writeFloatLE(V[p[k] + d], o + 12 + 12 * k + 4 * d);
    }
    const target = out ?? file.replace(/\.psim$/i, '') + '.stl';
    writeFileSync(target, buf);
    console.log(`${target}: ${n} triangles, ${kb(buf.length)}`);
  } else if (command === 'forces') {
    const f = await readPsim(load(file), { want: ['INFO', 'RFEA', 'RAIR'] });
    const rows = [['study', 'quantity', 'value', 'unit']];
    const flat = (prefix, o) => {
      for (const [k, v] of Object.entries(o ?? {})) {
        if (v && typeof v === 'object' && !Array.isArray(v)) flat(`${prefix}.${k}`, v);
        else if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') rows.push([prefix.split('.')[0], `${prefix}.${k}`.split('.').slice(1).join('.'), v, '']);
      }
    };
    if (f.rair) flat('airflow', f.rair.meta.airflow);
    if (f.rfea) for (const s of f.rfea.meta.results ?? []) flat(s, f.rfea.meta[s]);
    const csv = rows.map((r) => r.map(csvCell).join(',')).join('\n') + '\n';
    if (out) { writeFileSync(out, csv); console.log(`${out}: ${rows.length - 1} values`); } else process.stdout.write(csv);
  } else if (command === 'fixtures') {
    const dir = resolve(root, 'test/fixtures/psim');
    mkdirSync(dir, { recursive: true });
    const bytes = await writePsim(sampleContent(), { created: CREATED });
    writeFileSync(resolve(dir, 'v1-tube.psim'), bytes);
    console.log(`test/fixtures/psim/v1-tube.psim: ${bytes.length} bytes`);
  } else {
    console.log('usage: node scripts/psim.mjs inspect|verify|to-stl|forces|fixtures <file> [out]');
    process.exit(command ? 2 : 0);
  }
}

main().catch((err) => {
  console.error(err instanceof PsimError ? `${file ?? ''}: ${err.message}` : err);
  process.exit(1);
});
