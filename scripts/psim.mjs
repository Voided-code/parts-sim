// Inspect, check and convert .psim files.
//
//   node scripts/psim.mjs inspect <file>          sections, sizes, info and error bounds
//   node scripts/psim.mjs verify <file>           reads every section; exit status 1 when the file is damaged
//   node scripts/psim.mjs to-stl <file> [out.stl] the part as binary STL
//   node scripts/psim.mjs forces <file> [out.csv] forces and results as CSV
//   node scripts/psim.mjs fixtures                writes test/fixtures/psim/*.psim (one per format version)
//   node scripts/psim.mjs dump <file>             a stable line-based text of the decoded content; native/build/psim_tool dump
//                                                 prints the same text, so the two apps can be compared with diff
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

// ---- dump: the same text as native/tools/psim_tool.cpp (numbers as C's %.9g)

function fmtG(v) {
  if (Number.isNaN(v)) return 'nan';
  if (v === Infinity) return 'inf';
  if (v === -Infinity) return '-inf';
  if (v === 0) return Object.is(v, -0) ? '-0' : '0';
  const [mant, exp] = v.toExponential(8).split('e');
  const x = Number(exp);
  if (x < -4 || x >= 9) {
    const m = mant.includes('.') ? mant.replace(/0+$/, '').replace(/\.$/, '') : mant;
    return `${m}e${x < 0 ? '-' : '+'}${String(Math.abs(x)).padStart(2, '0')}`;
  }
  const f = v.toFixed(8 - x);
  return f.includes('.') ? f.replace(/0+$/, '').replace(/\.$/, '') : f;
}

const hex8 = (n) => (n >>> 0).toString(16).padStart(8, '0');
const quote = (s) => `"${s.replace(/[\\"\n\r\t\u0000-\u001f]/g, (c) => (c === '"' ? '\\"' : c === '\\' ? '\\\\' : c === '\n' ? '\\n' : c === '\r' ? '\\r' : c === '\t' ? '\\t' : `\\u${c.charCodeAt(0).toString(16).padStart(4, '0')}`))}"`;

function fnvOf(words) {
  let h = 0x811c9dc5;
  for (let i = 0; i < words.length; i++) h = Math.imul(h ^ words[i], 16777619) >>> 0;
  return h;
}

/** Bit patterns of floats with one not-a-number. */
function floatWords(f32) {
  const u = new Uint32Array(f32.buffer, f32.byteOffset, f32.length);
  const out = new Uint32Array(f32.length);
  for (let i = 0; i < f32.length; i++) out[i] = Number.isNaN(f32[i]) ? 0x7fc00000 : u[i];
  return out;
}

function stats(values) {
  let mn = 0, mx = 0, sum = 0, nonfinite = 0, first = true;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) { nonfinite++; continue; }
    if (first) { mn = mx = v; first = false; }
    if (v < mn) mn = v;
    if (v > mx) mx = v;
    sum += v;
  }
  return { mn, mx, sum, nonfinite };
}

function flatten(lines, prefix, v) {
  if (v === null || v === undefined) lines.push(`${prefix} = null`);
  else if (typeof v === 'boolean') lines.push(`${prefix} = ${v}`);
  else if (typeof v === 'number') lines.push(`${prefix} = ${fmtG(v)}`);
  else if (typeof v === 'string') lines.push(`${prefix} = ${quote(v)}`);
  else if (Array.isArray(v)) {
    if (!v.length) lines.push(`${prefix} = []`);
    v.forEach((x, i) => flatten(lines, `${prefix}[${i}]`, x));
  } else {
    const keys = Object.keys(v).sort();
    if (!keys.length) lines.push(`${prefix} = {}`);
    for (const k of keys) flatten(lines, `${prefix}.${k}`, v[k]);
  }
}

function dumpArrays(lines, tag, r) {
  flatten(lines, `${tag}.meta`, r.meta);
  for (const [name, a] of r.arrays) {
    const s = stats(a.data);
    const words = a.enc === 'i32' ? new Uint32Array(a.data.buffer, a.data.byteOffset, a.data.length) : a.enc === 'u8' ? a.data : floatWords(a.data);
    lines.push(`${tag}.array ${name} enc=${a.enc} n=${a.data.length} dims=${a.dims ? a.dims.join(',') : '-'} min=${fmtG(s.mn)} max=${fmtG(s.mx)} sum=${fmtG(s.sum)} nonfinite=${s.nonfinite} fnv=${hex8(fnvOf(words))}`);
  }
}

function dumpText(f) {
  const lines = ['psim-dump 1', `format ${f.version} ${f.minReader}`];
  for (const e of f.table) lines.push(`table ${e.id} codec=${e.codec} flags=${e.flags} raw=${e.raw} crc=${hex8(e.crc)}`);
  if (f.skipped.length) lines.push(`skipped ${f.skipped.join(',')}`);
  flatten(lines, 'info', f.info);
  if (f.thumb) lines.push(`thumb bytes=${f.thumb.length} fnv=${hex8(fnvOf(f.thumb))}`);
  if (f.geometry) {
    const g = f.geometry;
    const nV = g.vertices.length / 3;
    lines.push(`geom.mode = ${g.exact ? 'exact' : 'quantised16'}`, `geom.brep = ${g.brepFaces ? 1 : 0}`, `geom.vertices = ${nV}`, `geom.triangles = ${g.tris.length / 3}`, `geom.faceCount = ${g.faceCount}`, `geom.faceAngle = ${fmtG(g.faceAngle)}`);
    lines.push(`geom.bbox.min = ${g.bbox.min.map(fmtG).join(' ')}`, `geom.bbox.max = ${g.bbox.max.map(fmtG).join(' ')}`);
    for (let d = 0; d < 3; d++) {
      const axis = new Float32Array(nV);
      for (let i = 0; i < nV; i++) axis[i] = g.vertices[3 * i + d];
      const s = stats(axis);
      lines.push(`geom.pos.${'xyz'[d]} = min=${fmtG(s.mn)} max=${fmtG(s.mx)} sum=${fmtG(s.sum)}`);
    }
    lines.push(`geom.vertexFnv = ${hex8(fnvOf(floatWords(g.vertices)))}`, `geom.triangleFnv = ${hex8(fnvOf(g.tris))}`);
    if (g.brepFaces) lines.push(`geom.faceFnv = ${hex8(fnvOf(new Uint32Array(g.faceOf.buffer, g.faceOf.byteOffset, g.faceOf.length)))}`);
  }
  if (f.cad) lines.push(`cad bytes=${f.cad.length} fnv=${hex8(fnvOf(f.cad))}`);
  if (f.setup) flatten(lines, 'setup', f.setup);
  if (f.rfea) dumpArrays(lines, 'rfea', f.rfea);
  if (f.rair) dumpArrays(lines, 'rair', f.rair);
  if (f.view) flatten(lines, 'view', f.view);
  return lines.join('\n') + '\n';
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
  } else if (command === 'dump') {
    process.stdout.write(dumpText(await readPsim(load(file))));
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
    console.log('usage: node scripts/psim.mjs inspect|verify|dump|to-stl|forces|fixtures <file> [out]');
    process.exit(command ? 2 : 0);
  }
}

main().catch((err) => {
  console.error(err instanceof PsimError ? `${file ?? ''}: ${err.message}` : err);
  process.exit(1);
});
