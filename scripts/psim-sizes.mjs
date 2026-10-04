// The size ladder: what each step of the .psim coding saves, on real parts and real results.
//   node scripts/psim-sizes.mjs [file.psim ...]
// Geometry: the seven sample parts, built as the app builds them. Fields: the arrays of the given .psim files.
// Every number is bytes; "deflated" is deflate-raw level 6 (what the file uses).
import { readFileSync } from 'node:fs';
import { deflateRaw, putPlanes, predictEncode, readPsim, encodeGeometry } from '../src/core/psim.js';
import { SAMPLES } from '../src/core/samples.js';
import { buildPart } from '../src/core/mesh.js';

const z = async (b) => (await deflateRaw(b)).length;
const kb = (n) => (n / 1024).toFixed(n > 1e6 ? 0 : 1).padStart(9);
const row = (cols) => console.log(cols.join(' '));
const bytesOf = (typed) => new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);

async function geometryLadder() {
  console.log('\nGeometry (KiB). Binary STL of the same built part = 84 + 50 bytes per triangle.');
  row(['part'.padEnd(10), 'verts'.padStart(8), 'tris'.padStart(8), 'STL'.padStart(9), 'JSON'.padStart(9), 'binary'.padStart(9), '+quantise'.padStart(9), '+predict'.padStart(9), '+shuffle'.padStart(9), '+deflate'.padStart(9), 'exact+defl'.padStart(10), 'quant %STL', 'exact %STL']);
  for (const s of SAMPLES) {
    const p = buildPart({ ...s.make(), name: s.name });
    const stl = 84 + 50 * p.nTri;
    const json = new TextEncoder().encode(JSON.stringify({ v: Array.from(p.vertices), t: Array.from(p.tris) })).length;
    const binary = 4 * p.vertices.length + 4 * p.tris.length;
    const quant = 2 * p.vertices.length + 4 * p.tris.length;
    // positions: 16-bit codes as they come, then delta coded (still interleaved), then byte planes (the format)
    const g = { vertices: p.vertices, tris: p.tris, faceOf: null, brepFaces: false, faceCount: p.faceCount, faceAngle: 20 };
    const predictOnly = await stagesWithoutShuffle(p);
    const q = encodeGeometry(g, 'quantised').bytes;
    const e = encodeGeometry(g, 'exact').bytes;
    const [qz, ez] = [await z(q), await z(e)];
    row([s.id.padEnd(10), String(p.nVert).padStart(8), String(p.nTri).padStart(8), kb(stl), kb(json), kb(binary), kb(quant), kb(predictOnly), kb(q.length), kb(qz), kb(ez), `${((100 * qz) / stl).toFixed(1)}%`.padStart(10), `${((100 * ez) / stl).toFixed(1)}%`.padStart(10)]);
  }
}

/** Deflated size of the geometry coded with delta prediction but without the byte-plane split. */
async function stagesWithoutShuffle(p) {
  const nV = p.nVert, nT = p.nTri;
  const out = new Uint16Array(3 * nV + 2 * 3 * nT);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nV; i++) for (let d = 0; d < 3; d++) { min[d] = Math.min(min[d], p.vertices[3 * i + d]); max[d] = Math.max(max[d], p.vertices[3 * i + d]); }
  let o = 0;
  for (let i = 0; i < nV; i++) {
    for (let d = 0; d < 3; d++) {
      const span = max[d] - min[d];
      const q = span > 0 ? Math.round(((p.vertices[3 * i + d] - min[d]) / span) * 65535) : 0;
      const prev = i ? Math.round(((p.vertices[3 * (i - 1) + d] - min[d]) / span) * 65535) : 0;
      let r = (q - prev) & 0xffff;
      if (r >= 32768) r -= 65536;
      out[o++] = r >= 0 ? 2 * r : -2 * r - 1;
    }
  }
  const t32 = new Uint32Array(3 * nT);
  let prev = [0, 0, 0];
  for (let t = 0; t < nT; t++) for (let k = 0; k < 3; k++) { const c = p.tris[3 * t + k]; const d = (c - prev[k]) | 0; t32[3 * t + k] = ((d << 1) ^ (d >> 31)) >>> 0; prev[k] = c; }
  const buf = new Uint8Array(out.byteLength + t32.byteLength);
  buf.set(bytesOf(out), 0);
  buf.set(bytesOf(t32), out.byteLength);
  return z(buf);
}

async function fieldLadder(file) {
  const f = await readPsim(new Uint8Array(readFileSync(file)), { want: ['INFO', 'RFEA', 'RAIR'] });
  const sets = [];
  if (f.rfea) sets.push(['structural/thermal', f.rfea.arrays]);
  if (f.rair) sets.push(['airflow', f.rair.arrays]);
  for (const [label, arrays] of sets) {
    console.log(`\nFields in ${file} (${label}), KiB`);
    row(['array'.padEnd(24), 'n'.padStart(9), 'JSON'.padStart(9), 'float32'.padStart(9), 'f32 defl'.padStart(9), 'q16'.padStart(9), 'q16 defl'.padStart(9), '+predict'.padStart(9), '+shuffle'.padStart(9)]);
    let tot = { json: 0, f32: 0, f32z: 0, q: 0, qz: 0, pz: 0, sz: 0 };
    for (const [name, a] of arrays) {
      if (!(a.data instanceof Float32Array)) continue;
      const n = a.data.length;
      const json = JSON.stringify(Array.from(a.data)).length;
      const f32z = await z(bytesOf(a.data));
      let min = Infinity, max = -Infinity;
      for (const v of a.data) if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
      const span = max - min;
      const codes = new Uint16Array(n);
      for (let i = 0; i < n; i++) codes[i] = Number.isFinite(a.data[i]) ? (span > 0 ? 1 + Math.round(((a.data[i] - min) / span) * 65534) : 1) : 0;
      const qz = await z(bytesOf(codes));
      const res = predictEncode(codes, 65536, a.dims);
      const pz = await z(bytesOf(res));
      const planes = new Uint8Array(2 * n);
      putPlanes(planes, 0, res, 2);
      const sz = await z(planes);
      row([name.padEnd(24), String(n).padStart(9), kb(json), kb(4 * n), kb(f32z), kb(2 * n), kb(qz), kb(pz), kb(sz)]);
      tot = { json: tot.json + json, f32: tot.f32 + 4 * n, f32z: tot.f32z + f32z, q: tot.q + 2 * n, qz: tot.qz + qz, pz: tot.pz + pz, sz: tot.sz + sz };
    }
    row(['total'.padEnd(24), ''.padStart(9), kb(tot.json), kb(tot.f32), kb(tot.f32z), kb(tot.q), kb(tot.qz), kb(tot.pz), kb(tot.sz)]);
  }
}

await geometryLadder();
for (const file of process.argv.slice(2)) await fieldLadder(file);
