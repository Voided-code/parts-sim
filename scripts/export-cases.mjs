// Writes the airflow validation cases (src/cfd/validation.js) for the native benchmark: one binary
// STL per case (and per angle of attack) and cases.json with the tunnel and reference data, so both
// apps run the very same meshes.
//   node scripts/export-cases.mjs [output folder]   (default: test-artifacts/cases)
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { CASES } from '../src/cfd/validation.js';

const root = path.resolve(import.meta.dirname, '..');
const out = path.resolve(process.argv[2] || path.join(root, 'test-artifacts', 'cases'));
await mkdir(out, { recursive: true });

function stl(positions) {
  const n = positions.length / 9;
  const buf = Buffer.alloc(84 + 50 * n);
  buf.write('parts-sim validation case (mm)', 0, 'ascii');
  buf.writeUInt32LE(n, 80);
  for (let t = 0; t < n; t++) {
    const o = 84 + 50 * t;
    const p = (v, a) => positions[9 * t + 3 * v + a];
    const u = [0, 1, 2].map((a) => p(1, a) - p(0, a)), w = [0, 1, 2].map((a) => p(2, a) - p(0, a));
    const nn = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
    const l = Math.hypot(...nn) || 1;
    for (let a = 0; a < 3; a++) buf.writeFloatLE(nn[a] / l, o + 4 * a);
    for (let v = 0; v < 3; v++) for (let a = 0; a < 3; a++) buf.writeFloatLE(p(v, a), o + 12 + 12 * v + 4 * a);
  }
  return buf;
}

const cases = [];
for (const c of CASES) {
  const meshes = {};
  for (const alpha of c.alphas || [null]) {
    const file = alpha === null ? `${c.id}.stl` : `${c.id}-a${alpha}.stl`;
    await writeFile(path.join(out, file), stl(alpha === null ? c.make() : c.make(alpha)));
    meshes[alpha === null ? 'default' : String(alpha)] = file;
  }
  cases.push({
    id: c.id, name: c.name, note: c.note, meshes, alphas: c.alphas || null,
    speed: c.speed, lref: c.lref, aref: c.aref, across: c.across, spanCells: c.spanCells || 0,
    tunnel: c.tunnel || {}, friction: !!c.friction, perSpan: !!c.perSpan, ref: c.ref,
  });
}
await writeFile(path.join(out, 'cases.json'), JSON.stringify({ format: 'parts-sim-cases/1', cases }, null, 1));
console.log(`${cases.length} cases in ${out}`);
