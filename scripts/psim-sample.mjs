// Builds the built-in sample parts like the app does (src/core/mesh.js buildPart, src/core/samples.js) and writes each
// as a .psim with its geometry, its setup and a synthetic static result, for testing and timing other readers and writers.
//
//   node scripts/psim-sample.mjs <outdir> [sampleId ...]      writes <outdir>/<id>.psim for the given samples (default: all)
//
// The "result" is not a solved one: smooth functions of position per vertex, so the numbers are real-looking fields of
// the real mesh size (vm, p1, p3, fos in q16; displacement u, three values per vertex).
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { buildPart } from '../src/core/mesh.js';
import { SAMPLES } from '../src/core/samples.js';
import { readPsim, writePsim } from '../src/core/psim.js';

const [outDir, ...ids] = process.argv.slice(2);
if (!outDir) {
  console.log('usage: node scripts/psim-sample.mjs <outdir> [sampleId ...]');
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

/** Sorted triangle numbers as runs [start, count, ...]. */
function runs(tris) {
  const out = [];
  const sorted = Array.from(tris).sort((a, b) => a - b);
  for (let i = 0; i < sorted.length;) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1] === sorted[j] + 1) j++;
    out.push(sorted[i], j - i + 1);
    i = j + 1;
  }
  return out;
}

const ms = (t0) => (performance.now() - t0).toFixed(0);

for (const s of SAMPLES.filter((x) => !ids.length || ids.includes(x.id))) {
  const t0 = performance.now();
  const part = buildPart({ ...s.make(), name: s.id });
  const tBuild = ms(t0);
  const setup = s.setup(part);
  const nV = part.vertices.length / 3;
  const { min, max, diag } = part.bbox;
  const vm = new Float32Array(nV), p1 = new Float32Array(nV), p3 = new Float32Array(nV), fos = new Float32Array(nV), u = new Float32Array(3 * nV);
  for (let i = 0; i < nV; i++) {
    const x = (part.vertices[3 * i] - min[0]) / diag, y = (part.vertices[3 * i + 1] - min[1]) / diag, z = (part.vertices[3 * i + 2] - min[2]) / diag;
    const stress = 1e8 * (0.3 + 0.7 * Math.abs(Math.sin(9 * x + 3 * y) * Math.cos(7 * z + x)));
    vm[i] = stress;
    p1[i] = 0.6 * stress * (1 + 0.2 * Math.sin(5 * y));
    p3[i] = -0.5 * stress * (1 + 0.2 * Math.cos(6 * z));
    fos[i] = 3.5e8 / Math.max(stress, 1);
    u[3 * i] = 2e-3 * x * x;
    u[3 * i + 1] = -3e-3 * x * (1 + 0.1 * Math.sin(8 * z));
    u[3 * i + 2] = 5e-4 * Math.sin(4 * x + y);
  }
  const content = {
    info: {
      app: { name: 'Parts Sim', version: '1.1.0', kind: 'web' },
      name: s.name,
      notes: s.note,
      units: 'mm',
      contains: { geometry: 'quantised16', setup: true, results: ['static'], cad: false },
      part: { vertices: nV, triangles: part.nTri, bbox: { min: Array.from(min), max: Array.from(max) } },
    },
    geometry: { vertices: part.vertices, tris: part.tris, faceOf: part.brepFaces ? part.faceOf : null, brepFaces: part.brepFaces, faceCount: part.faceCount, faceAngle: part.brepFaces ? 0 : 20 },
    setup: {
      units: 'mm',
      structural: {
        study: 'static',
        resolution: 56,
        gravity: false,
        fixtures: (setup.fixtures ?? []).map((f) => ({ name: f.name, patches: f.patches.map((p) => ({ tris: runs(p.tris), clip: p.clip })) })),
        loads: (setup.loads ?? []).map((l) => ({ name: l.name, type: l.type, magnitude: l.magnitude, dir: l.dir, patches: l.patches.map((p) => ({ tris: runs(p.tris), clip: p.clip })) })),
      },
    },
    rfea: {
      meta: { results: ['static'], static: { maxVM: vm.reduce((a, b) => Math.max(a, b), 0), lamBreak: Infinity, minFos: fos.reduce((a, b) => Math.min(a, b), Infinity), resolution: 56 } },
      arrays: [
        { name: 'static.vm', data: vm, enc: 'q16' },
        { name: 'static.p1', data: p1, enc: 'q16' },
        { name: 'static.p3', data: p3, enc: 'q16' },
        { name: 'static.fos', data: fos, enc: 'q16' },
        { name: 'static.u', data: u, enc: 'q16' },
      ],
    },
    view: { tab: 'structural', study: 'static', plot: 'vm' },
  };
  const t1 = performance.now();
  const bytes = await writePsim(content, { created: '2026-10-04T00:00:00Z' });
  const tWrite = ms(t1);
  const t2 = performance.now();
  await readPsim(bytes);
  const tRead = ms(t2);
  const file = resolve(outDir, `${s.id}.psim`);
  writeFileSync(file, bytes);
  console.log(`${file}: ${nV} vertices, ${part.nTri} triangles, ${bytes.length} bytes; JS build ${tBuild} ms, write ${tWrite} ms, read ${tRead} ms`);
}
