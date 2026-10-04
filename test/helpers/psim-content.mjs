// Small deterministic content for the .psim tests and fixtures.

export const CREATED = '2026-10-04T00:00:00Z';

// a small deterministic generator, so failures repeat
export function rng(seed) {
  let s = seed >>> 0;
  return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; };
}

/** A closed grid surface: nu x nv vertices wrapped into a torus-like tube, with its triangles. */
export function tube(nu = 40, nv = 24) {
  const vertices = new Float32Array(3 * nu * nv);
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const a = (2 * Math.PI * i) / nu, b = (2 * Math.PI * j) / nv;
    const o = 3 * (i * nv + j);
    vertices[o] = (50 + 10 * Math.cos(b)) * Math.cos(a);
    vertices[o + 1] = 10 * Math.sin(b) + 12;
    vertices[o + 2] = (50 + 10 * Math.cos(b)) * Math.sin(a);
  }
  const tris = [];
  for (let i = 0; i < nu; i++) for (let j = 0; j < nv; j++) {
    const a = i * nv + j, b = ((i + 1) % nu) * nv + j, c = ((i + 1) % nu) * nv + ((j + 1) % nv), d = i * nv + ((j + 1) % nv);
    tris.push(a, b, c, a, c, d);
  }
  return { vertices, tris: Uint32Array.from(tris), nV: nu * nv };
}

export function sampleContent({ brep = false } = {}) {
  const t = tube();
  const faceOf = new Int32Array(t.tris.length / 3).map((_, k) => (brep ? Math.floor(k / 100) : 0));
  const geometry = { vertices: t.vertices, tris: t.tris, faceOf: brep ? faceOf : null, brepFaces: brep, faceCount: brep ? Math.floor((faceOf.length - 1) / 100) + 1 : 5, faceAngle: 20 };
  const vm = Float32Array.from({ length: t.nV }, (_, i) => 1e6 + 5e5 * Math.sin(i / 7));
  const u = Float32Array.from({ length: 3 * t.nV }, (_, i) => 0.01 * Math.cos(i / 11));
  return {
    info: { app: { name: 'Parts Sim', version: '1.1.0', kind: 'web' }, name: 'Tube', notes: 'n', units: 'mm', contains: { geometry: 'quantised16', setup: true, results: ['static'], cad: false } },
    thumb: Uint8Array.of(0xff, 0xd8, 0xff, 0xd9),
    geometry,
    setup: { units: 'mm', structural: { study: 'static', fixtures: [{ name: 'F', patches: [{ tris: [0, 12], clip: null }] }] } },
    rfea: { meta: { results: ['static'], static: { maxVM: 1.5e6, lamBreak: Infinity, minFos: NaN } }, arrays: [{ name: 'static.vm', data: vm, enc: 'q16' }, { name: 'static.u', data: u, enc: 'q16', stride: 3 }] },
    rair: { meta: { airflow: { cd: 0.3 } }, arrays: [{ name: 'airflow.avg.ux', data: Float32Array.from({ length: 6 * 5 * 4 }, (_, i) => Math.sin(i / 9)), enc: 'q16', dims: [6, 5, 4] }] },
    view: { tab: 'structural' },
  };
}

