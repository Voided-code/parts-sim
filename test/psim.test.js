import assert from 'node:assert/strict';
import { test } from 'node:test';
import {
  MAGIC, PsimError, LIMITS, crc32, deflateRaw, encodeArrays, decodeArrays, encodeGeometry, decodeGeometry,
  writePsim, readPsim, readTable, inspectPsim, stringifyJson, parseJson,
} from '../src/core/psim.js';

import { rng, tube, sampleContent, CREATED } from './helpers/psim-content.mjs';

test('crc32 matches the known value', () => {
  assert.equal(crc32(new TextEncoder().encode('123456789')), 0xcbf43926);
});

test('json keeps infinity and not-a-number', () => {
  const back = parseJson(stringifyJson({ a: Infinity, b: -Infinity, c: NaN, d: [1, { e: 2 }] }));
  assert.equal(back.a, Infinity);
  assert.equal(back.b, -Infinity);
  assert.ok(Number.isNaN(back.c));
  assert.deepEqual(back.d, [1, { e: 2 }]);
});

test('arrays: exact encodings round trip bit for bit', () => {
  const f = Float32Array.from([0, -0, 1.5, 3.4e38, -1e-30, NaN, Infinity]);
  const i = Int32Array.from([0, -1, 5, 2147483647, -2147483648]);
  const u = Uint8Array.from([0, 1, 255, 7]);
  const { bytes } = encodeArrays({ k: 1 }, [{ name: 'f', data: f, enc: 'f32' }, { name: 'i', data: i, enc: 'i32' }, { name: 'u', data: u, enc: 'u8' }]);
  const { meta, arrays } = decodeArrays(bytes);
  assert.equal(meta.k, 1);
  assert.deepEqual(Array.from(new Uint32Array(arrays.get('f').data.buffer)), Array.from(new Uint32Array(f.buffer)));
  assert.deepEqual(Array.from(arrays.get('i').data), Array.from(i));
  assert.deepEqual(Array.from(arrays.get('u').data), Array.from(u));
});

test('arrays: quantised values stay inside the stated error, NaN stays NaN', () => {
  const rand = rng(7);
  for (const enc of ['q16', 'q8']) {
    for (const dims of [null, [7, 5, 3]]) {
      const n = 105;
      const data = Float32Array.from({ length: n }, (_, k) => (k % 17 === 3 ? NaN : -40 + 90 * rand()));
      const { bytes, bounds } = encodeArrays({}, [{ name: 'a', data, enc, dims }]);
      const back = decodeArrays(bytes).arrays.get('a').data;
      const bound = bounds[0].absolute;
      for (let k = 0; k < n; k++) {
        if (Number.isNaN(data[k])) assert.ok(Number.isNaN(back[k]));
        else assert.ok(Math.abs(back[k] - data[k]) <= bound * (1 + 1e-6) + 1e-6, `${enc} ${k}: ${back[k]} vs ${data[k]} (bound ${bound})`);
      }
    }
  }
});

test('arrays: constant and all-NaN fields', () => {
  const { bytes } = encodeArrays({}, [{ name: 'c', data: new Float32Array(10).fill(3), enc: 'q16' }, { name: 'n', data: new Float32Array(4).fill(NaN), enc: 'q8' }]);
  const { arrays } = decodeArrays(bytes);
  assert.ok(arrays.get('c').data.every((v) => v === 3));
  assert.ok(arrays.get('n').data.every(Number.isNaN));
});

test('geometry: exact mode is bit for bit, quantised mode within its bound', () => {
  const t = tube();
  const g = { vertices: t.vertices, tris: t.tris, faceOf: null, brepFaces: false, faceCount: 4, faceAngle: 25 };
  const exact = decodeGeometry(encodeGeometry(g, 'exact').bytes);
  assert.deepEqual(Array.from(exact.vertices), Array.from(t.vertices));
  assert.deepEqual(Array.from(exact.tris), Array.from(t.tris));
  assert.equal(exact.faceAngle, 25);
  const q = encodeGeometry(g, 'quantised');
  const lossy = decodeGeometry(q.bytes);
  assert.deepEqual(Array.from(lossy.tris), Array.from(t.tris));
  let worst = 0;
  for (let i = 0; i < t.vertices.length; i++) worst = Math.max(worst, Math.abs(lossy.vertices[i] - t.vertices[i]));
  assert.ok(worst <= q.bounds.absolute * 1.0001, `${worst} > ${q.bounds.absolute}`);
});

test('geometry: CAD face ids round trip', () => {
  const c = sampleContent({ brep: true });
  const back = decodeGeometry(encodeGeometry(c.geometry, 'exact').bytes);
  assert.ok(back.brepFaces);
  assert.deepEqual(Array.from(back.faceOf), Array.from(c.geometry.faceOf));
  assert.equal(back.faceCount, c.geometry.faceCount);
});

test('a file with every section round trips and is deterministic', async () => {
  const c = sampleContent();
  const a = await writePsim(c, { created: CREATED });
  const b = await writePsim(c, { created: CREATED });
  assert.deepEqual(a, b, 'same content, same bytes');
  const f = await readPsim(a);
  assert.equal(f.info.name, 'Tube');
  assert.equal(f.info.created, CREATED);
  assert.equal(f.info.format, 1);
  assert.ok(f.info.bounds.some((x) => x.field === 'geometry.position'));
  assert.ok(f.info.bounds.some((x) => x.field === 'static.vm'));
  assert.deepEqual(Array.from(f.thumb), [0xff, 0xd8, 0xff, 0xd9]);
  assert.equal(f.geometry.tris.length, c.geometry.tris.length);
  assert.deepEqual(f.setup, c.setup);
  assert.equal(f.rfea.meta.static.lamBreak, Infinity);
  assert.ok(Number.isNaN(f.rfea.meta.static.minFos));
  assert.equal(f.rfea.arrays.get('static.vm').data.length, c.rfea.arrays[0].data.length);
  assert.equal(f.rair.arrays.get('airflow.avg.ux').dims.join(), '6,5,4');
  assert.deepEqual(f.view, c.view);
  assert.deepEqual(f.table.map((e) => e.id), ['INFO', 'THMB', 'GEOM', 'SETP', 'RFEA', 'RAIR', 'VIEW']);
});

test('inspect reads only the info and the thumbnail', async () => {
  const bytes = await writePsim(sampleContent(), { created: CREATED });
  const i = await inspectPsim(bytes);
  assert.equal(i.info.name, 'Tube');
  assert.equal(i.thumb.length, 4);
  assert.equal(i.size, bytes.length);
});

test('sections the reader does not know are skipped', async () => {
  const bytes = await writePsim(sampleContent(), { created: CREATED });
  // rename THMB to ZZZZ and fix the checksums
  const patched = bytes.slice();
  const dv = new DataView(patched.buffer);
  const n = dv.getUint32(16, true);
  for (let i = 0; i < n; i++) {
    const e = 32 + 24 * i;
    if (String.fromCharCode(...patched.subarray(e, e + 4)) === 'THMB') patched.set([0x5a, 0x5a, 0x5a, 0x5a], e);
  }
  reseal(patched);
  const f = await readPsim(patched);
  assert.deepEqual(f.skipped, ['ZZZZ']);
  assert.equal(f.thumb, null);
  assert.equal(f.info.name, 'Tube');
});

function reseal(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const n = dv.getUint32(16, true);
  let c = crc32(bytes, 0, 24);
  c = crc32(bytes, 32, Math.min(bytes.length - 8, 32 + 24 * n), c);
  dv.setUint32(24, c, true);
  dv.setUint32(bytes.length - 8, crc32(bytes, 0, bytes.length - 8), true);
}

async function expectPsimError(fn, pattern) {
  await assert.rejects(fn, (err) => {
    assert.ok(err instanceof PsimError, `expected a PsimError, got ${err?.stack ?? err}`);
    if (pattern) assert.match(err.message, pattern);
    return true;
  });
}

test('safety: wrong files and text-mode damage are named', async () => {
  await expectPsimError(() => readPsim(new Uint8Array(10)), /not a Parts Sim file/);
  await expectPsimError(() => readPsim(new TextEncoder().encode('x'.repeat(100))), /not a Parts Sim file/);
  const bytes = await writePsim(sampleContent(), { created: CREATED });
  const crlf = bytes.slice();
  crlf[5] = 0x0a; // CR LF turned into LF
  await expectPsimError(() => readPsim(crlf), /text-mode/);
});

test('safety: a newer file says so', async () => {
  const bytes = await writePsim(sampleContent(), { created: CREATED });
  new DataView(bytes.buffer).setUint16(12, 9, true);
  reseal(bytes);
  await expectPsimError(() => readPsim(bytes), /needs a newer Parts Sim/);
});

test('safety: every truncation is rejected', async () => {
  const bytes = await writePsim({ ...sampleContent(), rfea: undefined, rair: undefined }, { created: CREATED });
  const cuts = new Set([0, 1, 8, 9, 31, 32, 33, 55, bytes.length - 9, bytes.length - 8, bytes.length - 4, bytes.length - 1]);
  const rand = rng(3);
  while (cuts.size < 300) cuts.add(Math.floor(rand() * bytes.length));
  for (const cut of cuts) await expectPsimError(() => readPsim(bytes.subarray(0, cut)));
});

test('safety: every flipped bit is caught', async () => {
  const bytes = await writePsim({ ...sampleContent(), rfea: undefined, rair: undefined }, { created: CREATED });
  const rand = rng(11);
  for (let k = 0; k < 400; k++) {
    const copy = bytes.slice();
    const at = Math.floor(rand() * copy.length);
    copy[at] ^= 1 << Math.floor(rand() * 8);
    await expectPsimError(() => readPsim(copy));
  }
});

test('safety: header lies about counts and lengths (checksums repaired)', async () => {
  const bytes = await writePsim(sampleContent(), { created: CREATED });
  const lies = [
    (dv) => dv.setUint32(16, 0xffffffff, true), // section count
    (dv) => dv.setUint32(16, 3, true), // fewer sections than the data holds
    (dv) => dv.setUint32(20, 9999, true), // table offset
    (dv) => dv.setUint32(32 + 8, 40, true), // first section offset
    (dv) => dv.setUint32(32 + 12, 0xfffffff0, true), // stored length
    (dv) => dv.setUint32(32 + 16, 0xffffffff, true), // raw length
    (dv) => dv.setUint32(32 + 24 + 16, 0x7fffffff, true), // raw length of the second section
  ];
  for (const lie of lies) {
    const copy = bytes.slice();
    lie(new DataView(copy.buffer));
    reseal(copy);
    await expectPsimError(() => readPsim(copy));
  }
});

/** Builds a one-section container by hand, with whatever lengths the test wants. */
function handmade({ id = 'INFO', codec = 1, stored, raw, crc = 0 }) {
  const out = new Uint8Array(32 + 24 + stored.length + 8);
  const dv = new DataView(out.buffer);
  out.set(MAGIC, 0);
  dv.setUint16(10, 1, true);
  dv.setUint16(12, 1, true);
  dv.setUint32(16, 1, true);
  dv.setUint32(20, 32, true);
  for (let k = 0; k < 4; k++) out[32 + k] = id.charCodeAt(k);
  dv.setUint16(36, codec, true);
  dv.setUint32(40, 56, true);
  dv.setUint32(44, stored.length, true);
  dv.setUint32(48, raw, true);
  dv.setUint32(52, crc, true);
  out.set(stored, 56);
  dv.setUint32(out.length - 4, out.length, true);
  reseal(out);
  return out;
}

test('safety: a decompression bomb is stopped, not inflated', async () => {
  const zeros = new Uint8Array(64 * 2 ** 20); // 64 MiB of zeros deflate to about 64 KB
  const bomb = await deflateRaw(zeros);
  // honest about nothing: claims 1 MiB
  const lying = handmade({ stored: bomb, raw: 2 ** 20 });
  const t0 = performance.now();
  await expectPsimError(() => readPsim(lying), /larger than it says|bomb/);
  assert.ok(performance.now() - t0 < 5000);
  // claims the real size but a ratio no real file has
  const small = handmade({ stored: await deflateRaw(new Uint8Array(40 * 2 ** 20)), raw: 40 * 2 ** 20 });
  const ratio = (40 * 2 ** 20) / (small.length - 32 - 24 - 8);
  if (ratio > LIMITS.ratio) await expectPsimError(() => readPsim(small), /bomb|compressed far more/);
  // claims a section bigger than the cap
  await expectPsimError(() => readPsim(handmade({ stored: Uint8Array.of(0), raw: 0x7fffffff })), /more data than this reader accepts|bomb/);
});

test('safety: a section that is shorter or has a wrong checksum is rejected', async () => {
  const info = new TextEncoder().encode('{"format":1}');
  const z = await deflateRaw(info);
  await expectPsimError(() => readPsim(handmade({ stored: z, raw: info.length + 5, crc: crc32(info) })), /shorter|damaged/);
  await expectPsimError(() => readPsim(handmade({ stored: z, raw: info.length, crc: 1 })), /checksum/);
  const ok = await readPsim(handmade({ stored: z, raw: info.length, crc: crc32(info) }));
  assert.equal(ok.info.format, 1);
});

test('safety: oversized and inconsistent counts inside sections', async () => {
  const t = tube(6, 6);
  const g = encodeGeometry({ vertices: t.vertices, tris: t.tris, faceOf: null, brepFaces: false, faceCount: 1, faceAngle: 20 }, 'exact').bytes;
  const big = g.slice();
  new DataView(big.buffer).setUint32(4, 0xffffff00, true);
  assert.throws(() => decodeGeometry(big), PsimError);
  const off = g.slice();
  new DataView(off.buffer).setUint32(8, t.tris.length / 3 + 1, true);
  assert.throws(() => decodeGeometry(off), PsimError);
  // an index past the last vertex
  const bad = decodeGeometry(g);
  bad.tris[0] = 99999;
  assert.throws(() => decodeGeometry(encodeGeometry({ vertices: t.vertices, tris: bad.tris, faceOf: null, brepFaces: false, faceCount: 1, faceAngle: 20 }, 'exact').bytes), /vertex that does not exist/);
  // a degenerate triangle
  const deg = t.tris.slice();
  deg[1] = deg[0];
  assert.throws(() => decodeGeometry(encodeGeometry({ vertices: t.vertices, tris: deg, faceOf: null, brepFaces: false, faceCount: 1, faceAngle: 20 }, 'exact').bytes), /one vertex twice/);
  // arrays: header longer than the section, byte counts that lie
  const { bytes } = encodeArrays({}, [{ name: 'a', data: Float32Array.of(1, 2, 3), enc: 'q16' }]);
  const hl = bytes.slice();
  new DataView(hl.buffer).setUint32(0, 0xffffffff, true);
  assert.throws(() => decodeArrays(hl), PsimError);
  assert.throws(() => decodeArrays(bytes.subarray(0, bytes.length - 1)), PsimError);
  assert.throws(() => decodeArrays(bytes.subarray(0, 3)), PsimError);
});

test('fuzz: random damage never produces anything but a PsimError, and never hangs', async () => {
  const base = await writePsim(sampleContent(), { created: CREATED });
  const rand = rng(2026);
  const t0 = performance.now();
  let accepted = 0;
  for (let k = 0; k < 600; k++) {
    const copy = base.slice();
    const edits = 1 + Math.floor(rand() * 6);
    for (let e = 0; e < edits; e++) {
      const at = Math.floor(rand() * copy.length);
      const kind = rand();
      if (kind < 0.4) copy[at] = Math.floor(rand() * 256);
      else if (kind < 0.7) copy[at] ^= 1 << Math.floor(rand() * 8);
      else copy.fill(Math.floor(rand() * 256), at, Math.min(copy.length, at + 1 + Math.floor(rand() * 64)));
    }
    // half the time, make the checksums valid so the damage reaches the parsers
    if (rand() < 0.5 && copy.length > 40) {
      try { reseal(copy); } catch { /* the table is too damaged to seal */ }
    }
    try {
      await readPsim(copy);
      accepted++;
    } catch (err) {
      assert.ok(err instanceof PsimError, `fuzz ${k}: ${err?.stack ?? err}`);
    }
  }
  assert.ok(performance.now() - t0 < 25000, 'the fuzz run took too long');
  assert.ok(accepted < 600);
});

test('a file written by format version 1 still reads (fixture)', async () => {
  const { readFile } = await import('node:fs/promises');
  const bytes = new Uint8Array(await readFile(new URL('./fixtures/psim/v1-tube.psim', import.meta.url)));
  const f = await readPsim(bytes);
  assert.equal(f.version, 1);
  assert.equal(f.info.name, 'Tube');
  assert.equal(f.geometry.tris.length, 40 * 24 * 6);
  assert.equal(f.rfea.arrays.get('static.vm').data.length, 40 * 24);
  assert.equal(readTable(bytes).table.length, 7);
});
