// The .psim file: container, section codecs and array coding. The format is specified in
// docs/psim-format.md; this file is the JavaScript reader and writer. It does not know about the apps'
// panels: it turns plain data (a built mesh, JSON, typed arrays) into bytes and back, and checks
// everything it reads, because files come from strangers.
//
// Everything is async only because the browser's (De)CompressionStream is: Node 18+ has the same API.

export const PSIM_VERSION = 1;
export const MAGIC = Uint8Array.of(0x89, 0x50, 0x53, 0x49, 0x4d, 0x0d, 0x0a, 0x1a, 0x0a);
export const MIME = 'application/x-parts-sim';

export const CODEC_STORED = 0;
export const CODEC_DEFLATE = 1;
const CODEC_NAMES = { 2: 'zstd', 3: 'brotli' };

/** Section ids in the order they are written. */
export const SECTION_ORDER = ['INFO', 'THMB', 'GEOM', 'CADS', 'SETP', 'RFEA', 'RAIR', 'VIEW'];
const SECTION_CODEC = { INFO: CODEC_DEFLATE, THMB: CODEC_STORED, GEOM: CODEC_DEFLATE, CADS: CODEC_DEFLATE, SETP: CODEC_DEFLATE, RFEA: CODEC_DEFLATE, RAIR: CODEC_DEFLATE, VIEW: CODEC_DEFLATE };
/** Sections a file can still be shown without when their codec is unknown. */
const OPTIONAL = new Set(['THMB', 'CADS', 'VIEW']);

export const LIMITS = Object.freeze({
  sections: 64,
  rawPerSection: 512 * 2 ** 20,
  rawTotal: 1024 * 2 ** 20,
  ratio: 1100, // deflate cannot beat about 1032 : 1
  ratioSlack: 64 * 1024,
  jsonSection: 64 * 2 ** 20,
  vertices: 50e6,
  triangles: 50e6,
  arrayElements: 500e6,
  arrays: 100000,
});

const HEADER = 32;
const ENTRY = 24;
const TRAILER = 8;

export class PsimError extends Error {
  constructor(message, code = 'invalid') {
    super(message);
    this.name = 'PsimError';
    this.code = code;
  }
}
const fail = (message, code) => { throw new PsimError(message, code); };

// ---------------------------------------------------------------- CRC-32

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(bytes, start = 0, end = bytes.length, crc = 0) {
  let c = ~crc;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return ~c >>> 0;
}

// ---------------------------------------------------------------- deflate

async function collect(readable, limit) {
  const reader = readable.getReader();
  const chunks = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    n += value.length;
    if (limit !== undefined && n > limit) {
      reader.cancel().catch(() => {});
      fail('A section is larger than it says (possible decompression bomb).', 'bomb');
    }
    chunks.push(value);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

const once = (bytes) => new ReadableStream({ start(c) { c.enqueue(bytes); c.close(); } });

export async function deflateRaw(bytes) {
  return collect(once(bytes).pipeThrough(new CompressionStream('deflate-raw')));
}

/** Inflates `bytes`, which must come out exactly `rawLength` long. */
export async function inflateRaw(bytes, rawLength) {
  let out;
  try {
    out = await collect(once(bytes).pipeThrough(new DecompressionStream('deflate-raw')), rawLength);
  } catch (err) {
    if (err instanceof PsimError) throw err;
    fail('A section could not be decompressed; the file is damaged.', 'damaged');
  }
  if (out.length !== rawLength) fail('A section is shorter than it says; the file is damaged.', 'damaged');
  return out;
}

// ---------------------------------------------------------------- byte helpers

const zigzag = (v) => ((v << 1) ^ (v >> 31)) >>> 0;
const unzigzag = (u) => (u >>> 1) ^ -(u & 1);

/** Writes `values` (unsigned, `width` bytes each) as byte planes at `out[o...]`. Returns the new offset. */
function putPlanes(out, o, values, width) {
  const n = values.length;
  for (let b = 0; b < width; b++) {
    const shift = 8 * b;
    for (let i = 0; i < n; i++) out[o + i] = (values[i] >>> shift) & 0xff;
    o += n;
  }
  return o;
}

/** Reads `n` values of `width` bytes from byte planes at `bytes[o...]`. */
function getPlanes(bytes, o, n, width) {
  const out = new Uint32Array(n);
  for (let b = 0; b < width; b++) {
    const shift = 8 * b;
    for (let i = 0; i < n; i++) out[i] |= bytes[o + i] << shift;
    o += n;
  }
  return out;
}

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8', { fatal: true });

/** JSON with infinity and not-a-number written as {"$num": ...}. */
export function stringifyJson(value) {
  return JSON.stringify(value, (k, v) => {
    if (typeof v === 'number' && !Number.isFinite(v)) return { $num: Number.isNaN(v) ? 'nan' : v > 0 ? 'inf' : '-inf' };
    if (ArrayBuffer.isView(v) && !(v instanceof DataView)) return Array.from(v);
    return v;
  });
}

export function parseJson(text) {
  try {
    return JSON.parse(text, (k, v) => {
      if (v && typeof v === 'object' && !Array.isArray(v) && '$num' in v) {
        return v.$num === 'nan' ? NaN : v.$num === 'inf' ? Infinity : v.$num === '-inf' ? -Infinity : v;
      }
      return v;
    });
  } catch {
    return fail('A section holds text that is not valid JSON; the file is damaged.', 'damaged');
  }
}

function jsonBytes(value) { return textEncoder.encode(stringifyJson(value)); }
function jsonOf(bytes) {
  let text;
  try { text = textDecoder.decode(bytes); } catch { return fail('A section holds text that is not valid UTF-8.', 'damaged'); }
  return parseJson(text);
}

// ---------------------------------------------------------------- arrays container

const LEVELS = { q16: 65536, q8: 256 };
const WIDTH = { q16: 2, q8: 1, f32: 4, i32: 4, u8: 1 };

/** Prediction residuals of quantised codes, in place order x fastest. Returns zig-zag residuals. */
function predictEncode(codes, L, dims) {
  const n = codes.length;
  const res = new Uint16Array(n);
  const half = L >> 1;
  const put = (i, p) => {
    let r = (codes[i] - p) % L;
    if (r < 0) r += L;
    if (r >= half) r -= L;
    res[i] = r >= 0 ? 2 * r : -2 * r - 1;
  };
  if (!dims) {
    let prev = 0;
    for (let i = 0; i < n; i++) { put(i, prev); prev = codes[i]; }
    return res;
  }
  const [nx, ny, nz] = dims;
  const sy = nx, sz = nx * ny;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const i = x + sy * y + sz * z;
        const a = x > 0 ? codes[i - 1] : 0;
        const b = y > 0 ? codes[i - sy] : 0;
        const c = z > 0 ? codes[i - sz] : 0;
        const ab = x > 0 && y > 0 ? codes[i - 1 - sy] : 0;
        const ac = x > 0 && z > 0 ? codes[i - 1 - sz] : 0;
        const bc = y > 0 && z > 0 ? codes[i - sy - sz] : 0;
        const abc = x > 0 && y > 0 && z > 0 ? codes[i - 1 - sy - sz] : 0;
        put(i, a + b + c - ab - ac - bc + abc);
      }
    }
  }
  return res;
}

function predictDecode(res, L, dims, Type) {
  const n = res.length;
  const codes = new Type(n);
  const get = (i, p) => {
    const r = res[i] & 1 ? -((res[i] + 1) >> 1) : res[i] >> 1;
    let c = (p + r) % L;
    if (c < 0) c += L;
    codes[i] = c;
  };
  if (!dims) {
    let prev = 0;
    for (let i = 0; i < n; i++) { get(i, prev); prev = codes[i]; }
    return codes;
  }
  const [nx, ny, nz] = dims;
  const sy = nx, sz = nx * ny;
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const i = x + sy * y + sz * z;
        const a = x > 0 ? codes[i - 1] : 0;
        const b = y > 0 ? codes[i - sy] : 0;
        const c = z > 0 ? codes[i - sz] : 0;
        const ab = x > 0 && y > 0 ? codes[i - 1 - sy] : 0;
        const ac = x > 0 && z > 0 ? codes[i - 1 - sz] : 0;
        const bc = y > 0 && z > 0 ? codes[i - sy - sz] : 0;
        const abc = x > 0 && y > 0 && z > 0 ? codes[i - 1 - sy - sz] : 0;
        get(i, a + b + c - ab - ac - bc + abc);
      }
    }
  }
  return codes;
}

function checkDims(dims, n) {
  if (!Array.isArray(dims) || dims.length !== 3 || dims.some((d) => !Number.isInteger(d) || d < 1) || dims[0] * dims[1] * dims[2] !== n) {
    fail('An array has grid dimensions that do not match its length.', 'invalid');
  }
}

/**
 * Builds the body of a result section.
 * arrays: [{ name, data, enc: 'f32'|'i32'|'u8'|'q16'|'q8', dims?: [nx, ny, nz] }]
 * Returns { bytes, bounds } where bounds lists the error bound of each lossy array.
 */
export function encodeArrays(meta, arrays) {
  const entries = [];
  const blobs = [];
  const bounds = [];
  for (const a of arrays) {
    const { name, data, enc } = a;
    if (!(enc in WIDTH)) fail(`Unknown array encoding ${enc}.`);
    const n = data.length;
    const entry = { name, enc, n };
    if (a.dims) { checkDims(a.dims, n); entry.dims = a.dims; }
    let blob;
    if (enc === 'f32') {
      const bits = new Uint32Array(new Float32Array(data).buffer);
      blob = new Uint8Array(4 * n);
      putPlanes(blob, 0, bits, 4);
    } else if (enc === 'i32') {
      const z = new Uint32Array(n);
      for (let i = 0; i < n; i++) z[i] = zigzag(data[i]);
      blob = new Uint8Array(4 * n);
      putPlanes(blob, 0, z, 4);
    } else if (enc === 'u8') {
      blob = Uint8Array.from(data);
    } else {
      const L = LEVELS[enc];
      let min = Infinity, max = -Infinity;
      for (let i = 0; i < n; i++) {
        const v = data[i];
        if (Number.isFinite(v)) { if (v < min) min = v; if (v > max) max = v; }
      }
      if (min > max) { min = 0; max = 0; }
      const codes = enc === 'q16' ? new Uint16Array(n) : new Uint8Array(n);
      const span = max - min;
      for (let i = 0; i < n; i++) {
        const v = data[i];
        codes[i] = Number.isFinite(v) ? (span > 0 ? 1 + Math.round(((v - min) / span) * (L - 2)) : 1) : 0;
      }
      const res = predictEncode(codes, L, a.dims);
      blob = new Uint8Array(WIDTH[enc] * n);
      putPlanes(blob, 0, res, WIDTH[enc]);
      entry.min = min;
      entry.max = max;
      entry.err = span / (2 * (L - 2));
      bounds.push({ field: name, absolute: entry.err });
    }
    entry.bytes = blob.length;
    entries.push(entry);
    blobs.push(blob);
  }
  const head = jsonBytes({ meta, arrays: entries });
  let total = 4 + head.length;
  for (const b of blobs) total += b.length;
  const out = new Uint8Array(total);
  new DataView(out.buffer).setUint32(0, head.length, true);
  out.set(head, 4);
  let o = 4 + head.length;
  for (const b of blobs) { out.set(b, o); o += b.length; }
  return { bytes: out, bounds };
}

/** Reads a result section body: { meta, arrays: Map(name -> { data, enc, dims, err }) }. */
export function decodeArrays(bytes) {
  if (bytes.length < 4) fail('A result section is too short.', 'damaged');
  const hl = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
  if (hl > bytes.length - 4 || hl > LIMITS.jsonSection) fail('A result section has a header longer than the section.', 'damaged');
  const head = jsonOf(bytes.subarray(4, 4 + hl));
  if (!head || typeof head !== 'object' || !Array.isArray(head.arrays)) fail('A result section has no array list.', 'damaged');
  if (head.arrays.length > LIMITS.arrays) fail('A result section lists too many arrays.', 'limit');
  const arrays = new Map();
  let o = 4 + hl;
  let elements = 0;
  for (const e of head.arrays) {
    if (!e || typeof e.name !== 'string' || !(e.enc in WIDTH) || !Number.isInteger(e.n) || e.n < 0 || !Number.isInteger(e.bytes)) fail('An array entry is malformed.', 'damaged');
    elements += e.n;
    if (elements > LIMITS.arrayElements) fail('A result section holds too many numbers.', 'limit');
    if (e.bytes !== WIDTH[e.enc] * e.n) fail('An array entry has the wrong byte count.', 'damaged');
    if (o + e.bytes > bytes.length) fail('An array runs past the end of its section.', 'damaged');
    if (e.dims) checkDims(e.dims, e.n);
    let data;
    if (e.enc === 'f32') {
      data = new Float32Array(new Uint32Array(getPlanes(bytes, o, e.n, 4)).buffer);
    } else if (e.enc === 'i32') {
      const z = getPlanes(bytes, o, e.n, 4);
      data = new Int32Array(e.n);
      for (let i = 0; i < e.n; i++) data[i] = unzigzag(z[i]);
    } else if (e.enc === 'u8') {
      data = bytes.slice(o, o + e.n);
    } else {
      const L = LEVELS[e.enc];
      if (!Number.isFinite(e.min) || !Number.isFinite(e.max)) fail('A quantised array has no valid range.', 'damaged');
      const res = getPlanes(bytes, o, e.n, WIDTH[e.enc]);
      const codes = predictDecode(res, L, e.dims, e.enc === 'q16' ? Uint16Array : Uint8Array);
      data = new Float32Array(e.n);
      const k = (e.max - e.min) / (L - 2);
      for (let i = 0; i < e.n; i++) data[i] = codes[i] === 0 ? NaN : e.min + (codes[i] - 1) * k;
    }
    if (arrays.has(e.name)) fail(`The array ${e.name} appears twice.`, 'damaged');
    arrays.set(e.name, { data, enc: e.enc, dims: e.dims ?? null, err: e.err ?? 0 });
    o += e.bytes;
  }
  if (o !== bytes.length) fail('A result section has bytes the array list does not account for.', 'damaged');
  return { meta: head.meta ?? {}, arrays };
}

// ---------------------------------------------------------------- geometry

/**
 * geometry: { vertices: Float32Array, tris: Uint32Array, faceOf: Int32Array|null, brepFaces, faceCount, faceAngle }
 * mode: 'exact' or 'quantised'. Returns { bytes, bounds }.
 */
export function encodeGeometry(g, mode = 'quantised') {
  const nV = g.vertices.length / 3, nT = g.tris.length / 3;
  const quant = mode === 'quantised';
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nV; i++) for (let d = 0; d < 3; d++) {
    const v = g.vertices[3 * i + d];
    if (v < min[d]) min[d] = v;
    if (v > max[d]) max[d] = v;
  }
  if (nV === 0) { min.fill(0); max.fill(0); }
  const brep = !!g.brepFaces && !!g.faceOf;
  const size = 68 + (quant ? 2 * 3 * nV : 4 * 3 * nV) + 4 * 3 * nT + (brep ? 4 * nT : 0);
  const out = new Uint8Array(size);
  const dv = new DataView(out.buffer);
  out[0] = quant ? 1 : 0;
  out[1] = quant ? 16 : 0;
  out[2] = brep ? 1 : 0;
  dv.setUint32(4, nV, true);
  dv.setUint32(8, nT, true);
  dv.setUint32(12, g.faceCount ?? 0, true);
  dv.setFloat32(16, brep ? 0 : g.faceAngle ?? 20, true);
  for (let d = 0; d < 3; d++) { dv.setFloat64(20 + 8 * d, min[d], true); dv.setFloat64(44 + 8 * d, max[d], true); }
  let o = 68;
  const bounds = [];
  const axis = new Uint32Array(nV);
  for (let d = 0; d < 3; d++) {
    if (quant) {
      const span = max[d] - min[d];
      let prev = 0;
      for (let i = 0; i < nV; i++) {
        const q = span > 0 ? Math.round(((g.vertices[3 * i + d] - min[d]) / span) * 65535) : 0;
        let r = (q - prev) & 0xffff;
        if (r >= 32768) r -= 65536;
        axis[i] = r >= 0 ? 2 * r : -2 * r - 1;
        prev = q;
      }
      o = putPlanes(out, o, axis, 2);
      bounds.push(span / 131070);
    } else {
      const f = new Float32Array(axis.buffer);
      for (let i = 0; i < nV; i++) f[i] = g.vertices[3 * i + d];
      o = putPlanes(out, o, axis, 4);
    }
  }
  const slot = new Uint32Array(nT);
  for (let k = 0; k < 3; k++) {
    let prev = 0;
    for (let t = 0; t < nT; t++) {
      const c = g.tris[3 * t + k];
      slot[t] = zigzag((c - prev) | 0);
      prev = c;
    }
    o = putPlanes(out, o, slot, 4);
  }
  if (brep) {
    let prev = 0;
    for (let t = 0; t < nT; t++) { slot[t] = zigzag((g.faceOf[t] - prev) | 0); prev = g.faceOf[t]; }
    o = putPlanes(out, o, slot, 4);
  }
  return { bytes: out, bounds: quant ? { field: 'geometry.position', absolute: Math.max(...bounds) } : null };
}

export function decodeGeometry(bytes) {
  if (bytes.length < 68) fail('The geometry section is too short.', 'damaged');
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const mode = bytes[0], bits = bytes[1], brep = (bytes[2] & 1) === 1;
  if (mode > 1 || (mode === 1 && bits !== 16) || (mode === 0 && bits !== 0)) fail('The geometry uses a coding this version does not know.', 'unsupported');
  const nV = dv.getUint32(4, true), nT = dv.getUint32(8, true), faceCount = dv.getUint32(12, true);
  if (nV > LIMITS.vertices || nT > LIMITS.triangles) fail('The part has more vertices or triangles than this reader accepts.', 'limit');
  const expect = 68 + (mode ? 2 * 3 * nV : 4 * 3 * nV) + 4 * 3 * nT + (brep ? 4 * nT : 0);
  if (expect !== bytes.length) fail('The geometry section does not match its vertex and triangle counts.', 'damaged');
  const faceAngle = dv.getFloat32(16, true);
  const min = [0, 0, 0], max = [0, 0, 0];
  for (let d = 0; d < 3; d++) {
    min[d] = dv.getFloat64(20 + 8 * d, true);
    max[d] = dv.getFloat64(44 + 8 * d, true);
    if (!Number.isFinite(min[d]) || !Number.isFinite(max[d]) || max[d] < min[d]) fail('The geometry has an invalid bounding box.', 'damaged');
  }
  const vertices = new Float32Array(3 * nV);
  let o = 68;
  for (let d = 0; d < 3; d++) {
    if (mode) {
      const z = getPlanes(bytes, o, nV, 2);
      o += 2 * nV;
      const span = max[d] - min[d];
      let prev = 0;
      for (let i = 0; i < nV; i++) {
        const r = z[i] & 1 ? -((z[i] + 1) >> 1) : z[i] >> 1;
        prev = (prev + r) & 0xffff;
        vertices[3 * i + d] = min[d] + (prev / 65535) * span;
      }
    } else {
      const bits32 = getPlanes(bytes, o, nV, 4);
      o += 4 * nV;
      const f = new Float32Array(bits32.buffer);
      for (let i = 0; i < nV; i++) vertices[3 * i + d] = f[i];
    }
  }
  const tris = new Uint32Array(3 * nT);
  for (let k = 0; k < 3; k++) {
    const z = getPlanes(bytes, o, nT, 4);
    o += 4 * nT;
    let prev = 0;
    for (let t = 0; t < nT; t++) {
      prev = (prev + unzigzag(z[t])) >>> 0;
      if (prev >= nV) fail('A triangle refers to a vertex that does not exist.', 'damaged');
      tris[3 * t + k] = prev;
    }
  }
  for (let t = 0; t < nT; t++) {
    const a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
    if (a === b || b === c || a === c) fail('A triangle uses one vertex twice.', 'damaged');
  }
  let faceOf = null;
  if (brep) {
    faceOf = new Int32Array(nT);
    const z = getPlanes(bytes, o, nT, 4);
    let prev = 0;
    for (let t = 0; t < nT; t++) {
      prev = (prev + unzigzag(z[t])) | 0;
      if (prev < 0 || prev >= Math.max(faceCount, 1)) fail('A triangle belongs to a face that does not exist.', 'damaged');
      faceOf[t] = prev;
    }
  }
  return { vertices, tris, faceOf, brepFaces: brep, faceCount, faceAngle: brep ? 0 : faceAngle, bbox: { min, max }, exact: mode === 0 };
}

// ---------------------------------------------------------------- container

/**
 * Writes a file. content: { info, thumb?: Uint8Array (JPEG), geometry?: {...}, cad?: Uint8Array, setup?, rfea?: { meta, arrays },
 * rair?: { meta, arrays }, view? }. options: { geometryMode: 'quantised'|'exact', created }.
 * `info` is completed with `format`, `created`, `bounds` and the section sizes are left to the table.
 * Returns the file bytes.
 */
export async function writePsim(content, options = {}) {
  const parts = new Map(); // id -> { raw, lossy }
  const bounds = [];
  const geometryMode = options.geometryMode ?? 'quantised';
  if (content.thumb) parts.set('THMB', { raw: content.thumb });
  if (content.geometry) {
    const g = encodeGeometry(content.geometry, geometryMode);
    parts.set('GEOM', { raw: g.bytes, lossy: !!g.bounds });
    if (g.bounds) bounds.push(g.bounds);
  }
  if (content.cad) parts.set('CADS', { raw: content.cad });
  if (content.setup) parts.set('SETP', { raw: jsonBytes(content.setup) });
  for (const [id, key] of [['RFEA', 'rfea'], ['RAIR', 'rair']]) {
    if (!content[key]) continue;
    const r = encodeArrays(content[key].meta, content[key].arrays);
    parts.set(id, { raw: r.bytes, lossy: r.bounds.length > 0 });
    bounds.push(...r.bounds);
  }
  if (content.view) parts.set('VIEW', { raw: jsonBytes(content.view) });

  const info = {
    format: PSIM_VERSION,
    ...content.info,
    created: options.created ?? content.info?.created ?? null,
    bounds,
  };
  parts.set('INFO', { raw: jsonBytes(info) });

  const ids = SECTION_ORDER.filter((id) => parts.has(id));
  const stored = [];
  for (const id of ids) {
    const p = parts.get(id);
    const codec = SECTION_CODEC[id];
    stored.push(codec === CODEC_DEFLATE ? await deflateRaw(p.raw) : p.raw);
  }
  const n = ids.length;
  let size = HEADER + ENTRY * n + TRAILER;
  for (const s of stored) size += s.length;
  if (size > 0xffffffff) fail('The file would be larger than 4 GB.', 'limit');
  const out = new Uint8Array(size);
  const dv = new DataView(out.buffer);
  out.set(MAGIC, 0);
  dv.setUint16(10, PSIM_VERSION, true);
  dv.setUint16(12, 1, true);
  dv.setUint32(16, n, true);
  dv.setUint32(20, HEADER, true);
  let offset = HEADER + ENTRY * n;
  ids.forEach((id, i) => {
    const e = HEADER + ENTRY * i;
    const p = parts.get(id);
    for (let k = 0; k < 4; k++) out[e + k] = id.charCodeAt(k);
    dv.setUint16(e + 4, SECTION_CODEC[id], true);
    dv.setUint16(e + 6, p.lossy ? 1 : 0, true);
    dv.setUint32(e + 8, offset, true);
    dv.setUint32(e + 12, stored[i].length, true);
    dv.setUint32(e + 16, p.raw.length, true);
    dv.setUint32(e + 20, crc32(p.raw), true);
    out.set(stored[i], offset);
    offset += stored[i].length;
  });
  let c = crc32(out, 0, 24);
  c = crc32(out, HEADER, HEADER + ENTRY * n, c);
  dv.setUint32(24, c, true);
  dv.setUint32(size - 8, crc32(out, 0, size - 8), true);
  dv.setUint32(size - 4, size, true);
  return out;
}

/** Checks the container and returns { version, minReader, flags, table }. Does not decompress. */
export function readTable(bytes) {
  if (bytes.length < HEADER + TRAILER) fail('This is not a Parts Sim file (it is too short).', 'notpsim');
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC[i]) {
      fail(i < 5 ? 'This is not a Parts Sim file.' : 'This file was changed by a text-mode transfer (line endings were converted); download it again as binary.', i < 5 ? 'notpsim' : 'damaged');
    }
  }
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = dv.getUint16(10, true), minReader = dv.getUint16(12, true), flags = dv.getUint16(14, true);
  if (minReader > PSIM_VERSION) fail(`This file needs a newer Parts Sim (file format ${minReader}; this app reads up to ${PSIM_VERSION}).`, 'newer');
  const size = bytes.length;
  if (dv.getUint32(size - 4, true) !== size) fail('The file is cut short or has extra bytes at the end.', 'damaged');
  if (dv.getUint32(size - 8, true) !== crc32(bytes, 0, size - 8)) fail('The file is damaged (checksum mismatch).', 'damaged');
  const n = dv.getUint32(16, true);
  if (n > LIMITS.sections) fail('The file lists too many sections.', 'limit');
  if (dv.getUint32(20, true) !== HEADER) fail('The section table is not where it should be.', 'damaged');
  const tableEnd = HEADER + ENTRY * n;
  if (tableEnd + TRAILER > size) fail('The section table runs past the end of the file.', 'damaged');
  let c = crc32(bytes, 0, 24);
  c = crc32(bytes, HEADER, tableEnd, c);
  if (dv.getUint32(24, true) !== c) fail('The header is damaged (checksum mismatch).', 'damaged');
  const table = [];
  let next = tableEnd, rawTotal = 0;
  for (let i = 0; i < n; i++) {
    const e = HEADER + ENTRY * i;
    const id = String.fromCharCode(bytes[e], bytes[e + 1], bytes[e + 2], bytes[e + 3]);
    const entry = {
      id,
      codec: dv.getUint16(e + 4, true),
      flags: dv.getUint16(e + 6, true),
      offset: dv.getUint32(e + 8, true),
      stored: dv.getUint32(e + 12, true),
      raw: dv.getUint32(e + 16, true),
      crc: dv.getUint32(e + 20, true),
    };
    if (entry.offset !== next) fail('The sections overlap or leave gaps.', 'damaged');
    next += entry.stored;
    if (next > size - TRAILER) fail('A section runs past the end of the file.', 'damaged');
    rawTotal += entry.raw;
    if (entry.raw > LIMITS.rawPerSection || rawTotal > LIMITS.rawTotal) fail('The file holds more data than this reader accepts.', 'limit');
    if (entry.raw > entry.stored * LIMITS.ratio + LIMITS.ratioSlack) fail('A section is compressed far more than any real file (possible decompression bomb).', 'bomb');
    if (entry.codec === CODEC_STORED && entry.stored !== entry.raw) fail('A stored section has two different lengths.', 'damaged');
    table.push(entry);
  }
  if (next !== size - TRAILER) fail('The file has bytes after its last section.', 'damaged');
  return { version, minReader, flags, table };
}

/** Returns a section's uncompressed bytes (checked against its CRC-32). */
export async function readSection(bytes, entry) {
  const stored = bytes.subarray(entry.offset, entry.offset + entry.stored);
  let raw;
  if (entry.codec === CODEC_STORED) raw = stored;
  else if (entry.codec === CODEC_DEFLATE) raw = await inflateRaw(stored, entry.raw);
  else fail(`This file uses the ${CODEC_NAMES[entry.codec] ?? `codec ${entry.codec}`} compression for its ${entry.id} section, which this version cannot read.`, 'unsupported');
  if (crc32(raw) !== entry.crc) fail(`The ${entry.id} section is damaged (checksum mismatch).`, 'damaged');
  return raw;
}

/** Reads the INFO section and the thumbnail only. */
export async function inspectPsim(bytes) {
  const t = readTable(bytes);
  const get = (id) => t.table.find((e) => e.id === id);
  const info = get('INFO');
  if (!info) fail('The file has no INFO section.', 'damaged');
  const thumb = get('THMB');
  return {
    ...t,
    info: jsonOf(await readSection(bytes, info)),
    thumb: thumb && thumb.codec === CODEC_STORED ? await readSection(bytes, thumb) : null,
    size: bytes.length,
  };
}

/**
 * Reads the whole file. `want` limits which sections are decoded (default: all the app uses).
 * Returns { version, info, table, size, thumb, geometry, cad, setup, rfea, rair, view } with absent sections null.
 */
export async function readPsim(bytes, { want = SECTION_ORDER } = {}) {
  const t = readTable(bytes);
  const out = { version: t.version, minReader: t.minReader, table: t.table, size: bytes.length, info: null, thumb: null, geometry: null, cad: null, setup: null, rfea: null, rair: null, view: null };
  const skipped = [];
  for (const e of t.table) {
    if (!SECTION_ORDER.includes(e.id)) { skipped.push(e.id); continue; }
    if (!want.includes(e.id)) continue;
    let raw;
    try {
      raw = await readSection(bytes, e);
    } catch (err) {
      if (err.code === 'unsupported' && OPTIONAL.has(e.id)) { skipped.push(e.id); continue; }
      throw err;
    }
    if (e.id === 'INFO') out.info = jsonOf(raw);
    else if (e.id === 'THMB') out.thumb = raw;
    else if (e.id === 'GEOM') out.geometry = decodeGeometry(raw);
    else if (e.id === 'CADS') out.cad = raw;
    else if (e.id === 'SETP') out.setup = jsonOf(raw);
    else if (e.id === 'RFEA') out.rfea = decodeArrays(raw);
    else if (e.id === 'RAIR') out.rair = decodeArrays(raw);
    else if (e.id === 'VIEW') out.view = jsonOf(raw);
  }
  out.skipped = skipped;
  if (want.includes('INFO') && !out.info) fail('The file has no INFO section.', 'damaged');
  return out;
}
