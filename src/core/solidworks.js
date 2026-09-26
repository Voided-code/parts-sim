// SolidWorks .SLDPRT / .SLDASM reader (SolidWorks 2015 and newer), without SolidWorks.
//
// The exact B-rep inside these files is Parasolid, which only the licensed Parasolid kernel
// can read. What we can read is the display tessellation SolidWorks saves with the file:
// one triangle-strip table per face, so parts arrive with their CAD faces intact for
// click-selection. Assemblies store an instance tree of rigid placements and, in newer
// versions, a cached mesh per component; components without one are read from their own
// part files when those are supplied.
//
// Format facts and approach follow sldprt-export (MIT, (c) 2026 XRTC5,
// https://github.com/XRTC5/sldprt-export), which credits the container layout to the
// cadmpeg project's docs/formats/sldprt.md (CC BY 4.0). This is an independent JavaScript
// implementation.
//
// Container: a flat run of blocks, each introduced by 14 00 06 00 08 00:
//   +6 type_id u32 | +10 crc32 of the inflated payload | +14 compressed size
//   +18 inflated size | +22 name length | +26 nibble-swapped name, then raw-DEFLATE data
import { inflateSync, unzlibSync } from 'three/examples/jsm/libs/fflate.module.js';

const SIGNATURE = [0x14, 0x00, 0x06, 0x00, 0x08, 0x00];
const HEADER = 26;
const OLE_MAGIC = [0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1];
const MESH_BLOCKS = ['Contents/DisplayLists', 'FaceTessellations'];
// Strip table: six descriptors (element size, kind, 2, count) - counts, positions (m),
// normals, per-strip extras, per-strip 2n-2, byte extras.
const DESCRIPTORS = [[4, 8], [12, 100], [12, 100], [4, 8], [4, 8], [1, 8]];
const MAX_STRIPS = 250000;
const MAX_STRIP_VERTS = 1000000;
const MAX_DESC_COUNT = 2000000;
const MAX_COMPONENTS = 2000;
const MAX_FACE_TABLES = 250000;
const METERS_TO_MM = 1000;
const STRING_MARKER = [0xff, 0xfe, 0xff];

export function isSolidWorksName(name) {
  return /\.(sldprt|sldasm|slddrw)$/i.test(name);
}

// ---------- container ----------

let crcTable = null;
export function crc32(data) {
  if (!crcTable) {
    crcTable = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
      let c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      crcTable[n] = c >>> 0;
    }
  }
  let c = 0xffffffff;
  for (let i = 0; i < data.length; i++) c = crcTable[(c ^ data[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function startsWith(bytes, magic, at = 0) {
  for (let i = 0; i < magic.length; i++) if (bytes[at + i] !== magic[i]) return false;
  return true;
}

function indexOf(bytes, pattern, from = 0, to = bytes.length) {
  const first = pattern[0];
  const last = Math.min(to, bytes.length) - pattern.length;
  for (let i = from; i <= last; i++) {
    if (bytes[i] === first && startsWith(bytes, pattern, i)) return i;
  }
  return -1;
}

/** Every named block, in file order. */
export function readBlocks(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = [];
  let pos = 8;
  while (true) {
    const at = indexOf(bytes, SIGNATURE, pos);
    if (at < 0 || at + HEADER > bytes.length) break;
    pos = at + 1;
    const typeId = view.getUint32(at + 6, true);
    const crc = view.getUint32(at + 10, true);
    const compressed = view.getUint32(at + 14, true);
    const expanded = view.getUint32(at + 18, true);
    const nameLen = view.getUint32(at + 22, true);
    // A cache-cell index grid reuses the marker, encoding one value L as 2L, L/2 and L.
    if (crc === expanded * 2 && compressed === Math.floor(expanded / 2)) continue;
    if (!compressed || nameLen < 1 || nameLen > 1024) continue;
    if (at + HEADER + nameLen + compressed > bytes.length) continue;
    let name = '';
    let printable = true;
    for (let i = 0; i < nameLen; i++) {
      const b = bytes[at + HEADER + i];
      const c = ((b << 4) | (b >> 4)) & 0xff; // undo the nibble swap
      if (c < 0x20 || c > 0x7e) { printable = false; break; }
      name += String.fromCharCode(c);
    }
    if (!printable) continue;
    out.push({ name, typeId, crc, compressed, expanded, dataOffset: at + HEADER + nameLen });
  }
  return out;
}

/** Inflate a block's payload and verify it against the CRC-32 in its header. */
export function inflateBlock(bytes, block) {
  const raw = bytes.subarray(block.dataOffset, block.dataOffset + block.compressed);
  let data = null;
  for (const inflate of [(d) => inflateSync(d, { out: new Uint8Array(block.expanded) }), (d) => unzlibSync(d)]) {
    try {
      data = inflate(raw);
      if (data.length) break;
    } catch {
      data = null;
    }
  }
  if (!data) data = raw; // stored
  if (data.length !== block.expanded || crc32(data) !== block.crc) {
    throw new Error(`The SolidWorks block "${block.name}" is corrupt (checksum mismatch).`);
  }
  return data;
}

function findBlock(blocks, name) {
  return blocks.find((b) => b.name === name) || null;
}

// ---------- strip tables ----------

const TABLE_SIGNATURE = [4, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0];

/** Parse a strip table at `at`, or null when the layout does not hold. */
function readTable(data, view, at) {
  let pos = at;
  const desc = [];
  for (const [size, kind] of DESCRIPTORS) {
    if (pos + 16 > data.length) return null;
    const dSize = view.getUint32(pos, true), dKind = view.getUint32(pos + 4, true);
    const two = view.getUint32(pos + 8, true), count = view.getUint32(pos + 12, true);
    if (dSize !== size || dKind !== kind || two !== 2) return null;
    if (count > MAX_DESC_COUNT || pos + 16 + count * size > data.length) return null;
    desc.push([pos + 16, count]);
    pos += 16 + size * count;
  }
  const strips = desc[0][1];
  if (!strips || strips > MAX_STRIPS) return null;
  const lengths = new Uint32Array(strips);
  let total = 0;
  for (let j = 0; j < strips; j++) {
    const n = view.getUint32(desc[0][0] + 4 * j, true);
    if (n < 3 || n > MAX_STRIP_VERTS) return null;
    lengths[j] = n;
    total += n;
  }
  if (desc[1][1] !== total || (desc[2][1] !== 0 && desc[2][1] !== total) || desc[4][1] !== strips) return null;
  for (let j = 0; j < strips; j++) {
    if (view.getUint32(desc[4][0] + 4 * j, true) !== 2 * lengths[j] - 2) return null;
  }
  return { start: at, end: pos, positions: desc[1][0], lengths, triangles: total - 2 * strips };
}

function* iterTables(data, from = 0, to = data.length) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let pos = from;
  while (true) {
    const at = indexOf(data, TABLE_SIGNATURE, pos, to);
    if (at < 0) return;
    const table = readTable(data, view, at);
    if (!table || table.end > to) {
      pos = at + 1;
      continue;
    }
    yield table;
    pos = table.end;
  }
}

/**
 * Append a strip table's triangles (mm) to `out`, optionally transformed by a row-vector
 * 4x4 placement whose translation is in metres. Every table is one CAD face.
 */
function emitTable(data, table, face, out, matrix = null, mirrored = false) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const read = (i, p) => {
    const o = table.positions + 12 * i;
    const x = view.getFloat32(o, true), y = view.getFloat32(o + 4, true), z = view.getFloat32(o + 8, true);
    if (matrix) {
      const m = matrix;
      p[0] = METERS_TO_MM * (x * m[0] + y * m[4] + z * m[8] + m[12]);
      p[1] = METERS_TO_MM * (x * m[1] + y * m[5] + z * m[9] + m[13]);
      p[2] = METERS_TO_MM * (x * m[2] + y * m[6] + z * m[10] + m[14]);
    } else {
      p[0] = METERS_TO_MM * x; p[1] = METERS_TO_MM * y; p[2] = METERS_TO_MM * z;
    }
    return p;
  };
  const a = [0, 0, 0], b = [0, 0, 0], c = [0, 0, 0];
  let at = 0;
  for (const n of table.lengths) {
    for (let k = 0; k < n - 2; k++) {
      read(at + k, a); read(at + k + 1, b); read(at + k + 2, c);
      // strips alternate winding; a mirroring placement flips it once more
      const flip = (k % 2 === 1) !== mirrored;
      out.positions.push(...a, ...(flip ? c : b), ...(flip ? b : c));
      out.faces.push(face);
    }
    at += n;
  }
}

/** Bounding box (metres) of a set of strip tables: [xmin, ymin, zmin, xmax, ymax, zmax]. */
function tablesBBox(data, tables) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
  for (const t of tables) {
    const n = t.lengths.reduce((a, b) => a + b, 0);
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        const v = view.getFloat32(t.positions + 12 * i + 4 * d, true);
        if (v < box[d]) box[d] = v;
        if (v > box[d + 3]) box[d + 3] = v;
      }
    }
  }
  return box;
}

/**
 * Does a part file's saved mesh have the extent the assembly expects? A multi-configuration
 * part (Toolbox bolts, nuts) saves only its last-used size, which may not be the one used here.
 */
function matchesBoundingBox(box, expectedText) {
  const e = (expectedText || '').trim().split(/\s+/).map(Number);
  if (e.length !== 6 || !e.every(Number.isFinite)) return true; // nothing to compare against
  for (let d = 0; d < 3; d++) {
    const tol = Math.max(0.0005, 0.03 * Math.max(e[d + 3] - e[d], box[d + 3] - box[d]));
    if (Math.abs(box[d] - e[d]) > tol || Math.abs(box[d + 3] - e[d + 3]) > tol) return false;
  }
  return true;
}

function savedMeshTables(bytes, blocks) {
  for (const name of MESH_BLOCKS) {
    const block = findBlock(blocks, name);
    if (!block) continue;
    const data = inflateBlock(bytes, block);
    const tables = [...iterTables(data)];
    if (tables.length) return { data, tables };
  }
  return null;
}

// ---------- material ----------

function keywordsXML(bytes, blocks) {
  const block = findBlock(blocks, 'swXmlContents/KeyWords');
  if (!block) return '';
  try {
    return new TextDecoder().decode(inflateBlock(bytes, block));
  } catch {
    return '';
  }
}

/** Number of configurations a part defines (Toolbox fasteners hold one per size). */
function configurationCount(bytes, blocks) {
  return (keywordsXML(bytes, blocks).match(/Type="ConfigurationManager"/g) || []).length || 1;
}

/** SolidWorks material name assigned to the part, if any. */
export function readMaterialName(bytes, blocks) {
  const text = keywordsXML(bytes, blocks);
  if (!text) return null;
  const m = /Type="ConfigurationManager"[^>]*?\sMaterial="([^"]*)"/.exec(text) || /\sMaterial="([^"]*)"[^>]*Type="ConfigurationManager"/.exec(text);
  if (!m) return null;
  const name = decodeEntities(m[1]).trim();
  return !name || /not specified/i.test(name) ? null : name;
}

// ---------- assemblies ----------

function decodeEntities(s) {
  return s.replace(/&(lt|gt|amp|quot|apos|#\d+|#x[0-9a-f]+);/gi, (_, e) => {
    const map = { lt: '<', gt: '>', amp: '&', quot: '"', apos: "'" };
    if (map[e.toLowerCase()]) return map[e.toLowerCase()];
    return String.fromCodePoint(e[1].toLowerCase() === 'x' ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10));
  });
}

/** Minimal XML element tree (tags without namespace prefix, attributes, children). */
export function parseXML(text) {
  const root = { tag: '#root', attrs: {}, children: [] };
  const stack = [root];
  const re = /<!--[\s\S]*?-->|<\?[\s\S]*?\?>|<!\[CDATA\[[\s\S]*?\]\]>|<!DOCTYPE[^>]*>|<\/\s*([^\s>]+)\s*>|<([^\s/>]+)((?:\s+[^\s=/>]+\s*=\s*(?:"[^"]*"|'[^']*'))*)\s*(\/?)>/g;
  let m;
  while ((m = re.exec(text))) {
    if (m[1]) {
      if (stack.length > 1) stack.pop();
    } else if (m[2]) {
      const attrs = {};
      const ar = /([^\s=]+)\s*=\s*(?:"([^"]*)"|'([^']*)')/g;
      let a;
      while ((a = ar.exec(m[3]))) attrs[a[1].replace(/^.*:/, '')] = decodeEntities(a[2] ?? a[3]);
      const node = { tag: m[2].replace(/^.*:/, ''), attrs, children: [] };
      stack[stack.length - 1].children.push(node);
      if (!m[4]) stack.push(node);
    }
  }
  return root;
}

function readDirectory(data) {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let at = 0;
  const need = (n) => {
    if (at + n > data.length) throw new Error('The assembly mesh directory is truncated.');
    const s = at;
    at += n;
    return s;
  };
  const u32 = () => view.getUint32(need(4), true);
  const key = () => Array.from(data.subarray(need(8), at), (b) => b.toString(16).padStart(2, '0')).join('');
  const atString = () => startsWith(data, STRING_MARKER, at);
  const string = () => {
    if (!atString()) throw new Error('Unsupported string layout in the assembly mesh directory.');
    need(3);
    let n = data[need(1)];
    if (n === 0xff) {
      n = view.getUint16(need(2), true);
      if (n === 0xffff) n = u32();
    }
    if (n > 4096) throw new Error('A component name in the assembly is too long.');
    const s = need(2 * n);
    let out = '';
    for (let i = 0; i < n; i++) out += String.fromCharCode(view.getUint16(s + 2 * i, true));
    return out;
  };
  if (u32() !== 2) throw new Error('Unsupported assembly mesh directory version.');
  const capacity = u32(), count = u32();
  if (!capacity || capacity > 100000 || !count || count > MAX_COMPONENTS) throw new Error('The assembly exceeds the component limits.');
  // Entries are found by their own signature: a sequence number followed by a string marker.
  const firstMarker = indexOf(data, STRING_MARKER, 12);
  if (firstMarker < 4) return [];
  const base = view.getUint32(firstMarker - 4, true);
  const entries = [];
  for (let i = 0; i < count; i++) {
    const sig = [...new Uint8Array(new Uint32Array([base + i]).buffer), ...STRING_MARKER];
    const start = indexOf(data, sig, i ? at - 4 : firstMarker - 4);
    if (start < 0) break;
    at = start + 4;
    const name = string();
    u32(); u32(); u32(); // stamp, configuration, reserved
    const alias = key();
    const id = key();
    for (let k = 0; k < 4 && !atString() && at + 8 <= data.length; k++) at += 8; // writer-dependent padding
    const chunk = string();
    const index = u32();
    entries.push({ name, alias, id, chunk, index });
  }
  return entries;
}

function rigidTransform(text) {
  const v = (text || '').trim().split(/\s+/).map(Number);
  if (v.length !== 16 || !v.every(Number.isFinite)) return null;
  if ([3, 7, 11].some((i) => Math.abs(v[i]) > 1e-12) || Math.abs(v[15] - 1) > 1e-12) return null;
  const r = [v.slice(0, 3), v.slice(4, 7), v.slice(8, 11)];
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
  const tol = 1e-6;
  if (r.some((x) => Math.abs(dot(x, x) - 1) > tol) || Math.abs(dot(r[0], r[1])) > tol || Math.abs(dot(r[0], r[2])) > tol || Math.abs(dot(r[1], r[2])) > tol || Math.abs(Math.abs(det) - 1) > tol) {
    return null;
  }
  return { matrix: v, mirrored: det < 0 };
}

function compose(child, parent) {
  const out = new Array(16).fill(0);
  for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) for (let k = 0; k < 4; k++) out[i * 4 + j] += child[i * 4 + k] * parent[k * 4 + j];
  return out;
}

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Each stored component mesh: { data, tables } keyed by directory entry id. */
function componentMeshes(bytes, blocks, entries) {
  const byChunk = new Map();
  for (const e of entries) if (e.chunk) (byChunk.get(e.chunk) || byChunk.set(e.chunk, []).get(e.chunk)).push(e);
  const out = new Map();
  for (const [chunk, group] of byChunk) {
    const block = findBlock(blocks, `FaceTessellations/${chunk}`);
    if (!block) continue;
    const data = inflateBlock(bytes, block);
    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
    const records = [];
    for (const entry of group) {
      const marker = [...new Uint8Array(new Uint32Array([entry.index]).buffer), ...entry.id.match(/../g).map((h) => parseInt(h, 16))];
      const start = indexOf(data, marker);
      if (start < 0 || start + 16 > data.length || indexOf(data, marker, start + 1) >= 0) continue;
      const count = view.getUint32(start + 12, true);
      if (!count || count > MAX_FACE_TABLES) continue;
      records.push({ start, count, entry });
    }
    records.sort((a, b) => a.start - b.start);
    records.forEach((r, i) => {
      // A record owns exactly `count` tables; never read into the next record.
      const end = i + 1 < records.length ? records[i + 1].start : data.length;
      const tables = [];
      for (const t of iterTables(data, r.start + 16, end)) {
        tables.push(t);
        if (tables.length === r.count) break;
      }
      if (tables.length === r.count) out.set(r.entry.id, { data, tables });
    });
  }
  return out;
}

function readAssembly(bytes, blocks, resolvePart) {
  const treeBlock = findBlock(blocks, 'swXmlContents/COMPINSTANCETREE');
  const dirBlock = findBlock(blocks, 'FaceTessellations/Directory');
  if (!treeBlock) throw new Error('This assembly has no component tree. Export it from SolidWorks as STEP instead.');
  const xml = parseXML(new TextDecoder().decode(inflateBlock(bytes, treeBlock)));
  const doc = xml.children.find((c) => c.children.length) || xml;
  const one = (parent, tag) => {
    const found = parent.children.filter((c) => c.tag === tag);
    if (found.length !== 1) throw new Error(`The assembly has no unambiguous ${tag} record.`);
    return found[0];
  };
  const files = new Map(one(doc, 'swHeader').children.filter((c) => c.tag === 'swFile').map((f) => [f.attrs.id, f]));
  const models = new Map(one(doc, 'swModelList').children.filter((c) => c.tag === 'swModel').map((m) => [m.attrs.id, m]));
  const configs = one(doc, 'swConfigurationList').children.filter((c) => c.tag === 'swConfiguration');
  if (!configs.length) throw new Error('The assembly has no saved configuration.');
  const chosen = configs.find((c) => c.attrs.swMostRecentConfiguration === 'YES') || configs[0];
  const root = models.get(chosen.attrs.swModelRef);
  if (!root || files.get(root.attrs.swFileRef)?.attrs.swDocType !== 'ASSEMBLY') throw new Error('The assembly configuration could not be resolved.');

  // Newer files cache each component's mesh (FaceTessellations); older ones (e.g. 2019) do
  // not, and every component is then read from its own part file.
  const entries = dirBlock ? readDirectory(inflateBlock(bytes, dirBlock)) : [];
  const byName = new Map(entries.map((e) => [e.name, e]));
  const byId = new Map(entries.map((e) => [e.id, e]));
  const stored = componentMeshes(bytes, blocks, entries);
  const partCache = new Map();
  const report = { references: 0, placed: 0, fromParts: 0, suppressed: 0, hidden: 0, missing: [], otherConfig: [], configuration: chosen.attrs.swName || '' };
  const out = { positions: [], faces: [] };
  let face = 0;

  const fromPartFile = (model) => {
    const file = files.get(model.attrs.swFileRef);
    const fileName = (file?.attrs.swPath || '').replace(/\\/g, '/').split('/').pop();
    if (!fileName || !resolvePart) return null;
    if (!partCache.has(fileName)) {
      let mesh = null;
      const data = resolvePart(fileName);
      if (data) {
        try {
          const b = toBytes(data);
          const partBlocks = readBlocks(b);
          mesh = savedMeshTables(b, partBlocks);
          if (mesh) {
            mesh.box = tablesBBox(mesh.data, mesh.tables);
            mesh.configurations = configurationCount(b, partBlocks);
          }
        } catch {
          mesh = null;
        }
      }
      partCache.set(fileName, mesh);
    }
    const mesh = partCache.get(fileName);
    // A part redesigned since the assembly was saved is shown as it is now (as SolidWorks would);
    // only a multi-configuration part whose saved size differs is left out.
    if (mesh && mesh.configurations > 1 && !matchesBoundingBox(mesh.box, model.attrs.swBoundingBox)) return 'other-config';
    return mesh;
  };

  const walk = (model, path, matrix, depth) => {
    if (depth > 8) return;
    for (const ref of model.children.filter((c) => c.tag === 'swReference')) {
      const a = ref.attrs;
      report.references++;
      if (a.swSuppressed === 'YES') { report.suppressed++; continue; }
      const child = models.get(a.swModelRef);
      const step = `${a.swName}-${a.swReferenceNumber}@${model.attrs.swName}`;
      const full = path ? `${path}/${step}` : step;
      const placement = child && rigidTransform(a.swTransform);
      if (!child || !placement) { report.missing.push(a.swName || full); continue; }
      const world = compose(placement.matrix, matrix);
      if (files.get(child.attrs.swFileRef)?.attrs.swDocType === 'ASSEMBLY') {
        walk(child, full, world, depth + 1);
        continue;
      }
      if (a.swHidden === 'YES') { report.hidden++; continue; }
      let mesh = null;
      let target = byName.get(full);
      const seen = new Set();
      while (target && !target.chunk && !seen.has(target.id)) {
        seen.add(target.id);
        target = byId.get(target.alias); // repeated components alias one stored mesh
      }
      if (target) mesh = stored.get(target.id) || null;
      if (!mesh) {
        mesh = fromPartFile(child);
        if (mesh === 'other-config') {
          report.otherConfig.push(`${a.swName}${child.attrs.swConfigurationName ? ` (${child.attrs.swConfigurationName})` : ''}`);
          continue;
        }
        if (mesh) report.fromParts++;
      }
      if (!mesh) { report.missing.push(a.swName || full); continue; }
      const det = world[0] * (world[5] * world[10] - world[6] * world[9]) - world[1] * (world[4] * world[10] - world[6] * world[8]) + world[2] * (world[4] * world[9] - world[5] * world[8]);
      for (const t of mesh.tables) emitTable(mesh.data, t, face++, out, world, det < 0);
      report.placed++;
    }
  };
  walk(root, '', IDENTITY, 0);
  if (!out.faces.length) {
    throw new Error(
      'None of the assembly components has a saved mesh. Select the assembly together with its part files ' +
        '(or open it from its folder in the desktop app), or export the assembly as STEP.',
    );
  }
  return { out, report };
}

/** File names of the part documents an assembly references (to load alongside it). */
export function assemblyPartNames(buffer) {
  const bytes = toBytes(buffer);
  if (startsWith(bytes, OLE_MAGIC)) return [];
  const block = findBlock(readBlocks(bytes), 'swXmlContents/COMPINSTANCETREE');
  if (!block) return [];
  const xml = parseXML(new TextDecoder().decode(inflateBlock(bytes, block)));
  const doc = xml.children.find((c) => c.children.length) || xml;
  const header = doc.children.find((c) => c.tag === 'swHeader');
  const names = (header?.children || [])
    .filter((f) => f.tag === 'swFile' && f.attrs.swDocType === 'PART')
    .map((f) => (f.attrs.swPath || '').replace(/\\/g, '/').split('/').pop())
    .filter(Boolean);
  return [...new Set(names)];
}

// ---------- entry point ----------

function toBytes(data) {
  return data instanceof Uint8Array ? data : new Uint8Array(data);
}

const EXPORT_HINT = 'In SolidWorks use File ▸ Save As ▸ STEP AP242 (*.step) and open that instead.';

/**
 * @param {ArrayBuffer|Uint8Array} buffer
 * @param {{name?: string, resolvePart?: (fileName: string) => (ArrayBuffer|Uint8Array|null)}} opts
 *        resolvePart returns a component part file's bytes by file name (assemblies only).
 * @returns {{positions: Float32Array, index: null, faceIds: Int32Array, units: 'mm', info: string, material: string|null, warnings: string[]}}
 */
export function readSolidWorks(buffer, { name = '', resolvePart = null } = {}) {
  const bytes = toBytes(buffer);
  const kind = /\.sldasm$/i.test(name) ? 'assembly' : /\.slddrw$/i.test(name) ? 'drawing' : 'part';
  if (kind === 'drawing') throw new Error('SolidWorks drawings (.SLDDRW) hold 2D sheets, not the 3D model. Open the part (.SLDPRT) or assembly (.SLDASM) instead.');
  if (startsWith(bytes, OLE_MAGIC)) {
    throw new Error(`This file uses the pre-2015 SolidWorks format, which Parts Sim cannot read. Open and save it in SolidWorks 2015 or newer, or export it. ${EXPORT_HINT}`);
  }
  const blocks = readBlocks(bytes);
  if (!blocks.length) throw new Error('This does not look like a SolidWorks file (no SolidWorks data blocks were found).');
  const warnings = [];
  let out, info, material = null;
  if (kind === 'assembly') {
    const { out: o, report } = readAssembly(bytes, blocks, resolvePart);
    out = o;
    info = `SolidWorks assembly, ${report.placed} of ${report.placed + report.missing.length + report.otherConfig.length} components`;
    if (report.fromParts) info += ` (${report.fromParts} from part files)`;
    if (report.otherConfig.length) {
      const names = [...new Set(report.otherConfig)].slice(0, 3).join('; ');
      warnings.push(`${report.otherConfig.length} component(s) were left out because their part file is saved in a different configuration (size) than this assembly uses - usually Toolbox fasteners: ${names}${report.otherConfig.length > 3 ? '; …' : ''}.`);
    }
    if (report.missing.length) {
      const names = [...new Set(report.missing)].slice(0, 4).join(', ');
      warnings.push(`${report.missing.length} component(s) have no saved mesh (${names}${report.missing.length > 4 ? ', …' : ''}). Add their .SLDPRT files, or export the assembly as STEP.`);
    }
  } else {
    const mesh = savedMeshTables(bytes, blocks);
    if (!mesh) {
      throw new Error(
        'This SolidWorks part was saved without display data, so its geometry cannot be read without SolidWorks. ' +
          `Re-save it with Options ▸ Document Properties ▸ Image Quality ▸ “Save tessellation with part document” turned on, or: ${EXPORT_HINT}`,
      );
    }
    out = { positions: [], faces: [] };
    mesh.tables.forEach((t, i) => emitTable(mesh.data, t, i, out));
    material = readMaterialName(bytes, blocks);
    info = `SolidWorks part, ${mesh.tables.length} faces (saved display mesh)`;
  }
  return { positions: Float32Array.from(out.positions), index: null, faceIds: Int32Array.from(out.faces), units: 'mm', info, material, warnings };
}

/** Best match in the Parts Sim library for a SolidWorks material name, or null. */
export function matchMaterial(swName, materials) {
  if (!swName) return null;
  const s = swName.toLowerCase();
  const rules = [
    [/ti-?6al-?4v|titanium/, 'ti-64'],
    [/7075/, 'al-7075'],
    [/6061|6063|alumin(i)?um|al\s?alloy|1060|5052/, 'al-6061'],
    [/304|316|stainless/, 'ss-304'],
    [/cast.*iron|grey iron|gray iron/, 'cast-iron'],
    [/4140|4340|alloy steel|chrome|cr-?v/, 'steel-alloy'],
    [/1020|1018|1045|carbon steel|plain.*steel|steel/, 'steel-1020'],
    [/brass/, 'brass'],
    [/copper/, 'copper'],
    [/petg/, 'petg'],
    [/\bpla\b/, 'pla'],
    [/\babs\b/, 'abs'],
    [/nylon|\bpa ?6|\bpa ?66|polyamide/, 'nylon'],
    [/polycarbonate|\bpc\b/, 'pc'],
    [/carbon fib|cfrp/, 'cfrp'],
    [/glass/, 'glass'],
    [/wood|pine|oak|balsa/, 'wood'],
  ];
  for (const [re, id] of rules) if (re.test(s) && materials.some((m) => m.id === id)) return id;
  return null;
}
