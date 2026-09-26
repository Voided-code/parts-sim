// Synthetic SolidWorks files for tests: a 20 mm cube part and a small assembly of it.
// Every header field is written deliberately, so a reader that is subtly wrong fails.
import { deflateSync } from 'three/examples/jsm/libs/fflate.module.js';
import { crc32 } from '../../src/core/solidworks.js';

const enc = new TextEncoder();

export function block(name, payload, { corrupt = false } = {}) {
  const data = typeof payload === 'string' ? enc.encode(payload) : payload;
  const comp = deflateSync(data);
  const nameBytes = Uint8Array.from(enc.encode(name), (b) => ((b << 4) | (b >> 4)) & 0xff);
  const out = new Uint8Array(26 + nameBytes.length + comp.length);
  const v = new DataView(out.buffer);
  out.set([0x14, 0, 6, 0, 8, 0]);
  v.setUint32(6, 1, true);
  v.setUint32(10, corrupt ? crc32(data) ^ 1 : crc32(data), true);
  v.setUint32(14, comp.length, true);
  v.setUint32(18, data.length, true);
  v.setUint32(22, nameBytes.length, true);
  out.set(nameBytes, 26);
  out.set(comp, 26 + nameBytes.length);
  return out;
}

export function container(...blocks) {
  const head = Uint8Array.of(0x12, 0x34, 0x56, 0x78, 0, 0, 0, 4);
  const size = blocks.reduce((n, b) => n + b.length, head.length);
  const out = new Uint8Array(size);
  let o = 0;
  for (const b of [head, ...blocks]) { out.set(b, o); o += b.length; }
  return out;
}

/** One strip table: each strip is [vertices] in metres. */
function stripTable(strips) {
  const total = strips.reduce((n, s) => n + s.length, 0);
  const parts = [];
  const desc = (size, kind, count, fill) => {
    const b = new DataView(new ArrayBuffer(16 + size * count));
    b.setUint32(0, size, true); b.setUint32(4, kind, true); b.setUint32(8, 2, true); b.setUint32(12, count, true);
    fill?.(b);
    parts.push(new Uint8Array(b.buffer));
  };
  desc(4, 8, strips.length, (b) => strips.forEach((s, j) => b.setUint32(16 + 4 * j, s.length, true)));
  desc(12, 100, total, (b) => strips.flat().forEach((p, i) => p.forEach((c, d) => b.setFloat32(16 + 12 * i + 4 * d, c, true))));
  desc(12, 100, 0);
  desc(4, 8, 0);
  desc(4, 8, strips.length, (b) => strips.forEach((s, j) => b.setUint32(16 + 4 * j, 2 * s.length - 2, true)));
  desc(1, 8, 0);
  const n = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

function concat(...arrays) {
  const out = new Uint8Array(arrays.reduce((n, a) => n + a.length, 0));
  let o = 0;
  for (const a of arrays) { out.set(a, o); o += a.length; }
  return out;
}

// 20 mm cube as six faces, each a 4-vertex strip (2 triangles), wound outward.
const s = 0.02;
const P = (x, y, z) => [x * s, y * s, z * s];
const cubeFaces = [
  [P(0, 0, 0), P(0, 1, 0), P(1, 0, 0), P(1, 1, 0)], // z = 0
  [P(0, 0, 1), P(1, 0, 1), P(0, 1, 1), P(1, 1, 1)], // z = 1
  [P(0, 0, 0), P(1, 0, 0), P(0, 0, 1), P(1, 0, 1)], // y = 0
  [P(0, 1, 0), P(0, 1, 1), P(1, 1, 0), P(1, 1, 1)], // y = 1
  [P(0, 0, 0), P(0, 0, 1), P(0, 1, 0), P(0, 1, 1)], // x = 0
  [P(1, 0, 0), P(1, 1, 0), P(1, 0, 1), P(1, 1, 1)], // x = 1
];
export const displayLists = concat(Uint8Array.of(9, 9, 9, 9), ...cubeFaces.map((f) => stripTable([f])));
export const keywords = '<?xml version="1.0"?><Keywords><Configuration id="0" Name="Default" Type="ConfigurationManager" Material="AISI 1020"/></Keywords>';

export function assemblyTree(refs) {
  return `<?xml version="1.0" encoding="UTF-8"?><swSolidWorks xmlns="http://www.solidworks.com/sw2003/schema"><swHeader>
<swFile id="1" swDocType="ASSEMBLY" swPath="C:\\cad\\robot.SLDASM"/><swFile id="2" swDocType="PART" swPath="C:\\cad\\Cube.SLDPRT"/></swHeader>
<swModelList><swModel id="10" swName="robot" swFileRef="1">${refs}</swModel><swModel id="11" swName="Cube" swFileRef="2"/></swModelList>
<swConfigurationList><swConfiguration swName="Default" swModelRef="10" swMostRecentConfiguration="YES"/></swConfigurationList></swSolidWorks>`;
}
export const ref = (n, t, extra = '') => `<swReference swName="Cube" swReferenceNumber="${n}" swModelRef="11" swTransform="${t}" ${extra}/>`;
export const move = (x, y, z) => `1 0 0 0 0 1 0 0 0 0 1 0 ${x} ${y} ${z} 1`;


/** A cube part file and an assembly placing it three times (one mirrored). */
export function cubeAssemblyFiles() {
  const part = container(block('Contents/DisplayLists', displayLists), block('swXmlContents/KeyWords', keywords));
  const tree = assemblyTree(ref(1, move(0, 0, 0)) + ref(2, move(0.1, 0, 0)) + ref(3, '-1 0 0 0 0 1 0 0 0 0 1 0 0 0.1 0 1'));
  return { part, assembly: container(block('swXmlContents/COMPINSTANCETREE', tree)) };
}
