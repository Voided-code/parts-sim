// SolidWorks reader tests on synthetic files: every header field is written deliberately,
// so a reader that is subtly wrong fails here instead of looking plausible on real parts.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readSolidWorks, assemblyPartNames, readBlocks, matchMaterial } from '../src/core/solidworks.js';
import { block, container, displayLists, keywords, assemblyTree, ref, move } from './fixtures/solidworks.js';
import { primaryFile } from '../src/core/importers.js';
import { MATERIALS } from '../src/core/materials.js';
import { buildPart } from '../src/core/mesh.js';

const enc = new TextEncoder();

test('part: display mesh, CAD faces, millimetres and material', () => {
  const file = container(block('Contents/Config-0-Partition', 'PS\0\0\0 not parsed'), block('Contents/DisplayLists', displayLists), block('swXmlContents/KeyWords', keywords));
  assert.deepEqual(readBlocks(file).map((b) => b.name), ['Contents/Config-0-Partition', 'Contents/DisplayLists', 'swXmlContents/KeyWords']);
  const src = readSolidWorks(file, { name: 'cube.SLDPRT' });
  assert.equal(src.units, 'mm');
  assert.equal(src.positions.length / 9, 12);
  assert.deepEqual([...new Set(src.faceIds)], [0, 1, 2, 3, 4, 5]);
  assert.equal(src.material, 'AISI 1020');
  assert.equal(matchMaterial(src.material, MATERIALS), 'steel-1020');
  const part = buildPart({ ...src, name: 'cube' });
  assert.deepEqual(part.bbox.size.map((v) => Math.round(v * 1000) / 1000), [20, 20, 20]);
  assert.ok(Math.abs(part.volume - 8000) < 1e-3, `volume ${part.volume}`);
  assert.equal(part.faceCount, 6);
  assert.ok(part.neighbors.every((n) => n >= 0), 'watertight');
});

test('unreadable SolidWorks files explain what to do', () => {
  const noMesh = container(block('Contents/Config-0-Partition', 'PS\0\0\0 geometry only'));
  assert.throws(() => readSolidWorks(noMesh, { name: 'a.SLDPRT' }), /saved without display data.*STEP/s);
  const ole = Uint8Array.of(0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1, 0, 0);
  assert.throws(() => readSolidWorks(ole, { name: 'old.SLDPRT' }), /pre-2015/);
  assert.throws(() => readSolidWorks(noMesh, { name: 'a.SLDDRW' }), /drawings/);
  assert.throws(() => readSolidWorks(enc.encode('hello world, not CAD'), { name: 'x.SLDPRT' }), /does not look like a SolidWorks file/);
  const corrupt = container(block('Contents/DisplayLists', displayLists, { corrupt: true }));
  assert.throws(() => readSolidWorks(corrupt, { name: 'c.SLDPRT' }), /checksum/);
});

test('material names map onto the library', () => {
  const cases = { 'Alloy Steel (SS)': 'steel-alloy', '6061-T6 (SS)': 'al-6061', 'AISI 304': 'ss-304', 'ABS PC': 'abs', 'PLA': 'pla', 'Ti-6Al-4V Solution treated and aged (SS)': 'ti-64', 'Plastic Rubber': null };
  for (const [name, id] of Object.entries(cases)) assert.equal(matchMaterial(name, MATERIALS), id, name);
});

test('assembly: components placed from their part files, mirrored ones re-wound', () => {
  const tree = assemblyTree(ref(1, move(0, 0, 0)) + ref(2, move(0.1, 0, 0)) + ref(3, '-1 0 0 0 0 1 0 0 0 0 1 0 0 0.1 0 1') + ref(4, move(9, 9, 9), 'swSuppressed="YES"') + ref(5, move(0, 0, 0.3), 'swHidden="YES"'));
  const asm = container(block('swXmlContents/COMPINSTANCETREE', tree));
  assert.deepEqual(assemblyPartNames(asm), ['Cube.SLDPRT']);
  assert.throws(() => readSolidWorks(asm, { name: 'robot.SLDASM' }), /part files/);
  const cube = container(block('Contents/DisplayLists', displayLists));
  const src = readSolidWorks(asm, { name: 'robot.SLDASM', resolvePart: (n) => (n.toLowerCase() === 'cube.sldprt' ? cube : null) });
  assert.match(src.info, /3 of 3 components/);
  assert.equal(src.positions.length / 9, 36);
  assert.equal(new Set(src.faceIds).size, 18);
  const part = buildPart({ ...src, name: 'robot' });
  // cubes at x 0..20, 100..120 (mm), and a mirrored one at x -20..0, y 100..120
  assert.deepEqual(part.bbox.size.map((v) => Math.round(v)), [140, 120, 20]);
  assert.ok(Math.abs(part.volume - 3 * 8000) < 1e-2, `mirrored copies keep positive volume: ${part.volume}`);
});

test('multi-file selections open the assembly and pass the parts along', () => {
  const files = [new File(['a'], 'bracket.SLDPRT'), new File(['b'], 'robot.SLDASM'), new File(['c'], 'notes.txt')];
  const { primary, companions } = primaryFile(files);
  assert.equal(primary.name, 'robot.SLDASM');
  assert.deepEqual(companions.map((f) => f.name), ['bracket.SLDPRT', 'notes.txt']);
  assert.equal(primaryFile([new File(['x'], 'readme.md'), new File(['y'], 'part.stl')]).primary.name, 'part.stl');
});

test('assembly: multi-configuration parts saved at another size are left out, redesigned parts kept', () => {
  // The model record says the cube is 10 mm; the part file holds a 20 mm cube.
  const tree = assemblyTree(ref(1, move(0, 0, 0))).replace('<swModel id="11" swName="Cube" swFileRef="2"/>',
    '<swModel id="11" swName="Cube" swConfigurationName="M10" swFileRef="2" swBoundingBox="0 0 0 0.01 0.01 0.01"/>');
  const asm = container(block('swXmlContents/COMPINSTANCETREE', tree));
  const configs = (n) => `<?xml version="1.0"?><Keywords>${Array.from({ length: n }, (_, i) => `<Configuration id="${i}" Name="C${i}" Type="ConfigurationManager"/>`).join('')}</Keywords>`;
  const toolbox = container(block('Contents/DisplayLists', displayLists), block('swXmlContents/KeyWords', configs(3)));
  assert.throws(() => readSolidWorks(asm, { name: 'a.SLDASM', resolvePart: () => toolbox }), /part files/);
  const redesigned = container(block('Contents/DisplayLists', displayLists), block('swXmlContents/KeyWords', configs(1)));
  const src = readSolidWorks(asm, { name: 'a.SLDASM', resolvePart: () => redesigned });
  assert.match(src.info, /1 of 1 components/);
  const mixed = assemblyTree(ref(1, move(0, 0, 0)) + ref(2, move(0.1, 0, 0))).replace('<swModel id="11" swName="Cube" swFileRef="2"/>',
    '<swModel id="11" swName="Cube" swConfigurationName="M10" swFileRef="2" swBoundingBox="0 0 0 0.01 0.01 0.01"/>');
  // every component excluded -> a clear error; with a matching size -> placed
  const exact = '<swModel id="11" swName="Cube" swConfigurationName="M20" swFileRef="2" swBoundingBox="0 0 0 0.02 0.02 0.02"/>';
  const ok = readSolidWorks(container(block('swXmlContents/COMPINSTANCETREE', mixed.replace(/<swModel id="11"[^>]*\/>/, exact))), { name: 'b.SLDASM', resolvePart: () => toolbox });
  assert.match(ok.info, /2 of 2 components/);
  assert.deepEqual(ok.warnings, []);
});
