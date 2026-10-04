// File import: every format is reduced to { positions, index?, faceIds?, name }.
import * as THREE from 'three';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader.js';
import { ThreeMFLoader } from 'three/examples/jsm/loaders/3MFLoader.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { unzipSync, strFromU8 } from 'three/examples/jsm/libs/fflate.module.js';
import { readSolidWorks, assemblyPartNames } from './solidworks.js';

export const ACCEPT = '.psim,.step,.stp,.iges,.igs,.brep,.brp,.sldprt,.sldasm,.slddrw,.stl,.obj,.3mf,.ply,.glb,.gltf';

/** Pick the file to open from a multi-file selection; the rest travel along as companions (assembly parts). */
export function primaryFile(files) {
  const list = Array.from(files);
  const known = (f) => ACCEPT.split(',').includes(`.${f.name.split('.').pop().toLowerCase()}`);
  const primary = list.find((f) => /\.sldasm$/i.test(f.name)) || list.find(known) || list[0] || null;
  return { primary, companions: list.filter((f) => f !== primary) };
}

/**
 * @param {File} file
 * @param {{companions?: File[], readSibling?: (fileName: string) => Promise<ArrayBuffer|Uint8Array|null>}} opts
 *        companions: other selected files (a SolidWorks assembly's parts); readSibling: desktop app
 *        reads a file from the same folder as `file`.
 * @returns {Promise<{positions: Float32Array, index: Uint32Array|null, faceIds: Int32Array|null, name: string, units: string|null, info: string, material?: string|null, warnings?: string[]}>}
 */
export async function importFile(file, { companions = [], readSibling = null } = {}) {
  const ext = file.name.split('.').pop().toLowerCase();
  const name = file.name.replace(/\.[^.]+$/, '');
  const buffer = await file.arrayBuffer();
  if (!buffer.byteLength) throw new Error('The selected file is empty.');
  switch (ext) {
    case 'step':
    case 'stp':
      return { ...(await readCAD('step', buffer)), name };
    case 'iges':
    case 'igs':
      return { ...(await readCAD('iges', buffer)), name };
    case 'brep':
    case 'brp':
      return { ...(await readCAD('brep', buffer)), name };
    case 'sldprt':
    case 'sldasm':
    case 'slddrw':
      return { ...(await readSolidWorksFile(buffer, file.name, companions, readSibling)), name };
    case 'stl':
      return { ...fromGeometry(new STLLoader().parse(buffer)), name, units: null, info: 'STL mesh' };
    case 'ply':
      return { ...fromGeometry(new PLYLoader().parse(buffer)), name, units: null, info: 'PLY mesh' };
    case 'obj':
      return { ...fromObject(new OBJLoader().parse(new TextDecoder().decode(buffer))), name, units: null, info: 'OBJ mesh' };
    case '3mf': {
      const scale = threeMFScale(buffer);
      const root = new ThreeMFLoader().parse(buffer);
      root.scale.multiplyScalar(scale);
      return { ...fromObject(root), name, units: 'mm', info: '3MF mesh' };
    }
    case 'glb':
    case 'gltf': {
      validateGLTFResources(buffer);
      const manager = new THREE.LoadingManager();
      manager.setURLModifier((url) => {
        if (!/^(data:|blob:)/i.test(url)) throw new Error('External glTF resources are not supported. Export a self-contained GLB or embed its resources.');
        return url;
      });
      const gltf = await new Promise((res, rej) => new GLTFLoader(manager).parse(buffer, '', res, rej));
      return { ...fromObject(gltf.scene), name, units: 'm', info: 'glTF mesh' };
    }
    default:
      throw new Error(`Unsupported file type ".${ext}". Use STEP, IGES, SolidWorks (SLDPRT/SLDASM), STL, OBJ, 3MF, PLY or GLB.`);
  }
}

async function readSolidWorksFile(buffer, fileName, companions, readSibling) {
  const parts = new Map();
  if (/\.sldasm$/i.test(fileName)) {
    const byName = new Map(companions.map((f) => [f.name.toLowerCase(), f]));
    for (const n of assemblyPartNames(buffer)) {
      const f = byName.get(n.toLowerCase());
      let data = f ? await f.arrayBuffer() : null;
      if (!data && readSibling) data = await readSibling(n).catch(() => null);
      if (data) parts.set(n.toLowerCase(), new Uint8Array(data));
    }
  }
  return readSolidWorks(buffer, { name: fileName, resolvePart: (n) => parts.get(n.toLowerCase()) || null });
}

function readCAD(format, buffer) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./occt.worker.js', import.meta.url), { type: 'module' });
    worker.onmessage = (ev) => {
      worker.terminate();
      const d = ev.data;
      if (!d.ok) return reject(new Error(d.error));
      resolve({
        positions: d.positions,
        index: d.index,
        faceIds: d.faceIds,
        units: format === 'brep' ? null : 'mm',
        info: `${format.toUpperCase()} B-rep, ${d.bodies} bod${d.bodies === 1 ? 'y' : 'ies'}`,
      });
    };
    worker.onerror = (e) => {
      worker.terminate();
      reject(new Error(e.message || 'CAD import worker failed'));
    };
    worker.postMessage({ format, buffer }, [buffer]);
  });
}

function fromGeometry(geometry) {
  try {
    const pos = geometry.getAttribute('position');
    if (!pos?.count) throw new Error('No meshes found in file.');
    return {
      positions: Float32Array.from(pos.array),
      index: geometry.index ? Uint32Array.from(geometry.index.array) : null,
      faceIds: null,
    };
  } finally {
    geometry.dispose();
  }
}

function fromObject(root) {
  root.updateMatrixWorld(true);
  const positions = [];
  const v = new THREE.Vector3();
  try {
    root.traverse((obj) => {
      if (!obj.isMesh || !obj.geometry) return;
      const g = obj.geometry;
      const pos = g.getAttribute('position');
      if (!pos) return;
      const idx = g.index;
      const n = idx ? idx.count : pos.count;
      if (n % 3 !== 0) throw new Error('The mesh does not contain complete triangles.');
      const mirrored = obj.matrixWorld.determinant() < 0;
      if (obj.isSkinnedMesh) obj.skeleton.update();
      for (let t = 0; t < n; t += 3) {
        for (const corner of mirrored ? [0, 2, 1] : [0, 1, 2]) {
          const i = t + corner;
          obj.getVertexPosition(idx ? idx.getX(i) : i, v).applyMatrix4(obj.matrixWorld);
          positions.push(v.x, v.y, v.z);
        }
      }
    });
  } finally {
    disposeObject(root);
  }
  if (!positions.length) throw new Error('No meshes found in file.');
  return { positions: Float32Array.from(positions), index: null, faceIds: null };
}

function disposeObject(root) {
  const disposed = new Set();
  const dispose = (resource) => {
    if (!resource?.dispose || disposed.has(resource)) return;
    disposed.add(resource);
    resource.dispose();
  };
  root.traverse((obj) => {
    dispose(obj.geometry);
    for (const material of Array.isArray(obj.material) ? obj.material : [obj.material]) {
      if (!material) continue;
      for (const value of Object.values(material)) if (value?.isTexture) {
        if (value.source?.data?.close) value.source.data.close();
        dispose(value);
      }
      dispose(material);
    }
  });
}

// ThreeMFLoader retains source coordinates; normalize the model's declared units.
function threeMFScale(buffer) {
  const files = unzipSync(new Uint8Array(buffer));
  const parser = new DOMParser();
  const relationships = files['_rels/.rels'];
  if (!relationships) throw new Error('The 3MF archive is missing its model relationship.');
  const rels = parser.parseFromString(strFromU8(relationships), 'application/xml');
  const target = Array.from(rels.getElementsByTagName('Relationship'))
    .find((r) => /\.model$/i.test(r.getAttribute('Target') || ''))?.getAttribute('Target')?.replace(/^\//, '');
  if (!target || !files[target]) throw new Error('The 3MF archive is missing its model.');
  const unitOf = (data) => parser.parseFromString(strFromU8(data), 'application/xml').documentElement.getAttribute('unit') || 'millimeter';
  const unit = unitOf(files[target]);
  for (const [path, data] of Object.entries(files)) {
    if (/\.model$/i.test(path) && unitOf(data) !== unit) throw new Error('3MF assemblies with mixed model units are not supported. Export the assembly using one unit.');
  }
  const scale = { micron: 0.001, millimeter: 1, centimeter: 10, inch: 25.4, foot: 304.8, meter: 1000 }[unit];
  if (!scale) throw new Error(`Unsupported 3MF unit: ${unit}.`);
  return scale;
}

// Validate before parsing so imports never request a remote buffer or texture.
function validateGLTFResources(buffer) {
  const data = new DataView(buffer);
  let json;
  if (buffer.byteLength >= 20 && data.getUint32(0, true) === 0x46546c67) {
    const size = data.getUint32(12, true);
    if (data.getUint32(16, true) !== 0x4e4f534a || size > buffer.byteLength - 20) throw new Error('The GLB JSON chunk is invalid.');
    json = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 20, size)).trim());
  } else {
    json = JSON.parse(new TextDecoder().decode(buffer));
  }
  for (const resource of [...(json.buffers || []), ...(json.images || [])]) {
    if (resource.uri && !/^data:/i.test(resource.uri)) throw new Error('External glTF resources are not supported. Export a self-contained GLB or embed its resources.');
  }
}
