// Shared conversion used by the CAD worker and the import regression tests.
export function readCADMesh(occt, format, buffer, quality) {
  const result = occt.ReadFile(format, new Uint8Array(buffer), {
    linearUnit: 'millimeter',
    linearDeflectionType: 'bounding_box_ratio',
    linearDeflection: quality === 'fine' ? 0.0005 : 0.002,
    angularDeflection: quality === 'fine' ? 0.2 : 0.4,
  });
  if (!result?.success) throw new Error('OpenCascade could not read this file.');
  const meshes = result.meshes || [];
  let nVert = 0, nIdx = 0;
  for (const m of meshes) {
    nVert += m.attributes.position.array.length;
    nIdx += m.index.array.length;
  }
  if (!nVert || !nIdx) throw new Error('The CAD file contains no tessellated surfaces. Export a solid or surface body.');
  const positions = new Float32Array(nVert);
  const index = new Uint32Array(nIdx);
  const faceIds = new Int32Array(nIdx / 3).fill(-1);
  let vOff = 0, iOff = 0, face = 0;
  for (const m of meshes) {
    const p = m.attributes.position.array;
    positions.set(p, vOff);
    const base = vOff / 3;
    const idx = m.index.array;
    for (let i = 0; i < idx.length; i++) index[iOff + i] = idx[i] + base;
    const triOff = iOff / 3;
    const nTri = idx.length / 3;
    for (const f of m.brep_faces || []) {
      for (let t = f.first; t <= f.last; t++) faceIds[triOff + t] = face;
      face++;
    }
    for (let t = 0; t < nTri; t++) if (faceIds[triOff + t] < 0) faceIds[triOff + t] = face;
    face++;
    vOff += p.length;
    iOff += idx.length;
  }
  return { positions, index, faceIds, bodies: meshes.length };
}
