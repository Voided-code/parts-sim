// Triangle selections (patches) as they are written into a .psim file: sorted runs [start, count, ...].

/** { tris: Int32Array, clip, label } -> JSON-able object. */
export function patchToJson(p) {
  const sorted = Int32Array.from(p.tris).sort();
  const runs = [];
  for (let i = 0; i < sorted.length;) {
    let j = i + 1;
    while (j < sorted.length && sorted[j] === sorted[j - 1] + 1) j++;
    runs.push(sorted[i], j - i);
    i = j;
  }
  return { tris: runs, clip: p.clip ? { center: Array.from(p.clip.center), radius: p.clip.radius } : null, ...(p.label ? { label: p.label } : {}) };
}

/** Inverse of patchToJson. Checks every triangle number against `nTri`. */
export function patchFromJson(j, nTri) {
  const runs = j?.tris;
  if (!Array.isArray(runs) || runs.length % 2) throw new Error('A selection in the file is malformed.');
  let n = 0;
  for (let i = 0; i < runs.length; i += 2) {
    if (!Number.isInteger(runs[i]) || !Number.isInteger(runs[i + 1]) || runs[i] < 0 || runs[i + 1] < 1 || runs[i] + runs[i + 1] > nTri) {
      throw new Error('A selection in the file refers to triangles the part does not have.');
    }
    n += runs[i + 1];
    if (n > nTri) throw new Error('A selection in the file is longer than the part.');
  }
  const tris = new Int32Array(n);
  let o = 0;
  for (let i = 0; i < runs.length; i += 2) for (let k = 0; k < runs[i + 1]; k++) tris[o++] = runs[i] + k;
  const c = j.clip;
  const clip = c && Array.isArray(c.center) && c.center.length === 3 && Number.isFinite(c.radius) ? { center: c.center.map(Number), radius: Number(c.radius) } : null;
  return { tris, clip, ...(j.label ? { label: String(j.label) } : {}) };
}
