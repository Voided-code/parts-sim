// Mesh processing: welding, orientation, refinement, faces, normals, edges, sampling.
//
// A "Part" is the single source of truth for geometry. All studies (structural, airflow)
// read from it, and all results are stored per unique vertex of part.vertices.

/**
 * @typedef {object} Part
 * @property {string} name
 * @property {Float32Array} vertices      unique vertex positions (xyz), model units
 * @property {Uint32Array} tris           3 vertex indices per triangle (outward CCW)
 * @property {number} nVert
 * @property {number} nTri
 * @property {Float32Array} triNormal
 * @property {Float32Array} triArea
 * @property {Float32Array} vertNormal    area-weighted smooth normals
 * @property {Int32Array} neighbors       triangle across each edge (-1 if none)
 * @property {Int32Array} faceOf          face id per triangle
 * @property {number} faceCount
 * @property {boolean} brepFaces          faces come from the CAD B-rep (STEP/IGES)
 * @property {{min:number[],max:number[],size:number[],diag:number}} bbox
 * @property {number} volume
 * @property {number} area
 * @property {Uint32Array} edges          feature edge vertex pairs
 * @property {{position: Float32Array, normal: Float32Array, srcVert: Uint32Array}} display
 */

/**
 * @param {{positions: ArrayLike<number>, index?: ArrayLike<number>|null, faceIds?: Int32Array|null, name?: string}} src
 * @param {{maxTris?: number, faceAngle?: number}} opts
 * @returns {Part}
 */
export function buildPart(src, { maxTris = 350000, faceAngle = 20 } = {}) {
  validateMesh(src);
  let { vertices, tris, faceIds } = weld(src.positions, src.index, src.faceIds);
  ({ tris, faceIds } = dropDegenerate(vertices, tris, faceIds));
  if (tris.length === 0) throw new Error('The file contains no usable triangles.');
  vertices = compactVertices(vertices, tris);
  placeOnGround(vertices);
  orientShells(vertices, tris);

  const bbox = boundingBox(vertices);
  const area0 = totalArea(vertices, tris);
  const target = Math.max(bbox.diag / 140, Math.sqrt(area0 / (0.4 * maxTris)));
  ({ vertices, tris, faceIds } = refine(vertices, tris, faceIds, target, maxTris));

  const part = {
    name: src.name || 'Part',
    vertices,
    tris,
    nVert: vertices.length / 3,
    nTri: tris.length / 3,
    bbox,
    brepFaces: !!faceIds,
  };
  computeTriangleData(part);
  part.neighbors = triangleNeighbors(tris, part.nVert);
  part.vertNormal = smoothNormals(part);
  if (faceIds) {
    part.faceOf = faceIds;
    part.faceCount = compactFaceIds(faceIds);
  } else {
    setSmoothFaces(part, faceAngle);
  }
  part.volume = Math.abs(signedVolume(vertices, tris));
  part.edges = featureEdges(part, 30);
  part.display = creasedDisplay(part, 35);
  return part;
}

/**
 * Rebuilds a part from the arrays of an already built one (a .psim file), without welding, refining
 * or moving it: vertex and triangle numbers stay exactly as saved, because results and selections
 * refer to them. Faces come from `faceOf` for CAD parts, otherwise from the smooth-face angle.
 */
export function restorePart({ name, vertices, tris, faceOf = null, brepFaces = false, faceCount = 0, faceAngle = 20 }) {
  const part = {
    name: name || 'Part',
    vertices,
    tris,
    nVert: vertices.length / 3,
    nTri: tris.length / 3,
    bbox: boundingBox(vertices),
    brepFaces: !!brepFaces,
  };
  computeTriangleData(part);
  part.neighbors = triangleNeighbors(tris, part.nVert);
  part.vertNormal = smoothNormals(part);
  if (brepFaces && faceOf) {
    part.faceOf = faceOf;
    part.faceCount = faceCount;
  } else {
    setSmoothFaces(part, faceAngle || 20);
  }
  part.volume = Math.abs(signedVolume(vertices, tris));
  part.edges = featureEdges(part, 30);
  part.display = creasedDisplay(part, 35);
  return part;
}

function validateMesh({ positions, index, faceIds }) {
  if (!positions?.length || positions.length % 3 !== 0) throw new Error('The file contains no valid vertex positions.');
  for (const v of positions) if (!Number.isFinite(v)) throw new Error('The mesh contains non-finite vertex coordinates.');
  const nVert = positions.length / 3;
  const nCorner = index ? index.length : nVert;
  if (!nCorner || nCorner % 3 !== 0) throw new Error('The mesh does not contain complete triangles.');
  if (index) for (const i of index) {
    if (!Number.isInteger(i) || i < 0 || i >= nVert) throw new Error('The mesh contains an invalid triangle index.');
  }
  if (faceIds && faceIds.length !== nCorner / 3) throw new Error('The CAD face data does not match its triangles.');
}

function weld(positions, index, faceIds) {
  const nIn = positions.length / 3;
  const used = index ? new Uint8Array(nIn) : null;
  if (used) for (const i of index) used[i] = 1;
  let min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nIn; i++) {
    if (used && !used[i]) continue;
    for (let d = 0; d < 3; d++) {
      const v = positions[3 * i + d];
      if (v < min[d]) min[d] = v;
      if (v > max[d]) max[d] = v;
    }
  }
  const diag = Math.hypot(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
  const inv = 100000 / diag;
  const P = 131072;
  const map = new Map();
  const remap = new Uint32Array(nIn);
  const out = [];
  for (let i = 0; i < nIn; i++) {
    if (used && !used[i]) continue;
    const x = positions[3 * i], y = positions[3 * i + 1], z = positions[3 * i + 2];
    const key = Math.round((x - min[0]) * inv) + P * (Math.round((y - min[1]) * inv) + P * Math.round((z - min[2]) * inv));
    let id = map.get(key);
    if (id === undefined) {
      id = out.length / 3;
      map.set(key, id);
      out.push(x, y, z);
    }
    remap[i] = id;
  }
  const nTri = index ? index.length / 3 : nIn / 3;
  const tris = new Uint32Array(nTri * 3);
  for (let t = 0; t < nTri * 3; t++) tris[t] = remap[index ? index[t] : t];
  return { vertices: Float32Array.from(out), tris, faceIds: faceIds ? Int32Array.from(faceIds) : null };
}

function compactVertices(vertices, tris) {
  const remap = new Int32Array(vertices.length / 3).fill(-1);
  const out = [];
  for (let i = 0; i < tris.length; i++) {
    const v = tris[i];
    if (remap[v] < 0) {
      remap[v] = out.length / 3;
      out.push(vertices[3 * v], vertices[3 * v + 1], vertices[3 * v + 2]);
    }
    tris[i] = remap[v];
  }
  return Float32Array.from(out);
}

/** Make each shell coherent, with enclosed cavity shells facing into the cavity. */
function orientShells(V, tris) {
  const neighbors = triangleNeighbors(tris);
  const seen = new Uint8Array(tris.length / 3);
  const shells = [];
  for (let seed = 0; seed < seen.length; seed++) {
    if (seen[seed]) continue;
    const triangles = [seed];
    seen[seed] = 1;
    let closed = true;
    const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
    for (let q = 0; q < triangles.length; q++) {
      const t = triangles[q];
      // Find neighbours by vertex pair because the current triangle may have been flipped.
      for (let e = 0; e < 3; e++) {
        const a = tris[3 * t + e], b = tris[3 * t + (e + 1) % 3];
        for (let d = 0; d < 3; d++) {
          min[d] = Math.min(min[d], V[3 * a + d]);
          max[d] = Math.max(max[d], V[3 * a + d]);
        }
        const other = neighbors[3 * t + e];
        if (other < 0) { closed = false; continue; }
        if (seen[other]) continue;
        for (let oe = 0; oe < 3; oe++) {
          if (tris[3 * other + oe] === a && tris[3 * other + (oe + 1) % 3] === b) {
            [tris[3 * other + 1], tris[3 * other + 2]] = [tris[3 * other + 2], tris[3 * other + 1]];
            [neighbors[3 * other], neighbors[3 * other + 2]] = [neighbors[3 * other + 2], neighbors[3 * other]];
            break;
          }
        }
        seen[other] = 1;
        triangles.push(other);
      }
    }
    shells.push({ triangles, closed, min, max });
  }
  for (const shell of shells) {
    if (!shell.closed) continue;
    const first = shell.triangles[0];
    const point = [0, 0, 0];
    for (let d = 0; d < 3; d++) {
      for (let c = 0; c < 3; c++) point[d] += V[3 * tris[3 * first + c] + d] / 3;
    }
    let depth = 0;
    for (const other of shells) {
      if (other === shell || !other.closed || !other.min.every((m, d) => m < shell.min[d] && other.max[d] > shell.max[d])) continue;
      if (insideShell(V, tris, other.triangles, point)) depth++;
    }
    let volume = 0;
    // A local origin avoids cancellation in models far from their source origin.
    for (const t of shell.triangles) {
      const a = 3 * tris[3 * t];
      const n = triCross(V, tris[3 * t], tris[3 * t + 1], tris[3 * t + 2]);
      volume += (V[a] - point[0]) * n[0] + (V[a + 1] - point[1]) * n[1] + (V[a + 2] - point[2]) * n[2];
    }
    if ((volume < 0) !== (depth % 2 === 1)) {
      for (const t of shell.triangles) [tris[3 * t + 1], tris[3 * t + 2]] = [tris[3 * t + 2], tris[3 * t + 1]];
    }
  }
}

function insideShell(V, tris, triangles, point) {
  let angle = 0;
  for (const t of triangles) {
    const v = [0, 1, 2].map((c) => {
      const a = 3 * tris[3 * t + c];
      return [V[a] - point[0], V[a + 1] - point[1], V[a + 2] - point[2]];
    });
    const [a, b, c] = v, [la, lb, lc] = v.map((p) => Math.hypot(...p));
    const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
    const det = a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
    angle += 2 * Math.atan2(det, la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb);
  }
  return Math.abs(angle) > 2 * Math.PI;
}

function dropDegenerate(V, tris, faceIds) {
  const keep = [];
  const keepFace = [];
  for (let t = 0; t < tris.length / 3; t++) {
    const a = tris[3 * t], b = tris[3 * t + 1], c = tris[3 * t + 2];
    if (a === b || b === c || a === c) continue;
    const n = triCross(V, a, b, c);
    if (n[0] === 0 && n[1] === 0 && n[2] === 0) continue;
    keep.push(a, b, c);
    if (faceIds) keepFace.push(faceIds[t]);
  }
  return { tris: Uint32Array.from(keep), faceIds: faceIds ? Int32Array.from(keepFace) : null };
}

function triCross(V, a, b, c) {
  const ax = V[3 * a], ay = V[3 * a + 1], az = V[3 * a + 2];
  const ux = V[3 * b] - ax, uy = V[3 * b + 1] - ay, uz = V[3 * b + 2] - az;
  const vx = V[3 * c] - ax, vy = V[3 * c + 1] - ay, vz = V[3 * c + 2] - az;
  return [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx];
}

function signedVolume(V, tris) {
  let vol = 0;
  for (let t = 0; t < tris.length; t += 3) {
    const a = 3 * tris[t], b = 3 * tris[t + 1], c = 3 * tris[t + 2];
    vol +=
      V[a] * (V[b + 1] * V[c + 2] - V[b + 2] * V[c + 1]) -
      V[a + 1] * (V[b] * V[c + 2] - V[b + 2] * V[c]) +
      V[a + 2] * (V[b] * V[c + 1] - V[b + 1] * V[c]);
  }
  return vol / 6;
}

function totalArea(V, tris) {
  let a = 0;
  for (let t = 0; t < tris.length; t += 3) {
    const n = triCross(V, tris[t], tris[t + 1], tris[t + 2]);
    a += 0.5 * Math.hypot(n[0], n[1], n[2]);
  }
  return a;
}

export function boundingBox(V) {
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < V.length; i += 3) {
    for (let d = 0; d < 3; d++) {
      if (V[i + d] < min[d]) min[d] = V[i + d];
      if (V[i + d] > max[d]) max[d] = V[i + d];
    }
  }
  const size = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  return { min, max, size, diag: Math.hypot(...size) };
}

/** Center on X/Z and rest the part on the Y = 0 ground plane. */
function placeOnGround(V) {
  const { min, max } = boundingBox(V);
  const off = [-(min[0] + max[0]) / 2, -min[1], -(min[2] + max[2]) / 2];
  for (let i = 0; i < V.length; i += 3) {
    V[i] += off[0];
    V[i + 1] += off[1];
    V[i + 2] += off[2];
  }
}

/** Rotate vertices 90 degrees about a world axis (0 = X, 1 = Y, 2 = Z) and re-seat on the ground. */
export function rotatePositions90(positions, axis) {
  const V = Float32Array.from(positions);
  for (let i = 0; i < V.length; i += 3) {
    const x = V[i], y = V[i + 1], z = V[i + 2];
    if (axis === 0) { V[i + 1] = -z; V[i + 2] = y; }
    else if (axis === 1) { V[i] = z; V[i + 2] = -x; }
    else { V[i] = -y; V[i + 1] = x; }
  }
  return V;
}

/**
 * Conforming refinement: split every edge longer than `maxLen` at its midpoint until none remain.
 * Keeps the mesh watertight (no T-junctions) so deformed shapes never crack.
 */
function refine(V0, tris0, faceIds0, maxLen, maxTris) {
  let V = Array.from(V0);
  let tris = tris0;
  let faceIds = faceIds0;
  const maxLen2 = maxLen * maxLen;
  for (let pass = 0; pass < 12; pass++) {
    const mids = new Map();
    const nT = tris.length / 3;
    if (nT >= maxTris) break;
    const previousLength = V.length;
    let refinedCount = nT;
    const len2 = (a, b) => {
      const dx = V[3 * a] - V[3 * b], dy = V[3 * a + 1] - V[3 * b + 1], dz = V[3 * a + 2] - V[3 * b + 2];
      return dx * dx + dy * dy + dz * dz;
    };
    const K = 67108864; // 2^26
    const key = (a, b) => (a < b ? a * K + b : b * K + a);
    let splits = 0;
    for (let t = 0; t < nT; t++) {
      for (let e = 0; e < 3; e++) {
        const a = tris[3 * t + e], b = tris[3 * t + ((e + 1) % 3)];
        if (len2(a, b) > maxLen2) {
          refinedCount++;
          const k = key(a, b);
          if (!mids.has(k)) {
            mids.set(k, V.length / 3);
            V.push((V[3 * a] + V[3 * b]) / 2, (V[3 * a + 1] + V[3 * b + 1]) / 2, (V[3 * a + 2] + V[3 * b + 2]) / 2);
            splits++;
          }
        }
      }
    }
    if (splits === 0) break;
    // Reject the complete pass, keeping both the cap and a conforming surface.
    if (refinedCount > maxTris) { V.length = previousLength; break; }
    const out = [];
    const outFace = [];
    const v = [0, 0, 0], m = [0, 0, 0];
    for (let t = 0; t < nT; t++) {
      let count = 0;
      for (let e = 0; e < 3; e++) {
        v[e] = tris[3 * t + e];
      }
      for (let e = 0; e < 3; e++) {
        const mm = mids.get(key(v[e], v[(e + 1) % 3]));
        m[e] = mm === undefined ? -1 : mm;
        if (mm !== undefined) count++;
      }
      const f = faceIds ? faceIds[t] : 0;
      const emit = (a, b, c) => {
        out.push(a, b, c);
        outFace.push(f);
      };
      if (count === 0) {
        emit(v[0], v[1], v[2]);
      } else if (count === 3) {
        emit(v[0], m[0], m[2]);
        emit(m[0], v[1], m[1]);
        emit(m[2], m[1], v[2]);
        emit(m[0], m[1], m[2]);
      } else if (count === 1) {
        const r = m[0] >= 0 ? 0 : m[1] >= 0 ? 1 : 2; // rotate so the split edge is e0
        const v0 = v[r], v1 = v[(r + 1) % 3], v2 = v[(r + 2) % 3], m0 = m[r];
        emit(v0, m0, v2);
        emit(m0, v1, v2);
      } else {
        const u = m[0] < 0 ? 0 : m[1] < 0 ? 1 : 2; // unsplit edge; rotate it to e2
        const r = (u + 1) % 3;
        const v0 = v[r], v1 = v[(r + 1) % 3], v2 = v[(r + 2) % 3];
        const m0 = m[r], m1 = m[(r + 1) % 3];
        emit(m0, v1, m1);
        if (len2(v0, m1) < len2(m0, v2)) {
          emit(v0, m0, m1);
          emit(v0, m1, v2);
        } else {
          emit(v0, m0, v2);
          emit(m0, m1, v2);
        }
      }
    }
    tris = Uint32Array.from(out);
    if (faceIds) faceIds = Int32Array.from(outFace);
  }
  return { vertices: Float32Array.from(V), tris, faceIds };
}

function computeTriangleData(part) {
  const { vertices: V, tris, nTri } = part;
  part.triNormal = new Float32Array(3 * nTri);
  part.triArea = new Float32Array(nTri);
  let area = 0;
  for (let t = 0; t < nTri; t++) {
    const n = triCross(V, tris[3 * t], tris[3 * t + 1], tris[3 * t + 2]);
    const l = Math.hypot(n[0], n[1], n[2]) || 1;
    part.triNormal[3 * t] = n[0] / l;
    part.triNormal[3 * t + 1] = n[1] / l;
    part.triNormal[3 * t + 2] = n[2] / l;
    part.triArea[t] = 0.5 * l;
    area += 0.5 * l;
  }
  part.area = area;
}

function triangleNeighbors(tris, nVert) {
  const nTri = tris.length / 3;
  const nb = new Int32Array(3 * nTri).fill(-1);
  const edgeMap = new Map();
  const K = 67108864;
  for (let t = 0; t < nTri; t++) {
    for (let e = 0; e < 3; e++) {
      const a = tris[3 * t + e], b = tris[3 * t + ((e + 1) % 3)];
      const k = a < b ? a * K + b : b * K + a;
      const other = edgeMap.get(k);
      if (other === undefined) {
        edgeMap.set(k, 3 * t + e);
      } else if (other >= 0) {
        const ot = (other / 3) | 0;
        nb[other] = t;
        nb[3 * t + e] = ot;
        edgeMap.set(k, -1); // non-manifold edges beyond two triangles stay unlinked
      }
    }
  }
  return nb;
}

function smoothNormals(part) {
  const { tris, nTri, triNormal, triArea } = part;
  const N = new Float32Array(3 * part.nVert);
  for (let t = 0; t < nTri; t++) {
    for (let c = 0; c < 3; c++) {
      const v = tris[3 * t + c];
      for (let d = 0; d < 3; d++) N[3 * v + d] += triNormal[3 * t + d] * triArea[t];
    }
  }
  for (let v = 0; v < part.nVert; v++) {
    const l = Math.hypot(N[3 * v], N[3 * v + 1], N[3 * v + 2]) || 1;
    N[3 * v] /= l;
    N[3 * v + 1] /= l;
    N[3 * v + 2] /= l;
  }
  return N;
}

function compactFaceIds(faceOf) {
  const map = new Map();
  for (let t = 0; t < faceOf.length; t++) {
    let id = map.get(faceOf[t]);
    if (id === undefined) {
      id = map.size;
      map.set(faceOf[t], id);
    }
    faceOf[t] = id;
  }
  return map.size;
}

/** Group triangles into faces by region-growing across edges bent less than `angleDeg`. */
export function setSmoothFaces(part, angleDeg) {
  const { nTri, neighbors, triNormal } = part;
  const cosT = Math.cos((angleDeg * Math.PI) / 180);
  const faceOf = new Int32Array(nTri).fill(-1);
  const stack = new Int32Array(nTri);
  let count = 0;
  for (let s = 0; s < nTri; s++) {
    if (faceOf[s] >= 0) continue;
    let sp = 0;
    stack[sp++] = s;
    faceOf[s] = count;
    while (sp > 0) {
      const t = stack[--sp];
      for (let e = 0; e < 3; e++) {
        const o = neighbors[3 * t + e];
        if (o < 0 || faceOf[o] >= 0) continue;
        const dot =
          triNormal[3 * t] * triNormal[3 * o] + triNormal[3 * t + 1] * triNormal[3 * o + 1] + triNormal[3 * t + 2] * triNormal[3 * o + 2];
        if (dot >= cosT) {
          faceOf[o] = count;
          stack[sp++] = o;
        }
      }
    }
    count++;
  }
  part.faceOf = faceOf;
  part.faceCount = count;
  part.faceAngle = angleDeg;
}

/** Triangle indices of a face. */
export function trianglesOfFace(part, faceId) {
  const out = [];
  for (let t = 0; t < part.nTri; t++) if (part.faceOf[t] === faceId) out.push(t);
  return Int32Array.from(out);
}

/** Triangles connected to `seed` with at least one vertex within `radius` of `center`. */
export function trianglesInSphere(part, seed, center, radius) {
  const { tris, vertices: V, neighbors, nTri } = part;
  const r2 = radius * radius;
  const seen = new Uint8Array(nTri);
  const out = [seed];
  seen[seed] = 1;
  const inside = (t) => {
    for (let c = 0; c < 3; c++) {
      const v = 3 * tris[3 * t + c];
      const dx = V[v] - center[0], dy = V[v + 1] - center[1], dz = V[v + 2] - center[2];
      if (dx * dx + dy * dy + dz * dz <= r2) return true;
    }
    return false;
  };
  for (let q = 0; q < out.length; q++) {
    const t = out[q];
    for (let e = 0; e < 3; e++) {
      const o = neighbors[3 * t + e];
      if (o >= 0 && !seen[o]) {
        seen[o] = 1;
        if (inside(o)) out.push(o);
      }
    }
  }
  return Int32Array.from(out);
}

/**
 * Evenly spaced points over triangles (about one per spacing^2 of area), each carrying its share of area
 * and the triangle it lies on. Points outside an optional clip sphere are dropped.
 */
export function sampleTriangles(part, triList, spacing, clip = null) {
  const { tris, vertices: V, triArea } = part;
  const pts = [], wts = [], ids = [];
  const r2 = clip ? clip.radius * clip.radius : 0;
  for (const t of triList) {
    const a = 3 * tris[3 * t], b = 3 * tris[3 * t + 1], c = 3 * tris[3 * t + 2];
    const m = Math.max(1, Math.min(40, Math.ceil(Math.sqrt(triArea[t] / (spacing * spacing)))));
    const w = triArea[t] / (m * m);
    const push = (s, u) => {
      const r = 1 - s - u;
      const x = r * V[a] + s * V[b] + u * V[c];
      const y = r * V[a + 1] + s * V[b + 1] + u * V[c + 1];
      const z = r * V[a + 2] + s * V[b + 2] + u * V[c + 2];
      if (clip) {
        const dx = x - clip.center[0], dy = y - clip.center[1], dz = z - clip.center[2];
        if (dx * dx + dy * dy + dz * dz > r2) return;
      }
      pts.push(x, y, z);
      wts.push(w);
      ids.push(t);
    };
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < m - i; j++) {
        push((i + 1 / 3) / m, (j + 1 / 3) / m);
        if (i + j < m - 1) push((i + 2 / 3) / m, (j + 2 / 3) / m);
      }
    }
  }
  return { points: Float32Array.from(pts), weights: Float32Array.from(wts), tris: Int32Array.from(ids) };
}

/** Area-weighted centroid and average normal of a set of triangles. */
export function patchFrame(part, triList, clip = null) {
  const { tris, vertices: V, triNormal, triArea } = part;
  const c = [0, 0, 0], n = [0, 0, 0];
  let A = 0;
  for (const t of triList) {
    const w = triArea[t];
    for (let d = 0; d < 3; d++) {
      c[d] += (w * (V[3 * tris[3 * t] + d] + V[3 * tris[3 * t + 1] + d] + V[3 * tris[3 * t + 2] + d])) / 3;
      n[d] += w * triNormal[3 * t + d];
    }
    A += w;
  }
  if (clip) return { center: clip.center.slice(), normal: normalize(n), area: A };
  return { center: c.map((v) => v / (A || 1)), normal: normalize(n), area: A };
}

function normalize(v) {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
}

function featureEdges(part, angleDeg) {
  const { tris, nTri, neighbors, triNormal, faceOf } = part;
  const cosT = Math.cos((angleDeg * Math.PI) / 180);
  const out = [];
  for (let t = 0; t < nTri; t++) {
    for (let e = 0; e < 3; e++) {
      const o = neighbors[3 * t + e];
      if (o >= 0 && o < t) continue; // each shared edge once
      let sharp = o < 0;
      if (!sharp) {
        const dot =
          triNormal[3 * t] * triNormal[3 * o] + triNormal[3 * t + 1] * triNormal[3 * o + 1] + triNormal[3 * t + 2] * triNormal[3 * o + 2];
        sharp = dot < cosT || (part.brepFaces && faceOf[t] !== faceOf[o]);
      }
      if (sharp) out.push(tris[3 * t + e], tris[3 * t + ((e + 1) % 3)]);
    }
  }
  return Uint32Array.from(out);
}

/** Non-indexed display geometry with normals split at creases sharper than `angleDeg`. */
function creasedDisplay(part, angleDeg) {
  const { tris, nTri, nVert, triNormal, triArea, vertices: V } = part;
  const cosT = Math.cos((angleDeg * Math.PI) / 180);
  // vertex -> incident triangles (CSR)
  const start = new Uint32Array(nVert + 1);
  for (let i = 0; i < tris.length; i++) start[tris[i] + 1]++;
  for (let v = 0; v < nVert; v++) start[v + 1] += start[v];
  const fill = start.slice(0, nVert);
  const inc = new Uint32Array(tris.length);
  for (let t = 0; t < nTri; t++) for (let c = 0; c < 3; c++) inc[fill[tris[3 * t + c]]++] = t;

  const position = new Float32Array(9 * nTri);
  const normal = new Float32Array(9 * nTri);
  const srcVert = new Uint32Array(3 * nTri);
  for (let t = 0; t < nTri; t++) {
    const nx = triNormal[3 * t], ny = triNormal[3 * t + 1], nz = triNormal[3 * t + 2];
    for (let c = 0; c < 3; c++) {
      const v = tris[3 * t + c];
      let sx = 0, sy = 0, sz = 0;
      for (let q = start[v]; q < start[v + 1]; q++) {
        const s = inc[q];
        const dot = nx * triNormal[3 * s] + ny * triNormal[3 * s + 1] + nz * triNormal[3 * s + 2];
        if (dot >= cosT) {
          sx += triNormal[3 * s] * triArea[s];
          sy += triNormal[3 * s + 1] * triArea[s];
          sz += triNormal[3 * s + 2] * triArea[s];
        }
      }
      const l = Math.hypot(sx, sy, sz) || 1;
      const o = 9 * t + 3 * c;
      normal[o] = sx / l;
      normal[o + 1] = sy / l;
      normal[o + 2] = sz / l;
      position[o] = V[3 * v];
      position[o + 1] = V[3 * v + 1];
      position[o + 2] = V[3 * v + 2];
      srcVert[3 * t + c] = v;
    }
  }
  return { position, normal, srcVert };
}

/**
 * Scale a part in place by `s` about the centre of its footprint, keeping it on the ground.
 * Triangles keep their order, so face selections, fixtures and loads stay valid.
 * Returns the pivot, so callers can move points (e.g. brush centres) the same way.
 */
export function scalePart(part, s) {
  if (!(s > 0) || !Number.isFinite(s)) throw new Error('Scale factor must be a positive number.');
  const { min, max } = part.bbox;
  const pivot = [(min[0] + max[0]) / 2, min[1], (min[2] + max[2]) / 2];
  const move = (arr) => {
    for (let i = 0; i < arr.length; i += 3) {
      for (let d = 0; d < 3; d++) arr[i + d] = pivot[d] + (arr[i + d] - pivot[d]) * s;
    }
  };
  move(part.vertices);
  move(part.display.position);
  for (let t = 0; t < part.nTri; t++) part.triArea[t] *= s * s;
  part.area *= s * s;
  part.volume *= s * s * s;
  part.bbox = boundingBox(part.vertices);
  // cached ray-cast structures and measurements belong to the old size
  delete part._bvh;
  delete part._wallThickness;
  return pivot;
}
