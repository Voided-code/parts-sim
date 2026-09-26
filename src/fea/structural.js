// Structural study (main thread): voxel model, boundary-condition mapping, worker runs,
// and mapping of nodal results back onto the part's surface vertices.
import { voxelize, gridForBox } from '../core/voxelize.js';
import { sampleTriangles } from '../core/mesh.js';
import { typicalWallThickness, thinWallDensity } from '../core/shell.js';
import { vertexWeights, interpolate } from './mapping.js';

const MIN_FILL = 0.3;
const G = 9.81;

export function validateMaterial(material) {
  for (const key of ['E', 'yield', 'uts', 'density']) {
    if (!Number.isFinite(material[key]) || material[key] <= 0) throw new Error(`${key === 'E' ? "Young's modulus" : key} must be a positive finite number.`);
  }
  if (!Number.isFinite(material.nu) || material.nu <= -1 || material.nu >= 0.5) {
    throw new Error("Poisson's ratio must be greater than −1 and less than 0.5.");
  }
  if (!material.brittle && material.uts < material.yield) throw new Error('Tensile strength must be at least the yield strength for a ductile material.');
}

export class StructuralModel {
  /**
   * @param {import('../core/mesh.js').Part} part
   * @param {number} resolution voxels along the longest side
   */
  constructor(part, resolution) {
    this.part = part;
    this.resolution = resolution;
    const grid = gridForBox(part.bbox.min, part.bbox.max, resolution);
    Object.assign(this, grid);
    const [nx, ny, nz] = this.dims;
    // Sample finely enough to resolve the thinnest walls (a few samples through each), within
    // a memory budget for the sample grid.
    this.wallThickness = typicalWallThickness(part);
    // Partly filled voxels on the outer faces carry most of the bending stress, so their fill
    // fraction needs several samples per axis (2 per axis rounds 0.7 down to 0.5).
    let sub = 5;
    if (Number.isFinite(this.wallThickness)) sub = Math.max(sub, Math.min(6, Math.ceil((3 * this.h) / this.wallThickness)));
    while (sub > 2 && nx * ny * nz * sub ** 3 > 48e6) sub--;
    const frac = voxelize(part.vertices, part.tris, grid, sub);
    // Walls under ~1.5 voxels thick get a connected layer of voxels with their true
    // cross-section (see core/shell.js); elsewhere a voxel counts when it is mostly filled.
    const shell = this.wallThickness < 2.5 * this.h ? thinWallDensity(part, grid, 1.5 * this.h).density : null;
    this.density = new Float32Array(frac.length);
    let count = 0, thin = 0;
    for (let e = 0; e < frac.length; e++) {
      const f = frac[e], w = shell ? shell[e] : 0;
      let rho = 0;
      if (w > 0) {
        rho = f >= 0.9 ? f : Math.max(w, 0.02);
        if (f < 0.9) thin++;
      } else if (f >= MIN_FILL) rho = f;
      if (rho > 0) {
        this.density[e] = Math.min(1, rho);
        count++;
      }
    }
    this.voxelCount = count;
    this.thinVoxels = thin;
    this.NX = nx + 1;
    this.NY = ny + 1;
    this.NZ = nz + 1;
    this.nNodes = this.NX * this.NY * this.NZ;
    this.activeNode = new Uint8Array(this.nNodes);
    const full = new Uint8Array(this.nNodes); // number of solid voxels around each node
    for (let k = 0; k < nz; k++) {
      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          if (!(this.density[i + nx * (j + ny * k)] > 0)) continue;
          for (let c = 0; c < 8; c++) {
            const n = this.node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
            this.activeNode[n] = 1;
            full[n]++;
          }
        }
      }
    }
    this.surfaceNode = new Uint8Array(this.nNodes);
    for (let n = 0; n < this.nNodes; n++) if (this.activeNode[n] && full[n] < 8) this.surfaceNode[n] = 1;
  }

  node(i, j, k) {
    return i + this.NX * (j + this.NY * k);
  }

  nodePosition(n, out = [0, 0, 0]) {
    const i = n % this.NX, j = ((n / this.NX) | 0) % this.NY, k = (n / (this.NX * this.NY)) | 0;
    out[0] = this.origin[0] + i * this.h;
    out[1] = this.origin[1] + j * this.h;
    out[2] = this.origin[2] + k * this.h;
    return out;
  }

  /** Nearest active surface node to a point (model units), or -1. */
  nearestSurfaceNode(x, y, z, maxCells = 2.5) {
    const gx = (x - this.origin[0]) / this.h, gy = (y - this.origin[1]) / this.h, gz = (z - this.origin[2]) / this.h;
    const r = Math.ceil(maxCells);
    const ci = Math.round(gx), cj = Math.round(gy), ck = Math.round(gz);
    let best = -1, bestD = maxCells * maxCells;
    for (let k = Math.max(0, ck - r); k <= Math.min(this.NZ - 1, ck + r); k++) {
      for (let j = Math.max(0, cj - r); j <= Math.min(this.NY - 1, cj + r); j++) {
        for (let i = Math.max(0, ci - r); i <= Math.min(this.NX - 1, ci + r); i++) {
          const n = this.node(i, j, k);
          if (!this.surfaceNode[n]) continue;
          const d = (i - gx) ** 2 + (j - gy) ** 2 + (k - gz) ** 2;
          if (d < bestD) { bestD = d; best = n; }
        }
      }
    }
    return best;
  }

  /** Call fn(node) for every active surface node within `radius` voxels of a point. */
  forSurfaceNodesNear(x, y, z, radius, fn) {
    const gx = (x - this.origin[0]) / this.h, gy = (y - this.origin[1]) / this.h, gz = (z - this.origin[2]) / this.h;
    const r2 = radius * radius;
    for (let k = Math.max(0, Math.ceil(gz - radius)); k <= Math.min(this.NZ - 1, Math.floor(gz + radius)); k++) {
      for (let j = Math.max(0, Math.ceil(gy - radius)); j <= Math.min(this.NY - 1, Math.floor(gy + radius)); j++) {
        for (let i = Math.max(0, Math.ceil(gx - radius)); i <= Math.min(this.NX - 1, Math.floor(gx + radius)); i++) {
          const n = this.node(i, j, k);
          if (this.surfaceNode[n] && (i - gx) ** 2 + (j - gy) ** 2 + (k - gz) ** 2 <= r2) fn(n);
        }
      }
    }
  }

  /** Surface samples of a list of patches ({tris, clip}). */
  samplePatches(patches) {
    const pts = [], wts = [], tris = [];
    const seen = new Set();
    for (const p of patches) {
      const s = sampleTriangles(this.part, p.tris, this.h * 0.4, p.clip);
      // Repeated face clicks and overlapping brush strokes describe a union of
      // surface areas. Counting them twice would double a pressure load.
      for (let i = 0; i < s.weights.length; i++) {
        const x = s.points[3 * i], y = s.points[3 * i + 1], z = s.points[3 * i + 2];
        const key = `${s.tris[i]}:${x},${y},${z}`;
        if (seen.has(key)) continue;
        seen.add(key);
        pts.push(x, y, z); wts.push(s.weights[i]); tris.push(s.tris[i]);
      }
    }
    return { points: Float32Array.from(pts), weights: Float32Array.from(wts), tris: Int32Array.from(tris) };
  }

  /**
   * Build the held-DOF mask and nodal force vector.
   * @param {object} o
   * @param {Array} o.fixtures  [{patches}]
   * @param {Array} o.loads     [{type: 'force'|'pressure'|'wind', patches, magnitude, dir, forces?}]
   * @param {boolean} o.gravity
   * @param {object} o.material
   * @param {number} o.toMeters
   */
  assemble({ fixtures, loads, gravity, material, toMeters }) {
    validateMaterial(material);
    if (!Number.isFinite(toMeters) || toMeters <= 0) throw new Error('Invalid model units.');
    const bc = new Uint8Array(3 * this.nNodes);
    const f = new Float64Array(3 * this.nNodes);
    const warnings = [];
    let fixedNodes = 0;
    const hold = (n) => {
      if (!bc[3 * n]) fixedNodes++;
      bc[3 * n] = bc[3 * n + 1] = bc[3 * n + 2] = 1;
    };
    for (const fx of fixtures) {
      const s = this.samplePatches(fx.patches);
      let hit = 0;
      for (let i = 0; i < s.weights.length; i++) {
        const x = s.points[3 * i], y = s.points[3 * i + 1], z = s.points[3 * i + 2];
        const n = this.nearestSurfaceNode(x, y, z);
        if (n < 0) continue;
        hit++;
        hold(n);
        // The voxel grid can overhang the selected face by part of a voxel; hold every surface
        // node lying on the face (within 3/4 voxel of it), not only the nearest one per sample,
        // or the edge rows of a clamp stay free and the part pivots at its support.
        this.forSurfaceNodesNear(x, y, z, 0.75, hold);
      }
      if (!hit) warnings.push(`${fx.name}: no solid voxels under the selection - try a finer mesh.`);
    }
    const total = [0, 0, 0];
    for (const ld of loads) {
      if (ld.type === 'wind') {
        const F = ld.forces; // per part triangle, N (world)
        if (!F || F.length !== this.part.nTri * 3 || F.some((v) => !Number.isFinite(v))) throw new Error(`${ld.name}: invalid airflow forces.`);
        let lost = 0;
        for (let t = 0; t < this.part.nTri; t++) {
          const fx = F[3 * t], fy = F[3 * t + 1], fz = F[3 * t + 2];
          if (fx === 0 && fy === 0 && fz === 0) continue;
          const c = centroid(this.part, t);
          const n = this.nearestSurfaceNode(c[0], c[1], c[2], 3.5);
          if (n < 0) { lost++; continue; }
          f[3 * n] += fx; f[3 * n + 1] += fy; f[3 * n + 2] += fz;
          total[0] += fx; total[1] += fy; total[2] += fz;
        }
        if (lost) warnings.push(`${ld.name}: ${lost} surface triangles had no nearby voxel.`);
        continue;
      }
      if (!['force', 'pressure'].includes(ld.type)) throw new Error(`Unknown load type: ${ld.type}`);
      if (!Number.isFinite(ld.magnitude) || ld.magnitude < 0) throw new Error(`${ld.name}: magnitude must be a finite, nonnegative number.`);
      let direction;
      if (ld.type === 'force') {
        const length = ld.dir?.length === 3 && ld.dir.every(Number.isFinite) ? Math.hypot(...ld.dir) : 0;
        if (!(length > 0)) throw new Error(`${ld.name}: choose a nonzero force direction.`);
        direction = ld.dir.map((v) => v / length);
      }
      const s = this.samplePatches(ld.patches);
      const nodes = new Map();
      let W = 0;
      for (let i = 0; i < s.weights.length; i++) {
        const n = this.nearestSurfaceNode(s.points[3 * i], s.points[3 * i + 1], s.points[3 * i + 2]);
        if (n < 0) continue;
        const w = s.weights[i];
        const e = nodes.get(n) || [0, 0, 0, 0];
        e[0] += w;
        if (ld.type === 'pressure') {
          // pressure pushes into the surface: -normal * p * area
          const t = s.tris[i];
          for (let d = 0; d < 3; d++) e[1 + d] -= this.part.triNormal[3 * t + d] * w;
        }
        nodes.set(n, e);
        W += w;
      }
      if (!nodes.size) {
        warnings.push(`${ld.name}: no solid voxels under the selection - try a finer mesh.`);
        continue;
      }
      for (const [n, e] of nodes) {
        let v;
        if (ld.type === 'pressure') {
          const a = ld.magnitude * 1e6 * toMeters * toMeters; // MPa * model-unit^2 -> N
          v = [e[1] * a, e[2] * a, e[3] * a];
        } else {
          const s2 = (ld.magnitude * e[0]) / W;
          v = direction.map((d) => d * s2);
        }
        for (let d = 0; d < 3; d++) {
          f[3 * n + d] += v[d];
          total[d] += v[d];
        }
      }
    }
    if (gravity) {
      const [nx, ny] = this.dims;
      const hm = this.h * toMeters;
      let mass = 0;
      for (let e = 0; e < this.density.length; e++) {
        const d = this.density[e];
        if (!(d > 0)) continue;
        const m = material.density * d * hm * hm * hm;
        mass += m;
        const i = e % nx, j = ((e / nx) | 0) % ny, k = (e / (nx * ny)) | 0;
        for (let c = 0; c < 8; c++) f[3 * this.node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)) + 1] -= (m * G) / 8;
      }
      total[1] -= mass * G;
    }
    if (fixedNodes > 0 && fixedNodes < 3) warnings.push('Very small fixture - the part may pivot. Select a larger area.');
    return { bc, f, fixedNodes, total, warnings };
  }

  /** Trilinear interpolation weights from grid nodes to every part vertex, skipping inactive nodes. */
  vertexWeights(activeNode) {
    return vertexWeights(this.part.vertices, this, activeNode);
  }

  /** Interpolate a nodal field with `comps` components onto part vertices (NaN where no data). */
  interpolate(W, field, comps = 1) {
    return interpolate(W, field, comps);
  }

  /** Grid description the worker needs to map its results onto the part's vertices. */
  mapping() {
    return { vertices: Float32Array.from(this.part.vertices), origin: this.origin, hModel: this.h, dims: this.dims };
  }

  /** World-space centers of voxels whose index passes `test(e)`. */
  voxelCenters(test) {
    const [nx, ny] = this.dims;
    const out = [];
    for (let e = 0; e < this.density.length; e++) {
      if (!test(e)) continue;
      const i = e % nx, j = ((e / nx) | 0) % ny, k = (e / (nx * ny)) | 0;
      out.push(this.origin[0] + (i + 0.5) * this.h, this.origin[1] + (j + 0.5) * this.h, this.origin[2] + (k + 0.5) * this.h);
    }
    return Float32Array.from(out);
  }

  /** Voxels on the outside of the model (for the mesh preview). */
  isSurfaceVoxel(e) {
    const [nx, ny, nz] = this.dims;
    if (!(this.density[e] > 0)) return false;
    const i = e % nx, j = ((e / nx) | 0) % ny, k = (e / (nx * ny)) | 0;
    if (i === 0 || j === 0 || k === 0 || i === nx - 1 || j === ny - 1 || k === nz - 1) return true;
    return !(
      this.density[e - 1] > 0 && this.density[e + 1] > 0 &&
      this.density[e - nx] > 0 && this.density[e + nx] > 0 &&
      this.density[e - nx * ny] > 0 && this.density[e + nx * ny] > 0
    );
  }
}

function centroid(part, t) {
  const V = part.vertices, T = part.tris;
  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++) out[d] += V[3 * T[3 * t + c] + d] / 3;
  return out;
}

/** Runs one job on a fresh worker; cancel() terminates it immediately. */
export class FEAJob {
  constructor(message, { onProgress, onStep } = {}) {
    this.worker = new Worker(new URL('./fea.worker.js', import.meta.url), { type: 'module' });
    this.settled = false;
    this.promise = new Promise((resolve, reject) => {
      this.reject = reject;
      const finish = (error, value) => {
        if (this.settled) return;
        this.settled = true;
        this.worker.terminate();
        if (error) reject(error);
        else resolve(value);
      };
      this.worker.onmessage = (ev) => {
        if (this.settled) return;
        const d = ev.data;
        try {
          if (d.type === 'progress') onProgress?.(d);
          else if (d.type === 'breakStep' || d.type === 'step') onStep?.(d);
          else if (d.type === 'result' || d.type === 'breakDone' || d.type === 'done') finish(null, d);
          else if (d.type === 'error') finish(new Error(d.message));
        } catch (err) { finish(err); }
      };
      this.worker.onerror = (e) => finish(new Error(e.message || 'Solver crashed'));
      this.worker.onmessageerror = () => finish(new Error('Could not read the solver result.'));
      try { this.worker.postMessage(message, message.f ? [message.f.buffer] : []); }
      catch (err) { finish(err); }
    });
  }

  cancel() {
    if (this.settled) return;
    this.settled = true;
    this.worker.terminate();
    this.reject(Object.assign(new Error('Cancelled'), { cancelled: true }));
  }
}
