// Airflow study: builds a wind tunnel around the part and runs the v1 flow engine on it (flow.js:
// WebGPU, or the CPU in a worker), keeps the forces' statistics, and draws particles, streamlines,
// a slice plane and the surface pressure from the engine's reduced view fields.
import * as THREE from 'three';
import { voxelize } from '../core/voxelize.js';
import { webgpuAvailable } from './lbm-gpu.js';
import { hardwareAdapter } from '../core/webgpu.js';
import { buildFlowGrid, RAMP_STEPS, SOLID, FACE } from './flow.js';
import { wallRayCaster } from './links.js';
import { batchMeans } from './stats.js';
import { AIR, airNu } from './validation.js';
import { rainbow } from '../viewer/colormap.js';

export const U_LAT = 0.08; // lattice free-stream speed (Mach ~0.14)
// the v0.6 solver's lowest lattice viscosity (the v1 engine reaches real air: V1_NU_FLOOR)
export const MIN_NU_LAT = 0.0005;
export const V1_NU_FLOOR = 1e-6;
// above this Reynolds number (on the part's length along the wind) the boundary layer is taken as
// turbulent and the wall model is on ('auto')
export const TURBULENT_RE = 5e5;
// the tunnel's margins in multiples of the part's blockage size (planTunnel)
export const DEFAULT_MARGINS = { up: 2.25, down: 3.75, side: 1.35 };
// grid size (cells): the slider's range and defaults
export const FLOW_CELLS = { min: 50e3, gpuDefault: 4e6, cpuDefault: 250e3, cpuMax: 2e6, gpuMax: 300e6 };
// memory per cell: GPU (16-bit populations in place, cell kind; wall records are a few % more) and
// the page (cell kinds, the voxelizer's part box, the view fields)
export const FLOW_BYTES = { gpu: 19 * 2 + 1 + 2, page: 1 + 2 };

/** Grid-size range for an engine: {min, max, default} cells. */
export async function flowCapacity(useGPU) {
  if (!useGPU) return { min: FLOW_CELLS.min, max: FLOW_CELLS.cpuMax, default: FLOW_CELLS.cpuDefault };
  let max = FLOW_CELLS.gpuMax;
  try {
    const adapter = await hardwareAdapter({ powerPreference: 'high-performance' });
    // the populations are split over up to six storage buffers of at most one binding each
    const bind = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
    max = Math.min(max, (0.95 * bind * Math.min(6, adapter.limits.maxStorageBuffersPerShaderStage - 2)) / (19 * 2));
  } catch { /* keep the default cap */ }
  max = Math.max(max, FLOW_CELLS.min);
  return { min: FLOW_CELLS.min, max, default: Math.min(FLOW_CELLS.gpuDefault, max) };
}

/**
 * The wind-tunnel grid for a part and wind direction at about `cells` cells (without meshing it).
 * Options: h, a fixed cell size instead of the cell budget; ground, the gap (model units) between the
 * part's underside and a ground plane on the tunnel floor (lattice y = up); periodicSpan with
 * spanCells, a 2D section whose span (lattice z) wraps around.
 * @returns {{basis: THREE.Vector3[], q: Float32Array, min: number[], max: number[], L: number[], up: number, dims: number[], h: number, origin: number[]}}
 */
export function planTunnel(part, dir, cells, opts = {}) {
  const e1 = dir.clone().normalize();
  const ref = Math.abs(e1.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const e2 = ref.clone().addScaledVector(e1, -ref.dot(e1)).normalize();
  const e3 = new THREE.Vector3().crossVectors(e1, e2);
  const basis = [e1, e2, e3];
  // part in the wind frame
  const V = part.vertices;
  const q = new Float32Array(V.length);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < V.length; i += 3) {
    for (let a = 0; a < 3; a++) {
      const b = basis[a];
      const v = V[i] * b.x + V[i + 1] * b.y + V[i + 2] * b.z;
      q[i + a] = v;
      if (v < min[a]) min[a] = v;
      if (v > max[a]) max[a] = v;
    }
  }
  const L = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
  // Tunnel margins scale with the blockage size (square root of the cross-section) and the
  // length, not the largest dimension: a long-span wing needs room around its chord, and
  // sizing by span would leave its thickness only a cell or two across.
  const span = !!opts.periodicSpan;
  // a periodic section's blockage comes from its thickness only
  const B = span ? Math.max(L[1], 0.3 * L[0]) : Math.max(Math.sqrt(L[1] * L[2]), 0.3 * L[0]);
  if (!L.every(Number.isFinite) || !(B > 0)) throw new Error('The part has no finite volume to mesh.');
  // margins in multiples of B, or of the part's length along the wind for slender parts (wings)
  // the app's margins ('app'), or v0.6's (1.5, 2.5, 0.9) with any of them changed
  const m = opts.margins === 'app' ? { ...DEFAULT_MARGINS } : { up: 1.5, down: 2.5, side: 0.9, ...opts.margins };
  const scale = m.ofLength ? L[0] : B;
  const up = m.up * scale, down = m.down * scale, side = m.side * scale;
  const ground = opts.ground ?? null;
  // with a ground plane the floor sits `ground` below the part, plus one layer of cells under it
  const D = [L[0] + up + down, ground !== null ? ground + L[1] + side : L[1] + 2 * side, L[2] + 2 * side];
  const spanCells = span ? Math.max(2, Math.round(opts.spanCells || 8)) : 0;
  const dimsFor = (hv) => [
    Math.max(16, Math.ceil(D[0] / hv)),
    ground !== null ? Math.max(16, Math.ceil(D[1] / hv) + 1) : Math.max(16, Math.ceil(D[1] / hv)),
    span ? spanCells : Math.max(16, Math.ceil(D[2] / hv)),
  ];
  let h, dims;
  if (opts.h > 0) {
    h = opts.h;
    dims = dimsFor(h);
  } else {
    const budget = Math.max(cells, 4096);
    h = span ? Math.sqrt((D[0] * D[1]) / (budget / spanCells)) : Math.cbrt((D[0] * D[1] * D[2]) / budget);
    dims = dimsFor(h);
    // Ceil/minimum-axis rounding must not exceed the memory budget on thin/long parts.
    while (dims[0] * dims[1] * dims[2] > budget) {
      h *= Math.max(1.01, Math.cbrt(dims[0] * dims[1] * dims[2] / budget));
      dims = dimsFor(h);
    }
  }
  const origin = [
    min[0] - up,
    // the ground plane lies on the face between cell layers 0 (solid floor) and 1
    ground !== null ? min[1] - ground - h : min[1] - (dims[1] * h - L[1]) / 2,
    span ? (min[2] + max[2]) / 2 - (dims[2] * h) / 2 : min[2] - (dims[2] * h - L[2]) / 2,
  ];
  return { basis, q, min, max, L, up, dims, h, origin, ground, periodicSpan: span };
}

/**
 * Area of the part's shadow along the wind (model units^2): its triangles projected on the plane across
 * the wind, filled into a bitmap of up to 1024 pixels along the part's larger side (thin plates too).
 */
export function frontalArea(plan, tris) {
  const { q, min, L } = plan;
  if (!(L[1] > 0) || !(L[2] > 0)) return 0;
  const px = Math.max(L[1], L[2]) / 1024;
  const W = Math.max(1, Math.ceil(L[1] / px)), H = Math.max(1, Math.ceil(L[2] / px));
  const hit = new Uint8Array(W * H);
  const y = [0, 0, 0], z = [0, 0, 0];
  for (let t = 0; t < tris.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      y[k] = (q[3 * tris[t + k] + 1] - min[1]) / px;
      z[k] = (q[3 * tris[t + k] + 2] - min[2]) / px;
    }
    const area2 = (y[1] - y[0]) * (z[2] - z[0]) - (y[2] - y[0]) * (z[1] - z[0]);
    if (Math.abs(area2) < 1e-12) continue;
    const s = area2 > 0 ? 1 : -1;
    const i0 = Math.max(0, Math.floor(Math.min(...y))), i1 = Math.min(W - 1, Math.ceil(Math.max(...y)));
    const j0 = Math.max(0, Math.floor(Math.min(...z))), j1 = Math.min(H - 1, Math.ceil(Math.max(...z)));
    for (let j = j0; j <= j1; j++) {
      for (let i = i0; i <= i1; i++) {
        const cy = i + 0.5, cz = j + 0.5;
        let inside = true;
        for (let e = 0; e < 3 && inside; e++) {
          const a = e, b = (e + 1) % 3;
          inside = s * ((y[b] - y[a]) * (cz - z[a]) - (cy - y[a]) * (z[b] - z[a])) >= 0;
        }
        if (inside) hit[i + W * j] = 1;
      }
    }
  }
  let n = 0;
  for (const v of hit) n += v;
  return n * px * px;
}

/** Direction the air travels. yaw 0 = toward -Z (hits the +Z face), 90 = toward +X; pitch > 0 tilts upward. */
export function windDirection(yawDeg, pitchDeg) {
  const y = THREE.MathUtils.degToRad(yawDeg), p = THREE.MathUtils.degToRad(pitchDeg);
  return new THREE.Vector3(Math.sin(y) * Math.cos(p), Math.sin(p), -Math.cos(y) * Math.cos(p)).normalize();
}


// the engine's surface pressure and view fields are refreshed this often while it runs (ms)
const FIELDS_EVERY = 300, SURFACE_EVERY = 1000;
// steps between samples of the view fields on the GPU
const SAMPLE_EVERY = 20;

export class AirflowStudy {
  constructor(viewer, { onStatus, onUpdate } = {}) {
    this.viewer = viewer;
    this.onStatus = onStatus || (() => {});
    this.onUpdate = onUpdate || (() => {});
    this.group = new THREE.Group();
    this.group.matrixAutoUpdate = false;
    viewer.flowGroup.add(this.group);
    this.running = false;
    this.frozen = false; // results loaded from a file (importResults), no solver
    this.frozenCp = null;
    this.generation = 0;
    this.runGeneration = 0;
    this.show = { particles: true, streamlines: false, slice: false, domain: true };
    this.sliceAxis = 'xz';
    this.slicePos = 0.5;
    this.sliceQuantity = 'speed';
    this.particleCount = 3000;
    this.autoStop = true;
    this.unsubFrame = viewer.onFrame((dt) => this.frame(dt));
  }

  get ready() {
    return !!this.sim && !this.initializing;
  }

  /** True when there is a surface pressure to colour the part with (live, or loaded from a file). */
  get hasSurface() {
    return !!this.surface || !!this.frozenCp;
  }

  ensureCurrent(generation) {
    if (generation !== this.generation) throw new DOMException('Airflow setup was cancelled.', 'AbortError');
  }

  /**
   * @param {import('../core/mesh.js').Part} part
   * @param {{dir: THREE.Vector3, speed: number, airDensity: number, cells: number, engine: string, toMeters: number,
   *   ground?: number|null, boundaryLayer?: 'auto'|'turbulent'|'laminar'}} o  ground: clearance under the part
   *   to a moving road (model units), or null for free air
   */
  async setup(part, o) {
    for (const [key, label] of [['speed', 'Wind speed'], ['airDensity', 'Air density'], ['toMeters', 'Unit scale']]) {
      if (!Number.isFinite(o[key]) || o[key] <= 0) throw new Error(`${label} must be a finite positive number.`);
    }
    if (!o.dir || ![o.dir.x, o.dir.y, o.dir.z].every(Number.isFinite) || o.dir.lengthSq() === 0) {
      throw new Error('Wind direction must be a finite nonzero vector.');
    }
    this.dispose(false);
    const generation = this.generation;
    this.part = part;
    this.opts = o;
    const useGPU = o.engine !== 'cpu' && (await webgpuAvailable());
    this.ensureCurrent(generation);
    const cap = await flowCapacity(useGPU);
    this.ensureCurrent(generation);
    const wanted = Number.isFinite(o.cells) && o.cells > 0 ? o.cells : cap.default;
    const ground = Number.isFinite(o.ground) && o.ground >= 0 ? o.ground : null;
    const plan = planTunnel(part, o.dir, Math.min(cap.max, Math.max(cap.min, wanted)), { margins: 'app', ground });
    const { basis, min, max, L, dims, h, origin } = plan;
    Object.assign(this, { plan, basis, h, dims, origin });
    const N = (this.N = dims[0] * dims[1] * dims[2]);
    this.onStatus(`Meshing wind tunnel (${dims.join(' x ')} = ${(N / 1e6).toFixed(2)}M cells)...`);
    await new Promise((r) => setTimeout(r, 20));
    this.ensureCurrent(generation);
    const grid = buildFlowGrid(plan, part.tris, { voxelize, wallRayCaster: wallRayCaster(plan.q, part.tris) });
    if (!grid.solidCount && !grid.rec.count) throw new Error('The part is too thin for this flow grid. Move the grid-size slider to more cells.');
    this.grid = grid;
    this.ensureCurrent(generation);

    this.partRange = [
      [(min[1] - origin[1]) / h, (max[1] - origin[1]) / h],
      [(min[2] - origin[2]) / h, (max[2] - origin[2]) / h],
    ];
    const hm = (this.hm = h * o.toMeters);
    // the air: viscosity from the ISA atmosphere at the chosen density
    const nu = airNu(o.airDensity);
    this.nuAir = nu;
    // Reynolds number on the part's size across the wind (as shown) and on its length along it
    // (which decides whether the boundary layer is turbulent)
    this.reynolds = (o.speed * Math.max(L[1], L[2]) * o.toMeters) / nu;
    this.reynoldsLength = (o.speed * L[0] * o.toMeters) / nu;
    const nuReal = (nu * U_LAT) / (o.speed * hm);
    this.nuLat = Math.max(nuReal, V1_NU_FLOOR);
    this.simReynolds = (this.reynolds * nuReal) / this.nuLat;
    const bl = o.boundaryLayer || 'auto';
    this.wallModel = bl === 'turbulent' || (bl === 'auto' && this.reynoldsLength >= TURBULENT_RE);
    this.q = 0.5 * o.airDensity * o.speed * o.speed;
    this.frontal = frontalArea(plan, part.tris) / (h * h);
    this.groundGap = ground !== null ? ground / h : -1;

    const M = new THREE.Matrix4().makeBasis(basis[0], basis[1], basis[2])
      .multiply(new THREE.Matrix4().makeTranslation(origin[0], origin[1], origin[2]))
      .multiply(new THREE.Matrix4().makeScale(h, h, h));
    this.group.matrix.copy(M);
    this.group.matrixWorldNeedsUpdate = true;
    this.vertexRecords = this.mapVertices(part, grid);

    this.engine = useGPU ? 'WebGPU' : 'CPU';
    try { await this.createSolver(generation); }
    catch (err) {
      this.ensureCurrent(generation);
      if (!useGPU) throw err;
      this.onStatus(`WebGPU solver failed (${err.message}); rebuilding a smaller CPU grid.`);
      return this.setup(part, { ...o, engine: 'cpu', cells: Math.min(o.cells || FLOW_CELLS.cpuDefault, FLOW_CELLS.cpuDefault) });
    }
    this.ensureCurrent(generation);
    this.clearResults();
    this.buildDomainBox();
    this.initParticles();
    this.onStatus(`Wind tunnel ready: ${dims.join(' x ')} cells on ${this.engine}${this.wallModel ? ', turbulent boundary layer (wall model)' : ''}.`);
    this.onUpdate();
  }


  /** For each part vertex, the wall record next to it (for the surface pressure), or -1. */
  mapVertices(part, grid) {
    const { origin, h, dims } = grid;
    const [nx, ny] = dims;
    const byCell = new Map();
    for (let r = 0; r < grid.rec.count; r++) byCell.set(grid.rec.cell[r], r);
    const out = new Int32Array(part.nVert).fill(-1);
    const [e1, e2, e3] = this.basis;
    const p = new THREE.Vector3(), n = new THREE.Vector3();
    for (let v = 0; v < part.nVert; v++) {
      p.fromArray(part.vertices, 3 * v);
      n.fromArray(part.vertNormal, 3 * v);
      // a point just outside the surface, in lattice units
      const l = [(p.dot(e1) - origin[0]) / h + 0.7 * n.dot(e1), (p.dot(e2) - origin[1]) / h + 0.7 * n.dot(e2), (p.dot(e3) - origin[2]) / h + 0.7 * n.dot(e3)];
      let best = -1, bd = Infinity;
      const cx = Math.floor(l[0]), cy = Math.floor(l[1]), cz = Math.floor(l[2]);
      for (let dz = -1; dz <= 1; dz++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dx = -1; dx <= 1; dx++) {
            const r = byCell.get(cx + dx + nx * (cy + dy + ny * (cz + dz)));
            if (r === undefined) continue;
            const d = (cx + dx + 0.5 - l[0]) ** 2 + (cy + dy + 0.5 - l[1]) ** 2 + (cz + dz + 0.5 - l[2]) ** 2;
            if (d < bd) { bd = d; best = r; }
          }
        }
      }
      out[v] = best;
    }
    return out;
  }

  async createSolver(generation = this.generation) {
    this.ensureCurrent(generation);
    this.initializing = true;
    const params = { uLat: U_LAT, nuLat: this.nuLat, collision: 'rr', wallModel: this.wallModel };
    try {
      if (this.engine === 'WebGPU') {
        const { FlowGPU } = await import('./flow-gpu.js');
        const sim = await FlowGPU.create(this.grid, params);
        if (generation !== this.generation) { sim.destroy(); this.ensureCurrent(generation); }
        this.sim = sim;
      } else {
        const worker = new Worker(new URL('./flow.worker.js', import.meta.url), { type: 'module' });
        const sim = this.sim = { worker, steps: 0 };
        // coarse view cells: at most about a million
        const factor = Math.max(1, Math.ceil(Math.cbrt(this.N / 1e6)));
        await new Promise((resolve, reject) => {
          let initialized = false;
          const timer = setTimeout(() => finish(new Error('The CPU flow worker did not start.')), 60000);
          const finish = (err) => {
            clearTimeout(timer);
            sim.cancelInit = null;
            if (err) reject(err);
            else { initialized = true; resolve(); }
          };
          sim.cancelInit = () => finish(new DOMException('Airflow setup was cancelled.', 'AbortError'));
          worker.onerror = (ev) => {
            const err = new Error(ev.message || 'The CPU flow worker failed.');
            if (!initialized) finish(err);
            else if (this.sim === sim) this.fail(err);
          };
          worker.onmessage = (ev) => {
            if (this.sim !== sim || generation !== this.generation) return;
            const d = ev.data;
            if (d.type === 'ready') finish();
            else if (d.type === 'error') {
              const err = new Error(d.message);
              if (!initialized) finish(err);
              else this.fail(err);
            } else if (d.type === 'forces') {
              sim.steps = d.steps;
              this.mlups = d.mlups;
              this.onForces(d.f, d.steps);
            } else if (d.type === 'fields') {
              this.onFields({ dims: d.dims, factor: d.factor, inst: d.inst, avg: d.avg, samples: d.samples });
            } else if (d.type === 'surface') this.surface = d.rho;
          };
          worker.postMessage({ type: 'init', grid: this.grid, params, factor });
        });
      }
      this.ensureCurrent(generation);
    } catch (err) {
      if (generation === this.generation) this.destroySolver();
      throw err;
    } finally {
      if (generation === this.generation) this.initializing = false;
    }
  }

  start() {
    if (!this.ready || this.running) return;
    this.running = true;
    this.converged = false;
    const runGeneration = ++this.runGeneration;
    if (this.engine === 'WebGPU') this.gpuTask = (this.gpuTask || Promise.resolve()).then(() => this.gpuLoop(runGeneration));
    else {
      this.sim.worker.postMessage({ type: 'run' });
      clearInterval(this.surfaceTimer);
      this.surfaceTimer = setInterval(() => this.sim?.worker?.postMessage({ type: 'surface' }), SURFACE_EVERY);
    }
    this.onUpdate();
  }

  pause() {
    this.running = false;
    this.runGeneration++;
    clearInterval(this.surfaceTimer);
    if (this.sim?.worker) this.sim.worker.postMessage({ type: 'pause' });
    this.onUpdate();
  }

  /**
   * The GPU loop: two batches in flight, so the GPU always has the next one queued while the page
   * reads the last one's forces (a few bytes). Batches aim at ~30 ms of GPU work, so the desktop and
   * this window keep drawing. The view fields and surface pressure are read back now and then.
   */
  async gpuLoop(runGeneration) {
    const sim = this.sim;
    const live = () => this.running && this.sim === sim && this.runGeneration === runGeneration;
    const { firstSubmitSteps, SUBMIT_MS } = await import('./flow-gpu.js');
    let batch = firstSubmitSteps(this.N);
    const inFlight = [];
    let lastFields = 0, lastSurface = 0, lastDone = performance.now();
    try {
      while (live()) {
        inFlight.push({ done: sim.submitBatch(batch, { sampleEvery: SAMPLE_EVERY }), n: batch });
        if (inFlight.length < 2) continue;
        const { done, n } = inFlight.shift();
        const f = await done;
        if (!live()) break;
        const now = performance.now(), dt = Math.max(1, now - lastDone);
        lastDone = now;
        this.mlups = (this.N * n) / (dt * 1000);
        // the next batches: about SUBMIT_MS each (dt is the time per completed batch, with two queued;
        // next to nothing when the GPU finished it while the page was busy). They at most double at a
        // time: one that ran for seconds would have the OS reset the GPU.
        batch = Math.max(Math.max(1, n >> 1), Math.min(256, 2 * n, Math.round((n * SUBMIT_MS) / dt)));
        this.onForces(f, sim.steps - inFlight.reduce((a, b) => a + b.n, 0));
        if (now - lastFields > FIELDS_EVERY && !this.readingFields) {
          lastFields = now;
          this.readingFields = true;
          sim.readFields().then((fl) => { if (this.sim === sim) this.onFields(fl); }).catch(() => {}).finally(() => { this.readingFields = false; });
        }
        if (now - lastSurface > SURFACE_EVERY && !this.readingSurface) {
          lastSurface = now;
          this.readingSurface = true;
          sim.surfaceRho().then((rho) => { if (this.sim === sim) this.surface = rho; }).catch(() => {}).finally(() => { this.readingSurface = false; });
        }
      }
      // let the batches still in flight finish (they are part of the state)
      for (const b of inFlight) await b.done.catch(() => {});
    } catch (err) {
      if (this.sim === sim && this.runGeneration === runGeneration) this.fail(err);
    }
  }

  /** A batch of forces (lattice units, summed over f.steps steps) ending at step `steps`. */
  onForces(f, steps) {
    this.steps = steps;
    const k = 1 / (0.5 * U_LAT * U_LAT * f.steps); // mean force in the units of the pressure coefficient's area (cells^2)
    const C = f.me.map((v) => v * k);
    if (!C.every(Number.isFinite)) {
      return this.fail(new Error('the flow became unstable (the forces are no longer finite). Use more cells, a lower wind speed, or the laminar boundary layer setting.'));
    }
    const flowThrough = this.dims[0] / U_LAT;
    const developing = steps < RAMP_STEPS + 1.5 * flowThrough;
    if (developing) {
      this.developing = true;
    } else {
      if (this.developing) {
        // the flow has developed: start the averages
        this.developing = false;
        this.series = { x: [], y: [], z: [], fx: [] };
        if (this.engine === 'WebGPU') this.sim.resetAverages();
        else this.sim.worker.postMessage({ type: 'resetAverages' });
        this.averageFrom = steps;
      }
      const s = this.series;
      s.x.push(C[0]); s.y.push(C[1]); s.z.push(C[2]);
      s.fx.push(f.friction[0] * k);
      this.samples = s.x.length;
    }
    this.computeForces(C);
    this.onUpdate();
  }

  onFields(fields) {
    this.fields = fields;
    if (this.show.slice) this.updateSlice();
    if (this.show.streamlines && performance.now() - (this.lastStream || 0) > 2500) this.buildStreamlines();
    this.onUpdate();
  }

  fail(err) {
    this.running = false;
    this.runGeneration++;
    clearInterval(this.surfaceTimer);
    this.destroySolver();
    this.clearResults();
    this.onStatus(`Airflow stopped: ${err.message}`);
    this.onUpdate();
  }

  clearResults() {
    this.fields = null;
    this.surface = null;
    this.results = null;
    this.series = null;
    this.samples = 0;
    this.steps = 0;
    this.developing = true;
    this.converged = false;
    this.mlups = 0;
    this.updateSlice();
    this.buildStreamlines();
  }

  async reset() {
    if (!this.ready) return;
    const wasRunning = this.running;
    this.pause();
    if (this.engine === 'WebGPU') {
      await this.gpuTask;
      this.sim.reset();
    } else this.sim.worker.postMessage({ type: 'reset' });
    this.clearResults();
    this.onUpdate();
    if (wasRunning) this.start();
  }

  /**
   * Forces in newtons from a sample (C: mean force in pressure-coefficient units, cells^2), and once
   * averaging, the mean over the samples with its 95% confidence interval (flow statistics: stats.js).
   */
  computeForces(C) {
    const k = this.q * this.hm * this.hm;
    const [e1, e2, e3] = this.basis;
    const s = this.series;
    let mean = C, ci = [NaN, NaN, NaN], friction = NaN;
    if (s && s.x.length >= 8) {
      const b = [batchMeans(s.x), batchMeans(s.y), batchMeans(s.z)];
      mean = b.map((v) => v.mean);
      ci = b.map((v) => v.ci);
      friction = batchMeans(s.fx).mean * k;
    }
    const world = new THREE.Vector3().addScaledVector(e1, mean[0] * k).addScaledVector(e2, mean[1] * k).addScaledVector(e3, mean[2] * k);
    const frontal = Math.max(1, this.frontal);
    this.results = {
      drag: mean[0] * k, lift: mean[1] * k, side: mean[2] * k,
      dragCI: ci[0] * k, liftCI: ci[1] * k, sideCI: ci[2] * k,
      frictionDrag: friction,
      cd: mean[0] / frontal, cl: mean[1] / frontal,
      cdCI: ci[0] / frontal, clCI: ci[1] / frontal,
      frontalArea: frontal * this.hm * this.hm,
      force: world,
      averaged: !!(s && s.x.length >= 8),
    };
    // settled: the drag and lift within about 1% (or 0.005 and 0.01 in coefficient) at 95% confidence
    if (this.results.averaged && this.steps - this.averageFrom > this.dims[0] / U_LAT) {
      const settled = ci[0] / frontal <= Math.max(0.01 * Math.abs(mean[0] / frontal), 0.005) && ci[1] / frontal <= Math.max(0.02 * Math.abs(mean[1] / frontal), 0.01);
      if (settled && !this.converged) {
        this.converged = true;
        if (this.autoStop && this.running) {
          this.onStatus('Airflow converged: the forces have settled (95% confidence interval within about 1%).');
          this.pause();
        }
      }
    }
  }

  // ---------- sampling ----------

  toLattice(p, out = new THREE.Vector3()) {
    const [e1, e2, e3] = this.basis;
    return out.set((p.dot(e1) - this.origin[0]) / this.h, (p.dot(e2) - this.origin[1]) / this.h, (p.dot(e3) - this.origin[2]) / this.h);
  }

  /**
   * Trilinear sample of a view field (fields.inst or .avg: [rho - 1, ux, uy, uz] per coarse cell) at
   * lattice point (x, y, z), over coarse cells with fluid. Returns the weight of fluid cells used.
   */
  sample(field, x, y, z, out) {
    const fl = this.fields;
    const [nx, ny, nz] = fl.dims, f = fl.factor;
    const gx = x / f - 0.5, gy = y / f - 0.5, gz = z / f - 0.5;
    const i0 = Math.floor(gx), j0 = Math.floor(gy), k0 = Math.floor(gz);
    const tx = gx - i0, ty = gy - j0, tz = gz - k0;
    out[0] = out[1] = out[2] = out[3] = 0;
    let wsum = 0;
    for (let q = 0; q < 8; q++) {
      const i = i0 + (q & 1), j = j0 + ((q >> 1) & 1), k = k0 + ((q >> 2) & 1);
      if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) continue;
      const cell = i + nx * (j + ny * k);
      if (field[4 * cell] === -2) continue;
      const w = (q & 1 ? tx : 1 - tx) * ((q >> 1) & 1 ? ty : 1 - ty) * ((q >> 2) & 1 ? tz : 1 - tz);
      for (let c = 0; c < 4; c++) out[c] += w * field[4 * cell + c];
      wsum += w;
    }
    if (wsum > 0) for (let c = 0; c < 4; c++) out[c] /= wsum;
    return wsum;
  }

  isSolidAt(x, y, z) {
    const [nx, ny, nz] = this.dims;
    const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z);
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return true;
    return this.grid.kind[i + nx * (j + ny * k)] === SOLID;
  }

  /** Time-averaged pressure coefficient at every part vertex (NaN where unavailable). */
  surfaceCp() {
    const part = this.part;
    if (this.frozenCp) return Float32Array.from(this.frozenCp);
    const out = new Float32Array(part.nVert).fill(NaN);
    if (!this.surface || !this.vertexRecords) return out;
    const inv = 1 / (3 * 0.5 * U_LAT * U_LAT);
    for (let v = 0; v < part.nVert; v++) {
      const r = this.vertexRecords[v];
      if (r >= 0 && r < this.surface.length) out[v] = this.surface[r] * inv;
    }
    return out;
  }

  /** Aerodynamic pressure force on each part triangle (N, world) for the structural study. */
  triangleForces(cp = this.surfaceCp()) {
    const part = this.part;
    const F = new Float32Array(3 * part.nTri);
    const a2 = this.opts.toMeters * this.opts.toMeters;
    for (let t = 0; t < part.nTri; t++) {
      let s = 0, c = 0;
      for (let k = 0; k < 3; k++) {
        const v = cp[part.tris[3 * t + k]];
        if (!Number.isNaN(v)) { s += v; c++; }
      }
      if (!c) continue;
      const pa = (s / c) * this.q;
      const A = part.triArea[t] * a2;
      for (let d = 0; d < 3; d++) F[3 * t + d] = -pa * A * part.triNormal[3 * t + d];
    }
    return F;
  }

  // ---------- saving and loading results (.psim, RAIR section) ----------

  /**
   * The result in the form the file stores it: {meta, arrays}, or null when there is nothing to save yet.
   * arrays: [{name, data, enc: 'q16', dims?}] for encodeArrays (src/core/psim.js). Only the time-averaged
   * coarse flow is stored; the slice, the streamlines and (on load) the particles are drawn from it.
   */
  exportResults() {
    const r = this.results, fl = this.fields;
    if (!r || !fl || !fl.avg || !this.part) return null;
    const o = this.opts, d = o.dir;
    const num = (v) => (Number.isFinite(v) ? v : NaN);
    const meta = {
      schema: 1,
      results: {
        drag: r.drag, lift: r.lift, side: r.side, dragCI: r.dragCI, liftCI: r.liftCI, sideCI: r.sideCI,
        frictionDrag: r.frictionDrag, cd: r.cd, cl: r.cl, cdCI: r.cdCI, clCI: r.clCI,
        frontalArea: r.frontalArea, force: [r.force.x, r.force.y, r.force.z], averaged: !!r.averaged,
      },
      reynolds: this.reynolds, reynoldsLength: this.reynoldsLength, simReynolds: this.simReynolds,
      nuAir: this.nuAir, nuLat: this.nuLat, uLat: U_LAT,
      wallModel: !!this.wallModel, groundGap: this.groundGap, q: this.q, frontal: this.frontal,
      steps: this.steps, samples: this.samples, converged: !!this.converged, developing: !!this.developing,
      mlups: num(this.mlups) || 0, engine: this.engine,
      dims: [...this.dims], h: this.h, factor: fl.factor, fieldDims: [...fl.dims], fieldSamples: fl.samples ?? 0,
      toMeters: o.toMeters, nVert: this.part.nVert,
      settings: {
        dir: [d.x, d.y, d.z],
        yaw: (Math.atan2(d.x, -d.z) * 180) / Math.PI, pitch: (Math.asin(Math.max(-1, Math.min(1, d.y))) * 180) / Math.PI,
        speed: o.speed, airDensity: o.airDensity, ground: Number.isFinite(o.ground) && o.ground >= 0 ? o.ground : null,
        boundaryLayer: o.boundaryLayer || 'auto', cells: o.cells ?? null, engine: o.engine ?? null,
      },
    };
    const n = fl.dims[0] * fl.dims[1] * fl.dims[2];
    const avg = fl.avg;
    const soa = [0, 1, 2, 3].map(() => new Float32Array(n));
    for (let c = 0; c < n; c++) {
      if (avg[4 * c] === -2) { for (let k = 0; k < 4; k++) soa[k][c] = NaN; continue; }
      for (let k = 0; k < 4; k++) soa[k][c] = avg[4 * c + k];
    }
    const dims = [...fl.dims];
    const arrays = [{ name: 'airflow.cp', data: this.surfaceCp(), enc: 'q16' }];
    ['rho', 'ux', 'uy', 'uz'].forEach((k, i) => arrays.push({ name: `airflow.avg.${k}`, data: soa[i], enc: 'q16', dims }));
    return { meta, arrays };
  }

  /**
   * Shows a stored result without a solver: a "frozen" study (frozen = true). `arrays` is the Map of
   * decodeArrays (name -> {data}), an object of name -> typed array, or a list of {name, data}. The tunnel
   * plan is planned again from the part with the stored cell size (deterministic), and the solid cells for the
   * particles and streamlines come from buildFlowGrid's voxelisation (the same code as a live run, without
   * the wall rays and without the solver). Particles move in the stored time-averaged field.
   * Call showLoaded() on the panel afterwards. Run airflow sets the study up fresh and clears the flag.
   */
  async importResults(meta, arrays, part = this.part) {
    const bad = (m) => { throw new Error(`The stored airflow result is not usable: ${m}.`); };
    if (!part) bad('no part to show it on');
    if (!meta || typeof meta !== 'object' || !meta.results || !meta.settings) bad('the header is incomplete');
    const pos = (v) => Number.isFinite(v) && v > 0;
    const st = meta.settings;
    const dir = new THREE.Vector3(...(Array.isArray(st.dir) ? st.dir : []));
    if (![dir.x, dir.y, dir.z].every(Number.isFinite) || dir.lengthSq() === 0) bad('the wind direction');
    if (!pos(st.speed) || !pos(st.airDensity) || !pos(meta.toMeters) || !pos(meta.h)) bad('speed, density or scale');
    const ok3 = (a, big) => Array.isArray(a) && a.length === 3 && a.every((v) => Number.isInteger(v) && v >= 1 && v <= big);
    if (!ok3(meta.dims, 1e5) || !ok3(meta.fieldDims, 1e5)) bad('the grid sizes');
    const N = meta.dims[0] * meta.dims[1] * meta.dims[2];
    if (N > FLOW_CELLS.gpuMax) bad('the grid is too large');
    if (!Number.isInteger(meta.factor) || meta.factor < 1) bad('the coarse factor');
    if (!meta.dims.every((v, i) => meta.fieldDims[i] === Math.ceil(v / meta.factor))) bad('the coarse grid does not fit the tunnel');
    if (meta.nVert !== part.nVert) bad('it was computed for a different part');
    const f = meta.results;
    if (!Array.isArray(f.force) || f.force.length !== 3) bad('the force vector');
    const get = (name) => {
      let a = arrays instanceof Map ? arrays.get(name) : Array.isArray(arrays) ? arrays.find((x) => x.name === name) : arrays?.[name];
      if (a && !ArrayBuffer.isView(a)) a = a.data;
      if (!a || !ArrayBuffer.isView(a)) bad(`the array ${name} is missing`);
      return a;
    };
    const cp = get('airflow.cp');
    if (cp.length !== part.nVert) bad('the pressure array does not match the part');
    const nc = meta.fieldDims[0] * meta.fieldDims[1] * meta.fieldDims[2];
    const comp = ['rho', 'ux', 'uy', 'uz'].map((k) => {
      const a = get(`airflow.avg.${k}`);
      if (a.length !== nc) bad(`the array airflow.avg.${k} has the wrong length`);
      return a;
    });

    this.dispose(false);
    const generation = this.generation;
    const ground = Number.isFinite(st.ground) && st.ground >= 0 ? st.ground : null;
    // the plan again, with the stored cell size: the same dims and origin as the run had
    const plan = planTunnel(part, dir, 0, { margins: 'app', ground, h: meta.h });
    if (!plan.dims.every((v, i) => v === meta.dims[i])) bad('the tunnel no longer matches the part');
    const { basis, min, max, dims, h, origin } = plan;
    this.onStatus('Preparing the loaded airflow result...');
    await new Promise((r) => setTimeout(r, 0));
    this.ensureCurrent(generation);
    const full = buildFlowGrid(plan, part.tris, { voxelize });
    this.ensureCurrent(generation);

    this.part = part;
    this.opts = { dir, speed: st.speed, airDensity: st.airDensity, cells: st.cells ?? undefined, engine: st.engine ?? undefined, toMeters: meta.toMeters, ground, boundaryLayer: st.boundaryLayer || 'auto' };
    Object.assign(this, { plan, basis, h, dims, origin });
    this.N = N;
    this.grid = { dims, N, kind: full.kind }; // the cell kinds are all the redraws need
    this.partRange = [
      [(min[1] - origin[1]) / h, (max[1] - origin[1]) / h],
      [(min[2] - origin[2]) / h, (max[2] - origin[2]) / h],
    ];
    this.hm = h * meta.toMeters;
    Object.assign(this, {
      nuAir: meta.nuAir, nuLat: meta.nuLat, reynolds: meta.reynolds, reynoldsLength: meta.reynoldsLength, simReynolds: meta.simReynolds,
      wallModel: !!meta.wallModel, q: meta.q, frontal: meta.frontal, groundGap: meta.groundGap,
      engine: meta.engine, steps: meta.steps, samples: meta.samples, converged: !!meta.converged, developing: !!meta.developing,
      mlups: meta.mlups || 0, series: null,
    });
    this.results = {
      drag: f.drag, lift: f.lift, side: f.side, dragCI: f.dragCI, liftCI: f.liftCI, sideCI: f.sideCI,
      frictionDrag: f.frictionDrag, cd: f.cd, cl: f.cl, cdCI: f.cdCI, clCI: f.clCI, frontalArea: f.frontalArea,
      force: new THREE.Vector3(...f.force), averaged: !!f.averaged,
    };
    const avg = new Float32Array(4 * nc);
    for (let c = 0; c < nc; c++) {
      if (Number.isNaN(comp[0][c])) { avg[4 * c] = -2; continue; }
      for (let k = 0; k < 4; k++) avg[4 * c + k] = comp[k][c];
    }
    // the particles move in the averaged field too: no instantaneous field is stored
    this.fields = { dims: [...meta.fieldDims], factor: meta.factor, inst: avg, avg, samples: meta.fieldSamples ?? 0 };
    this.frozenCp = Float32Array.from(cp);
    this.surface = null;
    this.frozen = true;
    const M = new THREE.Matrix4().makeBasis(basis[0], basis[1], basis[2])
      .multiply(new THREE.Matrix4().makeTranslation(origin[0], origin[1], origin[2]))
      .multiply(new THREE.Matrix4().makeScale(h, h, h));
    this.group.matrix.copy(M);
    this.group.matrixWorldNeedsUpdate = true;
    this.buildDomainBox();
    this.initParticles();
    this.onStatus('Loaded airflow result shown (not computed here).');
    this.onUpdate();
  }

  // ---------- visuals ----------

  buildDomainBox() {
    const old = this.group.getObjectByName('domain');
    if (old) { old.geometry.dispose(); old.material.dispose(); this.group.remove(old); }
    const [nx, ny, nz] = this.dims;
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(nx, ny, nz).translate(nx / 2, ny / 2, nz / 2)),
      new THREE.LineBasicMaterial({ color: 0x5b8def, transparent: true, opacity: 0.35 }),
    );
    box.name = 'domain';
    box.visible = this.show.domain;
    this.group.add(box);
  }

  spawn(i) {
    const [ry, rz] = this.partRange;
    const [, ny, nz] = this.dims;
    const padY = (ry[1] - ry[0]) * 0.35 + 2, padZ = (rz[1] - rz[0]) * 0.35 + 2;
    const y = Math.min(ny - 1.5, Math.max(1.5, ry[0] - padY + Math.random() * (ry[1] - ry[0] + 2 * padY)));
    const z = Math.min(nz - 1.5, Math.max(1.5, rz[0] - padZ + Math.random() * (rz[1] - rz[0] + 2 * padZ)));
    const x = 1 + Math.random() * 2;
    const T = this.trail;
    for (let s = 0; s < T; s++) {
      const o = 3 * (i * T + s);
      this.hist[o] = x; this.hist[o + 1] = y; this.hist[o + 2] = z;
    }
    this.age[i] = 0;
    this.pos[3 * i] = x; this.pos[3 * i + 1] = y; this.pos[3 * i + 2] = z;
  }

  initParticles() {
    const old = this.group.getObjectByName('particles');
    if (old) { old.geometry.dispose(); old.material.dispose(); this.group.remove(old); }
    const P = this.particleCount, T = (this.trail = 12);
    this.pos = new Float32Array(3 * P);
    this.hist = new Float32Array(3 * P * T);
    this.age = new Float32Array(P);
    this.speed = new Float32Array(P);
    this.head = 0;
    for (let i = 0; i < P; i++) {
      this.spawn(i);
      this.pos[3 * i] = 1 + Math.random() * (this.dims[0] - 3); // pre-fill the tunnel
      for (let s = 0; s < T; s++) this.hist.set(this.pos.subarray(3 * i, 3 * i + 3), 3 * (i * T + s));
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(P * (T - 1) * 6), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(P * (T - 1) * 8), 4));
    const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, depthWrite: false }));
    lines.name = 'particles';
    lines.frustumCulled = false;
    lines.visible = this.show.particles;
    this.particles = lines;
    this.group.add(lines);
  }

  frame(dt) {
    if (!this.particles || !this.show.particles || !this.fields) {
      if (this.particles) this.particles.visible = false;
      return;
    }
    this.particles.visible = true;
    const field = this.fields.inst;
    const P = this.particleCount, T = this.trail;
    const k = (this.dims[0] / (4.5 * U_LAT)) * dt; // free stream crosses the tunnel in ~4.5 s
    const m = this.tmp4 || (this.tmp4 = [0, 0, 0, 0]);
    const c = [0, 0, 0];
    this.head = (this.head + 1) % T;
    const head = this.head;
    const pos = this.particles.geometry.attributes.position.array;
    const col = this.particles.geometry.attributes.color.array;
    const [nx] = this.dims;
    for (let i = 0; i < P; i++) {
      let x = this.pos[3 * i], y = this.pos[3 * i + 1], z = this.pos[3 * i + 2];
      let ux, uy, uz;
      if (this.sample(field, x, y, z, m) < 0.05) { ux = U_LAT * 0.3; uy = 0; uz = 0; }
      else { ux = m[1]; uy = m[2]; uz = m[3]; }
      // midpoint step
      const mx = x + 0.5 * k * ux, my = y + 0.5 * k * uy, mz = z + 0.5 * k * uz;
      if (this.sample(field, mx, my, mz, m) > 0.05) { ux = m[1]; uy = m[2]; uz = m[3]; }
      x += k * ux; y += k * uy; z += k * uz;
      this.age[i] += dt;
      if (x >= nx - 1.5 || this.isSolidAt(x, y, z) || this.age[i] > 20 || y < 1 || z < 1 || y > this.dims[1] - 1 || z > this.dims[2] - 1) {
        this.spawn(i);
        x = this.pos[3 * i]; y = this.pos[3 * i + 1]; z = this.pos[3 * i + 2];
      }
      this.pos[3 * i] = x; this.pos[3 * i + 1] = y; this.pos[3 * i + 2] = z;
      const o = 3 * (i * T + head);
      this.hist[o] = x; this.hist[o + 1] = y; this.hist[o + 2] = z;
      this.speed[i] = Math.hypot(ux, uy, uz);
      rainbow(this.speed[i] / (1.6 * U_LAT), c);
      for (let s = 0; s < T - 1; s++) {
        const a = 3 * (i * T + ((head - s + T) % T));
        const b = 3 * (i * T + ((head - s - 1 + T) % T));
        const seg = i * (T - 1) + s;
        pos[6 * seg] = this.hist[a]; pos[6 * seg + 1] = this.hist[a + 1]; pos[6 * seg + 2] = this.hist[a + 2];
        pos[6 * seg + 3] = this.hist[b]; pos[6 * seg + 4] = this.hist[b + 1]; pos[6 * seg + 5] = this.hist[b + 2];
        const al = 1 - s / (T - 1), al2 = 1 - (s + 1) / (T - 1);
        const r = c[0] / 255, gg = c[1] / 255, bb = c[2] / 255;
        col.set([r, gg, bb, al * 0.95, r, gg, bb, al2 * 0.95], 8 * seg);
      }
    }
    this.particles.geometry.attributes.position.needsUpdate = true;
    this.particles.geometry.attributes.color.needsUpdate = true;
  }

  buildStreamlines() {
    this.lastStream = performance.now();
    const old = this.group.getObjectByName('streamlines');
    if (old) { old.geometry.dispose(); old.material.dispose(); this.group.remove(old); }
    if (!this.fields || !this.show.streamlines) return;
    const field = this.fields.avg;
    const [nx, ny, nz] = this.dims;
    const [ry, rz] = this.partRange;
    const S = 16;
    const pos = [], col = [];
    const u = [0, 0, 0, 0], c = [0, 0, 0];
    const padY = (ry[1] - ry[0]) * 0.3 + 1, padZ = (rz[1] - rz[0]) * 0.3 + 1;
    for (let a = 0; a < S; a++) {
      for (let b = 0; b < S; b++) {
        let x = 1.5;
        let y = Math.min(ny - 2, Math.max(1.5, ry[0] - padY + ((a + 0.5) / S) * (ry[1] - ry[0] + 2 * padY)));
        let z = Math.min(nz - 2, Math.max(1.5, rz[0] - padZ + ((b + 0.5) / S) * (rz[1] - rz[0] + 2 * padZ)));
        for (let s = 0; s < 6 * nx; s++) {
          if (this.sample(field, x, y, z, u) < 0.05) break;
          let sp = Math.hypot(u[1], u[2], u[3]);
          if (sp < 1e-5) break;
          const st = 0.6 / sp;
          const mx = x + 0.5 * st * u[1], my = y + 0.5 * st * u[2], mz = z + 0.5 * st * u[3];
          if (this.sample(field, mx, my, mz, u) < 0.05) break;
          sp = Math.hypot(u[1], u[2], u[3]) || 1e-5;
          const st2 = 0.6 / sp;
          const X = x + st2 * u[1], Y = y + st2 * u[2], Z = z + st2 * u[3];
          if (X >= nx - 1 || this.isSolidAt(X, Y, Z) || Y < 1 || Z < 1 || Y > ny - 1 || Z > nz - 1) break;
          rainbow(sp / (1.6 * U_LAT), c);
          pos.push(x, y, z, X, Y, Z);
          for (let r = 0; r < 2; r++) col.push(c[0] / 255, c[1] / 255, c[2] / 255);
          x = X; y = Y; z = Z;
        }
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(pos), 3));
    g.setAttribute('color', new THREE.BufferAttribute(Float32Array.from(col), 3));
    const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true }));
    lines.name = 'streamlines';
    this.group.add(lines);
  }

  updateSlice() {
    const old = this.group.getObjectByName('slice');
    if (old) { old.geometry.dispose(); old.material.map?.dispose(); old.material.dispose(); this.group.remove(old); }
    if (!this.show.slice || !this.fields) return;
    const field = this.fields.avg;
    const [cnx, cny, cnz] = this.fields.dims, f = this.fields.factor;
    const [nx, ny, nz] = this.dims;
    const axis = this.sliceAxis;
    // in-plane axes (a, b) and the fixed coordinate, on the coarse view grid
    const cfg = {
      xy: { A: cnx, B: cny, n: cnz, cell: (a, b, k) => a + cnx * (b + cny * k) },
      xz: { A: cnx, B: cnz, n: cny, cell: (a, b, k) => a + cnx * (k + cny * b) },
      yz: { A: cny, B: cnz, n: cnx, cell: (a, b, k) => k + cnx * (a + cny * b) },
    }[axis];
    const k = Math.min(cfg.n - 1, Math.max(0, Math.floor(this.slicePos * cfg.n)));
    const data = new Uint8Array(cfg.A * cfg.B * 4);
    const c = [0, 0, 0];
    const inv = 1 / (3 * 0.5 * U_LAT * U_LAT);
    for (let b = 0; b < cfg.B; b++) {
      for (let a = 0; a < cfg.A; a++) {
        const cell = cfg.cell(a, b, k);
        const o = 4 * (a + cfg.A * b);
        if (field[4 * cell] === -2) { data.set([45, 50, 60, 255], o); continue; }
        let t;
        if (this.sliceQuantity === 'pressure') t = (field[4 * cell] * inv + 1.2) / 2.2;
        else t = Math.hypot(field[4 * cell + 1], field[4 * cell + 2], field[4 * cell + 3]) / (1.6 * U_LAT);
        rainbow(t, c);
        data.set([c[0], c[1], c[2], 225], o);
      }
    }
    const tex = new THREE.DataTexture(data, cfg.A, cfg.B, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    const fc = (k + 0.5) * f;
    const X = cnx * f, Y = cny * f, Z = cnz * f;
    const corners = {
      xy: [[0, 0, fc], [X, 0, fc], [X, Y, fc], [0, Y, fc]],
      xz: [[0, fc, 0], [X, fc, 0], [X, fc, Z], [0, fc, Z]],
      yz: [[fc, 0, 0], [fc, Y, 0], [fc, Y, Z], [fc, 0, Z]],
    }[axis];
    void nx; void ny; void nz;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(Float32Array.from(corners.flat()), 3));
    g.setAttribute('uv', new THREE.BufferAttribute(Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1), 2));
    g.setIndex([0, 1, 2, 0, 2, 3]);
    const mesh = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true, depthWrite: false }));
    mesh.name = 'slice';
    mesh.renderOrder = 2;
    this.group.add(mesh);
  }

  setVisible(key, on) {
    this.show[key] = on;
    if (key === 'particles' && this.particles) this.particles.visible = on && !!this.fields;
    if (key === 'domain') { const d = this.group.getObjectByName('domain'); if (d) d.visible = on; }
    if (key === 'streamlines') this.buildStreamlines();
    if (key === 'slice') this.updateSlice();
  }

  setGroupVisible(on) {
    this.group.visible = on;
  }

  destroySolver() {
    if (!this.sim) return;
    clearInterval(this.surfaceTimer);
    this.sim.cancelInit?.();
    if (this.sim.worker) this.sim.worker.terminate();
    else {
      const sim = this.sim;
      // after any batches in flight
      (this.gpuTask || Promise.resolve()).then(() => sim.destroy());
    }
    this.sim = null;
  }

  dispose(full = true) {
    this.running = false;
    this.generation++;
    this.runGeneration++;
    this.initializing = false;
    this.destroySolver();
    for (const o of [...this.group.children]) {
      o.geometry?.dispose();
      o.material?.map?.dispose();
      o.material?.dispose();
      this.group.remove(o);
    }
    this.particles = null;
    this.fields = null;
    this.surface = null;
    this.results = null;
    this.series = null;
    this.samples = 0;
    this.steps = 0;
    this.mlups = 0;
    this.frozen = false;
    this.frozenCp = null;
    if (full) {
      this.unsubFrame();
      this.viewer.flowGroup.remove(this.group);
    }
  }
}
