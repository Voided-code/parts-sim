// Airflow study: builds a wind tunnel around the part, runs the LBM solver (WebGPU or CPU
// worker), time-averages the flow, and draws particles / streamlines / a slice plane.
import * as THREE from 'three';
import { voxelize } from '../core/voxelize.js';
import { LBMGPU, webgpuAvailable, gpuMaxCells } from './lbm-gpu.js';
import { RAMP_STEPS } from './lbm-cpu.js';
import { pressureForceCoefficients } from './forces.js';
import { wallLinks } from './links.js';
import { rainbow } from '../viewer/colormap.js';

export const U_LAT = 0.08; // lattice free-stream speed (Mach ~0.14)
const NU_AIR = 1.5e-5; // m^2/s
const MIN_NU_LAT = 0.0005;
// grid size (cells): the slider's range and defaults. The CPU worker runs on one core, so it is capped low.
export const FLOW_CELLS = { min: 50e3, gpuDefault: 1.0e6, cpuDefault: 170e3, cpuMax: 1.0e6, gpuMax: 60e6 };
// memory per cell: GPU (in/out distributions, moments, solid flag, wall links, read-back) and the page
// (snapshot, time averages, solid flag, wall links, voxelizer)
export const FLOW_BYTES = { gpu: 2 * 76 + 16 + 4 + 4 + 19 + 16, page: 16 + 16 + 1 + 19 + 4 };

/** Grid-size range for an engine: {min, max, default} cells. */
export async function flowCapacity(useGPU) {
  if (!useGPU) return { min: FLOW_CELLS.min, max: FLOW_CELLS.cpuMax, default: FLOW_CELLS.cpuDefault };
  let max = FLOW_CELLS.gpuMax;
  try { max = Math.min(max, 0.95 * (await gpuMaxCells())); } catch { /* keep the default cap */ }
  max = Math.max(max, FLOW_CELLS.min);
  return { min: FLOW_CELLS.min, max, default: Math.min(FLOW_CELLS.gpuDefault, max) };
}

/**
 * The wind-tunnel grid for a part and wind direction at about `cells` cells (without meshing it).
 * @returns {{basis: THREE.Vector3[], q: Float32Array, min: number[], max: number[], L: number[], up: number, dims: number[], h: number, origin: number[]}}
 */
export function planTunnel(part, dir, cells) {
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
  const B = Math.max(Math.sqrt(L[1] * L[2]), 0.3 * L[0]);
  if (!L.every(Number.isFinite) || !(B > 0)) throw new Error('The part has no finite volume to mesh.');
  const up = 1.5 * B, down = 2.5 * B, side = 0.9 * B;
  const D = [L[0] + up + down, L[1] + 2 * side, L[2] + 2 * side];
  const budget = Math.max(cells, 4096);
  let h = Math.cbrt((D[0] * D[1] * D[2]) / budget);
  let dims = D.map((d) => Math.max(16, Math.ceil(d / h)));
  // Ceil/minimum-axis rounding must not exceed the memory budget on thin/long parts.
  while (dims[0] * dims[1] * dims[2] > budget) {
    h *= Math.max(1.01, Math.cbrt(dims[0] * dims[1] * dims[2] / budget));
    dims = D.map((d) => Math.max(16, Math.ceil(d / h)));
  }
  const origin = [min[0] - up, min[1] - (dims[1] * h - L[1]) / 2, min[2] - (dims[2] * h - L[2]) / 2];
  return { basis, q, min, max, L, up, dims, h, origin };
}

/** Direction the air travels. yaw 0 = toward -Z (hits the +Z face), 90 = toward +X; pitch > 0 tilts upward. */
export function windDirection(yawDeg, pitchDeg) {
  const y = THREE.MathUtils.degToRad(yawDeg), p = THREE.MathUtils.degToRad(pitchDeg);
  return new THREE.Vector3(Math.sin(y) * Math.cos(p), Math.sin(p), -Math.cos(y) * Math.cos(p)).normalize();
}

export class AirflowStudy {
  constructor(viewer, { onStatus, onUpdate } = {}) {
    this.viewer = viewer;
    this.onStatus = onStatus || (() => {});
    this.onUpdate = onUpdate || (() => {});
    this.group = new THREE.Group();
    this.group.matrixAutoUpdate = false;
    viewer.flowGroup.add(this.group);
    this.running = false;
    this.generation = 0;
    this.runGeneration = 0;
    this.show = { particles: true, streamlines: false, slice: false, domain: true };
    this.sliceAxis = 'xz';
    this.slicePos = 0.5;
    this.sliceQuantity = 'speed';
    this.particleCount = 3000;
    this.unsubFrame = viewer.onFrame((dt) => this.frame(dt));
  }

  get ready() {
    return !!this.sim && !this.initializing;
  }

  ensureCurrent(generation) {
    if (generation !== this.generation) throw new DOMException('Airflow setup was cancelled.', 'AbortError');
  }

  /**
   * @param {import('../core/mesh.js').Part} part
   * @param {{dir: THREE.Vector3, speed: number, airDensity: number, cells: number, engine: string, toMeters: number}} o
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
    this.recoveries = 0;
    this.part = part;
    this.opts = o;
    const useGPU = o.engine !== 'cpu' && (await webgpuAvailable());
    this.ensureCurrent(generation);
    const cap = await flowCapacity(useGPU);
    this.ensureCurrent(generation);
    const wanted = Number.isFinite(o.cells) && o.cells > 0 ? o.cells : cap.default;
    const { basis, q, min, max, L, dims, h, origin } = planTunnel(part, o.dir, Math.min(cap.max, Math.max(cap.min, wanted)));
    const [e1, e2, e3] = basis;
    this.basis = basis;
    this.h = h;
    this.dims = dims;
    this.origin = origin;
    const N = dims[0] * dims[1] * dims[2];
    this.N = N;
    this.onStatus(`Meshing wind tunnel (${dims.join(' x ')} = ${(N / 1e6).toFixed(2)}M cells)...`);
    await new Promise((r) => setTimeout(r, 20));
    this.ensureCurrent(generation);
    const frac = voxelize(q, part.tris, { origin, h, dims }, 2);
    const solid = new Uint8Array(N);
    let solidCount = 0;
    for (let c = 0; c < N; c++) if (frac[c] >= 0.5) { solid[c] = 1; solidCount++; }
    if (!solidCount) throw new Error('The part is too thin for this flow grid. Move the grid-size slider to more cells.');
    this.solid = solid;
    this.onStatus('Measuring wall distances for curved surfaces...');
    await new Promise((r) => setTimeout(r, 20));
    this.ensureCurrent(generation);
    this.links = wallLinks(q, part.tris, { origin, h, dims }, solid).links;

    this.partRange = [
      [(min[1] - origin[1]) / h, (max[1] - origin[1]) / h],
      [(min[2] - origin[2]) / h, (max[2] - origin[2]) / h],
    ];
    const hm = h * o.toMeters;
    const Lc = Math.max(L[1], L[2]) * o.toMeters;
    this.reynolds = (o.speed * Lc) / NU_AIR;
    const LcCells = Math.max(L[1], L[2]) / h;
    this.nuLat = Math.max((U_LAT * LcCells) / this.reynolds, MIN_NU_LAT);
    this.simReynolds = (U_LAT * LcCells) / this.nuLat;
    this.hm = hm;
    this.q = 0.5 * o.airDensity * o.speed * o.speed;

    const M = new THREE.Matrix4().makeBasis(e1, e2, e3)
      .multiply(new THREE.Matrix4().makeTranslation(origin[0], origin[1], origin[2]))
      .multiply(new THREE.Matrix4().makeScale(h, h, h));
    this.group.matrix.copy(M);
    this.group.matrixWorldNeedsUpdate = true;

    this.engine = useGPU ? 'WebGPU' : 'CPU';
    try { await this.createSolver(generation); }
    catch (err) {
      this.ensureCurrent(generation);
      if (!useGPU) throw err;
      this.onStatus(`WebGPU solver failed (${err.message}); rebuilding a smaller CPU grid.`);
      return this.setup(part, { ...o, engine: 'cpu', cells: Math.min(o.cells || FLOW_CELLS.cpuDefault, FLOW_CELLS.cpuDefault) });
    }
    this.ensureCurrent(generation);
    this.macro = null;
    this.avgRho = null;
    this.avgU = null;
    this.samples = 0;
    this.buildDomainBox();
    this.initParticles();
    this.onStatus(`Wind tunnel ready: ${dims.join(' x ')} cells on ${this.engine}.`);
    this.onUpdate();
  }

  async createSolver(generation = this.generation) {
    this.ensureCurrent(generation);
    this.initializing = true;
    const params = { dims: this.dims, solid: this.solid, links: this.links, uLat: U_LAT, nuLat: this.nuLat };
    try {
      if (this.engine === 'WebGPU') {
        const sim = await LBMGPU.create(params);
        if (generation !== this.generation) { sim.destroy(); this.ensureCurrent(generation); }
        this.sim = sim;
      } else {
        const worker = new Worker(new URL('./cfd.worker.js', import.meta.url), { type: 'module' });
        const sim = this.sim = { worker, steps: 0 };
        await new Promise((resolve, reject) => {
          let initialized = false;
          const timer = setTimeout(() => finish(new Error('The CPU flow worker did not start.')), 20000);
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
            } else if (d.type === 'snapshot') this.handleSnapshot(d.macro, d.steps, d.mlups);
          };
          worker.postMessage({ type: 'init', params });
        });
      }
      this.ensureCurrent(generation);
      this.steps = 0;
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
    const runGeneration = ++this.runGeneration;
    if (this.engine === 'WebGPU') {
      this.gpuTask = (this.gpuTask || Promise.resolve()).then(() => this.gpuLoop(runGeneration));
    }
    else this.sim.worker.postMessage({ type: 'run' });
    this.onUpdate();
  }

  pause() {
    this.running = false;
    this.runGeneration++;
    if (this.sim?.worker) this.sim.worker.postMessage({ type: 'pause' });
    this.onUpdate();
  }

  async gpuLoop(runGeneration) {
    let batch = 10;
    let lastRead = 0;
    let readInterval = 220;
    const sim = this.sim;
    try {
    while (this.running && this.sim === sim && this.runGeneration === runGeneration) {
      const t0 = performance.now();
      await sim.step(batch);
      if (this.sim !== sim || this.runGeneration !== runGeneration) return;
      const dt = Math.max(1, performance.now() - t0);
      const mlups = (this.N * batch) / (dt * 1000);
      batch = Math.max(1, Math.min(400, Math.round((batch * 25) / dt)));
      // big grids take a while to read back and average: read less often so the GPU keeps stepping
      // at least 80% of the time
      if (performance.now() - lastRead > readInterval) {
        lastRead = performance.now();
        const macro = await sim.readMacro();
        if (this.sim !== sim || this.runGeneration !== runGeneration) return;
        this.handleSnapshot(macro, sim.steps, mlups);
        readInterval = Math.max(220, 4 * (performance.now() - lastRead));
      }
      await new Promise((r) => requestAnimationFrame(r));
    }
    } catch (err) {
      if (this.sim === sim && this.runGeneration === runGeneration) this.fail(err);
    }
  }

  fail(err) {
    this.running = false;
    this.runGeneration++;
    this.destroySolver();
    this.clearResults();
    this.onStatus(`Airflow stopped: ${err.message}`);
    this.onUpdate();
  }

  clearResults() {
    this.macro = null;
    this.avgRho = null;
    this.avgU = null;
    this.results = null;
    this.samples = 0;
    this.steps = 0;
    this.developing = true;
    this.mlups = 0;
    this.updateSlice();
    this.buildStreamlines();
  }

  handleSnapshot(macro, steps, mlups) {
    if (!this.running && this.engine !== 'WebGPU') return;
    let bad = false;
    if (!(macro instanceof Float32Array) || macro.length !== 4 * this.N) return this.fail(new Error('Invalid flow snapshot.'));
    for (let c = 0; c < this.N; c++) {
      const u = Math.abs(macro[4 * c + 1]) + Math.abs(macro[4 * c + 2]) + Math.abs(macro[4 * c + 3]);
      const rho = macro[4 * c];
      if (!(u < 0.6) || !Number.isFinite(rho) || rho <= 0) { bad = true; break; }
    }
    if (bad) return this.recover();
    this.macro = macro;
    this.steps = steps;
    this.mlups = mlups;
    const N = this.N;
    const flowThrough = this.dims[0] / U_LAT;
    this.developing = steps < RAMP_STEPS + 0.6 * flowThrough;
    if (!this.avgRho || this.developing) {
      this.avgRho = new Float32Array(N);
      this.avgU = new Float32Array(3 * N);
      for (let c = 0; c < N; c++) {
        this.avgRho[c] = macro[4 * c];
        this.avgU[3 * c] = macro[4 * c + 1];
        this.avgU[3 * c + 1] = macro[4 * c + 2];
        this.avgU[3 * c + 2] = macro[4 * c + 3];
      }
      this.samples = 0;
    } else {
      const a = 1 / (this.samples + 1);
      for (let c = 0; c < N; c++) {
        this.avgRho[c] += a * (macro[4 * c] - this.avgRho[c]);
        this.avgU[3 * c] += a * (macro[4 * c + 1] - this.avgU[3 * c]);
        this.avgU[3 * c + 1] += a * (macro[4 * c + 2] - this.avgU[3 * c + 1]);
        this.avgU[3 * c + 2] += a * (macro[4 * c + 3] - this.avgU[3 * c + 2]);
      }
      this.samples++;
    }
    this.computeForces();
    if (this.show.slice) this.updateSlice();
    if (this.show.streamlines && performance.now() - (this.lastStream || 0) > 2500) this.buildStreamlines();
    this.onUpdate();
  }

  async recover() {
    if (this.recovering) return;
    if ((this.recoveries || 0) >= 3) return this.fail(new Error('The flow is still unstable after three retries. Reduce the wind speed or use more cells.'));
    this.recovering = true;
    this.recoveries = (this.recoveries || 0) + 1;
    const wasRunning = this.running;
    this.pause();
    const generation = ++this.generation;
    this.nuLat *= 2;
    this.simReynolds /= 2;
    this.onStatus('Flow became unstable - doubled the viscosity and restarted.');
    this.destroySolver();
    this.clearResults();
    try {
      await this.createSolver(generation);
      if (wasRunning) this.start();
    } catch (err) {
      if (generation === this.generation) this.fail(err);
    } finally {
      if (generation === this.generation) this.recovering = false;
      this.onUpdate();
    }
  }

  async reset() {
    if (!this.ready) return;
    const wasRunning = this.running;
    this.pause();
    const generation = ++this.generation;
    this.destroySolver();
    this.clearResults();
    this.onUpdate();
    try {
      await this.createSolver(generation);
      if (wasRunning) this.start();
    } catch (err) {
      if (generation === this.generation) this.fail(err);
    } finally { this.onUpdate(); }
  }

  computeForces() {
    const { C, frontal } = pressureForceCoefficients(this.avgRho, this.solid, this.dims, U_LAT);
    const k = this.q * this.hm * this.hm;
    const [e1, e2, e3] = this.basis;
    const world = new THREE.Vector3().addScaledVector(e1, C[0] * k).addScaledVector(e2, C[1] * k).addScaledVector(e3, C[2] * k);
    this.results = {
      drag: C[0] * k,
      lift: C[1] * k,
      side: C[2] * k,
      cd: C[0] / frontal,
      cl: C[1] / frontal,
      frontalArea: frontal * this.hm * this.hm,
      force: world,
    };
  }

  // ---------- sampling ----------

  toLattice(p, out = new THREE.Vector3()) {
    const [e1, e2, e3] = this.basis;
    return out.set((p.dot(e1) - this.origin[0]) / this.h, (p.dot(e2) - this.origin[1]) / this.h, (p.dot(e3) - this.origin[2]) / this.h);
  }

  /** Trilinear sample of fluid cells at lattice point (x,y,z). Returns weight of fluid cells used. */
  sample(field, comps, x, y, z, out) {
    const [nx, ny, nz] = this.dims;
    const gx = x - 0.5, gy = y - 0.5, gz = z - 0.5;
    const i0 = Math.floor(gx), j0 = Math.floor(gy), k0 = Math.floor(gz);
    const tx = gx - i0, ty = gy - j0, tz = gz - k0;
    for (let c = 0; c < comps; c++) out[c] = 0;
    let wsum = 0;
    for (let q = 0; q < 8; q++) {
      const i = i0 + (q & 1), j = j0 + ((q >> 1) & 1), k = k0 + ((q >> 2) & 1);
      if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) continue;
      const cell = i + nx * (j + ny * k);
      if (this.solid[cell]) continue;
      const w = (q & 1 ? tx : 1 - tx) * ((q >> 1) & 1 ? ty : 1 - ty) * ((q >> 2) & 1 ? tz : 1 - tz);
      for (let c = 0; c < comps; c++) out[c] += w * field[comps * cell + c];
      wsum += w;
    }
    if (wsum > 0) for (let c = 0; c < comps; c++) out[c] /= wsum;
    return wsum;
  }

  isSolidAt(x, y, z) {
    const [nx, ny, nz] = this.dims;
    const i = Math.floor(x), j = Math.floor(y), k = Math.floor(z);
    if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return true;
    return this.solid[i + nx * (j + ny * k)] === 1;
  }

  /** Time-averaged pressure coefficient at every part vertex (NaN if unavailable). */
  surfaceCp() {
    const part = this.part;
    const out = new Float32Array(part.nVert).fill(NaN);
    if (!this.avgRho) return out;
    const p = new THREE.Vector3(), n = new THREE.Vector3(), l = new THREE.Vector3();
    const [e1, e2, e3] = this.basis;
    const tmp = [0];
    const inv = 1 / (3 * 0.5 * U_LAT * U_LAT);
    for (let v = 0; v < part.nVert; v++) {
      p.fromArray(part.vertices, 3 * v);
      n.fromArray(part.vertNormal, 3 * v);
      this.toLattice(p, l);
      const nl = [n.dot(e1), n.dot(e2), n.dot(e3)];
      for (const off of [0.9, 1.6, 2.6]) {
        const w = this.sample(this.avgRho, 1, l.x + nl[0] * off, l.y + nl[1] * off, l.z + nl[2] * off, tmp);
        if (w > 0.15) {
          out[v] = (tmp[0] - 1) * inv;
          break;
        }
      }
    }
    return out;
  }

  /** Aerodynamic pressure force on each part triangle (N, world) for the structural study. */
  triangleForces() {
    const part = this.part;
    const cp = this.surfaceCp();
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
    if (!this.particles || !this.show.particles || !this.macro) {
      if (this.particles) this.particles.visible = false;
      return;
    }
    this.particles.visible = true;
    const P = this.particleCount, T = this.trail;
    const k = (this.dims[0] / (4.5 * U_LAT)) * dt; // free stream crosses the tunnel in ~4.5 s
    const u = [0, 0, 0];
    const c = [0, 0, 0];
    this.head = (this.head + 1) % T;
    const head = this.head;
    const pos = this.particles.geometry.attributes.position.array;
    const col = this.particles.geometry.attributes.color.array;
    const [nx] = this.dims;
    for (let i = 0; i < P; i++) {
      let x = this.pos[3 * i], y = this.pos[3 * i + 1], z = this.pos[3 * i + 2];
      const w = this.sample(this.macro, 4, x, y, z, this.tmp4 || (this.tmp4 = [0, 0, 0, 0]));
      const m = this.tmp4;
      u[0] = m[1]; u[1] = m[2]; u[2] = m[3];
      if (w < 0.05) { u[0] = U_LAT * 0.3; u[1] = 0; u[2] = 0; }
      // midpoint step
      const mx = x + 0.5 * k * u[0], my = y + 0.5 * k * u[1], mz = z + 0.5 * k * u[2];
      if (this.sample(this.macro, 4, mx, my, mz, m) > 0.05) { u[0] = m[1]; u[1] = m[2]; u[2] = m[3]; }
      x += k * u[0]; y += k * u[1]; z += k * u[2];
      this.age[i] += dt;
      if (x >= nx - 1.5 || this.isSolidAt(x, y, z) || this.age[i] > 20 || y < 1 || z < 1 || y > this.dims[1] - 1 || z > this.dims[2] - 1) {
        this.spawn(i);
        x = this.pos[3 * i]; y = this.pos[3 * i + 1]; z = this.pos[3 * i + 2];
      }
      this.pos[3 * i] = x; this.pos[3 * i + 1] = y; this.pos[3 * i + 2] = z;
      const o = 3 * (i * T + head);
      this.hist[o] = x; this.hist[o + 1] = y; this.hist[o + 2] = z;
      this.speed[i] = Math.hypot(u[0], u[1], u[2]);
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
    if (!this.avgU || !this.show.streamlines) return;
    const [nx, ny, nz] = this.dims;
    const [ry, rz] = this.partRange;
    const S = 16;
    const pos = [], col = [];
    const u = [0, 0, 0], c = [0, 0, 0];
    const padY = (ry[1] - ry[0]) * 0.3 + 1, padZ = (rz[1] - rz[0]) * 0.3 + 1;
    for (let a = 0; a < S; a++) {
      for (let b = 0; b < S; b++) {
        let x = 1.5;
        let y = Math.min(ny - 2, Math.max(1.5, ry[0] - padY + ((a + 0.5) / S) * (ry[1] - ry[0] + 2 * padY)));
        let z = Math.min(nz - 2, Math.max(1.5, rz[0] - padZ + ((b + 0.5) / S) * (rz[1] - rz[0] + 2 * padZ)));
        for (let s = 0; s < 6 * nx; s++) {
          if (this.sample(this.avgU, 3, x, y, z, u) < 0.05) break;
          let sp = Math.hypot(u[0], u[1], u[2]);
          if (sp < 1e-5) break;
          const st = 0.6 / sp;
          const mx = x + 0.5 * st * u[0], my = y + 0.5 * st * u[1], mz = z + 0.5 * st * u[2];
          if (this.sample(this.avgU, 3, mx, my, mz, u) < 0.05) break;
          sp = Math.hypot(u[0], u[1], u[2]) || 1e-5;
          const st2 = 0.6 / sp;
          const X = x + st2 * u[0], Y = y + st2 * u[1], Z = z + st2 * u[2];
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
    if (!this.show.slice || !this.avgU) return;
    const [nx, ny, nz] = this.dims;
    const axis = this.sliceAxis;
    // in-plane axes (a, b) and the fixed coordinate
    const cfg = {
      xy: { A: nx, B: ny, fixed: 2, n: nz, cell: (a, b, f) => a + nx * (b + ny * f) },
      xz: { A: nx, B: nz, fixed: 1, n: ny, cell: (a, b, f) => a + nx * (f + ny * b) },
      yz: { A: ny, B: nz, fixed: 0, n: nx, cell: (a, b, f) => f + nx * (a + ny * b) },
    }[axis];
    const f = Math.min(cfg.n - 1, Math.max(0, Math.floor(this.slicePos * cfg.n)));
    const data = new Uint8Array(cfg.A * cfg.B * 4);
    const c = [0, 0, 0];
    const inv = 1 / (3 * 0.5 * U_LAT * U_LAT);
    for (let b = 0; b < cfg.B; b++) {
      for (let a = 0; a < cfg.A; a++) {
        const cell = cfg.cell(a, b, f);
        const o = 4 * (a + cfg.A * b);
        if (this.solid[cell]) { data.set([45, 50, 60, 255], o); continue; }
        let t;
        if (this.sliceQuantity === 'pressure') t = ((this.avgRho[cell] - 1) * inv + 1.2) / 2.2;
        else t = Math.hypot(this.avgU[3 * cell], this.avgU[3 * cell + 1], this.avgU[3 * cell + 2]) / (1.6 * U_LAT);
        rainbow(t, c);
        data.set([c[0], c[1], c[2], 225], o);
      }
    }
    const tex = new THREE.DataTexture(data, cfg.A, cfg.B, THREE.RGBAFormat);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.magFilter = THREE.LinearFilter;
    tex.needsUpdate = true;
    const fc = f + 0.5;
    const corners = {
      xy: [[0, 0, fc], [nx, 0, fc], [nx, ny, fc], [0, ny, fc]],
      xz: [[0, fc, 0], [nx, fc, 0], [nx, fc, nz], [0, fc, nz]],
      yz: [[fc, 0, 0], [fc, ny, 0], [fc, ny, nz], [fc, 0, nz]],
    }[axis];
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
    if (key === 'particles' && this.particles) this.particles.visible = on && !!this.macro;
    if (key === 'domain') { const d = this.group.getObjectByName('domain'); if (d) d.visible = on; }
    if (key === 'streamlines') this.buildStreamlines();
    if (key === 'slice') this.updateSlice();
  }

  setGroupVisible(on) {
    this.group.visible = on;
  }

  destroySolver() {
    if (!this.sim) return;
    this.sim.cancelInit?.();
    if (this.sim.worker) this.sim.worker.terminate();
    else this.sim.destroy();
    this.sim = null;
  }

  dispose(full = true) {
    this.running = false;
    this.generation++;
    this.runGeneration++;
    this.initializing = false;
    this.recovering = false;
    this.destroySolver();
    for (const o of [...this.group.children]) {
      o.geometry?.dispose();
      o.material?.map?.dispose();
      o.material?.dispose();
      this.group.remove(o);
    }
    this.particles = null;
    this.macro = null;
    this.avgRho = null;
    this.avgU = null;
    this.results = null;
    this.samples = 0;
    this.steps = 0;
    this.mlups = 0;
    if (full) {
      this.unsubFrame();
      this.viewer.flowGroup.remove(this.group);
    }
  }
}
