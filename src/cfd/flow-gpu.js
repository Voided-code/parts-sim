// GPU solver of the v1 flow engine (WebGPU): the shader is native/shaders/flow.wgsl, shared with the
// native app, after a prelude written here for the population buffers.
import FLOW_WGSL from '../../native/shaders/flow.wgsl?raw';
import FLOW_GEN_WGSL from '../../native/shaders/flow_gen.wgsl?raw';
import { hardwareAdapter } from '../core/webgpu.js';
import { W, FACE, flowParams, inletVelocity } from './flow.js';
import { faceTable } from './flow-cpu.js';

const PARAM_BYTES = 64;
const MAX_STEPS = 256; // steps in one submission
// Each submission holds about this much GPU work (ms), so the desktop keeps drawing, and Windows
// (which resets a GPU after 2 s in one submission) and macOS (which ends long command buffers)
// never step in. The first one is sized for a slow GPU: 2e7 cell updates.
export const SUBMIT_MS = 40;
export const firstSubmitSteps = (cells) => Math.max(1, Math.min(MAX_STEPS, Math.round(2e7 / cells)));
const REC = 59, R_Q = 3, R_N = 8, R_AUX = 10, R_RHO = 32, R_BB = 36, R_SAMP = 54, R_Y2 = 55, R_AREA = 56; // wall record layout (flow.wgsl)
const REDUCE_GROUPS = 64;
const RING = 4; // force-sum slots read back while later batches run
const MAX_COARSE = 1.0e6; // cells of the reduced view grid
const f16bits = (() => {
  const f = new Float32Array(1), u = new Uint32Array(f.buffer);
  return (v) => {
    // round to nearest 16-bit float (normal range)
    f[0] = v;
    const x = u[0], sign = (x >>> 16) & 0x8000, e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff;
    if (e <= 0) return sign;
    if (e >= 31) return sign | 0x7c00;
    const r = (e << 10) | (m >>> 13);
    return sign | (r + ((m >>> 12) & 1));
  };
})();

/**
 * The population buffers: slot d (direction d's populations of every cell) belongs to one buffer, and
 * the two slots of a direction pair (i, i + 1) always to the same one. As many slots per buffer as
 * one binding holds.
 */
export function populationBuffers(N, half, maxBytes) {
  const slotBytes = N * (half ? 2 : 4);
  const units = [[0], ...Array.from({ length: 9 }, (_, p) => [2 * p + 1, 2 * p + 2])];
  const bufs = [];
  for (const u of units) {
    const last = bufs[bufs.length - 1];
    if (last && (last.count + u.length) * slotBytes <= maxBytes) last.count += u.length;
    else {
      if (u.length * slotBytes > maxBytes) throw new Error('The flow grid is too large for this GPU.');
      bufs.push({ base: u[0], count: u.length });
    }
  }
  return bufs.map((b) => ({ ...b, bytes: Math.ceil((b.count * slotBytes) / 4) * 4 }));
}

/** WGSL for the population buffers and their accessors ld<P>(s, k) / st<P>(s, k, v) (flow.wgsl). */
export function preludeWGSL(half, bufs) {
  const T = half ? 'f16' : 'f32';
  const lines = half ? ['enable f16;'] : [];
  bufs.forEach((b, i) => lines.push(`@group(0) @binding(${i + 1}) var<storage, read_write> F${i}: array<${T}>;`));
  const bufOf = (d) => bufs.findIndex((b) => d >= b.base && d < b.base + b.count);
  for (const P of [0, 1, 3, 5, 7, 9, 11, 13, 15, 17]) {
    const b = bufOf(P), base = bufs[b].base;
    const w = W[P].toPrecision(9);
    const idx = P === 0 ? `(${P - base}u + s) * P.n + k` : `(${P - base}u + s) * P.n + k`;
    if (half) {
      lines.push(`fn ld${P}(s: u32, k: u32) -> f32 { return f32(F${b}[${idx}]) + ${w}; }`);
      lines.push(`fn st${P}(s: u32, k: u32, v: f32) { F${b}[${idx}] = f16(v - ${w}); }`);
    } else {
      lines.push(`fn ld${P}(s: u32, k: u32) -> f32 { return F${b}[${idx}]; }`);
      lines.push(`fn st${P}(s: u32, k: u32, v: f32) { F${b}[${idx}] = v; }`);
    }
  }
  return lines.join('\n');
}

/** The shader for a set of population buffers. */
export const flowShader = (half, bufs) => `${preludeWGSL(half, bufs)}\n${FLOW_GEN_WGSL}\n${FLOW_WGSL}`;

/** Wall records packed for the GPU (flow.wgsl REC layout). */
function packRecords(rec) {
  const out = new Uint32Array(Math.max(1, rec.count) * REC);
  const f32 = new Float32Array(out.buffer);
  for (let r = 0; r < rec.count; r++) {
    const b = r * REC;
    out[b] = rec.cell[r];
    out[b + 1] = rec.mask[r] | (rec.onBelt?.[r] ? 1 : 0);
    out[b + 2] = rec.groundMask[r];
    for (let k = 0; k < 18; k++) out[b + R_Q + (k >> 2)] |= rec.q[18 * r + k] << ((k & 3) * 8);
    out[b + R_N] = f16bits(rec.normal[3 * r]) | (f16bits(rec.normal[3 * r + 1]) << 16);
    out[b + R_N + 1] = f16bits(rec.normal[3 * r + 2]) | (f16bits(rec.dist[r]) << 16);
    for (let k = 1; k < 19; k++) { f32[b + R_AUX + k - 1] = W[k]; f32[b + R_BB + k - 1] = W[k]; }
    out[b + R_SAMP] = rec.samp?.[r] || 0;
    f32[b + R_Y2] = rec.y2?.[r] || 0;
    if (rec.area) for (let a = 0; a < 3; a++) f32[b + R_AREA + a] = rec.area[3 * r + a];
  }
  return out;
}

export class FlowGPU {
  /**
   * @param {object} grid  flow.js buildFlowGrid()
   * @param {{uLat: number, nuLat: number, smagorinsky?: number, collision?: 'rr'|'bgk', wallModel?: boolean, half?: boolean, timing?: boolean, belt?: boolean, wgx?: number, wgy?: number}} o
   */
  static async create(grid, o) {
    const adapter = await hardwareAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    const half = o.half !== false && adapter.features.has('shader-f16');
    // benchmarks: GPU time from timestamp queries (FlowGPU.timing turns it on for every solver)
    const timing = !!(o.timing ?? FlowGPU.timing) && adapter.features.has('timestamp-query');
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredFeatures: [...(half ? ['shader-f16'] : []), ...(timing ? ['timestamp-query'] : [])],
      requiredLimits: {
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxBufferSize: lim.maxBufferSize,
        maxStorageBuffersPerShaderStage: Math.min(lim.maxStorageBuffersPerShaderStage, 10),
        maxComputeWorkgroupStorageSize: Math.min(lim.maxComputeWorkgroupStorageSize, 16384),
      },
    });
    let sim;
    try {
      device.pushErrorScope('out-of-memory');
      device.pushErrorScope('validation');
      sim = new FlowGPU(device, grid, o, { half, timing, info: adapter.info });
      const info = await sim.module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      const invalid = await device.popErrorScope();
      const oom = await device.popErrorScope();
      if (errors.length || invalid || oom) throw new Error(errors.map((m) => `${m.lineNum}: ${m.message}`).join('; ') || (oom || invalid).message);
      await device.queue.onSubmittedWorkDone();
      if (sim.lost) throw new Error('WebGPU device was lost during initialization.');
      return sim;
    } catch (err) {
      if (sim) sim.destroy();
      else device.destroy();
      throw err;
    }
  }

  constructor(device, grid, o, { half, timing, info }) {
    this.device = device;
    this.half = half;
    this.adapterInfo = info;
    this.grid = grid;
    this.dims = grid.dims;
    const [nx, ny, nz] = grid.dims;
    const N = (this.N = nx * ny * nz);
    const p = flowParams(o);
    this.p = p;
    this.collision = (o.collision || 'rr') === 'rr' ? 'rr' : 'bgk';
    this.wallModel = o.wallModel !== false && this.collision === 'rr';
    this.belt = o.belt !== false;
    this.wallMode = o.wallMode || 'model';
    this.steps = 0;
    this.accSteps = 0;
    this.rhoSteps = 0;
    this.gpuSeconds = 0;
    this.lost = false;
    device.lost.then(() => (this.lost = true));
    // a GPU error (out of memory, an invalid command) must stop the run, not leave it counting steps
    // that never ran
    this.error = null;
    device.addEventListener('uncapturederror', (e) => { this.error ||= e.error?.message || 'GPU error'; });

    const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, C = GPUBufferUsage.COPY_SRC;
    const maxBytes = Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize);
    this.bufs = populationBuffers(N, half, maxBytes);
    // the sampling kernel binds the population buffers, the cell kinds and the view fields
    if (this.bufs.length + 2 > device.limits.maxStorageBuffersPerShaderStage) throw new Error('The flow grid is too large for this GPU.');
    this.F = this.bufs.map((b) => device.createBuffer({ size: b.bytes, usage: S }));
    const kinds = new Uint32Array(Math.ceil(N / 4));
    new Uint8Array(kinds.buffer).set(grid.kind);
    this.cells = device.createBuffer({ size: kinds.byteLength, usage: S | D });
    device.queue.writeBuffer(this.cells, 0, kinds);
    const rec = grid.rec;
    if (!rec.onBelt) rec.onBelt = Uint8Array.from(rec.dist, (d, r) => (rec.groundMask[r] && d === 0.5 ? 1 : 0));
    this.nRec = rec.count;
    this.recWords = packRecords(rec);
    this.rec = device.createBuffer({ size: this.recWords.byteLength, usage: S | D | C });
    device.queue.writeBuffer(this.rec, 0, this.recWords);
    const faceCells = [];
    for (let c = 0; c < N; c++) if (grid.kind[c] === FACE) faceCells.push(c);
    const ft = faceTable(grid.kind, grid.dims, faceCells, grid.periodicZ);
    const faces = new Uint32Array(Math.max(1, faceCells.length) * 3);
    faceCells.forEach((c, j) => {
      faces[3 * j] = c;
      faces[3 * j + 1] = ft.n0[j] === -1 ? 0xffffffff : ft.n0[j] === -2 ? 0xfffffffe : ft.n0[j];
      faces[3 * j + 2] = ft.out[j];
    });
    this.nFace = faceCells.length;
    this.faces = device.createBuffer({ size: faces.byteLength, usage: S | D });
    device.queue.writeBuffer(this.faces, 0, faces);
    const partial = new Float32Array(9 * REDUCE_GROUPS + 1);
    partial[partial.length - 1] = REDUCE_GROUPS;
    this.partial = device.createBuffer({ size: partial.byteLength, usage: S | D });
    device.queue.writeBuffer(this.partial, 0, partial);
    this.history = device.createBuffer({ size: 12 * 4 * 16, usage: S | C });
    this.readback = device.createBuffer({ size: 12 * 4, usage: GPUBufferUsage.MAP_READ | D });
    this.ring = Array.from({ length: RING }, () => device.createBuffer({ size: 12 * 4 + 16, usage: GPUBufferUsage.MAP_READ | D }));
    this.ringNext = 0;
    // the reduced view grid: coarse cells of cf^3 cells, their latest sample and summed samples
    this.cf = Math.max(1, Math.ceil(Math.cbrt(N / MAX_COARSE)));
    this.coarseDims = [nx, ny, nz].map((d) => Math.ceil(d / this.cf));
    this.nCoarse = this.coarseDims[0] * this.coarseDims[1] * this.coarseDims[2];
    this.coarse = device.createBuffer({ size: 2 * 16 * this.nCoarse, usage: S | D | C });
    this.samples = 0;
    this.paramStride = Math.ceil(PARAM_BYTES / device.limits.minUniformBufferOffsetAlignment) * device.limits.minUniformBufferOffsetAlignment;
    this.params = device.createBuffer({ size: this.paramStride * (MAX_STEPS + 1), usage: GPUBufferUsage.UNIFORM | D });
    if (timing) {
      // two timestamps for step(), and two per ring slot for submitBatch()
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 + 2 * RING });
      // resolved at 256-byte offsets (WebGPU's rule): step() at 0, ring slot k at 256 (k + 1)
      this.queryBuf = device.createBuffer({ size: 256 * (1 + RING), usage: GPUBufferUsage.QUERY_RESOLVE | C });
      this.queryRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.MAP_READ | D });
    }

    this.module = device.createShaderModule({ code: flowShader(half, this.bufs) });
    const wgx = o.wgx || 64, wgy = o.wgy || 1;
    this.wg = [wgx, wgy];
    const uniform = { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: PARAM_BYTES } };
    const storage = (binding, type = 'storage') => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type } });
    const fEntries = this.F.map((_, i) => storage(i + 1));
    const make = (entries, bindings, entry, constants) => {
      const layout = device.createBindGroupLayout({ entries: [uniform, ...entries] });
      const pipeline = device.createComputePipeline({
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
        compute: { module: this.module, entryPoint: entry, constants },
      });
      const group = device.createBindGroup({
        layout,
        entries: [{ binding: 0, resource: { buffer: this.params, size: PARAM_BYTES } }, ...bindings.map(([binding, buffer]) => ({ binding, resource: { buffer } }))],
      });
      return { pipeline, group };
    };
    const fBind = this.F.map((b, i) => [i + 1, b]);
    const RR = this.collision === 'rr';
    this.kernels = {
      init: make(fEntries, fBind, 'init', { RR }),
      bulk: make([...fEntries, storage(10, 'read-only-storage')], [...fBind, [10, this.cells]], 'bulk', { WGX: wgx, WGY: wgy, RR }),
      wall: make([...fEntries, storage(11)], [...fBind, [11, this.rec]], 'wall', { RR }),
      face: make([...fEntries, storage(12, 'read-only-storage')], [...fBind, [12, this.faces]], 'face', { RR }),
      reduce1: make([storage(11), storage(13)], [[11, this.rec], [13, this.partial]], 'reduce1', { RR }),
      reduce2: make([storage(13), storage(14)], [[13, this.partial], [14, this.history]], 'reduce2', { RR }),
      sample: make([...fEntries, storage(10, 'read-only-storage'), storage(15)], [...fBind, [10, this.cells], [15, this.coarse]], 'sample', { RR, CF: this.cf }),
      clearRho: make([storage(11)], [[11, this.rec]], 'clearRho', { RR }),
    };
    this.reset();
  }

  /** Dispatch size for a 1D kernel of `count` threads in groups of 64 (wrapping into y past 65535). */
  groups1(count) {
    const g = Math.max(1, Math.ceil(count / 64));
    return g <= 65535 ? [g, 1] : [65535, Math.ceil(g / 65535)];
  }

  writeParams(count, slot = 0) {
    const [nx, ny, nz] = this.dims;
    const buf = new ArrayBuffer(this.paramStride * count);
    for (let s = 0; s < count; s++) {
      const off = s * this.paramStride, step = this.steps + s;
      const uin = inletVelocity(this.p.uLat, step);
      new Uint32Array(buf, off, 8).set([nx, ny, nz, this.N, this.nRec, this.nFace, this.grid.periodicZ ? 1 : 0, step & 1]);
      new Float32Array(buf, off + 32, 5).set([this.p.tau0, this.p.smag, uin, this.belt ? uin : 0, this.p.nuLat]);
      new Uint32Array(buf, off + 52, 3).set([this.wallModel ? { slip: 1, replace: 2, max: 3, model: 4 }[this.wallMode] : 0, slot, 0]);
    }
    this.device.queue.writeBuffer(this.params, 0, buf);
  }

  reset() {
    this.steps = 0;
    this.accSteps = 0;
    this.rhoSteps = 0;
    this.samples = 0;
    this.device.queue.writeBuffer(this.coarse, 0, new Float32Array(8 * this.nCoarse));
    this.writeParams(1);
    this.device.queue.writeBuffer(this.rec, 0, this.recWords);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const k = this.kernels.init;
    pass.setPipeline(k.pipeline);
    pass.setBindGroup(0, k.group, [0]);
    pass.dispatchWorkgroups(...this.groups1(this.N));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /** Run `count` steps, in submissions of about SUBMIT_MS each; resolves when the GPU has run them. */
  async step(count) {
    if (this.lost) throw new Error('WebGPU device was lost. Try the CPU engine.');
    if (this.error) throw new Error(`GPU error: ${this.error}`);
    this.submitSteps ||= firstSubmitSteps(this.N);
    while (count > 0) {
      const n = Math.min(count, this.submitSteps);
      const t0 = performance.now();
      this.writeParams(n);
      const enc = this.device.createCommandEncoder();
      const timed = !!this.querySet;
      const pass = enc.beginComputePass(timed ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined);
      const [nx, ny, nz] = this.dims;
      const bulkGroups = [Math.ceil(nx / this.wg[0]), Math.ceil(ny / this.wg[1]), nz];
      const wallGroups = this.groups1(this.nRec), faceGroups = this.groups1(this.nFace);
      const { bulk, wall, face } = this.kernels;
      for (let s = 0; s < n; s++) {
        const off = [s * this.paramStride];
        pass.setPipeline(bulk.pipeline);
        pass.setBindGroup(0, bulk.group, off);
        pass.dispatchWorkgroups(...bulkGroups);
        if (this.nRec) {
          pass.setPipeline(wall.pipeline);
          pass.setBindGroup(0, wall.group, off);
          pass.dispatchWorkgroups(...wallGroups);
        }
        pass.setPipeline(face.pipeline);
        pass.setBindGroup(0, face.group, off);
        pass.dispatchWorkgroups(...faceGroups);
      }
      pass.end();
      if (timed) {
        enc.resolveQuerySet(this.querySet, 0, 2, this.queryBuf, 0);
        enc.copyBufferToBuffer(this.queryBuf, 0, this.queryRead, 0, 16);
      }
      this.device.queue.submit([enc.finish()]);
      this.steps += n;
      this.accSteps += n;
      count -= n;
      await this.device.queue.onSubmittedWorkDone();
      if (this.error) throw new Error(`GPU error: ${this.error}`);
      const dt = performance.now() - t0;
      this.submitSteps = Math.max(1, Math.min(MAX_STEPS, Math.round((n * SUBMIT_MS) / Math.max(1, dt))));
      if (timed) {
        await this.queryRead.mapAsync(GPUMapMode.READ);
        const t = new BigInt64Array(this.queryRead.getMappedRange().slice(0));
        this.queryRead.unmap();
        const ns = Number(t[1] - t[0]);
        if (ns > 0) this.gpuSeconds += ns / 1e9;
      }
    }
  }

  /**
   * The app's pipelined stepping: queue `count` steps (at most MAX_STEPS) without waiting, with a
   * sample of the view fields every `sampleEvery` steps, and the forces summed over the batch.
   * Resolves once the GPU has run the batch, to {steps, me, pressure, friction, gpuSeconds}.
   * Keep one or two batches in flight so the GPU never waits for the page.
   */
  submitBatch(count, { sampleEvery = 0 } = {}) {
    if (this.lost) return Promise.reject(new Error('WebGPU device was lost. Try the CPU engine.'));
    if (this.error) return Promise.reject(new Error(`GPU error: ${this.error}`));
    const n = Math.max(1, Math.min(count, MAX_STEPS));
    const slot = this.ringNext;
    this.ringNext = (slot + 1) % RING;
    const staging = this.ring[slot];
    // params for each step and the one after (samples use the next step's loads)
    this.writeParams(n + 1, slot);
    const enc = this.device.createCommandEncoder();
    const q = 2 + 2 * slot;
    const pass = enc.beginComputePass(this.querySet ? { timestampWrites: { querySet: this.querySet, beginningOfPassWriteIndex: q, endOfPassWriteIndex: q + 1 } } : undefined);
    const [nx, ny, nz] = this.dims;
    const bulkGroups = [Math.ceil(nx / this.wg[0]), Math.ceil(ny / this.wg[1]), nz];
    const wallGroups = this.groups1(this.nRec), faceGroups = this.groups1(this.nFace), sampleGroups = this.groups1(this.nCoarse);
    const { bulk, wall, face, sample, reduce1, reduce2 } = this.kernels;
    let samples = 0;
    for (let s = 0; s < n; s++) {
      const off = [s * this.paramStride];
      pass.setPipeline(bulk.pipeline);
      pass.setBindGroup(0, bulk.group, off);
      pass.dispatchWorkgroups(...bulkGroups);
      if (this.nRec) {
        pass.setPipeline(wall.pipeline);
        pass.setBindGroup(0, wall.group, off);
        pass.dispatchWorkgroups(...wallGroups);
      }
      pass.setPipeline(face.pipeline);
      pass.setBindGroup(0, face.group, off);
      pass.dispatchWorkgroups(...faceGroups);
      if (sampleEvery > 0 && (this.steps + s + 1) % sampleEvery === 0) {
        pass.setPipeline(sample.pipeline);
        pass.setBindGroup(0, sample.group, [(s + 1) * this.paramStride]);
        pass.dispatchWorkgroups(...sampleGroups);
        samples++;
      }
    }
    // this batch's force sums into history[slot] (reduce1 reads the records, reduce2 the partials)
    pass.setPipeline(reduce1.pipeline);
    pass.setBindGroup(0, reduce1.group, [0]);
    pass.dispatchWorkgroups(REDUCE_GROUPS);
    pass.setPipeline(reduce2.pipeline);
    pass.setBindGroup(0, reduce2.group, [0]);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(this.history, slot * 48, staging, 0, 48);
    if (this.querySet) {
      enc.resolveQuerySet(this.querySet, q, 2, this.queryBuf, 256 * (slot + 1));
      enc.copyBufferToBuffer(this.queryBuf, 256 * (slot + 1), staging, 48, 16);
    }
    this.device.queue.submit([enc.finish()]);
    this.steps += n;
    this.samples += samples;
    this.rhoSteps += n;
    return staging.mapAsync(GPUMapMode.READ).then(() => {
      const range = staging.getMappedRange();
      const v = new Float32Array(range.slice(0, 48));
      let gpuSeconds = NaN;
      if (this.querySet) {
        const t = new BigInt64Array(range.slice(48, 64));
        const ns = Number(t[1] - t[0]);
        if (ns > 0) { gpuSeconds = ns / 1e9; this.gpuSeconds += gpuSeconds; }
      }
      staging.unmap();
      return { steps: n, me: [v[0], v[1], v[2]], pressure: [v[3], v[4], v[5]], friction: [v[6], v[7], v[8]], gpuSeconds };
    });
  }

  /** Start the time averages again (the view fields and the surface pressure). */
  resetAverages() {
    this.samples = 0;
    this.rhoSteps = 0;
    this.device.queue.writeBuffer(this.coarse, 16 * this.nCoarse, new Float32Array(4 * this.nCoarse));
    // the wall records' density sums
    if (!this.nRec) return;
    this.writeParams(1, 0);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(this.kernels.clearRho.pipeline);
    pass.setBindGroup(0, this.kernels.clearRho.group, [0]);
    pass.dispatchWorkgroups(...this.groups1(this.nRec));
    pass.end();
    this.device.queue.submit([enc.finish()]);
  }

  /**
   * The reduced view fields: {dims, factor, inst, avg} with [rho - 1, ux, uy, uz] per coarse cell
   * (rho - 1 = -2 where a coarse cell holds no bulk fluid); avg over the samples since resetAverages.
   */
  async readFields() {
    const bytes = 32 * this.nCoarse;
    const read = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.coarse, 0, read, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    const samples = this.samples;
    await read.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();
    const inst = all.subarray(0, 4 * this.nCoarse), avg = all.subarray(4 * this.nCoarse);
    if (samples) for (let i = 0; i < avg.length; i++) avg[i] /= samples;
    for (let i = 0; i < this.nCoarse; i++) if (inst[4 * i] === -2) avg[4 * i] = -2;
    return { dims: this.coarseDims, factor: this.cf, inst, avg, samples };
  }

  /**
   * Forces on the part since the last call, summed over its steps (lattice units): momentum
   * exchange and the pressure integral (flow-cpu.js takeForces).
   */
  async takeForces() {
    this.writeParams(1, 0);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    const { reduce1, reduce2 } = this.kernels;
    pass.setPipeline(reduce1.pipeline);
    pass.setBindGroup(0, reduce1.group, [0]);
    pass.dispatchWorkgroups(REDUCE_GROUPS);
    pass.setPipeline(reduce2.pipeline);
    pass.setBindGroup(0, reduce2.group, [0]);
    pass.dispatchWorkgroups(1);
    pass.end();
    enc.copyBufferToBuffer(this.history, 0, this.readback, 0, 48);
    this.device.queue.submit([enc.finish()]);
    await this.readback.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(this.readback.getMappedRange().slice(0));
    this.readback.unmap();
    const steps = this.accSteps;
    this.rhoSteps += steps;
    this.accSteps = 0;
    return { steps, me: [v[0], v[1], v[2]], pressure: [v[3], v[4], v[5]], friction: [v[6], v[7], v[8]] };
  }

  /** Time-averaged density minus one at each wall cell (for the surface pressure). */
  async surfaceRho() {
    const bytes = this.nRec * REC * 4;
    const read = this.device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.rec, 0, read, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const words = new Float32Array(read.getMappedRange());
    const out = new Float32Array(this.nRec);
    if (this.rhoSteps) for (let r = 0; r < this.nRec; r++) out[r] = words[r * REC + R_RHO] / this.rhoSteps;
    read.unmap();
    read.destroy();
    return out;
  }

  destroy() {
    for (const b of [...(this.F || []), ...(this.ring || []), this.cells, this.rec, this.faces, this.partial, this.history, this.readback, this.params, this.queryBuf, this.queryRead, this.coarse]) b?.destroy();
    this.querySet?.destroy();
    this.device.destroy();
  }
}
