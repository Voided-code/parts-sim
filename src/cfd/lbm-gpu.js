// WebGPU implementation of the D3Q19 LBM in lbm-cpu.js (same boundary conditions,
// interpolated bounce-back and collision model).
import { W, lbmParams, inletVelocity, validateGrid } from './lbm-cpu.js';

const WG = 64;
const MAX_BATCH = 400;

const SHADER = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, n: u32,
  stride: u32, p0: u32, p1: u32, p2: u32,
  tau0: f32, uin: f32, smag: f32, p3: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
@group(0) @binding(3) var<storage, read> solid: array<u32>;
@group(0) @binding(4) var<storage, read_write> moments: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> links: array<u32>; // 19*N bytes, 4 per word

fn linkByte(k: u32) -> u32 { return (links[k >> 2u] >> ((k & 3u) * 8u)) & 0xffu; }

var<private> CX: array<i32, 19> = array<i32, 19>(0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
var<private> CY: array<i32, 19> = array<i32, 19>(0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
var<private> CZ: array<i32, 19> = array<i32, 19>(0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);
var<private> OPP: array<u32, 19> = array<u32, 19>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u, 16u, 15u, 18u, 17u);
var<private> WT: array<f32, 19> = array<f32, 19>(
  ${W.map((w) => w.toFixed(10)).join(', ')});

fn feq(i: u32, rho: f32, u: vec3<f32>, usq: f32) -> f32 {
  let cu = f32(CX[i]) * u.x + f32(CY[i]) * u.y + f32(CZ[i]) * u.z;
  return WT[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq);
}

fn cellIndex(gid: vec3<u32>) -> u32 { return gid.x + gid.y * P.stride; }

@compute @workgroup_size(${WG})
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = cellIndex(gid);
  if (c >= P.n) { return; }
  for (var i = 0u; i < 19u; i++) { fout[i * P.n + c] = WT[i]; }
  moments[c] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(${WG})
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = cellIndex(gid);
  let n = P.n;
  if (c >= n) { return; }
  let nx = P.nx; let ny = P.ny; let nz = P.nz;
  let x = c % nx; let y = (c / nx) % ny; let z = c / (nx * ny);
  if (solid[c] != 0u) { moments[c] = vec4<f32>(1.0, 0.0, 0.0, 0.0); return; }
  if (x == 0u) {
    let u = vec3<f32>(P.uin, 0.0, 0.0);
    let usq = 1.5 * dot(u, u);
    for (var i = 0u; i < 19u; i++) { fout[i * n + c] = feq(i, 1.0, u, usq); }
    moments[c] = vec4<f32>(1.0, u);
    return;
  }
  if (x == nx - 1u || y == 0u || z == 0u || y == ny - 1u || z == nz - 1u) {
    // open boundary: velocity of the nearest interior cell, ambient pressure
    let n0 = min(x, nx - 2u) + nx * (clamp(y, 1u, ny - 2u) + ny * clamp(z, 1u, nz - 2u));
    var u = vec3<f32>(P.uin, 0.0, 0.0);
    if (solid[n0] == 0u) {
      var r = 0.0; var m = vec3<f32>(0.0);
      for (var i = 0u; i < 19u; i++) {
        let v = fin[i * n + n0];
        r += v;
        m += v * vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
      }
      u = m / r;
    }
    let usq = 1.5 * dot(u, u);
    for (var i = 0u; i < 19u; i++) { fout[i * n + c] = feq(i, 1.0, u, usq); }
    moments[c] = vec4<f32>(1.0, u);
    return;
  }
  var f: array<f32, 19>;
  var rho = 0.0;
  var m = vec3<f32>(0.0);
  for (var i = 0u; i < 19u; i++) {
    let s = u32(i32(x) - CX[i]) + nx * (u32(i32(y) - CY[i]) + ny * u32(i32(z) - CZ[i]));
    var v: f32;
    if (solid[s] == 0u) {
      v = fin[i * n + s];
    } else {
      let j = OPP[i];
      v = fin[j * n + c];
      let qb = linkByte(i * n + c);
      if (qb != 0u) {
        let q = f32(qb - 1u) / 254.0;
        if (q < 0.5) {
          let n2 = u32(i32(x) + CX[i]) + nx * (u32(i32(y) + CY[i]) + ny * u32(i32(z) + CZ[i]));
          if (solid[n2] == 0u) { v = 2.0 * q * v + (1.0 - 2.0 * q) * fin[j * n + n2]; }
        } else {
          v = (0.5 / q) * v + (1.0 - 0.5 / q) * fin[i * n + c];
        }
      }
    }
    f[i] = v;
    rho += v;
    m += v * vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
  }
  let u = m / rho;
  let usq = 1.5 * dot(u, u);
  var fe: array<f32, 19>;
  var pxx = 0.0; var pyy = 0.0; var pzz = 0.0; var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
  for (var i = 0u; i < 19u; i++) {
    let e = feq(i, rho, u, usq);
    fe[i] = e;
    let d = f[i] - e;
    let cx = f32(CX[i]); let cy = f32(CY[i]); let cz = f32(CZ[i]);
    pxx += cx * cx * d; pyy += cy * cy * d; pzz += cz * cz * d;
    pxy += cx * cy * d; pxz += cx * cz * d; pyz += cy * cz * d;
  }
  let q = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
  let tau = 0.5 * (P.tau0 + sqrt(P.tau0 * P.tau0 + P.smag * q / rho));
  let om = 1.0 / tau;
  for (var i = 0u; i < 19u; i++) { fout[i * n + c] = f[i] - om * (f[i] - fe[i]); }
  moments[c] = vec4<f32>(rho, u);
}
`;

export async function webgpuAvailable() {
  try {
    if (!navigator.gpu) return false;
    const adapter = await navigator.gpu.requestAdapter();
    return !!adapter;
  } catch {
    return false;
  }
}

/** Largest grid (cells) the GPU can hold, given 19 floats per cell in one storage buffer. */
export async function gpuMaxCells() {
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const lim = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  return Math.floor(lim / (19 * 4));
}

export class LBMGPU {
  static async create(o) {
    validateGrid(o);
    lbmParams(o);
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    let sim;
    try {
      device.pushErrorScope('validation');
      sim = new LBMGPU(device, o);
      const info = await sim.module.getCompilationInfo();
      const errors = info.messages.filter((m) => m.type === 'error');
      const err = await device.popErrorScope();
      if (errors.length || err) throw new Error(errors.map((m) => m.message).join('; ') || err.message);
      await device.queue.onSubmittedWorkDone();
      if (sim.lost) throw new Error('WebGPU device was lost during initialization.');
      return sim;
    } catch (err) {
      if (sim) sim.destroy();
      else device.destroy();
      throw err;
    }
  }

  constructor(device, o) {
    this.device = device;
    this.dims = o.dims;
    const [nx, ny, nz] = o.dims;
    const N = (this.N = nx * ny * nz);
    Object.assign(this, lbmParams(o));
    this.steps = 0;
    this.lost = false;
    device.lost.then(() => (this.lost = true));

    const S = GPUBufferUsage.STORAGE;
    this.fBufs = [0, 1].map(() => device.createBuffer({ size: 19 * N * 4, usage: S }));
    this.solidBuf = device.createBuffer({ size: N * 4, usage: S | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.solidBuf, 0, Uint32Array.from(o.solid));
    this.macroBuf = device.createBuffer({ size: N * 16, usage: S | GPUBufferUsage.COPY_SRC });
    if (o.links && o.links.length !== 19 * N) throw new Error('Wall links do not match the flow grid.');
    const linkBytes = Math.ceil((19 * N) / 4) * 4;
    this.linkBuf = device.createBuffer({ size: linkBytes, usage: S | GPUBufferUsage.COPY_DST });
    if (o.links) {
      const padded = new Uint8Array(linkBytes);
      padded.set(o.links);
      device.queue.writeBuffer(this.linkBuf, 0, padded);
    }
    this.readBuf = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const alignment = device.limits.minUniformBufferOffsetAlignment;
    this.paramStride = Math.ceil(48 / alignment) * alignment;
    this.paramBuf = device.createBuffer({ size: this.paramStride * MAX_BATCH, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const groups = Math.ceil(N / WG);
    this.wgX = Math.min(groups, 65535);
    this.wgY = Math.ceil(groups / this.wgX);

    const module = (this.module = device.createShaderModule({ code: SHADER }));
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 48 } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    this.stepPipe = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: 'step' } });
    this.initPipe = device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint: 'init' } });
    this.bindGroups = [0, 1].map((p) =>
      device.createBindGroup({
        layout,
        entries: [
          { binding: 0, resource: { buffer: this.paramBuf, size: 48 } },
          { binding: 1, resource: { buffer: this.fBufs[p] } },
          { binding: 2, resource: { buffer: this.fBufs[1 - p] } },
          { binding: 3, resource: { buffer: this.solidBuf } },
          { binding: 4, resource: { buffer: this.macroBuf } },
          { binding: 5, resource: { buffer: this.linkBuf } },
        ],
      }),
    );
    this.reset();
  }

  writeParams(count = 1) {
    const [nx, ny, nz] = this.dims;
    const buf = new ArrayBuffer(this.paramStride * count);
    for (let s = 0; s < count; s++) {
      const offset = s * this.paramStride;
      new Uint32Array(buf, offset, 8).set([nx, ny, nz, this.N, this.wgX * WG, 0, 0, 0]);
      new Float32Array(buf, offset + 32, 4).set([this.tau0, inletVelocity(this.uLat, this.steps + s), this.smag, 0]);
    }
    this.device.queue.writeBuffer(this.paramBuf, 0, buf);
  }

  reset() {
    this.steps = 0;
    this.parity = 0;
    this.writeParams();
    const enc = this.device.createCommandEncoder();
    // init writes into fBufs[1 - parity]; run it for both buffers
    for (const p of [1, 0]) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.initPipe);
      pass.setBindGroup(0, this.bindGroups[p], [0]);
      pass.dispatchWorkgroups(this.wgX, this.wgY);
      pass.end();
    }
    this.device.queue.submit([enc.finish()]);
  }

  /** Queue time steps with the same per-step inlet ramp as the CPU solver. */
  step(count) {
    if (this.lost) throw new Error('WebGPU device was lost. Try the CPU engine.');
    if (!Number.isInteger(count) || count < 1 || count > MAX_BATCH) throw new Error(`GPU batch must contain 1 to ${MAX_BATCH} steps.`);
    this.writeParams(count);
    const enc = this.device.createCommandEncoder();
    for (let s = 0; s < count; s++) {
      const pass = enc.beginComputePass();
      pass.setPipeline(this.stepPipe);
      pass.setBindGroup(0, this.bindGroups[this.parity], [s * this.paramStride]);
      pass.dispatchWorkgroups(this.wgX, this.wgY);
      pass.end();
      this.parity ^= 1;
    }
    this.device.queue.submit([enc.finish()]);
    this.steps += count;
    return this.device.queue.onSubmittedWorkDone();
  }

  /** Copy of [rho, ux, uy, uz] per cell. */
  async readMacro() {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.macroBuf, 0, this.readBuf, 0, this.N * 16);
    this.device.queue.submit([enc.finish()]);
    await this.readBuf.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.readBuf.getMappedRange().slice(0));
    this.readBuf.unmap();
    return out;
  }

  destroy() {
    for (const b of [...this.fBufs, this.solidBuf, this.macroBuf, this.linkBuf, this.readBuf, this.paramBuf]) b?.destroy();
    this.device.destroy();
  }
}
