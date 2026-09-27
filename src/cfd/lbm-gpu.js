// WebGPU implementation of the D3Q19 LBM in lbm-cpu.js (same boundary conditions,
// interpolated bounce-back and collision model), with the native app's kernel: unrolled over the
// directions, a per-cell flags word for walls, and 16-bit population storage where supported.
import { CX, CY, CZ, lbmParams, inletVelocity, validateGrid } from './lbm-cpu.js';
import { hardwareAdapter } from '../core/webgpu.js';

const WG = 64;
const MAX_BATCH = 400;

// population storage and its accessors: 32-bit, or 16-bit shifted by the rest weight ("FP16S",
// Lehmann et al. 2022: half the memory traffic, computing still in 32 bits)
const PRELUDE32 = /* wgsl */ `
@group(0) @binding(1) var<storage, read> fin: array<f32>;
@group(0) @binding(2) var<storage, read_write> fout: array<f32>;
fn ld(k: u32, w: f32) -> f32 { return fin[k]; }
fn st(k: u32, w: f32, v: f32) { fout[k] = v; }
`;
const PRELUDE16 = /* wgsl */ `enable f16;
@group(0) @binding(1) var<storage, read> fin: array<f16>;
@group(0) @binding(2) var<storage, read_write> fout: array<f16>;
fn ld(k: u32, w: f32) -> f32 { return f32(fin[k]) + w; }
fn st(k: u32, w: f32, v: f32) { fout[k] = f16(v - w); }
`;

// the same kernel as the native app's lbm.wgsl
const BODY = /* wgsl */ `
// D3Q19 lattice Boltzmann, pull streaming, BGK with a Smagorinsky subgrid model (workgroup size 64).
// The step kernel is unrolled over the 19 directions so the populations stay in registers.

struct Params {
  nx: u32, ny: u32, nz: u32, n: u32,
  stride: u32, writeMoments: u32, p1: u32, p2: u32,
  tau0: f32, uin: f32, smag: f32, p3: f32,
};
@group(0) @binding(0) var<uniform> P: Params;
// fin / fout and their accessors ld(k, w) / st(k, w, v) come from a prelude: 32-bit populations, or
// 16-bit ones stored shifted by their rest weight w (FP16S)
@group(0) @binding(3) var<storage, read> solid: array<u32>;
@group(0) @binding(4) var<storage, read_write> moments: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> links: array<u32>; // 19*N bytes, 4 per word
// per cell: bit i (1..18) = the neighbour at c - c_i is solid, plus the cell's own kind. One read
// replaces 19 neighbour lookups (a locally encoded boundary); solid[] is only read at walls.
@group(0) @binding(6) var<storage, read> flags: array<u32>;
const SOLID_CELL: u32 = 1u << 29u;
const INLET_CELL: u32 = 1u << 30u;
const OPEN_CELL: u32 = 1u << 31u;

fn linkByte(k: u32) -> u32 { return (links[k >> 2u] >> ((k & 3u) * 8u)) & 0xffu; }

const CX: array<i32, 19> = array<i32, 19>(0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
const CY: array<i32, 19> = array<i32, 19>(0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
const CZ: array<i32, 19> = array<i32, 19>(0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);
const OPP: array<u32, 19> = array<u32, 19>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u, 16u, 15u, 18u, 17u);
const WT: array<f32, 19> = array<f32, 19>(
  0.3333333333, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778);

fn feq(i: u32, rho: f32, u: vec3<f32>, usq: f32) -> f32 {
  let cu = f32(CX[i]) * u.x + f32(CY[i]) * u.y + f32(CZ[i]) * u.z;
  return WT[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq);
}

fn cellIndex(gid: vec3<u32>) -> u32 { return gid.x + gid.y * P.stride; }

@compute @workgroup_size(${WG})
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = cellIndex(gid);
  if (c >= P.n) { return; }
  for (var i = 0u; i < 19u; i++) { st(i * P.n + c, WT[i], WT[i]); }
  moments[c] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
}

// incoming population i at a fluid cell whose neighbour c - c_i is solid: bounce-back, with the
// interpolated (Bouzidi) variant where the wall's position along the link is known
fn wall(i: u32, c: u32, n: u32, sy: i32, sz: i32) -> f32 {
  let j = OPP[i];
  var v = ld(j * n + c, WT[j]);
  let qb = linkByte(i * n + c);
  if (qb != 0u) {
    let q = f32(qb - 1u) / 254.0;
    if (q < 0.5) {
      let n2 = u32(i32(c) + CX[i] + sy * CY[i] + sz * CZ[i]);
      if (solid[n2] == 0u) { v = 2.0 * q * v + (1.0 - 2.0 * q) * ld(j * n + n2, WT[j]); }
    } else {
      v = (0.5 / q) * v + (1.0 - 0.5 / q) * ld(i * n + c, WT[i]);
    }
  }
  return v;
}

@compute @workgroup_size(${WG})
fn step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = cellIndex(gid);
  let n = P.n;
  if (c >= n) { return; }
  let nx = P.nx; let ny = P.ny; let nz = P.nz;
  let fl = flags[c];
  // the moments are only read with snapshots: written on the last step of a batch
  let write = P.writeMoments != 0u;
  if ((fl & SOLID_CELL) != 0u) { if (write) { moments[c] = vec4<f32>(1.0, 0.0, 0.0, 0.0); } return; }
  if ((fl & INLET_CELL) != 0u) {
    let u = vec3<f32>(P.uin, 0.0, 0.0);
    let usq = 1.5 * dot(u, u);
    for (var i = 0u; i < 19u; i++) { st(i * n + c, WT[i], feq(i, 1.0, u, usq)); }
    if (write) { moments[c] = vec4<f32>(1.0, u); }
    return;
  }
  let x = c % nx; let y = (c / nx) % ny; let z = c / (nx * ny);
  if ((fl & OPEN_CELL) != 0u) {
    // open boundary: velocity of the nearest interior cell, ambient pressure
    let n0 = min(x, nx - 2u) + nx * (clamp(y, 1u, ny - 2u) + ny * clamp(z, 1u, nz - 2u));
    var u = vec3<f32>(P.uin, 0.0, 0.0);
    if (solid[n0] == 0u) {
      var r = 0.0; var m = vec3<f32>(0.0);
      for (var i = 0u; i < 19u; i++) {
        let v = ld(i * n + n0, WT[i]);
        r += v;
        m += v * vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
      }
      u = m / r;
    }
    let usq = 1.5 * dot(u, u);
    for (var i = 0u; i < 19u; i++) { st(i * n + c, WT[i], feq(i, 1.0, u, usq)); }
    if (write) { moments[c] = vec4<f32>(1.0, u); }
    return;
  }
  let sy = i32(nx); let sz = i32(nx * ny);
  let f0 = ld(c, WT[0]);
  var f1 = ld(1u * n + u32(i32(c) + -1), WT[1]);
  if ((fl & 2u) != 0u) { f1 = wall(1u, c, n, sy, sz); }
  var f2 = ld(2u * n + u32(i32(c) + 1), WT[2]);
  if ((fl & 4u) != 0u) { f2 = wall(2u, c, n, sy, sz); }
  var f3 = ld(3u * n + u32(i32(c) + -sy), WT[3]);
  if ((fl & 8u) != 0u) { f3 = wall(3u, c, n, sy, sz); }
  var f4 = ld(4u * n + u32(i32(c) + sy), WT[4]);
  if ((fl & 16u) != 0u) { f4 = wall(4u, c, n, sy, sz); }
  var f5 = ld(5u * n + u32(i32(c) + -sz), WT[5]);
  if ((fl & 32u) != 0u) { f5 = wall(5u, c, n, sy, sz); }
  var f6 = ld(6u * n + u32(i32(c) + sz), WT[6]);
  if ((fl & 64u) != 0u) { f6 = wall(6u, c, n, sy, sz); }
  var f7 = ld(7u * n + u32(i32(c) + -1 - sy), WT[7]);
  if ((fl & 128u) != 0u) { f7 = wall(7u, c, n, sy, sz); }
  var f8 = ld(8u * n + u32(i32(c) + 1 + sy), WT[8]);
  if ((fl & 256u) != 0u) { f8 = wall(8u, c, n, sy, sz); }
  var f9 = ld(9u * n + u32(i32(c) + -1 + sy), WT[9]);
  if ((fl & 512u) != 0u) { f9 = wall(9u, c, n, sy, sz); }
  var f10 = ld(10u * n + u32(i32(c) + 1 - sy), WT[10]);
  if ((fl & 1024u) != 0u) { f10 = wall(10u, c, n, sy, sz); }
  var f11 = ld(11u * n + u32(i32(c) + -1 - sz), WT[11]);
  if ((fl & 2048u) != 0u) { f11 = wall(11u, c, n, sy, sz); }
  var f12 = ld(12u * n + u32(i32(c) + 1 + sz), WT[12]);
  if ((fl & 4096u) != 0u) { f12 = wall(12u, c, n, sy, sz); }
  var f13 = ld(13u * n + u32(i32(c) + -1 + sz), WT[13]);
  if ((fl & 8192u) != 0u) { f13 = wall(13u, c, n, sy, sz); }
  var f14 = ld(14u * n + u32(i32(c) + 1 - sz), WT[14]);
  if ((fl & 16384u) != 0u) { f14 = wall(14u, c, n, sy, sz); }
  var f15 = ld(15u * n + u32(i32(c) + -sy - sz), WT[15]);
  if ((fl & 32768u) != 0u) { f15 = wall(15u, c, n, sy, sz); }
  var f16 = ld(16u * n + u32(i32(c) + sy + sz), WT[16]);
  if ((fl & 65536u) != 0u) { f16 = wall(16u, c, n, sy, sz); }
  var f17 = ld(17u * n + u32(i32(c) + -sy + sz), WT[17]);
  if ((fl & 131072u) != 0u) { f17 = wall(17u, c, n, sy, sz); }
  var f18 = ld(18u * n + u32(i32(c) + sy - sz), WT[18]);
  if ((fl & 262144u) != 0u) { f18 = wall(18u, c, n, sy, sz); }
  let rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18;
  let u = vec3<f32>(f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14, f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18, f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18) / rho;
  let usq = 1.5 * dot(u, u);
  let e0 = WT[0] * rho * (1.0 - usq);
  let cu1 = u.x;
  let e1 = WT[1] * rho * (1.0 + 3.0 * cu1 + 4.5 * cu1 * cu1 - usq);
  let cu2 = -u.x;
  let e2 = WT[2] * rho * (1.0 + 3.0 * cu2 + 4.5 * cu2 * cu2 - usq);
  let cu3 = u.y;
  let e3 = WT[3] * rho * (1.0 + 3.0 * cu3 + 4.5 * cu3 * cu3 - usq);
  let cu4 = -u.y;
  let e4 = WT[4] * rho * (1.0 + 3.0 * cu4 + 4.5 * cu4 * cu4 - usq);
  let cu5 = u.z;
  let e5 = WT[5] * rho * (1.0 + 3.0 * cu5 + 4.5 * cu5 * cu5 - usq);
  let cu6 = -u.z;
  let e6 = WT[6] * rho * (1.0 + 3.0 * cu6 + 4.5 * cu6 * cu6 - usq);
  let cu7 = u.x + u.y;
  let e7 = WT[7] * rho * (1.0 + 3.0 * cu7 + 4.5 * cu7 * cu7 - usq);
  let cu8 = -u.x - u.y;
  let e8 = WT[8] * rho * (1.0 + 3.0 * cu8 + 4.5 * cu8 * cu8 - usq);
  let cu9 = u.x - u.y;
  let e9 = WT[9] * rho * (1.0 + 3.0 * cu9 + 4.5 * cu9 * cu9 - usq);
  let cu10 = -u.x + u.y;
  let e10 = WT[10] * rho * (1.0 + 3.0 * cu10 + 4.5 * cu10 * cu10 - usq);
  let cu11 = u.x + u.z;
  let e11 = WT[11] * rho * (1.0 + 3.0 * cu11 + 4.5 * cu11 * cu11 - usq);
  let cu12 = -u.x - u.z;
  let e12 = WT[12] * rho * (1.0 + 3.0 * cu12 + 4.5 * cu12 * cu12 - usq);
  let cu13 = u.x - u.z;
  let e13 = WT[13] * rho * (1.0 + 3.0 * cu13 + 4.5 * cu13 * cu13 - usq);
  let cu14 = -u.x + u.z;
  let e14 = WT[14] * rho * (1.0 + 3.0 * cu14 + 4.5 * cu14 * cu14 - usq);
  let cu15 = u.y + u.z;
  let e15 = WT[15] * rho * (1.0 + 3.0 * cu15 + 4.5 * cu15 * cu15 - usq);
  let cu16 = -u.y - u.z;
  let e16 = WT[16] * rho * (1.0 + 3.0 * cu16 + 4.5 * cu16 * cu16 - usq);
  let cu17 = u.y - u.z;
  let e17 = WT[17] * rho * (1.0 + 3.0 * cu17 + 4.5 * cu17 * cu17 - usq);
  let cu18 = -u.y + u.z;
  let e18 = WT[18] * rho * (1.0 + 3.0 * cu18 + 4.5 * cu18 * cu18 - usq);
  let d0 = f0 - e0;
  let d1 = f1 - e1;
  let d2 = f2 - e2;
  let d3 = f3 - e3;
  let d4 = f4 - e4;
  let d5 = f5 - e5;
  let d6 = f6 - e6;
  let d7 = f7 - e7;
  let d8 = f8 - e8;
  let d9 = f9 - e9;
  let d10 = f10 - e10;
  let d11 = f11 - e11;
  let d12 = f12 - e12;
  let d13 = f13 - e13;
  let d14 = f14 - e14;
  let d15 = f15 - e15;
  let d16 = f16 - e16;
  let d17 = f17 - e17;
  let d18 = f18 - e18;
  let pxx = d1 + d2 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14;
  let pyy = d3 + d4 + d7 + d8 + d9 + d10 + d15 + d16 + d17 + d18;
  let pzz = d5 + d6 + d11 + d12 + d13 + d14 + d15 + d16 + d17 + d18;
  let pxy = d7 + d8 - d9 - d10;
  let pxz = d11 + d12 - d13 - d14;
  let pyz = d15 + d16 - d17 - d18;
  let q = sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz));
  let tau = 0.5 * (P.tau0 + sqrt(P.tau0 * P.tau0 + P.smag * q / rho));
  let om = 1.0 / tau;
  st(0u * n + c, WT[0], f0 - om * d0);
  st(1u * n + c, WT[1], f1 - om * d1);
  st(2u * n + c, WT[2], f2 - om * d2);
  st(3u * n + c, WT[3], f3 - om * d3);
  st(4u * n + c, WT[4], f4 - om * d4);
  st(5u * n + c, WT[5], f5 - om * d5);
  st(6u * n + c, WT[6], f6 - om * d6);
  st(7u * n + c, WT[7], f7 - om * d7);
  st(8u * n + c, WT[8], f8 - om * d8);
  st(9u * n + c, WT[9], f9 - om * d9);
  st(10u * n + c, WT[10], f10 - om * d10);
  st(11u * n + c, WT[11], f11 - om * d11);
  st(12u * n + c, WT[12], f12 - om * d12);
  st(13u * n + c, WT[13], f13 - om * d13);
  st(14u * n + c, WT[14], f14 - om * d14);
  st(15u * n + c, WT[15], f15 - om * d15);
  st(16u * n + c, WT[16], f16 - om * d16);
  st(17u * n + c, WT[17], f17 - om * d17);
  st(18u * n + c, WT[18], f18 - om * d18);
  if (write) { moments[c] = vec4<f32>(rho, u); }
}
`;

export const shaderSource = (half) => (half ? PRELUDE16 : PRELUDE32) + BODY;

const SOLID_CELL = 1 << 29, INLET_CELL = 1 << 30, OPEN_CELL = 2 ** 31;

/** Per cell: bit i = the neighbour at c - c_i is solid (fluid cells), or the cell's own kind. */
function cellFlags(dims, solid) {
  const [nx, ny, nz] = dims;
  const off = CX.map((cx, i) => cx + nx * (CY[i] + ny * CZ[i]));
  const fl = new Uint32Array(nx * ny * nz);
  for (let z = 0, c = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++, c++) {
        if (solid[c]) fl[c] = SOLID_CELL;
        else if (x === 0) fl[c] = INLET_CELL;
        else if (x === nx - 1 || y === 0 || z === 0 || y === ny - 1 || z === nz - 1) fl[c] = OPEN_CELL;
        else {
          let f = 0;
          for (let i = 1; i < 19; i++) if (solid[c - off[i]]) f |= 1 << i;
          fl[c] = f;
        }
      }
    }
  }
  return fl;
}

export async function webgpuAvailable() {
  try {
    return !!(await hardwareAdapter());
  } catch {
    return false;
  }
}

/** Largest grid (cells) the GPU can hold: 19 populations per cell (16-bit where supported) in one storage buffer. */
export async function gpuMaxCells() {
  const adapter = await hardwareAdapter();
  if (!adapter) throw new Error('No WebGPU adapter');
  const lim = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
  return Math.floor(lim / (19 * (adapter.features.has('shader-f16') ? 2 : 4)));
}

export class LBMGPU {
  static async create(o) {
    validateGrid(o);
    lbmParams(o);
    const adapter = await hardwareAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    const half = o.half !== false && adapter.features.has('shader-f16');
    const device = await adapter.requestDevice({
      requiredFeatures: half ? ['shader-f16'] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    let sim;
    try {
      device.pushErrorScope('validation');
      sim = new LBMGPU(device, o, half);
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

  constructor(device, o, half = false) {
    this.device = device;
    this.half = half;
    this.dims = o.dims;
    const [nx, ny, nz] = o.dims;
    const N = (this.N = nx * ny * nz);
    Object.assign(this, lbmParams(o));
    this.steps = 0;
    this.lost = false;
    device.lost.then(() => (this.lost = true));

    const S = GPUBufferUsage.STORAGE;
    const fBytes = Math.ceil((19 * N * (half ? 2 : 4)) / 4) * 4;
    this.fBufs = [0, 1].map(() => device.createBuffer({ size: fBytes, usage: S }));
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
    this.flagBuf = device.createBuffer({ size: N * 4, usage: S | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(this.flagBuf, 0, cellFlags(o.dims, o.solid));
    this.readBuf = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const alignment = device.limits.minUniformBufferOffsetAlignment;
    this.paramStride = Math.ceil(48 / alignment) * alignment;
    this.paramBuf = device.createBuffer({ size: this.paramStride * MAX_BATCH, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const groups = Math.ceil(N / WG);
    this.wgX = Math.min(groups, 65535);
    this.wgY = Math.ceil(groups / this.wgX);

    const module = (this.module = device.createShaderModule({ code: shaderSource(half) }));
    const layout = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform', hasDynamicOffset: true, minBindingSize: 48 } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
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
          { binding: 6, resource: { buffer: this.flagBuf } },
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
      // the moments are only read after a batch: written on its last step
      new Uint32Array(buf, offset, 8).set([nx, ny, nz, this.N, this.wgX * WG, s === count - 1 ? 1 : 0, 0, 0]);
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
    for (const b of [...this.fBufs, this.solidBuf, this.flagBuf, this.macroBuf, this.linkBuf, this.readBuf, this.paramBuf]) b?.destroy();
    this.device.destroy();
  }
}
