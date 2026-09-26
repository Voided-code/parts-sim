// WebGPU version of the multigrid-preconditioned conjugate-gradient solve in solver.js.
//
// The multigrid hierarchy (Galerkin coarse operators, smoothing weights, held DOFs) is built
// once on the CPU by VoxelFEA and uploaded; the whole CG loop then runs on the GPU:
//   - matrix-free K*x with one thread per grid node, gathering from its 8 surrounding voxels
//     (no atomics needed),
//   - V-cycle: damped Jacobi smoothing, full-weighting restriction, trilinear prolongation,
//     and a dense inverse on the (small) coarsest level,
//   - dot products by workgroup reduction, with the CG scalars kept on the GPU.
// The GPU computes in 32-bit floats, so the GPU solve is used as the preconditioner of an
// outer flexible CG in 64 bits on the CPU, which reaches the same accuracy as solver.js.
import { HEX_NODES } from './hex8.js';

const WG = 64;
const RED_GROUPS = 1024;
const SMOOTH_SWEEPS = 2; // same as solver.js
export const GPU_COARSEST_DOF = 300;

// scalar slots on the GPU
const RZ = 0, PQ = 1, RR = 4, RZN = 5, BB = 6;

const SHADER = /* wgsl */ `
struct Level {
  nx: u32, ny: u32, nz: u32, NX: u32,
  NY: u32, NZ: u32, nNodes: u32, nDof: u32,
  sharedK: u32, strideN: u32, strideD: u32, m: u32,
  omega: f32, p1: f32, p2: f32, p3: f32,
  CNX: u32, CNY: u32, CNZ: u32, cNodes: u32,
  cStrideN: u32, hasDiag: u32, p5: u32, p6: u32,
};
struct Red { slot: u32, count: u32, q0: u32, q1: u32 };

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> vin: array<f32>;
@group(0) @binding(2) var<storage, read_write> vout: array<f32>;
@group(0) @binding(3) var<storage, read> invD: array<f32>;
@group(0) @binding(4) var<storage, read> emap: array<i32>;
@group(0) @binding(5) var<storage, read> edata: array<f32>;
@group(0) @binding(6) var<storage, read> K0: array<f32>;
@group(0) @binding(7) var<storage, read> rhs: array<f32>;
@group(0) @binding(8) var<storage, read_write> vout2: array<f32>;
@group(0) @binding(9) var<storage, read> vin2: array<f32>;
@group(0) @binding(10) var<storage, read_write> S: array<f32>;
@group(0) @binding(11) var<storage, read_write> partials: array<f32>;
@group(0) @binding(12) var<uniform> R: Red;
@group(0) @binding(13) var<storage, read> dadd: array<f32>;

var<private> OFF: array<vec3<u32>, 8> = array<vec3<u32>, 8>(
  ${HEX_NODES.map(([x, y, z]) => `vec3<u32>(${x}u, ${y}u, ${z}u)`).join(', ')});

fn corner(x: u32, y: u32, z: u32) -> u32 {
  let c = select(select(0u, 1u, x == 1u), select(3u, 2u, x == 1u), y == 1u);
  return c + 4u * z;
}

@compute @workgroup_size(${WG})
fn matvec(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * L.strideN;
  if (n >= L.nNodes) { return; }
  let NX = L.NX; let NY = L.NY;
  let i = n % NX; let j = (n / NX) % NY; let k = n / (NX * NY);
  let free = vec3<f32>(invD[3u * n], invD[3u * n + 1u], invD[3u * n + 2u]);
  var acc = vec3<f32>(0.0);
  if (any(free != vec3<f32>(0.0))) {
    for (var dk = 0u; dk < 2u; dk++) {
      if (k + dk < 1u || k + dk > L.nz) { continue; }
      let ek = k + dk - 1u;
      for (var dj = 0u; dj < 2u; dj++) {
        if (j + dj < 1u || j + dj > L.ny) { continue; }
        let ej = j + dj - 1u;
        for (var di = 0u; di < 2u; di++) {
          if (i + di < 1u || i + di > L.nx) { continue; }
          let ei = i + di - 1u;
          let e = emap[ei + L.nx * (ej + L.ny * ek)];
          if (e < 0) { continue; }
          let a = corner(1u - di, 1u - dj, 1u - dk);
          let nb = ei + NX * (ej + NY * ek);
          var ue: array<f32, 24>;
          for (var c = 0u; c < 8u; c++) {
            let o = OFF[c];
            let m = 3u * (nb + o.x + NX * (o.y + NY * o.z));
            ue[3u * c] = vin[m];
            ue[3u * c + 1u] = vin[m + 1u];
            ue[3u * c + 2u] = vin[m + 2u];
          }
          if (L.sharedK == 1u) {
            let s = edata[u32(e)];
            for (var d = 0u; d < 3u; d++) {
              let row = (3u * a + d) * 24u;
              var sum = 0.0;
              for (var c = 0u; c < 24u; c++) { sum += K0[row + c] * ue[c]; }
              acc[d] += s * sum;
            }
          } else {
            let base = u32(e) * 576u;
            for (var d = 0u; d < 3u; d++) {
              let row = base + (3u * a + d) * 24u;
              var sum = 0.0;
              for (var c = 0u; c < 24u; c++) { sum += edata[row + c] * ue[c]; }
              acc[d] += sum;
            }
          }
        }
      }
    }
  }
  if (L.hasDiag == 1u) {
    for (var d = 0u; d < 3u; d++) { acc[d] += dadd[3u * n + d] * vin[3u * n + d]; }
  }
  for (var d = 0u; d < 3u; d++) { vout[3u * n + d] = select(0.0, acc[d], free[d] != 0.0); }
}

@compute @workgroup_size(${WG})
fn jacobi_first(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = L.omega * invD[q] * rhs[q];
}

@compute @workgroup_size(${WG})
fn jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = vout[q] + L.omega * invD[q] * (rhs[q] - vin[q]);
}

@compute @workgroup_size(${WG})
fn resid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = select(0.0, rhs[q] - vout[q], invD[q] != 0.0);
}

// coarse r = P^T fine t, one thread per coarse node (invD is the coarse level's)
@compute @workgroup_size(${WG})
fn restrict_(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cn = gid.x + gid.y * L.cStrideN;
  if (cn >= L.cNodes) { return; }
  let I = cn % L.CNX; let J = (cn / L.CNX) % L.CNY; let K = cn / (L.CNX * L.CNY);
  var s = vec3<f32>(0.0);
  for (var dk = -1; dk <= 1; dk++) {
    let kf = i32(2u * K) + dk;
    if (kf < 0 || kf >= i32(L.NZ)) { continue; }
    let wk = select(0.5, 1.0, dk == 0);
    for (var dj = -1; dj <= 1; dj++) {
      let jf = i32(2u * J) + dj;
      if (jf < 0 || jf >= i32(L.NY)) { continue; }
      let wj = select(0.5, 1.0, dj == 0);
      for (var di = -1; di <= 1; di++) {
        let if_ = i32(2u * I) + di;
        if (if_ < 0 || if_ >= i32(L.NX)) { continue; }
        let w = wk * wj * select(0.5, 1.0, di == 0);
        let fn_ = 3u * (u32(if_) + L.NX * (u32(jf) + L.NY * u32(kf)));
        s += w * vec3<f32>(vin[fn_], vin[fn_ + 1u], vin[fn_ + 2u]);
      }
    }
  }
  for (var d = 0u; d < 3u; d++) { vout[3u * cn + d] = select(0.0, s[d], invD[3u * cn + d] != 0.0); }
}

// fine z += P coarse z, one thread per fine node (invD is the fine level's)
@compute @workgroup_size(${WG})
fn prolong(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * L.strideN;
  if (n >= L.nNodes) { return; }
  let i = n % L.NX; let j = (n / L.NX) % L.NY; let k = n / (L.NX * L.NY);
  let io = i % 2u; let jo = j % 2u; let ko = k % 2u;
  var s = vec3<f32>(0.0);
  for (var kk = 0u; kk <= ko; kk++) {
    let K = k / 2u + kk;
    let wk = select(1.0, 0.5, ko == 1u);
    for (var jj = 0u; jj <= jo; jj++) {
      let J = j / 2u + jj;
      let wj = select(1.0, 0.5, jo == 1u);
      for (var ii = 0u; ii <= io; ii++) {
        let I = i / 2u + ii;
        let w = wk * wj * select(1.0, 0.5, io == 1u);
        let cn = 3u * (I + L.CNX * (J + L.CNY * K));
        s += w * vec3<f32>(vin[cn], vin[cn + 1u], vin[cn + 2u]);
      }
    }
  }
  for (var d = 0u; d < 3u; d++) {
    let q = 3u * n + d;
    if (invD[q] != 0.0) { vout[q] = vout[q] + s[d]; }
  }
}

// coarsest level: z = A^-1 r over the free DOFs (emap = free DOF list, edata = dense inverse)
@compute @workgroup_size(${WG})
fn coarsest(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x + gid.y * L.strideD;
  if (row >= L.m) { return; }
  var s = 0.0;
  for (var c = 0u; c < L.m; c++) { s += edata[row * L.m + c] * vin[u32(emap[c])]; }
  vout[u32(emap[row])] = s;
}

@compute @workgroup_size(${WG})
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = vin[q];
}

@compute @workgroup_size(${WG})
fn init_r(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = select(0.0, rhs[q] - vin[q], invD[q] != 0.0);
}

// x += alpha p ; r -= alpha q
@compute @workgroup_size(${WG})
fn update_xr(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  let a = S[2];
  vout[q] = vout[q] + a * vin[q];
  vout2[q] = vout2[q] - a * vin2[q];
}

// p = z + beta p
@compute @workgroup_size(${WG})
fn update_p(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = vin[q] + S[3] * vout[q];
}

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn dot_partial(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>,
               @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let stride = nwg.x * 256u;
  var s = 0.0;
  for (var q = gid.x; q < L.nDof; q += stride) { s += vin[q] * vin2[q]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var o = 128u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { partials[wid.x] = red[0]; }
}

@compute @workgroup_size(256)
fn reduce(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  for (var q = lid.x; q < R.count; q += 256u) { s += partials[q]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var o = 128u; o > 0u; o >>= 1u) {
    if (lid.x < o) { red[lid.x] += red[lid.x + o]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { S[R.slot] = red[0]; }
}

@compute @workgroup_size(1)
fn cg_alpha() { S[2] = select(0.0, S[0] / S[1], S[1] > 0.0); }

@compute @workgroup_size(1)
fn cg_beta() {
  S[3] = select(0.0, S[5] / S[0], S[0] != 0.0);
  S[0] = S[5];
}
`;

let devicePromise = null;

/** A WebGPU device with the adapter's full buffer limits, or null when WebGPU is unavailable. */
export function gpuDevice() {
  devicePromise ??= (async () => {
    try {
      if (typeof navigator === 'undefined' || !navigator.gpu) return null;
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return null;
      const device = await adapter.requestDevice({
        requiredLimits: {
          maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
          maxBufferSize: adapter.limits.maxBufferSize,
        },
      });
      device.lost.then(() => { devicePromise = null; });
      return device;
    } catch {
      return null;
    }
  })();
  return devicePromise;
}

function dispatchSize(count, device) {
  const groups = Math.max(1, Math.ceil(count / WG));
  const x = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension);
  return [x, Math.ceil(groups / x), x * WG];
}

/** Dense inverse of the coarsest level's free-DOF matrix from its Cholesky factor (row-major lower L). */
function coarseInverse(coarse) {
  const { m, A: Lf } = coarse;
  const inv = new Float32Array(m * m);
  const y = new Float64Array(m);
  for (let col = 0; col < m; col++) {
    y.fill(0);
    y[col] = 1;
    for (let i = 0; i < m; i++) {
      const ri = i * m;
      let s = y[i];
      for (let k = 0; k < i; k++) s -= Lf[ri + k] * y[k];
      y[i] = s / Lf[ri + i];
    }
    for (let i = m - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < m; k++) s -= Lf[k * m + i] * y[k];
      y[i] = s / Lf[i * m + i];
    }
    for (let i = 0; i < m; i++) inv[i * m + col] = y[i];
  }
  return inv;
}

export class GPUFEASolver {
  /** Relative accuracy asked of each float32 GPU solve inside the float64 outer iteration. */
  static innerTol = 0.1;

  /** @param {import('./solver.js').VoxelFEA} fea built with coarsestMaxDof <= GPU_COARSEST_DOF */
  static async create(fea) {
    const device = await gpuDevice();
    if (!device) throw new Error('WebGPU is not available.');
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    let solver;
    try {
      solver = new GPUFEASolver(device, fea);
    } finally {
      const oom = await device.popErrorScope();
      const invalid = await device.popErrorScope();
      if (oom || invalid) {
        solver?.destroy();
        throw new Error((oom || invalid).message);
      }
    }
    return solver;
  }

  constructor(device, fea) {
    this.device = device;
    this.fea = fea;
    const levels = fea.levels;
    const coarse = fea.coarse;
    if (!coarse.A || coarse.m === 0) throw new Error('The coarsest level has no direct solver.');
    this.buffers = [];
    const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, C = GPUBufferUsage.COPY_SRC;
    const make = (size, usage = S | D | C) => {
      const b = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
      this.buffers.push(b);
      return b;
    };
    const upload = (arr, usage) => {
      const b = make(arr.byteLength, usage);
      device.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
      return b;
    };
    this.module = device.createShaderModule({ code: SHADER });
    const pipe = (entryPoint) => device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint } });
    this.p = Object.fromEntries(
      ['matvec', 'jacobi_first', 'jacobi', 'resid', 'restrict_', 'prolong', 'coarsest', 'copy', 'init_r', 'update_xr', 'update_p', 'dot_partial', 'reduce', 'cg_alpha', 'cg_beta']
        .map((e) => [e, pipe(e)]),
    );
    const K0 = upload(Float32Array.from(fea.K0));

    this.L = levels.map((lv, l) => {
      const nx = lv.nx, ny = lv.ny, nz = lv.nz;
      const emap = new Int32Array(nx * ny * nz).fill(-1);
      for (let q = 0; q < lv.elems.length; q++) emap[lv.elems[q]] = q;
      const invD = new Float32Array(lv.nDof);
      for (let i = 0; i < lv.nDof; i++) invD[i] = lv.fixed[i] ? 0 : lv.invDiag[i];
      const last = l === levels.length - 1;
      const next = levels[l + 1];
      const [, , strideN] = dispatchSize(lv.nNodes, device);
      const [, , strideD] = dispatchSize(lv.nDof, device);
      const [, , cStrideN] = next ? dispatchSize(next.nNodes, device) : [0, 0, 0];
      const params = new ArrayBuffer(96);
      const u = new Uint32Array(params), f = new Float32Array(params);
      u.set([nx, ny, nz, lv.NX, lv.NY, lv.NZ, lv.nNodes, lv.nDof, lv.K ? 0 : 1, strideN, strideD, last ? coarse.m : 0]);
      f[12] = lv.omega;
      if (next) u.set([next.NX, next.NY, next.NZ, next.nNodes, cStrideN], 16);
      u[21] = lv.diagAdd ? 1 : 0;
      const L = {
        lv, last,
        params: upload(new Uint8Array(params), GPUBufferUsage.UNIFORM | D),
        invD: upload(invD),
        emap: upload(emap),
        edata: upload(lv.K ? Float32Array.from(lv.K) : Float32Array.from(lv.rho)),
        dadd: upload(lv.diagAdd ? Float32Array.from(lv.diagAdd) : new Float32Array(4)),
        r: make(4 * lv.nDof), z: make(4 * lv.nDof), t: make(4 * lv.nDof),
      };
      if (last) {
        const free = new Int32Array(coarse.m);
        for (let i = 0; i < lv.nDof; i++) if (coarse.map[i] >= 0) free[coarse.map[i]] = i;
        L.free = upload(free);
        L.ainv = upload(coarseInverse(coarse));
      }
      return L;
    });
    const L0 = this.L[0];
    const n = levels[0].nDof;
    this.n = n;
    this.x = make(4 * n);
    this.b = make(4 * n);
    this.pv = make(4 * n);
    this.qv = make(4 * n);
    this.S = make(64);
    this.partials = make(4 * RED_GROUPS);
    // reduction slots, one 256-byte-aligned entry per scalar
    const red = new Uint32Array(64 * 8);
    for (let s = 0; s < 8; s++) red.set([s, 0], s * 64);
    this.red = upload(red, GPUBufferUsage.UNIFORM | D);
    this.readBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.MAP_READ | D });
    this.xRead = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.MAP_READ | D });
    this.buffers.push(this.readBuf, this.xRead);

    // bind groups
    const bg = (pipeline, entries) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: Object.entries(entries).map(([binding, res]) =>
        ({ binding: Number(binding), resource: res instanceof GPUBuffer ? { buffer: res } : res })),
    });
    const P = this.p;
    this.L.forEach((L, l) => {
      const N = this.L[l + 1];
      L.bg = {
        mvZT: bg(P.matvec, { 0: L.params, 1: L.z, 2: L.t, 3: L.invD, 4: L.emap, 5: L.edata, 6: K0, 13: L.dadd }),
        smoothFirst: bg(P.jacobi_first, { 0: L.params, 2: L.z, 3: L.invD, 7: L.r }),
        smooth: bg(P.jacobi, { 0: L.params, 1: L.t, 2: L.z, 3: L.invD, 7: L.r }),
        resid: bg(P.resid, { 0: L.params, 2: L.t, 3: L.invD, 7: L.r }),
      };
      if (N) {
        L.bg.restrict = bg(P.restrict_, { 0: L.params, 1: L.t, 2: N.r, 3: N.invD });
        L.bg.prolong = bg(P.prolong, { 0: L.params, 1: N.z, 2: L.z, 3: L.invD });
      }
      if (L.last) L.bg.coarsest = bg(P.coarsest, { 0: L.params, 1: L.r, 2: L.z, 4: L.free, 5: L.ainv });
    });
    this.bg = {
      mvP: bg(P.matvec, { 0: L0.params, 1: this.pv, 2: this.qv, 3: L0.invD, 4: L0.emap, 5: L0.edata, 6: K0, 13: L0.dadd }),
      mvX: bg(P.matvec, { 0: L0.params, 1: this.x, 2: this.qv, 3: L0.invD, 4: L0.emap, 5: L0.edata, 6: K0, 13: L0.dadd }),
      initR: bg(P.init_r, { 0: L0.params, 1: this.qv, 2: L0.r, 3: L0.invD, 7: this.b }),
      copyZP: bg(P.copy, { 0: L0.params, 1: L0.z, 2: this.pv }),
      updXR: bg(P.update_xr, { 0: L0.params, 1: this.pv, 2: this.x, 8: L0.r, 9: this.qv, 10: this.S }),
      updP: bg(P.update_p, { 0: L0.params, 1: L0.z, 2: this.pv, 10: this.S }),
      alpha: bg(P.cg_alpha, { 10: this.S }),
      beta: bg(P.cg_beta, { 10: this.S }),
      // one bind group per scalar slot (auto layouts cannot take dynamic offsets)
      reduce: Array.from({ length: 8 }, (_, slot) => bg(P.reduce, { 10: this.S, 11: this.partials, 12: { buffer: this.red, offset: slot * 256, size: 16 } })),
    };
    this.dotBG = new Map();
    this.dotGroups = Math.min(RED_GROUPS, Math.ceil(n / 256));
    const counts = new Uint32Array(64 * 8);
    for (let s = 0; s < 8; s++) counts.set([s, this.dotGroups], s * 64);
    device.queue.writeBuffer(this.red, 0, counts);
  }

  dotBindGroup(a, b) {
    const key = `${this.buffers.indexOf(a)}:${this.buffers.indexOf(b)}`;
    if (!this.dotBG.has(key)) {
      this.dotBG.set(key, this.device.createBindGroup({
        layout: this.p.dot_partial.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.L[0].params } },
          { binding: 1, resource: { buffer: a } },
          { binding: 9, resource: { buffer: b } },
          { binding: 11, resource: { buffer: this.partials } },
        ],
      }));
    }
    return this.dotBG.get(key);
  }

  run(pass, pipeline, group, count) {
    const [x, y] = dispatchSize(count, this.device);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y);
  }

  dot(pass, a, b, slot) {
    pass.setPipeline(this.p.dot_partial);
    pass.setBindGroup(0, this.dotBindGroup(a, b));
    pass.dispatchWorkgroups(this.dotGroups);
    pass.setPipeline(this.p.reduce);
    pass.setBindGroup(0, this.bg.reduce[slot]);
    pass.dispatchWorkgroups(1);
  }

  vcycle(pass, l) {
    const L = this.L[l], lv = L.lv;
    if (L.last) {
      this.run(pass, this.p.coarsest, L.bg.coarsest, this.fea.coarse.m);
      return;
    }
    const nD = lv.nDof, nN = lv.nNodes;
    this.run(pass, this.p.jacobi_first, L.bg.smoothFirst, nD);
    for (let s = 1; s < SMOOTH_SWEEPS; s++) {
      this.run(pass, this.p.matvec, L.bg.mvZT, nN);
      this.run(pass, this.p.jacobi, L.bg.smooth, nD);
    }
    this.run(pass, this.p.matvec, L.bg.mvZT, nN);
    this.run(pass, this.p.resid, L.bg.resid, nD);
    this.run(pass, this.p.restrict_, L.bg.restrict, this.L[l + 1].lv.nNodes);
    this.vcycle(pass, l + 1);
    this.run(pass, this.p.prolong, L.bg.prolong, nN);
    for (let s = 0; s < SMOOTH_SWEEPS; s++) {
      this.run(pass, this.p.matvec, L.bg.mvZT, nN);
      this.run(pass, this.p.jacobi, L.bg.smooth, nD);
    }
  }

  async readScalars() {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.S, 0, this.readBuf, 0, 64);
    this.device.queue.submit([enc.finish()]);
    await this.readBuf.mapAsync(GPUMapMode.READ);
    const s = new Float32Array(this.readBuf.getMappedRange().slice(0));
    this.readBuf.unmap();
    return s;
  }

  /** Preconditioned CG on the GPU from x = 0 for K x = rhs (float32). */
  async pcg(rhs, { tol, maxIter, onProgress }) {
    const dev = this.device, L0 = this.L[0], lv = L0.lv, n = this.n;
    const b32 = new Float32Array(n);
    for (let i = 0; i < n; i++) b32[i] = lv.fixed[i] ? 0 : rhs[i];
    dev.queue.writeBuffer(this.b, 0, b32);
    dev.queue.writeBuffer(this.x, 0, new Float32Array(n));
    let enc = dev.createCommandEncoder();
    let pass = enc.beginComputePass();
    this.run(pass, this.p.matvec, this.bg.mvX, lv.nNodes);
    this.run(pass, this.p.init_r, this.bg.initR, n);
    this.vcycle(pass, 0);
    this.run(pass, this.p.copy, this.bg.copyZP, n);
    this.dot(pass, L0.r, L0.z, RZ);
    this.dot(pass, this.b, this.b, BB);
    this.dot(pass, L0.r, L0.r, RR);
    pass.end();
    dev.queue.submit([enc.finish()]);
    let s = await this.readScalars();
    const bnorm = Math.sqrt(s[BB]);
    if (!(bnorm > 0)) return { x: new Float64Array(n), iterations: 0, residual: 0, broke: false };
    let res = Math.sqrt(s[RR]) / bnorm;
    let it = 0, broke = false;
    const CHECK = 6;
    while (it < maxIter && res > tol) {
      enc = dev.createCommandEncoder();
      pass = enc.beginComputePass();
      const batch = Math.min(CHECK, maxIter - it);
      for (let k = 0; k < batch; k++) {
        this.run(pass, this.p.matvec, this.bg.mvP, lv.nNodes);
        this.dot(pass, this.pv, this.qv, PQ);
        this.run(pass, this.p.cg_alpha, this.bg.alpha, 1);
        this.run(pass, this.p.update_xr, this.bg.updXR, n);
        this.vcycle(pass, 0);
        this.dot(pass, L0.r, L0.z, RZN);
        this.run(pass, this.p.cg_beta, this.bg.beta, 1);
        this.run(pass, this.p.update_p, this.bg.updP, n);
      }
      this.dot(pass, L0.r, L0.r, RR);
      pass.end();
      dev.queue.submit([enc.finish()]);
      it += batch;
      s = await this.readScalars();
      if (!(s[PQ] > 0) || !Number.isFinite(s[RR])) { broke = true; break; }
      res = Math.sqrt(s[RR]) / bnorm;
      onProgress?.(it, res);
    }
    enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.x, 0, this.xRead, 0, 4 * n);
    dev.queue.submit([enc.finish()]);
    await this.xRead.mapAsync(GPUMapMode.READ);
    const x32 = new Float32Array(this.xRead.getMappedRange().slice(0));
    this.xRead.unmap();
    return { x: Float64Array.from(x32), iterations: it, residual: res, broke };
  }

  /**
   * Solve K u = f (normalized units, like VoxelFEA.solve).
   * Outer loop: flexible conjugate gradients in float64 on the CPU (one exact K*p per step).
   * Preconditioner: a float32 multigrid-PCG solve on the GPU. Because the outer iteration only
   * asks the GPU for an approximate correction, float32 round-off - which grows with the
   * stiffness condition number, large for thin walls - never limits the final accuracy.
   */
  async solve(f, { tol = 1e-6, maxIter = 500, x0 = null, onProgress = null } = {}) {
    const fea = this.fea, L = fea.levels[0], n = L.nDof, fixed = L.fixed;
    const u = x0 ? Float64Array.from(x0) : new Float64Array(n);
    for (let i = 0; i < n; i++) if (fixed[i]) u[i] = 0;
    let bnorm = 0;
    for (let i = 0; i < n; i++) if (!fixed[i]) bnorm += f[i] * f[i];
    bnorm = Math.sqrt(bnorm);
    if (bnorm === 0) return { u, iterations: 0, residual: 0, converged: true, engine: 'GPU' };
    const dot = (a, c) => { let t = 0; for (let i = 0; i < n; i++) t += a[i] * c[i]; return t; };
    const q = new Float64Array(n), r = new Float64Array(n), rOld = new Float64Array(n), p = new Float64Array(n);
    fea.apply(L, u, q);
    for (let i = 0; i < n; i++) r[i] = fixed[i] ? 0 : f[i] - q[i];
    let rel = Math.sqrt(dot(r, r)) / bnorm;
    let its = 0;
    const precondition = async (res) => {
      const inner = await this.pcg(res, {
        tol: GPUFEASolver.innerTol,
        maxIter: Math.max(1, Math.min(60, maxIter - its)),
        onProgress: (it, rr) => onProgress?.(its + it, Math.min(rel, rr * rel)),
      });
      its += Math.max(1, inner.iterations);
      return inner.x;
    };
    if (rel <= tol) return { u, iterations: 0, residual: rel, converged: true, engine: 'GPU' };
    let z = await precondition(r);
    p.set(z);
    let rz = dot(r, z);
    for (let outer = 0; outer < 40 && its < maxIter; outer++) {
      fea.apply(L, p, q);
      const pq = dot(p, q);
      if (!(pq > 0)) break;
      const alpha = rz / pq;
      rOld.set(r);
      for (let i = 0; i < n; i++) { u[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
      rel = Math.sqrt(dot(r, r)) / bnorm;
      onProgress?.(its, rel);
      if (rel <= tol) break;
      const zNew = await precondition(r);
      // flexible (Polak-Ribiere) beta: the preconditioner changes from step to step
      let num = 0;
      for (let i = 0; i < n; i++) num += zNew[i] * (r[i] - rOld[i]);
      const beta = Math.max(0, num / rz);
      rz = dot(r, zNew);
      for (let i = 0; i < n; i++) p[i] = zNew[i] + beta * p[i];
      z = zNew;
    }
    return { u, iterations: its, residual: rel, converged: rel <= tol * 10, engine: 'GPU' };
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
  }
}
