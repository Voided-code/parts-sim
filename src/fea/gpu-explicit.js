// WebGPU version of the drop test (explicit.js): the whole time loop runs on the GPU.
//
// Per time step: u += dt v and w = u + beta v; q = K w (one thread per node gathering from its
// 8 voxels, as in the static solver); then each node's acceleration from -E h q / m, gravity and
// penalty contact with the floor updates v. Every 4th step the nodal von Mises stress is
// recovered from the voxel corners, its running maximum kept, and the total floor force summed.
// The CPU only reads back the force history (to see when the part leaves the floor) and the
// occasional animation frame.
import { gpuDevice } from './gpu-solver.js';
import { HEX_NODES } from './hex8.js';
import { lumpedMass } from './eigen.js';
import { maxEigenvalue } from './explicit.js';

const WG = 64;
const SAMPLE = 4;

const SHADER = /* wgsl */ `
struct Params {
  nx: u32, ny: u32, nz: u32, NX: u32,
  NY: u32, NZ: u32, nNodes: u32, strideN: u32,
  dt: f32, beta: f32, Eh: f32, g: f32,
  nPart: u32, strideD: u32, nDof: u32, p3: u32,
};
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> emap: array<i32>;
@group(0) @binding(2) var<storage, read> rho: array<f32>;
@group(0) @binding(3) var<storage, read> K0: array<f32>;
@group(0) @binding(4) var<storage, read_write> u: array<f32>;
@group(0) @binding(5) var<storage, read_write> v: array<f32>;
@group(0) @binding(6) var<storage, read_write> w: array<f32>;
@group(0) @binding(7) var<storage, read_write> q: array<f32>;
@group(0) @binding(8) var<storage, read> invM: array<f32>;
@group(0) @binding(9) var<storage, read> contact: array<f32>;
@group(0) @binding(10) var<storage, read_write> fc: array<f32>;
@group(0) @binding(11) var<storage, read> S: array<f32>;
@group(0) @binding(12) var<storage, read_write> vmNow: array<f32>;
@group(0) @binding(13) var<storage, read_write> vmMax: array<f32>;
@group(0) @binding(14) var<storage, read_write> tPeak: array<f32>;
@group(0) @binding(15) var<storage, read_write> clock: array<f32>;
@group(0) @binding(16) var<storage, read_write> partial: array<f32>;
@group(0) @binding(17) var<storage, read_write> hist: array<f32>;

var<private> OFF: array<vec3<u32>, 8> = array<vec3<u32>, 8>(
  ${HEX_NODES.map(([x, y, z]) => `vec3<u32>(${x}u, ${y}u, ${z}u)`).join(', ')});

fn corner(x: u32, y: u32, z: u32) -> u32 {
  let c = select(select(0u, 1u, x == 1u), select(3u, 2u, x == 1u), y == 1u);
  return c + 4u * z;
}

@compute @workgroup_size(${WG})
fn advance(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * P.strideD;
  if (i >= P.nDof) { return; }
  let ui = u[i] + P.dt * v[i];
  u[i] = ui;
  w[i] = ui + P.beta * v[i];
}

@compute @workgroup_size(${WG})
fn matvec(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * P.strideN;
  if (n >= P.nNodes) { return; }
  let NX = P.NX; let NY = P.NY;
  let i = n % NX; let j = (n / NX) % NY; let k = n / (NX * NY);
  var acc = vec3<f32>(0.0);
  for (var dk = 0u; dk < 2u; dk++) {
    if (k + dk < 1u || k + dk > P.nz) { continue; }
    let ek = k + dk - 1u;
    for (var dj = 0u; dj < 2u; dj++) {
      if (j + dj < 1u || j + dj > P.ny) { continue; }
      let ej = j + dj - 1u;
      for (var di = 0u; di < 2u; di++) {
        if (i + di < 1u || i + di > P.nx) { continue; }
        let ei = i + di - 1u;
        let e = emap[ei + P.nx * (ej + P.ny * ek)];
        if (e < 0) { continue; }
        let a = corner(1u - di, 1u - dj, 1u - dk);
        let nb = ei + NX * (ej + NY * ek);
        var ue: array<f32, 24>;
        for (var c = 0u; c < 8u; c++) {
          let o = OFF[c];
          let m = 3u * (nb + o.x + NX * (o.y + NY * o.z));
          ue[3u * c] = w[m]; ue[3u * c + 1u] = w[m + 1u]; ue[3u * c + 2u] = w[m + 2u];
        }
        let s = rho[u32(e)];
        for (var d = 0u; d < 3u; d++) {
          let row = (3u * a + d) * 24u;
          var sum = 0.0;
          for (var c = 0u; c < 24u; c++) { sum += K0[row + c] * ue[c]; }
          acc[d] += s * sum;
        }
      }
    }
  }
  q[3u * n] = acc.x; q[3u * n + 1u] = acc.y; q[3u * n + 2u] = acc.z;
}

// kick: v += dt * a (dt is P.dt, or half of it for the start-up half step via a second params buffer)
@compute @workgroup_size(${WG})
fn update(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * P.strideN;
  if (n >= P.nNodes) { return; }
  var force = 0.0;
  for (var d = 0u; d < 3u; d++) {
    let i = 3u * n + d;
    let im = invM[i];
    if (im == 0.0) { continue; }
    var a = -P.Eh * q[i] * im;
    if (d == 1u) {
      a -= P.g;
      let kc = contact[3u * n];
      if (kc > 0.0) {
        let pen = -(contact[3u * n + 2u] + u[i]);
        if (pen > 0.0) {
          let f = max(0.0, kc * pen - contact[3u * n + 1u] * v[i]);
          a += f * im;
          force = f;
        }
      }
    }
    v[i] = v[i] + P.dt * a;
  }
  fc[n] = force;
}

@compute @workgroup_size(${WG})
fn stress(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * P.strideN;
  if (n >= P.nNodes) { return; }
  let NX = P.NX; let NY = P.NY;
  let i = n % NX; let j = (n / NX) % NY; let k = n / (NX * NY);
  var acc = 0.0;
  var wsum = 0.0;
  for (var dk = 0u; dk < 2u; dk++) {
    if (k + dk < 1u || k + dk > P.nz) { continue; }
    let ek = k + dk - 1u;
    for (var dj = 0u; dj < 2u; dj++) {
      if (j + dj < 1u || j + dj > P.ny) { continue; }
      let ej = j + dj - 1u;
      for (var di = 0u; di < 2u; di++) {
        if (i + di < 1u || i + di > P.nx) { continue; }
        let ei = i + di - 1u;
        let e = emap[ei + P.nx * (ej + P.ny * ek)];
        if (e < 0) { continue; }
        let a = corner(1u - di, 1u - dj, 1u - dk);
        let nb = ei + NX * (ej + NY * ek);
        var ue: array<f32, 24>;
        for (var c = 0u; c < 8u; c++) {
          let o = OFF[c];
          let m = 3u * (nb + o.x + NX * (o.y + NY * o.z));
          ue[3u * c] = u[m]; ue[3u * c + 1u] = u[m + 1u]; ue[3u * c + 2u] = u[m + 2u];
        }
        var sg: array<f32, 6>;
        for (var r = 0u; r < 6u; r++) {
          var sum = 0.0;
          let row = a * 144u + r * 24u;
          for (var c = 0u; c < 24u; c++) { sum += S[row + c] * ue[c]; }
          sg[r] = sum;
        }
        let vm = sqrt(0.5 * ((sg[0] - sg[1]) * (sg[0] - sg[1]) + (sg[1] - sg[2]) * (sg[1] - sg[2]) + (sg[2] - sg[0]) * (sg[2] - sg[0]))
          + 3.0 * (sg[3] * sg[3] + sg[4] * sg[4] + sg[5] * sg[5]));
        let r = rho[u32(e)];
        acc += r * vm;
        wsum += r;
      }
    }
  }
  var now = 0.0;
  if (wsum > 0.0) { now = acc / wsum; }
  vmNow[n] = now;
  if (now > vmMax[n]) { vmMax[n] = now; tPeak[n] = clock[0]; }
}

var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_a(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  var s = 0.0;
  var idx = wid.x * 256u + lid.x;
  loop {
    if (idx >= P.nNodes) { break; }
    s += fc[idx];
    idx += P.nPart * 256u;
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] += red[lid.x + st]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { partial[wid.x] = red[0]; }
}

@compute @workgroup_size(256)
fn reduce_b(@builtin(local_invocation_id) lid: vec3<u32>) {
  var s = 0.0;
  for (var t = lid.x; t < P.nPart; t += 256u) { s += partial[t]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var st = 128u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] += red[lid.x + st]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) {
    let k = u32(clock[1]);
    hist[k] = red[0];
    clock[1] = clock[1] + 1.0;
  }
}
`;

function dispatchSize(count, device) {
  const groups = Math.max(1, Math.ceil(count / WG));
  const x = Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension);
  return [x, Math.ceil(groups / x), x * WG];
}

export class GPUExplicit {
  /** Run a drop test on the GPU; same options and result as dropTestCPU. */
  static async run(fea, opts) {
    const device = await gpuDevice();
    if (!device) throw new Error('WebGPU is not available.');
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    let sim;
    try {
      sim = new GPUExplicit(device, fea, opts);
    } finally {
      const oom = await device.popErrorScope();
      const invalid = await device.popErrorScope();
      if (oom || invalid) {
        sim?.destroy();
        throw new Error((oom || invalid).message);
      }
    }
    const info = await sim.module.getCompilationInfo?.();
    if (info?.messages?.some((msg) => msg.type === 'error')) {
      sim.destroy();
      throw new Error(`GPU shader failed to compile: ${info.messages.find((msg) => msg.type === 'error').message}`);
    }
    try {
      return await sim.simulate();
    } finally {
      sim.destroy();
    }
  }

  constructor(device, fea, { E, rho, h, speed, nodeY, surface, g = 9.81, frames = 48, maxSteps = 40000, maxTime = Infinity, onFrame = null, onProgress = null }) {
    this.device = device;
    this.fea = fea;
    this.opts = { frames, maxSteps, maxTime, onFrame, onProgress };
    const L = fea.levels[0], n = L.nDof, nN = L.nNodes;
    this.n = n; this.nN = nN;
    const Mn = lumpedMass(fea);
    const lamMax = maxEigenvalue(fea, Mn, 25) * 1.05;
    const wMax = Math.sqrt((lamMax * E) / (rho * h * h));
    const xiHigh = 0.1;
    this.beta = (2 * xiHigh) / wMax;
    this.dt = 0.9 * (2 / wMax) * (Math.sqrt(1 + xiHigh * xiHigh) - xiHigh);
    this.wMax = wMax;
    const invM = new Float32Array(n);
    const contact = new Float32Array(3 * nN);
    for (let i = 0; i < n; i++) invM[i] = Mn[i] > 0 ? 1 / (rho * h * h * h * Mn[i]) : 0;
    for (let q = 0; q < nN; q++) {
      const m = rho * h * h * h * Mn[3 * q + 1];
      contact[3 * q + 2] = nodeY[q];
      if (!surface[q] || !(m > 0)) continue;
      contact[3 * q] = 0.25 * wMax * wMax * m;
      contact[3 * q + 1] = 2 * 0.05 * Math.sqrt(contact[3 * q] * m);
    }
    const v0 = new Float32Array(n);
    for (let i = 1; i < n; i += 3) if (invM[i] > 0) v0[i] = -speed;
    const emap = new Int32Array(L.nx * L.ny * L.nz).fill(-1);
    for (let q = 0; q < L.elems.length; q++) emap[L.elems[q]] = q;
    const Sm = new Float32Array(8 * 144);
    fea.cornerStress.forEach((M, a) => { for (let t = 0; t < 144; t++) Sm[a * 144 + t] = (M[t] * E) / h; });

    this.buffers = [];
    const SU = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, C = GPUBufferUsage.COPY_SRC;
    const make = (size, usage = SU | D | C) => {
      const b = device.createBuffer({ size: Math.max(16, Math.ceil(size / 4) * 4), usage });
      this.buffers.push(b);
      return b;
    };
    const upload = (arr, usage = SU | D | C) => {
      const b = make(arr.byteLength, usage);
      device.queue.writeBuffer(b, 0, arr.buffer, arr.byteOffset, arr.byteLength);
      return b;
    };
    const [, , strideN] = dispatchSize(nN, device);
    const [, , strideD] = dispatchSize(n, device);
    this.nPart = Math.min(1024, Math.ceil(nN / 256));
    const params = (dt) => {
      const buf = new ArrayBuffer(64), u = new Uint32Array(buf), f = new Float32Array(buf);
      u.set([L.nx, L.ny, L.nz, L.NX, L.NY, L.NZ, nN, strideN]);
      f[8] = dt; f[9] = this.beta; f[10] = E * h; f[11] = g;
      u[12] = this.nPart; u[13] = strideD; u[14] = n;
      return upload(new Uint8Array(buf), GPUBufferUsage.UNIFORM | D);
    };
    this.maxSamples = Math.ceil(maxSteps / SAMPLE) + 8;
    const B = {
      P: params(this.dt), Phalf: params(0.5 * this.dt),
      emap: upload(emap), rho: upload(Float32Array.from(L.rho)), K0: upload(Float32Array.from(fea.K0)),
      u: make(4 * n), v: upload(v0), w: make(4 * n), q: make(4 * n),
      invM: upload(invM), contact: upload(contact), fc: make(4 * nN), S: upload(Sm),
      vmNow: make(4 * nN), vmMax: make(4 * nN), tPeak: make(4 * nN), clock: make(16),
      partial: make(4 * this.nPart), hist: make(4 * this.maxSamples),
    };
    this.B = B;
    this.module = device.createShaderModule({ code: SHADER });
    const pipe = (entryPoint) => device.createComputePipeline({ layout: 'auto', compute: { module: this.module, entryPoint } });
    const bg = (pipeline, entries) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: Object.entries(entries).map(([binding, buffer]) => ({ binding: Number(binding), resource: { buffer } })),
    });
    const p = this.p = Object.fromEntries(['advance', 'matvec', 'update', 'stress', 'reduce_a', 'reduce_b'].map((e) => [e, pipe(e)]));
    this.bg = {
      advance: bg(p.advance, { 0: B.P, 4: B.u, 5: B.v, 6: B.w }),
      matvec: bg(p.matvec, { 0: B.P, 1: B.emap, 2: B.rho, 3: B.K0, 6: B.w, 7: B.q }),
      update: bg(p.update, { 0: B.P, 4: B.u, 5: B.v, 7: B.q, 8: B.invM, 9: B.contact, 10: B.fc }),
      updateHalf: bg(p.update, { 0: B.Phalf, 4: B.u, 5: B.v, 7: B.q, 8: B.invM, 9: B.contact, 10: B.fc }),
      stress: bg(p.stress, { 0: B.P, 1: B.emap, 2: B.rho, 4: B.u, 11: B.S, 12: B.vmNow, 13: B.vmMax, 14: B.tPeak, 15: B.clock }),
      reduceA: bg(p.reduce_a, { 0: B.P, 10: B.fc, 16: B.partial }),
      reduceB: bg(p.reduce_b, { 0: B.P, 15: B.clock, 16: B.partial, 17: B.hist }),
    };
    this.histRead = device.createBuffer({ size: 4 * this.maxSamples, usage: GPUBufferUsage.MAP_READ | D });
    this.uRead = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.MAP_READ | D });
    this.vRead = device.createBuffer({ size: 4 * nN, usage: GPUBufferUsage.MAP_READ | D });
    this.buffers.push(this.histRead, this.uRead, this.vRead);
  }

  run1(pass, name, group, count) {
    const [x, y] = dispatchSize(count, this.device);
    pass.setPipeline(this.p[name]);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(x, y);
  }

  async readback(src, dst, bytes, offset = 0) {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(src, offset, dst, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await dst.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(dst.getMappedRange(0, bytes).slice(0));
    dst.unmap();
    return out;
  }

  async simulate() {
    const { device, n, nN, dt } = this;
    const { frames, maxSteps, maxTime, onFrame, onProgress } = this.opts;
    const maxDim = Math.max(this.fea.levels[0].nx, this.fea.levels[0].ny, this.fea.levels[0].nz);
    const batch = Math.max(4, Math.min(64, Math.round(maxDim / 12 / SAMPLE) * SAMPLE || SAMPLE));
    // start-up: a(0) and the first half-step kick
    let enc = device.createCommandEncoder(), pass = enc.beginComputePass();
    this.run1(pass, 'matvec', this.bg.matvec, nN); // w = 0 at t = 0 (u = 0; beta v ignored at the start)
    this.run1(pass, 'update', this.bg.updateHalf, nN);
    pass.end();
    device.queue.submit([enc.finish()]);
    let t = 0, step = 0, samples = 0;
    let contactStarted = false, contactStart = 0, contactEnd = -1, peakForce = 0, estEnd = maxTime, lastFrame = -Infinity;
    const history = [];
    while (step < maxSteps) {
      const count = Math.min(batch, maxSteps - step);
      device.queue.writeBuffer(this.B.clock, 0, Float32Array.of(t + dt, samples));
      enc = device.createCommandEncoder();
      pass = enc.beginComputePass();
      let taken = 0;
      for (let s = 0; s < count; s++) {
        this.run1(pass, 'advance', this.bg.advance, n);
        this.run1(pass, 'matvec', this.bg.matvec, nN);
        this.run1(pass, 'update', this.bg.update, nN);
        taken++;
        if ((step + taken) % SAMPLE === 0) {
          this.run1(pass, 'stress', this.bg.stress, nN);
          pass.setPipeline(this.p.reduce_a); pass.setBindGroup(0, this.bg.reduceA); pass.dispatchWorkgroups(this.nPart);
          pass.setPipeline(this.p.reduce_b); pass.setBindGroup(0, this.bg.reduceB); pass.dispatchWorkgroups(1);
        }
      }
      pass.end();
      device.queue.submit([enc.finish()]);
      const newSamples = Math.floor((step + taken) / SAMPLE) - Math.floor(step / SAMPLE);
      const t0 = t;
      step += taken;
      t += taken * dt;
      let forces = new Float32Array(0);
      if (newSamples > 0) forces = await this.readback(this.B.hist, this.histRead, 4 * newSamples, 4 * samples);
      else await device.queue.onSubmittedWorkDone();
      forces.forEach((fc, k) => {
        const ts = t0 + (SAMPLE * (Math.floor((step - taken) / SAMPLE) + k + 1) - (step - taken)) * dt;
        history.push({ t: ts, force: fc });
        if (fc > 0 && !contactStarted) { contactStarted = true; contactStart = ts; }
        if (fc > peakForce) peakForce = fc;
        if (contactStarted && fc > 0) contactEnd = -1;
        else if (contactStarted && fc === 0 && contactEnd < 0) contactEnd = ts;
      });
      samples += newSamples;
      if (contactEnd > 0 && estEnd === maxTime) estEnd = Math.min(maxTime, contactEnd + Math.max(0.3 * (contactEnd - contactStart), 50 * dt));
      if (contactEnd < 0 && estEnd !== maxTime) estEnd = maxTime;
      const done = t >= estEnd || step >= maxSteps || samples >= this.maxSamples - 2;
      const frameGap = Math.max(dt, (Number.isFinite(estEnd) ? estEnd : Math.max(t, 1e-6) * 2) / frames);
      if (onFrame && (t - lastFrame >= frameGap || done)) {
        lastFrame = t;
        const u = await this.readback(this.B.u, this.uRead, 4 * n);
        const vm = await this.readback(this.B.vmNow, this.vRead, 4 * nN);
        onFrame({ t, u, vm, force: history.length ? history[history.length - 1].force : 0 });
      }
      if (onProgress && onProgress(Math.min(0.99, Number.isFinite(estEnd) ? t / estEnd : step / maxSteps)) === true) {
        throw Object.assign(new Error('Cancelled'), { cancelled: true });
      }
      if (done) break;
    }
    const vmMax = await this.readback(this.B.vmMax, this.vRead, 4 * nN);
    const tPeak = await this.readback(this.B.tPeak, this.vRead, 4 * nN);
    return {
      vmMax, tPeak, history, dt, steps: step, duration: t, peakForce,
      contactTime: contactStarted ? (contactEnd > 0 ? contactEnd : t) - contactStart : 0,
      wMax: this.wMax, rebounded: contactEnd > 0,
    };
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
  }
}
