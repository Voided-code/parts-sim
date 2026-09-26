// Explicit drop-test dynamics (from src/fea/gpu-explicit.js; workgroup size 64)

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
  vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u), vec3<u32>(1u, 1u, 0u), vec3<u32>(0u, 1u, 0u), vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u), vec3<u32>(1u, 1u, 1u), vec3<u32>(0u, 1u, 1u));

fn corner(x: u32, y: u32, z: u32) -> u32 {
  let c = select(select(0u, 1u, x == 1u), select(3u, 2u, x == 1u), y == 1u);
  return c + 4u * z;
}

@compute @workgroup_size(64)
fn advance(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x + gid.y * P.strideD;
  if (i >= P.nDof) { return; }
  let ui = u[i] + P.dt * v[i];
  u[i] = ui;
  w[i] = ui + P.beta * v[i];
}

@compute @workgroup_size(64)
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
@compute @workgroup_size(64)
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

@compute @workgroup_size(64)
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
