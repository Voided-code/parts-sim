// Multigrid V-cycle (degree-2 Chebyshev smoothing) and float32 conjugate gradients for the voxel
// FEA on the GPU. Workgroup size 64.
//
// Matrix products gather, for each node, the rows of its 8 surrounding voxels from a 3x3x3 block of
// neighbour values loaded once. Each voxel is one word: 0 = empty, > 0 = the f32 bits of s for an
// s * K0 element, < 0 = sign bit | index of its own 24x24 matrix in Ke (coarse levels). A node whose
// 8 voxels are the same s * K0 uses the assembled 27-point stencil (243 multiply-adds, not 576).

struct Level {
  nx: u32, ny: u32, nz: u32, NX: u32,
  NY: u32, NZ: u32, nNodes: u32, nDof: u32,
  p0: u32, strideN: u32, strideD: u32, m: u32,
  first: f32, c1: f32, c2: f32, p3: f32,     // Chebyshev smoother coefficients
  CNX: u32, CNY: u32, CNZ: u32, cNodes: u32,
  cStrideN: u32, hasDiag: u32, p5: u32, p6: u32,
};
struct Red { slot: u32, count: u32, q0: u32, q1: u32 };

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> vin: array<f32>;
@group(0) @binding(2) var<storage, read_write> vout: array<f32>;
@group(0) @binding(3) var<storage, read> invD: array<f32>;
@group(0) @binding(4) var<storage, read> ew: array<i32>;          // voxel words (coarsest: free DOF list)
@group(0) @binding(5) var<storage, read> Ke: array<vec4<f32>>;    // element matrices, 144 vec4 each
@group(0) @binding(6) var<uniform> K0: array<vec4<f32>, 144>;     // unit element, rows as 6 vec4
@group(0) @binding(7) var<storage, read> rhs: array<f32>;
@group(0) @binding(8) var<storage, read_write> vout2: array<f32>;
@group(0) @binding(9) var<storage, read> vin2: array<f32>;
@group(0) @binding(10) var<storage, read_write> S: array<f32>;
@group(0) @binding(11) var<storage, read_write> partials: array<f32>;
@group(0) @binding(12) var<uniform> R: Red;
@group(0) @binding(13) var<storage, read> dadd: array<f32>;
@group(0) @binding(14) var<uniform> ST: array<vec4<f32>, 81>;     // stencil: neighbour m, row d -> ST[3m + d].xyz
@group(0) @binding(15) var<storage, read> ainv: array<f32>;       // coarsest: dense inverse (symmetric)

fn nodeU(i: i32, j: i32, k: i32) -> vec3<f32> {
  let x = u32(clamp(i, 0, i32(L.NX) - 1));
  let y = u32(clamp(j, 0, i32(L.NY) - 1));
  let z = u32(clamp(k, 0, i32(L.NZ) - 1));
  let m = 3u * (x + L.NX * (y + L.NY * z));
  return vec3<f32>(vin[m], vin[m + 1u], vin[m + 2u]);
}

fn word(i: i32, j: i32, k: i32) -> i32 {
  if (i < 0 || j < 0 || k < 0 || i >= i32(L.nx) || j >= i32(L.ny) || k >= i32(L.nz)) { return 0; }
  return ew[u32(i) + L.nx * (u32(j) + L.ny * u32(k))];
}

// the three rows of corner a (from vec4 index r = 18a) of a voxel's matrix times its corner values
fn rows(w: i32, r: u32, v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>, v4: vec4<f32>, v5: vec4<f32>) -> vec3<f32> {
  if (w > 0) {
    return bitcast<f32>(w) * vec3<f32>(
      dot(K0[r + 0u], v0) + dot(K0[r + 1u], v1) + dot(K0[r + 2u], v2) + dot(K0[r + 3u], v3) + dot(K0[r + 4u], v4) + dot(K0[r + 5u], v5),
      dot(K0[r + 6u], v0) + dot(K0[r + 7u], v1) + dot(K0[r + 8u], v2) + dot(K0[r + 9u], v3) + dot(K0[r + 10u], v4) + dot(K0[r + 11u], v5),
      dot(K0[r + 12u], v0) + dot(K0[r + 13u], v1) + dot(K0[r + 14u], v2) + dot(K0[r + 15u], v3) + dot(K0[r + 16u], v4) + dot(K0[r + 17u], v5));
  }
  let b = u32(w & 0x7fffffff) * 144u + r;
  return vec3<f32>(
      dot(Ke[b + 0u], v0) + dot(Ke[b + 1u], v1) + dot(Ke[b + 2u], v2) + dot(Ke[b + 3u], v3) + dot(Ke[b + 4u], v4) + dot(Ke[b + 5u], v5),
      dot(Ke[b + 6u], v0) + dot(Ke[b + 7u], v1) + dot(Ke[b + 8u], v2) + dot(Ke[b + 9u], v3) + dot(Ke[b + 10u], v4) + dot(Ke[b + 11u], v5),
      dot(Ke[b + 12u], v0) + dot(Ke[b + 13u], v1) + dot(Ke[b + 14u], v2) + dot(Ke[b + 15u], v3) + dot(Ke[b + 16u], v4) + dot(Ke[b + 17u], v5));
}

fn elem(w: i32, a: u32, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, p4: vec3<f32>, p5: vec3<f32>, p6: vec3<f32>, p7: vec3<f32>) -> vec3<f32> {
  if (w == 0) { return vec3<f32>(0.0); }
  return rows(w, 18u * a, vec4<f32>(p0, p1.x), vec4<f32>(p1.yz, p2.xy), vec4<f32>(p2.z, p3), vec4<f32>(p4, p5.x), vec4<f32>(p5.yz, p6.xy),
              vec4<f32>(p6.z, p7));
}

fn st(m: u32, u: vec3<f32>) -> vec3<f32> {
  return vec3<f32>(dot(ST[3u * m].xyz, u), dot(ST[3u * m + 1u].xyz, u), dot(ST[3u * m + 2u].xyz, u));
}

// (K u) at node n, from neighbour differences u_m - u_n (K annihilates translations): for smooth
// fields they are small and exact, which keeps float32 products accurate to many more digits
fn applyK(n: u32) -> vec3<f32> {
  let i = i32(n % L.NX);
  let j = i32((n / L.NX) % L.NY);
  let k = i32(n / (L.NX * L.NY));
  let w0 = word(i + -1, j + -1, k + -1);
  let w1 = word(i + 0, j + -1, k + -1);
  let w2 = word(i + -1, j + 0, k + -1);
  let w3 = word(i + 0, j + 0, k + -1);
  let w4 = word(i + -1, j + -1, k + 0);
  let w5 = word(i + 0, j + -1, k + 0);
  let w6 = word(i + -1, j + 0, k + 0);
  let w7 = word(i + 0, j + 0, k + 0);
  let own = nodeU(i + 0, j + 0, k + 0);
  let u13 = vec3<f32>(0.0);
  let u0 = nodeU(i + -1, j + -1, k + -1) - own;
  let u1 = nodeU(i + 0, j + -1, k + -1) - own;
  let u2 = nodeU(i + 1, j + -1, k + -1) - own;
  let u3 = nodeU(i + -1, j + 0, k + -1) - own;
  let u4 = nodeU(i + 0, j + 0, k + -1) - own;
  let u5 = nodeU(i + 1, j + 0, k + -1) - own;
  let u6 = nodeU(i + -1, j + 1, k + -1) - own;
  let u7 = nodeU(i + 0, j + 1, k + -1) - own;
  let u8 = nodeU(i + 1, j + 1, k + -1) - own;
  let u9 = nodeU(i + -1, j + -1, k + 0) - own;
  let u10 = nodeU(i + 0, j + -1, k + 0) - own;
  let u11 = nodeU(i + 1, j + -1, k + 0) - own;
  let u12 = nodeU(i + -1, j + 0, k + 0) - own;
  let u14 = nodeU(i + 1, j + 0, k + 0) - own;
  let u15 = nodeU(i + -1, j + 1, k + 0) - own;
  let u16 = nodeU(i + 0, j + 1, k + 0) - own;
  let u17 = nodeU(i + 1, j + 1, k + 0) - own;
  let u18 = nodeU(i + -1, j + -1, k + 1) - own;
  let u19 = nodeU(i + 0, j + -1, k + 1) - own;
  let u20 = nodeU(i + 1, j + -1, k + 1) - own;
  let u21 = nodeU(i + -1, j + 0, k + 1) - own;
  let u22 = nodeU(i + 0, j + 0, k + 1) - own;
  let u23 = nodeU(i + 1, j + 0, k + 1) - own;
  let u24 = nodeU(i + -1, j + 1, k + 1) - own;
  let u25 = nodeU(i + 0, j + 1, k + 1) - own;
  let u26 = nodeU(i + 1, j + 1, k + 1) - own;
  var acc: vec3<f32>;
  if (w0 > 0 && w1 == w0 && w2 == w0 && w3 == w0 && w4 == w0 && w5 == w0 && w6 == w0 && w7 == w0) {
    acc = bitcast<f32>(w0) * (
      st(0u, u0) + st(1u, u1) + st(2u, u2) + st(3u, u3) + st(4u, u4) +
      st(5u, u5) + st(6u, u6) + st(7u, u7) + st(8u, u8) + st(9u, u9) +
      st(10u, u10) + st(11u, u11) + st(12u, u12) + st(14u, u14) +
      st(15u, u15) + st(16u, u16) + st(17u, u17) + st(18u, u18) + st(19u, u19) +
      st(20u, u20) + st(21u, u21) + st(22u, u22) + st(23u, u23) + st(24u, u24) +
      st(25u, u25) + st(26u, u26));
  } else {
    acc = elem(w0, 6u, u0, u1, u4, u3, u9, u10, u13, u12) +
          elem(w1, 7u, u1, u2, u5, u4, u10, u11, u14, u13) +
          elem(w2, 5u, u3, u4, u7, u6, u12, u13, u16, u15) +
          elem(w3, 4u, u4, u5, u8, u7, u13, u14, u17, u16) +
          elem(w4, 2u, u9, u10, u13, u12, u18, u19, u22, u21) +
          elem(w5, 3u, u10, u11, u14, u13, u19, u20, u23, u22) +
          elem(w6, 1u, u12, u13, u16, u15, u21, u22, u25, u24) +
          elem(w7, 0u, u13, u14, u17, u16, u22, u23, u26, u25);
  }
  if (L.hasDiag == 1u) { acc += vec3<f32>(dadd[3u * n], dadd[3u * n + 1u], dadd[3u * n + 2u]) * own; }
  return acc;
}

fn freeOf(n: u32) -> vec3<f32> { return vec3<f32>(invD[3u * n], invD[3u * n + 1u], invD[3u * n + 2u]); }

fn put(n: u32, y: vec3<f32>) {
  vout[3u * n] = y.x;
  vout[3u * n + 1u] = y.y;
  vout[3u * n + 2u] = y.z;
}

// vout = K vin (held DOFs zero)
@compute @workgroup_size(64)
fn matvec(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * L.strideN;
  if (n >= L.nNodes) { return; }
  let d = freeOf(n);
  var y = vec3<f32>(0.0);
  if (any(d != vec3<f32>(0.0))) { y = select(vec3<f32>(0.0), applyK(n), d != vec3<f32>(0.0)); }
  put(n, y);
}

// Chebyshev smoother step into a second buffer: d = c1 d + c2 D^-1 (rhs - K vin), vout = vin + d
// (d, per DOF, is updated in place in vout2)
fn chebStep(n: u32, c1: f32, c2: f32) {
  let dI = freeOf(n);
  let q = 3u * n;
  var z = vec3<f32>(0.0);
  var d = vec3<f32>(0.0);
  if (any(dI != vec3<f32>(0.0))) {
    let zo = vec3<f32>(vin[q], vin[q + 1u], vin[q + 2u]);
    let r = vec3<f32>(rhs[q], rhs[q + 1u], rhs[q + 2u]);
    let dOld = vec3<f32>(vout2[q], vout2[q + 1u], vout2[q + 2u]);
    d = select(vec3<f32>(0.0), c1 * dOld + c2 * dI * (r - applyK(n)), dI != vec3<f32>(0.0));
    z = zo + d;
  }
  put(n, z);
  vout2[q] = d.x;
  vout2[q + 1u] = d.y;
  vout2[q + 2u] = d.z;
}

// first step from a nonzero vin
@compute @workgroup_size(64)
fn cheb_a(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * L.strideN;
  if (n >= L.nNodes) { return; }
  chebStep(n, 0.0, L.first);
}

// second step
@compute @workgroup_size(64)
fn cheb_b(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * L.strideN;
  if (n >= L.nNodes) { return; }
  chebStep(n, L.c1, L.c2);
}

// vout = rhs - K vin (held DOFs zero)
@compute @workgroup_size(64)
fn resid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * L.strideN;
  if (n >= L.nNodes) { return; }
  let d = freeOf(n);
  var y = vec3<f32>(0.0);
  if (any(d != vec3<f32>(0.0))) {
    let q = 3u * n;
    let r = vec3<f32>(rhs[q], rhs[q + 1u], rhs[q + 2u]);
    y = select(vec3<f32>(0.0), r - applyK(n), d != vec3<f32>(0.0));
  }
  put(n, y);
}

// first step from zero: vout = d = first D^-1 rhs
@compute @workgroup_size(64)
fn cheb_first(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  let d = L.first * invD[q] * rhs[q];
  vout[q] = d;
  vout2[q] = d;
}

// coarse r = P^T fine t, one thread per coarse node (invD is the coarse level's)
@compute @workgroup_size(64)
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
@compute @workgroup_size(64)
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

// coarsest level: z = A^-1 r over the free DOFs (ew = free DOF list, at most 3000; ainv symmetric).
// A workgroup takes 32 rows, 8 threads per row each summing every 8th column (neighbouring threads
// read neighbouring words of a column), then adds up the 8 parts. r is gathered into workgroup
// memory first.
var<workgroup> cr: array<f32, 3000>;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn coarsest(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  for (var c = lid.x; c < L.m; c += 256u) { cr[c] = vin[u32(ew[c])]; }
  workgroupBarrier();
  let row = wid.x * 32u + lid.x % 32u;
  let k = lid.x / 32u;
  var s = 0.0;
  if (row < L.m) {
    for (var c = k; c < L.m; c += 8u) { s += ainv[c * L.m + row] * cr[c]; }
  }
  part[lid.x] = s;
  workgroupBarrier();
  if (lid.x < 32u && row < L.m) {
    var t = 0.0;
    for (var q = 0u; q < 8u; q++) { t += part[lid.x + 32u * q]; }
    vout[u32(ew[row])] = t;
  }
}

// Lanczos steps on D^-1/2 K D^-1/2 for the smoother's eigenvalue estimate (per DOF of one level):
// x = D^-1/2 v; w = K x; w = D^-1/2 w - beta vPrev; alpha = w.v; w -= alpha v; beta^2 = w.w;
// vPrev = v; v = w / beta. S[8] = alpha, S[9] = beta^2, S[10] = the previous beta^2, S[11] = step;
// step j keeps alpha in S[16 + j] and beta^2 in S[32 + j].
@compute @workgroup_size(64)
fn lz_x(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = sqrt(invD[q]) * vin[q];
}

@compute @workgroup_size(64)
fn lz_w(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = sqrt(invD[q]) * vout[q] - sqrt(max(S[10], 0.0)) * vin[q];
}

@compute @workgroup_size(64)
fn lz_axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = vout[q] - S[8] * vin[q];
}

@compute @workgroup_size(64)
fn lz_next(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  let b = sqrt(max(S[9], 0.0));
  vout2[q] = vout[q];
  vout[q] = select(0.0, vin[q] / b, b > 0.0);
}

@compute @workgroup_size(1)
fn lz_store() {
  let j = u32(S[11]);
  S[16u + j] = S[8];
  S[32u + j] = S[9];
  S[10] = S[9];
  S[11] = S[11] + 1.0;
}

// x += alpha p ; r -= alpha q
@compute @workgroup_size(64)
fn update_xr(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  let a = S[2];
  vout[q] = vout[q] + a * vin[q];
  vout2[q] = vout2[q] - a * vin2[q];
}

// p = z + beta p
@compute @workgroup_size(64)
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

// flexible (Polak-Ribiere) beta = z.(r - r_old) / rz_old with r - r_old = -alpha q
@compute @workgroup_size(1)
fn cg_beta() {
  S[3] = select(0.0, max(0.0, -S[2] * S[7] / S[0]), S[0] != 0.0);
  S[0] = S[5];
}
