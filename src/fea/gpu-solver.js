// WebGPU version of the multigrid-preconditioned conjugate-gradient solve in solver.js (the native
// app's GPU solver, same shader).
//
// The multigrid hierarchy (Galerkin coarse operators, smoother eigenvalue ranges, held DOFs) is
// built once on the CPU by VoxelFEA and uploaded; the CG loop then runs on the GPU in 32-bit:
//   - matrix-free K*x with one thread per grid node, gathering from its 8 surrounding voxels from a
//     3x3x3 block of neighbour differences (accurate in float32 for smooth fields), with the
//     assembled 27-point stencil inside uniform regions,
//   - V-cycle: degree-2 Chebyshev smoothing fused into the products, full-weighting restriction,
//     trilinear prolongation and a dense inverse on the (small) coarsest level,
//   - dot products by workgroup reduction, with the CG scalars kept on the GPU.
// 64-bit accuracy comes from reliable updates: whenever the GPU's residual has dropped tenfold, its
// solution is added to a float64 one on the CPU and the true residual f - K u, computed there,
// replaces the GPU's.
import { hardwareAdapter } from '../core/webgpu.js';
import { tridiagonalMax, lanczosStart } from './solver.js';

const WG = 64;
const RED_GROUPS = 1024;
export const GPU_COARSEST_DOF = 300;

// scalar slots on the GPU (0 = rz, 3 = beta)
const PQ = 1, ALPHA = 2, RR = 4, RZN = 5, ZQ = 7;

const SHADER = /* wgsl */ `
// Multigrid V-cycle (degree-2 Chebyshev smoothing) and float32 conjugate gradients for the voxel
// FEA on the GPU. Workgroup size 64.
//
// Matrix products gather, for each node, the rows of its 8 surrounding voxels from a 3x3x3 block of
// neighbour values loaded once. Each voxel is one word: 0 = empty, > 0 = the f32 bits of s for an
// s * Kb element (Kb, the level's uniform element, is the start of Ke), < 0 = sign bit | index of
// its own 24x24 matrix after it in Ke (coarse levels; pairs of 16-bit halves times the level's
// scale L.hs, which halves the memory traffic of these bandwidth-bound products). A node whose 8
// voxels are the same s * Kb uses the
// assembled 27-point stencil (243 multiply-adds, not 576). The product and the smoothing steps built
// on it share one kernel ("op"), so drivers compile the large unrolled code once.

struct Level {
  nx: u32, ny: u32, nz: u32, NX: u32,
  NY: u32, NZ: u32, nNodes: u32, nDof: u32,
  p0: u32, strideN: u32, strideD: u32, m: u32,
  first: f32, c1: f32, c2: f32, hs: f32,     // Chebyshev smoother coefficients; scale of the halves in Ke
  CNX: u32, CNY: u32, CNZ: u32, cNodes: u32,
  cStrideN: u32, hasDiag: u32, p5: u32, p6: u32,
};
struct Red { slot: u32, count: u32, q0: u32, q1: u32 };

@group(0) @binding(0) var<uniform> L: Level;
@group(0) @binding(1) var<storage, read> vin: array<f32>;
@group(0) @binding(2) var<storage, read_write> vout: array<f32>;
@group(0) @binding(3) var<storage, read> invD: array<f32>;
@group(0) @binding(4) var<storage, read> ew: array<i32>;          // voxel words (coarsest: free DOF list)
@group(0) @binding(5) var<storage, read> Ke: array<vec4<u32>>;    // Kb (f32 bits, 144 vec4), then own matrices (halves, 72 vec4 each)
@group(0) @binding(7) var<storage, read> rhs: array<f32>;
@group(0) @binding(8) var<storage, read_write> vout2: array<f32>;
@group(0) @binding(9) var<storage, read> vin2: array<f32>;
@group(0) @binding(10) var<storage, read_write> S: array<f32>;
@group(0) @binding(11) var<storage, read_write> partials: array<f32>;
@group(0) @binding(12) var<uniform> R: Red;
@group(0) @binding(13) var<storage, read> dadd: array<f32>;
@group(0) @binding(14) var<uniform> ST: array<vec4<f32>, 81>;     // stencil: neighbour m, row d -> ST[3m + d].xyz
@group(0) @binding(15) var<storage, read> ainv: array<f32>;       // coarsest: dense inverse (symmetric)
struct Op { mode: u32, q0: u32, q1: u32, q2: u32 };
@group(0) @binding(16) var<uniform> OP: Op;                        // op kernel: what to do with K vin

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

// eight halves of a matrix row times eight values
fn h8(q: vec4<u32>, lo: vec4<f32>, hi: vec4<f32>) -> f32 {
  return dot(vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y)), lo) + dot(vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w)), hi);
}

// row b of Kb (six vec4 from index b) times 24 values
fn kb(b: u32, v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>, v4: vec4<f32>, v5: vec4<f32>) -> f32 {
  return dot(bitcast<vec4<f32>>(Ke[b]), v0) + dot(bitcast<vec4<f32>>(Ke[b + 1u]), v1) + dot(bitcast<vec4<f32>>(Ke[b + 2u]), v2) +
    dot(bitcast<vec4<f32>>(Ke[b + 3u]), v3) + dot(bitcast<vec4<f32>>(Ke[b + 4u]), v4) + dot(bitcast<vec4<f32>>(Ke[b + 5u]), v5);
}

// the three rows of corner a (from vec4 index r = 18a) of a voxel's matrix times its corner values
fn rows(w: i32, r: u32, v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>, v4: vec4<f32>, v5: vec4<f32>) -> vec3<f32> {
  if (w < 0) {
    let h = 144u + u32(w & 0x7fffffff) * 72u + r / 2u;
    return L.hs * vec3<f32>(
      h8(Ke[h], v0, v1) + h8(Ke[h + 1u], v2, v3) + h8(Ke[h + 2u], v4, v5),
      h8(Ke[h + 3u], v0, v1) + h8(Ke[h + 4u], v2, v3) + h8(Ke[h + 5u], v4, v5),
      h8(Ke[h + 6u], v0, v1) + h8(Ke[h + 7u], v2, v3) + h8(Ke[h + 8u], v4, v5));
  }
  return bitcast<f32>(w) * vec3<f32>(
    kb(r, v0, v1, v2, v3, v4, v5), kb(r + 6u, v0, v1, v2, v3, v4, v5), kb(r + 12u, v0, v1, v2, v3, v4, v5));
}

fn elem(w: i32, a: u32, p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, p4: vec3<f32>, p5: vec3<f32>, p6: vec3<f32>, p7: vec3<f32>) -> vec3<f32> {
  if (w == 0) { return vec3<f32>(0.0); }
  return rows(w, 18u * a, vec4<f32>(p0, p1.x), vec4<f32>(p1.yz, p2.xy), vec4<f32>(p2.z, p3), vec4<f32>(p4, p5.x), vec4<f32>(p5.yz, p6.xy),
              vec4<f32>(p6.z, p7));
}

// voxel v = (di, dj, dk) around a node: the node is its corner CA[v]
const CA: array<u32, 8> = array<u32, 8>(6u, 7u, 5u, 4u, 2u, 3u, 1u, 0u);

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
    // near a surface: voxel by voxel, from the block copied into an array (one loop body keeps the
    // shader small; interior nodes, the bulk, take the unrolled stencil above)
    let ua = array<vec3<f32>, 27>(u0, u1, u2, u3, u4, u5, u6, u7, u8, u9, u10, u11, u12, u13, u14, u15, u16, u17, u18, u19, u20, u21, u22, u23, u24, u25, u26);
    let wa = array<i32, 8>(w0, w1, w2, w3, w4, w5, w6, w7);
    acc = vec3<f32>(0.0);
    for (var v = 0u; v < 8u; v++) {
      let w = wa[v];
      if (w == 0) { continue; }
      let b = (v & 1u) + 3u * ((v >> 1u) & 1u) + 9u * (v >> 2u);
      acc += elem(w, CA[v], ua[b], ua[b + 1u], ua[b + 4u], ua[b + 3u], ua[b + 9u], ua[b + 10u], ua[b + 13u], ua[b + 12u]);
    }
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

// One kernel for everything built on K vin (held DOFs zero), by OP.mode:
//   0: vout = K vin
//   1, 2: Chebyshev step into a second buffer, d = c1 d + c2 D^-1 (rhs - K vin), vout = vin + d, with
//         (c1, c2) = (0, first) for the first step from a nonzero vin (1) and (c1, c2) for the second (2);
//         d (per DOF) is updated in place in vout2
//   3: vout = rhs - K vin
@compute @workgroup_size(64)
fn op(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * L.strideN;
  if (n >= L.nNodes) { return; }
  let dI = freeOf(n);
  let q = 3u * n;
  let mode = OP.mode;
  var y = vec3<f32>(0.0);
  var d = vec3<f32>(0.0);
  if (any(dI != vec3<f32>(0.0))) {
    let Ku = applyK(n);
    if (mode == 0u) {
      y = Ku;
    } else {
      let r = vec3<f32>(rhs[q], rhs[q + 1u], rhs[q + 2u]);
      if (mode == 3u) {
        y = r - Ku;
      } else {
        let c1 = select(0.0, L.c1, mode == 2u);
        let c2 = select(L.first, L.c2, mode == 2u);
        let dOld = vec3<f32>(vout2[q], vout2[q + 1u], vout2[q + 2u]);
        d = select(vec3<f32>(0.0), c1 * dOld + c2 * dI * (r - Ku), dI != vec3<f32>(0.0));
        y = vec3<f32>(vin[q], vin[q + 1u], vin[q + 2u]) + d;
      }
    }
    y = select(vec3<f32>(0.0), y, dI != vec3<f32>(0.0));
  }
  put(n, y);
  if (mode == 1u || mode == 2u) {
    vout2[q] = d.x;
    vout2[q + 1u] = d.y;
    vout2[q + 2u] = d.z;
  }
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

// coarsest level: z = A^-1 r over the free DOFs (ew = free DOF list; ainv symmetric). First the
// free entries of r are gathered into vout2 (a compact vector), then a workgroup takes 32 rows, 8
// threads per row each summing every 8th column (neighbouring threads read neighbouring words of
// a column), and adds up the 8 parts.
@compute @workgroup_size(64)
fn coarse_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= L.m) { return; }
  vout2[c] = vin[u32(ew[c])];
}

var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn coarsest(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wid.x * 32u + lid.x % 32u;
  let k = lid.x / 32u;
  var s = 0.0;
  if (row < L.m) {
    for (var c = k; c < L.m; c += 8u) { s += ainv[c * L.m + row] * vin2[c]; }
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
`;

let devicePromise = null;

/** A WebGPU device with the adapter's full buffer limits, or null when WebGPU is unavailable. */
export function gpuDevice() {
  devicePromise ??= (async () => {
    try {
      const adapter = await hardwareAdapter({ powerPreference: 'high-performance' });
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
    for (let i = col; i < m; i++) {
      const ri = i * m;
      let s = y[i];
      for (let k = col; k < i; k++) s -= Lf[ri + k] * y[k];
      y[i] = s / Lf[ri + i];
    }
    // backward by column sweeps, reading rows of L
    for (let i = m - 1; i >= 0; i--) {
      const ri = i * m, xi = y[i] / Lf[ri + i];
      y[i] = xi;
      for (let k = 0; k < i; k++) y[k] -= Lf[ri + k] * xi;
    }
    for (let i = 0; i < m; i++) inv[i * m + col] = y[i];
  }
  return inv;
}

// compiled once per device: every study builds a new solver
const pipelineCache = new WeakMap();
function pipelines(device) {
  if (!pipelineCache.has(device)) {
    const module = device.createShaderModule({ code: SHADER });
    const pipe = (entryPoint) => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } });
    pipelineCache.set(device, Object.fromEntries(
      ['op', 'cheb_first', 'restrict_', 'prolong', 'coarse_gather', 'coarsest', 'update_xr', 'update_p', 'dot_partial', 'reduce', 'cg_alpha', 'cg_beta',
        'lz_x', 'lz_w', 'lz_axpy', 'lz_next', 'lz_store']
        .map((e) => [e, pipe(e)]),
    ));
  }
  return pipelineCache.get(device);
}

const f32 = new Float32Array(1), u32 = new Uint32Array(f32.buffer);
/** IEEE half-precision bits of v (rounded to float32, then to the nearest half, ties to even). */
export function toHalf(v) {
  f32[0] = v;
  const x = u32[0], sign = (x >>> 16) & 0x8000, mant = x & 0x7fffff, exp = ((x >>> 23) & 0xff) - 112;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) { // subnormal half
    if (exp < -10) return sign;
    const m = mant | 0x800000, shift = 14 - exp;
    let h = m >>> shift;
    const rem = m & ((1 << shift) - 1), half = 1 << (shift - 1);
    if (rem > half || (rem === half && (h & 1))) h++;
    return sign | h;
  }
  let h = (exp << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++; // a carry correctly bumps the exponent
  return sign | h;
}

/** Degree-2 Chebyshev coefficients over [0.1, 1.15] * lmax (as solver.js). */
function chebyshev(lmax) {
  const b = 1.15 * lmax, a = 0.1 * lmax;
  const theta = (b + a) / 2, delta = (b - a) / 2, sigma = theta / delta;
  const rho0 = 1 / sigma, rho1 = 1 / (2 * sigma - rho0);
  return [1 / theta, rho1 * rho0, (2 * rho1) / delta];
}

export class GPUFEASolver {
  /** @param {import('./solver.js').VoxelFEA} fea built with coarsestMaxDof <= GPU_COARSEST_DOF */
  static async create(fea) {
    const device = await gpuDevice();
    if (!device) throw new Error('WebGPU is not available.');
    device.pushErrorScope('validation');
    device.pushErrorScope('out-of-memory');
    let solver;
    try {
      solver = new GPUFEASolver(device, fea);
      await solver.estimateSmoothers();
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
    const U = GPUBufferUsage.UNIFORM | D;
    this.p = pipelines(device);
    const bits = new Int32Array(1), bitsF = new Float32Array(bits.buffer);

    this.L = levels.map((lv, l) => {
      const { nx, ny, nz } = lv;
      // one word per voxel: 0 empty, f32 bits of s for s * Kb, sign bit | the index of its own matrix
      // (after Kb in Ke) otherwise
      const words = new Int32Array(nx * ny * nz), own = [];
      for (let e = 0; e < lv.elems.length; e++) {
        if (lv.scale[e] > 0) {
          bitsF[0] = lv.scale[e];
          words[lv.elems[e]] = bits[0];
        } else words[lv.elems[e]] = (0x80000000 | own.push(e) - 1) | 0;
      }
      const invD = new Float32Array(lv.nDof);
      for (let i = 0; i < lv.nDof; i++) invD[i] = lv.fixed[i] ? 0 : lv.invDiag[i];
      const last = l === levels.length - 1;
      const next = levels[l + 1];
      const [, , strideN] = dispatchSize(lv.nNodes, device);
      const [, , strideD] = dispatchSize(lv.nDof, device);
      const [, , cStrideN] = next ? dispatchSize(next.nNodes, device) : [0, 0, 0];
      const params = new ArrayBuffer(96);
      const u = new Uint32Array(params), f = new Float32Array(params);
      u.set([nx, ny, nz, lv.NX, lv.NY, lv.NZ, lv.nNodes, lv.nDof, 0, strideN, strideD, last ? coarse.m : 0]);
      if (!last && lv.lmax > 0) f.set(chebyshev(lv.lmax), 12); // otherwise set by estimateSmoothers()
      if (next) u.set([next.NX, next.NY, next.NZ, next.nNodes, cStrideN], 16);
      u[21] = lv.diagAdd ? 1 : 0;
      // the level's own element matrices in 16-bit halves (their products are bandwidth-bound), with
      // a power-of-two scale that keeps the largest entries near 2^14, well inside the half range
      let kmax = 0;
      for (const e of own) for (let i = 576 * e; i < 576 * (e + 1); i++) kmax = Math.max(kmax, Math.abs(lv.K[i]));
      const hs = kmax > 0 ? 2 ** Math.ceil(Math.log2(kmax / 16384)) : 1;
      f[15] = hs;
      // the level's uniform element (rows as 6 vec4) and its 27-point stencil: neighbour
      // m = ox + 3 oy + 9 oz, row d -> [3m + d].xyz (from solver.js's layout [(row 3 + d) 9 + 3 ox + c])
      const st = new Float32Array(81 * 4);
      for (let row = 0; row < 9; row++) {
        for (let col = 0; col < 9; col++) {
          const m = ((col / 3) | 0) + 3 * row, c = col % 3;
          for (let d = 0; d < 3; d++) st[(3 * m + d) * 4 + c] = lv.S[(row * 3 + d) * 9 + col];
        }
      }
      const L = {
        lv, last, out: last ? 0 : 1,
        params: upload(new Uint8Array(params), U),
        invD: upload(invD),
        words: upload(words),
        // Kb in 32 bits, then the own matrices as halves / hs
        Ke: upload((() => {
          const ke = new Uint16Array(576 * (2 + own.length));
          new Float32Array(ke.buffer, 0, 576).set(lv.Kb.subarray(0, 576));
          own.forEach((e, q) => { for (let i = 0; i < 576; i++) ke[576 * (2 + q) + i] = toHalf(lv.K[576 * e + i] / hs); });
          return ke;
        })()),
        stencil: upload(st, U),
        dadd: upload(lv.diagAdd ? Float32Array.from(lv.diagAdd) : new Float32Array(4)),
        r: make(4 * lv.nDof), z: make(4 * lv.nDof), t: make(4 * lv.nDof), d: make(4 * lv.nDof),
      };
      L.buf = (p) => (p ? L.t : L.z);
      if (last) {
        const free = new Int32Array(coarse.m);
        for (let i = 0; i < lv.nDof; i++) if (coarse.map[i] >= 0) free[coarse.map[i]] = i;
        L.free = upload(free);
        L.ainv = upload(coarseInverse(coarse));
        L.rc = make(4 * coarse.m);
      }
      return L;
    });
    const L0 = this.L[0];
    const n = levels[0].nDof;
    this.n = n;
    this.x = make(4 * n);
    this.pv = make(4 * n);
    this.qv = make(4 * n);
    this.S = make(256);
    this.partials = make(4 * RED_GROUPS);
    this.dotGroups = Math.min(RED_GROUPS, Math.ceil(n / 256));
    // reduction slots, one 256-byte-aligned entry per scalar
    const red = new Uint32Array(64 * 10);
    for (let s = 0; s < 10; s++) red.set([s, this.dotGroups], s * 64);
    this.red = upload(red, U);
    this.readBuf = device.createBuffer({ size: 256, usage: GPUBufferUsage.MAP_READ | D });
    this.xRead = device.createBuffer({ size: 4 * n, usage: GPUBufferUsage.MAP_READ | D });
    this.buffers.push(this.readBuf, this.xRead);

    // bind groups: [p] reads buffer p (0 = z, 1 = t) and writes the other
    const bg = (pipeline, entries) => device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: Object.entries(entries).map(([binding, res]) =>
        ({ binding: Number(binding), resource: res instanceof GPUBuffer ? { buffer: res } : res })),
    });
    const P = this.p;
    // the op kernel's modes: 0 product, 1 / 2 Chebyshev steps, 3 residual
    this.opMode = [0, 1, 2, 3].map((m) => upload(new Uint32Array([m, 0, 0, 0]), U));
    this.L.forEach((L, l) => {
      const N = this.L[l + 1];
      const op = (mode, from, to) => bg(P.op, {
        0: L.params, 1: from, 2: to, 3: L.invD, 4: L.words, 5: L.Ke, 7: L.r, 8: L.d, 13: L.dadd, 14: L.stencil, 16: this.opMode[mode],
      });
      L.bg = {
        first: bg(P.cheb_first, { 0: L.params, 2: L.z, 3: L.invD, 7: L.r, 8: L.d }),
        chebA: [0, 1].map((p) => op(1, L.buf(p), L.buf(1 - p))),
        chebB: [0, 1].map((p) => op(2, L.buf(p), L.buf(1 - p))),
        resid: [0, 1].map((p) => op(3, L.buf(p), L.buf(1 - p))),
      };
      if (N) {
        // Lanczos steps (estimateSmoothers): v = z, vPrev = d, w = t, x = r
        L.bg.lz = {
          x: bg(P.lz_x, { 0: L.params, 1: L.z, 2: L.r, 3: L.invD }),
          mv: op(0, L.r, L.t),
          w: bg(P.lz_w, { 0: L.params, 1: L.d, 2: L.t, 3: L.invD, 10: this.S }),
          axpy: bg(P.lz_axpy, { 0: L.params, 1: L.z, 2: L.t, 10: this.S }),
          next: bg(P.lz_next, { 0: L.params, 1: L.t, 2: L.z, 8: L.d, 10: this.S }),
        };
        L.bg.restrict = [0, 1].map((p) => bg(P.restrict_, { 0: L.params, 1: L.buf(p), 2: N.r, 3: N.invD }));
        L.bg.prolong = [0, 1].map((p) => bg(P.prolong, { 0: L.params, 1: N.buf(N.out), 2: L.buf(p), 3: L.invD }));
      } else {
        L.bg.gather = bg(P.coarse_gather, { 0: L.params, 1: L.r, 4: L.free, 8: L.rc });
        L.bg.coarsest = bg(P.coarsest, { 0: L.params, 9: L.rc, 2: L.z, 4: L.free, 15: L.ainv });
      }
    });
    const z0 = L0.buf(L0.out);
    this.bg = {
      mvP: bg(P.op, { 0: L0.params, 1: this.pv, 2: this.qv, 3: L0.invD, 4: L0.words, 5: L0.Ke, 7: L0.r, 8: L0.d, 13: L0.dadd, 14: L0.stencil, 16: this.opMode[0] }),
      updXR: bg(P.update_xr, { 0: L0.params, 1: this.pv, 2: this.x, 8: L0.r, 9: this.qv, 10: this.S }),
      updP: bg(P.update_p, { 0: L0.params, 1: z0, 2: this.pv, 10: this.S }),
      alpha: bg(P.cg_alpha, { 10: this.S }),
      beta: bg(P.cg_beta, { 10: this.S }),
      lzStore: bg(P.lz_store, { 10: this.S }),
      // one bind group per scalar slot (auto layouts cannot take dynamic offsets)
      reduce: Array.from({ length: 10 }, (_, slot) => bg(P.reduce, { 10: this.S, 11: this.partials, 12: { buffer: this.red, offset: slot * 256, size: 16 } })),
    };
    this.dotBG = new Map();
  }

  dotBindGroup(a, b, params = this.L[0].params) {
    const key = `${this.buffers.indexOf(a)}:${this.buffers.indexOf(b)}:${this.buffers.indexOf(params)}`;
    if (!this.dotBG.has(key)) {
      this.dotBG.set(key, this.device.createBindGroup({
        layout: this.p.dot_partial.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: params } },
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

  /** S[slot] = a.b over the DOFs of level l (the same number of partial sums on every level). */
  dot(pass, a, b, slot, l = 0) {
    pass.setPipeline(this.p.dot_partial);
    pass.setBindGroup(0, this.dotBindGroup(a, b, this.L[l].params));
    pass.dispatchWorkgroups(this.dotGroups);
    pass.setPipeline(this.p.reduce);
    pass.setBindGroup(0, this.bg.reduce[slot]);
    pass.dispatchWorkgroups(1);
  }

  /** V-cycle as in solver.js: Chebyshev pre-smoothing from zero, coarse correction, post-smoothing. */
  vcycle(pass, l) {
    const L = this.L[l], P = this.p;
    if (L.last) {
      this.run(pass, P.coarse_gather, L.bg.gather, this.fea.coarse.m);
      pass.setPipeline(P.coarsest);
      pass.setBindGroup(0, L.bg.coarsest);
      pass.dispatchWorkgroups(Math.ceil(this.fea.coarse.m / 32)); // 32 rows per workgroup
      return;
    }
    const nD = L.lv.nDof, nN = L.lv.nNodes;
    this.run(pass, P.cheb_first, L.bg.first, nD); //       z
    this.run(pass, P.op, L.bg.chebB[0], nN); //            z -> t
    this.run(pass, P.op, L.bg.resid[1], nN); //            t -> residual in z
    this.run(pass, P.restrict_, L.bg.restrict[0], this.L[l + 1].lv.nNodes);
    this.vcycle(pass, l + 1);
    this.run(pass, P.prolong, L.bg.prolong[1], nN); //     t += P z_coarse
    this.run(pass, P.op, L.bg.chebA[1], nN); //            t -> z
    this.run(pass, P.op, L.bg.chebB[0], nN); //            z -> t
  }

  /**
   * The smoothers' eigenvalue estimates for the levels that have none (as the native GPU solver):
   * 10 Lanczos steps on each, all levels in one submission and one read-back. They also serve a
   * CPU solve of the same hierarchy.
   */
  async estimateSmoothers() {
    const todo = this.L.filter((L) => !L.last && !(L.lv.lmax > 0));
    if (!todo.length) return;
    const dev = this.device, P = this.p, STEPS = 10;
    const read = dev.createBuffer({ size: 256 * todo.length, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = dev.createCommandEncoder();
    todo.forEach((L, i) => {
      const lv = L.lv, m = lv.nDof, l = this.L.indexOf(L);
      // the CPU estimate's start vector (VoxelFEA.estimateOmega), so both give the same estimates
      dev.queue.writeBuffer(L.z, 0, Float32Array.from(lanczosStart(lv)));
      enc.clearBuffer(L.d);
      enc.clearBuffer(this.S);
      const pass = enc.beginComputePass();
      for (let j = 0; j < STEPS; j++) {
        this.run(pass, P.lz_x, L.bg.lz.x, m);
        this.run(pass, P.op, L.bg.lz.mv, lv.nNodes);
        this.run(pass, P.lz_w, L.bg.lz.w, m);
        this.dot(pass, L.t, L.z, 8, l);
        this.run(pass, P.lz_axpy, L.bg.lz.axpy, m);
        this.dot(pass, L.t, L.t, 9, l);
        this.run(pass, P.lz_next, L.bg.lz.next, m);
        this.run(pass, P.lz_store, this.bg.lzStore, 1);
      }
      pass.end();
      enc.copyBufferToBuffer(this.S, 0, read, 256 * i, 256);
    });
    dev.queue.submit([enc.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    const sc = new Float32Array(read.getMappedRange().slice(0));
    read.unmap();
    read.destroy();
    todo.forEach((L, i) => {
      const o = 64 * i, alpha = [], beta = [];
      for (let j = 0; j < STEPS; j++) {
        alpha.push(sc[o + 16 + j]);
        if (!(sc[o + 32 + j] > 1e-24 * sc[o + 16 + j] * sc[o + 16 + j]) || j + 1 === STEPS) break;
        beta.push(Math.sqrt(sc[o + 32 + j]));
      }
      const lv = L.lv;
      lv.lmax = tridiagonalMax(alpha, beta);
      lv.omega = 1.2 / (lv.lmax * 1.05);
      dev.queue.writeBuffer(L.params, 48, new Float32Array(chebyshev(lv.lmax)));
    });
  }

  async read(buffer, staging, bytes) {
    const enc = this.device.createCommandEncoder();
    enc.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(staging.getMappedRange(0, bytes).slice(0));
    staging.unmap();
    return out;
  }

  /** dst = float32(scale * v), held DOFs zero. */
  upload(dst, v, scale) {
    const fixed = this.fea.levels[0].fixed, n = this.n, h = new Float32Array(n);
    for (let i = 0; i < n; i++) h[i] = fixed[i] ? 0 : v[i] * scale;
    this.device.queue.writeBuffer(dst, 0, h);
  }

  clear(...bufs) {
    const enc = this.device.createCommandEncoder();
    for (const b of bufs) enc.clearBuffer(b);
    this.device.queue.submit([enc.finish()]);
  }

  /** One multigrid V-cycle on the GPU: z ~ K^-1 r (float64 in and out; the eigen preconditioner). */
  async precondition(r) {
    const fixed = this.fea.levels[0].fixed, n = this.n;
    // the V-cycle is linear: scale r to order one so float32 neither underflows nor loses digits
    let rmax = 0;
    for (let i = 0; i < n; i++) if (!fixed[i]) rmax = Math.max(rmax, Math.abs(r[i]));
    const z = new Float64Array(n);
    if (!(rmax > 0)) return z;
    const L0 = this.L[0];
    this.upload(L0.r, r, 1 / rmax);
    const enc = this.device.createCommandEncoder();
    const pass = enc.beginComputePass();
    this.vcycle(pass, 0);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    const z32 = await this.read(L0.buf(L0.out), this.xRead, 4 * n);
    for (let i = 0; i < n; i++) z[i] = fixed[i] ? 0 : z32[i] * rmax;
    return z;
  }

  /**
   * Approximate K^-1 r for each vector of a block: `iterations` steps of the GPU's multigrid-CG from
   * zero (0 = one V-cycle), all in one submission and one read-back - the eigen solvers'
   * preconditioner (a GPU round trip per vector would cost more than the work).
   */
  async preconditionBlock(R, iterations = 0) {
    const fixed = this.fea.levels[0].fixed, n = this.n, k = R.length, dev = this.device;
    if (!this.blockIn || this.blockK < k) {
      for (const b of [this.blockIn, this.blockOut, this.blockRead]) if (b) { b.destroy(); this.buffers.splice(this.buffers.indexOf(b), 1); }
      this.blockK = k;
      const S = GPUBufferUsage.STORAGE, D = GPUBufferUsage.COPY_DST, C = GPUBufferUsage.COPY_SRC;
      this.blockIn = dev.createBuffer({ size: 4 * n * k, usage: S | D | C });
      this.blockOut = dev.createBuffer({ size: 4 * n * k, usage: S | D | C });
      this.blockRead = dev.createBuffer({ size: 4 * n * k, usage: GPUBufferUsage.MAP_READ | D });
      this.buffers.push(this.blockIn, this.blockOut, this.blockRead);
    }
    // each vector scaled to order one (the V-cycle is linear)
    const scales = R.map((r) => {
      let m = 0;
      for (let i = 0; i < n; i++) if (!fixed[i]) m = Math.max(m, Math.abs(r[i]));
      return m;
    });
    const h = new Float32Array(n * k);
    R.forEach((r, j) => {
      const s = scales[j] > 0 ? 1 / scales[j] : 0;
      for (let i = 0; i < n; i++) h[j * n + i] = fixed[i] ? 0 : r[i] * s;
    });
    dev.queue.writeBuffer(this.blockIn, 0, h);
    const L0 = this.L[0], P = this.p, z0 = L0.buf(L0.out), nN = this.fea.levels[0].nNodes;
    const enc = dev.createCommandEncoder();
    for (let j = 0; j < k; j++) {
      enc.copyBufferToBuffer(this.blockIn, 4 * n * j, L0.r, 0, 4 * n);
      if (iterations) for (const b of [this.x, this.pv, this.qv, this.S]) enc.clearBuffer(b);
      const pass = enc.beginComputePass();
      if (!iterations) this.vcycle(pass, 0);
      for (let s = 0; s < iterations; s++) {
        this.vcycle(pass, 0);
        this.dot(pass, L0.r, z0, RZN);
        this.dot(pass, z0, this.qv, ZQ);
        this.run(pass, P.cg_beta, this.bg.beta, 1);
        this.run(pass, P.update_p, this.bg.updP, n);
        this.run(pass, P.op, this.bg.mvP, nN);
        this.dot(pass, this.pv, this.qv, PQ);
        this.run(pass, P.cg_alpha, this.bg.alpha, 1);
        this.run(pass, P.update_xr, this.bg.updXR, n);
      }
      pass.end();
      enc.copyBufferToBuffer(iterations ? this.x : z0, 0, this.blockOut, 4 * n * j, 4 * n);
    }
    enc.copyBufferToBuffer(this.blockOut, 0, this.blockRead, 0, 4 * n * k);
    dev.queue.submit([enc.finish()]);
    await this.blockRead.mapAsync(GPUMapMode.READ, 0, 4 * n * k);
    const z32 = new Float32Array(this.blockRead.getMappedRange(0, 4 * n * k).slice(0));
    this.blockRead.unmap();
    return R.map((_, j) => {
      const z = new Float64Array(n);
      for (let i = 0; i < n; i++) z[i] = fixed[i] ? 0 : z32[j * n + i] * scales[j];
      return z;
    });
  }

  /**
   * Solve K u = f (normalized units, like VoxelFEA.solve): conjugate gradients preconditioned by
   * the V-cycle on the GPU in float32, kept to float64 accuracy by reliable updates (Sleijpen &
   * van der Vorst 1996): whenever the GPU residual has dropped tenfold since the last fold, its
   * solution is added to the float64 one and the true residual f - K u replaces the GPU's.
   */
  async solve(f, { tol = 1e-6, maxIter = 500, x0 = null, onProgress = null } = {}) {
    const fea = this.fea, L = fea.levels[0], n = L.nDof, fixed = L.fixed;
    const u = x0 ? Float64Array.from(x0) : new Float64Array(n);
    for (let i = 0; i < n; i++) if (fixed[i]) u[i] = 0;
    let bnorm = 0;
    for (let i = 0; i < n; i++) if (!fixed[i]) bnorm += f[i] * f[i];
    bnorm = Math.sqrt(bnorm);
    if (bnorm === 0) return { u, iterations: 0, residual: 0, converged: true, engine: 'GPU' };
    const r = new Float64Array(n), q = new Float64Array(n);
    const trueResidual = () => {
      fea.apply(L, u, q);
      let s = 0;
      for (let i = 0; i < n; i++) { r[i] = fixed[i] ? 0 : f[i] - q[i]; s += r[i] * r[i]; }
      return Math.sqrt(s) / bnorm;
    };
    let rel = trueResidual(), it = 0, cancelled = false;
    if (rel > tol) {
      // float32 units: the first residual scaled to order one
      let rmax = 0;
      for (let i = 0; i < n; i++) rmax = Math.max(rmax, Math.abs(r[i]));
      const scale = 1 / rmax;
      const L0 = this.L[0], P = this.p, z0 = L0.buf(L0.out);
      this.upload(L0.r, r, scale);
      this.clear(this.x, this.pv, this.qv, this.S); // beta = 0 on the first pass
      const foldIn = async () => {
        const x32 = await this.read(this.x, this.xRead, 4 * n);
        for (let i = 0; i < n; i++) u[i] += x32[i] / scale;
        this.clear(this.x);
        return trueResidual();
      };
      let peak = rel, lastEst = rel, relUpdated = rel, batch = 1, dirty = false;
      while (it < maxIter) {
        const t0 = performance.now();
        const enc = this.device.createCommandEncoder();
        const pass = enc.beginComputePass();
        const k = Math.min(batch, maxIter - it);
        for (let s = 0; s < k; s++) {
          // z = M r; beta = max(0, z.(r - r_old) / rz) with r - r_old = -alpha q; p = z + beta p
          this.vcycle(pass, 0);
          this.dot(pass, L0.r, z0, RZN);
          this.dot(pass, z0, this.qv, ZQ);
          this.run(pass, P.cg_beta, this.bg.beta, 1);
          this.run(pass, P.update_p, this.bg.updP, n);
          // q = K p; alpha = rz / p.q; x += alpha p; r -= alpha q
          this.run(pass, P.op, this.bg.mvP, L.nNodes);
          this.dot(pass, this.pv, this.qv, PQ);
          this.run(pass, P.cg_alpha, this.bg.alpha, 1);
          this.run(pass, P.update_xr, this.bg.updXR, n);
        }
        this.dot(pass, L0.r, L0.r, RR);
        pass.end();
        this.device.queue.submit([enc.finish()]);
        const sc = await this.read(this.S, this.readBuf, 64);
        it += k;
        dirty = true;
        if (!(sc[PQ] > 0) || !Number.isFinite(sc[RR]) || !Number.isFinite(sc[ALPHA])) break; // lost positive-definiteness
        const est = Math.sqrt(sc[RR]) / scale / bnorm;
        if (onProgress && onProgress(it, Math.min(est, relUpdated)) === true) { cancelled = true; break; }
        peak = Math.max(peak, est);
        // per-iteration reduction, for sizing the next batch
        const rho = Math.min(0.95, Math.max(0.05, (est / lastEst) ** (1 / k)));
        lastEst = est;
        if (est <= tol || est <= 0.1 * peak) {
          rel = relUpdated = peak = lastEst = await foldIn();
          dirty = false;
          if (rel <= tol) break;
          this.upload(L0.r, r, scale);
        }
        // wait for the GPU about every 8 ms (rarely enough to keep it busy), and not past the next
        // fold or convergence (both only happen between batches)
        const per = (performance.now() - t0) / k;
        // (predicted with convergence speeding up by half again: overshooting a fold lets the float32
        // residual drift from the true one, which costs iterations)
        const toGo = Math.ceil(Math.log(Math.max(tol, 0.1 * peak) / lastEst) / (1.5 * Math.log(rho)));
        batch = Math.max(1, Math.min(8, 2 * k, Math.floor(8 / Math.max(per, 0.01)), toGo)); // growing at most twofold
      }
      if (dirty) rel = await foldIn();
    }
    return { u, iterations: it, residual: rel, converged: !cancelled && rel <= tol * 10, cancelled, engine: 'GPU' };
  }

  destroy() {
    for (const b of this.buffers) b.destroy();
    this.buffers = [];
  }
}
