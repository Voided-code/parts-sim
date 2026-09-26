// Multigrid-preconditioned CG for the voxel FEA (from src/fea/gpu-solver.js; workgroup size 64)

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
  vec3<u32>(0u, 0u, 0u), vec3<u32>(1u, 0u, 0u), vec3<u32>(1u, 1u, 0u), vec3<u32>(0u, 1u, 0u), vec3<u32>(0u, 0u, 1u), vec3<u32>(1u, 0u, 1u), vec3<u32>(1u, 1u, 1u), vec3<u32>(0u, 1u, 1u));

fn corner(x: u32, y: u32, z: u32) -> u32 {
  let c = select(select(0u, 1u, x == 1u), select(3u, 2u, x == 1u), y == 1u);
  return c + 4u * z;
}

@compute @workgroup_size(64)
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

@compute @workgroup_size(64)
fn jacobi_first(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = L.omega * invD[q] * rhs[q];
}

@compute @workgroup_size(64)
fn jacobi(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = vout[q] + L.omega * invD[q] * (rhs[q] - vin[q]);
}

@compute @workgroup_size(64)
fn resid(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = select(0.0, rhs[q] - vout[q], invD[q] != 0.0);
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

// coarsest level: z = A^-1 r over the free DOFs (emap = free DOF list, edata = dense inverse)
@compute @workgroup_size(64)
fn coarsest(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x + gid.y * L.strideD;
  if (row >= L.m) { return; }
  var s = 0.0;
  for (var c = 0u; c < L.m; c++) { s += edata[row * L.m + c] * vin[u32(emap[c])]; }
  vout[u32(emap[row])] = s;
}

@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = vin[q];
}

@compute @workgroup_size(64)
fn init_r(@builtin(global_invocation_id) gid: vec3<u32>) {
  let q = gid.x + gid.y * L.strideD;
  if (q >= L.nDof) { return; }
  vout[q] = select(0.0, rhs[q] - vin[q], invD[q] != 0.0);
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

@compute @workgroup_size(1)
fn cg_beta() {
  S[3] = select(0.0, S[5] / S[0], S[0] != 0.0);
  S[0] = S[5];
}
