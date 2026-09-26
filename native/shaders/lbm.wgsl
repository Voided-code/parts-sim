// D3Q19 lattice Boltzmann (from src/cfd/lbm-gpu.js; workgroup size 64)

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
  0.3333333333, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778);

fn feq(i: u32, rho: f32, u: vec3<f32>, usq: f32) -> f32 {
  let cu = f32(CX[i]) * u.x + f32(CY[i]) * u.y + f32(CZ[i]) * u.z;
  return WT[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - usq);
}

fn cellIndex(gid: vec3<u32>) -> u32 { return gid.x + gid.y * P.stride; }

@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = cellIndex(gid);
  if (c >= P.n) { return; }
  for (var i = 0u; i < 19u; i++) { fout[i * P.n + c] = WT[i]; }
  moments[c] = vec4<f32>(1.0, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(64)
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
