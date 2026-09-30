// v1 flow engine on the GPU: the same scheme as src/cfd/flow-cpu.js (read that file for the method).
// Shared by the web app (src/cfd/flow-gpu.js) and the native app (native/src/cfd/flow_gpu.cpp).
//
// Before this file come a prelude written by the host (the population buffers, their 16- or 32-bit
// storage and the accessors ld<P>(s, k) / st<P>(s, k, v) for direction pair P: slot P + s of cell k,
// with P = 0 for the rest population) and flow_gen.wgsl (the collision models).
//
// One step is three dispatches over disjoint cells (bulk, wall records, faces), all streaming in
// place: with odd = the step's parity and o = the offset of the neighbour n + c_i (odd i), a cell
// loads    f_i = ld_i(1 - odd, n),  f_{i+1} = ld_i(odd, n + o)
// and stores g_i -> st_i(odd, n + o),  g_{i+1} -> st_i(1 - odd, n),
// and a population from a solid cell is the same load with the parity swapped (bounce-back).

struct Params {
  nx: u32, ny: u32, nz: u32, n: u32,
  nRec: u32, nFace: u32, periodic: u32, odd: u32,
  tau0: f32, smag: f32, uin: f32, uBelt: f32,
  nu0: f32, wallModel: u32, slot: u32, pad: u32,
};
@group(0) @binding(0) var<uniform> P: Params;

override WGX: u32 = 64u;
override WGY: u32 = 1u;
override RR: bool = true; // recursive regularised collision, or plain BGK

const BULK: u32 = 0u;
const WALL: u32 = 1u;
const SOLID: u32 = 2u;
const FACE: u32 = 3u;

const CX: array<i32, 19> = array<i32, 19>(0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0);
const CY: array<i32, 19> = array<i32, 19>(0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1);
const CZ: array<i32, 19> = array<i32, 19>(0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1);
const WT: array<f32, 19> = array<f32, 19>(
  0.3333333333, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0555555556, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778, 0.0277777778);
const OPP: array<u32, 19> = array<u32, 19>(0u, 2u, 1u, 4u, 3u, 6u, 5u, 8u, 7u, 10u, 9u, 12u, 11u, 14u, 13u, 16u, 15u, 18u, 17u);


fn at(n: u32, o: i32) -> u32 { return u32(i32(n) + o); }

// all 19 populations of cell n (the offsets of the odd directions in o)
fn loadCell(n: u32, o: array<i32, 19>, f: ptr<function, array<f32, 19>>) {
  let e = 1u - P.odd;
  let d = P.odd;
  (*f)[0] = ld0(0u, n);
  (*f)[1] = ld1(e, n); (*f)[2] = ld1(d, at(n, o[1]));
  (*f)[3] = ld3(e, n); (*f)[4] = ld3(d, at(n, o[3]));
  (*f)[5] = ld5(e, n); (*f)[6] = ld5(d, at(n, o[5]));
  (*f)[7] = ld7(e, n); (*f)[8] = ld7(d, at(n, o[7]));
  (*f)[9] = ld9(e, n); (*f)[10] = ld9(d, at(n, o[9]));
  (*f)[11] = ld11(e, n); (*f)[12] = ld11(d, at(n, o[11]));
  (*f)[13] = ld13(e, n); (*f)[14] = ld13(d, at(n, o[13]));
  (*f)[15] = ld15(e, n); (*f)[16] = ld15(d, at(n, o[15]));
  (*f)[17] = ld17(e, n); (*f)[18] = ld17(d, at(n, o[17]));
}

// its post-collision populations, into the slots it loaded from; skip[i]: slot outside the tunnel
fn storeCell(n: u32, o: array<i32, 19>, f: ptr<function, array<f32, 19>>, out: u32) {
  let e = 1u - P.odd;
  let d = P.odd;
  st0(0u, n, (*f)[0]);
  if ((out & 2u) == 0u) { st1(d, at(n, o[1]), (*f)[1]); } st1(e, n, (*f)[2]);
  if ((out & 8u) == 0u) { st3(d, at(n, o[3]), (*f)[3]); } st3(e, n, (*f)[4]);
  if ((out & 32u) == 0u) { st5(d, at(n, o[5]), (*f)[5]); } st5(e, n, (*f)[6]);
  if ((out & 128u) == 0u) { st7(d, at(n, o[7]), (*f)[7]); } st7(e, n, (*f)[8]);
  if ((out & 512u) == 0u) { st9(d, at(n, o[9]), (*f)[9]); } st9(e, n, (*f)[10]);
  if ((out & 2048u) == 0u) { st11(d, at(n, o[11]), (*f)[11]); } st11(e, n, (*f)[12]);
  if ((out & 8192u) == 0u) { st13(d, at(n, o[13]), (*f)[13]); } st13(e, n, (*f)[14]);
  if ((out & 32768u) == 0u) { st15(d, at(n, o[15]), (*f)[15]); } st15(e, n, (*f)[16]);
  if ((out & 131072u) == 0u) { st17(d, at(n, o[17]), (*f)[17]); } st17(e, n, (*f)[18]);
}

// offsets of the neighbours n + c_i (odd i) in layer z; the span wraps around when periodic
fn offsets(z: u32) -> array<i32, 19> {
  let nx = i32(P.nx);
  let sxy = i32(P.nx * P.ny);
  var zp = sxy;
  var zm = -sxy;
  if (P.periodic != 0u) {
    if (z + 1u == P.nz) { zp = sxy * (1 - i32(P.nz)); }
    if (z == 0u) { zm = sxy * (i32(P.nz) - 1); }
  }
  var o: array<i32, 19>;
  o[1] = 1; o[3] = nx; o[5] = zp; o[7] = 1 + nx; o[9] = 1 - nx;
  o[11] = 1 + zp; o[13] = 1 + zm; o[15] = nx + zp; o[17] = nx + zm;
  return o;
}

fn kindOf(n: u32) -> u32 { return (cells[n >> 2u] >> ((n & 3u) * 8u)) & 0xffu; }

fn collide(f: ptr<function, array<f32, 19>>, tauMin: f32) -> vec4<f32> {
  if (RR) { return collide_rr(f, P.tau0, P.smag, tauMin); }
  return collide_bgk(f, P.tau0, P.smag, tauMin);
}

// ---------- start: every population at rest ----------

@compute @workgroup_size(64)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n = gid.x + gid.y * 65535u * 64u;
  if (n >= P.n) { return; }
  st0(0u, n, WT[0]);
  st1(0u, n, WT[1]); st1(1u, n, WT[2]);
  st3(0u, n, WT[3]); st3(1u, n, WT[4]);
  st5(0u, n, WT[5]); st5(1u, n, WT[6]);
  st7(0u, n, WT[7]); st7(1u, n, WT[8]);
  st9(0u, n, WT[9]); st9(1u, n, WT[10]);
  st11(0u, n, WT[11]); st11(1u, n, WT[12]);
  st13(0u, n, WT[13]); st13(1u, n, WT[14]);
  st15(0u, n, WT[15]); st15(1u, n, WT[16]);
  st17(0u, n, WT[17]); st17(1u, n, WT[18]);
}

// ---------- bulk fluid ----------

@group(0) @binding(10) var<storage, read> cells: array<u32>; // kinds, 4 per word

@compute @workgroup_size(WGX, WGY, 1)
fn bulk(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let z = gid.z;
  if (x < 1u || x + 1u >= P.nx || y >= P.ny || z >= P.nz) { return; }
  let n = x + P.nx * (y + P.ny * z);
  if (kindOf(n) != BULK) { return; }
  let o = offsets(z);
  var f: array<f32, 19>;
  loadCell(n, o, &f);
  _ = collide(&f, 0.0);
  storeCell(n, o, &f, 0u);
}

// ---------- wall cells ----------
// record r (REC words): cell, mask (bit k: link k from a solid cell; bit 0: the nearest wall is the
// moving ground), ground mask, q bytes (links 1-18), normal xy, normal z + distance (16-bit pairs),
// aux (the last step's outgoing populations 1-18), the force and density accumulators, the wall
// model's filtered wall velocity and bb (the last step's outgoing populations toward the walls, for
// bounce-back: across a thin wall the other side is fluid, so its slots are not free to read), f32 bits.
const REC: u32 = 59u;
const R_SLIP: u32 = 33u;
const R_BB: u32 = 36u;
const R_SAMP: u32 = 54u; // the wall model's sample direction (0: none), its distance y2, the area vector
const R_Y2: u32 = 55u;
const R_AREA: u32 = 56u;
const SLIP_FILTER: f32 = 0.05;
const R_Q: u32 = 3u;
const R_N: u32 = 8u;
const R_AUX: u32 = 10u;
const R_ACC: u32 = 28u;
const R_RHO: u32 = 32u;
@group(0) @binding(11) var<storage, read_write> rec: array<u32>;

fn qOf(r: u32, k: u32) -> f32 {
  let b = (rec[r * REC + R_Q + ((k - 1u) >> 2u)] >> (((k - 1u) & 3u) * 8u)) & 0xffu;
  return select(-1.0, f32(b - 1u) / 254.0, b != 0u);
}
fn auxOf(r: u32, k: u32) -> f32 { return bitcast<f32>(rec[r * REC + R_AUX + k - 1u]); }

const KAPPA: f32 = 0.41;
const EKB: f32 = 0.11836; // exp(-0.41 * 5.2)

// friction velocity from Spalding's law of the wall (flow.js spaldingUtau)
fn spalding(ut: f32, y: f32, nu: f32) -> f32 {
  if (ut <= 0.0 || y <= 0.0) { return 0.0; }
  var utau = sqrt(nu * ut / y);
  if (utau * y / nu > 11.0) { utau = ut / (log(y * ut / nu) / KAPPA + 5.2 - 1.5); }
  for (var it = 0; it < 8; it++) {
    let up = ut / utau;
    let k = KAPPA * up;
    let e = exp(min(k, 60.0));
    let g = up + EKB * (e - 1.0 - k - k * k / 2.0 - k * k * k / 6.0) - y * utau / nu;
    let dgdup = 1.0 + EKB * KAPPA * (e - 1.0 - k - k * k / 2.0);
    let dd = dgdup * (-up / utau) - y / nu;
    let next = utau - g / dd;
    utau = select(0.5 * utau, next, next > 0.0);
    if (abs(g) < 1e-5 * (1.0 + y * utau / nu)) { break; }
  }
  return utau;
}

// Reichardt's law of the wall (flow.js reichardt): (u+, du+/dy+) at y+
fn reichardt(yp: f32) -> vec2<f32> {
  let e11 = exp(-yp / 11.0);
  let e3 = exp(-yp / 3.0);
  return vec2<f32>(log(1.0 + KAPPA * yp) / KAPPA + 7.8 * (1.0 - e11 - (yp / 11.0) * e3),
                   1.0 / (1.0 + KAPPA * yp) + 7.8 * (e11 / 11.0 - e3 / 11.0 + (yp / 33.0) * e3));
}

// friction velocity with u_tau u+(y u_tau / nu) = ut (flow.js reichardtUtau)
fn reichardtUtau(ut: f32, y: f32, nu: f32) -> f32 {
  if (ut <= 0.0 || y <= 0.0) { return 0.0; }
  var utau = max(sqrt(nu * ut / y), ut / 30.0);
  for (var it = 0; it < 16; it++) {
    let yp = y * utau / nu;
    let r = reichardt(yp);
    let F = utau * r.x - ut;
    let next = utau - F / (r.x + yp * r.y);
    utau = select(0.5 * utau, next, next > 0.0);
    if (abs(F) < 1e-6 * ut) { break; }
  }
  return utau;
}

// post-collision populations of the regularised state at rho, u with non-equilibrium stress pi
// (xx, yy, zz, xy, xz, yz): feq + k W (H2 : pi / (2 cs^4) + recursive third-order terms)
fn regularized(f: ptr<function, array<f32, 19>>, rho: f32, u: vec3<f32>, pi: array<f32, 6>, k: f32) {
  if (RR) { equilibrium_rr(f, rho, u.x, u.y, u.z); } else { equilibrium_bgk(f, rho, u.x, u.y, u.z); }
  let tr = (pi[0] + pi[1] + pi[2]) / 3.0;
  var a1: array<f32, 6>;
  if (RR) {
    let xxy = 2.0 * u.x * pi[3] + u.y * pi[0];
    let yzz = 2.0 * u.z * pi[5] + u.y * pi[2];
    let xzz = 2.0 * u.z * pi[4] + u.x * pi[2];
    let xyy = 2.0 * u.y * pi[3] + u.x * pi[1];
    let yyz = 2.0 * u.y * pi[5] + u.z * pi[1];
    let xxz = 2.0 * u.x * pi[4] + u.z * pi[0];
    a1 = array<f32, 6>(xxy + yzz, xzz + xyy, yyz + xxz, xxy - yzz, xzz - xyy, yyz - xxz);
  }
  for (var i = 0u; i < 19u; i++) {
    let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
    let h2 = c.x * c.x * pi[0] + c.y * c.y * pi[1] + c.z * c.z * pi[2] + 2.0 * (c.x * c.y * pi[3] + c.x * c.z * pi[4] + c.y * c.z * pi[5]) - tr;
    var t3 = 0.0;
    if (RR) { for (var j = 0u; j < 6u; j++) { t3 += P3C[i][j] * a1[j]; } }
    (*f)[i] += k * WT[i] * (4.5 * h2 + t3);
  }
}

// the Smagorinsky relaxation time of populations f at rho, u (the collision's, without colliding)
fn smagorinskyTau(f: ptr<function, array<f32, 19>>, rho: f32, u: vec3<f32>) -> f32 {
  var pxx = 0.0; var pyy = 0.0; var pzz = 0.0; var pxy = 0.0; var pxz = 0.0; var pyz = 0.0;
  for (var i = 0u; i < 19u; i++) {
    let v = (*f)[i];
    let c = vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i]));
    pxx += c.x * c.x * v; pyy += c.y * c.y * v; pzz += c.z * c.z * v;
    pxy += c.x * c.y * v; pxz += c.x * c.z * v; pyz += c.y * c.z * v;
  }
  let nxx = pxx - rho * (u.x * u.x + 1.0 / 3.0);
  let nyy = pyy - rho * (u.y * u.y + 1.0 / 3.0);
  let nzz = pzz - rho * (u.z * u.z + 1.0 / 3.0);
  let nxy = pxy - rho * u.x * u.y;
  let nxz = pxz - rho * u.x * u.z;
  let nyz = pyz - rho * u.y * u.z;
  let q = sqrt(nxx * nxx + nyy * nyy + nzz * nzz + 2.0 * (nxy * nxy + nxz * nxz + nyz * nyz));
  return 0.5 * (P.tau0 + sqrt(P.tau0 * P.tau0 + P.smag * q / rho));
}

// Wall model (flow-cpu.js wallModelCell, Malaspinas & Sagaut 2014): sample the flow at the bulk cell
// n + c_samp, take u_tau from Reichardt's law, and rebuild this cell's populations as the regularised
// state with the law's velocity and shear at its own distance from the wall.
fn modelCell(r: u32, n: u32, z: u32, o: array<i32, 19>, mask: u32, j: u32) {
  let base = r * REC;
  // the sample cell (its z wraps around with a periodic span)
  var zm = i32(z) + CZ[j];
  if (P.periodic != 0u) { zm = (zm + i32(P.nz)) % i32(P.nz); }
  let m = u32(i32(n) + CX[j] + i32(P.nx) * CY[j] + i32(P.nx * P.ny) * (zm - i32(z)));
  let mo = storedMoments(m);
  let rho2 = mo.x;
  let nxy = unpack2x16float(rec[base + R_N]);
  let nzd = unpack2x16float(rec[base + R_N + 1u]);
  let nrm = vec3<f32>(nxy.x, nxy.y, nzd.x);
  let y1 = nzd.y;
  let y2 = bitcast<f32>(rec[base + R_Y2]);
  var wv = vec3<f32>(0.0);
  if ((mask & 1u) != 0u) { wv.x = P.uBelt; }
  let u2 = mo.yzw / rho2 - wv;
  let un2 = dot(u2, nrm);
  // tangential velocity, low-passed in time
  let prev = vec3<f32>(bitcast<f32>(rec[base + R_SLIP]), bitcast<f32>(rec[base + R_SLIP + 1u]), bitcast<f32>(rec[base + R_SLIP + 2u]));
  let tv = prev + SLIP_FILTER * (u2 - un2 * nrm - prev);
  rec[base + R_SLIP] = bitcast<u32>(tv.x);
  rec[base + R_SLIP + 1u] = bitcast<u32>(tv.y);
  rec[base + R_SLIP + 2u] = bitcast<u32>(tv.z);
  let ut2 = length(tv);
  var utau = 0.0; var ut1 = 0.0; var dudn = 0.0; var tauN = 0.5 + 3.0 * P.nu0;
  var e = vec3<f32>(0.0);
  if (ut2 > 1e-12) {
    utau = reichardtUtau(ut2, y2, P.nu0);
    let r1 = reichardt(y1 * utau / P.nu0);
    ut1 = utau * r1.x;
    dudn = utau * utau / P.nu0 * r1.y;
    tauN = 0.5 + 3.0 * P.nu0 / r1.y;
    e = tv / ut2;
  }
  let u1 = wv + ut1 * e + (un2 * y1 / y2) * nrm;
  let c = -rho2 * tauN * dudn / 3.0;
  let pi = array<f32, 6>(2.0 * c * e.x * nrm.x, 2.0 * c * e.y * nrm.y, 2.0 * c * e.z * nrm.z,
    c * (e.x * nrm.y + e.y * nrm.x), c * (e.x * nrm.z + e.z * nrm.x), c * (e.y * nrm.z + e.z * nrm.y));
  var f: array<f32, 19>;
  regularized(&f, rho2, u1, pi, 1.0 - 1.0 / tauN);
  storeCell(n, o, &f, 0u);
  for (var k = 1u; k < 19u; k++) {
    if ((mask & (1u << k)) != 0u) {
      rec[base + R_AUX + k - 1u] = bitcast<u32>(f[k]);
      rec[base + R_BB + k - 1u] = bitcast<u32>(f[OPP[k]]);
    }
  }
  // force on the part: pressure on the surface next to the cell, and the wall shear along the flow
  let av = vec3<f32>(bitcast<f32>(rec[base + R_AREA]), bitcast<f32>(rec[base + R_AREA + 1u]), bitcast<f32>(rec[base + R_AREA + 2u]));
  let p = (rho2 - 1.0) / 3.0;
  let fm = -p * av + rho2 * utau * utau * length(av) * e;
  let acc = vec4<f32>(bitcast<f32>(rec[base + R_ACC]), bitcast<f32>(rec[base + R_ACC + 1u]), bitcast<f32>(rec[base + R_ACC + 2u]), bitcast<f32>(rec[base + R_ACC + 3u]))
    + vec4<f32>(fm, rho2 - 1.0);
  rec[base + R_ACC] = bitcast<u32>(acc.x);
  rec[base + R_ACC + 1u] = bitcast<u32>(acc.y);
  rec[base + R_ACC + 2u] = bitcast<u32>(acc.z);
  rec[base + R_ACC + 3u] = bitcast<u32>(acc.w);
}

@compute @workgroup_size(64)
fn wall(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x + gid.y * 65535u * 64u;
  if (r >= P.nRec) { return; }
  let base = r * REC;
  let n = rec[base];
  let mask = rec[base + 1u];
  let gmask = rec[base + 2u];
  let z = n / (P.nx * P.ny);
  let o = offsets(z);
  let samp = rec[base + R_SAMP];
  if (P.wallModel == 4u && samp != 0u) { modelCell(r, n, z, o, mask, samp); return; }
  var f: array<f32, 19>;
  loadCell(n, o, &f);
  var fb: array<f32, 19>;
  for (var k = 1u; k < 19u; k++) {
    if ((mask & (1u << k)) != 0u) { fb[k] = bitcast<f32>(rec[base + R_BB + k - 1u]); }
  }
  // interpolated bounce-back (the part is at rest, the ground belt moves); fq: moving-wall factor
  var fq: array<f32, 19>;
  for (var k = 1u; k < 19u; k++) {
    if ((mask & (1u << k)) == 0u) { continue; }
    let v = fb[k];
    var fk = v;
    fq[k] = 6.0;
    if ((gmask & (1u << k)) != 0u) {
      fk = v + 6.0 * WT[k] * f32(CX[k]) * P.uBelt;
    } else {
      let q = qOf(r, k);
      if (q >= 0.0) {
        if (q < 0.5) { fk = 2.0 * q * v + (1.0 - 2.0 * q) * f[OPP[k]]; }
        else { fk = (0.5 / q) * v + (1.0 - 0.5 / q) * auxOf(r, k); fq[k] = 3.0 / q; }
      }
    }
    f[k] = fk;
  }
  // wall model (flow-cpu.js wallCells): 1 = slip, the wall moves along the near-wall flow at the
  // speed that gives the log law's wall shear with the cell's own viscosity; 2 = the log-law eddy
  // viscosity sets the cell's relaxation; 3 = at least that viscosity
  var w = vec3<f32>(0.0);
  var tauMin = 0.0;
  var tauWall = 0.0;
  if (P.wallModel != 0u) {
    var rho = 0.0;
    var j = vec3<f32>(0.0);
    for (var i = 0u; i < 19u; i++) { rho += f[i]; j += vec3<f32>(f32(CX[i]), f32(CY[i]), f32(CZ[i])) * f[i]; }
    let nxy = unpack2x16float(rec[base + R_N]);
    let nzd = unpack2x16float(rec[base + R_N + 1u]);
    let nrm = vec3<f32>(nxy.x, nxy.y, nzd.x);
    let d = nzd.y;
    let u = j / rho;
    var ur = u;
    if ((mask & 1u) != 0u) { ur.x -= P.uBelt; }
    let tv = ur - dot(ur, nrm) * nrm;
    let ut = length(tv);
    let utau = spalding(ut, d, P.nu0);
    if (ut > 1e-9 && utau > 0.0) {
      if (P.wallModel == 1u) {
        let nuEff = (smagorinskyTau(&f, rho, u) - 0.5) / 3.0;
        let us = min(ut, max(0.0, ut - utau * utau * d / nuEff));
        // low-pass in time (flow-cpu.js wallCells)
        let prev = vec3<f32>(bitcast<f32>(rec[base + R_SLIP]), bitcast<f32>(rec[base + R_SLIP + 1u]), bitcast<f32>(rec[base + R_SLIP + 2u]));
        let sl = prev + SLIP_FILTER * (tv * (us / ut) - prev);
        rec[base + R_SLIP] = bitcast<u32>(sl.x);
        rec[base + R_SLIP + 1u] = bitcast<u32>(sl.y);
        rec[base + R_SLIP + 2u] = bitcast<u32>(sl.z);
        // along the current flow only, and never faster than it
        w = tv * (clamp(dot(sl, tv) / ut, 0.0, ut) / ut);
        for (var k = 1u; k < 19u; k++) {
          if ((mask & (1u << k)) != 0u) { f[k] += fq[k] * WT[k] * rho * dot(vec3<f32>(f32(CX[k]), f32(CY[k]), f32(CZ[k])), w); }
        }
      } else {
        let nuT = max(utau * utau * d / ut - P.nu0, 0.0);
        tauWall = 3.0 * (P.nu0 + nuT) + 0.5;
        if (P.wallModel == 3u) { tauMin = tauWall; tauWall = 0.0; }
      }
    }
  }
  // momentum to the part (Galilean invariant for a moving wall, Wen et al. 2014), c_opp = -c_k
  var m = vec3<f32>(0.0);
  for (var k = 1u; k < 19u; k++) {
    if ((mask & (1u << k)) == 0u || (gmask & (1u << k)) != 0u) { continue; }
    let v = fb[k];
    m -= vec3<f32>(f32(CX[k]), f32(CY[k]), f32(CZ[k])) * (v + f[k]) + w * (v - f[k]);
  }
  var mac: vec4<f32>;
  if (tauWall > 0.0) {
    if (RR) { mac = collide_rr(&f, tauWall, 0.0, 0.0); } else { mac = collide_bgk(&f, tauWall, 0.0, 0.0); }
  } else { mac = collide(&f, tauMin); }
  storeCell(n, o, &f, 0u);
  // for the next step: the outgoing populations away from the walls (interpolation) and toward them
  // (bounce-back)
  for (var k = 1u; k < 19u; k++) {
    if ((mask & (1u << k)) != 0u) {
      rec[base + R_AUX + k - 1u] = bitcast<u32>(f[k]);
      rec[base + R_BB + k - 1u] = bitcast<u32>(f[OPP[k]]);
    }
  }
  let acc = vec4<f32>(bitcast<f32>(rec[base + R_ACC]), bitcast<f32>(rec[base + R_ACC + 1u]), bitcast<f32>(rec[base + R_ACC + 2u]), bitcast<f32>(rec[base + R_ACC + 3u]))
    + vec4<f32>(m, mac.x - 1.0);
  rec[base + R_ACC] = bitcast<u32>(acc.x);
  rec[base + R_ACC + 1u] = bitcast<u32>(acc.y);
  rec[base + R_ACC + 2u] = bitcast<u32>(acc.z);
  rec[base + R_ACC + 3u] = bitcast<u32>(acc.w);
}

// density and momentum of cell m from where its post-collision populations of this step now live
// (m must already have run this step)
fn storedMoments(m: u32) -> vec4<f32> {
  let o0 = offsets(m / (P.nx * P.ny));
  let e = 1u - P.odd;
  let d = P.odd;
  var rho = ld0(0u, m);
  var jv = vec3<f32>(0.0);
  var a: f32; var b: f32;
  a = ld1(d, at(m, o0[1])); b = ld1(e, m); rho += a + b; jv += vec3<f32>(1.0, 0.0, 0.0) * (a - b);
  a = ld3(d, at(m, o0[3])); b = ld3(e, m); rho += a + b; jv += vec3<f32>(0.0, 1.0, 0.0) * (a - b);
  a = ld5(d, at(m, o0[5])); b = ld5(e, m); rho += a + b; jv += vec3<f32>(0.0, 0.0, 1.0) * (a - b);
  a = ld7(d, at(m, o0[7])); b = ld7(e, m); rho += a + b; jv += vec3<f32>(1.0, 1.0, 0.0) * (a - b);
  a = ld9(d, at(m, o0[9])); b = ld9(e, m); rho += a + b; jv += vec3<f32>(1.0, -1.0, 0.0) * (a - b);
  a = ld11(d, at(m, o0[11])); b = ld11(e, m); rho += a + b; jv += vec3<f32>(1.0, 0.0, 1.0) * (a - b);
  a = ld13(d, at(m, o0[13])); b = ld13(e, m); rho += a + b; jv += vec3<f32>(1.0, 0.0, -1.0) * (a - b);
  a = ld15(d, at(m, o0[15])); b = ld15(e, m); rho += a + b; jv += vec3<f32>(0.0, 1.0, 1.0) * (a - b);
  a = ld17(d, at(m, o0[17])); b = ld17(e, m); rho += a + b; jv += vec3<f32>(0.0, 1.0, -1.0) * (a - b);
  return vec4<f32>(rho, jv);
}

// ---------- tunnel faces ----------
// face j (3 words): cell; the nearest interior cell (0xffffffff at the inlet, 0xfffffffe when it is
// solid); directions that leave the tunnel.
@group(0) @binding(12) var<storage, read> faces: array<u32>;

@compute @workgroup_size(64)
fn face(@builtin(global_invocation_id) gid: vec3<u32>) {
  let j = gid.x + gid.y * 65535u * 64u;
  if (j >= P.nFace) { return; }
  let n = faces[3u * j];
  let n0 = faces[3u * j + 1u];
  let out = faces[3u * j + 2u];
  var u = vec3<f32>(P.uin, 0.0, 0.0);
  if (n0 < 0xfffffffeu) {
    // velocity of the nearest interior cell, from where its post-collision populations now live
    let mo = storedMoments(n0);
    u = mo.yzw / mo.x;
  }
  var f: array<f32, 19>;
  if (RR) { equilibrium_rr(&f, 1.0, u.x, u.y, u.z); } else { equilibrium_bgk(&f, 1.0, u.x, u.y, u.z); }
  storeCell(n, offsets(n / (P.nx * P.ny)), &f, out);
}

// ---------- forces: deterministic two-pass sums over the wall records ----------
// pass 1: each workgroup sums a fixed share of the records in a fixed order into partial[wg]
// (momentum exchange xyz, pressure integral xyz, friction xyz: each record's force along its wall),
// adds each record's density sum to its long-time total and clears its accumulators; pass 2 adds
// the partials in order into history[slot] (12 floats a slot).
@group(0) @binding(13) var<storage, read_write> partial: array<f32>;
@group(0) @binding(14) var<storage, read_write> history: array<f32>;
const SUMS: u32 = 9u;
var<workgroup> red: array<array<f32, 9>, 256>;

@compute @workgroup_size(256)
fn reduce1(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  var s: array<f32, 9>;
  let stride = nwg.x * 256u;
  for (var r = wid.x * 256u + lid.x; r < P.nRec; r += stride) {
    let base = r * REC;
    let mask = rec[base + 1u] & ~rec[base + 2u];
    let fm = vec3<f32>(bitcast<f32>(rec[base + R_ACC]), bitcast<f32>(rec[base + R_ACC + 1u]), bitcast<f32>(rec[base + R_ACC + 2u]));
    s[0] += fm.x; s[1] += fm.y; s[2] += fm.z;
    let drho = bitcast<f32>(rec[base + R_ACC + 3u]);
    // pressure (rho - 1) c_s^2 on the voxel faces toward the part (axis links 1-6)
    let p = drho / 3.0;
    if ((mask & 2u) != 0u) { s[3] -= p; }
    if ((mask & 4u) != 0u) { s[3] += p; }
    if ((mask & 8u) != 0u) { s[4] -= p; }
    if ((mask & 16u) != 0u) { s[4] += p; }
    if ((mask & 32u) != 0u) { s[5] -= p; }
    if ((mask & 64u) != 0u) { s[5] += p; }
    let nxy = unpack2x16float(rec[base + R_N]);
    let nrm = vec3<f32>(nxy.x, nxy.y, unpack2x16float(rec[base + R_N + 1u]).x);
    let ft = fm - dot(fm, nrm) * nrm;
    s[6] += ft.x; s[7] += ft.y; s[8] += ft.z;
    rec[base + R_RHO] = bitcast<u32>(bitcast<f32>(rec[base + R_RHO]) + drho);
    rec[base + R_ACC] = 0u; rec[base + R_ACC + 1u] = 0u; rec[base + R_ACC + 2u] = 0u; rec[base + R_ACC + 3u] = 0u;
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (lid.x < h) { for (var c = 0u; c < SUMS; c++) { red[lid.x][c] += red[lid.x + h][c]; } }
    workgroupBarrier();
  }
  if (lid.x == 0u) { for (var c = 0u; c < SUMS; c++) { partial[wid.x * SUMS + c] = red[0][c]; } }
}

@compute @workgroup_size(1)
fn reduce2() {
  var s: array<f32, 9>;
  let groups = u32(partial[arrayLength(&partial) - 1u]);
  for (var g = 0u; g < groups; g++) { for (var c = 0u; c < SUMS; c++) { s[c] += partial[g * SUMS + c]; } }
  for (var c = 0u; c < SUMS; c++) { history[P.slot * 12u + c] = s[c]; }
}


// ---------- reduced fields for the view (and the saved results): coarse cells of CF^3 cells ----------
// coarse[i] = this sample's mean (rho - 1, u) over the bulk cells in coarse cell i (x = -2: none),
// coarse[nc + i] += the same (the time average is that sum over the number of samples). Run between
// steps with the next step's parameters (its loads find each cell's populations).
override CF: u32 = 1u;
@group(0) @binding(15) var<storage, read_write> coarse: array<vec4<f32>>;

@compute @workgroup_size(64)
fn sample(@builtin(global_invocation_id) gid: vec3<u32>) {
  let cnx = (P.nx + CF - 1u) / CF;
  let cny = (P.ny + CF - 1u) / CF;
  let cnz = (P.nz + CF - 1u) / CF;
  let nc = cnx * cny * cnz;
  let i = gid.x + gid.y * 65535u * 64u;
  if (i >= nc) { return; }
  let cx = i % cnx;
  let cy = (i / cnx) % cny;
  let cz = i / (cnx * cny);
  var sum = vec4<f32>(0.0);
  var count = 0.0;
  for (var dz = 0u; dz < CF; dz++) {
    let z = cz * CF + dz;
    if (z >= P.nz) { break; }
    let o = offsets(z);
    for (var dy = 0u; dy < CF; dy++) {
      let y = cy * CF + dy;
      if (y >= P.ny) { break; }
      for (var dx = 0u; dx < CF; dx++) {
        let x = cx * CF + dx;
        if (x >= P.nx) { break; }
        let n = x + P.nx * (y + P.ny * z);
        if (kindOf(n) != BULK) { continue; }
        var f: array<f32, 19>;
        loadCell(n, o, &f);
        var rho = 0.0;
        var j = vec3<f32>(0.0);
        for (var k = 0u; k < 19u; k++) { rho += f[k]; j += vec3<f32>(f32(CX[k]), f32(CY[k]), f32(CZ[k])) * f[k]; }
        sum += vec4<f32>(rho - 1.0, j / rho);
        count += 1.0;
      }
    }
  }
  if (count > 0.0) {
    let v = sum / count;
    coarse[i] = v;
    coarse[nc + i] += v;
  } else {
    coarse[i] = vec4<f32>(-2.0, 0.0, 0.0, 0.0);
  }
}

// start the wall records' long-time density sums again
@compute @workgroup_size(64)
fn clearRho(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x + gid.y * 65535u * 64u;
  if (r < P.nRec) { rec[r * REC + R_RHO] = 0u; }
}
