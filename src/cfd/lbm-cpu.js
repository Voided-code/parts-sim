// D3Q19 lattice-Boltzmann flow solver (CPU reference implementation / fallback).
//
// - BGK collision with a Smagorinsky sub-grid model, so moderately high Reynolds numbers
//   stay stable on coarse grids (a simple LES).
// - Wind blows along +x. The inlet (x = 0) imposes the free stream. The outlet and the four
//   side faces are open boundaries: equilibrium at ambient pressure (rho = 1) with the
//   velocity of the nearest interior cell. Air displaced by the part can leave sideways as it
//   would in open air, instead of being squeezed as in a closed tunnel (blockage).
// - Walls are no-slip. With `links` (see links.js) the wall sits at its true position on
//   each lattice link (Bouzidi interpolated bounce-back); without it, half-way bounce-back.
// Layout: f[i * N + cell], cell = x + nx * (y + ny * z). Output macro: [rho, ux, uy, uz] per cell.
//
// lbm-gpu.js implements exactly the same scheme in WGSL.

export const CX = [0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0];
export const CY = [0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1];
export const CZ = [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1];
export const W = [1 / 3, ...Array(6).fill(1 / 18), ...Array(12).fill(1 / 36)];
export const OPP = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17];

export const RAMP_STEPS = 400;

/** Relaxation parameters shared by the CPU and GPU solvers. */
export function lbmParams({ uLat, nuLat, smagorinsky = 0.16 }) {
  if (!Number.isFinite(uLat) || uLat < 0 || uLat >= 0.3) throw new Error('Lattice speed must be finite and between 0 and 0.3.');
  if (!Number.isFinite(nuLat) || nuLat <= 0) throw new Error('Lattice viscosity must be finite and positive.');
  if (!Number.isFinite(smagorinsky) || smagorinsky < 0) throw new Error('Smagorinsky constant must be finite and non-negative.');
  return { uLat, tau0: 3 * nuLat + 0.5, smag: 18 * Math.SQRT2 * smagorinsky * smagorinsky };
}

export function validateGrid({ dims, solid }) {
  if (!Array.isArray(dims) || dims.length !== 3 || dims.some((n) => !Number.isSafeInteger(n) || n < 3)) {
    throw new Error('The flow grid needs three integer dimensions of at least 3 cells.');
  }
  const cells = dims[0] * dims[1] * dims[2];
  if (!Number.isSafeInteger(cells) || solid?.length !== cells) throw new Error('Solid mask does not match the flow grid.');
  return cells;
}

export function inletVelocity(uLat, step) {
  return uLat * Math.min(1, (step + 1) / RAMP_STEPS);
}

// Cells away from walls and the domain boundary: the same pull-and-collide step without branches or
// per-direction table lookups, unrolled over the directions. src[i] = i N - off[i], so f[src[i] + c]
// is the population arriving at c.
function collideRun(f, g, macro, N, src, c0, c1, tau0, smag, writeMacro) {
  const [s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15, s16, s17, s18] = src;
  const w0 = W[0], w1 = W[1], w2 = W[7];
  for (let c = c0; c < c1; c++) {
    const f0 = f[s0 + c];
    const f1 = f[s1 + c];
    const f2 = f[s2 + c];
    const f3 = f[s3 + c];
    const f4 = f[s4 + c];
    const f5 = f[s5 + c];
    const f6 = f[s6 + c];
    const f7 = f[s7 + c];
    const f8 = f[s8 + c];
    const f9 = f[s9 + c];
    const f10 = f[s10 + c];
    const f11 = f[s11 + c];
    const f12 = f[s12 + c];
    const f13 = f[s13 + c];
    const f14 = f[s14 + c];
    const f15 = f[s15 + c];
    const f16 = f[s16 + c];
    const f17 = f[s17 + c];
    const f18 = f[s18 + c];
    const rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18;
    const ux = (f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14) / rho;
    const uy = (f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18) / rho;
    const uz = (f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18) / rho;
    const usq = 1.5 * (ux * ux + uy * uy + uz * uz);
    const d0 = f0 - w0 * rho * (1 - usq);
    const cu1 = ux;
    const d1 = f1 - w1 * rho * (1 + 3 * cu1 + 4.5 * cu1 * cu1 - usq);
    const cu2 = -ux;
    const d2 = f2 - w1 * rho * (1 + 3 * cu2 + 4.5 * cu2 * cu2 - usq);
    const cu3 = uy;
    const d3 = f3 - w1 * rho * (1 + 3 * cu3 + 4.5 * cu3 * cu3 - usq);
    const cu4 = -uy;
    const d4 = f4 - w1 * rho * (1 + 3 * cu4 + 4.5 * cu4 * cu4 - usq);
    const cu5 = uz;
    const d5 = f5 - w1 * rho * (1 + 3 * cu5 + 4.5 * cu5 * cu5 - usq);
    const cu6 = -uz;
    const d6 = f6 - w1 * rho * (1 + 3 * cu6 + 4.5 * cu6 * cu6 - usq);
    const cu7 = ux + uy;
    const d7 = f7 - w2 * rho * (1 + 3 * cu7 + 4.5 * cu7 * cu7 - usq);
    const cu8 = -ux - uy;
    const d8 = f8 - w2 * rho * (1 + 3 * cu8 + 4.5 * cu8 * cu8 - usq);
    const cu9 = ux - uy;
    const d9 = f9 - w2 * rho * (1 + 3 * cu9 + 4.5 * cu9 * cu9 - usq);
    const cu10 = -ux + uy;
    const d10 = f10 - w2 * rho * (1 + 3 * cu10 + 4.5 * cu10 * cu10 - usq);
    const cu11 = ux + uz;
    const d11 = f11 - w2 * rho * (1 + 3 * cu11 + 4.5 * cu11 * cu11 - usq);
    const cu12 = -ux - uz;
    const d12 = f12 - w2 * rho * (1 + 3 * cu12 + 4.5 * cu12 * cu12 - usq);
    const cu13 = ux - uz;
    const d13 = f13 - w2 * rho * (1 + 3 * cu13 + 4.5 * cu13 * cu13 - usq);
    const cu14 = -ux + uz;
    const d14 = f14 - w2 * rho * (1 + 3 * cu14 + 4.5 * cu14 * cu14 - usq);
    const cu15 = uy + uz;
    const d15 = f15 - w2 * rho * (1 + 3 * cu15 + 4.5 * cu15 * cu15 - usq);
    const cu16 = -uy - uz;
    const d16 = f16 - w2 * rho * (1 + 3 * cu16 + 4.5 * cu16 * cu16 - usq);
    const cu17 = uy - uz;
    const d17 = f17 - w2 * rho * (1 + 3 * cu17 + 4.5 * cu17 * cu17 - usq);
    const cu18 = -uy + uz;
    const d18 = f18 - w2 * rho * (1 + 3 * cu18 + 4.5 * cu18 * cu18 - usq);
    const pxx = d1 + d2 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14;
    const pyy = d3 + d4 + d7 + d8 + d9 + d10 + d15 + d16 + d17 + d18;
    const pzz = d5 + d6 + d11 + d12 + d13 + d14 + d15 + d16 + d17 + d18;
    const pxy = d7 + d8 - d9 - d10;
    const pxz = d11 + d12 - d13 - d14;
    const pyz = d15 + d16 - d17 - d18;
    const q = Math.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2 * (pxy * pxy + pxz * pxz + pyz * pyz));
    const om = 1 / (0.5 * (tau0 + Math.sqrt(tau0 * tau0 + (smag * q) / rho)));
    g[0 * N + c] = f0 - om * d0;
    g[1 * N + c] = f1 - om * d1;
    g[2 * N + c] = f2 - om * d2;
    g[3 * N + c] = f3 - om * d3;
    g[4 * N + c] = f4 - om * d4;
    g[5 * N + c] = f5 - om * d5;
    g[6 * N + c] = f6 - om * d6;
    g[7 * N + c] = f7 - om * d7;
    g[8 * N + c] = f8 - om * d8;
    g[9 * N + c] = f9 - om * d9;
    g[10 * N + c] = f10 - om * d10;
    g[11 * N + c] = f11 - om * d11;
    g[12 * N + c] = f12 - om * d12;
    g[13 * N + c] = f13 - om * d13;
    g[14 * N + c] = f14 - om * d14;
    g[15 * N + c] = f15 - om * d15;
    g[16 * N + c] = f16 - om * d16;
    g[17 * N + c] = f17 - om * d17;
    g[18 * N + c] = f18 - om * d18;
    if (writeMacro) { macro[4 * c] = rho; macro[4 * c + 1] = ux; macro[4 * c + 2] = uy; macro[4 * c + 3] = uz; }
  }
}

export class LBMCPU {
  /**
   * @param {{dims: number[], solid: Uint8Array, uLat: number, nuLat: number, smagorinsky?: number, links?: Uint8Array}} o
   */
  constructor(o) {
    validateGrid(o);
    if (o.links && o.links.length !== 19 * o.solid.length) throw new Error('Wall links do not match the flow grid.');
    this.links = o.links || null;
    this.threads = null; // lbm-threads.js
    const [nx, ny, nz] = o.dims;
    this.dims = o.dims;
    this.N = nx * ny * nz;
    this.solid = o.solid;
    Object.assign(this, lbmParams(o));
    this.f = new Float32Array(19 * this.N);
    this.g = new Float32Array(19 * this.N);
    this.macro = new Float32Array(4 * this.N);
    this.off = CX.map((c, i) => CX[i] + nx * (CY[i] + ny * CZ[i]));
    this.src = this.off.map((o, i) => i * this.N - o);
    // per row (y + ny z), runs [x0, x1) of interior fluid cells with no solid neighbour (the fast path)
    const runs = [];
    this.runStart = new Int32Array(ny * nz + 1);
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        if (y > 0 && z > 0 && y < ny - 1 && z < nz - 1) {
          let x0 = -1;
          for (let x = 1; x <= nx - 1; x++) {
            let simple = x < nx - 1;
            if (simple) {
              const c = x + nx * (y + ny * z);
              simple = !o.solid[c];
              for (let i = 1; i < 19 && simple; i++) simple = !o.solid[c - this.off[i]];
            }
            if (simple && x0 < 0) x0 = x;
            if (!simple && x0 >= 0) { runs.push(x0, x); x0 = -1; }
          }
        }
        this.runStart[y + ny * z + 1] = runs.length / 2;
      }
    }
    this.runs = Int32Array.from(runs);
    this.reset();
  }

  reset() {
    for (let i = 0; i < 19; i++) this.f.fill(W[i], i * this.N, (i + 1) * this.N);
    this.macro.fill(0);
    for (let c = 0; c < this.N; c++) this.macro[4 * c] = 1;
    this.steps = 0;
  }

  /** The fields stepRows needs (shared with helper threads by lbm-threads.js). */
  kernelState() {
    const { dims, N, solid, off, tau0, smag, macro, links, runs, runStart, src } = this;
    return { dims, N, solid, off, tau0, smag, macro, links, runs, runStart, src };
  }

  step(writeMacro = false) {
    const [, ny, nz] = this.dims;
    const uin = inletVelocity(this.uLat, this.steps);
    const feqIn = new Float64Array(19);
    for (let i = 0; i < 19; i++) {
      const cu = CX[i] * uin;
      feqIn[i] = W[i] * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uin * uin);
    }
    if (this.threads) this.threads.step(this, uin, feqIn, writeMacro);
    else stepRows(this, this.f, this.g, uin, feqIn, writeMacro, 0, ny * nz);
    [this.f, this.g] = [this.g, this.f];
    this.steps++;
  }
}

// one thread's scratch for the cells off the fast path
const fi = new Float64Array(19), fe = new Float64Array(19);

/**
 * One step (pull from f, collide into g) on the rows r0 <= row < r1 (row = y + ny z) of the flow
 * grid, with inlet speed uin and its equilibrium feqIn. Cells only read f, so rows can be split
 * between threads. sim: LBMCPU.kernelState().
 */
export function stepRows(sim, f, g, uin, feqIn, writeMacro, r0, r1) {
  const [nx, ny, nz] = sim.dims;
  const { N, solid, off, tau0, smag, macro, links, runs, runStart, src } = sim;
  for (let row = r0; row < r1; row++) {
    const y = row % ny, z = (row / ny) | 0;
    const edge = y === 0 || z === 0 || y === ny - 1 || z === nz - 1;
    let r = runStart[row];
    const rEnd = runStart[row + 1];
    for (let x = 0; x < nx; x++) {
      if (r < rEnd && x === runs[2 * r]) {
        collideRun(f, g, macro, N, src, nx * row + x, nx * row + runs[2 * r + 1], tau0, smag, writeMacro);
        x = runs[2 * r + 1] - 1;
        r++;
        continue;
      }
      const c = x + nx * row;
      if (solid[c]) {
        if (writeMacro) { macro[4 * c] = 1; macro[4 * c + 1] = 0; macro[4 * c + 2] = 0; macro[4 * c + 3] = 0; }
        continue;
      }
      if (x === 0) {
        for (let i = 0; i < 19; i++) g[i * N + c] = feqIn[i];
        if (writeMacro) {
          macro[4 * c] = 1; macro[4 * c + 1] = uin; macro[4 * c + 2] = 0; macro[4 * c + 3] = 0;
        }
        continue;
      }
      if (edge || x === nx - 1) {
        // open boundary: velocity of the nearest interior cell (previous step), ambient pressure
        const u0 = Math.min(x, nx - 2) + nx * (Math.min(Math.max(y, 1), ny - 2) + ny * Math.min(Math.max(z, 1), nz - 2));
        let ux = uin, uy = 0, uz = 0;
        if (!solid[u0]) {
          let r = 0, px = 0, py = 0, pz = 0;
          for (let i = 0; i < 19; i++) {
            const v = f[i * N + u0];
            r += v; px += CX[i] * v; py += CY[i] * v; pz += CZ[i] * v;
          }
          ux = px / r; uy = py / r; uz = pz / r;
        }
        const usq = 1.5 * (ux * ux + uy * uy + uz * uz);
        for (let i = 0; i < 19; i++) {
          const cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
          g[i * N + c] = W[i] * (1 + 3 * cu + 4.5 * cu * cu - usq);
        }
        if (writeMacro) {
          macro[4 * c] = 1; macro[4 * c + 1] = ux; macro[4 * c + 2] = uy; macro[4 * c + 3] = uz;
        }
        continue;
      }
      let rho = 0, mx = 0, my = 0, mz = 0;
      for (let i = 0; i < 19; i++) {
        const s = c - off[i];
        let v;
        if (!solid[s]) v = f[i * N + s];
        else {
          // population that left toward the wall (direction j) comes back as direction i
          const j = OPP[i];
          v = f[j * N + c];
          const qb = links ? links[i * N + c] : 0;
          if (qb) {
            const q = (qb - 1) / 254;
            if (q < 0.5) {
              const n2 = c + off[i];
              if (!solid[n2]) v = 2 * q * v + (1 - 2 * q) * f[j * N + n2];
            } else {
              v = (0.5 / q) * v + (1 - 0.5 / q) * f[i * N + c];
            }
          }
        }
        fi[i] = v;
        rho += v;
        mx += CX[i] * v;
        my += CY[i] * v;
        mz += CZ[i] * v;
      }
      const ux = mx / rho, uy = my / rho, uz = mz / rho;
      const usq = 1.5 * (ux * ux + uy * uy + uz * uz);
      let pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
      for (let i = 0; i < 19; i++) {
        const cx = CX[i], cy = CY[i], cz = CZ[i];
        const cu = cx * ux + cy * uy + cz * uz;
        const e = W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - usq);
        fe[i] = e;
        const d = fi[i] - e;
        pxx += cx * cx * d; pyy += cy * cy * d; pzz += cz * cz * d;
        pxy += cx * cy * d; pxz += cx * cz * d; pyz += cy * cz * d;
      }
      const q = Math.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2 * (pxy * pxy + pxz * pxz + pyz * pyz));
      const tau = 0.5 * (tau0 + Math.sqrt(tau0 * tau0 + (smag * q) / rho));
      const om = 1 / tau;
      for (let i = 0; i < 19; i++) g[i * N + c] = fi[i] - om * (fi[i] - fe[i]);
      if (writeMacro) {
        macro[4 * c] = rho; macro[4 * c + 1] = ux; macro[4 * c + 2] = uy; macro[4 * c + 3] = uz;
      }
    }
  }
}
