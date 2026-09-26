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

export class LBMCPU {
  /**
   * @param {{dims: number[], solid: Uint8Array, uLat: number, nuLat: number, smagorinsky?: number, links?: Uint8Array}} o
   */
  constructor(o) {
    validateGrid(o);
    if (o.links && o.links.length !== 19 * o.solid.length) throw new Error('Wall links do not match the flow grid.');
    this.links = o.links || null;
    const [nx, ny, nz] = o.dims;
    this.dims = o.dims;
    this.N = nx * ny * nz;
    this.solid = o.solid;
    Object.assign(this, lbmParams(o));
    this.f = new Float32Array(19 * this.N);
    this.g = new Float32Array(19 * this.N);
    this.macro = new Float32Array(4 * this.N);
    this.off = CX.map((c, i) => CX[i] + nx * (CY[i] + ny * CZ[i]));
    this.reset();
  }

  reset() {
    for (let i = 0; i < 19; i++) this.f.fill(W[i], i * this.N, (i + 1) * this.N);
    this.macro.fill(0);
    for (let c = 0; c < this.N; c++) this.macro[4 * c] = 1;
    this.steps = 0;
  }

  step(writeMacro = false) {
    const [nx, ny, nz] = this.dims;
    const { N, f, g, solid, off, tau0, smag, macro, links } = this;
    const uin = inletVelocity(this.uLat, this.steps);
    const feqIn = new Float64Array(19);
    for (let i = 0; i < 19; i++) {
      const cu = CX[i] * uin;
      feqIn[i] = W[i] * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uin * uin);
    }
    const fi = new Float64Array(19);
    const fe = new Float64Array(19);
    for (let z = 0; z < nz; z++) {
      for (let y = 0; y < ny; y++) {
        const edge = y === 0 || z === 0 || y === ny - 1 || z === nz - 1;
        for (let x = 0; x < nx; x++) {
          const c = x + nx * (y + ny * z);
          if (solid[c]) continue;
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
    this.f = g;
    this.g = f;
    this.steps++;
    if (writeMacro) {
      for (let c = 0; c < N; c++) if (solid[c]) { macro[4 * c] = 1; macro[4 * c + 1] = 0; macro[4 * c + 2] = 0; macro[4 * c + 3] = 0; }
    }
  }
}
