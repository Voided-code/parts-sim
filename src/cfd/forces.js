// Surface pressure integration on the voxel boundary of the solid.
// Every fluid cell face that touches a solid cell contributes Cp * area along the face
// direction (pressure pushes the solid away from the fluid).

/**
 * @param {Float32Array} rho     density per cell (time-averaged or instantaneous)
 * @param {Uint8Array} solid
 * @param {number[]} dims
 * @param {number} uLat          free-stream lattice velocity
 * @returns {{C: number[], frontal: number}} C = sum of Cp * n over wetted voxel faces (in cell^2);
 *          force [N] = C * (0.5 rho U^2) * h^2 ; frontal = projected solid area along x (cell^2)
 */
export function pressureForceCoefficients(rho, solid, dims, uLat) {
  const [nx, ny, nz] = dims;
  const inv = 1 / (3 * 0.5 * uLat * uLat); // Cp = (rho - 1) c_s^2 / (0.5 u^2)
  const C = [0, 0, 0];
  const strides = [1, nx, nx * ny];
  const shadow = new Uint8Array(ny * nz);
  for (let z = 1; z < nz - 1; z++) {
    for (let y = 1; y < ny - 1; y++) {
      for (let x = 1; x < nx - 1; x++) {
        const c = x + nx * (y + ny * z);
        if (solid[c]) {
          shadow[y + ny * z] = 1;
          continue;
        }
        const cp = (rho[c] - 1) * inv;
        for (let a = 0; a < 3; a++) {
          if (solid[c + strides[a]]) C[a] += cp;
          if (solid[c - strides[a]]) C[a] -= cp;
        }
      }
    }
  }
  let frontal = 0;
  for (const s of shadow) frontal += s;
  return { C, frontal };
}
