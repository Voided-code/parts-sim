// 8-node hexahedral ("brick") element on a unit cube with E = 1.
//
// Every voxel in the structural model is the same cube, so the element stiffness
// matrix is computed once and scaled by (E * h * density) per voxel.
//
// We use Wilson's incompatible-mode element (a.k.a. Q6 / "hex8 with bubble modes"):
// the 9 internal bending modes are statically condensed out, so the final matrix is
// still 24x24. This removes the shear locking of plain trilinear bricks, which matters
// a lot for bend tests where only a few voxels span the thickness of a part.

// Node order: bottom face counter-clockwise, then top face.
export const HEX_NODES = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];

/** Isotropic elasticity matrix (6x6, row-major) for E = 1. Voigt order xx, yy, zz, xy, yz, zx (engineering shear). */
export function elasticityMatrix(nu) {
  if (!Number.isFinite(nu) || nu <= -1 || nu >= 0.5) {
    throw new Error("Poisson's ratio must be greater than −1 and less than 0.5.");
  }
  const lam = nu / ((1 + nu) * (1 - 2 * nu));
  const mu = 1 / (2 * (1 + nu));
  const D = new Float64Array(36);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) D[i * 6 + j] = lam;
    D[i * 6 + i] = lam + 2 * mu;
  }
  D[3 * 6 + 3] = mu;
  D[4 * 6 + 4] = mu;
  D[5 * 6 + 5] = mu;
  return D;
}

/** Strain-displacement matrix B (6x24) at natural coordinates (xi, eta, zeta) in [-1, 1] for a cube of side 1. */
export function strainDisplacement(xi, eta, zeta) {
  const B = new Float64Array(6 * 24);
  for (let a = 0; a < 8; a++) {
    const sx = 2 * HEX_NODES[a][0] - 1;
    const sy = 2 * HEX_NODES[a][1] - 1;
    const sz = 2 * HEX_NODES[a][2] - 1;
    // dN/dx = dN/dxi * dxi/dx with dxi/dx = 2 on a unit cube.
    const dx = (2 * sx * (1 + sy * eta) * (1 + sz * zeta)) / 8;
    const dy = (2 * sy * (1 + sx * xi) * (1 + sz * zeta)) / 8;
    const dz = (2 * sz * (1 + sx * xi) * (1 + sy * eta)) / 8;
    fillStrainColumns(B, 24, 3 * a, dx, dy, dz);
  }
  return B;
}

function fillStrainColumns(B, cols, c, dx, dy, dz) {
  B[0 * cols + c] = dx;
  B[1 * cols + c + 1] = dy;
  B[2 * cols + c + 2] = dz;
  B[3 * cols + c] = dy;
  B[3 * cols + c + 1] = dx;
  B[4 * cols + c + 1] = dz;
  B[4 * cols + c + 2] = dy;
  B[5 * cols + c] = dz;
  B[5 * cols + c + 2] = dx;
}

// Incompatible (bubble) modes 1 - xi^2, 1 - eta^2, 1 - zeta^2 for each displacement component.
function incompatibleModes(xi, eta, zeta) {
  const G = new Float64Array(6 * 9);
  const grads = [
    [-2 * xi * 2, 0, 0],
    [0, -2 * eta * 2, 0],
    [0, 0, -2 * zeta * 2],
  ];
  for (let m = 0; m < 3; m++) fillStrainColumns(G, 9, 3 * m, grads[m][0], grads[m][1], grads[m][2]);
  return G;
}

// C (r x c) += s * A^T (k x r) * B (k x c)
function addAtB(C, A, B, k, r, c, s) {
  for (let i = 0; i < r; i++) {
    for (let j = 0; j < c; j++) {
      let sum = 0;
      for (let q = 0; q < k; q++) sum += A[q * r + i] * B[q * c + j];
      C[i * c + j] += s * sum;
    }
  }
}

function matMul(A, B, n, k, m) {
  const C = new Float64Array(n * m);
  for (let i = 0; i < n; i++) {
    for (let q = 0; q < k; q++) {
      const a = A[i * k + q];
      if (a === 0) continue;
      for (let j = 0; j < m; j++) C[i * m + j] += a * B[q * m + j];
    }
  }
  return C;
}

function invert(M, n) {
  const A = Float64Array.from(M);
  const I = new Float64Array(n * n);
  for (let i = 0; i < n; i++) I[i * n + i] = 1;
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(A[r * n + col]) > Math.abs(A[piv * n + col])) piv = r;
    if (piv !== col) {
      for (let j = 0; j < n; j++) {
        [A[col * n + j], A[piv * n + j]] = [A[piv * n + j], A[col * n + j]];
        [I[col * n + j], I[piv * n + j]] = [I[piv * n + j], I[col * n + j]];
      }
    }
    const d = A[col * n + col];
    for (let j = 0; j < n; j++) {
      A[col * n + j] /= d;
      I[col * n + j] /= d;
    }
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = A[r * n + col];
      if (f === 0) continue;
      for (let j = 0; j < n; j++) {
        A[r * n + j] -= f * A[col * n + j];
        I[r * n + j] -= f * I[col * n + j];
      }
    }
  }
  return I;
}

/**
 * Element stiffness (24x24, row-major) of a unit cube with E = 1.
 * The physical matrix of a voxel of side h and modulus E is E * h * K.
 */
export function hexStiffness(nu, opts) {
  return hexElement(nu, opts).K;
}

/**
 * Unit-cube element with E = 1:
 *  - K: 24x24 condensed stiffness
 *  - cornerStress: 8 matrices (6x24) giving the stress at each corner node from the
 *    element's nodal displacements, including the condensed incompatible modes.
 *    For a voxel of side h and modulus E, sigma = (E / h) * S_a * u_e.
 */
export function hexElement(nu, { incompatibleModes: useModes = true } = {}) {
  const D = elasticityMatrix(nu);
  const g = 1 / Math.sqrt(3);
  const detJ = 1 / 8;
  const Kuu = new Float64Array(24 * 24);
  const Kua = new Float64Array(24 * 9);
  const Kaa = new Float64Array(9 * 9);
  for (const xi of [-g, g]) {
    for (const eta of [-g, g]) {
      for (const zeta of [-g, g]) {
        const B = strainDisplacement(xi, eta, zeta);
        const DB = matMul(D, B, 6, 6, 24);
        addAtB(Kuu, B, DB, 6, 24, 24, detJ);
        if (useModes) {
          const G = incompatibleModes(xi, eta, zeta);
          const DG = matMul(D, G, 6, 6, 9);
          addAtB(Kua, B, DG, 6, 24, 9, detJ);
          addAtB(Kaa, G, DG, 6, 9, 9, detJ);
        }
      }
    }
  }
  // alpha = -Kaa^-1 * Kau * u  (internal mode amplitudes)
  let A = null;
  if (useModes) {
    const KaaInv = invert(Kaa, 9);
    const Kau = new Float64Array(9 * 24);
    for (let i = 0; i < 24; i++) for (let q = 0; q < 9; q++) Kau[q * 24 + i] = Kua[i * 9 + q];
    A = matMul(KaaInv, Kau, 9, 9, 24);
    for (let i = 0; i < A.length; i++) A[i] = -A[i];
    // Static condensation: K = Kuu + Kua * A
    const X = matMul(Kua, A, 24, 9, 24);
    for (let i = 0; i < 576; i++) Kuu[i] += X[i];
    for (let i = 0; i < 24; i++) {
      for (let j = i + 1; j < 24; j++) {
        const m = 0.5 * (Kuu[i * 24 + j] + Kuu[j * 24 + i]);
        Kuu[i * 24 + j] = m;
        Kuu[j * 24 + i] = m;
      }
    }
  }
  // strain-displacement matrix including the condensed internal modes: B + G * A
  const enhanced = (xi, eta, zeta) => {
    const Bt = strainDisplacement(xi, eta, zeta);
    if (A) {
      const GA = matMul(incompatibleModes(xi, eta, zeta), A, 6, 9, 24);
      for (let i = 0; i < Bt.length; i++) Bt[i] += GA[i];
    }
    return Bt;
  };
  const cornerStress = HEX_NODES.map(([x, y, z]) => matMul(D, enhanced(2 * x - 1, eta0(y), eta0(z)), 6, 6, 24));
  // Gauss points ordered like the corners they are nearest to. sum_g Bbar_g^T D Bbar_g / 8 = K.
  const gaussB = HEX_NODES.map(([x, y, z]) => enhanced((2 * x - 1) * g, (2 * y - 1) * g, (2 * z - 1) * g));
  return { K: Kuu, cornerStress, gaussB };
}

const eta0 = (v) => 2 * v - 1;
