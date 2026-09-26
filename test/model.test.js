// The full voxel-model pipeline (voxelise -> fixtures/loads -> solve) must not depend on how
// the grid happens to line up with the part: partly filled surface voxels and clamps whose
// face the grid overhangs are where errors creep in.
import test from 'node:test';
import assert from 'node:assert/strict';
import { SAMPLES } from '../src/core/samples.js';
import { buildPart } from '../src/core/mesh.js';
import { StructuralModel } from '../src/fea/structural.js';
import { VoxelFEA } from '../src/fea/solver.js';

test('beam sample deflection matches theory at every voxel resolution', () => {
  const s = SAMPLES.find((x) => x.id === 'beam');
  const part = buildPart({ ...s.make(), name: 'beam' });
  const setup = s.setup(part);
  const E = 200e9, nu = 0.29;
  // 200 x 20 x 10 mm steel, 800 N at the tip: FL^3/3EI + shear
  const I = (0.01 * 0.02 ** 3) / 12;
  const theory = (800 * 0.2 ** 3) / (3 * E * I) + (800 * 0.2) / ((5 / 6) * (E / (2 * (1 + nu))) * 0.02 * 0.01);
  for (const res of [40, 64, 90, 104, 128]) {
    const m = new StructuralModel(part, res);
    const asm = m.assemble({ ...setup, gravity: false, material: { E: 200, nu, yield: 351, uts: 420, density: 7900 }, toMeters: 0.001 });
    const fea = new VoxelFEA({ dims: m.dims, density: m.density, nu, bc: asm.bc });
    const sol = fea.solve(asm.f, { tol: 1e-7 });
    let tip = 0;
    for (let n = 0; n < m.nNodes; n++) tip = Math.max(tip, Math.abs(sol.u[3 * n + 1]) / (E * m.h * 0.001));
    const ratio = tip / theory;
    console.log(`  ${res} voxels: tip ${(tip * 1e3).toFixed(3)} mm vs ${(theory * 1e3).toFixed(3)} mm (x${ratio.toFixed(3)})`);
    assert.ok(ratio > 0.93 && ratio < 1.07, `${res} voxels: ratio ${ratio}`);
  }
});
