// Structural and thermal solves off the main thread.
//
// 'solve': one linear static analysis.
// 'break': progressive damage. Solve, find the voxel that reaches the material strength
//          first, remove it (and its equally-stressed neighbours), re-solve, repeat until the
//          load path is severed. These elastic voxel-removal steps are a qualitative
//          damage illustration, not a calibrated plasticity or fracture simulation.
// The other studies (frequency, buckling, nonlinear, drop test, dynamics, optimization, thermal)
// live in studies.js and map their results onto the part's vertices before sending them back.
import { VoxelFEA, pruneFloating } from './solver.js';
import { GPUFEASolver, GPU_COARSEST_DOF } from './gpu-solver.js';
import { runStudy } from './studies.js';
import { useThreads } from './threads.js';

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'solve') await solveOnce(m);
    else if (m.type === 'break') await breakTest(m);
    else await runStudy(m, { post: (msg, transfer) => self.postMessage(msg, transfer || []) });
  } catch (err) {
    if (err?.cancelled) return;
    self.postMessage({ type: 'error', message: err?.message || String(err) });
  }
};

// Set when the GPU failed once in this job, so later break-test steps go straight to the CPU.
let gpuFailure = null;

/**
 * Build the multigrid hierarchy and solve, on the GPU when allowed and available (m.engine
 * 'auto'), otherwise - or if the GPU path fails - with the CPU solver on the same hierarchy.
 */
async function buildAndSolve(m, density, f, x0, stage) {
  const useGPU = m.engine !== 'cpu' && !gpuFailure;
  const fea = new VoxelFEA({ dims: m.dims, density, nu: m.nu, bc: m.bc, ...(useGPU ? { coarsestMaxDof: GPU_COARSEST_DOF } : {}) });
  // helper threads (the CPU solve, the GPU solve's 64-bit residual checks and the stresses),
  // starting while the GPU sets up
  const started = useThreads(fea);
  const opts = (label) => ({
    tol: 1e-6,
    maxIter: label === 'GPU' ? 3000 : 400,
    x0,
    onProgress: (it, res) => {
      if (it % 2 === 0) self.postMessage({ type: 'progress', stage: `${stage} on the ${label}`, it, res });
    },
  });
  if (useGPU) {
    let gpu = null;
    try {
      gpu = await GPUFEASolver.create(fea);
      await started;
      const sol = await gpu.solve(f, opts('GPU'));
      if (sol.converged) return { fea, sol, engine: 'GPU' };
      gpuFailure = 'the GPU solve did not converge';
    } catch (err) {
      gpuFailure = err?.message || String(err);
    } finally {
      gpu?.destroy();
    }
  }
  await started;
  const sol = fea.solve(f, opts('CPU'));
  return { fea, sol, engine: 'CPU' };
}

function heldNodes(bc) {
  const held = new Uint8Array(bc.length / 3);
  for (let n = 0; n < held.length; n++) held[n] = bc[3 * n] | bc[3 * n + 1] | bc[3 * n + 2];
  return held;
}

function lostLoadFraction(fea, f) {
  const act = fea.levels[0].activeNode;
  let lost = 0, total = 0;
  for (let n = 0; n < act.length; n++) {
    const m = Math.hypot(f[3 * n], f[3 * n + 1], f[3 * n + 2]);
    total += m;
    if (!act[n]) lost += m;
  }
  return total > 0 ? lost / total : 0;
}

function analyze(fea, sol, m) {
  const t0 = performance.now();
  const scale = 1 / (m.E * m.h);
  const u = new Float64Array(sol.u.length);
  for (let i = 0; i < u.length; i++) u[i] = sol.u[i] * scale;
  const st = fea.stresses(u, m.E, m.h);
  return { sol, u, st, ms: performance.now() - t0 };
}

function maxDisp(u) {
  let best = 0;
  for (let i = 0; i < u.length; i += 3) best = Math.max(best, u[i] * u[i] + u[i + 1] * u[i + 1] + u[i + 2] * u[i + 2]);
  return Math.sqrt(best);
}

async function solveOnce(m) {
  const t0 = performance.now();
  const held = heldNodes(m.bc);
  const { density, removed } = pruneFloating(m.dims, m.density, held);
  self.postMessage({ type: 'progress', stage: 'Building multigrid', it: 0, res: 1 });
  const { fea, sol, engine } = await buildAndSolve(m, density, m.f, null, 'Solving');
  const lost = lostLoadFraction(fea, m.f);
  const { u, st } = analyze(fea, sol, m);
  if (!sol.converged) throw new Error('The structural solve did not converge. Enlarge the support area, check disconnected parts, or change the mesh resolution.');
  if (lost > 0.99) throw new Error('The applied load is on geometry disconnected from the supports.');
  const R = fea.reactions(sol.u, m.f);
  const u32 = Float32Array.from(u);
  self.postMessage(
    {
      type: 'result',
      u: u32,
      nodeVM: st.nodeVM,
      nodeP1: st.nodeP1,
      nodeP3: st.nodeP3,
      activeNode: fea.levels[0].activeNode,
      density,
      iterations: sol.iterations,
      residual: sol.residual,
      converged: sol.converged,
      removed,
      lostLoad: lost,
      reaction: R,
      levels: fea.levels.length,
      voxels: fea.levels[0].elems.length,
      ms: performance.now() - t0,
      engine,
      gpuNote: engine === 'CPU' && m.engine !== 'cpu' ? gpuFailure : null,
    },
    [u32.buffer, st.nodeVM.buffer, st.nodeP1.buffer, st.nodeP3.buffer],
  );
}

async function breakTest(m) {
  if (!Number.isFinite(m.strength) || m.strength <= 0) throw new Error('Failure strength must be positive.');
  const { dims, nu } = m;
  const [nx, ny] = dims;
  const NX = nx + 1, NY = ny + 1;
  const off = [0, 1, 1 + NX, NX, NX * NY, 1 + NX * NY, 1 + NX + NX * NY, NX + NX * NY];
  const held = heldNodes(m.bc);
  let { density, removed } = pruneFloating(dims, m.density, held);
  const nE = density.length;
  const brokenAt = new Int16Array(nE).fill(-1); // step at which a voxel cracked
  const detachedAt = new Int16Array(nE).fill(-1); // step at which a voxel fell off
  let x0 = null;
  let peak = 0;
  const maxSteps = m.maxSteps || 60;
  let reason = 'step limit';

  let engine = 'CPU';
  for (let step = 0; step < maxSteps; step++) {
    if (!density.some((d) => d > 0)) {
      reason = step > 0 ? 'separated' : 'unsupported load';
      break;
    }
    const solved = await buildAndSolve(m, density, m.f, x0, `Break test step ${step + 1}`);
    const { fea, sol } = solved;
    engine = solved.engine;
    const L = fea.levels[0];
    if (L.elems.length === 0 || lostLoadFraction(fea, m.f) > 0.5) {
      reason = step > 0 ? 'separated' : 'unsupported load';
      break;
    }
    const { u, st } = analyze(fea, sol, m);
    fea.threads?.release(); // the next step's model takes the helpers over
    if (!sol.converged) {
      reason = 'solver did not converge';
      break;
    }
    x0 = sol.u;
    const nodal = m.criterion === 'p1' ? st.nodeP1 : st.nodeVM;
    // voxel failure measure: the worst of its corner (nodal) stresses
    const measure = new Float32Array(L.elems.length);
    let smax = 0;
    for (let q = 0; q < L.elems.length; q++) {
      let s = 0;
      for (let a = 0; a < 8; a++) s = Math.max(s, nodal[L.base[q] + off[a]]);
      measure[q] = s;
      if (s > smax) smax = s;
    }
    if (!(smax > 0)) {
      reason = 'no stress';
      break;
    }
    const lambda = m.strength / smax; // multiple of the applied load that cracks the next voxel
    peak = Math.max(peak, lambda);

    const cap = Math.max(1, Math.round(0.02 * Math.pow(L.elems.length, 2 / 3)));
    const order = Array.from(measure.keys()).sort((a, b) => measure[b] - measure[a]);
    const cracked = [];
    for (const q of order) {
      if (cracked.length >= cap || measure[q] < 0.95 * smax) break;
      cracked.push(L.elems[q]);
    }
    const next = Float32Array.from(density);
    for (const e of cracked) {
      next[e] = 0;
      brokenAt[e] = step;
    }
    const pruned = pruneFloating(dims, next, held);
    const detached = [];
    for (let e = 0; e < nE; e++) {
      if (next[e] > 0 && !(pruned.density[e] > 0)) {
        detachedAt[e] = step;
        detached.push(e);
      }
    }
    density = pruned.density;

    const uScaled = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) uScaled[i] = u[i] * lambda;
    const vm = new Float32Array(st.nodeVM.length);
    for (let i = 0; i < vm.length; i++) vm[i] = st.nodeVM[i] * lambda;
    self.postMessage(
      {
        type: 'breakStep',
        step,
        lambda,
        maxDisp: maxDisp(u) * lambda,
        cracked: Int32Array.from(cracked),
        detached: Int32Array.from(detached),
        nodeVM: vm,
        u: uScaled,
        activeNode: L.activeNode,
        voxels: L.elems.length,
      },
      [vm.buffer, uScaled.buffer],
    );
  }
  self.postMessage({ type: 'breakDone', reason, peak, brokenAt, detachedAt, removed, engine });
}
