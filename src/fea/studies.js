// Worker side of the studies beyond linear static: frequency, buckling, nonlinear static, drop
// test, linear dynamics (modal basis), optimization and heat transfer. Each study solves on the
// voxel grid and maps its results onto the part's surface vertices before posting them, so the
// main thread only has to draw.
import { VoxelFEA, pruneFloating, vonMisesAt, principalStresses } from './solver.js';
import { GPUFEASolver, GPU_COARSEST_DOF } from './gpu-solver.js';
import { naturalFrequencies, bucklingFactors, elementStresses, cpuPreconditioner, lumpedMass } from './eigen.js';
import { NonlinearModel, loadRamp, hardeningFor, equilibrate } from './nonlinear.js';
import { dropTestCPU, maxEigenvalue } from './explicit.js';
import { GPUExplicit } from './gpu-explicit.js';
import { optimizeTopology } from './topology.js';
import { solveHeat, heatFlux } from './thermal.js';
import { vertexWeights, interpolate } from './mapping.js';

const cancelled = () => Object.assign(new Error('Cancelled'), { cancelled: true });

export async function runStudy(m, { post }) {
  const studies = { modal, buckling, nonlinear, drop, topology, sizing, thermal };
  const run = studies[m.type];
  if (!run) throw new Error(`Unknown study: ${m.type}`);
  await run(m, post);
}

// ---------- shared helpers ----------

function heldNodes(bc) {
  const held = new Uint8Array(bc.length / 3);
  for (let n = 0; n < held.length; n++) held[n] = bc[3 * n] | bc[3 * n + 1] | bc[3 * n + 2];
  return held;
}

function weights(m, activeNode) {
  return vertexWeights(m.map.vertices, { origin: m.map.origin, h: m.map.hModel, dims: m.dims }, activeNode);
}

/** Build the voxel model (pruned to what is connected to the fixtures unless free-floating). */
function buildModel(m, { free = false, diagAdd = null } = {}) {
  let density = m.density, removed = 0;
  if (!free) ({ density, removed } = pruneFloating(m.dims, m.density, heldNodes(m.bc)));
  const gpu = m.engine !== 'cpu';
  const fea = new VoxelFEA({ dims: m.dims, density, nu: m.nu, bc: m.bc, diagAdd: diagAdd ? diagAdd(density) : null, ...(gpu ? { coarsestMaxDof: GPU_COARSEST_DOF } : {}) });
  if (!fea.levels[0].elems.length) throw new Error('No voxels are connected to the fixtures.');
  return { fea, density, removed };
}

/** Solver engine: GPU multigrid-CG (WebGPU) when allowed and working, otherwise the CPU multigrid. */
async function engineFor(m, fea) {
  let note = null;
  if (m.engine !== 'cpu') {
    try {
      const gpu = await GPUFEASolver.create(fea);
      return {
        name: 'GPU',
        // a few multigrid-CG steps per vector on the GPU, the whole block in one round trip
        precond: (R) => gpu.preconditionBlock(R, 4),
        async solve(f, o) {
          const sol = await gpu.solve(f, o);
          return sol.converged ? sol : fea.solve(f, o);
        },
        destroy: () => gpu.destroy(),
      };
    } catch (err) {
      note = err?.message || String(err);
    }
  }
  const pre = cpuPreconditioner(fea);
  return { name: 'CPU', note, precond: async (R) => pre(R), solve: async (f, o) => fea.solve(f, o), destroy() {} };
}

function progress(post, stage, frac) {
  post({ type: 'progress', stage, it: 0, res: 1, frac });
}

/** Normalize a vertex vector field so its largest length is 1. */
function unitShape(u) {
  let mx = 0;
  for (let i = 0; i < u.length; i += 3) if (!Number.isNaN(u[i])) mx = Math.max(mx, Math.hypot(u[i], u[i + 1], u[i + 2]));
  if (mx > 0) for (let i = 0; i < u.length; i++) u[i] /= mx;
  return u;
}

function nodalVM(fea, u, E, h) {
  return fea.stresses(u, E, h).nodeVM;
}

// ---------- frequency (modal) and the modal basis for linear dynamics ----------

async function modal(m, post) {
  const free = !m.bc.some((v) => v);
  const nev = Math.max(1, Math.min(20, m.nev || 5));
  let shift = 0;
  const { fea, removed } = buildModel(m, {
    free,
    diagAdd: free
      ? (density) => {
        // free-floating: K + s M keeps the preconditioner invertible; s is far below the first
        // flexible eigenvalue, and the 6 rigid-body modes come out at ~0 Hz
        const probe = new VoxelFEA({ dims: m.dims, density, nu: m.nu, bc: m.bc, coarsestMaxDof: Infinity });
        const mass = lumpedMass(probe);
        shift = 1e-6 * maxEigenvalue(probe, mass, 20);
        return mass.map((v) => v * shift);
      }
      : null,
  });
  const engine = await engineFor(m, fea);
  try {
    progress(post, `Finding natural frequencies on the ${engine.name}`, 0);
    const want = nev + (free ? 6 : 0);
    const r = await naturalFrequencies(fea, {
      nev: want, E: m.E, density: m.rho, h: m.h, shift, precondFull: (R) => engine.precond(R),
      onProgress: (it, res, conv) => { post({ type: 'progress', stage: `Natural frequencies on the ${engine.name} · ${conv}/${want} converged`, it, res }); },
    });
    const L = fea.levels[0];
    const W = weights(m, L.activeNode);
    const mass = r.mass;
    let totalMass = 0;
    for (let i = 0; i < mass.length; i += 3) totalMass += mass[i];
    const modes = [];
    const rigidTol = Math.max(1e-3 * shift, 1e-12);
    const sq = Math.sqrt(m.rho * m.h ** 3);
    for (let i = 0; i < r.modes.length && modes.length < nev; i++) {
      if (free && r.lambdas[i] < rigidTol) continue;
      const phi = r.modes[i];
      // effective mass fractions along x, y, z
      const part = [0, 0, 0];
      for (let n = 0; n < phi.length; n += 3) for (let d = 0; d < 3; d++) part[d] += phi[n + d] * mass[n + d];
      const eff = part.map((p) => (p * p) / totalMass);
      modes.push({ freq: r.freqs[i], lambda: r.lambdas[i], eff, phi, participation: part.map((p) => p * sq) });
    }
    const out = {
      type: 'done', study: 'modal', free, engine: engine.name, gpuNote: engine.note, removed,
      converged: r.converged, iterations: r.iterations,
      totalMass: totalMass * m.rho * m.h ** 3, activeNode: L.activeNode,
      modes: modes.map((md) => ({ freq: md.freq, eff: md.eff, shape: unitShape(interpolate(W, md.phi, 3)) })),
    };
    const transfer = out.modes.map((md) => md.shape.buffer);
    if (m.dynamic) {
      // modal basis for linear dynamics: shapes per unit modal coordinate, pattern and static response
      progress(post, `Static response for the dynamic load on the ${engine.name}`, 0.9);
      let f;
      if (m.dynamic.kind === 'base') {
        const r = m.dynamic.dir;
        f = new Float64Array(L.nDof);
        const M = m.rho * m.h ** 3;
        for (let n = 0; n < L.nNodes; n++) for (let d = 0; d < 3; d++) f[3 * n + d] = -M * mass[3 * n + d] * r[d];
      } else f = m.f;
      const sol = await engine.solve(f, { tol: 1e-7, maxIter: 3000 });
      const uSt = sol.u.map((v) => v / (m.E * m.h));
      const toModel = 1 / m.toMeters;
      const basis = {
        omegas: modes.map((md) => 2 * Math.PI * md.freq),
        gamma: modes.map((md) => { let g = 0; for (let i = 0; i < f.length; i++) g += (md.phi[i] / sq) * f[i]; return g; }),
        modeU: [], modeS: [],
        staticU: interpolate(W, uSt, 3, toModel),
        staticS: interpolate(W, fea.stressTensors(uSt, m.E, m.h), 6),
      };
      for (const md of modes) {
        const phys = md.phi.map((v) => v / sq);
        basis.modeU.push(interpolate(W, phys, 3, toModel));
        basis.modeS.push(interpolate(W, fea.stressTensors(phys, m.E, m.h), 6));
      }
      let total = 0;
      for (let i = 0; i < f.length; i++) if (!L.fixed[i]) total += f[i];
      out.basis = basis;
      out.patternForce = m.dynamic.kind === 'base' ? totalMass * m.rho * m.h ** 3 : null;
      transfer.push(basis.staticU.buffer, basis.staticS.buffer, ...basis.modeU.map((a) => a.buffer), ...basis.modeS.map((a) => a.buffer));
    }
    post(out, transfer);
  } finally {
    engine.destroy();
  }
}

// ---------- linear buckling ----------

async function buckling(m, post) {
  const { fea, removed } = buildModel(m);
  const engine = await engineFor(m, fea);
  try {
    progress(post, `Pre-buckling stresses on the ${engine.name}`, 0);
    const sol = await engine.solve(m.f, { tol: 1e-8, maxIter: 3000, onProgress: (it, res) => post({ type: 'progress', stage: `Pre-buckling stresses on the ${engine.name}`, it, res }) });
    const u = sol.u.map((v) => v / (m.E * m.h));
    const sigma = elementStresses(fea, u, m.h);
    const nev = Math.max(1, Math.min(8, m.nev || 3));
    // Search up to 100 times the load that makes it yield (at least 1000 x the loads): a buckling
    // load far beyond yielding is never reached, and near-zero eigenvalues are slow to resolve.
    const nodeVM = nodalVM(fea, u, m.E, m.h);
    let vmMax = 0;
    for (const v of nodeVM) vmMax = Math.max(vmMax, v);
    const yieldFactor = vmMax > 0 && m.strength > 0 ? m.strength / vmMax : Infinity;
    const maxFactor = Number.isFinite(yieldFactor) ? Math.max(1000, 100 * yieldFactor) : Infinity;
    const r = await bucklingFactors(fea, sigma, {
      nev, maxFactor, precondFull: (R) => engine.precond(R),
      onProgress: (it, res, conv) => { post({ type: 'progress', stage: `Buckling modes on the ${engine.name} · ${conv}/${nev} converged`, it, res }); },
    });
    const L = fea.levels[0];
    const W = weights(m, L.activeNode);
    const vm = interpolate(W, nodeVM);
    const modes = r.factors.map((factor, i) => ({ factor, shape: unitShape(interpolate(W, r.modes[i], 3)) }));
    post({
      type: 'done', study: 'buckling', engine: engine.name, gpuNote: engine.note, removed, maxFactor,
      converged: r.converged, iterations: r.iterations, modes, vm, activeNode: L.activeNode,
    }, [vm.buffer, ...modes.map((md) => md.shape.buffer)]);
  } finally {
    engine.destroy();
  }
}

// ---------- nonlinear static ----------

async function nonlinear(m, post) {
  const { fea, removed } = buildModel(m);
  const engine = await engineFor(m, fea);
  try {
    const L = fea.levels[0];
    const W = weights(m, L.activeNode);
    const mat = m.material;
    const plastic = m.plasticity && !mat.brittle ? hardeningFor(mat) : null;
    const Eh2 = m.E * m.h * m.h;
    const f = m.f.map((v) => v / Eh2);
    let fAbs = 0;
    for (let i = 0; i < f.length; i++) if (!L.fixed[i]) fAbs += Math.abs(m.f[i]);
    const maxDim = Math.max(...m.dims);
    const precond = async (r) => (await engine.precond([r]))[0];
    let steps = 0, lastIt = '';
    const curve = [];
    const toModel = m.h / m.toMeters; // voxel units -> model units
    const sendStep = (s, kind = 'step') => {
      let work = 0;
      for (let i = 0; i < f.length; i++) if (!L.fixed[i]) work += m.f[i] * s.u[i];
      const D = fAbs > 0 ? (work / fAbs) * toModel : 0; // load-weighted displacement along the loads
      const fields = s.fields;
      const u = interpolate(W, s.u, 3, toModel);
      const vm = interpolate(W, fields.vm, 1, m.E);
      const pe = interpolate(W, fields.pe, 1);
      const p1 = interpolate(W, fields.p1, 1, m.E);
      let maxVM = 0, maxPE = 0;
      for (const v of fields.vm) maxVM = Math.max(maxVM, v * m.E);
      for (const v of fields.pe) maxPE = Math.max(maxPE, v);
      const point = { lam: s.lam, D, maxVM, maxPE, maxDisp: s.maxDisp * m.h / m.toMeters };
      if (kind === 'step') curve.push(point);
      post({ type: 'step', kind, ...point, u, vm, pe, p1, iterations: s.iterations }, [u.buffer, vm.buffer, pe.buffer, p1.buffer]);
    };
    const model = new NonlinearModel(fea, { largeDisplacement: m.largeDisplacement !== false, plastic });
    const res = await loadRamp(model, f, precond, {
      target: m.untilFailure ? Infinity : 1,
      steps: m.steps || 10,
      maxSteps: m.untilFailure ? 60 : 40,
      rupture: plastic ? mat.elongation : Infinity,
      crackStress: mat.brittle && m.untilFailure ? (mat.uts * 1e6) / m.E : Infinity,
      // "failed" also means grossly bent: the part moved more than 15% of its size
      maxDisp: m.untilFailure ? 0.15 * maxDim : Infinity,
      maxTime: m.untilFailure ? 240 : Infinity,
      stepAlpha: plastic ? Math.max(0.002, mat.elongation / 12) : Infinity,
      stepDisp: m.untilFailure ? 0.04 * maxDim : Infinity,
      onIter: (lam, it, rel) => {
        lastIt = `load ×${lam.toFixed(3)} · Newton ${it}`;
        post({ type: 'progress', stage: `Nonlinear on the ${engine.name} · ${lastIt}`, it, res: rel });
      },
      onStep: (s) => { steps++; sendStep(s); },
    });
    // spring-back: unload with the plastic strains kept to show the permanent deformation
    let permanent = false;
    if (plastic && model.alpha.some((a) => a > 0)) {
      const u = Float64Array.from(res.u);
      post({ type: 'progress', stage: 'Unloading (spring-back)', it: 0, res: 1 });
      const un = await equilibrate(model, u, new Float64Array(f.length), precond, { onIter: (it, rel) => post({ type: 'progress', stage: `Unloading (spring-back) · Newton ${it}`, it, res: rel }) });
      if (un.converged) {
        let dmax = 0;
        for (let i = 0; i < u.length; i += 3) dmax = Math.max(dmax, Math.hypot(u[i], u[i + 1], u[i + 2]));
        sendStep({ lam: 0, u, fields: model.nodalFields(u), maxDisp: dmax, iterations: un.iterations }, 'unloaded');
        permanent = true;
      }
    }
    post({ type: 'done', study: 'nonlinear', reason: res.reason, lam: res.lam, curve, steps, engine: engine.name, gpuNote: engine.note, removed, permanent, plastic: !!plastic, activeNode: L.activeNode });
  } finally {
    engine.destroy();
  }
}

// ---------- drop test ----------

async function drop(m, post) {
  const [nx, ny, nz] = m.dims;
  const NX = nx + 1, NY = ny + 1, NZ = nz + 1, nN = NX * NY * NZ;
  const fea = new VoxelFEA({ dims: m.dims, density: m.density, nu: m.nu, bc: new Uint8Array(3 * nN), coarsestMaxDof: Infinity });
  const L = fea.levels[0];
  if (!L.elems.length) throw new Error('The part produced no voxels.');
  // nodes that can touch the floor: active nodes with an empty neighbouring voxel
  const count = new Uint8Array(nN);
  for (let q = 0; q < L.elems.length; q++) for (let a = 0; a < 8; a++) count[L.base[q] + L.off[a]]++;
  const surface = new Uint8Array(nN);
  let lowest = Infinity;
  for (let n = 0; n < nN; n++) {
    if (!count[n] || count[n] === 8) continue;
    surface[n] = 1;
    const j = ((n / NX) | 0) % NY;
    lowest = Math.min(lowest, j);
  }
  const nodeY = new Float64Array(nN);
  for (let n = 0; n < nN; n++) nodeY[n] = ((((n / NX) | 0) % NY) - lowest) * m.h;
  const W = weights(m, L.activeNode);
  const toModel = 1 / m.toMeters;
  let mass = 0;
  for (let q = 0; q < L.elems.length; q++) mass += L.rho[q];
  mass *= m.rho * m.h ** 3;
  const speed = Math.sqrt(2 * 9.81 * m.height);
  const onFrame = (fr) => {
    const u = interpolate(W, fr.u, 3, toModel), vm = interpolate(W, fr.vm);
    post({ type: 'step', t: fr.t, force: fr.force, u, vm }, [u.buffer, vm.buffer]);
  };
  const onProgress = (frac) => post({ type: 'progress', stage: `Drop test · ${Math.round(frac * 100)}%`, it: 0, res: 1, frac });
  let r = null, engine = 'CPU', gpuNote = null;
  if (m.engine !== 'cpu') {
    try {
      r = await GPUExplicit.run(fea, { E: m.E, rho: m.rho, h: m.h, speed, nodeY, surface, onFrame, onProgress });
      engine = 'GPU';
    } catch (err) {
      if (err?.cancelled) throw err;
      gpuNote = err?.message || String(err);
    }
  }
  if (!r) r = dropTestCPU(fea, { E: m.E, rho: m.rho, h: m.h, speed, nodeY, surface, onFrame, onProgress });
  const vmMax = interpolate(W, r.vmMax);
  const tPeak = interpolate(W, r.tPeak);
  post({
    type: 'done', study: 'drop', engine, gpuNote, vmMax, tPeak, speed, mass,
    times: Float32Array.from(r.history, (p) => p.t), forces: Float32Array.from(r.history, (p) => p.force),
    peakForce: r.peakForce, contactTime: r.contactTime, duration: r.duration, steps: r.steps, dt: r.dt, rebounded: r.rebounded,
    activeNode: L.activeNode,
  }, [vmMax.buffer, tPeak.buffer]);
}

// ---------- optimization ----------

async function topology(m, post) {
  let engineName = 'CPU', note = null;
  const res = await optimizeTopology({
    dims: m.dims, fill: m.density, keep: m.keep, volFrac: m.volFrac, maxIter: m.maxIter || 40, rmin: m.rmin || 1.5,
    solve: async (density, x0) => {
      const fea = new VoxelFEA({ dims: m.dims, density, nu: m.nu, bc: m.bc, ...(m.engine !== 'cpu' ? { coarsestMaxDof: GPU_COARSEST_DOF } : {}) });
      const engine = await engineFor(m, fea);
      engineName = engine.name;
      note = engine.note;
      try {
        const sol = await engine.solve(m.f, { tol: 1e-5, maxIter: 3000, x0 });
        return { u: sol.u, fea, f: m.f };
      } finally {
        engine.destroy();
      }
    },
    onIter: (s) => {
      const density = new Float32Array(m.density.length);
      for (let q = 0; q < s.solid.length; q++) density[s.solid[q]] = s.xPhys[q];
      post({ type: 'step', it: s.it, compliance: s.compliance, volume: s.volume, change: s.change, density }, [density.buffer]);
      post({ type: 'progress', stage: `Topology on the ${engineName} · iteration ${s.it + 1}`, it: s.it, res: 1, frac: (s.it + 1) / (m.maxIter || 40) });
    },
  });
  post({ type: 'done', study: 'topology', density: res.density, history: res.history, keptFraction: res.keptFraction, engine: engineName, gpuNote: note }, [res.density.buffer]);
}

/**
 * Lightest (or cheapest) material and size that keeps the safety factor. Stresses are linear in
 * the loads, so each load group is solved once and rescaled: for geometric scale s and a material
 * of density rho, force loads give stress / s^2, pressure loads keep their stress, and self-weight
 * gives stress * s * rho / rho0 (displacements also scale with 1/E).
 */
async function sizing(m, post) {
  const { fea, removed } = buildModel(m);
  const engine = await engineFor(m, fea);
  const L = fea.levels[0];
  const groups = [];
  try {
    for (const g of m.groups) {
      progress(post, `Solving load group “${g.name}” on the ${engine.name}`, groups.length / m.groups.length);
      const sol = await engine.solve(g.f, { tol: 1e-7, maxIter: 3000 });
      const u = sol.u.map((v) => v / (m.E * m.h));
      groups.push({ ...g, S: fea.stressTensors(u, m.E, m.h), u });
    }
  } finally {
    engine.destroy();
  }
  // candidate nodes: the most stressed / displaced of each group (the combination's peak is there
  // unless the bound check below says otherwise)
  const act = [];
  for (let n = 0; n < L.nNodes; n++) if (L.activeNode[n]) act.push(n);
  const vmOf = (S, n) => vonMisesAt(S, 6 * n);
  const cand = new Set();
  const thr = [];
  for (const g of groups) {
    const vals = act.map((n) => vmOf(g.S, n));
    const order = act.map((_, i) => i).sort((a, b) => vals[b] - vals[a]);
    const K = Math.min(order.length, 2500);
    for (let i = 0; i < K; i++) cand.add(act[order[i]]);
    thr.push(K < order.length ? vals[order[K]] : 0);
    const dv = act.map((n) => Math.hypot(g.u[3 * n], g.u[3 * n + 1], g.u[3 * n + 2]));
    const dorder = act.map((_, i) => i).sort((a, b) => dv[b] - dv[a]);
    for (let i = 0; i < Math.min(dorder.length, 500); i++) cand.add(act[dorder[i]]);
  }
  const nodes = Int32Array.from(cand);
  const s6 = new Float64Array(6), pr = [0, 0, 0];
  const E0 = m.E, rho0 = m.rho;
  const coefs = (s, mat) => groups.map((g) => (g.kind === 'force' ? 1 / (s * s) : g.kind === 'gravity' ? (s * mat.density) / rho0 : 1));
  const dcoefs = (s, mat) => groups.map((g, i) => coefs(s, mat)[i] * s * (E0 / (mat.E * 1e9)));
  const evaluate = (s, mat) => {
    const c = coefs(s, mat);
    let peak = 0;
    for (const n of nodes) {
      s6.fill(0);
      for (let gi = 0; gi < groups.length; gi++) { const S = groups[gi].S; for (let k = 0; k < 6; k++) s6[k] += c[gi] * S[6 * n + k]; }
      const v = mat.brittle ? principalStresses(s6[0], s6[1], s6[2], s6[3], s6[4], s6[5], pr)[0] : vonMisesAt(s6);
      if (v > peak) peak = v;
    }
    let bound = 0;
    groups.forEach((g, i) => { bound += Math.abs(c[i]) * thr[i]; });
    const strength = (mat.brittle ? mat.uts : mat.yield) * 1e6;
    const dc = dcoefs(s, mat);
    let disp = 0;
    for (const n of nodes) {
      let x = 0, y = 0, z = 0;
      for (let gi = 0; gi < groups.length; gi++) { const u = groups[gi].u; x += dc[gi] * u[3 * n]; y += dc[gi] * u[3 * n + 1]; z += dc[gi] * u[3 * n + 2]; }
      disp = Math.max(disp, Math.hypot(x, y, z));
    }
    return { fos: peak > 0 ? strength / Math.max(peak, mat.brittle ? 0 : 0) : Infinity, peak: Math.max(peak, 0), exact: peak >= bound, disp: disp / m.toMeters };
  };
  const ok = (r) => r.fos >= m.fosTarget && (!(m.maxDisp > 0) || r.disp <= m.maxDisp);
  const rows = [];
  const volume = m.volume; // model-unit^3 at scale 1
  m.materials.forEach((mat, mi) => {
    progress(post, `Comparing materials · ${mat.name}`, mi / m.materials.length);
    // scan scales on a log grid, then refine the smallest passing scale by bisection
    const [lo, hi] = m.scaleRange;
    const grid = [];
    for (let k = 0; k <= 48; k++) grid.push(lo * Math.pow(hi / lo, k / 48));
    let pass = -1;
    for (let k = 0; k < grid.length; k++) if (ok(evaluate(grid[k], mat))) { pass = k; break; }
    let s = null;
    if (pass === 0) s = grid[0];
    else if (pass > 0) {
      let a = grid[pass - 1], b = grid[pass];
      for (let it = 0; it < 30; it++) { const c = Math.sqrt(a * b); if (ok(evaluate(c, mat))) b = c; else a = c; }
      s = b;
    }
    if (s === null) { rows.push({ id: mat.id, name: mat.name, feasible: false }); return; }
    const r = evaluate(s, mat);
    const massKg = mat.density * volume * s ** 3 * m.toMeters ** 3;
    rows.push({ id: mat.id, name: mat.name, feasible: true, scale: s, fos: r.fos, disp: r.disp, mass: massKg, cost: massKg * (mat.cost ?? 0), exact: r.exact });
  });
  const base = evaluate(1, m.baseMaterial);
  post({ type: 'done', study: 'sizing', rows, base, engine: engine.name, gpuNote: engine.note, removed });
}

// ---------- heat transfer ----------

async function thermal(m, post) {
  let active = null, W = null;
  const history = [];
  const res = solveHeat({
    dims: m.dims, density: m.density, fixedNode: m.fixedNode, fixedValue: m.fixedValue, source: m.source,
    convH: m.convH, convT: m.convT, k: m.k, h: m.h, rhoCp: m.rhoCp, duration: m.duration, steps: m.steps, initial: m.initial,
    onStep: (i, t, T) => {
      if (!W) return;
      const v = interpolate(W, T);
      let lo = Infinity, hi = -Infinity;
      for (const x of v) if (!Number.isNaN(x)) { lo = Math.min(lo, x); hi = Math.max(hi, x); }
      history.push({ t, min: lo, max: hi });
      post({ type: 'step', i, t, T: v, min: lo, max: hi }, [v.buffer]);
      post({ type: 'progress', stage: `Heat transfer · step ${i}/${m.steps}`, it: i, res: 1, frac: i / m.steps });
    },
    // the solver is built before the first step; weights need its active nodes
    onBuilt: (solver) => { active = solver.levels[0].active; W = weights(m, active); },
  });
  if (!W) { active = res.solver.levels[0].active; W = weights(m, active); }
  const T = interpolate(W, res.T);
  const flux = interpolate(W, heatFlux(res.solver, res.T, m.k, m.h));
  let lo = Infinity, hi = -Infinity, hot = -1;
  T.forEach((x, v) => { if (!Number.isNaN(x)) { if (x < lo) lo = x; if (x > hi) { hi = x; hot = v; } } });
  if (!history.length) history.push({ t: 0, min: lo, max: hi });
  post({ type: 'done', study: 'thermal', T, flux, min: lo, max: hi, hot, history, converged: res.converged, activeNode: active }, [T.buffer, flux.buffer]);
}
