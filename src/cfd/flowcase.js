// Headless airflow runs for validation and benchmarks: builds the same wind tunnel as the app
// (planTunnel, voxelize, wall links), steps the solver until the forces have converged, and reports
// them in newtons and as coefficients with 95% confidence intervals.
import * as THREE from 'three';
import { buildPart } from '../core/mesh.js';
import { voxelize } from '../core/voxelize.js';
import { planTunnel, U_LAT, MIN_NU_LAT, V1_NU_FLOOR, DEFAULT_MARGINS, TURBULENT_RE } from './airflow.js';
export { V1_NU_FLOOR };
import { wallLinks } from './links.js';
import { pressureForceCoefficients } from './forces.js';
import { LBMCPU, RAMP_STEPS } from './lbm-cpu.js';
import { LBMGPU } from './lbm-gpu.js';
import { AIR, airNu, frictionReference } from './validation.js';
import { batchMeans } from './stats.js';
import { buildFlowGrid } from './flow.js';
import { FlowCPU } from './flow-cpu.js';
import { wallRayCaster } from './links.js';

const now = () => performance.now();
const TO_METERS = 0.001; // validation parts are in millimetres


/**
 * Mesh the tunnel for a case: the part, the grid and, for the v0.6 solver, the solid cells with
 * their wall links, or for the v1 engine its cell kinds and wall records.
 */
export function buildTunnel(c, { across = c.across, alpha, legacy = true, marginScale = 1 } = {}) {
  const part = buildPart({ positions: c.make(alpha), name: c.id });
  const h = (c.lref / TO_METERS) / across;
  // the tunnel's margins: the app's unless the case sets its own, optionally scaled (a margin sweep)
  const m = { ...(legacy ? { up: 1.5, down: 2.5, side: 0.9 } : DEFAULT_MARGINS), ...c.tunnel?.margins };
  const margins = { ...m, up: m.up * marginScale, down: m.down * marginScale, side: m.side * marginScale };
  const plan = planTunnel(part, new THREE.Vector3(1, 0, 0), 0, { h, ...c.tunnel, margins, spanCells: c.spanCells });
  const { q, dims, origin } = plan;
  const N = dims[0] * dims[1] * dims[2];
  if (!legacy) {
    const grid = buildFlowGrid(plan, part.tris, { voxelize, wallRayCaster: wallRayCaster(q, part.tris) });
    return { part, plan, grid, N };
  }
  if (plan.periodicSpan || plan.ground !== null) return { part, plan, N };
  const frac = voxelize(q, part.tris, { origin, h, dims }, 2);
  const solid = new Uint8Array(N);
  for (let i = 0; i < N; i++) solid[i] = frac[i] >= 0.5 ? 1 : 0;
  const { links } = wallLinks(q, part.tris, { origin, h, dims }, solid);
  return { part, plan, solid, links, N };
}

/**
 * A tunnel of about n cells with a cube in it (a quarter of the tunnel's height across, a quarter of
 * its length from the inlet): the benchmarks' grid for the v1 engine, walls half-way.
 */
export function syntheticFlowGrid(n) {
  const ny = Math.max(16, Math.round(Math.cbrt(n / 2.5))), nz = ny, nx = Math.max(16, Math.round(n / (ny * nz)));
  const s = Math.floor(ny / 4), lo = [Math.floor(nx / 4), (ny - s) >> 1, (nz - s) >> 1];
  const plan = { dims: [nx, ny, nz], h: 1, origin: [0, 0, 0], min: lo, max: lo.map((v) => v + s), q: null, ground: null, periodicSpan: false };
  // the "voxelizer" of the box: the solid cells of the cube in the sub-grid it is given
  const voxelizeBox = (_q, _tris, g) => {
    const frac = new Float32Array(g.dims[0] * g.dims[1] * g.dims[2]);
    for (let z = 0; z < g.dims[2]; z++) {
      for (let y = 0; y < g.dims[1]; y++) {
        for (let x = 0; x < g.dims[0]; x++) {
          const c = [x, y, z].map((v, a) => g.origin[a] + v + 0.5);
          if (c.every((v, a) => v > lo[a] && v < lo[a] + s)) frac[x + g.dims[0] * (y + g.dims[1] * z)] = 1;
        }
      }
    }
    return frac;
  };
  return buildFlowGrid(plan, null, { voxelize: voxelizeBox, wallRayCaster: null });
}

/** The v1 engine (flow-cpu.js, flow-gpu.js): forces from momentum exchange, kept on the solver. */
async function v1Engine(kind, t, params) {
  if (kind === 'webgpu') {
    const { FlowGPU } = await import('./flow-gpu.js');
    const sim = await FlowGPU.create(t.grid, params);
    return {
      name: `WebGPU (v1 ${sim.collision}${sim.half ? ', 16-bit' : ''})`,
      get steps() { return sim.steps; },
      step: (n) => sim.step(n),
      sample: () => sim.takeForces(),
      destroy: () => sim.destroy(),
    };
  }
  const sim = new FlowCPU(t.grid, params);
  return {
    name: `CPU (v1 ${params.collision || 'rr'})`,
    get steps() { return sim.steps; },
    async step(n) { sim.step(n); },
    async sample() { return sim.takeForces(); },
    destroy() {},
  };
}

/** The v0.6 solvers (pressure-only forces from the density field). */
async function legacyEngine(kind, t, nuLat) {
  const { plan, solid, links } = t;
  const params = { dims: plan.dims, solid, links, uLat: U_LAT, nuLat };
  if (kind === 'webgpu') {
    const sim = await LBMGPU.create(params);
    return {
      name: 'WebGPU (v0.6 solver)',
      get steps() { return sim.steps; },
      async step(n) {
        // submissions of about 40 ms (see flow-gpu.js SUBMIT_MS)
        let k = Math.max(1, Math.min(400, Math.round(2e7 / sim.N)));
        while (n > 0) {
          const m = Math.min(n, k), t0 = performance.now();
          await sim.step(m);
          n -= m;
          k = Math.max(1, Math.min(400, Math.round((m * 40) / Math.max(1, performance.now() - t0))));
        }
      },
      async sample() {
        const macro = await sim.readMacro();
        return { macro };
      },
      destroy: () => sim.destroy(),
    };
  }
  const sim = new LBMCPU(params);
  return {
    name: 'CPU (v0.6 solver)',
    get steps() { return sim.steps; },
    async step(n) { for (let s = 0; s < n; s++) sim.step(s === n - 1); },
    async sample() { return { macro: sim.macro }; },
    destroy() {},
  };
}

function unstable(macro) {
  for (let c = 0; c < macro.length; c += 4) {
    const u = Math.abs(macro[c + 1]) + Math.abs(macro[c + 2]) + Math.abs(macro[c + 3]);
    if (!(u < 0.6) || !(macro[c] > 0)) return true;
  }
  return false;
}

/**
 * Run one case to a converged force.
 * @param {object} c                     a case from validation.js
 * @param {object} o
 * @param {'webgpu'|'cpu'} o.engine
 * @param {number} [o.across]            cells across the reference length (default: the case's)
 * @param {number} [o.nuScale]           multiply the viscosity (1 = the solver's normal choice)
 * @param {number} [o.alpha]             angle of attack for wing sections
 * @param {number} [o.tol]               stop when the drag's 95% interval is within this fraction
 * @param {number} [o.maxFlowThroughs]   give up after this many flow-throughs
 * @param {(p: object) => void} [o.onProgress]
 */
export async function runCase(c, o = {}) {
  const t0 = now();
  const across = o.across || c.across;
  const t = buildTunnel(c, { across, alpha: o.alpha, legacy: !o.solver || o.solver === 'v0.6', marginScale: o.marginScale || 1 });
  const { plan } = t;
  const { dims, h } = plan;
  const hm = h * TO_METERS;
  const nu = airNu(AIR.density);
  const reynolds = (c.speed * c.lref) / nu;
  // lattice viscosity of real air at this grid and wind speed, and the solver's floor
  const nuReal = (nu * U_LAT) / (c.speed * hm);
  let nuLat = Math.max(nuReal, MIN_NU_LAT) * (o.nuScale || 1);
  const q = 0.5 * AIR.density * c.speed * c.speed;
  const span = plan.periodicSpan ? dims[2] * hm : 1;
  const result = {
    id: c.id, name: c.name, alpha: o.alpha ?? null, engine: o.engine, across, dims, cells: t.N, h_mm: h,
    reynolds, nuScale: o.nuScale || 1, viscosityDoublings: 0, steps: 0, setupSeconds: (now() - t0) / 1000,
    reference: c.ref, lref: c.lref, aref: c.aref, speed: c.speed, marginScale: o.marginScale || 1,
  };
  const legacy = !o.solver || o.solver === 'v0.6';
  if (legacy && (plan.periodicSpan || plan.ground !== null)) {
    return { ...result, skipped: plan.periodicSpan ? 'the v0.6 solver has no periodic span' : 'the v0.6 solver has no ground plane' };
  }
  if (!legacy) nuLat = Math.max(nuReal, o.nuFloor ?? V1_NU_FLOOR) * (o.nuScale || 1);
  // the app's boundary-layer choice: turbulent (wall model) from Re 5e5 on the part's length along the
  // wind, unless the run sets it
  const reynoldsLength = (c.speed * plan.L[0] * TO_METERS) / nu;
  const wallModel = !legacy && (o.wallModel ?? reynoldsLength >= TURBULENT_RE);
  Object.assign(result, { reynoldsLength, wallModel });
  const flowThrough = dims[0] / U_LAT;
  const devSteps = RAMP_STEPS + Math.round((o.developFlowThroughs ?? 1.5) * flowThrough);
  const sampleEvery = Math.max(50, Math.round(flowThrough / 40));
  const minAvg = Math.round((o.minAvgFlowThroughs ?? 2) * flowThrough);
  const maxSteps = Math.round((o.maxFlowThroughs ?? 10) * flowThrough);
  const tol = o.tol ?? 0.02;
  let engine = null;
  try {
    for (;;) {
      engine = legacy ? await legacyEngine(o.engine, t, nuLat) : await v1Engine(o.engine, t, { uLat: U_LAT, nuLat, collision: o.collision, wallModel, wallMode: o.wallMode, smagorinsky: o.smagorinsky });
      result.engineName = engine.name;
      const tRun = now();
      const series = { x: [], y: [], z: [] }, pseries = { x: [], y: [], z: [] }, fseries = { x: [], y: [], z: [] };
      let failed = false;
      while (engine.steps < maxSteps) {
        const n = engine.steps < devSteps ? Math.min(devSteps - engine.steps, 2000) : sampleEvery;
        await engine.step(n);
        if (engine.steps < devSteps) {
          if (!legacy) await engine.sample(); // discard the developing flow's forces
          o.onProgress?.({ phase: 'developing', steps: engine.steps, of: devSteps });
          continue;
        }
        let C, P = null;
        if (legacy) {
          const { macro } = await engine.sample();
          if (unstable(macro)) { failed = true; break; }
          const rho = new Float32Array(t.N);
          for (let i = 0; i < t.N; i++) rho[i] = macro[4 * i];
          C = pressureForceCoefficients(rho, t.solid, dims, U_LAT).C;
        } else {
          // mean force over the interval, in the same units: F / (0.5 U^2) (cells^2)
          const s = await engine.sample();
          const k = 1 / (0.5 * U_LAT * U_LAT * s.steps);
          C = s.me.map((v) => v * k);
          P = s.pressure.map((v) => v * k);
          const Fr = s.friction.map((v) => v * k);
          fseries.x.push(Fr[0]); fseries.y.push(Fr[1]); fseries.z.push(Fr[2]);
          if (!C.every(Number.isFinite) || s.unstable) { failed = true; break; }
          pseries.x.push(P[0]); pseries.y.push(P[1]); pseries.z.push(P[2]);
        }
        series.x.push(C[0]);
        series.y.push(C[1]);
        series.z.push(C[2]);
        if (engine.steps - devSteps >= minAvg && series.x.length >= 16) {
          // settled when the drag and lift coefficients' 95% intervals are within tol of their values,
          // or within 0.004 (drag) and 0.01 (lift) for small ones, and the halves agree as well
          const toC = (hm * hm * (plan.periodicSpan ? 1 / (dims[2] * hm) : 1)) / c.aref; // cells^2 -> coefficient
          const ok = [[series.x, 0.004], [series.y, 0.01]].every(([xs, floor]) => {
            const s = batchMeans(xs), band = Math.max(tol * Math.abs(s.mean * toC), floor);
            return s.ci * toC <= band && Math.abs(s.drift * toC) <= Math.max(s.ci * toC, band);
          });
          const s = batchMeans(series.x);
          o.onProgress?.({ phase: 'averaging', steps: engine.steps, ci: s.ci / Math.abs(s.mean) });
          if (ok) break;
        } else o.onProgress?.({ phase: 'averaging', steps: engine.steps });
      }
      if (failed) {
        // what the app does: double the viscosity and start again (at most three times)
        engine.destroy();
        engine = null;
        if (++result.viscosityDoublings > 3) return { ...result, error: 'unstable after three viscosity doublings' };
        nuLat *= 2;
        continue;
      }
      const k = (q * hm * hm) / span; // coefficient units (cell^2) -> N (or N per metre of span)
      const sx = batchMeans(series.x), sy = batchMeans(series.y), sz = batchMeans(series.z);
      const drag = sx.mean * k, lift = sy.mean * k, side = sz.mean * k;
      Object.assign(result, {
        steps: engine.steps,
        seconds: (now() - tRun) / 1000,
        flowThroughs: engine.steps / flowThrough,
        converged: engine.steps < maxSteps,
        nuLat, simReynolds: (U_LAT * (c.lref / hm)) / nuLat,
        samples: series.x.length,
        forces: legacy ? 'pressure only' : 'momentum exchange (pressure and friction)',
        drag, lift, side, dragCI: sx.ci * k, liftCI: sy.ci * k,
        cd: drag / (q * c.aref), cl: lift / (q * c.aref),
        cdCI: (sx.ci * k) / (q * c.aref), clCI: (sy.ci * k) / (q * c.aref),
      });
      if (!legacy) {
        const px = batchMeans(pseries.x), py = batchMeans(pseries.y), fx = batchMeans(fseries.x);
        // friction: the momentum exchanged along the walls; the pressure integral is a check
        Object.assign(result, { pressureIntegralDrag: px.mean * k, pressureIntegralLift: py.mean * k, frictionDrag: fx.mean * k, pressureDrag: (sx.mean - fx.mean) * k });
      }
      if (c.friction) {
        // skin friction: the momentum-exchange drag minus the pressure on the plate's ends
        result.cf = legacy ? null : result.frictionDrag / (q * c.aref);
        result.cfReference = frictionReference(c, reynolds);
      }
      return result;
    }
  } finally {
    engine?.destroy();
  }
}

/** Least-squares slope of y over x. */
export function slope(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n, my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) { num += (xs[i] - mx) * (ys[i] - my); den += (xs[i] - mx) ** 2; }
  return num / den;
}
