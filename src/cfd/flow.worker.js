// Runs the v1 CPU flow solver (flow-cpu.js) off the main thread, with helper threads where memory
// can be shared, and streams its forces and reduced view fields (the GPU solver's counterparts).
import { FlowCPU } from './flow-cpu.js';
import { FlowThreads } from './flow-threads.js';

let sim = null;
let running = false;
let timer = 0;
let cf = 1;
let avg = null, samples = 0, lastFields = 0;

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      running = false;
      clearTimeout(timer);
      sim?.threads?.stop();
      sim = null;
      const next = new FlowCPU(m.grid, m.params);
      await FlowThreads.start(next);
      sim = next;
      cf = m.factor || 1;
      avg = null;
      samples = 0;
      self.postMessage({ type: 'ready', threads: !!sim.threads });
      if (running) loop();
    } else if (m.type === 'run') {
      if (!running) { running = true; loop(); }
    } else if (m.type === 'pause') {
      running = false;
      clearTimeout(timer);
    } else if (m.type === 'reset') {
      sim?.reset();
      avg = null;
      samples = 0;
    } else if (m.type === 'resetAverages') {
      sim?.resetAverages();
      avg = null;
      samples = 0;
    } else if (m.type === 'surface') {
      const rho = sim.surfaceRho();
      self.postMessage({ type: 'surface', rho }, [rho.buffer]);
    }
  } catch (err) { fail(err); }
};

function fail(err) {
  running = false;
  clearTimeout(timer);
  self.postMessage({ type: 'error', message: err.message || String(err) });
}

function loop() {
  if (!running || !sim) return;
  try {
    const t0 = performance.now();
    let n = 0;
    while (performance.now() - t0 < 150) { sim.step(1); n++; }
    const ms = performance.now() - t0;
    const f = sim.takeForces();
    self.postMessage({ type: 'forces', f, steps: sim.steps, mlups: (sim.N * n) / (ms * 1000) });
    // the view fields about twice a second (their time average kept here)
    if (performance.now() - lastFields > 500) {
      lastFields = performance.now();
      const fields = sim.coarseFields(cf);
      const inst = fields.inst;
      if (!avg) avg = new Float64Array(inst.length);
      for (let i = 0; i < inst.length; i++) avg[i] += inst[i];
      samples++;
      const mean = new Float32Array(inst.length);
      for (let i = 0; i < inst.length; i++) mean[i] = inst[i & ~3] === -2 ? (i & 3 ? 0 : -2) : avg[i] / samples;
      self.postMessage({ type: 'fields', dims: fields.dims, factor: cf, inst, avg: mean, samples }, [inst.buffer, mean.buffer]);
    }
    timer = setTimeout(loop, 0);
  } catch (err) { fail(err); }
}
