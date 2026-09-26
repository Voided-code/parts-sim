// Runs the CPU lattice-Boltzmann solver off the main thread and streams flow snapshots.
import { LBMCPU } from './lbm-cpu.js';

let sim = null;
let running = false;
let timer = 0;

self.onmessage = (ev) => {
  const m = ev.data;
  try {
  if (m.type === 'init') {
    running = false;
    clearTimeout(timer);
    sim = new LBMCPU(m.params);
    self.postMessage({ type: 'ready' });
  } else if (m.type === 'run') {
    if (!running) {
      running = true;
      loop();
    }
  } else if (m.type === 'pause') {
    running = false;
    clearTimeout(timer);
  } else if (m.type === 'reset') {
    sim?.reset();
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
  while (performance.now() - t0 < 180) {
    sim.step(false);
    n++;
  }
  sim.step(true);
  n++;
  const ms = performance.now() - t0;
  const macro = sim.macro.slice();
  self.postMessage({ type: 'snapshot', macro, steps: sim.steps, mlups: (sim.N * n) / (ms * 1000) }, [macro.buffer]);
  timer = setTimeout(loop, 0);
  } catch (err) { fail(err); }
}
