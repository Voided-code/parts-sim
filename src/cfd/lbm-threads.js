// Shared-memory threads (core/threads.js) for the CPU flow solver: the helpers and the calling
// thread share each step's rows. Every cell is computed as on one thread (each only reads the
// previous step), so the results are the same.
import { SharedThreads, availableThreads, takeChunks, COUNT_WORD, FIRST_FREE_WORD } from '../core/threads.js';
import { stepRows } from './lbm-cpu.js';

// control words of a step: which buffer holds the current populations, and whether to write macro
const SWAPPED = FIRST_FREE_WORD, MACRO = FIRST_FREE_WORD + 1;
const CHUNKS_PER_THREAD = 8;

/** Runs this thread's share of step `gen` (helpers and the calling thread alike). */
export function work(s, gen) {
  const { ctrl } = s;
  const rows = ctrl[COUNT_WORD], writeMacro = ctrl[MACRO] === 1;
  const f = ctrl[SWAPPED] ? s.g : s.f, g = ctrl[SWAPPED] ? s.f : s.g;
  const uin = s.scal[0], feqIn = s.scal.subarray(1, 20);
  takeChunks(s, gen, (c, chunks) => {
    stepRows(s.sim, f, g, uin, feqIn, writeMacro, Math.floor((rows * c) / chunks), Math.floor((rows * (c + 1)) / chunks));
  });
}

export class LBMThreads {
  /**
   * Helper threads for an LBMCPU (set as sim.threads), or null when memory cannot be shared, the
   * machine has too few cores (availableThreads) or the grid is too small to be worth it. Moves its
   * arrays into shared memory. `helpers` sets their number.
   */
  static async start(sim, { minCells = 20000, maxThreads = 15, helpers = null } = {}) {
    const count = availableThreads(maxThreads, helpers);
    if (count < 1 || sim.N < minCells) return null;
    const share = (a) => {
      if (!a || a.buffer instanceof SharedArrayBuffer) return a;
      const c = new a.constructor(new SharedArrayBuffer(a.byteLength));
      c.set(a);
      return c;
    };
    for (const k of ['f', 'g', 'macro', 'solid', 'links']) sim[k] = share(sim[k]);
    const s = { sim: sim.kernelState(), f: sim.f, g: sim.g, scal: new Float64Array(new SharedArrayBuffer(8 * 20)) };
    const pool = await SharedThreads.start(() => new Worker(new URL('./lbm-threads.worker.js', import.meta.url), { type: 'module' }), s, count, work);
    if (!pool) return null;
    sim.threads = new LBMThreads(pool, CHUNKS_PER_THREAD * (count + 1));
    return sim.threads;
  }

  constructor(pool, maxChunks) {
    this.pool = pool;
    this.maxChunks = maxChunks;
  }

  /** LBMCPU.step's rows on all threads. */
  step(sim, uin, feqIn, writeMacro) {
    const s = this.pool.s, rows = sim.dims[1] * sim.dims[2];
    s.scal[0] = uin;
    s.scal.set(feqIn, 1);
    s.ctrl[SWAPPED] = sim.f === s.f ? 0 : 1;
    s.ctrl[MACRO] = writeMacro ? 1 : 0;
    this.pool.run(0, rows, Math.min(this.maxChunks, rows));
  }

  stop() {
    this.pool.stop();
  }
}
