// Shared-memory threads (core/threads.js) for the v1 CPU flow solver (flow-cpu.js): each step's
// three passes (bulk rows, wall records, faces) are split between the helpers and the calling
// thread. Cells of a pass touch disjoint memory (in-place streaming), so the results are the same
// as on one thread.
import { SharedThreads, availableThreads, takeChunks, OP_WORD, COUNT_WORD } from '../core/threads.js';
import { bulkRows, wallCells, faceCells, scratch } from './flow-cpu.js';

const CHUNKS_PER_THREAD = 8;
const PASSES = [bulkRows, wallCells, faceCells];

/** Runs this thread's share of call `gen` (helpers and the calling thread alike). */
export function work(s, gen) {
  const { ctrl, scal } = s;
  const pass = PASSES[ctrl[OP_WORD]], count = ctrl[COUNT_WORD];
  // this step's parity, inlet and belt speeds
  s.odd = scal[0];
  s.uin = scal[1];
  s.uBelt = scal[2];
  const t = s.scratch || (s.scratch = scratch());
  takeChunks(s, gen, (c, chunks) => pass(s, t, Math.floor((count * c) / chunks), Math.floor((count * (c + 1)) / chunks)));
}

const shared = (a) => {
  if (!ArrayBuffer.isView(a) || a.buffer instanceof SharedArrayBuffer) return a;
  const c = new a.constructor(new SharedArrayBuffer(Math.max(8, a.byteLength)));
  c.set(a);
  return c;
};

export class FlowThreads {
  /**
   * Helper threads for a FlowCPU (set as sim.threads), or null when memory cannot be shared, the
   * machine has too few cores or the grid is small. Moves the solver's arrays into shared memory.
   */
  static async start(sim, { minCells = 20000, maxThreads = 15, helpers = null } = {}) {
    const count = availableThreads(maxThreads, helpers);
    if (count < 1 || sim.N < minCells || sim.s.ab) return null;
    const s = sim.s;
    for (const k of ['F', 'kind', 'faces', 'aux', 'bb', 'acc', 'slip', 'tauWall', 'rhoSum']) s[k] = shared(s[k]);
    for (const k of Object.keys(s.rec)) s.rec[k] = shared(s.rec[k]);
    for (const k of Object.keys(s.bulk)) s.bulk[k] = shared(s.bulk[k]);
    for (const k of Object.keys(s.face)) s.face[k] = shared(s.face[k]);
    s.scal = new Float64Array(new SharedArrayBuffer(8 * 4));
    const state = { ...s, offsets: undefined, scratch: undefined };
    const pool = await SharedThreads.start(() => new Worker(new URL('./flow-threads.worker.js', import.meta.url), { type: 'module' }), state, count, work);
    if (!pool) return null;
    sim.threads = new FlowThreads(pool, CHUNKS_PER_THREAD * (count + 1));
    return sim.threads;
  }

  constructor(pool, maxChunks) {
    this.pool = pool;
    this.maxChunks = maxChunks;
  }

  /** FlowCPU.step's three passes on all threads. */
  step(sim) {
    const st = this.pool.s, s = sim.s;
    st.scal[0] = s.odd;
    st.scal[1] = s.uin;
    st.scal[2] = s.uBelt;
    const counts = [s.ny * s.nz, s.rec.count, s.faces.length];
    for (let op = 0; op < 3; op++) if (counts[op]) this.pool.run(op, counts[op], Math.min(this.maxChunks, counts[op]));
  }

  stop() {
    this.pool.stop();
  }
}
