// Helper threads that share one call at a time over shared memory. The calling thread sets an op
// code and its arguments, wakes the helpers through Atomics, and then every thread, the caller
// included, takes chunks of the call until none are left. The caller waits only until every chunk
// is done, not for helpers that wake late: the chunk counter carries the call's tag (its
// generation), so a helper still looking for work from an earlier call cannot take a chunk of a
// later one. Between calls the helpers sleep in Atomics.wait.
//
// SharedArrayBuffer needs cross-origin isolation, which the desktop app and the dev and preview
// servers provide; elsewhere the solvers simply stay on one thread.
//
// Control words 0-7 belong to this module; users keep their own arguments from FIRST_FREE_WORD on.

const GEN = 0, OP = 1, NEXT = 2, DONE = 3, CHUNKS = 4, COUNT = 5, LISTENED = 6;
export const OP_WORD = OP, COUNT_WORD = COUNT, FIRST_FREE_WORD = 8;
// NEXT: the call's tag (its generation, 22 bits, which repeat only after four million calls) over
// the next chunk (10 bits: at most 1023 chunks a call)
const CHUNK_BITS = 10, CHUNK_MASK = (1 << CHUNK_BITS) - 1;
const tagOf = (gen) => gen << CHUNK_BITS;
// sends the helpers back to their event loop to receive more shared state (share())
const LISTEN = -1;

/** Whether this context can share memory with helper threads (cross-origin isolated, with workers). */
export function sharedMemory() {
  return typeof SharedArrayBuffer !== 'undefined' && !!globalThis.crossOriginIsolated && typeof Worker !== 'undefined';
}

/**
 * How many helper threads to start: all cores but two, one for the calling worker and one for the
 * page and the GPU process (with every core busy they starve: the window freezes during a solve on
 * a small machine), or none without shared memory. `helpers` overrides the count.
 */
export function availableThreads(maxThreads, helpers = null) {
  if (!sharedMemory()) return 0;
  return helpers ?? Math.max(0, Math.min(maxThreads, (globalThis.navigator?.hardwareConcurrency || 1) - 2));
}

/**
 * Calls fn(c, chunks) for each chunk c of call `gen` that this thread takes, until none are left
 * (helpers and the calling thread alike; a user's work(s, gen) calls this).
 */
export function takeChunks(s, gen, fn) {
  const { ctrl } = s;
  const tag = tagOf(gen), chunks = Atomics.load(ctrl, CHUNKS);
  for (;;) {
    const w = Atomics.load(ctrl, NEXT);
    if ((w & ~CHUNK_MASK) !== tag || (w & CHUNK_MASK) >= chunks) return;
    if (Atomics.compareExchange(ctrl, NEXT, w, w + 1) !== w) continue;
    fn(w & CHUNK_MASK, chunks);
    if (Atomics.add(ctrl, DONE, 1) === chunks - 1) Atomics.notify(ctrl, DONE);
  }
}

export class SharedThreads {
  /**
   * Starts `count` helpers from spawn() (module workers that call serve()) and gives them the
   * shared state s, which gets its control words here. work(s, gen) runs a call (takeChunks).
   * Resolves to null if the helpers cannot start.
   */
  static async start(spawn, s, count, work) {
    s.ctrl = new Int32Array(new SharedArrayBuffer(4 * 32));
    const workers = [];
    try {
      await Promise.all(Array.from({ length: count }, () => new Promise((resolve, reject) => {
        const w = spawn();
        workers.push(w);
        w.onmessage = () => resolve();
        w.onerror = (e) => { e.preventDefault?.(); reject(new Error(e.message || 'Helper thread failed')); };
        w.postMessage(s);
      })));
    } catch {
      for (const w of workers) w.terminate();
      return null;
    }
    return new SharedThreads(s, workers, work);
  }

  constructor(s, workers, work) {
    this.s = s;
    this.workers = workers;
    this.work = work;
    this.gen = 0;
    this.chunks = 0;
  }

  /** Runs op over `chunks` chunks on all threads; its arguments are set. count is passed along. */
  run(op, count, chunks) {
    const { ctrl } = this.s;
    chunks = this.chunks = Math.max(1, Math.min(CHUNK_MASK, chunks));
    const gen = (this.gen = (this.gen + 1) | 0);
    // the new tag before the call's chunk count: a helper still looking for work from the last call
    // that reads this count then also sees the new tag, so it cannot take a chunk with it
    Atomics.store(ctrl, DONE, 0);
    Atomics.store(ctrl, NEXT, tagOf(gen));
    Atomics.store(ctrl, OP, op);
    Atomics.store(ctrl, COUNT, count);
    Atomics.store(ctrl, CHUNKS, chunks);
    Atomics.store(ctrl, GEN, gen);
    Atomics.notify(ctrl, GEN);
    this.work(this.s, gen);
    for (let d; (d = Atomics.load(ctrl, DONE)) < chunks;) Atomics.wait(ctrl, DONE, d);
  }

  /**
   * Adds shared state for later calls (the helpers leave their wait loop to receive it). The values
   * must be SharedArrayBuffer views, or small constants, that the caller keeps using in place.
   */
  async share(extra) {
    const { ctrl } = this.s;
    Atomics.store(ctrl, LISTENED, 0);
    Atomics.store(ctrl, OP, LISTEN);
    Atomics.store(ctrl, GEN, (this.gen = (this.gen + 1) | 0));
    Atomics.notify(ctrl, GEN);
    for (let d; (d = Atomics.load(ctrl, LISTENED)) < this.workers.length;) Atomics.wait(ctrl, LISTENED, d);
    await Promise.all(this.workers.map((w) => new Promise((resolve) => {
      w.onmessage = () => resolve();
      w.postMessage(extra);
    })));
    Object.assign(this.s, extra);
  }

  stop() {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}

/**
 * The body of a helper worker: takes the shared state from the first message (and additions from
 * later ones, after which onShare(s, extra) runs), then runs calls with work(s, gen).
 */
export function serve(work, onShare = null) {
  let s = null;
  self.onmessage = (ev) => {
    if (s) Object.assign(s, ev.data);
    else s = ev.data;
    onShare?.(s, ev.data);
    const { ctrl } = s;
    let gen = Atomics.load(ctrl, GEN);
    self.postMessage('ready');
    for (;;) {
      Atomics.wait(ctrl, GEN, gen);
      gen = Atomics.load(ctrl, GEN);
      if (Atomics.load(ctrl, OP) === LISTEN) {
        Atomics.add(ctrl, LISTENED, 1);
        Atomics.notify(ctrl, LISTENED);
        return;
      }
      work(s, gen);
    }
  };
}
