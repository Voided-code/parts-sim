// Runs the CPU multigrid preconditioner and stiffness products for blocks of vectors (the
// eigenvalue solvers' LOBPCG) on several threads. Each helper worker builds its own copy of the
// multigrid hierarchy from the model's inputs, and this thread takes a share of every block too.

/** Memory allowed for the helpers' copies of the hierarchy. */
const BUDGET = 768 * 2 ** 20;

export class BlockPool {
  /**
   * A pool for `fea` (a VoxelFEA), or null when threads are unavailable, the machine has a single
   * core, or the copies would not fit in memory.
   */
  static async create(fea, { maxThreads = 15 } = {}) {
    if (typeof Worker === 'undefined' || typeof navigator === 'undefined') return null;
    const bytes = hierarchyBytes(fea);
    const count = Math.min((navigator.hardwareConcurrency || 1) - 1, maxThreads, Math.floor(BUDGET / bytes));
    if (!(count >= 1)) return null;
    const pool = new BlockPool(fea);
    try {
      for (let i = 0; i < count; i++) pool.workers.push(pool.spawn());
      await Promise.all(pool.workers.map((w) => pool.request(w, { type: 'init', options: fea.options })));
      return pool;
    } catch {
      pool.destroy();
      return null;
    }
  }

  constructor(fea) {
    this.fea = fea;
    this.workers = [];
    this.pending = new Map();
    this.id = 0;
  }

  spawn() {
    const w = new Worker(new URL('./pool.worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (ev) => {
      const d = ev.data, p = this.pending.get(d.id);
      if (!p) return;
      this.pending.delete(d.id);
      if (d.type === 'done') p.resolve(d.vectors);
      else p.reject(new Error(d.message));
    };
    w.onerror = (e) => {
      e.preventDefault?.();
      for (const [id, p] of this.pending) if (p.worker === w) { this.pending.delete(id); p.reject(new Error(e.message || 'Helper thread failed')); }
    };
    return w;
  }

  request(worker, message) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject, worker });
      worker.postMessage({ ...message, id });
    });
  }

  /** One V-cycle for each full-length vector. */
  precondition(R) {
    return this.run('precondition', R, (r) => this.fea.precondition(r));
  }

  /** K x for each full-length vector (held DOFs zeroed). */
  apply(X) {
    const L = this.fea.levels[0];
    return this.run('apply', X, (x) => { const y = new Float64Array(L.nDof); this.fea.apply(L, x, y); return y; });
  }

  async run(type, V, local) {
    // contiguous shares, one per helper and one for this thread (the last, often smallest)
    const size = Math.ceil(V.length / (this.workers.length + 1));
    const jobs = [];
    for (let w = 0; w < this.workers.length && w * size < V.length; w++) {
      const share = V.slice(w * size, (w + 1) * size);
      // a helper that fails hands its share back to this thread
      jobs.push(this.request(this.workers[w], { type, vectors: share }).catch(() => share.map(local)));
    }
    const mine = V.slice(this.workers.length * size).map(local);
    return [...(await Promise.all(jobs)).flat(), ...mine];
  }

  destroy() {
    for (const w of this.workers) w.terminate();
    this.workers = [];
    for (const p of this.pending.values()) p.reject(new Error('Pool closed'));
    this.pending.clear();
  }
}

/** Rough size of a VoxelFEA's typed arrays (what each helper copies). */
function hierarchyBytes(fea) {
  let bytes = 0;
  const seen = new Set();
  for (const L of fea.levels) {
    for (const v of Object.values(L)) {
      if (ArrayBuffer.isView(v) && !seen.has(v.buffer)) { seen.add(v.buffer); bytes += v.byteLength; }
    }
  }
  return Math.max(bytes, 1);
}
