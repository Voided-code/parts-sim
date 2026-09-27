// Shared-memory threads for the finest level's work on large models: the stiffness product (every
// conjugate-gradient iteration, smoothing step and 64-bit residual check) and the stress recovery.
//
// The level's arrays are copied once into SharedArrayBuffers; each call then splits the nodes into
// chunks that the helper workers and the calling thread take in turn, and the helpers wait on
// Atomics between calls. Results are the same as on one thread (each node's row is computed the
// same way). SharedArrayBuffer needs cross-origin isolation, which the desktop app and the dev and
// preview servers provide; elsewhere the solvers simply stay on one thread.
import { applyNodes, stressNodes } from './solver.js';

// control words
const GEN = 0, OP = 1, DONE = 2, NEXT = 3, RAW = 4;
export const OPS = { apply: 1, stresses: 2 };
const CHUNKS_PER_THREAD = 8;

/** The share of the finest level that the threads need, backed by SharedArrayBuffers. */
function sharedLevel(L) {
  const share = (a) => {
    if (!a) return a;
    const c = new a.constructor(new SharedArrayBuffer(a.byteLength));
    c.set(a);
    return c;
  };
  const { NX, NY, NZ, nx, ny, nz, nNodes, nDof } = L;
  return {
    NX, NY, NZ, nx, ny, nz, nNodes, nDof, K: null,
    off: L.off, S: L.S,
    emap: share(L.emap), nodeScale: share(L.nodeScale), activeNode: share(L.activeNode), fixed: share(L.fixed),
    rho: share(L.rho), diagAdd: share(L.diagAdd),
  };
}

/** Takes chunks of the nodes until none are left (helpers and the calling thread alike). */
export function work(s) {
  const { ctrl, level, chunks } = s;
  const op = Atomics.load(ctrl, OP), raw = ctrl[RAW] === 1, n = level.nNodes;
  for (;;) {
    const c = Atomics.add(ctrl, NEXT, 1);
    if (c >= chunks) break;
    const n0 = Math.floor((n * c) / chunks), n1 = Math.floor((n * (c + 1)) / chunks);
    if (op === OPS.apply) applyNodes(level, s.K0, s.x, s.y, raw, n0, n1);
    else stressNodes(level, s.C, s.scale[0], s.x, s.vm, s.p1, s.p3, n0, n1);
  }
}

export class Threads {
  /**
   * Helper threads for fea's finest level, or null when memory cannot be shared, the machine has
   * one core or the model is too small to be worth it.
   */
  static async start(fea, { minDof = 60000, maxThreads = 15 } = {}) {
    if (typeof SharedArrayBuffer === 'undefined' || !globalThis.crossOriginIsolated || typeof Worker === 'undefined') return null;
    const L = fea.levels[0];
    const count = Math.min(maxThreads, (navigator.hardwareConcurrency || 1) - 1);
    if (count < 1 || L.nDof < minDof) return null;
    const vec = (Type, len) => new Type(new SharedArrayBuffer(len * Type.BYTES_PER_ELEMENT));
    const s = {
      ctrl: vec(Int32Array, 8),
      level: sharedLevel(L),
      chunks: CHUNKS_PER_THREAD * (count + 1),
      K0: fea.K0,
      C: fea.cornerStress,
      scale: vec(Float64Array, 1),
      x: vec(Float64Array, L.nDof), y: vec(Float64Array, L.nDof),
      vm: vec(Float32Array, L.nNodes), p1: vec(Float32Array, L.nNodes), p3: vec(Float32Array, L.nNodes),
    };
    const workers = [];
    try {
      await Promise.all(Array.from({ length: count }, () => new Promise((resolve, reject) => {
        const w = new Worker(new URL('./threads.worker.js', import.meta.url), { type: 'module' });
        workers.push(w);
        w.onmessage = () => resolve();
        w.onerror = (e) => { e.preventDefault?.(); reject(new Error(e.message || 'Helper thread failed')); };
        w.postMessage(s);
      })));
    } catch {
      for (const w of workers) w.terminate();
      return null;
    }
    return new Threads(s, workers);
  }

  constructor(s, workers) {
    this.s = s;
    this.workers = workers;
  }

  run(op) {
    const { ctrl } = this.s;
    Atomics.store(ctrl, NEXT, 0);
    Atomics.store(ctrl, DONE, 0);
    Atomics.store(ctrl, OP, op);
    Atomics.add(ctrl, GEN, 1);
    Atomics.notify(ctrl, GEN);
    work(this.s);
    for (let d; (d = Atomics.load(ctrl, DONE)) < this.workers.length;) Atomics.wait(ctrl, DONE, d);
  }

  /** y = K x on the finest level (VoxelFEA.apply). */
  apply(x, y, raw) {
    const s = this.s;
    s.x.set(x);
    s.ctrl[RAW] = raw ? 1 : 0;
    this.run(OPS.apply);
    y.set(s.y);
  }

  /** VoxelFEA.stresses, with the corner stresses scaled by E / h. */
  stresses(u, scale) {
    const s = this.s;
    s.x.set(u);
    s.scale[0] = scale;
    s.vm.fill(0);
    s.p1.fill(0);
    s.p3.fill(0);
    this.run(OPS.stresses);
    return { nodeVM: s.vm.slice(), nodeP1: s.p1.slice(), nodeP3: s.p3.slice() };
  }

  stop() {
    for (const w of this.workers) w.terminate();
    this.workers = [];
  }
}

/** Starts threads for fea when they would help (see Threads.start); fea keeps using one thread otherwise. */
export async function useThreads(fea, options) {
  fea.threads = await Threads.start(fea, options);
  return fea.threads;
}
