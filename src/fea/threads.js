// Shared-memory threads (core/threads.js) for all but small models. The helpers and the calling
// thread share each step of the CPU solver: every product, smoothing step, transfer and vector
// update of the multigrid solve and preconditioner (solver.js's range kernels, on each level big
// enough to be worth it), the stress recovery, the nonlinear study's element loops and the
// eigenvalue solver's block operations (eigen.js's dense kernels, on vectors in a shared arena).
//
// The model's arrays move into SharedArrayBuffers once, in place, so the calling thread works on
// the same memory; a call then only sets an op code, a level, vectors (by their index in a table of
// shared vectors) and scalars. Results are the same as on one thread: each node's values are
// computed the same way, dot products add up fixed blocks of nodes, element loops write each
// element's own forces, which the nodes then sum in order, and the dense kernels sum fixed chunks.
import { SharedThreads, availableThreads, takeChunks, OP_WORD, COUNT_WORD, FIRST_FREE_WORD } from '../core/threads.js';
import { stressNodes, KERNELS, REDUCES, BLOCK, OP as KERNEL } from './solver.js';
import { forceElements, tangentElements, gatherNodes } from './nonlinear.js';
import { DENSE, geometricNodes } from './eigen.js';

// control words of a call: its level and up to four vectors (indices into s.vecs)
const LEVEL = FIRST_FREE_WORD, VEC = FIRST_FREE_WORD + 1, NVEC = 4;
// ops besides solver.js's range kernels (1-99) and eigen.js's dense kernels (dense + DENSE_OP)
export const OPS = { stresses: 100, nlForce: 101, nlTangent: 102, gather: 103, geometric: 105, dense: 110 };
const CHUNKS_PER_THREAD = 8;
const MIN_CHUNK = 64; // nodes or elements
// helpers whose model is done (Threads.release), kept for the job's next model (break test and
// optimization steps build a new model each time)
let spare = null;

/**
 * Moves the arrays of a level that the kernels use into shared memory, in place, and returns them
 * (activeNode only in a shared copy: studies post it to the page).
 */
function shareLevel(L, finest) {
  const share = (a) => {
    if (!a || a.buffer instanceof SharedArrayBuffer) return a;
    const c = new a.constructor(new SharedArrayBuffer(a.byteLength));
    c.set(a);
    return c;
  };
  for (const k of ['base', 'emap', 'nodeScale', 'fixed', 'invDiag', 'scale', 'kIdx', 'Kown', 'diagAdd']) L[k] = share(L[k]);
  if (finest) L.rho = L.scale;
  // the work vectors hold nothing between calls; the finest level also has the conjugate gradients'
  for (const k of finest ? ['r', 'z', 't', 'd', 'x', 'p', 'q'] : ['r', 'z', 't', 'd']) L[k] = new Float64Array(new SharedArrayBuffer(8 * L.nDof));
  const { NX, NY, NZ, nx, ny, nz, nNodes, nDof, off, S, Kb, base, emap, nodeScale, fixed, invDiag, scale, rho, kIdx, Kown, diagAdd, r, z, t, d } = L;
  return { NX, NY, NZ, nx, ny, nz, nNodes, nDof, off, S, Kb, base, emap, nodeScale, activeNode: share(L.activeNode), fixed, invDiag, scale, rho, kIdx, Kown, diagAdd, r, z, t, d };
}

/** The vector lists of a dense call: views of the arena, by slot. */
function denseLists(s) {
  const d = s.dlist, n = s.arenaN, views = (s.views ??= []);
  const lists = [];
  for (let l = 0, o = 1; l < d[0]; l++) {
    const V = [];
    for (let q = d[o++]; q > 0; q--) {
      const i = d[o++];
      V.push((views[i] ??= s.arena.subarray(i * n, (i + 1) * n)));
    }
    lists.push(V);
  }
  return lists;
}

/** Runs this thread's share of call `gen` (helpers and the calling thread alike). */
export function work(s, gen) {
  const { ctrl } = s;
  const op = ctrl[OP_WORD], count = ctrl[COUNT_WORD], l = ctrl[LEVEL];
  const blocks = REDUCES.has(op) ? Math.ceil(count / BLOCK) : 0;
  const v = [];
  for (let k = 0; k < NVEC; k++) v.push(ctrl[VEC + k] >= 0 ? s.vecs[ctrl[VEC + k]] : null);
  const kernel = KERNELS[op], dense = op >= OPS.dense ? DENSE[op - OPS.dense] : null;
  const lists = dense ? denseLists(s) : null;
  takeChunks(s, gen, (c, chunks) => {
    let n0, n1;
    if (blocks) { // reducing kernels: whole blocks
      n0 = BLOCK * Math.floor((blocks * c) / chunks);
      n1 = Math.min(count, BLOCK * Math.floor((blocks * (c + 1)) / chunks));
    } else {
      n0 = Math.floor((count * c) / chunks);
      n1 = Math.floor((count * (c + 1)) / chunks);
    }
    if (kernel) kernel(s.levels, l, v, s.scal, s.part, n0, n1);
    else if (dense) dense(lists, s.dcoef, s.dint, s.dpart, s.arenaN, n0, n1);
    else if (op === OPS.stresses) stressNodes(s.levels[0], s.C, s.scal[0], s.x, s.vm, s.p1, s.p3, n0, n1);
    else if (op === OPS.nlForce) s.chunkMax[c] = forceElements(s.nl, s.levels[0], s.x, s.fe, n0, n1);
    else if (op === OPS.nlTangent) tangentElements(s.nl, s.levels[0], s.x, s.fe, n0, n1);
    else if (op === OPS.gather) gatherNodes(s.levels[0], s.fe, s.y, n0, n1);
    else if (op === OPS.geometric) geometricNodes(s.levels[0], s.kg, s.x, s.y, n0, n1);
  });
}

/** Resets a thread's arena views when a new arena arrives. */
export function onShare(s, extra) {
  if ('arena' in extra) s.views = [];
}

export class Threads {
  /**
   * Helper threads for fea, or null when memory cannot be shared, the machine has too few cores
   * (availableThreads) or the model is too small to be worth it. `helpers` sets their number.
   */
  static async start(fea, { minDof = 20000, maxThreads = 15, helpers = null } = {}) {
    const count = availableThreads(maxThreads, helpers);
    const L0 = fea.levels[0];
    if (count < 1 || L0.nDof < minDof) return null;
    const vec = (Type, len) => new Type(new SharedArrayBuffer(len * Type.BYTES_PER_ELEMENT));
    const maxChunks = CHUNKS_PER_THREAD * (count + 1);
    const s = {
      levels: fea.levels.map((L, l) => shareLevel(L, l === 0)),
      scal: vec(Float64Array, 8),
      part: vec(Float64Array, Math.ceil(L0.nNodes / BLOCK)),
      chunkMax: vec(Float64Array, maxChunks),
      C: fea.cornerStress,
      x: vec(Float64Array, L0.nDof), y: vec(Float64Array, L0.nDof),
      vm: vec(Float32Array, L0.nNodes), p1: vec(Float32Array, L0.nNodes), p3: vec(Float32Array, L0.nNodes),
    };
    // the vectors that calls name by index: a scratch pair for the callers' own vectors, then every
    // level's work vectors
    s.vecs = [s.x, s.y];
    for (const L of fea.levels) for (const k of ['r', 'z', 't', 'd', 'x', 'p', 'q']) if (L[k]) s.vecs.push(L[k]);
    let pool;
    if (spare?.workers.length === count) {
      // the last model's helpers: this model's state replaces all of that model's
      pool = spare;
      spare = null;
      const extra = { nl: null, fe: null, kg: null, arena: null, dlist: null, dcoef: null, dint: null, dpart: null, ...s };
      await pool.share(extra);
      onShare(pool.s, extra);
    } else {
      pool = await SharedThreads.start(() => new Worker(new URL('./threads.worker.js', import.meta.url), { type: 'module' }), s, count, work);
    }
    return pool && new Threads(pool, maxChunks, fea);
  }

  constructor(pool, maxChunks, fea) {
    this.pool = pool;
    this.s = pool.s;
    this.maxChunks = maxChunks;
    this.fea = fea;
    this.ids = new Map(this.s.vecs.map((v, i) => [v, i]));
  }

  /**
   * The model is done with its helpers: it goes back to one thread, and they wait for the job's
   * next model (start()).
   */
  release() {
    if (this.fea.threads === this) this.fea.threads = null;
    if (spare && spare !== this.pool) spare.stop();
    spare = this.pool;
  }

  /** Runs op on count nodes, elements or chunks (dense ops) on all threads; its arguments are set. */
  run(op, count) {
    const units = op >= OPS.dense ? count : Math.ceil(count / (REDUCES.has(op) ? BLOCK : MIN_CHUNK));
    this.pool.run(op, count, Math.min(this.maxChunks, units));
  }

  /** VoxelFEA.each: range kernel op on count nodes of level l with shared vectors v and scalars sc. */
  each(op, l, v, sc, count) {
    const { ctrl, scal } = this.s;
    for (let k = 0; k < NVEC; k++) {
      const id = k < v.length ? this.ids.get(v[k]) : -1;
      if (id === undefined) throw new Error('Helper threads: the vector is not shared.');
      ctrl[VEC + k] = id;
    }
    for (let k = 0; k < sc.length; k++) scal[k] = sc[k];
    ctrl[LEVEL] = l;
    this.run(op, count);
    return this.s.part;
  }

  /** y = K x on level l (VoxelFEA.apply); vectors that are not shared go through the scratch pair. */
  apply(l, x, y, raw) {
    const s = this.s;
    let xi = this.ids.get(x), yi = this.ids.get(y);
    if (xi === undefined) { s.x.set(x); xi = 0; }
    const copy = yi === undefined;
    if (copy) yi = 1;
    s.ctrl[VEC] = xi;
    s.ctrl[VEC + 1] = yi;
    s.ctrl[LEVEL] = l;
    s.scal[0] = raw ? 1 : 0;
    this.run(KERNEL.apply, s.levels[l].nNodes);
    if (copy) y.set(y.length === s.y.length ? s.y : s.y.subarray(0, y.length));
  }

  /** VoxelFEA.stresses, with the corner stresses scaled by E / h. */
  stresses(u, scale) {
    const s = this.s;
    s.x.set(u);
    s.scal[0] = scale;
    s.vm.fill(0);
    s.p1.fill(0);
    s.p3.fill(0);
    this.run(OPS.stresses, s.levels[0].nNodes);
    return { nodeVM: s.vm.slice(), nodeP1: s.p1.slice(), nodeP3: s.p3.slice() };
  }

  /** Adds shared state for later calls (SharedThreads.share). */
  async share(extra) {
    await this.pool.share(extra);
    onShare(this.s, extra);
  }

  /**
   * An eigen.js dense kernel on count chunks of vectors in the shared arena (share()), given by
   * their slots; returns the partial sums.
   */
  dense(op, slots, coef, ints, count) {
    const s = this.s, d = s.dlist;
    let o = 0;
    d[o++] = slots.length;
    for (const V of slots) {
      d[o++] = V.length;
      for (const i of V) d[o++] = i;
    }
    s.dcoef.set(coef);
    s.dint.set(ints);
    this.run(OPS.dense + op, count);
    return s.dpart;
  }

  /** NonlinearModel.internalForce with its state shared (NonlinearModel.share). */
  internalForce(u, out) {
    const s = this.s;
    s.x.set(u);
    this.run(OPS.nlForce, s.levels[0].base.length);
    let m = 0;
    for (let c = 0; c < this.pool.chunks; c++) if (s.chunkMax[c] > m) m = s.chunkMax[c];
    this.run(OPS.gather, s.levels[0].nNodes);
    out.set(s.y);
    return m;
  }

  /** GeometricStiffness.apply with its matrices shared (GeometricStiffness.share). */
  geometric(x, y) {
    const s = this.s;
    s.x.set(x);
    this.run(OPS.geometric, s.levels[0].nNodes);
    y.set(s.y);
  }

  /** NonlinearModel.applyTangent with its state shared. */
  applyTangent(x, y) {
    const s = this.s;
    s.x.set(x);
    this.run(OPS.nlTangent, s.levels[0].base.length);
    this.run(OPS.gather, s.levels[0].nNodes);
    y.set(s.y);
  }

  stop() {
    this.pool.stop();
  }
}

/** Starts threads for fea when they would help (see Threads.start); fea keeps using one thread otherwise. */
export async function useThreads(fea, options) {
  fea.threads = await Threads.start(fea, options);
  return fea.threads;
}
