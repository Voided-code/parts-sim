// A helper thread of BlockPool (pool.js): holds its own copy of the multigrid hierarchy and runs
// V-cycles or stiffness products on the vectors it is sent.
import { VoxelFEA } from './solver.js';

let fea = null;

self.onmessage = (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'init') {
      fea = new VoxelFEA(m.options);
      self.postMessage({ type: 'done', id: m.id });
      return;
    }
    const L = fea.levels[0];
    const out = m.vectors.map((x) => {
      const y = new Float64Array(L.nDof);
      if (m.type === 'precondition') fea.precondition(x, y);
      else fea.apply(L, x, y);
      return y;
    });
    self.postMessage({ type: 'done', id: m.id, vectors: out }, out.map((v) => v.buffer));
  } catch (err) {
    self.postMessage({ type: 'error', id: m.id, message: err?.message || String(err) });
  }
};
