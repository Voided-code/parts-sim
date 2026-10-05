// Arrays that follow each other in time (the steps of a break test, the frames of a drop test) change
// little from one to the next. They are written with one shared range and each one predicted from the
// one before it, which makes them about half the size (see docs/psim-format.md, `base`).

/**
 * datas: Float32Arrays in time order. Returns encodeArrays specs: [{ name, data, enc: 'q16', range, base?, stride? }].
 * name(i) gives the array name of item i.
 */
export function seriesSpecs(datas, name, { stride = 1 } = {}) {
  let lo = Infinity, hi = -Infinity;
  for (const d of datas) for (let i = 0; i < d.length; i++) { const v = d[i]; if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } }
  if (lo > hi) { lo = 0; hi = 0; }
  return datas.map((data, i) => ({ name: name(i), data, enc: 'q16', range: [lo, hi], ...(stride > 1 ? { stride } : {}), ...(i ? { base: name(i - 1) } : {}) }));
}
