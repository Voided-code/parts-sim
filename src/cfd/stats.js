// Statistics of a force history: the running mean and its 95% confidence interval.
//
// Successive force samples of a turbulent wake are correlated, so the naive standard error would
// be far too small. The history is split into batches that are long compared with the correlation
// time; the batch means are then nearly independent and their spread gives the uncertainty of the
// mean (the method of batch means).

// two-sided 95% Student-t quantiles for 1..30 degrees of freedom
const T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16, 2.145, 2.131, 2.12, 2.11, 2.101,
  2.093, 2.086, 2.08, 2.074, 2.069, 2.064, 2.06, 2.056, 2.052, 2.048, 2.045, 2.042];
export const t95 = (dof) => (dof < 1 ? Infinity : dof <= 30 ? T95[dof - 1] : 1.96 + 2.4 / dof);

/** Lag-1 autocorrelation of a series. */
function lag1(xs, mean) {
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    const d = xs[i] - mean;
    den += d * d;
    if (i) num += d * (xs[i - 1] - mean);
  }
  return den > 0 ? num / den : 0;
}

/**
 * Mean and 95% confidence half-width of a correlated series by batch means. The batch count halves
 * (down to 4) while neighbouring batch means are still correlated.
 * @returns {{mean: number, ci: number, batches: number, drift: number}} drift = second-half mean minus first-half mean
 */
export function batchMeans(xs, maxBatches = 16) {
  const n = xs.length;
  let mean = 0;
  for (const x of xs) mean += x;
  mean /= n || 1;
  const half = n >> 1;
  let a = 0, b = 0;
  for (let i = 0; i < n; i++) i < half ? (a += xs[i]) : (b += xs[i]);
  const drift = n >= 2 ? b / (n - half) - a / half : 0;
  if (n < 8) return { mean, ci: Infinity, batches: 0, drift };
  let k = Math.min(maxBatches, Math.floor(n / 2));
  for (;;) {
    const size = Math.floor(n / k);
    const means = [];
    for (let j = 0; j < k; j++) {
      let s = 0;
      for (let i = n - (k - j) * size; i < n - (k - j - 1) * size; i++) s += xs[i];
      means.push(s / size);
    }
    let m = 0;
    for (const v of means) m += v;
    m /= k;
    if (k > 4 && lag1(means, m) > 0.3) { k = Math.max(4, k >> 1); continue; }
    let v = 0;
    for (const x of means) v += (x - m) ** 2;
    const se = Math.sqrt(v / (k - 1) / k);
    return { mean, ci: t95(k - 1) * se, batches: k, drift };
  }
}
