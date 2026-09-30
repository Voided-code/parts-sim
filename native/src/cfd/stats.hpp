// Statistics of a force history: the running mean and its 95% confidence interval (src/cfd/stats.js).
//
// Successive force samples of a turbulent wake are correlated, so the naive standard error would
// be far too small. The history is split into batches that are long compared with the correlation
// time; the batch means are then nearly independent and their spread gives the uncertainty of the
// mean (the method of batch means).
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace ps {

/** Two-sided 95% Student-t quantile for `dof` degrees of freedom. */
inline double t95(int dof) {
    static const double T[30] = {12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.16,  2.145, 2.131,
                                 2.12,   2.11,  2.101, 2.093, 2.086, 2.08,  2.074, 2.069, 2.064, 2.06,  2.056, 2.052, 2.048, 2.045, 2.042};
    if (dof < 1) return std::numeric_limits<double>::infinity();
    return dof <= 30 ? T[dof - 1] : 1.96 + 2.4 / dof;
}

struct BatchStats {
    double mean = 0, ci = std::numeric_limits<double>::infinity();
    int batches = 0;
    double drift = 0;  // second-half mean minus first-half mean
};

/**
 * Mean and 95% confidence half-width of a correlated series by batch means. The batch count halves
 * (down to 4) while neighbouring batch means are still correlated.
 */
inline BatchStats batchMeans(const std::vector<double>& xs, int maxBatches = 16) {
    BatchStats r;
    const int n = int(xs.size());
    for (double x : xs) r.mean += x;
    r.mean /= std::max(n, 1);
    const int half = n / 2;
    double a = 0, b = 0;
    for (int i = 0; i < n; i++) (i < half ? a : b) += xs[i];
    if (n >= 2) r.drift = b / (n - half) - a / half;
    if (n < 8) return r;
    int k = std::min(maxBatches, n / 2);
    for (;;) {
        const int size = n / k;
        std::vector<double> means(k);
        for (int j = 0; j < k; j++) {
            double s = 0;
            for (int i = n - (k - j) * size; i < n - (k - j - 1) * size; i++) s += xs[i];
            means[j] = s / size;
        }
        double m = 0;
        for (double v : means) m += v;
        m /= k;
        double num = 0, den = 0;
        for (int i = 0; i < k; i++) {
            const double d = means[i] - m;
            den += d * d;
            if (i) num += d * (means[i - 1] - m);
        }
        if (k > 4 && den > 0 && num / den > 0.3) {
            k = std::max(4, k >> 1);
            continue;
        }
        const double se = std::sqrt(den / (k - 1) / k);
        r.ci = t95(k - 1) * se;
        r.batches = k;
        return r;
    }
}

}  // namespace ps
