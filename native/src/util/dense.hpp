// Small dense kernels for the multigrid coarsest levels.
#pragma once

#include <cstdint>

namespace ps {

/** a . b with four partial sums (shorter dependency chains; not bitwise the sequential sum). */
inline double dot4(const double* a, const double* b, int64_t n) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    int64_t k = 0;
    for (; k + 4 <= n; k += 4) {
        s0 += a[k] * b[k];
        s1 += a[k + 1] * b[k + 1];
        s2 += a[k + 2] * b[k + 2];
        s3 += a[k + 3] * b[k + 3];
    }
    for (; k < n; k++) s0 += a[k] * b[k];
    return (s0 + s1) + (s2 + s3);
}

/**
 * Solves L L^T y = b in place (L: m x m Cholesky factor, row-major lower). Both passes read L by
 * rows: forward as dot products, backward as column sweeps (x_i, then y_k -= L_ik x_i for k < i).
 */
inline void choleskySolve(const double* L, double* y, int64_t m) {
    for (int64_t i = 0; i < m; i++) y[i] = (y[i] - dot4(L + i * m, y, i)) / L[i * m + i];
    for (int64_t i = m - 1; i >= 0; i--) {
        const double* row = L + i * m;
        const double x = y[i] / row[i];
        y[i] = x;
        for (int64_t k = 0; k < i; k++) y[k] -= row[k] * x;
    }
}

}  // namespace ps
