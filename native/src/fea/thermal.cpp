#include "thermal.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "../util/parallel.hpp"
#include "eigen.hpp"
#include "hex8.hpp"

namespace ps {

namespace {

constexpr int SMOOTH_SWEEPS = 2;

// prolongation weights: fine corner a of child c from the coarse element's 8 corners
const std::array<std::array<std::array<double, 8>, 8>, 8>& childP() {
    static const auto P = [] {
        std::array<std::array<std::array<double, 8>, 8>, 8> all{};
        for (int c = 0; c < 8; c++) {
            const int ci = c & 1, cj = (c >> 1) & 1, ck = (c >> 2) & 1;
            for (int a = 0; a < 8; a++) {
                const double sx = (ci + HEX_NODES[a][0]) / 2.0, sy = (cj + HEX_NODES[a][1]) / 2.0, sz = (ck + HEX_NODES[a][2]) / 2.0;
                for (int b = 0; b < 8; b++) {
                    const int bx = HEX_NODES[b][0], by = HEX_NODES[b][1], bz = HEX_NODES[b][2];
                    all[c][a][b] = (bx ? sx : 1 - sx) * (by ? sy : 1 - sy) * (bz ? sz : 1 - sz);
                }
            }
        }
        return all;
    }();
    return P;
}

ScalarVoxelSolver::Level makeLevel(int nx, int ny, int nz) {
    ScalarVoxelSolver::Level L;
    L.nx = nx; L.ny = ny; L.nz = nz;
    L.NX = nx + 1; L.NY = ny + 1; L.NZ = nz + 1;
    const int64_t NX = L.NX, NXY = int64_t(L.NX) * L.NY;
    L.nNodes = NXY * L.NZ;
    L.off = {0, 1, 1 + NX, NX, NXY, 1 + NXY, 1 + NX + NXY, NX + NXY};
    return L;
}

void transferMap(int nFine, std::vector<int>& c0, std::vector<int>& c1) {
    const int n = nFine + 1;
    c0.resize(n);
    c1.resize(n);
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) { c0[i] = i / 2; c1[i] = -1; }
        else { c0[i] = (i - 1) / 2; c1[i] = (i + 1) / 2; }
    }
}

}  // namespace

const std::array<double, 64>& conductionMatrix() {
    static const auto K = [] {
        const auto& T = geometricTables();
        std::array<double, 64> k{};
        for (int i = 0; i < 64; i++) k[i] = T[0][i] + T[1][i] + T[2][i];
        return k;
    }();
    return K;
}

ScalarVoxelSolver::ScalarVoxelSolver(std::array<int, 3> dims, const std::vector<float>& density, const std::vector<uint8_t>& fixed,
                                     const std::vector<double>* diagAdd, int64_t coarsestMax) {
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    Level L0 = makeLevel(nx, ny, nz);
    if (density.size() != size_t(nx) * ny * nz || int64_t(fixed.size()) != L0.nNodes) throw std::invalid_argument("Thermal grid sizes do not match.");
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                const int e = i + nx * (j + ny * k);
                if (density[e] > 0) {
                    L0.elems.push_back(e);
                    L0.base.push_back(i + int64_t(L0.NX) * (j + int64_t(L0.NY) * k));
                    L0.rho.push_back(density[e]);
                }
            }
    L0.bc = fixed;
    if (diagAdd) L0.diagAdd = *diagAdd;
    finishLevel(L0);
    levels.push_back(std::move(L0));
    while (true) {
        Level& L = levels.back();
        if (L.freeDof <= coarsestMax || levels.size() >= 12 || (L.nx <= 1 && L.ny <= 1 && L.nz <= 1)) break;
        Level C = coarsen(L);
        levels.push_back(std::move(C));
    }
    for (size_t l = 0; l + 1 < levels.size(); l++) estimateOmega(levels[l]);
    buildCoarse(levels.back());
}

void ScalarVoxelSolver::finishLevel(Level& L) {
    const auto& K0 = conductionMatrix();
    L.active.assign(L.nNodes, 0);
    std::vector<double> diag(L.nNodes, 0.0);
    for (auto& c : L.colours) c.clear();
    for (size_t e = 0; e < L.elems.size(); e++) {
        for (int a = 0; a < 8; a++) {
            const int64_t n = L.base[e] + L.off[a];
            L.active[n] = 1;
            diag[n] += L.K.empty() ? L.rho[e] * K0[9 * a] : L.K[64 * e + 9 * a];
        }
        const int v = L.elems[e];
        const int i = v % L.nx, j = (v / L.nx) % L.ny, k = v / (L.nx * L.ny);
        L.colours[(i & 1) | ((j & 1) << 1) | ((k & 1) << 2)].push_back(int(e));
    }
    if (!L.diagAdd.empty())
        for (int64_t n = 0; n < L.nNodes; n++) diag[n] += L.diagAdd[n];
    L.fixed.assign(L.nNodes, 0);
    L.invDiag.assign(L.nNodes, 0.0);
    L.freeDof = 0;
    for (int64_t n = 0; n < L.nNodes; n++) {
        if (!L.active[n] || L.bc[n] || !(diag[n] > 0)) L.fixed[n] = 1;
        else { L.invDiag[n] = 1 / diag[n]; L.freeDof++; }
    }
    L.r.assign(L.nNodes, 0.0);
    L.z.assign(L.nNodes, 0.0);
    L.t.assign(L.nNodes, 0.0);
}

ScalarVoxelSolver::Level ScalarVoxelSolver::coarsen(Level& F) {
    Level C = makeLevel((F.nx + 1) >> 1, (F.ny + 1) >> 1, (F.nz + 1) >> 1);
    std::vector<int> cmap(size_t(C.nx) * C.ny * C.nz, -1), order, fc(F.elems.size());
    std::vector<uint8_t> child(F.elems.size());
    for (size_t q = 0; q < F.elems.size(); q++) {
        const int e = F.elems[q];
        const int i = e % F.nx, j = (e / F.nx) % F.ny, k = e / (F.nx * F.ny);
        const int ce = (i >> 1) + C.nx * ((j >> 1) + C.ny * (k >> 1));
        if (cmap[ce] < 0) { cmap[ce] = int(order.size()); order.push_back(ce); }
        fc[q] = cmap[ce];
        child[q] = uint8_t((i & 1) | ((j & 1) << 1) | ((k & 1) << 2));
    }
    C.elems = order;
    C.base.resize(order.size());
    for (size_t q = 0; q < order.size(); q++) {
        const int ce = order[q];
        C.base[q] = (ce % C.nx) + int64_t(C.NX) * (((ce / C.nx) % C.ny) + int64_t(C.NY) * (ce / (C.nx * C.ny)));
    }
    C.K.assign(64 * order.size(), 0.0);
    const auto& K0 = conductionMatrix();
    const auto& CP = childP();
    double T[64];
    for (size_t q = 0; q < F.elems.size(); q++) {
        const auto& P = CP[child[q]];
        const size_t o = 64 * size_t(fc[q]);
        const double* Kf = F.K.empty() ? K0.data() : &F.K[64 * q];
        const double s = F.K.empty() ? F.rho[q] : 1;
        // T = Kf P, then C += s P^T T
        std::fill(T, T + 64, 0.0);
        for (int a = 0; a < 8; a++)
            for (int b = 0; b < 8; b++) {
                const double kab = Kf[a * 8 + b];
                if (kab == 0) continue;
                for (int c = 0; c < 8; c++) T[a * 8 + c] += kab * P[b][c];
            }
        for (int a = 0; a < 8; a++)
            for (int d = 0; d < 8; d++) {
                const double w = P[a][d] * s;
                if (w == 0) continue;
                for (int c = 0; c < 8; c++) C.K[o + d * 8 + c] += w * T[a * 8 + c];
            }
    }
    transferMap(F.nx, F.mx0, F.mx1);
    transferMap(F.ny, F.my0, F.my1);
    transferMap(F.nz, F.mz0, F.mz1);
    C.bc.assign(C.nNodes, 0);
    for (int k = 0; k < F.NZ; k++)
        for (int j = 0; j < F.NY; j++)
            for (int i = 0; i < F.NX; i++) {
                if (!F.bc[i + int64_t(F.NX) * (j + int64_t(F.NY) * k)]) continue;
                for (int K : {F.mz0[k], F.mz1[k]})
                    if (K >= 0)
                        for (int J : {F.my0[j], F.my1[j]})
                            if (J >= 0)
                                for (int I : {F.mx0[i], F.mx1[i]})
                                    if (I >= 0) C.bc[I + int64_t(C.NX) * (J + int64_t(C.NY) * K)] = 1;
            }
    if (!F.diagAdd.empty()) {
        C.diagAdd.assign(C.nNodes, 0.0);
        restrict(F, C, F.diagAdd.data(), C.diagAdd.data(), false);
    }
    finishLevel(C);
    return C;
}

void ScalarVoxelSolver::apply(const Level& L, const double* x, double* y, bool raw) const {
    std::fill(y, y + L.nNodes, 0.0);
    const auto& K0 = conductionMatrix();
    for (const auto& list : L.colours)
        parallelFor(int64_t(list.size()), [&](int64_t lo, int64_t hi) {
            double xe[8];
            for (int64_t t = lo; t < hi; t++) {
                const int e = list[t];
                const int64_t n0 = L.base[e];
                for (int a = 0; a < 8; a++) xe[a] = x[n0 + L.off[a]];
                const double* Ke = L.K.empty() ? K0.data() : &L.K[64 * size_t(e)];
                const double s = L.K.empty() ? L.rho[e] : 1;
                for (int a = 0; a < 8; a++) {
                    double sum = 0;
                    for (int b = 0; b < 8; b++) sum += Ke[a * 8 + b] * xe[b];
                    y[n0 + L.off[a]] += s * sum;
                }
            }
        }, 2048);
    parallelFor(L.nNodes, [&](int64_t lo, int64_t hi) {
        for (int64_t n = lo; n < hi; n++) {
            if (!L.diagAdd.empty()) y[n] += L.diagAdd[n] * x[n];
            if (!raw && L.fixed[n]) y[n] = 0;
        }
    });
}

void ScalarVoxelSolver::estimateOmega(Level& L) {
    std::vector<double> v(L.nNodes), w(L.nNodes);
    uint32_t seed = 7;
    for (int64_t i = 0; i < L.nNodes; i++) {
        seed = (seed * 1103515245u + 12345u) & 0x7fffffffu;
        v[i] = L.fixed[i] ? 0 : double(seed) / 0x7fffffff - 0.5;
    }
    double lambda = 1;
    for (int it = 0; it < 12; it++) {
        double nv = std::sqrt(dot(v.data(), v.data(), L.nNodes));
        if (!(nv > 0)) nv = 1;
        for (auto& x : v) x /= nv;
        apply(L, v.data(), w.data());
        for (int64_t i = 0; i < L.nNodes; i++) w[i] *= L.invDiag[i];
        lambda = std::sqrt(dot(w.data(), w.data(), L.nNodes));
        std::swap(v, w);
    }
    L.omega = 1.2 / (lambda * 1.05);
}

void ScalarVoxelSolver::buildCoarse(Level& L) {
    auto& c = coarse_;
    c.map.assign(L.nNodes, -1);
    c.m = 0;
    for (int64_t n = 0; n < L.nNodes; n++)
        if (!L.fixed[n]) c.map[n] = c.m++;
    c.A.clear();
    if (c.m == 0 || c.m > 4000) return;
    const int64_t m = c.m;
    std::vector<double> A(size_t(m * m), 0.0);
    const auto& K0 = conductionMatrix();
    for (size_t e = 0; e < L.elems.size(); e++) {
        const double* Ke = L.K.empty() ? K0.data() : &L.K[64 * e];
        const double s = L.K.empty() ? L.rho[e] : 1;
        for (int a = 0; a < 8; a++) {
            const int64_t ga = c.map[L.base[e] + L.off[a]];
            if (ga < 0) continue;
            for (int b = 0; b < 8; b++) {
                const int64_t gb = c.map[L.base[e] + L.off[b]];
                if (gb >= 0) A[ga * m + gb] += s * Ke[a * 8 + b];
            }
        }
    }
    if (!L.diagAdd.empty())
        for (int64_t n = 0; n < L.nNodes; n++)
            if (c.map[n] >= 0) A[c.map[n] * m + c.map[n]] += L.diagAdd[n];
    double maxDiag = 0;
    for (int64_t i = 0; i < m; i++) maxDiag = std::max(maxDiag, A[i * m + i]);
    for (int64_t j = 0; j < m; j++) {
        const int64_t rj = j * m;
        double s = A[rj + j];
        for (int64_t k = 0; k < j; k++) s -= A[rj + k] * A[rj + k];
        if (!(s > 1e-12 * maxDiag)) s = maxDiag;
        const double d = std::sqrt(s);
        A[rj + j] = d;
        parallelFor(m - j - 1, [&](int64_t lo, int64_t hi) {
            for (int64_t i = j + 1 + lo; i < j + 1 + hi; i++) {
                const int64_t ri = i * m;
                double t = A[ri + j];
                for (int64_t k = 0; k < j; k++) t -= A[ri + k] * A[rj + k];
                A[ri + j] = t / d;
            }
        }, 64);
    }
    c.A = std::move(A);
    c.y.assign(m, 0.0);
}

void ScalarVoxelSolver::coarseSolve(Level& L) {
    auto& c = coarse_;
    std::fill(L.z.begin(), L.z.end(), 0.0);
    if (c.A.empty()) {
        for (int s = 0; s < 40; s++) jacobi(L, s == 0);
        return;
    }
    const int64_t m = c.m;
    for (int64_t n = 0; n < L.nNodes; n++)
        if (c.map[n] >= 0) c.y[c.map[n]] = L.r[n];
    for (int64_t i = 0; i < m; i++) {
        double s = c.y[i];
        for (int64_t k = 0; k < i; k++) s -= c.A[i * m + k] * c.y[k];
        c.y[i] = s / c.A[i * m + i];
    }
    for (int64_t i = m - 1; i >= 0; i--) {
        double s = c.y[i];
        for (int64_t k = i + 1; k < m; k++) s -= c.A[k * m + i] * c.y[k];
        c.y[i] = s / c.A[i * m + i];
    }
    for (int64_t n = 0; n < L.nNodes; n++)
        if (c.map[n] >= 0) L.z[n] = c.y[c.map[n]];
}

void ScalarVoxelSolver::jacobi(Level& L, bool first) {
    if (first) {
        parallelFor(L.nNodes, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.z[i] = L.omega * L.invDiag[i] * L.r[i]; });
        return;
    }
    apply(L, L.z.data(), L.t.data());
    parallelFor(L.nNodes, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.z[i] += L.omega * L.invDiag[i] * (L.r[i] - L.t[i]); });
}

void ScalarVoxelSolver::restrict(const Level& F, const Level& C, const double* rf, double* rc, bool zeroFixed) const {
    std::fill(rc, rc + C.nNodes, 0.0);
    // gather: each coarse node sums the fine nodes that interpolate from it
    parallelFor(C.NZ, [&](int64_t klo, int64_t khi) {
        for (int K = int(klo); K < int(khi); K++)
            for (int J = 0; J < C.NY; J++)
                for (int I = 0; I < C.NX; I++) {
                    double s = 0;
                    for (int k = std::max(0, 2 * K - 1); k <= std::min(F.NZ - 1, 2 * K + 1); k++) {
                        const double wk = F.mz1[k] < 0 ? (F.mz0[k] == K) : (F.mz0[k] == K || F.mz1[k] == K) ? 0.5 : 0;
                        if (wk == 0) continue;
                        for (int j = std::max(0, 2 * J - 1); j <= std::min(F.NY - 1, 2 * J + 1); j++) {
                            const double wj = F.my1[j] < 0 ? (F.my0[j] == J) : (F.my0[j] == J || F.my1[j] == J) ? 0.5 : 0;
                            if (wj == 0) continue;
                            for (int i = std::max(0, 2 * I - 1); i <= std::min(F.NX - 1, 2 * I + 1); i++) {
                                const double wi = F.mx1[i] < 0 ? (F.mx0[i] == I) : (F.mx0[i] == I || F.mx1[i] == I) ? 0.5 : 0;
                                if (wi == 0) continue;
                                s += wi * wj * wk * rf[i + int64_t(F.NX) * (j + int64_t(F.NY) * k)];
                            }
                        }
                    }
                    const int64_t cn = I + int64_t(C.NX) * (J + int64_t(C.NY) * K);
                    rc[cn] = zeroFixed && C.fixed[cn] ? 0 : s;
                }
    }, 1);
}

void ScalarVoxelSolver::prolongAdd(const Level& F, const Level& C, const double* zc, double* zf) const {
    parallelFor(F.NZ, [&](int64_t klo, int64_t khi) {
        for (int k = int(klo); k < int(khi); k++)
            for (int j = 0; j < F.NY; j++)
                for (int i = 0; i < F.NX; i++) {
                    const int64_t fn = i + int64_t(F.NX) * (j + int64_t(F.NY) * k);
                    if (F.fixed[fn]) continue;
                    double s = 0;
                    for (int kk = 0; kk < 2; kk++) {
                        const int K = kk ? F.mz1[k] : F.mz0[k];
                        if (K < 0) continue;
                        const double wk = F.mz1[k] < 0 ? 1 : 0.5;
                        for (int jj = 0; jj < 2; jj++) {
                            const int J = jj ? F.my1[j] : F.my0[j];
                            if (J < 0) continue;
                            const double wj = F.my1[j] < 0 ? 1 : 0.5;
                            for (int ii = 0; ii < 2; ii++) {
                                const int I = ii ? F.mx1[i] : F.mx0[i];
                                if (I < 0) continue;
                                const double wi = F.mx1[i] < 0 ? 1 : 0.5;
                                s += wi * wj * wk * zc[I + int64_t(C.NX) * (J + int64_t(C.NY) * K)];
                            }
                        }
                    }
                    zf[fn] += s;
                }
    }, 1);
}

void ScalarVoxelSolver::vcycle(size_t l) {
    Level& L = levels[l];
    if (l == levels.size() - 1) { coarseSolve(L); return; }
    Level& C = levels[l + 1];
    jacobi(L, true);
    for (int s = 1; s < SMOOTH_SWEEPS; s++) jacobi(L, false);
    apply(L, L.z.data(), L.t.data());
    for (int64_t i = 0; i < L.nNodes; i++) L.t[i] = L.r[i] - L.t[i];
    restrict(L, C, L.t.data(), C.r.data());
    vcycle(l + 1);
    prolongAdd(L, C, C.z.data(), L.z.data());
    for (int s = 0; s < SMOOTH_SWEEPS; s++) jacobi(L, false);
}

ScalarVoxelSolver::Result ScalarVoxelSolver::solve(const std::vector<double>& f, const std::vector<double>& x0, double tol, int maxIter) {
    Level& L = levels[0];
    const int64_t n = L.nNodes;
    Result res;
    res.x = x0;
    for (int64_t i = 0; i < n; i++)
        if (L.fixed[i] && !L.active[i]) res.x[i] = 0;
    auto& x = res.x;
    std::vector<double> r(n), p(n), q(n);
    apply(L, x.data(), q.data(), true);
    double bn = 0;
    for (int64_t i = 0; i < n; i++) {
        r[i] = L.fixed[i] ? 0 : f[i] - q[i];
        if (!L.fixed[i]) bn += f[i] * f[i] + q[i] * q[i];
    }
    bn = std::sqrt(bn);
    if (!(bn > 0)) bn = 1;
    double rel = std::sqrt(dot(r.data(), r.data(), n)) / bn;
    int it = 0;
    if (rel <= tol) { res.residual = rel; res.converged = true; return res; }
    auto pre = [&] {
        std::copy(r.begin(), r.end(), L.r.begin());
        vcycle(0);
        return L.z.data();
    };
    const double* z = pre();
    std::copy(z, z + n, p.begin());
    double rz = dot(r.data(), z, n);
    for (; it < maxIter && rel > tol; it++) {
        apply(L, p.data(), q.data());
        const double pq = dot(p.data(), q.data(), n);
        if (!(pq > 0)) break;
        const double a = rz / pq;
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) { x[i] += a * p[i]; r[i] -= a * q[i]; } });
        rel = std::sqrt(dot(r.data(), r.data(), n)) / bn;
        if (rel <= tol) { it++; break; }
        z = pre();
        const double rzn = dot(r.data(), z, n);
        const double beta = rzn / rz;
        rz = rzn;
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) p[i] = z[i] + beta * p[i]; });
    }
    res.iterations = it;
    res.residual = rel;
    res.converged = rel <= tol * 10;
    return res;
}

std::vector<double> nodeCapacity(std::array<int, 3> dims, const std::vector<float>& density) {
    const int nx = dims[0], ny = dims[1], nz = dims[2], NX = nx + 1, NY = ny + 1;
    std::vector<double> c(size_t(NX) * NY * (nz + 1), 0.0);
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                const double d = density[i + size_t(nx) * (j + size_t(ny) * k)];
                if (!(d > 0)) continue;
                for (int a = 0; a < 8; a++) c[i + (a & 1) + size_t(NX) * (j + ((a >> 1) & 1) + size_t(NY) * (k + ((a >> 2) & 1)))] += d / 8;
            }
    return c;
}

std::vector<float> heatFlux(const ScalarVoxelSolver& s, const std::vector<double>& T, double k, double h) {
    const auto& L = s.levels[0];
    std::vector<float> out(L.nNodes, 0.f), w(L.nNodes, 0.f);
    for (size_t e = 0; e < L.elems.size(); e++) {
        const int64_t n0 = L.base[e];
        double gx = 0, gy = 0, gz = 0;
        for (int a = 0; a < 8; a++) {
            const double t = T[n0 + L.off[a]] / 4;  // centroid gradient of the trilinear field: +-1/4 per node
            gx += HEX_NODES[a][0] ? t : -t;
            gy += HEX_NODES[a][1] ? t : -t;
            gz += HEX_NODES[a][2] ? t : -t;
        }
        const double q = k * std::sqrt(gx * gx + gy * gy + gz * gz) / h;
        const double rho = L.rho[e];
        for (int a = 0; a < 8; a++) {
            out[n0 + L.off[a]] += float(rho * q);
            w[n0 + L.off[a]] += float(rho);
        }
    }
    for (size_t n = 0; n < out.size(); n++)
        if (w[n] > 0) out[n] /= w[n];
    return out;
}

HeatResult solveHeat(const HeatInput& in, const std::function<bool(int, double, const std::vector<double>&)>& onStep,
                     const std::function<void(const ScalarVoxelSolver&)>& onBuilt) {
    if (!(in.k > 0) || !(in.h > 0)) throw std::invalid_argument("Thermal conductivity and voxel size must be positive.");
    const double kh = in.k * in.h;
    const int64_t nNodes = int64_t(in.dims[0] + 1) * (in.dims[1] + 1) * (in.dims[2] + 1);
    std::vector<double> conv(nNodes), rhsBase(nNodes);
    bool hasSink = false;
    for (int64_t n = 0; n < nNodes; n++) {
        conv[n] = in.convH[n] / kh;
        rhsBase[n] = (in.source[n] + in.convH[n] * in.convT[n]) / kh;
        hasSink |= in.fixedNode[n] || conv[n] > 0;
    }
    const bool transient = in.duration > 0 && in.rhoCp > 0;
    if (!transient && !hasSink) throw std::invalid_argument("Steady state needs somewhere for the heat to go: add a fixed temperature or convection.");
    std::vector<double> x0(nNodes, transient ? in.initial : 20.0);
    for (int64_t n = 0; n < nNodes; n++)
        if (in.fixedNode[n]) x0[n] = in.fixedValue[n];
    HeatResult out;
    if (!transient) {
        out.solver = std::make_unique<ScalarVoxelSolver>(in.dims, in.density, in.fixedNode, &conv);
        if (onBuilt) onBuilt(*out.solver);
        auto sol = out.solver->solve(rhsBase, x0);
        out.T = std::move(sol.x);
        out.converged = sol.converged;
        out.iterations = sol.iterations;
        if (onStep) onStep(0, 0, out.T);
        return out;
    }
    const double dt = in.duration / in.steps;
    // heat capacity per node / (dt k h): rho cp h^3 fill/8 / (dt k h) = rho cp h^2 / (k dt) * fill/8
    const auto cap = nodeCapacity(in.dims, in.density);
    const double cs = in.rhoCp * in.h * in.h / (in.k * dt);
    std::vector<double> diag(nNodes);
    for (int64_t n = 0; n < nNodes; n++) diag[n] = conv[n] + cs * cap[n];
    out.solver = std::make_unique<ScalarVoxelSolver>(in.dims, in.density, in.fixedNode, &diag);
    if (onBuilt) onBuilt(*out.solver);
    out.T = x0;
    if (onStep && onStep(0, 0, out.T)) throw std::runtime_error("Cancelled");
    std::vector<double> rhs(nNodes);
    for (int s = 1; s <= in.steps; s++) {
        for (int64_t n = 0; n < nNodes; n++) rhs[n] = rhsBase[n] + cs * cap[n] * out.T[n];
        auto sol = out.solver->solve(rhs, out.T);
        out.converged = out.converged && sol.converged;
        out.iterations += sol.iterations;
        out.T = std::move(sol.x);
        if (onStep && onStep(s, s * dt, out.T)) throw std::runtime_error("Cancelled");
    }
    return out;
}

}  // namespace ps
