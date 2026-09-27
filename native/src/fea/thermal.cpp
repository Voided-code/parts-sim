#include "thermal.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "../util/dense.hpp"
#include "../util/parallel.hpp"
#include "../util/trace.hpp"
#include "eigen.hpp"
#include "hex8.hpp"

namespace ps {

namespace {

inline int corner(int x, int y, int z) { return (y ? (x ? 2 : 3) : (x ? 1 : 0)) + 4 * z; }

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

namespace {

// K0 assembled around a node inside a uniform block: coefficient of the neighbour at offset
// (ox, oy, oz) - 1, index ox + 3 oy + 9 oz
const std::array<double, 27>& conductionStencil() {
    static const auto S = [] {
        const auto& K0 = conductionMatrix();
        std::array<double, 27> st{};
        for (int v = 0; v < 8; v++) {
            const int di = v & 1, dj = (v >> 1) & 1, dk = v >> 2, a = corner(1 - di, 1 - dj, 1 - dk);
            for (int w = 0; w < 8; w++) {
                const int bx = w & 1, by = (w >> 1) & 1, bz = w >> 2;
                st[(di + bx) + 3 * (dj + by) + 9 * (dk + bz)] += K0[a * 8 + corner(bx, by, bz)];
            }
        }
        return st;
    }();
    return S;
}

// P_c^T K0 P_c per child position c
const std::array<std::array<double, 64>, 8>& childGalerkin() {
    static const auto M = [] {
        const auto& K0 = conductionMatrix();
        const auto& CP = childP();
        std::array<std::array<double, 64>, 8> m{};
        for (int c = 0; c < 8; c++)
            for (int d = 0; d < 8; d++)
                for (int e = 0; e < 8; e++) {
                    double s = 0;
                    for (int a = 0; a < 8; a++)
                        for (int b = 0; b < 8; b++) s += CP[c][a][d] * K0[a * 8 + b] * CP[c][b][e];
                    m[c][d * 8 + e] = s;
                }
        return m;
    }();
    return M;
}

}  // namespace

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
                    L0.scale.push_back(density[e]);
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
    L.emap.assign(size_t(L.nx) * L.ny * L.nz, -1);
    L.voxelScale.assign(L.emap.size(), 0.f);
    for (size_t e = 0; e < L.elems.size(); e++) {
        for (int a = 0; a < 8; a++) {
            const int64_t n = L.base[e] + L.off[a];
            L.active[n] = 1;
            diag[n] += L.K.empty() ? L.rho[e] * K0[9 * a] : L.K[64 * e + 9 * a];
        }
        L.emap[L.elems[e]] = int(e);
        L.voxelScale[L.elems[e]] = float(L.scale[e]);
    }
    L.nodeScale.assign(L.nNodes, 0.f);
    for (int k = 1; k < L.nz; k++)
        for (int j = 1; j < L.ny; j++)
            for (int i = 1; i < L.nx; i++) {
                const float s0 = L.voxelScale[(i - 1) + size_t(L.nx) * ((j - 1) + size_t(L.ny) * (k - 1))];
                bool u = s0 > 0;
                for (int v = 1; v < 8 && u; v++)
                    u = L.voxelScale[(i - 1 + (v & 1)) + size_t(L.nx) * ((j - 1 + ((v >> 1) & 1)) + size_t(L.ny) * (k - 1 + (v >> 2)))] == s0;
                if (u) L.nodeScale[i + int64_t(L.NX) * (j + int64_t(L.NY) * k)] = s0;
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
    // Galerkin products per coarse element from its children (in parallel): s * P^T K0 P for the
    // s * K0 ones; a coarse element whose 8 children are the same s * K0 is 2s * K0
    const size_t nC = order.size();
    std::vector<int> start(nC + 1, 0), kids(F.elems.size());
    for (size_t q = 0; q < F.elems.size(); q++) start[fc[q] + 1]++;
    for (size_t c = 0; c < nC; c++) start[c + 1] += start[c];
    {
        std::vector<int> fill(start.begin(), start.end() - 1);
        for (size_t q = 0; q < F.elems.size(); q++) kids[fill[fc[q]]++] = int(q);
    }
    C.K.assign(64 * nC, 0.0);
    C.scale.assign(nC, -1.0);
    const auto& CP = childP();
    const auto& M0 = childGalerkin();
    parallelFor(int64_t(nC), [&](int64_t lo, int64_t hi) {
        double T[64];
        for (int64_t c = lo; c < hi; c++) {
            double* out = &C.K[64 * size_t(c)];
            const int b = start[c], e = start[c + 1];
            bool uniform = e - b == 8 && F.scale[kids[b]] > 0;
            for (int t = b; t < e; t++) {
                const int q = kids[t];
                uniform = uniform && F.scale[q] == F.scale[kids[b]];
                if (F.scale[q] > 0) {
                    const double s = F.scale[q];
                    const double* M = M0[child[q]].data();
                    for (int u = 0; u < 64; u++) out[u] += s * M[u];
                    continue;
                }
                // T = Kf P, then C += P^T T
                const auto& P = CP[child[q]];
                const double* Kf = &F.K[64 * size_t(q)];
                std::fill(T, T + 64, 0.0);
                for (int a = 0; a < 8; a++)
                    for (int bb = 0; bb < 8; bb++) {
                        const double kab = Kf[a * 8 + bb];
                        if (kab == 0) continue;
                        for (int cc = 0; cc < 8; cc++) T[a * 8 + cc] += kab * P[bb][cc];
                    }
                for (int a = 0; a < 8; a++)
                    for (int d = 0; d < 8; d++) {
                        const double w = P[a][d];
                        if (w == 0) continue;
                        for (int cc = 0; cc < 8; cc++) out[d * 8 + cc] += w * T[a * 8 + cc];
                    }
            }
            if (uniform) C.scale[c] = 2 * F.scale[kids[b]];
        }
    }, 64);
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
    const auto& K0 = conductionMatrix();
    const auto& S0 = conductionStencil();
    const int NX = L.NX, NY = L.NY, NZ = L.NZ, nx = L.nx, ny = L.ny, nz = L.nz;
    const int64_t NXY = int64_t(NX) * NY;
    // gather per node, one row of nodes along x per task: node n = i + NX * row, its 8 voxels
    // (di, dj, dk) = bits of v in 4 voxel rows (q = v >> 1) at columns i - 1 + di
    parallelFor(int64_t(NY) * NZ, [&](int64_t lo, int64_t hi) {
        for (int64_t row = lo; row < hi; row++) {
            const int j = int(row % NY), k = int(row / NY);
            int64_t vrow[4];
            bool has[4];
            for (int q = 0; q < 4; q++) {
                const int ej = j - 1 + (q & 1), ek = k - 1 + (q >> 1);
                has[q] = ej >= 0 && ek >= 0 && ej < ny && ek < nz;
                vrow[q] = has[q] ? int64_t(nx) * (ej + int64_t(ny) * ek) : 0;
            }
            for (int i = 0; i < NX; i++) {
                const int64_t n = i + NX * row;
                double acc = 0;
                if (L.active[n]) {
                    const float s0 = L.nodeScale[n];
                    if (s0 > 0) {
                        // 27-point stencil: nine rows of three neighbours
                        const double* x0 = x + (n - 1 - NX - NXY);
                        for (int r = 0; r < 9; r++) {
                            const double* xr = x0 + (r % 3) * NX + (r / 3) * NXY;
                            acc += S0[3 * r] * xr[0] + S0[3 * r + 1] * xr[1] + S0[3 * r + 2] * xr[2];
                        }
                        acc *= s0;
                    } else {
                        for (int v = 0; v < 8; v++) {
                            const int q = v >> 1, di = v & 1, ei = i - 1 + di;
                            if (!has[q] || ei < 0 || ei >= nx) continue;
                            const int e = L.emap[vrow[q] + ei];
                            if (e < 0) continue;
                            const int a = corner(1 - di, 1 - (q & 1), 1 - (q >> 1));
                            const int64_t nb = n - (1 - di) - NX * (1 - (q & 1)) - NXY * (1 - (q >> 1));
                            const double* Ke = L.K.empty() ? K0.data() : &L.K[64 * size_t(e)];
                            double sum = 0;
                            for (int b = 0; b < 8; b++) sum += Ke[a * 8 + b] * x[nb + L.off[b]];
                            acc += (L.K.empty() ? L.rho[e] : 1.0) * sum;
                        }
                    }
                    if (!L.diagAdd.empty()) acc += L.diagAdd[n] * x[n];
                }
                y[n] = !raw && L.fixed[n] ? 0 : acc;
            }
        }
    }, std::max<int64_t>(1, 2048 / NX));
}

// largest eigenvalue of D^-1 A: 10 Lanczos steps on D^-1/2 A D^-1/2 (as VoxelFEA::estimateOmega)
void ScalarVoxelSolver::estimateOmega(Level& L) {
    const int64_t n = L.nNodes;
    std::vector<double> sq(n), v(n), vPrev(n, 0.0), w(n), x(n);
    for (int64_t i = 0; i < n; i++) sq[i] = std::sqrt(L.invDiag[i]);
    uint32_t seed = 7;
    for (int64_t i = 0; i < n; i++) {
        seed = (seed * 1103515245u + 12345u) & 0x7fffffffu;
        v[i] = L.fixed[i] ? 0 : double(seed) / 0x7fffffff - 0.5;
    }
    const double nv = std::sqrt(dot(v.data(), v.data(), n));
    for (auto& e : v) e /= nv > 0 ? nv : 1;
    std::vector<double> alpha, beta;
    double b = 0;
    for (int j = 0; j < 10; j++) {
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) x[i] = sq[i] * v[i]; });
        apply(L, x.data(), w.data());
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) w[i] = sq[i] * w[i] - b * vPrev[i]; });
        const double a = dot(w.data(), v.data(), n);
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) w[i] -= a * v[i]; });
        alpha.push_back(a);
        b = std::sqrt(dot(w.data(), w.data(), n));
        if (!(b > 1e-12 * std::abs(a))) break;
        beta.push_back(b);
        std::swap(vPrev, v);
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) v[i] = w[i] / b; });
    }
    beta.resize(alpha.size() - 1);
    L.omega = 1.2 / (tridiagonalMax(alpha, beta) * 1.05);
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
    choleskySolve(c.A.data(), c.y.data(), m);
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
    // gather: coarse node I takes fine 2I (weight 1) and 2I +- 1 (weight 1/2) along each axis
    parallelFor(int64_t(C.NY) * C.NZ, [&](int64_t lo, int64_t hi) {
        for (int64_t row = lo; row < hi; row++) {
            const int J = int(row % C.NY), K = int(row / C.NY);
            for (int I = 0; I < C.NX; I++) {
                double s = 0;
                for (int dk = -1; dk <= 1; dk++) {
                    const int kf = 2 * K + dk;
                    if (kf < 0 || kf >= F.NZ) continue;
                    const double wk = dk == 0 ? 1 : 0.5;
                    for (int dj = -1; dj <= 1; dj++) {
                        const int jf = 2 * J + dj;
                        if (jf < 0 || jf >= F.NY) continue;
                        const double wjk = (dj == 0 ? 1 : 0.5) * wk;
                        const double* p = rf + int64_t(F.NX) * (jf + int64_t(F.NY) * kf);
                        const int i0 = 2 * I;  // past the fine grid's end when its element count is odd
                        double t = i0 < F.NX ? p[i0] : 0;
                        if (i0 > 0) t += 0.5 * p[i0 - 1];
                        if (i0 + 1 < F.NX) t += 0.5 * p[i0 + 1];
                        s += wjk * t;
                    }
                }
                const int64_t cn = I + int64_t(C.NX) * row;
                rc[cn] = zeroFixed && C.fixed[cn] ? 0 : s;
            }
        }
    }, std::max<int64_t>(1, 1024 / C.NX));
}

void ScalarVoxelSolver::prolongAdd(const Level& F, const Level& C, const double* zc, double* zf) const {
    parallelFor(int64_t(F.NY) * F.NZ, [&](int64_t lo, int64_t hi) {
        for (int64_t row = lo; row < hi; row++) {
            const int j = int(row % F.NY), k = int(row / F.NY);
            const int j0 = F.my0[j], j1 = F.my1[j], k0 = F.mz0[k], k1 = F.mz1[k];
            const double wj = j1 < 0 ? 1 : 0.5, wk = k1 < 0 ? 1 : 0.5;
            for (int i = 0; i < F.NX; i++) {
                const int64_t fn = i + int64_t(F.NX) * row;
                if (F.fixed[fn]) continue;
                const int i0 = F.mx0[i], i1 = F.mx1[i];
                const double wi = i1 < 0 ? 1 : 0.5;
                double s = 0;
                for (int kk = 0; kk < 2; kk++) {
                    const int K = kk ? k1 : k0;
                    if (K < 0) continue;
                    for (int jj = 0; jj < 2; jj++) {
                        const int J = jj ? j1 : j0;
                        if (J < 0) continue;
                        const double* p = zc + int64_t(C.NX) * (J + int64_t(C.NY) * K);
                        s += wj * wk * (i1 < 0 ? p[i0] : wi * (p[i0] + p[i1]));
                    }
                }
                zf[fn] += s;
            }
        }
    }, std::max<int64_t>(1, 1024 / F.NX));
}

std::string ScalarVoxelSolver::profile(int reps) {
    std::string out;
    for (size_t l = 0; l < levels.size(); l++) {
        Level& L = levels[l];
        std::vector<double> x(L.nNodes, 1.0), y(L.nNodes);
        apply(L, x.data(), y.data());
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; k++) apply(L, x.data(), y.data());
        const double ta = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; k++) vcycle(l);
        const double tv = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        char line[160];
        std::snprintf(line, sizeof line, "heat L%zu nodes=%lld: matvec %.3f ms, V-cycle from here %.3f ms\n", l, (long long)L.nNodes, ta, tv);
        out += line;
    }
    return out;
}

void ScalarVoxelSolver::vcycle(size_t l) {
    Level& L = levels[l];
    if (l == levels.size() - 1) { coarseSolve(L); return; }
    Level& C = levels[l + 1];
    // two damped Jacobi sweeps each side (on these scalar problems they beat a degree-2 Chebyshev
    // smoother, which the structural solver uses)
    jacobi(L, true);
    jacobi(L, false);
    apply(L, L.z.data(), L.t.data());
    parallelFor(L.nNodes, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.t[i] = L.r[i] - L.t[i]; });
    restrict(L, C, L.t.data(), C.r.data());
    vcycle(l + 1);
    prolongAdd(L, C, C.z.data(), L.z.data());
    jacobi(L, false);
    jacobi(L, false);
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
        parallelFor(n, [&](int64_t lo, int64_t hi) { std::copy(r.begin() + lo, r.begin() + hi, L.r.begin() + lo); });
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
        TraceTimer trace("heat multigrid build");
        out.solver = std::make_unique<ScalarVoxelSolver>(in.dims, in.density, in.fixedNode, &conv);
        trace.lap("heat solve");
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
    TraceTimer trace("heat multigrid build");
    out.solver = std::make_unique<ScalarVoxelSolver>(in.dims, in.density, in.fixedNode, &diag);
    trace.lap("heat time steps");
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
