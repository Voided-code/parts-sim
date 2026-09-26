#include "voxel_fea.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../util/parallel.hpp"

namespace ps {

namespace {

constexpr int SMOOTH_SWEEPS = 2;

// corner index of the element node at unit offset (x, y, z)
inline int corner(int x, int y, int z) { return (y ? (x ? 2 : 3) : (x ? 1 : 0)) + 4 * z; }

// CHILD_P[c][a]: fine child-node a of child c interpolates coarse node b with weight w
struct ChildP {
    int count[8][8];
    int node[8][8][8];
    double w[8][8][8];
};

const ChildP& childP() {
    static const ChildP P = [] {
        ChildP p{};
        for (int c = 0; c < 8; c++) {
            const int ci = c & 1, cj = (c >> 1) & 1, ck = (c >> 2) & 1;
            for (int a = 0; a < 8; a++) {
                const double sx = (ci + HEX_NODES[a][0]) / 2.0, sy = (cj + HEX_NODES[a][1]) / 2.0, sz = (ck + HEX_NODES[a][2]) / 2.0;
                int n = 0;
                for (int b = 0; b < 8; b++) {
                    const double w = (HEX_NODES[b][0] ? sx : 1 - sx) * (HEX_NODES[b][1] ? sy : 1 - sy) * (HEX_NODES[b][2] ? sz : 1 - sz);
                    if (w > 1e-12) {
                        p.node[c][a][n] = b;
                        p.w[c][a][n] = w;
                        n++;
                    }
                }
                p.count[c][a] = n;
            }
        }
        return p;
    }();
    return P;
}

// out += scale * P_c^T K P_c
void galerkinAdd(double* out, const double* K, double scale, int c, double* T) {
    const ChildP& P = childP();
    std::fill(T, T + 576, 0.0);
    for (int a = 0; a < 8; a++)
        for (int t = 0; t < P.count[c][a]; t++) {
            const int b3 = 3 * P.node[c][a][t], a3 = 3 * a;
            const double w = P.w[c][a][t];
            for (int r = 0; r < 24; r++) {
                const double* kr = K + r * 24 + a3;
                double* tr = T + r * 24 + b3;
                tr[0] += kr[0] * w;
                tr[1] += kr[1] * w;
                tr[2] += kr[2] * w;
            }
        }
    for (int a = 0; a < 8; a++)
        for (int t = 0; t < P.count[c][a]; t++) {
            const int b3 = 3 * P.node[c][a][t], a3 = 3 * a;
            const double ws = P.w[c][a][t] * scale;
            for (int d = 0; d < 3; d++) {
                double* o = out + (b3 + d) * 24;
                const double* tr = T + (a3 + d) * 24;
                for (int q = 0; q < 24; q++) o[q] += ws * tr[q];
            }
        }
}

Level makeLevel(int nx, int ny, int nz) {
    Level L;
    L.nx = nx; L.ny = ny; L.nz = nz;
    L.NX = nx + 1; L.NY = ny + 1; L.NZ = nz + 1;
    const int64_t NXY = int64_t(L.NX) * L.NY;
    L.nNodes = NXY * L.NZ;
    L.nDof = 3 * L.nNodes;
    L.off = {0, 1, 1 + L.NX, L.NX, NXY, 1 + NXY, 1 + L.NX + NXY, L.NX + NXY};
    return L;
}

TransferMap transferMap(int nFineElems) {
    TransferMap m;
    const int n = nFineElems + 1;
    m.c0.resize(n);
    m.c1.resize(n);
    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) { m.c0[i] = i / 2; m.c1[i] = -1; }
        else { m.c0[i] = (i - 1) / 2; m.c1[i] = (i + 1) / 2; }
    }
    return m;
}

}  // namespace

VoxelFEA::VoxelFEA(std::array<int, 3> dims, const std::vector<float>& density, double nu_, const std::vector<uint8_t>& bc,
                   int64_t coarsestMaxDof, const std::vector<double>* diagAdd)
    : nu(nu_) {
    if (dims[0] < 1 || dims[1] < 1 || dims[2] < 1) throw std::invalid_argument("The structural grid must have three positive dimensions.");
    const int64_t nE = int64_t(dims[0]) * dims[1] * dims[2];
    if (int64_t(density.size()) != nE || int64_t(bc.size()) != 3 * int64_t(dims[0] + 1) * (dims[1] + 1) * (dims[2] + 1))
        throw std::invalid_argument("Structural grid, density and fixture sizes do not match.");
    for (float r : density)
        if (!std::isfinite(r) || r < 0 || r > 1) throw std::invalid_argument("Voxel density must be between zero and one.");
    const HexElement el = hexElement(nu);
    K0 = el.K;
    cornerStress = el.cornerStress;
    gaussB = el.gaussB;
    std::vector<double> T(576);
    for (int c = 0; c < 8; c++) {
        M0_[c].fill(0.0);
        galerkinAdd(M0_[c].data(), K0.data(), 1.0, c, T.data());
    }

    Level L0 = makeLevel(dims[0], dims[1], dims[2]);
    L0.emap.assign(nE, -1);
    for (int k = 0; k < L0.nz; k++)
        for (int j = 0; j < L0.ny; j++)
            for (int i = 0; i < L0.nx; i++) {
                const int e = i + L0.nx * (j + L0.ny * k);
                if (density[e] > 0) {
                    L0.emap[e] = int(L0.elems.size());
                    L0.elems.push_back(e);
                    L0.base.push_back(i + int64_t(L0.NX) * (j + int64_t(L0.NY) * k));
                    L0.rho.push_back(density[e]);
                }
            }
    L0.bc = bc;
    if (diagAdd) {
        if (int64_t(diagAdd->size()) != L0.nDof) throw std::invalid_argument("Diagonal shift size does not match.");
        L0.diagAdd = *diagAdd;
    }
    finishLevel(L0);
    levels.push_back(std::move(L0));
    for (;;) {
        Level& L = levels.back();
        if (L.freeDof <= coarsestMaxDof || levels.size() >= 12) break;
        if (L.nx <= 1 && L.ny <= 1 && L.nz <= 1) break;
        Level C = coarsen(L);
        levels.push_back(std::move(C));
    }
    for (int l = 0; l + 1 < int(levels.size()); l++) estimateOmega(l);
    buildCoarseSolver(levels.back());
}

void VoxelFEA::finishLevel(Level& L) {
    L.activeNode.assign(L.nNodes, 0);
    std::vector<double> diag(L.nDof, 0.0);
    const int64_t nE = L.elems.size();
    for (int64_t e = 0; e < nE; e++) {
        const int64_t n0 = L.base[e];
        const double* K = L.shared() ? K0.data() : L.K.data() + e * 576;
        const double s = L.shared() ? L.rho[e] : 1.0;
        for (int a = 0; a < 8; a++) {
            const int64_t n = n0 + L.off[a];
            L.activeNode[n] = 1;
            for (int d = 0; d < 3; d++) diag[3 * n + d] += s * K[(3 * a + d) * 25];
        }
    }
    if (!L.diagAdd.empty())
        for (int64_t i = 0; i < L.nDof; i++) diag[i] += L.diagAdd[i];
    L.fixed.assign(L.nDof, 0);
    L.invDiag.assign(L.nDof, 0.0);
    int64_t free = 0;
    for (int64_t n = 0; n < L.nNodes; n++)
        for (int d = 0; d < 3; d++) {
            const int64_t i = 3 * n + d;
            if (!L.activeNode[n] || L.bc[i] || !(diag[i] > 0)) L.fixed[i] = 1;
            else { L.invDiag[i] = 1 / diag[i]; free++; }
        }
    L.freeDof = free;
    L.r.assign(L.nDof, 0.0);
    L.z.assign(L.nDof, 0.0);
    L.t.assign(L.nDof, 0.0);
}

Level VoxelFEA::coarsen(Level& F) {
    Level C = makeLevel((F.nx + 1) >> 1, (F.ny + 1) >> 1, (F.nz + 1) >> 1);
    const int64_t nCv = int64_t(C.nx) * C.ny * C.nz;
    C.emap.assign(nCv, -1);
    const int64_t nF = F.elems.size();
    std::vector<int> fineCoarse(nF);
    std::vector<uint8_t> fineChild(nF);
    for (int64_t q = 0; q < nF; q++) {
        const int e = F.elems[q];
        const int i = e % F.nx, j = (e / F.nx) % F.ny, k = e / (F.nx * F.ny);
        const int ce = (i >> 1) + C.nx * ((j >> 1) + C.ny * (k >> 1));
        if (C.emap[ce] < 0) { C.emap[ce] = int(C.elems.size()); C.elems.push_back(ce); }
        fineCoarse[q] = C.emap[ce];
        fineChild[q] = uint8_t((i & 1) | ((j & 1) << 1) | ((k & 1) << 2));
    }
    // the elements list is in first-seen order; re-sort by voxel index for memory locality
    {
        std::vector<int> order(C.elems);
        std::sort(order.begin(), order.end());
        std::vector<int> remap(C.elems.size());
        for (size_t q = 0; q < order.size(); q++) C.emap[order[q]] = int(q);
        for (size_t q = 0; q < C.elems.size(); q++) remap[q] = C.emap[C.elems[q]];
        for (auto& v : fineCoarse) v = remap[v];
        C.elems = std::move(order);
    }
    const int64_t nE = C.elems.size();
    C.base.resize(nE);
    for (int64_t q = 0; q < nE; q++) {
        const int ce = C.elems[q];
        const int I = ce % C.nx, J = (ce / C.nx) % C.ny, K = ce / (C.nx * C.ny);
        C.base[q] = I + int64_t(C.NX) * (J + int64_t(C.NY) * K);
    }
    // children lists, then Galerkin products per coarse element (parallel, no conflicts)
    C.childStart.assign(nE + 1, 0);
    for (int64_t q = 0; q < nF; q++) C.childStart[fineCoarse[q] + 1]++;
    for (int64_t q = 0; q < nE; q++) C.childStart[q + 1] += C.childStart[q];
    C.children.resize(nF);
    {
        std::vector<int> fill(C.childStart.begin(), C.childStart.end() - 1);
        for (int64_t q = 0; q < nF; q++) C.children[fill[fineCoarse[q]]++] = int(q);
    }
    C.K.assign(nE * 576, 0.0);
    parallelFor(nE, [&](int64_t lo, int64_t hi) {
        std::vector<double> T(576);
        for (int64_t q = lo; q < hi; q++) {
            double* out = C.K.data() + q * 576;
            for (int t = C.childStart[q]; t < C.childStart[q + 1]; t++) {
                const int f = C.children[t];
                const int c = fineChild[f];
                if (F.shared()) {
                    const double s = F.rho[f];
                    const double* M = M0_[c].data();
                    for (int u = 0; u < 576; u++) out[u] += s * M[u];
                } else {
                    galerkinAdd(out, F.K.data() + int64_t(f) * 576, 1.0, c, T.data());
                }
            }
        }
    }, 64);
    // a coarse DOF is held if it interpolates onto any held fine DOF
    F.mx = transferMap(F.nx);
    F.my = transferMap(F.ny);
    F.mz = transferMap(F.nz);
    C.bc.assign(C.nDof, 0);
    for (int k = 0; k < F.NZ; k++)
        for (int j = 0; j < F.NY; j++)
            for (int i = 0; i < F.NX; i++) {
                const int64_t fn = 3 * (i + int64_t(F.NX) * (j + int64_t(F.NY) * k));
                if (!(F.bc[fn] | F.bc[fn + 1] | F.bc[fn + 2])) continue;
                for (int K : {F.mz.c0[k], F.mz.c1[k]}) {
                    if (K < 0) continue;
                    for (int J : {F.my.c0[j], F.my.c1[j]}) {
                        if (J < 0) continue;
                        for (int I : {F.mx.c0[i], F.mx.c1[i]}) {
                            if (I < 0) continue;
                            const int64_t cn = 3 * (I + int64_t(C.NX) * (J + int64_t(C.NY) * K));
                            C.bc[cn] |= F.bc[fn];
                            C.bc[cn + 1] |= F.bc[fn + 1];
                            C.bc[cn + 2] |= F.bc[fn + 2];
                        }
                    }
                }
            }
    if (!F.diagAdd.empty()) {
        // lumped Galerkin product of a diagonal: row sums of P^T D P = P^T (D 1)
        C.diagAdd.assign(C.nDof, 0.0);
        restrict(F, C, F.diagAdd.data(), C.diagAdd.data(), false);
    }
    finishLevel(C);
    return C;
}

void VoxelFEA::apply(int l, const double* x, double* y, bool raw) const {
    const Level& L = levels[l];
    const int NX = L.NX, NY = L.NY, nx = L.nx, ny = L.ny, nz = L.nz;
    const bool shared = L.shared();
    const double* K0p = K0.data();
    const int64_t NXY = int64_t(NX) * NY;
    parallelFor(L.nNodes, [&](int64_t lo, int64_t hi) {
        double ue[24];
        for (int64_t n = lo; n < hi; n++) {
            double acc0 = 0, acc1 = 0, acc2 = 0;
            if (L.activeNode[n]) {
                const int i = int(n % NX), j = int((n / NX) % NY), k = int(n / NXY);
                for (int dk = 0; dk < 2; dk++) {
                    const int ek = k + dk - 1;
                    if (ek < 0 || ek >= nz) continue;
                    for (int dj = 0; dj < 2; dj++) {
                        const int ej = j + dj - 1;
                        if (ej < 0 || ej >= ny) continue;
                        for (int di = 0; di < 2; di++) {
                            const int ei = i + di - 1;
                            if (ei < 0 || ei >= nx) continue;
                            const int e = L.emap[ei + nx * (ej + ny * ek)];
                            if (e < 0) continue;
                            const int a = corner(1 - di, 1 - dj, 1 - dk);
                            const int64_t nb = ei + NX * (ej + int64_t(NY) * ek);
                            for (int c = 0; c < 8; c++) {
                                const double* xp = x + 3 * (nb + L.off[c]);
                                ue[3 * c] = xp[0];
                                ue[3 * c + 1] = xp[1];
                                ue[3 * c + 2] = xp[2];
                            }
                            const double* Ke = shared ? K0p : L.K.data() + int64_t(e) * 576;
                            const double s = shared ? L.rho[e] : 1.0;
                            const double* r0 = Ke + (3 * a) * 24;
                            double s0 = 0, s1 = 0, s2 = 0;
                            for (int c = 0; c < 24; c++) {
                                s0 += r0[c] * ue[c];
                                s1 += r0[24 + c] * ue[c];
                                s2 += r0[48 + c] * ue[c];
                            }
                            acc0 += s * s0;
                            acc1 += s * s1;
                            acc2 += s * s2;
                        }
                    }
                }
                if (!L.diagAdd.empty()) {
                    acc0 += L.diagAdd[3 * n] * x[3 * n];
                    acc1 += L.diagAdd[3 * n + 1] * x[3 * n + 1];
                    acc2 += L.diagAdd[3 * n + 2] * x[3 * n + 2];
                }
            }
            double* yp = y + 3 * n;
            if (raw) { yp[0] = acc0; yp[1] = acc1; yp[2] = acc2; }
            else {
                const uint8_t* fx = L.fixed.data() + 3 * n;
                yp[0] = fx[0] ? 0 : acc0;
                yp[1] = fx[1] ? 0 : acc1;
                yp[2] = fx[2] ? 0 : acc2;
            }
        }
    }, 512);
}

void VoxelFEA::estimateOmega(int l) {
    Level& L = levels[l];
    std::vector<double> v(L.nDof), w(L.nDof);
    uint32_t seed = 12345;
    for (int64_t i = 0; i < L.nDof; i++) {
        seed = (seed * 1103515245u + 12345u) & 0x7fffffffu;
        v[i] = L.fixed[i] ? 0 : double(seed) / 0x7fffffff - 0.5;
    }
    double lambda = 1;
    for (int it = 0; it < 12; it++) {
        const double nv = std::sqrt(dot(v.data(), v.data(), L.nDof));
        const double inv = nv > 0 ? 1 / nv : 1;
        parallelFor(L.nDof, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) v[i] *= inv; });
        apply(l, v.data(), w.data());
        parallelFor(L.nDof, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) w[i] *= L.invDiag[i]; });
        lambda = std::sqrt(dot(w.data(), w.data(), L.nDof));
        std::swap(v, w);
    }
    L.omega = 1.2 / (lambda * 1.05);
}

void VoxelFEA::buildCoarseSolver(Level& L) {
    coarse.map.assign(L.nDof, -1);
    int64_t m = 0;
    for (int64_t i = 0; i < L.nDof; i++)
        if (!L.fixed[i]) coarse.map[i] = m++;
    coarse.m = m;
    coarse.A.clear();
    if (m == 0 || m > 3000) return;  // falls back to Jacobi sweeps
    std::vector<double>& A = coarse.A;
    A.assign(m * m, 0.0);
    int64_t dofs[24];
    for (size_t e = 0; e < L.elems.size(); e++) {
        const int64_t n0 = L.base[e];
        for (int a = 0; a < 8; a++)
            for (int d = 0; d < 3; d++) dofs[3 * a + d] = coarse.map[3 * (n0 + L.off[a]) + d];
        const double* K = L.shared() ? K0.data() : L.K.data() + e * 576;
        const double s = L.shared() ? L.rho[e] : 1.0;
        for (int r = 0; r < 24; r++) {
            if (dofs[r] < 0) continue;
            for (int c = 0; c < 24; c++)
                if (dofs[c] >= 0) A[dofs[r] * m + dofs[c]] += s * K[r * 24 + c];
        }
    }
    if (!L.diagAdd.empty())
        for (int64_t i = 0; i < L.nDof; i++)
            if (coarse.map[i] >= 0) A[coarse.map[i] * m + coarse.map[i]] += L.diagAdd[i];
    double maxDiag = 0;
    for (int64_t i = 0; i < m; i++) maxDiag = std::max(maxDiag, A[i * m + i]);
    // right-looking Cholesky (lower triangle), rows below the pivot updated in parallel
    for (int64_t j = 0; j < m; j++) {
        double s = A[j * m + j];
        if (!(s > 1e-10 * maxDiag)) s = maxDiag;  // regularize near-mechanisms at the coarsest level
        const double ljj = std::sqrt(s);
        A[j * m + j] = ljj;
        const double inv = 1 / ljj;
        for (int64_t i = j + 1; i < m; i++) A[i * m + j] *= inv;
        parallelFor(m - j - 1, [&](int64_t lo, int64_t hi) {
            for (int64_t ii = lo; ii < hi; ii++) {
                const int64_t i = j + 1 + ii;
                const double lij = A[i * m + j];
                if (lij == 0) continue;
                double* row = A.data() + i * m;
                const double* rj = A.data();
                for (int64_t k = j + 1; k <= i; k++) row[k] -= lij * rj[k * m + j];
            }
        }, 32);
    }
    coarse.y.assign(m, 0.0);
}

void VoxelFEA::coarseSolve(Level& L) {
    std::fill(L.z.begin(), L.z.end(), 0.0);
    if (coarse.A.empty()) {
        for (int s = 0; s < 40; s++) jacobi(int(levels.size()) - 1, s == 0);
        return;
    }
    const int64_t m = coarse.m;
    const double* A = coarse.A.data();
    double* y = coarse.y.data();
    for (int64_t i = 0; i < L.nDof; i++)
        if (coarse.map[i] >= 0) y[coarse.map[i]] = L.r[i];
    for (int64_t i = 0; i < m; i++) {
        double s = y[i];
        const double* row = A + i * m;
        for (int64_t k = 0; k < i; k++) s -= row[k] * y[k];
        y[i] = s / row[i];
    }
    for (int64_t i = m - 1; i >= 0; i--) {
        double s = y[i];
        for (int64_t k = i + 1; k < m; k++) s -= A[k * m + i] * y[k];
        y[i] = s / A[i * m + i];
    }
    for (int64_t i = 0; i < L.nDof; i++)
        if (coarse.map[i] >= 0) L.z[i] = y[coarse.map[i]];
}

void VoxelFEA::jacobi(int l, bool first) {
    Level& L = levels[l];
    const double om = L.omega;
    if (first) {
        parallelFor(L.nDof, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.z[i] = om * L.invDiag[i] * L.r[i]; });
        return;
    }
    apply(l, L.z.data(), L.t.data());
    parallelFor(L.nDof, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.z[i] += om * L.invDiag[i] * (L.r[i] - L.t[i]); });
}

// rc = P^T rf, gathered per coarse node: coarse I takes fine 2I (weight 1) and 2I +- 1 (weight 1/2)
void VoxelFEA::restrict(const Level& F, const Level& C, const double* rf, double* rc, bool zeroFixed) const {
    const int64_t CNXY = int64_t(C.NX) * C.NY;
    parallelFor(C.nNodes, [&](int64_t lo, int64_t hi) {
        for (int64_t cn = lo; cn < hi; cn++) {
            const int I = int(cn % C.NX), J = int((cn / C.NX) % C.NY), K = int(cn / CNXY);
            double s0 = 0, s1 = 0, s2 = 0;
            for (int dk = -1; dk <= 1; dk++) {
                const int kf = 2 * K + dk;
                if (kf < 0 || kf >= F.NZ) continue;
                const double wk = dk == 0 ? 1 : 0.5;
                for (int dj = -1; dj <= 1; dj++) {
                    const int jf = 2 * J + dj;
                    if (jf < 0 || jf >= F.NY) continue;
                    const double wj = dj == 0 ? 1 : 0.5;
                    for (int di = -1; di <= 1; di++) {
                        const int if_ = 2 * I + di;
                        if (if_ < 0 || if_ >= F.NX) continue;
                        const double w = (di == 0 ? 1 : 0.5) * wj * wk;
                        const double* p = rf + 3 * (if_ + int64_t(F.NX) * (jf + int64_t(F.NY) * kf));
                        s0 += w * p[0];
                        s1 += w * p[1];
                        s2 += w * p[2];
                    }
                }
            }
            double* o = rc + 3 * cn;
            if (zeroFixed) {
                const uint8_t* fx = C.fixed.data() + 3 * cn;
                o[0] = fx[0] ? 0 : s0;
                o[1] = fx[1] ? 0 : s1;
                o[2] = fx[2] ? 0 : s2;
            } else {
                o[0] = s0; o[1] = s1; o[2] = s2;
            }
        }
    }, 512);
}

void VoxelFEA::prolongAdd(const Level& F, const Level& C, const double* zc, double* zf) const {
    const int64_t FNXY = int64_t(F.NX) * F.NY;
    parallelFor(F.nNodes, [&](int64_t lo, int64_t hi) {
        for (int64_t fn = lo; fn < hi; fn++) {
            const uint8_t* fx = F.fixed.data() + 3 * fn;
            if (fx[0] && fx[1] && fx[2]) continue;
            const int i = int(fn % F.NX), j = int((fn / F.NX) % F.NY), k = int(fn / FNXY);
            const int i0 = F.mx.c0[i], i1 = F.mx.c1[i], j0 = F.my.c0[j], j1 = F.my.c1[j], k0 = F.mz.c0[k], k1 = F.mz.c1[k];
            const double wi = i1 < 0 ? 1 : 0.5, wj = j1 < 0 ? 1 : 0.5, wk = k1 < 0 ? 1 : 0.5;
            double s0 = 0, s1 = 0, s2 = 0;
            for (int kk = 0; kk < 2; kk++) {
                const int K = kk ? k1 : k0;
                if (K < 0) continue;
                for (int jj = 0; jj < 2; jj++) {
                    const int J = jj ? j1 : j0;
                    if (J < 0) continue;
                    for (int ii = 0; ii < 2; ii++) {
                        const int I = ii ? i1 : i0;
                        if (I < 0) continue;
                        const double w = wi * wj * wk;
                        const double* p = zc + 3 * (I + int64_t(C.NX) * (J + int64_t(C.NY) * K));
                        s0 += w * p[0];
                        s1 += w * p[1];
                        s2 += w * p[2];
                    }
                }
            }
            double* o = zf + 3 * fn;
            if (!fx[0]) o[0] += s0;
            if (!fx[1]) o[1] += s1;
            if (!fx[2]) o[2] += s2;
        }
    }, 512);
}

void VoxelFEA::vcycle(int l) {
    Level& L = levels[l];
    if (l == int(levels.size()) - 1) { coarseSolve(L); return; }
    Level& C = levels[l + 1];
    jacobi(l, true);
    for (int s = 1; s < SMOOTH_SWEEPS; s++) jacobi(l, false);
    apply(l, L.z.data(), L.t.data());
    parallelFor(L.nDof, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.t[i] = L.r[i] - L.t[i]; });
    restrict(L, C, L.t.data(), C.r.data());
    vcycle(l + 1);
    prolongAdd(L, C, C.z.data(), L.z.data());
    for (int s = 0; s < SMOOTH_SWEEPS; s++) jacobi(l, false);
}

void VoxelFEA::precondition(const double* r, double* z) {
    Level& L = levels[0];
    parallelFor(L.nDof, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.r[i] = L.fixed[i] ? 0 : r[i]; });
    vcycle(0);
    std::copy(L.z.begin(), L.z.end(), z);
}

SolveResult VoxelFEA::solve(const std::vector<double>& f, const SolveOptions& opts) {
    Level& L = levels[0];
    const int64_t n = L.nDof;
    if (int64_t(f.size()) != n || (opts.x0 && int64_t(opts.x0->size()) != n)) throw std::invalid_argument("Force or displacement vector has the wrong size.");
    for (int64_t i = 0; i < n; i++)
        if (!std::isfinite(f[i]) || (opts.x0 && !std::isfinite((*opts.x0)[i]))) throw std::invalid_argument("Forces and displacements must be finite numbers.");
    SolveResult res;
    res.u = opts.x0 ? *opts.x0 : std::vector<double>(n, 0.0);
    auto& x = res.u;
    const auto& fixed = L.fixed;
    for (int64_t i = 0; i < n; i++)
        if (fixed[i]) x[i] = 0;
    std::vector<double> r(n), p(n), q(n);
    double bnorm = 0;
    for (int64_t i = 0; i < n; i++)
        if (!fixed[i]) bnorm += f[i] * f[i];
    bnorm = std::sqrt(bnorm);
    if (bnorm == 0) {
        std::fill(x.begin(), x.end(), 0.0);
        res.converged = true;
        return res;
    }
    apply(0, x.data(), q.data());
    parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) r[i] = fixed[i] ? 0 : f[i] - q[i]; });
    auto precond = [&]() {
        L.r = r;
        vcycle(0);
        return L.z.data();
    };
    const double* z = precond();
    std::copy(z, z + n, p.begin());
    double rz = dot(r.data(), z, n);
    double rel = std::sqrt(dot(r.data(), r.data(), n)) / bnorm;
    int it = 0;
    for (; it < opts.maxIter && rel > opts.tol; it++) {
        apply(0, p.data(), q.data());
        const double pq = dot(p.data(), q.data(), n);
        if (!(pq > 0)) break;  // loss of positive-definiteness (mechanism)
        const double alpha = rz / pq;
        parallelFor(n, [&](int64_t lo, int64_t hi) {
            for (int64_t i = lo; i < hi; i++) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        });
        rel = std::sqrt(dot(r.data(), r.data(), n)) / bnorm;
        if (opts.onProgress && opts.onProgress(it + 1, rel)) { res.cancelled = true; it++; break; }
        if (rel <= opts.tol) { it++; break; }
        z = precond();
        const double rzNew = dot(r.data(), z, n);
        const double beta = rzNew / rz;
        rz = rzNew;
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) p[i] = z[i] + beta * p[i]; });
    }
    // report convergence against the actual equilibrium equations
    apply(0, x.data(), q.data());
    parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) r[i] = fixed[i] ? 0 : f[i] - q[i]; });
    res.residual = std::sqrt(dot(r.data(), r.data(), n)) / bnorm;
    res.iterations = it;
    res.converged = !res.cancelled && std::isfinite(res.residual) && res.residual <= opts.tol * 10;
    return res;
}

std::array<double, 3> VoxelFEA::reactions(const std::vector<double>& u, const std::vector<double>* f) const {
    const Level& L = levels[0];
    std::vector<double> y(L.nDof);
    apply(0, u.data(), y.data(), true);
    std::array<double, 3> R{0, 0, 0};
    for (int64_t n = 0; n < L.nNodes; n++)
        for (int d = 0; d < 3; d++) {
            const int64_t i = 3 * n + d;
            if (L.bc[i] && L.activeNode[n]) R[d] += y[i] - (f ? (*f)[i] : 0.0);
        }
    return R;
}

double vonMises(const double* s) {
    return std::sqrt(0.5 * ((s[0] - s[1]) * (s[0] - s[1]) + (s[1] - s[2]) * (s[1] - s[2]) + (s[2] - s[0]) * (s[2] - s[0])) +
                     3 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
}

void principalStresses(double sx, double sy, double sz, double txy, double tyz, double tzx, double out[3]) {
    const double p1 = txy * txy + tyz * tyz + tzx * tzx;
    const double q = (sx + sy + sz) / 3;
    const double scale = std::abs(sx) + std::abs(sy) + std::abs(sz) + std::sqrt(p1);
    if (p1 <= 1e-24 * scale * scale) {
        double v[3] = {sx, sy, sz};
        std::sort(v, v + 3, [](double a, double b) { return a > b; });
        out[0] = v[0]; out[1] = v[1]; out[2] = v[2];
        return;
    }
    const double p2 = (sx - q) * (sx - q) + (sy - q) * (sy - q) + (sz - q) * (sz - q) + 2 * p1;
    const double p = std::sqrt(p2 / 6);
    const double b11 = (sx - q) / p, b22 = (sy - q) / p, b33 = (sz - q) / p, b12 = txy / p, b23 = tyz / p, b13 = tzx / p;
    const double det = b11 * (b22 * b33 - b23 * b23) - b12 * (b12 * b33 - b23 * b13) + b13 * (b12 * b23 - b22 * b13);
    const double r = std::min(1.0, std::max(-1.0, det / 2));
    const double phi = std::acos(r) / 3;
    out[0] = q + 2 * p * std::cos(phi);
    out[2] = q + 2 * p * std::cos(phi + 2 * M_PI / 3);
    out[1] = 3 * q - out[0] - out[2];
}

NodalStresses VoxelFEA::stresses(const std::vector<double>& u, double E, double h) const {
    const Level& L = levels[0];
    std::array<std::array<double, 144>, 8> S;
    for (int a = 0; a < 8; a++)
        for (int i = 0; i < 144; i++) S[a][i] = cornerStress[a][i] * E / h;
    NodalStresses out;
    const int64_t nE = L.elems.size();
    out.elemVM.assign(nE, 0.f);
    out.nodeVM.assign(L.nNodes, 0.f);
    out.nodeP1.assign(L.nNodes, 0.f);
    out.nodeP3.assign(L.nNodes, 0.f);
    // element centroid von Mises
    parallelFor(nE, [&](int64_t lo, int64_t hi) {
        double ue[24], c[6];
        for (int64_t e = lo; e < hi; e++) {
            for (int a = 0; a < 8; a++)
                for (int d = 0; d < 3; d++) ue[3 * a + d] = u[3 * (L.base[e] + L.off[a]) + d];
            for (int i = 0; i < 6; i++) {
                double s = 0;
                for (int a = 0; a < 8; a++)
                    for (int q = 0; q < 24; q++) s += S[a][i * 24 + q] * ue[q];
                c[i] = s / 8;
            }
            out.elemVM[e] = float(vonMises(c));
        }
    }, 256);
    // nodal values: density-weighted average of the surrounding voxels' corner stresses
    const int NX = L.NX, NY = L.NY, nx = L.nx, ny = L.ny, nz = L.nz;
    const int64_t NXY = int64_t(NX) * NY;
    parallelFor(L.nNodes, [&](int64_t lo, int64_t hi) {
        double ue[24], sig[6], pr[3];
        for (int64_t n = lo; n < hi; n++) {
            if (!L.activeNode[n]) continue;
            const int i = int(n % NX), j = int((n / NX) % NY), k = int(n / NXY);
            double vm = 0, p1 = 0, p3 = 0, w = 0;
            for (int dk = 0; dk < 2; dk++) {
                const int ek = k + dk - 1;
                if (ek < 0 || ek >= nz) continue;
                for (int dj = 0; dj < 2; dj++) {
                    const int ej = j + dj - 1;
                    if (ej < 0 || ej >= ny) continue;
                    for (int di = 0; di < 2; di++) {
                        const int ei = i + di - 1;
                        if (ei < 0 || ei >= nx) continue;
                        const int e = L.emap[ei + nx * (ej + ny * ek)];
                        if (e < 0) continue;
                        const int a = corner(1 - di, 1 - dj, 1 - dk);
                        const int64_t nb = ei + NX * (ej + int64_t(NY) * ek);
                        for (int c = 0; c < 8; c++)
                            for (int d = 0; d < 3; d++) ue[3 * c + d] = u[3 * (nb + L.off[c]) + d];
                        for (int r = 0; r < 6; r++) {
                            double s = 0;
                            for (int q = 0; q < 24; q++) s += S[a][r * 24 + q] * ue[q];
                            sig[r] = s;
                        }
                        const double rho = L.rho[e];
                        principalStresses(sig[0], sig[1], sig[2], sig[3], sig[4], sig[5], pr);
                        vm += rho * vonMises(sig);
                        p1 += rho * pr[0];
                        p3 += rho * pr[2];
                        w += rho;
                    }
                }
            }
            if (w > 0) {
                out.nodeVM[n] = float(vm / w);
                out.nodeP1[n] = float(p1 / w);
                out.nodeP3[n] = float(p3 / w);
            }
        }
    }, 256);
    return out;
}

std::vector<float> VoxelFEA::stressTensors(const std::vector<double>& u, double E, double h) const {
    const Level& L = levels[0];
    std::array<std::array<double, 144>, 8> S;
    for (int a = 0; a < 8; a++)
        for (int i = 0; i < 144; i++) S[a][i] = cornerStress[a][i] * E / h;
    std::vector<float> out(6 * L.nNodes, 0.f);
    const int NX = L.NX, NY = L.NY, nx = L.nx, ny = L.ny, nz = L.nz;
    const int64_t NXY = int64_t(NX) * NY;
    parallelFor(L.nNodes, [&](int64_t lo, int64_t hi) {
        double ue[24], acc[6];
        for (int64_t n = lo; n < hi; n++) {
            if (!L.activeNode[n]) continue;
            const int i = int(n % NX), j = int((n / NX) % NY), k = int(n / NXY);
            double w = 0;
            std::fill(acc, acc + 6, 0.0);
            for (int dk = 0; dk < 2; dk++) {
                const int ek = k + dk - 1;
                if (ek < 0 || ek >= nz) continue;
                for (int dj = 0; dj < 2; dj++) {
                    const int ej = j + dj - 1;
                    if (ej < 0 || ej >= ny) continue;
                    for (int di = 0; di < 2; di++) {
                        const int ei = i + di - 1;
                        if (ei < 0 || ei >= nx) continue;
                        const int e = L.emap[ei + nx * (ej + ny * ek)];
                        if (e < 0) continue;
                        const int a = corner(1 - di, 1 - dj, 1 - dk);
                        const int64_t nb = ei + NX * (ej + int64_t(NY) * ek);
                        for (int c = 0; c < 8; c++)
                            for (int d = 0; d < 3; d++) ue[3 * c + d] = u[3 * (nb + L.off[c]) + d];
                        const double rho = L.rho[e];
                        for (int r = 0; r < 6; r++) {
                            double s = 0;
                            for (int q = 0; q < 24; q++) s += S[a][r * 24 + q] * ue[q];
                            acc[r] += rho * s;
                        }
                        w += rho;
                    }
                }
            }
            if (w > 0)
                for (int r = 0; r < 6; r++) out[6 * n + r] = float(acc[r] / w);
        }
    }, 256);
    return out;
}

PruneResult pruneFloating(std::array<int, 3> dims, const std::vector<float>& density, const std::vector<uint8_t>& heldNode) {
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    const int64_t NX = nx + 1, NY = ny + 1;
    const int64_t nE = int64_t(nx) * ny * nz;
    std::vector<uint8_t> seen(nE, 0);
    std::vector<int64_t> queue;
    queue.reserve(nE);
    const int64_t off[8] = {0, 1, 1 + NX, NX, NX * NY, 1 + NX * NY, 1 + NX + NX * NY, NX + NX * NY};
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                const int64_t e = i + int64_t(nx) * (j + int64_t(ny) * k);
                if (!(density[e] > 0)) continue;
                const int64_t n0 = i + NX * (j + NY * k);
                for (int a = 0; a < 8; a++)
                    if (heldNode[n0 + off[a]]) { seen[e] = 1; queue.push_back(e); break; }
            }
    for (size_t h = 0; h < queue.size(); h++) {
        const int64_t e = queue[h];
        const int i = int(e % nx), j = int((e / nx) % ny), k = int(e / (int64_t(nx) * ny));
        const int64_t nb[6] = {i > 0 ? e - 1 : -1, i < nx - 1 ? e + 1 : -1, j > 0 ? e - nx : -1, j < ny - 1 ? e + nx : -1,
                               k > 0 ? e - int64_t(nx) * ny : -1, k < nz - 1 ? e + int64_t(nx) * ny : -1};
        for (int64_t m : nb)
            if (m >= 0 && !seen[m] && density[m] > 0) { seen[m] = 1; queue.push_back(m); }
    }
    PruneResult out;
    out.density.assign(nE, 0.f);
    for (int64_t e = 0; e < nE; e++)
        if (density[e] > 0) {
            if (seen[e]) out.density[e] = density[e];
            else out.removed++;
        }
    return out;
}

}  // namespace ps
