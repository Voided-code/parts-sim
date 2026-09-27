#include "voxel_fea.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "../util/dense.hpp"
#include "../util/parallel.hpp"
#include "../util/trace.hpp"

namespace ps {

namespace {

// 4-lane vectors for the matrix products: compiler vector extensions (NEON, SSE/AVX), else scalars
#if defined(__GNUC__) || defined(__clang__)
typedef float F4 __attribute__((vector_size(16)));
typedef double D4 __attribute__((vector_size(32)));
template <class T> struct Lanes;
template <> struct Lanes<float> { using V = F4; };
template <> struct Lanes<double> { using V = D4; };
template <class T> using V4 = typename Lanes<T>::V;
template <class T> inline V4<T> load4(const T* p) {
    V4<T> v;
    std::memcpy(&v, p, sizeof v);
    return v;
}
template <class T> inline V4<T> load4(const float* p) requires std::is_same_v<T, double> {
    return V4<T>{p[0], p[1], p[2], p[3]};
}
#else
template <class T> struct V4 {
    T v[4] = {0, 0, 0, 0};
    T operator[](int i) const { return v[i]; }
    V4& operator+=(const V4& o) { for (int i = 0; i < 4; i++) v[i] += o.v[i]; return *this; }
    V4 operator*(T s) const { V4 r; for (int i = 0; i < 4; i++) r.v[i] = v[i] * s; return r; }
};
template <class T, class S> inline V4<T> load4(const S* p) {
    V4<T> v;
    for (int i = 0; i < 4; i++) v.v[i] = T(p[i]);
    return v;
}
#endif

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
    TraceTimer trace("  fine level");
    Level L0 = makeLevel(dims[0], dims[1], dims[2]);
    setBase(L0, K0.data());
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
    trace.lap("  coarsen");
    for (;;) {
        Level& L = levels.back();
        if (L.freeDof <= coarsestMaxDof || levels.size() >= 12) break;
        if (L.nx <= 1 && L.ny <= 1 && L.nz <= 1) break;
        Level C = coarsen(L);
        levels.push_back(std::move(C));
    }
    trace.lap("  coarsest factor");
    buildCoarseSolver(levels.back());
}

// a level's uniform element matrix, padded copies and stencil
void VoxelFEA::setBase(Level& L, const double* K) {
    L.Kb.fill(0.0);
    L.Kbf.fill(0.f);
    L.Sb.fill(0.0);
    for (int i = 0; i < 576; i++) {
        L.Kb[i] = K[i];
        L.Kbf[i] = float(K[i]);
    }
    // node n is corner a = corner(1-di, 1-dj, 1-dk) of voxel (di, dj, dk) around it; that voxel's
    // corner b = corner(bx, by, bz) is the neighbour at offset (di+bx, dj+by, dk+bz) - 1
    for (int v = 0; v < 8; v++) {
        const int di = v & 1, dj = (v >> 1) & 1, dk = v >> 2, a = corner(1 - di, 1 - dj, 1 - dk);
        for (int w = 0; w < 8; w++) {
            const int bx = w & 1, by = (w >> 1) & 1, bz = w >> 2, b = corner(bx, by, bz);
            const int row = (dj + by) + 3 * (dk + bz), col = 3 * (di + bx);
            for (int d = 0; d < 3; d++)
                for (int c = 0; c < 3; c++) L.Sb[(row * 9 + col + c) * 4 + d] += K[(3 * a + d) * 24 + 3 * b + c];
        }
    }
    for (int i = 0; i < 324; i++) L.Sbf[i] = float(L.Sb[i]);
}

void VoxelFEA::finishLevel(Level& L) {
    L.activeNode.assign(L.nNodes, 0);
    std::vector<double> diag(L.nDof, 0.0);
    const int64_t nE = L.elems.size();
    for (int64_t e = 0; e < nE; e++) {
        const int64_t n0 = L.base[e];
        const bool sc = L.scaled(e);
        const float* K = sc ? nullptr : L.matrix(e);
        for (int a = 0; a < 8; a++) {
            const int64_t n = n0 + L.off[a];
            L.activeNode[n] = 1;
            for (int d = 0; d < 3; d++) diag[3 * n + d] += sc ? L.rho[e] * L.Kb[(3 * a + d) * 25] : double(K[(3 * a + d) * 25]);
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
    L.invDiag32.assign(L.invDiag.begin(), L.invDiag.end());
    L.rowFirst.assign(int64_t(L.NY) * L.NZ, -1);
    L.rowLast.assign(int64_t(L.NY) * L.NZ, -1);
    L.activeRows.clear();
    for (int64_t row = 0; row < int64_t(L.NY) * L.NZ; row++) {
        for (int i = 0; i < L.NX; i++)
            if (L.activeNode[row * L.NX + i]) {
                if (L.rowFirst[row] < 0) L.rowFirst[row] = i;
                L.rowLast[row] = i;
            }
        if (L.rowFirst[row] >= 0) L.activeRows.push_back(row);
    }
    L.voxelScale.assign(L.emap.size(), 0.f);
    for (size_t v = 0; v < L.emap.size(); v++)
        if (L.emap[v] >= 0) L.voxelScale[v] = L.scaled(L.emap[v]) ? float(L.rho[L.emap[v]]) : -1.f;
    L.nodeScale.assign(L.nNodes, 0.f);
    parallelFor(int64_t(L.nz - 1) * (L.ny - 1), [&](int64_t lo, int64_t hi) {
        for (int64_t q = lo; q < hi; q++) {
            const int j = 1 + int(q % (L.ny - 1)), k = 1 + int(q / (L.ny - 1));
            for (int i = 1; i < L.nx; i++) {
                const float s0 = L.voxelScale[(i - 1) + int64_t(L.nx) * ((j - 1) + int64_t(L.ny) * (k - 1))];
                bool u = s0 > 0;
                for (int v = 1; v < 8 && u; v++)
                    u = L.voxelScale[(i - 1 + (v & 1)) + int64_t(L.nx) * ((j - 1 + ((v >> 1) & 1)) + int64_t(L.ny) * (k - 1 + (v >> 2)))] == s0;
                if (u) L.nodeScale[i + int64_t(L.NX) * (j + int64_t(L.NY) * k)] = s0;
            }
        }
    });
    L.r.assign(L.nDof, 0.f);
    L.z.assign(L.nDof, 0.f);
    L.t.assign(L.nDof, 0.f);
    L.d.assign(L.nDof, 0.f);
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
    // Galerkin products of the fine level's uniform element per child position: M[c] = P_c^T Kb P_c.
    // A coarse element whose 8 children are s * Kb is s * (sum of M[c]), the coarse level's own
    // uniform element; only the others (partly filled, or mixed densities) need their own matrix.
    std::array<std::array<double, 576>, 8> M{};
    {
        std::vector<double> T(576), Kc(576, 0.0);
        for (int c = 0; c < 8; c++) {
            galerkinAdd(M[c].data(), F.Kb.data(), 1.0, c, T.data());
            for (int u = 0; u < 576; u++) Kc[u] += M[c][u];
        }
        setBase(C, Kc.data());
    }
    C.rho.assign(nE, 0.0);
    C.kIdx.assign(nE, -1);
    int nK = 0;
    for (int64_t q = 0; q < nE; q++) {
        const int b = C.childStart[q], e = C.childStart[q + 1];
        bool uniform = e - b == 8 && F.scaled(C.children[b]);
        const double s = uniform ? F.rho[C.children[b]] : 0;
        for (int t = b; t < e && uniform; t++) uniform = F.scaled(C.children[t]) && F.rho[C.children[t]] == s;
        if (uniform) C.rho[q] = s;
        else C.kIdx[q] = nK++;
    }
    C.K.assign(int64_t(nK) * 576 + 4, 0.f);
    parallelFor(nE, [&](int64_t lo, int64_t hi) {
        std::vector<double> T(576), out(576), Kf(576);
        for (int64_t q = lo; q < hi; q++) {
            if (C.kIdx[q] < 0) continue;
            std::fill(out.begin(), out.end(), 0.0);
            for (int t = C.childStart[q]; t < C.childStart[q + 1]; t++) {
                const int f = C.children[t];
                const int c = fineChild[f];
                if (F.scaled(f)) {
                    const double s = F.rho[f];
                    const double* Mc = M[c].data();
                    for (int u = 0; u < 576; u++) out[u] += s * Mc[u];
                } else {
                    const float* Kc = F.matrix(f);
                    for (int u = 0; u < 576; u++) Kf[u] = Kc[u];
                    galerkinAdd(out.data(), Kf.data(), 1.0, c, T.data());
                }
            }
            float* dst = C.K.data() + int64_t(C.kIdx[q]) * 576;
            for (int u = 0; u < 576; u++) dst[u] = float(out[u]);
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

void VoxelFEA::apply(int l, const double* x, double* y, bool raw) const { applyT<double>(l, x, y, raw); }

// Column-oriented products (out[0..3] += column * x) vectorize without reassociating sums: the
// fourth lane is padding. Element matrices and the stencil are symmetric, so the 3 rows of a node
// are read as 3 consecutive entries of each column.
template <class T>
void VoxelFEA::applyT(int l, const T* x, T* y, bool raw) const {
    const Level& L = levels[l];
    const int NX = L.NX, NY = L.NY, NZ = L.NZ, nx = L.nx, ny = L.ny, nz = L.nz;
    const int64_t NXY = int64_t(NX) * NY;
    const T* K0t;
    const T* S0t;
    if constexpr (std::is_same_v<T, float>) { K0t = L.Kbf.data(); S0t = L.Sbf.data(); }
    else { K0t = L.Kb.data(); S0t = L.Sb.data(); }
    // one task per row of nodes along x: node n = i + NX * row, and its 8 voxels (di, dj, dk) =
    // bits of v lie in 4 voxel rows (q = v >> 1 = dj + 2 dk) at columns i - 1 + di
    parallelFor(int64_t(NY) * NZ, [&](int64_t lo, int64_t hi) {
        for (int64_t row = lo; row < hi; row++) {
            // nodes outside the row's active span are 0
            const int first = L.rowFirst[row], last = L.rowLast[row];
            if (first < 0) {
                std::fill(y + 3 * NX * row, y + 3 * NX * (row + 1), T(0));
                continue;
            }
            std::fill(y + 3 * NX * row, y + 3 * (NX * row + first), T(0));
            std::fill(y + 3 * (NX * row + last + 1), y + 3 * NX * (row + 1), T(0));
            const int j = int(row % NY), k = int(row / NY);
            int64_t vrow[4];
            bool has[4];
            for (int q = 0; q < 4; q++) {
                const int ej = j - 1 + (q & 1), ek = k - 1 + (q >> 1);
                has[q] = ej >= 0 && ek >= 0 && ej < ny && ek < nz;
                vrow[q] = has[q] ? int64_t(nx) * (ej + int64_t(ny) * ek) : 0;
            }
            for (int i = first; i <= last; i++) {
                const int64_t n = i + NX * row;
                T acc0 = 0, acc1 = 0, acc2 = 0;
                if (L.activeNode[n]) {
                    // inside a uniform region (the 8 voxels the same s * K0): the 27-point stencil
                    const float s0 = L.nodeScale[n];
                    if (s0 > 0) {
                        // nine rows of three neighbours, 9 contiguous DOFs each
                        const T* x0 = x + 3 * (n - 1 - NX - NXY);
                        V4<T> acc{};
                        for (int r = 0; r < 9; r++) {
                            const T* xr = x0 + 3 * ((r % 3) * NX + (r / 3) * NXY);
                            const T* col = S0t + r * 36;
                            // the row's 9 values in two vector loads and one scalar, used lane by lane
                            const V4<T> xa = load4<T>(xr), xb = load4<T>(xr + 4);
                            V4<T> b = load4<T>(col) * xa[0];
                            b += load4<T>(col + 4) * xa[1];
                            b += load4<T>(col + 8) * xa[2];
                            b += load4<T>(col + 12) * xa[3];
                            b += load4<T>(col + 16) * xb[0];
                            b += load4<T>(col + 20) * xb[1];
                            b += load4<T>(col + 24) * xb[2];
                            b += load4<T>(col + 28) * xb[3];
                            b += load4<T>(col + 32) * xr[8];
                            acc += b;
                        }
                        acc0 = T(s0) * acc[0];
                        acc1 = T(s0) * acc[1];
                        acc2 = T(s0) * acc[2];
                    } else {
                        for (int v = 0; v < 8; v++) {
                            const int q = v >> 1, di = v & 1, ei = i - 1 + di;
                            if (!has[q] || ei < 0 || ei >= nx) continue;
                            const int e = L.emap[vrow[q] + ei];
                            if (e < 0) continue;
                            const int dj = q & 1, dk = q >> 1, a = corner(1 - di, 1 - dj, 1 - dk);
                            // the voxel's corners (bx, by, bz) lie in 4 node rows (by, bz), bx = 0 and 1 side
                            // by side: 6 contiguous values each; column c of its matrix at col + 24 c
                            V4<T> b0{}, b1{};
                            auto columns = [&](const auto* col) {
                                for (int f = 0; f < 4; f++) {
                                    const int by = f & 1, bz = f >> 1;
                                    const T* xs = x + 3 * (n - (1 - di) + NX * (by - (1 - dj)) + NXY * (bz - (1 - dk)));
                                    const auto* c0 = col + 72 * corner(0, by, bz);
                                    const auto* c1 = col + 72 * corner(1, by, bz);
                                    const V4<T> xa = load4<T>(xs);
                                    b0 += load4<T>(c0) * xa[0];
                                    b1 += load4<T>(c0 + 24) * xa[1];
                                    b0 += load4<T>(c0 + 48) * xa[2];
                                    b1 += load4<T>(c1) * xa[3];
                                    b0 += load4<T>(c1 + 24) * xs[4];
                                    b1 += load4<T>(c1 + 48) * xs[5];
                                }
                            };
                            T s = 1;
                            if (L.scaled(e)) {
                                columns(K0t + 3 * a);
                                s = T(L.rho[e]);
                            } else columns(L.matrix(e) + 3 * a);
                            b0 += b1;
                            acc0 += s * b0[0];
                            acc1 += s * b0[1];
                            acc2 += s * b0[2];
                        }
                    }
                    if (!L.diagAdd.empty()) {
                        acc0 += T(L.diagAdd[3 * n]) * x[3 * n];
                        acc1 += T(L.diagAdd[3 * n + 1]) * x[3 * n + 1];
                        acc2 += T(L.diagAdd[3 * n + 2]) * x[3 * n + 2];
                    }
                }
                T* yp = y + 3 * n;
                if (raw) { yp[0] = acc0; yp[1] = acc1; yp[2] = acc2; }
                else {
                    const uint8_t* fx = L.fixed.data() + 3 * n;
                    yp[0] = fx[0] ? 0 : acc0;
                    yp[1] = fx[1] ? 0 : acc1;
                    yp[2] = fx[2] ? 0 : acc2;
                }
            }
        }
    }, std::max<int64_t>(1, 1024 / NX));
}

double tridiagonalMax(const std::vector<double>& a, const std::vector<double>& b) {
    const int k = int(a.size());
    double lo = INFINITY, hi = -INFINITY;
    for (int i = 0; i < k; i++) {
        const double r = (i > 0 ? std::abs(b[i - 1]) : 0) + (i + 1 < k ? std::abs(b[i]) : 0);
        lo = std::min(lo, a[i] - r);
        hi = std::max(hi, a[i] + r);
    }
    auto above = [&](double x) {  // eigenvalues greater than x
        int count = 0;
        double q = 1;
        for (int i = 0; i < k; i++) {
            q = a[i] - x - (i > 0 ? b[i - 1] * b[i - 1] / q : 0);
            if (q == 0) q = 1e-300;
            if (q > 0) count++;
        }
        return count;
    };
    for (int it = 0; it < 100 && hi - lo > 1e-12 * std::abs(hi); it++) {
        const double mid = 0.5 * (lo + hi);
        if (above(mid) > 0) lo = mid;
        else hi = mid;
    }
    return hi;
}

// Largest eigenvalue of D^-1 K on level l, for the smoother: 10 Lanczos steps on the symmetric
// D^-1/2 K D^-1/2 (Ritz values approach the top of the spectrum from below much faster than
// power iteration does)
std::string VoxelFEA::profile(int reps) {
    prepareSmoothers();
    std::string out;
    for (size_t l = 0; l < levels.size(); l++) {
        Level& L = levels[l];
        std::fill(L.z.begin(), L.z.end(), 1.f);
        applyT<float>(int(l), L.z.data(), L.t.data(), false);
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; k++) applyT<float>(int(l), L.z.data(), L.t.data(), false);
        const double t32 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::vector<double> xd(L.nDof, 1.0), yd(L.nDof);
        t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; k++) applyT<double>(int(l), xd.data(), yd.data(), false);
        const double t64 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        int64_t uniformNodes = 0, active = 0;
        for (int64_t n = 0; n < L.nNodes; n++) {
            if (!L.activeNode[n]) continue;
            active++;
            const int i = int(n % L.NX), j = int((n / L.NX) % L.NY), k = int(n / (int64_t(L.NX) * L.NY));
            if (i == 0 || j == 0 || k == 0 || i == L.nx || j == L.ny || k == L.nz) continue;
            const float s0 = L.voxelScale[(i - 1) + int64_t(L.nx) * ((j - 1) + int64_t(L.ny) * (k - 1))];
            bool u = s0 > 0;
            for (int v = 0; v < 8 && u; v++)
                u = L.voxelScale[(i - 1 + (v & 1)) + int64_t(L.nx) * ((j - 1 + ((v >> 1) & 1)) + int64_t(L.ny) * (k - 1 + (v >> 2)))] == s0;
            if (u) uniformNodes++;
        }
        char line[240];
        std::snprintf(line, sizeof line, "CPU L%zu nodes=%lld active=%lld uniform=%lld elems=%zu own matrices=%zu: matvec32 %.3f ms  matvec64 %.3f ms\n", l,
                      (long long)L.nNodes, (long long)active, (long long)uniformNodes, L.elems.size(), L.K.size() / 576, t32, t64);
        out += line;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < reps; k++) vcycle(0);
    char line[100];
    std::snprintf(line, sizeof line, "CPU V-cycle %.3f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
    return out + line;
}

void VoxelFEA::prepareSmoothers() {
    TraceTimer trace("  smoother eigenvalues");
    for (int l = 0; l + 1 < int(levels.size()); l++)
        if (!(levels[l].lmax > 0)) estimateOmega(l);
}

void VoxelFEA::estimateOmega(int l) {
    Level& L = levels[l];
    const int64_t n = L.nDof;
    std::vector<double> sq(n), v(n), vPrev(n, 0.0), w(n), x(n);
    for (int64_t i = 0; i < n; i++) sq[i] = std::sqrt(L.invDiag[i]);
    uint32_t seed = 12345;
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
        apply(l, x.data(), w.data());
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
    const double lambda = tridiagonalMax(alpha, beta);
    L.omega = 1.2 / (lambda * 1.05);
    L.lmax = lambda;
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
        const bool sc = L.scaled(e);
        const float* K = sc ? nullptr : L.matrix(e);
        const double s = L.scale(e);
        for (int r = 0; r < 24; r++) {
            if (dofs[r] < 0) continue;
            for (int c = 0; c < 24; c++)
                if (dofs[c] >= 0) A[dofs[r] * m + dofs[c]] += sc ? s * L.Kb[r * 24 + c] : double(K[r * 24 + c]);
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

void VoxelFEA::coarseSolve64(const double* r, double* z) {
    const Level& L = levels.back();
    const int64_t m = coarse.m;
    const double* A = coarse.A.data();
    std::vector<double> y(m);
    for (int64_t i = 0; i < L.nDof; i++)
        if (coarse.map[i] >= 0) y[coarse.map[i]] = r[i];
    choleskySolve(A, y.data(), m);
    for (int64_t i = 0; i < L.nDof; i++) z[i] = coarse.map[i] >= 0 ? y[coarse.map[i]] : 0;
}

void VoxelFEA::coarseSolve(Level& L) {
    std::fill(L.z.begin(), L.z.end(), 0.f);
    if (coarse.A.empty()) {
        for (int s = 0; s < 40; s++) jacobi(int(levels.size()) - 1, s == 0);
        return;
    }
    const int64_t m = coarse.m;
    const double* A = coarse.A.data();
    double* y = coarse.y.data();
    for (int64_t i = 0; i < L.nDof; i++)
        if (coarse.map[i] >= 0) y[coarse.map[i]] = L.r[i];
    choleskySolve(A, y, m);
    for (int64_t i = 0; i < L.nDof; i++)
        if (coarse.map[i] >= 0) L.z[i] = float(y[coarse.map[i]]);
}

// Chebyshev polynomial smoother (Adams et al. 2003): damps the eigencomponents of D^-1 K in
// [0.1, 1.15] * lmax as evenly as a degree-2 polynomial can, for the cost of two Jacobi sweeps.
ChebyshevCoefficients chebyshevCoefficients(double lmax) {
    const double b = 1.15 * lmax, a = 0.1 * lmax;
    const double theta = (b + a) / 2, delta = (b - a) / 2, sigma = theta / delta;
    ChebyshevCoefficients c;
    c.first = 1 / theta;
    const double rho0 = 1 / sigma, rho1 = 1 / (2 * sigma - rho0);
    c.c1 = rho1 * rho0;
    c.c2 = 2 * rho1 / delta;
    return c;
}

void VoxelFEA::chebyshev(int l, bool first) {
    Level& L = levels[l];
    const auto cc = chebyshevCoefficients(L.lmax);
    const float c0 = float(cc.first), c1 = float(cc.c1), c2 = float(cc.c2);
    const float* D = L.invDiag32.data();
    float *r = L.r.data(), *z = L.z.data(), *t = L.t.data(), *d = L.d.data();
    // step 0: d = D^-1 (r - K z) / theta, z += d
    if (first) {
        L.forActive([&](int64_t lo, int64_t hi) {
            for (int64_t i = lo; i < hi; i++) { d[i] = c0 * D[i] * r[i]; z[i] = d[i]; }
        });
    } else {
        applyT<float>(l, z, t, false);
        L.forActive([&](int64_t lo, int64_t hi) {
            for (int64_t i = lo; i < hi; i++) { d[i] = c0 * D[i] * (r[i] - t[i]); z[i] += d[i]; }
        });
    }
    // step 1: d = c1 d + c2 D^-1 (r - K z), z += d
    applyT<float>(l, z, t, false);
    L.forActive([&](int64_t lo, int64_t hi) {
        for (int64_t i = lo; i < hi; i++) { d[i] = c1 * d[i] + c2 * D[i] * (r[i] - t[i]); z[i] += d[i]; }
    });
}

void VoxelFEA::jacobi(int l, bool first) {
    Level& L = levels[l];
    const float om = float(L.omega);
    if (first) {
        L.forActive([&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.z[i] = om * L.invDiag32[i] * L.r[i]; });
        return;
    }
    applyT<float>(l, L.z.data(), L.t.data(), false);
    L.forActive([&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.z[i] += om * L.invDiag32[i] * (L.r[i] - L.t[i]); });
}

// rc = P^T rf, gathered per coarse node: coarse I takes fine 2I (weight 1) and 2I +- 1 (weight 1/2)
template <class T>
void VoxelFEA::restrict(const Level& F, const Level& C, const T* rf, T* rc, bool zeroFixed) const {
    const int64_t CNXY = int64_t(C.NX) * C.NY;
    auto body = [&](int64_t lo, int64_t hi) {
        for (int64_t cn = lo; cn < hi; cn++) {
            const int I = int(cn % C.NX), J = int((cn / C.NX) % C.NY), K = int(cn / CNXY);
            T s0 = 0, s1 = 0, s2 = 0;
            for (int dk = -1; dk <= 1; dk++) {
                const int kf = 2 * K + dk;
                if (kf < 0 || kf >= F.NZ) continue;
                const T wk = dk == 0 ? 1 : 0.5;
                for (int dj = -1; dj <= 1; dj++) {
                    const int jf = 2 * J + dj;
                    if (jf < 0 || jf >= F.NY) continue;
                    const T wj = dj == 0 ? 1 : 0.5;
                    for (int di = -1; di <= 1; di++) {
                        const int if_ = 2 * I + di;
                        if (if_ < 0 || if_ >= F.NX) continue;
                        const T w = T(di == 0 ? 1 : 0.5) * wj * wk;
                        const T* p = rf + 3 * (if_ + int64_t(F.NX) * (jf + int64_t(F.NY) * kf));
                        s0 += w * p[0];
                        s1 += w * p[1];
                        s2 += w * p[2];
                    }
                }
            }
            T* o = rc + 3 * cn;
            if (zeroFixed) {
                const uint8_t* fx = C.fixed.data() + 3 * cn;
                o[0] = fx[0] ? 0 : s0;
                o[1] = fx[1] ? 0 : s1;
                o[2] = fx[2] ? 0 : s2;
            } else {
                o[0] = s0; o[1] = s1; o[2] = s2;
            }
        }
    };
    // for the V-cycle (held DOFs zeroed) inactive coarse nodes keep 0; otherwise every node
    if (zeroFixed) C.forActiveNodes(body);
    else parallelFor(C.nNodes, body, 512);
}

template <class T>
void VoxelFEA::prolongAdd(const Level& F, const Level& C, const T* zc, T* zf) const {
    const int64_t FNXY = int64_t(F.NX) * F.NY;
    F.forActiveNodes([&](int64_t lo, int64_t hi) {
        for (int64_t fn = lo; fn < hi; fn++) {
            const uint8_t* fx = F.fixed.data() + 3 * fn;
            if (fx[0] && fx[1] && fx[2]) continue;
            const int i = int(fn % F.NX), j = int((fn / F.NX) % F.NY), k = int(fn / FNXY);
            const int i0 = F.mx.c0[i], i1 = F.mx.c1[i], j0 = F.my.c0[j], j1 = F.my.c1[j], k0 = F.mz.c0[k], k1 = F.mz.c1[k];
            const T wi = i1 < 0 ? 1 : 0.5, wj = j1 < 0 ? 1 : 0.5, wk = k1 < 0 ? 1 : 0.5;
            T s0 = 0, s1 = 0, s2 = 0;
            for (int kk = 0; kk < 2; kk++) {
                const int K = kk ? k1 : k0;
                if (K < 0) continue;
                for (int jj = 0; jj < 2; jj++) {
                    const int J = jj ? j1 : j0;
                    if (J < 0) continue;
                    for (int ii = 0; ii < 2; ii++) {
                        const int I = ii ? i1 : i0;
                        if (I < 0) continue;
                        const T w = wi * wj * wk;
                        const T* p = zc + 3 * (I + int64_t(C.NX) * (J + int64_t(C.NY) * K));
                        s0 += w * p[0];
                        s1 += w * p[1];
                        s2 += w * p[2];
                    }
                }
            }
            T* o = zf + 3 * fn;
            if (!fx[0]) o[0] += s0;
            if (!fx[1]) o[1] += s1;
            if (!fx[2]) o[2] += s2;
        }
    });
}

void VoxelFEA::vcycle(int l) {
    Level& L = levels[l];
    if (l == int(levels.size()) - 1) { coarseSolve(L); return; }
    Level& C = levels[l + 1];
    chebyshev(l, true);
    applyT<float>(l, L.z.data(), L.t.data(), false);
    L.forActive([&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.t[i] = L.r[i] - L.t[i]; });
    restrict(L, C, L.t.data(), C.r.data());
    vcycle(l + 1);
    prolongAdd(L, C, C.z.data(), L.z.data());
    chebyshev(l, false);
}

void VoxelFEA::precondition(const double* r, double* z) {
    prepareSmoothers();
    precondition32(r, z);
}

void VoxelFEA::precondition32(const double* r, double* z) {
    Level& L = levels[0];
    const int64_t n = L.nDof;
    if (levels.size() == 1 && !coarse.A.empty()) {
        // the whole model is the coarsest level: its direct solve, exact in 64-bit
        coarseSolve64(r, z);
        return;
    }
    // the V-cycle is linear: scale r to order one so float32 neither underflows nor loses digits
    double rmax = 0;
    {
        std::mutex m;
        L.forActive([&](int64_t lo, int64_t hi) {
            double mx = 0;
            for (int64_t i = lo; i < hi; i++)
                if (!L.fixed[i]) mx = std::max(mx, std::abs(r[i]));
            std::lock_guard<std::mutex> g(m);
            rmax = std::max(rmax, mx);
        });
    }
    if (!(rmax > 0)) { std::fill(z, z + n, 0.0); return; }
    const double s = 1 / rmax;
    L.forActive([&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) L.r[i] = L.fixed[i] ? 0.f : float(r[i] * s); });
    vcycle(0);
    std::fill(z, z + n, 0.0);
    L.forActive([&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) z[i] = double(L.z[i]) * rmax; });
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
    std::vector<double> r(n), p(n), q(n), z(n);
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
    prepareSmoothers();
    // held and inactive DOFs of r, z, p and q stay 0: vector work runs over the active node spans only
    auto dotA = [&](const std::vector<double>& a, const std::vector<double>& b) {
        return L.sumActive([&](int64_t lo, int64_t hi) {
            double s = 0;
            for (int64_t i = lo; i < hi; i++) s += a[i] * b[i];
            return s;
        });
    };
    precondition32(r.data(), z.data());
    p = z;
    double rz = dotA(r, z);
    double rel = std::sqrt(dotA(r, r)) / bnorm;
    int it = 0;
    for (; it < opts.maxIter && rel > opts.tol; it++) {
        apply(0, p.data(), q.data());
        const double pq = dotA(p, q);
        if (!(pq > 0)) break;  // loss of positive-definiteness (mechanism)
        const double alpha = rz / pq;
        L.forActive([&](int64_t lo, int64_t hi) {
            for (int64_t i = lo; i < hi; i++) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        });
        rel = std::sqrt(dotA(r, r)) / bnorm;
        if (opts.onProgress && opts.onProgress(it + 1, rel)) { res.cancelled = true; it++; break; }
        if (rel <= opts.tol) { it++; break; }
        precondition32(r.data(), z.data());
        // flexible (Polak-Ribiere) beta = z.(r - r_old) / rz with r - r_old = -alpha q: robust to the
        // 32-bit preconditioner's rounding
        const double rzNew = dotA(r, z);
        const double beta = std::max(0.0, -alpha * dotA(z, q) / rz);
        rz = rzNew;
        L.forActive([&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) p[i] = z[i] + beta * p[i]; });
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

NodalStresses VoxelFEA::stresses(const std::vector<double>& u, double E, double h, bool principals) const {
    const Level& L = levels[0];
    // corner-stress matrices by columns: ST[a][q] = the 6 stress rows (and 2 of padding) that DOF q
    // of the voxel contributes to at its corner a
    std::vector<double> ST(8 * 24 * 8, 0.0);
    for (int a = 0; a < 8; a++)
        for (int r = 0; r < 6; r++)
            for (int q = 0; q < 24; q++) ST[(a * 24 + q) * 8 + r] = cornerStress[a][r * 24 + q] * E / h;
    NodalStresses out;
    out.nodeVM.assign(L.nNodes, 0.f);
    if (principals) {
        out.nodeP1.assign(L.nNodes, 0.f);
        out.nodeP3.assign(L.nNodes, 0.f);
    }
    // nodal values: density-weighted average of the surrounding voxels' corner stresses
    const int NX = L.NX, nx = L.nx, ny = L.ny;
    const int64_t NXY = int64_t(NX) * L.NY;
    const double* x = u.data();
    L.forActiveNodes([&](int64_t lo, int64_t hi) {
        double sig[6], pr[3];
        for (int64_t n = lo; n < hi; n++) {
            if (!L.activeNode[n]) continue;
            const int i = int(n % NX), j = int((n / NX) % L.NY), k = int(n / NXY);
            double vm = 0, p1 = 0, p3 = 0, w = 0;
            for (int v = 0; v < 8; v++) {
                const int di = v & 1, dj = (v >> 1) & 1, dk = v >> 2;
                const int ei = i + di - 1, ej = j + dj - 1, ek = k + dk - 1;
                if (ei < 0 || ej < 0 || ek < 0 || ei >= nx || ej >= ny || ek >= L.nz) continue;
                const int e = L.emap[ei + nx * (ej + int64_t(ny) * ek)];
                if (e < 0) continue;
                const int a = corner(1 - di, 1 - dj, 1 - dk);
                const double* col = ST.data() + a * 24 * 8;
                // the voxel's corners (bx, by, bz): rows (by, bz) of two nodes, 6 contiguous values
                V4<double> s0{}, s1{};
                for (int f = 0; f < 4; f++) {
                    const int by = f & 1, bz = f >> 1;
                    const double* xs = x + 3 * (n - (1 - di) + NX * (by - (1 - dj)) + NXY * (bz - (1 - dk)));
                    const int q0 = 3 * corner(0, by, bz), q1 = 3 * corner(1, by, bz);
                    for (int d = 0; d < 3; d++) {
                        const double* c0 = col + (q0 + d) * 8;
                        const double* c1 = col + (q1 + d) * 8;
                        s0 += load4<double>(c0) * xs[d];
                        s1 += load4<double>(c0 + 4) * xs[d];
                        s0 += load4<double>(c1) * xs[3 + d];
                        s1 += load4<double>(c1 + 4) * xs[3 + d];
                    }
                }
                for (int r = 0; r < 4; r++) sig[r] = s0[r];
                sig[4] = s1[0];
                sig[5] = s1[1];
                const double rho = L.rho[e];
                vm += rho * vonMises(sig);
                if (principals) {
                    principalStresses(sig[0], sig[1], sig[2], sig[3], sig[4], sig[5], pr);
                    p1 += rho * pr[0];
                    p3 += rho * pr[2];
                }
                w += rho;
            }
            if (w > 0) {
                out.nodeVM[n] = float(vm / w);
                if (principals) {
                    out.nodeP1[n] = float(p1 / w);
                    out.nodeP3[n] = float(p3 / w);
                }
            }
        }
    });
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
