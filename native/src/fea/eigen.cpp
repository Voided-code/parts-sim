#include "eigen.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "../util/parallel.hpp"
#include "hex8.hpp"

namespace ps {

Vec lumpedMass(const VoxelFEA& fea) {
    const Level& L = fea.levels[0];
    Vec m(L.nDof, 0.0);
    for (size_t e = 0; e < L.elems.size(); e++) {
        const double w = L.rho[e] / 8;
        for (int a = 0; a < 8; a++) {
            const int64_t n = 3 * (L.base[e] + L.off[a]);
            m[n] += w;
            m[n + 1] += w;
            m[n + 2] += w;
        }
    }
    return m;
}

static std::array<double, 24> shapeGradients(double xi, double eta, double zeta) {
    std::array<double, 24> g{};
    for (int a = 0; a < 8; a++) {
        const double sx = 2 * HEX_NODES[a][0] - 1, sy = 2 * HEX_NODES[a][1] - 1, sz = 2 * HEX_NODES[a][2] - 1;
        g[3 * a] = 2 * sx * (1 + sy * eta) * (1 + sz * zeta) / 8;
        g[3 * a + 1] = 2 * sy * (1 + sx * xi) * (1 + sz * zeta) / 8;
        g[3 * a + 2] = 2 * sz * (1 + sx * xi) * (1 + sy * eta) / 8;
    }
    return g;
}

const std::array<std::array<double, 64>, 6>& geometricTables() {
    static const auto T = [] {
        std::array<std::array<double, 64>, 9> H{};
        const double g = 1 / std::sqrt(3.0);
        for (double xi : {-g, g})
            for (double eta : {-g, g})
                for (double zeta : {-g, g}) {
                    const auto G = shapeGradients(xi, eta, zeta);
                    for (int k = 0; k < 3; k++)
                        for (int l = 0; l < 3; l++)
                            for (int a = 0; a < 8; a++)
                                for (int b = 0; b < 8; b++) H[3 * k + l][a * 8 + b] += G[3 * a + k] * G[3 * b + l] / 8;
                }
        std::array<std::array<double, 64>, 6> out{};
        auto sum = [&](int p, int q) { std::array<double, 64> r; for (int i = 0; i < 64; i++) r[i] = H[p][i] + H[q][i]; return r; };
        out[0] = H[0]; out[1] = H[4]; out[2] = H[8];
        out[3] = sum(1, 3); out[4] = sum(5, 7); out[5] = sum(6, 2);
        return out;
    }();
    return T;
}

const std::array<double, 144>& centroidB() {
    static const auto B = [] {
        const auto G = shapeGradients(0, 0, 0);
        std::array<double, 144> b{};
        for (int a = 0; a < 8; a++) {
            const int c = 3 * a;
            const double dx = G[c], dy = G[c + 1], dz = G[c + 2];
            b[c] = dx; b[24 + c + 1] = dy; b[48 + c + 2] = dz;
            b[72 + c] = dy; b[72 + c + 1] = dx;
            b[96 + c + 1] = dz; b[96 + c + 2] = dy;
            b[120 + c] = dz; b[120 + c + 2] = dx;
        }
        return b;
    }();
    return B;
}

std::vector<double> elementStresses(const VoxelFEA& fea, const Vec& u, double h) {
    const Level& L = fea.levels[0];
    const auto D = elasticityMatrix(fea.nu);
    const auto& B0 = centroidB();
    std::vector<double> out(6 * L.elems.size());
    parallelFor(int64_t(L.elems.size()), [&](int64_t lo, int64_t hi) {
        for (int64_t e = lo; e < hi; e++) {
            double eps[6] = {0, 0, 0, 0, 0, 0};
            for (int a = 0; a < 8; a++) {
                const int64_t n = 3 * (L.base[e] + L.off[a]);
                for (int d = 0; d < 3; d++) {
                    const double v = u[n + d] / h;
                    if (v == 0) continue;
                    const int c = 3 * a + d;
                    for (int i = 0; i < 6; i++) eps[i] += B0[i * 24 + c] * v;
                }
            }
            for (int i = 0; i < 6; i++) {
                double s = 0;
                for (int j = 0; j < 6; j++) s += D[i * 6 + j] * eps[j];
                out[6 * e + i] = s;
            }
        }
    }, 1024);
    return out;
}

GeometricStiffness::GeometricStiffness(const VoxelFEA& fea, const std::vector<double>& sigma) : L_(fea.levels[0]) {
    const auto& T = geometricTables();
    G_.assign(64 * L_.elems.size(), 0.0);
    for (size_t e = 0; e < L_.elems.size(); e++) {
        const double rho = L_.rho[e];
        for (int c = 0; c < 6; c++) {
            const double s = rho * sigma[6 * e + c];
            if (s == 0) continue;
            for (int i = 0; i < 64; i++) G_[64 * e + i] += s * T[c][i];
        }
    }
}

void GeometricStiffness::apply(const double* x, double* y) const {
    const Level& L = L_;
    std::fill(y, y + L.nDof, 0.0);
    // element scatter, by colour so voxels updating the same node never run together
    for (int colour = 0; colour < 8; colour++) {
        parallelFor(int64_t(L.elems.size()), [&](int64_t lo, int64_t hi) {
            double ue[24];
            for (int64_t e = lo; e < hi; e++) {
                const int v = L.elems[e];
                const int i = v % L.nx, j = (v / L.nx) % L.ny, k = v / (L.nx * L.ny);
                if (((i & 1) | ((j & 1) << 1) | ((k & 1) << 2)) != colour) continue;
                const int64_t n0 = L.base[e];
                for (int a = 0; a < 8; a++) {
                    const int64_t n = 3 * (n0 + L.off[a]);
                    ue[3 * a] = x[n]; ue[3 * a + 1] = x[n + 1]; ue[3 * a + 2] = x[n + 2];
                }
                const double* G = &G_[64 * e];
                for (int a = 0; a < 8; a++) {
                    double sx = 0, sy = 0, sz = 0;
                    for (int b = 0; b < 8; b++) {
                        const double g = G[a * 8 + b];
                        sx += g * ue[3 * b]; sy += g * ue[3 * b + 1]; sz += g * ue[3 * b + 2];
                    }
                    const int64_t n = 3 * (n0 + L.off[a]);
                    y[n] += sx; y[n + 1] += sy; y[n + 2] += sz;
                }
            }
        }, 2048);
    }
    for (int64_t i = 0; i < L.nDof; i++)
        if (L.fixed[i]) y[i] = 0;
}

// ---------- small dense linear algebra ----------

static bool cholesky(std::vector<double>& A, int k) {
    for (int j = 0; j < k; j++) {
        double s = A[j * k + j];
        for (int q = 0; q < j; q++) s -= A[j * k + q] * A[j * k + q];
        if (!(s > 0)) return false;
        const double d = std::sqrt(s);
        A[j * k + j] = d;
        for (int i = j + 1; i < k; i++) {
            double t = A[i * k + j];
            for (int q = 0; q < j; q++) t -= A[i * k + q] * A[j * k + q];
            A[i * k + j] = t / d;
        }
        for (int i = 0; i < j; i++) A[i * k + j] = 0;
    }
    return true;
}

void symmetricEigen(const std::vector<double>& Ain, int k, std::vector<double>& values, std::vector<double>& vectors) {
    std::vector<double> A = Ain, V(size_t(k * k), 0.0);
    for (int i = 0; i < k; i++) V[i * k + i] = 1;
    for (int sweep = 0; sweep < 100; sweep++) {
        double off = 0, total = 0;
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++) {
                const double a = A[i * k + j] * A[i * k + j];
                total += a;
                if (i != j) off += a;
            }
        if (off <= 1e-30 * total || off == 0) break;
        for (int p = 0; p < k - 1; p++)
            for (int q = p + 1; q < k; q++) {
                const double apq = A[p * k + q];
                if (std::abs(apq) < 1e-300) continue;
                const double theta = (A[q * k + q] - A[p * k + p]) / (2 * apq);
                const double t = (theta >= 0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1));
                const double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int r = 0; r < k; r++) {
                    const double arp = A[r * k + p], arq = A[r * k + q];
                    A[r * k + p] = c * arp - s * arq;
                    A[r * k + q] = s * arp + c * arq;
                }
                for (int r = 0; r < k; r++) {
                    const double apr = A[p * k + r], aqr = A[q * k + r];
                    A[p * k + r] = c * apr - s * aqr;
                    A[q * k + r] = s * apr + c * aqr;
                }
                for (int r = 0; r < k; r++) {
                    const double vrp = V[r * k + p], vrq = V[r * k + q];
                    V[r * k + p] = c * vrp - s * vrq;
                    V[r * k + q] = s * vrp + c * vrq;
                }
            }
    }
    std::vector<int> order(k);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return A[a * k + a] < A[b * k + b]; });
    values.resize(k);
    vectors.assign(size_t(k * k), 0.0);
    for (int dst = 0; dst < k; dst++) {
        values[dst] = A[order[dst] * k + order[dst]];
        for (int r = 0; r < k; r++) vectors[r * k + dst] = V[r * k + order[dst]];
    }
}

// gA c = theta gB c (gB SPD); false when gB is not positive definite
static bool generalizedEigen(const std::vector<double>& gA, const std::vector<double>& gB, int k, std::vector<double>& values, std::vector<double>& vectors) {
    std::vector<double> Lc = gB;
    if (!cholesky(Lc, k)) return false;
    std::vector<double> Y(size_t(k * k)), C(size_t(k * k));
    for (int col = 0; col < k; col++)
        for (int i = 0; i < k; i++) {
            double s = gA[i * k + col];
            for (int q = 0; q < i; q++) s -= Lc[i * k + q] * Y[q * k + col];
            Y[i * k + col] = s / Lc[i * k + i];
        }
    for (int row = 0; row < k; row++)
        for (int i = 0; i < k; i++) {
            double s = Y[row * k + i];
            for (int q = 0; q < i; q++) s -= Lc[i * k + q] * C[row * k + q];
            C[row * k + i] = s / Lc[i * k + i];
        }
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++) C[i * k + j] = C[j * k + i] = 0.5 * (C[i * k + j] + C[j * k + i]);
    std::vector<double> ev;
    symmetricEigen(C, k, values, ev);
    vectors.assign(size_t(k * k), 0.0);
    for (int col = 0; col < k; col++)
        for (int i = k - 1; i >= 0; i--) {
            double s = ev[i * k + col];
            for (int q = i + 1; q < k; q++) s -= Lc[q * k + i] * vectors[q * k + col];
            vectors[i * k + col] = s / Lc[i * k + i];
        }
    return true;
}

// ---------- LOBPCG ----------

static Block combine(const Block& V, const std::vector<double>& C, int k, int rowOff, int cols, int64_t n) {
    Block out(cols, Vec(n, 0.0));
    parallelFor(n, [&](int64_t lo, int64_t hi) {
        for (int j = 0; j < cols; j++) {
            double* o = out[j].data();
            for (size_t i = 0; i < V.size(); i++) {
                const double c = C[(rowOff + i) * k + j];
                if (c == 0) continue;
                const double* v = V[i].data();
                for (int64_t t = lo; t < hi; t++) o[t] += c * v[t];
            }
        }
    }, 8192);
    return out;
}

static void addInto(Block& a, const Block& b) {
    for (size_t j = 0; j < a.size(); j++) {
        double* x = a[j].data();
        const double* y = b[j].data();
        parallelFor(int64_t(a[j].size()), [&](int64_t lo, int64_t hi) { for (int64_t t = lo; t < hi; t++) x[t] += y[t]; });
    }
}

// B-orthonormalize a block in place (with its A and B images) by Cholesky of its Gram matrix
static bool bOrthonormalize(Block& V, Block* AV, Block& BV) {
    const int k = int(V.size());
    if (!k) return false;
    const int64_t n = int64_t(V[0].size());
    std::vector<double> G(size_t(k * k));
    for (int i = 0; i < k; i++)
        for (int j = 0; j <= i; j++) G[i * k + j] = G[j * k + i] = dot(V[i].data(), BV[j].data(), n);
    double scale = 0;
    for (int i = 0; i < k; i++) scale = std::max(scale, G[i * k + i]);
    if (!(scale > 0) || !cholesky(G, k)) return false;
    for (int i = 0; i < k; i++)
        if (!(G[i * k + i] > 1e-7 * std::sqrt(scale))) return false;
    for (Block* M : {&V, AV, &BV}) {
        if (!M) continue;
        for (int j = 0; j < k; j++) {
            double* v = (*M)[j].data();
            for (int q = 0; q < j; q++) {
                const double c = G[j * k + q];
                const double* w = (*M)[q].data();
                for (int64_t t = 0; t < n; t++) v[t] -= c * w[t];
            }
            const double d = G[j * k + j];
            for (int64_t t = 0; t < n; t++) v[t] /= d;
        }
    }
    return true;
}

LobpcgResult lobpcg(int64_t n, const std::function<void(const double*, double*)>& applyA, const std::function<void(const double*, double*)>& applyB,
                    const std::function<Block(const Block&)>& precond, const LobpcgOptions& o) {
    const int nev = o.nev;
    const int m = std::max(nev, o.block > 0 ? o.block : nev + std::min(4, std::max(2, nev)));
    auto image = [&](const Block& V, const std::function<void(const double*, double*)>& op) {
        Block out(V.size(), Vec(n));
        for (size_t j = 0; j < V.size(); j++) op(V[j].data(), out[j].data());
        return out;
    };
    uint32_t seed = 12345;
    auto rnd = [&] { seed = (seed * 1103515245u + 12345u) & 0x7fffffffu; return double(seed) / 0x7fffffff - 0.5; };
    Block X(m, Vec(n));
    for (auto& v : X)
        for (auto& x : v) x = rnd();
    X = precond(X);  // smooth, low-energy starting shapes
    Block BX = image(X, applyB);
    if (!bOrthonormalize(X, nullptr, BX)) throw std::runtime_error("Could not start the eigenvalue solver (model has too few free nodes).");
    Block AX = image(X, applyA);
    std::vector<double> theta;
    {
        std::vector<double> gA(size_t(m * m)), gB(size_t(m * m)), vals, vecs;
        for (int i = 0; i < m; i++)
            for (int j = 0; j <= i; j++) {
                gA[i * m + j] = gA[j * m + i] = dot(X[i].data(), AX[j].data(), n);
                gB[i * m + j] = gB[j * m + i] = dot(X[i].data(), BX[j].data(), n);
            }
        if (!generalizedEigen(gA, gB, m, vals, vecs)) throw std::runtime_error("Eigenvalue solver failed to start.");
        X = combine(X, vecs, m, 0, m, n);
        AX = combine(AX, vecs, m, 0, m, n);
        BX = combine(BX, vecs, m, 0, m, n);
        theta.assign(vals.begin(), vals.begin() + m);
    }
    Block P, AP, BP;
    std::vector<double> res(m, 1.0);
    LobpcgResult out;
    int it = 0;
    for (; it < o.maxIter; it++) {
        Block R;
        std::vector<int> active;
        for (int j = 0; j < m; j++) {
            Vec r(n);
            const double th = theta[j];
            const double* ax = AX[j].data();
            const double* bx = BX[j].data();
            double rr = 0, na = 0, nb = 0;
            for (int64_t t = 0; t < n; t++) {
                const double v = ax[t] - th * bx[t];
                r[t] = v;
                rr += v * v;
                na += ax[t] * ax[t];
                nb += bx[t] * bx[t];
            }
            const double den = std::sqrt(na) + std::abs(th) * std::sqrt(nb);
            res[j] = std::sqrt(rr) / (den > 0 ? den : 1);
            if (res[j] > o.tol) { active.push_back(j); R.push_back(std::move(r)); }
        }
        int conv = 0;
        double worst = 0;
        for (int j = 0; j < nev; j++) { conv += res[j] <= o.tol; worst = std::max(worst, res[j]); }
        if (o.onProgress && o.onProgress(it, worst, conv)) throw std::runtime_error("Cancelled");
        if (conv == nev) break;
        Block W = precond(R);
        // B-orthogonal to X
        for (auto& w : W)
            for (int i = 0; i < m; i++) {
                const double c = dot(BX[i].data(), w.data(), n);
                const double* x = X[i].data();
                for (int64_t t = 0; t < n; t++) w[t] -= c * x[t];
            }
        Block BW = image(W, applyB);
        if (!bOrthonormalize(W, nullptr, BW)) {
            // dependent search directions: drop the momentum and try once more with a jitter
            P.clear(); AP.clear(); BP.clear();
            for (auto& w : W)
                for (auto& x : w) x += 1e-8 * rnd();
            BW = image(W, applyB);
            if (!bOrthonormalize(W, nullptr, BW)) break;
        }
        Block AW = image(W, applyA);
        bool usedP = false;
        if (!P.empty()) {
            Block P2, AP2, BP2;
            for (int j : active) { P2.push_back(P[j]); AP2.push_back(AP[j]); BP2.push_back(BP[j]); }
            P = std::move(P2); AP = std::move(AP2); BP = std::move(BP2);
            usedP = !P.empty() && bOrthonormalize(P, &AP, BP);
        }
        const int nW = int(W.size()), nP = usedP ? int(P.size()) : 0;
        int k = m + nW + nP;
        auto vecAt = [&](const Block& a, const Block& b, const Block& c, int i) -> const Vec& { return i < m ? a[i] : i < m + nW ? b[i - m] : c[i - m - nW]; };
        std::vector<double> gA(size_t(k * k)), gB(size_t(k * k));
        for (int i = 0; i < k; i++)
            for (int j = 0; j <= i; j++) {
                const Vec& Si = vecAt(X, W, P, i);
                const Vec& Sj = vecAt(X, W, P, j);
                gA[i * k + j] = gA[j * k + i] = 0.5 * (dot(Si.data(), vecAt(AX, AW, AP, j).data(), n) + dot(Sj.data(), vecAt(AX, AW, AP, i).data(), n));
                gB[i * k + j] = gB[j * k + i] = 0.5 * (dot(Si.data(), vecAt(BX, BW, BP, j).data(), n) + dot(Sj.data(), vecAt(BX, BW, BP, i).data(), n));
            }
        std::vector<double> vals, C;
        bool ok = generalizedEigen(gA, gB, k, vals, C);
        if (!ok && usedP) {
            // ill-conditioned with the momentum block: drop it
            const int kk = m + nW;
            std::vector<double> a2(size_t(kk * kk)), b2(size_t(kk * kk));
            for (int i = 0; i < kk; i++)
                for (int j = 0; j < kk; j++) { a2[i * kk + j] = gA[i * k + j]; b2[i * kk + j] = gB[i * k + j]; }
            k = kk;
            usedP = false;
            ok = generalizedEigen(a2, b2, k, vals, C);
        }
        if (!ok) break;
        theta.assign(vals.begin(), vals.begin() + m);
        Block newP = combine(W, C, k, m, m, n), newAP = combine(AW, C, k, m, m, n), newBP = combine(BW, C, k, m, m, n);
        if (usedP) {
            addInto(newP, combine(P, C, k, m + nW, m, n));
            addInto(newAP, combine(AP, C, k, m + nW, m, n));
            addInto(newBP, combine(BP, C, k, m + nW, m, n));
        }
        X = combine(X, C, k, 0, m, n);
        addInto(X, newP);
        AX = combine(AX, C, k, 0, m, n);
        addInto(AX, newAP);
        BX = combine(BX, C, k, 0, m, n);
        addInto(BX, newBP);
        P = std::move(newP);
        AP = std::move(newAP);
        BP = std::move(newBP);
    }
    out.values.assign(theta.begin(), theta.begin() + nev);
    out.vectors.assign(X.begin(), X.begin() + nev);
    out.residuals.assign(res.begin(), res.begin() + nev);
    out.iterations = it;
    out.converged = std::all_of(out.residuals.begin(), out.residuals.end(), [&](double r) { return r <= o.tol * 10; });
    return out;
}

// ---------- studies ----------

FreeDofs::FreeDofs(const VoxelFEA& fea) {
    const Level& L = fea.levels[0];
    for (int64_t i = 0; i < L.nDof; i++)
        if (!L.fixed[i]) idx.push_back(i);
    n = int64_t(idx.size());
}

void FreeDofs::scatter(const double* x, double* full, int64_t nFull) const {
    std::fill(full, full + nFull, 0.0);
    for (int64_t t = 0; t < n; t++) full[idx[t]] = x[t];
}

void FreeDofs::gather(const double* full, double* x) const {
    for (int64_t t = 0; t < n; t++) x[t] = full[idx[t]];
}

FullPreconditioner cpuPreconditioner(VoxelFEA& fea) {
    return [&fea](const Block& R) {
        Block Z;
        for (const auto& r : R) {
            Vec z(r.size());
            fea.precondition(r.data(), z.data());
            Z.push_back(std::move(z));
        }
        return Z;
    };
}

namespace {
struct Compact {
    VoxelFEA& fea;
    FreeDofs map;
    mutable Vec full, out;
    const FullPreconditioner& pre;
    Compact(VoxelFEA& f, const FullPreconditioner& p) : fea(f), map(f), full(f.nDof()), out(f.nDof()), pre(p) {}
    void applyK(const double* x, double* y) const {
        map.scatter(x, full.data(), fea.nDof());
        fea.apply(0, full.data(), out.data());
        map.gather(out.data(), y);
    }
    Block precond(const Block& R) const {
        Block F;
        for (const auto& r : R) {
            Vec f(fea.nDof());
            map.scatter(r.data(), f.data(), fea.nDof());
            F.push_back(std::move(f));
        }
        Block Z = pre(F), out2;
        for (const auto& z : Z) {
            Vec c(map.n);
            map.gather(z.data(), c.data());
            out2.push_back(std::move(c));
        }
        return out2;
    }
    Vec expand(const Vec& x) const {
        Vec f(fea.nDof());
        map.scatter(x.data(), f.data(), fea.nDof());
        return f;
    }
};
}  // namespace

FrequencyResult naturalFrequencies(VoxelFEA& fea, int nev, double E, double density, double h, double shift, const FullPreconditioner& pre,
                                   const std::function<bool(int, double, int)>& onProgress, double tol) {
    FrequencyResult r;
    r.mass = lumpedMass(fea);
    Compact c(fea, pre);
    Vec mc(c.map.n);
    c.map.gather(r.mass.data(), mc.data());
    LobpcgOptions o;
    o.nev = nev;
    o.tol = tol;
    o.onProgress = onProgress;
    auto res = lobpcg(c.map.n, [&](const double* x, double* y) { c.applyK(x, y); },
                      [&](const double* x, double* y) { for (int64_t t = 0; t < c.map.n; t++) y[t] = mc[t] * x[t]; },
                      [&](const Block& R) { return c.precond(R); }, o);
    for (double v : res.values) {
        const double l = std::max(0.0, v - shift);  // lambda = w^2 rho h^2 / E
        r.lambdas.push_back(l);
        r.freqs.push_back(std::sqrt(l * E / (density * h * h)) / (2 * M_PI));
    }
    for (auto& v : res.vectors) r.modes.push_back(c.expand(v));
    r.converged = res.converged;
    r.iterations = res.iterations;
    return r;
}

BucklingResult bucklingFactors(VoxelFEA& fea, const std::vector<double>& sigma, int nev, const FullPreconditioner& pre,
                               const std::function<bool(int, double, int)>& onProgress, double tol) {
    GeometricStiffness KG(fea, sigma);
    Compact c(fea, pre);
    Vec full(fea.nDof()), out(fea.nDof());
    LobpcgOptions o;
    o.nev = nev;
    o.tol = tol;
    o.onProgress = onProgress;
    // smallest (most negative) theta of K_G x = theta K x  <=>  buckling factor -1/theta
    auto res = lobpcg(c.map.n,
                      [&](const double* x, double* y) {
                          c.map.scatter(x, full.data(), fea.nDof());
                          KG.apply(full.data(), out.data());
                          c.map.gather(out.data(), y);
                      },
                      [&](const double* x, double* y) { c.applyK(x, y); }, [&](const Block& R) { return c.precond(R); }, o);
    BucklingResult r;
    for (double t : res.values) r.factors.push_back(t < 0 ? -1 / t : INFINITY);
    for (auto& v : res.vectors) r.modes.push_back(c.expand(v));
    r.converged = res.converged;
    r.iterations = res.iterations;
    return r;
}

}  // namespace ps
