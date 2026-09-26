#include "nonlinear.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <stdexcept>

#include "../util/parallel.hpp"
#include "eigen.hpp"
#include "hex8.hpp"

namespace ps {

namespace {

constexpr double REF[8][3] = {{-0.5, -0.5, -0.5}, {0.5, -0.5, -0.5}, {0.5, 0.5, -0.5}, {-0.5, 0.5, -0.5},
                              {-0.5, -0.5, 0.5},  {0.5, -0.5, 0.5},  {0.5, 0.5, 0.5},  {-0.5, 0.5, 0.5}};

inline double vonMises6(const double* s) {
    return std::sqrt(0.5 * ((s[0] - s[1]) * (s[0] - s[1]) + (s[1] - s[2]) * (s[1] - s[2]) + (s[2] - s[0]) * (s[2] - s[0])) +
                     3 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
}

}  // namespace

void polarRotation(const double* F, double* out) {
    double R[9], inv[9];
    std::copy(F, F + 9, R);
    for (int it = 0; it < 30; it++) {
        const double a = R[0], b = R[1], c = R[2], d = R[3], e = R[4], f = R[5], g = R[6], h = R[7], i = R[8];
        const double A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
        const double det = a * A + b * B + c * C;
        if (!(std::abs(det) > 1e-300)) break;
        // inverse transpose = cofactor / det
        inv[0] = A / det; inv[1] = B / det; inv[2] = C / det;
        inv[3] = -(b * i - c * h) / det; inv[4] = (a * i - c * g) / det; inv[5] = -(a * h - b * g) / det;
        inv[6] = (b * f - c * e) / det; inv[7] = -(a * f - c * d) / det; inv[8] = (a * e - b * d) / det;
        // scaling (Frobenius) speeds up convergence far from orthogonal
        double nR = 0, nI = 0;
        for (int k = 0; k < 9; k++) { nR += R[k] * R[k]; nI += inv[k] * inv[k]; }
        double g2 = std::sqrt(std::sqrt(nI / nR));
        if (!(g2 > 0) || !std::isfinite(g2)) g2 = 1;
        double diff = 0;
        for (int k = 0; k < 9; k++) {
            const double v = 0.5 * (g2 * R[k] + inv[k] / g2);
            diff = std::max(diff, std::abs(v - R[k]));
            R[k] = v;
        }
        if (diff < 1e-13) break;
    }
    std::copy(R, R + 9, out);
}

std::optional<Hardening> hardeningFor(const Material& m) {
    if (m.brittle) return std::nullopt;
    const double E = m.E * 1e3;  // GPa -> MPa
    const double eu = m.elongation > 0 ? m.elongation : 0.15;
    const double epsPlastic = std::max(1e-3, eu - m.uts / E);
    const double H = std::max(0.0, (m.uts - m.yield) / epsPlastic);  // MPa
    return Hardening{m.yield / E, std::max(H / E, 1e-5), eu, m.uts / E};
}

FcgResult fcg(const std::function<void(const double*, double*)>& op, const Precond& precond, const std::vector<double>& b, double tol, int maxIter) {
    const int64_t n = int64_t(b.size());
    FcgResult res;
    res.x.assign(n, 0.0);
    std::vector<double> r = b, q(n), p(n), rOld(n), z(n);
    const double bn = std::sqrt(dot(b.data(), b.data(), n));
    if (!(bn > 0)) return res;
    double rel = 1;
    precond(r.data(), z.data());
    p = z;
    double rz = dot(r.data(), z.data(), n);
    int it = 0;
    for (; it < maxIter; it++) {
        op(p.data(), q.data());
        const double pq = dot(p.data(), q.data(), n);
        if (!(pq > 0)) break;
        const double alpha = rz / pq;
        rOld = r;
        parallelFor(n, [&](int64_t lo, int64_t hi) {
            for (int64_t i = lo; i < hi; i++) { res.x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        });
        rel = std::sqrt(dot(r.data(), r.data(), n)) / bn;
        if (rel <= tol) { it++; break; }
        precond(r.data(), z.data());
        const double num = parallelSum(n, [&](int64_t lo, int64_t hi) {
            double s = 0;
            for (int64_t i = lo; i < hi; i++) s += z[i] * (r[i] - rOld[i]);
            return s;
        });
        const double beta = std::max(0.0, num / rz);
        rz = dot(r.data(), z.data(), n);
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) p[i] = z[i] + beta * p[i]; });
    }
    res.iterations = it;
    res.residual = rel;
    return res;
}

// ---------- model ----------

NonlinearModel::NonlinearModel(const VoxelFEA& fea, bool largeDisplacement, std::optional<Hardening> pl)
    : L(fea.levels[0]), large(largeDisplacement), plastic(pl), fea_(fea) {
    nE_ = int64_t(L.elems.size());
    D_ = elasticityMatrix(fea.nu);
    G_ = 1 / (2 * (1 + fea.nu));
    if (plastic) {
        ep_.assign(48 * nE_, 0.0);
        epTrial_.assign(48 * nE_, 0.0);
        alpha.assign(8 * nE_, 0.0);
        alphaTrial_.assign(8 * nE_, 0.0);
        gpStress_.assign(48 * nE_, 0.0);
        flow_.assign(64 * nE_, 0.0);  // consistent tangent data per yielding point: n (6), c1, c2
        yielding_.assign(nE_, 0);     // bit g set = Gauss point g is yielding in this iteration
        everPlastic_.assign(nE_, 0);
    }
    R_.assign(9 * nE_, 0.0);
    for (int64_t e = 0; e < nE_; e++) R_[9 * e] = R_[9 * e + 4] = R_[9 * e + 8] = 1;
    if (large) geo_.assign(64 * nE_, 0.0);  // stress stiffness of each voxel (8 x 8)
    for (int64_t e = 0; e < nE_; e++) {
        const int v = L.elems[e];
        const int i = v % L.nx, j = (v / L.nx) % L.ny, k = v / (L.nx * L.ny);
        colours_[(i & 1) | ((j & 1) << 1) | ((k & 1) << 2)].push_back(e);
    }
}

// fn(e) for every voxel, in parallel; voxels running together never share a node
template <class F>
void NonlinearModel::forElements(F&& fn) const {
    for (const auto& list : colours_)
        parallelFor(int64_t(list.size()), [&](int64_t lo, int64_t hi) { for (int64_t t = lo; t < hi; t++) fn(list[t]); }, 256);
}

void NonlinearModel::localDisplacement(int64_t e, const std::vector<double>& u, double* ue, double* ul, bool storeRotation) {
    const int64_t n0 = L.base[e];
    for (int a = 0; a < 8; a++) {
        const int64_t n = 3 * (n0 + L.off[a]);
        ue[3 * a] = u[n]; ue[3 * a + 1] = u[n + 1]; ue[3 * a + 2] = u[n + 2];
    }
    if (!large) { std::copy(ue, ue + 24, ul); return; }
    // mean deformation gradient F = I + sum_a u_a (x) grad N_a(centre)
    const auto& B0 = centroidB();
    double F[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
    for (int a = 0; a < 8; a++) {
        const double gx = B0[3 * a], gy = B0[24 + 3 * a + 1], gz = B0[48 + 3 * a + 2];
        for (int i = 0; i < 3; i++) {
            const double v = ue[3 * a + i];
            F[3 * i] += v * gx; F[3 * i + 1] += v * gy; F[3 * i + 2] += v * gz;
        }
    }
    double Rl[9];
    polarRotation(F, Rl);
    if (storeRotation) std::copy(Rl, Rl + 9, &R_[9 * e]);
    const double* R = Rl;
    double cx = 0, cy = 0, cz = 0;
    for (int a = 0; a < 8; a++) { cx += ue[3 * a]; cy += ue[3 * a + 1]; cz += ue[3 * a + 2]; }
    cx /= 8; cy /= 8; cz /= 8;
    for (int a = 0; a < 8; a++) {
        const double* X = REF[a];
        const double dx = X[0] + ue[3 * a] - cx, dy = X[1] + ue[3 * a + 1] - cy, dz = X[2] + ue[3 * a + 2] - cz;
        // R^T d - X
        ul[3 * a] = R[0] * dx + R[3] * dy + R[6] * dz - X[0];
        ul[3 * a + 1] = R[1] * dx + R[4] * dy + R[7] * dz - X[1];
        ul[3 * a + 2] = R[2] * dx + R[5] * dy + R[8] * dz - X[2];
    }
}

double NonlinearModel::internalForce(const std::vector<double>& u, std::vector<double>& out) {
    std::fill(out.begin(), out.end(), 0.0);
    const auto& K = fea_.K0;
    const auto& D = D_;
    const auto& B0 = centroidB();
    const auto& T = geometricTables();
    const double G = G_;
    std::mutex mx;
    double maxAlpha = 0;
    for (const auto& list : colours_)
        parallelFor(int64_t(list.size()), [&](int64_t lo, int64_t hi) {
            double ue[24], ul[24], fl[24], eps[6], sig[6], mean[6];
            double localMax = 0;
            for (int64_t t = lo; t < hi; t++) {
                const int64_t e = list[t];
                localDisplacement(e, u, ue, ul, true);
                std::fill(mean, mean + 6, 0.0);
                if (!plastic) {
                    for (int r = 0; r < 24; r++) {
                        double s = 0;
                        for (int c = 0; c < 24; c++) s += K[r * 24 + c] * ul[c];
                        fl[r] = s;
                    }
                    if (large) {
                        // mean stress D B0 u for the stress stiffness
                        std::fill(eps, eps + 6, 0.0);
                        for (int c = 0; c < 24; c++) {
                            const double v = ul[c];
                            if (v != 0)
                                for (int i = 0; i < 6; i++) eps[i] += B0[i * 24 + c] * v;
                        }
                        for (int i = 0; i < 6; i++) {
                            double s = 0;
                            for (int j = 0; j < 6; j++) s += D[i * 6 + j] * eps[j];
                            mean[i] = s;
                        }
                    }
                } else {
                    const Hardening& pl = *plastic;
                    std::fill(fl, fl + 24, 0.0);
                    uint8_t bits = 0;
                    for (int g = 0; g < 8; g++) {
                        const auto& B = fea_.gaussB[g];
                        const int64_t o = 48 * e + 6 * g, oa = 8 * e + g;
                        std::fill(eps, eps + 6, 0.0);
                        for (int c = 0; c < 24; c++) {
                            const double v = ul[c];
                            if (v != 0)
                                for (int i = 0; i < 6; i++) eps[i] += B[i * 24 + c] * v;
                        }
                        for (int i = 0; i < 6; i++) eps[i] -= ep_[o + i];
                        for (int i = 0; i < 6; i++) {
                            double s = 0;
                            for (int j = 0; j < 6; j++) s += D[i * 6 + j] * eps[j];
                            sig[i] = s;
                        }
                        double al = alpha[oa];
                        for (int i = 0; i < 6; i++) epTrial_[o + i] = ep_[o + i];
                        const double p = (sig[0] + sig[1] + sig[2]) / 3;
                        const double s0 = sig[0] - p, s1 = sig[1] - p, s2 = sig[2] - p;
                        const double q = std::sqrt(1.5 * (s0 * s0 + s1 * s1 + s2 * s2 + 2 * (sig[3] * sig[3] + sig[4] * sig[4] + sig[5] * sig[5])));
                        const double fy = q - (pl.yield + pl.H * al);
                        if (fy > 0 && q > 0) {
                            // radial return with its consistent (algorithmic) tangent:
                            // D_ep = D - 2G(1 - th) I_dev - 2G thb n (x) n, n = s / |s|
                            const double dg = fy / (3 * G + pl.H);
                            const double k = (1.5 / q) * dg;
                            const double th = 1 - (3 * G * dg) / q, thb = 1 / (1 + pl.H / (3 * G)) - (1 - th);
                            double* f = &flow_[64 * e + 8 * g];
                            const double ns = 1 / (std::sqrt(2.0 / 3) * q);
                            f[0] = s0 * ns; f[1] = s1 * ns; f[2] = s2 * ns; f[3] = sig[3] * ns; f[4] = sig[4] * ns; f[5] = sig[5] * ns;
                            f[6] = 2 * G * (1 - th); f[7] = 2 * G * thb;
                            epTrial_[o] += k * s0; epTrial_[o + 1] += k * s1; epTrial_[o + 2] += k * s2;
                            epTrial_[o + 3] += 2 * k * sig[3]; epTrial_[o + 4] += 2 * k * sig[4]; epTrial_[o + 5] += 2 * k * sig[5];
                            const double shrink = 1 - (3 * G * dg) / q;
                            sig[0] = p + s0 * shrink; sig[1] = p + s1 * shrink; sig[2] = p + s2 * shrink;
                            sig[3] *= shrink; sig[4] *= shrink; sig[5] *= shrink;
                            al += dg;
                            bits |= uint8_t(1u << g);
                        }
                        alphaTrial_[oa] = al;
                        localMax = std::max(localMax, al);
                        for (int i = 0; i < 6; i++) { gpStress_[o + i] = sig[i]; mean[i] += sig[i] / 8; }
                        // f += Bbar^T sigma / 8
                        for (int c = 0; c < 24; c++) {
                            double v = 0;
                            for (int i = 0; i < 6; i++) v += B[i * 24 + c] * sig[i];
                            fl[c] += v / 8;
                        }
                    }
                    yielding_[e] = bits;
                }
                if (large) {
                    double* gm = &geo_[64 * e];
                    std::fill(gm, gm + 64, 0.0);
                    for (int c = 0; c < 6; c++) {
                        const double sc = mean[c];
                        if (sc == 0) continue;
                        for (int i = 0; i < 64; i++) gm[i] += sc * T[c][i];
                    }
                }
                // rotate local element forces to global and add them (times rho)
                const double rho = L.rho[e];
                const double* R = &R_[9 * e];
                for (int a = 0; a < 8; a++) {
                    const int64_t n = 3 * (L.base[e] + L.off[a]);
                    const double x = fl[3 * a], y = fl[3 * a + 1], z = fl[3 * a + 2];
                    if (!large) {
                        out[n] += rho * x; out[n + 1] += rho * y; out[n + 2] += rho * z;
                    } else {
                        out[n] += rho * (R[0] * x + R[1] * y + R[2] * z);
                        out[n + 1] += rho * (R[3] * x + R[4] * y + R[5] * z);
                        out[n + 2] += rho * (R[6] * x + R[7] * y + R[8] * z);
                    }
                }
            }
            std::lock_guard<std::mutex> lock(mx);
            maxAlpha = std::max(maxAlpha, localMax);
        }, 256);
    for (int64_t i = 0; i < L.nDof; i++)
        if (L.fixed[i]) out[i] = 0;
    return maxAlpha;
}

void NonlinearModel::applyTangent(const double* x, double* y) const {
    std::fill(y, y + L.nDof, 0.0);
    const auto& K = fea_.K0;
    forElements([&](int64_t e) {
        double ue[24], ul[24], fl[24], eps[6], t[6];
        const int64_t n0 = L.base[e];
        const double* R = &R_[9 * e];
        for (int a = 0; a < 8; a++) {
            const int64_t n = 3 * (n0 + L.off[a]);
            ue[3 * a] = x[n]; ue[3 * a + 1] = x[n + 1]; ue[3 * a + 2] = x[n + 2];
        }
        if (large) {
            for (int a = 0; a < 8; a++) {
                const double vx = ue[3 * a], vy = ue[3 * a + 1], vz = ue[3 * a + 2];
                ul[3 * a] = R[0] * vx + R[3] * vy + R[6] * vz;
                ul[3 * a + 1] = R[1] * vx + R[4] * vy + R[7] * vz;
                ul[3 * a + 2] = R[2] * vx + R[5] * vy + R[8] * vz;
            }
        } else std::copy(ue, ue + 24, ul);
        for (int r = 0; r < 24; r++) {
            double s = 0;
            for (int c = 0; c < 24; c++) s += K[r * 24 + c] * ul[c];
            fl[r] = s;
        }
        if (large) {
            const double* g = &geo_[64 * e];
            for (int a = 0; a < 8; a++) {
                double sx = 0, sy = 0, sz = 0;
                for (int b = 0; b < 8; b++) {
                    const double v = g[a * 8 + b];
                    sx += v * ul[3 * b]; sy += v * ul[3 * b + 1]; sz += v * ul[3 * b + 2];
                }
                fl[3 * a] += sx; fl[3 * a + 1] += sy; fl[3 * a + 2] += sz;
            }
        }
        const uint8_t bits = plastic ? yielding_[e] : 0;
        if (bits) {
            for (int gp = 0; gp < 8; gp++) {
                if (!(bits & (1u << gp))) continue;
                const auto& B = fea_.gaussB[gp];
                const double* f = &flow_[64 * e + 8 * gp];
                std::fill(eps, eps + 6, 0.0);
                for (int c = 0; c < 24; c++) {
                    const double v = ul[c];
                    if (v != 0)
                        for (int i = 0; i < 6; i++) eps[i] += B[i * 24 + c] * v;
                }
                const double em = (eps[0] + eps[1] + eps[2]) / 3;
                const double ne = f[0] * eps[0] + f[1] * eps[1] + f[2] * eps[2] + f[3] * eps[3] + f[4] * eps[4] + f[5] * eps[5];
                const double c1 = f[6], c2 = f[7] * ne;
                // stress-like loss t = c1 dev(eps) + c2 n (tensor shear = engineering / 2)
                t[0] = c1 * (eps[0] - em) + c2 * f[0]; t[1] = c1 * (eps[1] - em) + c2 * f[1]; t[2] = c1 * (eps[2] - em) + c2 * f[2];
                t[3] = c1 * 0.5 * eps[3] + c2 * f[3]; t[4] = c1 * 0.5 * eps[4] + c2 * f[4]; t[5] = c1 * 0.5 * eps[5] + c2 * f[5];
                for (int c = 0; c < 24; c++) {
                    double v = 0;
                    for (int i = 0; i < 6; i++) v += B[i * 24 + c] * t[i];
                    fl[c] -= v / 8;
                }
            }
        }
        const double rho = L.rho[e];
        for (int a = 0; a < 8; a++) {
            const int64_t n = 3 * (n0 + L.off[a]);
            const double fx = fl[3 * a], fy = fl[3 * a + 1], fz = fl[3 * a + 2];
            if (!large) {
                y[n] += rho * fx; y[n + 1] += rho * fy; y[n + 2] += rho * fz;
            } else {
                y[n] += rho * (R[0] * fx + R[1] * fy + R[2] * fz);
                y[n + 1] += rho * (R[3] * fx + R[4] * fy + R[5] * fz);
                y[n + 2] += rho * (R[6] * fx + R[7] * fy + R[8] * fz);
            }
        }
    });
    for (int64_t i = 0; i < L.nDof; i++)
        if (L.fixed[i]) y[i] = 0;
}

void NonlinearModel::commit() {
    if (!plastic) return;
    ep_ = epTrial_;
    alpha = alphaTrial_;
    for (int64_t e = 0; e < nE_; e++) {
        if (everPlastic_[e]) continue;
        for (int g = 0; g < 8; g++)
            if (alpha[8 * e + g] > 0) { everPlastic_[e] = 1; break; }
    }
}

NodalFields NonlinearModel::nodalFields(const std::vector<double>& u) {
    NodalFields out;
    const int64_t nN = L.nNodes;
    out.vm.assign(nN, 0.f); out.p1.assign(nN, 0.f); out.p3.assign(nN, 0.f); out.pe.assign(nN, 0.f);
    std::vector<float> w(nN, 0.f);
    std::mutex mx;
    for (const auto& list : colours_)
        parallelFor(int64_t(list.size()), [&](int64_t lo, int64_t hi) {
            double ue[24], ul[24], sig[6], pr[3];
            double localMax = 0;
            for (int64_t t = lo; t < hi; t++) {
                const int64_t e = list[t];
                const double rho = L.rho[e];
                // voxels that have yielded or are close to it report their Gauss-point stresses, which never
                // leave the yield surface (corner extrapolation next to a plastic zone would overshoot it)
                bool nearYield = false;
                if (plastic && !everPlastic_[e])
                    for (int g = 0; g < 8 && !nearYield; g++) nearYield = vonMises6(&gpStress_[48 * e + 6 * g]) > 0.85 * plastic->yield;
                const bool yielded = plastic && (everPlastic_[e] || nearYield);
                if (!yielded) localDisplacement(e, u, ue, ul, false);
                double aMean = 0;
                if (plastic)
                    for (int g = 0; g < 8; g++) aMean += alpha[8 * e + g] / 8;
                for (int a = 0; a < 8; a++) {
                    if (yielded) {
                        std::copy(&gpStress_[48 * e + 6 * a], &gpStress_[48 * e + 6 * a] + 6, sig);
                    } else {
                        const auto& S = fea_.cornerStress[a];
                        for (int i = 0; i < 6; i++) {
                            double s = 0;
                            for (int c = 0; c < 24; c++) s += S[i * 24 + c] * ul[c];
                            sig[i] = s;
                        }
                    }
                    const double v = vonMises6(sig);
                    principalStresses(sig[0], sig[1], sig[2], sig[3], sig[4], sig[5], pr);
                    localMax = std::max(localMax, pr[0]);
                    const int64_t n = L.base[e] + L.off[a];
                    out.vm[n] += float(rho * v); out.p1[n] += float(rho * pr[0]); out.p3[n] += float(rho * pr[2]);
                    out.pe[n] += float(rho * aMean);
                    w[n] += float(rho);
                }
            }
            std::lock_guard<std::mutex> lock(mx);
            out.maxP1 = std::max(out.maxP1, localMax);
        }, 256);
    for (int64_t n = 0; n < nN; n++)
        if (w[n] > 0) { out.vm[n] /= w[n]; out.p1[n] /= w[n]; out.p3[n] /= w[n]; out.pe[n] /= w[n]; }
    return out;
}

// ---------- equilibrium and load ramp ----------

EquilibriumResult equilibrate(NonlinearModel& model, std::vector<double>& u, const std::vector<double>& fLam, const Precond& precond, double alphaLimit,
                              const std::function<bool(int, double)>& onIter, double tol, int maxNewton) {
    const int64_t n = int64_t(u.size());
    std::vector<double> fint(n), r(n);
    const auto& fixed = model.L.fixed;
    double fn = 0;
    for (int64_t i = 0; i < n; i++)
        if (!fixed[i]) fn += fLam[i] * fLam[i];
    fn = std::sqrt(fn);
    if (!(fn > 0)) fn = 1;
    EquilibriumResult res;
    double rel = INFINITY, prev = INFINITY;
    int grow = 0, it = 0;
    auto op = [&](const double* x, double* y) { model.applyTangent(x, y); };
    for (; it < maxNewton; it++) {
        res.maxAlpha = model.internalForce(u, fint);
        double rr = 0;
        for (int64_t i = 0; i < n; i++) {
            r[i] = fixed[i] ? 0 : fLam[i] - fint[i];
            rr += r[i] * r[i];
        }
        rel = std::sqrt(rr) / fn;
        if (onIter && onIter(it, rel)) throw std::runtime_error("Cancelled");
        if (!std::isfinite(rel)) break;
        if (rel <= tol) {
            res.converged = true;
            break;
        }
        // diverging: let the caller cut the increment (the first corrector may overshoot, so only
        // persistent growth, a residual that stays above the load, or absurd plastic strains count)
        if (it >= 2 && rel > 0.9 * prev) {
            if (++grow >= 3) break;
        } else grow = 0;
        if ((it >= 6 && rel > 1) || res.maxAlpha > alphaLimit) break;
        prev = rel;
        const auto sol = fcg(op, precond, r, 0.05, 80);
        for (int64_t i = 0; i < n; i++) u[i] += sol.x[i];
    }
    res.iterations = it;
    res.residual = rel;
    return res;
}

RampResult loadRamp(NonlinearModel& model, const std::vector<double>& f, const Precond& precond, const RampOptions& o) {
    const auto started = std::chrono::steady_clock::now();
    const int64_t n = int64_t(f.size());
    const auto& fixed = model.L.fixed;
    // generalized displacement along the loads, to watch the structure's stiffness
    double fAbs = 0;
    for (int64_t i = 0; i < n; i++)
        if (!fixed[i]) fAbs += std::abs(f[i]);
    auto along = [&](const std::vector<double>& u) {
        double w = 0;
        for (int64_t i = 0; i < n; i++)
            if (!fixed[i]) w += f[i] * u[i];
        return w / (fAbs > 0 ? fAbs : 1);
    };
    auto maxLength = [&](const std::vector<double>& u) {
        double d = 0;
        for (int64_t i = 0; i < n; i += 3) d = std::max(d, u[i] * u[i] + u[i + 1] * u[i + 1] + u[i + 2] * u[i + 2]);
        return std::sqrt(d);
    };
    const bool finiteTarget = std::isfinite(o.target);
    double k0 = 0, Dprev = 0, lamPrev = 0;
    RampResult out;
    out.u.assign(n, 0.0);
    double lam = 0;
    double dlam = finiteTarget ? o.target / o.steps : 1.0 / o.steps;
    const double minStep = 1e-3 * (finiteTarget ? o.target : 1);
    std::vector<double> fl(n);
    std::string reason = "reached";
    int count = 0;
    double lastAlpha = 0, lastDisp = 0;
    while (lam < o.target - 1e-12 && count < o.maxSteps) {
        const double lamTry = std::min(o.target, lam + dlam);
        // start from the last equilibrium (the first Newton step is the tangent predictor; a secant
        // extrapolation would stretch rotating voxels and start far from equilibrium)
        std::vector<double> uTry = out.u;
        for (int64_t i = 0; i < n; i++) fl[i] = lamTry * f[i];
        const double alphaLimit = std::min(std::isfinite(o.rupture) ? 2 * o.rupture : INFINITY, std::isfinite(o.stepAlpha) ? lastAlpha + 10 * o.stepAlpha : INFINITY);
        const auto res = equilibrate(model, uTry, fl, precond, alphaLimit, [&](int it, double rel) { return o.onIter && o.onIter(lamTry, it, rel); });
        // keep increments small enough to trace the load-displacement curve (and find the collapse load)
        const double dmaxTry = res.converged ? maxLength(uTry) : 0;
        // a converged increment is kept even when it yielded or moved more than wanted (re-solving it
        // in smaller pieces costs far more than it gains); the next increment is just made smaller
        const bool tooBig = res.converged && (res.maxAlpha - lastAlpha > o.stepAlpha || dmaxTry - lastDisp > o.stepDisp);
        if (!res.converged) {
            dlam *= 0.35;
            if (dlam < minStep) { reason = "collapse"; break; }
            continue;
        }
        model.commit();
        out.u = std::move(uTry);
        lam = lamTry;
        const double prevAlpha = lastAlpha, prevDisp = lastDisp;
        lastAlpha = res.maxAlpha;
        lastDisp = dmaxTry;
        count++;
        const double dmax = maxLength(out.u);
        auto fields = model.nodalFields(out.u);
        double smax = 0;
        for (float v : fields.vm) smax = std::max(smax, double(v));
        if (o.onStep) o.onStep(RampStep{lam, out.u, fields, res.iterations, res.maxAlpha, dmax, smax});
        if (res.maxAlpha >= o.rupture) { reason = "rupture"; break; }
        if (fields.maxP1 >= o.crackStress) { reason = "crack"; break; }
        if (dmax > o.maxDisp) { reason = "large deformation"; break; }
        if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() > o.maxTime) { reason = "time limit"; break; }
        // collapse: the load hardly rises any more while the part keeps deflecting (a mechanism has
        // formed; with strain hardening the load creeps up a little but the part has failed)
        const double D = along(out.u);
        if (count == 1 && D > 0) k0 = lam / D;
        else if (k0 > 0 && D > Dprev) {
            const double k = (lam - lamPrev) / (D - Dprev);
            if (k < 0.03 * k0) { reason = "collapse"; break; }
        }
        Dprev = D;
        lamPrev = lam;
        if (tooBig) {
            const double over = std::max((lastAlpha - prevAlpha) / o.stepAlpha, (lastDisp - prevDisp) / o.stepDisp);
            dlam = std::max(minStep, dlam / std::max(1.5, over));
        } else if (res.iterations <= 4) dlam *= 1.5;
        else if (res.iterations > 10) dlam *= 0.6;
    }
    if (count >= o.maxSteps && lam < o.target) reason = "step limit";
    out.lam = lam;
    out.reason = reason;
    return out;
}

}  // namespace ps
