// CPU solver of the v1 flow engine: the web app's src/cfd/flow-cpu.js (which explains the method),
// with the generated unrolled bulk kernel (flow_kernel.inc) and every pass split between threads
// (cells of a pass touch disjoint memory, so the results do not depend on the split).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../util/parallel.hpp"
#include "flow.hpp"

namespace ps::flow {

template <bool RR, bool Odd>
void bulkRun(float* __restrict F, int64_t N, int64_t c0, int64_t c1, const int64_t* o, float tau0, float smag);
#include "flow_kernel.inc"

namespace {

constexpr double CS2 = 1.0 / 3.0;

void equilibrium(double* g, double rho, double ux, double uy, double uz, bool rr) {
    const double usq = 1.5 * (ux * ux + uy * uy + uz * uz);
    double a0[6] = {0, 0, 0, 0, 0, 0};
    if (rr) {
        const double xxy = ux * ux * uy, yzz = uy * uz * uz, xzz = ux * uz * uz, xyy = ux * uy * uy, yyz = uy * uy * uz, xxz = ux * ux * uz;
        a0[0] = xxy + yzz; a0[1] = xzz + xyy; a0[2] = yyz + xxz; a0[3] = xxy - yzz; a0[4] = xzz - xyy; a0[5] = yyz - xxz;
    }
    for (int i = 0; i < 19; i++) {
        const double cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
        double t3 = 0;
        if (rr)
            for (int j = 0; j < 6; j++) t3 += P3C[i][j] * a0[j];
        g[i] = W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - usq + t3);
    }
}

/** g = feq + k W (H2 : pi / (2 cs^4) + recursive third order), pi = (xx, yy, zz, xy, xz, yz). */
void regularized(double* g, double rho, double ux, double uy, double uz, const double* pi, double k, bool rr) {
    equilibrium(g, rho, ux, uy, uz, rr);
    const double tr = CS2 * (pi[0] + pi[1] + pi[2]);
    double a1[6] = {0, 0, 0, 0, 0, 0};
    if (rr) {
        const double xxy = 2 * ux * pi[3] + uy * pi[0], yzz = 2 * uz * pi[5] + uy * pi[2], xzz = 2 * uz * pi[4] + ux * pi[2];
        const double xyy = 2 * uy * pi[3] + ux * pi[1], yyz = 2 * uy * pi[5] + uz * pi[1], xxz = 2 * ux * pi[4] + uz * pi[0];
        a1[0] = xxy + yzz; a1[1] = xzz + xyy; a1[2] = yyz + xxz; a1[3] = xxy - yzz; a1[4] = xzz - xyy; a1[5] = yyz - xxz;
    }
    for (int i = 0; i < 19; i++) {
        const int cx = CX[i], cy = CY[i], cz = CZ[i];
        const double h2 = cx * cx * pi[0] + cy * cy * pi[1] + cz * cz * pi[2] + 2 * (cx * cy * pi[3] + cx * cz * pi[4] + cy * cz * pi[5]) - tr;
        double t3 = 0;
        if (rr)
            for (int j = 0; j < 6; j++) t3 += P3C[i][j] * a1[j];
        g[i] += k * W[i] * (4.5 * h2 + t3);
    }
}

/** Non-equilibrium stress of f at rho, u (xx, yy, zz, xy, xz, yz). */
void nonEquilibrium(const double* f, double rho, double ux, double uy, double uz, double* pi) {
    double p[6] = {0, 0, 0, 0, 0, 0};
    for (int i = 0; i < 19; i++) {
        const int cx = CX[i], cy = CY[i], cz = CZ[i];
        p[0] += cx * cx * f[i]; p[1] += cy * cy * f[i]; p[2] += cz * cz * f[i];
        p[3] += cx * cy * f[i]; p[4] += cx * cz * f[i]; p[5] += cy * cz * f[i];
    }
    pi[0] = p[0] - rho * (ux * ux + CS2); pi[1] = p[1] - rho * (uy * uy + CS2); pi[2] = p[2] - rho * (uz * uz + CS2);
    pi[3] = p[3] - rho * ux * uy; pi[4] = p[4] - rho * ux * uz; pi[5] = p[5] - rho * uy * uz;
}

/** Collision (flow-cpu.js collide): post-collision g from f; returns rho, u in out. */
void collide(const double* f, double* g, double* out, double tau0, double smag, bool rr, double tauMin) {
    double rho = 0, jx = 0, jy = 0, jz = 0;
    for (int i = 0; i < 19; i++) { rho += f[i]; jx += CX[i] * f[i]; jy += CY[i] * f[i]; jz += CZ[i] * f[i]; }
    const double ux = jx / rho, uy = jy / rho, uz = jz / rho;
    double pi[6];
    nonEquilibrium(f, rho, ux, uy, uz, pi);
    const double q = std::sqrt(pi[0] * pi[0] + pi[1] * pi[1] + pi[2] * pi[2] + 2 * (pi[3] * pi[3] + pi[4] * pi[4] + pi[5] * pi[5]));
    const double tau = std::max(0.5 * (tau0 + std::sqrt(tau0 * tau0 + smag * q / rho)), tauMin);
    const double om = 1 / tau;
    out[0] = rho; out[1] = ux; out[2] = uy; out[3] = uz;
    if (!rr) {
        double e[19];
        equilibrium(e, rho, ux, uy, uz, false);
        for (int i = 0; i < 19; i++) g[i] = f[i] - om * (f[i] - e[i]);
        return;
    }
    regularized(g, rho, ux, uy, uz, pi, 1 - om, true);
}

class FlowCpu : public Solver {
public:
    FlowCpu(const Grid& g, const Params& p) : kind_(g.kind), rec_(g.rec) {
        dims_ = g.dims;
        nx_ = dims_[0]; ny_ = dims_[1]; nz_ = dims_[2];
        N_ = cells = g.N;
        periodic_ = g.periodicZ;
        if (!(p.nuLat > 0) || !(p.uLat >= 0 && p.uLat < 0.3)) throw std::invalid_argument("Invalid flow parameters.");
        uLat_ = p.uLat;
        nu0_ = p.nuLat;
        tau0_ = 3 * p.nuLat + 0.5;
        smag_ = 18 * std::sqrt(2.0) * p.smagorinsky * p.smagorinsky;
        rr_ = p.rr;
        wallModel_ = p.wallModel && p.rr;
        belt_ = p.belt;
        name = "CPU";
        F_.resize(19 * size_t(N_));
        const int64_t n = rec_.count;
        aux_.resize(19 * size_t(n));
        bb_.resize(19 * size_t(n));
        acc_.resize(4 * size_t(n));
        rhoSum_.resize(size_t(n));
        // neighbour offsets per layer: z = 0, inside, z = nz - 1 (the span wraps when periodic)
        for (int l = 0; l < 3; l++) {
            const int z = l == 0 ? 0 : l == 1 ? std::min(1, nz_ - 1) : nz_ - 1;
            for (int i = 0; i < 19; i++) {
                int dz = CZ[i];
                if (periodic_ && z + dz < 0) dz += nz_;
                if (periodic_ && z + dz >= nz_) dz -= nz_;
                off_[l][i] = CX[i] + int64_t(nx_) * (CY[i] + int64_t(ny_) * dz);
            }
        }
        // runs of bulk cells per row, and the faces
        runStart_.assign(size_t(ny_) * nz_ + 1, 0);
        for (int64_t row = 0; row < int64_t(ny_) * nz_; row++) {
            int x0 = -1;
            for (int x = 1; x < nx_; x++) {
                const bool bulk = x < nx_ - 1 && kind_[x + nx_ * row] == BULK;
                if (bulk && x0 < 0) x0 = x;
                if (!bulk && x0 >= 0) { runs_.push_back({x0, x}); x0 = -1; }
            }
            runStart_[row + 1] = int64_t(runs_.size());
        }
        for (int64_t c = 0; c < N_; c++) {
            if (kind_[c] != FACE) continue;
            Face f;
            f.cell = c;
            const int x = int(c % nx_), y = int((c / nx_) % ny_), z = int(c / (int64_t(nx_) * ny_));
            for (int i = 1; i < 19; i++) {
                const int xx = x + CX[i], yy = y + CY[i], zz = z + CZ[i];
                if (xx < 0 || xx >= nx_ || yy < 0 || yy >= ny_ || (!periodic_ && (zz < 0 || zz >= nz_))) f.out |= 1u << i;
            }
            f.layer = layerOf(z);
            if (x == 0) f.n0 = -1;
            else {
                const int z0 = periodic_ ? z : std::clamp(z, 1, nz_ - 2);
                const int64_t c0 = std::min(x, nx_ - 2) + int64_t(nx_) * (std::clamp(y, 1, ny_ - 2) + int64_t(ny_) * z0);
                f.n0 = kind_[c0] == BULK || kind_[c0] == WALL ? c0 : -2;
                f.layer0 = layerOf(z0);
            }
            faces_.push_back(f);
        }
        reset();
    }

    void reset() override {
        for (int i = 0; i < 19; i++) std::fill(F_.begin() + i * N_, F_.begin() + (i + 1) * N_, float(W[i]));
        for (int64_t r = 0; r < rec_.count; r++)
            for (int k = 0; k < 19; k++) { aux_[19 * r + k] = float(W[k]); bb_[19 * r + k] = float(W[k]); }
        std::fill(acc_.begin(), acc_.end(), 0.0);
        std::fill(rhoSum_.begin(), rhoSum_.end(), 0.0);
        steps = 0;
        accSteps_ = rhoSteps_ = 0;
        avg_.clear();
        samples_ = 0;
    }

    void step(int count) override {
        for (int s = 0; s < count; s++) {
            odd_ = int(steps & 1);
            uin_ = inletVelocity(uLat_, steps);
            uBelt_ = belt_ ? uin_ : 0;
            parallelFor(int64_t(ny_) * nz_, [&](int64_t a, int64_t b) { bulkRows(a, b); }, 8);
            if (rec_.count) parallelFor(rec_.count, [&](int64_t a, int64_t b) { wallCells(a, b); }, 256);
            parallelFor(int64_t(faces_.size()), [&](int64_t a, int64_t b) { faceCells(a, b); }, 2048);
            steps++;
            accSteps_++;
            if (sampleEvery > 0 && steps % sampleEvery == 0) sampleFields();
        }
    }

    Forces takeForces() override {
        Forces out;
        for (int64_t r = 0; r < rec_.count; r++) {
            const double fx = acc_[4 * r], fy = acc_[4 * r + 1], fz = acc_[4 * r + 2];
            out.me[0] += fx; out.me[1] += fy; out.me[2] += fz;
            const double nx = rec_.normal[3 * r], ny = rec_.normal[3 * r + 1], nz = rec_.normal[3 * r + 2];
            const double fn = fx * nx + fy * ny + fz * nz;
            out.friction[0] += fx - fn * nx; out.friction[1] += fy - fn * ny; out.friction[2] += fz - fn * nz;
            const double drho = acc_[4 * r + 3];
            rhoSum_[r] += drho;
            const uint32_t m = rec_.mask[r] & ~rec_.groundMask[r];
            for (int k = 1; k <= 6; k++) {
                if (!(m & (1u << k))) continue;
                out.pressure[0] -= CX[k] * drho * CS2; out.pressure[1] -= CY[k] * drho * CS2; out.pressure[2] -= CZ[k] * drho * CS2;
            }
        }
        std::fill(acc_.begin(), acc_.end(), 0.0);
        out.steps = accSteps_;
        rhoSteps_ += accSteps_;
        accSteps_ = 0;
        return out;
    }

    void resetAverages() override {
        std::fill(rhoSum_.begin(), rhoSum_.end(), 0.0);
        rhoSteps_ = 0;
        avg_.clear();
        samples_ = 0;
    }

    std::vector<float> surfaceRho() override {
        std::vector<float> out(size_t(rec_.count), 0.0f);
        if (rhoSteps_)
            for (int64_t r = 0; r < rec_.count; r++) out[r] = float(rhoSum_[r] / double(rhoSteps_));
        return out;
    }

    Fields fields() override {
        Fields f;
        f.factor = factor();
        f.dims = {(nx_ + f.factor - 1) / f.factor, (ny_ + f.factor - 1) / f.factor, (nz_ + f.factor - 1) / f.factor};
        f.inst = coarse(f.factor, int(steps & 1));
        if (!avg_.empty()) {
            f.avg.resize(avg_.size());
            for (size_t i = 0; i < avg_.size(); i++) f.avg[i] = float(avg_[i] / std::max(1, samples_));
            for (size_t c = 0; c < f.avg.size() / 4; c++)
                if (f.inst[4 * c] == -2) f.avg[4 * c] = -2;
        } else f.avg = f.inst;
        f.samples = samples_;
        return f;
    }

private:
    struct Face {
        int64_t cell = 0, n0 = 0;
        uint32_t out = 0;
        int layer = 1, layer0 = 1;
    };
    int layerOf(int z) const { return z == 0 ? 0 : z == nz_ - 1 ? 2 : 1; }
    // coarse view cells of at most about a million
    int factor() const { return std::max(1, int(std::ceil(std::cbrt(double(N_) / 1e6)))); }

    void neighbours(int64_t n, int z, int64_t* nb) const {
        const int64_t* o = off_[layerOf(z)];
        for (int i = 0; i < 19; i++) nb[i] = n + o[i];
    }

    // this step's loads, or those of a step of parity `parity`
    void load(int64_t n, const int64_t* nb, double* f, int parity = -1) const {
        const float* F = F_.data();
        const int64_t N = N_;
        const bool odd = parity < 0 ? odd_ != 0 : parity != 0;
        f[0] = F[n];
        for (int i = 1; i < 19; i += 2) {
            const int64_t m = nb[i];
            if (odd) { f[i] = F[i * N + n]; f[i + 1] = F[(i + 1) * N + m]; }
            else { f[i] = F[(i + 1) * N + n]; f[i + 1] = F[i * N + m]; }
        }
    }

    void store(int64_t n, const int64_t* nb, const double* g, uint32_t out = 0) {
        float* F = F_.data();
        const int64_t N = N_;
        F[n] = float(g[0]);
        for (int i = 1; i < 19; i += 2) {
            const int64_t m = nb[i];
            if (!(out & (1u << i))) F[(odd_ ? i + 1 : i) * N + m] = float(g[i]);
            F[(odd_ ? i : i + 1) * N + n] = float(g[i + 1]);
        }
    }

    /** Density and momentum of cell m from where its post-collision populations now live. */
    std::array<double, 4> storedMoments(int64_t m, int z) const {
        const float* F = F_.data();
        const int64_t N = N_;
        const int64_t* o = off_[layerOf(z)];
        double rho = F[m], jx = 0, jy = 0, jz = 0;
        for (int i = 1; i < 19; i += 2) {
            const double a = odd_ ? F[(i + 1) * N + m + o[i]] : F[i * N + m + o[i]];
            const double b = odd_ ? F[i * N + m] : F[(i + 1) * N + m];
            rho += a + b;
            jx += CX[i] * (a - b); jy += CY[i] * (a - b); jz += CZ[i] * (a - b);
        }
        return {rho, jx, jy, jz};
    }

    void bulkRows(int64_t r0, int64_t r1) {
        for (int64_t row = r0; row < r1; row++) {
            const int z = int(row / ny_);
            const int64_t* o = off_[layerOf(z)];
            const int64_t base = int64_t(nx_) * row;
            for (int64_t r = runStart_[row]; r < runStart_[row + 1]; r++) {
                const int64_t c0 = base + runs_[r].first, c1 = base + runs_[r].second;
                if (rr_) odd_ ? bulkRun<true, true>(F_.data(), N_, c0, c1, o, float(tau0_), float(smag_)) : bulkRun<true, false>(F_.data(), N_, c0, c1, o, float(tau0_), float(smag_));
                else odd_ ? bulkRun<false, true>(F_.data(), N_, c0, c1, o, float(tau0_), float(smag_)) : bulkRun<false, false>(F_.data(), N_, c0, c1, o, float(tau0_), float(smag_));
            }
        }
    }

    void wallCells(int64_t r0, int64_t r1) {
        double f[19], g[19], fb[19], fq[19], out[4];
        int64_t nb[19];
        for (int64_t r = r0; r < r1; r++) {
            const int64_t n = rec_.cell[r];
            const uint32_t mask = rec_.mask[r], gm = rec_.groundMask[r];
            const int z = int(n / (int64_t(nx_) * ny_));
            neighbours(n, z, nb);
            // the wall model on the part when it is on, and always on the moving ground (plain
            // bounce-back from a belt moving with the air leaves grid-scale waves undamped)
            if (rec_.samp[r] && (wallModel_ || rec_.onBelt[r]) && modelCell(r, n, z, nb)) continue;
            load(n, nb, f);
            // bounce-back: the cell's own outgoing populations toward its walls from the last step
            for (int k = 1; k < 19; k++)
                if (mask & (1u << k)) fb[k] = bb_[19 * r + k];
            for (int k = 1; k < 19; k++) {
                if (!(mask & (1u << k))) continue;
                const double v = fb[k];
                double fk = v;
                fq[k] = 6;
                if (gm & (1u << k)) fk = v + 6 * W[k] * CX[k] * uBelt_;
                else if (const int qb = rec_.q[18 * r + k - 1]) {
                    const double qq = (qb - 1) / 254.0;
                    if (qq < 0.5) fk = 2 * qq * v + (1 - 2 * qq) * f[OPP[k]];
                    else { fk = (0.5 / qq) * v + (1 - 0.5 / qq) * aux_[19 * r + k]; fq[k] = 3 / qq; }
                }
                f[k] = fk;
            }
            // momentum to the part: what left toward it minus what came back, c_opp = -c_k, less the
            // air at rest's (the reference pressure, as the wall model's surface integral has it: a
            // part with both kinds of wall cell must not feel the ambient pressure on one side only)
            double mx = 0, my = 0, mz = 0;
            for (int k = 1; k < 19; k++) {
                if (!(mask & (1u << k)) || (gm & (1u << k))) continue;
                const double e = fb[k] + f[k] - 2 * W[k];
                mx -= CX[k] * e; my -= CY[k] * e; mz -= CZ[k] * e;
            }
            collide(f, g, out, tau0_, smag_, rr_, 0);
            store(n, nb, g);
            for (int k = 1; k < 19; k++) {
                if (!(mask & (1u << k))) continue;
                aux_[19 * r + k] = float(g[k]);
                bb_[19 * r + k] = float(g[OPP[k]]);
            }
            acc_[4 * r] += mx; acc_[4 * r + 1] += my; acc_[4 * r + 2] += mz; acc_[4 * r + 3] += out[0] - 1;
        }
    }

    // wall model (flow-cpu.js wallModelCell)
    bool modelCell(int64_t r, int64_t n, int z, const int64_t* nb) {
        const int j = rec_.samp[r];
        const int zm = periodic_ ? (z + CZ[j] + nz_) % nz_ : z + CZ[j];
        const int64_t m = nb[j];
        const auto mo = storedMoments(m, zm);
        const double rho2 = mo[0];
        if (!(rho2 > 0)) return false;
        const double wx = rec_.onBelt[r] ? uBelt_ : 0;
        const double nx = rec_.normal[3 * r], ny = rec_.normal[3 * r + 1], nz = rec_.normal[3 * r + 2];
        const double ux = mo[1] / rho2 - wx, uy = mo[2] / rho2, uz = mo[3] / rho2;
        // the sample's velocity along the wall (not filtered in time: a delayed wall law lets the two
        // walls of a narrow gap drive each other into growing oscillations)
        const double un2 = ux * nx + uy * ny + uz * nz;
        const double tx = ux - un2 * nx, ty = uy - un2 * ny, tz = uz - un2 * nz;
        const double ut2 = std::sqrt(tx * tx + ty * ty + tz * tz);
        const double y1 = rec_.dist[r], y2 = rec_.y2[r];
        double utau = 0, ut1 = 0, dudn = 0, tauN = 0.5 + 3 * nu0_, ex = 0, ey = 0, ez = 0;
        if (ut2 > 1e-12) {
            utau = reichardtUtau(ut2, y2, nu0_);
            const auto r1 = reichardt(y1 * utau / nu0_);
            ut1 = utau * r1[0];
            dudn = utau * utau / nu0_ * r1[1];
            tauN = 0.5 + 3 * nu0_ / r1[1];
            ex = tx / ut2; ey = ty / ut2; ez = tz / ut2;
        }
        // the cell's velocity from the law of the wall, along the wall only (a velocity toward the wall
        // pumps pressure waves in narrow gaps)
        const double c = -rho2 * tauN * dudn / 3;
        const double pi[6] = {2 * c * ex * nx, 2 * c * ey * ny, 2 * c * ez * nz, c * (ex * ny + ey * nx), c * (ex * nz + ez * nx), c * (ey * nz + ez * ny)};
        double g[19];
        regularized(g, rho2, wx + ut1 * ex, ut1 * ey, ut1 * ez, pi, 1 - 1 / tauN, rr_);
        store(n, nb, g);
        const uint32_t mask = rec_.mask[r];
        for (int k = 1; k < 19; k++) {
            if (!(mask & (1u << k))) continue;
            aux_[19 * r + k] = float(g[k]);
            bb_[19 * r + k] = float(g[OPP[k]]);
        }
        const double ax = rec_.area[3 * r], ay = rec_.area[3 * r + 1], az = rec_.area[3 * r + 2];
        const double area = std::sqrt(ax * ax + ay * ay + az * az), p = (rho2 - 1) * CS2, tw = rho2 * utau * utau * area;
        acc_[4 * r] += -p * ax + tw * ex;
        acc_[4 * r + 1] += -p * ay + tw * ey;
        acc_[4 * r + 2] += -p * az + tw * ez;
        acc_[4 * r + 3] += rho2 - 1;
        return true;
    }

    void faceCells(int64_t i0, int64_t i1) {
        double gin[19], g[19];
        equilibrium(gin, 1, uin_, 0, 0, rr_);
        int64_t nb[19];
        for (int64_t j = i0; j < i1; j++) {
            const Face& fc = faces_[j];
            const double* src = gin;
            if (fc.n0 != -1) {
                double ux = uin_, uy = 0, uz = 0;
                if (fc.n0 >= 0) {
                    const int z0 = int(fc.n0 / (int64_t(nx_) * ny_));
                    const auto mo = storedMoments(fc.n0, z0);
                    ux = mo[1] / mo[0]; uy = mo[2] / mo[0]; uz = mo[3] / mo[0];
                }
                equilibrium(g, 1, ux, uy, uz, rr_);
                src = g;
            }
            const int64_t* o = off_[fc.layer];
            for (int i = 0; i < 19; i++) nb[i] = fc.cell + o[i];
            store(fc.cell, nb, src, fc.out);
        }
    }

    // the mean (rho - 1, u) of the bulk cells in coarse cells of cf^3 cells (-2: none), from the
    // loads of a step of parity `parity` (the next one)
    std::vector<float> coarse(int cf, int parity) const {
        const int cnx = (nx_ + cf - 1) / cf, cny = (ny_ + cf - 1) / cf, cnz = (nz_ + cf - 1) / cf;
        const int64_t nc = int64_t(cnx) * cny * cnz;
        std::vector<double> sum(4 * size_t(nc), 0.0);
        std::vector<int> count(size_t(nc), 0);
        double f[19];
        int64_t nb[19];
        // (one thread: the next step's loads)
        for (int z = 0; z < nz_; z++)
            for (int y = 0; y < ny_; y++)
                for (int x = 0; x < nx_; x++) {
                    const int64_t n = x + int64_t(nx_) * (y + int64_t(ny_) * z);
                    if (kind_[n] != BULK) continue;
                    neighbours(n, z, nb);
                    load(n, nb, f, parity);
                    double rho = 0, jx = 0, jy = 0, jz = 0;
                    for (int i = 0; i < 19; i++) { rho += f[i]; jx += CX[i] * f[i]; jy += CY[i] * f[i]; jz += CZ[i] * f[i]; }
                    const int64_t c = x / cf + int64_t(cnx) * (y / cf + int64_t(cny) * (z / cf));
                    sum[4 * c] += rho - 1; sum[4 * c + 1] += jx / rho; sum[4 * c + 2] += jy / rho; sum[4 * c + 3] += jz / rho;
                    count[c]++;
                }
        std::vector<float> out(4 * size_t(nc));
        for (int64_t c = 0; c < nc; c++) {
            if (!count[c]) { out[4 * c] = -2; continue; }
            for (int k = 0; k < 4; k++) out[4 * c + k] = float(sum[4 * c + k] / count[c]);
        }
        return out;
    }

    void sampleFields() {
        // the loads of the next step
        const auto inst = coarse(factor(), int(steps & 1));
        if (avg_.size() != inst.size()) avg_.assign(inst.size(), 0.0);
        for (size_t i = 0; i < inst.size(); i++) avg_[i] += inst[i];
        samples_++;
    }

    std::vector<uint8_t> kind_;
    Records rec_;
    std::array<int, 3> dims_{};
    int nx_ = 0, ny_ = 0, nz_ = 0;
    int64_t N_ = 0;
    bool periodic_ = false, rr_ = true, wallModel_ = true, belt_ = true;
    double uLat_ = 0.08, nu0_ = 1e-5, tau0_ = 0.5, smag_ = 0, uin_ = 0, uBelt_ = 0;
    int odd_ = 0;
    std::vector<float> F_, aux_, bb_;
    std::vector<double> acc_, rhoSum_, avg_;
    int64_t off_[3][19];
    std::vector<std::pair<int, int>> runs_;
    std::vector<int64_t> runStart_;
    std::vector<Face> faces_;
    int64_t accSteps_ = 0, rhoSteps_ = 0;
    int samples_ = 0;
};

}  // namespace

std::unique_ptr<Solver> makeCpu(const Grid& grid, const Params& p) { return std::make_unique<FlowCpu>(grid, p); }

}  // namespace ps::flow
