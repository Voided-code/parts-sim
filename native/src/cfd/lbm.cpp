#include "lbm.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>

#include "../core/bvh.hpp"
#include "../util/parallel.hpp"

namespace ps {

const double LBM_W[19] = {1.0 / 3,  1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 36, 1.0 / 36, 1.0 / 36,
                          1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};

double lbmInletVelocity(double uLat, int64_t step) { return uLat * std::min(1.0, double(step + 1) / LBM_RAMP_STEPS); }

namespace {

void validate(const LbmSetup& s) {
    for (int n : s.dims)
        if (n < 3) throw std::invalid_argument("The flow grid needs three dimensions of at least 3 cells.");
    const int64_t cells = int64_t(s.dims[0]) * s.dims[1] * s.dims[2];
    if (int64_t(s.solid.size()) != cells) throw std::invalid_argument("Solid mask does not match the flow grid.");
    if (!s.links.empty() && int64_t(s.links.size()) != 19 * cells) throw std::invalid_argument("Wall links do not match the flow grid.");
    if (!std::isfinite(s.uLat) || s.uLat < 0 || s.uLat >= 0.3) throw std::invalid_argument("Lattice speed must be finite and between 0 and 0.3.");
    if (!std::isfinite(s.nuLat) || s.nuLat <= 0) throw std::invalid_argument("Lattice viscosity must be finite and positive.");
    if (!std::isfinite(s.smagorinsky) || s.smagorinsky < 0) throw std::invalid_argument("Smagorinsky constant must be finite and non-negative.");
}


// Cells away from walls and the domain boundary: the same pull-and-collide step without branches,
// in 32-bit and unrolled over the directions, so the compiler can vectorize it along x.
// src[i] = f_i shifted so that src[i][c] is the population arriving at c; dst[i] = g_i.
template <bool WriteMacro>
void collideRun(const float* const* src, float* const* dst, float* __restrict macro, int64_t c0, int64_t c1, float tau0, float smag) {
    const float w0 = float(LBM_W[0]), w1 = float(LBM_W[1]), w2 = float(LBM_W[7]);
    const float* __restrict s0 = src[0];
    const float* __restrict s1 = src[1];
    const float* __restrict s2 = src[2];
    const float* __restrict s3 = src[3];
    const float* __restrict s4 = src[4];
    const float* __restrict s5 = src[5];
    const float* __restrict s6 = src[6];
    const float* __restrict s7 = src[7];
    const float* __restrict s8 = src[8];
    const float* __restrict s9 = src[9];
    const float* __restrict s10 = src[10];
    const float* __restrict s11 = src[11];
    const float* __restrict s12 = src[12];
    const float* __restrict s13 = src[13];
    const float* __restrict s14 = src[14];
    const float* __restrict s15 = src[15];
    const float* __restrict s16 = src[16];
    const float* __restrict s17 = src[17];
    const float* __restrict s18 = src[18];
    float* __restrict t0 = dst[0];
    float* __restrict t1 = dst[1];
    float* __restrict t2 = dst[2];
    float* __restrict t3 = dst[3];
    float* __restrict t4 = dst[4];
    float* __restrict t5 = dst[5];
    float* __restrict t6 = dst[6];
    float* __restrict t7 = dst[7];
    float* __restrict t8 = dst[8];
    float* __restrict t9 = dst[9];
    float* __restrict t10 = dst[10];
    float* __restrict t11 = dst[11];
    float* __restrict t12 = dst[12];
    float* __restrict t13 = dst[13];
    float* __restrict t14 = dst[14];
    float* __restrict t15 = dst[15];
    float* __restrict t16 = dst[16];
    float* __restrict t17 = dst[17];
    float* __restrict t18 = dst[18];
#if defined(__clang__) && defined(__aarch64__)
#pragma clang loop vectorize(enable) interleave(disable)
#elif defined(__clang__)
    // x86 has 16 vector registers for the 19 populations: the vector code spills, and scalar code
    // (what MSVC made of it) runs faster
#pragma clang loop vectorize(disable)
#endif
    for (int64_t c = c0; c < c1; c++) {
        const float f0 = s0[c];
        const float f1 = s1[c];
        const float f2 = s2[c];
        const float f3 = s3[c];
        const float f4 = s4[c];
        const float f5 = s5[c];
        const float f6 = s6[c];
        const float f7 = s7[c];
        const float f8 = s8[c];
        const float f9 = s9[c];
        const float f10 = s10[c];
        const float f11 = s11[c];
        const float f12 = s12[c];
        const float f13 = s13[c];
        const float f14 = s14[c];
        const float f15 = s15[c];
        const float f16 = s16[c];
        const float f17 = s17[c];
        const float f18 = s18[c];
        const float rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18;
        const float inv = 1.0f / rho;
        const float ux = (f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14) * inv;
        const float uy = (f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18) * inv;
        const float uz = (f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18) * inv;
        const float usq = 1.5f * (ux * ux + uy * uy + uz * uz);
        const float e0 = w0 * rho * (1.0f - usq);
        const float cu1 = ux;
        const float e1 = w1 * rho * (1.0f + 3.0f * cu1 + 4.5f * cu1 * cu1 - usq);
        const float cu2 = -ux;
        const float e2 = w1 * rho * (1.0f + 3.0f * cu2 + 4.5f * cu2 * cu2 - usq);
        const float cu3 = uy;
        const float e3 = w1 * rho * (1.0f + 3.0f * cu3 + 4.5f * cu3 * cu3 - usq);
        const float cu4 = -uy;
        const float e4 = w1 * rho * (1.0f + 3.0f * cu4 + 4.5f * cu4 * cu4 - usq);
        const float cu5 = uz;
        const float e5 = w1 * rho * (1.0f + 3.0f * cu5 + 4.5f * cu5 * cu5 - usq);
        const float cu6 = -uz;
        const float e6 = w1 * rho * (1.0f + 3.0f * cu6 + 4.5f * cu6 * cu6 - usq);
        const float cu7 = ux + uy;
        const float e7 = w2 * rho * (1.0f + 3.0f * cu7 + 4.5f * cu7 * cu7 - usq);
        const float cu8 = -ux - uy;
        const float e8 = w2 * rho * (1.0f + 3.0f * cu8 + 4.5f * cu8 * cu8 - usq);
        const float cu9 = ux - uy;
        const float e9 = w2 * rho * (1.0f + 3.0f * cu9 + 4.5f * cu9 * cu9 - usq);
        const float cu10 = -ux + uy;
        const float e10 = w2 * rho * (1.0f + 3.0f * cu10 + 4.5f * cu10 * cu10 - usq);
        const float cu11 = ux + uz;
        const float e11 = w2 * rho * (1.0f + 3.0f * cu11 + 4.5f * cu11 * cu11 - usq);
        const float cu12 = -ux - uz;
        const float e12 = w2 * rho * (1.0f + 3.0f * cu12 + 4.5f * cu12 * cu12 - usq);
        const float cu13 = ux - uz;
        const float e13 = w2 * rho * (1.0f + 3.0f * cu13 + 4.5f * cu13 * cu13 - usq);
        const float cu14 = -ux + uz;
        const float e14 = w2 * rho * (1.0f + 3.0f * cu14 + 4.5f * cu14 * cu14 - usq);
        const float cu15 = uy + uz;
        const float e15 = w2 * rho * (1.0f + 3.0f * cu15 + 4.5f * cu15 * cu15 - usq);
        const float cu16 = -uy - uz;
        const float e16 = w2 * rho * (1.0f + 3.0f * cu16 + 4.5f * cu16 * cu16 - usq);
        const float cu17 = uy - uz;
        const float e17 = w2 * rho * (1.0f + 3.0f * cu17 + 4.5f * cu17 * cu17 - usq);
        const float cu18 = -uy + uz;
        const float e18 = w2 * rho * (1.0f + 3.0f * cu18 + 4.5f * cu18 * cu18 - usq);
        const float d0 = f0 - e0;
        const float d1 = f1 - e1;
        const float d2 = f2 - e2;
        const float d3 = f3 - e3;
        const float d4 = f4 - e4;
        const float d5 = f5 - e5;
        const float d6 = f6 - e6;
        const float d7 = f7 - e7;
        const float d8 = f8 - e8;
        const float d9 = f9 - e9;
        const float d10 = f10 - e10;
        const float d11 = f11 - e11;
        const float d12 = f12 - e12;
        const float d13 = f13 - e13;
        const float d14 = f14 - e14;
        const float d15 = f15 - e15;
        const float d16 = f16 - e16;
        const float d17 = f17 - e17;
        const float d18 = f18 - e18;
        const float pxx = d1 + d2 + d7 + d8 + d9 + d10 + d11 + d12 + d13 + d14;
        const float pyy = d3 + d4 + d7 + d8 + d9 + d10 + d15 + d16 + d17 + d18;
        const float pzz = d5 + d6 + d11 + d12 + d13 + d14 + d15 + d16 + d17 + d18;
        const float pxy = d7 + d8 - d9 - d10;
        const float pxz = d11 + d12 - d13 - d14;
        const float pyz = d15 + d16 - d17 - d18;
        const float q = std::sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0f * (pxy * pxy + pxz * pxz + pyz * pyz));
        const float om = 2.0f / (tau0 + std::sqrt(tau0 * tau0 + smag * q * inv));
        t0[c] = f0 - om * d0;
        t1[c] = f1 - om * d1;
        t2[c] = f2 - om * d2;
        t3[c] = f3 - om * d3;
        t4[c] = f4 - om * d4;
        t5[c] = f5 - om * d5;
        t6[c] = f6 - om * d6;
        t7[c] = f7 - om * d7;
        t8[c] = f8 - om * d8;
        t9[c] = f9 - om * d9;
        t10[c] = f10 - om * d10;
        t11[c] = f11 - om * d11;
        t12[c] = f12 - om * d12;
        t13[c] = f13 - om * d13;
        t14[c] = f14 - om * d14;
        t15[c] = f15 - om * d15;
        t16[c] = f16 - om * d16;
        t17[c] = f17 - om * d17;
        t18[c] = f18 - om * d18;
        if constexpr (WriteMacro) { macro[4 * c] = rho; macro[4 * c + 1] = ux; macro[4 * c + 2] = uy; macro[4 * c + 3] = uz; }
    }
}

}  // namespace

LbmCpu::LbmCpu(const LbmSetup& s) : dims_(s.dims), solid_(s.solid), links_(s.links) {
    validate(s);
    cells = int64_t(dims_[0]) * dims_[1] * dims_[2];
    uLat_ = s.uLat;
    tau0_ = 3 * s.nuLat + 0.5;
    smag_ = 18 * std::sqrt(2.0) * s.smagorinsky * s.smagorinsky;
    f_.resize(19 * cells);
    g_.resize(19 * cells);
    macro_.resize(4 * cells);
    const int64_t nx = dims_[0], ny = dims_[1];
    for (int i = 0; i < 19; i++) off_[i] = LBM_CX[i] + nx * (LBM_CY[i] + ny * LBM_CZ[i]);
    // per row, runs [x0, x1) of interior fluid cells with no solid neighbour (the fast path)
    const int nz = dims_[2];
    runStart_.assign(size_t(ny) * nz + 1, 0);
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++) {
            if (y > 0 && z > 0 && y < ny - 1 && z < nz - 1) {
                int x0 = -1;
                for (int x = 1; x <= nx - 1; x++) {
                    bool simple = x < nx - 1;
                    if (simple) {
                        const int64_t c = x + nx * (y + ny * int64_t(z));
                        simple = !solid_[c];
                        for (int i = 1; i < 19 && simple; i++) simple = !solid_[c - off_[i]];
                    }
                    if (simple && x0 < 0) x0 = x;
                    if (!simple && x0 >= 0) { runs_.push_back({x0, x}); x0 = -1; }
                }
            }
            runStart_[y + size_t(ny) * z + 1] = int64_t(runs_.size());
        }
    reset();
}

void LbmCpu::reset() {
    for (int i = 0; i < 19; i++) std::fill(f_.begin() + i * cells, f_.begin() + (i + 1) * cells, float(LBM_W[i]));
    for (int64_t c = 0; c < cells; c++) {
        macro_[4 * c] = 1;
        macro_[4 * c + 1] = macro_[4 * c + 2] = macro_[4 * c + 3] = 0;
    }
    steps = 0;
    macroFresh_ = true;
}

void LbmCpu::step(int count) {
    for (int s = 0; s < count; s++) stepOnce(s == count - 1);
}

std::vector<float> LbmCpu::macro() { return macro_; }

void LbmCpu::stepOnce(bool writeMacro) {
    const int nx = dims_[0], ny = dims_[1], nz = dims_[2];
    const int64_t N = cells;
    const float* f = f_.data();
    float* g = g_.data();
    const uint8_t* solid = solid_.data();
    const uint8_t* links = links_.empty() ? nullptr : links_.data();
    float* macro = macro_.data();
    const double uin = lbmInletVelocity(uLat_, steps);
    double feqIn[19];
    for (int i = 0; i < 19; i++) {
        const double cu = LBM_CX[i] * uin;
        feqIn[i] = LBM_W[i] * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * uin * uin);
    }
    const double tau0 = tau0_, smag = smag_;
    const float* src[19];
    float* dst[19];
    for (int i = 0; i < 19; i++) {
        src[i] = f + i * N - off_[i];
        dst[i] = g + i * N;
    }
    parallelFor(int64_t(nz) * ny, [&](int64_t lo, int64_t hi) {
        double fi[19], fe[19];
        for (int64_t row = lo; row < hi; row++) {
            const int y = int(row % ny), z = int(row / ny);
            const bool edge = y == 0 || z == 0 || y == ny - 1 || z == nz - 1;
            const int64_t rowStart = int64_t(nx) * row;
            int64_t r = runStart_[row], rEnd = runStart_[row + 1];
            for (int x = 0; x < nx; x++) {
                if (r < rEnd && x == runs_[r].first) {
                    if (writeMacro) collideRun<true>(src, dst, macro, rowStart + x, rowStart + runs_[r].second, float(tau0), float(smag));
                    else collideRun<false>(src, dst, macro, rowStart + x, rowStart + runs_[r].second, float(tau0), float(smag));
                    x = runs_[r].second - 1;
                    r++;
                    continue;
                }
                const int64_t c = x + int64_t(nx) * (y + int64_t(ny) * z);
                if (solid[c]) continue;
                if (x == 0) {
                    for (int i = 0; i < 19; i++) g[i * N + c] = float(feqIn[i]);
                    if (writeMacro) { macro[4 * c] = 1; macro[4 * c + 1] = float(uin); macro[4 * c + 2] = 0; macro[4 * c + 3] = 0; }
                    continue;
                }
                if (edge || x == nx - 1) {
                    // open boundary: velocity of the nearest interior cell (previous step), ambient pressure
                    const int64_t u0 = std::min(x, nx - 2) + int64_t(nx) * (std::clamp(y, 1, ny - 2) + int64_t(ny) * std::clamp(z, 1, nz - 2));
                    double ux = uin, uy = 0, uz = 0;
                    if (!solid[u0]) {
                        double r = 0, px = 0, py = 0, pz = 0;
                        for (int i = 0; i < 19; i++) {
                            const double v = f[i * N + u0];
                            r += v; px += LBM_CX[i] * v; py += LBM_CY[i] * v; pz += LBM_CZ[i] * v;
                        }
                        ux = px / r; uy = py / r; uz = pz / r;
                    }
                    const double usq = 1.5 * (ux * ux + uy * uy + uz * uz);
                    for (int i = 0; i < 19; i++) {
                        const double cu = LBM_CX[i] * ux + LBM_CY[i] * uy + LBM_CZ[i] * uz;
                        g[i * N + c] = float(LBM_W[i] * (1 + 3 * cu + 4.5 * cu * cu - usq));
                    }
                    if (writeMacro) { macro[4 * c] = 1; macro[4 * c + 1] = float(ux); macro[4 * c + 2] = float(uy); macro[4 * c + 3] = float(uz); }
                    continue;
                }
                double rho = 0, mx = 0, my = 0, mz = 0;
                for (int i = 0; i < 19; i++) {
                    const int64_t s = c - off_[i];
                    double v;
                    if (!solid[s]) v = f[i * N + s];
                    else {
                        // population that left toward the wall (direction j) comes back as direction i
                        const int j = LBM_OPP[i];
                        v = f[j * N + c];
                        const int qb = links ? links[i * N + c] : 0;
                        if (qb) {
                            const double q = (qb - 1) / 254.0;
                            if (q < 0.5) {
                                const int64_t n2 = c + off_[i];
                                if (!solid[n2]) v = 2 * q * v + (1 - 2 * q) * f[j * N + n2];
                            } else {
                                v = (0.5 / q) * v + (1 - 0.5 / q) * f[i * N + c];
                            }
                        }
                    }
                    fi[i] = v;
                    rho += v;
                    mx += LBM_CX[i] * v;
                    my += LBM_CY[i] * v;
                    mz += LBM_CZ[i] * v;
                }
                const double ux = mx / rho, uy = my / rho, uz = mz / rho;
                const double usq = 1.5 * (ux * ux + uy * uy + uz * uz);
                double pxx = 0, pyy = 0, pzz = 0, pxy = 0, pxz = 0, pyz = 0;
                for (int i = 0; i < 19; i++) {
                    const double cx = LBM_CX[i], cy = LBM_CY[i], cz = LBM_CZ[i];
                    const double cu = cx * ux + cy * uy + cz * uz;
                    const double e = LBM_W[i] * rho * (1 + 3 * cu + 4.5 * cu * cu - usq);
                    fe[i] = e;
                    const double d = fi[i] - e;
                    pxx += cx * cx * d; pyy += cy * cy * d; pzz += cz * cz * d;
                    pxy += cx * cy * d; pxz += cx * cz * d; pyz += cy * cz * d;
                }
                const double q = std::sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2 * (pxy * pxy + pxz * pxz + pyz * pyz));
                const double tau = 0.5 * (tau0 + std::sqrt(tau0 * tau0 + smag * q / rho));
                const double om = 1 / tau;
                for (int i = 0; i < 19; i++) g[i * N + c] = float(fi[i] - om * (fi[i] - fe[i]));
                if (writeMacro) { macro[4 * c] = float(rho); macro[4 * c + 1] = float(ux); macro[4 * c + 2] = float(uy); macro[4 * c + 3] = float(uz); }
            }
        }
    }, 4);
    std::swap(f_, g_);
    steps++;
    if (writeMacro)
        for (int64_t c = 0; c < N; c++)
            if (solid[c]) { macro[4 * c] = 1; macro[4 * c + 1] = macro[4 * c + 2] = macro[4 * c + 3] = 0; }
}

std::vector<uint8_t> wallLinks(const std::vector<float>& positions, const std::vector<uint32_t>& tris, const std::array<double, 3>& origin, double h,
                               const std::array<int, 3>& dims, const std::vector<uint8_t>& solid, int* count) {
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    const int64_t N = int64_t(nx) * ny * nz;
    std::vector<uint8_t> links(19 * N, 0);
    const BVH bvh(positions, tris);
    int64_t off[19];
    double dir[19][3], len[19];
    for (int i = 0; i < 19; i++) {
        off[i] = LBM_CX[i] + int64_t(nx) * (LBM_CY[i] + int64_t(ny) * LBM_CZ[i]);
        // toward the solid neighbour c - c_i, one link long (distance in units of |dir|)
        dir[i][0] = -LBM_CX[i] * h; dir[i][1] = -LBM_CY[i] * h; dir[i][2] = -LBM_CZ[i] * h;
        len[i] = 1;
    }
    std::atomic<int> total{0};
    parallelFor(int64_t(nz - 2) * (ny - 2), [&](int64_t lo, int64_t hi) {
        int local = 0;
        for (int64_t row = lo; row < hi; row++) {
            const int y = 1 + int(row % (ny - 2)), z = 1 + int(row / (ny - 2));
            for (int x = 1; x < nx - 1; x++) {
                const int64_t c = x + int64_t(nx) * (y + int64_t(ny) * z);
                if (solid[c]) continue;
                const double o[3] = {origin[0] + (x + 0.5) * h, origin[1] + (y + 0.5) * h, origin[2] + (z + 0.5) * h};
                for (int i = 1; i < 19; i++) {
                    if (!solid[c - off[i]]) continue;
                    RayHit hit;
                    if (!bvh.raycast(o, dir[i], 0, len[i], hit)) continue;
                    const double q = std::min(1.0, std::max(0.01, hit.distance / len[i]));
                    links[i * N + c] = uint8_t(1 + std::lround(254 * q));
                    local++;
                }
            }
        }
        total += local;
    }, 8);
    if (count) *count = total;
    return links;
}

ForceCoefficients pressureForceCoefficients(const std::vector<float>& rho, const std::vector<uint8_t>& solid, const std::array<int, 3>& dims, double uLat) {
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    const double inv = 1 / (3 * 0.5 * uLat * uLat);  // Cp = (rho - 1) c_s^2 / (0.5 u^2)
    ForceCoefficients out;
    const int64_t strides[3] = {1, nx, int64_t(nx) * ny};
    std::vector<uint8_t> shadow(size_t(ny) * nz, 0);
    for (int z = 1; z < nz - 1; z++)
        for (int y = 1; y < ny - 1; y++)
            for (int x = 1; x < nx - 1; x++) {
                const int64_t c = x + int64_t(nx) * (y + int64_t(ny) * z);
                if (solid[c]) { shadow[y + size_t(ny) * z] = 1; continue; }
                const double cp = (rho[c] - 1) * inv;
                for (int a = 0; a < 3; a++) {
                    if (solid[c + strides[a]]) out.C[a] += cp;
                    if (solid[c - strides[a]]) out.C[a] -= cp;
                }
            }
    for (uint8_t s : shadow) out.frontal += s;
    return out;
}

}  // namespace ps
