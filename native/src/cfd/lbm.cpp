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
    parallelFor(int64_t(nz) * ny, [&](int64_t lo, int64_t hi) {
        double fi[19], fe[19];
        for (int64_t row = lo; row < hi; row++) {
            const int y = int(row % ny), z = int(row / ny);
            const bool edge = y == 0 || z == 0 || y == ny - 1 || z == nz - 1;
            for (int x = 0; x < nx; x++) {
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
