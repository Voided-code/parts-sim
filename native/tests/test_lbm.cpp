// Lattice-Boltzmann flow solver validation (mirrors test/lbm.test.js), plus GPU = CPU.
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "cfd/lbm.hpp"
#include "check.hpp"
#include "core/voxelize.hpp"
#include "gpu/gpu.hpp"

using namespace ps;

static LbmSetup emptyGrid(std::array<int, 3> dims) {
    LbmSetup s;
    s.dims = dims;
    s.solid.assign(size_t(dims[0]) * dims[1] * dims[2], 0);
    s.uLat = 0.08;
    s.nuLat = 0.002;
    return s;
}

TEST("reset reproduces the same inlet ramp and flow field") {
    LbmCpu sim(emptyGrid({9, 7, 6}));
    sim.step(15);
    const auto first = sim.macro();
    sim.reset();
    CHECK(sim.steps == 0);
    sim.step(15);
    CHECK(sim.macro() == first);
    CHECK_NEAR(sim.macro()[1], lbmInletVelocity(0.08, 14), 1e-7);
}

TEST("invalid solver inputs are rejected") {
    auto bad = [](auto change) {
        LbmSetup s = emptyGrid({3, 3, 3});
        change(s);
        try {
            LbmCpu sim(s);
        } catch (const std::exception&) {
            return true;
        }
        return false;
    };
    CHECK(bad([](LbmSetup& s) { s.dims = {2, 3, 3}; }));
    CHECK(bad([](LbmSetup& s) { s.solid.resize(1); }));
    CHECK(bad([](LbmSetup& s) { s.nuLat = NAN; }));
    CHECK(bad([](LbmSetup& s) { s.uLat = INFINITY; }));
}

TEST("pressure integration cancels uniform pressure and resolves a known face load") {
    const std::array<int, 3> dims{5, 5, 5};
    const int center = 2 + 5 * (2 + 5 * 2);
    std::vector<uint8_t> solid(125, 0);
    std::vector<float> rho(125, 1.01f);
    solid[center] = 1;
    const auto u = pressureForceCoefficients(rho, solid, dims, 0.08);
    CHECK(u.C[0] == 0 && u.C[1] == 0 && u.C[2] == 0);
    CHECK(u.frontal == 1);
    std::fill(rho.begin(), rho.end(), 1.f);
    rho[center - 1] = float(1 + 3 * 0.5 * 0.08 * 0.08 * 2);
    const auto l = pressureForceCoefficients(rho, solid, dims, 0.08);
    CHECK_NEAR(l.C[0], 2, 3e-5);
    CHECK(l.C[1] == 0 && l.C[2] == 0);
}

static LbmSetup cubeTunnel() {
    LbmSetup s = emptyGrid({72, 36, 36});
    const int nx = 72, ny = 36, sz = 8, x0 = 18, y0 = 14, z0 = 14;
    for (int z = z0; z < z0 + sz; z++)
        for (int y = y0; y < y0 + sz; y++)
            for (int x = x0; x < x0 + sz; x++) s.solid[x + nx * (y + ny * z)] = 1;
    return s;
}

static double cubeDrag(LbmSolver& sim, const LbmSetup& s, double* mlups, double* maxU) {
    const int64_t N = int64_t(s.dims[0]) * s.dims[1] * s.dims[2];
    std::vector<float> rhoAvg(N, 0.f);
    int samples = 0;
    const auto t0 = std::chrono::steady_clock::now();
    sim.step(900);
    for (int i = 900; i < 1400; i += 10) {
        sim.step(10);
        const auto m = sim.macro();
        for (int64_t c = 0; c < N; c++) rhoAvg[c] += m[4 * c];
        samples++;
    }
    const double ms = pstest::ms(t0);
    for (auto& r : rhoAvg) r /= samples;
    const auto m = sim.macro();
    *maxU = 0;
    for (int64_t c = 0; c < N; c++) *maxU = std::max(*maxU, std::sqrt(double(m[4 * c + 1]) * m[4 * c + 1] + double(m[4 * c + 2]) * m[4 * c + 2] + double(m[4 * c + 3]) * m[4 * c + 3]));
    *mlups = double(N) * 1400 / (ms * 1000);
    const auto fc = pressureForceCoefficients(rhoAvg, s.solid, s.dims, s.uLat);
    CHECK(fc.frontal == 64);
    CHECK(std::abs(fc.C[1] / fc.frontal) < 0.1);
    return fc.C[0] / fc.frontal;
}

TEST("flow past a cube stays stable and gives a bluff-body drag coefficient (CPU)") {
    const auto s = cubeTunnel();
    LbmCpu sim(s);
    double mlups, maxU;
    const double cd = cubeDrag(sim, s, &mlups, &maxU);
    std::printf("  CPU: %.0f MLUPS, max |u| %.2f U, Cd %.3f\n", mlups, maxU / s.uLat, cd);
    CHECK(maxU < 3 * s.uLat);
    CHECK(cd > 0.7 && cd < 2.0);
}

TEST("GPU flow solver matches the CPU solver") {
    if (!gpuAvailable()) { std::printf("  (no GPU: skipped)\n"); return; }
    const auto s = cubeTunnel();
    auto gpu = makeLbmGpu(s);
    double mlups, maxU;
    const double cd = cubeDrag(*gpu, s, &mlups, &maxU);
    LbmCpu cpu(s);
    double m2, u2;
    const double cdCpu = cubeDrag(cpu, s, &m2, &u2);
    std::printf("  GPU: %.0f MLUPS, Cd %.4f vs CPU %.4f\n", mlups, cd, cdCpu);
    CHECK(std::abs(cd - cdCpu) < 0.01 * std::abs(cdCpu));
}

TEST("wall links measure the true wall position along lattice links") {
    // box occupying x in [10.3, 14.3], y and z in [3, 7] on a unit grid
    const float x0 = 10.3f, x1 = 14.3f, y0 = 3, y1 = 7, z0 = 3, z1 = 7;
    std::vector<float> P = {x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0, x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1};
    std::vector<uint32_t> T = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5};
    Grid grid;
    grid.origin = {0, 0, 0};
    grid.h = 1;
    grid.dims = {20, 10, 10};
    const int N = 2000;
    const auto frac = voxelize(P, T, grid, 2);
    std::vector<uint8_t> solid(N);
    for (int c = 0; c < N; c++) solid[c] = frac[c] >= 0.5f;
    int count = 0;
    const auto links = wallLinks(P, T, {0, 0, 0}, 1, grid.dims, solid, &count);
    CHECK(count > 0);
    // fluid cell (9, 5, 5) sees the face x = 10.3 through its +x link, stored under direction 2 (c = -x)
    const int c = 9 + 20 * (5 + 10 * 5);
    CHECK(solid[c] == 0);
    CHECK(solid[c + 1] == 1);
    CHECK_NEAR((links[2 * N + c] - 1) / 254.0, 0.8, 0.01);
    // diagonal link (+x, +y) from the same cell also crosses the face at x = 10.3
    CHECK_NEAR((links[8 * N + c] - 1) / 254.0, 0.8, 0.01);
}

TEST_MAIN
