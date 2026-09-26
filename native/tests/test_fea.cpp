// Structural solver validation (mirrors test/fea.test.js).
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "check.hpp"
#include "fea/hex8.hpp"
#include "fea/voxel_fea.hpp"

using namespace ps;

TEST("element stiffness has rigid-body null space and is symmetric") {
    const auto K = hexElement(0.3).K;
    for (int d = 0; d < 3; d++)
        for (int r = 0; r < 24; r++) {
            double s = 0;
            for (int a = 0; a < 8; a++) s += K[r * 24 + 3 * a + d];
            CHECK(std::abs(s) < 1e-12);
        }
    for (int i = 0; i < 24; i++)
        for (int j = 0; j < 24; j++) CHECK(K[i * 24 + j] == K[j * 24 + i]);
}

TEST("Gauss-point strain matrices reproduce the condensed stiffness") {
    const auto el = hexElement(0.3);
    const auto D = elasticityMatrix(0.3);
    double err = 0, mx = 0;
    for (int i = 0; i < 24; i++)
        for (int j = 0; j < 24; j++) {
            double k = 0;
            for (const auto& B : el.gaussB)
                for (int p = 0; p < 6; p++)
                    for (int q = 0; q < 6; q++) k += B[p * 24 + i] * D[p * 6 + q] * B[q * 24 + j] / 8;
            err = std::max(err, std::abs(k - el.K[i * 24 + j]));
            mx = std::max(mx, std::abs(el.K[i * 24 + j]));
        }
    CHECK(err / mx < 1e-12);
}

TEST("principal stresses of a known tensor") {
    double out[3];
    principalStresses(50, -20, 10, 30, 0, 0, out);
    const double r = std::hypot(35.0, 30.0);
    CHECK_NEAR(out[0], 15 + r, 1e-9);
    CHECK_NEAR(out[1], 10, 1e-9);
    CHECK_NEAR(out[2], 15 - r, 1e-9);
}

struct Cantilever {
    double tip, expected, ms;
    SolveResult sol;
    NodalStresses st;
    std::array<double, 3> R;
    size_t levels;
    int nElem;
};

// length L along x, square section b, clamped at x = 0, tip load F in -y
static Cantilever cantilever(int n, double Lm = 0.1, double b = 0.01, double F = 100, double E = 200e9, double nu = 0.3) {
    const double h = b / n;
    const int nx = int(std::lround(Lm / h)), ny = n, nz = n, NX = nx + 1, NY = ny + 1, NZ = nz + 1;
    std::vector<float> density(size_t(nx) * ny * nz, 1.f);
    std::vector<uint8_t> bc(3 * size_t(NX) * NY * NZ, 0);
    std::vector<double> f(bc.size(), 0.0);
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++) {
            const size_t n0 = size_t(NX) * (j + size_t(NY) * k);
            bc[3 * n0] = bc[3 * n0 + 1] = bc[3 * n0 + 2] = 1;
            const double wj = (j == 0 || j == ny) ? 0.5 : 1, wk = (k == 0 || k == nz) ? 0.5 : 1;
            f[3 * (nx + n0) + 1] = -F * wj * wk / (ny * nz);
        }
    const auto t0 = std::chrono::steady_clock::now();
    VoxelFEA fea({nx, ny, nz}, density, nu, bc);
    SolveOptions o;
    o.tol = 1e-7;
    auto sol = fea.solve(f, o);
    const double ms = pstest::ms(t0);
    std::vector<double> u(sol.u.size());
    for (size_t i = 0; i < u.size(); i++) u[i] = sol.u[i] / (E * h);
    double tip = 0;
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++) tip += u[3 * (nx + size_t(NX) * (j + size_t(NY) * k)) + 1];
    tip /= NY * NZ;
    const double I = b * b * b * b / 12, G = E / (2 * (1 + nu));
    const double expected = -F * Lm * Lm * Lm / (3 * E * I) - F * Lm / ((5.0 / 6) * G * b * b);
    Cantilever c{tip, expected, ms, sol, fea.stresses(u, E, h), fea.reactions(sol.u), fea.levels.size(), nx * ny * nz};
    return c;
}

TEST("cantilever tip deflection matches Timoshenko beam theory") {
    for (int n : {2, 4, 8}) {
        const auto r = cantilever(n);
        const double err = std::abs(r.tip / r.expected - 1);
        std::printf("    n=%d (%d voxels, %zu MG levels): tip %.4f mm, theory %.4f mm, err %.2f%%, %d PCG its, %.0f ms\n", n, r.nElem,
                    r.levels, r.tip * 1e3, r.expected * 1e3, err * 100, r.sol.iterations, r.ms);
        CHECK(r.sol.converged);
        CHECK(err < 0.06);
        CHECK(std::abs(r.R[1] - 100) < 1e-3 * 100);
    }
}

TEST("cantilever surface bending stress matches M*c/I") {
    const int n = 8;
    const auto r = cantilever(n);
    const int nx = 80, NX = nx + 1, NY = n + 1;
    const size_t mid = nx / 2 + NX * (n + size_t(NY) * (n / 2));
    const double theoryMid = 100 * 0.05 * 0.005 / (0.01 * 1e-6 / 12);
    std::printf("    mid-span surface von Mises %.2f MPa, beam theory %.2f MPa\n", r.st.nodeVM[mid] / 1e6, theoryMid / 1e6);
    CHECK(std::abs(r.st.nodeVM[mid] / theoryMid - 1) < 0.05);
    double mx = 0;
    for (float v : r.st.nodeVM) mx = std::max(mx, double(v));
    CHECK(mx > 0.95 * 60e6 && mx < 1.4 * 60e6);
}

TEST("pruneFloating removes voxels not connected to supports") {
    std::vector<float> density = {1, 1, 0, 1};
    std::vector<uint8_t> held(5 * 2 * 2, 0);
    held[0] = 1;
    const auto r = pruneFloating({4, 1, 1}, density, held);
    CHECK(r.density[0] == 1 && r.density[1] == 1 && r.density[2] == 0 && r.density[3] == 0);
    CHECK(r.removed == 1);
}

TEST("support reaction includes loads applied directly to fixed nodes") {
    std::vector<uint8_t> bc(24, 1);
    VoxelFEA fea({1, 1, 1}, {1.f}, 0.3, bc);
    std::vector<double> f(24, 0.0);
    f[1] = -100;
    const auto sol = fea.solve(f);
    const auto R = fea.reactions(sol.u, &f);
    CHECK(R[0] == 0 && R[1] == 100 && R[2] == 0);
}

TEST("large cantilever: speed of the multithreaded multigrid solve") {
    // 200 x 20 x 20 voxels = 80,000 voxels, like a default-resolution run
    const auto r = cantilever(20, 0.1, 0.01);
    std::printf("    %d voxels: %d PCG iterations, %.0f ms total (setup + solve), tip error %.2f%%\n", r.nElem, r.sol.iterations, r.ms,
                std::abs(r.tip / r.expected - 1) * 100);
    CHECK(r.sol.converged);
}

TEST_MAIN
