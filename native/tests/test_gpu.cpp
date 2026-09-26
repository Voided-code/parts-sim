// GPU solver against the CPU solver (the same checks as the web app's GPU tests).
#include <algorithm>
#include <cmath>
#include <cstdio>

#include "check.hpp"
#include "fea/voxel_fea.hpp"
#include "gpu/gpu.hpp"
#include "gpu/gpu_fea.hpp"

using namespace ps;

TEST("GPU multigrid-CG matches the CPU solver") {
    if (!gpuAvailable()) {
        std::printf("    (skipped: %s)\n", gpuUnavailableReason().c_str());
        return;
    }
    std::printf("    adapter: %s\n", gpuName().c_str());
    for (int n : {4, 10}) {
        const double b = 0.01, Lm = 0.1, h = b / n;
        const int nx = int(std::lround(Lm / h)), ny = n, nz = n, NX = nx + 1, NY = ny + 1, NZ = nz + 1;
        std::vector<float> density(size_t(nx) * ny * nz, 1.f);
        std::vector<uint8_t> bc(3 * size_t(NX) * NY * NZ, 0);
        std::vector<double> f(bc.size(), 0.0);
        for (int k = 0; k < NZ; k++)
            for (int j = 0; j < NY; j++) {
                const size_t n0 = size_t(NX) * (j + size_t(NY) * k);
                bc[3 * n0] = bc[3 * n0 + 1] = bc[3 * n0 + 2] = 1;
                f[3 * (nx + n0) + 1] = -100.0 / (NY * NZ);
            }
        VoxelFEA cpu({nx, ny, nz}, density, 0.3, bc);
        SolveOptions o;
        o.tol = 1e-8;
        auto t0 = std::chrono::steady_clock::now();
        const auto a = cpu.solve(f, o);
        const double cms = pstest::ms(t0);
        VoxelFEA forGpu({nx, ny, nz}, density, 0.3, bc, GPU_COARSEST_DOF);
        t0 = std::chrono::steady_clock::now();
        GpuFeaSolver g(forGpu);
        o.maxIter = 3000;
        const auto r = g.solve(f, o);
        const double gms = pstest::ms(t0);
        double diff = 0, mx = 0;
        for (size_t i = 0; i < a.u.size(); i++) {
            diff = std::max(diff, std::abs(a.u[i] - r.u[i]));
            mx = std::max(mx, std::abs(a.u[i]));
        }
        std::printf("    %d voxels: GPU %d its %.0f ms (with upload), CPU %d its %.0f ms, max relative difference %.1e\n", nx * ny * nz, r.iterations, gms,
                    a.iterations, cms, diff / mx);
        CHECK(r.converged);
        CHECK(diff / mx < 1e-6);
    }
}

TEST_MAIN
