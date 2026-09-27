// Performance benchmark of the solver kernels at realistic sizes (not a test: run it by hand).
//   bench [filter]   runs the cases whose name contains filter
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "cfd/lbm.hpp"
#include "core/voxelize.hpp"
#include "fea/eigen.hpp"
#include "fea/explicit.hpp"
#include "fea/nonlinear.hpp"
#include "fea/thermal.hpp"
#include "fea/voxel_fea.hpp"
#include "gpu/gpu.hpp"
#include "gpu/gpu_fea.hpp"
#include "util/parallel.hpp"

using namespace ps;
using Clock = std::chrono::steady_clock;

static double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

// best of `reps` runs of fn, in seconds
static double best(int reps, const std::function<void()>& fn) {
    double b = 1e300;
    for (int r = 0; r < reps; r++) {
        const auto t0 = Clock::now();
        fn();
        b = std::min(b, since(t0));
    }
    return b;
}

// cantilever block (full density), clamped at x = 0, tip load along -y
struct Beam {
    std::array<int, 3> dims;
    std::vector<float> density;
    std::vector<uint8_t> bc;
    std::vector<double> f;
};

static Beam cantilever(int nx, int ny, int nz) {
    Beam b;
    b.dims = {nx, ny, nz};
    const int NX = nx + 1, NY = ny + 1, NZ = nz + 1;
    b.density.assign(size_t(nx) * ny * nz, 1.f);
    b.bc.assign(3 * size_t(NX) * NY * NZ, 0);
    b.f.assign(b.bc.size(), 0.0);
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++) {
            const size_t n0 = size_t(NX) * (j + size_t(NY) * k);
            b.bc[3 * n0] = b.bc[3 * n0 + 1] = b.bc[3 * n0 + 2] = 1;
            b.f[3 * (nx + n0) + 1] = -1.0 / (NY * NZ);
        }
    return b;
}

static void fea(const char* name, int nx, int ny, int nz, bool gpu) {
    auto b = cantilever(nx, ny, nz);
    const int voxels = nx * ny * nz;
    auto t0 = Clock::now();
    VoxelFEA m(b.dims, b.density, 0.3, b.bc, gpu ? GPU_COARSEST_DOF : 1100);
    const double build = since(t0);
    const int64_t n = m.nDof();
    std::vector<double> x(n, 1e-3), y(n);
    const double tApply = best(5, [&] { m.apply(0, x.data(), y.data()); });
    const double tPre = best(3, [&] { m.precondition(b.f.data(), y.data()); });
    SolveOptions o;
    o.tol = 1e-6;
    SolveResult s;
    double tSolve;
    if (gpu) {
        GpuFeaSolver g(m);
        if (const char* pr = std::getenv("BENCH_PROFILE")) std::fputs(g.profile(std::atoi(pr)).c_str(), stdout);
        o.maxIter = 3000;
        tSolve = best(2, [&] { s = g.solve(b.f, o); });
    } else tSolve = best(2, [&] { s = m.solve(b.f, o); });
    // 8 elements x 3 rows x 24 multiply-adds per node
    const double flops = 2.0 * 576 * double(voxels);
    std::printf("%-22s %7d voxels  build %6.0f ms  matvec %6.2f ms (%5.1f GFLOP/s)  V-cycle %6.2f ms  solve %7.0f ms (%d it, %s)\n", name, voxels,
                1e3 * build, 1e3 * tApply, flops / tApply / 1e9, 1e3 * tPre, 1e3 * tSolve, s.iterations, s.converged ? "ok" : "NOT CONVERGED");
}

static LbmSetup tunnel(int nx, int ny, int nz) {
    LbmSetup s;
    s.dims = {nx, ny, nz};
    s.solid.assign(size_t(nx) * ny * nz, 0);
    const int sz = ny / 4, x0 = nx / 4, y0 = (ny - sz) / 2, z0 = (nz - sz) / 2;
    for (int z = z0; z < z0 + sz; z++)
        for (int y = y0; y < y0 + sz; y++)
            for (int x = x0; x < x0 + sz; x++) s.solid[x + size_t(nx) * (y + size_t(ny) * z)] = 1;
    s.uLat = 0.08;
    s.nuLat = 0.002;
    return s;
}

static void lbm(const char* name, int nx, int ny, int nz, bool gpu, int steps, bool half = true) {
    auto s = tunnel(nx, ny, nz);
    s.gpuHalf = half;
    const double cells = double(nx) * ny * nz;
    auto t0 = Clock::now();
    std::unique_ptr<LbmSolver> sim = gpu ? makeLbmGpu(s) : std::make_unique<LbmCpu>(s);
    const double setup = since(t0);
    sim->step(10);  // warm up
    t0 = Clock::now();
    sim->step(steps);
    const double t = since(t0);
    t0 = Clock::now();
    const auto m = sim->macro();
    const double read = since(t0);
    // pull scheme: 19 populations read + 19 written per cell and step
    const double bytes = 38.0 * (gpu && half && GpuContext::get()->shaderF16 ? 2 : 4);
    std::printf("%-22s %7.2f M cells  setup %6.0f ms  %6.1f MLUPS  (%5.1f GB/s)  read-back %5.0f ms\n", name, cells / 1e6, 1e3 * setup,
                cells * steps / t / 1e6, cells * steps * bytes / t / 1e9, 1e3 * read);
}

static void eigen(const char* name, int nx, int ny, int nz) {
    auto b = cantilever(nx, ny, nz);
    VoxelFEA m(b.dims, b.density, 0.3, b.bc);
    const double h = 1e-3;
    auto t0 = Clock::now();
    const auto r = naturalFrequencies(m, 5, 200e9, 7850, h, 0, cpuPreconditioner(m));
    const double tModal = since(t0);
    // compression buckling: push the tip along -x
    std::vector<double> f(b.f.size(), 0.0);
    const int NX = nx + 1, NY = ny + 1, NZ = nz + 1;
    for (int k = 0; k < NZ; k++)
        for (int j = 0; j < NY; j++) f[3 * (nx + size_t(NX) * (j + size_t(NY) * k))] = -1.0 / (NY * NZ);
    SolveOptions o;
    o.tol = 1e-8;
    auto sol = m.solve(f, o);
    for (auto& v : sol.u) v /= 200e9 * h;
    t0 = Clock::now();
    const auto bk = bucklingFactors(m, elementStresses(m, sol.u, h), 3, cpuPreconditioner(m), {}, 1e-5, 1e9);
    const double tBuck = since(t0);
    std::printf("%-22s %7d voxels  modal (5) %6.0f ms, %d it   buckling (3) %6.0f ms, %d it\n", name, nx * ny * nz, 1e3 * tModal, r.iterations,
                1e3 * tBuck, bk.iterations);
}

static void thermal(const char* name, int nx, int ny, int nz) {
    const int64_t N = int64_t(nx + 1) * (ny + 1) * (nz + 1);
    HeatInput in;
    in.dims = {nx, ny, nz};
    in.density.assign(size_t(nx) * ny * nz, 1.f);
    in.fixedNode.assign(N, 0);
    in.fixedValue.assign(N, 20.0);
    in.source.assign(N, 0.0);
    in.convH.assign(N, 1e-4);
    in.convT.assign(N, 20.0);
    for (int k = 0; k <= nz; k++)
        for (int j = 0; j <= ny; j++) in.source[size_t(nx + 1) * (j + size_t(ny + 1) * k)] = 1.0;
    in.k = 50;
    in.h = 1e-3;
    auto t0 = Clock::now();
    const auto r = solveHeat(in);
    const double tSteady = since(t0);
    in.duration = 10;
    in.steps = 50;
    in.rhoCp = 3.5e6;
    in.initial = 20;
    t0 = Clock::now();
    const auto rt = solveHeat(in, {}, [](const ScalarVoxelSolver& sv) {
        if (std::getenv("BENCH_PROFILE")) std::fputs(const_cast<ScalarVoxelSolver&>(sv).profile().c_str(), stdout);
    });
    std::printf("%-22s %7d voxels  steady solve %6.0f ms (%d it)  transient 50 steps %6.0f ms (%d it)\n", name, nx * ny * nz, 1e3 * tSteady, r.iterations,
                1e3 * since(t0), rt.iterations);
}

static void nonlinearCase(const char* name, int nx, int ny, int nz) {
    auto b = cantilever(nx, ny, nz);
    VoxelFEA m(b.dims, b.density, 0.3, b.bc);
    for (auto& v : b.f) v *= 2e-3;  // large deflection
    NonlinearModel model(m, true, std::nullopt);
    RampOptions o;
    o.steps = 5;
    int newton = 0;
    o.onIter = [&](double, int, double) { newton++; return false; };
    auto t0 = Clock::now();
    const auto r = loadRamp(model, b.f, [&](const double* x, double* z) { m.precondition(x, z); }, o);
    std::printf("%-22s %7d voxels  load ramp %6.0f ms (%d Newton, %s)\n", name, nx * ny * nz, 1e3 * since(t0), newton, r.reason.c_str());
}

static void dropCase(const char* name, int n) {
    const int nx = n, ny = 10 * n, nz = n, NX = nx + 1, NY = ny + 1, NZ = nz + 1;
    const int64_t N = int64_t(NX) * NY * NZ;
    const double h = 0.01 / n;
    VoxelFEA m({nx, ny, nz}, std::vector<float>(size_t(nx) * ny * nz, 1.f), 0.3, std::vector<uint8_t>(3 * N, 0), INT64_MAX);
    DropOptions o;
    o.E = 200e9; o.rho = 7850; o.h = h; o.speed = 4.43;
    o.nodeY.assign(N, 0.0);
    o.surface.assign(N, 0);
    for (int k = 0; k <= nz; k++)
        for (int j = 0; j <= ny; j++)
            for (int i = 0; i <= nx; i++) {
                const int64_t q = i + NX * (j + int64_t(NY) * k);
                o.nodeY[q] = j * h;
                o.surface[q] = i == 0 || j == 0 || k == 0 || i == nx || j == ny || k == nz;
            }
    auto t0 = Clock::now();
    const auto r = dropTestCPU(m, o);
    const double t = since(t0);
    std::vector<double> u(m.nDof(), 1e-9), y(m.nDof());
    const double tApply = best(5, [&] { m.apply(0, u.data(), y.data(), true); });
    const double tStress = best(3, [&] { m.stresses(u, o.E, o.h, false); });
    std::printf("%-22s %7d voxels  %6.0f ms (%d steps, %.2f ms/step; matvec %.2f ms, stresses %.2f ms)\n", name, nx * ny * nz, 1e3 * t, r.steps,
                1e3 * t / r.steps, 1e3 * tApply, 1e3 * tStress);
}

static void voxelizeCase(const char* name, int res) {
    // UV sphere of ~80k triangles
    const int seg = 200, rings = 200;
    std::vector<float> pos;
    std::vector<uint32_t> idx;
    for (int r = 0; r <= rings; r++)
        for (int s = 0; s <= seg; s++) {
            const double th = M_PI * r / rings, ph = 2 * M_PI * s / seg;
            pos.insert(pos.end(), {float(std::sin(th) * std::cos(ph)), float(std::cos(th)), float(std::sin(th) * std::sin(ph))});
        }
    for (int r = 0; r < rings; r++)
        for (int s = 0; s < seg; s++) {
            const uint32_t a = r * (seg + 1) + s, b2 = a + seg + 1;
            idx.insert(idx.end(), {a, b2, a + 1, a + 1, b2, b2 + 1});
        }
    const Grid g = gridForBox({-1, -1, -1}, {1, 1, 1}, res, 1);
    auto t0 = Clock::now();
    const auto frac = voxelize(pos, idx, g, 2);
    const double tv = since(t0);
    std::vector<uint8_t> solid(frac.size());
    for (size_t c = 0; c < frac.size(); c++) solid[c] = frac[c] >= 0.5f;
    t0 = Clock::now();
    int links = 0;
    wallLinks(pos, idx, {g.origin[0], g.origin[1], g.origin[2]}, g.h, g.dims, solid, &links);
    const double tl = since(t0);
    std::printf("%-22s %7.2f M cells  voxelize %6.0f ms   wall links %6.0f ms (%d links)\n", name, frac.size() / 1e6, 1e3 * tv, 1e3 * tl, links);
}

int main(int argc, char** argv) {
    const std::string only = argc > 1 ? argv[1] : "";
    auto want = [&](const char* n) { return only.empty() || std::string(n).find(only) != std::string::npos; };
    std::printf("%u threads, GPU: %s\n", ThreadPool::instance().size(), gpuAvailable() ? gpuName().c_str() : "none");
    if (want("fea-80k")) fea("fea-80k cpu", 320, 16, 16, false);
    if (want("fea-280k")) fea("fea-280k cpu", 480, 24, 24, false);
    if (want("fea-20k-gpu") && gpuAvailable()) fea("fea-20k-gpu", 200, 10, 10, true);
    if (want("fea-20k cpu")) fea("fea-20k cpu", 200, 10, 10, false);
    if (want("fea-80k-gpu") && gpuAvailable()) fea("fea-80k-gpu", 320, 16, 16, true);
    if (want("fea-280k-gpu") && gpuAvailable()) fea("fea-280k-gpu", 480, 24, 24, true);
    if (want("eigen")) eigen("eigen-40k", 250, 16, 10);
    if (want("thermal")) thermal("thermal-250k", 250, 40, 25);
    if (want("nonlinear")) nonlinearCase("nonlinear-10k", 160, 8, 8);
    if (want("drop")) dropCase("drop-41k", 16);
    if (want("lbm-cpu")) lbm("lbm-cpu-1M", 200, 72, 72, false, 60);
    if (want("lbm-gpu") && gpuAvailable()) {
        lbm("lbm-gpu-4M", 320, 112, 112, true, 200);
        lbm("lbm-gpu-12M", 460, 160, 160, true, 100);
        lbm("lbm-gpu-12M-fp32", 460, 160, 160, true, 100, false);
    }
    if (want("voxelize")) voxelizeCase("voxelize-8M", 200);
}
