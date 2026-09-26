// Validation of the studies beyond linear static against textbook results (mirrors test/studies.test.js).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <memory>
#include <string>

#include "check.hpp"
#include "fea/eigen.hpp"
#include "fea/explicit.hpp"
#include "fea/fatigue.hpp"
#include "fea/nonlinear.hpp"
#include "fea/response.hpp"
#include "fea/thermal.hpp"
#include "fea/topology.hpp"
#include "fea/voxel_fea.hpp"

using namespace ps;

static const double E = 200e9, RHO = 7850, NU = 0.3;

namespace {

// prismatic bar along x (length L, square side b, n voxels across), optionally clamped at x = 0
struct Bar {
    std::array<int, 3> dims;
    double h;
    int NX, NY, NZ, len;
    std::vector<uint8_t> bc;
    std::vector<float> density;
    int64_t node(int i, int j, int k) const { return i + int64_t(NX) * (j + int64_t(NY) * k); }
};

Bar bar(int n = 2, double b = 0.01, double L = 0.1, bool clamp = true) {
    Bar g;
    g.h = b / n;
    g.len = int(std::lround(L / g.h));
    g.dims = {g.len, n, n};
    g.NX = g.len + 1; g.NY = n + 1; g.NZ = n + 1;
    g.bc.assign(3 * size_t(g.NX) * g.NY * g.NZ, 0);
    if (clamp)
        for (int k = 0; k < g.NZ; k++)
            for (int j = 0; j < g.NY; j++)
                for (int d = 0; d < 3; d++) g.bc[3 * g.node(0, j, k) + d] = 1;
    g.density.assign(size_t(g.len) * n * n, 1.f);
    return g;
}

// uniform traction on the x = L face along dir (total F) with consistent nodal weights
std::vector<double> tipForce(const Bar& g, double F, int dir) {
    std::vector<double> f(g.bc.size(), 0.0);
    const int ny = g.dims[1], nz = g.dims[2];
    for (int k = 0; k <= nz; k++)
        for (int j = 0; j <= ny; j++) {
            const double w = (j == 0 || j == ny ? 0.5 : 1) * (k == 0 || k == nz ? 0.5 : 1);
            f[3 * g.node(g.len, j, k) + dir] = F * w / (ny * nz);
        }
    return f;
}

std::vector<double> physical(const std::vector<double>& u, double h) {
    std::vector<double> out(u.size());
    for (size_t i = 0; i < u.size(); i++) out[i] = u[i] / (E * h);
    return out;
}

}  // namespace

TEST("natural frequencies of a cantilever match Euler-Bernoulli beam theory") {
    const auto g = bar(2);
    VoxelFEA fea(g.dims, g.density, NU, g.bc);
    const auto r = naturalFrequencies(fea, 2, E, RHO, g.h, 0, cpuPreconditioner(fea));
    const double I = std::pow(0.01, 4) / 12, A = 1e-4;
    const double f1 = std::pow(1.8751, 2) / (2 * M_PI) * std::sqrt(E * I / (RHO * A * std::pow(0.1, 4)));
    std::printf("  modes %.1f, %.1f Hz vs %.1f Hz (twice: square section), %d iterations\n", r.freqs[0], r.freqs[1], f1, r.iterations);
    CHECK(r.converged);
    for (double f : r.freqs) CHECK(std::abs(f / f1 - 1) < 0.02);
}

TEST("free-floating part: rigid-body modes are separated from the first flexible mode") {
    const auto g = bar(2, 0.01, 0.1, false);
    VoxelFEA probe(g.dims, g.density, NU, g.bc, INT64_MAX);
    const auto M = lumpedMass(probe);
    const double shift = 1e-6 * maxEigenvalue(probe, M, 20);
    std::vector<double> add(M.size());
    for (size_t i = 0; i < M.size(); i++) add[i] = M[i] * shift;
    VoxelFEA fea(g.dims, g.density, NU, g.bc, 1100, &add);
    const auto r = naturalFrequencies(fea, 8, E, RHO, g.h, shift, cpuPreconditioner(fea));
    int rigid = 0;
    for (double l : r.lambdas) rigid += l < 1e-3 * shift;
    const double I = std::pow(0.01, 4) / 12, A = 1e-4;
    const double f1 = std::pow(4.7300, 2) / (2 * M_PI) * std::sqrt(E * I / (RHO * A * std::pow(0.1, 4)));
    std::printf("  %d rigid-body modes; first flexible %.0f Hz vs free-free theory %.0f Hz\n", rigid, r.freqs[6], f1);
    CHECK(rigid == 6);
    CHECK(std::abs(r.freqs[6] / f1 - 1) < 0.05);
}

TEST("buckling load of a clamped column matches Euler") {
    const auto g = bar(2, 0.01, 0.2);
    VoxelFEA fea(g.dims, g.density, NU, g.bc);
    const double F = 1000;
    SolveOptions o;
    o.tol = 1e-9;
    const auto u = physical(fea.solve(tipForce(g, -F, 0), o).u, g.h);
    const auto r = bucklingFactors(fea, elementStresses(fea, u, g.h), 1, cpuPreconditioner(fea));
    const double Pcr = M_PI * M_PI * E * (std::pow(0.01, 4) / 12) / (4 * 0.2 * 0.2);
    std::printf("  load factor %.3f vs %.3f (%d iterations)\n", r.factors[0], Pcr / F, r.iterations);
    CHECK(std::abs(r.factors[0] / (Pcr / F) - 1) < 0.03);
    // pulling the column never buckles it
    const auto t = physical(fea.solve(tipForce(g, F, 0), o).u, g.h);
    const auto r2 = bucklingFactors(fea, elementStresses(fea, t, g.h), 1, cpuPreconditioner(fea));
    CHECK(std::isinf(r2.factors[0]));
}

TEST("nonlinear: large deflection of a cantilever follows the elastica") {
    const auto g = bar(2, 0.01, 0.2);
    VoxelFEA fea(g.dims, g.density, NU, g.bc);
    const double P = E * (std::pow(0.01, 4) / 12) / (0.2 * 0.2);  // PL^2/EI = 1
    auto f = tipForce(g, -P, 1);
    for (auto& v : f) v /= E * g.h * g.h;
    NonlinearModel model(fea, true, std::nullopt);
    RampOptions o;
    o.target = 1;
    o.steps = 5;
    const auto r = loadRamp(model, f, [&](const double* x, double* z) { fea.precondition(x, z); }, o);
    double tip = 0;
    for (int k = 0; k <= 2; k++)
        for (int j = 0; j <= 2; j++) tip += r.u[3 * g.node(g.len, j, k) + 1];
    tip = -tip / 9 * g.h / 0.2;
    std::printf("  tip deflection %.4f L (elastica 0.3017 L, linear theory 0.333 L)\n", tip);
    CHECK(r.reason == "reached");
    CHECK(std::abs(tip - 0.3017) < 0.01);
}

TEST("nonlinear: plastic collapse near the limit load and a permanent bend after unloading") {
    const auto g = bar(4, 0.01, 0.2);
    VoxelFEA fea(g.dims, g.density, NU, g.bc);
    Material mat;
    mat.E = 200; mat.yield = 250; mat.uts = 300; mat.elongation = 0.3;
    const double Plim = 250e6 * std::pow(0.01, 3) / 4 / 0.2;
    auto f = tipForce(g, -Plim, 1);
    for (auto& v : f) v /= E * g.h * g.h;
    NonlinearModel model(fea, false, hardeningFor(mat));
    const Precond pre = [&](const double* x, double* z) { fea.precondition(x, z); };
    RampOptions o;
    o.target = RampOptions::inf;
    o.steps = 8;
    o.stepAlpha = 0.01;
    o.rupture = 0.3;
    o.maxDisp = 0.15 * g.len;
    const auto r = loadRamp(model, f, pre, o);
    std::printf("  stopped at %.3f x the plastic limit load (%s); first yield at 0.667\n", r.lam, r.reason.c_str());
    CHECK(r.reason == "collapse" || r.reason == "large deformation");
    CHECK(r.lam > 0.95 && r.lam < 1.3);
    auto u = r.u;
    const auto un = equilibrate(model, u, std::vector<double>(u.size(), 0.0), pre);
    double tip = 0;
    for (int k = 0; k <= 4; k++)
        for (int j = 0; j <= 4; j++) tip += u[3 * g.node(g.len, j, k) + 1];
    CHECK(un.converged);
    CHECK(tip < 0);
}

TEST("heat conduction: linear bar, cooling fin and transient energy balance") {
    const int n = 4, nx = 40, N = 41 * 25;
    const double b = 0.01, h = b / n;
    auto idx = [](int i, int j, int k) { return i + 41 * (j + 5 * k); };
    HeatInput in;
    in.dims = {nx, n, n};
    in.density.assign(size_t(nx) * n * n, 1.f);
    in.fixedNode.assign(N, 0);
    in.fixedValue.assign(N, 0.0);
    in.source.assign(N, 0.0);
    in.convH.assign(N, 0.0);
    in.convT.assign(N, 0.0);
    in.k = 50;
    in.h = h;
    for (int k = 0; k <= n; k++)
        for (int j = 0; j <= n; j++) {
            in.fixedNode[idx(0, j, k)] = in.fixedNode[idx(nx, j, k)] = 1;
            in.fixedValue[idx(0, j, k)] = 100;
        }
    const auto r = solveHeat(in);
    CHECK_NEAR(r.T[idx(20, 2, 2)], 50, 1e-6);
    CHECK_NEAR(heatFlux(*r.solver, r.T, 50, h)[idx(20, 2, 2)], 50000, 1);
    // fin: base 100 C, h = 25 W/m2K all round, ambient 20 C
    HeatInput fin = in;
    fin.fixedNode.assign(N, 0);
    fin.fixedValue.assign(N, 100.0);
    fin.convT.assign(N, 20.0);
    fin.k = 200;
    for (int k = 0; k <= n; k++)
        for (int j = 0; j <= n; j++) fin.fixedNode[idx(0, j, k)] = 1;
    auto face = [&](std::initializer_list<int> list) { for (int m : list) fin.convH[m] += 25 * h * h / 4; };
    for (int i = 0; i < nx; i++)
        for (int t = 0; t < n; t++) {
            face({idx(i, 0, t), idx(i + 1, 0, t), idx(i, 0, t + 1), idx(i + 1, 0, t + 1)});
            face({idx(i, n, t), idx(i + 1, n, t), idx(i, n, t + 1), idx(i + 1, n, t + 1)});
            face({idx(i, t, 0), idx(i + 1, t, 0), idx(i, t + 1, 0), idx(i + 1, t + 1, 0)});
            face({idx(i, t, n), idx(i + 1, t, n), idx(i, t + 1, n), idx(i + 1, t + 1, n)});
        }
    for (int j = 0; j < n; j++)
        for (int k = 0; k < n; k++) face({idx(nx, j, k), idx(nx, j + 1, k), idx(nx, j, k + 1), idx(nx, j + 1, k + 1)});
    const auto fr = solveHeat(fin);
    const double m = std::sqrt(25 * 4 * b / (200 * b * b)), hmk = 25 / (m * 200);
    const double tipTheory = 20 + 80 / (std::cosh(m * 0.1) + hmk * std::sinh(m * 0.1));
    double tip = 0;
    for (int k = 0; k <= n; k++)
        for (int j = 0; j <= n; j++) tip += fr.T[idx(nx, j, k)] / 25;
    std::printf("  fin tip %.3f C vs %.3f C\n", tip, tipTheory);
    CHECK(std::abs(tip - tipTheory) < 0.05);
    // insulated bar heated at one end: mean temperature rise = Q t / (rho cp V)
    HeatInput tr = in;
    tr.fixedNode.assign(N, 0);
    tr.fixedValue.assign(N, 0.0);
    tr.source.assign(N, 0.0);
    for (int k = 0; k <= n; k++)
        for (int j = 0; j <= n; j++) tr.source[idx(0, j, k)] = 10.0 / 25;
    tr.k = 167;
    tr.rhoCp = 2700 * 896;
    tr.duration = 60;
    tr.steps = 20;
    tr.initial = 20;
    const auto t = solveHeat(tr);
    double mean = 0, w = 0;
    for (int k = 0; k <= n; k++)
        for (int j = 0; j <= n; j++)
            for (int i = 0; i <= nx; i++) {
                const double c = (i == 0 || i == nx ? 0.5 : 1) * (j == 0 || j == n ? 0.5 : 1) * (k == 0 || k == n ? 0.5 : 1);
                mean += c * t.T[idx(i, j, k)];
                w += c;
            }
    CHECK_NEAR(mean / w, 20 + 600 / (tr.rhoCp * b * b * 0.1), 1e-6);
}

TEST("drop test: bar hitting the floor end-on matches 1D impact theory") {
    const int n = 4, ny = 40, NX = n + 1, NY = ny + 1, N = NX * NY * NX;
    const double b = 0.01, h = b / n;
    VoxelFEA fea({n, ny, n}, std::vector<float>(size_t(n) * ny * n, 1.f), NU, std::vector<uint8_t>(3 * N, 0), INT64_MAX);
    DropOptions o;
    o.E = E; o.rho = RHO; o.h = h;
    o.nodeY.assign(N, 0.0);
    o.surface.assign(N, 0);
    for (int k = 0; k <= n; k++)
        for (int j = 0; j <= ny; j++)
            for (int i = 0; i <= n; i++) {
                const int q = i + NX * (j + NY * k);
                o.nodeY[q] = j * h;
                o.surface[q] = i == 0 || j == 0 || k == 0 || i == n || j == ny || k == n;
            }
    o.speed = std::sqrt(2 * 9.81);
    const auto r = dropTestCPU(fea, o);
    const double c = std::sqrt(E / RHO);
    std::printf("  contact %.1f us vs 2L/c %.1f us; peak force %.0f N vs rho c v A %.0f N (%d steps)\n", r.contactTime * 1e6, 0.2 / c * 1e6, r.peakForce,
                RHO * c * o.speed * b * b, r.steps);
    CHECK(r.rebounded);
    CHECK(std::abs(r.contactTime / (0.2 / c) - 1) < 0.1);
    CHECK(std::abs(r.peakForce / (RHO * c * o.speed * b * b) - 1) < 0.12);
}

TEST("modal superposition: harmonic and step responses of one mode are exact") {
    const double w = 2 * M_PI * 100;
    ModalBasis basis;
    basis.omegas = {w};
    basis.gamma = {1};
    basis.modeU = {{1, 0, 0}};
    basis.modeS = {{1, 0, 0, 0, 0, 0}};
    basis.staticU = {float(1 / w / w), 0, 0};
    basis.staticS = {float(1 / w / w), 0, 0, 0, 0, 0};
    const double z = 0.02, st = 1 / w / w;
    for (double f : {0.0, 50.0, 100.0, 200.0}) {
        const double exact = st / std::hypot(1 - std::pow(f / 100, 2), 2 * z * f / 100);
        CHECK(std::abs(harmonicField(basis, 2 * M_PI * f, z).disp[0] / exact - 1) < 1e-3);
    }
    Excitation ex;
    ex.kind = "step";
    const auto hist = timeHistory(basis, excitation(ex), z, 0.05);
    double peak = 0;
    for (size_t k = 0; k < hist.times.size(); k++) peak = std::max(peak, double(shapeAt(basis, hist, k)[0]));
    CHECK(std::abs(peak / st - (1 + std::exp(-M_PI * z / std::sqrt(1 - z * z)))) < 2e-3);
}

TEST("fatigue: S-N curve, Goodman correction and life field") {
    Material steel;
    steel.uts = 420;
    steel.fatigue = {210, 1e6, true, true};
    const auto c = snCurve(steel, "polished");
    CHECK_NEAR(strengthAt(c, 1e3), 378, 1e-9);
    CHECK_NEAR(strengthAt(c, 1e7), 210, 1e-9);
    CHECK(std::isinf(cyclesToFailure(c, 200)));
    CHECK(std::abs(std::log10(cyclesToFailure(c, strengthAt(c, 3e4))) - std::log10(3e4)) < 1e-9);
    CHECK(std::abs(goodman(100, 100, 420) - 100 / (1 - 100.0 / 420)) < 1e-9);
    CHECK(snCurve(steel, "machined").Se < 210);
    const auto f = fatigueField({300e6f, 100e6f}, {300e6f, 100e6f}, {0, 0}, steel, "polished", -1, 1e6);
    CHECK(f.life[0] < 1e6 && std::isinf(f.life[1]));
    CHECK(f.worst == 0);
    CHECK(std::abs(f.fos[1] - 2.1) < 1e-6);
}

TEST("topology optimization keeps the volume budget, stiffens the design and meshes a closed surface") {
    const int nx = 30, ny = 10, nz = 2, NX = 31, NY = 11, N = NX * NY * 3;
    std::vector<uint8_t> bc(3 * N, 0);
    for (int k = 0; k <= nz; k++)
        for (int j = 0; j <= ny; j++)
            for (int d = 0; d < 3; d++) bc[3 * (NX * (j + NY * k)) + d] = 1;
    std::vector<double> f(3 * N, 0.0);
    for (int k = 0; k <= nz; k++) f[3 * (nx + NX * NY * k) + 1] = -1;
    TopologyOptions o;
    o.dims = {nx, ny, nz};
    o.fill.assign(size_t(nx) * ny * nz, 1.f);
    o.keep.assign(o.fill.size(), 0);
    for (int k = 0; k < nz; k++) {
        o.keep[nx - 1 + nx * ny * k] = 1;
        for (int j = 0; j < ny; j++) o.keep[nx * (j + ny * k)] = 1;
    }
    o.volFrac = 0.5;
    o.maxIter = 25;
    std::unique_ptr<VoxelFEA> fea;
    o.solve = [&](const std::vector<float>& density, const std::vector<double>* x0) {
        fea = std::make_unique<VoxelFEA>(o.dims, density, NU, bc);
        SolveOptions so;
        so.x0 = x0;
        so.tol = 1e-6;
        so.maxIter = 800;
        return TopologySolve{fea->solve(f, so).u, f, fea.get()};
    };
    const auto r = optimizeTopology(o);
    const auto& first = r.history.front();
    const auto& last = r.history.back();
    std::printf("  compliance %.1f -> %.1f, volume %.3f\n", first.compliance, last.compliance, last.volume);
    CHECK(std::abs(last.volume - 0.5) < 0.01);
    CHECK(last.compliance < 0.6 * first.compliance);
    const auto s = surfaceNets(r.density, o.dims, {0, 0, 0}, 1, 0.5);
    // closed, consistently oriented surface: every directed edge appears once and its reverse once
    std::map<std::pair<uint32_t, uint32_t>, int> edges;
    for (size_t t = 0; t < s.index.size(); t += 3)
        for (int e = 0; e < 3; e++) edges[{s.index[t + e], s.index[t + (e + 1) % 3]}]++;
    int bad = 0;
    for (const auto& [k, cnt] : edges) {
        auto rev = edges.find({k.second, k.first});
        if (cnt != 1 || rev == edges.end() || rev->second != 1) bad++;
    }
    CHECK(!s.index.empty());
    CHECK(double(bad) / edges.size() < 0.02);
    const auto stl = toSTL(s);
    uint32_t count;
    std::memcpy(&count, &stl[80], 4);
    CHECK(count == s.index.size() / 3);
}

TEST_MAIN
