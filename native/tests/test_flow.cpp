// The v1 flow engine (cfd/flow*.cpp; mirrors test/flow.test.js): wall records, thin walls, the law
// of the wall, GPU = CPU, and a coarse drag regression.
#include <cmath>
#include <cstdio>

#include "cfd/flow.hpp"
#include "check.hpp"
#include "gpu/gpu.hpp"

using namespace ps;
using namespace ps::flow;

// a closed box mesh [x0, x1] x [y0, y1] x [z0, z1] (outward triangles)
static void box(std::vector<float>& P, std::vector<uint32_t>& T, double x0, double x1, double y0, double y1, double z0, double z1) {
    const uint32_t b = uint32_t(P.size() / 3);
    const double v[8][3] = {{x0, y0, z0}, {x1, y0, z0}, {x1, y1, z0}, {x0, y1, z0}, {x0, y0, z1}, {x1, y0, z1}, {x1, y1, z1}, {x0, y1, z1}};
    for (auto& p : v)
        for (double c : p) P.push_back(float(c));
    const uint32_t t[36] = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3};
    for (uint32_t i : t) T.push_back(b + i);
}

// a UV sphere of radius r at c
static void sphere(std::vector<float>& P, std::vector<uint32_t>& T, double r, const double c[3], int seg = 48, int rings = 32) {
    for (int i = 0; i <= rings; i++)
        for (int j = 0; j <= seg; j++) {
            const double th = M_PI * i / rings, ph = 2 * M_PI * j / seg;
            P.insert(P.end(), {float(c[0] + r * std::sin(th) * std::cos(ph)), float(c[1] + r * std::cos(th)), float(c[2] + r * std::sin(th) * std::sin(ph))});
        }
    for (int i = 0; i < rings; i++)
        for (int j = 0; j < seg; j++) {
            const uint32_t a = i * (seg + 1) + j, b = a + seg + 1;
            T.insert(T.end(), {a, a + 1, b, a + 1, b + 1, b});
        }
}

static GridSpec spec(std::array<int, 3> dims, std::array<double, 3> mn, std::array<double, 3> mx) {
    GridSpec s;
    s.dims = dims;
    s.h = 1;
    s.origin = {0, 0, 0};
    s.mn = mn;
    s.mx = mx;
    return s;
}

TEST("wall records measure the true wall position along lattice links") {
    // box occupying x in [10.3, 14.3], y and z in [3, 7] on a unit grid
    std::vector<float> P;
    std::vector<uint32_t> T;
    box(P, T, 10.3, 14.3, 3, 7, 3, 7);
    const Grid g = buildGrid(spec({20, 10, 10}, {10.3, 3, 3}, {14.3, 7, 7}), P, T);
    CHECK(g.rec.count > 0);
    // fluid cell (9, 5, 5) sees the face x = 10.3 through its +x link (direction 2: c = -x)
    const uint32_t c = 9 + 20 * (5 + 10 * 5);
    int r = -1;
    for (int64_t i = 0; i < g.rec.count; i++)
        if (g.rec.cell[i] == c) r = int(i);
    CHECK(r >= 0);
    CHECK(g.rec.mask[r] & (1u << 2));
    CHECK_NEAR((g.rec.q[18 * r + 1] - 1) / 254.0, 0.8, 0.01);
    CHECK_NEAR(g.rec.normal[3 * r], -1.0, 1e-6);
    CHECK_NEAR(g.rec.dist[r], 0.8, 0.01);
    // the surface areas add up to the box's faces (4 x 4 each), shared among the records along them
    double out[3] = {0, 0, 0}, in[3] = {0, 0, 0};
    int holding = 0;
    for (int64_t i = 0; i < g.rec.count; i++) {
        for (int a = 0; a < 3; a++) (g.rec.area[3 * i + a] > 0 ? out : in)[a] += g.rec.area[3 * i + a];
        holding += g.rec.area[3 * i] != 0 || g.rec.area[3 * i + 1] != 0 || g.rec.area[3 * i + 2] != 0;
    }
    for (int a = 0; a < 3; a++) {
        CHECK_NEAR(out[a], 16.0, 0.3);
        CHECK_NEAR(in[a], -16.0, 0.3);
    }
    CHECK(holding > 60);  // not one record per triangle
}

TEST("a plate with no cell inside it still blocks the flow (thin-wall links)") {
    std::vector<float> P;
    std::vector<uint32_t> T;
    box(P, T, 10.4, 10.6, 3, 9, 3, 9);
    const Grid g = buildGrid(spec({24, 12, 12}, {10.4, 3, 3}, {10.6, 9, 9}), P, T);
    CHECK(g.solidCount == 0);
    CHECK(g.rec.count >= 2 * 36);
    auto sim = makeCpu(g, Params{0.08, 0.01, 0.16, false, false});
    sim->step(400);
    sim->takeForces();
    sim->step(200);
    const auto f = sim->takeForces();
    const double cd = f.me[0] / f.steps / (0.5 * 0.08 * 0.08 * 36);
    std::printf("    thin plate Cd %.2f\n", cd);
    CHECK(cd > 0.5 && cd < 2.5);
}

TEST("the law of the wall: Reichardt from the sublayer to the log layer") {
    CHECK_NEAR(reichardt(1)[0], 1.0, 0.02);
    CHECK_NEAR(reichardt(100)[0], std::log(100.0) / 0.41 + 5.6, 0.3);
    for (double ut : {0.05, 0.001}) {
        const double y = 0.5, nu = 1e-5;
        const double utau = reichardtUtau(ut, y, nu);
        CHECK_NEAR(utau * reichardt(y * utau / nu)[0], ut, 1e-6 * ut);
    }
}

TEST("GPU flow engine matches the CPU engine (walls, wall model, faces)") {
    if (!gpuAvailable()) {
        std::printf("    (skipped: no GPU)\n");
        return;
    }
    std::vector<float> P;
    std::vector<uint32_t> T;
    const double c[3] = {14, 14, 14};
    sphere(P, T, 5, c);
    const Grid g = buildGrid(spec({50, 28, 28}, {9, 9, 9}, {19, 19, 19}), P, T);
    for (bool rr : {false, true}) {
        Params p;
        p.nuLat = rr ? 1e-5 : 0.002;
        p.rr = rr;
        p.wallModel = rr;
        p.half = false;
        auto cpu = makeCpu(g, p);
        auto gpu = makeGpu(g, p);
        p.half = true;
        auto gpu16 = makeGpu(g, p);
        cpu->step(300);
        gpu->step(300);
        gpu16->step(300);
        const auto a = cpu->takeForces(), b = gpu->takeForces(), h = gpu16->takeForces();
        std::printf("    %s: drag CPU %.5f, GPU 32-bit %.5f, 16-bit %.5f\n", rr ? "rr" : "bgk", a.me[0], b.me[0], h.me[0]);
        CHECK(a.steps == 300 && b.steps == 300);
        CHECK_NEAR(b.me[0], a.me[0], 1e-4 * std::abs(a.me[0]));
        CHECK_NEAR(h.me[0], a.me[0], 2e-3 * std::abs(a.me[0]));
    }
}

TEST("GPU flow engine matches the CPU engine over a moving ground") {
    if (!gpuAvailable()) {
        std::printf("    (skipped: no GPU)\n");
        return;
    }
    // a slab 3 cells above the road (part walls without the wall model there) and one 9 cells above
    std::vector<float> P;
    std::vector<uint32_t> T;
    box(P, T, 12, 40, 4.2, 8.2, 6, 18);
    box(P, T, 12, 40, 10.2, 13.2, 6, 18);
    GridSpec sp = spec({64, 22, 24}, {12, 4.2, 6}, {40, 13.2, 18});
    sp.ground = true;
    const Grid g = buildGrid(sp, P, T);
    Params p;
    p.nuLat = 1e-5;
    p.half = false;
    auto cpu = makeCpu(g, p);
    auto gpu = makeGpu(g, p);
    for (int k = 0; k < 4; k++) {
        cpu->step(100);
        gpu->step(100);
        const auto a = cpu->takeForces(), b = gpu->takeForces();
        std::printf("    step %d: lift CPU %.6f, GPU %.6f; drag CPU %.6f, GPU %.6f\n", 100 * (k + 1), a.me[1], b.me[1], a.me[0], b.me[0]);
        CHECK_NEAR(b.me[1], a.me[1], 1e-4 * std::abs(a.me[1]) + 1e-4);
        CHECK_NEAR(b.me[0], a.me[0], 1e-4 * std::abs(a.me[0]) + 1e-4);
    }
}

TEST("a ground moving with the wind leaves the free stream undisturbed") {
    GridSpec sp = spec({64, 20, 12}, {30, 8, 5}, {31, 9, 6});
    sp.ground = true;
    const Grid g = buildGrid(sp, {}, {});
    for (double nu : {0.01, 3e-6})
    for (bool gpu : {false, true}) {
        if (gpu && !gpuAvailable()) continue;
        Params p;
        p.nuLat = nu;
        p.wallModel = false;
        auto sim = gpu ? makeGpu(g, p) : makeCpu(g, p);
        sim->sampleEvery = 1;
        sim->step(3000);
        const auto f = sim->fields();
        std::printf("    %s, nu %g: ux/U up from the ground at x = 48:", gpu ? "GPU" : "CPU", nu);
        for (int y = 2; y < 19; y++) std::printf(" %.3f", f.inst[4 * (48 + 64 * (y + 20 * 6)) + 1] / 0.08);
        std::printf("\n");
        for (int y = 2; y < 19; y++) CHECK_NEAR(f.inst[4 * (48 + 64 * (y + 20 * 6)) + 1] / 0.08, 1.0, 0.03);
    }
}

TEST("flow past a coarse cube: a bluff-body drag coefficient from momentum exchange") {
    const Grid g = syntheticGrid(40e3);
    auto sim = makeCpu(g, Params{0.08, 0.002, 0.16, true, false});
    sim->step(1500);
    sim->takeForces();
    sim->step(600);
    const auto f = sim->takeForces();
    const int s = g.dims[1] / 4;
    const double cd = f.me[0] / f.steps / (0.5 * 0.08 * 0.08 * s * s);
    std::printf("    %d x %d x %d cells, Cd %.2f, Cl %.3f\n", g.dims[0], g.dims[1], g.dims[2], cd, f.me[1] / f.steps / (0.5 * 0.08 * 0.08 * s * s));
    CHECK(cd > 0.6 && cd < 1.5);
}

TEST_MAIN
