// Voxel model, importers and the whole bend-test pipeline (mirrors test/model.test.js,
// test/voxelize.test.js, test/import.test.js and test/solidworks.test.js).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

#include "check.hpp"
#include "core/importers.hpp"
#include "core/mesh.hpp"
#include "core/solidworks.hpp"
#include "core/voxelize.hpp"
#include "fea/structural.hpp"
#include "fea/voxel_fea.hpp"

using namespace ps;

#ifndef PS_TEST_DATA
#define PS_TEST_DATA "native/tests/data"
#endif

static Bytes readData(const std::string& name) {
    std::ifstream in(std::string(PS_TEST_DATA) + "/" + name, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// axis-aligned box as 12 triangles (outward), like THREE.BoxGeometry
static MeshSource box(double sx, double sy, double sz) {
    MeshSource m;
    const double x = sx / 2, y = sy / 2, z = sz / 2;
    const double v[8][3] = {{-x, -y, -z}, {x, -y, -z}, {x, y, -z}, {-x, y, -z}, {-x, -y, z}, {x, -y, z}, {x, y, z}, {-x, y, z}};
    for (auto& p : v)
        for (double c : p) m.positions.push_back(float(c));
    m.index = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3};
    return m;
}

static std::vector<int32_t> selectTris(const Part& p, const std::function<bool(const double*, const float*)>& test) {
    std::vector<int32_t> out;
    for (int t = 0; t < p.nTri; t++) {
        double c[3] = {0, 0, 0};
        for (int k = 0; k < 3; k++)
            for (int d = 0; d < 3; d++) c[d] += p.vertices[3 * p.tris[3 * t + k] + d] / 3.0;
        if (test(c, &p.triNormal[3 * t])) out.push_back(t);
    }
    return out;
}

TEST("sphere volume from voxel fractions") {
    // icosphere-like UV sphere, r = 10
    MeshSource m;
    const int nu = 64, nv = 32;
    const double r = 10;
    auto P = [&](int i, int j) {
        const double th = M_PI * j / nv, ph = 2 * M_PI * i / nu;
        return std::array<double, 3>{r * std::sin(th) * std::cos(ph), r * std::cos(th), r * std::sin(th) * std::sin(ph)};
    };
    for (int j = 0; j < nv; j++)
        for (int i = 0; i < nu; i++) {
            const auto a = P(i, j), b = P(i + 1, j), c = P(i + 1, j + 1), d = P(i, j + 1);
            for (auto q : {a, c, b, a, d, c})
                for (double x : q) m.positions.push_back(float(x));
        }
    auto part = buildPart(m);
    const Grid g = gridForBox(part->bbox.min, part->bbox.max, 40);
    const auto frac = voxelize(part->vertices, part->tris, g, 4);
    double vol = 0;
    for (float f : frac) vol += f;
    vol *= g.h * g.h * g.h;
    const double exact = 4.0 / 3 * M_PI * r * r * r;
    std::printf("    voxel volume %.1f vs mesh %.1f (sphere %.1f)\n", vol, part->volume, exact);
    CHECK(std::abs(vol / part->volume - 1) < 0.01);
}

TEST("beam sample deflection matches theory at every voxel resolution") {
    auto src = box(200, 20, 10);
    src.name = "beam";
    auto part = buildPart(src);
    const double eps = 0.02;
    Fixture clamp{"Clamped end", {{selectTris(*part, [&](const double* c, const float* n) { return c[0] < part->bbox.min[0] + eps && n[0] < -0.9; }), {}}}};
    Load tip;
    tip.name = "Tip load";
    tip.type = Load::Force;
    tip.magnitude = 800;
    tip.dir = {0, -1, 0};
    tip.patches = {{selectTris(*part, [&](const double* c, const float* n) { return c[0] > part->bbox.max[0] - eps && n[0] > 0.9; }), {}}};
    const double E = 200e9, nu = 0.29;
    const double I = 0.01 * 0.02 * 0.02 * 0.02 / 12;
    const double theory = 800 * 0.008 / (3 * E * I) + 800 * 0.2 / ((5.0 / 6) * (E / (2 * (1 + nu))) * 0.02 * 0.01);
    Material mat = *findMaterial("steel-1020");
    mat.E = 200;
    mat.nu = nu;
    for (int res : {40, 64, 90, 104, 128}) {
        StructuralModel m(part, res);
        const auto a = m.assemble({clamp}, {tip}, false, mat, 0.001);
        VoxelFEA fea(m.grid.dims, m.density, nu, a.bc);
        SolveOptions o;
        o.tol = 1e-7;
        const auto sol = fea.solve(a.f, o);
        double tipU = 0;
        for (int64_t n = 0; n < m.nNodes; n++) tipU = std::max(tipU, std::abs(sol.u[3 * n + 1]) / (E * m.grid.h * 0.001));
        const double ratio = tipU / theory;
        std::printf("    %d voxels: tip %.3f mm vs %.3f mm (x%.3f)\n", res, tipU * 1e3, theory * 1e3, ratio);
        CHECK(ratio > 0.93 && ratio < 1.07);
    }
}

TEST("STL: binary round trip and ASCII") {
    const auto src = box(10, 20, 30);
    Bytes bin(84, 0);
    const uint32_t n = uint32_t(src.index.size() / 3);
    std::memcpy(&bin[80], &n, 4);
    for (uint32_t t = 0; t < n; t++) {
        uint8_t rec[50] = {};
        for (int k = 0; k < 3; k++)
            for (int d = 0; d < 3; d++) {
                const float v = src.positions[3 * src.index[3 * t + k] + d];
                std::memcpy(rec + 12 + 12 * k + 4 * d, &v, 4);
            }
        bin.insert(bin.end(), rec, rec + 50);
    }
    auto part = buildPart(importBytes("box.stl", bin));
    CHECK(part->faceCount == 6);
    CHECK(std::abs(part->volume - 6000) < 1e-2);
    const std::string ascii = "solid t\nfacet normal 0 0 1\nouter loop\nvertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\nendloop\nendfacet\nendsolid t\n";
    const auto m = importBytes("tri.stl", Bytes(ascii.begin(), ascii.end()));
    CHECK(m.positions.size() == 9);
}

TEST("OBJ with polygons and negative indices") {
    const std::string obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\nf -4 -2 -1\n";
    const auto m = importBytes("q.obj", Bytes(obj.begin(), obj.end()));
    CHECK(m.positions.size() == 9 * 3);
}

TEST("importFile opens UTF-8 directories and file names") {
    namespace fs = std::filesystem;
    std::random_device rd;
    fs::path dir = fs::temp_directory_path() / fs::path(u8"parts-sim-r\u00e9pertoire-\u96f6\u4ef6-");
    dir += std::to_string(rd()) + std::to_string(rd());
    if (!fs::create_directory(dir)) throw std::runtime_error("Could not create the test directory.");
    struct Cleanup {
        fs::path dir;
        ~Cleanup() { std::error_code ec; fs::remove_all(dir, ec); }
    } cleanup{dir};
    const fs::path path = dir / fs::path(u8"pi\u00e8ce \u96f6\u4ef6.OBJ");
    {
        std::ofstream out(path, std::ios::binary);
        out << "v 0 0 0\nv 2 0 0\nv 0 3 0\nf 1 2 3\n";
        if (!out) throw std::runtime_error("Could not write the test mesh.");
    }
    const std::u8string utf8 = path.u8string();
    const auto m = importFile(std::string(utf8.begin(), utf8.end()));
    const std::u8string stem = u8"pi\u00e8ce \u96f6\u4ef6";
    CHECK(m.name == std::string(stem.begin(), stem.end()));
    CHECK(m.positions == std::vector<float>({0, 0, 0, 2, 0, 0, 0, 3, 0}));
}

TEST("STEP import through OpenCascade keeps the CAD faces") {
#ifdef PARTS_SIM_HAS_OCCT
    const auto m = importBytes("cube.stp", readData("cube.stp"));
    auto part = buildPart(m);
    std::printf("    %s: %d triangles, %d faces, size %.1f x %.1f x %.1f mm\n", m.info.c_str(), part->nTri, part->faceCount, part->bbox.size[0], part->bbox.size[1], part->bbox.size[2]);
    CHECK(part->faceCount == 6);
    CHECK(part->brepFaces);
    CHECK(m.units == "mm");
#else
    std::printf("    (skipped: built without OpenCascade)\n");
#endif
}

TEST("SolidWorks part: display mesh, CAD faces, millimetres and material") {
    const auto m = readSolidWorks(readData("cube.SLDPRT"), "cube.SLDPRT");
    CHECK(m.units == "mm");
    CHECK(m.positions.size() / 9 == 12);
    CHECK(m.material == "AISI 1020");
    CHECK(matchMaterial(m.material) == "steel-1020");
    auto part = buildPart(m);
    CHECK(std::abs(part->bbox.size[0] - 20) < 1e-3 && std::abs(part->bbox.size[1] - 20) < 1e-3 && std::abs(part->bbox.size[2] - 20) < 1e-3);
    CHECK(std::abs(part->volume - 8000) < 1e-2);
    CHECK(part->faceCount == 6);
}

TEST("SolidWorks assembly: parts found beside it, placements and mirroring") {
    const auto asmData = readData("robot.SLDASM");
    const auto names = assemblyPartNames(asmData);
    CHECK(names.size() == 1);
    const auto m = readSolidWorks(asmData, "robot.SLDASM", [](const std::string& name) {
        (void)name;
        return readData("cube.SLDPRT");
    });
    std::printf("    %s\n", m.info.c_str());
    CHECK(m.info.find("3 of 3 components") != std::string::npos);
    auto part = buildPart(m);
    CHECK(std::lround(part->bbox.size[0]) == 140 && std::lround(part->bbox.size[1]) == 120 && std::lround(part->bbox.size[2]) == 20);
}

TEST_MAIN
