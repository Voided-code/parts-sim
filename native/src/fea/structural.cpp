#include "structural.hpp"

#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

#include "../util/parallel.hpp"

namespace ps {

namespace {
constexpr double MIN_FILL = 0.3;
constexpr double GRAVITY = 9.81;
}

StructuralModel::StructuralModel(std::shared_ptr<Part> part_, int resolution_) : part(std::move(part_)), resolution(resolution_) {
    grid = gridForBox(part->bbox.min, part->bbox.max, resolution);
    const int nx = grid.dims[0], ny = grid.dims[1], nz = grid.dims[2];
    // sample finely enough to resolve thin walls, within a memory budget for the sample grid
    wallThickness = typicalWallThickness(*part);
    int sub = 5;
    if (std::isfinite(wallThickness)) sub = std::max(sub, std::min(6, int(std::ceil(3 * grid.h / wallThickness))));
    while (sub > 2 && double(nx) * ny * nz * sub * sub * sub > 48e6) sub--;
    const auto frac = voxelize(part->vertices, part->tris, grid, sub);
    // walls under ~1.5 voxels thick get a connected layer with their true cross-section
    std::vector<float> shell;
    if (wallThickness < 2.5 * grid.h) shell = thinWallDensity(*part, grid, 1.5 * grid.h);
    density.assign(frac.size(), 0.f);
    for (size_t e = 0; e < frac.size(); e++) {
        const float f = frac[e], w = shell.empty() ? 0.f : shell[e];
        float rho = 0;
        if (w > 0) {
            rho = f >= 0.9f ? f : std::max(w, 0.02f);
            if (f < 0.9f) thinVoxels++;
        } else if (f >= MIN_FILL) rho = f;
        if (rho > 0) {
            density[e] = std::min(1.f, rho);
            voxelCount++;
        }
    }
    NX = nx + 1;
    NY = ny + 1;
    NZ = nz + 1;
    nNodes = int64_t(NX) * NY * NZ;
    activeNode.assign(nNodes, 0);
    std::vector<uint8_t> full(nNodes, 0);
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) {
                if (!(density[i + size_t(nx) * (j + size_t(ny) * k)] > 0)) continue;
                for (int c = 0; c < 8; c++) {
                    const int64_t n = node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                    activeNode[n] = 1;
                    full[n]++;
                }
            }
    surfaceNode.assign(nNodes, 0);
    for (int64_t n = 0; n < nNodes; n++)
        if (activeNode[n] && full[n] < 8) surfaceNode[n] = 1;
}

Vec3 StructuralModel::nodePosition(int64_t n) const {
    const int i = int(n % NX), j = int((n / NX) % NY), k = int(n / (int64_t(NX) * NY));
    return {grid.origin[0] + i * grid.h, grid.origin[1] + j * grid.h, grid.origin[2] + k * grid.h};
}

int64_t StructuralModel::nearestSurfaceNode(double x, double y, double z, double maxCells) const {
    const double gx = (x - grid.origin[0]) / grid.h, gy = (y - grid.origin[1]) / grid.h, gz = (z - grid.origin[2]) / grid.h;
    const int r = int(std::ceil(maxCells));
    const int ci = int(std::lround(gx)), cj = int(std::lround(gy)), ck = int(std::lround(gz));
    int64_t best = -1;
    double bestD = maxCells * maxCells;
    for (int k = std::max(0, ck - r); k <= std::min(NZ - 1, ck + r); k++)
        for (int j = std::max(0, cj - r); j <= std::min(NY - 1, cj + r); j++)
            for (int i = std::max(0, ci - r); i <= std::min(NX - 1, ci + r); i++) {
                const int64_t n = node(i, j, k);
                if (!surfaceNode[n]) continue;
                const double d = (i - gx) * (i - gx) + (j - gy) * (j - gy) + (k - gz) * (k - gz);
                if (d < bestD) { bestD = d; best = n; }
            }
    return best;
}

Samples StructuralModel::samplePatches(const std::vector<Patch>& patches) const {
    Samples out;
    // repeated clicks and overlapping brush strokes describe a union of areas: count each point once
    std::set<std::tuple<int32_t, float, float, float>> seen;
    for (const auto& p : patches) {
        const Samples s = sampleTriangles(*part, p.tris, grid.h * 0.4, p.clip);
        for (size_t i = 0; i < s.weights.size(); i++) {
            const float x = s.points[3 * i], y = s.points[3 * i + 1], z = s.points[3 * i + 2];
            if (!seen.emplace(s.tris[i], x, y, z).second) continue;
            out.points.insert(out.points.end(), {x, y, z});
            out.weights.push_back(s.weights[i]);
            out.tris.push_back(s.tris[i]);
        }
    }
    return out;
}

Assembly StructuralModel::assemble(const std::vector<Fixture>& fixtures, const std::vector<Load>& loads, bool gravity, const Material& material,
                                   double toMeters) const {
    validateMaterial(material);
    if (!std::isfinite(toMeters) || toMeters <= 0) throw std::invalid_argument("Invalid model units.");
    Assembly a;
    a.bc.assign(3 * nNodes, 0);
    a.f.assign(3 * nNodes, 0.0);
    auto hold = [&](int64_t n) {
        if (!a.bc[3 * n]) a.fixedNodes++;
        a.bc[3 * n] = a.bc[3 * n + 1] = a.bc[3 * n + 2] = 1;
    };
    for (const auto& fx : fixtures) {
        const Samples s = samplePatches(fx.patches);
        int hit = 0;
        for (size_t i = 0; i < s.weights.size(); i++) {
            const double x = s.points[3 * i], y = s.points[3 * i + 1], z = s.points[3 * i + 2];
            const int64_t n = nearestSurfaceNode(x, y, z);
            if (n < 0) continue;
            hit++;
            hold(n);
            // the grid can overhang the selected face by part of a voxel: hold every surface node on
            // the face (within 3/4 voxel of it), or the clamp's edge rows stay free and the part pivots
            forSurfaceNodesNear(x, y, z, 0.75, hold);
        }
        if (!hit) a.warnings.push_back(fx.name + ": no solid voxels under the selection - try a finer mesh.");
    }
    for (const auto& ld : loads) {
        if (ld.type == Load::Wind) {
            const auto& F = ld.forces;
            if (int(F.size()) != part->nTri * 3) throw std::invalid_argument(ld.name + ": invalid airflow forces.");
            int lost = 0;
            for (int t = 0; t < part->nTri; t++) {
                const double fx = F[3 * t], fy = F[3 * t + 1], fz = F[3 * t + 2];
                if (fx == 0 && fy == 0 && fz == 0) continue;
                double c[3] = {0, 0, 0};
                for (int k = 0; k < 3; k++)
                    for (int d = 0; d < 3; d++) c[d] += part->vertices[3 * part->tris[3 * t + k] + d] / 3.0;
                const int64_t n = nearestSurfaceNode(c[0], c[1], c[2], 3.5);
                if (n < 0) { lost++; continue; }
                a.f[3 * n] += fx; a.f[3 * n + 1] += fy; a.f[3 * n + 2] += fz;
                a.total[0] += fx; a.total[1] += fy; a.total[2] += fz;
            }
            if (lost) a.warnings.push_back(ld.name + ": " + std::to_string(lost) + " surface triangles had no nearby voxel.");
            continue;
        }
        if (!std::isfinite(ld.magnitude) || ld.magnitude < 0) throw std::invalid_argument(ld.name + ": magnitude must be a finite, nonnegative number.");
        Vec3 direction{0, 0, 0};
        if (ld.type == Load::Force) {
            const double len = std::sqrt(ld.dir[0] * ld.dir[0] + ld.dir[1] * ld.dir[1] + ld.dir[2] * ld.dir[2]);
            if (!(len > 0) || !std::isfinite(len)) throw std::invalid_argument(ld.name + ": choose a nonzero force direction.");
            for (int d = 0; d < 3; d++) direction[d] = ld.dir[d] / len;
        }
        const Samples s = samplePatches(ld.patches);
        std::map<int64_t, std::array<double, 4>> nodes;
        double W = 0;
        for (size_t i = 0; i < s.weights.size(); i++) {
            const int64_t n = nearestSurfaceNode(s.points[3 * i], s.points[3 * i + 1], s.points[3 * i + 2]);
            if (n < 0) continue;
            const double w = s.weights[i];
            auto& e = nodes[n];
            e[0] += w;
            if (ld.type == Load::Pressure) {
                // pressure pushes into the surface: -normal * p * area
                const int t = s.tris[i];
                for (int d = 0; d < 3; d++) e[1 + d] -= part->triNormal[3 * t + d] * w;
            }
            W += w;
        }
        if (nodes.empty()) {
            a.warnings.push_back(ld.name + ": no solid voxels under the selection - try a finer mesh.");
            continue;
        }
        for (const auto& [n, e] : nodes) {
            double v[3];
            if (ld.type == Load::Pressure) {
                const double k = ld.magnitude * 1e6 * toMeters * toMeters;  // MPa * model-unit^2 -> N
                for (int d = 0; d < 3; d++) v[d] = e[1 + d] * k;
            } else {
                const double s2 = ld.magnitude * e[0] / W;
                for (int d = 0; d < 3; d++) v[d] = direction[d] * s2;
            }
            for (int d = 0; d < 3; d++) {
                a.f[3 * n + d] += v[d];
                a.total[d] += v[d];
            }
        }
    }
    if (gravity) {
        const int nx = grid.dims[0], ny = grid.dims[1];
        const double hm = grid.h * toMeters;
        double mass = 0;
        for (size_t e = 0; e < density.size(); e++) {
            const double d = density[e];
            if (!(d > 0)) continue;
            const double m = material.density * d * hm * hm * hm;
            mass += m;
            const int i = int(e % nx), j = int((e / nx) % ny), k = int(e / (size_t(nx) * ny));
            for (int c = 0; c < 8; c++) a.f[3 * node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1)) + 1] -= m * GRAVITY / 8;
        }
        a.total[1] -= mass * GRAVITY;
    }
    if (a.fixedNodes > 0 && a.fixedNodes < 3) a.warnings.push_back("Very small fixture - the part may pivot. Select a larger area.");
    return a;
}

VertexWeights vertexWeights(const std::vector<float>& V, const Grid& grid, const std::vector<uint8_t>& activeNode) {
    VertexWeights W;
    const int nV = int(V.size() / 3);
    W.nV = nV;
    W.idx.assign(8 * size_t(nV), -1);
    W.wts.assign(8 * size_t(nV), 0.f);
    const auto& dims = grid.dims;
    const int64_t NX = dims[0] + 1, NY = dims[1] + 1, NZ = dims[2] + 1;
    auto node = [&](int i, int j, int k) { return i + NX * (j + NY * k); };
    parallelFor(nV, [&](int64_t lo, int64_t hi) {
        for (int64_t v = lo; v < hi; v++) {
            double g[3], t[3];
            int c[3];
            for (int d = 0; d < 3; d++) {
                g[d] = (V[3 * v + d] - grid.origin[d]) / grid.h;
                c[d] = std::min(dims[d] - 1, std::max(0, int(std::floor(g[d]))));
                t[d] = std::min(1.0, std::max(0.0, g[d] - c[d]));
            }
            double sum = 0;
            for (int q = 0; q < 8; q++) {
                const int64_t n = node(c[0] + (q & 1), c[1] + ((q >> 1) & 1), c[2] + ((q >> 2) & 1));
                if (!activeNode[n]) continue;
                const double w = ((q & 1) ? t[0] : 1 - t[0]) * (((q >> 1) & 1) ? t[1] : 1 - t[1]) * (((q >> 2) & 1) ? t[2] : 1 - t[2]) + 1e-6;
                W.idx[8 * v + q] = n;
                W.wts[8 * v + q] = float(w);
                sum += w;
            }
            if (sum > 0) {
                for (int q = 0; q < 8; q++) W.wts[8 * v + q] = float(W.wts[8 * v + q] / sum);
                continue;
            }
            // a sliver just outside the voxels borrows the nearest node within 1.5 voxels
            int64_t best = -1;
            double bestD = 2.25;
            const int ci = int(std::lround(g[0])), cj = int(std::lround(g[1])), ck = int(std::lround(g[2]));
            for (int k = std::max(0, ck - 2); k <= std::min(int(NZ) - 1, ck + 2); k++)
                for (int j = std::max(0, cj - 2); j <= std::min(int(NY) - 1, cj + 2); j++)
                    for (int i = std::max(0, ci - 2); i <= std::min(int(NX) - 1, ci + 2); i++) {
                        const int64_t n = node(i, j, k);
                        if (!activeNode[n]) continue;
                        const double d = (i - g[0]) * (i - g[0]) + (j - g[1]) * (j - g[1]) + (k - g[2]) * (k - g[2]);
                        if (d < bestD) { bestD = d; best = n; }
                    }
            if (best >= 0) { W.idx[8 * v] = best; W.wts[8 * v] = 1; }
        }
    }, 1024);
    return W;
}

template <class T>
static std::vector<float> interpolateT(const VertexWeights& W, const T* field, int comps, double scale) {
    std::vector<float> out(size_t(comps) * W.nV, 0.f);
    parallelFor(W.nV, [&](int64_t lo, int64_t hi) {
        for (int64_t v = lo; v < hi; v++) {
            bool any = false;
            double acc[6] = {0, 0, 0, 0, 0, 0};
            for (int q = 0; q < 8; q++) {
                const int64_t n = W.idx[8 * v + q];
                if (n < 0) continue;
                any = true;
                const double w = W.wts[8 * v + q] * scale;
                for (int c = 0; c < comps; c++) acc[c] += w * field[comps * n + c];
            }
            for (int c = 0; c < comps; c++) out[comps * v + c] = any ? float(acc[c]) : NAN;
        }
    }, 2048);
    return out;
}

std::vector<float> interpolate(const VertexWeights& W, const float* field, int comps, double scale) { return interpolateT(W, field, comps, scale); }
std::vector<float> interpolate(const VertexWeights& W, const double* field, int comps, double scale) { return interpolateT(W, field, comps, scale); }

VertexWeights StructuralModel::vertexWeights(const std::vector<uint8_t>& active) const { return ps::vertexWeights(part->vertices, grid, active); }

bool StructuralModel::isSurfaceVoxel(int64_t e) const {
    const int nx = grid.dims[0], ny = grid.dims[1], nz = grid.dims[2];
    if (!(density[e] > 0)) return false;
    const int i = int(e % nx), j = int((e / nx) % ny), k = int(e / (int64_t(nx) * ny));
    if (i == 0 || j == 0 || k == 0 || i == nx - 1 || j == ny - 1 || k == nz - 1) return true;
    const int64_t sxy = int64_t(nx) * ny;
    return !(density[e - 1] > 0 && density[e + 1] > 0 && density[e - nx] > 0 && density[e + nx] > 0 && density[e - sxy] > 0 && density[e + sxy] > 0);
}

std::vector<float> StructuralModel::voxelCenters(bool surfaceOnly) const {
    const int nx = grid.dims[0], ny = grid.dims[1];
    std::vector<float> out;
    for (size_t e = 0; e < density.size(); e++) {
        if (!(density[e] > 0) || (surfaceOnly && !isSurfaceVoxel(int64_t(e)))) continue;
        const int i = int(e % nx), j = int((e / nx) % ny), k = int(e / (size_t(nx) * ny));
        out.push_back(float(grid.origin[0] + (i + 0.5) * grid.h));
        out.push_back(float(grid.origin[1] + (j + 0.5) * grid.h));
        out.push_back(float(grid.origin[2] + (k + 0.5) * grid.h));
    }
    return out;
}

}  // namespace ps
