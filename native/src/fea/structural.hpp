// Structural model: the part's voxel grid, fixtures and loads mapped onto grid nodes, and
// grid results mapped back onto the part's surface vertices.
#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../core/materials.hpp"
#include "../core/mesh.hpp"
#include "../core/voxelize.hpp"

namespace ps {

struct Patch {
    std::vector<int32_t> tris;
    std::optional<Clip> clip;  // brush strokes keep only the part inside the sphere
};

struct Fixture {
    std::string name;
    std::vector<Patch> patches;
};

struct Load {
    enum Type { Force, Pressure, Wind } type = Force;
    std::string name;
    double magnitude = 0;    // N (force), MPa (pressure)
    Vec3 dir{0, -1, 0};
    std::vector<Patch> patches;
    std::vector<float> forces;  // wind: N per part triangle (xyz)
};

struct Assembly {
    std::vector<uint8_t> bc;      // per DOF, held
    std::vector<double> f;        // per DOF, N
    int fixedNodes = 0;
    std::array<double, 3> total{0, 0, 0};
    std::vector<std::string> warnings;
};

/** Trilinear weights from grid nodes to part vertices (8 per vertex; -1 = none). */
struct VertexWeights {
    std::vector<int64_t> idx;
    std::vector<float> wts;
    int nV = 0;
};

class StructuralModel {
public:
    /** resolution: voxels along the longest side */
    StructuralModel(std::shared_ptr<Part> part, int resolution);

    std::shared_ptr<Part> part;
    int resolution;
    Grid grid;
    std::vector<float> density;
    int voxelCount = 0, thinVoxels = 0;
    double wallThickness;
    int NX, NY, NZ;
    int64_t nNodes;
    std::vector<uint8_t> activeNode, surfaceNode;

    int64_t node(int i, int j, int k) const { return i + int64_t(NX) * (j + int64_t(NY) * k); }
    Vec3 nodePosition(int64_t n) const;
    /** Nearest active surface node within maxCells voxels of a point (model units), or -1. */
    int64_t nearestSurfaceNode(double x, double y, double z, double maxCells = 2.5) const;
    template <class F> void forSurfaceNodesNear(double x, double y, double z, double radius, F&& fn) const;
    Samples samplePatches(const std::vector<Patch>& patches) const;
    Assembly assemble(const std::vector<Fixture>& fixtures, const std::vector<Load>& loads, bool gravity, const Material& material, double toMeters) const;
    VertexWeights vertexWeights(const std::vector<uint8_t>& activeNode) const;
    std::vector<float> voxelCenters(bool surfaceOnly) const;
    bool isSurfaceVoxel(int64_t e) const;
};

/** Interpolate a nodal field with `comps` components onto vertices (NaN where there is no data). */
std::vector<float> interpolate(const VertexWeights& W, const float* field, int comps = 1, double scale = 1);
std::vector<float> interpolate(const VertexWeights& W, const double* field, int comps = 1, double scale = 1);
VertexWeights vertexWeights(const std::vector<float>& vertices, const Grid& grid, const std::vector<uint8_t>& activeNode);

template <class F>
void StructuralModel::forSurfaceNodesNear(double x, double y, double z, double radius, F&& fn) const {
    const double gx = (x - grid.origin[0]) / grid.h, gy = (y - grid.origin[1]) / grid.h, gz = (z - grid.origin[2]) / grid.h;
    const double r2 = radius * radius;
    for (int k = std::max(0, int(std::ceil(gz - radius))); k <= std::min(NZ - 1, int(std::floor(gz + radius))); k++)
        for (int j = std::max(0, int(std::ceil(gy - radius))); j <= std::min(NY - 1, int(std::floor(gy + radius))); j++)
            for (int i = std::max(0, int(std::ceil(gx - radius))); i <= std::min(NX - 1, int(std::floor(gx + radius))); i++) {
                const int64_t n = node(i, j, k);
                if (surfaceNode[n] && (i - gx) * (i - gx) + (j - gy) * (j - gy) + (k - gz) * (k - gz) <= r2) fn(n);
            }
}

}  // namespace ps
