// Voxelization of a triangle mesh and thin-wall handling.
#pragma once

#include <array>
#include <vector>

#include "mesh.hpp"

namespace ps {

struct Grid {
    Vec3 origin{0, 0, 0};
    double h = 1;
    std::array<int, 3> dims{1, 1, 1};
};

/** A grid enclosing a box with n voxels along its longest side (plus `pad` voxels each side). */
Grid gridForBox(const Vec3& min, const Vec3& max, int n, int pad = 0);

/**
 * Filled fraction per voxel by supersampled ray casting: sub^3 samples per voxel, each solid when
 * at least two of its X, Y and Z rays agree (even-odd rule), which tolerates small holes and flipped
 * triangles. Index i + nx*(j + ny*k).
 */
std::vector<float> voxelize(const std::vector<float>& pos, const std::vector<uint32_t>& index, const Grid& grid, int sub = 2);

/** Typical thin-wall thickness (20th percentile of an area-weighted sample), or +inf for chunky parts. */
double typicalWallThickness(Part& part, int samples = 1500);

/** Density thin walls (thinner than maxThickness) contribute to each voxel; 0 where there is none. */
std::vector<float> thinWallDensity(const Part& part, const Grid& grid, double maxThickness);

}  // namespace ps
