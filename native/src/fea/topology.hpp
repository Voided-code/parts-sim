// Topology optimization: remove material where it carries little load, keeping the part as stiff
// as possible for a target mass.
//
// SIMP (solid isotropic material with penalization): every voxel gets a design density x in
// [xmin, 1]; its stiffness scales with x^p (p = 3), so intermediate densities are inefficient and
// the optimizer is driven to solid/void. Compliance (the work of the loads, the inverse of
// stiffness) is minimized for a volume budget with the optimality-criteria update and a density
// filter of radius rmin voxels (no checkerboards, mesh-independent members).
// Voxels at fixtures and loads are kept solid so the supports and load paths stay attached.
// Each iteration is one linear static solve, warm-started from the previous one.
#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "voxel_fea.hpp"

namespace ps {

constexpr double TOPO_XMIN = 1e-3;

/** u_e^T K0 u_e for every solved voxel (unit stiffness, no density scale). */
std::vector<double> elementEnergies(const VoxelFEA& fea, const std::vector<double>& u);

struct TopologySolve {
    std::vector<double> u, f;
    const VoxelFEA* fea;  // valid during the iteration that returned it
};

struct TopologyIter {
    int it;
    double compliance, volume, change;
    const std::vector<double>& xPhys;
    const std::vector<int>& solid;
};

struct TopologyOptions {
    std::array<int, 3> dims;
    std::vector<float> fill;     // solid fraction per voxel from voxelization (0 = outside)
    std::vector<uint8_t> keep;   // per voxel, 1 = must stay solid
    double volFrac = 0.4;
    double penal = 3, rmin = 1.5, move = 0.2, tol = 0.01;
    int maxIter = 40;
    /** One static solve with the given voxel densities (x0: warm start or null). */
    std::function<TopologySolve(const std::vector<float>& density, const std::vector<double>* x0)> solve;
    std::function<bool(const TopologyIter&)> onIter;  // true = stop
};

struct TopologyResult {
    std::vector<float> density;
    struct Point { int it; double compliance, volume, change; };
    std::vector<Point> history;
    double keptFraction = 0;
};

TopologyResult optimizeTopology(const TopologyOptions& o);

struct TriMesh {
    std::vector<float> positions;
    std::vector<uint32_t> index;
};
/**
 * Smooth closed surface of a voxel density field (naive surface nets with Laplacian smoothing).
 * origin: world position of voxel (0,0,0)'s min corner; h: voxel size.
 */
TriMesh surfaceNets(const std::vector<float>& density, std::array<int, 3> dims, std::array<double, 3> origin, double h, double level = 0.5,
                    int smooth = 2);
/** Binary STL of an indexed triangle mesh. */
std::vector<uint8_t> toSTL(const TriMesh& mesh, const std::string& name = "part");

}  // namespace ps
