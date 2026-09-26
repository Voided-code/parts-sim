// Linear static study and the progressive-damage break test (engine side; no UI).
#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>

#include "voxel_fea.hpp"

namespace ps {

/** Everything a structural run needs from the voxel model. */
struct StructuralInput {
    std::array<int, 3> dims;
    std::vector<float> density;
    std::vector<uint8_t> bc;
    std::vector<double> f;     // N per DOF
    double nu = 0.3;
    double E = 200e9;          // Pa
    double h = 0.001;          // voxel size [m]
    double rho = 7850;         // kg/m^3
    bool useGPU = true;
};

/** Progress callback: (fraction 0..1 or <0 when unknown, text); return true to cancel. */
using ProgressFn = std::function<bool(double, const std::string&)>;

struct StaticResult {
    std::vector<float> u;  // nodal displacement [m]
    std::vector<float> nodeVM, nodeP1, nodeP3;
    std::vector<uint8_t> activeNode;
    std::vector<float> density;
    int iterations = 0;
    double residual = 0;
    bool converged = false;
    int removed = 0;
    double lostLoad = 0;
    std::array<double, 3> reaction{0, 0, 0};
    int levels = 0, voxels = 0;
    std::string engine = "CPU", gpuNote;
};

/** One linear static solve (multigrid CG on the GPU when available, else all CPU cores). */
StaticResult solveStatic(const StructuralInput& in, const ProgressFn& progress);

struct BreakStep {
    int step;
    double lambda;      // load multiple that cracks the next voxels
    double maxDisp;     // [m] at that load
    std::vector<int> cracked, detached;
    std::vector<float> nodeVM, u;  // scaled to lambda
    std::vector<uint8_t> activeNode;
    int voxels;
};

struct BreakResult {
    std::string reason;
    double peak = 0;
    int removed = 0;
    std::string engine = "CPU";
};

/**
 * Progressive damage: solve, remove the voxels that reach `strength` first (von Mises or max
 * principal), drop detached fragments, repeat until the load path is severed.
 */
BreakResult breakTest(const StructuralInput& in, double strength, bool principal, int maxSteps,
                      const std::function<void(BreakStep&&)>& onStep, const ProgressFn& progress);

/** Held nodes (any DOF held). */
std::vector<uint8_t> heldNodes(const std::vector<uint8_t>& bc);
/** Share of the total load magnitude applied to nodes outside the solved model. */
double lostLoadFraction(const VoxelFEA& fea, const std::vector<double>& f);

}  // namespace ps
