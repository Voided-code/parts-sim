// Drop test: explicit dynamics of the part hitting a rigid floor.
//
// Central-difference time integration with a lumped mass matrix, the same linear-elastic voxel
// elements as the static study, a small stiffness-proportional damping that removes numerical
// ringing at the highest frequencies, and frictionless penalty contact with a rigid floor under
// the part (gravity along -Y). The part starts touching the floor, moving at the impact speed
// sqrt(2 g H) of a drop from height H. The time step is set from the highest natural frequency
// (power iteration), so the run is stable without user tuning.
//
// Physical units throughout: u [m], v [m/s], t [s], forces [N], stresses [Pa].
#pragma once

#include <functional>
#include <limits>
#include <vector>

#include "voxel_fea.hpp"

namespace ps {

/** Highest eigenvalue of M^-1 K (normalized units) by power iteration. */
double maxEigenvalue(const VoxelFEA& fea, const std::vector<double>& mass, int iterations = 40);

struct DropOptions {
    double E = 200e9, rho = 7850, h = 1e-3;  // Pa, kg/m^3, m
    double speed = 1;                        // impact speed [m/s]
    std::vector<double> nodeY;               // height of every node above the floor [m]
    std::vector<uint8_t> surface;            // 1 for nodes that can touch the floor
    double g = 9.81;
    int frames = 48, maxSteps = 20000;
    double maxTime = std::numeric_limits<double>::infinity();
    struct Frame { double t; std::vector<float> u, vm; double force; };
    std::function<void(Frame&&)> onFrame;
    std::function<bool(double)> onProgress;  // true = cancel
};

struct DropResult {
    std::vector<float> vmMax, tPeak;  // per node
    struct Sample { double t, force; };
    std::vector<Sample> history;
    double dt = 0, duration = 0, peakForce = 0, contactTime = 0, wMax = 0;
    int steps = 0;
    bool rebounded = false;
};

/** Explicit drop simulation on the CPU (fea: free, unheld voxel model). */
DropResult dropTestCPU(const VoxelFEA& fea, const DropOptions& o);

/** Time step, damping and contact constants shared by the CPU and GPU drop solvers. */
struct DropSetup {
    std::vector<double> m, kc, cc;  // DOF masses [kg], node contact stiffness / damping
    double wMax = 0, beta = 0, dt = 0;
};
DropSetup dropSetup(const VoxelFEA& fea, const DropOptions& o);

}  // namespace ps
