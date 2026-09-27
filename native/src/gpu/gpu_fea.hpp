// GPU structural solve (fea.wgsl). The multigrid hierarchy is built by VoxelFEA on the CPU and
// uploaded. solve() runs the CPU solver's multigrid-preconditioned conjugate gradients on the GPU
// in 32-bit, with 64-bit reliable updates on the CPU, so it converges like the CPU solver and
// reaches the same accuracy.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "../fea/voxel_fea.hpp"

namespace ps {

constexpr int64_t GPU_COARSEST_DOF = 1100;  // same hierarchy as the CPU solver (thin parts converge badly with one more level)

class GpuFeaSolver {
public:
    /** Throws when the GPU is unavailable or the upload fails. fea must use coarsestMaxDof <= GPU_COARSEST_DOF. */
    explicit GpuFeaSolver(VoxelFEA& fea);
    ~GpuFeaSolver();

    /** One multigrid V-cycle on the GPU (float32): z ~ K^-1 r, like VoxelFEA::precondition. */
    void precondition(const double* r, double* z);
    /** K u = f to the requested tolerance (32-bit CG on the GPU with 64-bit reliable updates on the CPU). */
    SolveResult solve(const std::vector<double>& f, const SolveOptions& opts);
    /** Diagnostics: GPU time of each kernel and of the whole V-cycle, one line each. */
    std::string profile(int reps = 20);


private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    VoxelFEA& fea_;
};

}  // namespace ps
