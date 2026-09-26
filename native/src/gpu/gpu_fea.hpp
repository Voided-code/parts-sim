// GPU version of the multigrid-preconditioned CG solve (fea.wgsl, the same shader the web app
// used). The multigrid hierarchy is built by VoxelFEA on the CPU and uploaded; each GPU solve
// runs in 32-bit floats and serves as the preconditioner of a 64-bit flexible CG on the CPU, so
// the result reaches the CPU solver's accuracy.
#pragma once

#include <memory>
#include <vector>

#include "../fea/voxel_fea.hpp"

namespace ps {

constexpr int64_t GPU_COARSEST_DOF = 300;

class GpuFeaSolver {
public:
    /** Throws when the GPU is unavailable or the upload fails. fea must use coarsestMaxDof <= GPU_COARSEST_DOF. */
    explicit GpuFeaSolver(VoxelFEA& fea);
    ~GpuFeaSolver();

    /** Approximate K x = rhs on the GPU from x = 0 (float32 multigrid-PCG). */
    std::vector<double> pcg(const std::vector<double>& rhs, double tol, int maxIter, int* iterations = nullptr);
    /** K u = f to the requested tolerance (flexible CG in float64, GPU-preconditioned). */
    SolveResult solve(const std::vector<double>& f, const SolveOptions& opts);

    static double innerTol;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    VoxelFEA& fea_;
};

}  // namespace ps
