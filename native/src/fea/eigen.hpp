// Eigenvalue problems on the voxel model: natural frequencies (K x = w^2 M x) and linear buckling
// (K x = -lambda K_G x), solved with LOBPCG (Knyazev 2001) preconditioned by the multigrid.
#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <vector>

#include "voxel_fea.hpp"

namespace ps {

using Vec = std::vector<double>;
using Block = std::vector<Vec>;

/** Lumped mass per DOF in normalized units (voxel side 1, density 1): fill fraction / 8 per node. */
Vec lumpedMass(const VoxelFEA& fea);
/** Centroid strain-displacement matrix (6x24) of the unit cube. */
const std::array<double, 144>& centroidB();
/** 6 matrices (8x8): sum_c s_c T[c] = integral of grad N_a . S . grad N_b for a constant stress S. */
const std::array<std::array<double, 64>, 6>& geometricTables();
/** Mean stress of every solved voxel in units of E from physical displacements u [m], voxel size h [m]. */
std::vector<double> elementStresses(const VoxelFEA& fea, const Vec& u, double h);

/** Geometric (stress) stiffness in the solver's normalized units, from dimensionless voxel stresses. */
class GeometricStiffness {
public:
    GeometricStiffness(const VoxelFEA& fea, const std::vector<double>& sigma);
    void apply(const double* x, double* y) const;

private:
    const Level& L_;
    std::vector<double> G_;
    std::array<std::vector<int64_t>, 8> colours_;  // voxels of the same parity share no nodes
};

/** Symmetric eigen-decomposition (cyclic Jacobi): ascending values, column eigenvectors. */
void symmetricEigen(const std::vector<double>& A, int k, std::vector<double>& values, std::vector<double>& vectors);

struct LobpcgOptions {
    int nev = 5;
    int block = 0;  // 0 = nev + guard vectors
    double tol = 1e-5;
    int maxIter = 300;
    // Ritz values above this that have settled (moved less than 1% of |settleAbove| over the last
    // 10 iterations) count as converged: buckling only needs to know an eigenvalue lies outside the
    // searched range, and a cluster near zero would never meet the relative tolerance. Ritz values
    // only decrease towards the eigenvalues, so a settled value above the bound stays above it.
    double settleAbove = INFINITY;
    std::function<bool(int it, double res, int converged)> onProgress;  // true = cancel
};

struct LobpcgResult {
    std::vector<double> values;
    Block vectors;
    std::vector<double> residuals;
    int iterations = 0;
    bool converged = false;
};

/** Smallest eigenpairs of A x = theta B x (A symmetric, B SPD); precond approximates A^-1. */
LobpcgResult lobpcg(int64_t n, const std::function<void(const double*, double*)>& applyA, const std::function<void(const double*, double*)>& applyB,
                    const std::function<Block(const Block&)>& precond, const LobpcgOptions& opts);

/** Maps between full DOF vectors and vectors of the free DOFs only. */
struct FreeDofs {
    std::vector<int64_t> idx;
    int64_t n = 0;
    explicit FreeDofs(const VoxelFEA& fea);
    void scatter(const double* x, double* full, int64_t nFull) const;
    void gather(const double* full, double* x) const;
};

/** Preconditioner on full-length vectors: z = M^-1 r for each r. */
using FullPreconditioner = std::function<Block(const Block&)>;
/** One multigrid V-cycle per vector (CPU). */
FullPreconditioner cpuPreconditioner(VoxelFEA& fea);

struct FrequencyResult {
    std::vector<double> freqs, lambdas;
    Block modes;  // full length, mode^T M mode = 1 with the normalized mass
    Vec mass;
    bool converged = false;
    int iterations = 0;
};
FrequencyResult naturalFrequencies(VoxelFEA& fea, int nev, double E, double density, double h, double shift, const FullPreconditioner& pre,
                                   const std::function<bool(int, double, int)>& onProgress = {}, double tol = 1e-5);

struct BucklingResult {
    std::vector<double> factors;  // +inf: no buckling below maxFactor for this mode
    double maxFactor = INFINITY;  // the range that was searched
    Block modes;
    bool converged = false;
    int iterations = 0;
};
/**
 * Lowest buckling load factors up to maxFactor (factors above it come back as +inf: a part that
 * yields long before would not reach them, and the search there is slow and ill-conditioned).
 */
BucklingResult bucklingFactors(VoxelFEA& fea, const std::vector<double>& sigma, int nev, const FullPreconditioner& pre,
                               const std::function<bool(int, double, int)>& onProgress = {}, double tol = 1e-5,
                               double maxFactor = INFINITY);

}  // namespace ps
