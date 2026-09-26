// Linear-elastic voxel finite-element solver.
//
// The part is voxelized into a regular grid of cubic 8-node elements. Each element has a density
// in (0, 1] (its filled volume fraction) that scales its stiffness, which smooths out the
// staircase along curved or inclined surfaces.
//
// K u = f is solved matrix-free with conjugate gradients preconditioned by a geometric multigrid
// V-cycle. Coarse operators are exact Galerkin products (P^T K P) computed element by element,
// which keeps the iteration count low (typically 15-60) even for long slender parts in bending.
// Matrix products gather, for each grid node, the rows of its (up to 8) surrounding voxels, so
// they run on all CPU cores without write conflicts.
//
// Units: the solver works in a normalized system (E = 1, voxel size = 1). With physical E [Pa]
// and voxel size h [m], u_physical = u_normalized / (E * h) for forces in N.
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <vector>

#include "hex8.hpp"

namespace ps {

struct TransferMap {
    std::vector<int> c0, c1;  // coarse nodes interpolating fine node i (c1 = -1: coincident)
};

struct Level {
    int nx = 0, ny = 0, nz = 0, NX = 0, NY = 0, NZ = 0;
    int64_t nNodes = 0, nDof = 0;
    std::array<int64_t, 8> off{};
    std::vector<int> elems;          // voxel index of each element
    std::vector<int64_t> base;       // node index of each element's first corner
    std::vector<double> rho;         // density scale (finest level, shared K0)
    std::vector<double> K;           // per-element 24x24 (coarse levels)
    std::vector<int> emap;           // voxel -> element (or -1)
    std::vector<uint8_t> bc, fixed, activeNode;
    std::vector<double> invDiag, diagAdd;
    double omega = 0.5;
    int64_t freeDof = 0;
    std::vector<double> r, z, t;
    TransferMap mx, my, mz;          // to the next coarser level
    std::vector<int> childStart, children;  // coarse element -> fine elements (built by coarsen)

    bool shared() const { return K.empty(); }
};

struct SolveOptions {
    double tol = 1e-6;
    int maxIter = 500;
    const std::vector<double>* x0 = nullptr;
    std::function<bool(int, double)> onProgress;  // return true to cancel
};

struct SolveResult {
    std::vector<double> u;
    int iterations = 0;
    double residual = 0;
    bool converged = false, cancelled = false;
};

struct NodalStresses {
    std::vector<float> elemVM, nodeVM, nodeP1, nodeP3;
};

class VoxelFEA {
public:
    /**
     * dims: element counts; density per voxel (index i + nx*(j + ny*k)), 0 = empty;
     * bc per DOF (3 per node), 1 = held at zero; diagAdd (optional) per DOF, added to K's diagonal.
     */
    VoxelFEA(std::array<int, 3> dims, const std::vector<float>& density, double nu, const std::vector<uint8_t>& bc,
             int64_t coarsestMaxDof = 1100, const std::vector<double>* diagAdd = nullptr);

    int64_t nNodes() const { return levels[0].nNodes; }
    int64_t nDof() const { return levels[0].nDof; }

    /** y = K x on level l (held/inactive DOFs of y zeroed unless raw). */
    void apply(int l, const double* x, double* y, bool raw = false) const;
    void vcycle(int l);
    /** One V-cycle as a preconditioner: z = M^-1 r on the finest level (full-length vectors). */
    void precondition(const double* r, double* z);

    SolveResult solve(const std::vector<double>& f, const SolveOptions& opts = {});
    std::array<double, 3> reactions(const std::vector<double>& u, const std::vector<double>* f = nullptr) const;
    /** Stresses [Pa] from physical displacements u [m], modulus E [Pa], voxel size h [m]. */
    NodalStresses stresses(const std::vector<double>& u, double E, double h) const;
    /** Nodal stress tensors (6 per node, Pa), corner-averaged like stresses(). */
    std::vector<float> stressTensors(const std::vector<double>& u, double E, double h) const;

    double nu;
    std::array<double, 576> K0{};
    std::array<std::array<double, 144>, 8> cornerStress{}, gaussB{};
    std::vector<Level> levels;

    struct Coarse {
        std::vector<int64_t> map;  // level DOF -> dense index or -1
        int64_t m = 0;
        std::vector<double> A;     // Cholesky factor (row-major lower)
        std::vector<double> y;
    } coarse;

private:
    void finishLevel(Level& L);
    Level coarsen(Level& F);
    void estimateOmega(int l);
    void buildCoarseSolver(Level& L);
    void coarseSolve(Level& L);
    void jacobi(int l, bool first);
    void restrict(const Level& F, const Level& C, const double* rf, double* rc, bool zeroFixed = true) const;
    void prolongAdd(const Level& F, const Level& C, const double* zc, double* zf) const;

    std::array<std::array<double, 576>, 8> M0_{};  // P_c^T K0 P_c per child position
};

/** Principal stresses of a symmetric tensor, sorted descending. */
void principalStresses(double sx, double sy, double sz, double txy, double tyz, double tzx, double out[3]);
double vonMises(const double* s);

struct PruneResult {
    std::vector<float> density;
    int removed = 0;
};
/** Drop voxels not face-connected to a voxel touching a held node. */
PruneResult pruneFloating(std::array<int, 3> dims, const std::vector<float>& density, const std::vector<uint8_t>& heldNode);

}  // namespace ps
