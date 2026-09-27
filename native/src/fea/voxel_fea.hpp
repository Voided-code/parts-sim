// Linear-elastic voxel finite-element solver.
//
// The part is voxelized into a regular grid of cubic 8-node elements. Each element has a density
// in (0, 1] (its filled volume fraction) that scales its stiffness, which smooths out the
// staircase along curved or inclined surfaces.
//
// K u = f is solved matrix-free with conjugate gradients (64-bit) preconditioned by a geometric
// multigrid V-cycle with degree-2 Chebyshev smoothing (32-bit: it only approximates K^-1, and
// flexible CG absorbs its rounding). Coarse operators are exact Galerkin products (P^T K P) computed element by element,
// which keeps the iteration count low (typically 15-60) even for long slender parts in bending.
// Matrix products gather, for each grid node, the rows of its (up to 8) surrounding voxels, so
// they run on all CPU cores without write conflicts. Nodes inside a uniform region (their 8 voxels
// all s * Kb, Kb = K0 on the finest level) use the assembled 27-point stencil instead: 243
// multiply-adds rather than 576. A coarse element whose 8 children are s * Kb is s times the next
// level's Kb (their exact Galerkin product), so only the others store a matrix.
//
// Units: the solver works in a normalized system (E = 1, voxel size = 1). With physical E [Pa]
// and voxel size h [m], u_physical = u_normalized / (E * h) for forces in N.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "hex8.hpp"
#include "../util/parallel.hpp"

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
    std::vector<double> rho;         // stiffness scale of the elements that are rho * Kb (all of the finest level)
    std::vector<float> K;            // 24x24 matrices of the other elements (coarse levels; 32-bit, 4 floats of padding)
    std::vector<int> kIdx;           // coarse levels: element -> its matrix in K, or -1 for rho * Kb
    // the level's uniform element Kb: K0 on the finest level, then the Galerkin product of 8 uniform
    // children (exact, and not 2 K0: the element is not a plain trilinear one). Padded for 4-wide
    // loads past a row's end, in both precisions, and its assembled stencil by columns: for each
    // neighbour row (oy, oz) of three nodes (ox = 0..2, contiguous) and each of their 9 DOFs, the 3
    // rows it contributes to (and one of padding)
    std::array<double, 580> Kb{};
    std::array<float, 580> Kbf{};
    std::array<double, 324> Sb{};
    std::array<float, 324> Sbf{};
    std::vector<int> emap;           // voxel -> element (or -1)
    std::vector<float> voxelScale;   // voxel -> s for s * Kb elements, -1 for the others, 0 if empty
    std::vector<float> nodeScale;    // node -> s when its 8 voxels are all s * Kb (27-point stencil), else 0
    // rows of nodes along x (row = j + NY k): first and last active i (-1 if none), and the rows with any
    std::vector<int> rowFirst, rowLast;
    std::vector<int64_t> activeRows;
    std::vector<uint8_t> bc, fixed, activeNode;
    std::vector<double> invDiag, diagAdd;
    std::vector<float> invDiag32;    // for the 32-bit V-cycle
    double omega = 0.5;              // damped Jacobi (coarsest-level fallback)
    double lmax = 0;                 // estimate of the largest eigenvalue of D^-1 K (0 = not yet)
    int64_t freeDof = 0;
    std::vector<float> r, z, t, d;   // V-cycle work vectors (32-bit)
    TransferMap mx, my, mz;          // to the next coarser level
    std::vector<int> childStart, children;  // coarse element -> fine elements (built by coarsen)

    bool scaled(int64_t e) const { return kIdx.empty() || kIdx[e] < 0; }
    /** fn(lo, hi) in parallel over the DOF ranges of the active node spans (every other DOF stays 0). */
    template <class F> void forActive(F&& fn) const;
    template <class F> void forActiveNodes(F&& fn) const;  // the same over node ranges
    template <class F> double sumActive(F&& fn) const;
    const float* matrix(int64_t e) const { return K.data() + int64_t(kIdx[e]) * 576; }  // elements that are not scaled
    double scale(int64_t e) const { return scaled(e) ? rho[e] : 1.0; }
};

/** Degree-2 Chebyshev smoother steps: z = d = first D^-1 r, then d = c1 d + c2 D^-1 (r - K z), z += d. */
struct ChebyshevCoefficients {
    double first, c1, c2;
};
ChebyshevCoefficients chebyshevCoefficients(double lmax);
/** Largest eigenvalue of a symmetric tridiagonal matrix (diagonal a, off-diagonal b; bisection). */
double tridiagonalMax(const std::vector<double>& a, const std::vector<double>& b);

template <class F> void Level::forActiveNodes(F&& fn) const {
    parallelFor(int64_t(activeRows.size()), [&](int64_t a, int64_t b) {
        for (int64_t q = a; q < b; q++) {
            const int64_t row = activeRows[q];
            fn(row * NX + rowFirst[row], row * NX + rowLast[row] + 1);
        }
    }, std::max<int64_t>(1, 2048 / NX));
}

template <class F> void Level::forActive(F&& fn) const {
    forActiveNodes([&](int64_t lo, int64_t hi) { fn(3 * lo, 3 * hi); });
}

template <class F> double Level::sumActive(F&& fn) const {
    return parallelSum(int64_t(activeRows.size()), [&](int64_t a, int64_t b) {
        double s = 0;
        for (int64_t q = a; q < b; q++) {
            const int64_t row = activeRows[q];
            s += fn(3 * (row * NX + rowFirst[row]), 3 * (row * NX + rowLast[row] + 1));
        }
        return s;
    }, std::max<int64_t>(1, 4096 / NX));
}

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
    std::vector<float> nodeVM, nodeP1, nodeP3;  // P1 / P3 empty unless principals were asked for
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
    /** Estimates the smoothers' eigenvalue ranges where not known yet (the GPU solver estimates
     *  them on the GPU instead; the CPU V-cycle calls this first). */
    void prepareSmoothers();
    /** Diagnostics: time of the 32-bit matrix product on each level, one line each. */
    std::string profile(int reps = 10);
    /** One V-cycle as a preconditioner: z = M^-1 r on the finest level (full-length vectors). */
    void precondition(const double* r, double* z);

    SolveResult solve(const std::vector<double>& f, const SolveOptions& opts = {});
    std::array<double, 3> reactions(const std::vector<double>& u, const std::vector<double>* f = nullptr) const;
    /** Stresses [Pa] from physical displacements u [m], modulus E [Pa], voxel size h [m]
     *  (principal stresses only when asked: they cost more than the rest). */
    NodalStresses stresses(const std::vector<double>& u, double E, double h, bool principals = true) const;
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
    void setBase(Level& L, const double* K);
    void finishLevel(Level& L);
    Level coarsen(Level& F);
    void estimateOmega(int l);
    void buildCoarseSolver(Level& L);
    void coarseSolve(Level& L);
    void coarseSolve64(const double* r, double* z);
    void jacobi(int l, bool first);
    void chebyshev(int l, bool first);
    template <class T> void applyT(int l, const T* x, T* y, bool raw) const;
    template <class T> void restrict(const Level& F, const Level& C, const T* rf, T* rc, bool zeroFixed = true) const;
    template <class T> void prolongAdd(const Level& F, const Level& C, const T* zc, T* zf) const;
    // one V-cycle on the finest level from r (64-bit), in 32-bit with r scaled to order one
    void precondition32(const double* r, double* z);

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
