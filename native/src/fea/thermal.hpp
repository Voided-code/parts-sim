// Heat transfer on the voxel model: steady-state and transient conduction with fixed
// temperatures, heat sources and surface convection.
//
// One temperature per grid node, 8-node brick conduction elements (conductivity scaled by each
// voxel's fill fraction), lumped heat capacity, backward-Euler time steps. The linear systems are
// solved with conjugate gradients preconditioned by a scalar geometric multigrid V-cycle with
// Galerkin coarse operators, the same scheme as the structural solver (including its 27-point
// stencil inside uniform regions).
//
// Normalized units: the system is divided by k*h (conductivity x voxel size), so temperatures
// come out directly in degrees and a node's convection term is h_c * A / (k h).
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ps {

/** Conduction matrix (8x8) of the unit cube with k = 1: integral of grad N_a . grad N_b. */
const std::array<double, 64>& conductionMatrix();

class ScalarVoxelSolver {
public:
    struct Level {
        int nx, ny, nz, NX, NY, NZ;
        int64_t nNodes;
        std::array<int64_t, 8> off;
        std::vector<int> elems;
        std::vector<int64_t> base;
        std::vector<double> rho, K, diagAdd, invDiag, r, z, t;
        std::vector<uint8_t> bc, fixed, active;
        std::vector<int> emap;          // voxel -> element (or -1)
        std::vector<double> scale;      // element -> s when it is s * K0 (all of the finest level), else -1
        std::vector<float> voxelScale;  // voxel -> scale (0 if empty)
        std::vector<float> nodeScale;   // node -> s when its 8 voxels are all s * K0 (27-point stencil), else 0
        double omega = 0.6;             // damped Jacobi
        int64_t freeDof = 0;
        std::vector<int> mx0, mx1, my0, my1, mz0, mz1;  // to the next coarser level
    };

    /** density: conductivity scale per voxel; fixed: per node, 1 = prescribed; diagAdd: per node (optional). */
    ScalarVoxelSolver(std::array<int, 3> dims, const std::vector<float>& density, const std::vector<uint8_t>& fixed, const std::vector<double>* diagAdd,
                      int64_t coarsestMax = 1500);

    struct Result {
        std::vector<double> x;
        int iterations = 0;
        double residual = 0;
        bool converged = false;
    };
    /** Solve A T = f with T fixed at x0 on prescribed nodes; x0 also warm-starts the free nodes. */
    Result solve(const std::vector<double>& f, const std::vector<double>& x0, double tol = 1e-8, int maxIter = 400);

    std::vector<Level> levels;
    /** Diagnostics: time of the matrix product and of a V-cycle per level, one line each. */
    std::string profile(int reps = 20);

private:
    void finishLevel(Level& L);
    Level coarsen(Level& F);
    void apply(const Level& L, const double* x, double* y, bool raw = false) const;
    void estimateOmega(Level& L);
    void buildCoarse(Level& L);
    void coarseSolve(Level& L);
    void jacobi(Level& L, bool first);
    void restrict(const Level& F, const Level& C, const double* rf, double* rc, bool zeroFixed = true) const;
    void prolongAdd(const Level& F, const Level& C, const double* zc, double* zf) const;
    void vcycle(size_t l);

    struct Coarse {
        std::vector<int64_t> map;
        int64_t m = 0;
        std::vector<double> A, y;
    } coarse_;
};

/** Lumped heat-capacity share per node (normalized: voxel volume 1, rho*cp 1). */
std::vector<double> nodeCapacity(std::array<int, 3> dims, const std::vector<float>& density);
/** Heat flux magnitude [W/m^2] per node; T in degrees, k [W/m K], h voxel size [m]. */
std::vector<float> heatFlux(const ScalarVoxelSolver& s, const std::vector<double>& T, double k, double h);

struct HeatInput {
    std::array<int, 3> dims;
    std::vector<float> density;
    std::vector<uint8_t> fixedNode;    // 1 where the temperature is prescribed
    std::vector<double> fixedValue;    // prescribed temperatures [deg]
    std::vector<double> source;        // heat input per node [W]
    std::vector<double> convH;         // h_c * area per node [W/K]
    std::vector<double> convT;         // ambient temperature per node [deg]
    double k = 50, h = 1e-3;           // W/m K, m
    double rhoCp = 0;                  // J/m^3 K (transient)
    double duration = 0;               // s; 0 = steady state
    int steps = 30;
    double initial = 20;
};

struct HeatResult {
    std::unique_ptr<ScalarVoxelSolver> solver;
    std::vector<double> T;
    bool converged = true;
    int iterations = 0;
};

/** onStep(i, t, T) after each step (i = 0 is the start / steady result); returns true to cancel. */
HeatResult solveHeat(const HeatInput& in, const std::function<bool(int, double, const std::vector<double>&)>& onStep = {},
                     const std::function<void(const ScalarVoxelSolver&)>& onBuilt = {});

}  // namespace ps
