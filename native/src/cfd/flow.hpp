// The v1 flow engine: the same scheme as the web app's src/cfd/flow*.js (read flow-cpu.js for the
// method). D3Q19 lattice Boltzmann with in-place ("esoteric pull") streaming, recursive regularised
// collision with a Smagorinsky subgrid model, walls from sparse records (interpolated bounce-back,
// thin walls, a moving ground belt, a wall model for turbulent boundary layers), forces by
// momentum exchange. The CPU solver (flow_cpu.cpp) and the GPU solver (flow_gpu.cpp, the shared
// shader native/shaders/flow.wgsl) compute the same thing.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ps::flow {

constexpr uint8_t BULK = 0, WALL = 1, SOLID = 2, FACE = 3;
constexpr int RAMP_STEPS = 400;
constexpr int MODEL_CLEAR = 3;  // cells of fluid the wall model needs along its sample direction on the part
extern const double W[19];
extern const int CX[19], CY[19], CZ[19], OPP[19];
/** 13.5 x (H_xxy + H_yzz, H_xzz + H_xyy, H_yyz + H_xxz) and 4.5 x their differences, per direction. */
extern const std::array<std::array<double, 6>, 19> P3C;

/** Inlet speed at a step (ramped up over the first RAMP_STEPS steps). */
double inletVelocity(double uLat, int64_t step);

/** Reichardt's law of the wall: {u+, du+/dy+} at y+. */
std::array<double, 2> reichardt(double yp);
/** Friction velocity with u_tau u+(y u_tau / nu) = ut. */
double reichardtUtau(double ut, double y, double nu);

/** Wall records (flow.js buildFlowGrid): one per fluid cell with a link blocked by the part or ground. */
struct Records {
    int64_t count = 0;
    std::vector<uint32_t> cell, mask, groundMask;
    std::vector<uint8_t> q;          // 18 per record: 1 + round(254 q), 0 = half-way
    std::vector<float> normal, dist; // wall normal into the fluid (3 per record), distance (cells)
    std::vector<uint8_t> samp;       // wall model's sample direction (0: plain bounce-back)
    std::vector<float> y2, area;     // sample's distance from the wall; part surface next to it (3)
    std::vector<uint8_t> onBelt;     // the nearest wall is the moving ground
};

struct Grid {
    std::array<int, 3> dims{0, 0, 0};
    int64_t N = 0;
    std::vector<uint8_t> kind;
    Records rec;
    bool ground = false, periodicZ = false;
    int64_t solidCount = 0;
    double h = 1;
    std::array<double, 3> origin{0, 0, 0};
};

/** What buildGrid needs of a tunnel plan (airflow.hpp planTunnel). */
struct GridSpec {
    std::array<int, 3> dims;
    double h;
    std::array<double, 3> origin, mn, mx;  // grid origin; the part's extent in the wind frame
    bool ground = false, periodicZ = false;
};

/**
 * Cell kinds and wall records for a tunnel: the part (vertices in the wind frame, `q`) voxelized
 * within its box, wall links cast against its triangles (including thin walls), the wall model's
 * sample cells and surface areas.
 */
Grid buildGrid(const GridSpec& spec, const std::vector<float>& q, const std::vector<uint32_t>& tris);
/** A tunnel of about n cells with a cube in it and half-way walls (benchmarks). */
Grid syntheticGrid(double n);

struct Params {
    double uLat = 0.08, nuLat = 1e-5, smagorinsky = 0.16;
    bool rr = true;         // recursive regularised collision (false: BGK)
    bool wallModel = true;  // turbulent boundary layer (only with rr)
    bool belt = true;       // the ground moves with the wind
    bool half = true;       // GPU: 16-bit population storage
    int wgx = 64, wgy = 1;  // GPU: the bulk kernel's workgroup
    bool timing = false;    // GPU: time the batches with timestamp queries (Solver::gpuSeconds)
};

/** Forces on the part over some steps (lattice units, summed over them). */
struct Forces {
    int64_t steps = 0;
    std::array<double, 3> me{0, 0, 0}, pressure{0, 0, 0}, friction{0, 0, 0};
};

/** The reduced view fields: [rho - 1, ux, uy, uz] per coarse cell (rho - 1 = -2: no fluid). */
struct Fields {
    std::array<int, 3> dims{0, 0, 0};
    int factor = 1;
    std::vector<float> inst, avg;
    int samples = 0;
};

class Solver {
public:
    virtual ~Solver() = default;
    virtual void reset() = 0;
    /** Advance `count` steps (the GPU in submissions of ~40 ms). */
    virtual void step(int count) = 0;
    /** Forces since the last call (and adds to the wall cells' long-time density sums). */
    virtual Forces takeForces() = 0;
    /** Start the time averages again (view fields, surface density). */
    virtual void resetAverages() = 0;
    /** Time-averaged density minus one per wall record. */
    virtual std::vector<float> surfaceRho() = 0;
    /** The view fields now (instantaneous and averaged). */
    virtual Fields fields() = 0;
    /**
     * Diagnostics (the CPU solver; empty elsewhere): per wall record since the last resetAverages,
     * the mean force on the part per step [fx, fy, fz] and the wall model's mean friction velocity
     * (0 where it is off), 4 values per record, lattice units.
     */
    virtual std::vector<double> recordForces() { return {}; }
    /**
     * Pipelined stepping (the GPU keeps running while the caller reads results): queue `count` steps
     * with their forces summed; returns a ticket for collect(). The CPU solver runs them at once.
     */
    virtual int submit(int count) { step(count); return 0; }
    /** Wait for a ticket's steps and return their forces. */
    virtual Forces collect(int ticket) { (void)ticket; return takeForces(); }
    /** Sample the view fields every `every` steps (0: never). */
    int sampleEvery = 20;
    int64_t steps = 0;
    int64_t cells = 0;
    double gpuSeconds = 0;  // GPU time of the batches (Params::timing, when the GPU has timestamp queries)
    bool timed = false;     // gpuSeconds is measured
    std::string name;
};

std::unique_ptr<Solver> makeCpu(const Grid& grid, const Params& p);
/** Throws when the GPU is unavailable or the grid does not fit. */
std::unique_ptr<Solver> makeGpu(const Grid& grid, const Params& p);
/** Largest grid the GPU can hold (population buffers split across bindings); 0 without a GPU. */
int64_t gpuMaxCells(bool half = true);

}  // namespace ps::flow
