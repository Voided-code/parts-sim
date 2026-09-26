// Airflow study: builds a wind tunnel around the part, runs the lattice-Boltzmann solver (GPU or
// all CPU cores) on a background thread, time-averages the flow, and integrates the forces.
// The UI reads immutable snapshots for particles, streamlines and slice planes.
#pragma once

#include <array>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../core/mesh.hpp"
#include "lbm.hpp"

namespace ps {

constexpr double AIR_U_LAT = 0.08;  // lattice free-stream speed (Mach ~0.14)

/** Direction the air travels. yaw 0 = toward -Z (hits the +Z face), 90 = toward +X; pitch > 0 tilts upward. */
Vec3 windDirection(double yawDeg, double pitchDeg);

struct AirflowOptions {
    Vec3 dir{0, 0, -1};
    double speed = 20;        // m/s
    double airDensity = 1.2;  // kg/m^3
    double cells = 1e6;       // grid size: the tunnel is meshed with about this many cells
    int engine = 0;           // 0 auto, 1 GPU, 2 CPU
    double toMeters = 0.001;
};

/** How large a flow grid this machine can take, and what each cell costs. */
struct AirflowCapacity {
    double minCells, maxCells, defaultCells;
    double gpuBytesPerCell;  // GPU memory (0 for the CPU engine)
    double ramBytesPerCell;  // system memory (on unified-memory Macs this includes the GPU's share)
};
AirflowCapacity airflowCapacity(bool gpu);

/** The wind-tunnel grid for a part and wind direction at about `cells` cells (without meshing it). */
struct TunnelPlan {
    std::array<Vec3, 3> basis;               // wind frame: e1 = wind direction
    std::array<double, 3> origin, mn, mx;    // grid origin; part extent in the wind frame
    std::array<int, 3> dims;
    double h;                                // cell size [model units]
};
TunnelPlan planTunnel(const Part& part, const Vec3& dir, double cells);

struct AirResults {
    double drag = 0, lift = 0, side = 0, cd = 0, cl = 0, frontalArea = 0;  // N, -, m^2
    Vec3 force{0, 0, 0};                                                   // world frame [N]
};

struct AirSnapshot {
    std::vector<float> macro;         // [rho, ux, uy, uz] per cell, instantaneous
    std::vector<float> avgRho, avgU;  // time-averaged (1 and 3 per cell)
    int64_t steps = 0;
    int samples = 0;
    double mlups = 0;
    bool developing = true;
    AirResults results;
};

class AirflowSim {
public:
    AirflowSim() = default;
    ~AirflowSim();

    using Progress = std::function<bool(const std::string&)>;  // true = cancel
    /** Mesh the tunnel and create the solver (worker thread OK). Throws on failure. */
    void setup(std::shared_ptr<const Part> part, const AirflowOptions& o, const Progress& progress);

    void start();
    void pause();
    /** Restart the flow from rest (keeps the tunnel). */
    void reset();
    bool running() const { return running_; }
    bool ready() const { return solver_ != nullptr; }

    /** Latest snapshot (null before the first). */
    std::shared_ptr<const AirSnapshot> snapshot() const;
    /** Called on the solver thread after each new snapshot, and with a message on status changes / failures. */
    std::function<void()> onSnapshot;
    std::function<void(const std::string&, bool failed)> onStatus;

    // ---- grid and sampling (lattice coordinates: cell (i,j,k) spans [i, i+1]) ----
    std::array<int, 3> dims{0, 0, 0};
    std::array<double, 3> origin{0, 0, 0};
    double h = 1;                          // cell size [model units]
    std::array<Vec3, 3> basis;             // wind frame: e1 = wind direction
    std::vector<uint8_t> solid;
    std::array<std::array<double, 2>, 2> partRange{};  // part extent in lattice y and z
    double reynolds = 0, simReynolds = 0, nuLat = 0;
    std::string engine;
    AirflowOptions opts;
    std::shared_ptr<const Part> part;

    /** Trilinear sample of fluid cells at lattice point (x, y, z); returns the weight of fluid cells used. */
    double sample(const std::vector<float>& field, int comps, double x, double y, double z, double* out) const;
    bool isSolidAt(double x, double y, double z) const;
    std::array<double, 3> toLattice(const Vec3& p) const;
    Vec3 toWorld(double x, double y, double z) const;

    /** Time-averaged pressure coefficient at every part vertex (NaN where unavailable). */
    std::vector<float> surfaceCp(const AirSnapshot& s) const;
    /** Aerodynamic pressure force on each part triangle [N, world] for the structural study. */
    std::vector<float> triangleForces(const AirSnapshot& s) const;

private:
    void createSolver();
    void loop();
    void stopThread();
    bool process(std::vector<float>&& macro, double mlups);

    std::unique_ptr<LbmSolver> solver_;
    std::vector<uint8_t> links_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    mutable std::mutex snapMutex_;
    std::shared_ptr<const AirSnapshot> snap_;
    int recoveries_ = 0;
    double hm_ = 0, q_ = 0;
    bool useGPU_ = false;
};

}  // namespace ps
