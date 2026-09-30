// Airflow study: builds a wind tunnel around the part and runs the v1 flow engine on it (flow.hpp:
// the GPU, or all CPU cores) on a background thread, keeps the forces' statistics, and publishes
// immutable snapshots (forces with their confidence intervals, the reduced view fields, the surface
// pressure) for the UI's particles, streamlines, slice plane and surface colours.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "../core/mesh.hpp"
#include "flow.hpp"

namespace ps {

constexpr double AIR_U_LAT = 0.08;       // lattice free-stream speed (Mach ~0.14)
constexpr double V1_NU_FLOOR = 1e-6;     // the engine's lowest lattice viscosity
constexpr double TURBULENT_RE = 5e5;     // 'auto' boundary layer: turbulent from this Re along the part
constexpr double AIR_VISCOSITY = 1.789e-5;  // ISA sea level, 15 °C [Pa s]

/** Kinematic viscosity of air (m^2/s) at a density (the dynamic viscosity barely changes with it). */
inline double airNu(double density = 1.225) { return AIR_VISCOSITY / density; }

/** Direction the air travels. yaw 0 = toward -Z (hits the +Z face), 90 = toward +X; pitch > 0 tilts upward. */
Vec3 windDirection(double yawDeg, double pitchDeg);

enum class BoundaryLayer { Auto, Turbulent, Laminar };

struct AirflowOptions {
    Vec3 dir{0, 0, -1};
    double speed = 20;          // m/s
    double airDensity = 1.225;  // kg/m^3
    double cells = 4e6;         // grid size: the tunnel is meshed with about this many cells
    int engine = 0;             // 0 auto, 1 GPU, 2 CPU
    double toMeters = 0.001;
    double ground = -1;         // gap under the part to a road moving with the wind (model units); < 0: free air
    BoundaryLayer boundaryLayer = BoundaryLayer::Auto;
};

/** How large a flow grid this machine can take, and what each cell costs. */
struct AirflowCapacity {
    double minCells, maxCells, defaultCells;
    double gpuBytesPerCell;  // GPU memory (0 for the CPU engine)
    double ramBytesPerCell;  // system memory (on unified-memory Macs this includes the GPU's share)
};
AirflowCapacity airflowCapacity(bool gpu);

/** The tunnel's margins in multiples of the part's blockage size (or of its length, `ofLength`). */
struct TunnelMargins {
    double up = 2.25, down = 3.75, side = 1.35;
    bool ofLength = false;
};
struct TunnelOptions {
    double h = 0;               // a fixed cell size instead of the cell budget
    double ground = -1;         // see AirflowOptions::ground
    bool periodicSpan = false;  // a 2D section: the span (lattice z) wraps around, spanCells across
    int spanCells = 8;
    TunnelMargins margins;
};

/** The wind-tunnel grid for a part and wind direction at about `cells` cells (without meshing it). */
struct TunnelPlan {
    std::array<Vec3, 3> basis;                  // wind frame: e1 = wind direction, e2 up (lattice y)
    std::array<double, 3> origin, mn, mx, L;    // grid origin; part extent and size in the wind frame
    std::array<int, 3> dims;
    double h;                                   // cell size [model units]
    double ground = -1;
    bool periodicSpan = false;
    std::vector<float> q;                       // the part's vertices in the wind frame
    flow::GridSpec spec() const;
};
TunnelPlan planTunnel(const Part& part, const Vec3& dir, double cells, const TunnelOptions& o = {});

/** Area of the part's shadow along the wind (model units^2), rasterised from its triangles. */
double frontalArea(const TunnelPlan& plan, const std::vector<uint32_t>& tris);

struct AirResults {
    static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
    double drag = 0, lift = 0, side = 0, cd = 0, cl = 0, frontalArea = 0;  // N, -, m^2
    // 95% confidence half-widths of the averages (NaN until averaging)
    double dragCI = NaN, liftCI = NaN, sideCI = NaN, cdCI = NaN, clCI = NaN;
    double frictionDrag = NaN;  // the part of the drag along the walls (reliable on flat surfaces)
    bool averaged = false;
    Vec3 force{0, 0, 0};  // world frame [N]
};

struct AirSnapshot {
    std::shared_ptr<const flow::Fields> fields;          // reduced view fields (null before the first read)
    std::shared_ptr<const std::vector<float>> surface;   // time-averaged density - 1 per wall record
    int64_t steps = 0;
    int samples = 0;  // force samples in the averages
    double mlups = 0;
    bool developing = true, converged = false;
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
    /** Pause once the forces have settled (95% interval within about 1%). */
    std::atomic<bool> autoStop{true};

    // ---- grid and sampling (lattice coordinates: cell (i,j,k) spans [i, i+1]) ----
    std::array<int, 3> dims{0, 0, 0};
    std::array<double, 3> origin{0, 0, 0};
    double h = 1;                          // cell size [model units]
    std::array<Vec3, 3> basis;             // wind frame: e1 = wind direction
    std::vector<uint8_t> kind;             // flow::BULK, WALL, SOLID, FACE per cell
    std::array<std::array<double, 2>, 2> partRange{};  // part extent in lattice y and z
    double reynolds = 0, reynoldsLength = 0, simReynolds = 0, nuLat = 0, nuAir = 0;
    bool wallModel = false;
    int64_t records = 0;                   // wall records (cells next to the part or the ground)
    std::string engine;
    AirflowOptions opts;
    std::shared_ptr<const Part> part;

    /**
     * Trilinear sample of a view field (Fields::inst or ::avg: [rho - 1, ux, uy, uz] per coarse cell)
     * at lattice point (x, y, z) over coarse cells with fluid. Returns the weight of fluid cells used.
     */
    static double sample(const flow::Fields& f, const std::vector<float>& field, double x, double y, double z, double out[4]);
    bool isSolidAt(double x, double y, double z) const;
    std::array<double, 3> toLattice(const Vec3& p) const;
    Vec3 toWorld(double x, double y, double z) const;

    /** Time-averaged pressure coefficient at every part vertex (NaN where unavailable). */
    std::vector<float> surfaceCp(const AirSnapshot& s) const;
    /** Aerodynamic pressure force on each part triangle [N, world] for the structural study. */
    std::vector<float> triangleForces(const AirSnapshot& s) const;

private:
    void createSolver(const flow::Grid& grid);
    void loop();
    void stopThread();
    void onForces(const flow::Forces& f, int64_t steps, double mlups);
    void publish(bool notify);

    std::unique_ptr<flow::Solver> solver_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    mutable std::mutex snapMutex_;
    std::shared_ptr<const AirSnapshot> snap_;
    std::vector<int32_t> vertexRecords_;  // the wall record next to each part vertex, or -1
    double hm_ = 0, q_ = 0, frontal_ = 0;  // cell size [m], dynamic pressure [Pa], frontal area [cells^2]
    bool useGPU_ = false;
    // solver-thread state
    std::vector<double> sx_, sy_, sz_, sfx_;  // force samples since the flow developed (coefficient units)
    int64_t averageFrom_ = 0;
    bool developing_ = true, converged_ = false;
    double mlups_ = 0;
    int64_t steps_ = 0;
    AirResults results_;
    std::shared_ptr<const flow::Fields> fields_;
    std::shared_ptr<const std::vector<float>> surface_;
    std::chrono::steady_clock::time_point lastNotify_{};
    int notifiedPhase_ = -1;
};

}  // namespace ps
