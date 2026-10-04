#include "airflow.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <deque>
#include <limits>
#include <stdexcept>

#include "../gpu/gpu.hpp"
#include "../util/parallel.hpp"
#include "../util/system.hpp"
#include "stats.hpp"

namespace ps {

namespace {

// view fields and surface pressure are read back this often while the flow runs [s]
constexpr double FIELDS_EVERY = 0.3, SURFACE_EVERY = 1.0;
// work per batch [s]: short enough to pause at once and to leave the GPU to the desktop in between
constexpr double BATCH_SECONDS = 0.040;
// host memory per cell: the study's cell kinds, the grid's while it is built, the view fields
constexpr double HOST_BYTES = 3;
// the CPU solver: populations (in place), its copy of the cell kinds, wall records
constexpr double CPU_SOLVER_BYTES = 19 * 4 + 1 + 3;

double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 normalize(Vec3 v) {
    const double l = std::sqrt(dot(v, v));
    for (auto& x : v) x /= l;
    return v;
}
Vec3 cross(const Vec3& a, const Vec3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

double gpuBytesPerCell() { return 19 * (gpuShaderF16() ? 2 : 4) + 1 + 2; }

}  // namespace

Vec3 windDirection(double yawDeg, double pitchDeg) {
    const double y = yawDeg * M_PI / 180, p = pitchDeg * M_PI / 180;
    return normalize({std::sin(y) * std::cos(p), std::sin(p), -std::cos(y) * std::cos(p)});
}

AirflowSim::~AirflowSim() { stopThread(); }

AirflowCapacity airflowCapacity(bool gpu) {
    AirflowCapacity c;
#ifdef __APPLE__
    const bool unified = true;  // Apple GPUs share system memory
#else
    const bool unified = false;
#endif
    c.gpuBytesPerCell = gpu ? gpuBytesPerCell() : 0;
    c.ramBytesPerCell = HOST_BYTES + (gpu ? (unified ? c.gpuBytesPerCell : 0) : CPU_SOLVER_BYTES);
    double ram = double(physicalMemory());
    if (!(ram > 0)) ram = 8e9;
    // leave half the memory to the system, the app and the other studies
    const double ramCap = 0.5 * ram / c.ramBytesPerCell;
    c.minCells = 50e3;
    if (gpu) {
        c.maxCells = std::min({ramCap, 0.95 * double(flow::gpuMaxCells(!std::getenv("PARTS_SIM_LBM_FP32"))), 300e6});
        c.defaultCells = 4e6;
    } else {
        // all cores manage ~250 million cell updates a second (10 cores): bigger grids take minutes
        // per flow-through
        c.maxCells = std::min(ramCap, 16e6);
        c.defaultCells = 1e6;
    }
    c.maxCells = std::max(c.maxCells, c.minCells);
    c.defaultCells = std::clamp(c.defaultCells, c.minCells, c.maxCells);
    return c;
}

flow::GridSpec TunnelPlan::spec() const {
    flow::GridSpec s;
    s.dims = dims;
    s.h = h;
    s.origin = origin;
    s.mn = mn;
    s.mx = mx;
    s.ground = ground >= 0;
    s.periodicZ = periodicSpan;
    return s;
}

TunnelPlan planTunnel(const Part& part, const Vec3& dir, double cells, const TunnelOptions& o) {
    TunnelPlan t;
    const Vec3 e1 = normalize(dir);
    const Vec3 ref = std::abs(e1[1]) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    const double r = dot(ref, e1);
    const Vec3 e2 = normalize({ref[0] - r * e1[0], ref[1] - r * e1[1], ref[2] - r * e1[2]});
    t.basis = {e1, e2, cross(e1, e2)};
    // the part in the wind frame
    const auto& V = part.vertices;
    t.q.resize(V.size());
    t.mn = {INFINITY, INFINITY, INFINITY};
    t.mx = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i + 2 < V.size(); i += 3)
        for (int a = 0; a < 3; a++) {
            const double v = V[i] * t.basis[a][0] + V[i + 1] * t.basis[a][1] + V[i + 2] * t.basis[a][2];
            t.q[i + a] = float(v);
            t.mn[a] = std::min(t.mn[a], v);
            t.mx[a] = std::max(t.mx[a], v);
        }
    auto& L = t.L;
    L = {t.mx[0] - t.mn[0], t.mx[1] - t.mn[1], t.mx[2] - t.mn[2]};
    // Tunnel margins scale with the blockage size (square root of the cross-section) and the
    // length, not the largest dimension: a long-span wing needs room around its chord, and
    // sizing by span would leave its thickness only a cell or two across.
    const bool span = o.periodicSpan;
    // a periodic section's blockage comes from its thickness only
    const double B = span ? std::max(L[1], 0.3 * L[0]) : std::max(std::sqrt(L[1] * L[2]), 0.3 * L[0]);
    if (!std::isfinite(L[0] + L[1] + L[2]) || !(B > 0)) throw std::runtime_error("The part has no finite volume to mesh.");
    // margins in multiples of B, or of the part's length along the wind for slender parts (wings)
    const double scale = o.margins.ofLength ? L[0] : B;
    const double up = o.margins.up * scale, down = o.margins.down * scale, side = o.margins.side * scale;
    const bool ground = o.ground >= 0;
    // with a ground plane the floor sits `ground` below the part, plus one layer of cells under it
    const double D[3] = {L[0] + up + down, ground ? o.ground + L[1] + side : L[1] + 2 * side, L[2] + 2 * side};
    const int spanCells = span ? std::max(2, o.spanCells) : 0;
    auto dimsFor = [&](double hv) {
        return std::array<int, 3>{std::max(16, int(std::ceil(D[0] / hv))),
                                  ground ? std::max(16, int(std::ceil(D[1] / hv)) + 1) : std::max(16, int(std::ceil(D[1] / hv))),
                                  span ? spanCells : std::max(16, int(std::ceil(D[2] / hv)))};
    };
    double hh;
    if (o.h > 0) {
        hh = o.h;
        t.dims = dimsFor(hh);
    } else {
        const double budget = std::max(cells, 4096.0);
        hh = span ? std::sqrt(D[0] * D[1] / (budget / spanCells)) : std::cbrt(D[0] * D[1] * D[2] / budget);
        t.dims = dimsFor(hh);
        // rounding up and the minimum per axis must not exceed the budget on thin or long parts
        while (double(t.dims[0]) * t.dims[1] * t.dims[2] > budget) {
            hh *= std::max(1.01, std::cbrt(double(t.dims[0]) * t.dims[1] * t.dims[2] / budget));
            t.dims = dimsFor(hh);
        }
    }
    t.h = hh;
    t.ground = ground ? o.ground : -1;
    t.periodicSpan = span;
    t.origin = {t.mn[0] - up,
                // the ground plane lies on the face between cell layers 0 (solid floor) and 1
                ground ? t.mn[1] - o.ground - hh : t.mn[1] - (t.dims[1] * hh - L[1]) / 2,
                span ? (t.mn[2] + t.mx[2]) / 2 - t.dims[2] * hh / 2 : t.mn[2] - (t.dims[2] * hh - L[2]) / 2};
    return t;
}

double frontalArea(const TunnelPlan& plan, const std::vector<uint32_t>& tris) {
    // the triangles projected on the plane across the wind, filled into a bitmap of up to 1024
    // pixels along the part's larger side
    const double ly = plan.L[1], lz = plan.L[2];
    if (!(ly > 0) || !(lz > 0)) return 0;
    const double px = std::max(ly, lz) / 1024;
    const int W = std::max(1, int(std::ceil(ly / px))), H = std::max(1, int(std::ceil(lz / px)));
    std::vector<uint8_t> hit(size_t(W) * H, 0);
    const auto& q = plan.q;
    for (size_t t = 0; t + 2 < tris.size(); t += 3) {
        double y[3], z[3];
        for (int k = 0; k < 3; k++) {
            y[k] = (q[3 * tris[t + k] + 1] - plan.mn[1]) / px;
            z[k] = (q[3 * tris[t + k] + 2] - plan.mn[2]) / px;
        }
        const double area2 = (y[1] - y[0]) * (z[2] - z[0]) - (y[2] - y[0]) * (z[1] - z[0]);
        if (std::abs(area2) < 1e-12) continue;
        const int i0 = std::max(0, int(std::floor(std::min({y[0], y[1], y[2]})))), i1 = std::min(W - 1, int(std::ceil(std::max({y[0], y[1], y[2]}))));
        const int j0 = std::max(0, int(std::floor(std::min({z[0], z[1], z[2]})))), j1 = std::min(H - 1, int(std::ceil(std::max({z[0], z[1], z[2]}))));
        const double s = area2 > 0 ? 1 : -1;
        for (int j = j0; j <= j1; j++)
            for (int i = i0; i <= i1; i++) {
                const double cy = i + 0.5, cz = j + 0.5;
                bool inside = true;
                for (int e = 0; e < 3 && inside; e++) {
                    const int a = e, b = (e + 1) % 3;
                    inside = s * ((y[b] - y[a]) * (cz - z[a]) - (cy - y[a]) * (z[b] - z[a])) >= 0;
                }
                if (inside) hit[i + size_t(W) * j] = 1;
            }
    }
    size_t n = 0;
    for (uint8_t v : hit) n += v;
    return double(n) * px * px;
}

void AirflowSim::setup(std::shared_ptr<const Part> p, const AirflowOptions& o, const Progress& progress) {
    stopThread();
    solver_.reset();
    {
        std::lock_guard<std::mutex> lock(snapMutex_);
        snap_.reset();
    }
    if (!(o.speed > 0) || !(o.airDensity > 0) || !(o.toMeters > 0) || !std::isfinite(o.speed) || !std::isfinite(o.airDensity))
        throw std::invalid_argument("Wind speed, air density and unit scale must be finite positive numbers.");
    if (!(dot(o.dir, o.dir) > 0)) throw std::invalid_argument("Wind direction must be a finite nonzero vector.");
    if (!(o.cells > 0) || !std::isfinite(o.cells)) throw std::invalid_argument("The flow grid size must be a positive number of cells.");
    if (!std::isfinite(o.ground)) throw std::invalid_argument("Ground clearance must be zero or more.");
    part = std::move(p);
    opts = o;
    useGPU_ = o.engine != 2 && gpuAvailable();
    const auto cap = airflowCapacity(useGPU_);
    TunnelOptions to;
    to.ground = o.ground;
    const TunnelPlan plan = planTunnel(*part, o.dir, std::clamp(o.cells, cap.minCells, cap.maxCells), to);
    basis = plan.basis;
    dims = plan.dims;
    h = plan.h;
    origin = plan.origin;
    const int64_t N = int64_t(dims[0]) * dims[1] * dims[2];
    char buf[160];
    std::snprintf(buf, sizeof buf, "Meshing wind tunnel (%d x %d x %d = %.2fM cells)...", dims[0], dims[1], dims[2], N / 1e6);
    if (progress && progress(buf)) throw std::runtime_error("Cancelled");
    flow::Grid grid = flow::buildGrid(plan.spec(), plan.q, part->tris);
    bool touches = grid.solidCount > 0;
    for (int64_t r = 0; r < grid.rec.count && !touches; r++) touches = (grid.rec.mask[r] & ~grid.rec.groundMask[r]) != 0;
    if (!touches) throw std::runtime_error("The part is too thin for this flow grid. Move the grid-size slider to more cells.");
    if (progress && progress("Measuring the surface...")) throw std::runtime_error("Cancelled");
    partRange = {{{(plan.mn[1] - origin[1]) / h, (plan.mx[1] - origin[1]) / h}, {(plan.mn[2] - origin[2]) / h, (plan.mx[2] - origin[2]) / h}}};
    hm_ = h * o.toMeters;
    // the air: viscosity from the ISA atmosphere at the chosen density
    nuAir = airNu(o.airDensity);
    // Reynolds number on the part's size across the wind (as shown) and on its length along it
    // (which decides whether the boundary layer is turbulent)
    reynolds = o.speed * std::max(plan.L[1], plan.L[2]) * o.toMeters / nuAir;
    reynoldsLength = o.speed * plan.L[0] * o.toMeters / nuAir;
    const double nuReal = nuAir * AIR_U_LAT / (o.speed * hm_);
    nuLat = std::max(nuReal, V1_NU_FLOOR);
    simReynolds = reynolds * nuReal / nuLat;
    wallModel = o.boundaryLayer == BoundaryLayer::Turbulent || (o.boundaryLayer == BoundaryLayer::Auto && reynoldsLength >= TURBULENT_RE);
    q_ = 0.5 * o.airDensity * o.speed * o.speed;
    frontal_ = frontalArea(plan, part->tris) / (h * h);
    groundGap = o.ground >= 0 ? o.ground / h : -1;
    records = grid.rec.count;
    // for each part vertex, the wall record next to it (the surface pressure)
    {
        std::vector<std::pair<uint32_t, int32_t>> byCell(size_t(grid.rec.count));
        for (int64_t r = 0; r < grid.rec.count; r++) byCell[r] = {grid.rec.cell[r], int32_t(r)};
        std::sort(byCell.begin(), byCell.end());
        const int nx = dims[0], ny = dims[1], nz = dims[2];
        vertexRecords_.assign(part->nVert, -1);
        parallelFor(part->nVert, [&](int64_t lo, int64_t hi) {
            for (int64_t v = lo; v < hi; v++) {
                const Vec3 pos{part->vertices[3 * v], part->vertices[3 * v + 1], part->vertices[3 * v + 2]};
                const Vec3 n{part->vertNormal[3 * v], part->vertNormal[3 * v + 1], part->vertNormal[3 * v + 2]};
                // a point just outside the surface, in lattice units
                auto l = toLattice(pos);
                for (int a = 0; a < 3; a++) l[a] += 0.7 * dot(n, basis[a]);
                const int cx = int(std::floor(l[0])), cy = int(std::floor(l[1])), cz = int(std::floor(l[2]));
                int32_t best = -1;
                double bd = INFINITY;
                for (int dz = -1; dz <= 1; dz++)
                    for (int dy = -1; dy <= 1; dy++)
                        for (int dx = -1; dx <= 1; dx++) {
                            const int x = cx + dx, y = cy + dy, z = cz + dz;
                            if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) continue;
                            const uint32_t c = uint32_t(x + int64_t(nx) * (y + int64_t(ny) * z));
                            const auto it = std::lower_bound(byCell.begin(), byCell.end(), std::make_pair(c, int32_t(-1)));
                            if (it == byCell.end() || it->first != c) continue;
                            const double d = std::pow(x + 0.5 - l[0], 2) + std::pow(y + 0.5 - l[1], 2) + std::pow(z + 0.5 - l[2], 2);
                            if (d < bd) { bd = d; best = it->second; }
                        }
                vertexRecords_[v] = best;
            }
        }, 1024);
    }
    if (progress && progress("Starting the flow solver...")) throw std::runtime_error("Cancelled");
    try {
        createSolver(grid);
    } catch (const std::exception& e) {
        if (!useGPU_) throw;
        // the GPU could not take it: rebuild a smaller grid for the CPU
        if (onStatus) onStatus(std::string("GPU flow solver failed (") + e.what() + "); rebuilding a smaller CPU grid.", false);
        AirflowOptions cpu = o;
        cpu.engine = 2;
        cpu.cells = std::min(o.cells, airflowCapacity(false).defaultCells);
        grid = flow::Grid();
        setup(part, cpu, progress);
        return;
    }
    kind = std::move(grid.kind);
    sx_.clear(); sy_.clear(); sz_.clear(); sfx_.clear();
    developing_ = true;
    converged_ = false;
    steps_ = 0;
    mlups_ = 0;
    results_ = AirResults();
    fields_.reset();
    surface_.reset();
}

void AirflowSim::createSolver(const flow::Grid& grid) {
    flow::Params p;
    p.uLat = AIR_U_LAT;
    p.nuLat = nuLat;
    p.rr = true;
    p.wallModel = wallModel;
    p.belt = true;
    p.half = !std::getenv("PARTS_SIM_LBM_FP32");  // diagnostics: 32-bit populations
    p.timing = opts.timing;
    if (useGPU_) {
        solver_ = flow::makeGpu(grid, p);
        engine = "GPU";
    } else {
        solver_ = flow::makeCpu(grid, p);
        engine = "CPU";
    }
}

void AirflowSim::loadFrozen(std::shared_ptr<const Part> p, const AirflowOptions& o, FrozenResult f) {
    stopThread();
    solver_.reset();
    frozen = false;
    if (!p || !(f.h > 0) || !(o.toMeters > 0) || !(dot(o.dir, o.dir) > 0)) throw std::runtime_error("The stored airflow result is not usable: speed, density or scale.");
    if (f.cp.size() != size_t(p->nVert)) throw std::runtime_error("The stored airflow result is not usable: the pressure array does not match the part.");
    TunnelOptions to;
    to.h = f.h;
    to.ground = o.ground;
    const TunnelPlan plan = planTunnel(*p, o.dir, 0, to);
    if (plan.dims != f.dims) throw std::runtime_error("The stored airflow result is not usable: the tunnel no longer matches the part.");
    for (int a = 0; a < 3; a++)
        if (f.fields.dims[a] != (f.dims[a] + f.fields.factor - 1) / f.fields.factor) throw std::runtime_error("The stored airflow result is not usable: the coarse grid does not fit the tunnel.");
    flow::Grid grid = flow::buildGrid(plan.spec(), plan.q, p->tris);
    part = std::move(p);
    opts = o;
    basis = plan.basis;
    dims = plan.dims;
    h = plan.h;
    origin = plan.origin;
    kind = std::move(grid.kind);
    partRange = {{{(plan.mn[1] - origin[1]) / h, (plan.mx[1] - origin[1]) / h}, {(plan.mn[2] - origin[2]) / h, (plan.mx[2] - origin[2]) / h}}};
    hm_ = h * o.toMeters;
    nuAir = f.nuAir; nuLat = f.nuLat; reynolds = f.reynolds; reynoldsLength = f.reynoldsLength; simReynolds = f.simReynolds;
    wallModel = f.wallModel; groundGap = f.groundGap; engine = f.engine;
    q_ = f.q; frontal_ = f.frontal;
    records = 0;
    useGPU_ = false;
    frozenCp_ = std::move(f.cp);
    f.fields.inst = f.fields.avg;
    auto snap = std::make_shared<AirSnapshot>();
    snap->fields = std::make_shared<const flow::Fields>(std::move(f.fields));
    snap->surface = std::make_shared<const std::vector<float>>();  // non-null marks "pressure available"
    snap->steps = f.steps;
    snap->samples = snap->fields->samples;
    snap->mlups = f.mlups;
    snap->developing = f.developing;
    snap->converged = f.converged;
    snap->results = f.results;
    frozen = true;
    {
        std::lock_guard<std::mutex> lock(snapMutex_);
        snap_ = snap;
    }
}

void AirflowSim::start() {
    if (!solver_ || running_ || frozen) return;
    stopThread();
    running_ = true;
    converged_ = false;
    thread_ = std::thread([this] { loop(); });
}

void AirflowSim::pause() {
    running_ = false;
    stopThread();
}

void AirflowSim::stopThread() {
    running_ = false;
    if (thread_.joinable() && thread_.get_id() != std::this_thread::get_id()) thread_.join();
}

void AirflowSim::reset() {
    const bool was = running_;
    pause();
    if (!solver_) return;
    solver_->reset();
    sx_.clear(); sy_.clear(); sz_.clear(); sfx_.clear();
    developing_ = true;
    converged_ = false;
    steps_ = 0;
    mlups_ = 0;
    results_ = AirResults();
    fields_.reset();
    surface_.reset();
    {
        std::lock_guard<std::mutex> lock(snapMutex_);
        snap_.reset();
    }
    if (onSnapshot) onSnapshot();
    if (was) start();
}

std::shared_ptr<const AirSnapshot> AirflowSim::snapshot() const {
    std::lock_guard<std::mutex> lock(snapMutex_);
    return snap_;
}

// The solver thread. On the GPU two batches are in flight, so the GPU always has the next one queued
// while this thread reads the last one's forces (a few bytes); batches aim at ~40 ms of work, so the
// desktop keeps drawing. The view fields and surface pressure are read back now and then.
void AirflowSim::loop() {
    flow::Solver& sim = *solver_;
    const double N = double(sim.cells);
    int batch = std::clamp(int(2e7 / N), 1, 256);
    struct Pending { int ticket, steps; };
    std::deque<Pending> inFlight;
    auto lastDone = Clock::now();
    auto lastFields = lastDone - std::chrono::seconds(10), lastSurface = lastFields;
    auto busyFrom = lastDone, rateFrom = lastDone;
    double gpuFrom = sim.gpuSeconds;
    int64_t rateSteps = 0;
    try {
        while (running_) {
            flow::Forces f;
            if (useGPU_) {
                inFlight.push_back({sim.submit(batch), batch});
                if (inFlight.size() < 2) continue;
                const Pending p = inFlight.front();
                inFlight.pop_front();
                f = sim.collect(p.ticket);
                // the time per completed batch (with the next one queued behind it; next to nothing
                // when the GPU finished it while this thread read results). Batches at most double
                // at a time: one that ran for seconds would have the OS reset the GPU.
                const double dt = std::max(1e-4, seconds(lastDone));
                lastDone = Clock::now();
                batch = std::clamp(int(std::lround(p.steps * BATCH_SECONDS / dt)), std::max(1, p.steps / 2), std::min(256, 2 * p.steps));
                rateSteps += p.steps;
            } else {
                const auto t0 = Clock::now();
                sim.step(batch);
                f = sim.takeForces();
                const double dt = std::max(1e-4, seconds(t0));
                rateSteps += batch;
                batch = std::clamp(int(std::lround(batch * BATCH_SECONDS / dt)), 1, 1000);
            }
            int64_t at = sim.steps;
            for (const auto& p : inFlight) at -= p.steps;
            // speed over the last half second or so (a single batch's time says little)
            if (seconds(rateFrom) > 0.5) {
                mlups_ = N * double(rateSteps) / seconds(rateFrom) / 1e6;
                rateSteps = 0;
                rateFrom = Clock::now();
            }
            if (sim.timed && seconds(busyFrom) > 1) {
                gpuBusy_ = (sim.gpuSeconds - gpuFrom) / seconds(busyFrom);
                gpuFrom = sim.gpuSeconds;
                busyFrom = Clock::now();
            }
            onForces(f, at, mlups_);
            if (running_ && seconds(lastFields) > FIELDS_EVERY) {
                fields_ = std::make_shared<const flow::Fields>(sim.fields());
                lastFields = Clock::now();
            }
            if (running_ && seconds(lastSurface) > SURFACE_EVERY) {
                surface_ = std::make_shared<const std::vector<float>>(sim.surfaceRho());
                lastSurface = Clock::now();
            }
            publish(false);
            // leave the cores to other work (a bend test) in between CPU batches
            if (!useGPU_) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // the batches still in flight are part of the state
        for (const auto& p : inFlight) sim.collect(p.ticket);
        // the latest fields and surface for the paused view
        fields_ = std::make_shared<const flow::Fields>(sim.fields());
        surface_ = std::make_shared<const std::vector<float>>(sim.surfaceRho());
        publish(true);
    } catch (const std::exception& e) {
        running_ = false;
        // start again from rest next time
        try { solver_->reset(); } catch (const std::exception&) {}
        sx_.clear(); sy_.clear(); sz_.clear(); sfx_.clear();
        developing_ = true;
        steps_ = 0;
        results_ = AirResults();
        fields_.reset();
        surface_.reset();
        {
            std::lock_guard<std::mutex> lock(snapMutex_);
            snap_.reset();
        }
        if (onSnapshot) onSnapshot();
        if (onStatus) onStatus(std::string("Airflow stopped: ") + e.what(), true);
    }
}

// A batch of forces (lattice units, summed over f.steps steps) ending at step `steps`.
void AirflowSim::onForces(const flow::Forces& f, int64_t steps, double mlups) {
    steps_ = steps;
    mlups_ = mlups;
    if (!f.steps) return;
    // mean force in the units of the pressure coefficient's area (cells^2)
    const double k = 1 / (0.5 * AIR_U_LAT * AIR_U_LAT * double(f.steps));
    const double C[3] = {f.me[0] * k, f.me[1] * k, f.me[2] * k};
    if (!std::isfinite(C[0] + C[1] + C[2]))
        throw std::runtime_error("the flow became unstable (the forces are no longer finite). Use more cells, a lower wind speed, or the laminar boundary layer setting.");
    const double flowThrough = dims[0] / AIR_U_LAT;
    if (steps < flow::RAMP_STEPS + 1.5 * flowThrough) developing_ = true;
    else {
        if (developing_) {
            // the flow has developed: start the averages
            developing_ = false;
            sx_.clear(); sy_.clear(); sz_.clear(); sfx_.clear();
            solver_->resetAverages();
            averageFrom_ = steps;
        }
        sx_.push_back(C[0]);
        sy_.push_back(C[1]);
        sz_.push_back(C[2]);
        sfx_.push_back(f.friction[0] * k);
    }
    // forces in newtons, and once averaging, the mean over the samples with its 95% interval
    const double kN = q_ * hm_ * hm_;
    double mean[3] = {C[0], C[1], C[2]}, ci[3] = {AirResults::NaN, AirResults::NaN, AirResults::NaN}, friction = AirResults::NaN;
    const bool averaged = sx_.size() >= 8;
    if (averaged) {
        const std::vector<double>* s[3] = {&sx_, &sy_, &sz_};
        for (int a = 0; a < 3; a++) {
            const auto b = batchMeans(*s[a]);
            mean[a] = b.mean;
            ci[a] = b.ci;
        }
        friction = batchMeans(sfx_).mean * kN;
    }
    const double frontal = std::max(1.0, frontal_);
    AirResults& r = results_;
    r.drag = mean[0] * kN;
    r.lift = mean[1] * kN;
    r.side = mean[2] * kN;
    r.dragCI = ci[0] * kN;
    r.liftCI = ci[1] * kN;
    r.sideCI = ci[2] * kN;
    r.frictionDrag = friction;
    r.cd = mean[0] / frontal;
    r.cl = mean[1] / frontal;
    r.cdCI = ci[0] / frontal;
    r.clCI = ci[1] / frontal;
    r.frontalArea = frontal * hm_ * hm_;
    r.averaged = averaged;
    for (int d = 0; d < 3; d++) r.force[d] = basis[0][d] * r.drag + basis[1][d] * r.lift + basis[2][d] * r.side;
    // settled: the drag and lift within about 1% (or 0.005 and 0.01 in coefficient) at 95% confidence,
    // after at least a flow-through of averaging
    if (averaged && steps - averageFrom_ > flowThrough && !converged_) {
        const bool settled = r.cdCI <= std::max(0.01 * std::abs(r.cd), 0.005) && r.clCI <= std::max(0.02 * std::abs(r.cl), 0.01);
        if (settled) {
            converged_ = true;
            if (autoStop && running_) {
                running_ = false;
                if (onStatus) onStatus("Airflow converged: the forces have settled (95% confidence interval within about 1%).", false);
            }
        }
    }
}

void AirflowSim::publish(bool notify) {
    auto s = std::make_shared<AirSnapshot>();
    s->fields = fields_;
    s->surface = surface_;
    s->steps = steps_;
    s->samples = int(sx_.size());
    s->mlups = mlups_;
    s->gpuBusy = gpuBusy_;
    s->developing = developing_;
    s->converged = converged_;
    s->results = results_;
    {
        std::lock_guard<std::mutex> lock(snapMutex_);
        snap_ = std::move(s);
    }
    // the UI redraws its numbers a few times a second, and at once when the phase changes
    const int phase = developing_ ? 0 : converged_ ? 2 : 1;
    if (notify || phase != notifiedPhase_ || seconds(lastNotify_) > 0.1) {
        lastNotify_ = Clock::now();
        notifiedPhase_ = phase;
        if (onSnapshot) onSnapshot();
    }
}

// ---------- sampling ----------

std::array<double, 3> AirflowSim::toLattice(const Vec3& p) const {
    return {(dot(p, basis[0]) - origin[0]) / h, (dot(p, basis[1]) - origin[1]) / h, (dot(p, basis[2]) - origin[2]) / h};
}

Vec3 AirflowSim::toWorld(double x, double y, double z) const {
    const double a = origin[0] + x * h, b = origin[1] + y * h, c = origin[2] + z * h;
    return {basis[0][0] * a + basis[1][0] * b + basis[2][0] * c, basis[0][1] * a + basis[1][1] * b + basis[2][1] * c,
            basis[0][2] * a + basis[1][2] * b + basis[2][2] * c};
}

double AirflowSim::sample(const flow::Fields& fl, const std::vector<float>& field, double x, double y, double z, double out[4]) {
    const int nx = fl.dims[0], ny = fl.dims[1], nz = fl.dims[2], f = fl.factor;
    const double gx = x / f - 0.5, gy = y / f - 0.5, gz = z / f - 0.5;
    const int i0 = int(std::floor(gx)), j0 = int(std::floor(gy)), k0 = int(std::floor(gz));
    const double tx = gx - i0, ty = gy - j0, tz = gz - k0;
    for (int c = 0; c < 4; c++) out[c] = 0;
    double wsum = 0;
    for (int q = 0; q < 8; q++) {
        const int i = i0 + (q & 1), j = j0 + ((q >> 1) & 1), k = k0 + ((q >> 2) & 1);
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) continue;
        const int64_t cell = i + int64_t(nx) * (j + int64_t(ny) * k);
        if (field[4 * cell] == -2.0f) continue;
        const double w = (q & 1 ? tx : 1 - tx) * ((q >> 1) & 1 ? ty : 1 - ty) * ((q >> 2) & 1 ? tz : 1 - tz);
        for (int c = 0; c < 4; c++) out[c] += w * field[4 * cell + c];
        wsum += w;
    }
    if (wsum > 0)
        for (int c = 0; c < 4; c++) out[c] /= wsum;
    return wsum;
}

bool AirflowSim::isSolidAt(double x, double y, double z) const {
    const int i = int(std::floor(x)), j = int(std::floor(y)), k = int(std::floor(z));
    if (i < 0 || j < 0 || k < 0 || i >= dims[0] || j >= dims[1] || k >= dims[2]) return true;
    return kind[i + int64_t(dims[0]) * (j + int64_t(dims[1]) * k)] == flow::SOLID;
}

std::vector<float> AirflowSim::surfaceCp(const AirSnapshot& s) const {
    if (frozen) return frozenCp_;
    const Part& p = *part;
    std::vector<float> out(p.nVert, std::numeric_limits<float>::quiet_NaN());
    if (!s.surface) return out;
    const auto& rho = *s.surface;
    const double inv = 1 / (3 * 0.5 * AIR_U_LAT * AIR_U_LAT);
    for (int v = 0; v < p.nVert && v < int(vertexRecords_.size()); v++) {
        const int32_t r = vertexRecords_[v];
        if (r >= 0 && size_t(r) < rho.size()) out[v] = float(rho[r] * inv);
    }
    return out;
}

std::vector<float> AirflowSim::triangleForces(const AirSnapshot& s) const {
    const Part& p = *part;
    const auto cp = surfaceCp(s);
    std::vector<float> F(3 * size_t(p.nTri), 0.f);
    const double a2 = opts.toMeters * opts.toMeters;
    for (int t = 0; t < p.nTri; t++) {
        double sum = 0;
        int c = 0;
        for (int k = 0; k < 3; k++) {
            const float v = cp[p.tris[3 * t + k]];
            if (!std::isnan(v)) { sum += v; c++; }
        }
        if (!c) continue;
        const double pa = sum / c * q_;
        const double A = p.triArea[t] * a2;
        for (int d = 0; d < 3; d++) F[3 * t + d] = float(-pa * A * p.triNormal[3 * t + d]);
    }
    return F;
}

}  // namespace ps
