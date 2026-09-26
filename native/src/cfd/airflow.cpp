#include "airflow.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

#include "../core/voxelize.hpp"
#include "../gpu/gpu.hpp"
#include "../util/parallel.hpp"
#include "../util/system.hpp"

namespace ps {

namespace {

constexpr double NU_AIR = 1.5e-5;  // m^2/s
constexpr double MIN_NU_LAT = 0.0005;
// memory per cell: GPU f-in / f-out (2 x 19 floats), moments, solid flag, wall links, read-back copy
constexpr double GPU_BYTES = 2 * 76 + 16 + 4 + 19 + 16;
// host: solid flag, wall links, up to three snapshots (moments + averages), a read-back copy, the voxelizer
constexpr double HOST_BYTES = 1 + 19 + 3 * 32 + 16 + 4;
constexpr double CPU_SOLVER_BYTES = 2 * 76 + 16;  // the CPU solver's own distributions and moments

double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
Vec3 normalize(Vec3 v) {
    const double l = std::sqrt(dot(v, v));
    for (auto& x : v) x /= l;
    return v;
}
Vec3 cross(const Vec3& a, const Vec3& b) { return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}; }

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

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
    c.gpuBytesPerCell = gpu ? GPU_BYTES : 0;
    c.ramBytesPerCell = HOST_BYTES + (gpu ? (unified ? GPU_BYTES : 0) : CPU_SOLVER_BYTES);
    double ram = double(physicalMemory());
    if (!(ram > 0)) ram = 8e9;
    // leave half the memory to the system, the app and the other studies
    const double ramCap = 0.5 * ram / c.ramBytesPerCell;
    c.minCells = 50e3;
    if (gpu) {
        c.maxCells = std::min(ramCap, 0.95 * double(lbmGpuMaxCells()));
        c.defaultCells = 1e6;
    } else {
        // the CPU manages ~70 million cell updates a second: bigger grids take minutes per flow-through
        c.maxCells = std::min(ramCap, 8e6);
        c.defaultCells = 3e5;
    }
    c.maxCells = std::max(c.maxCells, c.minCells);
    c.defaultCells = std::clamp(c.defaultCells, c.minCells, c.maxCells);
    return c;
}

TunnelPlan planTunnel(const Part& part, const Vec3& dir, double cells) {
    TunnelPlan t;
    const Vec3 e1 = normalize(dir);
    const Vec3 ref = std::abs(e1[1]) < 0.9 ? Vec3{0, 1, 0} : Vec3{1, 0, 0};
    const double r = dot(ref, e1);
    const Vec3 e2 = normalize({ref[0] - r * e1[0], ref[1] - r * e1[1], ref[2] - r * e1[2]});
    t.basis = {e1, e2, cross(e1, e2)};
    // part extent in the wind frame
    t.mn = {INFINITY, INFINITY, INFINITY};
    t.mx = {-INFINITY, -INFINITY, -INFINITY};
    const auto& V = part.vertices;
    for (size_t i = 0; i + 2 < V.size(); i += 3)
        for (int a = 0; a < 3; a++) {
            const double v = V[i] * t.basis[a][0] + V[i + 1] * t.basis[a][1] + V[i + 2] * t.basis[a][2];
            t.mn[a] = std::min(t.mn[a], v);
            t.mx[a] = std::max(t.mx[a], v);
        }
    const double L[3] = {t.mx[0] - t.mn[0], t.mx[1] - t.mn[1], t.mx[2] - t.mn[2]};
    // Tunnel margins scale with the blockage size (square root of the cross-section) and the
    // length, not the largest dimension: a long-span wing needs room around its chord, and
    // sizing by span would leave its thickness only a cell or two across.
    const double B = std::max(std::sqrt(L[1] * L[2]), 0.3 * L[0]);
    if (!std::isfinite(L[0] + L[1] + L[2]) || !(B > 0)) throw std::runtime_error("The part has no finite volume to mesh.");
    const double up = 1.5 * B, down = 2.5 * B, side = 0.9 * B;
    const double D[3] = {L[0] + up + down, L[1] + 2 * side, L[2] + 2 * side};
    const double budget = std::max(cells, 4096.0);
    double hh = std::cbrt(D[0] * D[1] * D[2] / budget);
    auto dimsFor = [&](double hv) {
        std::array<int, 3> d;
        for (int a = 0; a < 3; a++) d[a] = std::max(16, int(std::ceil(D[a] / hv)));
        return d;
    };
    t.dims = dimsFor(hh);
    // rounding up and the minimum per axis must not exceed the budget on thin or long parts
    while (double(t.dims[0]) * t.dims[1] * t.dims[2] > budget) {
        hh *= std::max(1.01, std::cbrt(double(t.dims[0]) * t.dims[1] * t.dims[2] / budget));
        t.dims = dimsFor(hh);
    }
    t.h = hh;
    t.origin = {t.mn[0] - up, t.mn[1] - (t.dims[1] * hh - L[1]) / 2, t.mn[2] - (t.dims[2] * hh - L[2]) / 2};
    return t;
}

void AirflowSim::setup(std::shared_ptr<const Part> p, const AirflowOptions& o, const Progress& progress) {
    stopThread();
    solver_.reset();
    snap_.reset();
    recoveries_ = 0;
    if (!(o.speed > 0) || !(o.airDensity > 0) || !(o.toMeters > 0) || !std::isfinite(o.speed) || !std::isfinite(o.airDensity))
        throw std::invalid_argument("Wind speed, air density and unit scale must be finite positive numbers.");
    if (!(dot(o.dir, o.dir) > 0)) throw std::invalid_argument("Wind direction must be a finite nonzero vector.");
    if (!(o.cells > 0) || !std::isfinite(o.cells)) throw std::invalid_argument("The flow grid size must be a positive number of cells.");
    part = std::move(p);
    opts = o;
    useGPU_ = o.engine != 2 && gpuAvailable();
    const auto cap = airflowCapacity(useGPU_);
    const TunnelPlan plan = planTunnel(*part, o.dir, std::clamp(o.cells, cap.minCells, cap.maxCells));
    basis = plan.basis;
    dims = plan.dims;
    h = plan.h;
    origin = plan.origin;
    const auto& mn = plan.mn;
    const auto& mx = plan.mx;
    const double L[3] = {mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2]};
    // part in the wind frame
    const auto& V = part->vertices;
    std::vector<float> q(V.size());
    for (size_t i = 0; i + 2 < V.size(); i += 3)
        for (int a = 0; a < 3; a++) q[i + a] = float(V[i] * basis[a][0] + V[i + 1] * basis[a][1] + V[i + 2] * basis[a][2]);
    const int64_t N = int64_t(dims[0]) * dims[1] * dims[2];
    char buf[160];
    std::snprintf(buf, sizeof buf, "Meshing wind tunnel (%d x %d x %d = %.2fM cells)...", dims[0], dims[1], dims[2], N / 1e6);
    if (progress && progress(buf)) throw std::runtime_error("Cancelled");
    Grid grid;
    grid.origin = {origin[0], origin[1], origin[2]};
    grid.h = h;
    grid.dims = dims;
    const auto frac = voxelize(q, part->tris, grid, 2);
    solid.assign(N, 0);
    int64_t solidCount = 0;
    for (int64_t c = 0; c < N; c++)
        if (frac[c] >= 0.5f) { solid[c] = 1; solidCount++; }
    if (!solidCount) throw std::runtime_error("The part is too thin for this flow grid. Move the grid-size slider to more cells.");
    if (progress && progress("Measuring wall distances for curved surfaces...")) throw std::runtime_error("Cancelled");
    links_ = wallLinks(q, part->tris, origin, h, dims, solid);
    partRange = {{{(mn[1] - origin[1]) / h, (mx[1] - origin[1]) / h}, {(mn[2] - origin[2]) / h, (mx[2] - origin[2]) / h}}};
    hm_ = h * o.toMeters;
    const double Lc = std::max(L[1], L[2]) * o.toMeters;
    reynolds = o.speed * Lc / NU_AIR;
    const double LcCells = std::max(L[1], L[2]) / h;
    nuLat = std::max(AIR_U_LAT * LcCells / reynolds, MIN_NU_LAT);
    simReynolds = AIR_U_LAT * LcCells / nuLat;
    q_ = 0.5 * o.airDensity * o.speed * o.speed;
    if (progress && progress("Starting the flow solver...")) throw std::runtime_error("Cancelled");
    try {
        createSolver();
    } catch (const std::exception& e) {
        if (!useGPU_) throw;
        // the GPU could not take it: rebuild a smaller grid for the CPU
        if (onStatus) onStatus(std::string("GPU flow solver failed (") + e.what() + "); rebuilding a smaller CPU grid.", false);
        AirflowOptions cpu = o;
        cpu.engine = 2;
        cpu.cells = std::min(o.cells, 1e6);
        setup(part, cpu, progress);
    }
}

void AirflowSim::createSolver() {
    LbmSetup s;
    s.dims = dims;
    s.solid = solid;
    s.links = links_;
    s.uLat = AIR_U_LAT;
    s.nuLat = nuLat;
    if (useGPU_) {
        solver_ = makeLbmGpu(s);
        engine = "GPU";
    } else {
        solver_ = std::make_unique<LbmCpu>(s);
        engine = "CPU";
    }
}

void AirflowSim::start() {
    if (!solver_ || running_) return;
    stopThread();
    running_ = true;
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

void AirflowSim::loop() {
    int batch = 5;
    double readInterval = 0.2;
    auto lastRead = Clock::now() - std::chrono::seconds(1);
    try {
        while (running_) {
            const auto t0 = Clock::now();
            solver_->step(batch);
            const double dt = std::max(1e-4, seconds(t0));
            // let other GPU / CPU work (a structural solve) in between batches
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            const double mlups = double(solver_->cells) * batch / (dt * 1e6);
            // batches of ~40 ms keep pausing responsive
            batch = std::clamp(int(std::lround(batch * 0.04 / dt)), 1, 400);
            // big grids take a while to read back and average: read less often so the solver keeps
            // most of its time (at least 80%) for stepping
            if (seconds(lastRead) > readInterval) {
                lastRead = Clock::now();
                const auto r0 = Clock::now();
                const bool ok = process(solver_->macro(), mlups);
                readInterval = std::max(0.2, 4 * seconds(r0));
                if (!ok) {
                    // unstable: double the viscosity and restart
                    if (++recoveries_ > 3) {
                        running_ = false;
                        if (onStatus) onStatus("Airflow stopped: the flow is still unstable after three retries. Reduce the wind speed or use more cells.", true);
                        return;
                    }
                    nuLat *= 2;
                    simReynolds /= 2;
                    createSolver();
                    {
                        std::lock_guard<std::mutex> lock(snapMutex_);
                        snap_.reset();
                    }
                    if (onStatus) onStatus("Flow became unstable - doubled the viscosity and restarted.", false);
                    batch = 5;
                }
            }
        }
    } catch (const std::exception& e) {
        running_ = false;
        if (onStatus) onStatus(std::string("Airflow stopped: ") + e.what(), true);
    }
}

bool AirflowSim::process(std::vector<float>&& macro, double mlups) {
    const int64_t N = int64_t(dims[0]) * dims[1] * dims[2];
    if (int64_t(macro.size()) != 4 * N) throw std::runtime_error("Invalid flow snapshot.");
    std::atomic<bool> bad{false};
    parallelFor(N, [&](int64_t lo, int64_t hi) {
        for (int64_t c = lo; c < hi && !bad; c++) {
            const float u = std::abs(macro[4 * c + 1]) + std::abs(macro[4 * c + 2]) + std::abs(macro[4 * c + 3]);
            const float rho = macro[4 * c];
            if (!(u < 0.6f) || !std::isfinite(rho) || rho <= 0) bad = true;
        }
    });
    if (bad) return false;
    auto prev = snapshot();
    auto s = std::make_shared<AirSnapshot>();
    s->steps = solver_->steps;
    s->mlups = mlups;
    const double flowThrough = dims[0] / AIR_U_LAT;
    s->developing = s->steps < LBM_RAMP_STEPS + 0.6 * flowThrough;
    s->avgRho.resize(N);
    s->avgU.resize(3 * N);
    if (!prev || s->developing || prev->avgRho.size() != size_t(N)) {
        parallelFor(N, [&](int64_t lo, int64_t hi) {
            for (int64_t c = lo; c < hi; c++) {
                s->avgRho[c] = macro[4 * c];
                for (int d = 0; d < 3; d++) s->avgU[3 * c + d] = macro[4 * c + 1 + d];
            }
        });
        s->samples = 0;
    } else {
        const float a = 1.0f / float(prev->samples + 1);
        parallelFor(N, [&](int64_t lo, int64_t hi) {
            for (int64_t c = lo; c < hi; c++) {
                s->avgRho[c] = prev->avgRho[c] + a * (macro[4 * c] - prev->avgRho[c]);
                for (int d = 0; d < 3; d++) s->avgU[3 * c + d] = prev->avgU[3 * c + d] + a * (macro[4 * c + 1 + d] - prev->avgU[3 * c + d]);
            }
        });
        s->samples = prev->samples + 1;
    }
    const auto fc = pressureForceCoefficients(s->avgRho, solid, dims, AIR_U_LAT);
    const double k = q_ * hm_ * hm_;
    auto& r = s->results;
    r.drag = fc.C[0] * k;
    r.lift = fc.C[1] * k;
    r.side = fc.C[2] * k;
    r.cd = fc.frontal > 0 ? fc.C[0] / fc.frontal : 0;
    r.cl = fc.frontal > 0 ? fc.C[1] / fc.frontal : 0;
    r.frontalArea = fc.frontal * hm_ * hm_;
    for (int d = 0; d < 3; d++) r.force[d] = basis[0][d] * r.drag + basis[1][d] * r.lift + basis[2][d] * r.side;
    s->macro = std::move(macro);
    {
        std::lock_guard<std::mutex> lock(snapMutex_);
        snap_ = std::move(s);
    }
    if (onSnapshot) onSnapshot();
    return true;
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

double AirflowSim::sample(const std::vector<float>& field, int comps, double x, double y, double z, double* out) const {
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    const double gx = x - 0.5, gy = y - 0.5, gz = z - 0.5;
    const int i0 = int(std::floor(gx)), j0 = int(std::floor(gy)), k0 = int(std::floor(gz));
    const double tx = gx - i0, ty = gy - j0, tz = gz - k0;
    for (int c = 0; c < comps; c++) out[c] = 0;
    double wsum = 0;
    for (int q = 0; q < 8; q++) {
        const int i = i0 + (q & 1), j = j0 + ((q >> 1) & 1), k = k0 + ((q >> 2) & 1);
        if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) continue;
        const int64_t cell = i + int64_t(nx) * (j + int64_t(ny) * k);
        if (solid[cell]) continue;
        const double w = (q & 1 ? tx : 1 - tx) * ((q >> 1) & 1 ? ty : 1 - ty) * ((q >> 2) & 1 ? tz : 1 - tz);
        for (int c = 0; c < comps; c++) out[c] += w * field[comps * cell + c];
        wsum += w;
    }
    if (wsum > 0)
        for (int c = 0; c < comps; c++) out[c] /= wsum;
    return wsum;
}

bool AirflowSim::isSolidAt(double x, double y, double z) const {
    const int i = int(std::floor(x)), j = int(std::floor(y)), k = int(std::floor(z));
    if (i < 0 || j < 0 || k < 0 || i >= dims[0] || j >= dims[1] || k >= dims[2]) return true;
    return solid[i + int64_t(dims[0]) * (j + int64_t(dims[1]) * k)] == 1;
}

std::vector<float> AirflowSim::surfaceCp(const AirSnapshot& s) const {
    const Part& p = *part;
    std::vector<float> out(p.nVert, std::numeric_limits<float>::quiet_NaN());
    if (s.avgRho.empty()) return out;
    const double inv = 1 / (3 * 0.5 * AIR_U_LAT * AIR_U_LAT);
    parallelFor(p.nVert, [&](int64_t lo, int64_t hi) {
        for (int64_t v = lo; v < hi; v++) {
            const Vec3 pos{p.vertices[3 * v], p.vertices[3 * v + 1], p.vertices[3 * v + 2]};
            const Vec3 n{p.vertNormal[3 * v], p.vertNormal[3 * v + 1], p.vertNormal[3 * v + 2]};
            const auto l = toLattice(pos);
            const double nl[3] = {dot(n, basis[0]), dot(n, basis[1]), dot(n, basis[2])};
            double tmp = 0;
            for (double off : {0.9, 1.6, 2.6}) {
                const double w = sample(s.avgRho, 1, l[0] + nl[0] * off, l[1] + nl[1] * off, l[2] + nl[2] * off, &tmp);
                if (w > 0.15) {
                    out[v] = float((tmp - 1) * inv);
                    break;
                }
            }
        }
    }, 1024);
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
