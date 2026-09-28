#include "static_study.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <mutex>
#include <numeric>
#include <stdexcept>

#include "../gpu/gpu.hpp"
#include "../gpu/gpu_fea.hpp"
#include "../util/trace.hpp"

namespace ps {

std::vector<uint8_t> heldNodes(const std::vector<uint8_t>& bc) {
    std::vector<uint8_t> held(bc.size() / 3);
    for (size_t n = 0; n < held.size(); n++) held[n] = bc[3 * n] | bc[3 * n + 1] | bc[3 * n + 2];
    return held;
}

double lostLoadFraction(const VoxelFEA& fea, const std::vector<double>& f) {
    const auto& act = fea.levels[0].activeNode;
    double lost = 0, total = 0;
    for (size_t n = 0; n < act.size(); n++) {
        const double m = std::sqrt(f[3 * n] * f[3 * n] + f[3 * n + 1] * f[3 * n + 1] + f[3 * n + 2] * f[3 * n + 2]);
        total += m;
        if (!act[n]) lost += m;
    }
    return total > 0 ? lost / total : 0;
}

namespace {

struct Solved {
    std::unique_ptr<VoxelFEA> fea;
    std::unique_ptr<GpuFeaSolver> gpu;
    int64_t coarsest = 0;
    SolveResult sol;
    std::string engine, gpuNote;
};

// The last linear static model (multigrid hierarchy and GPU solver), which the next run takes over
// when it solves the same mesh again (only the loads or the modulus changed): building them is a
// tenth to a fifth of such a run on large parts. Never destroyed (its GPU buffers must not outlive
// the GPU context at exit); dropKeptModel() frees it.
struct KeptModel {
    std::mutex lock;
    std::array<int, 3> dims{};
    std::vector<float> density;
    std::vector<uint8_t> bc;
    double nu = 0;
    int64_t coarsest = 0;
    std::unique_ptr<VoxelFEA> fea;
    std::unique_ptr<GpuFeaSolver> gpu;
};
KeptModel& kept() {
    static KeptModel* k = new KeptModel;
    return *k;
}

// build the multigrid hierarchy (or take the kept one when `reuse`) and solve: GPU when allowed and
// working, else all CPU cores
Solved buildAndSolve(const StructuralInput& in, const std::vector<float>& density, const std::vector<double>* x0, const std::string& stage,
                     const ProgressFn& progress, bool& gpuFailed, std::string& gpuNote, bool reuse = false) {
    Solved s;
    const bool gpu = in.useGPU && !gpuFailed && gpuAvailable();
    const int64_t coarsest = gpu ? GPU_COARSEST_DOF : 1100;
    TraceTimer trace("multigrid build");
    if (reuse) {
        KeptModel& k = kept();
        std::lock_guard<std::mutex> g(k.lock);
        if (k.fea && k.dims == in.dims && k.nu == in.nu && k.coarsest == coarsest && k.density == density && k.bc == in.bc) {
            s.fea = std::move(k.fea);
            s.gpu = std::move(k.gpu);
            trace.lap("  kept from the last run");
        }
    }
    if (!s.fea) s.fea = std::make_unique<VoxelFEA>(in.dims, density, in.nu, in.bc, coarsest);
    s.coarsest = coarsest;
    trace.lap("GPU setup");
    SolveOptions o;
    o.tol = 1e-6;
    o.x0 = x0;
    bool cancelled = false;
    auto report = [&](const std::string& label) {
        return [&, label](int it, double res) {
            const double frac = std::min(1.0, std::log10(std::max(res, 1e-7)) / -6);
            cancelled = progress && progress(frac, stage + " on the " + label + " · iteration " + std::to_string(it));
            return cancelled;
        };
    };
    if (gpu) {
        try {
            if (!s.gpu) s.gpu = std::make_unique<GpuFeaSolver>(*s.fea);
            if (std::getenv("PARTS_SIM_PROFILE")) std::fputs(s.gpu->profile(50).c_str(), stderr);
            trace.lap("GPU solve");
            o.maxIter = 3000;
            o.onProgress = report("GPU");
            s.sol = s.gpu->solve(in.f, o);
            if (s.sol.cancelled) throw std::runtime_error("Cancelled");
            if (s.sol.converged) {
                s.engine = "GPU";
                return s;
            }
            gpuFailed = true;
            gpuNote = "the GPU solve did not converge";
        } catch (const std::exception& e) {
            if (cancelled) throw;
            gpuFailed = true;
            gpuNote = e.what();
            s.gpu.reset();
        }
    }
    if (std::getenv("PARTS_SIM_PROFILE")) std::fputs(s.fea->profile().c_str(), stderr);
    trace.lap("CPU solve");
    o.maxIter = 400;
    o.onProgress = report("CPU");
    s.sol = s.fea->solve(in.f, o);
    s.engine = "CPU";
    s.gpuNote = gpuNote;
    return s;
}

double maxDisp(const std::vector<double>& u) {
    double best = 0;
    for (size_t i = 0; i + 2 < u.size(); i += 3) best = std::max(best, u[i] * u[i] + u[i + 1] * u[i + 1] + u[i + 2] * u[i + 2]);
    return std::sqrt(best);
}

}  // namespace

void dropKeptModel() {
    KeptModel& k = kept();
    std::unique_ptr<GpuFeaSolver> gpu;
    std::unique_ptr<VoxelFEA> fea;
    {
        std::lock_guard<std::mutex> g(k.lock);
        gpu = std::move(k.gpu);
        fea = std::move(k.fea);
        k.density = {};
        k.bc = {};
    }
}

StaticResult solveStatic(const StructuralInput& in, const ProgressFn& progress) {
    TraceTimer trace("prune");
    const auto held = heldNodes(in.bc);
    auto pruned = pruneFloating(in.dims, in.density, held);
    if (progress && progress(0, "Building multigrid")) throw std::runtime_error("Cancelled");
    bool gpuFailed = false;
    std::string gpuNote;
    trace.lap("build + solve");
    Solved s = buildAndSolve(in, pruned.density, nullptr, "Solving", progress, gpuFailed, gpuNote, true);
    trace.lap("stresses");
    if (s.sol.cancelled) throw std::runtime_error("Cancelled");
    const double lost = lostLoadFraction(*s.fea, in.f);
    if (!s.sol.converged)
        throw std::runtime_error("The structural solve did not converge. Enlarge the support area, check disconnected parts, or change the mesh resolution.");
    if (lost > 0.99) throw std::runtime_error("The applied load is on geometry disconnected from the supports.");
    StaticResult r;
    std::vector<double> u(s.sol.u.size());
    const double scale = 1 / (in.E * in.h);
    for (size_t i = 0; i < u.size(); i++) u[i] = s.sol.u[i] * scale;
    auto st = s.fea->stresses(u, in.E, in.h);
    trace.lap("result");
    r.u.assign(u.begin(), u.end());
    r.nodeVM = std::move(st.nodeVM);
    r.nodeP1 = std::move(st.nodeP1);
    r.nodeP3 = std::move(st.nodeP3);
    r.activeNode = s.fea->levels[0].activeNode;
    r.density = std::move(pruned.density);
    r.iterations = s.sol.iterations;
    r.residual = s.sol.residual;
    r.converged = s.sol.converged;
    r.removed = pruned.removed;
    r.lostLoad = lost;
    r.reaction = s.fea->reactions(s.sol.u, &in.f);
    r.levels = int(s.fea->levels.size());
    r.voxels = int(s.fea->levels[0].elems.size());
    r.engine = s.engine;
    r.gpuNote = s.engine == "CPU" && in.useGPU ? s.gpuNote : "";
    // kept for the next run (replacing the last kept model, which is freed outside the lock)
    {
        KeptModel& k = kept();
        std::lock_guard<std::mutex> g(k.lock);
        std::swap(k.fea, s.fea);
        std::swap(k.gpu, s.gpu);
        k.dims = in.dims;
        k.nu = in.nu;
        k.coarsest = s.coarsest;
        k.density = r.density;
        k.bc = in.bc;
    }
    return r;
}

BreakResult breakTest(const StructuralInput& in, double strength, bool principal, int maxSteps, const std::function<void(BreakStep&&)>& onStep,
                      const ProgressFn& progress) {
    if (!(strength > 0)) throw std::invalid_argument("Failure strength must be positive.");
    const int nx = in.dims[0], ny = in.dims[1];
    const int64_t NX = nx + 1, NY = ny + 1;
    const int64_t off[8] = {0, 1, 1 + NX, NX, NX * NY, 1 + NX * NY, 1 + NX + NX * NY, NX + NX * NY};
    const auto held = heldNodes(in.bc);
    auto pr = pruneFloating(in.dims, in.density, held);
    std::vector<float> density = pr.density;
    BreakResult out;
    out.removed = pr.removed;
    out.reason = "step limit";
    std::vector<double> x0;
    bool gpuFailed = false;
    std::string gpuNote;
    for (int step = 0; step < maxSteps; step++) {
        if (std::none_of(density.begin(), density.end(), [](float d) { return d > 0; })) {
            out.reason = step > 0 ? "separated" : "unsupported load";
            break;
        }
        Solved s = buildAndSolve(in, density, x0.empty() ? nullptr : &x0, "Break test step " + std::to_string(step + 1), progress, gpuFailed, gpuNote);
        if (s.sol.cancelled) throw std::runtime_error("Cancelled");
        out.engine = s.engine;
        const Level& L = s.fea->levels[0];
        if (L.elems.empty() || lostLoadFraction(*s.fea, in.f) > 0.5) {
            out.reason = step > 0 ? "separated" : "unsupported load";
            break;
        }
        if (!s.sol.converged) {
            out.reason = "solver did not converge";
            break;
        }
        std::vector<double> u(s.sol.u.size());
        const double scale = 1 / (in.E * in.h);
        for (size_t i = 0; i < u.size(); i++) u[i] = s.sol.u[i] * scale;
        auto st = s.fea->stresses(u, in.E, in.h);
        x0 = s.sol.u;
        const auto& nodal = principal ? st.nodeP1 : st.nodeVM;
        // voxel failure measure: the worst of its corner (nodal) stresses
        std::vector<float> measure(L.elems.size());
        double smax = 0;
        for (size_t q = 0; q < L.elems.size(); q++) {
            float m = 0;
            for (int a = 0; a < 8; a++) m = std::max(m, nodal[L.base[q] + off[a]]);
            measure[q] = m;
            smax = std::max(smax, double(m));
        }
        if (!(smax > 0)) {
            out.reason = "no stress";
            break;
        }
        const double lambda = strength / smax;
        out.peak = std::max(out.peak, lambda);
        const size_t cap = std::max<size_t>(1, size_t(std::lround(0.02 * std::pow(double(L.elems.size()), 2.0 / 3))));
        std::vector<int> order(L.elems.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) { return measure[a] > measure[b]; });
        BreakStep bs;
        bs.step = step;
        bs.lambda = lambda;
        bs.maxDisp = maxDisp(u) * lambda;
        for (int q : order) {
            if (bs.cracked.size() >= cap || measure[q] < 0.95 * smax) break;
            bs.cracked.push_back(L.elems[q]);
        }
        std::vector<float> next = density;
        for (int e : bs.cracked) next[e] = 0;
        auto pruned = pruneFloating(in.dims, next, held);
        for (size_t e = 0; e < next.size(); e++)
            if (next[e] > 0 && !(pruned.density[e] > 0)) bs.detached.push_back(int(e));
        density = std::move(pruned.density);
        bs.u.resize(u.size());
        for (size_t i = 0; i < u.size(); i++) bs.u[i] = float(u[i] * lambda);
        bs.nodeVM = st.nodeVM;
        for (auto& v : bs.nodeVM) v = float(v * lambda);
        bs.activeNode = L.activeNode;
        bs.voxels = int(L.elems.size());
        onStep(std::move(bs));
    }
    return out;
}

}  // namespace ps
