#include "study_runner.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

#include "../gpu/gpu.hpp"
#include "../gpu/gpu_fea.hpp"
#include "../util/parallel.hpp"
#include "eigen.hpp"
#include "explicit.hpp"
#include "nonlinear.hpp"
#include "topology.hpp"

namespace ps {

namespace {

struct Cancel : std::runtime_error {
    Cancel() : std::runtime_error("Cancelled") {}
};

void report(const ProgressFn& p, double frac, const std::string& text) {
    if (p && p(frac, text)) throw Cancel();
}

// iteration progress: fraction from the residual (1 -> 1e-6)
double residualFraction(double res) { return std::min(1.0, std::log10(std::max(res, 1e-7)) / -6); }

// the voxel model, pruned to what is connected to the fixtures unless free-floating
std::unique_ptr<VoxelFEA> buildModel(const StructuralInput& in, bool free, const std::vector<double>* diagAdd, int* removed,
                                     std::vector<float>* densityOut = nullptr) {
    std::vector<float> density = in.density;
    if (removed) *removed = 0;
    if (!free) {
        auto pr = pruneFloating(in.dims, in.density, heldNodes(in.bc));
        density = std::move(pr.density);
        if (removed) *removed = pr.removed;
    }
    auto fea = std::make_unique<VoxelFEA>(in.dims, density, in.nu, in.bc, in.useGPU && gpuAvailable() ? GPU_COARSEST_DOF : 1100, diagAdd);
    if (fea->levels[0].elems.empty()) throw std::runtime_error("No voxels are connected to the fixtures.");
    if (densityOut) *densityOut = std::move(density);
    return fea;
}

// solver engine: GPU multigrid-CG when allowed and working, otherwise the CPU multigrid
struct Engine {
    std::string name = "CPU", note;
    VoxelFEA* fea = nullptr;
    std::unique_ptr<GpuFeaSolver> gpu;

    Engine(VoxelFEA& f, bool useGPU) : fea(&f) {
        if (!useGPU) return;
        if (!gpuAvailable()) { note = "no compatible GPU"; return; }
        try {
            gpu = std::make_unique<GpuFeaSolver>(f);
            name = "GPU";
        } catch (const std::exception& e) {
            note = e.what();
        }
    }
    Block precond(const Block& R) {
        Block out;
        for (const auto& r : R) {
            Vec z(r.size());
            // one multigrid V-cycle, on the GPU when there is one
            if (gpu) gpu->precondition(r.data(), z.data());
            else fea->precondition(r.data(), z.data());
            out.push_back(std::move(z));
        }
        return out;
    }
    void precond1(const double* r, double* z) {
        if (gpu) gpu->precondition(r, z);
        else fea->precondition(r, z);
    }
    SolveResult solve(const std::vector<double>& f, SolveOptions o) {
        if (gpu) {
            auto s = gpu->solve(f, o);
            if (s.converged || s.cancelled) return s;
        }
        if (!gpu) o.maxIter = std::min(o.maxIter, 800);
        return fea->solve(f, o);
    }
};

std::vector<float>& unitShape(std::vector<float>& u) {
    double mx = 0;
    for (size_t i = 0; i + 2 < u.size(); i += 3)
        if (!std::isnan(u[i])) mx = std::max(mx, std::sqrt(double(u[i]) * u[i] + double(u[i + 1]) * u[i + 1] + double(u[i + 2]) * u[i + 2]));
    if (mx > 0)
        for (auto& x : u) x = float(x / mx);
    return u;
}

std::vector<double> physical(const std::vector<double>& u, const StructuralInput& in) {
    std::vector<double> out(u.size());
    const double s = 1 / (in.E * in.h);
    for (size_t i = 0; i < u.size(); i++) out[i] = u[i] * s;
    return out;
}

}  // namespace

// ---------- frequency and the modal basis for linear dynamics ----------

ModalRun runModal(const StructuralModel& m, const StructuralInput& in, int nev, const DynamicLoad* dynamic, double toMeters, const ProgressFn& progress) {
    ModalRun out;
    out.free = std::none_of(in.bc.begin(), in.bc.end(), [](uint8_t v) { return v; });
    nev = std::max(1, std::min(20, nev));
    double shift = 0;
    std::vector<double> diag;
    if (out.free) {
        // free-floating: K + s M keeps the preconditioner invertible; s is far below the first
        // flexible eigenvalue, and the 6 rigid-body modes come out at ~0 Hz
        report(progress, 0, "Estimating the stiffest vibration");
        VoxelFEA probe(in.dims, in.density, in.nu, in.bc, INT64_MAX);
        const auto mass = lumpedMass(probe);
        shift = 1e-6 * maxEigenvalue(probe, mass, 20);
        diag.resize(mass.size());
        for (size_t i = 0; i < mass.size(); i++) diag[i] = mass[i] * shift;
    }
    auto fea = buildModel(in, out.free, out.free ? &diag : nullptr, &out.removed);
    Engine engine(*fea, in.useGPU);
    out.engine = engine.name;
    out.gpuNote = engine.note;
    const int want = nev + (out.free ? 6 : 0);
    report(progress, 0, "Finding natural frequencies on the " + engine.name);
    auto r = naturalFrequencies(*fea, want, in.E, in.rho, in.h, shift, [&](const Block& R) { return engine.precond(R); },
                                [&](int it, double res, int conv) {
                                    return progress && progress(residualFraction(res), "Natural frequencies on the " + engine.name + " · " +
                                                                                           std::to_string(conv) + "/" + std::to_string(want) +
                                                                                           " converged · iteration " + std::to_string(it));
                                });
    const Level& L = fea->levels[0];
    const auto W = m.vertexWeights(L.activeNode);
    const auto& mass = r.mass;
    double totalMass = 0;
    for (size_t i = 0; i < mass.size(); i += 3) totalMass += mass[i];
    const double rigidTol = std::max(1e-3 * shift, 1e-12);
    const double sq = std::sqrt(in.rho * in.h * in.h * in.h);
    std::vector<size_t> kept;
    for (size_t i = 0; i < r.modes.size() && int(kept.size()) < nev; i++) {
        if (out.free && r.lambdas[i] < rigidTol) continue;
        const auto& phi = r.modes[i];
        // effective mass fractions along x, y, z
        double part[3] = {0, 0, 0};
        for (size_t n = 0; n < phi.size(); n += 3)
            for (int d = 0; d < 3; d++) part[d] += phi[n + d] * mass[n + d];
        ModalMode md;
        md.freq = r.freqs[i];
        for (int d = 0; d < 3; d++) md.eff[d] = part[d] * part[d] / totalMass;
        md.shape = interpolate(W, phi.data(), 3);
        unitShape(md.shape);
        out.modes.push_back(std::move(md));
        kept.push_back(i);
    }
    out.converged = r.converged;
    out.iterations = r.iterations;
    out.totalMass = totalMass * in.rho * in.h * in.h * in.h;
    if (dynamic && !out.free) {
        // modal basis for linear dynamics: shapes per unit modal coordinate, pattern and static response
        report(progress, 0.9, "Static response for the dynamic load on the " + engine.name);
        std::vector<double> f;
        if (dynamic->base) {
            f.assign(L.nDof, 0.0);
            const double M = in.rho * in.h * in.h * in.h;
            for (int64_t n = 0; n < L.nNodes; n++)
                for (int d = 0; d < 3; d++) f[3 * n + d] = -M * mass[3 * n + d] * dynamic->dir[d];
        } else f = in.f;
        SolveOptions so;
        so.tol = 1e-7;
        so.maxIter = 3000;
        so.onProgress = [&](int it, double res) { return progress && progress(residualFraction(res), "Static response · iteration " + std::to_string(it)); };
        const auto sol = engine.solve(f, so);
        if (sol.cancelled) throw Cancel();
        const auto uSt = physical(sol.u, in);
        const double toModel = 1 / toMeters;
        auto& b = out.basis;
        b.staticU = interpolate(W, uSt.data(), 3, toModel);
        b.staticS = interpolate(W, fea->stressTensors(uSt, in.E, in.h).data(), 6);
        for (size_t k = 0; k < kept.size(); k++) {
            const auto& phi = r.modes[kept[k]];
            std::vector<double> phys(phi.size());
            double g = 0;
            for (size_t i = 0; i < phi.size(); i++) {
                phys[i] = phi[i] / sq;
                g += phys[i] * f[i];
            }
            b.omegas.push_back(2 * M_PI * out.modes[k].freq);
            b.gamma.push_back(g);
            b.modeU.push_back(interpolate(W, phys.data(), 3, toModel));
            b.modeS.push_back(interpolate(W, fea->stressTensors(phys, in.E, in.h).data(), 6));
        }
        out.hasBasis = true;
    }
    return out;
}

// ---------- linear buckling ----------

BucklingRun runBuckling(const StructuralModel& m, const StructuralInput& in, int nev, double strength, const ProgressFn& progress) {
    BucklingRun out;
    auto fea = buildModel(in, false, nullptr, &out.removed);
    Engine engine(*fea, in.useGPU);
    out.engine = engine.name;
    out.gpuNote = engine.note;
    const std::string stage = "Pre-buckling stresses on the " + engine.name;
    report(progress, 0, stage);
    SolveOptions so;
    so.tol = 1e-8;
    so.maxIter = 3000;
    so.onProgress = [&](int it, double res) { return progress && progress(residualFraction(res), stage + " · iteration " + std::to_string(it)); };
    const auto sol = engine.solve(in.f, so);
    if (sol.cancelled) throw Cancel();
    if (!sol.converged) throw std::runtime_error("The pre-buckling solve did not converge. Check the fixtures and loads.");
    const auto u = physical(sol.u, in);
    const auto sigma = elementStresses(*fea, u, in.h);
    nev = std::max(1, std::min(8, nev));
    // Search up to 100 times the load that makes it yield (at least 1000 x the loads): a buckling
    // load far beyond yielding is never reached, and near-zero eigenvalues are slow to resolve.
    const auto st = fea->stresses(u, in.E, in.h, false);
    double vmMax = 0;
    for (float v : st.nodeVM) vmMax = std::max(vmMax, double(v));
    const double yieldFactor = vmMax > 0 && strength > 0 ? strength / vmMax : INFINITY;
    out.maxFactor = std::isfinite(yieldFactor) ? std::max(1000.0, 100 * yieldFactor) : INFINITY;
    auto r = bucklingFactors(*fea, sigma, nev, [&](const Block& R) { return engine.precond(R); }, [&](int it, double res, int conv) {
        return progress && progress(residualFraction(res), "Buckling modes on the " + engine.name + " · " + std::to_string(conv) + "/" + std::to_string(nev) +
                                                               " converged · iteration " + std::to_string(it));
    }, 1e-5, out.maxFactor);
    const auto W = m.vertexWeights(fea->levels[0].activeNode);
    out.vm = interpolate(W, st.nodeVM.data());
    for (size_t i = 0; i < r.factors.size(); i++) {
        BucklingRun::Mode md;
        md.factor = r.factors[i];
        md.shape = interpolate(W, r.modes[i].data(), 3);
        unitShape(md.shape);
        out.modes.push_back(std::move(md));
    }
    out.converged = r.converged;
    out.iterations = r.iterations;
    return out;
}

// ---------- nonlinear static ----------

NonlinearRun runNonlinear(const StructuralModel& m, const StructuralInput& in, const Material& mat, const NonlinearOptions& o, double toMeters,
                          const std::function<void(NonlinearStep&&)>& onStep, const ProgressFn& progress) {
    NonlinearRun out;
    auto fea = buildModel(in, false, nullptr, &out.removed);
    Engine engine(*fea, in.useGPU);
    out.engine = engine.name;
    out.gpuNote = engine.note;
    const Level& L = fea->levels[0];
    const auto W = m.vertexWeights(L.activeNode);
    const auto plastic = o.plasticity && !mat.brittle ? hardeningFor(mat) : std::nullopt;
    out.plastic = plastic.has_value();
    const double Eh2 = in.E * in.h * in.h;
    std::vector<double> f(in.f.size());
    for (size_t i = 0; i < f.size(); i++) f[i] = in.f[i] / Eh2;
    double fAbs = 0;
    for (int64_t i = 0; i < L.nDof; i++)
        if (!L.fixed[i]) fAbs += std::abs(in.f[i]);
    const int maxDim = std::max({in.dims[0], in.dims[1], in.dims[2]});
    const Precond pre = [&](const double* r, double* z) { engine.precond1(r, z); };
    const double toModel = in.h / toMeters;  // voxel units -> model units
    auto send = [&](double lam, const std::vector<double>& u, const NodalFields& fields, double maxDisp, int iterations, bool unloaded) {
        NonlinearStep s;
        double work = 0;
        for (int64_t i = 0; i < L.nDof; i++)
            if (!L.fixed[i]) work += in.f[i] * u[i];
        s.unloaded = unloaded;
        s.lam = lam;
        s.D = fAbs > 0 ? work / fAbs * toModel : 0;  // load-weighted displacement along the loads
        s.u = interpolate(W, u.data(), 3, toModel);
        s.vm = interpolate(W, fields.vm.data(), 1, in.E);
        s.pe = interpolate(W, fields.pe.data(), 1);
        s.p1 = interpolate(W, fields.p1.data(), 1, in.E);
        s.maxVM = 0;
        s.maxPE = 0;
        for (float v : fields.vm) s.maxVM = std::max(s.maxVM, double(v) * in.E);
        for (float v : fields.pe) s.maxPE = std::max(s.maxPE, double(v));
        s.maxDisp = maxDisp * in.h / toMeters;
        s.iterations = iterations;
        onStep(std::move(s));
    };
    NonlinearModel model(*fea, o.largeDisplacement, plastic);
    RampOptions ro;
    ro.target = o.untilFailure ? RampOptions::inf : 1;
    ro.steps = o.steps;
    ro.maxSteps = o.untilFailure ? 60 : 40;
    ro.rupture = plastic ? mat.elongation : RampOptions::inf;
    ro.crackStress = mat.brittle && o.untilFailure ? mat.uts * 1e6 / in.E : RampOptions::inf;
    // "failed" also means grossly bent: the part moved more than 15% of its size
    ro.maxDisp = o.untilFailure ? 0.15 * maxDim : RampOptions::inf;
    ro.maxTime = o.untilFailure ? 240 : RampOptions::inf;
    ro.stepAlpha = plastic ? std::max(0.002, mat.elongation / 12) : RampOptions::inf;
    ro.stepDisp = o.untilFailure ? 0.04 * maxDim : RampOptions::inf;
    ro.onIter = [&](double lam, int it, double rel) {
        char buf[96];
        std::snprintf(buf, sizeof buf, "load ×%.3f · Newton %d", lam, it);
        return progress && progress(-1, "Nonlinear on the " + engine.name + " · " + buf);
    };
    ro.onStep = [&](const RampStep& s) {
        out.steps++;
        send(s.lam, s.u, s.fields, s.maxDisp, s.iterations, false);
    };
    auto res = loadRamp(model, f, pre, ro);
    out.reason = res.reason;
    out.lam = res.lam;
    // spring-back: unload with the plastic strains kept to show the permanent deformation
    if (plastic && std::any_of(model.alpha.begin(), model.alpha.end(), [](double a) { return a > 0; })) {
        auto u = res.u;
        report(progress, -1, "Unloading (spring-back)");
        const auto un = equilibrate(model, u, std::vector<double>(f.size(), 0.0), pre, RampOptions::inf, [&](int it, double) {
            return progress && progress(-1, "Unloading (spring-back) · Newton " + std::to_string(it));
        });
        if (un.converged) {
            double dmax = 0;
            for (size_t i = 0; i + 2 < u.size(); i += 3) dmax = std::max(dmax, std::sqrt(u[i] * u[i] + u[i + 1] * u[i + 1] + u[i + 2] * u[i + 2]));
            send(0, u, model.nodalFields(u), dmax, un.iterations, true);
            out.permanent = true;
        }
    }
    return out;
}

// ---------- drop test ----------

DropRun runDrop(const StructuralModel& m, const StructuralInput& in, double height, double toMeters, const std::function<void(DropFrame&&)>& onFrame,
                const ProgressFn& progress) {
    const int nx = in.dims[0], ny = in.dims[1], nz = in.dims[2];
    const int64_t NX = nx + 1, NY = ny + 1, NZ = nz + 1, nN = NX * NY * NZ;
    VoxelFEA fea(in.dims, in.density, in.nu, std::vector<uint8_t>(3 * nN, 0), INT64_MAX);
    const Level& L = fea.levels[0];
    if (L.elems.empty()) throw std::runtime_error("The part produced no voxels.");
    // nodes that can touch the floor: active nodes with an empty neighbouring voxel
    std::vector<uint8_t> count(nN, 0);
    for (size_t q = 0; q < L.elems.size(); q++)
        for (int a = 0; a < 8; a++) count[L.base[q] + L.off[a]]++;
    DropOptions o;
    o.surface.assign(nN, 0);
    int64_t lowest = INT64_MAX;
    for (int64_t n = 0; n < nN; n++) {
        if (!count[n] || count[n] == 8) continue;
        o.surface[n] = 1;
        lowest = std::min(lowest, (n / NX) % NY);
    }
    o.nodeY.resize(nN);
    for (int64_t n = 0; n < nN; n++) o.nodeY[n] = double((n / NX) % NY - lowest) * in.h;
    const auto W = m.vertexWeights(L.activeNode);
    const double toModel = 1 / toMeters;
    DropRun out;
    for (size_t q = 0; q < L.elems.size(); q++) out.mass += L.rho[q];
    out.mass *= in.rho * in.h * in.h * in.h;
    out.speed = std::sqrt(2 * 9.81 * height);
    o.E = in.E;
    o.rho = in.rho;
    o.h = in.h;
    o.speed = out.speed;
    o.onFrame = [&](DropOptions::Frame&& fr) {
        DropFrame d;
        d.t = fr.t;
        d.force = fr.force;
        d.u = interpolate(W, fr.u.data(), 3, toModel);
        d.vm = interpolate(W, fr.vm.data());
        onFrame(std::move(d));
    };
    o.onProgress = [&](double frac) { return progress && progress(frac, "Drop test · " + std::to_string(int(std::lround(frac * 100))) + "%"); };
    out.engine = "CPU";
    const auto r = dropTestCPU(fea, o);
    out.vmMax = interpolate(W, r.vmMax.data());
    out.tPeak = interpolate(W, r.tPeak.data());
    for (const auto& p : r.history) {
        out.times.push_back(float(p.t));
        out.forces.push_back(float(p.force));
    }
    out.peakForce = r.peakForce;
    out.contactTime = r.contactTime;
    out.duration = r.duration;
    out.steps = r.steps;
    out.dt = r.dt;
    out.rebounded = r.rebounded;
    return out;
}

// ---------- topology ----------

TopologyRun runTopology(const StructuralModel&, const StructuralInput& in, const std::vector<uint8_t>& keep, double volFrac, int maxIter,
                        const std::function<void(TopologyStep&&)>& onIter, const ProgressFn& progress) {
    TopologyRun out;
    TopologyOptions o;
    o.dims = in.dims;
    o.fill = in.density;
    o.keep = keep;
    o.volFrac = volFrac;
    o.maxIter = maxIter;
    std::unique_ptr<VoxelFEA> fea;
    std::unique_ptr<Engine> engine;
    o.solve = [&](const std::vector<float>& density, const std::vector<double>* x0) {
        engine.reset();
        fea = std::make_unique<VoxelFEA>(in.dims, density, in.nu, in.bc, in.useGPU && gpuAvailable() ? GPU_COARSEST_DOF : 1100);
        if (fea->levels[0].elems.empty()) throw std::runtime_error("No voxels are connected to the fixtures.");
        engine = std::make_unique<Engine>(*fea, in.useGPU);
        out.engine = engine->name;
        out.gpuNote = engine->note;
        SolveOptions so;
        so.tol = 1e-5;
        so.maxIter = 3000;
        so.x0 = x0;
        so.onProgress = [&](int, double) { return progress && progress(-1, ""); };
        auto sol = engine->solve(in.f, so);
        if (sol.cancelled) throw Cancel();
        return TopologySolve{std::move(sol.u), in.f, fea.get()};
    };
    o.onIter = [&](const TopologyIter& s) {
        TopologyStep st;
        st.it = s.it;
        st.compliance = s.compliance;
        st.volume = s.volume;
        st.change = s.change;
        st.density.assign(in.density.size(), 0.f);
        for (size_t q = 0; q < s.solid.size(); q++) st.density[s.solid[q]] = float(s.xPhys[q]);
        onIter(std::move(st));
        report(progress, double(s.it + 1) / maxIter, "Topology on the " + out.engine + " · iteration " + std::to_string(s.it + 1));
        return false;
    };
    auto res = optimizeTopology(o);
    out.density = std::move(res.density);
    for (const auto& p : res.history) out.history.push_back({p.it, p.compliance, p.volume});
    out.keptFraction = res.keptFraction;
    return out;
}

// ---------- material & size ----------

// Stresses are linear in the loads, so each load group is solved once and rescaled: for geometric
// scale s and a material of density rho, force loads give stress / s^2, pressure loads keep their
// stress, and self-weight gives stress * s * rho / rho0 (displacements also scale with 1/E).
SizingRun runSizing(const StructuralModel&, const StructuralInput& in, const SizingOptions& o, double toMeters, const ProgressFn& progress) {
    SizingRun out;
    auto fea = buildModel(in, false, nullptr, &out.removed);
    Engine engine(*fea, in.useGPU);
    out.engine = engine.name;
    out.gpuNote = engine.note;
    const Level& L = fea->levels[0];
    struct Group { std::string kind; std::vector<float> S; std::vector<double> u; };
    std::vector<Group> groups;
    for (const auto& g : o.groups) {
        report(progress, double(groups.size()) / o.groups.size(), "Solving load group “" + g.name + "” on the " + engine.name);
        SolveOptions so;
        so.tol = 1e-7;
        so.maxIter = 3000;
        const auto sol = engine.solve(g.f, so);
        if (sol.cancelled) throw Cancel();
        Group gr;
        gr.kind = g.kind;
        gr.u = physical(sol.u, in);
        gr.S = fea->stressTensors(gr.u, in.E, in.h);
        groups.push_back(std::move(gr));
    }
    if (groups.empty()) throw std::runtime_error("No loads to size against.");
    // candidate nodes: the most stressed / displaced of each group (the combination's peak is there
    // unless the bound check below says otherwise)
    std::vector<int64_t> act;
    for (int64_t n = 0; n < L.nNodes; n++)
        if (L.activeNode[n]) act.push_back(n);
    auto vmOf = [](const float* S) {
        double s[6];
        for (int k = 0; k < 6; k++) s[k] = S[k];
        return vonMises(s);
    };
    std::unordered_set<int64_t> cand;
    std::vector<double> thr;
    for (const auto& g : groups) {
        std::vector<double> vals(act.size()), dv(act.size());
        for (size_t i = 0; i < act.size(); i++) {
            const int64_t n = act[i];
            vals[i] = vmOf(&g.S[6 * n]);
            dv[i] = std::sqrt(g.u[3 * n] * g.u[3 * n] + g.u[3 * n + 1] * g.u[3 * n + 1] + g.u[3 * n + 2] * g.u[3 * n + 2]);
        }
        std::vector<size_t> order(act.size());
        std::iota(order.begin(), order.end(), 0);
        const size_t K = std::min<size_t>(order.size(), 2500);
        std::partial_sort(order.begin(), order.begin() + std::min(order.size(), K + 1), order.end(), [&](size_t a, size_t b) { return vals[a] > vals[b]; });
        for (size_t i = 0; i < K; i++) cand.insert(act[order[i]]);
        thr.push_back(K < order.size() ? vals[order[K]] : 0);
        const size_t Kd = std::min<size_t>(order.size(), 500);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + Kd, order.end(), [&](size_t a, size_t b) { return dv[a] > dv[b]; });
        for (size_t i = 0; i < Kd; i++) cand.insert(act[order[i]]);
    }
    const std::vector<int64_t> nodes(cand.begin(), cand.end());
    const double E0 = in.E, rho0 = in.rho;
    struct Eval { double fos, peak, disp; bool exact; };
    auto evaluate = [&](double s, const Material& mat) {
        std::vector<double> c(groups.size()), dc(groups.size());
        for (size_t i = 0; i < groups.size(); i++) {
            c[i] = groups[i].kind == "force" ? 1 / (s * s) : groups[i].kind == "gravity" ? s * mat.density / rho0 : 1;
            dc[i] = c[i] * s * (E0 / (mat.E * 1e9));
        }
        double peak = 0, disp = 0;
        double s6[6], pr[3];
        for (int64_t n : nodes) {
            std::fill(s6, s6 + 6, 0.0);
            double x = 0, y = 0, z = 0;
            for (size_t gi = 0; gi < groups.size(); gi++) {
                const float* S = &groups[gi].S[6 * n];
                for (int k = 0; k < 6; k++) s6[k] += c[gi] * S[k];
                const auto& u = groups[gi].u;
                x += dc[gi] * u[3 * n]; y += dc[gi] * u[3 * n + 1]; z += dc[gi] * u[3 * n + 2];
            }
            double v;
            if (mat.brittle) {
                principalStresses(s6[0], s6[1], s6[2], s6[3], s6[4], s6[5], pr);
                v = pr[0];
            } else v = vonMises(s6);
            peak = std::max(peak, v);
            disp = std::max(disp, std::sqrt(x * x + y * y + z * z));
        }
        double bound = 0;
        for (size_t i = 0; i < groups.size(); i++) bound += std::abs(c[i]) * thr[i];
        const double strength = (mat.brittle ? mat.uts : mat.yield) * 1e6;
        return Eval{peak > 0 ? strength / peak : INFINITY, std::max(peak, 0.0), disp / toMeters, peak >= bound};
    };
    auto ok = [&](const Eval& r) { return r.fos >= o.fosTarget && (!(o.maxDisp > 0) || r.disp <= o.maxDisp); };
    for (size_t mi = 0; mi < o.materials.size(); mi++) {
        const Material& mat = o.materials[mi];
        report(progress, double(mi) / o.materials.size(), "Comparing materials · " + mat.name);
        // scan scales on a log grid, then refine the smallest passing scale by bisection
        std::vector<double> grid;
        for (int k = 0; k <= 48; k++) grid.push_back(o.scaleLo * std::pow(o.scaleHi / o.scaleLo, k / 48.0));
        int pass = -1;
        for (int k = 0; k < int(grid.size()); k++)
            if (ok(evaluate(grid[k], mat))) { pass = k; break; }
        SizingRow row;
        row.id = mat.id;
        row.name = mat.name;
        if (pass < 0) { out.rows.push_back(row); continue; }
        double s = grid[0];
        if (pass > 0) {
            double a = grid[pass - 1], b = grid[pass];
            for (int it = 0; it < 30; it++) {
                const double c = std::sqrt(a * b);
                if (ok(evaluate(c, mat))) b = c;
                else a = c;
            }
            s = b;
        }
        const auto r = evaluate(s, mat);
        row.feasible = true;
        row.scale = s;
        row.fos = r.fos;
        row.disp = r.disp;
        row.mass = mat.density * o.volume * s * s * s * toMeters * toMeters * toMeters;
        row.cost = row.mass * mat.cost;
        row.exact = r.exact;
        out.rows.push_back(row);
    }
    const auto b = evaluate(1, o.base);
    out.base.id = o.base.id;
    out.base.name = o.base.name;
    out.base.feasible = true;
    out.base.scale = 1;
    out.base.fos = b.fos;
    out.base.disp = b.disp;
    return out;
}

}  // namespace ps
