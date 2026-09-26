#include "gpu_fea.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../util/parallel.hpp"
#include "gpu.hpp"

namespace ps {

double GpuFeaSolver::innerTol = 0.1;

#ifdef PARTS_SIM_HAS_GPU

namespace {
constexpr int SMOOTH_SWEEPS = 2;
constexpr int RED_GROUPS = 1024;
// scalar slots on the GPU
constexpr int RZ = 0, PQ = 1, RR = 4, RZN = 5, BB = 6;
}  // namespace

struct GpuFeaSolver::Impl {
    GpuContext& ctx;
    std::unique_ptr<std::lock_guard<std::mutex>> guard;
    std::unique_ptr<GpuProgram> prog;
    GpuBuffer K0;
    struct Lvl {
        const Level* lv;
        bool last;
        GpuBuffer params, invD, emap, edata, dadd, r, z, t, freeIdx, ainv;
        GpuBindGroup mvZT, smoothFirst, smooth, resid, restrictBG, prolongBG, coarsest;
    };
    std::vector<Lvl> L;
    GpuBuffer x, b, pv, qv, S, partials, red, readBuf, xRead;
    GpuBindGroup mvP, mvX, initR, copyZP, updXR, updP, alpha, beta;
    std::vector<GpuBindGroup> reduce;
    std::map<std::pair<WGPUBuffer, WGPUBuffer>, GpuBindGroup> dotBG;
    uint32_t dotGroups = 1;
    int64_t n = 0;
    int64_t coarseM = 0;

    explicit Impl(GpuContext& c) : ctx(c), guard(std::make_unique<std::lock_guard<std::mutex>>(c.lock)) {}

    void run(WGPUComputePassEncoder pass, const char* entry, WGPUBindGroup group, uint64_t count) {
        const auto d = gpuDispatchSize(ctx, count);
        wgpuComputePassEncoderSetPipeline(pass, prog->pipeline(entry));
        wgpuComputePassEncoderSetBindGroup(pass, 0, group, 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, d[0], d[1], 1);
    }

    WGPUBindGroup dotGroup(WGPUBuffer a, WGPUBuffer c) {
        auto key = std::make_pair(a, c);
        auto it = dotBG.find(key);
        if (it == dotBG.end())
            it = dotBG.emplace(key, prog->bindGroup("dot_partial", {{0, L[0].params.get()}, {1, a}, {9, c}, {11, partials.get()}})).first;
        return it->second.get();
    }

    void dot(WGPUComputePassEncoder pass, WGPUBuffer a, WGPUBuffer c, int slot) {
        wgpuComputePassEncoderSetPipeline(pass, prog->pipeline("dot_partial"));
        wgpuComputePassEncoderSetBindGroup(pass, 0, dotGroup(a, c), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, dotGroups, 1, 1);
        wgpuComputePassEncoderSetPipeline(pass, prog->pipeline("reduce"));
        wgpuComputePassEncoderSetBindGroup(pass, 0, reduce[slot].get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    }

    void vcycle(WGPUComputePassEncoder pass, size_t l) {
        Lvl& V = L[l];
        if (V.last) { run(pass, "coarsest", V.coarsest.get(), uint64_t(coarseM)); return; }
        const uint64_t nD = V.lv->nDof, nN = V.lv->nNodes;
        run(pass, "jacobi_first", V.smoothFirst.get(), nD);
        for (int s = 1; s < SMOOTH_SWEEPS; s++) {
            run(pass, "matvec", V.mvZT.get(), nN);
            run(pass, "jacobi", V.smooth.get(), nD);
        }
        run(pass, "matvec", V.mvZT.get(), nN);
        run(pass, "resid", V.resid.get(), nD);
        run(pass, "restrict_", V.restrictBG.get(), L[l + 1].lv->nNodes);
        vcycle(pass, l + 1);
        run(pass, "prolong", V.prolongBG.get(), nN);
        for (int s = 0; s < SMOOTH_SWEEPS; s++) {
            run(pass, "matvec", V.mvZT.get(), nN);
            run(pass, "jacobi", V.smooth.get(), nD);
        }
    }

    std::vector<float> scalars() {
        auto bytes = gpuRead(ctx, S.get(), readBuf.get(), 64);
        std::vector<float> s(16);
        std::memcpy(s.data(), bytes.data(), 64);
        return s;
    }

    WGPUCommandEncoder encoder() { return wgpuDeviceCreateCommandEncoder(ctx.device, nullptr); }
    WGPUComputePassEncoder begin(WGPUCommandEncoder enc) { return wgpuCommandEncoderBeginComputePass(enc, nullptr); }
    void submit(WGPUCommandEncoder enc, WGPUComputePassEncoder pass) {
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(ctx.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }
};

// dense inverse of the coarsest free-DOF matrix from its Cholesky factor (row-major lower)
static std::vector<float> coarseInverse(const VoxelFEA::Coarse& c) {
    const int64_t m = c.m;
    std::vector<float> inv(size_t(m * m));
    parallelFor(m, [&](int64_t lo, int64_t hi) {
        std::vector<double> y(static_cast<size_t>(m));
        for (int64_t col = lo; col < hi; col++) {
            std::fill(y.begin(), y.end(), 0.0);
            y[col] = 1;
            for (int64_t i = 0; i < m; i++) {
                double s = y[i];
                for (int64_t k = 0; k < i; k++) s -= c.A[i * m + k] * y[k];
                y[i] = s / c.A[i * m + i];
            }
            for (int64_t i = m - 1; i >= 0; i--) {
                double s = y[i];
                for (int64_t k = i + 1; k < m; k++) s -= c.A[k * m + i] * y[k];
                y[i] = s / c.A[i * m + i];
            }
            for (int64_t i = 0; i < m; i++) inv[size_t(i * m + col)] = float(y[i]);
        }
    }, 4);
    return inv;
}

GpuFeaSolver::GpuFeaSolver(VoxelFEA& fea) : fea_(fea) {
    GpuContext* ctx = GpuContext::get();
    if (!ctx) throw std::runtime_error("no usable GPU adapter was found");
    if (fea.coarse.A.empty() || fea.coarse.m == 0) throw std::runtime_error("the coarsest level has no direct solver");
    impl_ = std::make_unique<Impl>(*ctx);
    Impl& I = *impl_;
    gpuPushErrors(*ctx);
    I.prog = std::make_unique<GpuProgram>(*ctx, wgslSource("fea"),
                                          std::vector<std::string>{"matvec", "jacobi_first", "jacobi", "resid", "restrict_", "prolong", "coarsest", "copy", "init_r",
                                                                   "update_xr", "update_p", "dot_partial", "reduce", "cg_alpha", "cg_beta"});
    std::vector<float> k0(fea.K0.begin(), fea.K0.end());
    I.K0 = gpuBuffer(*ctx, k0.size() * 4, k0.data());
    I.coarseM = fea.coarse.m;
    const auto& levels = fea.levels;
    I.L.resize(levels.size());
    for (size_t l = 0; l < levels.size(); l++) {
        const Level& lv = levels[l];
        auto& V = I.L[l];
        V.lv = &lv;
        V.last = l + 1 == levels.size();
        const Level* next = V.last ? nullptr : &levels[l + 1];
        std::vector<float> invD(lv.nDof);
        for (int64_t i = 0; i < lv.nDof; i++) invD[i] = lv.fixed[i] ? 0.f : float(lv.invDiag[i]);
        const auto sN = gpuDispatchSize(*ctx, lv.nNodes), sD = gpuDispatchSize(*ctx, lv.nDof);
        uint32_t params[24] = {};
        params[0] = lv.nx; params[1] = lv.ny; params[2] = lv.nz; params[3] = lv.NX;
        params[4] = lv.NY; params[5] = lv.NZ; params[6] = uint32_t(lv.nNodes); params[7] = uint32_t(lv.nDof);
        params[8] = lv.shared() ? 1 : 0; params[9] = sN[2]; params[10] = sD[2]; params[11] = V.last ? uint32_t(fea.coarse.m) : 0;
        const float omega = float(lv.omega);
        std::memcpy(&params[12], &omega, 4);
        if (next) {
            const auto sC = gpuDispatchSize(*ctx, next->nNodes);
            params[16] = next->NX; params[17] = next->NY; params[18] = next->NZ; params[19] = uint32_t(next->nNodes); params[20] = sC[2];
        }
        params[21] = lv.diagAdd.empty() ? 0 : 1;
        V.params = gpuUniform(*ctx, sizeof(params), params);
        V.invD = gpuBuffer(*ctx, invD.size() * 4, invD.data());
        V.emap = gpuBuffer(*ctx, lv.emap.size() * 4, lv.emap.data());
        std::vector<float> edata(lv.shared() ? lv.rho.begin() : lv.K.begin(), lv.shared() ? lv.rho.end() : lv.K.end());
        V.edata = gpuBuffer(*ctx, edata.size() * 4, edata.data());
        std::vector<float> dadd(lv.diagAdd.begin(), lv.diagAdd.end());
        if (dadd.empty()) dadd.assign(4, 0.f);
        V.dadd = gpuBuffer(*ctx, dadd.size() * 4, dadd.data());
        V.r = gpuBuffer(*ctx, lv.nDof * 4);
        V.z = gpuBuffer(*ctx, lv.nDof * 4);
        V.t = gpuBuffer(*ctx, lv.nDof * 4);
        if (V.last) {
            std::vector<int32_t> freeIdx(fea.coarse.m);
            for (int64_t i = 0; i < lv.nDof; i++)
                if (fea.coarse.map[i] >= 0) freeIdx[fea.coarse.map[i]] = int32_t(i);
            V.freeIdx = gpuBuffer(*ctx, freeIdx.size() * 4, freeIdx.data());
            const auto inv = coarseInverse(fea.coarse);
            V.ainv = gpuBuffer(*ctx, inv.size() * 4, inv.data());
        }
    }
    const int64_t n = levels[0].nDof;
    I.n = n;
    I.x = gpuBuffer(*ctx, n * 4);
    I.b = gpuBuffer(*ctx, n * 4);
    I.pv = gpuBuffer(*ctx, n * 4);
    I.qv = gpuBuffer(*ctx, n * 4);
    I.S = gpuBuffer(*ctx, 64);
    I.partials = gpuBuffer(*ctx, 4 * RED_GROUPS);
    I.dotGroups = uint32_t(std::min<int64_t>(RED_GROUPS, (n + 255) / 256));
    // reduction slots: one 256-byte-aligned uniform entry per scalar
    std::vector<uint32_t> red(64 * 8, 0);
    for (int s = 0; s < 8; s++) { red[s * 64] = s; red[s * 64 + 1] = I.dotGroups; }
    I.red = gpuBuffer(*ctx, red.size() * 4, red.data(), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    I.readBuf = gpuReadback(*ctx, 64);
    I.xRead = gpuReadback(*ctx, n * 4);
    auto& P = *I.prog;
    for (size_t l = 0; l < I.L.size(); l++) {
        auto& V = I.L[l];
        V.mvZT = P.bindGroup("matvec", {{0, V.params.get()}, {1, V.z.get()}, {2, V.t.get()}, {3, V.invD.get()}, {4, V.emap.get()}, {5, V.edata.get()},
                                         {6, I.K0.get()}, {13, V.dadd.get()}});
        V.smoothFirst = P.bindGroup("jacobi_first", {{0, V.params.get()}, {2, V.z.get()}, {3, V.invD.get()}, {7, V.r.get()}});
        V.smooth = P.bindGroup("jacobi", {{0, V.params.get()}, {1, V.t.get()}, {2, V.z.get()}, {3, V.invD.get()}, {7, V.r.get()}});
        V.resid = P.bindGroup("resid", {{0, V.params.get()}, {2, V.t.get()}, {3, V.invD.get()}, {7, V.r.get()}});
        if (!V.last) {
            auto& N = I.L[l + 1];
            V.restrictBG = P.bindGroup("restrict_", {{0, V.params.get()}, {1, V.t.get()}, {2, N.r.get()}, {3, N.invD.get()}});
            V.prolongBG = P.bindGroup("prolong", {{0, V.params.get()}, {1, N.z.get()}, {2, V.z.get()}, {3, V.invD.get()}});
        } else {
            V.coarsest = P.bindGroup("coarsest", {{0, V.params.get()}, {1, V.r.get()}, {2, V.z.get()}, {4, V.freeIdx.get()}, {5, V.ainv.get()}});
        }
    }
    auto& L0 = I.L[0];
    I.mvP = P.bindGroup("matvec", {{0, L0.params.get()}, {1, I.pv.get()}, {2, I.qv.get()}, {3, L0.invD.get()}, {4, L0.emap.get()}, {5, L0.edata.get()},
                                    {6, I.K0.get()}, {13, L0.dadd.get()}});
    I.mvX = P.bindGroup("matvec", {{0, L0.params.get()}, {1, I.x.get()}, {2, I.qv.get()}, {3, L0.invD.get()}, {4, L0.emap.get()}, {5, L0.edata.get()},
                                    {6, I.K0.get()}, {13, L0.dadd.get()}});
    I.initR = P.bindGroup("init_r", {{0, L0.params.get()}, {1, I.qv.get()}, {2, L0.r.get()}, {3, L0.invD.get()}, {7, I.b.get()}});
    I.copyZP = P.bindGroup("copy", {{0, L0.params.get()}, {1, L0.z.get()}, {2, I.pv.get()}});
    I.updXR = P.bindGroup("update_xr", {{0, L0.params.get()}, {1, I.pv.get()}, {2, I.x.get()}, {8, L0.r.get()}, {9, I.qv.get()}, {10, I.S.get()}});
    I.updP = P.bindGroup("update_p", {{0, L0.params.get()}, {1, L0.z.get()}, {2, I.pv.get()}, {10, I.S.get()}});
    I.alpha = P.bindGroup("cg_alpha", {{10, I.S.get()}});
    I.beta = P.bindGroup("cg_beta", {{10, I.S.get()}});
    for (int s = 0; s < 8; s++) I.reduce.push_back(P.bindGroup("reduce", {{10, I.S.get()}, {11, I.partials.get()}, {12, I.red.get(), uint64_t(s) * 256, 16}}));
    gpuPopErrors(*ctx, "GPU solver setup failed");
}

GpuFeaSolver::~GpuFeaSolver() = default;

std::vector<double> GpuFeaSolver::pcg(const std::vector<double>& rhs, double tol, int maxIter, int* iterations) {
    Impl& I = *impl_;
    auto& ctx = I.ctx;
    const Level& lv = fea_.levels[0];
    const int64_t n = I.n;
    std::vector<float> b32(n);
    for (int64_t i = 0; i < n; i++) b32[i] = lv.fixed[i] ? 0.f : float(rhs[i]);
    wgpuQueueWriteBuffer(ctx.queue, I.b.get(), 0, b32.data(), n * 4);
    std::vector<float> zero(n, 0.f);
    wgpuQueueWriteBuffer(ctx.queue, I.x.get(), 0, zero.data(), n * 4);
    auto enc = I.encoder();
    auto pass = I.begin(enc);
    I.run(pass, "matvec", I.mvX.get(), lv.nNodes);
    I.run(pass, "init_r", I.initR.get(), n);
    I.vcycle(pass, 0);
    I.run(pass, "copy", I.copyZP.get(), n);
    I.dot(pass, I.L[0].r.get(), I.L[0].z.get(), RZ);
    I.dot(pass, I.b.get(), I.b.get(), BB);
    I.dot(pass, I.L[0].r.get(), I.L[0].r.get(), RR);
    I.submit(enc, pass);
    auto s = I.scalars();
    const double bnorm = std::sqrt(double(s[BB]));
    if (iterations) *iterations = 0;
    if (!(bnorm > 0)) return std::vector<double>(n, 0.0);
    double res = std::sqrt(double(s[RR])) / bnorm;
    int it = 0;
    const int CHECK = 6;
    while (it < maxIter && res > tol) {
        enc = I.encoder();
        pass = I.begin(enc);
        const int batch = std::min(CHECK, maxIter - it);
        for (int k = 0; k < batch; k++) {
            I.run(pass, "matvec", I.mvP.get(), lv.nNodes);
            I.dot(pass, I.pv.get(), I.qv.get(), PQ);
            I.run(pass, "cg_alpha", I.alpha.get(), 1);
            I.run(pass, "update_xr", I.updXR.get(), n);
            I.vcycle(pass, 0);
            I.dot(pass, I.L[0].r.get(), I.L[0].z.get(), RZN);
            I.run(pass, "cg_beta", I.beta.get(), 1);
            I.run(pass, "update_p", I.updP.get(), n);
        }
        I.dot(pass, I.L[0].r.get(), I.L[0].r.get(), RR);
        I.submit(enc, pass);
        it += batch;
        s = I.scalars();
        if (!(s[PQ] > 0) || !std::isfinite(s[RR])) break;
        res = std::sqrt(double(s[RR])) / bnorm;
    }
    if (iterations) *iterations = it;
    const auto bytes = gpuRead(ctx, I.x.get(), I.xRead.get(), uint64_t(n) * 4);
    const float* x32 = reinterpret_cast<const float*>(bytes.data());
    return std::vector<double>(x32, x32 + n);
}

SolveResult GpuFeaSolver::solve(const std::vector<double>& f, const SolveOptions& opts) {
    const Level& L = fea_.levels[0];
    const int64_t n = L.nDof;
    SolveResult res;
    res.u = opts.x0 ? *opts.x0 : std::vector<double>(n, 0.0);
    auto& u = res.u;
    for (int64_t i = 0; i < n; i++)
        if (L.fixed[i]) u[i] = 0;
    double bnorm = 0;
    for (int64_t i = 0; i < n; i++)
        if (!L.fixed[i]) bnorm += f[i] * f[i];
    bnorm = std::sqrt(bnorm);
    if (bnorm == 0) { res.converged = true; return res; }
    std::vector<double> q(n), r(n), rOld(n), p(n);
    fea_.apply(0, u.data(), q.data());
    for (int64_t i = 0; i < n; i++) r[i] = L.fixed[i] ? 0 : f[i] - q[i];
    double rel = std::sqrt(dot(r.data(), r.data(), n)) / bnorm;
    int its = 0;
    auto precondition = [&](const std::vector<double>& rr) {
        int inner = 0;
        auto z = pcg(rr, innerTol, std::max(1, std::min(60, opts.maxIter - its)), &inner);
        its += std::max(1, inner);
        return z;
    };
    if (rel <= opts.tol) { res.residual = rel; res.converged = true; return res; }
    auto z = precondition(r);
    p = z;
    double rz = dot(r.data(), z.data(), n);
    for (int outer = 0; outer < 40 && its < opts.maxIter; outer++) {
        fea_.apply(0, p.data(), q.data());
        const double pq = dot(p.data(), q.data(), n);
        if (!(pq > 0)) break;
        const double alpha = rz / pq;
        rOld = r;
        parallelFor(n, [&](int64_t lo, int64_t hi) {
            for (int64_t i = lo; i < hi; i++) { u[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        });
        rel = std::sqrt(dot(r.data(), r.data(), n)) / bnorm;
        if (opts.onProgress && opts.onProgress(its, rel)) { res.cancelled = true; break; }
        if (rel <= opts.tol) break;
        auto zn = precondition(r);
        // flexible (Polak-Ribiere) beta: the preconditioner changes from step to step
        double num = 0;
        for (int64_t i = 0; i < n; i++) num += zn[i] * (r[i] - rOld[i]);
        const double beta = std::max(0.0, num / rz);
        rz = dot(r.data(), zn.data(), n);
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) p[i] = zn[i] + beta * p[i]; });
    }
    res.iterations = its;
    res.residual = rel;
    res.converged = !res.cancelled && rel <= opts.tol * 10;
    return res;
}

#else

struct GpuFeaSolver::Impl {};
GpuFeaSolver::GpuFeaSolver(VoxelFEA& fea) : fea_(fea) { throw std::runtime_error("this build has no GPU support"); }
GpuFeaSolver::~GpuFeaSolver() = default;
std::vector<double> GpuFeaSolver::pcg(const std::vector<double>&, double, int, int*) { return {}; }
SolveResult GpuFeaSolver::solve(const std::vector<double>&, const SolveOptions&) { return {}; }

#endif

}  // namespace ps
