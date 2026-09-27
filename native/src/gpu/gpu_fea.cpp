#include "gpu_fea.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <tuple>
#include <stdexcept>

#include "../util/parallel.hpp"
#include "gpu.hpp"
#include "../util/trace.hpp"

namespace ps {


#ifdef PARTS_SIM_HAS_GPU

namespace {
constexpr int RED_GROUPS = 1024;
// scalar slots on the GPU (2 = alpha, 3 = beta)
constexpr int PQ = 1, ALPHA = 2, RR = 4, RZN = 5, ZQ = 7;  // 0 = rz
}  // namespace

struct GpuFeaSolver::Impl {
    GpuContext& ctx;
    std::unique_ptr<std::lock_guard<std::mutex>> guard;
    std::unique_ptr<GpuProgram> prog;
    struct Lvl {
        const Level* lv;
        bool last;
        int out = 0;  // buffer holding the V-cycle result: 0 = z, 1 = t
        GpuBuffer params, invD, words, Ke, dadd, r, z, t, d, freeIdx, ainv, K0, stencil;  // K0 / stencil: the level's uniform element
        // [p]: from buffer p (0 = z, 1 = t) to the other
        GpuBindGroup first, chebA[2], chebB[2], resid[2], restrictBG[2], prolongBG[2], coarsest, mv;
        GpuBindGroup lzX, lzMv, lzW, lzAxpy, lzNext;  // Lanczos: v = z, vPrev = d, w = t, x = r
        WGPUBuffer buf(int p) const { return p ? t.get() : z.get(); }
    };
    std::vector<Lvl> L;
    GpuBuffer x, pv, qv, S, partials, red, readBuf, xRead;
    GpuBindGroup mvP, updXR, updP, alpha, beta, lzStore;
    std::vector<float> host;  // conversion buffer for uploads
    std::vector<GpuBindGroup> reduce;
    std::map<std::tuple<WGPUBuffer, WGPUBuffer, WGPUBuffer>, GpuBindGroup> dotBG;
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

    WGPUBindGroup dotGroup(WGPUBuffer a, WGPUBuffer c, WGPUBuffer params) {
        auto key = std::make_tuple(a, c, params);
        auto it = dotBG.find(key);
        if (it == dotBG.end())
            it = dotBG.emplace(key, prog->bindGroup("dot_partial", {{0, params}, {1, a}, {9, c}, {11, partials.get()}})).first;
        return it->second.get();
    }

    // S[slot] = a.c over the DOFs of level l (the same number of partial sums for every level)
    void dot(WGPUComputePassEncoder pass, WGPUBuffer a, WGPUBuffer c, int slot, size_t l = 0) {
        wgpuComputePassEncoderSetPipeline(pass, prog->pipeline("dot_partial"));
        wgpuComputePassEncoderSetBindGroup(pass, 0, dotGroup(a, c, L[l].params.get()), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, dotGroups, 1, 1);
        wgpuComputePassEncoderSetPipeline(pass, prog->pipeline("reduce"));
        wgpuComputePassEncoderSetBindGroup(pass, 0, reduce[slot].get(), 0, nullptr);
        wgpuComputePassEncoderDispatchWorkgroups(pass, 1, 1, 1);
    }

    // pre-smoothing from zero, residual to the coarser level, its correction, post-smoothing (both
    // degree-2 Chebyshev, like VoxelFEA::vcycle); steps alternate between the level's z and t
    void vcycle(WGPUComputePassEncoder pass, size_t l) {
        Lvl& V = L[l];
        if (V.last) {
            wgpuComputePassEncoderSetPipeline(pass, prog->pipeline("coarsest"));
            wgpuComputePassEncoderSetBindGroup(pass, 0, V.coarsest.get(), 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, uint32_t((coarseM + 31) / 32), 1, 1);  // 32 rows each
            return;
        }
        const uint64_t nD = V.lv->nDof, nN = V.lv->nNodes;
        run(pass, "cheb_first", V.first.get(), nD);                   // z
        run(pass, "cheb_b", V.chebB[0].get(), nN);                    // z -> t
        run(pass, "resid", V.resid[1].get(), nN);                     // t -> residual in z
        run(pass, "restrict_", V.restrictBG[0].get(), L[l + 1].lv->nNodes);
        vcycle(pass, l + 1);
        run(pass, "prolong", V.prolongBG[1].get(), nN);               // t += P z_coarse
        run(pass, "cheb_a", V.chebA[1].get(), nN);                    // t -> z
        run(pass, "cheb_b", V.chebB[0].get(), nN);                    // z -> t
    }
    WGPUBuffer result(size_t l) const { return L[l].buf(L[l].out); }

    std::array<float, 64> scalars() {
        std::array<float, 64> s{};
        gpuReadInto(ctx, S.get(), readBuf.get(), 256, [&](const void* p) { std::memcpy(s.data(), p, 256); });
        return s;
    }

    // largest eigenvalue of D^-1 K on level l: 10 Lanczos steps on the GPU (VoxelFEA::estimateOmega)
    double lanczosMax(size_t l) {
        Lvl& V = L[l];
        const Level& lv = *V.lv;
        const int64_t m = lv.nDof;
        // pseudo-random start (a hash of the index, so it fills in parallel), unit length
        std::vector<float> v(m);
        const double nv = parallelSum(m, [&](int64_t lo, int64_t hi) {
            double acc = 0;
            for (int64_t i = lo; i < hi; i++) {
                uint32_t h = uint32_t(i) * 2654435761u;
                h ^= h >> 15;
                h *= 2246822519u;
                h ^= h >> 13;
                v[i] = lv.fixed[i] ? 0.f : float(h) / 4294967296.f - 0.5f;
                acc += double(v[i]) * v[i];
            }
            return acc;
        });
        const float inv = float(1 / std::sqrt(nv > 0 ? nv : 1));
        parallelFor(m, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) v[i] *= inv; });
        wgpuQueueWriteBuffer(ctx.queue, V.z.get(), 0, v.data(), uint64_t(m) * 4);
        clear({V.d.get(), S.get()});
        constexpr int STEPS = 10;
        auto enc = encoder();
        auto pass = begin(enc);
        for (int j = 0; j < STEPS; j++) {
            run(pass, "lz_x", V.lzX.get(), uint64_t(m));
            run(pass, "matvec", V.lzMv.get(), uint64_t(lv.nNodes));
            run(pass, "lz_w", V.lzW.get(), uint64_t(m));
            dot(pass, V.t.get(), V.z.get(), 8, l);
            run(pass, "lz_axpy", V.lzAxpy.get(), uint64_t(m));
            dot(pass, V.t.get(), V.t.get(), 9, l);
            run(pass, "lz_next", V.lzNext.get(), uint64_t(m));
            run(pass, "lz_store", lzStore.get(), 1);
        }
        submit(enc, pass);
        const auto sc = scalars();
        std::vector<double> alpha, beta;
        for (int j = 0; j < STEPS; j++) {
            alpha.push_back(sc[16 + j]);
            if (!(sc[32 + j] > 1e-24 * sc[16 + j] * sc[16 + j]) || j + 1 == STEPS) break;
            beta.push_back(std::sqrt(double(sc[32 + j])));
        }
        return tridiagonalMax(alpha, beta);
    }

    // dst = float32(scale * v), held DOFs zero
    void upload(WGPUBuffer dst, const double* v, double scale, const Level& lv) {
        host.resize(size_t(n));
        parallelFor(n, [&](int64_t lo, int64_t hi) {
            for (int64_t i = lo; i < hi; i++) host[i] = lv.fixed[i] ? 0.f : float(v[i] * scale);
        });
        wgpuQueueWriteBuffer(ctx.queue, dst, 0, host.data(), uint64_t(n) * 4);
    }

    // out(i, value) for every DOF of a float32 GPU vector
    template <class F>
    void download(WGPUBuffer src, F out) {
        gpuReadInto(ctx, src, xRead.get(), uint64_t(n) * 4, [&](const void* p) {
            const float* v = static_cast<const float*>(p);
            parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) out(i, v[i]); });
        });
    }

    void clear(std::initializer_list<WGPUBuffer> bufs) {
        WGPUCommandEncoder enc = encoder();
        for (WGPUBuffer b : bufs) wgpuCommandEncoderClearBuffer(enc, b, 0, WGPU_WHOLE_SIZE);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(ctx.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
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
    TraceTimer trace("  GPU shaders");
    gpuPushErrors(*ctx);
    I.prog = std::make_unique<GpuProgram>(*ctx, wgslSource("fea"),
                                          std::vector<std::string>{"matvec", "cheb_first", "cheb_a", "cheb_b", "resid", "restrict_", "prolong", "coarsest", "update_xr",
                                                                   "update_p", "dot_partial", "reduce", "cg_alpha", "cg_beta", "lz_x", "lz_w", "lz_axpy",
                                                                   "lz_next", "lz_store"});
    trace.lap("  GPU upload");
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
        parallelFor(lv.nDof, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) invD[i] = lv.fixed[i] ? 0.f : float(lv.invDiag[i]); });
        const auto sN = gpuDispatchSize(*ctx, lv.nNodes), sD = gpuDispatchSize(*ctx, lv.nDof);
        uint32_t params[24] = {};
        params[0] = lv.nx; params[1] = lv.ny; params[2] = lv.nz; params[3] = lv.NX;
        params[4] = lv.NY; params[5] = lv.NZ; params[6] = uint32_t(lv.nNodes); params[7] = uint32_t(lv.nDof);
        params[9] = sN[2]; params[10] = sD[2]; params[11] = V.last ? uint32_t(fea.coarse.m) : 0;
        if (next) {
            const auto sC = gpuDispatchSize(*ctx, next->nNodes);
            params[16] = next->NX; params[17] = next->NY; params[18] = next->NZ; params[19] = uint32_t(next->nNodes); params[20] = sC[2];
        }
        params[21] = lv.diagAdd.empty() ? 0 : 1;
        V.params = gpuUniform(*ctx, sizeof(params), params);
        V.invD = gpuBuffer(*ctx, invD.size() * 4, invD.data());
        // one word per voxel: 0 empty, f32 bits of s for s * K0, sign bit | matrix index otherwise
        std::vector<int32_t> words(lv.emap.size(), 0);
        parallelFor(int64_t(words.size()), [&](int64_t lo, int64_t hi) {
            for (int64_t v = lo; v < hi; v++) {
                const int e = lv.emap[v];
                if (e < 0) continue;
                if (lv.scaled(e)) {
                    const float s = float(lv.rho[e]);
                    std::memcpy(&words[v], &s, 4);
                } else words[v] = int32_t(0x80000000u | uint32_t(lv.kIdx[e]));
            }
        });
        V.words = gpuBuffer(*ctx, words.size() * 4, words.data());
        V.Ke = gpuBuffer(*ctx, lv.K.size() * 4, lv.K.data());
        // the level's uniform element (rows as 6 vec4) and its 27-point stencil: neighbour
        // m = ox + 3 oy + 9 oz, row d -> [3m + d].xyz
        V.K0 = gpuUniform(*ctx, 576 * 4, lv.Kbf.data());
        {
            std::vector<float> st(81 * 4, 0.f);
            for (int row = 0; row < 9; row++)
                for (int col = 0; col < 9; col++)
                    for (int d = 0; d < 3; d++) {
                        const int ox = col / 3, c = col % 3, m = ox + 3 * row;
                        st[(3 * m + d) * 4 + c] = lv.Sbf[(row * 9 + col) * 4 + d];
                    }
            V.stencil = gpuUniform(*ctx, st.size() * 4, st.data());
        }
        std::vector<float> dadd(lv.diagAdd.begin(), lv.diagAdd.end());
        if (dadd.empty()) dadd.assign(4, 0.f);
        V.dadd = gpuBuffer(*ctx, dadd.size() * 4, dadd.data());
        V.r = gpuBuffer(*ctx, lv.nDof * 4);
        V.z = gpuBuffer(*ctx, lv.nDof * 4);
        V.t = gpuBuffer(*ctx, lv.nDof * 4);
        V.d = gpuBuffer(*ctx, lv.nDof * 4);
        if (V.last) {
            V.out = 0;
            std::vector<int32_t> freeIdx(fea.coarse.m);
            for (int64_t i = 0; i < lv.nDof; i++)
                if (fea.coarse.map[i] >= 0) freeIdx[fea.coarse.map[i]] = int32_t(i);
            V.freeIdx = gpuBuffer(*ctx, freeIdx.size() * 4, freeIdx.data());
            const auto inv = coarseInverse(fea.coarse);
            V.ainv = gpuBuffer(*ctx, inv.size() * 4, inv.data());
        } else V.out = 1;  // t
    }
    const int64_t n = levels[0].nDof;
    I.n = n;
    I.x = gpuBuffer(*ctx, n * 4);
    I.pv = gpuBuffer(*ctx, n * 4);
    I.qv = gpuBuffer(*ctx, n * 4);
    I.S = gpuBuffer(*ctx, 256);
    I.partials = gpuBuffer(*ctx, 4 * RED_GROUPS);
    I.dotGroups = uint32_t(std::min<int64_t>(RED_GROUPS, (n + 255) / 256));
    // reduction slots: one 256-byte-aligned uniform entry per scalar
    std::vector<uint32_t> red(64 * 10, 0);
    for (int s = 0; s < 10; s++) { red[s * 64] = s; red[s * 64 + 1] = I.dotGroups; }
    I.red = gpuBuffer(*ctx, red.size() * 4, red.data(), WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
    I.readBuf = gpuReadback(*ctx, 256);
    I.xRead = gpuReadback(*ctx, n * 4);
    auto& P = *I.prog;
    for (size_t l = 0; l < I.L.size(); l++) {
        auto& V = I.L[l];
        auto op = [&](const char* entry, WGPUBuffer in, WGPUBuffer out, bool withD) {
            std::vector<GpuProgram::Bind> b = {{0, V.params.get()}, {1, in}, {2, out}, {3, V.invD.get()}, {4, V.words.get()}, {5, V.Ke.get()},
                                               {6, V.K0.get()}, {7, V.r.get()}, {13, V.dadd.get()}, {14, V.stencil.get()}};
            if (withD) b.push_back({8, V.d.get()});
            return P.bindGroup(entry, b);
        };
        V.first = P.bindGroup("cheb_first", {{0, V.params.get()}, {2, V.z.get()}, {3, V.invD.get()}, {7, V.r.get()}, {8, V.d.get()}});
        for (int p = 0; p < 2; p++) {
            V.chebA[p] = op("cheb_a", V.buf(p), V.buf(p ^ 1), true);
            V.chebB[p] = op("cheb_b", V.buf(p), V.buf(p ^ 1), true);
            V.resid[p] = op("resid", V.buf(p), V.buf(p ^ 1), false);
        }
        if (!V.last) {
            V.lzX = P.bindGroup("lz_x", {{0, V.params.get()}, {1, V.z.get()}, {2, V.r.get()}, {3, V.invD.get()}});
            V.lzMv = P.bindGroup("matvec", {{0, V.params.get()}, {1, V.r.get()}, {2, V.t.get()}, {3, V.invD.get()}, {4, V.words.get()}, {5, V.Ke.get()},
                                            {6, V.K0.get()}, {13, V.dadd.get()}, {14, V.stencil.get()}});
            V.lzW = P.bindGroup("lz_w", {{0, V.params.get()}, {1, V.d.get()}, {2, V.t.get()}, {3, V.invD.get()}, {10, I.S.get()}});
            V.lzAxpy = P.bindGroup("lz_axpy", {{0, V.params.get()}, {1, V.z.get()}, {2, V.t.get()}, {10, I.S.get()}});
            V.lzNext = P.bindGroup("lz_next", {{0, V.params.get()}, {1, V.t.get()}, {2, V.z.get()}, {8, V.d.get()}, {10, I.S.get()}});
            auto& N = I.L[l + 1];
            for (int p = 0; p < 2; p++) {
                V.restrictBG[p] = P.bindGroup("restrict_", {{0, V.params.get()}, {1, V.buf(p)}, {2, N.r.get()}, {3, N.invD.get()}});
                V.prolongBG[p] = P.bindGroup("prolong", {{0, V.params.get()}, {1, N.buf(N.out)}, {2, V.buf(p)}, {3, V.invD.get()}});
            }
            V.mv = P.bindGroup("matvec", {{0, V.params.get()}, {1, V.z.get()}, {2, V.t.get()}, {3, V.invD.get()}, {4, V.words.get()}, {5, V.Ke.get()},
                                          {6, V.K0.get()}, {13, V.dadd.get()}, {14, V.stencil.get()}});
        } else {
            V.coarsest = P.bindGroup("coarsest", {{0, V.params.get()}, {1, V.r.get()}, {2, V.z.get()}, {4, V.freeIdx.get()}, {15, V.ainv.get()}});
        }
    }
    auto& L0 = I.L[0];
    const WGPUBuffer z0 = I.result(0);
    auto mv = [&](WGPUBuffer in, WGPUBuffer out) {
        return P.bindGroup("matvec", {{0, L0.params.get()}, {1, in}, {2, out}, {3, L0.invD.get()}, {4, L0.words.get()}, {5, L0.Ke.get()},
                                      {6, L0.K0.get()}, {13, L0.dadd.get()}, {14, L0.stencil.get()}});
    };
    I.mvP = mv(I.pv.get(), I.qv.get());
    I.updXR = P.bindGroup("update_xr", {{0, L0.params.get()}, {1, I.pv.get()}, {2, I.x.get()}, {8, L0.r.get()}, {9, I.qv.get()}, {10, I.S.get()}});
    I.updP = P.bindGroup("update_p", {{0, L0.params.get()}, {1, z0}, {2, I.pv.get()}, {10, I.S.get()}});
    I.alpha = P.bindGroup("cg_alpha", {{10, I.S.get()}});
    I.beta = P.bindGroup("cg_beta", {{10, I.S.get()}});
    for (int s = 0; s < 10; s++) I.reduce.push_back(P.bindGroup("reduce", {{10, I.S.get()}, {11, I.partials.get()}, {12, I.red.get(), uint64_t(s) * 256, 16}}));
    I.lzStore = P.bindGroup("lz_store", {{10, I.S.get()}});
    trace.lap("  GPU smoother eigenvalues");
    // the smoothers' eigenvalue estimates, where VoxelFEA has none yet (it leaves them to the GPU),
    // then their Chebyshev coefficients
    for (size_t l = 0; l + 1 < I.L.size(); l++) {
        Level& lv = fea.levels[l];
        if (!(lv.lmax > 0)) {
            lv.lmax = I.lanczosMax(l);
            lv.omega = 1.2 / (lv.lmax * 1.05);
        }
        const auto cc = chebyshevCoefficients(lv.lmax);
        const float cheb[3] = {float(cc.first), float(cc.c1), float(cc.c2)};
        wgpuQueueWriteBuffer(ctx->queue, I.L[l].params.get(), 48, cheb, 12);
    }
    gpuPopErrors(*ctx, "GPU solver setup failed");
}

GpuFeaSolver::~GpuFeaSolver() = default;

void GpuFeaSolver::precondition(const double* r, double* z) {
    Impl& I = *impl_;
    const Level& lv = fea_.levels[0];
    const int64_t n = I.n;
    // the V-cycle is linear: scale r to order one so float32 neither underflows nor loses digits
    const double rmax = parallelMax(n, [&](int64_t lo, int64_t hi) {
        double m = 0;
        for (int64_t i = lo; i < hi; i++)
            if (!lv.fixed[i]) m = std::max(m, std::abs(r[i]));
        return m;
    });
    if (!(rmax > 0)) { std::fill(z, z + n, 0.0); return; }
    I.upload(I.L[0].r.get(), r, 1 / rmax, lv);
    auto enc = I.encoder();
    auto pass = I.begin(enc);
    I.vcycle(pass, 0);
    I.submit(enc, pass);
    I.download(I.result(0), [&](int64_t i, float v) { z[i] = lv.fixed[i] ? 0 : double(v) * rmax; });
}

SolveResult GpuFeaSolver::solve(const std::vector<double>& f, const SolveOptions& opts) {
    // Conjugate gradients preconditioned by the multigrid V-cycle, all on the GPU in 32-bit (the
    // CPU solver's algorithm, flexible Polak-Ribiere beta included), kept to 64-bit accuracy by
    // reliable updates (Sleijpen & van der Vorst 1996, as in mixed-precision GPU solvers): each
    // time the GPU's residual has dropped tenfold, its solution is added to the 64-bit one on the
    // CPU and the true residual f - K u, computed there in 64-bit, replaces the GPU's.
    Impl& I = *impl_;
    const Level& L = fea_.levels[0];
    const int64_t n = L.nDof;
    if (int64_t(f.size()) != n || (opts.x0 && int64_t(opts.x0->size()) != n)) throw std::invalid_argument("Force or displacement vector has the wrong size.");
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
    std::vector<double> r(n);
    auto trueResidual = [&] {
        fea_.apply(0, u.data(), r.data());
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) r[i] = L.fixed[i] ? 0 : f[i] - r[i]; });
        return std::sqrt(dot(r.data(), r.data(), n)) / bnorm;
    };
    double rel = trueResidual();
    int it = 0;
    if (rel > opts.tol) {
        // float32 units: the first residual scaled to order one
        const double rmax = parallelMax(n, [&](int64_t lo, int64_t hi) {
            double m = 0;
            for (int64_t i = lo; i < hi; i++) m = std::max(m, std::abs(r[i]));
            return m;
        });
        const double scale = 1 / rmax;
        I.upload(I.L[0].r.get(), r.data(), scale, L);
        I.clear({I.x.get(), I.pv.get(), I.qv.get(), I.S.get()});  // beta = 0 on the first pass
        auto foldIn = [&] {
            I.download(I.x.get(), [&](int64_t i, float v) { u[i] += double(v) / scale; });
            I.clear({I.x.get()});
            return trueResidual();
        };
        // fold when the GPU residual is a tenth of its peak since the last fold: its drift from the
        // true residual stays a small fraction of the peak
        double relUpdated = rel, peak = rel, lastEst = rel;
        int batch = 1;
        bool dirty = false;
        while (it < opts.maxIter) {
            const auto t0 = std::chrono::steady_clock::now();
            auto enc = I.encoder();
            auto pass = I.begin(enc);
            const int k = std::min(batch, opts.maxIter - it);
            for (int q = 0; q < k; q++) {
                // z = M r; beta = max(0, z.(r - r_old) / rz) with r - r_old = -alpha q; p = z + beta p
                I.vcycle(pass, 0);
                I.dot(pass, I.L[0].r.get(), I.result(0), RZN);
                I.dot(pass, I.result(0), I.qv.get(), ZQ);
                I.run(pass, "cg_beta", I.beta.get(), 1);
                I.run(pass, "update_p", I.updP.get(), n);
                // q = K p; alpha = rz / p.q; x += alpha p; r -= alpha q
                I.run(pass, "matvec", I.mvP.get(), L.nNodes);
                I.dot(pass, I.pv.get(), I.qv.get(), PQ);
                I.run(pass, "cg_alpha", I.alpha.get(), 1);
                I.run(pass, "update_xr", I.updXR.get(), n);
            }
            I.dot(pass, I.L[0].r.get(), I.L[0].r.get(), RR);
            I.submit(enc, pass);
            const auto sc = I.scalars();
            it += k;
            dirty = true;
            if (!(sc[PQ] > 0) || !std::isfinite(sc[RR]) || !std::isfinite(sc[ALPHA])) break;  // lost positive-definiteness
            const double est = std::sqrt(double(sc[RR])) / scale / bnorm;
            if (opts.onProgress && opts.onProgress(it, std::min(est, relUpdated))) { res.cancelled = true; break; }
            peak = std::max(peak, est);
            // per-iteration reduction, for sizing the next batch
            const double rho = std::clamp(std::pow(est / lastEst, 1.0 / k), 0.05, 0.95);
            lastEst = est;
            if (est <= opts.tol || est <= 0.1 * peak) {
                rel = relUpdated = peak = lastEst = foldIn();
                dirty = false;
                if (rel <= opts.tol) break;
                I.upload(I.L[0].r.get(), r.data(), scale, L);
            }
            // wait for the GPU about every 4 ms (rarely enough to keep it busy), and not past convergence
            const double per = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / k;
            const int toGo = int(std::ceil(std::log(opts.tol / lastEst) / std::log(rho)));
            batch = std::clamp(std::min(int(0.004 / std::max(per, 1e-5)), toGo), 1, 8);
        }
        if (dirty) rel = foldIn();
    }
    res.iterations = it;
    res.residual = rel;
    res.converged = !res.cancelled && rel <= opts.tol * 10;
    return res;
}

std::string GpuFeaSolver::profile(int reps) {
    Impl& I = *impl_;
    auto time = [&](const std::function<void(WGPUComputePassEncoder)>& record) {
        auto enc = I.encoder();
        auto pass = I.begin(enc);
        record(pass);
        I.submit(enc, pass);
        I.ctx.wait();
        const auto t0 = std::chrono::steady_clock::now();
        enc = I.encoder();
        pass = I.begin(enc);
        for (int k = 0; k < reps; k++) record(pass);
        I.submit(enc, pass);
        I.ctx.wait();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
    };
    std::string out;
    char line[200];
    for (size_t l = 0; l < I.L.size(); l++) {
        auto& V = I.L[l];
        const uint64_t nN = V.lv->nNodes;
        if (V.last) {
            std::snprintf(line, sizeof line, "L%zu coarsest m=%lld: %.3f ms\n", l, (long long)I.coarseM, time([&](auto p) { I.vcycle(p, l); }));
            out += line;
            continue;
        }
        const double mv = time([&](auto p) { I.run(p, "matvec", V.mv.get(), nN); });
        const double sm = time([&](auto p) { I.run(p, "cheb_b", V.chebB[0].get(), nN); });
        const double rs = time([&](auto p) { I.run(p, "restrict_", V.restrictBG[0].get(), I.L[l + 1].lv->nNodes); });
        const double pr = time([&](auto p) { I.run(p, "prolong", V.prolongBG[0].get(), nN); });
        const double vc = time([&](auto p) { I.vcycle(p, l); });
        std::snprintf(line, sizeof line, "L%zu nodes=%lld elems=%zu own matrices=%zu: matvec %.3f  smooth %.3f  restrict %.3f  prolong %.3f  | V-cycle from here %.3f ms\n", l,
                      (long long)nN, V.lv->elems.size(), V.lv->K.size() / 576, mv, sm, rs, pr, vc);
        out += line;
    }
    {
        I.scalars();
        auto t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; k++) I.scalars();
        const double tRead = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::vector<float> h(size_t(I.n), 1.f);
        t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; k++) { wgpuQueueWriteBuffer(I.ctx.queue, I.L[0].r.get(), 0, h.data(), uint64_t(I.n) * 4); I.ctx.wait(); }
        const double tWrite = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        t0 = std::chrono::steady_clock::now();
        for (int k = 0; k < reps; k++) I.download(I.x.get(), [&](int64_t i, float v) { h[i] = v; });
        const double tDown = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        std::snprintf(line, sizeof line, "read 64 B %.3f ms, write vector %.3f ms, read vector %.3f ms\n", tRead, tWrite, tDown);
        out += line;
    }
    std::vector<double> r(I.n, 1.0), z(I.n);
    precondition(r.data(), z.data());
    const auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < reps; k++) precondition(r.data(), z.data());
    std::snprintf(line, sizeof line, "precondition() incl. transfers: %.3f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps);
    return out + line;
}

#else

struct GpuFeaSolver::Impl {};
GpuFeaSolver::GpuFeaSolver(VoxelFEA& fea) : fea_(fea) { throw std::runtime_error("this build has no GPU support"); }
GpuFeaSolver::~GpuFeaSolver() = default;
void GpuFeaSolver::precondition(const double*, double*) {}
SolveResult GpuFeaSolver::solve(const std::vector<double>&, const SolveOptions&) { return {}; }
std::string GpuFeaSolver::profile(int) { return {}; }

#endif

}  // namespace ps
