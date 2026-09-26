// GPU lattice-Boltzmann solver (lbm.wgsl, the same shader the web app used).
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../gpu/gpu.hpp"
#include "lbm.hpp"

namespace ps {

#ifdef PARTS_SIM_HAS_GPU

namespace {

constexpr uint64_t PARAM_BYTES = 48;

class LbmGpu : public LbmSolver {
public:
    explicit LbmGpu(const LbmSetup& s) : ctx_(*GpuContext::get()), dims_(s.dims) {
        cells = int64_t(dims_[0]) * dims_[1] * dims_[2];
        const int64_t N = cells;
        const uint64_t fBytes = uint64_t(19) * N * 4;
        if (fBytes > ctx_.limits.maxStorageBufferBindingSize || fBytes > ctx_.limits.maxBufferSize)
            throw std::runtime_error("The flow grid is too large for this GPU.");
        uLat_ = s.uLat;
        tau0_ = float(3 * s.nuLat + 0.5);
        smag_ = float(18 * std::sqrt(2.0) * s.smagorinsky * s.smagorinsky);
        std::lock_guard<std::mutex> lock(ctx_.lock);
        gpuPushErrors(ctx_);
        prog_ = std::make_unique<GpuProgram>(ctx_, wgslSource("lbm"), std::vector<std::string>{"init", "step"});
        f_[0] = gpuBuffer(ctx_, fBytes);
        f_[1] = gpuBuffer(ctx_, fBytes);
        std::vector<uint32_t> solid(s.solid.begin(), s.solid.end());
        solid_ = gpuBuffer(ctx_, uint64_t(N) * 4, solid.data());
        macro_ = gpuBuffer(ctx_, uint64_t(N) * 16);
        std::vector<uint8_t> links((19 * N + 3) / 4 * 4, 0);
        if (!s.links.empty()) std::memcpy(links.data(), s.links.data(), s.links.size());
        links_ = gpuBuffer(ctx_, links.size(), links.data());
        read_ = gpuReadback(ctx_, uint64_t(N) * 16);
        const auto d = gpuDispatchSize(ctx_, uint64_t(N));
        wg_ = {d[0], d[1]};
        stride_ = d[2];
        params_ = gpuUniform(ctx_, PARAM_BYTES, paramBytes(0).data());
        // auto layouts hold only the bindings an entry point uses
        for (int p = 0; p < 2; p++) {
            initBG_[p] = prog_->bindGroup("init", {{0, params_.get()}, {2, f_[1 - p].get()}, {4, macro_.get()}});
            stepBG_[p] = prog_->bindGroup("step", {{0, params_.get()}, {1, f_[p].get()}, {2, f_[1 - p].get()},
                                                   {3, solid_.get()}, {4, macro_.get()}, {5, links_.get()}});
        }
        gpuPopErrors(ctx_, "Could not set up the GPU flow solver");
        resetLocked();
        ctx_.wait();
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(ctx_.lock);
        resetLocked();
        ctx_.wait();
    }

    void step(int count) override {
        std::lock_guard<std::mutex> lock(ctx_.lock);
        gpuPushErrors(ctx_);
        int done = 0;
        while (done < count) {
            // the inlet ramps up over the first steps: one submission per step with its own inlet
            // speed; afterwards the parameters are constant and a whole batch goes in one submission
            const bool ramp = steps < LBM_RAMP_STEPS;
            const int batch = ramp ? 1 : count - done;
            wgpuQueueWriteBuffer(ctx_.queue, params_.get(), 0, paramBytes(steps).data(), PARAM_BYTES);
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx_.device, nullptr);
            for (int s = 0; s < batch; s++) {
                WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
                wgpuComputePassEncoderSetPipeline(pass, prog_->pipeline("step"));
                wgpuComputePassEncoderSetBindGroup(pass, 0, stepBG_[parity_].get(), 0, nullptr);
                wgpuComputePassEncoderDispatchWorkgroups(pass, wg_[0], wg_[1], 1);
                wgpuComputePassEncoderEnd(pass);
                wgpuComputePassEncoderRelease(pass);
                parity_ ^= 1;
            }
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            wgpuQueueSubmit(ctx_.queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
            steps += batch;
            done += batch;
        }
        ctx_.wait();
        gpuPopErrors(ctx_, "GPU flow step failed");
    }

    std::vector<float> macro() override {
        std::lock_guard<std::mutex> lock(ctx_.lock);
        const auto bytes = gpuRead(ctx_, macro_.get(), read_.get(), uint64_t(cells) * 16);
        std::vector<float> out(4 * cells);
        std::memcpy(out.data(), bytes.data(), bytes.size());
        return out;
    }

private:
    std::vector<uint8_t> paramBytes(int64_t step) const {
        std::vector<uint8_t> b(PARAM_BYTES, 0);
        const uint32_t u[8] = {uint32_t(dims_[0]), uint32_t(dims_[1]), uint32_t(dims_[2]), uint32_t(cells), stride_, 0, 0, 0};
        const float f[4] = {tau0_, float(lbmInletVelocity(uLat_, step)), smag_, 0};
        std::memcpy(b.data(), u, 32);
        std::memcpy(b.data() + 32, f, 16);
        return b;
    }

    void resetLocked() {
        steps = 0;
        parity_ = 0;
        wgpuQueueWriteBuffer(ctx_.queue, params_.get(), 0, paramBytes(0).data(), PARAM_BYTES);
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx_.device, nullptr);
        // init writes into the "out" buffer of a parity; run it for both
        for (int p : {1, 0}) {
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
            wgpuComputePassEncoderSetPipeline(pass, prog_->pipeline("init"));
            wgpuComputePassEncoderSetBindGroup(pass, 0, initBG_[p].get(), 0, nullptr);
            wgpuComputePassEncoderDispatchWorkgroups(pass, wg_[0], wg_[1], 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
        }
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(ctx_.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }

    GpuContext& ctx_;
    std::array<int, 3> dims_;
    double uLat_;
    float tau0_, smag_;
    std::unique_ptr<GpuProgram> prog_;
    GpuBuffer f_[2], solid_, macro_, links_, read_, params_;
    GpuBindGroup initBG_[2], stepBG_[2];
    std::array<uint32_t, 2> wg_{};
    uint32_t stride_ = 0;
    int parity_ = 0;
};

}  // namespace

std::unique_ptr<LbmSolver> makeLbmGpu(const LbmSetup& s) {
    if (!gpuAvailable()) throw std::runtime_error(gpuUnavailableReason().empty() ? "no compatible GPU" : gpuUnavailableReason());
    return std::make_unique<LbmGpu>(s);
}

int64_t lbmGpuMaxCells() {
    GpuContext* ctx = gpuAvailable() ? GpuContext::get() : nullptr;
    if (!ctx) return 0;
    return int64_t(std::min(ctx->limits.maxStorageBufferBindingSize, ctx->limits.maxBufferSize) / (19 * 4));
}

#else

std::unique_ptr<LbmSolver> makeLbmGpu(const LbmSetup&) { throw std::runtime_error("built without GPU support"); }
int64_t lbmGpuMaxCells() { return 0; }

#endif

}  // namespace ps
