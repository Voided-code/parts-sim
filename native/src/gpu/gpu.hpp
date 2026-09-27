// GPU compute through wgpu-native (WebGPU's C API): Metal on macOS, Vulkan or DirectX 12 on
// Windows, Vulkan on Linux. One shared device; small RAII helpers for buffers, pipelines and
// bind groups; blocking read-backs for the solvers' worker threads.
#pragma once

#include <array>
#include <functional>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#ifdef PARTS_SIM_HAS_GPU
#include <webgpu.h>
#endif

namespace ps {

/** Is a usable GPU adapter present (tries once)? */
bool gpuAvailable();
/** Why the GPU is not used ("" when it is available). */
std::string gpuUnavailableReason();
/** Adapter name for the UI (e.g. "Apple M5 (Metal)"). */
std::string gpuName();

/** WGSL sources embedded at build time: "fea", "explicit", "lbm". */
const char* wgslSource(const std::string& name);

#ifdef PARTS_SIM_HAS_GPU

struct GpuContext {
    WGPUInstance instance = nullptr;
    WGPUAdapter adapter = nullptr;
    WGPUDevice device = nullptr;
    WGPUQueue queue = nullptr;
    WGPULimits limits{};
    std::string name, error;
    bool shaderF16 = false;  // 16-bit floats in shaders (WGSL "enable f16")
    std::mutex lock;  // one GPU job at a time

    static GpuContext* get();  // nullptr when unavailable
    void wait();               // block until submitted work completes
};

/** Owning handle for a WebGPU object. */
template <class T, void (*Release)(T)>
class Handle {
public:
    Handle() = default;
    explicit Handle(T v) : v_(v) {}
    Handle(Handle&& o) noexcept : v_(o.v_) { o.v_ = nullptr; }
    Handle& operator=(Handle&& o) noexcept {
        if (this != &o) { reset(); v_ = o.v_; o.v_ = nullptr; }
        return *this;
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { reset(); }
    void reset() {
        if (v_) Release(v_);
        v_ = nullptr;
    }
    T get() const { return v_; }
    operator T() const { return v_; }
    explicit operator bool() const { return v_ != nullptr; }

private:
    T v_ = nullptr;
};

using GpuBuffer = Handle<WGPUBuffer, wgpuBufferRelease>;
using GpuPipeline = Handle<WGPUComputePipeline, wgpuComputePipelineRelease>;
using GpuBindGroup = Handle<WGPUBindGroup, wgpuBindGroupRelease>;
using GpuModule = Handle<WGPUShaderModule, wgpuShaderModuleRelease>;

/** A compiled WGSL module with one compute pipeline per entry point (auto layouts). */
class GpuProgram {
public:
    GpuProgram(GpuContext& ctx, const std::string& wgsl, const std::vector<std::string>& entryPoints);
    WGPUComputePipeline pipeline(const std::string& entry) const { return pipelines_.at(entry).get(); }
    /** Bind group for group 0 of an entry point: binding -> (buffer, offset, size; size 0 = whole buffer). */
    struct Bind { uint32_t binding; WGPUBuffer buffer; uint64_t offset = 0; uint64_t size = 0; };
    GpuBindGroup bindGroup(const std::string& entry, const std::vector<Bind>& binds) const;

private:
    GpuContext& ctx_;
    GpuModule module_;
    std::map<std::string, GpuPipeline> pipelines_;
};

/** Storage buffer (also copy source/destination) of `bytes`, optionally initialized. */
GpuBuffer gpuBuffer(GpuContext& ctx, uint64_t bytes, const void* data = nullptr, WGPUBufferUsage usage = 0);
GpuBuffer gpuUniform(GpuContext& ctx, uint64_t bytes, const void* data);
GpuBuffer gpuReadback(GpuContext& ctx, uint64_t bytes);
/** Copy `bytes` from src (at srcOffset) through the read-back buffer and return them. */
std::vector<uint8_t> gpuRead(GpuContext& ctx, WGPUBuffer src, WGPUBuffer staging, uint64_t bytes, uint64_t srcOffset = 0);
/** The same, handing `use` the mapped bytes in place (no copy; valid only during the call). */
void gpuReadInto(GpuContext& ctx, WGPUBuffer src, WGPUBuffer staging, uint64_t bytes, const std::function<void(const void*)>& use, uint64_t srcOffset = 0);
/** Waits for the work already queued to fill the read-back buffer, then hands `use` its bytes. */
void gpuMapRead(GpuContext& ctx, WGPUBuffer staging, uint64_t bytes, const std::function<void(const void*)>& use);
/** Processes GPU events until `done` is set by a callback (low latency for short jobs). */
void gpuWaitFor(GpuContext& ctx, const bool& done);
/** Dispatch geometry for `count` invocations at workgroup size 64: {x, y, stride}. */
std::array<uint32_t, 3> gpuDispatchSize(GpuContext& ctx, uint64_t count);
/** Throws with the message of any validation or out-of-memory error raised since `push`. */
void gpuPushErrors(GpuContext& ctx);
void gpuPopErrors(GpuContext& ctx, const char* what);

#endif

}  // namespace ps
