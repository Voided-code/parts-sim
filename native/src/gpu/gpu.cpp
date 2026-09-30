#include "gpu.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#ifdef PARTS_SIM_HAS_GPU
#include <wgpu.h>
#endif

namespace ps {

#ifdef PARTS_SIM_HAS_GPU

namespace {

std::string str(WGPUStringView v) { return v.data ? std::string(v.data, v.length == WGPU_STRLEN ? std::strlen(v.data) : v.length) : std::string(); }

GpuContext* create() {
    auto* ctx = new GpuContext();
    // PARTS_SIM_GPU_BACKEND=vulkan | dx12 | metal picks the backend (to compare them); by default
    // wgpu chooses (Vulkan first on Windows and Linux)
    WGPUInstanceExtras extras = {};
    extras.chain.sType = static_cast<WGPUSType>(WGPUSType_InstanceExtras);
    WGPUInstanceDescriptor idesc = WGPU_INSTANCE_DESCRIPTOR_INIT;
    if (const char* env = std::getenv("PARTS_SIM_GPU_BACKEND"); env && *env) {
        std::string b(env);
        for (auto& ch : b) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        extras.backends = b == "vulkan" ? WGPUInstanceBackend_Vulkan : b == "dx12" || b == "d3d12" ? WGPUInstanceBackend_DX12 : b == "metal" ? WGPUInstanceBackend_Metal : 0;
        idesc.nextInChain = &extras.chain;
    }
    ctx->instance = wgpuCreateInstance(idesc.nextInChain ? &idesc : nullptr);
    if (!ctx->instance) { ctx->error = "no WebGPU instance"; return ctx; }
    WGPURequestAdapterOptions opts = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    opts.powerPreference = WGPUPowerPreference_HighPerformance;
    struct AdapterReq { WGPUAdapter adapter = nullptr; bool done = false; std::string msg; } areq;
    WGPURequestAdapterCallbackInfo acb = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    acb.mode = WGPUCallbackMode_AllowProcessEvents;
    acb.callback = [](WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* u1, void*) {
        auto* r = static_cast<AdapterReq*>(u1);
        if (status == WGPURequestAdapterStatus_Success) r->adapter = adapter;
        else r->msg = str(message);
        r->done = true;
    };
    acb.userdata1 = &areq;
    wgpuInstanceRequestAdapter(ctx->instance, &opts, acb);
    for (int i = 0; i < 10000 && !areq.done; i++) wgpuInstanceProcessEvents(ctx->instance);
    if (!areq.adapter) { ctx->error = areq.msg.empty() ? "no GPU adapter" : areq.msg; return ctx; }
    ctx->adapter = areq.adapter;
    WGPULimits alimits = WGPU_LIMITS_INIT;
    wgpuAdapterGetLimits(ctx->adapter, &alimits);
    WGPUAdapterInfo info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(ctx->adapter, &info) == WGPUStatus_Success) {
        const char* backend = info.backendType == WGPUBackendType_Metal ? "Metal" : info.backendType == WGPUBackendType_Vulkan ? "Vulkan"
                              : info.backendType == WGPUBackendType_D3D12 ? "DirectX 12" : "GPU";
        ctx->name = str(info.device) + " (" + backend + ")";
        wgpuAdapterInfoFreeMembers(info);
    }
    // the solvers need the adapter's full buffer sizes; everything else stays at defaults
    WGPULimits req = WGPU_LIMITS_INIT;
    req.maxStorageBufferBindingSize = alimits.maxStorageBufferBindingSize;
    req.maxBufferSize = alimits.maxBufferSize;
    req.maxComputeWorkgroupsPerDimension = alimits.maxComputeWorkgroupsPerDimension;
    req.maxStorageBuffersPerShaderStage = std::min<uint32_t>(alimits.maxStorageBuffersPerShaderStage, 10);
    WGPUDeviceDescriptor dd = WGPU_DEVICE_DESCRIPTOR_INIT;
    dd.requiredLimits = &req;
    // 16-bit floats where the adapter has them (the flow solver stores its populations in them), and
    // timestamp queries (the benchmarks' GPU time)
    const WGPUFeatureName f16 = WGPUFeatureName_ShaderF16, ts = WGPUFeatureName_TimestampQuery;
    std::vector<WGPUFeatureName> features;
    if (wgpuAdapterHasFeature(ctx->adapter, f16)) features.push_back(f16);
    if (wgpuAdapterHasFeature(ctx->adapter, ts)) features.push_back(ts);
    dd.requiredFeatureCount = features.size();
    dd.requiredFeatures = features.data();
    dd.uncapturedErrorCallbackInfo.callback = [](WGPUDevice const*, WGPUErrorType, WGPUStringView, void*, void*) {};
    struct DeviceReq { WGPUDevice device = nullptr; bool done = false; std::string msg; } dreq;
    WGPURequestDeviceCallbackInfo dcb = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    dcb.mode = WGPUCallbackMode_AllowProcessEvents;
    dcb.callback = [](WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* u1, void*) {
        auto* r = static_cast<DeviceReq*>(u1);
        if (status == WGPURequestDeviceStatus_Success) r->device = device;
        else r->msg = str(message);
        r->done = true;
    };
    dcb.userdata1 = &dreq;
    wgpuAdapterRequestDevice(ctx->adapter, &dd, dcb);
    for (int i = 0; i < 10000 && !dreq.done; i++) wgpuInstanceProcessEvents(ctx->instance);
    if (!dreq.device) { ctx->error = dreq.msg.empty() ? "could not open the GPU device" : dreq.msg; return ctx; }
    ctx->device = dreq.device;
    ctx->queue = wgpuDeviceGetQueue(ctx->device);
    ctx->shaderF16 = wgpuDeviceHasFeature(ctx->device, f16);
    ctx->timestamps = wgpuDeviceHasFeature(ctx->device, ts);
    if (ctx->timestamps) ctx->timestampPeriod = wgpuQueueGetTimestampPeriod(ctx->queue);
    wgpuDeviceGetLimits(ctx->device, &ctx->limits);
    return ctx;
}

}  // namespace

GpuContext* GpuContext::get() {
    static GpuContext* ctx = create();
    return ctx->device ? ctx : nullptr;
}

void GpuContext::wait() {
    // an empty completion callback on the queue marks when everything submitted so far is done
    bool done = false;
    WGPUQueueWorkDoneCallbackInfo cb = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUQueueWorkDoneStatus, WGPUStringView, void* u1, void*) { *static_cast<bool*>(u1) = true; };
    cb.userdata1 = &done;
    wgpuQueueOnSubmittedWorkDone(queue, cb);
    gpuWaitFor(*this, done);
}

void gpuWaitFor(GpuContext& ctx, const bool& done) {
    // A blocking poll sleeps in 1 ms steps on some backends (Metal), which dominates short GPU
    // jobs such as one solver iteration: spin on non-blocking polls for the first few
    // milliseconds, then block for long jobs. On Metal a blocking poll can also oversleep by
    // seconds with work queued behind the awaited submission (the flow solver keeps two batches in
    // flight), so there it keeps polling with short naps.
    const auto t0 = std::chrono::steady_clock::now();
    while (!done) {
        const bool spin = std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(4);
#ifdef __APPLE__
        wgpuDevicePoll(ctx.device, false, nullptr);
        wgpuInstanceProcessEvents(ctx.instance);
        if (!done) {
            if (spin) std::this_thread::yield();
            else std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
#else
        wgpuDevicePoll(ctx.device, !spin, nullptr);
        wgpuInstanceProcessEvents(ctx.instance);
        if (spin && !done) std::this_thread::yield();
#endif
    }
}

GpuProgram::GpuProgram(GpuContext& ctx, const std::string& wgsl, const std::vector<std::string>& entries) : ctx_(ctx) {
    gpuPushErrors(ctx);
    WGPUShaderSourceWGSL src = WGPU_SHADER_SOURCE_WGSL_INIT;
    src.code = WGPUStringView{wgsl.c_str(), wgsl.size()};
    WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    md.nextInChain = &src.chain;
    module_ = GpuModule(wgpuDeviceCreateShaderModule(ctx.device, &md));
    for (const auto& e : entries) {
        WGPUComputePipelineDescriptor pd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        pd.compute.module = module_.get();
        pd.compute.entryPoint = WGPUStringView{e.c_str(), e.size()};
        pipelines_.emplace(e, GpuPipeline(wgpuDeviceCreateComputePipeline(ctx.device, &pd)));
    }
    gpuPopErrors(ctx, "GPU shader failed to compile");
}

GpuBindGroup GpuProgram::bindGroup(const std::string& entry, const std::vector<Bind>& binds) const {
    const WGPUComputePipeline p = pipelines_.at(entry).get();
    WGPUBindGroupLayout layout = wgpuComputePipelineGetBindGroupLayout(p, 0);
    std::vector<WGPUBindGroupEntry> es;
    for (const auto& b : binds) {
        WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
        e.binding = b.binding;
        e.buffer = b.buffer;
        e.offset = b.offset;
        e.size = b.size ? b.size : WGPU_WHOLE_SIZE;
        es.push_back(e);
    }
    WGPUBindGroupDescriptor d = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    d.layout = layout;
    d.entryCount = es.size();
    d.entries = es.data();
    GpuBindGroup g(wgpuDeviceCreateBindGroup(ctx_.device, &d));
    wgpuBindGroupLayoutRelease(layout);
    return g;
}

GpuBuffer gpuBuffer(GpuContext& ctx, uint64_t bytes, const void* data, WGPUBufferUsage usage) {
    WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
    d.size = std::max<uint64_t>(16, (bytes + 3) / 4 * 4);
    d.usage = usage ? usage : WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    GpuBuffer b(wgpuDeviceCreateBuffer(ctx.device, &d));
    if (data && bytes) wgpuQueueWriteBuffer(ctx.queue, b.get(), 0, data, (bytes + 3) / 4 * 4 == bytes ? bytes : bytes);
    return b;
}

GpuBuffer gpuUniform(GpuContext& ctx, uint64_t bytes, const void* data) {
    return gpuBuffer(ctx, bytes, data, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);
}

GpuBuffer gpuReadback(GpuContext& ctx, uint64_t bytes) {
    WGPUBufferDescriptor d = WGPU_BUFFER_DESCRIPTOR_INIT;
    d.size = std::max<uint64_t>(16, (bytes + 3) / 4 * 4);
    d.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    return GpuBuffer(wgpuDeviceCreateBuffer(ctx.device, &d));
}

std::vector<uint8_t> gpuRead(GpuContext& ctx, WGPUBuffer src, WGPUBuffer staging, uint64_t bytes, uint64_t srcOffset) {
    std::vector<uint8_t> out;
    gpuReadInto(ctx, src, staging, bytes, [&](const void* p) { out.assign(static_cast<const uint8_t*>(p), static_cast<const uint8_t*>(p) + bytes); }, srcOffset);
    return out;
}

void gpuReadInto(GpuContext& ctx, WGPUBuffer src, WGPUBuffer staging, uint64_t bytes, const std::function<void(const void*)>& use, uint64_t srcOffset) {
    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx.device, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(enc, src, srcOffset, staging, 0, bytes);
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
    wgpuQueueSubmit(ctx.queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    gpuMapRead(ctx, staging, bytes, use);
}

void gpuMapRead(GpuContext& ctx, WGPUBuffer staging, uint64_t bytes, const std::function<void(const void*)>& use) {
    struct Req { bool done = false; WGPUMapAsyncStatus status = WGPUMapAsyncStatus_Error; } req;
    WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
    cb.mode = WGPUCallbackMode_AllowProcessEvents;
    cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
        auto* r = static_cast<Req*>(u1);
        r->status = status;
        r->done = true;
    };
    cb.userdata1 = &req;
    wgpuBufferMapAsync(staging, WGPUMapMode_Read, 0, size_t(bytes), cb);
    gpuWaitFor(ctx, req.done);
    if (req.status != WGPUMapAsyncStatus_Success) throw std::runtime_error("Could not read results back from the GPU.");
    const void* p = wgpuBufferGetConstMappedRange(staging, 0, size_t(bytes));
    try {
        use(p);
    } catch (...) {
        wgpuBufferUnmap(staging);
        throw;
    }
    wgpuBufferUnmap(staging);
}

std::array<uint32_t, 3> gpuDispatchSize(GpuContext& ctx, uint64_t count) {
    const uint64_t groups = std::max<uint64_t>(1, (count + 63) / 64);
    const uint32_t x = uint32_t(std::min<uint64_t>(groups, ctx.limits.maxComputeWorkgroupsPerDimension));
    return {x, uint32_t((groups + x - 1) / x), x * 64};
}

void gpuPushErrors(GpuContext& ctx) {
    wgpuDevicePushErrorScope(ctx.device, WGPUErrorFilter_Validation);
    wgpuDevicePushErrorScope(ctx.device, WGPUErrorFilter_OutOfMemory);
}

void gpuPopErrors(GpuContext& ctx, const char* what) {
    struct Req { bool done = false; std::string msg; } oom, invalid;
    auto pop = [&](Req& r) {
        WGPUPopErrorScopeCallbackInfo cb = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
        cb.callback = [](WGPUPopErrorScopeStatus, WGPUErrorType type, WGPUStringView message, void* u1, void*) {
            auto* q = static_cast<Req*>(u1);
            if (type != WGPUErrorType_NoError) q->msg = str(message).empty() ? "GPU error" : str(message);
            q->done = true;
        };
        cb.userdata1 = &r;
        wgpuDevicePopErrorScope(ctx.device, cb);
        for (int i = 0; i < 100000 && !r.done; i++) {
            wgpuDevicePoll(ctx.device, false, nullptr);
            wgpuInstanceProcessEvents(ctx.instance);
        }
    };
    pop(oom);
    pop(invalid);
    const std::string& m = !oom.msg.empty() ? oom.msg : invalid.msg;
    if (!m.empty()) throw std::runtime_error(std::string(what) + ": " + m.substr(0, 400));
}

#endif

bool gpuAvailable() {
#ifdef PARTS_SIM_HAS_GPU
    if (const char* env = std::getenv("PARTS_SIM_NO_GPU"); env && *env) return false;
    return GpuContext::get() != nullptr;
#else
    return false;
#endif
}

std::string gpuUnavailableReason() {
#ifdef PARTS_SIM_HAS_GPU
    if (gpuAvailable()) return "";
    static GpuContext* dummy = nullptr;
    (void)dummy;
    return "no usable GPU adapter was found";
#else
    return "this build has no GPU support";
#endif
}

std::string gpuName() {
#ifdef PARTS_SIM_HAS_GPU
    if (auto* c = GpuContext::get()) return c->name;
#endif
    return "";
}

bool gpuShaderF16() {
#ifdef PARTS_SIM_HAS_GPU
    return gpuAvailable() && GpuContext::get()->shaderF16;
#else
    return false;
#endif
}

}  // namespace ps
