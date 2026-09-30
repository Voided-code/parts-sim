// GPU solver of the v1 flow engine: native/shaders/flow.wgsl (shared with the web app) after a
// prelude for the population buffers (src/cfd/flow-gpu.js does the same in the browser).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "../gpu/gpu.hpp"
#include "flow.hpp"

namespace ps::flow {

#ifdef PARTS_SIM_HAS_GPU

namespace {

constexpr uint32_t REC = 59, R_Q = 3, R_N = 8, R_AUX = 10, R_RHO = 32, R_BB = 36, R_SAMP = 54, R_Y2 = 55, R_AREA = 56;
constexpr int MAX_STEPS = 256, RING = 4, REDUCE_GROUPS = 64;
constexpr uint64_t PARAM_BYTES = 64;
constexpr double SUBMIT_SECONDS = 0.040;  // GPU work per submission (see flow-gpu.js SUBMIT_MS)
constexpr double MAX_COARSE = 1e6;

uint16_t f16bits(float v) {
    uint32_t x;
    std::memcpy(&x, &v, 4);
    const uint32_t sign = (x >> 16) & 0x8000;
    const int e = int((x >> 23) & 0xff) - 112;
    const uint32_t m = x & 0x7fffff;
    if (e <= 0) return uint16_t(sign);
    if (e >= 31) return uint16_t(sign | 0x7c00);
    const uint32_t r = (uint32_t(e) << 10) | (m >> 13);
    return uint16_t(sign | (r + ((m >> 12) & 1)));
}

struct PopBuf {
    int base, count;
    uint64_t bytes;
};

// slots of a direction pair share a buffer; as many as one binding holds (flow-gpu.js populationBuffers)
std::vector<PopBuf> populationBuffers(int64_t N, bool half, uint64_t maxBytes) {
    const uint64_t slotBytes = uint64_t(N) * (half ? 2 : 4);
    std::vector<std::vector<int>> units{{0}};
    for (int p = 0; p < 9; p++) units.push_back({2 * p + 1, 2 * p + 2});
    std::vector<PopBuf> bufs;
    for (const auto& u : units) {
        if (!bufs.empty() && (bufs.back().count + u.size()) * slotBytes <= maxBytes) bufs.back().count += int(u.size());
        else {
            if (u.size() * slotBytes > maxBytes) throw std::runtime_error("The flow grid is too large for this GPU.");
            bufs.push_back({u[0], int(u.size()), 0});
        }
    }
    for (auto& b : bufs) b.bytes = (b.count * slotBytes + 3) / 4 * 4;
    return bufs;
}

std::string prelude(bool half, const std::vector<PopBuf>& bufs) {
    std::string s = half ? "enable f16;\n" : "";
    const char* T = half ? "f16" : "f32";
    for (size_t i = 0; i < bufs.size(); i++)
        s += "@group(0) @binding(" + std::to_string(i + 1) + ") var<storage, read_write> F" + std::to_string(i) + ": array<" + T + ">;\n";
    for (int P : {0, 1, 3, 5, 7, 9, 11, 13, 15, 17}) {
        size_t b = 0;
        while (!(P >= bufs[b].base && P < bufs[b].base + bufs[b].count)) b++;
        char w[32];
        std::snprintf(w, sizeof w, "%.9g", W[P]);
        const std::string idx = "(" + std::to_string(P - bufs[b].base) + "u + s) * P.n + k", fb = "F" + std::to_string(b);
        if (half) {
            s += "fn ld" + std::to_string(P) + "(s: u32, k: u32) -> f32 { return f32(" + fb + "[" + idx + "]) + " + w + "; }\n";
            s += "fn st" + std::to_string(P) + "(s: u32, k: u32, v: f32) { " + fb + "[" + idx + "] = f16(v - " + w + "); }\n";
        } else {
            s += "fn ld" + std::to_string(P) + "(s: u32, k: u32) -> f32 { return " + fb + "[" + idx + "]; }\n";
            s += "fn st" + std::to_string(P) + "(s: u32, k: u32, v: f32) { " + fb + "[" + idx + "] = v; }\n";
        }
    }
    return s;
}

WGPUStringView sv(const char* s) { return WGPUStringView{s, std::strlen(s)}; }

class FlowGpu : public Solver {
public:
    FlowGpu(const Grid& g, const Params& p) : ctx_(*GpuContext::get()), grid_(g) {
        dims_ = g.dims;
        N_ = cells = g.N;
        nRec_ = uint32_t(g.rec.count);
        uLat_ = p.uLat;
        nu0_ = p.nuLat;
        tau0_ = float(3 * p.nuLat + 0.5);
        smag_ = float(18 * std::sqrt(2.0) * p.smagorinsky * p.smagorinsky);
        rr_ = p.rr;
        wallModel_ = p.wallModel && p.rr;
        belt_ = p.belt;
        half_ = p.half && ctx_.shaderF16;
        wg_ = {uint32_t(std::max(1, p.wgx)), uint32_t(std::max(1, p.wgy))};
        name = ctx_.name;
        std::lock_guard<std::mutex> lock(ctx_.lock);
        gpuPushErrors(ctx_);
        const uint64_t maxBytes = std::min<uint64_t>(ctx_.limits.maxStorageBufferBindingSize, ctx_.limits.maxBufferSize);
        bufs_ = populationBuffers(N_, half_, maxBytes);
        // the sampling kernel binds the population buffers, the cell kinds and the view fields
        if (bufs_.size() + 2 > ctx_.limits.maxStorageBuffersPerShaderStage) throw std::runtime_error("The flow grid is too large for this GPU.");
        for (const auto& b : bufs_) F_.push_back(gpuBuffer(ctx_, b.bytes));
        std::vector<uint32_t> kinds((N_ + 3) / 4, 0);
        std::memcpy(kinds.data(), g.kind.data(), size_t(N_));
        cells_ = gpuBuffer(ctx_, kinds.size() * 4, kinds.data());
        recWords_ = packRecords(g.rec);
        rec_ = gpuBuffer(ctx_, recWords_.size() * 4, recWords_.data());
        std::vector<uint32_t> faces;
        for (int64_t c = 0; c < N_; c++) {
            if (g.kind[c] != FACE) continue;
            const int nx = dims_[0], ny = dims_[1], nz = dims_[2];
            const int x = int(c % nx), y = int((c / nx) % ny), z = int(c / (int64_t(nx) * ny));
            uint32_t out = 0;
            for (int i = 1; i < 19; i++) {
                const int xx = x + CX[i], yy = y + CY[i], zz = z + CZ[i];
                if (xx < 0 || xx >= nx || yy < 0 || yy >= ny || (!g.periodicZ && (zz < 0 || zz >= nz))) out |= 1u << i;
            }
            uint32_t n0 = 0xffffffffu;
            if (x > 0) {
                const int z0 = g.periodicZ ? z : std::clamp(z, 1, nz - 2);
                const int64_t c0 = std::min(x, nx - 2) + int64_t(nx) * (std::clamp(y, 1, ny - 2) + int64_t(ny) * z0);
                n0 = g.kind[c0] == BULK || g.kind[c0] == WALL ? uint32_t(c0) : 0xfffffffeu;
            }
            faces.insert(faces.end(), {uint32_t(c), n0, out});
        }
        nFace_ = uint32_t(faces.size() / 3);
        if (faces.empty()) faces.assign(3, 0);
        faces_ = gpuBuffer(ctx_, faces.size() * 4, faces.data());
        std::vector<float> partial(9 * REDUCE_GROUPS + 1, 0.0f);
        partial.back() = float(REDUCE_GROUPS);
        partial_ = gpuBuffer(ctx_, partial.size() * 4, partial.data());
        history_ = gpuBuffer(ctx_, 12 * 4 * 16);
        for (auto& r : ring_) r.staging = gpuReadback(ctx_, 48);
        cf_ = std::max(1, int(std::ceil(std::cbrt(double(N_) / MAX_COARSE))));
        cdims_ = {(dims_[0] + cf_ - 1) / cf_, (dims_[1] + cf_ - 1) / cf_, (dims_[2] + cf_ - 1) / cf_};
        nCoarse_ = int64_t(cdims_[0]) * cdims_[1] * cdims_[2];
        coarse_ = gpuBuffer(ctx_, uint64_t(nCoarse_) * 32);
        coarseRead_ = gpuReadback(ctx_, uint64_t(nCoarse_) * 32);
        recRead_ = gpuReadback(ctx_, recWords_.size() * 4);
        const uint32_t align = ctx_.limits.minUniformBufferOffsetAlignment;
        paramStride_ = (PARAM_BYTES + align - 1) / align * align;
        params_ = gpuBuffer(ctx_, paramStride_ * (MAX_STEPS + 1), nullptr, WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst);

        const std::string src = prelude(half_, bufs_) + wgslSource("flow_gen") + wgslSource("flow");
        WGPUShaderSourceWGSL wsrc = WGPU_SHADER_SOURCE_WGSL_INIT;
        wsrc.code = WGPUStringView{src.c_str(), src.size()};
        WGPUShaderModuleDescriptor md = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
        md.nextInChain = &wsrc.chain;
        module_ = GpuModule(wgpuDeviceCreateShaderModule(ctx_.device, &md));
        std::vector<uint32_t> fb;
        for (size_t i = 0; i < F_.size(); i++) fb.push_back(uint32_t(i + 1));
        auto bindsF = [&](std::vector<std::pair<uint32_t, WGPUBuffer>> extra) {
            std::vector<std::pair<uint32_t, WGPUBuffer>> b;
            for (size_t i = 0; i < F_.size(); i++) b.push_back({uint32_t(i + 1), F_[i].get()});
            b.insert(b.end(), extra.begin(), extra.end());
            return b;
        };
        const double RR = rr_ ? 1 : 0;
        k_.init = make("init", bindsF({}), {}, {{"RR", RR}});
        k_.bulk = make("bulk", bindsF({{10, cells_.get()}}), {10}, {{"WGX", double(wg_[0])}, {"WGY", double(wg_[1])}, {"RR", RR}});
        k_.wall = make("wall", bindsF({{11, rec_.get()}}), {}, {{"RR", RR}});
        k_.face = make("face", bindsF({{12, faces_.get()}}), {12}, {{"RR", RR}});
        k_.reduce1 = make("reduce1", {{11, rec_.get()}, {13, partial_.get()}}, {}, {{"RR", RR}});
        k_.reduce2 = make("reduce2", {{13, partial_.get()}, {14, history_.get()}}, {}, {{"RR", RR}});
        k_.sample = make("sample", bindsF({{10, cells_.get()}, {15, coarse_.get()}}), {10}, {{"RR", RR}, {"CF", double(cf_)}});
        k_.clearRho = make("clearRho", {{11, rec_.get()}}, {}, {{"RR", RR}});
        gpuPopErrors(ctx_, "Could not set up the GPU flow solver");
        resetLocked();
        ctx_.wait();
    }

    ~FlowGpu() override {
        for (auto* k : {&k_.init, &k_.bulk, &k_.wall, &k_.face, &k_.reduce1, &k_.reduce2, &k_.sample, &k_.clearRho}) {
            if (k->playout) wgpuPipelineLayoutRelease(k->playout);
            if (k->layout) wgpuBindGroupLayoutRelease(k->layout);
        }
    }

    void reset() override {
        std::lock_guard<std::mutex> lock(ctx_.lock);
        resetLocked();
        ctx_.wait();
    }

    int submit(int count) override {
        std::lock_guard<std::mutex> lock(ctx_.lock);
        const int n = std::clamp(count, 1, MAX_STEPS);
        const int slot = ringNext_;
        ringNext_ = (slot + 1) % RING;
        auto& rs = ring_[slot];
        if (rs.pending) throw std::runtime_error("GPU flow batches submitted faster than collected.");
        writeParams(n + 1, slot);
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx_.device, nullptr);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
        const uint32_t bulk[3] = {(uint32_t(dims_[0]) + wg_[0] - 1) / wg_[0], (uint32_t(dims_[1]) + wg_[1] - 1) / wg_[1], uint32_t(dims_[2])};
        const auto wallG = groups1(nRec_), faceG = groups1(nFace_), sampleG = groups1(uint64_t(nCoarse_));
        int samples = 0;
        for (int s = 0; s < n; s++) {
            const uint32_t off = uint32_t(s * paramStride_);
            dispatch(pass, k_.bulk, off, bulk[0], bulk[1], bulk[2]);
            if (nRec_) dispatch(pass, k_.wall, off, wallG[0], wallG[1], 1);
            dispatch(pass, k_.face, off, faceG[0], faceG[1], 1);
            if (sampleEvery > 0 && (steps + s + 1) % sampleEvery == 0) {
                dispatch(pass, k_.sample, uint32_t((s + 1) * paramStride_), sampleG[0], sampleG[1], 1);
                samples++;
            }
        }
        dispatch(pass, k_.reduce1, 0, REDUCE_GROUPS, 1, 1);
        dispatch(pass, k_.reduce2, 0, 1, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderCopyBufferToBuffer(enc, history_.get(), uint64_t(slot) * 48, rs.staging.get(), 0, 48);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(ctx_.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
        rs.pending = true;
        rs.done = false;
        rs.steps = n;
        rs.t0 = std::chrono::steady_clock::now();
        WGPUBufferMapCallbackInfo cb = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
        cb.mode = WGPUCallbackMode_AllowProcessEvents;
        cb.callback = [](WGPUMapAsyncStatus status, WGPUStringView, void* u1, void*) {
            auto* r = static_cast<Slot*>(u1);
            r->ok = status == WGPUMapAsyncStatus_Success;
            r->done = true;
        };
        cb.userdata1 = &rs;
        wgpuBufferMapAsync(rs.staging.get(), WGPUMapMode_Read, 0, 48, cb);
        steps += n;
        samples_ += samples;
        rhoSteps_ += n;
        return slot;
    }

    Forces collect(int slot) override {
        auto& rs = ring_[slot];
        Forces f;
        if (!rs.pending) return f;
        {
            std::lock_guard<std::mutex> lock(ctx_.lock);
            gpuWaitFor(ctx_, rs.done);
            rs.pending = false;
            if (!rs.ok) throw std::runtime_error("Could not read the flow's forces back from the GPU.");
            float v[12];
            std::memcpy(v, wgpuBufferGetConstMappedRange(rs.staging.get(), 0, 48), 48);
            wgpuBufferUnmap(rs.staging.get());
            f.steps = rs.steps;
            for (int a = 0; a < 3; a++) { f.me[a] = v[a]; f.pressure[a] = v[3 + a]; f.friction[a] = v[6 + a]; }
        }
        lastBatchSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - rs.t0).count();
        return f;
    }

    void step(int count) override {
        if (!submitSteps_) submitSteps_ = std::clamp(int(2e7 / double(N_)), 1, MAX_STEPS);
        while (count > 0) {
            const int n = std::min(count, submitSteps_);
            const auto t0 = std::chrono::steady_clock::now();
            const Forces f = collect(submit(n));
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            submitSteps_ = std::clamp(int(std::lround(n * SUBMIT_SECONDS / std::max(1e-4, dt))), 1, MAX_STEPS);
            acc_.steps += f.steps;
            for (int a = 0; a < 3; a++) { acc_.me[a] += f.me[a]; acc_.pressure[a] += f.pressure[a]; acc_.friction[a] += f.friction[a]; }
            count -= n;
        }
    }

    Forces takeForces() override {
        Forces f = acc_;
        acc_ = Forces();
        return f;
    }

    void resetAverages() override {
        std::lock_guard<std::mutex> lock(ctx_.lock);
        samples_ = 0;
        rhoSteps_ = 0;
        const std::vector<float> zero(4 * size_t(nCoarse_), 0.0f);
        wgpuQueueWriteBuffer(ctx_.queue, coarse_.get(), 16 * uint64_t(nCoarse_), zero.data(), zero.size() * 4);
        if (nRec_) {
            writeParams(1, 0);
            WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx_.device, nullptr);
            WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
            const auto g = groups1(nRec_);
            dispatch(pass, k_.clearRho, 0, g[0], g[1], 1);
            wgpuComputePassEncoderEnd(pass);
            wgpuComputePassEncoderRelease(pass);
            WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
            wgpuQueueSubmit(ctx_.queue, 1, &cmd);
            wgpuCommandBufferRelease(cmd);
            wgpuCommandEncoderRelease(enc);
        }
    }

    std::vector<float> surfaceRho() override {
        std::lock_guard<std::mutex> lock(ctx_.lock);
        std::vector<float> out(nRec_, 0.0f);
        if (!nRec_) return out;
        const int64_t rs = rhoSteps_;
        gpuReadInto(ctx_, rec_.get(), recRead_.get(), uint64_t(nRec_) * REC * 4, [&](const void* p) {
            const float* w = static_cast<const float*>(p);
            if (rs)
                for (uint32_t r = 0; r < nRec_; r++) out[r] = w[size_t(r) * REC + R_RHO] / float(rs);
        });
        return out;
    }

    Fields fields() override {
        std::lock_guard<std::mutex> lock(ctx_.lock);
        Fields f;
        f.dims = cdims_;
        f.factor = cf_;
        f.samples = samples_;
        gpuReadInto(ctx_, coarse_.get(), coarseRead_.get(), uint64_t(nCoarse_) * 32, [&](const void* p) {
            const float* v = static_cast<const float*>(p);
            f.inst.assign(v, v + 4 * nCoarse_);
            f.avg.assign(v + 4 * nCoarse_, v + 8 * nCoarse_);
        });
        const float k = samples_ ? 1.0f / float(samples_) : 1.0f;
        for (int64_t c = 0; c < nCoarse_; c++) {
            for (int a = 0; a < 4; a++) f.avg[4 * c + a] *= k;
            if (f.inst[4 * c] == -2.0f) f.avg[4 * c] = -2.0f;
        }
        return f;
    }

    double lastBatchSeconds = 0;

private:
    struct Kernel {
        WGPUBindGroupLayout layout = nullptr;
        WGPUPipelineLayout playout = nullptr;
        GpuPipeline pipeline;
        GpuBindGroup group;
    };
    struct Slot {
        GpuBuffer staging;
        bool pending = false, done = false, ok = false;
        int steps = 0;
        std::chrono::steady_clock::time_point t0;
    };

    Kernel make(const char* entry, const std::vector<std::pair<uint32_t, WGPUBuffer>>& binds, const std::vector<uint32_t>& readOnly,
                const std::vector<std::pair<const char*, double>>& constants) {
        Kernel k;
        std::vector<WGPUBindGroupLayoutEntry> le;
        WGPUBindGroupLayoutEntry u = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        u.binding = 0;
        u.visibility = WGPUShaderStage_Compute;
        u.buffer.type = WGPUBufferBindingType_Uniform;
        u.buffer.hasDynamicOffset = 1;
        u.buffer.minBindingSize = PARAM_BYTES;
        le.push_back(u);
        for (const auto& [b, buf] : binds) {
            WGPUBindGroupLayoutEntry e = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
            e.binding = b;
            e.visibility = WGPUShaderStage_Compute;
            e.buffer.type = std::find(readOnly.begin(), readOnly.end(), b) != readOnly.end() ? WGPUBufferBindingType_ReadOnlyStorage : WGPUBufferBindingType_Storage;
            le.push_back(e);
        }
        WGPUBindGroupLayoutDescriptor ld = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        ld.entryCount = le.size();
        ld.entries = le.data();
        k.layout = wgpuDeviceCreateBindGroupLayout(ctx_.device, &ld);
        WGPUPipelineLayoutDescriptor pd = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
        pd.bindGroupLayoutCount = 1;
        pd.bindGroupLayouts = &k.layout;
        k.playout = wgpuDeviceCreatePipelineLayout(ctx_.device, &pd);
        std::vector<WGPUConstantEntry> ce;
        for (const auto& [key, value] : constants) {
            // only the overrides this entry point uses may be given
            if (!usesOverride(entry, key)) continue;
            WGPUConstantEntry c = WGPU_CONSTANT_ENTRY_INIT;
            c.key = sv(key);
            c.value = value;
            ce.push_back(c);
        }
        WGPUComputePipelineDescriptor cd = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
        cd.layout = k.playout;
        cd.compute.module = module_.get();
        cd.compute.entryPoint = sv(entry);
        cd.compute.constantCount = ce.size();
        cd.compute.constants = ce.data();
        k.pipeline = GpuPipeline(wgpuDeviceCreateComputePipeline(ctx_.device, &cd));
        std::vector<WGPUBindGroupEntry> be;
        WGPUBindGroupEntry pb = WGPU_BIND_GROUP_ENTRY_INIT;
        pb.binding = 0;
        pb.buffer = params_.get();
        pb.size = PARAM_BYTES;
        be.push_back(pb);
        for (const auto& [b, buf] : binds) {
            WGPUBindGroupEntry e = WGPU_BIND_GROUP_ENTRY_INIT;
            e.binding = b;
            e.buffer = buf;
            e.size = WGPU_WHOLE_SIZE;
            be.push_back(e);
        }
        WGPUBindGroupDescriptor bd = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        bd.layout = k.layout;
        bd.entryCount = be.size();
        bd.entries = be.data();
        k.group = GpuBindGroup(wgpuDeviceCreateBindGroup(ctx_.device, &bd));
        return k;
    }

    // WebGPU rejects constants for overrides an entry point does not reach; WGSL's are known here
    static bool usesOverride(const std::string& entry, const std::string& key) {
        if (key == "WGX" || key == "WGY") return entry == "bulk";
        if (key == "CF") return entry == "sample";
        return entry == "bulk" || entry == "wall" || entry == "face" || entry == "sample";  // RR: through collide / equilibrium
    }

    void dispatch(WGPUComputePassEncoder pass, const Kernel& k, uint32_t offset, uint32_t x, uint32_t y, uint32_t z) {
        wgpuComputePassEncoderSetPipeline(pass, k.pipeline.get());
        wgpuComputePassEncoderSetBindGroup(pass, 0, k.group.get(), 1, &offset);
        wgpuComputePassEncoderDispatchWorkgroups(pass, x, y, z);
    }

    std::array<uint32_t, 2> groups1(uint64_t count) const {
        const uint64_t g = std::max<uint64_t>(1, (count + 63) / 64);
        return g <= 65535 ? std::array<uint32_t, 2>{uint32_t(g), 1} : std::array<uint32_t, 2>{65535, uint32_t((g + 65534) / 65535)};
    }

    std::vector<uint32_t> packRecords(const Records& rec) const {
        std::vector<uint32_t> out(std::max<int64_t>(1, rec.count) * REC, 0);
        auto f32 = [&](size_t i, float v) { std::memcpy(&out[i], &v, 4); };
        for (int64_t r = 0; r < rec.count; r++) {
            const size_t b = size_t(r) * REC;
            out[b] = rec.cell[r];
            out[b + 1] = rec.mask[r] | (rec.onBelt[r] ? 1u : 0u);
            out[b + 2] = rec.groundMask[r];
            for (int k = 0; k < 18; k++) out[b + R_Q + k / 4] |= uint32_t(rec.q[18 * r + k]) << ((k % 4) * 8);
            out[b + R_N] = f16bits(rec.normal[3 * r]) | (uint32_t(f16bits(rec.normal[3 * r + 1])) << 16);
            out[b + R_N + 1] = f16bits(rec.normal[3 * r + 2]) | (uint32_t(f16bits(rec.dist[r])) << 16);
            for (int k = 1; k < 19; k++) { f32(b + R_AUX + k - 1, float(W[k])); f32(b + R_BB + k - 1, float(W[k])); }
            out[b + R_SAMP] = rec.samp.empty() ? 0 : rec.samp[r];
            f32(b + R_Y2, rec.y2.empty() ? 0.0f : rec.y2[r]);
            if (!rec.area.empty())
                for (int a = 0; a < 3; a++) f32(b + R_AREA + a, rec.area[3 * r + a]);
        }
        return out;
    }

    void writeParams(int count, int slot) {
        std::vector<uint8_t> buf(size_t(paramStride_) * count, 0);
        for (int s = 0; s < count; s++) {
            const int64_t step = steps + s;
            const float uin = float(inletVelocity(uLat_, step));
            const uint32_t u[8] = {uint32_t(dims_[0]), uint32_t(dims_[1]), uint32_t(dims_[2]), uint32_t(N_), nRec_, nFace_, grid_.periodicZ ? 1u : 0u, uint32_t(step & 1)};
            const float f[5] = {tau0_, smag_, uin, belt_ ? uin : 0.0f, float(nu0_)};
            const uint32_t w[3] = {wallModel_ ? 4u : 0u, uint32_t(slot), 0u};
            uint8_t* p = buf.data() + size_t(s) * paramStride_;
            std::memcpy(p, u, 32);
            std::memcpy(p + 32, f, 20);
            std::memcpy(p + 52, w, 12);
        }
        wgpuQueueWriteBuffer(ctx_.queue, params_.get(), 0, buf.data(), buf.size());
    }

    void resetLocked() {
        steps = 0;
        rhoSteps_ = 0;
        samples_ = 0;
        acc_ = Forces();
        wgpuQueueWriteBuffer(ctx_.queue, rec_.get(), 0, recWords_.data(), recWords_.size() * 4);
        const std::vector<float> zero(8 * size_t(nCoarse_), 0.0f);
        wgpuQueueWriteBuffer(ctx_.queue, coarse_.get(), 0, zero.data(), zero.size() * 4);
        writeParams(1, 0);
        WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(ctx_.device, nullptr);
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, nullptr);
        const auto g = groups1(uint64_t(N_));
        dispatch(pass, k_.init, 0, g[0], g[1], 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);
        WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, nullptr);
        wgpuQueueSubmit(ctx_.queue, 1, &cmd);
        wgpuCommandBufferRelease(cmd);
        wgpuCommandEncoderRelease(enc);
    }

    GpuContext& ctx_;
    Grid grid_;
    std::array<int, 3> dims_{};
    int64_t N_ = 0;
    uint32_t nRec_ = 0, nFace_ = 0;
    double uLat_ = 0.08, nu0_ = 1e-5;
    float tau0_ = 0.5f, smag_ = 0;
    bool rr_ = true, wallModel_ = true, belt_ = true, half_ = true;
    std::array<uint32_t, 2> wg_{64, 1};
    std::vector<PopBuf> bufs_;
    std::vector<GpuBuffer> F_;
    GpuBuffer cells_, rec_, faces_, partial_, history_, coarse_, coarseRead_, recRead_, params_;
    std::vector<uint32_t> recWords_;
    Slot ring_[RING];
    int ringNext_ = 0;
    int cf_ = 1;
    std::array<int, 3> cdims_{};
    int64_t nCoarse_ = 0;
    uint64_t paramStride_ = 256;
    GpuModule module_;
    struct {
        Kernel init, bulk, wall, face, reduce1, reduce2, sample, clearRho;
    } k_;
    int submitSteps_ = 0;
    int samples_ = 0;
    int64_t rhoSteps_ = 0;
    Forces acc_;
};

}  // namespace

std::unique_ptr<Solver> makeGpu(const Grid& grid, const Params& p) {
    if (!gpuAvailable()) throw std::runtime_error(gpuUnavailableReason().empty() ? "no compatible GPU" : gpuUnavailableReason());
    return std::make_unique<FlowGpu>(grid, p);
}

int64_t gpuMaxCells(bool half) {
    GpuContext* ctx = gpuAvailable() ? GpuContext::get() : nullptr;
    if (!ctx) return 0;
    const bool h = half && ctx->shaderF16;
    const double bind = double(std::min<uint64_t>(ctx->limits.maxStorageBufferBindingSize, ctx->limits.maxBufferSize));
    // population buffers of whole direction pairs, two storage bindings left for the other data
    const double buffers = std::min(6.0, double(ctx->limits.maxStorageBuffersPerShaderStage) - 2);
    return int64_t(0.95 * bind * buffers / (19.0 * (h ? 2 : 4)));
}

#else

std::unique_ptr<Solver> makeGpu(const Grid&, const Params&) { throw std::runtime_error("built without GPU support"); }
int64_t gpuMaxCells(bool) { return 0; }

#endif

}  // namespace ps::flow
