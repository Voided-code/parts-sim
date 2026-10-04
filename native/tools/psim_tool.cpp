// psim_tool: check, list and dump .psim files, and write the test fixture.
//
//   psim_tool verify <file>            reads every section; exit status 1 when the file is damaged
//   psim_tool inspect <file>           the same table as `node scripts/psim.mjs inspect`
//   psim_tool dump <file>              a stable, line-based text of the decoded content (see scripts/psim.mjs dump)
//   psim_tool make-fixture <out> [--brep]   writes the content of test/helpers/psim-content.mjs sampleContent()
//   psim_tool bench <file> [out.psim]  times a read and a write of the file's content (and keeps the written file)
//
// `dump` prints exactly what `node scripts/psim.mjs dump` prints for the same file, so the two apps can be
// compared with diff. Numbers are printed with %.9g.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/psim.hpp"
#include "psim_sample.hpp"

using namespace ps;
using namespace ps::psim;
using json::Value;

namespace {

Bytes loadFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw Error("cannot open " + path, "invalid");
    return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void saveFile(const std::string& path, const Bytes& b) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(b.data()), std::streamsize(b.size()));
    if (!out) throw Error("cannot write " + path, "invalid");
}

std::string kb(double n) {
    char b[40];
    if (n >= 1e6) std::snprintf(b, sizeof b, "%.2f MB", n / 1e6);
    else std::snprintf(b, sizeof b, "%.1f KB", n / 1e3);
    return b;
}

std::string pad(std::string s, size_t n) { while (s.size() < n) s += ' '; return s; }

// ---- %.9g, the same text as scripts/psim.mjs fmtG

std::string g9(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
    char b[40];
    std::snprintf(b, sizeof b, "%.9g", v);
    return b;
}

std::string quote(const std::string& s) {
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += char(c);
    }
    return o + "\"";
}

std::string hex8(uint32_t v) { char b[16]; std::snprintf(b, sizeof b, "%08x", v); return b; }

struct Fnv {
    uint32_t h = 0x811c9dc5u;
    void add(uint32_t w) { h ^= w; h *= 16777619u; }
};

uint32_t floatBits(float f) {
    if (f != f) return 0x7fc00000u;  // one not-a-number, whatever its payload
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

void flatten(const std::string& prefix, const Value& v) {
    switch (v.type) {
        case Value::Null: std::printf("%s = null\n", prefix.c_str()); break;
        case Value::Bool: std::printf("%s = %s\n", prefix.c_str(), v.b ? "true" : "false"); break;
        case Value::Number: std::printf("%s = %s\n", prefix.c_str(), g9(v.num).c_str()); break;
        case Value::String: std::printf("%s = %s\n", prefix.c_str(), quote(v.str).c_str()); break;
        case Value::Array:
            if (v.arr.empty()) std::printf("%s = []\n", prefix.c_str());
            for (size_t i = 0; i < v.arr.size(); i++) flatten(prefix + "[" + std::to_string(i) + "]", v.arr[i]);
            break;
        case Value::Object:
            if (v.obj.empty()) std::printf("%s = {}\n", prefix.c_str());
            for (const auto& [k, x] : v.obj) flatten(prefix + "." + k, x);
            break;
    }
}

struct Stats {
    double mn = 0, mx = 0, sum = 0;
    size_t nonfinite = 0;
    void add(double v, bool& first) {
        if (!std::isfinite(v)) { nonfinite++; return; }
        if (first) { mn = mx = v; first = false; }
        if (v < mn) mn = v;
        if (v > mx) mx = v;
        sum += v;
    }
};

void dumpArrays(const char* tag, const Arrays& r) {
    flatten(std::string(tag) + ".meta", r.meta);
    for (const Array& a : r.list) {
        Stats s;
        bool first = true;
        Fnv h;
        if (a.enc == Enc::I32) for (int32_t v : a.i) { s.add(v, first); h.add(uint32_t(v)); }
        else if (a.enc == Enc::U8) for (uint8_t v : a.u) { s.add(v, first); h.add(v); }
        else for (float v : a.f) { s.add(v, first); h.add(floatBits(v)); }
        std::string dims = "-";
        if (!a.dims.empty()) dims = std::to_string(a.dims[0]) + "," + std::to_string(a.dims[1]) + "," + std::to_string(a.dims[2]);
        std::printf("%s.array %s enc=%s n=%zu dims=%s min=%s max=%s sum=%s nonfinite=%zu fnv=%s\n", tag, a.name.c_str(), encName(a.enc), a.size(), dims.c_str(),
                    g9(s.mn).c_str(), g9(s.mx).c_str(), g9(s.sum).c_str(), s.nonfinite, hex8(h.h).c_str());
    }
}

int cmdDump(const std::string& path) {
    const Bytes bytes = loadFile(path);
    const File f = readPsim(bytes);
    std::printf("psim-dump 1\n");
    std::printf("format %d %d\n", f.version, f.minReader);
    for (const auto& e : f.table) {
        std::printf("table %s codec=%u flags=%u raw=%u crc=%s\n", e.id.c_str(), e.codec, e.flags, e.raw, hex8(e.crc).c_str());
    }
    if (!f.skipped.empty()) {
        std::string s;
        for (size_t i = 0; i < f.skipped.size(); i++) s += (i ? "," : "") + f.skipped[i];
        std::printf("skipped %s\n", s.c_str());
    }
    flatten("info", f.info);
    if (f.thumb) {
        Fnv h;
        for (uint8_t b : *f.thumb) h.add(b);
        std::printf("thumb bytes=%zu fnv=%s\n", f.thumb->size(), hex8(h.h).c_str());
    }
    if (f.geometry) {
        const Geometry& g = *f.geometry;
        const size_t nV = g.vertices.size() / 3, nT = g.tris.size() / 3;
        std::printf("geom.mode = %s\n", g.exact ? "exact" : "quantised16");
        std::printf("geom.brep = %d\n", g.brepFaces ? 1 : 0);
        std::printf("geom.vertices = %zu\n", nV);
        std::printf("geom.triangles = %zu\n", nT);
        std::printf("geom.faceCount = %u\n", g.faceCount);
        std::printf("geom.faceAngle = %s\n", g9(g.faceAngle).c_str());
        std::printf("geom.bbox.min = %s %s %s\n", g9(g.bboxMin[0]).c_str(), g9(g.bboxMin[1]).c_str(), g9(g.bboxMin[2]).c_str());
        std::printf("geom.bbox.max = %s %s %s\n", g9(g.bboxMax[0]).c_str(), g9(g.bboxMax[1]).c_str(), g9(g.bboxMax[2]).c_str());
        const char* axes = "xyz";
        for (int d = 0; d < 3; d++) {
            Stats s;
            bool first = true;
            for (size_t i = 0; i < nV; i++) s.add(g.vertices[3 * i + d], first);
            std::printf("geom.pos.%c = min=%s max=%s sum=%s\n", axes[d], g9(s.mn).c_str(), g9(s.mx).c_str(), g9(s.sum).c_str());
        }
        Fnv hv, ht, hf;
        for (float v : g.vertices) hv.add(floatBits(v));
        for (uint32_t v : g.tris) ht.add(v);
        std::printf("geom.vertexFnv = %s\n", hex8(hv.h).c_str());
        std::printf("geom.triangleFnv = %s\n", hex8(ht.h).c_str());
        if (g.brepFaces) {
            for (int32_t v : g.faceOf) hf.add(uint32_t(v));
            std::printf("geom.faceFnv = %s\n", hex8(hf.h).c_str());
        }
    }
    if (f.cad) {
        Fnv h;
        for (uint8_t b : *f.cad) h.add(b);
        std::printf("cad bytes=%zu fnv=%s\n", f.cad->size(), hex8(h.h).c_str());
    }
    if (f.setup) flatten("setup", *f.setup);
    if (f.rfea) dumpArrays("rfea", *f.rfea);
    if (f.rair) dumpArrays("rair", *f.rair);
    if (f.view) flatten("view", *f.view);
    return 0;
}

// ---- indented JSON like JSON.stringify(v, null, 2)

void pretty(std::string& o, const Value& v, int level, const KeyOrder* order, const std::string& parent) {
    const std::string pad1(size_t(2 * (level + 1)), ' '), pad0(size_t(2 * level), ' ');
    switch (v.type) {
        case Value::Null: o += "null"; break;
        case Value::Bool: o += v.b ? "true" : "false"; break;
        case Value::Number: o += std::isfinite(v.num) ? stringifyJson(v) : "null"; break;
        case Value::String: o += stringifyJson(v); break;
        case Value::Array:
            if (v.arr.empty()) { o += "[]"; break; }
            o += "[\n";
            for (size_t i = 0; i < v.arr.size(); i++) {
                o += pad1;
                pretty(o, v.arr[i], level + 1, order, parent);
                o += i + 1 < v.arr.size() ? ",\n" : "\n";
            }
            o += pad0 + "]";
            break;
        case Value::Object: {
            if (v.obj.empty()) { o += "{}"; break; }
            std::vector<std::string> keys;
            if (order) {
                auto it = order->find(parent);
                if (it != order->end()) for (const auto& k : it->second) if (v.obj.count(k)) keys.push_back(k);
            }
            for (const auto& [k, x] : v.obj) { bool seen = false; for (const auto& s : keys) seen |= s == k; if (!seen) keys.push_back(k); }
            o += "{\n";
            for (size_t i = 0; i < keys.size(); i++) {
                o += pad1 + stringifyJson(jstr(keys[i])) + ": ";
                pretty(o, v.obj.at(keys[i]), level + 1, order, keys[i]);
                o += i + 1 < keys.size() ? ",\n" : "\n";
            }
            o += pad0 + "}";
            break;
        }
    }
}

int cmdInspect(const std::string& path) {
    const Bytes bytes = loadFile(path);
    const File i = inspectPsim(bytes);
    std::printf("%s: %s, format %d (reader needs %d)\n", path.c_str(), kb(double(bytes.size())).c_str(), i.version, i.minReader);
    std::printf("section  codec  stored      raw         ratio\n");
    for (const auto& e : i.table) {
        const std::string codec = e.codec == 0 ? "stored" : e.codec == 1 ? "deflate" : std::to_string(e.codec);
        char ratio[32];
        std::snprintf(ratio, sizeof ratio, "%.1f", double(e.raw) / std::max<double>(1, e.stored));
        std::printf("%s     %s  %s  %s  %s%s\n", e.id.c_str(), codec.c_str(), pad(kb(e.stored), 10).c_str(), pad(kb(e.raw), 10).c_str(), ratio, (e.flags & 1) ? "  lossy" : "");
    }
    std::string text;
    pretty(text, i.info, 0, &infoKeyOrder(), "");
    std::printf("%s\n", text.c_str());
    if (i.thumb) std::printf("thumbnail: %zu bytes\n", i.thumb->size());
    return 0;
}

int cmdVerify(const std::string& path) {
    const Bytes bytes = loadFile(path);
    const File f = readPsim(bytes);
    std::string parts;
    auto add = [&](const std::string& s) { parts += (parts.empty() ? "" : ", ") + s; };
    if (f.geometry) add("geometry " + std::to_string(f.geometry->vertices.size() / 3) + " vertices, " + std::to_string(f.geometry->tris.size() / 3) + " triangles");
    if (f.setup) add("setup");
    if (f.rfea) add("structural/thermal results (" + std::to_string(f.rfea->list.size()) + " arrays)");
    if (f.rair) add("airflow results (" + std::to_string(f.rair->list.size()) + " arrays)");
    std::string skipped;
    if (!f.skipped.empty()) {
        skipped = "; skipped unknown sections ";
        for (size_t i = 0; i < f.skipped.size(); i++) skipped += (i ? ", " : "") + f.skipped[i];
    }
    std::printf("%s: OK, %s%s\n", path.c_str(), parts.empty() ? "info only" : parts.c_str(), skipped.c_str());
    return 0;
}

int cmdMakeFixture(const std::string& out, bool brep) {
    WriteOptions opt;
    opt.created = "2026-10-04T00:00:00Z";
    const Bytes bytes = writePsim(sampleContent(brep), opt);
    saveFile(out, bytes);
    std::printf("%s: %zu bytes\n", out.c_str(), bytes.size());
    return 0;
}

double nowMs() { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

int cmdBench(const std::string& path, const std::string& outPath) {
    const Bytes bytes = loadFile(path);
    File f;
    double bestRead = 1e30;
    for (int k = 0; k < 3; k++) {
        const double t0 = nowMs();
        f = readPsim(bytes);
        bestRead = std::min(bestRead, nowMs() - t0);
    }
    f.thumb.reset();
    double bestWrite = 1e30;
    Bytes out;
    WriteOptions opt;
    opt.created = "2026-10-04T00:00:00Z";
    opt.quantised = f.geometry ? !f.geometry->exact : true;
    for (int k = 0; k < 3; k++) {
        const double t0 = nowMs();
        out = writePsim(f, opt);
        bestWrite = std::min(bestWrite, nowMs() - t0);
    }
    std::printf("%s: file %zu bytes; read %.1f ms; write %.1f ms -> %zu bytes\n", path.c_str(), bytes.size(), bestRead, bestWrite, out.size());
    if (!outPath.empty()) saveFile(outPath, out);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string cmd = argc > 1 ? argv[1] : "";
    const std::string file = argc > 2 ? argv[2] : "";
    try {
        if (cmd == "verify" && argc > 2) return cmdVerify(file);
        if (cmd == "inspect" && argc > 2) return cmdInspect(file);
        if (cmd == "dump" && argc > 2) return cmdDump(file);
        if (cmd == "make-fixture" && argc > 2) return cmdMakeFixture(file, argc > 3 && std::string(argv[3]) == "--brep");
        if (cmd == "bench" && argc > 2) return cmdBench(file, argc > 3 ? argv[3] : "");
    } catch (const Error& e) {
        std::fprintf(stderr, "%s: %s\n", file.c_str(), e.what());
        return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s: %s\n", file.c_str(), e.what());
        return 1;
    }
    std::fprintf(stderr, "usage: psim_tool verify|inspect|dump <file>\n       psim_tool bench <file> [out.psim]\n       psim_tool make-fixture <out.psim> [--brep]\n");
    return cmd.empty() ? 0 : 2;
}
