// The .psim file (see psim.hpp and docs/psim-format.md). Mirrors src/core/psim.js.
//
// The decoders must give the same floats as the JavaScript ones, so no fused multiply-add here:
// every expression is evaluated in double precision exactly as written.
#include "core/psim.hpp"

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

namespace ps::psim {

const char* const kSectionOrder[8] = {"INFO", "THMB", "GEOM", "CADS", "SETP", "RFEA", "RAIR", "VIEW"};

namespace {

constexpr size_t kHeader = 32, kEntry = 24, kTrailer = 8;
constexpr uint16_t kCodecStored = 0, kCodecDeflate = 1;
const uint8_t kMagic[9] = {0x89, 0x50, 0x53, 0x49, 0x4d, 0x0d, 0x0a, 0x1a, 0x0a};

[[noreturn]] void fail(const std::string& message, const char* code = "invalid") { throw Error(message, code); }

uint16_t rd16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
uint32_t rd32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
void wr16(uint8_t* p, uint32_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
void wr32(uint8_t* p, uint32_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24); }
void wr64f(uint8_t* p, double d) { uint64_t u; std::memcpy(&u, &d, 8); for (int k = 0; k < 8; k++) p[k] = uint8_t(u >> (8 * k)); }
double rd64f(const uint8_t* p) { uint64_t u = 0; for (int k = 0; k < 8; k++) u |= uint64_t(p[k]) << (8 * k); double d; std::memcpy(&d, &u, 8); return d; }
void wr32f(uint8_t* p, float f) { uint32_t u; std::memcpy(&u, &f, 4); wr32(p, u); }
float rd32f(const uint8_t* p) { uint32_t u = rd32(p); float f; std::memcpy(&f, &u, 4); return f; }

uint32_t zigzag(int32_t v) { return (uint32_t(v) << 1) ^ uint32_t(v >> 31); }
int32_t unzigzag(uint32_t u) { return int32_t((u >> 1) ^ (0u - (u & 1))); }

/** Writes `values` (`width` bytes each) as byte planes at out; returns the new pointer. */
uint8_t* putPlanes(uint8_t* out, const uint32_t* values, size_t n, int width) {
    for (int b = 0; b < width; b++) {
        const int shift = 8 * b;
        for (size_t i = 0; i < n; i++) out[i] = uint8_t(values[i] >> shift);
        out += n;
    }
    return out;
}

/** Reads `n` values of `width` bytes from byte planes. */
std::vector<uint32_t> getPlanes(const uint8_t* in, size_t n, int width) {
    std::vector<uint32_t> out(n, 0);
    for (int b = 0; b < width; b++) {
        const int shift = 8 * b;
        for (size_t i = 0; i < n; i++) out[i] |= uint32_t(in[i]) << shift;
        in += n;
    }
    return out;
}

bool truthy(const json::Value& v) {
    switch (v.type) {
        case json::Value::Null: return false;
        case json::Value::Bool: return v.b;
        case json::Value::Number: return v.num != 0 && !std::isnan(v.num);
        case json::Value::String: return !v.str.empty();
        default: return true;
    }
}

bool isInteger(const json::Value& v) {
    return v.type == json::Value::Number && std::isfinite(v.num) && std::floor(v.num) == v.num;
}

// ---------------------------------------------------------------- UTF-8

/** Decodes one code point; returns the length or 0 when the bytes are not well-formed UTF-8. */
int utf8At(const unsigned char* s, size_t n, uint32_t* cp) {
    const unsigned c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    int len;
    uint32_t v, min;
    if (c >= 0xc2 && c <= 0xdf) { len = 2; v = c & 0x1f; min = 0x80; }
    else if (c >= 0xe0 && c <= 0xef) { len = 3; v = c & 0x0f; min = 0x800; }
    else if (c >= 0xf0 && c <= 0xf4) { len = 4; v = c & 0x07; min = 0x10000; }
    else return 0;
    if (size_t(len) > n) return 0;
    for (int k = 1; k < len; k++) {
        if ((s[k] & 0xc0) != 0x80) return 0;
        v = (v << 6) | (s[k] & 0x3f);
    }
    if (v < min || v > 0x10ffff || (v >= 0xd800 && v < 0xe000)) return 0;
    *cp = v;
    return len;
}

bool utf8Valid(const uint8_t* p, size_t n) {
    size_t i = 0;
    while (i < n) {
        uint32_t cp;
        const int l = utf8At(p + i, n - i, &cp);
        if (!l) return false;
        i += size_t(l);
    }
    return true;
}

/** Replaces ill-formed sequences by U+FFFD so that what is written can be read back. */
std::string utf8Repair(const std::string& s) {
    if (utf8Valid(reinterpret_cast<const uint8_t*>(s.data()), s.size())) return s;
    std::string out;
    size_t i = 0;
    while (i < s.size()) {
        uint32_t cp;
        const int l = utf8At(reinterpret_cast<const unsigned char*>(s.data()) + i, s.size() - i, &cp);
        if (!l) { out += "\xef\xbf\xbd"; i++; }
        else { out.append(s, i, size_t(l)); i += size_t(l); }
    }
    return out;
}

// ---------------------------------------------------------------- JSON writing

/** The shortest text that reads back as `v`, in the layout JavaScript uses (so both writers give the same bytes). */
std::string numberText(double v) {
    if (v == 0) return "0";
    char buf[48];
    int p = 1;
    for (; p <= 17; p++) {
        std::snprintf(buf, sizeof buf, "%.*e", p - 1, std::fabs(v));
        if (std::strtod(buf, nullptr) == std::fabs(v)) break;
    }
    // buf is d.ddddde[+-]xx: take the digits and the decimal exponent
    std::string digits;
    const char* e = std::strchr(buf, 'e');
    for (const char* q = buf; q < e; q++) if (*q != '.') digits += *q;
    while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
    const int k = int(digits.size());
    const int n = std::atoi(e + 1) + 1;  // the decimal point sits after n digits
    std::string out = v < 0 ? "-" : "";
    if (k <= n && n <= 21) out += digits + std::string(size_t(n - k), '0');
    else if (0 < n && n <= 21) out += digits.substr(0, size_t(n)) + "." + digits.substr(size_t(n));
    else if (-6 < n && n <= 0) out += "0." + std::string(size_t(-n), '0') + digits;
    else {
        out += digits.substr(0, 1);
        if (k > 1) out += "." + digits.substr(1);
        out += std::string("e") + (n - 1 < 0 ? "-" : "+") + std::to_string(std::abs(n - 1));
    }
    return out;
}

void writeString(std::string& o, const std::string& raw) {
    const std::string s = utf8Repair(raw);
    o += '"';
    for (const char ch : s) {
        const unsigned char c = static_cast<unsigned char>(ch);
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c == '\b') o += "\\b";
        else if (c == '\f') o += "\\f";
        else if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += ch;
    }
    o += '"';
}

void writeValue(std::string& o, const json::Value& v, const KeyOrder* order, const std::string& parent) {
    switch (v.type) {
        case json::Value::Null: o += "null"; break;
        case json::Value::Bool: o += v.b ? "true" : "false"; break;
        case json::Value::Number:
            if (std::isfinite(v.num)) o += numberText(v.num);
            else o += std::isnan(v.num) ? "{\"$num\":\"nan\"}" : v.num > 0 ? "{\"$num\":\"inf\"}" : "{\"$num\":\"-inf\"}";
            break;
        case json::Value::String: writeString(o, v.str); break;
        case json::Value::Array: {
            o += '[';
            for (size_t k = 0; k < v.arr.size(); k++) {
                if (k) o += ',';
                writeValue(o, v.arr[k], order, parent);
            }
            o += ']';
            break;
        }
        case json::Value::Object: {
            o += '{';
            bool first = true;
            auto put = [&](const std::string& key, const json::Value& val) {
                if (!first) o += ',';
                first = false;
                writeString(o, key);
                o += ':';
                writeValue(o, val, order, key);
            };
            const std::vector<std::string>* pref = nullptr;
            if (order) {
                auto it = order->find(parent);
                if (it != order->end()) pref = &it->second;
            }
            if (pref) for (const auto& key : *pref) { auto it = v.obj.find(key); if (it != v.obj.end()) put(key, it->second); }
            for (const auto& [key, val] : v.obj) {
                if (pref && std::find(pref->begin(), pref->end(), key) != pref->end()) continue;
                put(key, val);
            }
            o += '}';
            break;
        }
    }
}

json::Value num(double d) { json::Value v; v.type = json::Value::Number; v.num = d; return v; }
json::Value str(const std::string& s) { json::Value v; v.type = json::Value::String; v.str = s; return v; }
json::Value obj() { json::Value v; v.type = json::Value::Object; return v; }
json::Value arr() { json::Value v; v.type = json::Value::Array; return v; }

const KeyOrder& infoOrder() {
    static const KeyOrder o = {
        // as the JavaScript writer lays it out: format, the caller's keys, then created and bounds
        {"", {"format", "app", "name", "notes", "units", "contains", "part", "created", "bounds"}},
        {"app", {"name", "version", "kind"}},
        {"contains", {"geometry", "setup", "results", "cad", "cadName"}},
        {"part", {"vertices", "triangles", "bbox"}},
        {"bbox", {"min", "max"}},
        {"bounds", {"field", "absolute", "unit"}},
    };
    return o;
}

}  // namespace
const KeyOrder& infoKeyOrder() { return infoOrder(); }
namespace {

const KeyOrder& setupOrder() {
    static const KeyOrder o = {
        {"", {"units", "material", "structural", "thermal", "airflow"}},
        {"material", {"id", "name", "E", "nu", "yield", "uts", "density", "k", "cp", "alpha", "fatigue", "cost"}},
        {"structural", {"study", "resolution", "gravity", "fixtures", "loads", "options"}},
        {"fixtures", {"name", "patches"}},
        {"loads", {"name", "type", "magnitude", "dir", "patches"}},
        {"patches", {"tris", "clip", "label"}},
        {"thermal", {"mode", "items", "ambient", "duration", "steps", "initial", "resolution"}},
        {"items", {"name", "type", "value", "ambient", "patches"}},
        {"airflow", {"yaw", "pitch", "speed", "density", "cells", "ground", "boundaryLayer", "engine"}},
    };
    return o;
}

const KeyOrder& viewOrder() {
    static const KeyOrder o = {
        {"", {"tab", "study", "camera", "plot", "scalePct", "level", "step", "mode", "flowDisplay"}},
        {"camera", {"position", "target"}},
    };
    return o;
}

const KeyOrder& arraysHeaderOrder() {
    static const KeyOrder o = {
        {"", {"meta", "arrays"}},
        {"arrays", {"name", "enc", "n", "dims", "stride", "min", "max", "err", "bytes"}},
    };
    return o;
}

Bytes textBytes(const std::string& s) { return Bytes(s.begin(), s.end()); }

Bytes jsonBytes(const json::Value& v, const KeyOrder* order) { return textBytes(stringifyJson(v, order)); }

void convertNums(json::Value& v) {
    if (v.type == json::Value::Array) { for (auto& x : v.arr) convertNums(x); return; }
    if (v.type != json::Value::Object) return;
    for (auto& [k, x] : v.obj) convertNums(x);
    auto it = v.obj.find("$num");
    if (it != v.obj.end() && it->second.type == json::Value::String) {
        const std::string& s = it->second.str;
        double d;
        if (s == "nan") d = std::numeric_limits<double>::quiet_NaN();
        else if (s == "inf") d = std::numeric_limits<double>::infinity();
        else if (s == "-inf") d = -std::numeric_limits<double>::infinity();
        else return;
        v = num(d);
    }
}

json::Value jsonOf(const uint8_t* p, size_t n) {
    if (!utf8Valid(p, n)) fail("A section holds text that is not valid UTF-8.", "damaged");
    if (n >= 3 && p[0] == 0xef && p[1] == 0xbb && p[2] == 0xbf) { p += 3; n -= 3; }  // a decoder drops the byte order mark
    return parseJson(std::string(reinterpret_cast<const char*>(p), n));
}
json::Value jsonOf(const Bytes& b) { return jsonOf(b.data(), b.size()); }

}  // namespace

// ---------------------------------------------------------------- primitives

uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc) {
    uLong c = crc;
    while (n) {
        const uInt chunk = uInt(std::min<size_t>(n, 1u << 30));
        c = ::crc32(c, p, chunk);
        p += chunk;
        n -= chunk;
    }
    return uint32_t(c);
}

Bytes deflateRaw(const uint8_t* p, size_t n) {
    if (n > 0x7fffffffu) fail("A section is too large to compress.", "limit");
    z_stream zs{};
    if (deflateInit2(&zs, 6, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK) fail("The compressor could not start.", "invalid");
    Bytes out(deflateBound(&zs, uLong(n)) + 16);
    zs.next_in = const_cast<Bytef*>(p);
    zs.avail_in = uInt(n);
    zs.next_out = out.data();
    zs.avail_out = uInt(out.size());
    const int rc = deflate(&zs, Z_FINISH);
    const size_t produced = out.size() - zs.avail_out;
    deflateEnd(&zs);
    if (rc != Z_STREAM_END) fail("A section could not be compressed.", "invalid");
    out.resize(produced);
    return out;
}

Bytes inflateRaw(const uint8_t* p, size_t stored, size_t rawLength) {
    z_stream zs{};
    if (inflateInit2(&zs, -15) != Z_OK) fail("A section could not be decompressed; the file is damaged.", "damaged");
    struct Guard { z_stream* z; ~Guard() { inflateEnd(z); } } guard{&zs};
    zs.next_in = const_cast<Bytef*>(p);
    zs.avail_in = uInt(stored);
    Bytes out;
    size_t produced = 0;
    for (;;) {
        if (produced > rawLength) fail("A section is larger than it says (possible decompression bomb).", "bomb");
        if (produced == out.size()) {
            // grow in steps, so a lying header cannot make us reserve its claim before the data shows up
            const size_t want = std::min<size_t>(rawLength + 1, std::max<size_t>(size_t(64) << 10, out.size() * 2));
            out.resize(want);
        }
        const size_t room = out.size() - produced;
        zs.next_out = out.data() + produced;
        zs.avail_out = uInt(std::min<size_t>(room, 1u << 30));
        const uInt before = zs.avail_out;
        const int rc = inflate(&zs, Z_NO_FLUSH);
        produced += before - zs.avail_out;
        if (rc == Z_STREAM_END) break;
        if (rc != Z_OK) fail("A section could not be decompressed; the file is damaged.", "damaged");
        if (zs.avail_in == 0 && zs.avail_out > 0) fail("A section could not be decompressed; the file is damaged.", "damaged");
    }
    if (produced > rawLength) fail("A section is larger than it says (possible decompression bomb).", "bomb");
    if (zs.avail_in != 0) fail("A section could not be decompressed; the file is damaged.", "damaged");
    if (produced != rawLength) fail("A section is shorter than it says; the file is damaged.", "damaged");
    out.resize(rawLength);
    return out;
}

std::string stringifyJson(const json::Value& v, const KeyOrder* order) {
    std::string o;
    writeValue(o, v, order, "");
    return o;
}

json::Value parseJson(const std::string& text) {
    json::ParseOptions opt;
    opt.strict = true;
    opt.maxNodes = Limits::jsonNodes;
    json::Value v;
    try {
        v = json::parse(text, opt);
    } catch (const std::runtime_error&) {
        fail("A section holds text that is not valid JSON; the file is damaged.", "damaged");
    }
    convertNums(v);
    return v;
}

// ---------------------------------------------------------------- arrays container

namespace {

constexpr int levels(Enc e) { return e == Enc::Q16 ? 65536 : 256; }
constexpr int widthOf(Enc e) { return e == Enc::Q16 ? 2 : e == Enc::Q8 ? 1 : e == Enc::U8 ? 1 : 4; }

bool encFromName(const std::string& s, Enc* e) {
    static const struct { const char* n; Enc e; } all[] = {{"f32", Enc::F32}, {"i32", Enc::I32}, {"u8", Enc::U8}, {"q16", Enc::Q16}, {"q8", Enc::Q8}};
    for (const auto& a : all) if (s == a.n) { *e = a.e; return true; }
    return false;
}

void checkDims(const std::vector<uint32_t>& d, size_t n) {
    if (d.size() != 3 || d[0] < 1 || d[1] < 1 || d[2] < 1 || uint64_t(d[0]) * d[1] * d[2] != n) {
        fail("An array has grid dimensions that do not match its length.", "invalid");
    }
}

/** 3-D Lorenzo prediction at cell i (x fastest); `codes` holds the cells before i. */
inline int predictAt(const uint16_t* codes, size_t i, size_t x, size_t y, size_t z, size_t sy, size_t sz) {
    const int a = x > 0 ? codes[i - 1] : 0;
    const int b = y > 0 ? codes[i - sy] : 0;
    const int c = z > 0 ? codes[i - sz] : 0;
    const int ab = x > 0 && y > 0 ? codes[i - 1 - sy] : 0;
    const int ac = x > 0 && z > 0 ? codes[i - 1 - sz] : 0;
    const int bc = y > 0 && z > 0 ? codes[i - sy - sz] : 0;
    const int abc = x > 0 && y > 0 && z > 0 ? codes[i - 1 - sy - sz] : 0;
    return a + b + c - ab - ac - bc + abc;
}

std::vector<uint32_t> predictEncode(const std::vector<uint16_t>& codes, int L, const std::vector<uint32_t>& dims, size_t stride) {
    const size_t n = codes.size();
    std::vector<uint32_t> res(n);
    const int half = L >> 1;
    auto put = [&](size_t i, int p) {
        int r = (int(codes[i]) - p) % L;
        if (r < 0) r += L;
        if (r >= half) r -= L;
        res[i] = uint32_t(r >= 0 ? 2 * r : -2 * r - 1);
    };
    if (dims.empty()) {
        for (size_t i = 0; i < n; i++) put(i, i >= stride ? codes[i - stride] : 0);
        return res;
    }
    const size_t nx = dims[0], ny = dims[1], nz = dims[2], sy = nx, sz = nx * ny;
    for (size_t z = 0; z < nz; z++)
        for (size_t y = 0; y < ny; y++)
            for (size_t x = 0; x < nx; x++) {
                const size_t i = x + sy * y + sz * z;
                put(i, predictAt(codes.data(), i, x, y, z, sy, sz));
            }
    return res;
}

std::vector<uint16_t> predictDecode(const std::vector<uint32_t>& res, int L, const std::vector<uint32_t>& dims, size_t stride) {
    const size_t n = res.size();
    std::vector<uint16_t> codes(n);
    auto get = [&](size_t i, int p) {
        const uint32_t q = res[i];
        const int r = (q & 1) ? -int((q + 1) >> 1) : int(q >> 1);
        int c = (p + r) % L;
        if (c < 0) c += L;
        codes[i] = uint16_t(c);
    };
    if (dims.empty()) {
        for (size_t i = 0; i < n; i++) get(i, i >= stride ? codes[i - stride] : 0);
        return codes;
    }
    const size_t nx = dims[0], ny = dims[1], nz = dims[2], sy = nx, sz = nx * ny;
    for (size_t z = 0; z < nz; z++)
        for (size_t y = 0; y < ny; y++)
            for (size_t x = 0; x < nx; x++) {
                const size_t i = x + sy * y + sz * z;
                get(i, predictAt(codes.data(), i, x, y, z, sy, sz));
            }
    return codes;
}

}  // namespace

const char* encName(Enc e) {
    switch (e) {
        case Enc::F32: return "f32";
        case Enc::I32: return "i32";
        case Enc::U8: return "u8";
        case Enc::Q16: return "q16";
        case Enc::Q8: return "q8";
    }
    return "f32";
}

const Array* Arrays::find(const std::string& name) const {
    for (const auto& a : list) if (a.name == name) return &a;
    return nullptr;
}

Bytes encodeArrays(const json::Value& meta, const std::vector<Array>& arrays, std::vector<Bound>* bounds) {
    json::Value entries = arr();
    std::vector<Bytes> blobs;
    for (const Array& a : arrays) {
        const size_t n = a.size();
        json::Value entry = obj();
        entry.obj["name"] = str(a.name);
        entry.obj["enc"] = str(encName(a.enc));
        entry.obj["n"] = num(double(n));
        if (!a.dims.empty()) {
            checkDims(a.dims, n);
            json::Value d = arr();
            for (uint32_t x : a.dims) d.arr.push_back(num(x));
            entry.obj["dims"] = d;
        }
        if (a.stride != 1) {
            if (!(a.enc == Enc::Q16 || a.enc == Enc::Q8) || !a.dims.empty() || a.stride < 2 || a.stride > 4 || n % a.stride) fail("An array has a stride its encoding cannot use.", "invalid");
            entry.obj["stride"] = num(a.stride);
        }
        Bytes blob(size_t(widthOf(a.enc)) * n);
        if (a.enc == Enc::F32) {
            std::vector<uint32_t> bits(n);
            if (n) std::memcpy(bits.data(), a.f.data(), 4 * n);
            putPlanes(blob.data(), bits.data(), n, 4);
        } else if (a.enc == Enc::I32) {
            std::vector<uint32_t> z(n);
            for (size_t i = 0; i < n; i++) z[i] = zigzag(a.i[i]);
            putPlanes(blob.data(), z.data(), n, 4);
        } else if (a.enc == Enc::U8) {
            if (n) std::memcpy(blob.data(), a.u.data(), n);
        } else {
            const int L = levels(a.enc);
            double mn = std::numeric_limits<double>::infinity(), mx = -mn;
            for (size_t i = 0; i < n; i++) {
                const double v = a.f[i];
                if (std::isfinite(v)) { if (v < mn) mn = v; if (v > mx) mx = v; }
            }
            if (mn > mx) { mn = 0; mx = 0; }
            const double span = mx - mn;
            std::vector<uint16_t> codes(n);
            for (size_t i = 0; i < n; i++) {
                const double v = a.f[i];
                codes[i] = std::isfinite(v) ? uint16_t(span > 0 ? 1 + std::round(((v - mn) / span) * (L - 2)) : 1) : uint16_t(0);
            }
            const std::vector<uint32_t> res = predictEncode(codes, L, a.dims, a.stride);
            putPlanes(blob.data(), res.data(), n, widthOf(a.enc));
            const double err = span / (2.0 * (L - 2));
            entry.obj["min"] = num(mn);
            entry.obj["max"] = num(mx);
            entry.obj["err"] = num(err);
            if (bounds) bounds->push_back({a.name, err});
        }
        entry.obj["bytes"] = num(double(blob.size()));
        entries.arr.push_back(std::move(entry));
        blobs.push_back(std::move(blob));
    }
    json::Value head = obj();
    head.obj["meta"] = meta;
    head.obj["arrays"] = std::move(entries);
    const Bytes hb = jsonBytes(head, &arraysHeaderOrder());
    size_t total = 4 + hb.size();
    for (const auto& b : blobs) total += b.size();
    Bytes out(total);
    wr32(out.data(), uint32_t(hb.size()));
    std::memcpy(out.data() + 4, hb.data(), hb.size());
    size_t o = 4 + hb.size();
    for (const auto& b : blobs) { if (!b.empty()) std::memcpy(out.data() + o, b.data(), b.size()); o += b.size(); }
    return out;
}

Arrays decodeArrays(const uint8_t* bytes, size_t length) {
    if (length < 4) fail("A result section is too short.", "damaged");
    const uint32_t hl = rd32(bytes);
    if (hl > length - 4 || hl > Limits::jsonSection) fail("A result section has a header longer than the section.", "damaged");
    const json::Value head = jsonOf(bytes + 4, hl);
    if (head.type != json::Value::Object || !head.has("arrays") || head["arrays"].type != json::Value::Array) fail("A result section has no array list.", "damaged");
    const auto& list = head["arrays"].arr;
    if (list.size() > Limits::arrays) fail("A result section lists too many arrays.", "limit");
    Arrays out;
    size_t o = 4 + size_t(hl);
    double elements = 0;
    for (const json::Value& e : list) {
        Enc enc = Enc::F32;
        if (e.type != json::Value::Object || e["name"].type != json::Value::String || e["enc"].type != json::Value::String ||
            !encFromName(e["enc"].str, &enc) || !isInteger(e["n"]) || e["n"].num < 0 || !isInteger(e["bytes"])) {
            fail("An array entry is malformed.", "damaged");
        }
        const double nd = e["n"].num;
        elements += nd;
        if (elements > double(Limits::arrayElements)) fail("A result section holds too many numbers.", "limit");
        if (e["bytes"].num != double(widthOf(enc)) * nd) fail("An array entry has the wrong byte count.", "damaged");
        if (double(o) + e["bytes"].num > double(length)) fail("An array runs past the end of its section.", "damaged");
        const size_t n = size_t(nd);
        Array a;
        a.name = e["name"].str;
        a.enc = enc;
        const json::Value& dv = e["dims"];
        if (truthy(dv)) {
            bool ok = dv.type == json::Value::Array && dv.arr.size() == 3;
            double prod = 1;
            if (ok) for (const auto& d : dv.arr) { if (!isInteger(d) || d.num < 1) ok = false; else prod *= d.num; }
            if (!ok || prod != nd) fail("An array has grid dimensions that do not match its length.", "invalid");
            for (const auto& d : dv.arr) a.dims.push_back(uint32_t(d.num));
        }
        const json::Value& sv = e["stride"];
        double strideD = 1;
        if (sv.type != json::Value::Null) strideD = sv.type == json::Value::Number ? sv.num : std::nan("");
        if (!(std::floor(strideD) == strideD) || strideD < 1 || strideD > 4 || (strideD > 1 && (truthy(dv) || std::fmod(nd, strideD) != 0))) fail("An array entry has an invalid stride.", "damaged");
        a.stride = uint32_t(strideD);
        const uint8_t* src = bytes + o;
        if (enc == Enc::F32) {
            const auto bits = getPlanes(src, n, 4);
            a.f.resize(n);
            if (n) std::memcpy(a.f.data(), bits.data(), 4 * n);
        } else if (enc == Enc::I32) {
            const auto z = getPlanes(src, n, 4);
            a.i.resize(n);
            for (size_t i = 0; i < n; i++) a.i[i] = unzigzag(z[i]);
        } else if (enc == Enc::U8) {
            a.u.assign(src, src + n);
        } else {
            const int L = levels(enc);
            if (e["min"].type != json::Value::Number || !std::isfinite(e["min"].num) || e["max"].type != json::Value::Number || !std::isfinite(e["max"].num)) {
                fail("A quantised array has no valid range.", "damaged");
            }
            const double mn = e["min"].num, mx = e["max"].num;
            const auto res = getPlanes(src, n, widthOf(enc));
            const auto codes = predictDecode(res, L, a.dims, a.stride);
            a.f.resize(n);
            const double k = (mx - mn) / (L - 2);
            for (size_t i = 0; i < n; i++) a.f[i] = codes[i] == 0 ? std::numeric_limits<float>::quiet_NaN() : float(mn + (codes[i] - 1) * k);
        }
        if (out.find(a.name)) fail("The array " + a.name + " appears twice.", "damaged");
        a.err = e["err"].type == json::Value::Number ? e["err"].num : 0;
        o += size_t(e["bytes"].num);
        out.list.push_back(std::move(a));
    }
    if (o != length) fail("A result section has bytes the array list does not account for.", "damaged");
    out.meta = head.has("meta") && head["meta"].type != json::Value::Null ? head["meta"] : obj();
    return out;
}

// ---------------------------------------------------------------- geometry

Bytes encodeGeometry(const Geometry& g, bool quant, Bound* bound) {
    if (g.vertices.size() % 3 || g.tris.size() % 3) fail("The geometry arrays are not a multiple of three.", "invalid");
    const size_t nV = g.vertices.size() / 3, nT = g.tris.size() / 3;
    double mn[3] = {INFINITY, INFINITY, INFINITY}, mx[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (size_t i = 0; i < nV; i++)
        for (int d = 0; d < 3; d++) {
            const double v = g.vertices[3 * i + d];
            if (v < mn[d]) mn[d] = v;
            if (v > mx[d]) mx[d] = v;
        }
    if (nV == 0) for (int d = 0; d < 3; d++) mn[d] = mx[d] = 0;
    const bool brep = g.brepFaces && !g.faceOf.empty();
    if (brep && g.faceOf.size() != nT) fail("The face ids do not match the triangles.", "invalid");
    const size_t size = 68 + (quant ? 2 * 3 * nV : 4 * 3 * nV) + 4 * 3 * nT + (brep ? 4 * nT : 0);
    Bytes out(size, 0);
    uint8_t* b = out.data();
    b[0] = quant ? 1 : 0;
    b[1] = quant ? 16 : 0;
    b[2] = brep ? 1 : 0;
    wr32(b + 4, uint32_t(nV));
    wr32(b + 8, uint32_t(nT));
    wr32(b + 12, g.faceCount);
    wr32f(b + 16, brep ? 0.0f : float(g.faceAngle));
    for (int d = 0; d < 3; d++) { wr64f(b + 20 + 8 * d, mn[d]); wr64f(b + 44 + 8 * d, mx[d]); }
    uint8_t* o = b + 68;
    double worst = 0;
    std::vector<uint32_t> axis(nV);
    for (int d = 0; d < 3; d++) {
        if (quant) {
            const double span = mx[d] - mn[d];
            int prev = 0;
            for (size_t i = 0; i < nV; i++) {
                const int q = span > 0 ? int(std::round(((double(g.vertices[3 * i + d]) - mn[d]) / span) * 65535)) : 0;
                int r = (q - prev) & 0xffff;
                if (r >= 32768) r -= 65536;
                axis[i] = uint32_t(r >= 0 ? 2 * r : -2 * r - 1);
                prev = q;
            }
            o = putPlanes(o, axis.data(), nV, 2);
            worst = std::max(worst, span / 131070);
        } else {
            for (size_t i = 0; i < nV; i++) std::memcpy(&axis[i], &g.vertices[3 * i + d], 4);
            o = putPlanes(o, axis.data(), nV, 4);
        }
    }
    std::vector<uint32_t> slot(nT);
    for (int k = 0; k < 3; k++) {
        uint32_t prev = 0;
        for (size_t t = 0; t < nT; t++) {
            const uint32_t c = g.tris[3 * t + k];
            slot[t] = zigzag(int32_t(c - prev));
            prev = c;
        }
        o = putPlanes(o, slot.data(), nT, 4);
    }
    if (brep) {
        int32_t prev = 0;
        for (size_t t = 0; t < nT; t++) { slot[t] = zigzag(int32_t(uint32_t(g.faceOf[t]) - uint32_t(prev))); prev = g.faceOf[t]; }
        putPlanes(o, slot.data(), nT, 4);
    }
    if (bound && quant) *bound = {"geometry.position", worst};
    return out;
}

Geometry decodeGeometry(const uint8_t* bytes, size_t length) {
    if (length < 68) fail("The geometry section is too short.", "damaged");
    const int mode = bytes[0], bits = bytes[1];
    const bool brep = (bytes[2] & 1) == 1;
    if (mode > 1 || (mode == 1 && bits != 16) || (mode == 0 && bits != 0)) fail("The geometry uses a coding this version does not know.", "unsupported");
    const uint64_t nV = rd32(bytes + 4), nT = rd32(bytes + 8);
    Geometry g;
    g.faceCount = rd32(bytes + 12);
    if (nV > Limits::vertices || nT > Limits::triangles) fail("The part has more vertices or triangles than this reader accepts.", "limit");
    const uint64_t expect = 68 + (mode ? 2 * 3 * nV : 4 * 3 * nV) + 4 * 3 * nT + (brep ? 4 * nT : 0);
    if (expect != length) fail("The geometry section does not match its vertex and triangle counts.", "damaged");
    const float faceAngle = rd32f(bytes + 16);
    for (int d = 0; d < 3; d++) {
        g.bboxMin[d] = rd64f(bytes + 20 + 8 * d);
        g.bboxMax[d] = rd64f(bytes + 44 + 8 * d);
        if (!std::isfinite(g.bboxMin[d]) || !std::isfinite(g.bboxMax[d]) || g.bboxMax[d] < g.bboxMin[d]) fail("The geometry has an invalid bounding box.", "damaged");
    }
    g.vertices.assign(size_t(3 * nV), 0.0f);
    const uint8_t* o = bytes + 68;
    for (int d = 0; d < 3; d++) {
        if (mode) {
            const auto z = getPlanes(o, size_t(nV), 2);
            o += 2 * nV;
            const double span = g.bboxMax[d] - g.bboxMin[d];
            uint32_t prev = 0;
            for (size_t i = 0; i < nV; i++) {
                const int r = (z[i] & 1) ? -int((z[i] + 1) >> 1) : int(z[i] >> 1);
                prev = uint32_t(int(prev) + r) & 0xffff;
                g.vertices[3 * i + d] = float(g.bboxMin[d] + (double(prev) / 65535) * span);
            }
        } else {
            const auto w = getPlanes(o, size_t(nV), 4);
            o += 4 * nV;
            for (size_t i = 0; i < nV; i++) std::memcpy(&g.vertices[3 * i + d], &w[i], 4);
        }
    }
    g.tris.assign(size_t(3 * nT), 0u);
    for (int k = 0; k < 3; k++) {
        const auto z = getPlanes(o, size_t(nT), 4);
        o += 4 * nT;
        uint32_t prev = 0;
        for (size_t t = 0; t < nT; t++) {
            prev += uint32_t(unzigzag(z[t]));
            if (prev >= nV) fail("A triangle refers to a vertex that does not exist.", "damaged");
            g.tris[3 * t + k] = prev;
        }
    }
    for (size_t t = 0; t < nT; t++) {
        const uint32_t a = g.tris[3 * t], b = g.tris[3 * t + 1], c = g.tris[3 * t + 2];
        if (a == b || b == c || a == c) fail("A triangle uses one vertex twice.", "damaged");
    }
    if (brep) {
        g.faceOf.assign(size_t(nT), 0);
        const auto z = getPlanes(o, size_t(nT), 4);
        int32_t prev = 0;
        const int64_t limit = std::max<int64_t>(g.faceCount, 1);
        for (size_t t = 0; t < nT; t++) {
            prev = int32_t(uint32_t(prev) + uint32_t(unzigzag(z[t])));
            if (prev < 0 || prev >= limit) fail("A triangle belongs to a face that does not exist.", "damaged");
            g.faceOf[t] = prev;
        }
    }
    g.brepFaces = brep;
    g.faceAngle = brep ? 0 : faceAngle;
    g.exact = mode == 0;
    return g;
}

// ---------------------------------------------------------------- container

Bytes writePsim(const File& content, const WriteOptions& opt) {
    struct Part { std::string id; Bytes raw; bool lossy = false; uint16_t codec = kCodecDeflate; };
    std::vector<Part> parts;
    std::vector<Bound> bounds;
    if (content.thumb) parts.push_back({"THMB", *content.thumb, false, kCodecStored});
    if (content.geometry) {
        Bound b;
        Bytes raw = encodeGeometry(*content.geometry, opt.quantised, &b);
        if (opt.quantised) bounds.push_back(b);
        parts.push_back({"GEOM", std::move(raw), opt.quantised, kCodecDeflate});
    }
    if (content.cad) parts.push_back({"CADS", *content.cad, false, kCodecDeflate});
    if (content.setup) parts.push_back({"SETP", jsonBytes(*content.setup, &setupOrder()), false, kCodecDeflate});
    for (int k = 0; k < 2; k++) {
        const auto& r = k == 0 ? content.rfea : content.rair;
        if (!r) continue;
        std::vector<Bound> bb;
        Bytes raw = encodeArrays(r->meta, r->list, &bb);
        parts.push_back({k == 0 ? "RFEA" : "RAIR", std::move(raw), !bb.empty(), kCodecDeflate});
        bounds.insert(bounds.end(), bb.begin(), bb.end());
    }
    if (content.view) parts.push_back({"VIEW", jsonBytes(*content.view, &viewOrder()), false, kCodecDeflate});

    json::Value info = obj();
    info.obj["format"] = num(kVersion);
    if (content.info.type == json::Value::Object) for (const auto& [k, v] : content.info.obj) info.obj[k] = v;
    if (opt.created) info.obj["created"] = str(*opt.created);
    else if (!(content.info.has("created"))) info.obj["created"] = json::Value{};
    json::Value bl = arr();
    for (const Bound& b : bounds) { json::Value e = obj(); e.obj["field"] = str(b.field); e.obj["absolute"] = num(b.absolute); bl.arr.push_back(e); }
    info.obj["bounds"] = bl;
    parts.push_back({"INFO", jsonBytes(info, &infoOrder()), false, kCodecDeflate});

    // write in the documented order
    std::vector<const Part*> ordered;
    for (const char* id : kSectionOrder) for (const Part& p : parts) if (p.id == id) ordered.push_back(&p);
    std::vector<Bytes> stored;
    for (const Part* p : ordered) stored.push_back(p->codec == kCodecDeflate ? deflateRaw(p->raw.data(), p->raw.size()) : p->raw);
    const size_t n = ordered.size();
    uint64_t size = kHeader + kEntry * n + kTrailer;
    for (const auto& s : stored) size += s.size();
    if (size > 0xffffffffull) fail("The file would be larger than 4 GB.", "limit");
    Bytes out(size_t(size), 0);
    uint8_t* b = out.data();
    std::memcpy(b, kMagic, 9);
    wr16(b + 10, kVersion);
    wr16(b + 12, 1);
    wr32(b + 16, uint32_t(n));
    wr32(b + 20, kHeader);
    size_t offset = kHeader + kEntry * n;
    for (size_t i = 0; i < n; i++) {
        uint8_t* e = b + kHeader + kEntry * i;
        const Part& p = *ordered[i];
        std::memcpy(e, p.id.data(), 4);
        wr16(e + 4, p.codec);
        wr16(e + 6, p.lossy ? 1 : 0);
        wr32(e + 8, uint32_t(offset));
        wr32(e + 12, uint32_t(stored[i].size()));
        wr32(e + 16, uint32_t(p.raw.size()));
        wr32(e + 20, crc32(p.raw.data(), p.raw.size()));
        if (!stored[i].empty()) std::memcpy(b + offset, stored[i].data(), stored[i].size());
        offset += stored[i].size();
    }
    uint32_t c = crc32(b, 24);
    c = crc32(b + kHeader, kEntry * n, c);
    wr32(b + 24, c);
    wr32(b + size - 8, crc32(b, size_t(size) - 8));
    wr32(b + size - 4, uint32_t(size));
    return out;
}

Table readTable(const uint8_t* bytes, size_t size) {
    if (size < kHeader + kTrailer) fail("This is not a Parts Sim file (it is too short).", "notpsim");
    for (size_t i = 0; i < 9; i++) {
        if (bytes[i] != kMagic[i]) {
            if (i < 5) fail("This is not a Parts Sim file.", "notpsim");
            fail("This file was changed by a text-mode transfer (line endings were converted); download it again as binary.", "damaged");
        }
    }
    Table t;
    t.version = rd16(bytes + 10);
    t.minReader = rd16(bytes + 12);
    t.flags = rd16(bytes + 14);
    if (t.minReader > kVersion) {
        fail("This file needs a newer Parts Sim (file format " + std::to_string(t.minReader) + "; this app reads up to " + std::to_string(kVersion) + ").", "newer");
    }
    if (rd32(bytes + size - 4) != size) fail("The file is cut short or has extra bytes at the end.", "damaged");
    if (rd32(bytes + size - 8) != crc32(bytes, size - 8)) fail("The file is damaged (checksum mismatch).", "damaged");
    const uint32_t n = rd32(bytes + 16);
    if (n > Limits::sections) fail("The file lists too many sections.", "limit");
    if (rd32(bytes + 20) != kHeader) fail("The section table is not where it should be.", "damaged");
    const uint64_t tableEnd = kHeader + uint64_t(kEntry) * n;
    if (tableEnd + kTrailer > size) fail("The section table runs past the end of the file.", "damaged");
    uint32_t c = crc32(bytes, 24);
    c = crc32(bytes + kHeader, size_t(tableEnd - kHeader), c);
    if (rd32(bytes + 24) != c) fail("The header is damaged (checksum mismatch).", "damaged");
    uint64_t next = tableEnd, rawTotal = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t* e = bytes + kHeader + kEntry * i;
        TableEntry en;
        en.id.assign(reinterpret_cast<const char*>(e), 4);
        en.codec = rd16(e + 4);
        en.flags = rd16(e + 6);
        en.offset = rd32(e + 8);
        en.stored = rd32(e + 12);
        en.raw = rd32(e + 16);
        en.crc = rd32(e + 20);
        if (en.offset != next) fail("The sections overlap or leave gaps.", "damaged");
        next += en.stored;
        if (next > size - kTrailer) fail("A section runs past the end of the file.", "damaged");
        rawTotal += en.raw;
        if (en.raw > Limits::rawPerSection || rawTotal > Limits::rawTotal) fail("The file holds more data than this reader accepts.", "limit");
        if (uint64_t(en.raw) > uint64_t(en.stored) * Limits::ratio + Limits::ratioSlack) fail("A section is compressed far more than any real file (possible decompression bomb).", "bomb");
        if (en.codec == kCodecStored && en.stored != en.raw) fail("A stored section has two different lengths.", "damaged");
        t.entries.push_back(en);
    }
    if (next != size - kTrailer) fail("The file has bytes after its last section.", "damaged");
    return t;
}

Bytes readSection(const uint8_t* bytes, size_t size, const TableEntry& e) {
    if (uint64_t(e.offset) + e.stored > size) fail("A section runs past the end of the file.", "damaged");
    const uint8_t* stored = bytes + e.offset;
    Bytes raw;
    if (e.codec == kCodecStored) raw.assign(stored, stored + e.stored);
    else if (e.codec == kCodecDeflate) raw = inflateRaw(stored, e.stored, e.raw);
    else {
        const std::string name = e.codec == 2 ? "zstd" : e.codec == 3 ? "brotli" : "codec " + std::to_string(e.codec);
        fail("This file uses the " + name + " compression for its " + e.id + " section, which this version cannot read.", "unsupported");
    }
    if (crc32(raw.data(), raw.size()) != e.crc) fail("The " + e.id + " section is damaged (checksum mismatch).", "damaged");
    return raw;
}

File inspectPsim(const uint8_t* bytes, size_t size) {
    try {
        const Table t = readTable(bytes, size);
        File out;
        out.version = t.version;
        out.minReader = t.minReader;
        out.table = t.entries;
        out.size = size;
        const TableEntry* info = nullptr;
        const TableEntry* thumb = nullptr;
        for (const auto& e : t.entries) {
            if (!info && e.id == "INFO") info = &e;
            if (!thumb && e.id == "THMB") thumb = &e;
        }
        if (!info) fail("The file has no INFO section.", "damaged");
        out.info = jsonOf(readSection(bytes, size, *info));
        if (thumb && thumb->codec == kCodecStored) out.thumb = readSection(bytes, size, *thumb);
        return out;
    } catch (const std::bad_alloc&) {
        fail("The file needs more memory than is available.", "limit");
    }
}

File readPsim(const uint8_t* bytes, size_t size, unsigned want) {
    try {
        const Table t = readTable(bytes, size);
        File out;
        out.version = t.version;
        out.minReader = t.minReader;
        out.table = t.entries;
        out.size = size;
        for (const TableEntry& e : t.entries) {
            int idx = -1;
            for (int k = 0; k < 8; k++) if (e.id == kSectionOrder[k]) idx = k;
            if (idx < 0) { out.skipped.push_back(e.id); continue; }
            if (!(want & (1u << idx))) continue;
            Bytes raw;
            try {
                raw = readSection(bytes, size, e);
            } catch (const Error& err) {
                const bool optional = e.id == "THMB" || e.id == "CADS" || e.id == "VIEW";
                if (err.code() == "unsupported" && optional) { out.skipped.push_back(e.id); continue; }
                throw;
            }
            switch (idx) {
                case 0: out.info = jsonOf(raw); break;
                case 1: out.thumb = std::move(raw); break;
                case 2: out.geometry = decodeGeometry(raw.data(), raw.size()); break;
                case 3: out.cad = std::move(raw); break;
                case 4: out.setup = jsonOf(raw); break;
                case 5: out.rfea = decodeArrays(raw.data(), raw.size()); break;
                case 6: out.rair = decodeArrays(raw.data(), raw.size()); break;
                case 7: out.view = jsonOf(raw); break;
            }
        }
        if ((want & WantInfo) && !truthy(out.info)) fail("The file has no INFO section.", "damaged");
        return out;
    } catch (const std::bad_alloc&) {
        fail("The file needs more memory than is available.", "limit");
    }
}

}  // namespace ps::psim
