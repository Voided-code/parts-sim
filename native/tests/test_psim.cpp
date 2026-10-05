// The .psim file (native reader and writer): the same cases as test/psim.test.js, plus the JS-written fixture.
// Compile with -fsanitize=address,undefined to run it under the sanitizers (see native/README.md).
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

#include "check.hpp"
#include "core/mesh.hpp"
#include "core/psim.hpp"
#include "../tools/psim_sample.hpp"

using namespace ps;
using namespace ps::psim;

namespace {

struct Rng {
    uint32_t s;
    explicit Rng(uint32_t seed) : s(seed) {}
    double next() { s = s * 1664525u + 1013904223u; return s / 4294967296.0; }
};

uint32_t rd32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
void wr16(uint8_t* p, uint32_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); }
void wr32(uint8_t* p, uint32_t v) { p[0] = uint8_t(v); p[1] = uint8_t(v >> 8); p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24); }
void wr64f(uint8_t* p, double d) { uint64_t u; std::memcpy(&u, &d, 8); for (int k = 0; k < 8; k++) p[k] = uint8_t(u >> (8 * k)); }

const char* kCreated = "2026-10-04T00:00:00Z";

WriteOptions fixedOptions() {
    WriteOptions o;
    o.created = kCreated;
    return o;
}

/** True when `f` threw a psim::Error (matching code and text when given); anything else is a failure. */
template <class F>
bool throwsPsim(F&& f, const char* code = nullptr, const char* contains = nullptr, const char* what = "") {
    try {
        f();
    } catch (const Error& e) {
        if (code && e.code() != code) { std::printf("    wrong code %s (wanted %s) for %s: %s\n", e.code().c_str(), code, what, e.what()); return false; }
        if (contains && std::string(e.what()).find(contains) == std::string::npos) { std::printf("    wrong message for %s: %s\n", what, e.what()); return false; }
        return true;
    } catch (const std::exception& e) {
        std::printf("    NOT a psim::Error (%s): %s\n", what, e.what());
        return false;
    } catch (...) {
        std::printf("    NOT a psim::Error (%s)\n", what);
        return false;
    }
    return false;
}

#define CHECK_THROWS(expr, ...) CHECK(throwsPsim([&] { (void)(expr); }, ##__VA_ARGS__))

void reseal(Bytes& b) {
    const uint32_t n = rd32(b.data() + 16);
    uint32_t c = crc32(b.data(), 24);
    const size_t end = std::min<size_t>(b.size() - 8, 32 + 24 * size_t(n));
    c = crc32(b.data() + 32, end - 32, c);
    wr32(b.data() + 24, c);
    wr32(b.data() + b.size() - 8, crc32(b.data(), b.size() - 8));
}

Bytes handmade(const char* id, uint16_t codec, const Bytes& stored, uint32_t raw, uint32_t crc) {
    static const uint8_t magic[9] = {0x89, 0x50, 0x53, 0x49, 0x4d, 0x0d, 0x0a, 0x1a, 0x0a};
    Bytes out(32 + 24 + stored.size() + 8, 0);
    std::memcpy(out.data(), magic, 9);
    wr16(out.data() + 10, 1);
    wr16(out.data() + 12, 1);
    wr32(out.data() + 16, 1);
    wr32(out.data() + 20, 32);
    std::memcpy(out.data() + 32, id, 4);
    wr16(out.data() + 36, codec);
    wr32(out.data() + 40, 56);
    wr32(out.data() + 44, uint32_t(stored.size()));
    wr32(out.data() + 48, raw);
    wr32(out.data() + 52, crc);
    if (!stored.empty()) std::memcpy(out.data() + 56, stored.data(), stored.size());
    wr32(out.data() + out.size() - 4, uint32_t(out.size()));
    reseal(out);
    return out;
}

Bytes text(const std::string& s) { return Bytes(s.begin(), s.end()); }

Array qarray(const std::string& name, Enc enc, std::vector<float> data, std::vector<uint32_t> dims = {}) {
    Array a;
    a.name = name;
    a.enc = enc;
    a.f = std::move(data);
    a.dims = std::move(dims);
    return a;
}

Geometry tubeGeometry(bool brep) { return *sampleContent(brep).geometry; }

Bytes loadFixture(const char* name = "v1-tube.psim") {
    std::ifstream in(std::string(PS_TEST_DATA) + "/../../../test/fixtures/psim/" + name, std::ios::binary);
    return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

}  // namespace

TEST("crc32 matches the known value") {
    const Bytes b = text("123456789");
    CHECK(crc32(b.data(), b.size()) == 0xcbf43926u);
    // chaining, as the header CRC does
    uint32_t c = crc32(b.data(), 4);
    c = crc32(b.data() + 4, 5, c);
    CHECK(c == 0xcbf43926u);
}

TEST("json keeps infinity and not-a-number, rejects what JSON does not allow") {
    json::Value v = jobj();
    v.obj["a"] = jnum(INFINITY);
    v.obj["b"] = jnum(-INFINITY);
    v.obj["c"] = jnum(NAN);
    v.obj["d"] = jarr();
    v.obj["d"].arr = {jnum(1), jobj(), jstr("h\xc3\xa9llo \"q\" \\ \n \x01")};
    const json::Value back = parseJson(stringifyJson(v));
    CHECK(back["a"].num == INFINITY);
    CHECK(back["b"].num == -INFINITY);
    CHECK(std::isnan(back["c"].num));
    CHECK(back["d"].size() == 3);
    CHECK(back["d"][2].str == v.obj["d"].arr[2].str);
    CHECK(stringifyJson(v).find("{\"$num\":\"inf\"}") != std::string::npos);
    // doubles survive exactly
    for (double d : {0.1, 1.0 / 3, 1e-7, 123456789.123456789, 5e-324, 1.7976931348623157e308, -2.5e21}) {
        CHECK(parseJson(stringifyJson(jnum(d))).num == d);
    }
    // not JSON
    for (const char* bad : {"", "nan", "inf", "0x10", "01", "1.", ".5", "+1", "[1,]", "{\"a\":1,}", "{'a':1}", "\"\x01\"", "\"\\x\"", "\"\\u12\"", "[1] 2", "tru", "{\"a\"}", "\"abc"}) {
        CHECK(throwsPsim([&] { parseJson(bad); }, "damaged", nullptr, bad));
    }
    CHECK(parseJson("  [1, 2.5e3, -0, true, null]  ").size() == 5);
    CHECK(parseJson("\"\\ud83d\\ude00 \\u00e9\"").str == "\xf0\x9f\x98\x80 \xc3\xa9");
    // nesting and size limits against stack and memory
    CHECK(throwsPsim([&] { parseJson(std::string(100000, '[')); }, "damaged", nullptr, "deep"));
    CHECK(throwsPsim([&] { parseJson(std::string(257, '[') + std::string(257, ']')); }, "damaged", nullptr, "257 deep"));
    CHECK(!throwsPsim([&] { parseJson(std::string(200, '[') + std::string(200, ']')); }, nullptr, nullptr, "200 deep"));
    std::string many = "[";
    for (int i = 0; i < 4100000; i++) many += "1,";
    many += "1]";
    CHECK(throwsPsim([&] { parseJson(many); }, "damaged", nullptr, "too many values"));
}

TEST("arrays: exact encodings round trip bit for bit") {
    Array f = qarray("f", Enc::F32, {0.0f, -0.0f, 1.5f, 3.4e38f, -1e-30f, NAN, INFINITY});
    Array i;
    i.name = "i"; i.enc = Enc::I32; i.i = {0, -1, 5, 2147483647, INT32_MIN};
    Array u;
    u.name = "u"; u.enc = Enc::U8; u.u = {0, 1, 255, 7};
    json::Value meta = jobj();
    meta.obj["k"] = jnum(1);
    const Bytes bytes = encodeArrays(meta, {f, i, u});
    const Arrays r = decodeArrays(bytes.data(), bytes.size());
    CHECK(r.meta["k"].num == 1);
    CHECK(r.list.size() == 3);
    CHECK(std::memcmp(r.find("f")->f.data(), f.f.data(), 4 * f.f.size()) == 0);
    CHECK(r.find("i")->i == i.i);
    CHECK(r.find("u")->u == u.u);
}

TEST("arrays: quantised values stay inside the stated error, NaN stays NaN") {
    Rng rng(7);
    for (Enc enc : {Enc::Q16, Enc::Q8}) {
        for (int withDims = 0; withDims < 2; withDims++) {
            const size_t n = 105;
            std::vector<float> data(n);
            for (size_t k = 0; k < n; k++) data[k] = k % 17 == 3 ? NAN : float(-40 + 90 * rng.next());
            std::vector<Bound> bounds;
            const Bytes bytes = encodeArrays(jobj(), {qarray("a", enc, data, withDims ? std::vector<uint32_t>{7, 5, 3} : std::vector<uint32_t>{})}, &bounds);
            const Arrays decoded = decodeArrays(bytes.data(), bytes.size());
            const Array& back = decoded.list[0];
            const double bound = bounds[0].absolute;
            CHECK(back.err == bound);
            for (size_t k = 0; k < n; k++) {
                if (std::isnan(data[k])) CHECK(std::isnan(back.f[k]));
                else CHECK(std::abs(back.f[k] - data[k]) <= bound * (1 + 1e-6) + 1e-6);
            }
        }
    }
}

TEST("arrays: constant and all-NaN fields, empty arrays") {
    const Bytes bytes = encodeArrays(jobj(), {qarray("c", Enc::Q16, std::vector<float>(10, 3.0f)), qarray("n", Enc::Q8, std::vector<float>(4, NAN)), qarray("e", Enc::Q16, {}), qarray("inf", Enc::Q16, {1.0f, INFINITY, -INFINITY, 2.0f})});
    const Arrays r = decodeArrays(bytes.data(), bytes.size());
    for (float v : r.find("c")->f) CHECK(v == 3.0f);
    for (float v : r.find("n")->f) CHECK(std::isnan(v));
    CHECK(r.find("e")->f.empty());
    CHECK(std::isnan(r.find("inf")->f[1]) && std::isnan(r.find("inf")->f[2]));
}

TEST("arrays: a large grid with Lorenzo prediction stays inside its bound") {
    const uint32_t nx = 37, ny = 29, nz = 23;
    std::vector<float> data(size_t(nx) * ny * nz);
    for (uint32_t z = 0; z < nz; z++)
        for (uint32_t y = 0; y < ny; y++)
            for (uint32_t x = 0; x < nx; x++) data[x + nx * (y + ny * z)] = (x + y + z) % 11 == 0 ? NAN : float(std::sin(0.3 * x) * std::cos(0.2 * y) + 0.1 * z);
    std::vector<Bound> bounds;
    const Bytes bytes = encodeArrays(jobj(), {qarray("g", Enc::Q16, data, {nx, ny, nz})}, &bounds);
    const Arrays decoded = decodeArrays(bytes.data(), bytes.size());
            const Array& back = decoded.list[0];
    CHECK(back.dims.size() == 3 && back.dims[2] == nz);
    for (size_t k = 0; k < data.size(); k++) {
        if (std::isnan(data[k])) CHECK(std::isnan(back.f[k]));
        else CHECK(std::abs(back.f[k] - data[k]) <= bounds[0].absolute * (1 + 1e-6) + 1e-6);
    }
    // the dims must match the length
    CHECK_THROWS(encodeArrays(jobj(), {qarray("g", Enc::Q16, data, {nx, ny, nz + 1})}), "invalid");
}

TEST("geometry: exact mode is bit for bit, quantised mode within its bound") {
    Geometry g = tubeGeometry(false);
    g.faceAngle = 25;
    const Geometry exact = decodeGeometry(encodeGeometry(g, false).data(), encodeGeometry(g, false).size());
    CHECK(exact.vertices.size() == g.vertices.size());
    CHECK(std::memcmp(exact.vertices.data(), g.vertices.data(), 4 * g.vertices.size()) == 0);
    CHECK(exact.tris == g.tris);
    CHECK(exact.faceAngle == 25);
    CHECK(exact.exact);
    Bound bound;
    const Bytes q = encodeGeometry(g, true, &bound);
    const Geometry lossy = decodeGeometry(q.data(), q.size());
    CHECK(lossy.tris == g.tris);
    CHECK(!lossy.exact);
    double worst = 0;
    for (size_t i = 0; i < g.vertices.size(); i++) worst = std::max(worst, double(std::abs(lossy.vertices[i] - g.vertices[i])));
    CHECK(worst <= bound.absolute * 1.0001);
    CHECK(bound.field == "geometry.position");
}

TEST("geometry: CAD face ids round trip") {
    const Geometry g = tubeGeometry(true);
    const Bytes b = encodeGeometry(g, false);
    const Geometry back = decodeGeometry(b.data(), b.size());
    CHECK(back.brepFaces);
    CHECK(back.faceOf == g.faceOf);
    CHECK(back.faceCount == g.faceCount);
    CHECK(back.faceAngle == 0);
}

TEST("geometry: empty part, one triangle") {
    Geometry g;
    const Bytes b = encodeGeometry(g, true);
    const Geometry e = decodeGeometry(b.data(), b.size());
    CHECK(e.vertices.empty() && e.tris.empty());
    g.vertices = {0, 0, 0, 1, 0, 0, 0, 2, 0};
    g.tris = {0, 1, 2};
    for (bool q : {false, true}) {
        const Bytes c = encodeGeometry(g, q);
        const Geometry t = decodeGeometry(c.data(), c.size());
        CHECK(t.tris == g.tris);
        for (size_t i = 0; i < 9; i++) CHECK_NEAR(t.vertices[i], g.vertices[i], 1e-4);
    }
}

TEST("a file with every section round trips and is deterministic") {
    const File c = sampleContent(false);
    const Bytes a = writePsim(c, fixedOptions());
    const Bytes b = writePsim(c, fixedOptions());
    CHECK(a == b);
    const File f = readPsim(a);
    CHECK(f.info["name"].str == "Tube");
    CHECK(f.info["created"].str == kCreated);
    CHECK(f.info["format"].num == 1);
    bool geomBound = false, vmBound = false;
    for (const auto& x : f.info["bounds"].arr) {
        if (x["field"].str == "geometry.position") geomBound = true;
        if (x["field"].str == "static.vm") vmBound = true;
    }
    CHECK(geomBound && vmBound);
    CHECK(f.thumb && *f.thumb == (Bytes{0xff, 0xd8, 0xff, 0xd9}));
    CHECK(f.geometry && f.geometry->tris == c.geometry->tris);
    CHECK(f.setup && f.setup->has("structural"));
    CHECK(f.setup->obj.at("structural")["fixtures"][0]["patches"][0]["tris"][1].num == 12);
    CHECK(f.rfea && f.rfea->meta["static"]["lamBreak"].num == INFINITY);
    CHECK(f.rfea && std::isnan(f.rfea->meta["static"]["minFos"].num));
    CHECK(f.rfea && f.rfea->find("static.vm")->f.size() == c.rfea->list[0].f.size());
    CHECK(f.rair && f.rair->find("airflow.avg.ux")->dims == (std::vector<uint32_t>{6, 5, 4}));
    CHECK(f.view && (*f.view)["tab"].str == "structural");
    std::string ids;
    for (const auto& e : f.table) ids += e.id + " ";
    CHECK(ids == "INFO THMB GEOM SETP RFEA RAIR VIEW ");
    CHECK(f.version == 1 && f.minReader == 1);
    // the info section reads as the documented order
    const Bytes info = readSection(a.data(), a.size(), f.table[0]);
    CHECK(std::string(info.begin(), info.end()).rfind("{\"format\":1,\"app\":{\"name\":\"Parts Sim\",\"version\":\"1.1.0\",\"kind\":\"web\"},\"name\":\"Tube\"", 0) == 0);
}

TEST("every section including the CAD bytes and exact geometry with faces") {
    File c = sampleContent(true);
    c.cad = Bytes(5000);
    for (size_t i = 0; i < c.cad->size(); i++) (*c.cad)[i] = uint8_t(i * 7);
    c.info.obj["contains"].obj["cadName"] = jstr("part.step");
    Array iarr;
    iarr.name = "ids"; iarr.enc = Enc::I32; iarr.i = {-5, 7, 1 << 30, -1};
    c.rfea->list.push_back(iarr);
    WriteOptions o = fixedOptions();
    o.quantised = false;
    const Bytes bytes = writePsim(c, o);
    const File f = readPsim(bytes);
    CHECK(f.cad && *f.cad == *c.cad);
    CHECK(f.geometry->exact && f.geometry->brepFaces && f.geometry->faceOf == c.geometry->faceOf);
    CHECK(std::memcmp(f.geometry->vertices.data(), c.geometry->vertices.data(), 4 * c.geometry->vertices.size()) == 0);
    CHECK(f.rfea->find("ids")->i == iarr.i);
    // GEOM is exact so it is not marked lossy; RFEA is
    for (const auto& e : f.table) CHECK(((e.flags & 1) != 0) == (e.id == "RFEA" || e.id == "RAIR"));
}

TEST("inspect reads only the info and the thumbnail") {
    const Bytes bytes = writePsim(sampleContent(false), fixedOptions());
    const File i = inspectPsim(bytes);
    CHECK(i.info["name"].str == "Tube");
    CHECK(i.thumb && i.thumb->size() == 4);
    CHECK(i.size == bytes.size());
    CHECK(!i.geometry);
}

TEST("sections the reader does not know are skipped") {
    Bytes bytes = writePsim(sampleContent(false), fixedOptions());
    const uint32_t n = rd32(bytes.data() + 16);
    for (uint32_t i = 0; i < n; i++) {
        uint8_t* e = bytes.data() + 32 + 24 * i;
        if (std::memcmp(e, "THMB", 4) == 0) std::memcpy(e, "ZZZZ", 4);
    }
    reseal(bytes);
    const File f = readPsim(bytes);
    CHECK(f.skipped == std::vector<std::string>{"ZZZZ"});
    CHECK(!f.thumb);
    CHECK(f.info["name"].str == "Tube");
    // an optional section with a codec this version does not know is skipped too; a needed one is named
    Bytes b2 = writePsim(sampleContent(false), fixedOptions());
    for (uint32_t i = 0; i < n; i++) {
        uint8_t* e = b2.data() + 32 + 24 * i;
        if (std::memcmp(e, "VIEW", 4) == 0) wr16(e + 4, 2);
        if (std::memcmp(e, "SETP", 4) == 0) wr16(e + 4, 3);
    }
    reseal(b2);
    CHECK_THROWS(readPsim(b2), "unsupported", "brotli compression for its SETP section");
    const File g = readPsim(b2, WantInfo | WantGeometry | WantView);
    CHECK(g.skipped == std::vector<std::string>{"VIEW"});
}

TEST("safety: wrong files and text-mode damage are named") {
    CHECK_THROWS(readPsim(Bytes(10)), "notpsim", "not a Parts Sim file");
    CHECK_THROWS(readPsim(text(std::string(100, 'x'))), "notpsim", "not a Parts Sim file");
    Bytes bytes = writePsim(sampleContent(false), fixedOptions());
    Bytes crlf = bytes;
    crlf[5] = 0x0a;
    CHECK_THROWS(readPsim(crlf), "damaged", "text-mode");
}

TEST("safety: a newer file says so") {
    Bytes bytes = writePsim(sampleContent(false), fixedOptions());
    wr16(bytes.data() + 12, 9);
    reseal(bytes);
    CHECK_THROWS(readPsim(bytes), "newer", "needs a newer Parts Sim");
}

static File withoutResults() {
    File c = sampleContent(false);
    c.rfea.reset();
    c.rair.reset();
    return c;
}

TEST("safety: every truncation is rejected (300 cut points)") {
    const Bytes bytes = writePsim(withoutResults(), fixedOptions());
    std::vector<size_t> cuts = {0, 1, 8, 9, 31, 32, 33, 55, bytes.size() - 9, bytes.size() - 8, bytes.size() - 4, bytes.size() - 1};
    Rng rng(3);
    while (cuts.size() < 300) cuts.push_back(size_t(rng.next() * bytes.size()));
    for (size_t cut : cuts) {
        const Bytes part(bytes.begin(), bytes.begin() + std::ptrdiff_t(cut));
        CHECK(throwsPsim([&] { readPsim(part); }, nullptr, nullptr, "truncation"));
    }
}

TEST("safety: every flipped bit is caught (400 flips)") {
    const Bytes bytes = writePsim(withoutResults(), fixedOptions());
    Rng rng(11);
    for (int k = 0; k < 400; k++) {
        Bytes copy = bytes;
        copy[size_t(rng.next() * copy.size())] ^= uint8_t(1u << int(rng.next() * 8));
        CHECK(throwsPsim([&] { readPsim(copy); }, nullptr, nullptr, "bit flip"));
    }
}

TEST("safety: header lies about counts and lengths (checksums repaired)") {
    const Bytes bytes = writePsim(sampleContent(false), fixedOptions());
    const std::vector<std::pair<size_t, uint32_t>> lies = {
        {16, 0xffffffffu}, {16, 3}, {20, 9999}, {32 + 8, 40}, {32 + 12, 0xfffffff0u}, {32 + 16, 0xffffffffu}, {32 + 24 + 16, 0x7fffffffu},
        {32 + 24 + 8, 1}, {32 + 4, 0x0505}, {16, 0}, {32 + 20, 12345},
    };
    for (const auto& [at, value] : lies) {
        Bytes copy = bytes;
        wr32(copy.data() + at, value);
        reseal(copy);
        CHECK(throwsPsim([&] { readPsim(copy); }, nullptr, nullptr, "header lie"));
    }
}

TEST("safety: a decompression bomb is stopped, not inflated") {
    const Bytes zeros(size_t(64) << 20, 0);
    const Bytes bomb = deflateRaw(zeros.data(), zeros.size());
    const auto t0 = std::chrono::steady_clock::now();
    // claims 1 MiB
    CHECK_THROWS(readPsim(handmade("INFO", 1, bomb, 1u << 20, 0)), "bomb");
    CHECK(pstest::ms(t0) < 5000);
    // claims the real size, but a ratio no real file has
    const Bytes z40 = Bytes(size_t(40) << 20, 0);
    const Bytes small = deflateRaw(z40.data(), z40.size());
    if (double(z40.size()) / double(small.size()) > 1100) CHECK_THROWS(readPsim(handmade("INFO", 1, small, uint32_t(z40.size()), 0)), "bomb");
    // claims a section bigger than the cap
    CHECK_THROWS(readPsim(handmade("INFO", 1, Bytes{0}, 0x7fffffffu, 0)), "limit");
    // stored sections with two lengths
    CHECK_THROWS(readPsim(handmade("INFO", 0, Bytes{1, 2, 3}, 5, 0)), "damaged");
}

TEST("safety: a section that is shorter or has a wrong checksum is rejected") {
    const Bytes info = text("{\"format\":1}");
    const Bytes z = deflateRaw(info.data(), info.size());
    const uint32_t crc = crc32(info.data(), info.size());
    CHECK_THROWS(readPsim(handmade("INFO", 1, z, uint32_t(info.size()) + 5, crc)), "damaged", "shorter");
    CHECK_THROWS(readPsim(handmade("INFO", 1, z, uint32_t(info.size()), 1)), "damaged", "checksum");
    CHECK(readPsim(handmade("INFO", 1, z, uint32_t(info.size()), crc)).info["format"].num == 1);
    // trailing bytes after the deflate stream, a cut stream, garbage
    Bytes trail = z;
    trail.push_back(0);
    CHECK_THROWS(readPsim(handmade("INFO", 1, trail, uint32_t(info.size()), crc)), "damaged");
    const Bytes cut(z.begin(), z.end() - 2);
    CHECK_THROWS(readPsim(handmade("INFO", 1, cut, uint32_t(info.size()), crc)), "damaged");
    CHECK_THROWS(readPsim(handmade("INFO", 1, Bytes(30, 0xff), 100, crc)), "damaged");
    // a section that is not JSON, not UTF-8, or not an object with INFO content
    for (const std::string& body : {std::string("{\"format\":"), std::string("\xff\xfe"), std::string("0"), std::string("null")}) {
        const Bytes raw = text(body);
        const Bytes zz = deflateRaw(raw.data(), raw.size());
        CHECK(throwsPsim([&] { readPsim(handmade("INFO", 1, zz, uint32_t(raw.size()), crc32(raw.data(), raw.size()))); }, "damaged", nullptr, body.c_str()));
    }
    // a byte order mark is dropped, as a text decoder does
    const Bytes bom = text("\xef\xbb\xbf{\"format\":2}");
    const Bytes zb = deflateRaw(bom.data(), bom.size());
    CHECK(readPsim(handmade("INFO", 1, zb, uint32_t(bom.size()), crc32(bom.data(), bom.size()))).info["format"].num == 2);
}

TEST("safety: oversized and inconsistent counts inside sections") {
    Geometry t;
    t.vertices = tubeGeometry(false).vertices;
    t.tris = tubeGeometry(false).tris;
    t.vertices.resize(36 * 3);
    t.tris.assign({0, 1, 2, 3, 4, 5, 6, 7, 8});
    const Bytes g = encodeGeometry(t, false);
    Bytes big = g;
    wr32(big.data() + 4, 0xffffff00u);
    CHECK_THROWS(decodeGeometry(big.data(), big.size()), "limit");
    Bytes off = g;
    wr32(off.data() + 8, 4);
    CHECK_THROWS(decodeGeometry(off.data(), off.size()), "damaged");
    Bytes mode = g;
    mode[0] = 5;
    CHECK_THROWS(decodeGeometry(mode.data(), mode.size()), "unsupported");
    mode = g;
    mode[1] = 8;
    CHECK_THROWS(decodeGeometry(mode.data(), mode.size()), "unsupported");
    Bytes bbox = g;
    wr64f(bbox.data() + 20, NAN);
    CHECK_THROWS(decodeGeometry(bbox.data(), bbox.size()), "damaged", "bounding box");
    CHECK_THROWS(decodeGeometry(g.data(), 40), "damaged");
    // an index past the last vertex, a degenerate triangle
    Geometry bad = t;
    bad.tris[0] = 99999;
    const Bytes b1 = encodeGeometry(bad, false);
    CHECK_THROWS(decodeGeometry(b1.data(), b1.size()), "damaged", "vertex that does not exist");
    Geometry deg = t;
    deg.tris[1] = deg.tris[0];
    const Bytes b2 = encodeGeometry(deg, false);
    CHECK_THROWS(decodeGeometry(b2.data(), b2.size()), "damaged", "one vertex twice");
    // a face id past the face count
    Geometry face = tubeGeometry(true);
    face.faceCount = 3;
    const Bytes b3 = encodeGeometry(face, false);
    CHECK_THROWS(decodeGeometry(b3.data(), b3.size()), "damaged", "face that does not exist");
    // arrays: header longer than the section, byte counts that lie
    const Bytes ok = encodeArrays(jobj(), {qarray("a", Enc::Q16, {1, 2, 3})});
    Bytes hl = ok;
    wr32(hl.data(), 0xffffffffu);
    CHECK_THROWS(decodeArrays(hl.data(), hl.size()), "damaged");
    CHECK_THROWS(decodeArrays(ok.data(), ok.size() - 1), "damaged");
    CHECK_THROWS(decodeArrays(ok.data(), 3), "damaged");
}

TEST("safety: array headers that lie") {
    auto section = [](const std::string& head, size_t dataBytes) {
        Bytes b(4 + head.size() + dataBytes, 0);
        wr32(b.data(), uint32_t(head.size()));
        std::memcpy(b.data() + 4, head.data(), head.size());
        return b;
    };
    struct Case { std::string head; size_t data; const char* code; };
    const Case cases[] = {
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f32\",\"n\":3,\"bytes\":12}]}", 12, nullptr},  // honest
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f32\",\"n\":3,\"bytes\":8}]}", 12, "damaged"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f32\",\"n\":3,\"bytes\":12}]}", 11, "damaged"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f32\",\"n\":3,\"bytes\":12}]}", 13, "damaged"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f32\",\"n\":1e300,\"bytes\":12}]}", 12, "limit"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f32\",\"n\":600000000,\"bytes\":2400000000}]}", 12, "limit"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f32\",\"n\":-1,\"bytes\":-4}]}", 0, "damaged"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f32\",\"n\":1.5,\"bytes\":6}]}", 6, "damaged"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"f64\",\"n\":3,\"bytes\":12}]}", 12, "damaged"},
        {"{\"arrays\":[{\"name\":7,\"enc\":\"f32\",\"n\":3,\"bytes\":12}]}", 12, "damaged"},
        {"{\"arrays\":[3]}", 0, "damaged"},
        {"{\"arrays\":{}}", 0, "damaged"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"u8\",\"n\":3,\"bytes\":3},{\"name\":\"a\",\"enc\":\"u8\",\"n\":3,\"bytes\":3}]}", 6, "damaged"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":3,\"bytes\":6}]}", 6, "damaged"},  // no range
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":3,\"bytes\":6,\"min\":{\"$num\":\"inf\"},\"max\":1}]}", 6, "damaged"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":3,\"bytes\":6,\"min\":0,\"max\":1}]}", 6, nullptr},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":3,\"bytes\":6,\"min\":0,\"max\":1,\"dims\":[3,1,2]}]}", 6, "invalid"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":3,\"bytes\":6,\"min\":0,\"max\":1,\"dims\":[3,1]}]}", 6, "invalid"},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":3,\"bytes\":6,\"min\":0,\"max\":1,\"dims\":[3,1,1]}]}", 6, nullptr},
        {"{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":3,\"bytes\":6,\"min\":0,\"max\":1,\"dims\":[0.5,6,1]}]}", 6, "invalid"},
        {"{\"nothing\":1}", 0, "damaged"},
        {"[1]", 0, "damaged"},
        {"{\"arrays\":[]", 0, "damaged"},
    };
    for (const Case& c : cases) {
        const Bytes b = section(c.head, c.data);
        if (c.code) CHECK(throwsPsim([&] { decodeArrays(b.data(), b.size()); }, c.code, nullptr, c.head.c_str()));
        else CHECK(!throwsPsim([&] { decodeArrays(b.data(), b.size()); }, nullptr, nullptr, c.head.c_str()));
    }
    // too many arrays
    std::string many = "{\"arrays\":[";
    for (int i = 0; i < 100001; i++) many += (i ? "," : "") + std::string("{}");
    many += "]}";
    const Bytes b = section(many, 0);
    CHECK_THROWS(decodeArrays(b.data(), b.size()), "limit");
}

TEST("fuzz: random damage never produces anything but a psim::Error, and never hangs (600 cases)") {
    const Bytes base = writePsim(sampleContent(false), fixedOptions());
    Rng rng(2026);
    const auto t0 = std::chrono::steady_clock::now();
    int accepted = 0;
    const char* env = std::getenv("PSIM_FUZZ_CASES");  // a longer run by hand, e.g. under the sanitizers
    const int cases = env ? std::atoi(env) : 600;
    for (int k = 0; k < cases; k++) {
        Bytes copy = base;
        const int edits = 1 + int(rng.next() * 6);
        for (int e = 0; e < edits; e++) {
            const size_t at = size_t(rng.next() * copy.size());
            const double kind = rng.next();
            if (kind < 0.4) copy[at] = uint8_t(rng.next() * 256);
            else if (kind < 0.7) copy[at] ^= uint8_t(1u << int(rng.next() * 8));
            else {
                const size_t len = 1 + size_t(rng.next() * 64);
                const uint8_t v = uint8_t(rng.next() * 256);
                for (size_t q = at; q < std::min(copy.size(), at + len); q++) copy[q] = v;
            }
        }
        if (rng.next() < 0.5 && copy.size() > 40) reseal(copy);
        try {
            readPsim(copy);
            accepted++;
        } catch (const Error&) {
        } catch (...) {
            std::printf("    fuzz %d: not a psim::Error\n", k);
            CHECK(false);
        }
    }
    CHECK(env || pstest::ms(t0) < 25000);
    CHECK(accepted < cases);
}

TEST("fuzz: damage inside the decompressed sections (checksums valid)") {
    // rewrite each section's bytes with edits and honest CRCs, so the parsers see the damage
    const File c = sampleContent(true);
    Rng rng(99);
    const Bytes base = writePsim(c, fixedOptions());
    const Table t = readTable(base);
    int rejected = 0;
    const char* env = std::getenv("PSIM_FUZZ_CASES");
    const int cases = env ? std::atoi(env) / 2 : 400;
    for (int k = 0; k < cases; k++) {
        const TableEntry& e = t.entries[size_t(rng.next() * t.entries.size())];
        if (e.id == "THMB") continue;
        Bytes raw = readSection(base.data(), base.size(), e);
        const int edits = 1 + int(rng.next() * 4);
        for (int q = 0; q < edits; q++) raw[size_t(rng.next() * raw.size())] = uint8_t(rng.next() * 256);
        // build a one-section file around the damaged bytes
        const Bytes z = deflateRaw(raw.data(), raw.size());
        const Bytes file = handmade(e.id.c_str(), 1, z, uint32_t(raw.size()), crc32(raw.data(), raw.size()));
        try {
            readPsim(file.data(), file.size(), WantAll & ~unsigned(WantInfo));
        } catch (const Error&) {
            rejected++;
        } catch (...) {
            std::printf("    section fuzz %d (%s): not a psim::Error\n", k, e.id.c_str());
            CHECK(false);
        }
    }
    CHECK(rejected > 20);
}

TEST("the file written by the JavaScript code reads, with the right counts and values") {
    const Bytes bytes = loadFixture();
    CHECK(bytes.size() > 1000);
    const File f = readPsim(bytes);
    CHECK(f.version == 1);
    CHECK(f.info["name"].str == "Tube");
    CHECK(f.info["created"].str == kCreated);
    CHECK(f.geometry && f.geometry->tris.size() == 40 * 24 * 6);
    CHECK(f.geometry && f.geometry->vertices.size() == 3 * 40 * 24);
    CHECK(f.rfea && f.rfea->find("static.vm") && f.rfea->find("static.vm")->f.size() == 40 * 24);
    CHECK(f.rfea && f.rfea->find("static.u")->f.size() == 3 * 40 * 24);
    CHECK(readTable(bytes).entries.size() == 7);
    // the values are those of the formulas, within the stated bounds
    const File mine = sampleContent(false);
    double posBound = 0, vmBound = 0, uBound = 0, uxBound = 0;
    for (const auto& b : f.info["bounds"].arr) {
        if (b["field"].str == "geometry.position") posBound = b["absolute"].num;
        if (b["field"].str == "static.vm") vmBound = b["absolute"].num;
        if (b["field"].str == "static.u") uBound = b["absolute"].num;
        if (b["field"].str == "airflow.avg.ux") uxBound = b["absolute"].num;
    }
    CHECK(posBound > 0 && vmBound > 0 && uBound > 0 && uxBound > 0);
    CHECK(f.geometry->tris == mine.geometry->tris);
    for (size_t i = 0; i < mine.geometry->vertices.size(); i++) CHECK_NEAR(f.geometry->vertices[i], mine.geometry->vertices[i], posBound * 1.0001 + 1e-5);
    const Array* vm = f.rfea->find("static.vm");
    for (size_t i = 0; i < vm->f.size(); i++) CHECK_NEAR(vm->f[i], mine.rfea->list[0].f[i], vmBound * 1.0001 + 1e-3);
    const Array* u = f.rfea->find("static.u");
    for (size_t i = 0; i < u->f.size(); i++) CHECK_NEAR(u->f[i], mine.rfea->list[1].f[i], uBound * 1.0001 + 1e-9);
    const Array* ux = f.rair->find("airflow.avg.ux");
    CHECK(ux->dims == (std::vector<uint32_t>{6, 5, 4}));
    for (size_t i = 0; i < ux->f.size(); i++) CHECK_NEAR(ux->f[i], mine.rair->list[0].f[i], uxBound * 1.0001 + 1e-7);
    CHECK(f.rfea->meta["static"]["lamBreak"].num == INFINITY);
    CHECK(std::isnan(f.rfea->meta["static"]["minFos"].num));
    CHECK(f.setup->obj.at("structural")["study"].str == "static");
    CHECK((*f.view)["tab"].str == "structural");
}

TEST("the native file has the same uncompressed sections as the JavaScript one") {
    const Bytes js = loadFixture("v1-tube-stride.psim");
    const Bytes mine = writePsim(sampleContent(false), fixedOptions());
    const Table a = readTable(js), b = readTable(mine);
    CHECK(a.entries.size() == b.entries.size());
    for (size_t i = 0; i < a.entries.size() && i < b.entries.size(); i++) {
        CHECK(a.entries[i].id == b.entries[i].id);
        CHECK(a.entries[i].raw == b.entries[i].raw);
        // GEOM and the arrays agree when the sine and cosine round the same way on this machine
        const bool same = a.entries[i].crc == b.entries[i].crc;
        std::printf("    %s: raw %u, crc %s\n", a.entries[i].id.c_str(), a.entries[i].raw, same ? "identical" : "differs");
        if (a.entries[i].id != "RFEA") CHECK(same);  // RFEA: the JS writer keeps the order its meta was built in, the native one sorts the keys
    }
}

TEST("a read and a write of the sample take well under a second") {
    const Geometry g = tubeGeometry(false);
    File c = sampleContent(false);
    const auto t0 = std::chrono::steady_clock::now();
    const Bytes b = writePsim(c, fixedOptions());
    const File f = readPsim(b);
    CHECK(pstest::ms(t0) < 1000);
    CHECK(f.geometry->vertices.size() == g.vertices.size());
}

TEST("stride: vector fields round trip within the bound and shrink") {
    Rng rng(5);
    std::vector<float> u(3 * 500);
    for (size_t i = 0; i < u.size(); i++) u[i] = float(0.01 * std::cos(double(i / 3) / 11) * (1 + (i % 3)) + 1e-5 * rng.next());
    Array plain = qarray("u", Enc::Q16, u), strided = plain;
    strided.stride = 3;
    for (Enc enc : {Enc::Q16, Enc::Q8}) {
        plain.enc = strided.enc = enc;
        std::vector<Bound> b;
        const Bytes bp = encodeArrays(jobj(), {plain}), bs = encodeArrays(jobj(), {strided}, &b);
        const Arrays r = decodeArrays(bs.data(), bs.size());
        CHECK(r.list[0].stride == 3);
        for (size_t i = 0; i < u.size(); i++) CHECK(std::abs(r.list[0].f[i] - u[i]) <= b[0].absolute * (1 + 1e-6) + 1e-9);
        if (enc == Enc::Q16) {
            CHECK(decodeArrays(bp.data(), bp.size()).list[0].stride == 1);
            CHECK(deflateRaw(bs.data(), bs.size()).size() < deflateRaw(bp.data(), bp.size()).size());  // 3 interleaved components: the vector predictor wins
        }
    }
    // bad strides are refused when writing
    Array bad = strided;
    bad.stride = 5;
    CHECK_THROWS(encodeArrays(jobj(), {bad}), "invalid");
    bad.stride = 2;
    bad.f.resize(501);
    CHECK_THROWS(encodeArrays(jobj(), {bad}), "invalid");
    bad = qarray("f", Enc::F32, {1, 2, 3, 4, 5, 6});
    bad.stride = 3;
    CHECK_THROWS(encodeArrays(jobj(), {bad}), "invalid");
    bad = qarray("g", Enc::Q16, {1, 2, 3, 4, 5, 6}, {3, 2, 1});
    bad.stride = 3;
    CHECK_THROWS(encodeArrays(jobj(), {bad}), "invalid");
    // and when reading
    auto section = [](const std::string& head, size_t dataBytes) {
        Bytes b(4 + head.size() + dataBytes, 0);
        wr32(b.data(), uint32_t(head.size()));
        std::memcpy(b.data() + 4, head.data(), head.size());
        return b;
    };
    for (const char* extra : {"\"stride\":5", "\"stride\":0", "\"stride\":2.5", "\"stride\":\"3\"", "\"stride\":2", "\"stride\":3,\"dims\":[6,1,1]"}) {
        const std::string head = std::string("{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":6,\"bytes\":12,\"min\":0,\"max\":1,") + extra + "}]}";
        const Bytes b = section(head, 12);
        const bool ok = std::string(extra) == "\"stride\":2";
        CHECK(ok ? !throwsPsim([&] { decodeArrays(b.data(), b.size()); }, nullptr, nullptr, extra) : throwsPsim([&] { decodeArrays(b.data(), b.size()); }, "damaged", nullptr, extra));
    }
    const Bytes b5 = section("{\"arrays\":[{\"name\":\"a\",\"enc\":\"q16\",\"n\":5,\"bytes\":10,\"min\":0,\"max\":1,\"stride\":2}]}", 10);
    CHECK_THROWS(decodeArrays(b5.data(), b5.size()), "damaged");
}

TEST("the stride fixture written by JavaScript reads (and the old one without stride)") {
    const File f = readPsim(loadFixture("v1-tube-stride.psim"));
    const Array* u = f.rfea->find("static.u");
    CHECK(u && u->stride == 3 && u->f.size() == 3 * 40 * 24);
    const File mine = sampleContent(false);
    double bound = 0;
    for (const auto& b : f.info["bounds"].arr) if (b["field"].str == "static.u") bound = b["absolute"].num;
    CHECK(bound > 0);
    for (size_t i = 0; i < u->f.size(); i++) CHECK_NEAR(u->f[i], mine.rfea->list[1].f[i], bound * 1.0001 + 1e-9);
    CHECK(readPsim(loadFixture("v1-tube.psim")).rfea->find("static.u")->stride == 1);
}

TEST("restorePart rebuilds a built part exactly, without refining") {
    MeshSource m;
    const double x = 20, y = 5, z = 3;
    const double v[8][3] = {{-x, -y, -z}, {x, -y, -z}, {x, y, -z}, {-x, y, -z}, {-x, -y, z}, {x, -y, z}, {x, y, z}, {-x, y, z}};
    for (auto& q : v) for (double d : q) m.positions.push_back(float(d));
    m.index = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3};
    for (bool brep : {false, true}) {
        MeshSource s = m;
        if (brep) s.faceIds = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
        const auto a = buildPart(s, BuildOptions{3000, 20});
        CHECK(a->nTri > 200);  // refined
        const auto b = restorePart(a->name, a->vertices, a->tris, a->brepFaces ? a->faceOf : std::vector<int32_t>{}, a->brepFaces, a->faceCount, a->faceAngle);
        CHECK(b->nTri == a->nTri && b->nVert == a->nVert);
        CHECK(b->brepFaces == a->brepFaces);
        CHECK(b->neighbors == a->neighbors);
        CHECK(b->faceOf == a->faceOf);
        CHECK(b->faceCount == a->faceCount);
        CHECK(b->triNormal == a->triNormal);
        CHECK(b->triArea == a->triArea);
        CHECK(b->vertNormal == a->vertNormal);
        CHECK(b->edges == a->edges);
        CHECK(b->displayPosition == a->displayPosition && b->displayNormal == a->displayNormal && b->displaySrc == a->displaySrc);
        CHECK(b->volume == a->volume && b->area == a->area);
        CHECK(b->bbox.min == a->bbox.min && b->bbox.max == a->bbox.max);
        CHECK(b->faceAngle == a->faceAngle);
    }
}

TEST("series: arrays predicted from the previous step round trip and code smaller; bad bases are refused") {
    const size_t n = 3000, steps = 6;
    std::vector<std::vector<float>> frames(steps, std::vector<float>(n));
    for (size_t k = 0; k < steps; k++)
        for (size_t i = 0; i < n; i++) frames[k][i] = float(std::sin(double(i) / 30) * (1 + double(k) * 0.05) + (i % 97 == 0 ? double(k) * 0.01 : 0));
    std::vector<const std::vector<float>*> ptrs;
    std::vector<Array> plain;
    double lo = 1e300, hi = -1e300;
    for (auto& f : frames) { ptrs.push_back(&f); for (float v : f) { lo = std::min(lo, double(v)); hi = std::max(hi, double(v)); } }
    for (size_t k = 0; k < steps; k++) plain.push_back(qarray("f." + std::to_string(k), Enc::Q16, frames[k]));
    const auto series = seriesArrays(ptrs, [](size_t i) { return "f." + std::to_string(i); });
    const Bytes bp = encodeArrays(jobj(), plain), bs = encodeArrays(jobj(), series);
    const Arrays back = decodeArrays(bs.data(), bs.size());
    const double bound = (hi - lo) / (2 * 65534);
    for (size_t k = 0; k < steps; k++) {
        const Array* a = back.find("f." + std::to_string(k));
        CHECK(a && a->f.size() == n);
        for (size_t i = 0; a && i < n; i++) CHECK(std::abs(a->f[i] - frames[k][i]) <= bound * 1.0001 + 1e-6);
    }
    CHECK(deflateRaw(bs.data(), bs.size()).size() < 0.8 * double(deflateRaw(bp.data(), bp.size()).size()));
    // refused when writing: range differs from the base, missing base, non-quantised
    Array a = qarray("a", Enc::Q16, std::vector<float>(4, 0.f)), b = qarray("b", Enc::Q16, std::vector<float>(4, 0.f));
    b.base = "a"; b.hasRange = true; b.rangeMin = 0; b.rangeMax = 1;
    CHECK_THROWS(encodeArrays(jobj(), {a, b}), "invalid");
    b.hasRange = false; b.base = "missing";
    CHECK_THROWS(encodeArrays(jobj(), {b}), "invalid");
    Array f1 = qarray("a", Enc::F32, {1, 2, 3, 4}), f2 = qarray("b", Enc::F32, {1, 2, 3, 4});
    f2.base = "a";
    CHECK_THROWS(encodeArrays(jobj(), {f1, f2}), "invalid");
    // a header that names a base that does not exist is refused
    std::string text(bs.begin(), bs.end());
    const size_t at = text.find("\"base\":\"f.0\"");
    CHECK(at != std::string::npos);
    Bytes bad = bs;
    bad[at + 10] = '9';
    CHECK_THROWS(decodeArrays(bad.data(), bad.size()), "damaged");
    bad = bs;
    const size_t at1 = text.find("\"base\":\"f.1\"");
    bad[at1 + 10] = '0';  // f.2 predicted from f.0 now: same range, so it decodes, to different values; but base "f.5" (later) must fail
    const size_t at2 = text.find("\"base\":\"f.2\"");
    bad = bs;
    bad[at2 + 10] = '5';
    CHECK_THROWS(decodeArrays(bad.data(), bad.size()), "damaged");
}

TEST_MAIN
