// The .psim file: container, section codecs and array coding (native reader and writer).
//
// The format is specified in docs/psim-format.md (version 1); src/core/psim.js is the JavaScript
// reference and the two must stay byte-compatible. This code knows nothing about the app's panels or
// about Qt: it turns plain data (a built mesh, JSON, float arrays) into bytes and back, and checks
// everything it reads, because files come from strangers. Every failure is a ps::psim::Error whose
// code() is one of: notpsim, damaged, newer, unsupported, limit, bomb, invalid.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/json.hpp"

namespace ps::psim {

constexpr int kVersion = 1;
using Bytes = std::vector<uint8_t>;

/** Same limits as the JavaScript reader (docs/psim-format.md, Safety). */
struct Limits {
    static constexpr size_t sections = 64;
    static constexpr uint64_t rawPerSection = 512ull << 20;
    static constexpr uint64_t rawTotal = 1024ull << 20;
    static constexpr uint64_t ratio = 1100;
    static constexpr uint64_t ratioSlack = 64 * 1024;
    static constexpr uint64_t jsonSection = 64ull << 20;
    static constexpr uint64_t vertices = 50000000;
    static constexpr uint64_t triangles = 50000000;
    static constexpr uint64_t arrayElements = 500000000;
    static constexpr size_t arrays = 100000;
    /** Native only: a parsed JSON value costs about 120 bytes, so a 64 MB text of "1,1,1..." must not become 8 GB. */
    static constexpr size_t jsonNodes = 4000000;
};

class Error : public std::runtime_error {
public:
    Error(const std::string& message, const std::string& code = "invalid") : std::runtime_error(message), code_(code) {}
    const std::string& code() const { return code_; }

private:
    std::string code_;
};

// ---------------------------------------------------------------- primitives

uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0);
/** zlib, raw deflate, level 6 (the same as the browser's CompressionStream('deflate-raw')). */
Bytes deflateRaw(const uint8_t* p, size_t n);
/** Inflates `stored` bytes, which must come out exactly `rawLength` long; a longer stream is a bomb. */
Bytes inflateRaw(const uint8_t* p, size_t stored, size_t rawLength);

/** Preferred key order of JSON objects, keyed by the name of the parent key ("" for the top object). */
using KeyOrder = std::map<std::string, std::vector<std::string>>;
/** Compact JSON; keys in `order` first (in that order), the rest alphabetical; non-finite numbers as {"$num":...}. */
std::string stringifyJson(const json::Value& v, const KeyOrder* order = nullptr);
/** Strict JSON with depth and size limits; {"$num":"inf"|"-inf"|"nan"} become numbers. Throws Error(damaged). */
json::Value parseJson(const std::string& text);

/** Small builders for json::Value. */
inline json::Value jnum(double d) { json::Value v; v.type = json::Value::Number; v.num = d; return v; }
inline json::Value jstr(const std::string& s) { json::Value v; v.type = json::Value::String; v.str = s; return v; }
inline json::Value jbool(bool b) { json::Value v; v.type = json::Value::Bool; v.b = b; return v; }
inline json::Value jarr() { json::Value v; v.type = json::Value::Array; return v; }
inline json::Value jobj() { json::Value v; v.type = json::Value::Object; return v; }
/** The documented key order of the INFO section (also used to print it). */
const KeyOrder& infoKeyOrder();

// ---------------------------------------------------------------- arrays container

enum class Enc { F32, I32, U8, Q16, Q8 };
const char* encName(Enc e);

/** One result array. F32, Q16 and Q8 keep their values in `f` (decoded quantised arrays hold the dequantised values). */
struct Array {
    std::string name;
    Enc enc = Enc::F32;
    std::vector<float> f;
    std::vector<int32_t> i;
    std::vector<uint8_t> u;
    std::vector<uint32_t> dims;  // empty, or {nx, ny, nz} with nx*ny*nz = size(), x fastest
    double err = 0;              // worst-case error of a quantised array (set by the reader and the writer)
    size_t size() const { return enc == Enc::I32 ? i.size() : enc == Enc::U8 ? u.size() : f.size(); }
};

struct Arrays {
    json::Value meta;
    std::vector<Array> list;  // in file order
    const Array* find(const std::string& name) const;
};

struct Bound {
    std::string field;
    double absolute = 0;
};

/** Body of a result section. `bounds` receives the error bound of each quantised array. */
Bytes encodeArrays(const json::Value& meta, const std::vector<Array>& arrays, std::vector<Bound>* bounds = nullptr);
Arrays decodeArrays(const uint8_t* p, size_t n);

// ---------------------------------------------------------------- geometry

struct Geometry {
    std::vector<float> vertices;   // 3 per vertex
    std::vector<uint32_t> tris;    // 3 per triangle
    std::vector<int32_t> faceOf;   // one per triangle; used (and stored) only when brepFaces
    bool brepFaces = false;
    uint32_t faceCount = 0;
    double faceAngle = 20;         // degrees; 0 for B-rep faces
    double bboxMin[3] = {0, 0, 0}; // set by the reader
    double bboxMax[3] = {0, 0, 0};
    bool exact = false;            // set by the reader
};

/** quantised = 16 bits per axis (error span/131070), otherwise exact float32. */
Bytes encodeGeometry(const Geometry& g, bool quantised, Bound* bound = nullptr);
Geometry decodeGeometry(const uint8_t* p, size_t n);

// ---------------------------------------------------------------- container

struct TableEntry {
    std::string id;
    uint16_t codec = 0, flags = 0;
    uint32_t offset = 0, stored = 0, raw = 0, crc = 0;
};

struct Table {
    int version = 0, minReader = 0, flags = 0;
    std::vector<TableEntry> entries;
};

/** The file's content; what writePsim takes and readPsim returns. Absent sections are empty optionals. */
struct File {
    json::Value info;  // completed on writing with format, created and bounds
    std::optional<Bytes> thumb;  // JPEG
    std::optional<Geometry> geometry;
    std::optional<Bytes> cad;
    std::optional<json::Value> setup;
    std::optional<Arrays> rfea, rair;
    std::optional<json::Value> view;
    // filled by the reader
    int version = 0, minReader = 0;
    std::vector<TableEntry> table;
    std::vector<std::string> skipped;
    size_t size = 0;
};

struct WriteOptions {
    bool quantised = true;  // geometry mode
    std::optional<std::string> created;  // ISO time; falls back to info.created, then null
};

enum Want : unsigned { WantInfo = 1, WantThumb = 2, WantGeometry = 4, WantCad = 8, WantSetup = 16, WantRfea = 32, WantRair = 64, WantView = 128, WantAll = 255 };

Bytes writePsim(const File& content, const WriteOptions& opt = {});
/** Checks the container (header, table, trailer, CRCs, limits). Does not decompress. */
Table readTable(const uint8_t* p, size_t n);
/** A section's uncompressed bytes, checked against its CRC-32. */
Bytes readSection(const uint8_t* p, size_t n, const TableEntry& e);
/** Reads the INFO section and the thumbnail only. */
File inspectPsim(const uint8_t* p, size_t n);
/** Reads the sections in `want`. */
File readPsim(const uint8_t* p, size_t n, unsigned want = WantAll);

inline Table readTable(const Bytes& b) { return readTable(b.data(), b.size()); }
inline File inspectPsim(const Bytes& b) { return inspectPsim(b.data(), b.size()); }
inline File readPsim(const Bytes& b, unsigned want = WantAll) { return readPsim(b.data(), b.size(), want); }

/** The sections in the order they are written. */
extern const char* const kSectionOrder[8];

}  // namespace ps::psim
