#include "solidworks.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>

#include "../util/xml.hpp"
#include "../util/zip.hpp"
#include "materials.hpp"

namespace ps {

namespace {

const uint8_t SIGNATURE[] = {0x14, 0x00, 0x06, 0x00, 0x08, 0x00};
constexpr size_t HEADER = 26;
const uint8_t OLE_MAGIC[] = {0xd0, 0xcf, 0x11, 0xe0, 0xa1, 0xb1, 0x1a, 0xe1};
const char* MESH_BLOCKS[] = {"Contents/DisplayLists", "FaceTessellations"};
// strip table: six descriptors (element size, kind, 2, count)
const uint32_t DESCRIPTORS[6][2] = {{4, 8}, {12, 100}, {12, 100}, {4, 8}, {4, 8}, {1, 8}};
constexpr uint32_t MAX_STRIPS = 250000, MAX_STRIP_VERTS = 1000000, MAX_DESC_COUNT = 2000000, MAX_COMPONENTS = 2000, MAX_FACE_TABLES = 250000;
constexpr double METERS_TO_MM = 1000;
const uint8_t STRING_MARKER[] = {0xff, 0xfe, 0xff};
const uint8_t TABLE_SIGNATURE[] = {4, 0, 0, 0, 8, 0, 0, 0, 2, 0, 0, 0};
const char* EXPORT_HINT = "In SolidWorks use File > Save As > STEP AP242 (*.step) and open that instead.";

uint32_t u32(const Bytes& b, size_t at) {
    uint32_t v;
    std::memcpy(&v, &b[at], 4);
    return v;
}

bool startsWith(const Bytes& b, const uint8_t* magic, size_t n, size_t at = 0) {
    if (at + n > b.size()) return false;
    return std::memcmp(&b[at], magic, n) == 0;
}

int64_t indexOf(const Bytes& b, const uint8_t* pat, size_t n, size_t from = 0, size_t to = SIZE_MAX) {
    to = std::min(to, b.size());
    if (to < n || from > to - n) return -1;
    const uint8_t* base = b.data();
    const uint8_t* p = base + from;
    const uint8_t* end = base + (to - n) + 1;
    while (p < end) {
        p = static_cast<const uint8_t*>(std::memchr(p, pat[0], size_t(end - p)));
        if (!p) return -1;
        if (std::memcmp(p, pat, n) == 0) return int64_t(p - base);
        p++;
    }
    return -1;
}

struct Block {
    std::string name;
    uint32_t typeId, crc, compressed, expanded;
    size_t dataOffset;
};

std::vector<Block> readBlocks(const Bytes& b) {
    std::vector<Block> out;
    size_t pos = 8;
    for (;;) {
        const int64_t at = indexOf(b, SIGNATURE, 6, pos);
        if (at < 0 || size_t(at) + HEADER > b.size()) break;
        pos = size_t(at) + 1;
        Block k;
        k.typeId = u32(b, at + 6);
        k.crc = u32(b, at + 10);
        k.compressed = u32(b, at + 14);
        k.expanded = u32(b, at + 18);
        const uint32_t nameLen = u32(b, at + 22);
        // a cache-cell index grid reuses the marker, encoding one value L as 2L, L/2 and L
        if (k.crc == uint64_t(k.expanded) * 2 && k.compressed == k.expanded / 2) continue;
        if (!k.compressed || nameLen < 1 || nameLen > 1024) continue;
        if (size_t(at) + HEADER + nameLen + k.compressed > b.size()) continue;
        bool printable = true;
        for (uint32_t i = 0; i < nameLen; i++) {
            const uint8_t x = b[at + HEADER + i];
            const uint8_t c = uint8_t(((x << 4) | (x >> 4)) & 0xff);  // undo the nibble swap
            if (c < 0x20 || c > 0x7e) { printable = false; break; }
            k.name += char(c);
        }
        if (!printable) continue;
        k.dataOffset = size_t(at) + HEADER + nameLen;
        out.push_back(std::move(k));
    }
    return out;
}

Bytes inflateBlock(const Bytes& b, const Block& k) {
    const uint8_t* raw = &b[k.dataOffset];
    Bytes data = inflateRaw(raw, k.compressed, k.expanded);
    if (data.empty()) data = inflateZlib(raw, k.compressed, k.expanded);
    if (data.empty()) data.assign(raw, raw + k.compressed);  // stored
    if (data.size() != k.expanded || crc32(data.data(), data.size()) != k.crc)
        throw std::runtime_error("The SolidWorks block \"" + k.name + "\" is corrupt (checksum mismatch).");
    return data;
}

const Block* findBlock(const std::vector<Block>& blocks, const std::string& name) {
    for (const auto& k : blocks)
        if (k.name == name) return &k;
    return nullptr;
}

struct Table {
    size_t start, end, positions;
    std::vector<uint32_t> lengths;
};

std::optional<Table> readTable(const Bytes& d, size_t at) {
    size_t pos = at;
    size_t descStart[6];
    uint32_t descCount[6];
    for (int k = 0; k < 6; k++) {
        if (pos + 16 > d.size()) return std::nullopt;
        const uint32_t size = u32(d, pos), kind = u32(d, pos + 4), two = u32(d, pos + 8), count = u32(d, pos + 12);
        if (size != DESCRIPTORS[k][0] || kind != DESCRIPTORS[k][1] || two != 2) return std::nullopt;
        if (count > MAX_DESC_COUNT || pos + 16 + size_t(count) * size > d.size()) return std::nullopt;
        descStart[k] = pos + 16;
        descCount[k] = count;
        pos += 16 + size_t(size) * count;
    }
    const uint32_t strips = descCount[0];
    if (!strips || strips > MAX_STRIPS) return std::nullopt;
    Table t;
    t.lengths.resize(strips);
    uint64_t total = 0;
    for (uint32_t j = 0; j < strips; j++) {
        const uint32_t n = u32(d, descStart[0] + 4 * size_t(j));
        if (n < 3 || n > MAX_STRIP_VERTS) return std::nullopt;
        t.lengths[j] = n;
        total += n;
    }
    if (descCount[1] != total || (descCount[2] != 0 && descCount[2] != total) || descCount[4] != strips) return std::nullopt;
    for (uint32_t j = 0; j < strips; j++)
        if (u32(d, descStart[4] + 4 * size_t(j)) != 2 * t.lengths[j] - 2) return std::nullopt;
    t.start = at;
    t.end = pos;
    t.positions = descStart[1];
    return t;
}

std::vector<Table> tablesIn(const Bytes& d, size_t from = 0, size_t to = SIZE_MAX, size_t limit = SIZE_MAX) {
    std::vector<Table> out;
    to = std::min(to, d.size());
    size_t pos = from;
    while (out.size() < limit) {
        const int64_t at = indexOf(d, TABLE_SIGNATURE, 12, pos, to);
        if (at < 0) break;
        auto t = readTable(d, size_t(at));
        if (!t || t->end > to) { pos = size_t(at) + 1; continue; }
        pos = t->end;
        out.push_back(std::move(*t));
    }
    return out;
}

struct Out {
    std::vector<float> positions;
    std::vector<int32_t> faces;
};

using Mat = std::array<double, 16>;  // row-vector 4x4, translation in metres (m[12..14])

void emitTable(const Bytes& d, const Table& t, int face, Out& out, const Mat* m = nullptr, bool mirrored = false) {
    auto read = [&](size_t i, double p[3]) {
        float xyz[3];
        std::memcpy(xyz, &d[t.positions + 12 * i], 12);
        if (m) {
            const Mat& M = *m;
            p[0] = METERS_TO_MM * (xyz[0] * M[0] + xyz[1] * M[4] + xyz[2] * M[8] + M[12]);
            p[1] = METERS_TO_MM * (xyz[0] * M[1] + xyz[1] * M[5] + xyz[2] * M[9] + M[13]);
            p[2] = METERS_TO_MM * (xyz[0] * M[2] + xyz[1] * M[6] + xyz[2] * M[10] + M[14]);
        } else
            for (int k = 0; k < 3; k++) p[k] = METERS_TO_MM * xyz[k];
    };
    double a[3], b[3], c[3];
    size_t at = 0;
    for (uint32_t n : t.lengths) {
        for (uint32_t k = 0; k + 2 < n; k++) {
            read(at + k, a);
            read(at + k + 1, b);
            read(at + k + 2, c);
            // strips alternate winding; a mirroring placement flips it once more
            const bool flip = (k % 2 == 1) != mirrored;
            const double* second = flip ? c : b;
            const double* third = flip ? b : c;
            for (const double* p : {static_cast<const double*>(a), second, third})
                for (int q = 0; q < 3; q++) out.positions.push_back(float(p[q]));
            out.faces.push_back(face);
        }
        at += n;
    }
}

std::array<double, 6> tablesBBox(const Bytes& d, const std::vector<Table>& tables) {
    std::array<double, 6> box = {1e300, 1e300, 1e300, -1e300, -1e300, -1e300};
    for (const auto& t : tables) {
        size_t n = 0;
        for (auto l : t.lengths) n += l;
        for (size_t i = 0; i < n; i++)
            for (int k = 0; k < 3; k++) {
                float v;
                std::memcpy(&v, &d[t.positions + 12 * i + 4 * k], 4);
                box[k] = std::min(box[k], double(v));
                box[k + 3] = std::max(box[k + 3], double(v));
            }
    }
    return box;
}

std::vector<double> numbers(const std::string& s) {
    std::vector<double> v;
    std::istringstream in(s);
    std::string tok;
    while (in >> tok) {
        char* end;
        const double x = std::strtod(tok.c_str(), &end);
        v.push_back(*end ? NAN : x);
    }
    return v;
}

// does a part file's saved mesh have the extent the assembly expects? (multi-configuration parts)
bool matchesBoundingBox(const std::array<double, 6>& box, const std::string& expected) {
    const auto e = numbers(expected);
    if (e.size() != 6 || std::any_of(e.begin(), e.end(), [](double x) { return !std::isfinite(x); })) return true;
    for (int d = 0; d < 3; d++) {
        const double tol = std::max(0.0005, 0.03 * std::max(e[d + 3] - e[d], box[d + 3] - box[d]));
        if (std::abs(box[d] - e[d]) > tol || std::abs(box[d + 3] - e[d + 3]) > tol) return false;
    }
    return true;
}

struct SavedMesh {
    Bytes data;
    std::vector<Table> tables;
    std::array<double, 6> box{};
    int configurations = 1;
};

std::optional<SavedMesh> savedMeshTables(const Bytes& b, const std::vector<Block>& blocks) {
    for (const char* name : MESH_BLOCKS) {
        const Block* k = findBlock(blocks, name);
        if (!k) continue;
        SavedMesh m;
        m.data = inflateBlock(b, *k);
        m.tables = tablesIn(m.data);
        if (!m.tables.empty()) return m;
    }
    return std::nullopt;
}

std::string keywordsXML(const Bytes& b, const std::vector<Block>& blocks) {
    const Block* k = findBlock(blocks, "swXmlContents/KeyWords");
    if (!k) return "";
    try {
        const Bytes d = inflateBlock(b, *k);
        return std::string(d.begin(), d.end());
    } catch (...) {
        return "";
    }
}

int configurationCount(const Bytes& b, const std::vector<Block>& blocks) {
    const std::string text = keywordsXML(b, blocks);
    int n = 0;
    for (size_t p = 0; (p = text.find("Type=\"ConfigurationManager\"", p)) != std::string::npos; p++) n++;
    return std::max(1, n);
}

std::string readMaterialName(const Bytes& b, const std::vector<Block>& blocks) {
    const std::string text = keywordsXML(b, blocks);
    if (text.empty()) return "";
    std::smatch m;
    static const std::regex a("Type=\"ConfigurationManager\"[^>]*?\\sMaterial=\"([^\"]*)\"");
    static const std::regex c("\\sMaterial=\"([^\"]*)\"[^>]*Type=\"ConfigurationManager\"");
    if (!std::regex_search(text, m, a) && !std::regex_search(text, m, c)) return "";
    std::string name = xml::decodeEntities(m[1].str());
    while (!name.empty() && std::isspace((unsigned char)name.back())) name.pop_back();
    while (!name.empty() && std::isspace((unsigned char)name.front())) name.erase(name.begin());
    std::string low = name;
    for (auto& ch : low) ch = char(std::tolower((unsigned char)ch));
    return low.find("not specified") != std::string::npos ? "" : name;
}

// ---------- assemblies ----------

struct DirEntry {
    std::string name, alias, id, chunk;
    uint32_t index = 0;
};

std::vector<DirEntry> readDirectory(const Bytes& d) {
    size_t at = 0;
    auto need = [&](size_t n) {
        if (at + n > d.size()) throw std::runtime_error("The assembly mesh directory is truncated.");
        const size_t s = at;
        at += n;
        return s;
    };
    auto rd32 = [&]() { return u32(d, need(4)); };
    auto key = [&]() {
        const size_t s = need(8);
        static const char* hex = "0123456789abcdef";
        std::string k;
        for (size_t i = s; i < s + 8; i++) { k += hex[d[i] >> 4]; k += hex[d[i] & 15]; }
        return k;
    };
    auto atString = [&]() { return startsWith(d, STRING_MARKER, 3, at); };
    auto str = [&]() {
        if (!atString()) throw std::runtime_error("Unsupported string layout in the assembly mesh directory.");
        need(3);
        uint32_t n = d[need(1)];
        if (n == 0xff) {
            n = d[at] | (d[at + 1] << 8);
            need(2);
            if (n == 0xffff) n = rd32();
        }
        if (n > 4096) throw std::runtime_error("A component name in the assembly is too long.");
        const size_t s = need(2 * size_t(n));
        std::string out;
        for (uint32_t i = 0; i < n; i++) {
            const uint16_t ch = uint16_t(d[s + 2 * i] | (d[s + 2 * i + 1] << 8));
            if (ch < 0x80) out += char(ch);
            else if (ch < 0x800) { out += char(0xc0 | (ch >> 6)); out += char(0x80 | (ch & 0x3f)); }
            else { out += char(0xe0 | (ch >> 12)); out += char(0x80 | ((ch >> 6) & 0x3f)); out += char(0x80 | (ch & 0x3f)); }
        }
        return out;
    };
    if (rd32() != 2) throw std::runtime_error("Unsupported assembly mesh directory version.");
    const uint32_t capacity = rd32(), count = rd32();
    if (!capacity || capacity > 100000 || !count || count > MAX_COMPONENTS) throw std::runtime_error("The assembly exceeds the component limits.");
    // entries are found by their own signature: a sequence number followed by a string marker
    const int64_t firstMarker = indexOf(d, STRING_MARKER, 3, 12);
    if (firstMarker < 4) return {};
    const uint32_t base = u32(d, size_t(firstMarker) - 4);
    std::vector<DirEntry> entries;
    for (uint32_t i = 0; i < count; i++) {
        uint8_t sig[7];
        const uint32_t seq = base + i;
        std::memcpy(sig, &seq, 4);
        std::memcpy(sig + 4, STRING_MARKER, 3);
        const int64_t start = indexOf(d, sig, 7, i ? at - 4 : size_t(firstMarker) - 4);
        if (start < 0) break;
        at = size_t(start) + 4;
        DirEntry e;
        e.name = str();
        rd32(); rd32(); rd32();  // stamp, configuration, reserved
        e.alias = key();
        e.id = key();
        for (int k = 0; k < 4 && !atString() && at + 8 <= d.size(); k++) at += 8;  // writer-dependent padding
        e.chunk = str();
        e.index = rd32();
        entries.push_back(std::move(e));
    }
    return entries;
}

std::optional<std::pair<Mat, bool>> rigidTransform(const std::string& text) {
    const auto v = numbers(text);
    if (v.size() != 16 || std::any_of(v.begin(), v.end(), [](double x) { return !std::isfinite(x); })) return std::nullopt;
    for (int i : {3, 7, 11})
        if (std::abs(v[i]) > 1e-12) return std::nullopt;
    if (std::abs(v[15] - 1) > 1e-12) return std::nullopt;
    const double r[3][3] = {{v[0], v[1], v[2]}, {v[4], v[5], v[6]}, {v[8], v[9], v[10]}};
    auto dot = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    const double det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1]) - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0]) +
                       r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    const double tol = 1e-6;
    for (auto& row : r)
        if (std::abs(dot(row, row) - 1) > tol) return std::nullopt;
    if (std::abs(dot(r[0], r[1])) > tol || std::abs(dot(r[0], r[2])) > tol || std::abs(dot(r[1], r[2])) > tol || std::abs(std::abs(det) - 1) > tol)
        return std::nullopt;
    Mat m;
    std::copy(v.begin(), v.end(), m.begin());
    return std::make_pair(m, det < 0);
}

Mat compose(const Mat& child, const Mat& parent) {
    Mat out{};
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++)
            for (int k = 0; k < 4; k++) out[i * 4 + j] += child[i * 4 + k] * parent[k * 4 + j];
    return out;
}

const Mat IDENTITY = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};

struct Stored {
    std::shared_ptr<Bytes> data;
    std::vector<Table> tables;
};

std::map<std::string, Stored> componentMeshes(const Bytes& b, const std::vector<Block>& blocks, const std::vector<DirEntry>& entries) {
    std::map<std::string, std::vector<const DirEntry*>> byChunk;
    for (const auto& e : entries)
        if (!e.chunk.empty()) byChunk[e.chunk].push_back(&e);
    std::map<std::string, Stored> out;
    for (const auto& [chunk, group] : byChunk) {
        const Block* k = findBlock(blocks, "FaceTessellations/" + chunk);
        if (!k) continue;
        auto data = std::make_shared<Bytes>(inflateBlock(b, *k));
        const Bytes& d = *data;
        struct Rec { size_t start; uint32_t count; const DirEntry* entry; };
        std::vector<Rec> records;
        for (const DirEntry* e : group) {
            uint8_t marker[12];
            std::memcpy(marker, &e->index, 4);
            for (int i = 0; i < 8; i++) marker[4 + i] = uint8_t(std::stoi(e->id.substr(2 * i, 2), nullptr, 16));
            const int64_t start = indexOf(d, marker, 12);
            if (start < 0 || size_t(start) + 16 > d.size() || indexOf(d, marker, 12, size_t(start) + 1) >= 0) continue;
            const uint32_t count = u32(d, size_t(start) + 12);
            if (!count || count > MAX_FACE_TABLES) continue;
            records.push_back({size_t(start), count, e});
        }
        std::sort(records.begin(), records.end(), [](const Rec& a, const Rec& c) { return a.start < c.start; });
        for (size_t i = 0; i < records.size(); i++) {
            // a record owns exactly `count` tables; never read into the next record
            const size_t end = i + 1 < records.size() ? records[i + 1].start : d.size();
            auto tables = tablesIn(d, records[i].start + 16, end, records[i].count);
            if (tables.size() == records[i].count) out[records[i].entry->id] = {data, std::move(tables)};
        }
    }
    return out;
}

std::string baseName(std::string p) {
    std::replace(p.begin(), p.end(), '\\', '/');
    const size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}

std::string lowerCopy(std::string s) {
    for (auto& c : s) c = char(std::tolower((unsigned char)c));
    return s;
}

struct Report {
    int references = 0, placed = 0, fromParts = 0, suppressed = 0, hidden = 0;
    std::vector<std::string> missing, otherConfig;
};

Out readAssembly(const Bytes& b, const std::vector<Block>& blocks, const SiblingReader& resolvePart, Report& report) {
    const Block* treeBlock = findBlock(blocks, "swXmlContents/COMPINSTANCETREE");
    const Block* dirBlock = findBlock(blocks, "FaceTessellations/Directory");
    if (!treeBlock) throw std::runtime_error("This assembly has no component tree. Export it from SolidWorks as STEP instead.");
    const Bytes treeData = inflateBlock(b, *treeBlock);
    auto root = xml::parse(std::string(treeData.begin(), treeData.end()));
    const xml::Node* doc = root.get();
    for (const auto& c : root->children)
        if (!c->children.empty()) { doc = c.get(); break; }
    auto one = [](const xml::Node* parent, const std::string& tag) {
        auto found = parent->childrenNamed(tag);
        if (found.size() != 1) throw std::runtime_error("The assembly has no unambiguous " + tag + " record.");
        return found[0];
    };
    std::map<std::string, const xml::Node*> files, models;
    for (const xml::Node* f : one(doc, "swHeader")->childrenNamed("swFile")) files[f->attr("id")] = f;
    for (const xml::Node* m : one(doc, "swModelList")->childrenNamed("swModel")) models[m->attr("id")] = m;
    auto configs = one(doc, "swConfigurationList")->childrenNamed("swConfiguration");
    if (configs.empty()) throw std::runtime_error("The assembly has no saved configuration.");
    const xml::Node* chosen = configs[0];
    for (const xml::Node* c : configs)
        if (c->attr("swMostRecentConfiguration") == "YES") { chosen = c; break; }
    auto docType = [&](const xml::Node* model) {
        auto f = files.find(model->attr("swFileRef"));
        return f == files.end() ? std::string() : f->second->attr("swDocType");
    };
    auto rootIt = models.find(chosen->attr("swModelRef"));
    if (rootIt == models.end() || docType(rootIt->second) != "ASSEMBLY") throw std::runtime_error("The assembly configuration could not be resolved.");

    const auto entries = dirBlock ? readDirectory(inflateBlock(b, *dirBlock)) : std::vector<DirEntry>{};
    std::map<std::string, const DirEntry*> byName, byId;
    for (const auto& e : entries) { byName[e.name] = &e; byId[e.id] = &e; }
    const auto stored = componentMeshes(b, blocks, entries);
    std::map<std::string, std::optional<SavedMesh>> partCache;
    Out out;
    int face = 0;

    auto fromPartFile = [&](const xml::Node* model, bool& otherConfig) -> const SavedMesh* {
        otherConfig = false;
        auto f = files.find(model->attr("swFileRef"));
        const std::string fileName = f == files.end() ? "" : baseName(f->second->attr("swPath"));
        if (fileName.empty() || !resolvePart) return nullptr;
        auto it = partCache.find(fileName);
        if (it == partCache.end()) {
            std::optional<SavedMesh> mesh;
            try {
                const Bytes data = resolvePart(fileName);
                if (!data.empty()) {
                    const auto partBlocks = readBlocks(data);
                    mesh = savedMeshTables(data, partBlocks);
                    if (mesh) {
                        mesh->box = tablesBBox(mesh->data, mesh->tables);
                        mesh->configurations = configurationCount(data, partBlocks);
                    }
                }
            } catch (...) {
                mesh.reset();
            }
            it = partCache.emplace(fileName, std::move(mesh)).first;
        }
        if (!it->second) return nullptr;
        if (it->second->configurations > 1 && !matchesBoundingBox(it->second->box, model->attr("swBoundingBox"))) { otherConfig = true; return nullptr; }
        return &*it->second;
    };

    std::function<void(const xml::Node*, const std::string&, const Mat&, int)> walk = [&](const xml::Node* model, const std::string& path, const Mat& matrix, int depth) {
        if (depth > 8) return;
        for (const xml::Node* ref : model->childrenNamed("swReference")) {
            report.references++;
            if (ref->attr("swSuppressed") == "YES") { report.suppressed++; continue; }
            auto childIt = models.find(ref->attr("swModelRef"));
            const std::string nameA = ref->attr("swName");
            const std::string step = nameA + "-" + ref->attr("swReferenceNumber") + "@" + model->attr("swName");
            const std::string full = path.empty() ? step : path + "/" + step;
            const auto placement = childIt == models.end() ? std::nullopt : rigidTransform(ref->attr("swTransform"));
            if (childIt == models.end() || !placement) { report.missing.push_back(nameA.empty() ? full : nameA); continue; }
            const xml::Node* child = childIt->second;
            const Mat world = compose(placement->first, matrix);
            if (docType(child) == "ASSEMBLY") { walk(child, full, world, depth + 1); continue; }
            if (ref->attr("swHidden") == "YES") { report.hidden++; continue; }
            const Bytes* data = nullptr;
            const std::vector<Table>* tables = nullptr;
            auto target = byName.count(full) ? byName[full] : nullptr;
            std::set<std::string> seen;
            while (target && target->chunk.empty() && !seen.count(target->id)) {
                seen.insert(target->id);
                target = byId.count(target->alias) ? byId[target->alias] : nullptr;  // repeated components alias one stored mesh
            }
            if (target) {
                auto s = stored.find(target->id);
                if (s != stored.end()) { data = s->second.data.get(); tables = &s->second.tables; }
            }
            if (!tables) {
                bool other = false;
                const SavedMesh* mesh = fromPartFile(child, other);
                if (other) {
                    const std::string cfg = child->attr("swConfigurationName");
                    report.otherConfig.push_back(nameA + (cfg.empty() ? "" : " (" + cfg + ")"));
                    continue;
                }
                if (mesh) { data = &mesh->data; tables = &mesh->tables; report.fromParts++; }
            }
            if (!tables) { report.missing.push_back(nameA.empty() ? full : nameA); continue; }
            const Mat& w = world;
            const double det = w[0] * (w[5] * w[10] - w[6] * w[9]) - w[1] * (w[4] * w[10] - w[6] * w[8]) + w[2] * (w[4] * w[9] - w[5] * w[8]);
            for (const auto& t : *tables) emitTable(*data, t, face++, out, &world, det < 0);
            report.placed++;
        }
    };
    walk(rootIt->second, "", IDENTITY, 0);
    if (out.faces.empty())
        throw std::runtime_error("None of the assembly components has a saved mesh. Open the assembly from the folder that holds its part files, or export the assembly as STEP.");
    return out;
}

}  // namespace

std::vector<std::string> assemblyPartNames(const Bytes& b) {
    if (startsWith(b, OLE_MAGIC, 8)) return {};
    const auto blocks = readBlocks(b);
    const Block* k = findBlock(blocks, "swXmlContents/COMPINSTANCETREE");
    if (!k) return {};
    const Bytes d = inflateBlock(b, *k);
    auto root = xml::parse(std::string(d.begin(), d.end()));
    const xml::Node* doc = root.get();
    for (const auto& c : root->children)
        if (!c->children.empty()) { doc = c.get(); break; }
    std::vector<std::string> names;
    for (const xml::Node* h : doc->childrenNamed("swHeader"))
        for (const xml::Node* f : h->childrenNamed("swFile"))
            if (f->attr("swDocType") == "PART") {
                const std::string n = baseName(f->attr("swPath"));
                if (!n.empty() && std::find(names.begin(), names.end(), n) == names.end()) names.push_back(n);
            }
    return names;
}

MeshSource readSolidWorks(const Bytes& b, const std::string& fileName, const SiblingReader& readSibling) {
    const std::string low = lowerCopy(fileName);
    auto ends = [&](const char* s) { const size_t n = std::strlen(s); return low.size() >= n && low.compare(low.size() - n, n, s) == 0; };
    const bool assembly = ends(".sldasm");
    if (ends(".slddrw")) throw std::runtime_error("SolidWorks drawings (.SLDDRW) hold 2D sheets, not the 3D model. Open the part (.SLDPRT) or assembly (.SLDASM) instead.");
    if (startsWith(b, OLE_MAGIC, 8))
        throw std::runtime_error(std::string("This file uses the pre-2015 SolidWorks format, which Parts Sim cannot read. Open and save it in SolidWorks 2015 or newer, or export it. ") + EXPORT_HINT);
    const auto blocks = readBlocks(b);
    if (blocks.empty()) throw std::runtime_error("This does not look like a SolidWorks file (no SolidWorks data blocks were found).");
    MeshSource m;
    m.units = "mm";
    Out out;
    if (assembly) {
        // parts are found by name in the assembly's folder (case-insensitively on case-sensitive disks)
        SiblingReader resolve = readSibling;
        Report report;
        out = readAssembly(b, blocks, resolve, report);
        const int total = report.placed + int(report.missing.size() + report.otherConfig.size());
        m.info = "SolidWorks assembly, " + std::to_string(report.placed) + " of " + std::to_string(total) + " components";
        if (report.fromParts) m.info += " (" + std::to_string(report.fromParts) + " from part files)";
        auto listNames = [](std::vector<std::string> v, size_t n) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
            std::string s;
            for (size_t i = 0; i < std::min(n, v.size()); i++) s += (i ? ", " : "") + v[i];
            if (v.size() > n) s += ", ...";
            return s;
        };
        if (!report.otherConfig.empty())
            m.warnings.push_back(std::to_string(report.otherConfig.size()) +
                                 " component(s) were left out because their part file is saved in a different configuration (size) than this assembly uses - usually Toolbox fasteners: " +
                                 listNames(report.otherConfig, 3) + ".");
        if (!report.missing.empty())
            m.warnings.push_back(std::to_string(report.missing.size()) + " component(s) have no saved mesh (" + listNames(report.missing, 4) +
                                 "). Add their .SLDPRT files to the assembly's folder, or export the assembly as STEP.");
    } else {
        auto mesh = savedMeshTables(b, blocks);
        if (!mesh)
            throw std::runtime_error(std::string("This SolidWorks part was saved without display data, so its geometry cannot be read without SolidWorks. "
                                                 "Re-save it with Options > Document Properties > Image Quality > \"Save tessellation with part document\" turned on, or: ") +
                                     EXPORT_HINT);
        for (size_t i = 0; i < mesh->tables.size(); i++) emitTable(mesh->data, mesh->tables[i], int(i), out);
        m.material = readMaterialName(b, blocks);
        m.info = "SolidWorks part, " + std::to_string(mesh->tables.size()) + " faces (saved display mesh)";
    }
    m.positions = std::move(out.positions);
    m.faceIds = std::move(out.faces);
    return m;
}

std::string matchMaterial(const std::string& swName) {
    if (swName.empty()) return "";
    const std::string s = lowerCopy(swName);
    static const std::vector<std::pair<std::regex, std::string>> rules = {
        {std::regex("ti-?6al-?4v|titanium"), "ti-64"},
        {std::regex("7075"), "al-7075"},
        {std::regex("6061|6063|alumin(i)?um|al\\s?alloy|1060|5052"), "al-6061"},
        {std::regex("304|316|stainless"), "ss-304"},
        {std::regex("cast.*iron|grey iron|gray iron"), "cast-iron"},
        {std::regex("4140|4340|alloy steel|chrome|cr-?v"), "steel-alloy"},
        {std::regex("1020|1018|1045|carbon steel|plain.*steel|steel"), "steel-1020"},
        {std::regex("brass"), "brass"},
        {std::regex("copper"), "copper"},
        {std::regex("petg"), "petg"},
        {std::regex("\\bpla\\b"), "pla"},
        {std::regex("\\babs\\b"), "abs"},
        {std::regex("nylon|\\bpa ?6|\\bpa ?66|polyamide"), "nylon"},
        {std::regex("polycarbonate|\\bpc\\b"), "pc"},
        {std::regex("carbon fib|cfrp"), "cfrp"},
        {std::regex("glass"), "glass"},
        {std::regex("wood|pine|oak|balsa"), "wood"},
    };
    for (const auto& [re, id] : rules)
        if (std::regex_search(s, re) && findMaterial(id)) return id;
    return "";
}

}  // namespace ps
