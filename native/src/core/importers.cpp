#include "importers.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "../util/json.hpp"
#include "../util/xml.hpp"
#include "../util/zip.hpp"
#include "solidworks.hpp"

namespace ps {

namespace {

std::string lower(std::string s) {
    for (auto& c : s) c = char(std::tolower((unsigned char)c));
    return s;
}

std::string extensionOf(const std::string& name) {
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? "" : lower(name.substr(dot + 1));
}

std::string stemOf(const std::string& path) {
    const size_t slash = path.find_last_of("/\\");
    std::string file = slash == std::string::npos ? path : path.substr(slash + 1);
    const size_t dot = file.find_last_of('.');
    return dot == std::string::npos ? file : file.substr(0, dot);
}

template <class T>
T readLE(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

}  // namespace

const std::vector<std::string>& supportedExtensions() {
    static const std::vector<std::string> ext = {"step", "stp", "iges", "igs", "brep", "brp", "sldprt", "sldasm", "slddrw",
                                                 "stl", "obj", "3mf", "ply", "glb", "gltf"};
    return ext;
}

MeshSource importFile(const std::string& path, const SiblingReader& readSibling) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Could not open the file.");
    Bytes data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t slash = path.find_last_of("/\\");
    return importBytes(slash == std::string::npos ? path : path.substr(slash + 1), data, readSibling);
}

MeshSource importBytes(const std::string& fileName, const Bytes& data, const SiblingReader& readSibling) {
    if (data.empty()) throw std::runtime_error("The selected file is empty.");
    const std::string ext = extensionOf(fileName);
    MeshSource m;
    if (ext == "step" || ext == "stp") m = readCAD("step", data);
    else if (ext == "iges" || ext == "igs") m = readCAD("iges", data);
    else if (ext == "brep" || ext == "brp") m = readCAD("brep", data);
    else if (ext == "sldprt" || ext == "sldasm" || ext == "slddrw") m = readSolidWorks(data, fileName, readSibling);
    else if (ext == "stl") m = readSTL(data);
    else if (ext == "obj") m = readOBJ(data);
    else if (ext == "ply") m = readPLY(data);
    else if (ext == "3mf") m = read3MF(data);
    else if (ext == "glb" || ext == "gltf") m = readGLB(data);
    else throw std::runtime_error("Unsupported file type \"." + ext + "\". Use STEP, IGES, SolidWorks (SLDPRT/SLDASM), STL, OBJ, 3MF, PLY or GLB.");
    m.name = stemOf(fileName);
    return m;
}

// ---------- STL ----------

MeshSource readSTL(const Bytes& d) {
    MeshSource m;
    m.info = "STL mesh";
    // binary when the size matches the triangle count (ASCII files may also start with "solid")
    if (d.size() >= 84) {
        const uint32_t n = readLE<uint32_t>(&d[80]);
        if (84 + size_t(n) * 50 == d.size()) {
            m.positions.resize(size_t(n) * 9);
            for (uint32_t t = 0; t < n; t++)
                for (int k = 0; k < 9; k++) m.positions[9 * size_t(t) + k] = readLE<float>(&d[84 + 50 * size_t(t) + 12 + 4 * k]);
            if (m.positions.empty()) throw std::runtime_error("No meshes found in file.");
            return m;
        }
    }
    const std::string text(d.begin(), d.end());
    std::istringstream in(text);
    std::string word;
    while (in >> word) {
        if (lower(word) == "vertex") {
            float x, y, z;
            if (!(in >> x >> y >> z)) throw std::runtime_error("The ASCII STL file has a malformed vertex.");
            m.positions.insert(m.positions.end(), {x, y, z});
        }
    }
    if (m.positions.empty()) throw std::runtime_error("No meshes found in file.");
    if (m.positions.size() % 9) throw std::runtime_error("The mesh does not contain complete triangles.");
    return m;
}

// ---------- OBJ ----------

MeshSource readOBJ(const Bytes& d) {
    MeshSource m;
    m.info = "OBJ mesh";
    std::vector<float> V;
    const std::string text(d.begin(), d.end());
    std::istringstream in(text);
    std::string line;
    std::vector<long> poly;
    while (std::getline(in, line)) {
        if (line.size() < 2) continue;
        if (line[0] == 'v' && (line[1] == ' ' || line[1] == '\t')) {
            std::istringstream ls(line.substr(2));
            float x = 0, y = 0, z = 0;
            ls >> x >> y >> z;
            V.insert(V.end(), {x, y, z});
        } else if (line[0] == 'f' && (line[1] == ' ' || line[1] == '\t')) {
            std::istringstream ls(line.substr(2));
            std::string tok;
            poly.clear();
            while (ls >> tok) {
                const long idx = std::strtol(tok.c_str(), nullptr, 10);
                const long nv = long(V.size() / 3);
                const long i = idx < 0 ? nv + idx : idx - 1;
                if (i < 0 || i >= nv) throw std::runtime_error("The OBJ file references a vertex that does not exist.");
                poly.push_back(i);
            }
            for (size_t k = 1; k + 1 < poly.size(); k++)
                for (long v : {poly[0], poly[k], poly[k + 1]})
                    m.positions.insert(m.positions.end(), {V[3 * v], V[3 * v + 1], V[3 * v + 2]});
        }
    }
    if (m.positions.empty()) throw std::runtime_error("No meshes found in file.");
    return m;
}

// ---------- PLY ----------

MeshSource readPLY(const Bytes& d) {
    MeshSource m;
    m.info = "PLY mesh";
    const std::string head(d.begin(), d.begin() + std::min<size_t>(d.size(), 65536));
    const size_t endHeader = head.find("end_header");
    if (head.compare(0, 3, "ply") != 0 || endHeader == std::string::npos) throw std::runtime_error("This is not a PLY file.");
    size_t body = head.find('\n', endHeader);
    if (body == std::string::npos) throw std::runtime_error("The PLY header is incomplete.");
    body++;
    std::istringstream hs(head.substr(0, endHeader));
    std::string line, format;
    struct Prop { std::string name, type, countType; bool list = false; };
    struct Element { std::string name; size_t count = 0; std::vector<Prop> props; };
    std::vector<Element> elements;
    while (std::getline(hs, line)) {
        std::istringstream ls(line);
        std::string kw;
        ls >> kw;
        if (kw == "format") ls >> format;
        else if (kw == "element") { Element e; ls >> e.name >> e.count; elements.push_back(e); }
        else if (kw == "property" && !elements.empty()) {
            Prop p;
            std::string t;
            ls >> t;
            if (t == "list") { p.list = true; ls >> p.countType >> p.type >> p.name; }
            else { p.type = t; ls >> p.name; }
            elements.back().props.push_back(p);
        }
    }
    const bool ascii = format == "ascii", big = format == "binary_big_endian";
    auto size = [](const std::string& t) -> int {
        if (t == "char" || t == "uchar" || t == "int8" || t == "uint8") return 1;
        if (t == "short" || t == "ushort" || t == "int16" || t == "uint16") return 2;
        if (t == "double" || t == "float64") return 8;
        return 4;
    };
    size_t at = body;
    std::istringstream text(ascii ? std::string(d.begin() + body, d.end()) : std::string());
    auto readNum = [&](const std::string& t) -> double {
        if (ascii) { double v = 0; text >> v; return v; }
        const int n = size(t);
        if (at + n > d.size()) throw std::runtime_error("The PLY file is truncated.");
        uint8_t b[8];
        for (int k = 0; k < n; k++) b[k] = big ? d[at + n - 1 - k] : d[at + k];
        at += n;
        if (t == "char" || t == "int8") return int8_t(b[0]);
        if (t == "uchar" || t == "uint8") return b[0];
        if (t == "short" || t == "int16") return readLE<int16_t>(b);
        if (t == "ushort" || t == "uint16") return readLE<uint16_t>(b);
        if (t == "int" || t == "int32") return readLE<int32_t>(b);
        if (t == "uint" || t == "uint32") return readLE<uint32_t>(b);
        if (t == "double" || t == "float64") return readLE<double>(b);
        return readLE<float>(b);
    };
    std::vector<float> V;
    for (const auto& e : elements) {
        for (size_t r = 0; r < e.count; r++) {
            double xyz[3] = {0, 0, 0};
            std::vector<long> face;
            for (const auto& p : e.props) {
                if (p.list) {
                    const long n = long(readNum(p.countType));
                    if (n < 0 || n > 1000000) throw std::runtime_error("The PLY file has an invalid face.");
                    for (long k = 0; k < n; k++) {
                        const double v = readNum(p.type);
                        if (e.name == "face" && (p.name == "vertex_indices" || p.name == "vertex_index")) face.push_back(long(v));
                    }
                } else {
                    const double v = readNum(p.type);
                    if (e.name == "vertex") {
                        if (p.name == "x") xyz[0] = v;
                        else if (p.name == "y") xyz[1] = v;
                        else if (p.name == "z") xyz[2] = v;
                    }
                }
            }
            if (e.name == "vertex") V.insert(V.end(), {float(xyz[0]), float(xyz[1]), float(xyz[2])});
            else if (e.name == "face") {
                for (size_t k = 1; k + 1 < face.size(); k++)
                    for (long v : {face[0], face[k], face[k + 1]}) {
                        if (v < 0 || size_t(v) >= V.size() / 3) throw std::runtime_error("The PLY file references a vertex that does not exist.");
                        m.positions.insert(m.positions.end(), {V[3 * v], V[3 * v + 1], V[3 * v + 2]});
                    }
            }
        }
    }
    if (m.positions.empty()) throw std::runtime_error("No meshes found in file.");
    return m;
}

// ---------- glTF / GLB ----------

namespace {

using Mat4 = std::array<double, 16>;  // column-major like glTF
Mat4 identity() { return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; }
Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            double s = 0;
            for (int k = 0; k < 4; k++) s += a[k * 4 + rr] * b[c * 4 + k];
            r[c * 4 + rr] = s;
        }
    return r;
}

Mat4 nodeMatrix(const json::Value& n) {
    if (n.has("matrix") && n["matrix"].size() == 16) {
        Mat4 m;
        for (int i = 0; i < 16; i++) m[i] = n["matrix"][i].number();
        return m;
    }
    double t[3] = {0, 0, 0}, q[4] = {0, 0, 0, 1}, s[3] = {1, 1, 1};
    for (int i = 0; i < 3; i++) {
        t[i] = n["translation"][i].number(0);
        s[i] = n["scale"][i].number(1);
    }
    for (int i = 0; i < 4; i++) q[i] = n["rotation"][i].number(i == 3 ? 1 : 0);
    const double x = q[0], y = q[1], z = q[2], w = q[3];
    Mat4 m = {1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w), 0,
              2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w), 0,
              2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y), 0,
              t[0], t[1], t[2], 1};
    for (int c = 0; c < 3; c++)
        for (int r = 0; r < 3; r++) m[c * 4 + r] *= s[c];
    return m;
}

}  // namespace

MeshSource readGLB(const Bytes& d) {
    MeshSource m;
    m.info = "glTF mesh";
    m.units = "m";
    json::Value doc;
    const uint8_t* bin = nullptr;
    size_t binSize = 0;
    if (d.size() >= 20 && readLE<uint32_t>(&d[0]) == 0x46546c67) {
        const uint32_t jsonSize = readLE<uint32_t>(&d[12]);
        if (readLE<uint32_t>(&d[16]) != 0x4e4f534a || 20 + size_t(jsonSize) > d.size()) throw std::runtime_error("The GLB JSON chunk is invalid.");
        doc = json::parse(std::string(d.begin() + 20, d.begin() + 20 + jsonSize));
        size_t at = 20 + jsonSize;
        if (at + 8 <= d.size() && readLE<uint32_t>(&d[at + 4]) == 0x004e4942) {
            binSize = readLE<uint32_t>(&d[at]);
            bin = &d[at + 8];
            if (at + 8 + binSize > d.size()) throw std::runtime_error("The GLB binary chunk is truncated.");
        }
    } else {
        doc = json::parse(std::string(d.begin(), d.end()));
    }
    // only self-contained files: external buffers would need other files (or the network)
    std::vector<std::vector<uint8_t>> buffers;
    for (size_t b = 0; b < doc["buffers"].size(); b++) {
        const auto& buf = doc["buffers"][b];
        if (buf.has("uri")) {
            const std::string& uri = buf["uri"].string();
            const size_t comma = uri.find(',');
            if (uri.rfind("data:", 0) != 0 || uri.find(";base64") == std::string::npos || comma == std::string::npos)
                throw std::runtime_error("External glTF resources are not supported. Export a self-contained GLB or embed its resources.");
            static const std::string B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
            std::vector<uint8_t> out;
            uint32_t acc = 0;
            int bits = 0;
            for (size_t k = comma + 1; k < uri.size(); k++) {
                const size_t p = B64.find(uri[k]);
                if (p == std::string::npos) continue;
                acc = (acc << 6) | uint32_t(p);
                bits += 6;
                if (bits >= 8) { bits -= 8; out.push_back(uint8_t((acc >> bits) & 0xff)); }
            }
            buffers.push_back(std::move(out));
        } else {
            buffers.emplace_back(bin, bin + binSize);
        }
    }
    auto accessor = [&](int index, int& count, int& comps) -> std::vector<double> {
        const auto& acc = doc["accessors"][size_t(index)];
        const auto& view = doc["bufferViews"][size_t(acc["bufferView"].integer(-1))];
        const int buffer = view["buffer"].integer(-1);
        if (buffer < 0 || size_t(buffer) >= buffers.size()) throw std::runtime_error("The glTF file references a missing buffer.");
        const auto& data = buffers[size_t(buffer)];
        const std::string type = acc["type"].string();
        comps = type == "SCALAR" ? 1 : type == "VEC2" ? 2 : type == "VEC3" ? 3 : type == "VEC4" ? 4 : 0;
        count = acc["count"].integer();
        const int ct = acc["componentType"].integer();
        const int cs = ct == 5126 || ct == 5125 ? 4 : ct == 5123 || ct == 5122 ? 2 : 1;
        const size_t stride = view["byteStride"].integer(comps * cs);
        const size_t base = size_t(view["byteOffset"].integer(0)) + size_t(acc["byteOffset"].integer(0));
        if (!comps || count < 0 || base + (count ? (count - 1) * stride + comps * cs : 0) > data.size()) throw std::runtime_error("A glTF accessor is out of range.");
        std::vector<double> out(size_t(count) * comps);
        for (int i = 0; i < count; i++)
            for (int c = 0; c < comps; c++) {
                const uint8_t* p = &data[base + i * stride + c * cs];
                out[size_t(i) * comps + c] = ct == 5126 ? readLE<float>(p) : ct == 5125 ? readLE<uint32_t>(p) : ct == 5123 ? readLE<uint16_t>(p)
                                             : ct == 5122 ? readLE<int16_t>(p) : ct == 5121 ? p[0] : int8_t(p[0]);
            }
        return out;
    };
    std::function<void(int, const Mat4&, int)> visit = [&](int ni, const Mat4& parent, int depth) {
        if (depth > 64) return;
        const auto& node = doc["nodes"][size_t(ni)];
        const Mat4 world = mul(parent, nodeMatrix(node));
        if (node.has("mesh")) {
            const auto& mesh = doc["meshes"][size_t(node["mesh"].integer())];
            const double det = world[0] * (world[5] * world[10] - world[9] * world[6]) - world[4] * (world[1] * world[10] - world[9] * world[2]) +
                               world[8] * (world[1] * world[6] - world[5] * world[2]);
            for (size_t p = 0; p < mesh["primitives"].size(); p++) {
                const auto& prim = mesh["primitives"][p];
                if (prim["mode"].integer(4) != 4 || !prim["attributes"].has("POSITION")) continue;
                int n = 0, c = 0;
                const auto pos = accessor(prim["attributes"]["POSITION"].integer(), n, c);
                if (c != 3) continue;
                std::vector<double> idx;
                if (prim.has("indices")) { int ni2, c2; idx = accessor(prim["indices"].integer(), ni2, c2); }
                const size_t corners = idx.empty() ? size_t(n) : idx.size();
                for (size_t t = 0; t + 2 < corners; t += 3) {
                    const int order[3] = {0, det < 0 ? 2 : 1, det < 0 ? 1 : 2};
                    for (int k : order) {
                        const size_t v = idx.empty() ? t + k : size_t(idx[t + k]);
                        if (v >= size_t(n)) throw std::runtime_error("A glTF index is out of range.");
                        const double x = pos[3 * v], y = pos[3 * v + 1], z = pos[3 * v + 2];
                        for (int r = 0; r < 3; r++) m.positions.push_back(float(world[r] * x + world[4 + r] * y + world[8 + r] * z + world[12 + r]));
                    }
                }
            }
        }
        for (size_t k = 0; k < node["children"].size(); k++) visit(node["children"][k].integer(), world, depth + 1);
    };
    const auto& scenes = doc["scenes"];
    const auto& scene = scenes[size_t(doc["scene"].integer(0))];
    if (scene.has("nodes"))
        for (size_t k = 0; k < scene["nodes"].size(); k++) visit(scene["nodes"][k].integer(), identity(), 0);
    else
        for (size_t k = 0; k < doc["nodes"].size(); k++) visit(int(k), identity(), 0);
    if (m.positions.empty()) throw std::runtime_error("No meshes found in file.");
    return m;
}

// ---------- 3MF ----------

MeshSource read3MF(const Bytes& d) {
    MeshSource m;
    m.info = "3MF mesh";
    m.units = "mm";
    const auto files = unzip(d);
    auto rels = files.find("_rels/.rels");
    if (rels == files.end()) throw std::runtime_error("The 3MF archive is missing its model relationship.");
    auto relDoc = xml::parse(std::string(rels->second.begin(), rels->second.end()));
    std::string target;
    std::function<void(const xml::Node&)> findRel = [&](const xml::Node& n) {
        if (n.tag == "Relationship") {
            const std::string t = n.attr("Target");
            if (t.size() > 6 && lower(t.substr(t.size() - 6)) == ".model" && target.empty()) target = t[0] == '/' ? t.substr(1) : t;
        }
        for (const auto& c : n.children) findRel(*c);
    };
    findRel(*relDoc);
    auto model = files.find(target);
    if (target.empty() || model == files.end()) throw std::runtime_error("The 3MF archive is missing its model.");
    auto doc = xml::parse(std::string(model->second.begin(), model->second.end()));
    const xml::Node* root = nullptr;
    for (const auto& c : doc->children)
        if (c->tag == "model") root = c.get();
    if (!root) throw std::runtime_error("The 3MF model is empty.");
    const std::string unit = root->attr("unit", "millimeter");
    const std::map<std::string, double> scales = {{"micron", 0.001}, {"millimeter", 1}, {"centimeter", 10}, {"inch", 25.4}, {"foot", 304.8}, {"meter", 1000}};
    auto su = scales.find(unit);
    if (su == scales.end()) throw std::runtime_error("Unsupported 3MF unit: " + unit + ".");
    const double scale = su->second;
    using M3 = std::array<double, 12>;  // 3MF row-major 3x4 affine (m00 m01 m02 m10 ... m30 m31 m32)
    auto parseM = [](const std::string& s) {
        M3 t = {1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0};
        if (s.empty()) return t;
        std::istringstream in(s);
        for (auto& v : t) in >> v;
        return t;
    };
    auto compose = [](const M3& a, const M3& b) {  // apply a, then b
        M3 r{};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 3; j++) {
                double s = i == 3 ? b[9 + j] : 0;
                for (int k = 0; k < 3; k++) s += a[i * 3 + k] * b[k * 3 + j];
                r[i * 3 + j] = s;
            }
        return r;
    };
    std::map<std::string, const xml::Node*> objects;
    const xml::Node* build = nullptr;
    for (const auto& c : root->children) {
        if (c->tag == "resources")
            for (const auto& o : c->children)
                if (o->tag == "object") objects[o->attr("id")] = o.get();
        if (c->tag == "build") build = c.get();
    }
    std::function<void(const std::string&, const M3&, int)> emit = [&](const std::string& id, const M3& T, int depth) {
        auto it = objects.find(id);
        if (it == objects.end() || depth > 32) return;
        const double det = T[0] * (T[4] * T[8] - T[5] * T[7]) - T[1] * (T[3] * T[8] - T[5] * T[6]) + T[2] * (T[3] * T[7] - T[4] * T[6]);
        for (const auto& part : it->second->children) {
            if (part->tag == "mesh") {
                std::vector<double> V;
                for (const auto& sec : part->children) {
                    if (sec->tag == "vertices")
                        for (const auto& v : sec->children) {
                            const double x = std::atof(v->attr("x").c_str()), y = std::atof(v->attr("y").c_str()), z = std::atof(v->attr("z").c_str());
                            for (int j = 0; j < 3; j++) V.push_back(x * T[j] + y * T[3 + j] + z * T[6 + j] + T[9 + j]);
                        }
                }
                for (const auto& sec : part->children) {
                    if (sec->tag != "triangles") continue;
                    for (const auto& t : sec->children) {
                        long idx[3] = {std::atol(t->attr("v1").c_str()), std::atol(t->attr("v2").c_str()), std::atol(t->attr("v3").c_str())};
                        if (det < 0) std::swap(idx[1], idx[2]);
                        for (long v : idx) {
                            if (v < 0 || size_t(v) >= V.size() / 3) throw std::runtime_error("The 3MF file references a vertex that does not exist.");
                            for (int j = 0; j < 3; j++) m.positions.push_back(float(V[3 * v + j] * scale));
                        }
                    }
                }
            } else if (part->tag == "components") {
                for (const auto& c : part->children)
                    if (c->tag == "component") emit(c->attr("objectid"), compose(parseM(c->attr("transform")), T), depth + 1);
            }
        }
    };
    if (build)
        for (const auto& item : build->children)
            if (item->tag == "item") emit(item->attr("objectid"), parseM(item->attr("transform")), 0);
    if (m.positions.empty()) throw std::runtime_error("No meshes found in file.");
    return m;
}

}  // namespace ps
