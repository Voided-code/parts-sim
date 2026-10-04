// Plain-data helpers for .psim files shared by the panels: materials, triangle selections (patches) and
// number lists as json::Value, in the layout of docs/psim-format.md (JS: src/core/psim-patches.js).
#pragma once

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "core/materials.hpp"
#include "core/psim.hpp"
#include "fea/structural.hpp"

namespace ps::pj {

using json::Value;
using psim::jarr;
using psim::jbool;
using psim::jnum;
using psim::jobj;
using psim::jstr;

inline Value vec(const std::vector<double>& v) { Value a = jarr(); for (double d : v) a.arr.push_back(jnum(d)); return a; }
inline Value vec3(const Vec3& v) { return vec({v[0], v[1], v[2]}); }
inline bool isNum(const Value& v) { return v.type == Value::Number && std::isfinite(v.num); }
inline double numOr(const Value& v, double d) { return isNum(v) ? v.num : d; }
inline bool boolOr(const Value& v, bool d) { return v.type == Value::Bool ? v.b : d; }
inline std::string strOr(const Value& v, const std::string& d = {}) { return v.type == Value::String ? v.str : d; }

inline Value materialToJson(const Material& m) {
    Value o = jobj();
    o.obj["id"] = jstr(m.id);
    o.obj["name"] = jstr(m.name);
    o.obj["E"] = jnum(m.E);
    o.obj["nu"] = jnum(m.nu);
    o.obj["yield"] = jnum(m.yield);
    o.obj["uts"] = jnum(m.uts);
    o.obj["density"] = jnum(m.density);
    if (m.brittle) o.obj["brittle"] = jbool(true);
    o.obj["elongation"] = jnum(m.elongation);
    o.obj["k"] = jnum(m.k);
    o.obj["cp"] = jnum(m.cp);
    o.obj["alpha"] = jnum(m.alpha);
    Value f = jobj();
    f.obj["Se"] = jnum(m.fatigue.Se);
    f.obj["Ne"] = jnum(m.fatigue.Ne);
    f.obj["endurance"] = jbool(m.fatigue.endurance);
    if (m.fatigue.metal) f.obj["metal"] = jbool(true);
    o.obj["fatigue"] = f;
    o.obj["cost"] = jnum(m.cost);
    return o;
}

inline Material materialFromJson(const Value& j) {
    if (j.type != Value::Object || !isNum(j["E"]) || !isNum(j["density"])) throw std::runtime_error("The material in the file is incomplete.");
    Material m;
    m.id = strOr(j["id"], "custom");
    m.name = strOr(j["name"], "Custom");
    m.E = j["E"].num;
    m.nu = numOr(j["nu"], m.nu);
    m.yield = numOr(j["yield"], m.yield);
    m.uts = numOr(j["uts"], m.uts);
    m.density = j["density"].num;
    m.brittle = boolOr(j["brittle"], false);
    m.elongation = numOr(j["elongation"], m.elongation);
    m.k = numOr(j["k"], m.k);
    m.cp = numOr(j["cp"], m.cp);
    m.alpha = numOr(j["alpha"], m.alpha);
    const Value& f = j["fatigue"];
    m.fatigue.Se = numOr(f["Se"], 0);
    m.fatigue.Ne = numOr(f["Ne"], 1e7);
    m.fatigue.endurance = boolOr(f["endurance"], false);
    m.fatigue.metal = boolOr(f["metal"], false);
    m.cost = numOr(j["cost"], m.cost);
    return m;
}

/** { tris: runs [start, count, ...], clip } like patchToJson in src/core/psim-patches.js. */
inline Value patchToJson(const Patch& p) {
    std::vector<int32_t> s = p.tris;
    std::sort(s.begin(), s.end());
    Value runs = jarr();
    for (size_t i = 0; i < s.size();) {
        size_t j = i + 1;
        while (j < s.size() && s[j] == s[j - 1] + 1) j++;
        runs.arr.push_back(jnum(s[i]));
        runs.arr.push_back(jnum(double(j - i)));
        i = j;
    }
    Value o = jobj();
    o.obj["tris"] = runs;
    if (p.clip) {
        Value c = jobj();
        c.obj["center"] = vec3(p.clip->center);
        c.obj["radius"] = jnum(p.clip->radius);
        o.obj["clip"] = c;
    } else {
        o.obj["clip"] = Value{};
    }
    return o;
}

inline Patch patchFromJson(const Value& j, int nTri) {
    const Value& runs = j["tris"];
    if (runs.type != Value::Array || runs.arr.size() % 2) throw std::runtime_error("A selection in the file is malformed.");
    Patch p;
    double n = 0;
    for (size_t i = 0; i < runs.arr.size(); i += 2) {
        const Value &a = runs.arr[i], &b = runs.arr[i + 1];
        if (!isNum(a) || !isNum(b) || std::floor(a.num) != a.num || std::floor(b.num) != b.num || a.num < 0 || b.num < 1 || a.num + b.num > nTri)
            throw std::runtime_error("A selection in the file refers to triangles the part does not have.");
        n += b.num;
        if (n > nTri) throw std::runtime_error("A selection in the file is longer than the part.");
        for (int k = 0; k < int(b.num); k++) p.tris.push_back(int32_t(a.num) + k);
    }
    const Value& c = j["clip"];
    if (c.type == Value::Object && c["center"].size() == 3 && isNum(c["radius"]))
        p.clip = Clip{{numOr(c["center"][0], 0), numOr(c["center"][1], 0), numOr(c["center"][2], 0)}, c["radius"].num};
    return p;
}

}  // namespace ps::pj
