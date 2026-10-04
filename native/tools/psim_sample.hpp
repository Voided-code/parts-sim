// The content of test/helpers/psim-content.mjs sampleContent(), built with the same formulas (shared by psim_tool and test_psim).
// This file must be compiled without fused multiply-add (-ffp-contract=off) so the floats equal the JavaScript ones.
#pragma once

#include <cmath>

#include "core/psim.hpp"

namespace ps::psim {

using json::Value;

inline File sampleContent(bool brep) {
    const int nu = 40, nv = 24;
    File c;
    Geometry g;
    g.vertices.resize(size_t(3 * nu * nv));
    for (int i = 0; i < nu; i++)
        for (int j = 0; j < nv; j++) {
            const double a = (2 * M_PI * i) / nu, b = (2 * M_PI * j) / nv;
            const size_t o = size_t(3 * (i * nv + j));
            g.vertices[o] = float((50 + 10 * std::cos(b)) * std::cos(a));
            g.vertices[o + 1] = float(10 * std::sin(b) + 12);
            g.vertices[o + 2] = float((50 + 10 * std::cos(b)) * std::sin(a));
        }
    for (int i = 0; i < nu; i++)
        for (int j = 0; j < nv; j++) {
            const uint32_t a = uint32_t(i * nv + j), b = uint32_t(((i + 1) % nu) * nv + j), cc = uint32_t(((i + 1) % nu) * nv + ((j + 1) % nv)), d = uint32_t(i * nv + ((j + 1) % nv));
            for (uint32_t t : {a, b, cc, a, cc, d}) g.tris.push_back(t);
        }
    const size_t nT = g.tris.size() / 3, nV = size_t(nu * nv);
    g.brepFaces = brep;
    if (brep) {
        g.faceOf.resize(nT);
        for (size_t k = 0; k < nT; k++) g.faceOf[k] = int32_t(k / 100);
        g.faceCount = uint32_t((nT - 1) / 100 + 1);
    } else {
        g.faceCount = 5;
    }
    g.faceAngle = 20;
    c.geometry = g;

    Array vm, u;
    vm.name = "static.vm";
    vm.enc = Enc::Q16;
    for (size_t i = 0; i < nV; i++) vm.f.push_back(float(1e6 + 5e5 * std::sin(double(i) / 7)));
    u.name = "static.u";
    u.enc = Enc::Q16;
    u.stride = 3;
    for (size_t i = 0; i < 3 * nV; i++) u.f.push_back(float(0.01 * std::cos(double(i) / 11)));
    Arrays rfea;
    rfea.meta = jobj();
    Value results = jarr();
    results.arr.push_back(jstr("static"));
    rfea.meta.obj["results"] = results;
    Value st = jobj();
    st.obj["maxVM"] = jnum(1.5e6);
    st.obj["lamBreak"] = jnum(INFINITY);
    st.obj["minFos"] = jnum(NAN);
    rfea.meta.obj["static"] = st;
    rfea.list = {vm, u};
    c.rfea = rfea;

    Array ux;
    ux.name = "airflow.avg.ux";
    ux.enc = Enc::Q16;
    ux.dims = {6, 5, 4};
    for (int i = 0; i < 6 * 5 * 4; i++) ux.f.push_back(float(std::sin(double(i) / 9)));
    Arrays rair;
    rair.meta = jobj();
    Value af = jobj();
    af.obj["cd"] = jnum(0.3);
    rair.meta.obj["airflow"] = af;
    rair.list = {ux};
    c.rair = rair;

    c.info = jobj();
    Value app = jobj();
    app.obj["name"] = jstr("Parts Sim");
    app.obj["version"] = jstr("1.1.0");
    app.obj["kind"] = jstr("web");
    c.info.obj["app"] = app;
    c.info.obj["name"] = jstr("Tube");
    c.info.obj["notes"] = jstr("n");
    c.info.obj["units"] = jstr("mm");
    Value contains = jobj();
    contains.obj["geometry"] = jstr("quantised16");
    contains.obj["setup"] = jbool(true);
    Value cr = jarr();
    cr.arr.push_back(jstr("static"));
    contains.obj["results"] = cr;
    contains.obj["cad"] = jbool(false);
    c.info.obj["contains"] = contains;

    c.thumb = Bytes{0xff, 0xd8, 0xff, 0xd9};

    Value setup = jobj();
    setup.obj["units"] = jstr("mm");
    Value structural = jobj();
    structural.obj["study"] = jstr("static");
    Value fixture = jobj();
    fixture.obj["name"] = jstr("F");
    Value patch = jobj();
    Value tris = jarr();
    tris.arr = {jnum(0), jnum(12)};
    patch.obj["tris"] = tris;
    patch.obj["clip"] = Value{};
    Value patches = jarr();
    patches.arr.push_back(patch);
    fixture.obj["patches"] = patches;
    Value fixtures = jarr();
    fixtures.arr.push_back(fixture);
    structural.obj["fixtures"] = fixtures;
    setup.obj["structural"] = structural;
    c.setup = setup;

    Value view = jobj();
    view.obj["tab"] = jstr("structural");
    c.view = view;
    return c;
}

}  // namespace ps::psim
