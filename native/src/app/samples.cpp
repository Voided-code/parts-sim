#include "samples.hpp"

#include <cmath>

#ifdef PARTS_SIM_HAS_OCCT
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <GC_MakeArcOfCircle.hxx>
#include <GC_MakeCircle.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <Geom_BSplineCurve.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Wire.hxx>
#include <gp_Ax2.hxx>
#include <gp_Circ.hxx>

#include "core/occt_mesh.hpp"
#endif

namespace ps {

namespace {

constexpr double EPS = 0.02;

Patch where(const Part& p, const std::function<bool(const double* c, const float* n, const BBox& b)>& test) {
    Patch patch;
    for (int t = 0; t < p.nTri; t++) {
        double c[3] = {0, 0, 0};
        for (int k = 0; k < 3; k++)
            for (int d = 0; d < 3; d++) c[d] += p.vertices[3 * p.tris[3 * t + k] + d] / 3.0;
        if (test(c, &p.triNormal[3 * t], p.bbox)) patch.tris.push_back(t);
    }
    return patch;
}

Load force(const std::string& name, double magnitude, Vec3 dir, Patch patch) {
    Load l;
    l.name = name;
    l.type = Load::Force;
    l.magnitude = magnitude;
    l.dir = dir;
    l.patches = {std::move(patch)};
    return l;
}

MeshSource boxMesh(double sx, double sy, double sz) {
    MeshSource m;
    const double x = sx / 2, y = sy / 2, z = sz / 2;
    const double v[8][3] = {{-x, -y, -z}, {x, -y, -z}, {x, y, -z}, {-x, y, -z}, {-x, -y, z}, {x, -y, z}, {x, y, z}, {-x, y, z}};
    for (auto& q : v)
        for (double c : q) m.positions.push_back(float(c));
    m.index = {0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 0, 4, 7, 0, 7, 3};
    return m;
}

// Ahmed body: rectangular sections with all four nose edges rounded (R 100) and a 25 degree slant
// 222 mm long, measured along the slanted face (Ahmed et al. 1984)
MeshSource ahmedBody() {
    const double L = 1044, H = 288, W = 389, R = 100, sl = 222 * std::cos(25 * M_PI / 180), tanA = std::tan(25 * M_PI / 180);
    std::vector<double> xs;
    for (int i = 0; i <= 32; i++) xs.push_back(R * (1 - std::cos(i / 32.0 * M_PI / 2)));
    xs.push_back(L - sl);
    xs.push_back(L);
    std::vector<std::array<std::array<double, 3>, 4>> rings;
    for (double x : xs) {
        const double d = x < R ? R - std::sqrt(R * R - (R - x) * (R - x)) : 0;
        const double top = x > L - sl ? H - (x - (L - sl)) * tanA : H;
        const double hw = W / 2 - d;
        rings.push_back({{{x, d, -hw}, {x, d, hw}, {x, top - d, hw}, {x, top - d, -hw}}});
    }
    MeshSource m;
    auto tri = [&](const std::array<double, 3>& a, const std::array<double, 3>& b, const std::array<double, 3>& c) {
        for (auto* p : {&a, &b, &c})
            for (double v : *p) m.positions.push_back(float(v));
    };
    for (size_t k = 0; k + 1 < rings.size(); k++)
        for (int e = 0; e < 4; e++) {
            const auto &a = rings[k][e], &b = rings[k][(e + 1) % 4], &c = rings[k + 1][(e + 1) % 4], &d = rings[k + 1][e];
            tri(a, d, c);
            tri(a, c, b);
        }
    const auto &f = rings.front(), &r = rings.back();
    tri(f[0], f[1], f[2]);
    tri(f[0], f[2], f[3]);
    tri(r[0], r[2], r[1]);
    tri(r[0], r[3], r[2]);
    return m;
}

#ifdef PARTS_SIM_HAS_OCCT

TopoDS_Wire polygonWire(const std::vector<gp_Pnt>& pts) {
    BRepBuilderAPI_MakePolygon poly;
    for (const auto& p : pts) poly.Add(p);
    poly.Close();
    return poly.Wire();
}

TopoDS_Shape extrude(const TopoDS_Face& face, double depth) { return BRepPrimAPI_MakePrism(face, gp_Vec(0, 0, depth)).Shape(); }

TopoDS_Shape rotateX90(const TopoDS_Shape& s) {
    gp_Trsf t;
    t.SetRotation(gp::OX(), M_PI / 2);
    return BRepBuilderAPI_Transform(s, t, true).Shape();
}

MeshSource lBracket() {
    const double t = 8, r = 4;
    BRepBuilderAPI_MakeWire w;
    auto line = [&](gp_Pnt a, gp_Pnt b) { w.Add(BRepBuilderAPI_MakeEdge(a, b).Edge()); };
    line({0, 0, 0}, {80, 0, 0});
    line({80, 0, 0}, {80, t, 0});
    line({80, t, 0}, {t + r, t, 0});
    // inside fillet between the base and the upright
    const Handle(Geom_TrimmedCurve) arc = GC_MakeArcOfCircle(gp_Pnt(t + r, t, 0), gp_Pnt(t + r - r * std::sqrt(0.5), t + r - r * std::sqrt(0.5), 0), gp_Pnt(t, t + r, 0)).Value();
    w.Add(BRepBuilderAPI_MakeEdge(arc).Edge());
    line({t, t + r, 0}, {t, 60, 0});
    line({t, 60, 0}, {0, 60, 0});
    line({0, 60, 0}, {0, 0, 0});
    return tessellate(extrude(BRepBuilderAPI_MakeFace(w.Wire()).Face(), 40));
}

MeshSource plateBracket() {
    const TopoDS_Wire outer = polygonWire({{0, -25, 0}, {80, -25, 0}, {140, -12, 0}, {140, 12, 0}, {80, 25, 0}, {0, 25, 0}});
    BRepBuilderAPI_MakeFace face(outer);
    for (double y : {-12.0, 12.0}) {
        gp_Circ c(gp_Ax2(gp_Pnt(15, y, 0), gp::DZ()), 5);
        TopoDS_Wire hole = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(c).Edge()).Wire();
        hole.Reverse();
        face.Add(hole);
    }
    // lightening slot: two half circles joined by straight sides
    BRepBuilderAPI_MakeWire slot;
    slot.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(50, -6, 0), gp_Pnt(95, -6, 0)).Edge());
    slot.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(gp_Pnt(95, -6, 0), gp_Pnt(101, 0, 0), gp_Pnt(95, 6, 0)).Value()).Edge());
    slot.Add(BRepBuilderAPI_MakeEdge(gp_Pnt(95, 6, 0), gp_Pnt(50, 6, 0)).Edge());
    slot.Add(BRepBuilderAPI_MakeEdge(GC_MakeArcOfCircle(gp_Pnt(50, 6, 0), gp_Pnt(44, 0, 0), gp_Pnt(50, -6, 0)).Value()).Edge());
    TopoDS_Wire sw = slot.Wire();
    sw.Reverse();
    face.Add(sw);
    return tessellate(rotateX90(extrude(face.Face(), 6)));
}

MeshSource wrench() {
    const double R = 17, w = 6.5, hw = 7, L = 150;
    const double xj = -std::sqrt(R * R - w * w);
    const double tip = std::atan2(w, xj), neck = std::asin(hw / R);
    std::vector<gp_Pnt> pts;
    auto arc = [&](double cx, double cy, double r, double a0, double a1, int n) {
        for (int i = 0; i <= n; i++) {
            const double a = a0 + (a1 - a0) * i / n;
            pts.emplace_back(cx + r * std::cos(a), cy + r * std::sin(a), 0);
        }
    };
    arc(0, 0, R, 2 * M_PI - tip, 2 * M_PI - neck, 28);
    arc(L, 0, hw, -M_PI / 2, M_PI / 2, 18);
    arc(0, 0, R, neck, tip, 28);
    pts.emplace_back(2, w, 0);
    pts.emplace_back(2, -w, 0);
    return tessellate(rotateX90(extrude(BRepBuilderAPI_MakeFace(polygonWire(pts)).Face(), 5)));
}

MeshSource hook() {
    std::vector<gp_Pnt> pts;
    for (double y = 90; y > 30; y -= 6) pts.emplace_back(0, y, 0);
    for (double a = 180; a <= 400; a += 8) {
        const double r = a * M_PI / 180;
        pts.emplace_back(22 + 22 * std::cos(r), 30 + 22 * std::sin(r), 0);
    }
    Handle(TColgp_HArray1OfPnt) arr = new TColgp_HArray1OfPnt(1, int(pts.size()));
    for (size_t i = 0; i < pts.size(); i++) arr->SetValue(int(i) + 1, pts[i]);
    GeomAPI_Interpolate interp(arr, false, 1e-6);
    interp.Perform();
    const TopoDS_Wire spine = BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(interp.Curve()).Edge()).Wire();
    gp_Pnt p0;
    gp_Vec t0;
    interp.Curve()->D1(interp.Curve()->FirstParameter(), p0, t0);
    const gp_Circ c(gp_Ax2(p0, gp_Dir(t0)), 6);
    const TopoDS_Face profile = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(BRepBuilderAPI_MakeEdge(c).Edge()).Wire()).Face();
    BRepOffsetAPI_MakePipe pipe(spine, profile);
    pipe.Build();
    return tessellate(pipe.Shape());
}

MeshSource wing() {
    const double m = 0.02, p = 0.4, t = 0.12, c = 100;
    const int n = 60;
    std::vector<gp_Pnt> up, lo;
    for (int i = 0; i <= n; i++) {
        const double x = (1 - std::cos(double(i) / n * M_PI)) / 2;
        const double yt = 5 * t * (0.2969 * std::sqrt(x) - 0.126 * x - 0.3516 * x * x + 0.2843 * x * x * x - 0.1036 * x * x * x * x);
        const double yc = x < p ? m / (p * p) * (2 * p * x - x * x) : m / ((1 - p) * (1 - p)) * (1 - 2 * p + 2 * p * x - x * x);
        const double dy = x < p ? 2 * m / (p * p) * (p - x) : 2 * m / ((1 - p) * (1 - p)) * (p - x);
        const double th = std::atan(dy);
        up.emplace_back((x - yt * std::sin(th)) * c, (yc + yt * std::cos(th)) * c, 0);
        lo.emplace_back((x + yt * std::sin(th)) * c, (yc - yt * std::cos(th)) * c, 0);
    }
    // one smooth curve from the lower trailing edge round the nose to the upper trailing edge
    std::vector<gp_Pnt> loop;
    for (int i = n; i >= 1; i--) loop.push_back(lo[i]);
    for (int i = 0; i <= n; i++) loop.push_back(up[i]);
    Handle(TColgp_HArray1OfPnt) arr = new TColgp_HArray1OfPnt(1, int(loop.size()));
    for (size_t i = 0; i < loop.size(); i++) arr->SetValue(int(i) + 1, loop[i]);
    GeomAPI_Interpolate interp(arr, false, 1e-7);
    interp.Perform();
    BRepBuilderAPI_MakeWire w;
    w.Add(BRepBuilderAPI_MakeEdge(interp.Curve()).Edge());
    // this profile closes at the trailing edge; a blunt one needs a closing segment
    if (loop.back().Distance(loop.front()) > 1e-6) w.Add(BRepBuilderAPI_MakeEdge(loop.back(), loop.front()).Edge());
    return tessellate(extrude(BRepBuilderAPI_MakeFace(w.Wire()).Face(), 300));
}

MeshSource beamCAD() {
    return tessellate(BRepPrimAPI_MakeBox(gp_Pnt(-100, -10, -5), gp_Pnt(100, 10, 5)).Shape());
}

#endif

}  // namespace

const std::vector<Sample>& samples() {
    static const std::vector<Sample> list = [] {
        std::vector<Sample> s;
#ifdef PARTS_SIM_HAS_OCCT
        auto beamMake = [] { return beamCAD(); };
#else
        auto beamMake = [] { return boxMesh(200, 20, 10); };
#endif
        s.push_back({"beam", "Cantilever beam", "200 x 20 x 10 mm steel bar, clamped at one end, 800 N at the tip", beamMake, [](const Part& p) {
                         SampleSetup st;
                         st.material = "steel-1020";
                         st.fixtures.push_back({"Clamped end", {where(p, [](const double* c, const float* n, const BBox& b) { return c[0] < b.min[0] + EPS && n[0] < -0.9; })}});
                         st.loads.push_back(force("Tip load", 800, {0, -1, 0}, where(p, [](const double* c, const float* n, const BBox& b) { return c[0] > b.max[0] - EPS && n[0] > 0.9; })));
                         return st;
                     }});
#ifdef PARTS_SIM_HAS_OCCT
        s.push_back({"lbracket", "L-bracket (PLA print)", "3D-printed PLA shelf bracket, bolted base, 180 N pulling the top outward", lBracket, [](const Part& p) {
                         SampleSetup st;
                         st.material = "pla";
                         st.fixtures.push_back({"Bolted base", {where(p, [](const double* c, const float* n, const BBox& b) { return c[1] < b.min[1] + EPS && n[1] < -0.9 && c[0] > b.min[0] + 30; })}});
                         st.loads.push_back(force("Pull on top", 180, {-1, 0, 0}, where(p, [](const double* c, const float* n, const BBox& b) {
                                                      return c[1] > b.max[1] - 12 && n[0] > 0.9 && c[0] < b.min[0] + 9;
                                                  })));
                         return st;
                     }});
        s.push_back({"bracket", "Bolted mounting bracket", "Aluminium 6061 plate with two bolt holes and a lightening slot, 250 N at the tip", plateBracket, [](const Part& p) {
                         SampleSetup st;
                         st.material = "al-6061";
                         const double x0 = p.bbox.min[0], zc = (p.bbox.min[2] + p.bbox.max[2]) / 2;
                         st.fixtures.push_back({"Bolt holes", {where(p, [=](const double* c, const float* n, const BBox&) {
                                                    bool in = false;
                                                    for (double z : {-12.0, 12.0}) in = in || std::hypot(c[0] - (x0 + 15), c[2] - (zc + z)) < 5.4;
                                                    return in && std::abs(n[1]) < 0.5;
                                                })}});
                         st.loads.push_back(force("Tip load", 250, {0, -1, 0}, where(p, [](const double* c, const float* n, const BBox& b) { return c[0] > b.max[0] - EPS && n[0] > 0.9; })));
                         return st;
                     }});
        s.push_back({"wrench", "Open-end wrench", "13 mm Cr-V spanner gripping a nut, 500 N pushed at the end of the handle", wrench, [](const Part& p) {
                         SampleSetup st;
                         st.material = "steel-alloy";
                         const double x0 = p.bbox.min[0], zc = (p.bbox.min[2] + p.bbox.max[2]) / 2;
                         st.fixtures.push_back({"Nut flats", {where(p, [=](const double* c, const float* n, const BBox&) {
                                                   return c[0] < x0 + 18.5 && std::abs(std::abs(c[2] - zc) - 6.5) < 0.3 && std::abs(n[2]) > 0.9;
                                               })}});
                         st.loads.push_back(force("Hand force", 500, {0, 0, 1}, where(p, [](const double* c, const float* n, const BBox& b) { return c[0] > b.max[0] - 25 && n[2] < -0.7; })));
                         return st;
                     }});
        s.push_back({"hook", "Crane hook", "12 mm steel rod bent into a J, hung from the top, 1.5 kN load in the throat", hook, [](const Part& p) {
                         SampleSetup st;
                         st.material = "steel-1020";
                         const BBox b = p.bbox;
                         const double cx = b.min[0] + 6 + 22;
                         st.fixtures.push_back({"Top eye", {where(p, [=](const double* c, const float* n, const BBox&) { return c[1] > b.max[1] - 0.5 && n[1] > 0.9; })}});
                         st.loads.push_back(force("Hanging load", 1500, {0, -1, 0}, where(p, [=](const double* c, const float* n, const BBox&) {
                                                      return c[1] < b.min[1] + 16 && std::abs(c[0] - cx) < 7 && n[1] > 0.6;
                                                  })));
                         return st;
                     }});
        s.push_back({"wing", "Wing (NACA 2412)", "100 mm chord, 300 mm span at 6° - see the flow and tip vortices, then apply the wind load", wing, [](const Part& p) {
                         SampleSetup st;
                         st.material = "al-6061";
                         st.airflow = SampleSetup::Air{90, 6, 30};
                         st.fixtures.push_back({"Wing root", {where(p, [](const double* c, const float* n, const BBox& b) { return c[2] < b.min[2] + EPS && n[2] < -0.9; })}});
                         Load l;
                         l.name = "Lift (distributed)";
                         l.type = Load::Pressure;
                         l.magnitude = 0.004;
                         l.patches = {where(p, [](const double*, const float* n, const BBox&) { return n[1] < -0.3; })};
                         st.loads.push_back(l);
                         return st;
                     }});
#endif
        s.push_back({"ahmed", "Ahmed body (car)", "Standard car aerodynamics benchmark: rounded nose, 25° rear slant", ahmedBody, [](const Part& p) {
                         SampleSetup st;
                         st.material = "al-6061";
                         st.airflow = SampleSetup::Air{90, 0, 40};
                         st.fixtures.push_back({"Underside", {where(p, [](const double* c, const float* n, const BBox& b) { return c[1] < b.min[1] + EPS && n[1] < -0.9; })}});
                         return st;
                     }});
        return s;
    }();
    return list;
}

const Sample* findSample(const std::string& id) {
    for (const auto& s : samples())
        if (s.id == id) return &s;
    return nullptr;
}

}  // namespace ps
