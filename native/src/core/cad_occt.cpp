// STEP / IGES / BREP import through OpenCascade: read the B-rep, tessellate every face, and
// keep the CAD face of each triangle so faces can be clicked in the viewport.
#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include "importers.hpp"
#include "occt_mesh.hpp"

#ifdef PARTS_SIM_HAS_OCCT
#include <BRepBndLib.hxx>
#include <BRepMesh_IncrementalMesh.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BRep_Tool.hxx>
#include <Bnd_Box.hxx>
#include <IFSelect_ReturnStatus.hxx>
#include <IGESControl_Reader.hxx>
#include <Interface_Static.hxx>
#include <Poly_Triangulation.hxx>
#include <STEPControl_Reader.hxx>
#include <Standard_Failure.hxx>
#include <TopExp_Explorer.hxx>
#include <TopoDS.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shape.hxx>
#include <XSControl_WorkSession.hxx>
#include <cstdio>
#include <filesystem>
#include <random>
#include <fstream>
#endif

namespace ps {

#ifdef PARTS_SIM_HAS_OCCT

MeshSource tessellate(const TopoDS_Shape& shape, double relDeflection, double angDeflection) {
    Bnd_Box box;
    BRepBndLib::Add(shape, box);
    if (box.IsVoid()) throw std::runtime_error("The CAD file contains no geometry.");
    double x0, y0, z0, x1, y1, z1;
    box.Get(x0, y0, z0, x1, y1, z1);
    const double diag = std::sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + (z1 - z0) * (z1 - z0));
    BRepMesh_IncrementalMesh mesher(shape, std::max(1e-6, relDeflection * diag), false, angDeflection, true);
    MeshSource m;
    int face = 0, bodies = 0;
    for (TopExp_Explorer ex(shape, TopAbs_SOLID); ex.More(); ex.Next()) bodies++;
    if (!bodies)
        for (TopExp_Explorer ex(shape, TopAbs_SHELL); ex.More(); ex.Next()) bodies++;
    for (TopExp_Explorer ex(shape, TopAbs_FACE); ex.More(); ex.Next(), face++) {
        const TopoDS_Face& f = TopoDS::Face(ex.Current());
        TopLoc_Location loc;
        const Handle(Poly_Triangulation) tri = BRep_Tool::Triangulation(f, loc);
        if (tri.IsNull()) continue;
        const gp_Trsf T = loc.Transformation();
        const uint32_t base = uint32_t(m.positions.size() / 3);
        for (int i = 1; i <= tri->NbNodes(); i++) {
            const gp_Pnt p = tri->Node(i).Transformed(T);
            m.positions.insert(m.positions.end(), {float(p.X()), float(p.Y()), float(p.Z())});
        }
        const bool reversed = f.Orientation() == TopAbs_REVERSED;
        for (int t = 1; t <= tri->NbTriangles(); t++) {
            int a, b, c;
            tri->Triangle(t).Get(a, b, c);
            if (reversed) std::swap(b, c);
            m.index.insert(m.index.end(), {base + uint32_t(a - 1), base + uint32_t(b - 1), base + uint32_t(c - 1)});
            m.faceIds.push_back(face);
        }
    }
    if (m.index.empty()) throw std::runtime_error("The CAD file contains no tessellated surfaces. Export a solid or surface body.");
    m.units = "mm";
    m.info = std::to_string(bodies) + (bodies == 1 ? " body" : " bodies");
    return m;
}

MeshSource readCAD(const std::string& format, const Bytes& data, bool fine) {
    // OpenCascade's readers take file names; hand them a private temporary copy
    namespace fs = std::filesystem;
    std::random_device rd;
    const fs::path tmp = fs::temp_directory_path() / ("parts-sim-" + std::to_string(rd()) + std::to_string(rd()) + "." + format);
    {
        std::ofstream out(tmp, std::ios::binary);
        out.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
        if (!out) throw std::runtime_error("Could not stage the CAD file for reading.");
    }
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove(p, ec); } } cleanup{tmp};
    // OpenCascade takes UTF-8 file names (the temp folder can contain a non-English user name)
    const std::u8string u8 = tmp.u8string();
    const std::string fileName(reinterpret_cast<const char*>(u8.data()), u8.size());
    TopoDS_Shape shape;
    try {
        if (format == "step") {
            STEPControl_Reader reader;
            Interface_Static::SetCVal("xstep.cascade.unit", "MM");
            if (reader.ReadFile(fileName.c_str()) != IFSelect_RetDone) throw std::runtime_error("OpenCascade could not read this STEP file.");
            reader.TransferRoots();
            shape = reader.OneShape();
        } else if (format == "iges") {
            IGESControl_Reader reader;
            Interface_Static::SetCVal("xstep.cascade.unit", "MM");
            if (reader.ReadFile(fileName.c_str()) != IFSelect_RetDone) throw std::runtime_error("OpenCascade could not read this IGES file.");
            reader.TransferRoots();
            shape = reader.OneShape();
        } else {
            BRep_Builder builder;
            if (!BRepTools::Read(shape, fileName.c_str(), builder)) throw std::runtime_error("OpenCascade could not read this BREP file.");
        }
    } catch (const Standard_Failure& e) {
        throw std::runtime_error(std::string("OpenCascade could not read this file: ") + e.GetMessageString());
    }
    if (shape.IsNull()) throw std::runtime_error("The CAD file contains no shapes.");
    MeshSource m = tessellate(shape, fine ? 0.0005 : 0.002, fine ? 0.2 : 0.4);
    std::string upper = format;
    for (auto& c : upper) c = char(std::toupper((unsigned char)c));
    m.info = upper + " B-rep, " + m.info;
    if (format == "brep") m.units.clear();
    return m;
}

#else

MeshSource readCAD(const std::string& format, const Bytes&, bool) {
    throw std::runtime_error("This build has no STEP/IGES support (OpenCascade was not found when it was compiled).");
}

#endif

}  // namespace ps
