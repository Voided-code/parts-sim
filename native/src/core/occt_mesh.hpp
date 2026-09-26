// Tessellation of OpenCascade shapes (shared by the STEP importer and the sample parts).
#pragma once

#include "mesh.hpp"

#ifdef PARTS_SIM_HAS_OCCT
#include <TopoDS_Shape.hxx>

namespace ps {
/** Triangles of every face, with the B-rep face index per triangle. Deflection relative to the size. */
MeshSource tessellate(const TopoDS_Shape& shape, double relDeflection = 0.002, double angDeflection = 0.4);
}  // namespace ps
#endif
