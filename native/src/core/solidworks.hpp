// SolidWorks .SLDPRT / .SLDASM reader (SolidWorks 2015 and newer), without SolidWorks.
//
// The exact B-rep inside these files is Parasolid, which only the licensed Parasolid kernel can
// read. What we read is the display tessellation SolidWorks saves with the file: one triangle-
// strip table per face, so parts arrive with their CAD faces intact. Assemblies store an
// instance tree of rigid placements and, in newer versions, a cached mesh per component;
// components without one are read from their own part files.
//
// Format facts follow sldprt-export (MIT, (c) 2026 XRTC5), which credits the container layout to
// the cadmpeg project's docs/formats/sldprt.md (CC BY 4.0). Independent implementation.
#pragma once

#include <string>
#include <vector>

#include "importers.hpp"

namespace ps {

MeshSource readSolidWorks(const Bytes& data, const std::string& fileName, const SiblingReader& readSibling = {});
/** File names of the part documents an assembly references. */
std::vector<std::string> assemblyPartNames(const Bytes& data);
/** Best library material id for a SolidWorks material name, or empty. */
std::string matchMaterial(const std::string& swName);

}  // namespace ps
