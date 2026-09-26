// File import: every format is reduced to a MeshSource (triangles, optional CAD face ids, units).
#pragma once

#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "mesh.hpp"

namespace ps {

using Bytes = std::vector<uint8_t>;

/** Reads a file sitting next to the one being opened (a SolidWorks assembly's parts), or empty. */
using SiblingReader = std::function<Bytes(const std::string& fileName)>;

/** File-dialog filter patterns of every supported format. */
const std::vector<std::string>& supportedExtensions();

/**
 * Import a file by extension. Throws std::runtime_error with a user-facing message on failure.
 * STEP/IGES/BREP need OpenCascade (PARTS_SIM_STEP); the rest are read here.
 */
MeshSource importFile(const std::string& path, const SiblingReader& readSibling = {});
MeshSource importBytes(const std::string& fileName, const Bytes& data, const SiblingReader& readSibling = {});

MeshSource readSTL(const Bytes& data);
MeshSource readOBJ(const Bytes& data);
MeshSource readPLY(const Bytes& data);
MeshSource readGLB(const Bytes& data);
MeshSource read3MF(const Bytes& data);

/** OpenCascade B-rep import and tessellation (format: "step", "iges" or "brep"). */
MeshSource readCAD(const std::string& format, const Bytes& data, bool fine = false);

}  // namespace ps
