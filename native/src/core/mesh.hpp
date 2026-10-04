// Mesh processing: welding, orientation, refinement, faces, normals, edges, sampling.
//
// A Part is the single source of truth for geometry. All studies read from it, and results are
// stored per unique vertex of Part::vertices.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ps {

class BVH;
using Vec3 = std::array<double, 3>;

struct BBox {
    Vec3 min{0, 0, 0}, max{0, 0, 0}, size{0, 0, 0};
    double diag = 0;
};

/** Raw triangles from an importer. index empty = triangle soup; faceIds empty = no CAD faces. */
struct MeshSource {
    std::vector<float> positions;
    std::vector<uint32_t> index;
    std::vector<int32_t> faceIds;
    std::string name;
    std::string units;  // "mm", "m", ... or empty when the file has none
    std::string info;
    std::string material;
    std::vector<std::string> warnings;
};

struct Part {
    std::string name;
    std::vector<float> vertices;       // unique positions, model units
    std::vector<uint32_t> tris;        // 3 per triangle, outward CCW
    int nVert = 0, nTri = 0;
    std::vector<float> triNormal, triArea, vertNormal;
    std::vector<int32_t> neighbors;    // triangle across each edge, -1 if none
    std::vector<int32_t> faceOf;
    int faceCount = 0;
    bool brepFaces = false;
    double faceAngle = 20;
    BBox bbox;
    double volume = 0, area = 0;
    std::vector<uint32_t> edges;       // feature-edge vertex pairs
    // non-indexed display copy with normals split at creases
    std::vector<float> displayPosition, displayNormal;
    std::vector<uint32_t> displaySrc;  // unique vertex of each display vertex
    // cached measurements (reset when the part changes)
    std::optional<double> wallThickness;
    mutable std::shared_ptr<BVH> bvhCache;
};

struct BuildOptions {
    int maxTris = 350000;
    double faceAngle = 20;
};

/** Weld, clean, orient, refine and analyse raw triangles. Throws std::runtime_error on bad input. */
std::shared_ptr<Part> buildPart(const MeshSource& src, const BuildOptions& opts = {});

/** Rebuilds a Part from an already built mesh (a .psim file): no welding, orientation or refinement, so
 *  vertex and triangle numbers stay those of the file. faceOf is used when brepFaces, else faces grow from faceAngle. */
std::shared_ptr<Part> restorePart(const std::string& name, std::vector<float> vertices, std::vector<uint32_t> tris,
                                  std::vector<int32_t> faceOf, bool brepFaces, int faceCount, double faceAngle);

BBox boundingBox(const std::vector<float>& V);
/** Regroup triangles into faces across edges bent less than angleDeg (mesh files without CAD faces). */
void setSmoothFaces(Part& part, double angleDeg);
std::vector<int32_t> trianglesOfFace(const Part& part, int faceId);
/** Triangles connected to `seed` with a vertex within `radius` of `center`. */
std::vector<int32_t> trianglesInSphere(const Part& part, int seed, const Vec3& center, double radius);

struct Clip {
    Vec3 center;
    double radius;
};

struct Samples {
    std::vector<float> points, weights;
    std::vector<int32_t> tris;
};
/** Evenly spaced points (about one per spacing^2) with their area share and triangle. */
Samples sampleTriangles(const Part& part, const std::vector<int32_t>& triList, double spacing, const std::optional<Clip>& clip = {});

struct Frame {
    Vec3 center, normal;
    double area;
};
Frame patchFrame(const Part& part, const std::vector<int32_t>& triList, const std::optional<Clip>& clip = {});

/** Rotate positions 90 degrees about a world axis (0 X, 1 Y, 2 Z). */
std::vector<float> rotatePositions90(const std::vector<float>& positions, int axis);
/** Scale in place about the footprint centre, keeping the part on the ground; returns the pivot. */
Vec3 scalePart(Part& part, double s);

}  // namespace ps
