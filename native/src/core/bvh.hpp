// Bounding-volume hierarchy over a triangle mesh for fast ray casts (wall thickness, airflow
// wall distances, picking in the viewport). Binned-SAH build; nearest hit in [near, far].
#pragma once

#include <cstdint>
#include <vector>

namespace ps {

struct RayHit {
    int tri = -1;
    double distance = 0, u = 0, v = 0;  // barycentrics of the hit (w = 1 - u - v on vertex 0)
};

class BVH {
public:
    BVH() = default;
    BVH(const std::vector<float>& vertices, const std::vector<uint32_t>& tris);
    /** Nearest hit of origin + t*dir (dir need not be unit; distance is in units of |dir|) with t in [tmin, tmax]. */
    bool raycast(const double origin[3], const double dir[3], double tmin, double tmax, RayHit& hit) const;
    bool empty() const { return nodes_.empty(); }

private:
    struct Node {
        float min[3], max[3];
        int left = -1, right = -1;  // children (internal) or -1
        int start = 0, count = 0;   // triangle range (leaf)
    };
    int build(int start, int count, int depth);

    const std::vector<float>* V_ = nullptr;
    const std::vector<uint32_t>* T_ = nullptr;
    std::vector<Node> nodes_;
    std::vector<int> order_;
    std::vector<float> centroid_, lo_, hi_;
};

}  // namespace ps
