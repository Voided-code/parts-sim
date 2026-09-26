#include "bvh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ps {

BVH::BVH(const std::vector<float>& V, const std::vector<uint32_t>& T) : V_(&V), T_(&T) {
    const int n = int(T.size() / 3);
    order_.resize(n);
    centroid_.resize(3 * n);
    lo_.resize(3 * n);
    hi_.resize(3 * n);
    for (int t = 0; t < n; t++) {
        order_[t] = t;
        for (int d = 0; d < 3; d++) {
            const float a = V[3 * T[3 * t] + d], b = V[3 * T[3 * t + 1] + d], c = V[3 * T[3 * t + 2] + d];
            lo_[3 * t + d] = std::min({a, b, c});
            hi_[3 * t + d] = std::max({a, b, c});
            centroid_[3 * t + d] = (a + b + c) / 3;
        }
    }
    nodes_.reserve(2 * n / 2 + 1);
    if (n) build(0, n, 0);
    centroid_.clear();
    centroid_.shrink_to_fit();
}

int BVH::build(int start, int count, int depth) {
    const int id = int(nodes_.size());
    nodes_.push_back({});
    Node node;
    for (int d = 0; d < 3; d++) { node.min[d] = std::numeric_limits<float>::max(); node.max[d] = -std::numeric_limits<float>::max(); }
    float cmin[3] = {1e30f, 1e30f, 1e30f}, cmax[3] = {-1e30f, -1e30f, -1e30f};
    for (int i = start; i < start + count; i++) {
        const int t = order_[i];
        for (int d = 0; d < 3; d++) {
            node.min[d] = std::min(node.min[d], lo_[3 * t + d]);
            node.max[d] = std::max(node.max[d], hi_[3 * t + d]);
            cmin[d] = std::min(cmin[d], centroid_[3 * t + d]);
            cmax[d] = std::max(cmax[d], centroid_[3 * t + d]);
        }
    }
    if (count <= 4 || depth > 60) {
        node.start = start;
        node.count = count;
        nodes_[id] = node;
        return id;
    }
    // binned SAH along the widest centroid axis
    int axis = 0;
    for (int d = 1; d < 3; d++)
        if (cmax[d] - cmin[d] > cmax[axis] - cmin[axis]) axis = d;
    const float extent = cmax[axis] - cmin[axis];
    int mid = start + count / 2;
    if (extent > 0) {
        constexpr int B = 16;
        struct Bin { float lo[3], hi[3]; int n; } bins[B];
        for (auto& b : bins) { b.n = 0; for (int d = 0; d < 3; d++) { b.lo[d] = 1e30f; b.hi[d] = -1e30f; } }
        const float scale = B / extent * 0.99999f;
        for (int i = start; i < start + count; i++) {
            const int t = order_[i];
            const int b = std::min(B - 1, int((centroid_[3 * t + axis] - cmin[axis]) * scale));
            bins[b].n++;
            for (int d = 0; d < 3; d++) { bins[b].lo[d] = std::min(bins[b].lo[d], lo_[3 * t + d]); bins[b].hi[d] = std::max(bins[b].hi[d], hi_[3 * t + d]); }
        }
        auto area = [](const float* lo, const float* hi) {
            const float x = hi[0] - lo[0], y = hi[1] - lo[1], z = hi[2] - lo[2];
            return x < 0 ? 0.f : 2 * (x * y + y * z + z * x);
        };
        float leftA[B], rightA[B];
        int leftN[B], rightN[B];
        float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
        int acc = 0;
        for (int b = 0; b < B; b++) {
            acc += bins[b].n;
            for (int d = 0; d < 3; d++) { lo[d] = std::min(lo[d], bins[b].lo[d]); hi[d] = std::max(hi[d], bins[b].hi[d]); }
            leftA[b] = area(lo, hi);
            leftN[b] = acc;
        }
        for (int d = 0; d < 3; d++) { lo[d] = 1e30f; hi[d] = -1e30f; }
        acc = 0;
        for (int b = B - 1; b >= 0; b--) {
            acc += bins[b].n;
            for (int d = 0; d < 3; d++) { lo[d] = std::min(lo[d], bins[b].lo[d]); hi[d] = std::max(hi[d], bins[b].hi[d]); }
            rightA[b] = area(lo, hi);
            rightN[b] = acc;
        }
        float best = std::numeric_limits<float>::max();
        int split = -1;
        for (int b = 0; b + 1 < B; b++) {
            if (!leftN[b] || !rightN[b + 1]) continue;
            const float cost = leftA[b] * leftN[b] + rightA[b + 1] * rightN[b + 1];
            if (cost < best) { best = cost; split = b; }
        }
        if (split >= 0) {
            auto it = std::partition(order_.begin() + start, order_.begin() + start + count, [&](int t) {
                return std::min(B - 1, int((centroid_[3 * t + axis] - cmin[axis]) * scale)) <= split;
            });
            mid = int(it - order_.begin());
        }
    }
    if (mid <= start || mid >= start + count) {
        mid = start + count / 2;
        std::nth_element(order_.begin() + start, order_.begin() + mid, order_.begin() + start + count,
                         [&](int a, int b) { return centroid_[3 * a + axis] < centroid_[3 * b + axis]; });
    }
    node.left = build(start, mid - start, depth + 1);
    node.right = build(mid, start + count - mid, depth + 1);
    nodes_[id] = node;
    return id;
}

bool BVH::raycast(const double o[3], const double d[3], double tmin, double tmax, RayHit& hit) const {
    if (nodes_.empty()) return false;
    const auto& V = *V_;
    const auto& T = *T_;
    double inv[3];
    for (int k = 0; k < 3; k++) inv[k] = d[k] != 0 ? 1 / d[k] : std::copysign(1e300, d[k] == 0 ? 1.0 : d[k]);
    bool found = false;
    double best = tmax;
    int stack[128], sp = 0;
    stack[sp++] = 0;
    while (sp) {
        const Node& n = nodes_[stack[--sp]];
        double t0 = tmin, t1 = best;
        bool miss = false;
        for (int k = 0; k < 3 && !miss; k++) {
            double a = (n.min[k] - o[k]) * inv[k], b = (n.max[k] - o[k]) * inv[k];
            if (a > b) std::swap(a, b);
            t0 = std::max(t0, a);
            t1 = std::min(t1, b);
            miss = t0 > t1 * (1 + 1e-12) + 1e-300;
        }
        if (miss) continue;
        if (n.left < 0) {
            for (int i = n.start; i < n.start + n.count; i++) {
                const int t = order_[i];
                const float* A = &V[3 * T[3 * t]];
                const float* Bv = &V[3 * T[3 * t + 1]];
                const float* C = &V[3 * T[3 * t + 2]];
                const double e1[3] = {double(Bv[0]) - A[0], double(Bv[1]) - A[1], double(Bv[2]) - A[2]};
                const double e2[3] = {double(C[0]) - A[0], double(C[1]) - A[1], double(C[2]) - A[2]};
                const double p[3] = {d[1] * e2[2] - d[2] * e2[1], d[2] * e2[0] - d[0] * e2[2], d[0] * e2[1] - d[1] * e2[0]};
                const double det = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2];
                if (std::abs(det) < 1e-300) continue;
                const double idet = 1 / det;
                const double s[3] = {o[0] - A[0], o[1] - A[1], o[2] - A[2]};
                const double u = (s[0] * p[0] + s[1] * p[1] + s[2] * p[2]) * idet;
                if (u < 0 || u > 1) continue;
                const double q[3] = {s[1] * e1[2] - s[2] * e1[1], s[2] * e1[0] - s[0] * e1[2], s[0] * e1[1] - s[1] * e1[0]};
                const double v = (d[0] * q[0] + d[1] * q[1] + d[2] * q[2]) * idet;
                if (v < 0 || u + v > 1) continue;
                const double t2 = (e2[0] * q[0] + e2[1] * q[1] + e2[2] * q[2]) * idet;
                if (t2 < tmin || t2 > best) continue;
                best = t2;
                hit.tri = t;
                hit.distance = t2;
                hit.u = u;
                hit.v = v;
                found = true;
            }
        } else if (sp + 2 <= 128) {
            stack[sp++] = n.left;
            stack[sp++] = n.right;
        }
    }
    return found;
}

}  // namespace ps
