#include "mesh.hpp"

#include "bvh.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace ps {

namespace {

Vec3 triCross(const std::vector<float>& V, uint32_t a, uint32_t b, uint32_t c) {
    const double ax = V[3 * a], ay = V[3 * a + 1], az = V[3 * a + 2];
    const double ux = V[3 * b] - ax, uy = V[3 * b + 1] - ay, uz = V[3 * b + 2] - az;
    const double vx = V[3 * c] - ax, vy = V[3 * c + 1] - ay, vz = V[3 * c + 2] - az;
    return {uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx};
}

double norm(const Vec3& v) { return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

inline uint64_t edgeKey(uint32_t a, uint32_t b) { return a < b ? (uint64_t(a) << 32) | b : (uint64_t(b) << 32) | a; }

void validate(const MeshSource& s) {
    if (s.positions.empty() || s.positions.size() % 3) throw std::runtime_error("The file contains no valid vertex positions.");
    for (float v : s.positions)
        if (!std::isfinite(v)) throw std::runtime_error("The mesh contains non-finite vertex coordinates.");
    const size_t nVert = s.positions.size() / 3;
    const size_t nCorner = s.index.empty() ? nVert : s.index.size();
    if (!nCorner || nCorner % 3) throw std::runtime_error("The mesh does not contain complete triangles.");
    for (uint32_t i : s.index)
        if (i >= nVert) throw std::runtime_error("The mesh contains an invalid triangle index.");
    if (!s.faceIds.empty() && s.faceIds.size() != nCorner / 3) throw std::runtime_error("The CAD face data does not match its triangles.");
}

struct Mesh {
    std::vector<float> V;
    std::vector<uint32_t> T;
    std::vector<int32_t> F;
};

Mesh weld(const MeshSource& s) {
    const size_t nIn = s.positions.size() / 3;
    std::vector<uint8_t> used;
    if (!s.index.empty()) {
        used.assign(nIn, 0);
        for (uint32_t i : s.index) used[i] = 1;
    }
    double mn[3] = {1e300, 1e300, 1e300}, mx[3] = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < nIn; i++) {
        if (!used.empty() && !used[i]) continue;
        for (int d = 0; d < 3; d++) {
            mn[d] = std::min(mn[d], double(s.positions[3 * i + d]));
            mx[d] = std::max(mx[d], double(s.positions[3 * i + d]));
        }
    }
    double diag = std::sqrt((mx[0] - mn[0]) * (mx[0] - mn[0]) + (mx[1] - mn[1]) * (mx[1] - mn[1]) + (mx[2] - mn[2]) * (mx[2] - mn[2]));
    if (!(diag > 0)) diag = 1;
    const double inv = 100000 / diag;
    constexpr uint64_t P = 131072;
    std::unordered_map<uint64_t, uint32_t> map;
    map.reserve(nIn);
    std::vector<uint32_t> remap(nIn, 0);
    Mesh m;
    for (size_t i = 0; i < nIn; i++) {
        if (!used.empty() && !used[i]) continue;
        const float x = s.positions[3 * i], y = s.positions[3 * i + 1], z = s.positions[3 * i + 2];
        const uint64_t key = uint64_t(std::llround((x - mn[0]) * inv)) + P * (uint64_t(std::llround((y - mn[1]) * inv)) + P * uint64_t(std::llround((z - mn[2]) * inv)));
        auto it = map.find(key);
        uint32_t id;
        if (it == map.end()) {
            id = uint32_t(m.V.size() / 3);
            map.emplace(key, id);
            m.V.insert(m.V.end(), {x, y, z});
        } else id = it->second;
        remap[i] = id;
    }
    const size_t nTri = s.index.empty() ? nIn / 3 : s.index.size() / 3;
    m.T.resize(nTri * 3);
    for (size_t t = 0; t < nTri * 3; t++) m.T[t] = remap[s.index.empty() ? t : s.index[t]];
    m.F = s.faceIds;
    return m;
}

void dropDegenerate(Mesh& m) {
    std::vector<uint32_t> keep;
    std::vector<int32_t> keepFace;
    keep.reserve(m.T.size());
    for (size_t t = 0; t < m.T.size() / 3; t++) {
        const uint32_t a = m.T[3 * t], b = m.T[3 * t + 1], c = m.T[3 * t + 2];
        if (a == b || b == c || a == c) continue;
        const Vec3 n = triCross(m.V, a, b, c);
        if (n[0] == 0 && n[1] == 0 && n[2] == 0) continue;
        keep.insert(keep.end(), {a, b, c});
        if (!m.F.empty()) keepFace.push_back(m.F[t]);
    }
    m.T = std::move(keep);
    if (!m.F.empty()) m.F = std::move(keepFace);
}

void compactVertices(Mesh& m) {
    std::vector<int64_t> remap(m.V.size() / 3, -1);
    std::vector<float> out;
    out.reserve(m.V.size());
    for (auto& v : m.T) {
        if (remap[v] < 0) {
            remap[v] = int64_t(out.size() / 3);
            out.insert(out.end(), {m.V[3 * v], m.V[3 * v + 1], m.V[3 * v + 2]});
        }
        v = uint32_t(remap[v]);
    }
    m.V = std::move(out);
}

std::vector<int32_t> triangleNeighbors(const std::vector<uint32_t>& tris) {
    const size_t nTri = tris.size() / 3;
    std::vector<int32_t> nb(3 * nTri, -1);
    std::unordered_map<uint64_t, int64_t> edges;
    edges.reserve(tris.size());
    for (size_t t = 0; t < nTri; t++)
        for (int e = 0; e < 3; e++) {
            const uint32_t a = tris[3 * t + e], b = tris[3 * t + (e + 1) % 3];
            const uint64_t k = edgeKey(a, b);
            auto it = edges.find(k);
            if (it == edges.end()) edges.emplace(k, int64_t(3 * t + e));
            else if (it->second >= 0) {
                const int64_t other = it->second;
                nb[other] = int32_t(t);
                nb[3 * t + e] = int32_t(other / 3);
                it->second = -1;  // non-manifold edges beyond two triangles stay unlinked
            }
        }
    return nb;
}

bool insideShell(const std::vector<float>& V, const std::vector<uint32_t>& tris, const std::vector<int32_t>& triangles, const Vec3& p) {
    double angle = 0;
    for (int32_t t : triangles) {
        double v[3][3], l[3];
        for (int c = 0; c < 3; c++) {
            const uint32_t a = 3 * tris[3 * t + c];
            for (int d = 0; d < 3; d++) v[c][d] = V[a + d] - p[d];
            l[c] = std::sqrt(v[c][0] * v[c][0] + v[c][1] * v[c][1] + v[c][2] * v[c][2]);
        }
        auto dotp = [](const double* a, const double* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
        const double* a = v[0];
        const double* b = v[1];
        const double* c = v[2];
        const double det = a[0] * (b[1] * c[2] - b[2] * c[1]) + a[1] * (b[2] * c[0] - b[0] * c[2]) + a[2] * (b[0] * c[1] - b[1] * c[0]);
        angle += 2 * std::atan2(det, l[0] * l[1] * l[2] + dotp(a, b) * l[2] + dotp(b, c) * l[0] + dotp(c, a) * l[1]);
    }
    return std::abs(angle) > 2 * M_PI;
}

// Make each shell coherent, with enclosed cavity shells facing into the cavity.
void orientShells(const std::vector<float>& V, std::vector<uint32_t>& tris) {
    auto neighbors = triangleNeighbors(tris);
    const size_t nT = tris.size() / 3;
    std::vector<uint8_t> seen(nT, 0);
    struct Shell {
        std::vector<int32_t> triangles;
        bool closed = true;
        Vec3 min{1e300, 1e300, 1e300}, max{-1e300, -1e300, -1e300};
    };
    std::vector<Shell> shells;
    for (size_t seed = 0; seed < nT; seed++) {
        if (seen[seed]) continue;
        Shell sh;
        sh.triangles.push_back(int32_t(seed));
        seen[seed] = 1;
        for (size_t q = 0; q < sh.triangles.size(); q++) {
            const int32_t t = sh.triangles[q];
            for (int e = 0; e < 3; e++) {
                const uint32_t a = tris[3 * t + e], b = tris[3 * t + (e + 1) % 3];
                for (int d = 0; d < 3; d++) {
                    sh.min[d] = std::min(sh.min[d], double(V[3 * a + d]));
                    sh.max[d] = std::max(sh.max[d], double(V[3 * a + d]));
                }
                const int32_t other = neighbors[3 * t + e];
                if (other < 0) { sh.closed = false; continue; }
                if (seen[other]) continue;
                for (int oe = 0; oe < 3; oe++)
                    if (tris[3 * other + oe] == a && tris[3 * other + (oe + 1) % 3] == b) {
                        std::swap(tris[3 * other + 1], tris[3 * other + 2]);
                        std::swap(neighbors[3 * other], neighbors[3 * other + 2]);
                        break;
                    }
                seen[other] = 1;
                sh.triangles.push_back(other);
            }
        }
        shells.push_back(std::move(sh));
    }
    for (auto& sh : shells) {
        if (!sh.closed) continue;
        const int32_t first = sh.triangles[0];
        Vec3 p{0, 0, 0};
        for (int d = 0; d < 3; d++)
            for (int c = 0; c < 3; c++) p[d] += V[3 * tris[3 * first + c] + d] / 3.0;
        int depth = 0;
        for (auto& other : shells) {
            if (&other == &sh || !other.closed) continue;
            bool encloses = true;
            for (int d = 0; d < 3; d++) encloses = encloses && other.min[d] < sh.min[d] && other.max[d] > sh.max[d];
            if (encloses && insideShell(V, tris, other.triangles, p)) depth++;
        }
        double volume = 0;
        for (int32_t t : sh.triangles) {
            const uint32_t a = 3 * tris[3 * t];
            const Vec3 n = triCross(V, tris[3 * t], tris[3 * t + 1], tris[3 * t + 2]);
            volume += (V[a] - p[0]) * n[0] + (V[a + 1] - p[1]) * n[1] + (V[a + 2] - p[2]) * n[2];
        }
        if ((volume < 0) != (depth % 2 == 1))
            for (int32_t t : sh.triangles) std::swap(tris[3 * t + 1], tris[3 * t + 2]);
    }
}

void placeOnGround(std::vector<float>& V) {
    const BBox b = boundingBox(V);
    const double off[3] = {-(b.min[0] + b.max[0]) / 2, -b.min[1], -(b.min[2] + b.max[2]) / 2};
    for (size_t i = 0; i < V.size(); i += 3)
        for (int d = 0; d < 3; d++) V[i + d] = float(V[i + d] + off[d]);
}

double totalArea(const std::vector<float>& V, const std::vector<uint32_t>& T) {
    double a = 0;
    for (size_t t = 0; t < T.size(); t += 3) a += 0.5 * norm(triCross(V, T[t], T[t + 1], T[t + 2]));
    return a;
}

// Conforming refinement: split every edge longer than maxLen at its midpoint (no T-junctions).
void refine(Mesh& m, double maxLen, int maxTris) {
    std::vector<float>& V = m.V;
    const double maxLen2 = maxLen * maxLen;
    auto len2 = [&](uint32_t a, uint32_t b) {
        const double dx = V[3 * a] - V[3 * b], dy = V[3 * a + 1] - V[3 * b + 1], dz = V[3 * a + 2] - V[3 * b + 2];
        return dx * dx + dy * dy + dz * dz;
    };
    for (int pass = 0; pass < 12; pass++) {
        const size_t nT = m.T.size() / 3;
        if (int64_t(nT) >= maxTris) break;
        const size_t previous = V.size();
        std::unordered_map<uint64_t, uint32_t> mids;
        size_t refinedCount = nT, splits = 0;
        for (size_t t = 0; t < nT; t++)
            for (int e = 0; e < 3; e++) {
                const uint32_t a = m.T[3 * t + e], b = m.T[3 * t + (e + 1) % 3];
                if (len2(a, b) > maxLen2) {
                    refinedCount++;
                    const uint64_t k = edgeKey(a, b);
                    if (!mids.count(k)) {
                        mids.emplace(k, uint32_t(V.size() / 3));
                        V.push_back((V[3 * a] + V[3 * b]) / 2);
                        V.push_back((V[3 * a + 1] + V[3 * b + 1]) / 2);
                        V.push_back((V[3 * a + 2] + V[3 * b + 2]) / 2);
                        splits++;
                    }
                }
            }
        if (!splits) break;
        if (int64_t(refinedCount) > maxTris) { V.resize(previous); break; }  // keep the cap and a conforming surface
        std::vector<uint32_t> out;
        std::vector<int32_t> outFace;
        out.reserve(refinedCount * 3 + 3);
        auto mid = [&](uint32_t a, uint32_t b) -> int64_t {
            auto it = mids.find(edgeKey(a, b));
            return it == mids.end() ? -1 : int64_t(it->second);
        };
        for (size_t t = 0; t < nT; t++) {
            uint32_t v[3];
            int64_t mm[3];
            int count = 0;
            for (int e = 0; e < 3; e++) v[e] = m.T[3 * t + e];
            for (int e = 0; e < 3; e++) {
                mm[e] = mid(v[e], v[(e + 1) % 3]);
                if (mm[e] >= 0) count++;
            }
            const int32_t f = m.F.empty() ? 0 : m.F[t];
            auto emit = [&](int64_t a, int64_t b, int64_t c) {
                out.insert(out.end(), {uint32_t(a), uint32_t(b), uint32_t(c)});
                outFace.push_back(f);
            };
            if (count == 0) emit(v[0], v[1], v[2]);
            else if (count == 3) {
                emit(v[0], mm[0], mm[2]);
                emit(mm[0], v[1], mm[1]);
                emit(mm[2], mm[1], v[2]);
                emit(mm[0], mm[1], mm[2]);
            } else if (count == 1) {
                const int r = mm[0] >= 0 ? 0 : mm[1] >= 0 ? 1 : 2;
                emit(v[r], mm[r], v[(r + 2) % 3]);
                emit(mm[r], v[(r + 1) % 3], v[(r + 2) % 3]);
            } else {
                const int u = mm[0] < 0 ? 0 : mm[1] < 0 ? 1 : 2;
                const int r = (u + 1) % 3;
                const uint32_t v0 = v[r], v1 = v[(r + 1) % 3], v2 = v[(r + 2) % 3];
                const int64_t m0 = mm[r], m1 = mm[(r + 1) % 3];
                emit(m0, v1, m1);
                if (len2(v0, uint32_t(m1)) < len2(uint32_t(m0), v2)) {
                    emit(v0, m0, m1);
                    emit(v0, m1, v2);
                } else {
                    emit(v0, m0, v2);
                    emit(m0, m1, v2);
                }
            }
        }
        m.T = std::move(out);
        if (!m.F.empty()) m.F = std::move(outFace);
    }
}

void computeTriangleData(Part& p) {
    p.triNormal.assign(3 * p.nTri, 0.f);
    p.triArea.assign(p.nTri, 0.f);
    double area = 0;
    for (int t = 0; t < p.nTri; t++) {
        const Vec3 n = triCross(p.vertices, p.tris[3 * t], p.tris[3 * t + 1], p.tris[3 * t + 2]);
        double l = norm(n);
        const double a = 0.5 * l;
        if (!(l > 0)) l = 1;
        for (int d = 0; d < 3; d++) p.triNormal[3 * t + d] = float(n[d] / l);
        p.triArea[t] = float(a);
        area += a;
    }
    p.area = area;
}

std::vector<float> smoothNormals(const Part& p) {
    std::vector<float> N(3 * p.nVert, 0.f);
    for (int t = 0; t < p.nTri; t++)
        for (int c = 0; c < 3; c++) {
            const uint32_t v = p.tris[3 * t + c];
            for (int d = 0; d < 3; d++) N[3 * v + d] += p.triNormal[3 * t + d] * p.triArea[t];
        }
    for (int v = 0; v < p.nVert; v++) {
        double l = std::sqrt(double(N[3 * v]) * N[3 * v] + double(N[3 * v + 1]) * N[3 * v + 1] + double(N[3 * v + 2]) * N[3 * v + 2]);
        if (!(l > 0)) l = 1;
        for (int d = 0; d < 3; d++) N[3 * v + d] = float(N[3 * v + d] / l);
    }
    return N;
}

double signedVolume(const std::vector<float>& V, const std::vector<uint32_t>& T) {
    double vol = 0;
    for (size_t t = 0; t < T.size(); t += 3) {
        const size_t a = 3 * T[t], b = 3 * T[t + 1], c = 3 * T[t + 2];
        vol += double(V[a]) * (double(V[b + 1]) * V[c + 2] - double(V[b + 2]) * V[c + 1]) -
               double(V[a + 1]) * (double(V[b]) * V[c + 2] - double(V[b + 2]) * V[c]) +
               double(V[a + 2]) * (double(V[b]) * V[c + 1] - double(V[b + 1]) * V[c]);
    }
    return vol / 6;
}

std::vector<uint32_t> featureEdges(const Part& p, double angleDeg) {
    const double cosT = std::cos(angleDeg * M_PI / 180);
    std::vector<uint32_t> out;
    for (int t = 0; t < p.nTri; t++)
        for (int e = 0; e < 3; e++) {
            const int32_t o = p.neighbors[3 * t + e];
            if (o >= 0 && o < t) continue;  // each shared edge once
            bool sharp = o < 0;
            if (!sharp) {
                const double dot = p.triNormal[3 * t] * p.triNormal[3 * o] + p.triNormal[3 * t + 1] * p.triNormal[3 * o + 1] +
                                   p.triNormal[3 * t + 2] * p.triNormal[3 * o + 2];
                sharp = dot < cosT || (p.brepFaces && p.faceOf[t] != p.faceOf[o]);
            }
            if (sharp) {
                out.push_back(p.tris[3 * t + e]);
                out.push_back(p.tris[3 * t + (e + 1) % 3]);
            }
        }
    return out;
}

void creasedDisplay(Part& p, double angleDeg) {
    const double cosT = std::cos(angleDeg * M_PI / 180);
    std::vector<uint32_t> start(p.nVert + 1, 0);
    for (uint32_t v : p.tris) start[v + 1]++;
    for (int v = 0; v < p.nVert; v++) start[v + 1] += start[v];
    std::vector<uint32_t> fill(start.begin(), start.end() - 1), inc(p.tris.size());
    for (int t = 0; t < p.nTri; t++)
        for (int c = 0; c < 3; c++) inc[fill[p.tris[3 * t + c]]++] = uint32_t(t);
    p.displayPosition.assign(9 * size_t(p.nTri), 0.f);
    p.displayNormal.assign(9 * size_t(p.nTri), 0.f);
    p.displaySrc.assign(3 * size_t(p.nTri), 0);
    for (int t = 0; t < p.nTri; t++) {
        const float* nt = &p.triNormal[3 * t];
        for (int c = 0; c < 3; c++) {
            const uint32_t v = p.tris[3 * t + c];
            double sx = 0, sy = 0, sz = 0;
            for (uint32_t q = start[v]; q < start[v + 1]; q++) {
                const uint32_t s = inc[q];
                const float* ns = &p.triNormal[3 * s];
                if (nt[0] * ns[0] + nt[1] * ns[1] + nt[2] * ns[2] >= cosT) {
                    sx += ns[0] * p.triArea[s];
                    sy += ns[1] * p.triArea[s];
                    sz += ns[2] * p.triArea[s];
                }
            }
            double l = std::sqrt(sx * sx + sy * sy + sz * sz);
            if (!(l > 0)) l = 1;
            const size_t o = 9 * size_t(t) + 3 * c;
            p.displayNormal[o] = float(sx / l);
            p.displayNormal[o + 1] = float(sy / l);
            p.displayNormal[o + 2] = float(sz / l);
            for (int d = 0; d < 3; d++) p.displayPosition[o + d] = p.vertices[3 * v + d];
            p.displaySrc[3 * size_t(t) + c] = v;
        }
    }
}

int compactFaceIds(std::vector<int32_t>& faceOf) {
    std::unordered_map<int32_t, int32_t> map;
    for (auto& f : faceOf) {
        auto it = map.find(f);
        if (it == map.end()) it = map.emplace(f, int32_t(map.size())).first;
        f = it->second;
    }
    return int(map.size());
}

}  // namespace

BBox boundingBox(const std::vector<float>& V) {
    BBox b;
    b.min = {1e300, 1e300, 1e300};
    b.max = {-1e300, -1e300, -1e300};
    for (size_t i = 0; i < V.size(); i += 3)
        for (int d = 0; d < 3; d++) {
            b.min[d] = std::min(b.min[d], double(V[i + d]));
            b.max[d] = std::max(b.max[d], double(V[i + d]));
        }
    for (int d = 0; d < 3; d++) b.size[d] = b.max[d] - b.min[d];
    b.diag = norm(b.size);
    return b;
}

std::shared_ptr<Part> buildPart(const MeshSource& src, const BuildOptions& opts) {
    validate(src);
    Mesh m = weld(src);
    dropDegenerate(m);
    if (m.T.empty()) throw std::runtime_error("The file contains no usable triangles.");
    compactVertices(m);
    placeOnGround(m.V);
    orientShells(m.V, m.T);
    const BBox bbox = boundingBox(m.V);
    const double area0 = totalArea(m.V, m.T);
    const double target = std::max(bbox.diag / 140, std::sqrt(area0 / (0.4 * opts.maxTris)));
    refine(m, target, opts.maxTris);

    auto p = std::make_shared<Part>();
    p->name = src.name.empty() ? "Part" : src.name;
    p->vertices = std::move(m.V);
    p->tris = std::move(m.T);
    p->nVert = int(p->vertices.size() / 3);
    p->nTri = int(p->tris.size() / 3);
    p->bbox = bbox;
    p->brepFaces = !m.F.empty();
    computeTriangleData(*p);
    p->neighbors = triangleNeighbors(p->tris);
    p->vertNormal = smoothNormals(*p);
    if (p->brepFaces) {
        p->faceOf = std::move(m.F);
        p->faceCount = compactFaceIds(p->faceOf);
    } else {
        setSmoothFaces(*p, opts.faceAngle);
    }
    p->volume = std::abs(signedVolume(p->vertices, p->tris));
    p->edges = featureEdges(*p, 30);
    creasedDisplay(*p, 35);
    return p;
}

std::shared_ptr<Part> restorePart(const std::string& name, std::vector<float> vertices, std::vector<uint32_t> tris,
                                  std::vector<int32_t> faceOf, bool brepFaces, int faceCount, double faceAngle) {
    auto p = std::make_shared<Part>();
    p->name = name.empty() ? "Part" : name;
    p->vertices = std::move(vertices);
    p->tris = std::move(tris);
    p->nVert = int(p->vertices.size() / 3);
    p->nTri = int(p->tris.size() / 3);
    p->bbox = boundingBox(p->vertices);
    p->brepFaces = brepFaces && faceOf.size() == size_t(p->nTri);
    computeTriangleData(*p);
    p->neighbors = triangleNeighbors(p->tris);
    p->vertNormal = smoothNormals(*p);
    if (p->brepFaces) {
        p->faceOf = std::move(faceOf);
        p->faceCount = faceCount;
    } else {
        setSmoothFaces(*p, faceAngle > 0 ? faceAngle : 20);
    }
    p->volume = std::abs(signedVolume(p->vertices, p->tris));
    p->edges = featureEdges(*p, 30);
    creasedDisplay(*p, 35);
    return p;
}

void setSmoothFaces(Part& p, double angleDeg) {
    const double cosT = std::cos(angleDeg * M_PI / 180);
    p.faceOf.assign(p.nTri, -1);
    std::vector<int32_t> stack;
    int count = 0;
    for (int s = 0; s < p.nTri; s++) {
        if (p.faceOf[s] >= 0) continue;
        stack.clear();
        stack.push_back(s);
        p.faceOf[s] = count;
        while (!stack.empty()) {
            const int32_t t = stack.back();
            stack.pop_back();
            for (int e = 0; e < 3; e++) {
                const int32_t o = p.neighbors[3 * t + e];
                if (o < 0 || p.faceOf[o] >= 0) continue;
                const double dot = p.triNormal[3 * t] * p.triNormal[3 * o] + p.triNormal[3 * t + 1] * p.triNormal[3 * o + 1] +
                                   p.triNormal[3 * t + 2] * p.triNormal[3 * o + 2];
                if (dot >= cosT) {
                    p.faceOf[o] = count;
                    stack.push_back(o);
                }
            }
        }
        count++;
    }
    p.faceCount = count;
    p.faceAngle = angleDeg;
}

std::vector<int32_t> trianglesOfFace(const Part& p, int faceId) {
    std::vector<int32_t> out;
    for (int t = 0; t < p.nTri; t++)
        if (p.faceOf[t] == faceId) out.push_back(t);
    return out;
}

std::vector<int32_t> trianglesInSphere(const Part& p, int seed, const Vec3& c, double radius) {
    const double r2 = radius * radius;
    std::vector<uint8_t> seen(p.nTri, 0);
    std::vector<int32_t> out{seed};
    seen[seed] = 1;
    auto inside = [&](int t) {
        for (int k = 0; k < 3; k++) {
            const size_t v = 3 * size_t(p.tris[3 * t + k]);
            const double dx = p.vertices[v] - c[0], dy = p.vertices[v + 1] - c[1], dz = p.vertices[v + 2] - c[2];
            if (dx * dx + dy * dy + dz * dz <= r2) return true;
        }
        return false;
    };
    for (size_t q = 0; q < out.size(); q++) {
        const int32_t t = out[q];
        for (int e = 0; e < 3; e++) {
            const int32_t o = p.neighbors[3 * t + e];
            if (o >= 0 && !seen[o]) {
                seen[o] = 1;
                if (inside(o)) out.push_back(o);
            }
        }
    }
    return out;
}

Samples sampleTriangles(const Part& p, const std::vector<int32_t>& triList, double spacing, const std::optional<Clip>& clip) {
    Samples s;
    const double r2 = clip ? clip->radius * clip->radius : 0;
    for (int32_t t : triList) {
        const size_t a = 3 * size_t(p.tris[3 * t]), b = 3 * size_t(p.tris[3 * t + 1]), c = 3 * size_t(p.tris[3 * t + 2]);
        const int m = int(std::max(1.0, std::min(40.0, std::ceil(std::sqrt(p.triArea[t] / (spacing * spacing))))));
        const double w = p.triArea[t] / double(m * m);
        auto push = [&](double su, double u) {
            const double r = 1 - su - u;
            const double x = r * p.vertices[a] + su * p.vertices[b] + u * p.vertices[c];
            const double y = r * p.vertices[a + 1] + su * p.vertices[b + 1] + u * p.vertices[c + 1];
            const double z = r * p.vertices[a + 2] + su * p.vertices[b + 2] + u * p.vertices[c + 2];
            if (clip) {
                const double dx = x - clip->center[0], dy = y - clip->center[1], dz = z - clip->center[2];
                if (dx * dx + dy * dy + dz * dz > r2) return;
            }
            s.points.insert(s.points.end(), {float(x), float(y), float(z)});
            s.weights.push_back(float(w));
            s.tris.push_back(t);
        };
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m - i; j++) {
                push((i + 1.0 / 3) / m, (j + 1.0 / 3) / m);
                if (i + j < m - 1) push((i + 2.0 / 3) / m, (j + 2.0 / 3) / m);
            }
    }
    return s;
}

Frame patchFrame(const Part& p, const std::vector<int32_t>& triList, const std::optional<Clip>& clip) {
    Vec3 c{0, 0, 0}, n{0, 0, 0};
    double A = 0;
    for (int32_t t : triList) {
        const double w = p.triArea[t];
        for (int d = 0; d < 3; d++) {
            c[d] += w * (p.vertices[3 * p.tris[3 * t] + d] + p.vertices[3 * p.tris[3 * t + 1] + d] + p.vertices[3 * p.tris[3 * t + 2] + d]) / 3;
            n[d] += w * p.triNormal[3 * t + d];
        }
        A += w;
    }
    double l = norm(n);
    if (!(l > 0)) l = 1;
    for (auto& v : n) v /= l;
    if (clip) return {clip->center, n, A};
    for (auto& v : c) v /= (A > 0 ? A : 1);
    return {c, n, A};
}

std::vector<float> rotatePositions90(const std::vector<float>& positions, int axis) {
    std::vector<float> V(positions);
    for (size_t i = 0; i < V.size(); i += 3) {
        const float x = V[i], y = V[i + 1], z = V[i + 2];
        if (axis == 0) { V[i + 1] = -z; V[i + 2] = y; }
        else if (axis == 1) { V[i] = z; V[i + 2] = -x; }
        else { V[i] = -y; V[i + 1] = x; }
    }
    return V;
}

Vec3 scalePart(Part& p, double s) {
    if (!(s > 0) || !std::isfinite(s)) throw std::invalid_argument("Scale factor must be a positive number.");
    const Vec3 pivot{(p.bbox.min[0] + p.bbox.max[0]) / 2, p.bbox.min[1], (p.bbox.min[2] + p.bbox.max[2]) / 2};
    auto move = [&](std::vector<float>& arr) {
        for (size_t i = 0; i < arr.size(); i += 3)
            for (int d = 0; d < 3; d++) arr[i + d] = float(pivot[d] + (arr[i + d] - pivot[d]) * s);
    };
    move(p.vertices);
    move(p.displayPosition);
    for (auto& a : p.triArea) a = float(a * s * s);
    p.area *= s * s;
    p.volume *= s * s * s;
    p.bbox = boundingBox(p.vertices);
    p.wallThickness.reset();
    p.bvhCache.reset();
    return pivot;
}

}  // namespace ps
