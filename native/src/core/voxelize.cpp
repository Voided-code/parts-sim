#include "voxelize.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "../util/parallel.hpp"
#include "bvh.hpp"

namespace ps {

Grid gridForBox(const Vec3& min, const Vec3& max, int n, int pad) {
    Grid g;
    const Vec3 size{max[0] - min[0], max[1] - min[1], max[2] - min[2]};
    g.h = std::max({size[0], size[1], size[2]}) / n;
    for (int d = 0; d < 3; d++) {
        g.dims[d] = std::max(1, int(std::ceil(size[d] / g.h - 1e-6))) + 2 * pad;
        g.origin[d] = min[d] - (g.dims[d] * g.h - size[d]) / 2;
    }
    return g;
}

std::vector<float> voxelize(const std::vector<float>& pos, const std::vector<uint32_t>& index, const Grid& grid, int sub) {
    const int S[3] = {grid.dims[0] * sub, grid.dims[1] * sub, grid.dims[2] * sub};
    const double ds = grid.h / sub;
    std::vector<uint8_t> votes(size_t(S[0]) * S[1] * S[2], 0);
    const int64_t strides[3] = {1, S[0], int64_t(S[0]) * S[1]};
    const size_t nTri = index.size() / 3;
    // tiny irrational offsets keep rays off shared edges and vertices of axis-aligned CAD geometry
    const double jitter[3] = {0.000137 * ds, 0.000291 * ds, 0.000213 * ds};

    for (int a = 0; a < 3; a++) {
        const int u = (a + 1) % 3, v = (a + 2) % 3;
        const int Nu = S[u], Nv = S[v], Na = S[a];
        const double ou = grid.origin[u] + jitter[u], ov = grid.origin[v] + jitter[v];
        const size_t nRays = size_t(Nu) * Nv;
        std::vector<uint32_t> counts(nRays + 1, 0);
        std::vector<float> hits;
        // pass 0 counts hits per ray, pass 1 stores them
        for (int pass = 0; pass < 2; pass++) {
            std::vector<uint32_t> fill;
            if (pass == 1) fill.assign(counts.begin(), counts.end() - 1);
            for (size_t t = 0; t < nTri; t++) {
                const size_t i0 = 3 * size_t(index[3 * t]), i1 = 3 * size_t(index[3 * t + 1]), i2 = 3 * size_t(index[3 * t + 2]);
                const double au = pos[i0 + u], av = pos[i0 + v], aa = pos[i0 + a];
                const double bu = pos[i1 + u], bv = pos[i1 + v], ba = pos[i1 + a];
                const double cu = pos[i2 + u], cv = pos[i2 + v], ca = pos[i2 + a];
                const double det = (bv - cv) * (au - cu) + (cu - bu) * (av - cv);
                if (det == 0) continue;
                const int su0 = std::max(0, int(std::ceil((std::min({au, bu, cu}) - ou) / ds - 0.5)));
                const int su1 = std::min(Nu - 1, int(std::floor((std::max({au, bu, cu}) - ou) / ds - 0.5)));
                const int sv0 = std::max(0, int(std::ceil((std::min({av, bv, cv}) - ov) / ds - 0.5)));
                const int sv1 = std::min(Nv - 1, int(std::floor((std::max({av, bv, cv}) - ov) / ds - 0.5)));
                if (su0 > su1 || sv0 > sv1) continue;
                const double inv = 1 / det;
                for (int sv = sv0; sv <= sv1; sv++) {
                    const double pv = ov + (sv + 0.5) * ds;
                    for (int su = su0; su <= su1; su++) {
                        const double pu = ou + (su + 0.5) * ds;
                        const double l1 = ((bv - cv) * (pu - cu) + (cu - bu) * (pv - cv)) * inv;
                        if (l1 < 0) continue;
                        const double l2 = ((cv - av) * (pu - cu) + (au - cu) * (pv - cv)) * inv;
                        if (l2 < 0) continue;
                        const double l3 = 1 - l1 - l2;
                        if (l3 < 0) continue;
                        const size_t ray = size_t(su) + size_t(Nu) * sv;
                        if (pass == 0) counts[ray]++;
                        else hits[fill[ray]++] = float(l1 * aa + l2 * ba + l3 * ca);
                    }
                }
            }
            if (pass == 0) {
                uint32_t acc = 0;
                for (size_t r = 0; r < nRays; r++) {
                    const uint32_t c = counts[r];
                    counts[r] = acc;
                    acc += c;
                }
                counts[nRays] = acc;
                hits.assign(acc, 0.f);
            }
        }
        const double oa = grid.origin[a];
        parallelFor(int64_t(Nv), [&](int64_t lo, int64_t hi) {
            for (int64_t sv = lo; sv < hi; sv++)
                for (int su = 0; su < Nu; su++) {
                    const size_t ray = size_t(su) + size_t(Nu) * sv;
                    const uint32_t start = counts[ray], end = counts[ray + 1];
                    if (end - start < 2) continue;
                    std::sort(hits.begin() + start, hits.begin() + end);
                    const int64_t base = su * strides[u] + sv * strides[v];
                    for (uint32_t q = start; q + 1 < end; q += 2) {
                        const int s0 = std::max(0, int(std::ceil((hits[q] - oa) / ds - 0.5)));
                        const int s1 = std::min(Na - 1, int(std::floor((hits[q + 1] - oa) / ds - 0.5)));
                        for (int s = s0; s <= s1; s++) votes[base + s * strides[a]]++;
                    }
                }
        }, 1);
    }
    const int nx = grid.dims[0], ny = grid.dims[1], nz = grid.dims[2];
    std::vector<float> frac(size_t(nx) * ny * nz, 0.f);
    const double inv = 1.0 / (sub * sub * sub);
    parallelFor(int64_t(nz), [&](int64_t lo, int64_t hi) {
        for (int64_t k = lo; k < hi; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    int c = 0;
                    for (int dk = 0; dk < sub; dk++)
                        for (int dj = 0; dj < sub; dj++) {
                            const int64_t row = int64_t(i) * sub + S[0] * (int64_t(j) * sub + dj + int64_t(S[1]) * (k * sub + dk));
                            for (int di = 0; di < sub; di++) c += votes[row + di] >= 2;
                        }
                    frac[i + size_t(nx) * (j + size_t(ny) * k)] = float(c * inv);
                }
    }, 1);
    return frac;
}

namespace {

const BVH& bvhOf(const Part& part) {
    if (!part.bvhCache) part.bvhCache = std::make_shared<BVH>(part.vertices, part.tris);
    return *part.bvhCache;
}

// distance through the material behind triangle t (to the opposite face), or +inf beyond `far`
double wallThicknessAt(const Part& part, const BVH& bvh, int t, double far) {
    const auto& V = part.vertices;
    const auto& T = part.tris;
    const auto& N = part.triNormal;
    const double eps = part.bbox.diag * 1e-6;
    double c[3] = {0, 0, 0};
    for (int k = 0; k < 3; k++)
        for (int d = 0; d < 3; d++) c[d] += V[3 * T[3 * t + k] + d] / 3.0;
    const double n[3] = {N[3 * t], N[3 * t + 1], N[3 * t + 2]};
    const double o[3] = {c[0] - n[0] * eps, c[1] - n[1] * eps, c[2] - n[2] * eps};
    const double dir[3] = {-n[0], -n[1], -n[2]};
    RayHit hit;
    if (!bvh.raycast(o, dir, 0, far, hit) || hit.tri == t) return std::numeric_limits<double>::infinity();
    const int q = hit.tri;
    // the far side of a wall faces the other way
    if (N[3 * q] * n[0] + N[3 * q + 1] * n[1] + N[3 * q + 2] * n[2] > -0.3) return std::numeric_limits<double>::infinity();
    return hit.distance + eps;
}

// Akenine-Moller triangle / axis-aligned box overlap (box by centre and half size)
bool triBoxOverlap(const double c[3], double hs, const double a[3], const double b[3], const double d[3]) {
    double v0[3], v1[3], v2[3], e[3][3];
    for (int k = 0; k < 3; k++) {
        v0[k] = a[k] - c[k];
        v1[k] = b[k] - c[k];
        v2[k] = d[k] - c[k];
    }
    for (int k = 0; k < 3; k++) {
        e[0][k] = v1[k] - v0[k];
        e[1][k] = v2[k] - v1[k];
        e[2][k] = v0[k] - v2[k];
    }
    for (auto& ed : e)
        for (int k = 0; k < 3; k++) {
            double ax[3];
            if (k == 0) { ax[0] = 0; ax[1] = -ed[2]; ax[2] = ed[1]; }
            else if (k == 1) { ax[0] = ed[2]; ax[1] = 0; ax[2] = -ed[0]; }
            else { ax[0] = -ed[1]; ax[1] = ed[0]; ax[2] = 0; }
            const double p0 = ax[0] * v0[0] + ax[1] * v0[1] + ax[2] * v0[2];
            const double p1 = ax[0] * v1[0] + ax[1] * v1[1] + ax[2] * v1[2];
            const double p2 = ax[0] * v2[0] + ax[1] * v2[1] + ax[2] * v2[2];
            const double r = hs * (std::abs(ax[0]) + std::abs(ax[1]) + std::abs(ax[2]));
            if (std::min({p0, p1, p2}) > r || std::max({p0, p1, p2}) < -r) return false;
        }
    for (int k = 0; k < 3; k++)
        if (std::min({v0[k], v1[k], v2[k]}) > hs || std::max({v0[k], v1[k], v2[k]}) < -hs) return false;
    const double n[3] = {e[0][1] * e[1][2] - e[0][2] * e[1][1], e[0][2] * e[1][0] - e[0][0] * e[1][2], e[0][0] * e[1][1] - e[0][1] * e[1][0]};
    const double dist = n[0] * v0[0] + n[1] * v0[1] + n[2] * v0[2];
    return std::abs(dist) <= hs * (std::abs(n[0]) + std::abs(n[1]) + std::abs(n[2]));
}

}  // namespace

double typicalWallThickness(Part& part, int samples) {
    if (part.wallThickness) return *part.wallThickness;
    const BVH& bvh = bvhOf(part);
    const double far = part.bbox.diag * 0.15;
    std::vector<double> cdf(part.nTri);
    double acc = 0;
    for (int t = 0; t < part.nTri; t++) cdf[t] = acc += part.triArea[t];
    std::vector<int> picks(samples);
    uint32_t seed = 7;
    for (int s = 0; s < samples; s++) {
        seed = (seed * 1103515245u + 12345u) & 0x7fffffffu;
        const double r = double(seed) / 0x7fffffff * acc;
        picks[s] = int(std::lower_bound(cdf.begin(), cdf.end(), r) - cdf.begin());
        if (picks[s] >= part.nTri) picks[s] = part.nTri - 1;
    }
    std::vector<double> thick(samples);
    parallelFor(samples, [&](int64_t lo, int64_t hi) {
        for (int64_t s = lo; s < hi; s++) thick[s] = wallThicknessAt(part, bvh, picks[s], far);
    }, 64);
    std::sort(thick.begin(), thick.end());
    int finite = 0;
    for (double t : thick) finite += std::isfinite(t);
    // walls must cover a meaningful share of the surface to count
    part.wallThickness = finite > samples * 0.15 ? thick[int(samples * 0.2)] : std::numeric_limits<double>::infinity();
    return *part.wallThickness;
}

std::vector<float> thinWallDensity(const Part& part, const Grid& grid, double maxThickness) {
    const int nx = grid.dims[0], ny = grid.dims[1], nz = grid.dims[2];
    const double h = grid.h;
    std::vector<double> thick(part.nTri);
    const BVH& bvh = bvhOf(part);
    parallelFor(part.nTri, [&](int64_t lo, int64_t hi) {
        for (int64_t t = lo; t < hi; t++) thick[t] = wallThicknessAt(part, bvh, int(t), maxThickness);
    }, 256);
    std::vector<float> vol(size_t(nx) * ny * nz, 0.f);
    std::vector<int64_t> hit;
    for (int t = 0; t < part.nTri; t++) {
        if (!std::isfinite(thick[t])) continue;
        double a[3], b[3], d[3], c[3];
        for (int k = 0; k < 3; k++) {
            a[k] = part.vertices[3 * part.tris[3 * t] + k];
            b[k] = part.vertices[3 * part.tris[3 * t + 1] + k];
            d[k] = part.vertices[3 * part.tris[3 * t + 2] + k];
        }
        int lo[3], hi[3];
        for (int k = 0; k < 3; k++) {
            lo[k] = std::max(0, int(std::floor((std::min({a[k], b[k], d[k]}) - grid.origin[k]) / h)));
            hi[k] = std::min(grid.dims[k] - 1, int(std::floor((std::max({a[k], b[k], d[k]}) - grid.origin[k]) / h)));
        }
        hit.clear();
        for (int k = lo[2]; k <= hi[2]; k++)
            for (int j = lo[1]; j <= hi[1]; j++)
                for (int i = lo[0]; i <= hi[0]; i++) {
                    c[0] = grid.origin[0] + (i + 0.5) * h;
                    c[1] = grid.origin[1] + (j + 0.5) * h;
                    c[2] = grid.origin[2] + (k + 0.5) * h;
                    if (triBoxOverlap(c, h / 2, a, b, d)) hit.push_back(i + int64_t(nx) * (j + int64_t(ny) * k));
                }
        const double share = part.triArea[t] * thick[t] / 2 / std::max<size_t>(1, hit.size());
        for (int64_t e : hit) vol[e] += float(share);
    }
    std::vector<float> out(vol.size(), 0.f);
    const double h3 = h * h * h;
    parallelFor(int64_t(nz), [&](int64_t klo, int64_t khi) {
        for (int64_t k = klo; k < khi; k++)
            for (int j = 0; j < ny; j++)
                for (int i = 0; i < nx; i++) {
                    const size_t e = i + size_t(nx) * (j + size_t(ny) * k);
                    if (!vol[e]) continue;
                    double sum = 0;
                    int n = 0;
                    for (int dk = -1; dk <= 1; dk++)
                        for (int dj = -1; dj <= 1; dj++)
                            for (int di = -1; di <= 1; di++) {
                                const int ii = i + di, jj = j + dj;
                                const int64_t kk = k + dk;
                                if (ii < 0 || jj < 0 || kk < 0 || ii >= nx || jj >= ny || kk >= nz) continue;
                                const float v = vol[ii + size_t(nx) * (jj + size_t(ny) * kk)];
                                if (v) { sum += v; n++; }
                            }
                    out[e] = float(std::min(1.0, sum / (n * h3)));
                }
    }, 1);
    return out;
}

}  // namespace ps
