// The v1 flow engine's lattice, law of the wall and tunnel description (the web app's flow.js).
#include "flow.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

#include "../core/bvh.hpp"
#include "../core/voxelize.hpp"

namespace ps::flow {

const double W[19] = {1.0 / 3,  1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 18, 1.0 / 36, 1.0 / 36, 1.0 / 36,
                      1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};
const int CX[19] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
const int CY[19] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
const int CZ[19] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
const int OPP[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};

namespace {
// H_bba for a velocity with components a (odd) and b
double h3(int a, int b) { return a * (b * b - 1.0 / 3.0); }
std::array<std::array<double, 6>, 19> makeP3() {
    std::array<std::array<double, 6>, 19> c{};
    for (int i = 0; i < 19; i++) {
        const int x = CX[i], y = CY[i], z = CZ[i];
        const double xxy = h3(y, x), yzz = h3(y, z), xzz = h3(x, z), xyy = h3(x, y), yyz = h3(z, y), xxz = h3(z, x);
        const double p[6] = {xxy + yzz, xzz + xyy, yyz + xxz, xxy - yzz, xzz - xyy, yyz - xxz};
        // exact simple fractions: 13.5 P+ and 4.5 P- are multiples of 1.5
        for (int j = 0; j < 6; j++) c[i][j] = std::round((j < 3 ? 13.5 : 4.5) * p[j] * 2) / 2;
    }
    return c;
}
}  // namespace

const std::array<std::array<double, 6>, 19> P3C = makeP3();

double inletVelocity(double uLat, int64_t step) { return uLat * std::min(1.0, double(step + 1) / RAMP_STEPS); }

constexpr double KAPPA = 0.41, C_REICHARDT = 7.8;

std::array<double, 2> reichardt(double yp) {
    const double e11 = std::exp(-yp / 11), e3 = std::exp(-yp / 3);
    return {std::log(1 + KAPPA * yp) / KAPPA + C_REICHARDT * (1 - e11 - (yp / 11) * e3),
            1 / (1 + KAPPA * yp) + C_REICHARDT * (e11 / 11 - e3 / 11 + (yp / 33) * e3)};
}

double reichardtUtau(double ut, double y, double nu) {
    if (!(ut > 0) || !(y > 0) || !(nu > 0)) return 0;
    double utau = std::max(std::sqrt(nu * ut / y), ut / 30);
    for (int it = 0; it < 16; it++) {
        const double yp = y * utau / nu;
        const auto r = reichardt(yp);
        const double F = utau * r[0] - ut;
        const double next = utau - F / (r[0] + yp * r[1]);
        utau = next > 0 ? next : 0.5 * utau;
        if (std::abs(F) < 1e-7 * ut) break;
    }
    return utau;
}

namespace {

/** The wall model's sample cells and the part's surface next to each record (flow.js wallModelData). */
void wallModelData(const GridSpec& spec, const std::vector<float>& q, const std::vector<uint32_t>& tris, Grid& g,
                   const std::array<int, 3>& lo, const std::array<int, 3>& hi) {
    const int nx = spec.dims[0], ny = spec.dims[1], nz = spec.dims[2];
    auto& rec = g.rec;
    const int64_t n = rec.count;
    rec.samp.assign(n, 0);
    rec.y2.assign(n, 0);
    rec.area.assign(3 * n, 0);
    auto zOff = [&](int z, int dz) { return g.periodicZ ? (z + dz + nz) % nz : z + dz; };
    for (int64_t r = 0; r < n; r++) {
        const int64_t c = rec.cell[r];
        const int x = int(c % nx), y = int((c / nx) % ny), z = int(c / (int64_t(nx) * ny));
        const double nv[3] = {rec.normal[3 * r], rec.normal[3 * r + 1], rec.normal[3 * r + 2]};
        double best = 0;
        int bj = 0;
        for (int j = 1; j < 19; j++) {
            const double dot = (CX[j] * nv[0] + CY[j] * nv[1] + CZ[j] * nv[2]) / std::sqrt(double(CX[j] * CX[j] + CY[j] * CY[j] + CZ[j] * CZ[j]));
            if (dot <= best + 1e-6 || dot < 0.5) continue;
            const int xx = x + CX[j], yy = y + CY[j], zz = zOff(z, CZ[j]);
            if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) continue;
            if (g.kind[xx + int64_t(nx) * (yy + int64_t(ny) * zz)] != BULK) continue;
            best = dot;
            bj = j;
        }
        // on the part the model also needs MODEL_CLEAR cells of fluid along c_j (flow.js wallModelData)
        const bool belt = rec.groundMask[r] && rec.dist[r] == 0.5f;
        for (int k = 2; bj && !belt && k <= MODEL_CLEAR; k++) {
            const int xx = x + k * CX[bj], yy = y + k * CY[bj], zz = g.periodicZ ? ((z + k * CZ[bj]) % nz + nz) % nz : z + k * CZ[bj];
            if (xx < 0 || yy < 0 || zz < 0 || xx >= nx || yy >= ny || zz >= nz) break;
            const uint8_t kd = g.kind[xx + int64_t(nx) * (yy + int64_t(ny) * zz)];
            if (kd == WALL || kd == SOLID) bj = 0;
        }
        rec.samp[r] = uint8_t(bj);
        if (bj) rec.y2[r] = float(rec.dist[r] + CX[bj] * nv[0] + CY[bj] * nv[1] + CZ[bj] * nv[2]);
    }
    if (tris.empty() || !n) return;
    // records by cell within the part's box
    const std::array<int, 3> box{hi[0] - lo[0] + 2, hi[1] - lo[1] + 2, hi[2] - lo[2] + 2};
    std::vector<int32_t> at(size_t(box[0]) * box[1] * box[2], -1);
    auto boxIndex = [&](int x, int y, int z) -> int64_t {
        const int bx = x - lo[0] + 1, by = y - lo[1] + 1, bz = z - lo[2] + 1;
        if (bx < 0 || by < 0 || bz < 0 || bx >= box[0] || by >= box[1] || bz >= box[2]) return -1;
        return bx + int64_t(box[0]) * (by + int64_t(box[1]) * bz);
    };
    for (int64_t r = 0; r < n; r++) {
        const int64_t c = rec.cell[r];
        const int64_t i = boxIndex(int(c % nx), int((c / nx) % ny), int(c / (int64_t(nx) * ny)));
        if (i >= 0 && !(rec.groundMask[r] && rec.dist[r] == 0.5f)) at[i] = int32_t(r);
    }
    const double h = spec.h;
    for (size_t t = 0; t + 2 < tris.size(); t += 3) {
        double v[3][3];
        for (int k = 0; k < 3; k++)
            for (int a = 0; a < 3; a++) v[k][a] = (q[3 * tris[t + k] + a] - spec.origin[a]) / h;
        const double e1[3] = {v[1][0] - v[0][0], v[1][1] - v[0][1], v[1][2] - v[0][2]};
        const double e2[3] = {v[2][0] - v[0][0], v[2][1] - v[0][1], v[2][2] - v[0][2]};
        const double A[3] = {(e1[1] * e2[2] - e1[2] * e2[1]) / 2, (e1[2] * e2[0] - e1[0] * e2[2]) / 2, (e1[0] * e2[1] - e1[1] * e2[0]) / 2};
        const double len = std::sqrt(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
        if (!(len > 0)) continue;
        // the triangle's area is shared among the records along it: points on a grid of about half a
        // cell over the triangle, each to the nearest record just outside it (a large CAD face would
        // otherwise hang on one record); with a periodic span only the points inside it count
        const double l1 = std::sqrt(e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]), l2 = std::sqrt(e2[0] * e2[0] + e2[1] * e2[1] + e2[2] * e2[2]);
        const int m1 = std::clamp(int(std::ceil(l1 / 0.5)), 1, 4096), m2 = std::clamp(int(std::ceil(l2 / 0.5)), 1, 4096);
        int64_t inside = 0;
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < m2; j++) inside += (i + 0.5) / m1 + (j + 0.5) / m2 < 1;
        const bool centroid = inside == 0;
        const double w = centroid ? 1 : 1.0 / double(inside);
        auto assign = [&](double a, double b) {
            double p[3];
            for (int k = 0; k < 3; k++) p[k] = v[0][k] + a * e1[k] + b * e2[k] + 0.6 * A[k] / len;
            if (g.periodicZ && (p[2] - 0.6 * A[2] / len < 0 || p[2] - 0.6 * A[2] / len >= nz)) return;
            const int cx = int(std::floor(p[0])), cy = int(std::floor(p[1])), czz = int(std::floor(p[2]));
            int bestR = -1;
            double bestD = std::numeric_limits<double>::infinity();
            for (int dz = -1; dz <= 1; dz++)
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        const int zz = g.periodicZ ? ((czz + dz) % nz + nz) % nz : czz + dz;
                        const int64_t i = boxIndex(cx + dx, cy + dy, zz);
                        if (i < 0 || at[i] < 0) continue;
                        const double d = std::pow(cx + dx + 0.5 - p[0], 2) + std::pow(cy + dy + 0.5 - p[1], 2) + std::pow(czz + dz + 0.5 - p[2], 2);
                        if (d < bestD) { bestD = d; bestR = at[i]; }
                    }
            if (bestR < 0) return;
            for (int k = 0; k < 3; k++) rec.area[3 * bestR + k] += float(A[k] * w);
        };
        if (centroid) assign(1.0 / 3, 1.0 / 3);
        else
            for (int i = 0; i < m1; i++)
                for (int j = 0; j < m2; j++) {
                    const double a = (i + 0.5) / m1, b = (j + 0.5) / m2;
                    if (a + b < 1) assign(a, b);
                }
    }
}

}  // namespace

Grid buildGrid(const GridSpec& spec, const std::vector<float>& q, const std::vector<uint32_t>& tris) {
    Grid g;
    g.dims = spec.dims;
    const int nx = spec.dims[0], ny = spec.dims[1], nz = spec.dims[2];
    g.N = int64_t(nx) * ny * nz;
    g.ground = spec.ground;
    g.periodicZ = spec.periodicZ;
    g.h = spec.h;
    g.origin = spec.origin;
    auto& kind = g.kind;
    kind.assign(g.N, BULK);
    // the tunnel's faces, and the ground layer
    for (int z = 0; z < nz; z++)
        for (int y = 0; y < ny; y++) {
            const int64_t row = int64_t(nx) * (y + int64_t(ny) * z);
            const bool edge = y == 0 || y == ny - 1 || (!g.periodicZ && (z == 0 || z == nz - 1));
            if (g.ground && y == 0) std::fill(kind.begin() + row, kind.begin() + row + nx, SOLID);
            else if (edge) std::fill(kind.begin() + row, kind.begin() + row + nx, FACE);
            else { kind[row] = FACE; kind[row + nx - 1] = FACE; }
        }
    // the part, voxelized within its bounding box (one cell of margin)
    std::array<int, 3> lo, hi;
    for (int a = 0; a < 3; a++) {
        lo[a] = std::max(0, int(std::floor((spec.mn[a] - spec.origin[a]) / spec.h)) - 1);
        hi[a] = std::min(spec.dims[a], int(std::ceil((spec.mx[a] - spec.origin[a]) / spec.h)) + 1);
    }
    if (g.periodicZ) { lo[2] = 0; hi[2] = nz; }
    const std::array<int, 3> sub{hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    if (sub[0] > 0 && sub[1] > 0 && sub[2] > 0 && !tris.empty()) {
        ps::Grid box;  // the voxelizer's grid over the part's box
        box.origin = {spec.origin[0] + lo[0] * spec.h, spec.origin[1] + lo[1] * spec.h, spec.origin[2] + lo[2] * spec.h};
        box.h = spec.h;
        box.dims = sub;
        const auto frac = voxelize(q, tris, box, 2);
        for (int z = 0; z < sub[2]; z++)
            for (int y = 0; y < sub[1]; y++)
                for (int x = 0; x < sub[0]; x++) {
                    if (frac[x + size_t(sub[0]) * (y + size_t(sub[1]) * z)] < 0.5f) continue;
                    const int64_t c = lo[0] + x + int64_t(nx) * (lo[1] + y + int64_t(ny) * (lo[2] + z));
                    if (kind[c] != FACE) { kind[c] = SOLID; g.solidCount++; }
                }
    }
    // wall cells: interior fluid with a link blocked by the part (toward a solid cell or across the
    // surface: thin walls) or by the ground; thin-wall candidates lie within a cell of a triangle
    auto zOff = [&](int z, int dz) { return g.periodicZ ? (z + dz + nz) % nz : z + dz; };
    const std::array<int, 3> box{hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
    std::vector<uint8_t> near(std::max<size_t>(1, size_t(std::max(0, box[0])) * std::max(0, box[1]) * std::max(0, box[2])), 0);
    for (size_t t = 0; t + 2 < tris.size(); t += 3) {
        int a[3], b[3];
        for (int ax = 0; ax < 3; ax++) {
            double mn = 1e300, mx = -1e300;
            for (int v = 0; v < 3; v++) {
                const double p = (q[3 * tris[t + v] + ax] - spec.origin[ax]) / spec.h - 0.5;
                mn = std::min(mn, p);
                mx = std::max(mx, p);
            }
            a[ax] = std::max(lo[ax], int(std::floor(mn)) - 1) - lo[ax];
            b[ax] = std::min(hi[ax] - 1, int(std::ceil(mx)) + 1) - lo[ax];
        }
        for (int z = a[2]; z <= b[2]; z++)
            for (int y = a[1]; y <= b[1]; y++)
                for (int x = a[0]; x <= b[0]; x++) near[x + size_t(box[0]) * (y + size_t(box[1]) * z)] = 1;
    }
    const BVH bvh = tris.empty() ? BVH() : BVH(q, tris);
    auto& R = g.rec;
    uint8_t qb[18];
    auto scanCell = [&](int x, int y, int z, bool candidate) {
        const int64_t c = x + int64_t(nx) * (y + int64_t(ny) * z);
        if (kind[c] != BULK) return;
        uint32_t mask = 0, gmask = 0;
        double best = std::numeric_limits<double>::infinity();
        std::array<double, 3> bn{0, 0, 0};
        bool haveNormal = false;
        std::fill(qb, qb + 18, 0);
        for (int i = 1; i < 19; i++) {
            const int64_t s = x - CX[i] + int64_t(nx) * (y - CY[i] + int64_t(ny) * zOff(z, -CZ[i]));
            const bool solid = kind[s] == SOLID;
            if (solid && g.ground && y - CY[i] == 0) {
                mask |= 1u << i;
                gmask |= 1u << i;
                // the ground plane sits half-way between layers 0 and 1
                if (0.5 < best) { best = 0.5; bn = {0, 1, 0}; haveNormal = true; }
                continue;
            }
            if (!solid && !candidate) continue;
            RayHit hit;
            bool found = false;
            if (!bvh.empty()) {
                const double o[3] = {spec.origin[0] + (x + 0.5) * spec.h, spec.origin[1] + (y + 0.5) * spec.h, spec.origin[2] + (z + 0.5) * spec.h};
                const double d[3] = {-CX[i] * spec.h, -CY[i] * spec.h, -CZ[i] * spec.h};
                found = bvh.raycast(o, d, 0, 1, hit);
            }
            if (!found) { if (solid) mask |= 1u << i; continue; }
            mask |= 1u << i;
            const double qq = std::min(1.0, std::max(0.01, hit.distance));
            qb[i - 1] = uint8_t(1 + std::lround(254 * qq));
            // the hit triangle's normal and the perpendicular distance to its plane
            const uint32_t* tv = &tris[3 * size_t(hit.tri)];
            double e1[3], e2[3], nt[3];
            for (int a = 0; a < 3; a++) { e1[a] = q[3 * tv[1] + a] - q[3 * tv[0] + a]; e2[a] = q[3 * tv[2] + a] - q[3 * tv[0] + a]; }
            nt[0] = e1[1] * e2[2] - e1[2] * e2[1]; nt[1] = e1[2] * e2[0] - e1[0] * e2[2]; nt[2] = e1[0] * e2[1] - e1[1] * e2[0];
            const double l = std::sqrt(nt[0] * nt[0] + nt[1] * nt[1] + nt[2] * nt[2]);
            if (!(l > 0)) continue;
            for (double& v : nt) v /= l;
            const double cn = -CX[i] * nt[0] - CY[i] * nt[1] - CZ[i] * nt[2];
            const double dist = qq * std::abs(cn);
            if (dist < best) {
                best = dist;
                // the normal points into the fluid: against the link, which runs toward the wall
                const double sg = cn > 0 ? -1 : 1;
                bn = {sg * nt[0], sg * nt[1], sg * nt[2]};
                haveNormal = true;
            }
        }
        if (!mask) return;
        // interpolation (q < 1/2) needs the population from the next fluid cell away from the wall
        for (int i = 1; i < 19; i++) {
            if (!qb[i - 1] || (qb[i - 1] - 1) / 254.0 >= 0.5) continue;
            const int64_t n2 = x + CX[i] + int64_t(nx) * (y + CY[i] + int64_t(ny) * zOff(z, CZ[i]));
            if ((kind[n2] != BULK && kind[n2] != WALL) || (mask & (1u << OPP[i]))) qb[i - 1] = 128;
        }
        if (!haveNormal) {
            // no surface found (a staircase corner): half a cell toward the solid neighbours
            double sx = 0, sy = 0, sz = 0;
            for (int i = 1; i < 19; i++)
                if (mask & (1u << i)) { sx += CX[i]; sy += CY[i]; sz += CZ[i]; }
            const double l = std::sqrt(sx * sx + sy * sy + sz * sz);
            best = 0.5;
            bn = l > 0 ? std::array<double, 3>{sx / l, sy / l, sz / l} : std::array<double, 3>{0, 1, 0};
        }
        kind[c] = WALL;
        R.cell.push_back(uint32_t(c));
        R.mask.push_back(mask);
        R.groundMask.push_back(gmask);
        R.q.insert(R.q.end(), qb, qb + 18);
        for (double v : bn) R.normal.push_back(float(v));
        R.dist.push_back(float(std::max(0.05, best)));
    };
    const int x0 = std::max(1, lo[0]), x1 = std::min(nx - 1, hi[0]);
    const int y0 = std::max(1, lo[1]), y1 = std::min(ny - 1, hi[1]);
    const int z0 = g.periodicZ ? 0 : std::max(1, lo[2]), z1 = g.periodicZ ? nz : std::min(nz - 1, hi[2]);
    for (int z = z0; z < z1; z++)
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) scanCell(x, y, z, near[x - lo[0] + size_t(box[0]) * (y - lo[1] + size_t(box[1]) * (z - lo[2]))] == 1);
    // the cells above the ground layer, outside the part's box
    if (g.ground)
        for (int z = g.periodicZ ? 0 : 1; z < (g.periodicZ ? nz : nz - 1); z++)
            for (int x = 1; x < nx - 1; x++)
                if (!(x >= x0 && x < x1 && z >= z0 && z < z1 && y0 <= 1 && y1 > 1)) scanCell(x, 1, z, false);
    R.count = int64_t(R.cell.size());
    R.onBelt.assign(R.count, 0);
    for (int64_t r = 0; r < R.count; r++) R.onBelt[r] = R.groundMask[r] && R.dist[r] == 0.5f;
    wallModelData(spec, q, tris, g, lo, hi);
    return g;
}

Grid syntheticGrid(double n) {
    const int ny = std::max(16, int(std::lround(std::cbrt(n / 2.5)))), nz = ny, nx = std::max(16, int(std::lround(n / (double(ny) * nz))));
    const int s = ny / 4;
    GridSpec spec;
    spec.dims = {nx, ny, nz};
    spec.h = 1;
    spec.origin = {0, 0, 0};
    spec.mn = {double(nx / 4), double((ny - s) / 2), double((nz - s) / 2)};
    spec.mx = {spec.mn[0] + s, spec.mn[1] + s, spec.mn[2] + s};
    Grid g = buildGrid(spec, {}, {});
    // the cube's cells, and its wall cells (half-way links)
    for (int z = int(spec.mn[2]); z < int(spec.mx[2]); z++)
        for (int y = int(spec.mn[1]); y < int(spec.mx[1]); y++)
            for (int x = int(spec.mn[0]); x < int(spec.mx[0]); x++) { g.kind[x + int64_t(nx) * (y + int64_t(ny) * z)] = SOLID; g.solidCount++; }
    auto& R = g.rec;
    for (int z = int(spec.mn[2]) - 1; z <= int(spec.mx[2]); z++)
        for (int y = int(spec.mn[1]) - 1; y <= int(spec.mx[1]); y++)
            for (int x = int(spec.mn[0]) - 1; x <= int(spec.mx[0]); x++) {
                const int64_t c = x + int64_t(nx) * (y + int64_t(ny) * z);
                if (g.kind[c] != BULK) continue;
                uint32_t mask = 0;
                double sx = 0, sy = 0, sz = 0;
                for (int i = 1; i < 19; i++)
                    if (g.kind[c - CX[i] - int64_t(nx) * (CY[i] + int64_t(ny) * CZ[i])] == SOLID) { mask |= 1u << i; sx += CX[i]; sy += CY[i]; sz += CZ[i]; }
                if (!mask) continue;
                g.kind[c] = WALL;
                const double l = std::max(1e-9, std::sqrt(sx * sx + sy * sy + sz * sz));
                R.cell.push_back(uint32_t(c));
                R.mask.push_back(mask);
                R.groundMask.push_back(0);
                R.q.insert(R.q.end(), 18, uint8_t(0));
                R.normal.insert(R.normal.end(), {float(sx / l), float(sy / l), float(sz / l)});
                R.dist.push_back(0.5f);
            }
    R.count = int64_t(R.cell.size());
    R.onBelt.assign(R.count, 0);
    R.samp.assign(R.count, 0);
    R.y2.assign(R.count, 0);
    R.area.assign(3 * R.count, 0);
    return g;
}

}  // namespace ps::flow
