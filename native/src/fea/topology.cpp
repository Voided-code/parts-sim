#include "topology.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_set>

#include "../util/parallel.hpp"

namespace ps {

namespace {

struct Filter {
    std::vector<int> start, nbr;
    std::vector<double> wts, W;
};

// neighbour lists of the density filter: weights max(0, rmin - distance) over solid voxels
Filter densityFilter(std::array<int, 3> dims, const std::vector<int>& ids, double rmin) {
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    const int r = int(std::ceil(rmin)) - 1;
    struct Off { int i, j, k; double w; };
    std::vector<Off> offs;
    for (int k = -r - 1; k <= r + 1; k++)
        for (int j = -r - 1; j <= r + 1; j++)
            for (int i = -r - 1; i <= r + 1; i++) {
                const double w = rmin - std::sqrt(double(i * i + j * j + k * k));
                if (w > 0) offs.push_back({i, j, k, w});
            }
    std::vector<int> pos(size_t(nx) * ny * nz, -1);
    for (size_t q = 0; q < ids.size(); q++) pos[ids[q]] = int(q);
    Filter F;
    F.start.assign(ids.size() + 1, 0);
    for (size_t q = 0; q < ids.size(); q++) {
        const int e = ids[q];
        const int i = e % nx, j = (e / nx) % ny, k = e / (nx * ny);
        for (const auto& o : offs) {
            const int a = i + o.i, b = j + o.j, c = k + o.k;
            if (a < 0 || b < 0 || c < 0 || a >= nx || b >= ny || c >= nz) continue;
            const int p = pos[a + nx * (b + ny * c)];
            if (p >= 0) { F.nbr.push_back(p); F.wts.push_back(o.w); }
        }
        F.start[q + 1] = int(F.nbr.size());
    }
    F.W.assign(ids.size(), 0.0);
    for (size_t q = 0; q < ids.size(); q++)
        for (int t = F.start[q]; t < F.start[q + 1]; t++) F.W[q] += F.wts[t];
    return F;
}

void applyFilter(const Filter& F, const std::vector<double>& x, std::vector<double>& out) {
    parallelFor(int64_t(x.size()), [&](int64_t lo, int64_t hi) {
        for (int64_t q = lo; q < hi; q++) {
            double s = 0;
            for (int t = F.start[q]; t < F.start[q + 1]; t++) s += F.wts[t] * x[F.nbr[t]];
            out[q] = s / F.W[q];
        }
    });
}

// sensitivities of a filtered quantity back to the design variables (the filter's transpose;
// the filter is symmetric in its weights, so this gathers instead of scattering)
void applyFilterT(const Filter& F, const std::vector<double>& g, std::vector<double>& out) {
    parallelFor(int64_t(g.size()), [&](int64_t lo, int64_t hi) {
        for (int64_t q = lo; q < hi; q++) {
            double s = 0;
            for (int t = F.start[q]; t < F.start[q + 1]; t++) {
                const int p = F.nbr[t];
                s += F.wts[t] * g[p] / F.W[p];
            }
            out[q] = s;
        }
    });
}

}  // namespace

std::vector<double> elementEnergies(const VoxelFEA& fea, const std::vector<double>& u) {
    const Level& L = fea.levels[0];
    const auto& K = fea.K0;
    std::vector<double> out(L.elems.size());
    parallelFor(int64_t(L.elems.size()), [&](int64_t lo, int64_t hi) {
        double ue[24];
        for (int64_t e = lo; e < hi; e++) {
            for (int a = 0; a < 8; a++) {
                const int64_t n = 3 * (L.base[e] + L.off[a]);
                ue[3 * a] = u[n]; ue[3 * a + 1] = u[n + 1]; ue[3 * a + 2] = u[n + 2];
            }
            double s = 0;
            for (int r = 0; r < 24; r++) {
                double t = 0;
                for (int c = 0; c < 24; c++) t += K[r * 24 + c] * ue[c];
                s += ue[r] * t;
            }
            out[e] = s;
        }
    }, 1024);
    return out;
}

TopologyResult optimizeTopology(const TopologyOptions& o) {
    std::vector<int> solid;
    for (size_t e = 0; e < o.fill.size(); e++)
        if (o.fill[e] > 0) solid.push_back(int(e));
    const size_t m = solid.size();
    const Filter F = densityFilter(o.dims, solid, o.rmin);
    std::vector<double> vol(m);
    std::vector<uint8_t> isFree(m, 0);
    double totalVol = 0, keptVol = 0;
    for (size_t q = 0; q < m; q++) {
        vol[q] = o.fill[solid[q]];
        totalVol += vol[q];
        if (o.keep[solid[q]]) keptVol += vol[q];
        else isFree[q] = 1;
    }
    const double target = std::max(o.volFrac * totalVol, keptVol * 1.02);
    // start uniform at the budget left after the kept voxels
    const double start = std::min(1.0, std::max(TOPO_XMIN, (target - keptVol) / std::max(1e-12, totalVol - keptVol)));
    std::vector<double> x(m), xPhys(m), dc(m), dcx(m), dv(m), dvx(m), xNew(m), phys(m);
    for (size_t q = 0; q < m; q++) x[q] = isFree[q] ? start : 1;
    std::vector<float> density(o.fill.size(), 0.f);
    std::vector<double> u0;
    std::vector<double> byVoxel(o.fill.size(), 0.0);
    TopologyResult res;
    double change = 1;
    const double penal = o.penal;
    for (int it = 0; it < o.maxIter; it++) {
        applyFilter(F, x, xPhys);
        for (size_t q = 0; q < m; q++)
            if (!isFree[q]) xPhys[q] = 1;
        for (size_t q = 0; q < m; q++)
            density[solid[q]] = float(std::min(1.0, o.fill[solid[q]] * (TOPO_XMIN + (1 - TOPO_XMIN) * std::pow(xPhys[q], penal))));
        auto sol = o.solve(density, u0.empty() ? nullptr : &u0);
        const auto energy = elementEnergies(*sol.fea, sol.u);
        // map by voxel index (voxels may have been pruned from the solved model)
        std::fill(byVoxel.begin(), byVoxel.end(), 0.0);
        const Level& L = sol.fea->levels[0];
        for (size_t t = 0; t < L.elems.size(); t++) byVoxel[L.elems[t]] = energy[t];
        double c = 0;
        for (size_t i = 0; i < sol.f.size(); i++) c += sol.f[i] * sol.u[i];
        double used = 0;
        for (size_t q = 0; q < m; q++) {
            dc[q] = -penal * (1 - TOPO_XMIN) * o.fill[solid[q]] * std::pow(xPhys[q], penal - 1) * byVoxel[solid[q]];
            dv[q] = vol[q];
            used += vol[q] * xPhys[q];
        }
        u0 = std::move(sol.u);
        res.history.push_back({it, c, used / totalVol, change});
        if (o.onIter && o.onIter(TopologyIter{it, c, used / totalVol, change, xPhys, solid})) break;
        if (it > 4 && change < o.tol) break;
        applyFilterT(F, dc, dcx);
        applyFilterT(F, dv, dvx);
        // optimality criteria: bisection on the volume multiplier
        double l1 = 0, l2 = 1e9;
        while ((l2 - l1) / (l1 + l2 + 1e-30) > 1e-4) {
            const double lm = 0.5 * (l1 + l2);
            for (size_t q = 0; q < m; q++) {
                if (!isFree[q]) { xNew[q] = 1; continue; }
                const double B = std::sqrt(std::max(0.0, -dcx[q]) / (lm * dvx[q] + 1e-300));
                xNew[q] = std::max(0.0, std::max(x[q] - o.move, std::min(1.0, std::min(x[q] + o.move, x[q] * B))));
            }
            applyFilter(F, xNew, phys);
            double v = 0;
            for (size_t q = 0; q < m; q++) v += vol[q] * (isFree[q] ? phys[q] : 1);
            if (v > target) l1 = lm;
            else l2 = lm;
        }
        change = 0;
        for (size_t q = 0; q < m; q++) {
            change = std::max(change, std::abs(xNew[q] - x[q]));
            x[q] = xNew[q];
        }
    }
    applyFilter(F, x, xPhys);
    for (size_t q = 0; q < m; q++)
        if (!isFree[q]) xPhys[q] = 1;
    res.density.assign(o.fill.size(), 0.f);
    for (size_t q = 0; q < m; q++) res.density[solid[q]] = float(xPhys[q]);
    res.keptFraction = keptVol / totalVol;
    return res;
}

TriMesh surfaceNets(const std::vector<float>& density, std::array<int, 3> dims, std::array<double, 3> origin, double h, double level, int smooth) {
    const int nx = dims[0], ny = dims[1], nz = dims[2];
    // padded sample grid: sample (a,b,c) = voxel (a-1,b-1,c-1) centre
    const int SX = nx + 2, SY = ny + 2, SZ = nz + 2;
    std::vector<float> f(size_t(SX) * SY * SZ, 0.f);
    for (int k = 0; k < nz; k++)
        for (int j = 0; j < ny; j++)
            for (int i = 0; i < nx; i++) f[(i + 1) + size_t(SX) * ((j + 1) + size_t(SY) * (k + 1))] = density[i + size_t(nx) * (j + size_t(ny) * k)];
    auto at = [&](int a, int b, int c) { return f[a + size_t(SX) * (b + size_t(SY) * c)]; };
    const int CX = SX - 1, CY = SY - 1, CZ = SZ - 1;
    std::vector<int> cellVert(size_t(CX) * CY * CZ, -1);
    std::vector<float> P;
    static const int corners[8][3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {1, 1, 0}, {0, 0, 1}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    static const int edges[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    float v[8];
    for (int c = 0; c < CZ; c++)
        for (int b = 0; b < CY; b++)
            for (int a = 0; a < CX; a++) {
                int inside = 0;
                for (int q = 0; q < 8; q++) {
                    v[q] = at(a + corners[q][0], b + corners[q][1], c + corners[q][2]);
                    if (v[q] > level) inside++;
                }
                if (inside == 0 || inside == 8) continue;
                double sx = 0, sy = 0, sz = 0;
                int cnt = 0;
                for (const auto& e : edges) {
                    const int p = e[0], q = e[1];
                    if ((v[p] > level) == (v[q] > level)) continue;
                    const double t = (level - v[p]) / (v[q] - v[p]);
                    sx += corners[p][0] + t * (corners[q][0] - corners[p][0]);
                    sy += corners[p][1] + t * (corners[q][1] - corners[p][1]);
                    sz += corners[p][2] + t * (corners[q][2] - corners[p][2]);
                    cnt++;
                }
                cellVert[a + size_t(CX) * (b + size_t(CY) * c)] = int(P.size() / 3);
                P.push_back(float(a + sx / cnt));
                P.push_back(float(b + sy / cnt));
                P.push_back(float(c + sz / cnt));
            }
    TriMesh out;
    auto cell = [&](int a, int b, int c) { return a < 0 || b < 0 || c < 0 || a >= CX || b >= CY || c >= CZ ? -1 : cellVert[a + size_t(CX) * (b + size_t(CY) * c)]; };
    // each grid edge between two samples with a sign change gets a quad of the 4 cells around it
    for (int c = 0; c < SZ; c++)
        for (int b = 0; b < SY; b++)
            for (int a = 0; a < SX; a++) {
                const bool inA = at(a, b, c) > level;
                for (int axis = 0; axis < 3; axis++) {
                    const int a2 = a + (axis == 0), b2 = b + (axis == 1), c2 = c + (axis == 2);
                    if (a2 >= SX || b2 >= SY || c2 >= SZ) continue;
                    if (inA == (at(a2, b2, c2) > level)) continue;
                    int q[4];
                    if (axis == 0) { q[0] = cell(a, b - 1, c - 1); q[1] = cell(a, b, c - 1); q[2] = cell(a, b, c); q[3] = cell(a, b - 1, c); }
                    else if (axis == 1) { q[0] = cell(a - 1, b, c - 1); q[1] = cell(a - 1, b, c); q[2] = cell(a, b, c); q[3] = cell(a, b, c - 1); }
                    else { q[0] = cell(a - 1, b - 1, c); q[1] = cell(a, b - 1, c); q[2] = cell(a, b, c); q[3] = cell(a - 1, b, c); }
                    if (q[0] < 0 || q[1] < 0 || q[2] < 0 || q[3] < 0) continue;
                    // orient outward: normals point from inside to outside
                    const uint32_t A = q[0], B = q[1], C = q[2], D = q[3];
                    if (inA) out.index.insert(out.index.end(), {A, B, C, A, C, D});
                    else out.index.insert(out.index.end(), {A, C, B, A, D, C});
                }
            }
    // Laplacian smoothing over the mesh graph
    if (smooth > 0 && !out.index.empty()) {
        const size_t nV = P.size() / 3;
        std::vector<std::vector<uint32_t>> nb(nV);
        auto link = [&](uint32_t i, uint32_t j) {
            if (std::find(nb[i].begin(), nb[i].end(), j) == nb[i].end()) nb[i].push_back(j);
        };
        for (size_t t = 0; t < out.index.size(); t += 3)
            for (int e = 0; e < 3; e++) {
                const uint32_t i = out.index[t + e], j = out.index[t + (e + 1) % 3];
                link(i, j);
                link(j, i);
            }
        std::vector<float> Q(P.size());
        for (int s = 0; s < smooth; s++) {
            for (size_t i = 0; i < nV; i++) {
                double x = 0, y = 0, z = 0;
                for (uint32_t j : nb[i]) { x += P[3 * j]; y += P[3 * j + 1]; z += P[3 * j + 2]; }
                const double c = nb[i].empty() ? 1 : double(nb[i].size());
                Q[3 * i] = float(0.5 * P[3 * i] + 0.5 * (x / c));
                Q[3 * i + 1] = float(0.5 * P[3 * i + 1] + 0.5 * (y / c));
                Q[3 * i + 2] = float(0.5 * P[3 * i + 2] + 0.5 * (z / c));
            }
            std::swap(P, Q);
        }
    }
    for (size_t i = 0; i < P.size(); i += 3)
        for (int d = 0; d < 3; d++) P[i + d] = float(origin[d] + (P[i + d] - 0.5) * h);
    out.positions = std::move(P);
    return out;
}

std::vector<uint8_t> toSTL(const TriMesh& mesh, const std::string& name) {
    const auto& P = mesh.positions;
    const uint32_t nT = uint32_t(mesh.index.size() / 3);
    std::vector<uint8_t> buf(84 + size_t(50) * nT, 0);
    const std::string head = (name + " - Parts Sim").substr(0, 79);
    for (size_t i = 0; i < head.size(); i++) buf[i] = uint8_t(head[i]) & 0x7f;
    std::memcpy(&buf[80], &nT, 4);  // little-endian hosts
    size_t o = 84;
    auto put = [&](float v) { std::memcpy(&buf[o], &v, 4); o += 4; };
    for (uint32_t t = 0; t < nT; t++) {
        const size_t a = 3 * mesh.index[3 * t], b = 3 * mesh.index[3 * t + 1], c = 3 * mesh.index[3 * t + 2];
        const double ux = P[b] - P[a], uy = P[b + 1] - P[a + 1], uz = P[b + 2] - P[a + 2];
        const double vx = P[c] - P[a], vy = P[c + 1] - P[a + 1], vz = P[c + 2] - P[a + 2];
        double nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        double l = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (!(l > 0)) l = 1;
        put(float(nx / l)); put(float(ny / l)); put(float(nz / l));
        for (size_t p : {a, b, c})
            for (int d = 0; d < 3; d++) put(P[p + d]);
        o += 2;
    }
    return buf;
}

}  // namespace ps
