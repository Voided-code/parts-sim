#include "explicit.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "../util/parallel.hpp"
#include "eigen.hpp"

namespace ps {

double maxEigenvalue(const VoxelFEA& fea, const std::vector<double>& mass, int iterations) {
    const Level& L = fea.levels[0];
    const int64_t n = L.nDof;
    std::vector<double> v(n), w(n);
    uint32_t s = 99;
    for (int64_t i = 0; i < n; i++) {
        s = (s * 1103515245u + 12345u) & 0x7fffffffu;
        v[i] = mass[i] > 0 && !L.fixed[i] ? double(s) / 0x7fffffff - 0.5 : 0;
    }
    double lam = 0;
    for (int it = 0; it < iterations; it++) {
        double nv = std::sqrt(dot(v.data(), v.data(), n));
        if (!(nv > 0)) nv = 1;
        for (auto& x : v) x /= nv;
        fea.apply(0, v.data(), w.data());
        for (int64_t i = 0; i < n; i++) w[i] = mass[i] > 0 ? w[i] / mass[i] : 0;
        lam = std::sqrt(dot(w.data(), w.data(), n));
        std::swap(v, w);
    }
    return lam;
}

DropSetup dropSetup(const VoxelFEA& fea, const DropOptions& o) {
    const Level& L = fea.levels[0];
    const int64_t n = L.nDof, nN = L.nNodes;
    const auto Mn = lumpedMass(fea);
    DropSetup d;
    const double lamMax = maxEigenvalue(fea, Mn) * 1.05;
    d.wMax = std::sqrt(lamMax * o.E / (o.rho * o.h * o.h));
    const double xiHigh = 0.1;  // damping ratio at the highest frequency
    d.beta = 2 * xiHigh / d.wMax;
    d.dt = 0.9 * (2 / d.wMax) * (std::sqrt(1 + xiHigh * xiHigh) - xiHigh);
    d.m.resize(n);
    for (int64_t i = 0; i < n; i++) d.m[i] = o.rho * o.h * o.h * o.h * Mn[i];
    // penalty contact: each node's contact frequency is half the highest element frequency
    d.kc.assign(nN, 0.0);
    d.cc.assign(nN, 0.0);
    for (int64_t q = 0; q < nN; q++) {
        if (!o.surface[q] || !(d.m[3 * q + 1] > 0)) continue;
        d.kc[q] = 0.25 * d.wMax * d.wMax * d.m[3 * q + 1];
        d.cc[q] = 2 * 0.05 * std::sqrt(d.kc[q] * d.m[3 * q + 1]);
    }
    return d;
}

DropResult dropTestCPU(const VoxelFEA& fea, const DropOptions& o) {
    const Level& L = fea.levels[0];
    const int64_t n = L.nDof, nN = L.nNodes;
    const DropSetup d = dropSetup(fea, o);
    const double dt = d.dt, beta = d.beta;
    const auto& m = d.m;
    std::vector<double> u(n, 0.0), v(n, 0.0), w(n), q(n), a(n);
    for (int64_t i = 1; i < n; i += 3)
        if (m[i] > 0) v[i] = -o.speed;
    const double Eh = o.E * o.h;
    bool contactStarted = false;
    double contactEnd = -1, contactStart = 0, peakForce = 0, t = 0;
    DropResult r;
    r.vmMax.assign(nN, 0.f);
    r.tPeak.assign(nN, 0.f);
    const int sampleEvery = 4;
    double lastFrame = -INFINITY;
    auto forces = [&] {
        // internal + damping forces, contact, gravity -> acceleration
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) w[i] = u[i] + beta * v[i]; });
        fea.apply(0, w.data(), q.data(), true);
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) a[i] = m[i] > 0 ? -Eh * q[i] / m[i] : 0; });
        double fc = 0;
        for (int64_t k = 0; k < nN; k++) {
            const int64_t i = 3 * k + 1;
            if (!(m[i] > 0)) continue;
            a[i] -= o.g;
            if (d.kc[k] > 0) {
                const double pen = -(o.nodeY[k] + u[i]);
                if (pen > 0) {
                    const double f = std::max(0.0, d.kc[k] * pen - d.cc[k] * v[i]);
                    a[i] += f / m[i];
                    fc += f;
                }
            }
        }
        return fc;
    };
    double fc = forces();
    for (int64_t i = 0; i < n; i++) v[i] += 0.5 * dt * a[i];
    auto sampleStress = [&] {
        auto st = fea.stresses(u, o.E, o.h);
        for (int64_t k = 0; k < nN; k++)
            if (st.nodeVM[k] > r.vmMax[k]) { r.vmMax[k] = st.nodeVM[k]; r.tPeak[k] = float(t); }
        return st;
    };
    double estEnd = o.maxTime;
    int step = 1;
    for (; step <= o.maxSteps; step++) {
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) u[i] += dt * v[i]; });
        t += dt;
        fc = forces();
        parallelFor(n, [&](int64_t lo, int64_t hi) { for (int64_t i = lo; i < hi; i++) v[i] += dt * a[i]; });
        if (fc > 0 && !contactStarted) { contactStarted = true; contactStart = t; }
        peakForce = std::max(peakForce, fc);
        if (contactStarted && fc > 0) contactEnd = -1;
        else if (contactStarted && fc == 0 && contactEnd < 0) contactEnd = t;
        // stop a while after the part has left the floor (or bounced off it)
        if (contactEnd > 0 && estEnd == o.maxTime) estEnd = std::min(o.maxTime, contactEnd + std::max(0.3 * (contactEnd - contactStart), 50 * dt));
        if (contactEnd < 0 && estEnd != o.maxTime) estEnd = o.maxTime;  // touched down again
        const bool done = t >= estEnd || step == o.maxSteps;
        if (step % sampleEvery == 0 || done) {
            auto st = sampleStress();
            r.history.push_back({t, fc});
            const double frameGap = std::max(dt, (std::isfinite(estEnd) ? estEnd : std::max(t, 1e-6) * 2) / o.frames);
            if (o.onFrame && (t - lastFrame >= frameGap || done)) {
                lastFrame = t;
                o.onFrame({t, std::vector<float>(u.begin(), u.end()), std::move(st.nodeVM), fc});
            }
            if (o.onProgress && o.onProgress(std::min(0.99, std::isfinite(estEnd) ? t / estEnd : double(step) / o.maxSteps)))
                throw std::runtime_error("Cancelled");
        }
        if (done) break;
    }
    r.dt = dt;
    r.steps = std::min(step, o.maxSteps);
    r.duration = t;
    r.peakForce = peakForce;
    r.contactTime = contactStarted ? (contactEnd > 0 ? contactEnd : t) - contactStart : 0;
    r.wMax = d.wMax;
    r.rebounded = contactEnd > 0;
    return r;
}

}  // namespace ps
