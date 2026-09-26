#include "response.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include "../util/parallel.hpp"

namespace ps {

namespace {

double vonMises6(const double* s) {
    return std::sqrt(0.5 * ((s[0] - s[1]) * (s[0] - s[1]) + (s[1] - s[2]) * (s[1] - s[2]) + (s[2] - s[0]) * (s[2] - s[0])) +
                     3 * (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]));
}

// complex modal coefficients (re, im) at forcing frequency W, including the static correction
void harmonicCoefficients(const ModalBasis& b, double W, double zeta, std::vector<double>& re, std::vector<double>& im) {
    re.clear();
    im.clear();
    for (size_t i = 0; i < b.omegas.size(); i++) {
        const double w = b.omegas[i];
        const double a = w * w - W * W, c = 2 * zeta * w * W, d = a * a + c * c;
        const double G = b.gamma[i];
        re.push_back(G * (a / d - 1 / (w * w)));
        im.push_back(G * (-c / d));
    }
}

}  // namespace

HarmonicField harmonicField(const ModalBasis& b, double W, double zeta, int phases, const std::vector<int>* vertices) {
    std::vector<double> re, im;
    harmonicCoefficients(b, W, zeta, re, im);
    const int64_t nV = int64_t(b.staticU.size() / 3);
    const int64_t count = vertices ? int64_t(vertices->size()) : nV;
    HarmonicField out;
    out.disp.resize(count);
    out.vm.resize(count);
    std::vector<double> cs(phases), sn(phases);
    for (int p = 0; p < phases; p++) { cs[p] = std::cos(M_PI * p / phases); sn[p] = std::sin(M_PI * p / phases); }
    const size_t m = re.size();
    parallelFor(count, [&](int64_t lo, int64_t hi) {
        double ur[3], ui[3], sr[6], si[6], s[6];
        for (int64_t q = lo; q < hi; q++) {
            const int64_t v = vertices ? (*vertices)[q] : q;
            for (int d = 0; d < 3; d++) { ur[d] = b.staticU[3 * v + d]; ui[d] = 0; }
            for (int c = 0; c < 6; c++) { sr[c] = b.staticS[6 * v + c]; si[c] = 0; }
            for (size_t i = 0; i < m; i++) {
                const auto& U = b.modeU[i];
                const auto& S = b.modeS[i];
                for (int d = 0; d < 3; d++) { ur[d] += re[i] * U[3 * v + d]; ui[d] += im[i] * U[3 * v + d]; }
                for (int c = 0; c < 6; c++) { sr[c] += re[i] * S[6 * v + c]; si[c] += im[i] * S[6 * v + c]; }
            }
            if (std::isnan(ur[0])) { out.disp[q] = out.vm[q] = std::numeric_limits<float>::quiet_NaN(); continue; }
            // |u| amplitude: the largest length over the cycle of Re(u e^{i t})
            double best = 0, bestS = 0;
            for (int p = 0; p < phases; p++) {
                const double x = ur[0] * cs[p] - ui[0] * sn[p], y = ur[1] * cs[p] - ui[1] * sn[p], z = ur[2] * cs[p] - ui[2] * sn[p];
                best = std::max(best, x * x + y * y + z * z);
                for (int c = 0; c < 6; c++) s[c] = sr[c] * cs[p] - si[c] * sn[p];
                bestS = std::max(bestS, vonMises6(s));
            }
            out.disp[q] = float(std::sqrt(best));
            out.vm[q] = float(bestS);
        }
    }, 1024);
    return out;
}

std::vector<float> harmonicShape(const ModalBasis& b, double W, double zeta, double phase) {
    std::vector<double> re, im;
    harmonicCoefficients(b, W, zeta, re, im);
    const double c = std::cos(phase), s = std::sin(phase);
    std::vector<float> u(b.staticU.size());
    for (size_t t = 0; t < u.size(); t++) u[t] = float(b.staticU[t] * c);
    for (size_t i = 0; i < re.size(); i++) {
        const double k = re[i] * c - im[i] * s;
        const auto& U = b.modeU[i];
        for (size_t t = 0; t < u.size(); t++) u[t] += float(k * U[t]);
    }
    return u;
}

std::function<double(double)> excitation(const Excitation& e) {
    const double A = e.amplitude, f = e.freq, dur = e.duration, total = e.total;
    if (e.kind == "sine") return [=](double t) { return t <= total ? A * std::sin(2 * M_PI * f * t) : 0.0; };
    if (e.kind == "step") return [=](double t) { return t >= 0 ? A : 0.0; };
    if (e.kind == "quake") {
        // synthetic ground motion: filtered noise (1-10 Hz band) under a build-up / decay envelope
        uint32_t seed = 2024;
        auto rnd = [&] { seed = (seed * 1103515245u + 12345u) & 0x7fffffffu; return double(seed) / 0x7fffffff - 0.5; };
        struct Comp { double f, ph, a; };
        std::vector<Comp> comps;
        for (int k = 0; k < 40; k++) {
            const double ph = 2 * M_PI * rnd();
            comps.push_back({1 + 9.0 * k / 39, ph, 0.5 + rnd()});
        }
        auto raw = [comps, total](double t) {
            const double env = t < 0.15 * total ? std::pow(t / (0.15 * total), 2) : std::exp(-3 * (t - 0.15 * total) / total);
            double s = 0;
            for (const auto& c : comps) s += c.a * std::sin(2 * M_PI * c.f * t + c.ph);
            return env * s;
        };
        double peak = 0;
        for (double t = 0; t < total; t += total / 2000) peak = std::max(peak, std::abs(raw(t)));
        if (!(peak > 0)) peak = 1;
        return [=](double t) { return t >= 0 && t <= total ? A * raw(t) / peak : 0.0; };
    }
    // half-sine shock pulse
    return [=](double t) { return t >= 0 && t <= dur ? A * std::sin(M_PI * t / dur) : 0.0; };
}

TimeHistory timeHistory(const ModalBasis& b, const std::function<double(double)>& g, double zeta, double total, int maxSteps) {
    const double wMax = *std::max_element(b.omegas.begin(), b.omegas.end());
    TimeHistory h;
    h.dt = std::max(total / maxSteps, std::min(2 * M_PI / wMax / 24, total / 400));
    const size_t steps = size_t(std::ceil(total / h.dt));
    h.times.resize(steps + 1);
    h.g.resize(steps + 1);
    for (size_t k = 0; k <= steps; k++) { h.times[k] = k * h.dt; h.g[k] = g(k * h.dt); }
    const double dt = h.dt;
    for (size_t i = 0; i < b.omegas.size(); i++) {
        const double w = b.omegas[i], G = b.gamma[i], c = 2 * zeta * w, k = w * w;
        const double kh = k + 2 * c / dt + 4 / (dt * dt), A = 4 / dt + 2 * c;
        std::vector<float> out(steps + 1);
        double x = 0, v = 0, acc = G * h.g[0];
        out[0] = float(-(G * h.g[0]) / k);
        for (size_t n = 0; n < steps; n++) {
            const double dp = G * (h.g[n + 1] - h.g[n]) + A * v + 2 * acc;
            const double dx = dp / kh;
            const double dv = 2 * dx / dt - 2 * v;
            const double da = 4 * dx / (dt * dt) - 4 * v / dt - 2 * acc;
            x += dx; v += dv; acc += da;
            out[n + 1] = float(x - G * h.g[n + 1] / k);
        }
        h.dyn.push_back(std::move(out));
    }
    return h;
}

std::vector<float> shapeAt(const ModalBasis& b, const TimeHistory& h, size_t k) {
    const double gk = h.g[k];
    std::vector<float> u(b.staticU.size());
    for (size_t t = 0; t < u.size(); t++) u[t] = float(b.staticU[t] * gk);
    for (size_t i = 0; i < h.dyn.size(); i++) {
        const float c = h.dyn[i][k];
        if (c == 0) continue;
        const auto& U = b.modeU[i];
        for (size_t t = 0; t < u.size(); t++) u[t] += c * U[t];
    }
    return u;
}

std::vector<float> stressAt(const ModalBasis& b, const TimeHistory& h, size_t k, const std::vector<int>* vertices) {
    const double gk = h.g[k];
    const int64_t nV = int64_t(b.staticS.size() / 6);
    const int64_t count = vertices ? int64_t(vertices->size()) : nV;
    std::vector<float> vm(count);
    std::vector<double> coef;
    for (const auto& q : h.dyn) coef.push_back(q[k]);
    parallelFor(count, [&](int64_t lo, int64_t hi) {
        double s[6];
        for (int64_t q = lo; q < hi; q++) {
            const int64_t v = vertices ? (*vertices)[q] : q;
            for (int c = 0; c < 6; c++) s[c] = b.staticS[6 * v + c] * gk;
            for (size_t i = 0; i < coef.size(); i++) {
                const double a = coef[i];
                if (a == 0) continue;
                const auto& S = b.modeS[i];
                for (int c = 0; c < 6; c++) s[c] += a * S[6 * v + c];
            }
            vm[q] = std::isnan(s[0]) ? std::numeric_limits<float>::quiet_NaN() : float(vonMises6(s));
        }
    }, 2048);
    return vm;
}

}  // namespace ps
