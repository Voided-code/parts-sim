#include "fatigue.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace ps {

namespace {
// Marin surface factor ka = a * UTS[MPa]^b
const std::map<std::string, std::pair<double, double>> SURFACE = {
    {"ground", {1.58, -0.085}}, {"machined", {4.51, -0.265}}, {"hot-rolled", {57.7, -0.718}}, {"forged", {272, -0.995}}};
constexpr double INF = std::numeric_limits<double>::infinity();
}  // namespace

const std::vector<std::pair<std::string, std::string>>& fatigueFinishes() {
    static const std::vector<std::pair<std::string, std::string>> list = {
        {"polished", "Polished"}, {"ground", "Ground"}, {"machined", "Machined / cold drawn"}, {"hot-rolled", "Hot rolled"}, {"forged", "As forged"}};
    return list;
}

SNCurve snCurve(const Material& m, const std::string& finish) {
    const double uts = m.uts;
    Fatigue fat = m.fatigue;
    if (!(fat.Se > 0)) fat = {0.4 * uts, 1e7, false, false};
    double ka = 1;
    auto s = SURFACE.find(finish);
    if (fat.metal && s != SURFACE.end()) ka = std::min(1.0, s->second.first * std::pow(uts, s->second.second));
    SNCurve c;
    c.uts = uts;
    c.Se = std::max(1e-3, fat.Se * ka);
    c.Ne = fat.Ne;
    c.S3 = std::max(c.Se * 1.001, std::min(0.9 * uts, uts));  // strength at 1000 cycles
    c.b = std::log10(c.Se / c.S3) / std::log10(c.Ne / 1e3);    // negative slope
    c.ka = ka;
    c.endurance = fat.endurance;
    return c;
}

double strengthAt(const SNCurve& c, double N) {
    if (N <= 1) return c.uts;
    if (N <= 1e3) return c.uts * std::pow(c.S3 / c.uts, std::log10(N) / 3);
    if (N >= c.Ne && c.endurance) return c.Se;
    return c.S3 * std::pow(N / 1e3, c.b);
}

double cyclesToFailure(const SNCurve& c, double S) {
    if (!(S > 0)) return INF;
    if (S >= c.uts) return 1;
    if (S >= c.S3) return std::pow(10, 3 * std::log10(S / c.uts) / std::log10(c.S3 / c.uts));
    if (c.endurance && S <= c.Se) return INF;
    return 1e3 * std::pow(S / c.S3, 1 / c.b);
}

double goodman(double sa, double sm, double uts) {
    if (sm <= 0) return sa;
    if (sm >= uts) return INF;
    return sa / (1 - sm / uts);
}

FatigueField fatigueField(const std::vector<float>& vm, const std::vector<float>& p1, const std::vector<float>& p3, const Material& m,
                          const std::string& finish, double R, double cycles, double scale) {
    FatigueField out;
    out.curve = snCurve(m, finish);
    const auto& c = out.curve;
    const size_t n = vm.size();
    out.life.resize(n);
    out.damage.resize(n);
    out.fos.resize(n);
    out.strengthAtLife = strengthAt(c, cycles);
    out.minLife = INF;
    out.minFos = INF;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (size_t v = 0; v < n; v++) {
        if (std::isnan(vm[v])) { out.life[v] = out.damage[v] = out.fos[v] = nan; continue; }
        auto sgn = [](double x) { return x > 0 ? 1.0 : x < 0 ? -1.0 : 1.0; };
        const double sign = std::abs(p1[v]) >= std::abs(p3[v]) ? sgn(p1[v]) : sgn(p3[v]);
        const double smax = sign * vm[v] * scale / 1e6;  // MPa
        const double sa = std::abs(smax) * (1 - R) / 2, sm = smax * (1 + R) / 2;
        const double N = cyclesToFailure(c, goodman(sa, sm, c.uts));
        out.life[v] = float(N);
        out.damage[v] = std::isfinite(N) ? float(cycles / N) : 0.f;
        const double denom = sa / out.strengthAtLife + std::max(0.0, sm) / c.uts;
        out.fos[v] = denom > 0 ? float(1 / denom) : float(INF);
        if (N < out.minLife) { out.minLife = N; out.worst = int(v); }
        out.minFos = std::min(out.minFos, double(out.fos[v]));
    }
    return out;
}

}  // namespace ps
