// Linear dynamic response by modal superposition (from a modal study).
//
// Each mode i (natural frequency w_i, mass-normalized shape phi_i) obeys
//   q_i'' + 2 z w_i q_i' + w_i^2 q_i = G_i g(t),   G_i = phi_i^T f
// for a load pattern f scaled in time by g(t). The response uses the mode-acceleration method:
//   u = u_static g(t) + sum_i phi_i (q_i - G_i g / w_i^2)
// so the modes that were not computed still contribute their static share exactly (u_static is
// K^-1 f). Stresses superpose the same way. For base shaking the pattern is the inertia load
// f = -M r of a unit ground acceleration along r, and u is the motion relative to the base.
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace ps {

struct ModalBasis {
    std::vector<double> omegas;               // rad/s
    std::vector<double> gamma;                // G_i for the load pattern
    std::vector<std::vector<float>> modeU;    // per mode, vertex displacement per unit modal coordinate [model units]
    std::vector<std::vector<float>> modeS;    // per mode, vertex stress tensor per unit modal coordinate [Pa]
    std::vector<float> staticU, staticS;      // vertex displacement / stress under the pattern (g = 1)
};

struct HarmonicField {
    std::vector<float> disp, vm;
};
/** Peak (over the cycle) displacement magnitude and von Mises at every vertex (or the listed ones) at W [rad/s]. */
HarmonicField harmonicField(const ModalBasis& b, double W, double zeta, int phases = 16, const std::vector<int>* vertices = nullptr);
/** Real displacement field at one phase angle (for animating the steady vibration). */
std::vector<float> harmonicShape(const ModalBasis& b, double W, double zeta, double phase);

struct Excitation {
    std::string kind = "pulse";  // pulse, sine, step, quake
    double amplitude = 1, freq = 10, duration = 0.01, total = 0.05;
};
/** Excitation time function g(t) (dimensionless multiplier of the load pattern). */
std::function<double(double)> excitation(const Excitation& e);

struct TimeHistory {
    std::vector<double> times, g;
    std::vector<std::vector<float>> dyn;  // per mode: q_i(t) - G_i g(t) / w_i^2
    double dt = 0;
};
/** Newmark (average acceleration) per mode on a step of at most 1/24 of the shortest period. */
TimeHistory timeHistory(const ModalBasis& b, const std::function<double(double)>& g, double zeta, double total, int maxSteps = 20000);
/** Vertex displacement (3 per vertex) at time index k. */
std::vector<float> shapeAt(const ModalBasis& b, const TimeHistory& h, size_t k);
/** von Mises at every vertex (or the listed ones) at time index k. */
std::vector<float> stressAt(const ModalBasis& b, const TimeHistory& h, size_t k, const std::vector<int>* vertices = nullptr);

}  // namespace ps
