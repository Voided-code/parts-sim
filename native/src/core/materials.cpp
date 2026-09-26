#include "materials.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace ps {

const std::vector<Material>& materials() {
    static const std::vector<Material> list = [] {
        auto M = [](const char* id, const char* name, double E, double nu, double ys, double uts, double rho, bool brittle, double el, double k,
                    double cp, double alpha, Fatigue f, double cost) {
            Material m;
            m.id = id; m.name = name; m.E = E; m.nu = nu; m.yield = ys; m.uts = uts; m.density = rho; m.brittle = brittle;
            m.elongation = el; m.k = k; m.cp = cp; m.alpha = alpha; m.fatigue = f; m.cost = cost;
            return m;
        };
        return std::vector<Material>{
            M("steel-1020", "Steel, AISI 1020", 200, 0.29, 351, 420, 7900, false, 0.25, 51.9, 486, 11.7, {210, 1e6, true, true}, 1.5),
            M("steel-alloy", "Alloy steel (4140 Q&T)", 205, 0.29, 655, 1020, 7850, false, 0.18, 42.6, 473, 12.3, {510, 1e6, true, true}, 2.5),
            M("ss-304", "Stainless steel 304", 193, 0.29, 215, 505, 8000, false, 0.45, 16.2, 500, 17.3, {240, 1e7, true, true}, 4),
            M("al-6061", "Aluminium 6061-T6", 69, 0.33, 276, 310, 2700, false, 0.12, 167, 896, 23.6, {96.5, 5e8, false, true}, 4),
            M("al-7075", "Aluminium 7075-T6", 71.7, 0.33, 503, 572, 2810, false, 0.11, 130, 960, 23.6, {159, 5e8, false, true}, 8),
            M("ti-64", "Titanium Ti-6Al-4V", 114, 0.34, 880, 950, 4430, false, 0.14, 6.7, 526, 8.6, {510, 1e7, true, true}, 45),
            M("brass", "Brass C360", 97, 0.31, 310, 385, 8500, false, 0.25, 115, 380, 20.5, {138, 1e8, false, true}, 8),
            M("copper", "Copper C110 (annealed)", 115, 0.33, 69, 220, 8900, false, 0.45, 388, 385, 17, {76, 1e8, false, true}, 10),
            M("cast-iron", "Grey cast iron", 110, 0.26, 150, 150, 7200, true, 0.005, 46, 490, 11, {69, 1e7, true, true}, 1.2),
            M("pla", "PLA (3D printed, 100% infill)", 3.5, 0.36, 45, 50, 1240, true, 0.05, 0.13, 1800, 68, {15, 1e7, false, false}, 20),
            M("petg", "PETG (3D printed)", 2.1, 0.38, 47, 50, 1270, false, 0.1, 0.2, 1200, 60, {15, 1e7, false, false}, 22),
            M("abs", "ABS", 2.2, 0.35, 40, 44, 1050, false, 0.1, 0.17, 1400, 90, {11, 1e7, false, false}, 20),
            M("nylon", "Nylon PA6/66", 2.8, 0.39, 70, 80, 1140, false, 0.4, 0.25, 1700, 90, {25, 1e7, false, false}, 30),
            M("pc", "Polycarbonate", 2.4, 0.37, 62, 70, 1200, false, 0.6, 0.2, 1200, 68, {14, 1e7, false, false}, 35),
            M("cfrp", "Carbon fibre (quasi-isotropic)", 60, 0.3, 570, 570, 1600, true, 0.01, 5, 900, 2, {200, 1e7, false, false}, 60),
            M("wood", "Pine wood (along grain, approx.)", 9, 0.3, 40, 40, 500, true, 0.01, 0.12, 1700, 5, {12, 1e7, false, false}, 1),
            M("glass", "Soda-lime glass", 70, 0.22, 45, 45, 2500, true, 0.001, 1, 840, 9, {15, 1e7, false, false}, 1.5),
        };
    }();
    return list;
}

const Material* findMaterial(const std::string& id) {
    for (const auto& m : materials())
        if (m.id == id) return &m;
    return nullptr;
}

void validateMaterial(const Material& m) {
    const std::pair<double, const char*> positive[] = {{m.E, "Young's modulus"}, {m.yield, "Yield strength"}, {m.uts, "Tensile strength"}, {m.density, "Density"}};
    for (auto& [v, name] : positive)
        if (!std::isfinite(v) || v <= 0) throw std::invalid_argument(std::string(name) + " must be a positive finite number.");
    if (!std::isfinite(m.nu) || m.nu <= -1 || m.nu >= 0.5) throw std::invalid_argument("Poisson's ratio must be greater than -1 and less than 0.5.");
    if (!m.brittle && m.uts < m.yield) throw std::invalid_argument("Tensile strength must be at least the yield strength for a ductile material.");
}

}  // namespace ps
