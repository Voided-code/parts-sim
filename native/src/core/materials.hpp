// Material library. Typical room-temperature properties; real values vary with alloy, temper,
// print settings and supplier ("Custom" lets you enter datasheet numbers).
#pragma once

#include <stdexcept>
#include <string>
#include <vector>

namespace ps {

struct Fatigue {
    double Se = 0;      // fatigue strength [MPa] at Ne cycles (fully reversed, polished)
    double Ne = 1e7;
    bool endurance = false;  // no failure below Se (steel, titanium)
    bool metal = false;      // surface-finish factors apply
};

struct Material {
    std::string id, name;
    double E = 200;        // GPa
    double nu = 0.3;
    double yield = 250;    // MPa
    double uts = 400;      // MPa
    double density = 7850; // kg/m^3
    bool brittle = false;  // failure by max principal stress instead of von Mises
    double elongation = 0.15;  // strain at break
    double k = 50;         // W/m K
    double cp = 500;       // J/kg K
    double alpha = 12;     // 1/K x 1e-6
    Fatigue fatigue;
    double cost = 5;       // rough USD/kg
};

const std::vector<Material>& materials();
const Material* findMaterial(const std::string& id);
/** Throws std::invalid_argument when a property is missing or impossible. */
void validateMaterial(const Material& m);

}  // namespace ps
