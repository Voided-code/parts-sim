// Fatigue (stress-life) evaluation of a linear static result under repeated loading.
//
// S-N curve: a straight line in log-log from the tensile strength at one cycle, through
// 0.9 UTS at 1000 cycles, to the fatigue strength Se at Ne cycles (Basquin), with Se reduced
// by a surface-finish factor for metals (Marin, Shigley's Mechanical Engineering Design).
// Below Se, materials with an endurance limit (steel, titanium) last indefinitely; others keep
// following the line. Mean stress: Goodman correction for tensile means; compressive means are
// not credited. The equivalent stress is von Mises signed by the dominant principal stress.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "../core/materials.hpp"

namespace ps {

/** (id, label) of the surface finishes. */
const std::vector<std::pair<std::string, std::string>>& fatigueFinishes();

struct SNCurve {
    double uts, Se, Ne, S3, b, ka;
    bool endurance;
};

SNCurve snCurve(const Material& m, const std::string& finish = "machined");
/** Strength [MPa] at N cycles. */
double strengthAt(const SNCurve& c, double N);
/** Cycles to failure at a fully reversed equivalent amplitude S [MPa] (inf = infinite life). */
double cyclesToFailure(const SNCurve& c, double S);
/** Goodman equivalent fully reversed amplitude; inf when the mean alone reaches UTS. */
double goodman(double sa, double sm, double uts);

struct FatigueField {
    SNCurve curve;
    std::vector<float> life, damage, fos;
    double minLife, minFos, strengthAtLife;
    int worst = -1;
};

/**
 * Per-vertex fatigue results from the stress at the peak of the load cycle.
 * vm, p1, p3 [Pa]; R = min / max load ratio (-1 fully reversed, 0 zero-based); cycles = design life.
 */
FatigueField fatigueField(const std::vector<float>& vm, const std::vector<float>& p1, const std::vector<float>& p3, const Material& m,
                          const std::string& finish, double R, double cycles, double scale = 1);

}  // namespace ps
