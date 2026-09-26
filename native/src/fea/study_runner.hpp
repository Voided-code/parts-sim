// The studies beyond linear static, engine side: frequency (and the modal basis for linear
// dynamics), buckling, nonlinear static, drop test, topology optimization and material/size
// optimization. Each solves on the voxel grid and maps its results onto the part's surface
// vertices, so the UI only has to draw. Runs on a worker thread; progress callbacks may cancel.
#pragma once

#include <array>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "../core/materials.hpp"
#include "response.hpp"
#include "static_study.hpp"
#include "structural.hpp"

namespace ps {

// ---------- frequency ----------

struct ModalMode {
    double freq;
    std::array<double, 3> eff;  // effective mass fraction along x, y, z
    std::vector<float> shape;   // per vertex, largest length 1
};

struct DynamicLoad {
    bool base = false;  // base shaking along dir (else the load pattern in.f)
    std::array<double, 3> dir{0, 1, 0};
};

struct ModalRun {
    std::vector<ModalMode> modes;
    bool free = false, converged = false;
    int iterations = 0, removed = 0;
    double totalMass = 0;  // kg
    std::string engine, gpuNote;
    bool hasBasis = false;
    ModalBasis basis;
};
ModalRun runModal(const StructuralModel& m, const StructuralInput& in, int nev, const DynamicLoad* dynamic, double toMeters, const ProgressFn& progress);

// ---------- buckling ----------

struct BucklingRun {
    struct Mode { double factor; std::vector<float> shape; };  // factor inf: does not buckle
    std::vector<Mode> modes;
    std::vector<float> vm;  // pre-buckling von Mises [Pa] per vertex
    bool converged = false;
    int iterations = 0, removed = 0;
    std::string engine, gpuNote;
};
BucklingRun runBuckling(const StructuralModel& m, const StructuralInput& in, int nev, const ProgressFn& progress);

// ---------- nonlinear static ----------

struct NonlinearOptions {
    bool largeDisplacement = true, plasticity = true, untilFailure = false;
    int steps = 10;
};

struct NonlinearStep {
    bool unloaded = false;
    double lam, D, maxVM, maxPE, maxDisp;  // D: load-weighted displacement along the loads [model units]
    std::vector<float> u, vm, pe, p1;       // per vertex (u in model units, stresses in Pa)
    int iterations;
};

struct NonlinearRun {
    std::string reason;
    double lam = 0;
    int steps = 0, removed = 0;
    bool permanent = false, plastic = false;
    std::string engine, gpuNote;
};
NonlinearRun runNonlinear(const StructuralModel& m, const StructuralInput& in, const Material& mat, const NonlinearOptions& o, double toMeters,
                          const std::function<void(NonlinearStep&&)>& onStep, const ProgressFn& progress);

// ---------- drop test ----------

struct DropFrame {
    double t, force;
    std::vector<float> u, vm;  // per vertex (model units, Pa)
};

struct DropRun {
    std::vector<float> vmMax, tPeak;  // per vertex
    std::vector<float> times, forces;
    double speed = 0, mass = 0, peakForce = 0, contactTime = 0, duration = 0, dt = 0;
    int steps = 0;
    bool rebounded = false;
    std::string engine, gpuNote;
};
DropRun runDrop(const StructuralModel& m, const StructuralInput& in, double height, double toMeters, const std::function<void(DropFrame&&)>& onFrame,
                const ProgressFn& progress);

// ---------- topology ----------

struct TopologyStep {
    int it;
    double compliance, volume, change;
    std::vector<float> density;  // per voxel
};

struct TopologyRun {
    std::vector<float> density;
    struct Point { int it; double compliance, volume; };
    std::vector<Point> history;
    double keptFraction = 0;
    std::string engine, gpuNote;
};
TopologyRun runTopology(const StructuralModel& m, const StructuralInput& in, const std::vector<uint8_t>& keep, double volFrac, int maxIter,
                        const std::function<void(TopologyStep&&)>& onIter, const ProgressFn& progress);

// ---------- material & size ----------

struct LoadGroup {
    std::string name, kind;  // kind: force, pressure, gravity
    std::vector<double> f;
};

struct SizingRow {
    std::string id, name;
    bool feasible = false;
    double scale = 0, fos = 0, disp = 0, mass = 0, cost = 0;
    bool exact = true;
};

struct SizingOptions {
    std::vector<LoadGroup> groups;
    std::vector<Material> materials;
    Material base;
    double fosTarget = 2, maxDisp = 0;  // maxDisp in model units, 0 = any
    double scaleLo = 0.25, scaleHi = 4;
    double volume = 0;                  // part volume at scale 1 [model units^3]
};

struct SizingRun {
    std::vector<SizingRow> rows;
    SizingRow base;
    int removed = 0;
    std::string engine, gpuNote;
};
SizingRun runSizing(const StructuralModel& m, const StructuralInput& in, const SizingOptions& o, double toMeters, const ProgressFn& progress);

}  // namespace ps
