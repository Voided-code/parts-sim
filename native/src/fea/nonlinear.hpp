// Nonlinear static analysis: large displacements and metal plasticity.
//
// Geometry: element-independent co-rotational formulation. Each voxel's rigid rotation is taken
// from the polar decomposition of its mean deformation gradient; the (small) strain left after
// removing it is resisted by the same incompatible-mode element used in the linear study. That
// keeps the linear study's accuracy for bending and adds large rotations and stress stiffening.
//
// Material: von Mises (J2) plasticity with linear isotropic hardening from the yield strength to
// the tensile strength over the elongation at break, evaluated at the 8 Gauss points of each
// voxel (radial return). Brittle materials stay elastic and fail at their tensile strength.
//
// Equilibrium: Newton iterations under load control with adaptive increments. Each Newton step
// solves with the consistent tangent (rotated elastic stiffness + stress stiffness - plastic
// stiffness loss) by flexible conjugate gradients preconditioned with the elastic multigrid.
//
// Normalized units (E = 1, voxel = 1): displacement u~ = u / h, force f~ = f / (E h^2),
// stress s~ = s / E. These make K u~ = f~ exactly the linear solver's system.
#pragma once

#include <array>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "../core/materials.hpp"
#include "voxel_fea.hpp"

namespace ps {

/** Rotation part of a 3x3 matrix (row-major) by Higham's scaled Newton iteration. */
void polarRotation(const double* F, double* R);

struct Hardening {
    double yield, H, eu, uts;  // normalized by E (eu: elongation at break)
};
/** Linear hardening from yield to UTS at the elongation; nullopt for brittle materials. */
std::optional<Hardening> hardeningFor(const Material& m);

using Precond = std::function<void(const double* r, double* z)>;

struct FcgResult {
    std::vector<double> x;
    int iterations = 0;
    double residual = 0;
};
/** Flexible preconditioned conjugate gradients. */
FcgResult fcg(const std::function<void(const double*, double*)>& op, const Precond& precond, const std::vector<double>& b, double tol = 1e-3,
              int maxIter = 200);

struct NodalFields {
    std::vector<float> vm, p1, p3, pe;
    double maxP1 = 0;
};

class NonlinearModel {
public:
    NonlinearModel(const VoxelFEA& fea, bool largeDisplacement, std::optional<Hardening> plastic);

    /** Internal forces F_int(u~); updates the trial plastic state. Returns the largest plastic strain. */
    double internalForce(const std::vector<double>& u, std::vector<double>& out);
    /** y = J x with the rotations, stresses and plastic state of the last internalForce call. */
    void applyTangent(const double* x, double* y) const;
    void commit();
    /** Nodal von Mises / principal stresses (units of E) and plastic strain at u~. */
    NodalFields nodalFields(const std::vector<double>& u);

    const Level& L;
    bool large;
    std::optional<Hardening> plastic;
    std::vector<double> alpha;  // committed equivalent plastic strain per Gauss point

private:
    void localDisplacement(int64_t e, const std::vector<double>& u, double* ue, double* ul, bool storeRotation);
    template <class F> void forElements(F&& fn) const;

    const VoxelFEA& fea_;
    int64_t nE_;
    std::array<double, 36> D_;
    double G_;
    std::vector<double> R_, geo_, ep_, epTrial_, alphaTrial_, gpStress_, flow_;
    std::vector<uint8_t> yielding_, everPlastic_;
    std::array<std::vector<int64_t>, 8> colours_;  // voxels of the same parity share no nodes
};

struct EquilibriumResult {
    bool converged = false;
    int iterations = 0;
    double residual = 0, maxAlpha = 0;
};
/** Newton solve of F_int(u) = fLam from u (updated in place). onIter(it, rel) returns true to cancel. */
EquilibriumResult equilibrate(NonlinearModel& model, std::vector<double>& u, const std::vector<double>& fLam, const Precond& precond,
                              double alphaLimit = std::numeric_limits<double>::infinity(), const std::function<bool(int, double)>& onIter = {},
                              double tol = 1e-4, int maxNewton = 25);

struct RampStep {
    double lam;
    const std::vector<double>& u;
    NodalFields& fields;
    int iterations;
    double maxAlpha, maxDisp, maxStress;
};

struct RampOptions {
    static constexpr double inf = std::numeric_limits<double>::infinity();
    double target = 1;  // inf = until failure
    int steps = 10, maxSteps = 80;
    double maxDisp = inf, rupture = inf, crackStress = inf, stepAlpha = 0.01, stepDisp = inf, maxTime = inf;
    std::function<void(const RampStep&)> onStep;
    std::function<bool(double lam, int it, double rel)> onIter;  // true = cancel
};

struct RampResult {
    std::vector<double> u;
    double lam = 0;
    std::string reason;
};
/** Ramp the load factor from 0 with adaptive increments (see RampOptions::target). */
RampResult loadRamp(NonlinearModel& model, const std::vector<double>& f, const Precond& precond, const RampOptions& o);

}  // namespace ps
