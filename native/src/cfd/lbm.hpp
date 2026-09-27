// D3Q19 lattice-Boltzmann flow solver.
//
// - BGK collision with a Smagorinsky sub-grid model, so moderately high Reynolds numbers
//   stay stable on coarse grids (a simple LES).
// - Wind blows along +x. The inlet (x = 0) imposes the free stream. The outlet and the four
//   side faces are open boundaries: equilibrium at ambient pressure (rho = 1) with the
//   velocity of the nearest interior cell, so air displaced by the part can leave sideways as
//   it would in open air instead of being squeezed as in a closed tunnel (blockage).
// - Walls are no-slip. With wall links the wall sits at its true position on each lattice link
//   (Bouzidi interpolated bounce-back); without them, half-way bounce-back.
// Layout: f[i * N + cell], cell = x + nx * (y + ny * z). Macro output: [rho, ux, uy, uz] per cell.
// LbmGpu (lbm.wgsl) implements exactly the same scheme.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace ps {

constexpr int LBM_CX[19] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0};
constexpr int LBM_CY[19] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1};
constexpr int LBM_CZ[19] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1};
constexpr int LBM_OPP[19] = {0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17};
extern const double LBM_W[19];
constexpr int LBM_RAMP_STEPS = 400;

struct LbmSetup {
    std::array<int, 3> dims;
    std::vector<uint8_t> solid;  // per cell
    std::vector<uint8_t> links;  // 19 * N or empty (see wallLinks)
    double uLat = 0.08, nuLat = 0.01, smagorinsky = 0.16;
    // GPU: store the populations in 16 bits (shifted by their rest weights, "FP16S": Lehmann et al.
    // 2022) where the GPU supports it - half the memory traffic and size, computing still in 32 bits
    bool gpuHalf = true;
};

/** Inlet speed at a step (ramped up over the first LBM_RAMP_STEPS steps). */
double lbmInletVelocity(double uLat, int64_t step);

/** Common interface of the CPU and GPU solvers. */
class LbmSolver {
public:
    virtual ~LbmSolver() = default;
    virtual void reset() = 0;
    /** Advance `count` steps. */
    virtual void step(int count) = 0;
    /** [rho, ux, uy, uz] per cell after the last step (solid cells: 1, 0, 0, 0). */
    virtual std::vector<float> macro() = 0;
    int64_t steps = 0;
    int64_t cells = 0;
};

class LbmCpu : public LbmSolver {
public:
    explicit LbmCpu(const LbmSetup& s);
    void reset() override;
    void step(int count) override;
    std::vector<float> macro() override;

private:
    void stepOnce(bool writeMacro);
    std::array<int, 3> dims_;
    std::vector<uint8_t> solid_, links_;
    std::vector<float> f_, g_, macro_;
    int64_t off_[19];
    std::vector<std::pair<int, int>> runs_;  // [x0, x1) of cells with no solid neighbour, row by row
    std::vector<int64_t> runStart_;          // per row (y + ny z), its first run
    double uLat_, tau0_, smag_;
    bool macroFresh_ = false;
};

/** GPU solver; throws when the GPU is unavailable or the grid does not fit. */
std::unique_ptr<LbmSolver> makeLbmGpu(const LbmSetup& s);
/** Largest grid (cells) the GPU can hold (19 floats per cell in one storage buffer); 0 without a GPU. */
int64_t lbmGpuMaxCells(bool half = true);

/**
 * Exact wall distances for interpolated (Bouzidi) bounce-back: links[i * N + c] = 1 + round(254 q)
 * for the link from fluid cell c toward its solid neighbour c - c_i (q = fluid fraction of the
 * link); 0 means no link. positions: part vertices in the lattice (wind) frame, model units.
 */
std::vector<uint8_t> wallLinks(const std::vector<float>& positions, const std::vector<uint32_t>& tris, const std::array<double, 3>& origin, double h,
                               const std::array<int, 3>& dims, const std::vector<uint8_t>& solid, int* count = nullptr);

/**
 * Surface pressure integration on the voxel boundary of the solid: C = sum of Cp * n over wetted
 * voxel faces (cell^2); force [N] = C * (0.5 rho U^2) * h^2. frontal = projected solid area along x.
 */
struct ForceCoefficients {
    std::array<double, 3> C{0, 0, 0};
    double frontal = 0;
};
ForceCoefficients pressureForceCoefficients(const std::vector<float>& rho, const std::vector<uint8_t>& solid, const std::array<int, 3>& dims, double uLat);

}  // namespace ps
