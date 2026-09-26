// 8-node hexahedral ("brick") element on a unit cube with E = 1.
//
// Every voxel is the same cube, so the element matrices are computed once and scaled per voxel.
// Wilson's incompatible-mode element: 9 internal bubble modes are statically condensed out, so
// the matrix stays 24x24 but bending no longer locks when only a few voxels span a wall.
#pragma once

#include <array>
#include <vector>

namespace ps {

// Node order: bottom face counter-clockwise, then top face.
constexpr int HEX_NODES[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};

/** Isotropic elasticity matrix (6x6 row-major) for E = 1; Voigt xx, yy, zz, xy, yz, zx (engineering shear). */
std::array<double, 36> elasticityMatrix(double nu);

/** Strain-displacement matrix B (6x24) at natural coordinates in [-1, 1] on a unit cube. */
std::array<double, 144> strainDisplacement(double xi, double eta, double zeta);

struct HexElement {
    std::array<double, 576> K{};                       // 24x24 condensed stiffness
    std::array<std::array<double, 144>, 8> cornerStress{};  // D * Bbar at each corner (6x24)
    std::array<std::array<double, 144>, 8> gaussB{};       // Bbar at the Gauss point nearest each corner
};

/** Unit-cube element; the physical matrix of a voxel of side h and modulus E is E*h*K. */
HexElement hexElement(double nu, bool incompatibleModes = true);

}  // namespace ps
