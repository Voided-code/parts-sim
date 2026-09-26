#include "hex8.hpp"

#include <cmath>
#include <stdexcept>

namespace ps {

std::array<double, 36> elasticityMatrix(double nu) {
    if (!std::isfinite(nu) || nu <= -1 || nu >= 0.5) throw std::invalid_argument("Poisson's ratio must be greater than -1 and less than 0.5.");
    const double lam = nu / ((1 + nu) * (1 - 2 * nu)), mu = 1 / (2 * (1 + nu));
    std::array<double, 36> D{};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) D[i * 6 + j] = lam;
        D[i * 6 + i] = lam + 2 * mu;
    }
    D[3 * 6 + 3] = D[4 * 6 + 4] = D[5 * 6 + 5] = mu;
    return D;
}

static void fillStrainColumns(double* B, int cols, int c, double dx, double dy, double dz) {
    B[0 * cols + c] = dx;
    B[1 * cols + c + 1] = dy;
    B[2 * cols + c + 2] = dz;
    B[3 * cols + c] = dy;
    B[3 * cols + c + 1] = dx;
    B[4 * cols + c + 1] = dz;
    B[4 * cols + c + 2] = dy;
    B[5 * cols + c] = dz;
    B[5 * cols + c + 2] = dx;
}

std::array<double, 144> strainDisplacement(double xi, double eta, double zeta) {
    std::array<double, 144> B{};
    for (int a = 0; a < 8; a++) {
        const double sx = 2 * HEX_NODES[a][0] - 1, sy = 2 * HEX_NODES[a][1] - 1, sz = 2 * HEX_NODES[a][2] - 1;
        // dN/dx = dN/dxi * dxi/dx with dxi/dx = 2 on a unit cube
        const double dx = 2 * sx * (1 + sy * eta) * (1 + sz * zeta) / 8;
        const double dy = 2 * sy * (1 + sx * xi) * (1 + sz * zeta) / 8;
        const double dz = 2 * sz * (1 + sx * xi) * (1 + sy * eta) / 8;
        fillStrainColumns(B.data(), 24, 3 * a, dx, dy, dz);
    }
    return B;
}

// bubble modes 1 - xi^2, 1 - eta^2, 1 - zeta^2 per displacement component (6x9)
static std::array<double, 54> incompatibleModes(double xi, double eta, double zeta) {
    std::array<double, 54> G{};
    const double g[3][3] = {{-4 * xi, 0, 0}, {0, -4 * eta, 0}, {0, 0, -4 * zeta}};
    for (int m = 0; m < 3; m++) fillStrainColumns(G.data(), 9, 3 * m, g[m][0], g[m][1], g[m][2]);
    return G;
}

// C (r x c) += s * A^T (k x r) * B (k x c)
static void addAtB(double* C, const double* A, const double* B, int k, int r, int c, double s) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            double sum = 0;
            for (int q = 0; q < k; q++) sum += A[q * r + i] * B[q * c + j];
            C[i * c + j] += s * sum;
        }
}

static std::vector<double> matMul(const double* A, const double* B, int n, int k, int m) {
    std::vector<double> C(n * m, 0.0);
    for (int i = 0; i < n; i++)
        for (int q = 0; q < k; q++) {
            const double a = A[i * k + q];
            if (a == 0) continue;
            for (int j = 0; j < m; j++) C[i * m + j] += a * B[q * m + j];
        }
    return C;
}

static std::vector<double> invert(const std::vector<double>& M, int n) {
    std::vector<double> A = M, I(n * n, 0.0);
    for (int i = 0; i < n; i++) I[i * n + i] = 1;
    for (int col = 0; col < n; col++) {
        int piv = col;
        for (int r = col + 1; r < n; r++)
            if (std::abs(A[r * n + col]) > std::abs(A[piv * n + col])) piv = r;
        if (piv != col)
            for (int j = 0; j < n; j++) {
                std::swap(A[col * n + j], A[piv * n + j]);
                std::swap(I[col * n + j], I[piv * n + j]);
            }
        const double d = A[col * n + col];
        for (int j = 0; j < n; j++) {
            A[col * n + j] /= d;
            I[col * n + j] /= d;
        }
        for (int r = 0; r < n; r++) {
            if (r == col) continue;
            const double f = A[r * n + col];
            if (f == 0) continue;
            for (int j = 0; j < n; j++) {
                A[r * n + j] -= f * A[col * n + j];
                I[r * n + j] -= f * I[col * n + j];
            }
        }
    }
    return I;
}

HexElement hexElement(double nu, bool useModes) {
    const auto D = elasticityMatrix(nu);
    const double g = 1 / std::sqrt(3.0), detJ = 1.0 / 8;
    std::vector<double> Kuu(576, 0.0), Kua(24 * 9, 0.0), Kaa(81, 0.0);
    for (double xi : {-g, g})
        for (double eta : {-g, g})
            for (double zeta : {-g, g}) {
                const auto B = strainDisplacement(xi, eta, zeta);
                const auto DB = matMul(D.data(), B.data(), 6, 6, 24);
                addAtB(Kuu.data(), B.data(), DB.data(), 6, 24, 24, detJ);
                if (useModes) {
                    const auto G = incompatibleModes(xi, eta, zeta);
                    const auto DG = matMul(D.data(), G.data(), 6, 6, 9);
                    addAtB(Kua.data(), B.data(), DG.data(), 6, 24, 9, detJ);
                    addAtB(Kaa.data(), G.data(), DG.data(), 6, 9, 9, detJ);
                }
            }
    std::vector<double> A;  // alpha = A u (internal mode amplitudes), 9x24
    if (useModes) {
        const auto KaaInv = invert(Kaa, 9);
        std::vector<double> Kau(9 * 24);
        for (int i = 0; i < 24; i++)
            for (int q = 0; q < 9; q++) Kau[q * 24 + i] = Kua[i * 9 + q];
        A = matMul(KaaInv.data(), Kau.data(), 9, 9, 24);
        for (double& v : A) v = -v;
        const auto X = matMul(Kua.data(), A.data(), 24, 9, 24);
        for (int i = 0; i < 576; i++) Kuu[i] += X[i];
        for (int i = 0; i < 24; i++)
            for (int j = i + 1; j < 24; j++) {
                const double m = 0.5 * (Kuu[i * 24 + j] + Kuu[j * 24 + i]);
                Kuu[i * 24 + j] = Kuu[j * 24 + i] = m;
            }
    }
    auto enhanced = [&](double xi, double eta, double zeta) {
        auto Bt = strainDisplacement(xi, eta, zeta);
        if (!A.empty()) {
            const auto G = incompatibleModes(xi, eta, zeta);
            const auto GA = matMul(G.data(), A.data(), 6, 9, 24);
            for (int i = 0; i < 144; i++) Bt[i] += GA[i];
        }
        return Bt;
    };
    HexElement el;
    for (int i = 0; i < 576; i++) el.K[i] = Kuu[i];
    for (int a = 0; a < 8; a++) {
        const auto B = enhanced(2 * HEX_NODES[a][0] - 1, 2 * HEX_NODES[a][1] - 1, 2 * HEX_NODES[a][2] - 1);
        const auto S = matMul(D.data(), B.data(), 6, 6, 24);
        for (int i = 0; i < 144; i++) el.cornerStress[a][i] = S[i];
        el.gaussB[a] = enhanced((2 * HEX_NODES[a][0] - 1) * g, (2 * HEX_NODES[a][1] - 1) * g, (2 * HEX_NODES[a][2] - 1) * g);
    }
    return el;
}

}  // namespace ps
