// Minimal test harness: TEST("name") { ... CHECK(cond) ... } and a main that runs them all.
#pragma once

#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace pstest {

struct Case {
    const char* name;
    std::function<void()> fn;
};

inline std::vector<Case>& cases() {
    static std::vector<Case> c;
    return c;
}

inline int& failures() {
    static int f = 0;
    return f;
}

struct Register {
    Register(const char* name, std::function<void()> fn) { cases().push_back({name, std::move(fn)}); }
};

inline double ms(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace pstest

#define PS_CAT2(a, b) a##b
#define PS_CAT(a, b) PS_CAT2(a, b)
#define TEST(name)                                                                   \
    static void PS_CAT(test_fn_, __LINE__)();                                        \
    static pstest::Register PS_CAT(test_reg_, __LINE__)(name, PS_CAT(test_fn_, __LINE__)); \
    static void PS_CAT(test_fn_, __LINE__)()

#define CHECK(cond)                                                                             \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            std::printf("    FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond);                   \
            pstest::failures()++;                                                               \
        }                                                                                       \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                    \
    do {                                                                                         \
        const double _a = (a), _b = (b);                                                         \
        if (!(std::abs(_a - _b) <= (tol))) {                                                     \
            std::printf("    FAILED %s:%d: %s = %.9g, expected %.9g +- %.3g\n", __FILE__, __LINE__, #a, _a, _b, double(tol)); \
            pstest::failures()++;                                                                \
        }                                                                                        \
    } while (0)

#define TEST_MAIN                                                                     \
    int main() {                                                                      \
        int failedCases = 0;                                                          \
        for (auto& c : pstest::cases()) {                                             \
            const int before = pstest::failures();                                    \
            const auto t0 = std::chrono::steady_clock::now();                         \
            c.fn();                                                                   \
            const bool ok = pstest::failures() == before;                             \
            if (!ok) failedCases++;                                                   \
            std::printf("%s %s (%.0f ms)\n", ok ? "ok  " : "FAIL", c.name, pstest::ms(t0)); \
        }                                                                             \
        std::printf("%d/%zu passed\n", int(pstest::cases().size()) - failedCases, pstest::cases().size()); \
        return failedCases ? 1 : 0;                                                   \
    }
