// Opt-in stage timings for profiling: run with PARTS_SIM_TRACE=1 to print them to stderr.
// Other diagnostics: PARTS_SIM_PROFILE=1 prints per-level solver kernel timings before a bend test,
// PARTS_SIM_LBM_FP32=1 keeps the GPU flow solver's populations in 32 bits, PARTS_SIM_NO_GPU=1 and
// PARTS_SIM_THREADS=n limit the engines. tests/bench.cpp times every solver (BENCH_PROFILE=reps
// adds the GPU kernel breakdown).
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace ps {

inline bool traceOn() {
    static const bool on = std::getenv("PARTS_SIM_TRACE") != nullptr;
    return on;
}

/** Prints the time from construction (or the last lap) to destruction under a stage name. */
class TraceTimer {
public:
    explicit TraceTimer(const char* stage) : stage_(stage), t0_(std::chrono::steady_clock::now()) {}
    ~TraceTimer() { print(); }
    TraceTimer(const TraceTimer&) = delete;
    TraceTimer& operator=(const TraceTimer&) = delete;
    /** Ends the current stage and starts the next. */
    void lap(const char* next) {
        print();
        stage_ = next;
        t0_ = std::chrono::steady_clock::now();
    }

private:
    void print() const {
        if (traceOn())
            std::fprintf(stderr, "[trace] %-26s %9.1f ms\n", stage_, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count());
    }
    const char* stage_;
    std::chrono::steady_clock::time_point t0_;
};

}  // namespace ps
