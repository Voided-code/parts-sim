// Shared worker pool and parallel loops.
//
// The solvers call parallel_for thousands of times per second (every matrix product, smoothing
// sweep and vector update of a conjugate-gradient run), so the threads are created once and
// wait on a condition variable between jobs. Work is split into contiguous chunks; reductions
// sum per-chunk partials in chunk order so results do not depend on thread timing.
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace ps {

class ThreadPool {
public:
    static ThreadPool& instance();

    explicit ThreadPool(unsigned threads);
    ~ThreadPool();
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    unsigned size() const { return static_cast<unsigned>(workers_.size()) + 1; }

    /** Run fn(chunk) for chunk in [0, chunks) on all threads (the caller helps); returns when done. */
    void run(int chunks, const std::function<void(int)>& fn);

private:
    void loop();

    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::mutex owner_;  // held by the thread whose job the pool is running
    std::condition_variable wake_, done_;
    const std::function<void(int)>* job_ = nullptr;
    int chunks_ = 0;
    std::atomic<int> next_{0};
    int active_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

/** Number of chunks worth splitting n items into (1 below `grain` items per chunk). */
inline int chunkCount(int64_t n, int64_t grain = 4096) {
    const int64_t threads = ThreadPool::instance().size();
    const int64_t byGrain = std::max<int64_t>(1, n / std::max<int64_t>(1, grain));
    return static_cast<int>(std::min<int64_t>(byGrain, threads * 4));
}

/** fn(begin, end) over [0, n) split into contiguous ranges. */
template <class F>
void parallelFor(int64_t n, F&& fn, int64_t grain = 4096) {
    if (n <= 0) return;
    const int chunks = chunkCount(n, grain);
    if (chunks <= 1) {
        fn(int64_t{0}, n);
        return;
    }
    std::function<void(int)> job = [&](int c) {
        const int64_t b = n * c / chunks, e = n * (c + 1) / chunks;
        if (b < e) fn(b, e);
    };
    ThreadPool::instance().run(chunks, job);
}

/** Deterministic parallel sum of fn(begin, end) partials. */
template <class F>
double parallelSum(int64_t n, F&& fn, int64_t grain = 8192) {
    if (n <= 0) return 0.0;
    const int chunks = chunkCount(n, grain);
    if (chunks <= 1) return fn(int64_t{0}, n);
    std::vector<double> part(chunks, 0.0);
    std::function<void(int)> job = [&](int c) {
        const int64_t b = n * c / chunks, e = n * (c + 1) / chunks;
        part[c] = b < e ? fn(b, e) : 0.0;
    };
    ThreadPool::instance().run(chunks, job);
    double s = 0;
    for (double v : part) s += v;
    return s;
}

/** Parallel maximum of fn(begin, end) partials (0 when empty). */
template <class F>
double parallelMax(int64_t n, F&& fn, int64_t grain = 8192) {
    if (n <= 0) return 0.0;
    const int chunks = chunkCount(n, grain);
    if (chunks <= 1) return fn(int64_t{0}, n);
    std::vector<double> part(chunks, 0.0);
    std::function<void(int)> job = [&](int c) {
        const int64_t b = n * c / chunks, e = n * (c + 1) / chunks;
        part[c] = b < e ? fn(b, e) : 0.0;
    };
    ThreadPool::instance().run(chunks, job);
    double m = 0;
    for (double v : part) m = v > m ? v : m;
    return m;
}

inline double dot(const double* a, const double* b, int64_t n) {
    return parallelSum(n, [&](int64_t lo, int64_t hi) {
        double s = 0;
        for (int64_t i = lo; i < hi; i++) s += a[i] * b[i];
        return s;
    });
}

}  // namespace ps
