#include "parallel.hpp"

#include <cstdlib>

namespace ps {

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool([] {
        if (const char* env = std::getenv("PARTS_SIM_THREADS")) {
            const int n = std::atoi(env);
            if (n > 0) return static_cast<unsigned>(n);
        }
        return std::max(1u, std::thread::hardware_concurrency());
    }());
    return pool;
}

ThreadPool::ThreadPool(unsigned threads) {
    for (unsigned i = 1; i < threads; i++) workers_.emplace_back([this] { loop(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    wake_.notify_all();
    for (auto& t : workers_) t.join();
}

namespace {
// set while a thread executes pool chunks: a parallel loop inside one runs inline
thread_local bool inPool = false;
}  // namespace

void ThreadPool::run(int chunks, const std::function<void(int)>& fn) {
    if (workers_.empty() || chunks <= 1 || inPool) {
        for (int c = 0; c < chunks; c++) fn(c);
        return;
    }
    // several solver threads (airflow, structural, thermal) share the pool: one job at a time
    std::lock_guard<std::mutex> owner(owner_);
    inPool = true;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        job_ = &fn;
        chunks_ = chunks;
        next_.store(0);
        active_ = static_cast<int>(workers_.size());
        generation_++;
    }
    wake_.notify_all();
    // the calling thread takes chunks too
    for (int c; (c = next_.fetch_add(1)) < chunks;) fn(c);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return active_ == 0; });
    job_ = nullptr;
    inPool = false;
}

void ThreadPool::loop() {
    uint64_t seen = 0;
    for (;;) {
        const std::function<void(int)>* job;
        int chunks;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            wake_.wait(lock, [&] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            job = job_;
            chunks = chunks_;
        }
        inPool = true;
        for (int c; (c = next_.fetch_add(1)) < chunks;) (*job)(c);
        inPool = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--active_ == 0) done_.notify_all();
        }
    }
}

}  // namespace ps
