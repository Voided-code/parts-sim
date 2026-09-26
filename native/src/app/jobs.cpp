#include "jobs.hpp"

#include <chrono>
#include <cstdint>
#include <string>
#include <utility>

namespace ps {

void JobControl::progress(double frac, const std::string& text) {
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    int64_t last = lastProgress_.load();
    if (now - last < 33 && frac < 1) return;
    lastProgress_ = now;
    if (!context_ || !onProgress_) return;
    auto fn = onProgress_;
    std::weak_ptr<JobControl> self = weak_from_this();
    QPointer<QObject> ctx = context_;
    QMetaObject::invokeMethod(context_, [fn, frac, text, ctx, self]() {
        auto s = self.lock();
        if (ctx && s && !s->cancelled()) fn(frac, text);
    }, Qt::QueuedConnection);
}

void JobControl::post(std::function<void()> fn) {
    if (!context_) return;
    QPointer<QObject> ctx = context_;
    QMetaObject::invokeMethod(context_, [fn = std::move(fn), ctx]() {
        if (ctx) fn();
    }, Qt::QueuedConnection);
}

}  // namespace ps
