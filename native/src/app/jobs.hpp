// Background solver jobs: the work runs on its own thread, reports progress and intermediate
// results to the UI thread, and can be cancelled (the work polls `cancelled()`).
#pragma once

#include <QObject>
#include <QPointer>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ps {

struct Cancelled : std::runtime_error {
    Cancelled() : std::runtime_error("Cancelled") {}
};

class JobControl : public std::enable_shared_from_this<JobControl> {
public:
    bool cancelled() const { return cancel_.load(); }
    void cancel() { cancel_ = true; }
    /** Throws Cancelled when the user cancelled. */
    void check() const {
        if (cancelled()) throw Cancelled();
    }
    /** Progress to show (thread-safe; throttled to ~30 updates a second). */
    void progress(double frac, const std::string& text);
    /** Run fn on the UI thread (intermediate results). */
    void post(std::function<void()> fn);

    std::function<void(double, std::string)> onProgress_;
    QPointer<QObject> context_;

private:
    std::atomic<bool> cancel_{false};
    std::atomic<int64_t> lastProgress_{0};
};

/** Jobs whose threads are still running (they are detached: nothing waits for them at exit). */
std::atomic<int>& runningJobs();

/**
 * Start work on a new thread. done(result) or fail(message, cancelled) run on the UI thread in
 * `context`'s thread, and are dropped if `context` was destroyed or the job was cancelled.
 */
template <class R>
std::shared_ptr<JobControl> runJob(QObject* context, std::function<R(JobControl&)> work, std::function<void(R)> done,
                                    std::function<void(QString, bool)> fail, std::function<void(double, QString)> progress);

}  // namespace ps

#include <QMetaObject>
#include <QString>

namespace ps {

template <class R>
std::shared_ptr<JobControl> runJob(QObject* context, std::function<R(JobControl&)> work, std::function<void(R)> done,
                                    std::function<void(QString, bool)> fail, std::function<void(double, QString)> progress) {
    auto ctl = std::make_shared<JobControl>();
    ctl->context_ = context;
    ctl->onProgress_ = [progress](double f, std::string t) {
        if (progress) progress(f, QString::fromStdString(t));
    };
    runningJobs()++;
    std::thread([ctl, context, work = std::move(work), done = std::move(done), fail = std::move(fail)]() mutable {
        struct Running { ~Running() { runningJobs()--; } } running;
        QPointer<QObject> ctx(context);
        try {
            auto result = std::make_shared<R>(work(*ctl));
            QMetaObject::invokeMethod(context, [ctl, ctx, result, done]() {
                if (!ctx || ctl->cancelled()) return;
                done(std::move(*result));
            }, Qt::QueuedConnection);
        } catch (const Cancelled&) {
            QMetaObject::invokeMethod(context, [ctl, ctx, fail]() {
                if (ctx && fail) fail(QStringLiteral("Cancelled"), true);
            }, Qt::QueuedConnection);
        } catch (const std::exception& e) {
            const QString msg = QString::fromUtf8(e.what());
            QMetaObject::invokeMethod(context, [ctl, ctx, fail, msg]() {
                if (ctx && !ctl->cancelled() && fail) fail(msg, false);
            }, Qt::QueuedConnection);
        }
    }).detach();
    return ctl;
}

}  // namespace ps
