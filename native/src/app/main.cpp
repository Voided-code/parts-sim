// Parts Sim desktop app entry point.
#include <QApplication>
#include <QFile>
#include <QElapsedTimer>
#include <QFileOpenEvent>
#include <QIcon>
#include <QPalette>
#include <QStyleFactory>
#include <QStyleHints>
#include <QTimer>
#include <cstdio>
#include <cstdlib>

#include "jobs.hpp"
#include "mainwindow.hpp"
#include "studies.hpp"
#include "structuralpanel.hpp"
#include "thermalpanel.hpp"
#include "airflowpanel.hpp"
#include "viewport.hpp"
#include "widgets.hpp"

namespace {

// Light / dark style sheets on top of the platform style; colours match the web version.
QString styleSheet(bool dark) {
    const QString panel = dark ? "#1a1d22" : "#ffffff";
    const QString panel2 = dark ? "#21252b" : "#f5f7fa";
    const QString bg = dark ? "#111317" : "#e9edf2";
    const QString border = dark ? "#2c3139" : "#dde3ea";
    const QString text = dark ? "#eef1f5" : "#121820";
    const QString muted = dark ? "#8b939e" : "#6f7986";
    const QString accent = dark ? "#3987e5" : "#2a78d6";
    return QString(R"(
QTabWidget::pane { border: none; }
QScrollArea, QScrollArea > QWidget > QWidget { background: %1; }
QFrame#card { background: %2; border: 1px solid %3; border-radius: 10px; }
QLabel#cardTitle { color: %4; font-weight: 600; font-size: 11px; letter-spacing: 0.5px; }
QLabel#note, QLabel#meta { color: %4; font-size: 11px; }
QLabel#fieldLabel { color: %4; font-size: 11px; }
QFrame#kpi { background: %5; border: 1px solid %3; border-radius: 8px; }
QLabel#kpiLabel { color: %4; font-size: 11px; }
QLabel#kpiValue { font-family: Menlo, Consolas, monospace; font-size: 17px; font-weight: 700; color: %6; }
QLabel#kpiSub { color: %4; font-size: 11px; }
QLabel#alert[kind="error"] { background: rgba(207,52,52,0.10); border: 1px solid rgba(207,52,52,0.45); border-radius: 8px; padding: 8px; }
QLabel#alert[kind="info"] { background: %5; border: 1px solid %3; border-radius: 8px; padding: 8px; }
QLabel#status[kind="error"] { color: #cf3434; }
QLabel#status[kind="warn"] { color: #b7791f; }
QPushButton#primary { background: %7; color: white; border: 1px solid %7; border-radius: 8px; font-weight: 600; padding: 6px 12px; }
QPushButton#primary:hover { background: %8; }
QFrame#busy, QFrame#picker { background: %2; border: 1px solid %3; border-radius: 10px; }
QFrame#editor { background: %5; border: 1px solid %3; border-radius: 8px; }
QLabel#probe { background: rgba(18,24,32,0.88); color: white; border-radius: 6px; padding: 4px 8px; font-family: Menlo, monospace; font-size: 11px; }
QLabel#hint { background: rgba(18,24,32,0.82); color: white; border-radius: 8px; padding: 6px 12px; }
QLabel#emptyTitle { font-size: 18px; font-weight: 600; color: %6; }
QListWidget#items { background: transparent; border: none; }
QListWidget#items::item { border: 1px solid %3; border-radius: 7px; margin: 2px 0; }
QListWidget#items::item:selected { border: 1px solid %7; background: transparent; }
QToolButton#segFirst, QToolButton#segMid, QToolButton#segLast { border: 1px solid %3; padding: 4px 8px; background: %2; }
QToolButton#segFirst { border-top-left-radius: 7px; border-bottom-left-radius: 7px; }
QToolButton#segLast { border-top-right-radius: 7px; border-bottom-right-radius: 7px; }
QToolButton#segFirst:checked, QToolButton#segMid:checked, QToolButton#segLast:checked { background: %9; color: %7; font-weight: 600; }
)")
        .arg(bg, panel, border, muted, panel2, text, accent, dark ? "#4a95ee" : "#1f68c0", dark ? "#1d2a3b" : "#e3eefb");
}

class App : public QApplication {
public:
    using QApplication::QApplication;
    ps::MainWindow* window = nullptr;
    QStringList pending;

protected:
    bool event(QEvent* e) override {
        // Finder "Open With" and double-clicked documents arrive as file-open events
        if (e->type() == QEvent::FileOpen) {
            const QString f = static_cast<QFileOpenEvent*>(e)->file();
            if (window) window->openFiles({f});
            else pending << f;
            return true;
        }
        return QApplication::event(e);
    }
};

void scriptFailure(const QString& message) {
    std::fprintf(stderr, "PARTS_SIM_SCRIPT_FAILED: %s\n", qPrintable(message));
    std::fflush(stderr);
    QApplication::exit(1);
}

// Test automation: PARTS_SIM_SCRIPT="sample:beam;idle;assert:part;run;idle;assert:static;shot:/tmp/a.png;quit".
// Assertions exit with failure. A static pass also writes PARTS_SIM_SCRIPT_RESULT when set, so
// packaged GUI executables can report success even on platforms without a console.
void runScript(ps::MainWindow* w, QStringList steps) {
    if (steps.isEmpty()) return;
    const QString step = steps.takeFirst().trimmed();
    const QString cmd = step.section(':', 0, 0), arg = step.section(':', 1);
    auto next = [w, steps](int delay) { QTimer::singleShot(delay, w, [w, steps] { runScript(w, steps); }); };
    if (cmd == "sample") w->loadSample(arg);
    else if (cmd == "open") w->openFiles({arg});
    else if (cmd == "savepsim") {
        // savepsim:<path>[:geometry=exact][:noresults][:nosetup] - write the current state as a .psim file
        QStringList parts = arg.split(':');
        ps::PsimSaveOptions o;
        while (parts.size() > 1 && (parts.last().startsWith("geometry=") || parts.last() == "noresults" || parts.last() == "nosetup")) {
            const QString opt = parts.takeLast();
            if (opt == "geometry=exact") o.exact = true;
            else if (opt == "noresults") { o.allResults = false; o.results.clear(); }
            else if (opt == "nosetup") o.setup = false;
        }
        if (!w->savePsim(parts.join(':'), o)) return scriptFailure("Could not save the .psim file. " + w->statusText());
        std::printf("psim save: %.0f ms\n", w->psimSaveMs);
        std::fflush(stdout);
    } else if (cmd == "psimtime") {
        std::printf("psim open: %.0f ms, save: %.0f ms\n", w->psimOpenMs, w->psimSaveMs);
        std::fflush(stdout);
    } else if (cmd == "fileinfo") {
        if (!w->loadedFile) return scriptFailure("No file is open.");
        std::printf("psim file: %s, %zu bytes, format %d\n", qPrintable(w->loadedFile->name), w->loadedFile->size, w->loadedFile->version);
        std::fflush(stdout);
    } else if (cmd == "rerun") w->rerunLoaded();
    else if (cmd == "tab") w->setTab(arg);
    else if (cmd == "study") w->structural->selectStudy(arg);
    else if (cmd == "setopt") {
        // setopt:<study>:<json> - merge { opts: {...}, view: {...} } into a study's options as a loaded file would
        const QString id = arg.section(':', 0, 0);
        auto* s = w->structural->studyById(id);
        if (!s) return scriptFailure("setopt: unknown study " + id);
        try { s->importOptions(ps::psim::parseJson(arg.section(':', 1).toStdString())); } catch (const std::exception& e) { return scriptFailure(QString("setopt: ") + e.what()); }
        w->structural->renderStudyOptions();
    } else if (cmd == "getopt") {
        auto* s = w->structural->studyById(arg);
        if (!s) return scriptFailure("getopt: unknown study " + arg);
        std::printf("options %s: %s\n", qPrintable(arg), ps::psim::stringifyJson(s->exportOptions()).c_str());
        std::fflush(stdout);
    }
    else if (cmd == "studyopt") { if (auto* s = w->structural->currentStudy()) s->scriptOption(arg); }
    else if (cmd == "run") {
        if (w->tab() == "thermal") w->thermal->run();
        else if (w->tab() == "airflow") w->airflow->run();
        else w->structural->runStudy();
    } else if (cmd == "windload") w->airflow->useAsLoad();
    else if (cmd == "cells") w->airflow->setCells(arg.toDouble());
    else if (cmd == "airengine") w->airflow->setEngine(arg == "gpu" ? 1 : arg == "cpu" ? 2 : 0);
    else if (cmd == "aero") {
        std::printf("aero: %s\n", qPrintable(w->airflow->aeroSummary()));
        std::fflush(stdout);
    } else if (cmd == "result") {
        std::printf("result: %s\n", qPrintable(w->structural->resultSummary()));
        std::fflush(stdout);
    }
    else if (cmd == "voxels") w->structural->setVoxelPreview(arg == "on");
    else if (cmd == "mesh") w->structural->setTargetVoxels(arg.toDouble());  // mesh:<total voxels>
    else if (cmd == "engine") w->structural->setEngine(arg == "gpu" ? 1 : arg == "cpu" ? 2 : 0);
    else if (cmd == "meshinfo") {
        std::printf("mesh: %s\n", qPrintable(w->structural->meshSummary()));
        std::fflush(stdout);
    }
    else if (cmd == "thtemp" || cmd == "thheat" || cmd == "thconv") {
        // thheat:<face>:<value> - a heat input on one CAD face
        const auto type = cmd == "thtemp" ? ps::ThermalPanel::Item::Temp : cmd == "thheat" ? ps::ThermalPanel::Item::Heat : ps::ThermalPanel::Item::Conv;
        w->thermal->addOnFace(type, arg.section(':', 0, 0).toInt(), arg.section(':', 1, 1).toDouble());
    } else if (cmd == "thmode") w->thermal->setTransient(arg == "transient");
    else if (cmd == "break") w->structural->runBreak();
    else if (cmd == "view") w->viewer->setView(arg);
    else if (cmd == "level") w->structural->setLevelPublic(arg);
    else if (cmd == "wait") return next(arg.toInt());
    else if (cmd == "airwait") {
        // airwait:<seconds> - wait until the airflow stops by itself (converged) or the time is up
        const double limit = arg.toDouble();
        auto deadline = std::make_shared<QElapsedTimer>();
        deadline->start();
        auto poll = std::make_shared<std::function<void()>>();
        *poll = [w, steps, limit, deadline, poll] {
            if (w->airflow->flowRunning() && deadline->elapsed() < limit * 1000) return void(QTimer::singleShot(250, w, *poll));
            runScript(w, steps);
            *poll = nullptr;
        };
        return void(QTimer::singleShot(250, w, *poll));
    }
    else if (cmd == "idle") {
        // wait until no solver job is running
        if (w->structural->job || w->thermal->job || w->airflow->busy() || w->busy->busy()) {
            steps.prepend(step);
            QTimer::singleShot(100, w, [w, steps] { runScript(w, steps); });
            return;
        }
    } else if (cmd == "assert") {
        if (arg == "part") {
            if (!w->part || w->part->nVert <= 0 || w->part->nTri <= 0)
                return scriptFailure("No part was loaded. " + w->statusText());
        } else if (arg == "psim") {
            if (!w->psimShowingResults()) return scriptFailure("No .psim file with results is loaded. " + w->statusText());
            std::printf("PARTS_SIM_PSIM_OK %s\n", qPrintable(w->structural->resultSummary()));
            std::fflush(stdout);
        } else if (arg == "static") {
            if (!w->part || !w->structural->hasSuccessfulStaticResult())
                return scriptFailure("No successful current static result. " + w->statusText());
            const QByteArray marker("PARTS_SIM_SMOKE_OK\n");
            if (qEnvironmentVariableIsSet("PARTS_SIM_SCRIPT_RESULT")) {
                QFile file(qEnvironmentVariable("PARTS_SIM_SCRIPT_RESULT"));
                if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate) || file.write(marker) != marker.size() || !file.flush())
                    return scriptFailure("Could not write the script result marker: " + file.errorString());
            }
            std::printf("%s", marker.constData());
            std::fflush(stdout);
        } else return scriptFailure("Unknown assertion: " + arg);
    } else if (cmd == "shot") {
        if (!w->grab().save(arg)) return scriptFailure("Could not save screenshot: " + arg);
    } else if (cmd == "loaddir") {
        // loaddir:x,y,z - point the selected load (or the first) along a direction
        const QStringList v = arg.split(',');
        auto& loads = w->structural->loads;
        if (v.size() != 3 || loads.empty()) return scriptFailure("loaddir needs x,y,z and a load");
        auto& l = loads[std::max(0, w->structural->selectedLoad)];
        l.dir = {v[0].toDouble(), v[1].toDouble(), v[2].toDouble()};
        w->structural->markStale();
    } else if (cmd == "clock") {
        // elapsed time since start-up, to time runs: clock;run;idle;clock
        static QElapsedTimer timer;
        if (!timer.isValid()) timer.start();
        std::printf("clock: %lld ms\n", static_cast<long long>(timer.elapsed()));
        std::fflush(stdout);
    } else if (cmd == "status") {
        std::printf("status: %s\n", qPrintable(w->statusText()));
        std::fflush(stdout);
    } else if (cmd == "quit") {
        QApplication::quit();
        return;
    } else return scriptFailure("Unknown script command: " + cmd);
    next(cmd == "sample" || cmd == "open" ? 300 : 50);
}

}  // namespace

int main(int argc, char** argv) {
    QApplication::setApplicationName("Parts Sim");
    QApplication::setOrganizationName("Parts Sim");
    QApplication::setApplicationVersion(PARTS_SIM_VERSION);
    App app(argc, argv);
    app.setWindowIcon(QIcon(QStringLiteral(":/icon.png")));
    const bool dark = app.styleHints()->colorScheme() == Qt::ColorScheme::Dark;
    app.setStyleSheet(styleSheet(dark));
    int code = 0;
    {
        ps::MainWindow w;
        app.window = &w;
        w.show();
        QStringList files = app.pending;
        const QStringList args = app.arguments();
        for (int i = 1; i < args.size(); i++)
            if (!args[i].startsWith('-')) files << args[i];
        if (qEnvironmentVariableIsSet("PARTS_SIM_SCRIPT")) {
            const QStringList steps = qEnvironmentVariable("PARTS_SIM_SCRIPT").split(';', Qt::SkipEmptyParts);
            QTimer::singleShot(500, &w, [&w, steps] { runScript(&w, steps); });
        } else if (!files.isEmpty()) QTimer::singleShot(0, &w, [&w, files] { w.openFiles(files); });
        else if (qEnvironmentVariableIsSet("PARTS_SIM_SAMPLE")) {
            const QString id = qEnvironmentVariable("PARTS_SIM_SAMPLE");
            QTimer::singleShot(0, &w, [&w, id] { w.loadSample(id); });
        }
        code = app.exec();
    }
    app.window = nullptr;
    // A job still running on its detached thread (such as the start-up speed calibration, inside a
    // slow software GPU driver's shader compiler) must not run into the libraries' exit-time
    // teardown: leave without it.
    if (ps::runningJobs() > 0) {
        std::fflush(nullptr);
        std::_Exit(code);
    }
    return code;
}
