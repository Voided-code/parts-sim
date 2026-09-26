// Parts Sim desktop app entry point.
#include <QApplication>
#include <QFileOpenEvent>
#include <QIcon>
#include <QPalette>
#include <QStyleFactory>
#include <QStyleHints>
#include <QTimer>
#include <cstdio>

#include "mainwindow.hpp"
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

// Test automation: PARTS_SIM_SCRIPT="sample:beam;run;idle;shot:/tmp/a.png;quit" runs the steps in order.
void runScript(ps::MainWindow* w, QStringList steps) {
    if (steps.isEmpty()) return;
    const QString step = steps.takeFirst().trimmed();
    const QString cmd = step.section(':', 0, 0), arg = step.section(':', 1);
    auto next = [w, steps](int delay) { QTimer::singleShot(delay, w, [w, steps] { runScript(w, steps); }); };
    if (cmd == "sample") w->loadSample(arg);
    else if (cmd == "open") w->openFiles({arg});
    else if (cmd == "tab") w->setTab(arg);
    else if (cmd == "study") w->structural->selectStudy(arg);
    else if (cmd == "run") {
        if (w->tab() == "thermal") w->thermal->run();
        else if (w->tab() == "airflow") w->airflow->run();
        else w->structural->runStudy();
    } else if (cmd == "windload") w->airflow->useAsLoad();
    else if (cmd == "cells") w->airflow->setCells(arg.toDouble());
    else if (cmd == "voxels") w->structural->setVoxelPreview(arg == "on");
    else if (cmd == "thtemp" || cmd == "thheat" || cmd == "thconv") {
        // thheat:<face>:<value> - a heat input on one CAD face
        const auto type = cmd == "thtemp" ? ps::ThermalPanel::Item::Temp : cmd == "thheat" ? ps::ThermalPanel::Item::Heat : ps::ThermalPanel::Item::Conv;
        w->thermal->addOnFace(type, arg.section(':', 0, 0).toInt(), arg.section(':', 1, 1).toDouble());
    } else if (cmd == "thmode") w->thermal->setTransient(arg == "transient");
    else if (cmd == "break") w->structural->runBreak();
    else if (cmd == "view") w->viewer->setView(arg);
    else if (cmd == "level") w->structural->setLevelPublic(arg);
    else if (cmd == "wait") return next(arg.toInt());
    else if (cmd == "idle") {
        // wait until no solver job is running
        if (w->structural->job || w->thermal->job || w->airflow->busy() || w->busy->busy()) {
            steps.prepend(step);
            QTimer::singleShot(100, w, [w, steps] { runScript(w, steps); });
            return;
        }
    } else if (cmd == "shot") {
        w->grab().save(arg);
    } else if (cmd == "status") {
        std::printf("status: %s\n", qPrintable(w->statusText()));
        std::fflush(stdout);
    } else if (cmd == "quit") {
        QApplication::quit();
        return;
    }
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
    return app.exec();
}
