// Structural studies beyond linear static, shown in the Structural tab's study picker: nonlinear
// static, frequency, buckling, fatigue, drop test, linear dynamic and optimization. Each study
// renders its options, runs its job and draws its results; StructuralPanel owns the shared setup
// (material, fixtures, loads, mesh) and the linear static study.
#pragma once

#include <QString>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "widgets.hpp"

class QVBoxLayout;
class QWidget;

namespace ps {

class StructuralPanel;
class MainWindow;
struct Part;

struct StudyInfo {
    QString id, label, desc, button;
    bool fixtures, fixturesOptional, loads, loadsOptional;
    double budget;  // voxel budget for the suggested mesh resolution
};

const std::vector<StudyInfo>& studyInfos();

class Study {
public:
    explicit Study(StructuralPanel* panel) : panel_(panel) {}
    virtual ~Study() = default;
    virtual QString id() const = 0;
    /** Fill the options card; return false when the study has no options. */
    virtual bool options(QVBoxLayout*) { return false; }
    virtual void run() = 0;
    /** Draw the result (builds the result card's body when it belongs to another study or result). */
    void show();
    virtual void tick(double) {}
    virtual QString probe(int tri, const double bary[3]);
    virtual void clear() = 0;
    virtual bool hasResult() const = 0;
    virtual void leave() {}

protected:
    /** Build the result card body (widgets that stay while the result is shown). */
    virtual void build(QVBoxLayout* body) = 0;
    /** Update the card (KPIs, chart, labels) and draw the part. */
    virtual void refresh() = 0;
    /** Rebuild the body on the next show() (new result or a different layout). */
    void invalidate() { built_ = false; }
    bool visible() const;

    MainWindow* app() const;
    Part& part() const;
    void setTitle(const QString& title);
    void setKpis(const std::vector<KpiGrid::Kpi>& kpis);
    void alert(const QString& text, bool error = true);
    /** Colour the part with per-vertex values and draw the legend (bands / heat from the panel's view). */
    void paint(const std::vector<float>& values, LegendSpec spec, std::function<QString(double)> probeFmt = {});
    void marker(int vertex, const std::vector<float>* u, double scale, const QString& label);
    /** Contour bands / heat colours / show loads toggles. */
    QWidget* displayToggles();
    void status(const QString& msg, const QString& kind = {});
    QString engineNote(const std::string& engine, const std::string& gpuNote) const;

    StructuralPanel* panel_;
    std::vector<float> shown_;
    std::function<QString(double)> shownFmt_;
    int runGen_ = 0;  // bumped by every run; late messages from older runs are dropped

private:
    bool built_ = false;
};

/** The studies at indices 1.. of studyInfos() (index 0 is linear static, built into the panel). */
std::vector<std::unique_ptr<Study>> createStudies(StructuralPanel* panel);

}  // namespace ps
