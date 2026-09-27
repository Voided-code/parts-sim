// Structural tab: study picker, fixtures, loads, mesh, the linear static study (bend test,
// load levels, break test) and hosting of the other studies.
#pragma once

#include <QWidget>
#include <any>
#include <memory>
#include <functional>
#include <map>
#include <optional>
#include <set>

#include "fea/static_study.hpp"
#include "fea/structural.hpp"
#include "jobs.hpp"

class QCheckBox;
class QComboBox;
class QFrame;
class QLabel;
class QListWidget;
class QPushButton;
class QSlider;
class QSpinBox;
class QVBoxLayout;

namespace ps {

class MainWindow;
class KpiGrid;
class LineChart;
struct SampleSetup;
class Study;

class StructuralPanel : public QWidget {
    Q_OBJECT
public:
    explicit StructuralPanel(MainWindow* app);
    ~StructuralPanel() override;

    void reset(const SampleSetup* setup);
    void activate();
    void deactivate();
    void tick(double dt);
    QString probeText(const QPointF& pos);
    void markStale();
    void onMaterialChanged();
    void onPartScaled(double s, const Vec3& pivot);
    void updateMeshInfo();
    /** Mesh size as a total voxel count (the grid is sized to match it). */
    void setTargetVoxels(double n, bool markChanged = true);
    double maxVoxels() const;
    /** Automation: the mesh card's estimate and mesh lines; choose the engine (0 auto, 1 GPU, 2 CPU). */
    QString meshSummary() const;
    void setEngine(int index);
    /** Voxel count at which "Automatic" switches to the GPU. */
    static constexpr int AUTO_GPU_VOXELS = 40000;
    /** Show the voxel mesh instead of the part while "Preview voxel mesh" is on (setup view only). */
    void showVoxelPreview();
    void setVoxelPreview(bool on);
    bool hasSetup() const { return !fixtures.empty() || !loads.empty(); }

    void runStudy();
    void runBreak();
    void renderStudyOptions();
    void applyDisplay();
    void selectStudy(const QString& id);
    void setLevelPublic(const QString& level) { setLevel(level); }
    void cancelJob();
    /** Replace the airflow wind load (N per part triangle) and select it. */
    void addWindLoad(const std::vector<float>& forces, const Vec3& net);

    // ---- shared with the studies ----
    MainWindow* app() const { return app_; }
    std::vector<Fixture> fixtures;
    std::vector<Load> loads;
    int selectedLoad = -1;
    bool gravity() const;
    int resolution() const;
    std::shared_ptr<StructuralModel> modelFor(int res);
    std::shared_ptr<StructuralModel> model();
    struct Prepared {
        std::shared_ptr<StructuralModel> model;
        Assembly asm_;
        StructuralInput input;
        Material material;
        double toMeters;
        QString units;
    };
    bool checkSetup(bool requireFixtures = true, bool requireLoads = true);
    std::shared_ptr<JobControl> job;
    void drawOverlays();
    bool useGPU() const;

    struct View {
        QString plot = "vm";
        int scalePct = 1;
        QString level = "applied";
        bool animate = false, bands = false, heat = false, bcs = false, marker = true, breakStress = true;
    } view;

private:
    void buildUi();
    void syncVoxelControls();
    void learnVoxelFactor(const StructuralModel& m);
    void recordRunTime(int voxels, const std::string& engine, double seconds);
    void calibrate();
    QString runEstimate() const;
    void renderLists();
    void renderLoadEditor();
    void setStudy(int index);
    void pick(int fixtureIndex, int loadIndex, bool isNew);
    void addFixture();
    void addLoad(Load::Type type);
    void run();
    void showResults(bool full = true);
    void showBreakStep();
    void renderKpis();
    void renderBreakKpis();
    void renderAlert();
    void renderLevelNote();
    void setLevel(const QString& level, bool refresh = true);
    double loadFactor() const;
    double deformationScale(double autoScale) const;
    double currentScale() const;
    void placeMarker(double scale);
    void mapResult(const StaticResult& res, const Prepared& prep);
    Frame loadFrame(const Load& l) const;

    MainWindow* app_;
    std::map<int, std::shared_ptr<StructuralModel>> models_;
    std::set<int> isNewForce_;  // forces whose direction follows the picked face until the user sets one
    std::vector<float> shown_;  // values currently coloured (probe)
    std::function<QString(double)> shownFmt_;
    int study_ = 0;
    QString display_ = "setup";
    bool stale_ = false;
    double phase_ = 0;

    // result of the linear static study, mapped onto the part's vertices
    struct Result {
        std::vector<float> vm, p1, p3, u, dmag, fos;
        double maxVM = 0, maxDisp = 0, minP1 = 0, maxP1 = 0, minP3 = 0, maxP3 = 0;
        double minFos = 0, lamYield = 0, lamBreak = 0;
        int weakest = -1;
        double totalF = 0, autoScale = 1;
        Material material;
        QString units;
        int iterations = 0, voxels = 0;
        bool incomplete = false, unreliable = false, converged = true;
        double solvedShare = 1, lostLoad = 0;
        int thinVoxels = 0, resolution = 0;
        double voxelSize = 0, wallThickness = 0;
    };
    std::optional<Result> result_;
    struct Anim { double t = 0, dur = 3.5, lambda = 0, lastLegend = 0; };
    std::optional<Anim> anim_;

    // break test
    struct Break {
        std::vector<BreakStep> steps;
        struct Mapped { std::vector<float> vm, u; };
        std::vector<std::optional<Mapped>> mapped;
        double totalF = 0;
        Material material;
        std::shared_ptr<StructuralModel> model;
        double toMeters = 0.001;
        int current = 0;
        bool done = false;
        QString reason;
        double scale = 0;
    };
    std::optional<Break> brk_;
    bool playing_ = false;
    double playT_ = 0;

    // widgets
    QComboBox* studyBox_;
    QLabel* studyDesc_;
    QFrame* optionsCard_;
    QVBoxLayout* optionsLayout_;
    QFrame* fixturesCard_;
    QFrame* loadsCard_;
    QListWidget* fixtureList_;
    QListWidget* loadList_;
    QWidget* loadEditor_;
    QCheckBox* gravity_;
    // mesh size: the user sets a total voxel count; the grid is sized to match it
    QSlider* voxelsSlider_;
    QSpinBox* voxelsBox_;
    QLabel* etaLabel_;
    int resolution_ = 56;       // voxels on the longest side, derived from the target
    double targetVoxels_ = 5e4;
    double voxelFactor_ = 1;    // actual / estimated voxel count, learned from the meshes built for this part
    // seconds per voxel of one bend test (voxelize, build and solve) on each engine: from a start-up
    // calibration run, then from the real runs
    double secPerVoxel_[2] = {3e-6, 2.5e-6};  // CPU, GPU
    bool measured_[2] = {false, false};
    bool calibrated_ = false;
    std::shared_ptr<JobControl> calibration_;
    QCheckBox* voxelsChk_;
    std::shared_ptr<JobControl> previewJob_;
    QComboBox* engine_;
    QLabel* meshInfo_;
    QPushButton* runBtn_;
    QPushButton* breakBtn_;
    QFrame* resultsCard_;
    QLabel* alert_;
    KpiGrid* kpis_;
    QComboBox* plot_;
    QWidget* levelSeg_;
    QPushButton* bendBreak_;
    QLabel* levelNote_;
    QSlider* scale_;
    QLabel* scaleOut_;
    QCheckBox* animateChk_;
    QFrame* breakCard_;
    KpiGrid* breakKpis_;
    LineChart* breakChart_;
    QSlider* breakStep_;
    QLabel* breakStepOut_;
    QPushButton* breakPlay_;

public:
    // other studies render into this card
    QFrame* studyCard = nullptr;
    QLabel* studyTitle = nullptr;
    QLabel* studyAlert = nullptr;
    KpiGrid* studyKpis = nullptr;
    QVBoxLayout* studyBody = nullptr;
    QWidget* studyContent = nullptr;  // rebuilt by the study that owns it
    const void* studyContentOwner = nullptr;
    std::vector<std::unique_ptr<Study>> studies;
    int studyIndex() const { return study_; }
    Study* currentStudy() const { return study_ > 0 ? studies[study_ - 1].get() : nullptr; }
    bool isStale() const { return stale_; }
    /** True only while a completed, converged linear static result is current and displayed. */
    bool hasSuccessfulStaticResult() const {
        return study_ == 0 && display_ == "results" && !job && !stale_ && result_ &&
               result_->converged && !result_->unreliable && result_->voxels > 0;
    }
    /** Key numbers of the static result, for scripted comparisons (the "result" script command). */
    QString resultSummary() const {
        if (!result_) return "no result";
        return QString("voxels %1, %2 iterations, max von Mises %3, max displacement %4")
            .arg(result_->voxels).arg(result_->iterations).arg(result_->maxVM, 0, 'g', 12).arg(result_->maxDisp, 0, 'g', 12);
    }
    void clearStale() { stale_ = false; }
    void rememberModel(const std::shared_ptr<StructuralModel>& m);
    /** The voxel model at this resolution if it is already built for the current part. */
    std::shared_ptr<StructuralModel> cachedModel(int res) const;

    using StudyWork = std::function<std::any(Prepared&, JobControl&)>;
    using StudyDone = std::function<void(std::any&, Prepared&)>;
    /**
     * Check the setup, snapshot it and run `work` on a worker thread after voxelizing and assembling;
     * `done` runs on the UI thread. `failed(cancelled)` runs after the error is reported.
     * Returns false when the setup is incomplete (nothing started).
     */
    bool startStudy(const QString& busyText, bool requireFixtures, bool requireLoads, StudyWork work, StudyDone done,
                    std::function<void(bool)> failed = {});
    void setDisplay(const QString& d) { display_ = d; }
    QString display() const { return display_; }
};

}  // namespace ps
