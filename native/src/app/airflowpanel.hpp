// Airflow tab: wind setup, flow run control, aerodynamic results and flow visualisation
// (particles, streamlines, slice plane, surface pressure), and the wind load for the bend test.
#pragma once

#include <QWidget>
#include <atomic>
#include <memory>
#include <optional>
#include <vector>

#include "cfd/airflow.hpp"
#include "jobs.hpp"
#include "samples.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
class QLabel;
class QPushButton;
class QSlider;
class QToolButton;

namespace ps {

class MainWindow;
class KpiGrid;

class AirflowPanel : public QWidget {
    Q_OBJECT
public:
    explicit AirflowPanel(MainWindow* app);
    ~AirflowPanel() override;

    void reset(const std::optional<SampleSetup::Air>& preset = std::nullopt);
    void activate();
    void deactivate();
    void tick(double dt);
    QString probeText(const QPointF& pos);
    void onPartScaled() { reset(); }
    void run();
    void stop();
    bool busy() const { return bool(job_); }
    bool hasResults() const;
    void useAsLoad();
    /** Automation: set the grid size (cells). */
    void setCells(double n);
    /** Script diagnostics: step count and force coefficients of the latest snapshot. */
    QString aeroSummary() const;

private:
    void buildUi();
    void syncWind();
    void markDirty();
    void updateButtons();
    void update();
    void applyColoring(bool force = false);
    void renderLegends();
    void drawWindArrow();
    void clearVisuals();
    void initParticles();
    void spawn(int i);
    void stepParticles(double dt);
    void buildStreamlines();
    void buildSlice();
    void buildDomain();
    AirflowOptions settings() const;
    bool gpuEngine() const;
    AirflowCapacity capacity() const;
    void syncCells();
    void updateCellsInfo();

    MainWindow* app_;
    std::shared_ptr<AirflowSim> sim_;
    std::shared_ptr<JobControl> job_;
    std::shared_ptr<std::atomic<bool>> pending_ = std::make_shared<std::atomic<bool>>(false);
    int yaw_ = 90, pitch_ = 0;
    double cells_ = 1e6;   // grid size chosen on the slider
    double lastMlups_ = 0;  // measured speed of the last run on this engine
    QString lastEngine_;
    bool dirty_ = true;
    std::vector<float> cp_;
    double cpMin_ = 0, cpMax_ = 0;
    double lastCp_ = -1e9, lastStream_ = -1e9, lastInfo_ = -1e9, clock_ = 0;
    int64_t shownSteps_ = -1;

    // particles in lattice coordinates, with a short trail each
    static constexpr int TRAIL = 12;
    int particles_ = 3000;
    std::vector<float> pos_, hist_, age_;
    int head_ = 0;
    uint32_t seed_ = 12345;
    double rnd();

    QToolButton* presets_[5];
    QSlider* yaw_s_;
    QSlider* pitch_s_;
    QLabel* yawOut_;
    QLabel* pitchOut_;
    QDoubleSpinBox* speed_;
    QDoubleSpinBox* density_;
    QSlider* cellsSlider_ = nullptr;
    QLabel* cellsOut_ = nullptr;
    QLabel* cellsInfo_ = nullptr;
    QComboBox* engine_;
    QLabel* gpuInfo_;
    QPushButton* runBtn_;
    QPushButton* resetBtn_;
    QFrame* flowCard_;
    KpiGrid* kpis_;
    QLabel* state_;
    QFrame* displayCard_;
    QCheckBox* cpChk_;
    QCheckBox* particlesChk_;
    QCheckBox* streamChk_;
    QCheckBox* domainChk_;
    QCheckBox* sliceChk_;
    QWidget* sliceControls_;
    QComboBox* sliceAxis_;
    QComboBox* sliceQty_;
    QSlider* slicePos_;
    QLabel* slicePosOut_;
    QPushButton* windLoad_;
};

}  // namespace ps
