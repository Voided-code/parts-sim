// Thermal tab: fixed temperatures, heat inputs and convection picked on faces, steady-state or
// transient heat conduction on the voxel model, and temperature / heat-flux results.
#pragma once

#include <QWidget>
#include <memory>
#include <optional>
#include <vector>

#include "core/materials.hpp"
#include "fea/structural.hpp"
#include "jobs.hpp"

class QCheckBox;
class QComboBox;
class QDoubleSpinBox;
class QFrame;
class QLabel;
class QListWidget;
class QPushButton;
class QSlider;
class QVBoxLayout;

namespace ps {

class MainWindow;
class KpiGrid;
class LineChart;

class ThermalPanel : public QWidget {
    Q_OBJECT
public:
    explicit ThermalPanel(MainWindow* app);
    ~ThermalPanel() override;

    void reset();
    void activate();
    void deactivate();
    void tick(double dt);
    QString probeText(const QPointF& pos);
    void onPartScaled(double s, const Vec3& pivot);
    void renderMaterial();
    void markStale();
    void cancel();
    void run();

    struct Item {
        enum Type { Temp, Heat, Conv } type;
        QString name;
        double value = 0;     // °C, W or W/m²K
        double ambient = 20;  // °C (convection)
        std::vector<Patch> patches;
    };
    std::vector<Item> items;
    /** Automation: add a heat input on one CAD face; switch steady / transient. */
    void addOnFace(Item::Type type, int face, double value);
    void setTransient(bool on);
    std::shared_ptr<JobControl> job;

private:
    void buildUi();
    void addItem(Item::Type type);
    void pick(int index, bool isNew);
    void renderList();
    void renderEditor();
    void drawOverlays();
    void updateMeshInfo();
    void showSetup();
    void show();
    void setFrame(int i);

    MainWindow* app_;
    int selected_ = -1;
    bool transient_ = false;
    QString display_ = "setup";

    struct Frame { double t; std::vector<float> T; double min, max; };
    struct Result {
        std::vector<Frame> frames;  // transient: every step; steady: one
        std::vector<float> flux;
        Material material;
        bool transient = false, converged = true, stale = false;
        double heatIn = 0;
        int voxels = 0;
    };
    std::optional<Result> result_;
    struct View { QString plot = "T"; int frame = 0; bool playing = false, bands = false, bcs = false; double t = 0; } view_;
    std::vector<float> shown_;
    std::function<QString(double)> shownFmt_;

    QLabel* matName_;
    QLabel* matProps_;
    QListWidget* list_;
    QWidget* editor_;
    QCheckBox* ambient_;
    QWidget* ambientProps_;
    QDoubleSpinBox* ambientH_;
    QDoubleSpinBox* ambientT_;
    QWidget* modeSeg_;
    QWidget* transientBox_;
    QDoubleSpinBox* duration_;
    QDoubleSpinBox* steps_;
    QDoubleSpinBox* initial_;
    QSlider* res_;
    QLabel* resOut_;
    QLabel* meshInfo_;
    QFrame* resultsCard_;
    QLabel* alert_;
    KpiGrid* kpis_;
    QComboBox* plot_;
    QWidget* timeBox_;
    LineChart* chart_;
    QSlider* frame_;
    QLabel* frameOut_;
    QPushButton* play_;
};

}  // namespace ps
