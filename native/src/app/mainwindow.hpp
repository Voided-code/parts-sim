// Main window: toolbar, menus, sidebar tabs, 3D viewport, status bar and the shared app state
// (current part, units, material) the panels read.
#pragma once

#include <QLabel>
#include <QMainWindow>
#include <QTabWidget>
#include <memory>

#include "core/importers.hpp"
#include "core/materials.hpp"
#include "core/mesh.hpp"

namespace ps {

class Viewport;
class Legend;
class BusyOverlay;
class Picker;
class PartPanel;
class StructuralPanel;
class ThermalPanel;
class AirflowPanel;
struct SampleSetup;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    // ---- shared state ----
    std::shared_ptr<Part> part;
    QString partInfo;
    QString units = "mm";
    Material material;
    double toMeters() const;

    Viewport* viewer = nullptr;
    Legend* legend = nullptr;
    BusyOverlay* busy = nullptr;
    Picker* picker = nullptr;
    QLabel* probe = nullptr;
    QLabel* hintLabel = nullptr;
    PartPanel* partPanel = nullptr;
    StructuralPanel* structural = nullptr;
    ThermalPanel* thermal = nullptr;
    AirflowPanel* airflow = nullptr;

    QString tab() const;
    QString statusText() const;
    void setTab(const QString& name);
    void status(const QString& msg, const QString& kind = {});
    void hint(const QString& text);
    void setXRay(bool on);
    void setMaterial(const Material& m);
    void setUnits(const QString& u, bool notify = true);
    void applyScale(double s);
    void rotatePart(int axis);
    void loadGeneratedPart(MeshSource src, const QString& info);

    void openFiles(const QStringList& paths);
    void loadSample(const QString& id);

signals:
    void materialChanged();
    void unitsChanged();
    void partLoaded();

protected:
    void dragEnterEvent(QDragEnterEvent* e) override;
    void dropEvent(QDropEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;
    void closeEvent(QCloseEvent* e) override;

private:
    void buildUi();
    void buildMenus();
    bool loadPart(const MeshSource& src, const QString& units, const QString& info, const SampleSetup* setup, bool keepTab);
    void tabChanged(int index);
    void layoutOverlays();
    void saveScreenshot();
    void showHelp();

    QTabWidget* tabs_ = nullptr;
    QWidget* viewportHost_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QWidget* emptyState_ = nullptr;
    QString currentTab_ = "part";
    int loadRequest_ = 0;
};

}  // namespace ps
