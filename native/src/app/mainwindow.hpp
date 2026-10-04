// Main window: toolbar, menus, sidebar tabs, 3D viewport, status bar and the shared app state
// (current part, units, material) the panels read.
#pragma once

#include <QFrame>
#include <QLabel>
#include <QMainWindow>
#include <QTabWidget>
#include <map>
#include <memory>
#include <optional>

#include "core/psim.hpp"

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

/** What goes into a .psim file (the Save dialog's choices). */
struct PsimSaveOptions {
    QString name, notes;
    bool exact = false;          // geometry: exact float32 instead of 16-bit compact
    bool setup = true;
    bool allResults = true;      // every result present, unless `results` is non-empty
    QStringList results;         // ids: static, thermal, airflow...
    bool cad = false, thumb = true;
};

/** A file that is open: its INFO, sections and the numbers it showed, for the banner and Re-run. */
struct LoadedFile {
    QString name;
    json::Value info;
    std::vector<psim::TableEntry> table;
    size_t size = 0;
    int version = 0;
    bool hasResults = false, hasSetup = false, edited = false;
    std::map<QString, std::map<QString, double>> stored;  // key numbers of the stored results
    std::vector<std::string> skipped;
    bool rerun = false;
};

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

    // ---- .psim files (mainwindow_psim.cpp) ----
    std::optional<LoadedFile> loadedFile;
    QByteArray sourceBytes;  // the imported CAD file, for "Original CAD file" in the Save dialog
    QString sourceName;
    double psimOpenMs = 0, psimSaveMs = 0;
    bool restoring = false;  // true while a file is being put into the panels
    bool openPsim(const QString& path);
    bool savePsim(const QString& path, const PsimSaveOptions& o);
    void showSaveDialog();
    void showFileInfo();
    void rerunLoaded();
    /** Called by the panels: the setup changed / a run finished (with its key numbers). */
    void psimEdited();
    void psimRunDone(const QString& kind, const std::map<QString, double>& now);
    bool psimShowingResults() const;

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
    bool installPart(std::shared_ptr<Part> p, const QString& units, const QString& info, const SampleSetup* setup, bool keepTab);
    void showBanner();
    psim::File gatherPsim(const PsimSaveOptions& o, QStringList* resultIds = nullptr);
    bool loadPart(const MeshSource& src, const QString& units, const QString& info, const SampleSetup* setup, bool keepTab);
    void tabChanged(int index);
    void layoutOverlays();
    void saveScreenshot();
    void showHelp();

    QTabWidget* tabs_ = nullptr;
    QWidget* viewportHost_ = nullptr;
    QLabel* statusLabel_ = nullptr;
    QFrame* banner_ = nullptr;
    QWidget* emptyState_ = nullptr;
    QString currentTab_ = "part";
    int loadRequest_ = 0;
};

}  // namespace ps
