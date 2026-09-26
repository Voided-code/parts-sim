#include "mainwindow.hpp"

#include <QAction>
#include <QApplication>
#include <QCloseEvent>
#include <QDesktopServices>
#include <QDialog>
#include <QDialogButtonBox>
#include <QDragEnterEvent>
#include <QFileDialog>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QLabel>
#include <QMenuBar>
#include <QMessageBox>
#include <QMimeData>
#include <QPainter>
#include <QStyle>
#include <QDir>
#include <QPushButton>
#include <QScrollArea>
#include <QShortcut>
#include <QSplitter>
#include <QStandardPaths>
#include <QStatusBar>
#include <QTextBrowser>
#include <QToolBar>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <fstream>

#include "core/solidworks.hpp"
#include "gpu/gpu.hpp"
#include "partpanel.hpp"
#include "picker.hpp"
#include "samples.hpp"
#include "structuralpanel.hpp"
#include "thermalpanel.hpp"
#include "airflowpanel.hpp"
#include "viewport.hpp"
#include "widgets.hpp"

namespace ps {

namespace {

double unitToMeters(const QString& u) {
    if (u == "cm") return 0.01;
    if (u == "m") return 1;
    if (u == "in") return 0.0254;
    return 0.001;
}

QString extensionsFilter() {
    QStringList pats;
    for (const auto& e : supportedExtensions()) pats << "*." + QString::fromStdString(e) << "*." + QString::fromStdString(e).toUpper();
    return QObject::tr("CAD and mesh files (%1);;STEP / IGES (*.step *.stp *.iges *.igs *.brep);;SolidWorks (*.sldprt *.sldasm);;Meshes (*.stl *.obj *.3mf *.ply *.glb)")
        .arg(pats.join(' '));
}

}  // namespace

MainWindow::MainWindow(QWidget* parent) : QMainWindow(parent) {
    material = materials()[0];
    setWindowTitle("Parts Sim");
    setAcceptDrops(true);
    resize(1440, 920);
    buildUi();
    buildMenus();
}

MainWindow::~MainWindow() = default;

double MainWindow::toMeters() const { return unitToMeters(units); }

void MainWindow::buildUi() {
    // ---- toolbar ----
    auto* tb = addToolBar(tr("Main"));
    tb->setMovable(false);
    tb->setObjectName("mainToolbar");
    tb->setToolButtonStyle(Qt::ToolButtonTextOnly);
    auto* open = tb->addAction(tr("Import part…"), this, [this] {
        const auto files = QFileDialog::getOpenFileNames(this, tr("Import part"), QString(), extensionsFilter());
        if (!files.isEmpty()) openFiles(files);
    });
    open->setShortcut(QKeySequence::Open);
    auto* samplesBtn = new QToolButton(this);
    samplesBtn->setText(tr("Samples"));
    samplesBtn->setPopupMode(QToolButton::InstantPopup);
    auto* sampleMenu = new QMenu(samplesBtn);
    for (const auto& s : samples()) {
        auto* a = sampleMenu->addAction(QString::fromStdString(s.name), this, [this, id = s.id] { loadSample(QString::fromStdString(id)); });
        a->setToolTip(QString::fromStdString(s.note));
    }
    samplesBtn->setMenu(sampleMenu);
    tb->addWidget(samplesBtn);
    tb->addSeparator();
    for (auto [name, label] : std::initializer_list<std::pair<const char*, const char*>>{{"iso", "Iso"}, {"front", "Front"}, {"top", "Top"}, {"right", "Right"}})
        tb->addAction(tr(label), this, [this, v = QString(name)] { viewer->setView(v); });
    tb->addSeparator();
    auto* edges = tb->addAction(tr("Edges"));
    edges->setCheckable(true);
    edges->setChecked(true);
    connect(edges, &QAction::toggled, this, [this](bool on) { viewer->setEdgesVisible(on); });
    auto* xray = tb->addAction(tr("X-ray"));
    xray->setCheckable(true);
    xray->setObjectName("xrayAction");
    connect(xray, &QAction::toggled, this, [this](bool on) { viewer->setXRay(on); });
    tb->addAction(tr("Screenshot"), this, [this] { saveScreenshot(); });

    // ---- sidebar + viewport ----
    auto* split = new QSplitter(Qt::Horizontal, this);
    split->setChildrenCollapsible(false);
    tabs_ = new QTabWidget(split);
    tabs_->setDocumentMode(true);
    tabs_->setMinimumWidth(320);
    tabs_->setMaximumWidth(600);
    auto scroll = [this](QWidget* w) {
        auto* sa = new QScrollArea(tabs_);
        sa->setWidgetResizable(true);
        sa->setFrameShape(QFrame::NoFrame);
        sa->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        sa->setWidget(w);
        return sa;
    };
    viewportHost_ = new QWidget(split);
    auto* vl = new QVBoxLayout(viewportHost_);
    vl->setContentsMargins(0, 0, 0, 0);
    viewer = new Viewport(viewportHost_);
    vl->addWidget(viewer);
    legend = new Legend(viewer);
    legend->move(14, 14);
    busy = new BusyOverlay(viewer);
    picker = new Picker(this, viewer);
    probe = new QLabel(viewer);
    probe->setObjectName("probe");
    probe->setAttribute(Qt::WA_TransparentForMouseEvents);
    probe->hide();
    hintLabel = new QLabel(viewer);
    hintLabel->setObjectName("hint");
    hintLabel->hide();
    emptyState_ = new QWidget(viewer);
    {
        auto* el = new QVBoxLayout(emptyState_);
        auto* t = new QLabel(tr("Drop a STEP, SolidWorks, STL or OBJ file here"), emptyState_);
        t->setObjectName("emptyTitle");
        t->setAlignment(Qt::AlignCenter);
        auto* row = new QHBoxLayout;
        auto* browse = new QPushButton(tr("Browse…"), emptyState_);
        connect(browse, &QPushButton::clicked, open, &QAction::trigger);
        auto* sample = new QPushButton(tr("Open a sample"), emptyState_);
        connect(sample, &QPushButton::clicked, this, [this] { loadSample("beam"); });
        row->addStretch();
        row->addWidget(browse);
        row->addWidget(sample);
        row->addStretch();
        el->addStretch();
        el->addWidget(t);
        el->addLayout(row);
        el->addStretch();
    }

    partPanel = new PartPanel(this);
    structural = new StructuralPanel(this);
    tabs_->addTab(scroll(partPanel), tr("Part"));
    tabs_->addTab(scroll(structural), tr("Structural"));
    thermal = new ThermalPanel(this);
    tabs_->addTab(scroll(thermal), tr("Thermal"));
    airflow = new AirflowPanel(this);
    tabs_->addTab(scroll(airflow), tr("Airflow"));
    connect(tabs_, &QTabWidget::currentChanged, this, &MainWindow::tabChanged);
    split->addWidget(tabs_);
    split->addWidget(viewportHost_);
    split->setStretchFactor(1, 1);
    split->setSizes({400, 1040});
    setCentralWidget(split);

    // ---- status bar ----
    statusLabel_ = new QLabel(tr("Ready."), this);
    statusLabel_->setObjectName("status");
    statusBar()->addWidget(statusLabel_, 1);
    const QString gpu = gpuAvailable() ? tr("GPU: %1").arg(QString::fromStdString(gpuName())) : tr("GPU unavailable: solvers run on the CPU");
    statusBar()->addPermanentWidget(new QLabel(gpu, this));

    // ---- viewport interaction ----
    viewer->onHover = [this](QMouseEvent* e) {
        if (!part) return;
        if (picker->active()) return picker->move(e);
        QString text;
        if (currentTab_ == "structural") text = structural->probeText(e->position());
        else if (currentTab_ == "thermal") text = thermal->probeText(e->position());
        else if (currentTab_ == "airflow") text = airflow->probeText(e->position());
        probe->setVisible(!text.isEmpty());
        if (!text.isEmpty()) {
            probe->setText(text);
            probe->adjustSize();
            probe->move((e->position() + QPointF(14, 14)).toPoint());
        }
    };
    viewer->onClick = [this](QMouseEvent* e) {
        if (picker->active()) picker->click(e);
    };
    viewer->onLeave = [this] {
        probe->hide();
        if (picker->active()) viewer->clearLayer("hover");
    };
    auto* enter = new QShortcut(QKeySequence(Qt::Key_Return), this);
    connect(enter, &QShortcut::activated, this, [this] { if (picker->active()) picker->finish(true); });
    auto* esc = new QShortcut(QKeySequence(Qt::Key_Escape), this);
    connect(esc, &QShortcut::activated, this, [this] { if (picker->active()) picker->finish(false); });
    auto* fit = new QShortcut(QKeySequence(Qt::Key_F), viewer);
    connect(fit, &QShortcut::activated, this, [this] { viewer->setView("iso"); });
    connect(viewer, &Viewport::frame, this, [this](double dt) {
        if (currentTab_ == "structural") structural->tick(dt);
        else if (currentTab_ == "thermal") thermal->tick(dt);
        else if (currentTab_ == "airflow") airflow->tick(dt);
    });
    layoutOverlays();
}

void MainWindow::buildMenus() {
    auto* file = menuBar()->addMenu(tr("&File"));
    file->addAction(tr("Open…"), QKeySequence::Open, this, [this] {
        const auto files = QFileDialog::getOpenFileNames(this, tr("Import part"), QString(), extensionsFilter());
        if (!files.isEmpty()) openFiles(files);
    });
    auto* sm = file->addMenu(tr("Open Sample"));
    for (const auto& s : samples()) sm->addAction(QString::fromStdString(s.name), this, [this, id = s.id] { loadSample(QString::fromStdString(id)); });
    file->addSeparator();
    file->addAction(tr("Save Screenshot…"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_S), this, [this] { saveScreenshot(); });
    file->addSeparator();
    file->addAction(tr("Quit"), QKeySequence::Quit, qApp, &QApplication::quit);

    auto* view = menuBar()->addMenu(tr("&View"));
    int k = 1;
    for (auto [name, label] : std::initializer_list<std::pair<const char*, const char*>>{{"iso", "Isometric"}, {"front", "Front"}, {"top", "Top"}, {"right", "Right"}})
        view->addAction(tr(label), QKeySequence(Qt::CTRL | Qt::Key(Qt::Key_0 + k++)), this, [this, v = QString(name)] { viewer->setView(v); });

    auto* sim = menuBar()->addMenu(tr("&Simulate"));
    sim->addAction(tr("Run Current Study"), QKeySequence(Qt::CTRL | Qt::Key_R), this, [this] {
        setTab("structural");
        structural->runStudy();
    });
    sim->addAction(tr("Run Break Test"), QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_R), this, [this] {
        setTab("structural");
        structural->runBreak();
    });

    auto* help = menuBar()->addMenu(tr("&Help"));
    help->addAction(tr("How Parts Sim Works"), this, [this] { showHelp(); });
    help->addAction(tr("Open-Source Licences"), this, [this] {
        QMessageBox::information(this, tr("Open-source licences"),
                                 tr("Parts Sim %1 is MIT-licensed. It uses Qt (LGPL-3.0), Open CASCADE Technology (LGPL-2.1), wgpu-native (MIT/Apache-2.0) "
                                    "and zlib (zlib licence). SolidWorks reading follows the format research in sldprt-export (MIT) and the cadmpeg "
                                    "specification (CC BY 4.0). See THIRD_PARTY_NOTICES.md.")
                                     .arg(QApplication::applicationVersion()));
    });
}

void MainWindow::layoutOverlays() {
    if (!viewer) return;
    emptyState_->setGeometry(viewer->rect());
    legend->move(14, 14);
    hintLabel->adjustSize();
    hintLabel->move((viewer->width() - hintLabel->width()) / 2, viewer->height() - hintLabel->height() - 14);
    if (busy->isVisible()) busy->move((viewer->width() - busy->width()) / 2, viewer->height() - busy->height() - 40);
    if (picker->isVisible()) picker->QWidget::move((viewer->width() - picker->width()) / 2, 12);
}

void MainWindow::resizeEvent(QResizeEvent* e) {
    QMainWindow::resizeEvent(e);
    layoutOverlays();
}

void MainWindow::closeEvent(QCloseEvent* e) {
    structural->cancelJob();
    thermal->cancel();
    airflow->stop();
    QMainWindow::closeEvent(e);
}

QString MainWindow::tab() const { return currentTab_; }
QString MainWindow::statusText() const { return statusLabel_->text(); }

void MainWindow::setTab(const QString& name) {
    static const QStringList names = {"part", "structural", "thermal", "airflow"};
    const int i = names.indexOf(name);
    if (i >= 0 && tabs_->currentIndex() != i) tabs_->setCurrentIndex(i);
    else tabChanged(i);
}

void MainWindow::tabChanged(int index) {
    static const QStringList names = {"part", "structural", "thermal", "airflow"};
    if (picker->active()) picker->finish(false);
    const QString prev = currentTab_;
    currentTab_ = names.value(index, "part");
    if (prev == "structural" && currentTab_ != "structural") structural->deactivate();
    if (prev == "thermal" && currentTab_ != "thermal") thermal->deactivate();
    if (prev == "airflow" && currentTab_ != "airflow") airflow->deactivate();
    legend->clear();
    probe->hide();
    if (currentTab_ == "structural") structural->activate();
    if (currentTab_ == "thermal") thermal->activate();
    if (currentTab_ == "airflow") airflow->activate();
    if (currentTab_ == "part") partPanel->refresh();
}

void MainWindow::status(const QString& msg, const QString& kind) {
    statusLabel_->setText(msg);
    statusLabel_->setProperty("kind", kind);
    statusLabel_->style()->unpolish(statusLabel_);
    statusLabel_->style()->polish(statusLabel_);
}

void MainWindow::hint(const QString& text) {
    hintLabel->setVisible(!text.isEmpty());
    hintLabel->setText(text);
    layoutOverlays();
    hintLabel->raise();
}

void MainWindow::setXRay(bool on) {
    if (auto* a = findChild<QAction*>("xrayAction")) a->setChecked(on);
    viewer->setXRay(on);
}

void MainWindow::setMaterial(const Material& m) {
    material = m;
    partPanel->refreshMaterial();
    structural->onMaterialChanged();
    thermal->renderMaterial();
    emit materialChanged();
}

void MainWindow::setUnits(const QString& u, bool notify) {
    units = u;
    if (notify) structural->markStale();
    structural->updateMeshInfo();
    partPanel->refresh();
    emit unitsChanged();
}

void MainWindow::applyScale(double s) {
    if (!part) return status(tr("Import a part or open a sample first."), "error");
    if (!(s > 0) || !std::isfinite(s) || s == 1) return;
    const double size = std::max({part->bbox.size[0], part->bbox.size[1], part->bbox.size[2]}) * s;
    if (size > 1e7 || size < 1e-6) return status(tr("That size is outside what the simulations can handle."), "error");
    if (picker->active()) picker->finish(false);
    const Vec3 pivot = scalePart(*part, s);
    structural->onPartScaled(s, pivot);
    thermal->onPartScaled(s, pivot);
    airflow->onPartScaled();
    structural->deactivate();
    viewer->setPart(part);
    setTab(currentTab_);
    partPanel->refresh();
    status(tr("Scaled × %1: now %2 × %3 × %4 %5. Fixtures and loads were kept; run the tests again.")
               .arg(num(s), num(part->bbox.size[0]), num(part->bbox.size[1]), num(part->bbox.size[2]), units));
}

void MainWindow::rotatePart(int axis) {
    if (!part) return;
    MeshSource src;
    src.positions = rotatePositions90(part->vertices, axis);
    src.index = part->tris;
    if (part->brepFaces) src.faceIds = part->faceOf;
    src.name = part->name;
    const bool hadSetup = structural->hasSetup();
    if (loadPart(src, {}, partInfo, nullptr, true) && hadSetup) status(tr("Part rotated - fixtures and loads were cleared."), "warn");
}

bool MainWindow::loadPart(const MeshSource& src, const QString& u, const QString& info, const SampleSetup* setup, bool keepTab) {
    std::shared_ptr<Part> p;
    try {
        p = buildPart(src, {350000, 20});
    } catch (const std::exception& e) {
        status(tr("Could not process the mesh: %1").arg(e.what()), "error");
        return false;
    }
    if (picker->active()) picker->finish(false);
    structural->deactivate();
    part = p;
    partInfo = info;
    if (!u.isEmpty()) setUnits(u, false);
    viewer->setPart(part);
    emptyState_->hide();
    structural->reset(setup);
    thermal->reset();
    airflow->reset(setup ? setup->airflow : std::nullopt);
    partPanel->refresh();
    setWindowTitle(QString::fromStdString(part->name) + " — Parts Sim");
    if (keepTab) setTab(currentTab_);
    emit partLoaded();
    return true;
}

void MainWindow::loadGeneratedPart(MeshSource src, const QString& info) {
    if (loadPart(src, {}, info, nullptr, true))
        status(tr("Loaded “%1”. Add fixtures and loads, then run a static study to check it.").arg(QString::fromStdString(src.name)));
}

void MainWindow::openFiles(const QStringList& paths) {
    if (paths.isEmpty()) return;
    // an assembly opens with its parts found in the same folder
    QString primary = paths.first();
    for (const auto& p : paths)
        if (p.endsWith(".sldasm", Qt::CaseInsensitive)) primary = p;
    const QString name = QFileInfo(primary).fileName();
    const QString dir = QFileInfo(primary).absolutePath();
    busy->showBusy(tr("Reading %1…").arg(name));
    QApplication::processEvents();
    MeshSource src;
    try {
        SiblingReader sibling = [dir](const std::string& fileName) -> Bytes {
            const QString wanted = QString::fromStdString(fileName);
            if (wanted.contains('/') || wanted.contains('\\') || !wanted.endsWith(".sldprt", Qt::CaseInsensitive)) return {};
            QDir d(dir);
            QString match;
            for (const auto& f : d.entryList(QDir::Files))
                if (f.compare(wanted, Qt::CaseInsensitive) == 0) { match = f; break; }
            if (match.isEmpty()) return {};
            std::ifstream in(d.filePath(match).toStdString(), std::ios::binary);
            return Bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        };
        src = importFile(primary.toStdString(), sibling);
    } catch (const std::exception& e) {
        busy->hideBusy();
        status(tr("Import failed: %1").arg(e.what()), "error");
        return;
    }
    busy->hideBusy();
    if (!loadPart(src, QString::fromStdString(src.units), QString::fromStdString(src.info), nullptr, true)) return;
    QStringList notes;
    for (const auto& w : src.warnings) notes << QString::fromStdString(w);
    const std::string matId = matchMaterial(src.material);
    if (!matId.empty()) {
        setMaterial(*findMaterial(matId));
        notes.prepend(tr("Material from SolidWorks: “%1” → %2.").arg(QString::fromStdString(src.material), QString::fromStdString(material.name)));
    } else if (!src.material.empty()) {
        notes.prepend(tr("SolidWorks material “%1” is not in the library - pick the closest one or enter its properties (Part tab).").arg(QString::fromStdString(src.material)));
    }
    if (src.units.empty()) notes.prepend(tr("Loaded %1. Mesh files have no units - check the unit (Part tab) matches your model.").arg(name));
    if (!notes.isEmpty()) status(notes.join(' '), !src.warnings.empty() || src.units.empty() ? "warn" : "");
    else status(tr("Loaded “%1”: %2 triangles, %3 × %4 × %5 %6.").arg(QString::fromStdString(part->name)).arg(part->nTri).arg(num(part->bbox.size[0]), num(part->bbox.size[1]), num(part->bbox.size[2]), units));
}

void MainWindow::loadSample(const QString& id) {
    const Sample* s = findSample(id.toStdString());
    if (!s) return;
    busy->showBusy(tr("Building %1…").arg(QString::fromStdString(s->name)));
    QApplication::processEvents();
    MeshSource src;
    try {
        src = s->make();
        src.name = s->name;
    } catch (const std::exception& e) {
        busy->hideBusy();
        return status(tr("Could not load sample: %1").arg(e.what()), "error");
    }
    busy->hideBusy();
    // build the part first so the setup can select its faces
    std::shared_ptr<Part> p;
    try {
        p = buildPart(src, {350000, 20});
    } catch (const std::exception& e) {
        return status(tr("Could not process the sample: %1").arg(e.what()), "error");
    }
    SampleSetup setup = s->setup(*p);
    if (!loadPart(src, "mm", QString::fromStdString(s->note), &setup, false)) return;
    if (const Material* m = findMaterial(setup.material)) setMaterial(*m);
    if (currentTab_ == "part") setTab("structural");
    else setTab(currentTab_);
    status(tr("%1: %2. Press “Run bend test”.").arg(QString::fromStdString(s->name), QString::fromStdString(s->note)));
}

void MainWindow::dragEnterEvent(QDragEnterEvent* e) {
    if (e->mimeData()->hasUrls()) e->acceptProposedAction();
}

void MainWindow::dropEvent(QDropEvent* e) {
    QStringList paths;
    for (const auto& u : e->mimeData()->urls())
        if (u.isLocalFile()) paths << u.toLocalFile();
    openFiles(paths);
}

void MainWindow::saveScreenshot() {
    if (!part) return;
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::PicturesLocation);
    const QString file = QFileDialog::getSaveFileName(this, tr("Save screenshot"), dir + "/" + QString::fromStdString(part->name) + ".png", tr("PNG image (*.png)"));
    if (file.isEmpty()) return;
    QImage img = viewer->screenshot();
    // include the legend
    QPainter p(&img);
    const double dpr = img.width() / double(std::max(1, viewer->width()));
    if (legend->isVisible()) {
        QPixmap lp = legend->grab();
        p.drawPixmap(QRectF(legend->x() * dpr, legend->y() * dpr, legend->width() * dpr, legend->height() * dpr), lp, lp.rect());
    }
    p.end();
    if (img.save(file)) status(tr("Screenshot saved."));
    else status(tr("Could not save the screenshot."), "error");
}

void MainWindow::showHelp() {
    QDialog d(this);
    d.setWindowTitle(tr("How Parts Sim works"));
    auto* v = new QVBoxLayout(&d);
    auto* t = new QTextBrowser(&d);
    t->setOpenExternalLinks(true);
    t->setHtml(tr(
        "<h2>How Parts Sim works</h2>"
        "<p><b>Linear static (bend test).</b> The part is voxelised into 8-node hexahedral elements with incompatible bending modes and solved with a "
        "multigrid-preconditioned conjugate-gradient solver on the GPU (Metal, Vulkan or DirectX 12), or on all CPU cores. Stresses are recovered at "
        "the element corners and averaged, like the nodal stress plots in commercial FEA.</p>"
        "<p><b>Break test.</b> A progressive-damage model: the voxel that first reaches the material's strength is removed, the model is re-solved, "
        "and this repeats until the load path is severed.</p>"
        "<p><b>SolidWorks files.</b> Parts and assemblies from SolidWorks 2015 or newer open without SolidWorks, using the display mesh saved in the "
        "file. Assemblies find their part files in the same folder.</p>"
        "<p><b>Picking.</b> Add a fixture or load, then click faces (or switch to Brush and paint an area). Drag the round handle on the force arrow "
        "to aim it, or use the direction buttons.</p>"
        "<p style='color:gray'>Everything runs on your computer. Nothing is uploaded. Results are engineering estimates: check critical parts "
        "with a physical test or a certified analysis.</p>"));
    t->setMinimumSize(560, 420);
    v->addWidget(t);
    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok, &d);
    connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
    v->addWidget(bb);
    d.exec();
}

}  // namespace ps
