#include "structuralpanel.hpp"

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QTimer>
#include <QSlider>
#include <QSpinBox>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <set>

#include "gpu/gpu.hpp"
#include "gpu/gpu_fea.hpp"
#include "util/system.hpp"
#include "util/trace.hpp"
#include "mainwindow.hpp"
#include "picker.hpp"
#include "samples.hpp"
#include "studies.hpp"
#include "viewport.hpp"
#include "widgets.hpp"

namespace ps {

namespace {

const QColor FIX_COLOR(0x1a, 0x9f, 0x55);
const QColor LOAD_COLOR(0x9b, 0x46, 0xd4);
const QColor LOAD_SELECTED(0xc0, 0x26, 0xd3);
const QColor WIND_COLOR(0x2a, 0x78, 0xd6);
const QColor CRACK_COLOR(0xe0, 0x26, 0x2b);

QVector3D rgb(const QColor& c) { return QVector3D(float(c.redF()), float(c.greenF()), float(c.blueF())); }

std::vector<int32_t> allTris(const std::vector<Patch>& patches) {
    if (patches.size() == 1) return patches[0].tris;
    std::set<int32_t> s;
    for (const auto& p : patches) s.insert(p.tris.begin(), p.tris.end());
    return std::vector<int32_t>(s.begin(), s.end());
}

// voxels on the longest side: ~5 across chunky sections and ~2 through thin walls, within a budget
int suggestResolution(Part& part, double budget) {
    const auto& size = part.bbox.size;
    const double maxDim = std::max({size[0], size[1], size[2]});
    const double bulk = 3 * part.volume / part.area;
    const double wall = typicalWallThickness(part);
    const double h = std::min(bulk / 5, std::isfinite(wall) ? wall / 2 : 1e300);
    int n = int(std::min(480.0, std::max(32.0, std::round(maxDim / h))));
    auto fits = [&](int n) {
        const double hv = maxDim / n;
        double nodes = 1;
        for (double s : size) nodes *= std::ceil(s / hv) + 1;
        const double voxels = std::max(part.volume / (hv * hv * hv), 0.75 * part.area / (hv * hv));
        return nodes <= 3e5 * std::pow(budget / 8e4, 0.8) && voxels <= budget;
    };
    while (n > 24 && !fits(n)) n -= 4;
    return int(std::lround(n / 4.0) * 4);
}

// voxels of a part at a resolution: its volume plus the partly filled layer along the surface
double estimateVoxels(const Part& part, int res) {
    const auto& size = part.bbox.size;
    const double hv = std::max({size[0], size[1], size[2]}) / res;
    return part.volume / (hv * hv * hv) + 0.5 * part.area / (hv * hv);
}

// the resolution whose (corrected) voxel estimate is closest to a target count
int resolutionFor(const Part& part, double voxels, double factor) {
    int lo = 4, hi = 4000;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (factor * estimateVoxels(part, mid) <= voxels) lo = mid;
        else hi = mid - 1;
    }
    const double below = factor * estimateVoxels(part, lo), above = factor * estimateVoxels(part, lo + 1);
    return std::abs(std::log(above / voxels)) < std::abs(std::log(voxels / below)) ? lo + 1 : lo;
}

QString duration(double s) {
    if (s < 1) return QObject::tr("%1 s").arg(QString::number(s, 'f', 1));
    if (s < 60) return QObject::tr("%1 s").arg(std::lround(s));
    const long m = std::lround(s) / 60, r = std::lround(s) % 60;
    return r ? QObject::tr("%1 min %2 s").arg(m).arg(r) : QObject::tr("%1 min").arg(m);
}

// how many bend tests (one static solve) a study costs, roughly (measured on the samples)
double studyCost(const QString& id, double voxels) {
    if (id == "nonlinear") return 30;
    if (id == "modal") return 15;
    if (id == "buckling") return 10;
    if (id == "drop") return 10 * std::cbrt(voxels / 4e4);  // the time step shrinks with the voxels
    if (id == "dynamic") return 16;
    if (id == "optimize") return 20;
    return 1;
}

}  // namespace

StructuralPanel::StructuralPanel(MainWindow* app) : app_(app) {
    buildUi();
    QTimer::singleShot(1500, this, [this] { calibrate(); });
    studies = createStudies(this);
    renderStudyOptions();
}

StructuralPanel::~StructuralPanel() { cancelJob(); }

bool StructuralPanel::gravity() const { return gravity_->isChecked(); }
int StructuralPanel::resolution() const { return resolution_; }

bool StructuralPanel::useGPU() const {
    const int mode = engine_->currentIndex();
    return mode != 2 && gpuAvailable();
}

// ---------- UI ----------

void StructuralPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(10);
    QVBoxLayout* c;

    root->addWidget(card(tr("Study"), &c, this));
    studyBox_ = new QComboBox(this);
    for (const auto& s : studyInfos()) studyBox_->addItem(s.label);
    connect(studyBox_, &QComboBox::activated, this, [this](int i) { setStudy(i); });
    c->addWidget(studyBox_);
    studyDesc_ = note("", this);
    c->addWidget(studyDesc_);

    optionsCard_ = card("", &optionsLayout_, this);
    root->addWidget(optionsCard_);

    fixturesCard_ = card(tr("Fixtures"), &c, this);
    auto* fh = new QHBoxLayout;
    fh->addStretch();
    auto* addFix = new QPushButton(tr("+ Fixed support"), this);
    connect(addFix, &QPushButton::clicked, this, [this] { addFixture(); });
    fh->addWidget(addFix);
    c->addLayout(fh);
    fixtureList_ = new QListWidget(this);
    fixtureList_->setObjectName("items");

    c->addWidget(fixtureList_);
    root->addWidget(fixturesCard_);

    loadsCard_ = card(tr("Loads"), &c, this);
    auto* lh = new QHBoxLayout;
    lh->addStretch();
    auto* addForce = new QPushButton(tr("+ Force"), this);
    auto* addPressure = new QPushButton(tr("+ Pressure"), this);
    connect(addForce, &QPushButton::clicked, this, [this] { addLoad(Load::Force); });
    connect(addPressure, &QPushButton::clicked, this, [this] { addLoad(Load::Pressure); });
    lh->addWidget(addForce);
    lh->addWidget(addPressure);
    c->addLayout(lh);
    loadList_ = new QListWidget(this);
    loadList_->setObjectName("items");

    connect(loadList_, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row == selectedLoad || row < 0) return;
        selectedLoad = row;
        renderLoadEditor();
        drawOverlays();
    });
    c->addWidget(loadList_);
    loadEditor_ = new QWidget(this);
    new QVBoxLayout(loadEditor_);
    loadEditor_->layout()->setContentsMargins(0, 0, 0, 0);
    c->addWidget(loadEditor_);
    gravity_ = new QCheckBox(tr("Include self-weight (gravity −Y)"), this);
    connect(gravity_, &QCheckBox::toggled, this, [this] { markStale(); });
    c->addWidget(gravity_);
    root->addWidget(loadsCard_);

    root->addWidget(card(tr("Mesh"), &c, this));
    auto* rl = new QHBoxLayout;
    rl->addWidget(new QLabel(tr("Total voxels"), this));
    rl->addStretch();
    voxelsBox_ = new QSpinBox(this);
    voxelsBox_->setRange(1000, 100000000);
    voxelsBox_->setSingleStep(10000);
    voxelsBox_->setGroupSeparatorShown(true);
    voxelsBox_->setKeyboardTracking(false);  // apply when typing is finished
    voxelsBox_->setToolTip(tr("How many voxels the whole part is meshed with. More voxels: finer detail, longer runs."));
    connect(voxelsBox_, &QSpinBox::valueChanged, this, [this](int v) { setTargetVoxels(v); });
    rl->addWidget(voxelsBox_);
    c->addLayout(rl);
    // log scale: every step is the same ratio of voxels; the mesh follows when the slider is released
    voxelsSlider_ = new QSlider(Qt::Horizontal, this);
    voxelsSlider_->setRange(0, 1000);
    auto fromSlider = [this](int v) {
        const double raw = 1000 * std::pow(maxVoxels() / 1000, v / 1000.0);
        const double p = std::pow(10.0, std::floor(std::log10(raw)) - 1);
        return std::round(raw / p) * p;
    };
    connect(voxelsSlider_, &QSlider::valueChanged, this, [this, fromSlider](int v) {
        const double n = fromSlider(v);
        voxelsBox_->blockSignals(true);
        voxelsBox_->setValue(int(n));
        voxelsBox_->blockSignals(false);
        if (!voxelsSlider_->isSliderDown()) setTargetVoxels(n);
    });
    connect(voxelsSlider_, &QSlider::sliderReleased, this, [this, fromSlider] { setTargetVoxels(fromSlider(voxelsSlider_->value())); });
    c->addWidget(voxelsSlider_);
    voxelsChk_ = new QCheckBox(tr("Preview voxel mesh"), this);
    connect(voxelsChk_, &QCheckBox::toggled, this, [this] { showVoxelPreview(); });
    c->addWidget(voxelsChk_);
    auto* el = new QHBoxLayout;
    el->addWidget(new QLabel(tr("Solver"), this));
    engine_ = new QComboBox(this);
    engine_->addItems({tr("Automatic (GPU for large models)"), tr("GPU"), tr("CPU (all cores)")});
    engine_->setItemData(1, gpuAvailable() ? QVariant() : QVariant(0), Qt::UserRole - 1);
    el->addWidget(engine_, 1);
    c->addLayout(el);
    connect(engine_, &QComboBox::activated, this, [this](int) { updateMeshInfo(); });
    etaLabel_ = note("", this);
    etaLabel_->setStyleSheet("font-weight: 600;");
    c->addWidget(etaLabel_);
    meshInfo_ = note("", this);
    c->addWidget(meshInfo_);

    auto* run = new QHBoxLayout;
    runBtn_ = new QPushButton(tr("Run bend test"), this);
    runBtn_->setObjectName("primary");
    runBtn_->setMinimumHeight(36);
    breakBtn_ = new QPushButton(tr("Break test"), this);
    breakBtn_->setMinimumHeight(36);
    connect(runBtn_, &QPushButton::clicked, this, [this] { runStudy(); });
    connect(breakBtn_, &QPushButton::clicked, this, [this] { runBreak(); });
    run->addWidget(runBtn_, 2);
    run->addWidget(breakBtn_, 1);
    root->addLayout(run);

    // ---- linear static results ----
    resultsCard_ = card(tr("Results"), &c, this);
    alert_ = new QLabel(this);
    alert_->setObjectName("alert");
    alert_->setWordWrap(true);
    alert_->setTextFormat(Qt::RichText);
    alert_->hide();
    c->addWidget(alert_);
    kpis_ = new KpiGrid(this);
    c->addWidget(kpis_);
    auto* pl = new QHBoxLayout;
    pl->addWidget(new QLabel(tr("Plot"), this));
    plot_ = new QComboBox(this);
    plot_->addItem(tr("von Mises stress"), "vm");
    plot_->addItem(tr("Displacement"), "disp");
    plot_->addItem(tr("Factor of safety"), "fos");
    plot_->addItem(tr("Max principal stress (tension)"), "p1");
    plot_->addItem(tr("Min principal stress (compression)"), "p3");
    connect(plot_, &QComboBox::activated, this, [this](int i) { view.plot = plot_->itemData(i).toString(); showResults(); });
    pl->addWidget(plot_, 1);
    c->addLayout(pl);
    c->addWidget(new QLabel(tr("Show the part at"), this));
    levelSeg_ = segmented({tr("Applied load"), tr("First yield"), tr("Breaking load")}, 0, [this](int i) {
        static const char* lv[3] = {"applied", "yield", "break"};
        setLevel(lv[i]);
    }, this);
    c->addWidget(levelSeg_);
    bendBreak_ = new QPushButton(tr("▶ Bend until it breaks"), this);
    connect(bendBreak_, &QPushButton::clicked, this, [this] {
        if (!result_ || result_->unreliable || !std::isfinite(result_->lamBreak)) return;
        if (!view.scalePct) scale_->setValue(view.scalePct = 1);
        view.animate = false;
        animateChk_->setChecked(false);
        anim_ = Anim{};
        app_->viewer->clearMarker();
    });
    c->addWidget(bendBreak_);
    levelNote_ = note("", this);
    c->addWidget(levelNote_);
    auto* sl = new QHBoxLayout;
    sl->addWidget(new QLabel(tr("Deformed shape (0 = keep the part's shape)"), this));
    sl->addStretch();
    scaleOut_ = new QLabel(tr("off"), this);
    sl->addWidget(scaleOut_);
    c->addLayout(sl);
    scale_ = new QSlider(Qt::Horizontal, this);
    scale_->setRange(0, 100);
    connect(scale_, &QSlider::valueChanged, this, [this](int v) { view.scalePct = v; showResults(false); });
    c->addWidget(scale_);
    auto* checks = new QGridLayout;
    animateChk_ = new QCheckBox(tr("Animate"), this);
    connect(animateChk_, &QCheckBox::toggled, this, [this](bool on) { view.animate = on; phase_ = 0; showResults(false); });
    auto* bands = new QCheckBox(tr("Contour bands"), this);
    connect(bands, &QCheckBox::toggled, this, [this](bool on) { view.bands = on; applyDisplay(); });
    auto* heat = new QCheckBox(tr("Heat colours"), this);
    connect(heat, &QCheckBox::toggled, this, [this](bool on) { view.heat = on; applyDisplay(); });
    auto* bcs = new QCheckBox(tr("Show loads"), this);
    connect(bcs, &QCheckBox::toggled, this, [this](bool on) { view.bcs = on; drawOverlays(); });
    auto* marker = new QCheckBox(tr("Weakest point"), this);
    marker->setChecked(true);
    connect(marker, &QCheckBox::toggled, this, [this](bool on) { view.marker = on; applyDisplay(); });
    checks->addWidget(animateChk_, 0, 0);
    checks->addWidget(bands, 0, 1);
    checks->addWidget(heat, 1, 0);
    checks->addWidget(bcs, 1, 1);
    checks->addWidget(marker, 2, 0);
    c->addLayout(checks);
    resultsCard_->hide();
    root->addWidget(resultsCard_);

    // ---- break test ----
    breakCard_ = card(tr("Break test"), &c, this);
    breakKpis_ = new KpiGrid(this);
    c->addWidget(breakKpis_);
    c->addWidget(note(tr("Load needed to grow the crack"), this));
    LineChart::Options co;
    co.xLabel = tr("Crack growth step");
    co.formatY = [](double v) { return force(v); };
    breakChart_ = new LineChart(this, co);
    breakChart_->onPick = [this](int i) {
        if (!brk_ || i >= int(brk_->steps.size())) return;
        brk_->current = i;
        breakStep_->setValue(i);
        showBreakStep();
    };
    c->addWidget(breakChart_);
    auto* bl = new QHBoxLayout;
    bl->addWidget(new QLabel(tr("Crack step"), this));
    breakStep_ = new QSlider(Qt::Horizontal, this);
    connect(breakStep_, &QSlider::valueChanged, this, [this](int v) {
        if (!brk_ || v >= int(brk_->steps.size())) return;
        brk_->current = v;
        display_ = "break";
        showBreakStep();
    });
    bl->addWidget(breakStep_, 1);
    breakStepOut_ = new QLabel("0", this);
    bl->addWidget(breakStepOut_);
    c->addLayout(bl);
    auto* br = new QHBoxLayout;
    breakPlay_ = new QPushButton(tr("▶ Play"), this);
    connect(breakPlay_, &QPushButton::clicked, this, [this] {
        if (playing_) { playing_ = false; breakPlay_->setText(tr("▶ Play")); return; }
        if (!brk_ || brk_->steps.empty()) return;
        if (brk_->current >= int(brk_->steps.size()) - 1) breakStep_->setValue(0);
        playing_ = true;
        playT_ = 0;
        breakPlay_->setText(tr("❚❚ Pause"));
    });
    br->addWidget(breakPlay_);
    auto* bs = new QCheckBox(tr("Stress colours"), this);
    bs->setChecked(true);
    connect(bs, &QCheckBox::toggled, this, [this](bool on) { view.breakStress = on; showBreakStep(); });
    br->addWidget(bs);
    br->addStretch();
    c->addLayout(br);
    breakCard_->hide();
    root->addWidget(breakCard_);

    // ---- other studies ----
    studyCard = card("", &studyBody, this);
    studyTitle = new QLabel(this);
    studyTitle->setObjectName("cardTitle");
    studyBody->insertWidget(0, studyTitle);
    studyAlert = new QLabel(this);
    studyAlert->setObjectName("alert");
    studyAlert->setWordWrap(true);
    studyAlert->hide();
    studyBody->addWidget(studyAlert);
    studyKpis = new KpiGrid(this);
    studyBody->addWidget(studyKpis);
    studyContent = new QWidget(this);
    auto* sc = new QVBoxLayout(studyContent);
    sc->setContentsMargins(0, 0, 0, 0);
    sc->setSpacing(8);
    studyBody->addWidget(studyContent);
    studyCard->hide();
    root->addWidget(studyCard);
    root->addStretch();
    compact(this);
    for (auto* b : {runBtn_, breakBtn_}) b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
}

void StructuralPanel::setStudy(int index) {
    if (index == study_) return;
    cancelJob();
    playing_ = false;
    if (study_ > 0) studies[study_ - 1]->leave();
    study_ = index;
    studyBox_->setCurrentIndex(index);
    if (app_->part) {
        resolution_ = suggestResolution(*app_->part, studyInfos()[index].budget);
        targetVoxels_ = voxelFactor_ * estimateVoxels(*app_->part, resolution_);
        syncVoxelControls();
        updateMeshInfo();
    }
    display_ = index == 0 ? (result_ ? "results" : brk_ && !brk_->steps.empty() ? "break" : "setup")
                          : studies[index - 1]->hasResult() ? "study" : "setup";
    app_->viewer->clearLayer("shape");
    app_->viewer->clearLayer("voxels");
    app_->viewer->setPartVisible(true);
    renderStudyOptions();
    applyDisplay();
}

void StructuralPanel::selectStudy(const QString& id) {
    const auto& infos = studyInfos();
    for (size_t i = 0; i < infos.size(); i++)
        if (infos[i].id == id) return setStudy(int(i));
}

void StructuralPanel::renderStudyOptions() {
    const auto& info = studyInfos()[study_];
    studyDesc_->setText(info.desc);
    runBtn_->setText(info.button);
    breakBtn_->setVisible(study_ == 0);
    fixturesCard_->setVisible(info.fixtures);
    loadsCard_->setVisible(info.loads);
    resultsCard_->setVisible(study_ == 0 && result_.has_value());
    breakCard_->setVisible(study_ == 0 && brk_.has_value());
    studyCard->setVisible(study_ > 0 && studies[study_ - 1]->hasResult());
    while (QLayoutItem* it = optionsLayout_->takeAt(0)) {
        if (it->widget()) it->widget()->deleteLater();
        else if (it->layout()) {
            while (QLayoutItem* sub = it->layout()->takeAt(0)) {
                if (sub->widget()) sub->widget()->deleteLater();
                delete sub;
            }
        }
        delete it;
    }
    const bool has = study_ > 0 && studies[study_ - 1]->options(optionsLayout_);
    optionsCard_->setVisible(has);
    if (has) compact(optionsCard_);
}

void StructuralPanel::reset(const SampleSetup* setup) {
    cancelJob();
    playing_ = false;
    voxelsChk_->blockSignals(true);
    voxelsChk_->setChecked(false);
    voxelsChk_->blockSignals(false);
    app_->viewer->clearLayer("meshpreview");
    fixtures.clear();
    loads.clear();
    selectedLoad = -1;
    models_.clear();
    result_.reset();
    brk_.reset();
    anim_.reset();
    display_ = "setup";
    stale_ = false;
    for (auto& s : studies) s->clear();
    app_->viewer->clearLayer("shape");
    resultsCard_->hide();
    breakCard_->hide();
    studyCard->hide();
    if (setup) {
        for (const auto& f : setup->fixtures) {
            Fixture fx = f;
            fx.patches.erase(std::remove_if(fx.patches.begin(), fx.patches.end(), [](const Patch& p) { return p.tris.empty(); }), fx.patches.end());
            fixtures.push_back(fx);
        }
        for (const auto& l : setup->loads) loads.push_back(l);
        selectedLoad = loads.empty() ? -1 : 0;
    }
    voxelFactor_ = 1;
    if (app_->part) {
        resolution_ = suggestResolution(*app_->part, studyInfos()[study_].budget);
        targetVoxels_ = estimateVoxels(*app_->part, resolution_);
        syncVoxelControls();
    }
    renderLists();
    renderStudyOptions();
    updateMeshInfo();
    if (app_->tab() == "structural") activate();
}

void StructuralPanel::markStale() {
    if (job) cancelJob();
    const bool has = result_ || brk_ || (study_ > 0 && studies[study_ - 1]->hasResult());
    if (has) {
        stale_ = true;
        app_->status(tr("Setup changed - run the study again to update the results."), "warn");
    }
    drawOverlays();
}

void StructuralPanel::onMaterialChanged() {
    markStale();
    if (study_ > 0) renderStudyOptions();
}

void StructuralPanel::onPartScaled(double s, const Vec3& pivot) {
    cancelJob();
    playing_ = false;
    auto move = [&](Vec3 c) {
        for (int d = 0; d < 3; d++) c[d] = pivot[d] + (c[d] - pivot[d]) * s;
        return c;
    };
    for (auto& f : fixtures)
        for (auto& p : f.patches)
            if (p.clip) p.clip = Clip{move(p.clip->center), p.clip->radius * s};
    for (auto& l : loads) {
        for (auto& p : l.patches)
            if (p.clip) p.clip = Clip{move(p.clip->center), p.clip->radius * s};
        if (l.type == Load::Wind) {
            for (auto& v : l.forces) v = float(v * s * s);  // same pressure on s^2 the area
            l.magnitude *= s * s;
        }
    }
    models_.clear();
    result_.reset();
    brk_.reset();
    anim_.reset();
    display_ = "setup";
    for (auto& st : studies) st->clear();
    resultsCard_->hide();
    breakCard_->hide();
    studyCard->hide();
    renderLists();
    renderStudyOptions();
    updateMeshInfo();
}

// ---------- fixtures & loads ----------

void StructuralPanel::renderLists() {
    auto row = [this](QListWidget* list, const QString& name, const QString& meta, const QColor& swatch, std::function<void()> onAdd,
                      std::function<void()> onDelete) {
        auto* item = new QListWidgetItem(list);
        auto* w = new QWidget(list);
        auto* h = new QHBoxLayout(w);
        h->setContentsMargins(6, 2, 4, 2);
        auto* sw = new QLabel(w);
        sw->setFixedSize(10, 10);
        sw->setStyleSheet(QString("background:%1;border-radius:3px").arg(swatch.name()));
        h->addWidget(sw);
        auto* n = new QLabel(name, w);
        h->addWidget(n, 1);
        auto* m = new QLabel(meta, w);
        m->setObjectName("meta");
        h->addWidget(m);
        if (onAdd) {
            auto* add = new QToolButton(w);
            add->setText("+");
            add->setToolTip(tr("Add more areas"));
            connect(add, &QToolButton::clicked, this, onAdd);
            h->addWidget(add);
        }
        auto* del = new QToolButton(w);
        del->setText("×");
        del->setToolTip(tr("Delete"));
        connect(del, &QToolButton::clicked, this, onDelete);
        h->addWidget(del);
        item->setSizeHint(w->sizeHint());
        list->setItemWidget(item, w);
    };
    fixtureList_->clear();
    for (size_t i = 0; i < fixtures.size(); i++) {
        const auto& f = fixtures[i];
        row(fixtureList_, QString::fromStdString(f.name), tr("%n area(s)", "", int(f.patches.size())), FIX_COLOR,
            [this, i] { pick(int(i), -1, false); },
            [this, i] {
                QMetaObject::invokeMethod(this, [this, i] {
                    if (i < fixtures.size()) fixtures.erase(fixtures.begin() + long(i));
                    markStale();
                    renderLists();
                }, Qt::QueuedConnection);
            });
    }
    loadList_->blockSignals(true);
    loadList_->clear();
    for (size_t i = 0; i < loads.size(); i++) {
        const auto& l = loads[i];
        const QString meta = l.type == Load::Pressure ? num(l.magnitude) + " MPa" : force(l.magnitude);
        row(loadList_, QString::fromStdString(l.name), meta, l.type == Load::Wind ? WIND_COLOR : LOAD_COLOR, {},
            [this, i] {
                QMetaObject::invokeMethod(this, [this, i] {
                    if (i < loads.size()) loads.erase(loads.begin() + long(i));
                    if (selectedLoad >= int(loads.size())) selectedLoad = int(loads.size()) - 1;
                    markStale();
                    renderLists();
                }, Qt::QueuedConnection);
            });
    }
    loadList_->setCurrentRow(selectedLoad);
    loadList_->blockSignals(false);
    fitList(fixtureList_);
    fitList(loadList_);
    renderLoadEditor();
    drawOverlays();
}

Frame StructuralPanel::loadFrame(const Load& l) const {
    const auto tris = allTris(l.patches);
    std::optional<Clip> clip;
    if (l.patches.size() == 1) clip = l.patches[0].clip;
    return patchFrame(*app_->part, tris, clip);
}

void StructuralPanel::renderLoadEditor() {
    auto* lay = static_cast<QVBoxLayout*>(loadEditor_->layout());
    while (QLayoutItem* it = lay->takeAt(0)) {
        if (it->widget()) it->widget()->deleteLater();
        delete it;
    }
    if (selectedLoad < 0 || selectedLoad >= int(loads.size())) { loadEditor_->hide(); return; }
    loadEditor_->show();
    Load& l = loads[selectedLoad];
    auto* box = new QFrame(loadEditor_);
    box->setObjectName("editor");
    auto* v = new QVBoxLayout(box);
    v->setContentsMargins(10, 8, 10, 8);
    v->addWidget(new QLabel(tr("Name"), box));
    auto* name = new QLineEdit(QString::fromStdString(l.name), box);
    connect(name, &QLineEdit::editingFinished, this, [this, name] {
        if (selectedLoad < 0) return;
        if (!name->text().isEmpty()) loads[selectedLoad].name = name->text().toStdString();
        QMetaObject::invokeMethod(this, [this] { renderLists(); }, Qt::QueuedConnection);
    });
    v->addWidget(name);
    if (l.type == Load::Wind) {
        v->addWidget(note(tr("Surface pressure from the airflow study. Net aerodynamic force %1.").arg(force(l.magnitude)), box));
    } else {
        v->addWidget(new QLabel(l.type == Load::Force ? tr("Total force (N)") : tr("Pressure (MPa)"), box));
        auto* mag = numberBox(l.magnitude, 0, 1e12, 1, 4, [this](double x) {
            if (selectedLoad < 0) return;
            loads[selectedLoad].magnitude = x;
            markStale();
            QMetaObject::invokeMethod(this, [this] { renderLists(); }, Qt::QueuedConnection);
        }, box);
        v->addWidget(mag);
    }
    if (l.type == Load::Force) {
        v->addWidget(new QLabel(tr("Direction"), box));
        auto* grid = new QGridLayout;
        auto setDir = [this](Vec3 d) {
            const double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
            if (!(len > 1e-12)) return app_->status(tr("Choose a nonzero force direction."), "error");
            for (auto& x : d) x /= len;
            loads[selectedLoad].dir = d;
            markStale();
            QMetaObject::invokeMethod(this, [this] { renderLoadEditor(); drawOverlays(); }, Qt::QueuedConnection);
        };
        auto btn = [&](const QString& label, std::function<Vec3()> d, const QString& tip = {}) {
            auto* b = new QPushButton(label, box);
            b->setToolTip(tip);
            connect(b, &QPushButton::clicked, this, [setDir, d] { setDir(d()); });
            return b;
        };
        auto normal = [this] { return loadFrame(loads[selectedLoad]).normal; };
        grid->addWidget(btn(tr("Push ⊥"), [normal] { auto n = normal(); return Vec3{-n[0], -n[1], -n[2]}; }, tr("Into the surface")), 0, 0);
        grid->addWidget(btn(tr("Pull ⊥"), normal, tr("Out of the surface")), 0, 1);
        grid->addWidget(btn(tr("Reverse"), [this] { auto d = loads[selectedLoad].dir; return Vec3{-d[0], -d[1], -d[2]}; }), 0, 2);
        auto* area = new QPushButton(tr("+ Area"), box);
        connect(area, &QPushButton::clicked, this, [this] { pick(-1, selectedLoad, false); });
        grid->addWidget(area, 0, 3);
        const char* names[6] = {"+X", "−X", "+Y", "−Y", "+Z", "−Z"};
        for (int k = 0; k < 6; k++) {
            Vec3 d{0, 0, 0};
            d[k / 2] = k % 2 ? -1 : 1;
            grid->addWidget(btn(names[k], [d] { return d; }), 1 + k / 4, k % 4);
        }
        v->addLayout(grid);
        auto* vec = new QHBoxLayout;
        std::array<QDoubleSpinBox*, 3> comps;
        for (int d = 0; d < 3; d++) {
            comps[d] = new QDoubleSpinBox(box);
            comps[d]->setRange(-1e6, 1e6);
            comps[d]->setDecimals(3);
            comps[d]->setSingleStep(0.1);
            comps[d]->setValue(l.dir[d]);
            comps[d]->setKeyboardTracking(false);
            vec->addWidget(comps[d]);
        }
        for (int d = 0; d < 3; d++)
            connect(comps[d], &QDoubleSpinBox::valueChanged, this, [comps, setDir] { setDir({comps[0]->value(), comps[1]->value(), comps[2]->value()}); });
        v->addLayout(vec);
        v->addWidget(note(tr("Tip: drag the round handle on the arrow in the view to aim the force."), box));
    } else if (l.type == Load::Pressure) {
        v->addWidget(note(tr("Acts normal to the selected faces; positive pushes into the part."), box));
        auto* area = new QPushButton(tr("+ Area"), box);
        connect(area, &QPushButton::clicked, this, [this] { pick(-1, selectedLoad, false); });
        v->addWidget(area);
    }
    lay->addWidget(box);
    compact(box);
}

void StructuralPanel::pick(int fi, int li, bool isNew) {
    if (!app_->part) return;
    display_ = "setup";
    applyDisplay();
    const size_t before = fi >= 0 ? fixtures[fi].patches.size() : loads[li].patches.size();
    const std::optional<Vec3> beforeDir = li >= 0 ? std::optional<Vec3>(loads[li].dir) : std::nullopt;
    Picker::Session s;
    s.title = fi >= 0 ? QString::fromStdString(fixtures[fi].name) : QString::fromStdString(loads[li].name);
    s.color = fi >= 0 ? FIX_COLOR : LOAD_SELECTED;
    s.onPatch = [this, fi, li](const Patch& p) {
        auto& patches = fi >= 0 ? fixtures[fi].patches : loads[li].patches;
        // clicking the same face twice adds nothing
        if (!p.clip)
            for (const auto& o : patches)
                if (!o.clip && o.tris == p.tris) return;
        patches.push_back(p);
        if (li >= 0 && loads[li].type == Load::Force && isNewForce_.count(li)) {
            const Frame fr = loadFrame(loads[li]);
            if (fr.normal[0] * fr.normal[0] + fr.normal[1] * fr.normal[1] + fr.normal[2] * fr.normal[2] > 1e-18)
                loads[li].dir = {-fr.normal[0], -fr.normal[1], -fr.normal[2]};
        }
        markStale();
        renderLists();
    };
    s.onDone = [this, fi, li, before, beforeDir, isNew](bool commit) {
        auto& patches = fi >= 0 ? fixtures[fi].patches : loads[li].patches;
        if (!commit) {
            patches.resize(before);
            if (beforeDir) loads[li].dir = *beforeDir;
        }
        if (li >= 0) isNewForce_.erase(li);
        if (patches.empty() && isNew) {
            if (fi >= 0) fixtures.erase(fixtures.begin() + fi);
            else {
                loads.erase(loads.begin() + li);
                if (selectedLoad >= int(loads.size())) selectedLoad = int(loads.size()) - 1;
            }
        }
        renderLists();
        drawOverlays();
    };
    app_->picker->start(std::move(s));
}

void StructuralPanel::addFixture() {
    if (!app_->part) return app_->status(tr("Import a part or open a sample first."), "error");
    Fixture f;
    f.name = tr("Fixed support %1").arg(fixtures.size() + 1).toStdString();
    fixtures.push_back(f);
    renderLists();
    pick(int(fixtures.size()) - 1, -1, true);
}

void StructuralPanel::addLoad(Load::Type type) {
    if (!app_->part) return app_->status(tr("Import a part or open a sample first."), "error");
    int n = 1;
    for (const auto& l : loads) n += l.type == type;
    Load l;
    l.type = type;
    l.name = (type == Load::Force ? tr("Force %1") : tr("Pressure %1")).arg(n).toStdString();
    l.magnitude = type == Load::Force ? 500 : 1;
    l.dir = {0, -1, 0};
    loads.push_back(l);
    selectedLoad = int(loads.size()) - 1;
    if (type == Load::Force) isNewForce_.insert(selectedLoad);
    renderLists();
    pick(-1, selectedLoad, true);
}

void StructuralPanel::drawOverlays() {
    Viewport* v = app_->viewer;
    v->clearLayer("overlays");
    const bool show = app_->tab() == "structural" && (display_ == "setup" || view.bcs || app_->picker->active());
    if (!app_->part || !show) return;
    const Part& part = *app_->part;
    const auto& info = studyInfos()[study_];
    const float diag = float(part.bbox.diag);
    std::vector<Geometry> items;
    std::vector<Viewport::Handle> handles;
    // a few glyphs spread over a patch
    auto spread = [&](const std::vector<Patch>& patches, int n) {
        std::vector<std::pair<QVector3D, int>> pts;
        const auto tris = allTris(patches);
        double area = 0;
        for (int32_t t : tris) area += part.triArea[t];
        if (tris.empty()) return pts;
        for (const auto& p : patches) {
            const auto s = sampleTriangles(part, p.tris, std::sqrt(area / (n * 2)), p.clip);
            for (size_t i = 0; i < s.weights.size(); i++) pts.push_back({QVector3D(s.points[3 * i], s.points[3 * i + 1], s.points[3 * i + 2]), s.tris[i]});
        }
        if (int(pts.size()) <= n) return pts;
        std::vector<std::pair<QVector3D, int>> out;
        for (int i = 0; i < n; i++) out.push_back(pts[size_t((i + 0.5) * pts.size() / n)]);
        return out;
    };
    if (info.fixtures)
        for (const auto& f : fixtures) {
            const auto tris = allTris(f.patches);
            if (tris.empty()) continue;
            items.push_back(shapes::patch(part, tris, rgb(FIX_COLOR), 0.5f));
            for (auto& [p, t] : spread(f.patches, 6))
                items.push_back(shapes::anchor(p, QVector3D(part.triNormal[3 * t], part.triNormal[3 * t + 1], part.triNormal[3 * t + 2]), diag * 0.03f, rgb(FIX_COLOR)));
        }
    if (info.loads)
        for (size_t i = 0; i < loads.size(); i++) {
            const Load& l = loads[i];
            const bool sel = int(i) == selectedLoad;
            const QColor color = sel ? LOAD_SELECTED : LOAD_COLOR;
            if (l.type == Load::Wind) {
                const QVector3D c(float((part.bbox.min[0] + part.bbox.max[0]) / 2), float((part.bbox.min[1] + part.bbox.max[1]) / 2), float((part.bbox.min[2] + part.bbox.max[2]) / 2));
                const QVector3D d = QVector3D(float(l.dir[0]), float(l.dir[1]), float(l.dir[2])).normalized();
                items.push_back(shapes::arrow(c - d * diag * 0.05f, d, diag * 0.25f, rgb(WIND_COLOR)));
                continue;
            }
            const auto tris = allTris(l.patches);
            if (tris.empty()) continue;
            items.push_back(shapes::patch(part, tris, rgb(color), sel ? 0.55f : 0.4f));
            double area = 0;
            for (int32_t t : tris) area += part.triArea[t];
            const int n = std::max(1, std::min(7, int(std::lround(std::sqrt(area) / (0.06 * diag)))));
            if (!(sel && l.type == Load::Force && n <= 2))
                for (auto& [p, t] : spread(l.patches, n)) {
                    const QVector3D d = l.type == Load::Pressure ? -QVector3D(part.triNormal[3 * t], part.triNormal[3 * t + 1], part.triNormal[3 * t + 2])
                                                                  : QVector3D(float(l.dir[0]), float(l.dir[1]), float(l.dir[2]));
                    items.push_back(shapes::arrow(p, d, diag * 0.07f, rgb(color)));
                }
            if (sel && l.type == Load::Force) {
                const Frame fr = loadFrame(l);
                const QVector3D tip(float(fr.center[0]), float(fr.center[1]), float(fr.center[2]));
                const QVector3D dir = QVector3D(float(l.dir[0]), float(l.dir[1]), float(l.dir[2])).normalized();
                const float len = diag * 0.18f;
                items.push_back(shapes::arrow(tip, dir, len, rgb(color), len * 0.03f));
                auto handle = shapes::sphere(tip - dir * len, len * 0.075f, QVector3D(1, 1, 1));
                items.push_back(std::move(handle));
                Viewport::Handle h;
                h.center = tip - dir * len;
                h.radius = len * 0.075f;
                h.pivot = tip;
                const int li = int(i);
                h.onDrag = [this, li, tip](const QVector3D& p) {
                    QVector3D d = tip - p;
                    if (d.lengthSquared() < 1e-12f) return;
                    d.normalize();
                    loads[li].dir = {d.x(), d.y(), d.z()};
                    drawOverlays();
                };
                h.onEnd = [this] { markStale(); renderLoadEditor(); };
                handles.push_back(h);
            }
        }
    v->setLayer("overlays", std::move(items));
    v->setHandles(std::move(handles));
}

// ---------- model & runs ----------

std::shared_ptr<StructuralModel> StructuralPanel::modelFor(int res) {
    auto it = models_.find(res);
    if (it != models_.end() && it->second->part == app_->part) return it->second;
    auto m = std::make_shared<StructuralModel>(app_->part, res);
    models_[res] = m;
    while (models_.size() > 3) models_.erase(models_.begin());
    return m;
}

std::shared_ptr<StructuralModel> StructuralPanel::model() { return modelFor(resolution()); }

void StructuralPanel::updateMeshInfo() {
    if (!app_->part) { meshInfo_->clear(); etaLabel_->clear(); return; }
    Part& p = *app_->part;
    const int res = resolution();
    const double hv = std::max({p.bbox.size[0], p.bbox.size[1], p.bbox.size[2]}) / res;
    const double wall = typicalWallThickness(p);
    QString text = tr("Voxel size ≈ %1 %2 (%3 on the longest side).").arg(num(hv), app_->units).arg(res);
    if (std::isfinite(wall))
        text += tr(" Walls ≈ %1 %2 thick (%3 voxels across)%4.")
                    .arg(num(wall), app_->units, num(wall / hv), wall < 1.5 * hv ? tr("; thinner walls are kept as connected layers") : QString());
    if (auto m = cachedModel(res)) text += tr(" This mesh: %1 voxels.").arg(QLocale().toString(m->voxelCount));
    meshInfo_->setText(text);
    etaLabel_->setText(runEstimate());
}

void StructuralPanel::setTargetVoxels(double n, bool markChanged) {
    targetVoxels_ = std::clamp(n, 1000.0, maxVoxels());
    const int before = resolution_;
    if (app_->part) resolution_ = resolutionFor(*app_->part, targetVoxels_, voxelFactor_);
    syncVoxelControls();
    if (markChanged && resolution_ != before) {
        markStale();
        showVoxelPreview();
    }
    updateMeshInfo();
}

QString StructuralPanel::meshSummary() const { return etaLabel_->text() + " | " + meshInfo_->text(); }

void StructuralPanel::setEngine(int index) {
    engine_->setCurrentIndex(index);
    updateMeshInfo();
}

double StructuralPanel::maxVoxels() const {
    // about 1.2 kB per voxel for the solver and its multigrid levels; keep 60% of the memory free
    double ram = double(physicalMemory());
    if (!(ram > 0)) ram = 8e9;
    return std::clamp(0.4 * ram / 1200, 1e5, 1e8);
}

void StructuralPanel::syncVoxelControls() {
    voxelsBox_->blockSignals(true);
    voxelsBox_->setMaximum(int(maxVoxels()));
    voxelsBox_->setValue(int(std::lround(targetVoxels_)));
    voxelsBox_->blockSignals(false);
    voxelsSlider_->blockSignals(true);
    voxelsSlider_->setValue(int(std::lround(1000 * std::log(targetVoxels_ / 1000) / std::log(maxVoxels() / 1000))));
    voxelsSlider_->blockSignals(false);
}

void StructuralPanel::learnVoxelFactor(const StructuralModel& m) {
    if (!app_->part || m.part != app_->part || !(m.voxelCount > 0)) return;
    const double est = estimateVoxels(*m.part, m.resolution);
    if (est > 0) voxelFactor_ = m.voxelCount / est;
}

void StructuralPanel::recordRunTime(int voxels, const std::string& engine, double seconds) {
    if (voxels < 2000 || !(seconds > 0)) return;
    const int e = engine == "GPU" ? 1 : 0;
    const double rate = seconds / voxels;
    secPerVoxel_[e] = measured_[e] ? 0.5 * (secPerVoxel_[e] + rate) : rate;
    measured_[e] = true;
    updateMeshInfo();
}

QString StructuralPanel::runEstimate() const {
    if (!app_->part) return {};
    auto m = cachedModel(resolution_);
    const double n = m ? m->voxelCount : targetVoxels_;
    const QString id = studyInfos()[study_].id;
    const double cost = studyCost(id, n);
    const double cpu = cost * (0.05 + secPerVoxel_[0] * n), gpu = cost * (0.05 + secPerVoxel_[1] * n);
    const int mode = engine_->currentIndex();
    const bool gpuOk = gpuAvailable() && id != "drop";  // the drop test runs on the CPU
    const bool onGPU = gpuOk && (mode == 1 || (mode == 0 && n >= AUTO_GPU_VOXELS));
    QString text;
    if (gpuOk)
        text = onGPU ? tr("Estimated run ≈ %1 on the GPU (used) · %2 on the CPU").arg(duration(gpu), duration(cpu))
                     : tr("Estimated run ≈ %1 on the CPU (used) · %2 on the GPU").arg(duration(cpu), duration(gpu));
    else text = tr("Estimated run ≈ %1 on the CPU").arg(duration(cpu));
    if (id == "static") text += tr("; break test ≈ %1").arg(duration(20 * (onGPU ? gpu : cpu)));
    if (!measured_[0] && !measured_[1] && !calibrated_) text += calibration_ ? tr(" (measuring this computer…)") : tr(" (rough)");
    return text + ".";
}

void StructuralPanel::calibrate() {
    // a small bend test on each engine at start-up, so the estimates fit this computer
    struct Out { double rate[2] = {-1, -1}; };
    calibration_ = runJob<Out>(this,
        [](JobControl& ctl) {
            Out o;
            // cantilever beam fixed at x = 0 and loaded at the far end
            auto solveBeam = [](int nx, int ny, int nz, bool gpu) {
                const int NX = nx + 1, NY = ny + 1, NZ = nz + 1;
                std::vector<uint8_t> bc(3 * size_t(NX) * NY * NZ, 0);
                std::vector<double> f(bc.size(), 0.0);
                for (int k = 0; k < NZ; k++)
                    for (int j = 0; j < NY; j++) {
                        const size_t n0 = size_t(NX) * (j + size_t(NY) * k);
                        bc[3 * n0] = bc[3 * n0 + 1] = bc[3 * n0 + 2] = 1;
                        f[3 * (nx + n0) + 1] = -1.0 / (NY * NZ);
                    }
                VoxelFEA fea({nx, ny, nz}, std::vector<float>(size_t(nx) * ny * nz, 1.f), 0.3, bc, gpu ? GPU_COARSEST_DOF : 1100);
                SolveOptions so;
                so.tol = 1e-6;
                so.maxIter = 400;
                if (gpu) GpuFeaSolver(fea).solve(f, so);
                else fea.solve(f, so);
            };
            const int nx = 200, ny = 18, nz = 18;
            for (int e = 0; e < 2; e++) {
                if (e == 1 && !gpuAvailable()) break;
                ctl.check();
                // the first GPU solve compiles the shaders (once per session): keep that out of the rate
                if (e) solveBeam(24, 4, 4, true);
                const auto t0 = std::chrono::steady_clock::now();
                solveBeam(nx, ny, nz, e == 1);
                // a real part needs voxelizing and results mapping, and its thin features and corners
                // take more iterations than this plain beam: real parts run about 2.5x slower per voxel
                o.rate[e] = 2.5 * std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / (nx * ny * nz);
            }
            return o;
        },
        [this](Out o) {
            calibration_.reset();
            for (int e = 0; e < 2; e++)
                if (!measured_[e] && o.rate[e] > 0) secPerVoxel_[e] = o.rate[e];
            calibrated_ = o.rate[0] > 0;
            updateMeshInfo();
        },
        [this](QString, bool) { calibration_.reset(); }, {});
}

bool StructuralPanel::checkSetup(bool requireFixtures, bool requireLoads) {
    if (!app_->part) { app_->status(tr("Import a part or open a sample first."), "error"); return false; }
    try {
        validateMaterial(app_->material);
    } catch (const std::exception& e) {
        app_->status(e.what(), "error");
        return false;
    }
    const auto& info = studyInfos()[study_];
    if (requireFixtures && info.fixtures && !info.fixturesOptional && std::none_of(fixtures.begin(), fixtures.end(), [](const Fixture& f) { return !f.patches.empty(); })) {
        app_->status(tr("Add at least one fixed support (click “+ Fixed support”, then click faces)."), "error");
        return false;
    }
    if (requireLoads && info.loads && loads.empty() && !gravity()) {
        app_->status(tr("Add a force or pressure load (or enable self-weight)."), "error");
        return false;
    }
    return true;
}

void StructuralPanel::addWindLoad(const std::vector<float>& forces, const Vec3& net) {
    loads.erase(std::remove_if(loads.begin(), loads.end(), [](const Load& l) { return l.type == Load::Wind; }), loads.end());
    Load l;
    l.type = Load::Wind;
    l.name = tr("Wind load (airflow)").toStdString();
    l.forces = forces;
    l.magnitude = std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]);
    l.dir = l.magnitude > 0 ? Vec3{net[0] / l.magnitude, net[1] / l.magnitude, net[2] / l.magnitude} : Vec3{0, -1, 0};
    loads.push_back(std::move(l));
    selectedLoad = int(loads.size()) - 1;
    if (study_ != 0 && !studyInfos()[study_].loads) setStudy(0);
    markStale();
    renderLists();
}

void StructuralPanel::cancelJob() {
    const bool active = bool(job);
    if (job) job->cancel();
    job.reset();
    if (active) {
        app_->busy->hideBusy();
        if (brk_ && !brk_->done) {
            brk_->done = true;
            brk_->reason = "stopped";
        }
        app_->status(tr("Simulation cancelled."), "warn");
    }
}

void StructuralPanel::runStudy() {
    if (study_ == 0) return run();
    studies[study_ - 1]->run();
}

// Everything a job needs, copied on the UI thread so the solver thread never touches panel state.
struct Snapshot {
    std::shared_ptr<Part> part;
    std::vector<Fixture> fixtures;
    std::vector<Load> loads;
    bool gravity;
    Material material;
    double toMeters;
    QString units;
    int res;
    std::shared_ptr<StructuralModel> cached;
    bool useGPU;
    bool requireFixtures, requireLoads, fixturesUsed, loadsUsed, fixturesOptional;
};

static StructuralPanel::Prepared prepareOn(const Snapshot& s, JobControl& ctl) {
    ctl.progress(0, "Voxelizing part…");
    StructuralPanel::Prepared p;
    p.model = s.cached ? s.cached : std::make_shared<StructuralModel>(s.part, s.res);
    ctl.check();
    if (!p.model->voxelCount) throw std::runtime_error("The part produced no voxels. Increase the mesh resolution.");
    p.asm_ = p.model->assemble(s.fixturesUsed ? s.fixtures : std::vector<Fixture>{}, s.loadsUsed ? s.loads : std::vector<Load>{},
                               s.loadsUsed && s.gravity, s.material, s.toMeters);
    if (s.requireFixtures && s.fixturesUsed && !s.fixturesOptional && !p.asm_.fixedNodes)
        throw std::runtime_error("No voxels found under the fixtures. Try a higher mesh resolution.");
    if (s.requireLoads && s.loadsUsed) {
        bool any = false;
        for (size_t i = 0; i < p.asm_.f.size() && !any; i++) any = p.asm_.f[i] != 0 && !p.asm_.bc[i];
        if (!any) throw std::runtime_error("No nonzero load reaches a free node. Set a load magnitude and select an area away from the fixed support.");
    }
    p.material = s.material;
    p.toMeters = s.toMeters;
    p.units = s.units;
    auto& in = p.input;
    in.dims = p.model->grid.dims;
    in.density = p.model->density;
    in.bc = p.asm_.bc;
    in.f = p.asm_.f;
    in.nu = s.material.nu;
    in.E = s.material.E * 1e9;
    in.h = p.model->grid.h * s.toMeters;
    in.rho = s.material.density;
    in.useGPU = s.useGPU;
    return p;
}

void StructuralPanel::run() {
    if (!checkSetup()) return;
    cancelJob();
    playing_ = false;
    app_->picker->finish(true);
    const auto& info = studyInfos()[study_];
    Snapshot s{app_->part, fixtures, loads, gravity(), app_->material, app_->toMeters(), app_->units, resolution(), nullptr, false, true, true,
               info.fixtures, info.loads, info.fixturesOptional};
    auto it = models_.find(s.res);
    if (it != models_.end() && it->second->part == app_->part) s.cached = it->second;
    const int engine = engine_->currentIndex();
    s.useGPU = engine == 1 || (engine == 0 && gpuAvailable());
    struct Out {
        Prepared prep;
        StaticResult res;
        double seconds;
    };
    const auto t0 = std::chrono::steady_clock::now();
    app_->busy->showBusy(tr("Voxelizing part…"), [this] { cancelJob(); });
    job = runJob<Out>(this,
        [s, engine](JobControl& ctl) {
            Out o;
            {
                TraceTimer trace("prepare (model + loads)");
                o.prep = prepareOn(s, ctl);
            }
            // the CPU solver wins on small models (GPU set-up and transfers dominate)
            if (engine == 0) o.prep.input.useGPU = s.useGPU && o.prep.model->voxelCount >= AUTO_GPU_VOXELS;
            ctl.progress(0, "Solving " + std::to_string(o.prep.model->voxelCount) + " voxels…");
            o.res = solveStatic(o.prep.input, [&ctl](double f, const std::string& t) {
                ctl.progress(f, t);
                return ctl.cancelled();
            });
            return o;
        },
        [this, t0](Out o) {
            job.reset();
            app_->busy->hideBusy();
            models_[o.prep.model->resolution] = o.prep.model;
            updateMeshInfo();
            for (const auto& w : o.prep.asm_.warnings) app_->status(QString::fromStdString(w), "warn");
            TraceTimer trace("map result");
            mapResult(o.res, o.prep);
            trace.lap("show result");
            stale_ = false;
            display_ = "results";
            resultsCard_->show();
            scale_->setValue(view.scalePct = 1);
            setLevel("applied", false);
            QStringList notes;
            if (o.res.removed) notes << tr("%1 voxels not connected to a fixture were ignored").arg(o.res.removed);
            if (o.res.lostLoad > 0.01) notes << tr("%1% of the load was on unsupported bits").arg(std::lround(o.res.lostLoad * 100));
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            recordRunTime(o.res.voxels, o.res.engine, secs);
            QString msg = (result_->unreliable ? tr("Results not reliable - see the note in Results. ") : QString()) +
                          tr("Solved %1 voxels on the %2 in %3 s (%4 multigrid-CG iterations).")
                              .arg(QLocale().toString(o.res.voxels), QString::fromStdString(o.res.engine), QString::number(secs, 'f', 2))
                              .arg(o.res.iterations);
            if (!o.res.gpuNote.empty()) msg += tr(" The GPU was not used: %1.").arg(QString::fromStdString(o.res.gpuNote));
            if (!notes.isEmpty()) msg += tr(" Note: %1.").arg(notes.join("; "));
            app_->status(msg, result_->unreliable ? "error" : notes.isEmpty() ? "" : "warn");
            applyDisplay();
        },
        [this](QString msg, bool cancelled) {
            job.reset();
            app_->busy->hideBusy();
            app_->status(cancelled ? tr("Cancelled.") : tr("Solver error: %1").arg(msg), cancelled ? "" : "error");
        },
        [this](double f, QString text) { app_->busy->progress(f, text); });
}

void StructuralPanel::rememberModel(const std::shared_ptr<StructuralModel>& m) {
    learnVoxelFactor(*m);
    models_[m->resolution] = m;
    while (models_.size() > 3) models_.erase(models_.begin());
    updateMeshInfo();
}

std::shared_ptr<StructuralModel> StructuralPanel::cachedModel(int res) const {
    auto it = models_.find(res);
    return it != models_.end() && it->second->part == app_->part ? it->second : nullptr;
}

bool StructuralPanel::startStudy(const QString& busyText, bool requireFixtures, bool requireLoads, StudyWork work, StudyDone done,
                                 std::function<void(bool)> failed) {
    if (!checkSetup(requireFixtures, requireLoads)) return false;
    cancelJob();
    playing_ = false;
    app_->picker->finish(true);
    const auto& info = studyInfos()[study_];
    Snapshot s{app_->part, fixtures, loads, gravity(), app_->material, app_->toMeters(), app_->units, resolution(), nullptr, false,
               requireFixtures, requireLoads, info.fixtures, info.loads, info.fixturesOptional || !requireFixtures};
    auto it = models_.find(s.res);
    if (it != models_.end() && it->second->part == app_->part) s.cached = it->second;
    const int engine = engine_->currentIndex();
    s.useGPU = engine == 1 || (engine == 0 && gpuAvailable());
    struct Out {
        Prepared prep;
        std::any value;
    };
    app_->busy->showBusy(busyText, [this] { cancelJob(); });
    job = runJob<Out>(this,
        [s, engine, work](JobControl& ctl) {
            Out o;
            o.prep = prepareOn(s, ctl);
            // the CPU solver wins on small models (GPU set-up and transfers dominate)
            if (engine == 0) o.prep.input.useGPU = s.useGPU && o.prep.model->voxelCount >= AUTO_GPU_VOXELS;
            o.value = work(o.prep, ctl);
            return o;
        },
        [this, done](Out o) {
            job.reset();
            app_->busy->hideBusy();
            rememberModel(o.prep.model);
            for (const auto& w : o.prep.asm_.warnings) app_->status(QString::fromStdString(w), "warn");
            stale_ = false;
            done(o.value, o.prep);
            // bring the new results into view
            if (studyCard->isVisible())
                for (QWidget* w = studyCard->parentWidget(); w; w = w->parentWidget())
                    if (auto* sa = qobject_cast<QScrollArea*>(w)) {
                        // after the layout settles: the card's top at the top of the sidebar
                        QTimer::singleShot(60, studyCard, [sa, c = studyCard] {
                            if (sa->widget() && c->isVisible()) sa->verticalScrollBar()->setValue(c->mapTo(sa->widget(), QPoint(0, 0)).y() - 8);
                        });
                        break;
                    }
        },
        [this, failed](QString msg, bool cancelled) {
            job.reset();
            app_->busy->hideBusy();
            app_->status(cancelled ? tr("Cancelled.") : tr("Solver error: %1").arg(msg), cancelled ? "" : "error");
            if (failed) failed(cancelled);
        },
        [this](double f, QString text) { app_->busy->progress(f, text); });
    return true;
}

void StructuralPanel::mapResult(const StaticResult& res, const Prepared& prep) {
    const auto& m = *prep.model;
    const Part& part = *m.part;
    const auto W = m.vertexWeights(res.activeNode);
    Result r;
    r.vm = interpolate(W, res.nodeVM.data());
    r.p1 = interpolate(W, res.nodeP1.data());
    r.p3 = interpolate(W, res.nodeP3.data());
    r.u = interpolate(W, res.u.data(), 3, 1 / prep.toMeters);
    const int nV = part.nVert;
    r.dmag.assign(nV, 0.f);
    r.fos.assign(nV, 0.f);
    const Material& mat = prep.material;
    const double ys = mat.yield * 1e6, uts = mat.uts * 1e6;
    double maxRatio = 0;
    r.minP1 = r.minP3 = 1e300;
    r.maxP1 = r.maxP3 = -1e300;
    for (int v = 0; v < nV; v++) {
        r.dmag[v] = std::sqrt(r.u[3 * v] * r.u[3 * v] + r.u[3 * v + 1] * r.u[3 * v + 1] + r.u[3 * v + 2] * r.u[3 * v + 2]);
        if (std::isnan(r.vm[v])) { r.fos[v] = NAN; continue; }
        r.maxVM = std::max(r.maxVM, double(r.vm[v]));
        r.maxDisp = std::max(r.maxDisp, double(r.dmag[v]));
        r.minP1 = std::min(r.minP1, double(r.p1[v]));
        r.maxP1 = std::max(r.maxP1, double(r.p1[v]));
        r.minP3 = std::min(r.minP3, double(r.p3[v]));
        r.maxP3 = std::max(r.maxP3, double(r.p3[v]));
        const double ratio = mat.brittle ? std::max(double(r.p1[v]), 0.0) / uts : r.vm[v] / ys;
        r.fos[v] = ratio > 0 ? float(1 / ratio) : 1e3f;
        if (ratio > maxRatio) { maxRatio = ratio; r.weakest = v; }
    }
    const double diag = part.bbox.diag;
    r.autoScale = r.maxDisp > 0 ? std::min(1000.0, std::max(1.0, 0.05 * diag / r.maxDisp)) : 1;
    const double solved = double(res.voxels) / std::max(1, res.voxels + res.removed);
    r.minFos = maxRatio > 0 ? 1 / maxRatio : INFINITY;
    r.lamYield = maxRatio > 0 ? 1 / maxRatio : INFINITY;
    r.lamBreak = mat.brittle ? (maxRatio > 0 ? 1 / maxRatio : INFINITY) : r.maxVM > 0 ? uts / r.maxVM : INFINITY;
    r.totalF = std::sqrt(prep.asm_.total[0] * prep.asm_.total[0] + prep.asm_.total[1] * prep.asm_.total[1] + prep.asm_.total[2] * prep.asm_.total[2]);
    r.material = mat;
    r.units = prep.units;
    r.iterations = res.iterations;
    r.voxels = res.voxels;
    r.incomplete = !res.converged || res.lostLoad > 0.01 || res.removed > 0;
    // most of the part missing from the solved model: do not present stresses as a result
    r.unreliable = !res.converged || res.lostLoad > 0.05 || solved < 0.85;
    r.solvedShare = solved;
    r.lostLoad = res.lostLoad;
    r.converged = res.converged;
    r.thinVoxels = m.thinVoxels;
    r.voxelSize = m.grid.h;
    r.resolution = m.resolution;
    r.wallThickness = m.wallThickness;
    result_ = std::move(r);
}

// ---------- display ----------

void StructuralPanel::activate() { applyDisplay(); }

void StructuralPanel::deactivate() {
    Viewport* v = app_->viewer;
    playing_ = false;
    v->clearLayer("overlays");
    v->clearLayer("cracks");
    v->clearLayer("shape");
    v->clearLayer("voxels");
    v->clearLayer("meshpreview");
    v->clearMarker();
    v->setPartVisible(true);
    v->setDeformation(nullptr);
    v->setScalars(nullptr);
    app_->legend->clear();
}

void StructuralPanel::setVoxelPreview(bool on) { voxelsChk_->setChecked(on); }

void StructuralPanel::showVoxelPreview() {
    Viewport* v = app_->viewer;
    if (previewJob_) { previewJob_->cancel(); previewJob_.reset(); }
    v->clearLayer("meshpreview");
    const bool on = voxelsChk_->isChecked() && app_->part && app_->tab() == "structural" && display_ == "setup";
    if (!on) {
        if (display_ == "setup") v->setPartVisible(true);
        return;
    }
    auto show = [this](const std::shared_ptr<StructuralModel>& m) {
        std::vector<Geometry> g;
        g.push_back(shapes::cubes(m->voxelCenters(true), float(m->grid.h * 0.94), QVector3D(0x86 / 255.f, 0xa8 / 255.f, 0xd8 / 255.f)));
        app_->viewer->setLayer("meshpreview", std::move(g));
        app_->viewer->setPartVisible(false);
        updateMeshInfo();
    };
    if (auto m = cachedModel(resolution())) return show(m);
    // voxelizing can take a moment: do it off the UI thread
    auto part = app_->part;
    const int res = resolution();
    app_->busy->showBusy(tr("Voxelizing…"));
    previewJob_ = runJob<std::shared_ptr<StructuralModel>>(this,
        [part, res](JobControl&) { return std::make_shared<StructuralModel>(part, res); },
        [this, show](std::shared_ptr<StructuralModel> m) {
            previewJob_.reset();
            app_->busy->hideBusy();
            if (m->part != app_->part) return;
            rememberModel(m);
            if (voxelsChk_->isChecked() && app_->tab() == "structural" && display_ == "setup" && m->resolution == resolution()) show(m);
        },
        [this](QString msg, bool) {
            previewJob_.reset();
            app_->busy->hideBusy();
            voxelsChk_->setChecked(false);
            app_->status(tr("Mesh preview failed: %1").arg(msg), "error");
        },
        {});
}

void StructuralPanel::applyDisplay() {
    if (app_->tab() != "structural") return;
    Viewport* v = app_->viewer;
    renderAlert();
    Study* st = study_ > 0 ? studies[study_ - 1].get() : nullptr;
    studyCard->setVisible(st && st->hasResult());
    if (st && display_ == "study" && st->hasResult()) {
        v->clearLayer("cracks");
        st->show();
    } else if (!st && display_ == "results" && result_) showResults();
    else if (!st && display_ == "break" && brk_ && !brk_->steps.empty()) showBreakStep();
    else {
        display_ = "setup";
        v->clearLayer("shape");
        v->clearLayer("cracks");
        v->setPartVisible(true);
        v->setDeformation(nullptr);
        v->setScalars(nullptr);
        v->clearMarker();
        app_->legend->clear();
        drawOverlays();
        showVoxelPreview();
        return;
    }
    v->clearLayer("meshpreview");
}

double StructuralPanel::deformationScale(double autoScale) const {
    const int v = view.scalePct;
    if (!v) return 0;
    if (v <= 1) return 1;
    const double top = std::max(1.0, std::min(2000.0, 2 * autoScale));
    return std::pow(top, (v - 1) / 99.0);
}

double StructuralPanel::loadFactor() const {
    if (!result_) return 1;
    if (anim_) return anim_->lambda;
    return view.level == "yield" ? result_->lamYield : view.level == "break" ? result_->lamBreak : 1;
}

double StructuralPanel::currentScale() const {
    if (!result_ || result_->unreliable) return 0;
    return deformationScale(result_->autoScale);
}

void StructuralPanel::setLevel(const QString& level, bool refresh) {
    view.level = level;
    anim_.reset();
    static const QStringList levels = {"applied", "yield", "break"};
    const auto buttons = levelSeg_->findChildren<QToolButton*>();
    for (int i = 0; i < buttons.size(); i++) buttons[i]->setChecked(levels.value(i) == level);
    // showing the real bend: make sure the shape is drawn (at true scale if it was off)
    if (level != "applied" && !view.scalePct) scale_->setValue(view.scalePct = 1);
    if (refresh) showResults();
}

namespace {
struct PlotSpec {
    const std::vector<float>* values;
    std::vector<float> owned;
    bool scaled = false;
    LegendSpec legend;
    std::function<QString(double)> fmt;
};
}  // namespace

void StructuralPanel::showResults(bool full) {
    if (!result_ || app_->tab() != "structural" || study_ != 0) return;
    const Result& r = *result_;
    Viewport* v = app_->viewer;
    const double lam = loadFactor();
    const double scale = currentScale();
    scaleOut_->setText(scale == 0 ? tr("off") : scale < 1.05 ? tr("true scale") : QString("×%1").arg(num(scale)));
    renderAlert();
    renderLevelNote();
    if (!view.animate) v->setDeformation(&r.u, scale * lam);
    if (full) {
        // what to colour and how (values stored at the applied load; other load levels rescale the range)
        const bool beyond = lam != 1 || anim_.has_value();
        const double uts = r.material.uts * 1e6;
        PlotSpec s;
        s.legend.heat = view.heat;
        s.legend.bands = view.bands ? 12 : 0;
        const QString units = r.units;
        if (view.plot == "disp") {
            const double mx = (r.maxDisp > 0 ? r.maxDisp : 1) * (anim_ ? r.lamBreak : lam);
            s.values = &r.dmag;
            s.scaled = true;
            s.legend.title = tr("Displacement");
            s.legend.min = 0;
            s.legend.max = mx;
            s.legend.format = [units](double x) { return num(x) + " " + units; };
            s.fmt = [units, lam](double x) { return num(x * lam) + " " + units; };
        } else if (view.plot == "fos") {
            const double mx = std::min(10.0, std::max(2.0, std::ceil(r.minFos / lam * 3)));
            s.owned.resize(r.fos.size());
            for (size_t i = 0; i < r.fos.size(); i++) s.owned[i] = std::isnan(r.fos[i]) ? NAN : float(std::min(r.fos[i] / lam, mx));
            s.values = &s.owned;
            s.legend.title = tr("Factor of safety");
            s.legend.min = 0;
            s.legend.max = mx;
            s.legend.reverse = true;
            s.legend.format = [](double x) { return num(x, 2); };
            s.legend.markers = {{1, "FOS = 1"}};
            s.fmt = [mx](double x) { return x >= mx ? QString("> %1").arg(mx) : num(x, 3); };
        } else if (view.plot == "p1") {
            const double hi = beyond ? std::max(r.maxP1 * lam, uts) : std::max(r.maxP1, 1.0);
            const auto f = stressFormatter(std::max(std::abs(r.minP1 * lam), std::abs(hi)));
            s.values = &r.p1;
            s.scaled = true;
            s.legend.title = tr("Max principal stress");
            s.legend.min = std::min(0.0, r.minP1 * lam);
            s.legend.max = hi;
            s.legend.format = f;
            s.legend.markers = {{uts, QString("UTS %1").arg(num(r.material.uts))}};
            s.fmt = [f, lam](double x) { return f(x * lam); };
        } else if (view.plot == "p3") {
            const auto f = stressFormatter(std::max(std::abs(r.minP3), std::abs(r.maxP3)) * lam);
            s.values = &r.p3;
            s.scaled = true;
            s.legend.title = tr("Min principal stress");
            s.legend.min = std::min(r.minP3, -1.0) * lam;
            s.legend.max = std::max(0.0, r.maxP3) * lam;
            s.legend.format = f;
            s.fmt = [f, lam](double x) { return f(x * lam); };
        } else {
            // beyond the applied load, keep the scale fixed at the tensile strength so red means breaking
            const double hi = beyond ? std::max(r.maxVM * lam, uts) : (r.maxVM > 0 ? r.maxVM : 1);
            const auto f = stressFormatter(hi);
            s.values = &r.vm;
            s.scaled = true;
            s.legend.title = tr("von Mises stress");
            s.legend.min = 0;
            s.legend.max = hi;
            s.legend.format = f;
            s.legend.markers = {{r.material.yield * 1e6, QString("Yield %1").arg(num(r.material.yield))}};
            if (beyond && !r.material.brittle) s.legend.markers.push_back({uts, QString("UTS %1").arg(num(r.material.uts))});
            s.fmt = [f, lam](double x) { return f(x * lam); };
        }
        const double k = s.scaled ? lam : 1;
        ColorSpec cs;
        cs.min = float(s.legend.min / k);
        cs.max = float(s.legend.max / k);
        cs.bands = s.legend.bands;
        cs.reverse = s.legend.reverse;
        cs.heat = view.heat;
        v->setScalars(s.values, cs);
        v->clearLayer("cracks");
        v->setPartVisible(true);
        const QString shapeText = scale ? (scale < 1.05 ? tr("true-scale shape") : tr("deformation ×%1").arg(num(scale))) : tr("undeformed shape");
        QString level;
        const QString load = r.totalF > 1e-9 ? force(lam * r.totalF) : QString("%1 × loads").arg(num(lam));
        if (anim_) level = tr("load %1").arg(load);
        else if (view.level == "yield") level = tr("at first yield (%1)").arg(load);
        else if (view.level == "break") level = tr("at breaking load (%1)").arg(load);
        else level = tr("at applied load");
        s.legend.sub = QString("%1 · %2 · %3").arg(QString::fromStdString(r.material.name), level, shapeText);
        app_->legend->setSpecs({s.legend});
        shown_ = s.values == &s.owned ? s.owned : *s.values;
        shownFmt_ = s.fmt;
        drawOverlays();
        renderKpis();
    }
    placeMarker(scale * lam);
}

void StructuralPanel::renderLevelNote() {
    const auto& r = result_;
    if (!r || r->unreliable || !std::isfinite(r->lamBreak)) { levelNote_->clear(); return; }
    const QString load = r->totalF > 1e-9 ? force(r->lamBreak * r->totalF) : tr("%1 × the loads").arg(num(r->lamBreak));
    const QString bend = num(r->maxDisp * r->lamBreak) + " " + r->units;
    const QString subtle = r->maxDisp * r->lamBreak < 0.02 * app_->part->bbox.diag
                               ? tr(" At true scale that is hard to see; slide “Deformed shape” right to exaggerate it.")
                               : QString();
    levelNote_->setText((r->material.brittle ? tr("It breaks at about %1, after bending %2 (brittle: it cracks at the marker with little warning).").arg(load, bend)
                                             : tr("It reaches its tensile strength at about %1, after bending %2 elastically. Ductile metals bend further (permanently) before they tear, so treat this as the point where it is badly bent or failing.").arg(load, bend)) +
                        subtle);
}

void StructuralPanel::placeMarker(double scale) {
    Viewport* v = app_->viewer;
    const auto& r = result_;
    if (!r || r->unreliable || !view.marker || !std::isfinite(r->lamYield) || r->weakest < 0) { v->clearMarker(); return; }
    const int i = r->weakest;
    const auto& V = app_->part->vertices;
    const QVector3D p(float(V[3 * i] + scale * r->u[3 * i]), float(V[3 * i + 1] + scale * r->u[3 * i + 1]), float(V[3 * i + 2] + scale * r->u[3 * i + 2]));
    const QString level = anim_ ? "anim" : view.level;
    const QString label = level == "break" ? tr("Breaks here") : level == "yield" ? tr("Yields here") : r->minFos < 1 ? tr("Strength exceeded here") : tr("Estimated weakest point");
    if (v->hasMarker() && v->markerLabel() == label) v->moveMarker(p);
    else v->setMarker(p, label);
}

void StructuralPanel::renderAlert() {
    const auto& r = result_;
    if (!r || display_ != "results" || !(r->unreliable || r->thinVoxels)) { alert_->hide(); return; }
    const QString units = r->units;
    if (r->unreliable) {
        const QString why = !r->converged
                                ? tr("the solver could not balance the loads (usually pieces that are not connected to a fixture, or a fixture too small to stop the part pivoting)")
                                : tr("only %1% of the part was connected to the fixtures in the voxel model%2")
                                      .arg(std::lround(r->solvedShare * 100))
                                      .arg(r->lostLoad > 0.05 ? tr(" and %1% of the load fell on the unconnected bits").arg(std::lround(r->lostLoad * 100)) : QString());
        const bool thin = r->wallThickness < 2 * r->voxelSize;
        const QString cause = thin ? tr("The walls (≈ %1 %2) are thin compared with the voxels (%3 %2); a finer mesh usually fixes this.").arg(num(r->wallThickness), units, num(r->voxelSize))
                                   : tr("Check that every body touches the rest of the part or has its own fixed support, and that the load is on the supported part.");
        alert_->setText(tr("<b>These results are not reliable:</b> %1. %2 The shape is shown undeformed.").arg(why, cause));
        alert_->setProperty("kind", "error");
    } else {
        alert_->setText(tr("Some walls are thinner than a voxel (%1 %2); they are modelled as connected layers with their true cross-section. Stiffness along the walls is accurate; bending of those walls themselves is approximate. For sheet-metal bending, raise the voxel count until this note disappears.")
                            .arg(num(r->voxelSize), units));
        alert_->setProperty("kind", "info");
    }
    alert_->style()->unpolish(alert_);
    alert_->style()->polish(alert_);
    alert_->show();
}

void StructuralPanel::renderKpis() {
    const Result& r = *result_;
    const Material& mat = r.material;
    const char* st = r.unreliable ? "bad" : r.incomplete ? "warn" : r.minFos >= 2 ? "good" : r.minFos >= 1 ? "warn" : "bad";
    const QString stText = r.unreliable ? tr("Not reliable") : r.incomplete ? tr("Partial model") : r.minFos >= 2 ? tr("Below strength") : r.minFos >= 1 ? tr("Low margin") : tr("Strength exceeded");
    auto loadAt = [&](double lam) { return r.totalF > 1e-9 ? force(lam * r.totalF) : QString("%1 × loads").arg(num(lam)); };
    kpis_->setKpis({
        {tr("Max von Mises"), stress(r.maxVM), tr("yield %1 MPa").arg(num(mat.yield))},
        {tr("Max displacement"), num(r.maxDisp) + " " + r.units, tr("net applied %1").arg(force(r.totalF))},
        {tr("Min factor of safety"), std::isfinite(r.minFos) ? num(r.minFos) : "∞", "", st, stText},
        {mat.brittle ? tr("Estimated first crack") : tr("Estimated first yield"), std::isfinite(r.lamYield) ? loadAt(r.lamYield) : "–",
         mat.brittle ? tr("max principal = UTS") : tr("von Mises = yield")},
    });
}

void StructuralPanel::tick(double dt) {
    if (app_->tab() != "structural") return;
    if (study_ > 0) {
        if (display_ == "study") studies[study_ - 1]->tick(dt);
        return;
    }
    if (anim_ && result_ && display_ == "results") {
        auto& a = *anim_;
        const auto& r = *result_;
        a.t = std::min(a.dur, a.t + dt);
        const double x = a.t / a.dur;
        a.lambda = r.lamBreak * x * x * (3 - 2 * x);  // ease in and out
        if (a.t - a.lastLegend > 0.1 || a.t >= a.dur) {
            a.lastLegend = a.t;
            showResults(true);
        } else {
            app_->viewer->setDeformation(&r.u, currentScale() * a.lambda);
        }
        if (a.t >= a.dur) {
            anim_.reset();
            setLevel("break");
            app_->status((r.material.brittle ? tr("Breaks at about %1, after bending %2 %3.") : tr("Reaches its tensile strength at about %1, after bending %2 %3."))
                             .arg(r.totalF > 1e-9 ? force(r.lamBreak * r.totalF) : tr("%1 × the loads").arg(num(r.lamBreak)))
                             .arg(num(r.maxDisp * r.lamBreak), r.units));
        }
        return;
    }
    if (display_ == "results" && result_ && view.animate) {
        phase_ += dt;
        const double peak = (currentScale() ? currentScale() : (result_->unreliable ? 0 : result_->autoScale)) * loadFactor();
        const double s = peak * (0.5 - 0.5 * std::cos(phase_ * M_PI));
        app_->viewer->setDeformation(&result_->u, s);
        placeMarker(s);
    }
    if (playing_ && brk_) {
        playT_ += dt;
        if (playT_ > 0.35) {
            playT_ = 0;
            const int next = brk_->current + 1;
            if (next >= int(brk_->steps.size())) {
                playing_ = false;
                breakPlay_->setText(tr("▶ Play"));
            } else breakStep_->setValue(next);
        }
    }
}

QString StructuralPanel::probeText(const QPointF& pos) {
    auto hit = app_->viewer->pickPart(pos);
    if (!hit) return {};
    if (display_ == "study" && study_ > 0) return studies[study_ - 1]->probe(hit->tri, hit->bary);
    const std::vector<float>* values = nullptr;
    std::function<QString(double)> fmt;
    if (display_ == "results" && result_) { values = &shown_; fmt = shownFmt_; }
    else if (display_ == "break" && brk_ && view.breakStress && brk_->mapped.size() > size_t(brk_->current) && brk_->mapped[brk_->current]) {
        values = &brk_->mapped[brk_->current]->vm;
        fmt = [](double x) { return stress(x); };
    }
    if (!values || values->empty() || !fmt) return {};
    const auto& T = app_->part->tris;
    double s = 0;
    for (int k = 0; k < 3; k++) s += hit->bary[k] * (*values)[T[3 * hit->tri + k]];
    return std::isnan(s) ? tr("no data") : fmt(s);
}

// ---------- break test ----------

void StructuralPanel::runBreak() {
    if (study_ != 0) setStudy(0);
    if (!checkSetup()) return;
    cancelJob();
    app_->picker->finish(true);
    const auto& info = studyInfos()[0];
    Snapshot s{app_->part, fixtures, loads, gravity(), app_->material, app_->toMeters(), app_->units, resolution(), nullptr, false, true, true,
               info.fixtures, info.loads, info.fixturesOptional};
    auto it = models_.find(s.res);
    if (it != models_.end() && it->second->part == app_->part) s.cached = it->second;
    const int engine = engine_->currentIndex();
    s.useGPU = engine == 1 || (engine == 0 && gpuAvailable());
    playing_ = false;
    brk_ = Break{};
    brk_->material = app_->material;
    brk_->toMeters = app_->toMeters();
    breakCard_->show();
    display_ = "break";
    app_->setXRay(true);
    renderBreakKpis();
    const double strength = app_->material.uts * 1e6;
    const bool principal = app_->material.brittle;
    struct Out { BreakResult r; std::shared_ptr<StructuralModel> model; double totalF; };
    app_->busy->showBusy(tr("Break test: growing the crack…"), [this] { cancelJob(); });
    job = runJob<Out>(this,
        [this, s, engine, strength, principal](JobControl& ctl) {
            Out o;
            auto prep = prepareOn(s, ctl);
            if (engine == 0) prep.input.useGPU = s.useGPU && prep.model->voxelCount >= AUTO_GPU_VOXELS;
            o.model = prep.model;
            o.totalF = std::sqrt(prep.asm_.total[0] * prep.asm_.total[0] + prep.asm_.total[1] * prep.asm_.total[1] + prep.asm_.total[2] * prep.asm_.total[2]);
            auto model = prep.model;
            const double totalF = o.totalF;
            std::weak_ptr<JobControl> weak = ctl.weak_from_this();
            o.r = breakTest(prep.input, strength, principal, 60,
                            [this, &ctl, model, totalF, weak](BreakStep&& step) {
                                // each crack step goes to the UI thread as soon as it is computed
                                auto shared = std::make_shared<BreakStep>(std::move(step));
                                ctl.post([this, shared, model, totalF, weak]() {
                                    auto c = weak.lock();
                                    if (!c || c->cancelled() || !brk_) return;
                                    brk_->model = model;
                                    brk_->totalF = totalF;
                                    brk_->steps.push_back(std::move(*shared));
                                    brk_->mapped.emplace_back();
                                    breakStep_->blockSignals(true);
                                    breakStep_->setMaximum(int(brk_->steps.size()) - 1);
                                    brk_->current = int(brk_->steps.size()) - 1;
                                    breakStep_->setValue(brk_->current);
                                    breakStep_->blockSignals(false);
                                    showBreakStep();
                                });
                            },
                            [&ctl](double f, const std::string& t) {
                                ctl.progress(f, t);
                                return ctl.cancelled();
                            });
            return o;
        },
        [this](Out o) {
            job.reset();
            app_->busy->hideBusy();
            if (!brk_) return;
            brk_->done = true;
            brk_->reason = QString::fromStdString(o.r.reason);
            app_->status(o.r.reason == "separated" ? tr("Damage illustration finished on the %1: the voxel load path separated after %2 steps.").arg(QString::fromStdString(o.r.engine)).arg(brk_->steps.size())
                                                   : tr("Damage illustration stopped after %1 steps (%2).").arg(brk_->steps.size()).arg(QString::fromStdString(o.r.reason)),
                         o.r.reason == "separated" ? "" : "warn");
            stale_ = false;
            if (!brk_->steps.empty()) {
                brk_->current = 0;
                breakStep_->blockSignals(true);
                breakStep_->setValue(0);
                breakStep_->blockSignals(false);
                showBreakStep();
            }
            renderBreakKpis();
        },
        [this](QString msg, bool cancelled) {
            job.reset();
            app_->busy->hideBusy();
            if (brk_) {
                brk_->done = true;
                brk_->reason = cancelled ? "stopped" : "error";
                renderBreakKpis();
            }
            app_->status(cancelled ? tr("Break test stopped - showing the steps computed so far.") : tr("Solver error: %1").arg(msg), cancelled ? "warn" : "error");
        },
        [this](double f, QString text) { app_->busy->progress(f, text); });
}

void StructuralPanel::renderBreakKpis() {
    if (!brk_) return;
    const auto& b = *brk_;
    const Material& mat = b.material;
    auto loadAt = [&](double lam) { return b.totalF > 1e-9 ? force(lam * b.totalF) : QString("%1 × loads").arg(num(lam)); };
    const BreakStep* s0 = b.steps.empty() ? nullptr : &b.steps[0];
    double peak = 0;
    for (const auto& s : b.steps) peak = std::max(peak, s.lambda);
    const QString outcome = !b.done ? tr("running…") : b.reason == "separated" ? tr("load path separates") : b.reason;
    breakKpis_->setKpis({
        {tr("Estimated first crack"), s0 ? loadAt(s0->lambda) : b.done ? "–" : "…",
         mat.brittle ? tr("max principal = UTS") : tr("von Mises = UTS; yields at %1").arg(s0 ? loadAt(s0->lambda * mat.yield / mat.uts) : "…")},
        {tr("Peak in computed steps"), b.steps.empty() ? (b.done ? "–" : "…") : loadAt(peak), tr("%1 steps · %2").arg(b.steps.size()).arg(outcome)},
    });
    std::vector<QPointF> pts;
    for (size_t i = 0; i < b.steps.size(); i++) pts.push_back(QPointF(double(i + 1), b.steps[i].lambda * (b.totalF > 1e-9 ? b.totalF : 1)));
    breakChart_->setPoints(pts, b.current);
    breakStepOut_->setText(QString("%1/%2").arg(b.steps.empty() ? 0 : b.current + 1).arg(b.steps.size()));
}

void StructuralPanel::showBreakStep() {
    if (!brk_ || brk_->steps.empty() || app_->tab() != "structural" || study_ != 0) return;
    auto& b = *brk_;
    alert_->hide();
    Viewport* v = app_->viewer;
    const StructuralModel& m = *b.model;
    const int k = std::min(b.current, int(b.steps.size()) - 1);
    const BreakStep& s = b.steps[k];
    if (!b.mapped[k]) {
        const auto W = m.vertexWeights(s.activeNode);
        Break::Mapped mp;
        mp.u = interpolate(W, s.u.data(), 3, 1 / b.toMeters);
        mp.vm = interpolate(W, s.nodeVM.data());
        b.mapped[k] = std::move(mp);
    }
    if (!b.scale) {
        const double d0 = b.steps[0].maxDisp / b.toMeters;
        b.scale = d0 > 0 ? std::min(1000.0, std::max(1.0, 0.05 * app_->part->bbox.diag / d0)) : 1;
    }
    const double scale = deformationScale(b.scale);
    const double uts = b.material.uts * 1e6;
    ColorSpec cs;
    cs.min = 0;
    cs.max = float(uts);
    cs.bands = view.bands ? 12 : 0;
    cs.heat = view.heat;
    v->setScalars(view.breakStress ? &b.mapped[k]->vm : nullptr, cs);
    v->setDeformation(&b.mapped[k]->u, scale);
    // cracked and detached voxels up to this step, moved with the deformation
    auto positions = [&](const std::vector<int>& list) {
        std::vector<float> out;
        const int nx = m.grid.dims[0], ny = m.grid.dims[1];
        for (int e : list) {
            const int i = e % nx, j = (e / nx) % ny, kk = e / (nx * ny);
            double ux = 0, uy = 0, uz = 0;
            int c = 0;
            for (int a = 0; a < 8; a++) {
                const int64_t n = m.node(i + (a & 1), j + ((a >> 1) & 1), kk + ((a >> 2) & 1));
                if (!s.activeNode[n]) continue;
                ux += s.u[3 * n]; uy += s.u[3 * n + 1]; uz += s.u[3 * n + 2];
                c++;
            }
            const double f = c ? scale / (c * b.toMeters) : 0;
            out.push_back(float(m.grid.origin[0] + (i + 0.5) * m.grid.h + ux * f));
            out.push_back(float(m.grid.origin[1] + (j + 0.5) * m.grid.h + uy * f));
            out.push_back(float(m.grid.origin[2] + (kk + 0.5) * m.grid.h + uz * f));
        }
        return out;
    };
    std::vector<int> crack, gone;
    for (int j = 0; j <= k; j++) {
        crack.insert(crack.end(), b.steps[j].cracked.begin(), b.steps[j].cracked.end());
        gone.insert(gone.end(), b.steps[j].detached.begin(), b.steps[j].detached.end());
    }
    std::vector<Geometry> cr;
    cr.push_back(shapes::cubes(positions(crack), float(m.grid.h * 1.02), rgb(CRACK_COLOR)));
    cr.push_back(shapes::cubes(positions(gone), float(m.grid.h * 0.98), QVector3D(0.48f, 0.52f, 0.58f), 0.3f));
    v->setLayer("cracks", std::move(cr));
    const QString load = b.totalF > 1e-9 ? force(s.lambda * b.totalF) : QString("%1 × loads").arg(num(s.lambda));
    if (view.breakStress) {
        LegendSpec ls;
        ls.title = tr("von Mises stress");
        ls.sub = tr("Step %1 · load %2 · ×%3").arg(k + 1).arg(load, num(scale));
        ls.min = 0;
        ls.max = uts;
        ls.format = stressFormatter(uts);
        ls.bands = cs.bands;
        ls.heat = view.heat;
        ls.markers = {{b.material.yield * 1e6, QString("Yield %1").arg(num(b.material.yield))}};
        app_->legend->setSpecs({ls});
    } else app_->legend->clear();
    if (!b.steps[0].cracked.empty() && view.marker) {
        const auto p = positions({b.steps[0].cracked[0]});
        v->setMarker(QVector3D(p[0], p[1], p[2]), k == 0 ? tr("Crack starts here") : tr("Crack origin"));
    } else v->clearMarker();
    renderBreakKpis();
    drawOverlays();
}

}  // namespace ps
