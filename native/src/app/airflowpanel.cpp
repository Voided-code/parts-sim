#include "airflowpanel.hpp"

#include <QButtonGroup>
#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <chrono>
#include <cmath>

#include "gpu/gpu.hpp"
#include "mainwindow.hpp"
#include "structuralpanel.hpp"
#include "viewport.hpp"
#include "widgets.hpp"

namespace ps {

namespace {

const QVector3D WIND_COLOR(0x2a / 255.f, 0x78 / 255.f, 0xd6 / 255.f);
const char* LAYERS[] = {"flow:arrow", "flow:domain", "flow:particles", "flow:streamlines", "flow:slice"};

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

QVector3D q3(const Vec3& v) { return QVector3D(float(v[0]), float(v[1]), float(v[2])); }

struct Preset { int yaw, pitch; const char* label; const char* tip; };
const Preset PRESETS[5] = {{90, 0, "→ +X", "Air moves toward +X"},
                           {-90, 0, "← −X", "Air moves toward −X"},
                           {0, 0, "−Z", "Air moves toward −Z (hits the front)"},
                           {180, 0, "+Z", "Air moves toward +Z"},
                           {0, -90, "↓ −Y", "Air falls onto the part from above"}};

}  // namespace

AirflowPanel::AirflowPanel(MainWindow* app) : app_(app) { buildUi(); }

AirflowPanel::~AirflowPanel() { stop(); }

double AirflowPanel::rnd() {
    seed_ = seed_ * 1664525u + 1013904223u;
    return (seed_ >> 8) / double(1u << 24);
}

void AirflowPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(10);
    QVBoxLayout* c;

    root->addWidget(card(tr("Wind"), &c, this));
    c->addWidget(note(tr("Direction the air travels. The blue arrow in the view shows it."), this));
    auto* pr = new QHBoxLayout;
    pr->setSpacing(0);
    for (int i = 0; i < 5; i++) {
        auto* b = new QToolButton(this);
        b->setText(PRESETS[i].label);
        b->setToolTip(tr(PRESETS[i].tip));
        b->setCheckable(true);
        b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        b->setObjectName(i == 0 ? "segFirst" : i == 4 ? "segLast" : "segMid");
        connect(b, &QToolButton::clicked, this, [this, i] {
            yaw_ = PRESETS[i].yaw;
            pitch_ = PRESETS[i].pitch;
            syncWind();
            markDirty();
        });
        presets_[i] = b;
        pr->addWidget(b);
    }
    c->addLayout(pr);
    auto slider = [&](const QString& label, int min, int max, QSlider*& s, QLabel*& out, int* target) {
        auto* h = new QHBoxLayout;
        h->addWidget(new QLabel(label, this));
        h->addStretch();
        out = new QLabel(this);
        h->addWidget(out);
        c->addLayout(h);
        s = new QSlider(Qt::Horizontal, this);
        s->setRange(min, max);
        connect(s, &QSlider::valueChanged, this, [this, target](int v) {
            if (*target == v) return;
            *target = v;
            syncWind();
            markDirty();
        });
        c->addWidget(s);
    };
    slider(tr("Yaw"), -180, 180, yaw_s_, yawOut_, &yaw_);
    slider(tr("Pitch (angle of attack)"), -90, 90, pitch_s_, pitchOut_, &pitch_);
    auto prop = [&](const QString& label, QDoubleSpinBox* box) {
        auto* h = new QHBoxLayout;
        h->addWidget(new QLabel(label, this), 1);
        box->setParent(this);
        box->setMaximumWidth(120);
        h->addWidget(box);
        c->addLayout(h);
    };
    speed_ = numberBox(20, 0.1, 1000, 1, 1, [this](double) { markDirty(); });
    density_ = numberBox(1.225, 0.01, 2000, 0.01, 3, [this](double) { markDirty(); });
    prop(tr("Speed (m/s)"), speed_);
    prop(tr("Air density (kg/m³)"), density_);

    root->addWidget(card(tr("Solver"), &c, this));
    auto* eh = new QHBoxLayout;
    eh->addWidget(new QLabel(tr("Engine"), this));
    engine_ = new QComboBox(this);
    engine_->addItems({tr("Automatic (GPU if available)"), tr("GPU"), tr("CPU (all cores)")});
    connect(engine_, &QComboBox::activated, this, [this](int) {
        // the CPU takes smaller grids: keep the slider within what this engine can do
        cells_ = std::clamp(cells_, capacity().minCells, capacity().maxCells);
        syncCells();
        markDirty();
    });
    eh->addWidget(engine_, 1);
    c->addLayout(eh);
    auto* rh = new QHBoxLayout;
    rh->addWidget(new QLabel(tr("Grid size"), this));
    rh->addStretch();
    cellsOut_ = new QLabel(this);
    rh->addWidget(cellsOut_);
    c->addLayout(rh);
    // log scale: every step of the slider is the same ratio of cells
    cellsSlider_ = new QSlider(Qt::Horizontal, this);
    cellsSlider_->setRange(0, 1000);
    connect(cellsSlider_, &QSlider::valueChanged, this, [this](int v) {
        const auto cap = capacity();
        const double raw = cap.minCells * std::pow(cap.maxCells / cap.minCells, v / 1000.0);
        // round to two significant digits so the numbers read cleanly
        const double p = std::pow(10.0, std::floor(std::log10(raw)) - 1);
        cells_ = std::clamp(std::round(raw / p) * p, cap.minCells, cap.maxCells);
        updateCellsInfo();
        markDirty();
    });
    c->addWidget(cellsSlider_);
    cellsInfo_ = note("", this);
    c->addWidget(cellsInfo_);
    gpuInfo_ = note(gpuAvailable() ? tr("GPU: %1 - the flow runs on your graphics card.").arg(QString::fromStdString(gpuName()))
                                   : tr("No compatible GPU was found, so the flow runs on all CPU cores with a coarser grid."),
                    this);
    c->addWidget(gpuInfo_);

    auto* runRow = new QHBoxLayout;
    runBtn_ = new QPushButton(tr("Run airflow"), this);
    runBtn_->setObjectName("primary");
    runBtn_->setMinimumHeight(36);
    connect(runBtn_, &QPushButton::clicked, this, [this] { run(); });
    resetBtn_ = new QPushButton(tr("Reset"), this);
    resetBtn_->setMinimumHeight(36);
    connect(resetBtn_, &QPushButton::clicked, this, [this] {
        if (!sim_) return;
        sim_->reset();
        cp_.clear();
        initParticles();
        update();
    });
    runRow->addWidget(runBtn_, 3);
    runRow->addWidget(resetBtn_, 1);
    root->addLayout(runRow);

    flowCard_ = card(tr("Aerodynamics"), &c, this);
    kpis_ = new KpiGrid(this);
    c->addWidget(kpis_);
    state_ = note("", this);
    c->addWidget(state_);
    flowCard_->hide();
    root->addWidget(flowCard_);

    displayCard_ = card(tr("Display"), &c, this);
    auto check = [&](const QString& label, bool on) {
        auto* b = new QCheckBox(label, this);
        b->setChecked(on);
        return b;
    };
    cpChk_ = check(tr("Surface pressure"), true);
    particlesChk_ = check(tr("Particles"), true);
    streamChk_ = check(tr("Streamlines"), false);
    domainChk_ = check(tr("Tunnel box"), true);
    auto* r1 = new QHBoxLayout;
    r1->addWidget(cpChk_);
    r1->addWidget(particlesChk_);
    r1->addStretch();
    c->addLayout(r1);
    auto* r2 = new QHBoxLayout;
    r2->addWidget(streamChk_);
    r2->addWidget(domainChk_);
    r2->addStretch();
    c->addLayout(r2);
    connect(cpChk_, &QCheckBox::toggled, this, [this] { applyColoring(true); });
    connect(particlesChk_, &QCheckBox::toggled, this, [this](bool on) {
        if (!on) app_->viewer->clearLayer("flow:particles");
        renderLegends();
    });
    connect(streamChk_, &QCheckBox::toggled, this, [this] { buildStreamlines(); renderLegends(); });
    connect(domainChk_, &QCheckBox::toggled, this, [this] { buildDomain(); });
    sliceChk_ = check(tr("Slice plane"), false);
    c->addWidget(sliceChk_);
    sliceControls_ = new QWidget(this);
    auto* sl = new QVBoxLayout(sliceControls_);
    sl->setContentsMargins(0, 0, 0, 0);
    auto* sr = new QHBoxLayout;
    sliceAxis_ = new QComboBox(sliceControls_);
    sliceAxis_->addItem(tr("Horizontal"), "xz");
    sliceAxis_->addItem(tr("Vertical (along wind)"), "xy");
    sliceAxis_->addItem(tr("Cross-section"), "yz");
    sliceQty_ = new QComboBox(sliceControls_);
    sliceQty_->addItem(tr("Speed"), "speed");
    sliceQty_->addItem(tr("Pressure"), "pressure");
    sr->addWidget(sliceAxis_, 1);
    sr->addWidget(sliceQty_, 1);
    sl->addLayout(sr);
    auto* ph = new QHBoxLayout;
    slicePos_ = new QSlider(Qt::Horizontal, sliceControls_);
    slicePos_->setRange(0, 100);
    slicePos_->setValue(50);
    slicePosOut_ = new QLabel("50%", sliceControls_);
    ph->addWidget(slicePos_, 1);
    ph->addWidget(slicePosOut_);
    sl->addLayout(ph);
    sliceControls_->hide();
    c->addWidget(sliceControls_);
    connect(sliceChk_, &QCheckBox::toggled, this, [this](bool on) {
        sliceControls_->setVisible(on);
        buildSlice();
        renderLegends();
    });
    connect(sliceAxis_, &QComboBox::activated, this, [this] { buildSlice(); });
    connect(sliceQty_, &QComboBox::activated, this, [this] { buildSlice(); renderLegends(); });
    connect(slicePos_, &QSlider::valueChanged, this, [this](int v) {
        slicePosOut_->setText(QString("%1%").arg(v));
        buildSlice();
    });
    windLoad_ = new QPushButton(tr("Use as load in bend test →"), this);
    connect(windLoad_, &QPushButton::clicked, this, [this] { useAsLoad(); });
    c->addWidget(windLoad_);
    displayCard_->hide();
    root->addWidget(displayCard_);
    root->addStretch();
    compact(this);
    runBtn_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    cells_ = capacity().defaultCells;
    syncCells();
    syncWind();
    updateButtons();
}

void AirflowPanel::syncWind() {
    yawOut_->setText(QString("%1°").arg(yaw_));
    pitchOut_->setText(QString("%1°").arg(pitch_));
    yaw_s_->blockSignals(true);
    yaw_s_->setValue(yaw_);
    yaw_s_->blockSignals(false);
    pitch_s_->blockSignals(true);
    pitch_s_->setValue(pitch_);
    pitch_s_->blockSignals(false);
    for (int i = 0; i < 5; i++) presets_[i]->setChecked(PRESETS[i].yaw == yaw_ && PRESETS[i].pitch == pitch_);
    if (cellsInfo_) updateCellsInfo();
}

AirflowOptions AirflowPanel::settings() const {
    AirflowOptions o;
    o.dir = windDirection(yaw_, pitch_);
    o.speed = speed_->value();
    o.airDensity = density_->value();
    o.cells = std::clamp(cells_, capacity().minCells, capacity().maxCells);
    o.engine = engine_->currentIndex();
    o.toMeters = app_->toMeters();
    return o;
}

void AirflowPanel::setCells(double n) {
    cells_ = std::clamp(n, capacity().minCells, capacity().maxCells);
    syncCells();
    markDirty();
}

bool AirflowPanel::gpuEngine() const { return engine_->currentIndex() != 2 && gpuAvailable(); }

AirflowCapacity AirflowPanel::capacity() const { return airflowCapacity(gpuEngine()); }

void AirflowPanel::syncCells() {
    const auto cap = capacity();
    cellsSlider_->blockSignals(true);
    cellsSlider_->setValue(int(std::lround(1000 * std::log(cells_ / cap.minCells) / std::log(cap.maxCells / cap.minCells))));
    cellsSlider_->blockSignals(false);
    updateCellsInfo();
}

void AirflowPanel::updateCellsInfo() {
    const auto cap = capacity();
    auto count = [](double n) { return n >= 1e6 ? tr("%1 M").arg(num(n / 1e6, 2)) : tr("%1 k").arg(num(n / 1e3, 2)); };
    cellsOut_->setText(tr("%1 cells").arg(count(cells_)));
    QStringList parts;
    if (app_->part) {
        try {
            const auto plan = planTunnel(*app_->part, windDirection(yaw_, pitch_), cells_);
            parts << tr("Tunnel %1 × %2 × %3, cells %4 %5 across.")
                         .arg(plan.dims[0]).arg(plan.dims[1]).arg(plan.dims[2])
                         .arg(num(plan.h), app_->units);
        } catch (const std::exception&) {
        }
    }
    const double gb = 1e9;
    if (cap.gpuBytesPerCell > 0)
#ifdef __APPLE__
        parts << tr("About %1 GB of memory (shared with the GPU).").arg(num(cells_ * cap.ramBytesPerCell / gb, 2));
#else
        parts << tr("About %1 GB of GPU memory.").arg(num(cells_ * cap.gpuBytesPerCell / gb, 2));
#endif
    else parts << tr("About %1 GB of memory.").arg(num(cells_ * cap.ramBytesPerCell / gb, 2));
    // from the speed measured on the last run: how long until the flow has developed
    const QString engine = gpuEngine() ? "GPU" : "CPU";
    if (lastMlups_ > 0 && lastEngine_ == engine) {
        const double stepsPerSecond = lastMlups_ * 1e6 / cells_;
        double nx = std::cbrt(cells_);
        if (app_->part) {
            try { nx = planTunnel(*app_->part, windDirection(yaw_, pitch_), cells_).dims[0]; } catch (const std::exception&) {}
        }
        const double developSteps = LBM_RAMP_STEPS + 0.6 * nx / AIR_U_LAT;
        parts << tr("At the last run's %1 MLUPS: about %2 steps/s, developed flow after about %3 s.")
                     .arg(num(lastMlups_), num(stepsPerSecond), num(developSteps / stepsPerSecond, 2));
    }
    parts << tr("Range here: %1 to %2 cells.").arg(count(cap.minCells), count(cap.maxCells));
    cellsInfo_->setText(parts.join(" "));
}

void AirflowPanel::markDirty() {
    dirty_ = true;
    drawWindArrow();
    updateButtons();
    update();
}

void AirflowPanel::stop() {
    if (job_) {
        job_->cancel();
        job_.reset();
        app_->busy->hideBusy();
    }
    if (sim_) {
        sim_->onSnapshot = {};
        sim_->onStatus = {};
        sim_->pause();
        sim_.reset();
    }
}

void AirflowPanel::reset(const std::optional<SampleSetup::Air>& preset) {
    stop();
    dirty_ = true;
    cp_.clear();
    pos_.clear();
    if (preset) {
        yaw_ = int(std::lround(preset->yaw));
        pitch_ = int(std::lround(preset->pitch));
        speed_->blockSignals(true);
        speed_->setValue(preset->speed);
        speed_->blockSignals(false);
        syncWind();
    }
    flowCard_->hide();
    displayCard_->hide();
    updateCellsInfo();
    updateButtons();
    if (app_->tab() == "airflow") {
        clearVisuals();
        activate();
    }
}

bool AirflowPanel::hasResults() const {
    auto s = sim_ ? sim_->snapshot() : nullptr;
    return s && !s->developing;
}

void AirflowPanel::updateButtons() {
    const bool building = bool(job_);
    runBtn_->setEnabled(!building);
    const bool ready = sim_ && sim_->ready();
    if (sim_ && sim_->running()) runBtn_->setText(dirty_ ? tr("Apply & restart") : tr("Pause"));
    else if (ready && !dirty_) runBtn_->setText(tr("Resume"));
    else runBtn_->setText(ready ? tr("Apply & run") : tr("Run airflow"));
    resetBtn_->setEnabled(ready && !building);
    auto s = sim_ ? sim_->snapshot() : nullptr;
    windLoad_->setEnabled(!dirty_ && s && !s->developing);
}

void AirflowPanel::run() {
    if (job_) return;
    if (!app_->part) return app_->status(tr("Import a part or open a sample first."), "error");
    if (sim_ && sim_->running() && !dirty_) {
        sim_->pause();
        updateButtons();
        update();
        return;
    }
    if (dirty_ || !sim_ || !sim_->ready()) {
        const AirflowOptions o = settings();
        stop();
        clearVisuals();
        auto building = std::make_shared<AirflowSim>();
        std::shared_ptr<const Part> part = app_->part;
        app_->busy->showBusy(tr("Building wind tunnel…"), [this] {
            stop();
            updateButtons();
            app_->status(tr("Airflow setup cancelled."));
        });
        job_ = runJob<bool>(this,
            [building, part, o](JobControl& ctl) {
                building->setup(part, o, [&ctl](const std::string& msg) {
                    ctl.progress(-1, msg);
                    return ctl.cancelled();
                });
                return true;
            },
            [this, building, part](bool) {
                job_.reset();
                app_->busy->hideBusy();
                if (app_->part != part) return;
                sim_ = building;
                // solver-thread callbacks: coalesce snapshots into one UI update
                auto pending = pending_;
                QPointer<AirflowPanel> self(this);
                sim_->onSnapshot = [self, pending] {
                    if (pending->exchange(true)) return;
                    QMetaObject::invokeMethod(self, [self, pending] {
                        pending->store(false);
                        if (self) self->update();
                    }, Qt::QueuedConnection);
                };
                sim_->onStatus = [self](const std::string& msg, bool failed) {
                    const QString m = QString::fromStdString(msg);
                    QMetaObject::invokeMethod(self, [self, m, failed] {
                        if (!self) return;
                        self->app_->status(m, failed ? "error" : "warn");
                        self->updateButtons();
                    }, Qt::QueuedConnection);
                };
                dirty_ = false;
                cp_.clear();
                initParticles();
                buildDomain();
                drawWindArrow();
                sim_->start();
                flowCard_->show();
                displayCard_->show();
                app_->status(tr("Wind tunnel ready: %1 × %2 × %3 cells on the %4.")
                                 .arg(sim_->dims[0])
                                 .arg(sim_->dims[1])
                                 .arg(sim_->dims[2])
                                 .arg(QString::fromStdString(sim_->engine)));
                updateButtons();
                update();
                renderLegends();
            },
            [this](QString msg, bool cancelled) {
                job_.reset();
                app_->busy->hideBusy();
                updateButtons();
                if (!cancelled) app_->status(tr("Airflow setup failed: %1").arg(msg), "error");
            },
            [this](double, QString text) { app_->busy->progress(-1, text); });
        updateButtons();
        return;
    }
    sim_->start();
    flowCard_->show();
    displayCard_->show();
    updateButtons();
    renderLegends();
}

void AirflowPanel::update() {
    updateButtons();
    auto s = sim_ ? sim_->snapshot() : nullptr;
    if (s) {
        const auto& r = s->results;
        const double tm = app_->toMeters();
        const double area = r.frontalArea / (tm * tm);
        // the part's own weight (from its material) against the aerodynamic force
        const Material& mat = app_->material;
        const double weight = (app_->part ? app_->part->volume : 0) * tm * tm * tm * mat.density * 9.81;
        const double ratio = weight > 0 ? r.force[1] / weight : 0;
        const QString matName = QString::fromStdString(mat.name).section(" (", 0, 0);
        KpiGrid::Kpi w{tr("Weight (%1)").arg(matName), force(weight), ""};
        if (ratio >= 1) { w.status = "bad"; w.statusText = tr("the wind lifts it"); }
        else if (ratio >= 0.5) { w.status = "warn"; w.statusText = tr("lift is %1% of its weight").arg(std::lround(ratio * 100)); }
        else w.sub = tr("lift is %1% of its weight").arg(std::max(0L, std::lround(ratio * 100)));
        kpis_->setKpis({
            {tr("Drag force"), force(r.drag), QString("Cd %1").arg(num(r.cd))},
            {tr("Lift force"), force(r.lift), QString("Cl %1").arg(num(r.cl))},
            w,
            {tr("Frontal area"), num(area) + " " + app_->units + "²", tr("side force %1").arg(force(r.side))},
            {tr("Reynolds number"), num(sim_->reynolds), tr("simulated ≈ %1").arg(num(sim_->simReynolds))},
        });
    } else kpis_->setKpis({});
    if (s && s->mlups > 0 && !s->developing) {
        lastMlups_ = s->mlups;
        lastEngine_ = QString::fromStdString(sim_->engine);
        if (now() - lastInfo_ > 2) {
            lastInfo_ = now();
            updateCellsInfo();
        }
    }
    if (sim_ && sim_->ready()) {
        const QString phase = !s ? tr("starting") : s->developing ? tr("developing flow") : tr("averaging (%1 samples)").arg(s->samples);
        state_->setText(tr("%1%2 · %3×%4×%5 cells · step %6 · %7 MLUPS · %8. Forces are pressure-only (skin friction is not resolved).")
                            .arg(dirty_ ? tr("Settings changed; apply to update results. ") : QString(), QString::fromStdString(sim_->engine))
                            .arg(sim_->dims[0])
                            .arg(sim_->dims[1])
                            .arg(sim_->dims[2])
                            .arg(QLocale().toString(qlonglong(s ? s->steps : 0)), num(s ? s->mlups : 0), phase));
    } else state_->clear();
    if (app_->tab() != "airflow") return;
    if (!s) {
        if (!cp_.empty()) { cp_.clear(); app_->viewer->setScalars(nullptr); }
        app_->viewer->clearLayer("flow:streamlines");
        app_->viewer->clearLayer("flow:slice");
        return;
    }
    if (s->steps == shownSteps_) return;
    shownSteps_ = s->steps;
    const double t = now();
    if (t - lastCp_ > 0.9) applyColoring();
    if (sliceChk_->isChecked()) buildSlice();
    if (streamChk_->isChecked() && t - lastStream_ > 2.5) buildStreamlines();
}

void AirflowPanel::applyColoring(bool force) {
    if (app_->tab() != "airflow") return;
    Viewport* v = app_->viewer;
    auto s = sim_ ? sim_->snapshot() : nullptr;
    if (cpChk_->isChecked() && s) {
        if (force || now() - lastCp_ > 0.9 || cp_.empty()) {
            cp_ = sim_->surfaceCp(*s);
            lastCp_ = now();
        }
        double mn = INFINITY, mx = -INFINITY;
        for (float x : cp_)
            if (!std::isnan(x)) { mn = std::min(mn, double(x)); mx = std::max(mx, double(x)); }
        cpMin_ = std::max(-3.0, std::min(mn, -0.2));
        cpMax_ = std::min(1.2, std::max(mx, 0.2));
        ColorSpec cs;
        cs.min = float(cpMin_);
        cs.max = float(cpMax_);
        v->setScalars(&cp_, cs);
    } else {
        cp_.clear();
        v->setScalars(nullptr);
    }
    renderLegends();
}

void AirflowPanel::renderLegends() {
    if (app_->tab() != "airflow") return;
    std::vector<LegendSpec> specs;
    const double U = sim_ ? sim_->opts.speed : speed_->value();
    const double rhoAir = sim_ ? sim_->opts.airDensity : density_->value();
    const double q = 0.5 * rhoAir * U * U;
    if (cpChk_->isChecked() && !cp_.empty()) {
        LegendSpec l;
        l.title = tr("Surface pressure (Cp)");
        l.sub = tr("1 Cp = %1 at %2 m/s").arg(stress(q), num(U));
        l.min = cpMin_;
        l.max = cpMax_;
        l.format = [](double x) { return num(x, 2); };
        specs.push_back(l);
    }
    auto s = sim_ ? sim_->snapshot() : nullptr;
    const bool ready = sim_ && sim_->ready();
    const bool speedShown = (s && (particlesChk_->isChecked() || streamChk_->isChecked())) || (sliceChk_->isChecked() && sliceQty_->currentData() == "speed");
    if (speedShown && ready) {
        LegendSpec l;
        l.title = tr("Air speed");
        l.sub = tr("particles, streamlines, slice");
        l.min = 0;
        l.max = 1.6 * U;
        l.format = [](double x) { return num(x) + " m/s"; };
        specs.push_back(l);
    }
    if (sliceChk_->isChecked() && sliceQty_->currentData() == "pressure" && ready) {
        LegendSpec l;
        l.title = tr("Slice pressure (Cp)");
        l.min = -1.2;
        l.max = 1;
        l.format = [](double x) { return num(x, 2); };
        specs.push_back(l);
    }
    app_->legend->setSpecs(specs);
}

void AirflowPanel::drawWindArrow() {
    Viewport* v = app_->viewer;
    v->clearLayer("flow:arrow");
    if (!app_->part || app_->tab() != "airflow") return;
    const auto& b = app_->part->bbox;
    const QVector3D c(float((b.min[0] + b.max[0]) / 2), float((b.min[1] + b.max[1]) / 2), float((b.min[2] + b.max[2]) / 2));
    const QVector3D dir = q3(windDirection(yaw_, pitch_));
    const float diag = float(b.diag);
    std::vector<Geometry> g;
    g.push_back(shapes::arrow(c - dir * 0.6f * diag, dir, 0.35f * diag, WIND_COLOR, 0.35f * diag * 0.04f));
    v->setLayer("flow:arrow", std::move(g));
}

void AirflowPanel::clearVisuals() {
    for (const char* l : LAYERS) app_->viewer->clearLayer(l);
}

void AirflowPanel::buildDomain() {
    app_->viewer->clearLayer("flow:domain");
    if (!sim_ || !domainChk_->isChecked() || app_->tab() != "airflow") return;
    const double X[2] = {0, double(sim_->dims[0])}, Y[2] = {0, double(sim_->dims[1])}, Z[2] = {0, double(sim_->dims[2])};
    Geometry g;
    g.kind = Geometry::Lines;
    g.lit = false;
    g.color = QVector3D(0x5b / 255.f, 0x8d / 255.f, 0xef / 255.f);
    g.opacity = 0.35f;
    auto corner = [&](int m) { return q3(sim_->toWorld(X[m & 1], Y[(m >> 1) & 1], Z[(m >> 2) & 1])); };
    for (int a = 0; a < 8; a++)
        for (int bit : {1, 2, 4})
            if (!(a & bit)) {
                g.vertex(corner(a));
                g.vertex(corner(a | bit));
            }
    std::vector<Geometry> list;
    list.push_back(std::move(g));
    app_->viewer->setLayer("flow:domain", std::move(list));
}

// ---------- particles ----------

void AirflowPanel::spawn(int i) {
    const auto& pr = sim_->partRange;
    const int ny = sim_->dims[1], nz = sim_->dims[2];
    const double padY = (pr[0][1] - pr[0][0]) * 0.35 + 2, padZ = (pr[1][1] - pr[1][0]) * 0.35 + 2;
    const double y = std::min(ny - 1.5, std::max(1.5, pr[0][0] - padY + rnd() * (pr[0][1] - pr[0][0] + 2 * padY)));
    const double z = std::min(nz - 1.5, std::max(1.5, pr[1][0] - padZ + rnd() * (pr[1][1] - pr[1][0] + 2 * padZ)));
    const double x = 1 + rnd() * 2;
    for (int s = 0; s < TRAIL; s++) {
        float* h = &hist_[3 * (size_t(i) * TRAIL + s)];
        h[0] = float(x); h[1] = float(y); h[2] = float(z);
    }
    age_[i] = 0;
    pos_[3 * i] = float(x); pos_[3 * i + 1] = float(y); pos_[3 * i + 2] = float(z);
}

void AirflowPanel::initParticles() {
    if (!sim_) return;
    pos_.assign(3 * size_t(particles_), 0.f);
    hist_.assign(3 * size_t(particles_) * TRAIL, 0.f);
    age_.assign(particles_, 0.f);
    head_ = 0;
    for (int i = 0; i < particles_; i++) {
        spawn(i);
        pos_[3 * i] = float(1 + rnd() * (sim_->dims[0] - 3));  // pre-fill the tunnel
        for (int s = 0; s < TRAIL; s++) std::copy(&pos_[3 * i], &pos_[3 * i] + 3, &hist_[3 * (size_t(i) * TRAIL + s)]);
    }
}

void AirflowPanel::stepParticles(double dt) {
    auto snap = sim_ ? sim_->snapshot() : nullptr;
    if (!snap || !particlesChk_->isChecked() || pos_.empty()) return;
    const auto& macro = snap->macro;
    const auto& sim = *sim_;
    const int nx = sim.dims[0], ny = sim.dims[1], nz = sim.dims[2];
    const double k = nx / (4.5 * AIR_U_LAT) * std::min(dt, 0.05);  // free stream crosses the tunnel in ~4.5 s
    head_ = (head_ + 1) % TRAIL;
    Geometry g;
    g.kind = Geometry::Lines;
    g.vertexColors = true;
    g.lit = false;
    g.opacity = 0.95f;
    g.verts.reserve(size_t(particles_) * (TRAIL - 1) * 2 * 7);
    double m[4];
    for (int i = 0; i < particles_; i++) {
        double x = pos_[3 * i], y = pos_[3 * i + 1], z = pos_[3 * i + 2];
        double u[3];
        if (sim.sample(macro, 4, x, y, z, m) < 0.05) { u[0] = AIR_U_LAT * 0.3; u[1] = u[2] = 0; }
        else { u[0] = m[1]; u[1] = m[2]; u[2] = m[3]; }
        // midpoint step
        if (sim.sample(macro, 4, x + 0.5 * k * u[0], y + 0.5 * k * u[1], z + 0.5 * k * u[2], m) > 0.05) { u[0] = m[1]; u[1] = m[2]; u[2] = m[3]; }
        x += k * u[0]; y += k * u[1]; z += k * u[2];
        age_[i] += float(dt);
        if (x >= nx - 1.5 || sim.isSolidAt(x, y, z) || age_[i] > 20 || y < 1 || z < 1 || y > ny - 1 || z > nz - 1) {
            spawn(i);
            x = pos_[3 * i]; y = pos_[3 * i + 1]; z = pos_[3 * i + 2];
        }
        pos_[3 * i] = float(x); pos_[3 * i + 1] = float(y); pos_[3 * i + 2] = float(z);
        float* hp = &hist_[3 * (size_t(i) * TRAIL + head_)];
        hp[0] = float(x); hp[1] = float(y); hp[2] = float(z);
        const double speed = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
        const QVector3D c = colorRamp(false, float(speed / (1.6 * AIR_U_LAT)));
        for (int s = 0; s < TRAIL - 1; s++) {
            const float* a = &hist_[3 * (size_t(i) * TRAIL + (head_ - s + TRAIL) % TRAIL)];
            const float* b = &hist_[3 * (size_t(i) * TRAIL + (head_ - s - 1 + TRAIL) % TRAIL)];
            const float al = 1 - float(s) / (TRAIL - 1), al2 = 1 - float(s + 1) / (TRAIL - 1);
            g.vertex(q3(sim.toWorld(a[0], a[1], a[2])), c, al);
            g.vertex(q3(sim.toWorld(b[0], b[1], b[2])), c, al2);
        }
    }
    std::vector<Geometry> list;
    list.push_back(std::move(g));
    app_->viewer->setLayer("flow:particles", std::move(list));
}

void AirflowPanel::buildStreamlines() {
    lastStream_ = now();
    app_->viewer->clearLayer("flow:streamlines");
    auto snap = sim_ ? sim_->snapshot() : nullptr;
    if (!snap || !streamChk_->isChecked() || app_->tab() != "airflow") return;
    const auto& sim = *sim_;
    const auto& avgU = snap->avgU;
    const int nx = sim.dims[0], ny = sim.dims[1], nz = sim.dims[2];
    const auto& pr = sim.partRange;
    const int S = 16;
    // about 250 steps along the tunnel however fine the grid is
    const double stride = std::max(1.0, nx / 250.0);
    const double padY = (pr[0][1] - pr[0][0]) * 0.3 + 1, padZ = (pr[1][1] - pr[1][0]) * 0.3 + 1;
    Geometry g;
    g.kind = Geometry::Lines;
    g.vertexColors = true;
    g.lit = false;
    double u[3];
    for (int a = 0; a < S; a++)
        for (int b = 0; b < S; b++) {
            double x = 1.5;
            double y = std::min(ny - 2.0, std::max(1.5, pr[0][0] - padY + (a + 0.5) / S * (pr[0][1] - pr[0][0] + 2 * padY)));
            double z = std::min(nz - 2.0, std::max(1.5, pr[1][0] - padZ + (b + 0.5) / S * (pr[1][1] - pr[1][0] + 2 * padZ)));
            for (int s = 0; s < 6 * nx / stride; s++) {
                if (sim.sample(avgU, 3, x, y, z, u) < 0.05) break;
                double sp = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                if (sp < 1e-5) break;
                const double st = 0.6 * stride / sp;
                if (sim.sample(avgU, 3, x + 0.5 * st * u[0], y + 0.5 * st * u[1], z + 0.5 * st * u[2], u) < 0.05) break;
                sp = std::sqrt(u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
                if (!(sp > 0)) sp = 1e-5;
                const double st2 = 0.6 * stride / sp;
                const double X = x + st2 * u[0], Y = y + st2 * u[1], Z = z + st2 * u[2];
                if (X >= nx - 1 || sim.isSolidAt(X, Y, Z) || Y < 1 || Z < 1 || Y > ny - 1 || Z > nz - 1) break;
                const QVector3D c = colorRamp(false, float(sp / (1.6 * AIR_U_LAT)));
                g.vertex(q3(sim.toWorld(x, y, z)), c, 1);
                g.vertex(q3(sim.toWorld(X, Y, Z)), c, 1);
                x = X; y = Y; z = Z;
            }
        }
    std::vector<Geometry> list;
    list.push_back(std::move(g));
    app_->viewer->setLayer("flow:streamlines", std::move(list));
}

void AirflowPanel::buildSlice() {
    app_->viewer->clearLayer("flow:slice");
    auto snap = sim_ ? sim_->snapshot() : nullptr;
    if (!snap || !sliceChk_->isChecked() || app_->tab() != "airflow") return;
    const auto& sim = *sim_;
    const int nx = sim.dims[0], ny = sim.dims[1], nz = sim.dims[2];
    const QString axis = sliceAxis_->currentData().toString();
    const bool pressure = sliceQty_->currentData() == "pressure";
    // in-plane axes (a, b) and the fixed coordinate f
    int A, B, n;
    if (axis == "xy") { A = nx; B = ny; n = nz; }
    else if (axis == "xz") { A = nx; B = nz; n = ny; }
    else { A = ny; B = nz; n = nx; }
    const int f = std::clamp(int(std::floor(slicePos_->value() / 100.0 * n)), 0, n - 1);
    auto cell = [&](int a, int b) -> int64_t {
        if (axis == "xy") return a + int64_t(nx) * (b + int64_t(ny) * f);
        if (axis == "xz") return a + int64_t(nx) * (f + int64_t(ny) * b);
        return f + int64_t(nx) * (a + int64_t(ny) * b);
    };
    auto world = [&](double a, double b) {
        const double fc = f + 0.5;
        if (axis == "xy") return q3(sim.toWorld(a, b, fc));
        if (axis == "xz") return q3(sim.toWorld(a, fc, b));
        return q3(sim.toWorld(fc, a, b));
    };
    const double inv = 1 / (3 * 0.5 * AIR_U_LAT * AIR_U_LAT);
    std::vector<QVector3D> col(size_t(A) * B);
    for (int b = 0; b < B; b++)
        for (int a = 0; a < A; a++) {
            const int64_t c = cell(a, b);
            if (sim.solid[c]) { col[a + size_t(A) * b] = QVector3D(45 / 255.f, 50 / 255.f, 60 / 255.f); continue; }
            double t;
            if (pressure) t = ((snap->avgRho[c] - 1) * inv + 1.2) / 2.2;
            else t = std::sqrt(double(snap->avgU[3 * c]) * snap->avgU[3 * c] + double(snap->avgU[3 * c + 1]) * snap->avgU[3 * c + 1] +
                               double(snap->avgU[3 * c + 2]) * snap->avgU[3 * c + 2]) / (1.6 * AIR_U_LAT);
            col[a + size_t(A) * b] = colorRamp(false, float(t));
        }
    // a quad per cell with its own colour (cell-sized pixels, like a nearest-neighbour texture);
    // fine grids are drawn in square blocks of cells so the slice stays under ~250k quads
    const int k = std::max(1, int(std::ceil(std::sqrt(double(A) * B / 250e3))));
    Geometry g;
    g.kind = Geometry::Triangles;
    g.vertexColors = true;
    g.lit = false;
    g.opacity = 0.88f;
    g.verts.reserve(size_t((A + k - 1) / k) * ((B + k - 1) / k) * 6 * 7);
    for (int b = 0; b < B; b += k)
        for (int a = 0; a < A; a += k) {
            const QVector3D c = col[std::min(A - 1, a + k / 2) + size_t(A) * std::min(B - 1, b + k / 2)];
            const int a1 = std::min(A, a + k), b1 = std::min(B, b + k);
            const QVector3D p00 = world(a, b), p10 = world(a1, b), p11 = world(a1, b1), p01 = world(a, b1);
            for (const QVector3D& p : {p00, p10, p11, p00, p11, p01}) g.vertex(p, c, 1);
        }
    std::vector<Geometry> list;
    list.push_back(std::move(g));
    app_->viewer->setLayer("flow:slice", std::move(list));
}

// ---------- tab ----------

void AirflowPanel::activate() {
    drawWindArrow();
    buildDomain();
    shownSteps_ = -1;
    applyColoring(true);
    buildSlice();
    buildStreamlines();
    renderLegends();
}

void AirflowPanel::deactivate() {
    clearVisuals();
    app_->viewer->setScalars(nullptr);
    app_->legend->clear();
}

void AirflowPanel::tick(double dt) {
    if (app_->tab() != "airflow") return;
    stepParticles(dt);
}

QString AirflowPanel::probeText(const QPointF& pos) {
    if (cp_.empty() || !cpChk_->isChecked() || !sim_) return {};
    auto hit = app_->viewer->pickPart(pos);
    if (!hit) return {};
    const auto& T = app_->part->tris;
    double s = 0;
    for (int k = 0; k < 3; k++) s += hit->bary[k] * cp_[T[3 * hit->tri + k]];
    if (std::isnan(s)) return tr("no data");
    const double q = 0.5 * sim_->opts.airDensity * sim_->opts.speed * sim_->opts.speed;
    return QString("Cp %1 · %2").arg(num(s, 2), stress(s * q));
}

void AirflowPanel::useAsLoad() {
    if (dirty_) return app_->status(tr("Apply the changed airflow settings before transferring the wind load."), "warn");
    auto s = sim_ ? sim_->snapshot() : nullptr;
    if (!s) return app_->status(tr("Run the airflow first."), "error");
    if (s->developing) return app_->status(tr("The flow is still developing - wait until it says “averaging”, then try again."), "warn");
    const auto F = sim_->triangleForces(*s);
    Vec3 net{0, 0, 0};
    for (size_t t = 0; t < F.size(); t += 3)
        for (int d = 0; d < 3; d++) net[d] += F[t + d];
    app_->structural->addWindLoad(F, net);
    app_->setTab("structural");
    app_->status(tr("Wind load added (%1 net). Add a fixture if needed, then run the bend test.").arg(force(std::sqrt(net[0] * net[0] + net[1] * net[1] + net[2] * net[2]))));
}

}  // namespace ps
