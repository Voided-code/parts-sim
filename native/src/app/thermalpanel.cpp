#include "thermalpanel.hpp"
#include "psimjson.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QPushButton>
#include <QScrollArea>
#include <QScrollBar>
#include <QSlider>
#include <QStyle>
#include <QTimer>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <set>

#include "fea/thermal.hpp"
#include "mainwindow.hpp"
#include "picker.hpp"
#include "structuralpanel.hpp"
#include "viewport.hpp"
#include "widgets.hpp"

namespace ps {

namespace {

const QColor COLORS[3] = {QColor(0xe0, 0x58, 0x2b), QColor(0xc0, 0x26, 0xd3), QColor(0x1f, 0x9b, 0xd6)};

QString label(ThermalPanel::Item::Type t) {
    switch (t) {
        case ThermalPanel::Item::Temp: return QObject::tr("Temperature");
        case ThermalPanel::Item::Heat: return QObject::tr("Heat power");
        default: return QObject::tr("Convection");
    }
}

QString metaText(const ThermalPanel::Item& it) {
    const QString unit = it.type == ThermalPanel::Item::Temp ? "°C" : it.type == ThermalPanel::Item::Heat ? "W" : "W/m²K";
    QString meta = num(it.value) + " " + unit;
    if (it.type == ThermalPanel::Item::Conv) meta += QString(" → %1 °C").arg(num(it.ambient));
    return meta;
}

QVector3D rgb(const QColor& c) { return QVector3D(float(c.redF()), float(c.greenF()), float(c.blueF())); }

QWidget* labelled(const QString& text, QWidget* control, QWidget* parent) {
    auto* w = new QWidget(parent);
    auto* h = new QHBoxLayout(w);
    h->setContentsMargins(0, 0, 0, 0);
    auto* l = new QLabel(text, w);
    h->addWidget(l, 1);
    control->setParent(w);
    control->setMaximumWidth(120);
    h->addWidget(control);
    return w;
}

// what a job needs, copied on the UI thread
struct Setup {
    std::shared_ptr<Part> part;
    std::shared_ptr<StructuralModel> cached;
    int res;
    std::vector<ThermalPanel::Item> items;
    bool ambient;
    double ambientH, ambientT;
    double toMeters;
    Material material;
    bool transient;
    double duration, initial;
    int steps;
};

struct Assembled {
    std::vector<uint8_t> fixedNode;
    std::vector<double> fixedValue, source, convH, convT;
    double heatIn = 0;
    std::vector<std::string> warnings;
};

Assembled assemble(const StructuralModel& model, const Setup& s) {
    const int64_t nN = model.nNodes;
    const double A = s.toMeters * s.toMeters;
    Assembled a;
    a.fixedNode.assign(nN, 0);
    a.fixedValue.assign(nN, 0.0);
    a.source.assign(nN, 0.0);
    a.convH.assign(nN, 0.0);
    std::vector<double> convHT(nN, 0.0);
    std::set<int32_t> used;
    for (const auto& it : s.items) {
        for (const auto& p : it.patches)
            if (!p.clip) used.insert(p.tris.begin(), p.tris.end());
        const Samples smp = model.samplePatches(it.patches);
        const std::string name = it.name.toStdString();
        if (it.type == ThermalPanel::Item::Temp) {
            int hit = 0;
            auto hold = [&](int64_t n) { a.fixedNode[n] = 1; a.fixedValue[n] = it.value; };
            for (size_t i = 0; i < smp.weights.size(); i++) {
                const double x = smp.points[3 * i], y = smp.points[3 * i + 1], z = smp.points[3 * i + 2];
                const int64_t n = model.nearestSurfaceNode(x, y, z);
                if (n < 0) continue;
                hit++;
                hold(n);
                model.forSurfaceNodesNear(x, y, z, 0.75, hold);
            }
            if (!hit) a.warnings.push_back(name + ": no voxels under the selection.");
            continue;
        }
        double W = 0;
        std::vector<std::pair<int64_t, double>> nodes;
        for (size_t i = 0; i < smp.weights.size(); i++) {
            const int64_t n = model.nearestSurfaceNode(smp.points[3 * i], smp.points[3 * i + 1], smp.points[3 * i + 2]);
            if (n < 0) continue;
            nodes.push_back({n, smp.weights[i]});
            W += smp.weights[i];
        }
        if (nodes.empty()) { a.warnings.push_back(name + ": no voxels under the selection."); continue; }
        for (const auto& [n, w] : nodes) {
            if (it.type == ThermalPanel::Item::Heat) a.source[n] += it.value * w / W;
            else {
                a.convH[n] += it.value * w * A;
                convHT[n] += it.value * w * A * it.ambient;
            }
        }
        if (it.type == ThermalPanel::Item::Heat) a.heatIn += it.value;
    }
    if (s.ambient && s.ambientH > 0) {
        Patch others;
        for (int t = 0; t < s.part->nTri; t++)
            if (!used.count(t)) others.tris.push_back(t);
        const Samples smp = model.samplePatches({others});
        for (size_t i = 0; i < smp.weights.size(); i++) {
            const int64_t n = model.nearestSurfaceNode(smp.points[3 * i], smp.points[3 * i + 1], smp.points[3 * i + 2]);
            if (n < 0) continue;
            a.convH[n] += s.ambientH * smp.weights[i] * A;
            convHT[n] += s.ambientH * smp.weights[i] * A * s.ambientT;
        }
    }
    a.convT.assign(nN, 0.0);
    for (int64_t n = 0; n < nN; n++) a.convT[n] = a.convH[n] > 0 ? convHT[n] / a.convH[n] : 0;
    return a;
}

}  // namespace

ThermalPanel::ThermalPanel(MainWindow* app) : app_(app) { buildUi(); }

ThermalPanel::~ThermalPanel() {
    if (job) job->cancel();
}

void ThermalPanel::buildUi() {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(10);
    QVBoxLayout* c;

    root->addWidget(card(tr("Material"), &c, this));
    auto* mr = new QHBoxLayout;
    matName_ = new QLabel(this);
    matName_->setStyleSheet("font-weight: 600;");
    mr->addWidget(matName_, 1);
    auto* change = new QPushButton(tr("Change"), this);
    connect(change, &QPushButton::clicked, this, [this] { app_->setTab("part"); });
    mr->addWidget(change);
    c->addLayout(mr);
    matProps_ = note("", this);
    c->addWidget(matProps_);

    root->addWidget(card(tr("Heat"), &c, this));
    auto* br = new QHBoxLayout;
    const std::pair<QString, Item::Type> adds[] = {{tr("+ Temperature"), Item::Temp}, {tr("+ Heat power"), Item::Heat}, {tr("+ Convection"), Item::Conv}};
    const QString tips[] = {tr("Hold faces at a temperature"), tr("Heat power into faces (e.g. a chip, a heater)"), tr("Faces cooled by air or liquid")};
    int k = 0;
    for (const auto& [text, type] : adds) {
        auto* b = new QPushButton(text, this);
        b->setToolTip(tips[k++]);
        const Item::Type t = type;
        connect(b, &QPushButton::clicked, this, [this, t] { addItem(t); });
        br->addWidget(b);
    }
    br->addStretch();
    c->addLayout(br);
    list_ = new QListWidget(this);
    list_->setObjectName("items");
    connect(list_, &QListWidget::currentRowChanged, this, [this](int row) {
        if (row < 0 || row == selected_) return;
        selected_ = row;
        renderEditor();
        drawOverlays();
    });
    c->addWidget(list_);
    editor_ = new QWidget(this);
    new QVBoxLayout(editor_);
    editor_->layout()->setContentsMargins(0, 0, 0, 0);
    c->addWidget(editor_);
    ambient_ = new QCheckBox(tr("Convection on all other faces"), this);
    ambient_->setChecked(true);
    c->addWidget(ambient_);
    ambientProps_ = new QWidget(this);
    auto* ap = new QVBoxLayout(ambientProps_);
    ap->setContentsMargins(0, 0, 0, 0);
    ambientH_ = numberBox(10, 0, 1e6, 1, 1, [this](double) { markStale(); });
    ambientT_ = numberBox(20, -273, 1e4, 1, 1, [this](double) { markStale(); });
    ap->addWidget(labelled(tr("Film coefficient (W/m²K)"), ambientH_, ambientProps_));
    ap->addWidget(labelled(tr("Air temperature (°C)"), ambientT_, ambientProps_));
    c->addWidget(ambientProps_);
    connect(ambient_, &QCheckBox::toggled, this, [this](bool on) { ambientProps_->setVisible(on); markStale(); });
    c->addWidget(note(tr("Still air ≈ 5–10 W/m²K, a fan ≈ 25–100, water ≈ 500–5000."), this));

    root->addWidget(card(tr("Analysis"), &c, this));
    modeSeg_ = segmented({tr("Steady state"), tr("Over time")}, 0, [this](int i) {
        transient_ = i == 1;
        transientBox_->setVisible(transient_);
        markStale();
    }, this);
    c->addWidget(modeSeg_);
    transientBox_ = new QWidget(this);
    auto* tb = new QVBoxLayout(transientBox_);
    tb->setContentsMargins(0, 4, 0, 0);
    duration_ = numberBox(600, 0.001, 1e8, 10, 3, [this](double) { markStale(); });
    steps_ = numberBox(40, 2, 400, 1, 0, [this](double) { markStale(); });
    initial_ = numberBox(20, -273, 1e4, 1, 1, [this](double) { markStale(); });
    tb->addWidget(labelled(tr("Duration (s)"), duration_, transientBox_));
    tb->addWidget(labelled(tr("Time steps"), steps_, transientBox_));
    tb->addWidget(labelled(tr("Starting temperature (°C)"), initial_, transientBox_));
    transientBox_->hide();
    c->addWidget(transientBox_);

    root->addWidget(card(tr("Mesh"), &c, this));
    auto* rr = new QHBoxLayout;
    rr->addWidget(new QLabel(tr("Voxels on longest side"), this), 1);
    resOut_ = new QLabel("56", this);
    rr->addWidget(resOut_);
    c->addLayout(rr);
    res_ = new QSlider(Qt::Horizontal, this);
    res_->setRange(16, 480);
    res_->setSingleStep(4);
    res_->setPageStep(16);
    res_->setValue(56);
    connect(res_, &QSlider::valueChanged, this, [this](int v) {
        const int snapped = int(std::lround(v / 4.0) * 4);
        if (snapped != v) { res_->setValue(snapped); return; }
        resOut_->setText(QString::number(v));
        updateMeshInfo();
    });
    connect(res_, &QSlider::sliderReleased, this, [this] { markStale(); });
    c->addWidget(res_);
    meshInfo_ = note("", this);
    c->addWidget(meshInfo_);

    auto* runBtn = new QPushButton(tr("Run heat transfer"), this);
    runBtn->setObjectName("primary");
    runBtn->setMinimumHeight(36);
    connect(runBtn, &QPushButton::clicked, this, [this] { run(); });
    root->addWidget(runBtn);

    resultsCard_ = card(tr("Results"), &c, this);
    alert_ = new QLabel(this);
    alert_->setObjectName("alert");
    alert_->setProperty("kind", "error");
    alert_->setWordWrap(true);
    alert_->hide();
    c->addWidget(alert_);
    kpis_ = new KpiGrid(this);
    c->addWidget(kpis_);
    plot_ = new QComboBox(this);
    plot_->addItem(tr("Temperature"), "T");
    plot_->addItem(tr("Heat flux"), "flux");
    connect(plot_, &QComboBox::activated, this, [this](int i) { view_.plot = plot_->itemData(i).toString(); show(); });
    auto* pr = new QHBoxLayout;
    pr->addWidget(new QLabel(tr("Plot"), this));
    pr->addWidget(plot_, 1);
    c->addLayout(pr);
    timeBox_ = new QWidget(this);
    auto* tl = new QVBoxLayout(timeBox_);
    tl->setContentsMargins(0, 0, 0, 0);
    tl->addWidget(note(tr("Hottest point over time"), timeBox_));
    LineChart::Options co;
    co.xLabel = tr("time (s)");
    co.integerX = false;
    co.formatX = [](double x) { return num(x); };
    co.formatY = [](double y) { return num(y) + " °C"; };
    co.tipText = [](double x, double y, int) { return tr("%1 s · hottest %2 °C").arg(num(x), num(y)); };
    chart_ = new LineChart(timeBox_, co);
    chart_->onPick = [this](int i) { view_.playing = false; setFrame(i); };
    tl->addWidget(chart_);
    auto* fr = new QHBoxLayout;
    fr->addWidget(new QLabel(tr("Time"), timeBox_));
    frame_ = new QSlider(Qt::Horizontal, timeBox_);
    connect(frame_, &QSlider::valueChanged, this, [this](int v) { view_.playing = false; setFrame(v); });
    fr->addWidget(frame_, 1);
    frameOut_ = new QLabel("0 s", timeBox_);
    fr->addWidget(frameOut_);
    tl->addLayout(fr);
    play_ = new QPushButton(tr("▶ Play"), timeBox_);
    connect(play_, &QPushButton::clicked, this, [this] {
        if (!result_ || result_->frames.empty()) return;
        view_.playing = !view_.playing;
        if (view_.playing && view_.frame >= int(result_->frames.size()) - 1) view_.frame = 0;
        view_.t = 0;
        play_->setText(view_.playing ? tr("❚❚ Pause") : tr("▶ Play"));
    });
    auto* pl = new QHBoxLayout;
    pl->addWidget(play_);
    pl->addStretch();
    tl->addLayout(pl);
    c->addWidget(timeBox_);
    auto* toggles = new QHBoxLayout;
    auto* bands = new QCheckBox(tr("Contour bands"), this);
    connect(bands, &QCheckBox::toggled, this, [this](bool on) { view_.bands = on; show(); });
    auto* bcs = new QCheckBox(tr("Show heat inputs"), this);
    connect(bcs, &QCheckBox::toggled, this, [this](bool on) { view_.bcs = on; drawOverlays(); });
    toggles->addWidget(bands);
    toggles->addWidget(bcs);
    toggles->addStretch();
    c->addLayout(toggles);
    resultsCard_->hide();
    root->addWidget(resultsCard_);
    root->addStretch();
    compact(this);
    runBtn->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    renderList();
}

void ThermalPanel::renderMaterial() {
    const Material& m = app_->material;
    matName_->setText(QString::fromStdString(m.name));
    matProps_->setText(tr("k %1 W/mK · cp %2 J/kgK · density %3 kg/m³").arg(num(m.k), num(m.cp), num(m.density)));
}

void ThermalPanel::updateMeshInfo() {
    if (!app_->part) { meshInfo_->clear(); return; }
    const auto& s = app_->part->bbox.size;
    meshInfo_->setText(tr("Voxel size ≈ %1 %2.").arg(num(std::max({s[0], s[1], s[2]}) / res_->value()), app_->units));
}

void ThermalPanel::reset() {
    cancel();
    items.clear();
    selected_ = -1;
    result_.reset();
    display_ = "setup";
    resultsCard_->hide();
    if (app_->part) res_->setValue(app_->structural->resolution());
    renderList();
    renderMaterial();
    updateMeshInfo();
}

void ThermalPanel::onPartScaled(double s, const Vec3& pivot) {
    cancel();
    for (auto& it : items)
        for (auto& p : it.patches)
            if (p.clip) {
                Vec3 c = p.clip->center;
                for (int d = 0; d < 3; d++) c[d] = pivot[d] + (c[d] - pivot[d]) * s;
                p.clip = Clip{c, p.clip->radius * s};
            }
    result_.reset();
    display_ = "setup";
    resultsCard_->hide();
    renderList();
    updateMeshInfo();
}

void ThermalPanel::markStale() {
    app_->psimEdited();
    cancel();
    if (result_ && !result_->stale) {
        result_->stale = true;
        app_->status(tr("Thermal setup changed - run it again to update the results."), "warn");
        if (display_ == "results") show();
    }
    drawOverlays();
}

void ThermalPanel::cancel() {
    if (!job) return;
    job->cancel();
    job.reset();
    app_->busy->hideBusy();
    app_->status(tr("Heat transfer cancelled."), "warn");
}

// ---------- heat inputs ----------

void ThermalPanel::addItem(Item::Type type) {
    if (!app_->part) return app_->status(tr("Import a part or open a sample first."), "error");
    int n = 1;
    for (const auto& it : items) n += it.type == type;
    Item it;
    it.type = type;
    it.name = QString("%1 %2").arg(label(type)).arg(n);
    it.value = type == Item::Temp ? 80 : type == Item::Heat ? 10 : 25;
    it.ambient = 20;
    items.push_back(std::move(it));
    selected_ = int(items.size()) - 1;
    renderList();
    pick(selected_, true);
}

void ThermalPanel::addOnFace(Item::Type type, int face, double value) {
    if (!app_->part) return;
    Item it;
    it.type = type;
    it.name = QString("%1 %2").arg(label(type)).arg(items.size() + 1);
    it.value = value;
    Patch p;
    p.tris = trianglesOfFace(*app_->part, face);
    if (p.tris.empty()) return app_->status(tr("No face %1 on this part.").arg(face), "error");
    it.patches.push_back(std::move(p));
    items.push_back(std::move(it));
    selected_ = int(items.size()) - 1;
    markStale();
    renderList();
}

void ThermalPanel::setTransient(bool on) {
    transient_ = on;
    transientBox_->setVisible(on);
    if (auto* b = modeSeg_->findChildren<QToolButton*>().value(on ? 1 : 0)) b->setChecked(true);
}

void ThermalPanel::pick(int index, bool isNew) {
    showSetup();
    const size_t before = items[index].patches.size();
    Picker::Session s;
    s.title = items[index].name;
    s.color = COLORS[items[index].type];
    s.onPatch = [this, index](const Patch& p) {
        if (p.tris.empty() || index >= int(items.size())) return;
        auto& patches = items[index].patches;
        if (!p.clip)
            for (const auto& o : patches)
                if (!o.clip && o.tris == p.tris) return;
        patches.push_back(p);
        markStale();
        renderList();
    };
    s.onDone = [this, index, before, isNew](bool commit) {
        if (index >= int(items.size())) return;
        if (!commit) items[index].patches.resize(before);
        if (items[index].patches.empty() && isNew) {
            items.erase(items.begin() + index);
            if (selected_ >= int(items.size())) selected_ = int(items.size()) - 1;
        }
        renderList();
        drawOverlays();
    };
    app_->picker->start(s);
}

void ThermalPanel::renderList() {
    list_->blockSignals(true);
    list_->clear();
    for (size_t i = 0; i < items.size(); i++) {
        const auto& it = items[i];
        const QString meta = metaText(it);
        auto* item = new QListWidgetItem(list_);
        auto* w = new QWidget(list_);
        auto* h = new QHBoxLayout(w);
        h->setContentsMargins(6, 2, 4, 2);
        auto* sw = new QLabel(w);
        sw->setFixedSize(10, 10);
        sw->setStyleSheet(QString("background:%1;border-radius:3px").arg(COLORS[it.type].name()));
        h->addWidget(sw);
        h->addWidget(new QLabel(it.name, w), 1);
        auto* m = new QLabel(meta, w);
        m->setObjectName("meta");
        h->addWidget(m);
        auto* add = new QToolButton(w);
        add->setText("+");
        add->setToolTip(tr("Add more areas"));
        connect(add, &QToolButton::clicked, this, [this, i] { pick(int(i), false); });
        h->addWidget(add);
        auto* del = new QToolButton(w);
        del->setText("×");
        del->setToolTip(tr("Delete"));
        connect(del, &QToolButton::clicked, this, [this, i] {
            QMetaObject::invokeMethod(this, [this, i] {
                if (i < items.size()) items.erase(items.begin() + long(i));
                if (selected_ >= int(items.size())) selected_ = int(items.size()) - 1;
                markStale();
                renderList();
            }, Qt::QueuedConnection);
        });
        h->addWidget(del);
        item->setSizeHint(w->sizeHint());
        list_->setItemWidget(item, w);
    }
    list_->setCurrentRow(selected_);
    list_->blockSignals(false);
    fitList(list_);
    renderEditor();
    drawOverlays();
}

void ThermalPanel::renderEditor() {
    auto* layout = static_cast<QVBoxLayout*>(editor_->layout());
    while (QLayoutItem* it = layout->takeAt(0)) {
        if (it->widget()) { it->widget()->hide(); it->widget()->deleteLater(); }
        delete it;
    }
    if (selected_ < 0 || selected_ >= int(items.size())) { editor_->hide(); return; }
    editor_->show();
    const int i = selected_;
    Item& it = items[i];
    auto box = [this, i](double value, double min, std::function<void(Item&, double)> set) {
        return numberBox(value, min, 1e9, 1, 2, [this, i, set](double v) {
            if (i >= int(items.size())) return;
            set(items[i], v);
            markStale();
            // refresh the list's meta text without rebuilding this editor
            const Item& x = items[i];
            if (auto* w = list_->itemWidget(list_->item(i)))
                if (auto* meta = w->findChild<QLabel*>("meta")) meta->setText(metaText(x));
        });
    };
    if (it.type == Item::Temp) layout->addWidget(labelled(tr("Temperature (°C)"), box(it.value, -273, [](Item& x, double v) { x.value = v; }), editor_));
    if (it.type == Item::Heat) layout->addWidget(labelled(tr("Heat power, total (W)"), box(it.value, 0, [](Item& x, double v) { x.value = v; }), editor_));
    if (it.type == Item::Conv) {
        layout->addWidget(labelled(tr("Film coefficient (W/m²K)"), box(it.value, 0, [](Item& x, double v) { x.value = v; }), editor_));
        layout->addWidget(labelled(tr("Fluid temperature (°C)"), box(it.ambient, -273, [](Item& x, double v) { x.ambient = v; }), editor_));
    }
    auto* area = new QPushButton(tr("+ Area"), editor_);
    connect(area, &QPushButton::clicked, this, [this, i] { pick(i, false); });
    auto* row = new QWidget(editor_);
    auto* h = new QHBoxLayout(row);
    h->setContentsMargins(0, 0, 0, 0);
    h->addWidget(area);
    h->addStretch();
    layout->addWidget(row);
    compact(editor_);
}

void ThermalPanel::drawOverlays() {
    Viewport* v = app_->viewer;
    if (app_->tab() != "thermal") return;
    v->clearLayer("overlays");
    if (!app_->part || !(display_ != "results" || view_.bcs || app_->picker->active())) return;
    std::vector<Geometry> g;
    for (size_t i = 0; i < items.size(); i++) {
        std::set<int32_t> tris;
        for (const auto& p : items[i].patches) tris.insert(p.tris.begin(), p.tris.end());
        if (tris.empty()) continue;
        g.push_back(shapes::patch(*app_->part, std::vector<int32_t>(tris.begin(), tris.end()), rgb(COLORS[items[i].type]), int(i) == selected_ ? 0.6f : 0.42f));
    }
    v->setLayer("overlays", std::move(g));
}

// ---------- run ----------

void ThermalPanel::run() {
    if (!app_->part) return app_->status(tr("Import a part or open a sample first."), "error");
    cancel();
    app_->picker->finish(true);
    Setup s;
    s.part = app_->part;
    s.res = res_->value();
    s.cached = app_->structural->cachedModel(s.res);
    s.items = items;
    s.ambient = ambient_->isChecked();
    s.ambientH = ambientH_->value();
    s.ambientT = ambientT_->value();
    s.toMeters = app_->toMeters();
    s.material = app_->material;
    s.transient = transient_;
    s.duration = duration_->value();
    s.steps = int(steps_->value());
    s.initial = initial_->value();
    if (s.transient && (!(s.duration > 0) || s.steps < 2)) return app_->status(tr("Set a positive duration and at least 2 time steps."), "error");
    if (!(s.material.k > 0) || (s.transient && !(s.material.cp > 0 && s.material.density > 0)))
        return app_->status(tr("The material needs a thermal conductivity%1 (Part tab, Custom material).").arg(s.transient ? tr(", heat capacity and density") : QString()), "error");
    struct Out {
        std::shared_ptr<StructuralModel> model;
        Result r;
        std::vector<std::string> warnings;
    };
    app_->busy->showBusy(tr("Voxelizing part…"), [this] { cancel(); });
    job = runJob<Out>(this,
        [s](JobControl& ctl) {
            Out o;
            ctl.progress(0, "Voxelizing part…");
            o.model = s.cached ? s.cached : std::make_shared<StructuralModel>(s.part, s.res);
            ctl.check();
            const StructuralModel& m = *o.model;
            if (!m.voxelCount) throw std::runtime_error("The part produced no voxels. Increase the mesh resolution.");
            auto bc = assemble(m, s);
            o.warnings = bc.warnings;
            const bool anyFixed = std::any_of(bc.fixedNode.begin(), bc.fixedNode.end(), [](uint8_t x) { return x; });
            const bool anyConv = std::any_of(bc.convH.begin(), bc.convH.end(), [](double x) { return x > 0; });
            if (!s.transient && !anyFixed && !anyConv)
                throw std::runtime_error("Steady state needs somewhere for the heat to go: add a fixed temperature or convection (or run it over time).");
            HeatInput in;
            in.dims = m.grid.dims;
            in.density = m.density;
            in.fixedNode = std::move(bc.fixedNode);
            in.fixedValue = std::move(bc.fixedValue);
            in.source = std::move(bc.source);
            in.convH = std::move(bc.convH);
            in.convT = std::move(bc.convT);
            in.k = s.material.k;
            in.h = m.grid.h * s.toMeters;
            in.rhoCp = s.material.density * s.material.cp;
            in.duration = s.transient ? s.duration : 0;
            in.steps = s.transient ? s.steps : 1;
            in.initial = s.initial;
            Result& r = o.r;
            r.material = s.material;
            r.transient = s.transient;
            r.heatIn = bc.heatIn;
            r.voxels = m.voxelCount;
            VertexWeights W;
            auto res = solveHeat(
                in,
                [&](int i, double t, const std::vector<double>& T) {
                    Frame f;
                    f.t = t;
                    f.T = interpolate(W, T.data());
                    f.min = INFINITY;
                    f.max = -INFINITY;
                    for (float x : f.T)
                        if (!std::isnan(x)) { f.min = std::min(f.min, double(x)); f.max = std::max(f.max, double(x)); }
                    r.frames.push_back(std::move(f));
                    ctl.progress(s.transient ? double(i) / s.steps : 0.9, s.transient ? "Heat transfer · step " + std::to_string(i) + "/" + std::to_string(s.steps) : "Heat transfer");
                    return ctl.cancelled();
                },
                [&](const ScalarVoxelSolver& solver) {
                    ctl.progress(0.05, s.transient ? "Heat transfer over time…" : "Heat transfer…");
                    W = m.vertexWeights(solver.levels[0].active);
                });
            ctl.check();
            r.flux = interpolate(W, heatFlux(*res.solver, res.T, s.material.k, in.h).data());
            r.converged = res.converged;
            return o;
        },
        [this](Out o) {
            job.reset();
            app_->busy->hideBusy();
            app_->structural->rememberModel(o.model);
            for (const auto& w : o.warnings) app_->status(QString::fromStdString(w), "warn");
            result_ = std::move(o.r);
            app_->psimRunDone("thermal", {{"min", result_->frames.back().min}, {"max", result_->frames.back().max}});
            display_ = "results";
            view_.frame = std::max(0, int(result_->frames.size()) - 1);
            view_.playing = false;
            resultsCard_->show();
            timeBox_->setVisible(result_->transient);
            show();
            const auto& last = result_->frames.back();
            app_->status(tr("Heat transfer on %1 voxels: %2–%3 °C%4.%5")
                             .arg(QLocale().toString(result_->voxels), num(last.min), num(last.max),
                                  result_->transient ? tr(" after %1 s").arg(num(last.t)) : QString(),
                                  result_->converged ? QString() : tr(" The solver did not fully converge.")),
                         result_->converged ? "" : "warn");
            for (QWidget* w = resultsCard_->parentWidget(); w; w = w->parentWidget())
                if (auto* sa = qobject_cast<QScrollArea*>(w)) {
                    QTimer::singleShot(60, resultsCard_, [sa, c = resultsCard_] { sa->verticalScrollBar()->setValue(c->mapTo(sa->widget(), QPoint(0, 0)).y() - 8); });
                    break;
                }
        },
        [this](QString msg, bool cancelled) {
            job.reset();
            app_->busy->hideBusy();
            app_->status(cancelled ? tr("Cancelled.") : msg, cancelled ? "" : "error");
        },
        [this](double f, QString text) { app_->busy->progress(f, text); });
}

// ---------- display ----------

void ThermalPanel::showSetup() {
    display_ = "setup";
    Viewport* v = app_->viewer;
    v->setScalars(nullptr);
    v->setDeformation(nullptr);
    v->clearMarker();
    app_->legend->clear();
    drawOverlays();
}

void ThermalPanel::activate() {
    renderMaterial();
    updateMeshInfo();
    if (display_ == "results" && result_) show();
    else showSetup();
}

void ThermalPanel::deactivate() {
    view_.playing = false;
    play_->setText(tr("▶ Play"));
    Viewport* v = app_->viewer;
    v->clearLayer("overlays");
    v->clearMarker();
    v->setScalars(nullptr);
    app_->legend->clear();
}

void ThermalPanel::setFrame(int i) {
    if (!result_ || result_->frames.empty()) return;
    view_.frame = std::clamp(i, 0, int(result_->frames.size()) - 1);
    show();
}

void ThermalPanel::show() {
    if (!result_ || app_->tab() != "thermal" || display_ != "results") return;
    const Result& r = *result_;
    const Frame& cur = r.frames[std::min(view_.frame, int(r.frames.size()) - 1)];
    const bool last = view_.frame >= int(r.frames.size()) - 1;
    const bool flux = view_.plot == "flux" && last;
    LegendSpec spec;
    spec.bands = view_.bands ? 12 : 0;
    spec.heat = true;
    const std::vector<float>* values;
    if (flux) {
        values = &r.flux;
        double mx = 0;
        for (float x : r.flux)
            if (!std::isnan(x)) mx = std::max(mx, double(x));
        if (!(mx > 0)) mx = 1;
        spec.min = 0;
        spec.max = mx;
        spec.title = tr("Heat flux");
        spec.format = [mx](double x) { return mx >= 1e4 ? num(x / 1e3) + " kW/m²" : num(x) + " W/m²"; };
    } else {
        values = &cur.T;
        // fixed colour scale over the whole run so the animation compares like with like
        double lo = INFINITY, hi = -INFINITY;
        for (const auto& f : r.frames) { lo = std::min(lo, f.min); hi = std::max(hi, f.max); }
        if (!(hi > lo)) hi = lo + 1;
        spec.min = lo;
        spec.max = hi;
        spec.title = tr("Temperature");
        spec.format = [](double x) { return num(x) + " °C"; };
    }
    spec.sub = QString::fromStdString(r.material.name) + (r.transient ? tr(" · t = %1 s").arg(num(cur.t)) : tr(" · steady state")) +
               (view_.plot == "flux" && !last ? tr(" · flux shown at the end only") : QString());
    ColorSpec cs;
    cs.min = float(spec.min);
    cs.max = float(spec.max);
    cs.bands = spec.bands;
    cs.heat = true;
    Viewport* v = app_->viewer;
    v->setScalars(values, cs);
    v->setDeformation(nullptr);
    shown_ = *values;
    shownFmt_ = spec.format;
    app_->legend->setSpecs({spec});
    // hottest point
    double hi = -INFINITY;
    int hot = -1;
    for (size_t i = 0; i < values->size(); i++)
        if (!std::isnan((*values)[i]) && (*values)[i] > hi) { hi = (*values)[i]; hot = int(i); }
    const auto& V = app_->part->vertices;
    if (hot >= 0) {
        const QString text = flux ? tr("Highest heat flux") : tr("Hottest point");
        const QVector3D p(V[3 * hot], V[3 * hot + 1], V[3 * hot + 2]);
        if (v->hasMarker() && v->markerLabel() == text) v->moveMarker(p);
        else v->setMarker(p, text);
    } else v->clearMarker();
    drawOverlays();
    kpis_->setKpis({
        {tr("Hottest"), num(cur.max) + " °C", r.transient ? tr("at %1 s").arg(num(cur.t)) : tr("steady state")},
        {tr("Coolest"), num(cur.min) + " °C", tr("spread %1 °C").arg(num(cur.max - cur.min))},
        {tr("Heat input"), num(r.heatIn) + " W", tr("from heat-power faces")},
    });
    alert_->setVisible(r.stale);
    if (r.stale) alert_->setText(tr("The setup changed since this run - run it again to update."));
    if (r.transient) {
        frame_->blockSignals(true);
        frame_->setRange(0, std::max(0, int(r.frames.size()) - 1));
        frame_->setValue(view_.frame);
        frame_->blockSignals(false);
        frameOut_->setText(num(cur.t) + " s");
        std::vector<QPointF> pts;
        for (const auto& f : r.frames) pts.push_back(QPointF(f.t, f.max));
        chart_->setPoints(pts, view_.frame);
    }
}

QString ThermalPanel::probeText(const QPointF& pos) {
    if (display_ != "results" || shown_.empty() || !shownFmt_) return {};
    auto hit = app_->viewer->pickPart(pos);
    if (!hit) return {};
    const auto& T = app_->part->tris;
    double s = 0;
    for (int k = 0; k < 3; k++) s += hit->bary[k] * shown_[T[3 * hit->tri + k]];
    return std::isnan(s) ? tr("no data") : shownFmt_(s);
}

void ThermalPanel::tick(double dt) {
    if (!view_.playing || app_->tab() != "thermal" || !result_) return;
    view_.t += dt;
    if (view_.t < 0.12) return;
    view_.t = 0;
    if (view_.frame >= int(result_->frames.size()) - 1) {
        view_.playing = false;
        play_->setText(tr("▶ Play"));
        return;
    }
    setFrame(view_.frame + 1);
}


// ---------- .psim files ----------

json::Value ThermalPanel::exportState() const {
    using namespace pj;
    Value o = jobj();
    o.obj["mode"] = jstr(transient_ ? "transient" : "steady");
    Value its = jarr();
    for (const auto& it : items) {
        Value e = jobj();
        e.obj["name"] = jstr(it.name.toStdString());
        e.obj["type"] = jstr(it.type == Item::Temp ? "temp" : it.type == Item::Heat ? "heat" : "conv");
        e.obj["value"] = jnum(it.value);
        e.obj["ambient"] = jnum(it.ambient);
        Value ps = jarr();
        for (const auto& p : it.patches) ps.arr.push_back(patchToJson(p));
        e.obj["patches"] = ps;
        its.arr.push_back(e);
    }
    o.obj["items"] = its;
    Value a = jobj();
    a.obj["enabled"] = jbool(ambient_->isChecked());
    a.obj["h"] = jnum(ambientH_->value());
    a.obj["t"] = jnum(ambientT_->value());
    o.obj["ambient"] = a;
    o.obj["duration"] = jnum(duration_->value());
    o.obj["steps"] = jnum(steps_->value());
    o.obj["initial"] = jnum(initial_->value());
    o.obj["resolution"] = jnum(res_->value());
    return o;
}

void ThermalPanel::importState(const json::Value& s) {
    using namespace pj;
    const int nTri = app_->part->nTri;
    std::vector<Item> list;
    for (const auto& j : s["items"].arr) {
        const std::string t = strOr(j["type"]);
        if (t != "temp" && t != "heat" && t != "conv") throw std::runtime_error("The file has a thermal condition of an unknown type (" + t + ").");
        Item it;
        it.type = t == "temp" ? Item::Temp : t == "heat" ? Item::Heat : Item::Conv;
        it.name = QString::fromStdString(strOr(j["name"], "Condition"));
        it.value = numOr(j["value"], 0);
        it.ambient = numOr(j["ambient"], 20);
        for (const auto& p : j["patches"].arr) {
            Patch q = patchFromJson(p, nTri);
            if (!q.tris.empty()) it.patches.push_back(std::move(q));
        }
        list.push_back(std::move(it));
    }
    items = std::move(list);
    selected_ = items.empty() ? -1 : 0;
    setTransient(strOr(s["mode"]) == "transient");
    const Value& a = s["ambient"];
    ambient_->setChecked(boolOr(a["enabled"], true));
    ambientProps_->setVisible(ambient_->isChecked());
    if (isNum(a["h"])) ambientH_->setValue(a["h"].num);
    if (isNum(a["t"])) ambientT_->setValue(a["t"].num);
    if (isNum(s["duration"])) duration_->setValue(s["duration"].num);
    if (isNum(s["steps"])) steps_->setValue(s["steps"].num);
    if (isNum(s["initial"])) initial_->setValue(s["initial"].num);
    if (isNum(s["resolution"])) res_->setValue(int(s["resolution"].num));
    renderList();
    renderEditor();
    updateMeshInfo();
}

std::optional<psim::Arrays> ThermalPanel::exportResult() const {
    using namespace pj;
    if (!result_ || result_->frames.empty()) return std::nullopt;
    const Result& r = *result_;
    const Frame& last = r.frames.back();
    psim::Arrays out;
    Value m = jobj();
    m.obj["min"] = jnum(last.min);
    m.obj["max"] = jnum(last.max);
    m.obj["transient"] = jbool(r.transient);
    m.obj["converged"] = jbool(r.converged);
    m.obj["heatIn"] = jnum(r.heatIn);
    m.obj["voxels"] = jnum(r.voxels);
    m.obj["material"] = materialToJson(r.material);
    Value ft = jarr();
    for (const auto& f : r.frames) {
        Value e = jobj();
        e.obj["t"] = jnum(f.t);
        e.obj["min"] = jnum(f.min);
        e.obj["max"] = jnum(f.max);
        ft.arr.push_back(e);
    }
    m.obj["frameTimes"] = ft;
    out.meta = m;
    auto add = [&](const std::string& name, const std::vector<float>& d) {
        psim::Array a;
        a.name = name;
        a.enc = psim::Enc::Q16;
        a.f = d;
        out.list.push_back(std::move(a));
    };
    add("thermal.T", last.T);
    add("thermal.flux", r.flux);
    for (size_t i = 0; i < r.frames.size(); i++) add("thermal.frame." + std::to_string(i), r.frames[i].T);
    return out;
}

void ThermalPanel::importResult(const json::Value& m, const psim::Arrays& arrays) {
    using namespace pj;
    const size_t nV = size_t(app_->part->nVert);
    auto get = [&](const std::string& name) -> const std::vector<float>& {
        const psim::Array* a = arrays.find(name);
        if (!a || a->f.size() != nV) throw std::runtime_error("The thermal result in the file does not fit this part.");
        return a->f;
    };
    Result r;
    r.flux = get("thermal.flux");
    const auto& times = m["frameTimes"].arr;
    for (size_t i = 0; i < times.size(); i++) {
        Frame f;
        f.t = numOr(times[i]["t"], 0);
        f.min = numOr(times[i]["min"], 0);
        f.max = numOr(times[i]["max"], 0);
        f.T = get("thermal.frame." + std::to_string(i));
        r.frames.push_back(std::move(f));
    }
    if (r.frames.empty()) {
        Frame f;
        f.t = 0;
        f.min = numOr(m["min"], 0);
        f.max = numOr(m["max"], 0);
        f.T = get("thermal.T");
        r.frames.push_back(std::move(f));
    }
    r.material = m["material"].type == Value::Object ? materialFromJson(m["material"]) : app_->material;
    r.transient = boolOr(m["transient"], false);
    r.converged = boolOr(m["converged"], true);
    r.heatIn = numOr(m["heatIn"], 0);
    r.voxels = int(numOr(m["voxels"], 0));
    result_ = std::move(r);
    display_ = "results";
    view_.frame = int(result_->frames.size()) - 1;
    view_.playing = false;
    resultsCard_->show();
    timeBox_->setVisible(result_->transient);
}

json::Value ThermalPanel::exportView() const {
    using namespace pj;
    Value v = jobj();
    v.obj["plot"] = jstr(view_.plot.toStdString());
    v.obj["frame"] = jnum(view_.frame);
    v.obj["bands"] = jbool(view_.bands);
    v.obj["bcs"] = jbool(view_.bcs);
    return v;
}

void ThermalPanel::importView(const json::Value& v) {
    using namespace pj;
    const QString plot = QString::fromStdString(strOr(v["plot"]));
    if (plot == "T" || plot == "flux") { view_.plot = plot; plot_->setCurrentIndex(plot_->findData(plot)); }
    if (result_ && isNum(v["frame"])) view_.frame = std::clamp(int(v["frame"].num), 0, int(result_->frames.size()) - 1);
    auto setBox = [this](const QString& text, bool on) {
        for (auto* b : findChildren<QCheckBox*>()) if (b->text() == text) b->setChecked(on);
    };
    view_.bands = boolOr(v["bands"], view_.bands);
    view_.bcs = boolOr(v["bcs"], view_.bcs);
    setBox(tr("Contour bands"), view_.bands);
    setBox(tr("Show heat inputs"), view_.bcs);
}

}  // namespace ps
