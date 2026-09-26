#include "partpanel.hpp"

#include <QButtonGroup>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFormLayout>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QToolButton>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <functional>

#include "mainwindow.hpp"
#include "widgets.hpp"

namespace ps {

PartPanel::PartPanel(MainWindow* app) : app_(app) {
    auto* root = new QVBoxLayout(this);
    root->setContentsMargins(10, 10, 10, 10);
    root->setSpacing(10);

    QVBoxLayout* c;
    root->addWidget(card(tr("Geometry"), &c, this));
    empty_ = note(tr("Import a CAD file or open a sample to get started. You can also drop a file onto the viewport."), this);
    c->addWidget(empty_);
    statsBox_ = new QWidget(this);
    stats_ = new QFormLayout(statsBox_);
    stats_->setContentsMargins(0, 0, 0, 0);
    c->addWidget(statsBox_);

    root->addWidget(card(tr("Material"), &c, this));
    c->addWidget(note(tr("Used by every study: stiffness and strength (structural), conductivity and heat capacity (thermal), and the part's weight against the wind (airflow)."), this));
    materialBox_ = new QComboBox(this);
    for (const auto& m : materials()) materialBox_->addItem(QString::fromStdString(m.name), QString::fromStdString(m.id));
    materialBox_->addItem(tr("Custom…"), "custom");
    connect(materialBox_, &QComboBox::activated, this, [this](int i) {
        const QString id = materialBox_->itemData(i).toString();
        if (id == "custom") {
            Material m = app_->material;
            m.id = "custom";
            if (!QString::fromStdString(m.name).startsWith("Custom")) m.name = "Custom (" + m.name + ")";
            app_->setMaterial(m);
        } else if (const Material* m = findMaterial(id.toStdString())) {
            app_->setMaterial(*m);
        }
    });
    c->addWidget(materialBox_);
    materialFields_ = new QWidget(this);
    c->addWidget(materialFields_);
    buildMaterialFields();

    root->addWidget(card(tr("Size"), &c, this));
    c->addWidget(note(tr("Scale the part up or down. Fixtures and loads stay on the same faces."), this));
    auto* grid = new QGridLayout;
    scaleFactor_ = new QDoubleSpinBox(this);
    scaleFactor_->setRange(0.0001, 10000);
    scaleFactor_->setDecimals(4);
    scaleFactor_->setValue(1);
    auto* scaleBtn = new QPushButton(tr("Scale"), this);
    connect(scaleBtn, &QPushButton::clicked, this, [this] { app_->applyScale(scaleFactor_->value()); scaleFactor_->setValue(1); });
    targetSize_ = new QDoubleSpinBox(this);
    targetSize_->setRange(0, 1e7);
    targetSize_->setDecimals(3);
    sizeUnit_ = new QLabel("mm", this);
    auto* sizeBtn = new QPushButton(tr("Set size"), this);
    connect(sizeBtn, &QPushButton::clicked, this, [this] {
        if (!app_->part || !(targetSize_->value() > 0)) return app_->status(tr("Enter the new length of the longest side."), "error");
        const auto& s = app_->part->bbox.size;
        app_->applyScale(targetSize_->value() / std::max({s[0], s[1], s[2]}));
    });
    grid->addWidget(new QLabel(tr("Scale by"), this), 0, 0);
    grid->addWidget(scaleFactor_, 0, 1);
    grid->addWidget(scaleBtn, 0, 2);
    auto* ls = new QHBoxLayout;
    ls->addWidget(new QLabel(tr("Longest side"), this));
    ls->addWidget(sizeUnit_);
    grid->addLayout(ls, 1, 0);
    grid->addWidget(targetSize_, 1, 1);
    grid->addWidget(sizeBtn, 1, 2);
    c->addLayout(grid);
    auto* quick = new QHBoxLayout;
    for (double f : {0.1, 0.5, 2.0, 10.0}) {
        auto* b = new QPushButton(QString("× %1").arg(f), this);
        connect(b, &QPushButton::clicked, this, [this, f] { app_->applyScale(f); });
        quick->addWidget(b);
    }
    c->addLayout(quick);

    root->addWidget(card(tr("Units"), &c, this));
    c->addWidget(note(tr("STEP, IGES and SolidWorks files carry their own units. For STL/OBJ, choose the unit the file was modelled in."), this));
    unitsSeg_ = segmented({"mm", "cm", "m", "in"}, 0, [this](int i) {
        static const char* u[4] = {"mm", "cm", "m", "in"};
        if (!updating_) app_->setUnits(u[i]);
    }, this);
    c->addWidget(unitsSeg_);

    root->addWidget(card(tr("Orientation"), &c, this));
    c->addWidget(note(tr("Y is up and gravity points down (−Y). Rotate the part if it was modelled Z-up."), this));
    auto* rot = new QHBoxLayout;
    const char* axes[3] = {"X", "Y", "Z"};
    for (int a = 0; a < 3; a++) {
        auto* b = new QPushButton(tr("Rotate %1 90°").arg(axes[a]), this);
        connect(b, &QPushButton::clicked, this, [this, a] { app_->rotatePart(a); });
        rot->addWidget(b);
    }
    c->addLayout(rot);

    faceCard_ = card(tr("Face detection"), &c, this);
    root->addWidget(faceCard_);
    c->addWidget(note(tr("Mesh files have no CAD faces, so faces are grown across edges bent less than this angle. Click-selection uses these faces."), this));
    auto* fr = new QHBoxLayout;
    faceAngle_ = new QSlider(Qt::Horizontal, this);
    faceAngle_->setRange(2, 60);
    faceAngle_->setValue(20);
    faceAngleOut_ = new QLabel("20°", this);
    connect(faceAngle_, &QSlider::valueChanged, this, [this](int v) { faceAngleOut_->setText(QString("%1°").arg(v)); });
    connect(faceAngle_, &QSlider::sliderReleased, this, [this] {
        if (!app_->part || app_->part->brepFaces) return;
        setSmoothFaces(*app_->part, faceAngle_->value());
        refresh();
    });
    fr->addWidget(faceAngle_);
    fr->addWidget(faceAngleOut_);
    c->addLayout(fr);
    root->addStretch();
    compact(this);
    refresh();
}

void PartPanel::buildMaterialFields() {
    delete materialFields_->layout();
    for (auto* w : materialFields_->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly)) delete w;
    auto* g = new QGridLayout(materialFields_);
    g->setContentsMargins(0, 4, 0, 0);
    g->setHorizontalSpacing(10);
    const Material& m = app_->material;
    // each field edits one property; the percent field shows the elongation as %
    struct F { QString label, unit; std::function<double&(Material&)> ref; double min, max; int dec; double scale; };
    const std::vector<F> fields = {
        {tr("Young's modulus"), "GPa", [](Material& x) -> double& { return x.E; }, 1e-6, 1e4, 3, 1},
        {tr("Poisson's ratio"), "", [](Material& x) -> double& { return x.nu; }, -0.99, 0.499, 3, 1},
        {tr("Yield strength"), "MPa", [](Material& x) -> double& { return x.yield; }, 1e-6, 1e5, 1, 1},
        {tr("Tensile strength"), "MPa", [](Material& x) -> double& { return x.uts; }, 1e-6, 1e5, 1, 1},
        {tr("Density"), "kg/m³", [](Material& x) -> double& { return x.density; }, 1e-3, 1e5, 0, 1},
        {tr("Elongation at break"), "%", [](Material& x) -> double& { return x.elongation; }, 0.01, 200, 1, 100},
        {tr("Conductivity"), "W/m·K", [](Material& x) -> double& { return x.k; }, 1e-4, 1e4, 3, 1},
        {tr("Heat capacity"), "J/kg·K", [](Material& x) -> double& { return x.cp; }, 1, 1e5, 0, 1},
        {tr("Fatigue strength"), "MPa", [](Material& x) -> double& { return x.fatigue.Se; }, 1e-3, 1e5, 1, 1},
        {tr("Price"), "$/kg", [](Material& x) -> double& { return x.cost; }, 0, 1e6, 2, 1},
    };
    int row = 0, col = 0;
    for (const auto& f : fields) {
        auto* box = new QWidget(materialFields_);
        auto* v = new QVBoxLayout(box);
        v->setContentsMargins(0, 0, 0, 0);
        v->setSpacing(2);
        auto* lab = new QLabel(f.unit.isEmpty() ? f.label : QString("%1 <span style='color:gray'>%2</span>").arg(f.label, f.unit), box);
        lab->setObjectName("fieldLabel");
        v->addWidget(lab);
        Material copy = m;
        const double value = f.ref(copy) * f.scale;
        auto ref = f.ref;
        const double scale = f.scale;
        auto* s = numberBox(value, f.min, f.max, 0, f.dec, [this, ref, scale](double x) {
            if (updating_) return;
            Material mm = app_->material;
            ref(mm) = x / scale;
            if (mm.id != "custom") {
                mm.id = "custom";
                mm.name = "Custom (" + mm.name + ")";
            }
            app_->setMaterial(mm);
        }, box);
        v->addWidget(s);
        g->addWidget(box, row, col);
        if (++col == 2) { col = 0; row++; }
    }
    auto* modeBox = new QWidget(materialFields_);
    auto* mv = new QVBoxLayout(modeBox);
    mv->setContentsMargins(0, 0, 0, 0);
    mv->setSpacing(2);
    mv->addWidget(new QLabel(tr("Failure mode"), modeBox));
    auto* mode = new QComboBox(modeBox);
    mode->addItems({tr("Ductile (von Mises)"), tr("Brittle (principal)")});
    mode->setCurrentIndex(m.brittle ? 1 : 0);
    connect(mode, &QComboBox::activated, this, [this](int i) {
        Material mm = app_->material;
        mm.brittle = i == 1;
        app_->setMaterial(mm);
    });
    mv->addWidget(mode);
    g->addWidget(modeBox, row, col);
    compact(materialFields_);
}

void PartPanel::refreshMaterial() {
    updating_ = true;
    const int idx = materialBox_->findData(QString::fromStdString(app_->material.id));
    materialBox_->setCurrentIndex(idx >= 0 ? idx : materialBox_->count() - 1);
    buildMaterialFields();
    updating_ = false;
    refresh();
}

void PartPanel::refresh() {
    const auto& p = app_->part;
    empty_->setVisible(!p);
    statsBox_->setVisible(bool(p));
    faceCard_->setVisible(p && !p->brepFaces);
    while (stats_->rowCount()) stats_->removeRow(0);
    updating_ = true;
    static const char* units[4] = {"mm", "cm", "m", "in"};
    for (int i = 0; i < 4; i++)
        if (auto* b = unitsSeg_->findChildren<QToolButton*>().value(i)) b->setChecked(app_->units == units[i]);
    updating_ = false;
    if (!p) return;
    const QString u = app_->units;
    const double m3 = std::pow(app_->toMeters(), 3);
    const double mass = p->volume * m3 * app_->material.density;
    const QString matName = QString::fromStdString(app_->material.name).section(" (", 0, 0);
    auto add = [&](const QString& k, const QString& v) {
        auto* val = new QLabel(v, statsBox_);
        val->setTextInteractionFlags(Qt::TextSelectableByMouse);
        val->setToolTip(v);
        stats_->addRow(k, val);
    };
    add(tr("Name"), QString::fromStdString(p->name));
    add(tr("Source"), app_->partInfo.isEmpty() ? "–" : app_->partInfo);
    add(tr("Size"), QString("%1 × %2 × %3 %4").arg(num(p->bbox.size[0]), num(p->bbox.size[1]), num(p->bbox.size[2]), u));
    add(tr("Volume"), QString("%1 %2³").arg(num(p->volume), u));
    add(tr("Surface"), QString("%1 %2²").arg(num(p->area), u));
    add(tr("Mass"), QString("%1 g (%2)").arg(num(mass * 1000), matName));
    add(tr("Triangles"), QLocale().toString(p->nTri));
    add(tr("Faces"), QLocale().toString(p->faceCount) + (p->brepFaces ? tr(" (CAD)") : ""));
    targetSize_->setValue(std::max({p->bbox.size[0], p->bbox.size[1], p->bbox.size[2]}));
    sizeUnit_->setText(u);
}

}  // namespace ps
