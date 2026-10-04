#include "studies.hpp"
#include "psimjson.hpp"

#include <QCheckBox>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFile>
#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QObject>
#include <QPushButton>
#include <QScrollArea>
#include <QTimer>
#include <QSlider>
#include <QStyle>
#include <QVBoxLayout>
#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <numeric>
#include <optional>
#include <set>

#include "core/materials.hpp"
#include "fea/fatigue.hpp"
#include "fea/response.hpp"
#include "fea/study_runner.hpp"
#include "fea/topology.hpp"
#include "mainwindow.hpp"
#include "structuralpanel.hpp"
#include "viewport.hpp"

namespace ps {

const std::vector<StudyInfo>& studyInfos() {
    static const std::vector<StudyInfo> list = {
        {"static", QObject::tr("Linear static · bend & break"),
         QObject::tr("Stress, deflection and safety factor under steady loads; the break test shows where cracks start and run."), QObject::tr("Run bend test"),
         true, false, true, false, 8e4},
        {"nonlinear", QObject::tr("Nonlinear static · large bending, yielding"),
         QObject::tr("Large bending and metal yielding that a linear study misses. Can raise the load until the part collapses or tears, then shows the permanent bend."),
         QObject::tr("Run nonlinear"), true, false, true, false, 1e4},
        {"modal", QObject::tr("Frequency · natural vibration modes"),
         QObject::tr("Natural vibration frequencies and mode shapes. Uses the fixtures; loads are ignored. With no fixtures the part is free-floating."),
         QObject::tr("Find frequencies"), true, true, false, false, 4e4},
        {"buckling", QObject::tr("Buckling · critical load factor"),
         QObject::tr("How many times the loads can grow before a slender or compressed part suddenly buckles sideways."), QObject::tr("Run buckling"), true, false,
         true, false, 4e4},
        {"fatigue", QObject::tr("Fatigue · life under repeated loads"),
         QObject::tr("How many load cycles the part survives when the loads are applied over and over (S-N curve)."), QObject::tr("Run fatigue"), true, false, true,
         false, 8e4},
        {"drop", QObject::tr("Drop test · impact on the floor"),
         QObject::tr("The part falls onto a rigid floor, as it is oriented now (gravity −Y). Rotate it in the Part tab to change which side hits. Fixtures and loads are not used."),
         QObject::tr("Run drop test"), false, false, false, false, 4e4},
        {"dynamic", QObject::tr("Linear dynamic · vibration, shock, earthquake"),
         QObject::tr("Response to vibration, shock or earthquake shaking over time or frequency, from the natural modes (modal superposition)."),
         QObject::tr("Run dynamic"), true, false, true, true, 3e4},
        {"optimize", QObject::tr("Optimization · lighter or cheaper design"),
         QObject::tr("Remove material that does little (topology), or find the lightest / cheapest material and size that keeps the safety factor."),
         QObject::tr("Optimize"), true, false, true, false, 2.4e4},
    };
    return list;
}

namespace {

QString tr(const char* s) { return QObject::tr(s); }

QString pct(double x) { return x < 5e-4 ? QStringLiteral("0%") : num(x * 100, 2) + "%"; }
QString freqText(double f) { return f >= 1000 ? num(f / 1000) + " kHz" : num(f) + " Hz"; }
QString timeText(double t) { return t >= 1 ? num(t) + " s" : t >= 1e-3 ? num(t * 1e3) + " ms" : num(t * 1e6) + " µs"; }
QString sup(int n) {
    static const QString digits = QString::fromUtf8("⁰¹²³⁴⁵⁶⁷⁸⁹");
    QString out;
    for (QChar c : QString::number(n)) out += c == '-' ? QString::fromUtf8("⁻") : QString(digits[c.digitValue()]);
    return out;
}
QString pow10(int e) { return "10" + sup(e); }
QString cyclesText(double n) {
    if (!std::isfinite(n)) return tr("unlimited");
    if (n >= 1e9) return "> " + pow10(9);
    if (n < 1) return "< 1";
    if (n < 1e4) return QString::number(std::lround(n));
    const int e = int(std::floor(std::log10(n)));
    const double m = n / std::pow(10, e);
    return m < 1.05 ? pow10(e) : num(m, 2) + "×" + pow10(e);
}

std::vector<float> magnitudes(const std::vector<float>& u) {
    std::vector<float> out(u.size() / 3);
    for (size_t v = 0; v < out.size(); v++)
        out[v] = std::isnan(u[3 * v]) ? NAN : std::sqrt(u[3 * v] * u[3 * v] + u[3 * v + 1] * u[3 * v + 1] + u[3 * v + 2] * u[3 * v + 2]);
    return out;
}

double maxAbs3(const std::vector<float>& u) {
    double m = 0;
    for (size_t i = 0; i + 2 < u.size(); i += 3)
        if (!std::isnan(u[i])) m = std::max(m, std::sqrt(double(u[i]) * u[i] + double(u[i + 1]) * u[i + 1] + double(u[i + 2]) * u[i + 2]));
    return m;
}

struct Range {
    double lo = INFINITY, hi = -INFINITY;
    int arg = -1;
};
Range range(const std::vector<float>& values) {
    Range r;
    for (size_t i = 0; i < values.size(); i++) {
        const double v = values[i];
        if (std::isnan(v)) continue;
        r.lo = std::min(r.lo, v);
        if (v > r.hi) { r.hi = v; r.arg = int(i); }
    }
    return r;
}

double norm3(const std::array<double, 3>& a) { return std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

QLabel* heading(const QString& text) {
    auto* l = new QLabel(text);
    l->setObjectName("cardTitle");
    return l;
}

QWidget* field(const QString& label, QWidget* control) {
    auto* w = new QWidget;
    auto* v = new QVBoxLayout(w);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(3);
    auto* l = new QLabel(label, w);
    l->setObjectName("meta");
    v->addWidget(l);
    control->setParent(w);
    v->addWidget(control);
    return w;
}

QSlider* hslider(int min, int max, int value, std::function<void(int)> fn) {
    auto* s = new QSlider(Qt::Horizontal);
    s->setRange(min, max);
    s->setValue(value);
    QObject::connect(s, &QSlider::valueChanged, s, [fn](int v) { fn(v); });
    return s;
}

QComboBox* combo(const std::vector<std::pair<QString, QString>>& items, const QString& current, std::function<void(QString)> fn) {
    auto* c = new QComboBox;
    for (const auto& [id, label] : items) {
        c->addItem(label, id);
        if (id == current) c->setCurrentIndex(c->count() - 1);
    }
    QObject::connect(c, &QComboBox::activated, c, [c, fn](int i) { fn(c->itemData(i).toString()); });
    return c;
}

QCheckBox* check(const QString& label, bool on, std::function<void(bool)> fn, bool enabled = true) {
    auto* c = new QCheckBox(label);
    c->setChecked(on);
    c->setEnabled(enabled);
    QObject::connect(c, &QCheckBox::toggled, c, [fn](bool v) { fn(v); });
    return c;
}

QPushButton* button(const QString& text, std::function<void()> fn) {
    auto* b = new QPushButton(text);
    QObject::connect(b, &QPushButton::clicked, b, [fn] { fn(); });
    return b;
}

QHBoxLayout* row(std::initializer_list<QWidget*> widgets) {
    auto* h = new QHBoxLayout;
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(8);
    for (auto* w : widgets)
        if (w) h->addWidget(w);
    h->addStretch();
    return h;
}

QLabel* chartTitle(const QString& text) {
    auto* l = note(text);
    l->setStyleSheet("font-weight: 600;");
    return l;
}

// one of a list of items (modes): a plain list widget sized to its rows
QListWidget* itemList(const QStringList& rows, int current, std::function<void(int)> onPick) {
    auto* list = new QListWidget;
    list->setObjectName("items");
    for (const auto& r : rows) list->addItem(r);
    list->setCurrentRow(current);
    QObject::connect(list, &QListWidget::currentRowChanged, list, [onPick](int r) {
        if (r >= 0) onPick(r);
    });
    fitList(list, 10);
    return list;
}

std::vector<float> scaled(const std::vector<float>& v, double k) {
    std::vector<float> out(v.size());
    for (size_t i = 0; i < v.size(); i++) out[i] = float(v[i] * k);
    return out;
}

ProgressFn progressOf(JobControl& ctl) {
    return [&ctl](double f, const std::string& t) {
        ctl.progress(f, t);
        return ctl.cancelled();
    };
}

double totalForce(const Assembly& a) { return norm3(a.total); }

}  // namespace

// ---------- Study base ----------

MainWindow* Study::app() const { return panel_->app(); }
Part& Study::part() const { return *panel_->app()->part; }

bool Study::visible() const { return app()->tab() == "structural" && panel_->currentStudy() == this; }

void Study::show() {
    if (!hasResult() || !visible()) return;
    panel_->studyCard->show();
    if (!built_ || panel_->studyContentOwner != this) {
        auto* layout = static_cast<QVBoxLayout*>(panel_->studyContent->layout());
        while (QLayoutItem* it = layout->takeAt(0)) {
            if (QWidget* w = it->widget()) {
                w->hide();
                w->deleteLater();
            } else if (QLayout* l = it->layout()) {
                while (QLayoutItem* sub = l->takeAt(0)) {
                    if (sub->widget()) { sub->widget()->hide(); sub->widget()->deleteLater(); }
                    delete sub;
                }
            }
            delete it;
        }
        panel_->studyAlert->hide();
        build(layout);
        compact(panel_->studyContent);
        built_ = true;
        panel_->studyContentOwner = this;
    }
    refresh();
}

QString Study::probe(int tri, const double bary[3]) {
    if (shown_.empty() || !shownFmt_ || !app()->part) return {};
    const auto& T = part().tris;
    double s = 0;
    for (int k = 0; k < 3; k++) s += bary[k] * shown_[T[3 * tri + k]];
    return std::isnan(s) ? tr("no data") : shownFmt_(s);
}

void Study::setTitle(const QString& title) { panel_->studyTitle->setText(title.toUpper()); }
void Study::setKpis(const std::vector<KpiGrid::Kpi>& kpis) { panel_->studyKpis->setKpis(kpis); }

void Study::alert(const QString& text, bool error) {
    auto* a = panel_->studyAlert;
    a->setVisible(!text.isEmpty());
    if (text.isEmpty()) return;
    a->setText(text);
    a->setProperty("kind", error ? "error" : "info");
    a->style()->unpolish(a);
    a->style()->polish(a);
}

void Study::paint(const std::vector<float>& values, LegendSpec spec, std::function<QString(double)> probeFmt) {
    const auto& v = panel_->view;
    spec.bands = v.bands ? 12 : 0;
    spec.heat = v.heat;
    ColorSpec cs;
    cs.min = float(spec.min);
    cs.max = float(spec.max);
    cs.bands = spec.bands;
    cs.reverse = spec.reverse;
    cs.heat = spec.heat;
    app()->viewer->setScalars(&values, cs);
    shown_ = values;
    shownFmt_ = probeFmt ? probeFmt : spec.format;
    app()->legend->setSpecs({std::move(spec)});
}

void Study::marker(int vertex, const std::vector<float>* u, double scale, const QString& label) {
    Viewport* viewer = app()->viewer;
    if (vertex < 0 || !panel_->view.marker) { viewer->clearMarker(); return; }
    const auto& V = part().vertices;
    QVector3D p(V[3 * vertex], V[3 * vertex + 1], V[3 * vertex + 2]);
    if (u && !u->empty() && !std::isnan((*u)[3 * vertex])) p += QVector3D((*u)[3 * vertex], (*u)[3 * vertex + 1], (*u)[3 * vertex + 2]) * float(scale);
    if (viewer->hasMarker() && viewer->markerLabel() == label) viewer->moveMarker(p);
    else viewer->setMarker(p, label);
}

QWidget* Study::displayToggles() {
    auto* w = new QWidget;
    auto* h = new QHBoxLayout(w);
    h->setContentsMargins(0, 0, 0, 0);
    auto& v = panel_->view;
    h->addWidget(check(tr("Contour bands"), v.bands, [this](bool on) { panel_->view.bands = on; show(); }));
    h->addWidget(check(tr("Heat colours"), v.heat, [this](bool on) { panel_->view.heat = on; show(); }));
    h->addWidget(check(tr("Show loads"), v.bcs, [this](bool on) { panel_->view.bcs = on; panel_->drawOverlays(); }));
    h->addStretch();
    return w;
}

void Study::status(const QString& msg, const QString& kind) { app()->status(msg, kind); }

QString Study::engineNote(const std::string& engine, const std::string& gpuNote) const {
    QString s = tr("on the %1").arg(QString::fromStdString(engine.empty() ? "CPU" : engine));
    if (!gpuNote.empty()) s += tr(" (GPU not used: %1)").arg(QString::fromStdString(gpuNote));
    return s;
}

namespace {

// ---------------------------------------------------------------------------------------------
// Nonlinear static

class NonlinearStudy : public Study {
public:
    using Study::Study;
    QString id() const override { return "nonlinear"; }
    bool hasResult() const override { return r_ && !r_->steps.empty(); }
    void clear() override { r_.reset(); view_.playing = false; }

    std::optional<psim::Arrays> exportResult() const override {
        using namespace pj;
        if (!r_ || r_->steps.empty()) return std::nullopt;
        psim::Arrays out;
        auto stepMeta = [](const NonlinearStep& s) {
            Value e = jobj();
            e.obj["lam"] = jnum(s.lam); e.obj["D"] = jnum(s.D); e.obj["maxVM"] = jnum(s.maxVM); e.obj["maxPE"] = jnum(s.maxPE);
            e.obj["maxDisp"] = jnum(s.maxDisp); e.obj["iterations"] = jnum(s.iterations);
            return e;
        };
        auto put = [&](const std::string& prefix, const NonlinearStep& s) {
            for (auto [n, data, stride] : {std::tuple<const char*, const std::vector<float>*, uint32_t>{"u", &s.u, 3}, {"vm", &s.vm, 1}, {"pe", &s.pe, 1}}) {
                psim::Array a;
                a.name = prefix + "." + n;
                a.enc = psim::Enc::Q16;
                a.f = *data;
                a.stride = stride;
                out.list.push_back(std::move(a));
            }
        };
        Value m = jobj();
        m.obj["nVert"] = jnum(part().nVert);
        m.obj["units"] = jstr(r_->units.toStdString());
        m.obj["material"] = materialToJson(r_->material);
        m.obj["totalF"] = jnum(r_->totalF);
        m.obj["reason"] = jstr(r_->reason.toStdString());
        m.obj["engine"] = jstr("");
        m.obj["gpuNote"] = jstr("");
        m.obj["plastic"] = jbool(o_.plastic);
        Value opts = jobj();
        opts.obj["large"] = jbool(o_.large);
        opts.obj["plastic"] = jbool(o_.plastic);
        opts.obj["mode"] = jstr(r_->untilFailure ? "failure" : "applied");
        opts.obj["steps"] = jnum(o_.steps);
        m.obj["opts"] = opts;
        Value steps = jarr();
        for (size_t i = 0; i < r_->steps.size(); i++) {
            steps.arr.push_back(stepMeta(r_->steps[i]));
            put("nonlinear.step." + std::to_string(i), r_->steps[i]);
        }
        m.obj["steps"] = steps;
        if (r_->unloaded) { m.obj["unloaded"] = stepMeta(*r_->unloaded); put("nonlinear.unloaded", *r_->unloaded); }
        else m.obj["unloaded"] = Value{};
        out.meta = m;
        return out;
    }
    bool canImport() const override { return true; }
    void importResult(const json::Value& m, const psim::Arrays& arrays) override {
        using namespace pj;
        const size_t nV = size_t(part().nVert);
        if (numOr(m["nVert"], -1) != double(nV)) throw std::runtime_error("The nonlinear result in the file was computed for a different part.");
        auto load = [&](const Value& e, const std::string& prefix) {
            NonlinearStep s;
            s.lam = numOr(e["lam"], 0); s.D = numOr(e["D"], 0); s.maxVM = numOr(e["maxVM"], 0); s.maxPE = numOr(e["maxPE"], 0);
            s.maxDisp = numOr(e["maxDisp"], 0); s.iterations = int(numOr(e["iterations"], 0));
            for (auto [n, data, len] : {std::tuple<const char*, std::vector<float>*, size_t>{"u", &s.u, 3 * nV}, {"vm", &s.vm, nV}, {"pe", &s.pe, nV}}) {
                const psim::Array* a = arrays.find(prefix + "." + n);
                if (!a || a->f.size() != len) throw std::runtime_error("The nonlinear result in the file does not fit this part.");
                *data = a->f;
            }
            return s;
        };
        Result r;
        for (size_t i = 0; i < m["steps"].arr.size(); i++) r.steps.push_back(load(m["steps"].arr[i], "nonlinear.step." + std::to_string(i)));
        if (r.steps.empty()) throw std::runtime_error("The nonlinear result in the file has no steps.");
        if (m["unloaded"].type == Value::Object) { r.unloaded = load(m["unloaded"], "nonlinear.unloaded"); r.unloaded->unloaded = true; }
        r.material = m["material"].type == Value::Object ? materialFromJson(m["material"]) : app()->material;
        r.totalF = numOr(m["totalF"], 0);
        r.units = QString::fromStdString(strOr(m["units"], app()->units.toStdString()));
        r.reason = QString::fromStdString(strOr(m["reason"]));
        r.done = true;
        r.untilFailure = strOr(m["opts"]["mode"]) == "failure";
        r_ = std::move(r);
        view_.step = int(r_->steps.size()) - 1;
        view_.unloaded = false;
        view_.playing = false;
        invalidate();
    }

    bool options(QVBoxLayout* l) override {
        const Material& mat = app()->material;
        l->addWidget(heading(tr("NONLINEAR OPTIONS")));
        l->addWidget(check(tr("Large deflection (geometry updates as it bends)"), o_.large, [this](bool on) { o_.large = on; panel_->markStale(); }));
        l->addWidget(check(mat.brittle ? tr("Plasticity (off: the material is brittle)")
                                       : tr("Plasticity (yields at %1 MPa, tears at %2% strain)").arg(num(mat.yield), num(mat.elongation * 100)),
                           o_.plastic && !mat.brittle, [this](bool on) { o_.plastic = on; panel_->markStale(); }, !mat.brittle));
        l->addWidget(field(tr("Load"), segmented({tr("Applied loads"), tr("Until it fails")}, o_.failure ? 1 : 0, [this](int i) {
                               o_.failure = i == 1;
                               panel_->markStale();
                           })));
        l->addWidget(field(tr("Load steps"), numberBox(o_.steps, 2, 60, 1, 0, [this](double v) { o_.steps = int(v); panel_->markStale(); })));
        l->addWidget(note(tr("Steps adapt automatically near yielding and collapse. Nonlinear runs take longer than linear ones: start with a coarse mesh.")));
        return true;
    }

    void run() override {
        const int gen = ++runGen_;
        const NonlinearOptions opts{o_.large, o_.plastic, o_.failure, o_.steps};
        const Material mat = app()->material;
        auto started = panel_->startStudy(
            tr("Nonlinear static: ramping the load…"), true, true,
            [this, gen, opts, mat](StructuralPanel::Prepared& p, JobControl& ctl) -> std::any {
                const double totalF = totalForce(p.asm_);
                const QString units = p.units;
                ctl.post([this, gen, totalF, units, mat, opts] {
                    if (gen != runGen_) return;
                    r_ = Result{};
                    r_->totalF = totalF;
                    r_->units = units;
                    r_->material = mat;
                    r_->untilFailure = opts.untilFailure;
                    view_.step = -1;
                    view_.unloaded = false;
                    invalidate();
                });
                auto res = runNonlinear(*p.model, p.input, mat, opts, p.toMeters,
                                        [this, gen, &ctl](NonlinearStep&& s) {
                                            auto shared = std::make_shared<NonlinearStep>(std::move(s));
                                            ctl.post([this, gen, shared] {
                                                if (gen != runGen_ || !r_) return;
                                                if (shared->unloaded) {
                                                    r_->unloaded = std::move(*shared);
                                                    invalidate();
                                                    return;
                                                }
                                                const bool first = r_->steps.empty();
                                                r_->steps.push_back(std::move(*shared));
                                                view_.step = int(r_->steps.size()) - 1;
                                                if (first) panel_->setDisplay("study");
                                                show();
                                            });
                                        },
                                        progressOf(ctl));
                return res;
            },
            [this, gen](std::any& value, StructuralPanel::Prepared&) {
                if (gen != runGen_ || !r_) return;
                const auto res = std::any_cast<NonlinearRun>(value);
                r_->done = true;
                r_->reason = QString::fromStdString(res.reason);
                panel_->setDisplay("study");
                view_.step = int(r_->steps.size()) - 1;
                invalidate();
                show();
                const auto* last = r_->steps.empty() ? nullptr : &r_->steps.back();
                const QString load = last && r_->totalF > 0 ? force(last->lam * r_->totalF) : "–";
                QString why = r_->reason;
                if (res.reason == "reached") why = tr("Reached the applied loads");
                else if (res.reason == "collapse") why = tr("Collapsed at about %1: it keeps bending without taking more load").arg(load);
                else if (res.reason == "rupture") why = tr("Tore: plastic strain reached the elongation at break at %1").arg(load);
                else if (res.reason == "crack") why = tr("Cracked: tensile stress reached the strength at %1").arg(load);
                else if (res.reason == "large deformation") why = tr("Failed by gross bending at %1 (it moved more than 15% of its size)").arg(load);
                else if (res.reason == "time limit") why = tr("Stopped after 4 minutes at %1 without failing; try a coarser mesh").arg(load);
                else if (res.reason == "step limit") why = tr("Stopped after the maximum number of load steps at %1").arg(load);
                status(tr("%1 (%2 steps %3).").arg(why).arg(r_->steps.size()).arg(engineNote(res.engine, res.gpuNote)), res.reason == "reached" ? "" : "warn");
            },
            [this, gen](bool) {
                if (gen != runGen_ || !r_) return;
                r_->done = true;
                r_->reason = "stopped";
                if (!r_->steps.empty()) { invalidate(); show(); }
                else r_.reset();
            });
        if (!started) --runGen_;
    }

    void tick(double dt) override {
        if (!view_.playing || !hasResult()) return;
        view_.t += dt;
        if (view_.t < 0.3) return;
        view_.t = 0;
        if (view_.step >= int(r_->steps.size()) - 1) { view_.playing = false; show(); return; }
        view_.step++;
        show();
    }

protected:
    void build(QVBoxLayout* body) override {
        setTitle(tr("Nonlinear static results"));
        body->addWidget(chartTitle(tr("Load vs displacement (along the loads, %1)").arg(r_->units)));
        LineChart::Options co;
        co.xLabel = tr("displacement (%1)").arg(r_->units);
        co.integerX = false;
        co.formatX = [](double x) { return num(x); };
        const double F = r_->totalF;
        const QString units = r_->units;
        co.formatY = [F](double y) { return F > 1e-9 ? force(y) : num(y); };
        co.tipText = [F, units](double x, double y, int i) {
            return QString("%1 · %2 · %3 %4").arg(i ? tr("step %1").arg(i) : tr("start"), F > 1e-9 ? force(y) : num(y), num(x), units);
        };
        chart_ = new LineChart(nullptr, co);
        chart_->onPick = [this](int i) {
            view_.step = std::max(0, i - 1);
            view_.unloaded = false;
            show();
        };
        body->addWidget(chart_);
        stepLabel_ = new QLabel;
        stepLabel_->setObjectName("meta");
        body->addWidget(stepLabel_);
        slider_ = hslider(0, 0, 0, [this](int v) {
            view_.step = v;
            view_.unloaded = false;
            view_.playing = false;
            show();
        });
        body->addWidget(slider_);
        play_ = button(tr("▶ Play"), [this] { togglePlay(); });
        QCheckBox* unl = r_->unloaded ? check(tr("After unloading (spring-back)"), view_.unloaded, [this](bool on) {
            view_.unloaded = on;
            view_.playing = false;
            show();
        })
                                      : nullptr;
        body->addLayout(row({play_, unl}));
        body->addWidget(field(tr("Plot"), combo({{"vm", tr("von Mises stress")}, {"disp", tr("Displacement")}, {"pe", tr("Plastic (permanent) strain")}}, view_.plot,
                                                [this](QString v) { view_.plot = v; show(); })));
        body->addWidget(field(tr("Exaggerate shape"), hslider(0, 100, int(std::lround(std::log(view_.exaggerate) / std::log(50.0) * 100)), [this](int v) {
                                  view_.exaggerate = std::pow(50.0, v / 100.0);
                                  draw();
                              })));
        body->addWidget(displayToggles());
    }

    void refresh() override {
        const auto& r = *r_;
        const auto& s = current();
        const Material& mat = r.material;
        auto loadAt = [&](double lam) { return r.totalF > 1e-9 ? force(lam * r.totalF) : tr("%1 × loads").arg(num(lam)); };
        const auto& last = r.steps.back();
        double maxPE = 0;
        const NonlinearStep* firstYield = nullptr;
        for (const auto& st : r.steps) {
            maxPE = std::max(maxPE, st.maxPE);
            if (!firstYield && st.maxPE > 0) firstYield = &st;
        }
        QString sKind, sText;
        if (!r.done) { sKind = "warn"; sText = tr("running"); }
        else if (r.reason == "reached") { sKind = maxPE > 0 ? "warn" : "good"; sText = maxPE > 0 ? tr("yielded") : tr("elastic"); }
        else { sKind = "bad"; sText = r.reason; }
        std::vector<KpiGrid::Kpi> k = {
            {r.untilFailure ? tr("Failure load") : tr("Load reached"), loadAt(last.lam), "", sKind, sText},
            {tr("Max von Mises"), stress(s.maxVM), tr("yield %1 MPa").arg(num(mat.yield))},
            {tr("Max displacement"), num(s.maxDisp) + " " + r.units, s.unloaded ? tr("after unloading") : tr("at %1").arg(loadAt(s.lam))},
            {tr("Max plastic strain"), num(maxPE * 100) + " %", firstYield ? tr("yielding from %1").arg(loadAt(firstYield->lam)) : tr("no yielding")},
        };
        if (r.unloaded) k.push_back({tr("Permanent bend"), num(r.unloaded->maxDisp) + " " + r.units, tr("after the load is removed")});
        setKpis(k);
        std::vector<QPointF> pts = {QPointF(0, 0)};
        for (const auto& st : r.steps) pts.push_back(QPointF(st.D, st.lam * (r.totalF > 1e-9 ? r.totalF : 1)));
        chart_->setPoints(pts, view_.unloaded ? -1 : view_.step + 1);
        slider_->blockSignals(true);
        slider_->setMaximum(int(r.steps.size()) - 1);
        slider_->setValue(view_.step);
        slider_->blockSignals(false);
        stepLabel_->setText(tr("Step (%1)").arg(view_.unloaded ? tr("unloaded") : QString("%1/%2").arg(view_.step + 1).arg(r.steps.size())));
        play_->setText(view_.playing ? tr("❚❚ Pause") : tr("▶ Play"));
        draw();
    }

private:
    const NonlinearStep& current() const {
        if (view_.unloaded && r_->unloaded) return *r_->unloaded;
        return r_->steps[std::clamp(view_.step, 0, int(r_->steps.size()) - 1)];
    }

    void draw() {
        if (!hasResult() || !visible()) return;
        const auto& r = *r_;
        const auto& s = current();
        const Material& mat = r.material;
        LegendSpec spec;
        std::vector<float> values;
        if (view_.plot == "disp") {
            values = magnitudes(s.u);
            double mx = 1e-12;
            for (const auto& st : r.steps) mx = std::max(mx, st.maxDisp);
            const QString units = r.units;
            spec.title = tr("Displacement");
            spec.max = mx;
            spec.format = [units](double x) { return num(x) + " " + units; };
        } else if (view_.plot == "pe") {
            values = scaled(s.pe, 100);
            double mx = std::max(mat.elongation * 100, 1e-6);
            for (const auto& st : r.steps) mx = std::max(mx, st.maxPE * 100);
            spec.title = tr("Plastic strain");
            spec.max = mx;
            spec.format = [](double x) { return num(x) + " %"; };
            spec.markers = {{mat.elongation * 100, tr("breaks %1%").arg(num(mat.elongation * 100))}};
        } else {
            values = s.vm;
            double hi = mat.uts * 1e6;
            for (const auto& st : r.steps) hi = std::max(hi, st.maxVM);
            spec.title = tr("von Mises stress");
            spec.max = hi;
            spec.format = stressFormatter(hi);
            spec.markers = {{mat.yield * 1e6, QString("Yield %1").arg(num(mat.yield))}, {mat.uts * 1e6, QString("UTS %1").arg(num(mat.uts))}};
        }
        spec.min = 0;
        spec.sub = QString("%1 · %2 · %3")
                       .arg(QString::fromStdString(mat.name), s.unloaded ? tr("after unloading") : tr("load ×%1").arg(num(s.lam)),
                            view_.exaggerate < 1.05 ? tr("true-scale shape") : tr("shape ×%1").arg(num(view_.exaggerate)));
        paint(values, spec);
        app()->viewer->setDeformation(&s.u, view_.exaggerate);
        marker(range(values).arg, &s.u, view_.exaggerate, view_.plot == "pe" ? tr("Most permanent strain") : tr("Highest stress"));
        panel_->drawOverlays();
    }

    void togglePlay() {
        if (view_.playing) { view_.playing = false; show(); return; }
        if (!hasResult()) return;
        view_.unloaded = false;
        if (view_.step >= int(r_->steps.size()) - 1) view_.step = 0;
        view_.playing = true;
        view_.t = 0;
        show();
    }

    struct Options { bool large = true, plastic = true, failure = false; int steps = 10; } o_;
    struct View { QString plot = "vm"; int step = -1; double exaggerate = 1; bool playing = false, unloaded = false; double t = 0; } view_;
    struct Result {
        std::vector<NonlinearStep> steps;
        std::optional<NonlinearStep> unloaded;
        Material material;
        double totalF = 0;
        QString units, reason;
        bool done = false, untilFailure = false;
    };
    std::optional<Result> r_;
    LineChart* chart_ = nullptr;
    QSlider* slider_ = nullptr;
    QLabel* stepLabel_ = nullptr;
    QPushButton* play_ = nullptr;
};

// ---------------------------------------------------------------------------------------------
// Frequency (modal) and buckling share the mode list and the animated shape

class ModeStudy : public Study {
public:
    using Study::Study;

    void tick(double dt) override {
        if (!hasResult() || !view_.animate || panel_->display() != "study") return;
        view_.phase += dt * M_PI * speed_;
        const std::vector<float>* shape = modeShape();
        if (shape) app()->viewer->setDeformation(shape, std::sin(view_.phase) * view_.amp * diag_);
    }

protected:
    virtual const std::vector<float>* modeShape() const = 0;

    void addModeControls(QVBoxLayout* body, const QStringList& rows) {
        body->addWidget(itemList(rows, view_.mode, [this](int i) {
            view_.mode = i;
            show();
        }));
        body->addLayout(row({check(tr("Animate"), view_.animate, [this](bool on) { view_.animate = on; draw(); })}));
        body->addWidget(field(tr("Amplitude (display only)"), hslider(1, 30, int(std::lround(view_.amp * 100)), [this](int v) {
                                  view_.amp = v / 100.0;
                                  draw();
                              })));
    }

    void drawShape(const QString& sub) {
        const std::vector<float>* shape = modeShape();
        if (!visible()) return;
        std::vector<float> values = shape ? magnitudes(*shape) : std::vector<float>(part().nVert, 0.f);
        LegendSpec spec;
        spec.title = tr("Relative displacement");
        spec.sub = sub;
        spec.min = 0;
        spec.max = 1;
        spec.format = [](double x) { return num(x, 2); };
        paint(values, spec, [](double x) { return tr("%1 % of max").arg(num(x * 100)); });
        const double s = view_.animate ? std::sin(view_.phase) : 1;
        app()->viewer->setDeformation(shape, s * view_.amp * diag_);
        app()->viewer->clearMarker();
        panel_->drawOverlays();
    }

    virtual void draw() = 0;

    struct View { int mode = 0; double amp = 0.08, phase = 0; bool animate = true; } view_;
    double diag_ = 1, speed_ = 1.4;
};

class ModalStudy : public ModeStudy {
public:
    using ModeStudy::ModeStudy;
    QString id() const override { return "modal"; }
    bool hasResult() const override { return r_.has_value(); }
    void clear() override { r_.reset(); }

    std::optional<psim::Arrays> exportResult() const override {
        using namespace pj;
        if (!r_) return std::nullopt;
        psim::Arrays out;
        Value m = jobj();
        m.obj["nVert"] = jnum(part().nVert);
        m.obj["units"] = jstr(units_.toStdString());
        m.obj["material"] = materialToJson(app()->material);
        m.obj["diag"] = jnum(diag_);
        m.obj["free"] = jbool(r_->free);
        m.obj["converged"] = jbool(r_->converged);
        m.obj["iterations"] = jnum(r_->iterations);
        m.obj["totalMass"] = jnum(r_->totalMass);
        m.obj["engine"] = jstr(r_->engine);
        m.obj["gpuNote"] = jstr(r_->gpuNote);
        Value modes = jarr();
        for (size_t i = 0; i < r_->modes.size(); i++) {
            Value e = jobj();
            e.obj["freq"] = jnum(r_->modes[i].freq);
            e.obj["eff"] = vec({r_->modes[i].eff[0], r_->modes[i].eff[1], r_->modes[i].eff[2]});
            modes.arr.push_back(e);
            psim::Array a;
            a.name = "modal.shape." + std::to_string(i);
            a.enc = psim::Enc::Q16;
            a.f = r_->modes[i].shape;
            a.stride = 3;
            out.list.push_back(std::move(a));
        }
        m.obj["modes"] = modes;
        out.meta = m;
        return out;
    }
    bool canImport() const override { return true; }
    void importResult(const json::Value& m, const psim::Arrays& arrays) override {
        using namespace pj;
        if (numOr(m["nVert"], -1) != part().nVert) throw std::runtime_error("The frequency result in the file was computed for a different part.");
        ModalRun r;
        for (size_t i = 0; i < m["modes"].arr.size(); i++) {
            const Value& e = m["modes"].arr[i];
            const psim::Array* a = arrays.find("modal.shape." + std::to_string(i));
            if (!a || a->f.size() != 3 * size_t(part().nVert)) throw std::runtime_error("The frequency result in the file does not fit this part.");
            ModalMode mode;
            mode.freq = numOr(e["freq"], 0);
            for (int d = 0; d < 3; d++) mode.eff[d] = numOr(e["eff"][d], 0);
            mode.shape = a->f;
            r.modes.push_back(std::move(mode));
        }
        r.free = boolOr(m["free"], false);
        r.converged = boolOr(m["converged"], true);
        r.iterations = int(numOr(m["iterations"], 0));
        r.totalMass = numOr(m["totalMass"], 0);
        r.engine = strOr(m["engine"]);
        r.gpuNote = strOr(m["gpuNote"]);
        r_ = std::move(r);
        units_ = QString::fromStdString(strOr(m["units"], app()->units.toStdString()));
        diag_ = part().bbox.diag;
        view_.mode = 0;
        invalidate();
    }

    bool options(QVBoxLayout* l) override {
        l->addWidget(heading(tr("FREQUENCY OPTIONS")));
        l->addWidget(field(tr("Number of modes"), numberBox(nev_, 1, 20, 1, 0, [this](double v) { nev_ = int(v); panel_->markStale(); })));
        return true;
    }

    void run() override {
        const int gen = ++runGen_;
        const int nev = nev_;
        if (!panel_->startStudy(
                tr("Finding natural frequencies…"), false, false,
                [nev](StructuralPanel::Prepared& p, JobControl& ctl) -> std::any { return runModal(*p.model, p.input, nev, nullptr, p.toMeters, progressOf(ctl)); },
                [this, gen](std::any& value, StructuralPanel::Prepared& p) {
                    if (gen != runGen_) return;
                    r_ = std::any_cast<ModalRun>(std::move(value));
                    units_ = p.units;
                    diag_ = part().bbox.diag;
                    view_.mode = 0;
                    panel_->setDisplay("study");
                    invalidate();
                    show();
                    QStringList f;
                    for (const auto& m : r_->modes) f << freqText(m.freq);
                    status(tr("%1 natural frequencies %2%3: %4.%5")
                               .arg(r_->modes.size())
                               .arg(engineNote(r_->engine, r_->gpuNote), r_->free ? tr(" (free-floating: rigid-body modes skipped)") : QString(), f.join(", "),
                                    r_->converged ? QString() : tr(" Some modes did not fully converge.")),
                           r_->converged ? "" : "warn");
                }))
            --runGen_;
    }

protected:
    const std::vector<float>* modeShape() const override {
        return r_ && view_.mode < int(r_->modes.size()) ? &r_->modes[view_.mode].shape : nullptr;
    }

    void build(QVBoxLayout* body) override {
        setTitle(tr("Natural frequencies"));
        QStringList rows;
        for (size_t i = 0; i < r_->modes.size(); i++) {
            const auto& m = r_->modes[i];
            const int d = int(std::max_element(m.eff.begin(), m.eff.end()) - m.eff.begin());
            const double e = m.eff[d];
            rows << tr("Mode %1   %2 · %3").arg(i + 1).arg(freqText(m.freq), e > 0.01 ? tr("%1 mass %2").arg(pct(e), QString(QLatin1Char("XYZ"[d]))) : tr("little mass"));
        }
        addModeControls(body, rows);
        body->addWidget(note(tr("Mode shapes show the pattern of vibration; their size is arbitrary. Colours show where the part moves most in that mode. Effective mass tells how strongly shaking along X, Y or Z excites it.")));
        body->addWidget(displayToggles());
    }

    void refresh() override {
        const auto& r = *r_;
        double sum[3] = {0, 0, 0};
        for (const auto& m : r.modes)
            for (int d = 0; d < 3; d++) sum[d] += m.eff[d];
        setKpis({
            {tr("Lowest frequency"), r.modes.empty() ? "–" : freqText(r.modes[0].freq), r.free ? tr("free-floating part") : tr("with the fixtures")},
            {tr("Modes found"), QString::number(r.modes.size()), tr("highest %1").arg(freqText(r.modes.empty() ? 0 : r.modes.back().freq))},
            {tr("Mass captured by these modes"), QString("X %1 · Y %2 · Z %3").arg(pct(sum[0]), pct(sum[1]), pct(sum[2])),
             tr("shaking along an axis with little captured mass excites higher modes"), "", "", true},
        });
        draw();
    }

    void draw() override {
        if (!r_ || r_->modes.empty()) return;
        const auto& m = r_->modes[std::min(view_.mode, int(r_->modes.size()) - 1)];
        drawShape(tr("Mode %1 · %2").arg(view_.mode + 1).arg(freqText(m.freq)));
    }

private:
    int nev_ = 5;
    std::optional<ModalRun> r_;
    QString units_;
};

class BucklingStudy : public ModeStudy {
public:
    explicit BucklingStudy(StructuralPanel* p) : ModeStudy(p) {
        view_.animate = false;
        speed_ = 1;
    }
    QString id() const override { return "buckling"; }
    bool hasResult() const override { return r_.has_value(); }
    void clear() override { r_.reset(); }

    std::optional<psim::Arrays> exportResult() const override {
        using namespace pj;
        if (!r_) return std::nullopt;
        psim::Arrays out;
        Value m = jobj();
        m.obj["nVert"] = jnum(part().nVert);
        m.obj["units"] = jstr(app()->units.toStdString());
        m.obj["material"] = materialToJson(material_);
        m.obj["diag"] = jnum(diag_);
        m.obj["totalF"] = jnum(totalF_);
        m.obj["maxVM"] = jnum(maxVM_);
        m.obj["maxFactor"] = jnum(r_->maxFactor);
        m.obj["converged"] = jbool(r_->converged);
        m.obj["iterations"] = jnum(r_->iterations);
        m.obj["engine"] = jstr(r_->engine);
        m.obj["gpuNote"] = jstr(r_->gpuNote);
        Value f = jarr();
        for (size_t i = 0; i < r_->modes.size(); i++) {
            f.arr.push_back(jnum(r_->modes[i].factor));
            psim::Array a;
            a.name = "buckling.shape." + std::to_string(i);
            a.enc = psim::Enc::Q16;
            a.f = r_->modes[i].shape;
            a.stride = 3;
            out.list.push_back(std::move(a));
        }
        m.obj["factors"] = f;
        out.meta = m;
        return out;
    }
    bool canImport() const override { return true; }
    void importResult(const json::Value& m, const psim::Arrays& arrays) override {
        using namespace pj;
        if (numOr(m["nVert"], -1) != part().nVert) throw std::runtime_error("The buckling result in the file was computed for a different part.");
        BucklingRun r;
        for (size_t i = 0; i < m["factors"].arr.size(); i++) {
            const psim::Array* a = arrays.find("buckling.shape." + std::to_string(i));
            if (!a || a->f.size() != 3 * size_t(part().nVert)) throw std::runtime_error("The buckling result in the file does not fit this part.");
            BucklingRun::Mode mode;
            mode.factor = m["factors"].arr[i].type == Value::Number ? m["factors"].arr[i].num : INFINITY;
            mode.shape = a->f;
            r.modes.push_back(std::move(mode));
        }
        r.maxFactor = m["maxFactor"].type == Value::Number ? m["maxFactor"].num : INFINITY;
        r.converged = boolOr(m["converged"], true);
        r.iterations = int(numOr(m["iterations"], 0));
        r.engine = strOr(m["engine"]);
        r.gpuNote = strOr(m["gpuNote"]);
        r_ = std::move(r);
        material_ = m["material"].type == Value::Object ? materialFromJson(m["material"]) : app()->material;
        totalF_ = numOr(m["totalF"], 0);
        maxVM_ = numOr(m["maxVM"], 0);
        diag_ = part().bbox.diag;
        view_.mode = 0;
        invalidate();
    }

    bool options(QVBoxLayout* l) override {
        l->addWidget(heading(tr("BUCKLING OPTIONS")));
        l->addWidget(field(tr("Number of modes"), numberBox(nev_, 1, 8, 1, 0, [this](double v) { nev_ = int(v); panel_->markStale(); })));
        return true;
    }

    void run() override {
        const int gen = ++runGen_;
        const int nev = nev_;
        if (!panel_->startStudy(
                tr("Buckling analysis…"), true, true,
                [nev](StructuralPanel::Prepared& p, JobControl& ctl) -> std::any {
                    const double strength = (p.material.brittle ? p.material.uts : p.material.yield) * 1e6;
                    return runBuckling(*p.model, p.input, nev, strength, progressOf(ctl));
                },
                [this, gen](std::any& value, StructuralPanel::Prepared& p) {
                    if (gen != runGen_) return;
                    r_ = std::any_cast<BucklingRun>(std::move(value));
                    material_ = p.material;
                    totalF_ = totalForce(p.asm_);
                    diag_ = part().bbox.diag;
                    maxVM_ = std::max(0.0, range(r_->vm).hi);
                    view_.mode = 0;
                    panel_->setDisplay("study");
                    invalidate();
                    show();
                    const double bl = r_->modes.empty() ? INFINITY : r_->modes[0].factor;
                    status(std::isfinite(bl) ? tr("Lowest buckling load factor %1 %2: it buckles at about %3.").arg(num(bl), engineNote(r_->engine, r_->gpuNote), force(bl * totalF_))
                                             : tr("No buckling below %1 %2.").arg(beyond(), engineNote(r_->engine, r_->gpuNote)),
                           std::isfinite(bl) && bl < 1 ? "error" : "");
                }))
            --runGen_;
    }

protected:
    const std::vector<float>* modeShape() const override {
        if (!r_ || view_.mode >= int(r_->modes.size()) || !std::isfinite(r_->modes[view_.mode].factor)) return nullptr;
        return &r_->modes[view_.mode].shape;
    }

    // "no buckling" means none below the searched range (it yields long before)
    QString beyond() const {
        return std::isfinite(r_->maxFactor) ? tr("%1 × the loads (it yields long before that)").arg(num(r_->maxFactor)) : tr("any load (nothing is compressed)");
    }

    double yieldFactor() const {
        return maxVM_ > 0 ? (material_.brittle ? material_.uts : material_.yield) * 1e6 / maxVM_ : INFINITY;
    }

    void build(QVBoxLayout* body) override {
        setTitle(tr("Buckling"));
        QStringList rows;
        for (size_t i = 0; i < r_->modes.size(); i++)
            rows << tr("Mode %1   %2").arg(i + 1).arg(std::isfinite(r_->modes[i].factor) ? tr("factor %1").arg(num(r_->modes[i].factor))
                                                                                           : std::isfinite(r_->maxFactor) ? tr("factor > %1").arg(num(r_->maxFactor)) : tr("no buckling"));
        addModeControls(body, rows);
        const double bl = r_->modes.empty() ? INFINITY : r_->modes[0].factor;
        const double yl = yieldFactor();
        if (std::isfinite(bl) && yl < bl)
            body->addWidget(note(tr("The material yields at %1 × the loads, before the buckling load: expect it to fail by yielding (see the nonlinear study).").arg(num(yl))));
        body->addWidget(note(tr("Linear buckling assumes a perfect part. Real parts with small imperfections often buckle at 50–80% of this load, so aim for a factor of 3 or more.")));
        body->addWidget(displayToggles());
    }

    void refresh() override {
        const double bl = r_->modes.empty() ? INFINITY : r_->modes[0].factor;
        const double yl = yieldFactor();
        QString k, t;
        if (!std::isfinite(bl)) { k = "good"; t = tr("no buckling"); }
        else if (bl < 1) { k = "bad"; t = tr("buckles under the applied loads"); }
        else if (bl < 3) { k = "warn"; t = tr("low margin"); }
        else { k = "good"; t = tr("safe margin"); }
        setKpis({
            {tr("Buckling load factor"), std::isfinite(bl) ? num(bl) : std::isfinite(r_->maxFactor) ? "> " + num(r_->maxFactor) : "∞", "", k, t},
            {tr("Buckling load"), std::isfinite(bl) && totalF_ > 0 ? force(bl * totalF_) : "–", tr("loads × factor")},
            {tr("Yields first at"), std::isfinite(yl) ? "× " + num(yl) : "–", yl < bl ? tr("before it buckles") : tr("after buckling")},
        });
        draw();
    }

    void draw() override {
        if (!r_ || r_->modes.empty()) return;
        const auto& m = r_->modes[std::min(view_.mode, int(r_->modes.size()) - 1)];
        drawShape(tr("Buckling mode %1 · factor %2").arg(view_.mode + 1).arg(std::isfinite(m.factor) ? num(m.factor) : std::isfinite(r_->maxFactor) ? "> " + num(r_->maxFactor) : "∞"));
    }

private:
    int nev_ = 3;
    std::optional<BucklingRun> r_;
    Material material_;
    double totalF_ = 0, maxVM_ = 0;
};

// ---------------------------------------------------------------------------------------------
// Fatigue

class FatigueStudy : public Study {
public:
    using Study::Study;
    QString id() const override { return "fatigue"; }
    bool hasResult() const override { return r_.has_value(); }
    void clear() override { r_.reset(); }

    std::optional<psim::Arrays> exportResult() const override {
        using namespace pj;
        if (!r_) return std::nullopt;
        psim::Arrays out;
        Value m = jobj();
        m.obj["nVert"] = jnum(part().nVert);
        m.obj["units"] = jstr(app()->units.toStdString());
        m.obj["material"] = materialToJson(r_->material);
        m.obj["totalF"] = jnum(0);
        m.obj["engine"] = jstr(r_->m.engine);
        m.obj["gpuNote"] = jstr(r_->m.gpuNote);
        out.meta = m;
        for (auto [name, data] : {std::pair<const char*, const std::vector<float>*>{"vm", &r_->m.vm}, {"p1", &r_->m.p1}, {"p3", &r_->m.p3}}) {
            psim::Array a;
            a.name = std::string("fatigue.") + name;
            a.enc = psim::Enc::Q16;
            a.f = *data;
            out.list.push_back(std::move(a));
        }
        return out;
    }
    bool canImport() const override { return true; }
    void importResult(const json::Value& m, const psim::Arrays& arrays) override {
        using namespace pj;
        if (numOr(m["nVert"], -1) != part().nVert) throw std::runtime_error("The fatigue result in the file was computed for a different part.");
        Mapped mp;
        for (auto [name, data] : {std::pair<const char*, std::vector<float>*>{"vm", &mp.vm}, {"p1", &mp.p1}, {"p3", &mp.p3}}) {
            const psim::Array* a = arrays.find(std::string("fatigue.") + name);
            if (!a || a->f.size() != size_t(part().nVert)) throw std::runtime_error("The fatigue result in the file does not fit this part.");
            *data = a->f;
        }
        mp.engine = strOr(m["engine"]);
        mp.gpuNote = strOr(m["gpuNote"]);
        r_ = Result{std::move(mp), m["material"].type == Value::Object ? materialFromJson(m["material"]) : app()->material, {}};
        invalidate();
        r_->fat = fatigueField(r_->m.vm, r_->m.p1, r_->m.p3, r_->material, o_.finish.toStdString(), o_.R, o_.cycles, o_.scale);
    }

    bool options(QVBoxLayout* l) override {
        const Material& mat = app()->material;
        l->addWidget(heading(tr("FATIGUE OPTIONS")));
        rBox_ = numberBox(o_.R, -5, 0.99, 0.1, 2, [this](double v) { o_.R = v; recompute(); });
        rBox_->setEnabled(o_.loading == 2);
        l->addWidget(field(tr("Each cycle"), segmented({tr("0 → load"), tr("± load"), tr("Custom")}, o_.loading, [this](int i) {
                               o_.loading = i;
                               if (i == 0) o_.R = 0;
                               if (i == 1) o_.R = -1;
                               if (rBox_) {
                                   rBox_->blockSignals(true);
                                   rBox_->setValue(o_.R);
                                   rBox_->blockSignals(false);
                                   rBox_->setEnabled(i == 2);
                               }
                               recompute();
                           })));
        l->addWidget(field(tr("Load ratio R (min / max)"), rBox_));
        l->addWidget(field(tr("Design life (cycles)"), numberBox(o_.cycles, 1, 1e12, 1e5, 0, [this](double v) { o_.cycles = std::max(1.0, v); recompute(); })));
        l->addWidget(field(tr("Load scale"), numberBox(o_.scale, 0, 1e6, 0.1, 2, [this](double v) { o_.scale = v; recompute(); })));
        if (mat.fatigue.metal) {
            std::vector<std::pair<QString, QString>> items;
            for (const auto& [id, label] : fatigueFinishes()) items.push_back({QString::fromStdString(id), QString::fromStdString(label)});
            l->addWidget(field(tr("Surface finish"), combo(items, o_.finish, [this](QString v) { o_.finish = v; recompute(); })));
        }
        const double Se = mat.fatigue.Se > 0 ? mat.fatigue.Se : 0.4 * mat.uts;
        l->addWidget(note(tr("%1: fatigue strength ≈ %2 MPa at %3 cycles (polished)%4. Edit it under Material (Custom) if you have datasheet values.")
                              .arg(QString::fromStdString(mat.name), num(Se), cyclesText(mat.fatigue.Ne), mat.fatigue.endurance ? tr(", with an endurance limit") : QString())));
        return true;
    }

    void run() override {
        const int gen = ++runGen_;
        if (!panel_->startStudy(
                tr("Fatigue: solving the stresses…"), true, true,
                [](StructuralPanel::Prepared& p, JobControl& ctl) -> std::any {
                    auto res = solveStatic(p.input, progressOf(ctl));
                    const auto W = p.model->vertexWeights(res.activeNode);
                    Mapped m;
                    m.vm = interpolate(W, res.nodeVM.data());
                    m.p1 = interpolate(W, res.nodeP1.data());
                    m.p3 = interpolate(W, res.nodeP3.data());
                    m.engine = res.engine;
                    m.gpuNote = res.gpuNote;
                    return m;
                },
                [this, gen](std::any& value, StructuralPanel::Prepared& p) {
                    if (gen != runGen_) return;
                    r_ = Result{std::any_cast<Mapped>(std::move(value)), p.material, {}};
                    panel_->setDisplay("study");
                    invalidate();
                    recompute();
                    // an all-blue "unlimited life" plot says little: show the margin instead
                    if (!std::isfinite(r_->fat.minLife) && view_ == "life") { view_ = "fos"; invalidate(); show(); }
                    const auto& f = r_->fat;
                    status(tr("Fatigue %1: shortest life %2 cycles; safety factor %3 for %4 cycles.")
                               .arg(engineNote(r_->m.engine, r_->m.gpuNote), cyclesText(f.minLife), num(f.minFos), cyclesText(o_.cycles)),
                           f.minLife < o_.cycles ? "error" : "");
                }))
            --runGen_;
    }

protected:
    void build(QVBoxLayout* body) override {
        setTitle(tr("Fatigue"));
        body->addWidget(chartTitle(tr("S-N curve (stress amplitude vs cycles to failure)")));
        LineChart::Options co;
        co.xLabel = tr("cycles to failure");
        co.integerX = true;
        co.formatX = [](double x) { return pow10(int(std::lround(x))); };
        co.formatY = [](double y) { return num(y); };
        co.tipText = [](double x, double y, int) { return tr("%1 cycles · %2 MPa").arg(cyclesText(std::pow(10, x)), num(y)); };
        chart_ = new LineChart(nullptr, co);
        body->addWidget(chart_);
        body->addWidget(note(tr("The marked point is the most damaged spot (equivalent fully-reversed amplitude, Goodman-corrected).")));
        body->addWidget(field(tr("Plot"), combo({{"life", tr("Life (cycles)")}, {"damage", tr("Damage at design life")}, {"fos", tr("Safety factor")}}, view_,
                                                [this](QString v) { view_ = v; show(); })));
        body->addWidget(displayToggles());
    }

    void refresh() override {
        const auto& f = r_->fat;
        const bool ok = f.minLife >= o_.cycles;
        setKpis({
            {tr("Shortest life"), std::isfinite(f.minLife) ? cyclesText(f.minLife) : tr("Unlimited"), tr("cycles"), ok ? "good" : "bad",
             ok ? tr("meets design life") : tr("fails before design life")},
            {tr("Damage at design life"), num(o_.cycles / f.minLife), tr("≥ 1 means failure")},
            {tr("Safety factor"), num(f.minFos), tr("for %1 cycles").arg(cyclesText(o_.cycles))},
            {tr("Fatigue strength"), num(f.strengthAtLife) + " MPa",
             tr("at %1 cycles%2").arg(cyclesText(o_.cycles), f.curve.ka < 1 ? tr(" · finish ×%1").arg(num(f.curve.ka)) : QString())},
        });
        // S-N curve (x = log10 cycles) with the worst point's equivalent amplitude
        std::vector<QPointF> pts;
        for (int i = 0; i <= 90; i++) pts.push_back(QPointF(i / 10.0, strengthAt(f.curve, std::pow(10, i / 10.0))));
        int cur = -1;
        const int v = f.worst;
        const auto& m = r_->m;
        if (v >= 0) {
            auto sgn = [](double x) { return x > 0 ? 1.0 : x < 0 ? -1.0 : 1.0; };
            const double sign = std::abs(m.p1[v]) >= std::abs(m.p3[v]) ? sgn(m.p1[v]) : sgn(m.p3[v]);
            const double smax = sign * m.vm[v] * o_.scale / 1e6;
            const double sa = std::abs(smax) * (1 - o_.R) / 2, sm = smax * (1 + o_.R) / 2;
            const double sar = goodman(sa, sm, f.curve.uts);
            const double N = cyclesToFailure(f.curve, sar);
            if (std::isfinite(N)) {
                const QPointF p(std::log10(std::max(1.0, N)), sar);
                pts.push_back(p);
                std::sort(pts.begin(), pts.end(), [](const QPointF& a, const QPointF& b) { return a.x() < b.x(); });
                for (size_t i = 0; i < pts.size(); i++)
                    if (pts[i] == p) cur = int(i);
            }
        }
        chart_->setPoints(pts, cur);
        draw();
    }

private:
    void recompute() {
        if (!r_) return;
        r_->fat = fatigueField(r_->m.vm, r_->m.p1, r_->m.p3, r_->material, o_.finish.toStdString(), o_.R, o_.cycles, o_.scale);
        show();
    }

    void draw() {
        if (!visible()) return;
        const auto& f = r_->fat;
        std::vector<float> values;
        LegendSpec spec;
        spec.min = 0;
        std::function<QString(double)> probe;
        if (view_ == "damage") {
            values = f.damage;
            spec.title = tr("Damage at design life");
            spec.max = std::max(1.0, std::min(10.0, range(values).hi));
            spec.format = [](double x) { return num(x, 2); };
            spec.markers = {{1, tr("fails")}};
        } else if (view_ == "fos") {
            const double mx = std::min(10.0, std::max(2.0, std::ceil(f.minFos * 3)));
            values.resize(f.fos.size());
            for (size_t i = 0; i < values.size(); i++) values[i] = std::isnan(f.fos[i]) ? NAN : float(std::min(double(f.fos[i]), mx));
            spec.title = tr("Fatigue safety factor");
            spec.max = mx;
            spec.reverse = true;
            spec.format = [](double x) { return num(x, 2); };
            spec.markers = {{1, "FOS = 1"}};
        } else {
            // log10 life, capped at 1e9 (unlimited)
            values.resize(f.life.size());
            for (size_t i = 0; i < values.size(); i++) values[i] = std::isnan(f.life[i]) ? NAN : float(std::min(9.0, std::log10(std::max(1.0, double(f.life[i])))));
            spec.title = tr("Life (cycles)");
            spec.max = 9;
            spec.reverse = true;
            spec.format = [](double x) { return std::abs(x - std::round(x)) < 0.01 ? pow10(int(std::lround(x))) : QString("10^%1").arg(num(x, 2)); };
            spec.markers = {{std::log10(o_.cycles), tr("design life")}};
            probe = [](double x) { return tr("%1 cycles").arg(cyclesText(std::pow(10, x))); };
        }
        spec.sub = QString("%1 · R = %2 · ×%3 loads").arg(QString::fromStdString(r_->material.name), num(o_.R), num(o_.scale));
        paint(values, spec, probe);
        app()->viewer->setDeformation(nullptr);
        marker(f.worst, nullptr, 0, tr("Fatigue crack starts here"));
        panel_->drawOverlays();
    }

    struct Mapped {
        std::vector<float> vm, p1, p3;
        std::string engine, gpuNote;
    };
    struct Result {
        Mapped m;
        Material material;
        FatigueField fat;
    };
    struct Options { int loading = 0; double R = 0, cycles = 1e6, scale = 1; QString finish = "machined"; } o_;
    QString view_ = "life";
    std::optional<Result> r_;
    QPointer<QDoubleSpinBox> rBox_;
    LineChart* chart_ = nullptr;
};

// ---------------------------------------------------------------------------------------------
// Drop test

class DropStudy : public Study {
public:
    using Study::Study;
    QString id() const override { return "drop"; }
    bool hasResult() const override { return r_.has_value(); }
    void clear() override { r_.reset(); view_.playing = false; }

    bool options(QVBoxLayout* l) override {
        l->addWidget(heading(tr("DROP OPTIONS")));
        auto* speed = note("");
        auto update = [speed, this] { speed->setText(tr("Impact speed %1 m/s onto a rigid, frictionless floor.").arg(num(std::sqrt(2 * 9.81 * height_)))); };
        l->addWidget(field(tr("Drop height (m)"), numberBox(height_, 0.001, 1000, 0.1, 3, [this, update](double v) {
                               height_ = std::max(0.001, v);
                               update();
                               panel_->markStale();
                           })));
        update();
        l->addWidget(speed);
        return true;
    }

    void run() override {
        const int gen = ++runGen_;
        const double height = height_;
        if (!panel_->startStudy(
                tr("Drop test…"), false, false,
                [height](StructuralPanel::Prepared& p, JobControl& ctl) -> std::any {
                    Out o;
                    o.run = runDrop(*p.model, p.input, height, p.toMeters, [&o](DropFrame&& f) { o.frames.push_back(std::move(f)); }, progressOf(ctl));
                    return o;
                },
                [this, gen, height](std::any& value, StructuralPanel::Prepared& p) {
                    if (gen != runGen_) return;
                    auto o = std::any_cast<Out>(std::move(value));
                    Result r{std::move(o.run), std::move(o.frames), p.material, p.units, height};
                    const auto rg = range(r.run.vmMax);
                    r.peak = std::max(0.0, rg.hi);
                    r.peakAt = rg.arg;
                    double maxU = 0;
                    for (const auto& fr : r.frames) maxU = std::max(maxU, maxAbs3(fr.u));
                    r.autoScale = maxU > 0 ? std::min(2000.0, std::max(1.0, 0.05 * part().bbox.diag / maxU)) : 1;
                    view_.frame = int(r.frames.size()) - 1;
                    if (r.peakAt >= 0)
                        for (size_t i = 0; i < r.frames.size(); i++)
                            if (r.frames[i].t >= r.run.tPeak[r.peakAt]) { view_.frame = int(i); break; }
                    view_.frame = std::max(0, view_.frame);
                    view_.exaggerate = r.autoScale;
                    view_.plot = "peak";
                    r_ = std::move(r);
                    panel_->setDisplay("study");
                    invalidate();
                    show();
                    const Material& mat = r_->material;
                    const double hi = r_->peak;
                    const QString verdict = hi >= mat.uts * 1e6 ? tr("it breaks") : hi >= mat.yield * 1e6 ? (mat.brittle ? tr("it cracks") : tr("it bends permanently")) : tr("it survives");
                    status(tr("Drop from %1 m %2: peak stress %3 - %4. %5 time steps of %6.")
                               .arg(num(height), engineNote(r_->run.engine, r_->run.gpuNote), stress(hi), verdict)
                               .arg(r_->run.steps)
                               .arg(timeText(r_->run.dt)),
                           hi >= mat.yield * 1e6 ? "error" : "");
                }))
            --runGen_;
    }

    void tick(double dt) override {
        if (!view_.playing || !r_) return;
        view_.t += dt;
        if (view_.t < 0.08) return;
        view_.t = 0;
        if (view_.frame >= int(r_->frames.size()) - 1) { view_.playing = false; show(); return; }
        view_.frame++;
        show();
    }

protected:
    void build(QVBoxLayout* body) override {
        setTitle(tr("Drop test"));
        body->addWidget(chartTitle(tr("Floor force during the impact")));
        LineChart::Options co;
        co.xLabel = tr("time (ms)");
        co.integerX = false;
        co.formatX = [](double x) { return num(x); };
        co.formatY = [](double y) { return force(y); };
        co.tipText = [](double x, double y, int) { return QString("%1 ms · %2").arg(num(x), force(y)); };
        chart_ = new LineChart(nullptr, co);
        chart_->onPick = [this](int i) {
            const double t = r_->run.times[i];
            view_.frame = 0;
            for (size_t k = 0; k < r_->frames.size(); k++)
                if (r_->frames[k].t >= t) { view_.frame = int(k); break; }
            view_.plot = "frame";
            view_.playing = false;
            show();
        };
        body->addWidget(chart_);
        plot_ = combo({{"peak", tr("Peak stress over the whole impact")}, {"frame", tr("Stress at this moment")}}, view_.plot, [this](QString v) {
            view_.plot = v;
            show();
        });
        body->addWidget(field(tr("Plot"), plot_));
        moment_ = new QLabel;
        moment_->setObjectName("meta");
        body->addWidget(moment_);
        slider_ = hslider(0, std::max(0, int(r_->frames.size()) - 1), view_.frame, [this](int v) {
            view_.frame = v;
            view_.plot = "frame";
            view_.playing = false;
            show();
        });
        body->addWidget(slider_);
        play_ = button(tr("▶ Play impact"), [this] {
            view_.playing = !view_.playing;
            if (view_.playing) {
                if (view_.frame >= int(r_->frames.size()) - 1) view_.frame = 0;
                view_.plot = "frame";
                view_.t = 0;
            }
            show();
        });
        body->addLayout(row({play_}));
        const double top = std::max(2.0, 2 * r_->autoScale);
        body->addWidget(field(tr("Exaggerate shape"), hslider(0, 100, int(std::lround(std::log(std::max(1.0, view_.exaggerate)) / std::log(top) * 100)), [this, top](int v) {
                                  view_.exaggerate = std::pow(top, v / 100.0);
                                  draw();
                              })));
        body->addWidget(note(tr("Stress is recovered at voxel corners like the static study; sharp corners that hit the floor show local peaks. Linear-elastic: beyond yield it shows where it would dent or crack, not how far.")));
        body->addWidget(displayToggles());
    }

    void refresh() override {
        const auto& r = *r_;
        const Material& mat = r.material;
        QString k, t;
        if (r.peak >= mat.uts * 1e6) { k = "bad"; t = tr("breaks"); }
        else if (r.peak >= mat.yield * 1e6) { k = "bad"; t = mat.brittle ? tr("cracks") : tr("dents / bends"); }
        else { k = "good"; t = tr("survives"); }
        setKpis({
            {tr("Peak stress"), stress(r.peak), tr("yield %1 · UTS %2 MPa").arg(num(mat.yield), num(mat.uts)), k, t},
            {tr("Peak impact force"), force(r.run.peakForce), tr("%1 g deceleration").arg(num(r.run.peakForce / (r.run.mass * 9.81)))},
            {tr("Contact time"), timeText(r.run.contactTime), r.run.rebounded ? tr("bounced off the floor") : tr("still in contact at the end")},
            {tr("Impact speed"), num(r.run.speed) + " m/s", tr("mass %1 g").arg(num(r.run.mass * 1000))},
        });
        std::vector<QPointF> pts;
        for (size_t i = 0; i < r.run.times.size(); i++) pts.push_back(QPointF(r.run.times[i] * 1e3, r.run.forces[i]));
        int cur = -1;
        if (view_.frame < int(r.frames.size())) {
            const double t0 = r.frames[view_.frame].t * 1e3;
            for (size_t i = 0; i < pts.size(); i++)
                if (pts[i].x() >= t0) { cur = int(i); break; }
        }
        chart_->setPoints(pts, cur);
        for (int i = 0; i < plot_->count(); i++)
            if (plot_->itemData(i).toString() == view_.plot) plot_->setCurrentIndex(i);
        slider_->blockSignals(true);
        slider_->setValue(view_.frame);
        slider_->blockSignals(false);
        moment_->setText(tr("Moment (%1)").arg(view_.frame < int(r.frames.size()) ? timeText(r.frames[view_.frame].t) : "–"));
        play_->setText(view_.playing ? tr("❚❚ Pause") : tr("▶ Play impact"));
        draw();
    }

private:
    void draw() {
        if (!r_ || !visible()) return;
        const auto& r = *r_;
        const Material& mat = r.material;
        const DropFrame* fr = view_.frame < int(r.frames.size()) ? &r.frames[view_.frame] : nullptr;
        const double hi = std::max(r.peak, mat.yield * 1e6);
        const bool frame = view_.plot == "frame" && fr;
        const auto& values = frame ? fr->vm : r.run.vmMax;
        LegendSpec spec;
        spec.min = 0;
        spec.max = hi;
        spec.title = frame ? tr("von Mises stress") : tr("Peak von Mises (whole impact)");
        spec.format = stressFormatter(hi);
        spec.sub = tr("%1 · drop %2 m%3 · shape ×%4")
                       .arg(QString::fromStdString(mat.name), num(r.height), frame ? tr(" · t = %1").arg(timeText(fr->t)) : QString(), num(view_.exaggerate));
        spec.markers = {{mat.yield * 1e6, QString("Yield %1").arg(num(mat.yield))}, {mat.uts * 1e6, QString("UTS %1").arg(num(mat.uts))}};
        paint(values, spec);
        app()->viewer->setDeformation(fr ? &fr->u : nullptr, view_.exaggerate);
        marker(frame ? range(values).arg : r.peakAt, fr ? &fr->u : nullptr, view_.exaggerate, tr("Highest impact stress"));
        panel_->drawOverlays();
    }

    struct Out {
        DropRun run;
        std::vector<DropFrame> frames;
    };
    struct Result {
        DropRun run;
        std::vector<DropFrame> frames;
        Material material;
        QString units;
        double height = 1, peak = 0, autoScale = 1;
        int peakAt = -1;
    };
    double height_ = 1;
    struct View { QString plot = "peak"; int frame = 0; bool playing = false; double t = 0, exaggerate = 1; } view_;
    std::optional<Result> r_;
    LineChart* chart_ = nullptr;
    QComboBox* plot_ = nullptr;
    QSlider* slider_ = nullptr;
    QLabel* moment_ = nullptr;
    QPushButton* play_ = nullptr;
};

// ---------------------------------------------------------------------------------------------
// Linear dynamic

class DynamicStudy : public Study {
public:
    using Study::Study;
    QString id() const override { return "dynamic"; }
    bool hasResult() const override { return r_.has_value(); }
    void clear() override { r_.reset(); view_.playing = false; }

    bool options(QVBoxLayout* l) override {
        auto& o = o_;
        const bool base = o.base;
        l->addWidget(heading(tr("DYNAMIC OPTIONS")));
        l->addWidget(field(tr("Excitation"), segmented({tr("The loads"), tr("Base shaking")}, base ? 1 : 0, [this](int i) {
                               o_.base = i == 1;
                               if (o_.base && o_.type == "harmonic") o_.amp = 1;
                               QMetaObject::invokeMethod(panel_, [p = panel_] { p->renderStudyOptions(); }, Qt::QueuedConnection);
                               panel_->markStale();
                           })));
        if (base)
            l->addWidget(field(tr("Shaking direction"), segmented({"X", "Y", "Z"}, o.dir, [this](int i) { o_.dir = i; panel_->markStale(); })));
        l->addWidget(field(tr("Type"), combo({{"harmonic", tr("Frequency sweep (steady vibration)")}, {"shock", tr("Shock pulse (half-sine)")}, {"sine", tr("Sine burst")},
                                              {"quake", tr("Earthquake (synthetic)")}},
                                             o.type, [this](QString v) {
                                                 o_.type = v;
                                                 QMetaObject::invokeMethod(panel_, [p = panel_] { p->renderStudyOptions(); }, Qt::QueuedConnection);
                                                 recompute();
                                             })));
        l->addWidget(field(base ? tr("Peak acceleration (g)") : tr("Load multiplier"), numberBox(o.amp, -1e6, 1e6, 0.1, 3, [this](double v) {
                               o_.amp = v != 0 ? v : 1;
                               recompute();
                           })));
        if (o.type == "shock")
            l->addWidget(field(tr("Pulse duration (ms)"), numberBox(o.pulse, 0.01, 1e5, 0.5, 2, [this](double v) { o_.pulse = std::max(0.01, v); recompute(); })));
        if (o.type == "sine") {
            l->addWidget(field(tr("Frequency (Hz)"), numberBox(o.sineF, 0.01, 1e6, 1, 2, [this](double v) { o_.sineF = std::max(0.01, v); recompute(); })));
            l->addWidget(field(tr("Duration (s)"), numberBox(o.duration, 1e-4, 1e4, 0.05, 4, [this](double v) { o_.duration = std::max(1e-4, v); recompute(); })));
        }
        if (o.type == "quake")
            l->addWidget(field(tr("Duration (s)"), numberBox(o.quake, 0.5, 600, 1, 1, [this](double v) { o_.quake = std::max(0.5, v); recompute(); })));
        l->addWidget(field(tr("Damping (% of critical)"), numberBox(o.zeta, 0.01, 50, 0.5, 2, [this](double v) { o_.zeta = v; recompute(); })));
        l->addWidget(field(tr("Modes used"), numberBox(o.nev, 1, 20, 1, 0, [this](double v) { o_.nev = int(v); panel_->markStale(); })));
        l->addWidget(note(tr("Metals ≈ 1–2 % damping, bolted assemblies 3–5 %, plastics and rubber mounts 5–10 %.")));
        return true;
    }

    void run() override {
        const int gen = ++runGen_;
        const int nev = o_.nev;
        DynamicLoad dyn;
        dyn.base = o_.base;
        dyn.dir = {0, 0, 0};
        dyn.dir[o_.dir] = 1;
        if (!panel_->startStudy(
                tr("Linear dynamic: natural modes…"), true, !o_.base,
                [nev, dyn](StructuralPanel::Prepared& p, JobControl& ctl) -> std::any { return runModal(*p.model, p.input, nev, &dyn, p.toMeters, progressOf(ctl)); },
                [this, gen](std::any& value, StructuralPanel::Prepared& p) {
                    if (gen != runGen_) return;
                    auto res = std::any_cast<ModalRun>(std::move(value));
                    if (res.free || !res.hasBasis) { status(tr("Linear dynamic needs fixtures (what the part is mounted on)."), "error"); return; }
                    if (res.modes.empty()) { status(tr("No vibration modes were found."), "error"); return; }
                    Result r;
                    r.run = std::move(res);
                    r.units = p.units;
                    r.material = p.material;
                    r.diag = part().bbox.diag;
                    // sample vertices for curves: the most stressed / displaced in the static and modal fields
                    const auto& b = r.run.basis;
                    const int nV = part().nVert;
                    std::set<int> pick;
                    auto top = [&](const std::vector<float>& vals, int k) {
                        std::vector<int> idx;
                        for (int i = 0; i < nV; i++)
                            if (!std::isnan(vals[i])) idx.push_back(i);
                        const int K = std::min(k, int(idx.size()));
                        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int c) { return vals[a] > vals[c]; });
                        for (int i = 0; i < K; i++) pick.insert(idx[i]);
                    };
                    auto vmOf = [nV](const std::vector<float>& S) {
                        std::vector<float> out(nV);
                        for (int v = 0; v < nV; v++) {
                            double s[6];
                            for (int c = 0; c < 6; c++) s[c] = S[6 * v + c];
                            out[v] = std::isnan(s[0]) ? NAN : float(vonMises(s));
                        }
                        return out;
                    };
                    top(vmOf(b.staticS), 150);
                    top(magnitudes(b.staticU), 30);
                    for (const auto& S : b.modeS) top(vmOf(S), 60);
                    for (const auto& U : b.modeU) top(magnitudes(U), 15);
                    r.samples.assign(pick.begin(), pick.end());
                    r_ = std::move(r);
                    view_.freq = 0;
                    panel_->setDisplay("study");
                    recompute();
                    QStringList f;
                    for (size_t i = 0; i < std::min<size_t>(4, r_->run.modes.size()); i++) f << freqText(r_->run.modes[i].freq);
                    status(tr("Linear dynamic %1 with %2 modes (%3%4).")
                               .arg(engineNote(r_->run.engine, r_->run.gpuNote))
                               .arg(r_->run.basis.omegas.size())
                               .arg(f.join(", "), r_->run.modes.size() > 4 ? ", …" : ""));
                }))
            --runGen_;
    }

    void tick(double dt) override {
        if (!r_ || panel_->display() != "study") return;
        if (o_.type == "harmonic" && view_.animate && r_->hasField) {
            view_.phase += dt * M_PI * 1.2;
            const auto u = harmonicShape(r_->run.basis, 2 * M_PI * r_->fieldF, o_.zeta / 100, view_.phase);
            app()->viewer->setDeformation(&u, drawScale_);
        } else if (o_.type != "harmonic" && view_.playing && r_->hasHist) {
            const int n = int(r_->hist.times.size());
            view_.index = std::min(n - 1, view_.index + std::max(1, int(std::lround(n * dt / 6))));
            if (view_.index >= n - 1) view_.playing = false;
            show();
        }
    }

protected:
    void build(QVBoxLayout* body) override {
        const QString unitIn = o_.base ? tr("%1 g base shaking").arg(num(o_.amp)) : tr("%1 × the loads").arg(num(o_.amp));
        LineChart::Options co;
        co.integerX = false;
        co.formatX = [](double x) { return num(x); };
        co.formatY = [](double y) { return stress(y); };
        if (o_.type == "harmonic") {
            setTitle(tr("Frequency response"));
            body->addWidget(chartTitle(tr("Peak stress vs frequency (%1)").arg(unitIn)));
            co.xLabel = tr("frequency (Hz)");
            co.logY = true;
            co.tipText = [](double x, double y, int) { return QString("%1 · %2").arg(freqText(x), stress(y)); };
            chart_ = new LineChart(nullptr, co);
            chart_->onPick = [this](int i) {
                view_.freq = r_->curve[i].f;
                fieldAtFreq();
                show();
            };
            body->addWidget(chart_);
            body->addWidget(note(tr("Click the curve to see the part vibrating at that frequency.")));
            freqBox_ = numberBox(view_.freq, 0.001, 1e7, 1, 3, [this](double v) {
                if (v > 0) { view_.freq = v; fieldAtFreq(); show(); }
            });
            body->addWidget(field(tr("Frequency (Hz)"), freqBox_));
            body->addWidget(field(tr("Plot"), combo({{"vm", tr("Stress amplitude (peak von Mises)")}, {"disp", tr("Vibration amplitude")}}, view_.plot == "disp" ? "disp" : "vm",
                                                    [this](QString v) { view_.plot = v; show(); })));
            body->addLayout(row({check(tr("Animate vibration"), view_.animate, [this](bool on) { view_.animate = on; draw(); })}));
        } else {
            setTitle(tr("Time response"));
            body->addWidget(chartTitle(tr("Peak stress over time (%1)").arg(unitIn)));
            co.xLabel = tr("time (ms)");
            co.tipText = [](double x, double y, int) { return QString("%1 ms · %2").arg(num(x), stress(y)); };
            chart_ = new LineChart(nullptr, co);
            chart_->onPick = [this](int i) {
                view_.index = r_->tcurve[i].i;
                view_.plot = "now";
                view_.playing = false;
                show();
            };
            body->addWidget(chart_);
            timeLabel_ = new QLabel;
            timeLabel_->setObjectName("meta");
            body->addWidget(timeLabel_);
            slider_ = hslider(0, int(r_->hist.times.size()) - 1, view_.index, [this](int v) {
                view_.index = v;
                view_.plot = "now";
                view_.playing = false;
                show();
            });
            body->addWidget(slider_);
            play_ = button(tr("▶ Play"), [this] {
                view_.playing = !view_.playing;
                if (view_.playing && view_.index >= int(r_->hist.times.size()) - 1) view_.index = 0;
                view_.plot = "now";
                show();
            });
            body->addLayout(row({play_}));
            plotBox_ = combo({{"envelope", tr("Peak stress over the whole event")}, {"now", tr("Stress at this moment")}}, view_.plot == "now" ? "now" : "envelope",
                             [this](QString v) { view_.plot = v; show(); });
            body->addWidget(field(tr("Plot"), plotBox_));
        }
        body->addWidget(displayToggles());
    }

    void refresh() override {
        const auto& r = *r_;
        const Material& mat = r.material;
        if (o_.type == "harmonic") {
            const auto worst = *std::max_element(r.curve.begin(), r.curve.end(), [](const auto& a, const auto& b) { return a.vm < b.vm; });
            const double mx = std::max(0.0, range(r.fieldVM).hi), du = std::max(0.0, range(r.fieldDisp).hi);
            setKpis({
                {tr("First resonance"), freqText(r.run.modes[0].freq), tr("%1 modes").arg(r.run.modes.size())},
                {tr("Worst resonance"), freqText(worst.f), tr("%1 peak stress").arg(stress(worst.vm)), worst.vm >= mat.yield * 1e6 ? "bad" : "",
                 worst.vm >= mat.yield * 1e6 ? tr("yields") : QString()},
                {tr("At %1").arg(freqText(r.fieldF)), stress(mx), tr("%1 %2 vibration amplitude").arg(num(du), r.units)},
                {tr("Amplification"), r.staticVM > 0 ? "× " + num(worst.vm / r.staticVM) : "–", tr("worst resonance vs steady load")},
            });
            std::vector<QPointF> pts;
            int cur = 0;
            for (size_t i = 0; i < r.curve.size(); i++) {
                pts.push_back(QPointF(r.curve[i].f, r.curve[i].vm));
                if (std::abs(r.curve[i].f - r.fieldF) < std::abs(r.curve[cur].f - r.fieldF)) cur = int(i);
            }
            chart_->setPoints(pts, cur);
            if (freqBox_) {
                freqBox_->blockSignals(true);
                freqBox_->setValue(r.fieldF);
                freqBox_->blockSignals(false);
            }
        } else {
            const auto worst = *std::max_element(r.tcurve.begin(), r.tcurve.end(), [](const auto& a, const auto& b) { return a.vm < b.vm; });
            const double env = std::max(0.0, range(r.envelope).hi);
            const auto u = shapeAt(r.run.basis, r.hist, view_.index);
            const double k = std::abs(scale());
            QString sk, st;
            if (env >= mat.uts * 1e6) { sk = "bad"; st = tr("breaks"); }
            else if (env >= mat.yield * 1e6) { sk = "bad"; st = tr("yields"); }
            else { sk = "good"; st = tr("elastic"); }
            setKpis({
                {tr("Peak stress"), stress(env), tr("at %1").arg(timeText(worst.t)), sk, st},
                {tr("Displacement now"), num(maxAbs3(u) * k) + " " + r.units, tr("t = %1").arg(timeText(r.hist.times[view_.index]))},
                {tr("Lowest mode"), freqText(r.run.modes[0].freq), tr("%1 modes, %2% damping").arg(r.run.modes.size()).arg(num(o_.zeta))},
            });
            std::vector<QPointF> pts;
            int cur = 0;
            for (size_t i = 0; i < r.tcurve.size(); i++) {
                pts.push_back(QPointF(r.tcurve[i].t * 1e3, r.tcurve[i].vm));
                if (std::abs(r.tcurve[i].i - view_.index) < std::abs(r.tcurve[cur].i - view_.index)) cur = int(i);
            }
            chart_->setPoints(pts, cur);
            slider_->blockSignals(true);
            slider_->setValue(view_.index);
            slider_->blockSignals(false);
            timeLabel_->setText(tr("Time (%1)").arg(timeText(r.hist.times[view_.index])));
            play_->setText(view_.playing ? tr("❚❚ Pause") : tr("▶ Play"));
            for (int i = 0; i < plotBox_->count(); i++)
                if (plotBox_->itemData(i).toString() == (view_.plot == "now" ? "now" : "envelope")) plotBox_->setCurrentIndex(i);
        }
        draw();
    }

private:
    double scale() const { return o_.base ? o_.amp * 9.81 : o_.amp; }  // pattern per unit load or per 1 m/s^2

    void recompute() {
        if (!r_) return;
        auto& r = *r_;
        const auto& b = r.run.basis;
        const double zeta = o_.zeta / 100, k = std::abs(scale());
        if (o_.type == "harmonic") {
            double fMaxMode = 0;
            for (const auto& m : r.run.modes) fMaxMode = std::max(fMaxMode, m.freq);
            const double fMax = fMaxMode * 1.3;
            std::set<double> freqs;
            for (int i = 1; i <= 300; i++) freqs.insert(fMax * i / 300);
            for (const auto& m : r.run.modes)
                for (int d = -8; d <= 8; d++) freqs.insert(std::max(1e-3, m.freq * (1 + d * zeta * 0.5)));
            r.curve.clear();
            for (double f : freqs) {
                const auto fld = harmonicField(b, 2 * M_PI * f, zeta, 8, &r.samples);
                double s = 0, u = 0;
                for (size_t i = 0; i < r.samples.size(); i++) {
                    if (fld.vm[i] > s) s = fld.vm[i];
                    if (fld.disp[i] > u) u = fld.disp[i];
                }
                r.curve.push_back({f, s * k, u * k});
            }
            const auto worst = *std::max_element(r.curve.begin(), r.curve.end(), [](const auto& a, const auto& c) { return a.vm < c.vm; });
            r.staticVM = r.curve[0].vm;
            if (!(view_.freq > 0)) view_.freq = worst.f;
            r.hasHist = false;
            fieldAtFreq();
        } else {
            Excitation ex;
            ex.kind = o_.type == "shock" ? "pulse" : o_.type.toStdString();
            ex.amplitude = 1;
            ex.freq = o_.sineF;
            ex.duration = o_.pulse / 1000;
            ex.total = o_.type == "quake" ? o_.quake : o_.duration;
            double fMin = INFINITY;
            for (const auto& m : r.run.modes) fMin = std::min(fMin, m.freq);
            const double T1 = 1 / fMin;
            const double total = o_.type == "shock" ? std::max(4 * o_.pulse / 1000, 6 * T1) : o_.type == "quake" ? o_.quake * 1.1 : o_.duration + 3 * T1;
            r.hist = timeHistory(b, excitation(ex), zeta, total);
            r.hasHist = true;
            const int n = int(r.hist.times.size());
            const int every = std::max(1, int(std::ceil(n / 1500.0)));
            r.tcurve.clear();
            for (int i = 0; i < n; i += every) {
                const auto vm = stressAt(b, r.hist, i, &r.samples);
                double s = 0;
                for (float v : vm) s = std::max(s, double(v));
                r.tcurve.push_back({i, r.hist.times[i], s * k});
            }
            const auto worst = *std::max_element(r.tcurve.begin(), r.tcurve.end(), [](const auto& a, const auto& c) { return a.vm < c.vm; });
            r.worstIndex = worst.i;
            // envelope over the largest local peaks of the response
            std::vector<TPoint> peaks;
            for (size_t j = 1; j + 1 < r.tcurve.size(); j++)
                if (r.tcurve[j].vm >= r.tcurve[j - 1].vm && r.tcurve[j].vm >= r.tcurve[j + 1].vm) peaks.push_back(r.tcurve[j]);
            std::sort(peaks.begin(), peaks.end(), [](const auto& a, const auto& c) { return a.vm > c.vm; });
            if (peaks.size() > 10) peaks.resize(10);
            peaks.insert(peaks.begin(), worst);
            r.envelope.assign(part().nVert, 0.f);
            for (const auto& p : peaks) {
                const auto vm = stressAt(b, r.hist, p.i);
                for (size_t v = 0; v < r.envelope.size(); v++) r.envelope[v] = std::isnan(vm[v]) ? NAN : std::max(r.envelope[v], float(vm[v] * k));
            }
            view_.index = worst.i;
            view_.plot = "envelope";
        }
        invalidate();
        show();
    }

    void fieldAtFreq() {
        auto& r = *r_;
        const double k = std::abs(scale());
        const auto fld = harmonicField(r.run.basis, 2 * M_PI * view_.freq, o_.zeta / 100, 16);
        r.fieldVM = scaled(fld.vm, k);
        r.fieldDisp = scaled(fld.disp, k);
        r.fieldF = view_.freq;
        r.hasField = true;
    }

    void draw() {
        if (!r_ || !visible()) return;
        auto& r = *r_;
        const Material& mat = r.material;
        const double k = std::abs(scale());
        std::vector<float> values, u;
        double sc = 1;
        QString sub;
        const bool harmonic = o_.type == "harmonic";
        if (harmonic) {
            values = view_.plot == "disp" ? r.fieldDisp : r.fieldVM;
            u = harmonicShape(r.run.basis, 2 * M_PI * r.fieldF, o_.zeta / 100, view_.animate ? view_.phase : 0);
            // scale from the vibration amplitude (the shape at one phase can be near zero)
            const double amp = std::max(0.0, range(r.fieldDisp).hi) / (k > 0 ? k : 1);
            sc = amp > 0 ? std::min(1e6, 0.05 * r.diag / amp) : 1;
            sub = tr("%1 · %2 · %3% damping · shape exaggerated").arg(QString::fromStdString(mat.name), freqText(r.fieldF), num(o_.zeta));
        } else {
            values = view_.plot == "now" ? scaled(stressAt(r.run.basis, r.hist, view_.index), k) : r.envelope;
            u = shapeAt(r.run.basis, r.hist, view_.index);
            const double peakU = std::max(maxAbs3(shapeAt(r.run.basis, r.hist, r.worstIndex)), 1e-30);
            sc = std::min(1e6, 0.05 * r.diag / peakU);
            sub = tr("%1 · %2 · shape exaggerated").arg(QString::fromStdString(mat.name), view_.plot == "now" ? tr("t = %1").arg(timeText(r.hist.times[view_.index])) : tr("peak over the event"));
        }
        const bool disp = harmonic && view_.plot == "disp";
        // time plots keep one scale (the event's peak) so the animation compares like with like
        const double peak = harmonic || disp ? range(values).hi : std::max(range(r.envelope).hi, range(values).hi);
        const double hi = peak > 0 ? peak : 1;
        const QString units = r.units;
        LegendSpec spec;
        spec.min = 0;
        spec.max = hi;
        spec.title = disp ? tr("Vibration amplitude") : harmonic ? tr("Stress amplitude") : tr("von Mises stress");
        spec.format = disp ? std::function<QString(double)>([units](double x) { return num(x) + " " + units; }) : stressFormatter(hi);
        spec.sub = sub;
        if (!disp && hi >= mat.yield * 1e6) spec.markers = {{mat.yield * 1e6, QString("Yield %1").arg(num(mat.yield))}};
        paint(values, spec);
        app()->viewer->setDeformation(&u, sc);
        drawScale_ = sc;
        marker(range(values).arg, &u, sc, tr("Highest dynamic stress"));
        panel_->drawOverlays();
    }

    struct FPoint { double f, vm, disp; };
    struct TPoint { int i; double t, vm; };
    struct Result {
        ModalRun run;
        QString units;
        Material material;
        double diag = 1;
        std::vector<int> samples;
        std::vector<FPoint> curve;
        double staticVM = 0;
        bool hasField = false, hasHist = false;
        std::vector<float> fieldVM, fieldDisp;
        double fieldF = 0;
        TimeHistory hist;
        std::vector<TPoint> tcurve;
        int worstIndex = 0;
        std::vector<float> envelope;
    };
    struct Options {
        bool base = false;
        int dir = 1;
        QString type = "harmonic";
        double amp = 1, zeta = 2;
        int nev = 10;
        double pulse = 5, sineF = 50, duration = 0.2, quake = 10;
    } o_;
    struct View { double freq = 0; int index = 0; QString plot = "vm"; double phase = 0; bool playing = false, animate = true; } view_;
    std::optional<Result> r_;
    double drawScale_ = 1;
    LineChart* chart_ = nullptr;
    QDoubleSpinBox* freqBox_ = nullptr;
    QSlider* slider_ = nullptr;
    QLabel* timeLabel_ = nullptr;
    QPushButton* play_ = nullptr;
    QComboBox* plotBox_ = nullptr;
};

// ---------------------------------------------------------------------------------------------
// Optimization

class OptimizeStudy : public Study {
public:
    using Study::Study;
    QString id() const override { return "optimize"; }
    bool hasResult() const override { return topo_.has_value() || sizing_.has_value(); }
    void clear() override {
        topo_.reset();
        sizing_.reset();
        app()->viewer->clearLayer("shape");
        app()->viewer->clearLayer("voxels");
    }
    void leave() override {
        if (autoXRay_) { app()->setXRay(false); autoXRay_ = false; }
        app()->viewer->clearLayer("shape");
        app()->viewer->clearLayer("voxels");
    }

    bool options(QVBoxLayout* l) override {
        l->addWidget(heading(tr("OPTIMIZATION")));
        l->addWidget(field(tr("Goal"), segmented({tr("Remove material"), tr("Material & size")}, o_.sizing ? 1 : 0, [this](int i) {
                               o_.sizing = i == 1;
                               QMetaObject::invokeMethod(panel_, [p = panel_] { p->renderStudyOptions(); }, Qt::QueuedConnection);
                               panel_->markStale();
                           })));
        if (!o_.sizing) {
            l->addWidget(field(tr("Keep (% of the mass)"), numberBox(o_.keep, 5, 95, 5, 0, [this](double v) { o_.keep = v; panel_->markStale(); })));
            l->addWidget(field(tr("Iterations"), numberBox(o_.iters, 5, 100, 5, 0, [this](double v) { o_.iters = int(v); panel_->markStale(); })));
            l->addWidget(note(tr("Finds the stiffest shape that uses this much material for your fixtures and loads. Material at the fixtures and loads is always kept. Each iteration is one full solve, so use a coarse mesh first.")));
        } else {
            l->addWidget(field(tr("Minimum safety factor"), numberBox(o_.fos, 0.1, 100, 0.1, 2, [this](double v) { o_.fos = v; panel_->markStale(); })));
            l->addWidget(field(tr("Max displacement (%1, 0 = any)").arg(app()->units), numberBox(o_.maxDisp, 0, 1e9, 0.1, 3, [this](double v) { o_.maxDisp = v; panel_->markStale(); })));
            l->addWidget(field(tr("Minimize"), segmented({tr("Weight"), tr("Material cost")}, o_.cost ? 1 : 0, [this](int i) {
                                   o_.cost = i == 1;
                                   if (sizing_) { invalidate(); show(); }
                               })));
            l->addWidget(field(tr("Materials"), segmented({tr("All"), tr("Metals"), tr("Plastics")}, o_.family, [this](int i) { o_.family = i; panel_->markStale(); })));
            l->addWidget(note(tr("Scales the whole part (0.25× to 4×) for each material to the smallest size that keeps the safety factor. Costs are rough raw-material prices.")));
        }
        return true;
    }

    void run() override {
        if (o_.sizing) runSizingStudy();
        else runTopologyStudy();
    }

protected:
    void build(QVBoxLayout* body) override {
        if (sizing_) return buildSizing(body);
        const auto& r = *topo_;
        setTitle(tr("Topology optimization"));
        body->addWidget(chartTitle(tr("Compliance per iteration (lower is stiffer)")));
        LineChart::Options co;
        co.xLabel = tr("iteration");
        co.formatX = [](double x) { return QString::number(std::lround(x)); };
        co.formatY = [](double y) { return num(y); };
        co.tipText = [](double x, double y, int) { return tr("iteration %1 · %2").arg(std::lround(x) + 1).arg(num(y)); };
        chart_ = new LineChart(nullptr, co);
        body->addWidget(chart_);
        auto* s = hslider(20, 80, int(std::lround(level_ * 100)), [this](int v) {
            level_ = v / 100.0;
            if (topo_->done) {
                buildShape();
                show();
            } else previewVoxels(topo_->density);
        });
        s->setTracking(false);
        body->addWidget(field(tr("Keep voxels denser than"), s));
        if (r.done)
            body->addLayout(row({button(tr("Export STL…"), [this] { exportSTL(); }), button(tr("Use as new part"), [this] { useAsPart(); })}));
        body->addWidget(note(tr("The blue shape is the optimized design; the original part is shown see-through. Re-model it in CAD (or export the STL) and check it with a static study.")));
    }

    void refresh() override {
        if (sizing_) return refreshSizing();
        const auto& r = *topo_;
        const Material& mat = r.material;
        const double massFull = part().volume * std::pow(r.toMeters, 3) * mat.density;
        const double kept = r.shapeFraction >= 0 ? r.shapeFraction : r.history.empty() ? 1 : r.history.back().volume;
        const auto* h0 = r.history.empty() ? nullptr : &r.history.front();
        const auto* h1 = r.history.empty() ? nullptr : &r.history.back();
        setKpis({
            {tr("Material kept"), num(kept * 100) + " %", tr("≈ %1 g of %2 g").arg(num(massFull * kept * 1000), num(massFull * 1000))},
            {tr("Iterations"), QString::number(r.history.size()), r.done ? tr("converged or limit") : tr("running…")},
            {tr("Compliance"), h0 && h1 ? num(h1->compliance / h0->compliance * 100) + " %" : "–", tr("of the uniform start (lower = stiffer)")},
        });
        std::vector<QPointF> pts;
        for (const auto& p : r.history) pts.push_back(QPointF(p.it, p.compliance));
        chart_->setPoints(pts, int(pts.size()) - 1);
        draw();
    }

private:
    void runTopologyStudy() {
        const int gen = ++runGen_;
        const double volFrac = o_.keep / 100;
        const int iters = o_.iters;
        if (!panel_->startStudy(
                tr("Topology optimization…"), true, true,
                [this, gen, volFrac, iters](StructuralPanel::Prepared& p, JobControl& ctl) -> std::any {
                    const auto& m = *p.model;
                    const auto& asm_ = p.asm_;
                    // keep voxels touching a fixture or a loaded node
                    const int nx = m.grid.dims[0], ny = m.grid.dims[1], nz = m.grid.dims[2];
                    std::vector<uint8_t> keep(m.density.size(), 0);
                    for (int k = 0; k < nz; k++)
                        for (int j = 0; j < ny; j++)
                            for (int i = 0; i < nx; i++) {
                                const size_t e = i + size_t(nx) * (j + size_t(ny) * k);
                                if (!(m.density[e] > 0)) continue;
                                for (int c = 0; c < 8; c++) {
                                    const int64_t n = m.node(i + (c & 1), j + ((c >> 1) & 1), k + ((c >> 2) & 1));
                                    if (asm_.bc[3 * n] || asm_.f[3 * n] || asm_.f[3 * n + 1] || asm_.f[3 * n + 2]) { keep[e] = 1; break; }
                                }
                            }
                    auto model = p.model;
                    const Material mat = p.material;
                    const double toMeters = p.toMeters;
                    ctl.post([this, gen, model, mat, toMeters] {
                        if (gen != runGen_) return;
                        topo_ = Topo{};
                        topo_->model = model;
                        topo_->material = mat;
                        topo_->toMeters = toMeters;
                        sizing_.reset();
                        invalidate();
                        panel_->setDisplay("study");
                        if (!app()->viewer->xray()) { app()->setXRay(true); autoXRay_ = true; }
                    });
                    auto last = std::make_shared<std::chrono::steady_clock::time_point>();
                    return runTopology(*p.model, p.input, keep, volFrac, iters,
                                       [this, gen, &ctl, last](TopologyStep&& s) {
                                           const auto now = std::chrono::steady_clock::now();
                                           const bool drawIt = now - *last > std::chrono::milliseconds(400);
                                           if (drawIt) *last = now;
                                           auto shared = std::make_shared<TopologyStep>(std::move(s));
                                           ctl.post([this, gen, shared, drawIt] {
                                               if (gen != runGen_ || !topo_) return;
                                               topo_->history.push_back({shared->it, shared->compliance, shared->volume});
                                               topo_->density = std::move(shared->density);
                                               if (drawIt) show();
                                           });
                                       },
                                       progressOf(ctl));
                },
                [this, gen](std::any& value, StructuralPanel::Prepared&) {
                    if (gen != runGen_ || !topo_) return;
                    auto res = std::any_cast<TopologyRun>(std::move(value));
                    topo_->density = std::move(res.density);
                    topo_->history = std::move(res.history);
                    topo_->done = true;
                    buildShape();
                    invalidate();
                    show();
                    if (topo_->history.empty()) return;
                    const auto& h0 = topo_->history.front();
                    const auto& h1 = topo_->history.back();
                    status(tr("Topology optimized %1 in %2 iterations: %3% of the material kept; compliance %4% of the uniform start.")
                               .arg(engineNote(res.engine, res.gpuNote))
                               .arg(topo_->history.size())
                               .arg(num(h1.volume * 100), num(h1.compliance / h0.compliance * 100)));
                },
                [this, gen](bool) {
                    if (gen != runGen_ || !topo_) return;
                    if (topo_->density.empty()) topo_.reset();
                    else { invalidate(); show(); }
                }))
            --runGen_;
    }

    void runSizingStudy() {
        const int gen = ++runGen_;
        const auto fixtures = panel_->fixtures;
        const auto loads = panel_->loads;
        const bool gravity = panel_->gravity();
        const int family = o_.family;
        SizingOptions so;
        so.fosTarget = o_.fos;
        so.maxDisp = o_.maxDisp;
        so.volume = part().volume;
        so.base = app()->material;
        auto isPlastic = [](const Material& m) { return m.E < 10; };
        for (const auto& m : materials())
            if (family == 0 || (family == 1 ? !isPlastic(m) && m.fatigue.metal : isPlastic(m))) so.materials.push_back(m);
        if (std::none_of(so.materials.begin(), so.materials.end(), [&](const Material& m) { return m.id == so.base.id; })) so.materials.insert(so.materials.begin(), so.base);
        if (!panel_->startStudy(
                tr("Comparing materials and sizes…"), true, true,
                [so, fixtures, loads, gravity](StructuralPanel::Prepared& p, JobControl& ctl) mutable -> std::any {
                    auto add = [&](const std::string& name, const std::string& kind, const std::vector<Load>& ls, bool g) {
                        if (ls.empty() && !g) return;
                        const auto a = p.model->assemble(fixtures, ls, g, p.material, p.toMeters);
                        bool any = false;
                        for (size_t i = 0; i < a.f.size() && !any; i++) any = a.f[i] != 0 && !a.bc[i];
                        if (any) so.groups.push_back({name, kind, a.f});
                    };
                    std::vector<Load> forces, other;
                    for (const auto& l : loads) (l.type == Load::Force ? forces : other).push_back(l);
                    add("forces", "force", forces, false);
                    add("pressures & wind", "pressure", other, false);
                    add("self-weight", "gravity", {}, gravity);
                    return runSizing(*p.model, p.input, so, p.toMeters, progressOf(ctl));
                },
                [this, gen](std::any& value, StructuralPanel::Prepared& p) {
                    if (gen != runGen_) return;
                    sizing_ = Sizing{std::any_cast<SizingRun>(std::move(value)), p.material, p.units};
                    topo_.reset();
                    app()->viewer->clearLayer("shape");
                    app()->viewer->clearLayer("voxels");
                    panel_->setDisplay("study");
                    invalidate();
                    show();
                    const auto list = ranked();
                    if (!list.empty())
                        status(tr("Best: %1 at %2× size, %3 g (%4).")
                                   .arg(QString::fromStdString(list[0].name), num(list[0].scale), num(list[0].mass * 1000), engineNote(sizing_->run.engine, sizing_->run.gpuNote)));
                    else status(tr("No material meets the safety factor within 0.25×–4× size."), "warn");
                }))
            --runGen_;
    }

    std::vector<SizingRow> ranked() const {
        std::vector<SizingRow> list;
        for (const auto& r : sizing_->run.rows)
            if (r.feasible) list.push_back(r);
        std::sort(list.begin(), list.end(), [this](const SizingRow& a, const SizingRow& b) { return o_.cost ? a.cost < b.cost : a.mass < b.mass; });
        return list;
    }

    void buildSizing(QVBoxLayout* body) {
        setTitle(tr("Material & size"));
        auto list = ranked();
        for (const auto& r : sizing_->run.rows)
            if (!r.feasible) list.push_back(r);
        auto* table = new QWidget;
        auto* g = new QGridLayout(table);
        g->setContentsMargins(0, 0, 0, 0);
        g->setHorizontalSpacing(8);
        g->setVerticalSpacing(4);
        const QStringList heads = {tr("Material"), tr("Size"), tr("Mass"), tr("Cost"), tr("FOS"), ""};
        for (int c = 0; c < heads.size(); c++) {
            auto* l = new QLabel(heads[c]);
            l->setObjectName("meta");
            g->addWidget(l, 0, c);
        }
        for (size_t i = 0; i < list.size(); i++) {
            const auto& r = list[i];
            QString name = QString::fromStdString(r.name);
            name = name.section(" (", 0, 0);
            auto* nl = new QLabel(name);
            nl->setToolTip(QString::fromStdString(r.name));
            if (i == 0 && r.feasible) nl->setStyleSheet("font-weight: 600;");
            g->addWidget(nl, int(i) + 1, 0);
            g->addWidget(new QLabel(r.feasible ? num(r.scale) + "×" : "–"), int(i) + 1, 1);
            g->addWidget(new QLabel(r.feasible ? num(r.mass * 1000) + " g" : "–"), int(i) + 1, 2);
            g->addWidget(new QLabel(r.feasible ? "$" + num(r.cost) : "–"), int(i) + 1, 3);
            g->addWidget(new QLabel(r.feasible ? num(r.fos) : tr("no")), int(i) + 1, 4);
            if (r.feasible) {
                auto* b = button(tr("Use"), [this, r] { apply(r); });
                b->setToolTip(tr("Use this material and size"));
                g->addWidget(b, int(i) + 1, 5);
            }
        }
        g->setColumnStretch(0, 1);
        body->addWidget(table);
        body->addWidget(note(tr("Safety factor ≥ %1%2. Stress scales exactly with size and load type; differences in Poisson's ratio are ignored, so re-run the static study after applying.")
                                 .arg(num(o_.fos), o_.maxDisp > 0 ? tr(", displacement ≤ %1 %2").arg(num(o_.maxDisp), sizing_->units) : QString())));
    }

    void refreshSizing() {
        const auto list = ranked();
        const auto& s = *sizing_;
        const double massNow = part().volume * std::pow(app()->toMeters(), 3) * s.material.density;
        const SizingRow* best = list.empty() ? nullptr : &list[0];
        setKpis({
            {tr("Current design"), "FOS " + num(s.run.base.fos), tr("%1 · %2 g").arg(QString::fromStdString(s.material.name), num(massNow * 1000))},
            {o_.cost ? tr("Cheapest") : tr("Lightest"), best ? QString::fromStdString(best->name).section(" (", 0, 0) : tr("none"),
             best ? tr("%1× size · %2 g · $%3").arg(num(best->scale), num(best->mass * 1000), num(best->cost)) : tr("nothing meets the target")},
        });
        draw();
    }

    void apply(const SizingRow& row) {
        const Material* m = findMaterial(row.id);
        if (m) app()->setMaterial(*m);
        if (std::abs(row.scale - 1) > 1e-3) app()->applyScale(row.scale);
        status(tr("Applied %1 at %2× size. Run the static study to check it.").arg(QString::fromStdString(row.name), num(row.scale)));
    }

    void previewVoxels(const std::vector<float>& density) {
        if (!topo_ || !visible() || density.empty()) return;
        const auto& m = *topo_->model;
        const int nx = m.grid.dims[0], ny = m.grid.dims[1];
        std::vector<float> pts;
        for (size_t e = 0; e < density.size(); e++) {
            if (!(density[e] > level_)) continue;
            const int i = int(e % nx), j = int((e / nx) % ny), k = int(e / (size_t(nx) * ny));
            pts.push_back(float(m.grid.origin[0] + (i + 0.5) * m.grid.h));
            pts.push_back(float(m.grid.origin[1] + (j + 0.5) * m.grid.h));
            pts.push_back(float(m.grid.origin[2] + (k + 0.5) * m.grid.h));
        }
        std::vector<Geometry> g;
        g.push_back(shapes::cubes(pts, float(m.grid.h * 0.95), QVector3D(0x86 / 255.f, 0xa8 / 255.f, 0xd8 / 255.f)));
        app()->viewer->setLayer("voxels", std::move(g));
    }

    void buildShape() {
        auto& r = *topo_;
        const auto& m = *r.model;
        r.shape = surfaceNets(r.density, m.grid.dims, m.grid.origin, m.grid.h, level_, 3);
        double vol = 0, total = 0;
        for (size_t e = 0; e < r.density.size(); e++)
            if (m.density[e] > 0) {
                total += m.density[e];
                if (r.density[e] > level_) vol += m.density[e];
            }
        r.shapeFraction = total > 0 ? vol / total : 0;
    }

    void draw() {
        if (!visible()) return;
        Viewport* v = app()->viewer;
        if (sizing_) {
            v->clearLayer("shape");
            v->clearLayer("voxels");
            return;
        }
        v->setScalars(nullptr);
        v->setDeformation(nullptr);
        v->clearMarker();
        app()->legend->clear();
        shown_.clear();
        const auto& r = *topo_;
        if (r.done && !r.shape.index.empty()) {
            v->clearLayer("voxels");
            // smooth normals for the optimized shape
            const auto& P = r.shape.positions;
            const auto& I = r.shape.index;
            std::vector<QVector3D> normals(P.size() / 3);
            for (size_t t = 0; t + 2 < I.size(); t += 3) {
                const QVector3D a(P[3 * I[t]], P[3 * I[t] + 1], P[3 * I[t] + 2]);
                const QVector3D b(P[3 * I[t + 1]], P[3 * I[t + 1] + 1], P[3 * I[t + 1] + 2]);
                const QVector3D c(P[3 * I[t + 2]], P[3 * I[t + 2] + 1], P[3 * I[t + 2] + 2]);
                const QVector3D n = QVector3D::crossProduct(b - a, c - a);
                for (int k = 0; k < 3; k++) normals[I[t + k]] += n;
            }
            Geometry geo;
            geo.color = QVector3D(0x3f / 255.f, 0x7f / 255.f, 0xd9 / 255.f);
            for (size_t t = 0; t < I.size(); t++) {
                const uint32_t q = I[t];
                geo.vertex(QVector3D(P[3 * q], P[3 * q + 1], P[3 * q + 2]), normals[q].normalized());
            }
            std::vector<Geometry> list;
            list.push_back(std::move(geo));
            v->setLayer("shape", std::move(list));
        } else previewVoxels(r.density);
        panel_->drawOverlays();
    }

    void exportSTL() {
        if (!topo_ || topo_->shape.index.empty()) return;
        const QString name = QString::fromStdString(part().name.empty() ? "part" : part().name) + "-optimized";
        const QString path = QFileDialog::getSaveFileName(app(), tr("Export STL"), name + ".stl", tr("STL files (*.stl)"));
        if (path.isEmpty()) return;
        const auto bytes = toSTL(topo_->shape, name.toStdString());
        QFile f(path);
        if (!f.open(QIODevice::WriteOnly) || f.write(reinterpret_cast<const char*>(bytes.data()), qint64(bytes.size())) != qint64(bytes.size())) {
            status(tr("Could not save %1.").arg(path), "error");
            return;
        }
        status(tr("STL saved to %1.").arg(path));
    }

    void useAsPart() {
        if (!topo_ || topo_->shape.index.empty()) return;
        MeshSource src;
        src.positions = topo_->shape.positions;
        src.index = topo_->shape.index;
        src.name = (part().name.empty() ? std::string("part") : part().name) + " (optimized)";
        app()->loadGeneratedPart(std::move(src), tr("Optimized shape"));
    }

    struct Topo {
        std::shared_ptr<StructuralModel> model;
        Material material;
        double toMeters = 0.001;
        std::vector<float> density;
        std::vector<TopologyRun::Point> history;
        TriMesh shape;
        double shapeFraction = -1;
        bool done = false;
    };
    struct Sizing {
        SizingRun run;
        Material material;
        QString units;
    };
    struct Options {
        bool sizing = false;
        double keep = 40;
        int iters = 30;
        double fos = 2, maxDisp = 0;
        bool cost = false;
        int family = 0;
    } o_;
    std::optional<Topo> topo_;
    std::optional<Sizing> sizing_;
    double level_ = 0.5;
    bool autoXRay_ = false;
    LineChart* chart_ = nullptr;
};

}  // namespace

std::vector<std::unique_ptr<Study>> createStudies(StructuralPanel* panel) {
    std::vector<std::unique_ptr<Study>> list;
    list.push_back(std::make_unique<NonlinearStudy>(panel));
    list.push_back(std::make_unique<ModalStudy>(panel));
    list.push_back(std::make_unique<BucklingStudy>(panel));
    list.push_back(std::make_unique<FatigueStudy>(panel));
    list.push_back(std::make_unique<DropStudy>(panel));
    list.push_back(std::make_unique<DynamicStudy>(panel));
    list.push_back(std::make_unique<OptimizeStudy>(panel));
    return list;
}

}  // namespace ps
