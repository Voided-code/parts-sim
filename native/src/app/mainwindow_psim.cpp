// Saving and opening .psim files in the app (JS reference: src/ui/psim-io.js). The container and codecs are
// in core/psim.*; this file gathers the part, setup and results from the panels and puts them back.
#include <QApplication>
#include <QBuffer>
#include <QCheckBox>
#include <QDateTime>
#include <QDialog>
#include <QDialogButtonBox>
#include <QElapsedTimer>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QMessageBox>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QRadioButton>
#include <QStandardPaths>
#include <QTableWidget>
#include <QHeaderView>
#include <QVBoxLayout>
#include <cmath>

#include "airflowpanel.hpp"
#include "mainwindow.hpp"
#include "psimjson.hpp"
#include "structuralpanel.hpp"
#include "thermalpanel.hpp"
#include "viewport.hpp"
#include "widgets.hpp"

namespace ps {

namespace {

using json::Value;
using namespace pj;

QString kb(double n) {
    if (n >= 1e6) return QString::number(n / 1e6, 'f', 2) + " MB";
    if (n >= 1e3) return QString::number(n / 1e3, 'f', 1) + " KB";
    return QString::number(int(n)) + " B";
}

const std::map<QString, QString>& resultLabels() {
    static const std::map<QString, QString> m = {
        {"static", "Bend test (static stresses)"}, {"break", "Break test"}, {"nonlinear", "Nonlinear static"}, {"modal", "Frequency"},
        {"buckling", "Buckling"}, {"fatigue", "Fatigue"}, {"drop", "Drop test"}, {"optimize", "Optimization"}, {"thermal", "Thermal"}, {"airflow", "Airflow"}};
    return m;
}
QString labelOf(const QString& id) { auto it = resultLabels().find(id); return it == resultLabels().end() ? id : it->second; }

psim::Geometry geometryOf(const Part& p) {
    psim::Geometry g;
    g.vertices = p.vertices;
    g.tris = p.tris;
    g.brepFaces = p.brepFaces;
    if (p.brepFaces) g.faceOf = p.faceOf;
    g.faceCount = uint32_t(p.faceCount);
    g.faceAngle = p.faceAngle;
    return g;
}

size_t deflated(const psim::Bytes& b) { return psim::deflateRaw(b.data(), b.size()).size(); }

}  // namespace

// ---------------------------------------------------------------- gathering

psim::File MainWindow::gatherPsim(const PsimSaveOptions& o, QStringList* resultIds) {
    psim::File f;
    const Part& p = *part;
    f.geometry = geometryOf(p);
    Value info = jobj();
    Value app = jobj();
    app.obj["name"] = jstr("Parts Sim");
    app.obj["version"] = jstr(QApplication::applicationVersion().toStdString());
    app.obj["kind"] = jstr("native");
    info.obj["app"] = app;
    info.obj["name"] = jstr((o.name.isEmpty() ? QString::fromStdString(p.name) : o.name).toStdString());
    info.obj["notes"] = jstr(o.notes.toStdString());
    info.obj["units"] = jstr(units.toStdString());
    Value part_ = jobj();
    part_.obj["vertices"] = jnum(p.nVert);
    part_.obj["triangles"] = jnum(p.nTri);
    Value bb = jobj();
    bb.obj["min"] = vec3(p.bbox.min);
    bb.obj["max"] = vec3(p.bbox.max);
    part_.obj["bbox"] = bb;
    info.obj["part"] = part_;
    if (o.setup && !partInfo.isEmpty()) info.obj["partSource"] = jstr(partInfo.left(200).toStdString());

    // results
    std::vector<std::pair<QString, psim::Arrays>> results;
    if (auto r = structural->exportStatic()) results.push_back({"static", std::move(*r)});
    if (auto r = structural->exportBreak()) results.push_back({"break", std::move(*r)});
    if (auto r = thermal->exportResult()) results.push_back({"thermal", std::move(*r)});
    std::optional<psim::Arrays> air = airflow->exportResults();
    if (resultIds) {
        for (auto& r : results) *resultIds << r.first;
        if (air) *resultIds << "airflow";
    }
    Value rlist = jarr();
    psim::Arrays rfea;
    rfea.meta = jobj();
    for (auto& [id, r] : results) {
        if (!o.allResults && !o.results.contains(id)) continue;
        rlist.arr.push_back(jstr(id.toStdString()));
        rfea.meta.obj[id.toStdString()] = r.meta;
        for (auto& a : r.list) rfea.list.push_back(std::move(a));
    }
    if (air && (o.allResults || o.results.contains("airflow"))) {
        rlist.arr.push_back(jstr("airflow"));
        f.rair = std::move(*air);
    }
    rfea.meta.obj["results"] = rlist;
    if (o.setup)
        for (auto& a : structural->windArrays()) rfea.list.push_back(std::move(a));
    Value contains = jobj();
    contains.obj["geometry"] = jstr(o.exact ? "exact" : "quantised16");
    contains.obj["setup"] = jbool(o.setup);
    contains.obj["results"] = rlist;
    contains.obj["cad"] = jbool(false);
    // RFEA lists only the studies it holds
    {
        Value rl = jarr();
        for (const auto& x : rlist.arr) if (x.str != "airflow") rl.arr.push_back(x);
        rfea.meta.obj["results"] = rl;
        if (!rl.arr.empty() || !rfea.list.empty()) f.rfea = std::move(rfea);
    }

    if (o.cad && !sourceBytes.isEmpty()) {
        f.cad = psim::Bytes(sourceBytes.begin(), sourceBytes.end());
        contains.obj["cad"] = jbool(true);
        contains.obj["cadName"] = jstr(QFileInfo(sourceName).fileName().toStdString());
    }
    info.obj["contains"] = contains;
    f.info = info;

    if (o.setup) {
        Value s = jobj();
        s.obj["units"] = jstr(units.toStdString());
        s.obj["material"] = materialToJson(material);
        s.obj["structural"] = structural->exportSetup();
        s.obj["thermal"] = thermal->exportState();
        s.obj["airflow"] = airflow->exportState();
        f.setup = s;
    }
    if (o.setup || !rlist.arr.empty()) {
        Value v = jobj();
        v.obj["tab"] = jstr(currentTab_.toStdString());
        v.obj["study"] = jstr(structural->exportSetup()["study"].str);
        const auto c = viewer->cameraState();
        Value cam = jobj();
        cam.obj["position"] = vec({c[0], c[1], c[2]});
        cam.obj["target"] = vec({c[3], c[4], c[5]});
        cam.obj["up"] = vec({c[6], c[7], c[8]});
        v.obj["camera"] = cam;
        v.obj["structural"] = structural->exportView();
        v.obj["thermal"] = thermal->exportView();
        f.view = v;
    }
    if (o.thumb) {
        QImage img = viewer->screenshot();
        if (!img.isNull()) {
            img = img.scaledToWidth(320, Qt::SmoothTransformation);
            QByteArray jpg;
            QBuffer buf(&jpg);
            buf.open(QIODevice::WriteOnly);
            if (img.convertToFormat(QImage::Format_RGB32).save(&buf, "JPEG", 72)) f.thumb = psim::Bytes(jpg.begin(), jpg.end());
        }
    }
    return f;
}

bool MainWindow::savePsim(const QString& path, const PsimSaveOptions& o) {
    if (!part) { status(tr("Open a part or a sample first."), "error"); return false; }
    QElapsedTimer t;
    t.start();
    try {
        psim::WriteOptions w;
        w.quantised = !o.exact;
        w.created = QDateTime::currentDateTimeUtc().toString("yyyy-MM-ddTHH:mm:ss'Z'").toStdString();
        const psim::Bytes bytes = psim::writePsim(gatherPsim(o), w);
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Truncate) || file.write(reinterpret_cast<const char*>(bytes.data()), qint64(bytes.size())) != qint64(bytes.size())) {
            status(tr("Could not write %1: %2").arg(path, file.errorString()), "error");
            return false;
        }
        psimSaveMs = double(t.elapsed());
        status(tr("Saved %1 as a .psim file.").arg(kb(double(bytes.size()))));
        return true;
    } catch (const std::exception& e) {
        status(tr("Could not save: %1").arg(e.what()), "error");
        return false;
    }
}

void MainWindow::showSaveDialog() {
    if (!part) return status(tr("Open a part or a sample first."), "error");
    QStringList ids;
    PsimSaveOptions all;
    all.thumb = false;
    psim::File draft = gatherPsim(all, &ids);
    // compressed sizes of each piece
    const double geomQ = double(deflated(psim::encodeGeometry(*draft.geometry, true))), geomX = double(deflated(psim::encodeGeometry(*draft.geometry, false)));
    double setupSize = draft.setup ? double(deflated(psim::Bytes(psim::stringifyJson(*draft.setup).begin(), psim::stringifyJson(*draft.setup).end()))) : 0;
    std::map<QString, double> resSize;
    if (draft.rfea) {
        for (const QString& id : ids) {
            psim::Arrays one;
            one.meta = draft.rfea->meta.obj.count(id.toStdString()) ? draft.rfea->meta.obj.at(id.toStdString()) : jobj();
            const std::string prefix = id.toStdString() + ".";
            for (const auto& a : draft.rfea->list) if (a.name.rfind(prefix, 0) == 0) one.list.push_back(a);
            resSize[id] = double(deflated(psim::encodeArrays(one.meta, one.list)));
        }
    }
    if (draft.rair) resSize["airflow"] = double(deflated(psim::encodeArrays(draft.rair->meta, draft.rair->list)));
    QDialog d(this);
    d.setWindowTitle(tr("Save as .psim"));
    auto* v = new QVBoxLayout(&d);
    auto* name = new QLineEdit(QString::fromStdString(part->name), &d);
    auto* notes = new QPlainTextEdit(&d);
    notes->setPlaceholderText(tr("Notes for whoever opens the file (optional)"));
    notes->setMaximumHeight(60);
    v->addWidget(new QLabel(tr("Name"), &d));
    v->addWidget(name);
    v->addWidget(notes);
    auto* gbox = new QGroupBox(tr("Part"), &d);
    auto* gl = new QVBoxLayout(gbox);
    auto* compact = new QRadioButton(tr("Geometry, compact (positions to 16 bits)  %1").arg(kb(geomQ)), gbox);
    auto* exact = new QRadioButton(tr("Geometry, exact (32-bit positions)  %1").arg(kb(geomX)), gbox);
    compact->setChecked(true);
    gl->addWidget(compact);
    gl->addWidget(exact);
    v->addWidget(gbox);
    auto* ibox = new QGroupBox(tr("Include"), &d);
    auto* il = new QVBoxLayout(ibox);
    auto* setupChk = new QCheckBox(tr("Setup (material, supports, loads, study settings)  %1").arg(kb(setupSize)), ibox);
    setupChk->setChecked(true);
    il->addWidget(setupChk);
    std::map<QString, QCheckBox*> resChk;
    for (const QString& id : ids) {
        auto* c = new QCheckBox(tr("%1  %2").arg(labelOf(id), kb(resSize[id])), ibox);
        c->setChecked(true);
        resChk[id] = c;
        il->addWidget(c);
    }
    if (ids.isEmpty()) il->addWidget(new QLabel(tr("No results yet: run a study to include its results."), ibox));
    auto* cadChk = new QCheckBox(sourceBytes.isEmpty() ? tr("Original CAD file  (not available)") : tr("Original CAD file  %1").arg(kb(double(sourceBytes.size()))), ibox);
    cadChk->setEnabled(!sourceBytes.isEmpty());
    il->addWidget(cadChk);
    auto* thumbChk = new QCheckBox(tr("Preview picture"), ibox);
    thumbChk->setChecked(true);
    il->addWidget(thumbChk);
    v->addWidget(ibox);
    auto* priv = new QLabel(tr("Lossy fields are listed with their largest error in the file’s info panel. The file never holds your user name, folders or computer "
                               "details, but it does hold the part’s geometry: share it only with people who may see the part."), &d);
    priv->setWordWrap(true);
    priv->setEnabled(false);
    v->addWidget(priv);
    auto* total = new QLabel(&d);
    auto refresh = [&] {
        double n = 2000 + (thumbChk->isChecked() ? 8000 : 0) + (exact->isChecked() ? geomX : geomQ);
        if (setupChk->isChecked()) n += setupSize;
        for (auto& [id, c] : resChk) if (c->isChecked()) n += resSize[id];
        if (cadChk->isChecked()) n += double(sourceBytes.size());
        total->setText(tr("About %1").arg(kb(n)));
    };
    for (auto* c : std::vector<QAbstractButton*>{compact, exact, setupChk, cadChk, thumbChk}) QObject::connect(c, &QAbstractButton::toggled, &d, refresh);
    for (auto& [id, c] : resChk) QObject::connect(c, &QAbstractButton::toggled, &d, refresh);
    refresh();
    auto* row = new QHBoxLayout;
    row->addWidget(total, 1);
    auto* bb = new QDialogButtonBox(QDialogButtonBox::Save | QDialogButtonBox::Cancel, &d);
    row->addWidget(bb);
    v->addLayout(row);
    QObject::connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
    QObject::connect(bb, &QDialogButtonBox::rejected, &d, &QDialog::reject);
    if (d.exec() != QDialog::Accepted) return;
    PsimSaveOptions o;
    o.name = name->text().trimmed();
    o.notes = notes->toPlainText().trimmed();
    o.exact = exact->isChecked();
    o.setup = setupChk->isChecked();
    o.allResults = false;
    for (auto& [id, c] : resChk) if (c->isChecked()) o.results << id;
    o.cad = cadChk->isChecked();
    o.thumb = thumbChk->isChecked();
    QString base = o.name.isEmpty() ? QString::fromStdString(part->name) : o.name;
    base.replace(QRegularExpression("[\\\\/:*?\"<>|]+"), "_");
    const QString dir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation);
    const QString file = QFileDialog::getSaveFileName(this, tr("Save as .psim"), dir + "/" + base + ".psim", tr("Parts Sim file (*.psim)"));
    if (file.isEmpty()) return;
    savePsim(file.endsWith(".psim", Qt::CaseInsensitive) ? file : file + ".psim", o);
}

// ---------------------------------------------------------------- opening

bool MainWindow::openPsim(const QString& path) {
    QElapsedTimer t;
    t.start();
    const QString fileName = QFileInfo(path).fileName();
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) { status(tr("Could not open %1: %2").arg(fileName, file.errorString()), "error"); return false; }
    const QByteArray raw = file.readAll();
    psim::File f;
    try {
        f = psim::readPsim(reinterpret_cast<const uint8_t*>(raw.constData()), size_t(raw.size()));
    } catch (const psim::Error& e) {
        status(tr("%1: %2").arg(fileName, e.what()), "error");
        return false;
    }
    LoadedFile lf;
    lf.name = fileName;
    lf.info = f.info;
    lf.table = f.table;
    lf.size = size_t(raw.size());
    lf.version = f.version;
    lf.skipped = f.skipped;
    if (!f.geometry) {
        loadedFile = lf;
        showFileInfo();
        loadedFile.reset();
        status(tr("%1 holds no geometry, so there is nothing to show. Its info panel is open.").arg(fileName), "warn");
        return false;
    }
    const Value& info = f.info;
    const psim::Geometry& g = *f.geometry;
    std::shared_ptr<Part> p;
    try {
        std::string name = strOr(info["name"], fileName.section('.', 0, 0).toStdString());
        p = restorePart(name, g.vertices, g.tris, g.faceOf, g.brepFaces, int(g.faceCount), g.faceAngle);
    } catch (const std::exception& e) {
        status(tr("%1: could not build the part: %2").arg(fileName, e.what()), "error");
        return false;
    }
    const double tRestore = double(t.elapsed());
    const QString u = QString::fromStdString(strOr(f.setup ? (*f.setup)["units"] : Value{}, strOr(info["units"], "mm")));
    restoring = true;
    struct Reset { bool& r; ~Reset() { r = false; } } resetGuard{restoring};
    const QString partSource = QString::fromStdString(strOr(info["partSource"], ("From " + fileName).toStdString()));
    if (!installPart(p, u, partSource, nullptr, true)) return false;
    QStringList problems;
    auto guard = [&](const QString& what, auto fn) {
        try { fn(); } catch (const std::exception& e) { problems << what + ": " + e.what(); }
    };
    const Value setup = f.setup ? *f.setup : Value{};
    const psim::Arrays* rfea = f.rfea ? &*f.rfea : nullptr;
    if (setup["material"].type == Value::Object) guard("material", [&] { structural->importMaterial(setup["material"]); });
    if (setup["structural"].type == Value::Object) guard("setup", [&] { structural->importSetup(setup["structural"], rfea); });
    if (setup["thermal"].type == Value::Object) guard("thermal setup", [&] { thermal->importState(setup["thermal"]); });
    if (rfea) {
        const Value& rf = rfea->meta;
        for (const auto& idv : rf["results"].arr) {
            const QString id = QString::fromStdString(strOr(idv));
            const Value& m = rf[id.toStdString()];
            if (id == "static") guard("static result", [&] {
                structural->importStatic(m, *rfea);
                lf.stored["static"] = {{"maxVM", numOr(m["maxVM"], NAN)}, {"maxDisp", numOr(m["maxDisp"], NAN)}, {"minFos", numOr(m["minFos"], NAN)}};
            });
            else if (id == "break") guard("break test", [&] { structural->importBreak(m, *rfea); lf.stored["break"] = {{"firstCrack", numOr(m["steps"][0]["lambda"], NAN)}}; });
            else if (id == "thermal") guard("thermal result", [&] {
                thermal->importResult(m, *rfea);
                lf.stored["thermal"] = {{"min", numOr(m["min"], NAN)}, {"max", numOr(m["max"], NAN)}};
            });
            else problems << labelOf(id) + ": this version cannot show it yet";
        }
    }
    if (f.rair) guard("airflow", [&] {
        const Value& a = f.rair->meta["results"];
        const Value* st = setup["airflow"].type == Value::Object ? &setup["airflow"] : nullptr;
        airflow->loadResults(f.rair->meta, *f.rair, st);
        lf.stored["airflow"] = {{"drag", numOr(a["drag"], NAN)}, {"lift", numOr(a["lift"], NAN)}, {"cd", numOr(a["cd"], NAN)}, {"cl", numOr(a["cl"], NAN)}};
    });
    else if (setup["airflow"].type == Value::Object) guard("airflow setup", [&] { airflow->importState(setup["airflow"]); });
    const Value view = f.view ? *f.view : Value{};
    guard("view", [&] {
        structural->importView(view["structural"]);
        thermal->importView(view["thermal"]);
        structural->restoreStudy(QString::fromStdString(strOr(view["study"], strOr(setup["structural"]["study"], "static"))));
        const Value& c = view["camera"];
        if (c["position"].size() == 3 && c["target"].size() == 3) {
            std::array<float, 9> s{};
            for (int k = 0; k < 3; k++) {
                s[k] = float(numOr(c["position"][k], 0));
                s[3 + k] = float(numOr(c["target"][k], 0));
                s[6 + k] = float(numOr(c["up"][k], k == 1 ? 1 : 0));
            }
            viewer->setCameraState(s);
        }
    });
    const bool hasResults = !lf.stored.empty();
    const QString tabWanted = QString::fromStdString(strOr(view["tab"]));
    setTab(QStringList{"part", "structural", "thermal", "airflow"}.contains(tabWanted) ? tabWanted : lf.stored.count("airflow") ? "airflow" : lf.stored.count("thermal") && !lf.stored.count("static") ? "thermal" : hasResults ? "structural" : "part");
    lf.hasResults = hasResults;
    lf.hasSetup = f.setup.has_value();
    loadedFile = lf;
    showBanner();
    psimOpenMs = double(t.elapsed());
    status(tr("Opened %1: %2.%3").arg(fileName, hasResults ? tr("results shown from the file, not computed here") : tr("part and setup"),
                                      problems.isEmpty() ? QString() : tr(" Could not restore: %1.").arg(problems.join("; "))),
           problems.isEmpty() ? "" : "warn");
    (void)tRestore;
    return true;
}

// ---------------------------------------------------------------- banner, info, re-run

void MainWindow::showBanner() {
    if (!banner_) return;
    qDeleteAll(banner_->findChildren<QWidget*>(QString(), Qt::FindDirectChildrenOnly));
    delete banner_->layout();
    if (!loadedFile) { banner_->hide(); return; }
    auto* l = new QHBoxLayout(banner_);
    l->setContentsMargins(10, 4, 10, 4);
    const LoadedFile& f = *loadedFile;
    auto* text = new QLabel(QString("<b>%1</b> · %2").arg(f.hasResults ? (f.edited ? tr("Results are from the file; the setup has changed since") : tr("Loaded from file, not computed here")) : tr("Opened from file"), f.name.toHtmlEscaped()), banner_);
    l->addWidget(text, 1);
    auto* info = new QPushButton(tr("File info"), banner_);
    connect(info, &QPushButton::clicked, this, [this] { showFileInfo(); });
    l->addWidget(info);
    if (f.hasSetup && f.hasResults) {
        auto* re = new QPushButton(tr("Re-run"), banner_);
        connect(re, &QPushButton::clicked, this, [this] { rerunLoaded(); });
        l->addWidget(re);
    }
    banner_->show();
}

void MainWindow::psimEdited() {
    if (!loadedFile || restoring || loadedFile->edited) return;
    loadedFile->edited = true;
    showBanner();
}

bool MainWindow::psimShowingResults() const {
    return loadedFile && loadedFile->hasResults && !structural->job && !thermal->job;
}

void MainWindow::rerunLoaded() {
    if (!loadedFile) return;
    loadedFile->rerun = true;
    banner_->hide();
    if (currentTab_ == "thermal") thermal->run();
    else if (currentTab_ == "airflow") airflow->rerun();
    else structural->runStudy();
}

void MainWindow::psimRunDone(const QString& kind, const std::map<QString, double>& now) {
    if (!loadedFile || restoring) return;
    const LoadedFile f = *loadedFile;
    loadedFile.reset();
    banner_->hide();
    auto it = f.stored.find(kind);
    if (!f.rerun || it == f.stored.end()) return;
    QStringList bits;
    for (const auto& [k, was] : it->second) {
        auto n = now.find(k);
        if (n == now.end() || !std::isfinite(was) || !std::isfinite(n->second)) continue;
        const double d = was != 0 ? (n->second - was) / was * 100 : 0;
        bits << QString("%1 %2 (file %3, %4)").arg(k, num(n->second), num(was), std::abs(d) < 0.05 ? tr("same") : QString("%1%2%").arg(d >= 0 ? "+" : "−").arg(std::abs(d), 0, 'f', 1));
    }
    if (!bits.isEmpty()) status(tr("Re-run finished. Against %1: %2.").arg(f.name, bits.join("; ")));
}

void MainWindow::showFileInfo() {
    if (!loadedFile) return;
    const LoadedFile& f = *loadedFile;
    const Value& info = f.info;
    QDialog d(this);
    d.setWindowTitle(f.name);
    auto* v = new QVBoxLayout(&d);
    QString html = QString("<h3>%1</h3><table>").arg(f.name.toHtmlEscaped());
    auto row = [&](const QString& k, const QString& val) { if (!val.isEmpty()) html += QString("<tr><td><b>%1</b>&nbsp;&nbsp;</td><td>%2</td></tr>").arg(k, val.toHtmlEscaped()); };
    auto S = [](const Value& x) { return QString::fromStdString(strOr(x)); };
    row(tr("Part"), S(info["name"]));
    row(tr("Notes"), S(info["notes"]));
    const Value& app = info["app"];
    if (app.type == Value::Object) row(tr("Made by"), QString("%1 %2 (%3)").arg(S(app["name"]), S(app["version"]), S(app["kind"])));
    row(tr("Created"), S(info["created"]));
    row(tr("Format"), tr("version %1").arg(f.version));
    row(tr("Units"), S(info["units"]));
    if (info["part"].type == Value::Object) row(tr("Part size"), tr("%1 vertices, %2 triangles").arg(QLocale().toString(qlonglong(numOr(info["part"]["vertices"], 0))), QLocale().toString(qlonglong(numOr(info["part"]["triangles"], 0)))));
    const Value& c = info["contains"];
    const QString geom = S(c["geometry"]);
    row(tr("Geometry"), geom == "exact" ? tr("exact") : geom.isEmpty() ? tr("not included") : tr("compact (16-bit positions)"));
    row(tr("Setup"), boolOr(c["setup"], false) ? tr("included") : tr("not included"));
    QStringList rs;
    for (const auto& r : c["results"].arr) rs << labelOf(S(r));
    row(tr("Results"), rs.isEmpty() ? tr("none") : rs.join(", "));
    row(tr("CAD source"), boolOr(c["cad"], false) ? (S(c["cadName"]).isEmpty() ? tr("included") : S(c["cadName"])) : tr("not included"));
    row(tr("File size"), kb(double(f.size)));
    html += "</table><h4>" + tr("Sections") + "</h4><table cellpadding=3>";
    static const std::map<std::string, QString> names = {{"INFO", "Info"}, {"THMB", "Preview picture"}, {"GEOM", "Geometry"}, {"CADS", "CAD source"}, {"SETP", "Setup"}, {"RFEA", "Structural and thermal results"}, {"RAIR", "Airflow results"}, {"VIEW", "View"}};
    for (const auto& e : f.table) {
        auto n = names.find(e.id);
        html += QString("<tr><td>%1</td><td>%2</td><td>%3</td></tr>").arg(n == names.end() ? QString::fromStdString(e.id) + " (unknown)" : n->second, kb(e.stored), kb(e.raw));
    }
    html += "</table><h4>" + tr("Largest error of each lossy field") + "</h4>";
    if (info["bounds"].arr.empty()) html += "<p>" + tr("Everything in this file is exact.") + "</p>";
    else {
        html += "<table cellpadding=3>";
        for (const auto& b : info["bounds"].arr) html += QString("<tr><td>%1</td><td>%2</td></tr>").arg(S(b["field"]).toHtmlEscaped(), num(numOr(b["absolute"], 0), 3));
        html += "</table>";
    }
    if (!f.skipped.empty()) {
        QStringList sk;
        for (const auto& s : f.skipped) sk << QString::fromStdString(s);
        html += "<p>" + tr("Sections this version does not know were skipped: %1.").arg(sk.join(", ")) + "</p>";
    }
    auto* label = new QLabel(html, &d);
    label->setTextFormat(Qt::RichText);
    label->setTextInteractionFlags(Qt::TextSelectableByMouse);
    v->addWidget(label);
    auto* bb = new QDialogButtonBox(QDialogButtonBox::Ok, &d);
    QObject::connect(bb, &QDialogButtonBox::accepted, &d, &QDialog::accept);
    v->addWidget(bb);
    d.exec();
}

}  // namespace ps
