#include "picker.hpp"

#include <QHBoxLayout>
#include <QLabel>
#include <QMouseEvent>
#include <QPushButton>
#include <QSlider>

#include "mainwindow.hpp"
#include "viewport.hpp"
#include "widgets.hpp"

namespace ps {

static QVector3D rgb(const QColor& c) { return QVector3D(float(c.redF()), float(c.greenF()), float(c.blueF())); }

Picker::Picker(MainWindow* app, QWidget* parent) : QFrame(parent), app_(app) {
    setObjectName("picker");
    setAutoFillBackground(true);
    auto* h = new QHBoxLayout(this);
    h->setContentsMargins(10, 6, 10, 6);
    title_ = new QLabel(this);
    QFont f = title_->font();
    f.setBold(true);
    title_->setFont(f);
    h->addWidget(title_);
    modeSeg_ = segmented({tr("Face"), tr("Brush")}, 0, [this](int i) { setMode(i == 1); }, this);
    h->addWidget(modeSeg_);
    radiusRow_ = new QWidget(this);
    auto* rr = new QHBoxLayout(radiusRow_);
    rr->setContentsMargins(0, 0, 0, 0);
    rr->addWidget(new QLabel(tr("Radius"), radiusRow_));
    radiusSlider_ = new QSlider(Qt::Horizontal, radiusRow_);
    radiusSlider_->setRange(1, 30);
    radiusSlider_->setValue(5);
    radiusSlider_->setFixedWidth(90);
    rr->addWidget(radiusSlider_);
    radiusRow_->hide();
    h->addWidget(radiusRow_);
    auto* done = new QPushButton(tr("Done"), this);
    done->setDefault(true);
    auto* cancel = new QPushButton(tr("Cancel"), this);
    connect(done, &QPushButton::clicked, this, [this] { finish(true); });
    connect(cancel, &QPushButton::clicked, this, [this] { finish(false); });
    h->addWidget(done);
    h->addWidget(cancel);
    hide();
}

double Picker::radius() const { return app_->part ? app_->part->bbox.diag * radiusSlider_->value() / 100.0 : 1; }

const std::vector<int32_t>& Picker::faceTris(int face) {
    const Part& p = *app_->part;
    if (indexedPart_ != &p || indexedFaces_ != p.faceCount) {
        faces_.assign(p.faceCount, {});
        for (int t = 0; t < p.nTri; t++) faces_[p.faceOf[t]].push_back(t);
        indexedPart_ = &p;
        indexedFaces_ = p.faceCount;
    }
    static const std::vector<int32_t> none;
    return face >= 0 && face < int(faces_.size()) ? faces_[face] : none;
}

void Picker::setMode(bool brush) {
    brush_ = brush;
    radiusRow_->setVisible(brush);
    hoverFace_ = -1;
    app_->viewer->clearLayer("hover");
    adjustSize();
    updateHint();
}

void Picker::start(Session s) {
    if (session_) finish(false);
    session_ = std::move(s);
    title_->setText(session_->title);
    setMode(brush_);
    show();
    raise();
    adjustSize();
    if (parentWidget()) QFrame::move((parentWidget()->width() - width()) / 2, 12);
    app_->viewer->setCursor(Qt::CrossCursor);
}

void Picker::updateHint() {
    if (!session_) return;
    const QString what = brush_ ? tr("Click to paint circular areas") : tr("Click faces to add them");
    app_->hint(tr("%1. Selected: %2. Press Enter or Done to finish, Esc to cancel.").arg(what).arg(session_->count));
}

void Picker::move(QMouseEvent* e) {
    if (!session_ || !app_->part) return;
    auto hit = app_->viewer->pickPart(e->position());
    if (!hit) {
        hoverFace_ = -1;
        app_->viewer->clearLayer("hover");
        return;
    }
    if (!brush_) {
        if (hit->face == hoverFace_) return;
        hoverFace_ = hit->face;
        app_->viewer->setLayer("hover", {shapes::patch(*app_->part, faceTris(hit->face), rgb(session_->color), 0.45f)});
    } else {
        auto ring = shapes::ring(hit->rest + hit->normal * float(app_->part->bbox.diag * 0.001), hit->normal, float(radius()), rgb(session_->color));
        app_->viewer->setLayer("hover", {std::move(ring)});
    }
}

void Picker::click(QMouseEvent* e) {
    if (!session_ || !app_->part || e->button() != Qt::LeftButton) return;
    auto hit = app_->viewer->pickPart(e->position());
    if (!hit) return;
    Patch patch;
    if (!brush_) patch.tris = faceTris(hit->face);
    else {
        const Vec3 c{hit->rest.x(), hit->rest.y(), hit->rest.z()};
        patch.tris = trianglesInSphere(*app_->part, hit->tri, c, radius());
        patch.clip = Clip{c, radius()};
    }
    if (patch.tris.empty()) return;
    session_->count++;
    session_->onPatch(patch);
    hoverFace_ = -1;
    updateHint();
}

void Picker::finish(bool commit) {
    if (!session_) return;
    Session s = std::move(*session_);
    session_.reset();
    hide();
    app_->viewer->unsetCursor();
    app_->viewer->clearLayer("hover");
    app_->hint({});
    s.onDone(commit);
}

}  // namespace ps
