#include "viewport.hpp"

#include <QFile>
#include <QMouseEvent>
#include <QPainter>
#include <QWheelEvent>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace ps {

namespace {

constexpr float NO_DATA = -3.0e38f;

// SolidWorks-style rainbow (blue low -> red high) and a colour-blind-safe sequential ramp
const float RAINBOW[5][4] = {{0, 0, 0, 255}, {0.25f, 0, 255, 255}, {0.5f, 0, 255, 0}, {0.75f, 255, 255, 0}, {1, 255, 0, 0}};
const char* HEAT_HEX[9] = {"#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"};

QVector3D ramp(bool heat, float t) {
    t = std::clamp(t, 0.f, 1.f);
    if (heat) {
        const float x = t * 8;
        const int i = std::min(7, int(x));
        const QColor a(HEAT_HEX[i]), b(HEAT_HEX[i + 1]);
        const float s = x - i;
        return QVector3D(a.redF() + s * (b.redF() - a.redF()), a.greenF() + s * (b.greenF() - a.greenF()), a.blueF() + s * (b.blueF() - a.blueF()));
    }
    for (int i = 1; i < 5; i++)
        if (t <= RAINBOW[i][0]) {
            const float s = (t - RAINBOW[i - 1][0]) / (RAINBOW[i][0] - RAINBOW[i - 1][0]);
            QVector3D c;
            for (int k = 0; k < 3; k++) c[k] = (RAINBOW[i - 1][k + 1] + s * (RAINBOW[i][k + 1] - RAINBOW[i - 1][k + 1])) / 255.f;
            return c;
        }
    return {1, 0, 0};
}

QShader loadShader(const QString& path) {
    QFile f(path);
    return f.open(QIODevice::ReadOnly) ? QShader::fromSerialized(f.readAll()) : QShader();
}

QVector3D anyPerpendicular(const QVector3D& d) {
    const QVector3D a = std::abs(d.x()) < 0.9f ? QVector3D(1, 0, 0) : QVector3D(0, 1, 0);
    return QVector3D::crossProduct(d, a).normalized();
}

}  // namespace

QVector3D colorRamp(bool heat, float t) { return ramp(heat, t); }

// ---------- geometry ----------

void Geometry::triangle(const QVector3D& a, const QVector3D& b, const QVector3D& c) {
    const QVector3D n = QVector3D::crossProduct(b - a, c - a).normalized();
    vertex(a, n);
    vertex(b, n);
    vertex(c, n);
}

namespace shapes {

static void cylinder(Geometry& g, const QVector3D& base, const QVector3D& axis, float len, float r0, float r1, int seg) {
    const QVector3D d = axis.normalized(), u = anyPerpendicular(d), v = QVector3D::crossProduct(d, u);
    const QVector3D top = base + d * len;
    for (int i = 0; i < seg; i++) {
        const float a0 = 2 * float(M_PI) * i / seg, a1 = 2 * float(M_PI) * (i + 1) / seg;
        const QVector3D e0 = u * std::cos(a0) + v * std::sin(a0), e1 = u * std::cos(a1) + v * std::sin(a1);
        const QVector3D p0 = base + e0 * r0, p1 = base + e1 * r0, q0 = top + e0 * r1, q1 = top + e1 * r1;
        // smooth side normals (slanted for cones)
        const float slope = (r0 - r1) / len;
        const QVector3D n0 = (e0 + d * slope).normalized(), n1 = (e1 + d * slope).normalized();
        g.vertex(p0, n0); g.vertex(p1, n1); g.vertex(q1, n1);
        if (r1 > 0) { g.vertex(p0, n0); g.vertex(q1, n1); g.vertex(q0, n0); }
        g.vertex(base, -d); g.vertex(p1, -d); g.vertex(p0, -d);
        if (r1 > 0) { g.vertex(top, d); g.vertex(q0, d); g.vertex(q1, d); }
    }
}

Geometry arrow(const QVector3D& tip, const QVector3D& dir, float length, const QVector3D& color, float radius) {
    Geometry g;
    g.color = color;
    if (radius <= 0) radius = length * 0.035f;
    const QVector3D d = dir.normalized();
    const float head = length * 0.28f;
    cylinder(g, tip - d * length, d, length - head, radius, radius, 12);
    cylinder(g, tip - d * head, d, head, radius * 2.6f, 0, 18);
    return g;
}

Geometry anchor(const QVector3D& point, const QVector3D& normal, float size, const QVector3D& color) {
    Geometry g;
    g.color = color;
    const QVector3D n = normal.normalized();
    // four-sided cone pointing into the surface
    cylinder(g, point + n * size, -n, size, size * 0.35f, 0, 4);
    return g;
}

Geometry sphere(const QVector3D& c, float r, const QVector3D& color, int seg) {
    Geometry g;
    g.color = color;
    const int rings = seg / 2;
    auto P = [&](int i, int j) {
        const float th = float(M_PI) * j / rings, ph = 2 * float(M_PI) * i / seg;
        return QVector3D(std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph));
    };
    for (int j = 0; j < rings; j++)
        for (int i = 0; i < seg; i++) {
            const QVector3D a = P(i, j), b = P(i + 1, j), cc = P(i + 1, j + 1), d = P(i, j + 1);
            for (const QVector3D& q : {a, cc, b, a, d, cc}) g.vertex(c + q * r, q);
        }
    return g;
}

Geometry patch(const Part& part, const std::vector<int32_t>& tris, const QVector3D& color, float opacity) {
    Geometry g;
    g.color = color;
    g.opacity = opacity;
    g.lit = false;
    const float off = float(part.bbox.diag * 0.0008);
    g.verts.reserve(tris.size() * 21);
    for (int32_t t : tris) {
        const QVector3D n(part.triNormal[3 * t], part.triNormal[3 * t + 1], part.triNormal[3 * t + 2]);
        for (int c = 0; c < 3; c++) {
            const uint32_t v = part.tris[3 * t + c];
            g.vertex(QVector3D(part.vertices[3 * v], part.vertices[3 * v + 1], part.vertices[3 * v + 2]) + n * off, n);
        }
    }
    return g;
}

Geometry ring(const QVector3D& c, const QVector3D& normal, float r, const QVector3D& color) {
    Geometry g;
    g.color = color;
    g.lit = false;
    g.onTop = true;
    g.opacity = 0.9f;
    const QVector3D n = normal.normalized(), u = anyPerpendicular(n), v = QVector3D::crossProduct(n, u);
    for (int i = 0; i < 48; i++) {
        const float a0 = 2 * float(M_PI) * i / 48, a1 = 2 * float(M_PI) * (i + 1) / 48;
        const QVector3D e0 = u * std::cos(a0) + v * std::sin(a0), e1 = u * std::cos(a1) + v * std::sin(a1);
        g.triangle(c + e0 * r * 0.92f, c + e0 * r, c + e1 * r);
        g.triangle(c + e0 * r * 0.92f, c + e1 * r, c + e1 * r * 0.92f);
    }
    return g;
}

Geometry cubes(const std::vector<float>& centers, float size, const QVector3D& color, float opacity) {
    Geometry g;
    g.color = color;
    g.opacity = opacity;
    const float s = size / 2;
    static const int F[6][4] = {{0, 3, 2, 1}, {4, 5, 6, 7}, {0, 1, 5, 4}, {2, 3, 7, 6}, {1, 2, 6, 5}, {0, 4, 7, 3}};
    g.verts.reserve(centers.size() / 3 * 36 * 7);
    for (size_t i = 0; i + 2 < centers.size(); i += 3) {
        const QVector3D c(centers[i], centers[i + 1], centers[i + 2]);
        QVector3D p[8];
        for (int k = 0; k < 8; k++) p[k] = c + QVector3D((k & 1) ? s : -s, (k & 2) ? s : -s, (k & 4) ? s : -s);
        // remap to the corner order used by F (0..3 bottom ccw, 4..7 top)
        const QVector3D q[8] = {p[0], p[1], p[3], p[2], p[4], p[5], p[7], p[6]};
        for (auto& f : F) {
            g.triangle(q[f[0]], q[f[1]], q[f[2]]);
            g.triangle(q[f[0]], q[f[2]], q[f[3]]);
        }
    }
    return g;
}

}  // namespace shapes

// ---------- labels drawn over the 3D view ----------

class LabelLayer : public QWidget {
public:
    explicit LabelLayer(Viewport* v) : QWidget(v), view_(v) {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        setAttribute(Qt::WA_NoSystemBackground);
    }
    void paintEvent(QPaintEvent*) override {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        // marker label
        if (view_->marker_) {
            bool visible = false;
            const QPointF s = view_->project(view_->marker_->p, &visible);
            if (visible) {
                QFont f = font();
                f.setBold(true);
                f.setPointSizeF(f.pointSizeF() * 0.95);
                p.setFont(f);
                const QRectF text = p.fontMetrics().boundingRect(view_->marker_->label);
                const QRectF box(s.x() + 14, s.y() - 14 - text.height() / 2 - 4, text.width() + 16, text.height() + 8);
                p.setPen(Qt::NoPen);
                p.setBrush(QColor(207, 38, 38, 235));
                p.drawRoundedRect(box, 5, 5);
                p.setPen(Qt::white);
                p.drawText(box, Qt::AlignCenter, view_->marker_->label);
            }
        }
        // axis letters of the orientation triad (bottom-left)
        const QMatrix4x4 view = view_->viewMatrix();
        const QPointF o(8 + 48, height() - 8 - 48);
        const QVector3D axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        const QColor colors[3] = {QColor(0xe5, 0x48, 0x4d), QColor(0x30, 0xa4, 0x6c), QColor(0x3e, 0x63, 0xdd)};
        const char* names[3] = {"X", "Y", "Z"};
        QFont f = font();
        f.setBold(true);
        p.setFont(f);
        for (int a = 0; a < 3; a++) {
            const QVector3D d = view.mapVector(axes[a]);
            const QPointF end = o + QPointF(d.x(), -d.y()) * 34;
            p.setPen(QPen(colors[a], 2.4, Qt::SolidLine, Qt::RoundCap));
            p.drawLine(o, end);
            p.setPen(colors[a]);
            const QPointF lp = o + QPointF(d.x(), -d.y()) * 44;
            p.drawText(QRectF(lp.x() - 8, lp.y() - 8, 16, 16), Qt::AlignCenter, names[a]);
        }
    }

private:
    Viewport* view_;
};

// ---------- viewport ----------

Viewport::Viewport(QWidget* parent) : QRhiWidget(parent) {
    setMouseTracking(true);
    setFocusPolicy(Qt::StrongFocus);
    setSampleCount(4);
#if defined(Q_OS_MACOS)
    setApi(QRhiWidget::Api::Metal);
#elif defined(Q_OS_WIN)
    setApi(QRhiWidget::Api::Direct3D11);
#endif
    labels_ = new LabelLayer(this);
    partItem_ = std::make_unique<Item>();
    edgeItem_ = std::make_unique<Item>();
    gridItem_ = std::make_unique<Item>();
    rebuildGrid();
    // orientation triad is drawn by the label layer (2D), the frame timer drives animations
    clock_.start();
    connect(&timer_, &QTimer::timeout, this, &Viewport::tick);
    timer_.start(16);
}

Viewport::~Viewport() = default;

void Viewport::tick() {
    static qint64 last = clock_.elapsed();
    const qint64 now = clock_.elapsed();
    const double dt = std::min(0.1, (now - last) / 1000.0);
    last = now;
    emit frame(dt);
    if (marker_) update();  // pulsing halo
}

void Viewport::resizeEvent(QResizeEvent* e) {
    QRhiWidget::resizeEvent(e);
    labels_->setGeometry(rect());
}

void Viewport::setPart(std::shared_ptr<Part> part) {
    part_ = std::move(part);
    hasScalars_ = hasDisp_ = false;
    scalars_.clear();
    disp_.clear();
    layers_.clear();
    marker_.reset();
    pickDirty_ = true;
    sceneSize_ = float(part_->bbox.diag);
    rebuildPartVertices();
    rebuildEdges();
    rebuildGrid();
    setView("iso");
}

void Viewport::clearPart() {
    part_.reset();
    layers_.clear();
    marker_.reset();
    partItem_->g.verts.clear();
    partItem_->dirty = true;
    edgeItem_->g.verts.clear();
    edgeItem_->dirty = true;
    update();
}

void Viewport::setScalars(const std::vector<float>* values, const ColorSpec& spec) {
    hasScalars_ = values != nullptr;
    if (values) scalars_ = *values;
    spec_ = spec;
    rebuildPartVertices();
}

void Viewport::setDeformation(const std::vector<float>* disp, double scale) {
    hasDisp_ = disp != nullptr && scale != 0;
    if (hasDisp_) disp_ = *disp;
    dispScale_ = scale;
    pickDirty_ = true;
    rebuildPartVertices();
    rebuildEdges();
}

void Viewport::setEdgesVisible(bool on) {
    edges_ = on;
    update();
}

void Viewport::setXRay(bool on) {
    xray_ = on;
    update();
}

void Viewport::setPartVisible(bool on) {
    partVisible_ = on;
    update();
}

void Viewport::rebuildPartVertices() {
    Item& it = *partItem_;
    it.g = Geometry{};
    it.g.color = QVector3D(0xc9 / 255.f, 0xd2 / 255.f, 0xdc / 255.f);
    it.scalar = hasScalars_;
    it.spec = spec_;
    if (!part_) { it.dirty = true; update(); return; }
    const Part& p = *part_;
    const size_t n = p.displaySrc.size();
    it.g.verts.resize(n * 7);
    const bool def = hasDisp_ && disp_.size() == size_t(3 * p.nVert);
    const bool sc = hasScalars_ && scalars_.size() == size_t(p.nVert);
    for (size_t i = 0; i < n; i++) {
        const uint32_t v = p.displaySrc[i];
        float* o = &it.g.verts[7 * i];
        for (int d = 0; d < 3; d++) {
            float x = p.displayPosition[3 * i + d];
            if (def) {
                const float dv = disp_[3 * v + d];
                if (std::isfinite(dv)) x += float(dispScale_ * dv);
            }
            o[d] = x;
            o[3 + d] = p.displayNormal[3 * i + d];
        }
        o[6] = sc ? (std::isnan(scalars_[v]) ? NO_DATA : scalars_[v]) : 0.f;
    }
    it.dirty = true;
    update();
}

void Viewport::rebuildEdges() {
    Item& it = *edgeItem_;
    it.g = Geometry{};
    it.g.kind = Geometry::Lines;
    it.g.color = QVector3D(0x1b / 255.f, 0x23 / 255.f, 0x30 / 255.f);
    it.g.opacity = 0.55f;
    it.g.lit = false;
    if (part_) {
        const Part& p = *part_;
        const bool def = hasDisp_ && disp_.size() == size_t(3 * p.nVert);
        it.g.verts.reserve(p.edges.size() * 7);
        for (uint32_t v : p.edges) {
            QVector3D q(p.vertices[3 * v], p.vertices[3 * v + 1], p.vertices[3 * v + 2]);
            if (def) {
                const QVector3D d(disp_[3 * v], disp_[3 * v + 1], disp_[3 * v + 2]);
                if (std::isfinite(d.x())) q += d * float(dispScale_);
            }
            it.g.vertex(q);
        }
    }
    it.dirty = true;
    update();
}

void Viewport::rebuildGrid() {
    Item& it = *gridItem_;
    it.g = Geometry{};
    it.g.kind = Geometry::Lines;
    it.g.lit = false;
    it.g.color = QVector3D(0x8b / 255.f, 0x98 / 255.f, 0xa8 / 255.f);
    it.g.opacity = 0.45f;
    const float size = part_ ? float(std::pow(10, std::ceil(std::log10(part_->bbox.diag * 2)))) : 400;
    const float y = part_ ? float(-part_->bbox.diag * 0.002) : 0;
    const int div = 20;
    for (int i = 0; i <= div; i++) {
        const float s = -size / 2 + size * i / div;
        it.g.vertex({s, y, -size / 2});
        it.g.vertex({s, y, size / 2});
        it.g.vertex({-size / 2, y, s});
        it.g.vertex({size / 2, y, s});
    }
    it.dirty = true;
}

void Viewport::setLayer(const std::string& name, std::vector<Geometry> items) {
    auto& list = layers_[name];
    list.clear();
    for (auto& g : items) {
        if (g.verts.empty()) continue;
        auto it = std::make_unique<Item>();
        it->g = std::move(g);
        list.push_back(std::move(it));
    }
    update();
}

void Viewport::clearLayer(const std::string& name) {
    layers_.erase(name);
    if (name == "overlays") handles_.clear();
    update();
}

void Viewport::setMarker(const QVector3D& p, const QString& label) {
    marker_ = Marker{p, label};
    labels_->update();
    update();
}

void Viewport::moveMarker(const QVector3D& p) {
    if (marker_) marker_->p = p;
    update();
}

void Viewport::clearMarker() {
    marker_.reset();
    labels_->update();
    update();
}

// ---------- camera ----------

QMatrix4x4 Viewport::viewMatrix() const {
    QMatrix4x4 m;
    m.lookAt(eye_, target_, up_);
    return m;
}

QMatrix4x4 Viewport::projMatrix() const {
    QMatrix4x4 m;
    const float aspect = height() > 0 ? float(width()) / float(height()) : 1.f;
    m.perspective(fov_, aspect, sceneSize_ / 1000.f, sceneSize_ * 200.f);
    return m;
}

void Viewport::setView(const QString& name) {
    if (!part_) return;
    const BBox& b = part_->bbox;
    const QVector3D center(float((b.min[0] + b.max[0]) / 2), float((b.min[1] + b.max[1]) / 2), float((b.min[2] + b.max[2]) / 2));
    const float radius = float(b.diag / 2);
    const float vertical = qDegreesToRadians(fov_ / 2);
    const float aspect = height() > 0 ? float(width()) / float(height()) : 1.f;
    const float horizontal = std::atan(std::tan(vertical) * aspect);
    const float dist = radius / std::sin(std::min(vertical, horizontal)) * 1.08f;
    QVector3D d(1, 0.75f, 1.25f);
    up_ = QVector3D(0, 1, 0);
    if (name == "front") d = {0, 0, 1};
    else if (name == "back") d = {0, 0, -1};
    else if (name == "right") d = {1, 0, 0};
    else if (name == "left") d = {-1, 0, 0};
    else if (name == "top") { d = {0, 1, 0}; up_ = {0, 0, -1}; }
    else if (name == "bottom") { d = {0, -1, 0}; up_ = {0, 0, 1}; }
    target_ = center;
    eye_ = center + d.normalized() * dist;
    labels_->update();
    update();
}

void Viewport::orbit(float dx, float dy) {
    QVector3D off = eye_ - target_;
    const QVector3D worldUp(0, 1, 0);
    // yaw about the world up axis, pitch about the camera's right axis (clamped short of the poles)
    QMatrix4x4 yaw;
    yaw.rotate(-dx * 0.4f, worldUp);
    off = yaw.map(off);
    up_ = yaw.mapVector(up_);
    const QVector3D right = QVector3D::crossProduct(up_, off).normalized();
    QMatrix4x4 pitch;
    pitch.rotate(-dy * 0.4f, right);
    const QVector3D next = pitch.map(off);
    const float cosUp = QVector3D::dotProduct(next.normalized(), worldUp);
    if (std::abs(cosUp) < 0.995f) {
        off = next;
        up_ = QVector3D::crossProduct(off, right).normalized();
    }
    if (std::abs(QVector3D::dotProduct(up_, worldUp)) > 0.01f && std::abs(QVector3D::dotProduct(off.normalized(), worldUp)) < 0.995f) {
        // keep the horizon level
        const QVector3D r = QVector3D::crossProduct(worldUp, off).normalized();
        up_ = QVector3D::crossProduct(off, r).normalized();
        if (QVector3D::dotProduct(up_, worldUp) < 0) up_ = -up_;
    }
    eye_ = target_ + off;
    labels_->update();
    update();
}

void Viewport::pan(float dx, float dy) {
    const QVector3D off = eye_ - target_;
    const float dist = off.length();
    const QVector3D right = QVector3D::crossProduct(up_, off).normalized();
    const QVector3D up = QVector3D::crossProduct(off, right).normalized();
    const float scale = 2 * dist * std::tan(qDegreesToRadians(fov_ / 2)) / std::max(1, height());
    const QVector3D move = (-right * dx + up * dy) * scale;
    target_ += move;
    eye_ += move;
    update();
}

QPointF Viewport::project(const QVector3D& p, bool* visible) const {
    const QVector4D c = projMatrix() * viewMatrix() * QVector4D(p, 1);
    if (visible) *visible = c.w() > 0 && std::abs(c.z() / c.w()) < 1;
    if (c.w() == 0) return {};
    return QPointF((c.x() / c.w() + 1) / 2 * width(), (1 - c.y() / c.w()) / 2 * height());
}

void Viewport::cameraRay(const QPointF& pos, QVector3D& origin, QVector3D& dir) const {
    const QMatrix4x4 inv = (projMatrix() * viewMatrix()).inverted();
    const float x = float(2 * pos.x() / std::max(1, width()) - 1), y = float(1 - 2 * pos.y() / std::max(1, height()));
    const QVector4D a = inv * QVector4D(x, y, -1, 1), b = inv * QVector4D(x, y, 1, 1);
    origin = a.toVector3D() / a.w();
    dir = (b.toVector3D() / b.w() - origin).normalized();
}

QVector3D Viewport::pointOnPlane(const QPointF& pos, const QVector3D& pivot) const {
    QVector3D o, d;
    cameraRay(pos, o, d);
    const QVector3D n = (target_ - eye_).normalized();
    const float denom = QVector3D::dotProduct(n, d);
    if (std::abs(denom) < 1e-9f) return pivot;
    const float t = QVector3D::dotProduct(pivot - o, n) / denom;
    return o + d * t;
}

const BVH& Viewport::pickBVH() {
    if (pickDirty_ || !pickBVH_) {
        const Part& p = *part_;
        pickVerts_ = p.vertices;
        if (hasDisp_ && disp_.size() == pickVerts_.size())
            for (size_t i = 0; i < pickVerts_.size(); i++)
                if (std::isfinite(disp_[i])) pickVerts_[i] += float(dispScale_ * disp_[i]);
        pickBVH_ = std::make_unique<BVH>(pickVerts_, p.tris);
        pickDirty_ = false;
    }
    return *pickBVH_;
}

std::optional<PickHit> Viewport::pickPart(const QPointF& pos) {
    if (!part_ || !partVisible_) return std::nullopt;
    QVector3D o, d;
    cameraRay(pos, o, d);
    const double od[3] = {o.x(), o.y(), o.z()}, dd[3] = {d.x(), d.y(), d.z()};
    RayHit h;
    if (!pickBVH().raycast(od, dd, 0, 1e30, h)) return std::nullopt;
    const Part& p = *part_;
    PickHit hit;
    hit.tri = h.tri;
    hit.bary[0] = 1 - h.u - h.v;
    hit.bary[1] = h.u;
    hit.bary[2] = h.v;
    hit.point = o + d * float(h.distance);
    for (int k = 0; k < 3; k++) {
        const uint32_t v = p.tris[3 * h.tri + k];
        hit.rest += QVector3D(p.vertices[3 * v], p.vertices[3 * v + 1], p.vertices[3 * v + 2]) * float(hit.bary[k]);
    }
    hit.normal = QVector3D(p.triNormal[3 * h.tri], p.triNormal[3 * h.tri + 1], p.triNormal[3 * h.tri + 2]);
    hit.face = p.faceOf[h.tri];
    return hit;
}

QImage Viewport::screenshot() { return grabFramebuffer(); }

// ---------- input ----------

void Viewport::mousePressEvent(QMouseEvent* e) {
    pressPos_ = lastPos_ = e->position();
    pressButton_ = e->button();
    if (e->button() == Qt::LeftButton && !handles_.empty()) {
        QVector3D o, d;
        cameraRay(e->position(), o, d);
        for (size_t i = 0; i < handles_.size(); i++) {
            const QVector3D oc = o - handles_[i].center;
            const float b = QVector3D::dotProduct(oc, d), c = oc.lengthSquared() - handles_[i].radius * handles_[i].radius * 2.25f;
            if (b * b - c >= 0) { activeHandle_ = int(i); drag_ = Drag::Handle; return; }
        }
    }
    const bool panButton = e->button() == Qt::RightButton || e->button() == Qt::MiddleButton || (e->modifiers() & Qt::ShiftModifier);
    drag_ = panButton ? Drag::Pan : Drag::Orbit;
}

void Viewport::mouseMoveEvent(QMouseEvent* e) {
    const QPointF delta = e->position() - lastPos_;
    lastPos_ = e->position();
    if (drag_ == Drag::Handle && activeHandle_ >= 0) {
        auto& h = handles_[activeHandle_];
        if (h.onDrag) h.onDrag(pointOnPlane(e->position(), h.pivot));
        return;
    }
    if (e->buttons() && drag_ == Drag::Orbit) { orbit(float(delta.x()), float(delta.y())); return; }
    if (e->buttons() && drag_ == Drag::Pan) { pan(float(delta.x()), float(delta.y())); return; }
    if (!e->buttons() && onHover) onHover(e);
}

void Viewport::mouseReleaseEvent(QMouseEvent* e) {
    if (drag_ == Drag::Handle && activeHandle_ >= 0) {
        auto end = handles_[activeHandle_].onEnd;
        activeHandle_ = -1;
        drag_ = Drag::None;
        if (end) end();
        return;
    }
    const bool click = QLineF(e->position(), pressPos_).length() < 5;
    drag_ = Drag::None;
    if (click && onClick) onClick(e);
}

void Viewport::wheelEvent(QWheelEvent* e) {
    const float steps = e->angleDelta().y() / 120.f + e->pixelDelta().y() / 60.f;
    const float k = std::pow(0.88f, steps);
    const QVector3D off = eye_ - target_;
    const float len = std::clamp(off.length() * k, sceneSize_ * 0.02f, sceneSize_ * 50);
    eye_ = target_ + off.normalized() * len;
    update();
}

void Viewport::leaveEvent(QEvent*) {
    if (onLeave) onLeave();
}

// ---------- rendering ----------

void Viewport::initialize(QRhiCommandBuffer*) {
    if (rhi_ != rhi()) {
        pipelines_.clear();
        rhi_ = rhi();
        vs_ = loadShader(":/shaders/mesh.vert.qsb");
        fs_ = loadShader(":/shaders/mesh.frag.qsb");
        sampler_.reset(rhi_->newSampler(QRhiSampler::Linear, QRhiSampler::Linear, QRhiSampler::None, QRhiSampler::ClampToEdge, QRhiSampler::ClampToEdge));
        sampler_->create();
        for (auto* tex : {&rainbow_, &heat_}) {
            tex->reset(rhi_->newTexture(QRhiTexture::RGBA8, QSize(256, 1)));
            (*tex)->create();
        }
        layoutUbuf_.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 128));
        layoutUbuf_->create();
        layoutSrb_.reset(rhi_->newShaderResourceBindings());
        layoutSrb_->setBindings({QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage, layoutUbuf_.get()),
                                 QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, rainbow_.get(), sampler_.get())});
        layoutSrb_->create();
        // every item's GPU resources belong to the old QRhi
        auto reset = [](Item& it) { it.vbuf.reset(); it.ubuf.reset(); it.srb.reset(); it.srbTex = nullptr; it.capacity = 0; it.dirty = true; };
        for (Item* it : {partItem_.get(), edgeItem_.get(), gridItem_.get()}) reset(*it);
        for (auto& [n, list] : layers_)
            for (auto& it : list) reset(*it);
        texturesUploaded_ = false;
        markerItem_.reset();
        markerHalo_.reset();
    }
    if (rpDesc_ != renderTarget()->renderPassDescriptor() || sampleCount_ != sampleCount()) {
        pipelines_.clear();
        rpDesc_ = renderTarget()->renderPassDescriptor();
        sampleCount_ = sampleCount();
    }
}

QRhiGraphicsPipeline* Viewport::pipeline(Geometry::Kind kind, bool blend, bool depthTest, bool depthWrite, bool bias) {
    const int key = (kind == Geometry::Lines) | (blend << 1) | (depthTest << 2) | (depthWrite << 3) | (bias << 4);
    auto& ps = pipelines_[key];
    if (ps) return ps.get();
    ps.reset(rhi_->newGraphicsPipeline());
    ps->setTopology(kind == Geometry::Lines ? QRhiGraphicsPipeline::Lines : QRhiGraphicsPipeline::Triangles);
    ps->setCullMode(QRhiGraphicsPipeline::None);
    ps->setDepthTest(depthTest);
    ps->setDepthWrite(depthWrite);
    ps->setDepthOp(QRhiGraphicsPipeline::LessOrEqual);
    if (bias) {
        ps->setDepthBias(1);
        ps->setSlopeScaledDepthBias(1.0f);
    }
    QRhiGraphicsPipeline::TargetBlend b;
    b.enable = blend;
    b.srcColor = QRhiGraphicsPipeline::One;
    b.dstColor = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    b.srcAlpha = QRhiGraphicsPipeline::One;
    b.dstAlpha = QRhiGraphicsPipeline::OneMinusSrcAlpha;
    ps->setTargetBlends({b});
    ps->setSampleCount(sampleCount_);
    ps->setShaderStages({{QRhiShaderStage::Vertex, vs_}, {QRhiShaderStage::Fragment, fs_}});
    QRhiVertexInputLayout layout;
    layout.setBindings({{7 * sizeof(float)}});
    layout.setAttributes({{0, 0, QRhiVertexInputAttribute::Float3, 0},
                          {0, 1, QRhiVertexInputAttribute::Float3, 3 * sizeof(float)},
                          {0, 2, QRhiVertexInputAttribute::Float, 6 * sizeof(float)}});
    ps->setVertexInputLayout(layout);
    ps->setShaderResourceBindings(layoutSrb_.get());
    ps->setRenderPassDescriptor(rpDesc_);
    ps->create();
    return ps.get();
}

void Viewport::ensureItem(Item& it, QRhiResourceUpdateBatch* u) {
    const size_t bytes = it.g.verts.size() * sizeof(float);
    if (!it.ubuf) {
        it.ubuf.reset(rhi_->newBuffer(QRhiBuffer::Dynamic, QRhiBuffer::UniformBuffer, 128));
        it.ubuf->create();
    }
    if (!it.dirty && it.vbuf) return;
    if (!bytes) { it.dirty = false; return; }
    if (!it.vbuf || it.capacity < bytes) {
        it.vbuf.reset(rhi_->newBuffer(QRhiBuffer::Static, QRhiBuffer::VertexBuffer, quint32(bytes)));
        it.vbuf->create();
        it.capacity = bytes;
    }
    u->uploadStaticBuffer(it.vbuf.get(), 0, quint32(bytes), it.g.verts.data());
    it.dirty = false;
}

void Viewport::drawItem(QRhiCommandBuffer* cb, Item& it, const QMatrix4x4& mvp, const QVector3D& toViewer, QRhiResourceUpdateBatch*, int) {
    const quint32 n = quint32(it.g.count());
    if (!n || !it.vbuf) return;
    QRhiTexture* tex = it.scalar && it.spec.heat ? heat_.get() : rainbow_.get();
    if (!it.srb || it.srbTex != tex) {
        it.srbTex = tex;
        it.srb.reset(rhi_->newShaderResourceBindings());
        it.srb->setBindings({QRhiShaderResourceBinding::uniformBuffer(0, QRhiShaderResourceBinding::VertexStage | QRhiShaderResourceBinding::FragmentStage, it.ubuf.get()),
                             QRhiShaderResourceBinding::sampledTexture(1, QRhiShaderResourceBinding::FragmentStage, tex, sampler_.get())});
        it.srb->create();
    }
    const bool isPart = &it == partItem_.get();
    const bool blend = it.g.opacity < 0.999f || (isPart && xray_);
    const float opacity = isPart && xray_ ? (it.scalar ? 0.45f : 0.28f) : it.g.opacity;
    QRhiGraphicsPipeline* ps = pipeline(it.g.kind, blend, it.g.depthTest && !it.g.onTop, !blend && !it.g.onTop, isPart);
    cb->setGraphicsPipeline(ps);
    cb->setViewport({0, 0, float(renderTarget()->pixelSize().width()), float(renderTarget()->pixelSize().height())});
    cb->setShaderResources(it.srb.get());
    const QRhiCommandBuffer::VertexInput vin(it.vbuf.get(), 0);
    cb->setVertexInput(0, 1, &vin);
    cb->draw(n);
    (void)mvp;
    (void)toViewer;
}

void Viewport::render(QRhiCommandBuffer* cb) {
    QRhiResourceUpdateBatch* u = rhi_->nextResourceUpdateBatch();
    if (!texturesUploaded_) {
        for (int h = 0; h < 2; h++) {
            QImage img(256, 1, QImage::Format_RGBA8888);
            for (int i = 0; i < 256; i++) {
                const QVector3D c = ramp(h == 1, i / 255.f);
                img.setPixelColor(i, 0, QColor::fromRgbF(c.x(), c.y(), c.z()));
            }
            u->uploadTexture((h ? heat_ : rainbow_).get(), img);
        }
        texturesUploaded_ = true;
    }
    const QMatrix4x4 mvp = rhi_->clipSpaceCorrMatrix() * projMatrix() * viewMatrix();
    const QVector3D toViewer = (eye_ - target_).normalized();
    // marker sphere and pulsing halo
    if (marker_) {
        const float r = sceneSize_ * 0.012f;
        const float pulse = 1 + 0.6f * (0.5f + 0.5f * std::sin(clock_.elapsed() / 220.f));
        auto mk = [&](std::unique_ptr<Item>& it, float radius, float opacity) {
            if (!it) it = std::make_unique<Item>();
            it->g = shapes::sphere(marker_->p, radius, {1, 0.165f, 0.165f}, 20);
            it->g.lit = false;
            it->g.onTop = true;
            it->g.opacity = opacity;
            it->dirty = true;
        };
        mk(markerItem_, r, 0.95f);
        mk(markerHalo_, r * pulse * 1.6f, 0.35f);
    }
    // collect items in draw order
    std::vector<Item*> opaque, transparent, top;
    auto classify = [&](Item* it) {
        if (it->g.verts.empty()) return;
        const bool isPart = it == partItem_.get();
        if (it->g.onTop) top.push_back(it);
        else if (it->g.opacity < 0.999f || (isPart && xray_)) transparent.push_back(it);
        else opaque.push_back(it);
    };
    classify(gridItem_.get());
    if (part_ && partVisible_) classify(partItem_.get());
    if (part_ && edges_) classify(edgeItem_.get());
    for (auto& [n, list] : layers_)
        for (auto& it : list) classify(it.get());
    if (marker_) {
        classify(markerHalo_.get());
        classify(markerItem_.get());
    }
    auto prepare = [&](Item* it) {
        ensureItem(*it, u);
        const bool isPart = it == partItem_.get();
        const float opacity = isPart && xray_ ? (it->scalar ? 0.45f : 0.28f) : it->g.opacity;
        struct { float mvp[16]; float color[4]; float light[4]; float range[4]; } ub;
        std::memcpy(ub.mvp, mvp.constData(), sizeof(ub.mvp));
        ub.color[0] = it->g.color.x(); ub.color[1] = it->g.color.y(); ub.color[2] = it->g.color.z(); ub.color[3] = opacity;
        ub.light[0] = toViewer.x(); ub.light[1] = toViewer.y(); ub.light[2] = toViewer.z(); ub.light[3] = it->g.lit ? 1 : 0;
        ub.range[0] = it->spec.min; ub.range[1] = it->spec.max; ub.range[2] = float(it->spec.bands);
        ub.range[3] = float((it->scalar ? 1 : 0) | (it->spec.reverse ? 2 : 0) | (it->g.vertexColors ? 4 : 0));
        u->updateDynamicBuffer(it->ubuf.get(), 0, sizeof(ub), &ub);
    };
    for (Item* it : opaque) prepare(it);
    for (Item* it : transparent) prepare(it);
    for (Item* it : top) prepare(it);
    const QColor clear = palette().color(QPalette::Base).lightness() < 128 ? QColor(0x1f, 0x24, 0x2b) : QColor(0xe4, 0xe9, 0xef);
    cb->beginPass(renderTarget(), clear, {1.0f, 0}, u);
    for (Item* it : opaque) drawItem(cb, *it, mvp, toViewer, nullptr, 0);
    for (Item* it : transparent) drawItem(cb, *it, mvp, toViewer, nullptr, 1);
    for (Item* it : top) drawItem(cb, *it, mvp, toViewer, nullptr, 2);
    cb->endPass();
    labels_->update();
}

void Viewport::releaseResources() {
    pipelines_.clear();
    auto reset = [](Item& it) { it.vbuf.reset(); it.ubuf.reset(); it.srb.reset(); it.srbTex = nullptr; it.capacity = 0; it.dirty = true; };
    for (Item* it : {partItem_.get(), edgeItem_.get(), gridItem_.get()}) reset(*it);
    for (auto& [n, list] : layers_)
        for (auto& it : list) reset(*it);
    markerItem_.reset();
    markerHalo_.reset();
    layoutSrb_.reset();
    layoutUbuf_.reset();
    rainbow_.reset();
    heat_.reset();
    sampler_.reset();
    rhi_ = nullptr;
}

}  // namespace ps
