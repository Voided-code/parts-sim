// 3D viewport: part display, result colouring, deformation, overlays, markers and picking.
// Drawn through Qt's RHI, which uses Metal on macOS, Direct3D on Windows and Vulkan or OpenGL
// on Linux.
#pragma once

#include <array>

#include <QElapsedTimer>
#include <QMatrix4x4>
#include <QRhiWidget>
#include <QTimer>
#include <QVector3D>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <rhi/qrhi.h>
#include <vector>

#include "core/bvh.hpp"
#include "core/mesh.hpp"

namespace ps {

/** Triangles or lines in world space, 7 floats per vertex: position, normal, value. */
struct Geometry {
    enum Kind { Triangles, Lines } kind = Triangles;
    std::vector<float> verts;
    QVector3D color{0.8f, 0.8f, 0.8f};
    float opacity = 1;
    bool lit = true;
    bool depthTest = true;
    bool onTop = false;  // drawn last, ignoring depth (markers, brush ring)
    bool vertexColors = false;  // colour per vertex: rgb in the normal slot, alpha in the value slot (unlit)

    void vertex(const QVector3D& p, const QVector3D& n = {}, float value = 0) {
        verts.insert(verts.end(), {p.x(), p.y(), p.z(), n.x(), n.y(), n.z(), value});
    }
    void triangle(const QVector3D& a, const QVector3D& b, const QVector3D& c);
    size_t count() const { return verts.size() / 7; }
};

namespace shapes {
Geometry arrow(const QVector3D& tip, const QVector3D& dir, float length, const QVector3D& color, float radius = -1);
Geometry anchor(const QVector3D& point, const QVector3D& normal, float size, const QVector3D& color);
Geometry sphere(const QVector3D& c, float r, const QVector3D& color, int seg = 24);
Geometry patch(const Part& part, const std::vector<int32_t>& tris, const QVector3D& color, float opacity);
Geometry ring(const QVector3D& c, const QVector3D& normal, float r, const QVector3D& color);
Geometry cubes(const std::vector<float>& centers, float size, const QVector3D& color, float opacity = 1);
}  // namespace shapes

/** The colour ramps the viewport uses (rainbow, or the heat ramp), t in [0, 1]. */
QVector3D colorRamp(bool heat, float t);

struct ColorSpec {
    float min = 0, max = 1;
    int bands = 0;
    bool reverse = false;
    bool heat = false;  // colour-blind-safe ramp instead of the rainbow
};

struct PickHit {
    int tri = -1;
    QVector3D point, rest, normal;
    double bary[3] = {0, 0, 0};
    int face = -1;
};

class LabelLayer;

class Viewport : public QRhiWidget {
    Q_OBJECT
public:
    explicit Viewport(QWidget* parent = nullptr);
    ~Viewport() override;

    void setPart(std::shared_ptr<Part> part);
    void clearPart();
    std::shared_ptr<Part> part() const { return part_; }

    /** Per-unique-vertex values coloured through the ramp; nullptr restores the plain look. NaN = no data. */
    void setScalars(const std::vector<float>* values, const ColorSpec& spec = {});
    /** Displace by per-unique-vertex vectors (model units) times scale; nullptr restores the shape. */
    void setDeformation(const std::vector<float>* disp, double scale = 1);
    void setEdgesVisible(bool on);
    void setXRay(bool on);
    bool xray() const { return xray_; }
    void setPartVisible(bool on);

    // named overlay layers ("overlays", "hover", "shape", "voxels:mesh", ...)
    void setLayer(const std::string& name, std::vector<Geometry> items);
    void clearLayer(const std::string& name);

    void setMarker(const QVector3D& p, const QString& label);
    void moveMarker(const QVector3D& p);
    void clearMarker();
    bool hasMarker() const { return marker_.has_value(); }
    QString markerLabel() const { return marker_ ? marker_->label : QString(); }

    void setView(const QString& name);
    /** Camera as {eye xyz, target xyz, up xyz} (saved in .psim files). */
    std::array<float, 9> cameraState() const { return {eye_.x(), eye_.y(), eye_.z(), target_.x(), target_.y(), target_.z(), up_.x(), up_.y(), up_.z()}; }
    void setCameraState(const std::array<float, 9>& c) { eye_ = {c[0], c[1], c[2]}; target_ = {c[3], c[4], c[5]}; up_ = {c[6], c[7], c[8]}; update(); }
    QPointF project(const QVector3D& p, bool* visible = nullptr) const;
    std::optional<PickHit> pickPart(const QPointF& pos);
    /** Point on the camera-facing plane through `pivot` under the pointer (dragging handles). */
    QVector3D pointOnPlane(const QPointF& pos, const QVector3D& pivot) const;
    QImage screenshot();

    /** A draggable handle: a sphere; onDrag gets the pointer's point on a camera-facing plane. */
    struct Handle {
        QVector3D center;
        float radius;
        QVector3D pivot;
        std::function<void(const QVector3D&)> onDrag;
        std::function<void()> onEnd;
    };
    void setHandles(std::vector<Handle> handles) { handles_ = std::move(handles); }

    std::function<void(QMouseEvent*)> onHover;
    std::function<void(QMouseEvent*)> onClick;
    std::function<void()> onLeave;

signals:
    void frame(double dt);

protected:
    void initialize(QRhiCommandBuffer* cb) override;
    void render(QRhiCommandBuffer* cb) override;
    void releaseResources() override;
    void mousePressEvent(QMouseEvent* e) override;
    void mouseMoveEvent(QMouseEvent* e) override;
    void mouseReleaseEvent(QMouseEvent* e) override;
    void wheelEvent(QWheelEvent* e) override;
    void leaveEvent(QEvent* e) override;
    void resizeEvent(QResizeEvent* e) override;

private:
    struct Item {
        Geometry g;
        std::unique_ptr<QRhiBuffer> vbuf, ubuf;
        std::unique_ptr<QRhiShaderResourceBindings> srb;
        size_t capacity = 0;
        QRhiTexture* srbTex = nullptr;
        bool dirty = true;
        bool scalar = false;
        ColorSpec spec;
    };
    struct Marker {
        QVector3D p;
        QString label;
    };

    void rebuildPartVertices();
    void rebuildEdges();
    void rebuildGrid();
    void ensureItem(Item& it, QRhiResourceUpdateBatch* u);
    void drawItem(QRhiCommandBuffer* cb, Item& it, const QMatrix4x4& mvp, const QVector3D& toViewer, QRhiResourceUpdateBatch* u, int pass);
    QRhiGraphicsPipeline* pipeline(Geometry::Kind kind, bool blend, bool depthTest, bool depthWrite, bool bias);
    QMatrix4x4 viewMatrix() const;
    QMatrix4x4 projMatrix() const;
    void cameraRay(const QPointF& pos, QVector3D& origin, QVector3D& dir) const;
    void orbit(float dx, float dy);
    void pan(float dx, float dy);
    const BVH& pickBVH();
    void tick();

    std::shared_ptr<Part> part_;
    std::vector<float> scalars_, disp_;
    bool hasScalars_ = false, hasDisp_ = false;
    double dispScale_ = 1;
    ColorSpec spec_;
    bool edges_ = true, xray_ = false, partVisible_ = true;

    std::unique_ptr<Item> partItem_, edgeItem_, gridItem_, markerItem_, markerHalo_;
    std::map<std::string, std::vector<std::unique_ptr<Item>>> layers_;
    std::vector<std::unique_ptr<Item>> triad_;
    std::optional<Marker> marker_;
    std::vector<Handle> handles_;

    // deformed positions for picking (rebuilt lazily)
    std::vector<float> pickVerts_;
    std::unique_ptr<BVH> pickBVH_;
    bool pickDirty_ = true;

    QVector3D target_{0, 0, 0}, eye_{200, 160, 240}, up_{0, 1, 0};
    float fov_ = 35;
    float sceneSize_ = 100;

    QRhi* rhi_ = nullptr;
    QShader vs_, fs_;
    std::unique_ptr<QRhiSampler> sampler_;
    std::unique_ptr<QRhiTexture> rainbow_, heat_;
    bool texturesUploaded_ = false;
    std::unique_ptr<QRhiShaderResourceBindings> layoutSrb_;
    std::unique_ptr<QRhiBuffer> layoutUbuf_;
    std::map<int, std::unique_ptr<QRhiGraphicsPipeline>> pipelines_;
    QRhiRenderPassDescriptor* rpDesc_ = nullptr;
    int sampleCount_ = 1;

    enum class Drag { None, Orbit, Pan, Handle } drag_ = Drag::None;
    QPointF pressPos_, lastPos_;
    Qt::MouseButton pressButton_ = Qt::NoButton;
    int activeHandle_ = -1;

    QTimer timer_;
    QElapsedTimer clock_;
    LabelLayer* labels_ = nullptr;
    friend class LabelLayer;
};

}  // namespace ps
