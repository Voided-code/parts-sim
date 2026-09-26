// Interactive selection of surface patches: click CAD faces, or paint a circular area.
#pragma once

#include <QColor>
#include <QFrame>
#include <functional>
#include <optional>

#include "fea/structural.hpp"

class QLabel;
class QSlider;
class QWidget;

namespace ps {

class MainWindow;

class Picker : public QFrame {
    Q_OBJECT
public:
    Picker(MainWindow* app, QWidget* parent);

    struct Session {
        QString title;
        QColor color;
        std::function<void(const Patch&)> onPatch;
        std::function<void(bool commit)> onDone;
        int count = 0;
    };

    bool active() const { return session_.has_value(); }
    void start(Session s);
    void finish(bool commit);
    void move(QMouseEvent* e);
    void click(QMouseEvent* e);

private:
    void setMode(bool brush);
    void updateHint();
    double radius() const;
    const std::vector<int32_t>& faceTris(int face);

    MainWindow* app_;
    std::optional<Session> session_;
    bool brush_ = false;
    int hoverFace_ = -1;
    QLabel* title_;
    QWidget* radiusRow_;
    QSlider* radiusSlider_;
    QWidget* modeSeg_;
    // face -> triangles index, rebuilt when the part or its faces change
    const void* indexedPart_ = nullptr;
    int indexedFaces_ = -1;
    std::vector<std::vector<int32_t>> faces_;
};

}  // namespace ps
