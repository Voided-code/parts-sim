// Part tab: geometry summary, material, size, units, orientation and face detection.
#pragma once

#include <QWidget>

class QComboBox;
class QFormLayout;
class QLabel;
class QDoubleSpinBox;
class QSlider;
class QFrame;

namespace ps {

class MainWindow;

class PartPanel : public QWidget {
    Q_OBJECT
public:
    explicit PartPanel(MainWindow* app);
    void refresh();          // part summary
    void refreshMaterial();  // material fields from app state

private:
    void buildMaterialFields();

    MainWindow* app_;
    QLabel* empty_;
    QFormLayout* stats_;
    QWidget* statsBox_;
    QComboBox* materialBox_;
    QWidget* materialFields_;
    QDoubleSpinBox* scaleFactor_;
    QDoubleSpinBox* targetSize_;
    QLabel* sizeUnit_;
    QWidget* unitsSeg_;
    QFrame* faceCard_;
    QSlider* faceAngle_;
    QLabel* faceAngleOut_;
    bool updating_ = false;
};

}  // namespace ps
