// Shared UI pieces: number formatting, the colour legend, the line chart, the busy overlay,
// KPI tiles and small form helpers.
#pragma once

#include <QFrame>
#include <QLabel>
#include <QPointer>
#include <QProgressBar>
#include <QPushButton>
#include <QWidget>
#include <functional>
#include <vector>

class QGridLayout;
class QVBoxLayout;
class QFormLayout;
class QComboBox;
class QDoubleSpinBox;
class QSlider;
class QListWidget;

namespace ps {

QString num(double v, int digits = 3);
QString stress(double pa);
std::function<QString(double)> stressFormatter(double maxAbs);
QString force(double n);

struct LegendSpec {
    QString title, sub;
    double min = 0, max = 1;
    std::function<QString(double)> format;
    int bands = 0;
    bool reverse = false, heat = false;
    struct Marker { double value; QString label; };
    std::vector<Marker> markers;
};

/** Colour legend floating over the viewport. */
class Legend : public QWidget {
    Q_OBJECT
public:
    explicit Legend(QWidget* parent);
    void setSpecs(std::vector<LegendSpec> specs);
    void clear() { setSpecs({}); }

protected:
    void paintEvent(QPaintEvent*) override;

private:
    std::vector<LegendSpec> specs_;
};

/** One-series line chart with hover tooltip and a marked current point. */
class LineChart : public QWidget {
    Q_OBJECT
public:
    struct Options {
        QString xLabel;
        std::function<QString(double)> formatX = [](double v) { return QString::number(v); };
        std::function<QString(double)> formatY = [](double v) { return QString::number(v); };
        std::function<QString(double, double, int)> tipText;
        bool integerX = true, logY = false;
        int xTicks = 4;
    };
    explicit LineChart(QWidget* parent, Options o);
    void setPoints(std::vector<QPointF> pts, int current = -1);
    std::function<void(int)> onPick;

protected:
    void paintEvent(QPaintEvent*) override;
    void mouseMoveEvent(QMouseEvent*) override;
    void mousePressEvent(QMouseEvent*) override;
    void leaveEvent(QEvent*) override;

private:
    QRectF plotRect() const;
    QPointF map(const QPointF& p) const;
    Options o_;
    std::vector<QPointF> pts_;
    int current_ = -1, hover_ = -1;
    double xMin_ = 0, xMax_ = 1, yMin_ = 0, yMax_ = 1;
};

/** Progress box shown over the viewport while a solver runs. */
class BusyOverlay : public QFrame {
    Q_OBJECT
public:
    explicit BusyOverlay(QWidget* parent);
    void showBusy(const QString& text, std::function<void()> onCancel = {});
    void progress(double frac, const QString& text = {});
    void hideBusy();
    bool busy() const { return isVisible(); }

private:
    QLabel* text_;
    QProgressBar* bar_;
    QPushButton* cancel_;
    std::function<void()> onCancel_;
};

/** Grid of KPI tiles (label, big value, small note with optional status dot). */
class KpiGrid : public QWidget {
    Q_OBJECT
public:
    explicit KpiGrid(QWidget* parent = nullptr);
    struct Kpi {
        QString label, value, sub;
        QString status;      // "good", "warn", "bad" or empty
        QString statusText;
        bool wide = false;
    };
    void setKpis(const std::vector<Kpi>& kpis);

private:
    QGridLayout* grid_;
};

/** A titled card (QGroupBox-like) used for sidebar sections. */
QFrame* card(const QString& title, QVBoxLayout** content, QWidget* parent = nullptr);
QLabel* note(const QString& text, QWidget* parent = nullptr);
/** Row of exclusive toggle buttons; returns the container. onPick(index) on change. */
QWidget* segmented(const QStringList& labels, int current, std::function<void(int)> onPick, QWidget* parent = nullptr);
/** Let every control under `root` shrink to the sidebar width (combos, spin boxes, button grids). */
void compact(QWidget* root);
/** Size a list widget to its rows (no empty space, no inner scrolling for short lists). */
void fitList(QListWidget* list, int maxRows = 8);
QDoubleSpinBox* numberBox(double value, double min, double max, double step, int decimals, std::function<void(double)> onChange, QWidget* parent = nullptr);

}  // namespace ps
