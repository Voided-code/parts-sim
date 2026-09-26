#include "widgets.hpp"

#include <QButtonGroup>
#include <QComboBox>
#include <QListWidget>
#include <QDoubleSpinBox>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLocale>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QToolButton>
#include <QToolTip>
#include <QVBoxLayout>
#include <algorithm>
#include <cmath>
#include <utility>

namespace ps {

QString num(double v, int digits) {
    if (!std::isfinite(v)) return QStringLiteral("–");
    if (v != 0 && (std::abs(v) >= 1e6 || std::abs(v) < 1e-3)) {
        QString s = QString::number(v, 'e', digits - 1);
        s.replace("e+0", "e").replace("e+", "e").replace("e-0", "e-");
        return s;
    }
    // significant digits, grouping like the web version (Intl.NumberFormat)
    const int mag = v == 0 ? 0 : int(std::floor(std::log10(std::abs(v))));
    const int decimals = std::max(0, digits - 1 - mag);
    const double f = std::pow(10.0, decimals);
    const double r = std::round(v * f) / f;
    QString s = QLocale().toString(r, 'f', decimals);
    if (s.contains(QLocale().decimalPoint())) {
        while (s.endsWith('0')) s.chop(1);
        if (s.endsWith(QLocale().decimalPoint())) s.chop(1);
    }
    return s;
}

QString stress(double pa) {
    if (!std::isfinite(pa)) return QStringLiteral("–");
    const double a = std::abs(pa);
    if (a >= 1e9) return num(pa / 1e9) + " GPa";
    if (a >= 1e5) return num(pa / 1e6) + " MPa";
    if (a >= 100) return num(pa / 1e3) + " kPa";
    return num(pa) + " Pa";
}

std::function<QString(double)> stressFormatter(double maxAbs) {
    double div = 1;
    QString unit = "Pa";
    if (maxAbs >= 1e9) { div = 1e9; unit = "GPa"; }
    else if (maxAbs >= 1e5) { div = 1e6; unit = "MPa"; }
    else if (maxAbs >= 100) { div = 1e3; unit = "kPa"; }
    return [div, unit](double pa) { return std::isfinite(pa) ? num(pa / div) + " " + unit : QStringLiteral("–"); };
}

QString force(double n) {
    if (!std::isfinite(n)) return QStringLiteral("–");
    const double a = std::abs(n);
    if (a >= 1e6) return num(n / 1e6) + " MN";
    if (a >= 1e4) return num(n / 1e3) + " kN";
    return num(n) + " N";
}

// ---------- legend ----------

static QColor rampColor(bool heat, double t) {
    static const double R[5][4] = {{0, 0, 0, 255}, {0.25, 0, 255, 255}, {0.5, 0, 255, 0}, {0.75, 255, 255, 0}, {1, 255, 0, 0}};
    static const char* H[9] = {"#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84", "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"};
    t = std::clamp(t, 0.0, 1.0);
    if (heat) {
        const double x = t * 8;
        const int i = std::min(7, int(x));
        const QColor a(H[i]), b(H[i + 1]);
        const double s = x - i;
        return QColor::fromRgbF(a.redF() + s * (b.redF() - a.redF()), a.greenF() + s * (b.greenF() - a.greenF()), a.blueF() + s * (b.blueF() - a.blueF()));
    }
    for (int i = 1; i < 5; i++)
        if (t <= R[i][0]) {
            const double s = (t - R[i - 1][0]) / (R[i][0] - R[i - 1][0]);
            return QColor::fromRgbF((R[i - 1][1] + s * (R[i][1] - R[i - 1][1])) / 255, (R[i - 1][2] + s * (R[i][2] - R[i - 1][2])) / 255,
                                    (R[i - 1][3] + s * (R[i][3] - R[i - 1][3])) / 255);
        }
    return Qt::red;
}

Legend::Legend(QWidget* parent) : QWidget(parent) {
    setAttribute(Qt::WA_TransparentForMouseEvents);
    hide();
}

void Legend::setSpecs(std::vector<LegendSpec> specs) {
    specs_ = std::move(specs);
    setVisible(!specs_.empty());
    int h = 12;
    for (size_t i = 0; i < specs_.size(); i++) h += (specs_.size() > 1 ? 150 : 220) + 48 + (specs_[i].sub.isEmpty() ? 0 : 28) + (i ? 12 : 0);
    resize(220, h);
    update();
}

void Legend::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QPalette pal = palette();
    const bool dark = pal.color(QPalette::Window).lightness() < 128;
    const QColor panel = dark ? QColor(26, 29, 34, 235) : QColor(255, 255, 255, 235);
    const QColor text = dark ? QColor(238, 241, 245) : QColor(18, 24, 32);
    const QColor muted = dark ? QColor(139, 147, 158) : QColor(111, 121, 134);
    p.setPen(QColor(dark ? 44 : 221, dark ? 49 : 227, dark ? 57 : 234));
    p.setBrush(panel);
    p.drawRoundedRect(rect().adjusted(0, 0, -1, -1), 10, 10);
    int y = 12;
    for (size_t i = 0; i < specs_.size(); i++) {
        const auto& s = specs_[i];
        if (i) y += 12;
        const int H = specs_.size() > 1 ? 150 : 220;
        QFont bold = font();
        bold.setBold(true);
        p.setFont(bold);
        p.setPen(text);
        p.drawText(QRect(12, y, width() - 24, 18), Qt::AlignLeft | Qt::AlignVCenter, s.title);
        y += 20;
        if (!s.sub.isEmpty()) {
            QFont small = font();
            small.setPointSizeF(small.pointSizeF() * 0.85);
            p.setFont(small);
            p.setPen(muted);
            p.drawText(QRect(12, y, width() - 24, 28), Qt::AlignLeft | Qt::TextWordWrap, s.sub);
            y += 28;
        }
        y += 6;
        // gradient bar (bottom = low)
        const QRect bar(14, y, 14, H);
        const int n = s.bands > 0 ? s.bands : 64;
        for (int k = 0; k < n; k++) {
            const double t0 = double(k) / n, t = s.bands > 0 ? (k + 0.5) / n : t0;
            const QColor c = rampColor(s.heat, s.reverse ? 1 - t : t);
            const int yy0 = bar.bottom() - int(std::round((k + 1) * double(H) / n)), yy1 = bar.bottom() - int(std::round(k * double(H) / n));
            p.fillRect(QRect(bar.left(), yy0, bar.width(), yy1 - yy0 + 1), c);
        }
        p.setPen(QColor(0, 0, 0, 60));
        p.setBrush(Qt::NoBrush);
        p.drawRect(bar);
        // ticks; markers win over ticks they would overlap
        QFont mono("Menlo");
        mono.setStyleHint(QFont::Monospace);
        mono.setPointSizeF(font().pointSizeF() * 0.85);
        p.setFont(mono);
        const auto fmt = s.format ? s.format : [](double v) { return num(v); };
        std::vector<std::pair<double, QString>> marks;
        for (const auto& m : s.markers) {
            if (!(m.value >= s.min && m.value <= s.max)) continue;
            const double yy = bar.bottom() - (m.value - s.min) / (s.max - s.min != 0 ? s.max - s.min : 1) * H;
            bool clash = false;
            for (auto& o : marks) clash = clash || std::abs(o.first - yy) < 14;
            if (!clash) marks.push_back({yy, m.label});
        }
        for (int k = 0; k <= 6; k++) {
            const double v = s.min + (s.max - s.min) * k / 6;
            const double yy = bar.bottom() - double(k) / 6 * H;
            bool clash = false;
            for (auto& m : marks) clash = clash || std::abs(m.first - yy) < 14;
            if (clash) continue;
            p.setPen(text);
            p.drawText(QRectF(bar.right() + 8, yy - 8, width() - bar.right() - 16, 16), Qt::AlignLeft | Qt::AlignVCenter, fmt(v));
        }
        for (auto& m : marks) {
            p.setPen(QPen(QColor(220, 38, 38), 1.5));
            p.drawLine(QPointF(bar.left() - 4, m.first), QPointF(bar.right() + 4, m.first));
            p.drawText(QRectF(bar.right() + 8, m.first - 8, width() - bar.right() - 16, 16), Qt::AlignLeft | Qt::AlignVCenter, m.second);
        }
        y += H + 16;
    }
}

// ---------- line chart ----------

LineChart::LineChart(QWidget* parent, Options o) : QWidget(parent), o_(std::move(o)) {
    setMouseTracking(true);
    setMinimumHeight(150);
    setMaximumHeight(170);
    setCursor(Qt::CrossCursor);
}

void LineChart::setPoints(std::vector<QPointF> pts, int current) {
    pts_ = std::move(pts);
    current_ = current;
    xMin_ = 1e300; xMax_ = -1e300; yMin_ = 0; yMax_ = 1e-12;
    std::vector<double> ys;
    for (auto& q : pts_) {
        xMin_ = std::min(xMin_, q.x());
        xMax_ = std::max(xMax_, q.x());
        if (std::isfinite(q.y())) ys.push_back(q.y());
    }
    if (pts_.empty()) { xMin_ = 0; xMax_ = 1; }
    if (xMax_ <= xMin_) xMax_ = xMin_ + (o_.integerX ? 1 : 1e-12);
    if (o_.logY) {
        double pos = 0, mn = 1e300;
        for (double y : ys) if (y > 0) { pos = std::max(pos, y); mn = std::min(mn, y); }
        yMax_ = std::max(pos, 1e-30) * 1.5;
        yMin_ = std::max(std::min(mn, yMax_) / 1.5, yMax_ * 1e-6);
    } else {
        for (double y : ys) { yMax_ = std::max(yMax_, y * 1.1); yMin_ = std::min(yMin_, y * 1.1); }
    }
    update();
}

QRectF LineChart::plotRect() const { return QRectF(56, 8, width() - 66, height() - 40); }

QPointF LineChart::map(const QPointF& q) const {
    const QRectF r = plotRect();
    const double x = r.left() + (q.x() - xMin_) / (xMax_ - xMin_) * r.width();
    double t;
    if (o_.logY) t = (std::log10(std::max(q.y(), yMin_)) - std::log10(yMin_)) / (std::log10(yMax_) - std::log10(yMin_));
    else t = (q.y() - yMin_) / (yMax_ - yMin_ != 0 ? yMax_ - yMin_ : 1);
    return QPointF(x, r.bottom() - t * r.height());
}

void LineChart::paintEvent(QPaintEvent*) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const bool dark = palette().color(QPalette::Window).lightness() < 128;
    const QColor grid = dark ? QColor(44, 49, 57) : QColor(228, 232, 238);
    const QColor muted = dark ? QColor(139, 147, 158) : QColor(111, 121, 134);
    const QColor line = dark ? QColor(57, 135, 229) : QColor(42, 120, 214);
    if (pts_.empty()) return;
    const QRectF r = plotRect();
    QFont mono("Menlo");
    mono.setStyleHint(QFont::Monospace);
    mono.setPointSizeF(font().pointSizeF() * 0.78);
    p.setFont(mono);
    std::vector<double> yt;
    if (o_.logY) {
        for (int e = int(std::ceil(std::log10(yMin_))); e <= int(std::floor(std::log10(yMax_))); e++) yt.push_back(std::pow(10, e));
        while (yt.size() > 5) yt.erase(yt.begin() + 1);
    } else
        for (int k = 0; k <= 3; k++) yt.push_back(yMin_ + (yMax_ - yMin_) * k / 3.3);
    for (double v : yt) {
        const double y = map({xMin_, v}).y();
        p.setPen(grid);
        p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y));
        p.setPen(muted);
        p.drawText(QRectF(0, y - 8, r.left() - 5, 16), Qt::AlignRight | Qt::AlignVCenter, o_.formatY(v));
    }
    const double span = xMax_ - xMin_;
    double every;
    if (o_.integerX) every = std::max(1.0, std::ceil(span / o_.xTicks));
    else {
        const double raw = span / o_.xTicks, mag = std::pow(10, std::floor(std::log10(raw)));
        const double f = raw / mag;
        every = (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * mag;
    }
    const double x0 = o_.integerX ? xMin_ : std::ceil(xMin_ / every) * every;
    p.setPen(muted);
    for (double xv = x0; xv <= xMax_ + every * 1e-9; xv += every) {
        const double x = map({xv, yMin_}).x();
        p.drawText(QRectF(x - 40, r.bottom() + 4, 80, 14), Qt::AlignCenter, o_.formatX(xv));
    }
    p.drawText(QRectF(r.left(), height() - 16, r.width(), 14), Qt::AlignCenter, o_.xLabel);
    QPainterPath path;
    bool started = false;
    for (auto& q : pts_) {
        if (!std::isfinite(q.y())) { started = false; continue; }
        const QPointF s = map(q);
        if (started) path.lineTo(s); else { path.moveTo(s); started = true; }
    }
    p.setPen(QPen(line, 2, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin));
    p.setBrush(Qt::NoBrush);
    p.drawPath(path);
    auto mark = [&](int i, double rad) {
        if (i < 0 || i >= int(pts_.size()) || !std::isfinite(pts_[i].y())) return;
        const QPointF s = map(pts_[i]);
        p.setPen(Qt::NoPen);
        p.setBrush(palette().color(QPalette::Base));
        p.drawEllipse(s, rad + 2, rad + 2);
        p.setBrush(line);
        p.drawEllipse(s, rad, rad);
    };
    if (hover_ >= 0 && hover_ != current_) mark(hover_, 3);
    mark(current_, 4);
}

void LineChart::mouseMoveEvent(QMouseEvent* e) {
    if (pts_.empty()) return;
    int best = -1;
    double bd = 1e300;
    for (size_t i = 0; i < pts_.size(); i++) {
        const double d = std::abs(map(pts_[i]).x() - e->position().x());
        if (d < bd) { bd = d; best = int(i); }
    }
    hover_ = best;
    const auto& q = pts_[best];
    const QString tip = o_.tipText ? o_.tipText(q.x(), q.y(), best) : QString("step %1 · %2").arg(best + 1).arg(o_.formatY(q.y()));
    QToolTip::showText(e->globalPosition().toPoint() + QPoint(10, -30), tip, this);
    update();
}

void LineChart::mousePressEvent(QMouseEvent*) {
    if (hover_ >= 0 && onPick) onPick(hover_);
}

void LineChart::leaveEvent(QEvent*) {
    hover_ = -1;
    QToolTip::hideText();
    update();
}

// ---------- busy overlay ----------

BusyOverlay::BusyOverlay(QWidget* parent) : QFrame(parent) {
    setObjectName("busy");
    setFrameShape(QFrame::StyledPanel);
    setAutoFillBackground(true);
    auto* lay = new QVBoxLayout(this);
    text_ = new QLabel(this);
    text_->setWordWrap(true);
    bar_ = new QProgressBar(this);
    bar_->setRange(0, 1000);
    bar_->setTextVisible(false);
    bar_->setMaximumHeight(6);
    cancel_ = new QPushButton(tr("Cancel"), this);
    connect(cancel_, &QPushButton::clicked, this, [this] { if (onCancel_) onCancel_(); });
    lay->addWidget(text_);
    lay->addWidget(bar_);
    auto* row = new QHBoxLayout;
    row->addStretch();
    row->addWidget(cancel_);
    lay->addLayout(row);
    setFixedWidth(340);
    hide();
}

void BusyOverlay::showBusy(const QString& text, std::function<void()> onCancel) {
    text_->setText(text);
    bar_->setValue(0);
    onCancel_ = std::move(onCancel);
    cancel_->setVisible(bool(onCancel_));
    adjustSize();
    if (parentWidget()) move((parentWidget()->width() - width()) / 2, parentWidget()->height() - height() - 40);
    show();
    raise();
}

void BusyOverlay::progress(double frac, const QString& text) {
    if (frac >= 0) bar_->setValue(int(std::clamp(frac, 0.0, 1.0) * 1000));  // < 0: fraction unknown, keep the bar
    if (!text.isEmpty()) text_->setText(text);
}

void BusyOverlay::hideBusy() {
    hide();
    onCancel_ = {};
}

// ---------- KPI tiles ----------

KpiGrid::KpiGrid(QWidget* parent) : QWidget(parent) {
    grid_ = new QGridLayout(this);
    grid_->setContentsMargins(0, 0, 0, 0);
    grid_->setSpacing(8);
}

void KpiGrid::setKpis(const std::vector<Kpi>& kpis) {
    while (QLayoutItem* it = grid_->takeAt(0)) {
        delete it->widget();
        delete it;
    }
    int row = 0, col = 0;
    for (const auto& k : kpis) {
        auto* f = new QFrame(this);
        f->setObjectName("kpi");
        auto* v = new QVBoxLayout(f);
        v->setContentsMargins(10, 8, 10, 8);
        v->setSpacing(2);
        auto* l = new QLabel(k.label, f);
        l->setObjectName("kpiLabel");
        l->setWordWrap(true);
        auto* val = new QLabel(k.value, f);
        val->setObjectName("kpiValue");
        val->setWordWrap(true);
        v->addWidget(l);
        v->addWidget(val);
        if (!k.sub.isEmpty() || !k.status.isEmpty()) {
            QString html;
            if (!k.status.isEmpty()) {
                const QString c = k.status == "good" ? "#1a8a4a" : k.status == "warn" ? "#b7791f" : "#cf3434";
                html += QString("<span style='color:%1'>● %2</span> ").arg(c, k.statusText.toHtmlEscaped());
            }
            html += k.sub.toHtmlEscaped();
            auto* s = new QLabel(html, f);
            s->setObjectName("kpiSub");
            s->setWordWrap(true);
            v->addWidget(s);
        }
        if (k.wide) {
            if (col) { row++; col = 0; }
            grid_->addWidget(f, row, 0, 1, 2);
            row++;
        } else {
            grid_->addWidget(f, row, col);
            if (++col == 2) { col = 0; row++; }
        }
    }
}

// ---------- form helpers ----------

QFrame* card(const QString& title, QVBoxLayout** content, QWidget* parent) {
    auto* f = new QFrame(parent);
    f->setObjectName("card");
    auto* v = new QVBoxLayout(f);
    v->setContentsMargins(14, 12, 14, 12);
    v->setSpacing(8);
    if (!title.isEmpty()) {
        auto* h = new QLabel(title.toUpper(), f);
        h->setObjectName("cardTitle");
        v->addWidget(h);
    }
    *content = v;
    return f;
}

QLabel* note(const QString& text, QWidget* parent) {
    auto* l = new QLabel(text, parent);
    l->setObjectName("note");
    l->setWordWrap(true);
    return l;
}

QWidget* segmented(const QStringList& labels, int current, std::function<void(int)> onPick, QWidget* parent) {
    auto* w = new QWidget(parent);
    auto* h = new QHBoxLayout(w);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(0);
    auto* group = new QButtonGroup(w);
    group->setExclusive(true);
    for (int i = 0; i < labels.size(); i++) {
        auto* b = new QToolButton(w);
        b->setText(QString(labels[i]).replace("&", "&&"));  // no mnemonics
        b->setCheckable(true);
        b->setChecked(i == current);
        b->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        b->setObjectName(i == 0 ? "segFirst" : i == labels.size() - 1 ? "segLast" : "segMid");
        group->addButton(b, i);
        h->addWidget(b);
    }
    QObject::connect(group, &QButtonGroup::idClicked, w, [onPick](int id) { onPick(id); });
    return w;
}

void compact(QWidget* root) {
    for (auto* c : root->findChildren<QComboBox*>()) {
        c->setSizeAdjustPolicy(QComboBox::AdjustToMinimumContentsLengthWithIcon);
        c->setMinimumContentsLength(8);
        c->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    }
    for (auto* s : root->findChildren<QAbstractSpinBox*>()) {
        s->setMinimumWidth(56);
        s->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    }
    for (auto* b : root->findChildren<QPushButton*>()) {
        if (b->objectName() == "primary") continue;
        // as narrow as the label allows, so button grids fit the sidebar
        b->setMinimumWidth(b->fontMetrics().horizontalAdvance(b->text()) + 18);
        b->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Fixed);
        b->setMinimumHeight(26);
    }
    for (auto* l : root->findChildren<QLabel*>()) l->setMinimumWidth(0);
}

void fitList(QListWidget* list, int maxRows) {
    int h = 0;
    for (int i = 0; i < list->count() && i < maxRows; i++) h += list->sizeHintForRow(i) + 2 * list->spacing() + 4;
    list->setFixedHeight(std::max(h + 2 * list->frameWidth() + 2, list->count() ? 0 : 6));
    list->setVisible(list->count() > 0);
}

QDoubleSpinBox* numberBox(double value, double min, double max, double step, int decimals, std::function<void(double)> onChange, QWidget* parent) {
    auto* s = new QDoubleSpinBox(parent);
    s->setRange(min, max);
    s->setDecimals(decimals);
    s->setSingleStep(step);
    s->setValue(value);
    s->setKeyboardTracking(false);
    QObject::connect(s, &QDoubleSpinBox::valueChanged, s, [onChange](double v) { onChange(v); });
    return s;
}

}  // namespace ps
