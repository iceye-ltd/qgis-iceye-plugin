"""IRF analysis result UI: profile plot and metrics dialog."""

from __future__ import annotations

import math

import numpy as np
from qgis.PyQt.QtCore import QCoreApplication, QPointF, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QPainter, QPen
from qgis.PyQt.QtWidgets import (
    QDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.irf import IrfResult


def _tr(message: str) -> str:
    """Translate message for ICEYE Toolbox context."""
    return QCoreApplication.translate("ICEYE Toolbox", message)


class ProfileWidget(QWidget):
    """QPainter-based 1D dB profile plot.

    Draws the amplitude profile (normalised to 0 dB) together with a dashed
    -3 dB reference line, axis labels, and a white background.
    """

    _ML, _MR, _MT, _MB = 52, 14, 28, 38  # plot-area margins in pixels
    _X_WINDOW_M = 4.0  # x-axis display range: -2 m to +2 m

    def __init__(
        self,
        title: str,
        profile_db: np.ndarray,
        spacing_m: float,
        line_color: QColor,
        parent=None,
    ) -> None:
        """Initialise the profile widget.

        Args:
            title: Chart title text.
            profile_db: 1D dB profile (peak normalised to 0 dB).
            spacing_m: Pixel spacing in metres (sets x-axis scale).
            line_color: Colour for the profile curve.
            parent: Parent widget.
        """
        super().__init__(parent)
        self._title = title
        self._profile = np.asarray(profile_db, dtype=np.float32)
        self._spacing_m = spacing_m
        self._line_color = QColor(line_color)
        self.setMinimumSize(360, 240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def paintEvent(self, event) -> None:
        """Draw axes, grid, reference line, and profile curve."""
        ml, mr, mt, mb = self._ML, self._MR, self._MT, self._MB
        w, h = self.width(), self.height()
        pw = w - ml - mr
        ph = h - mt - mb

        y_max = 5.0
        y_min = max(-60.0, float(np.min(self._profile)) - 5.0)
        y_span = y_max - y_min

        n = len(self._profile)
        half = n // 2
        x_min_m = -self._X_WINDOW_M / 2.0  # -2 m
        x_span_m = self._X_WINDOW_M  # 4 m total (-2 to +2)

        def to_px(x_m: float, y_db: float) -> QPointF:
            px = ml + (x_m - x_min_m) / x_span_m * pw
            py = mt + (1.0 - (y_db - y_min) / y_span) * ph
            return QPointF(px, py)

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Backgrounds
        p.fillRect(0, 0, w, h, QColor(255, 255, 255))
        p.fillRect(ml, mt, pw, ph, QColor(255, 255, 255))

        small_font = QFont("Arial", 8)
        p.setFont(small_font)
        fm = p.fontMetrics()

        # Horizontal grid lines + y-axis tick labels
        db_step = 10
        tick_start = int(y_max // db_step) * db_step
        for db_val in range(tick_start, int(y_min) - 1, -db_step):
            if db_val < y_min:
                break
            pt = to_px(x_min_m, float(db_val))
            p.setPen(QPen(QColor(200, 200, 200), 1, Qt.SolidLine))
            p.drawLine(QPointF(ml, pt.y()), QPointF(ml + pw, pt.y()))
            label = str(db_val)
            lw = fm.horizontalAdvance(label)
            p.setPen(QColor(80, 80, 80))
            p.drawText(
                QRectF(ml - lw - 6, pt.y() - 8, lw + 4, 16),
                Qt.AlignRight | Qt.AlignVCenter,
                label,
            )

        # −3 dB reference line
        ref_pt = to_px(x_min_m, -3.0)
        p.setPen(QPen(QColor(230, 70, 70), 1.3, Qt.DashLine))
        p.drawLine(QPointF(ml, ref_pt.y()), QPointF(ml + pw, ref_pt.y()))
        p.setPen(QColor(230, 70, 70))
        p.setFont(QFont("Arial", 7))
        p.drawText(
            QRectF(ml + pw + 2, ref_pt.y() - 7, mr + 10, 14),
            Qt.AlignLeft | Qt.AlignVCenter,
            "-3",
        )
        p.setFont(small_font)

        # Profile curve (only within -2 m to +2 m window)
        half_win_m = self._X_WINDOW_M / 2.0
        i_min = max(0, int(half - half_win_m / self._spacing_m))
        i_max = min(n, int(half + half_win_m / self._spacing_m) + 1)
        p.setPen(QPen(self._line_color, 1.8, Qt.SolidLine))
        prev: QPointF | None = None
        for i in range(i_min, i_max):
            val = self._profile[i]
            pt = to_px((i - half) * self._spacing_m, float(val))
            if prev is not None:
                p.drawLine(prev, pt)
            prev = pt

        # Plot area border
        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.drawRect(ml, mt, pw, ph)

        # Title
        p.setFont(QFont("Arial", 10, QFont.Bold))
        p.setPen(QColor(40, 40, 40))
        p.drawText(QRectF(ml, 2, pw, mt - 2), Qt.AlignCenter, self._title)

        # Y-axis label (rotated, left of plot)
        p.save()
        p.translate(ml / 2, mt + ph / 2)
        p.rotate(-90)
        p.setFont(QFont("Arial", 9))
        p.setPen(QColor(80, 80, 80))
        p.drawText(QRectF(-ph / 2, -10, ph, 20), Qt.AlignCenter, "Amplitude (dB)")
        p.restore()

        # X-axis label
        p.setFont(small_font)
        p.setPen(QColor(80, 80, 80))
        p.drawText(QRectF(ml, mt + ph + 20, pw, 14), Qt.AlignCenter, "Offset (m)")

        # X-axis tick labels (three ticks: left, centre, right)
        p.setPen(QColor(80, 80, 80))
        x_max_m = x_min_m + x_span_m
        for frac, x_m in ((0.0, x_min_m), (0.5, 0.0), (1.0, x_max_m)):
            label = f"{x_m:.1f}"
            lw = fm.horizontalAdvance(label)
            px = ml + frac * pw
            p.drawText(
                QRectF(px - lw / 2, mt + ph + 4, lw + 4, 14), Qt.AlignCenter, label
            )

        p.end()


class IRFResultDialog(QDialog):
    """Non-modal dialog showing IRF analysis results.

    Displays range and azimuth amplitude profiles (in dB) with -3 dB reference
    lines, plus a summary table of resolution, PSLR, and ISLR metrics.
    """

    _RANGE_COLOR = QColor(100, 180, 255)
    _AZ_COLOR = QColor(100, 230, 150)

    def __init__(self, result: IrfResult, parent=None) -> None:
        """Build the dialog from an IrfResult.

        Args:
            result: IRF analysis output from analyze_point().
            parent: Parent widget (usually the QGIS main window).
        """
        super().__init__(parent)
        self.setWindowTitle(_tr("IRF Analysis"))
        self.setMinimumSize(860, 560)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowFlags(self.windowFlags() | Qt.Window)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        charts_row = QHBoxLayout()
        charts_row.addWidget(
            ProfileWidget(
                _tr("Azimuth IRF"),
                result.az_profile_db,
                result.pixel_spacing_azimuth,
                self._AZ_COLOR,
            )
        )
        charts_row.addWidget(
            ProfileWidget(
                _tr("Range IRF"),
                result.range_profile_db,
                result.pixel_spacing_range,
                self._RANGE_COLOR,
            )
        )
        root.addLayout(charts_row, stretch=1)

        root.addWidget(self._make_metrics_widget(result))

        close_btn = QPushButton(_tr("Close"))
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.close)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        root.addLayout(btn_row)

    def _make_metrics_widget(self, result: IrfResult) -> QWidget:
        """Build a styled grid widget showing all six IRF metrics."""
        container = QWidget()
        container.setStyleSheet(
            "background: #1e1e23; border-radius: 6px; padding: 8px;"
        )

        grid = QGridLayout(container)
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(6)

        header_style = "color: #888; font-size: 11px;"
        value_style = "color: #e0e0e0; font-size: 13px; font-weight: bold;"

        def _format_irf_metric(val: float, unit: str = "", *, decimals: int = 2) -> str:
            if not math.isfinite(val):
                return "N/A"
            return f"{val:.{decimals}f}{unit}"

        metrics = [
            (
                _tr("Range HPBW"),
                _format_irf_metric(result.range_resolution_m, " m", decimals=3),
            ),
            (
                _tr("Azimuth HPBW"),
                _format_irf_metric(result.az_resolution_m, " m", decimals=3),
            ),
            (_tr("Range PSLR"), _format_irf_metric(result.range_pslr_db, " dB")),
            (_tr("Azimuth PSLR"), _format_irf_metric(result.az_pslr_db, " dB")),
            (_tr("Range ISLR"), _format_irf_metric(result.range_islr_db, " dB")),
            (_tr("Azimuth ISLR"), _format_irf_metric(result.az_islr_db, " dB")),
        ]

        for col, (label_text, value_text) in enumerate(metrics):
            lbl = QLabel(label_text)
            lbl.setStyleSheet(header_style)
            lbl.setAlignment(Qt.AlignCenter)
            val = QLabel(value_text)
            val.setStyleSheet(value_style)
            val.setAlignment(Qt.AlignCenter)
            grid.addWidget(lbl, 0, col)
            grid.addWidget(val, 1, col)

        return container
