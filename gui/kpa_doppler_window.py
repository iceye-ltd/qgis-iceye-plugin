"""Popup window showing 2D Doppler spectrum after KPA compensation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qgis.PyQt.QtCore import QCoreApplication, QRectF, Qt
from qgis.PyQt.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from qgis.PyQt.QtWidgets import (
    QDialog,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


def _tr(message: str) -> str:
    """Translate message for ICEYE Toolbox context."""
    return QCoreApplication.translate("ICEYE Toolbox", message)


def log_doppler_spectrum_to_qimage(
    log_doppler: NDArray[np.floating],
) -> QImage | None:
    """Convert 2D log-magnitude Doppler spectrum to grayscale QImage (origin lower)."""
    if log_doppler.size == 0:
        return None

    min_val = float(np.min(log_doppler))
    max_val = float(np.max(log_doppler))
    if max_val <= min_val:
        normalized = np.zeros(log_doppler.shape, dtype=np.uint8)
    else:
        scaled = (log_doppler - min_val) / (max_val - min_val)
        normalized = (scaled * 255).astype(np.uint8)

    normalized = np.ascontiguousarray(np.flipud(normalized))
    rows, cols = normalized.shape
    return QImage(
        normalized.data,
        cols,
        rows,
        cols,
        QImage.Format.Format_Grayscale8,
    ).copy()


class DopplerSpectrumImageWidget(QWidget):
    """2D Doppler spectrum heatmap with axis labels."""

    _ML, _MR, _MT, _MB = 58, 16, 32, 44

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._title = _tr("Doppler spectrum after KPA")
        self._pixmap = QPixmap()
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_spectrum_image(self, image: QImage | None) -> None:
        """Update the displayed spectrum image."""
        if image is None or image.isNull():
            self._pixmap = QPixmap()
        else:
            self._pixmap = QPixmap.fromImage(image)
        self.update()

    def paintEvent(self, event) -> None:
        """Draw title, axis labels, and the spectrum heatmap."""
        ml, mr, mt, mb = self._ML, self._MR, self._MT, self._MB
        w, h = self.width(), self.height()
        pw = max(1, w - ml - mr)
        ph = max(1, h - mt - mb)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.fillRect(0, 0, w, h, QColor(255, 255, 255))

        p.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        p.setPen(QColor(40, 40, 40))
        p.drawText(QRectF(ml, 2, pw, mt - 2), Qt.AlignmentFlag.AlignCenter, self._title)

        plot_rect = QRectF(ml, mt, pw, ph)
        p.fillRect(plot_rect, QColor(245, 245, 245))

        if self._pixmap.isNull():
            p.setFont(QFont("Arial", 9))
            p.setPen(QColor(120, 120, 120))
            p.drawText(plot_rect, Qt.AlignmentFlag.AlignCenter, _tr("No spectrum data"))
        else:
            scaled = self._pixmap.scaled(
                int(pw),
                int(ph),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            p.drawPixmap(int(ml), int(mt), scaled)

        p.setPen(QPen(QColor(180, 180, 180), 1))
        p.drawRect(plot_rect)

        small_font = QFont("Arial", 8)
        p.setFont(small_font)
        p.setPen(QColor(80, 80, 80))
        p.drawText(
            QRectF(ml, mt + ph + 22, pw, 14),
            Qt.AlignmentFlag.AlignCenter,
            _tr("Range"),
        )

        p.save()
        p.translate(ml / 2, mt + ph / 2)
        p.rotate(-90)
        p.setFont(QFont("Arial", 9))
        p.drawText(
            QRectF(-ph / 2, -10, ph, 20),
            Qt.AlignmentFlag.AlignCenter,
            _tr("Azimuth frequency"),
        )
        p.restore()

        p.end()


class KpaDopplerWindow(QDialog):
    """Non-modal popup with 2D after-KPA Doppler spectrum for tuning."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(_tr("Doppler Spectrum"))
        self.setMinimumSize(520, 420)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        self._coeff_label = QLabel()
        self._coeff_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._coeff_label)

        self._velocity_label = QLabel()
        self._velocity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self._velocity_label)

        self._spectrum_widget = DopplerSpectrumImageWidget()
        root.addWidget(self._spectrum_widget)

    def update_spectrum(
        self,
        log_doppler: NDArray[np.floating],
        *,
        a1: float,
        a2: float,
        velocity: float | None = None,
    ) -> None:
        """Refresh the after-KPA 2D Doppler spectrum display."""
        self._coeff_label.setText(
            _tr("KPA coefficients: linear={a1:.3f}, quadratic={a2:.3f}").format(
                a1=a1, a2=a2
            )
        )
        if velocity is None:
            self._velocity_label.setText(
                _tr("Velocity: unavailable (missing metadata)")
            )
        else:
            self._velocity_label.setText(
                _tr("Velocity: {velocity:.3f} m/s").format(velocity=velocity)
            )
        image = log_doppler_spectrum_to_qimage(log_doppler)
        self._spectrum_widget.set_spectrum_image(image)
