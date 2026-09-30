"""Curve editor: a line you bend by dragging its ends and its 1/3 and 2/3 handles.

Fitted over an SLC chip of the current map view, the curve marks an imaged (displaced)
moving target. "Use as target" finds the target's hull along the curve
(``core.mover_relocation.locate_imaged_target``) and hands the imaged position to the
Mover Relocation tool, where the band is drawn and the constraint is clicked.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
)
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QPointF,
    QRectF,
    Qt,
    pyqtSignal,
)
from qgis.PyQt.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import (
    QApplication,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..core.cropper import get_extend_image_coords
from ..core.metadata import MetadataProvider
from ..core.mover_relocation import ImagedTarget, locate_imaged_target
from ..core.target_finder import (
    ProductGeometry,
    RelocationParameters,
    SlcChip,
    gcp_mean_height,
    gcp_pixel_to_lonlat,
    patch_to_file_layout,
    read_iceye_properties,
)
from ..core.typing_compat import NDArray
from .lens_tool import read_slc_data

# Largest chip read from the view (complex samples); ~160 MB of complex64.
MAX_CHIP_PIXELS = 20_000_000
# Longest side of the background image drawn in the editor.
_DISPLAY_SIZE = 1024


def _tr(message: str) -> str:
    """Translate message for ICEYE Toolbox context."""
    return QCoreApplication.translate("ICEYE Toolbox", message)


def _event_pos(event) -> QPointF:
    """Widget-local cursor position (Qt6 ``position`` vs Qt5 ``pos``)."""
    if hasattr(event, "position"):
        return QPointF(event.position())
    return QPointF(event.pos())


def _block_reduce(values: NDArray[Any], max_size: int, reducer) -> NDArray[Any]:
    """Shrink a 2D array by whole blocks so neither side exceeds ``max_size``."""
    fr = max(1, math.ceil(values.shape[0] / max_size))
    fc = max(1, math.ceil(values.shape[1] / max_size))
    rows, cols = values.shape[0] // fr * fr, values.shape[1] // fc * fc
    if rows == 0 or cols == 0:
        return values
    blocks = values[:rows, :cols].reshape(rows // fr, fr, cols // fc, fc)
    return reducer(blocks, axis=(1, 3))


def chip_display_image(data: NDArray[np.complexfloating[Any]]) -> QImage:
    """Grayscale dB image of a chip, block-averaged to at most ``_DISPLAY_SIZE``."""
    power = _block_reduce(np.abs(data) ** 2, _DISPLAY_SIZE, np.mean)
    db = 10.0 * np.log10(power + 1e-12)
    lo, hi = np.percentile(db, [2.0, 99.5])
    scaled = np.clip((db - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    pixels = np.ascontiguousarray((scaled * 255).astype(np.uint8))
    rows, cols = pixels.shape
    return QImage(pixels.data, cols, rows, cols, QImage.Format.Format_Grayscale8).copy()


def mask_overlay_image(
    hull: NDArray[np.bool_] | None, ring: NDArray[np.bool_] | None = None
) -> QImage | None:
    """Translucent image of the hull (orange) and optional clutter ring (cyan) masks."""
    if hull is None:
        return None
    hull_small = _block_reduce(hull, _DISPLAY_SIZE, np.any)
    rgba = np.zeros((*hull_small.shape, 4), dtype=np.uint8)
    if ring is not None:
        rgba[_block_reduce(ring, _DISPLAY_SIZE, np.any)] = (0, 170, 200, 60)
    rgba[hull_small] = (230, 126, 34, 150)
    rgba = np.ascontiguousarray(rgba)
    rows, cols = hull_small.shape
    return QImage(rgba.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888).copy()


def canvas_extent_in_layer_crs(canvas, layer: QgsRasterLayer) -> QgsRectangle:
    """Return the canvas extent expressed in the layer CRS."""
    extent = canvas.extent()
    canvas_crs = canvas.mapSettings().destinationCrs()
    if canvas_crs != layer.crs():
        transform = QgsCoordinateTransform(
            canvas_crs, layer.crs(), QgsProject.instance()
        )
        extent = transform.transformBoundingBox(extent)
    return extent


def read_slc_chip(
    layer: QgsRasterLayer,
    extent: QgsRectangle,
    metadata_provider: MetadataProvider,
) -> SlcChip:
    """Complex SLC chip of ``layer`` under ``extent`` (layer CRS), in file layout.

    Raises
    ------
    ValueError
        With a user-facing message when the layer or extent cannot be read.
    """
    if not isinstance(layer, QgsRasterLayer) or layer.bandCount() < 2:
        raise ValueError(_tr("Select an ICEYE SLC layer first."))
    metadata = metadata_provider.get(layer)
    if metadata is None:
        raise ValueError(_tr("The active layer has no ICEYE metadata."))
    bounds = get_extend_image_coords(layer, extent)
    if bounds is None:
        raise ValueError(_tr("Could not map the view to SLC pixels."))
    if (
        bounds.xMinimum() < 0
        or bounds.yMinimum() < 0
        or bounds.xMaximum() > layer.width()
        or bounds.yMaximum() > layer.height()
    ):
        raise ValueError(_tr("Zoom in so the view lies inside the SLC."))
    if bounds.width() * bounds.height() > MAX_CHIP_PIXELS:
        raise ValueError(_tr("The view is too large; zoom in on the target."))
    slc = read_slc_data(layer, extent, metadata_provider)
    if slc is None:
        raise ValueError(_tr("Failed to read the SLC samples."))
    source = layer.dataProvider().dataSourceUri()
    left = (metadata.sar_observation_direction or "").lower() == "left"
    return SlcChip(
        data=patch_to_file_layout(slc.data_patch, left),
        col0=int(bounds.xMinimum()),
        row0=int(bounds.yMinimum()),
        geometry=ProductGeometry.from_properties(read_iceye_properties(source)),
        pixel_to_lonlat=gcp_pixel_to_lonlat(source),
    )


class CurveEditorWidget(QWidget):
    """A cubic Bezier with four draggable handles: both ends, plus 1/3 and 2/3.

    The inner handles are the Bezier control points. Sitting exactly at 1/3 and 2/3
    of the chord they describe the straight segment, which is the starting state;
    dragging them away from there is what bends the line.

    Points are held as 0..1 fractions of the plot area, so resizing needs no
    bookkeeping and handles can never be dragged out of reach. With a background
    image the plot area is that image, so fractions are also chip coordinates.
    """

    curve_changed = pyqtSignal()

    _MARGIN = 20.0
    _HIT_RADIUS = 12.0
    _RADIUS = 6.0

    _CURVE_COLOR = QColor(31, 119, 180)
    _END_COLOR = QColor(44, 62, 80)
    _HANDLE_COLOR = QColor(230, 126, 34)
    _MARKER_COLOR = QColor(46, 204, 113)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the editor showing a straight horizontal line."""
        super().__init__(parent)
        self.setMinimumSize(320, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._drag_index: int | None = None
        self._points = self._straight_points()
        self._background: QImage | None = None
        self._overlay: QImage | None = None
        self._markers: list[QPointF] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def control_points(self) -> tuple[QPointF, QPointF, QPointF, QPointF]:
        """Return the four control points as 0..1 fractions of the plot area."""
        return tuple(QPointF(p) for p in self._points)

    def set_control_points(
        self, p0: QPointF, c1: QPointF, c2: QPointF, p3: QPointF
    ) -> None:
        """Replace the curve; points are 0..1 fractions of the plot area."""
        self._points = [QPointF(p0), QPointF(c1), QPointF(c2), QPointF(p3)]
        self.update()
        self.curve_changed.emit()

    def reset(self) -> None:
        """Restore the straight line with the handles back at 1/3 and 2/3."""
        self.set_control_points(*self._straight_points())

    def set_background(self, image: QImage | None) -> None:
        """Draw ``image`` stretched over the plot area (None clears it)."""
        self._background = image
        self.update()

    def set_overlay(self, image: QImage | None) -> None:
        """Draw a translucent ``image`` over the background (None clears it)."""
        self._overlay = image
        self.update()

    def set_markers(self, points: list[QPointF]) -> None:
        """Mark 0..1 positions, e.g. the imaged target centroid."""
        self._markers = [QPointF(p) for p in points]
        self.update()

    def curve_path(self) -> QPainterPath:
        """Return the curve as a QPainterPath in widget pixels."""
        p0, c1, c2, p3 = (self._to_pixels(p) for p in self._points)
        path = QPainterPath(p0)
        path.cubicTo(c1, c2, p3)
        return path

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _straight_points() -> list[QPointF]:
        """Control points of a horizontal line, inner handles at 1/3 and 2/3."""
        return [
            QPointF(0.0, 0.5),
            QPointF(1.0 / 3.0, 0.5),
            QPointF(2.0 / 3.0, 0.5),
            QPointF(1.0, 0.5),
        ]

    def _plot_rect(self) -> QRectF:
        """Drawing area, inset so handles never sit flush against the edge."""
        m = self._MARGIN
        return QRectF(
            m, m, max(1.0, self.width() - 2 * m), max(1.0, self.height() - 2 * m)
        )

    def _to_pixels(self, point: QPointF) -> QPointF:
        """Map a 0..1 point to widget pixels."""
        rect = self._plot_rect()
        return QPointF(
            rect.left() + point.x() * rect.width(),
            rect.top() + point.y() * rect.height(),
        )

    def _from_pixels(self, point: QPointF) -> QPointF:
        """Map widget pixels back to a 0..1 point, clamped to the plot area."""
        rect = self._plot_rect()
        return QPointF(
            min(max((point.x() - rect.left()) / rect.width(), 0.0), 1.0),
            min(max((point.y() - rect.top()) / rect.height(), 0.0), 1.0),
        )

    def _index_at(self, pos: QPointF) -> int | None:
        """Index of the handle under *pos*, or None."""
        best_index: int | None = None
        best_d2 = self._HIT_RADIUS**2
        # Inner handles first, so one resting on an endpoint still wins the hit.
        for index in (1, 2, 0, 3):
            delta = self._to_pixels(self._points[index]) - pos
            d2 = delta.x() ** 2 + delta.y() ** 2
            if d2 <= best_d2:
                best_index, best_d2 = index, d2
        return best_index

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        """Grab the handle under the cursor."""
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        self._drag_index = self._index_at(_event_pos(event))
        if self._drag_index is None:
            super().mousePressEvent(event)
            return
        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        """Move the grabbed handle."""
        if self._drag_index is None:
            super().mouseMoveEvent(event)
            return
        self._points[self._drag_index] = self._from_pixels(_event_pos(event))
        self.update()
        self.curve_changed.emit()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        """Release the grabbed handle."""
        if self._drag_index is None or event.button() != Qt.MouseButton.LeftButton:
            super().mouseReleaseEvent(event)
            return
        self._drag_index = None
        self.unsetCursor()
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:
        """Draw the chip, the leader lines, the curve, the handles and markers."""
        p0, c1, c2, p3 = (self._to_pixels(p) for p in self._points)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(252, 252, 252))
        if self._background is not None:
            painter.drawImage(self._plot_rect(), self._background)
        if self._overlay is not None:
            painter.drawImage(self._plot_rect(), self._overlay)

        painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.PenStyle.DotLine))
        painter.drawLine(p0, c1)
        painter.drawLine(p3, c2)

        curve_pen = QPen(self._CURVE_COLOR, 2.5)
        curve_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(curve_pen)
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        painter.drawPath(self.curve_path())

        for index, pos in enumerate((p0, c1, c2, p3)):
            inner = index in (1, 2)
            color = self._HANDLE_COLOR if inner else self._END_COLOR
            radius = self._RADIUS + (2.0 if index == self._drag_index else 0.0)
            painter.setPen(QPen(color, 2))
            painter.setBrush(QBrush(QColor(255, 255, 255) if inner else color))
            painter.drawEllipse(pos, radius, radius)

        painter.setPen(QPen(self._MARKER_COLOR, 2))
        painter.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        for marker in self._markers:
            pos = self._to_pixels(marker)
            painter.drawLine(pos + QPointF(-5, -5), pos + QPointF(5, 5))
            painter.drawLine(pos + QPointF(-5, 5), pos + QPointF(5, -5))

        painter.end()


class CurveEditorDialog(QDialog):
    """Non-modal window: fit the curve to a moving target in an SLC chip.

    Load view reads the active ICEYE SLC layer under the current canvas extent; Use as
    target finds the hull along the curve and emits ``target_located(target, layer)``
    for the Mover Relocation tool. Without ``iface`` the dialog is a plain curve editor.
    """

    target_located = pyqtSignal(object, object)  # ImagedTarget, QgsRasterLayer

    def __init__(
        self,
        iface=None,
        metadata_provider: MetadataProvider | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the dialog around a fresh curve editor."""
        super().__init__(parent)
        self.iface = iface
        self.metadata_provider = metadata_provider or MetadataProvider()
        self.chip: SlcChip | None = None
        self.layer: QgsRasterLayer | None = None
        self.display_height = 0.0
        self.last_target: ImagedTarget | None = None

        self.setWindowTitle(_tr("Curve Editor"))
        self.setMinimumSize(520, 460)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        self.editor = CurveEditorWidget(self)

        load_btn = QPushButton(_tr("Load view"))
        load_btn.setToolTip(
            _tr("Read the active ICEYE SLC layer under the current map view")
        )
        load_btn.clicked.connect(self.load_from_canvas)
        self._chip_label = QLabel(_tr("No SLC loaded (columns: azimuth, rows: range)"))
        self._chip_label.setWordWrap(True)

        self.corridor_spin = QDoubleSpinBox()
        self.corridor_spin.setRange(1.0, 500.0)
        self.corridor_spin.setSuffix(" m")
        self.corridor_spin.setValue(RelocationParameters.corridor_half_width_m)
        self.corridor_spin.setToolTip(_tr("Half-width of the target corridor"))

        self.search_button = QPushButton(_tr("Use as target"))
        self.search_button.setToolTip(
            _tr(
                "Find the target along the curve and relocate it with the Mover "
                "Relocation tool"
            )
        )
        self.search_button.setDefault(True)
        self.search_button.clicked.connect(self._on_search_clicked)

        reset_btn = QPushButton(_tr("Reset"))
        reset_btn.clicked.connect(self.editor.reset)

        close_btn = QPushButton(_tr("Close"))
        close_btn.clicked.connect(self.close)

        self._status = QLabel()
        self._status.setWordWrap(True)
        self._status.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        top = QHBoxLayout()
        top.addWidget(load_btn)
        top.addWidget(self._chip_label, 1)

        form = QFormLayout()
        form.addRow(_tr("Corridor half-width"), self.corridor_spin)

        buttons = QHBoxLayout()
        buttons.addWidget(self.search_button)
        buttons.addStretch(1)
        buttons.addWidget(reset_btn)
        buttons.addWidget(close_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.addLayout(top)
        root.addWidget(self.editor, 1)
        root.addLayout(form)
        root.addWidget(self._status)
        root.addLayout(buttons)

    # ------------------------------------------------------------------
    # Chip
    # ------------------------------------------------------------------

    def parameters(self) -> RelocationParameters:
        """Hull detection parameters from the dialog controls."""
        return RelocationParameters(corridor_half_width_m=self.corridor_spin.value())

    def set_chip(
        self,
        chip: SlcChip,
        layer: QgsRasterLayer | None = None,
        display_height: float | None = None,
    ) -> None:
        """Show ``chip`` behind the curve and clear any previous result."""
        self.chip = chip
        self.layer = layer
        self.display_height = (
            chip.geometry.scene_height if display_height is None else display_height
        )
        self.last_target = None
        self.editor.set_background(chip_display_image(chip.data))
        self.editor.set_overlay(None)
        self.editor.set_markers([])
        rows, cols = chip.shape
        self._chip_label.setText(
            _tr(
                "{cols} azimuth x {rows} range samples at pixel ({x}, {y}) "
                "(columns: azimuth, rows: range)"
            ).format(cols=cols, rows=rows, x=chip.col0, y=chip.row0)
        )
        self._status.setText(
            _tr("Bend the curve along the target, then Use as target.")
        )

    def load_from_canvas(self) -> bool:
        """Read the active SLC layer under the canvas extent into the editor."""
        if self.iface is None:
            self._status.setText(_tr("No QGIS interface available."))
            return False
        layer = self.iface.activeLayer()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            extent = canvas_extent_in_layer_crs(self.iface.mapCanvas(), layer)
            chip = read_slc_chip(layer, extent, self.metadata_provider)
            source = layer.dataProvider().dataSourceUri()
            height = gcp_mean_height(source, chip.geometry.scene_height)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Curve editor failed to load SLC: {e}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )
            self._status.setText(str(e))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.set_chip(chip, layer, height)
        return True

    # ------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------

    def _on_search_clicked(self) -> None:
        """Locate the target's hull along the curve and hand it on."""
        if self.chip is None:
            self._status.setText(_tr("Load an SLC view first."))
            return
        try:
            target = locate_imaged_target(
                self.chip,
                self.editor.control_points(),
                self.display_height,
                self.parameters(),
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Target detection failed: {e}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )
            self._status.setText(_tr("Target detection failed: {e}").format(e=e))
            return
        self.last_target = target
        rows, cols = self.chip.shape
        self.editor.set_markers(
            [QPointF(target.col / max(cols - 1, 1), target.row / max(rows - 1, 1))]
        )
        self.editor.set_overlay(mask_overlay_image(target.hull_mask))
        lon, lat = target.lonlat
        self._status.setText(
            _tr(
                "Imaged position {lon:.6f}, {lat:.6f} (half-extent {ext:.1f} m). "
                "Click the constraint in the Mover Relocation tool."
            ).format(lon=lon, lat=lat, ext=target.half_extent_m)
        )
        self.target_located.emit(target, self.layer)
