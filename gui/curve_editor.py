"""Curve editor: a line you bend by dragging its ends and its 1/3 and 2/3 handles.

Fitted over an SLC chip of the current map view, the curve marks an imaged (displaced)
moving ship; Search relocates it with ``core.target_finder.relocate_mover`` and writes
the true position and the imaged-to-true displacement to memory layers.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QMetaType,
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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..core.cropper import get_extend_image_coords
from ..core.metadata import MetadataProvider
from ..core.target_finder import (
    MoverEstimate,
    ProductGeometry,
    RelocationParameters,
    SlcChip,
    gcp_pixel_to_lonlat,
    patch_to_file_layout,
    read_iceye_properties,
    relocate_mover,
)
from ..core.typing_compat import NDArray
from .lens_tool import read_slc_data

# Largest chip read from the view (complex samples); ~160 MB of complex64.
MAX_CHIP_PIXELS = 20_000_000
# Longest side of the background image drawn in the editor.
_DISPLAY_SIZE = 1024

_POINT_LAYER_NAME = "Mover true positions"
_LINE_LAYER_NAME = "Mover displacements"
_STRING_FIELDS = frozenset({"method", "flags"})


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
    hull: NDArray[np.bool_] | None, ring: NDArray[np.bool_] | None
) -> QImage | None:
    """Translucent image of the hull (orange) and clutter ring (cyan) masks."""
    if hull is None or ring is None:
        return None
    hull_small = _block_reduce(hull, _DISPLAY_SIZE, np.any)
    ring_small = _block_reduce(ring, _DISPLAY_SIZE, np.any)
    rgba = np.zeros((*hull_small.shape, 4), dtype=np.uint8)
    rgba[ring_small] = (0, 170, 200, 60)
    rgba[hull_small] = (230, 126, 34, 150)
    rgba = np.ascontiguousarray(rgba)
    rows, cols = hull_small.shape
    return QImage(rgba.data, cols, rows, cols * 4, QImage.Format.Format_RGBA8888).copy()


def format_estimate(estimate: MoverEstimate) -> str:
    """Short multi-line summary of a relocation result."""

    def opt(value: float | None, fmt: str) -> str:
        return "n/a" if value is None else format(value, fmt)

    lines = [
        _tr("Method: {method}, confidence {conf:.2f}").format(
            method=estimate.method, conf=estimate.confidence
        ),
        _tr("v_r {v_r:.2f} m/s, v_gr {v_gr:.2f} m/s, dx {dx:.1f} m").format(
            v_r=estimate.v_r, v_gr=estimate.v_gr, dx=estimate.dx_m
        ),
        _tr("Speed {speed} m/s, heading {heading} deg, v_along {va} m/s").format(
            speed=opt(estimate.speed, ".2f"),
            heading=opt(estimate.heading_deg, ".0f"),
            va=opt(estimate.v_along, ".2f"),
        ),
        _tr(
            "Doppler: ship {fs:.0f} Hz, ring {fr:.0f} Hz, metadata {fm:.0f} Hz, "
            "amplification {amp:.0f}"
        ).format(
            fs=estimate.f_ship_hz,
            fr=estimate.f_ref_ring_hz,
            fm=estimate.f_ref_meta_hz,
            amp=estimate.amplification,
        ),
        _tr("Looks found: {n}; flags: {flags}").format(
            n=len(estimate.targets), flags=", ".join(estimate.flags) or "-"
        ),
    ]
    return "\n".join(lines)


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
        """Mark 0..1 positions, e.g. the ship found in each sub-aperture look."""
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
    """Non-modal window: fit the curve to a ship in an SLC chip, then relocate it.

    Load view reads the active ICEYE SLC layer under the current canvas extent;
    Search estimates the ship's true position and adds it to two memory layers.
    Without ``iface`` the dialog is a plain curve editor.
    """

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
        self.last_estimate: MoverEstimate | None = None
        self._point_layer_id: str | None = None
        self._line_layer_id: str | None = None

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
        self.corridor_spin.setToolTip(_tr("Half-width of the ship corridor"))
        self.looks_spin = QSpinBox()
        self.looks_spin.setRange(2, 16)
        self.looks_spin.setValue(RelocationParameters.n_looks)
        self.looks_spin.setToolTip(_tr("Sub-aperture looks for the map-drift estimate"))

        self.search_button = QPushButton(_tr("Search"))
        self.search_button.setToolTip(
            _tr("Estimate the true position of the ship along the curve")
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
        form.addRow(_tr("Looks"), self.looks_spin)

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
        """Relocation parameters from the dialog controls."""
        return RelocationParameters(
            corridor_half_width_m=self.corridor_spin.value(),
            n_looks=self.looks_spin.value(),
        )

    def set_chip(self, chip: SlcChip) -> None:
        """Show ``chip`` behind the curve and clear any previous result."""
        self.chip = chip
        self.last_estimate = None
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
        self._status.setText(_tr("Bend the curve along the ship, then Search."))

    def load_from_canvas(self) -> bool:
        """Read the active SLC layer under the canvas extent into the editor."""
        if self.iface is None:
            self._status.setText(_tr("No QGIS interface available."))
            return False
        layer = self.iface.activeLayer()
        if not isinstance(layer, QgsRasterLayer) or layer.bandCount() < 2:
            self._status.setText(_tr("Select an ICEYE SLC layer first."))
            return False
        metadata = self.metadata_provider.get(layer)
        if metadata is None:
            self._status.setText(_tr("The active layer has no ICEYE metadata."))
            return False

        canvas = self.iface.mapCanvas()
        extent = canvas.extent()
        canvas_crs = canvas.mapSettings().destinationCrs()
        if canvas_crs != layer.crs():
            transform = QgsCoordinateTransform(
                canvas_crs, layer.crs(), QgsProject.instance()
            )
            extent = transform.transformBoundingBox(extent)

        bounds = get_extend_image_coords(layer, extent)
        if bounds is None:
            self._status.setText(_tr("Could not map the view to SLC pixels."))
            return False
        width, height = layer.width(), layer.height()
        if (
            bounds.xMinimum() < 0
            or bounds.yMinimum() < 0
            or bounds.xMaximum() > width
            or bounds.yMaximum() > height
        ):
            self._status.setText(_tr("Zoom in so the view lies inside the SLC."))
            return False
        if bounds.width() * bounds.height() > MAX_CHIP_PIXELS:
            self._status.setText(_tr("The view is too large; zoom in on the ship."))
            return False

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            slc = read_slc_data(layer, extent, self.metadata_provider)
            if slc is None:
                self._status.setText(_tr("Failed to read the SLC samples."))
                return False
            source = layer.dataProvider().dataSourceUri()
            left = (metadata.sar_observation_direction or "").lower() == "left"
            chip = SlcChip(
                data=patch_to_file_layout(slc.data_patch, left),
                col0=int(bounds.xMinimum()),
                row0=int(bounds.yMinimum()),
                geometry=ProductGeometry.from_properties(read_iceye_properties(source)),
                pixel_to_lonlat=gcp_pixel_to_lonlat(source),
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Curve editor failed to load SLC: {e}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )
            self._status.setText(_tr("Failed to load the SLC: {e}").format(e=e))
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self.set_chip(chip)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def _on_search_clicked(self) -> None:
        """Relocate the ship along the current curve."""
        if self.chip is None:
            self._status.setText(_tr("Load an SLC view first."))
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            estimate = relocate_mover(
                self.chip, self.editor.control_points(), self.parameters()
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Mover relocation failed: {e}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )
            self._status.setText(_tr("Search failed: {e}").format(e=e))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.last_estimate = estimate
        self.editor.set_markers(
            [QPointF(t.x_fraction, t.y_fraction) for t in estimate.targets]
        )
        self.editor.set_overlay(
            mask_overlay_image(estimate.hull_mask, estimate.ring_mask)
        )
        self._status.setText(format_estimate(estimate))
        if self.iface is not None:
            self._write_layers(estimate)

    # ------------------------------------------------------------------
    # Output layers
    # ------------------------------------------------------------------

    @staticmethod
    def _fields(estimate: MoverEstimate) -> QgsFields:
        fields = QgsFields()
        for name in estimate.attributes():
            kind = (
                QMetaType.Type.QString
                if name in _STRING_FIELDS
                else QMetaType.Type.Double
            )
            fields.append(QgsField(name, kind))
        return fields

    def _layer(
        self, layer_id: str | None, geometry: str, name: str, estimate
    ) -> QgsVectorLayer:
        """Existing output layer, or a new EPSG:4326 memory layer added to the project."""
        project = QgsProject.instance()
        layer = project.mapLayer(layer_id) if layer_id else None
        if isinstance(layer, QgsVectorLayer) and layer.isValid():
            return layer
        layer = QgsVectorLayer(f"{geometry}?crs=EPSG:4326", name, "memory")
        layer.dataProvider().addAttributes(self._fields(estimate))
        layer.updateFields()
        project.addMapLayer(layer)
        return layer

    def _write_layers(self, estimate: MoverEstimate) -> None:
        """Append the true position and the displacement line to the output layers."""
        points = self._layer(self._point_layer_id, "Point", _POINT_LAYER_NAME, estimate)
        lines = self._layer(
            self._line_layer_id, "LineString", _LINE_LAYER_NAME, estimate
        )
        self._point_layer_id, self._line_layer_id = points.id(), lines.id()

        imaged = QgsPointXY(*estimate.imaged_lonlat)
        true = QgsPointXY(*estimate.true_lonlat)
        attributes = estimate.attributes()
        for layer, geometry in (
            (points, QgsGeometry.fromPointXY(true)),
            (lines, QgsGeometry.fromPolylineXY([imaged, true])),
        ):
            feature = QgsFeature(layer.fields())
            for name, value in attributes.items():
                if layer.fields().indexOf(name) >= 0:
                    feature.setAttribute(name, value)
            feature.setGeometry(geometry)
            layer.dataProvider().addFeature(feature)
            layer.updateExtents()
            layer.triggerRepaint()
