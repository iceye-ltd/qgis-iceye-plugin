"""Tests for the curve editor dialog used to relocate moving ships."""

from __future__ import annotations

import numpy as np
from qgis.core import QgsProject, QgsRectangle
from qgis.PyQt.QtCore import QPointF

from iceye_toolbox.core.metadata import MetadataProvider
from iceye_toolbox.gui.curve_editor import (
    CurveEditorDialog,
    CurveEditorWidget,
    chip_display_image,
    mask_overlay_image,
)


def _zoom_to_centre(qgis_iface, layer, half_deg: float = 0.0005) -> None:
    canvas = qgis_iface.mapCanvas()
    canvas.setDestinationCrs(layer.crs())
    c = layer.extent().center()
    canvas.setExtent(
        QgsRectangle(
            c.x() - half_deg, c.y() - half_deg, c.x() + half_deg, c.y() + half_deg
        )
    )
    canvas.refresh()


class TestDisplayHelpers:
    """Chip and mask images drawn behind the curve."""

    def test_chip_display_image_is_bounded(self, qgis_iface):
        """Large chips are block-averaged to at most 1024 pixels per side."""
        data = np.ones((300, 5000), dtype=np.complex64)
        image = chip_display_image(data)
        assert image.width() <= 1024 and image.height() == 300

    def test_mask_overlay_image(self, qgis_iface):
        """Hull and ring masks become one translucent RGBA image."""
        hull = np.zeros((10, 20), bool)
        ring = np.zeros((10, 20), bool)
        hull[5, 5] = ring[1, 1] = True
        image = mask_overlay_image(hull, ring)
        assert image.width() == 20 and image.height() == 10
        assert image.pixelColor(5, 5).alpha() > image.pixelColor(1, 1).alpha() > 0
        assert mask_overlay_image(None, None) is None


class TestCurveEditorWidget:
    """Bezier handles and markers."""

    def test_reset_restores_straight_line(self, qgis_iface):
        """Reset puts the inner handles back at 1/3 and 2/3."""
        widget = CurveEditorWidget()
        widget.set_control_points(
            QPointF(0.1, 0.1), QPointF(0.2, 0.9), QPointF(0.8, 0.1), QPointF(0.9, 0.9)
        )
        widget.reset()
        points = widget.control_points()
        assert points[1] == QPointF(1 / 3, 0.5)
        assert points[2] == QPointF(2 / 3, 0.5)


class TestCurveEditorDialog:
    """Loading the SLC view and relocating along the curve."""

    def test_search_without_chip_reports(self, qgis_iface):
        """Search before Load view explains what to do."""
        dialog = CurveEditorDialog()
        dialog._on_search_clicked()
        assert "Load" in dialog._status.text()

    def test_load_view_and_search_writes_layers(self, qgis_iface, base_crop_layer):
        """The view is read as a chip and Search adds point and line features."""
        project = QgsProject.instance()
        project.addMapLayer(base_crop_layer)
        qgis_iface.setActiveLayer(base_crop_layer)
        _zoom_to_centre(qgis_iface, base_crop_layer)

        dialog = CurveEditorDialog(qgis_iface, metadata_provider=MetadataProvider())
        assert dialog.load_from_canvas(), dialog._status.text()
        rows, cols = dialog.chip.shape
        assert rows > 10 and cols > 100
        assert np.iscomplexobj(dialog.chip.data)

        dialog._on_search_clicked()
        estimate = dialog.last_estimate
        assert estimate is not None, dialog._status.text()
        assert "Method" in dialog._status.text()

        points = project.mapLayer(dialog._point_layer_id)
        lines = project.mapLayer(dialog._line_layer_id)
        assert points.featureCount() == 1
        assert lines.featureCount() == 1
        feature = next(points.getFeatures())
        assert feature["method"] == estimate.method
        assert "sign_unverified" in feature["flags"]

        # A second search appends to the same layers.
        dialog._on_search_clicked()
        assert points.featureCount() == 2
        project.removeMapLayers([points.id(), lines.id(), base_crop_layer.id()])
