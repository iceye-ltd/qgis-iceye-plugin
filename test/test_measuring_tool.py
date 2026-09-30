"""Tests for the measuring toolbar's map-tool handling."""

from __future__ import annotations

from qgis.gui import QgsMapToolPan

from iceye_toolbox.core.metadata import MetadataProvider
from iceye_toolbox.gui.measuring_tool import MeasuringToolbarAction


class TestPolicyDisableKeepsOtherTools:
    """Turning the measuring tools off must not replace another active map tool.

    The toolbar policy turns them off whenever the active layer stops being an
    ICEYE layer (e.g. a new memory layer is added), even when they are not in use.
    """

    def test_disable_leaves_foreign_tool_active(self, qgis_iface):
        """IRF and height-ruler disable hooks leave an unrelated tool in place."""
        action = MeasuringToolbarAction(
            qgis_iface, metadata_provider=MetadataProvider()
        )
        action.setup()
        canvas = qgis_iface.mapCanvas()
        other = QgsMapToolPan(canvas)
        canvas.setMapTool(other)
        try:
            action._on_irf_policy_disabled()
            action._on_height_ruler_policy_disabled()
            assert canvas.mapTool() is other
        finally:
            canvas.unsetMapTool(other)
            action.unload()
