"""Tests for the Mover Relocation panel and map tool."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qgis.core import Qgis, QgsMarkerLineSymbolLayer, QgsPointXY, QgsProject

from iceye_toolbox.core.metadata import MetadataProvider
from iceye_toolbox.core.mover_relocation import ConstraintAxis
from iceye_toolbox.core.target_finder import ecef_to_lonlat
from iceye_toolbox.gui.mover_relocation_tool import (
    STEP_CONSTRAINT,
    STEP_DONE,
    MoverRelocationDialog,
    MoverScene,
    OutputLayers,
    target_from_click,
)

from .test_mover_relocation import _ground_velocity, simulate_mover

CENTRE_COL, CENTRE_ROW = 4271.5, 434.5


@pytest.fixture
def slc_layer(qgis_iface, base_crop_layer):
    """Add the crop fixture as the active layer."""
    project = QgsProject.instance()
    project.addMapLayer(base_crop_layer)
    qgis_iface.setActiveLayer(base_crop_layer)
    qgis_iface.mapCanvas().setDestinationCrs(base_crop_layer.crs())
    yield base_crop_layer
    project.removeAllMapLayers()


@pytest.fixture
def dialog(qgis_iface, slc_layer):
    """Mover Relocation panel waiting for a target, hull snapping off."""
    panel = MoverRelocationDialog(qgis_iface, metadata_provider=MetadataProvider())
    panel.detect_check.setChecked(False)
    assert panel.start(), panel._step_label.text()
    yield panel
    panel.close()


def _point(point: np.ndarray) -> QgsPointXY:
    return QgsPointXY(*ecef_to_lonlat(point))


def _mover(scene: MoverScene, heading: float = 100.0, speed: float = 10.0):
    """Simulate a car on a straight road through the fixture centre."""
    p0 = scene.ecef(*scene.pixel_to_lonlat(CENTRE_COL, CENTRE_ROW))
    mover = simulate_mover(scene.geometry, p0, _ground_velocity(p0, speed, heading))
    p_img = scene.geometry.orbit.geocode(
        mover.r_img, mover.t_img, scene.display_height, p0
    )
    u = mover.velocity / np.linalg.norm(mover.velocity)
    return mover, _point(p_img), lambda offset: _point(p0 + offset * u)


def test_click_snaps_to_hull(slc_layer):
    """A target click reads the SLC around it and snaps to a nearby hull."""
    scene = MoverScene.from_layer(slc_layer)
    assert scene.display_height == pytest.approx(-2.653, abs=1e-3)
    lon, lat = scene.pixel_to_lonlat(CENTRE_COL, CENTRE_ROW)
    target = target_from_click(scene, lon, lat, MetadataProvider())
    assert np.linalg.norm(target.position - scene.ecef(lon, lat)) < 16.0


class TestMoverRelocationDialog:
    """Click workflows on the fixture layer with simulated cars."""

    def test_two_click_relocation(self, qgis_iface, slc_layer, dialog):
        """Target click, band, two road clicks, outputs; opposite cars read apart.

        New layers are made active and knock out the map tool, as QGIS and the
        toolbar policy can do; the tool must keep both the SLC and its map tool.
        """
        canvas = qgis_iface.mapCanvas()

        def knock_out_tool(layers):
            qgis_iface.setActiveLayer(layers[-1])
            canvas.unsetMapTool(canvas.mapTool())

        QgsProject.instance().layersAdded.connect(knock_out_tool)
        try:
            texts = []
            for heading in (100.0, 280.0):
                mover, imaged, on_road = _mover(dialog.scene, heading)
                dialog.handle_click(imaged)
                assert dialog.step == STEP_CONSTRAINT
                assert dialog.outputs.layer("band").featureCount() == 1
                assert canvas.mapTool() is dialog.map_tool
                dialog.handle_move(imaged)
                assert "|v_r| 0.0" in dialog._readout.text()
                dialog.handle_click(on_road(-60.0))
                dialog.handle_click(on_road(60.0))
                assert dialog.step == STEP_DONE, dialog._result.text()
                result = dialog.last_result
                assert result.v_t == pytest.approx(10.0, rel=0.02)
                assert math.copysign(1, result.v_r) == math.copysign(1, mover.v_r)
                texts.append(dialog._result.text())
        finally:
            QgsProject.instance().layersAdded.disconnect(knock_out_tool)

        assert qgis_iface.activeLayer() is slc_layer
        assert sum("approaching the radar" in t for t in texts) == 1
        assert sum("receding from the radar" in t for t in texts) == 1
        assert dialog.outputs.layer("band") is None
        assert dialog.outputs.layer("ticks") is None
        assert dialog.outputs.layer("true").featureCount() == 2
        track = dialog.outputs.layer("track")
        arrows = [
            sl
            for sl in track.renderer().symbol().symbolLayers()
            if isinstance(sl, QgsMarkerLineSymbolLayer)
        ]
        assert arrows[0].placements() == Qgis.MarkerLinePlacement.LastVertex

    @pytest.mark.parametrize("coherence", [0.8, 0.05])
    def test_single_click(self, dialog, monkeypatch, coherence):
        """One constraint click: rough heading from a clear axis, else minimum speed."""
        import iceye_toolbox.gui.mover_relocation_tool as tool

        h = math.radians(100.0)
        monkeypatch.setattr(
            tool,
            "estimate_constraint_axis",
            lambda *a, **k: ConstraintAxis(
                np.array([math.sin(h), math.cos(h)]), coherence
            ),
        )
        dialog.single_check.setChecked(True)
        _, imaged, on_road = _mover(dialog.scene)
        dialog.handle_click(imaged)
        dialog.handle_click(on_road(0.0))
        assert dialog.step == STEP_DONE, dialog._result.text()
        result = dialog.last_result
        if coherence > 0.5:
            assert result.mode == "single_click_auto"
            assert result.v_t == pytest.approx(10.0, rel=0.02)
            assert dialog._points_rb.numberOfVertices() == 2
        else:
            assert result.mode == "single_click" and result.v_t is None
            assert "No clear linear feature" in dialog._result.text()

    def test_band_cleanup_and_reset(self, qgis_iface, slc_layer, dialog):
        """Leftover and cancelled bands disappear; Reset removes every Mover layer."""
        _, imaged, on_road = _mover(dialog.scene)
        earlier = MoverRelocationDialog(
            qgis_iface, metadata_provider=MetadataProvider()
        )
        earlier.detect_check.setChecked(False)
        earlier.start()
        earlier.handle_click(imaged)
        earlier.close()
        dialog.start()

        dialog.handle_click(imaged)
        assert len(OutputLayers.layers("band")) == 1
        dialog.map_tool.cancelled.emit()
        assert dialog.outputs.layer("band") is None

        dialog.handle_click(imaged)
        dialog.handle_click(on_road(-60.0))
        dialog.handle_click(on_road(60.0))
        assert OutputLayers.layers("band") == []
        dialog.reset_all()
        assert OutputLayers.layers() == []
        assert QgsProject.instance().mapLayer(slc_layer.id()) is slc_layer

    def test_constraint_must_straddle(self, dialog):
        """Two clicks on one side are rejected and the constraint step restarts."""
        _, imaged, on_road = _mover(dialog.scene)
        dialog.handle_click(imaged)
        dialog.handle_click(on_road(20.0))
        dialog.handle_click(on_road(60.0))
        assert dialog.step == STEP_CONSTRAINT
        assert "straddle" in dialog._result.text()
        assert dialog.outputs.layer("true") is None
