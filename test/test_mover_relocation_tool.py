"""Tests for the Mover Relocation panel and map tool (two-click workflow)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from qgis.core import QgsPointXY, QgsProject

from iceye_toolbox.core.metadata import MetadataProvider
from iceye_toolbox.core.mover_relocation import FLAG_BAND_CLIPPED, INDICATOR_AMBER
from iceye_toolbox.core.target_finder import ecef_to_geodetic, lonlat_to_ecef
from iceye_toolbox.gui.curve_editor import CurveEditorDialog
from iceye_toolbox.gui.mover_relocation_tool import (
    STEP_CONSTRAINT,
    STEP_DONE,
    MoverRelocationDialog,
    MoverScene,
    target_from_click,
)

from .test_curve_editor import _zoom_to_centre
from .test_mover_relocation import _ground_velocity, simulate_mover

CENTRE_COL, CENTRE_ROW = 4271.5, 434.5


@pytest.fixture
def slc_layer(qgis_iface, base_crop_layer):
    """Add the crop fixture as the active layer, zoomed to its centre."""
    project = QgsProject.instance()
    project.addMapLayer(base_crop_layer)
    qgis_iface.setActiveLayer(base_crop_layer)
    _zoom_to_centre(qgis_iface, base_crop_layer)
    yield base_crop_layer
    project.removeAllMapLayers()


def _lonlat_point(point: np.ndarray) -> QgsPointXY:
    lat, lon, _ = ecef_to_geodetic(point)
    return QgsPointXY(lon, lat)


def _mover(scene: MoverScene, speed: float = 10.0, heading: float = 100.0):
    """Simulate a car on a straight road through the fixture centre."""
    lon, lat = scene.pixel_to_lonlat(CENTRE_COL, CENTRE_ROW)
    p0 = scene.ecef(lon, lat)
    mover = simulate_mover(scene.geometry, p0, _ground_velocity(p0, speed, heading))
    p_img = scene.geometry.orbit.geocode(
        mover.r_img, mover.t_img, scene.display_height, p0
    )
    u = mover.velocity / np.linalg.norm(mover.velocity)

    def on_road(offset: float) -> QgsPointXY:
        return _lonlat_point(p0 + offset * u)

    return mover, _lonlat_point(p_img), on_road


class TestMoverScene:
    """Display surface and image extent of the SLC layer."""

    def test_display_height_is_gcp_height(self, slc_layer):
        """The GCP-warped display lies ~0.35 m above the average scene height."""
        scene = MoverScene.from_layer(slc_layer)
        assert scene.display_height == pytest.approx(-2.653, abs=1e-3)
        assert scene.geometry.scene_height == -3.0

    def test_click_snaps_to_hull(self, slc_layer):
        """A click reads a chip around itself and returns a nearby hull centroid."""
        scene = MoverScene.from_layer(slc_layer)
        lon, lat = scene.pixel_to_lonlat(CENTRE_COL, CENTRE_ROW)
        target = target_from_click(scene, lon, lat, MetadataProvider())
        assert target.hull_mask is not None
        moved = np.linalg.norm(target.position - scene.ecef(lon, lat))
        assert moved < 16.0
        plain = target_from_click(
            scene, lon, lat, MetadataProvider(), detect_hull=False
        )
        assert np.linalg.norm(plain.position - scene.ecef(lon, lat)) < 1e-6


class TestMoverRelocationDialog:
    """Two-click workflow on the fixture layer with a simulated car."""

    def _dialog(self, qgis_iface) -> MoverRelocationDialog:
        dialog = MoverRelocationDialog(qgis_iface, metadata_provider=MetadataProvider())
        dialog.detect_check.setChecked(False)
        return dialog

    def test_start_requires_slc_layer(self, qgis_iface, monkeypatch):
        """Without an SLC layer the tool explains what to select."""
        monkeypatch.setattr(qgis_iface, "activeLayer", lambda: None)
        dialog = self._dialog(qgis_iface)
        assert not dialog.start()
        assert "SLC" in dialog._step_label.text()

    def test_two_click_relocation(self, qgis_iface, slc_layer):
        """Target click draws the band; two road clicks give the true position."""
        dialog = self._dialog(qgis_iface)
        assert dialog.start(), dialog._step_label.text()
        assert qgis_iface.mapCanvas().mapTool() is dialog.map_tool
        mover, imaged, on_road = _mover(dialog.scene)

        dialog.handle_click(imaged)
        assert dialog.step == STEP_CONSTRAINT
        band = dialog.outputs.layer("band")
        ticks = dialog.outputs.layer("ticks")
        assert band.featureCount() == 1 and ticks.featureCount() > 0
        assert dialog.outputs.layer("imaged").featureCount() == 1
        # The 0.05 s crop is far shorter than a car band: clipped.
        assert next(band.getFeatures())["clipped"]
        assert dialog.outputs.layer("true") is None

        dialog.handle_move(imaged)
        assert "|v_r| 0.0" in dialog._readout.text()

        dialog.handle_click(on_road(-60.0))
        dialog.handle_click(on_road(60.0))
        assert dialog.step == STEP_DONE, dialog._result.text()
        result = dialog.last_result
        assert result.v_t == pytest.approx(10.0, rel=0.02)
        assert math.copysign(1, result.v_r) == math.copysign(1, mover.v_r)
        assert result.flags == [FLAG_BAND_CLIPPED]
        assert result.indicator == INDICATOR_AMBER
        assert "AMBER" in dialog._result.text()

        true_layer = dialog.outputs.layer("true")
        feature = next(true_layer.getFeatures())
        assert feature["target_class"] == "car"
        assert feature["cue"] == "road"
        assert feature["indicator"] == INDICATOR_AMBER
        assert feature["v_t"] == pytest.approx(result.v_t)
        assert dialog.outputs.layer("displacement").featureCount() == 1
        assert dialog.outputs.layer("track").featureCount() == 1

        true_ecef = lonlat_to_ecef(*result.true_lonlat, dialog.scene.display_height)
        assert np.linalg.norm(true_ecef - mover.p_true) < 0.5

        dialog.close()
        assert qgis_iface.mapCanvas().mapTool() is not dialog.map_tool

    def test_constraint_must_straddle(self, qgis_iface, slc_layer):
        """Two clicks on one side are rejected and the constraint step restarts."""
        dialog = self._dialog(qgis_iface)
        assert dialog.start()
        _, imaged, on_road = _mover(dialog.scene)
        dialog.handle_click(imaged)
        dialog.handle_click(on_road(20.0))
        dialog.handle_click(on_road(60.0))
        assert dialog.step == STEP_CONSTRAINT
        assert "straddle" in dialog._result.text()
        assert dialog.clicks == []
        assert dialog.outputs.layer("true") is None

    def test_curve_editor_hands_over_target(self, qgis_iface, slc_layer):
        """Use as target in the Curve Editor feeds the relocation panel."""
        editor = CurveEditorDialog(qgis_iface, metadata_provider=MetadataProvider())
        assert editor.load_from_canvas(), editor._status.text()
        dialog = self._dialog(qgis_iface)
        editor.target_located.connect(dialog.set_target)
        editor._on_search_clicked()
        assert editor.last_target is not None, editor._status.text()
        assert dialog.target is editor.last_target
        assert dialog.step == STEP_CONSTRAINT
        assert dialog.outputs.layer("band").featureCount() == 1
