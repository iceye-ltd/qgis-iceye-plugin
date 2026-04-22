# coding=utf-8
"""Tests for auto_styler module."""

import uuid

import numpy as np
import pytest
from osgeo import gdal
from qgis.core import QgsProject, QgsRasterLayer

from iceye_toolbox.core.auto_styler import AutoStyler, _find_alpha_band

# Import resources so Qt resource paths work in tests
from iceye_toolbox.resources import resources

# QGIS enum values for minMaxOrigin
# CumulativeCut=3; extent: UpdatedCanvas=1, CurrentCanvas=2 (both are dynamic)
CUMULATIVE_CUT = 3
DYNAMIC_EXTENTS = (1, 2)  # UpdatedCanvas, CurrentCanvas


def _create_raster_with_alpha():
    """Create a temporary 2-band raster (grayscale + alpha) for testing."""
    path = f"/vsimem/autostyler_alpha_test_{uuid.uuid4().hex}.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(path, 10, 10, 2, gdal.GDT_Byte)
    band1_data = np.array(
        [[i % 256 for i in range(10)] for _ in range(10)], dtype=np.uint8
    )
    band2_data = np.full((10, 10), 255, dtype=np.uint8)  # fully opaque alpha
    ds.GetRasterBand(1).WriteArray(band1_data)
    ds.GetRasterBand(2).WriteArray(band2_data)
    ds.GetRasterBand(2).SetColorInterpretation(gdal.GCI_AlphaBand)
    ds.SetGeoTransform([0, 1, 0, 0, 0, -1])
    ds.SetProjection(
        'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
    )
    ds = None
    return path


class TestAutoStyler:
    """Tests for AutoStyler class."""

    def test_applies_style_to_iceye_layer(self, base_crop_layer, qgis_iface):
        """AutoStyler should apply singlebandgray style to ICEYE raster layers."""
        styler = AutoStyler(qgis_iface)
        layer = base_crop_layer
        initial_renderer = type(layer.renderer()).__name__

        # Add layer to project - this triggers layerWasAdded signal
        QgsProject.instance().addMapLayer(layer)

        # Verify renderer changed to singlebandgray
        styled_renderer = type(layer.renderer()).__name__

        print(f"initial_renderer: {initial_renderer}")
        print(f"styled_renderer: {styled_renderer}")
        print(f"styler: {styler}")  # add this so its ignored in ruff w/o changing rules
        assert styled_renderer == "QgsSingleBandGrayRenderer", (
            f"Expected QgsSingleBandGrayRenderer, got {styled_renderer}"
        )
        assert initial_renderer != styled_renderer, (
            f"Renderer should have changed from {initial_renderer}"
        )

        # Cleanup
        QgsProject.instance().removeMapLayer(layer.id())

    def test_layer_without_alpha_keeps_alpha_band_unset(
        self, base_crop_layer, qgis_iface
    ):
        """Layer without alpha band should have renderer.alphaBand() == -1."""
        _ = AutoStyler(qgis_iface)
        layer = base_crop_layer
        QgsProject.instance().addMapLayer(layer)

        renderer = layer.renderer()
        assert renderer.alphaBand() == -1, "No alpha band: alphaBand should be -1"

        QgsProject.instance().removeMapLayer(layer.id())

    def test_layer_with_alpha_band_gets_alpha_set(self, qgis_iface):
        """Layer with alpha band should have renderer.alphaBand() set to that band."""
        path = _create_raster_with_alpha()
        try:
            _ = AutoStyler(qgis_iface)
            layer = QgsRasterLayer(path, "ICEYE_TEST_ALPHA", "gdal")
            assert layer.isValid(), "Test raster with alpha should load"
            # Rename to trigger ICEYE detection (is_iceye_layer checks name/source)
            layer.setName("ICEYE_TEST_ALPHA")

            alpha_band = _find_alpha_band(layer)
            assert alpha_band == 2, f"Expected alpha band 2, got {alpha_band}"

            QgsProject.instance().addMapLayer(layer)

            renderer = layer.renderer()
            assert renderer.alphaBand() == 2, "Alpha band should be set to 2"

            QgsProject.instance().removeMapLayer(layer.id())
        finally:
            gdal.Unlink(path)

    def test_dynamic_min_max_uses_min_max_origin(self, base_crop_layer, qgis_iface):
        """Styled layer should use minMaxOrigin (CumulativeCut) for dynamic stretch."""
        _ = AutoStyler(qgis_iface)
        layer = base_crop_layer
        QgsProject.instance().addMapLayer(layer)

        renderer = layer.renderer()
        min_max_origin = renderer.minMaxOrigin()
        assert min_max_origin.limits() == CUMULATIVE_CUT, (
            "minMaxOrigin limits should be CumulativeCut"
        )
        assert min_max_origin.extent() in DYNAMIC_EXTENTS, (
            "minMaxOrigin extent should be UpdatedCanvas or CurrentCanvas"
        )

        QgsProject.instance().removeMapLayer(layer.id())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
