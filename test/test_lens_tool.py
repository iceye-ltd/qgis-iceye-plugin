"""Tests for lens_tool module."""

from pathlib import Path

import numpy as np
import pytest
from qgis.core import QgsRasterLayer, QgsRectangle
from qgis.PyQt.QtGui import QImage

from iceye_toolbox.core.cropper import get_extend_image_coords
from iceye_toolbox.core.metadata import MetadataProvider
from iceye_toolbox.gui.lens_tool import (
    LensMapTool,
    _process_color_spectrum,
    _process_focus_data,
    compute_lens_extent,
    create_georeferenced_temp_raster,
    get_pixel_to_geo_corners,
)


@pytest.fixture
def test_render_file():
    """Path to the 4096-pixel render reference raster."""
    test_dir = Path(__file__).resolve().parent
    return str(test_dir / "test-output4096.tif")


@pytest.fixture
def test_render_layer(test_render_file):
    """QgsRasterLayer loaded from the 4096-pixel render reference raster."""
    layer = QgsRasterLayer(test_render_file, "ICEYE_TEST_RENDER")
    assert layer.isValid(), f"Failed to load test raster: {test_render_file}"
    return layer


class TestProcessColorSpectrum:
    """Tests for color spectrum processing helper."""

    def test_color_spectrum_processing(self, qgis_iface, complex_data, metadata):
        """Test color spectrum processing runs without error."""
        rgb = _process_color_spectrum(complex_data, metadata)

        assert rgb.ndim == 3, "Output should be 3D (height, width, 3)"
        assert rgb.shape[2] == 3, "Output should have 3 color channels"
        assert rgb.dtype == np.uint8, "Output should be uint8"
        assert rgb.size > 0, "Output should not be empty"


class TestProcessFocusData:
    """Tests for focus processing helper."""

    def test_basic_focus_processing(self, qgis_iface, complex_data, metadata):
        """Test focus data processing runs without error."""
        result = _process_focus_data(complex_data, metadata)

        assert result.ndim == 2, "Output should be 2D"
        assert result.dtype == np.uint8, "Output should be uint8"
        assert result.size > 0, "Output should not be empty"


class TestCreateGeoreferencedTempRaster:
    """Tests for create_georeferenced_temp_raster function."""

    def test_create_grayscale_raster(self, qgis_iface, base_crop_layer):
        """Test creating a grayscale georeferenced raster."""
        data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        extent = QgsRectangle(
            base_crop_layer.extent().xMinimum(),
            base_crop_layer.extent().yMinimum(),
            base_crop_layer.extent().xMinimum() + 0.001,
            base_crop_layer.extent().yMinimum() + 0.001,
        )
        pixel_bounds = get_extend_image_coords(base_crop_layer, extent)

        geo_corners = get_pixel_to_geo_corners(base_crop_layer, pixel_bounds)

        temp_path = create_georeferenced_temp_raster(
            data, geo_corners, base_crop_layer.crs().toWkt()
        )

        assert temp_path is not None, "Should create temp raster"
        assert temp_path.startswith("/vsimem/"), "Should use GDAL virtual filesystem"

        temp_layer = QgsRasterLayer(temp_path, "temp", "gdal")
        assert temp_layer.isValid(), "Created raster should be valid"
        assert temp_layer.bandCount() == 2, (
            "Should have 2 bands (1 for grayscale + 1 for alpha)"
        )

    def test_create_rgb_raster(self, qgis_iface, base_crop_layer):
        """Test creating an RGB georeferenced raster."""
        data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        extent = QgsRectangle(
            base_crop_layer.extent().xMinimum(),
            base_crop_layer.extent().yMinimum(),
            base_crop_layer.extent().xMinimum() + 0.001,
            base_crop_layer.extent().yMinimum() + 0.001,
        )
        pixel_bounds = get_extend_image_coords(base_crop_layer, extent)

        geo_corners = get_pixel_to_geo_corners(base_crop_layer, pixel_bounds)

        temp_path = create_georeferenced_temp_raster(
            data, geo_corners, base_crop_layer.crs().toWkt()
        )

        assert temp_path is not None
        temp_layer = QgsRasterLayer(temp_path, "temp", "gdal")
        assert temp_layer.isValid()
        assert temp_layer.bandCount() == 4, (
            "RGB should have 4 bands (3 for RGB + 1 for alpha)"
        )


class TestLensExtentCalculation:
    """Tests for compute_lens_extent function."""

    def test_lens_extent_with_valid_raster(self, qgis_iface, base_crop_layer):
        """Test lens extent calculation with a valid raster layer."""
        canvas = qgis_iface.mapCanvas()
        center = base_crop_layer.extent().center()
        extent = compute_lens_extent(
            center, base_crop_layer, 0.5, canvas, overlay_size=350
        )

        assert extent is not None
        assert extent.width() > 0
        assert extent.height() > 0
        assert extent.center() == center


class TestLensMapToolManagement:
    """Test LensMapTool initialization, activation, and state management."""

    def test_activation_creates_overlay_and_band(self, qgis_iface):
        """Test that activation creates overlay and extent band."""
        tool = LensMapTool(qgis_iface)

        tool.activate()

        assert tool._overlay is not None
        assert tool._extent_band is not None

    def test_deactivation_cleans_up(self, qgis_iface):
        """Test that deactivation removes overlay and resets state."""
        tool = LensMapTool(qgis_iface)
        tool.activate()

        tool.deactivate()

        assert tool._overlay is None
        assert tool._extent_band is None
        assert tool._pinned is False

    def test_render_mode_switching(self, qgis_iface):
        """Test switching between valid render modes."""
        tool = LensMapTool(qgis_iface)

        tool.set_render_mode("color")
        assert tool.render_mode() == "color"

        tool.set_render_mode("spectrum")
        assert tool.render_mode() == "spectrum"

        tool.set_render_mode("2d_spectrum")
        assert tool.render_mode() == "2d_spectrum"

        tool.set_render_mode("normal")
        assert tool.render_mode() == "normal"

        tool.set_render_mode("range_spectrum")
        assert tool.render_mode() == "range_spectrum"

        tool.set_render_mode("azimuth_viewer")
        assert tool.render_mode() == "azimuth_viewer"

        tool.set_render_mode("range_viewer")
        assert tool.render_mode() == "range_viewer"

    def test_viewer_mode_resets_center_frac(self, qgis_iface):
        """Test switching to a viewer mode resets viewer _center_frac to 0.5."""
        tool = LensMapTool(qgis_iface)
        azimuth_mode = tool._modes["azimuth_viewer"]
        range_mode = tool._modes["range_viewer"]
        azimuth_mode._center_frac = 0.8

        tool.set_render_mode("azimuth_viewer")
        assert azimuth_mode._center_frac == 0.5

        range_mode._center_frac = 0.2
        tool.set_render_mode("range_viewer")
        assert range_mode._center_frac == 0.5

    def test_invalid_render_mode_defaults_to_normal(self, qgis_iface):
        """Test invalid render mode defaults to normal."""
        tool = LensMapTool(qgis_iface)

        tool.set_render_mode("invalid")
        assert tool.render_mode() == "normal"

        tool.set_render_mode(None)
        assert tool.render_mode() == "normal"


def _render_kwargs(lens_tool: LensMapTool):
    return {
        "overlay_size": lens_tool.overlay_size,
        "metadata_provider": lens_tool.metadata_provider,
        "canvas": lens_tool.canvas,
    }


class TestRenderFunctions:
    """Tests for rendering functions that produce QImage output."""

    def test_render_color_returns_image(self, qgis_iface, test_render_layer):
        """Test color mode produces a valid QImage."""
        metadata_provider = MetadataProvider()
        lens_tool = LensMapTool(qgis_iface, metadata_provider=metadata_provider)

        center = test_render_layer.extent().center()
        extent = compute_lens_extent(
            center,
            test_render_layer,
            lens_tool._wheel.value,
            lens_tool.canvas,
            lens_tool.overlay_size,
        )
        result = lens_tool._modes["color"].render(
            test_render_layer, extent, **_render_kwargs(lens_tool)
        )

        assert result is not None
        image, is_georef = result
        assert isinstance(image, QImage)
        assert is_georef is True
        assert not image.isNull()
        assert image.width() > 0
        assert image.height() > 0

    def test_render_focus_returns_image(self, qgis_iface, test_render_layer):
        """Test focus mode produces a valid QImage."""
        metadata_provider = MetadataProvider()
        lens_tool = LensMapTool(qgis_iface, metadata_provider=metadata_provider)

        center = test_render_layer.extent().center()
        extent = compute_lens_extent(
            center,
            test_render_layer,
            lens_tool._wheel.value,
            lens_tool.canvas,
            lens_tool.overlay_size,
        )
        result = lens_tool._modes["spectrum"].render(
            test_render_layer, extent, **_render_kwargs(lens_tool)
        )

        assert result is not None
        image, is_georef = result
        assert isinstance(image, QImage)
        assert is_georef is True
        assert not image.isNull()
        assert image.width() > 0
        assert image.height() > 0

    def test_render_2d_spectrum_returns_image(self, qgis_iface, test_render_layer):
        """Test 2D spectrum mode produces a valid QImage."""
        metadata_provider = MetadataProvider()
        lens_tool = LensMapTool(qgis_iface, metadata_provider=metadata_provider)

        center = test_render_layer.extent().center()
        extent = compute_lens_extent(
            center,
            test_render_layer,
            lens_tool._wheel.value,
            lens_tool.canvas,
            lens_tool.overlay_size,
        )
        result = lens_tool._modes["2d_spectrum"].render(
            test_render_layer, extent, **_render_kwargs(lens_tool)
        )

        assert result is not None
        image, is_georef = result
        assert isinstance(image, QImage)
        assert is_georef is False
        assert not image.isNull()
        assert image.width() == lens_tool.overlay_size
        assert image.height() == lens_tool.overlay_size

    def test_render_range_spectrum_returns_image(self, qgis_iface, test_render_layer):
        """Test range spectrum mode produces a valid QImage."""
        metadata_provider = MetadataProvider()
        lens_tool = LensMapTool(qgis_iface, metadata_provider=metadata_provider)

        center = test_render_layer.extent().center()
        extent = compute_lens_extent(
            center,
            test_render_layer,
            lens_tool._wheel.value,
            lens_tool.canvas,
            lens_tool.overlay_size,
        )
        result = lens_tool._modes["range_spectrum"].render(
            test_render_layer, extent, **_render_kwargs(lens_tool)
        )

        assert result is not None
        image, is_georef = result
        assert isinstance(image, QImage)
        assert is_georef is False
        assert not image.isNull()
        assert image.width() == lens_tool.overlay_size
        assert image.height() == lens_tool.overlay_size

    def test_render_azimuth_viewer_returns_image(self, qgis_iface, test_render_layer):
        """Test azimuth viewer mode produces a valid QImage."""
        metadata_provider = MetadataProvider()
        lens_tool = LensMapTool(qgis_iface, metadata_provider=metadata_provider)

        center = test_render_layer.extent().center()
        extent = compute_lens_extent(
            center,
            test_render_layer,
            lens_tool._wheel.value,
            lens_tool.canvas,
            lens_tool.overlay_size,
        )
        result = lens_tool._modes["azimuth_viewer"].render(
            test_render_layer, extent, **_render_kwargs(lens_tool)
        )

        assert result is not None
        image, is_georef = result
        assert isinstance(image, QImage)
        assert is_georef is True
        assert not image.isNull()
        assert image.width() > 0
        assert image.height() > 0

    def test_render_range_viewer_returns_image(self, qgis_iface, test_render_layer):
        """Test range viewer mode produces a valid QImage."""
        metadata_provider = MetadataProvider()
        lens_tool = LensMapTool(qgis_iface, metadata_provider=metadata_provider)

        center = test_render_layer.extent().center()
        extent = compute_lens_extent(
            center,
            test_render_layer,
            lens_tool._wheel.value,
            lens_tool.canvas,
            lens_tool.overlay_size,
        )
        result = lens_tool._modes["range_viewer"].render(
            test_render_layer, extent, **_render_kwargs(lens_tool)
        )

        assert result is not None
        image, is_georef = result
        assert isinstance(image, QImage)
        assert is_georef is True
        assert not image.isNull()
        assert image.width() > 0
        assert image.height() > 0

    def test_update_overlay_dispatches_selected_mode(
        self, qgis_iface, test_render_layer, monkeypatch
    ):
        """Test selected render mode dispatches to matching mode class."""
        lens_tool = LensMapTool(qgis_iface)
        lens_tool.activate()
        # activate() syncs _layer from iface.activeLayer(); set explicitly for this test.
        lens_tool._layer = test_render_layer
        lens_tool._last_pos = qgis_iface.mapCanvas().rect().center()
        lens_tool._last_map_point = qgis_iface.mapCanvas().extent().center()

        called = {"count": 0}

        def _fake_render(_layer, _extent, **_kw):
            called["count"] += 1
            return (QImage(10, 10, QImage.Format_ARGB32), False)

        monkeypatch.setattr(lens_tool._modes["color"], "render", _fake_render)
        lens_tool.set_render_mode("color")
        lens_tool._update_overlay()

        assert called["count"] == 1
