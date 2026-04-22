# coding=utf-8
"""Tests for color module."""

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal
from qgis.core import QgsRasterLayer, QgsRectangle

from iceye_toolbox.core.color import (
    ColorTask,
    color_image,
    color_image_slow_time,
    create_color_raster_layer,
)
from iceye_toolbox.core.cropper import CropLayerTask
from iceye_toolbox.core.metadata import MetadataProvider
from iceye_toolbox.core.raster import read_slc_layer

# Fixtures


@pytest.fixture
def color_slow_file():
    """Path to the slow-time color reference raster in test/fixtures."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    return str(
        fixtures_dir
        / "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_COLOR_37a3f6c7.tif"
    )


@pytest.fixture
def color_slow_layer(color_slow_file):
    """QgsRasterLayer loaded from the bundled COLOR (slow-time) fixture."""
    layer = QgsRasterLayer(
        color_slow_file,
        "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_COLOR_37a3f6c7",
    )
    assert layer.isValid(), f"Failed to load test raster: {color_slow_file}"
    return layer


@pytest.fixture
def color_fast_file():
    """Path to the fast-time color reference raster in test/fixtures."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    return str(
        fixtures_dir
        / "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_COLOR_FAST_37a3f6c7.tif"
    )


@pytest.fixture
def color_fast_layer(color_fast_file):
    """QgsRasterLayer loaded from the bundled COLOR (fast-time) fixture."""
    layer = QgsRasterLayer(
        color_fast_file,
        "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_COLOR_FAST_37a3f6c7",
    )
    assert layer.isValid(), f"Failed to load test raster: {color_slow_file}"
    return layer


class TestColorWorkflow:
    """End-to-end workflow test - main validation."""

    def test_color_slowtime(self, qgis_iface, base_crop_layer, color_slow_layer):
        """Slow-time color workflow should match expected output."""
        metadata = MetadataProvider()
        # We dont need to run CropTask
        crop_task = CropLayerTask(base_crop_layer, QgsRectangle())
        crop_task.result_layer = base_crop_layer
        crop_layer = crop_task.result_layer

        crop_width, crop_height = crop_layer.width(), crop_layer.height()

        color_task = ColorTask(qgis_iface, metadata, crop_task, color_mode="slow_time")

        color_task.crop_subtask = crop_task

        color_success = color_task.run()

        if not color_success:
            pytest.fail(f"ColorTask failed: {color_task.exception}")

        color_task.finished(True)

        assert color_task.result_layer.isValid()
        assert color_task.result_layer.width() == crop_width
        assert color_task.result_layer.height() == crop_height

        expected_ds = gdal.Open(color_slow_layer.source())
        actual_ds = gdal.Open(color_task.result_layer.source())

        assert expected_ds.RasterCount == actual_ds.RasterCount

        for band_idx in range(1, expected_ds.RasterCount + 1):
            expected_band = expected_ds.GetRasterBand(band_idx)
            actual_band = actual_ds.GetRasterBand(band_idx)
            expected_data = expected_band.ReadAsArray()
            actual_data = actual_band.ReadAsArray()
            np.testing.assert_allclose(actual_data, expected_data, rtol=1e-3, atol=1e-5)

        # Cleanup after loop
        result_path = Path(color_task.result_layer.source())
        color_task.result_layer = None
        result_path.unlink(missing_ok=True)

    def test_color_fasttime(self, qgis_iface, base_crop_layer, color_fast_layer):
        """Fast-time color workflow vs reference."""
        metadata = MetadataProvider()
        # We dont need to run CropTask
        crop_task = CropLayerTask(base_crop_layer, QgsRectangle())
        crop_task.result_layer = base_crop_layer
        crop_layer = crop_task.result_layer

        crop_width, crop_height = crop_layer.width(), crop_layer.height()

        color_task = ColorTask(qgis_iface, metadata, crop_task, color_mode="fast_time")

        color_task.crop_subtask = crop_task

        color_success = color_task.run()

        if not color_success:
            pytest.fail(f"ColorTask failed: {color_task.exception}")

        color_task.finished(True)

        assert color_task.result_layer.isValid()
        assert color_task.result_layer.width() == crop_width
        assert color_task.result_layer.height() == crop_height

        expected_ds = gdal.Open(color_fast_layer.source())
        actual_ds = gdal.Open(color_task.result_layer.source())

        assert expected_ds.RasterCount == actual_ds.RasterCount

        for band_idx in range(1, expected_ds.RasterCount + 1):
            expected_band = expected_ds.GetRasterBand(band_idx)
            actual_band = actual_ds.GetRasterBand(band_idx)
            expected_data = expected_band.ReadAsArray()
            actual_data = actual_band.ReadAsArray()
            np.testing.assert_allclose(actual_data, expected_data, rtol=1e-3, atol=1e-5)

        # Cleanup after loop
        result_path = Path(color_task.result_layer.source())
        color_task.result_layer = None
        result_path.unlink(missing_ok=True)


class TestColorFunctions:
    """Tests for Color functions and raster creation."""

    def test_color_image(self, base_crop_layer, color_fast_layer):
        """color_image vs fast-time reference."""
        metadata = MetadataProvider().get(base_crop_layer)

        data, _ = read_slc_layer(
            base_crop_layer,
            metadata.sar_observation_direction.lower() == "left",
            metadata,
        )
        rgb = color_image(data, metadata)

        # We need to transpose and flip rgb
        rgb = np.transpose(rgb, (1, 0, 2))
        if metadata.sar_observation_direction.lower() == "left":
            rgb = np.flip(rgb, axis=1)

        assert rgb.shape == (data.shape[1], data.shape[0], 3)

        actual_ds = gdal.Open(color_fast_layer.source())

        assert rgb.dtype == np.float32

        for band_idx in range(1, actual_ds.RasterCount + 1):
            actual_band = actual_ds.GetRasterBand(band_idx)
            actual_data = actual_band.ReadAsArray()

            np.testing.assert_allclose(
                actual_data, rgb[:, :, band_idx - 1], rtol=1e-3, atol=1e-5
            )

    def test_color_image_slow_time(self, base_crop_layer, color_slow_layer):
        """color_image_slow_time should produce RGB matching slow-time reference."""
        metadata = MetadataProvider().get(base_crop_layer)

        data, _ = read_slc_layer(
            base_crop_layer,
            metadata.sar_observation_direction.lower() == "left",
            metadata,
        )
        rgb = color_image_slow_time(data, metadata)

        # We need to transpose and flip rgb
        rgb = np.transpose(rgb, (1, 0, 2))
        if metadata.sar_observation_direction.lower() == "left":
            rgb = np.flip(rgb, axis=1)

        assert rgb.shape == (data.shape[1], data.shape[0], 3)

        actual_ds = gdal.Open(color_slow_layer.source())

        assert rgb.dtype == np.float32

        for band_idx in range(1, actual_ds.RasterCount + 1):
            actual_band = actual_ds.GetRasterBand(band_idx)
            actual_data = actual_band.ReadAsArray()

            np.testing.assert_allclose(
                actual_data, rgb[:, :, band_idx - 1], rtol=1e-3, atol=1e-5
            )

    def test_create_color_raster_layer(self, base_crop_layer):
        """create_color_raster_layer should produce valid 3-band raster."""
        metadata = MetadataProvider().get(base_crop_layer)

        data, _ = read_slc_layer(
            base_crop_layer,
            metadata.sar_observation_direction.lower() == "left",
            metadata,
        )
        rgb = color_image(data, metadata)

        success, test_color_layer, error = create_color_raster_layer(
            rgb, base_crop_layer, metadata.sar_observation_direction.lower() == "left"
        )
        assert success
        assert test_color_layer.isValid()
        assert test_color_layer.bandCount() == 3

        result_path = Path(test_color_layer.source())
        test_color_layer = None
        result_path.unlink(missing_ok=True)
