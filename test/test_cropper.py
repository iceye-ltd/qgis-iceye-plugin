# coding=utf-8
"""Tests for cropper module."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal
from qgis.core import QgsProject, QgsRasterLayer, QgsRectangle

from iceye_toolbox.core.cropper import (
    CropLayerTask,
    CropTool,
    MaskLayerFactory,
    check_and_grow_pixel_extent,
    get_extend_image_coords,
)


# ============================================================================
# Utility Function Tests
# ============================================================================
@pytest.fixture
def sample_extent():
    """Geographic extent inside the WWGTZ2 test/fixtures crop (WGS 84)."""
    return QgsRectangle(
        117.7408,
        38.9958,
        117.7465,
        38.9985,
    )


class TestGetExtendImageCoords:
    """Tests for get_extend_image_coords function."""

    def test_returns_pixel_coordinates_for_gcp_layer(
        self, base_crop_layer, sample_extent
    ):
        """Should return QgsRectangle with valid pixel coordinates."""
        result = get_extend_image_coords(base_crop_layer, sample_extent)

        assert isinstance(result, QgsRectangle)
        assert result.width() > 0
        assert result.height() > 0
        # GCP-based pixel coordinates can be large negatives or positives (stripmap crops).
        for v in (
            result.xMinimum(),
            result.xMaximum(),
            result.yMinimum(),
            result.yMaximum(),
        ):
            assert np.isfinite(v)
            assert abs(v) < 1e6

    def test_returns_none_without_gcps(self, qgis_iface):
        """Should return None for layer without GCPs."""
        # Create a simple raster without GCPs
        layer = QgsRasterLayer("", "test", "gdal")  # Invalid/empty layer
        result = get_extend_image_coords(layer, QgsRectangle(0, 0, 10, 10))
        assert result is None


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestCropToolErrorHandling:
    """Tests for edge cases and error handling."""

    def test_crop_without_layer_handles_gracefully(self, qgis_iface):
        """_crop() should handle missing layer gracefully without crashing."""
        tool = CropTool(qgis_iface)
        tool.layer = None
        tool._crop(QgsRectangle(0, 0, 10, 10))  # Should not raise


# ============================================================================
# End-to-End Workflow Test
# ============================================================================


class TestCropperWorkflow:
    """End-to-end workflow test - main validation."""

    def test_full_crop_workflow(self, qgis_iface, base_crop_layer, sample_extent):
        """Complete crop workflow: create mask → crop → verify output.

        Do not add ``base_crop_layer`` to :class:`QgsProject` here: the project takes
        ownership and would break the pytest fixture lifecycle (and later tests).
        """
        # Create mask (adds a temporary vector layer to the project)
        mask = MaskLayerFactory(qgis_iface.mapCanvas()).create(sample_extent)
        assert mask.isValid()
        assert mask.featureCount() == 1

        # Run crop
        task = CropLayerTask(base_crop_layer, sample_extent)
        success = task.run()

        if not success:
            pytest.fail(f"CropLayerTask.run() failed: {task.error_msg}")

        task.finished(True)

        assert success, f"CropLayerTask failed: {task.error_msg}"
        assert task.result_layer is not None
        assert task.result_layer.isValid()
        assert task.result_layer.width() > 0
        assert task.result_layer.height() > 0
        ds = gdal.Open(task.result_layer.source())
        assert ds.GetMetadata("ICEYE_PROPERTIES") is not None
        assert len(ds.GetGCPs()) > 0

        project = QgsProject.instance()
        project.removeMapLayer(mask.id())
        project.removeMapLayer(task.result_layer.id())

    def test_crop_of_crop_consistency(self, qgis_iface, base_crop_layer, sample_extent):
        """Testing crop of crop consistency."""
        # Define two nested extents
        outer_extent = sample_extent
        # Taken from a crop of the sample_extent
        inner_extent = QgsRectangle(
            117.7415,
            38.9961,
            117.7442,
            38.9976,
        )

        # Crop 1: Large crop from original
        task1 = CropLayerTask(base_crop_layer, outer_extent)
        assert task1.run()
        task1.finished(True)
        first_crop = task1.result_layer

        # Crop 2: Small crop from first crop
        task2 = CropLayerTask(first_crop, inner_extent)
        assert task2.run()
        task2.finished(True)
        nested_crop = task2.result_layer

        # Crop 3: Small crop directly from original
        task3 = CropLayerTask(base_crop_layer, inner_extent)
        assert task3.run()
        task3.finished(True)
        direct_crop = task3.result_layer

        # Both methods should produce same dimensions (within tolerance)
        assert abs(nested_crop.width() - direct_crop.width()) <= 1
        assert abs(nested_crop.height() - direct_crop.height()) <= 1

        project = QgsProject.instance()
        for lyr in (first_crop, nested_crop, direct_crop):
            project.removeMapLayer(lyr.id())


# ============================================================================
# check_and_grow_pixel_extent Tests
# ============================================================================


class TestCheckAndGrowPixelExtent:
    """Tests for check_and_grow_pixel_extent function."""

    def test_returns_extent_and_growth_flag(self):
        """Should return extent and boolean indicating if growth occurred.

        Test values are in pixel coordinates.
        """
        # Test with small extent (should grow)
        small_extent = QgsRectangle(0, 0, 50, 50)
        result, was_grown = check_and_grow_pixel_extent(small_extent, min_pixels=100)

        print(f"result: {result}, was_grown: {was_grown}")

        assert isinstance(result, QgsRectangle)
        assert was_grown is True
        assert result.width() == 100
        assert result.height() == 100

    def test_no_growth_for_large_extent(self):
        """Should not grow extents that already meet minimum size."""
        large_extent = QgsRectangle(0, 0, 200, 150)
        result, was_grown = check_and_grow_pixel_extent(large_extent, min_pixels=100)

        assert was_grown is False
        assert result.width() == 200
        assert result.height() == 150
