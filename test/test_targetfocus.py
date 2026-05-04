# coding=utf-8
"""Tests for video module."""

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal
from qgis.core import QgsRasterLayer, QgsRectangle

from iceye_toolbox.core.autofocus import (
    AutofocusTask,
    entropy,
    focus_with_centered_looks_pga,
)
from iceye_toolbox.core.cropper import CropLayerTask
from iceye_toolbox.core.metadata import MetadataProvider

# Fixtures


@pytest.fixture
def focus_file():
    """Path to the multiband focus crop reference raster in test/fixtures."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    return str(
        fixtures_dir
        / "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_FOCUS_37a3f6c7.tif"
    )


@pytest.fixture
def focus_layer(focus_file):
    """QgsRasterLayer loaded from the bundled CROP fixture."""
    layer = QgsRasterLayer(
        focus_file, "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_FOCUS_37a3f6c7"
    )
    assert layer.isValid(), f"Failed to load test raster: {focus_file}"
    return layer


################################################################################
# Test autofocus workflow
################################################################################


class TestAutofocusWorkflow:
    """End-to-end workflow test. Tests whole workflow from reading the crop task to creating the focused raster layer."""

    def test_target_focus_singleband(self, qgis_iface, base_crop_layer, focus_layer):
        """Test autofocus workflow."""
        expected_layer = focus_layer

        crop_task = CropLayerTask(base_crop_layer, base_crop_layer.extent())
        crop_task.result_layer = base_crop_layer

        crop_layer = crop_task.result_layer
        crop_width, crop_height = crop_layer.width(), crop_layer.height()

        metadata_provider = MetadataProvider()

        target_focus_task = AutofocusTask(qgis_iface, metadata_provider, crop_task)

        target_success = target_focus_task.run()
        if not target_success:
            pytest.fail(f"AutofocusTask failed: {target_focus_task.exception}")

        target_focus_task.finished(True)

        result_layer = target_focus_task.result_layer

        assert result_layer.isValid()
        assert result_layer.width() == crop_width
        assert result_layer.height() == crop_height

        expected_ds = gdal.Open(expected_layer.source())
        actual_ds = gdal.Open(result_layer.source())
        assert expected_ds is not None
        assert actual_ds is not None
        assert expected_ds.RasterCount == actual_ds.RasterCount == 2

        for band_idx in (1, 2):
            expected = expected_ds.GetRasterBand(band_idx).ReadAsArray()
            actual = actual_ds.GetRasterBand(band_idx).ReadAsArray()
            assert np.all(np.isfinite(actual))
            rtol, atol = 1e-3, 2e-3
            close = np.isclose(actual, expected, rtol=rtol, atol=atol)
            if not np.all(close):
                n_bad = int(np.sum(~close))
                max_abs = float(np.max(np.abs(actual - expected)))
                assert n_bad <= 128 and max_abs <= 10.0, (
                    f"band {band_idx}: {n_bad}/{actual.size} pixels outside "
                    f"rtol={rtol}, atol={atol} (max |Δ|={max_abs})"
                )

        path = Path(result_layer.source())
        if path.exists():
            path.unlink()


################################################################################
# Test autofocus utility functions
################################################################################


class TestAutofocusFunctions:
    """Smoke tests for helpers still exposed from core.autofocus."""

    def test_calculate_entropy(self, complex_data):
        """Entropy of complex data should be computed."""
        entropy_value = entropy(complex_data)
        assert entropy_value is not None


class TestFocusWithCenteredLooksPga:
    """Tests for centered-look PGA autofocus helper."""

    def test_returns_same_shape_and_dtype(self):
        """Output matches input shape; stays complex (precision may widen in PGA path)."""
        rng = np.random.default_rng(42)
        z = (rng.standard_normal((96, 48)) + 1j * rng.standard_normal((96, 48))).astype(
            np.complex64
        )
        out = focus_with_centered_looks_pga(z)
        assert out.shape == z.shape
        assert np.issubdtype(out.dtype, np.complexfloating)
        assert np.all(np.isfinite(out.real))
        assert np.all(np.isfinite(out.imag))
