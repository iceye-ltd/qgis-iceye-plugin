# coding=utf-8
"""Tests for video module."""

from pathlib import Path

import numpy as np
import pytest
from osgeo import gdal
from qgis.core import QgsRasterLayer

from ICEYE_toolbox.core.cropper import CropLayerTask
from ICEYE_toolbox.core.metadata import IceyeMetadata
from ICEYE_toolbox.core.video import (
    VideoProcessingTask,
    calculate_valid_samples,
    create_multiband_raster,
    get_frames_slow_time,
    write_frame_to_band,
)


# Fixtures
@pytest.fixture
def video_file():
    """Path to the short multiframe video reference raster in test/fixtures."""
    fixtures_dir = Path(__file__).resolve().parent / "fixtures"
    return str(
        fixtures_dir
        / "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_SHORT_37a3f6c7.tif"
    )


@pytest.fixture
def video_layer(video_file):
    """QgsRasterLayer loaded from the bundled SHORT video fixture."""
    layer = QgsRasterLayer(
        video_file, "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED_SHORT_37a3f6c7"
    )
    assert layer.isValid(), f"Failed to load test raster: {video_file}"
    return layer


################################################################################
# Test Video Processing Task
################################################################################


class TestVideoWorkflow:
    """End-to-end workflow test - main validation."""

    def test_full_video_workflow_slow_time(
        self, qgis_iface, base_crop_layer, metadata, video_layer, tmp_path
    ):
        """Complete video workflow with slow time: run video task and verify output."""
        num_frames = 4

        # Create crop subtask and run it
        crop_task = CropLayerTask(base_crop_layer, base_crop_layer.extent())
        crop_task.result_layer = base_crop_layer

        # Save width and height here because crop is deleted after video_task.finished()
        crop_width, crop_height = (
            crop_task.result_layer.width(),
            crop_task.result_layer.height(),
        )

        video_task = VideoProcessingTask(
            qgis_iface,
            crop_task.result_layer,
            num_frames,
            metadata,
            temp_files=tmp_path,
        )
        video_task.crop_subtask = crop_task

        video_success = video_task.run()

        if not video_success:
            pytest.fail(f"VideoProcessingTask failed: {video_task.exception}")

        video_task.finished(True)

        assert video_task.temp_path.exists()
        assert video_task.temp_path.suffix == ".tif"

        task_video_layer = video_task.result_layer
        assert task_video_layer.isValid()
        assert task_video_layer.width() == crop_width
        assert task_video_layer.height() == crop_height

        expected_ds = gdal.Open(video_layer.source())
        actual_ds = gdal.Open(task_video_layer.source())

        # Verify they have the same number of bands
        assert expected_ds.RasterCount == actual_ds.RasterCount
        num_frames = expected_ds.RasterCount

        # Compare each band
        for band_idx in range(1, num_frames + 1):
            expected_band = expected_ds.GetRasterBand(band_idx)
            actual_band = actual_ds.GetRasterBand(band_idx)

            expected_data = expected_band.ReadAsArray()
            actual_data = actual_band.ReadAsArray()

            # Verify the data and shape are the same
            assert np.array_equal(actual_data, expected_data), (
                f"Band {band_idx} data mismatch"
            )

        # Clean up GDAL datasets
        expected_ds = None
        actual_ds = None


################################################################################
# Test Calculate Valid Samples
################################################################################


class TestCalculateValidSamples:
    """Tests for calculate_valid_samples function."""

    def test_reasonable_output(self):
        """Test that calculate_valid_samples produces reasonable outputs."""
        nx = 1000
        oversampling = 1.2

        start, stop, num = calculate_valid_samples(nx, oversampling)

        assert num < nx
        assert num > 0
        assert start >= 0
        assert stop <= nx
        assert stop - start == num

    def test_edge_cases(self):
        """Test edge cases don't crash."""
        # Very high oversampling
        start, stop, num = calculate_valid_samples(100, 50.0)
        assert num >= 1  # Should clamp to at least 1

        # Very low oversampling
        start, stop, num = calculate_valid_samples(100, 0.5)
        assert num <= 100  # Should clamp to max nx


################################################################################
# Test Get Frames
################################################################################


@pytest.mark.parametrize("observation_direction", ["left", "right"])
def test_observation_directions(
    complex_data: np.ndarray,
    metadata: IceyeMetadata,
    tmp_path: str,
    observation_direction: str,
):
    """Test both observation directions work for both functions."""
    metadata.sar_observation_direction = observation_direction
    num_frames = 2

    # Test slow time
    output_path_slow = Path(tmp_path) / f"test_slow_{observation_direction}.tif"
    output_raster_slow = create_multiband_raster(
        str(output_path_slow), num_frames, complex_data.shape, None
    )

    get_frames_slow_time(
        complex_data,
        num_frames,
        metadata,
        output_raster_slow,
        lambda: False,
        lambda x: None,
    )

    output_raster_slow = None


################################################################################
# Test Write Frame To Band
################################################################################


class TestWriteFrameToBand:
    """Basic tests for write_frame_to_band."""

    def test_writes_successfully(self, tmp_path: str):
        """Test basic frame writing."""
        output_path = Path(tmp_path) / "test_write.tif"
        frame_shape = (20, 20)

        raster = create_multiband_raster(str(output_path), 1, frame_shape, None)
        test_frame = np.ones(frame_shape, dtype=np.float32) * 42.0

        # Observation direction doesnt matter as we only test that it writes successfully
        success = write_frame_to_band(raster, test_frame, 1, False)

        assert success is True

        # Verify something was written
        band = raster.GetRasterBand(1)
        data = band.ReadAsArray()
        assert data.mean() > 0

        raster = None
