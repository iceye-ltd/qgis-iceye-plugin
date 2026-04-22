"""Video generation from SLC: frame extraction and multiband raster creation."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from osgeo import gdal
from qgis import processing
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsTask,
)
from qgis.gui import QgsMapToolExtent

from ..gui.video_dialog import VideoDialog
from .batch_runner import BatchExtentRunner, BatchStepResult, resolve_batch_source_layer
from .cropper import (
    CropLayerTask,
    MaskLayerFactory,
)
from .metadata import IceyeMetadata, parse_iso8601_datetime
from .raster import read_slc_layer
from .temporal_properties import apply_temporal_properties_for_frames

# ============================================================================
# Video Processing Task Class
# ============================================================================


class VideoProcessingTask(QgsTask):
    """Parent task for video processing."""

    def __init__(
        self,
        iface,
        tif_layer,
        num_frames,
        metadata,
        temp_files,
    ):
        super().__init__("Processing Video", QgsTask.CanCancel)

        # Store parameters
        self.iface = iface
        self.tif_layer = tif_layer
        self.num_frames = num_frames
        self.metadata = metadata
        self.temp_files = temp_files

        # Will be set when subtask is added
        self.crop_subtask = None

        # Results
        self.layer_name = None
        self.result_layer = None
        self.frames = None
        self.temp_path = None
        self.exception = None
        self.cropped_layer_path = None
        # For temporal properties (set in run, used in finished)
        self.crop_extent = None
        self.full_width = 0
        self.full_height = 0

    def run(self):
        """Run video processing."""

        def phase_progress_setter(start, end):
            return lambda x: self.setProgress(
                self.progress() + (x / 100) * (end - start)
            )

        try:
            if self.isCanceled():
                return False

            if not self.crop_subtask:
                self.exception = Exception("No crop subtask set")
                return False

            # Get the cropped layer from the completed subtask
            cropped_layer = self.crop_subtask.result_layer

            if not cropped_layer or not cropped_layer.isValid():
                self.exception = Exception("Cropped layer is not valid")
                return False

            QgsMessageLog.logMessage(
                "Cropping complete, starting frame generation...",
                "ICEYE Toolbox",
                Qgis.Info,
            )

            # Store for temporal properties in finished()
            self.crop_extent = self.crop_subtask.extend_image_coords
            self.full_width = self.tif_layer.width()
            self.full_height = self.tif_layer.height()

            total_time = calculate_approximate_imaging_time(
                cropped_layer, self.metadata
            )

            frame_time = cropped_layer.width() / self.tif_layer.width() * total_time
            QgsMessageLog.logMessage(
                f"Cropped layers approximate imaging time: {frame_time:.3f} seconds",
                "ICEYE Toolbox",
                Qgis.Info,
            )

            # Read cropped layer and reconstruct complex data
            try:
                sub_patch, self.cropped_layer_path = read_slc_layer(
                    cropped_layer,
                    self.metadata.sar_observation_direction.lower() == "left",
                    self.metadata,
                )
            except Exception as e:
                self.exception = Exception(f"Failed to read cropped layer: {e!s}")
                return False

            if self.isCanceled():
                return False
            QgsMessageLog.logMessage(
                f"current progress {self.progress()}",
                "ICEYE Toolbox",
                Qgis.Info,
            )
            self.setProgress(10)
            QgsMessageLog.logMessage(
                f"current progress {self.progress()}",
                "ICEYE Toolbox",
                Qgis.Info,
            )
            self.layer_name = cropped_layer.name().replace("_CROP_", "_SHORT_")

            output_dir = (
                Path(self.temp_files)
                if self.temp_files
                else Path(self.cropped_layer_path).parent
            )
            self.temp_path = output_dir / f"{self.layer_name}.tif"

            # Create empty multiband TIFF
            output_raster = create_multiband_raster(
                str(self.temp_path),
                self.num_frames,
                sub_patch.shape,
                self.cropped_layer_path,
            )

            if output_raster is None:
                self.exception = Exception("Failed to create output TIFF file")
                return False

            QgsMessageLog.logMessage(
                f"current progress {self.progress()}",
                "ICEYE Toolbox",
                Qgis.Info,
            )
            # Generate frames and write them directly to TIFF
            try:
                get_frames_slow_time(
                    sub_patch,
                    self.num_frames,
                    metadata=self.metadata,
                    output_raster=output_raster,
                    is_canceled=self.isCanceled,
                    set_progress=phase_progress_setter(self.progress(), 95),
                )
            finally:
                try:
                    src_ds = gdal.Open(self.cropped_layer_path)
                    if src_ds:
                        properties = src_ds.GetMetadata()["ICEYE_PROPERTIES"]
                        output_raster.SetMetadataItem("ICEYE_PROPERTIES", properties)
                        QgsMessageLog.logMessage(
                            f"Metadata copied to video layer {self.temp_path}",
                            "ICEYE Toolbox",
                            Qgis.Info,
                        )
                        src_ds = None
                except (KeyError, TypeError) as e:
                    QgsMessageLog.logMessage(
                        f"Failed to copy metadata to video layer: {e!s}",
                        "ICEYE Toolbox",
                        Qgis.Warning,
                    )

                # Always close the raster
                output_raster.FlushCache()
                output_raster = None

            # Clear data from memory
            sub_patch = None

            if self.isCanceled():
                return False

            self.setProgress(95)

            try:
                overview_params = {
                    "INPUT": str(self.temp_path),
                    "LEVELS": "2 4 8 16",
                    "RESAMPLING": 1,
                    "FORMAT": 0,
                    "EXTRA": "",
                }
                processing.run("gdal:overviews", overview_params)
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Failed to build video layer overviews: {e!s}",
                    "ICEYE Toolbox",
                    Qgis.Warning,
                )
                return False

            self.setProgress(100)
            return True

        except Exception as e:
            self.exception = e
            QgsMessageLog.logMessage(
                f"Video processing error: {e!s}", "ICEYE Toolbox", Qgis.Critical
            )
            return False

    def finished(self, result):
        """Handle completion in main thread after run() finishes."""
        if not result:
            if self.exception:
                QgsMessageLog.logMessage(
                    f"Video processing failed: {self.exception!s}",
                    "ICEYE Toolbox",
                    Qgis.Critical,
                )
            elif self.isCanceled():
                QgsMessageLog.logMessage(
                    "Video processing was cancelled", "ICEYE Toolbox", Qgis.Warning
                )
        else:
            try:
                # Remove cropped layer from project
                QgsProject.instance().removeMapLayer(self.crop_subtask.result_layer)
                self.crop_subtask.result_layer = None
                self.crop_subtask.result_layer_path = None

                # Add video layer to project
                video_layer = QgsRasterLayer(str(self.temp_path), self.layer_name)
                QgsProject.instance().addMapLayer(video_layer, True)

                self.result_layer = video_layer

                # Apply per-band temporal properties for short
                apply_temporal_properties_for_frames(
                    video_layer,
                    self.metadata,
                )

                QgsMessageLog.logMessage(
                    f"Video created successfully with {self.num_frames} frames!",
                    "ICEYE Toolbox",
                    Qgis.Info,
                )
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Failed to load video layer: {e!s}", "ICEYE Toolbox", Qgis.Critical
                )


################################################################################
# Video Tool Class
################################################################################


class VideoTool:
    """Video tool class for creating video from a selected area."""

    def __init__(self, iface, metadata_provider, temp_files):
        """Initialize the video tool."""
        self.iface = iface
        self.metadata_provider = metadata_provider
        self.temp_files = temp_files
        self.canvas = iface.mapCanvas()
        self.extent_tool = QgsMapToolExtent(self.canvas)
        self.mask_factory = MaskLayerFactory(self.canvas)

        self.extent_tool.extentChanged.connect(
            lambda extent: self._create_video(extent)
        )

        self.video_task = None
        self._batch_num_frames = 0
        self._batch_runner = BatchExtentRunner(
            iface,
            label="Batch video",
            get_task=lambda: self.video_task,
            set_task=lambda t: setattr(self, "video_task", t),
            is_sibling_task_running=self._video_task_running,
            batch_already_running_msg="Batch video is already running.",
            sibling_task_running_msg="A video task is already running.",
        )

    def _video_task_running(self) -> bool:
        if self.video_task is None:
            return False
        try:
            return self.video_task.status() not in (
                QgsTask.Complete,
                QgsTask.Terminated,
            )
        except RuntimeError:
            self.video_task = None
            return False

    def process_extents_batch(self, jobs: list[tuple[QgsRectangle, str]]) -> None:
        """Process each (extent, source_layer_id) as video; one dialog sets frame count for all."""
        if not jobs:
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                "No extents to process.",
                level=Qgis.Warning,
                duration=3,
            )
            return
        if self._batch_runner.active:
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                "Batch video is already running.",
                level=Qgis.Warning,
                duration=4,
            )
            return
        if self._video_task_running():
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                "A video task is already running.",
                level=Qgis.Warning,
                duration=4,
            )
            return

        first_layer = resolve_batch_source_layer(jobs[0][1])
        if first_layer is None:
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                "Could not resolve source layer for batch video.",
                level=Qgis.Critical,
                duration=5,
            )
            return

        metadata = self.metadata_provider.get(first_layer)
        if not metadata:
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                "No metadata found for batch video.",
                level=Qgis.Critical,
                duration=5,
            )
            return

        dialog = VideoDialog(self.iface.mainWindow())
        if not dialog.exec_():
            return

        self._batch_num_frames = dialog.get_num_frames()
        self._batch_runner.prepare_without_try_begin(jobs)
        n = len(jobs)
        QgsMessageLog.logMessage(
            f"Batch video: starting {n} extent(s)",
            "ICEYE Toolbox",
            Qgis.Info,
        )
        self.iface.messageBar().pushMessage(
            "ICEYE Toolbox",
            f"Batch video: {n} area(s), {self._batch_num_frames} frame(s) each…",
            level=Qgis.Info,
            duration=5,
        )
        self._batch_runner.run_next_after_prepare(
            self._batch_video_step,
            on_after_each_step=self._after_batch_video_step,
            on_finalize=self._finalize_video_batch,
        )

    def _batch_video_step(
        self,
        extent: QgsRectangle,
        layer_id: str,
        source_layer: QgsRasterLayer,
        index: int,
        total: int,
    ) -> BatchStepResult:
        metadata = self.metadata_provider.get(source_layer)
        if not metadata:
            QgsMessageLog.logMessage(
                f"Batch video: no metadata for step {index}/{total}",
                "ICEYE Toolbox",
                Qgis.Warning,
            )
            return BatchStepResult(skip=True)

        crop_task = CropLayerTask(source_layer, extent)
        self.video_task = VideoProcessingTask(
            self.iface,
            source_layer,
            self._batch_num_frames,
            metadata,
            self.temp_files,
        )
        self.video_task.crop_subtask = crop_task
        self.video_task.addSubTask(
            crop_task,
            [],
            QgsTask.ParentDependsOnSubTask,
        )
        return BatchStepResult(task=self.video_task)

    def _after_batch_video_step(self) -> None:
        QgsMessageLog.logMessage(
            f"Batch video: completed step {self._batch_runner.step_index}/{self._batch_runner.total}",
            "ICEYE Toolbox",
            Qgis.Info,
        )

    def _finalize_video_batch(self) -> None:
        self._batch_num_frames = 0

    def run(self):
        """Activate extent tool for selecting area for video creation."""
        self.canvas.setMapTool(self.extent_tool)
        self.iface.messageBar().pushMessage(
            "Click and drag to select area for video creation", level=Qgis.Info
        )

    def _create_video(self, extent):
        """Handle extent selection - create mask, show dialog, start task."""
        self.canvas.unsetMapTool(self.extent_tool)
        self.process_extent(extent)

    def process_extent(self, extent):
        """Run video logic for the given extent (no map tool handling)."""
        tif_layer = self.iface.activeLayer()

        if not tif_layer:
            self.iface.messageBar().pushMessage(
                "No TIF layer found", level=Qgis.Critical
            )
            return

        # Get metadata
        metadata = self.metadata_provider.get(tif_layer)
        if not metadata:
            self.iface.messageBar().pushMessage(
                "No metadata found", level=Qgis.Critical
            )
            return

        # Show dialog for settings
        dialog = VideoDialog(self.iface.mainWindow())

        if not dialog.exec_():
            return

        num_frames = dialog.get_num_frames()

        # Create the crop subtask
        crop_task = CropLayerTask(tif_layer, extent)

        # Create the main video processing task
        self.video_task = VideoProcessingTask(
            self.iface,
            tif_layer,
            num_frames,
            metadata,
            self.temp_files,
        )

        # Store reference to crop task
        self.video_task.crop_subtask = crop_task

        self.video_task.addSubTask(
            crop_task,
            [],
            QgsTask.ParentDependsOnSubTask,
        )

        # Now add the parent task to task manager
        QgsApplication.taskManager().addTask(self.video_task)

        self.iface.messageBar().pushMessage(
            f"Starting video processing with {num_frames} frames in background...",
            level=Qgis.Info,
        )


# ============================================================================
# Video Frame Generation Functions
# ============================================================================


def calculate_valid_samples(nx: int, oversampling: float) -> tuple[int, int, int]:
    """Calculate start and end samples to avoid garbage in edge bands.

    Parameters
    ----------
    nx : int
        Total number of samples.
    oversampling : float
        Oversampling factor.

    Returns
    -------
    tuple of int
        (start_sample, stop_sample, num_valid_samples).
    """
    # Calculate number of valid samples
    num_samples = int(nx / oversampling)

    # Clamp to valid range
    num_samples = max(1, min(num_samples, nx))

    # Calculate extra samples at edges
    extra_samples = nx - num_samples

    # Trim symmetrically from both ends
    start_sample = extra_samples // 2
    stop_sample = start_sample + num_samples

    return start_sample, stop_sample, num_samples


def get_frames_slow_time(
    complex_data: NDArray[np.complex64],
    num_frames: int,
    metadata: IceyeMetadata,
    output_raster: gdal.Dataset,
    is_canceled: Callable[[], bool],
    set_progress: Callable[[int], None],
) -> None:
    """Generate frames from complex data using FFT along azimuth direction."""
    rows, cols = complex_data.shape

    oversampling_azi = (
        metadata.sar_resolution_azimuth / metadata.sar_pixel_spacing_azimuth
    )
    start_sample, stop_sample, num_valid_samples = calculate_valid_samples(
        rows, oversampling_azi
    )

    fft_data = np.fft.fft(complex_data, axis=0)
    fft_data = np.fft.fftshift(fft_data, axes=0)

    # 30% of this function
    set_progress(30.0)
    complex_data = None

    segment_size = num_valid_samples * metadata.sar_resolution_azimuth / 0.5
    # segment_size = num_valid_samples // num_frames

    centers = np.round(
        np.linspace(
            start_sample + segment_size // 2,
            stop_sample - segment_size // 2,
            num_frames,
            dtype=int,
        )
    )

    frame_resolution = metadata.sar_resolution_azimuth * num_frames
    QgsMessageLog.logMessage(
        f"Frame resolution: {frame_resolution:.3f} meters, of size {segment_size}/{num_valid_samples}",
        "ICEYE Toolbox",
        Qgis.Info,
    )

    QgsMessageLog.logMessage("FFTing...", "ICEYE Toolbox", Qgis.Info)
    # rest 60 %
    for i, center in enumerate(centers):
        if is_canceled():
            QgsMessageLog.logMessage(
                "Frame generation cancelled", "ICEYE Toolbox", Qgis.Warning
            )
            return

        if set_progress:
            set_progress(30 + (i / len(centers) * 100))

        frame_spectrum = np.zeros_like(fft_data)

        start_idx = int(center - segment_size // 2)
        end_idx = int(start_idx + segment_size)
        QgsMessageLog.logMessage(
            f"from {start_idx} to {end_idx}", "ICEYE Toolbox", Qgis.Info
        )
        frame_spectrum[start_idx:end_idx, :] = fft_data[start_idx:end_idx, :]
        frame_spectrum = np.fft.ifftshift(frame_spectrum, axes=0)
        frame = np.fft.ifft(frame_spectrum, axis=0)

        if not write_frame_to_band(
            output_raster,
            np.abs(frame),
            i + 1,
            metadata.sar_observation_direction.lower() == "left",
        ):
            raise Exception(f"Failed to write frame {i} to band {i + 1}")


def calculate_approximate_imaging_time(
    cropped_layer: QgsRasterLayer, metadata: IceyeMetadata
) -> float:
    """Calculate the approximate imaging time of the cropped layer in seconds."""
    start_dt = parse_iso8601_datetime(metadata.start_datetime)
    end_dt = parse_iso8601_datetime(metadata.end_datetime)
    if start_dt is None or end_dt is None:
        raise ValueError(
            "Invalid metadata acquisition times: "
            f"start_datetime={metadata.start_datetime!r}, "
            f"end_datetime={metadata.end_datetime!r}"
        )
    total_time = (end_dt - start_dt).total_seconds()

    return total_time


def create_multiband_raster(
    path: str, num_frames: int, frame_shape: tuple[int, int], source_path: str
) -> gdal.Dataset | None:
    """Create a multiband raster file ready for frame writing.

    Parameters
    ----------
    path : str
        Output file path.
    num_frames : int
        Number of bands (frames).
    frame_shape : tuple of int
        (height, width) of each frame.
    source_path : str
        Source SLC path for georeferencing.

    Returns
    -------
    gdal.Dataset or None
        Open dataset or None on failure.
    """
    driver = gdal.GetDriverByName("GTiff")

    output_raster = driver.Create(
        path,
        frame_shape[0],
        frame_shape[1],
        num_frames,
        gdal.GDT_Float32,
        options=["BIGTIFF=YES", "COMPRESS=NONE", "PHOTOMETRIC=MINISBLACK", "ALPHA=NO"],
    )

    if output_raster is None:
        QgsMessageLog.logMessage(
            f"Failed to create raster file: {path}", "iceye_toolbox", Qgis.Critical
        )
        return None

    if source_path:
        src_ds = gdal.Open(source_path)
        if src_ds:
            proj = src_ds.GetGeoTransform()
            output_raster.SetGeoTransform(proj)

            gcps = src_ds.GetGCPs()
            gcp_projection = src_ds.GetGCPProjection()
            output_raster.SetGCPs(gcps, gcp_projection)
        else:
            QgsMessageLog.logMessage(
                f"Failed to open source path: {source_path}",
                "iceye_toolbox",
                Qgis.Critical,
            )
            return None

    return output_raster


def write_frame_to_band(
    output_raster: gdal.Dataset,
    frame_data: NDArray[np.float32],
    band_index: int,
    left: bool,
) -> bool:
    """Write a single frame to a specific band in the output raster."""
    try:
        band = output_raster.GetRasterBand(band_index)
        band.SetNoDataValue(0)
        if left:
            band.WriteArray(np.fliplr(frame_data.T))
        else:
            band.WriteArray(frame_data.T)
        return True
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to write frame to band {band_index}: {e!s}",
            "iceye_toolbox",
            Qgis.Critical,
        )
        return False
