"""Color workflow: RGB spectrum visualization and color raster creation."""

from __future__ import annotations

from datetime import datetime
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

from .batch_runner import BatchExtentRunner, BatchStepResult
from .cropper import (
    CropLayerTask,
    MaskLayerFactory,
)
from .metadata import IceyeMetadata
from .raster import read_slc_layer
from .video import calculate_valid_samples, create_multiband_raster, write_frame_to_band


class ColorTool:
    """Handler class for color workflow."""

    def __init__(self, iface, metadata_provider):
        """Initialize the color handler."""
        self.iface = iface
        self.canvas = iface.mapCanvas()

        self.metadata_provider = metadata_provider
        self.mapToolExtent = QgsMapToolExtent(self.canvas)
        self.maskLayerFactory = MaskLayerFactory(self.canvas)

        self.mapToolExtent.extentChanged.connect(lambda extent: self._run(extent))

        self.task = None
        self.layer = None

        self._batch_color_mode: str = "range_cmap"
        self._batch_runner = BatchExtentRunner(
            iface,
            label="Batch color",
            get_task=lambda: self.task,
            set_task=lambda t: setattr(self, "task", t),
            is_sibling_task_running=self._color_task_running,
            batch_already_running_msg="Batch color is already running.",
            sibling_task_running_msg="A color task is already running.",
        )

    def activate(self):
        """Activate extent tool for selecting the target area."""
        self.layer = self.iface.activeLayer()
        self.canvas.setMapTool(self.mapToolExtent)

    def _run(self, extent):
        """Handle the selected extent."""
        self.canvas.unsetMapTool(self.mapToolExtent)
        self.process_extent(extent)

    def process_extent(self, extent, color_mode: str = "range_cmap"):
        """Run color logic for the given extent."""
        layer = self.iface.activeLayer()
        if not layer:
            self.iface.messageBar().pushMessage(
                "No TIF layer found", level=Qgis.Warning, duration=3
            )
            return

        crop_task = CropLayerTask(layer, extent)
        self.task = ColorTask(self.iface, self.metadata_provider, crop_task, color_mode)

        self.task.addSubTask(crop_task, [], QgsTask.ParentDependsOnSubTask)
        QgsApplication.taskManager().addTask(self.task)

    def _color_task_running(self) -> bool:
        """Return True if a single or batch color QgsTask is queued or running."""
        if self.task is None:
            return False
        try:
            return self.task.status() not in (QgsTask.Complete, QgsTask.Terminated)
        except RuntimeError:
            self.task = None
            return False

    def process_extents_batch(
        self,
        jobs: list[tuple[QgsRectangle, str]],
        color_mode: str = "range_cmap",
    ) -> None:
        """Process each (extent, source_layer_id) as color composite, one task at a time."""
        self._batch_color_mode = color_mode
        self._batch_runner.start(
            jobs,
            self._batch_color_step,
            on_after_each_step=self._after_batch_color_step,
            start_log=f"Batch color: starting {len(jobs)} extent(s)",
            start_message=f"Batch color: {len(jobs)} area(s)…",
        )

    def _batch_color_step(
        self,
        extent: QgsRectangle,
        layer_id: str,
        source_layer: QgsRasterLayer,
        index: int,
        total: int,
    ) -> BatchStepResult:
        crop_task = CropLayerTask(source_layer, extent)
        self.task = ColorTask(
            self.iface,
            self.metadata_provider,
            crop_task,
            self._batch_color_mode,
        )
        self.task.addSubTask(crop_task, [], QgsTask.ParentDependsOnSubTask)
        return BatchStepResult(task=self.task)

    def _after_batch_color_step(self) -> None:
        QgsMessageLog.logMessage(
            f"Batch color: completed step {self._batch_runner.step_index}/{self._batch_runner.total}",
            "ICEYE Toolbox",
            Qgis.Info,
        )


class ColorTask(QgsTask):
    """Task to create a color image of a given extent."""

    def __init__(
        self, iface, metadata_provider, crop_task, color_mode: str = "range_cmap"
    ):
        super().__init__("Creating color image", QgsTask.CanCancel)
        self.iface = iface
        self.crop_task = crop_task
        self.metadata_provider = metadata_provider
        self.color_mode = color_mode
        self.result_layer = None

    def run(self):
        """Run the color image creation."""
        QgsMessageLog.logMessage(
            f"Running for {self.crop_task.result_layer.name()}",
            "ICEYE Toolbox",
            Qgis.Info,
        )
        metadata = self.metadata_provider.get(self.crop_task.result_layer)
        data, _ = read_slc_layer(
            self.crop_task.result_layer,
            metadata.sar_observation_direction.lower() == "left",
            metadata,
        )

        func = COLOR_FUNCS.get(self.color_mode, color_image)

        try:
            rgb = func(data, metadata, progress_callback=self.setProgress)
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Error in color image creation: {e!s}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return False

        success, rgb_layer, error = create_color_raster_layer(
            rgb,
            self.crop_task.result_layer,
            metadata.sar_observation_direction.lower() == "left",
        )
        if not success:
            QgsMessageLog.logMessage(
                f"Failed to create color layer: {error}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return False

        self.result_layer = rgb_layer

        rgb_layer.triggerRepaint()
        return True

    def finished(self, result):
        """Finished the color image creation."""
        QgsMessageLog.logMessage(
            "Finished color task",
            "ICEYE Toolbox",
            Qgis.Info,
        )
        QgsProject.instance().removeMapLayer(self.crop_task.result_layer)
        self.crop_task.result_layer = None
        self.crop_task.result_layer_path = None

        if not result:
            self.iface.messageBar().pushMessage(
                "Failed to create color image", level=Qgis.Critical, duration=3
            )

            return


def color_image(
    data: NDArray[np.complex64],
    metadata: IceyeMetadata,
    progress_callback: Callable[[int], None] = lambda x: None,
) -> NDArray[np.float32]:
    """Create RGB color image from SLC using range-direction spectrum.

    Parameters
    ----------
    data : ndarray of complex64
        SLC data (range x azimuth).
    metadata : IceyeMetadata
        ICEYE metadata (unused).
    progress_callback : callable, optional
        Progress callback (0-100).

    Returns
    -------
    ndarray of float32
        RGB image (height, width, 3).

    """
    rows, cols = data.shape

    fft_data = np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)

    positions = np.linspace(0.0, 1.0, cols, dtype=np.float32)
    weight_r = 1.0 - positions
    weight_g = np.clip(1.0 - np.abs(positions - 0.5) * 2.0, 0.0, 1.0)
    weight_b = positions

    mag_r = np.abs(
        np.fft.ifft(np.fft.ifftshift(fft_data * weight_r[None, :], axes=1), axis=1)
    ).astype(np.float32)
    mag_g = np.abs(
        np.fft.ifft(np.fft.ifftshift(fft_data * weight_g[None, :], axes=1), axis=1)
    ).astype(np.float32)
    mag_b = np.abs(
        np.fft.ifft(np.fft.ifftshift(fft_data * weight_b[None, :], axes=1), axis=1)
    ).astype(np.float32)

    progress_callback(70)

    r = normalize_channel(mag_r)
    g = normalize_channel(mag_g)
    b = normalize_channel(mag_b)

    progress_callback(90)

    return np.dstack([r, g, b])


def color_image_slow_time(
    data: NDArray[np.complex64],
    metadata: IceyeMetadata,
    progress_callback: Callable[[int], None] = lambda x: None,
) -> NDArray[np.float32]:
    """Create RGB color image from SLC using slow-time (azimuth) spectrum."""
    rows, cols = data.shape

    fft_data = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)

    positions = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    weight_r = 1.0 - positions
    weight_g = np.clip(1.0 - np.abs(positions - 0.5) * 2.0, 0.0, 1.0)
    weight_b = positions

    mag_r = np.abs(
        np.fft.ifft(np.fft.ifftshift(fft_data * weight_r[:, None], axes=0), axis=0)
    ).astype(np.float32)
    mag_g = np.abs(
        np.fft.ifft(np.fft.ifftshift(fft_data * weight_g[:, None], axes=0), axis=0)
    ).astype(np.float32)
    mag_b = np.abs(
        np.fft.ifft(np.fft.ifftshift(fft_data * weight_b[:, None], axes=0), axis=0)
    ).astype(np.float32)

    progress_callback(70)

    r = normalize_channel(mag_r)
    g = normalize_channel(mag_g)
    b = normalize_channel(mag_b)

    progress_callback(90)

    return np.dstack([r, g, b])


COLOR_FUNCS = {
    "fast_time": color_image,
    "slow_time": color_image_slow_time,
}


def create_color_raster_layer(
    rgb_data: NDArray[np.float32],
    source_layer: QgsRasterLayer,
    left: bool,
) -> tuple[bool, QgsRasterLayer | None, str | None]:
    """Create a color raster layer from RGB data.

    Parameters
    ----------
    rgb_data : ndarray of float32
        RGB image (height, width, 3).
    source_layer : QgsRasterLayer
        Source layer for metadata and path.
    left : bool
        True if left-looking SAR (for orientation).

    Returns
    -------
    tuple of (bool, QgsRasterLayer or None, str or None)
        Success, layer if successful, error message if failed.
    """
    source_path = source_layer.dataProvider().dataSourceUri()
    output_dir = Path(source_path).parent

    layer_name = source_layer.name().replace("_CROP_", "_COLOR_")
    output_path = output_dir / f"{layer_name}.tif"

    height, width, num_bands = rgb_data.shape

    output_raster = create_multiband_raster(
        str(output_path), num_bands, (height, width), source_path
    )

    src_ds = gdal.Open(source_path)
    if src_ds:
        QgsMessageLog.logMessage(
            f"Getting metadata from source layer {source_layer.name()}",
            "ICEYE Toolbox",
            Qgis.Info,
        )
        try:
            properties = src_ds.GetMetadata()
            if properties:
                properties = src_ds.GetMetadata()["ICEYE_PROPERTIES"]
                output_raster.SetMetadataItem("ICEYE_PROPERTIES", properties)

            else:
                error = "No metadata found in source layer"
                return False, None, error

        except KeyError:
            error = "Failed to get metadata from source layer"
            return False, None, error

    else:
        return False, None, "Failed to open source layer"

    write_frame_to_band(output_raster, rgb_data[:, :, 0], 1, left)
    write_frame_to_band(output_raster, rgb_data[:, :, 1], 2, left)
    write_frame_to_band(output_raster, rgb_data[:, :, 2], 3, left)

    output_raster.FlushCache()
    output_raster = None

    try:
        overview_params = {
            "INPUT": str(output_path),
            "LEVELS": "2 4 8 16",
            "RESAMPLING": 1,
            "FORMAT": 0,
            "EXTRA": "",
        }
        processing.run("gdal:overviews", overview_params)
        QgsMessageLog.logMessage(
            f"Built overviews for: {output_path}",
            "ICEYE Toolbox",
            Qgis.Info,
        )
    except Exception:
        return False, None, "Failed to build overviews"

    rgb_layer = QgsRasterLayer(str(output_path), layer_name)
    if not rgb_layer.isValid():
        return False, None, "Failed to create color layer"

    QgsProject.instance().addMapLayer(rgb_layer, True)

    return True, rgb_layer, None


def normalize_channel(channel: NDArray[np.float32]) -> NDArray[np.float32]:
    """Normalize channel to [0, 1] using log-scale."""
    log_data = np.log10(channel + 1e-10)
    min_val = float(np.min(log_data))
    max_val = float(np.max(log_data))
    if max_val <= min_val:
        return np.zeros_like(log_data, dtype=np.float32)
    return ((log_data - min_val) / (max_val - min_val)).astype(np.float32)
