"""Cropping and extent utilities for raster layers."""

from __future__ import annotations

import hashlib
import math

import numpy as np
from osgeo import gdal
from PyQt5.QtGui import QColor
from qgis import processing
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsFeature,
    QgsGeometry,
    QgsLayerTreeGroup,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsTask,
    QgsVectorLayer,
)
from qgis.gui import QgsMapToolExtent

from .batch_runner import BatchExtentRunner, BatchStepResult


class CropTool:
    """Cropper class for cropping layers."""

    def __init__(self, iface):
        """Initialize the cropper."""
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.layer = None
        self.task = None
        self.mapToolExtent = QgsMapToolExtent(self.canvas)
        self._mask_factory = MaskLayerFactory(self.canvas)

        self._batch_current_mask_id: str | None = None
        self._batch_runner = BatchExtentRunner(
            iface,
            label="Batch crop",
            get_task=lambda: self.task,
            set_task=lambda t: setattr(self, "task", t),
            is_sibling_task_running=self._crop_task_running,
            batch_already_running_msg="Batch crop is already running.",
            sibling_task_running_msg="A crop task is already running.",
        )

        self.mapToolExtent.extentChanged.connect(lambda extent: self._crop(extent))

    def activate(self):
        """Activate the crop tool."""
        self.layer = self.iface.activeLayer()
        self.canvas.setMapTool(self.mapToolExtent)

    def _crop(self, extent: QgsRectangle):
        """Handle the cropped area."""
        self.canvas.unsetMapTool(self.mapToolExtent)
        self.process_extent(extent)

    def process_extent(self, extent: QgsRectangle):
        """Run crop logic for the given extent (no map tool handling)."""
        layer = self.iface.activeLayer()
        if not layer:
            self.iface.messageBar().pushMessage(
                "Warning", "No TIF layer found", level=Qgis.Warning, duration=3
            )
            return

        QgsMessageLog.logMessage(
            f"Cropping layer: {layer.name()} with extent: {extent.toString()}",
            "ICEYE Toolbox",
            Qgis.Info,
        )

        self.task = CropLayerTask(layer, extent)
        self.mask = self._mask_factory.create(extent)
        self.task.taskCompleted.connect(
            lambda: QgsProject.instance().removeMapLayer(self.mask)
        )
        QgsApplication.taskManager().addTask(self.task)

    def _crop_task_running(self) -> bool:
        """Return True if a single or batch crop QgsTask is queued or running."""
        if self.task is None:
            return False
        try:
            return self.task.status() not in (QgsTask.Complete, QgsTask.Terminated)
        except RuntimeError:
            self.task = None
            return False

    def process_extents_batch(self, jobs: list[tuple[QgsRectangle, str]]) -> None:
        """Crop each (extent, source_layer_id) in order, one task at a time."""
        self._batch_runner.start(
            jobs,
            self._batch_crop_step,
            on_after_each_step=self._after_batch_crop_step,
            start_log=f"Batch crop: starting {len(jobs)} extent(s)",
            start_message=f"Batch crop: {len(jobs)} area(s)…",
        )

    def _batch_crop_step(
        self,
        extent: QgsRectangle,
        layer_id: str,
        source_layer: QgsRasterLayer,
        index: int,
        total: int,
    ) -> BatchStepResult:
        mask_name = f"Batch crop mask {index}"
        mask = self._mask_factory.create(extent, layer_name=mask_name)
        task = CropLayerTask(source_layer, extent)
        self._batch_current_mask_id = mask.id()
        return BatchStepResult(task=task)

    def _after_batch_crop_step(self) -> None:
        if self._batch_current_mask_id is not None:
            try:
                QgsProject.instance().removeMapLayer(self._batch_current_mask_id)
            except Exception:
                pass
            self._batch_current_mask_id = None
        QgsMessageLog.logMessage(
            f"Batch crop: completed step {self._batch_runner.step_index}/{self._batch_runner.total}",
            "ICEYE Toolbox",
            Qgis.Info,
        )


class CropLayerTask(QgsTask):
    """Task to clipping raster from a mask."""

    def __init__(self, layer: QgsRasterLayer, extent: QgsRectangle):
        super().__init__("Clipping raster layer", QgsTask.CanCancel)
        self.layer = layer
        self.extent = extent
        self.extend_image_coords = None
        self.result_path = None
        self.result_layer = None
        self.error_msg = None

    def run(self):
        """Run the clipping process."""
        if self.isCanceled():
            return False

        QgsMessageLog.logMessage(
            f"Running crop layer task for layer: {self.layer.name()} with extent: {self.extent.toString()}",
            "ICEYE Toolbox",
            Qgis.Info,
        )

        self.extend_image_coords = get_extend_image_coords(self.layer, self.extent)
        QgsMessageLog.logMessage(
            f"Extend image coords: {self.extend_image_coords.toString()}",
            "ICEYE Toolbox",
            Qgis.Info,
        )
        if not self.extend_image_coords:
            self.error_msg = "Failed to get extend image coords"
            QgsMessageLog.logMessage(
                "Failed to get extend image coords",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return False

        # Check and grow pixel extent if needed
        self.extend_image_coords, was_modified = check_and_grow_pixel_extent(
            self.extend_image_coords, min_pixels=100
        )

        params = {
            "INPUT": self.layer.dataProvider().dataSourceUri(),
            "OUTPUT": "TEMPORARY_OUTPUT",
            "EXTRA": "-r near -srcwin "
            + " ".join(
                [
                    str(math.floor(self.extend_image_coords.xMinimum())),
                    str(math.floor(self.extend_image_coords.yMinimum())),
                    str(math.floor(self.extend_image_coords.width())),
                    str(math.floor(self.extend_image_coords.height())),
                ]
            ),
        }

        try:
            result = processing.run("gdal:translate", params)
            self.result_path = result["OUTPUT"]
        except Exception as e:
            self.error_msg = "Failed to clip layer"
            QgsMessageLog.logMessage(
                f"Error in clipping layer: {str(e)}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return False

        if self.isCanceled():
            return False

        self.setProgress(50)
        overview_params = {
            "INPUT": self.result_path,
            "LEVELS": "2 4 8 16",
            "RESAMPLING": 1,
            "FORMAT": 0,
            "EXTRA": "",
        }

        try:
            processing.run("gdal:overviews", overview_params)
        except Exception as e:
            self.error_msg = "Failed to build overviews"
            QgsMessageLog.logMessage(
                f"Error in building overviews: {str(e)}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
        QgsMessageLog.logMessage(
            f"{overview_params}",
            "ICEYE Toolbox",
            Qgis.Info,
        )

        # Copy full metadata
        # TODO Crop metedata that need to be cropped
        try:
            input_dataset = gdal.Open(self.layer.dataProvider().dataSourceUri())
            if not input_dataset:
                QgsMessageLog.logMessage(
                    f"Failed to open input dataset {self.layer.name()}",
                    "ICEYE Toolbox",
                    Qgis.Critical,
                )
                self.error_msg = f"Failed to open input dataset {self.layer.name()}"
                return False
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Error in opening input dataset: {str(e)}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            self.error_msg = f"Error in opening input dataset: {str(e)}"
            return False

        try:
            out_dataset = gdal.Open(self.result_path, gdal.GA_Update)
            if not out_dataset:
                QgsMessageLog.logMessage(
                    f"Failed to open output dataset {self.result_path}",
                    "ICEYE Toolbox",
                    Qgis.Critical,
                )
                self.error_msg = f"Failed to open output dataset {self.result_path}"
                return False
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Error in opening output dataset: {str(e)}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            self.error_msg = f"Error in opening output dataset: {str(e)}"
            return False

        try:
            properties = input_dataset.GetMetadata()["ICEYE_PROPERTIES"]
            if not properties:
                QgsMessageLog.logMessage(
                    f"Missing ICEYE_PROPERTIES in input dataset {self.layer.name()}",
                    "ICEYE Toolbox",
                    Qgis.Warning,
                )
                self.error_msg = "Missing ICEYE_PROPERTIES in input dataset"
                return False

            out_dataset.SetMetadata(properties, "ICEYE_PROPERTIES")

        except Exception as e:
            QgsMessageLog.logMessage(
                f"Error in setting metadata: {str(e)}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            self.error_msg = f"Error in setting metadata: {str(e)}"
            return False

        finally:
            input_dataset = None
            out_dataset = None

        self.setProgress(100)
        return True

    def finished(self, result: bool):
        """Finished the clipping task."""
        QgsMessageLog.logMessage(
            f"Result path: {self.result_path}",
            "ICEYE Toolbox",
            Qgis.Info,
        )

        if not result or not self.result_path:
            QgsMessageLog.logMessage(
                f"Clipping failed: {self.error_msg}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return

        if "_CROP_" in self.layer.name():
            base_name = self.layer.name().split("_CROP_")[0]
        else:
            base_name = self.layer.name().rsplit("_", 1)[0]

        m = hashlib.sha1(self.extent.toString().encode()).hexdigest()[:8]
        result_name = base_name + "_CROP_" + m
        raster = QgsRasterLayer(self.result_path, result_name)
        if not raster.isValid():
            QgsMessageLog.logMessage(
                f"Generated clipped layer from {self.result_path} is not valid",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return
        self.result_layer = raster

        QgsMessageLog.logMessage(
            f"Adding raster to project: {raster.name()}",
            "ICEYE Toolbox",
            Qgis.Info,
        )
        QgsProject.instance().addMapLayer(raster, True)


class MaskLayerFactory:
    """Factory for creating mask vector layers from extents."""

    def __init__(self, canvas) -> None:
        self.canvas = canvas

    def create(
        self,
        rect: QgsRectangle,
        layer_name: str = "Mask Layer",
        parent_group: QgsLayerTreeGroup | None = None,
    ) -> QgsVectorLayer:
        """Create a polygon mask layer from the given extent.

        If *parent_group* is set, the layer is added under that legend group only.
        """
        crs = self.canvas.mapSettings().destinationCrs().authid()
        layer = QgsVectorLayer(f"Polygon?crs={crs}", layer_name, "memory")
        provider = layer.dataProvider()
        geom = QgsGeometry.fromRect(rect)

        feature = QgsFeature()
        feature.setGeometry(geom)
        provider.addFeatures([feature])
        layer.updateExtents()

        # Optional styling
        symbol = layer.renderer().symbol()
        symbol.setColor(QColor("olivedrab"))
        symbol.setOpacity(0.3)
        layer.triggerRepaint()

        project = QgsProject.instance()
        if parent_group is not None:
            project.addMapLayer(layer, False)
            parent_group.addLayer(layer)
        else:
            project.addMapLayer(layer, True)

        return layer


def get_extend_image_coords(
    layer: QgsRasterLayer, extent: QgsRectangle
) -> QgsRectangle | None:
    """Transform geographic extent to pixel coordinates for the layer.

    Parameters
    ----------
    layer : QgsRasterLayer
        Raster layer with georeferencing.
    extent : QgsRectangle
        Geographic extent to transform.

    Returns
    -------
    QgsRectangle or None
        Pixel extent (x, y, width, height) or None on failure.
    """
    dataset = gdal.Open(layer.dataProvider().dataSourceUri())
    if dataset is None:
        return None

    try:
        t = gdal.Transformer(dataset, None, ["METHOD=GCP_TPS"])
        if t is None:
            raise NoneOutputError("Not output from Transformer")

        upper_left = [extent.xMinimum(), extent.yMaximum(), 0.0]
        upper_right = [extent.xMaximum(), extent.yMaximum(), 0.0]
        lower_left = [extent.xMinimum(), extent.yMinimum(), 0.0]
        lower_right = [extent.xMaximum(), extent.yMinimum(), 0.0]

        (success, upper_left_dst) = t.TransformPoint(1, *upper_left)
        if not success:
            return None
        (success, upper_right_dst) = t.TransformPoint(1, *upper_right)
        if not success:
            return None
        (success, lower_left_dst) = t.TransformPoint(1, *lower_left)
        if not success:
            return None
        (success, lower_right_dst) = t.TransformPoint(1, *lower_right)
        if not success:
            return None

        offset_x = np.min(
            [
                upper_left_dst[0],
                upper_right_dst[0],
                lower_left_dst[0],
                lower_right_dst[0],
            ]
        )
        offset_y = np.min(
            [
                upper_left_dst[1],
                upper_right_dst[1],
                lower_left_dst[1],
                lower_right_dst[1],
            ]
        )
        dx = (
            np.max(
                [
                    upper_left_dst[0],
                    upper_right_dst[0],
                    lower_left_dst[0],
                    lower_right_dst[0],
                ]
            )
            - offset_x
        )
        dy = (
            np.max(
                [
                    upper_left_dst[1],
                    upper_right_dst[1],
                    lower_left_dst[1],
                    lower_right_dst[1],
                ]
            )
            - offset_y
        )

        return QgsRectangle(offset_x, offset_y, offset_x + dx, offset_y + dy)
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to compute extent {e}",
            "ICEYE Toolbox",
            Qgis.Critical,
        )
    finally:
        dataset = None

    return None


class NoneOutputError(Exception):
    """Raised when GDAL Transformer returns None instead of a valid output."""

    pass


def check_and_grow_pixel_extent(
    pixel_extent: QgsRectangle, min_pixels: int = 100
) -> tuple[QgsRectangle, bool]:
    """Check if pixel extent meets minimum size and grow if needed.

    Parameters
    ----------
    pixel_extent : QgsRectangle
        Pixel extent from get_extend_image_coords().
    min_pixels : int, optional
        Minimum pixels in each dimension. Default is 100.

    Returns
    -------
    tuple of (QgsRectangle, bool)
        Grown extent (or original if large enough) and whether it was modified.
    """
    width_pixels = pixel_extent.width()
    height_pixels = pixel_extent.height()

    QgsMessageLog.logMessage(
        f"Extent pixel dimensions - Width: {width_pixels}, Height: {height_pixels}",
        "ICEYE Toolbox",
        Qgis.Info,
    )

    # If extent is large enough, return unchanged
    if width_pixels >= min_pixels and height_pixels >= min_pixels:
        return pixel_extent, False

    # Grow the pixel extent to meet minimum requirements
    grown_pixel_extent = QgsRectangle(pixel_extent)  # Create a copy

    if width_pixels < min_pixels:
        grow_width = (min_pixels - width_pixels) / 2
        grown_pixel_extent.setXMinimum(pixel_extent.xMinimum() - grow_width)
        grown_pixel_extent.setXMaximum(pixel_extent.xMaximum() + grow_width)

    if height_pixels < min_pixels:
        grow_height = (min_pixels - height_pixels) / 2
        grown_pixel_extent.setYMinimum(pixel_extent.yMinimum() - grow_height)
        grown_pixel_extent.setYMaximum(pixel_extent.yMaximum() + grow_height)

    QgsMessageLog.logMessage(
        f"Extent was too small, growing from ({width_pixels}x{height_pixels}) to "
        f"({grown_pixel_extent.width()}x{grown_pixel_extent.height()}) pixels",
        "ICEYE Toolbox",
        Qgis.Info,
    )

    return grown_pixel_extent, True
