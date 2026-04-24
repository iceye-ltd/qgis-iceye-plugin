"""Export utilities: PNG, COG, GIF, MP4 from raster layers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PyQt5.QtCore import QSize, Qt
from PyQt5.QtGui import QColor, QImage
from PyQt5.QtWidgets import QFileDialog
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsMessageLog,
    QgsRasterLayer,
    QgsSingleBandGrayRenderer,
    QgsSingleBandPseudoColorRenderer,
    QgsTask,
)
from qgis.gui import QgsMapCanvas

from iceye_toolbox.core.raster import read_all_band_from_layer

from ..gui.export_dialog import ExportDialog

# Constants
GIF_DURATION = 250
MP4_FPS = 3


########################################################
# Export tool
########################################################
class ExportTool:
    """Tool for exporting layers and canvas."""

    def __init__(self, iface: Any):
        self.iface = iface
        self.task: QgsTask | None = None

    def export_canvas(self, iface: Any, canvas: QgsMapCanvas | None = None) -> bool:
        """Quick export of current canvas view."""
        if canvas is None:
            canvas = iface.mapCanvas()

        output_path, _ = QFileDialog.getSaveFileName(
            iface.mainWindow(),
            "Export Map Canvas",
            "canvas.png",
            "PNG Files (*.png);;JPEG Files (*.jpg)",
        )
        if not output_path:
            return False

        pixmap = canvas.grab()

        if pixmap.save(output_path):
            iface.messageBar().pushMessage(
                "Export",
                f"Canvas exported successfully to {output_path}",
                level=Qgis.Success,
                duration=5,
            )
            return True
        iface.messageBar().pushMessage(
            "Error", "Failed to save canvas image", level=Qgis.Critical, duration=5
        )
        return False

    def export_layer(self, layer: QgsRasterLayer | None = None) -> bool:
        """Export a layer with automatic format detection."""
        # Get and validate layer
        layer = layer or self.iface.activeLayer()

        if not isinstance(layer, QgsRasterLayer):
            QgsMessageLog.logMessage(
                "Layer is not a raster layer", "ICEYE Toolbox", level=Qgis.Warning
            )
            return False

        self.task = None

        default_name = f"{layer.name()}.tif"
        file_filter = "GIF files (*.gif);;MP4 files (*.mp4);;PNG files (*.png);;TIFF files (*.tif *.tiff)"

        output_path, _ = QFileDialog.getSaveFileName(
            self.iface.mainWindow(), "Select Output File", default_name, file_filter
        )

        # If user cancelled the file dialog, exit
        if not output_path:
            return False

        # Show export dialog
        dialog = ExportDialog(
            self.iface.mainWindow(),
            layer,
            default_path=output_path,
            file_filter=file_filter,
        )

        if dialog.exec_() != ExportDialog.Accepted:
            return False

        settings = dialog.get_export_settings()
        if not settings["output_path"]:
            QgsMessageLog.logMessage(
                "Settings are missing", "ICEYE Toolbox", level=Qgis.Warning
            )
            return False

        output_path = Path(settings["output_path"])
        output_extension = output_path.suffix
        if output_extension is None:
            return False

        if output_extension == ".png":
            self.task = ExportLayerAsPNG(
                f"Exporting {layer.name()}",
                layer,
                settings["output_path"],
                downscale_factor=settings["downscale_factor"],
            )

        if output_extension in [".tif", ".tiff"]:
            self.task = ExportLayerAsCOG(
                f"Exporting {layer.name()}",
                layer,
                settings["output_path"],
            )

        if output_extension in [".gif", ".mp4"]:
            self.task = ExportMultiBandLayer(
                f"Exporting {layer.name()}",
                layer,
                settings["output_path"],
                format="gif" if output_extension == ".gif" else "mp4",
                downscale_factor=settings["downscale_factor"],
            )

        if self.task is None:
            QgsMessageLog.logMessage(
                f"Nothing to be done for export {settings['output_path']}"
            )
            return False
        QgsApplication.taskManager().addTask(self.task)
        return True


########################################################
# Export layer task
########################################################


class ExportLayerAsPNG(QgsTask):
    """Export a raster layer as a rendered PNG image."""

    def __init__(
        self,
        description: str,
        layer: QgsRasterLayer,
        output_path: str,
        downscale_factor: float = 1.0,
    ) -> None:
        super().__init__(description, QgsTask.CanCancel)

        self.layer = layer
        self.output_path: str = output_path
        self.downscale_factor: float = downscale_factor

    def run(self) -> bool:
        """Export static layer as png."""
        if self.isCanceled():
            return False

        ms = QgsMapSettings()
        ms.setLayers([self.layer])
        ms.setExtent(self.layer.extent())
        ms.setOutputSize(
            QSize(
                int(self.layer.width()),
                int(self.layer.height()),
            )
        )
        ms.setFlag(QgsMapSettings.UseRenderingOptimization, True)
        ms.setFlag(QgsMapSettings.Antialiasing, True)
        ms.setFlag(QgsMapSettings.HighQualityImageTransforms, True)
        ms.setBackgroundColor(QColor(0, 0, 0, 0))  # RGBA with alpha=0 for transparency

        job = QgsMapRendererParallelJob(ms)
        job.start()
        job.waitForFinished()

        img = job.renderedImage()
        img = img.scaled(
            img.size() * self.downscale_factor,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if not img.save(self.output_path, "PNG", 100):
            QgsMessageLog.logMessage(f"failed to save {self.output_path}")
            return False

        return True

    def finished(self, result: bool) -> None:
        """Log success when export completes."""
        if result:
            QgsMessageLog.logMessage(
                f"Layer exported to {self.output_path}",
                "ICEYE Toolbox",
                level=Qgis.Success,
            )


class ExportLayerAsCOG(QgsTask):
    """Export a raster layer as a COG (Cloud Optimized GeoTIFF) by copying."""

    def __init__(
        self,
        description: str,
        layer: QgsRasterLayer,
        output_path: str,
    ) -> None:
        super().__init__(description, QgsTask.CanCancel)
        self.layer = layer
        self.output_path: Path = Path(output_path)

    def run(self) -> bool:
        """Copy the layer source to the output path."""
        shutil.copyfile(self.layer.dataProvider().dataSourceUri(), self.output_path)
        return True

    def finished(self, result: bool) -> None:
        """Log success when export completes."""
        if result:
            QgsMessageLog.logMessage(
                f"Layer exported to {self.output_path}",
                "ICEYE Toolbox",
                level=Qgis.Success,
            )


class ExportMultiBandLayer(QgsTask):
    """Export a multi-band raster layer as animated GIF or MP4."""

    def __init__(
        self,
        description: str,
        layer: QgsRasterLayer,
        output_path: str,
        format: Literal["gif", "mp4"] = "gif",
        downscale_factor: float = 1.0,
        frame_duration_ms: float = GIF_DURATION,
        fps: float | None = None,
    ) -> None:
        super().__init__(description, QgsTask.CanCancel)

        self.layer = layer
        self.output_path: str = output_path
        self.format: Literal["gif", "mp4"] = format
        self.downscale_factor: float = downscale_factor
        # fps overrides frame_duration_ms for GIF (fps=4 -> 250ms per frame)
        if fps is not None and fps > 0:
            self.frame_duration_ms = 1000 / fps
        else:
            self.frame_duration_ms = frame_duration_ms

    def run(self) -> bool:
        """Render each band as a frame and convert to GIF or MP4."""

        def update_progress(progress: float) -> None:
            progress_percent = 10 + int(progress * 85)
            self.setProgress(progress_percent)

        total_frames = self.layer.bandCount()
        format_name = self.format.upper()
        QgsMessageLog.logMessage(
            f"Starting {format_name} export: {total_frames} frames to {self.output_path}",
            "ICEYE Toolbox",
            level=Qgis.Info,
        )

        width = int(self.layer.width() * self.downscale_factor)
        height = int(self.layer.height() * self.downscale_factor)

        QgsMessageLog.logMessage(
            f"Export dimensions: {width}x{height} ({int(self.downscale_factor * 100)}% scale), bands: {total_frames}",
            "ICEYE Toolbox",
            level=Qgis.Info,
        )

        temp_dir = Path(tempfile.gettempdir()) / "qgis"
        temp_dir.mkdir(parents=True, exist_ok=True)
        frame_prefix = "multiband_frame"
        frame_paths = []

        def on_finished(
            job: QgsMapRendererParallelJob, path: Path, downscaling_factor: float
        ):
            img = job.renderedImage()
            scaled = img.scaled(img.size() * downscaling_factor)
            if not scaled.save(str(path), "PNG", 100):
                QgsMessageLog.logMessage(
                    f"failed to save {path}", "ICEYE Toolbox", Qgis.Warning
                )

        # Generate PNG frames for each band
        for band in range(1, total_frames + 1):
            if self.isCanceled():
                QgsMessageLog.logMessage(
                    f"{format_name} export cancelled", "ICEYE Toolbox", Qgis.Warning
                )
                return False

            progress = (band - 1) / total_frames
            update_progress(progress)

            renderer = self.layer.renderer()
            if isinstance(renderer, QgsSingleBandGrayRenderer):
                renderer.setGrayBand(band)
            elif isinstance(renderer, QgsSingleBandPseudoColorRenderer):
                renderer.setBand(band)
            else:
                renderer.setInputBand(band)

            ms = QgsMapSettings()
            ms.setLayers([self.layer])
            ms.setExtent(self.layer.extent())
            ms.setOutputSize(
                QSize(
                    int(self.layer.width()),
                    int(self.layer.height()),
                )
            )
            ms.setFlag(QgsMapSettings.UseRenderingOptimization, True)
            ms.setBackgroundColor(QColor(0, 0, 0, 0))

            frame_path = temp_dir / f"{frame_prefix}_{band}.png"
            job = QgsMapRendererParallelJob(ms)
            job.finished.connect(
                partial(on_finished, job, frame_path, self.downscale_factor)
            )
            job.start()
            job.waitForFinished()
            frame_paths.append(frame_path)

        # Convert PNG frames to output format
        try:
            existing_frames = [p for p in frame_paths if Path(p).exists()]
            if not existing_frames:
                QgsMessageLog.logMessage(
                    "No frame files were generated",
                    "ICEYE Toolbox",
                    Qgis.Critical,
                )
                return False

            if self.format == "gif":
                success = self._create_gif(existing_frames, temp_dir)
            else:
                success = self._create_mp4(existing_frames, temp_dir, frame_prefix)

        except FileNotFoundError:
            tool = "ImageMagick" if self.format == "gif" else "ffmpeg"
            QgsMessageLog.logMessage(
                f"{tool} not found. Please install {tool} for {format_name} export.",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return False
        finally:
            # Clean up temporary frame files
            for frame_path in frame_paths:
                try:
                    Path(frame_path).unlink(missing_ok=True)
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Failed to remove temporary file {frame_path}: {str(e)}",
                        "ICEYE Toolbox",
                        Qgis.Warning,
                    )

        return success

    def _create_gif(self, frame_paths: list[Path], temp_dir: Path) -> bool:
        QgsMessageLog.logMessage(
            f"Creating GIF from {len(frame_paths)} frames using ImageMagick",
            "ICEYE Toolbox",
            Qgis.Info,
        )

        for frame_path in frame_paths:
            if self.isCanceled():
                QgsMessageLog.logMessage(
                    "GIF export cancelled during frame processing",
                    "ICEYE Toolbox",
                    Qgis.Warning,
                )
                return False

        delay_cs = int(self.frame_duration_ms / 10)
        cmd = (
            ["convert", "-delay", str(delay_cs), "-loop", "0"]
            + [str(p) for p in frame_paths]
            + [self.output_path]
        )

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            QgsMessageLog.logMessage(
                f"GIF export completed: {self.output_path}",
                "ICEYE Toolbox",
                Qgis.Info,
            )
            return True
        else:
            QgsMessageLog.logMessage(
                f"ImageMagick convert failed: {result.stderr}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return False

    def _create_mp4(
        self, frame_paths: list[Path], temp_dir: Path, frame_prefix: str
    ) -> bool:
        QgsMessageLog.logMessage(
            f"Converting {len(frame_paths)} frames to MP4 using ffmpeg",
            "ICEYE Toolbox",
            Qgis.Info,
        )

        # ffmpeg image2 demuxer: -i multiband_frame_%d.png reads mp4_frame_1.png, mp4_frame_2.png, ...
        # scale filter ensures width/height divisible by 2 (required by libx264)
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(MP4_FPS),
            "-i",
            str(temp_dir / f"{frame_prefix}_%d.png"),
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(self.output_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            QgsMessageLog.logMessage(
                f"MP4 export completed: {self.output_path}",
                "ICEYE Toolbox",
                Qgis.Info,
            )
            return True
        else:
            QgsMessageLog.logMessage(
                f"ffmpeg failed: {result.stderr}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            return False

    def finished(self, result: bool) -> None:
        """Log success when export completes."""
        if result:
            QgsMessageLog.logMessage(
                f"Layer exported to {self.output_path}",
                "ICEYE Toolbox",
                level=Qgis.Success,
            )
