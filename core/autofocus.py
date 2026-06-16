"""Autofocus: centered-look PGA task, raster output, and helpers for lens PGA preview."""

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
    QgsVectorLayer,
)
from qgis.gui import QgsMapToolExtent

from .batch_runner import BatchExtentRunner, BatchStepResult
from .cropper import CropLayerTask, MaskLayerFactory
from .looks import extract_centered_look
from .raster import read_slc_layer
from .temporal_properties import (
    apply_temporal_properties,
    apply_temporal_properties_for_frames,
)
from .video import create_multiband_raster, write_frame_to_band

########################################################
# Autofocus
########################################################


class AutofocusTool:
    """Handler class for the autofocus workflow."""

    def __init__(self, iface, metadata_provider) -> None:
        """Initialize the autofocus handler."""
        self.iface = iface
        self.canvas = iface.mapCanvas()

        self.metadata_provider = metadata_provider
        self.mapToolExtent = QgsMapToolExtent(self.canvas)
        self.maskLayerFactory = MaskLayerFactory(self.canvas)

        self.mapToolExtent.extentChanged.connect(lambda extent: self._run(extent))

        self.task = None
        self.layer = None

        self._batch_current_mask_id: str | None = None
        self._batch_runner = BatchExtentRunner(
            iface,
            label="Batch focus",
            get_task=lambda: self.task,
            set_task=lambda t: setattr(self, "task", t),
            is_sibling_task_running=self._focus_task_running,
            batch_already_running_msg="Batch focus is already running.",
            sibling_task_running_msg="A focus task is already running.",
        )

    def activate(self):
        """Activate extent tool for selecting the target area."""
        self.layer = self.iface.activeLayer()
        self.canvas.setMapTool(self.mapToolExtent)

    def _run(self, extent):
        """Handle the selected extent."""
        self.canvas.unsetMapTool(self.mapToolExtent)
        self.process_extent(extent)

    def _focus_task_running(self) -> bool:
        """Return True if a crop/focus QgsTask is queued or running."""
        if self.task is None:
            return False
        try:
            return self.task.status() not in (
                QgsTask.TaskStatus.Complete,
                QgsTask.TaskStatus.Terminated,
            )
        except RuntimeError:
            # Qt has destroyed the underlying QgsTask; clear stale Python wrapper.
            self.task = None
            return False

    def _build_autofocus_task(
        self,
        extent: QgsRectangle,
        mask_name: str = "Mask Layer",
        source_layer: QgsRasterLayer | None = None,
    ) -> tuple[AutofocusTask, QgsVectorLayer] | tuple[None, None]:
        """Create mask, crop subtask, and parent AutofocusTask (not yet added to manager).

        If *source_layer* is None, the current active layer is used (single-click workflow).
        """
        layer = source_layer if source_layer is not None else self.iface.activeLayer()
        if not isinstance(layer, QgsRasterLayer):
            return None, None

        mask = self.maskLayerFactory.create(extent, layer_name=mask_name)
        if not mask:
            return None, None

        QgsMessageLog.logMessage(
            f"Mask layer created: {mask.name()}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )

        crop_task = CropLayerTask(layer, extent)
        task = AutofocusTask(self.iface, self.metadata_provider, crop_task)
        task.addSubTask(crop_task, [], QgsTask.SubTaskDependency.ParentDependsOnSubTask)
        return task, mask

    def process_extent(self, extent: QgsRectangle) -> None:
        """Run focus logic for the given extent (no map tool handling)."""
        if self._batch_runner.active:
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                "Batch focus is running — wait for it to finish.",
                level=Qgis.MessageLevel.Warning,
                duration=4,
            )
            return
        if self._focus_task_running():
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                "A focus task is already running.",
                level=Qgis.MessageLevel.Warning,
                duration=4,
            )
            return

        task, mask = self._build_autofocus_task(extent, "Mask Layer")
        if task is None or mask is None:
            active = self.iface.activeLayer()
            if not isinstance(active, QgsRasterLayer):
                self.iface.messageBar().pushMessage(
                    "No TIF layer found", level=Qgis.MessageLevel.Warning, duration=3
                )
            else:
                self.iface.messageBar().pushMessage(
                    "Failed to create mask layer",
                    level=Qgis.MessageLevel.Warning,
                    duration=3,
                )
            return

        self.task = task
        mask_id = mask.id()

        def _on_single_focus_done() -> None:
            try:
                QgsProject.instance().removeMapLayer(mask_id)
            except Exception:
                pass
            self.task = None

        self.task.taskCompleted.connect(_on_single_focus_done)
        self.task.taskTerminated.connect(_on_single_focus_done)
        QgsApplication.taskManager().addTask(self.task)

    def process_extents_batch(self, jobs: list[tuple[QgsRectangle, str]]) -> None:
        """Process each (extent, source_layer_id) in order: crop and focus, one QgsTask chain at a time.

        *source_layer_id* is the raster layer id from when the mask was drawn; layers are looked up per step.
        """
        self._batch_runner.start(
            jobs,
            self._batch_focus_step,
            on_after_each_step=self._after_batch_focus_step,
            start_log=f"Batch focus: starting {len(jobs)} extent(s)",
            start_message=f"Batch focus: {len(jobs)} area(s)…",
        )

    def _batch_focus_step(
        self,
        extent: QgsRectangle,
        layer_id: str,
        source_layer: QgsRasterLayer,
        index: int,
        total: int,
    ) -> BatchStepResult:
        mask_name = f"Batch focus mask {index}"
        task, mask = self._build_autofocus_task(
            extent, mask_name, source_layer=source_layer
        )
        if task is None or mask is None:
            QgsMessageLog.logMessage(
                f"Batch focus: step {index}/{total} failed (mask or task)",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )
            self.iface.messageBar().pushMessage(
                "ICEYE Toolbox",
                f"Batch aborted at step {index}: could not start crop/focus.",
                level=Qgis.MessageLevel.Critical,
                duration=5,
            )
            return BatchStepResult(abort=True)
        self._batch_current_mask_id = mask.id()
        return BatchStepResult(task=task)

    def _after_batch_focus_step(self) -> None:
        if self._batch_current_mask_id is not None:
            try:
                QgsProject.instance().removeMapLayer(self._batch_current_mask_id)
            except Exception:
                pass
            self._batch_current_mask_id = None
        QgsMessageLog.logMessage(
            f"Batch focus: completed step {self._batch_runner.step_index}/{self._batch_runner.total}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )


class AutofocusTask(QgsTask):
    """Background task that runs autofocus on the cropped extent."""

    def __init__(self, iface, metadata_provider, crop_task) -> None:
        """Initialize the focus task with crop result and metadata provider."""
        super().__init__("Autofocus", QgsTask.Flag.CanCancel)
        self.iface = iface
        self.crop_task = crop_task
        self.metadata_provider = metadata_provider

        self.result_layer = None

    def run(self):
        """Run autofocus."""
        QgsMessageLog.logMessage(
            f"Running for {self.crop_task.result_layer.name()}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )
        metadata = self.metadata_provider.get(self.crop_task.result_layer)
        data, _ = read_slc_layer(
            self.crop_task.result_layer,
            metadata.sar_observation_direction.lower() == "left",
            metadata,
        )

        QgsMessageLog.logMessage(
            f"Shape: {data.shape}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )

        try:
            QgsMessageLog.logMessage(
                "Autofocus: global range deviation correction (degree 2)",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Info,
            )
            data = apply_global_range_deviation_correction(
                data,
                progress_callback=lambda p: self.setProgress(p * 0.3),
            )
        except ValueError as e:
            QgsMessageLog.logMessage(
                f"Error in autofocus: {e!s}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Critical,
            )
            return False


        
        try:
            QgsMessageLog.logMessage(
                "Autofocus: centered-look PGA (10–25% of azimuth spectrum)",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Info,
            )
            focused_data = focus_with_centered_looks_pga(
                data,
                progress_callback=lambda p: self.setProgress(30.0 + p * 0.7),
            )
        except ValueError as e:
            QgsMessageLog.logMessage(
                f"Error in autofocus: {e!s}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Critical,
            )
            return False
        QgsMessageLog.logMessage(
            f"Focused data shape: {focused_data.shape}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )

        success, new_layer, error = create_focused_raster_layer(
            focused_data,
            self.crop_task.result_layer,
            metadata.sar_observation_direction.lower() == "left",
        )

        if not success:
            QgsMessageLog.logMessage(
                f"Failed to create focused layer: {error}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Critical,
            )
            return False

        self.result_layer = new_layer

        if len(focused_data.shape) == 3:
            apply_temporal_properties_for_frames(new_layer, metadata)
        else:
            apply_temporal_properties(new_layer, metadata)

        # Trigger repaint of the new layer
        new_layer.triggerRepaint()
        return True

    def finished(self, result):
        """Handle completion of autofocus."""
        QgsMessageLog.logMessage(
            "Finished focus task",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )
        QgsProject.instance().removeMapLayer(self.crop_task.result_layer)
        self.crop_task.result_layer = None
        self.crop_task.result_layer_path = None

        if not result:
            self.iface.messageBar().pushMessage(
                "Autofocus failed", level=Qgis.MessageLevel.Critical
            )

            return


def create_focused_raster_layer(
    focused_data: NDArray[np.complex64], source_layer: QgsRasterLayer, left: bool
) -> tuple[bool, QgsRasterLayer | None, str | None]:
    """Create a new raster layer and write focused data into it.

    Parameters
    ----------
    focused_data : ndarray of complex64
        Complex or real focused data (2D or 3D).
    source_layer : QgsRasterLayer
        Source layer for geospatial info.
    left : bool
        True if left-looking SAR.

    Returns
    -------
    tuple of (bool, QgsRasterLayer or None, str or None)
        Success, layer if successful, error message if failed.
    """
    try:
        source_path = source_layer.dataProvider().dataSourceUri()
        output_dir = Path(source_path).parent

        layer_name = source_layer.name().replace("_CROP_", "_FOCUS_")

        output_path = output_dir / f"{layer_name}.tif"
        # Use same directory as source layer so focused layer gets cleaned up

        if len(focused_data.shape) == 3:
            height, width, num_bands = focused_data.shape
        else:
            height, width = focused_data.shape
            num_bands = 2

        # Create output raster
        output_raster = create_multiband_raster(
            str(output_path), num_bands, (height, width), source_path
        )

        src_ds = gdal.Open(source_path)
        if src_ds:
            QgsMessageLog.logMessage(
                f"Getting metadata from source layer {source_layer.name()}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Info,
            )
            try:
                properties = src_ds.GetMetadata()
                if properties:
                    properties = src_ds.GetMetadata()["ICEYE_PROPERTIES"]
                    output_raster.SetMetadataItem("ICEYE_PROPERTIES", properties)
                    QgsMessageLog.logMessage(
                        f"Metadata set for output raster {output_path}",
                        "ICEYE Toolbox",
                        Qgis.MessageLevel.Info,
                    )
                else:
                    QgsMessageLog.logMessage(
                        f"No metadata found in source layer {source_layer.name()}",
                        "ICEYE Toolbox",
                        Qgis.MessageLevel.Warning,
                    )
            except KeyError:
                QgsMessageLog.logMessage(
                    f"Failed to get metadata from source layer {source_layer.name()}",
                    "ICEYE Toolbox",
                    Qgis.MessageLevel.Warning,
                )
            src_ds = None
        else:
            QgsMessageLog.logMessage(
                f"Failed to open source layer {source_layer.name()}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )

        QgsMessageLog.logMessage(
            f"observation direction: {left}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )

        if len(focused_data.shape) == 3:
            for i in range(num_bands):
                write_frame_to_band(output_raster, focused_data[:, :, i], i + 1, left)
        else:
            write_frame_to_band(output_raster, np.abs(focused_data), 1, left)
            write_frame_to_band(output_raster, np.angle(focused_data), 2, left)

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
                Qgis.MessageLevel.Info,
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to build overviews: {str(e)}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )

        # Create QgsRasterLayer
        result_layer = QgsRasterLayer(str(output_path), layer_name)

        if not result_layer.isValid():
            error_msg = f"Generated focused layer from {output_path} is not valid"
            QgsMessageLog.logMessage(
                error_msg, "ICEYE Toolbox", Qgis.MessageLevel.Critical
            )
            return False, None, error_msg

        QgsProject.instance().addMapLayer(result_layer, True)

        return True, result_layer, None

    except Exception as e:
        error_msg = f"Error creating focused raster: {str(e)}"
        QgsMessageLog.logMessage(error_msg, "ICEYE Toolbox", Qgis.MessageLevel.Critical)
        return False, None, error_msg


def weigthed_estimator(x):
    """Estimate phase gradient as a magnitude-weighted phase average."""
    s = np.conj(x[:-1, :]) * x[1:, :]
    return np.sum(
        np.angle(s) * np.abs(s),
        axis=1,
    ) / np.sum(np.abs(s), axis=1)


def phase_gradient_autofocus(
    data: NDArray[np.complex64],
    iter_num=1,
    estimator: Callable[[NDArray], NDArray] = weigthed_estimator,
    tolerance: float = 0.01,
) -> NDArray[np.complex64]:
    """Estimate and apply iterative phase-gradient autofocus corrections."""
    entropies = [entropy(data)]
    phase_corrections = np.zeros(data.shape[0])
    rms = []
    phase_changes = []
    for _ in range(iter_num):
        data_centered = center_on_strong_target(data, axis=1)

        # Window the data
        window = calculate_window(data_centered, axis=0)

        p = np.zeros_like(
            data_centered,
        )
        p[:, window] = data_centered[:, window]

        P = ft(p, axis=0)

        # Phase Gradient Estimation Maximum Likelihood
        phase_change = estimator(P)
        phase_change = np.unwrap([0, *np.cumsum(phase_change)])

        # Remove linear trend
        t = np.arange(0, phase_change.shape[0])
        trend = np.poly1d(np.polyfit(t, phase_change, 1))
        phase_change -= trend(t)
        rms.append(np.sqrt(np.mean(phase_change**2)))
        phase_changes.append(phase_change)

        if rms[-1] < tolerance:
            break

        # Apply phase correction
        data = ft(data, axis=0)
        data *= np.exp(-1j * phase_change[:, None])
        data = ift(data, axis=0)

        entropies.append(entropy(data))

        phase_corrections += phase_change

    phase_corrections = phase_corrections[:, None]
    return phase_corrections, rms, entropies


def ft(s: NDArray[np.complex64], axis: int = -1) -> NDArray[np.complex64]:
    """Compute centered FFT along the selected axis."""
    return np.fft.fftshift(np.fft.fft(s, axis=axis), axes=axis)


def ift(f: NDArray[np.complex64], axis: int = -1) -> NDArray[np.complex64]:
    """Compute inverse FFT from centered frequency-domain data."""
    return np.fft.ifft(np.fft.ifftshift(f, axes=axis), axis=axis)


def select_pulse_with_strong_target(
    s: NDArray[np.complex64], percentile: float = 95.0, axis: int = -1
) -> tuple[NDArray[np.int32], int]:
    """Select lines above a percentile strength threshold."""
    if axis not in [0, 1]:
        raise ValueError("Axis must be 0 or 1")
    line_max = np.amax(np.abs(s), axis=1 - axis)
    threshold = np.percentile(line_max, percentile)
    target_lines = np.where(line_max >= threshold)[0]
    if axis == 1:
        return s[:, target_lines], target_lines
    return s[target_lines], target_lines


def center_on_strong_target(x: NDArray[np.complex64], axis=-1):
    """Roll each line to place its strongest target at the center."""
    # Get the position of the strong signal on the azimuth axis
    H, W = x.shape
    max_index = np.argmax(np.abs(x), axis=axis)
    if axis == 1:
        center = W // 2
        shifts = center - max_index
        return x[np.arange(H)[:, None], (np.arange(W) - shifts[:, None]) % W]
    center = H // 2
    shifts = center - max_index
    return x[
        (np.arange(H) - shifts[:, None]) % H,
        np.arange(W)[:, None],
    ]


def calculate_window(
    s: NDArray[np.complex64], threshold: float = -20.0, min_width=50, axis=-1
) -> NDArray[np.int32]:
    """Select a target window from the power-spectrum profile."""
    p = np.sum(np.abs(s * np.conj(s)), axis=axis)
    p_max = p.max()
    if p_max > 0 and np.isfinite(p_max):
        p = 10.0 * np.log10(p / p_max)
        width = int(np.sum(p > threshold))
    else:
        # Degenerate patch (all-zero or non-finite power): fall back to min_width.
        width = min_width
    if width < min_width:
        width = min_width
    # Ensure width never exceeds the input dimension
    width = min(width, s.shape[axis] - 1)

    return np.arange(-width // 2, width // 2) + (s.shape[axis] - 1) // 2


def entropy(data: NDArray[np.complex64]) -> float:  # noqa: F811
    """Compute entropy of complex data (power-normalized)."""
    pwr = np.abs(data) ** 2
    pwr = pwr[pwr > 0]
    p = pwr / pwr.sum()
    return -np.sum(p * np.log(p))


def apply_phase_correction(data, phase_error):
    """Apply azimuth phase correction after interpolating phase error."""
    x = np.linspace(0.0, 1.0, data.shape[0])  # Changed to match azimuth dimension
    xp = np.linspace(0.0, 1.0, phase_error.shape[0])
    phase_error_interp = np.interp(x, xp, phase_error.squeeze())
    data = ft(data, axis=0)
    data *= np.exp(-1j * phase_error_interp[:, None])
    data = ift(data, axis=0)
    return data


def compute_contrast(image: NDArray[np.complex64]) -> float:
    """Image-intensity contrast: ``std(|x|^2) / mean(|x|^2)``.

    Higher values indicate sharper, more focused imagery. Returns 0 if the
    mean intensity is non-positive.
    """
    intensity = np.abs(image) ** 2
    mean = intensity.mean()
    if mean <= 0:
        return 0.0
    return float(intensity.std() / mean)

def compute_contrast_subaperture_sums(
    image: NDArray[np.complex64], N_subaperture: int
) -> NDArray[np.floating]:
    """Per-subaperture intensity contrast over the centered 80% of azimuth.

    Drops the outer 10% of azimuth rows on each side, splits the remaining
    centered 80% into ``N_subaperture`` equal chunks, and returns an array of
    ``std(|x|^2) / mean(|x|^2)`` per chunk. Returns zeros if the centered
    region has non-positive mean intensity.
    """
    H = image.shape[0]
    start = H // 10
    end = H - H // 10
    center = image[start:end]
    seg = (end - start) // N_subaperture
    intensity = np.zeros(N_subaperture, dtype=np.float32)
    for i in range(N_subaperture):
        intensity[i] = np.mean(np.abs(center[i * seg : (i + 1) * seg, :]) ** 2,axis = 0).std()
    return intensity

def shift_fitted(
    s: NDArray[np.complex64], fitted: NDArray[np.floating]
) -> NDArray[np.complex64]:
    """Apply a per-azimuth-row range shift defined by *fitted* (in samples).

    Each row ``k`` of *s* is shifted along the range axis by ``fitted[k]``
    samples via a Fourier-domain phase ramp. Vectorised over rows.
    Returns a new array; *s* is not modified.
    """
    N = s.shape[1]
    k_over_N = np.arange(N) / N - 0.5
    phase_ramp = np.exp(1j * 2.0 * np.pi * fitted[:, None] * k_over_N[None, :])
    shift_term = np.fft.fftshift(phase_ramp, axes=1)
    return np.fft.ifft(np.fft.fft(s, axis=1) * shift_term, axis=1)


def _contrast_ratio_sum(
    C: NDArray[np.floating], C_initial: NDArray[np.floating]
) -> float:
    """Sum of element-wise ``C / C_initial``, treating zero baselines as 1.

    Subapertures with non-positive mean intensity return 0 from
    :func:`compute_contrast_subaperture_sums`. For those, the ratio is
    undefined (``0/0``); we treat it as 1.0 (i.e. "no change") so the running
    max comparison stays stable and we never raise an "invalid value" warning.
    """
    ratio = np.divide(
        C,
        C_initial,
        out=np.ones_like(C, dtype=np.float64),
        where=C_initial > 0,
    )
    return float(np.sum(ratio))


def find_best_deviation(
    spatch_fft: NDArray[np.complex64],
    x: NDArray[np.floating],
    *,
    dev_min: float = -100.0,
    dev_max: float = 100.0,
    accuracy: float = 0.5,
    poly_degree: int = 2,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[float, int]:
    """Coarse-to-fine search for the polynomial range deviation maximizing contrast.

    Searches deviations in ``[dev_min, dev_max]`` with resolution *accuracy*
    (floats allowed) and returns ``(best_deviation, n_sub_contrast_improved)``,
    where the second value is the number of subapertures whose contrast
    improved over the previous best at the chosen deviation.
    """
    coarse_step = max(accuracy * 5.0, (dev_max - dev_min) / 20.0)
    best_deviation = 0.0
    N_subaperture = 10
    # Ratio-sum baseline: when C == C_initial, sum(C / C_initial) == N_subaperture.
    # Only updates on genuine improvement over the running best.
    max_contrast = float(N_subaperture)
    C_initial = compute_contrast_subaperture_sums(spatch_fft, N_subaperture)

    coarse = np.arange(dev_min, dev_max + coarse_step, coarse_step)
    n_coarse = max(len(coarse), 1)
    n_sub_contrast_improved = 0
    for i, deviation in enumerate(coarse):
        c = deviation / x[-1] ** poly_degree
        fitted = -c * x**poly_degree
        shifted = np.abs(shift_fitted(spatch_fft, fitted))
        C = compute_contrast_subaperture_sums(shifted, N_subaperture)
        contrast_gain = _contrast_ratio_sum(C, C_initial)
        QgsMessageLog.logMessage(
            f"[coarse {i+1}/{n_coarse}] deviation={float(deviation):.3f} "
            f"contrast_gain={contrast_gain:.3f}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )
        if contrast_gain > max_contrast:
            max_contrast = contrast_gain
            C_initial = np.copy(C)
            best_deviation = float(deviation)
        if progress_callback is not None:
            progress_callback(50.0 * (i + 1) / n_coarse)

    fine_min = max(dev_min, best_deviation - 2*coarse_step)
    fine_max = min(dev_max, best_deviation + 2*coarse_step)
    fine = np.arange(fine_min, fine_max + accuracy, accuracy)
    n_fine = max(len(fine), 1)
    for i, deviation in enumerate(fine):
        c = deviation / x[-1] ** poly_degree
        fitted = -c * x**poly_degree
        shifted = np.abs(shift_fitted(spatch_fft, fitted))
        C = compute_contrast_subaperture_sums(shifted, N_subaperture)
        contrast_gain = _contrast_ratio_sum(C, C_initial)
        QgsMessageLog.logMessage(
            f"[fine {i+1}/{n_fine}] deviation={float(deviation):.3f} "
            f"contrast_gain={contrast_gain:.3f}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )
        if contrast_gain > max_contrast:
            max_contrast = contrast_gain
            C_initial = np.copy(C)
            best_deviation = float(deviation)
        if progress_callback is not None:
            progress_callback(50.0 + 50.0 * (i + 1) / n_fine)

    return best_deviation, int(n_sub_contrast_improved)


def apply_global_range_deviation_correction(
    data: NDArray[np.complex64],
    *,
    dev_min: float = -100.0,
    dev_max: float = 100.0,
    accuracy: float = 0.5,
    poly_degree: int = 2,
    progress_callback: Callable[[float], None] | None = None,
) -> NDArray[np.complex64]:
    """Estimate and remove a global polynomial range deviation (range walk).

    The data is transformed to the azimuth-frequency / range-time domain;
    a polynomial range shift of degree *poly_degree* is fitted by maximizing
    intensity contrast (a coarse-to-fine grid search over ``[dev_min, dev_max]``
    to resolution *accuracy*), then applied. The time-domain corrected SLC is
    returned with the same dtype as the input.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D SLC data, got shape {data.shape}")

    rows = data.shape[0]
    spatch_fft = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    x = np.linspace(-0.5, 0.5, rows)
    best_deviation, n_sub_contrast_improved = find_best_deviation(
        spatch_fft,
        x,
        dev_min=dev_min,
        dev_max=dev_max,
        accuracy=accuracy,
        poly_degree=poly_degree,
        progress_callback=progress_callback,
    )

    QgsMessageLog.logMessage(
        f"Global range deviation correction: best_deviation={best_deviation:.3f} "
        f"(subapertures improved={n_sub_contrast_improved}/10, degree={poly_degree}, "
        f"search=[{dev_min}, {dev_max}] @ {accuracy})",
        "ICEYE Toolbox",
        Qgis.MessageLevel.Info,
    )

    fitted = -best_deviation / x[-1] ** poly_degree * x**poly_degree
    spatch_fft = shift_fitted(spatch_fft, fitted)
    corrected = np.fft.ifft(np.fft.ifftshift(spatch_fft, axes=0), axis=0)
    return corrected.astype(data.dtype, copy=False)


def _centered_look_row_counts_from_azimuth_fractions(
    spectrum_rows: int, azimuth_look_fractions: tuple[float, ...]
) -> list[int]:
    """Azimuth look heights as fixed fractions of the Doppler spectrum row count."""
    if spectrum_rows < 1:
        return []
    heights: list[int] = []
    for frac in azimuth_look_fractions:
        h = max(1, min(spectrum_rows, int(round(spectrum_rows * frac))))
        if not heights or h > heights[-1]:
            heights.append(h)
    return heights


def focus_with_centered_looks_pga(
    data: NDArray[np.complex64],
    *,
    azimuth_look_fractions: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25),
    progress_callback: Callable[[float], None] | None = None,
) -> NDArray[np.complex64]:
    """Autofocus via PGA on centered spectral looks; pick lowest-entropy look.

    Uses looks whose azimuth extent is specified by azimuth_look_fractions of the
    Doppler spectrum height (deduplicated if the crop is very small).

    Mirrors the lens-tool pipeline: 2D spectrum, centered azimuth looks, strong-pulse
    selection, :func:`phase_gradient_autofocus`, then :func:`apply_phase_correction`.
    The phase estimate from the winning look is applied to the full-resolution
    ``data``.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D SLC data, got shape {data.shape}")

    data = np.ascontiguousarray(data, dtype=np.complex64)
    spectrum = np.fft.fftshift(np.fft.fft2(data))
    rows, cols = spectrum.shape
    look_heights = _centered_look_row_counts_from_azimuth_fractions(
        rows, azimuth_look_fractions
    )
    QgsMessageLog.logMessage(
        "Centered-look PGA: Doppler spectrum "
        f"{rows} x {cols}, look azimuth rows {look_heights} "
        f"(from {', '.join(f'{f:.0%}' for f in azimuth_look_fractions)} of azimuth; duplicates dropped if crop is small)",
        "ICEYE Toolbox",
        Qgis.MessageLevel.Info,
    )
    if not look_heights:
        QgsMessageLog.logMessage(
            "Centered-look PGA: empty look list; returning input unchanged",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Warning,
        )
        return data

    best_entropy = float("inf")
    best_phase_error: NDArray[np.floating] | None = None
    best_look_idx: int | None = None
    best_look_rows: int | None = None
    n_eval = len(look_heights)
    best = None
    for i, azimuth_look_size in enumerate(look_heights):
        look = extract_centered_look(
            spectrum,
            center_row=rows // 2,
            center_col=cols // 2,
            look_rows=azimuth_look_size,
            look_cols=cols,
            apply_ifftshift=True,
        )
        patch, _ = select_pulse_with_strong_target(look, axis=0)
        if patch.size == 0:
            QgsMessageLog.logMessage(
                f"Centered-look PGA: look {i + 1}/{n_eval} "
                f"(azimuth_rows={azimuth_look_size}): no strong-pulse patch, skip",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )
            if progress_callback is not None:
                progress_callback(100.0 * (i + 1) / n_eval)
            continue

        phase_error, _, _ = phase_gradient_autofocus(patch)
        corrected_look = apply_phase_correction(look, phase_error)
        score = entropy(corrected_look)
        is_best = score < best_entropy
        if is_best:
            best_entropy = score
            best_phase_error = phase_error
            best_look_idx = i
            best_look_rows = azimuth_look_size
            best = corrected_look

        QgsMessageLog.logMessage(
            f"Centered-look PGA: look {i + 1}/{n_eval} "
            f"azimuth_rows={azimuth_look_size}, patch {patch.shape}, "
            f"entropy={score:.6f}{' (best so far)' if is_best else ''}",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Info,
        )

        if progress_callback is not None:
            progress_callback(100.0 * (i + 1) / n_eval)

    if best_phase_error is None:
        QgsMessageLog.logMessage(
            "Centered-look PGA: no valid phase estimate from any look; "
            "returning input unchanged",
            "ICEYE Toolbox",
            Qgis.MessageLevel.Warning,
        )
        return data

    QgsMessageLog.logMessage(
        "Centered-look PGA: applying phase correction from "
        f"look index {best_look_idx + 1}/{n_eval} "
        f"(azimuth_rows={best_look_rows}, entropy={best_entropy:.6f})",
        "ICEYE Toolbox",
        Qgis.MessageLevel.Info,
    )
    return best
