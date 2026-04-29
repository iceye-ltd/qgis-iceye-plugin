"""Lens tool: magnifying overlay with Normal, Focus, 2D Spectrum, and Color modes."""

from __future__ import annotations

import math
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from osgeo import gdal
from qgis.core import (
    Qgis,
    QgsGeometry,
    QgsMapLayer,
    QgsMapRendererParallelJob,
    QgsMapSettings,
    QgsMessageLog,
    QgsRasterLayer,
    QgsRectangle,
    QgsWkbTypes,
)
from qgis.gui import QgsMapToolPan, QgsRubberBand
from qgis.PyQt.QtCore import (
    QCoreApplication,
    QPoint,
    QPointF,
    QSize,
    Qt,
    QTimer,
    pyqtSignal,
)
from qgis.PyQt.QtGui import QBrush, QColor, QIcon, QImage, QKeySequence, QPen, QPixmap
from qgis.PyQt.QtWidgets import (
    QAction,
    QActionGroup,
    QGraphicsItemGroup,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QMenu,
    QShortcut,
    QToolButton,
)

from ..core.autofocus import (
    apply_phase_correction,
    phase_gradient_autofocus,
    select_pulse_with_strong_target,
)
from ..core.cropper import get_extend_image_coords
from ..core.looks import extract_centered_look
from ..core.metadata import MetadataProvider
from ..core.raster import read_slc_layer
from .toolbar_button_policy import ToolbarButtonPolicy

# --- Extent (meters) ---------------------------------------------------------------------------

MIN_EXTENT_M = 100.0
MAX_EXTENT_M = 400.0

_SLC_RENDER_MODES = frozenset(
    {
        "spectrum",
        "color",
        "2d_spectrum",
        "range_spectrum",
        "azimuth_viewer",
        "range_viewer",
    }
)


def compute_lens_extent(
    center: QPointF,
    layer: QgsRasterLayer | None,
    wheel_value: float,
    canvas,
    overlay_size: int,
) -> QgsRectangle:
    """Square extent in map units from center; size interpolates meters by wheel_value in [0, 1]."""
    size_m = MIN_EXTENT_M + float(wheel_value) * (MAX_EXTENT_M - MIN_EXTENT_M)
    half_m = size_m / 2.0

    if layer is not None and layer.isValid():
        crs = layer.crs()
    else:
        crs = canvas.mapSettings().destinationCrs()

    if crs.mapUnits() == Qgis.DistanceUnit.Degrees:
        lat_rad = math.radians(center.y())
        half_x = half_m / (111320.0 * max(math.cos(lat_rad), 1e-6))
        half_y = half_m / 110540.0
    else:
        half_x = half_y = half_m

    return QgsRectangle(
        center.x() - half_x,
        center.y() - half_y,
        center.x() + half_x,
        center.y() + half_y,
    )


class BoundedWheel:
    """Scalar in ``[lo, hi]``, adjusted in fixed steps (e.g. mouse wheel)."""

    def __init__(
        self,
        value: float = 0.5,
        *,
        step: float = 0.05,
        lo: float = 0.0,
        hi: float = 1.0,
    ) -> None:
        self._step = step
        self._lo = lo
        self._hi = hi
        self._value = max(lo, min(hi, float(value)))

    @property
    def value(self) -> float:
        """Current value clamped to ``[lo, hi]``."""
        return self._value

    def increment(self) -> bool:
        """Increase by one step; return True if the value changed."""
        old = self._value
        self._value = min(self._hi, self._value + self._step)
        return self._value != old

    def decrement(self) -> bool:
        """Decrease by one step; return True if the value changed."""
        old = self._value
        self._value = max(self._lo, self._value - self._step)
        return self._value != old

    def nudge_from_delta(self, delta: int) -> bool:
        """Apply wheel delta (typ. multiples of 120) as step(s). Returns True if value changed."""
        if delta == 0:
            return False
        steps = int(round(delta / 120.0))
        if steps == 0:
            steps = 1 if delta > 0 else -1
        changed = False
        for _ in range(abs(steps)):
            if delta > 0:
                changed = self.increment() or changed
            else:
                changed = self.decrement() or changed
        return changed


class LensOverlayItem(QGraphicsItemGroup):
    """Graphics item group for the lens overlay (pixmap + border)."""

    def __init__(self, size: int, parent=None) -> None:
        super().__init__(parent)

        self._pixmap_item = QGraphicsPixmapItem(self)
        self._pixmap_item.setOffset(0, 0)
        self.addToGroup(self._pixmap_item)

        self._border_item = QGraphicsRectItem(0, 0, size, size, self)
        pen = QPen(QColor("white"))
        pen.setWidth(1)
        self._border_item.setPen(pen)
        self._border_item.setBrush(QBrush(Qt.NoBrush))
        self.addToGroup(self._border_item)

        self.setZValue(10000)

    def set_image(self, image: QImage) -> None:
        """Set the overlay image from QImage."""
        if image is None:
            return
        self._pixmap_item.setPixmap(QPixmap.fromImage(image))


@dataclass
class LensSLCData:
    """SLC payload used by SLC-based lens modes."""

    extent: QgsRectangle
    layer: QgsRasterLayer
    metadata: Any
    data_patch: NDArray[np.complex64]
    geo_corners: list[tuple[float, float]] | None = None


def _normalize_to_uint8(data: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Normalize float image data to [0, 255] uint8."""
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    if max_val <= min_val:
        return np.zeros_like(data, dtype=np.uint8)
    scaled = (data - min_val) / (max_val - min_val)
    return (scaled * 255).astype(np.uint8)


def _as_uint8_georef(data: NDArray[Any]) -> NDArray[np.uint8]:
    """Coerce array to uint8 for GCP GeoTIFF writing."""
    if data.dtype == np.uint8:
        return np.ascontiguousarray(data)
    return _normalize_to_uint8(np.asarray(data, dtype=np.float32))


def _orient_slc_image(
    data: NDArray[Any], sar_observation_direction: str | None
) -> NDArray[Any]:
    """Apply left/right looking orientation transform."""
    if (sar_observation_direction or "").lower() == "left":
        return np.flipud(np.fliplr(data.T))
    return np.flipud(data.T)


def _to_scaled_grayscale(data: NDArray[np.uint8], overlay_size: int) -> QImage | None:
    """Convert uint8 2D array to scaled grayscale QImage."""
    if data.size == 0:
        return None
    normalized = np.ascontiguousarray(data)
    rows, cols = normalized.shape
    image = QImage(
        normalized.data,
        cols,
        rows,
        cols,
        QImage.Format_Grayscale8,
    ).copy()
    return image.scaled(
        overlay_size,
        overlay_size,
        Qt.IgnoreAspectRatio,
        Qt.SmoothTransformation,
    )


def get_pixel_to_geo_corners(
    layer: QgsRasterLayer, pixel_bounds: QgsRectangle
) -> list[tuple[float, float]]:
    """Transform pixel corners to geographic coordinates (GCP_TPS)."""
    source_path = layer.dataProvider().dataSourceUri()
    dataset = gdal.Open(source_path)
    t = gdal.Transformer(dataset, None, ["METHOD=GCP_TPS"])

    pixel_corners = [
        [pixel_bounds.xMinimum(), pixel_bounds.yMinimum()],
        [pixel_bounds.xMaximum(), pixel_bounds.yMinimum()],
        [pixel_bounds.xMaximum(), pixel_bounds.yMaximum()],
        [pixel_bounds.xMinimum(), pixel_bounds.yMaximum()],
    ]

    geo_corners = []
    for px, py in pixel_corners:
        success, point = t.TransformPoint(0, px, py, 0.0)
        if success:
            geo_corners.append((point[0], point[1]))

    dataset = None
    return geo_corners


def read_slc_data(
    layer: QgsRasterLayer,
    extent: QgsRectangle,
    metadata_provider: MetadataProvider,
    *,
    with_geo: bool = False,
) -> LensSLCData | None:
    """Read SLC patch from layer at extent."""
    if not isinstance(layer, QgsRasterLayer) or not layer.isValid():
        return None
    try:
        metadata = metadata_provider.get(layer)
        if metadata is None:
            return None
        left = (metadata.sar_observation_direction or "").lower() == "left"
        data_patch, _ = read_slc_layer(
            layer,
            left=left,
            metadata=metadata,
            extent=extent,
        )
        geo_corners = None
        if with_geo:
            pixel_bounds = get_extend_image_coords(layer, extent)
            geo_corners = get_pixel_to_geo_corners(layer, pixel_bounds)
        return LensSLCData(
            extent=extent,
            layer=layer,
            metadata=metadata,
            data_patch=data_patch,
            geo_corners=geo_corners,
        )
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to read SLC data: {e}", "ICEYE Toolbox", Qgis.Warning
        )
        return None


def create_georeferenced_temp_raster(
    data: NDArray[np.uint8],
    geo_corners: list[tuple[float, float]],
    crs_wkt: str,
) -> str | None:
    """Create a temporary GeoTIFF with GCP georeferencing."""
    if data.ndim == 2:
        height, width = data.shape
        num_bands = 1
    elif data.ndim == 3:
        height, width, num_bands = data.shape
    else:
        return None

    temp_path = f"/vsimem/lens_tool_{uuid.uuid4()}.tif"

    try:
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(temp_path, width, height, num_bands, gdal.GDT_Byte)

        if num_bands == 1:
            band = dataset.GetRasterBand(1)
            band.WriteArray(data)
            band.FlushCache()
        else:
            for i in range(num_bands):
                band = dataset.GetRasterBand(i + 1)
                band.WriteArray(data[:, :, i])
                band.FlushCache()

        gcps = [
            gdal.GCP(geo_corners[3][0], geo_corners[3][1], 0, 0, 0),
            gdal.GCP(geo_corners[2][0], geo_corners[2][1], 0, width, 0),
            gdal.GCP(geo_corners[1][0], geo_corners[1][1], 0, width, height),
            gdal.GCP(geo_corners[0][0], geo_corners[0][1], 0, 0, height),
        ]

        dataset.SetGCPs(gcps, crs_wkt)
        dataset = None

        return temp_path
    except Exception:
        Path(temp_path).unlink(missing_ok=True)
        return None


def render_layers_to_image(
    layers: list[QgsRasterLayer],
    extent: QgsRectangle,
    overlay_size: int,
    canvas,
) -> QImage:
    """QGIS parallel map render to QImage."""
    map_settings = canvas.mapSettings()
    map_settings.setLayers(layers)
    map_settings.setOutputSize(QSize(overlay_size, overlay_size))
    map_settings.setExtent(extent)
    map_settings.setBackgroundColor(QColor(0, 0, 0, 0))
    map_settings.setFlag(QgsMapSettings.Antialiasing, True)

    job = QgsMapRendererParallelJob(map_settings)
    job.start()
    job.waitForFinished()
    return job.renderedImage()


def render_georeferenced_data(
    data: NDArray[Any],
    geo_corners: list[tuple[float, float]],
    layer: QgsRasterLayer,
    extent: QgsRectangle,
    layer_name: str,
    overlay_size: int,
    canvas,
) -> QImage | None:
    """Write georeferenced uint8 (or normalized float) data and map-render to QImage."""
    data_u8 = _as_uint8_georef(data)
    temp_path = create_georeferenced_temp_raster(
        data_u8, geo_corners, layer.crs().toWkt()
    )
    if temp_path is None:
        return None
    temp_layer = QgsRasterLayer(temp_path, layer_name, "gdal")
    Path(temp_path).unlink(missing_ok=True)
    if not temp_layer.isValid():
        Path(temp_path).unlink(missing_ok=True)
        return None
    image = render_layers_to_image([temp_layer], extent, overlay_size, canvas)
    Path(temp_path).unlink(missing_ok=True)
    return image


def _process_focus_data(
    data_patch: NDArray[np.complex64], metadata: Any
) -> NDArray[np.uint8]:
    """Apply look extraction + phase correction and return display-ready image."""
    data_patch = np.fft.fftshift(np.fft.fft2(data_patch))

    rows, cols = data_patch.shape
    azimuth_look_size = max(1, int(rows * 0.15))
    look = extract_centered_look(
        data_patch,
        center_row=rows // 2,
        center_col=cols // 2,
        look_rows=azimuth_look_size,
        look_cols=cols,
        apply_ifftshift=True,
    )

    patch, _ = select_pulse_with_strong_target(look, axis=0)
    phase_error, _, _ = phase_gradient_autofocus(patch)
    corrected = apply_phase_correction(look, phase_error)

    magnitude = np.abs(corrected)

    thresh = np.mean(magnitude) + 4.0 * np.std(magnitude)
    magnitude[magnitude > thresh] = thresh
    if magnitude.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    normalized = _normalize_to_uint8(magnitude)
    return _orient_slc_image(
        normalized, getattr(metadata, "sar_observation_direction", "")
    )


def _process_color_spectrum(
    data_patch: NDArray[np.complex64], metadata: Any
) -> NDArray[np.uint8]:
    """Create RGB spectrum visualization from SLC patch."""
    spectrum = np.fft.fftshift(np.fft.fft(data_patch, axis=0), axes=0)
    rows, cols = spectrum.shape
    if rows < 2 or cols < 1:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    positions = np.linspace(0.0, 1.0, rows, dtype=np.float32)
    weight_r = (1.0 - positions)[:, None]
    weight_b = positions[:, None]
    weight_g = np.clip(1.0 - np.abs(positions - 0.5) * 2.0, 0.0, 1.0)[:, None]

    weighted_r = spectrum * weight_r
    weighted_g = spectrum * weight_g
    weighted_b = spectrum * weight_b

    mag_r = np.abs(np.fft.ifft(np.fft.ifftshift(weighted_r, axes=0), axis=0))
    mag_g = np.abs(np.fft.ifft(np.fft.ifftshift(weighted_g, axes=0), axis=0))
    mag_b = np.abs(np.fft.ifft(np.fft.ifftshift(weighted_b, axes=0), axis=0))

    rgb = np.dstack([mag_r, mag_g, mag_b])
    for channel in range(3):
        channel_data = rgb[:, :, channel]
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data)
        thresh = mean_val + 2.0 * std_val
        channel_data[channel_data > thresh] = thresh
    rgb = _normalize_to_uint8(rgb.astype(np.float32))

    if (getattr(metadata, "sar_observation_direction", "") or "").lower() == "left":
        rgb = np.transpose(rgb, (1, 0, 2))
        rgb = np.flip(rgb, (0, 1))
    else:
        rgb = np.transpose(rgb, (1, 0, 2))
        rgb = np.flip(rgb, axis=0)
    return np.ascontiguousarray(rgb)


class RenderMode(ABC):
    """Lens render strategy: full pipeline returns (QImage, is_georeferenced)."""

    mode_name: str
    render_delay_ms: ClassVar[int] = 60

    def __init__(self) -> None:
        self.render_delay_ms: int = max(0, int(type(self).render_delay_ms))

    @abstractmethod
    def render(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle,
        **kwargs: Any,
    ) -> tuple[QImage, bool] | None:
        """Return (image, is_georeferenced) or None. kwargs: metadata_provider, overlay_size, canvas."""

    def on_activated(self) -> None:
        """Run when this mode becomes active."""

    def on_deactivated(self) -> None:
        """Run when switching away from this mode."""

    def intercepts_wheel(self) -> bool:
        """If True, wheel adjusts mode state instead of extent wheel."""
        return False

    def handle_wheel(self, delta: int) -> bool:
        """Handle mouse wheel; return True if handled."""
        return False

    def handle_scroll(self, direction: int) -> bool:
        """Handle Shift+Up/Down; return True if handled."""
        return False


class NormalRenderMode(RenderMode):
    """Map-render the active raster extent into the lens square."""

    mode_name = "normal"

    def render(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle,
        **kwargs: Any,
    ) -> tuple[QImage, bool] | None:
        """Return a non-georeferenced map preview image."""
        canvas = kwargs.get("canvas")
        overlay_size = int(kwargs.get("overlay_size", 350))
        if canvas is None:
            return None
        layers: list[QgsRasterLayer] = (
            [layer] if layer is not None and layer.isValid() else []
        )
        img = render_layers_to_image(layers, extent, overlay_size, canvas)
        return (img, False)


class FocusRenderMode(RenderMode):
    """Phase-gradient autofocus spectrum view."""

    mode_name = "spectrum"

    def render(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle,
        **kwargs: Any,
    ) -> tuple[QImage, bool] | None:
        """Read SLC, process, and return a georeferenced-rendered image."""
        metadata_provider = kwargs.get("metadata_provider")
        canvas = kwargs.get("canvas")
        overlay_size = int(kwargs.get("overlay_size", 350))
        if metadata_provider is None or canvas is None:
            return None
        slc = read_slc_data(layer, extent, metadata_provider, with_geo=True)
        if slc is None or slc.data_patch.size == 0 or slc.geo_corners is None:
            return None
        normalized = _process_focus_data(slc.data_patch, slc.metadata)
        if normalized.size == 0:
            return None
        img = render_georeferenced_data(
            normalized,
            slc.geo_corners or [],
            slc.layer,
            slc.extent,
            "temp_focus",
            overlay_size,
            canvas,
        )
        if img is None:
            return None
        return (img, True)


class ColorRenderMode(RenderMode):
    """RGB azimuth-spectrum visualization."""

    mode_name = "color"

    def render(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle,
        **kwargs: Any,
    ) -> tuple[QImage, bool] | None:
        """Read SLC, build RGB spectrum, georeference-render."""
        metadata_provider = kwargs.get("metadata_provider")
        canvas = kwargs.get("canvas")
        overlay_size = int(kwargs.get("overlay_size", 350))
        if metadata_provider is None or canvas is None:
            return None
        slc = read_slc_data(layer, extent, metadata_provider, with_geo=True)
        if slc is None or slc.data_patch.size == 0 or slc.geo_corners is None:
            return None
        rgb = _process_color_spectrum(slc.data_patch, slc.metadata)
        if rgb.size == 0:
            return None
        img = render_georeferenced_data(
            rgb,
            slc.geo_corners or [],
            slc.layer,
            slc.extent,
            "temp_color",
            overlay_size,
            canvas,
        )
        if img is None:
            return None
        return (img, True)


class Spectrum2DRenderMode(RenderMode):
    """2D FFT log-magnitude view (screen-space)."""

    mode_name = "2d_spectrum"

    def render(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle,
        **kwargs: Any,
    ) -> tuple[QImage, bool] | None:
        """Read SLC and return a scaled grayscale spectrum image."""
        metadata_provider = kwargs.get("metadata_provider")
        overlay_size = int(kwargs.get("overlay_size", 350))
        if metadata_provider is None:
            return None
        slc = read_slc_data(layer, extent, metadata_provider, with_geo=False)
        if slc is None or slc.data_patch.size == 0:
            return None
        spectrum = np.fft.fftshift(np.fft.fft2(slc.data_patch))
        magnitude = np.abs(spectrum)
        log_data = np.log10(magnitude + 1e-10)
        u8 = _normalize_to_uint8(log_data.astype(np.float32))
        img = _to_scaled_grayscale(u8, overlay_size)
        if img is None:
            return None
        return (img, False)


class RangeSpectrumRenderMode(RenderMode):
    """Range FFT log-power spectrum (screen-space)."""

    mode_name = "range_spectrum"

    def render(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle,
        **kwargs: Any,
    ) -> tuple[QImage, bool] | None:
        """Read SLC and return a scaled grayscale range-spectrum image."""
        metadata_provider = kwargs.get("metadata_provider")
        overlay_size = int(kwargs.get("overlay_size", 350))
        if metadata_provider is None:
            return None
        slc = read_slc_data(layer, extent, metadata_provider, with_geo=False)
        if slc is None or slc.data_patch.size == 0:
            return None
        range_fft = np.fft.fftshift(np.fft.fft(slc.data_patch, axis=1), axes=1)
        power_2d = np.abs(range_fft) ** 2
        log_data = np.log10(power_2d + 1e-10)
        u8 = _normalize_to_uint8(log_data.astype(np.float32))
        img = _to_scaled_grayscale(u8, overlay_size)
        if img is None:
            return None
        return (img, False)


class AzimuthViewerRenderMode(RenderMode):
    """Sub-aperture viewer along azimuth (wheel moves window)."""

    mode_name = "azimuth_viewer"
    _look_frac: float = 0.15
    _step_frac: float = 0.05
    _margin_frac: float = 0.1

    def __init__(self) -> None:
        super().__init__()
        self._center_frac: float = 0.5

    def on_activated(self) -> None:
        """Reset sub-aperture center to mid track."""
        self._center_frac = 0.5

    def intercepts_wheel(self) -> bool:
        """Wheel adjusts sub-aperture, not lens extent."""
        return True

    def handle_wheel(self, delta: int) -> bool:
        """Move sub-aperture by one step per wheel direction."""
        return self.handle_scroll(int(np.sign(delta)))

    def handle_scroll(self, direction: int) -> bool:
        """Shift sub-aperture window; return True (always handled)."""
        self._center_frac = float(
            np.clip(
                self._center_frac + self._step_frac * direction,
                self._look_frac / 2 + self._margin_frac,
                1.0 - self._margin_frac - self._look_frac / 2,
            )
        )
        return True

    def render(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle,
        **kwargs: Any,
    ) -> tuple[QImage, bool] | None:
        """Read SLC, apply azimuth masking, georeference-render log power."""
        metadata_provider = kwargs.get("metadata_provider")
        canvas = kwargs.get("canvas")
        overlay_size = int(kwargs.get("overlay_size", 350))
        if metadata_provider is None or canvas is None:
            return None
        slc = read_slc_data(layer, extent, metadata_provider, with_geo=True)
        if slc is None or slc.data_patch.size == 0 or slc.geo_corners is None:
            return None
        rows, _ = slc.data_patch.shape
        used_rows = int(rows * (1.0 - self._margin_frac))
        az_spectrum = np.fft.fftshift(np.fft.fft(slc.data_patch, axis=0), axes=0)

        mask = np.zeros(rows, dtype=np.float32)
        start = max(
            int(rows * self._margin_frac),
            int(used_rows * (self._center_frac - self._look_frac / 2)),
        )
        end = min(used_rows, start + int(used_rows * self._look_frac))

        mask[start:end] = 1.0

        masked = az_spectrum * mask[:, None]
        output = np.fft.ifft(np.fft.ifftshift(masked, axes=0), axis=0)

        output = _orient_slc_image(output, slc.metadata.sar_observation_direction)
        normalized = np.ascontiguousarray(np.abs(output))
        data = 10.0 * np.log10(normalized**2 + 1e-30)
        img = render_georeferenced_data(
            data,
            slc.geo_corners or [],
            slc.layer,
            slc.extent,
            "temp_azimuth",
            overlay_size,
            canvas,
        )
        if img is None:
            return None
        return (img, True)


class RangeViewerRenderMode(RenderMode):
    """Sub-aperture viewer along range (wheel moves window)."""

    mode_name = "range_viewer"
    _look_frac: float = 0.15
    _step_frac: float = 0.05
    _margin_frac: float = 0.1

    def __init__(self) -> None:
        super().__init__()
        self._center_frac: float = 0.5

    def on_activated(self) -> None:
        """Reset sub-aperture center to mid track."""
        self._center_frac = 0.5

    def intercepts_wheel(self) -> bool:
        """Wheel adjusts sub-aperture, not lens extent."""
        return True

    def handle_wheel(self, delta: int) -> bool:
        """Move sub-aperture by one step per wheel direction."""
        return self.handle_scroll(int(np.sign(delta)))

    def handle_scroll(self, direction: int) -> bool:
        """Shift sub-aperture window; return True (always handled)."""
        self._center_frac = float(
            np.clip(
                self._center_frac + self._step_frac * direction,
                self._look_frac / 2 + self._margin_frac,
                1.0 - self._margin_frac - self._look_frac / 2,
            )
        )
        return True

    def render(
        self,
        layer: QgsRasterLayer,
        extent: QgsRectangle,
        **kwargs: Any,
    ) -> tuple[QImage, bool] | None:
        """Read SLC, apply range masking, georeference-render magnitude."""
        metadata_provider = kwargs.get("metadata_provider")
        canvas = kwargs.get("canvas")
        overlay_size = int(kwargs.get("overlay_size", 350))
        if metadata_provider is None or canvas is None:
            return None
        slc = read_slc_data(layer, extent, metadata_provider, with_geo=True)
        if slc is None or slc.data_patch.size == 0 or slc.geo_corners is None:
            return None
        _, cols = slc.data_patch.shape
        used_cols = int(cols * (1.0 - self._margin_frac))
        rg_spectrum = np.fft.fftshift(np.fft.fft(slc.data_patch, axis=1), axes=1)

        mask = np.zeros(cols, dtype=np.float32)
        start = max(
            int(cols * self._margin_frac),
            int(used_cols * (self._center_frac - self._look_frac / 2)),
        )
        end = min(used_cols, start + int(used_cols * self._look_frac))
        mask[start:end] = 1.0

        masked = rg_spectrum * mask[np.newaxis, :]
        output = np.fft.ifft(np.fft.ifftshift(masked, axes=1), axis=1)

        output = _orient_slc_image(output, slc.metadata.sar_observation_direction)
        img = render_georeferenced_data(
            np.abs(output),
            slc.geo_corners or [],
            slc.layer,
            slc.extent,
            "temp_range_viewer",
            overlay_size,
            canvas,
        )
        if img is None:
            return None
        return (img, True)


class LensMapTool(QgsMapToolPan):
    """Map tool for interactive lens overlay with multiple render modes."""

    deactivated = pyqtSignal()

    def __init__(
        self,
        iface,
        *,
        metadata_provider: MetadataProvider | None = None,
        overlay_size: int = 350,
    ) -> None:
        super().__init__(iface.mapCanvas())
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.metadata_provider = metadata_provider or MetadataProvider()
        self.overlay_size = overlay_size
        self._wheel = BoundedWheel(value=0.5)
        self._layer: QgsRasterLayer | None = self._coerce_raster_layer(
            self.iface.activeLayer()
        )
        self._extent: QgsRectangle | None = None
        self._overlay: LensOverlayItem | None = None
        self._extent_band: QgsRubberBand | None = None
        self._last_map_point: QPointF | None = None
        self._last_pos: QPoint | None = None
        self._press_pos: QPoint | None = None
        self._pinned = False
        self._pinned_pos: QPoint | None = None
        self._pinned_map_point: QPointF | None = None
        self._offset = QPoint(20, 20)
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._update_overlay)
        self.canvas.extentsChanged.connect(self._on_canvas_extent_changed)
        self._render_mode = "normal"
        self._modes: dict[str, RenderMode] = {
            "normal": NormalRenderMode(),
            "spectrum": FocusRenderMode(),
            "color": ColorRenderMode(),
            "2d_spectrum": Spectrum2DRenderMode(),
            "range_spectrum": RangeSpectrumRenderMode(),
            "azimuth_viewer": AzimuthViewerRenderMode(),
            "range_viewer": RangeViewerRenderMode(),
        }
        self._render_modes = set(self._modes.keys())
        self._active_mode: RenderMode = self._modes[self._render_mode]

    @staticmethod
    def _coerce_raster_layer(layer: QgsMapLayer | None) -> QgsRasterLayer | None:
        if isinstance(layer, QgsRasterLayer) and layer.isValid():
            return layer
        return None

    def _on_current_layer_changed(self, layer: QgsMapLayer | None) -> None:
        self._layer = self._coerce_raster_layer(layer)
        self._recompute_extent()
        self._schedule_render()

    def activate(self) -> None:
        """Activate the lens tool and show overlay."""
        super().activate()
        self.iface.currentLayerChanged.connect(self._on_current_layer_changed)
        self._layer = self._coerce_raster_layer(self.iface.activeLayer())
        if self._overlay is None:
            self._overlay = LensOverlayItem(self.overlay_size)
            self.canvas.scene().addItem(self._overlay)
        if self._extent_band is None:
            self._extent_band = QgsRubberBand(self.canvas, QgsWkbTypes.PolygonGeometry)
            self._extent_band.setColor(QColor(255, 255, 255, 200))
            self._extent_band.setFillColor(QColor(0, 0, 0, 0))
            self._extent_band.setWidth(1)
            self._extent_band.setLineStyle(Qt.DashLine)
            self._extent_band.setZValue(9999)
        self._overlay.show()
        if self._extent_band is not None:
            self._extent_band.show()
        self.canvas.setCursor(Qt.CrossCursor)

        self._shortcut_scroll_fwd = QShortcut(QKeySequence("Shift+Up"), self.canvas)
        self._shortcut_scroll_fwd.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_scroll_fwd.activated.connect(self._scroll_forward)

        self._shortcut_scroll_bwd = QShortcut(QKeySequence("Shift+Down"), self.canvas)
        self._shortcut_scroll_bwd.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_scroll_bwd.activated.connect(self._scroll_backward)

    def deactivate(self) -> None:
        """Deactivate the lens tool and remove overlay."""
        try:
            self.iface.currentLayerChanged.disconnect(self._on_current_layer_changed)
        except (TypeError, RuntimeError):
            pass
        if self._render_timer.isActive():
            self._render_timer.stop()
        for shortcut in ("_shortcut_scroll_fwd", "_shortcut_scroll_bwd"):
            sc = getattr(self, shortcut, None)
            if sc is not None:
                sc.deleteLater()
                setattr(self, shortcut, None)
        if self._overlay is not None:
            self.canvas.scene().removeItem(self._overlay)
            self._overlay = None
        if self._extent_band is not None:
            self._extent_band.reset(QgsWkbTypes.PolygonGeometry)
            self._extent_band.hide()
            self._extent_band = None

        self._pinned = False
        self._pinned_pos = None
        self._pinned_map_point = None

        self.deactivated.emit()
        super().deactivate()

    def canvasMoveEvent(self, event):
        """Handle canvas mouse move."""
        super().canvasMoveEvent(event)
        if self._pinned:
            return
        self._last_map_point = event.mapPoint()
        self._last_pos = event.pos()
        self._recompute_extent()
        self._update_position(self._last_pos)
        self._update_extent_band()
        self._schedule_render()

    def canvasPressEvent(self, event):
        """Handle canvas mouse press."""
        self._press_pos = event.pos()
        super().canvasPressEvent(event)

    def canvasReleaseEvent(self, event):
        """Handle canvas mouse release."""
        super().canvasReleaseEvent(event)
        if event.button() != Qt.LeftButton or self._press_pos is None:
            return
        moved = (event.pos() - self._press_pos).manhattanLength()
        self._press_pos = None
        if moved > 3:
            return
        if self._pinned:
            self._pinned = False
            self._pinned_pos = None
            self._pinned_map_point = None
            self._last_map_point = event.mapPoint()
            self._last_pos = event.pos()
            self._recompute_extent()
            self._update_position(self._last_pos)
            self._update_extent_band()
            self._schedule_render()
            return

        self._pinned = True
        self._pinned_map_point = event.mapPoint()
        self._pinned_pos = self._pos_from_map_point(self._pinned_map_point)
        if self._pinned_pos is not None:
            self._update_position(self._pinned_pos, clamp=False)
        self._recompute_extent()
        self._update_extent_band()
        self._schedule_render()

    def wheelEvent(self, event):
        """Handle mouse wheel for extent size or sub-aperture scroll."""
        delta = event.angleDelta().y()
        if delta == 0:
            return

        if self._active_mode.intercepts_wheel() and self._active_mode.handle_wheel(
            delta
        ):
            event.accept()
            self._schedule_render()
            return

        if self._wheel.nudge_from_delta(delta):
            self._recompute_extent()
        event.accept()
        self._schedule_render()

    def keyPressEvent(self, event):
        """Handle key press for extent wheel (+/-)."""
        key = event.key()
        if key in (Qt.Key_Plus, Qt.Key_Equal):
            delta = 120
        elif key == Qt.Key_Minus:
            delta = -120
        else:
            super().keyPressEvent(event)
            return
        if self._active_mode.intercepts_wheel():
            super().keyPressEvent(event)
            return
        if self._wheel.nudge_from_delta(delta):
            self._recompute_extent()
        event.accept()
        self._schedule_render()

    def _scroll_forward(self) -> None:
        if self._active_mode.handle_scroll(1):
            self._schedule_render()

    def _scroll_backward(self) -> None:
        if self._active_mode.handle_scroll(-1):
            self._schedule_render()

    def _recompute_extent(self) -> None:
        if self._pinned:
            center = self._pinned_map_point
        else:
            center = self._last_map_point
        if center is None:
            self._extent = None
            return
        self._extent = compute_lens_extent(
            center,
            self._layer,
            self._wheel.value,
            self.canvas,
            self.overlay_size,
        )

    def _schedule_render(self) -> None:
        if self._render_timer.isActive():
            self._render_timer.stop()
        delay = self._active_mode.render_delay_ms
        if delay == 0:
            self._update_overlay()
            return
        self._render_timer.start(delay)

    def _render_kwargs(self) -> dict[str, Any]:
        return {
            "overlay_size": self.overlay_size,
            "metadata_provider": self.metadata_provider,
            "canvas": self.canvas,
        }

    def _do_render(self) -> tuple[QImage, bool] | None:
        if self._extent is None:
            return None
        layer = self._layer
        if layer is None or not layer.isValid():
            layer = self._coerce_raster_layer(self.iface.activeLayer())
        if layer is None or not layer.isValid():
            return None
        return self._active_mode.render(layer, self._extent, **self._render_kwargs())

    def _update_overlay(self) -> None:
        if (
            self._overlay is None
            or (self._pinned is False and self._last_pos is None)
            or (self._pinned and self._pinned_pos is None)
        ):
            return
        if self._pinned:
            map_point = self._pinned_map_point
            pos = self._pos_from_map_point(map_point)
        else:
            pos = self._last_pos
            map_point = self._last_map_point or self._map_point_from_pos(pos)
        if pos is None or map_point is None:
            return
        self._update_position(pos, clamp=not self._pinned)

        self._recompute_extent()
        self._update_extent_band()

        result = self._do_render()
        if result is None:
            return
        image = result[0]
        if image is not None:
            self._overlay.set_image(image)

    def _update_position(self, pos: QPoint, *, clamp: bool = True) -> None:
        if self._overlay is None:
            return
        canvas_size = self.canvas.size()
        x = pos.x() + self._offset.x()
        y = pos.y() + self._offset.y()

        if clamp:
            if x + self.overlay_size > canvas_size.width():
                x = pos.x() - self.overlay_size - self._offset.x()
            if y + self.overlay_size > canvas_size.height():
                y = pos.y() - self.overlay_size - self._offset.y()

            x = max(0, min(x, canvas_size.width() - self.overlay_size))
            y = max(0, min(y, canvas_size.height() - self.overlay_size))

        self._overlay.setPos(QPointF(x, y))

    def _map_point_from_pos(self, pos: QPoint | None):
        if pos is None:
            return None
        return self.canvas.getCoordinateTransform().toMapCoordinates(pos)

    def _pos_from_map_point(self, map_point):
        if map_point is None:
            return None
        pixel = self.canvas.getCoordinateTransform().transform(map_point)
        return QPoint(int(pixel.x()), int(pixel.y()))

    def _on_canvas_extent_changed(self):
        if not self._pinned:
            return
        self._schedule_render()

    def _update_extent_band(self) -> None:
        if self._extent_band is None or self._extent is None:
            return
        geometry = QgsGeometry.fromRect(self._extent)
        self._extent_band.setToGeometry(geometry, None)

    def set_render_mode(self, mode: str) -> None:
        """Set render mode (normal, spectrum, color, 2d_spectrum)."""
        mode = (mode or "").strip().lower()
        if mode not in self._render_modes:
            mode = "normal"
        if mode == self._render_mode:
            return
        new_mode = self._modes.get(mode, self._modes["normal"])
        self._active_mode.on_deactivated()
        self._render_mode = mode
        self._active_mode = new_mode
        self._active_mode.on_activated()
        self._schedule_render()

    def render_mode(self) -> str:
        """Return current render mode."""
        return self._render_mode

    def active_mode_uses_slc(self) -> bool:
        """Return True if the active render mode reads SLC patch data."""
        return self._render_mode in _SLC_RENDER_MODES


def _tr(message: str) -> str:
    """Translate message for ICEYE Toolbox context."""
    return QCoreApplication.translate("ICEYE Toolbox", message)


class LensToolbarAction:
    """Encapsulates the lens toolbar, toggle action, and render mode actions."""

    def __init__(
        self,
        iface,
        metadata_provider: MetadataProvider | None = None,
        toolbar_button_policy: ToolbarButtonPolicy | None = None,
    ) -> None:
        self.iface = iface
        self.metadata_provider = metadata_provider or MetadataProvider()
        self._toolbar_button_policy = toolbar_button_policy
        self.lens_tool = LensMapTool(
            self.iface, metadata_provider=self.metadata_provider
        )
        self.lens_toolbar = None
        self.lens_action = None
        self.lens_render_actions: dict[str, QAction] = {}
        self._suppress_lens_toggle = False
        self._crop_toolbar_action = None
        self._current_spectrum_mode = "2d_spectrum"
        self._current_viewer_mode = "azimuth_viewer"

    def setup(self) -> None:
        """Create the toolbar and actions, connect signals."""
        self.lens_toolbar = self.iface.addToolBar("ICEYE Lens")
        self.lens_toolbar.setObjectName("ICEYE Lens")

        self.lens_action = QAction(
            QIcon(":/plugins/iceye_toolbox/lens-svgrepo-com.svg"),
            _tr("Lens"),
            self.iface.mainWindow(),
        )
        self.lens_action.setCheckable(True)
        self.lens_action.setStatusTip("Toggle lens tool")
        self.lens_action.toggled.connect(self._toggle_lens)
        self.lens_toolbar.addAction(self.lens_action)
        self.lens_tool.deactivated.connect(self._on_lens_tool_deactivated)

        normal_action = QAction(
            QIcon(":/plugins/iceye_toolbox/grayscale-svgrepo-com.svg"),
            _tr("Lens Render: Normal"),
            self.iface.mainWindow(),
        )
        normal_action.setStatusTip("Lens render mode: normal")
        normal_action.triggered.connect(lambda: self._set_lens_render_mode("normal"))
        self.lens_toolbar.addAction(normal_action)
        self.lens_render_actions["normal"] = normal_action

        spectrum_action = QAction(
            QIcon(
                ":/plugins/iceye_toolbox/focus-horizontal-round-round-840-svgrepo-com.svg"
            ),
            _tr("Lens Render: Focus"),
            self.iface.mainWindow(),
        )
        spectrum_action.setStatusTip("Lens render mode: focus")
        spectrum_action.triggered.connect(
            lambda: self._set_lens_render_mode("spectrum")
        )
        self.lens_toolbar.addAction(spectrum_action)
        self.lens_render_actions["spectrum"] = spectrum_action

        spectrum_2d_action = QAction(
            QIcon(":/plugins/iceye_toolbox/spectrum-svgrepo-com.svg"),
            _tr("2D Spectrum"),
            self.iface.mainWindow(),
        )
        spectrum_2d_action.setCheckable(True)
        spectrum_2d_action.triggered.connect(
            lambda: self._set_lens_render_mode("2d_spectrum")
        )

        range_spectrum_action = QAction(
            QIcon(":/plugins/iceye_toolbox/spectrum-svgrepo-com.svg"),
            _tr("Range Power Spectrum"),
            self.iface.mainWindow(),
        )
        range_spectrum_action.setCheckable(True)
        range_spectrum_action.triggered.connect(
            lambda: self._set_lens_render_mode("range_spectrum")
        )

        spectrum_menu = QMenu()
        spectrum_menu.addAction(spectrum_2d_action)
        spectrum_menu.addAction(range_spectrum_action)

        self._spectrum_dropdown_btn = QToolButton()
        self._spectrum_dropdown_btn.setIcon(
            QIcon(":/plugins/iceye_toolbox/spectrum-svgrepo-com.svg")
        )
        self._spectrum_dropdown_btn.setText(_tr("Lens Render: 2D Spectrum"))
        self._spectrum_dropdown_btn.setToolTip(_tr("Spectrum mode"))
        self._spectrum_dropdown_btn.setCheckable(True)
        self._spectrum_dropdown_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._spectrum_dropdown_btn.setMenu(spectrum_menu)
        self._spectrum_dropdown_btn.clicked.connect(
            lambda: self._set_lens_render_mode(self._current_spectrum_mode)
        )

        self.lens_toolbar.addWidget(self._spectrum_dropdown_btn)

        self.lens_render_actions["2d_spectrum"] = spectrum_2d_action
        self.lens_render_actions["range_spectrum"] = range_spectrum_action

        color_action = QAction(
            QIcon(":/plugins/iceye_toolbox/rgb-svgrepo-com.svg"),
            _tr("Lens Render: Color"),
            self.iface.mainWindow(),
        )
        color_action.setStatusTip("Lens render mode: color")
        color_action.triggered.connect(lambda: self._set_lens_render_mode("color"))
        self.lens_toolbar.addAction(color_action)
        self.lens_render_actions["color"] = color_action

        azimuth_viewer_action = QAction(
            QIcon(":/plugins/iceye_toolbox/subaperture-viewer.svg"),
            _tr("Azimuth Sub-aperture Viewer"),
            self.iface.mainWindow(),
        )
        azimuth_viewer_action.setCheckable(True)
        azimuth_viewer_action.triggered.connect(
            lambda: self._set_lens_render_mode("azimuth_viewer")
        )

        range_viewer_action = QAction(
            QIcon(":/plugins/iceye_toolbox/subaperture-viewer.svg"),
            _tr("Range Sub-aperture Viewer"),
            self.iface.mainWindow(),
        )
        range_viewer_action.setCheckable(True)
        range_viewer_action.triggered.connect(
            lambda: self._set_lens_render_mode("range_viewer")
        )

        viewer_menu = QMenu()
        viewer_menu.addAction(azimuth_viewer_action)
        viewer_menu.addAction(range_viewer_action)

        self._viewer_dropdown_btn = QToolButton()
        self._viewer_dropdown_btn.setIcon(
            QIcon(":/plugins/iceye_toolbox/subaperture-viewer.svg")
        )
        self._viewer_dropdown_btn.setText(_tr("Lens Render: Azimuth Viewer"))
        self._viewer_dropdown_btn.setToolTip(_tr("Sub-aperture viewer mode"))
        self._viewer_dropdown_btn.setCheckable(True)
        self._viewer_dropdown_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._viewer_dropdown_btn.setMenu(viewer_menu)
        self._viewer_dropdown_btn.clicked.connect(
            lambda: self._set_lens_render_mode(self._current_viewer_mode)
        )

        self.lens_toolbar.addWidget(self._viewer_dropdown_btn)

        self.lens_render_actions["azimuth_viewer"] = azimuth_viewer_action
        self.lens_render_actions["range_viewer"] = range_viewer_action

        normal_action.setCheckable(True)
        spectrum_action.setCheckable(True)
        spectrum_2d_action.setCheckable(True)
        color_action.setCheckable(True)

        self.lens_render_mode_group = QActionGroup(self.iface.mainWindow())
        self.lens_render_mode_group.setExclusive(True)
        for action in self.lens_render_actions.values():
            self.lens_render_mode_group.addAction(action)

        self._set_lens_render_mode("normal")

        for action in self.lens_render_actions.values():
            action.setEnabled(False)
        self._spectrum_dropdown_btn.setEnabled(False)
        self._viewer_dropdown_btn.setEnabled(False)

        if self._toolbar_button_policy is not None:
            self._toolbar_button_policy.register(
                self.lens_action,
                self._policy_lens_toggle,
                on_disable=self.deactivate,
            )
            self._toolbar_button_policy.register(
                self.lens_render_actions["normal"],
                self._policy_lens_normal,
                on_disable=self._on_normal_lens_policy_disabled,
            )
            for key in (
                "spectrum",
                "2d_spectrum",
                "range_spectrum",
                "color",
                "azimuth_viewer",
                "range_viewer",
            ):
                self._toolbar_button_policy.register(
                    self.lens_render_actions[key],
                    self._policy_lens_slc,
                    on_disable=self._on_slc_lens_policy_disabled,
                )
            self._toolbar_button_policy.register(
                self._spectrum_dropdown_btn,
                self._policy_lens_slc,
                on_disable=self._on_slc_lens_policy_disabled,
            )
            self._toolbar_button_policy.register(
                self._viewer_dropdown_btn,
                self._policy_lens_slc,
                on_disable=self._on_slc_lens_policy_disabled,
            )
            self._toolbar_button_policy.refresh()

    def _policy_lens_toggle(self, layer: QgsMapLayer | None) -> bool:
        """Return True if the layer has ICEYE metadata (Lens tool enabled)."""
        if self._toolbar_button_policy is None:
            return True
        return self._toolbar_button_policy.enabled_if_iceye_layer(layer)

    def _policy_lens_normal(self, layer: QgsMapLayer | None) -> bool:
        """Return True if the lens session is active and the layer has ICEYE metadata."""
        if self.lens_action is None or not self.lens_action.isChecked():
            return False
        if self._toolbar_button_policy is None:
            return True
        return self._toolbar_button_policy.enabled_if_iceye_layer(layer)

    def _policy_lens_slc(self, layer: QgsMapLayer | None) -> bool:
        """Focus / spectrum / color / viewers: SLC-COG while lens session is active."""
        if self.lens_action is None or not self.lens_action.isChecked():
            return False
        if self._toolbar_button_policy is None:
            return False
        return self._toolbar_button_policy.enabled_if_slc_cog(layer)

    def _on_normal_lens_policy_disabled(self) -> None:
        if self.lens_tool.render_mode() == "normal":
            self._set_lens_render_mode("normal")

    def _on_slc_lens_policy_disabled(self) -> None:
        m = self.lens_tool.render_mode()
        if self.lens_tool.active_mode_uses_slc():
            self._set_lens_render_mode(m)

    def unload(self) -> None:
        """Remove toolbar and actions, deactivate lens tool."""
        if self.lens_action is not None:
            try:
                if self.lens_toolbar is not None:
                    self.lens_toolbar.removeAction(self.lens_action)
            except Exception:
                pass
            self.lens_action = None

        if self.lens_tool is not None:
            try:
                self.lens_tool.deactivate()
            except Exception:
                pass

        if self.lens_toolbar is not None:
            try:
                del self.lens_toolbar
            except Exception:
                pass
            self.lens_toolbar = None

    def deactivate(self) -> None:
        """Deactivate lens tool."""
        if self.lens_action is not None and self.lens_action.isChecked():
            self.lens_action.setChecked(False)

    def _toggle_lens(self, enabled: bool) -> None:
        if self._suppress_lens_toggle:
            return
        canvas = self.iface.mapCanvas()

        if enabled:
            if (
                self._crop_toolbar_action is not None
                and canvas.mapTool() is self._crop_toolbar_action.map_tool
            ):
                self._crop_toolbar_action.deactivate()

            canvas.setMapTool(self.lens_tool)

            if self._toolbar_button_policy is not None:
                self._toolbar_button_policy.refresh()
            else:
                for action in self.lens_render_actions.values():
                    action.setEnabled(True)
                self._spectrum_dropdown_btn.setEnabled(True)
                self._viewer_dropdown_btn.setEnabled(True)
            self._set_lens_render_mode("normal")
            return

        if self._toolbar_button_policy is None:
            for action in self.lens_render_actions.values():
                action.setEnabled(False)
            self._spectrum_dropdown_btn.setEnabled(False)
            self._viewer_dropdown_btn.setEnabled(False)
        else:
            self._toolbar_button_policy.refresh()

        canvas.unsetMapTool(self.lens_tool)
        self.iface.actionPan().trigger()

        try:
            self.lens_tool.deactivate()
        except Exception:
            pass

    def _on_lens_tool_deactivated(self) -> None:
        if self.lens_action is None or not self.lens_action.isChecked():
            return
        self._suppress_lens_toggle = True
        self.lens_action.blockSignals(True)
        self.lens_action.setChecked(False)
        self.lens_action.blockSignals(False)
        self._suppress_lens_toggle = False
        for action in self.lens_render_actions.values():
            action.setEnabled(False)
        self._spectrum_dropdown_btn.setEnabled(False)
        self._viewer_dropdown_btn.setEnabled(False)

        if self._toolbar_button_policy is not None:
            self._toolbar_button_policy.refresh()

    def _set_lens_render_mode(self, mode: str) -> None:
        self.lens_tool.set_render_mode(mode)

        if mode in ("2d_spectrum", "range_spectrum"):
            self._current_spectrum_mode = mode
            self._spectrum_dropdown_btn.setChecked(True)
        else:
            self._spectrum_dropdown_btn.setChecked(False)

        if mode in ("azimuth_viewer", "range_viewer"):
            self._current_viewer_mode = mode
            self._viewer_dropdown_btn.setChecked(True)
        else:
            self._viewer_dropdown_btn.setChecked(False)

        for mode_name, action in self.lens_render_actions.items():
            action.blockSignals(True)
            action.setChecked(mode_name == mode)
            action.blockSignals(False)
