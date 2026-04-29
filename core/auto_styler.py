"""Auto-styling for ICEYE layers and tone curve colormaps."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from qgis.core import (
    Qgis,
    QgsColorRamp,
    QgsColorRampShader,
    QgsGradientColorRamp,
    QgsGradientStop,
    QgsMessageLog,
    QgsProject,
    QgsRaster,
    QgsRasterDataProvider,
    QgsRasterLayer,
    QgsRasterShader,
    QgsStyle,
)
from qgis.PyQt.QtCore import QObject
from qgis.PyQt.QtGui import QColor

from .metadata import is_iceye_layer
from .temporal_properties import (
    apply_temporal_properties,
    apply_temporal_properties_for_frames,
)


def _find_alpha_band(layer: QgsRasterLayer) -> int | None:
    """Find band index with AlphaBand color interpretation, or None if not found.

    Parameters
    ----------
    layer : QgsRasterLayer
        Raster layer to check.

    Returns
    -------
    int or None
        Band index (1-based) or None if no alpha band.
    """
    if not isinstance(layer, QgsRasterLayer):
        return None
    p = layer.dataProvider()
    if not isinstance(p, QgsRasterDataProvider):
        return None
    for band_idx in range(1, layer.bandCount() + 1):
        if p.colorInterpretation(band_idx) == QgsRaster.AlphaBand:
            return band_idx
    return None


class AutoStyler(QObject):
    """Apply default styles and temporal properties to ICEYE layers on add."""

    DEFAULT_QML = ":/plugins/iceye_toolbox/styles/iceye_default.qml"

    def __init__(self, iface, metadata_provider=None):
        super().__init__()

        self.canvas = iface.mapCanvas()
        self.metadata_provider = metadata_provider

        # self.canvas.extentsChanged.connect(self.on_canvas_updated)
        QgsProject.instance().layerWasAdded.connect(self.on_layer_was_added)

    def on_layer_was_added(self, layer):
        """Apply default style and temporal properties when an ICEYE layer is added."""
        if not is_iceye_layer(layer):
            return

        if "COLOR" in layer.name() or "CSI" in layer.name():
            return

        # Load and apply default style (QLK has its own styling)
        if "QLK" not in layer.name():
            msg, ok = layer.loadNamedStyle(self.DEFAULT_QML)
            if ok:
                # Apply alpha band for transparency when present (new product behavior)
                alpha_band = _find_alpha_band(layer)
                if alpha_band is not None:
                    renderer = layer.renderer()
                    if renderer is not None:
                        renderer.setAlphaBand(alpha_band)
                layer.triggerRepaint()
            else:
                QgsMessageLog.logMessage(
                    f"Failed to style {layer.name()}: {msg}",
                    "ICEYE Toolbox",
                    Qgis.Warning,
                )

        # Apply temporal properties to all ICEYE products including QLK (skip shorts - handled in VideoProcessingTask)
        if self.metadata_provider:
            metadata = self.metadata_provider.get(layer)
            if metadata:
                if "VID" in layer.name() or "SHORT" in layer.name():
                    apply_temporal_properties_for_frames(layer, metadata)
                elif "FOCUS" in layer.name() and layer.bandCount() > 2:
                    apply_temporal_properties_for_frames(layer, metadata)
                else:
                    # SLC, CSI, QLK, and others: fixed temporal range per layer
                    apply_temporal_properties(layer, metadata)


def build_shader_from_tonemap(tonemap: NDArray[np.float64]) -> QgsRasterShader:
    """Build a grayscale raster shader from a tone curve.

    Parameters
    ----------
    tonemap : ndarray of float64
        Array of values in [0, 1] (tone-mapped positions).

    Returns
    -------
    QgsRasterShader
        Shader with ColorRampItems at tone-mapped positions.

    Raises
    ------
    ValueError
        If tonemap values are outside [0, 1].
    """
    QgsMessageLog.logMessage(f"{tonemap}", "ICEYE Toolbox", Qgis.Info)
    if np.max(tonemap) > 1.0 or np.min(tonemap) < 0.0:
        raise ValueError("tonemap must be between 0 and 1")

    items = []
    g = np.floor(255.0 * tonemap).astype(np.uint8)
    QgsMessageLog.logMessage(f"{g}", "ICEYE Toolbox", Qgis.Info)

    for v, c in zip(tonemap, g):
        items.append(QgsColorRampShader.ColorRampItem(v, QColor(c, c, c)))

    ramp = QgsColorRampShader()
    ramp.setColorRampType(QgsColorRampShader.Interpolated)
    ramp.setColorRampItemList(items)

    shader = QgsRasterShader()
    shader.setRasterShaderFunction(ramp)
    return shader


def build_shader_from_color_ramp(
    color_ramp: QgsColorRamp,
    ramp_type: int = QgsColorRampShader.Interpolated,
    classification_mode: int = QgsColorRampShader.Continuous,
) -> QgsColorRampShader:
    """Build a QgsColorRampShader from a QgsColorRamp.

    Parameters
    ----------
    color_ramp : QgsColorRamp
        Color ramp (e.g. gradient) to sample.
    ramp_type : int, optional
        QgsColorRampShader ramp type. Default is Interpolated.
    classification_mode : int, optional
        QgsColorRampShader classification mode. Default is Continuous.

    Returns
    -------
    QgsColorRampShader
        Shader with ColorRampItems sampled from the ramp.
    """
    color_ramp_shader = QgsColorRampShader()
    items = []
    for i in range(color_ramp.count):
        color = color_ramp.color(color_ramp.value(i))
        value = float(i) / (color_ramp.count - 1)
        QgsMessageLog.logMessage(f"value {value} color {color.red()}")
        item = QgsColorRampShader.ColorRampItem(value, color)
        items.append(item)

    color_ramp_shader.setColorRampItemList(items)
    color_ramp_shader.setColorRampType(ramp_type)
    color_ramp_shader.setClassificationMode(classification_mode)

    return color_ramp_shader


def build_color_ramp_from_tone_curve(
    tone_curve: NDArray[np.float64],
    color1: QColor,
    color2: QColor,
) -> QgsGradientColorRamp:
    """
    Build a QgsGradientColorRamp with stops based on a tone curve.

    The tone curve controls how colors are distributed across the gradient.
    Each element in the tone curve determines the color blend at that position.

    Parameters
    ----------
    tone_curve : ndarray of float64
        Array of values in [0, 1] representing the tone mapping.
        Length determines the number of stops in the gradient.
    color1 : QColor
        Start color (at position 0)
    color2 : QColor
        End color (at position 1)

    Returns
    -------
    QgsGradientColorRamp
        A gradient color ramp with intermediate stops based on the tone curve
    """
    n = len(tone_curve)
    if n < 2:
        raise ValueError("tone_curve must have at least 2 elements")

    # Create stops for intermediate positions (excluding 0 and 1 which are color1 and color2)
    stops = []
    for i in range(1, n - 1):
        offset = float(i) / (n - 1)  # Position in the gradient [0, 1]
        t = float(tone_curve[i])  # Tone mapped value determines color blend

        # Interpolate color between color1 and color2 based on tone curve value
        r = int(color1.red() + t * (color2.red() - color1.red()))
        g = int(color1.green() + t * (color2.green() - color1.green()))
        b = int(color1.blue() + t * (color2.blue() - color1.blue()))
        a = int(color1.alpha() + t * (color2.alpha() - color1.alpha()))

        color = QColor(r, g, b, a)
        stops.append(QgsGradientStop(offset, color))

    return QgsGradientColorRamp(color1, color2, False, stops)


def build_multistop_gradient(
    stops: list[tuple[float, QColor]],
) -> QgsGradientColorRamp:
    """Build QgsGradientColorRamp from (position, color) stops.

    Parameters
    ----------
    stops : list of (float, QColor)
        List of (position, color) tuples. Positions must be in [0, 1],
        sorted ascending. Must have at least 2 entries.

    Returns
    -------
    QgsGradientColorRamp
        A gradient ramp with the specified color stops.
    """
    if len(stops) < 2:
        raise ValueError("stops must have at least 2 entries")
    color1 = stops[0][1]
    color2 = stops[-1][1]
    ramp_stops = [QgsGradientStop(pos, color) for pos, color in stops[1:-1]]
    return QgsGradientColorRamp(color1, color2, False, ramp_stops)


def db_tone_curve(n: int, base: float = 10.0) -> NDArray[np.float64]:
    """Decibel-style tone curve (log of squared linear ramp)."""
    linear = np.linspace(0.0, 1.0, n)
    return np.log1p((base - 1.0) * linear**2) / np.log(base)


def log_tone_curve(n: int, base: float = 10.0) -> NDArray[np.float64]:
    """Logarithmic tone curve. Return n samples in [0, 1]."""
    if base < 0.0 or n < 2:
        raise ValueError()

    linear = np.linspace(0.0, 1.0, n)

    return np.log1p((base - 1.0) * linear) / np.log(base)


def sqrt_tone_mapping(n: int) -> NDArray[np.float64]:
    """Square-root tone mapping. Returns n samples in [0, 1]."""
    if n < 2:
        raise ValueError("n must be >= 2")

    x = np.linspace(0.0, 1.0, n)
    return np.sqrt(x)


def asinh_tone_mapping(n: int, k: float = 5.0) -> NDArray[np.float64]:
    """
    Asinh tone mapping (soft logarithmic).

    Parameters
    ----------
    n : int
        Number of samples (>= 2)
    k : float
        Controls compression strength (typical: 3–10)

    Returns
    -------
    ndarray of float64
        Array of shape (n,) with values in [0, 1].
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if k <= 0:
        raise ValueError("k must be > 0")

    x = np.linspace(0.0, 1.0, n)
    return np.arcsinh(k * x) / np.arcsinh(k)


def loglog_tone_mapping(n: int, base: float = 10.0) -> NDArray[np.float64]:
    """
    Double-log (log ∘ log) tone mapping.

    Strong compression for very high dynamic range data (expert use).

    Parameters
    ----------
    n : int
        Number of samples (>= 2)
    base : float
        Log base (> 1). Typical: 10.

    Returns
    -------
    ndarray of float64
        Array of shape (n,) with values in [0, 1].
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if base <= 1.0:
        raise ValueError("base must be > 1")

    x = np.linspace(0.0, 1.0, n)

    # First log
    y1 = np.log1p((base - 1.0) * x) / np.log(base)

    # Second log (renormalized)
    y2 = np.log1p((base - 1.0) * y1) / np.log(base)

    return y2


N_COLORS = 256

# Register some styles
QgsStyle.defaultStyle().addColorRamp(
    "Grey Log",
    build_color_ramp_from_tone_curve(
        log_tone_curve(N_COLORS), QColor("black"), QColor("white")
    ),
    update=True,
)

QgsStyle.defaultStyle().addColorRamp(
    "Grey Square-Root",
    build_color_ramp_from_tone_curve(
        sqrt_tone_mapping(N_COLORS), QColor("black"), QColor("white")
    ),
    update=True,
)
QgsStyle.defaultStyle().addColorRamp(
    "Grey Asinh",
    build_color_ramp_from_tone_curve(
        asinh_tone_mapping(N_COLORS), QColor("black"), QColor("white")
    ),
    update=True,
)
QgsStyle.defaultStyle().addColorRamp(
    "Grey LogLog",
    build_color_ramp_from_tone_curve(
        loglog_tone_mapping(N_COLORS), QColor("black"), QColor("white")
    ),
    update=True,
)

# Teal gradient: black -> #2D6967 -> #ACD4CD -> white
TEAL_GRADIENT_STOPS = [
    (0.0, QColor("#000000")),
    (1.0 / 3.0, QColor("#2D6967")),
    (2.0 / 3.0, QColor("#ACD4CD")),
    (1.0, QColor("#FFFFFF")),
]

QgsStyle.defaultStyle().addColorRamp(
    "Teal",
    build_multistop_gradient(TEAL_GRADIENT_STOPS),
    update=True,
)
