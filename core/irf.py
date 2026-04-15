"""Impulse Response Function (IRF) analysis for SAR SLC data."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from osgeo import gdal
from qgis.core import Qgis, QgsMessageLog, QgsRasterLayer


@dataclass
class IrfResult:
    """Results from an IRF analysis at a single point."""

    range_profile_db: NDArray[np.float32]
    az_profile_db: NDArray[np.float32]
    range_resolution_m: float
    az_resolution_m: float
    range_pslr_db: float
    az_pslr_db: float
    range_islr_db: float
    az_islr_db: float
    pixel_spacing_range: float
    pixel_spacing_azimuth: float


def map_point_to_pixel(layer: QgsRasterLayer, map_point) -> tuple[int, int] | None:
    """Convert a map CRS point to raster pixel (col, row).

    Uses GCP_TPS transformer, matching the method used in get_extend_image_coords.

    Parameters
    ----------
    layer : QgsRasterLayer
        Raster layer with GCPs.
    map_point : QgsPointXY
        Point in map CRS.

    Returns
    -------
    tuple of (int, int) or None
        (col, row) pixel indices, or None on failure.
    """
    ds = gdal.Open(layer.dataProvider().dataSourceUri())
    if ds is None:
        return None
    try:
        t = gdal.Transformer(ds, None, ["METHOD=GCP_TPS"])
        if t is None:
            return None
        success, dst = t.TransformPoint(1, map_point.x(), map_point.y(), 0.0)
        if not success:
            return None
        return int(round(dst[0])), int(round(dst[1]))  # col, row
    except Exception as e:
        QgsMessageLog.logMessage(
            f"IRF: coordinate transform failed: {e}",
            "ICEYE Toolbox",
            Qgis.Warning,
        )
        return None
    finally:
        ds = None


def read_slc_chip(
    layer: QgsRasterLayer,
    row: int,
    col: int,
    half_window: int,
    meta,
) -> NDArray[np.complex64] | None:
    """Read a square SLC chip centred at (row, col) in pixel coordinates.

    Reconstructs complex data from amplitude (band 1) and phase (band 2),
    applying the same band-scale/offset and amplitude-mapping logic as
    read_slc_layer.  The window is clamped to raster bounds.

    Parameters
    ----------
    layer : QgsRasterLayer
        SLC raster layer (must have amplitude + phase bands).
    row, col : int
        Centre pixel (row = y-offset, col = x-offset from top-left corner).
    half_window : int
        Half-size of the extraction window in pixels.
    meta : IceyeMetadata
        Layer metadata (used for amplitude mapping).

    Returns
    -------
    ndarray of complex64 or None
        Complex chip array shape (h, w), or None on failure.
    """
    source_path = layer.dataProvider().dataSourceUri()
    ds = gdal.Open(source_path)
    if ds is None:
        return None
    try:
        n_cols = ds.RasterXSize
        n_rows = ds.RasterYSize

        col0 = max(0, col - half_window)
        row0 = max(0, row - half_window)
        col1 = min(n_cols, col + half_window)
        row1 = min(n_rows, row + half_window)
        w = col1 - col0
        h = row1 - row0
        if w <= 0 or h <= 0:
            return None

        amp_band = ds.GetRasterBand(1)
        phase_band = ds.GetRasterBand(2)

        amp_scale = amp_band.GetScale() or 1.0
        amp_offset = amp_band.GetOffset() or 0.0
        phase_scale = phase_band.GetScale() or 1.0
        phase_offset = phase_band.GetOffset() or 0.0

        amp_data = amp_band.ReadAsArray(col0, row0, w, h)
        phase_data = phase_band.ReadAsArray(col0, row0, w, h)

        mapping = meta.iceye_amplitude_mapping if meta else None
        function = mapping.get("function") if mapping else None

        if function == "power":
            params = mapping.get("parameters", {})
            scale = params.get("scale", 1.0)
            exponent = params.get("exponent", 1.0)
            amp_complex = (
                amp_data.astype(np.complex64) ** (scale / exponent) * amp_scale
                + amp_offset
            )
        else:
            amp_complex = amp_data.astype(np.complex64) * amp_scale + amp_offset

        data = amp_complex * np.exp(
            -1j * (phase_data.astype(np.float32) * phase_scale + phase_offset)
        )
        return data.astype(np.complex64)
    except Exception as e:
        QgsMessageLog.logMessage(
            f"IRF: chip read failed: {e}",
            "ICEYE Toolbox",
            Qgis.Warning,
        )
        return None
    finally:
        ds = None


def _fft2_complex_interpolation(
    data: NDArray[np.complex64],
    target_shape: tuple[int, int],
) -> tuple[NDArray[np.complex64], float, float]:
    """Interpolate 2D complex data by FFT zero-padding in the frequency domain.

    Parameters
    ----------
    data : ndarray of complex64
        2D complex SLC chip (rows × cols).
    target_shape : tuple of (int, int)
        (target_rows, target_cols) after interpolation.

    Returns
    -------
    tuple of (interpolated, oversampling_row, oversampling_col)
    """
    data = np.asarray(data, dtype=np.complex64)
    if data.ndim != 2:
        raise ValueError("Input data must be a 2D array.")
    orig_rows, orig_cols = data.shape
    target_rows, target_cols = target_shape
    if target_rows < orig_rows or target_cols < orig_cols:
        raise ValueError("Target shape must be >= original shape.")

    pad_rows = target_rows - orig_rows
    pad_cols = target_cols - orig_cols
    pad_before = (pad_rows // 2, pad_cols // 2)
    pad_after = (pad_rows - pad_before[0], pad_cols - pad_before[1])

    spectrum = np.fft.fftshift(np.fft.fft2(data))
    spectrum_padded = np.pad(
        spectrum,
        pad_width=((pad_before[0], pad_after[0]), (pad_before[1], pad_after[1])),
        mode="constant",
        constant_values=0,
    )
    interpolated = np.fft.ifft2(np.fft.ifftshift(spectrum_padded))
    scale_factor = (target_rows / orig_rows) * (target_cols / orig_cols)
    interpolated = (interpolated * scale_factor).astype(np.complex64)

    oversampling_row = target_rows / orig_rows
    oversampling_col = target_cols / orig_cols
    return interpolated, oversampling_row, oversampling_col


def _calculate_mainlobe_width(profile: NDArray[np.float64]) -> int:
    """Count samples with linear amplitude >= max - 0.5 (mainlobe extent)."""
    half_power_level = float(np.max(profile)) - 0.5
    above_half = profile >= half_power_level
    return int(np.sum(above_half))


def _compute_hpbw(profile_db: NDArray[np.float64], spacing_m: float) -> float:
    """Half-power beamwidth at -3 dB (metres)."""
    half_power_level = float(np.max(profile_db)) - 3.0
    above_half = profile_db >= half_power_level
    indices = np.where(above_half)[0]
    if len(indices) < 2:
        return 0.0
    return float(indices[-1] - indices[0]) * spacing_m


def _compute_pslr(profile: NDArray[np.float64]) -> float:
    """Peak sidelobe ratio (dB): exclude mainlobe, 10·log10(max_sidelobe/peak)."""
    power_profile = profile.astype(np.float64) ** 2
    peak_idx = int(np.argmax(power_profile))
    peak_val = float(power_profile[peak_idx])
    exclude_width = _calculate_mainlobe_width(profile)
    mask = np.ones_like(power_profile, dtype=bool)
    mask[max(0, peak_idx - exclude_width) : peak_idx + exclude_width + 1] = False
    sidelobe_power = power_profile[mask]
    if len(sidelobe_power) == 0:
        return float("-inf")
    max_sidelobe = float(np.max(sidelobe_power))
    return float(10.0 * np.log10(max_sidelobe / peak_val))


def _compute_islr(profile: NDArray[np.float64]) -> float:
    """Integrated sidelobe ratio (dB): mainlobe = peak ± mainlobe_width."""
    power_profile = profile.astype(np.float64) ** 2
    peak_idx = int(np.argmax(power_profile))
    mainlobe_width = _calculate_mainlobe_width(profile)
    start_mainlobe = max(peak_idx - mainlobe_width, 0)
    end_mainlobe = min(peak_idx + mainlobe_width + 1, len(profile))
    mainlobe_power = float(np.sum(power_profile[start_mainlobe:end_mainlobe]))
    sidelobe_power = float(np.sum(power_profile)) - mainlobe_power
    if mainlobe_power <= 0:
        return float("-inf")
    return float(10.0 * np.log10(sidelobe_power / mainlobe_power))


def analyze_point(
    data: NDArray[np.complex64],
    peak_row: int,
    peak_col: int,
    pixel_spacing_range: float,
    pixel_spacing_azimuth: float,
    half_window: int = 64,
    az_window_meters: float = 4.0,
    rg_window_meters: float = 4.0,
) -> IrfResult:
    """Perform IRF analysis on a chip of SLC data.

    Uses 2D FFT zero-padding interpolation, extracts profiles cropped to
    ±window_meters/2 around the peak, then computes HPBW, PSLR, and ISLR.

    Parameters
    ----------
    data : ndarray of complex64
        SLC chip (rows x cols).
    peak_row, peak_col : int
        Pixel location of the brightest target within *data*.
    pixel_spacing_range : float
        Ground range pixel spacing in metres.
    pixel_spacing_azimuth : float
        Azimuth pixel spacing in metres.
    half_window : int
        Half-length of the 1D analysis window on each side of the peak.
    az_window_meters : float
        Azimuth profile window extent in metres (default 4.0, i.e. ±2 m).
    rg_window_meters : float
        Range profile window extent in metres (default 4.0, i.e. ±2 m).

    Returns
    -------
    IrfResult
    """
    n_rows, n_cols = data.shape

    r0 = max(0, peak_row - half_window)
    r1 = min(n_rows, peak_row + half_window + 1)
    c0 = max(0, peak_col - half_window)
    c1 = min(n_cols, peak_col + half_window + 1)

    chip = data[r0:r1, c0:c1].astype(np.complex64)
    patch_rows, patch_cols = chip.shape

    target_rows = 2 ** math.ceil(math.log2(10 * patch_rows))
    target_cols = 2 ** math.ceil(math.log2(10 * patch_cols))
    interpolated, oversampling_row, oversampling_col = _fft2_complex_interpolation(
        chip, (target_rows, target_cols)
    )

    slc_abs = np.abs(interpolated)
    peak_row_int, peak_col_int = np.unravel_index(np.argmax(slc_abs), slc_abs.shape)
    peak_amplitude = float(slc_abs[peak_row_int, peak_col_int])
    slc_abs_norm = slc_abs / peak_amplitude

    az_profile_full = slc_abs_norm[:, peak_col_int].astype(np.float64)
    rng_profile_full = slc_abs_norm[peak_row_int, :].astype(np.float64)

    spacing_az = pixel_spacing_azimuth / oversampling_row
    spacing_rng = pixel_spacing_range / oversampling_col

    az_half_samples = int(0.5 * az_window_meters / spacing_az)
    rg_half_samples = int(0.5 * rg_window_meters / spacing_rng)

    az_start = max(peak_row_int - az_half_samples, 0)
    az_end = min(peak_row_int + az_half_samples + 1, len(az_profile_full))
    rg_start = max(peak_col_int - rg_half_samples, 0)
    rg_end = min(peak_col_int + rg_half_samples + 1, len(rng_profile_full))

    az_profile = az_profile_full[az_start:az_end]
    rng_profile = rng_profile_full[rg_start:rg_end]

    def _to_db(profile: NDArray[np.float64]) -> NDArray[np.float64]:
        with np.errstate(divide="ignore"):
            db = 20.0 * np.log10(profile + 1e-12)
        return np.where(np.isfinite(db), db, -100.0).astype(np.float64)

    az_profile_db = _to_db(az_profile)
    rng_profile_db = _to_db(rng_profile)

    az_res = _compute_hpbw(az_profile_db, spacing_az)
    rng_res = _compute_hpbw(rng_profile_db, spacing_rng)

    az_pslr = _compute_pslr(az_profile)
    rng_pslr = _compute_pslr(rng_profile)

    az_islr = _compute_islr(az_profile)
    rng_islr = _compute_islr(rng_profile)

    return IrfResult(
        range_profile_db=rng_profile_db.astype(np.float32),
        az_profile_db=az_profile_db.astype(np.float32),
        range_resolution_m=rng_res,
        az_resolution_m=az_res,
        range_pslr_db=rng_pslr,
        az_pslr_db=az_pslr,
        range_islr_db=rng_islr,
        az_islr_db=az_islr,
        pixel_spacing_range=spacing_rng,
        pixel_spacing_azimuth=spacing_az,
    )
