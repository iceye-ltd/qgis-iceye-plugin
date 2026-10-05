"""SLC geometry and target detection for moving-target relocation.

Orbit fit, range-Doppler inverse and geocoding, SLC chips with GCP geolocation, and
the click corridor, clutter ring and hull masks used by core.mover_relocation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from osgeo import gdal

from .geometry import ecef_to_geodetic, geodetic_to_ecef
from .metadata import parse_iso8601_datetime
from .typing_compat import NDArray

_WGS84_A = 6378137.0
_WGS84_B = 6356752.314245


# ----------------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------------


@dataclass
class HullParameters:
    """Hull detection settings (distances in ground metres)."""

    corridor_half_width_m: float = 15.0
    ring_gap_m: float = 10.0
    ring_width_m: float = 30.0
    hull_snr_db: float = 10.0
    hull_dynamic_range_db: float = 25.0


# ----------------------------------------------------------------------------------
# Coordinates
# ----------------------------------------------------------------------------------


def ecef_to_lonlat(point: NDArray[np.float64]) -> tuple[float, float]:
    """Convert ECEF (m) to (lon, lat) in degrees on WGS84."""
    lat, lon = ecef_to_geodetic(*(float(v) for v in point))
    return math.degrees(lon), math.degrees(lat)


def lonlat_to_ecef(lon: float, lat: float, height: float) -> NDArray[np.float64]:
    """Geodetic (lon, lat deg, height m) to ECEF (m)."""
    return np.array(geodetic_to_ecef(lat, lon, height), dtype=np.float64)


def enu_basis(lon: float, lat: float) -> NDArray[np.float64]:
    """Rows are the East, North, Up unit vectors in ECEF at (lon, lat)."""
    lo, la = math.radians(lon), math.radians(lat)
    return np.array(
        [
            [-math.sin(lo), math.cos(lo), 0.0],
            [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
            [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)],
        ]
    )


# ----------------------------------------------------------------------------------
# Orbit and product geometry
# ----------------------------------------------------------------------------------


class Orbit:
    """Polynomial fit to ECEF orbit states; times in seconds from a reference epoch."""

    def __init__(
        self,
        times: NDArray[np.float64],
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64] | None = None,
        degree: int = 7,
    ) -> None:
        """Fit each ECEF axis with a polynomial in normalised time."""
        times = np.asarray(times, dtype=np.float64)
        positions = np.asarray(positions, dtype=np.float64)
        self._t_mid = float(np.mean(times))
        self._t_scale = max(float(np.ptp(times)) / 2.0, 1e-9)
        tn = (times - self._t_mid) / self._t_scale
        degree = min(degree, len(times) - 1)
        self._pos = [np.polyfit(tn, positions[:, i], degree) for i in range(3)]
        self._vel = [np.polyder(c) for c in self._pos]
        self._acc = [np.polyder(c, 2) for c in self._pos]
        self.velocities = velocities

    def _eval(self, coeffs, t: float, power: int) -> NDArray[np.float64]:
        tn = (t - self._t_mid) / self._t_scale
        return np.array([np.polyval(c, tn) for c in coeffs]) / self._t_scale**power

    def position(self, t: float) -> NDArray[np.float64]:
        """Satellite position at time t."""
        return self._eval(self._pos, t, 0)

    def velocity(self, t: float) -> NDArray[np.float64]:
        """Satellite velocity at time t."""
        return self._eval(self._vel, t, 1)

    def acceleration(self, t: float) -> NDArray[np.float64]:
        """Satellite acceleration at time t."""
        return self._eval(self._acc, t, 2)

    def zero_doppler(
        self, point: NDArray[np.float64], t_guess: float
    ) -> tuple[float, float]:
        """Zero-Doppler time and slant range of an Earth-fixed ECEF point."""
        t = float(t_guess)
        for _ in range(30):
            d = point - self.position(t)
            v = self.velocity(t)
            g = float(v @ d)
            gp = float(self.acceleration(t) @ d - v @ v)
            step = g / gp
            t -= step
            if abs(step) < 1e-10:
                break
        return t, float(np.linalg.norm(point - self.position(t)))

    def geocode(
        self,
        slant_range: float,
        t: float,
        height: float,
        guess: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Zero-Doppler point at (slant_range, t) on the ellipsoid raised by height.

        ``guess`` picks the look side; the imaged position is always close enough.
        """
        s = self.position(t)
        v = self.velocity(t)
        ah, bh = _WGS84_A + height, _WGS84_B + height
        p = np.array(guess, dtype=np.float64)
        for _ in range(30):
            d = p - s
            f = np.array(
                [
                    v @ d,
                    d @ d - slant_range**2,
                    (p[0] ** 2 + p[1] ** 2) / ah**2 + p[2] ** 2 / bh**2 - 1.0,
                ]
            )
            jac = np.array(
                [
                    v,
                    2.0 * d,
                    [2.0 * p[0] / ah**2, 2.0 * p[1] / ah**2, 2.0 * p[2] / bh**2],
                ]
            )
            step = np.linalg.solve(jac, f)
            p = p - step
            if np.linalg.norm(step) < 1e-4:
                break
        return p


@dataclass
class ProductGeometry:
    """Metadata needed for relocation, parsed from ``ICEYE_PROPERTIES``.

    Times are seconds from ``reference_time`` (the zero-Doppler start).
    """

    reference_time: datetime
    scene_height: float
    orbit: Orbit
    dc_times: NDArray[np.float64]
    # Echo collection window (start_datetime .. end_datetime), seconds.
    acquisition_window: tuple[float, float] | None = None

    @classmethod
    def from_properties(cls, props: dict[str, Any]) -> ProductGeometry:
        """Build from the decoded ``ICEYE_PROPERTIES`` JSON."""
        t0 = parse_iso8601_datetime(props["iceye:zero_doppler_start_datetime"])
        if t0 is None:
            raise ValueError("Missing iceye:zero_doppler_start_datetime")

        def seconds(value: str) -> float:
            parsed = parse_iso8601_datetime(value)
            if parsed is None:
                raise ValueError(f"Unparseable datetime {value!r}")
            return (parsed - t0).total_seconds()

        states = props["iceye:orbit_states"]
        orbit = Orbit(
            np.array([seconds(s["time"]) for s in states]),
            np.array([s["position"] for s in states], dtype=np.float64),
            np.array([s["velocity"] for s in states], dtype=np.float64),
        )
        dc_times = np.array(
            [seconds(s) for s in props.get("iceye:doppler_centroid_datetimes") or []]
        )
        window = None
        if props.get("start_datetime") and props.get("end_datetime"):
            window = (seconds(props["start_datetime"]), seconds(props["end_datetime"]))
        return cls(
            reference_time=t0,
            scene_height=float(props.get("iceye:average_scene_height") or 0.0),
            orbit=orbit,
            dc_times=dc_times,
            acquisition_window=window,
        )


def read_iceye_properties(source_path: str) -> dict[str, Any]:
    """Decode ``ICEYE_PROPERTIES`` of an ICEYE GeoTIFF."""
    dataset = gdal.Open(source_path)
    if dataset is None:
        raise ValueError(f"Failed to open {source_path}")
    try:
        return json.loads(dataset.GetMetadata()["ICEYE_PROPERTIES"])
    finally:
        dataset = None


# ----------------------------------------------------------------------------------
# SLC chip
# ----------------------------------------------------------------------------------


PixelToLonLat = Callable[[float, float], tuple[float, float]]


class _GcpTransform:
    """GCP TPS transform of an ICEYE GeoTIFF: pixel -> lon/lat, or its inverse."""

    def __init__(self, source_path: str, inverse: bool) -> None:
        self._dataset = gdal.Open(source_path)
        if self._dataset is None:
            raise ValueError(f"Failed to open {source_path}")
        self._transformer = gdal.Transformer(self._dataset, None, ["METHOD=GCP_TPS"])
        self._inverse = int(inverse)

    def __call__(self, x: float, y: float) -> tuple[float, float]:
        """(col, row) -> (lon, lat), or (lon, lat) -> (col, row) when inverse."""
        ok, point = self._transformer.TransformPoint(
            self._inverse, float(x), float(y), 0.0
        )
        if not ok:
            raise ValueError(f"({x}, {y}) could not be transformed")
        return float(point[0]), float(point[1])

    def close(self) -> None:
        """Release the transformer before the dataset it references."""
        self._transformer = None
        self._dataset = None


def gcp_pixel_to_lonlat(source_path: str) -> PixelToLonLat:
    """Return a (file col, file row) -> (lon, lat) function from the GCP TPS model."""
    return _GcpTransform(source_path, inverse=False)


def gcp_lonlat_to_pixel(
    source_path: str,
) -> Callable[[float, float], tuple[float, float]]:
    """Return a (lon, lat) -> (file col, file row) function, inverse of the GCP model."""
    return _GcpTransform(source_path, inverse=True)


def gcp_mean_height(source_path: str, default: float = 0.0) -> float:
    """Return the mean GCP height, the surface the GCP-warped display lies on."""
    dataset = gdal.Open(source_path)
    if dataset is None:
        raise ValueError(f"Failed to open {source_path}")
    heights = [gcp.GCPZ for gcp in dataset.GetGCPs() or []]
    return float(np.mean(heights)) if heights else default


def patch_to_file_layout(
    data_patch: NDArray[np.complexfloating[Any]], left: bool
) -> NDArray[np.complexfloating[Any]]:
    """Undo ``core.raster.toggle_shadows_down``: back to file rows / columns."""
    data = data_patch.T
    return np.fliplr(data) if left else data


@dataclass
class SlcChip:
    """Complex SLC window in file layout (rows = range samples, cols = azimuth lines).

    ``col0`` / ``row0`` is the window origin in file pixels; ``pixel_to_lonlat`` maps
    file pixel coordinates (pixel centres at +0.5) to lon/lat.
    """

    data: NDArray[np.complex64]
    col0: int
    row0: int
    geometry: ProductGeometry
    pixel_to_lonlat: PixelToLonLat

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols)."""
        return self.data.shape  # type: ignore[return-value]

    def lonlat(self, row: float, col: float) -> tuple[float, float]:
        """Lon/lat of a chip pixel centre."""
        return self.pixel_to_lonlat(self.col0 + col + 0.5, self.row0 + row + 0.5)

    def ecef(self, row: float, col: float) -> NDArray[np.float64]:
        """ECEF of a chip pixel on the scene-height surface."""
        lon, lat = self.lonlat(row, col)
        return lonlat_to_ecef(lon, lat, self.geometry.scene_height)

    def pixel_enu_steps(self, row: float, col: float) -> NDArray[np.float64]:
        """Ground ENU metres per +1 row (first row) and per +1 column (second row)."""
        lon, lat = self.lonlat(row, col)
        basis = enu_basis(lon, lat)
        p0 = self.ecef(row, col)
        dr = max(1.0, min(20.0, self.shape[0] / 4.0))
        dc = max(1.0, min(200.0, self.shape[1] / 4.0))
        step_row = basis @ (self.ecef(row + dr, col) - p0) / dr
        step_col = basis @ (self.ecef(row, col + dc) - p0) / dc
        return np.array([step_row[:2], step_col[:2]])


# ----------------------------------------------------------------------------------
# Click masks
# ----------------------------------------------------------------------------------


def ellipse_mask(
    shape: tuple[int, int],
    row: float,
    col: float,
    half_rows: float,
    half_cols: float,
) -> NDArray[np.bool_]:
    """Return the pixels within an elliptical (half_rows, half_cols) distance of a point."""
    rr = (np.arange(shape[0])[:, None] - row) / max(half_rows, 0.5)
    cc = (np.arange(shape[1])[None, :] - col) / max(half_cols, 0.5)
    return rr**2 + cc**2 <= 1.0


@dataclass
class ClickMasks:
    """Corridor disc and clutter ring around a click, with the sub-window they cover."""

    corridor: NDArray[np.bool_]
    ring: NDArray[np.bool_]
    window: tuple[slice, slice]


def build_click_masks(
    chip: SlcChip,
    row: float,
    col: float,
    params: HullParameters,
    row_spacing_m: float,
    col_spacing_m: float,
) -> ClickMasks:
    """Build the corridor and ring masks around chip pixel (row, col).

    The corridor is corridor_half_width_m of ground around the click; the ring lies
    ring_gap_m beyond it and is ring_width_m wide.
    """
    outer = params.corridor_half_width_m + params.ring_gap_m + params.ring_width_m
    margin_r = int(math.ceil(outer / row_spacing_m)) + 1
    margin_c = int(math.ceil(outer / col_spacing_m)) + 1
    r_lo = max(int(math.floor(row)) - margin_r, 0)
    r_hi = min(int(math.ceil(row)) + margin_r + 1, chip.shape[0])
    c_lo = max(int(math.floor(col)) - margin_c, 0)
    c_hi = min(int(math.ceil(col)) + margin_c + 1, chip.shape[1])
    shape = (r_hi - r_lo, c_hi - c_lo)

    def disc(half_m: float) -> NDArray[np.bool_]:
        return ellipse_mask(
            shape,
            row - r_lo,
            col - c_lo,
            half_m / row_spacing_m,
            half_m / col_spacing_m,
        )

    inner = disc(params.corridor_half_width_m)
    ring = disc(outer) & ~disc(params.corridor_half_width_m + params.ring_gap_m)
    return ClickMasks(inner, ring, (slice(r_lo, r_hi), slice(c_lo, c_hi)))


def hull_mask(
    intensity: NDArray[np.floating[Any]],
    corridor: NDArray[np.bool_],
    ring: NDArray[np.bool_],
    params: HullParameters,
) -> NDArray[np.bool_]:
    """Return the bright hull pixels in the corridor.

    Pixels must exceed both the ring clutter by hull_snr_db and the corridor peak
    minus hull_dynamic_range_db, which drops most sidelobe energy.
    """
    ring_values = intensity[ring] if ring.any() else intensity[corridor]
    clutter = float(np.median(ring_values)) if ring_values.size else 0.0
    inside = intensity[corridor]
    peak = float(inside.max()) if inside.size else 0.0
    threshold = max(
        clutter * 10.0 ** (params.hull_snr_db / 10.0),
        peak * 10.0 ** (-params.hull_dynamic_range_db / 10.0),
    )
    return corridor & (intensity > threshold)
