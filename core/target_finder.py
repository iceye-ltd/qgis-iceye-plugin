"""SLC geometry, target detection and map drift for moving-target relocation.

Orbit fit, range-Doppler inverse and geocoding, SLC chips with GCP geolocation, and
the corridor, clutter ring and hull masks used by core.mover_relocation. The map
drift (find_targets_along_curve, map_drift) is kept only for an optional
along-track velocity check; its sign is not yet validated against a known mover.

The Doppler-centroid radial-velocity estimate was removed: in Spotlight / Dwell the
beam-steered centroid moves at almost exactly the FM rate, so a mover's own echoes
carry no usable radial velocity (doppler_amplification is ~3000 on the WWGTZ2
fixture). Azimuth sample spacing is 1 / iceye:processing_prf, and the time direction
along file columns is taken from the geolocation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from osgeo import gdal

from .geometry import geodetic_to_ecef
from .metadata import parse_iso8601_datetime
from .typing_compat import NDArray

SPEED_OF_LIGHT = 299792458.0

_WGS84_A = 6378137.0
_WGS84_B = 6356752.314245
_WGS84_E2 = 1.0 - (_WGS84_B / _WGS84_A) ** 2

# Physical Doppler = DOPPLER_SIGN * Doppler measured on read_slc_layer samples
# (s = A * exp(-j * phase)). Positive physical Doppler = approaching the radar.
DOPPLER_SIGN = 1.0

# ----------------------------------------------------------------------------------
# Parameters and results
# ----------------------------------------------------------------------------------


@dataclass
class RelocationParameters:
    """Hull detection and map-drift settings (distances in ground metres)."""

    corridor_half_width_m: float = 15.0
    ring_gap_m: float = 10.0
    ring_width_m: float = 30.0
    hull_snr_db: float = 10.0
    hull_dynamic_range_db: float = 25.0
    n_looks: int = 5
    streak_db: float = 15.0


@dataclass
class CurveTarget:
    """Ship position found in one sub-aperture look along the curve."""

    look: int
    doppler_hz: float
    slow_time_s: float
    row: float
    col: float
    x_fraction: float
    y_fraction: float
    intensity_db: float
    n_pixels: int


# ----------------------------------------------------------------------------------
# Coordinates
# ----------------------------------------------------------------------------------


def ecef_to_geodetic(point: NDArray[np.float64]) -> tuple[float, float, float]:
    """ECEF (m) to geodetic (lat deg, lon deg, height m) on WGS84 (Bowring)."""
    x, y, z = (float(v) for v in point)
    a, b, e2 = _WGS84_A, _WGS84_B, _WGS84_E2
    ep2 = (a * a - b * b) / (b * b)
    p = math.hypot(x, y)
    theta = math.atan2(z * a, p * b)
    lat = math.atan2(
        z + ep2 * b * math.sin(theta) ** 3, p - e2 * a * math.cos(theta) ** 3
    )
    for _ in range(3):
        n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - e2 * n / (n + h)))
    n = a / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
    h = p / math.cos(lat) - n
    return math.degrees(lat), math.degrees(math.atan2(y, x)), h


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
class Kinematics:
    """Zero-Doppler kinematics of an Earth-fixed point."""

    fm_rate: float
    v_eff: float
    v_ground: float
    slant_range: float
    sat_position: NDArray[np.float64]
    sat_velocity: NDArray[np.float64]


@dataclass
class ProductGeometry:
    """Metadata needed for relocation, parsed from ``ICEYE_PROPERTIES``.

    Times are seconds from ``reference_time`` (the zero-Doppler start).
    """

    reference_time: datetime
    wavelength: float
    azimuth_time_spacing: float
    acquisition_prf: float
    processed_azimuth_bandwidth: float
    range_near: float
    pixel_spacing_range: float
    pixel_spacing_azimuth: float
    incidence_angle_deg: float
    look_side: str
    scene_height: float
    instrument_mode: str
    orbit: Orbit
    dc_times: NDArray[np.float64]
    dc_coeffs: list[NDArray[np.float64]]
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
        dc_coeffs = [
            np.asarray(c, dtype=np.float64)
            for c in props.get("iceye:doppler_centroid_coeffs") or []
        ]
        processing_prf = props.get("iceye:processing_prf") or props.get(
            "iceye:acquisition_prf"
        )
        window = None
        if props.get("start_datetime") and props.get("end_datetime"):
            window = (seconds(props["start_datetime"]), seconds(props["end_datetime"]))
        incidence = props.get("view:incidence_angle")
        if incidence is None:
            incidence = 0.5 * (
                props["iceye:incidence_angle_near"] + props["iceye:incidence_angle_far"]
            )
        return cls(
            reference_time=t0,
            wavelength=SPEED_OF_LIGHT / float(props["sar:center_frequency"]),
            azimuth_time_spacing=1.0 / float(processing_prf),
            acquisition_prf=float(props.get("iceye:acquisition_prf") or processing_prf),
            processed_azimuth_bandwidth=float(
                props.get("iceye:processing_bandwidth_azimuth") or processing_prf
            ),
            range_near=float(props["iceye:range_near"]),
            pixel_spacing_range=float(props["sar:pixel_spacing_range"]),
            pixel_spacing_azimuth=float(props["sar:pixel_spacing_azimuth"]),
            incidence_angle_deg=float(incidence),
            look_side=(props.get("sar:observation_direction") or "right").lower(),
            scene_height=float(props.get("iceye:average_scene_height") or 0.0),
            instrument_mode=(props.get("sar:instrument_mode") or "").lower(),
            orbit=orbit,
            dc_times=dc_times,
            dc_coeffs=dc_coeffs,
            acquisition_window=window,
        )

    @property
    def ground_range_spacing(self) -> float:
        """Ground-range metres per range sample."""
        return self.pixel_spacing_range / math.sin(
            math.radians(self.incidence_angle_deg)
        )

    def doppler_centroid(self, t: float, slant_range: float) -> float:
        """Metadata Doppler centroid (Hz) at zero-Doppler time t and slant range.

        Each entry of ``iceye:doppler_centroid_coeffs`` is taken as a polynomial in
        slant-range time from near range (increasing powers); the fixture only has
        constants, so the range variable is an unverified assumption. Between entries
        the value is interpolated linearly in time and extrapolated from the end slopes.
        """
        if not self.dc_coeffs:
            return 0.0
        tau = 2.0 * (slant_range - self.range_near) / SPEED_OF_LIGHT
        values = np.array(
            [np.polyval(np.asarray(c)[::-1], tau) for c in self.dc_coeffs]
        )
        times = self.dc_times
        if len(values) == 1 or len(times) != len(values):
            return float(values[0])
        if t <= times[0]:
            slope = (values[1] - values[0]) / (times[1] - times[0])
            return float(values[0] + slope * (t - times[0]))
        if t >= times[-1]:
            slope = (values[-1] - values[-2]) / (times[-1] - times[-2])
            return float(values[-1] + slope * (t - times[-1]))
        return float(np.interp(t, times, values))

    def doppler_centroid_rate(self, t: float, slant_range: float) -> float:
        """Time derivative of the metadata Doppler centroid (Hz/s)."""
        h = 1e-3
        return (
            self.doppler_centroid(t + h, slant_range)
            - self.doppler_centroid(t - h, slant_range)
        ) / (2.0 * h)

    def kinematics(self, t: float, point: NDArray[np.float64]) -> Kinematics:
        """FM rate and effective / ground velocity at a zero-Doppler point.

        ``Ka = 2 * V_eff^2 / (lambda * R)`` with ``V_eff^2 = |v|^2 - a . (P - S)``, the
        exact second derivative of the range history in the Earth-fixed frame.
        """
        s = self.orbit.position(t)
        v = self.orbit.velocity(t)
        d = point - s
        r = float(np.linalg.norm(d))
        v_eff2 = float(v @ v - self.orbit.acceleration(t) @ d)
        v_ground = (
            float(np.linalg.norm(v))
            * float(np.linalg.norm(point))
            / float(np.linalg.norm(s))
        )
        return Kinematics(
            fm_rate=2.0 * v_eff2 / (self.wavelength * r),
            v_eff=math.sqrt(max(v_eff2, 0.0)),
            v_ground=v_ground,
            slant_range=r,
            sat_position=s,
            sat_velocity=v,
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

    def zero_doppler(self, row: float, col: float) -> tuple[float, float]:
        """Zero-Doppler time and slant range of a chip pixel."""
        guess = 0.5 * (
            self.geometry.dc_times[0] + self.geometry.dc_times[-1]
            if len(self.geometry.dc_times)
            else 0.0
        )
        return self.geometry.orbit.zero_doppler(self.ecef(row, col), guess)

    def azimuth_time_step(self, row: float, col: float) -> float:
        """Signed zero-Doppler time per column (magnitude from the processing PRF)."""
        span = max(1.0, min(50.0, self.shape[1] / 4.0))
        t_a, _ = self.zero_doppler(row, col - span)
        t_b, _ = self.zero_doppler(row, col + span)
        sign = 1.0 if t_b >= t_a else -1.0
        return sign * self.geometry.azimuth_time_spacing

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
# Curve and masks
# ----------------------------------------------------------------------------------


def _as_xy(point: Any) -> tuple[float, float]:
    """(x, y) from a QPointF-like object or a 2-sequence."""
    if hasattr(point, "x") and callable(point.x):
        return float(point.x()), float(point.y())
    return float(point[0]), float(point[1])


def bezier_points(control_points: Sequence[Any], n: int = 256) -> NDArray[np.float64]:
    """Sample a cubic Bezier given as four (x, y) control points; returns (n, 2)."""
    p0, c1, c2, p3 = (np.array(_as_xy(p)) for p in control_points)
    t = np.linspace(0.0, 1.0, n)[:, None]
    u = 1.0 - t
    return u**3 * p0 + 3 * u**2 * t * c1 + 3 * u * t**2 * c2 + t**3 * p3


def curve_pixels(
    control_points: Sequence[Any], shape: tuple[int, int], n: int = 256
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Curve samples as (rows, cols) of a chip; control points are 0..1 fractions."""
    xy = bezier_points(control_points, n)
    rows = xy[:, 1] * (shape[0] - 1)
    cols = xy[:, 0] * (shape[1] - 1)
    return rows, cols


def corridor_mask(
    shape: tuple[int, int],
    rows: NDArray[np.float64],
    cols: NDArray[np.float64],
    half_rows: float,
    half_cols: float,
) -> NDArray[np.bool_]:
    """Pixels within an elliptical (half_rows, half_cols) distance of a polyline."""
    mask = np.zeros(shape, dtype=bool)
    hr, hc = max(half_rows, 0.5), max(half_cols, 0.5)
    # Densify so consecutive stamps overlap by at least 3/4 of their radius.
    pts_r, pts_c = [rows[0]], [cols[0]]
    for r0, c0, r1, c1 in zip(rows[:-1], cols[:-1], rows[1:], cols[1:]):
        steps = int(math.ceil(math.hypot((r1 - r0) / hr, (c1 - c0) / hc) / 0.25))
        for k in range(1, max(steps, 1) + 1):
            pts_r.append(r0 + (r1 - r0) * k / max(steps, 1))
            pts_c.append(c0 + (c1 - c0) * k / max(steps, 1))
    ir, ic = int(math.ceil(hr)), int(math.ceil(hc))
    rr = np.arange(-ir, ir + 1)[:, None]
    cc = np.arange(-ic, ic + 1)[None, :]
    stamp = (rr / hr) ** 2 + (cc / hc) ** 2 <= 1.0
    centres = {(int(round(r)), int(round(c))) for r, c in zip(pts_r, pts_c)}
    for r, c in centres:
        r_lo, r_hi = max(r - ir, 0), min(r + ir + 1, shape[0])
        c_lo, c_hi = max(c - ic, 0), min(c + ic + 1, shape[1])
        if r_lo >= r_hi or c_lo >= c_hi:
            continue
        mask[r_lo:r_hi, c_lo:c_hi] |= stamp[
            r_lo - (r - ir) : r_hi - (r - ir), c_lo - (c - ic) : c_hi - (c - ic)
        ]
    return mask


@dataclass
class CurveMasks:
    """Corridor and clutter ring around the curve, with the sub-window they occupy."""

    corridor: NDArray[np.bool_]
    ring: NDArray[np.bool_]
    window: tuple[slice, slice]


def build_curve_masks(
    chip: SlcChip,
    control_points: Sequence[Any],
    params: RelocationParameters,
    row_spacing_m: float,
    col_spacing_m: float,
) -> CurveMasks:
    """Corridor (``corridor_half_width_m``) and ring masks on a sub-window of the chip."""
    rows, cols = curve_pixels(control_points, chip.shape)
    outer = params.corridor_half_width_m + params.ring_gap_m + params.ring_width_m
    margin_r = int(math.ceil(outer / row_spacing_m)) + 1
    margin_c = int(math.ceil(outer / col_spacing_m)) + 1
    r_lo = max(int(math.floor(rows.min())) - margin_r, 0)
    r_hi = min(int(math.ceil(rows.max())) + margin_r + 1, chip.shape[0])
    c_lo = max(int(math.floor(cols.min())) - margin_c, 0)
    c_hi = min(int(math.ceil(cols.max())) + margin_c + 1, chip.shape[1])
    shape = (r_hi - r_lo, c_hi - c_lo)
    rows, cols = rows - r_lo, cols - c_lo

    def corridor(half_m: float) -> NDArray[np.bool_]:
        return corridor_mask(
            shape, rows, cols, half_m / row_spacing_m, half_m / col_spacing_m
        )

    inner = corridor(params.corridor_half_width_m)
    gap = corridor(params.corridor_half_width_m + params.ring_gap_m)
    ring = corridor(outer) & ~gap
    return CurveMasks(inner, ring, (slice(r_lo, r_hi), slice(c_lo, c_hi)))


def hull_and_clutter_masks(
    intensity: NDArray[np.floating[Any]],
    corridor: NDArray[np.bool_],
    ring: NDArray[np.bool_],
    params: RelocationParameters,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], float]:
    """Bright hull pixels in the corridor, and ring pixels free of bright returns.

    The ring also drops the hull's sidelobe streaks: every range line the hull
    occupies (azimuth streaks) and the columns of its strongest scatterers (range
    streaks). Returns (hull, clean_ring, clutter_level).
    """
    ring_values = intensity[ring]
    if ring_values.size == 0:
        ring_values = intensity[corridor]
    clutter = float(np.median(ring_values)) if ring_values.size else 0.0
    snr = 10.0 ** (params.hull_snr_db / 10.0)
    inside = intensity[corridor]
    peak = float(inside.max()) if inside.size else 0.0
    threshold = max(
        clutter * snr, peak * 10.0 ** (-params.hull_dynamic_range_db / 10.0)
    )
    hull = corridor & (intensity > threshold)
    streak_rows = hull.any(axis=1)
    strongest = hull & (intensity > peak * 10.0 ** (-params.streak_db / 10.0))
    streak_cols = strongest.any(axis=0)
    clean_ring = (
        ring
        & (intensity <= clutter * snr)
        & ~streak_rows[:, None]
        & ~streak_cols[None, :]
    )
    return hull, clean_ring, clutter


# ----------------------------------------------------------------------------------
# Doppler diagnostic
# ----------------------------------------------------------------------------------


def doppler_amplification(
    geometry: ProductGeometry, t: float, point: NDArray[np.float64]
) -> float:
    """Return the error gain 1 / |1 - dfdc_dt / Ka| of a Doppler-centroid relocation.

    Diagnostic only; about 3000 on the WWGTZ2 Dwell fixture.
    """
    _, r = geometry.orbit.zero_doppler(point, t)
    gain = (
        1.0
        - geometry.doppler_centroid_rate(t, r) / geometry.kinematics(t, point).fm_rate
    )
    return 1.0 / abs(gain) if gain != 0 else float("inf")


# ----------------------------------------------------------------------------------
# Sub-aperture looks (map drift)
# ----------------------------------------------------------------------------------


def subaperture_looks(
    data: NDArray[np.complexfloating[Any]],
    n_looks: int,
    bandwidth: float,
    dt_col: float,
) -> list[tuple[float, NDArray[np.complex64]]]:
    """Split the azimuth spectrum into ``n_looks`` contiguous bands of ``bandwidth``.

    Returns (centre Doppler Hz, look image) per band, lowest Doppler first.
    """
    spectrum = np.fft.fft(data, axis=1)
    freqs = DOPPLER_SIGN * np.fft.fftfreq(data.shape[1], dt_col)
    edges = np.linspace(-bandwidth / 2.0, bandwidth / 2.0, n_looks + 1)
    looks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = (freqs >= lo) & (freqs < hi)
        looks.append((0.5 * (lo + hi), np.fft.ifft(spectrum * band[None, :], axis=1)))
    return looks


@dataclass
class _LocalFrame:
    """Geometry around the imaged ship, shared by the estimators."""

    row: float
    col: float
    t_img: float
    r_img: float
    p_img: NDArray[np.float64]
    dt_col: float
    kin: Kinematics
    row_spacing_m: float
    col_spacing_m: float
    enu_steps: NDArray[np.float64]
    ground_range_dir: NDArray[np.float64]
    along_track_dir: NDArray[np.float64]


def _local_frame(chip: SlcChip, row: float, col: float) -> _LocalFrame:
    t_img, r_img = chip.zero_doppler(row, col)
    p_img = chip.ecef(row, col)
    kin = chip.geometry.kinematics(t_img, p_img)
    lon, lat = chip.lonlat(row, col)
    basis = enu_basis(lon, lat)
    away = basis @ (p_img - kin.sat_position)
    ground_range_dir = away[:2] / np.linalg.norm(away[:2])
    vel = basis @ kin.sat_velocity
    along_track_dir = vel[:2] / np.linalg.norm(vel[:2])
    steps = chip.pixel_enu_steps(row, col)
    return _LocalFrame(
        row=row,
        col=col,
        t_img=t_img,
        r_img=r_img,
        p_img=p_img,
        dt_col=chip.azimuth_time_step(row, col),
        kin=kin,
        row_spacing_m=float(np.linalg.norm(steps[0])),
        col_spacing_m=float(np.linalg.norm(steps[1])),
        enu_steps=steps,
        ground_range_dir=ground_range_dir,
        along_track_dir=along_track_dir,
    )


def _curve_centre(chip: SlcChip, control_points: Sequence[Any]) -> tuple[float, float]:
    rows, cols = curve_pixels(control_points, chip.shape)
    mid = len(rows) // 2
    return float(rows[mid]), float(cols[mid])


def _find_targets(
    chip: SlcChip,
    params: RelocationParameters,
    frame: _LocalFrame,
    masks: CurveMasks,
) -> list[CurveTarget]:
    window = chip.data[masks.window]
    looks = subaperture_looks(
        window,
        params.n_looks,
        chip.geometry.processed_azimuth_bandwidth,
        frame.dt_col,
    )
    r_off, c_off = masks.window[0].start, masks.window[1].start
    targets = []
    for index, (f_center, look) in enumerate(looks):
        intensity = np.abs(look) ** 2
        hull, _, clutter = hull_and_clutter_masks(
            intensity, masks.corridor, masks.ring, params
        )
        n_pixels = int(hull.sum())
        if n_pixels == 0:
            continue
        w = intensity * hull
        total = float(w.sum())
        rr, cc = np.nonzero(hull)
        row = float(np.sum(rr * w[rr, cc]) / total) + r_off
        col = float(np.sum(cc * w[rr, cc]) / total) + c_off
        peak = float(intensity[hull].max())
        targets.append(
            CurveTarget(
                look=index,
                doppler_hz=f_center,
                slow_time_s=-f_center / frame.kin.fm_rate,
                row=row,
                col=col,
                x_fraction=col / max(chip.shape[1] - 1, 1),
                y_fraction=row / max(chip.shape[0] - 1, 1),
                intensity_db=10.0 * math.log10(peak / clutter)
                if clutter > 0
                else float("inf"),
                n_pixels=n_pixels,
            )
        )
    return targets


def find_targets_along_curve(
    chip: SlcChip,
    control_points: Sequence[Any],
    params: RelocationParameters | None = None,
) -> list[CurveTarget]:
    """Locate the ship in each sub-aperture look inside the corridor around the curve.

    Parameters
    ----------
    chip : SlcChip
        SLC window around the target.
    control_points : sequence
        The curve's four Bezier control points in order (start, c1, c2, end), as
        0..1 fractions of the chip (x along columns, y along rows).
    params : RelocationParameters or None
        Corridor width, look count and thresholds.

    Returns
    -------
    list of CurveTarget
        One entry per look in which the ship stands out from the clutter ring.
    """
    params = params or RelocationParameters()
    frame = _local_frame(chip, *_curve_centre(chip, control_points))
    masks = build_curve_masks(
        chip, control_points, params, frame.row_spacing_m, frame.col_spacing_m
    )
    return _find_targets(chip, params, frame, masks)


def map_drift(
    targets: Sequence[CurveTarget], frame: _LocalFrame
) -> tuple[float | None, float | None, float | None]:
    """Along-track velocity, fit R^2 and range walk rate from per-look positions.

    A Doppler-rate mismatch ``Ka_t = Ka * (1 - 2 v_a / V_eff)`` moves look k by
    ``tau_k * 2 v_a / V_eff`` in zero-Doppler time, so ``v_a = V_eff / 2 * slope``.
    """
    if len(targets) < 2:
        return None, None, None
    tau = np.array([t.slow_time_s for t in targets])
    t_img = np.array([(t.col - frame.col) * frame.dt_col for t in targets])
    ground_per_row = float(frame.enu_steps[0] @ frame.ground_range_dir)
    r_img = np.array([(t.row - frame.row) * ground_per_row for t in targets])
    weights = np.array([t.n_pixels for t in targets], dtype=float)
    slope, intercept = np.polyfit(tau, t_img, 1, w=np.sqrt(weights))
    resid = t_img - (slope * tau + intercept)
    total = np.sum(weights * (t_img - np.average(t_img, weights=weights)) ** 2)
    r2 = 1.0 - float(np.sum(weights * resid**2) / total) if total > 0 else 0.0
    walk = float(np.polyfit(tau, r_img, 1, w=np.sqrt(weights))[0])
    return 0.5 * frame.kin.v_eff * float(slope), r2, walk


def along_track_velocity(
    chip: SlcChip,
    control_points: Sequence[Any],
    params: RelocationParameters | None = None,
) -> tuple[float | None, float | None, list[CurveTarget]]:
    """Return the map-drift along-track velocity (m/s), its fit R^2 and the looks.

    For the optional v_a consistency check; its sign is not yet validated.
    """
    params = params or RelocationParameters()
    frame = _local_frame(chip, *_curve_centre(chip, control_points))
    masks = build_curve_masks(
        chip, control_points, params, frame.row_spacing_m, frame.col_spacing_m
    )
    targets = _find_targets(chip, params, frame, masks)
    v_along, r2, _ = map_drift(targets, frame)
    return v_along, r2, targets
