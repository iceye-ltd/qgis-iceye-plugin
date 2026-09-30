"""Moving-ship relocation for ICEYE Spotlight / Dwell SLCs, driven by a fitted curve.

The user bends the curve editor's Bezier along the imaged (displaced) ship. Everything
below uses only the SLC samples and the product metadata; no AIS, wakes or other
external references.

Pipeline (see ``relocate_mover``):

1. The curve, a corridor around it and a clutter ring outside it are rasterised in
   SLC pixel space (file layout: rows = range samples, columns = azimuth lines).
2. Bright hull pixels inside the corridor are kept (threshold against ring clutter and
   against the hull peak, which drops most sidelobe energy); bright pixels in the ring
   (streaks, other targets) are dropped from the clutter reference.
3. Doppler of hull and ring along azimuth, by the band-limited spectral centroid
   (default) or the correlation (Madsen) estimator ``C = sum(s[n+1] * conj(s[n]))``,
   corrected for truncation by the processing window. The reference Doppler comes
   from the metadata Doppler centroid (primary) and from the ring (cross-check); each
   is solved at the TRUE position by Newton iteration together with the zero-Doppler
   geocoding.
4. ``df -> v_r = lambda*df/2 -> dt = df/Ka``; the true zero-Doppler time is
   ``t_img - dt`` at the imaged slant range, geocoded onto the sea surface.
5. Sub-aperture looks along the curve (``find_targets_along_curve``) give a map-drift
   estimate of along-track velocity; with the hull heading this is a second, independent
   relocation that stays usable where the Doppler centroid is not.

Conventions and caveats found on real products (WWGTZ2 SLED fixture):

* Azimuth sample spacing is ``1 / iceye:processing_prf``; the time direction along file
  columns is taken from the geolocation (it decreases with column index there).
* The azimuth spectrum is flat over the processed bandwidth and centred on zero, so the
  SLC is treated as basebanded to the local Doppler centroid (``slc_basebanded``).
* In Spotlight / Dwell the Doppler centroid moves with azimuth position at almost exactly
  the azimuth FM rate (4918 vs 4920 Hz/s on the fixture). A mover's centroid then equals
  that of the clutter at its imaged position to first order, and solving for the true
  position amplifies any Doppler error by ``1 / |1 - dfdc_dt / Ka|`` (about 3500 on the
  fixture). This is reported as ``amplification`` and flagged; the map-drift estimate
  is then used for the output position when it is valid.
* Over the 17.6 deg Dwell aperture the Doppler of stationary port scatterers varies by
  kHz with their aspect, and a ring of flat-spectrum clutter is only good to ~50 Hz
  (~0.8 m/s); Doppler estimates carry that scatter.
* Map drift assumes the hull reflects over the whole aperture. A straight structure
  whose specular point slides with look angle drifts like a mover, so the curve must
  enclose an isolated ship at sea.
* The Doppler sign (``DOPPLER_SIGN``) relative to the phase convention of
  ``core.raster.read_slc_layer`` and the map-drift sign are not yet verified against a
  mover with known motion; every estimate carries ``FLAG_SIGN_UNVERIFIED``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
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

FLAG_SIGN_UNVERIFIED = "sign_unverified"
FLAG_ILL_CONDITIONED = "doppler_ill_conditioned"
FLAG_REFERENCE_DISAGREEMENT = "reference_disagreement"
FLAG_NEAR_BANDWIDTH_EDGE = "near_bandwidth_edge"
FLAG_NEAR_AMBIGUITY = "near_doppler_ambiguity"
FLAG_LOW_COHERENCE = "low_coherence"
FLAG_FEW_SAMPLES = "few_samples"
FLAG_NOT_CONVERGED = "not_converged"
FLAG_HEADING_UNRELIABLE = "heading_unreliable"
FLAG_HEADING_NEAR_AZIMUTH = "heading_near_azimuth"
FLAG_HEADING_NEAR_RANGE = "heading_near_range"
FLAG_FEW_LOOKS = "few_looks"
FLAG_NO_VALID_ESTIMATE = "no_valid_estimate"
FLAG_MODE_NOT_SPOTLIGHT = "mode_not_spotlight"
FLAG_RING_UNRELIABLE = "ring_unreliable"

METHOD_DOPPLER = "doppler"
METHOD_MAP_DRIFT = "map_drift"
METHOD_NONE = "none"


# ----------------------------------------------------------------------------------
# Parameters and results
# ----------------------------------------------------------------------------------


@dataclass
class RelocationParameters:
    """Tunable settings of the relocation pipeline (distances in ground metres)."""

    corridor_half_width_m: float = 15.0
    ring_gap_m: float = 10.0
    ring_width_m: float = 30.0
    hull_snr_db: float = 10.0
    hull_dynamic_range_db: float = 25.0
    hull_dilation_m: float = 2.0
    min_ring_coherence: float = 0.05
    n_looks: int = 5
    max_amplification: float = 5.0
    reference_disagreement_mps: float = 0.5
    min_coherence: float = 0.1
    min_samples: int = 50
    min_looks: int = 3
    min_heading_cosine: float = 0.2
    streak_db: float = 15.0
    doppler_estimator: str = "centroid"
    slc_basebanded: bool = True
    max_iterations: int = 10
    position_tolerance_m: float = 0.1


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


@dataclass
class DopplerEstimate:
    """Correlation Doppler estimate over a mask."""

    frequency_hz: float
    coherence: float
    n_samples: int


@dataclass
class DopplerSolution:
    """Relative Doppler offset solved at the true position for one reference."""

    reference: str
    df_hz: float
    v_r: float
    dt_s: float
    dx_m: float
    converged: bool


@dataclass
class MoverEstimate:
    """Outcome of ``relocate_mover``; velocities in m/s, v_r > 0 towards the radar."""

    method: str
    imaged_lonlat: tuple[float, float]
    true_lonlat: tuple[float, float]
    v_r: float
    v_gr: float
    df_hz: float
    dt_s: float
    dx_m: float
    speed: float | None
    heading_deg: float | None
    confidence: float
    flags: list[str]
    f_ship_hz: float
    f_ref_ring_hz: float
    f_ref_meta_hz: float
    coherence: float
    n_samples: int
    amplification: float
    doppler_spread_hz: float
    doppler_meta: DopplerSolution | None
    doppler_ring: DopplerSolution | None
    v_along: float | None
    v_along_fit_r2: float | None
    range_walk_rate: float | None
    hull_axis_deg: float | None
    targets: list[CurveTarget] = field(default_factory=list)
    hull_mask: NDArray[np.bool_] | None = None
    ring_mask: NDArray[np.bool_] | None = None
    iterations: int = 0

    def attributes(self) -> dict[str, Any]:
        """Flat attribute dict for the output point layer."""
        return {
            "method": self.method,
            "v_r": self.v_r,
            "v_gr": self.v_gr,
            "df_hz": self.df_hz,
            "dt_s": self.dt_s,
            "dx_m": self.dx_m,
            "speed": self.speed,
            "heading": self.heading_deg,
            "v_along": self.v_along,
            "confidence": self.confidence,
            "coherence": self.coherence,
            "n_samples": self.n_samples,
            "amplif": self.amplification,
            "f_ship": self.f_ship_hz,
            "f_ref_ring": self.f_ref_ring_hz,
            "f_ref_meta": self.f_ref_meta_hz,
            "dopp_sprd": self.doppler_spread_hz,
            "flags": ",".join(self.flags),
        }


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


def gcp_pixel_to_lonlat(source_path: str) -> PixelToLonLat:
    """Return a (file col, file row) -> (lon, lat) function from the GCP TPS model."""
    dataset = gdal.Open(source_path)
    if dataset is None:
        raise ValueError(f"Failed to open {source_path}")
    transformer = gdal.Transformer(dataset, None, ["METHOD=GCP_TPS"])

    def transform(col: float, row: float) -> tuple[float, float]:
        ok, point = transformer.TransformPoint(0, float(col), float(row), 0.0)
        if not ok:
            raise ValueError(f"Pixel ({col}, {row}) could not be geolocated")
        return float(point[0]), float(point[1])

    # Keep the dataset alive for as long as the transformer is used.
    transform.dataset = dataset  # type: ignore[attr-defined]
    return transform


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
# Doppler
# ----------------------------------------------------------------------------------


def dilate_columns(mask: NDArray[np.bool_], radius: int) -> NDArray[np.bool_]:
    """Binary dilation of a mask by ``radius`` pixels along columns (azimuth)."""
    if radius <= 0:
        return mask.copy()
    padded = np.pad(mask.astype(np.int32), ((0, 0), (radius + 1, radius)))
    csum = np.cumsum(padded, axis=1)
    window = csum[:, 2 * radius + 1 :] - csum[:, : -(2 * radius + 1)]
    return window > 0


def correlation_doppler(
    data: NDArray[np.complexfloating[Any]],
    mask: NDArray[np.bool_],
    dt_col: float,
) -> DopplerEstimate:
    """Correlation Doppler estimator along columns (azimuth) over a mask.

    ``dt_col`` is the signed time per column, so the result is the Doppler in time.
    """
    pair = mask[:, 1:] & mask[:, :-1]
    s0 = data[:, :-1][pair]
    s1 = data[:, 1:][pair]
    n = int(s0.size)
    if n == 0:
        return DopplerEstimate(0.0, 0.0, 0)
    c = np.sum(s1 * np.conj(s0))
    power = 0.5 * float(np.sum(np.abs(s0) ** 2 + np.abs(s1) ** 2))
    f = DOPPLER_SIGN * float(np.angle(c)) / (2.0 * math.pi * dt_col)
    return DopplerEstimate(f, float(abs(c)) / power if power > 0 else 0.0, n)


def spectral_centroid_doppler(
    data: NDArray[np.complexfloating[Any]],
    mask: NDArray[np.bool_],
    dt_col: float,
    bandwidth: float,
) -> float:
    """Power-weighted mean azimuth frequency (Hz) inside ``[-bandwidth/2, bandwidth/2]``.

    Valid for a basebanded SLC whose processed band leaves a gap at +-PRF/2, so the
    linear mean cannot wrap. The masked range lines are Fourier transformed whole.
    """
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return 0.0
    segments = np.where(mask[rows], data[rows], 0)
    power = np.sum(np.abs(np.fft.fft(segments, axis=1)) ** 2, axis=0)
    freqs = DOPPLER_SIGN * np.fft.fftfreq(data.shape[1], dt_col)
    band = np.abs(freqs) <= bandwidth / 2.0
    total = float(np.sum(power[band]))
    return float(np.sum(power[band] * freqs[band]) / total) if total > 0 else 0.0


def estimate_doppler(
    data: NDArray[np.complexfloating[Any]],
    mask: NDArray[np.bool_],
    dt_col: float,
    bandwidth: float,
    method: str = "centroid",
) -> DopplerEstimate:
    """Doppler over a mask by ``"correlation"`` (Madsen) or ``"centroid"``.

    The coherence is always the lag-one correlation coherence. When the processed band
    nearly fills the PRF (86 % on ICEYE Dwell) the lag-one coherence of any target is
    only about sinc(B / PRF) ~ 0.16 and its angle is easily pulled by scene structure,
    which is why the band-limited spectral centroid is the default.
    """
    corr = correlation_doppler(data, mask, dt_col)
    if method == "correlation" or corr.n_samples == 0:
        return corr
    if method != "centroid":
        raise ValueError(f"Unknown Doppler estimator {method!r}")
    return DopplerEstimate(
        spectral_centroid_doppler(data, mask, dt_col, bandwidth),
        corr.coherence,
        corr.n_samples,
    )


def doppler_spread(
    data: NDArray[np.complexfloating[Any]],
    mask: NDArray[np.bool_],
    dt_col: float,
    min_pairs: int = 8,
) -> float:
    """Power-weighted spread (Hz) of per-range-row Doppler across the hull.

    Rows see different parts of the hull, so the spread bounds rotational motion.
    """
    pair = mask[:, 1:] & mask[:, :-1]
    prod = data[:, 1:] * np.conj(data[:, :-1])
    c_rows = np.where(pair, prod, 0).sum(axis=1)
    counts = pair.sum(axis=1)
    keep = counts >= min_pairs
    if keep.sum() < 2:
        return 0.0
    f_rows = DOPPLER_SIGN * np.angle(c_rows[keep]) / (2.0 * math.pi * dt_col)
    w = np.abs(c_rows[keep])
    mean = np.angle(np.sum(c_rows[keep])) * DOPPLER_SIGN / (2.0 * math.pi * dt_col)
    return float(math.sqrt(np.sum(w * (f_rows - mean) ** 2) / np.sum(w)))


def truncation_bias_curve(
    processed_bandwidth: float,
    sampling_rate: float,
    offsets: NDArray[np.float64],
    target_bandwidth: float | None = None,
    n: int = 8192,
) -> NDArray[np.float64]:
    """Correlation-estimator reading for spectra offset by ``offsets`` (Hz).

    A point target with a flat spectrum of ``target_bandwidth`` (default: the processed
    bandwidth) centred at each offset is cut to the processing window
    ``[-B/2, B/2]``, and the lag-one correlation of what remains is evaluated.
    """
    bt = processed_bandwidth if target_bandwidth is None else target_bandwidth
    f = np.fft.fftfreq(n, 1.0 / sampling_rate)
    window = np.abs(f) <= processed_bandwidth / 2.0
    out = np.empty(len(offsets))
    for i, df in enumerate(offsets):
        support = window & (np.abs(f - df) <= bt / 2.0)
        if not support.any():
            out[i] = np.nan
            continue
        c = np.sum(np.exp(2j * math.pi * f[support] / sampling_rate))
        out[i] = np.angle(c) * sampling_rate / (2.0 * math.pi)
    return out


def correct_truncation(
    measured: float, processed_bandwidth: float, sampling_rate: float
) -> float:
    """Invert ``truncation_bias_curve`` for a measured relative Doppler offset."""
    offsets = np.linspace(-0.95, 0.95, 191) * processed_bandwidth
    readings = truncation_bias_curve(processed_bandwidth, sampling_rate, offsets)
    ok = np.isfinite(readings)
    offsets, readings = offsets[ok], readings[ok]
    order = np.argsort(readings)
    return float(np.interp(measured, readings[order], offsets[order]))


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
        SLC window the curve editor shows.
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


# ----------------------------------------------------------------------------------
# Heading and relocation
# ----------------------------------------------------------------------------------


def hull_axis(
    hull: NDArray[np.bool_],
    intensity: NDArray[np.floating[Any]],
    enu_steps: NDArray[np.float64],
) -> tuple[NDArray[np.float64] | None, float]:
    """Principal hull axis (unit ground EN vector) and its elongation (major/minor)."""
    rr, cc = np.nonzero(hull)
    if rr.size < 3:
        return None, 0.0
    w = intensity[rr, cc]
    xy = rr[:, None] * enu_steps[0][None, :] + cc[:, None] * enu_steps[1][None, :]
    mean = np.average(xy, axis=0, weights=w)
    d = xy - mean
    cov = (d * w[:, None]).T @ d / np.sum(w)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, 1]
    elong = math.sqrt(vals[1] / vals[0]) if vals[0] > 0 else float("inf")
    return major / np.linalg.norm(major), elong


def _heading_deg(direction: NDArray[np.float64]) -> float:
    """Compass heading (deg from North, clockwise) of an EN vector."""
    return math.degrees(math.atan2(direction[0], direction[1])) % 360.0


def _dedrifted_look_sum(
    looks: list[tuple[float, NDArray[np.complex64]]],
    targets: Sequence[CurveTarget],
) -> NDArray[np.float64] | None:
    """Incoherent sum of the looks, each shifted onto the mean look position.

    Removing the drift leaves a sharper hull for the heading fit.
    """
    if not targets:
        return None
    mean_col = float(np.mean([t.col for t in targets]))
    total = np.zeros(looks[0][1].shape)
    for t in targets:
        shift = int(round(mean_col - t.col))
        total += np.roll(np.abs(looks[t.look][1]) ** 2, shift, axis=1)
    return total


def _solve_doppler(
    chip: SlcChip,
    frame: _LocalFrame,
    f_ship: float,
    reference: Callable[[float], float],
    name: str,
    params: RelocationParameters,
) -> tuple[DopplerSolution, NDArray[np.float64], int]:
    """Newton iteration of reference Doppler at the true position (steps 3 to 5).

    Solves ``df = f_ship - f_ref(t_img - df / Ka)`` with Ka re-evaluated at each new
    true position, until that position moves less than ``position_tolerance_m``.
    """
    geometry = chip.geometry
    ka = frame.kin.fm_rate
    df = f_ship - reference(frame.t_img)
    p_true = frame.p_img
    converged = False
    iterations = 0
    for iterations in range(1, params.max_iterations + 1):
        t_true = frame.t_img - df / ka
        h = 1e-4
        slope = (reference(t_true + h) - reference(t_true - h)) / (2.0 * h)
        g = df - (f_ship - reference(t_true))
        gp = 1.0 - slope / ka
        df = df - g / gp if gp != 0 else df
        t_true = frame.t_img - df / ka
        p_new = geometry.orbit.geocode(
            frame.r_img, t_true, geometry.scene_height, p_true
        )
        ka = geometry.kinematics(t_true, p_new).fm_rate
        moved = float(np.linalg.norm(p_new - p_true))
        p_true = p_new
        if moved < params.position_tolerance_m and abs(g) < 1e-6 * max(1.0, abs(df)):
            converged = True
            break
    dt = df / ka
    solution = DopplerSolution(
        reference=name,
        df_hz=df,
        v_r=geometry.wavelength * df / 2.0,
        dt_s=dt,
        dx_m=frame.kin.v_ground * dt,
        converged=converged,
    )
    return solution, p_true, iterations


def relocate_mover(
    chip: SlcChip,
    control_points: Sequence[Any],
    params: RelocationParameters | None = None,
) -> MoverEstimate:
    """Estimate the true position of the ship imaged along the curve.

    See the module docstring for the pipeline; velocities are in m/s with ``v_r > 0``
    towards the radar and ``v_gr = v_r / sin(theta_inc)``.
    """
    params = params or RelocationParameters()
    geometry = chip.geometry
    flags = [FLAG_SIGN_UNVERIFIED]
    if geometry.instrument_mode not in ("spotlight", "dwell"):
        flags.append(FLAG_MODE_NOT_SPOTLIGHT)

    frame0 = _local_frame(chip, *_curve_centre(chip, control_points))
    masks = build_curve_masks(
        chip, control_points, params, frame0.row_spacing_m, frame0.col_spacing_m
    )
    window = chip.data[masks.window]
    intensity = np.abs(window) ** 2
    hull, ring, _ = hull_and_clutter_masks(
        intensity, masks.corridor, masks.ring, params
    )
    r_off, c_off = masks.window[0].start, masks.window[1].start

    # Imaged position: intensity centroid of the hull.
    if hull.any():
        rr, cc = np.nonzero(hull)
        w = intensity[rr, cc]
        row_c = float(np.sum(rr * w) / np.sum(w)) + r_off
        col_c = float(np.sum(cc * w) / np.sum(w)) + c_off
    else:
        row_c, col_c = frame0.row, frame0.col
    frame = _local_frame(chip, row_c, col_c)
    imaged_lonlat = chip.lonlat(row_c, col_c)

    # Doppler on hull and ring. The hull is widened along azimuth so the estimator
    # sees the scatterers' sidelobes too; a lag-one sum over the mainlobe alone is
    # biased by the sinc's sign changes.
    dilation = int(round(params.hull_dilation_m / frame.col_spacing_m))
    doppler_mask = dilate_columns(hull, dilation) & masks.corridor
    bandwidth = geometry.processed_azimuth_bandwidth
    ship = estimate_doppler(
        window, doppler_mask, frame.dt_col, bandwidth, params.doppler_estimator
    )
    ring_est = estimate_doppler(
        window, ring, frame.dt_col, bandwidth, params.doppler_estimator
    )
    spread = doppler_spread(window, doppler_mask, frame.dt_col)
    dc_img = geometry.doppler_centroid(frame.t_img, frame.r_img)
    base = dc_img if params.slc_basebanded else 0.0
    f_ship_raw = ship.frequency_hz + base
    if ring_est.coherence >= params.min_ring_coherence:
        f_ring = ring_est.frequency_hz + base
    else:
        flags.append(FLAG_RING_UNRELIABLE)
        f_ring = dc_img

    # Truncation bias acts on the offset from the processing window, which is
    # centred on the processor's Doppler centroid at the imaged position.
    sampling = 1.0 / geometry.azimuth_time_spacing
    relative = f_ship_raw - dc_img
    f_ship = dc_img + correct_truncation(relative, bandwidth, sampling)
    if abs(relative) > 0.4 * bandwidth:
        flags.append(FLAG_NEAR_BANDWIDTH_EDGE)
    if ship.coherence < params.min_coherence:
        flags.append(FLAG_LOW_COHERENCE)
    if ship.n_samples < params.min_samples:
        flags.append(FLAG_FEW_SAMPLES)

    dcdt = geometry.doppler_centroid_rate(frame.t_img, frame.r_img)
    gain = 1.0 - dcdt / frame.kin.fm_rate
    amplification = 1.0 / abs(gain) if gain != 0 else float("inf")
    if amplification > params.max_amplification:
        flags.append(FLAG_ILL_CONDITIONED)

    def meta_reference(t: float) -> float:
        return geometry.doppler_centroid(t, frame.r_img)

    ring_bias = f_ring - dc_img

    def ring_reference(t: float) -> float:
        return geometry.doppler_centroid(t, frame.r_img) + ring_bias

    sol_meta, p_meta, it_meta = _solve_doppler(
        chip, frame, f_ship, meta_reference, "metadata", params
    )
    sol_ring, _, _ = _solve_doppler(chip, frame, f_ship, ring_reference, "ring", params)
    # Once ill-conditioned, both solutions are amplified noise; skip their checks.
    if FLAG_ILL_CONDITIONED not in flags:
        if abs(sol_meta.v_r - sol_ring.v_r) > params.reference_disagreement_mps:
            flags.append(FLAG_REFERENCE_DISAGREEMENT)
        if not (sol_meta.converged and sol_ring.converged):
            flags.append(FLAG_NOT_CONVERGED)
        span = geometry.wavelength * geometry.acquisition_prf / 2.0
        if abs(sol_meta.v_r) > 0.25 * span:
            flags.append(FLAG_NEAR_AMBIGUITY)

    # Sub-aperture looks: map drift and a drift-free hull for the heading.
    targets = _find_targets(chip, params, frame, masks)
    v_along, r2, walk = map_drift(targets, frame)
    looks = subaperture_looks(window, params.n_looks, bandwidth, frame.dt_col)
    look_sum = _dedrifted_look_sum(looks, targets)
    if look_sum is None:
        look_sum = intensity
    hull_looks, _, _ = hull_and_clutter_masks(
        look_sum, masks.corridor, masks.ring, params
    )
    axis, elongation = hull_axis(hull_looks, look_sum, frame.enu_steps)
    if axis is None or elongation < 2.0:
        flags.append(FLAG_HEADING_UNRELIABLE)
    if len(targets) < params.min_looks:
        flags.append(FLAG_FEW_LOOKS)

    sin_inc = math.sin(math.radians(geometry.incidence_angle_deg))
    towards_radar = -frame.ground_range_dir
    doppler_valid = (
        FLAG_ILL_CONDITIONED not in flags
        and FLAG_LOW_COHERENCE not in flags
        and FLAG_FEW_SAMPLES not in flags
    )
    cos_along = float(axis @ frame.along_track_dir) if axis is not None else 0.0
    if v_along is not None and axis is not None:
        if abs(cos_along) < params.min_heading_cosine:
            flags.append(FLAG_HEADING_NEAR_RANGE)
    drift_valid = (
        v_along is not None
        and axis is not None
        and FLAG_HEADING_UNRELIABLE not in flags
        and len(targets) >= params.min_looks
        and abs(cos_along) >= params.min_heading_cosine
    )

    method = METHOD_NONE
    v_r = 0.0
    speed: float | None = None
    heading: float | None = None
    confidence = 0.0
    p_true = frame.p_img
    solution: DopplerSolution | None = None
    iterations = 0
    if doppler_valid:
        method = METHOD_DOPPLER
        solution, p_true, iterations = sol_meta, p_meta, it_meta
        v_r = solution.v_r
        if axis is not None:
            cos_alpha = float(axis @ towards_radar)
            if abs(cos_alpha) < params.min_heading_cosine:
                flags.append(FLAG_HEADING_NEAR_AZIMUTH)
            else:
                v_gr = v_r / sin_inc
                speed = abs(v_gr / cos_alpha)
                heading = _heading_deg(axis * math.copysign(1.0, v_gr * cos_alpha))
        confidence = (
            ship.coherence
            * min(1.0, ship.n_samples / (4.0 * params.min_samples))
            / max(1.0, amplification)
        )
    elif drift_valid:
        method = METHOD_MAP_DRIFT
        speed = abs(v_along / cos_along)
        direction = axis * math.copysign(1.0, v_along * cos_along)
        heading = _heading_deg(direction)
        v_r = speed * float(direction @ towards_radar) * sin_inc
        # Zero reference: df is known, only the geocoding iteration remains.
        solution, p_true, iterations = _solve_doppler(
            chip,
            frame,
            2.0 * v_r / geometry.wavelength,
            lambda _t: 0.0,
            "map_drift",
            params,
        )
        confidence = max(0.0, r2 or 0.0) * min(1.0, len(targets) / params.n_looks)
        confidence *= min(1.0, abs(cos_along))
    else:
        flags.append(FLAG_NO_VALID_ESTIMATE)

    lat, lon, _ = ecef_to_geodetic(p_true)
    hull_full = np.zeros(chip.shape, dtype=bool)
    hull_full[masks.window] = hull
    ring_full = np.zeros(chip.shape, dtype=bool)
    ring_full[masks.window] = ring
    return MoverEstimate(
        method=method,
        imaged_lonlat=imaged_lonlat,
        true_lonlat=(lon, lat),
        v_r=v_r,
        v_gr=v_r / sin_inc,
        df_hz=solution.df_hz if solution else 0.0,
        dt_s=solution.dt_s if solution else 0.0,
        dx_m=solution.dx_m if solution else 0.0,
        speed=speed,
        heading_deg=heading,
        confidence=float(min(max(confidence, 0.0), 1.0)),
        flags=flags,
        f_ship_hz=f_ship,
        f_ref_ring_hz=f_ring,
        f_ref_meta_hz=dc_img,
        coherence=ship.coherence,
        n_samples=ship.n_samples,
        amplification=amplification,
        doppler_spread_hz=spread,
        doppler_meta=sol_meta,
        doppler_ring=sol_ring,
        v_along=v_along,
        v_along_fit_r2=r2,
        range_walk_rate=walk,
        hull_axis_deg=_heading_deg(axis) % 180.0 if axis is not None else None,
        targets=targets,
        hull_mask=hull_full,
        ring_mask=ring_full,
        iterations=iterations,
    )
