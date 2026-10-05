"""Moving-target relocation for ICEYE Spotlight / Dwell SLCs from in-image constraints.

A mover is imaged on its own range line, displaced along track by
dt = t_img - t_true = R * v_r / V_eff^2 (v_r positive towards the radar). The true
position along that line comes from a road, rail, bridge deck or wake the user
clicks. The maths is documented in help/sar-mover-relocator-maths.md.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np

from .target_finder import (
    PixelToLonLat,
    ProductGeometry,
    RelocationParameters,
    SlcChip,
    build_curve_masks,
    curve_pixels,
    ecef_to_geodetic,
    enu_basis,
    hull_and_clutter_masks,
    lonlat_to_ecef,
)
from .typing_compat import NDArray

MPS_TO_KMH = 3.6
MPS_TO_KNOTS = 3600.0 / 1852.0

# Signs of the band / intersection chain are checked in simulation by
# test_mover_relocation.TestSignValidation.
SIGN_VALIDATED = True

FLAG_OUTSIDE_BAND = "outside_band"
FLAG_IMPLAUSIBLE_SPEED = "implausible_speed"
FLAG_PROBABLY_STATIONARY = "probably_stationary"
FLAG_CONSTRAINT_PARALLEL = "constraint_parallel_to_track"
FLAG_V_A_INCONSISTENT = "v_a_inconsistent"
FLAG_BAND_CLIPPED = "band_clipped"
FLAG_HEADING_UNKNOWN = "heading_unknown"
FLAG_HEADING_ESTIMATED = "heading_estimated"
SOFT_FLAGS = frozenset(
    {
        FLAG_BAND_CLIPPED,
        FLAG_PROBABLY_STATIONARY,
        FLAG_HEADING_UNKNOWN,
        FLAG_HEADING_ESTIMATED,
    }
)

MODE_TWO_CLICK = "two_click"
MODE_SINGLE_CLICK = "single_click"
MODE_SINGLE_CLICK_AUTO = "single_click_auto"

INDICATOR_GREEN = "green"
INDICATOR_AMBER = "amber"
INDICATOR_RED = "red"


@dataclass(frozen=True)
class TargetClass:
    """Target type; its maximum speed sets the band length and the speed check."""

    name: str
    v_max_mps: float


TARGET_CLASSES = {
    "car": TargetClass("car", 40.0),
    "train": TargetClass("train", 90.0),
    "ship": TargetClass("ship", 15.0),
}


@dataclass
class RelocationSettings:
    """Band, intersection, uncertainty and plausibility settings (metres, m/s)."""

    band_margin_m: float = 10.0
    tick_step_mps: float = 5.0
    band_samples: int = 41
    v_min_mps: float = 0.5
    min_constraint_track_angle_deg: float = 15.0
    sigma_centroid_m: float = 2.0
    sigma_click_m: float = 2.0
    constraint_width_m: float = 10.0
    range_residual: bool = False
    allow_extrapolation: bool = False
    axis_radius_m: float = 50.0
    axis_cell_m: float = 2.0
    # Speckle gives ~0.1, clean roads and wakes ~0.8-0.9.
    min_axis_coherence: float = 0.5
    axis_half_length_m: float = 40.0
    # Off until the map-drift v_a sign is validated.
    v_a_check: bool = False
    v_a_k_sigma: float = 3.0


class ConstraintError(ValueError):
    """The constraint clicks cannot be intersected with the band."""


@dataclass
class LocalGeometry:
    """Zero-Doppler geometry of an Earth-fixed point.

    ground_range_dir (away from the satellite) and along_track_dir (platform
    velocity) are horizontal (East, North) unit vectors.
    """

    slant_range: float
    v_eff2: float
    v_ground: float
    sin_incidence: float
    ground_range_dir: NDArray[np.float64]
    along_track_dir: NDArray[np.float64]
    enu: NDArray[np.float64]


def local_geometry(
    geometry: ProductGeometry, t: float, point: NDArray[np.float64]
) -> LocalGeometry:
    """Compute V_eff^2, V_g, local incidence and horizontal directions at a point.

    Parameters
    ----------
    geometry : ProductGeometry
        Orbit and product metadata.
    t : float
        Zero-Doppler time of *point* (s from the zero-Doppler start).
    point : ndarray
        ECEF position (m).

    Returns
    -------
    LocalGeometry
    """
    kin = geometry.kinematics(t, point)
    lat, lon, _ = ecef_to_geodetic(point)
    enu = enu_basis(lon, lat)
    away = enu @ (point - kin.sat_position)
    vel = enu @ kin.sat_velocity
    cos_inc = -away[2] / float(np.linalg.norm(away))
    return LocalGeometry(
        slant_range=kin.slant_range,
        v_eff2=kin.v_eff**2,
        v_ground=kin.v_ground,
        sin_incidence=math.sqrt(max(0.0, 1.0 - cos_inc**2)),
        ground_range_dir=away[:2] / np.linalg.norm(away[:2]),
        along_track_dir=vel[:2] / np.linalg.norm(vel[:2]),
        enu=enu,
    )


def _lonlat(point: NDArray[np.float64]) -> tuple[float, float]:
    """(lon, lat) of an ECEF point."""
    lat, lon, _ = ecef_to_geodetic(point)
    return lon, lat


def _heading_deg(en: NDArray[np.float64]) -> float:
    """Compass heading (deg from North, clockwise) of an EN vector."""
    return math.degrees(math.atan2(en[0], en[1])) % 360.0


def _mid_time(geometry: ProductGeometry) -> float:
    """Zero-Doppler time in the middle of the scene, as a Newton start value."""
    if len(geometry.dc_times):
        return 0.5 * float(geometry.dc_times[0] + geometry.dc_times[-1])
    return 0.0


@dataclass
class ImagedTarget:
    """Imaged (displaced) target in radar coordinates.

    height is the surface the display geocoding uses, so clicks on the map land on
    the pixels the user sees.
    """

    slant_range: float
    time: float
    position: NDArray[np.float64]
    height: float
    half_extent_m: float = 0.0

    @classmethod
    def from_ecef(
        cls,
        geometry: ProductGeometry,
        point: NDArray[np.float64],
        height: float,
        half_extent_m: float = 0.0,
    ) -> ImagedTarget:
        """Create a target at an ECEF point on the display surface."""
        point = np.asarray(point, np.float64)
        t, r = geometry.orbit.zero_doppler(point, _mid_time(geometry))
        return cls(r, t, point, height, half_extent_m)

    @property
    def lonlat(self) -> tuple[float, float]:
        """Imaged position (lon, lat)."""
        return _lonlat(self.position)


def locate_imaged_target(
    chip: SlcChip,
    control_points: Sequence[Any],
    height: float,
    params: RelocationParameters | None = None,
) -> ImagedTarget:
    """Find the intensity-weighted hull centroid around a click in an SLC chip.

    Parameters
    ----------
    chip : SlcChip
        SLC window around the click.
    control_points : sequence
        Four (x, y) Bezier points as 0..1 chip fractions; for a click pass the click
        fraction four times, which makes the corridor a disc.
    height : float
        Display surface height (m).
    params : RelocationParameters or None
        Corridor and hull thresholds.

    Returns
    -------
    ImagedTarget
        Hull centroid, with half the hull's ground-range extent as half_extent_m.
    """
    params = params or RelocationParameters()
    rows, cols = curve_pixels(control_points, chip.shape)
    mid = len(rows) // 2
    steps = chip.pixel_enu_steps(float(rows[mid]), float(cols[mid]))
    row_m, col_m = (float(np.linalg.norm(s)) for s in steps)
    masks = build_curve_masks(chip, control_points, params, row_m, col_m)
    intensity = np.abs(chip.data[masks.window]) ** 2
    hull, _, _ = hull_and_clutter_masks(intensity, masks.corridor, masks.ring, params)
    rr, cc = np.nonzero(hull)
    if rr.size == 0:
        lon, lat = chip.lonlat(float(rows[mid]), float(cols[mid]))
        return ImagedTarget.from_ecef(
            chip.geometry, lonlat_to_ecef(lon, lat, height), height
        )

    w = intensity[rr, cc]
    row = float(np.sum(rr * w) / np.sum(w)) + masks.window[0].start
    col = float(np.sum(cc * w) / np.sum(w)) + masks.window[1].start
    lon, lat = chip.lonlat(row, col)
    target = ImagedTarget.from_ecef(
        chip.geometry, lonlat_to_ecef(lon, lat, height), height
    )
    g_hat = local_geometry(chip.geometry, target.time, target.position).ground_range_dir
    across = rr * float(steps[0] @ g_hat) + cc * float(steps[1] @ g_hat)
    target.half_extent_m = 0.5 * float(np.ptp(across))
    return target


@dataclass
class BandTick:
    """Point on the band where v_r has a round value (signed, towards radar > 0)."""

    v_r: float
    t: float
    lonlat: tuple[float, float]
    v_ground_min: float
    label: str


@dataclass
class Band:
    """Possible true positions of a target: its range line within +-dt_max."""

    target: ImagedTarget
    target_class: TargetClass
    dt_max: float
    dx_max_m: float
    half_width_m: float
    half_width_slant_m: float
    t_samples: list[float]
    near_lonlat: list[tuple[float, float]]
    far_lonlat: list[tuple[float, float]]
    ticks: list[BandTick]
    clipped: bool
    v_eff2: float
    v_ground: float
    sin_incidence: float

    @property
    def polygon_lonlat(self) -> list[tuple[float, float]]:
        """Closed outer ring: near edge forward, far edge back."""
        ring = list(self.near_lonlat) + list(reversed(self.far_lonlat))
        return ring + ring[:1]


def band_for_target(
    geometry: ProductGeometry,
    target: ImagedTarget,
    target_class: TargetClass,
    settings: RelocationSettings | None = None,
    time_limits: tuple[float, float] | None = None,
) -> Band:
    """Build the possible-location band and its |v_r| ticks for a target.

    The band spans dt_max = R_img * v_max * sin(theta_inc) / V_eff^2 either side of
    t_img; its edges are range lines offset by the target half-extent plus
    band_margin_m in ground range.

    Parameters
    ----------
    geometry : ProductGeometry
        Orbit and product metadata.
    target : ImagedTarget
        Imaged target.
    target_class : TargetClass
        Sets v_max.
    settings : RelocationSettings or None
        Margin, tick step and sample count.
    time_limits : tuple of float or None
        Zero-Doppler span of the image (image_time_limits). Samples and ticks
        outside it are dropped and the band is flagged clipped.

    Returns
    -------
    Band
    """
    settings = settings or RelocationSettings()
    orbit = geometry.orbit
    local = local_geometry(geometry, target.time, target.position)
    r_img, t_img, h = target.slant_range, target.time, target.height
    dt_per_vr = r_img / local.v_eff2
    v_r_max = target_class.v_max_mps * local.sin_incidence
    dt_max = v_r_max * dt_per_vr
    half_width = target.half_extent_m + settings.band_margin_m
    half_width_slant = half_width * local.sin_incidence

    t_lo, t_hi = t_img - dt_max, t_img + dt_max
    clipped = False
    if time_limits is not None:
        lo, hi = min(time_limits), max(time_limits)
        clipped = t_lo < lo or t_hi > hi
        t_lo, t_hi = max(t_lo, lo), min(t_hi, hi)
    times = np.linspace(t_lo, t_hi, max(settings.band_samples, 2)).tolist()
    if t_lo <= t_img <= t_hi:
        times = sorted({*times, t_img})

    def line(slant: float) -> list[tuple[float, float]]:
        points, guess = [], target.position
        for t in times:
            guess = orbit.geocode(slant, t, h, guess)
            points.append(_lonlat(guess))
        return points

    ticks = []
    step = settings.tick_step_mps
    k = 1
    while step > 0 and k * step <= v_r_max + 1e-9:
        v = k * step
        for v_r in (v, -v):
            t = t_img - v_r * dt_per_vr
            if not t_lo - 1e-12 <= t <= t_hi + 1e-12:
                continue
            p = orbit.geocode(r_img, t, h, target.position)
            v_ground_min = v / local.sin_incidence
            ticks.append(
                BandTick(
                    v_r=v_r,
                    t=t,
                    lonlat=_lonlat(p),
                    v_ground_min=v_ground_min,
                    label=f"{v:.0f} m/s (>= {v_ground_min:.0f} m/s ground)",
                )
            )
        k += 1

    return Band(
        target=target,
        target_class=target_class,
        dt_max=dt_max,
        dx_max_m=local.v_ground * dt_max,
        half_width_m=half_width,
        half_width_slant_m=half_width_slant,
        t_samples=times,
        near_lonlat=line(r_img - half_width_slant),
        far_lonlat=line(r_img + half_width_slant),
        ticks=ticks,
        clipped=clipped,
        v_eff2=local.v_eff2,
        v_ground=local.v_ground,
        sin_incidence=local.sin_incidence,
    )


def image_time_limits(
    geometry: ProductGeometry,
    pixel_to_lonlat: PixelToLonLat,
    width: int,
    row: float,
    height: float,
) -> tuple[float, float]:
    """Zero-Doppler times of the first and last file column (azimuth line) at *row*."""
    times = []
    for col in (0.5, width - 0.5):
        lon, lat = pixel_to_lonlat(col, row)
        t, _ = geometry.orbit.zero_doppler(
            lonlat_to_ecef(lon, lat, height), _mid_time(geometry)
        )
        times.append(t)
    return min(times), max(times)


@dataclass
class CursorReadout:
    """Radial velocity a band position would imply."""

    v_r_abs: float
    kmh: float
    knots: float
    dx_abs_m: float


def cursor_readout(
    geometry: ProductGeometry,
    target: ImagedTarget,
    band: Band,
    point: NDArray[np.float64],
) -> CursorReadout | None:
    """Return |v_r| = V_eff^2 |t_img - t_c| / R_img if *point* lies inside the band."""
    t_c, r_c = geometry.orbit.zero_doppler(np.asarray(point, np.float64), target.time)
    if abs(r_c - target.slant_range) > band.half_width_slant_m:
        return None
    dt = abs(target.time - t_c)
    v = band.v_eff2 * dt / target.slant_range
    return CursorReadout(v, v * MPS_TO_KMH, v * MPS_TO_KNOTS, band.v_ground * dt)


def _segment_point(
    a: NDArray[np.float64], b: NDArray[np.float64], s: float, height: float
) -> NDArray[np.float64]:
    """Point at fraction *s* of the straight segment A-B on the height surface."""
    lat, lon, _ = ecef_to_geodetic(a + s * (b - a))
    return lonlat_to_ecef(lon, lat, height)


def intersect_constraint(
    geometry: ProductGeometry,
    target: ImagedTarget,
    point_a: NDArray[np.float64],
    point_b: NDArray[np.float64],
    settings: RelocationSettings | None = None,
    slant_range: float | None = None,
) -> tuple[float, float, NDArray[np.float64]]:
    """Find where the constraint A-B crosses the target's range line.

    Starts from s = (R_img - R_A) / (R_B - R_A) and refines it by a secant search
    along the segment, so long constraints keep their straight-line geometry.

    Parameters
    ----------
    geometry : ProductGeometry
        Orbit and product metadata.
    target : ImagedTarget
        Imaged target; its height is the display surface.
    point_a, point_b : ndarray
        ECEF constraint points on the display surface.
    settings : RelocationSettings or None
        allow_extrapolation lets A and B lie on one side of the band.
    slant_range : float or None
        Range line to intersect (default: the imaged slant range).

    Returns
    -------
    tuple of (float, float, ndarray)
        (t_true, s, crossing point in ECEF).

    Raises
    ------
    ConstraintError
        If A and B do not straddle the band or have the same slant range.
    """
    settings = settings or RelocationSettings()
    orbit = geometry.orbit
    r_img = target.slant_range if slant_range is None else slant_range
    a = np.asarray(point_a, np.float64)
    b = np.asarray(point_b, np.float64)
    _, r_a = orbit.zero_doppler(a, target.time)
    _, r_b = orbit.zero_doppler(b, target.time)
    if abs(r_b - r_a) < 1e-3:
        raise ConstraintError(
            "The constraint runs along the band; click two points across it."
        )
    if (r_a - r_img) * (r_b - r_img) > 0 and not settings.allow_extrapolation:
        raise ConstraintError("Points must straddle the band.")

    def residual(s: float) -> tuple[float, float, NDArray[np.float64]]:
        p = _segment_point(a, b, s, target.height)
        t, r = orbit.zero_doppler(p, target.time)
        return r - r_img, t, p

    s0, f0 = 0.0, r_a - r_img
    s1 = (r_img - r_a) / (r_b - r_a)
    f1, t1, p1 = residual(s1)
    for _ in range(20):
        if abs(f1) < 1e-4 or f1 == f0:
            break
        s0, s1 = s1, s1 - f1 * (s1 - s0) / (f1 - f0)
        f0 = f1
        f1, t1, p1 = residual(s1)
    return t1, s1, p1


def plausibility(
    *,
    dx_m: float,
    dx_max_m: float,
    v_t: float,
    target_class: TargetClass,
    constraint_track_angle_deg: float | None,
    band_clipped: bool,
    settings: RelocationSettings,
    v_a_measured: tuple[float, float] | None = None,
    v_a_predicted: float | None = None,
) -> tuple[list[str], str]:
    """Run the plausibility checks and pick the traffic light.

    Parameters
    ----------
    dx_m, dx_max_m : float
        Along-track displacement and the band half-length (m).
    v_t : float
        Ground speed, or the minimum ground speed in single-click mode (m/s).
    target_class : TargetClass
        Sets the speed limit.
    constraint_track_angle_deg : float or None
        Angle between constraint and track; None when the constraint direction is
        unknown (single click), which skips the geometry check.
    band_clipped : bool
        Whether the band was clipped by the image extent.
    settings : RelocationSettings
        Thresholds.
    v_a_measured : tuple of float or None
        (v_a, sigma) from map drift, used only when settings.v_a_check is on.
    v_a_predicted : float or None
        Along-track velocity implied by the relocation.

    Returns
    -------
    tuple of (list of str, str)
        Failed-check flags, and green, amber (soft flags only) or red.
    """
    flags = []
    if abs(dx_m) > dx_max_m:
        flags.append(FLAG_OUTSIDE_BAND)
    if v_t > target_class.v_max_mps:
        flags.append(FLAG_IMPLAUSIBLE_SPEED)
    elif v_t < settings.v_min_mps:
        flags.append(FLAG_PROBABLY_STATIONARY)
    if constraint_track_angle_deg is None:
        flags.append(FLAG_HEADING_UNKNOWN)
    elif constraint_track_angle_deg <= settings.min_constraint_track_angle_deg:
        flags.append(FLAG_CONSTRAINT_PARALLEL)
    if settings.v_a_check and v_a_measured is not None and v_a_predicted is not None:
        v_a, sigma = v_a_measured
        same_sign = v_a * v_a_predicted >= 0
        if not same_sign or abs(v_a - v_a_predicted) > settings.v_a_k_sigma * sigma:
            flags.append(FLAG_V_A_INCONSISTENT)
    if band_clipped:
        flags.append(FLAG_BAND_CLIPPED)
    if not flags:
        return flags, INDICATOR_GREEN
    if all(f in SOFT_FLAGS for f in flags):
        return flags, INDICATOR_AMBER
    return flags, INDICATOR_RED


@dataclass
class Relocation:
    """True position and motion of a relocated target; v_r > 0 towards the radar.

    In single-click mode without an image axis, v_t, heading_deg, phi_deg,
    constraint_track_angle_deg and sigma_v_t are None; only v_t_min is known.
    """

    target: ImagedTarget
    target_class: TargetClass
    t_true: float
    t_true_utc: str
    p_true: NDArray[np.float64]
    v_r: float
    v_gr: float
    dx_m: float
    v_t: float | None
    v_t_min: float
    heading_deg: float | None
    phi_deg: float | None
    constraint_track_angle_deg: float | None
    constraint_width_m: float
    sigma_dx_m: float
    sigma_v_r: float
    sigma_v_t: float | None
    flags: list[str]
    indicator: str
    constraint_lonlat: list[tuple[float, float]]
    track_lonlat: list[tuple[float, float]]
    mode: str = MODE_TWO_CLICK
    axis_coherence: float | None = None
    sign_validated: bool = SIGN_VALIDATED

    @property
    def true_lonlat(self) -> tuple[float, float]:
        """True position (lon, lat)."""
        return _lonlat(self.p_true)

    def attributes(self) -> dict[str, Any]:
        """Flat attribute dict for the true position layer."""
        v_t = self.v_t
        return {
            "mode": self.mode,
            "v_r": self.v_r,
            "v_gr": self.v_gr,
            "v_t": v_t,
            "v_t_kmh": None if v_t is None else v_t * MPS_TO_KMH,
            "v_t_kn": None if v_t is None else v_t * MPS_TO_KNOTS,
            "v_t_min": self.v_t_min,
            "heading_deg": self.heading_deg,
            "phi_deg": self.phi_deg,
            "dx_m": self.dx_m,
            "sigma_dx_m": self.sigma_dx_m,
            "sigma_v_r": self.sigma_v_r,
            "t_true_utc": self.t_true_utc,
            "target_class": self.target_class.name,
            "flags": ",".join(self.flags),
            "indicator": self.indicator,
            "axis_coherence": self.axis_coherence,
            "sign_validated": self.sign_validated,
        }


def _range_with_residual(
    geometry: ProductGeometry,
    target: ImagedTarget,
    t_true: float,
    point: NDArray[np.float64],
) -> float:
    """Return the true slant range R_img + R v_r^2 / (2 V_eff^2) for *t_true*."""
    local = local_geometry(geometry, t_true, point)
    v_r = local.v_eff2 * (target.time - t_true) / target.slant_range
    return target.slant_range + target.slant_range * v_r**2 / (2.0 * local.v_eff2)


def _sigma_dx(settings: RelocationSettings, sin_track: float = 1.0) -> float:
    """Along-track uncertainty: centroid, click and constraint width across the track."""
    width_term = settings.constraint_width_m / math.sqrt(12.0) / sin_track
    return math.sqrt(
        settings.sigma_centroid_m**2 + settings.sigma_click_m**2 + width_term**2
    )


def _utc(geometry: ProductGeometry, t: float) -> str:
    """ISO 8601 UTC time of a zero-Doppler time."""
    epoch = geometry.reference_time + timedelta(seconds=t)
    return epoch.isoformat().replace("+00:00", "Z")


def relocate(
    geometry: ProductGeometry,
    target: ImagedTarget,
    point_a: NDArray[np.float64],
    point_b: NDArray[np.float64],
    target_class: TargetClass,
    settings: RelocationSettings | None = None,
    band: Band | None = None,
    target_height: float | None = None,
    v_a_measured: tuple[float, float] | None = None,
) -> Relocation:
    """Relocate a target from two clicks along its road, rail, deck or wake.

    The travel direction is the constraint direction that points towards the radar
    when v_r > 0, so no target-direction logic is needed.

    Parameters
    ----------
    geometry : ProductGeometry
        Orbit and product metadata.
    target : ImagedTarget
        Imaged target.
    point_a, point_b : ndarray
        ECEF constraint clicks on the display surface.
    target_class : TargetClass
        Sets the speed limit.
    settings : RelocationSettings or None
        Uncertainty, residual and plausibility settings.
    band : Band or None
        Band of *target* (computed when not given).
    target_height : float or None
        Surface to geocode the true position on (default: the display surface).
    v_a_measured : tuple of float or None
        (v_a, sigma) from map drift for the optional v_a check.

    Returns
    -------
    Relocation

    Raises
    ------
    ConstraintError
        If the clicks cannot be intersected with the band.
    """
    settings = settings or RelocationSettings()
    band = band or band_for_target(geometry, target, target_class, settings)
    h_target = target.height if target_height is None else target_height
    a = np.asarray(point_a, np.float64)
    b = np.asarray(point_b, np.float64)

    t_true, _, p_cross = intersect_constraint(geometry, target, a, b, settings)
    if settings.range_residual:
        for _ in range(2):
            r_true = _range_with_residual(geometry, target, t_true, p_cross)
            t_true, _, p_cross = intersect_constraint(
                geometry, target, a, b, settings, slant_range=r_true
            )
    _, r_true = geometry.orbit.zero_doppler(p_cross, t_true)
    p_true = geometry.orbit.geocode(r_true, t_true, h_target, p_cross)

    local = local_geometry(geometry, t_true, p_true)
    dt = target.time - t_true
    v_r = local.v_eff2 * dt / target.slant_range

    road = local.enu[:2] @ (b - a)
    u_road = road / np.linalg.norm(road)
    cos_phi = float(u_road @ local.ground_range_dir)
    phi_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_phi))))
    cos_track = abs(float(u_road @ local.along_track_dir))
    track_angle = math.degrees(math.acos(min(1.0, cos_track)))
    abs_cos_phi = max(abs(cos_phi), 1e-9)
    v_t = abs(v_r) / (abs_cos_phi * local.sin_incidence)
    towards = float(u_road @ -local.ground_range_dir)
    u_dir = u_road if towards * v_r >= 0 else -u_road

    sigma_dx = _sigma_dx(settings, max(math.sin(math.radians(track_angle)), 1e-9))
    sigma_v_r = local.v_eff2 * sigma_dx / (local.v_ground * target.slant_range)
    flags, indicator = plausibility(
        dx_m=local.v_ground * dt,
        dx_max_m=band.dx_max_m,
        v_t=v_t,
        target_class=target_class,
        constraint_track_angle_deg=track_angle,
        band_clipped=band.clipped,
        settings=settings,
        v_a_measured=v_a_measured,
        v_a_predicted=v_t * float(u_dir @ local.along_track_dir),
    )

    track: list[tuple[float, float]] = []
    if geometry.acquisition_window is not None:
        velocity = v_t * (u_dir[0] * local.enu[0] + u_dir[1] * local.enu[1])
        track = [
            _lonlat(p_true + velocity * (t - t_true))
            for t in geometry.acquisition_window
        ]

    return Relocation(
        target=target,
        target_class=target_class,
        t_true=t_true,
        t_true_utc=_utc(geometry, t_true),
        p_true=p_true,
        v_r=v_r,
        v_gr=v_r / local.sin_incidence,
        dx_m=local.v_ground * dt,
        v_t=v_t,
        v_t_min=abs(v_r) / local.sin_incidence,
        heading_deg=_heading_deg(u_dir),
        phi_deg=phi_deg,
        constraint_track_angle_deg=track_angle,
        constraint_width_m=settings.constraint_width_m,
        sigma_dx_m=sigma_dx,
        sigma_v_r=sigma_v_r,
        sigma_v_t=sigma_v_r / (abs_cos_phi * local.sin_incidence),
        flags=flags,
        indicator=indicator,
        constraint_lonlat=[_lonlat(a), _lonlat(b)],
        track_lonlat=track,
    )


def relocate_single_click(
    geometry: ProductGeometry,
    target: ImagedTarget,
    point: NDArray[np.float64],
    target_class: TargetClass,
    settings: RelocationSettings | None = None,
    band: Band | None = None,
    target_height: float | None = None,
) -> Relocation:
    """Relocate from one click where the road, rail, deck or wake crosses the band.

    The click is snapped onto the band at its own zero-Doppler time. Without the
    constraint direction only the minimum ground speed |v_r| / sin(theta_inc) is
    known, and the result is flagged heading_unknown.

    Parameters
    ----------
    geometry : ProductGeometry
        Orbit and product metadata.
    target : ImagedTarget
        Imaged target.
    point : ndarray
        ECEF click on the display surface.
    target_class : TargetClass
        Sets the speed limit.
    settings : RelocationSettings or None
        Uncertainty, residual and plausibility settings.
    band : Band or None
        Band of *target* (computed when not given).
    target_height : float or None
        Surface to geocode the true position on (default: the display surface).

    Returns
    -------
    Relocation

    Raises
    ------
    ConstraintError
        If the click is not inside the band.
    """
    settings = settings or RelocationSettings()
    band = band or band_for_target(geometry, target, target_class, settings)
    h_target = target.height if target_height is None else target_height
    c = np.asarray(point, np.float64)
    t_true, r_c = geometry.orbit.zero_doppler(c, target.time)
    if abs(r_c - target.slant_range) > band.half_width_slant_m:
        raise ConstraintError(
            "Click inside the band, where the road / wake crosses it."
        )
    r_true = target.slant_range
    if settings.range_residual:
        r_true = _range_with_residual(geometry, target, t_true, c)
    p_true = geometry.orbit.geocode(r_true, t_true, h_target, c)

    local = local_geometry(geometry, t_true, p_true)
    dt = target.time - t_true
    v_r = local.v_eff2 * dt / target.slant_range
    v_t_min = abs(v_r) / local.sin_incidence
    sigma_dx = _sigma_dx(settings)
    flags, indicator = plausibility(
        dx_m=local.v_ground * dt,
        dx_max_m=band.dx_max_m,
        v_t=v_t_min,
        target_class=target_class,
        constraint_track_angle_deg=None,
        band_clipped=band.clipped,
        settings=settings,
    )

    return Relocation(
        target=target,
        target_class=target_class,
        t_true=t_true,
        t_true_utc=_utc(geometry, t_true),
        p_true=p_true,
        v_r=v_r,
        v_gr=v_r / local.sin_incidence,
        dx_m=local.v_ground * dt,
        v_t=None,
        v_t_min=v_t_min,
        heading_deg=None,
        phi_deg=None,
        constraint_track_angle_deg=None,
        constraint_width_m=settings.constraint_width_m,
        sigma_dx_m=sigma_dx,
        sigma_v_r=local.v_eff2 * sigma_dx / (local.v_ground * target.slant_range),
        sigma_v_t=None,
        flags=flags,
        indicator=indicator,
        constraint_lonlat=[_lonlat(c)],
        track_lonlat=[],
        mode=MODE_SINGLE_CLICK,
    )


@dataclass
class ConstraintAxis:
    """Dominant linear direction of the image around a click.

    direction_en is a horizontal (East, North) unit vector with arbitrary sign;
    coherence is (l1 - l2) / (l1 + l2) of the structure tensor, 0 for isotropic
    texture and 1 for a single direction.
    """

    direction_en: NDArray[np.float64]
    coherence: float


def estimate_constraint_axis(
    chip: SlcChip,
    row: float,
    col: float,
    settings: RelocationSettings | None = None,
) -> ConstraintAxis:
    """Estimate the road, rail, deck or wake axis around a chip pixel.

    The intensity is block-averaged to about axis_cell_m ground cells, its log is
    differentiated, and the gradients are mapped to ground metres. The minor
    eigenvector of the Gaussian-weighted structure tensor within axis_radius_m is
    the line direction. Strong straight edges of other structures and sidelobe
    streaks of bright targets can be mistaken for the road.

    Parameters
    ----------
    chip : SlcChip
        SLC window around the click.
    row, col : float
        Click position in chip pixels.
    settings : RelocationSettings or None
        Window radius and cell size.

    Returns
    -------
    ConstraintAxis
    """
    settings = settings or RelocationSettings()
    steps = chip.pixel_enu_steps(row, col)
    row_m, col_m = (float(np.linalg.norm(v)) for v in steps)
    br = max(1, int(round(settings.axis_cell_m / row_m)))
    bc = max(1, int(round(settings.axis_cell_m / col_m)))
    hr = int(math.ceil(settings.axis_radius_m / row_m)) + br
    hc = int(math.ceil(settings.axis_radius_m / col_m)) + bc
    r0, r1 = max(0, int(row) - hr), min(chip.shape[0], int(row) + hr + 1)
    c0, c1 = max(0, int(col) - hc), min(chip.shape[1], int(col) + hc + 1)
    nr, nc = (r1 - r0) // br, (c1 - c0) // bc
    if nr < 3 or nc < 3:
        return ConstraintAxis(np.array([1.0, 0.0]), 0.0)
    power = np.abs(chip.data[r0 : r0 + nr * br, c0 : c0 + nc * bc]) ** 2
    blocks = power.reshape(nr, br, nc, bc).mean(axis=(1, 3))
    image = np.log10(blocks + 1e-12 * max(float(blocks.max()), 1e-30))
    g_rows, g_cols = np.gradient(image)
    jac = np.column_stack([steps[0] * br, steps[1] * bc])
    grads = np.linalg.inv(jac).T @ np.vstack([g_rows.ravel(), g_cols.ravel()])
    ii, jj = np.meshgrid(
        ((np.arange(nr) + 0.5) * br - 0.5 + r0 - row) / br,
        ((np.arange(nc) + 0.5) * bc - 0.5 + c0 - col) / bc,
        indexing="ij",
    )
    dist = np.linalg.norm(jac @ np.vstack([ii.ravel(), jj.ravel()]), axis=0)
    sigma = settings.axis_radius_m / 2.0
    weights = np.exp(-0.5 * (dist / sigma) ** 2) * (dist <= settings.axis_radius_m)
    tensor = (grads * weights) @ grads.T
    vals, vecs = np.linalg.eigh(tensor)
    total = float(vals.sum())
    coherence = float((vals[1] - vals[0]) / total) if total > 0 else 0.0
    return ConstraintAxis(vecs[:, 0] / np.linalg.norm(vecs[:, 0]), coherence)


def relocate_from_axis(
    geometry: ProductGeometry,
    target: ImagedTarget,
    point: NDArray[np.float64],
    axis: ConstraintAxis,
    target_class: TargetClass,
    settings: RelocationSettings | None = None,
    band: Band | None = None,
    target_height: float | None = None,
) -> Relocation:
    """Relocate from one click plus an image-estimated constraint axis.

    Two points axis_half_length_m either side of the click along *axis* replace the
    two constraint clicks of relocate. The result is flagged heading_estimated.

    Parameters
    ----------
    geometry : ProductGeometry
        Orbit and product metadata.
    target : ImagedTarget
        Imaged target.
    point : ndarray
        ECEF click on the display surface.
    axis : ConstraintAxis
        Axis from estimate_constraint_axis.
    target_class : TargetClass
        Sets the speed limit.
    settings : RelocationSettings or None
        Spawned point spacing, uncertainty and plausibility settings.
    band : Band or None
        Band of *target* (computed when not given).
    target_height : float or None
        Surface to geocode the true position on (default: the display surface).

    Returns
    -------
    Relocation
    """
    settings = settings or RelocationSettings()
    c = np.asarray(point, np.float64)
    lat, lon, _ = ecef_to_geodetic(c)
    enu = enu_basis(lon, lat)
    step = settings.axis_half_length_m * (
        axis.direction_en[0] * enu[0] + axis.direction_en[1] * enu[1]
    )
    a, b = (
        lonlat_to_ecef(*_lonlat(c + sign * step), target.height) for sign in (-1, 1)
    )
    result = relocate(
        geometry,
        target,
        a,
        b,
        target_class,
        settings,
        band=band,
        target_height=target_height,
    )
    result.mode = MODE_SINGLE_CLICK_AUTO
    result.axis_coherence = axis.coherence
    result.flags.append(FLAG_HEADING_ESTIMATED)
    if result.indicator == INDICATOR_GREEN:
        result.indicator = INDICATOR_AMBER
    return result
