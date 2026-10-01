"""Two-click relocation of moving targets in ICEYE Spotlight / Dwell SLCs.

A constant-velocity target's range history is that of a stationary target displaced
purely along track, so every possible true position lies on the target's own range
line (constant slant range, varying zero-Doppler time). In Spotlight / Dwell the
target's echoes carry no usable radial velocity, so the position along that line comes
from a constraint the user reads off the image:

1. **Target.** The imaged position ``(R_img, t_img)`` (click or curve, see
   ``locate_imaged_target``). ``band_for_target`` draws the possible-location band,
   ``R = R_img`` for ``|t - t_img| <= dt_max``, with ``|v_r|`` ticks.
2. **Constraint.** Two clicks on a road / rail / bridge deck / wake axis on opposite
   sides of the band. ``relocate`` intersects that segment with the band and derives
   the true position, velocity and heading, with a plausibility indicator.

Relations (``v_r`` positive towards the radar, ``t`` zero-Doppler time)::

    dt  = t_img - t_true = R v_r / V_eff^2
    dx  = V_g dt
    v_t = |v_r| / (|cos(phi)| sin(theta_inc))

Only geolocated points and the orbit are used (clicks go to ECEF on the display
surface, then through the range-Doppler inverse), so no Doppler or column-time sign
enters. ``test/test_mover_relocation.py`` validates the chain, signs included, against
simulated range histories for left and right looks, approaching and receding.
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

# The band / intersection chain and its signs are validated in simulation by
# test_mover_relocation.TestSignValidation; see the module docstring.
SIGN_VALIDATED = True

FLAG_OUTSIDE_BAND = "outside_band"
FLAG_IMPLAUSIBLE_SPEED = "implausible_speed"
FLAG_PROBABLY_STATIONARY = "probably_stationary"
FLAG_CONSTRAINT_PARALLEL = "constraint_parallel_to_track"
FLAG_V_A_INCONSISTENT = "v_a_inconsistent"
FLAG_BAND_CLIPPED = "band_clipped"
SOFT_FLAGS = frozenset({FLAG_BAND_CLIPPED, FLAG_PROBABLY_STATIONARY})

INDICATOR_GREEN = "green"
INDICATOR_AMBER = "amber"
INDICATOR_RED = "red"


@dataclass(frozen=True)
class TargetClass:
    """Target type chosen in the tool panel; sets the band length and speed checks."""

    name: str
    v_max_mps: float


TARGET_CLASSES = {
    "car": TargetClass("car", 40.0),
    "train": TargetClass("train", 90.0),
    "ship": TargetClass("ship", 15.0),
}


@dataclass
class RelocationSettings:
    """Configuration of band, intersection, uncertainty and plausibility checks."""

    band_margin_m: float = 10.0
    tick_step_mps: float = 5.0
    band_samples: int = 41
    v_min_mps: float = 0.5
    min_constraint_track_angle_deg: float = 15.0
    sigma_centroid_m: float = 2.0
    sigma_click_m: float = 2.0
    # Width of the road / rail / deck / wake the clicks follow, for sigma_dx.
    constraint_width_m: float = 10.0
    # True slant range is R_img + R v_r^2 / (2 V_eff^2): below 1 m at car speeds.
    range_residual: bool = False
    allow_extrapolation: bool = False
    # Along-track velocity check from map drift; off until its sign is validated.
    v_a_check: bool = False
    v_a_k_sigma: float = 3.0


class ConstraintError(ValueError):
    """The constraint clicks cannot be intersected with the band."""


# ----------------------------------------------------------------------------------
# Local geometry
# ----------------------------------------------------------------------------------


@dataclass
class LocalGeometry:
    """Zero-Doppler geometry of an Earth-fixed point.

    ``ground_range_dir`` (away from the satellite) and ``along_track_dir`` (platform
    velocity) are horizontal unit vectors in local (East, North).
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
    """``V_eff^2``, ``V_g``, local incidence and horizontal directions at ``point``."""
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
    lat, lon, _ = ecef_to_geodetic(point)
    return lon, lat


def _heading_deg(en: NDArray[np.float64]) -> float:
    """Compass heading (deg from North, clockwise) of an EN vector."""
    return math.degrees(math.atan2(en[0], en[1])) % 360.0


# ----------------------------------------------------------------------------------
# A: imaged target
# ----------------------------------------------------------------------------------


@dataclass
class ImagedTarget:
    """Imaged (displaced) target in radar coordinates.

    ``height`` is the surface the display geocoding uses; clicks are mapped onto the
    same surface so they land on the pixels the user sees.
    """

    slant_range: float
    time: float
    position: NDArray[np.float64]
    height: float
    half_extent_m: float = 0.0
    hull_mask: NDArray[np.bool_] | None = None
    row: float | None = None
    col: float | None = None

    @classmethod
    def from_ecef(
        cls,
        geometry: ProductGeometry,
        point: NDArray[np.float64],
        height: float,
        half_extent_m: float = 0.0,
        t_guess: float | None = None,
    ) -> ImagedTarget:
        """Target at an ECEF point on the display surface."""
        guess = _mid_time(geometry) if t_guess is None else t_guess
        t, r = geometry.orbit.zero_doppler(np.asarray(point, np.float64), guess)
        return cls(r, t, np.asarray(point, np.float64), height, half_extent_m)

    @property
    def lonlat(self) -> tuple[float, float]:
        """Imaged position (lon, lat)."""
        return _lonlat(self.position)


def _mid_time(geometry: ProductGeometry) -> float:
    if len(geometry.dc_times):
        return 0.5 * float(geometry.dc_times[0] + geometry.dc_times[-1])
    return 0.0


def locate_imaged_target(
    chip: SlcChip,
    control_points: Sequence[Any],
    height: float,
    params: RelocationParameters | None = None,
) -> ImagedTarget:
    """Intensity-weighted hull centroid around a curve (or a click) in an SLC chip.

    ``control_points`` are the curve editor's four Bezier points as 0..1 chip
    fractions; for a single click pass the click fraction four times, which makes the
    corridor a disc of ``corridor_half_width_m``. Hull and ring thresholds are those
    of ``core.target_finder.hull_and_clutter_masks``.
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
    if rr.size:
        w = intensity[rr, cc]
        row = float(np.sum(rr * w) / np.sum(w)) + masks.window[0].start
        col = float(np.sum(cc * w) / np.sum(w)) + masks.window[1].start
    else:
        row, col = float(rows[mid]), float(cols[mid])
    lon, lat = chip.lonlat(row, col)
    target = ImagedTarget.from_ecef(
        chip.geometry, lonlat_to_ecef(lon, lat, height), height
    )
    if rr.size:
        # Half the hull's ground-range extent widens the band.
        g_hat = local_geometry(
            chip.geometry, target.time, target.position
        ).ground_range_dir
        across = rr * float(steps[0] @ g_hat) + cc * float(steps[1] @ g_hat)
        target.half_extent_m = 0.5 * float(np.ptp(across))
    target.row, target.col = row, col
    target.hull_mask = np.zeros(chip.shape, dtype=bool)
    target.hull_mask[masks.window] = hull
    return target


# ----------------------------------------------------------------------------------
# B: possible-location band
# ----------------------------------------------------------------------------------


@dataclass
class BandTick:
    """Point on the band where ``v_r`` has a round value (signed, towards radar > 0)."""

    v_r: float
    t: float
    lonlat: tuple[float, float]
    v_ground_min: float
    label: str


@dataclass
class Band:
    """Possible true positions of a target: its range line, ``|t - t_img| <= dt_max``."""

    target: ImagedTarget
    target_class: TargetClass
    dt_max: float
    dx_max_m: float
    half_width_m: float
    half_width_slant_m: float
    t_samples: list[float]
    centre_lonlat: list[tuple[float, float]]
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
    """Band, speed ticks and clipping for ``target``.

    ``dt_max = R_img v_max sin(theta_inc) / V_eff^2``; the band is sampled at
    ``band_samples`` zero-Doppler times, its edges are range lines offset by the
    target half-extent plus ``band_margin_m`` in ground range. ``time_limits`` is the
    image's zero-Doppler span (``image_time_limits``); samples and ticks outside are
    dropped and the band is flagged clipped.
    """
    settings = settings or RelocationSettings()
    orbit = geometry.orbit
    local = local_geometry(geometry, target.time, target.position)
    r_img, t_img, h = target.slant_range, target.time, target.height
    dt_per_vr = r_img / local.v_eff2
    dt_max = target_class.v_max_mps * local.sin_incidence * dt_per_vr
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
    while step > 0 and k * step <= target_class.v_max_mps * local.sin_incidence + 1e-9:
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
        centre_lonlat=line(r_img),
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
    """Zero-Doppler times of the first and last file column (azimuth line) at ``row``."""
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
    dt: float


def cursor_readout(
    geometry: ProductGeometry,
    target: ImagedTarget,
    band: Band,
    point: NDArray[np.float64],
) -> CursorReadout | None:
    """``|v_r| = V_eff^2 |t_img - t_c| / R_img`` if ``point`` lies inside the band."""
    t_c, r_c = geometry.orbit.zero_doppler(np.asarray(point, np.float64), target.time)
    if abs(r_c - target.slant_range) > band.half_width_slant_m:
        return None
    dt = target.time - t_c
    v = band.v_eff2 * abs(dt) / target.slant_range
    return CursorReadout(
        v, v * MPS_TO_KMH, v * MPS_TO_KNOTS, band.v_ground * abs(dt), dt
    )


# ----------------------------------------------------------------------------------
# C: constraint intersection
# ----------------------------------------------------------------------------------


def _segment_point(
    a: NDArray[np.float64], b: NDArray[np.float64], s: float, height: float
) -> NDArray[np.float64]:
    """Point at fraction ``s`` of the straight segment A-B on the height surface."""
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
    """Where the constraint A-B crosses the target's range line.

    ``s = (R_img - R_A) / (R_B - R_A)`` is refined by a secant search along the
    segment on the display surface, so long constraints keep their straight-line
    geometry. Returns ``(t_true, s, point)``.

    Raises
    ------
    ConstraintError
        If A and B do not straddle the band (unless ``allow_extrapolation``) or have
        the same slant range.
    """
    settings = settings or RelocationSettings()
    orbit = geometry.orbit
    r_img = target.slant_range if slant_range is None else slant_range
    h = target.height
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
        p = _segment_point(a, b, s, h)
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


# ----------------------------------------------------------------------------------
# D and E: derived quantities and plausibility
# ----------------------------------------------------------------------------------


def plausibility(
    *,
    dx_m: float,
    dx_max_m: float,
    v_t: float,
    target_class: TargetClass,
    constraint_track_angle_deg: float,
    band_clipped: bool,
    settings: RelocationSettings,
    v_a_measured: tuple[float, float] | None = None,
    v_a_predicted: float | None = None,
) -> tuple[list[str], str]:
    """Failed checks and the traffic light: green, amber (soft flags only) or red.

    ``v_a_measured`` is ``(v_a, sigma)`` from map drift, used only with ``v_a_check``.
    """
    flags = []
    if abs(dx_m) > dx_max_m:
        flags.append(FLAG_OUTSIDE_BAND)
    if v_t > target_class.v_max_mps:
        flags.append(FLAG_IMPLAUSIBLE_SPEED)
    elif v_t < settings.v_min_mps:
        flags.append(FLAG_PROBABLY_STATIONARY)
    if constraint_track_angle_deg <= settings.min_constraint_track_angle_deg:
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
    """True position and motion of a relocated target; ``v_r > 0`` towards the radar."""

    target: ImagedTarget
    target_class: TargetClass
    t_true: float
    t_true_utc: str
    p_true: NDArray[np.float64]
    dt_s: float
    v_r: float
    v_gr: float
    dx_m: float
    v_t: float
    heading_deg: float
    phi_deg: float
    constraint_track_angle_deg: float
    constraint_width_m: float
    sigma_dx_m: float
    sigma_v_r: float
    sigma_v_t: float
    v_a_predicted: float
    flags: list[str]
    indicator: str
    constraint_lonlat: tuple[tuple[float, float], tuple[float, float]]
    track_lonlat: list[tuple[float, float]]
    sign_validated: bool = SIGN_VALIDATED

    @property
    def true_lonlat(self) -> tuple[float, float]:
        """True position (lon, lat)."""
        return _lonlat(self.p_true)

    def attributes(self) -> dict[str, Any]:
        """Flat attribute dict for the ``true_position`` layer."""
        return {
            "v_r": self.v_r,
            "v_gr": self.v_gr,
            "v_t": self.v_t,
            "v_t_kmh": self.v_t * MPS_TO_KMH,
            "v_t_kn": self.v_t * MPS_TO_KNOTS,
            "heading_deg": self.heading_deg,
            "phi_deg": self.phi_deg,
            "dx_m": self.dx_m,
            "sigma_dx_m": self.sigma_dx_m,
            "sigma_v_r": self.sigma_v_r,
            "t_true_utc": self.t_true_utc,
            "target_class": self.target_class.name,
            "flags": ",".join(self.flags),
            "indicator": self.indicator,
            "sign_validated": self.sign_validated,
        }


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
    """Relocate the target: true position, velocity and heading from constraint A-B.

    ``point_a`` / ``point_b`` are ECEF on the display surface (``target.height``).
    ``target_height`` is the surface the true position is geocoded on (default: the
    display surface; ships: the sea surface). Travel direction is ``+-u_road`` chosen
    so it points towards the radar when ``v_r > 0``, so no target-direction logic is
    needed.
    """
    settings = settings or RelocationSettings()
    band = band or band_for_target(geometry, target, target_class, settings)
    h_target = target.height if target_height is None else target_height
    a = np.asarray(point_a, np.float64)
    b = np.asarray(point_b, np.float64)

    t_true, _, p_cross = intersect_constraint(geometry, target, a, b, settings)
    if settings.range_residual:
        # The true slant range is longer by R v_r^2 / (2 V_eff^2); two passes settle it.
        for _ in range(2):
            local = local_geometry(geometry, t_true, p_cross)
            v_r = local.v_eff2 * (target.time - t_true) / local.slant_range
            r_true = target.slant_range + target.slant_range * v_r**2 / (
                2.0 * local.v_eff2
            )
            t_true, _, p_cross = intersect_constraint(
                geometry, target, a, b, settings, slant_range=r_true
            )
    _, r_true = geometry.orbit.zero_doppler(p_cross, t_true)
    p_true = geometry.orbit.geocode(r_true, t_true, h_target, p_cross)

    local = local_geometry(geometry, t_true, p_true)
    dt = target.time - t_true
    v_r = local.v_eff2 * dt / target.slant_range
    dx = local.v_ground * dt

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
    v_a_pred = v_t * float(u_dir @ local.along_track_dir)

    width = settings.constraint_width_m
    sin_track = max(math.sin(math.radians(track_angle)), 1e-9)
    sigma_dx = math.sqrt(
        settings.sigma_centroid_m**2
        + settings.sigma_click_m**2
        + (width / math.sqrt(12.0)) ** 2 / sin_track**2
    )
    sigma_v_r = local.v_eff2 * sigma_dx / (local.v_ground * target.slant_range)

    flags, indicator = plausibility(
        dx_m=dx,
        dx_max_m=band.dx_max_m,
        v_t=v_t,
        target_class=target_class,
        constraint_track_angle_deg=track_angle,
        band_clipped=band.clipped,
        settings=settings,
        v_a_measured=v_a_measured,
        v_a_predicted=v_a_pred,
    )

    track: list[tuple[float, float]] = []
    if geometry.acquisition_window is not None:
        velocity = v_t * (u_dir[0] * local.enu[0] + u_dir[1] * local.enu[1])
        track = [
            _lonlat(p_true + velocity * (t - t_true))
            for t in geometry.acquisition_window
        ]

    epoch = geometry.reference_time + timedelta(seconds=t_true)
    return Relocation(
        target=target,
        target_class=target_class,
        t_true=t_true,
        t_true_utc=epoch.isoformat().replace("+00:00", "Z"),
        p_true=p_true,
        dt_s=dt,
        v_r=v_r,
        v_gr=v_r / local.sin_incidence,
        dx_m=dx,
        v_t=v_t,
        heading_deg=_heading_deg(u_dir),
        phi_deg=phi_deg,
        constraint_track_angle_deg=track_angle,
        constraint_width_m=width,
        sigma_dx_m=sigma_dx,
        sigma_v_r=sigma_v_r,
        sigma_v_t=sigma_v_r / (abs_cos_phi * local.sin_incidence),
        v_a_predicted=v_a_pred,
        flags=flags,
        indicator=indicator,
        constraint_lonlat=(_lonlat(a), _lonlat(b)),
        track_lonlat=track,
    )
