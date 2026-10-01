"""Tests for the two-click mover relocation in core.mover_relocation.

Geometry (orbit, GCP geolocation) comes from the WWGTZ2 SLED crop fixture. Movers are
simulated from their exact range history ``R(t) = |P(t) - S(t)|`` on the fitted orbit,
so no image synthesis is needed: a constant-velocity target is imaged at the minimum
of its range history, which is the stationary point a zero-Doppler processor focuses
it to.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from iceye_toolbox.core.mover_relocation import (
    FLAG_BAND_CLIPPED,
    FLAG_CONSTRAINT_PARALLEL,
    FLAG_HEADING_UNKNOWN,
    FLAG_IMPLAUSIBLE_SPEED,
    FLAG_OUTSIDE_BAND,
    FLAG_PROBABLY_STATIONARY,
    FLAG_V_A_INCONSISTENT,
    INDICATOR_AMBER,
    INDICATOR_GREEN,
    INDICATOR_RED,
    MODE_SINGLE_CLICK,
    TARGET_CLASSES,
    ConstraintError,
    ImagedTarget,
    RelocationSettings,
    band_for_target,
    cursor_readout,
    image_time_limits,
    intersect_constraint,
    local_geometry,
    plausibility,
    relocate,
    relocate_single_click,
)
from iceye_toolbox.core.target_finder import (
    ProductGeometry,
    ecef_to_geodetic,
    enu_basis,
    gcp_pixel_to_lonlat,
    lonlat_to_ecef,
    read_iceye_properties,
)

# Fixture scene centre (file col, row).
CENTRE_COL, CENTRE_ROW = 4271.5, 434.5
ROAD_HALF_LENGTH_M = 60.0


@pytest.fixture(scope="module")
def crop_path() -> str:
    """Path to the WWGTZ2 SLED crop fixture."""
    from .conftest import wwgtz2_fixture_tif

    return str(wwgtz2_fixture_tif("CROP"))


@pytest.fixture(scope="module")
def geometry(crop_path) -> ProductGeometry:
    """Relocation geometry parsed from the crop fixture metadata."""
    return ProductGeometry.from_properties(read_iceye_properties(crop_path))


@pytest.fixture(scope="module")
def pixel_to_lonlat(crop_path):
    """GCP geolocation of the crop fixture, released before GDAL shuts down."""
    transform = gcp_pixel_to_lonlat(crop_path)
    yield transform
    transform.close()


@pytest.fixture(scope="module")
def scene_point(geometry, pixel_to_lonlat) -> np.ndarray:
    """ECEF of the fixture centre on the scene-height surface (a left-looking point)."""
    lon, lat = pixel_to_lonlat(CENTRE_COL, CENTRE_ROW)
    return lonlat_to_ecef(lon, lat, geometry.scene_height)


# ----------------------------------------------------------------------------------
# Mover simulation helpers
# ----------------------------------------------------------------------------------


def _on_surface(point: np.ndarray, height: float) -> np.ndarray:
    lat, lon, _ = ecef_to_geodetic(point)
    return lonlat_to_ecef(lon, lat, height)


def _mirrored_point(geometry: ProductGeometry, point: np.ndarray) -> np.ndarray:
    """Return the point at the same (R, t) on the other side of the ground track."""
    t, r = geometry.orbit.zero_doppler(point, 0.35)
    s = geometry.orbit.position(t)
    v = geometry.orbit.velocity(t)
    cross = np.cross(v, s)
    cross /= np.linalg.norm(cross)
    d = point - s
    guess = s + d - 2.0 * (d @ cross) * cross
    return geometry.orbit.geocode(r, t, geometry.scene_height, guess)


def _ground_velocity(point: np.ndarray, speed: float, heading_deg: float) -> np.ndarray:
    """ECEF velocity of a horizontal motion with compass heading ``heading_deg``."""
    lat, lon, _ = ecef_to_geodetic(point)
    basis = enu_basis(lon, lat)
    h = math.radians(heading_deg)
    return speed * (math.sin(h) * basis[0] + math.cos(h) * basis[1])


@dataclasses.dataclass
class SimulatedMover:
    """A constant-velocity mover and where a zero-Doppler processor images it."""

    p_true: np.ndarray
    velocity: np.ndarray
    t_true: float
    r_true: float
    t_img: float
    r_img: float
    v_r: float  # positive towards the radar

    def position(self, t: float) -> np.ndarray:
        """Mover position at zero-Doppler time t."""
        return self.p_true + self.velocity * (t - self.t_true)


def simulate_mover(
    geometry: ProductGeometry, p_true: np.ndarray, velocity: np.ndarray
) -> SimulatedMover:
    """Range-history minimum of a mover that is at ``p_true`` at its zero-Doppler time.

    ``t_true`` is the zero-Doppler time of ``p_true``: the moment the satellite crosses
    the mover's zero-Doppler plane. ``(t_img, r_img)`` minimise ``|P(t) - S(t)|``.
    """
    orbit = geometry.orbit
    t_true, r_true = orbit.zero_doppler(p_true, 0.35)

    def rate(t: float) -> tuple[float, float]:
        d = p_true + velocity * (t - t_true) - orbit.position(t)
        dv = velocity - orbit.velocity(t)
        return float(d @ dv), float(dv @ dv - d @ orbit.acceleration(t))

    t = t_true
    for _ in range(50):
        f, fp = rate(t)
        step = f / fp
        t -= step
        if abs(step) < 1e-12:
            break
    d = p_true + velocity * (t - t_true) - orbit.position(t)
    u = (p_true - orbit.position(t_true)) / r_true
    return SimulatedMover(
        p_true=p_true,
        velocity=velocity,
        t_true=t_true,
        r_true=r_true,
        t_img=t,
        r_img=float(np.linalg.norm(d)),
        v_r=-float(u @ velocity),
    )


def _imaged_target(geometry: ProductGeometry, mover: SimulatedMover) -> ImagedTarget:
    p_img = geometry.orbit.geocode(
        mover.r_img, mover.t_img, geometry.scene_height, mover.p_true
    )
    return ImagedTarget.from_ecef(geometry, p_img, geometry.scene_height)


def _road(
    mover: SimulatedMover, height: float, half_length: float = ROAD_HALF_LENGTH_M
):
    """Two points on the mover's own straight track, either side of it."""
    u = mover.velocity / np.linalg.norm(mover.velocity)
    a = _on_surface(mover.p_true - half_length * u, height)
    b = _on_surface(mover.p_true + half_length * u, height)
    return a, b


def _expected_heading(point: np.ndarray, velocity: np.ndarray) -> float:
    lat, lon, _ = ecef_to_geodetic(point)
    en = enu_basis(lon, lat)[:2] @ velocity
    return math.degrees(math.atan2(en[0], en[1])) % 360.0


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ----------------------------------------------------------------------------------
# Test 1: sign and chain validation
# ----------------------------------------------------------------------------------

# (look side, speed m/s, heading deg); headings cover approaching and receding
# targets on both look sides, oblique and near-range-direction motion.
SIGN_CASES = [
    (side, speed, heading)
    for side in ("left", "right")
    for speed in (3.0, 12.5)
    for heading in (20.0, 100.0, 200.0, 290.0, 345.0)
]


class TestSignValidation:
    """Mover range history -> imaged coords -> two-click intersection -> truth."""

    @pytest.mark.parametrize(("side", "speed", "heading"), SIGN_CASES)
    def test_chain_recovers_true_motion(
        self, geometry, scene_point, side, speed, heading
    ):
        """t_true, v_r (magnitude and sign), dx, v_t and heading match the truth."""
        p0 = scene_point if side == "left" else _mirrored_point(geometry, scene_point)
        velocity = _ground_velocity(p0, speed, heading)
        mover = simulate_mover(geometry, p0, velocity)
        target = _imaged_target(geometry, mover)
        a, b = _road(mover, geometry.scene_height)

        settings = RelocationSettings(range_residual=True)
        result = relocate(geometry, target, a, b, TARGET_CLASSES["car"], settings)

        # Physics check independent of the tool: an approaching target (v_r > 0) is
        # imaged LATER in zero-Doppler time than its true position.
        assert math.copysign(1.0, mover.t_img - mover.t_true) == math.copysign(
            1.0, mover.v_r
        )
        assert mover.t_img - mover.t_true == pytest.approx(
            mover.r_true
            * mover.v_r
            / local_geometry(geometry, mover.t_true, p0).v_eff2,
            rel=0.01,
        )

        assert result.t_true == pytest.approx(mover.t_true, abs=2e-5)
        assert np.linalg.norm(result.p_true - p0) < 0.3
        assert math.copysign(1.0, result.v_r) == math.copysign(1.0, mover.v_r)
        assert result.v_r == pytest.approx(mover.v_r, rel=0.01)
        expected_dx = local_geometry(geometry, mover.t_true, p0).v_ground * (
            mover.t_img - mover.t_true
        )
        assert result.dx_m == pytest.approx(expected_dx, rel=0.01)
        assert abs(result.dx_m) == pytest.approx(
            np.linalg.norm(target.position - p0), rel=0.02
        )
        assert result.v_t == pytest.approx(speed, rel=0.015)
        assert _angle_diff(result.heading_deg, _expected_heading(p0, velocity)) < 0.5
        assert result.sign_validated

    def test_look_sides_differ(self, geometry, scene_point):
        """The mirrored point really is on the other side of the ground track."""
        right = _mirrored_point(geometry, scene_point)
        t, r = geometry.orbit.zero_doppler(right, 0.35)
        t0, r0 = geometry.orbit.zero_doppler(scene_point, 0.35)
        assert (t, r) == pytest.approx((t0, r0), abs=1e-4)
        g_left = local_geometry(geometry, t0, scene_point).ground_range_dir
        g_right = local_geometry(geometry, t, right).ground_range_dir
        assert g_left @ g_right < -0.9

    def test_click_order_does_not_matter(self, geometry, scene_point):
        """Swapping A and B gives the same position, v_r and heading."""
        velocity = _ground_velocity(scene_point, 8.0, 60.0)
        mover = simulate_mover(geometry, scene_point, velocity)
        target = _imaged_target(geometry, mover)
        a, b = _road(mover, geometry.scene_height)
        car = TARGET_CLASSES["car"]
        ab = relocate(geometry, target, a, b, car)
        ba = relocate(geometry, target, b, a, car)
        assert np.linalg.norm(ab.p_true - ba.p_true) < 1e-3
        assert ab.v_r == pytest.approx(ba.v_r)
        assert _angle_diff(ab.heading_deg, ba.heading_deg) < 1e-6

    def test_range_residual_is_small_at_car_speeds(self, geometry, scene_point):
        """Without the optional range residual the error stays below a metre."""
        velocity = _ground_velocity(scene_point, 12.5, 100.0)
        mover = simulate_mover(geometry, scene_point, velocity)
        target = _imaged_target(geometry, mover)
        a, b = _road(mover, geometry.scene_height)
        plain = relocate(geometry, target, a, b, TARGET_CLASSES["car"])
        assert not RelocationSettings().range_residual
        assert 0.05 < np.linalg.norm(plain.p_true - scene_point) < 1.0


# ----------------------------------------------------------------------------------
# B: possible-location band, ticks and live readout
# ----------------------------------------------------------------------------------


class TestBand:
    """Band centre line, width, ticks, clipping and cursor readout."""

    def _target(self, geometry, scene_point, half_extent=0.0) -> ImagedTarget:
        return ImagedTarget.from_ecef(
            geometry, scene_point, geometry.scene_height, half_extent_m=half_extent
        )

    def test_band_is_the_targets_range_line(self, geometry, scene_point):
        """Every centre-line sample has the imaged slant range; span is +-dt_max."""
        target = self._target(geometry, scene_point)
        band = band_for_target(geometry, target, TARGET_CLASSES["ship"])
        local = local_geometry(geometry, target.time, target.position)
        v_max = TARGET_CLASSES["ship"].v_max_mps
        assert band.dt_max == pytest.approx(
            target.slant_range * v_max * local.sin_incidence / local.v_eff2
        )
        for lon, lat in band.centre_lonlat:
            p = lonlat_to_ecef(lon, lat, target.height)
            _, r = geometry.orbit.zero_doppler(p, target.time)
            assert r == pytest.approx(target.slant_range, abs=0.01)
        assert band.t_samples[0] == pytest.approx(target.time - band.dt_max)
        assert band.t_samples[-1] == pytest.approx(target.time + band.dt_max)
        # Ships: ~90 m per m/s on this geometry, so the half-length is ~0.7 to 1.3 km.
        half_length_m = band.dt_max * local.v_ground
        assert 700 < half_length_m < 1300
        assert not band.clipped

    def test_band_edges_offset_by_half_width(self, geometry, scene_point):
        """Edges lie half-extent + margin away in ground range."""
        target = self._target(geometry, scene_point, half_extent=5.0)
        band = band_for_target(
            geometry,
            target,
            TARGET_CLASSES["car"],
            RelocationSettings(band_margin_m=10.0),
        )
        assert band.half_width_m == pytest.approx(15.0)
        mid = len(band.t_samples) // 2
        c = lonlat_to_ecef(*band.centre_lonlat[mid], target.height)
        near = lonlat_to_ecef(*band.near_lonlat[mid], target.height)
        far = lonlat_to_ecef(*band.far_lonlat[mid], target.height)
        assert np.linalg.norm(near - c) == pytest.approx(15.0, rel=0.02)
        assert np.linalg.norm(far - c) == pytest.approx(15.0, rel=0.02)
        ring = band.polygon_lonlat
        assert ring[0] == ring[-1]
        assert len(ring) == 2 * len(band.t_samples) + 1

    def test_ticks_on_both_sides(self, geometry, scene_point):
        """Ticks every 5 m/s of |v_r| on both sides, signed by the convention."""
        target = self._target(geometry, scene_point)
        band = band_for_target(geometry, target, TARGET_CLASSES["car"])
        local = local_geometry(geometry, target.time, target.position)
        v_r_max = TARGET_CLASSES["car"].v_max_mps * local.sin_incidence
        n_side = int(v_r_max // 5.0)
        assert len(band.ticks) == 2 * n_side
        for tick in band.ticks:
            assert abs(tick.v_r) % 5.0 == pytest.approx(0.0, abs=1e-9)
            expected_t = target.time - target.slant_range * tick.v_r / local.v_eff2
            assert tick.t == pytest.approx(expected_t)
            assert tick.v_ground_min == pytest.approx(
                abs(tick.v_r) / local.sin_incidence
            )
            assert f"{abs(tick.v_r):.0f} m/s" in tick.label
        # Approaching (v_r > 0) ticks are at earlier zero-Doppler time.
        approaching = [t for t in band.ticks if t.v_r > 0]
        assert approaching and all(t.t < target.time for t in approaching)

    def test_band_clipped_to_image(self, geometry, scene_point):
        """Samples and ticks outside the image time span are dropped and flagged."""
        target = self._target(geometry, scene_point)
        limits = (target.time - 0.2, target.time + 0.5)
        band = band_for_target(
            geometry, target, TARGET_CLASSES["car"], time_limits=limits
        )
        assert band.clipped
        assert min(band.t_samples) >= limits[0] - 1e-12
        assert all(limits[0] <= t.t <= limits[1] for t in band.ticks)
        assert any(t.v_r < 0 for t in band.ticks)

    def test_image_time_limits_on_fixture(self, geometry, pixel_to_lonlat):
        """The crop fixture spans ~0.05 s of zero-Doppler time (8542 columns)."""
        lo, hi = image_time_limits(
            geometry, pixel_to_lonlat, 8542, CENTRE_ROW, geometry.scene_height
        )
        # The GCP model spans ~1.3 % more time than 1 / processing_prf per column.
        assert hi - lo == pytest.approx(8541 * geometry.azimuth_time_spacing, rel=0.02)

    def test_cursor_readout(self, geometry, scene_point):
        """Inside the band the readout converts zero-Doppler offset to |v_r|."""
        target = self._target(geometry, scene_point)
        band = band_for_target(geometry, target, TARGET_CLASSES["car"])
        local = local_geometry(geometry, target.time, target.position)
        dt = 0.05
        p = geometry.orbit.geocode(
            target.slant_range + 2.0, target.time - dt, target.height, scene_point
        )
        readout = cursor_readout(geometry, target, band, p)
        assert readout is not None
        assert readout.v_r_abs == pytest.approx(
            local.v_eff2 * dt / target.slant_range, rel=1e-3
        )
        assert readout.kmh == pytest.approx(readout.v_r_abs * 3.6)
        assert readout.knots == pytest.approx(readout.v_r_abs * 3600 / 1852)
        assert readout.dx_abs_m == pytest.approx(local.v_ground * dt, rel=1e-3)
        off_band = geometry.orbit.geocode(
            target.slant_range + 200.0, target.time, target.height, scene_point
        )
        assert cursor_readout(geometry, target, band, off_band) is None


# ----------------------------------------------------------------------------------
# C and D: intersection, derived quantities, uncertainty
# ----------------------------------------------------------------------------------


class TestIntersection:
    """Straddle rules, parallel constraints, uncertainty and epoch."""

    def _case(self, geometry, scene_point, speed=10.0, heading=100.0):
        velocity = _ground_velocity(scene_point, speed, heading)
        mover = simulate_mover(geometry, scene_point, velocity)
        return mover, _imaged_target(geometry, mover)

    def test_points_must_straddle_band(self, geometry, scene_point):
        """Both clicks on one side of the band are rejected unless extrapolating."""
        mover, target = self._case(geometry, scene_point)
        a, b = _road(mover, geometry.scene_height)
        u = (b - a) / np.linalg.norm(b - a)
        a2 = _on_surface(b + 20.0 * u, geometry.scene_height)
        with pytest.raises(ConstraintError, match="straddle"):
            intersect_constraint(geometry, target, a2, b)
        settings = RelocationSettings(allow_extrapolation=True)
        t_true, _, p = intersect_constraint(geometry, target, a2, b, settings)
        assert t_true == pytest.approx(mover.t_true, abs=5e-5)
        assert np.linalg.norm(p - scene_point) < 2.0

    def test_constraint_parallel_to_range_line_rejected(self, geometry, scene_point):
        """A constraint with no range extent cannot be intersected."""
        _, target = self._case(geometry, scene_point)
        t = target.time
        a = geometry.orbit.geocode(
            target.slant_range, t - 0.01, target.height, target.position
        )
        b = geometry.orbit.geocode(
            target.slant_range, t + 0.01, target.height, target.position
        )
        with pytest.raises(ConstraintError):
            intersect_constraint(geometry, target, a, b)

    def test_near_parallel_constraint_is_flagged(self, geometry, scene_point):
        """A road within 15 deg of the track is flagged red."""
        mover, target = self._case(geometry, scene_point, speed=10.0, heading=0.0)
        local = local_geometry(geometry, mover.t_true, scene_point)
        track_heading = math.degrees(
            math.atan2(local.along_track_dir[0], local.along_track_dir[1])
        )
        velocity = _ground_velocity(scene_point, 10.0, track_heading + 8.0)
        mover = simulate_mover(geometry, scene_point, velocity)
        target = _imaged_target(geometry, mover)
        a, b = _road(mover, geometry.scene_height, half_length=400.0)
        result = relocate(geometry, target, a, b, TARGET_CLASSES["car"])
        assert FLAG_CONSTRAINT_PARALLEL in result.flags
        assert result.indicator == INDICATOR_RED
        assert result.constraint_track_angle_deg == pytest.approx(8.0, abs=0.5)

    def test_uncertainty(self, geometry, scene_point):
        """sigma_dx follows the configured budget; sigma_v_r scales from it."""
        mover, target = self._case(geometry, scene_point, heading=100.0)
        a, b = _road(mover, geometry.scene_height)
        settings = RelocationSettings(sigma_centroid_m=2.0, sigma_click_m=3.0)
        result = relocate(geometry, target, a, b, TARGET_CLASSES["car"], settings)
        sin_psi = math.sin(math.radians(result.constraint_track_angle_deg))
        expected = math.sqrt(4.0 + 9.0 + (10.0 / math.sqrt(12.0)) ** 2 / sin_psi**2)
        assert result.sigma_dx_m == pytest.approx(expected)
        local = local_geometry(geometry, result.t_true, result.p_true)
        assert result.sigma_v_r == pytest.approx(
            local.v_eff2 * expected / (local.v_ground * local.slant_range)
        )
        assert result.constraint_width_m == 10.0
        wide = relocate(
            geometry,
            target,
            a,
            b,
            TARGET_CLASSES["car"],
            RelocationSettings(constraint_width_m=20.0),
        )
        assert wide.constraint_width_m == 20.0
        assert wide.sigma_dx_m > result.sigma_dx_m

    def test_epoch_and_track(self, geometry, scene_point):
        """t_true is reported in UTC; the track spans the acquisition window."""
        mover, target = self._case(geometry, scene_point)
        a, b = _road(mover, geometry.scene_height)
        result = relocate(geometry, target, a, b, TARGET_CLASSES["car"])
        assert result.t_true_utc.startswith("2025-11-09T14:15:")
        assert geometry.acquisition_window is not None
        assert len(result.track_lonlat) == 2
        start = lonlat_to_ecef(*result.track_lonlat[0], target.height)
        end = lonlat_to_ecef(*result.track_lonlat[1], target.height)
        duration = geometry.acquisition_window[1] - geometry.acquisition_window[0]
        assert np.linalg.norm(end - start) == pytest.approx(
            result.v_t * duration, rel=0.01
        )

    def test_attributes(self, geometry, scene_point):
        """The true-position attributes carry every field of the handoff."""
        mover, target = self._case(geometry, scene_point)
        a, b = _road(mover, geometry.scene_height)
        attributes = relocate(
            geometry, target, a, b, TARGET_CLASSES["car"]
        ).attributes()
        for name in (
            "v_r",
            "v_gr",
            "v_t",
            "v_t_kmh",
            "v_t_kn",
            "heading_deg",
            "phi_deg",
            "dx_m",
            "sigma_dx_m",
            "t_true_utc",
            "target_class",
            "flags",
            "indicator",
            "sign_validated",
        ):
            assert name in attributes
        assert attributes["target_class"] == "car"
        assert attributes["v_t_kmh"] == pytest.approx(attributes["v_t"] * 3.6)


# ----------------------------------------------------------------------------------
# E: plausibility
# ----------------------------------------------------------------------------------


class TestSingleClick:
    """One click where the constraint crosses the band: position and v_r only."""

    @pytest.mark.parametrize(
        ("side", "heading"), [("left", 20.0), ("left", 200.0), ("right", 100.0)]
    )
    def test_click_on_crossing_recovers_position_and_v_r(
        self, geometry, scene_point, side, heading
    ):
        """t_true, v_r and dx match the truth; speed is a lower bound, no heading."""
        p0 = scene_point if side == "left" else _mirrored_point(geometry, scene_point)
        speed = 10.0
        mover = simulate_mover(geometry, p0, _ground_velocity(p0, speed, heading))
        target = _imaged_target(geometry, mover)
        result = relocate_single_click(
            geometry,
            target,
            p0,
            TARGET_CLASSES["car"],
            RelocationSettings(range_residual=True),
        )
        assert result.mode == MODE_SINGLE_CLICK
        assert result.t_true == pytest.approx(mover.t_true, abs=2e-5)
        assert np.linalg.norm(result.p_true - p0) < 0.3
        assert math.copysign(1.0, result.v_r) == math.copysign(1.0, mover.v_r)
        assert result.v_r == pytest.approx(mover.v_r, rel=0.01)
        local = local_geometry(geometry, mover.t_true, p0)
        assert result.dx_m == pytest.approx(
            local.v_ground * (mover.t_img - mover.t_true), rel=0.01
        )
        assert result.v_t is None and result.heading_deg is None
        assert result.v_t_min == pytest.approx(abs(result.v_r) / local.sin_incidence)
        assert result.v_t_min <= speed * 1.01
        assert result.track_lonlat == []
        assert result.flags == [FLAG_HEADING_UNKNOWN]
        assert result.indicator == INDICATOR_AMBER
        attributes = result.attributes()
        assert attributes["mode"] == MODE_SINGLE_CLICK
        assert attributes["v_t"] is None and attributes["v_t_min"] > 0

    def test_minimum_speed_is_exact_along_ground_range(self, geometry, scene_point):
        """A target driving straight at the radar has v_t_min equal to its speed."""
        local = local_geometry(
            geometry, geometry.orbit.zero_doppler(scene_point, 0.35)[0], scene_point
        )
        towards = -local.ground_range_dir
        heading = math.degrees(math.atan2(towards[0], towards[1]))
        mover = simulate_mover(
            geometry, scene_point, _ground_velocity(scene_point, 8.0, heading)
        )
        target = _imaged_target(geometry, mover)
        result = relocate_single_click(
            geometry, target, scene_point, TARGET_CLASSES["car"]
        )
        assert result.v_r > 0
        assert result.v_t_min == pytest.approx(8.0, rel=0.01)

    def test_click_outside_band_is_rejected(self, geometry, scene_point):
        """The click must lie inside the band."""
        mover = simulate_mover(
            geometry, scene_point, _ground_velocity(scene_point, 10.0, 100.0)
        )
        target = _imaged_target(geometry, mover)
        off = geometry.orbit.geocode(
            target.slant_range + 100.0, target.time, target.height, scene_point
        )
        with pytest.raises(ConstraintError, match="inside the band"):
            relocate_single_click(geometry, target, off, TARGET_CLASSES["car"])


class TestPlausibility:
    """Traffic-light indicator from hard and soft flags."""

    def test_green_for_a_plain_car(self, geometry, scene_point):
        """A 15 m/s car on an oblique road passes every check."""
        velocity = _ground_velocity(scene_point, 15.0, 100.0)
        mover = simulate_mover(geometry, scene_point, velocity)
        target = _imaged_target(geometry, mover)
        a, b = _road(mover, geometry.scene_height)
        result = relocate(geometry, target, a, b, TARGET_CLASSES["car"])
        assert result.flags == []
        assert result.indicator == INDICATOR_GREEN

    def test_speed_checks(self):
        """Too fast is red; almost stationary is amber."""
        ship = TARGET_CLASSES["ship"]
        settings = RelocationSettings()
        flags, indicator = plausibility(
            dx_m=10.0,
            dx_max_m=100.0,
            v_t=20.0,
            target_class=ship,
            constraint_track_angle_deg=60.0,
            band_clipped=False,
            settings=settings,
        )
        assert flags == [FLAG_IMPLAUSIBLE_SPEED] and indicator == INDICATOR_RED
        flags, indicator = plausibility(
            dx_m=1.0,
            dx_max_m=100.0,
            v_t=0.2,
            target_class=ship,
            constraint_track_angle_deg=60.0,
            band_clipped=True,
            settings=settings,
        )
        assert set(flags) == {FLAG_PROBABLY_STATIONARY, FLAG_BAND_CLIPPED}
        assert indicator == INDICATOR_AMBER

    def test_outside_band_is_red(self):
        """A displacement beyond the band half-length is a hard failure."""
        flags, indicator = plausibility(
            dx_m=-150.0,
            dx_max_m=100.0,
            v_t=5.0,
            target_class=TARGET_CLASSES["car"],
            constraint_track_angle_deg=60.0,
            band_clipped=False,
            settings=RelocationSettings(),
        )
        assert FLAG_OUTSIDE_BAND in flags and indicator == INDICATOR_RED

    def test_v_a_check_is_off_by_default(self):
        """The along-track check only runs when enabled."""
        common = dict(
            dx_m=10.0,
            dx_max_m=100.0,
            v_t=5.0,
            target_class=TARGET_CLASSES["ship"],
            constraint_track_angle_deg=60.0,
            band_clipped=False,
            v_a_measured=(-2.0, 0.2),
            v_a_predicted=2.0,
        )
        flags, _ = plausibility(settings=RelocationSettings(), **common)
        assert FLAG_V_A_INCONSISTENT not in flags
        flags, indicator = plausibility(
            settings=RelocationSettings(v_a_check=True), **common
        )
        assert FLAG_V_A_INCONSISTENT in flags and indicator == INDICATOR_RED
