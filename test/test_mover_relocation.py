"""Tests for the mover relocation in core.mover_relocation.

Geometry comes from the WWGTZ2 SLED crop fixture. Movers are simulated from their
exact range history |P(t) - S(t)| on the fitted orbit: a constant-velocity target is
imaged at the minimum of its range history.
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
    FLAG_PROBABLY_STATIONARY,
    INDICATOR_AMBER,
    INDICATOR_GREEN,
    INDICATOR_RED,
    RADIAL_APPROACHING,
    RADIAL_RECEDING,
    TARGET_CLASSES,
    ConstraintError,
    ImagedTarget,
    RelocationSettings,
    band_for_target,
    cursor_readout,
    estimate_constraint_axis,
    intersect_constraint,
    local_geometry,
    plausibility,
    relocate,
    relocate_single_click,
)
from iceye_toolbox.core.target_finder import (
    ProductGeometry,
    SlcChip,
    ecef_to_lonlat,
    enu_basis,
    gcp_pixel_to_lonlat,
    lonlat_to_ecef,
    read_iceye_properties,
)

CENTRE_COL, CENTRE_ROW = 4271.5, 434.5


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


def _on_surface(point: np.ndarray, height: float) -> np.ndarray:
    return lonlat_to_ecef(*ecef_to_lonlat(point), height)


def _mirrored_point(geometry: ProductGeometry, point: np.ndarray) -> np.ndarray:
    """Return the point at the same (R, t) on the other side of the ground track."""
    t, r = geometry.orbit.zero_doppler(point, 0.35)
    s, v = geometry.orbit.position(t), geometry.orbit.velocity(t)
    cross = np.cross(v, s) / np.linalg.norm(np.cross(v, s))
    d = point - s
    return geometry.orbit.geocode(
        r, t, geometry.scene_height, s + d - 2.0 * (d @ cross) * cross
    )


def _ground_velocity(point: np.ndarray, speed: float, heading_deg: float) -> np.ndarray:
    """ECEF velocity of a horizontal motion with compass heading heading_deg."""
    basis = enu_basis(*ecef_to_lonlat(point))
    h = math.radians(heading_deg)
    return speed * (math.sin(h) * basis[0] + math.cos(h) * basis[1])


@dataclasses.dataclass
class SimulatedMover:
    """A constant-velocity mover and where a zero-Doppler processor images it."""

    p_true: np.ndarray
    velocity: np.ndarray
    t_true: float
    t_img: float
    r_img: float
    v_r: float  # positive towards the radar


def simulate_mover(
    geometry: ProductGeometry, p_true: np.ndarray, velocity: np.ndarray
) -> SimulatedMover:
    """Find the range-history minimum of a mover that is at p_true at t_true."""
    orbit = geometry.orbit
    t_true, r_true = orbit.zero_doppler(p_true, 0.35)
    t = t_true
    for _ in range(50):
        d = p_true + velocity * (t - t_true) - orbit.position(t)
        dv = velocity - orbit.velocity(t)
        step = float(d @ dv) / float(dv @ dv - d @ orbit.acceleration(t))
        t -= step
        if abs(step) < 1e-12:
            break
    d = p_true + velocity * (t - t_true) - orbit.position(t)
    u = (p_true - orbit.position(t_true)) / r_true
    return SimulatedMover(
        p_true, velocity, t_true, t, float(np.linalg.norm(d)), -float(u @ velocity)
    )


def _imaged_target(geometry: ProductGeometry, mover: SimulatedMover) -> ImagedTarget:
    p_img = geometry.orbit.geocode(
        mover.r_img, mover.t_img, geometry.scene_height, mover.p_true
    )
    return ImagedTarget.from_ecef(geometry, p_img, geometry.scene_height)


def _road(mover: SimulatedMover, height: float, half_length: float = 60.0):
    """Two points on the mover's own straight track, either side of it."""
    u = mover.velocity / np.linalg.norm(mover.velocity)
    return (
        _on_surface(mover.p_true - half_length * u, height),
        _on_surface(mover.p_true + half_length * u, height),
    )


def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


class TestSignValidation:
    """Mover range history -> imaged position -> two-click relocation -> truth."""

    @pytest.mark.parametrize("side", ["left", "right"])
    @pytest.mark.parametrize("heading", [70.0, 250.0])
    def test_chain_recovers_true_motion(self, geometry, scene_point, side, heading):
        """Position, v_r sign and size, speed and heading match the simulation.

        Headings 70 and 250 deg are two cars going opposite ways along one road.
        """
        p0 = scene_point if side == "left" else _mirrored_point(geometry, scene_point)
        mover = simulate_mover(geometry, p0, _ground_velocity(p0, 12.5, heading))
        target = _imaged_target(geometry, mover)
        a, b = _road(mover, geometry.scene_height)
        result = relocate(
            geometry,
            target,
            a,
            b,
            TARGET_CLASSES["car"],
            RelocationSettings(range_residual=True),
        )
        # An approaching target is imaged later in zero-Doppler time.
        assert (mover.t_img > mover.t_true) == (mover.v_r > 0)
        assert result.t_true == pytest.approx(mover.t_true, abs=2e-5)
        assert np.linalg.norm(result.p_true - p0) < 0.3
        assert result.v_r == pytest.approx(mover.v_r, rel=0.01)
        expected_motion = RADIAL_APPROACHING if mover.v_r > 0 else RADIAL_RECEDING
        assert result.radial_motion == expected_motion
        assert result.v_t == pytest.approx(12.5, rel=0.015)
        assert _angle_diff(result.heading_deg, heading) < 0.5


class TestBand:
    """Possible-location band, ticks and cursor readout."""

    def test_band_ticks_and_clipping(self, geometry, scene_point):
        """The band spans +-dt_max with signed |v_r| ticks and clips to the image."""
        target = ImagedTarget.from_ecef(geometry, scene_point, geometry.scene_height)
        local = local_geometry(geometry, target.time, target.position)
        band = band_for_target(geometry, target, TARGET_CLASSES["ship"])
        assert band.dt_max == pytest.approx(
            target.slant_range * 15.0 * local.sin_incidence / local.v_eff2
        )
        # Ships: about 90 m of along-track shift per m/s on this geometry.
        assert 700 < band.dx_max_m < 1300
        assert not band.clipped
        for tick in band.ticks:
            assert tick.t == pytest.approx(
                target.time - target.slant_range * tick.v_r / local.v_eff2
            )
        assert all(t.t < target.time for t in band.ticks if t.v_r > 0)

        limits = (target.time - 0.2, target.time + 0.5)
        clipped = band_for_target(
            geometry, target, TARGET_CLASSES["car"], time_limits=limits
        )
        assert clipped.clipped and min(clipped.t_samples) >= limits[0] - 1e-12

    def test_cursor_readout(self, geometry, scene_point):
        """Inside the band the readout converts zero-Doppler offset to |v_r|."""
        target = ImagedTarget.from_ecef(geometry, scene_point, geometry.scene_height)
        band = band_for_target(geometry, target, TARGET_CLASSES["car"])
        local = local_geometry(geometry, target.time, target.position)
        inside = geometry.orbit.geocode(
            target.slant_range, target.time - 0.05, target.height, scene_point
        )
        readout = cursor_readout(geometry, target, band, inside)
        assert readout.v_r_abs == pytest.approx(
            local.v_eff2 * 0.05 / target.slant_range, rel=1e-3
        )
        outside = geometry.orbit.geocode(
            target.slant_range + 200.0, target.time, target.height, scene_point
        )
        assert cursor_readout(geometry, target, band, outside) is None


class TestConstraint:
    """Straddle rule and single-click relocation."""

    def test_points_must_straddle_band(self, geometry, scene_point):
        """Two clicks on one side of the band are rejected."""
        mover = simulate_mover(
            geometry, scene_point, _ground_velocity(scene_point, 10.0, 100.0)
        )
        target = _imaged_target(geometry, mover)
        a, b = _road(mover, geometry.scene_height)
        beyond = _on_surface(b + 20.0 * (b - a) / np.linalg.norm(b - a), target.height)
        with pytest.raises(ConstraintError, match="straddle"):
            intersect_constraint(geometry, target, beyond, b)

    def test_single_click(self, geometry, scene_point):
        """One click on the crossing gives position and v_r; speed is a lower bound."""
        mover = simulate_mover(
            geometry, scene_point, _ground_velocity(scene_point, 10.0, 200.0)
        )
        target = _imaged_target(geometry, mover)
        result = relocate_single_click(
            geometry,
            target,
            scene_point,
            TARGET_CLASSES["car"],
            RelocationSettings(range_residual=True),
        )
        assert result.t_true == pytest.approx(mover.t_true, abs=2e-5)
        assert result.v_r == pytest.approx(mover.v_r, rel=0.01)
        assert result.v_t is None and result.heading_deg is None
        assert result.v_t_min <= 10.0 * 1.01
        assert result.flags == [FLAG_HEADING_UNKNOWN]

        off_band = geometry.orbit.geocode(
            target.slant_range + 100.0, target.time, target.height, scene_point
        )
        with pytest.raises(ConstraintError, match="inside the band"):
            relocate_single_click(geometry, target, off_band, TARGET_CLASSES["car"])


class TestImageAxis:
    """Road direction estimated from the image around a single click."""

    @pytest.mark.parametrize("contrast", [3.0, 0.2, None])
    def test_axis_of_line_feature(self, geometry, pixel_to_lonlat, contrast):
        """A bright road or dark wake gives its axis; pure speckle gives no axis."""
        rows, cols, heading = 300, 2600, 100.0
        rng = np.random.default_rng(0)
        data = rng.standard_normal((rows, cols)) + 1j * rng.standard_normal(
            (rows, cols)
        )
        chip = SlcChip(
            data.astype(np.complex64),
            int(CENTRE_COL) - cols // 2,
            int(CENTRE_ROW) - rows // 2,
            geometry,
            pixel_to_lonlat,
        )
        h = math.radians(heading)
        if contrast is not None:
            steps = chip.pixel_enu_steps(rows / 2, cols / 2)
            rr, cc = np.mgrid[0:rows, 0:cols]
            en = (rr - rows / 2)[..., None] * steps[0] + (cc - cols / 2)[
                ..., None
            ] * steps[1]
            across = np.abs(en @ np.array([math.cos(h), -math.sin(h)]))
            chip.data[across <= 4.0] *= math.sqrt(contrast)

        axis = estimate_constraint_axis(chip, rows / 2, cols / 2)
        threshold = RelocationSettings().min_axis_coherence
        if contrast is None:
            assert axis.coherence < threshold
            return
        assert axis.coherence > threshold
        error = _angle_diff(math.degrees(math.atan2(*axis.direction_en)), heading)
        assert min(error, 180.0 - error) < 8.0


class TestPlausibility:
    """Traffic-light indicator from hard and soft flags."""

    @pytest.mark.parametrize(
        ("v_t", "angle", "clipped", "flags", "indicator"),
        [
            (15.0, 60.0, False, [], INDICATOR_GREEN),
            (60.0, 60.0, False, [FLAG_IMPLAUSIBLE_SPEED], INDICATOR_RED),
            (
                0.2,
                60.0,
                True,
                [FLAG_PROBABLY_STATIONARY, FLAG_BAND_CLIPPED],
                INDICATOR_AMBER,
            ),
            (15.0, 8.0, False, [FLAG_CONSTRAINT_PARALLEL], INDICATOR_RED),
            (15.0, None, False, [FLAG_HEADING_UNKNOWN], INDICATOR_AMBER),
        ],
    )
    def test_indicator(self, v_t, angle, clipped, flags, indicator):
        """Hard flags give red, soft flags only amber, none green."""
        result = plausibility(
            dx_m=10.0,
            dx_max_m=100.0,
            v_t=v_t,
            target_class=TARGET_CLASSES["car"],
            constraint_track_angle_deg=angle,
            band_clipped=clipped,
            settings=RelocationSettings(),
        )
        assert result == (flags, indicator)
