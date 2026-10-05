"""Tests for the SLC geometry, target detection and map drift in core.target_finder.

Geometry (orbit, Doppler centroid, GCP geolocation) comes from the WWGTZ2 SLED crop
fixture; the SLC samples are synthetic clutter with injected point-target ships, so the
expected positions and drift are known exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from iceye_toolbox.core.mover_relocation import locate_imaged_target
from iceye_toolbox.core.target_finder import (
    DOPPLER_SIGN,
    ProductGeometry,
    RelocationParameters,
    SlcChip,
    along_track_velocity,
    bezier_points,
    corridor_mask,
    doppler_amplification,
    ecef_to_geodetic,
    find_targets_along_curve,
    gcp_pixel_to_lonlat,
    lonlat_to_ecef,
    patch_to_file_layout,
    read_iceye_properties,
)

ROWS, COLS = 869, 8542
SHIP_ROW, SHIP_COL = 434, 4271
# A small ship: 25 scatterers over ~35 m along track and ~9 m across.
HULL_ROWS = SHIP_ROW + np.linspace(-10, 10, 25)
HULL_COLS = SHIP_COL + np.linspace(-400, 400, 25)


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


def _clutter(geometry: ProductGeometry, seed: int = 0) -> np.ndarray:
    """Complex Gaussian clutter with the SLC's azimuth spectrum (flat over the band)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((ROWS, COLS)) + 1j * rng.standard_normal((ROWS, COLS))
    f = np.fft.fftfreq(COLS, geometry.azimuth_time_spacing)
    band = np.abs(f) <= geometry.processed_azimuth_bandwidth / 2
    noise = np.fft.ifft(np.fft.fft(noise, axis=1) * band[None, :], axis=1)
    return (noise * 10.0).astype(np.complex64)


def _inject_ship(
    data: np.ndarray,
    geometry: ProductGeometry,
    dt_col: float,
    ka: float,
    *,
    drift_per_s: float = 0.0,
    rows: np.ndarray,
    cols: np.ndarray,
    amplitude: float = 3000.0,
) -> None:
    """Add band-limited point scatterers at (rows, cols).

    ``drift_per_s`` is the look position change d(t_img)/d(tau) caused by a
    Doppler-rate mismatch.
    """
    f = DOPPLER_SIGN * np.fft.fftfreq(COLS, dt_col)
    support = np.abs(f) <= geometry.processed_azimuth_bandwidth / 2
    # Group delay t_g(f) = drift * tau(f) with tau = -f / Ka, so phase = pi*drift*f^2/Ka.
    spectrum_phase = np.exp(1j * math.pi * drift_per_s * f**2 / ka)
    for r, c in zip(rows, cols):
        impulse = np.zeros(COLS, dtype=np.complex128)
        impulse[int(c)] = amplitude
        line = np.fft.ifft(np.fft.fft(impulse) * support * spectrum_phase)
        data[int(r), :] += line.astype(np.complex64)


def _chip(data, geometry, pixel_to_lonlat) -> SlcChip:
    return SlcChip(data, 0, 0, geometry, pixel_to_lonlat)


def _horizontal_curve(half_cols: int) -> list[tuple[float, float]]:
    y = SHIP_ROW / (ROWS - 1)
    x0 = (SHIP_COL - half_cols) / (COLS - 1)
    x1 = (SHIP_COL + half_cols) / (COLS - 1)
    return [
        (x0, y),
        (x0 + (x1 - x0) / 3, y),
        (x0 + 2 * (x1 - x0) / 3, y),
        (x1, y),
    ]


def _setup(geometry, pixel_to_lonlat):
    data = _clutter(geometry)
    chip = _chip(data, geometry, pixel_to_lonlat)
    dt_col = chip.azimuth_time_step(SHIP_ROW, SHIP_COL)
    t, _ = chip.zero_doppler(SHIP_ROW, SHIP_COL)
    ka = geometry.kinematics(t, chip.ecef(SHIP_ROW, SHIP_COL)).fm_rate
    return data, chip, dt_col, ka


class TestGeometry:
    """Orbit fit, range-Doppler inverse and geocoding on the fixture metadata."""

    def test_metadata_parsing(self, geometry):
        """ICEYE_PROPERTIES fields land in ProductGeometry."""
        assert geometry.instrument_mode == "spotlight"
        assert geometry.look_side == "left"
        assert geometry.azimuth_time_spacing == pytest.approx(1 / 163152.122)
        assert len(geometry.dc_coeffs) == len(geometry.dc_times) == 10
        start, end = geometry.acquisition_window
        assert end - start == pytest.approx(28.789, abs=1e-3)

    def test_zero_doppler_geocode_round_trip(self, geometry, pixel_to_lonlat):
        """Range-Doppler inverse then geocode returns the same point."""
        lon, lat = pixel_to_lonlat(4000.5, 400.5)
        point = lonlat_to_ecef(lon, lat, geometry.scene_height)
        t, r = geometry.orbit.zero_doppler(point, 0.35)
        assert 0.0 < t < 0.72
        assert 701000 < r < 705000
        back = geometry.orbit.geocode(r, t, geometry.scene_height, point + 50.0)
        assert np.linalg.norm(back - point) < 0.01
        lat2, lon2, h2 = ecef_to_geodetic(back)
        assert (lat2, lon2) == pytest.approx((lat, lon), abs=1e-8)
        assert h2 == pytest.approx(geometry.scene_height, abs=1e-3)

    def test_fm_rate_matches_rule_of_thumb(self, geometry, pixel_to_lonlat):
        """Ka and the azimuth shift per m/s have the expected size."""
        lon, lat = pixel_to_lonlat(4000.5, 400.5)
        point = lonlat_to_ecef(lon, lat, geometry.scene_height)
        t, r = geometry.orbit.zero_doppler(point, 0.35)
        kin = geometry.kinematics(t, point)
        assert kin.fm_rate == pytest.approx(
            2 * kin.v_eff**2 / (geometry.wavelength * r)
        )
        assert 4800 < kin.fm_rate < 5000
        # Azimuth shift per 1 m/s radial velocity: R * V_g / V_eff^2.
        assert 80 < r * kin.v_ground / kin.v_eff**2 < 100

    def test_dwell_doppler_is_degenerate(self, geometry, pixel_to_lonlat):
        """In Dwell the DC slope nearly equals Ka: why no Doppler v_r is used."""
        lon, lat = pixel_to_lonlat(4000.5, 400.5)
        point = lonlat_to_ecef(lon, lat, geometry.scene_height)
        t, _ = geometry.orbit.zero_doppler(point, 0.35)
        assert doppler_amplification(geometry, t, point) > 500


class TestPrimitives:
    """Curve and masks on synthetic arrays."""

    def test_bezier_endpoints_and_straight_line(self):
        """Handles at 1/3 and 2/3 give a straight, evenly sampled line."""
        pts = bezier_points([(0, 0.5), (1 / 3, 0.5), (2 / 3, 0.5), (1, 0.5)], n=11)
        assert pts[0] == pytest.approx([0, 0.5])
        assert pts[-1] == pytest.approx([1, 0.5])
        assert pts[:, 0] == pytest.approx(np.linspace(0, 1, 11))

    def test_corridor_mask_is_anisotropic(self):
        """The corridor honours separate row and column half-widths."""
        rows = np.array([50.0, 50.0])
        cols = np.array([20.0, 80.0])
        mask = corridor_mask((100, 100), rows, cols, half_rows=5, half_cols=10)
        assert mask[50, 20:81].all()
        assert mask[45, 50] and not mask[44, 50]
        assert mask[50, 10] and not mask[50, 9]

    def test_patch_to_file_layout_inverts_read_orientation(self):
        """Undoes the shadows-down orientation of read_slc_layer."""
        file_data = np.arange(12).reshape(3, 4)
        for left in (False, True):
            patch = (np.fliplr(file_data) if left else file_data).T
            assert np.array_equal(patch_to_file_layout(patch, left), file_data)


class TestTargetInput:
    """Imaged position of an injected ship from a curve or a single click."""

    @pytest.mark.parametrize("single_click", [False, True])
    def test_hull_centroid(self, geometry, pixel_to_lonlat, single_click):
        """The hull centroid lands on the injected ship, curve or click alike."""
        data, chip, dt_col, ka = _setup(geometry, pixel_to_lonlat)
        _inject_ship(
            data, geometry, dt_col, ka, rows=HULL_ROWS, cols=HULL_COLS, amplitude=3e4
        )
        if single_click:
            click = (SHIP_COL / (COLS - 1), SHIP_ROW / (ROWS - 1))
            points = [click] * 4
            params = RelocationParameters(corridor_half_width_m=30.0)
        else:
            points = _horizontal_curve(800)
            params = None
        target = locate_imaged_target(chip, points, geometry.scene_height, params)
        # One row is 0.43 m and one column 0.044 m of ground.
        assert np.linalg.norm(target.position - chip.ecef(SHIP_ROW, SHIP_COL)) < 1.5
        t, r = chip.zero_doppler(SHIP_ROW, SHIP_COL)
        assert target.time == pytest.approx(t, abs=2e-4)
        assert target.slant_range == pytest.approx(r, abs=0.5)
        # ~9 m across in rows (0.43 m ground each) plus the tiny column skew.
        assert 2.0 < target.half_extent_m < 10.0


class TestMapDrift:
    """Sub-aperture drift of an injected mover (for the optional v_a check)."""

    def test_map_drift_recovers_along_track_velocity(self, geometry, pixel_to_lonlat):
        """Sub-aperture drift of an injected mover gives its along-track speed."""
        data, chip, dt_col, ka = _setup(geometry, pixel_to_lonlat)
        t, _ = chip.zero_doppler(SHIP_ROW, SHIP_COL)
        v_eff = geometry.kinematics(t, chip.ecef(SHIP_ROW, SHIP_COL)).v_eff
        v_along = 1.0
        # A hull diagonal in ground geometry: ~40 m along track by ~40 m across.
        n = 25
        rows = SHIP_ROW + np.linspace(-45, 45, n)
        cols = SHIP_COL + np.linspace(-450, 450, n)
        _inject_ship(
            data,
            geometry,
            dt_col,
            ka,
            drift_per_s=2 * v_along / v_eff,
            rows=rows,
            cols=cols,
        )
        params = RelocationParameters(corridor_half_width_m=40.0)
        curve = _horizontal_curve(1800)

        targets = find_targets_along_curve(chip, curve, params)
        assert len(targets) == params.n_looks
        assert np.all(np.diff([t.col for t in targets]) != 0)

        measured, r2, looks = along_track_velocity(chip, curve, params)
        assert measured == pytest.approx(v_along, rel=0.15)
        assert r2 > 0.95
        assert len(looks) == params.n_looks
