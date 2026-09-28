"""Tests for the moving-ship relocation in core.target_finder.

Geometry (orbit, Doppler centroid, GCP geolocation) comes from the WWGTZ2 SLED crop
fixture; the SLC samples are synthetic clutter with injected point-target ships, so the
expected Doppler, drift and displacement are known exactly.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from iceye_toolbox.core.target_finder import (
    DOPPLER_SIGN,
    FLAG_ILL_CONDITIONED,
    FLAG_SIGN_UNVERIFIED,
    METHOD_DOPPLER,
    METHOD_MAP_DRIFT,
    ProductGeometry,
    RelocationParameters,
    SlcChip,
    bezier_points,
    correct_truncation,
    correlation_doppler,
    corridor_mask,
    dilate_columns,
    ecef_to_geodetic,
    find_targets_along_curve,
    gcp_pixel_to_lonlat,
    lonlat_to_ecef,
    patch_to_file_layout,
    read_iceye_properties,
    relocate_mover,
    truncation_bias_curve,
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
    """GCP geolocation of the crop fixture."""
    return gcp_pixel_to_lonlat(crop_path)


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
    doppler_hz: float = 0.0,
    drift_per_s: float = 0.0,
    rows: np.ndarray,
    cols: np.ndarray,
    amplitude: float = 3000.0,
) -> None:
    """Add band-limited point scatterers at (rows, cols).

    ``doppler_hz`` shifts their spectrum inside the processing window; ``drift_per_s``
    is the look position change d(t_img)/d(tau) caused by a Doppler-rate mismatch.
    """
    f = DOPPLER_SIGN * np.fft.fftfreq(COLS, dt_col)
    band = geometry.processed_azimuth_bandwidth
    support = (np.abs(f) <= band / 2) & (np.abs(f - doppler_hz) <= band / 2)
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


class TestGeometry:
    """Orbit fit, range-Doppler inverse and geocoding on the fixture metadata."""

    def test_metadata_parsing(self, geometry):
        """ICEYE_PROPERTIES fields land in ProductGeometry."""
        assert geometry.instrument_mode == "spotlight"
        assert geometry.look_side == "left"
        assert geometry.azimuth_time_spacing == pytest.approx(1 / 163152.122)
        assert len(geometry.dc_coeffs) == len(geometry.dc_times) == 10

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

    def test_spotlight_doppler_centroid_tracks_fm_rate(self, geometry, pixel_to_lonlat):
        """In Dwell the DC slope nearly equals Ka: the reason for FLAG_ILL_CONDITIONED."""
        lon, lat = pixel_to_lonlat(4000.5, 400.5)
        point = lonlat_to_ecef(lon, lat, geometry.scene_height)
        t, r = geometry.orbit.zero_doppler(point, 0.35)
        ratio = (
            geometry.doppler_centroid_rate(t, r) / geometry.kinematics(t, point).fm_rate
        )
        assert ratio == pytest.approx(1.0, abs=2e-3)


class TestPrimitives:
    """Curve, masks and estimators on synthetic arrays."""

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

    def test_dilate_columns(self):
        """Dilation grows a mask along columns only."""
        mask = np.zeros((2, 10), bool)
        mask[0, 5] = True
        out = dilate_columns(mask, 2)
        assert np.flatnonzero(out[0]).tolist() == [3, 4, 5, 6, 7]
        assert not out[1].any()

    def test_correlation_doppler_recovers_tone(self):
        """A pure tone is estimated exactly, signed by time direction."""
        dt = 1e-4
        f0 = 123.0
        n = np.arange(400)
        data = np.tile(np.exp(2j * np.pi * f0 * n * dt), (5, 1))
        est = correlation_doppler(data, np.ones_like(data, bool), dt)
        assert est.frequency_hz == pytest.approx(DOPPLER_SIGN * f0, rel=1e-6)
        assert est.coherence == pytest.approx(1.0)
        # Reversed time direction flips the sign.
        est_rev = correlation_doppler(data, np.ones_like(data, bool), -dt)
        assert est_rev.frequency_hz == pytest.approx(-DOPPLER_SIGN * f0, rel=1e-6)

    def test_truncation_bias_halves_full_band_offset(self):
        """A full-band target cut by the window reads half its offset."""
        band, fs = 140e3, 163e3
        offsets = np.array([-40e3, 0.0, 20e3])
        readings = truncation_bias_curve(band, fs, offsets)
        assert readings == pytest.approx(offsets / 2, abs=fs / 8192 * 2)
        assert correct_truncation(10e3, band, fs) == pytest.approx(20e3, rel=0.02)

    def test_patch_to_file_layout_inverts_read_orientation(self):
        """Undoes the shadows-down orientation of read_slc_layer."""
        file_data = np.arange(12).reshape(3, 4)
        for left in (False, True):
            patch = (np.fliplr(file_data) if left else file_data).T
            assert np.array_equal(patch_to_file_layout(patch, left), file_data)


class TestRelocation:
    """End-to-end relocation with injected ships on fixture geometry."""

    def _setup(self, geometry, pixel_to_lonlat):
        data = _clutter(geometry)
        chip = _chip(data, geometry, pixel_to_lonlat)
        dt_col = chip.azimuth_time_step(SHIP_ROW, SHIP_COL)
        t, _ = chip.zero_doppler(SHIP_ROW, SHIP_COL)
        ka = geometry.kinematics(t, chip.ecef(SHIP_ROW, SHIP_COL)).fm_rate
        return data, chip, dt_col, ka

    def test_dwell_doppler_is_flagged_ill_conditioned(self, geometry, pixel_to_lonlat):
        """Dwell DC tracks Ka, so the Doppler method is flagged and skipped."""
        data, chip, dt_col, ka = self._setup(geometry, pixel_to_lonlat)
        _inject_ship(
            data,
            geometry,
            dt_col,
            ka,
            doppler_hz=320.0,
            rows=HULL_ROWS,
            cols=HULL_COLS,
            amplitude=30000.0,
        )
        result = relocate_mover(chip, _horizontal_curve(800))
        assert FLAG_SIGN_UNVERIFIED in result.flags
        assert FLAG_ILL_CONDITIONED in result.flags
        assert result.amplification > 100
        assert result.method != METHOD_DOPPLER
        # The injected offset is still measured against the window (the centroid
        # reads ~10 % high on this synthetic hull).
        assert result.f_ship_hz - result.f_ref_meta_hz == pytest.approx(320.0, abs=50)

    def test_doppler_relocation_when_well_conditioned(self, geometry, pixel_to_lonlat):
        """With a constant DC (stripmap-like) df maps straight to the displacement."""
        flat = dataclasses.replace(
            geometry, dc_coeffs=[np.array([0.0])], dc_times=np.array([0.0])
        )
        data, chip, dt_col, ka = self._setup(flat, pixel_to_lonlat)
        df = 2 * 3.0 / flat.wavelength  # 3 m/s towards the radar
        _inject_ship(
            data,
            flat,
            dt_col,
            ka,
            doppler_hz=df,
            rows=HULL_ROWS,
            cols=HULL_COLS,
            amplitude=30000.0,
        )
        result = relocate_mover(chip, _horizontal_curve(800))
        assert result.method == METHOD_DOPPLER
        assert result.amplification == pytest.approx(1.0, abs=1e-6)
        assert result.v_r == pytest.approx(3.0, abs=0.4)
        # The sea-clutter ring reference is noise limited (~50 Hz on flat spectra).
        assert result.doppler_ring.v_r == pytest.approx(result.v_r, abs=1.0)

        p_img = lonlat_to_ecef(*result.imaged_lonlat, flat.scene_height)
        t_img, r_img = flat.orbit.zero_doppler(p_img, 0.35)
        kin = flat.kinematics(t_img, p_img)
        expected_dx = r_img * result.v_r * kin.v_ground / kin.v_eff**2
        assert result.dx_m == pytest.approx(expected_dx, rel=0.02)

        # True position sits dt earlier in zero-Doppler time at the same slant range.
        lon, lat = result.true_lonlat
        p_true = lonlat_to_ecef(lon, lat, flat.scene_height)
        t_true, r_true = flat.orbit.zero_doppler(p_true, t_img)
        assert t_true == pytest.approx(t_img - result.dt_s, abs=1e-5)
        assert r_true == pytest.approx(r_img, abs=0.05)
        assert np.linalg.norm(p_true - p_img) == pytest.approx(
            abs(expected_dx), rel=0.05
        )

    def test_map_drift_recovers_along_track_velocity(self, geometry, pixel_to_lonlat):
        """Sub-aperture drift of an injected mover gives its along-track speed."""
        data, chip, dt_col, ka = self._setup(geometry, pixel_to_lonlat)
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
        cols_found = [t.col for t in targets]
        assert np.all(np.diff(cols_found) != 0)

        result = relocate_mover(chip, curve, params)
        assert result.v_along == pytest.approx(v_along, rel=0.15)
        assert result.v_along_fit_r2 > 0.95
        assert result.method == METHOD_MAP_DRIFT
        assert result.speed is not None and result.heading_deg is not None
        # Displacement is consistent with the implied radial velocity.
        assert result.dt_s == pytest.approx(
            2 * result.v_r / geometry.wavelength / ka, rel=0.02
        )
        assert result.attributes()["method"] == METHOD_MAP_DRIFT
