"""Tests for the SLC geometry and target detection in core.target_finder.

Geometry (orbit, GCP geolocation) comes from the WWGTZ2 SLED crop fixture; the SLC
samples are synthetic clutter with an injected point-target ship, so the expected
position is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceye_toolbox.core.mover_relocation import locate_imaged_target
from iceye_toolbox.core.target_finder import (
    HullParameters,
    ProductGeometry,
    SlcChip,
    ecef_to_lonlat,
    ellipse_mask,
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
def properties(crop_path) -> dict:
    """Read the decoded ICEYE_PROPERTIES of the crop fixture."""
    return read_iceye_properties(crop_path)


@pytest.fixture(scope="module")
def geometry(properties) -> ProductGeometry:
    """Relocation geometry parsed from the crop fixture metadata."""
    return ProductGeometry.from_properties(properties)


@pytest.fixture(scope="module")
def pixel_to_lonlat(crop_path):
    """GCP geolocation of the crop fixture, released before GDAL shuts down."""
    transform = gcp_pixel_to_lonlat(crop_path)
    yield transform
    transform.close()


def _azimuth_band(properties: dict) -> np.ndarray:
    """Return the processed azimuth band as a mask over the column FFT bins."""
    f = np.fft.fftfreq(COLS, 1.0 / properties["iceye:processing_prf"])
    return np.abs(f) <= properties["iceye:processing_bandwidth_azimuth"] / 2


def _clutter(properties: dict, seed: int = 0) -> np.ndarray:
    """Complex Gaussian clutter with the SLC's azimuth spectrum (flat over the band)."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal((ROWS, COLS)) + 1j * rng.standard_normal((ROWS, COLS))
    band = _azimuth_band(properties)
    noise = np.fft.ifft(np.fft.fft(noise, axis=1) * band[None, :], axis=1)
    return (noise * 10.0).astype(np.complex64)


def _inject_ship(data: np.ndarray, properties: dict, amplitude: float) -> None:
    """Add band-limited point scatterers along the hull."""
    band = _azimuth_band(properties)
    for r, c in zip(HULL_ROWS, HULL_COLS):
        impulse = np.zeros(COLS, dtype=np.complex128)
        impulse[int(c)] = amplitude
        data[int(r), :] += np.fft.ifft(np.fft.fft(impulse) * band).astype(np.complex64)


class TestGeometry:
    """Orbit fit, range-Doppler inverse and geocoding on the fixture metadata."""

    def test_metadata_parsing(self, geometry):
        """ICEYE_PROPERTIES fields land in ProductGeometry."""
        assert geometry.scene_height == -3.0
        assert len(geometry.dc_times) == 10
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
        assert ecef_to_lonlat(back) == pytest.approx((lon, lat), abs=1e-8)

    def test_azimuth_shift_per_mps(self, geometry, pixel_to_lonlat):
        """Azimuth shift per 1 m/s radial velocity, R * V_g / V_eff^2, is ~90 m."""
        lon, lat = pixel_to_lonlat(4000.5, 400.5)
        point = lonlat_to_ecef(lon, lat, geometry.scene_height)
        t, r = geometry.orbit.zero_doppler(point, 0.35)
        kin = geometry.kinematics(t, point)
        assert kin.slant_range == pytest.approx(r)
        assert 80 < r * kin.v_ground / kin.v_eff**2 < 100


class TestPrimitives:
    """Masks and layout helpers on synthetic arrays."""

    def test_ellipse_mask_is_anisotropic(self):
        """The click disc honours separate row and column half-widths."""
        mask = ellipse_mask((100, 100), 50.0, 50.0, half_rows=5, half_cols=10)
        assert mask[45, 50] and not mask[44, 50]
        assert mask[55, 50] and not mask[56, 50]
        assert mask[50, 40] and not mask[50, 39]
        assert mask[50, 60] and not mask[50, 61]

    def test_patch_to_file_layout_inverts_read_orientation(self):
        """Undoes the shadows-down orientation of read_slc_layer."""
        file_data = np.arange(12).reshape(3, 4)
        for left in (False, True):
            patch = (np.fliplr(file_data) if left else file_data).T
            assert np.array_equal(patch_to_file_layout(patch, left), file_data)


class TestTargetInput:
    """Imaged position of an injected ship from a click."""

    @pytest.mark.parametrize("offset_cols", [0, 300])
    def test_hull_centroid(self, geometry, properties, pixel_to_lonlat, offset_cols):
        """The hull centroid lands on the injected ship, wherever on it the click is."""
        data = _clutter(properties)
        _inject_ship(data, properties, amplitude=3e4)
        chip = SlcChip(data, 0, 0, geometry, pixel_to_lonlat)
        target = locate_imaged_target(
            chip,
            SHIP_ROW,
            SHIP_COL + offset_cols,
            geometry.scene_height,
            HullParameters(corridor_half_width_m=30.0),
        )
        ship = chip.ecef(SHIP_ROW, SHIP_COL)
        # One row is 0.43 m and one column 0.044 m of ground.
        assert np.linalg.norm(target.position - ship) < 1.5
        t, r = geometry.orbit.zero_doppler(ship, 0.35)
        assert target.time == pytest.approx(t, abs=2e-4)
        assert target.slant_range == pytest.approx(r, abs=0.5)
        # ~9 m across in rows (0.43 m ground each) plus the tiny column skew.
        assert 2.0 < target.half_extent_m < 10.0
