"""Tests for the SLC geometry and target detection in core.target_finder.

Geometry (orbit, GCP geolocation) comes from the WWGTZ2 SLED crop fixture; the SLC
samples are synthetic clutter with an injected point-target ship, so the expected
position is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceye_toolbox.core.mover_relocation import local_geometry, locate_imaged_target
from iceye_toolbox.core.target_finder import (
    HullParameters,
    ProductGeometry,
    SlcChip,
    ecef_to_lonlat,
    gcp_pixel_to_lonlat,
    lonlat_to_ecef,
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

    def test_zero_doppler_geocode_round_trip(self, geometry, pixel_to_lonlat):
        """Range-Doppler inverse then geocode returns the same point."""
        assert geometry.scene_height == -3.0
        start, end = geometry.acquisition_window
        assert end - start == pytest.approx(28.789, abs=1e-3)
        lon, lat = pixel_to_lonlat(4000.5, 400.5)
        point = lonlat_to_ecef(lon, lat, geometry.scene_height)
        t, r = geometry.orbit.zero_doppler(point, 0.35)
        assert 701000 < r < 705000
        back = geometry.orbit.geocode(r, t, geometry.scene_height, point + 50.0)
        assert np.linalg.norm(back - point) < 0.01
        assert ecef_to_lonlat(back) == pytest.approx((lon, lat), abs=1e-8)
        # About 90 m of along-track shift per 1 m/s radial velocity.
        local = local_geometry(geometry, t, point)
        assert 80 < r * local.v_ground / local.v_eff2 < 100


class TestTargetInput:
    """Imaged position of an injected ship from a click."""

    def test_hull_centroid(self, geometry, properties, pixel_to_lonlat):
        """A click on the ship's end snaps to the hull centroid."""
        data = _clutter(properties)
        _inject_ship(data, properties, amplitude=3e4)
        chip = SlcChip(data, 0, 0, geometry, pixel_to_lonlat)
        target = locate_imaged_target(
            chip,
            SHIP_ROW,
            SHIP_COL + 300,
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
