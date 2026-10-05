"""Tests for the SAR image geometry in core.target_finder.

Geometry (orbit, GCP geolocation) comes from the WWGTZ2 SLED crop fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from iceye_toolbox.core.mover_relocation import local_geometry
from iceye_toolbox.core.target_finder import (
    ProductGeometry,
    ecef_to_lonlat,
    gcp_pixel_to_lonlat,
    lonlat_to_ecef,
    read_iceye_properties,
)


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
