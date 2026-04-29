import json
from pathlib import Path

import numpy as np
import pytest

from iceye_toolbox.core.geometry import (
    WGS84Norm,
    ecef_to_geodetic,
    geodetic_to_ecef,
    get_geometry_from_metadata,
    sar_vectors_to_geometry,
)


@pytest.fixture(scope="module")
def expected_values():
    """Load expected values from JSON file."""
    json_file = Path(__file__).parent / "geometry_expected_values.json"
    with Path(json_file).open() as f:
        return json.load(f)


class TestCoordinateConversion:
    """Robust coordinate conversion tests using expected values."""

    def test_all_coordinate_conversions(self, expected_values):
        """Test coordinate conversions against all expected values."""
        coord_data = expected_values["coordinate_conversions"]

        for location, data in coord_data.items():
            inp = data["input"]
            expected_ecef = data["expected_ecef"]
            expected_roundtrip = data["round_trip"]

            # Test forward conversion
            x, y, z = geodetic_to_ecef(inp["lat"], inp["lon"], inp["h"])

            # Allow small numerical differences
            assert abs(x - expected_ecef["x"]) < 1e-6, f"{location}: ECEF X mismatch"
            assert abs(y - expected_ecef["y"]) < 1e-6, f"{location}: ECEF Y mismatch"
            assert abs(z - expected_ecef["z"]) < 1e-6, f"{location}: ECEF Z mismatch"

            # Test round-trip conversion
            lat_back, lon_back = ecef_to_geodetic(x, y, z)
            lat_back_deg = np.degrees(lat_back)
            lon_back_deg = np.degrees(lon_back)

            # Use the expected error tolerance
            max_error = (
                max(expected_roundtrip["lat_error"], expected_roundtrip["lon_error"])
                + 1e-9
            )

            assert abs(lat_back_deg - inp["lat"]) < max(1e-6, max_error), (
                f"{location}: Latitude round-trip failed"
            )
            assert abs(lon_back_deg - inp["lon"]) < max(1e-6, max_error), (
                f"{location}: Longitude round-trip failed"
            )


class TestWGS84Norm:
    """Robust WGS84 normal vector tests."""

    def test_all_normals(self, expected_values):
        """Test WGS84 normals against all expected values."""
        norm_data = expected_values["wgs84_normals"]

        for location, data in norm_data.items():
            ecef = data["input_ecef"]
            expected_norm = np.array(data["expected_normal"])
            expected_mag = data["magnitude"]

            # Calculate normal
            norm = WGS84Norm()(ecef["x"], ecef["y"], ecef["z"])

            # Check magnitude is unity
            mag = np.linalg.norm(norm)
            assert abs(mag - expected_mag) < 1e-10, (
                f"{location}: Normal magnitude not unity: {mag}"
            )

            # Check normal vector matches expected (within tolerance)
            assert np.allclose(norm, expected_norm, atol=1e-10), (
                f"{location}: Normal vector mismatch"
            )

            # Additional property: normal should be a 3D vector
            assert norm.shape == (3,), f"{location}: Normal has wrong shape"


class TestSARGeometry:
    """Robust SAR geometry tests using expected values."""

    def test_all_sar_geometries(self, expected_values):
        """Test SAR geometry calculations against all expected scenarios."""
        sar_data = expected_values["sar_geometry"]

        for scenario, data in sar_data.items():
            inputs = data["inputs"]
            expected = data["expected_radians"]

            AIM = np.array(inputs["AIM"])
            P = np.array(inputs["P"])
            VEL = np.array(inputs["VEL"])

            # Calculate geometry
            geometry = sar_vectors_to_geometry(AIM, P, VEL)

            # Check all angles match expected values
            tolerance = 1e-9
            for angle in [
                "azimuth",
                "graze",
                "slope",
                "squint",
                "layover",
                "multipath",
                "dca",
                "tilt",
                "track",
                "felev",
                "shadow",
                "north",
            ]:
                actual = getattr(geometry, angle)
                expected_val = expected[angle]
                assert abs(actual - expected_val) < tolerance, (
                    f"{scenario}.{angle}: expected {expected_val}, got {actual}"
                )

            # Check sense
            assert geometry.sense == expected["sense"], f"{scenario}: sense mismatch"

            # Check look side and orbital node
            assert geometry.look_side == expected["look_side"], (
                f"{scenario}: look_side mismatch"
            )
            assert geometry.orbital_node == expected["orbital_node"], (
                f"{scenario}: orbital_node mismatch"
            )

            # Verify fundamental SAR property: slope >= graze
            assert geometry.slope >= geometry.graze - 1e-10, (
                f"{scenario}: slope should be >= graze"
            )

            # Verify units
            assert geometry.units == "radians"

    def test_unit_conversion_with_expected(self, expected_values):
        """Test that unit conversion produces expected degree values."""
        # Get the real metadata case
        sar_data = expected_values["sar_geometry"]["real_metadata"]
        inputs = sar_data["inputs"]
        expected_deg = sar_data["expected_degrees"]

        AIM = np.array(inputs["AIM"])
        P = np.array(inputs["P"])
        VEL = np.array(inputs["VEL"])

        geometry = sar_vectors_to_geometry(AIM, P, VEL)
        geometry.radian2deg()

        # Check conversion to degrees
        tolerance = 1e-8
        for angle in [
            "azimuth",
            "graze",
            "slope",
            "squint",
            "layover",
            "multipath",
            "dca",
            "tilt",
            "track",
            "felev",
            "shadow",
            "north",
        ]:
            actual = getattr(geometry, angle)
            expected_val = expected_deg[angle]
            assert abs(actual - expected_val) < tolerance, (
                f"degree.{angle}: expected {expected_val}, got {actual}"
            )

        assert geometry.units == "degrees"


class TestMetadataIntegration:
    """Test geometry extraction from metadata."""

    def test_geometry_from_metadata(self, metadata, expected_values):
        """Test get_geometry_from_metadata against expected values."""
        expected = expected_values["metadata_integration"]["from_metadata_object"][
            "expected_radians"
        ]

        geometry = get_geometry_from_metadata(metadata)

        tolerance = 1e-9
        for angle in [
            "azimuth",
            "graze",
            "slope",
            "squint",
            "layover",
            "multipath",
            "dca",
            "tilt",
            "track",
            "felev",
            "shadow",
            "north",
        ]:
            actual = getattr(geometry, angle)
            expected_val = expected[angle]
            assert abs(actual - expected_val) < tolerance, (
                f"from_metadata.{angle}: expected {expected_val}, got {actual}"
            )

        assert geometry.sense == expected["sense"]
        assert geometry.look_side == expected["look_side"]
        assert geometry.orbital_node == expected["orbital_node"]
        assert geometry.units == expected["units"]

    def test_consistency_multiple_calls(self, metadata):
        """Test that multiple calls produce identical results (deterministic)."""
        geom1 = get_geometry_from_metadata(metadata)
        geom2 = get_geometry_from_metadata(metadata)
        geom3 = get_geometry_from_metadata(metadata)

        # All values should be exactly identical
        for angle in [
            "azimuth",
            "graze",
            "slope",
            "squint",
            "layover",
            "multipath",
            "dca",
            "tilt",
            "track",
            "felev",
            "shadow",
            "north",
            "sense",
        ]:
            assert (
                getattr(geom1, angle) == getattr(geom2, angle) == getattr(geom3, angle)
            ), f"Non-deterministic {angle}"
