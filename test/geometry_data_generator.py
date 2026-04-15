import json
from pathlib import Path

import numpy as np
import pytest

from ICEYE_toolbox.core.geometry import (
    WGS84Norm,
    ecef_to_geodetic,
    geodetic_to_ecef,
    get_geometry_from_metadata,
    sar_vectors_to_geometry,
)


def test_generate_expected_values(metadata):
    """Run this manually when you need to regenerate expected values."""
    generate_geometry_expected_values(metadata)
    pytest.skip("Generated expected values, skipping test")

def generate_geometry_expected_values(metadata, output_file=None):
    """
    Run this once to generate expected values for geometry tests.
    
    Usage:
        # In your test or as a standalone script:
        from conftest import metadata
        generate_geometry_expected_values(metadata)
        
    Or to just print without saving:
        generate_geometry_expected_values(metadata, output_file=None)
    """
    if output_file is None:
        output_file = Path(__file__).parent / "geometry_expected_values.json"

    expected = {
        "metadata_info": {
            "source_file": "ICEYE_TEST_W7JZMJ_20251002T212716Z_6384106_X49_SLH_CROP_4fe216db.tif",
            "generated_date": "2026-02-05",
            "description": "Expected values for comprehensive geometry.py testing"
        }
    }

    # 1. Coordinate conversions - test various points
    coord_test_cases = {
        "real_iceye_metadata": {
            "lat": metadata.proj_centroid['lat'],
            "lon": metadata.proj_centroid['lon'],
            "h": metadata.iceye_average_scene_height
        },
        "equator_prime_meridian": {"lat": 0.0, "lon": 0.0, "h": 0.0},
        "north_pole": {"lat": 90.0, "lon": 0.0, "h": 0.0},
        "south_pole": {"lat": -90.0, "lon": 0.0, "h": 0.0},
        "helsinki": {"lat": 60.1699, "lon": 24.9384, "h": 0.0},
        "sydney": {"lat": -33.8688, "lon": 151.2093, "h": 50.0},
        "high_altitude_equator": {"lat": 0.0, "lon": 0.0, "h": 600000.0},
        "mid_latitude_positive": {"lat": 45.0, "lon": 45.0, "h": 100.0},
        "mid_latitude_negative": {"lat": -45.0, "lon": -45.0, "h": 100.0},
    }

    expected["coordinate_conversions"] = {}
    for name, coords in coord_test_cases.items():
        x, y, z = geodetic_to_ecef(coords["lat"], coords["lon"], coords["h"])
        lat_back, lon_back = ecef_to_geodetic(x, y, z)

        expected["coordinate_conversions"][name] = {
            "input": coords,
            "expected_ecef": {"x": float(x), "y": float(y), "z": float(z)},
            "round_trip": {
                "lat": float(np.degrees(lat_back)),
                "lon": float(np.degrees(lon_back)),
                "lat_error": abs(float(np.degrees(lat_back)) - coords["lat"]),
                "lon_error": abs(float(np.degrees(lon_back)) - coords["lon"])
            }
        }

    # 2. WGS84 Normals
    expected["wgs84_normals"] = {}
    for name, data in expected["coordinate_conversions"].items():
        ecef = data["expected_ecef"]
        norm = WGS84Norm()(ecef["x"], ecef["y"], ecef["z"])

        expected["wgs84_normals"][name] = {
            "input_ecef": ecef,
            "expected_normal": [float(n) for n in norm],
            "magnitude": float(np.linalg.norm(norm))
        }

    # 3. SAR Geometry calculations - various scenarios
    sar_test_cases = {
        "real_metadata": {
            "AIM": np.array(geodetic_to_ecef(
                metadata.proj_centroid['lat'],
                metadata.proj_centroid['lon'],
                metadata.iceye_average_scene_height
            )),
            "P": metadata.center_aperture_position,
            "VEL": metadata.center_aperture_velocity
        },
        "descending_right_looking": {
            "AIM": np.array([2897560.783099828, 1351154.7831290446, 5500477.133938444]),
            "P": np.array([2668520.2749206917, 1272818.8796461378, 6301494.513717764]),
            "VEL": np.array([0.0, 7500.0, -500.0])
        },
        "ascending_left_looking": {
            "AIM": np.array([2897560.783099828, 1351154.7831290446, 5500477.133938444]),
            "P": np.array([3649617.50718747, 1663226.5206767116, 5692874.749775472]),
            "VEL": np.array([0.0, 7500.0, 500.0])
        },
        "equatorial_pass": {
            "AIM": np.array([6378137.0, 0.0, 0.0]),
            "P": np.array([6951479.938282574, 60664.64637150409, 604477.4056763334]),
            "VEL": np.array([0.0, 7500.0, 100.0])
        }
    }

    expected["sar_geometry"] = {}
    for name, vectors in sar_test_cases.items():
        geometry = sar_vectors_to_geometry(
            vectors["AIM"], vectors["P"], vectors["VEL"]
        )

        test_data = {
            "inputs": {
                "AIM": [float(x) for x in vectors["AIM"]],
                "P": [float(x) for x in vectors["P"]],
                "VEL": [float(x) for x in vectors["VEL"]]
            },
            "expected_radians": {
                "azimuth": float(geometry.azimuth),
                "graze": float(geometry.graze),
                "slope": float(geometry.slope),
                "squint": float(geometry.squint),
                "layover": float(geometry.layover),
                "multipath": float(geometry.multipath),
                "dca": float(geometry.dca),
                "tilt": float(geometry.tilt),
                "track": float(geometry.track),
                "felev": float(geometry.felev),
                "shadow": float(geometry.shadow),
                "north": float(geometry.north),
                "sense": float(geometry.sense),
                "look_side": geometry.look_side,
                "orbital_node": geometry.orbital_node
            }
        }

        # Also get degrees version
        if name == "real_metadata":
            geometry_deg = sar_vectors_to_geometry(
                vectors["AIM"], vectors["P"], vectors["VEL"]
            )
            geometry_deg.radian2deg()
            test_data["expected_degrees"] = {
                "azimuth": float(geometry_deg.azimuth),
                "graze": float(geometry_deg.graze),
                "slope": float(geometry_deg.slope),
                "squint": float(geometry_deg.squint),
                "layover": float(geometry_deg.layover),
                "multipath": float(geometry_deg.multipath),
                "dca": float(geometry_deg.dca),
                "tilt": float(geometry_deg.tilt),
                "track": float(geometry_deg.track),
                "felev": float(geometry_deg.felev),
                "shadow": float(geometry_deg.shadow),
                "north": float(geometry_deg.north)
            }

        expected["sar_geometry"][name] = test_data

    # 4. Unit conversions
    # (Add similar logic for other test sections...)

    # 5. Metadata integration
    geom_from_meta = get_geometry_from_metadata(metadata)
    expected["metadata_integration"] = {
        "from_metadata_object": {
            "metadata_source": "ICEYE test file",
            "expected_radians": {
                "azimuth": float(geom_from_meta.azimuth),
                "graze": float(geom_from_meta.graze),
                "slope": float(geom_from_meta.slope),
                "squint": float(geom_from_meta.squint),
                "layover": float(geom_from_meta.layover),
                "multipath": float(geom_from_meta.multipath),
                "dca": float(geom_from_meta.dca),
                "tilt": float(geom_from_meta.tilt),
                "track": float(geom_from_meta.track),
                "felev": float(geom_from_meta.felev),
                "shadow": float(geom_from_meta.shadow),
                "north": float(geom_from_meta.north),
                "sense": float(geom_from_meta.sense),
                "look_side": geom_from_meta.look_side,
                "orbital_node": geom_from_meta.orbital_node,
                "units": geom_from_meta.units
            }
        }
    }

    # Save or print
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(expected, f, indent=2)
        print(f"✓ Expected values saved to {output_file}")
    else:
        print(json.dumps(expected, indent=2))

    return expected
