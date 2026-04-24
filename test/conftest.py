"""Pytest configuration and fixtures for ICEYE Toolbox tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from qgis.core import QgsCoordinateReferenceSystem, QgsRasterLayer, QgsRectangle
from qgis.gui import QgsMapCanvas

from iceye_toolbox.core.metadata import IceyeMetadata, MetadataProvider
from iceye_toolbox.core.raster import read_slc_layer

# Bundled regression rasters (see test/fixtures/).
_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_WWGTZ2_STEM = "ICEYE_WWGTZ2_20251109T141525Z_6987409_X44_SLED"


def wwgtz2_fixture_tif(suffix: str) -> Path:
    """Path to a WWGTZ2 fixture GeoTIFF (suffix e.g. CROP, SHORT, COLOR)."""
    return _FIXTURES_DIR / f"{_WWGTZ2_STEM}_{suffix}_37a3f6c7.tif"


@pytest.fixture(scope="session", autouse=True)
def qgis_processing() -> None:
    """Initialize QGIS Processing framework."""
    # Import here: `processing` exists only on QGIS's Python path, not in a bare venv.
    from processing.core.Processing import Processing

    Processing.initialize()
    yield


@pytest.fixture
def map_canvas() -> QgsMapCanvas:
    """Create a QgsMapCanvas for testing."""
    canvas = QgsMapCanvas()
    canvas.setDestinationCrs(QgsCoordinateReferenceSystem("EPSG:4326"))
    return canvas


@pytest.fixture
def metadata(base_crop_layer):
    """Extract ICEYE metadata from the base crop layer."""
    metadata = MetadataProvider().get(base_crop_layer)
    return metadata


@pytest.fixture
def complex_data(base_crop_layer, metadata):
    """Read SLC complex data array from the base crop layer."""
    complex_data, _ = read_slc_layer(
        base_crop_layer, metadata.sar_observation_direction.lower(), metadata
    )
    return complex_data


@pytest.fixture
def base_crop_file() -> str:
    """Path to base crop file for integration tests."""
    return str(wwgtz2_fixture_tif("CROP"))


@pytest.fixture
def base_crop_layer(base_crop_file: str) -> QgsRasterLayer:
    """Create a QgsRasterLayer from the bundled crop fixture."""
    layer = QgsRasterLayer(base_crop_file, "ICEYE_WWGTZ2_fixture_crop")
    assert layer.isValid(), f"Failed to load test raster: {base_crop_file}"
    return layer
