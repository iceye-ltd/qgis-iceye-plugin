"""ICEYE metadata container and provider for raster layers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray
from osgeo import gdal
from qgis.core import Qgis, QgsMessageLog, QgsRasterLayer


def parse_iso8601_datetime(value: str | None) -> datetime | None:
    """Parse ICEYE ISO 8601 datetime strings for Python 3.10+ compatibility.

    Python 3.10 ``datetime.fromisoformat`` rejects fractional seconds unless they
    use exactly six digits (e.g. ``.87+00:00`` fails; ``.870000+00:00`` works). ICEYE
    metadata may use shorter fractional parts. This normalizes the fraction, then
    parses.
    """
    if not value or not isinstance(value, str):
        return None
    s = value.strip().replace("Z", "+00:00")
    m = re.match(r"^(.*T\d{2}:\d{2}:\d{2})(\.\d+)?([+-]\d{2}:\d{2})?$", s)
    if not m:
        try:
            return datetime.fromisoformat(s)
        except (ValueError, TypeError):
            return None
    base, frac, tz = m.group(1), m.group(2), m.group(3) or ""
    if frac:
        digits = frac[1:]
        digits6 = (digits + "000000")[:6]
        s = base + "." + digits6 + tz
    else:
        s = base + tz
    try:
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


@dataclass
class IceyeMetadata:
    """Container for ICEYE metadata properties from STAC/GDAL."""

    # Basic properties
    datetime: str | None = None
    created: str | None = None
    start_datetime: str | None = None
    end_datetime: str | None = None
    platform: str | None = None
    constellation: str | None = None

    # SAR properties
    sar_instrument_mode: str | None = None
    sar_frequency_band: str | None = None
    sar_center_frequency: float | None = None
    sar_polarizations: list[str] | None = None
    sar_product_type: str | None = None
    sar_resolution_range: float | None = None
    sar_resolution_azimuth: float | None = None
    sar_pixel_spacing_range: float | None = None
    sar_pixel_spacing_azimuth: float | None = None
    sar_looks_range: int | None = None
    sar_looks_azimuth: int | None = None
    sar_observation_direction: str | None = None

    # Satellite properties
    sat_orbit_state: str | None = None

    # View properties
    view_off_nadir: float | None = None
    view_azimuth: float | None = None
    view_incidence_angle: float | None = None

    # Processing properties
    processing_software: dict[str, str] | None = None

    # Projection properties
    proj_code: str | None = None
    proj_shape: list[int] | None = None
    proj_centroid: dict[str, float] | None = None
    proj_transform: list[float] | None = None

    # Raster properties
    raster_bands: list[dict[str, Any]] | None = None

    # ICEYE specific properties
    iceye_image_id: int | None = None
    iceye_image_reference: str | None = None
    iceye_scene_id: str | None = None
    iceye_run_name: str | None = None
    iceye_frame_number: int | None = None
    iceye_filename: str | None = None
    iceye_resolution_range_near: float | None = None
    iceye_resolution_range_far: float | None = None
    iceye_looks_range_bandwidth: float | None = None
    iceye_looks_range_overlap: float | None = None
    iceye_looks_azimuth_bandwidth: float | None = None
    iceye_looks_azimuth_overlap: float | None = None
    iceye_synthetic_aperture_angle: float | None = None
    iceye_pulse_bandwidth: float | None = None
    iceye_pulse_duration: float | None = None
    iceye_acquisition_prf: float | None = None
    iceye_acquisition_range_sampling_rate: float | None = None
    iceye_zero_doppler_start_datetime: str | None = None
    iceye_zero_doppler_end_datetime: str | None = None
    iceye_doppler_centroid_datetimes: list[str] | None = None
    iceye_orientation: str | None = None
    iceye_coordinate_frame: str | None = None
    iceye_orbit_mean_altitude: float | None = None
    iceye_orbit_precision: str | None = None
    iceye_incidence_angle_near: float | None = None
    iceye_incidence_angle_far: float | None = None
    iceye_grazing_angle_near: float | None = None
    iceye_grazing_angle: float | None = None
    iceye_grazing_angle_far: float | None = None
    iceye_squint_angle: float | None = None
    iceye_layover_angle: float | None = None
    iceye_shadow_angle: float | None = None
    iceye_doppler_cone_angle: float | None = None
    iceye_range_near: float | None = None
    iceye_range: float | None = None
    iceye_range_far: float | None = None
    iceye_average_scene_height: float | None = None
    iceye_extent_range: float | None = None
    iceye_extent_azimuth: float | None = None
    iceye_processing_end_datetime: str | None = None
    iceye_processing_bandwidth_range: float | None = None
    iceye_processing_bandwidth_azimuth: float | None = None
    iceye_processing_prf: float | None = None
    iceye_processing_mode: str | None = None
    iceye_calibration_factor: float | None = None
    iceye_amplitude_mapping: dict[str, Any] | None = None

    center_aperture_position: NDArray[np.float64] | None = None
    center_aperture_velocity: NDArray[np.float64] | None = None

    def view(
        self,
        keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return a dictionary subset of the dataclass attributes.

        Parameters
        ----------
        keys : list of str or None, optional
            Field names to include. If None, returns all dataclass attributes.

        Returns
        -------
        dict
            Requested field values from the dataclass.
        """
        # Get all dataclass field names
        all_field_names = {f.name for f in fields(self)}

        # If no keys specified, return all attributes
        if keys is None:
            return {name: getattr(self, name) for name in all_field_names}

        # Filter to only include keys that exist in the dataclass
        valid_keys = [k for k in keys if k in all_field_names]

        # Return dictionary with requested field values
        return {name: getattr(self, name) for name in valid_keys}


class MetadataProvider:
    """Provider that extracts, parses, caches, and formats metadata for layers."""

    def __init__(self) -> None:
        """Initialize the metadata provider with an empty cache."""
        self._cache: dict[str, IceyeMetadata] = {}

    def get(self, layer: QgsRasterLayer | None) -> IceyeMetadata | None:
        """Get metadata for a layer, using cache if available.

        Parameters
        ----------
        layer : QgsRasterLayer or None
            The raster layer to get metadata for.

        Returns
        -------
        IceyeMetadata or None
            Parsed metadata, or None if not available.
        """
        if layer is None:
            return None

        if not isinstance(layer, QgsRasterLayer):
            QgsMessageLog.logMessage(
                f"Layer {layer.name()} is not a raster layer",
                "ICEYE Toolbox",
                level=Qgis.Warning,
            )
            return None

        # Return cached metadata if availables
        layer_id = layer.id()
        if layer_id in self._cache:
            QgsMessageLog.logMessage(
                f"Returning cached metadata for layer {layer_id}",
                "ICEYE Toolbox",
                level=Qgis.Info,
            )
            return self._cache[layer_id]

        # Load and cache metadata
        QgsMessageLog.logMessage(
            f"Loading metadata for layer {layer_id}",
            "ICEYE Toolbox",
            level=Qgis.Info,
        )
        return self.load(layer)

    def load(self, layer: QgsRasterLayer) -> IceyeMetadata | None:
        """Load and parse metadata from layer and cache it.

        Parameters
        ----------
        layer : QgsRasterLayer
            The raster layer to load metadata from.

        Returns
        -------
        IceyeMetadata or None
            Parsed metadata, or None on error.
        """
        if not isinstance(layer, QgsRasterLayer):
            QgsMessageLog.logMessage(
                f"Layer {layer.id()} is not a raster layer",
                "ICEYE Toolbox",
                level=Qgis.Critical,
            )
            return None

        layer_id = layer.id()
        if not is_iceye_layer(layer):
            QgsMessageLog.logMessage(
                f"Layer {layer.id()} is not an ICEYE layer",
                "ICEYE Toolbox",
                level=Qgis.Warning,
            )
            return None

        dataset = gdal.Open(layer.source())
        if dataset is None:
            QgsMessageLog.logMessage(
                f"Failed to open dataset for layer {layer.id()}",
                "ICEYE Toolbox",
                level=Qgis.Critical,
            )
            return None

        try:
            properties = json.loads(dataset.GetMetadata()["ICEYE_PROPERTIES"])
        except (json.JSONDecodeError, KeyError):
            QgsMessageLog.logMessage(
                "Error decoding ICEYE_PROPERTIES",
                "ICEYE Toolbox",
                level=Qgis.Warning,
            )
            return None
        try:
            # Get center of aperture position and velocity
            coa_idx = len(properties.get("iceye:orbit_states", [])) // 2
            coa_pos = np.array(
                properties.get("iceye:orbit_states", [])[coa_idx]["position"],
                dtype=np.float64,
            )
            coa_vel = np.array(
                properties.get("iceye:orbit_states", [])[coa_idx]["velocity"],
                dtype=np.float64,
            )

            # Get scene center position and height

            self._cache[layer_id] = IceyeMetadata(
                datetime=properties.get("datetime"),
                created=properties.get("created"),
                start_datetime=properties.get("start_datetime"),
                end_datetime=properties.get("end_datetime"),
                platform=properties.get("platform"),
                constellation=properties.get("constellation"),
                sar_instrument_mode=properties.get("sar:instrument_mode"),
                sar_frequency_band=properties.get("sar:frequency_band"),
                sar_center_frequency=properties.get("sar:center_frequency"),
                sar_polarizations=properties.get("sar:polarizations"),
                sar_product_type=properties.get("sar:product_type"),
                sar_resolution_range=properties.get("sar:resolution_range"),
                sar_resolution_azimuth=properties.get("sar:resolution_azimuth"),
                sar_pixel_spacing_range=properties.get("sar:pixel_spacing_range"),
                sar_pixel_spacing_azimuth=properties.get("sar:pixel_spacing_azimuth"),
                sar_looks_range=properties.get("sar:looks_range"),
                sar_looks_azimuth=properties.get("sar:looks_azimuth"),
                sar_observation_direction=properties.get("sar:observation_direction"),
                sat_orbit_state=properties.get("sat:orbit_state"),
                view_off_nadir=properties.get("view:off_nadir"),
                view_azimuth=properties.get("view:azimuth"),
                view_incidence_angle=properties.get("view:incidence_angle"),
                processing_software=properties.get("processing:software"),
                proj_code=properties.get("proj:code"),
                proj_shape=properties.get("proj:shape"),
                proj_centroid=properties.get("proj:centroid"),
                proj_transform=properties.get("proj:transform"),
                raster_bands=properties.get("raster:bands"),
                iceye_image_id=properties.get("iceye:image_id"),
                iceye_image_reference=properties.get("iceye:image_reference"),
                iceye_scene_id=properties.get("iceye:scene_id"),
                iceye_run_name=properties.get("iceye:run_name"),
                iceye_frame_number=properties.get("iceye:frame_number"),
                iceye_filename=properties.get("iceye:filename"),
                iceye_resolution_range_near=properties.get(
                    "iceye:resolution_range_near"
                ),
                iceye_resolution_range_far=properties.get("iceye:resolution_range_far"),
                iceye_looks_range_bandwidth=properties.get(
                    "iceye:looks_range_bandwidth"
                ),
                iceye_looks_range_overlap=properties.get("iceye:looks_range_overlap"),
                iceye_looks_azimuth_bandwidth=properties.get(
                    "iceye:looks_azimuth_bandwidth"
                ),
                iceye_looks_azimuth_overlap=properties.get(
                    "iceye:looks_azimuth_overlap"
                ),
                iceye_synthetic_aperture_angle=properties.get(
                    "iceye:synthetic_aperture_angle"
                ),
                iceye_pulse_bandwidth=properties.get("iceye:pulse_bandwidth"),
                iceye_pulse_duration=properties.get("iceye:pulse_duration"),
                iceye_acquisition_prf=properties.get("iceye:acquisition_prf"),
                iceye_acquisition_range_sampling_rate=properties.get(
                    "iceye:acquisition_range_sampling_rate"
                ),
                iceye_zero_doppler_start_datetime=properties.get(
                    "iceye:zero_doppler_start_datetime"
                ),
                iceye_zero_doppler_end_datetime=properties.get(
                    "iceye:zero_doppler_end_datetime"
                ),
                iceye_doppler_centroid_datetimes=properties.get(
                    "iceye:doppler_centroid_datetimes"
                ),
                iceye_orientation=properties.get("iceye:orientation"),
                iceye_coordinate_frame=properties.get("iceye:coordinate_frame"),
                iceye_orbit_mean_altitude=properties.get("iceye:orbit_mean_altitude"),
                iceye_orbit_precision=properties.get("iceye:orbit_precision"),
                iceye_incidence_angle_near=properties.get("iceye:incidence_angle_near"),
                iceye_incidence_angle_far=properties.get("iceye:incidence_angle_far"),
                iceye_grazing_angle_near=properties.get("iceye:grazing_angle_near"),
                iceye_grazing_angle=properties.get("iceye:grazing_angle"),
                iceye_grazing_angle_far=properties.get("iceye:grazing_angle_far"),
                iceye_squint_angle=properties.get("iceye:squint_angle"),
                iceye_layover_angle=properties.get("iceye:layover_angle"),
                iceye_shadow_angle=properties.get("iceye:shadow_angle"),
                iceye_doppler_cone_angle=properties.get("iceye:doppler_cone_angle"),
                iceye_range_near=properties.get("iceye:range_near"),
                iceye_range=properties.get("iceye:range"),
                iceye_range_far=properties.get("iceye:range_far"),
                iceye_average_scene_height=properties.get("iceye:average_scene_height"),
                iceye_extent_range=properties.get("iceye:extent_range"),
                iceye_extent_azimuth=properties.get("iceye:extent_azimuth"),
                iceye_processing_end_datetime=properties.get(
                    "iceye:processing_end_datetime"
                ),
                iceye_processing_bandwidth_range=properties.get(
                    "iceye:processing_bandwidth_range"
                ),
                iceye_processing_bandwidth_azimuth=properties.get(
                    "iceye:processing_bandwidth_azimuth"
                ),
                iceye_processing_prf=properties.get("iceye:processing_prf"),
                iceye_processing_mode=properties.get("iceye:processing_mode"),
                iceye_calibration_factor=properties.get("iceye:calibration_factor"),
                iceye_amplitude_mapping=properties.get("iceye:amplitude_mapping"),
                center_aperture_position=coa_pos,
                center_aperture_velocity=coa_vel,
            )
            return self._cache[layer_id]
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Error loading metadata for layer {layer.id()}: {e!s}",
                "ICEYE Toolbox",
                level=Qgis.Critical,
            )
            return None

    def on_layer_changed(self, layer: QgsRasterLayer) -> None:
        """Handle layer change event."""
        # load when new layer
        # clean when layer is dropped
        pass


def is_iceye_layer(layer: QgsRasterLayer | None) -> bool:
    """Check if layer has ICEYE identifier in name or source.

    Parameters
    ----------
    layer : QgsRasterLayer or None
        The layer to check.

    Returns
    -------
    bool
        True if layer is an ICEYE layer, False otherwise.
    """
    if layer is None or not isinstance(layer, QgsRasterLayer):
        layer_id = layer.id() if layer is not None else "None"
        QgsMessageLog.logMessage(
            f"Layer {layer_id} is not a raster layer",
            "ICEYE Toolbox",
            level=Qgis.Warning,
        )
        return False

    layer_name = layer.name()
    source = layer.source()

    return layer_name.startswith("ICEYE_") or "ICEYE_" in source.split("/")[-1]
