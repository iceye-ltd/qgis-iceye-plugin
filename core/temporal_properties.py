"""Temporal properties for ICEYE raster layers."""

from __future__ import annotations

from datetime import datetime, timedelta

from qgis.core import (
    Qgis,
    QgsDateTimeRange,
    QgsMessageLog,
    QgsRaster,
    QgsRasterDataProvider,
    QgsRasterLayer,
)
from qgis.PyQt.QtCore import QDateTime, Qt

from .metadata import IceyeMetadata, parse_iso8601_datetime


def _parse_datetime(iso_str: str | None) -> datetime | None:
    """Parse ISO datetime string to Python datetime (UTC).

    Parameters
    ----------
    iso_str : str or None
        ISO 8601 datetime string, or None.

    Returns
    -------
    datetime or None
        Parsed datetime in UTC, or None if invalid or missing.
    """
    return parse_iso8601_datetime(iso_str)


def _to_qdatetime_utc(dt: datetime) -> QDateTime:
    """Convert Python datetime (UTC) to QDateTime with explicit UTC timezone.

    Parameters
    ----------
    dt : datetime
        Python datetime in UTC.

    Returns
    -------
    QDateTime
        QDateTime in UTC.
    """
    msecs = int(dt.timestamp() * 1000)
    return QDateTime.fromMSecsSinceEpoch(msecs, Qt.UTC)


def apply_temporal_properties(layer: QgsRasterLayer, metadata: IceyeMetadata) -> bool:
    """Apply fixed temporal range to a standard ICEYE layer (SLC, crop, focus).

    Uses metadata.start_datetime and metadata.end_datetime for the acquisition range.

    Parameters
    ----------
    layer : QgsRasterLayer
        Raster layer to configure.
    metadata : IceyeMetadata
        ICEYE metadata with start_datetime and end_datetime.

    Returns
    -------
    bool
        True if temporal properties were applied, False otherwise.
    """
    if not metadata:
        QgsMessageLog.logMessage(
            "No metadata provided for temporal properties",
            "ICEYE Toolbox",
            level=Qgis.Warning,
        )
        return False

    if not layer:
        QgsMessageLog.logMessage(
            "No layer provided for temporal properties",
            "ICEYE Toolbox",
            level=Qgis.Warning,
        )
        return False

    if not isinstance(layer, QgsRasterLayer):
        QgsMessageLog.logMessage(
            f"Layer {layer.name() if hasattr(layer, 'name') else 'unknown'} is not a raster layer",
            "ICEYE Toolbox",
            level=Qgis.Warning,
        )
        return False

    start_dt = _parse_datetime(metadata.start_datetime)
    end_dt = _parse_datetime(metadata.end_datetime)

    if not start_dt or not end_dt:
        QgsMessageLog.logMessage(
            f"Invalid datetime values in metadata for layer {layer.name()}: start={metadata.start_datetime}, end={metadata.end_datetime}",
            "ICEYE Toolbox",
            level=Qgis.Warning,
        )
        return False

    try:
        t_props = layer.temporalProperties()
        if not t_props:
            QgsMessageLog.logMessage(
                f"No temporal properties available for layer {layer.name()}",
                "ICEYE Toolbox",
                level=Qgis.Warning,
            )
            return False

        # Use QDateTime with explicit UTC so Temporal Controller compares correctly
        start_qdt = _to_qdatetime_utc(start_dt)
        end_qdt = _to_qdatetime_utc(end_dt)
        dt_range = QgsDateTimeRange(start_qdt, end_qdt)
        t_props.setMode(Qgis.RasterTemporalMode.FixedTemporalRange)
        t_props.setFixedTemporalRange(dt_range)
        t_props.setIsActive(True)
        return True
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to apply temporal properties to {layer.name()}: {e!s}",
            "ICEYE Toolbox",
            level=Qgis.Warning,
        )
        return False


def apply_temporal_properties_for_frames(
    layer: QgsRasterLayer,
    metadata: IceyeMetadata,
) -> bool:
    """Apply per-band temporal ranges to a multi-frame layer (VID, short).

    Each band represents a fraction of the acquisition temporal range.
    Number of frames = layer.bandCount().

    Parameters
    ----------
    layer : QgsRasterLayer
        Multi-band raster layer.
    metadata : IceyeMetadata
        ICEYE metadata with start_datetime and end_datetime.

    Returns
    -------
    bool
        True if temporal properties were applied, False otherwise.
    """
    if not metadata or not layer or not isinstance(layer, QgsRasterLayer):
        return False

    num_bands = layer.bandCount()
    if num_bands < 2:
        return apply_temporal_properties(layer, metadata)

    p = layer.dataProvider()
    if isinstance(p, QgsRasterDataProvider):
        ci = p.colorInterpretation(num_bands)
        if ci == QgsRaster.AlphaBand:
            num_bands -= 1

    start_dt = _parse_datetime(metadata.start_datetime)
    end_dt = _parse_datetime(metadata.end_datetime)

    if not start_dt or not end_dt:
        return False

    total_time_seconds = (end_dt - start_dt).total_seconds()
    if total_time_seconds <= 0:
        return False

    band_range_seconds = total_time_seconds / num_bands
    try:
        t_props = layer.temporalProperties()
        if not t_props:
            return False

        band_ranges = {}
        for i in range(num_bands):
            band_start_offset = i * band_range_seconds
            band_end_offset = band_start_offset + band_range_seconds
            band_start = start_dt + timedelta(seconds=band_start_offset)
            band_end = start_dt + timedelta(seconds=band_end_offset)
            band_ranges[i + 1] = QgsDateTimeRange(
                _to_qdatetime_utc(band_start), _to_qdatetime_utc(band_end)
            )

        full_range = QgsDateTimeRange(
            _to_qdatetime_utc(start_dt), _to_qdatetime_utc(end_dt)
        )
        try:
            t_props.setMode(Qgis.RasterTemporalMode.FixedRangePerBand)
            t_props.setFixedRangePerBand(band_ranges)
            t_props.setFixedTemporalRange(full_range)
        except AttributeError:
            # QGIS < 3.38: FixedRangePerBand not available, fall back to full range
            t_props.setMode(Qgis.RasterTemporalMode.FixedTemporalRange)
            t_props.setFixedTemporalRange(full_range)

        t_props.setIsActive(True)
        return True
    except Exception as e:
        QgsMessageLog.logMessage(
            f"Failed to apply temporal properties for frames {layer.name()}: {e!s}",
            "ICEYE Toolbox",
            level=Qgis.Warning,
        )
        return False
