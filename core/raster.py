"""SLC raster I/O and band reading utilities."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray
from osgeo import gdal
from qgis.core import Qgis, QgsMessageLog, QgsRasterLayer, QgsRectangle

from iceye_toolbox.core.cropper import get_extend_image_coords
from iceye_toolbox.core.metadata import IceyeMetadata


def _get_band_name(band: gdal.Band) -> str | None:
    """Return the canonical name of a GDAL raster band, or None if unset.

    The original ICEYE SLC GeoTIFFs expose the band identity through the
    band-level ``NAME`` metadata key. Some downstream operations (notably
    ``gdal:translate``) drop this key; in that case we fall back to the
    band description, which most GDAL drivers do round-trip.
    """
    name = band.GetMetadata().get("NAME")
    if name:
        return name
    description = band.GetDescription()
    if description:
        return description
    return None


def read_slc_layer(
    layer: QgsRasterLayer | str,
    left: bool,
    metadata: IceyeMetadata,
    extent: QgsRectangle | None = None,
) -> tuple[NDArray[np.complex64], str]:
    """Read SLC (amplitude + phase) data from a raster layer.

    Parameters
    ----------
    layer : QgsRasterLayer or str
        Raster layer or path to SLC GeoTIFF.
    left : bool
        True if left-looking SAR (used for shadow orientation).
    metadata : IceyeMetadata
        ICEYE metadata for amplitude mapping.
    extent : QgsRectangle or None, optional
        Extent to read. If None, reads full layer.

    Returns
    -------
    tuple of (ndarray of complex64, str)
        Complex SLC data and source path.

    Raises
    ------
    ValueError
        If layer is not SLC format (band names invalid).
    """
    if isinstance(layer, QgsRasterLayer):
        source_path = layer.dataProvider().dataSourceUri()
    else:
        source_path = layer

    ds = gdal.Open(source_path)
    if ds.RasterCount < 2:
        raise ValueError(f"Layer {layer.name()} probably not SLC format")

    if ds:
        amplitude_band = ds.GetRasterBand(1)
        phase_band = ds.GetRasterBand(2)
        amp_band_scale = amplitude_band.GetScale() or 1.0
        amp_band_offset = amplitude_band.GetOffset() or 0.0
        phase_band_scale = phase_band.GetScale() or 1.0
        phase_band_offset = phase_band.GetOffset() or 0.0

        amplitude_band_name = _get_band_name(amplitude_band)
        phase_band_name = _get_band_name(phase_band)

        # Band names can be lost by intermediate GDAL operations (e.g.
        # gdal:translate during cropping). When a name is present we still
        # validate it strictly; when missing we fall back to the standard
        # ICEYE SLC layout (band 1 = amplitude, band 2 = phase).
        if amplitude_band_name is not None and amplitude_band_name.lower() != "amplitude":
            raise ValueError(
                f"Expected band 1 to be named 'amplitude', but got '{amplitude_band_name}'"
            )

        if phase_band_name is not None and phase_band_name.lower() != "phase":
            raise ValueError(
                f"Expected band 2 to be named 'phase', but got '{phase_band_name}'"
            )

        if amplitude_band_name is None or phase_band_name is None:
            QgsMessageLog.logMessage(
                f"SLC band name metadata missing for {source_path}; "
                "assuming band 1 = amplitude, band 2 = phase",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )

        mapping = metadata.iceye_amplitude_mapping
        function = mapping.get("function") if mapping else None

        # Read amplitude data with extent if provided
        if extent:
            bounds = get_extend_image_coords(layer, extent)
            amplitude_data = amplitude_band.ReadAsArray(
                int(bounds.xMinimum()),
                int(bounds.yMinimum()),
                int(bounds.width()),
                int(bounds.height()),
            )
            phase_data = phase_band.ReadAsArray(
                int(bounds.xMinimum()),
                int(bounds.yMinimum()),
                int(bounds.width()),
                int(bounds.height()),
            )
        else:
            amplitude_data = amplitude_band.ReadAsArray()
            phase_data = phase_band.ReadAsArray()

        if function == "power":
            scale = mapping.get("parameters", {}).get("scale")
            exponent = mapping.get("parameters", {}).get("exponent")

            QgsMessageLog.logMessage(
                f"Found amplitude mapping: function: {function}, scale: {scale}, exponent: {exponent}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Info,
            )

            data = (
                amplitude_data.astype(np.complex64) ** (scale / exponent)
                * amp_band_scale
                + amp_band_offset
            )
        else:
            if mapping:
                QgsMessageLog.logMessage(
                    f"Found amplitude mapping: function: {function}",
                    "ICEYE Toolbox",
                    Qgis.MessageLevel.Info,
                )

            data = (
                amplitude_data.astype(np.complex64) * amp_band_scale + amp_band_offset
            )

        data *= np.exp(
            -1j * phase_data.astype(np.float32) * phase_band_scale + phase_band_offset
        )

        return toggle_shadows_down(data, left), source_path
    else:
        raise Exception(f"Failed to open dataset {source_path}")


def toggle_shadows_down(
    data: NDArray[np.complex64], left: bool
) -> NDArray[np.complex64]:
    """Orient data so shadows point downward.

    Parameters
    ----------
    data : ndarray of complex64
        SLC data (range x azimuth).
    left : bool
        True if left-looking SAR.

    Returns
    -------
    ndarray of complex64
        Reoriented data.
    """
    if left:
        data = np.fliplr(data)
    return data.T


def read_all_band_from_layer(
    layer: QgsRasterLayer | str, extent: QgsRectangle | None = None
) -> Iterator[NDArray[np.float32]]:
    """Yield each band from a raster layer.

    Parameters
    ----------
    layer : QgsRasterLayer or str
        Raster layer or path.
    extent : QgsRectangle or None, optional
        Extent to read. If None, reads full layer.

    Yields
    ------
    ndarray of float32
        Each band's data.
    """
    if isinstance(layer, QgsRasterLayer):
        source_path = layer.dataProvider().dataSourceUri()
    else:
        source_path = layer

    with gdal.Open(source_path) as dataset:
        for band_num in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(band_num)
            if extent:
                bounds = get_extend_image_coords(layer, extent)
                yield band.ReadAsArray(
                    bounds.xMinimum, bounds.yMinimum, bounds.width(), bounds.height()
                )
            yield band.ReadAsArray()
