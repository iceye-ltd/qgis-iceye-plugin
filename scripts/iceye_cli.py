#!/usr/bin/env python3
r"""Headless ICEYE toolbox: crop, video, color, or focus on a GeoTIFF with optional KML ROI.

Run inside QGIS's Python environment (see run_cli.sh for Docker).

Examples
--------
  # Clip to a KML footprint (output _CROP_<hash>.tif; KML is required for crop)
  python scripts/iceye_cli.py crop --output-dir /tmp/out path/to/full.tif path/to/roi.kml

  # Video from an existing CROP GeoTIFF (identity extent), 4 frames, output next to input
  python scripts/iceye_cli.py video path/to/ICEYE_*_CROP_*.tif

  # Color slow-time spectrum, custom output directory
  python scripts/iceye_cli.py color --mode slow_time --output-dir /tmp/out path/to/ICEYE_*_CROP_*.tif

  # Focus on a full SLC using a KML footprint
  python scripts/iceye_cli.py focus path/to/full.tif path/to/roi.kml

Environment
-----------
  PYTHONPATH must include the parent directory of the ``iceye_toolbox`` package
  (same as pytest in Docker: ``PYTHONPATH=/plugins`` when the repo is mounted at
  ``/plugins/iceye_toolbox``).

  QT_QPA_PLATFORM=offscreen is set automatically if unset.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


def _ensure_package_on_path() -> None:
    try:
        import iceye_toolbox  # noqa: F401
    except ImportError:
        print(
            "Could not import iceye_toolbox. Set PYTHONPATH to the directory that "
            "contains the iceye_toolbox package (e.g. in Docker: PYTHONPATH=/plugins).",
            file=sys.stderr,
        )
        sys.exit(1)


def _bootstrap_qgis() -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qgis.core import QgsApplication

    qgis_prefix = os.environ.get("QGIS_PREFIX_PATH")
    if qgis_prefix:
        QgsApplication.setPrefixPath(qgis_prefix, True)
    app = QgsApplication([], False)
    app.initQgis()
    from processing.core.Processing import Processing

    Processing.initialize()
    # Keep reference on module to avoid GC before exitQgis
    global _QGIS_APP
    _QGIS_APP = app


def _shutdown_qgis() -> None:
    from qgis.core import QgsApplication

    QgsApplication.exitQgis()
    global _QGIS_APP
    _QGIS_APP = None


_QGIS_APP = None


def _load_raster(path: Path):
    from qgis.core import QgsRasterLayer

    p = path.resolve()
    if not p.is_file():
        print(
            f"Raster path is not a readable file: {p}\n"
            "If you use ./run_cli.sh inside Docker, only paths under the repo mount "
            "exist unless you also mount your data directory. Example:\n"
            "  ICEYE_CLI_DATA_ROOT=/path/to/data ./run_cli.sh crop --output-dir /tmp/out \\\n"
            "    /iceye_data/your.tif /plugins/iceye_toolbox/test/fixtures/minimal_roi.kml",
            file=sys.stderr,
        )
        sys.exit(1)

    # Explicit "gdal" provider: ICEYE SLC GeoTIFFs (GCP-based) load reliably as GDAL rasters.
    layer = QgsRasterLayer(str(p), p.stem, "gdal")
    valid = layer.isValid()
    err = ""
    if not valid:
        try:
            qe = layer.error()
            if qe is not None:
                err = qe.summary()
        except Exception:
            err = "(could not read QGIS layer error)"

    if not valid:
        print(f"Invalid raster layer: {p}", file=sys.stderr)
        if err:
            print(err, file=sys.stderr)
        sys.exit(1)
    return layer


def _extent_from_kml(raster_layer, kml_path: Path):
    from qgis.core import (
        QgsCoordinateTransform,
        QgsProject,
        QgsVectorLayer,
    )

    kml_layer = QgsVectorLayer(str(kml_path), "roi", "ogr")
    if not kml_layer.isValid():
        print(f"Invalid KML / vector layer: {kml_path}", file=sys.stderr)
        sys.exit(1)
    transform = QgsCoordinateTransform(
        kml_layer.crs(),
        raster_layer.crs(),
        QgsProject.instance(),
    )
    return transform.transformBoundingBox(kml_layer.extent())


def _identity_crop_task(raster_layer, extent):
    from iceye_toolbox.core.cropper import CropLayerTask

    crop_task = CropLayerTask(raster_layer, extent)
    crop_task.result_layer = raster_layer
    return crop_task


def _run_real_crop(raster_layer, extent):
    from iceye_toolbox.core.cropper import CropLayerTask

    crop_task = CropLayerTask(raster_layer, extent)
    ok = crop_task.run()
    if not ok:
        print(
            f"Cropping failed: {crop_task.error_msg or 'unknown error'}",
            file=sys.stderr,
        )
        sys.exit(1)
    crop_task.finished(True)
    if not crop_task.result_layer or not crop_task.result_layer.isValid():
        print("Cropping produced an invalid layer.", file=sys.stderr)
        sys.exit(1)
    return crop_task


def _prepare_crop(raster_layer, extent, use_kml: bool):
    from qgis.core import QgsProject

    QgsProject.instance().addMapLayer(raster_layer, True)
    if use_kml:
        return _run_real_crop(raster_layer, extent)
    return _identity_crop_task(raster_layer, extent)


def _resolve_output_dir(input_path: Path, output_dir: Path | None) -> Path:
    return Path(output_dir).resolve() if output_dir else input_path.parent.resolve()


def _maybe_move_output(src_uri: str, output_dir: Path) -> Path:
    """Move GDAL/QGIS output file to output_dir if it is not already there."""
    src = Path(src_uri.split("|")[0]).resolve()
    if not src.is_file():
        print(f"Expected output file missing: {src}", file=sys.stderr)
        sys.exit(1)
    dest = output_dir / src.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dest.resolve():
        return dest
    shutil.move(str(src), str(dest))
    return dest


def cmd_video(args: argparse.Namespace) -> None:
    """Run slow-time video (SHORT multiband GeoTIFF)."""
    from iceye_toolbox.core.metadata import MetadataProvider
    from iceye_toolbox.core.video import VideoProcessingTask

    input_path = Path(args.input_tif).resolve()
    output_dir = _resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raster = _load_raster(input_path)
    use_kml = args.roi_kml is not None
    extent = (
        _extent_from_kml(raster, Path(args.roi_kml).resolve())
        if use_kml
        else raster.extent()
    )

    crop_task = _prepare_crop(raster, extent, use_kml)
    metadata_layer = crop_task.result_layer
    metadata = MetadataProvider().get(metadata_layer)
    if not metadata:
        print("No ICEYE metadata found on raster.", file=sys.stderr)
        sys.exit(1)

    video_task = VideoProcessingTask(
        None,
        metadata_layer,
        args.frames,
        metadata,
        temp_files=str(output_dir),
    )
    video_task.crop_subtask = crop_task

    if not video_task.run():
        print(
            f"Video task failed: {video_task.exception or 'unknown error'}",
            file=sys.stderr,
        )
        sys.exit(1)

    out_path = Path(video_task.temp_path).resolve()
    if not out_path.is_file():
        print(f"Expected output file missing: {out_path}", file=sys.stderr)
        sys.exit(1)

    video_task.finished(True)

    print(out_path)


def cmd_color(args: argparse.Namespace) -> None:
    """Run color composite (RGB GeoTIFF)."""
    from iceye_toolbox.core.color import ColorTask
    from iceye_toolbox.core.metadata import MetadataProvider

    input_path = Path(args.input_tif).resolve()
    output_dir = _resolve_output_dir(input_path, args.output_dir)

    raster = _load_raster(input_path)
    use_kml = args.roi_kml is not None
    extent = (
        _extent_from_kml(raster, Path(args.roi_kml).resolve())
        if use_kml
        else raster.extent()
    )

    crop_task = _prepare_crop(raster, extent, use_kml)
    metadata_provider = MetadataProvider()
    if not metadata_provider.get(crop_task.result_layer):
        print("No ICEYE metadata found on raster.", file=sys.stderr)
        sys.exit(1)

    color_task = ColorTask(
        None,
        metadata_provider,
        crop_task,
        color_mode=args.mode,
    )
    if not color_task.run():
        print("Color task failed.", file=sys.stderr)
        sys.exit(1)

    result_layer = color_task.result_layer
    if not result_layer or not result_layer.isValid():
        print("Color task did not produce a valid layer.", file=sys.stderr)
        sys.exit(1)

    src_uri = result_layer.source()
    final_path = _maybe_move_output(src_uri, output_dir)

    print(final_path)


def cmd_focus(args: argparse.Namespace) -> None:
    """Run autofocus (FOCUS GeoTIFF)."""
    from iceye_toolbox.core.autofocus import AutofocusTask
    from iceye_toolbox.core.metadata import MetadataProvider

    input_path = Path(args.input_tif).resolve()
    output_dir = _resolve_output_dir(input_path, args.output_dir)

    raster = _load_raster(input_path)
    use_kml = args.roi_kml is not None
    extent = (
        _extent_from_kml(raster, Path(args.roi_kml).resolve())
        if use_kml
        else raster.extent()
    )

    crop_task = _prepare_crop(raster, extent, use_kml)
    metadata_provider = MetadataProvider()
    if not metadata_provider.get(crop_task.result_layer):
        print("No ICEYE metadata found on raster.", file=sys.stderr)
        sys.exit(1)

    focus_task = AutofocusTask(None, metadata_provider, crop_task)
    if not focus_task.run():
        print("Focus task failed.", file=sys.stderr)
        sys.exit(1)

    result_layer = focus_task.result_layer
    if not result_layer or not result_layer.isValid():
        print("Focus task did not produce a valid layer.", file=sys.stderr)
        sys.exit(1)

    src_uri = result_layer.source()
    final_path = _maybe_move_output(src_uri, output_dir)

    print(final_path)


def cmd_crop(args: argparse.Namespace) -> None:
    """Run gdal clip to KML ROI; writes ICEYE-style ``_CROP_<hash>.tif`` (always a new file)."""
    from qgis.core import QgsProject

    input_path = Path(args.input_tif).resolve()
    output_dir = _resolve_output_dir(input_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raster = _load_raster(input_path)
    extent = _extent_from_kml(raster, Path(args.roi_kml).resolve())

    QgsProject.instance().addMapLayer(raster, True)
    crop_task = _run_real_crop(raster, extent)

    result_layer = crop_task.result_layer
    src = Path(result_layer.dataProvider().dataSourceUri().split("|")[0]).resolve()
    if not src.is_file():
        print(f"Expected crop output file missing: {src}", file=sys.stderr)
        sys.exit(1)
    dest = output_dir / f"{result_layer.name()}.tif"
    output_dir.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))
    print(dest.resolve())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Headless ICEYE crop / video / color / focus on GeoTIFF (KML required for crop).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "input_tif",
            type=str,
            help="Input GeoTIFF (full scene or CROP product).",
        )
        p.add_argument(
            "roi_kml",
            nargs="?",
            default=None,
            help="Optional KML (or OGR-readable vector) defining ROI; bbox is reprojected to raster CRS.",
        )
        p.add_argument(
            "--output-dir",
            type=Path,
            default=None,
            help="Directory for outputs (default: next to input GeoTIFF).",
        )

    p_crop = sub.add_parser(
        "crop",
        help="Clip GeoTIFF to KML ROI bbox (gdal_translate → _CROP_<hash>.tif).",
    )
    p_crop.add_argument(
        "input_tif",
        type=str,
        help="Input GeoTIFF (full scene or CROP product).",
    )
    p_crop.add_argument(
        "roi_kml",
        type=str,
        help="KML or other OGR-readable vector; ROI bbox is reprojected to raster CRS.",
    )
    p_crop.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs (default: next to input GeoTIFF).",
    )
    p_crop.set_defaults(func=cmd_crop)

    p_video = sub.add_parser(
        "video", help="Multi-frame SHORT (slow-time) video GeoTIFF."
    )
    add_common(p_video)
    p_video.add_argument(
        "--frames",
        type=int,
        default=4,
        help="Number of frames / bands (default: 4).",
    )
    p_video.set_defaults(func=cmd_video)

    p_color = sub.add_parser("color", help="RGB color composite GeoTIFF.")
    add_common(p_color)
    p_color.add_argument(
        "--mode",
        choices=("fast_time", "slow_time", "range_cmap"),
        default="range_cmap",
        help="Color mode: fast_time, slow_time, or range_cmap (default).",
    )
    p_color.set_defaults(func=cmd_color)

    p_focus = sub.add_parser("focus", help="Autofocus (PGA) GeoTIFF output.")
    add_common(p_focus)
    p_focus.set_defaults(func=cmd_focus)

    return parser


def main() -> None:
    """CLI entry: init QGIS, dispatch subcommand, tear down."""
    _ensure_package_on_path()
    parser = _build_parser()
    args = parser.parse_args()
    _bootstrap_qgis()
    try:
        from qgis.core import QgsProject

        QgsProject.instance().clear()
        args.func(args)
    finally:
        from qgis.core import QgsProject

        QgsProject.instance().clear()
        _shutdown_qgis()


if __name__ == "__main__":
    main()
