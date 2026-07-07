"""Interactive ground-truth labeller for moving-target detections.

Given an ICEYE SLC ``.tif`` (with its sidecar ``.json``) or a ``.npy``
SLC patch, this tool

  1. Loads the SLC through the exact same code path that
     :mod:`scripts.shear_averaging` uses (``_load_slc_from_tiff``), so
     the resulting array has shape ``(N_az, N_range)`` in the same
     "shadows-down" orientation the detector consumes. This is the
     coordinate frame in which ground-truth boxes are stored.

  2. Displays a log-stretched, decimated ``|s|`` view (``s[::da, ::dr]``
     with ``da, dr`` from ``--decimate-az`` / ``--decimate-rg``,
     defaulting to 10 and 2 respectively).

  3. Lets you drag rectangles with the mouse to mark moving targets.
     Each drag adds one ground-truth box, whose corners are converted
     back to the full-resolution ``(azimuth, range)`` grid.

  4. Writes / updates a sidecar ``<tiff_stem>.gt.json`` next to the
     TIFF, containing image metadata (path, sha256, shape, left-look
     flag, pixel spacings) plus the list of GT boxes.

Coordinate convention
---------------------

Every box in the sidecar is stored as ``{y_c, x_c, h, w}`` in the
**full-resolution** ``(azimuth, range)`` frame that
:func:`scripts.shear_averaging._load_slc_from_tiff` produces. In
particular ``x_c`` is in ORIGINAL range pixels, NOT in the ``x_c`` of
``shear_averaging``'s ``_box_stats.csv`` (which is decimated by
``Number_of_Range_Looks``). The evaluator converts detections to
full-res before matching, so labels and detections share the same
frame.

Key bindings inside the plot window
-----------------------------------

* Drag left mouse            add a rectangle as a GT box
* ``z``                       undo the last box
* ``x``                       delete the box under the mouse cursor
* ``s``                       save to disk
* ``q``                       save and quit
* ``Q``                       quit WITHOUT saving
* ``r``                       reload GT sidecar from disk
* ``l``                       toggle box-id text labels on/off
* ``1``..``9``                set the class label for the NEXT drag
* ``c``                       clear the pending class label

Usage
-----

    python scripts/eval/label_ground_truth.py path/to/scene.tif
    python scripts/eval/label_ground_truth.py scene.tif --decimate-az 8 --decimate-rg 2
    python scripts/eval/label_ground_truth.py scene.tif --gt-json custom_gt.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# Local import of shear_averaging so we share the exact SLC-loading code
# path. scripts/eval/ sits alongside scripts/, so bump sys.path up one.
_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from shear_averaging import (  # noqa: E402
    _load_iceye_sidecar_metadata,
    _load_slc_from_tiff,
)

GT_SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Sidecar JSON I/O
# ---------------------------------------------------------------------------


def _default_gt_path(image_path: Path) -> Path:
    """`<stem>.gt.json` next to the image, whatever its suffix."""
    return image_path.with_suffix(".gt.json")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256; needed because ICEYE scenes are several GB."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_gt_json(path: Path) -> Optional[dict]:
    """Return the sidecar dict, or None if the file does not exist."""
    if not path.exists():
        return None
    with path.open() as f:
        doc = json.load(f)
    schema = int(doc.get("schema", 0))
    if schema != GT_SCHEMA_VERSION:
        raise ValueError(
            f"{path.name}: unsupported gt schema {schema} "
            f"(expected {GT_SCHEMA_VERSION})"
        )
    return doc


def save_gt_json(
    gt_path: Path,
    image_path: Path,
    sha256: str,
    shape_az_rg: tuple[int, int],
    left_look: bool,
    range_pixel_spacing_m: Optional[float],
    azimuth_pixel_spacing_m: Optional[float],
    boxes: list[dict],
    created_at: Optional[str] = None,
) -> None:
    """Write the sidecar atomically (via a ``.tmp`` sibling)."""
    doc = {
        "schema": GT_SCHEMA_VERSION,
        "image_path": str(image_path.resolve()),
        "image_stem": image_path.stem,
        "sha256": sha256,
        "shape_az_rg": [int(shape_az_rg[0]), int(shape_az_rg[1])],
        "left_look": bool(left_look),
        "range_pixel_spacing_m": (
            None if range_pixel_spacing_m is None
            else float(range_pixel_spacing_m)
        ),
        "azimuth_pixel_spacing_m": (
            None if azimuth_pixel_spacing_m is None
            else float(azimuth_pixel_spacing_m)
        ),
        "created_at": created_at or _now_iso(),
        "updated_at": _now_iso(),
        "coord_frame": (
            "shear_averaging (azimuth, range) full-resolution pixels, "
            "matches _load_slc_from_tiff output; x_c is at FULL range "
            "resolution (multiply by Number_of_Range_Looks to compare "
            "against shear_averaging _box_stats.csv)."
        ),
        "boxes": boxes,
    }
    tmp = gt_path.with_suffix(gt_path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(doc, f, indent=2)
    tmp.replace(gt_path)


# ---------------------------------------------------------------------------
# SLC loading (mirrors shear_averaging.main but without any processing)
# ---------------------------------------------------------------------------


def load_slc_any(
    path: Path,
) -> tuple[np.ndarray, bool, Optional[float], Optional[float]]:
    """Load a `.tif` (ICEYE SLC) or `.npy` patch as a complex 2-D array.

    Returns ``(s, left_look, range_pixel_spacing_m, azimuth_pixel_spacing_m)``.
    Pixel spacings are ``None`` for a bare `.npy` patch.
    """
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        sidecar = path.with_suffix(".json")
        if not sidecar.exists():
            raise FileNotFoundError(
                f"GeoTIFF {path.name} requires sidecar metadata JSON at "
                f"{sidecar}; not found."
            )
        md = _load_iceye_sidecar_metadata(sidecar)
        left = md["sar_observation_direction"].lower() == "left"
        s = _load_slc_from_tiff(path, left=left)
        return (
            s,
            left,
            float(md["sar_pixel_spacing_range"]),
            float(md["sar_pixel_spacing_azimuth"]),
        )
    if suffix == ".npy":
        s = np.load(path)
        if s.ndim != 2:
            raise ValueError(
                f"{path.name} loaded with shape {s.shape}; expected 2-D."
            )
        return s, False, None, None
    raise ValueError(
        f"Unsupported input extension {suffix!r} for {path}; "
        "expected .npy, .tif or .tiff."
    )


def build_display_image(
    s: np.ndarray,
    decimate_az: int,
    decimate_rg: int,
    dr_db: float = 40.0,
) -> np.ndarray:
    """Log-stretched, decimated |s| for display.

    Values below the (100 - dr_db*ish) percentile floor are clipped so
    the huge dynamic range of raw SAR amplitudes doesn't crush the
    weaker targets. Returns a real ``float32`` array in dB.
    """
    view = np.abs(s[::decimate_az, ::decimate_rg])
    view = view.astype(np.float32, copy=False)
    finite = view[np.isfinite(view) & (view > 0)]
    if finite.size == 0:
        return np.zeros_like(view)
    floor = np.percentile(finite, 5.0)
    ceil = np.percentile(finite, 99.5)
    if floor <= 0.0:
        floor = max(finite.min(), 1e-6)
    view_db = 20.0 * np.log10(np.maximum(view, floor))
    hi_db = 20.0 * np.log10(ceil)
    lo_db = hi_db - dr_db
    return np.clip(view_db, lo_db, hi_db)


# ---------------------------------------------------------------------------
# Interactive labeller
# ---------------------------------------------------------------------------


class _Labeller:
    """Stateful matplotlib UI wrapper.

    Everything the mouse / keyboard does is a mutation of ``self.boxes``
    (which is a list of full-resolution ``dict`` records). The display
    grid is decimated by ``(da, dr)``, so on every draw we render each
    box at ``x_display = x_full / dr`` (and similarly for y).
    """

    def __init__(
        self,
        s: np.ndarray,
        image_path: Path,
        gt_path: Path,
        decimate_az: int,
        decimate_rg: int,
        left_look: bool,
        range_pixel_spacing_m: Optional[float],
        azimuth_pixel_spacing_m: Optional[float],
        existing: Optional[dict],
    ) -> None:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RectangleSelector

        self.plt = plt
        self.Rectangle = plt.matplotlib.patches.Rectangle
        self.RectangleSelector = RectangleSelector

        self.image_path = image_path
        self.gt_path = gt_path
        self.decimate_az = int(decimate_az)
        self.decimate_rg = int(decimate_rg)
        self.left_look = bool(left_look)
        self.range_pixel_spacing_m = range_pixel_spacing_m
        self.azimuth_pixel_spacing_m = azimuth_pixel_spacing_m
        self.shape_az_rg = (int(s.shape[0]), int(s.shape[1]))

        print("  Computing SHA-256 of image (streaming)...")
        t0 = time.perf_counter()
        self.sha256 = _sha256_file(image_path) if image_path.is_file() else ""
        print(
            f"    sha256 = {self.sha256[:16]}\u2026  "
            f"({time.perf_counter() - t0:.1f} s)"
        )

        if existing is not None:
            if existing["sha256"] and existing["sha256"] != self.sha256:
                print(
                    "  (warning) sidecar sha256 does not match the current "
                    "image content. Loading boxes anyway, but the labels "
                    "may not correspond to what you're now seeing."
                )
            self.created_at = existing.get("created_at", _now_iso())
            self.boxes = list(existing.get("boxes", []))
            print(f"  Loaded {len(self.boxes)} existing GT boxes from "
                  f"{gt_path.name}")
        else:
            self.created_at = _now_iso()
            self.boxes = []
            print(f"  Starting fresh; sidecar {gt_path.name} will be created.")

        self._next_id = 1 + max((int(b["id"]) for b in self.boxes), default=-1)

        # Build the display view.
        print(
            f"  Building display view (decimate az={self.decimate_az}, "
            f"rg={self.decimate_rg})..."
        )
        t0 = time.perf_counter()
        self.view = build_display_image(
            s, self.decimate_az, self.decimate_rg
        )
        print(
            f"    display shape={self.view.shape}  "
            f"range=[{self.view.min():.1f}, {self.view.max():.1f}] dB  "
            f"({time.perf_counter() - t0:.1f} s)"
        )

        self.pending_class: Optional[str] = None
        self.labels_visible = True
        self._box_patches: list = []  # matplotlib artists, mirrors self.boxes
        self._label_artists: list = []

        # --- Figure ---
        # Landscape-ish figure. `imshow` renders row 0 (azimuth = 0) at
        # top which matches the shear_averaging convention.
        self.fig, self.ax = plt.subplots(figsize=(14, 9))
        self.ax.imshow(
            self.view, cmap="gray", aspect="auto", interpolation="nearest",
        )
        self.ax.set_xlabel(
            f"range pixel (display; \u00d7 {self.decimate_rg} to get full-res)"
        )
        self.ax.set_ylabel(
            f"azimuth pixel (display; \u00d7 {self.decimate_az} to get full-res)"
        )
        self._refresh_title()
        self._redraw_boxes()

        # RectangleSelector for drag-to-label. `useblit=True` is much
        # smoother on large images; the rest is styling.
        self.selector = self.RectangleSelector(
            self.ax,
            self._on_rectangle,
            useblit=True,
            button=[1],  # left mouse only
            minspanx=2,
            minspany=2,
            spancoords="pixels",
            interactive=False,
            props=dict(facecolor="none", edgecolor="cyan", linewidth=1.2),
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ----- Persistence helpers -----

    def save(self) -> None:
        save_gt_json(
            self.gt_path,
            self.image_path,
            self.sha256,
            self.shape_az_rg,
            self.left_look,
            self.range_pixel_spacing_m,
            self.azimuth_pixel_spacing_m,
            self.boxes,
            created_at=self.created_at,
        )
        print(
            f"  Saved {len(self.boxes)} boxes \u2192 {self.gt_path} "
            f"({self.gt_path.stat().st_size} bytes)"
        )

    def reload(self) -> None:
        existing = load_gt_json(self.gt_path)
        if existing is None:
            print(f"  (reload) {self.gt_path.name} does not exist; ignored.")
            return
        self.boxes = list(existing.get("boxes", []))
        self.created_at = existing.get("created_at", self.created_at)
        self._next_id = 1 + max(
            (int(b["id"]) for b in self.boxes), default=-1
        )
        self._redraw_boxes()
        self._refresh_title()
        self.fig.canvas.draw_idle()
        print(f"  Reloaded {len(self.boxes)} boxes from {self.gt_path.name}")

    # ----- Rendering helpers -----

    def _refresh_title(self) -> None:
        klass = (
            f"class \u2192 {self.pending_class}"
            if self.pending_class is not None
            else "class \u2192 (none)"
        )
        self.ax.set_title(
            f"{self.image_path.name}   |   {len(self.boxes)} boxes   |   "
            f"{klass}\n"
            "drag=add   z=undo   x=delete   s=save   q=save+quit   "
            "Q=quit no-save   r=reload   l=toggle labels   1..9=class   c=clear"
        )

    def _redraw_boxes(self) -> None:
        for p in self._box_patches:
            p.remove()
        for t in self._label_artists:
            t.remove()
        self._box_patches = []
        self._label_artists = []

        for b in self.boxes:
            y_c = float(b["y_c"]) / self.decimate_az
            x_c = float(b["x_c"]) / self.decimate_rg
            hh = float(b["h"]) / self.decimate_az
            ww = float(b["w"]) / self.decimate_rg
            rect = self.Rectangle(
                (x_c - ww / 2.0, y_c - hh / 2.0),
                ww, hh,
                linewidth=1.4,
                edgecolor="lime",
                facecolor="none",
                zorder=5,
            )
            self.ax.add_patch(rect)
            self._box_patches.append(rect)
            if self.labels_visible:
                label = f"#{b['id']}"
                klass = b.get("class")
                if klass:
                    label = f"{label}:{klass}"
                t = self.ax.text(
                    x_c - ww / 2.0, y_c - hh / 2.0 - 3,
                    label,
                    color="lime", fontsize=8, weight="bold",
                    ha="left", va="bottom", zorder=6,
                )
                self._label_artists.append(t)

    # ----- Event handlers -----

    def _on_rectangle(self, eclick, erelease) -> None:
        """RectangleSelector callback. Snap corners to display-pixel
        integers, convert to full-resolution ``(y_c, x_c, h, w)`` and
        append to self.boxes.
        """
        if eclick.xdata is None or erelease.xdata is None:
            return
        x0_d = min(eclick.xdata, erelease.xdata)
        x1_d = max(eclick.xdata, erelease.xdata)
        y0_d = min(eclick.ydata, erelease.ydata)
        y1_d = max(eclick.ydata, erelease.ydata)

        # Full-resolution corners.
        x0 = int(round(x0_d * self.decimate_rg))
        x1 = int(round(x1_d * self.decimate_rg))
        y0 = int(round(y0_d * self.decimate_az))
        y1 = int(round(y1_d * self.decimate_az))
        x0 = max(0, min(x0, self.shape_az_rg[1] - 1))
        x1 = max(0, min(x1, self.shape_az_rg[1] - 1))
        y0 = max(0, min(y0, self.shape_az_rg[0] - 1))
        y1 = max(0, min(y1, self.shape_az_rg[0] - 1))
        if x1 <= x0 or y1 <= y0:
            return
        y_c = (y0 + y1) / 2.0
        x_c = (x0 + x1) / 2.0
        h = y1 - y0
        w = x1 - x0

        box = {
            "id": self._next_id,
            "y_c": float(y_c),
            "x_c": float(x_c),
            "h": int(h),
            "w": int(w),
            "class": self.pending_class,
            "notes": "",
            "created_at": _now_iso(),
        }
        self.boxes.append(box)
        self._next_id += 1

        print(
            f"  + box #{box['id']}: y_c={box['y_c']:.0f} x_c={box['x_c']:.0f} "
            f"h={box['h']} w={box['w']} class={box['class']!r}"
        )
        self._redraw_boxes()
        self._refresh_title()
        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        key = event.key
        if key == "z":
            if not self.boxes:
                print("  (undo) no boxes to remove.")
                return
            gone = self.boxes.pop()
            print(f"  - undo box #{gone['id']}")
            self._redraw_boxes()
            self._refresh_title()
            self.fig.canvas.draw_idle()
        elif key == "x":
            self._delete_under_cursor(event)
        elif key == "s":
            self.save()
        elif key == "q":
            self.save()
            self.plt.close(self.fig)
        elif key == "Q":
            print("  Quitting WITHOUT saving.")
            self.plt.close(self.fig)
        elif key == "r":
            self.reload()
        elif key == "l":
            self.labels_visible = not self.labels_visible
            self._redraw_boxes()
            self.fig.canvas.draw_idle()
        elif key == "c":
            self.pending_class = None
            self._refresh_title()
            self.fig.canvas.draw_idle()
            print("  class label cleared.")
        elif key in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
            self.pending_class = key
            self._refresh_title()
            self.fig.canvas.draw_idle()
            print(f"  next box class \u2192 {key}")

    def _delete_under_cursor(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            print("  (delete) cursor outside plot; ignored.")
            return
        x_full = event.xdata * self.decimate_rg
        y_full = event.ydata * self.decimate_az
        for i in reversed(range(len(self.boxes))):
            b = self.boxes[i]
            if (
                abs(x_full - b["x_c"]) <= b["w"] / 2.0
                and abs(y_full - b["y_c"]) <= b["h"] / 2.0
            ):
                gone = self.boxes.pop(i)
                print(f"  - delete box #{gone['id']}")
                self._redraw_boxes()
                self._refresh_title()
                self.fig.canvas.draw_idle()
                return
        print("  (delete) no box under cursor.")

    def run(self) -> None:
        self.plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "image", type=Path,
        help="Path to ICEYE SLC `.tif` (with sidecar `.json`) or `.npy` patch.",
    )
    p.add_argument(
        "--gt-json", type=Path, default=None, dest="gt_json",
        help="Override sidecar GT JSON path. Default: `<image_stem>.gt.json` "
             "next to the image.",
    )
    p.add_argument(
        "--decimate-az", type=int, default=10, dest="decimate_az",
        help="Azimuth decimation factor for the on-screen preview "
             "(display samples every N azimuth pixels). Default: 10.",
    )
    p.add_argument(
        "--decimate-rg", type=int, default=2, dest="decimate_rg",
        help="Range decimation factor for the on-screen preview. "
             "Default: 2.",
    )
    p.add_argument(
        "--backend", type=str, default=None,
        help="Force a matplotlib backend (e.g. `TkAgg`, `Qt5Agg`). "
             "Default: whatever matplotlib picks.",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    image_path = args.image.resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    if args.backend:
        import matplotlib
        matplotlib.use(args.backend)
    # Import late so the (possibly forced) backend applies.
    import matplotlib.pyplot as plt  # noqa: F401

    gt_path = args.gt_json if args.gt_json else _default_gt_path(image_path)

    print(f"Image        : {image_path}")
    print(f"GT sidecar   : {gt_path}")
    print(f"Decimation   : az={args.decimate_az}  rg={args.decimate_rg}")

    s, left_look, rg_sp, az_sp = load_slc_any(image_path)
    print(
        f"Loaded SLC   : shape={s.shape} dtype={s.dtype} "
        f"left_look={left_look} range_spacing={rg_sp} az_spacing={az_sp}"
    )

    existing = load_gt_json(gt_path)
    ui = _Labeller(
        s=s,
        image_path=image_path,
        gt_path=gt_path,
        decimate_az=args.decimate_az,
        decimate_rg=args.decimate_rg,
        left_look=left_look,
        range_pixel_spacing_m=rg_sp,
        azimuth_pixel_spacing_m=az_sp,
        existing=existing,
    )
    ui.run()


if __name__ == "__main__":
    main()
