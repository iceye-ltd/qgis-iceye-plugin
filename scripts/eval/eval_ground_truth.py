"""Run :mod:`scripts.shear_averaging` on an image and score its
detections against a per-image ground-truth database created by
:mod:`scripts.eval.label_ground_truth`.

Pipeline per image
------------------

1. Locate the GT sidecar (``<image_stem>.gt.json`` by default).

2. Invoke ``python scripts/shear_averaging.py <image>`` as a
   subprocess. Extra CLI flags that :mod:`shear_averaging` understands
   can be forwarded via ``--`` (e.g. ``-- --slope-rad-thresh 1.5
   --debug``). We set ``--save`` ourselves to a deterministic path
   under ``--out-dir`` so we can locate the resulting
   ``<stem>_box_stats.csv`` afterwards.

3. Parse the CSV. Detection ``(y_c, x_c, h, w)`` are in
   ``s_degraded`` pixels (range decimated by ``Number_of_Range_Looks``).
   We recompute ``Number_of_Range_Looks = int(1.5 /
   sar_pixel_spacing_range)`` the same way :mod:`shear_averaging` does
   (``min_size_of_target = 1.5 m`` is hard-coded there) and multiply
   the range fields to get back into the same full-resolution
   ``(azimuth, range)`` frame the GT boxes live in.

4. Match detections to GT with the **centre-in-GT** rule.

   * A detection whose centre falls inside the ground-truth rectangle
     of exactly one GT box is a true positive (``TP``).
   * If the detection centre falls inside several GT boxes, it is
     assigned to the one whose CENTRE is closest.
   * If two or more detections all match the same GT box, only the
     nearest-centre one is counted as ``TP``; the rest are logged as
     ``TP_dup`` (they do NOT contribute to precision).
   * A detection whose centre falls inside no GT box is a false alarm
     (``FA``).
   * A GT box with no matching detection is a false negative (``FN``).

5. Write, next to the shear_averaging output dir:

   * ``<image_stem>_eval.csv``   : one row per DETECTION with the
     matching GT id + status.
   * ``<image_stem>_eval_gt.csv``: one row per GT box with
     matched detection idxs.
   * ``<image_stem>_eval_overlay.png`` : decimated |s| view with GT
     (green), TP (yellow), FA (red), FN (blue dashed).

6. Append a one-line summary
   ``image, n_gt, n_det, TP, TP_dup, FA, FN, precision, recall, F1,
   runtime_s`` to ``<out-dir>/eval_summary.csv``.

Usage
-----

    python scripts/eval/eval_ground_truth.py path/to/scene.tif
    python scripts/eval/eval_ground_truth.py path/to/scene_folder/
    python scripts/eval/eval_ground_truth.py scene.tif -- --slope-rad-thresh 1.5

Extra flags after ``--`` are forwarded verbatim to
:mod:`shear_averaging`. Do NOT forward ``--save`` or ``--no-save`` —
this script controls those.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from label_ground_truth import (  # noqa: E402
    build_display_image,
    load_slc_any,
    load_gt_json,
    _default_gt_path,
)

# Must match shear_averaging.main() hard-coded value.
_MIN_SIZE_OF_TARGET_M = 1.5


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def compute_range_looks(range_pixel_spacing_m: Optional[float]) -> int:
    """Reproduce :mod:`shear_averaging`\u2019s
    ``Number_of_Range_Looks = int(min_size_of_target / range_spacing)``.

    Returns ``1`` if the spacing is unknown (i.e. bare ``.npy`` input),
    which effectively disables the range-axis rescaling.
    """
    if range_pixel_spacing_m is None or range_pixel_spacing_m <= 0:
        return 1
    return max(1, int(_MIN_SIZE_OF_TARGET_M / float(range_pixel_spacing_m)))


def detection_to_full_res(
    row: dict[str, str], n_range_looks: int,
) -> dict[str, float]:
    """Return ``(y_c, x_c, h, w)`` at full range resolution + carry-over."""
    y_c = float(row["y_c"])
    x_c = float(row["x_c"]) * n_range_looks
    h = float(row["h"])
    w = float(row["w"]) * n_range_looks
    return {
        "idx": int(row["idx"]),
        "y_c": y_c,
        "x_c": x_c,
        "h": h,
        "w": w,
        "y_c_deg": float(row["y_c"]),
        "x_c_deg": float(row["x_c"]),
        "h_deg": float(row["h"]),
        "w_deg": float(row["w"]),
    }


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def read_box_stats_csv(csv_path: Path) -> list[dict[str, str]]:
    """Load shear_averaging\u2019s ``_box_stats.csv`` into a list of dicts."""
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


# ---------------------------------------------------------------------------
# Matching (centre-in-GT rule)
# ---------------------------------------------------------------------------


def match_center_in_gt(
    detections: list[dict], gt_boxes: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Run the centre-in-GT matching described in the module docstring.

    Parameters
    ----------
    detections : list of dicts, each with ``y_c, x_c, h, w`` at
        FULL-resolution (the return of :func:`detection_to_full_res`).
    gt_boxes   : list of dicts, each with ``id, y_c, x_c, h, w`` in the
        same frame.

    Returns
    -------
    per_det : list of dicts (one per detection) with keys
        ``det_idx, matched_gt_id, status`` where ``status`` is one of
        ``{"TP", "TP_dup", "FA"}``.
    per_gt  : list of dicts (one per GT box) with keys
        ``gt_id, matched_det_idxs, primary_det_idx`` where
        ``primary_det_idx`` is the nearest-centre detection assigned
        as the TP (or ``None`` if the GT has no matcher \u2192 FN).
    """
    gt_by_id = {int(g["id"]): g for g in gt_boxes}

    # Step 1: for each detection, list every GT whose rectangle contains
    #         the detection centre; pick the nearest-centre GT.
    det_to_gt: list[Optional[int]] = []
    for d in detections:
        best_gid: Optional[int] = None
        best_dist2 = float("inf")
        for gid, g in gt_by_id.items():
            if (
                abs(d["y_c"] - g["y_c"]) <= g["h"] / 2.0
                and abs(d["x_c"] - g["x_c"]) <= g["w"] / 2.0
            ):
                dy = d["y_c"] - g["y_c"]
                dx = d["x_c"] - g["x_c"]
                dist2 = dy * dy + dx * dx
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_gid = gid
        det_to_gt.append(best_gid)

    # Step 2: for each GT that received matches, elect the closest
    #         detection as the primary TP; every other match becomes
    #         a TP_dup.
    gt_to_dets: dict[int, list[int]] = {int(g["id"]): [] for g in gt_boxes}
    for di, gid in enumerate(det_to_gt):
        if gid is not None:
            gt_to_dets[gid].append(di)

    primary_by_gt: dict[int, int] = {}
    for gid, di_list in gt_to_dets.items():
        if not di_list:
            continue
        g = gt_by_id[gid]
        best_di = di_list[0]
        best_dist2 = float("inf")
        for di in di_list:
            d = detections[di]
            dy = d["y_c"] - g["y_c"]
            dx = d["x_c"] - g["x_c"]
            dist2 = dy * dy + dx * dx
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_di = di
        primary_by_gt[gid] = best_di

    per_det: list[dict] = []
    for di, gid in enumerate(det_to_gt):
        if gid is None:
            status = "FA"
        else:
            status = "TP" if primary_by_gt[gid] == di else "TP_dup"
        per_det.append(
            {"det_idx": di, "matched_gt_id": gid, "status": status}
        )

    per_gt: list[dict] = []
    for gid, di_list in gt_to_dets.items():
        per_gt.append(
            {
                "gt_id": gid,
                "matched_det_idxs": di_list,
                "primary_det_idx": primary_by_gt.get(gid),
            }
        )

    return per_det, per_gt


def summarise(per_det: list[dict], per_gt: list[dict]) -> dict[str, float]:
    n_det = len(per_det)
    n_gt = len(per_gt)
    tp = sum(1 for r in per_det if r["status"] == "TP")
    tp_dup = sum(1 for r in per_det if r["status"] == "TP_dup")
    fa = sum(1 for r in per_det if r["status"] == "FA")
    fn = sum(1 for r in per_gt if r["primary_det_idx"] is None)
    precision = tp / (tp + fa) if (tp + fa) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if (
        precision == precision and recall == recall
        and (precision + recall) > 0
    ):
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return {
        "n_det": n_det,
        "n_gt": n_gt,
        "TP": tp,
        "TP_dup": tp_dup,
        "FA": fa,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
    }


# ---------------------------------------------------------------------------
# Overlay figure
# ---------------------------------------------------------------------------


def render_overlay(
    s,
    out_path: Path,
    gt_boxes: list[dict],
    detections: list[dict],
    per_det: list[dict],
    per_gt: list[dict],
    image_name: str,
    decimate_az: int,
    decimate_rg: int,
    summary: dict[str, float],
) -> None:
    """Save `<stem>_eval_overlay.png` with GT + TP + FA + FN.

    Colour code: GT = lime (solid), TP = gold (solid), FA = red (solid),
    FN = royalblue (dashed). All boxes are drawn in the decimated
    display frame; only the ``matplotlib`` state is created here so we
    keep this function testable without a display.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    view = build_display_image(s, decimate_az, decimate_rg)
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(view, cmap="gray", aspect="auto", interpolation="nearest")

    def _add(x_c, y_c, w, h, edge, style="-", lw=1.4, alpha=1.0):
        rect = Rectangle(
            ((x_c - w / 2) / decimate_rg, (y_c - h / 2) / decimate_az),
            w / decimate_rg, h / decimate_az,
            linewidth=lw, edgecolor=edge, facecolor="none",
            linestyle=style, alpha=alpha, zorder=5,
        )
        ax.add_patch(rect)

    for g in gt_boxes:
        _add(g["x_c"], g["y_c"], g["w"], g["h"], "lime", "-", 1.3, 0.9)

    tp_di = {r["det_idx"] for r in per_det if r["status"] == "TP"}
    fa_di = {r["det_idx"] for r in per_det if r["status"] == "FA"}
    for i, d in enumerate(detections):
        if i in tp_di:
            _add(d["x_c"], d["y_c"], d["w"], d["h"], "gold", "-", 1.6, 1.0)
        elif i in fa_di:
            _add(d["x_c"], d["y_c"], d["w"], d["h"], "red", "-", 1.4, 1.0)

    for r in per_gt:
        if r["primary_det_idx"] is None:
            g = next(g for g in gt_boxes if int(g["id"]) == r["gt_id"])
            _add(g["x_c"], g["y_c"], g["w"], g["h"], "royalblue", "--", 1.6, 1.0)

    ax.set_title(
        f"{image_name}   |   "
        f"TP={summary['TP']}  TP_dup={summary['TP_dup']}  "
        f"FA={summary['FA']}  FN={summary['FN']}  "
        f"P={summary['precision']:.2f}  R={summary['recall']:.2f}  "
        f"F1={summary['F1']:.2f}\n"
        "green=GT   gold=TP   red=FA   blue-dashed=FN"
    )
    ax.set_xlabel(
        f"range pixel (display; \u00d7 {decimate_rg} to get full-res)"
    )
    ax.set_ylabel(
        f"azimuth pixel (display; \u00d7 {decimate_az} to get full-res)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-image driver
# ---------------------------------------------------------------------------


def run_shear_averaging(
    image_path: Path,
    save_png: Path,
    extra_args: list[str],
    python_exe: str,
    shear_script: Path,
) -> tuple[Path, float]:
    """Invoke shear_averaging.py; return (path to `_box_stats.csv`, runtime).

    The script writes ``<save_png.stem>_box_stats.csv`` next to
    ``save_png``, so we just return that.
    """
    save_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe, str(shear_script), str(image_path),
        "--save", str(save_png),
    ] + list(extra_args)
    print(f"  \u2192 running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, check=False)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"shear_averaging exited with code {proc.returncode} for "
            f"{image_path.name}"
        )
    csv_path = save_png.with_name(f"{save_png.stem}_box_stats.csv")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"shear_averaging did not produce {csv_path.name} "
            f"(expected next to {save_png.name})"
        )
    return csv_path, dt


def evaluate_one(
    image_path: Path,
    out_dir: Path,
    extra_shear_args: list[str],
    python_exe: str,
    shear_script: Path,
    decimate_az: int,
    decimate_rg: int,
    skip_existing_csv: bool,
) -> Optional[dict]:
    """Full end-to-end for a single image.

    Returns the per-image summary dict, or ``None`` if the image was
    skipped (e.g. missing GT sidecar).
    """
    print(f"\n=== {image_path.name} ===")

    gt_path = _default_gt_path(image_path)
    if not gt_path.exists():
        print(
            f"  (skip) no GT sidecar at {gt_path.name}; run "
            f"label_ground_truth.py first."
        )
        return None
    gt_doc = load_gt_json(gt_path)
    gt_boxes = list(gt_doc.get("boxes", []))
    print(f"  GT boxes: {len(gt_boxes)}   (sidecar: {gt_path.name})")

    per_image_dir = out_dir / image_path.stem
    per_image_dir.mkdir(parents=True, exist_ok=True)
    save_png = per_image_dir / "shear.png"
    csv_path = save_png.with_name(f"{save_png.stem}_box_stats.csv")

    if skip_existing_csv and csv_path.exists():
        print(
            f"  (--skip-existing-csv) reusing existing {csv_path.name}"
        )
        runtime_s = float("nan")
    else:
        csv_path, runtime_s = run_shear_averaging(
            image_path, save_png, extra_shear_args, python_exe, shear_script,
        )
        print(f"  shear_averaging runtime: {runtime_s:.1f} s")

    # Compute Number_of_Range_Looks the same way shear_averaging does.
    range_pixel_spacing_m = gt_doc.get("range_pixel_spacing_m")
    n_range_looks = compute_range_looks(range_pixel_spacing_m)
    print(
        f"  Number_of_Range_Looks = {n_range_looks} "
        f"(range_spacing={range_pixel_spacing_m} m)"
    )

    raw_rows = read_box_stats_csv(csv_path)
    detections = [detection_to_full_res(r, n_range_looks) for r in raw_rows]
    print(f"  Detections in CSV: {len(detections)}")

    per_det, per_gt = match_center_in_gt(detections, gt_boxes)
    summary = summarise(per_det, per_gt)
    print(
        f"  \u2192 TP={summary['TP']}  TP_dup={summary['TP_dup']}  "
        f"FA={summary['FA']}  FN={summary['FN']}  "
        f"P={summary['precision']:.3f}  R={summary['recall']:.3f}  "
        f"F1={summary['F1']:.3f}"
    )

    # Per-detection CSV.
    per_det_path = per_image_dir / f"{image_path.stem}_eval.csv"
    with per_det_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "det_idx", "csv_idx",
            "y_c_full", "x_c_full", "h_full", "w_full",
            "y_c_deg", "x_c_deg", "h_deg", "w_deg",
            "matched_gt_id", "status",
        ])
        for r, d in zip(per_det, detections):
            w.writerow([
                r["det_idx"], d["idx"],
                f"{d['y_c']:.1f}", f"{d['x_c']:.1f}",
                f"{d['h']:.1f}", f"{d['w']:.1f}",
                f"{d['y_c_deg']:.1f}", f"{d['x_c_deg']:.1f}",
                f"{d['h_deg']:.1f}", f"{d['w_deg']:.1f}",
                "" if r["matched_gt_id"] is None else r["matched_gt_id"],
                r["status"],
            ])

    # Per-GT CSV.
    per_gt_path = per_image_dir / f"{image_path.stem}_eval_gt.csv"
    with per_gt_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "gt_id", "y_c", "x_c", "h", "w", "class",
            "n_matches", "primary_det_idx", "matched_det_idxs",
        ])
        gt_by_id = {int(g["id"]): g for g in gt_boxes}
        for r in per_gt:
            g = gt_by_id[r["gt_id"]]
            w.writerow([
                r["gt_id"],
                f"{g['y_c']:.1f}", f"{g['x_c']:.1f}",
                g["h"], g["w"], g.get("class", "") or "",
                len(r["matched_det_idxs"]),
                "" if r["primary_det_idx"] is None else r["primary_det_idx"],
                ";".join(str(i) for i in r["matched_det_idxs"]),
            ])

    # Overlay figure. Needs the actual SLC re-loaded (we don't cache
    # across images to keep RAM sane; per-scene load is ~5-10 s).
    print("  Reloading SLC for overlay figure...")
    s, _left, _rg, _az = load_slc_any(image_path)
    overlay_path = per_image_dir / f"{image_path.stem}_eval_overlay.png"
    render_overlay(
        s, overlay_path,
        gt_boxes=gt_boxes,
        detections=detections,
        per_det=per_det, per_gt=per_gt,
        image_name=image_path.name,
        decimate_az=decimate_az,
        decimate_rg=decimate_rg,
        summary=summary,
    )
    print(f"  Saved \u2192 {overlay_path.name}")
    del s

    result = {
        "image": image_path.name,
        "image_path": str(image_path),
        "n_gt": summary["n_gt"],
        "n_det": summary["n_det"],
        "TP": summary["TP"],
        "TP_dup": summary["TP_dup"],
        "FA": summary["FA"],
        "FN": summary["FN"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "F1": summary["F1"],
        "runtime_s": runtime_s,
        "eval_at": _now_iso(),
        "shear_args": " ".join(extra_shear_args),
    }

    # Also drop a JSON summary next to the per-image outputs so
    # downstream sweeps can pick it up without parsing CSVs.
    with (per_image_dir / f"{image_path.stem}_eval.json").open("w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _split_argv(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv on the first ``--`` so anything after it is forwarded."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1:]
    return argv, []


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    argv_ours, argv_shear = _split_argv(argv)
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Extra shear_averaging flags go AFTER `--`.",
    )
    p.add_argument(
        "path", type=Path,
        help="Either a single `.tif`/`.tiff`/`.npy` image, or a directory "
             "whose immediate contents are scanned for `.tif`/`.tiff`.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("./eval_out"), dest="out_dir",
        help="Root output folder. One subfolder per image lands inside "
             "(along with `eval_summary.csv`). Default: ./eval_out",
    )
    p.add_argument(
        "--python", type=str, default=sys.executable,
        help="Python interpreter used to invoke shear_averaging.py. "
             "Default: sys.executable (this interpreter).",
    )
    p.add_argument(
        "--shear-script", type=Path,
        default=_SCRIPTS_DIR / "shear_averaging.py",
        dest="shear_script",
        help="Path to shear_averaging.py. Default: sibling in scripts/.",
    )
    p.add_argument(
        "--decimate-az", type=int, default=10, dest="decimate_az",
        help="Azimuth decimation for the overlay figure. Default: 10.",
    )
    p.add_argument(
        "--decimate-rg", type=int, default=2, dest="decimate_rg",
        help="Range decimation for the overlay figure. Default: 2.",
    )
    p.add_argument(
        "--skip-existing-csv", action="store_true",
        dest="skip_existing_csv",
        help="If the per-image `_box_stats.csv` already exists inside the "
             "output folder, DO NOT re-run shear_averaging \u2014 just "
             "re-score it. Useful when tuning the matching, not the "
             "detector.",
    )
    p.add_argument(
        "--pattern", type=str, default="*.tif",
        help="Glob used when `path` is a directory. Default: '*.tif'. "
             "Set to e.g. '*.tif' or '*.{tif,tiff}'.",
    )
    return p.parse_args(argv_ours), argv_shear


def main(argv: Optional[list[str]] = None) -> int:
    args, extra_shear_args = _parse_args(
        argv if argv is not None else sys.argv[1:]
    )
    if any(a in ("--save", "--no-save") for a in extra_shear_args):
        print(
            "  (error) do not forward --save/--no-save to shear_averaging;"
            " this script controls that.", file=sys.stderr,
        )
        return 2
    if not args.shear_script.exists():
        raise FileNotFoundError(args.shear_script)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "eval_summary.csv"
    summary_exists = summary_path.exists()

    if args.path.is_dir():
        images = sorted(args.path.glob(args.pattern))
        # Also fold in .tiff without duplicating what --pattern already caught.
        if args.pattern == "*.tif":
            images += [
                p for p in sorted(args.path.glob("*.tiff"))
                if p not in images
            ]
        if not images:
            print(
                f"  (error) no images matching {args.pattern!r} in "
                f"{args.path}", file=sys.stderr,
            )
            return 2
    else:
        images = [args.path]

    print(f"Evaluating {len(images)} image(s). Output \u2192 {args.out_dir}")

    all_rows: list[dict] = []
    for img in images:
        try:
            row = evaluate_one(
                image_path=img.resolve(),
                out_dir=args.out_dir.resolve(),
                extra_shear_args=extra_shear_args,
                python_exe=args.python,
                shear_script=args.shear_script.resolve(),
                decimate_az=args.decimate_az,
                decimate_rg=args.decimate_rg,
                skip_existing_csv=args.skip_existing_csv,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  (error) {img.name}: {exc!r}", file=sys.stderr)
            continue
        if row is not None:
            all_rows.append(row)

    # Append per-image rows to the running summary.
    if all_rows:
        with summary_path.open("a", newline="") as f:
            w = csv.writer(f)
            if not summary_exists:
                w.writerow([
                    "image", "n_gt", "n_det", "TP", "TP_dup",
                    "FA", "FN", "precision", "recall", "F1",
                    "runtime_s", "eval_at", "shear_args",
                ])
            for r in all_rows:
                w.writerow([
                    r["image"], r["n_gt"], r["n_det"], r["TP"], r["TP_dup"],
                    r["FA"], r["FN"],
                    f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}",
                    f"{r['F1']:.4f}",
                    f"{r['runtime_s']:.2f}",
                    r["eval_at"], r["shear_args"],
                ])
        print(f"\nAppended {len(all_rows)} row(s) \u2192 {summary_path}")

    # Aggregate print.
    if all_rows:
        tp = sum(r["TP"] for r in all_rows)
        fa = sum(r["FA"] for r in all_rows)
        fn = sum(r["FN"] for r in all_rows)
        p_agg = tp / (tp + fa) if (tp + fa) > 0 else float("nan")
        r_agg = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        print(
            "\nAggregate  "
            f"TP={tp} FA={fa} FN={fn}  "
            f"P={p_agg:.3f}  R={r_agg:.3f}  "
            f"(over {len(all_rows)} images)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
