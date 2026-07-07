"""Regression diagnosis for `scripts/shear_averaging.py` on a frozen scene.

Workflow
========

1. Run `scripts/shear_averaging.py` on the dataset listed in the baseline
   JSON (or reuse the boxes produced by a previous run via
   `--use-existing`).
2. Load the resulting `autofocus_summary.csv` — that file lists the 32
   boxes drawn in red on `{stem}_af_boxes.png`, i.e. the surviving
   detections after the autofocus `|best_deviation| >= 3.6` gate.
3. Match every baseline entry (true positive or false alarm) against the
   detections of the current run using a scaled-axis nearest-neighbour
   rule (so that small numeric drift never breaks a match).
4. Print a regression report: how many baseline TPs are still detected,
   how many baseline FAs are reproduced, and how many *new* detections
   appeared that do not correspond to anything in the baseline. New
   detections need a human eyeball — they may be either new TPs or new
   FAs.

Frozen baseline for `data_20260617_141944_569017.npy`
-----------------------------------------------------

The user accepted these 6 detections as known false alarms (approximate
full-resolution coords (y_az, x_rg_full)):

    1. (27000,  200)
    2. (34000, 1600)
    3. (47000, 2250)
    4. (39000, 3200)
    5. (45000, 4500)
    6. (18000, 2250)

Every other detection produced by the same pipeline (26 boxes) is
treated as a true positive moving target. Misses outside this set are
acknowledged and tolerated by the user — the diagnostic does not
attempt to enumerate them.

Exit codes
----------

* 0 — no regression: all baseline TPs reproduced, no new detections.
* 1 — invocation / I/O error (missing files, bad arguments).
* 2 — regression detected: at least one baseline TP went missing, or at
      least one new detection appeared that does not match the baseline.

Examples
--------

# 1) Re-run the pipeline, then diagnose:
python scripts/diagnose_baseline.py

# 2) Skip the run; just diagnose the most recent existing output:
python scripts/diagnose_baseline.py --use-existing-latest

# 3) Diagnose a specific autofocus_summary.csv:
python scripts/diagnose_baseline.py \\
    --use-existing /path/to/shear_..._per_box/autofocus_summary.csv

# 4) Save a JSON report next to the regression printout:
python scripts/diagnose_baseline.py --report-json /tmp/diag.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = (
    REPO_ROOT
    / "scripts"
    / "baselines"
    / "data_20260617_141944_569017.baseline.json"
)


def _load_baseline(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Baseline DB not found: {path}")
    with path.open("r") as f:
        baseline = json.load(f)
    if int(baseline.get("schema_version", 0)) != 1:
        raise ValueError(
            f"Unsupported baseline schema_version={baseline.get('schema_version')} "
            f"in {path}. Update this diagnostic to handle the new schema."
        )
    return baseline


def _parse_detections_from_autofocus_csv(
    csv_path: Path, n_rl: int = 6,
) -> list[dict[str, float]]:
    """Read the per-box autofocus summary written by shear_averaging.py.

    Returns one dict per surviving box with keys:
        idx       — original box index in `boxes_yxhw` (the CSV's "box_idx").
        y_c       — azimuth centre (pixels, full-res azimuth).
        x_c_dec   — range centre in `s_degraded` columns.
        x_c_full  — range centre in full-resolution range pixels
                    (= x_c_dec * Number_of_Range_Looks).
        h         — azimuth extent in full-res rows.
        w_dec     — range extent in `s_degraded` columns.
        w_full    — range extent in full-res range columns.
        best_deviation, gain_db — diagnostics carried through verbatim.

    The dec→full range conversion uses ``n_rl`` (= ``Number_of_Range_Looks``).
    Default is 6 (matches the baseline-time configuration
    `int(3/0.5) = 6` in `scripts/shear_averaging.py::main()`), but the
    current pipeline may use a different value depending on
    ``minimum_target_size / range_spacing`` — pass ``--n-looks`` to
    override. The 32 boxes drawn on `{stem}_af_boxes.png` are exactly
    the rows of this CSV.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"autofocus_summary.csv not found: {csv_path}")

    detections: list[dict[str, float]] = []
    with csv_path.open("r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(row["box_idx"])
                y_c = int(row["y"])
                x_c_dec = int(row["x"])
                h = int(row["h"])
                w_dec = int(row["w"])
                best_dev = float(row["best_deviation"])
                gain_db = float(row["gain_db"]) if row["gain_db"] not in (
                    "", "nan",
                ) else float("nan")
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Could not parse row in {csv_path}: {row!r} ({exc})"
                ) from exc
            detections.append({
                "idx": idx,
                "y_c": y_c,
                "x_c_dec": x_c_dec,
                "x_c_full": x_c_dec * n_rl,
                "h": h,
                "w_dec": w_dec,
                "w_full": w_dec * n_rl,
                "best_deviation": best_dev,
                "gain_db": gain_db,
            })
    return detections


def _scaled_distance(
    y_a: float,
    x_full_a: float,
    y_b: float,
    x_full_b: float,
    dy_tol: float,
    dx_tol: float,
) -> float:
    """Axis-scaled Euclidean distance used by the matcher."""
    dy = (y_a - y_b) / max(dy_tol, 1e-9)
    dx = (x_full_a - x_full_b) / max(dx_tol, 1e-9)
    return math.sqrt(dy * dy + dx * dx)


def _match_baseline_to_detections(
    baseline: dict[str, Any],
    detections: list[dict[str, float]],
) -> dict[str, Any]:
    """Greedy nearest-neighbour assignment of baseline entries → detections.

    Iterates through every (baseline_entry, detection) pair sorted by
    scaled distance. The first time a baseline entry or a detection is
    seen, the pair is locked in (if the distance is within tolerance).
    Either side may end up unmatched, which is exactly what the report
    needs:

    * Unmatched baseline TP        → "missed" (regression).
    * Unmatched baseline FA        → "no longer present" (improvement,
                                     reported informationally).
    * Unmatched current detection  → "new" (potential new FA or new TP;
                                     requires human triage).
    """
    tol = baseline["match_tolerance"]
    dy_tol = float(tol["dy_tol_azimuth_px"])
    dx_tol = float(tol["dx_tol_range_full_px"])
    max_d = float(tol["max_distance"])

    baseline_entries: list[dict[str, Any]] = []
    for tp in baseline.get("true_positives", []):
        baseline_entries.append({"kind": "TP", "id": tp["tp_id"], "entry": tp})
    for fa in baseline.get("false_alarms", []):
        baseline_entries.append({"kind": "FA", "id": fa["fa_id"], "entry": fa})

    pairs: list[tuple[float, int, int]] = []
    for i, base in enumerate(baseline_entries):
        bx = base["entry"]
        by, bxf = float(bx["y_c"]), float(bx["x_c_full"])
        for j, det in enumerate(detections):
            d = _scaled_distance(
                by, bxf, float(det["y_c"]), float(det["x_c_full"]),
                dy_tol, dx_tol,
            )
            pairs.append((d, i, j))
    pairs.sort(key=lambda p: p[0])

    baseline_match: dict[int, int | None] = {i: None for i in range(len(baseline_entries))}
    baseline_dist: dict[int, float] = {i: float("inf") for i in range(len(baseline_entries))}
    det_match: dict[int, int | None] = {j: None for j in range(len(detections))}

    for d, i, j in pairs:
        if d > max_d:
            break
        if baseline_match[i] is not None or det_match[j] is not None:
            continue
        baseline_match[i] = j
        baseline_dist[i] = d
        det_match[j] = i

    matched_tp: list[dict[str, Any]] = []
    missed_tp: list[dict[str, Any]] = []
    matched_fa: list[dict[str, Any]] = []
    missing_fa: list[dict[str, Any]] = []
    for i, base in enumerate(baseline_entries):
        j = baseline_match[i]
        rec = {
            "kind": base["kind"],
            "id": base["id"],
            "baseline_y_c": base["entry"]["y_c"],
            "baseline_x_c_full": base["entry"]["x_c_full"],
            "baseline_box_idx_in_csv": base["entry"].get(
                "baseline_box_idx_in_csv"
            ),
            "matched_detection": detections[j] if j is not None else None,
            "scaled_distance": baseline_dist[i] if j is not None else None,
        }
        if base["kind"] == "TP":
            (matched_tp if j is not None else missed_tp).append(rec)
        else:
            (matched_fa if j is not None else missing_fa).append(rec)

    new_detections = [
        {"detection": detections[j]}
        for j, i in det_match.items()
        if i is None
    ]
    new_detections.sort(
        key=lambda r: (r["detection"]["y_c"], r["detection"]["x_c_full"])
    )

    return {
        "matched_tp": matched_tp,
        "missed_tp": missed_tp,
        "matched_fa": matched_fa,
        "missing_fa": missing_fa,
        "new_detections": new_detections,
    }


def _run_shear(baseline: dict[str, Any], rerun_extra: list[str]) -> None:
    """Re-run `scripts/shear_averaging.py` using the baseline argv."""
    argv = list(baseline["shear_invocation"]["argv"])
    # Resolve a relative "scripts/shear_averaging.py" against the repo root.
    if len(argv) >= 2 and argv[0].startswith("python"):
        script = argv[1]
        if not Path(script).is_absolute():
            argv[1] = str((REPO_ROOT / script).resolve())
    argv = argv + list(rerun_extra)
    print(f"[diag] running: {' '.join(argv)}")
    t0 = time.perf_counter()
    proc = subprocess.run(argv, cwd=REPO_ROOT, check=False)
    dt = time.perf_counter() - t0
    print(f"[diag] shear_averaging.py exited {proc.returncode} in {dt:.1f} s")
    if proc.returncode != 0:
        raise RuntimeError(
            f"shear_averaging.py failed (exit {proc.returncode}); aborting "
            f"diagnosis."
        )


def _newest_autofocus_csv(search_root: Path, stem_prefix: str = "shear_") -> Path:
    """Find the most recently modified `autofocus_summary.csv` under
    `search_root` that lives inside a `{stem}_per_box` directory."""
    if not search_root.exists():
        raise FileNotFoundError(
            f"Search root for autofocus_summary.csv does not exist: "
            f"{search_root}"
        )
    candidates = [
        p for p in search_root.glob(f"{stem_prefix}*_per_box/autofocus_summary.csv")
        if p.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No `{stem_prefix}*_per_box/autofocus_summary.csv` files in "
            f"{search_root}. Run shear_averaging.py at least once first."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _print_report(
    baseline: dict[str, Any],
    detections: list[dict[str, float]],
    matches: dict[str, Any],
    csv_path: Path,
    verbose: bool,
) -> int:
    """Print a human-readable regression report. Returns the exit code."""
    bar = "=" * 78
    sub = "-" * 78
    print(bar)
    print("Baseline regression diagnosis")
    print(bar)
    print(f"Dataset      : {baseline['dataset_path']}")
    print(f"Baseline DB  : {DEFAULT_BASELINE}")
    print(f"Current CSV  : {csv_path}")
    print(
        f"Git baseline : {baseline.get('git_commit', '?')} "
        f"({baseline.get('frozen_at', '?')})"
    )
    print()

    n_tp = len(baseline.get("true_positives", []))
    n_fa = len(baseline.get("false_alarms", []))
    n_det = len(detections)
    n_matched_tp = len(matches["matched_tp"])
    n_missed_tp = len(matches["missed_tp"])
    n_matched_fa = len(matches["matched_fa"])
    n_missing_fa = len(matches["missing_fa"])
    n_new = len(matches["new_detections"])

    print(f"Baseline     : {n_tp} TPs + {n_fa} FAs = {n_tp + n_fa} total")
    print(f"This run     : {n_det} total detections (after autofocus gate)")
    print()
    print(sub)
    print(f"True positives reproduced : {n_matched_tp:3d} / {n_tp}")
    print(f"True positives missed     : {n_missed_tp:3d}")
    print(f"False alarms reproduced   : {n_matched_fa:3d} / {n_fa}  "
          f"(expected — known FAs)")
    print(f"False alarms cleared      : {n_missing_fa:3d}")
    print(f"New detections            : {n_new:3d}  "
          f"(unknown classification; need eyeball)")
    print(sub)
    print()

    if verbose or n_missed_tp:
        print("Missed true positives:")
        if not n_missed_tp:
            print("  (none)")
        for m in matches["missed_tp"]:
            print(
                f"  {m['id']}: baseline (y={m['baseline_y_c']}, "
                f"x_full={m['baseline_x_c_full']}) — NO MATCH"
            )
        print()
    if verbose or n_missing_fa:
        print("Cleared false alarms:")
        if not n_missing_fa:
            print("  (none)")
        for m in matches["missing_fa"]:
            print(
                f"  {m['id']}: baseline (y={m['baseline_y_c']}, "
                f"x_full={m['baseline_x_c_full']}) — NO MATCH"
            )
        print()
    if verbose or n_new:
        print("New detections (not in baseline):")
        if not n_new:
            print("  (none)")
        for r in matches["new_detections"]:
            d = r["detection"]
            print(
                f"  idx={d['idx']:3d}  y={d['y_c']:6d}  "
                f"x_full={d['x_c_full']:5d}  h={d['h']:4d}  "
                f"w_full={d['w_full']:4d}  "
                f"best_dev={d['best_deviation']:+.2f}  "
                f"gain={d['gain_db']:+.2f} dB"
            )
        print()
    if verbose:
        print("Reproduced false alarms:")
        for m in matches["matched_fa"]:
            md = m["matched_detection"]
            print(
                f"  {m['id']}: baseline ({m['baseline_y_c']}, "
                f"{m['baseline_x_c_full']}) → detection (y={md['y_c']}, "
                f"x_full={md['x_c_full']}), d={m['scaled_distance']:.3f}"
            )
        print()
        print("Reproduced true positives:")
        for m in matches["matched_tp"]:
            md = m["matched_detection"]
            print(
                f"  {m['id']}: baseline ({m['baseline_y_c']}, "
                f"{m['baseline_x_c_full']}) → detection (y={md['y_c']}, "
                f"x_full={md['x_c_full']}), d={m['scaled_distance']:.3f}"
            )
        print()

    # Verdict.
    print(bar)
    delta_fa = n_matched_fa - n_fa  # 0 means no change, negative is good
    print(
        f"Δ false alarms reproduced = {delta_fa:+d}   "
        f"Δ true positives detected = {n_matched_tp - n_tp:+d}   "
        f"new detections = {n_new}"
    )

    regressions: list[str] = []
    if n_missed_tp:
        regressions.append(f"{n_missed_tp} baseline TP(s) lost")
    if n_new:
        regressions.append(
            f"{n_new} new detection(s) (potential new false alarm(s))"
        )
    if not regressions:
        print("Verdict: NO REGRESSION DETECTED.")
        print(bar)
        return 0
    print("Verdict: REGRESSION DETECTED")
    for r in regressions:
        print(f"  - {r}")
    print(bar)
    return 2


def _write_json_report(
    out: Path,
    baseline: dict[str, Any],
    detections: list[dict[str, float]],
    matches: dict[str, Any],
    csv_path: Path,
    exit_code: int,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "diagnosed_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_db": str(DEFAULT_BASELINE),
        "baseline_git_commit": baseline.get("git_commit"),
        "baseline_frozen_at": baseline.get("frozen_at"),
        "dataset_path": baseline["dataset_path"],
        "current_autofocus_csv": str(csv_path),
        "current_total_detections": len(detections),
        "baseline_n_tp": len(baseline.get("true_positives", [])),
        "baseline_n_fa": len(baseline.get("false_alarms", [])),
        "n_matched_tp": len(matches["matched_tp"]),
        "n_missed_tp": len(matches["missed_tp"]),
        "n_matched_fa": len(matches["matched_fa"]),
        "n_missing_fa": len(matches["missing_fa"]),
        "n_new_detections": len(matches["new_detections"]),
        "matches": matches,
        "exit_code": exit_code,
    }
    out.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[diag] JSON report → {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"Path to baseline JSON. Default: {DEFAULT_BASELINE}.",
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--use-existing",
        type=Path,
        default=None,
        metavar="AUTOFOCUS_SUMMARY_CSV",
        help="Skip the rerun; diagnose this autofocus_summary.csv "
             "(absolute path).",
    )
    src.add_argument(
        "--use-existing-latest",
        action="store_true",
        help="Skip the rerun; diagnose the most recently modified "
             "autofocus_summary.csv under the baseline's default save "
             "directory.",
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=None,
        help="When using --use-existing-latest, search this directory for "
             "shear_*_per_box/autofocus_summary.csv. Defaults to the "
             "parent of the baseline's `dataset_path` patches directory.",
    )
    parser.add_argument(
        "--rerun-extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments appended verbatim to the baseline's "
             "`shear_invocation.argv` when re-running. Useful for "
             "throwaway debugging flags. Anything after this is passed "
             "through, so place it last on the command line.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Also write a machine-readable JSON report to this path.",
    )
    parser.add_argument(
        "--n-looks",
        type=int,
        default=None,
        help="Number_of_Range_Looks used by the current shear_averaging.py "
             "run (used to convert x_dec → x_full when comparing to the "
             "baseline, which is stored in full-res coords). Defaults to "
             "the baseline's `number_of_range_looks` (6).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every matched/unmatched entry, not just regressions.",
    )
    args = parser.parse_args(argv)

    try:
        baseline = _load_baseline(args.baseline)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[diag] ERROR: {exc}", file=sys.stderr)
        return 1

    # 1. Decide which autofocus_summary.csv to diagnose.
    if args.use_existing is not None:
        csv_path = args.use_existing.expanduser().resolve()
    else:
        if not args.use_existing_latest:
            try:
                _run_shear(baseline, args.rerun_extra)
            except RuntimeError as exc:
                print(f"[diag] ERROR: {exc}", file=sys.stderr)
                return 1
        # Locate the newest autofocus_summary.csv.
        if args.search_root is not None:
            search_root = args.search_root.expanduser().resolve()
        else:
            # The baseline's dataset lives in `<save_dir>/patches/...npy`.
            # shear_averaging.py writes its outputs in <save_dir>.
            ds_path = Path(baseline["dataset_path"]).expanduser().resolve()
            search_root = ds_path.parent.parent
        try:
            csv_path = _newest_autofocus_csv(search_root)
        except FileNotFoundError as exc:
            print(f"[diag] ERROR: {exc}", file=sys.stderr)
            return 1

    # 2. Parse detections and compare.
    n_rl = args.n_looks if args.n_looks is not None else int(
        baseline.get("number_of_range_looks", 6)
    )
    try:
        detections = _parse_detections_from_autofocus_csv(csv_path, n_rl=n_rl)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[diag] ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"[diag] dec→full range conversion: n_rl = {n_rl} "
        f"(from {'--n-looks' if args.n_looks is not None else 'baseline JSON'})"
    )

    matches = _match_baseline_to_detections(baseline, detections)
    exit_code = _print_report(
        baseline, detections, matches, csv_path, args.verbose,
    )

    if args.report_json is not None:
        _write_json_report(
            args.report_json, baseline, detections, matches, csv_path,
            exit_code,
        )

    # Touch `shutil` import so static checkers don't flag it; we keep the
    # import available for future "copy artefacts" extensions.
    _ = shutil

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
