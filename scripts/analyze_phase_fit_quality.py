"""Per-box goodness-of-fit metrics for the azimuth phase-slope line.

For each kept box we already store in ``<scene>.npz``:

    box_phi_rad[k, i]        wrapped per-row phase (rad, in (-pi, pi])
    box_coh[k, i]             per-row coherence weight in [0, 1]
    box_y0[k], box_n_rows[k]  valid slice length (rest is NaN)
    box_slope_rad_per_row[k]  linear fit slope
    box_intercept_rad[k]      linear fit intercept

The fitted line at local row ``i`` is ``slope * i + intercept``. Because both
the observed phase and the model live modulo 2*pi, the residual is computed
as ``wrap_to_pi(phi - fit)``.

This script produces, next to the input NPZ:

    <base>_fit_quality.csv       one row per box: RMS / MAD / mean|r| / frac<=Tr
    <base>_fit_quality.png       histograms + slope vs. frac-in-band scatter

Usage
-----
    python scripts/analyze_phase_fit_quality.py /path/to/<scene>.npz

If the paired ``<scene>_box_stats.csv`` exists next to the NPZ, the ``is_strong``
column is joined into the output CSV so you can slice moving vs. borderline
boxes without recomputing anything.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


TR_THRESHOLDS_RAD = (0.5, 1.0, 1.5)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def load_is_strong(npz_path: Path) -> dict[int, int]:
    """Return {box_idx: is_strong} from the sibling *_box_stats.csv, if any."""
    stats = npz_path.with_name(npz_path.stem + "_box_stats.csv")
    if not stats.is_file():
        return {}
    out: dict[int, int] = {}
    with stats.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out[int(row["idx"])] = int(row["is_strong"])
            except (KeyError, ValueError):
                continue
    return out


def analyze(npz_path: Path, out_dir: Path | None = None) -> None:
    d = np.load(npz_path, allow_pickle=False)
    phi_all = d["box_phi_rad"]                    # (K, N) wrapped phase per row
    coh_all = d["box_coh"]                        # (K, N)
    slope = d["box_slope_rad_per_row"]           # (K,)
    intercept = d["box_intercept_rad"]           # (K,)
    n_rows = d["box_n_rows"]                     # (K,) valid slice length
    slope_totals = d["slope_totals_rad"]         # (K,) |slope * n|-style motion strength

    K, N = phi_all.shape
    print(f"Loaded {npz_path}: K={K} boxes, up to N={N} rows per box")

    is_strong_map = load_is_strong(npz_path)

    row_hdr = [
        "idx", "n_rows",
        "slope_total_rad", "slope_rad_per_row",
        "rms_rad", "rms_rad_coh",
        "mad_rad", "mean_abs_rad", "median_abs_rad",
    ] + [f"frac_within_{t:.1f}rad" for t in TR_THRESHOLDS_RAD] + [
        f"frac_within_{t:.1f}rad_coh" for t in TR_THRESHOLDS_RAD
    ] + ["is_strong"]

    rows_out: list[list] = []
    # For plotting; keep the ±1 rad fraction and slope magnitudes globally.
    frac_1rad = np.full(K, np.nan)
    frac_1rad_coh = np.full(K, np.nan)
    rms_arr = np.full(K, np.nan)
    mad_arr = np.full(K, np.nan)

    for k in range(K):
        n_k = int(n_rows[k])
        if n_k <= 0:
            rows_out.append([k, 0] + [np.nan] * (len(row_hdr) - 3) + [is_strong_map.get(k, -1)])
            continue

        phi = phi_all[k, :n_k].astype(np.float64, copy=False)
        w = coh_all[k, :n_k].astype(np.float64, copy=False)
        valid = np.isfinite(phi) & np.isfinite(w)
        if not np.any(valid):
            rows_out.append([k, n_k] + [np.nan] * (len(row_hdr) - 3) + [is_strong_map.get(k, -1)])
            continue

        phi = phi[valid]
        w = np.clip(w[valid], 0.0, 1.0)
        idx = np.arange(n_k, dtype=np.float64)[valid]

        fit = slope[k] * idx + intercept[k]
        r = wrap_to_pi(phi - fit)
        ar = np.abs(r)

        rms = float(np.sqrt(np.mean(r * r)))
        w_sum = float(w.sum())
        rms_w = float(np.sqrt(np.sum(w * r * r) / w_sum)) if w_sum > 0 else np.nan
        mad = float(1.4826 * np.median(np.abs(r - np.median(r))))
        mean_abs = float(np.mean(ar))
        med_abs = float(np.median(ar))

        fracs = [float(np.mean(ar <= t)) for t in TR_THRESHOLDS_RAD]
        fracs_w = (
            [float(np.sum(w * (ar <= t)) / w_sum) for t in TR_THRESHOLDS_RAD]
            if w_sum > 0
            else [np.nan] * len(TR_THRESHOLDS_RAD)
        )

        rows_out.append([
            k, n_k,
            float(slope_totals[k]), float(slope[k]),
            rms, rms_w,
            mad, mean_abs, med_abs,
        ] + fracs + fracs_w + [is_strong_map.get(k, -1)])

        frac_1rad[k] = fracs[TR_THRESHOLDS_RAD.index(1.0)]
        frac_1rad_coh[k] = fracs_w[TR_THRESHOLDS_RAD.index(1.0)]
        rms_arr[k] = rms
        mad_arr[k] = mad

    base_dir = out_dir if out_dir is not None else npz_path.parent
    base_dir.mkdir(parents=True, exist_ok=True)
    out_csv = base_dir / (npz_path.stem + "_fit_quality.csv")
    with out_csv.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(row_hdr)
        writer.writerows(rows_out)
    print(f"Wrote {out_csv}  ({len(rows_out)} rows)")

    # Quick console summary
    finite = np.isfinite(frac_1rad)
    if np.any(finite):
        f = frac_1rad[finite]
        fw = frac_1rad_coh[finite]
        print("Fit-quality summary (over {} boxes):".format(int(finite.sum())))
        print(
            "  frac(|r|<=1 rad) : min={:.3f}  p25={:.3f}  med={:.3f}  p75={:.3f}  max={:.3f}".format(
                f.min(), np.percentile(f, 25), np.median(f), np.percentile(f, 75), f.max()
            )
        )
        print(
            "  frac(|r|<=1 rad)_coh-weighted : med={:.3f}  p25={:.3f}  p75={:.3f}".format(
                np.median(fw), np.percentile(fw, 25), np.percentile(fw, 75)
            )
        )
        rf = rms_arr[finite]
        mf = mad_arr[finite]
        print(
            "  RMS residual [rad] : med={:.3f}  p25={:.3f}  p75={:.3f}".format(
                np.median(rf), np.percentile(rf, 25), np.percentile(rf, 75)
            )
        )
        print(
            "  MAD residual [rad] : med={:.3f}  p25={:.3f}  p75={:.3f}".format(
                np.median(mf), np.percentile(mf, 25), np.percentile(mf, 75)
            )
        )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"[warn] matplotlib unavailable ({exc}); skipping figure")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # Top-left: histogram of frac(|r|<=1 rad)
    ax = axes[0, 0]
    ax.hist(frac_1rad[np.isfinite(frac_1rad)], bins=40, range=(0, 1), color="tab:blue", alpha=0.7,
            label="unweighted")
    ax.hist(frac_1rad_coh[np.isfinite(frac_1rad_coh)], bins=40, range=(0, 1), color="tab:orange",
            alpha=0.5, label="coh-weighted")
    ax.axvline(0.68, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("fraction of rows with |residual| <= 1 rad")
    ax.set_ylabel("box count")
    ax.set_title("Fit quality per box")
    ax.legend(loc="upper left", fontsize=9)

    # Top-right: histogram of RMS residual
    ax = axes[0, 1]
    ax.hist(rms_arr[np.isfinite(rms_arr)], bins=40, color="tab:green", alpha=0.75, label="RMS")
    ax.hist(mad_arr[np.isfinite(mad_arr)], bins=40, color="tab:red", alpha=0.55, label="MAD")
    ax.set_xlabel("residual scale [rad]")
    ax.set_ylabel("box count")
    ax.set_title("RMS vs. MAD of wrap-to-pi residuals")
    ax.legend(loc="upper right", fontsize=9)

    # Bottom-left: frac(|r|<=1 rad) vs. |slope*n| motion strength
    ax = axes[1, 0]
    good = np.isfinite(frac_1rad) & np.isfinite(slope_totals)
    ax.scatter(np.abs(slope_totals[good]), frac_1rad[good], s=8, alpha=0.55, color="tab:blue")
    ax.set_xscale("log")
    ax.set_xlabel("|slope * n_rows|  [rad] (motion strength)")
    ax.set_ylabel("fraction with |residual| <= 1 rad")
    ax.axhline(0.68, color="k", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_title("fit quality vs. motion strength")

    # Bottom-right: RMS vs. |slope*n|
    ax = axes[1, 1]
    good = np.isfinite(rms_arr) & np.isfinite(slope_totals)
    ax.scatter(np.abs(slope_totals[good]), rms_arr[good], s=8, alpha=0.55, color="tab:green")
    ax.set_xscale("log")
    ax.set_xlabel("|slope * n_rows|  [rad]")
    ax.set_ylabel("RMS residual [rad]")
    ax.set_title("RMS residual vs. motion strength")

    fig.suptitle(f"{npz_path.name}: azimuth phase-slope fit quality  ({K} boxes)")
    fig.tight_layout()

    out_png = base_dir / (npz_path.stem + "_fit_quality.png")
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", type=str, help="scene NPZ produced by shear_averaging.py")
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="directory for the CSV/PNG (defaults to the NPZ's folder)",
    )
    args = p.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        sys.exit(f"error: {npz_path} does not exist")
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else None
    analyze(npz_path, out_dir=out_dir)


if __name__ == "__main__":
    main()
