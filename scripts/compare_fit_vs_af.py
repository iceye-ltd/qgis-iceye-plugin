"""Join per-box linear-fit quality with the autofocus decision.

Inputs (all produced by the shear_averaging.py pipeline):

* ``<scene>.npz``                   -- scene bundle used by
  ``analyze_phase_fit_quality.py`` for the fit-quality metrics.
* ``<scene>_fit_quality.csv``       -- per-box fit-quality CSV.
* ``<scene>.log``                   -- stdout log of the run; contains
  ``[af] best_deviation=... contrast X.XX -> Y.YY (...dB; range-walk ...)``
  lines in box order (0..K-1).
* ``<scene>_per_box/autofocus_summary.csv`` (optional) -- adds y/x/h/w and the
  ``best_look_rows`` / ``n_sub_contrast_improved`` fields for the boxes that
  were actually saved (|best_deviation| >= threshold).

The script writes:

* ``<scene>_fit_vs_af.csv``  -- joined per-box table.
* ``<scene>_fit_vs_af.png``  -- 6-panel comparison figure.

Categorisation used throughout:

* ``needs_af``   := ``|best_deviation| >= 3.6`` (the pipeline threshold that
  decides whether the AF result is saved).
* ``af_helped``  := ``needs_af`` AND ``gain_db > 0``.
* ``pga_used``   := log line contains ``range-walk + PGA``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np


AF_LINE_RE = re.compile(
    r"^\[af\] best_deviation=([+-]?\d+\.\d+)\s+"
    r"contrast\s+([+-]?\d+\.\d+)\s*->\s*([+-]?\d+\.\d+)\s*"
    r"\(([+-]?\d+\.\d+)\s*dB;\s*(range-walk(?:\s*\+\s*PGA(?:\s*@\s*look_rows=(\d+))?)?)"
)
NEEDS_AF_THRESH = 3.6


def parse_log(log_path: Path) -> list[dict]:
    out: list[dict] = []
    with log_path.open("r") as fh:
        for line in fh:
            if not line.startswith("[af] "):
                continue
            m = AF_LINE_RE.match(line.strip())
            if not m:
                continue
            best_dev = float(m.group(1))
            c_before = float(m.group(2))
            c_after = float(m.group(3))
            gain_db = float(m.group(4))
            mode = m.group(5)
            look_rows = int(m.group(6)) if m.group(6) else -1
            pga = "PGA" in mode
            out.append(dict(
                best_deviation=best_dev,
                contrast_before=c_before,
                contrast_after=c_after,
                gain_db=gain_db,
                af_pga=int(pga),
                af_look_rows=look_rows,
            ))
    return out


def load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def to_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return -1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", type=str,
                   help="scene NPZ produced by shear_averaging.py (used only for its name)")
    p.add_argument("--fit-csv", type=str, required=True,
                   help="per-box fit-quality CSV from analyze_phase_fit_quality.py")
    p.add_argument("--log", type=str, default=None,
                   help="path to the run's .log (defaults to <scene>.log next to the NPZ)")
    p.add_argument("--af-summary", type=str, default=None,
                   help="path to autofocus_summary.csv (optional; only adds extras)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="directory for the joined CSV / figure")
    args = p.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    fit_csv = Path(args.fit_csv).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve() if args.log else npz_path.with_suffix(".log")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for pth, label in [(fit_csv, "fit-quality CSV"), (log_path, "log file")]:
        if not pth.is_file():
            sys.exit(f"error: {label} not found: {pth}")

    fit_rows = load_csv(fit_csv)
    af_rows = parse_log(log_path)
    if len(af_rows) != len(fit_rows):
        print(
            f"[warn] log has {len(af_rows)} [af] lines, fit CSV has {len(fit_rows)} rows;"
            " they should match one-to-one"
        )
    n = min(len(af_rows), len(fit_rows))

    # Optional autofocus_summary.csv for the boxes that were saved
    saved_by_idx: dict[int, dict] = {}
    if args.af_summary:
        summ_path = Path(args.af_summary).expanduser().resolve()
        if summ_path.is_file():
            for r in load_csv(summ_path):
                saved_by_idx[to_int(r["box_idx"])] = r
        else:
            print(f"[warn] autofocus_summary.csv not found at {summ_path}")

    merged: list[dict] = []
    for k in range(n):
        fq = fit_rows[k]
        af = af_rows[k]
        best_dev = af["best_deviation"]
        needs_af = int(abs(best_dev) >= NEEDS_AF_THRESH)
        saved = saved_by_idx.get(k)
        merged.append(dict(
            idx=k,
            n_rows=to_int(fq["n_rows"]),
            slope_total_rad=to_float(fq["slope_total_rad"]),
            slope_rad_per_row=to_float(fq["slope_rad_per_row"]),
            rms_rad=to_float(fq["rms_rad"]),
            mad_rad=to_float(fq["mad_rad"]),
            median_abs_rad=to_float(fq["median_abs_rad"]),
            frac_within_0p5=to_float(fq["frac_within_0.5rad"]),
            frac_within_1p0=to_float(fq["frac_within_1.0rad"]),
            frac_within_1p5=to_float(fq["frac_within_1.5rad"]),
            frac_within_1p0_coh=to_float(fq["frac_within_1.0rad_coh"]),
            is_strong=to_int(fq["is_strong"]),
            best_deviation=best_dev,
            contrast_before=af["contrast_before"],
            contrast_after=af["contrast_after"],
            af_gain_db=af["gain_db"],
            af_pga=af["af_pga"],
            af_look_rows=af["af_look_rows"],
            needs_af=needs_af,
            af_saved=int(saved is not None),
        ))

    out_csv = out_dir / f"{npz_path.stem}_fit_vs_af.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    print(f"Wrote {out_csv} ({len(merged)} rows)")

    # Summaries
    a = np.array([[m[k] for k in
                   ("frac_within_0p5", "frac_within_1p0", "frac_within_1p5",
                    "frac_within_1p0_coh", "rms_rad", "mad_rad",
                    "best_deviation", "af_gain_db", "needs_af",
                    "af_pga", "n_rows", "slope_total_rad")]
                  for m in merged], dtype=float)
    (frac05, frac10, frac15, frac10_coh, rms, mad,
     best_dev, gain_db, needs_af, pga, n_rows, slope_tot) = a.T

    print("\n=== decision counts ===")
    print(f"  total boxes            : {len(merged)}")
    print(f"  needs_af (|dev|>=3.6)  : {int(needs_af.sum())}"
          f"   of which PGA used : {int(pga[needs_af == 1].sum())}")
    print(f"  AF gain > 0 dB         : {int((gain_db > 0).sum())}"
          f"  AF gain > 1 dB : {int((gain_db > 1).sum())}"
          f"  AF gain > 3 dB : {int((gain_db > 3).sum())}")
    print(f"  AF gain < 0 dB (worse) : {int((gain_db < 0).sum())}")

    def med_iqr(x):
        x = x[np.isfinite(x)]
        return (np.median(x), np.percentile(x, 25), np.percentile(x, 75)) if len(x) else (np.nan,)*3

    print("\n=== frac(|r|<=1 rad)  by needs_af ===")
    for tag, mask in [("needs_af=1", needs_af == 1), ("needs_af=0", needs_af == 0)]:
        m, q1, q3 = med_iqr(frac10[mask])
        mw, qw1, qw3 = med_iqr(frac10_coh[mask])
        rm, r1, r3 = med_iqr(rms[mask])
        print(f"  {tag}: n={int(mask.sum()):3d}  "
              f"frac<=1: med={m:.3f} [{q1:.3f}, {q3:.3f}]  "
              f"coh: med={mw:.3f}  RMS: med={rm:.3f} [{r1:.3f}, {r3:.3f}]")

    print("\n=== frac(|r|<=1 rad)  by AF gain buckets ===")
    for tag, mask in [
        ("gain>=3 dB", gain_db >= 3),
        ("gain in [1,3) dB", (gain_db >= 1) & (gain_db < 3)),
        ("gain in [0,1) dB", (gain_db >= 0) & (gain_db < 1)),
        ("gain in [-1,0) dB", (gain_db >= -1) & (gain_db < 0)),
        ("gain <-1 dB", gain_db < -1),
    ]:
        if mask.sum() == 0:
            continue
        m, q1, q3 = med_iqr(frac10[mask])
        print(f"  {tag}: n={int(mask.sum()):3d}  frac<=1 med={m:.3f} [{q1:.3f}, {q3:.3f}]")

    # Figures
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[warn] matplotlib unavailable: {exc}")
        return

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))

    # (0,0) frac<=1 vs best_deviation
    ax = axes[0, 0]
    mask_needs = needs_af == 1
    ax.scatter(best_dev[~mask_needs], frac10[~mask_needs], s=10, alpha=0.5, color="tab:blue",
               label=f"|dev|<3.6 (n={int((~mask_needs).sum())})")
    ax.scatter(best_dev[mask_needs], frac10[mask_needs], s=18, alpha=0.75, color="tab:red",
               label=f"|dev|>=3.6 (n={int(mask_needs.sum())})")
    ax.axhline(0.68, color="k", linestyle="--", lw=0.7)
    ax.axvline(+3.6, color="k", linestyle=":", lw=0.7)
    ax.axvline(-3.6, color="k", linestyle=":", lw=0.7)
    ax.set_xlabel("best_deviation")
    ax.set_ylabel("frac(|r|<=1 rad)")
    ax.set_title("Fit quality vs. AF best_deviation")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)

    # (0,1) frac<=1 vs AF gain_db
    ax = axes[0, 1]
    pga_m = pga.astype(bool)
    ax.scatter(gain_db[~pga_m], frac10[~pga_m], s=10, alpha=0.5, color="tab:blue",
               label=f"range-walk only (n={int((~pga_m).sum())})")
    ax.scatter(gain_db[pga_m], frac10[pga_m], s=18, alpha=0.8, color="tab:orange",
               label=f"range-walk + PGA (n={int(pga_m.sum())})")
    ax.axhline(0.68, color="k", linestyle="--", lw=0.7)
    ax.axvline(0.0, color="k", linestyle=":", lw=0.7)
    ax.set_xlabel("AF gain [dB]")
    ax.set_ylabel("frac(|r|<=1 rad)")
    ax.set_title("Fit quality vs. AF contrast gain")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)

    # (0,2) RMS residual vs |best_deviation|
    ax = axes[0, 2]
    ax.scatter(np.abs(best_dev), rms, s=8, alpha=0.5, color="tab:green")
    ax.set_xlabel("|best_deviation|")
    ax.set_ylabel("RMS residual [rad]")
    ax.set_title("RMS residual vs. |best_deviation|")

    # (1,0) histogram of best_deviation, coloured by fit quality
    ax = axes[1, 0]
    bins = np.linspace(-40, 40, 41)
    ax.hist(best_dev[frac10 < 0.5], bins=bins, alpha=0.75, color="tab:red",
            label=f"frac<=1 < 0.5 (n={int((frac10 < 0.5).sum())})")
    ax.hist(best_dev[(frac10 >= 0.5) & (frac10 < 0.7)], bins=bins, alpha=0.6, color="tab:orange",
            label=f"0.5..0.7 (n={int(((frac10 >= 0.5) & (frac10 < 0.7)).sum())})")
    ax.hist(best_dev[frac10 >= 0.7], bins=bins, alpha=0.6, color="tab:green",
            label=f">=0.7 (n={int((frac10 >= 0.7).sum())})")
    ax.axvline(+3.6, color="k", linestyle=":", lw=0.7)
    ax.axvline(-3.6, color="k", linestyle=":", lw=0.7)
    ax.set_xlabel("best_deviation")
    ax.set_ylabel("box count")
    ax.set_title("best_deviation, split by fit quality")
    ax.legend(loc="upper left", fontsize=8)

    # (1,1) bar: median frac<=1 vs AF gain bucket
    ax = axes[1, 1]
    buckets = [
        ("<= -1 dB", gain_db < -1),
        ("(-1, 0)", (gain_db >= -1) & (gain_db < 0)),
        ("[0, 1)", (gain_db >= 0) & (gain_db < 1)),
        ("[1, 3)", (gain_db >= 1) & (gain_db < 3)),
        (">= 3 dB", gain_db >= 3),
    ]
    labels = [b[0] for b in buckets]
    meds = [np.median(frac10[b[1]]) if np.any(b[1]) else 0.0 for b in buckets]
    counts = [int(b[1].sum()) for b in buckets]
    bars = ax.bar(labels, meds, color=["#B22222", "#D2691E", "#DAA520", "#2E8B57", "#1E90FF"])
    for b_, c_ in zip(bars, counts):
        ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.01,
                f"n={c_}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.68, color="k", linestyle="--", lw=0.7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("median frac(|r|<=1 rad)")
    ax.set_xlabel("AF gain bucket")
    ax.set_title("Median fit quality across AF gain buckets")

    # (1,2) 2D density-ish: |slope*n| vs |best_deviation|, coloured by frac<=1
    ax = axes[1, 2]
    sc = ax.scatter(np.abs(slope_tot), np.abs(best_dev), c=frac10, cmap="viridis",
                    s=12, vmin=0.4, vmax=0.95)
    ax.set_xscale("log")
    ax.set_xlabel("|slope * n_rows|  [rad]  (motion strength)")
    ax.set_ylabel("|best_deviation|")
    ax.set_title("Motion strength vs. AF |best_deviation|,  colour = frac(|r|<=1)")
    fig.colorbar(sc, ax=ax, label="frac(|r|<=1 rad)")

    fig.suptitle(f"{npz_path.name}: phase-fit quality vs. autofocus decision "
                 f"({len(merged)} boxes)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_png = out_dir / f"{npz_path.stem}_fit_vs_af.png"
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
