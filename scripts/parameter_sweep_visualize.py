"""Visualise the baseline detections from shear_averaging.py colour-coded by
TP (inside s_degraded[:, 70:120]) vs FA (outside that range band).

Also dumps a clean per-target CSV with each filter's "kill threshold".
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parameter_sweep import (
    BASELINE, BoxStats, classify, compute_scene_to_post_merge, count_tp_fa,
)


def main():
    patch = Path("/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_143542_433755.npy")
    s = np.load(patch)
    cache = compute_scene_to_post_merge(s, n_subaperture=8, cov_th_mult=1.5)

    # Reconstruct s_degraded once for the visualisation.
    from shear_averaging import (
        apply_range_window, degrade_range_resolution_range_sum,
    )
    s_windowed = apply_range_window(s, sll_db=55.0, nbar=8)
    s_degraded, _ = degrade_range_resolution_range_sum(s_windowed, 6)
    amp = np.abs(s_degraded)

    # Find baseline-kept indices.
    _, _, kept = count_tp_fa(cache, **BASELINE)
    boxes = [(i, cache.stats[i]) for i in kept]
    tp_boxes = [(i, bs) for i, bs in boxes if classify(bs.x_c) == "TP"]
    fa_boxes = [(i, bs) for i, bs in boxes if classify(bs.x_c) == "FA"]
    print(f"TP={len(tp_boxes)}, FA={len(fa_boxes)}")

    # ---- Figure 1: full scene with TP/FA overlay + truth band ---------
    fig, axes = plt.subplots(1, 2, figsize=(14, 12), constrained_layout=True)
    fig.suptitle(
        "Baseline STRONG detections (|slope·n| \u2265 1.0 rad) — "
        "green = TP (in s_degraded[:, 70:120]), red = FA",
        fontsize=11,
    )
    mu, sd = float(amp.mean()), float(amp.std())
    vmax = min(float(amp.max()), mu + 4 * sd)
    vmin = max(float(amp.min()), mu - 4 * sd)

    for ax_, title in zip(axes, ("|s_degraded|", "|s_degraded|  (zoom on truth band)")):
        im = ax_.imshow(amp, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
        ax_.set_title(title)
        ax_.set_xlabel("range pixel (s_degraded)")
        ax_.set_ylabel("azimuth pixel")
        # truth band edges
        ax_.axvline(70, color="lime", lw=0.8, ls="--", alpha=0.8)
        ax_.axvline(120, color="lime", lw=0.8, ls="--", alpha=0.8)
        for i, bs in tp_boxes:
            ax_.add_patch(plt.Rectangle(
                (bs.x_c - bs.w / 2 - 0.5, bs.y_c - bs.h / 2 - 0.5),
                bs.w, bs.h,
                fill=False, edgecolor="lime", linewidth=1.4,
            ))
            ax_.text(bs.x_c, bs.y_c, str(i), color="lime",
                     ha="center", va="center", fontsize=8, fontweight="bold")
        for i, bs in fa_boxes:
            ax_.add_patch(plt.Rectangle(
                (bs.x_c - bs.w / 2 - 0.5, bs.y_c - bs.h / 2 - 0.5),
                bs.w, bs.h,
                fill=False, edgecolor="red", linewidth=1.0,
            ))
            ax_.text(bs.x_c, bs.y_c, str(i), color="red",
                     ha="center", va="center", fontsize=8)
    axes[1].set_xlim(40, 160)
    fig.colorbar(im, ax=axes, label="|s|")

    out_dir = Path("sweep_out"); out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / "tp_fa_overlay.png", dpi=140)
    print(f"saved {out_dir / 'tp_fa_overlay.png'}")

    # ---- Figure 2: scatter of |slope_tot| vs |v_max| per box, colour TP/FA
    fig2, ax2 = plt.subplots(figsize=(8, 7), constrained_layout=True)
    for i, bs in boxes:
        v_max = max(
            abs(bs.v_az_mps) if np.isfinite(bs.v_az_mps) else 0.0,
            abs(bs.v_rg_mps) if np.isfinite(bs.v_rg_mps) else 0.0,
        )
        st = abs(bs.slope_total_rad) if np.isfinite(bs.slope_total_rad) else 0.0
        col = "lime" if classify(bs.x_c) == "TP" else "red"
        ax2.scatter(v_max, st, s=40, c=col, edgecolor="k", linewidth=0.4)
        ax2.text(v_max, st, str(i), fontsize=7, ha="left", va="bottom")
    ax2.axhline(BASELINE["slope_rad_thresh"], color="k", ls="--", lw=0.7,
                label=f"slope_rad_thresh = {BASELINE['slope_rad_thresh']:g}")
    ax2.axvline(BASELINE["min_velocity_mps"], color="k", ls=":", lw=0.7,
                label=f"min_velocity_mps = {BASELINE['min_velocity_mps']:g}")
    ax2.set_xscale("symlog", linthresh=1.0)
    ax2.set_yscale("linear")
    ax2.set_xlabel(r"$\max(|v_{az}|, |v_{rg}|)$   [m/s]")
    ax2.set_ylabel(r"$|\,\mathrm{slope}\cdot n_{\rm rows}\,|$   [rad]")
    ax2.set_title("Motion-filter feature space\n"
                  "(Box passes if it is right of '|' OR above '--')")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left")
    fig2.savefig(out_dir / "motion_feature_space.png", dpi=140)
    print(f"saved {out_dir / 'motion_feature_space.png'}")

    # ---- Per-target CSV with kill thresholds ---------------------------
    csv_path = out_dir / "per_target_kill_thresholds.csv"
    with csv_path.open("w") as f:
        f.write("idx,class,y_c,x_c,h,w,n_rows,det_count,det_per_row,"
                "slope_per_row_deg,slope_total_rad,phase_residual_rad,"
                "v_az_mps,v_rg_mps,v_max_mps,"
                "x_band_kill_min,x_band_kill_max,"
                "kill_slope_deg_above,kill_residual_below_AND_skip_above,"
                "kill_v_min_above,kill_slope_rad_thresh_above\n")
        for i, bs in boxes:
            cls = classify(bs.x_c)
            slope_deg = (abs(np.degrees(bs.slope_per_row_rad))
                         if np.isfinite(bs.slope_per_row_rad) else 0.0)
            v_max = max(
                abs(bs.v_az_mps) if np.isfinite(bs.v_az_mps) else 0.0,
                abs(bs.v_rg_mps) if np.isfinite(bs.v_rg_mps) else 0.0,
            )
            st = abs(bs.slope_total_rad) if np.isfinite(bs.slope_total_rad) else 0.0
            xmin = bs.x_c if bs.x_c < 70 else ""
            xmax = bs.x_c if bs.x_c >= 120 else ""
            res = (f"{bs.residual_rad:.3f}|skip>{bs.det_count / max(1,bs.n_rows):.2f}"
                   if np.isfinite(bs.residual_rad) else "")
            f.write(",".join([
                str(i), cls, str(bs.y_c), str(bs.x_c), str(bs.h), str(bs.w),
                str(bs.n_rows), str(bs.det_count),
                f"{bs.det_count/max(1,bs.n_rows):.3f}",
                f"{slope_deg:.4f}",
                f"{bs.slope_total_rad:+.4f}" if np.isfinite(bs.slope_total_rad) else "",
                f"{bs.residual_rad:.4f}" if np.isfinite(bs.residual_rad) else "",
                f"{bs.v_az_mps:+.4f}" if np.isfinite(bs.v_az_mps) else "",
                f"{bs.v_rg_mps:+.4f}" if np.isfinite(bs.v_rg_mps) else "",
                f"{v_max:.4f}",
                str(xmin), str(xmax),
                f"{slope_deg:.4f}",
                res,
                f"{v_max:.4f}",
                f"{st:.4f}",
            ]) + "\n")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
