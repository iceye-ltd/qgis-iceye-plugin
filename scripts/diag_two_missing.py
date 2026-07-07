"""Investigate why two doubles in the WTW3YQ scene are still missing
after the mask+coh-gate fix:

  ROI A: y=80000..84000, x=850..870   (init boxes 666, 667)
  ROI B: y=72000..78000, x=870..890   (init boxes 789, 790, 851, 852)

The CSV shows the following dispositions (post-fix run):
  666: dropped_by_phase_fit_quality
  667: dropped_by_phase_fit_quality
  789: dropped_by_nested_box_overlap   (parent of 790)
  790: dropped_by_phase_fit_quality
  851: dropped_by_nested_box_overlap   (parent of 852)
  852: dropped_by_phase_fit_quality

The kept set has min score 0.925 (top-100 cap). We recompute the phase
estimate + coh-gated score for each of the 6 boxes and render an
amplitude+phase panel per box, so we can see whether the score would
land inside the top-100 with any relaxation, or whether the fit itself
looks weak.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shear_averaging import (  # noqa: E402
    apply_range_window,
    degrade_range_resolution_range_sum,
    compute_box_phase_estimates,
    compute_box_frac_within_thresh,
)
from diag_missing_doubles import load_azimuth_strip  # noqa: E402


TIF = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/"
    "ICEYE_WTW3YQ_20250104T180444Z_4439676_X7_SLED_SLC.tif"
)
NPZ = Path("/home/odogan/Desktop/ship_focusing/4439676/WTW3YQ.npz")
OUT_DIR = HERE / "missing_doubles_out"

RANGE_LOOKS = 4
RANGE_SLL_DB = 55.0
RANGE_TAYLOR_NBAR = 8
MIN_ROW_COHERENCE = 0.5
INLIER_TOL_RAD = 0.5
PHASE_FIT_THRESH_RAD = 1.5
N_SUBAPERTURE = 16
FILTER_TARGET_SIZE_M = 3.0
FILTER_MIN_DENSITY = 0.30

# (roi_tag, init_idx, disposition, y_c, x_c, h, w).
BOXES = [
    ("A", 666, "phase_fit_quality",  81928, 856, 2768,  8),
    ("A", 667, "phase_fit_quality",  81304, 860, 3728, 14),
    ("B", 789, "nested_overlap",     74888, 877, 6384, 15),
    ("B", 790, "phase_fit_quality",  76728, 868, 3088,  7),
    ("B", 851, "nested_overlap",     74344, 880, 5520, 12),
    ("B", 852, "phase_fit_quality",  74592, 879, 5472, 10),
]

MIN_KEPT_SCORE = 0.925  # from WTW3YQ.log for the current run


def _make_mask_filt(mask_dec: np.ndarray, az_m_dec: float, rg_m_dec: float) -> np.ndarray:
    """Reproduce the pipeline's isolated-pixel-filtered mask (mask_filt)."""
    az_win = max(1, int(round(FILTER_TARGET_SIZE_M / az_m_dec)))
    rg_win = max(1, int(round(FILTER_TARGET_SIZE_M / rg_m_dec)))
    density = uniform_filter(mask_dec.astype(np.float32),
                             size=(az_win, rg_win), mode="constant")
    return (mask_dec.astype(bool) & (density >= FILTER_MIN_DENSITY)).astype(np.float32)


def _slice_mask_for_strip(mask_full: np.ndarray, y_lo: int, y_hi: int, n_sub: int):
    r_lo = y_lo // n_sub
    r_hi = min(mask_full.shape[0], -(-y_hi // n_sub))
    return mask_full[r_lo:r_hi], y_lo - r_lo * n_sub


def _row_amp_profile(amp: np.ndarray) -> np.ndarray:
    """Mean amplitude per azimuth row inside a box."""
    return np.mean(amp, axis=1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    y_lo = min(b[3] - b[5] // 2 for b in BOXES) - 200
    y_hi = max(b[3] + b[5] // 2 for b in BOXES) + 200
    print(f"Loading azimuth strip [{y_lo}, {y_hi}) from {TIF.name} ...")
    strip = load_azimuth_strip(TIF, y_lo, y_hi, left=True)
    print(f"  strip shape = {strip.shape}")

    print("Range Taylor window + range degradation ...")
    strip_win = apply_range_window(strip, sll_db=RANGE_SLL_DB,
                                   nbar=RANGE_TAYLOR_NBAR)
    del strip
    s_deg = degrade_range_resolution_range_sum(strip_win, RANGE_LOOKS)
    del strip_win
    print(f"  s_degraded strip shape = {s_deg.shape}")

    print(f"Loading mask_dec from {NPZ.name} ...")
    npz = np.load(NPZ)
    mask_dec_full = np.asarray(npz["mask_dec"]).astype(np.float32)
    az_m_dec = float(npz["az_m_per_px_mask_dec"])
    rg_m_dec = float(npz["rg_m_per_px_mask_dec"])
    print(f"  mask_dec shape = {mask_dec_full.shape}  "
          f"az_m/px={az_m_dec:.2f}  rg_m/px={rg_m_dec:.2f}")

    print("Reconstructing mask_filt (isolated-pixel filter) ...")
    mask_filt_full = _make_mask_filt(mask_dec_full, az_m_dec, rg_m_dec)
    print(f"  mask_filt density: raw={mask_dec_full.mean():.4f}   "
          f"filt={mask_filt_full.mean():.4f}")

    mask_slice, strip_row_offset = _slice_mask_for_strip(
        mask_filt_full, y_lo, y_lo + s_deg.shape[0], N_SUBAPERTURE,
    )
    print(f"  strip-local mask_filt slice shape = {mask_slice.shape}   "
          f"row-offset inside strip = {strip_row_offset}")

    if strip_row_offset > 0:
        pad = np.zeros((strip_row_offset, s_deg.shape[1]), dtype=s_deg.dtype)
        s_deg_pad = np.vstack([pad, s_deg])
    else:
        s_deg_pad = s_deg

    fig, axes = plt.subplots(
        3, len(BOXES),
        figsize=(3.5 * len(BOXES), 12),
        constrained_layout=True,
    )
    fig.suptitle(
        f"Two doubles still missing after mask+coh fix.  "
        f"Current top-100 kept score min = {MIN_KEPT_SCORE:.3f}\n"
        "Row 1: mean|amp| profile along azimuth (with mask_filt overlay)\n"
        "Row 2: azimuth-mean amplitude image  (yellow: mask_filt=1)\n"
        "Row 3: Δφ vs azimuth row, colour = coh, red = wrap-aware fit",
        fontsize=11,
    )

    for col, (tag, init_idx, disp, y_c, x_c, h, w) in enumerate(BOXES):
        y_c_loc = int(y_c) - y_lo
        box_local = np.array([[y_c_loc + strip_row_offset, x_c, h, w]],
                             dtype=np.int64)
        phi_arr, coh_arr, sl_arr, ic_arr, y0_arr, n_arr, res_arr, in_arr = (
            compute_box_phase_estimates(
                box_local, s_deg_pad,
                min_row_coherence=MIN_ROW_COHERENCE,
                inlier_tol_rad=INLIER_TOL_RAD,
                mask_dec=mask_slice,
                N_subaperture=N_SUBAPERTURE,
            )
        )
        _, frac_coh = compute_box_frac_within_thresh(
            phi_arr, coh_arr, sl_arr, ic_arr, n_arr,
            thresh_rad=PHASE_FIT_THRESH_RAD,
            min_row_coherence=MIN_ROW_COHERENCE,
        )

        n_rows = int(min(n_arr[0], phi_arr.shape[1]))
        phi = np.asarray(phi_arr[0][:n_rows], dtype=np.float64)
        coh = np.asarray(coh_arr[0][:n_rows], dtype=np.float64)
        slope_pr = float(sl_arr[0])
        intercept = float(ic_arr[0])
        residual = float(res_arr[0])
        score = float(frac_coh[0])
        n_inliers = int(in_arr[0])
        y0_l = int(y0_arr[0]) - strip_row_offset
        y0_abs = y0_l + y_lo

        # Amplitude thumbnail inside the box (unwindowed strip is
        # gone, so use |s_deg|).
        y_top = max(0, y_c_loc - h // 2)
        y_bot = min(s_deg.shape[0], y_c_loc + h // 2 + 1)
        x_left = max(0, x_c - w // 2)
        x_right = min(s_deg.shape[1], x_c + w // 2 + 1)
        amp_chip = np.abs(s_deg[y_top:y_bot, x_left:x_right])

        # Mask overlay for the same chip.
        m_r_lo = (y_top + y_lo) // N_SUBAPERTURE
        m_r_hi = min(mask_dec_full.shape[0],
                     -(-(y_bot + y_lo) // N_SUBAPERTURE))
        mask_chip_dec = mask_filt_full[m_r_lo:m_r_hi, x_left:x_right]
        # Upsample mask (dec-frame) so it matches the amp chip rows.
        mask_chip = np.repeat(mask_chip_dec, N_SUBAPERTURE, axis=0)
        row_start = (y_top + y_lo) - m_r_lo * N_SUBAPERTURE
        mask_chip = mask_chip[row_start:row_start + amp_chip.shape[0]]

        # Row-1: amp mean-profile + mask hotness fraction (per row).
        ax0 = axes[0, col]
        row_amp = _row_amp_profile(amp_chip)
        row_hot = mask_chip.mean(axis=1) if mask_chip.size else np.zeros(row_amp.size)
        ax0.plot(np.arange(row_amp.size) + (y_top + y_lo), row_amp,
                 color="steelblue", lw=1.0, label="mean|amp|")
        ax0.set_ylabel("mean |amp|", color="steelblue")
        ax0.tick_params(axis="y", labelcolor="steelblue")
        ax1 = ax0.twinx()
        ax1.plot(np.arange(row_hot.size) + (y_top + y_lo), row_hot,
                 color="darkorange", lw=1.0, label="mask hot frac")
        ax1.set_ylabel("mask hot frac", color="darkorange")
        ax1.tick_params(axis="y", labelcolor="darkorange")
        ax1.set_ylim(-0.05, 1.05)
        ax0.set_title(f"[{tag}] init={init_idx}\nrow-profile", fontsize=9)
        ax0.grid(True, alpha=0.3)

        # Row-2: amp image + mask overlay.
        ax = axes[1, col]
        if amp_chip.size:
            vmin, vmax = np.percentile(amp_chip, [1, 99])
            ax.imshow(amp_chip, aspect="auto", cmap="gray",
                      vmin=vmin, vmax=vmax,
                      extent=[x_left, x_right,
                              (y_bot + y_lo), (y_top + y_lo)])
            m_over = np.where(mask_chip > 0, 1.0, np.nan)
            ax.imshow(m_over, aspect="auto", cmap="autumn",
                      alpha=0.35, vmin=0.0, vmax=1.0,
                      extent=[x_left, x_right,
                              (y_bot + y_lo), (y_top + y_lo)])
        ax.set_title(
            f"disp: {disp}\nh={h} w={w} n_det={0}",
            fontsize=8,
        )
        ax.set_xlabel("range col")
        ax.set_ylabel("azimuth row (s_deg abs)")

        # Row-3: Δφ vs row, colour = coh, red = fit.
        ax = axes[2, col]
        i_loc = np.arange(n_rows, dtype=np.float64)
        y_abs = y0_abs + i_loc
        valid = np.isfinite(phi)
        if valid.any():
            sc = ax.scatter(y_abs[valid], phi[valid], c=coh[valid],
                            cmap="viridis", s=8, vmin=0.0, vmax=1.0,
                            edgecolor="none", zorder=2)
            fig.colorbar(sc, ax=ax, location="right",
                         shrink=0.85, label="coh")
        if np.isfinite(slope_pr) and np.isfinite(intercept):
            line_w = np.angle(np.exp(1j * (slope_pr * i_loc + intercept)))
            if line_w.size > 1:
                jumps = np.where(np.abs(np.diff(line_w)) > np.pi)[0]
                if jumps.size:
                    line_w = line_w.copy()
                    line_w[jumps] = np.nan
            ax.plot(y_abs, line_w, color="red", lw=1.0, zorder=3)
        ax.set_ylim(-np.pi, np.pi)
        ax.axhline(0.0, color="0.5", lw=0.5, ls="--")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("azimuth row abs")
        ax.set_ylabel(r"$\Delta\varphi$  [rad]")
        finite_rows = int(np.isfinite(phi).sum())
        ax.set_title(
            f"score={score:.3f}  "
            f"{'PASS' if score >= MIN_KEPT_SCORE else 'FAIL'} "
            f"vs min={MIN_KEPT_SCORE:.3f}\n"
            f"slope·n={slope_pr * n_rows:+.2f} rad  res={residual:.2f}  "
            f"n_fin={finite_rows}/{n_rows}",
            fontsize=8,
        )

        print(
            f"[{tag}] init={init_idx:>4d}  disp={disp:>18s}  "
            f"score={score:.3f}  slope·n={slope_pr*n_rows:+.3f} rad  "
            f"res={residual:.2f}  n_inliers={n_inliers}/{n_rows}  "
            f"n_finite_rows={finite_rows}"
        )

    out = OUT_DIR / "two_missing_doubles_diag.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
