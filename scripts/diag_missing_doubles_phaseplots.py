"""Phase-plot panel for the 5 candidate boxes that phase_fit_quality
dropped in the four tracked ROIs (T1..T4).

The 5 boxes are all "dropped_by_phase_fit_quality" survivors of nested
cleanup — the real doubles that the top-100 cap kicked out:

  T1 -> box 852
  T2 -> box 989
  T3 -> box 1189
  T4 -> box 196
  T4 -> box 197   (T4 holds two opposing-Doppler ships)

The pipeline never wrote per-box dphase PNGs for dropped boxes (only
the final kept 100), so we recompute the phase estimate from
``s_degraded`` and reproduce the same viridis-scatter / wrapped-fit
line layout the pipeline uses for kept boxes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
NPZ = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/WTW3YQ.npz"
)
OUT_DIR = HERE / "missing_doubles_out"

RANGE_LOOKS = 4
RANGE_SLL_DB = 55.0
RANGE_TAYLOR_NBAR = 8
MIN_ROW_COHERENCE = 0.5
INLIER_TOL_RAD = 0.5
PHASE_FIT_THRESH_RAD = 1.5
N_SUBAPERTURE = 16

BOXES = [
    ("T1",  852,  74592, 879, 5472, 10, 0.828),
    ("T2",  989,  69672, 944, 5712, 10, 0.887),
    ("T3", 1189,  65552, 986, 6432, 15, 0.796),
    ("T4",  196, 101336, 661, 2992, 10, 0.859),
    ("T4",  197, 102896, 653, 3328, 11, 0.879),
]


def _slice_mask_for_strip(mask_dec_full: np.ndarray, y_lo: int, y_hi: int,
                          N_sub: int) -> tuple[np.ndarray, int]:
    """Slice ``mask_dec_full`` (shape (sub_size, N_rg)) to cover s_degraded rows [y_lo, y_hi) after upsampling by N_sub. Returns (mask_dec_slice, strip_row_offset) where strip_row_offset is the number of rows to trim off the top when materialising mask_full for the strip."""
    r_lo = y_lo // N_sub
    r_hi = min(mask_dec_full.shape[0], -(-y_hi // N_sub))
    return mask_dec_full[r_lo:r_hi], y_lo - r_lo * N_sub


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    y_lo = min(b[2] - b[4] // 2 for b in BOXES) - 200
    y_hi = max(b[2] + b[4] // 2 for b in BOXES) + 200
    print(f"Loading azimuth strip [{y_lo}, {y_hi}) from {TIF.name} ...")
    strip = load_azimuth_strip(TIF, y_lo, y_hi, left=True)
    print(f"  strip shape = {strip.shape}")
    print("Range Taylor window + range degradation ...")
    strip_win = apply_range_window(strip, sll_db=RANGE_SLL_DB, nbar=RANGE_TAYLOR_NBAR)
    del strip
    s_deg = degrade_range_resolution_range_sum(strip_win, RANGE_LOOKS)
    del strip_win
    print(f"  s_degraded strip shape = {s_deg.shape}")

    print(f"Loading mask_dec from {NPZ.name} ...")
    npz = np.load(NPZ)
    mask_dec_full = np.asarray(npz["mask_dec"])
    print(f"  mask_dec shape = {mask_dec_full.shape}  N_subaperture = {N_SUBAPERTURE}")
    mask_dec_slice, strip_row_offset = _slice_mask_for_strip(
        mask_dec_full, y_lo, y_lo + s_deg.shape[0], N_SUBAPERTURE,
    )
    print(
        f"  strip-local mask_dec slice shape = {mask_dec_slice.shape}   "
        f"row-offset inside strip = {strip_row_offset}"
    )

    fig, axes = plt.subplots(
        2, len(BOXES), figsize=(4.5 * len(BOXES), 10.2),
        constrained_layout=True,
    )
    fig.suptitle(
        "Phase plots of the 5 dropped-by-phase_fit_quality survivors\n"
        "TOP row = current pipeline (no mask, amplitude-only weighting)   "
        "BOTTOM row = mask-aware weighting "
        "(m[i,j]·m[i+1,j] factor)\n"
        "colour = per-row coherence   red = wrap-aware linear fit",
        fontsize=11,
    )

    for col, (tag, init_idx, y_c, x_c, h, w, score_csv) in enumerate(BOXES):
        y_c_loc = int(y_c) - y_lo
        box_local = np.array([[y_c_loc, x_c, h, w]], dtype=np.int64)

        # Reference (unmasked, matches current pipeline).
        phi_ref, coh_ref, sl_ref, ic_ref, y0_ref, n_ref, res_ref, in_ref = (
            compute_box_phase_estimates(
                box_local, s_deg,
                min_row_coherence=MIN_ROW_COHERENCE,
                inlier_tol_rad=INLIER_TOL_RAD,
            )
        )
        _, frac_ref_coh = compute_box_frac_within_thresh(
            phi_ref, coh_ref, sl_ref, ic_ref, n_ref,
            thresh_rad=PHASE_FIT_THRESH_RAD,
        )

        # Mask-aware version — s_deg's local row 0 corresponds to mask_dec_slice's local row (strip_row_offset), so we shift the box's local y up so that box_local + strip_row_offset lands on the correct mask_dec row. Simpler alternative: give compute_box_phase_estimates a mask_dec that starts at mask_dec row 0 corresponding to s_deg row (-strip_row_offset).
        # Instead: pass mask_dec_slice with N_sub, and shift the box y in a "mask-frame" where row 0 aligns to r_lo*N_sub in the original scene, i.e., row -strip_row_offset in s_deg.
        s_deg_padded = s_deg
        if strip_row_offset > 0:
            pad_top = np.zeros((strip_row_offset, s_deg.shape[1]), dtype=s_deg.dtype)
            s_deg_padded = np.vstack([pad_top, s_deg])
        box_local_padded = np.array(
            [[y_c_loc + strip_row_offset, x_c, h, w]], dtype=np.int64,
        )
        phi_arr, coh_arr, sl_arr, ic_arr, y0_arr, n_arr, res_arr, in_arr = (
            compute_box_phase_estimates(
                box_local_padded, s_deg_padded,
                min_row_coherence=MIN_ROW_COHERENCE,
                inlier_tol_rad=INLIER_TOL_RAD,
                mask_dec=mask_dec_slice,
                N_subaperture=N_SUBAPERTURE,
            )
        )
        _, frac_masked_coh = compute_box_frac_within_thresh(
            phi_arr, coh_arr, sl_arr, ic_arr, n_arr,
            thresh_rad=PHASE_FIT_THRESH_RAD,
        )
        n_k = int(min(n_arr[0], phi_arr.shape[1]))
        y0_l = int(y0_arr[0]) - strip_row_offset
        slope_pr = float(sl_arr[0])
        intercept = float(ic_arr[0])
        residual = float(res_arr[0])
        n_inliers = int(in_arr[0])
        y0_abs = y0_l + y_lo

        _plot_phase_panel(
            axes[0, col], fig,
            phi=phi_ref[0], coh=coh_ref[0],
            slope_pr=float(sl_ref[0]), intercept=float(ic_ref[0]),
            n_rows=int(min(n_ref[0], phi_ref.shape[1])),
            y0_abs=int(y0_ref[0]) + y_lo,
            title=(
                f"[{tag}]  init={init_idx}\n"
                f"OLD (no mask)\n"
                f"score={float(frac_ref_coh[0]):.3f}  "
                f"csv={score_csv:.3f}\n"
                f"slope·n={float(sl_ref[0]) * int(n_ref[0]):+.2f} rad  "
                f"res={float(res_ref[0]):.2f}"
            ),
        )

        _plot_phase_panel(
            axes[1, col], fig,
            phi=phi_arr[0], coh=coh_arr[0],
            slope_pr=slope_pr, intercept=intercept,
            n_rows=n_k, y0_abs=y0_abs,
            title=(
                f"[{tag}]  init={init_idx}\n"
                f"NEW (mask-aware)\n"
                f"score={float(frac_masked_coh[0]):.3f}\n"
                f"slope·n={slope_pr * n_k:+.2f} rad  res={residual:.2f}"
            ),
        )

        print(
            f"[{tag}] init={init_idx}  "
            f"old score={float(frac_ref_coh[0]):.3f}  "
            f"new score={float(frac_masked_coh[0]):.3f}  "
            f"(csv={score_csv:.3f})   "
            f"old res={float(res_ref[0]):.2f}  new res={residual:.2f}"
        )

    out = OUT_DIR / "phaseplots_5_dropped_survivors_masked_vs_unmasked.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"→ {out}")


def _plot_phase_panel(
    ax,
    fig,
    *,
    phi: np.ndarray,
    coh: np.ndarray,
    slope_pr: float,
    intercept: float,
    n_rows: int,
    y0_abs: int,
    title: str,
) -> None:
    phi = np.asarray(phi[:n_rows], dtype=np.float64)
    coh = np.asarray(coh[:n_rows], dtype=np.float64)
    i_loc = np.arange(n_rows, dtype=np.float64)
    y_abs = y0_abs + i_loc
    valid = np.isfinite(phi)
    if valid.any():
        sc = ax.scatter(
            y_abs[valid], phi[valid],
            c=coh[valid], cmap="viridis",
            s=8, vmin=0.0, vmax=1.0,
            edgecolor="none", zorder=2,
        )
        fig.colorbar(sc, ax=ax, location="right", shrink=0.85, label="coh")
    if np.isfinite(slope_pr) and np.isfinite(intercept):
        line_w = np.angle(np.exp(1j * (slope_pr * i_loc + intercept)))
        if line_w.size > 1:
            jumps = np.where(np.abs(np.diff(line_w)) > np.pi)[0]
            if jumps.size:
                line_w = line_w.copy()
                line_w[jumps] = np.nan
        ax.plot(y_abs, line_w, color="red", lw=1.0, zorder=3)
    ax.set_ylim(-np.pi, np.pi)
    ax.axhline(0.0, color="0.5", lw=0.5, ls="--", zorder=1)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("absolute azimuth row")
    ax.set_ylabel(r"$\Delta\varphi$  [rad]")
    ax.set_title(title, fontsize=8)


if __name__ == "__main__":
    main()
