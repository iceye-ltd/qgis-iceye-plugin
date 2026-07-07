"""Diagnostic: sweep `window_length` for estimate_phase_slope() on the
TARGET box only.

Reconstructs the exact pre-processing of shear_averaging.main()
(`apply_range_window` -> `degrade_range_resolution_range_sum` ->
CoV gate -> `mask_full`), applies `d_phase *= mask_full[:-1, :]` so
that `estimate_phase_slope` actually sees zero-bounded runs, then
runs the estimator at window_length ∈ {50, 500, 1000, 2000} and
plots:

  - amp_raw (range-degraded) inside the target box
  - mask_full inside the target box (so we can see runs)
  - one slope_map crop per window_length (rad/row, signed)
  - per-column |slope · n_run| over the target box, with the
    motion-filter threshold (1 rad) drawn for reference

This is a READ-ONLY diagnostic; nothing in shear_averaging.py changes.

Usage:
  python scripts/diag_slope_map_target.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shear_averaging import (
    apply_range_window,
    degrade_range_resolution_range_sum,
    compute_subapertures,
    estimate_phase_slope,
)


PATCH = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches/"
    "data_20260617_143542_433755.npy"
)

# Target box (y_top, y_bot, x_left, x_right) in s_degraded coords.
Y0, Y1 = 3571, 5571
X0, X1 = 109, 123

N_SUBAPERTURE = 8
COV_TH_MULT = 1.5
WINDOW_LENGTHS = [50, 500, 1000, 2000]
SAVE_PNG = Path(__file__).resolve().parent.parent / "target_slope_map_sweep.png"


def main() -> None:
    print(f"Loading {PATCH.name} ...")
    s = np.load(PATCH)
    print(f"  shape={s.shape}, dtype={s.dtype}")

    range_spacing = 0.5
    min_size_of_target = 3
    n_range_looks = int(min_size_of_target / range_spacing)  # 6

    s_windowed = apply_range_window(s, sll_db=55.0, nbar=8)
    s_degraded, d_phase = degrade_range_resolution_range_sum(
        s_windowed, n_range_looks,
    )
    print(f"  s_degraded {s_degraded.shape}  d_phase {d_phase.shape}")

    subs = compute_subapertures(s_degraded, N_SUBAPERTURE)
    sub_mean = subs.mean(axis=0)
    sub_var = subs.var(axis=0)
    del subs
    cov_sq = sub_var / (sub_mean ** 2 + 1e-12)
    th = COV_TH_MULT * np.median(cov_sq)
    mask_dec = (cov_sq > th).astype(np.float32)

    mask_full = np.repeat(mask_dec, N_SUBAPERTURE, axis=0)
    pad = s_degraded.shape[0] - mask_full.shape[0]
    if pad > 0:
        mask_full = np.vstack(
            [mask_full, np.repeat(mask_full[-1:], pad, axis=0)]
        )
    assert mask_full.shape == s_degraded.shape
    print(
        f"  mask_full: kept "
        f"{mask_full.sum():.0f}/{mask_full.size} "
        f"({100 * mask_full.sum() / mask_full.size:.2f}%)"
    )

    amp_raw = np.abs(s_degraded).astype(np.float32)

    # THE step that's currently commented out in shear_averaging.py:
    # apply the CoV mask onto d_phase so the column runs of non-zero
    # samples line up with the detection mask.
    d_phase_m = d_phase * mask_full[:-1, :]

    # Phasor-LS weight: |s[:-1]| * |s[1:]|. Zero whenever either
    # neighbour is masked, so masked samples drop out automatically.
    s_masked = s_degraded * mask_full
    weights = (
        np.abs(s_masked[:-1]) * np.abs(s_masked[1:])
    ).astype(np.float32)

    # Sweep window_length and compute slope_map for each value.
    slope_maps: dict[int, np.ndarray] = {}
    for wl in WINDOW_LENGTHS:
        print(f"\n-- estimate_phase_slope(window_length={wl}) --")
        slope_maps[wl] = estimate_phase_slope(
            d_phase_m, window_length=wl, weights=weights, n_iter_gn=2,
        )

    # Crop everything to the target box for plotting.
    def crop(a: np.ndarray) -> np.ndarray:
        return a[Y0:Y1, X0:X1]

    amp_crop = crop(amp_raw)
    mask_crop = crop(mask_full)
    slope_crops = {wl: crop(sm) for wl, sm in slope_maps.items()}

    # Per-column |slope·n_run| inside the target box. For each column
    # in the cropped slope_map, find the dominant run (longest
    # contiguous non-zero stretch) and report its slope · its length.
    def per_col_slope_n(slope_col: np.ndarray) -> tuple[float, int]:
        """Return (slope·n_rows, run_length) of the longest non-zero
        run in this column slice."""
        nz = slope_col != 0
        if not nz.any():
            return 0.0, 0
        change = np.flatnonzero(np.diff(nz.astype(np.int8))) + 1
        edges = np.concatenate([[0], change, [len(slope_col)]])
        best_len = 0
        best_slope = 0.0
        for s, e in zip(edges[:-1], edges[1:]):
            if not nz[s]:
                continue
            L = e - s
            if L > best_len:
                best_len = L
                best_slope = float(slope_col[s])
        return best_slope * best_len, best_len

    per_col_sn = {wl: [] for wl in WINDOW_LENGTHS}
    per_col_runlen = {wl: [] for wl in WINDOW_LENGTHS}
    for wl in WINDOW_LENGTHS:
        for c in range(slope_crops[wl].shape[1]):
            sn, L = per_col_slope_n(slope_crops[wl][:, c])
            per_col_sn[wl].append(sn)
            per_col_runlen[wl].append(L)

    # ---- Figure: top row 2 panels (amp, mask), middle row 4 panels
    #      (slope_map per wl), bottom row 1 panel (per-col |slope·n|).
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 4)
    ax_amp = fig.add_subplot(gs[0, 0:2])
    ax_mask = fig.add_subplot(gs[0, 2:4], sharex=ax_amp, sharey=ax_amp)
    ax_amp.imshow(
        amp_crop, cmap="gray",
        extent=[X0, X1, Y1, Y0], aspect="auto",
        vmax=np.percentile(amp_crop, 99),
    )
    ax_amp.set_title("amp_raw inside target box")
    ax_amp.set_xlabel("range col")
    ax_amp.set_ylabel("az row")
    ax_mask.imshow(
        mask_crop, cmap="gray",
        extent=[X0, X1, Y1, Y0], aspect="auto", vmin=0, vmax=1,
    )
    ax_mask.set_title(
        f"mask_full = 1 (CoV-kept)\n{int(mask_crop.sum())}/{mask_crop.size} px "
        f"({100*mask_crop.mean():.1f}%)"
    )
    ax_mask.set_xlabel("range col")

    sm_vmax = max(
        np.max(np.abs(slope_crops[wl])) for wl in WINDOW_LENGTHS
    ) or 1e-3
    for k, wl in enumerate(WINDOW_LENGTHS):
        ax = fig.add_subplot(
            gs[1, k], sharex=ax_amp, sharey=ax_amp,
        )
        im = ax.imshow(
            slope_crops[wl], cmap="RdBu_r",
            vmin=-sm_vmax, vmax=sm_vmax,
            extent=[X0, X1, Y1, Y0], aspect="auto",
        )
        nz_frac = (slope_crops[wl] != 0).mean()
        ax.set_title(
            f"window_length={wl}\nmin_len={wl//2}, "
            f"nz frac={100*nz_frac:.1f}%"
        )
        ax.set_xlabel("range col")
        if k == 0:
            ax.set_ylabel("az row")
        fig.colorbar(
            im, ax=ax, shrink=0.8, label="slope (rad/row)",
        )

    ax_bottom = fig.add_subplot(gs[2, :])
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(WINDOW_LENGTHS))
    x_idx = np.arange(slope_crops[WINDOW_LENGTHS[0]].shape[1])
    for off, wl in zip(offsets, WINDOW_LENGTHS):
        ax_bottom.bar(
            x_idx + off, np.abs(per_col_sn[wl]),
            width=width, label=f"wl={wl}",
        )
    ax_bottom.axhline(1.0, color="red", lw=1.0, ls="--",
                      label="motion gate (1 rad)")
    ax_bottom.set_xticks(x_idx)
    ax_bottom.set_xticklabels(
        [str(X0 + c) for c in x_idx],
        fontsize=8,
    )
    ax_bottom.set_xlabel("range col (absolute)")
    ax_bottom.set_ylabel("|slope · run_length|  (rad)")
    ax_bottom.set_title(
        "Per-column dominant-run |slope·n| (the per-column motion score)"
    )
    ax_bottom.legend(loc="upper right", ncols=5)
    ax_bottom.grid(alpha=0.3)

    fig.suptitle(
        f"slope_map sweep on target box  y={Y0}..{Y1}  x={X0}..{X1}  "
        f"(N_subaperture={N_SUBAPERTURE}, cov_mult={COV_TH_MULT})",
        fontsize=12,
    )
    fig.savefig(SAVE_PNG, dpi=110)
    print(f"\nSaved figure → {SAVE_PNG}")

    # Also print a tidy text table of per-column |slope·n| for each wl.
    print("\nPer-column |slope·n| (rad)  inside target box "
          f"(cols x={X0}..{X1}):")
    header = "  col |  " + "  ".join(
        f"wl={wl:5d}|nrun" for wl in WINDOW_LENGTHS
    )
    print(header)
    print("-" * len(header))
    for c in range(slope_crops[WINDOW_LENGTHS[0]].shape[1]):
        row = f"  {X0 + c:3d} |  "
        for wl in WINDOW_LENGTHS:
            sn = per_col_sn[wl][c]
            ln = per_col_runlen[wl][c]
            row += f"{sn:+7.2f}|{ln:4d}  "
        print(row)


if __name__ == "__main__":
    main()
