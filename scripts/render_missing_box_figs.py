"""Render `box_X_dphase.png` and `box_X_refocus.png` for the 5
low-backscatter targets that the shear_averaging pipeline misses.

For each labelled ROI we:

  1. Run the pipeline's early stages (Taylor window, range degradation,
     sub-aperture stack, CoV gate, seed peaks, grow_and_recenter_boxes)
     once.
  2. Pick the box the script's IoU + centre-distance NMS would have
     kept inside the ROI — the single highest-score grown box whose
     centre lies inside the ROI rectangle (score = max |s_degraded|
     inside the grown rectangle; identical to the script's NMS score).
  3. Re-fit the box's azimuth phase ramp with the same coh-weighted
     wrap-aware grid search the script uses (compute_box_phase_estimates).
  4. Apply the same centred-QPE refocus (`_refocus_box_chip`) and
     measure the normalised-variance contrast before / after.
  5. Plot exactly the same two figures the script writes for each
     surviving box:
       box_<LABEL>_dphase.png  — per-row phi scatter coloured by coh
                                 + wrapped grid-fit line (red)
       box_<LABEL>_refocus.png — |chip| before / |chip| after the
                                 centred-QPE refocus (1×2 panel)

Output goes to ``scripts/missing_targets_out/`` so it stays out of the
existing ``*_per_box/`` folders.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, maximum_filter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shear_averaging import (  # noqa: E402
    apply_range_window,
    degrade_range_resolution_range_sum,
    compute_subapertures,
    grow_and_recenter_boxes,
    compute_box_phase_estimates,
    _refocus_box_chip,
    _normalized_variance,
    _plot_refocus_box_1x2,
)


def _refocus_box_chip_fixed(
    chip: np.ndarray,
    slope_rad_per_row: float,
    intercept_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Corrected freq-domain centred-QPE refocus.

    Two-step model. dφ = slope·n + intercept integrates to
    φ_full(n) = ½·slope·n² + intercept·n + const.

      1. Linear part (intercept·n) ⇔ Doppler-centroid shift of the
         spectrum. Apply as a spatial pre-multiplication
         exp(-j·intercept·n) so the focused row lands at its true
         azimuth position.
      2. Quadratic part (½·slope·n²) ⇔ a chirp of rate 1/slope on the
         spectrum (Fourier dual of the spatial chirp of rate slope).
         Apply the conjugate exp(+j·f²/(2·slope)) on the centred
         spectrum, with f in rad/sample.
    """
    N = chip.shape[0]
    if N < 2 or abs(slope_rad_per_row) < 1e-12:
        return chip.copy(), np.zeros(N, dtype=np.float64)
    # Step 1 — linear spatial pre-mult (= spectrum shift back to centroid).
    n = np.arange(N, dtype=np.float64) - (N // 2)
    chip_shifted = chip * np.exp(-1j * float(intercept_rad) * n)[:, None]
    # Step 2 — Fourier-dual quadratic in centred frequency.
    f = np.fft.fftshift(np.fft.fftfreq(N) * 2.0 * np.pi)
    phi_freq = (f * f) / (2.0 * float(slope_rad_per_row)) + f*n
    phi_unshifted = np.fft.ifftshift(phi_freq)
    F = np.fft.fft(chip_shifted, axis=0)
    F = F * np.exp(1j * phi_unshifted)[:, None]
    corrected = np.fft.ifft(F, axis=0)
    # Also return the spatial QPE for downstream inspection.
    phi_spatial = 0.5 * float(slope_rad_per_row) * n * n
    return corrected, phi_spatial


PATCH = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_141944_569017.npy"
)
OUT_DIR = HERE / "missing_targets_out"

# (label, y_lo, y_hi, x_lo, x_hi) in s_degraded coordinates.
ROIS = [
    ("T1", 10500, 12500, 322, 330),
    ("T2", 24800, 26400, 360, 362),
    ("T3", 33000, 35000, 335, 345),
    ("T4", 38000, 40000, 326, 334),
    ("T5", 42500, 44250, 272, 278),
]

# Pipeline parameters matching the user's run.
RANGE_LOOKS = 6
RANGE_SLL_DB = 55.0
RANGE_TAYLOR_NBAR = 8
N_SUBAPERTURE = 8
COV_TH_MULT = 1.5
SEED_DENSITY = 0.5
SEED_WINDOW = (100, 3)
GROW_DENSITY = 0.25
N_TARGET_AZIMUTH_WIDTH = 100
N_TARGET_RANGE_LENGTH = 3
MAX_SIZE_AZIMUTH_BIN = 2000
MAX_SIZE_RANGE_BIN = 130 // 3
MIN_ROW_COHERENCE = 0.5
INLIER_TOL_RAD = 0.5


def plot_dphase_like_script(
    *,
    label: str,
    y_c: int,
    x_c: int,
    h: int,
    w: int,
    y0: int,
    n_rows: int,
    phi: np.ndarray,
    coh: np.ndarray,
    slope_rad_per_row: float,
    intercept_rad: float,
    slope_total_rad: float,
    residual_rad: float,
    n_inliers: int,
    out_path: Path,
) -> None:
    """Reproduces the script's `box_NNN_dphase.png` exactly."""
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    i_loc = np.arange(n_rows, dtype=np.float64)
    y_abs = y0 + i_loc
    valid = np.isfinite(phi)
    if valid.any():
        sc = ax.scatter(
            y_abs[valid], phi[valid],
            c=coh[valid], cmap="viridis",
            s=10, vmin=0.0, vmax=1.0,
            edgecolor="none", zorder=2,
        )
        fig.colorbar(sc, ax=ax, location="right", shrink=0.85, label="coh")

    used = (np.isfinite(phi) & np.isfinite(coh)
            & (coh > MIN_ROW_COHERENCE))
    n_used = int(used.sum())

    if np.isfinite(slope_rad_per_row) and np.isfinite(intercept_rad):
        line_wrapped = np.angle(
            np.exp(1j * (slope_rad_per_row * i_loc + intercept_rad))
        )
        if line_wrapped.size > 1:
            d_line = np.abs(np.diff(line_wrapped))
            jumps = np.where(d_line > np.pi)[0]
            if jumps.size:
                line_wrapped = line_wrapped.copy()
                line_wrapped[jumps] = np.nan
        ax.plot(y_abs, line_wrapped, color="red", lw=1.1, zorder=3)
    ax.set_ylim(-np.pi, np.pi)
    ax.axhline(0.0, color="0.5", lw=0.5, ls="--", zorder=1)

    coh_pct = (100.0 * n_used / n_rows) if n_rows > 0 else 0.0
    in_pct = (100.0 * n_inliers / n_rows) if n_rows > 0 else 0.0
    res_str = f"  res={residual_rad:.2f} rad" if np.isfinite(residual_rad) else ""
    coh_str = (f"  coh>{MIN_ROW_COHERENCE:g}: "
               f"{n_used}/{n_rows} ({coh_pct:.1f}%)")
    inlier_str = (f"  inliers(d≤{INLIER_TOL_RAD:g}): "
                  f"{n_inliers}/{n_rows} ({in_pct:.1f}%)")
    ax.set_title(
        f"{label}  y={y_c} x={x_c}\n"
        f"slope={np.degrees(slope_rad_per_row):+.2f}°/row, "
        f"slope·n={slope_total_rad:+.2f} rad"
        f"{res_str}{coh_str}{inlier_str}",
        fontsize=9,
    )
    ax.set_xlabel("absolute az row")
    ax.set_ylabel(r"$\Delta\varphi$  [rad]")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {PATCH.name} ...")
    s = np.load(PATCH)
    print(f"  SLC shape={s.shape}")

    print("Range Taylor window + range degradation ...")
    s_win = apply_range_window(s, sll_db=RANGE_SLL_DB, nbar=RANGE_TAYLOR_NBAR)
    s_deg = degrade_range_resolution_range_sum(s_win, RANGE_LOOKS)
    # d_phase used only by the debug plotting helpers below; the shear_-
    # averaging main pipeline computes d_phase per-box inside the phase
    # filters instead.
    d_phase = np.angle(s_deg[1:] * np.conj(s_deg[:-1]))
    del s, s_win
    H, W = s_deg.shape
    print(f"  s_degraded shape={s_deg.shape}")

    print("Sub-aperture CoV mask ...")
    sub_un = compute_subapertures(s_deg, N_SUBAPERTURE)
    sub_mean = sub_un.mean(axis=0)
    sub_var = sub_un.var(axis=0)
    cov_sq = sub_var / (sub_mean ** 2 + 1e-12)
    th = COV_TH_MULT * float(np.median(cov_sq))
    mask_dec = (cov_sq > th).astype(np.float32)
    mask_full = np.repeat(mask_dec, N_SUBAPERTURE, axis=0)
    pad = H - mask_full.shape[0]
    if pad > 0:
        mask_full = np.vstack([mask_full, np.repeat(mask_full[-1:], pad, axis=0)])
    assert mask_full.shape == s_deg.shape
    print(f"  th={th:.3g}, kept {int(mask_full.sum())}/{mask_full.size} pixels")

    print("Seed peaks ...")
    amp = np.abs(s_deg).astype(np.float32)
    box_area = SEED_WINDOW[0] * SEED_WINDOW[1]
    det_count = uniform_filter(
        mask_full.astype(np.float32), size=SEED_WINDOW, mode="constant"
    ) * box_area
    amp_max = maximum_filter(amp, size=SEED_WINDOW, mode="constant")
    seeds = (mask_full == 1) & (det_count > SEED_DENSITY * box_area) & (amp == amp_max)
    seeds_yx = np.argwhere(seeds)
    seed_amp = amp[seeds_yx[:, 0], seeds_yx[:, 1]]
    print(f"  {len(seeds_yx)} seed peaks in scene")

    amp_raw = amp  # script's amp_raw = |s_deg| BEFORE masking; identical here

    for label, y_lo, y_hi, x_lo, x_hi in ROIS:
        print(f"\n[{label}] ROI s_deg[{y_lo}:{y_hi}, {x_lo}:{x_hi}]")
        # Seeds inside the ROI.
        inside = (
            (seeds_yx[:, 0] >= y_lo) & (seeds_yx[:, 0] < y_hi)
            & (seeds_yx[:, 1] >= x_lo) & (seeds_yx[:, 1] < x_hi)
        )
        idx = np.flatnonzero(inside)
        if idx.size == 0:
            print(f"  no seed in ROI; skipping (target dies at the seed stage).")
            continue

        # Grow each ROI seed.
        seeds_in_roi = seeds_yx[idx]
        grown = grow_and_recenter_boxes(
            seeds_in_roi, mask=mask_full, amp=amp,
            initial_hw=(N_TARGET_AZIMUTH_WIDTH, N_TARGET_RANGE_LENGTH),
            az_step=5, rg_step=1,
            density_threshold=GROW_DENSITY,
            max_h=MAX_SIZE_AZIMUTH_BIN, max_w=MAX_SIZE_RANGE_BIN,
        )
        # Score = MAX |s_deg| inside the grown rectangle (same as the
        # NMS scoring in the script). Pick the brightest box whose
        # centre lies inside the ROI.
        scores = np.zeros(len(grown), dtype=np.float32)
        for i, (y, x, h, w) in enumerate(grown):
            yl = max(int(y) - int(h) // 2, 0)
            yh = min(yl + int(h), H)
            xl = max(int(x) - int(w) // 2, 0)
            xh = min(xl + int(w), W)
            sub = amp_raw[yl:yh, xl:xh]
            scores[i] = float(sub.max()) if sub.size else 0.0

        in_roi_box = (
            (grown[:, 0] >= y_lo) & (grown[:, 0] < y_hi)
            & (grown[:, 1] >= x_lo) & (grown[:, 1] < x_hi)
        )
        keep_pool = np.flatnonzero(in_roi_box) if in_roi_box.any() else np.arange(len(grown))
        best = keep_pool[int(np.argmax(scores[keep_pool]))]
        y_c, x_c, h, w = (int(v) for v in grown[best])
        print(f"  picked grown box (NMS-winner): y_c={y_c}, x_c={x_c}, "
              f"h={h}, w={w}, score={float(scores[best]):.2f}")

        # Per-box phase estimate + linear fit (same numerics as script).
        phi_arr, coh_arr, sl_arr, ic_arr, y0_arr, n_arr, res_arr, in_arr = (
            compute_box_phase_estimates(
                np.array([[y_c, x_c, h, w]], dtype=np.int64),
                s_deg,
                min_row_coherence=MIN_ROW_COHERENCE,
                inlier_tol_rad=INLIER_TOL_RAD,
            )
        )
        n_k = int(min(n_arr[0], phi_arr.shape[1]))
        y0 = int(y0_arr[0])
        slope_pr = float(sl_arr[0])
        intercept = float(ic_arr[0])
        residual = float(res_arr[0])
        n_inliers = int(in_arr[0])
        slope_total = slope_pr * n_k if np.isfinite(slope_pr) else float("nan")
        print(f"  phase fit: slope={slope_pr:+.4e} rad/row, "
              f"|slope·n|={abs(slope_total):.2f} rad, "
              f"residual={residual:.2f} rad, n_rows={n_k}")

        # --- box_LABEL_dphase.png -------------------------------------
        out_dphase = OUT_DIR / f"box_{label}_dphase.png"
        plot_dphase_like_script(
            label=label,
            y_c=y_c, x_c=x_c, h=h, w=w,
            y0=y0, n_rows=n_k,
            phi=phi_arr[0, :n_k], coh=coh_arr[0, :n_k],
            slope_rad_per_row=slope_pr,
            intercept_rad=intercept,
            slope_total_rad=slope_total,
            residual_rad=residual,
            n_inliers=n_inliers,
            out_path=out_dphase,
        )
        print(f"  → {out_dphase}")

        # --- box_LABEL_refocus.png ------------------------------------
        # Refocus uses the masked s_degraded chip (same as the pipeline).
        x0_c = max(0, x_c - w // 2)
        x1_c = min(W, x0_c + w)
        y1 = y0 + n_k
        chip_unmasked = s_deg[y0:y1, x0_c:x1_c]
        chip = chip_unmasked * mask_full[y0:y1, x0_c:x1_c]
        corrected, _ = _refocus_box_chip(chip, slope_pr, intercept)

        c_b = _normalized_variance(chip)
        c_a = _normalized_variance(corrected)
        gain_db = (20.0 * np.log10(c_a / c_b)
                   if (c_b > 0 and c_a > 0) else float("nan"))

        def _peak_to_mean(img: np.ndarray) -> float:
            prof = (np.abs(img) ** 2).sum(axis=1)
            mu = float(prof.mean())
            return float(prof.max() / mu) if mu > 0 else float("nan")

        ptm_b = _peak_to_mean(chip)
        ptm_a = _peak_to_mean(corrected)

        print(f"  refocus  C: {c_b:.3f} → {c_a:.3f} "
              f"({gain_db:+.2f} dB)  peak/mean(az): "
              f"{ptm_b:.2f} → {ptm_a:.2f}")

        # 2x2 panel: |I| before/after on top, azimuth profiles below.
        fig, axes = plt.subplots(2, 2, figsize=(11, 9.5),
                                 constrained_layout=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.0]})
        fig.suptitle(
            f"{label}  box (y={y_c}, x={x_c}, h={h}, w={w})  "
            f"slope = {slope_pr:+.4e} rad/row  "
            f"intercept = {intercept:+.3f} rad  "
            f"|slope·n| = {abs(slope_total):.2f} rad",
            fontsize=10,
        )
        panels = (
            ("|I|  before",
             chip, c_b, np.nan, ptm_b),
            (f"|I|  after refocus\n"
             f"C={c_a:.3f} ({gain_db:+.2f} dB)   peak/mean={ptm_a:.1f}",
             corrected, c_a, gain_db, ptm_a),
        )
        for col, (ttl, img, *_) in enumerate(panels):
            ax = axes[0, col]
            a = np.abs(img)
            if a.size:
                vmin = float(np.nanpercentile(a, 1.0))
                vmax = float(np.nanpercentile(a, 99.0))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                ax.imshow(a, aspect="auto", cmap="gray",
                          vmin=vmin, vmax=vmax)
            ax.set_xlabel("range")
            ax.set_ylabel("azimuth")
            ax.set_title(ttl, fontsize=9)

            # Azimuth-integrated intensity profile below.
            ax_p = axes[1, col]
            prof = (a ** 2).sum(axis=1)
            ax_p.plot(np.arange(prof.size), prof, lw=0.7)
            mu = float(prof.mean())
            peak = float(prof.max())
            ax_p.axhline(mu, color="0.6", ls="--", lw=0.6, label=f"mean = {mu:.0f}")
            ax_p.set_xlim(0, prof.size)
            ax_p.set_xlabel("azimuth row (within box)")
            ax_p.set_ylabel(r"$\sum_{\rm rg}|I|^2$")
            ax_p.set_title(f"az profile  —  peak {peak:.0f}, peak/mean = {peak/mu:.1f}",
                           fontsize=9)
            ax_p.legend(loc="upper right", fontsize=8)
            ax_p.grid(True, alpha=0.3)

        out_refocus = OUT_DIR / f"box_{label}_refocus_fixed.png"
        fig.savefig(out_refocus, dpi=130)
        plt.close(fig)
        print(f"  → {out_refocus}")


if __name__ == "__main__":
    main()
