"""Test multiple refocus formulations on T1 to find the bug.

T1's d_phase is unambiguously a linear ramp (|slope·n|=4.13 rad, residual
0.57 rad). A correct centred-QPE refocus should sharpen the chip.
The script's current implementation makes the contrast WORSE by 2.8 dB.

This script applies four refocus variants to T1's chip and reports the
contrast before/after for each. The winner tells us where the bug is.
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
    _normalized_variance,
)

PATCH = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_141944_569017.npy"
)
OUT_DIR = HERE / "missing_targets_out"

# T1 parameters from the previous diagnostic.
Y_LO, Y_HI, X_LO, X_HI = 10500, 12500, 322, 330

# Pipeline parameters matching the user's run.
RANGE_LOOKS = 6
RANGE_SLL_DB = 55.0
RANGE_TAYLOR_NBAR = 8
N_SUBAPERTURE = 8
COV_TH_MULT = 1.5
SEED_DENSITY = 0.5
SEED_WINDOW = (100, 3)
GROW_DENSITY = 0.25
MIN_ROW_COHERENCE = 0.5


def refocus_current_freq_plus(chip: np.ndarray, slope: float) -> np.ndarray:
    """The script's current implementation: exp(+j·½·slope·m²) in frequency."""
    N = chip.shape[0]
    m = np.arange(N, dtype=np.float64) - (N // 2)
    phi = 0.5 * slope * m * m
    F = np.fft.fft(np.fft.ifftshift(chip, axes=0), axis=0)
    F = F * np.exp(1j * phi)[:, None]
    return np.fft.fftshift(np.fft.ifft(F, axis=0), axes=0)


def refocus_freq_minus(chip: np.ndarray, slope: float) -> np.ndarray:
    """Sign-flipped: exp(-j·½·slope·m²) in frequency (matches docstring)."""
    N = chip.shape[0]
    m = np.arange(N, dtype=np.float64) - (N // 2)
    phi = 0.5 * slope * m * m
    F = np.fft.fft(np.fft.ifftshift(chip, axes=0), axis=0)
    F = F * np.exp(-1j * phi)[:, None]
    return np.fft.fftshift(np.fft.ifft(F, axis=0), axes=0)


def refocus_spatial(chip: np.ndarray, slope: float) -> np.ndarray:
    """The textbook spatial-domain QPE removal: chip · exp(-j·½·slope·n²).

    `d_phase = arg(s[n+1]·conj(s[n]))` measures the SPATIAL azimuth phase
    gradient. Integrating gives the SPATIAL QPE φ(n)=½·slope·n². To remove
    a chirp in the spatial domain you multiply by the conjugate chirp in
    the SPATIAL domain — full stop.
    """
    N = chip.shape[0]
    m = np.arange(N, dtype=np.float64) - (N // 2)
    phi = 0.5 * slope * m * m
    return chip * np.exp(-1j * phi)[:, None]


def refocus_freq_dual(chip: np.ndarray, slope: float) -> np.ndarray:
    """Fourier-dual chirp: exp(+j·k²/(2·slope)) on the (fft-shifted)
    spectrum. By stationary phase, FT[exp(-j·½·a·n²)] ∝ exp(+j·f²/(2a)),
    so the freq-domain equivalent of the spatial fix above uses rate 1/a,
    not a. We apply it in the natural centred-frequency convention.
    """
    N = chip.shape[0]
    if abs(slope) < 1e-12:
        return chip.copy()
    f = np.fft.fftshift(np.fft.fftfreq(N) * 2 * np.pi)   # centred radians/sample
    phi_dual = (f * f) / (2.0 * slope)
    # ifftshift after building phi (so it matches fft bin order).
    phi_dual_unshifted = np.fft.ifftshift(phi_dual)
    F = np.fft.fft(chip, axis=0)
    F = F * np.exp(1j * phi_dual_unshifted)[:, None]
    return np.fft.ifft(F, axis=0)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s = np.load(PATCH)
    s_win = apply_range_window(s, sll_db=RANGE_SLL_DB, nbar=RANGE_TAYLOR_NBAR)
    s_deg = degrade_range_resolution_range_sum(s_win, RANGE_LOOKS)
    del s, s_win
    H, W = s_deg.shape

    sub_un = compute_subapertures(s_deg, N_SUBAPERTURE)
    cov_sq = sub_un.var(axis=0) / (sub_un.mean(axis=0) ** 2 + 1e-12)
    th = COV_TH_MULT * float(np.median(cov_sq))
    mask_dec = (cov_sq > th).astype(np.float32)
    mask_full = np.repeat(mask_dec, N_SUBAPERTURE, axis=0)
    pad = H - mask_full.shape[0]
    if pad > 0:
        mask_full = np.vstack([mask_full, np.repeat(mask_full[-1:], pad, axis=0)])
    del sub_un

    amp = np.abs(s_deg).astype(np.float32)
    box_area = SEED_WINDOW[0] * SEED_WINDOW[1]
    det_count = uniform_filter(mask_full.astype(np.float32), size=SEED_WINDOW,
                               mode="constant") * box_area
    amp_max = maximum_filter(amp, size=SEED_WINDOW, mode="constant")
    seeds = (mask_full == 1) & (det_count > SEED_DENSITY * box_area) & (amp == amp_max)
    seeds_yx = np.argwhere(seeds)

    inside = ((seeds_yx[:, 0] >= Y_LO) & (seeds_yx[:, 0] < Y_HI)
              & (seeds_yx[:, 1] >= X_LO) & (seeds_yx[:, 1] < X_HI))
    seeds_in_roi = seeds_yx[inside]
    grown = grow_and_recenter_boxes(
        seeds_in_roi, mask=mask_full, amp=amp,
        initial_hw=(100, 3), az_step=5, rg_step=1,
        density_threshold=GROW_DENSITY, max_h=2000, max_w=43,
    )
    scores = np.array([
        float(amp[max(0, y - h // 2):min(H, y - h // 2 + h),
                  max(0, x - w // 2):min(W, x - w // 2 + w)].max())
        for (y, x, h, w) in grown
    ], dtype=np.float32)
    best = int(np.argmax(scores))
    y_c, x_c, h, w = (int(v) for v in grown[best])
    print(f"T1 box: y_c={y_c}, x_c={x_c}, h={h}, w={w}")

    phi_arr, coh_arr, sl_arr, ic_arr, y0_arr, n_arr, res_arr, _ = (
        compute_box_phase_estimates(
            np.array([[y_c, x_c, h, w]], dtype=np.int64),
            s_deg,
            min_row_coherence=MIN_ROW_COHERENCE,
            inlier_tol_rad=0.5,
        )
    )
    n_k = int(min(n_arr[0], phi_arr.shape[1]))
    y0 = int(y0_arr[0])
    slope = float(sl_arr[0])
    print(f"phase fit: slope={slope:+.4e} rad/row, "
          f"|slope·n|={abs(slope * n_k):.2f} rad, residual={float(res_arr[0]):.2f}")

    x0 = max(0, x_c - w // 2); x1 = min(W, x0 + w)
    y1 = y0 + n_k
    chip = s_deg[y0:y1, x0:x1] * mask_full[y0:y1, x0:x1]
    c_before = _normalized_variance(chip)

    variants = {
        "current (freq +sign, script)": refocus_current_freq_plus,
        "freq +sign DOC fix (-sign)  ": refocus_freq_minus,
        "spatial -sign  (proposed fix)": refocus_spatial,
        "freq Fourier-dual (1/slope) ": refocus_freq_dual,
    }

    print(f"\nContrast (std(|I|²)/mean(|I|²)) before = {c_before:.4f}\n")
    print(f"{'variant':32s}  {'after':>8s}   {'gain (dB)':>10s}")
    results = {}
    for name, fn in variants.items():
        corrected = fn(chip, slope)
        c_after = _normalized_variance(corrected)
        gain_db = (20.0 * np.log10(c_after / c_before)
                   if (c_before > 0 and c_after > 0) else float("nan"))
        results[name] = (corrected, c_after, gain_db)
        print(f"{name}   {c_after:8.4f}   {gain_db:+10.2f}")

    # Side-by-side panel: |I| before vs each refocus variant after.
    fig, axes = plt.subplots(1, 5, figsize=(20, 7), constrained_layout=True)
    panels = [("before", chip, c_before, 0.0)]
    for name, (corr, c_after, gain_db) in results.items():
        panels.append((name, corr, c_after, gain_db))
    for ax, (ttl, img, c_val, gain) in zip(axes, panels):
        a = np.abs(img)
        if a.size:
            vmin = float(np.nanpercentile(a, 1.0))
            vmax = float(np.nanpercentile(a, 99.0))
            if vmax <= vmin:
                vmax = vmin + 1.0
            ax.imshow(a, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xlabel("range"); ax.set_ylabel("azimuth")
        sub = (f"\nC={c_val:.3f}  ({gain:+.2f} dB)"
               if ttl != "before" else f"\nC={c_val:.3f}")
        ax.set_title(ttl + sub, fontsize=9)
    fig.suptitle(
        f"T1 refocus variants — y={y_c}, x={x_c}, h={h}, w={w}, "
        f"slope={slope:+.4e} rad/row, |slope·n|={abs(slope*n_k):.2f} rad",
        fontsize=11,
    )
    out_path = OUT_DIR / "T1_refocus_variants.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
