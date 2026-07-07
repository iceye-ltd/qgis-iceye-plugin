"""Apply the corrected freq-domain refocus to T1, and also compare:

  • Quadratic-only (current model, but the CORRECT freq-domain formula:
    exp(+j·f²/(2·slope)) on the centred spectrum).
  • Quadratic + linear (full integrated line φ = ½·slope·n² + intercept·n).
  • Full integral of d_phase (cumsum of the actual measured d_phase,
    coh-weighted, no parametric model at all). This is the "integrate
    the phase derivative twice" route.

For each, show |I| before / after and also the azimuth-summed intensity
profile (the most honest focus indicator).
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
)

PATCH = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_141944_569017.npy"
)
OUT_DIR = HERE / "missing_targets_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)

Y_LO, Y_HI, X_LO, X_HI = 10500, 12500, 322, 330


def freq_quadratic(chip: np.ndarray, slope: float) -> np.ndarray:
    """Correct freq-domain QPE: exp(+j·f²/(2·slope)) on the spectrum.

    f = centred azimuth-frequency (rad/sample), built with fftshift
    so DC sits at array index N//2. We ifftshift it before multiplying
    so it lands on the FFT bin order produced by fft(chip).
    """
    N = chip.shape[0]
    if abs(slope) < 1e-12:
        return chip.copy()
    f = np.fft.fftshift(np.fft.fftfreq(N) * 2 * np.pi)
    phi_freq = (f * f) / (2.0 * slope)
    phi_unshifted = np.fft.ifftshift(phi_freq)
    F = np.fft.fft(chip, axis=0)
    F = F * np.exp(1j * phi_unshifted)[:, None]
    return np.fft.ifft(F, axis=0)


def spatial_then_freq(chip: np.ndarray,
                      slope: float,
                      intercept: float) -> np.ndarray:
    """Remove ½·slope·n² + intercept·n.

    intercept·n in spatial = a Doppler-centroid SHIFT of the spectrum;
    multiplying chip by exp(-j·intercept·n) puts the spectrum back at
    its 'true' Doppler centre before we apply the freq-domain quadratic
    refocus.
    """
    N = chip.shape[0]
    n = np.arange(N) - N // 2
    shifted = chip * np.exp(-1j * float(intercept) * n)[:, None]
    return freq_quadratic(shifted, slope)


def cumsum_then_freq(chip: np.ndarray,
                     d_phase_box: np.ndarray,
                     weight: np.ndarray) -> np.ndarray:
    """Coherent integration of the per-row d_phase to get φ_full(n),
    apply exp(-j·φ_full(n)) in spatial (= shift spectrum to centroid),
    then perform an inverse refocus of any RESIDUAL chirp in the
    centroid-aligned spectrum.

    Steps:
      1. Per-row d_phase: take the coh-weighted circular mean across
         range, exactly as `compute_box_phase_estimates` does — gives
         dφ(i) for i = 0..L-2.
      2. φ(n) = cumsum(dφ) — the integrated spatial phase, no parametric
         model. Wraps in dφ already encode -pi..pi increments so the
         cumsum is naturally unwrapped (mod 2π drift accumulates, but
         that's a const after wrap → fine).
      3. Multiply chip rows by exp(-j·φ(n)) → demodulates the spatial
         azimuth phase. After this step the target's chirp is GONE in
         spatial, but the spectrum has been shifted to whatever centroid
         dφ traced; we don't yet know if a *residual* chirp survives,
         so we sweep one Newton-style adjustment by re-fitting a line
         to the residual d_phase and feeding its slope to `freq_quadratic`.
    """
    N, W = chip.shape
    L = d_phase_box.shape[0]                # L = N - 1
    # Coh-weighted per-row circular mean of d_phase (= same dφ(i) the
    # script fits a line to, just kept un-modelled here).
    z = (weight * np.exp(1j * d_phase_box)).sum(axis=1)
    dphi = np.angle(z)                       # (L,) ∈ (-π, π]
    # Cumulative integration: φ(n) for n=1..L, φ(0)=0.
    phi = np.concatenate([[0.0], np.cumsum(dphi)])
    # Re-centre so φ(N/2) = 0 (matches the rest of the codebase's
    # centred-line convention).
    phi -= phi[N // 2]
    # Demodulate spatial.
    chip2 = chip * np.exp(-1j * phi)[:, None]
    # Re-measure the residual d_phase on chip2 and fit a centred line;
    # whatever slope survives is the residual chirp to remove in freq.
    d_phase_res = np.angle(chip2[1:] * np.conj(chip2[:-1]))
    z_res = (weight * np.exp(1j * d_phase_res)).sum(axis=1)
    dphi_res = np.angle(z_res)
    # Simple least-squares slope on the residual (already nearly zero).
    n_idx = np.arange(L) - L / 2
    denom = (n_idx ** 2).sum()
    res_slope = float((n_idx * dphi_res).sum() / denom) if denom > 0 else 0.0
    return freq_quadratic(chip2, res_slope), phi


def main() -> None:
    s = np.load(PATCH)
    s_win = apply_range_window(s, sll_db=55.0, nbar=8)
    s_deg = degrade_range_resolution_range_sum(s_win, 6)
    del s, s_win
    H, W = s_deg.shape
    # d_phase is used locally for the cumsum/refocus experiments below;
    # the shear_averaging main pipeline computes d_phase per-box inside
    # the phase filters instead.
    d_phase = np.angle(s_deg[1:] * np.conj(s_deg[:-1]))

    # CoV mask (N=8 to match the user's run).
    N_sub = 8
    sub_un = compute_subapertures(s_deg, N_sub)
    cov_sq = sub_un.var(axis=0) / (sub_un.mean(axis=0) ** 2 + 1e-12)
    th = 1.5 * float(np.median(cov_sq))
    mask_dec = (cov_sq > th).astype(np.float32)
    mask_full = np.repeat(mask_dec, N_sub, axis=0)
    pad = H - mask_full.shape[0]
    if pad > 0:
        mask_full = np.vstack([mask_full, np.repeat(mask_full[-1:], pad, axis=0)])
    del sub_un

    amp = np.abs(s_deg).astype(np.float32)
    box_area = 100 * 3
    det_count = uniform_filter(mask_full.astype(np.float32), size=(100, 3),
                               mode="constant") * box_area
    amp_max = maximum_filter(amp, size=(100, 3), mode="constant")
    seeds = (mask_full == 1) & (det_count > 0.5 * box_area) & (amp == amp_max)
    seeds_yx = np.argwhere(seeds)
    in_roi = ((seeds_yx[:, 0] >= Y_LO) & (seeds_yx[:, 0] < Y_HI)
              & (seeds_yx[:, 1] >= X_LO) & (seeds_yx[:, 1] < X_HI))
    grown = grow_and_recenter_boxes(
        seeds_yx[in_roi], mask=mask_full, amp=amp,
        initial_hw=(100, 3), az_step=5, rg_step=1,
        density_threshold=0.25, max_h=2000, max_w=43,
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
            s_deg, min_row_coherence=0.5, inlier_tol_rad=0.5,
        )
    )
    n_k = int(min(n_arr[0], phi_arr.shape[1]))
    y0 = int(y0_arr[0])
    slope = float(sl_arr[0]); intercept = float(ic_arr[0])
    print(f"phase fit: slope={slope:+.4e} rad/row, intercept={intercept:+.4f} rad, "
          f"|slope·n|={abs(slope * n_k):.2f}, residual={float(res_arr[0]):.2f}")

    x0 = max(0, x_c - w // 2); x1 = min(W, x0 + w)
    y1 = y0 + n_k
    chip_complex = s_deg[y0:y1, x0:x1]
    chip = chip_complex * mask_full[y0:y1, x0:x1]   # masked, what the script uses

    weight = amp[y0:y1 - 1, x0:x1] * amp[y0 + 1:y1, x0:x1]
    d_phase_box = d_phase[y0:y1 - 1, x0:x1]

    variants = {}
    variants["before (no refocus)"] = chip
    variants["script (broken freq +sign)"] = (
        np.fft.fftshift(np.fft.ifft(
            np.fft.fft(np.fft.ifftshift(chip, axes=0), axis=0)
            * np.exp(1j * 0.5 * slope
                     * (np.arange(chip.shape[0]) - chip.shape[0] // 2) ** 2)[:, None],
            axis=0), axes=0)
    )
    variants["freq-dual quadratic only"] = freq_quadratic(chip, slope)
    variants["freq-dual + linear (intercept)"] = (
        spatial_then_freq(chip, slope, intercept)
    )
    cumsum_chip, _ = cumsum_then_freq(chip, d_phase_box, weight)
    variants["cumsum dφ + residual freq"] = cumsum_chip

    # Figure: rows = variants; left column = |I| image, right = az profile.
    fig, axes = plt.subplots(
        nrows=len(variants), ncols=2,
        figsize=(11, 3.2 * len(variants)), constrained_layout=True,
    )
    for r, (name, img) in enumerate(variants.items()):
        a = np.abs(img)
        prof = (a ** 2).sum(axis=1)
        ptm = float(prof.max() / max(prof.mean(), 1e-12))
        # contrast metric (the script's gate)
        intensity = a ** 2
        m = float(intensity.mean())
        c_metric = float(intensity.std() / m) if m > 0 else float("nan")

        ax_img, ax_prof = axes[r, 0], axes[r, 1]
        vmin = float(np.nanpercentile(a, 1.0))
        vmax = float(np.nanpercentile(a, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1.0
        ax_img.imshow(a, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax_img.set_xlabel("range"); ax_img.set_ylabel("azimuth")
        ax_img.set_title(f"{name}\nstd(|I|²)/mean(|I|²) = {c_metric:.3f}",
                         fontsize=9)

        ax_prof.plot(prof, np.arange(prof.size), lw=0.7)
        ax_prof.invert_yaxis()
        ax_prof.set_xlabel(r"$\sum_{\rm rg}|I|^2$")
        ax_prof.set_ylabel("azimuth")
        ax_prof.set_title(f"peak/mean = {ptm:.1f}", fontsize=9)
        ax_prof.grid(True, alpha=0.3)
        print(f"  {name:32s}  C={c_metric:5.3f}  peak/mean(az profile)={ptm:7.1f}")

    fig.suptitle(
        f"T1 refocus — y={y_c}, x={x_c}, h={h}, w={w}, "
        f"slope={slope:+.4e}, intercept={intercept:+.3f}, "
        f"|slope·n|={abs(slope*n_k):.2f}",
        fontsize=11,
    )
    out = OUT_DIR / "T1_full_refocus_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
