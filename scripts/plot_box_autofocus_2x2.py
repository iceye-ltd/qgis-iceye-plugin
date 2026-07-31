#!/usr/bin/env python3
"""Build a 2×3 before/mid/after diagnostic figure from a `box_NNN_autofocus.npz`.

Layout:

    ┌───────────────────────┬───────────────────────┬───────────────────────┐
    │ (0,0) Image Box       │ (0,1) Image Box       │ (0,2) Image Box       │
    │       unfocused       │       range-walk +    │       focused         │
    │       cmap = "gray"   │       PGA (no BW cut) │       cmap = "gray"   │
    │       (rotated CW90)  │       cmap = "gray"   │       (rotated CW90)  │
    │                       │       (rotated CW90)  │                       │
    ├───────────────────────┼───────────────────────┼───────────────────────┤
    │ (1,0) |FFT_az| raw    │ (1,1) |FFT_az|        │ (1,2) |FFT_az|        │
    │       cmap = "viridis"│       range-walk +    │       focused         │
    │       (rotated CW90)  │       PGA (no BW cut) │       (rotated CW90)  │
    │                       │       (rotated CW90)  │                       │
    └───────────────────────┴───────────────────────┴───────────────────────┘

All six panels share the same CW90 rotation. For the amp panels the
axes are (x = azimuth pixel, y = range pixel); for the FFT panels they
are (x = normalised Doppler frequency, y = range pixel), so range points
down uniformly across the figure.

The middle "range-walk + PGA (no bandlimit)" column is a hybrid stage
that does not exist in the AF pipeline itself: the polynomial range-walk
correction is applied (same as before), then the PGA azimuth phase
error is estimated from the same centred ``best_look_rows`` sub-band
the pipeline used, and that phase correction is applied to the
FULL-BANDWIDTH range-walked chip — no Doppler bins are zeroed. The
right "focused" column (``chip_after``) IS that same phase correction
combined WITH the centred-look bandlimit; the middle column therefore
isolates "what did the PGA phase correction do?" from "what did the
bandwidth truncation do?". When PGA did not fire for a box
(``best_look_rows == 0`` in the NPZ), the middle simplifies to plain
range-walk correction and equals the right column.

The ``|FFT_az|  unfocused`` panel (1,0) also has the fitted range-walk
polynomial overlaid as a red dashed curve. The curve is
``c(f) = n_rg/2 + fitted(f)``, with ``fitted(f) = -best_deviation/x[-1]^p · f^p``
matching the correction applied by ``_af_shift_fitted``. It shows the
AF corrective *shift* itself (a positive ``fitted`` shifts a Doppler
row toward larger range column indices, i.e. downward on the CW90-rotated
panel).

All six panels are stretched to ``[mean − k·std, mean + k·std]`` (linear
scale, no dB anywhere) with ``k`` set by ``--k-std`` (default 2.0):
    * Amp panels: shared stretch across all three chips' magnitudes so
      the before/mid/after comparison stays fair (AF preserves total
      energy — it just redistributes it — and linear-shared is the
      honest way to visualise that).
    * FFT panels: per-panel stretch on ``|fftshift(fft(chip, axis=0))|``,
      with exact-zero bins excluded from the stats for the "focused"
      panel — the AF pipeline zeroes a large fraction of the spectrum
      outside the retained sub-band, and letting those zeros feed the
      mean/std would collapse the display's dynamic range.

Usage
-----
    python scripts/plot_box_autofocus_2x2.py <box_NNN_autofocus.npz> \
        [--save <output.png>] \
        [--dpi 130] \
        [--k-std 2.0]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import os

import matplotlib

if "--show" not in os.sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _fft_az(chip: np.ndarray) -> np.ndarray:
    """|fftshift(fft(chip, axis=0))| — linear amplitude Doppler spectrum.

    Same FFT convention as `box_NNN_fft.png`. Linear scale (no dB).
    """
    return np.abs(
        np.fft.fftshift(np.fft.fft(chip, axis=0), axes=0)
    ).astype(np.float32, copy=False)


def _af_shift_fitted(s: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """Per-row range shift via Fourier-domain phase ramp.

    Standalone copy of ``_af_shift_fitted`` in ``shear_averaging.py``
    (which mirrors ``core.autofocus.shift_fitted``). Each row ``k`` of
    ``s`` is shifted along the range axis by ``fitted[k]`` samples via
    a Fourier-domain phase ramp — no interpolation, no data loss.
    """
    N = s.shape[1]
    k_over_N = (
        np.arange(N, dtype=np.float32) / np.float32(N) - np.float32(0.5)
    )
    k_shifted = np.fft.fftshift(k_over_N)
    phase = (
        np.float32(2.0 * np.pi)
        * fitted.astype(np.float32, copy=False)[:, None]
        * k_shifted[None, :]
    )
    shift_term = np.empty(phase.shape, dtype=np.complex64)
    np.cos(phase, out=shift_term.real)
    np.sin(phase, out=shift_term.imag)
    del phase
    fft_s = np.fft.fft(s, axis=1)
    fft_s *= shift_term
    del shift_term
    return np.fft.ifft(fft_s, axis=1)


# ─── Polynomial-fit search (copied verbatim from shear_averaging.py) ─── Contrast-based coarse-to-fine search for the range-walk polynomial. `_af_compute_contrast_subaperture_sums`, `_af_contrast_ratio_sum` and `_af_find_best_deviation` mirror the same-named functions in shear_averaging.py exactly; `_af_fit_polynomial_deviation_from_chip` is a thin wrapper that runs the fft + search on a raw complex chip so the plot script can recompute `best_deviation` from `chip_before` without trusting whatever is stored in the NPZ.

def _af_compute_contrast_subaperture_sums(
    image: np.ndarray, N_subaperture: int = 10,
) -> np.ndarray:
    """Per-subaperture focus metric over the centred 80 % of azimuth.

    Standalone copy of ``_af_compute_contrast_subaperture_sums`` in
    ``shear_averaging.py`` (which mirrors
    ``core.autofocus.compute_contrast_subaperture_sums``). Drops the
    outer 10 % of azimuth rows on each side, splits the remaining
    centred 80 % into ``N_subaperture`` equal chunks and returns
    ``std_range(mean_az(|x|²))`` per chunk. Higher = more focused.
    """
    H = image.shape[0]
    start = H // 10
    end = H - H // 10
    center = image[start:end]
    seg = (end - start) // N_subaperture
    intensity = np.zeros(N_subaperture, dtype=np.float32)
    if seg < 1:
        return intensity
    for i in range(N_subaperture):
        chunk = center[i * seg:(i + 1) * seg, :]
        if chunk.size == 0:
            continue
        intensity[i] = np.mean(np.abs(chunk) ** 2, axis=0).std()
    return intensity


def _af_contrast_ratio_sum(C: np.ndarray, C_initial: np.ndarray) -> float:
    """Sum of element-wise ``C / C_initial`` (1.0 where baseline is 0).

    Standalone copy of ``_af_contrast_ratio_sum`` in ``shear_averaging.py``.
    """
    ratio = np.divide(
        C, C_initial,
        out=np.ones_like(C, dtype=np.float64),
        where=C_initial > 0,
    )
    return float(np.sum(ratio))


def _mean_filter_1d(arr: np.ndarray, k: int) -> np.ndarray:
    """1D moving-average smoothing with edge-padding boundary handling. Returns a same-length array; edge samples are averaged against replicated boundary values (not zero-padded), which preserves non-zero endpoints of e.g. a parabolic `fitted` curve. `k < 2` is a no-op."""
    if k < 2 or arr.size == 0:
        return arr
    pad_left = k // 2
    pad_right = k - 1 - pad_left
    padded = np.pad(arr, (pad_left, pad_right), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / float(k)
    out = np.convolve(padded, kernel, mode="valid")
    return out.astype(arr.dtype, copy=False)


def _af_find_best_deviation(
    spatch_fft: np.ndarray,
    x: np.ndarray,
    *,
    dev_min: float = -100.0,
    dev_max: float = 100.0,
    accuracy: float = 0.25,
    poly_degree: int = 2,
) -> tuple[float, int]:
    """Coarse-to-fine search for the polynomial range deviation.

    Standalone copy of ``_af_find_best_deviation`` in ``shear_averaging.py``
    (identical numeric conventions: 10 subapertures, contrast-ratio-sum
    objective, coarse step = ``max(accuracy·5, (dev_max−dev_min)/20)``,
    then a ±2·coarse-step fine sweep at ``accuracy``). Returns
    ``(best_deviation, n_sub_contrast_improved)`` — the second value is
    the count of subapertures whose per-subaperture focus metric
    improved on the step that produced ``best_deviation``.
    """
    coarse_step = max(accuracy * 5.0, (dev_max - dev_min) / 20.0)
    best_deviation = 0.0
    N_subaperture = 10
    max_contrast = float(N_subaperture)
    C_initial = _af_compute_contrast_subaperture_sums(
        spatch_fft, N_subaperture,
    )
    n_sub_contrast_improved = 0

    coarse = np.arange(dev_min, dev_max + coarse_step, coarse_step)
    for deviation in coarse:
        c = deviation / x[-1] ** poly_degree
        fitted = -c * x ** poly_degree
        shifted = np.abs(_af_shift_fitted(spatch_fft, fitted))
        C = _af_compute_contrast_subaperture_sums(shifted, N_subaperture)
        contrast_gain = _af_contrast_ratio_sum(C, C_initial)
        if contrast_gain > max_contrast:
            n_sub_contrast_improved = int(np.sum(C > C_initial))
            max_contrast = contrast_gain
            C_initial = np.copy(C)
            best_deviation = float(deviation)
    if abs(best_deviation) <= 1.0:
        return best_deviation, int(n_sub_contrast_improved)
    # Local override vs the shear_averaging.py copy: reset `max_contrast` to the identity-ratio value (`sum(C_i / C_i) = N_subaperture = 10`) before the fine loop. `C_initial` now holds `C(coarse_winner)`, so the correct acceptance gate for a fine perturbation is "does this deviation strictly improve on the coarse winner's per-subaperture focus metric?" — i.e. `contrast_gain > 10.0`. Carrying `max_contrast` over from the coarse loop (where it is the step-to-step ratio of the last coarse acceptance, typically ≫ 10) makes the fine gate unreachable in practice and glues `best_deviation` to the coarse grid. See `_af_find_best_deviation` in `shear_averaging.py` L3325-3384 for the shared original.
    max_contrast = float(N_subaperture)
    fine_min = max(dev_min, best_deviation -  coarse_step)
    fine_max = min(dev_max, best_deviation +  coarse_step)
    fine = np.arange(fine_min, fine_max + accuracy, accuracy)
    for deviation in fine:

        c = deviation / x[-1] ** poly_degree
        fitted = -c * x ** poly_degree
        # Smooth the polynomial correction with a moving-average filter of length `N/10` (edge-padded boundaries so the parabola's non-zero endpoints are preserved). Replaces the earlier flat-middle hack — the mean filter softens the whole curve rather than only its centre.
        fitted = _mean_filter_1d(fitted, fitted.size // 3)
        shifted = np.abs(_af_shift_fitted(spatch_fft, fitted))
        C = _af_compute_contrast_subaperture_sums(shifted, N_subaperture)
        contrast_gain = _af_contrast_ratio_sum(C, C_initial)

        if contrast_gain > max_contrast:
            n_sub_contrast_improved = int(np.sum(C > C_initial))
            max_contrast = contrast_gain
            C_initial = np.copy(C)
            best_deviation = float(deviation)
            print("fine"+str(best_deviation))
    return best_deviation, int(n_sub_contrast_improved)


def _af_fit_polynomial_deviation_from_chip(
    chip_before: np.ndarray,
    *,
    dev_min: float = -100.0,
    dev_max: float = 100.0,
    accuracy: float = 0.5,
    poly_degree: int = 2,
) -> tuple[float, int]:
    """Fit ``best_deviation`` from a raw complex chip ``chip_before``.

    FFTs along azimuth (matches ``_af_apply_global_range_deviation_correction``
    in ``shear_averaging.py``), then runs the polynomial-fit search
    on the fftshift'd spectrum. Search bounds are in range SAMPLES (not
    metres) — callers convert from metres via
    ``int(dev_meters / range_spacing)`` if needed.
    """
    rows = int(chip_before.shape[0])
    spatch_fft = np.fft.fftshift(np.fft.fft(chip_before, axis=0), axes=0)
    x = np.linspace(-0.5, 0.5, rows).astype(np.float64, copy=False)
    return _af_find_best_deviation(
        spatch_fft, x,
        dev_min=dev_min, dev_max=dev_max,
        accuracy=accuracy, poly_degree=poly_degree,
    )


def _apply_range_walk_correction(
    chip_before: np.ndarray,
    best_deviation: float,
    poly_degree: int,
) -> np.ndarray:
    """Rebuild the range-walk-corrected chip (before PGA + bandlimiting).

    Reproduces the intermediate `corrected` that lives inside
    ``_af_apply_global_range_deviation_correction`` between the
    polynomial Fourier-domain range shift and the centred-look PGA
    call. Deterministic — same inputs (``chip_before``,
    ``best_deviation``, ``af_poly_degree``) always produce the same
    intermediate.
    """
    rows = int(chip_before.shape[0])
    x = np.linspace(-0.5, 0.5, rows).astype(np.float32, copy=False)
    fitted = (-float(best_deviation) / (x[-1] ** poly_degree)) * (x ** poly_degree)
    spatch_fft = np.fft.fftshift(np.fft.fft(chip_before, axis=0), axes=0)
    spatch_fft = _af_shift_fitted(spatch_fft, fitted)
    corrected = np.fft.ifft(np.fft.ifftshift(spatch_fft, axes=0), axis=0)
    return corrected.astype(chip_before.dtype, copy=False)


# ─── Local PGA re-implementation (no bandlimit) ────────────────────── Compact port of the standalone PGA helpers in `shear_averaging.py` (`_pga_*`) so we can compute a "range-walk + PGA phase correction WITHOUT the bandlimit" intermediate from the NPZ alone. Same numeric conventions (centred FFTs, magnitude-weighted phase estimator, single PGA iteration, deterministic in `best_look_rows`).

def _pga_ft(s: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(s, axis=axis), axes=axis)


def _pga_ift(f: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.fft.ifft(np.fft.ifftshift(f, axes=axis), axis=axis)


def _pga_select_strong_pulses(
    s: np.ndarray, percentile: float = 95.0,
) -> np.ndarray:
    """Rows of `s` whose max-amplitude profile is in the top ``100 - percentile`` percent."""
    line_max = np.amax(np.abs(s), axis=1)
    thr = np.percentile(line_max, percentile)
    idx = np.where(line_max >= thr)[0]
    return s[idx]


def _pga_center_on_strong_target_axis1(x: np.ndarray) -> np.ndarray:
    H, W = x.shape
    max_index = np.argmax(np.abs(x), axis=1)
    shifts = (W // 2) - max_index
    return x[np.arange(H)[:, None], (np.arange(W) - shifts[:, None]) % W]


def _pga_calculate_window_over_axis0(
    s: np.ndarray, threshold: float = -20.0, min_width: int = 50,
) -> np.ndarray:
    """`_pga_calculate_window(..., axis=0)` — collapses az, returns range window."""
    p = np.sum(np.abs(s) ** 2, axis=0)
    p_max = p.max()
    if p_max > 0 and np.isfinite(p_max):
        p_db = 10.0 * np.log10(p / p_max)
        width = int(np.sum(p_db > threshold))
    else:
        width = min_width
    if width < min_width:
        width = min_width
    n_rg = s.shape[1]
    width = min(width, n_rg - 1)
    if width < 1:
        width = max(1, n_rg - 1)
    center = (n_rg - 1) // 2
    return np.arange(-width // 2, width // 2) + center


def _pga_weighted_phase_estimator(x: np.ndarray) -> np.ndarray:
    s = np.conj(x[:-1, :]) * x[1:, :]
    denom = np.sum(np.abs(s), axis=1)
    denom = np.where(denom > 0, denom, 1.0)
    return np.sum(np.angle(s) * np.abs(s), axis=1) / denom


def _pga_phase_gradient_autofocus(
    data: np.ndarray, iter_num: int = 1,
) -> np.ndarray:
    """PGA azimuth-phase-error estimate. Returns a length-``H`` real array (broadcastable via ``[:, None]``)."""
    phase_corrections = np.zeros(data.shape[0])
    d = data
    for _ in range(iter_num):
        d_centered = _pga_center_on_strong_target_axis1(d)
        window = _pga_calculate_window_over_axis0(d_centered)
        p = np.zeros_like(d_centered)
        p[:, window] = d_centered[:, window]
        P = _pga_ft(p, axis=0)
        phase_change = _pga_weighted_phase_estimator(P)
        phase_change = np.unwrap([0.0, *np.cumsum(phase_change)])
        t = np.arange(phase_change.shape[0])
        trend = np.poly1d(np.polyfit(t, phase_change, 1))
        phase_change = phase_change - trend(t)
        d = _pga_ft(d, axis=0)
        d = d * np.exp(-1j * phase_change[:, None])
        d = _pga_ift(d, axis=0)
        phase_corrections = phase_corrections + phase_change
    return phase_corrections


def _pga_apply_phase_correction_to_full_bw(
    data: np.ndarray, phase_error: np.ndarray,
) -> np.ndarray:
    """Apply ``phase_error`` (length ``H_est``) to the full-bandwidth ``data`` (H, W).

    The phase estimate can have a different row count than ``data`` when
    it was estimated on a bandlimited look; we linearly interpolate onto
    ``data``'s row grid before multiplying in the Doppler domain (same
    convention as ``_pga_apply_phase_correction`` in shear_averaging.py).
    """
    x = np.linspace(0.0, 1.0, data.shape[0])
    xp = np.linspace(0.0, 1.0, phase_error.shape[0])
    phi = np.interp(x, xp, phase_error)
    d = _pga_ft(data, axis=0)
    d = d * np.exp(-1j * phi[:, None])
    return _pga_ift(d, axis=0)


def _apply_range_walk_plus_pga_no_bandlimit(
    chip_range_walk: np.ndarray,
    best_look_rows: int,
) -> np.ndarray:
    """Estimate the PGA azimuth phase error from the centred ``best_look_rows`` sub-band, then apply it to the FULL-bandwidth range-walked chip (no bandlimit).

    Mirrors the phase-estimation path of ``_focus_with_centered_looks_pga``
    but keeps every Doppler bin of ``chip_range_walk`` — the winning look
    is only used to derive the phase error, never to gate the output
    spectrum. When ``best_look_rows == 0`` (PGA did not fire for this
    box) the chip is returned unchanged.
    """
    if best_look_rows is None or best_look_rows <= 0:
        return chip_range_walk

    data = np.ascontiguousarray(chip_range_walk, dtype=np.complex64)
    n_az, n_rg = data.shape
    look_rows = int(min(best_look_rows, n_az))

    spectrum = np.fft.fftshift(np.fft.fft2(data))
    look_slab = np.zeros_like(spectrum)
    r0 = (n_az // 2) - (look_rows // 2)
    r1 = r0 + look_rows
    look_slab[r0:r1, :] = spectrum[r0:r1, :]
    look = np.fft.ifft2(np.fft.ifftshift(look_slab))

    patch = _pga_select_strong_pulses(look)
    if patch.size == 0 or patch.shape[0] < 2:
        return chip_range_walk

    try:
        phase_error = _pga_phase_gradient_autofocus(patch)
    except Exception:
        return chip_range_walk

    corrected = _pga_apply_phase_correction_to_full_bw(data, phase_error)
    return corrected.astype(chip_range_walk.dtype, copy=False)


def _mean_std_stretch(
    values: np.ndarray,
    k_std: float = 2.0,
    drop_zeros: bool = False,
) -> tuple[float, float]:
    """`[mean − k·std, mean + k·std]` stretch over finite `values`.

    Set `drop_zeros=True` for the "focused" FFT panel — the AF pipeline
    zeroes a large fraction of the Doppler spectrum outside the
    retained sub-band, and including those exact-zero bins in the
    stats drags the mean toward zero and inflates the std, which
    collapses the real dynamic range to a thin sliver on the display.
    Dropping them keeps the stretch anchored on the active spectrum.
    """
    v = values[np.isfinite(values)]
    if drop_zeros:
        v = v[v > 0.0]
    if v.size < 2:
        v = values[np.isfinite(values)]
        if v.size < 2:
            return 0.0, 1.0
    mu = float(v.mean())
    sd = float(v.std())
    if not np.isfinite(sd) or sd <= 0.0:
        return mu - 1.0, mu + 1.0
    return mu - float(k_std) * sd, mu + float(k_std) * sd


def _plot_simple_2x1(
    chip_before: np.ndarray,
    *,
    box_idx: int,
    y_center: int | None,
    x_center: int | None,
    h_box: int | None,
    w_box: int | None,
    strong: bool,
    best_dev: float,
    poly_degree: int,
    n_sub_imp: int,
    refit: bool,
    best_dev_npz: float | None,
    save: Path | None,
    dpi: int,
    k_std: float,
    show: bool = False,
) -> None:
    """Single-panel diagnostic: just the azimuth FFT of ``chip_before`` with the fitted polynomial overlaid.

    CW90 rotation so x = normalised Doppler frequency, y = range pixel.
    Linear ``mean ± k·std`` stretch. The polynomial curve is drawn as
    ``c(f) = c_center + fitted(f)``, with ``fitted(f) = -best_dev/x[-1]^p · f^p``
    matching the AF pipeline's ``_af_shift_fitted`` corrective shift.
    """
    n_az, n_rg = chip_before.shape

    fft = _fft_az(chip_before)
    vmin_f, vmax_f = _mean_std_stretch(fft, k_std=k_std, drop_zeros=False)
    fft_rot = fft.T[:, ::-1]

    f_top = -0.5
    f_bot = 0.5 - 1.0 / max(n_az, 1)
    fft_extent_rot = (float(f_bot), float(f_top), float(n_rg), 0.0)

    star = "*" if strong else " "
    src = "local refit" if refit else "NPZ stored"
    dev_str = f"{best_dev:+.3f}" if np.isfinite(best_dev) else "nan"
    npz_note = (
        f"   [NPZ stored: {best_dev_npz:+.3f}]"
        if refit and best_dev_npz is not None and np.isfinite(best_dev_npz)
        else ""
    )

    fig, ax = plt.subplots(1, 1, figsize=(11, 6), constrained_layout=True)
    fig.suptitle(
        f"{star}box {box_idx:03d}  y={y_center} x={x_center}  "
        f"h={h_box} w={w_box}\n"
        f"best_deviation={dev_str}  ({src})   "
        f"n_sub_contrast_improved={n_sub_imp}/10   "
        f"poly_degree={poly_degree}{npz_note}",
        fontsize=11,
    )

    ax.imshow(
        fft_rot, aspect="auto", cmap="viridis",
        vmin=vmin_f, vmax=vmax_f,
        extent=fft_extent_rot,
    )
    # `c(f) = c_center + fitted(f)` where `fitted(f) = -best_dev/x[-1]^p · f^p` — same convention as `_af_shift_fitted` in shear_averaging.py: positive `fitted` shifts a Doppler row toward larger range column indices (downward on the CW90-rotated panel), so the drawn curve traces the AF's corrective shift. Smoothed with a `N/10` moving-average filter to match the same operation applied inside `_af_find_best_deviation`'s fine loop.
    if np.isfinite(best_dev):
        f_curve = np.linspace(f_top, f_bot, 400)
        fitted_curve = (
            -float(best_dev) / (0.5 ** poly_degree)
        ) * (f_curve ** poly_degree)
        fitted_curve = _mean_filter_1d(fitted_curve, fitted_curve.size // 10)
        c_center = 0.5 * float(n_rg)
        c_curve = c_center + fitted_curve
        inside = (c_curve >= 0.0) & (c_curve <= float(n_rg))
        c_plot = np.where(inside, c_curve, np.nan)
        ax.plot(
            f_curve, c_plot, color="red", ls="--", lw=1.5,
            label=(
                rf"fitted poly (deg {poly_degree}, {src}, mean-filtered N/10): "
                rf"$c(f) = {c_center:.1f} + "
                rf"{-float(best_dev) / (0.5 ** poly_degree):+.2f}\,"
                rf"f^{{{poly_degree}}}$"
            ),
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.75)
    ax.set_title(r"$|\mathrm{FFT}_{\mathrm{az}}\,I|$  unfocused (with fitted polynomial)")
    ax.set_xlabel("normalised Doppler frequency")
    ax.set_ylabel("range pixel")

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi)
        print(f"Saved → {save}")
    if show:
        plt.show()
    plt.close(fig)


def plot_box_autofocus_2x2(
    npz_path: Path,
    save: Path | None,
    dpi: int = 130,
    k_std: float = 2.0,
    *,
    simple: bool = False,
    refit: bool = True,
    fit_dev_min_samples: float = -100.0,
    fit_dev_max_samples: float = 100.0,
    fit_accuracy: float = 0.5,
    fit_poly_degree: int | None = None,
    show: bool = False,
) -> None:
    with np.load(npz_path, allow_pickle=False) as arch:
        needed = ("chip_before", "chip_after")
        missing = [k for k in needed if k not in arch.files]
        if missing:
            raise KeyError(
                f"NPZ {npz_path.name} is missing required keys: {missing}. "
                f"Expected both `chip_before` and `chip_after` (complex64)."
            )
        chip_before = np.asarray(arch["chip_before"])
        chip_after = np.asarray(arch["chip_after"])

        def _scalar(name, cast=float, default=None):
            if name in arch.files:
                v = arch[name]
                return cast(v) if v.shape == () else cast(v.item())
            return default

        box_idx = _scalar("box_idx", int, default=-1)
        y_center = _scalar("y_center", int)
        x_center = _scalar("x_center", int)
        h_box = _scalar("h", int)
        w_box = _scalar("w", int)
        best_dev_npz = _scalar("best_deviation", float)
        c_before = _scalar("contrast_before", float)
        c_after = _scalar("contrast_after", float)
        gain_db = _scalar("gain_db", float)
        n_sub_imp_npz = _scalar("n_sub_contrast_improved", int)
        best_look_rows = _scalar("best_look_rows", int, default=0)
        strong = bool(_scalar("strong", bool, default=False))
        poly_degree = _scalar("af_poly_degree", int, default=2)

    if chip_before.shape != chip_after.shape:
        raise ValueError(
            f"chip_before {chip_before.shape} vs chip_after {chip_after.shape}"
            " shape mismatch."
        )
    n_az, n_rg = chip_before.shape

    # Recompute the polynomial fit from `chip_before` using the exact same coarse-to-fine contrast-based search as `_af_find_best_deviation` in shear_averaging.py. When `refit=True` (default), the locally-computed value drives the overlay so the plot is a self-contained diagnostic — no reliance on whatever `best_deviation` the pipeline happened to store in the NPZ. A diagnostic line prints the local fit alongside the NPZ's stored value so any drift is immediately visible.
    fit_p = int(fit_poly_degree if fit_poly_degree is not None else poly_degree)
    if refit:
        best_dev_local, n_sub_imp_local = _af_fit_polynomial_deviation_from_chip(
            chip_before,
            dev_min=float(fit_dev_min_samples),
            dev_max=float(fit_dev_max_samples),
            accuracy=float(fit_accuracy),
            poly_degree=fit_p,
        )
        best_dev = float(best_dev_local)
        n_sub_imp = int(n_sub_imp_local)
        poly_degree = fit_p
        print(
            f"[fit] box {box_idx:03d}: local refit "
            f"best_deviation={best_dev:+.3f}  "
            f"n_sub_contrast_improved={n_sub_imp}/10   "
            f"(NPZ stored: {best_dev_npz:+.3f}, {n_sub_imp_npz}/10; "
            f"search=[{fit_dev_min_samples:g}, {fit_dev_max_samples:g}] samples, "
            f"accuracy={fit_accuracy}, poly_degree={fit_p})"
        )
    else:
        best_dev = float(best_dev_npz) if best_dev_npz is not None else float("nan")
        n_sub_imp = int(n_sub_imp_npz or 0)

    if simple:
        _plot_simple_2x1(
            chip_before,
            box_idx=box_idx, y_center=y_center, x_center=x_center,
            h_box=h_box, w_box=w_box, strong=strong,
            best_dev=best_dev, poly_degree=poly_degree,
            n_sub_imp=n_sub_imp, refit=refit,
            best_dev_npz=best_dev_npz,
            save=save, dpi=dpi, k_std=k_std, show=show,
        )
        return

    # Middle stage: "range-walk + PGA (WITHOUT bandlimit)". First reconstruct the range-walk-corrected chip from `chip_before` + `best_deviation` + `af_poly_degree`, then estimate the PGA azimuth phase error from the same centred `best_look_rows` sub-band the AF pipeline used, and apply that phase correction to the FULL-bandwidth range-walked chip. This differs from the right column (`chip_after`) only by the omitted bandlimit — the PGA phase correction is identical. When PGA did not fire (`best_look_rows == 0`), `chip_range_walk_pga` and `chip_after` are numerically identical.
    chip_range_walk = _apply_range_walk_correction(
        chip_before,
        best_deviation=best_dev if np.isfinite(best_dev) else 0.0,
        poly_degree=poly_degree,
    )
    chip_range_walk_pga = _apply_range_walk_plus_pga_no_bandlimit(
        chip_range_walk, best_look_rows=int(best_look_rows or 0),
    )

    amp_before = np.abs(chip_before).astype(np.float32, copy=False)
    amp_mid = np.abs(chip_range_walk_pga).astype(np.float32, copy=False)
    amp_after = np.abs(chip_after).astype(np.float32, copy=False)
    # Shared mean/std across ALL THREE amp panels so before → range-walk+PGA (no BW cut) → focused stays on the same brightness scale (AF preserves total energy, just redistributes it, so linear-shared is the honest visualisation).
    vmin_i, vmax_i = _mean_std_stretch(
        np.concatenate([amp_before.ravel(),
                        amp_mid.ravel(),
                        amp_after.ravel()]),
        k_std=k_std,
    )
    if vmin_i < 0.0:
        vmin_i = 0.0

    fft_before = _fft_az(chip_before)
    fft_mid = _fft_az(chip_range_walk_pga)
    fft_after = _fft_az(chip_after)
    # Per-panel stretch for FFTs. The "focused" spectrum has a lot of exact-zero bins where the AF pipeline zeroed the spectrum outside the retained sub-band; `drop_zeros=True` keeps those out of the stats so the mean/std reflect the active spectrum, not a background of zeros. The "range-walk + PGA (no bandlimit)" spectrum keeps the full Doppler support (only the phase is corrected, no bins are zeroed), so `drop_zeros=False`.
    vmin_fb, vmax_fb = _mean_std_stretch(fft_before, k_std=k_std, drop_zeros=False)
    vmin_fm, vmax_fm = _mean_std_stretch(fft_mid, k_std=k_std, drop_zeros=False)
    vmin_fa, vmax_fa = _mean_std_stretch(fft_after, k_std=k_std, drop_zeros=True)

    # CW90 rotation on both the amplitude and FFT panels (same convention as `_show_slc_rot_cw90` in shear_averaging.py). `.T[:, ::-1]` is a strided view — no data copy. All six panels then share the same "azimuth (or Doppler frequency) → x, range → y (down)" layout.
    amp_before_rot = amp_before.T[:, ::-1]
    amp_mid_rot = amp_mid.T[:, ::-1]
    amp_after_rot = amp_after.T[:, ::-1]
    fft_before_rot = fft_before.T[:, ::-1]
    fft_mid_rot = fft_mid.T[:, ::-1]
    fft_after_rot = fft_after.T[:, ::-1]

    # Pixel-index extent for the rotated amp panels so the x-axis reads 0 → n_az (azimuth) and y reads 0 → n_rg (range).
    amp_extent = (0.0, float(n_az), float(n_rg), 0.0)

    # FFT panels: after CW90 rotation the x-axis is normalised Doppler frequency in (−½, +½] and the y-axis is range pixel. `.T[:, ::-1]` maps original row `k` (Doppler bin) to new column `n_az−1−k`, so the leftmost column (x = f_bot) is the most-positive Doppler and the rightmost (x = f_top = −0.5) is the most-negative. Setting `extent = (f_bot, f_top, n_rg, 0.0)` reproduces that "high Doppler left → low Doppler right" mapping (same convention as shear_averaging.py's `box_NNN_fft.png`, just rotated).
    f_top = -0.5
    f_bot = 0.5 - 1.0 / max(n_az, 1)
    fft_extent_rot = (float(f_bot), float(f_top), float(n_rg), 0.0)

    star = "*" if strong else " "
    pga_tag = (
        f"same-BW @ look_rows={best_look_rows}, range-walk + PGA"
        if (best_look_rows and best_look_rows > 0)
        else "full-BW, PGA not triggered"
    )
    gain_str = f"{gain_db:+.2f} dB" if np.isfinite(gain_db) else "nan dB"
    dev_str = f"{best_dev:+.3f}" if np.isfinite(best_dev) else "nan"

    pga_fired = best_look_rows and best_look_rows > 0
    mid_tag = (
        f"range-walk + PGA  (no bandlimit; phase from look_rows={best_look_rows})"
        if pga_fired
        else "range-walk only  (PGA not triggered — same as focused)"
    )

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    fig.suptitle(
        f"{star}box {box_idx:03d}  y={y_center} x={x_center}  "
        f"h={h_box} w={w_box}\n"
        f"contrast gain: {gain_str}   "
        f"({c_before:.3f} → {c_after:.3f};  {pga_tag})\n"
        f"best_deviation={dev_str}   "
        f"n_sub_contrast_improved={n_sub_imp}/10",
        fontsize=11,
    )

    axes[0, 0].imshow(
        amp_before_rot, aspect="auto", cmap="gray",
        vmin=vmin_i, vmax=vmax_i,
        extent=amp_extent,
    )
    axes[0, 0].set_title("Image Box — unfocused")
    axes[0, 0].set_xlabel("azimuth pixel")
    axes[0, 0].set_ylabel("range pixel")

    axes[0, 1].imshow(
        amp_mid_rot, aspect="auto", cmap="gray",
        vmin=vmin_i, vmax=vmax_i,
        extent=amp_extent,
    )
    axes[0, 1].set_title(f"Image Box — {mid_tag}")
    axes[0, 1].set_xlabel("azimuth pixel")
    axes[0, 1].set_ylabel("range pixel")

    axes[0, 2].imshow(
        amp_after_rot, aspect="auto", cmap="gray",
        vmin=vmin_i, vmax=vmax_i,
        extent=amp_extent,
    )
    axes[0, 2].set_title("Image Box — focused")
    axes[0, 2].set_xlabel("azimuth pixel")
    axes[0, 2].set_ylabel("range pixel")

    axes[1, 0].imshow(
        fft_before_rot, aspect="auto", cmap="viridis",
        vmin=vmin_fb, vmax=vmax_fb,
        extent=fft_extent_rot,
    )
    # Overlay the fitted polynomial that the AF search recovered — plotted as `c(f) = c_center + fitted(f)`, where `fitted(f) = -best_deviation / x[-1]^p · f^p` is the correction applied by `_af_shift_fitted`. The plotted polynomial contribution is the AF *shift* itself (positive `fitted` shifts a Doppler row toward larger range column indices, i.e. downward on the CW90-rotated panel), NOT the target's pre-correction walk. Anchored at the mid-range column (`c_center = n_rg / 2`). After the CW90 rotation the axes are (x = Doppler, y = range), so we plot (f_curve, c_curve). Smoothed with a `N/10` moving-average filter to match the same operation applied inside `_af_find_best_deviation`'s fine loop.
    if np.isfinite(best_dev):
        f_curve = np.linspace(f_top, f_bot, 400)
        fitted_curve = (
            -float(best_dev) / (0.5 ** poly_degree)
        ) * (f_curve ** poly_degree)
        fitted_curve = _mean_filter_1d(fitted_curve, fitted_curve.size // 10)
        c_center = 0.5 * float(n_rg)
        c_curve = c_center + fitted_curve
        inside = (c_curve >= 0.0) & (c_curve <= float(n_rg))
        c_plot = np.where(inside, c_curve, np.nan)
        axes[1, 0].plot(
            f_curve, c_plot, color="red", ls="--", lw=1.2,
            label=(
                rf"fitted poly (deg {poly_degree}, mean-filtered N/10): "
                rf"$c(f) = {c_center:.1f} + "
                rf"{-float(best_dev) / (0.5 ** poly_degree):+.2f}\,f^{{{poly_degree}}}$"
            ),
        )
        axes[1, 0].legend(loc="lower right", fontsize=8, framealpha=0.75)
    axes[1, 0].set_title(r"$|\mathrm{FFT}_{\mathrm{az}}\,I|$  unfocused")
    axes[1, 0].set_xlabel("normalised Doppler frequency")
    axes[1, 0].set_ylabel("range pixel")

    axes[1, 1].imshow(
        fft_mid_rot, aspect="auto", cmap="viridis",
        vmin=vmin_fm, vmax=vmax_fm,
        extent=fft_extent_rot,
    )
    axes[1, 1].set_title(
        r"$|\mathrm{FFT}_{\mathrm{az}}\,I|$  range-walk + PGA (no bandlimit)"
    )
    axes[1, 1].set_xlabel("normalised Doppler frequency")
    axes[1, 1].set_ylabel("range pixel")

    axes[1, 2].imshow(
        fft_after_rot, aspect="auto", cmap="viridis",
        vmin=vmin_fa, vmax=vmax_fa,
        extent=fft_extent_rot,
    )
    axes[1, 2].set_title(r"$|\mathrm{FFT}_{\mathrm{az}}\,I|$  focused")
    axes[1, 2].set_xlabel("normalised Doppler frequency")
    axes[1, 2].set_ylabel("range pixel")

    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=dpi)
        print(f"Saved → {save}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a 2×2 (amp | fft) before/after diagnostic figure from a "
            "`box_NNN_autofocus.npz`."
        ),
    )
    ap.add_argument(
        "npz", type=Path,
        help="Path to a `box_NNN_autofocus.npz` written by shear_averaging.py.",
    )
    ap.add_argument(
        "--save", type=Path, default=None,
        help=(
            "Output PNG path. Default: <npz_stem>_2x2.png next to the NPZ."
        ),
    )
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument(
        "--k-std", type=float, default=2.0,
        help=(
            "Half-width of the colormap stretch in standard deviations "
            "(so vmin/vmax = mean ± k·std). Applies to all panels. "
            "Default 2.0."
        ),
    )
    ap.add_argument(
        "--simple", action="store_true",
        help=(
            "Emit a single-panel diagnostic: just the |FFT_az| of the "
            "unfocused chip with the fitted polynomial overlaid."
        ),
    )
    ap.add_argument(
        "--no-refit", dest="refit", action="store_false", default=True,
        help=(
            "Use the `best_deviation` stored in the NPZ instead of "
            "recomputing it from `chip_before`. Default is to REFIT "
            "locally via the ported `_af_find_best_deviation` search."
        ),
    )
    ap.add_argument(
        "--fit-dev-min-samples", type=float, default=-100.0,
        help="Lower search bound for `best_deviation` in range samples (default -100).",
    )
    ap.add_argument(
        "--fit-dev-max-samples", type=float, default=100.0,
        help="Upper search bound for `best_deviation` in range samples (default +100).",
    )
    ap.add_argument(
        "--fit-accuracy", type=float, default=0.5,
        help="Fine-search step in range samples (default 0.5).",
    )
    ap.add_argument(
        "--fit-poly-degree", type=int, default=None,
        help=(
            "Polynomial degree for the fit. Default = value stored in "
            "the NPZ (`af_poly_degree`, typically 2)."
        ),
    )
    ap.add_argument(
        "--show", action="store_true",
        help=(
            "Open an interactive matplotlib window instead of writing to "
            "disk. When combined with --save, does both. When neither "
            "--save nor --show is set, defaults to saving next to the NPZ."
        ),
    )
    args = ap.parse_args()

    default_stem = "_fft_poly" if args.simple else "_2x2"
    if args.save is not None:
        save = args.save
    elif args.show:
        save = None
    else:
        save = args.npz.with_name(args.npz.stem + default_stem + ".png")
    plot_box_autofocus_2x2(
        args.npz, save,
        dpi=args.dpi, k_std=args.k_std,
        simple=args.simple, refit=args.refit,
        fit_dev_min_samples=args.fit_dev_min_samples,
        fit_dev_max_samples=args.fit_dev_max_samples,
        fit_accuracy=args.fit_accuracy,
        fit_poly_degree=args.fit_poly_degree,
        show=args.show,
    )


if __name__ == "__main__":
    main()
