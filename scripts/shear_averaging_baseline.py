"""SAR moving target detection via phase derivative analysis.

Given an SLC patch s with axes (azimuth=vertical, range=horizontal):

1. Range resolution is degraded by coherently summing range sub-bands.
   This suppresses range-dependent phase noise while preserving the
   azimuth phase structure of moving targets.

2. The azimuth phase derivative is computed on the degraded image:
       d_phase[i, v] = arg{ s_degraded[i+1, v] · conj(s_degraded[i, v]) }
   Result is real-valued, -pi to pi, axis 0 = azimuth pixels (spatial).

3. Moving target detection: each range column of d_phase is convolved
   with a sinusoidal kernel. A moving target produces an azimuth phase
   gradient with a sign inversion (ramps up then down, or vice versa),
   which matches the sinusoidal kernel and gives a strong response.
   Noise and stationary targets average to zero.
"""

from __future__ import annotations

import argparse
import time
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless: figures are saved to disk, never shown.
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np
from scipy.signal import fftconvolve


class _Stopwatch:
    """Lightweight pipeline timer used by `main()` to report total wall
    time plus per-stage durations. `mark()` prints/records the time
    spent since the previous mark (or `__init__`); `report()` prints
    the consolidated summary at the end.
    """

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.steps: list[tuple[str, float]] = []
        self._pause_start: float | None = None

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        dt = now - self.last
        self.steps.append((label, dt))
        print(f"  [time] {label:60s} {dt:8.2f} s")
        self.last = now

    def pause(self) -> None:
        """Stop counting wall time. Spans bracketed by `pause` / `resume`
        are excluded from both the current step's duration and from the
        final total reported by `report()`.
        """
        if self._pause_start is None:
            self._pause_start = time.perf_counter()

    def resume(self) -> None:
        """Resume counting wall time. The duration between the matching
        `pause` and this call is discarded by advancing both `self.last`
        and `self.t0` by that delta.
        """
        if self._pause_start is None:
            return
        delta = time.perf_counter() - self._pause_start
        self.last += delta
        self.t0 += delta
        self._pause_start = None

    def report(self) -> None:
        total = time.perf_counter() - self.t0
        bar = "=" * 82
        print(bar)
        print("Processing-time summary")
        print(bar)
        for label, dt in self.steps:
            frac = 100.0 * dt / total if total > 0 else 0.0
            print(f"  {label:60s} {dt:8.2f} s  ({frac:5.1f}%)")
        print("-" * 82)
        print(f"  {'TOTAL':60s} {total:8.2f} s")
        print(bar)


DEFAULT_PATCH = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_141944_569017.npy"
)
DEFAULT_SAVE_DIR = Path("/home/odogan/Desktop/ship_focusing/4439676")


def _default_save_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return DEFAULT_SAVE_DIR / f"shear_{ts}_{suffix}.png"


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def apply_range_window(
    s: np.ndarray,
    sll_db: float = 55.0,
    nbar: int = 8,
) -> np.ndarray:
    """Aggressively suppress range sidelobes by tapering the SLC range
    spectrum with a Taylor window.

    The SLC range impulse response is sinc-like with first sidelobes at
    only ≈ −13 dB because the SAR matched filter leaves a
    near-rectangular spectrum across the chirp bandwidth. Multiplying
    that spectrum by a Taylor window collapses every sidelobe to
    ≤ −`sll_db`: the inner `nbar` lobes on each side are held flat at
    exactly that level, the rest decay as a sinc tail. The cost is a
    broader main lobe (≈ 1.5× rectangular at −40 dB, ≈ 1.85× at −55 dB,
    ≈ 2.0× at −65 dB).

    Taylor is the SAR canonical choice because the sidelobe level is a
    single dial, with the inner-lobe count `nbar` controlling how many
    sidelobes are forced flat (higher `nbar` → closer to ideal
    Dolph-Chebyshev, slightly more energy in the wings). Axis 0
    (azimuth) is untouched.

    Parameters
    ----------
    s : complex ndarray, shape (N_az, N_range)
        Input SLC; axis 1 = range (the axis being windowed).
    sll_db : float
        Peak sidelobe level in dB **below** the main lobe.
        ≈ 35 → Hamming-equivalent.
        45  → typical airborne-SAR setting.
        55+ → very low; for high dynamic range scenes (ship vs wake,
              ship vs small boat). 0 disables and returns `s`.
    nbar : int
        Number of equal-level sidelobes flattened on each side of the
        main lobe. Must be large enough that the design is well-posed
        for the requested `sll_db`; otherwise the realised PSL falls a
        few dB short of the dial. Empirically `nbar=8` saturates the
        target up to ≈ −55 dB; bump to 12–16 only if pushing past
        −60 dB.

    Returns
    -------
    s_w : complex ndarray, same shape as `s`.
        Range-windowed SLC. Window is mean-normalised so global
        amplitude is preserved (no DC gain change vs. `s`).
    """
    if sll_db <= 0.0:
        return s

    from scipy.signal.windows import taylor

    N_range = s.shape[1]
    w = taylor(N_range, nbar=nbar, sll=sll_db, norm=False).astype(np.float64)
    w /= w.mean()                                       # preserve mean amplitude
    S = np.fft.fftshift(np.fft.fft(s, axis=1), axes=1)  # DC at centre
    S *= w[np.newaxis, :]
    return np.fft.ifft(np.fft.ifftshift(S, axes=1), axis=1)


def degrade_range_resolution_range_sum(ship: np.ndarray, Number_of_Range_Looks: int) -> tuple[np.ndarray, np.ndarray]:
    """Degrade range resolution by coherently summing range sub-bands,
    then compute the azimuth phase derivative.

    Parameters
    ----------
    ship : complex ndarray, shape (N_az, N_range)
        Input SLC patch. Axis 0 = azimuth (spatial), axis 1 = range (spatial).

    Returns
    -------
    s_degraded : complex ndarray, shape (N_az, N_range // Number_of_Range_Looks)
        Range-degraded SLC. Axis 0 is still azimuth spatial pixels.
    d_phase : real ndarray, shape (N_az - 1, N_range // Number_of_Range_Looks)
        Azimuth phase derivative: arg{ s_degraded[i+1] · conj(s_degraded[i]) }.
        Values in [-pi, pi]. Wrap-safe by construction.
    """
    # degradation_ratio = 0.8
    # Number_of_Range_Looks = int(1 / (1 - degradation_ratio))   # = 5
    N_range = ship.shape[1] // Number_of_Range_Looks

    # FFT along range only — azimuth axis untouched throughout
    ship_fft_range = np.fft.fftshift(np.fft.fft(ship, axis=1), axes=1)

    s_degraded = np.zeros((ship.shape[0], N_range), dtype=complex)
    for k in range(Number_of_Range_Looks):
        patch = ship_fft_range[:, k * N_range:(k + 1) * N_range]
        patch = np.fft.ifft(np.fft.ifftshift(patch, axes=1), axis=1)
        s_degraded += patch     # coherent sum of range sub-bands

    # Wrap-safe azimuth phase derivative
    d_phase = np.angle(s_degraded[1:] * np.conj(s_degraded[:-1]))
    return s_degraded, d_phase

def _phasor_ls_slope(
    phi: np.ndarray,
    w: np.ndarray,
    n_iter_gn: int = 2,
) -> float:
    """Weighted phasor least-squares slope of a wrapped phase signal.

    Fits  phi[n] ≈ a + b·n  (mod 2π)  by minimising

        J(a, b) = Σ w[n] · |exp(j·phi[n]) − exp(j·(a + b·n))|²
                = 2 · Σ w[n] · (1 − cos(phi[n] − a − b·n))

    over (a, b) and returns the slope `b`. The cost depends on `phi`
    only through sin/cos, so a real ±2π wrap of `phi` inside the run is
    indistinguishable from no wrap — no `np.unwrap` is involved.

    Two-stage fit:

      (A) Initial slope from Kay's weighted lag-1 estimator,

              b₀ = arg( Σ_n v[n] · e^{j·(phi[n+1] − phi[n])} ),
              v[n] = sqrt(w[n] · w[n+1]).

          At high SNR this is the maximum-likelihood slope of a
          linear-phase signal in Gaussian noise; the bright (high-w)
          pairs dominate the sum.

      (B) Newton refinement of (a, b) on the full weighted cost, with a
          centred parameterisation that decouples slope and intercept
          at the optimum. Two iterations are enough from the Kay init.

    Parameters
    ----------
    phi : (L,) float ndarray
        Wrapped phase samples (rad), in (-π, π].
    w : (L,) float ndarray, non-negative
        Per-sample weight. For phi = arg(s[n+1]·conj(s[n])) the
        natural choice is w[n] = |s[n]|·|s[n+1]|, i.e. the modulus of
        the complex number whose argument is phi[n]. Set entries to 0
        to drop noise-dominated samples entirely.
    n_iter_gn : int
        Number of Newton iterations to run after the Kay init. 0
        returns Kay's estimate alone.

    Returns
    -------
    b : float
        Slope estimate (rad/sample). 0.0 if the run is degenerate
        (length < 2 or all weights zero).
    """
    L = phi.size
    if L < 2:
        return 0.0
    w_total = float(w.sum())
    if w_total <= 0.0:
        return 0.0

    # --- Step A: weighted Kay's lag-1 slope -------------------------------
    v = np.sqrt(w[:-1] * w[1:])
    if v.sum() <= 0.0:
        return 0.0
    diff_phasor = np.exp(1j * (phi[1:] - phi[:-1]))
    b = float(np.angle((v * diff_phasor).sum()))

    if n_iter_gn <= 0:
        return b

    # Centred sample index — at the weighted centroid, slope and
    # intercept decouple in the Newton Hessian, which is numerically
    # much better behaved than the raw (a, b) parameterisation.
    n = np.arange(L, dtype=np.float64)
    n_bar = float((w * n).sum() / w_total)
    n_c = n - n_bar

    # Initial intercept at the centred origin from the weighted phasor
    # centroid evaluated at the current slope estimate.
    a_c = float(np.angle((w * np.exp(1j * (phi - b * n_c))).sum()))

    # --- Step B: Newton on J(a_c, b) = Σ w·(1 − cos(phi − a_c − b·n_c)) ---
    for _ in range(n_iter_gn):
        r = phi - a_c - b * n_c
        c = np.cos(r)
        s = np.sin(r)
        A  = float((w * c).sum())
        B  = float((w * n_c * c).sum())
        D  = float((w * n_c * n_c * c).sum())
        ga = float((w * s).sum())
        gb = float((w * n_c * s).sum())
        det = A * D - B * B
        # Bail out if the Hessian is not safely positive definite (can
        # happen far from optimum; here we trust the Kay init enough
        # that this is the only safety net we need).
        if A <= 0.0 or det <= 1e-12 * (abs(A) * abs(D) + 1e-30):
            break
        da = ( D * ga - B * gb) / det
        db = (-B * ga + A * gb) / det
        a_c += da
        b   += db

    return b


def _phasor_ls_slope_intercept(
    phi: np.ndarray,
    w: np.ndarray,
    n_iter_gn: int = 4,
) -> tuple[float, float]:
    """Wrap-immune weighted phasor LS that returns BOTH slope and intercept.

    Thin wrapper on :func:`_phasor_ls_slope` (which already does the
    wrap-immune optimisation but only returns the slope) that recovers
    the matching intercept at ``u = 0`` via the wrap-aware weighted
    circular mean

        intercept = arg( Σ_i w_i · exp(j · (phi_i - slope · i)) )    (∈ (-π, π])

    Rows can be excluded from the fit by setting their entry of ``w`` to
    zero — Kay's lag-1 estimator and the Newton step both ignore
    zero-weighted samples cleanly.

    Returns
    -------
    slope : float
        rad / sample; ``nan`` if degenerate (fewer than two non-zero
        weights, no consecutive non-zero pair for Kay's init, or the
        Newton step left a non-finite value).
    intercept : float
        rad, ∈ (-π, π]; ``nan`` if the fit was degenerate.
    """
    L = int(phi.size)
    if L < 2 or w.size != L:
        return float("nan"), float("nan")
    w_total = float(w.sum())
    if w_total <= 0.0:
        return float("nan"), float("nan")
    # Kay's init needs at least one consecutive pair with both weights > 0.
    if float(np.sqrt(w[:-1] * w[1:]).sum()) <= 0.0:
        return float("nan"), float("nan")
    slope = float(_phasor_ls_slope(phi, w, n_iter_gn=n_iter_gn))
    n = np.arange(L, dtype=np.float64)
    intercept = float(
        np.angle((w * np.exp(1j * (phi - slope * n))).sum())
    )
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return float("nan"), float("nan")
    return slope, intercept


def _min_distance_line_fit(
    phi: np.ndarray,
    u: np.ndarray,
    coh: np.ndarray,
    min_row_coherence: float = 0.5,
    inlier_tol_rad: float = 0.5,
    slope_max: float = 0.2,
    n_slope: int = 1001,
) -> tuple[float, float, int]:
    """Wrap-aware brute-force line fit by minimum wrapped-distance cost.

    For a wrapped phase trace ``phi(u)`` (sample indices ``u``), fits
    ``phi(u) ≈ slope·u + intercept`` by trying ``n_slope`` candidate
    slopes on an equispaced grid ``[-slope_max, +slope_max]`` and
    picking the candidate whose **wrapped** line is closest to the
    data — i.e. exactly the procedure asked for: "try 1000 different
    lines with incremental slopes, wrap if needed, find the Euclidean
    distance, and plot the lowest".

    Algorithm (fully vectorised, no Newton, no LS refit, no unwrap):

      1. For every candidate slope ``s``, the optimal intercept is the
         coh-weighted **circular** mean of ``phi - s·u`` evaluated at
         ``u = 0``::

             intercept(s) = arg( Σ_i coh[i] · exp(j·(phi[i] - s·u[i])) )

         This is the closed-form ``arg min_b Σ coh[i]·(1-cos(phi-s·u-b))``
         and is wrap-immune by construction.
      2. The per-sample wrapped distance to the candidate line is

             r[i] = arg( exp(j·(phi[i] - s·u[i] - intercept(s))) )
                  ∈ (-π, π]

         The candidate's **score** is the coh-weighted mean of
         ``|r[i]|`` over high-coh rows (rows with ``coh > min_row_coherence``;
         falls back to all rows if too few qualify)::

             score(s) = Σ_high coh[i]·|r[i]| / Σ_high coh[i]

         This is the "Euclidean wrapped distance" cost the user asked
         for, with the trivial improvement of weighting by coherence so
         pure-noise rows do not pull the optimum. It is robust to
         outliers (each row's contribution is bounded by π) and has no
         local-minimum pathology because we evaluate every candidate
         on the grid.
      3. The grid argmin is returned **directly** — no LS refit. The
         default grid step is ``2·slope_max / (n_slope-1) ≈ 4·10⁻⁴
         rad/sample`` which over a 1000-row patch resolves ``slope·n``
         to ~0.4 rad — well below ``2π``.

    Parameters
    ----------
    phi : (n,) float ndarray
        Wrapped phase samples in (-π, π].
    u : (n,) float ndarray
        Sample indices (e.g. ``np.arange(n)``).
    coh : (n,) float ndarray
        Non-negative per-sample weight in [0, 1] — the same per-row
        coherence ``compute_box_phase_estimates`` produces.
    min_row_coherence : float
        High-coh gate; only rows with ``coh > min_row_coherence``
        contribute to the score. If fewer than 2 rows qualify, the
        gate is lifted (all rows contribute). Default 0.5.
    inlier_tol_rad : float
        Diagnostic tolerance (rad) used **only** to report how many
        high-coh rows lie within ``±inlier_tol_rad`` of the chosen
        line. Does NOT affect slope selection. Default 0.5 (≈ ±29°).
    slope_max : float
        Half-width of the slope search range (rad / sample).
        Default 0.2.
    n_slope : int
        Number of equispaced slope candidates. Default 1001.

    Returns
    -------
    slope : float
        Grid-argmin slope (rad / sample); ``nan`` on degenerate input.
    intercept : float
        Optimal intercept at ``u = 0`` for that slope, in (-π, π];
        ``nan`` on degenerate input.
    n_inliers : int
        Diagnostic: number of high-coh rows within ``inlier_tol_rad``
        of the chosen line. 0 on degenerate input.
    """
    n = int(phi.size)
    if n < 2 or u.size != n or coh.size != n:
        return float("nan"), float("nan"), 0
    if float(coh.sum()) <= 0.0:
        return float("nan"), float("nan"), 0

    high_coh_mask = coh > min_row_coherence
    # Fall back to all rows if the coh gate is too aggressive for this box;
    # otherwise the score would be undefined or dominated by ≤1 row.
    if int(high_coh_mask.sum()) < 2:
        high_coh_mask = np.ones(n, dtype=bool)

    slopes = np.linspace(-slope_max, slope_max, n_slope)
    # phi - slope·u, broadcast to (n_slope, n)
    phi_minus_su = phi[None, :] - slopes[:, None] * u[None, :]
    # Optimal intercept per slope: wrap-aware coh-weighted circular mean.
    z = (coh[None, :] * np.exp(1j * phi_minus_su)).sum(axis=1)
    intercepts = np.angle(z)
    # Wrapped distance from every sample to every candidate line.
    r = np.angle(np.exp(1j * (phi_minus_su - intercepts[:, None])))
    abs_r = np.abs(r)

    # Cost: coh-weighted mean |wrapped r| over high-coh rows
    #         = Σ_high coh[i]·|r[i]| / Σ_high coh[i]
    w_row = np.where(high_coh_mask, coh, 0.0)
    denom = float(w_row.sum())
    if denom <= 0.0:
        return float("nan"), float("nan"), 0
    scores = (w_row[None, :] * abs_r).sum(axis=1) / denom
    best = int(np.argmin(scores))
    s_best = float(slopes[best])
    b_best = float(intercepts[best])
    # Diagnostic inlier count for plot titles / CSV; not used for selection.
    n_in_best = int(((abs_r[best] <= inlier_tol_rad) & high_coh_mask).sum())
    return s_best, b_best, n_in_best


def estimate_phase_slope(
    d_phase: np.ndarray,
    window_length: int,
    weights: np.ndarray | None = None,
    n_iter_gn: int = 2,
) -> np.ndarray:
    """
    Per-range-column slope of `d_phase` over runs of non-zero samples in
    azimuth, using a weighted phasor LS fit that is wrap-immune by
    construction.

    For each range column `r`:
      1. Find maximal runs of indices where `d_phase[:, r] != 0`.
      2. Discard runs shorter than `0.5 * window_length` azimuth pixels.
      3. On every kept run, fit  d_phase[n] ≈ a + b·n  (mod 2π) by
         minimising  Σ w[n]·(1 − cos(d_phase[n] − a − b·n))  via
         `_phasor_ls_slope`. The slope `b` (rad/pixel) is written into
         the output map at the run's positions; pixels not in any kept
         run remain 0.

    The fit is initialised by Kay's weighted lag-1 estimator and then
    refined with a couple of Newton steps on the weighted cosine cost.
    Both stages depend on the wrapped phase only through sin/cos, so a
    real ±2π wrap inside the run produces the same contribution as no
    wrap — no `np.unwrap` is used and isolated noise excursions across
    ±π cannot inject spurious 2π/L slope bias.

    Parameters
    ----------
    d_phase : (N_az, N_rg) float ndarray
        Wrapped azimuth phase derivative (rad). Zeros mark masked-out
        pixels and define the run boundaries.
    window_length : int
        Minimum kept run length is ⌊0.5·window_length⌋.
    weights : (N_az, N_rg) float ndarray, optional
        Per-pixel non-negative weight for the fit. For
            d_phase = arg(s_degraded[1:] · conj(s_degraded[:-1]))
        the natural choice is
            weights = |s_degraded[:-1]| * |s_degraded[1:]|     (same shape)
        i.e. the modulus of the complex product whose argument was
        taken. Multiplying by an amplitude / coherence mask, or zeroing
        out samples below a threshold, lets the caller keep low-SNR
        pixels out of the fit without changing this routine. If None,
        all kept samples receive equal weight (still wrap-immune).
    n_iter_gn : int
        Number of Newton refinement iterations after Kay's init.
        Default 2. Set to 0 to use Kay's estimator alone.

    Returns
    -------
    slope_map : (N_az, N_rg) float ndarray
        Slope `b` in rad/pixel at every position belonging to a kept
        run; 0 elsewhere.

    Notes
    -----
    * "Non-zero" is interpreted strictly: the upstream CoV gate produces
      exact zeros at masked pixels, which propagate into d_phase via
      arg(0 · conj(·)) = 0. The run boundaries follow that mask.
    * The (unresolved) global 2π ambiguity in `b` is the same as in any
      single-lag Doppler estimator: if a target really has |b·L| > 2π
      the estimate folds. Use a shorter degradation factor / smaller
      kernel if that regime matters.
    """
    n_az, n_rg = d_phase.shape
    if weights is not None:
        assert weights.shape == d_phase.shape, (weights.shape, d_phase.shape)
        w_full = np.asarray(weights, dtype=np.float64)
    else:
        w_full = None
    min_len = max(2, int(0.5 * window_length))
    slope_map = np.zeros_like(d_phase, dtype=float)

    n_runs_total = 0
    n_runs_kept = 0
    for r in range(n_rg):
        col = d_phase[:, r]
        nz = col != 0
        if not nz.any():
            continue
        # Run boundaries: indices where nz transitions, plus the array ends.
        change = np.flatnonzero(np.diff(nz.astype(np.int8))) + 1
        edges = np.concatenate([[0], change, [n_az]])
        for s, e in zip(edges[:-1], edges[1:]):
            if not nz[s]:
                continue              # this segment is zeros
            n_runs_total += 1
            if (e - s) < min_len:
                continue
            phi = col[s:e].astype(np.float64, copy=False)
            if w_full is not None:
                w = w_full[s:e, r]
            else:
                w = np.ones(e - s, dtype=np.float64)
            slope = _phasor_ls_slope(phi, w, n_iter_gn=n_iter_gn)
            slope_map[s:e, r] = slope
            n_runs_kept += 1

    print(
        f"  estimate_phase_slope: d_phase {d_phase.shape}, "
        f"window_length={window_length}, min_len={min_len}, "
        f"runs kept {n_runs_kept}/{n_runs_total}, "
        f"weighted={'yes' if w_full is not None else 'no'}, "
        f"n_iter_gn={n_iter_gn}"
    )
    return slope_map


def compute_box_slope_col_total(
    boxes_yxhw: np.ndarray,
    slope_map: np.ndarray,
) -> np.ndarray:
    """Per-box aggregate of `slope_map` along each range column.

    For each box, for each range column the box covers, find the
    longest contiguous non-zero run of ``slope_map`` inside the box's
    azimuth row range, and compute ``|slope[run_start] · run_length|``.
    The per-box score is the **maximum** of that quantity over the
    box's columns.

    Rationale: a large moving target whose single-line wrap-aware fit
    (`_min_distance_line_fit`) reports a below-threshold
    ``|slope · n_rows|`` because the box straddles a phase wrap or
    contains heterogeneous scatterers can still have one or more
    columns with a clean, wrap-immune phasor-LS slope (`slope_map`)
    above threshold. This aggregate captures that.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int ndarray
        Box centres + sizes ``[y_c, x_c, h, w]`` in `s_degraded`
        coordinates (axis 0 = azimuth row, axis 1 = range column).
    slope_map : (N_az - 1, N_rg) float ndarray
        Per-pixel slope (rad/row) from :func:`estimate_phase_slope`;
        zero outside any kept column-run.

    Returns
    -------
    slope_col_total_rad : (K,) float ndarray
        ``max_c |slope_c · L_c|`` over the box's covered columns,
        where ``L_c`` is the length of the longest non-zero run in
        column ``c`` inside the box's azimuth range and ``slope_c``
        is the (constant) slope on that run.
    """
    K = len(boxes_yxhw)
    out = np.zeros(K, dtype=np.float64)
    if K == 0 or slope_map.size == 0:
        return out
    n_az, n_rg = slope_map.shape
    for i, (y_c, x_c, h, w) in enumerate(np.atleast_2d(boxes_yxhw)):
        y0 = max(0, int(y_c) - int(h) // 2)
        y1 = min(n_az, y0 + int(h))
        x0 = max(0, int(x_c) - int(w) // 2)
        x1 = min(n_rg, x0 + int(w))
        if y1 <= y0 or x1 <= x0:
            continue
        best = 0.0
        for c in range(x0, x1):
            col = slope_map[y0:y1, c]
            nz = col != 0
            if not nz.any():
                continue
            change = np.flatnonzero(np.diff(nz.astype(np.int8))) + 1
            edges = np.concatenate([[0], change, [len(col)]])
            best_col = 0.0
            for s, e in zip(edges[:-1], edges[1:]):
                if not nz[s]:
                    continue
                L = e - s
                score = abs(float(col[s])) * L
                if score > best_col:
                    best_col = score
            if best_col > best:
                best = best_col
        out[i] = best
    return out


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _overlay_boxes(
    ax: plt.Axes,
    boxes_yxhw: np.ndarray,
    color: str = "cyan",
    lw: float = 0.8,
) -> None:
    """Draw axis-aligned rectangles, one per row of `boxes_yxhw`.

    boxes_yxhw : (K, 4) array of [(y_centre, x_centre, h, w), …] in pixel
                 coords (axis 0 = azimuth row, axis 1 = range column).
    """
    for y, x, h, w in np.atleast_2d(boxes_yxhw):
        ax.add_patch(plt.Rectangle(
            (x - w / 2 - 0.5, y - h / 2 - 0.5), w, h,
            fill=False, edgecolor=color, linewidth=lw,
        ))


def merge_overlapping_boxes(boxes_yxhw: np.ndarray) -> np.ndarray:
    """Merge any axis-aligned boxes whose rectangles intersect.

    Builds a graph where i and j are linked iff their boxes overlap by
    even one pixel (open-interval intersection > 0), then replaces each
    connected component with the union bounding box of its members.

    Useful as a final pass after IoU-NMS / distance-NMS / bridge-merge:
    earlier stages can leave touching-or-overlapping boxes (IoU > 0 but
    ≤ iou_thresh, or a bridge-merged union now overlapping a previously
    untouched neighbour).
    """
    n = len(boxes_yxhw)
    if n < 2:
        return boxes_yxhw.copy()

    yc = boxes_yxhw[:, 0].astype(np.int64)
    xc = boxes_yxhw[:, 1].astype(np.int64)
    h = boxes_yxhw[:, 2].astype(np.int64)
    w = boxes_yxhw[:, 3].astype(np.int64)
    y_lo = yc - h // 2
    y_hi = y_lo + h
    x_lo = xc - w // 2
    x_hi = x_lo + w

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if (y_lo[i] < y_hi[j] and y_lo[j] < y_hi[i]
                    and x_lo[i] < x_hi[j] and x_lo[j] < x_hi[i]):
                union(i, j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    out = []
    for members in comps.values():
        if len(members) == 1:
            out.append(boxes_yxhw[members[0]])
            continue
        ys_lo = int(y_lo[members].min())
        ys_hi = int(y_hi[members].max())
        xs_lo = int(x_lo[members].min())
        xs_hi = int(x_hi[members].max())
        out.append(np.array([
            (ys_lo + ys_hi) // 2,
            (xs_lo + xs_hi) // 2,
            ys_hi - ys_lo,
            xs_hi - xs_lo,
        ], dtype=np.int64))
    return np.array(out, dtype=np.int64)


def merge_by_strip_amplitude(
    boxes_yxhw: np.ndarray,
    amp_raw: np.ndarray,
    az_spacing_m: float,
    rg_spacing_m: float,
    bridge_strength: float = 2.0,
    max_rg_offset_m: float = 20.0,
    max_az_gap_m: float = 2000.0,
) -> np.ndarray:
    """Bridge-merge boxes joined by a high-amplitude trail in `amp_raw`.

    Designed to merge multiple boxes that fall on the same physical
    moving target whose azimuth signature has been split by gaps in the
    CoV mask, but whose unthresholded amplitude is continuous.

    Two boxes (i, j) are linked iff:
      • |Δrg|·rg_spacing_m ≤ `max_rg_offset_m`
        (i.e. they share a range column to within one target's width),
      • |Δaz|·az_spacing_m ≤ `max_az_gap_m`,
      • median amplitude of the strip joining them along azimuth in
        `amp_raw` is at least
        `bridge_strength × max(amp_raw[centre_i], amp_raw[centre_j])`.

    Why max + median? `max` means the strip must be a comparable fraction
    of the *brighter* endpoint, so a faint box can't pull a bright box
    into a merge by anchoring on a dim trail. `median` means at least
    half the strip pixels clear the threshold — a single bright outlier
    can't carry the bridge the way a mean would.

    Connected components in this link graph are replaced with their
    union bounding box. Singletons pass through unchanged. The strip
    spans the *azimuth gap* between the two boxes' inner edges, with
    range columns set to the union of both boxes' range extents.
    """
    n = len(boxes_yxhw)
    if n < 2:
        return boxes_yxhw.copy()

    H, W = amp_raw.shape
    yc = boxes_yxhw[:, 0].astype(np.int64)
    xc = boxes_yxhw[:, 1].astype(np.int64)
    h = boxes_yxhw[:, 2].astype(np.int64)
    w = boxes_yxhw[:, 3].astype(np.int64)
    centre_amp = amp_raw[np.clip(yc, 0, H - 1), np.clip(xc, 0, W - 1)]

    y_lo_arr = np.clip(yc - h // 2, 0, H)
    y_hi_arr = np.clip(y_lo_arr + h, 0, H)
    x_lo_arr = np.clip(xc - w // 2, 0, W)
    x_hi_arr = np.clip(x_lo_arr + w, 0, W)

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_links = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_rg_m = abs(int(xc[i]) - int(xc[j])) * rg_spacing_m
            if d_rg_m > max_rg_offset_m:
                continue
            d_az_m = abs(int(yc[i]) - int(yc[j])) * az_spacing_m
            if d_az_m > max_az_gap_m:
                continue
            # Strip = azimuth gap between the two boxes' inner edges.
            strip_top = min(int(y_hi_arr[i]), int(y_hi_arr[j]))
            strip_bot = max(int(y_lo_arr[i]), int(y_lo_arr[j]))
            if strip_top >= strip_bot:
                # Boxes overlap in azimuth — IoU NMS already kept both,
                # so they're not redundant by overlap. Skip; let other
                # pairs link them transitively if they really belong
                # together.
                continue
            sx_lo = min(int(x_lo_arr[i]), int(x_lo_arr[j]))
            sx_hi = max(int(x_hi_arr[i]), int(x_hi_arr[j]))
            strip = amp_raw[strip_top:strip_bot, sx_lo:sx_hi]
            if strip.size == 0:
                continue
            anchor = bridge_strength * max(float(centre_amp[i]),
                                           float(centre_amp[j]))
            if float(np.median(strip)) > anchor:
                union(i, j)
                n_links += 1

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    out = []
    for members in comps.values():
        if len(members) == 1:
            out.append(boxes_yxhw[members[0]])
            continue
        ys_lo = int(y_lo_arr[members].min())
        ys_hi = int(y_hi_arr[members].max())
        xs_lo = int(x_lo_arr[members].min())
        xs_hi = int(x_hi_arr[members].max())
        out.append(np.array([
            (ys_lo + ys_hi) // 2,
            (xs_lo + xs_hi) // 2,
            ys_hi - ys_lo,
            xs_hi - xs_lo,
        ], dtype=np.int64))
    return np.array(out, dtype=np.int64)


def nms_by_centre_distance(
    boxes_yxhw: np.ndarray,
    scores: np.ndarray,
    az_spacing_m: float,
    rg_spacing_m: float,
    max_dist_az_m: float,
    max_dist_rg_m: float,
) -> np.ndarray:
    """Greedy centre-distance suppression in *metres*, anisotropic ellipse.

    Walk boxes from highest score down. Keep each one and suppress every
    later box whose centre lies inside the axis-aligned ellipse

        (Δaz_m / max_dist_az_m)^2 + (Δrg_m / max_dist_rg_m)^2 < 1

    around any kept centre. The ellipse is the right shape for this
    domain: a SAR moving target's image signature is long along azimuth
    (Doppler smear) but stays near-pointlike in range, so legitimate
    "same target → multiple boxes" duplicates have small Δrg even when
    Δaz is large, while genuinely-distinct targets are typically well
    separated in range. A single circular radius can't capture both.

    Use `max_dist_az_m` ≈ a few × max target length (e.g. 1000 m) and
    `max_dist_rg_m` ≈ max target width (e.g. 20 m).

    Returns indices into the original `boxes_yxhw` (score-descending).
    """
    if len(boxes_yxhw) == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-scores)
    yxhw = boxes_yxhw[order]
    suppressed = np.zeros(len(yxhw), dtype=bool)
    keep = []
    inv_az_sq = 1.0 / float(max_dist_az_m) ** 2
    inv_rg_sq = 1.0 / float(max_dist_rg_m) ** 2
    for i in range(len(yxhw)):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 == len(yxhw):
            continue
        dy_m = (yxhw[i + 1:, 0] - yxhw[i, 0]) * az_spacing_m
        dx_m = (yxhw[i + 1:, 1] - yxhw[i, 1]) * rg_spacing_m
        norm = dy_m * dy_m * inv_az_sq + dx_m * dx_m * inv_rg_sq
        suppressed[i + 1:][norm < 1.0] = True
    return order[np.asarray(keep, dtype=np.int64)]


def nms_boxes(
    boxes_yxhw: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.3,
) -> np.ndarray:
    """Greedy non-maximum suppression of axis-aligned boxes.

    Walk boxes in descending `scores`. Keep each one and suppress every
    later box whose IoU with the kept one exceeds `iou_thresh`.

    Parameters
    ----------
    boxes_yxhw : (K, 4) array of (y_centre, x_centre, h, w).
    scores     : (K,)   per-box score (higher = better, kept first).
    iou_thresh : float  IoU above which a later box is dropped.

    Returns
    -------
    keep_idx : 1-D int array of indices into the *original* `boxes_yxhw`
               (in score-descending order).
    """
    if len(boxes_yxhw) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    boxes = boxes_yxhw[order].astype(np.float64)
    y_c, x_c, h, w = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    y_lo, y_hi = y_c - h / 2, y_c + h / 2
    x_lo, x_hi = x_c - w / 2, x_c + w / 2
    areas = h * w

    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        # Compute IoU of box i vs every still-alive box j > i.
        j = np.flatnonzero(~suppressed)
        j = j[j > i]
        if not j.size:
            continue
        oy = np.maximum(0.0, np.minimum(y_hi[i], y_hi[j]) - np.maximum(y_lo[i], y_lo[j]))
        ox = np.maximum(0.0, np.minimum(x_hi[i], x_hi[j]) - np.maximum(x_lo[i], x_lo[j]))
        inter = oy * ox
        iou = inter / np.maximum(areas[i] + areas[j] - inter, 1e-12)
        suppressed[j[iou > iou_thresh]] = True

    return order[np.asarray(keep, dtype=np.int64)]


def grow_and_recenter_boxes(
    peaks_yx: np.ndarray,
    mask: np.ndarray,
    amp: np.ndarray,
    initial_hw: tuple[int, int],
    az_step: int = 5,
    rg_step: int = 1,
    density_threshold: float = 0.25,
    az_tail: int = 20,
    rg_tail: int = 1,
    az_rescue_lookahead: int = 50,
    az_rescue_lookback: int = 50,
    az_rescue_threshold: float = 0.5,
    max_h: int | None = None,
    max_w: int | None = None,
    max_iter: int = 2000,
) -> np.ndarray:
    """Grow each peak's box per-edge — azimuth first, then range — and
    recenter on the amplitude centre-of-mass each step.

    Box state is tracked as edges (y_lo, y_hi, x_lo, x_hi); each of the
    four edges can expand independently. The box's centre is just the
    midpoint of its current bounds.

    Algorithm (per peak):
      1. Start with a box of size `initial_hw` centred on the peak.
      2. Each iteration (two-phase):
         (Phase 1 — azimuth)
         a. Try shifting the top edge:    y_lo ← y_lo - az_step.
         b. Try shifting the bottom edge: y_hi ← y_hi + az_step.
         (Phase 2 — range, only once both top AND bottom are dead)
         c. Try shifting the left edge:   x_lo ← x_lo - rg_step.
         d. Try shifting the right edge:  x_hi ← x_hi + rg_step.
            Each proposal is accepted iff the (image-clipped) box still
            has `mask` density ≥ `density_threshold`. A failed edge is
            "killed" — not retried in subsequent iterations.
         e. If no edge grew this iteration, stop.
         f. Recenter: shift bounds so the box (size h × w preserved) is
            centred on the amplitude-weighted CoM inside it.
         g. If post-recentre density falls below `density_threshold`, stop.

    Phase ordering (azimuth-first) keeps the box thin in range while
    azimuth grows, so the azimuth density check isn't diluted by a wide
    range strip that may contain partial-target columns.

    Returns
    -------
    boxes_yxhw : (K, 4) int array — final (y_c, x_c, h, w) per peak.
    """
    H, W = mask.shape
    h0, w0 = initial_hw
    out = np.zeros((len(peaks_yx), 4), dtype=np.int64)

    def _density_ok(yl, yh, xl, xh, threshold=None):
        """True iff `mask[yl:yh, xl:xh]` density ≥ threshold.

        Used for the per-edge GROWTH test (strip-only), the post-recentre
        FULL-BOX test, and the wider rescue look-ahead test (with a higher
        threshold). When `threshold` is None, falls back to the function's
        global `density_threshold` parameter.
        """
        sub = mask[yl:yh, xl:xh]
        if sub.size == 0:
            return False
        th = density_threshold if threshold is None else threshold
        return float(sub.sum()) >= th * sub.size

    for k, (y0, x0) in enumerate(peaks_yx):
        # Initial bounds centred on the seed peak, clipped to the image.
        y_lo = max(0, int(y0) - h0 // 2)
        y_hi = min(H, y_lo + h0)
        x_lo = max(0, int(x0) - w0 // 2)
        x_hi = min(W, x_lo + w0)

        # Edge proposals are tested fresh each iteration. Az edges are
        # primary (tried every iter); range only advances when az is
        # stuck on the current iter, after which the next iter retries
        # az with the wider strip — so a brief az-density dip can be
        # bridged by widening range.

        for _ in range(max_iter):
            # Each edge test runs on the *new strip plus a tail* of the
            # box's already-grown interior on the same side (az_tail rows
            # for top/bottom, rg_tail cols for left/right). The tail is
            # clamped so it never crosses the opposite edge.

            # ---- Phase A: try both azimuth edges ----------------------
            # The primary strip test (next az_step + last az_tail rows)
            # is permissive (density_threshold). If it fails on an edge,
            # a rescue test peeks much further ahead (az_rescue_lookahead
            # rows outside the box) plus a long tail (az_rescue_lookback
            # rows of the box's interior on the same side), with a
            # stricter threshold (az_rescue_threshold). The rescue is what
            # bridges sparse az dropouts inside an otherwise dense target.
            grew_az = False

            # (a) Top edge: strip = rows [ny_lo, y_lo + az_tail], cols [x_lo, x_hi]
            ny_lo = y_lo - az_step
            new_h = y_hi - ny_lo
            over_cap = max_h is not None and new_h > max_h
            if not over_cap and ny_lo >= 0:
                tail_hi = min(y_lo + az_tail, y_hi)
                if _density_ok(ny_lo, tail_hi, x_lo, x_hi):
                    y_lo = ny_lo
                    grew_az = True
                else:
                    # Rescue: wider look-ahead + longer tail, stricter th.
                    look_lo = max(y_lo - az_rescue_lookahead, 0)
                    tail_hi_resc = min(y_lo + az_rescue_lookback, y_hi)
                    if _density_ok(look_lo, tail_hi_resc, x_lo, x_hi,
                                   threshold=az_rescue_threshold):
                        y_lo = ny_lo
                        grew_az = True

            # (b) Bottom edge: strip = rows [y_hi - az_tail, ny_hi], cols [x_lo, x_hi]
            ny_hi = y_hi + az_step
            new_h = ny_hi - y_lo
            over_cap = max_h is not None and new_h > max_h
            if not over_cap and ny_hi <= H:
                tail_lo = max(y_hi - az_tail, y_lo)
                if _density_ok(tail_lo, ny_hi, x_lo, x_hi):
                    y_hi = ny_hi
                    grew_az = True
                else:
                    # Rescue: wider look-ahead + longer tail, stricter th.
                    look_hi = min(y_hi + az_rescue_lookahead, H)
                    tail_lo_resc = max(y_hi - az_rescue_lookback, y_lo)
                    if _density_ok(tail_lo_resc, look_hi, x_lo, x_hi,
                                   threshold=az_rescue_threshold):
                        y_hi = ny_hi
                        grew_az = True

            # ---- Phase B: only if az did NOT advance, take ONE range step.
            # This widens the box and lets next iter retry az with a wider
            # strip (and thus more cells per row to satisfy the density check).
            grew_rg = False
            if not grew_az:
                # (c) Left edge: strip = rows [y_lo, y_hi], cols [nx_lo, x_lo + rg_tail]
                nx_lo = x_lo - rg_step
                new_w = x_hi - nx_lo
                over_cap = max_w is not None and new_w > max_w
                tail_hi_x = min(x_lo + rg_tail, x_hi)
                if (not over_cap and nx_lo >= 0
                        and _density_ok(y_lo, y_hi, nx_lo, tail_hi_x)):
                    x_lo = nx_lo
                    grew_rg = True

                # (d) Right edge: strip = rows [y_lo, y_hi], cols [x_hi - rg_tail, nx_hi]
                nx_hi = x_hi + rg_step
                new_w = nx_hi - x_lo
                over_cap = max_w is not None and new_w > max_w
                tail_lo_x = max(x_hi - rg_tail, x_lo)
                if (not over_cap and nx_hi <= W
                        and _density_ok(y_lo, y_hi, tail_lo_x, nx_hi)):
                    x_hi = nx_hi
                    grew_rg = True

            # (e) Neither az nor rg can advance → done.
            if not (grew_az or grew_rg):
                break

            # (f) Recenter: shift bounds so they're centred on amp CoM,
            # preserving the current h, w.
            h_cur = y_hi - y_lo
            w_cur = x_hi - x_lo
            sub_amp = amp[y_lo:y_hi, x_lo:x_hi]
            total = float(sub_amp.sum())
            if total > 0:
                row_marg = sub_amp.sum(axis=1)
                col_marg = sub_amp.sum(axis=0)
                yy = np.arange(sub_amp.shape[0])
                xx = np.arange(sub_amp.shape[1])
                dy = float((row_marg * yy).sum() / total)
                dx = float((col_marg * xx).sum() / total)
                y_com = y_lo + int(round(dy))
                x_com = x_lo + int(round(dx))
                # Re-derive bounds, clipping to image but keeping h_cur, w_cur
                # if at all possible.
                y_lo = max(0, y_com - h_cur // 2)
                y_hi = min(H, y_lo + h_cur)
                y_lo = max(0, y_hi - h_cur)
                x_lo = max(0, x_com - w_cur // 2)
                x_hi = min(W, x_lo + w_cur)
                x_lo = max(0, x_hi - w_cur)

            # (g) Post-recentre density check.
            if not _density_ok(y_lo, y_hi, x_lo, x_hi):
                break

        h_final = y_hi - y_lo
        w_final = x_hi - x_lo
        y_c = (y_lo + y_hi) // 2
        x_c = (x_lo + x_hi) // 2
        out[k] = (y_c, x_c, h_final, w_final)

    return out


def extend_boxes_azimuth_strong_signal(
    boxes_yxhw: np.ndarray,
    mask: np.ndarray,
    *,
    step_frac: float = 0.05,
    max_steps: int = 2,
    density_threshold: float = 0.25,
    max_h: int | None = None,
) -> np.ndarray:
    """Post-growth azimuth-only extension of each box's top and bottom edges.

    After `grow_and_recenter_boxes` finalises a box, the density gate may
    have stopped a fraction below the loose-grow threshold even though the
    target's signature continues a little further in azimuth. For each box
    we therefore try, top and bottom independently, to expand the azimuth
    edge by ``step_frac`` of the **entry** box height (e.g. 5%). The trial
    is repeated up to ``max_steps`` times per side (e.g. 2 × 5% = 10%
    cap per side), and accepted iff the detection-mask density inside the
    candidate strip ``mask[strip_rows, x_lo:x_hi]`` is at least
    ``density_threshold`` (same loose 0.25 used during growth). The first
    failed step kills further extension on that side — we don't try to
    bridge gaps here; that's what the growth-phase rescue look-ahead is
    for.

    The box's range bounds and width are untouched. Range-direction
    extension is intentionally out of scope.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int array of (y_centre, x_centre, h, w).
    mask : 2-D detection mask (0/1) aligned with the boxes.
    step_frac : fractional step size, expressed as a fraction of the
        box's height **at entry** to this function (default 0.05 = 5%).
        Using the entry height keeps the total extension per side at
        exactly ``step_frac * max_steps`` of the original height,
        regardless of intermediate acceptances.
    max_steps : maximum number of steps per side (default 2).
    density_threshold : minimum fraction of mask==1 pixels in the
        candidate strip required to accept the step (default 0.25).
    max_h : optional global cap on box height in pixels; extension is
        skipped if it would push the box above this cap.

    Returns
    -------
    out : (K, 4) int array — boxes with possibly enlarged height and
        shifted centre (centre is recomputed from the new bounds).
    """
    if len(boxes_yxhw) == 0:
        return boxes_yxhw
    H, W = mask.shape
    out = np.array(boxes_yxhw, dtype=np.int64, copy=True)
    n_extended_top = 0
    n_extended_bot = 0
    total_extra_h = 0
    for k, (y_c, x_c, h, w) in enumerate(out):
        h_int, w_int = int(h), int(w)
        y_lo = max(0, int(y_c) - h_int // 2)
        y_hi = min(H, y_lo + h_int)
        x_lo = max(0, int(x_c) - w_int // 2)
        x_hi = min(W, x_lo + w_int)

        # Step is sized from the box's entry height so the total cap
        # per side equals exactly step_frac * max_steps of that height.
        step = max(1, int(round(step_frac * h_int)))

        # Top edge (smaller y).
        for _ in range(max_steps):
            ny_lo = y_lo - step
            if ny_lo < 0:
                break
            if max_h is not None and (y_hi - ny_lo) > max_h:
                break
            strip = mask[ny_lo:y_lo, x_lo:x_hi]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_lo = ny_lo
                n_extended_top += 1
            else:
                break

        # Bottom edge (larger y).
        for _ in range(max_steps):
            ny_hi = y_hi + step
            if ny_hi > H:
                break
            if max_h is not None and (ny_hi - y_lo) > max_h:
                break
            strip = mask[y_hi:ny_hi, x_lo:x_hi]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_hi = ny_hi
                n_extended_bot += 1
            else:
                break

        h_new = y_hi - y_lo
        w_new = x_hi - x_lo
        y_c_new = (y_lo + y_hi) // 2
        x_c_new = (x_lo + x_hi) // 2
        total_extra_h += (h_new - h_int)
        out[k] = (y_c_new, x_c_new, h_new, w_new)

    print(
        f"  az-extend: step={step_frac:.0%}×h, max_steps={max_steps}, "
        f"density≥{density_threshold:g}: "
        f"{n_extended_top}/{len(out)} top-steps, "
        f"{n_extended_bot}/{len(out)} bottom-steps, "
        f"total +{total_extra_h} az rows across all boxes"
    )
    return out


def _show_slc(ax: plt.Axes, s: np.ndarray, title: str, sigma: float = 4.0) -> None:
    """Display complex SLC amplitude with mean+`sigma`·σ clip."""
    img = np.abs(s)
    mu, sd = float(img.mean()), float(img.std())
    vmax = min(float(img.max()), mu + sigma * sd)
    vmin = max(float(img.min()), mu - sigma * sd)
    im = ax.imshow(img, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    plt.colorbar(im, ax=ax, label="|s|")


def _show_phase_derivative(ax: plt.Axes, d: np.ndarray, title: str) -> None:
    """Display a real phase array on the cyclic [-pi, pi] twilight colormap."""
    im = ax.imshow(d, cmap="twilight", aspect="auto", vmin=-np.pi, vmax=np.pi)
    ax.set_title(title)
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    plt.colorbar(im, ax=ax, label="rad")


def _show_detection_map(ax: plt.Axes, det: np.ndarray, title: str) -> None:
    """Display detection score map with robust 99th-percentile clip."""
    vmax = float(np.percentile(det, 99))
    im = ax.imshow(det, cmap="hot", aspect="auto", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    plt.colorbar(im, ax=ax, label="detection score")


def _show_image_domain_phase_derivative(ax: plt.Axes, s: np.ndarray) -> None:
    """Compute and display the azimuth phase derivative directly on the input SLC."""
    d = np.angle(s[1:, :] * np.conj(s[:-1, :]))
    _show_phase_derivative(
        ax, d,
        r"$\partial\varphi/\partial u$ of input SLC" + "\narg{s[u+1]·s*[u]}  (image domain)"
    )

from scipy.ndimage import uniform_filter

def compute_subaperture_coherence(
    s: np.ndarray,
    overlap: float = 0.5,
    window: tuple[int, int] = (7, 7),
) -> np.ndarray:
    """Local CCD coherence between two azimuth sub-apertures."""
    N_az = s.shape[0]
    S = np.fft.fft(s, axis=0)

    half = N_az // 2
    keep = int(half * (1.0 + overlap))   # how many bins each look spans
    sub1 = np.zeros_like(S); sub1[:keep] = S[:keep]
    sub2 = np.zeros_like(S); sub2[-keep:] = S[-keep:]

    s1 = np.fft.ifft(sub1, axis=0)
    s2 = np.fft.ifft(sub2, axis=0)

    cross = uniform_filter((s1 * np.conj(s2)).real, size=window) + 1j * \
            uniform_filter((s1 * np.conj(s2)).imag, size=window)
    pow1 = uniform_filter(np.abs(s1) ** 2, size=window)
    pow2 = uniform_filter(np.abs(s2) ** 2, size=window)
    return np.abs(cross) / np.sqrt(pow1 * pow2 + 1e-12)


def compute_subaperture_com_per_box(
    boxes_yxhw: np.ndarray,
    subapertures: np.ndarray,
    N_subaperture: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-box, per-subaperture amplitude-weighted centre of mass.

    For each detection box (in ``s_degraded`` pixel coordinates), the box is
    projected onto each subaperture's grid by dividing its azimuth centre
    and height by ``N_subaperture`` (the range axis is unchanged because
    subapertures share the range axis with ``s_degraded``). Inside the
    projected window, the centre of mass is computed on ``|subapertures[j]|``
    — i.e. on the amplitude data — for every Doppler look ``j``.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int array
        ``[(y_centre, x_centre, h, w), …]`` in ``s_degraded`` pixels.
    subapertures : (N_sub, sub_size, N_range_d) ndarray
        Subaperture amplitudes returned by :func:`compute_subapertures`.
    N_subaperture : int
        Number of Doppler sub-bands (== ``subapertures.shape[0]``).

    Returns
    -------
    com_az_sub : (K, N_sub) float64 ndarray
        Azimuth centre of mass per box per subaperture, in subaperture
        pixel coordinates (``0 <= com_az_sub < sub_size``). NaN where the
        cropped window has zero amplitude.
    com_rg : (K, N_sub) float64 ndarray
        Range centre of mass per box per subaperture, in ``s_degraded``
        range pixels (range is not decimated by subaperturing). NaN where
        the cropped window has zero amplitude.
    """
    n_sub, sub_size, n_rg_d = subapertures.shape
    K = len(boxes_yxhw)
    com_az = np.full((K, n_sub), np.nan, dtype=np.float64)
    com_rg = np.full((K, n_sub), np.nan, dtype=np.float64)
    if K == 0:
        return com_az, com_rg

    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        # Project the box into the subaperture grid (azimuth ÷ N_subaperture).
        y0_sub = max(0, int(round((y_c - h / 2) / N_subaperture)))
        y1_sub = min(sub_size, int(round((y_c + h / 2) / N_subaperture)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg_d, int(round(x_c + w / 2)))
        if y1_sub - y0_sub < 1 or x1 - x0 < 1:
            continue

        az_idx = np.arange(y0_sub, y1_sub, dtype=np.float64)
        rg_idx = np.arange(x0, x1, dtype=np.float64)
        for j in range(n_sub):
            img = np.abs(subapertures[j, y0_sub:y1_sub, x0:x1]).astype(np.float64)
            total = float(img.sum())
            if total <= 0.0 or not np.isfinite(total):
                continue
            com_az[k, j] = float((img.sum(axis=1) * az_idx).sum() / total)
            com_rg[k, j] = float((img.sum(axis=0) * rg_idx).sum() / total)
    return com_az, com_rg


def compute_subaperture_contrast_gain_per_box(
    boxes_yxhw: np.ndarray,
    s_degraded: np.ndarray,
    subapertures: np.ndarray,
    N_subaperture: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-box contrast gain across the N Doppler sub-apertures.

    The contrast metric is the amplitude coefficient of variation

        C(img) = std(|img|) / mean(|img|).

    For each detection box ``k`` two crops are taken:

    * ``full_k``: the box cropped from ``|s_degraded|`` in s_degraded
      pixel coordinates → ``contrast_full[k] = C(full_k)``.
    * ``sub_{k,j}``: the box projected onto subaperture ``j``'s grid
      (azimuth coordinates divided by ``N_subaperture``, range
      unchanged — exactly the projection used by
      :func:`compute_subaperture_com_per_box`) →
      ``contrast_sub[k, j] = C(sub_{k,j})``.

    The per-box gain is

        contrast_gain[k] = sum_j contrast_sub[k, j] / contrast_full[k]

    which equals ``sum_j (contrast_sub[k, j] / contrast_full[k])`` —
    the total per-subaperture sharpness ratio relative to the full
    aperture. A stationary point target has roughly equal contrast in
    every Doppler sub-band as in the full image, so the gain ≈
    ``N_subaperture``. A defocused mover smears in ``s_degraded``
    (low ``contrast_full``) but refocuses in one or a few sub-bands
    (high individual ``contrast_sub``), so the gain rises above
    ``N_subaperture``.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int array
        ``[(y_centre, x_centre, h, w), …]`` in ``s_degraded`` pixels.
    s_degraded : complex ndarray, shape (N_az_d, N_range_d)
        Full-aperture degraded SLC; only its magnitude is used.
    subapertures : (N_sub, sub_size, N_range_d) ndarray
        Subaperture amplitudes returned by :func:`compute_subapertures`.
    N_subaperture : int
        Number of Doppler sub-bands (== ``subapertures.shape[0]``).

    Returns
    -------
    contrast_gain : (K,) float64 ndarray
        Per-box sum-ratio metric. NaN if ``contrast_full[k]`` is
        non-finite / non-positive, or no subaperture had a finite
        contrast.
    contrast_sub : (K, N_sub) float64 ndarray
        Per-box, per-subaperture contrast. NaN where the projected
        window is empty or the crop has zero mean.
    contrast_full : (K,) float64 ndarray
        Per-box contrast of the ``|s_degraded|`` crop. NaN if the crop
        is empty or has zero mean.
    """
    n_sub, sub_size, n_rg_d = subapertures.shape
    n_az_d = s_degraded.shape[0]
    K = len(boxes_yxhw)
    contrast_sub = np.full((K, n_sub), np.nan, dtype=np.float64)
    contrast_full = np.full(K, np.nan, dtype=np.float64)
    contrast_gain = np.full(K, np.nan, dtype=np.float64)
    if K == 0:
        return contrast_gain, contrast_sub, contrast_full

    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg_d, int(round(x_c + w / 2)))
        if x1 - x0 < 1:
            continue

        # Full-aperture crop on the s_degraded grid (azimuth NOT divided).
        y0_full = max(0, int(round(y_c - h / 2)))
        y1_full = min(n_az_d, int(round(y_c + h / 2)))
        if y1_full - y0_full >= 1:
            full = np.abs(s_degraded[y0_full:y1_full, x0:x1]).astype(np.float64)
            mu_full = float(full.mean())
            if np.isfinite(mu_full) and mu_full > 0.0:
                contrast_full[k] = float(full.std()) / mu_full

        # Same projection as compute_subaperture_com_per_box.
        y0_sub = max(0, int(round((y_c - h / 2) / N_subaperture)))
        y1_sub = min(sub_size, int(round((y_c + h / 2) / N_subaperture)))
        if y1_sub - y0_sub >= 1:
            for j in range(n_sub):
                # subapertures already holds amplitudes (real, non-negative).
                img = subapertures[j, y0_sub:y1_sub, x0:x1].astype(np.float64)
                mu = float(img.mean())
                if np.isfinite(mu) and mu > 0.0:
                    contrast_sub[k, j] = float(img.std()) / mu

        if (np.isfinite(contrast_full[k])
                and contrast_full[k] > 0.0
                and np.isfinite(contrast_sub[k]).any()):
            contrast_gain[k] = (
                float(np.nansum(contrast_sub[k])) / contrast_full[k]
            )

    return contrast_gain, contrast_sub, contrast_full


def compute_box_com_velocities(
    com_az_sub: np.ndarray,
    com_rg: np.ndarray,
    sub_size: int,
    prf: float,
    N_subaperture: int,
    azimuth_spacing_m: float,
    Number_of_Range_Looks: int,
    range_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-box azimuth and range COM velocity (m/s) from a linear fit
    across the N subapertures.

    The k-th subaperture is an IFFT of the k-th of N_subaperture
    contiguous Doppler bands of the SLC azimuth spectrum. In slow time
    its centre lies at

        t_k = (k + 0.5) · (sub_size / PRF)      seconds

    where ``sub_size = N_az_s_degraded // N_subaperture`` is the number of
    raw azimuth samples per band (== ``subapertures.shape[1]``). The COM
    output of :func:`compute_subaperture_com_per_box` is in pixel
    coordinates; we convert each axis to ground metres before fitting:

        az_m[k, j] = com_az_sub[k, j] · N_subaperture · azimuth_spacing_m
        rg_m[k, j] = com_rg[k, j]    · Number_of_Range_Looks · range_spacing_m

    Then a 1-D linear fit ``y = v·t + b`` is performed across the
    subapertures with a finite COM (≥ 2 needed). The slope ``v`` is the
    apparent COM velocity along that axis in m/s. For along-track
    motion the target's amplitude centroid advances by v_az·(t_k+1 − t_k)
    between subaperture centres, so to first order the fit slope equals
    the target's azimuth ground velocity; the analogous logic holds in
    range.

    Returns
    -------
    v_az_mps : (K,) float64
        Azimuth COM velocity (m/s); NaN if fewer than 2 subapertures
        have a finite COM in this box.
    v_rg_mps : (K,) float64
        Range COM velocity (m/s); same NaN convention.
    """
    K, n_sub = com_az_sub.shape
    if K == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    dt = float(sub_size) / float(prf)
    t = (np.arange(n_sub, dtype=np.float64) + 0.5) * dt
    az_per_pix_m = float(N_subaperture) * float(azimuth_spacing_m)
    rg_per_pix_m = float(Number_of_Range_Looks) * float(range_spacing_m)
    v_az = np.full(K, np.nan, dtype=np.float64)
    v_rg = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        az_finite = np.isfinite(com_az_sub[k])
        if az_finite.sum() >= 2:
            try:
                slope, _ = np.polyfit(
                    t[az_finite],
                    com_az_sub[k, az_finite] * az_per_pix_m,
                    1,
                )
                v_az[k] = float(slope)
            except (np.linalg.LinAlgError, ValueError):
                pass
        rg_finite = np.isfinite(com_rg[k])
        if rg_finite.sum() >= 2:
            try:
                slope, _ = np.polyfit(
                    t[rg_finite],
                    com_rg[k, rg_finite] * rg_per_pix_m,
                    1,
                )
                v_rg[k] = float(slope)
            except (np.linalg.LinAlgError, ValueError):
                pass
    return v_az, v_rg


def compute_box_phase_estimates(
    boxes_yxhw: np.ndarray,
    d_phase: np.ndarray,
    amp_raw: np.ndarray,
    min_row_coherence: float = 0.0,
    inlier_tol_rad: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-box azimuth phase estimate, linear fit, and fit residual.

    For each box, the azimuth phase derivative inside the box is collapsed
    across range using an amplitude-weighted circular mean, giving one
    phase sample per azimuth row:

        z[i]   = Σ_j w[i, j] · exp(j · d_phase[y0+i, x0+j])
        w[i,j] = |s_degraded[y0+i, x0+j]| · |s_degraded[y0+i+1, x0+j]|
        phi[i] = arg z[i]                     (the estimated phase)
        coh[i] = |z[i]| / Σ_j w[i, j]         (per-row coherence ∈ [0, 1])

    The per-row phase ``phi(i)`` is the actual estimate we save per target.
    A coherence-weighted polyfit gives the linear model

        phi(i) ≈ slope · i + intercept            (i = 0 .. n_rows-1)

    which lets you reconstruct the fit at absolute azimuth row ``y0 + i``.
    The per-box residual score reported here is the unweighted mean of the
    wrapped distance from each sample to the fit,

        r[i] = arg{ exp(j · (phi[i] - (slope·i + intercept))) }   (∈ (-π, π])
        residual_mean = mean_i |r[i]|

    i.e. the mean Euclidean distance (in radians, with circular wrap) from
    each dot in Figure 5 to the red line. A coherent ramp has a small
    score; a noisy / clutter target has a large one. Same metric the
    ``--max-phase-residual-rad`` filter compares against.

    Identical numerics to :func:`filter_boxes_by_phase_slope` /
    :func:`filter_boxes_by_phase_residual`.

    Returns
    -------
    phi : (K, max_h) float64
        Per-row weighted-circular-mean azimuth phase (rad), NaN-padded
        beyond each box's actual row count.
    coh : (K, max_h) float64
        Per-row coherence in [0, 1], NaN-padded the same way.
    slope : (K,) float64
        Slope of the coherence-weighted linear fit (rad / azimuth row).
        NaN where the fit was not performed (< 2 valid rows, etc.).
    intercept : (K,) float64
        Intercept of the same fit at i = 0, i.e. at absolute row ``y0``
        (rad). NaN where the fit was not performed.
    y0 : (K,) int64
        Absolute azimuth row of the first sample of each box's strip in
        d_phase coordinates. Use this with ``i = 0 .. n_rows-1`` to
        reconstruct absolute azimuth rows.
    n_rows : (K,) int64
        Actual number of azimuth rows used per box (may be less than ``h``
        for boxes clipped at the image edge). 0 for boxes the fit could
        not run on.
    residual_mean : (K,) float64
        Coh-weighted mean wrapped Euclidean distance from each sample to
        the chosen line (rad), i.e. the actual cost the brute-force grid
        search minimised — the smaller the better. NaN where the fit
        was not performed.
    n_inliers : (K,) int64
        Diagnostic: number of high-coh rows that happen to land within
        ``inlier_tol_rad`` of the chosen line. 0 where the fit was not
        performed.
    """
    K = len(boxes_yxhw)
    if K == 0:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    # `int(...)` would truncate float box heights and can be one row short
    # of the actual span `L = int(round(y_c + h/2)) - int(round(y_c - h/2))`
    # for non-integer ``h`` (e.g. h=6391.7 → trunc=6391, but rounding can
    # produce L=6392). Use ``ceil`` with one extra row of headroom so the
    # (K, max_h) ``phi_arr`` / ``coh_arr`` buffers always fit `L`, and
    # ``n_rows_arr[k] = L`` stays consistent with ``phi_arr.shape[1]``
    # for every downstream `box_phi[k, :n_k]` slice.
    max_h = max(
        int(np.ceil(float(boxes_yxhw[:, 2].max()))) + 1, 1,
    )
    phi_arr = np.full((K, max_h), np.nan, dtype=np.float64)
    coh_arr = np.full((K, max_h), np.nan, dtype=np.float64)
    slope_arr = np.full(K, np.nan, dtype=np.float64)
    intercept_arr = np.full(K, np.nan, dtype=np.float64)
    y0_arr = np.zeros(K, dtype=np.int64)
    n_rows_arr = np.zeros(K, dtype=np.int64)
    residual_arr = np.full(K, np.nan, dtype=np.float64)
    n_inliers_arr = np.zeros(K, dtype=np.int64)
    n_az, n_rg = d_phase.shape
    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        y0 = max(0, int(round(y_c - h / 2)))
        y1 = min(n_az, int(round(y_c + h / 2)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg, int(round(x_c + w / 2)))
        y0_arr[k] = y0
        L = y1 - y0
        if L < 2 or x1 - x0 < 1:
            continue
        sub_phase = d_phase[y0:y1, x0:x1]
        # weight[i, j] = |s[y0+i, x0+j]|·|s[y0+i+1, x0+j]| — same as in the
        # filter; we read amp_raw[y0+1:y1+1] which is safe because d_phase
        # is one row shorter than s_degraded, so y1 ≤ N_az_d = N_az_s - 1.
        weight = amp_raw[y0:y1, x0:x1] * amp_raw[y0 + 1:y1 + 1, x0:x1]
        z = (weight * np.exp(1j * sub_phase)).sum(axis=1)
        w_sum = weight.sum(axis=1) + 1e-12
        phi = np.angle(z)
        coh = np.abs(z) / w_sum
        L_pad = min(L, max_h)
        phi_arr[k, :L_pad] = phi[:L_pad]
        coh_arr[k, :L_pad] = coh[:L_pad]
        n_rows_arr[k] = L
        # Centred index keeps slope and intercept numerically independent;
        # we then shift the intercept back to absolute row y0 (i.e. i=0)
        # for downstream reconstruction.
        u = np.arange(L, dtype=float)
        # Brute-force wrap-aware line fit on the wrapped phi values:
        # try every slope on an equispaced grid, take the wrap-aware
        # optimal intercept per candidate, and pick the candidate whose
        # wrapped line minimises the coh-weighted mean wrapped distance
        # to the data. No LS refit on top — the grid value is returned
        # directly, so the smooth-cost local-min failure mode (phasor LS
        # landing on a high-wrap-count line that does not actually pass
        # through the data) cannot happen.
        slope, intercept, n_in = _min_distance_line_fit(
            phi.astype(np.float64), u, coh.astype(np.float64),
            min_row_coherence=min_row_coherence,
            inlier_tol_rad=inlier_tol_rad,
        )
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue
        slope_arr[k] = float(slope)
        intercept_arr[k] = float(intercept)
        n_inliers_arr[k] = int(n_in)
        # Residual = the actual cost minimised by the grid search
        # = coh-weighted mean |wrapped(phi - line)| over high-coh rows,
        # with fall-back to all rows when too few are high-coh.
        r = np.angle(np.exp(1j * (phi - (slope * u + intercept))))
        high = coh > min_row_coherence
        if int(high.sum()) < 2:
            high = np.ones_like(coh, dtype=bool)
        w_row = np.where(high, coh, 0.0)
        denom = float(w_row.sum())
        if denom > 0.0:
            residual_arr[k] = float((w_row * np.abs(r)).sum() / denom)
    return (
        phi_arr, coh_arr, slope_arr, intercept_arr,
        y0_arr, n_rows_arr, residual_arr, n_inliers_arr,
    )


def filter_boxes_by_phase_residual(
    boxes_yxhw: np.ndarray,
    d_phase: np.ndarray,
    amp_raw: np.ndarray,
    max_residual_rad: float,
    mask_full: np.ndarray | None = None,
    detection_skip_frac: float = 0.5,
    min_row_coherence: float = 0.0,
    inlier_tol_rad: float = 0.5,
) -> np.ndarray:
    """Drop boxes whose mean wrapped phase residual to the linear fit
    exceeds ``max_residual_rad`` (rad).

    For each box, the per-row weighted-circular-mean phase ``phi[i]`` and
    the brute-force wrap-aware fit ``phi_fit[i] = slope·i + intercept``
    are computed exactly as in :func:`compute_box_phase_estimates`. The
    score is the coh-weighted mean wrapped Euclidean distance over the
    high-coh rows (with a fall-back to all rows when too few qualify),

        r[i]          = arg{ exp(j · (phi[i] - phi_fit[i])) }   (∈ (-π, π])
        residual_mean = Σ_high coh[i]·|r[i]| / Σ_high coh[i]

    i.e. the average distance (rad) from each high-coh sample to the red
    fit line in Figure 5, with circular wrap so a real ±2π roll-over
    inside the box isn't counted as a 2π error. A coherent ramp scores
    low; a noisy / clutter target scores high. Identical to the cost
    the grid-search line fit just minimised, so the filter compares the
    box against the actual optimum.

    Bypass on detection count
    -------------------------
    If ``mask_full`` is supplied and a box has

        mask_full[y0:y1, x0:x1].sum() > detection_skip_frac · (y1 - y0)

    — i.e. its CoV-gate detection count exceeds
    ``detection_skip_frac`` per azimuth row on average — the box is kept
    regardless of its phase residual. The idea is that a target with that
    many CoV hits is already strongly supported by amplitude evidence
    alone, so a noisy phase trace shouldn't kill it. Set
    ``detection_skip_frac`` to a very large value to disable the bypass.

    Non-positive ``max_residual_rad`` disables the filter (passes
    everything through unchanged).
    """
    if max_residual_rad <= 0 or len(boxes_yxhw) == 0:
        return boxes_yxhw
    n_az, n_rg = d_phase.shape
    keep = np.zeros(len(boxes_yxhw), dtype=bool)
    n_bypass = 0
    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        y0 = max(0, int(round(y_c - h / 2)))
        y1 = min(n_az, int(round(y_c + h / 2)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg, int(round(x_c + w / 2)))
        L = y1 - y0
        if L < 2 or x1 - x0 < 1:
            continue
        # Detection-count bypass. mask_full has shape (N_az_s, N_rg) while
        # d_phase is (N_az_s - 1, N_rg); since y1 ≤ n_az = N_az_s - 1 the
        # same (y0:y1, x0:x1) slice is valid for both. Any box with
        # > detection_skip_frac · L mask hits passes regardless of
        # residual — a bright target with a noisy phase trace shouldn't
        # be culled by this filter.
        if mask_full is not None:
            det_count = int(mask_full[y0:y1, x0:x1].sum())
            if det_count > detection_skip_frac * L:
                keep[k] = True
                n_bypass += 1
                continue
        sub_phase = d_phase[y0:y1, x0:x1]
        weight = amp_raw[y0:y1, x0:x1] * amp_raw[y0 + 1:y1 + 1, x0:x1]
        z = (weight * np.exp(1j * sub_phase)).sum(axis=1)
        w_sum = weight.sum(axis=1) + 1e-12
        phi = np.angle(z)
        coh = np.abs(z) / w_sum
        u = np.arange(L, dtype=float)
        slope, intercept, _n_in = _min_distance_line_fit(
            phi.astype(np.float64), u, coh.astype(np.float64),
            min_row_coherence=min_row_coherence,
            inlier_tol_rad=inlier_tol_rad,
        )
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue
        # Residual = the same coh-weighted mean wrapped distance the
        # grid search minimised; nothing more, nothing less.
        r = np.angle(np.exp(1j * (phi - (slope * u + intercept))))
        high = coh > min_row_coherence
        if int(high.sum()) < 2:
            high = np.ones_like(coh, dtype=bool)
        w_row = np.where(high, coh, 0.0)
        denom = float(w_row.sum())
        if denom <= 0.0:
            continue
        residual_mean = float((w_row * np.abs(r)).sum() / denom)
        if np.isfinite(residual_mean) and residual_mean <= max_residual_rad:
            keep[k] = True
    if mask_full is not None and n_bypass:
        print(
            f"  residual filter: bypassed {n_bypass}/{len(boxes_yxhw)} "
            f"boxes via detection count > "
            f"{detection_skip_frac:g} · n_azimuth_rows"
        )
    return boxes_yxhw[keep]

def compute_contrast(image: np.ndarray) -> float:
    """Image-intensity contrast ``std(|x|^2) / mean(|x|^2)``.

    Accepts a real or complex array of any shape. Higher values indicate
    sharper, more focused imagery. Computation is performed in ``float64``
    to avoid precision loss on large SLC magnitudes, and non-finite samples
    are dropped before the statistics are taken. Returns ``0.0`` if the
    input is empty, all non-finite, or has non-positive mean intensity.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return 0.0
    intensity = np.abs(arr).astype(np.float64, copy=False) ** 2
    if not np.all(np.isfinite(intensity)):
        intensity = intensity[np.isfinite(intensity)]
        if intensity.size == 0:
            return 0.0
    mean = intensity.mean()
    if not np.isfinite(mean) or mean <= 0.0:
        return 0.0
    return float(intensity.std() / mean)


def filter_boxes_by_phase_slope(
    boxes_yxhw: np.ndarray,
    d_phase: np.ndarray,
    amp_raw: np.ndarray,
    min_slope_deg: float,
    min_row_coherence: float = 0.0,
    inlier_tol_rad: float = 0.5,
) -> np.ndarray:
    """Drop boxes whose amp-weighted phase ramp is shallower than ``min_slope_deg``.

    For each box, computes the coherence-weighted circular mean of
    ``d_phase`` along range for every azimuth row inside the box, then
    fits a wrap-aware line ``phi(u) = slope·u + b`` via brute-force
    grid search (see :func:`_min_distance_line_fit` — the slope minimising
    the coh-weighted mean wrapped distance over an equispaced grid of
    1001 candidates, with no LS refit). Boxes with
    ``abs(np.degrees(slope)) < min_slope_deg`` (i.e. shallower than
    ``min_slope_deg`` per azimuth row) are removed, along with any box that
    cannot be fit (fewer than two valid azimuth rows or NaN slope).

    ``d_phase`` has shape ``(N_az - 1, N_rg)`` and ``amp_raw`` has shape
    ``(N_az, N_rg)``; box centres/sizes are in d_phase coordinates.

    Returns the kept rows of ``boxes_yxhw`` in their original order. If
    ``min_slope_deg <= 0`` the input is returned unchanged.
    """
    if min_slope_deg <= 0 or len(boxes_yxhw) == 0:
        return boxes_yxhw
    n_az, n_rg = d_phase.shape
    min_slope_rad = float(np.deg2rad(min_slope_deg))
    keep = np.zeros(len(boxes_yxhw), dtype=bool)
    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        y0 = max(0, int(round(y_c - h / 2)))
        y1 = min(n_az, int(round(y_c + h / 2)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg, int(round(x_c + w / 2)))
        if y1 - y0 < 2 or x1 - x0 < 1:
            continue
        sub_phase = d_phase[y0:y1, x0:x1]
        weight = amp_raw[y0:y1, x0:x1] * amp_raw[y0 + 1:y1 + 1, x0:x1]
        z = (weight * np.exp(1j * sub_phase)).sum(axis=1)
        w_sum = weight.sum(axis=1) + 1e-12
        phi = np.angle(z)
        coh = np.abs(z) / w_sum
        u_local = np.arange(y1 - y0, dtype=np.float64)
        slope, _, _n_in = _min_distance_line_fit(
            phi.astype(np.float64), u_local, coh.astype(np.float64),
            min_row_coherence=min_row_coherence,
            inlier_tol_rad=inlier_tol_rad,
        )
        if np.isfinite(slope) and abs(slope) >= min_slope_rad:
            keep[k] = True
    return boxes_yxhw[keep]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def compute_subapertures(s: np.ndarray, N: int = 8) -> np.ndarray:
    """Split SLC into N azimuth subapertures and return their amplitudes.

    Parameters
    ----------
    s : complex ndarray, shape (N_az, N_range)
        Input SLC. Axis 0 = azimuth (spatial), axis 1 = range.
    N : int
        Number of subapertures to split into.

    Returns
    -------
    amplitudes : real ndarray, shape (N, N_az // N, N_range)
        Amplitude of each subaperture image.
    """
    N_az = s.shape[0]
    sub_size = N_az // N

    # Azimuth FFT with DC at the centre so contiguous index slices map to
    # contiguous Doppler sub-bands ordered from -Nyq to +Nyq.
    S = np.fft.fftshift(np.fft.fft(s, axis=0), axes=0)

    amplitudes = np.zeros((N, sub_size, s.shape[1]), dtype=float)
    for k in range(N):
        band = S[k * sub_size:(k + 1) * sub_size]
        # IFFT only the band itself: produces a properly Nyquist-sampled
        # sub-image of size (sub_size, N_range), reduced azimuth resolution.
        # The constant Doppler offset of band k turns into a uniform linear
        # phase modulation along axis 0, which drops out of |s_sub|.
        s_sub = np.fft.ifft(band, axis=0)
        amplitudes[k] = np.abs(s_sub)

    return amplitudes


# ---------------------------------------------------------------------------
# Rough refocus helpers — line → quadratic phase → freq-domain correction
# ---------------------------------------------------------------------------
def _normalized_variance(chip: np.ndarray) -> float:
    """Image contrast = std(|I|²) / mean(|I|²) of a complex chip.

    This is the dimensionless intensity-contrast metric used in
    entropy-minimisation autofocus papers. Higher = sharper (more
    concentrated energy). NaN-safe; returns 0.0 if mean intensity
    is zero or non-finite.
    """
    intensity = np.abs(chip) ** 2
    m = float(np.mean(intensity))
    if not np.isfinite(m) or m <= 0.0:
        return 0.0
    return float(np.std(intensity)) / m


def _refocus_box_chip(
    chip: np.ndarray,
    slope_rad_per_row: float,
    intercept_rad: float = 0.0,  # kept for API symmetry; unused on purpose
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the centred quadratic phase of the fitted line as a refocus.

    Background. The per-box fit produced a line in d_phase space,

        d_phase_line(n) = slope · n + intercept,   n = 0..N-1.

    The intercept is the Doppler-centroid shift (= pure azimuth
    translation of the chip; irrelevant for contrast / focusing). The
    quadratic phase error itself comes from the *slope*: re-centring
    the line so its midpoint sits at zero gives

        d_phase_centered(m) = slope · m,   m = n − N/2,

    which integrates analytically to a centred, midpoint-zero
    quadratic phase

        φ(m) = ½ · slope · m²,    φ(0) = 0.

    Because the QPE is estimated in the azimuth-frequency domain in
    PGA, we apply the correction *there* following the reference
    `shear_focusing.apply_phase_correction` recipe:

        chip_F  = fft(ifftshift(chip, axis=0), axis=0)
        chip_F *= exp(-1j · φ)[:, None]
        out     = fftshift(ifft(chip_F, axis=0), axis=0)

    The purpose is NOT real autofocus (a much better one happens
    elsewhere) but a classification signal: if the slope we fitted
    really corresponds to a moving target's QPE, applying its
    conjugate on the spectrum will concentrate the chip's energy and
    raise time-domain image contrast; if the slope was noise, the
    contrast will not improve.

    Returns the corrected chip and the applied per-row phase φ(n).
    """
    N = chip.shape[0]
    if N < 2:
        return chip.copy(), np.zeros(N, dtype=np.float64)
    # Centred-line integral: φ(m) = ½ · slope · m²,  m = n − N//2.
    # φ is zero at the midpoint by construction.
    m = np.arange(N, dtype=np.float64) - (N // 2)
    phi = 0.5 * float(slope_rad_per_row) * m * m
    chip_F = np.fft.fft(np.fft.ifftshift(chip, axes=0), axis=0)
    chip_F = chip_F * np.exp(1j * phi)[:, None]
    corrected = np.fft.fftshift(np.fft.ifft(chip_F, axis=0), axes=0)
    return corrected, phi


# ----------------------------------------------------------------------
# Polynomial range-walk autofocus (mirrors core/autofocus.py)
# ----------------------------------------------------------------------
# The QGIS plugin's `apply_global_range_deviation_correction`
# (core/autofocus.py:760-804) estimates a polynomial range walk by
# maximising the per-subaperture intensity-contrast sum across a
# coarse-to-fine grid of polynomial deviations, then applies the
# corresponding per-row range shift on the azimuth-FFT and IFFTs back.
# This script is standalone (no QGIS), so the same helpers are
# replicated locally without the QgsMessageLog logging.


def _af_compute_contrast_subaperture_sums(
    image: np.ndarray, N_subaperture: int = 10
) -> np.ndarray:
    """Per-subaperture focus metric over the centred 80 % of azimuth.

    Mirrors ``core.autofocus.compute_contrast_subaperture_sums``. Drops
    the outer 10 % of azimuth rows on each side, splits the remaining
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


def _af_shift_fitted(
    s: np.ndarray, fitted: np.ndarray
) -> np.ndarray:
    """Per-row range shift via Fourier-domain phase ramp.

    Mirrors ``core.autofocus.shift_fitted``. Each row ``k`` of ``s`` is
    shifted along the range axis by ``fitted[k]`` samples.
    """
    N = s.shape[1]
    k_over_N = np.arange(N) / N - 0.5
    phase_ramp = np.exp(
        1j * 2.0 * np.pi * fitted[:, None] * k_over_N[None, :]
    )
    shift_term = np.fft.fftshift(phase_ramp, axes=1)
    return np.fft.ifft(np.fft.fft(s, axis=1) * shift_term, axis=1)


def _af_contrast_ratio_sum(
    C: np.ndarray, C_initial: np.ndarray
) -> float:
    """Sum of element-wise ``C / C_initial`` (1.0 where baseline is 0)."""
    ratio = np.divide(
        C, C_initial,
        out=np.ones_like(C, dtype=np.float64),
        where=C_initial > 0,
    )
    return float(np.sum(ratio))


def _af_find_best_deviation(
    spatch_fft: np.ndarray,
    x: np.ndarray,
    *,
    dev_min: float = -100.0,
    dev_max: float = 100.0,
    accuracy: float = 0.5,
    poly_degree: int = 2,
) -> tuple[float, int]:
    """Coarse-to-fine search for the polynomial range deviation that
    maximises ``sum(C / C_initial)`` over the 10 subapertures.

    Mirrors ``core.autofocus.find_best_deviation`` (no QGIS logging).
    Returns ``(best_deviation, n_sub_contrast_improved)``: the second
    value is the per-subaperture count of contrast improvements
    captured at the most recent baseline update — i.e. how many of the
    10 subapertures improved at the step that produced
    ``best_deviation``. The upstream implementation always returns 0
    for this field; here it is computed as documented so the per-box
    plot conveys a meaningful number.
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

    fine_min = max(dev_min, best_deviation - 2 * coarse_step)
    fine_max = min(dev_max, best_deviation + 2 * coarse_step)
    fine = np.arange(fine_min, fine_max + accuracy, accuracy)
    for deviation in fine:
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

    return best_deviation, int(n_sub_contrast_improved)


# ---------------------------------------------------------------------------
# Standalone phase-gradient-autofocus (PGA) pipeline
# ---------------------------------------------------------------------------
# Mirrors ``core.autofocus.focus_with_centered_looks_pga`` and the helpers
# it depends on (``phase_gradient_autofocus``, ``select_pulse_with_strong_target``,
# ``apply_phase_correction``, ``entropy``, ``center_on_strong_target``,
# ``calculate_window``, ``weigthed_estimator``) as well as
# ``core.looks.extract_centered_look``. The script-side copy drops all
# QGIS logging so this file remains importable without ``qgis.core``.
# Used by :func:`_af_apply_global_range_deviation_correction` to refine
# the polynomial range-walk correction with PGA on chips where the
# residual range walk is large (|best_deviation| > 3.6).

def _pga_ft(s: np.ndarray, axis: int = -1) -> np.ndarray:
    """Centered FFT along *axis*."""
    return np.fft.fftshift(np.fft.fft(s, axis=axis), axes=axis)


def _pga_ift(f: np.ndarray, axis: int = -1) -> np.ndarray:
    """Centered inverse FFT along *axis*."""
    return np.fft.ifft(np.fft.ifftshift(f, axes=axis), axis=axis)


def _pga_entropy(data: np.ndarray) -> float:
    """Power-normalised entropy of complex data."""
    pwr = np.abs(data) ** 2
    pwr = pwr[pwr > 0]
    if pwr.size == 0:
        return float("inf")
    p = pwr / pwr.sum()
    return float(-np.sum(p * np.log(p)))


def _pga_select_pulse_with_strong_target(
    s: np.ndarray, percentile: float = 95.0, axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows/columns whose max-amplitude profile is in the top
    ``100 - percentile`` percent.
    """
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 or 1")
    line_max = np.amax(np.abs(s), axis=1 - axis)
    threshold = np.percentile(line_max, percentile)
    target_lines = np.where(line_max >= threshold)[0]
    if axis == 1:
        return s[:, target_lines], target_lines
    return s[target_lines], target_lines


def _pga_center_on_strong_target(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Roll each line so its strongest target sits at the centre of *axis*."""
    H, W = x.shape
    max_index = np.argmax(np.abs(x), axis=axis)
    if axis == 1:
        center = W // 2
        shifts = center - max_index
        return x[np.arange(H)[:, None], (np.arange(W) - shifts[:, None]) % W]
    center = H // 2
    shifts = center - max_index
    return x[
        (np.arange(H) - shifts[:, None]) % H,
        np.arange(W)[:, None],
    ]


def _pga_calculate_window(
    s: np.ndarray,
    threshold: float = -20.0,
    min_width: int = 50,
    axis: int = -1,
) -> np.ndarray:
    """Pick a centred target window from the power profile along *axis*.

    Sums ``|s|^2`` along *axis*, then returns a 1-D index array of length
    ``width`` covering a centred window on the OTHER axis (the axis that
    survives the reduction and is what the caller will index, e.g.
    ``p[:, window]`` after ``axis=0``). ``width`` is the count of bins
    within ``threshold`` dB of the per-axis power peak, floored at
    ``min_width`` and clamped to ``s.shape[other_axis] - 1`` so the
    window never overruns the indexed axis.

    Mirrors ``core.autofocus.calculate_window`` but fixes its axis
    confusion (the upstream version clamps / centres on ``s.shape[axis]``
    instead of the surviving axis, so ``rows > cols`` patches blow up
    when the window indices are used on a smaller axis).
    """
    p = np.sum(np.abs(s * np.conj(s)), axis=axis)
    p_max = p.max()
    if p_max > 0 and np.isfinite(p_max):
        p_db = 10.0 * np.log10(p / p_max)
        width = int(np.sum(p_db > threshold))
    else:
        width = min_width
    if width < min_width:
        width = min_width
    # The window must index the axis that survived the reduction, i.e.
    # the OTHER axis from the one we summed over. This is the bug-fix
    # over upstream ``core.autofocus.calculate_window``.
    other_axis = (axis + 1) % s.ndim
    n_other = s.shape[other_axis]
    width = min(width, n_other - 1)
    if width < 1:
        width = max(1, n_other - 1)
    center = (n_other - 1) // 2
    return np.arange(-width // 2, width // 2) + center


def _pga_weighted_estimator(x: np.ndarray) -> np.ndarray:
    """Magnitude-weighted phase-difference estimator (axis 0 = azimuth)."""
    s = np.conj(x[:-1, :]) * x[1:, :]
    return np.sum(np.angle(s) * np.abs(s), axis=1) / np.sum(np.abs(s), axis=1)


def _pga_phase_gradient_autofocus(
    data: np.ndarray,
    iter_num: int = 1,
    tolerance: float = 0.01,
) -> tuple[np.ndarray, list[float], list[float]]:
    """PGA estimate of the azimuth phase error.

    Returns ``(phase_corrections, rms_history, entropy_history)``.
    """
    entropies = [_pga_entropy(data)]
    phase_corrections = np.zeros(data.shape[0])
    rms: list[float] = []
    for _ in range(iter_num):
        data_centered = _pga_center_on_strong_target(data, axis=1)
        window = _pga_calculate_window(data_centered, axis=0)

        p = np.zeros_like(data_centered)
        p[:, window] = data_centered[:, window]
        P = _pga_ft(p, axis=0)

        phase_change = _pga_weighted_estimator(P)
        phase_change = np.unwrap([0, *np.cumsum(phase_change)])

        t = np.arange(0, phase_change.shape[0])
        trend = np.poly1d(np.polyfit(t, phase_change, 1))
        phase_change -= trend(t)
        rms.append(float(np.sqrt(np.mean(phase_change ** 2))))

        if rms[-1] < tolerance:
            break

        data = _pga_ft(data, axis=0)
        data *= np.exp(-1j * phase_change[:, None])
        data = _pga_ift(data, axis=0)

        entropies.append(_pga_entropy(data))
        phase_corrections += phase_change

    return phase_corrections[:, None], rms, entropies


def _pga_apply_phase_correction(
    data: np.ndarray, phase_error: np.ndarray,
) -> np.ndarray:
    """Apply (interpolated) azimuth phase correction to *data*."""
    x = np.linspace(0.0, 1.0, data.shape[0])
    xp = np.linspace(0.0, 1.0, phase_error.shape[0])
    phase_error_interp = np.interp(x, xp, phase_error.squeeze())
    data = _pga_ft(data, axis=0)
    data *= np.exp(-1j * phase_error_interp[:, None])
    return _pga_ift(data, axis=0)


def _pga_insert_center(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Insert *src* into the centre of *dst* (in place); returns *dst*."""
    rows, cols = dst.shape
    src_rows, src_cols = src.shape
    if src_rows > rows or src_cols > cols:
        raise ValueError("Source array is larger than destination array")
    start_row = rows // 2 - src_rows // 2
    start_col = cols // 2 - src_cols // 2
    dst[start_row:start_row + src_rows, start_col:start_col + src_cols] = src
    return dst


def _pga_extract_centered_look(
    spectrum: np.ndarray,
    center_row: int,
    center_col: int,
    look_rows: int,
    look_cols: int,
    *,
    apply_ifftshift: bool = True,
) -> np.ndarray:
    """Zero-pad a centred (look_rows, look_cols) sub-spectrum and inverse-FFT.

    Standalone mirror of ``core.looks.extract_centered_look``.
    """
    if look_rows <= 0 or look_cols <= 0:
        raise ValueError("Look size must be positive in both dimensions")

    rows, cols = spectrum.shape
    if not (0 <= center_row < rows and 0 <= center_col < cols):
        raise ValueError("Center index is out of bounds for the spectrum")
    if look_rows > rows or look_cols > cols:
        raise ValueError("Look size cannot exceed spectrum dimensions")

    window_row_start = center_row - look_rows // 2
    window_row_end = window_row_start + look_rows
    window_col_start = center_col - look_cols // 2
    window_col_end = window_col_start + look_cols

    src_row_start = max(0, window_row_start)
    src_row_end = min(rows, window_row_end)
    src_col_start = max(0, window_col_start)
    src_col_end = min(cols, window_col_end)

    look_window = np.zeros((look_rows, look_cols), dtype=spectrum.dtype)
    dst_row_start = src_row_start - window_row_start
    dst_row_end = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = src_col_start - window_col_start
    dst_col_end = dst_col_start + (src_col_end - src_col_start)

    look_window[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = (
        spectrum[src_row_start:src_row_end, src_col_start:src_col_end]
    )

    centered_spectrum = _pga_insert_center(np.zeros_like(spectrum), look_window)
    if apply_ifftshift:
        centered_spectrum = np.fft.ifftshift(centered_spectrum)
    return np.fft.ifft2(centered_spectrum)


def _pga_centered_look_row_counts_from_azimuth_fractions(
    spectrum_rows: int, azimuth_look_fractions: tuple[float, ...],
) -> list[int]:
    """Azimuth look heights as fractions of the Doppler spectrum row count."""
    if spectrum_rows < 1:
        return []
    heights: list[int] = []
    for frac in azimuth_look_fractions:
        h = max(1, min(spectrum_rows, int(round(spectrum_rows * frac))))
        if not heights or h > heights[-1]:
            heights.append(h)
    return heights


def _focus_with_centered_looks_pga(
    data: np.ndarray,
    *,
    azimuth_look_fractions: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25),
) -> np.ndarray:
    """Standalone mirror of ``core.autofocus.focus_with_centered_looks_pga``.

    Builds centred azimuth looks at the requested fractions of the Doppler
    spectrum, runs PGA on each look's strong-pulse patch, and returns the
    PGA-corrected look with the lowest entropy. Falls back to *data* if no
    look yields a valid phase estimate.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D SLC data, got shape {data.shape}")

    data = np.ascontiguousarray(data, dtype=np.complex64)
    spectrum = np.fft.fftshift(np.fft.fft2(data))
    rows, cols = spectrum.shape
    look_heights = _pga_centered_look_row_counts_from_azimuth_fractions(
        rows, azimuth_look_fractions,
    )
    if not look_heights:
        return data

    best_entropy = float("inf")
    best: np.ndarray | None = None
    best_look_rows: int | None = None
    for azimuth_look_size in look_heights:
        try:
            look = _pga_extract_centered_look(
                spectrum,
                center_row=rows // 2,
                center_col=cols // 2,
                look_rows=azimuth_look_size,
                look_cols=cols,
                apply_ifftshift=True,
            )
        except ValueError:
            continue

        patch, _ = _pga_select_pulse_with_strong_target(look, axis=0)
        if patch.size == 0 or patch.shape[0] < 2:
            continue

        phase_error, _, _ = _pga_phase_gradient_autofocus(patch)
        corrected_look = _pga_apply_phase_correction(look, phase_error)
        score = _pga_entropy(corrected_look)
        if score < best_entropy:
            best_entropy = score
            best = corrected_look
            best_look_rows = azimuth_look_size

    if best is None:
        return data
    # Compensate for the azimuth bandwidth thrown away by the winning look:
    # zeroing rows-best_look_rows of the Doppler spectrum scales image-domain
    # amplitudes by best_look_rows/rows, so multiply by the inverse fraction
    # to keep point-target levels comparable to the full-bandwidth input.
    scale = rows / float(best_look_rows)
    return (best * scale).astype(data.dtype, copy=False)


def _af_apply_global_range_deviation_correction(
    data: np.ndarray,
    *,
    dev_min: float = -100.0,
    dev_max: float = 100.0,
    accuracy: float = 0.5,
    poly_degree: int = 2,
    pga_abs_deviation_threshold: float = 3.6,
) -> tuple[np.ndarray, float, int]:
    """Standalone copy of ``core.autofocus.apply_global_range_deviation_correction``.

    Returns ``(corrected, best_deviation, n_sub_contrast_improved)``.

    When ``abs(best_deviation) > pga_abs_deviation_threshold`` (default
    3.6), the polynomial range-walk correction is refined with the local
    centred-look PGA pipeline (mirrors the QGIS-side two-stage autofocus
    in :class:`core.autofocus.AutofocusTask`).
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D SLC data, got shape {data.shape}")
    rows = data.shape[0]
    spatch_fft = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    x = np.linspace(-0.5, 0.5, rows)
    best_deviation, n_sub_contrast_improved = _af_find_best_deviation(
        spatch_fft, x,
        dev_min=dev_min, dev_max=dev_max,
        accuracy=accuracy, poly_degree=poly_degree,
    )
    fitted = (
        -best_deviation / x[-1] ** poly_degree * x ** poly_degree
    )
    spatch_fft = _af_shift_fitted(spatch_fft, fitted)
    corrected = np.fft.ifft(
        np.fft.ifftshift(spatch_fft, axes=0), axis=0,
    )
    is_apply_focusing = True
    if is_apply_focusing == True and abs(best_deviation) > pga_abs_deviation_threshold:
        corrected = _focus_with_centered_looks_pga(corrected)
    return (
        corrected.astype(data.dtype, copy=False),
        best_deviation,
        n_sub_contrast_improved,
    )


def _plot_refocus_box_1x2(
    chip_before: np.ndarray,
    chip_after: np.ndarray,
    *,
    box_idx: int,
    box_yxhw: tuple[int, int, int, int],
    slope_rad_per_row: float,
    slope_total_rad: float,
    contrast_before: float,
    contrast_after: float,
    gain_db: float,
    out_path: Path,
) -> None:
    """One PNG per box: |chip| before / |chip| after the centred-QPE refocus.

    chip is taken from s_degraded over the fit window. The title shows
    the per-box slope and the time-domain contrast gain (a moving-target
    classification signal, NOT a focusing claim).
    """
    y, x, h, w = box_yxhw
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Refocus box {box_idx:03d} @ (y={y}, x={x}, h={h}, w={w})  "
        f"slope = {slope_rad_per_row:+.4f} rad/row  "
        f"slope·n = {slope_total_rad:+.3f} rad  "
        f"contrast {contrast_before:.3f} → {contrast_after:.3f}  "
        f"({gain_db:+.2f} dB)",
        fontsize=11,
    )
    for ax, img, ttl in (
        (axes[0], chip_before, "|I|  before"),
        (axes[1], chip_after,  "|I|  after"),
    ):
        amp = np.abs(img)
        if amp.size == 0:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
            ax.set_title(ttl)
            continue
        vmin = float(np.nanpercentile(amp, 1.0))
        vmax = float(np.nanpercentile(amp, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1.0
        ax.imshow(amp, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xlabel("range")
        ax.set_ylabel("azimuth")
        ax.set_title(ttl)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)




def _plot_shear_averaging_box(
    chip: np.ndarray,
    *,
    box_idx: int,
    box_yxhw: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    """Visualise the three inputs of the shear-averaging autofocus on one box chip.

    Given the complex chip ``image`` (axis 0 = azimuth, axis 1 = range),
    the function plots:

    * ``image``                                          — time-domain |chip|.
    * ``image_azfft = FFT_azimuth(image)``               — log-magnitude of
      the azimuth (Doppler) spectrum, fftshifted so DC sits at the centre.
    * ``image_azfft[:-1] * conj(image_azfft[1:])``       — the "shear"
      product of adjacent Doppler bins. Both the magnitude (where the
      shear product is reliable — bright bins dominate the average) and
      the wrapped phase (the quantity that the shear-averaging algorithm
      averages coherently across range to estimate the QPE) are shown.

    No averaging / weighting / unwrapping is performed here — the
    helper is purely diagnostic so the upstream steps of the
    shear-averaging method (Mancill & Swiger 1981; Wahl et al.) can be
    inspected per box.
    """
    y, x, h, w = box_yxhw

    if chip.size == 0 or chip.shape[0] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "(empty chip)", ha="center", va="center")
        ax.set_title(
            f"Shear box {box_idx:03d} @ (y={y}, x={x}, h={h}, w={w})"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
        plt.close(fig)
        return

    image = chip
    image_azfft = np.fft.fftshift(np.fft.fft(chip, axis=0), axes=0)
    shear = image_azfft[:-1, :] * np.conj(image_azfft[1:, :])

    # --- Amplitude-weighted circular mean of the shear phase across range ---
    # Mirrors the kernel of compute_box_phase_estimates: take the
    # amplitude-weighted circular mean across the "range" axis. The
    # natural per-cell weight for arg{S[k]·S*[k+1]} is the modulus of
    # the same complex product, i.e. |shear| = |S[k]|·|S[k+1]|. With
    # that choice the weighted sum collapses to a plain coherent sum:
    #     z[k]   = Σ_r |shear[k, r]| · exp(j · arg{shear[k, r]})
    #            = Σ_r shear[k, r]
    #     phi[k] = arg z[k]                         (per-Doppler-bin phase)
    #     coh[k] = |z[k]| / Σ_r |shear[k, r]|       (coherence ∈ [0, 1])
    # Low-|S| Doppler bins (noise floor) contribute weight ≈ 0 to z and
    # to Σ_r |shear|, so the mask is encoded in the amplitudes — no
    # explicit gate needed at this stage.
    shear_amp = np.abs(shear)
    z_k = shear.sum(axis=1)
    w_sum_k = shear_amp.sum(axis=1) + 1e-12
    phi_k = np.angle(z_k)
    coh_k = np.abs(z_k) / w_sum_k

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"Shear averaging box {box_idx:03d} @ (y={y}, x={x}, h={h}, w={w})  "
        f"shape={chip.shape}",
        fontsize=11,
    )

    amp = np.abs(image)
    vmin = float(np.nanpercentile(amp, 1.0))
    vmax = float(np.nanpercentile(amp, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1.0
    im0 = axes[0, 0].imshow(amp, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("image = |chip|")
    axes[0, 0].set_xlabel("range")
    axes[0, 0].set_ylabel("azimuth")
    plt.colorbar(im0, ax=axes[0, 0])

    amp_fft = np.abs(image_azfft)
    log_amp = 20.0 * np.log10(amp_fft + np.finfo(np.float64).eps)
    vmax_fft = float(np.nanpercentile(log_amp, 99.5))
    vmin_fft = vmax_fft - 40.0
    im1 = axes[0, 1].imshow(
        log_amp, aspect="auto", cmap="viridis",
        vmin=vmin_fft, vmax=vmax_fft,
    )
    axes[0, 1].set_title("image_azfft = |FFT_az(chip)|  (dB, fftshifted)")
    axes[0, 1].set_xlabel("range")
    axes[0, 1].set_ylabel("azimuth-freq bin")
    plt.colorbar(im1, ax=axes[0, 1], label="dB")

    log_shear = 20.0 * np.log10(shear_amp + np.finfo(np.float64).eps)
    vmax_sh = float(np.nanpercentile(log_shear, 99.5))
    vmin_sh = vmax_sh - 40.0
    im2 = axes[1, 0].imshow(
        log_shear, aspect="auto", cmap="viridis",
        vmin=vmin_sh, vmax=vmax_sh,
    )
    axes[1, 0].set_title(r"|S[k]·S*[k+1]|  (shear product magnitude, dB)")
    axes[1, 0].set_xlabel("range")
    axes[1, 0].set_ylabel("azimuth-freq bin (k)")
    plt.colorbar(im2, ax=axes[1, 0], label="dB")

    shear_phase = np.angle(shear)
    im3 = axes[1, 1].imshow(
        shear_phase, aspect="auto", cmap="twilight",
        vmin=-np.pi, vmax=np.pi,
    )
    axes[1, 1].set_title(r"arg{ S[k]·S*[k+1] }  (shear product phase)")
    axes[1, 1].set_xlabel("range")
    axes[1, 1].set_ylabel("azimuth-freq bin (k)")
    plt.colorbar(im3, ax=axes[1, 1], label="rad")

    # Bottom-left: per-Doppler-bin amplitude-weighted shear phase phi[k]
    # = arg{ Σ_r |S[k]·S*[k+1]| · exp(j · arg{S[k]·S*[k+1]}) }, scatter
    # coloured by per-bin coherence.
    k_axis = np.arange(phi_k.size, dtype=np.float64)
    sc = axes[2, 0].scatter(
        k_axis, phi_k, c=coh_k, cmap="viridis",
        s=12, vmin=0.0, vmax=1.0, edgecolor="none",
    )
    plt.colorbar(sc, ax=axes[2, 0], label="coh")
    axes[2, 0].axhline(0.0, color="0.5", lw=0.5, ls="--")
    axes[2, 0].set_ylim(-np.pi, np.pi)
    axes[2, 0].set_xlim(0, phi_k.size - 1)
    axes[2, 0].set_title(
        r"$\varphi[k] = \arg\!\left\{\sum_r S[k,r]\cdot S^*[k+1,r]\right\}$"
        "   (amp-weighted, range-collapsed)"
    )
    axes[2, 0].set_xlabel("azimuth-freq bin (k)")
    axes[2, 0].set_ylabel(r"$\varphi[k]$  [rad]")
    axes[2, 0].grid(True, alpha=0.3)

    # Bottom-right: per-bin coherence.
    axes[2, 1].plot(k_axis, coh_k, color="C0", lw=1.2)
    axes[2, 1].set_ylim(0.0, 1.05)
    axes[2, 1].set_xlim(0, phi_k.size - 1)
    axes[2, 1].set_title(
        r"$\mathrm{coh}[k] = |z[k]| / \sum_r |S[k,r]\cdot S^*[k+1,r]|$"
    )
    axes[2, 1].set_xlabel("azimuth-freq bin (k)")
    axes[2, 1].set_ylabel("coh")
    axes[2, 1].grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATCH,
        help=f"Path to a .npy SLC patch. Defaults to {DEFAULT_PATCH}.",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=None,
        dest="kernel",
        help="Sinusoidal kernel length for moving target detection (default: N_az // 20).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=_default_save_path(),
        help=f"PNG output path. Defaults to {DEFAULT_SAVE_DIR}/shear_<ts>_<rand>.png.",
    )
    parser.add_argument(
        "--no-save", dest="save", action="store_const", const=None,
        help="Disable PNG saving.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-mode artefacts. When set, also build / save "
             "the mask-and-boxes overlay figure "
             "(`{stem}_mask_boxes.png`) and the raw arrays "
             "(`{stem}_dphase.npy`, `{stem}_slopes.npy`). When unset "
             "(default), those three artefacts are neither built nor "
             "written. Orthogonal to `--display-all-mode`, which gates "
             "the three richer display figures (subapertures, "
             "dphase_fit, box_com) on its own.",
    )
    parser.add_argument(
        "--display-all-mode",
        action="store_true",
        help="Display / save gate for the three rich diagnostic "
             "figures: per-Doppler-band subaperture amplitudes "
             "(`{stem}_subapertures.png`), the d_phase + per-box "
             "slope·n overview (`{stem}_dphase_fit.png`) and the "
             "per-box subaperture-COM trajectory grid "
             "(`{stem}_box_com.png`). When unset (default), none of "
             "them are built or written; when set, all three are "
             "produced. Independent of `--debug`.",
    )
    parser.add_argument(
        "--refocus",
        action="store_true",
        help="If set, run a *classification* refocus on every box that "
             "survives the motion gate. The fitted line a·n + b is "
             "re-centred so its midpoint sits at zero (intercept dropped "
             "— pure Doppler-centroid shift) and integrated analytically "
             "to a centred quadratic phase φ(m) = ½·a·(n − N/2)². "
             "Following PGA, exp(-1j·φ) is applied on the box chip's "
             "azimuth spectrum and IFFT'd back. For each surviving box "
             "a `box_NNN_refocus.png` (|I| before / |I| after) is "
             "written alongside the existing `box_NNN_dphase.png` in "
             "the `{stem}_per_box/` folder, plus a `refocus_summary.csv` "
             "with per-box (contrast_before, contrast_after, gain_dB). "
             "When set, the refocus gain is ALSO used as an additional "
             "filter (see --refocus-min-gain-db). Off by default.",
    )
    parser.add_argument(
        "--refocus-min-gain-db",
        type=float,
        default=-5.0,
        help="When --refocus is enabled, also drop every surviving box "
             "whose time-domain contrast gain (after the centred-QPE "
             "correction) falls below this threshold, in dB. The "
             "rationale: if applying the fitted slope's quadratic "
             "to the box chip DECREASES contrast significantly, the "
             "slope was inconsistent with the chip's actual spectral "
             "structure and the detection is likely spurious. Boxes "
             "with NaN gain (e.g. degenerate chip) are kept. Set very "
             "negative (e.g. -1e9) to effectively disable. Default: -2 dB.",
    )
    parser.add_argument(
        "--subaperture-sharpness-min",
        type=float,
        default=float("-inf"),
        help="Drop every surviving box whose per-box sub-aperture "
             "sharpness ratio Σ_j C(sub_j)/C(s_degraded) is finite and "
             "below this threshold. Here C(img) = std(|img|)/mean(|img|) "
             "is the amplitude coefficient of variation, summed across "
             "all N_subaperture Doppler sub-bands and divided by the "
             "full-aperture (degraded) contrast. With N_subaperture "
             "sub-bands a roughly stationary point target sits near "
             "`N_subaperture` (each sub-band ≈ same contrast as the "
             "full image); larger values indicate that one or more "
             "sub-bands are individually sharper than the degraded "
             "full aperture, the signature of a refocused mover; "
             "smaller values indicate a smeared / extended target "
             "whose sub-bands all lose contrast vs. the full aperture. "
             "This is a DIMENSIONLESS RATIO (the `contrast_gain` column "
             "of the CSV / NPZ), NOT the dB refocus-gain value drawn "
             "on `box_NNN_refocus.png` (that one is gated by "
             "`--refocus-min-gain-db`). Boxes with NaN ratio (empty "
             "crop / zero mean) are kept. Set to -inf to disable. "
             "Default: -inf (off).",
    )
    parser.add_argument(
        "--min-slope-deg", type=float, default=0.0,
        help="Drop boxes whose amp-weighted phase ramp has per-row slope "
             "|np.degrees(slope)| < this many degrees. Set to 0 to disable. "
             "Default: 0.0 (off — use --slope-rad-thresh instead).",
    )
    parser.add_argument(
        "--inlier-tol-rad", type=float, default=0.5,
        help="Diagnostic inlier tolerance (rad) for the per-box wrap-"
             "aware brute-force grid-search line fit. The fit itself "
             "minimises the coh-weighted mean wrapped distance "
             "|arg(exp(j·(phi[i] - slope·u[i] - intercept)))| over a "
             "1001-point slope grid (no LS refit, no consensus gating), "
             "so this tolerance does NOT affect slope selection. It is "
             "only used to report 'n_inliers' per box — the count of "
             "high-coh (coh > --min-row-coherence) rows that happen to "
             "land within ±--inlier-tol-rad of the chosen line — in "
             "the per-box plot titles and CSV. 0.5 rad ≈ ±29°. "
             "Default: 0.5.",
    )
    parser.add_argument(
        "--min-row-coherence", type=float, default=0.5,
        help="Only azimuth rows whose per-row coherence coh[i] = "
             "|Σ_j w·exp(j·dφ_ij)| / Σ_j w_ij exceeds this threshold are "
             "used to fit the per-box linear phase model phi(i) ≈ "
             "slope·i + intercept (and to evaluate its residual). Rows "
             "with coh[i] ≤ threshold have high circular variance of "
             "dφ across range and are dropped from the fit, but are "
             "still drawn in the per-box d_phase scatter (as the dark "
             "viridis dots) for inspection. Applies identically inside "
             "--min-slope-deg, --max-phase-residual-rad, and the saved "
             "per-box slope / intercept / residual. Set to 0 to disable "
             "(keeps all rows in the fit, original behaviour). "
             "Default: 0.5.",
    )
    parser.add_argument(
        "--max-phase-residual-rad", type=float, default=1.0,
        help="Drop boxes whose mean wrapped Euclidean distance from the "
             "estimated phase trace phi[i] to the linear fit slope·i + "
             "intercept exceeds this many radians. The score is "
             "mean(|arg(exp(j·(phi - phi_fit)))|), i.e. the average "
             "distance from each dot to the red line in Figure 5 with "
             "circular wrap. A coherent ramp scores low; a noisy / "
             "clutter target scores high. Set to 0 to disable. Default: "
             "1.0 rad (~57°).",
    )
    parser.add_argument(
        "--residual-skip-det-frac", type=float, default=0.5,
        help="Bypass the --max-phase-residual-rad filter for boxes whose "
             "CoV-gate detection count inside them exceeds this fraction "
             "of the box's azimuth row count. A bright / extended target "
             "with many mask hits is then trusted on amplitude alone, "
             "even if its phase trace is noisy. Set to a very large "
             "number to disable the bypass. Default: 0.5.",
    )
    parser.add_argument(
        "--slope-rad-thresh", type=float, default=1.0,
        help="UPPER box-wide phase-slope gate: |slope · n_rows| ≥ this "
             "value (rad) ⇒ the box is unambiguously moving and is "
             "kept regardless of any other signal. Also used as the "
             "highlight threshold for the red overlays in Figure 1 and "
             "the `is_strong` column in the CSV. Default: 1.0 rad.",
    )
    parser.add_argument(
        "--slope-rad-lower", type=float, default=0.1,
        help="LOWER box-wide phase-slope gate: |slope · n_rows| < this "
             "value (rad) ⇒ the box is always filtered out, no matter "
             "what other signals say. Boxes in the ambiguous middle "
             "band [slope-rad-lower, slope-rad-thresh) are decided by "
             "the azimuth COM peak-to-peak (--az-pp-sub-thresh). "
             "Default: 0.1 rad.",
    )
    parser.add_argument(
        "--az-pp-sub-thresh", type=float, default=20.0,
        help="Peak-to-peak azimuth COM walk across the N subapertures "
             "(in subaperture-azimuth pixels; 1 sub-az px ≈ "
             "N_subaperture · azimuth_spacing on the ground). Used as "
             "the TIEBREAKER for boxes whose box-wide phase slope is in "
             "the ambiguous band [slope-rad-lower, slope-rad-thresh): "
             "they are kept only if az_pp ≥ this threshold. Boxes with "
             "|slope·n| < slope-rad-lower are dropped regardless; boxes "
             "with |slope·n| ≥ slope-rad-thresh are kept regardless. "
             "Set very large (e.g. 1e9) to drop every middle-band box. "
             "Default: 20.0 sub-az px (revisit once geometry / radar "
             "params are calibrated).",
    )
    parser.add_argument(
        "--prf", type=float, default=6000.0,
        help="Azimuth PRF in Hz, used to time-stamp subaperture centres "
             "for the COM linear-velocity fit. Each raw azimuth sample "
             "is 1/PRF seconds apart, so a subaperture of "
             "sub_size = N_az/N_subaperture rows spans sub_size/PRF "
             "seconds; subaperture k is centred at t_k = (k+0.5)·"
             "sub_size/PRF. Default: 6000 Hz.",
    )
    parser.add_argument(
        "--min-velocity-mps", type=float, default=2.0,
        help="Per-box COM linear-velocity threshold (m/s) for the motion "
             "filter (Case 1). A box is moving iff |v_az| ≥ this OR "
             "|v_rg| ≥ this, where v_az and v_rg come from a linear fit "
             "of the per-subaperture COM (converted to ground metres) "
             "vs subaperture-centre time. The fit slope is the apparent "
             "ground COM velocity in that axis (m/s). Default: 2.0 m/s.",
    )
    parser.add_argument(
        "--com-az-thresh", type=float, default=3.0,
        help="Pre-phase amplitude filter: keep a box only if its azimuth COM "
             "peak-to-peak shift across the N subapertures is at least this "
             "many SUBAPERTURE-azimuth pixels (1 sub-az px ≈ "
             "N_subaperture × azimuth_spacing on the ground). Stationary "
             "targets typically score below the threshold. Set to a very "
             "large number to disable the azimuth criterion. Default: 3.0.",
    )
    parser.add_argument(
        "--com-rg-thresh", type=float, default=0.8,
        help="Pre-phase amplitude filter: keep a box only if its range COM "
             "peak-to-peak shift across the N subapertures is at least this "
             "many s_degraded RANGE pixels (1 d-range px ≈ "
             "Number_of_Range_Looks × range_spacing on the ground). Set to "
             "a very large number to disable the range criterion. "
             "Default: 0.8.",
    )
    parser.add_argument(
        "--com-filter-mode", choices=("any", "both", "off"), default="off",
        help="Legacy COM peak-to-peak filter (sub-aperture pixel "
             "thresholds via --com-az-thresh / --com-rg-thresh). "
             "Superseded by the motion filter "
             "(--min-velocity-mps + --slope-rad-thresh) which fits a "
             "linear velocity instead. Default: 'off'. Set to 'any' / "
             "'both' to re-enable the legacy PP pre-cull in front of "
             "the motion filter.",
    )
    parser.add_argument(
        "--n-subaperture", type=int, default=16,
        help="Number of azimuth subapertures (Doppler sub-bands) used to "
             "build the CoV mask AND to compute the per-box COM "
             "trajectories. With fewer looks the sample-CoV² estimator is "
             "noisier (std ∝ 1/√(N-1)), each look is Doppler-wider so a "
             "mover's bright-look contrast drops, and the upsampled mask "
             "is blockier in azimuth (block size = N_subaperture rows). "
             "Lower N → less selective gate; if you halve N you typically "
             "need to lower --cov-th-mult to recover marginal targets. "
             "Default: 16.",
    )
    parser.add_argument(
        "--cov-th-mult", type=float, default=1.5,
        help="Multiplier on median(CoV²) for the CoV mask threshold "
             "(th = mult · median(cov_sq)). Lower → more pixels pass the "
             "gate, more sensitive, more false alarms. When you reduce "
             "--n-subaperture, the median floor and its sample noise both "
             "rise; drop this to ~1.0–1.2 at N=8 to compensate. Default: 1.5.",
    )
    parser.add_argument(
        "--debug-roi", type=str, default=None,
        help="Trace boxes overlapping a rectangle 'y_lo,y_hi,x_lo,x_hi' "
             "(s_degraded coords). At each filter stage the script prints "
             "how many boxes still overlap the ROI and which were dropped. "
             "Use to figure out where a missing target gets eliminated. "
             "Example: --debug-roi 2500,5500,100,115. Default: off.",
    )
    args = parser.parse_args()

    debug_roi = None
    if args.debug_roi:
        try:
            _y_lo, _y_hi, _x_lo, _x_hi = (
                int(s) for s in args.debug_roi.split(",")
            )
            debug_roi = (_y_lo, _y_hi, _x_lo, _x_hi)
            print(
                f"[roi] tracing boxes overlapping y={_y_lo}..{_y_hi}, "
                f"x={_x_lo}..{_x_hi}"
            )
        except (ValueError, TypeError):
            print(
                f"  (warning) --debug-roi could not parse "
                f"'{args.debug_roi}'; ignored."
            )

    def _roi_hits(boxes_in: np.ndarray) -> np.ndarray:
        """Indices of boxes overlapping the ROI rectangle (or empty)."""
        if debug_roi is None or len(boxes_in) == 0:
            return np.empty(0, dtype=np.int64)
        y_lo, y_hi, x_lo, x_hi = debug_roi
        y0 = boxes_in[:, 0] - boxes_in[:, 2] / 2
        y1 = boxes_in[:, 0] + boxes_in[:, 2] / 2
        x0 = boxes_in[:, 1] - boxes_in[:, 3] / 2
        x1 = boxes_in[:, 1] + boxes_in[:, 3] / 2
        ov = (y1 >= y_lo) & (y0 <= y_hi) & (x1 >= x_lo) & (x0 <= x_hi)
        return np.where(ov)[0]

    def _roi_print(label: str, boxes_in: np.ndarray) -> None:
        if debug_roi is None:
            return
        hit_idx = _roi_hits(boxes_in)
        if hit_idx.size == 0:
            print(f"[roi] after {label:30s}: 0 boxes overlap ROI")
            return
        bits = []
        for k in hit_idx[:6]:
            y_c, x_c, h, w = boxes_in[k]
            bits.append(
                f"#{int(k)} y={int(y_c)} x={int(x_c)} "
                f"h={int(h)} w={int(w)}"
            )
        extra = f" + {hit_idx.size - 6} more" if hit_idx.size > 6 else ""
        print(
            f"[roi] after {label:30s}: {hit_idx.size} boxes overlap ROI "
            f"→ {', '.join(bits)}{extra}"
        )

    sw = _Stopwatch()

    s = np.load(args.path)
    print(f"Loaded: {args.path.name}  shape={s.shape}  dtype={s.dtype}")

    # --- Processing pipeline ---

    max_length_of_target = 130
    max_width_of_target = 20
    min_size_of_target = 3
    range_spacing = 0.5
    azimuth_spacing = 0.5
    minimum_target_size = 3
    Number_of_Range_Looks = int(minimum_target_size / range_spacing) #
    N_subaperture = args.n_subaperture
    degrade_range_resolution = range_spacing * Number_of_Range_Looks
    max_size_range_bin = max_length_of_target / degrade_range_resolution
    max_size_azimuth_bin = 2000
    min_size_range_bin = min_size_of_target / degrade_range_resolution
    min_size_azimuth_bin = 100 / N_subaperture
    # Peak range sidelobe level (dB below main lobe) for the Taylor taper.
    # 35 ≈ Hamming, 45 = typical SAR, 55–65 = very low (broader main lobe).
    range_sll_db = 55.0
    range_taylor_nbar = 8

    # Range sidelobe suppression on the SLC. Tapering the range spectrum
    # before degradation stops bright scatterers (ships) from leaking
    # energy into neighbouring range bins and corrupting the azimuth
    # phase used by the downstream moving-target detector.
    s_windowed = apply_range_window(s, sll_db=range_sll_db, nbar=range_taylor_nbar)
    print(
        f"  range Taylor window applied: PSL ≤ -{range_sll_db:.0f} dB, "
        f"nbar={range_taylor_nbar}"
    )

    s_degraded, d_phase = degrade_range_resolution_range_sum(s_windowed, Number_of_Range_Looks)
    
    # CoV statistics need the *un-masked* sub-aperture amplitudes (the gate
    # is computed before masking happens). Only the 0-th and 2-nd moments
    # are needed, so the stack itself is dropped right after.
    _subaps_unmasked = compute_subapertures(s_degraded, N_subaperture)  # (N, sub_size, N_rg)
    sub_mean = _subaps_unmasked.mean(axis=0)                            # (sub_size, N_rg)
    sub_var = _subaps_unmasked.var(axis=0)                              # (sub_size, N_rg)
    del _subaps_unmasked

    # Build the gate at the *decimated* azimuth grid where the statistics live,
    # then upsample by N_subaperture along axis 0 so we can multiply it onto
    # s_degraded in-place. This avoids the (16x larger) full-resolution
    # subaperture stack the alternative would need.
    cov_sq = sub_var / (sub_mean ** 2 + 1e-12)         # (sub_size, N_rg)
    th = args.cov_th_mult * np.median(cov_sq)
    mask_dec = (cov_sq > th).astype(np.float32)        # (sub_size, N_rg), {0., 1.}

    # Upsample by N_subaperture: each decimated row covers N_subaperture rows
    # of s_degraded. (sub_size * N_subaperture) may be 1..N_subaperture-1 rows
    # short of N_az due to floor division in compute_subapertures; pad with the
    # last row so shapes match.
    mask_full = np.repeat(mask_dec, N_subaperture, axis=0)
    pad_rows = s_degraded.shape[0] - mask_full.shape[0]
    if pad_rows > 0:
        mask_full = np.vstack([mask_full, np.repeat(mask_full[-1:], pad_rows, axis=0)])
    assert mask_full.shape == s_degraded.shape, (mask_full.shape, s_degraded.shape)

    n_kept = int(mask_full.sum())
    cov_p50, cov_p90, cov_p99 = (
        float(np.percentile(cov_sq, p)) for p in (50.0, 90.0, 99.0)
    )
    print(
        f"  CoV gate: N_subaperture={N_subaperture}, "
        f"mult={args.cov_th_mult:g} · median(cov²)={cov_p50:.3g} → th={th:.3g} "
        f"(p90={cov_p90:.3g}, p99={cov_p99:.3g}), "
        f"keeping {n_kept}/{mask_full.size} pixels "
        f"({100*n_kept/mask_full.size:.2f}%)"
    )
    # Snapshot the un-masked amplitude before applying the CoV gate.
    # This is what the post-NMS bridge merge uses to detect a continuous
    # high-amplitude trail between boxes that the gated mask broke into
    # several short clusters along azimuth.
    amp_raw = np.abs(s_degraded).astype(np.float32)
    s_degraded = s_degraded * mask_full   # complex × float32 → complex

    # Canonical `subapertures` for Figure 3 / per-box COM / per-box 4×4
    # figures: re-compute from the *masked* s_degraded so each Doppler
    # sub-band only contains the pixels that passed the CoV gate.
    subapertures = compute_subapertures(s_degraded, N_subaperture)

    # Mark the local-amplitude peak of each dense detection cluster.
    # A pixel passes iff, inside a (N_Target_Azimuth_Width × N_Target_Range_Length)
    # box centred on it:
    #   (a) it is currently a detection             (mask_full == 1)
    #   (b) >50% of the box is also detected        (density test)
    #   (c) it is the local amplitude maximum       (peak test)
    # Vectorised with uniform_filter / maximum_filter — same logic as the
    # original nested loop but milliseconds instead of minutes, and without
    # the float-slice / complex-comparison bugs.
    from scipy.ndimage import uniform_filter, maximum_filter

    N_Target_Azimuth_Width = 100
    N_Target_Range_Length = 3
    box_size = (N_Target_Azimuth_Width, N_Target_Range_Length)
    box_area = N_Target_Azimuth_Width * N_Target_Range_Length

    # Density that the seed (N_Target_Azimuth_Width × N_Target_Range_Length)
    # window must contain before we declare its centre a peak. Higher →
    # fewer but more confident peaks.
    seed_density_threshold = 0.5       # try also: 0.25, 0.75

    amp = np.abs(s_degraded).astype(np.float32)
    det_count = uniform_filter(mask_full.astype(np.float32),
                               size=box_size, mode="constant") * box_area
    amp_max = maximum_filter(amp, size=box_size, mode="constant")

    is_detection = mask_full == 1
    is_dense = det_count > seed_density_threshold * box_area
    is_peak = amp == amp_max
    boundary_box = (is_detection & is_dense & is_peak).astype(np.float32)
    print(
        f"  boundary_box: window=({N_Target_Azimuth_Width},{N_Target_Range_Length}), "
        f"density>{seed_density_threshold}, peaks={int(boundary_box.sum())}"
    )


    # d_phase = np.angle(s_degraded[1:] * np.conj(s_degraded[:-1]))
    # d_phase = d_phase * mask_full[:-1, :]   # complex × float32 → complex
    # ↑ Don't recompute d_phase from the masked s_degraded: that would
    # stamp explicit zeros at every CoV-masked pixel, which then show
    # up as a horizontal band at φ=0 in every per-box phase scatter
    # plot and pull the per-box wrap-aware fit toward zero. d_phase
    # used downstream is the original unmasked one returned by
    # degrade_range_resolution_range_sum() — only the per-column
    # `estimate_phase_slope` diagnostic needs zero-bounded runs, and
    # that call is itself currently commented out below.



    print(f"  s_degraded: {s_degraded.shape},  d_phase: {d_phase.shape}")
    sw.mark("preprocessing (range window + degradation + CoV gate + subapertures)")
    # Run-length floor for estimate_phase_slope: only fit per-column
    # runs whose length is ≥ window_length/2 azimuth rows. Sized for
    # the typical target's longest contiguous-run length (~500 rows
    # after CoV masking) — so min_len = 250 keeps the target's
    # dominant run while rejecting short clutter runs. See
    # scripts/diag_slope_map_target.py for the per-column sweep.
    window_length = 500
    # Per-sample weight for the phasor LS fit: the modulus of the complex
    # product whose argument is d_phase, i.e. |s[n]|·|s[n+1]|. Zero
    # wherever either neighbour was masked by the CoV gate, so masked
    # samples drop out of the fit automatically. To additionally drop
    # noise-floor pixels, multiply by `(amp_pair > tau)` for some `tau`.
    # amp_pair = np.abs(s_degraded[:-1]) * np.abs(s_degraded[1:])
    # phase_weights = amp_pair.astype(np.float32)
    # detection_map_full_range = estimate_phase_slope(
    #     d_phase, window_length, weights=phase_weights,
    # )
    # detection_map = estimate_phase_slope(
    #     d_phase, window_length, weights=phase_weights,
    # )
    coherence = compute_subaperture_coherence(s_degraded)
    # detection_map *= coherence[:-1]

    # Per-column wrap-immune phasor-LS slope (rad/row), one slope per
    # maximal run of non-zero pixels in each range column. The input
    # `d_phase_for_slope_map` is a LOCAL masked copy of d_phase — the
    # original `d_phase` used for per-box phase scatter plots stays
    # unmasked so plots don't get a φ=0 band of dots. The aggregate
    # of `slope_map` per box (see compute_box_slope_col_total below)
    # is OR-ed into the motion gate to rescue large targets whose
    # box-wide single-line fit straddles a phase wrap.
    d_phase_for_slope_map = d_phase * mask_full[:-1, :]
    phase_weights = (
        np.abs(s_degraded[:-1]) * np.abs(s_degraded[1:])
    ).astype(np.float32)
    slope_map = estimate_phase_slope(
        d_phase_for_slope_map, window_length, weights=phase_weights,
    )
    sw.mark("coherence + per-column slope map (estimate_phase_slope)")

    # --- Main figure: |s_degraded| + boundary_box (always saved) ---
    # Two-panel layout, both in s_degraded pixel coords so box overlays
    # need no rescaling:
    #   left   |s_degraded|                  (range-degraded SLC amplitude)
    #   right  boundary_box  (peak per dense detection cluster)
    # The original 2×3 layout (input |s|, image-domain phase derivative,
    # d_phase, …) was slow to render on big patches and blocked
    # plt.show(); the remaining panels' draws live commented out below
    # so reverting is trivial.
    #
    #  Old layout, for reference:
    #    [0,0] Input SLC amplitude
    #    [0,1] Image-domain azimuth phase derivative (on raw s)
    #    [0,2] Detection map on full-range d_phase
    #    [1,0] Range-degraded SLC amplitude            ← left panel here
    #    [1,1] Azimuth phase derivative (d_phase)
    #    [1,2] boundary_box (peak per dense cluster)   ← right panel here
    fig, (ax_main, ax_bdy) = plt.subplots(
        1, 2, figsize=(20, 9), constrained_layout=True,
    )
    fig.suptitle(
        f"SAR moving target detection — {args.path.name}",
        fontsize=11,
    )
    # _show_slc(axes[0, 0], s, "Input SLC  |s|")
    # _show_image_domain_phase_derivative(axes[0, 1], s)
    # _show_detection_map(
    #     axes[0, 2],
    #     detection_map_full_range,
    #     "Detection map (full-range d_phase)",
    # )

    # Fraction of the box that must be detected (mask==1) to keep growing
    # and to pass the post-recentre check. Lower → boxes grow further into
    # sparse outskirts; higher → tighter, more conservative boxes.
    grow_density_threshold = 0.25      # try also: 0.4, 0.5

    peaks_yx = np.argwhere(boundary_box == 1.0)

    # Peak-level ROI hit count (peaks themselves are points, not boxes).
    if debug_roi is not None:
        y_lo, y_hi, x_lo, x_hi = debug_roi
        in_peaks = (
            (peaks_yx[:, 0] >= y_lo) & (peaks_yx[:, 0] <= y_hi)
            & (peaks_yx[:, 1] >= x_lo) & (peaks_yx[:, 1] <= x_hi)
        )
        n_peaks_roi = int(in_peaks.sum())
        print(
            f"[roi] {'peaks in boundary_box':38s}: "
            f"{n_peaks_roi}/{len(peaks_yx)} peaks inside ROI"
        )
        if n_peaks_roi == 0:
            print(
                "[roi]   → the CoV gate did not seed any peak inside the ROI; "
                "the target never enters the box pipeline. Inspect mask_full "
                "/ boundary_box, or lower --cov-th-mult to recover it."
            )

    boxes_yxhw = grow_and_recenter_boxes(
        peaks_yx,
        mask=mask_full,
        amp=amp,                                    # already = |s_degraded| float32
        initial_hw=(N_Target_Azimuth_Width, N_Target_Range_Length),
        az_step=5,
        rg_step=1,
        density_threshold=grow_density_threshold,
        max_h=int(max_size_azimuth_bin),
        max_w=int(max_size_range_bin),
    )
    _roi_print("grow_and_recenter_boxes", boxes_yxhw)
    if len(boxes_yxhw):
        med_h = int(np.median(boxes_yxhw[:, 2]))
        med_w = int(np.median(boxes_yxhw[:, 3]))
        max_h = int(boxes_yxhw[:, 2].max())
        max_w = int(boxes_yxhw[:, 3].max())
        print(
            f"  grown boxes: {len(boxes_yxhw)},  "
            f"median h×w = {med_h}×{med_w} px,  max h×w = {max_h}×{max_w} px"
        )

    # Post-growth azimuth-only extension: try to push each box's top and
    # bottom edges outward by 5% of the box's entry height, twice, accepting
    # each step iff the new strip's mask density ≥ extend_density_threshold.
    # The threshold is deliberately *looser* than grow_density_threshold
    # (0.10 vs 0.25): the growth pass has already done the hard work of
    # finding the dense core, so for the trailing 10% azimuth margin we
    # only need a faint hint that the target's signature continues —
    # otherwise narrow strips of mostly-black mask suppress the step even
    # when bright white pixels are visibly present further out. Capped at
    # +10% per side and at the global azimuth size cap.
    extend_density_threshold = 0.10
    boxes_yxhw = extend_boxes_azimuth_strong_signal(
        boxes_yxhw,
        mask=mask_full,
        step_frac=0.05,
        max_steps=2,
        density_threshold=extend_density_threshold,
        max_h=int(max_size_azimuth_bin),
    )
    _roi_print("extend_boxes_azimuth_strong_signal", boxes_yxhw)

    # NMS — two passes.
    #   (1) IoU NMS  — drops near-duplicate boxes from the same growth cluster.
    #   (2) centre-distance NMS — drops boxes whose centres are within one
    #       physical-target rectangle of a brighter kept centre, even when
    #       their boxes don't overlap enough for IoU > thresh.
    # Score each box by the MAX un-thresholded amplitude inside its
    # rectangle, not the centre-pixel amplitude. Recentring during growth
    # often drifts the centre off the actual peak (or onto a masked-out
    # pixel) — using amp[centre] then under-represents bright-but-large
    # targets and they get suppressed by dimmer neighbours in NMS.
    if len(boxes_yxhw):
        H_amp, W_amp = amp_raw.shape
        scores = np.empty(len(boxes_yxhw), dtype=np.float32)
        for i, (yc, xc, h, w) in enumerate(boxes_yxhw):
            yl = max(int(yc) - int(h) // 2, 0)
            yh = min(yl + int(h), H_amp)
            xl = max(int(xc) - int(w) // 2, 0)
            xh = min(xl + int(w), W_amp)
            sub = amp_raw[yl:yh, xl:xh]
            scores[i] = sub.max() if sub.size else 0.0
        n0 = len(scores)

        nms_iou_thresh = 0.3
        keep1 = nms_boxes(boxes_yxhw, scores, iou_thresh=nms_iou_thresh)
        boxes_yxhw = boxes_yxhw[keep1]
        scores = scores[keep1]
        n1 = len(boxes_yxhw)
        _roi_print(f"IoU NMS (>{nms_iou_thresh})", boxes_yxhw)

        # Anisotropic centre-distance gate (ellipse in metres). Loose
        # along azimuth (a moving target's signature smears hundreds of
        # metres along-track), strict in range (genuinely distinct
        # targets are usually well-separated cross-track, and a target
        # is only ~max_width_of_target metres wide).
        az_spacing_m = azimuth_spacing
        rg_spacing_m = range_spacing * Number_of_Range_Looks
        max_dist_az_m = 1000.0
        max_dist_rg_m = max_width_of_target
        keep2 = nms_by_centre_distance(
            boxes_yxhw, scores,
            az_spacing_m=az_spacing_m,
            rg_spacing_m=rg_spacing_m,
            max_dist_az_m=max_dist_az_m,
            max_dist_rg_m=max_dist_rg_m,
        )
        boxes_yxhw = boxes_yxhw[keep2]
        n2 = len(boxes_yxhw)
        _roi_print("centre-distance NMS", boxes_yxhw)

        # Bridge-merge boxes joined by a continuous high-amplitude trail
        # in the un-masked image: catches one physical target whose CoV
        # mask was split into several disjoint az clusters but whose raw
        # |s_degraded| stays bright across the gaps. Stricter anchor
        # (max of centres) and stat (strip median) make spurious merges
        # of distinct targets less likely.
        bridge_strength = 0.7
        max_rg_offset_m = max_width_of_target
        max_az_gap_m = 2000.0
        boxes_yxhw = merge_by_strip_amplitude(
            boxes_yxhw, amp_raw,
            az_spacing_m=az_spacing_m,
            rg_spacing_m=rg_spacing_m,
            bridge_strength=bridge_strength,
            max_rg_offset_m=max_rg_offset_m,
            max_az_gap_m=max_az_gap_m,
        )
        n3 = len(boxes_yxhw)
        _roi_print("bridge merge", boxes_yxhw)

        # Final geometric union of any boxes that still overlap.
        boxes_yxhw = merge_overlapping_boxes(boxes_yxhw)
        n4 = len(boxes_yxhw)
        _roi_print("overlap merge", boxes_yxhw)

        # Drop boxes whose range extent exceeds the physical range cap.
        # After the merging steps a few unions can grow wider than what a
        # single target can plausibly span in range; treat those as clutter.
        if len(boxes_yxhw):
            keep = boxes_yxhw[:, 3] <= max_size_range_bin
            boxes_yxhw = boxes_yxhw[keep]
        n5 = len(boxes_yxhw)
        _roi_print(f"range ≤ {int(max_size_range_bin)} px", boxes_yxhw)

        # Drop boxes with a near-flat amp-weighted phase ramp. A stationary
        # target (or pure noise) has no consistent azimuth phase gradient,
        # so its per-row slope from polyfit is ~0. A moving target shears
        # the phase along azimuth and the slope is non-trivial. The
        # threshold is on |np.degrees(slope)| (degrees per row); flag --min-
        # slope-deg=0 disables this filter.
        boxes_yxhw = filter_boxes_by_phase_slope(
            boxes_yxhw, d_phase, amp_raw, args.min_slope_deg,
            min_row_coherence=args.min_row_coherence,
            inlier_tol_rad=args.inlier_tol_rad,
        )
        n6 = len(boxes_yxhw)
        _roi_print(
            f"|slope| ≥ {args.min_slope_deg:g}°/row", boxes_yxhw,
        )

        # Drop boxes whose phase trace doesn't actually look like a line:
        # if the average (circular, wrapped) Euclidean distance from
        # phi[i] to the fit slope·i + intercept is large, the slope is
        # not describing the data well — clutter or noise rather than a
        # coherent moving-target ramp. Boxes with enough CoV-gate
        # detections inside them (> --residual-skip-det-frac · h_box)
        # bypass this filter and are kept regardless of residual — a
        # bright extended target is trusted on amplitude alone.
        # Flag --max-phase-residual-rad=0 disables this filter entirely.
        boxes_yxhw = filter_boxes_by_phase_residual(
            boxes_yxhw, d_phase, amp_raw, args.max_phase_residual_rad,
            mask_full=mask_full,
            detection_skip_frac=args.residual_skip_det_frac,
            min_row_coherence=args.min_row_coherence,
            inlier_tol_rad=args.inlier_tol_rad,
        )
        n6b = len(boxes_yxhw)
        _roi_print(
            f"mean |phi-fit| ≤ {args.max_phase_residual_rad:g} rad",
            boxes_yxhw,
        )

        # (Range-midpoint band filter --x-c-min / --x-c-max was removed
        # permanently — it silently emptied cropped patches whose
        # s_degraded had fewer columns than the hard-coded [65, 150]
        # default. Do NOT reintroduce a fixed range-pixel gate here.)

        # Pre-phase amplitude filter on subaperture COM motion. A truly
        # stationary target keeps its (amp-weighted) centre of mass in the
        # same place across all N Doppler sub-bands, so its peak-to-peak
        # COM shift is small. A range-only mover walks in range across
        # looks (target is at different range at different times); an
        # azimuth-only mover walks in subaperture-azimuth (Doppler offset
        # → linear-phase tilt → sub-pixel azimuth shift). Either way the
        # COM moves. We keep boxes whose az or rg COM PP exceeds the user
        # thresholds; this runs BEFORE any phase analysis.
        com_az_sub, com_rg = compute_subaperture_com_per_box(
            boxes_yxhw, subapertures, N_subaperture,
        )
        if args.com_filter_mode != "off" and len(boxes_yxhw):
            # NaN-safe peak-to-peak. Boxes with no valid look (all NaN) get 0.
            az_finite_any = np.isfinite(com_az_sub).any(axis=1)
            rg_finite_any = np.isfinite(com_rg).any(axis=1)
            az_pp = np.where(
                az_finite_any,
                np.nanmax(com_az_sub, axis=1) - np.nanmin(com_az_sub, axis=1),
                0.0,
            )
            rg_pp = np.where(
                rg_finite_any,
                np.nanmax(com_rg, axis=1) - np.nanmin(com_rg, axis=1),
                0.0,
            )
            az_ok = az_pp >= args.com_az_thresh
            rg_ok = rg_pp >= args.com_rg_thresh
            if args.com_filter_mode == "any":
                com_passes = az_ok | rg_ok
            else:                          # "both"
                com_passes = az_ok & rg_ok
            boxes_yxhw = boxes_yxhw[com_passes]
            com_az_sub = com_az_sub[com_passes]
            com_rg = com_rg[com_passes]
        n8 = len(boxes_yxhw)
        _roi_print(
            f"COM-PP filter ({args.com_filter_mode})", boxes_yxhw,
        )

        # Per-box, per-subaperture contrast gain on the boxes that
        # survived the COM filter. Computed here (rather than further
        # down) so it travels alongside com_az_sub / com_rg through the
        # remaining filters and lands in NPZ + CSV one-to-one with each
        # surviving box.
        contrast_gain, contrast_sub, contrast_full = (
            compute_subaperture_contrast_gain_per_box(
                boxes_yxhw, s_degraded, subapertures, N_subaperture,
            )
        )

        print(
            f"  NMS: {n0} → {n1} (IoU>{nms_iou_thresh}) "
            f"→ {n2} (centre ellipse: az<{max_dist_az_m:.0f} m, "
            f"rg<{max_dist_rg_m:.0f} m) "
            f"→ {n3} (bridge merge: median(strip) > "
            f"{bridge_strength}×max(centre amp), "
            f"|Δrg|<{max_rg_offset_m} m, |Δaz|<{max_az_gap_m} m) "
            f"→ {n4} (overlap merge) "
            f"→ {n5} (range ≤ {int(max_size_range_bin)} px) "
            f"→ {n6} (|slope| ≥ {args.min_slope_deg:g}°/row) "
            f"→ {n6b} (mean |phi - fit| ≤ {args.max_phase_residual_rad:g} rad "
            f"or det > {args.residual_skip_det_frac:g}·n_az) "
            f"→ {n8} (COM motion: {args.com_filter_mode}, "
            f"Δaz≥{args.com_az_thresh:g} sub-az px, "
            f"Δrg≥{args.com_rg_thresh:g} rg px) "
            f"— {100*(1 - n8/n0):.1f}% total suppressed"
        )
    else:
        # No seed peaks at all → make empty COM arrays for the save step.
        com_az_sub = np.empty((0, N_subaperture), dtype=np.float64)
        com_rg = np.empty((0, N_subaperture), dtype=np.float64)
        contrast_gain = np.empty(0, dtype=np.float64)
        contrast_sub = np.empty((0, N_subaperture), dtype=np.float64)
        contrast_full = np.empty(0, dtype=np.float64)

    if len(boxes_yxhw):
        az_pp_print = np.nanmax(com_az_sub, axis=1) - np.nanmin(com_az_sub, axis=1)
        rg_pp_print = np.nanmax(com_rg, axis=1)     - np.nanmin(com_rg, axis=1)
        print(
            f"  per-box subaperture COM (after filter): shape={com_az_sub.shape},"
            f" median Δaz={np.nanmedian(az_pp_print):.2f} sub-az px,"
            f" median Δrg={np.nanmedian(rg_pp_print):.2f} rg px"
        )
        # Contrast gain summary. With N_subaperture sub-bands a roughly
        # stationary target sits near ``N_subaperture`` (each sub-band ≈
        # same contrast as the full image); larger values indicate that
        # one or more sub-bands are individually sharper than the
        # degraded full aperture, the signature of a refocused mover.
        cg_finite = np.isfinite(contrast_gain)
        if cg_finite.any():
            cg_med = float(np.nanmedian(contrast_gain))
            cg_min = float(np.nanmin(contrast_gain))
            cg_max = float(np.nanmax(contrast_gain))
            print(
                f"  per-box contrast gain Σ_j C(sub_j)/C(s_degraded): "
                f"median={cg_med:.2f}, min={cg_min:.2f}, max={cg_max:.2f} "
                f"(reference N_subaperture={N_subaperture})"
            )
        else:
            print(
                f"  per-box contrast gain: no finite values "
                f"(reference N_subaperture={N_subaperture})"
            )

    sw.mark("box detection pipeline (grow + NMS + merges + slope/residual/COM filters)")

    # Per-box azimuth phase estimate + coherence-weighted linear fit.
    # Returns the actual per-row phase phi[i] used by the fit (the "phase
    # estimated for each target") plus its slope, intercept, and mean
    # wrapped Euclidean residual to the fit. The "slope·n" total phase
    # swing in radians is just slope · n_rows; box_residual_rad is the
    # exact same metric the --max-phase-residual-rad filter uses, kept
    # per box so it can be inspected in the CSV / NPZ.
    (box_phi, box_coh, box_slope_rad_per_row, box_intercept_rad,
     box_y0, box_n_rows, box_residual_rad, box_n_inliers) = (
        compute_box_phase_estimates(
            boxes_yxhw, d_phase, amp_raw,
            min_row_coherence=args.min_row_coherence,
            inlier_tol_rad=args.inlier_tol_rad,
        )
    )
    slope_totals = box_slope_rad_per_row * box_n_rows.astype(np.float64)

    # ------------------------------------------------------------------
    # Motion filter — three-tier rule on the box-wide phase slope.
    #
    #   |slope · n_rows| <  --slope-rad-lower    ⇒ DROP  (clear stationary)
    #   |slope · n_rows| >= --slope-rad-thresh   ⇒ KEEP  (clear mover)
    #   else (ambiguous middle band)             ⇒ KEEP iff
    #                                              az_pp ≥ --az-pp-sub-thresh
    #
    # The COM-velocity branch (|v_az| or |v_rg| ≥ --min-velocity-mps)
    # and the per-column wrap-immune slope_map aggregate are still
    # computed and saved in NPZ / CSV as diagnostics; they do NOT gate
    # the filter. Re-enable e.g. by OR-ing `is_moving_com` /
    # `is_moving_slope_col` into `is_moving` below.
    # ------------------------------------------------------------------
    sub_size = subapertures.shape[1]
    v_az_mps, v_rg_mps = compute_box_com_velocities(
        com_az_sub, com_rg,
        sub_size=sub_size,
        prf=args.prf,
        N_subaperture=N_subaperture,
        azimuth_spacing_m=azimuth_spacing,
        Number_of_Range_Looks=Number_of_Range_Looks,
        range_spacing_m=range_spacing,
    )

    # Diagnostic only — kept in CSV / NPZ but does NOT influence is_moving.
    is_moving_com = (
        (np.isfinite(v_az_mps) & (np.abs(v_az_mps) >= args.min_velocity_mps))
        | (np.isfinite(v_rg_mps) & (np.abs(v_rg_mps) >= args.min_velocity_mps))
    )
    # Per-box max |slope · run_length| over the box's range columns,
    # using the wrap-immune per-column phasor-LS slope_map. Kept as a
    # DIAGNOSTIC only (saved in NPZ / CSV): it does not drive the
    # motion gate any more — the three-tier rule below uses only the
    # box-wide single-line slope plus az_pp as the middle-band
    # tiebreaker.
    slope_col_total_rad = compute_box_slope_col_total(boxes_yxhw, slope_map)
    is_moving_slope_col = np.abs(slope_col_total_rad) >= args.slope_rad_thresh

    # Phase-free signal: peak-to-peak azimuth COM walk across the N
    # Doppler subapertures (NaN-safe). For a moving target the bright
    # spot shifts position from look to look; for stationary clutter
    # the COM is roughly constant.
    if len(com_az_sub):
        _az_finite = np.isfinite(com_az_sub).any(axis=1)
        az_pp_per_box = np.where(
            _az_finite,
            np.nanmax(com_az_sub, axis=1) - np.nanmin(com_az_sub, axis=1),
            0.0,
        )
    else:
        az_pp_per_box = np.zeros(len(boxes_yxhw), dtype=np.float64)
    is_moving_az_pp = az_pp_per_box >= args.az_pp_sub_thresh

    # ------------------------------------------------------------------
    # Three-tier motion gate on the box-wide phase slope:
    #   |slope·n| <  --slope-rad-lower    ⇒ DROP (clear stationary)
    #   |slope·n| >= --slope-rad-thresh   ⇒ KEEP (clear mover)
    #   else (ambiguous middle band)      ⇒ KEEP iff az_pp ≥ az-pp-sub-thresh
    # NaN |slope·n| is treated as "below lower" and dropped — the
    # phase fit failed entirely, we don't trust anything for that box.
    # ------------------------------------------------------------------
    abs_slope = np.where(
        np.isfinite(slope_totals), np.abs(slope_totals), 0.0,
    )
    is_moving_slope_box = abs_slope >= args.slope_rad_thresh           # clear yes
    slope_clear_no      = abs_slope <  args.slope_rad_lower            # clear no → drop
    slope_ambiguous     = (~is_moving_slope_box) & (~slope_clear_no)   # middle band

    is_moving = is_moving_slope_box | (slope_ambiguous & is_moving_az_pp)

    # Diagnostic union flag (any slope-route would have admitted it).
    # Used only for downstream save / NPZ; the actual is_moving above
    # is what gates the survival.
    is_moving_slope = is_moving_slope_box | (slope_ambiguous & is_moving_az_pp)

    n_pre_motion = len(boxes_yxhw)
    n_post_motion = int(is_moving.sum())

    # Pre-filter counts in each tier of the three-tier rule.
    n_clear_no    = int(slope_clear_no.sum())                       # dropped: |slope·n| < lower
    n_clear_yes   = int(is_moving_slope_box.sum())                  # kept:    |slope·n| ≥ thresh
    n_mid_keep    = int((slope_ambiguous & is_moving_az_pp).sum())  # kept by az_pp tiebreaker
    n_mid_drop    = int((slope_ambiguous & ~is_moving_az_pp).sum()) # middle band, failed az_pp

    if n_pre_motion and not is_moving.all():
        boxes_yxhw           = boxes_yxhw[is_moving]
        com_az_sub           = com_az_sub[is_moving]
        com_rg               = com_rg[is_moving]
        v_az_mps             = v_az_mps[is_moving]
        v_rg_mps             = v_rg_mps[is_moving]
        box_phi              = box_phi[is_moving]
        box_coh              = box_coh[is_moving]
        box_slope_rad_per_row = box_slope_rad_per_row[is_moving]
        box_intercept_rad    = box_intercept_rad[is_moving]
        box_y0               = box_y0[is_moving]
        box_n_rows           = box_n_rows[is_moving]
        box_residual_rad     = box_residual_rad[is_moving]
        box_n_inliers        = box_n_inliers[is_moving]
        slope_totals         = slope_totals[is_moving]
        contrast_gain        = contrast_gain[is_moving]
        contrast_sub         = contrast_sub[is_moving]
        contrast_full        = contrast_full[is_moving]
        slope_col_total_rad  = slope_col_total_rad[is_moving]
        az_pp_per_box        = az_pp_per_box[is_moving]
        # Per-box flags carried on the *post-filter* set as diagnostics.
        # After the three-tier gate every survivor satisfies
        # is_moving_slope == True (clear-yes OR ambiguous-and-az_pp).
        # is_moving_slope_col / is_moving_com record whether the box
        # would also have passed the per-column / COM-velocity routes.
        is_moving_com        = is_moving_com[is_moving]
        is_moving_slope      = is_moving_slope[is_moving]
        is_moving_slope_box  = is_moving_slope_box[is_moving]
        is_moving_slope_col  = is_moving_slope_col[is_moving]
        is_moving_az_pp      = is_moving_az_pp[is_moving]

    n_com_only = int((is_moving_com & ~is_moving_slope).sum())
    print(
        f"  motion filter: {n_pre_motion} → {n_post_motion} "
        f"(3-tier on |slope·n|: <{args.slope_rad_lower:g} drop, "
        f"≥{args.slope_rad_thresh:g} keep, else az_pp ≥ "
        f"{args.az_pp_sub_thresh:g} sub-az px)"
    )
    print(
        f"    clear-no drop (|slope·n| < {args.slope_rad_lower:g}): "
        f"{n_clear_no}, "
        f"clear-yes keep (|slope·n| ≥ {args.slope_rad_thresh:g}): "
        f"{n_clear_yes}, "
        f"middle-band kept by az_pp: {n_mid_keep}, "
        f"middle-band dropped: {n_mid_drop}, "
        f"COM-only diag (would have passed via |v_az| or |v_rg| ≥ "
        f"{args.min_velocity_mps:g} m/s but dropped here): {n_com_only}"
    )
    _roi_print(
        f"motion 3-tier on |slope·n| (low={args.slope_rad_lower:g}, "
        f"high={args.slope_rad_thresh:g})",
        boxes_yxhw,
    )

    strong_mask = is_moving_slope.copy()

    # ------------------------------------------------------------------
    # Optional refocus-gain gate (only when --refocus is set).
    # Drop boxes whose centred-QPE refocus *decreases* time-domain
    # contrast by more than |--refocus-min-gain-db| dB: the fitted
    # slope is inconsistent with the chip's spectral structure and
    # the detection is almost certainly spurious. NaN gains are kept
    # (no evidence either way).
    # ------------------------------------------------------------------
    if args.refocus and len(boxes_yxhw):
        H_full, W_full = s_degraded.shape
        gains_db_full = np.full(len(boxes_yxhw), np.nan, dtype=np.float64)
        for k in range(len(boxes_yxhw)):
            n_k = int(box_n_rows[k])
            y0_k = int(box_y0[k])
            y1_k = y0_k + n_k
            y_c_k = int(boxes_yxhw[k, 0])
            x_c_k = int(boxes_yxhw[k, 1])
            w_k = int(boxes_yxhw[k, 3])
            x0_k = max(0, x_c_k - w_k // 2)
            x1_k = min(W_full, x_c_k - w_k // 2 + w_k)
            s_k = float(box_slope_rad_per_row[k])
            if (
                n_k < 2 or x1_k <= x0_k or y1_k <= y0_k
                or y0_k < 0 or y1_k > H_full
                or not np.isfinite(s_k)
            ):
                continue
            chip_k = s_degraded[y0_k:y1_k, x0_k:x1_k]
            if chip_k.size == 0:
                continue
            corrected_k, _phi = _refocus_box_chip(chip_k, s_k)
            c_before = _normalized_variance(chip_k)
            c_after = _normalized_variance(corrected_k)
            if c_before > 0.0 and c_after > 0.0:
                gains_db_full[k] = 20.0 * np.log10(c_after / c_before)
        # keep iff gain >= threshold (or gain is NaN / no measurement)
        keep_mask = ~(
            np.isfinite(gains_db_full)
            & (gains_db_full < args.refocus_min_gain_db)
        )
        n_pre = len(boxes_yxhw)
        n_drop = int((~keep_mask).sum())
        if n_drop > 0:
            boxes_yxhw           = boxes_yxhw[keep_mask]
            com_az_sub           = com_az_sub[keep_mask]
            com_rg               = com_rg[keep_mask]
            v_az_mps             = v_az_mps[keep_mask]
            v_rg_mps             = v_rg_mps[keep_mask]
            box_phi              = box_phi[keep_mask]
            box_coh              = box_coh[keep_mask]
            box_slope_rad_per_row = box_slope_rad_per_row[keep_mask]
            box_intercept_rad    = box_intercept_rad[keep_mask]
            box_y0               = box_y0[keep_mask]
            box_n_rows           = box_n_rows[keep_mask]
            box_residual_rad     = box_residual_rad[keep_mask]
            box_n_inliers        = box_n_inliers[keep_mask]
            slope_totals         = slope_totals[keep_mask]
            contrast_gain        = contrast_gain[keep_mask]
            contrast_sub         = contrast_sub[keep_mask]
            contrast_full        = contrast_full[keep_mask]
            slope_col_total_rad  = slope_col_total_rad[keep_mask]
            az_pp_per_box        = az_pp_per_box[keep_mask]
            is_moving_com        = is_moving_com[keep_mask]
            is_moving_slope      = is_moving_slope[keep_mask]
            is_moving_slope_box  = is_moving_slope_box[keep_mask]
            is_moving_slope_col  = is_moving_slope_col[keep_mask]
            is_moving_az_pp      = is_moving_az_pp[keep_mask]
            strong_mask          = strong_mask[keep_mask]
        print(
            f"  refocus-gain filter: {n_pre} → {n_pre - n_drop} "
            f"(drop boxes with gain < {args.refocus_min_gain_db:g} dB)"
        )
        _roi_print(
            f"refocus-gain gate ≥ {args.refocus_min_gain_db:g} dB",
            boxes_yxhw,
        )

    # ------------------------------------------------------------------
    # Optional sub-aperture sharpness gate. Drop boxes whose per-box
    # sub-aperture sharpness ratio Σ_j C(sub_j) / C(s_degraded) (the
    # `contrast_gain` saved to CSV / NPZ; DIMENSIONLESS, not dB) is
    # finite and below --subaperture-sharpness-min. NaN ratios (empty
    # crop / zero mean) are kept (no evidence either way). Default
    # threshold -inf ⇒ filter is a no-op. Distinct from the dB refocus
    # gain drawn on `box_NNN_refocus.png`, which is gated by
    # --refocus-min-gain-db above.
    # ------------------------------------------------------------------
    if np.isfinite(args.subaperture_sharpness_min) and len(boxes_yxhw):
        keep_mask = ~(
            np.isfinite(contrast_gain)
            & (contrast_gain < args.subaperture_sharpness_min)
        )
        n_pre = len(boxes_yxhw)
        n_drop = int((~keep_mask).sum())
        if n_drop > 0:
            boxes_yxhw           = boxes_yxhw[keep_mask]
            com_az_sub           = com_az_sub[keep_mask]
            com_rg               = com_rg[keep_mask]
            v_az_mps             = v_az_mps[keep_mask]
            v_rg_mps             = v_rg_mps[keep_mask]
            box_phi              = box_phi[keep_mask]
            box_coh              = box_coh[keep_mask]
            box_slope_rad_per_row = box_slope_rad_per_row[keep_mask]
            box_intercept_rad    = box_intercept_rad[keep_mask]
            box_y0               = box_y0[keep_mask]
            box_n_rows           = box_n_rows[keep_mask]
            box_residual_rad     = box_residual_rad[keep_mask]
            box_n_inliers        = box_n_inliers[keep_mask]
            slope_totals         = slope_totals[keep_mask]
            contrast_gain        = contrast_gain[keep_mask]
            contrast_sub         = contrast_sub[keep_mask]
            contrast_full        = contrast_full[keep_mask]
            slope_col_total_rad  = slope_col_total_rad[keep_mask]
            az_pp_per_box        = az_pp_per_box[keep_mask]
            is_moving_com        = is_moving_com[keep_mask]
            is_moving_slope      = is_moving_slope[keep_mask]
            is_moving_slope_box  = is_moving_slope_box[keep_mask]
            is_moving_slope_col  = is_moving_slope_col[keep_mask]
            is_moving_az_pp      = is_moving_az_pp[keep_mask]
            strong_mask          = strong_mask[keep_mask]
        print(
            f"  sub-aperture sharpness filter: {n_pre} → {n_pre - n_drop} "
            f"(drop boxes with Σ_j C(sub_j)/C(s_degraded) < "
            f"{args.subaperture_sharpness_min:g})"
        )
        _roi_print(
            f"sub-aperture sharpness gate ≥ "
            f"{args.subaperture_sharpness_min:g}",
            boxes_yxhw,
        )

    sw.mark("phase-fit motion filter + refocus/sharpness gates")

    strong_boxes = boxes_yxhw[strong_mask]
    weak_boxes = boxes_yxhw[~strong_mask]
    print(
        f"  → boxes with |slope·n| ≥ {args.slope_rad_thresh:g} rad "
        f"(red overlays): {int(strong_mask.sum())}/{len(boxes_yxhw)}"
    )

    # Persist d_phase + final boxes for downstream analysis scripts
    # (e.g. scripts/view_dphase_boxes.py). Skipped on --no-save.
    if args.save is not None:
        data_path = args.save.with_suffix(".npz")
        # Per-box phase estimate. box_phi[k, i] is the amplitude-weighted
        # circular-mean azimuth phase derivative for box k at
        # azimuth row box_y0[k] + i, for i = 0 .. box_n_rows[k] - 1; NaN
        # beyond that. box_coh is the matching per-row coherence weight.
        # box_slope_rad_per_row[k] · i + box_intercept_rad[k] reconstructs
        # the linear fit at row box_y0[k] + i.
        npz_payload = dict(
            d_phase=d_phase,
            amp_raw=amp_raw,
            boxes_yxhw=boxes_yxhw,
            slope_totals_rad=slope_totals,
            slope_rad_thresh=np.float64(args.slope_rad_thresh),
            slope_rad_lower=np.float64(args.slope_rad_lower),
            s_degraded_shape=np.asarray(s_degraded.shape, dtype=np.int64),
            com_az_sub=com_az_sub,
            com_rg=com_rg,
            contrast_gain=contrast_gain,
            contrast_sub=contrast_sub,
            contrast_full=contrast_full,
            N_subaperture=np.int64(N_subaperture),
            box_phi_rad=box_phi,
            box_coh=box_coh,
            box_slope_rad_per_row=box_slope_rad_per_row,
            box_intercept_rad=box_intercept_rad,
            box_y0=box_y0,
            box_n_rows=box_n_rows,
            box_residual_rad=box_residual_rad,
            box_n_inliers=box_n_inliers,
            max_phase_residual_rad=np.float64(args.max_phase_residual_rad),
            min_row_coherence=np.float64(args.min_row_coherence),
            inlier_tol_rad=np.float64(args.inlier_tol_rad),
            v_az_mps=v_az_mps,
            v_rg_mps=v_rg_mps,
            is_moving_com=is_moving_com.astype(np.int8),
            is_moving_slope=is_moving_slope.astype(np.int8),
            is_moving_slope_box=is_moving_slope_box.astype(np.int8),
            is_moving_slope_col=is_moving_slope_col.astype(np.int8),
            is_moving_az_pp=is_moving_az_pp.astype(np.int8),
            slope_col_total_rad=slope_col_total_rad,
            az_pp_per_box=az_pp_per_box,
            az_pp_sub_thresh=np.float64(args.az_pp_sub_thresh),
            prf_hz=np.float64(args.prf),
            min_velocity_mps=np.float64(args.min_velocity_mps),
        )
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(data_path, **npz_payload)
            print(f"Saved d_phase + amp_raw + boxes → {data_path}")
        except PermissionError as exc:
            fallback = Path("/tmp") / data_path.name
            print(
                f"  (warning) cannot write {data_path} ({exc.strerror});"
                f" falling back to {fallback}"
            )
            np.savez_compressed(fallback, **npz_payload)
            print(f"Saved d_phase + amp_raw + boxes → {fallback}")

    n_strong = int(strong_mask.sum())

    # Main figure panels — both in s_degraded pixel coords, so the
    # strong-box overlay uses the box arrays as-is (no Number_of_Range_-
    # Looks rescaling needed). Left: range-degraded SLC amplitude.
    # Right: boundary_box peak map (one peak per dense detection
    # cluster — the inputs to grow_and_recenter_boxes).
    _show_slc(
        ax_main, s_degraded,
        f"|s_degraded| (range-degraded SLC): "
        f"{n_strong}/{len(boxes_yxhw)} boxes "
        f"|slope·n| > {args.slope_rad_thresh:g} rad",
    )
    _overlay_boxes(ax_main, strong_boxes, color="red", lw=1.4)

    _show_slc(
        ax_bdy, boundary_box,
        f"boundary_box (peak per dense cluster): "
        f"{int(boundary_box.sum())} peaks",
    )
    _overlay_boxes(ax_bdy, strong_boxes, color="red", lw=1.4)

    # The d_phase panel of the old 2×3 layout is disabled to keep the
    # main figure light. Re-enable by restoring `plt.subplots(2, 3, …)`
    # above and uncommenting this block.
    # _show_phase_derivative(
    #     axes[1, 1], d_phase,
    #     f"d_phase (degraded): {n_strong}/{len(boxes_yxhw)} boxes "
    #     f"|slope·n| > {args.slope_rad_thresh:g} rad",
    # )
    # _overlay_boxes(axes[1, 1], strong_boxes, color="red", lw=1.4)


    # n_rows = int(np.floor(np.sqrt(N_subaperture)))
    # n_cols = int(np.ceil(N_subaperture / n_rows))
    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14), constrained_layout=True)
    # axes_flat = np.atleast_1d(axes).ravel()
    # for i in range(N_subaperture):
    #     _show_slc(axes_flat[i], subapertures[i], f"subaperture {i}")
    # for j in range(N_subaperture, axes_flat.size):
    #     axes_flat[j].set_visible(False)

    # Per-pixel statistics across the N=8 sub-aperture amplitudes.
    # Stationary scatterers → consistent |s_k| across looks → low variance.
    # Moving targets / non-stationary scenes → |s_k| varies with Doppler band
    # → high variance (and high variance / mean^2).
    # sub_mean = subapertures.mean(axis=0)
    # sub_var = subapertures.var(axis=0)

    # Plot a handful of range columns of d_phase as 1-D azimuth traces.
    # plt.plot(2D) draws one line per column, with row index on the x-axis,
    # so axis 0 (azimuth) ends up on the horizontal axis as requested.
    # range_cols = slice(10, 15)
    # plt.figure()
    # plt.plot(d_phase[:, range_cols],'o')
    # plt.xlabel("azimuth pixel")
    # plt.ylabel(r"$\Delta\varphi$  [rad]")
    # plt.title(f"d_phase along azimuth, range cols {range_cols.start}:{range_cols.stop}")

    # range_cols = slice(124, 134)
    # plt.figure()
    # plt.plot(d_phase[:, range_cols],'o')
    # plt.xlabel("azimuth pixel")
    # plt.ylabel(r"$\Delta\varphi$  [rad]")
    # plt.title(f"d_phase along azimuth, range cols {range_cols.start}:{range_cols.stop}")

    # d = np.angle(s[1:, :] * np.conj(s[:-1, :]))
    # range_cols = slice(640, 645)
    # plt.figure()
    # plt.plot(d[:, range_cols],'o')
    # plt.xlabel("azimuth pixel")
    # plt.ylabel(r"$\Delta\varphi$  [rad]")
    # plt.title(f"d_phase along azimuth, range cols {range_cols.start}:{range_cols.stop}")


    # fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True)
    # _show_slc(axes_cmp[0], s_degraded, "|s_degraded|  (coherent)")
    # _show_slc(
    #     axes_cmp[1],
    #     sub_var/sub_mean**2,
    #     r"sub_var/sub_mean$_k\,|s_k|$  (incoherent average)",
    #     sigma=1.0,
    # )
    # _show_slc(
    #     axes_cmp[2],
    #     sub_var,
    #     r"var$_k\,|s_k|$  (per-pixel variance over $N=8$ looks)",
    #     sigma=1.0,
    # )


    

    # Number of grown boxes — used by Figure 4 (--display-all-mode) and
    # by the per-box save / NPZ blocks further down. Defined once,
    # outside any gate, so the rest of the function can rely on it
    # regardless of `--debug` / `--display-all-mode`.
    K = len(boxes_yxhw)

    # --- Debug-only: mask + grown boxes overlay ---
    # Diagnostic for the box pipeline. Cheap to build but irrelevant
    # outside of debugging the mask / growth / NMS chain, so it is gated
    # on --debug and skipped in the default run. The two raw arrays
    # (slopes, d_phase) saved further down share this gate.
    if args.debug:
        # --- Figure mb: CoV detection mask + grown boxes + seed peaks ---
        # Shows the binary detection mask used by the box-growth (and
        # post-growth azimuth-extension) density gates, overlaid with
        # the final bounding boxes (cyan) and the boundary_box seed
        # peaks (red dots) that grow_and_recenter_boxes started from.
        # Boxes and peaks live in s_degraded pixel coords, same as
        # mask_full, so no scaling is needed.
        fig_mb, ax_mb = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)
        fig_mb.suptitle(
            f"mask_full + grown boxes — {args.path.name}",
            fontsize=11,
        )
        ax_mb.imshow(
            mask_full, cmap="gray", aspect="auto",
            vmin=0.0, vmax=1.0, interpolation="nearest",
        )
        # Seed peaks (one per dense detection cluster) — the inputs to
        # grow_and_recenter_boxes. Showing them next to the grown boxes
        # makes it easy to see how far each seed grew in azimuth/range.
        peak_rows = np.argwhere(boundary_box == 1.0)
        if peak_rows.size:
            ax_mb.scatter(
                peak_rows[:, 1], peak_rows[:, 0],
                s=14, marker="x", c="red", linewidths=0.8,
                label=f"boundary_box peaks ({len(peak_rows)})",
            )
        _overlay_boxes(ax_mb, boxes_yxhw, color="cyan", lw=1.2)
        ax_mb.set_title(
            f"mask_full (white = detection), "
            f"{len(boxes_yxhw)} grown boxes (cyan)"
        )
        ax_mb.set_xlabel("range pixel (s_degraded)")
        ax_mb.set_ylabel("azimuth pixel")
        if peak_rows.size:
            ax_mb.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # --- Rich display figures (subapertures, d_phase fit, per-box COM) ---
    # The expensive grid figures: per-Doppler-band subaperture
    # amplitudes, the d_phase + per-box slope·n overview, and the
    # per-box subaperture-COM trajectory grid. They scale poorly with
    # K and N_subaperture, so they are gated on --display-all-mode and
    # skipped entirely when it is unset (default). Independent of
    # --debug.
    if args.display_all_mode:
        # --- Figure 2: d_phase  vs.  per-box slope·n total phase swing ---
        # Paint each box's slope_total (rad) into a NaN-filled canvas. Outside
        # the boxes is NaN ⇒ rendered transparent so the diverging colormap is
        # only "spent" on actual targets.
        per_box_slope = np.full(d_phase.shape, np.nan, dtype=np.float64)
        n_az_d, n_rg_d = d_phase.shape
        for (y_c, x_c, h, w), s_tot in zip(boxes_yxhw, slope_totals):
            if not np.isfinite(s_tot):
                continue
            y0 = max(0, int(round(y_c - h / 2)))
            y1 = min(n_az_d, int(round(y_c + h / 2)))
            x0 = max(0, int(round(x_c - w / 2)))
            x1 = min(n_rg_d, int(round(x_c + w / 2)))
            if y1 <= y0 or x1 <= x0:
                continue
            per_box_slope[y0:y1, x0:x1] = float(s_tot)

        finite = per_box_slope[np.isfinite(per_box_slope)]
        box_vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
        box_vmax = max(box_vmax, float(args.slope_rad_thresh))

        fig2, axes2 = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        fig2.suptitle(
            f"d_phase and per-box slope·n — {args.path.name}",
            fontsize=11,
        )
        _show_phase_derivative(
            axes2[0], d_phase,
            f"d_phase (degraded): {n_strong}/{len(boxes_yxhw)} boxes "
            f"|slope·n| > {args.slope_rad_thresh:g} rad",
        )
        _overlay_boxes(axes2[0], strong_boxes, color="red", lw=1.4)

        cmap = plt.get_cmap("seismic").copy()
        cmap.set_bad(color=(0, 0, 0, 0))   # NaN → transparent
        im2 = axes2[1].imshow(
            per_box_slope, cmap=cmap, aspect="auto",
            vmin=-box_vmax, vmax=+box_vmax,
            interpolation="nearest",
        )
        axes2[1].set_facecolor("0.92")
        axes2[1].set_title(
            f"per-box slope·n  (rad)  —  "
            f"{n_strong}/{len(boxes_yxhw)} boxes |slope·n| > {args.slope_rad_thresh:g} rad\n"
            f"max |slope·n| = {box_vmax:.2f} rad"
        )
        axes2[1].set_xlabel("range pixel")
        axes2[1].set_ylabel("azimuth pixel")
        plt.colorbar(im2, ax=axes2[1], label="slope · n_rows  [rad]")
        _overlay_boxes(axes2[1], strong_boxes, color="red", lw=1.4)

        # --- Figure 3: per-Doppler-band subaperture amplitudes ---
        # Lay them out on a near-square grid. Subapertures are shape
        # (N_subaperture, sub_size, N_range_degraded), already amplitudes.
        n_sub = subapertures.shape[0]
        n_cols_sub = int(np.ceil(np.sqrt(n_sub)))
        n_rows_sub = int(np.ceil(n_sub / n_cols_sub))
        fig3, axes3 = plt.subplots(
            n_rows_sub, n_cols_sub,
            figsize=(4 * n_cols_sub, 3 * n_rows_sub),
            constrained_layout=True,
        )
        fig3.suptitle(
            f"Subaperture amplitudes — {args.path.name}\n"
            f"N_subaperture = {n_sub} Doppler sub-bands, "
            f"sub_size = {subapertures.shape[1]} az px each",
            fontsize=11,
        )
        axes3_flat = np.atleast_1d(axes3).ravel()
        # Common amplitude clip so brightness is comparable across panels:
        # mean ± 2σ over the whole stack, with vmin floored at 0 (amplitudes
        # are non-negative).
        sub_mu = float(subapertures.mean())
        sub_sd = float(subapertures.std())
        sub_vmin = max(0.0, sub_mu - 2.0 * sub_sd)
        sub_vmax = max(sub_mu + 2.0 * sub_sd, sub_vmin + 1e-12)
        for k in range(n_sub):
            ax_k = axes3_flat[k]
            im_k = ax_k.imshow(
                subapertures[k], cmap="viridis", aspect="auto",
                vmin=sub_vmin, vmax=sub_vmax,
            )
            ax_k.set_title(f"subaperture {k}", fontsize=9)
            ax_k.set_xlabel("range pixel", fontsize=8)
            ax_k.set_ylabel("azimuth pixel", fontsize=8)
            ax_k.tick_params(labelsize=7)
        for j in range(n_sub, axes3_flat.size):
            axes3_flat[j].set_visible(False)
        fig3.colorbar(
            im_k, ax=axes3_flat.tolist(), location="right",
            label="|s_sub|", shrink=0.6,
        )

        # --- Figure 4: per-box COM trajectory across the N subapertures ---
        # One small panel per box (whole population, not just strong-slope).
        if K:
            n_cols_box = int(np.ceil(np.sqrt(K)))
            n_rows_box = int(np.ceil(K / n_cols_box))
            fig4, axes4 = plt.subplots(
                n_rows_box, n_cols_box,
                figsize=(2.4 * n_cols_box, 2.2 * n_rows_box),
                constrained_layout=True,
            )
            fig4.suptitle(
                f"Per-box subaperture COM trajectory — {args.path.name}\n"
                f"x = range pixel (s_degraded),  y = azimuth pixel (subaperture grid),"
                f"  colour = subaperture index 0..{n_sub - 1}",
                fontsize=11,
            )
            axes4_flat = np.atleast_1d(axes4).ravel()
            sub_idx = np.arange(n_sub, dtype=np.float64)
            for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                ax_k = axes4_flat[k]
                cx = com_rg[k]
                cy = com_az_sub[k]
                valid = np.isfinite(cx) & np.isfinite(cy)
                if valid.any():
                    ax_k.plot(cx[valid], cy[valid], color="0.6", lw=0.7, zorder=1)
                    ax_k.scatter(
                        cx[valid], cy[valid],
                        c=sub_idx[valid], cmap="viridis",
                        s=22, edgecolor="k", linewidth=0.3, zorder=2,
                        vmin=0, vmax=n_sub - 1,
                    )
                slope_k = slope_totals[k]
                star = "*" if strong_mask[k] else " "
                ax_k.set_title(
                    f"{star}box {k:02d}  y={int(y_c)} x={int(x_c)}\n"
                    f"slope·n={slope_k:+.2f} rad" if np.isfinite(slope_k)
                    else f"{star}box {k:02d}  y={int(y_c)} x={int(x_c)}\nslope·n=NaN",
                    fontsize=8,
                )
                ax_k.set_xlabel("range pix", fontsize=7)
                ax_k.set_ylabel("az pix (sub)", fontsize=7)
                ax_k.tick_params(labelsize=6)
                ax_k.grid(True, alpha=0.3)
            for j in range(K, axes4_flat.size):
                axes4_flat[j].set_visible(False)

    # The per-box azimuth phase-derivative trace + wrapped linear fit
    # used to live in a single Figure 5 grid (n_rows × n_cols of
    # subplots). That grid became unreadable on K > 50 boxes and slow
    # to render / open interactively, so it was split into one PNG per
    # box. Those PNGs are written inside the args.save block below,
    # together with the other per-box per_box_dir/box_NNN_*.png files.

    if args.save is not None:
        # Resolve the actual save directory once. Probe by trying to create
        # args.save.parent; on PermissionError fall back to /tmp. This way
        # main figures, per-box figures and data files all land together.
        try:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            (args.save.parent / ".write_probe").write_text("")
            (args.save.parent / ".write_probe").unlink(missing_ok=True)
            save_dir = args.save.parent
        except PermissionError as exc:
            print(
                f"  (warning) cannot write into {args.save.parent} "
                f"({exc.strerror}); falling back to /tmp"
            )
            save_dir = Path("/tmp")

        stem = args.save.stem
        path_main = save_dir / args.save.name
        path_csv = save_dir / f"{stem}_box_stats.csv"
        per_box_dir = save_dir / f"{stem}_per_box"

        # Save the main overview figure (always written).
        fig.savefig(path_main, dpi=150);          print(f"Saved → {path_main}")

        # Debug-only artefacts. The mask + grown-boxes overlay
        # (`_mask_boxes.png`) and the two raw arrays (slopes, d_phase)
        # are written only when `--debug` is set; they mirror the
        # `if args.debug:` block that constructs `fig_mb` further up.
        if args.debug:
            path_mask_boxes = save_dir / f"{stem}_mask_boxes.png"
            path_slopes = save_dir / f"{stem}_slopes.npy"
            path_dphase = save_dir / f"{stem}_dphase.npy"

            fig_mb.savefig(path_mask_boxes, dpi=150); print(f"Saved → {path_mask_boxes}")
            np.save(path_slopes, slope_totals);       print(f"Saved → {path_slopes}")
            np.save(path_dphase, d_phase);            print(f"Saved → {path_dphase}")

        # Rich display figures. Written only when `--display-all-mode`
        # is set; they mirror the `if args.display_all_mode:` block
        # that constructs `fig2`, `fig3` and `fig4` further up.
        if args.display_all_mode:
            path_dphase_fit = save_dir / f"{stem}_dphase_fit.png"
            path_subaps = save_dir / f"{stem}_subapertures.png"
            path_box_com = save_dir / f"{stem}_box_com.png"

            fig2.savefig(path_dphase_fit, dpi=150);   print(f"Saved → {path_dphase_fit}")
            fig3.savefig(path_subaps, dpi=150);       print(f"Saved → {path_subaps}")
            if K:
                fig4.savefig(path_box_com, dpi=150);  print(f"Saved → {path_box_com}")

        # Per-box CSV — slim view holding only the target position and the
        # numeric quantities that drive each elimination stage:
        #   * slope_rad_per_row   → --max-phase-slope filter
        #   * phase_residual_rad  → --max-phase-residual-rad filter
        #   * slope_total_rad     → 3-tier motion filter (slope·n vs
        #                           --slope-rad-lower / --slope-rad-thresh)
        #   * az_pp_sub_px        → middle-band tiebreaker
        #                           (--az-pp-sub-thresh)
        #   * contrast_gain       → --refocus-min-gain-db gate
        # Diagnostic-only fields (v_az_mps, contrast_full, is_moving_com,
        # …), the linear-fit reconstruction (intercept_rad, y0, n_rows,
        # n_inliers) and the per-subaperture arrays (com_az_sub_*,
        # com_rg_*, contrast_sub_*) all live in the NPZ payload — see the
        # ``npz_payload`` dict written further down.
        with path_csv.open("w") as f:
            f.write(
                "idx,y_c,x_c,h,w,"
                "slope_total_rad,slope_rad_per_row,phase_residual_rad,"
                "az_pp_sub_px,contrast_gain\n"
            )
            n_csv_rows = 0
            for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                # CSV holds only the post-final-filter targets — i.e. the
                # boxes flagged "strong" (|slope·n| ≥ slope_rad_thresh,
                # the red overlays in Figure 1). The full set of motion-
                # filter survivors lives in the NPZ. ``idx`` is left equal
                # to the post-motion-filter index so it still matches the
                # per-box PNG filenames (``box_<idx>.png``) and the NPZ
                # row order.
                if not bool(strong_mask[k]):
                    continue
                az_finite = np.isfinite(com_az_sub[k])
                az_pp_k = (
                    com_az_sub[k][az_finite].max() - com_az_sub[k][az_finite].min()
                    if az_finite.any() else float("nan")
                )
                fields: list[str] = [
                    str(k), str(int(y_c)), str(int(x_c)),
                    str(int(h)), str(int(w)),
                    f"{slope_totals[k]:+.6e}" if np.isfinite(slope_totals[k]) else "",
                    (f"{box_slope_rad_per_row[k]:+.6e}"
                     if np.isfinite(box_slope_rad_per_row[k]) else ""),
                    (f"{box_residual_rad[k]:.6f}"
                     if np.isfinite(box_residual_rad[k]) else ""),
                    f"{az_pp_k:.6f}" if np.isfinite(az_pp_k) else "",
                    (f"{contrast_gain[k]:.6f}"
                     if np.isfinite(contrast_gain[k]) else ""),
                ]
                f.write(",".join(fields) + "\n")
                n_csv_rows += 1
        print(f"Saved → {path_csv} ({n_csv_rows}/{len(boxes_yxhw)} strong boxes)")

        # Per-box artefacts. The 4×4 subaperture-crops figure
        # (box_NNN.png), the per-box azimuth-FFT figure
        # (box_NNN_fft.png) and the per-subaperture FFT figure
        # (box_NNN_sub_fft.png) were dropped — none of them are inspected
        # any more. Only the per-box d_phase phase trace + wrapped linear
        # fit (box_NNN_dphase.png) is still written.
        if K:
            per_box_dir.mkdir(parents=True, exist_ok=True)

            # --- Per-box d_phase phase trace + wrapped linear fit ---
            # One PNG per box, replacing the old dense Figure 5 grid.
            # Same content as before (scatter of phi vs absolute az row,
            # coloured by per-row coherence, plus the wrapped grid-fit
            # red line and the diagnostics in the title) but at a
            # readable size and individually openable.
            for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                n_k = int(box_n_rows[k])
                slope_k_pr = float(box_slope_rad_per_row[k])
                b_k = float(box_intercept_rad[k])
                star = "*" if strong_mask[k] else " "
                fig_dphi, ax_dphi = plt.subplots(
                    figsize=(10, 4.5), constrained_layout=True,
                )
                n_used_k = 0
                if n_k >= 1:
                    i_loc = np.arange(n_k, dtype=np.float64)
                    y_abs = int(box_y0[k]) + i_loc
                    phi_k = box_phi[k, :n_k]
                    coh_k = box_coh[k, :n_k]
                    valid = np.isfinite(phi_k)
                    if valid.any():
                        sc = ax_dphi.scatter(
                            y_abs[valid], phi_k[valid],
                            c=coh_k[valid], cmap="viridis",
                            s=10, vmin=0.0, vmax=1.0,
                            edgecolor="none", zorder=2,
                        )
                        fig_dphi.colorbar(
                            sc, ax=ax_dphi, location="right",
                            shrink=0.85, label="coh",
                        )
                    used_mask = (
                        np.isfinite(phi_k) & np.isfinite(coh_k)
                        & (coh_k > args.min_row_coherence)
                    )
                    n_used_k = int(used_mask.sum())
                    if np.isfinite(slope_k_pr) and np.isfinite(b_k):
                        line_wrapped = np.angle(
                            np.exp(1j * (slope_k_pr * i_loc + b_k))
                        )
                        if line_wrapped.size > 1:
                            d_line = np.abs(np.diff(line_wrapped))
                            jumps = np.where(d_line > np.pi)[0]
                            if jumps.size:
                                line_wrapped = line_wrapped.copy()
                                line_wrapped[jumps] = np.nan
                        ax_dphi.plot(
                            y_abs, line_wrapped,
                            color="red", lw=1.1, zorder=3,
                        )
                ax_dphi.set_ylim(-np.pi, np.pi)
                ax_dphi.axhline(0.0, color="0.5", lw=0.5, ls="--", zorder=1)
                slope_tot_k = slope_totals[k]
                res_k = float(box_residual_rad[k])
                vaz_k = float(v_az_mps[k])
                vrg_k = float(v_rg_mps[k])
                v_str = (
                    f"  v=({vaz_k:+.2f},{vrg_k:+.2f}) m/s"
                    if (np.isfinite(vaz_k) or np.isfinite(vrg_k))
                    else ""
                )
                coh_pct_k = (100.0 * n_used_k / n_k) if n_k > 0 else 0.0
                coh_str = (
                    f"  coh>{args.min_row_coherence:g}: "
                    f"{n_used_k}/{n_k} ({coh_pct_k:.1f}%)"
                )
                n_in_k = int(box_n_inliers[k])
                in_pct_k = (100.0 * n_in_k / n_k) if n_k > 0 else 0.0
                inlier_str = (
                    f"  inliers(d≤{args.inlier_tol_rad:g}): "
                    f"{n_in_k}/{n_k} ({in_pct_k:.1f}%)"
                )
                if np.isfinite(slope_tot_k) and np.isfinite(slope_k_pr):
                    res_str = (
                        f"  res={res_k:.2f} rad"
                        if np.isfinite(res_k) else ""
                    )
                    ax_dphi.set_title(
                        f"{star}box {k:02d}  y={int(y_c)} x={int(x_c)}"
                        f"{v_str}\n"
                        f"slope={np.degrees(slope_k_pr):+.2f}°/row, "
                        f"slope·n={slope_tot_k:+.2f} rad"
                        f"{res_str}{coh_str}{inlier_str}",
                        fontsize=9,
                    )
                else:
                    ax_dphi.set_title(
                        f"{star}box {k:02d}  y={int(y_c)} x={int(x_c)}"
                        f"{v_str}\nfit failed{coh_str}{inlier_str}",
                        fontsize=9,
                    )
                ax_dphi.set_xlabel("absolute az row")
                ax_dphi.set_ylabel(r"$\Delta\varphi$  [rad]")
                ax_dphi.grid(True, alpha=0.3)

                path_box_dphi_k = per_box_dir / f"box_{k:03d}_dphase.png"
                fig_dphi.savefig(path_box_dphi_k, dpi=130)
                plt.close(fig_dphi)
            print(
                f"Saved {K} per-box phase PNGs → "
                f"{per_box_dir}/box_000_dphase.png … "
                f"box_{K-1:03d}_dphase.png"
            )

            # --- Per-box azimuth FFT on the FULL-RESOLUTION SLC ---
            # The detections live on the `s_degraded` range grid (axis 1
            # decimated by `Number_of_Range_Looks`). To crop the same
            # boxes from the full-resolution SLC `s`, rescale the range
            # coordinates (x_centre and w) by `Number_of_Range_Looks`;
            # y_centre and h are unchanged because azimuth is untouched
            # by range degradation. The cropped chip's azimuth FFT
            # `|FFT_az|` is written to `box_NNN_fft.png` next to
            # `box_NNN_dphase.png`.
            H_az_full, W_rg_full = s.shape
            n_fft_written = 0
            for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                y_int = int(y_c)
                h_int = int(h)
                y0_full = max(0, y_int - h_int // 2)
                y1_full = min(H_az_full, y0_full + h_int)
                x_full = int(x_c) * Number_of_Range_Looks
                w_full = int(w) * Number_of_Range_Looks
                x0_full = max(0, x_full - w_full // 2)
                x1_full = min(W_rg_full, x0_full + w_full)
                if x1_full <= x0_full or y1_full - y0_full < 2:
                    continue
                chip_full = s[y0_full:y1_full, x0_full:x1_full]
                if chip_full.size == 0:
                    continue
                fft_az = np.fft.fftshift(
                    np.fft.fft(chip_full, axis=0), axes=0,
                )
                amp_fft = np.abs(fft_az).astype(np.float32)
                n_az_chip = chip_full.shape[0]
                # imshow extent: row 0 of the fftshifted FFT is the most
                # negative Doppler bin, so map it to the top of the axes
                # (top < bottom inverts the y-axis as desired).
                f_top = -0.5
                f_bot = 0.5 - 1.0 / n_az_chip

                fig_fft, ax_fft = plt.subplots(
                    figsize=(7, 5), constrained_layout=True,
                )
                vmin = float(np.nanpercentile(amp_fft, 1.0))
                vmax = float(np.nanpercentile(amp_fft, 99.0))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                im_fft = ax_fft.imshow(
                    amp_fft, aspect="auto", cmap="viridis",
                    vmin=vmin, vmax=vmax,
                    extent=(0, chip_full.shape[1], f_bot, f_top),
                )
                fig_fft.colorbar(im_fft, ax=ax_fft, label="|FFT|")
                star_fft = "*" if bool(strong_mask[k]) else " "
                ax_fft.set_title(
                    f"{star_fft}box {k:03d}  y={y_int} x={int(x_c)} "
                    f"h={h_int} w={int(w)}  "
                    f"|FFT_az(full-res chip)|",
                    fontsize=10,
                )
                ax_fft.set_xlabel("range column (full-res, in box)")
                ax_fft.set_ylabel("Doppler bin  [cycles / az pixel]")

                out_path_fft = per_box_dir / f"box_{k:03d}_fft.png"
                fig_fft.savefig(out_path_fft, dpi=110)
                plt.close(fig_fft)
                n_fft_written += 1
            print(
                f"Saved {n_fft_written} per-box azimuth FFT PNGs → "
                f"{per_box_dir}/box_000_fft.png … "
                f"box_{K-1:03d}_fft.png"
            )

            # --- Per-box polynomial range-walk autofocus ---
            # Mirrors `core.autofocus.apply_global_range_deviation_correction`
            # on each box chip cropped from the FULL-resolution SLC `s`.
            # For each surviving box write `box_NNN_autofocus.png` (1×2:
            # |chip| before / |chip| after) and a row in
            # `autofocus_summary.csv` listing `best_deviation` and
            # `n_sub_contrast_improved` (number of subapertures whose
            # contrast improved at the best deviation, 0..10).
            af_csv_lines = [
                "box_idx,y,x,h,w,h_chip,w_chip,"
                "best_deviation,n_sub_contrast_improved,"
                "contrast_before,contrast_after,gain_db"
            ]
            af_dev_min = -100.0
            af_dev_max = 100.0
            af_accuracy = 0.5
            af_poly_degree = 2
            # Drop any box whose |best_deviation| comes out below this
            # threshold: the polynomial range-walk correction is
            # negligible there, so the box is effectively already in
            # focus and the per-box artefacts (PNG / NPZ / CSV row)
            # are not written for it.
            af_min_abs_deviation = 3.6
            n_af_written = 0
            n_af_filtered = 0
            # Collect [y_centre, x_centre_full, h, w_full] for every box
            # that passes the autofocus gate so the surviving set can be
            # overlaid on `|s|` at the end of the loop.
            af_kept_boxes_full: list[tuple[float, float, float, float]] = []
            # Parallel list of (y0, y1, x0, x1, corrected_chip) for every
            # surviving box so a second full-image figure can paste the
            # PGA-refined chips back into `|s|`.
            af_kept_chip_data: list[
                tuple[int, int, int, int, np.ndarray]
            ] = []
            for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                y_int = int(y_c)
                h_int = int(h)
                y0_full = max(0, y_int - h_int // 2)
                y1_full = min(H_az_full, y0_full + h_int)
                x_full = int(x_c) * Number_of_Range_Looks
                w_full = int(w) * Number_of_Range_Looks
                x0_full = max(0, x_full - w_full // 2)
                x1_full = min(W_rg_full, x0_full + w_full)
                if x1_full <= x0_full or y1_full - y0_full < 2:
                    continue
                chip_full = s[y0_full:y1_full, x0_full:x1_full]
                if chip_full.size == 0 or chip_full.shape[0] < 2:
                    continue

                corrected_k, best_dev_k, n_sub_imp_k = (
                    _af_apply_global_range_deviation_correction(
                        chip_full,
                        dev_min=af_dev_min,
                        dev_max=af_dev_max,
                        accuracy=af_accuracy,
                        poly_degree=af_poly_degree,
                    )
                )

                # Skip boxes where the search found a near-zero
                # deviation: the correction is negligible and the box
                # is effectively already focused.
                if abs(best_dev_k) < af_min_abs_deviation:
                    n_af_filtered += 1
                    continue

                # Time-domain image contrast (std(|I|²)/mean(|I|²)) before
                # and after the polynomial range-walk correction. Same
                # metric the existing classification-refocus loop uses,
                # so the numbers on this PNG can be compared directly
                # with `box_NNN_refocus.png`. gain_dB > 0 ⇒ sharper.
                c_before_af = _normalized_variance(chip_full)
                c_after_af = _normalized_variance(corrected_k)
                if c_before_af > 0.0 and c_after_af > 0.0:
                    gain_db_af = 20.0 * np.log10(c_after_af / c_before_af)
                else:
                    gain_db_af = float("nan")
                gain_db_str = (
                    f"{gain_db_af:+.2f} dB"
                    if np.isfinite(gain_db_af) else "nan dB"
                )

                star_af = "*" if bool(strong_mask[k]) else " "
                fig_af, axes_af = plt.subplots(
                    1, 2, figsize=(11, 5), constrained_layout=True,
                )
                fig_af.suptitle(
                    f"{star_af}box {k:03d}  y={y_int} x={int(x_c)} "
                    f"h={h_int} w={int(w)}\n"
                    f"best_deviation={best_dev_k:+.3f}   "
                    f"n_sub_contrast_improved={n_sub_imp_k}/10   "
                    f"contrast {c_before_af:.3f} → {c_after_af:.3f}   "
                    f"({gain_db_str})",
                    fontsize=11,
                )
                for ax_af, img_af, ttl_af in (
                    (axes_af[0], chip_full,    "|I|  before"),
                    (axes_af[1], corrected_k,  "|I|  after"),
                ):
                    amp_af = np.abs(img_af)
                    if amp_af.size == 0:
                        ax_af.text(
                            0.5, 0.5, "(empty)",
                            ha="center", va="center",
                        )
                        ax_af.set_title(ttl_af)
                        continue
                    vmin_af = float(np.nanpercentile(amp_af, 1.0))
                    vmax_af = float(np.nanpercentile(amp_af, 99.0))
                    if vmax_af <= vmin_af:
                        vmax_af = vmin_af + 1.0
                    ax_af.imshow(
                        amp_af, aspect="auto", cmap="gray",
                        vmin=vmin_af, vmax=vmax_af,
                    )
                    ax_af.set_xlabel("range column (full-res, in box)")
                    ax_af.set_ylabel("azimuth")
                    ax_af.set_title(ttl_af)
                out_path_af = per_box_dir / f"box_{k:03d}_autofocus.png"
                fig_af.savefig(out_path_af, dpi=110)
                plt.close(fig_af)

                # Persist the per-box autofocus result as an NPZ so the
                # complex `chip_before` / `chip_after` arrays (and the
                # scalar diagnostics drawn on the PNG title) can be
                # reloaded later for re-analysis without re-running the
                # coarse-to-fine search.
                out_path_af_npz = (
                    per_box_dir / f"box_{k:03d}_autofocus.npz"
                )
                np.savez_compressed(
                    out_path_af_npz,
                    chip_before=chip_full.astype(np.complex64, copy=False),
                    chip_after=corrected_k.astype(np.complex64, copy=False),
                    best_deviation=np.float64(best_dev_k),
                    n_sub_contrast_improved=np.int32(n_sub_imp_k),
                    contrast_before=np.float64(c_before_af),
                    contrast_after=np.float64(c_after_af),
                    gain_db=np.float64(gain_db_af),
                    box_idx=np.int32(k),
                    y_center=np.int32(y_int),
                    x_center=np.int32(int(x_c)),
                    h=np.int32(h_int),
                    w=np.int32(int(w)),
                    y0_full=np.int32(y0_full),
                    y1_full=np.int32(y1_full),
                    x0_full=np.int32(x0_full),
                    x1_full=np.int32(x1_full),
                    Number_of_Range_Looks=np.int32(Number_of_Range_Looks),
                    af_dev_min=np.float64(af_dev_min),
                    af_dev_max=np.float64(af_dev_max),
                    af_accuracy=np.float64(af_accuracy),
                    af_poly_degree=np.int32(af_poly_degree),
                    strong=np.bool_(bool(strong_mask[k])),
                )

                gain_db_csv = (
                    f"{gain_db_af:.6f}"
                    if np.isfinite(gain_db_af) else "nan"
                )
                af_csv_lines.append(
                    f"{k},{y_int},{int(x_c)},{h_int},{int(w)},"
                    f"{chip_full.shape[0]},{chip_full.shape[1]},"
                    f"{best_dev_k:.6f},{n_sub_imp_k},"
                    f"{c_before_af:.6f},{c_after_af:.6f},{gain_db_csv}"
                )
                af_kept_boxes_full.append(
                    (float(y_int), float(x_full),
                     float(h_int), float(w_full))
                )
                af_kept_chip_data.append(
                    (y0_full, y1_full, x0_full, x1_full,
                     np.ascontiguousarray(corrected_k)),
                )
                n_af_written += 1
            (per_box_dir / "autofocus_summary.csv").write_text(
                "\n".join(af_csv_lines) + "\n"
            )
            print(
                f"Saved {n_af_written} per-box autofocus PNGs + NPZs → "
                f"{per_box_dir}/box_000_autofocus.{{png,npz}} … "
                f"box_{K-1:03d}_autofocus.{{png,npz}}  "
                f"(+ autofocus_summary.csv; "
                f"{n_af_filtered} boxes dropped with "
                f"|best_deviation| < {af_min_abs_deviation:g})"
            )

            # Building + saving the two full-image overview PNGs below
            # ({stem}_af_boxes.png and {stem}_af_corrected.png) is
            # excluded from the timing report — they are large-canvas
            # plot/save operations dominated by matplotlib + PNG
            # encoding cost, not pipeline work we care about benchmarking.
            sw.pause()

            # --- Full image with boundary boxes after the autofocus gate ---
            # Single-panel figure alongside the main overview:
            # `{stem}_af_boxes.png` shows `|s|` with the bounding
            # rectangles of every box that survived the
            # `|best_deviation| ≥ af_min_abs_deviation` autofocus gate,
            # drawn in red and already in full-resolution range coords.
            af_boxes_arr = (
                np.asarray(af_kept_boxes_full, dtype=np.float64)
                if af_kept_boxes_full else np.empty((0, 4), dtype=np.float64)
            )
            path_af_boxes = save_dir / f"{stem}_af_boxes.png"
            fig_af_boxes, ax_af_boxes = plt.subplots(
                figsize=(11, 9), constrained_layout=True,
            )
            _show_slc(
                ax_af_boxes, s,
                f"|s| with {len(af_boxes_arr)}/{K} boxes after "
                f"autofocus gate (|best_deviation| "
                f"≥ {af_min_abs_deviation:g})",
            )
            if len(af_boxes_arr):
                _overlay_boxes(
                    ax_af_boxes, af_boxes_arr,
                    color="red", lw=1.4,
                )
            fig_af_boxes.savefig(path_af_boxes, dpi=150)
            plt.close(fig_af_boxes)
            print(f"Saved → {path_af_boxes}")

            # --- Full image with corrected chips pasted into the boxes ---
            # Companion to `{stem}_af_boxes.png`. We start from a copy of
            # the full-resolution SLC `s`, then for every box that passed
            # the autofocus gate we overwrite its rectangle with the
            # corrected chip returned by
            # `_af_apply_global_range_deviation_correction` (polynomial
            # range-walk + centred-look PGA when |best_deviation| > 3.6).
            # The result is rendered with the same `_show_slc` helper as
            # the af-boxes figure, with the surviving boxes outlined in
            # red so the patched regions are easy to locate.
            s_with_corrected = s.copy()
            for y0c, y1c, x0c, x1c, chip_corr in af_kept_chip_data:
                target = s_with_corrected[y0c:y1c, x0c:x1c]
                if chip_corr.shape != target.shape:
                    # Defensive: should never happen because the chip was
                    # cropped from the same slice, but skip mismatches
                    # rather than crashing.
                    continue
                s_with_corrected[y0c:y1c, x0c:x1c] = chip_corr.astype(
                    s_with_corrected.dtype, copy=False,
                )

            path_af_corrected = save_dir / f"{stem}_af_corrected.png"
            fig_af_corr, ax_af_corr = plt.subplots(
                figsize=(11, 9), constrained_layout=True,
            )
            _show_slc(
                ax_af_corr, s_with_corrected,
                f"|s| with corrected chips pasted into "
                f"{len(af_kept_chip_data)}/{K} boxes "
                f"(|best_deviation| ≥ {af_min_abs_deviation:g})",
            )
            if len(af_boxes_arr):
                _overlay_boxes(
                    ax_af_corr, af_boxes_arr,
                    color="red", lw=1.4,
                )
            fig_af_corr.savefig(path_af_corrected, dpi=150)
            plt.close(fig_af_corr)
            print(f"Saved → {path_af_corrected}")

            # End of the af_boxes / af_corrected save block — resume timing.
            sw.resume()

            # --- Per-box rough refocus (classification signal) ---
            # For each surviving box write a `box_NNN_refocus.png`
            # (|chip| before / |chip| after the centred-QPE correction)
            # alongside the existing `box_NNN_dphase.png` in per_box_dir,
            # plus a `refocus_summary.csv` with per-box contrast & gain.
            if args.refocus:
                csv_lines = [
                    "box_idx,y,x,h,w,y0_fit,n_rows_fit,"
                    "slope_rad_per_row,slope_total_rad,"
                    "contrast_before,contrast_after,gain_db"
                ]
                gains_db = []
                n_written = 0
                H_full, W_full = s_degraded.shape
                for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                    n_k = int(box_n_rows[k])
                    y0 = int(box_y0[k])
                    y1 = y0 + n_k
                    x_int = int(x_c)
                    w_int = int(w)
                    x0 = max(0, x_int - w_int // 2)
                    x1 = min(W_full, x_int - w_int // 2 + w_int)
                    s_k = float(box_slope_rad_per_row[k])
                    if (
                        n_k < 2 or x1 <= x0 or y1 <= y0
                        or y0 < 0 or y1 > H_full
                        or not np.isfinite(s_k)
                    ):
                        continue
                    chip = s_degraded[y0:y1, x0:x1]
                    if chip.size == 0:
                        continue
                    corrected, _phi = _refocus_box_chip(chip, s_k)
                    c_before = _normalized_variance(chip)
                    c_after = _normalized_variance(corrected)
                    if c_before > 0.0 and c_after > 0.0:
                        db = 20.0 * np.log10(c_after / c_before)
                    else:
                        db = np.nan
                    gains_db.append(db)
                    db_str = f"{db:.6f}" if np.isfinite(db) else "nan"
                    csv_lines.append(
                        f"{k},{int(y_c)},{int(x_c)},{int(h)},{w_int},"
                        f"{y0},{n_k},"
                        f"{s_k:.6f},{float(slope_totals[k]):.6f},"
                        f"{c_before:.6f},{c_after:.6f},{db_str}"
                    )
                    _plot_refocus_box_1x2(
                        chip, corrected,
                        box_idx=k,
                        box_yxhw=(int(y_c), int(x_c), int(h), w_int),
                        slope_rad_per_row=s_k,
                        slope_total_rad=float(slope_totals[k]),
                        contrast_before=c_before,
                        contrast_after=c_after,
                        gain_db=db,
                        out_path=per_box_dir / f"box_{k:03d}_refocus.png",
                    )
                    n_written += 1
                (per_box_dir / "refocus_summary.csv").write_text(
                    "\n".join(csv_lines) + "\n"
                )
                gains_arr = np.asarray(gains_db, dtype=np.float64)
                ok = np.isfinite(gains_arr)
                if ok.any():
                    med_db = float(np.nanmedian(gains_arr[ok]))
                    max_db = float(np.nanmax(gains_arr[ok]))
                    print(
                        f"Saved {n_written} per-box refocus PNGs → "
                        f"{per_box_dir}/box_NNN_refocus.png "
                        f"(contrast gain: median {med_db:+.2f} dB, "
                        f"max {max_db:+.2f} dB)"
                    )
                else:
                    print(
                        f"Saved {n_written} per-box refocus PNGs → "
                        f"{per_box_dir}/box_NNN_refocus.png "
                        f"(no valid gains)"
                    )
    sw.mark("save outputs + per-box PNGs (main fig, d_phase, FFT, autofocus, refocus)")

    # Always close every figure we built so the interpreter can free the
    # matplotlib state cleanly. Figures intended for review have already
    # been written to disk by the args.save branch above; we never pop a
    # GUI window (the Agg backend is forced at import time).
    plt.close("all")

    sw.report()


if __name__ == "__main__":
    main()
