"""Sub-aperture-mean/variance target detection (clean, no plots).

Consumes a `*_subapertures.npz` produced by ``scripts/save_subapertures.py``
and returns one axis-aligned bounding box per detected physical target.

Pipeline
--------
  1. CoV² gate          ``mask = (sub_var / (sub_mean² + eps)
                              > cov_th_mult · median(cov²))``
                        (dark / bright branches available via
                        ``bright_mode``)
  2. Isolated-pixel     density in a 3 m × 3 m window ≥ 0.30
     filter
  3. Seed peaks         local-amplitude-max + (N_TAW × N_TRL) window
                        density > 0.5, on the decimated grid
  4. Clustering         ``scripts.clustering.cluster_peaks`` with the
                        mass-quantile tightening OFF
  5. Grow-and-recenter  per cluster, seed = brightest peak; box is
                        grown per edge with a density + rescue
                        gate, constrained to always contain the
                        cluster's peak bounding rectangle so the
                        amp-CoM recentre cannot drift onto a
                        neighbouring brighter target
  6. Post-growth        +5% × h per side × N_steps at a looser
     azimuth extension  density gate (default 0.05)
  7. (optional)         CA-CFAR on the grown boxes; disabled by
     CFAR              default (``cfar_snr_th=0``)

Public entry point
------------------
    from view_subaps_cov import detect_targets
    result = detect_targets(npz_path)

Returns a ``DetectionResult`` bundle with ``boxes``, ``labels``,
``peaks_yx``, ``seed_peaks_yx``, ``mask_filt`` and metadata. All
arrays live on the decimated sub-aperture grid whose pixel spacings
are ``result.az_m_per_px × result.rg_m_per_px`` metres.

A minimal CLI shell at the bottom just calls ``detect_targets`` on the
first positional argument (default: the shipped test scene) and
prints the per-stage statistics. No plots, no files written.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from clustering import cluster_peaks  # noqa: E402


DEFAULT_NPZ = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/"
    "ICEYE_WTW3YQ_20250104T180444Z_4439676_X7_SLED_SLC_subapertures.npz"
)


# --------------------------------------------------------------------- #
# Grow-and-recenter (local port of shear_averaging.grow_and_recenter_boxes)
# --------------------------------------------------------------------- #
# Two changes vs. the shear_averaging original:
#   1. Adds a `must_contain_yxyx` per-seed inclusive rectangle. The
#      grown box is always shifted (or grown, if size preservation
#      isn't enough) to still contain this rectangle. Used to pin the
#      grown box around a cluster's full peak bounding rectangle so
#      the amp-CoM recentre step cannot drift onto a neighbouring
#      brighter target.
#   2. When `must_contain_yxyx` is provided, the post-recentre FULL-
#      BOX density check is bypassed — the strip-based edge tests
#      still gate every growth step, but a large-cluster box that
#      only happens to have low global density is no longer killed
#      on the first iteration.
# Returned box format is `(y_lo, y_hi, x_lo, x_hi)` INCLUSIVE.
def _grow_and_recenter_boxes(
    peaks_yx: np.ndarray,
    mask: np.ndarray,
    amp: np.ndarray,
    initial_hw: tuple[int, int],
    az_step: int = 1,
    rg_step: int = 1,
    density_threshold: float = 0.15,
    az_tail: int = 3,
    rg_tail: int = 1,
    az_rescue_lookahead: int = 10,
    az_rescue_lookback: int = 10,
    az_rescue_threshold: float = 0.35,
    max_h: int | None = None,
    max_w: int | None = None,
    max_iter: int = 2000,
    must_contain_yxyx: np.ndarray | None = None,
) -> np.ndarray:
    """Grow each seed's box per-edge (az-first, then rg) with amp-CoM
    recentring, constrained to always contain ``must_contain_yxyx[k]``.

    Returns
    -------
    out : (K, 4) int64 array of (y_lo, y_hi, x_lo, x_hi) INCLUSIVE.
    """
    H, W = mask.shape
    h0, w0 = initial_hw
    K = len(peaks_yx)
    out = np.zeros((K, 4), dtype=np.int64)
    if K == 0:
        return out

    def _density_ok(yl, yh, xl, xh, threshold=None):
        """True iff ``mask[yl:yh, xl:xh]`` density ≥ threshold. Uses
        EXCLUSIVE upper bounds during growth (converted to inclusive
        only at the very end)."""
        sub = mask[yl:yh, xl:xh]
        if sub.size == 0:
            return False
        th = density_threshold if threshold is None else threshold
        return float(sub.sum()) >= th * sub.size

    def _contain_rect(y_lo, y_hi, x_lo, x_hi, k):
        """Shift-or-grow the box (EXCLUSIVE y_hi/x_hi) so it contains
        ``must_contain_yxyx[k]`` = (py_lo, py_hi, px_lo, px_hi) with
        BOTH bounds INCLUSIVE. Preserves box size when possible; only
        grows when the current size cannot fit the peak rect."""
        if must_contain_yxyx is None:
            return y_lo, y_hi, x_lo, x_hi
        py_lo, py_hi, px_lo, px_hi = (int(v) for v in must_contain_yxyx[k])
        h = y_hi - y_lo
        peak_h = py_hi - py_lo + 1
        if h >= peak_h:
            if y_lo > py_lo:
                shift = y_lo - py_lo
                y_lo -= shift
                y_hi -= shift
            if y_hi <= py_hi:
                shift = py_hi + 1 - y_hi
                y_lo += shift
                y_hi += shift
        else:
            if y_lo > py_lo:
                y_lo = py_lo
            if y_hi <= py_hi:
                y_hi = py_hi + 1
        w = x_hi - x_lo
        peak_w = px_hi - px_lo + 1
        if w >= peak_w:
            if x_lo > px_lo:
                shift = x_lo - px_lo
                x_lo -= shift
                x_hi -= shift
            if x_hi <= px_hi:
                shift = px_hi + 1 - x_hi
                x_lo += shift
                x_hi += shift
        else:
            if x_lo > px_lo:
                x_lo = px_lo
            if x_hi <= px_hi:
                x_hi = px_hi + 1
        y_lo = max(0, y_lo)
        y_hi = min(H, y_hi)
        x_lo = max(0, x_lo)
        x_hi = min(W, x_hi)
        return y_lo, y_hi, x_lo, x_hi

    for k, (y0, x0) in enumerate(peaks_yx):
        y_lo = max(0, int(y0) - h0 // 2)
        y_hi = min(H, y_lo + h0)
        x_lo = max(0, int(x0) - w0 // 2)
        x_hi = min(W, x_lo + w0)
        y_lo, y_hi, x_lo, x_hi = _contain_rect(y_lo, y_hi, x_lo, x_hi, k)

        for _ in range(max_iter):
            grew_az = False

            # Top edge
            ny_lo = y_lo - az_step
            new_h = y_hi - ny_lo
            over_cap = max_h is not None and new_h > max_h
            if not over_cap and ny_lo >= 0:
                tail_hi = min(y_lo + az_tail, y_hi)
                if _density_ok(ny_lo, tail_hi, x_lo, x_hi):
                    y_lo = ny_lo
                    grew_az = True
                else:
                    look_lo = max(y_lo - az_rescue_lookahead, 0)
                    tail_hi_resc = min(y_lo + az_rescue_lookback, y_hi)
                    if _density_ok(look_lo, tail_hi_resc, x_lo, x_hi,
                                   threshold=az_rescue_threshold):
                        y_lo = ny_lo
                        grew_az = True

            # Bottom edge
            ny_hi = y_hi + az_step
            new_h = ny_hi - y_lo
            over_cap = max_h is not None and new_h > max_h
            if not over_cap and ny_hi <= H:
                tail_lo = max(y_hi - az_tail, y_lo)
                if _density_ok(tail_lo, ny_hi, x_lo, x_hi):
                    y_hi = ny_hi
                    grew_az = True
                else:
                    look_hi = min(y_hi + az_rescue_lookahead, H)
                    tail_lo_resc = max(y_hi - az_rescue_lookback, y_lo)
                    if _density_ok(tail_lo_resc, look_hi, x_lo, x_hi,
                                   threshold=az_rescue_threshold):
                        y_hi = ny_hi
                        grew_az = True

            grew_rg = False
            if not grew_az:
                # Left edge
                nx_lo = x_lo - rg_step
                new_w = x_hi - nx_lo
                over_cap = max_w is not None and new_w > max_w
                tail_hi_x = min(x_lo + rg_tail, x_hi)
                if (
                    not over_cap and nx_lo >= 0
                    and _density_ok(y_lo, y_hi, nx_lo, tail_hi_x)
                ):
                    x_lo = nx_lo
                    grew_rg = True

                # Right edge
                nx_hi = x_hi + rg_step
                new_w = nx_hi - x_lo
                over_cap = max_w is not None and new_w > max_w
                tail_lo_x = max(x_hi - rg_tail, x_lo)
                if (
                    not over_cap and nx_hi <= W
                    and _density_ok(y_lo, y_hi, tail_lo_x, nx_hi)
                ):
                    x_hi = nx_hi
                    grew_rg = True

            if not (grew_az or grew_rg):
                break

            # Amp-CoM recentre (preserves current h × w).
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
                y_lo = max(0, y_com - h_cur // 2)
                y_hi = min(H, y_lo + h_cur)
                y_lo = max(0, y_hi - h_cur)
                x_lo = max(0, x_com - w_cur // 2)
                x_hi = min(W, x_lo + w_cur)
                x_lo = max(0, x_hi - w_cur)

            y_lo, y_hi, x_lo, x_hi = _contain_rect(y_lo, y_hi, x_lo, x_hi, k)

            # Full-box density safety net (bypassed when a must-
            # contain rect is present — the strip gate is what
            # actually decides whether growth continues).
            if must_contain_yxyx is None:
                if not _density_ok(y_lo, y_hi, x_lo, x_hi):
                    break

        out[k, 0] = y_lo
        out[k, 1] = y_hi - 1
        out[k, 2] = x_lo
        out[k, 3] = x_hi - 1
    return out


def _extend_boxes_azimuth_strong_signal(
    boxes: np.ndarray,
    mask: np.ndarray,
    *,
    step_frac: float = 0.05,
    max_steps: int = 4,
    density_threshold: float = 0.05,
    max_h: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Post-growth az-only extension. Boxes are ``(y_lo, y_hi, x_lo, x_hi)``
    INCLUSIVE. Tries to push top/bottom outward by ``step_frac × h_entry``
    rows, up to ``max_steps`` times per side, accepting each step iff
    the strip's mask density ≥ ``density_threshold``.

    Returns
    -------
    out : (K, 4) int64 array — same format as the input.
    """
    if len(boxes) == 0:
        return boxes
    H, _ = mask.shape
    out = np.array(boxes, dtype=np.int64, copy=True)
    n_top = n_bot = total_extra = 0
    for k in range(len(out)):
        y_lo_i, y_hi_i, x_lo_i, x_hi_i = (int(v) for v in out[k])
        y_lo = y_lo_i
        y_hi = y_hi_i + 1
        h_entry = y_hi - y_lo
        step = max(1, int(round(step_frac * h_entry)))

        for _ in range(max_steps):
            ny_lo = y_lo - step
            if ny_lo < 0:
                break
            if max_h is not None and (y_hi - ny_lo) > max_h:
                break
            strip = mask[ny_lo:y_lo, x_lo_i:x_hi_i + 1]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_lo = ny_lo
                n_top += 1
            else:
                break

        for _ in range(max_steps):
            ny_hi = y_hi + step
            if ny_hi > H:
                break
            if max_h is not None and (ny_hi - y_lo) > max_h:
                break
            strip = mask[y_hi:ny_hi, x_lo_i:x_hi_i + 1]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_hi = ny_hi
                n_bot += 1
            else:
                break

        total_extra += (y_hi - y_lo) - h_entry
        out[k, 0] = y_lo
        out[k, 1] = y_hi - 1
    if verbose:
        print(
            f"  az-extend: step={step_frac:.0%}×h, max_steps={max_steps}, "
            f"density≥{density_threshold:g}: {n_top}/{len(out)} top-steps, "
            f"{n_bot}/{len(out)} bottom-steps, "
            f"+{total_extra} az rows total"
        )
    return out


# --------------------------------------------------------------------- #
# Detection result bundle
# --------------------------------------------------------------------- #
@dataclass
class DetectionResult:
    """Return type of :func:`detect_targets`. All arrays live on the
    decimated sub-aperture grid whose pixel spacings are
    ``az_m_per_px × rg_m_per_px`` metres.
    """
    boxes: np.ndarray            # (N, 4) inclusive (y_lo, y_hi, x_lo, x_hi)
    labels: np.ndarray           # (K,) peak → cluster id in [0, N)
    peaks_yx: np.ndarray         # (K, 2) all seed peaks
    seed_peaks_yx: np.ndarray    # (N, 2) brightest peak per cluster
    mask_filt: np.ndarray        # (H, W) isolated-pixel-filtered CoV mask
    sub_mean: np.ndarray         # (H, W)
    sub_var: np.ndarray          # (H, W)
    az_m_per_px: float
    rg_m_per_px: float
    n_subaperture: int | None
    n_range_looks: int | None
    threshold: float             # applied CoV² threshold (dark path for
    #                              exclude / higher-th modes)


# --------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------- #
def detect_targets(
    npz_path: Path,
    *,
    # ---- CoV gate ----
    cov_th_mult: float = 1.5,
    bright_mode: str = "off",
    bright_cov_th_mult: float = 3.0,
    dark_amp_percentile: float = 50.0,
    eps: float = 1e-12,
    # ---- Isolated-pixel filter ----
    filter_target_size_m: float = 3.0,
    filter_min_density: float = 0.30,
    # ---- Seed detection ----
    seed_density_th: float = 0.5,
    n_target_azimuth_width: int = 100,
    n_target_range_length: int = 3,
    # ---- Clustering ----
    az_search_m: float = 500.0,
    rg_search_m: float = 30.0,
    enable_arc_split: bool = False,
    max_peak_gap_az_m: float = float("inf"),
    # ---- Same-corridor cluster merge (opt-in) ----
    # Complete-linkage's overlap gate refuses to merge two clusters
    # whenever ANY single pair of their per-peak azimuth intervals
    # fails to touch. Two arc-shaped smears from the SAME long rigid
    # body (e.g. two dominant scatterers on a big ship) can therefore
    # end up as two separate clusters even when their max pairwise
    # Chebyshev distance (in units of az/rg_search_m) is ≤ 1.0 — i.e.
    # even when linkage WOULD have merged them without the overlap
    # gate. Enable this to run a post-linkage merge with exactly the
    # same threshold as fcluster's cut but ignoring the gate, so the
    # two arcs become one cluster before the grow step and the
    # resulting grown box covers the full rigid body. Transitively
    # safe: distant clusters cannot chain because their max pairwise
    # distance vetoes the merge, same as complete-linkage.
    merge_same_corridor: bool = False,
    merge_max_norm_dist: float = 1.0,
    # ---- Grow-and-recenter (relaxed-az defaults) ----
    grow_density_th: float = 0.15,
    grow_az_step_m: float = 10.0,
    grow_az_tail_m: float | None = None,   # None → same as step
    grow_rescue_lookahead_m: float = 200.0,
    grow_rescue_lookback_m: float = 200.0,
    grow_rescue_th: float = 0.35,
    grow_max_h_m: float = 300.0,
    grow_max_w_m: float = 130.0,
    # ---- Post-growth az extension (relaxed-az defaults) ----
    extend_density_th: float = 0.05,
    extend_max_steps: int = 4,
    # ---- Optional CA-CFAR ----
    cfar_snr_th: float = 0.0,
    cfar_range_bins: int = 2,
    verbose: bool = True,
) -> DetectionResult:
    """Detect targets on a `*_subapertures.npz` and return grown boxes.

    See the module docstring for the pipeline overview. Every knob
    defaults to the tuned relaxed-azimuth values.
    """
    _log = print if verbose else (lambda *_a, **_k: None)

    # ---- Load ------------------------------------------------------
    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=False) as z:
        missing = [k for k in ("sub_mean", "sub_var") if k not in z.files]
        if missing:
            raise KeyError(
                f"{npz_path.name} is missing required keys {missing}. "
                "Regenerate with scripts/save_subapertures.py."
            )
        sub_mean = z["sub_mean"].astype(np.float64)
        sub_var = z["sub_var"].astype(np.float64)
        n_sub = int(z["N_subaperture"]) if "N_subaperture" in z.files else None
        n_rl = int(z["Number_of_Range_Looks"]) if "Number_of_Range_Looks" in z.files else None
        az_m = float(z["az_m_per_px_s_degraded"]) if "az_m_per_px_s_degraded" in z.files else None
        rg_m = float(z["rg_m_per_px_s_degraded"]) if "rg_m_per_px_s_degraded" in z.files else None

    az_m_dec = (az_m * n_sub) if (az_m is not None and n_sub is not None) else 0.64
    rg_m_dec = rg_m if rg_m is not None else 0.96

    _log(
        f"Loaded {npz_path.name}: shape={sub_mean.shape}, "
        f"az={az_m_dec:g} m/row × rg={rg_m_dec:g} m/px "
        f"(N_sub={n_sub}, N_RL={n_rl})"
    )

    # ---- CoV gate --------------------------------------------------
    cov_sq = sub_var / (sub_mean ** 2 + eps)
    th_applied: float
    if bright_mode == "off":
        th_applied = cov_th_mult * float(np.median(cov_sq))
        mask_dec = (cov_sq > th_applied).astype(np.float32)
    else:
        amp_dark_thresh = float(np.percentile(sub_mean, dark_amp_percentile))
        dark_dec = sub_mean <= amp_dark_thresh
        cov_sq_dark = cov_sq[dark_dec] if dark_dec.any() else cov_sq
        median_dark = float(np.median(cov_sq_dark))
        th_dark = cov_th_mult * median_dark
        th_applied = th_dark
        if bright_mode == "exclude":
            mask_dec = (dark_dec & (cov_sq > th_dark)).astype(np.float32)
        elif bright_mode == "higher-th":
            bright_dec = ~dark_dec
            median_bright = (
                float(np.median(cov_sq[bright_dec])) if bright_dec.any()
                else median_dark
            )
            th_bright = bright_cov_th_mult * median_bright
            mask_dec = (
                (dark_dec & (cov_sq > th_dark))
                | (bright_dec & (cov_sq > th_bright))
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown bright_mode: {bright_mode!r}")

    n_kept = int(mask_dec.sum())
    n_total = mask_dec.size
    _log(
        f"  CoV gate (mode='{bright_mode}', mult={cov_th_mult:g}): "
        f"kept {n_kept}/{n_total} = {100 * n_kept / n_total:.2f}%"
    )

    # ---- Isolated-pixel filter -------------------------------------
    az_filt_win = max(1, int(round(filter_target_size_m / az_m_dec)))
    rg_filt_win = max(1, int(round(filter_target_size_m / rg_m_dec)))
    density = uniform_filter(
        mask_dec, size=(az_filt_win, rg_filt_win), mode="constant",
    )
    mask_filt = (
        mask_dec.astype(bool) & (density >= filter_min_density)
    ).astype(np.float32)
    n_kept_after = int(mask_filt.sum())
    _log(
        f"  isolated-pixel filter ({filter_target_size_m:g} m × "
        f"{filter_target_size_m:g} m, density ≥ {filter_min_density:.2f}): "
        f"kept {n_kept_after}/{n_kept} = "
        f"{100 * n_kept_after / max(n_kept, 1):.2f}%"
    )

    # ---- Seed peaks -----------------------------------------------
    az_seed_win = max(
        1, int(round(n_target_azimuth_width / max(n_sub or 16, 1))),
    )
    rg_seed_win = int(n_target_range_length)
    box_size = (az_seed_win, rg_seed_win)
    box_area = az_seed_win * rg_seed_win

    amp_proxy = sub_mean.astype(np.float32)
    det_count = uniform_filter(
        mask_filt.astype(np.float32), size=box_size, mode="constant",
    ) * box_area
    amp_max = maximum_filter(amp_proxy, size=box_size, mode="constant")
    boundary = (
        (mask_filt == 1)
        & (det_count > seed_density_th * box_area)
        & (amp_proxy == amp_max)
    )
    peaks_yx = np.argwhere(boundary).astype(np.int64)
    _log(
        f"  seed peaks: window=({az_seed_win},{rg_seed_win}), "
        f"density > {seed_density_th:g} → {len(peaks_yx)} peaks"
    )

    # ---- Clustering (mass-quantile tightening OFF) ----------------
    labels = np.empty(0, dtype=np.int64)
    boxes = np.empty((0, 4), dtype=np.int64)
    if len(peaks_yx):
        arc_az_thr_m = 200.0 if enable_arc_split else float("inf")
        t0 = time.time()
        labels, boxes = cluster_peaks(
            mask_filt.astype(np.uint8), peaks_yx,
            az_m_per_px=az_m_dec, rg_m_per_px=rg_m_dec,
            az_search_m=az_search_m, rg_search_m=rg_search_m,
            arc_az_thr_m=arc_az_thr_m, arc_rg_max_m=20.0,
            max_peak_gap_az_m=max_peak_gap_az_m,
            tighten_boxes=False,
            merge_same_corridor=merge_same_corridor,
            merge_max_norm_dist=merge_max_norm_dist,
            enforce_max_size=False,
        )
        n_clusters_after = int(labels.max()) + 1
        _log(
            f"  cluster_peaks: {len(peaks_yx)} peaks → "
            f"{n_clusters_after} clusters in {time.time() - t0:.2f} s"
            + (
                f"  (merge_same_corridor: max_norm_dist ≤ "
                f"{merge_max_norm_dist:g})"
                if merge_same_corridor else ""
            )
        )

    # ---- Cluster → grow -----------------------------------------
    seed_peaks_yx = np.empty((0, 2), dtype=np.int64)
    if len(peaks_yx) and len(boxes):
        n_clusters_pre = int(labels.max()) + 1
        az_tail_m_eff = grow_az_step_m if grow_az_tail_m is None else grow_az_tail_m
        az_step_px = max(1, int(round(grow_az_step_m / az_m_dec)))
        az_tail_px = max(1, int(round(az_tail_m_eff / az_m_dec)))
        look_ahead_px = max(1, int(round(grow_rescue_lookahead_m / az_m_dec)))
        look_back_px = max(1, int(round(grow_rescue_lookback_m / az_m_dec)))
        max_h_px = max(1, int(round(grow_max_h_m / az_m_dec)))
        max_w_px = max(1, int(round(grow_max_w_m / rg_m_dec)))

        seed_list: list[np.ndarray] = []
        must_contain_list: list[tuple[int, int, int, int]] = []
        for c in range(n_clusters_pre):
            mem = np.where(labels == c)[0]
            if len(mem) == 0:
                continue
            mp = peaks_yx[mem]
            amps_at_peaks = amp_proxy[
                mp[:, 0].astype(np.int64), mp[:, 1].astype(np.int64),
            ]
            best_local = int(np.argmax(amps_at_peaks))
            seed_list.append(mp[best_local])
            must_contain_list.append((
                int(mp[:, 0].min()), int(mp[:, 0].max()),
                int(mp[:, 1].min()), int(mp[:, 1].max()),
            ))
        seed_peaks_yx = np.asarray(seed_list, dtype=np.int64)
        must_contain_yxyx = np.asarray(must_contain_list, dtype=np.int64)

        t_g = time.time()
        boxes = _grow_and_recenter_boxes(
            seed_peaks_yx,
            mask=mask_filt.astype(np.float32),
            amp=amp_proxy,
            initial_hw=(az_seed_win, rg_seed_win),
            az_step=az_step_px,
            rg_step=1,
            density_threshold=grow_density_th,
            az_tail=az_tail_px,
            rg_tail=1,
            az_rescue_lookahead=look_ahead_px,
            az_rescue_lookback=look_back_px,
            az_rescue_threshold=grow_rescue_th,
            max_h=max_h_px,
            max_w=max_w_px,
            must_contain_yxyx=must_contain_yxyx,
        )
        boxes = _extend_boxes_azimuth_strong_signal(
            boxes,
            mask=mask_filt.astype(np.float32),
            step_frac=0.05,
            max_steps=extend_max_steps,
            density_threshold=extend_density_th,
            max_h=max_h_px,
            verbose=verbose,
        )
        gh = boxes[:, 1] - boxes[:, 0] + 1
        gw = boxes[:, 3] - boxes[:, 2] + 1
        _log(
            f"  grow-and-recenter: {n_clusters_pre} clusters → "
            f"{len(boxes)} grown boxes in {time.time() - t_g:.2f} s "
            f"(step={az_step_px}px [{grow_az_step_m:g} m], "
            f"tail={az_tail_px}px [{az_tail_m_eff:g} m"
            + (" ← =step" if grow_az_tail_m is None else "")
            + f"], density≥{grow_density_th:g}, "
            f"rescue lookahead={look_ahead_px}px "
            f"[{grow_rescue_lookahead_m:g} m], "
            f"cap h×w={max_h_px}×{max_w_px}px "
            f"[{grow_max_h_m:g}×{grow_max_w_m:g} m])"
        )
        _log(
            f"    grown az: min={gh.min()}, med={int(np.median(gh))}, "
            f"max={gh.max()} px  "
            f"({gh.min() * az_m_dec:.0f} .. {gh.max() * az_m_dec:.0f} m)"
        )
        _log(
            f"    grown rg: min={gw.min()}, med={int(np.median(gw))}, "
            f"max={gw.max()} px  "
            f"({gw.min() * rg_m_dec:.0f} .. {gw.max() * rg_m_dec:.0f} m)"
        )

    # ---- Optional CA-CFAR filter ---------------------------------
    if len(boxes) and cfar_snr_th > 0:
        H_mask, W_mask = mask_filt.shape
        n_guard = int(max(1, cfar_range_bins))
        keep = np.ones(len(boxes), dtype=bool)
        for c in range(len(boxes)):
            y_lo, y_hi, x_lo, x_hi = (int(v) for v in boxes[c])
            box_amp = amp_proxy[y_lo:y_hi + 1, x_lo:x_hi + 1]
            box_msk = mask_filt[y_lo:y_hi + 1, x_lo:x_hi + 1]
            sig_pix = box_amp[box_msk == 1]
            if sig_pix.size == 0:
                sig_pix = box_amp
                if sig_pix.size == 0:
                    keep[c] = False
                    continue
            signal = float(sig_pix.mean())

            prev_lo = max(0, x_lo - n_guard)
            succ_hi = min(W_mask, x_hi + 1 + n_guard)
            noise_means: list[float] = []
            prev_strip = amp_proxy[y_lo:y_hi + 1, prev_lo:x_lo]
            succ_strip = amp_proxy[y_lo:y_hi + 1, x_hi + 1:succ_hi]
            if prev_strip.size:
                noise_means.append(float(prev_strip.mean()))
            if succ_strip.size:
                noise_means.append(float(succ_strip.mean()))
            if not noise_means:
                continue
            snr = signal / max(min(noise_means), 1e-12)
            if snr < cfar_snr_th:
                keep[c] = False

        n_before = len(boxes)
        boxes = boxes[keep]
        keep_pk = keep[labels]
        labels = labels[keep_pk]
        peaks_yx = peaks_yx[keep_pk]
        if len(labels):
            _, remap = np.unique(labels, return_inverse=True)
            labels = remap.astype(np.int64)
        seed_peaks_yx = seed_peaks_yx[keep]
        _log(
            f"  CA-CFAR (snr_th={cfar_snr_th:g}): "
            f"kept {len(boxes)}/{n_before} clusters"
        )

    return DetectionResult(
        boxes=boxes.astype(np.int64),
        labels=labels.astype(np.int64),
        peaks_yx=peaks_yx.astype(np.int64),
        seed_peaks_yx=seed_peaks_yx.astype(np.int64),
        mask_filt=mask_filt,
        sub_mean=sub_mean,
        sub_var=sub_var,
        az_m_per_px=az_m_dec,
        rg_m_per_px=rg_m_dec,
        n_subaperture=n_sub,
        n_range_looks=n_rl,
        threshold=th_applied,
    )


# --------------------------------------------------------------------- #
# Thin CLI shell (no plots, no file writes)
# --------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sub-aperture-mean/variance target detection "
                    "(clean, no plots).",
    )
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--cov-th-mult", type=float, default=1.5)
    parser.add_argument(
        "--bright-mode", choices=("off", "exclude", "higher-th"),
        default="off",
    )
    parser.add_argument("--bright-cov-th-mult", type=float, default=3.0)
    parser.add_argument("--dark-amp-percentile", type=float, default=50.0)
    parser.add_argument("--filter-target-size-m", type=float, default=3.0)
    parser.add_argument("--filter-min-density", type=float, default=0.30)
    parser.add_argument("--seed-density-th", type=float, default=0.5)
    parser.add_argument("--n-target-azimuth-width", type=int, default=100)
    parser.add_argument("--n-target-range-length", type=int, default=3)
    parser.add_argument("--az-search-m", type=float, default=500.0)
    parser.add_argument("--rg-search-m", type=float, default=30.0)
    parser.add_argument("--enable-arc-split", action="store_true")
    parser.add_argument("--max-peak-gap-az-m", type=float, default=float("inf"))
    parser.add_argument(
        "--merge-same-corridor", action="store_true",
        help="Merge cluster pairs whose MAX pairwise Chebyshev "
             "distance (normalised by --az-search-m / --rg-search-m) "
             "is ≤ --merge-max-norm-dist. Uses exactly the same "
             "threshold as fcluster's cut but bypasses the overlap "
             "gate, recovering cases where the gate spuriously kept "
             "two arc-shaped smears of the same rigid body apart. "
             "Transitively safe (no chaining). Default: off.",
    )
    parser.add_argument("--merge-max-norm-dist", type=float, default=1.0)
    parser.add_argument("--grow-density-th", type=float, default=0.15)
    parser.add_argument("--grow-az-step-m", type=float, default=10.0)
    parser.add_argument("--grow-az-tail-m", type=float, default=None)
    parser.add_argument("--grow-rescue-lookahead-m", type=float, default=200.0)
    parser.add_argument("--grow-rescue-lookback-m", type=float, default=200.0)
    parser.add_argument("--grow-rescue-th", type=float, default=0.35)
    parser.add_argument("--grow-max-h-m", type=float, default=300.0)
    parser.add_argument("--grow-max-w-m", type=float, default=130.0)
    parser.add_argument("--extend-density-th", type=float, default=0.05)
    parser.add_argument("--extend-max-steps", type=int, default=4)
    parser.add_argument("--cfar-snr-th", type=float, default=0.0)
    parser.add_argument("--cfar-range-bins", type=int, default=2)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    detect_targets(
        args.path,
        cov_th_mult=args.cov_th_mult,
        bright_mode=args.bright_mode,
        bright_cov_th_mult=args.bright_cov_th_mult,
        dark_amp_percentile=args.dark_amp_percentile,
        filter_target_size_m=args.filter_target_size_m,
        filter_min_density=args.filter_min_density,
        seed_density_th=args.seed_density_th,
        n_target_azimuth_width=args.n_target_azimuth_width,
        n_target_range_length=args.n_target_range_length,
        az_search_m=args.az_search_m,
        rg_search_m=args.rg_search_m,
        enable_arc_split=args.enable_arc_split,
        max_peak_gap_az_m=args.max_peak_gap_az_m,
        merge_same_corridor=args.merge_same_corridor,
        merge_max_norm_dist=args.merge_max_norm_dist,
        grow_density_th=args.grow_density_th,
        grow_az_step_m=args.grow_az_step_m,
        grow_az_tail_m=args.grow_az_tail_m,
        grow_rescue_lookahead_m=args.grow_rescue_lookahead_m,
        grow_rescue_lookback_m=args.grow_rescue_lookback_m,
        grow_rescue_th=args.grow_rescue_th,
        grow_max_h_m=args.grow_max_h_m,
        grow_max_w_m=args.grow_max_w_m,
        extend_density_th=args.extend_density_th,
        extend_max_steps=args.extend_max_steps,
        cfar_snr_th=args.cfar_snr_th,
        cfar_range_bins=args.cfar_range_bins,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
