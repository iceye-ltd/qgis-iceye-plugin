"""Display sub-aperture mean, CoV², CoV-mask and clusters side by side.

Consumes a `*_subapertures.npz` produced by ``scripts/save_subapertures.py``
(which mirrors the head of ``shear_averaging.main`` up to the sub-aperture
statistics — see lines 3877-3878 of ``scripts/shear_averaging.py``) and
computes the coefficient-of-variation squared exactly as line 3886 does::

    cov_sq = sub_var / (sub_mean ** 2 + 1e-12)

Then applies the same three-mode CoV threshold as
``scripts/shear_averaging.py:3908-3933``:

  * ``--bright-mode off``       : scene-wide  ``mult · median(cov²)``.
  * ``--bright-mode exclude``   : DARK pixels only, thresholded by
    ``mult · median_dark(cov²)``; bright pixels are zeroed.
  * ``--bright-mode higher-th`` : DARK pixels use ``mult · median_dark``,
    BRIGHT pixels use ``bright_mult · median_bright``.

Detects seed peaks with the same density × local-amplitude-max recipe as
``shear_averaging.py:4084-4092`` (rescaled from the full-res azimuth grid
to the decimated one by ``N_subaperture``), then runs
``clustering.cluster_peaks`` at the decimated-grid pixel spacing to group
peaks into per-target boxes.

Panels
------
    (a) log10(sub_mean)         Doppler-mean amplitude
    (b) log10(cov_sq)           raw CoV², threshold contour overlaid
    (c) CoV mask                binary CoV gate (after isolated-pixel filter)
    (d) mask + peaks + clusters cluster_peaks() result, one colour per cluster

All four panels share the y-scale so a given row lines up. The aspect
ratio is derived from ``az_m_per_px_s_degraded * N_subaperture`` (each
row of the decimated grid covers ``N_subaperture`` rows of ``s_degraded``)
and ``rg_m_per_px_s_degraded`` so the panels are shown at their physical
proportions.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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


def _clip_range(x: np.ndarray, lo_p: float = 2.0, hi_p: float = 99.5) -> tuple[float, float]:
    """Robust percentile-based (vmin, vmax) for `imshow`.

    2nd / 99.5th percentiles kill the log10-tail on the noise floor and
    a few extreme scatterers without hand-tuning per-scene limits.
    """
    return float(np.percentile(x, lo_p)), float(np.percentile(x, hi_p))


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
# The returned box format is `(y_lo, y_hi, x_lo, x_hi)` INCLUSIVE, to
# match the rest of this script (unlike shear_averaging's original
# which returned `(y_c, x_c, h, w)`).
def _grow_and_recenter_boxes(
    peaks_yx: np.ndarray,
    mask: np.ndarray,
    amp: np.ndarray,
    initial_hw: tuple[int, int],
    az_step: int = 1,
    rg_step: int = 1,
    density_threshold: float = 0.25,
    az_tail: int = 3,
    rg_tail: int = 1,
    az_rescue_lookahead: int = 10,
    az_rescue_lookback: int = 10,
    az_rescue_threshold: float = 0.5,
    max_h: int | None = None,
    max_w: int | None = None,
    max_iter: int = 2000,
    must_contain_yxyx: np.ndarray | None = None,
) -> np.ndarray:
    """Grow each seed's box per-edge (az-first, then rg) with amp-CoM
    recentring, constrained to always contain `must_contain_yxyx[k]`.

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

    # ------ density gate ------------------------------------------------
    def _density_ok(yl, yh, xl, xh, threshold=None):
        """True iff `mask[yl:yh, xl:xh]` density ≥ threshold. Uses
        EXCLUSIVE upper bounds during growth (converted to inclusive
        only at the very end)."""
        sub = mask[yl:yh, xl:xh]
        if sub.size == 0:
            return False
        th = density_threshold if threshold is None else threshold
        return float(sub.sum()) >= th * sub.size

    # ------ must-contain shift-or-grow helper --------------------------
    def _contain_rect(y_lo, y_hi, x_lo, x_hi, k):
        """Shift-or-grow the box (EXCLUSIVE y_hi/x_hi) so it contains
        `must_contain_yxyx[k]` = (py_lo, py_hi, px_lo, px_hi) with
        BOTH bounds INCLUSIVE. Preserves box size when possible; only
        grows when the current size cannot fit the peak rect."""
        if must_contain_yxyx is None:
            return y_lo, y_hi, x_lo, x_hi
        py_lo, py_hi, px_lo, px_hi = (int(v) for v in must_contain_yxyx[k])
        # --- azimuth ---
        h = y_hi - y_lo
        peak_h = py_hi - py_lo + 1
        if h >= peak_h:
            # Enough room; shift to keep the rect inside.
            if y_lo > py_lo:
                shift = y_lo - py_lo
                y_lo -= shift
                y_hi -= shift
            if y_hi <= py_hi:
                shift = py_hi + 1 - y_hi
                y_lo += shift
                y_hi += shift
        else:
            # Rect wider than the box; enlarge.
            if y_lo > py_lo:
                y_lo = py_lo
            if y_hi <= py_hi:
                y_hi = py_hi + 1
        # --- range ---
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
        # Clip. Peaks are always inside [0,H) × [0,W) so containment
        # survives clipping automatically.
        y_lo = max(0, y_lo)
        y_hi = min(H, y_hi)
        x_lo = max(0, x_lo)
        x_hi = min(W, x_hi)
        return y_lo, y_hi, x_lo, x_hi

    for k, (y0, x0) in enumerate(peaks_yx):
        # --- Initial box centred on the (brightest) seed, then padded
        # up to contain the cluster's peak bounding rect. Once the
        # containment step has fired the initial box may be much
        # bigger than `initial_hw`, but the strip-based density test
        # below is what actually gates growth.
        y_lo = max(0, int(y0) - h0 // 2)
        y_hi = min(H, y_lo + h0)
        x_lo = max(0, int(x0) - w0 // 2)
        x_hi = min(W, x_lo + w0)
        y_lo, y_hi, x_lo, x_hi = _contain_rect(y_lo, y_hi, x_lo, x_hi, k)

        for _ in range(max_iter):
            # ---- Phase A: azimuth (both edges) ----------------------
            grew_az = False

            # (a) Top edge
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

            # (b) Bottom edge
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

            # ---- Phase B: range (only if az stuck) ------------------
            grew_rg = False
            if not grew_az:
                # (c) Left edge
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

                # (d) Right edge
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

            # (e) Neither axis advanced → done
            if not (grew_az or grew_rg):
                break

            # (f) Amp-CoM recentre, preserving current h × w. Slides
            # the box to align with the local amplitude concentration;
            # `_contain_rect` right after clips the slide so the box
            # still covers every cluster peak.
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

            # Snap back so the peak rect stays inside after the
            # recentre slide.
            y_lo, y_hi, x_lo, x_hi = _contain_rect(y_lo, y_hi, x_lo, x_hi, k)

            # (g) Full-box density safety net. Bypassed when a must-
            # contain rect is present: we've already committed to a
            # box shape that covers every cluster peak; the strip
            # gate is what actually decides whether growth continues.
            if must_contain_yxyx is None:
                if not _density_ok(y_lo, y_hi, x_lo, x_hi):
                    break

        # Convert exclusive → inclusive on the way out.
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
    max_steps: int = 2,
    density_threshold: float = 0.10,
    max_h: int | None = None,
) -> np.ndarray:
    """Post-growth az-only extension. Boxes are `(y_lo, y_hi, x_lo, x_hi)`
    with both bounds INCLUSIVE (view_subaps_cov convention).

    Tries to push each box's top and bottom edges outward by
    `step_frac × h_entry` rows (with a minimum of 1 row), up to
    `max_steps` times per side. Each step is accepted iff the new
    strip's mask density ≥ `density_threshold`. Deliberately looser
    than the growth-phase density gate: growth has already found the
    dense core; here we only need a faint hint the target continues.

    Returns
    -------
    out : (K, 4) int64 array — same format as the input.
    """
    if len(boxes) == 0:
        return boxes
    H, _ = mask.shape
    out = np.array(boxes, dtype=np.int64, copy=True)
    n_extended_top = 0
    n_extended_bot = 0
    total_extra_h = 0
    for k in range(len(out)):
        y_lo_i, y_hi_i, x_lo_i, x_hi_i = (int(v) for v in out[k])
        # Work with an exclusive y_hi for slicing arithmetic.
        y_lo = y_lo_i
        y_hi = y_hi_i + 1
        h_entry = y_hi - y_lo
        step = max(1, int(round(step_frac * h_entry)))

        # Top edge (smaller y).
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
            strip = mask[y_hi:ny_hi, x_lo_i:x_hi_i + 1]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_hi = ny_hi
                n_extended_bot += 1
            else:
                break

        h_new = y_hi - y_lo
        total_extra_h += (h_new - h_entry)
        out[k, 0] = y_lo
        out[k, 1] = y_hi - 1
        # x untouched
    print(
        f"  az-extend: step={step_frac:.0%}×h, max_steps={max_steps}, "
        f"density≥{density_threshold:g}: "
        f"{n_extended_top}/{len(out)} top-steps, "
        f"{n_extended_bot}/{len(out)} bottom-steps, "
        f"total +{total_extra_h} az rows across all boxes"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_NPZ,
        help=f"Path to a `*_subapertures.npz`. Defaults to {DEFAULT_NPZ}.",
    )
    parser.add_argument(
        "--eps", type=float, default=1e-12,
        help="Numerical floor added to mean² before division (same as "
             "line 3886 of shear_averaging.py). Default: 1e-12.",
    )
    parser.add_argument(
        "--bright-mode",
        choices=("off", "exclude", "higher-th"),
        default="off",
        help="How to threshold the CoV mask (mirrors "
             "shear_averaging.py:3908-3933). 'off' (default): single "
             "scene-wide `--cov-th-mult · median(cov²)`. 'exclude': "
             "compute the median over DARK pixels only, apply the "
             "threshold to them, and zero the mask on bright pixels. "
             "'higher-th': same for dark, plus bright pixels get their "
             "own threshold `--bright-cov-th-mult · median_bright(cov²)`.",
    )
    parser.add_argument(
        "--cov-th-mult", type=float, default=1.5,
        help="Multiplier on `median(cov²)` for the CoV mask threshold "
             "(dark median when --bright-mode is not 'off'). "
             "Default: 1.5 (same as shear_averaging).",
    )
    parser.add_argument(
        "--dark-amp-percentile", type=float, default=50.0,
        help="Amplitude percentile that separates DARK from BRIGHT for "
             "the dark-aware gate. A pixel is 'dark' iff its `sub_mean` "
             "is ≤ np.percentile(sub_mean, p). Ignored when --bright-mode "
             "is 'off'. Default: 50.0.",
    )
    parser.add_argument(
        "--bright-cov-th-mult", type=float, default=3.0,
        help="Multiplier on `median_bright(cov²)` used only when "
             "--bright-mode is 'higher-th'. Default: 3.0.",
    )
    parser.add_argument(
        "--seed-density-th", type=float, default=0.5,
        help="Density (fraction of ones inside the seed window) that a "
             "pixel must sit at the centre of to be declared a boundary "
             "peak. Mirrors --seed-density-th in shear_averaging.py. "
             "Default: 0.5.",
    )
    parser.add_argument(
        "--n-target-azimuth-width", type=int, default=100,
        help="Seed-window azimuth length in FULL-RESOLUTION rows "
             "(shear_averaging default). Rescaled by N_subaperture for "
             "the decimated mask_dec grid. Default: 100.",
    )
    parser.add_argument(
        "--n-target-range-length", type=int, default=3,
        help="Seed-window range length in range pixels of the "
             "s_degraded grid (shear_averaging default). Default: 3.",
    )
    parser.add_argument(
        "--filter-target-size-m", type=float, default=3.0,
        help="Side length (m) of the isolated-pixel filter window on the "
             "mask (same 3 m × 3 m as shear_averaging & run_clustering). "
             "Default: 3.0.",
    )
    parser.add_argument(
        "--filter-min-density", type=float, default=0.30,
        help="Minimum density in the isolated-pixel filter window a "
             "mask==1 pixel must have to survive. Same convention as "
             "run_clustering.py. Default: 0.30.",
    )
    parser.add_argument(
        "--enable-arc-split",
        action="store_true",
        help="Enable clustering._split_arc_clusters — the post-linkage "
             "'wide-in-both-axes' azimuth-mover smear splitter. Off by "
             "default here because after sub-aperture averaging there "
             "are no more arc-shaped targets, and the gate then only "
             "spuriously breaks up wide-but-legitimate clusters (harbor "
             "infrastructure, long ships) along the range axis. Pass "
             "this flag to fall back to clustering.py's default arc "
             "thresholds (arc_az_thr_m=200, arc_rg_max_m=20).",
    )
    parser.add_argument(
        "--max-peak-gap-az-m", type=float, default=float("inf"),
        help="Post-linkage azimuth-gap split threshold "
             "(clustering._split_az_gap_clusters). A cluster whose "
             "sorted peak-y positions contain a gap larger than this "
             "many metres is broken at each such gap. Default here is "
             "+inf (i.e. gate DISABLED) because it was the main source "
             "of over-splitting on this scene — see also the "
             "--az-search-m knob for controlling the linkage radius. "
             "Set to a finite value (e.g. 300) to re-enable.",
    )
    parser.add_argument(
        "--az-search-m", type=float, default=500.0,
        help="Azimuth linkage radius for clustering.cluster_peaks. Two "
             "peaks can be merged if their centre-to-centre azimuth "
             "distance is ≤ this many metres (Chebyshev in normalised "
             "space, combined with the range gate). Raised from "
             "clustering.py's own default of 300 m because that value "
             "was aggressively over-splitting elongated but coherent "
             "structures (long ships, port infrastructure). Default: 500.",
    )
    parser.add_argument(
        "--rg-search-m", type=float, default=30.0,
        help="Range linkage radius for clustering.cluster_peaks. Raised "
             "from clustering.py's own default of 20 m for the same "
             "reason as --az-search-m: pairs of peaks that are less "
             "than half a ship apart in range should merge even if "
             "one has slightly biased range centring. Default: 30.",
    )
    # -----------------------------------------------------------------
    # Grow-and-recenter hybrid (opt-in). Turned on with --grow-boxes.
    # -----------------------------------------------------------------
    # When enabled, after `cluster_peaks` groups the seed peaks into
    # per-target clusters, each cluster is reduced to a SINGLE seed
    # (the brightest peak by `sub_mean`) and passed to a local port of
    # shear_averaging.grow_and_recenter_boxes. The grown box is
    # constrained to always contain the cluster's full peak bounding
    # rectangle (`must_contain_yxyx`), so the amp-CoM recentre step
    # cannot drift off onto a neighbouring brighter target. Growth
    # uses the isolated-pixel-filtered CoV mask as the density gate
    # and `sub_mean` as the amplitude for the CoM recentre — both
    # already computed above, no full-resolution SLC needed. Post-
    # growth an azimuth-only extension (5% × h × N_steps per side)
    # with a looser density gate pushes the box's tails further out.
    parser.add_argument(
        "--grow-boxes",
        action="store_true",
        help="Enable the cluster-then-grow hybrid: after "
             "`cluster_peaks` finalises the peak clusters, each "
             "cluster's peaks are reduced to a single seed (the "
             "brightest by `sub_mean`) and grown per-edge with a "
             "density gate + rescue look-ahead, constrained so the "
             "grown box always contains the cluster's full peak "
             "bounding rectangle. A final +5%%×h × 2 azimuth extension "
             "pass with a looser density gate pushes the tails out. "
             "This fixes both `cluster_peaks` failure modes: "
             "under-extended azimuth (per-peak profile cutoff at "
             "0.15·peak) and multiple targets merged at similar "
             "range (arc-shape / az-gap post-splits, still applied "
             "before growth). Default: off (unchanged behaviour).",
    )
    parser.add_argument(
        "--grow-density-th", type=float, default=0.25,
        help="Density gate used by the grow-and-recenter pass on "
             "every candidate azimuth (and range) strip: an edge "
             "only moves outward if the new strip's mask density ≥ "
             "this value. Azimuth is tried first (range only advances "
             "after both azimuth edges are stuck), so lowering this "
             "primarily lets boxes grow FURTHER IN AZIMUTH before "
             "they hit the density gate. Default: 0.25 (matches "
             "shear_averaging.py's --grow-density-th default).",
    )
    parser.add_argument(
        "--grow-az-step-m", type=float, default=10.0,
        help="Azimuth grow-step in metres. Converted to decimated "
             "rows via az_m_per_px_dec (with a minimum floor of 1 "
             "row so growth actually advances). At the typical "
             "decimated az spacing of ~10 m/row this means one row "
             "per iteration. Default: 10.0 m.",
    )
    parser.add_argument(
        "--grow-az-tail-m", type=float, default=None,
        help="Azimuth tail length in metres for the density strip "
             "test. Each candidate az edge is evaluated on "
             "(new_step_rows + tail_rows) of the box's existing "
             "interior on the same side, so a single sparse row "
             "on the box's inside doesn't collapse the density "
             "check. Default: same as --grow-az-step-m — i.e. the "
             "primary strip test spans `new_step_rows + "
             "step_rows_worth_of_interior`. Pass an explicit value "
             "here to decouple tail and step (larger tail → more "
             "permissive gate; smaller → stricter).",
    )
    parser.add_argument(
        "--grow-rescue-lookahead-m", type=float, default=100.0,
        help="Az rescue look-ahead in metres. When the primary "
             "density gate fails, a fallback test peeks this many "
             "metres OUTSIDE the box (plus --grow-rescue-lookback-m "
             "of the interior) with the stricter "
             "--grow-rescue-th threshold. Bridges sparse-mask ripple "
             "inside otherwise dense targets. Default: 100.0 m.",
    )
    parser.add_argument(
        "--grow-rescue-lookback-m", type=float, default=100.0,
        help="Az rescue look-back in metres — how much of the box's "
             "interior (on the same side) is included in the rescue "
             "density test. Larger → the rescue averages over a "
             "longer interior tail, making it more permissive. "
             "Default: 100.0 m.",
    )
    parser.add_argument(
        "--grow-rescue-th", type=float, default=0.5,
        help="Density threshold used by the rescue test (see "
             "--grow-rescue-lookahead-m). Kept stricter than "
             "--grow-density-th because the rescue evaluates a much "
             "wider strip and would otherwise fire on sparse noise. "
             "Default: 0.5.",
    )
    parser.add_argument(
        "--grow-max-h-m", type=float, default=2000.0,
        help="Hard cap on the grown box's azimuth extent, in metres. "
             "Growth stops before the box exceeds this size. Long "
             "enough to accept large ships / azimuth-mover smears; "
             "raise for extreme cases. Default: 2000.0 m.",
    )
    parser.add_argument(
        "--grow-max-w-m", type=float, default=100.0,
        help="Hard cap on the grown box's range extent, in metres. "
             "Growth stops before the box exceeds this size. "
             "Default: 100.0 m.",
    )
    parser.add_argument(
        "--extend-density-th", type=float, default=0.10,
        help="Density gate for the POST-growth azimuth-only "
             "extension: each box tries to push its top/bottom "
             "edges outward by 5%%×h with this looser threshold. "
             "Rationale: growth already found the dense core; here "
             "we only want a faint hint the target continues. "
             "Default: 0.10 (matches shear_averaging.py).",
    )
    parser.add_argument(
        "--extend-max-steps", type=int, default=2,
        help="Number of 5%%×height post-growth extension steps per "
             "azimuth side (top and bottom independent). Total "
             "maximum extension per side ≈ 5%% × N_steps × height at "
             "growth exit. Default: 2 (up to +10%% per side).",
    )
    parser.add_argument(
        "--cfar-snr-th", type=float, default=0.0,
        help="Post-clustering CA-CFAR SNR threshold. For each cluster: "
             "signal = mean(sub_mean over mask==1 pixels inside the "
             "box); noise = min of the mean of an "
             "`az_extent × --cfar-range-bins`-pixel strip immediately "
             "before / after the box in range. Clusters with "
             "signal/noise < th are deleted. Set to 0 (or a negative "
             "number) to disable. Default: 0.0 (DISABLED — arc-shaped "
             "azimuth-mover targets cannot pass a CFAR test whose noise "
             "reference is the same arc's tail in the adjacent range "
             "strip). Pass a positive value (e.g. 4.0) to re-enable.",
    )
    parser.add_argument(
        "--cfar-range-bins", type=int, default=2,
        help="Width (in range pixels of the mask_dec grid) of the CA-"
             "CFAR guard strips on either side of the cluster box. "
             "Default: 2.",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Optional PNG output path. If omitted (default), the figure "
             "is only shown interactively and not written to disk.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Skip interactive plt.show() (useful together with --save).",
    )
    args = parser.parse_args()

    with np.load(args.path, allow_pickle=False) as z:
        missing = [k for k in ("sub_mean", "sub_var") if k not in z.files]
        if missing:
            raise KeyError(
                f"{args.path.name} is missing required keys {missing}. "
                "Regenerate with scripts/save_subapertures.py."
            )
        sub_mean = z["sub_mean"].astype(np.float64)
        sub_var = z["sub_var"].astype(np.float64)
        n_sub = int(z["N_subaperture"]) if "N_subaperture" in z.files else None
        n_rl = int(z["Number_of_Range_Looks"]) if "Number_of_Range_Looks" in z.files else None
        az_m = float(z["az_m_per_px_s_degraded"]) if "az_m_per_px_s_degraded" in z.files else None
        rg_m = float(z["rg_m_per_px_s_degraded"]) if "rg_m_per_px_s_degraded" in z.files else None

    print(f"Loaded {args.path}")
    print(f"  sub_mean/sub_var shape={sub_mean.shape}, dtype={sub_mean.dtype}")
    if n_sub is not None:
        print(f"  N_subaperture={n_sub}, Number_of_Range_Looks={n_rl}")
    if az_m is not None:
        print(f"  s_degraded spacing: {az_m} m (az) × {rg_m} m (rg)")
        if n_sub is not None:
            print(f"  decimated az spacing: {az_m} × {n_sub} = {az_m * n_sub} m/row")

    cov_sq = sub_var / (sub_mean ** 2 + args.eps)

    cov_p50, cov_p90, cov_p99 = (float(np.percentile(cov_sq, p)) for p in (50.0, 90.0, 99.0))
    mean_p50 = float(np.median(sub_mean))
    print(f"  sub_mean p50 = {mean_p50:.3g}")
    print(f"  cov_sq   p50/p90/p99 = {cov_p50:.3g}/{cov_p90:.3g}/{cov_p99:.3g}")

    # --- CoV mask (mirrors shear_averaging.py:3908-3933) ------------------
    # Kept structurally identical to the pipeline gate so a mask produced
    # here is bit-identical to what shear_averaging would compute on the
    # same statistics, absent the downstream isolated-pixel filter.
    th_off: float | None = None
    th_dark: float | None = None
    th_bright: float | None = None
    amp_dark_thresh: float | None = None
    median_dark: float | None = None
    median_bright: float | None = None
    dark_dec: np.ndarray | None = None
    cov_th_mult = args.cov_th_mult
    if args.bright_mode == "off":
        th_off = cov_th_mult * float(np.median(cov_sq))
        mask_dec = (cov_sq > th_off).astype(np.float32)
    else:
        amp_dark_thresh = float(
            np.percentile(sub_mean, args.dark_amp_percentile)
        )
        dark_dec = sub_mean <= amp_dark_thresh
        cov_sq_dark = cov_sq[dark_dec] if dark_dec.any() else cov_sq
        median_dark = float(np.median(cov_sq_dark))
        th_dark = cov_th_mult * median_dark
        if args.bright_mode == "exclude":
            mask_dec = (dark_dec & (cov_sq > th_dark)).astype(np.float32)
        else:  # "higher-th"
            bright_dec = ~dark_dec
            median_bright = float(
                np.median(cov_sq[bright_dec]) if bright_dec.any() else median_dark
            )
            th_bright = args.bright_cov_th_mult * median_bright
            mask_dec = (
                (dark_dec & (cov_sq > th_dark))
                | (bright_dec & (cov_sq > th_bright))
            ).astype(np.float32)

    n_kept = int(mask_dec.sum())
    n_total = mask_dec.size
    if args.bright_mode == "off":
        print(
            f"  CoV gate (mode='off'): mult={args.cov_th_mult:g} · "
            f"median = {th_off:.3g}  (log10={np.log10(th_off):.2f}) → "
            f"kept {n_kept}/{n_total} ({100 * n_kept / n_total:.2f}%)"
        )
    else:
        n_dark = int(dark_dec.sum())
        n_bright = int((~dark_dec).sum())
        print(
            f"  CoV gate (mode='{args.bright_mode}'): "
            f"amp ≤ p{args.dark_amp_percentile:g} = {amp_dark_thresh:.3g}"
        )
        print(
            f"    dark   : {n_dark}/{n_total} px · median_dark = "
            f"{median_dark:.3g} · th_dark = {th_dark:.3g}"
        )
        if args.bright_mode == "higher-th":
            print(
                f"    bright : {n_bright}/{n_total} px · median_bright = "
                f"{median_bright:.3g} · th_bright = {th_bright:.3g}"
            )
        else:
            print(f"    bright : {n_bright}/{n_total} px · zeroed (mode='exclude')")
        print(f"    total kept: {n_kept}/{n_total} ({100 * n_kept / n_total:.2f}%)")

    # --- Isolated-pixel filter (parity with run_clustering.py) ------------
    # Same 3 m × 3 m physical window as shear_averaging & run_clustering,
    # but with a density (not count) survival test so the meaning is
    # scene-invariant. Pixel spacings on the decimated grid are
    # az_m × N_subaperture along azimuth, rg_m along range.
    if az_m is not None and n_sub is not None and rg_m is not None:
        az_m_dec = az_m * n_sub
        rg_m_dec = rg_m
    else:
        # Fallback so peak detection & clustering still run when spacings
        # are missing. Chose 0.64 × 0.96 m (ICEYE spot × 16 subapertures)
        # as a reasonable default matching the shipped test data.
        az_m_dec = 0.64
        rg_m_dec = 0.96
        print("  (spacings not in npz; using fallback 0.64 × 0.96 m/px)")

    az_filt_win = max(1, int(round(args.filter_target_size_m / az_m_dec)))
    rg_filt_win = max(1, int(round(args.filter_target_size_m / rg_m_dec)))
    density = uniform_filter(
        mask_dec, size=(az_filt_win, rg_filt_win), mode="constant",
    )
    mask_filt = (mask_dec.astype(bool) & (density >= args.filter_min_density)).astype(np.float32)
    n_kept_before = int(mask_dec.sum())
    n_kept_after = int(mask_filt.sum())
    print(
        f"  isolated-pixel filter "
        f"({args.filter_target_size_m:g} m × {args.filter_target_size_m:g} m "
        f"= {az_filt_win} az × {rg_filt_win} rg px, density ≥ "
        f"{args.filter_min_density:.2f}): kept {n_kept_after}/{n_kept_before} "
        f"({100 * n_kept_after / max(n_kept_before, 1):.2f}%)"
    )

    # --- Peak detection (parity with shear_averaging.py:4070-4092) --------
    # Seed window is (N_Target_Azimuth_Width, N_Target_Range_Length) on the
    # s_degraded grid. Rescale the azimuth extent to the decimated grid by
    # dividing by N_subaperture; range extent is unchanged.
    az_seed_win = max(
        1,
        int(round(args.n_target_azimuth_width / (n_sub if n_sub else 16))),
    )
    rg_seed_win = int(args.n_target_range_length)
    box_size = (az_seed_win, rg_seed_win)
    box_area = az_seed_win * rg_seed_win

    # sub_mean is the Doppler-mean amplitude on the SAME decimated grid as
    # mask_dec; it plays the role `|s_degraded|` plays in the pipeline.
    amp_proxy = sub_mean.astype(np.float32)
    det_count = uniform_filter(
        mask_filt.astype(np.float32), size=box_size, mode="constant",
    ) * box_area
    amp_max = maximum_filter(amp_proxy, size=box_size, mode="constant")

    is_detection = mask_filt == 1
    is_dense = det_count > args.seed_density_th * box_area
    is_peak = amp_proxy == amp_max
    boundary = is_detection & is_dense & is_peak
    peaks_yx = np.argwhere(boundary).astype(np.int64)
    print(
        f"  seed peaks: window=({az_seed_win},{rg_seed_win}) "
        f"[≈ ({args.n_target_azimuth_width},{rg_seed_win}) at s_degraded], "
        f"density > {args.seed_density_th:g} → {len(peaks_yx)} peaks"
    )

    # --- Clustering (clustering.cluster_peaks) ----------------------------
    labels = np.empty(0, dtype=np.int64)
    boxes = np.empty((0, 4), dtype=np.int64)
    if len(peaks_yx) == 0:
        print("  cluster_peaks: no peaks → no clusters")
    else:
        # The three post-linkage splitters below are the main causes
        # of over-splitting on this scene, so all three are dialled
        # toward "keep together" by default here (the user prefers
        # occasional under-splitting over over-splitting):
        #   * `_split_arc_clusters`   — disabled entirely (arc_az_thr=inf).
        #   * `_split_az_gap_clusters`— disabled by default via
        #                               --max-peak-gap-az-m = +inf.
        #   * hierarchical linkage    — search radii raised via
        #                               --az-search-m and --rg-search-m.
        arc_az_thr_m = 200.0 if args.enable_arc_split else float("inf")
        arc_rg_max_m = 20.0
        t0 = time.time()
        labels, boxes = cluster_peaks(
            mask_filt.astype(np.uint8), peaks_yx,
            az_m_per_px=az_m_dec, rg_m_per_px=rg_m_dec,
            az_search_m=args.az_search_m,
            rg_search_m=args.rg_search_m,
            arc_az_thr_m=arc_az_thr_m,
            arc_rg_max_m=arc_rg_max_m,
            max_peak_gap_az_m=args.max_peak_gap_az_m,
            enforce_max_size=False,
        )
        n_clusters = int(labels.max()) + 1 if len(labels) else 0
        sizes = np.bincount(labels) if len(labels) else np.array([], dtype=int)
        h = boxes[:, 1] - boxes[:, 0]
        w = boxes[:, 3] - boxes[:, 2]
        print(
            f"  cluster_peaks: {len(peaks_yx)} peaks → {n_clusters} clusters "
            f"in {time.time() - t0:.2f} s"
        )
        if n_clusters:
            print(
                f"    peaks/cluster : min={sizes.min()}, "
                f"median={int(np.median(sizes))}, max={sizes.max()}"
            )
            print(
                f"    box az extent : min={h.min()}, median={int(np.median(h))}, "
                f"max={h.max()} px  "
                f"({h.min()*az_m_dec:.0f} .. {h.max()*az_m_dec:.0f} m)"
            )
            print(
                f"    box rg extent : min={w.min()}, median={int(np.median(w))}, "
                f"max={w.max()} px  "
                f"({w.min()*rg_m_dec:.0f} .. {w.max()*rg_m_dec:.0f} m)"
            )

    # --- Cluster → grow hybrid ------------------------------------------
    # For each cluster:
    #   * seed = brightest peak (by sub_mean) in the cluster,
    #   * must_contain = axis-aligned bounding rectangle of the cluster's
    #     peaks (inclusive; grown box must always cover it).
    # Growth uses the isolated-pixel-filtered CoV mask as the density
    # gate and sub_mean as the amplitude for the CoM recentre — both
    # already computed above. No full-resolution SLC needed. The
    # `_extend_boxes_azimuth_strong_signal` post-pass pushes the top /
    # bottom edges out with a looser density gate.
    # The pre-growth cluster boxes and the growth output are both kept
    # (as `boxes_cluster_display` and `boxes` respectively) so the
    # rendering can put them side-by-side.
    boxes_cluster_display = boxes.copy()
    labels_cluster_display = labels.copy()
    peaks_cluster_display = peaks_yx.copy()
    grew_ran = False
    seed_peaks_yx: np.ndarray = np.empty((0, 2), dtype=np.int64)
    if args.grow_boxes and len(peaks_yx) and len(boxes):
        n_clusters_pre = int(labels.max()) + 1 if len(labels) else 0
        # Convert metre-based CLI to decimated-grid pixel counts. All
        # step / tail / lookahead sizes must be at least 1 row so the
        # loop actually advances. `--grow-az-tail-m` defaults to the
        # step size so the primary strip test is exactly
        # (new step-rows) + (equal-number-of-rows of interior) unless
        # the user overrides the tail explicitly.
        az_tail_m_eff = (
            args.grow_az_step_m
            if args.grow_az_tail_m is None
            else args.grow_az_tail_m
        )
        az_step_px = max(1, int(round(args.grow_az_step_m / az_m_dec)))
        az_tail_px = max(1, int(round(az_tail_m_eff / az_m_dec)))
        az_look_ahead_px = max(1, int(round(args.grow_rescue_lookahead_m / az_m_dec)))
        az_look_back_px = max(1, int(round(args.grow_rescue_lookback_m / az_m_dec)))
        max_h_px = max(1, int(round(args.grow_max_h_m / az_m_dec)))
        max_w_px = max(1, int(round(args.grow_max_w_m / rg_m_dec)))

        seed_list = []
        must_contain_list = []
        for c in range(n_clusters_pre):
            mem = np.where(labels == c)[0]
            if len(mem) == 0:
                continue
            member_peaks = peaks_yx[mem]  # (Nc, 2)
            # Brightest seed by sub_mean amplitude at each peak's
            # pixel. Ties broken by array order (argmax → first max).
            amps_at_peaks = amp_proxy[
                member_peaks[:, 0].astype(np.int64),
                member_peaks[:, 1].astype(np.int64),
            ]
            best_local = int(np.argmax(amps_at_peaks))
            seed_list.append(member_peaks[best_local])
            py_lo, py_hi = int(member_peaks[:, 0].min()), int(member_peaks[:, 0].max())
            px_lo, px_hi = int(member_peaks[:, 1].min()), int(member_peaks[:, 1].max())
            must_contain_list.append((py_lo, py_hi, px_lo, px_hi))
        seed_peaks_yx = np.asarray(seed_list, dtype=np.int64)
        must_contain_yxyx = np.asarray(must_contain_list, dtype=np.int64)

        # Growth uses the SAME seed window as peak detection as the
        # initial box footprint (so a single-peak cluster starts from
        # the exact box it would have gotten with --peaks-as-boxes).
        t_g = time.time()
        grown = _grow_and_recenter_boxes(
            seed_peaks_yx,
            mask=mask_filt.astype(np.float32),
            amp=amp_proxy,
            initial_hw=(az_seed_win, rg_seed_win),
            az_step=az_step_px,
            rg_step=1,
            density_threshold=args.grow_density_th,
            az_tail=az_tail_px,
            rg_tail=1,
            az_rescue_lookahead=az_look_ahead_px,
            az_rescue_lookback=az_look_back_px,
            az_rescue_threshold=args.grow_rescue_th,
            max_h=max_h_px,
            max_w=max_w_px,
            must_contain_yxyx=must_contain_yxyx,
        )
        grown = _extend_boxes_azimuth_strong_signal(
            grown,
            mask=mask_filt.astype(np.float32),
            step_frac=0.05,
            max_steps=args.extend_max_steps,
            density_threshold=args.extend_density_th,
            max_h=max_h_px,
        )
        boxes = grown
        grew_ran = True
        gh = boxes[:, 1] - boxes[:, 0] + 1
        gw = boxes[:, 3] - boxes[:, 2] + 1
        print(
            f"  grow-and-recenter: {n_clusters_pre} clusters → "
            f"{len(boxes)} grown boxes in {time.time() - t_g:.2f} s "
            f"(step={az_step_px}px [{args.grow_az_step_m:g} m], "
            f"tail={az_tail_px}px [{az_tail_m_eff:g} m"
            + (" ← =step" if args.grow_az_tail_m is None else "")
            + f"], density≥{args.grow_density_th:g}, "
            f"rescue lookahead={az_look_ahead_px}px "
            f"[{args.grow_rescue_lookahead_m:g} m], "
            f"cap h×w={max_h_px}×{max_w_px}px "
            f"[{args.grow_max_h_m:g}×{args.grow_max_w_m:g} m])"
        )
        print(
            f"    grown az extent : min={gh.min()}, median={int(np.median(gh))}, "
            f"max={gh.max()} px  "
            f"({gh.min()*az_m_dec:.0f} .. {gh.max()*az_m_dec:.0f} m)"
        )
        print(
            f"    grown rg extent : min={gw.min()}, median={int(np.median(gw))}, "
            f"max={gw.max()} px  "
            f"({gw.min()*rg_m_dec:.0f} .. {gw.max()*rg_m_dec:.0f} m)"
        )

    # --- CA-CFAR filter on the clusters -----------------------------------
    # For each cluster: signal = mean(sub_mean over mask==1 pixels inside
    # the box). Noise = min of the mean over an
    # `az_box_extent × cfar_range_bins` guard strip immediately before /
    # after the box in range. If signal / noise < --cfar-snr-th the
    # cluster is dropped. When the box sits at a range edge and one of
    # the strips is empty, the other one carries the noise estimate
    # alone; if both are empty (extremely narrow scene) the cluster is
    # kept unconditionally.
    cfar_snrs = np.full(len(boxes), np.nan, dtype=np.float64)
    cfar_signal = np.full(len(boxes), np.nan, dtype=np.float64)
    cfar_noise = np.full(len(boxes), np.nan, dtype=np.float64)
    keep_cluster = np.ones(len(boxes), dtype=bool)
    if len(boxes) and args.cfar_snr_th > 0:
        H_mask, W_mask = mask_filt.shape
        n_guard = int(max(1, args.cfar_range_bins))
        for c in range(len(boxes)):
            y_lo, y_hi, x_lo, x_hi = (int(v) for v in boxes[c])
            box_amp = amp_proxy[y_lo:y_hi + 1, x_lo:x_hi + 1]
            box_msk = mask_filt[y_lo:y_hi + 1, x_lo:x_hi + 1]
            sig_pix = box_amp[box_msk == 1]
            if sig_pix.size == 0:
                # Box devoid of mask pixels (post-tightening artefact);
                # falling back to the whole box keeps the CFAR ratio well
                # defined and never accidentally admits an empty cluster
                # since box_amp on such rows is essentially pure noise.
                sig_pix = box_amp
                if sig_pix.size == 0:
                    keep_cluster[c] = False
                    continue
            signal = float(sig_pix.mean())

            prev_lo = max(0, x_lo - n_guard)
            succ_hi = min(W_mask, x_hi + 1 + n_guard)
            prev_strip = amp_proxy[y_lo:y_hi + 1, prev_lo:x_lo]
            succ_strip = amp_proxy[y_lo:y_hi + 1, x_hi + 1:succ_hi]

            noise_means = []
            if prev_strip.size:
                noise_means.append(float(prev_strip.mean()))
            if succ_strip.size:
                noise_means.append(float(succ_strip.mean()))
            if not noise_means:
                # Box spans the full range extent; no CFAR reference
                # available. Keep the cluster rather than deleting
                # something we cannot evaluate.
                continue
            noise = min(noise_means)
            snr = signal / max(noise, 1e-12)

            cfar_signal[c] = signal
            cfar_noise[c] = noise
            cfar_snrs[c] = snr
            if snr < args.cfar_snr_th:
                keep_cluster[c] = False

        n_before = len(boxes)
        n_after = int(keep_cluster.sum())
        snrs_eval = cfar_snrs[~np.isnan(cfar_snrs)]
        if snrs_eval.size:
            snr_p10, snr_p50, snr_p90 = (
                float(np.percentile(snrs_eval, p)) for p in (10.0, 50.0, 90.0)
            )
        else:
            snr_p10 = snr_p50 = snr_p90 = float("nan")
        print(
            f"  CA-CFAR (guard = az_box × {n_guard} rg px, snr_th = "
            f"{args.cfar_snr_th:g}): kept {n_after}/{n_before} clusters "
            f"({100 * n_after / max(n_before, 1):.1f}%)  ·  "
            f"snr p10/p50/p90 = {snr_p10:.2g}/{snr_p50:.2g}/{snr_p90:.2g}"
        )

        # Apply the filter: drop killed clusters from labels/boxes and
        # dense-remap so `n_clusters` stays contiguous in [0, N).
        boxes = boxes[keep_cluster]
        keep_pk = keep_cluster[labels]
        peak_labels_kept = labels[keep_pk]
        peaks_kept = peaks_yx[keep_pk]
        # Remap surviving cluster ids to [0, N).
        _, remap = np.unique(peak_labels_kept, return_inverse=True)
        labels = remap.astype(np.int64)
        peaks_yx = peaks_kept
        cfar_snrs_kept = cfar_snrs[keep_cluster]
    elif len(boxes) and args.cfar_snr_th <= 0:
        print(f"  CA-CFAR: skipped (--cfar-snr-th = {args.cfar_snr_th:g})")
        cfar_snrs_kept = cfar_snrs

    # --- Rendering --------------------------------------------------------
    # Match shear_averaging.py's log10 rendering for the CoV panel; use
    # the same trick for the mean so both panels are contrast-comparable.
    log_mean = np.log10(np.maximum(sub_mean, 1e-6))
    log_cov = np.log10(np.maximum(cov_sq, 1e-6))

    vmin_m, vmax_m = _clip_range(log_mean)
    vmin_c, vmax_c = _clip_range(log_cov)

    # imshow aspect ratio: 'auto' stretches each panel to fill its axes
    # regardless of pixel spacing. Preferred here because the decimated
    # grid is very tall (7162 × 1961), so a physically-proportioned figure
    # is skinny and hard to inspect at a glance.
    aspect = "auto"

    # When the grow-and-recenter hybrid ran, add a 5th panel dedicated
    # to the grown boxes so cluster (tightened) and grown outputs sit
    # side-by-side for comparison. Otherwise the classic 4-panel
    # layout is used unchanged.
    n_panels = 5 if grew_ran else 4
    fig, axes = plt.subplots(
        1, n_panels, figsize=(6 * n_panels, 8),
        constrained_layout=True, sharey=True,
    )
    ax_m, ax_c, ax_k, ax_cl = axes[:4]
    ax_gr = axes[4] if grew_ran else None
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    n_clusters_display = (
        int(labels_cluster_display.max()) + 1
        if len(labels_cluster_display) else 0
    )
    suptitle_head = (
        f"sub-aperture mean  vs  CoV²  vs  mask  vs  clusters"
        + ("  vs  grown" if grew_ran else "")
        + f"  —  {args.path.name}"
    )
    title_bits = [
        suptitle_head,
        (
            f"{len(peaks_cluster_display)} peaks → "
            f"{n_clusters_display} clusters"
            + (
                f" → {len(boxes)} grown boxes"
                if grew_ran else ""
            )
        ),
    ]
    if args.bright_mode == "off":
        title_bits.append(
            f"mode='off', th = {args.cov_th_mult:g}·median = "
            f"{th_off:.3g} (log10={np.log10(th_off):.2f})"
        )
    else:
        bright_bit = (
            f", th_bright={th_bright:.3g} (log10={np.log10(th_bright):.2f})"
            if th_bright is not None else ""
        )
        title_bits.append(
            f"mode='{args.bright_mode}', "
            f"th_dark={th_dark:.3g} (log10={np.log10(th_dark):.2f})"
            + bright_bit
        )
    fig.suptitle("  —  ".join(title_bits), fontsize=10)

    im_m = ax_m.imshow(
        log_mean, cmap="gray", aspect=aspect,
        vmin=vmin_m, vmax=vmax_m, interpolation="nearest",
    )
    ax_m.set_title(
        f"log10(sub_mean)  [p50 = {mean_p50:.3g}]", fontsize=10,
    )
    ax_m.set_xlabel("range pixel (s_degraded)")
    ax_m.set_ylabel(
        f"sub-aperture-index row"
        + (f"  (× {n_sub} = azimuth pixel)" if n_sub is not None else "")
    )
    fig.colorbar(im_m, ax=ax_m, shrink=0.75).set_label("log10(mean amplitude)")

    im_c = ax_c.imshow(
        log_cov, cmap="viridis", aspect=aspect,
        vmin=vmin_c, vmax=vmax_c, interpolation="nearest",
    )
    # Overlay the applied threshold(s) as contour(s), same convention as
    # `fig_cv` in shear_averaging.py (red = dark/scene-wide, orange = bright).
    if args.bright_mode == "off":
        ax_c.contour(log_cov, levels=[np.log10(th_off)],
                     colors="red", linewidths=0.6)
    else:
        ax_c.contour(log_cov, levels=[np.log10(th_dark)],
                     colors="red", linewidths=0.6)
        if th_bright is not None:
            ax_c.contour(log_cov, levels=[np.log10(th_bright)],
                         colors="orange", linewidths=0.6)
    ax_c.set_title(
        f"log10(cov²)  [p50/p90/p99 = {cov_p50:.2g}/{cov_p90:.2g}/{cov_p99:.2g}]",
        fontsize=10,
    )
    ax_c.set_xlabel("range pixel (s_degraded)")
    fig.colorbar(im_c, ax=ax_c, shrink=0.75).set_label("log10(cov²)")

    ax_k.imshow(
        mask_filt, cmap="gray", aspect=aspect,
        vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    ax_k.set_title(
        f"CoV mask (mode='{args.bright_mode}')  "
        f"kept {n_kept_after}/{n_total} = "
        f"{100 * n_kept_after / n_total:.2f}%\n"
        f"(after {args.filter_target_size_m:g} m × "
        f"{args.filter_target_size_m:g} m density ≥ "
        f"{args.filter_min_density:.2f} filter)",
        fontsize=10,
    )
    ax_k.set_xlabel("range pixel (s_degraded)")

    # --- Panel (d) [+ optional (e)]: mask + peaks + boxes ---------------
    # Same rendering primitive for the tightened cluster boxes (panel d)
    # and the grown boxes (panel e, only when --grow-boxes ran). Both
    # panels share the mask background and the tab20 label→colour map;
    # the only difference is which (boxes, labels, peaks) triple they
    # consume.
    def _draw_boxes(
        ax,
        boxes_local: np.ndarray,
        labels_local: np.ndarray,
        peaks_local: np.ndarray,
        title: str,
        background: np.ndarray | None = None,
        bg_cmap: str = "gray",
        bg_vmin: float | None = None,
        bg_vmax: float | None = None,
    ) -> None:
        # Default background is the isolated-pixel-filtered CoV mask
        # (same as the historical behaviour). Pass `background=log_mean`
        # + bg_vmin/bg_vmax to overlay the boxes on the sub-aperture
        # mean amplitude instead.
        bg = mask_filt if background is None else background
        vmin = 0.0 if background is None else bg_vmin
        vmax = 1.0 if background is None else bg_vmax
        ax.imshow(
            bg, cmap=bg_cmap, aspect=aspect,
            vmin=vmin, vmax=vmax, interpolation="nearest",
        )
        n_local = int(labels_local.max()) + 1 if len(labels_local) else 0
        if n_local:
            cmap_cl = plt.get_cmap("tab20")
            for c in range(n_local):
                colour = cmap_cl(c % 20)
                sel = labels_local == c
                ax.scatter(
                    peaks_local[sel, 1], peaks_local[sel, 0],
                    s=10, c=[colour], edgecolor="none",
                )
                if c < len(boxes_local):
                    y_lo, y_hi, x_lo, x_hi = boxes_local[c]
                    ax.add_patch(Rectangle(
                        (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                        fill=False, edgecolor=colour, linewidth=0.7,
                    ))
        elif len(peaks_local):
            # No clusters (shouldn't happen if there are peaks) — still show
            # the peaks so it's visible how many the seed step produced.
            ax.scatter(peaks_local[:, 1], peaks_local[:, 0], s=10,
                       c="red", marker="x", linewidths=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("range pixel (s_degraded)")

    _draw_boxes(
        ax_cl,
        boxes_cluster_display,
        labels_cluster_display,
        peaks_cluster_display,
        f"cluster_peaks(): {len(peaks_cluster_display)} peaks "
        f"→ {n_clusters_display} clusters",
    )
    if grew_ran and ax_gr is not None:
        _draw_boxes(
            ax_gr,
            boxes,
            labels,
            peaks_yx,
            f"grow-and-recenter: {len(boxes)} grown boxes  "
            f"(density≥{args.grow_density_th:g}, "
            f"cap {args.grow_max_h_m:g}×{args.grow_max_w_m:g} m)",
        )
        # Mark the single seed used per cluster (brightest peak) so the
        # user can see WHICH seed each grown box was launched from.
        if seed_peaks_yx.size:
            ax_gr.scatter(
                seed_peaks_yx[:, 1], seed_peaks_yx[:, 0],
                s=25, facecolors="none", edgecolors="white",
                linewidths=0.8, marker="o",
            )

    # --- Standalone clusters figure (same content as panel (d)) ---------
    # Same aspect / imshow settings as the 4-panel version; taller figure
    # so the very tall decimated grid (7162 × 1961) is legible on its own.
    fig_cl, ax_cl_solo = plt.subplots(figsize=(8, 12), constrained_layout=True)
    _draw_boxes(
        ax_cl_solo,
        boxes_cluster_display,
        labels_cluster_display,
        peaks_cluster_display,
        f"cluster_peaks(): {len(peaks_cluster_display)} peaks "
        f"→ {n_clusters_display} clusters",
    )
    ax_cl_solo.set_ylabel(
        f"sub-aperture-index row"
        + (f"  (× {n_sub} = azimuth pixel)" if n_sub is not None else "")
    )
    fig_cl.suptitle(f"{args.path.name}", fontsize=10)

    # --- Standalone clusters-on-mean figure -----------------------------
    # Same boxes / peaks as `fig_cl`, but drawn on top of the sub-aperture
    # log10(mean) amplitude instead of the CoV mask. Makes it easier to
    # judge whether each cluster box actually encloses a bright signature
    # in the imagery — the mask panel can be misleading when the mask is
    # sparse / speckled, whereas the mean image shows the underlying
    # target structure directly.
    fig_cl_mean, ax_cl_mean_solo = plt.subplots(
        figsize=(8, 12), constrained_layout=True,
    )
    _draw_boxes(
        ax_cl_mean_solo,
        boxes_cluster_display,
        labels_cluster_display,
        peaks_cluster_display,
        f"cluster_peaks() on log10(sub_mean): "
        f"{len(peaks_cluster_display)} peaks "
        f"→ {n_clusters_display} clusters",
        background=log_mean,
        bg_cmap="gray",
        bg_vmin=vmin_m,
        bg_vmax=vmax_m,
    )
    ax_cl_mean_solo.set_ylabel(
        f"sub-aperture-index row"
        + (f"  (× {n_sub} = azimuth pixel)" if n_sub is not None else "")
    )
    fig_cl_mean.suptitle(f"{args.path.name}", fontsize=10)

    # --- Standalone grown-boxes figure (only when growth ran) -----------
    fig_gr = None
    if grew_ran:
        fig_gr, ax_gr_solo = plt.subplots(
            figsize=(8, 12), constrained_layout=True,
        )
        _draw_boxes(
            ax_gr_solo,
            boxes,
            labels,
            peaks_yx,
            f"grow-and-recenter: {len(boxes)} grown boxes  "
            f"(density≥{args.grow_density_th:g}, "
            f"cap {args.grow_max_h_m:g}×{args.grow_max_w_m:g} m)",
        )
        if seed_peaks_yx.size:
            ax_gr_solo.scatter(
                seed_peaks_yx[:, 1], seed_peaks_yx[:, 0],
                s=25, facecolors="none", edgecolors="white",
                linewidths=0.8, marker="o",
            )
        ax_gr_solo.set_ylabel(
            f"sub-aperture-index row"
            + (f"  (× {n_sub} = azimuth pixel)" if n_sub is not None else "")
        )
        fig_gr.suptitle(f"{args.path.name}", fontsize=10)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
        # Second output next to the first: <stem>_clusters<suffix>
        clusters_path = args.save.with_name(
            f"{args.save.stem}_clusters{args.save.suffix}"
        )
        fig_cl.savefig(clusters_path, dpi=150)
        print(f"Saved clusters-only figure to {clusters_path}")
        # Third output: <stem>_clusters_on_mean<suffix> — same clusters
        # drawn over log10(sub_mean) so the boxes can be checked against
        # the underlying amplitude imagery.
        clusters_mean_path = args.save.with_name(
            f"{args.save.stem}_clusters_on_mean{args.save.suffix}"
        )
        fig_cl_mean.savefig(clusters_mean_path, dpi=150)
        print(f"Saved clusters-on-mean figure to {clusters_mean_path}")
        # Fourth output (only when growth ran): <stem>_grown<suffix>
        if fig_gr is not None:
            grown_path = args.save.with_name(
                f"{args.save.stem}_grown{args.save.suffix}"
            )
            fig_gr.savefig(grown_path, dpi=150)
            print(f"Saved grown-boxes figure to {grown_path}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
