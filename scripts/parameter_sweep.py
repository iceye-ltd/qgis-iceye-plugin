"""Parameter sweep for shear_averaging.py: measure each filter's impact on
TP (boxes inside s_degraded[:, 70:120]) vs FA (boxes outside) and per-target
sensitivity (which filter, at what threshold, kills each baseline box).

The expensive CoV / NMS / merge stage is cached: we run it once for the given
(n_subaperture, cov_th_mult, scene), record per-box statistics for every
post-merge box, and then evaluate every parameter combination as a plain
predicate over those stats. This keeps the sweep instantaneous.

Filter order (must match shear_averaging.main()):
  (S0) post-merge box set                        (cached, depends on CoV stage)
  (S1) range-extent cap   :  w  <= max_w_px
  (S2) min-slope filter   :  |slope_per_row_deg| >= min_slope_deg
  (S3) residual filter    :  residual_rad <= max_phase_residual_rad
                              OR det_count > skip_frac * h_box
  (S4) range-band gate    :  x_c_min <= x_c <= x_c_max
  (S5) (COM PP filter — off in user's command)
  (S6) motion filter      :  |v_az| >= v_min OR |v_rg| >= v_min
                              OR |slope_total| >= slope_rad_thresh
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shear_averaging import (
    apply_range_window,
    degrade_range_resolution_range_sum,
    compute_subapertures,
    grow_and_recenter_boxes,
    nms_boxes,
    nms_by_centre_distance,
    merge_by_strip_amplitude,
    merge_overlapping_boxes,
    compute_subaperture_com_per_box,
    compute_box_com_velocities,
    compute_box_phase_estimates,
)
from scipy.ndimage import uniform_filter, maximum_filter


# Scene constants — must match shear_averaging.main()
RANGE_SPACING = 0.5
AZIMUTH_SPACING = 0.5
MIN_TARGET_SIZE = 3.0
NUMBER_OF_RANGE_LOOKS = int(MIN_TARGET_SIZE / RANGE_SPACING)        # 6
MAX_LENGTH_TARGET = 130
MAX_WIDTH_TARGET = 20
MAX_SIZE_RG_BIN = int(MAX_LENGTH_TARGET / (RANGE_SPACING * NUMBER_OF_RANGE_LOOKS))  # 43
MAX_SIZE_AZ_BIN = 2000
RANGE_SLL_DB = 55.0
RANGE_TAYLOR_NBAR = 8
N_TARGET_AZ_WIDTH = 100
N_TARGET_RG_LEN = 3
SEED_DENSITY_THRESHOLD = 0.5
GROW_DENSITY_THRESHOLD = 0.25
NMS_IOU_THRESH = 0.3
MAX_DIST_AZ_M = 1000.0
MAX_DIST_RG_M = MAX_WIDTH_TARGET
BRIDGE_STRENGTH = 0.7
MAX_RG_OFFSET_M = MAX_WIDTH_TARGET
MAX_AZ_GAP_M = 2000.0


@dataclass
class BoxStats:
    """All per-box features used by post-merge filters, computed once."""
    y_c: int
    x_c: int
    h: int
    w: int
    slope_per_row_rad: float       # rad / azimuth row
    slope_total_rad: float         # slope_per_row * n_rows
    intercept_rad: float
    residual_rad: float
    n_rows: int
    det_count: int                 # mask_full.sum() inside the box
    v_az_mps: float
    v_rg_mps: float
    az_pp_sub_px: float
    rg_pp_d_px: float


@dataclass
class SweepCache:
    """Everything needed to evaluate any post-CoV filter combo."""
    boxes_post_merge: np.ndarray   # (K, 4) yxhw, after merge_overlapping_boxes
    stats: list[BoxStats]


def compute_scene_to_post_merge(
    s: np.ndarray,
    n_subaperture: int,
    cov_th_mult: float,
) -> SweepCache:
    """Run the pipeline up to and including merge_overlapping_boxes,
    then compute every per-box feature used by the downstream filters."""
    s_windowed = apply_range_window(s, sll_db=RANGE_SLL_DB, nbar=RANGE_TAYLOR_NBAR)
    s_degraded = degrade_range_resolution_range_sum(
        s_windowed, NUMBER_OF_RANGE_LOOKS,
    )
    # d_phase is computed once here for the local per-box stats loop
    # below; the shear_averaging main pipeline computes d_phase per-box
    # inside the phase filters instead.
    d_phase = np.angle(s_degraded[1:] * np.conj(s_degraded[:-1]))
    _subs = compute_subapertures(s_degraded, n_subaperture)
    sub_mean = _subs.mean(axis=0)
    sub_var = _subs.var(axis=0)
    del _subs
    cov_sq = sub_var / (sub_mean ** 2 + 1e-12)
    th = cov_th_mult * float(np.median(cov_sq))
    mask_dec = (cov_sq > th).astype(np.float32)
    mask_full = np.repeat(mask_dec, n_subaperture, axis=0)
    pad = s_degraded.shape[0] - mask_full.shape[0]
    if pad > 0:
        mask_full = np.vstack([mask_full, np.repeat(mask_full[-1:], pad, axis=0)])
    amp_raw = np.abs(s_degraded).astype(np.float32)
    s_degraded = s_degraded * mask_full
    subapertures = compute_subapertures(s_degraded, n_subaperture)

    amp = np.abs(s_degraded).astype(np.float32)
    box_size = (N_TARGET_AZ_WIDTH, N_TARGET_RG_LEN)
    box_area = N_TARGET_AZ_WIDTH * N_TARGET_RG_LEN
    det_count = uniform_filter(mask_full.astype(np.float32),
                               size=box_size, mode="constant") * box_area
    amp_max = maximum_filter(amp, size=box_size, mode="constant")
    is_detection = mask_full == 1
    is_dense = det_count > SEED_DENSITY_THRESHOLD * box_area
    is_peak = amp == amp_max
    boundary_box = (is_detection & is_dense & is_peak).astype(np.float32)
    peaks_yx = np.argwhere(boundary_box == 1.0)

    boxes_yxhw = grow_and_recenter_boxes(
        peaks_yx, mask=mask_full, amp=amp,
        initial_hw=(N_TARGET_AZ_WIDTH, N_TARGET_RG_LEN),
        az_step=5, rg_step=1,
        density_threshold=GROW_DENSITY_THRESHOLD,
        max_h=MAX_SIZE_AZ_BIN, max_w=MAX_SIZE_RG_BIN,
    )
    if not len(boxes_yxhw):
        return SweepCache(boxes_post_merge=boxes_yxhw, stats=[])

    H_amp, W_amp = amp_raw.shape
    scores = np.empty(len(boxes_yxhw), dtype=np.float32)
    for i, (yc, xc, h, w) in enumerate(boxes_yxhw):
        yl = max(int(yc) - int(h) // 2, 0)
        yh = min(yl + int(h), H_amp)
        xl = max(int(xc) - int(w) // 2, 0)
        xh = min(xl + int(w), W_amp)
        sub = amp_raw[yl:yh, xl:xh]
        scores[i] = sub.max() if sub.size else 0.0

    keep = nms_boxes(boxes_yxhw, scores, iou_thresh=NMS_IOU_THRESH)
    boxes_yxhw, scores = boxes_yxhw[keep], scores[keep]
    keep = nms_by_centre_distance(
        boxes_yxhw, scores,
        az_spacing_m=AZIMUTH_SPACING,
        rg_spacing_m=RANGE_SPACING * NUMBER_OF_RANGE_LOOKS,
        max_dist_az_m=MAX_DIST_AZ_M,
        max_dist_rg_m=MAX_DIST_RG_M,
    )
    boxes_yxhw = boxes_yxhw[keep]
    boxes_yxhw = merge_by_strip_amplitude(
        boxes_yxhw, amp_raw,
        az_spacing_m=AZIMUTH_SPACING,
        rg_spacing_m=RANGE_SPACING * NUMBER_OF_RANGE_LOOKS,
        bridge_strength=BRIDGE_STRENGTH,
        max_rg_offset_m=MAX_RG_OFFSET_M,
        max_az_gap_m=MAX_AZ_GAP_M,
    )
    boxes_yxhw = merge_overlapping_boxes(boxes_yxhw)

    # Per-box stats: slope/residual on d_phase, COM velocities on subapertures.
    (box_phi, box_coh, box_slope_pr, box_intercept,
     box_y0, box_n_rows, box_residual, _box_n_inliers) = (
        compute_box_phase_estimates(boxes_yxhw, s_degraded)
    )
    com_az_sub, com_rg = compute_subaperture_com_per_box(
        boxes_yxhw, subapertures, n_subaperture,
    )
    v_az, v_rg = compute_box_com_velocities(
        com_az_sub, com_rg,
        sub_size=subapertures.shape[1],
        prf=6000.0,                           # baseline command value
        N_subaperture=n_subaperture,
        azimuth_spacing_m=AZIMUTH_SPACING,
        Number_of_Range_Looks=NUMBER_OF_RANGE_LOOKS,
        range_spacing_m=RANGE_SPACING,
    )

    n_az_d, n_rg_d = d_phase.shape
    stats: list[BoxStats] = []
    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        y0 = max(0, int(round(y_c - h / 2)))
        y1 = min(n_az_d, int(round(y_c + h / 2)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg_d, int(round(x_c + w / 2)))
        det_in_box = int(mask_full[y0:y1, x0:x1].sum())
        n_rows = int(box_n_rows[k]) if box_n_rows[k] else max(1, y1 - y0)
        slope_pr = float(box_slope_pr[k])
        slope_tot = float(slope_pr * n_rows) if np.isfinite(slope_pr) else float("nan")
        az_finite = np.isfinite(com_az_sub[k])
        rg_finite = np.isfinite(com_rg[k])
        az_pp = (float(com_az_sub[k, az_finite].max() - com_az_sub[k, az_finite].min())
                 if az_finite.any() else float("nan"))
        rg_pp = (float(com_rg[k, rg_finite].max() - com_rg[k, rg_finite].min())
                 if rg_finite.any() else float("nan"))
        stats.append(BoxStats(
            y_c=int(y_c), x_c=int(x_c), h=int(h), w=int(w),
            slope_per_row_rad=slope_pr,
            slope_total_rad=slope_tot,
            intercept_rad=float(box_intercept[k]),
            residual_rad=float(box_residual[k]),
            n_rows=n_rows,
            det_count=det_in_box,
            v_az_mps=float(v_az[k]),
            v_rg_mps=float(v_rg[k]),
            az_pp_sub_px=az_pp,
            rg_pp_d_px=rg_pp,
        ))
    return SweepCache(boxes_post_merge=boxes_yxhw, stats=stats)


# ---------------------------------------------------------------------------
# Filter predicates — apply exactly the conditions of shear_averaging.main()
# ---------------------------------------------------------------------------
def _passes(
    bs: BoxStats,
    *,
    max_phase_residual_rad: float,
    residual_skip_det_frac: float,
    min_slope_deg: float,
    x_c_min: float, x_c_max: float,
    min_velocity_mps: float,
    slope_rad_thresh: float,
    max_w_px: int = MAX_SIZE_RG_BIN,
    strong_only: bool = False,
    motion_mode: str = "or",          # "or" (current code) or "and"
    min_h_px: int = 0,                # azimuth-height floor (proposed filter)
    max_w_px_strict: int | None = None,   # tighter range-width cap
    min_v_az_mps: float = 0.0,        # hard |v_az| floor (proposed filter)
) -> tuple[bool, str]:
    """Return (kept, first_failing_filter).

    If ``strong_only`` is True, also require ``|slope_total_rad| >= slope_rad_thresh``
    (the "strong" criterion that draws the red overlays in Figure 1, i.e.
    Case 2 of the motion filter on its own).
    """
    # S1: range-extent cap
    cap = max_w_px_strict if max_w_px_strict is not None else max_w_px
    if bs.w > cap:
        return False, "S1_range_extent_cap"
    # S1b: azimuth-height floor (proposed scene-independent filter)
    if bs.h < min_h_px:
        return False, "S1b_min_h"
    # S1c: hard |v_az| floor (proposed scene-independent filter)
    if min_v_az_mps > 0:
        v_az_abs = abs(bs.v_az_mps) if np.isfinite(bs.v_az_mps) else 0.0
        if v_az_abs < min_v_az_mps:
            return False, "S1c_min_v_az"
    # S2: min-slope (per-row degrees)
    if min_slope_deg > 0:
        slope_deg = abs(np.degrees(bs.slope_per_row_rad)) if np.isfinite(bs.slope_per_row_rad) else 0.0
        if slope_deg < min_slope_deg:
            return False, "S2_min_slope_deg"
    # S3: residual filter (with detection-count bypass)
    if max_phase_residual_rad > 0:
        bypass = bs.det_count > residual_skip_det_frac * max(1, bs.n_rows)
        if not bypass:
            if not (np.isfinite(bs.residual_rad)
                    and bs.residual_rad <= max_phase_residual_rad):
                return False, "S3_max_phase_residual_rad"
    # S4: range band
    if not (x_c_min <= bs.x_c <= x_c_max):
        return False, "S4_x_c_band"
    # S6: motion filter
    moving_com = (
        (np.isfinite(bs.v_az_mps) and abs(bs.v_az_mps) >= min_velocity_mps)
        or (np.isfinite(bs.v_rg_mps) and abs(bs.v_rg_mps) >= min_velocity_mps)
    )
    moving_slope = (
        np.isfinite(bs.slope_total_rad)
        and abs(bs.slope_total_rad) >= slope_rad_thresh
    )
    if strong_only:
        # Only Case 2 (slope) counts towards the "strong" set — same condition
        # that drives the red overlay in figure 1 of shear_averaging.py.
        if not moving_slope:
            return False, "S6_strong_slope"
        # In strong_only, optionally also require the COM velocity branch
        # (turning the OR into an AND on the strong set).
        if motion_mode == "and" and not moving_com:
            return False, "S6_strong_and_velocity"
    else:
        if motion_mode == "and":
            if not (moving_com and moving_slope):
                return False, "S6_motion_filter"
        else:
            if not (moving_com or moving_slope):
                return False, "S6_motion_filter"
    return True, "kept"


def classify(x_c: float) -> str:
    return "TP" if 70 <= x_c < 120 else "FA"


def survivors(cache: SweepCache, **params) -> list[tuple[int, BoxStats, bool]]:
    out = []
    for i, bs in enumerate(cache.stats):
        ok, _why = _passes(bs, **params)
        out.append((i, bs, ok))
    return out


def count_tp_fa(cache: SweepCache, **params) -> tuple[int, int, list[int]]:
    tp = fa = 0
    kept_idx = []
    for i, bs, ok in survivors(cache, **params):
        if not ok:
            continue
        kept_idx.append(i)
        if classify(bs.x_c) == "TP":
            tp += 1
        else:
            fa += 1
    return tp, fa, kept_idx


BASELINE = dict(
    max_phase_residual_rad=1.5,
    residual_skip_det_frac=0.25,
    min_slope_deg=0.0,
    x_c_min=-1e9, x_c_max=1e9,
    min_velocity_mps=2.0,
    slope_rad_thresh=1.0,
    strong_only=True,                          # focus on the red-overlay set
)


def fmt_box_row(i: int, bs: BoxStats, cls: str) -> str:
    return (f"{i:2d} {cls} y={bs.y_c:>5d} x={bs.x_c:>3d} h={bs.h:>4d} w={bs.w:>2d}  "
            f"|slope_tot|={abs(bs.slope_total_rad):5.2f} "
            f"res={bs.residual_rad:4.2f} "
            f"|v_az|={abs(bs.v_az_mps):6.2f} |v_rg|={abs(bs.v_rg_mps):5.2f} "
            f"detN/h={bs.det_count}/{bs.n_rows} "
            f"(det/h_box={bs.det_count/max(1,bs.n_rows):.2f})")


def main():
    patch = Path("/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_143542_433755.npy")
    print(f"Loading {patch.name} …")
    s = np.load(patch)
    print(f"  shape={s.shape}")

    print("Running scene up to merge_overlapping_boxes (n_sub=8, cov_th_mult=1.5) …")
    cache = compute_scene_to_post_merge(s, n_subaperture=8, cov_th_mult=1.5)
    K = len(cache.stats)
    print(f"  post-merge boxes: {K}")

    # ----------------------------------------------------------------------
    # 1. Baseline TP/FA at user's command
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("BASELINE (user's command)")
    print("=" * 78)
    print(f"  params: {BASELINE}")
    tp, fa, kept = count_tp_fa(cache, **BASELINE)
    print(f"  surviving: {tp + fa}   TP={tp}   FA={fa}")
    print("\nBaseline surviving boxes (TP = inside [70,120) range columns):")
    for i in kept:
        bs = cache.stats[i]
        print("  " + fmt_box_row(i, bs, classify(bs.x_c)))

    # ----------------------------------------------------------------------
    # 2. Per-box: which filter (and threshold) would kill each baseline FA?
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PER-TARGET SENSITIVITY (baseline-surviving STRONG boxes only)")
    print("=" * 78)
    print(" For each kept box, the *smallest tightening* of each filter that")
    print(" would drop this box, holding the other params at baseline.")
    print(f"   res-skip-det-frac is set to a huge number to disable the bypass")
    print(f"   so we can compare boxes on raw residual.")
    print()
    hdr = (f"{'idx':>3s} {'cls':>3s}  {'x_c':>5s}  "
           f"{'kill_residual':>14s}  {'kill_slope_deg':>15s}  "
           f"{'kill_v_min(mps)':>15s}  {'kill_slope_rad':>14s}  "
           f"{'x_band_kill':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for i in kept:
        bs = cache.stats[i]
        cls = classify(bs.x_c)

        # smallest max-residual that drops this box (if det-bypass is OFF)
        if np.isfinite(bs.residual_rad):
            kill_res = f"<{bs.residual_rad:.3f}"
        else:
            kill_res = "n/a"
        # if bypass is active at baseline frac=0.25, residual filter won't fire
        det_per_h = bs.det_count / max(1, bs.n_rows)
        if det_per_h > BASELINE["residual_skip_det_frac"]:
            kill_res += "*"          # bypass active — must also raise skip_frac
        # smallest min-slope-deg that drops this box (S2)
        slope_deg = abs(np.degrees(bs.slope_per_row_rad)) if np.isfinite(bs.slope_per_row_rad) else 0.0
        kill_slope_deg = f">{slope_deg:.3f}"
        # smallest v_min that drops Case-1 motion (largest absolute COM velocity)
        v_largest = max(
            abs(bs.v_az_mps) if np.isfinite(bs.v_az_mps) else 0.0,
            abs(bs.v_rg_mps) if np.isfinite(bs.v_rg_mps) else 0.0,
        )
        kill_v = f">{v_largest:.3f}"
        # smallest slope_rad_thresh that drops Case-2 motion
        s_tot = abs(bs.slope_total_rad) if np.isfinite(bs.slope_total_rad) else 0.0
        kill_st = f">{s_tot:.3f}"
        # range-band edge that would drop this box
        if bs.x_c < 70:
            x_band_kill = f"min>{bs.x_c}"
        elif bs.x_c >= 120:
            x_band_kill = f"max<{bs.x_c}"
        else:
            x_band_kill = "(in TP)"
        print(f"{i:>3d} {cls:>3s}  {bs.x_c:>5d}  "
              f"{kill_res:>14s}  {kill_slope_deg:>15s}  "
              f"{kill_v:>15s}  {kill_st:>14s}  {x_band_kill:>11s}")

    print("\n  Reading: '<R' means 'set --max-phase-residual-rad below R' to kill this box.")
    print("           '>X' means 'set the threshold above X' to kill this box.")
    print("           A trailing '*' on residual means the box is bypassed at baseline frac=0.25;")
    print("           must also raise --residual-skip-det-frac above det/h_box to enable the residual cut.")

    # ----------------------------------------------------------------------
    # 3. One-parameter sweeps (everything else at baseline)
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("ONE-PARAMETER SWEEPS  (all other params held at baseline)")
    print("=" * 78)

    def sweep_one(name: str, key: str, values):
        print(f"\n--- sweep {name}  ({key}) ---")
        print(f"  {'value':>14s}  {'kept':>4s}  {'TP':>3s}  {'FA':>3s}  {'lost-TP-idx':<22s}  added-FA-idx")
        params0 = dict(BASELINE)
        tp0, fa0, kept0 = count_tp_fa(cache, **params0)
        for v in values:
            params = dict(BASELINE)
            params[key] = v
            tp, fa, kept = count_tp_fa(cache, **params)
            lost_tp = sorted(set(kept0) - set(kept))
            lost_tp = [i for i in lost_tp if classify(cache.stats[i].x_c) == "TP"]
            gained_fa = sorted(set(kept) - set(kept0))
            gained_fa = [i for i in gained_fa if classify(cache.stats[i].x_c) == "FA"]
            print(f"  {str(v):>14s}  {tp+fa:>4d}  {tp:>3d}  {fa:>3d}  "
                  f"{str(lost_tp):<22s}  {gained_fa}")

    sweep_one("max-phase-residual-rad", "max_phase_residual_rad",
              [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0, 3.14])
    sweep_one("residual-skip-det-frac", "residual_skip_det_frac",
              [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0,
               1.5, 2.0, 3.0, 5.0, 10.0, 1e9])
    sweep_one("min-slope-deg", "min_slope_deg",
              [0.0, 0.005, 0.01, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3,
               0.5, 1.0])
    sweep_one("slope-rad-thresh", "slope_rad_thresh",
              [0.5, 0.8, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5])
    sweep_one("min-velocity-mps  (only relevant if strong_only=False)",
              "min_velocity_mps",
              [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])
    sweep_one("x-c-min", "x_c_min",
              [-1e9, 0, 40, 50, 60, 65, 70])
    sweep_one("x-c-max", "x_c_max",
              [120, 125, 130, 140, 150, 175, 200, 1e9])

    # ----------------------------------------------------------------------
    # 4. Joint sweep WITHOUT the x-c-band: which combos retain all TPs?
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("JOINT SWEEP — strong-only, no x-band; keep all baseline TPs")
    print("=" * 78)
    tp0 = sum(1 for i in kept if classify(cache.stats[i].x_c) == "TP")
    print(f"  baseline strong TP = {tp0};  searching for combos that retain all {tp0}.")
    grid = []
    for s_th in [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
        for r_max in [3.14, 2.0, 1.5, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6]:
            for f_skip in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 1e9]:
                for ms_deg in [0.0, 0.005, 0.01, 0.02, 0.05, 0.08]:
                    params = dict(BASELINE)
                    params.update(
                        slope_rad_thresh=s_th,
                        max_phase_residual_rad=r_max,
                        residual_skip_det_frac=f_skip,
                        min_slope_deg=ms_deg,
                    )
                    tp, fa, _ = count_tp_fa(cache, **params)
                    if tp == tp0:
                        grid.append((fa, tp, s_th, r_max, f_skip, ms_deg))
    grid.sort()
    print(f"  {'FA':>3s} {'TP':>3s}  {'s_thr':>6s} {'res_max':>7s} {'res_skip':>8s} {'min_slope_deg':>13s}")
    for fa, tp, st, rm, fs, ms in grid[:15]:
        print(f"  {fa:>3d} {tp:>3d}  {st:>6.2f} {rm:>7.2f} {fs:>8.2g} {ms:>13.3f}")
    print(f"  … {len(grid)} total combos retain all {tp0} TPs")

    # ----------------------------------------------------------------------
    # 5. Joint sweep ADDING the x-c band
    # ----------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("JOINT SWEEP — strong-only, x-band IN; keep all baseline TPs")
    print("=" * 78)
    grid2 = []
    for s_th in [1.0, 1.5, 2.0, 3.0]:
        for r_max in [3.14, 1.5, 1.0]:
            for f_skip in [0.25, 1.0, 1e9]:
                for ms_deg in [0.0, 0.05]:
                    for xlo, xhi in [(-1e9, 1e9), (40, 1e9), (60, 1e9), (-1e9, 130),
                                     (-1e9, 200), (60, 130), (60, 200), (40, 200),
                                     (40, 130), (65, 125), (70, 120)]:
                        params = dict(BASELINE)
                        params.update(
                            slope_rad_thresh=s_th,
                            max_phase_residual_rad=r_max,
                            residual_skip_det_frac=f_skip,
                            min_slope_deg=ms_deg,
                            x_c_min=xlo, x_c_max=xhi,
                        )
                        tp, fa, _ = count_tp_fa(cache, **params)
                        if tp == tp0:
                            grid2.append((fa, tp, s_th, r_max, f_skip, ms_deg, xlo, xhi))
    grid2.sort()
    print(f"  {'FA':>3s} {'TP':>3s}  {'s_thr':>5s} {'r_max':>5s} {'r_skip':>6s} {'msdeg':>5s} {'xlo':>6s} {'xhi':>6s}")
    for row in grid2[:20]:
        fa, tp, st, rm, fs, ms, xl, xh = row
        print(f"  {fa:>3d} {tp:>3d}  {st:>5.2f} {rm:>5.2f} {fs:>6.2g} {ms:>5.3f} {xl:>6.0f} {xh:>6.0f}")
    print(f"  … {len(grid2)} total combos retain all {tp0} TPs")


if __name__ == "__main__":
    main()
