"""Scene-independent parameter sweep: which features can suppress strong-slope
FAs without using x_c (a scene-specific gate)?

Findings from the per-feature audit (see report):
  - |v_az| (COM-velocity in azimuth) gives a *clean* separation on this scene:
      TPs   ∈ [27.1, 148.9] m/s
      FAs   ∈ [ 0.49,   6.03] m/s
  - All other features (slope, residual, h, w, det-density) overlap.

The current motion filter is:
      is_moving = (|v_az| >= v_min OR |v_rg| >= v_min)  OR  (|slope·n| >= s_th)
so strong-slope FAs slip through regardless of |v_az|. Two proposed fixes
(both scene-independent):

  (A) Add a hard "|v_az| >= V_floor" filter applied to every box upstream
      of the motion gate. (--min-v-az-mps, exposed in this sweep.)
  (B) Switch the motion filter to AND so a box must satisfy both the slope
      and the velocity branch. (--motion-mode=and.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parameter_sweep import (
    BASELINE, classify, compute_scene_to_post_merge, count_tp_fa,
    fmt_box_row,
)


def main():
    patch = Path("/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_143542_433755.npy")
    s = np.load(patch)
    cache = compute_scene_to_post_merge(s, n_subaperture=8, cov_th_mult=1.5)

    # Strict baseline = same as before, strong-only.
    base = dict(BASELINE)
    tp0, fa0, kept0 = count_tp_fa(cache, **base)
    print(f"Baseline (strong-only, OR motion):   TP={tp0}  FA={fa0}\n")

    # ----------------------------------------------------------------
    # 1. Sweep |v_az| floor alone (other params at baseline, strong-only)
    # ----------------------------------------------------------------
    print("=" * 78)
    print("Sweep --min-v-az-mps (proposed hard |v_az| floor, scene-independent)")
    print("=" * 78)
    print(f"  {'min_v_az':>10s}  {'kept':>4s}  {'TP':>3s}  {'FA':>3s}  "
          f"{'lost-TP':<10s}  surviving-FA-idx")
    for v in [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 6.05, 7.0, 8.0, 10.0, 15.0, 20.0, 25.0, 27.0, 27.5]:
        p = dict(base); p["min_v_az_mps"] = v
        tp, fa, kept = count_tp_fa(cache, **p)
        lost_tp = sorted(set(kept0) - set(kept))
        lost_tp = [i for i in lost_tp if classify(cache.stats[i].x_c) == "TP"]
        fa_idx = [i for i in kept if classify(cache.stats[i].x_c) == "FA"]
        print(f"  {v:>10.2f}  {tp+fa:>4d}  {tp:>3d}  {fa:>3d}  "
              f"{str(lost_tp):<10s}  {fa_idx}")

    # ----------------------------------------------------------------
    # 2. Sweep motion-mode = AND (strong-only) with sweeping min_velocity_mps
    # ----------------------------------------------------------------
    print()
    print("=" * 78)
    print("Sweep --motion-mode=and  with min_velocity_mps  (strong & velocity)")
    print("=" * 78)
    print(f"  {'v_min':>6s}  {'kept':>4s}  {'TP':>3s}  {'FA':>3s}  "
          f"{'lost-TP':<10s}  surviving-FA-idx")
    for v in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.05, 7.0, 10.0, 15.0, 20.0, 25.0, 27.0, 27.5]:
        p = dict(base); p["motion_mode"] = "and"; p["min_velocity_mps"] = v
        tp, fa, kept = count_tp_fa(cache, **p)
        lost_tp = sorted(set(kept0) - set(kept))
        lost_tp = [i for i in lost_tp if classify(cache.stats[i].x_c) == "TP"]
        fa_idx = [i for i in kept if classify(cache.stats[i].x_c) == "FA"]
        print(f"  {v:>6.2f}  {tp+fa:>4d}  {tp:>3d}  {fa:>3d}  "
              f"{str(lost_tp):<10s}  {fa_idx}")

    # ----------------------------------------------------------------
    # 3. Sweep min azimuth height (--min-h-px)
    # ----------------------------------------------------------------
    print()
    print("=" * 78)
    print("Sweep --min-h-px (proposed azimuth-height floor)")
    print("=" * 78)
    print(f"  {'min_h':>7s}  {'kept':>4s}  {'TP':>3s}  {'FA':>3s}  "
          f"{'lost-TP':<10s}  surviving-FA-idx")
    for h in [0, 100, 200, 300, 500, 700, 750, 800, 850, 890, 900, 1000, 1200, 1500, 1700, 2000]:
        p = dict(base); p["min_h_px"] = h
        tp, fa, kept = count_tp_fa(cache, **p)
        lost_tp = sorted(set(kept0) - set(kept))
        lost_tp = [i for i in lost_tp if classify(cache.stats[i].x_c) == "TP"]
        fa_idx = [i for i in kept if classify(cache.stats[i].x_c) == "FA"]
        print(f"  {h:>7d}  {tp+fa:>4d}  {tp:>3d}  {fa:>3d}  "
              f"{str(lost_tp):<10s}  {fa_idx}")

    # ----------------------------------------------------------------
    # 4. Joint sweep: tighten the proposed knobs together, no x-band
    # ----------------------------------------------------------------
    print()
    print("=" * 78)
    print("Best combos (strong-only, NO x-band, retains all TPs)")
    print("=" * 78)
    grid = []
    for v_az in [0.0, 2.0, 6.5, 8.0, 10.0]:
        for mh in [0, 500, 800, 890, 1000]:
            for s_th in [1.0, 1.25, 1.5]:
                for mw in [MAX_W := 43, 30, 20, 15, 14]:
                    p = dict(base)
                    p["min_v_az_mps"] = v_az
                    p["min_h_px"]     = mh
                    p["slope_rad_thresh"] = s_th
                    p["max_w_px_strict"] = mw
                    tp, fa, _ = count_tp_fa(cache, **p)
                    if tp == tp0:
                        grid.append((fa, tp, v_az, mh, s_th, mw))
    grid.sort()
    print(f"  {'FA':>2s} {'TP':>2s}  {'min_v_az':>8s} {'min_h':>5s} {'s_th':>5s} {'max_w':>5s}")
    seen = set()
    for fa, tp, v, mh, st, mw in grid[:25]:
        print(f"  {fa:>2d} {tp:>2d}  {v:>8.1f} {mh:>5d} {st:>5.2f} {mw:>5d}")
        seen.add((fa, tp))

    # ----------------------------------------------------------------
    # 5. Per-target sensitivity to proposed filters
    # ----------------------------------------------------------------
    print()
    print("=" * 78)
    print("Per-target proposed-filter kill thresholds (strong-only set)")
    print("=" * 78)
    print(f"{'idx':>3s} {'cls':>3s} {'x_c':>4s} {'h':>5s} {'w':>3s} "
          f"{'|v_az|':>7s} {'|slope·n|':>10s}  "
          f"{'min_v_az_kills':>15s} {'min_h_kills':>11s} {'max_w_kills':>11s}")
    for i in kept0:
        bs = cache.stats[i]
        cls = classify(bs.x_c)
        v_az_abs = abs(bs.v_az_mps) if np.isfinite(bs.v_az_mps) else 0.0
        s_tot = abs(bs.slope_total_rad) if np.isfinite(bs.slope_total_rad) else 0.0
        kv = f">{v_az_abs:.2f}"
        kh = f">{bs.h}"
        kw = f"<{bs.w}"
        print(f"{i:>3d} {cls:>3s} {bs.x_c:>4d} {bs.h:>5d} {bs.w:>3d} "
              f"{v_az_abs:>7.2f} {s_tot:>10.2f}  "
              f"{kv:>15s} {kh:>11s} {kw:>11s}")


if __name__ == "__main__":
    main()
