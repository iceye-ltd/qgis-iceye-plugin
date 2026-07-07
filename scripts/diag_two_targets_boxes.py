"""Diagnose why two neighbouring targets around rows 6100:6600, cols ~590:615
are not one-box-per-target under the current view_subaps_cov settings.

This script *does not* change any behaviour: it replays the same pipeline
as `scripts/view_subaps_cov.py` (CoV mask → isolated-pixel filter → seed
peaks → cluster_peaks → CFAR) on the same npz, then prints every peak,
every cluster and every box that overlaps the region and renders a
zoom-in with the two operator-declared target rectangles overlaid on the
mask and on the sub_mean amplitude image.

Target 1: rows 6100:6350, cols 600:615
Target 2: rows 6350:6600, cols 590:610
"""

from __future__ import annotations

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


NPZ = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/"
    "ICEYE_WTW3YQ_20250104T180444Z_4439676_X7_SLED_SLC_subapertures.npz"
)

# Operator-declared targets (row_lo, row_hi, col_lo, col_hi) inclusive-lo,
# exclusive-hi, i.e. the same convention as `arr[y_lo:y_hi, x_lo:x_hi]`.
TARGETS = [
    ("T1", 6100, 6350, 600, 615),
    ("T2", 6350, 6600, 590, 610),
]

# Zoom-in window drawn generously around both targets so we also see any
# neighbouring peaks / clusters that might have leaked in.
ZOOM_Y_LO, ZOOM_Y_HI = 5950, 6750
ZOOM_X_LO, ZOOM_X_HI = 560, 650

# Same knob values view_subaps_cov.py uses by default.
BRIGHT_MODE = "off"
COV_TH_MULT = 2                # NOTE: view_subaps_cov overrides the CLI --cov-th-mult with 2
FILTER_TARGET_SIZE_M = 3.0
FILTER_MIN_DENSITY = 0.30
SEED_DENSITY_TH = 0.5
N_TARGET_AZ_WIDTH = 100         # full-res s_degraded rows
N_TARGET_RG_LEN = 3
AZ_SEARCH_M = 500.0
RG_SEARCH_M = 30.0
MAX_PEAK_GAP_AZ_M = float("inf")
ENABLE_ARC_SPLIT = False
CFAR_SNR_TH = 0.0   # DISABLED — matches view_subaps_cov's new default
CFAR_RANGE_BINS = 2


def _overlaps(box, y_lo, y_hi, x_lo, x_hi) -> bool:
    """box = (y_lo, y_hi, x_lo, x_hi) inclusive-hi (as returned by _build_boxes)."""
    by_lo, by_hi, bx_lo, bx_hi = box
    return not (by_hi < y_lo or by_lo >= y_hi or bx_hi < x_lo or bx_lo >= x_hi)


def main() -> None:
    print(f"Loading {NPZ.name}")
    with np.load(NPZ, allow_pickle=False) as z:
        sub_mean = z["sub_mean"].astype(np.float64)
        sub_var = z["sub_var"].astype(np.float64)
        n_sub = int(z["N_subaperture"])
        az_m = float(z["az_m_per_px_s_degraded"])
        rg_m = float(z["rg_m_per_px_s_degraded"])

    H, W = sub_mean.shape
    az_m_dec = az_m * n_sub
    rg_m_dec = rg_m
    print(f"  shape = {sub_mean.shape}  ·  az_m_dec = {az_m_dec:.3f}  ·  rg_m_dec = {rg_m_dec:.3f}")

    # ---- CoV mask (mode='off') ---------------------------------------
    cov_sq = sub_var / (sub_mean ** 2 + 1e-12)
    th_off = COV_TH_MULT * float(np.median(cov_sq))
    mask_dec = (cov_sq > th_off).astype(np.float32)
    print(f"  CoV th = {th_off:.3g}  ·  raw mask kept = {int(mask_dec.sum())}/{mask_dec.size}")

    # ---- isolated-pixel filter ---------------------------------------
    az_filt_win = max(1, int(round(FILTER_TARGET_SIZE_M / az_m_dec)))
    rg_filt_win = max(1, int(round(FILTER_TARGET_SIZE_M / rg_m_dec)))
    density = uniform_filter(mask_dec, size=(az_filt_win, rg_filt_win), mode="constant")
    mask_filt = (mask_dec.astype(bool) & (density >= FILTER_MIN_DENSITY)).astype(np.float32)
    print(
        f"  filter win = {az_filt_win} az × {rg_filt_win} rg px "
        f"(density ≥ {FILTER_MIN_DENSITY:.2f})  ·  after filter = {int(mask_filt.sum())}"
    )

    # ---- seed peaks --------------------------------------------------
    az_seed_win = max(1, int(round(N_TARGET_AZ_WIDTH / n_sub)))
    rg_seed_win = int(N_TARGET_RG_LEN)
    box_size = (az_seed_win, rg_seed_win)
    box_area = az_seed_win * rg_seed_win
    print(f"  seed window (az_seed_win, rg_seed_win) = {box_size}  (area={box_area})")

    amp_proxy = sub_mean.astype(np.float32)
    det_count = uniform_filter(mask_filt, size=box_size, mode="constant") * box_area
    amp_max = maximum_filter(amp_proxy, size=box_size, mode="constant")
    is_detection = mask_filt == 1
    is_dense = det_count > SEED_DENSITY_TH * box_area
    is_peak = amp_proxy == amp_max
    boundary = is_detection & is_dense & is_peak
    peaks_yx = np.argwhere(boundary).astype(np.int64)
    print(f"  seed peaks (scene-wide) = {len(peaks_yx)}")

    # ---- clustering --------------------------------------------------
    arc_az_thr_m = 200.0 if ENABLE_ARC_SPLIT else float("inf")
    t0 = time.time()
    labels, boxes = cluster_peaks(
        mask_filt.astype(np.uint8), peaks_yx,
        az_m_per_px=az_m_dec, rg_m_per_px=rg_m_dec,
        az_search_m=AZ_SEARCH_M, rg_search_m=RG_SEARCH_M,
        arc_az_thr_m=arc_az_thr_m, arc_rg_max_m=20.0,
        max_peak_gap_az_m=MAX_PEAK_GAP_AZ_M,
        enforce_max_size=False,
    )
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    print(f"  cluster_peaks → {n_clusters} clusters in {time.time() - t0:.2f} s")

    # ---- CFAR filter (same as view_subaps_cov) ------------------------
    kept = np.ones(len(boxes), dtype=bool)
    if CFAR_SNR_TH > 0 and len(boxes):
        for c in range(len(boxes)):
            y_lo, y_hi, x_lo, x_hi = (int(v) for v in boxes[c])
            box_amp = amp_proxy[y_lo:y_hi + 1, x_lo:x_hi + 1]
            box_msk = mask_filt[y_lo:y_hi + 1, x_lo:x_hi + 1]
            sig_pix = box_amp[box_msk == 1]
            if sig_pix.size == 0:
                sig_pix = box_amp
                if sig_pix.size == 0:
                    kept[c] = False
                    continue
            signal = float(sig_pix.mean())
            n_guard = max(1, CFAR_RANGE_BINS)
            prev_lo = max(0, x_lo - n_guard)
            succ_hi = min(W, x_hi + 1 + n_guard)
            prev_strip = amp_proxy[y_lo:y_hi + 1, prev_lo:x_lo]
            succ_strip = amp_proxy[y_lo:y_hi + 1, x_hi + 1:succ_hi]
            noise_means = []
            if prev_strip.size:
                noise_means.append(float(prev_strip.mean()))
            if succ_strip.size:
                noise_means.append(float(succ_strip.mean()))
            if not noise_means:
                continue
            noise = min(noise_means)
            snr = signal / max(noise, 1e-12)
            if snr < CFAR_SNR_TH:
                kept[c] = False
        print(f"  CFAR kept {int(kept.sum())}/{len(boxes)} clusters (snr_th = {CFAR_SNR_TH:g})")

    # Two views: full pre-CFAR result and post-CFAR result.
    views = {
        "pre-CFAR": (boxes.copy(), labels.copy(), peaks_yx.copy(),
                     np.ones(len(boxes), dtype=bool)),
        "post-CFAR": (boxes[kept], None, None, None),
    }

    # Rebuild the CFAR-kept peaks/labels the same way view_subaps_cov does.
    keep_pk = kept[labels]
    peaks_yx_post = peaks_yx[keep_pk]
    _, labels_post = np.unique(labels[keep_pk], return_inverse=True)
    labels_post = labels_post.astype(np.int64)
    views["post-CFAR"] = (boxes[kept], labels_post, peaks_yx_post,
                          np.ones(int(kept.sum()), dtype=bool))

    # For rendering later — the operator sees the post-CFAR figure.
    boxes_kept = boxes[kept]
    peaks_yx_kept = peaks_yx_post
    labels_kept = labels_post
    n_clusters_kept = int(labels_kept.max()) + 1 if len(labels_kept) else 0

    # ============================================================== #
    # Region diagnostics — for both pre-CFAR and post-CFAR
    # ============================================================== #
    for view_name, (v_boxes, v_labels, v_peaks, _) in views.items():
        print(f"\n=========== {view_name} view ===========")
        n_v = int(v_labels.max()) + 1 if v_labels is not None and len(v_labels) else 0
        print(f"  total clusters: {n_v}")

        if v_labels is None:
            continue

        print(f"\n  --- Peaks inside targets ({view_name}) ---")
        for name, y_lo, y_hi, x_lo, x_hi in TARGETS:
            sel = (
                (v_peaks[:, 0] >= y_lo) & (v_peaks[:, 0] < y_hi)
                & (v_peaks[:, 1] >= x_lo) & (v_peaks[:, 1] < x_hi)
            )
            print(f"  {name}  rows {y_lo}:{y_hi}  cols {x_lo}:{x_hi}  peaks = {int(sel.sum())}")
            if sel.any():
                for p, lab in zip(v_peaks[sel], v_labels[sel]):
                    print(f"      peak (y={int(p[0])}, x={int(p[1])})  →  cluster {int(lab)}")

        print(f"\n  --- All peaks in zoom window ({view_name}) ---")
        zoom = (
            (v_peaks[:, 0] >= ZOOM_Y_LO) & (v_peaks[:, 0] < ZOOM_Y_HI)
            & (v_peaks[:, 1] >= ZOOM_X_LO) & (v_peaks[:, 1] < ZOOM_X_HI)
        )
        n_in_zoom = int(zoom.sum())
        print(f"  {n_in_zoom} peaks in zoom window")
        if n_in_zoom and n_in_zoom <= 200:
            for p, lab in zip(v_peaks[zoom], v_labels[zoom]):
                print(f"    peak (y={int(p[0])}, x={int(p[1])})  →  cluster {int(lab)}")

        print(f"\n  --- Cluster boxes overlapping zoom window ({view_name}) ---")
        v_overlap_cids = []
        for c in range(n_v):
            by_lo, by_hi, bx_lo, bx_hi = (int(v) for v in v_boxes[c])
            if not (by_hi < ZOOM_Y_LO or by_lo > ZOOM_Y_HI
                    or bx_hi < ZOOM_X_LO or bx_lo > ZOOM_X_HI):
                v_overlap_cids.append(c)
                n_peaks = int((v_labels == c).sum())
                print(
                    f"    cluster {c:>3d}  box (y {by_lo:>5d}..{by_hi:>5d}, "
                    f"x {bx_lo:>4d}..{bx_hi:>4d})  "
                    f"az_ext = {by_hi - by_lo + 1:>4d} px = {(by_hi - by_lo + 1) * az_m_dec:>6.1f} m  "
                    f"rg_ext = {bx_hi - bx_lo + 1:>3d} px = {(bx_hi - bx_lo + 1) * rg_m_dec:>5.1f} m  "
                    f"n_peaks = {n_peaks}"
                )
        print(f"  (total: {len(v_overlap_cids)} clusters overlapping zoom)")

    # ---- Azimuth mask-density profile through the target columns ------
    # This is what per-peak `_contiguous_run` sees on cluster 48's peaks.
    print("\n=== Azimuth mask-density profile through cols 600..615 ===")
    from scipy.ndimage import gaussian_filter1d
    prof_lo, prof_hi = 5900, 6650
    slab = mask_filt[prof_lo:prof_hi, 600:616]
    az_profile = slab.mean(axis=1)            # in [0, 1]
    az_profile_sm = gaussian_filter1d(az_profile, sigma=10.0)
    # Also with a smaller sigma to see the "true" gap.
    az_profile_sm3 = gaussian_filter1d(az_profile, sigma=3.0)
    # Print thresholds & profile at a few key rows.
    key_rows = [6100, 6150, 6200, 6227, 6293, 6300, 6320, 6340, 6350,
                6360, 6380, 6399, 6450, 6501, 6550, 6600]
    print(f"{'row':>6}  {'raw':>5}  {'sm(σ=10)':>9}  {'sm(σ=3)':>8}")
    for r in key_rows:
        i = r - prof_lo
        if 0 <= i < len(az_profile):
            print(
                f"{r:>6d}  {az_profile[i]:>5.2f}  "
                f"{az_profile_sm[i]:>9.3f}  {az_profile_sm3[i]:>8.3f}"
            )
    # Threshold used by clustering.py: 0.15 × profile_at_peak.
    print(f"  peak of σ=10 profile in c48 rows (6399..6501): "
          f"{az_profile_sm[6399-prof_lo:6501-prof_lo+1].max():.3f}  "
          f"→ 0.15·peak = {0.15*az_profile_sm[6399-prof_lo:6501-prof_lo+1].max():.3f}")
    print(f"  peak of σ=10 profile in c64 rows (6227..6293): "
          f"{az_profile_sm[6227-prof_lo:6293-prof_lo+1].max():.3f}  "
          f"→ 0.15·peak = {0.15*az_profile_sm[6227-prof_lo:6293-prof_lo+1].max():.3f}")

    # ---- Per-target overlap for pre-CFAR clusters ---------------------
    print("\n=== Per-target overlap (pre-CFAR clusters overlapping zoom) ===")
    pre_overlap = []
    for c in range(int(labels.max()) + 1 if len(labels) else 0):
        by_lo, by_hi, bx_lo, bx_hi = (int(v) for v in boxes[c])
        if not (by_hi < ZOOM_Y_LO or by_lo > ZOOM_Y_HI
                or bx_hi < ZOOM_X_LO or bx_lo > ZOOM_X_HI):
            pre_overlap.append(c)
    for name, y_lo, y_hi, x_lo, x_hi in TARGETS:
        t_area = (y_hi - y_lo) * (x_hi - x_lo)
        t_az = y_hi - y_lo
        print(f"\n{name}  rows {y_lo}:{y_hi}  cols {x_lo}:{x_hi}  area = {t_area}")
        for c in pre_overlap:
            by_lo, by_hi, bx_lo, bx_hi = (int(v) for v in boxes[c])
            iy_lo = max(y_lo, by_lo)
            iy_hi = min(y_hi, by_hi + 1)
            ix_lo = max(x_lo, bx_lo)
            ix_hi = min(x_hi, bx_hi + 1)
            if iy_hi > iy_lo and ix_hi > ix_lo:
                area = (iy_hi - iy_lo) * (ix_hi - ix_lo)
                b_area = (by_hi - by_lo + 1) * (bx_hi - bx_lo + 1)
                az_frac = (iy_hi - iy_lo) / t_az
                print(
                    f"    cluster {c:>3d}: intersect area={area:>5d}  "
                    f"({100 * area / t_area:5.1f}% of {name}, "
                    f"{100 * area / b_area:5.1f}% of box)  "
                    f"az-overlap {iy_lo}..{iy_hi} = {100 * az_frac:5.1f}% of T rows"
                )

    # ---- Why isn't c48 merged with c64? Trace linkage distance -------
    print("\n=== Linkage distance c48 vs c64 (pre-CFAR) ===")
    c48_mask = labels == 48
    c64_mask = labels == 64
    c48_peaks = peaks_yx[c48_mask]
    c64_peaks = peaks_yx[c64_mask]
    print(f"  c48: {len(c48_peaks)} peaks  y ∈ [{c48_peaks[:,0].min()}, {c48_peaks[:,0].max()}]  "
          f"x ∈ [{c48_peaks[:,1].min()}, {c48_peaks[:,1].max()}]")
    print(f"  c64: {len(c64_peaks)} peaks  y ∈ [{c64_peaks[:,0].min()}, {c64_peaks[:,0].max()}]  "
          f"x ∈ [{c64_peaks[:,1].min()}, {c64_peaks[:,1].max()}]")
    az_search_px_500 = 500.0 / az_m_dec
    rg_search_px_30 = 30.0 / rg_m_dec
    # Pairwise Chebyshev distance (normalised). Complete linkage merges
    # only if MAX pairwise distance ≤ 1.0.
    max_d = 0.0
    max_d_pair = None
    for p1 in c48_peaks:
        for p2 in c64_peaks:
            d_az = abs(p1[0] - p2[0]) / az_search_px_500
            d_rg = abs(p1[1] - p2[1]) / rg_search_px_30
            d = max(d_az, d_rg)
            if d > max_d:
                max_d = d
                max_d_pair = (tuple(p1), tuple(p2), d_az, d_rg)
    print(f"  MAX pairwise (Chebyshev normalised) distance: {max_d:.3f}")
    print(f"    pair: {max_d_pair[0]} ↔ {max_d_pair[1]}  "
          f"(d_az_norm={max_d_pair[2]:.3f}, d_rg_norm={max_d_pair[3]:.3f})")
    print(f"  Cut threshold t=1.0  →  should have been merged? "
          f"{'YES' if max_d <= 1.0 else 'NO'}")

    # ---- Isolate: cluster just c48+c64 peaks alone and see ------------
    from scipy.spatial.distance import squareform as _sqf
    from scipy.cluster.hierarchy import linkage as _link, fcluster as _fcl
    pair_peaks = np.vstack([c64_peaks, c48_peaks])
    K = len(pair_peaks)
    dy = np.abs(pair_peaks[:, 0:1] - pair_peaks[:, 0:1].T) / az_search_px_500
    dx = np.abs(pair_peaks[:, 1:2] - pair_peaks[:, 1:2].T) / rg_search_px_30
    D_iso = np.maximum(dy, dx)
    np.fill_diagonal(D_iso, 0.0)
    Z_iso = _link(_sqf(D_iso, checks=False), method="complete")
    fl = _fcl(Z_iso, t=1.0, criterion="distance") - 1
    print(f"\n  If only these 16 peaks were clustered alone: "
          f"→ {len(np.unique(fl))} cluster(s)")

    # ---- Also check c48+c64 with the FULL peak set, same params -------
    # (i.e. does the presence of neighbours c38, c45, c54, c971, etc.
    # change what happens for these two?)
    all_pk = peaks_yx
    K_all = len(all_pk)
    print(f"  Scene-wide: {K_all} peaks, {int(labels.max())+1} clusters")
    # Find cophenetic distance c48<->c64 in the scene-wide linkage.
    # Easier: check whether any peak has d(peak, c48) < 1.0 AND d(peak, c64) < 1.0
    # but pulls the merge above 1.0 due to complete-linkage.
    yc = all_pk[:, 0]; xc = all_pk[:, 1]
    d_az = np.abs(yc[:, None] - yc[None, :]) / az_search_px_500
    d_rg = np.abs(xc[:, None] - xc[None, :]) / rg_search_px_30
    D_full = np.maximum(d_az, d_rg)
    np.fill_diagonal(D_full, 0.0)
    # Add overlap gate for parity with clustering.py? Not needed since
    # we're computing D as-is here. But clustering.py sets D=inf when
    # per-peak intervals don't overlap — cannot easily reproduce here
    # without recomputing all K intervals; assume finite for this test.
    Z_full = _link(_sqf(D_full, checks=False), method="complete")
    fl_full = _fcl(Z_full, t=1.0, criterion="distance") - 1
    # Look at what cluster each c48 / c64 peak falls into in this
    # simplified full-scene linkage.
    idx_c48 = np.where(np.isin(np.arange(K_all), np.where(c48_mask)[0]))[0]
    idx_c64 = np.where(np.isin(np.arange(K_all), np.where(c64_mask)[0]))[0]
    labs_c48 = np.unique(fl_full[idx_c48])
    labs_c64 = np.unique(fl_full[idx_c64])
    print(f"  Scene-wide linkage (no overlap gate): "
          f"c48 peaks → labels {labs_c48}  ·  c64 peaks → labels {labs_c64}  "
          f"→ {'MERGED' if set(labs_c48) & set(labs_c64) else 'SEPARATE'}")

    # ---- Reproduce per-peak intervals for the boundary peaks ----------
    # Compute y_lo/y_hi as clustering.py does for the two peaks that
    # sit closest to the "gap": last c64 peak (y=6293) and first c48
    # peak (y=6399). If their intervals don't overlap the pair D goes
    # to inf and complete-linkage refuses the merge.
    def _pp_interval(yp, xp, mask_arr, rg_half, sigma, thr_frac, az_min_px_, az_max_px_):
        H = mask_arr.shape[0]
        W = mask_arr.shape[1]
        x0 = max(0, xp - rg_half)
        x1 = min(W, xp + rg_half + 1)
        prof = mask_arr[:, x0:x1].mean(axis=1)
        prof_sm = gaussian_filter1d(prof, sigma=sigma)
        thr = thr_frac * (prof_sm[yp] + 1e-9)
        above = (prof_sm >= thr).astype(np.int8)
        lo = yp
        while lo > 0 and above[lo - 1]:
            lo -= 1
        hi = yp
        while hi < H - 1 and above[hi + 1]:
            hi += 1
        half = max((hi - lo) / 2.0, az_min_px_ / 2.0)
        half = min(half, az_max_px_ / 2.0)
        return max(0, yp - half), min(H - 1, yp + half)
    rg_half_repro = int(np.ceil(30.0 / rg_m_dec))
    az_min_px_repro = 20.0 / az_m_dec
    az_max_px_repro = 200.0 / az_m_dec
    print(f"\n  Per-peak intervals (clustering.py STEP 1):")
    for (yp, xp), lab in [
        ((6293, 613), 64),
        ((6288, 605), 64),
        ((6399, 606), 48),
        ((6419, 605), 48),
    ]:
        lo, hi = _pp_interval(
            yp, xp, mask_filt, rg_half_repro, 10.0, 0.15,
            az_min_px_repro, az_max_px_repro,
        )
        print(f"    peak ({yp:>4d}, {xp:>3d})  c{lab}  →  y ∈ [{int(lo)}, {int(hi)}]")
    # Cross-pair overlap: all 60 c48-c64 pairs
    print(f"\n  Cross-cluster interval overlap for all 60 c48×c64 pairs:")
    n_fail = 0
    for pc48 in c48_peaks:
        for pc64 in c64_peaks:
            la, ha = _pp_interval(int(pc48[0]), int(pc48[1]), mask_filt,
                                  rg_half_repro, 10.0, 0.15,
                                  az_min_px_repro, az_max_px_repro)
            lb, hb = _pp_interval(int(pc64[0]), int(pc64[1]), mask_filt,
                                  rg_half_repro, 10.0, 0.15,
                                  az_min_px_repro, az_max_px_repro)
            a_ov = min(ha, hb) - max(la, lb)
            r_ov = 1  # (range overlap not needed to trigger — az alone triggers)
            if a_ov < 0:
                n_fail += 1
                print(
                    f"    c48 ({int(pc48[0])},{int(pc48[1])}) y∈[{int(la)},{int(ha)}]"
                    f"  vs  c64 ({int(pc64[0])},{int(pc64[1])}) y∈[{int(lb)},{int(hb)}]"
                    f"  →  az_overlap = {int(a_ov)}  ✗ D=inf"
                )
    print(f"  {n_fail}/60 pairs have az_overlap < 0 (any single failure blocks the merge)")

    # ---- Trace the actual box formation for c48 ----------------------
    # Reproduce clustering.py logic locally for the two peaks of interest.
    print("\n=== Box-formation trace for c48 ===")
    from scipy.ndimage import gaussian_filter1d as _gf1
    az_m_per_px = az_m_dec
    rg_m_per_px = rg_m_dec
    rg_search_px = 30.0 / rg_m_per_px       # view_subaps_cov default
    az_search_px = 500.0 / az_m_per_px
    az_min_px = 20.0 / az_m_per_px
    az_max_px = 200.0 / az_m_per_px
    profile_thr = 0.15
    sigma_az_px = 10.0
    rg_half = int(np.ceil(rg_search_px))
    print(
        f"  rg_search_px = {rg_search_px:.1f}  →  rg_half = {rg_half}  "
        f"(slab width = {2*rg_half+1} px = {(2*rg_half+1)*rg_m_per_px:.1f} m)"
    )

    # Just look at the top-most peak of c48 (y=6399, x=606).
    yp, xp = 6399, 606
    x0 = max(0, xp - rg_half)
    x1 = min(mask_filt.shape[1], xp + rg_half + 1)
    az_prof = mask_filt[:, x0:x1].mean(axis=1)
    az_prof_sm = _gf1(az_prof, sigma=sigma_az_px)
    thr = profile_thr * (az_prof_sm[yp] + 1e-9)
    print(f"  peak (y=6399, x=606)  slab cols = [{x0}..{x1})  ({x1-x0} px)")
    print(f"  az_prof_sm[yp] = {az_prof_sm[yp]:.4f}  →  thr = 0.15 · profile[yp] = {thr:.4f}")
    print(f"  az_prof_sm[6300]={az_prof_sm[6300]:.4f}  6350={az_prof_sm[6350]:.4f}  "
          f"6395={az_prof_sm[6395]:.4f}  6400={az_prof_sm[6400]:.4f}")
    # Walk up from yp to find where the run ends.
    lo = yp
    while lo > 0 and az_prof_sm[lo - 1] >= thr:
        lo -= 1
    hi = yp
    while hi < len(az_prof_sm) - 1 and az_prof_sm[hi + 1] >= thr:
        hi += 1
    print(f"  contiguous run above thr: rows {lo}..{hi}  ({hi-lo+1} px)")
    print(
        f"  → per-peak az interval before clamps: "
        f"y = [{lo}, {hi}] centred on {yp}"
    )

    # Compare with narrower slabs.
    for slab_half in (32, 16, 8, 4):
        x0n = max(0, xp - slab_half)
        x1n = min(mask_filt.shape[1], xp + slab_half + 1)
        az_prof_narrow = mask_filt[:, x0n:x1n].mean(axis=1)
        az_prof_narrow_sm = _gf1(az_prof_narrow, sigma=sigma_az_px)
        thr_n = profile_thr * (az_prof_narrow_sm[yp] + 1e-9)
        lo_n = yp
        while lo_n > 0 and az_prof_narrow_sm[lo_n - 1] >= thr_n:
            lo_n -= 1
        hi_n = yp
        while hi_n < len(az_prof_narrow_sm) - 1 and az_prof_narrow_sm[hi_n + 1] >= thr_n:
            hi_n += 1
        # Apply the same symmetric az_max clamp view_subaps_cov / clustering do.
        half = max((hi_n - lo_n) / 2.0, az_min_px / 2.0)
        half = min(half, az_max_px / 2.0)
        clamped_lo = int(max(0, yp - half))
        clamped_hi = int(min(mask_filt.shape[0] - 1, yp + half))
        print(
            f"  slab_half = ±{slab_half:>2d} (width={2*slab_half+1:>3d} px = "
            f"{(2*slab_half+1)*rg_m_per_px:>5.1f} m)  "
            f"prof[yp]={az_prof_narrow_sm[yp]:.4f}  thr={thr_n:.4f}  "
            f"run={lo_n}..{hi_n} ({hi_n-lo_n+1} px)  "
            f"→ clamped={clamped_lo}..{clamped_hi}"
        )

    # ---- CFAR verdict for the two most interesting clusters -----------
    print("\n=== CFAR verdict for pre-CFAR clusters overlapping the zoom ===")
    for c in pre_overlap:
        y_lo, y_hi, x_lo, x_hi = (int(v) for v in boxes[c])
        box_amp = amp_proxy[y_lo:y_hi + 1, x_lo:x_hi + 1]
        box_msk = mask_filt[y_lo:y_hi + 1, x_lo:x_hi + 1]
        sig_pix = box_amp[box_msk == 1]
        signal = float(sig_pix.mean()) if sig_pix.size else float(box_amp.mean())
        n_guard = max(1, CFAR_RANGE_BINS)
        prev_lo = max(0, x_lo - n_guard)
        succ_hi = min(W, x_hi + 1 + n_guard)
        prev_strip = amp_proxy[y_lo:y_hi + 1, prev_lo:x_lo]
        succ_strip = amp_proxy[y_lo:y_hi + 1, x_hi + 1:succ_hi]
        noise_prev = float(prev_strip.mean()) if prev_strip.size else float("nan")
        noise_succ = float(succ_strip.mean()) if succ_strip.size else float("nan")
        noise = min([v for v in (noise_prev, noise_succ) if not np.isnan(v)] or [1e-12])
        snr = signal / max(noise, 1e-12)
        verdict = "KEPT " if snr >= CFAR_SNR_TH else "KILLED"
        print(
            f"  cluster {c:>3d}  signal={signal:>8.1f}  noise_prev={noise_prev:>7.1f}  "
            f"noise_succ={noise_succ:>7.1f}  →  SNR={snr:>4.2f}  {verdict}"
        )

    # Post-CFAR overlap_cids for the (post-CFAR) figure below.
    overlap_cids = []
    for c in range(n_clusters_kept):
        by_lo, by_hi, bx_lo, bx_hi = (int(v) for v in boxes_kept[c])
        if not (by_hi < ZOOM_Y_LO or by_lo > ZOOM_Y_HI
                or bx_hi < ZOOM_X_LO or bx_lo > ZOOM_X_HI):
            overlap_cids.append(c)

    # ---- coverage per (cluster, target) -------------------------------
    print("\n=== Overlap of cluster boxes with each declared target ===")
    for name, y_lo, y_hi, x_lo, x_hi in TARGETS:
        t_area = (y_hi - y_lo) * (x_hi - x_lo)
        print(f"{name}  rows {y_lo}:{y_hi}  cols {x_lo}:{x_hi}  area = {t_area}")
        for c in overlap_cids:
            by_lo, by_hi, bx_lo, bx_hi = (int(v) for v in boxes_kept[c])
            # Convert box to half-open to match target rectangle convention.
            iy_lo = max(y_lo, by_lo)
            iy_hi = min(y_hi, by_hi + 1)
            ix_lo = max(x_lo, bx_lo)
            ix_hi = min(x_hi, bx_hi + 1)
            if iy_hi > iy_lo and ix_hi > ix_lo:
                area = (iy_hi - iy_lo) * (ix_hi - ix_lo)
                b_area = (by_hi - by_lo + 1) * (bx_hi - bx_lo + 1)
                print(
                    f"    cluster {c:>3d}: intersect = {area}  "
                    f"→ {100 * area / t_area:5.1f}% of target, "
                    f"{100 * area / b_area:5.1f}% of box"
                )

    # ---- inspect the CoV mask density inside each target --------------
    print("\n=== Mask density inside declared targets ===")
    for name, y_lo, y_hi, x_lo, x_hi in TARGETS:
        sub = mask_filt[y_lo:y_hi, x_lo:x_hi]
        raw_sub = mask_dec[y_lo:y_hi, x_lo:x_hi]
        print(
            f"{name}  raw mask hits = {int(raw_sub.sum())}/{raw_sub.size} "
            f"({100 * raw_sub.mean():.1f}%)   "
            f"filtered = {int(sub.sum())}/{sub.size} ({100 * sub.mean():.1f}%)"
        )

    # ============================================================== #
    # Zoom-in rendering (four panels: sub_mean, mask, pre-CFAR, post-CFAR)
    # ============================================================== #
    y_lo_v, y_hi_v = ZOOM_Y_LO, ZOOM_Y_HI
    x_lo_v, x_hi_v = ZOOM_X_LO, ZOOM_X_HI

    sub_mean_v = np.log10(np.maximum(sub_mean[y_lo_v:y_hi_v, x_lo_v:x_hi_v], 1e-6))
    mask_v = mask_filt[y_lo_v:y_hi_v, x_lo_v:x_hi_v]

    fig, (ax_a, ax_m, ax_pre, ax_post) = plt.subplots(
        1, 4, figsize=(20, 10), constrained_layout=True, sharey=True,
    )
    extent = (x_lo_v, x_hi_v, y_hi_v, y_lo_v)

    ax_a.imshow(sub_mean_v, cmap="gray", aspect="auto",
                extent=extent, interpolation="nearest",
                vmin=np.percentile(sub_mean_v, 2),
                vmax=np.percentile(sub_mean_v, 99.5))
    ax_a.set_title("log10(sub_mean)  —  zoom")

    ax_m.imshow(mask_v, cmap="gray", aspect="auto",
                extent=extent, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_m.set_title("CoV mask (filtered)")

    def _draw_boxes(ax, boxes_arr, labels_arr, peaks_arr, cids, title):
        ax.imshow(mask_v, cmap="gray", aspect="auto",
                  extent=extent, vmin=0.0, vmax=1.0, interpolation="nearest")
        cmap_cl = plt.get_cmap("tab20")
        for c in cids:
            colour = cmap_cl(c % 20)
            sel = labels_arr == c
            ax.scatter(peaks_arr[sel, 1], peaks_arr[sel, 0],
                       s=25, c=[colour], edgecolor="k", linewidth=0.3)
            by_lo, by_hi, bx_lo, bx_hi = (int(v) for v in boxes_arr[c])
            ax.add_patch(Rectangle(
                (bx_lo, by_lo), bx_hi - bx_lo, by_hi - by_lo,
                fill=False, edgecolor=colour, linewidth=1.4,
            ))
            ax.text(bx_hi + 0.5, by_lo + 2, f"c{c}", color=colour,
                    fontsize=9, fontweight="bold")
        ax.set_title(title)

    _draw_boxes(ax_pre, boxes, labels, peaks_yx, pre_overlap,
                f"PRE-CFAR clusters overlapping zoom ({len(pre_overlap)})")
    _draw_boxes(ax_post, boxes_kept, labels_kept, peaks_yx_kept, overlap_cids,
                f"POST-CFAR clusters overlapping zoom ({len(overlap_cids)})")

    for ax in (ax_a, ax_m, ax_pre, ax_post):
        for name, y_lo, y_hi, x_lo, x_hi in TARGETS:
            ax.add_patch(Rectangle(
                (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                fill=False, edgecolor="red", linewidth=2.0, linestyle="--",
            ))
            ax.text(x_hi + 1, y_lo + 5, name, color="red", fontsize=10)
        ax.set_xlim(x_lo_v, x_hi_v)
        ax.set_ylim(y_hi_v, y_lo_v)
        ax.set_xlabel("range pixel (s_degraded)")
    ax_a.set_ylabel("sub-aperture-index row")

    fig.suptitle(
        f"Two-target diagnostic — {NPZ.name}\n"
        f"red dashed = operator-declared targets; coloured = cluster_peaks output",
        fontsize=10,
    )
    out = Path("/home/odogan/Desktop/ship_focus/qgis-iceye-plugin/diag_two_targets.png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved zoom figure to {out}")


if __name__ == "__main__":
    main()
