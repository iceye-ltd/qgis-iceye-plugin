"""Diagnostic: trace where each of 5 missing low-backscatter targets is
eliminated by the shear_averaging.py pipeline.

For each ROI in s_degraded coordinates:

  Step A — Range-windowing + range degradation: report mean / max
           amplitude inside the ROI vs the global noise level
           (= median |s_degraded|).

  Step B — Sub-aperture CoV gate: how many ROI pixels would have
           cov² > th = cov_th_mult * median(cov_sq) (= the CoV
           pre-filter that produces mask_full).

  Step C — boundary_box seed peaks: how many local-amp-max pixels
           inside the ROI also pass the 50%-density gate (= the seeds
           passed to grow_and_recenter_boxes).

  Step D — Single-line wrap-aware phase fit on the box: slope, slope·n,
           coh-weighted mean wrapped residual. Would the
           three-tier motion gate keep it? What about the residual
           filter (1.0 rad default)? CoV-mass bypass available?

  Step E — Subaperture COM peak-to-peak across the N sub-bands
           (the az_pp tiebreaker).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter

# Reuse functions from the main script.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shear_averaging import (  # noqa: E402
    apply_range_window,
    degrade_range_resolution_range_sum,
    compute_subapertures,
    grow_and_recenter_boxes,
    _min_distance_line_fit,
    _refocus_box_chip,
    _normalized_variance,
)


PATCH = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_141944_569017.npy"
)

# (label, y_lo, y_hi, x_lo, x_hi) in s_degraded coordinates.
ROIS = [
    ("T1", 10500, 12500, 322, 330),
    ("T2", 24800, 26400, 360, 362),
    ("T3", 33000, 35000, 335, 345),
    ("T4", 38000, 40000, 326, 334),
    ("T5", 42500, 44250, 272, 278),
]

# Defaults from main() of shear_averaging.py.
RANGE_LOOKS = 6
RANGE_SLL_DB = 55.0
RANGE_TAYLOR_NBAR = 8
N_SUBAPERTURE = 8           # matches what produced the latest NPZ
COV_TH_MULT = 1.5
SEED_DENSITY = 0.5
SEED_WINDOW = (100, 3)
GROW_DENSITY = 0.25
N_TARGET_AZIMUTH_WIDTH = 100   # seed window h
N_TARGET_RANGE_LENGTH = 3      # seed window w
MAX_SIZE_AZIMUTH_BIN = 2000
MAX_SIZE_RANGE_BIN = 130 // 3  # = 43 px (max_length_of_target/degrade_range_resolution)
NMS_AZ_M = 1000.0              # max_dist_az_m
NMS_RG_M = 20.0                # max_dist_rg_m = max_width_of_target
AZ_SPACING_M = 0.5             # m / s_degraded az pixel
RG_SPACING_M = 0.5 * 6         # m / s_degraded rg pixel (range_spacing * Number_of_Range_Looks)
SLOPE_RAD_THRESH = 1.5
SLOPE_RAD_LOWER = 0.1
AZ_PP_SUB_THRESH = 20.0
MAX_RESIDUAL_RAD = 1.5
RESIDUAL_SKIP_DET_FRAC = 0.25
REFOCUS_MIN_GAIN_DB = -5.0   # shear_averaging.py default
MIN_ROW_COHERENCE = 0.5
INLIER_TOL_RAD = 0.5


def hr(c: str = "=") -> str:
    return c * 78


def main() -> None:
    print(f"Loading {PATCH.name} ...")
    s = np.load(PATCH)
    print(f"  SLC shape={s.shape}, dtype={s.dtype}")

    print("\nRange Taylor window + range degradation ...")
    s_win = apply_range_window(s, sll_db=RANGE_SLL_DB, nbar=RANGE_TAYLOR_NBAR)
    s_deg, d_phase = degrade_range_resolution_range_sum(s_win, RANGE_LOOKS)
    del s, s_win   # free memory
    H, W = s_deg.shape
    print(f"  s_degraded shape={s_deg.shape}")

    amp = np.abs(s_deg).astype(np.float32)
    amp_global_med = float(np.median(amp))
    amp_global_p99 = float(np.percentile(amp, 99))
    print(f"  global |s_deg|  median={amp_global_med:.3g}  p99={amp_global_p99:.3g}")

    print("\nSub-aperture stack + CoV² gate ...")
    sub_unmasked = compute_subapertures(s_deg, N_SUBAPERTURE)  # (N, sub_size, W)
    sub_mean = sub_unmasked.mean(axis=0)
    sub_var = sub_unmasked.var(axis=0)
    cov_sq = sub_var / (sub_mean ** 2 + 1e-12)
    th = COV_TH_MULT * float(np.median(cov_sq))
    mask_dec = (cov_sq > th).astype(np.float32)
    print(f"  median(cov²)={float(np.median(cov_sq)):.3g}  threshold th={th:.3g}")

    # Upsample mask by N_SUBAPERTURE in azimuth so it matches s_deg.
    mask_full = np.repeat(mask_dec, N_SUBAPERTURE, axis=0)
    pad = H - mask_full.shape[0]
    if pad > 0:
        mask_full = np.vstack([mask_full, np.repeat(mask_full[-1:], pad, axis=0)])
    assert mask_full.shape == s_deg.shape
    n_kept = int(mask_full.sum())
    print(f"  mask_full kept {n_kept}/{mask_full.size} pixels "
          f"({100 * n_kept / mask_full.size:.2f}%)")

    print("\nboundary_box seed peaks (density>{:.2f} in 100x3 + local max) ...".format(SEED_DENSITY))
    box_area = SEED_WINDOW[0] * SEED_WINDOW[1]
    det_count = uniform_filter(
        mask_full.astype(np.float32), size=SEED_WINDOW, mode="constant"
    ) * box_area
    amp_max = maximum_filter(amp, size=SEED_WINDOW, mode="constant")
    is_det = mask_full == 1
    is_dense = det_count > SEED_DENSITY * box_area
    is_peak = amp == amp_max
    seeds = is_det & is_dense & is_peak
    seeds_yx = np.argwhere(seeds)        # (M, 2)
    seed_amp = amp[seeds_yx[:, 0], seeds_yx[:, 1]]
    print(f"  total seed peaks in scene: {int(seeds.sum())}")

    # Pre-compute mask coverage at the masked s_deg (for d_phase row weights).
    # The script uses unmasked d_phase for per-box phase, but rows are weighted
    # by amp_raw[*] * amp_raw[*+1], so masked-out pixels (amp=0 only matters
    # if masked is applied to amp_raw too; in the script amp_raw stays
    # unmasked, so we use the same here).
    amp_raw = amp  # in the script amp_raw = |s_degraded| BEFORE masking

    # --- Sub-aperture stack (post-mask in the script). For COM PP across
    # sub-bands we follow the same convention: mask s_degraded then re-sub.
    s_deg_masked = s_deg * mask_full
    subs_masked = compute_subapertures(s_deg_masked, N_SUBAPERTURE)
    sub_size = subs_masked.shape[1]
    del s_deg_masked

    for label, y0, y1, x0, x1 in ROIS:
        print("\n" + hr("="))
        print(f"ROI {label}: s_deg[{y0}:{y1}, {x0}:{x1}]   "
              f"(h={y1 - y0}, w={x1 - x0})")
        print(hr("-"))

        # --- A: amplitude vs noise ----------------------------------------
        roi = amp[y0:y1, x0:x1]
        roi_med = float(np.median(roi))
        roi_max = float(roi.max())
        roi_mean = float(roi.mean())
        # Surrounding 'background' just outside the ROI (same azimuth band,
        # ±30 rg columns wider). Helpful to see the local clutter floor.
        bx_lo = max(0, x0 - 30)
        bx_hi = min(W, x1 + 30)
        bg = amp[y0:y1, bx_lo:bx_hi]
        bg_med = float(np.median(bg))
        snr_db = 20.0 * np.log10(max(roi_max, 1e-30) / max(bg_med, 1e-30))
        peak_over_global = 20.0 * np.log10(max(roi_max, 1e-30) / max(amp_global_med, 1e-30))
        print(f"  [A] amplitude   ROI: med={roi_med:.3g}  mean={roi_mean:.3g}  "
              f"max={roi_max:.3g}")
        print(f"      background (same az, ±30 rg): med={bg_med:.3g}")
        print(f"      peak / local_bg = {snr_db:+5.1f} dB,   "
              f"peak / global_median = {peak_over_global:+5.1f} dB")

        # --- B: CoV² gate -------------------------------------------------
        roi_mask = mask_full[y0:y1, x0:x1]
        m_kept = int(roi_mask.sum())
        m_total = int(roi_mask.size)
        # CoV² actually lives on the decimated grid (sub_size, W). Project ROI:
        y0_d = max(0, y0 // N_SUBAPERTURE)
        y1_d = min(sub_size, (y1 + N_SUBAPERTURE - 1) // N_SUBAPERTURE)
        roi_cov_sq = cov_sq[y0_d:y1_d, x0:x1]
        roi_cov_med = float(np.median(roi_cov_sq))
        roi_cov_p90 = float(np.percentile(roi_cov_sq, 90))
        roi_cov_max = float(roi_cov_sq.max())
        print(f"  [B] CoV gate     mask kept {m_kept}/{m_total} pixels "
              f"({100 * m_kept / max(m_total, 1):.2f}%)")
        print(f"      ROI cov² (decim grid): med={roi_cov_med:.3g}, "
              f"p90={roi_cov_p90:.3g}, max={roi_cov_max:.3g}   "
              f"(global th = {th:.3g})")

        # --- C: seed peaks inside ROI -------------------------------------
        roi_seeds = int(seeds[y0:y1, x0:x1].sum())
        # Density inside the ROI (no 100x3 sliding window — purely a sanity
        # check that the 50% threshold could ever be met).
        density_overall = m_kept / max(m_total, 1)
        print(f"  [C] seeds in ROI : {roi_seeds}")
        print(f"      overall ROI mask density = {density_overall:.3f}  "
              f"(need ≥ {SEED_DENSITY:.2f} inside a {SEED_WINDOW[0]}×{SEED_WINDOW[1]} window)")

        # If no seed inside the ROI itself, also look for the closest seed
        # cell in azimuth that might still have grown into the ROI later.
        if roi_seeds == 0:
            seed_rows = np.flatnonzero(seeds[max(0, y0 - 1500):min(H, y1 + 1500),
                                              x0:x1].any(axis=1))
            if seed_rows.size:
                nearest = int(seed_rows[0]) + max(0, y0 - 1500)
                d_az = nearest - (y0 + y1) // 2
                print(f"      nearest seed in same range cols within ±1500 rows: "
                      f"az={nearest} (Δaz={d_az:+d})")
            else:
                print("      no seed within ±1500 rows in the same range cols")

        # --- D: per-box single-line phase fit (assume box ≈ ROI) ----------
        # Build the same coh / phi inputs filter_boxes_by_phase_residual uses.
        y0p, y1p = y0, min(d_phase.shape[0], y1)
        x0p, x1p = x0, min(d_phase.shape[1], x1)
        L = y1p - y0p
        weight = amp_raw[y0p:y1p, x0p:x1p] * amp_raw[y0p + 1:y1p + 1, x0p:x1p]
        sub_phase = d_phase[y0p:y1p, x0p:x1p]
        z = (weight * np.exp(1j * sub_phase)).sum(axis=1)
        w_sum = weight.sum(axis=1) + 1e-12
        phi = np.angle(z)
        coh = np.abs(z) / w_sum
        u = np.arange(L, dtype=float)
        slope, intercept, n_in = _min_distance_line_fit(
            phi.astype(np.float64), u, coh.astype(np.float64),
            min_row_coherence=MIN_ROW_COHERENCE,
            inlier_tol_rad=INLIER_TOL_RAD,
        )
        if np.isfinite(slope) and np.isfinite(intercept):
            r = np.angle(np.exp(1j * (phi - (slope * u + intercept))))
            high = coh > MIN_ROW_COHERENCE
            if int(high.sum()) < 2:
                high = np.ones_like(coh, dtype=bool)
            w_row = np.where(high, coh, 0.0)
            resid = float((w_row * np.abs(r)).sum() / max(w_row.sum(), 1e-12))
            slope_total = float(slope) * L
            print(f"  [D] phase fit    slope={slope:+.4e} rad/row, "
                  f"|slope·n|={abs(slope_total):.2f} rad   "
                  f"residual={resid:.2f} rad   "
                  f"high-coh rows={int(high.sum())}/{L}")
            print(f"      coh stats   med={float(np.median(coh)):.3f}, "
                  f"p90={float(np.percentile(coh, 90)):.3f}, "
                  f"max={float(coh.max()):.3f}")
            # Apply the three-tier rule (assume az_pp computed below):
            tier_drop = abs(slope_total) < SLOPE_RAD_LOWER
            tier_keep = abs(slope_total) >= SLOPE_RAD_THRESH
            print(f"      motion gate  |slope·n| vs (lower={SLOPE_RAD_LOWER}, "
                  f"upper={SLOPE_RAD_THRESH}):  ", end="")
            if tier_drop:
                print("CLEAR DROP (below lower)")
            elif tier_keep:
                print("CLEAR KEEP (above upper)")
            else:
                print("ambiguous (needs az_pp tiebreaker)")
            print(f"      residual gate ({MAX_RESIDUAL_RAD} rad): ", end="")
            if resid <= MAX_RESIDUAL_RAD:
                print(f"PASS ({resid:.2f} ≤ {MAX_RESIDUAL_RAD})")
            else:
                # Detection-bypass check
                det_in_box = int(mask_full[y0p:y1p, x0p:x1p].sum())
                if det_in_box > RESIDUAL_SKIP_DET_FRAC * L:
                    print(f"FAIL ({resid:.2f}) but BYPASSED by det count "
                          f"{det_in_box} > {RESIDUAL_SKIP_DET_FRAC}·{L}")
                else:
                    print(f"FAIL ({resid:.2f} > {MAX_RESIDUAL_RAD}), "
                          f"det count {det_in_box} ≤ {RESIDUAL_SKIP_DET_FRAC}·{L}")
        else:
            print("  [D] phase fit    DEGENERATE (NaN slope/intercept)")

        # --- F: NMS neighbourhood ----------------------------------------
        # The script uses centre-distance NMS in an ellipse:
        #   (Δaz_m / NMS_AZ_M)^2 + (Δrg_m / NMS_RG_M)^2 < 1
        # Scores are MAX |s_deg| inside each box's grown rectangle.
        # Approximation: use the seed amp as a per-seed score (since
        # boxes are grown from these seeds), and centre = seed.
        y_c, x_c = (y0 + y1) // 2, (x0 + x1) // 2
        # Use the ROI's brightest pixel as a stand-in for what its own
        # post-grow MAX score would be.
        roi_score = roi_max
        dy_m = (seeds_yx[:, 0] - y_c) * AZ_SPACING_M
        dx_m = (seeds_yx[:, 1] - x_c) * RG_SPACING_M
        inside = (
            (dy_m / NMS_AZ_M) ** 2 + (dx_m / NMS_RG_M) ** 2 < 1.0
        )
        nbrs_inside = int(inside.sum())
        # Look at the BRIGHTER neighbours (their seed-amp > roi_score):
        brighter = inside & (seed_amp > roi_score)
        n_brighter = int(brighter.sum())
        print(f"  [F] NMS ellipse  (Δaz≤{NMS_AZ_M:.0f} m, Δrg≤{NMS_RG_M:.0f} m): "
              f"{nbrs_inside} seed peaks inside,  "
              f"{n_brighter} brighter than ROI peak ({roi_score:.2g})")
        if n_brighter:
            br_idx = np.flatnonzero(brighter)
            # Show the 5 brightest brighter-neighbours.
            order = np.argsort(-seed_amp[br_idx])[:5]
            for kk in order:
                ii = br_idx[kk]
                ys, xs = seeds_yx[ii]
                d_az = ys - y_c
                d_rg = xs - x_c
                ellipse_norm = (d_az * AZ_SPACING_M / NMS_AZ_M) ** 2 + (
                    d_rg * RG_SPACING_M / NMS_RG_M
                ) ** 2
                print(f"      brighter seed @ az={ys:6d} rg={xs:3d} "
                      f"amp={float(seed_amp[ii]):6.1f}  "
                      f"Δaz={d_az:+6d} px (={d_az*AZ_SPACING_M:+7.0f} m)  "
                      f"Δrg={d_rg:+4d} px (={d_rg*RG_SPACING_M:+5.0f} m)  "
                      f"ellipse_norm={ellipse_norm:.2f}")

        # --- G: simulate grow_and_recenter from EVERY seed inside the ROI,
        # then compute the actual phase fit + motion-gate verdict on each.
        roi_seed_mask = (
            (seeds_yx[:, 0] >= y0) & (seeds_yx[:, 0] < y1)
            & (seeds_yx[:, 1] >= x0) & (seeds_yx[:, 1] < x1)
        )
        roi_seed_idx = np.flatnonzero(roi_seed_mask)
        if roi_seed_idx.size > 0:
            seeds_in_roi = seeds_yx[roi_seed_idx]
            grown = grow_and_recenter_boxes(
                seeds_in_roi, mask=mask_full, amp=amp,
                initial_hw=(N_TARGET_AZIMUTH_WIDTH, N_TARGET_RANGE_LENGTH),
                az_step=5, rg_step=1,
                density_threshold=GROW_DENSITY,
                max_h=MAX_SIZE_AZIMUTH_BIN, max_w=MAX_SIZE_RANGE_BIN,
            )
            print(f"  [G] grow + per-box phase fit + motion-gate verdict, "
                  f"one row per ROI seed (sorted by amp desc):")
            print(f"      {'seed_az':>8s} {'rg':>4s} {'amp':>7s} | "
                  f"{'h':>4s} {'w':>3s} {'|slope·n|':>10s} {'resid':>6s} "
                  f"{'az_pp':>6s} | verdict")
            order = np.argsort(-seed_amp[roi_seed_idx])
            for kk in order:
                ys, xs = seeds_in_roi[kk]
                y_g, x_g, h_g, w_g = grown[kk]
                # Phase fit on the exact grown box.
                y0g = max(0, int(round(y_g - h_g / 2)))
                y1g = min(d_phase.shape[0], int(round(y_g + h_g / 2)))
                x0g = max(0, int(round(x_g - w_g / 2)))
                x1g = min(d_phase.shape[1], int(round(x_g + w_g / 2)))
                Lg = y1g - y0g
                if Lg < 2 or x1g - x0g < 1:
                    print(f"      {int(ys):8d} {int(xs):4d} "
                          f"{float(seed_amp[roi_seed_idx[kk]]):7.1f} | "
                          f"{h_g:4d} {w_g:3d}    degenerate (h<2 or w<1)")
                    continue
                wt_g = amp_raw[y0g:y1g, x0g:x1g] * amp_raw[y0g+1:y1g+1, x0g:x1g]
                z_g = (wt_g * np.exp(1j * d_phase[y0g:y1g, x0g:x1g])).sum(axis=1)
                w_sum_g = wt_g.sum(axis=1) + 1e-12
                phi_g = np.angle(z_g)
                coh_g = np.abs(z_g) / w_sum_g
                u_g = np.arange(Lg, dtype=float)
                sl_g, ic_g, _ = _min_distance_line_fit(
                    phi_g.astype(np.float64), u_g, coh_g.astype(np.float64),
                    min_row_coherence=MIN_ROW_COHERENCE,
                    inlier_tol_rad=INLIER_TOL_RAD,
                )
                if np.isfinite(sl_g):
                    r_g = np.angle(np.exp(1j * (phi_g - (sl_g * u_g + ic_g))))
                    high_g = coh_g > MIN_ROW_COHERENCE
                    if int(high_g.sum()) < 2:
                        high_g = np.ones_like(coh_g, dtype=bool)
                    wr_g = np.where(high_g, coh_g, 0.0)
                    res_g = float((wr_g * np.abs(r_g)).sum() / max(wr_g.sum(), 1e-12))
                    sn_g = abs(float(sl_g) * Lg)
                else:
                    res_g = float("nan"); sn_g = float("nan")
                # Az COM PP on the GROWN box's azimuth range across N subaps.
                y0_sub_g = max(0, y_g // N_SUBAPERTURE - h_g // (2 * N_SUBAPERTURE))
                y1_sub_g = min(sub_size,
                               y_g // N_SUBAPERTURE + h_g // (2 * N_SUBAPERTURE))
                com_az_g = []
                for j in range(N_SUBAPERTURE):
                    img_g = subs_masked[j, y0_sub_g:y1_sub_g, x0g:x1g].astype(np.float64)
                    tot_g = float(img_g.sum())
                    if tot_g <= 0.0:
                        com_az_g.append(np.nan); continue
                    yy_g = np.arange(y0_sub_g, y1_sub_g, dtype=np.float64)
                    com_az_g.append(float((img_g.sum(axis=1) * yy_g).sum() / tot_g))
                com_az_g = np.asarray(com_az_g)
                if np.isfinite(com_az_g).any():
                    azpp_g = float(np.nanmax(com_az_g) - np.nanmin(com_az_g))
                else:
                    azpp_g = 0.0
                # Three-tier rule verdict.
                if not np.isfinite(sn_g):
                    verdict = "DROP (phase NaN)"
                elif sn_g >= SLOPE_RAD_THRESH:
                    verdict = "KEEP (slope·n ≥ upper)"
                elif sn_g < SLOPE_RAD_LOWER:
                    verdict = "DROP (slope·n < lower)"
                elif azpp_g >= AZ_PP_SUB_THRESH:
                    verdict = "KEEP (middle band, az_pp ≥ thresh)"
                else:
                    verdict = f"DROP (middle band, az_pp {azpp_g:.1f} < {AZ_PP_SUB_THRESH:g})"
                # Refocus-gain filter (when motion gate kept the box).
                refoc_str = ""
                if verdict.startswith("KEEP") and np.isfinite(sl_g):
                    # Same chip the script uses: s_degraded MASKED (the
                    # script's `chip_k = s_degraded[...]` is the masked
                    # complex array). We approximate with s_deg * mask.
                    chip = s_deg[y0g:y1g, x0g:x1g] * mask_full[y0g:y1g, x0g:x1g]
                    corrected, _ = _refocus_box_chip(chip, float(sl_g))
                    c_b = _normalized_variance(chip)
                    c_a = _normalized_variance(corrected)
                    if c_b > 0.0 and c_a > 0.0:
                        gain_db = 20.0 * np.log10(c_a / c_b)
                        if gain_db < REFOCUS_MIN_GAIN_DB:
                            refoc_str = (f"  → REFOCUS-GATE DROP "
                                         f"(gain={gain_db:+.2f} dB < "
                                         f"{REFOCUS_MIN_GAIN_DB} dB)")
                        else:
                            refoc_str = f"  → refocus gain={gain_db:+.2f} dB (kept)"
                    else:
                        refoc_str = "  → refocus gain=NaN (kept)"
                # Residual gate (separate; would drop before motion gate).
                resid_verdict = ""
                if np.isfinite(res_g):
                    det_in_box = int(mask_full[y0g:y1g, x0g:x1g].sum())
                    if res_g > MAX_RESIDUAL_RAD:
                        if det_in_box > RESIDUAL_SKIP_DET_FRAC * Lg:
                            resid_verdict = (f" [resid {res_g:.2f}>1.0 BYPASSED "
                                             f"by det {det_in_box}>{int(RESIDUAL_SKIP_DET_FRAC*Lg)}]")
                        else:
                            resid_verdict = (f" [resid GATE DROP: {res_g:.2f}>1.0, "
                                             f"det {det_in_box}≤{int(RESIDUAL_SKIP_DET_FRAC*Lg)}]")
                print(f"      {int(ys):8d} {int(xs):4d} "
                      f"{float(seed_amp[roi_seed_idx[kk]]):7.1f} | "
                      f"{h_g:4d} {w_g:3d} {sn_g:10.2f} {res_g:6.2f} "
                      f"{azpp_g:6.2f} | {verdict}{resid_verdict}{refoc_str}")
        else:
            print("  [G] no seed in ROI → grow stage never reached "
                  "(target dies at the boundary_box seed stage)")

        # --- E: subaperture COM peak-to-peak (az_pp tiebreaker) ----------
        y0_sub = max(0, y0 // N_SUBAPERTURE)
        y1_sub = min(sub_size, (y1 + N_SUBAPERTURE - 1) // N_SUBAPERTURE)
        com_az = []
        for j in range(N_SUBAPERTURE):
            img = subs_masked[j, y0_sub:y1_sub, x0:x1].astype(np.float64)
            tot = float(img.sum())
            if tot <= 0.0:
                com_az.append(np.nan)
                continue
            yy = np.arange(y0_sub, y1_sub, dtype=np.float64)
            com_az.append(float((img.sum(axis=1) * yy).sum() / tot))
        com_az = np.asarray(com_az)
        fin = np.isfinite(com_az)
        az_pp = float(np.nanmax(com_az) - np.nanmin(com_az)) if fin.any() else 0.0
        print(f"  [E] az COM PP across {N_SUBAPERTURE} subaps: "
              f"az_pp = {az_pp:.2f} sub-az px  "
              f"(threshold for middle-band keep = {AZ_PP_SUB_THRESH:g})")
        print(f"      finite subaps = {int(fin.sum())}/{N_SUBAPERTURE}")


if __name__ == "__main__":
    main()
