"""Diagnose why a target ROI is missed by the shear_averaging pipeline.

Walks the same processing chain as scripts/shear_averaging.py and, at every
stage, reports what survives inside a user-supplied ROI:

    az_lo:az_hi, rg_lo:rg_hi   (in s_degraded coordinates)

For each stage we print a one-line "verdict": how many pixels of the ROI
pass it, what fraction, and (where useful) the stage's threshold so you
can see what would tip it over.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

from shear_averaging import (
    DEFAULT_PATCH,
    compute_subapertures,
    degrade_range_resolution_range_sum,
    grow_and_recenter_boxes,
    merge_by_strip_amplitude,
    nms_boxes,
    nms_by_centre_distance,
)


# --- Pipeline parameters (must mirror main() in shear_averaging.py) -------
MAX_LENGTH_OF_TARGET = 130
MIN_SIZE_OF_TARGET = 3
RANGE_SPACING = 0.5
NUMBER_OF_RANGE_LOOKS = 5
N_SUBAPERTURE = 16
N_TARGET_AZIMUTH_WIDTH = 100
N_TARGET_RANGE_LENGTH = 3
SEED_DENSITY_THRESHOLD = 0.5
GROW_DENSITY_THRESHOLD = 0.25
NMS_IOU_THRESH = 0.3


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{n}/{total} ({100 * n / total:.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("--az-lo", type=int, default=6800)
    parser.add_argument("--az-hi", type=int, default=8000)
    parser.add_argument("--rg-lo", type=int, default=93)
    parser.add_argument("--rg-hi", type=int, default=95)
    args = parser.parse_args()

    s = np.load(args.path)
    print(f"Loaded: {args.path.name}  shape={s.shape}  dtype={s.dtype}")
    az_lo, az_hi = args.az_lo, args.az_hi
    rg_lo, rg_hi = args.rg_lo, args.rg_hi
    H_roi = az_hi - az_lo
    W_roi = rg_hi - rg_lo
    roi_size = H_roi * W_roi
    print(f"ROI: az [{az_lo}:{az_hi})  rg [{rg_lo}:{rg_hi})  → "
          f"{H_roi}×{W_roi} = {roi_size} px in s_degraded coords")
    print()

    # ------------------------------------------------------------------
    # 1. degrade range resolution
    # ------------------------------------------------------------------
    s_degraded, _ = degrade_range_resolution_range_sum(s, NUMBER_OF_RANGE_LOOKS)
    H_full, W_full = s_degraded.shape
    print(f"[1] s_degraded: shape={s_degraded.shape}")
    if not (0 <= az_lo < az_hi <= H_full and 0 <= rg_lo < rg_hi <= W_full):
        print(f"    !! ROI out of bounds: image is {H_full}×{W_full}")
        return

    amp = np.abs(s_degraded).astype(np.float32)
    amp_roi = amp[az_lo:az_hi, rg_lo:rg_hi]
    print(f"    |s_degraded| in ROI:  "
          f"min={amp_roi.min():.3g}  max={amp_roi.max():.3g}  "
          f"mean={amp_roi.mean():.3g}  median={np.median(amp_roi):.3g}")
    print(f"    global  amp percentiles: "
          f"50%={np.percentile(amp, 50):.3g}  "
          f"95%={np.percentile(amp, 95):.3g}  "
          f"99%={np.percentile(amp, 99):.3g}  "
          f"max={amp.max():.3g}")
    print()

    # ------------------------------------------------------------------
    # 2. CoV gate on subaperture stats (decimated grid, then upsampled)
    # ------------------------------------------------------------------
    subapertures = compute_subapertures(s_degraded, N_SUBAPERTURE)
    sub_mean = subapertures.mean(axis=0)
    sub_var = subapertures.var(axis=0)
    cov_sq = sub_var / (sub_mean ** 2 + 1e-12)
    th = 2.0 * np.median(cov_sq)
    mask_dec = (cov_sq > th).astype(np.float32)

    mask_full = np.repeat(mask_dec, N_SUBAPERTURE, axis=0)
    pad_rows = s_degraded.shape[0] - mask_full.shape[0]
    if pad_rows > 0:
        mask_full = np.vstack([mask_full, np.repeat(mask_full[-1:], pad_rows, axis=0)])
    assert mask_full.shape == s_degraded.shape

    # Snapshot raw amplitude before masking for the bridge merge.
    amp_raw = np.abs(s_degraded).astype(np.float32)

    print(f"[2] CoV gate:  th = 2·median(var/mean²) = {th:.3g}")
    # Map ROI rows to decimated rows (sub_size).
    sub_size = sub_mean.shape[0]
    dec_lo = az_lo // N_SUBAPERTURE
    dec_hi = min(sub_size, (az_hi + N_SUBAPERTURE - 1) // N_SUBAPERTURE)
    cov_roi = cov_sq[dec_lo:dec_hi, rg_lo:rg_hi]
    mask_dec_roi = mask_dec[dec_lo:dec_hi, rg_lo:rg_hi]
    n_dec_pass = int(mask_dec_roi.sum())
    print(f"    decimated grid rows {dec_lo}..{dec_hi}  ×  cols {rg_lo}..{rg_hi}  "
          f"= {mask_dec_roi.size} cells")
    print(f"    cov_sq in ROI:  min={cov_roi.min():.3g}  "
          f"max={cov_roi.max():.3g}  median={np.median(cov_roi):.3g}  "
          f"mean={cov_roi.mean():.3g}")
    print(f"    mask_dec passes (cov_sq > {th:.3g}):  {_pct(n_dec_pass, mask_dec_roi.size)}")

    mask_roi = mask_full[az_lo:az_hi, rg_lo:rg_hi]
    n_pass = int(mask_roi.sum())
    print(f"    mask_full passes (full-res ROI):       {_pct(n_pass, roi_size)}")

    # Per-row density profile, downsampled to 24 bins so it fits one screen.
    bins = 24
    bin_h = max(1, H_roi // bins)
    print(f"    per-row mask density (axis 0, 1 char per ~{bin_h} rows):")
    for k in range(bins):
        b_lo = az_lo + k * bin_h
        b_hi = min(az_hi, b_lo + bin_h)
        d = mask_full[b_lo:b_hi, rg_lo:rg_hi].mean()
        bar = "#" * int(round(d * 30))
        print(f"      az {b_lo:>5}..{b_hi:<5}  {d*100:5.1f}% |{bar}")
    print()

    # ------------------------------------------------------------------
    # 3. seed peak detection: is_detection & is_dense & is_peak
    # ------------------------------------------------------------------
    box_size = (N_TARGET_AZIMUTH_WIDTH, N_TARGET_RANGE_LENGTH)
    box_area = N_TARGET_AZIMUTH_WIDTH * N_TARGET_RANGE_LENGTH
    # Mirror main(): zero out s_degraded outside the mask BEFORE computing amp,
    # so is_peak ranks against the masked image (same as the pipeline).
    s_degraded = s_degraded * mask_full
    amp = np.abs(s_degraded).astype(np.float32)
    det_count = uniform_filter(mask_full.astype(np.float32),
                               size=box_size, mode="constant") * box_area
    amp_max = maximum_filter(amp, size=box_size, mode="constant")

    is_detection = mask_full == 1
    is_dense = det_count > SEED_DENSITY_THRESHOLD * box_area
    is_peak = amp == amp_max
    boundary_box = (is_detection & is_dense & is_peak)

    print(f"[3] seed test:  window={box_size}, "
          f"density > {SEED_DENSITY_THRESHOLD} × {box_area} = "
          f"{SEED_DENSITY_THRESHOLD * box_area:.0f}, "
          f"and pixel = local amp max in window")
    print(f"    is_detection (mask==1):                 "
          f"{_pct(int(is_detection[az_lo:az_hi, rg_lo:rg_hi].sum()), roi_size)}")
    dc_roi = det_count[az_lo:az_hi, rg_lo:rg_hi]
    print(f"    det_count in ROI:                       "
          f"min={dc_roi.min():.0f}  max={dc_roi.max():.0f}  "
          f"median={np.median(dc_roi):.0f}  "
          f"(threshold = {SEED_DENSITY_THRESHOLD * box_area:.0f})")
    print(f"    is_dense (det_count > thr) in ROI:      "
          f"{_pct(int(is_dense[az_lo:az_hi, rg_lo:rg_hi].sum()), roi_size)}")
    print(f"    is_peak (amp == max in window) in ROI:  "
          f"{_pct(int(is_peak[az_lo:az_hi, rg_lo:rg_hi].sum()), roi_size)}")
    seed_roi = boundary_box[az_lo:az_hi, rg_lo:rg_hi]
    n_seed = int(seed_roi.sum())
    print(f"    SEEDS (all three) in ROI:               "
          f"{_pct(n_seed, roi_size)}")
    if n_seed:
        ys, xs = np.where(seed_roi)
        for yy, xx in zip(ys[:10], xs[:10]):
            print(f"        seed @ az={az_lo + yy}, rg={rg_lo + xx}, "
                  f"amp={amp[az_lo + yy, rg_lo + xx]:.3g}, "
                  f"det_count={det_count[az_lo + yy, rg_lo + xx]:.0f}")
    print()

    # ------------------------------------------------------------------
    # 4. growth + NMS — but only if we got at least one seed.
    # ------------------------------------------------------------------
    peaks_yx = np.argwhere(boundary_box)
    print(f"[4] global seeds: {len(peaks_yx)}")
    boxes_yxhw = grow_and_recenter_boxes(
        peaks_yx,
        mask=mask_full,
        amp=amp,
        initial_hw=(N_TARGET_AZIMUTH_WIDTH, N_TARGET_RANGE_LENGTH),
        az_step=5,
        rg_step=1,
        density_threshold=GROW_DENSITY_THRESHOLD,
        max_h=2000,                                                      # az cap (main pipeline)
        max_w=int(MAX_LENGTH_OF_TARGET / (RANGE_SPACING / NUMBER_OF_RANGE_LOOKS)),
    )
    print(f"    grown boxes: {len(boxes_yxhw)}")

    # Boxes whose centre lies inside the ROI.
    if len(boxes_yxhw):
        in_roi = (
            (boxes_yxhw[:, 0] >= az_lo) & (boxes_yxhw[:, 0] < az_hi)
            & (boxes_yxhw[:, 1] >= rg_lo) & (boxes_yxhw[:, 1] < rg_hi)
        )
        idx_pre_nms = np.where(in_roi)[0]
        print(f"    boxes with centre in ROI (pre-NMS): {len(idx_pre_nms)}")
        for i in idx_pre_nms[:10]:
            yc, xc, h, w = boxes_yxhw[i]
            print(f"        box {i}: centre=({yc},{xc}) "
                  f"h×w={h}×{w}  amp={amp[yc, xc]:.3g}")

        scores = amp[boxes_yxhw[:, 0], boxes_yxhw[:, 1]]
        keep1 = nms_boxes(boxes_yxhw, scores, iou_thresh=NMS_IOU_THRESH)
        in_roi_post1 = np.intersect1d(keep1, idx_pre_nms)
        print(f"    survives IoU NMS (>{NMS_IOU_THRESH}):     "
              f"{len(in_roi_post1)} of {len(idx_pre_nms)}")

        boxes_post1 = boxes_yxhw[keep1]
        scores_post1 = scores[keep1]
        az_spacing_m = RANGE_SPACING                                # = azimuth_spacing
        rg_spacing_m = RANGE_SPACING * NUMBER_OF_RANGE_LOOKS
        keep2 = nms_by_centre_distance(
            boxes_post1, scores_post1,
            az_spacing_m=az_spacing_m,
            rg_spacing_m=rg_spacing_m,
            max_dist_m=MAX_LENGTH_OF_TARGET,
        )
        # Map keep2 (indices into post1) → original indices.
        kept_orig = keep1[keep2]
        in_roi_post2 = np.intersect1d(kept_orig, idx_pre_nms)
        print(f"    survives centre-dist NMS "
              f"(Δ<{MAX_LENGTH_OF_TARGET} m, az={az_spacing_m} m/px, "
              f"rg={rg_spacing_m} m/px): "
              f"{len(in_roi_post2)} of {len(idx_pre_nms)}")
        if len(in_roi_post2):
            for i in in_roi_post2[:5]:
                yc, xc, h, w = boxes_yxhw[i]
                print(f"        post-NMS box: centre=({yc},{xc}) "
                      f"h×w={h}×{w}  amp={amp[yc, xc]:.3g}")

            # Bridge merge stage.
            boxes_post2 = boxes_yxhw[kept_orig]
            boxes_post3 = merge_by_strip_amplitude(
                boxes_post2, amp_raw,
                az_spacing_m=az_spacing_m,
                rg_spacing_m=rg_spacing_m,
                bridge_strength=0.7,
                max_rg_offset_m=20.0,
                max_az_gap_m=2000.0,
            )
            in_roi3 = (
                (boxes_post3[:, 0] >= az_lo) & (boxes_post3[:, 0] < az_hi)
                & (boxes_post3[:, 1] >= rg_lo) & (boxes_post3[:, 1] < rg_hi)
            )
            print(f"    survives bridge merge: "
                  f"{int(in_roi3.sum())} (was {len(in_roi_post2)} before merge, "
                  f"globally {len(boxes_post3)} ↩ {len(boxes_post2)})")
            for yc, xc, h, w in boxes_post3[in_roi3][:5]:
                print(f"        FINAL box: centre=({yc},{xc}) h×w={h}×{w}  "
                      f"amp_raw[centre]={amp_raw[yc, xc]:.3g}")
        else:
            # Why was every ROI box killed by NMS? Find the suppressor.
            for i in idx_pre_nms:
                yc_i, xc_i, h_i, w_i = boxes_yxhw[i]
                amp_i = amp[yc_i, xc_i]
                for j in keep1:
                    if j == i:
                        continue
                    yc_j, xc_j, h_j, w_j = boxes_yxhw[j]
                    dy_m = abs(int(yc_i) - int(yc_j)) * az_spacing_m
                    dx_m = abs(int(xc_i) - int(xc_j)) * rg_spacing_m
                    d_m = (dy_m * dy_m + dx_m * dx_m) ** 0.5
                    if d_m < MAX_LENGTH_OF_TARGET and amp[yc_j, xc_j] >= amp_i:
                        print(f"        ROI box {i} (amp={amp_i:.3g}) suppressed by "
                              f"box {j} centre=({yc_j},{xc_j}) "
                              f"amp={amp[yc_j, xc_j]:.3g} "
                              f"(d={d_m:.1f} m: Δaz={dy_m:.0f} m, Δrg={dx_m:.0f} m)")
                        break
    print()

    # ------------------------------------------------------------------
    # 5. summary verdict
    # ------------------------------------------------------------------
    print("[5] SUMMARY")
    if n_pass == 0:
        print("    -> ROI dropped at the CoV gate (mask_full all zero).")
    elif n_seed == 0:
        print("    -> ROI passed CoV but no seed peak inside ROI.")
        print("       Likely cause: density (det_count) below threshold")
        print("       OR no pixel is the local amp max in its (200,3) box.")
    elif len(boxes_yxhw) == 0:
        print("    -> seeds existed but produced no grown box (shouldn't happen).")
    else:
        print("    -> seeds and boxes exist; check NMS lines above.")


if __name__ == "__main__":
    main()
