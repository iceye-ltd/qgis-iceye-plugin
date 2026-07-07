import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster
from typing import Tuple


def cluster_peaks(
    mask: np.ndarray,                # uint8 (H, W)
    peaks_yx: np.ndarray,            # int64 (K, 2)
    *,
    # --- pixel spacing ---
    az_m_per_px: float = 0.5,
    rg_m_per_px: float = 2.4,
    # --- search radii for centre-to-centre linkage ---
    az_search_m: float = 100.0,
    rg_search_m: float = 5.0,
    # --- per-peak interval sizing ---
    az_min_m: float = 20.0,
    rg_min_m: float = 8.0,
    az_max_m: float = 200.0,
    rg_max_m: float = 60.0,
    # --- profile extraction ---
    profile_thr: float = 0.15,       # fraction of local peak to call "support"
    sigma_az_px: float = 10.0,       # Gaussian sigma for az profile smoothing
    sigma_rg_px: float = 3.0,        # Gaussian sigma for rg profile smoothing
    # --- overlap gate ---
    overlap_frac: float = 0.0,       # 0 = touching OK; 0.5 = must share 50%
    # --- arc-shape gate (post-cluster split) ---
    # Physical intuition: a cluster elongated in azimuth (>arc_az_thr_m)
    # is an azimuth-mover smear ("arc"), which should be thin in range.
    # If both dimensions are large simultaneously, two different targets
    # sharing the same az corridor have been merged — split by range.
    arc_az_thr_m: float = 200.0,     # az extent above which the rg constraint kicks in
    arc_rg_max_m: float = 20.0,      # max rg extent allowed for arc-shaped clusters
    # --- azimuth-gap split (post-cluster split) ---
    # Complete linkage merges peaks whose max pairwise distance is
    # ≤ az_search_m; but the merged cluster can still have big INTERNAL
    # empty regions where no peaks sit (e.g. one clump of peaks near
    # y=27000 and another near y=32000, both within 300 m of each
    # other but with a 200 m dead zone in between). Split any cluster
    # whose sorted peak-y positions contain a gap larger than
    # `max_peak_gap_az_m`. Same idea as the arc-split-by-range, but
    # applied to azimuth without any pre-condition on cluster shape.
    max_peak_gap_az_m: float = 100.0,
    # --- box tightening (post-cluster shrink) ---
    # The union of per-peak intervals is generous — a range-profile
    # smoothed with sigma_rg_px≈3 makes each peak's interval extend
    # well beyond the target's actual footprint, and the cluster box is
    # the union of those. This step shrinks each box to the smallest
    # axis-aligned window containing `tighten_mass_frac` of the
    # mask-detection mass along each axis independently, then unions
    # with the member-peak bounding rectangle so no seed peak sits
    # outside its own cluster box.
    # The shrink is repeated up to `tighten_iters` times: after each
    # cut the marginals are recomputed on the *new* box, so stray
    # sparse detections initially inside the loose union box
    # progressively lose their influence.
    # Iteration converges when the box bounds stop changing, or after
    # `tighten_iters` iterations.
    tighten_boxes: bool = True,
    tighten_mass_frac: float = 0.90,
    tighten_iters: int = 5,
    # --- peakless-tail trim (post-tightening) ---
    # Even after mass-quantile tightening, a cluster's box can still
    # extend well beyond its own peak-y range when a neighbouring
    # target's arc leaks mask signal into the same column strip. The
    # symptom is a box whose "tail" (rows outside the peak bounding
    # rectangle) sits on ANOTHER target — see the two-target arc pair
    # around rows 6100..6600 in the WTW3YQ scene.
    #
    # Root cause: the STEP-1 azimuth profile is computed over a range
    # slab of width `2 * ceil(rg_search_px) + 1` (i.e. the LINKAGE
    # radius, not the target's own footprint). When rg_search_m is
    # e.g. 30 m and the target is ~10 m wide, the slab is 3× too wide,
    # the on-peak density gets diluted 3×, and the 0.15-relative
    # threshold ends up in the noise floor — the contiguous run above
    # threshold then stretches ~170 m past the peak.
    #
    # This step recomputes the row marginal INSIDE the cluster's own
    # peak column range (px_lo..px_hi) — no dilution — and clips the
    # box's y-extent to the contiguous run(s) above
    # `trim_valley_frac × max(row_marg)` that contain at least one
    # cluster peak, unioned with the peak bounding rectangle so no
    # peak sits outside its own box.
    trim_peakless_tails: bool = True,
    trim_valley_frac: float = 0.20,
    trim_sigma_az_px: float = 3.0,
    # --- same-corridor merge (post-linkage) ---
    # Complete-linkage refuses to merge two clusters if ANY single pair
    # of their per-peak azimuth intervals fails to overlap (D set to
    # inf by the overlap gate at STEP 2). Two neighbouring arc-shaped
    # smears from the SAME long rigid body (e.g. two dominant
    # scatterers ~70 m apart on a ~300 m container ship) can therefore
    # end up as two separate clusters — even when their peak-to-peak
    # positions would satisfy the linkage cut. This optional step
    # re-applies the SAME criterion complete-linkage uses (max
    # pairwise normalised Chebyshev distance ≤ merge_max_norm_dist)
    # but bypasses the overlap gate, recovering exactly those merges
    # the gate spuriously blocked. Transitively safe: chaining cannot
    # occur because a peak pair with distance > threshold vetoes any
    # merge involving both endpoints, matching complete-linkage.
    # Iterates until fixed point.
    merge_same_corridor: bool = False,
    merge_max_norm_dist: float = 1.0,
    # --- final hard ceiling (over-size cluster re-split) ---
    enforce_max_size: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster SAR moving-target seed peaks into one label per physical target.

    Mask-only algorithm: uses the binary CoV detection mask and the peak
    coordinates. The amplitude image is intentionally not consumed —
    per-peak support intervals are derived from mask density, and box
    tightening from mask marginals.

    Returns
    -------
    labels : int64 (K,)   cluster ids in [0, N_clusters)
    boxes  : int64 (N_clusters, 4)  rows = (y_lo, y_hi, x_lo, x_hi)
    """
    H, W = mask.shape
    K = len(peaks_yx)

    if K == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.int64)

    mask_f = mask.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Pixel-space thresholds derived from physical parameters
    # ------------------------------------------------------------------ #
    az_search_px = az_search_m / az_m_per_px   # 300 px
    rg_search_px = rg_search_m / rg_m_per_px   # 25 px
    az_min_px    = az_min_m    / az_m_per_px    # 40 px
    rg_min_px    = rg_min_m    / rg_m_per_px    # ~3 px
    az_max_px    = az_max_m    / az_m_per_px    # 400 px
    rg_max_px    = rg_max_m    / rg_m_per_px    # 25 px

    rg_half = int(np.ceil(rg_search_px))        # range slab half-width

    # ------------------------------------------------------------------ #
    # STEP 1 – per-peak support intervals
    # ------------------------------------------------------------------ #
    y_lo = np.empty(K, dtype=np.float32)
    y_hi = np.empty(K, dtype=np.float32)
    x_lo = np.empty(K, dtype=np.float32)
    x_hi = np.empty(K, dtype=np.float32)

    for k, (yp, xp) in enumerate(peaks_yx):
        yp, xp = int(yp), int(xp)

        # --- azimuth profile: mean mask density along a range slab ---
        x0 = max(0, xp - rg_half)
        x1 = min(W, xp + rg_half + 1)

        az_mask_slab  = mask_f[:, x0:x1]          # (H, slab)
        az_profile    = az_mask_slab.mean(axis=1)  # in [0, 1]
        az_profile_sm = gaussian_filter1d(az_profile, sigma=sigma_az_px)

        thr_az = profile_thr * (az_profile_sm[yp] + 1e-9)
        above  = (az_profile_sm >= thr_az).astype(np.int8)

        # Find contiguous run containing yp
        yl, yh = _contiguous_run(above, yp, H)

        # Enforce min / max half-extents symmetrically around yp
        half = max((yh - yl) / 2.0, az_min_px / 2.0)
        half = min(half, az_max_px / 2.0)
        y_lo[k] = max(0,   yp - half)
        y_hi[k] = min(H-1, yp + half)

        # --- range profile: mean mask density along an az slab ---
        az_half = int(np.ceil(az_min_px / 2.0))
        y0 = max(0, yp - az_half)
        y1 = min(H, yp + az_half + 1)

        rg_mask_slab  = mask_f[y0:y1, :]          # (slab, W)
        rg_profile    = rg_mask_slab.mean(axis=0)  # in [0, 1]
        rg_profile_sm = gaussian_filter1d(rg_profile, sigma=sigma_rg_px)

        thr_rg = profile_thr * (rg_profile_sm[xp] + 1e-9)
        above_rg = (rg_profile_sm >= thr_rg).astype(np.int8)

        xl, xh = _contiguous_run(above_rg, xp, W)

        half_rg = max((xh - xl) / 2.0, rg_min_px / 2.0)
        half_rg = min(half_rg, rg_max_px / 2.0)
        x_lo[k] = max(0,   xp - half_rg)
        x_hi[k] = min(W-1, xp + half_rg)

    # ------------------------------------------------------------------ #
    # STEP 2 – pairwise distance matrix (vectorised, Chebyshev normalised)
    # ------------------------------------------------------------------ #
    yc = peaks_yx[:, 0].astype(np.float64)
    xc = peaks_yx[:, 1].astype(np.float64)

    # Normalised centre-to-centre distances
    d_az_norm = np.abs(yc[:, None] - yc[None, :]) / az_search_px   # (K,K)
    d_rg_norm = np.abs(xc[:, None] - xc[None, :]) / rg_search_px

    D = np.maximum(d_az_norm, d_rg_norm)   # Chebyshev in normalised space

    # Overlap gate: set D = inf if intervals don't touch in BOTH axes
    az_overlap = (np.minimum(y_hi[:, None], y_hi[None, :])
                  - np.maximum(y_lo[:, None], y_lo[None, :]))       # (K,K)
    rg_overlap = (np.minimum(x_hi[:, None], x_hi[None, :])
                  - np.maximum(x_lo[:, None], x_lo[None, :]))

    min_az_span = np.minimum(y_hi - y_lo, az_min_px)[:, None] * overlap_frac
    min_rg_span = np.minimum(x_hi - x_lo, rg_min_px)[:, None] * overlap_frac

    no_merge = (az_overlap < min_az_span) | (rg_overlap < min_rg_span)
    D[no_merge] = np.inf
    np.fill_diagonal(D, 0.0)

    # ------------------------------------------------------------------ #
    # STEP 3 – hierarchical clustering (single linkage, cut at 1.0)
    # ------------------------------------------------------------------ #
    if K == 1:
        raw_labels = np.array([0])
    else:
        # scipy linkage needs finite condensed distances; replace inf safely
        D_fin = np.where(np.isinf(D), 1e9, D)
        condensed = squareform(D_fin, checks=False)
        Z = linkage(condensed, method='complete')
        # Cut at 1.0 in normalised space → merge only if within search radii
        raw_labels = fcluster(Z, t=1.0, criterion='distance') - 1  # 0-indexed

    # ------------------------------------------------------------------ #
    # STEP 4 – bounding boxes + optional hard-ceiling re-split
    # ------------------------------------------------------------------ #
    labels, boxes = _build_boxes(
        raw_labels, y_lo, y_hi, x_lo, x_hi, H, W
    )

    # ------------------------------------------------------------------ #
    # STEP 5 – arc-shape split: any cluster with az_span > arc_az_thr_m
    # AND rg_span > arc_rg_max_m is broken up along the range axis (a
    # true azimuth-mover arc is narrow in range; simultaneous width in
    # both axes signals two targets sharing the same az corridor).
    # ------------------------------------------------------------------ #
    labels, boxes = _split_arc_clusters(
        labels, boxes, peaks_yx,
        y_lo, y_hi, x_lo, x_hi, H, W,
        arc_az_thr_px=arc_az_thr_m / az_m_per_px,
        arc_rg_max_px=arc_rg_max_m / rg_m_per_px,
        rg_search_px=rg_search_px,
    )

    labels, boxes = _split_az_gap_clusters(
        labels, boxes, peaks_yx,
        y_lo, y_hi, x_lo, x_hi, H, W,
        max_gap_az_px=max_peak_gap_az_m / az_m_per_px,
    )

    if tighten_boxes:
        boxes = _tighten_boxes_by_mass(
            labels, boxes, mask_f, peaks_yx,
            mass_frac=tighten_mass_frac,
            max_iter=tighten_iters,
        )

    if trim_peakless_tails:
        boxes = _trim_peakless_tails(
            labels, boxes, mask_f, peaks_yx,
            valley_frac=trim_valley_frac,
            sigma_az_px=trim_sigma_az_px,
        )

    if merge_same_corridor:
        labels, boxes = _merge_same_corridor_clusters(
            labels, boxes, mask_f, peaks_yx,
            az_search_px=az_search_px,
            rg_search_px=rg_search_px,
            merge_max_norm_dist=merge_max_norm_dist,
        )
        # Retighten and retrim on the merged boxes so their extents
        # reflect the full new mask footprint.
        if tighten_boxes:
            # Reconstruct per-peak intervals to feed _build_boxes.
            # After merging labels are reassigned; y_lo/y_hi/x_lo/x_hi
            # arrays still map k → per-peak, so we can rebuild boxes
            # from them under the new labels then retighten.
            _, boxes = _build_boxes(labels, y_lo, y_hi, x_lo, x_hi, H, W)
            boxes = _tighten_boxes_by_mass(
                labels, boxes, mask_f, peaks_yx,
                mass_frac=tighten_mass_frac,
                max_iter=tighten_iters,
            )
        if trim_peakless_tails:
            boxes = _trim_peakless_tails(
                labels, boxes, mask_f, peaks_yx,
                valley_frac=trim_valley_frac,
                sigma_az_px=trim_sigma_az_px,
            )

    if enforce_max_size:
        labels, boxes = _enforce_max_size(
            labels, boxes, peaks_yx,
            y_lo, y_hi, x_lo, x_hi,
            az_max_px, rg_max_px,
            Z if K > 1 else None,
        )

    return labels.astype(np.int64), boxes.astype(np.int64)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _contiguous_run(above: np.ndarray, centre: int, length: int):
    """
    Find the start/end of the contiguous run of 1s in `above` that
    contains index `centre`. Returns (lo, hi) inclusive pixel coords.
    If centre itself is 0, returns a single-pixel interval.
    """
    if above[centre] == 0:
        return centre, centre
    lo = centre
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = centre
    while hi < length - 1 and above[hi + 1]:
        hi += 1
    return lo, hi


def _build_boxes(raw_labels, y_lo, y_hi, x_lo, x_hi, H, W):
    """Union of member intervals → bounding box per cluster."""
    n_clusters = raw_labels.max() + 1
    boxes = np.empty((n_clusters, 4), dtype=np.float32)
    boxes[:, 0] =  np.inf   # y_lo
    boxes[:, 1] = -np.inf   # y_hi
    boxes[:, 2] =  np.inf   # x_lo
    boxes[:, 3] = -np.inf   # x_hi

    for c in range(n_clusters):
        sel = raw_labels == c
        boxes[c, 0] = np.clip(y_lo[sel].min(), 0, H - 1)
        boxes[c, 1] = np.clip(y_hi[sel].max(), 0, H - 1)
        boxes[c, 2] = np.clip(x_lo[sel].min(), 0, W - 1)
        boxes[c, 3] = np.clip(x_hi[sel].max(), 0, W - 1)

    return raw_labels, boxes


def _split_az_gap_clusters(labels, boxes, peaks_yx,
                           y_lo, y_hi, x_lo, x_hi, H, W,
                           max_gap_az_px: float):
    """Split any cluster whose sorted peak-y positions contain a gap
    larger than ``max_gap_az_px``.

    The complete-linkage step ensures a cluster's max pairwise
    az-distance is ≤ ``az_search_m``, but says nothing about internal
    gaps. A cluster with peaks at y≈27000 and y≈32000 and nothing in
    between is almost certainly two independent targets that happened
    to sit within one ``az_search_m`` radius of each other; the mass-
    quantile tightening cannot cut such a gap because the box still
    contains mask-weighted signal at the fringes on both sides.
    Splitting on peak-position gaps first lets tightening run on
    each sub-cluster and produce two properly-shrunk boxes instead
    of one wide one straddling the empty region.
    """
    n_clusters = int(labels.max()) + 1
    # Cluster ids that will actually need splitting.
    to_split = []
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        if len(members) < 2:
            continue
        ys = peaks_yx[members, 0]
        order = np.argsort(ys)
        ys_sorted = ys[order]
        gaps = np.diff(ys_sorted)
        if gaps.max() > max_gap_az_px:
            to_split.append(c)
    if not to_split:
        return labels, boxes

    new_labels = labels.copy()
    next_id = int(new_labels.max()) + 1
    for c in to_split:
        members = np.where(labels == c)[0]
        ys = peaks_yx[members, 0].astype(np.int64)
        order = np.argsort(ys)
        m_sorted = members[order]
        ys_sorted = ys[order]
        gaps = np.diff(ys_sorted)
        # cumsum of break flags gives contiguous sub-cluster ids
        # 0, 0, ..., 1, 1, ..., 2, ...
        sub_ids = np.zeros(len(ys_sorted), dtype=np.int64)
        sub_ids[1:] = np.cumsum(gaps > max_gap_az_px)
        # First sub-cluster keeps id `c`; subsequent get fresh ids.
        for sid in np.unique(sub_ids):
            group_members = m_sorted[sub_ids == sid]
            if sid == 0:
                new_labels[group_members] = c
            else:
                new_labels[group_members] = next_id
                next_id += 1

    # Dense-remap and rebuild boxes.
    _, remap = np.unique(new_labels, return_inverse=True)
    _, new_boxes = _build_boxes(remap, y_lo, y_hi, x_lo, x_hi, H, W)
    return remap.astype(np.int64), new_boxes


def _tighten_boxes_by_mass(labels, boxes, mask_f, peaks_yx,
                           mass_frac: float = 0.90,
                           max_iter: int = 5):
    """Iteratively shrink each cluster's bounding box to the axis-marginal
    quantiles of the detection mask.

    For each cluster:
      1. Restrict attention to the current box; the per-pixel weight
         is the mask value itself (1 for detection, 0 for background).
      2. Marginalise onto azimuth and range independently.
      3. Keep the smallest axis-aligned window whose per-axis mass
         fraction is ``mass_frac`` — bounded by the
         ((1 − mass_frac) / 2, 1 − (1 − mass_frac) / 2) quantiles of
         each cumulative marginal (e.g. mass_frac=0.90 → [5th, 95th]).
      4. Union with the member peaks' bounding rectangle so every
         seed peak stays inside its own cluster's box.
      5. Repeat with the new box until bounds stop changing, or
         `max_iter` iterations are reached.

    Iterating matters when the initial (union) box is inflated by
    per-peak-interval slop: sparse stray detections near the box edge
    can pull the marginal outward on the first pass. After one shrink
    they're outside the box entirely, the marginal lands on the true
    target core, and a second pass tightens further. The peak-
    bounding-rectangle lower bound is checked every iteration, so
    the box can never shrink below the member peaks' own extent.

    If a cluster has zero mask mass in its box (the per-peak
    intervals covered blank space), we fall back to the pure peak
    bounding rectangle immediately.
    """
    H, W = mask_f.shape
    n_clusters = int(labels.max()) + 1
    new_boxes = boxes.copy()
    lo_q = (1.0 - mass_frac) / 2.0
    hi_q = 1.0 - lo_q
    for c in range(n_clusters):
        mem = np.where(labels == c)[0]
        if len(mem) == 0:
            continue

        # Peak bounding rectangle — the box must always contain it.
        py_lo, py_hi = int(peaks_yx[mem, 0].min()), int(peaks_yx[mem, 0].max())
        px_lo, px_hi = int(peaks_yx[mem, 1].min()), int(peaks_yx[mem, 1].max())

        y_lo, y_hi, x_lo, x_hi = (int(v) for v in boxes[c])
        if y_hi <= y_lo or x_hi <= x_lo:
            continue

        for _ in range(max_iter):
            weight = mask_f[y_lo:y_hi + 1, x_lo:x_hi + 1]

            if weight.sum() <= 0:
                y_lo, y_hi, x_lo, x_hi = py_lo, py_hi, px_lo, px_hi
                break

            # --- Step 1: shrink range (x) first ------------------------
            # A range-tight window discards clutter columns that would
            # otherwise dilute the azimuth marginal below, so the
            # az-marginal in step 2 is computed on cleaner data.
            col_marg = weight.sum(axis=0)
            cum_c = np.cumsum(col_marg) / col_marg.sum()
            xl_rel = int(np.searchsorted(cum_c, lo_q))
            xh_rel = int(np.searchsorted(cum_c, hi_q))
            new_x_lo = min(x_lo + xl_rel, px_lo)
            new_x_hi = max(x_lo + xh_rel, px_hi)

            # --- Step 2: shrink azimuth (y) using the range-tight sub-window
            weight_x = mask_f[y_lo:y_hi + 1, new_x_lo:new_x_hi + 1]
            if weight_x.sum() <= 0:
                # Range-tight window is empty; keep the range shrink but
                # fall back to the peak-bound rectangle for az.
                new_y_lo, new_y_hi = py_lo, py_hi
            else:
                row_marg = weight_x.sum(axis=1)
                cum_r = np.cumsum(row_marg) / row_marg.sum()
                yl_rel = int(np.searchsorted(cum_r, lo_q))
                yh_rel = int(np.searchsorted(cum_r, hi_q))
                new_y_lo = min(y_lo + yl_rel, py_lo)
                new_y_hi = max(y_lo + yh_rel, py_hi)

            if (new_y_lo == y_lo and new_y_hi == y_hi
                    and new_x_lo == x_lo and new_x_hi == x_hi):
                break
            y_lo, y_hi, x_lo, x_hi = new_y_lo, new_y_hi, new_x_lo, new_x_hi

        new_boxes[c] = (y_lo, y_hi, x_lo, x_hi)
    return new_boxes


def _trim_peakless_tails(labels, boxes, mask_f, peaks_yx,
                         valley_frac: float = 0.20,
                         sigma_az_px: float = 3.0):
    """Trim each box's y-extent to the contiguous mask-marginal run(s)
    that actually contain the cluster's own peaks.

    Marginal source: row-wise sum of ``mask`` inside the current box
    but restricted to the cluster's PEAK column range (``[px_lo, px_hi]``),
    not the box's full column range. This is the key change from
    ``_tighten_boxes_by_mass``: by narrowing the columns to the peaks'
    own footprint we make the marginal a much cleaner target-specific
    signature, immune to dilution by neighbouring bright ridges that
    happen to share the same wide range strip.

    A row is considered "supported" if its smoothed marginal is
    ``>= valley_frac * max(marginal)``. Any tail below that threshold
    that does not contain a peak is discarded. The final box is
    unioned with the peak bounding rectangle so no seed peak can end
    up outside its own cluster box.

    This step fixes the case where two neighbouring targets share an
    azimuth corridor: without it, the per-peak azimuth profile of
    STEP 1 uses a range slab of ``2 * rg_half + 1`` pixels wide, which
    is the linkage radius — typically 3–5× the target's own range
    footprint. The consequent on-peak dilution collapses the
    ``0.15 * profile[yp]`` threshold onto the mask noise floor, and
    the contiguous run above it extends well past the peak into the
    neighbouring target's territory.

    Numerically safe when the marginal is all-zero (falls back to the
    peak bounding rectangle) or when the box has zero height/width
    (no change).
    """
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    if n_clusters == 0:
        return boxes
    new_boxes = boxes.copy()
    for c in range(n_clusters):
        mem = np.where(labels == c)[0]
        if len(mem) == 0:
            continue
        peak_rows = peaks_yx[mem, 0].astype(np.int64)
        peak_cols = peaks_yx[mem, 1].astype(np.int64)
        py_lo, py_hi = int(peak_rows.min()), int(peak_rows.max())
        px_lo, px_hi = int(peak_cols.min()), int(peak_cols.max())

        y_lo, y_hi, x_lo, x_hi = (int(v) for v in boxes[c])
        if y_hi <= y_lo or x_hi <= x_lo:
            continue

        # Row marginal computed over the cluster's own peak column
        # range only. If the peak column range collapses to a single
        # pixel, widen it by 1 on each side so gaussian_filter1d has
        # something to work with.
        col_lo = max(0, px_lo)
        col_hi = min(mask_f.shape[1] - 1, px_hi)
        if col_hi == col_lo:
            col_lo = max(0, col_lo - 1)
            col_hi = min(mask_f.shape[1] - 1, col_hi + 1)

        row_marg = mask_f[y_lo:y_hi + 1, col_lo:col_hi + 1].sum(axis=1)
        row_marg_sm = gaussian_filter1d(row_marg.astype(np.float64), sigma=sigma_az_px)
        m = float(row_marg_sm.max())
        if m <= 0:
            # No mask mass in the peak-column strip — degenerate case
            # (should not happen since peaks are on mask==1 pixels), so
            # fall back to the peak bounding rectangle.
            new_boxes[c] = (py_lo, py_hi, x_lo, x_hi)
            continue

        thr = valley_frac * m
        above = row_marg_sm >= thr
        # Absolute row indices of "supported" rows in the current box.
        rows_abs = np.arange(y_lo, y_hi + 1)
        above_rows = rows_abs[above]
        if above_rows.size == 0:
            new_boxes[c] = (py_lo, py_hi, x_lo, x_hi)
            continue

        # Find contiguous runs in above_rows; keep only the runs that
        # contain at least one cluster peak. Union them together in
        # case multiple peaks sit in disjoint runs (rare but possible).
        gaps = np.where(np.diff(above_rows) > 1)[0]
        run_starts = np.concatenate(([0], gaps + 1))
        run_ends = np.concatenate((gaps, [len(above_rows) - 1]))
        keep_lo = None
        keep_hi = None
        for s, e in zip(run_starts, run_ends):
            r_lo, r_hi = int(above_rows[s]), int(above_rows[e])
            if np.any((peak_rows >= r_lo) & (peak_rows <= r_hi)):
                if keep_lo is None:
                    keep_lo, keep_hi = r_lo, r_hi
                else:
                    keep_lo = min(keep_lo, r_lo)
                    keep_hi = max(keep_hi, r_hi)
        if keep_lo is None:
            # No run contains a peak — extremely narrow-mass target;
            # keep only the peak bounding rectangle.
            new_boxes[c] = (py_lo, py_hi, x_lo, x_hi)
            continue

        # Union with peak bounding rectangle so no peak sits outside.
        new_y_lo = min(keep_lo, py_lo)
        new_y_hi = max(keep_hi, py_hi)
        new_boxes[c] = (new_y_lo, new_y_hi, x_lo, x_hi)
    return new_boxes


def _merge_same_corridor_clusters(labels, boxes, mask_f, peaks_yx,
                                  az_search_px: float,
                                  rg_search_px: float,
                                  merge_max_norm_dist: float = 1.0):
    """Merge cluster pairs that complete-linkage WOULD have merged if
    the STEP-2 overlap gate hadn't set their pairwise distances to inf.

    Criterion (identical to what ``fcluster(Z, t=1.0)`` uses, but
    computed on peak positions only, without the overlap gate):

        max_{i∈A, j∈B}  max(|Δy|/az_search_px, |Δx|/rg_search_px)
                 ≤  merge_max_norm_dist

    In other words: merge two clusters iff their MAX pairwise
    normalised-Chebyshev distance is ≤ the linkage cut threshold.

    This is transitively safe: if A+B merge (max=0.4) and C's max
    pairwise to A+B exceeds 1.0 (say via A-C=1.5), then C does not
    join, and no chaining occurs. The peak column range acts as an
    implicit "range corridor" test since the range term of Chebyshev
    is normalised by ``rg_search_px``.

    Iterates until fixed point. Labels are dense-remapped; boxes are
    rebuilt from the union of member peaks' bounding rectangles.
    Caller is expected to re-run tightening for mass-based extents.
    """
    H, W = mask_f.shape
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    if n_clusters <= 1:
        return labels, boxes

    def _cluster_bounds(lbls, n):
        py_lo = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        py_hi = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
        px_lo = np.full(n, np.iinfo(np.int64).max, dtype=np.int64)
        px_hi = np.full(n, np.iinfo(np.int64).min, dtype=np.int64)
        peaks_by_c: list[np.ndarray] = [
            np.empty((0, 2), dtype=np.int64) for _ in range(n)
        ]
        for c in range(n):
            mem = np.where(lbls == c)[0]
            if len(mem) == 0:
                continue
            pk = peaks_yx[mem]
            py_lo[c] = pk[:, 0].min()
            py_hi[c] = pk[:, 0].max()
            px_lo[c] = pk[:, 1].min()
            px_hi[c] = pk[:, 1].max()
            peaks_by_c[c] = pk
        return py_lo, py_hi, px_lo, px_hi, peaks_by_c

    # Fast bounding-box max-Chebyshev in normalised space:
    #   max over pairs |Δy|/az = (max_y_span) / az
    # where max_y_span between clusters i and j is
    #   max(py_hi[i], py_hi[j]) - min(py_lo[i], py_lo[j])
    # (i.e. the max distance between any peak of i and any peak of j
    # along y). Same for x. If this bbox-derived MAX is > threshold,
    # no need to enumerate the (up to |A|*|B|) pairs.
    def _bbox_max_norm_dist(i, j, py_lo, py_hi, px_lo, px_hi):
        d_az = (max(py_hi[i], py_hi[j]) - min(py_lo[i], py_lo[j])) / az_search_px
        d_rg = (max(px_hi[i], px_hi[j]) - min(px_lo[i], px_lo[j])) / rg_search_px
        return max(d_az, d_rg)

    lbls = labels.copy().astype(np.int64)
    n = n_clusters
    n_merges = 0
    while True:
        py_lo, py_hi, px_lo, px_hi, peaks_by_c = _cluster_bounds(lbls, n)

        # Iterate pairs by ascending bbox-max-distance. Sorted once
        # per pass; distances are RE-CHECKED against current bboxes
        # before every actual merge, so a pair whose union grew above
        # the threshold via an earlier merge in this pass is skipped
        # (prevents chaining that would blow past the cut).
        pair_dists = []
        for i in range(n):
            if len(peaks_by_c[i]) == 0:
                continue
            for j in range(i + 1, n):
                if len(peaks_by_c[j]) == 0:
                    continue
                d_bbox = _bbox_max_norm_dist(i, j, py_lo, py_hi, px_lo, px_hi)
                if d_bbox <= merge_max_norm_dist:
                    pair_dists.append((d_bbox, i, j))
        if not pair_dists:
            break
        pair_dists.sort()
        alive = np.ones(n, dtype=bool)
        merged_any = False
        for _d0, i, j in pair_dists:
            if not (alive[i] and alive[j]):
                continue
            # Re-check with CURRENT (possibly grown) bboxes — the
            # sorted `_d0` was computed pre-pass and may now be stale.
            d_now = _bbox_max_norm_dist(i, j, py_lo, py_hi, px_lo, px_hi)
            if d_now > merge_max_norm_dist:
                continue
            lbls[lbls == j] = i
            alive[j] = False
            merged_any = True
            n_merges += 1
            peaks_by_c[i] = np.vstack([peaks_by_c[i], peaks_by_c[j]])
            peaks_by_c[j] = np.empty((0, 2), dtype=np.int64)
            py_lo[i] = peaks_by_c[i][:, 0].min()
            py_hi[i] = peaks_by_c[i][:, 0].max()
            px_lo[i] = peaks_by_c[i][:, 1].min()
            px_hi[i] = peaks_by_c[i][:, 1].max()

        if not merged_any:
            break
        _, lbls = np.unique(lbls, return_inverse=True)
        lbls = lbls.astype(np.int64)
        n = int(lbls.max()) + 1

    py_lo, py_hi, px_lo, px_hi, peaks_by_c = _cluster_bounds(lbls, n)
    new_boxes = np.zeros((n, 4), dtype=np.float32)
    for c in range(n):
        if len(peaks_by_c[c]) == 0:
            continue
        new_boxes[c, 0] = np.clip(py_lo[c], 0, H - 1)
        new_boxes[c, 1] = np.clip(py_hi[c], 0, H - 1)
        new_boxes[c, 2] = np.clip(px_lo[c], 0, W - 1)
        new_boxes[c, 3] = np.clip(px_hi[c], 0, W - 1)
    return lbls, new_boxes


def _split_arc_clusters(labels, boxes, peaks_yx,
                        y_lo, y_hi, x_lo, x_hi, H, W,
                        arc_az_thr_px: float,
                        arc_rg_max_px: float,
                        rg_search_px: float):
    """Split any cluster that is elongated in az AND wide in rg.

    A cluster whose az extent exceeds `arc_az_thr_px` should be interpreted
    as an azimuth-mover smear (arc). Physically, an arc has a narrow
    range signature, so if the same cluster is also wide in rg
    (>arc_rg_max_px) two different targets have almost certainly been
    merged. We split such clusters along the range axis: peaks whose
    range coordinates differ by more than `rg_search_px` are re-labelled
    into separate sub-clusters (gap-based 1-D grouping on sorted x).

    Boxes are rebuilt from the updated labels.
    """
    # Trigger on the actual peak-position spread, not the (possibly
    # inflated) union of per-peak intervals: the box extent is bloated
    # by the range-profile smoothing, so a cluster whose PEAKS only
    # span 15 px can still yield a 68 px union box and spuriously
    # trigger. What we care about physically is where the seed peaks
    # actually sit.
    n_clusters = int(labels.max()) + 1
    az_peak_span = np.zeros(n_clusters, dtype=np.float64)
    rg_peak_span = np.zeros(n_clusters, dtype=np.float64)
    for c in range(n_clusters):
        mem = np.where(labels == c)[0]
        if len(mem) == 0:
            continue
        ys = peaks_yx[mem, 0]
        xs = peaks_yx[mem, 1]
        az_peak_span[c] = ys.max() - ys.min()
        rg_peak_span[c] = xs.max() - xs.min()
    oversize = np.where((az_peak_span > arc_az_thr_px)
                        & (rg_peak_span > arc_rg_max_px))[0]
    if len(oversize) == 0:
        return labels, boxes

    new_labels = labels.copy()
    next_id = int(new_labels.max()) + 1

    for c in oversize:
        members = np.where(labels == c)[0]
        if len(members) < 2:
            continue
        xs = peaks_yx[members, 1].astype(np.int64)
        order = np.argsort(xs)
        m_sorted = members[order]
        xs_sorted = xs[order]
        gaps = np.diff(xs_sorted)
        # Break wherever the gap between adjacent (in range) peaks
        # exceeds the linkage radius. Cumsum turns break-flags into
        # sub-cluster ids (0, 0, ..., 1, 1, ..., 2, ...).
        sub_ids = np.zeros(len(xs_sorted), dtype=np.int64)
        sub_ids[1:] = np.cumsum(gaps > rg_search_px)
        # First sub-cluster keeps the original id; subsequent groups
        # get fresh ids.
        for sid in np.unique(sub_ids):
            group_members = m_sorted[sub_ids == sid]
            if sid == 0:
                new_labels[group_members] = c
            else:
                new_labels[group_members] = next_id
                next_id += 1

    # Rebuild boxes over the (possibly larger) label set. Dense-remap
    # first so ids are contiguous in [0, N).
    uniq, remap = np.unique(new_labels, return_inverse=True)
    _, new_boxes = _build_boxes(remap, y_lo, y_hi, x_lo, x_hi, H, W)
    return remap.astype(np.int64), new_boxes


def _enforce_max_size(labels, boxes, peaks_yx,
                      y_lo, y_hi, x_lo, x_hi,
                      az_max_px, rg_max_px, Z):
    """
    Re-split any cluster whose merged bounding box exceeds the hard ceiling
    by re-cutting the linkage sub-tree at a tighter threshold.
    Simple fallback: assign each over-size-cluster member its own label.
    """
    if Z is None:
        return labels, boxes

    az_size = boxes[:, 1] - boxes[:, 0]
    rg_size = boxes[:, 3] - boxes[:, 2]
    oversize = np.where((az_size > az_max_px) | (rg_size > rg_max_px))[0]

    if len(oversize) == 0:
        return labels, boxes

    next_id = labels.max() + 1
    new_labels = labels.copy()

    for c in oversize:
        members = np.where(labels == c)[0]
        # Give each member its own id (conservative: prefer over-splitting)
        for m in members:
            new_labels[m] = next_id
            next_id += 1

    # Rebuild boxes for all (now possibly more) clusters
    _, new_boxes = _build_boxes(
        new_labels, y_lo, y_hi, x_lo, x_hi,
        int(boxes[:, 1].max()) + 1,
        int(boxes[:, 3].max()) + 1,
    )
    return new_labels, new_boxes