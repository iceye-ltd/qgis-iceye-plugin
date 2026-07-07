"""Run `shear_averaging.main()` up to line 4561-4562 and visualise the result.

Reproduces ONLY the head of :func:`shear_averaging.main` — from the loader
block through the ``cluster_targets`` call at lines 4547-4561 — and then
renders the sub-aperture mean amplitude and the cluster boxes exactly the
way :mod:`scripts.view_subaps_cov_baseline` does.

Nothing downstream of ``cluster_result`` is executed (no
``grow_and_recenter_boxes``, no NMS, no per-box refocus, no motion filter).

Two fast paths for the SLC → ``sub_mean`` / ``sub_var`` step:

  * ``<something>_subapertures.npz`` produced by
    ``scripts/save_subapertures.py`` is auto-detected: the cached
    ``sub_mean`` / ``sub_var`` and grid metadata are loaded directly and
    the range Taylor window / sub-aperture stack are NOT recomputed.
  * A raw SLC (``.tif`` / ``.tiff`` + sidecar ``.json``, ``.npz`` bundle,
    or ``.npy`` + ``--metadata-from``) is processed with the same
    ``apply_range_window`` → ``degrade_range_resolution_range_sum`` →
    ``compute_subapertures`` recipe as :func:`shear_averaging.main`.

Figures produced::

    (a) log10(sub_mean)         — Doppler-mean amplitude
    (b) log10(cov²)             — sub_var / (sub_mean² + eps), with the
                                  applied CoV threshold overlaid
    (c) CoV mask (mask_filt)    — CoV gate after isolated-pixel filter
    (d) mask_filt + peaks + boxes — one colour per cluster

Plus a standalone ``clusters-on-mean`` figure — the panel (d) boxes drawn
on top of ``log10(sub_mean)`` so the user can visually check whether each
cluster encloses a real amplitude signature.

Nothing in ``shear_averaging.py`` is modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np

_SHOW_FIGURES: bool = "--show" in sys.argv
if not _SHOW_FIGURES:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shear_averaging import (  # noqa: E402
    _ICEYE_JSON_FIELDS,
    _iso_duration_seconds,
    _load_iceye_sidecar_metadata,
    _load_slc_from_tiff,
    apply_range_window,
    cluster_targets,
    compute_subapertures,
    degrade_range_resolution_range_sum,
)


# --------------------------------------------------------------------------- #
# Input loading
# --------------------------------------------------------------------------- #
def _looks_like_subaps_cache(path: Path) -> bool:
    """`True` iff `path` is a ``*_subapertures.npz`` produced by
    ``save_subapertures.py`` (has `sub_mean`, `sub_var`, and grid
    metadata). Anything else — including the ``.npz`` SLC bundle used
    by shear_averaging.main() — returns `False`.
    """
    if path.suffix.lower() != ".npz":
        return False
    try:
        with np.load(path, allow_pickle=False) as arch:
            keys = set(arch.files)
    except Exception:
        return False
    required = {
        "sub_mean",
        "sub_var",
        "N_subaperture",
        "Number_of_Range_Looks",
        "az_m_per_px_s_degraded",
        "rg_m_per_px_s_degraded",
    }
    return required.issubset(keys)


def _load_subaps_cache(path: Path) -> dict:
    """Load a ``save_subapertures.py`` cache into the same fields the
    fresh pipeline produces (``sub_mean``, ``sub_var``, spacings, N_sub,
    N_range_looks).
    """
    with np.load(path, allow_pickle=False) as arch:
        return {
            "sub_mean": np.asarray(arch["sub_mean"], dtype=np.float64),
            "sub_var": np.asarray(arch["sub_var"], dtype=np.float64),
            "N_subaperture": int(arch["N_subaperture"]),
            "Number_of_Range_Looks": int(arch["Number_of_Range_Looks"]),
            "azimuth_spacing": float(arch["az_m_per_px_s_degraded"]),
            "rg_m_per_px_degraded": float(arch["rg_m_per_px_s_degraded"]),
        }


def _load_slc_and_metadata(
    path: Path,
    metadata_from: Path | None,
) -> tuple[np.ndarray, dict | None]:
    """Return ``(slc_complex, iceye_md)`` mirroring the loader block of
    :func:`shear_averaging.main` (lines 4348-4465).

    ``iceye_md`` is ``None`` for ``.npy`` inputs without ``--metadata-from``.
    """
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        sidecar_json = path.with_suffix(".json")
        if not sidecar_json.exists():
            raise FileNotFoundError(
                f"GeoTIFF input {path.name} requires a sidecar metadata "
                f"JSON at {sidecar_json}, but none was found."
            )
        iceye_md = _load_iceye_sidecar_metadata(sidecar_json)
        left = iceye_md["sar_observation_direction"].lower() == "left"
        s = _load_slc_from_tiff(path, left=left)
        print(
            f"Loaded: {path.name}  shape={s.shape}  dtype={s.dtype}  "
            f"(left-look={left}, sidecar={sidecar_json.name})"
        )
        return s, iceye_md

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as arch:
            if "data" not in arch.files:
                raise KeyError(
                    f".npz bundle {path} is missing the 'data' array."
                )
            s = np.ascontiguousarray(arch["data"])
            missing = [k for k in _ICEYE_JSON_FIELDS if k not in arch.files]
            if missing:
                raise KeyError(
                    f".npz bundle {path} is missing metadata keys: "
                    f"{missing}. Expected all of {_ICEYE_JSON_FIELDS}."
                )
            iceye_md = {
                "sar_resolution_range": float(arch["sar_resolution_range"]),
                "sar_resolution_azimuth": float(arch["sar_resolution_azimuth"]),
                "sar_pixel_spacing_range": float(arch["sar_pixel_spacing_range"]),
                "sar_pixel_spacing_azimuth": float(arch["sar_pixel_spacing_azimuth"]),
                "iceye_acquisition_prf": float(arch["iceye_acquisition_prf"]),
                "start_datetime": str(arch["start_datetime"]),
                "end_datetime": str(arch["end_datetime"]),
            }
        print(
            f"Loaded: {path.name}  shape={s.shape}  dtype={s.dtype}  "
            "(bundled ICEYE metadata)"
        )
        return s, iceye_md

    if suffix == ".npy":
        s = np.load(path)
        print(f"Loaded: {path.name}  shape={s.shape}  dtype={s.dtype}")
        iceye_md: dict | None = None
        if metadata_from is not None:
            md_suffix = metadata_from.suffix.lower()
            if md_suffix in (".tif", ".tiff"):
                sidecar_json = metadata_from.with_suffix(".json")
            elif md_suffix == ".json":
                sidecar_json = metadata_from
            else:
                raise ValueError(
                    f"--metadata-from {metadata_from!r} must point at a "
                    ".tif / .tiff (its sidecar `<stem>.json` is loaded) "
                    "or directly at a .json file."
                )
            if not sidecar_json.exists():
                raise FileNotFoundError(
                    f"--metadata-from points at {metadata_from} but the "
                    f"sidecar JSON {sidecar_json} does not exist."
                )
            iceye_md = _load_iceye_sidecar_metadata(sidecar_json)
            print(f"  metadata sidecar: {sidecar_json}")
        return s, iceye_md

    raise ValueError(
        f"Unsupported input extension {suffix!r} for {path}; expected "
        ".npy, .npz, .tif or .tiff (or a *_subapertures.npz cache)."
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _clip_range(x: np.ndarray, lo_p: float = 2.0, hi_p: float = 99.5) -> tuple[float, float]:
    """Robust percentile-based (vmin, vmax) for `imshow`."""
    return float(np.percentile(x, lo_p)), float(np.percentile(x, hi_p))


def _draw_boxes(
    ax,
    background: np.ndarray,
    bg_cmap: str,
    bg_vmin: float | None,
    bg_vmax: float | None,
    aspect,
    boxes: np.ndarray,
    labels: np.ndarray,
    peaks_yx: np.ndarray,
    title: str,
    seed_peaks_yx: np.ndarray | None = None,
) -> None:
    """Mirror ``view_subaps_cov_baseline._draw_boxes``: background image
    with per-cluster colour peaks + box outlines.
    """
    ax.imshow(
        background,
        cmap=bg_cmap,
        aspect=aspect,
        vmin=bg_vmin,
        vmax=bg_vmax,
        interpolation="nearest",
    )
    n_local = int(labels.max()) + 1 if len(labels) else 0
    if n_local:
        cmap_cl = plt.get_cmap("tab20")
        for c in range(n_local):
            colour = cmap_cl(c % 20)
            sel = labels == c
            ax.scatter(
                peaks_yx[sel, 1], peaks_yx[sel, 0],
                s=10, c=[colour], edgecolor="none",
            )
            if c < len(boxes):
                y_lo, y_hi, x_lo, x_hi = boxes[c]
                ax.add_patch(Rectangle(
                    (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                    fill=False, edgecolor=colour, linewidth=0.7,
                ))
    elif len(peaks_yx):
        ax.scatter(
            peaks_yx[:, 1], peaks_yx[:, 0],
            s=10, c="red", marker="x", linewidths=0.7,
        )
    if seed_peaks_yx is not None and seed_peaks_yx.size:
        ax.scatter(
            seed_peaks_yx[:, 1], seed_peaks_yx[:, 0],
            s=25, facecolors="none", edgecolors="white",
            linewidths=0.8, marker="o",
        )
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("range pixel (s_degraded)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", type=Path,
        help="ICEYE SLC input: .tif (+ sidecar .json), .npz bundle, .npy "
             "(with --metadata-from), OR a *_subapertures.npz cache "
             "produced by scripts/save_subapertures.py (fast path — "
             "skips the Taylor window / sub-aperture recomputation).",
    )
    parser.add_argument(
        "--metadata-from", type=Path, default=None, dest="metadata_from",
        help="Path to an ICEYE .tif or its sidecar .json. Required for "
             ".npy inputs (the historical patches carry no metadata).",
    )
    parser.add_argument(
        "--n-subaperture", type=int, default=16,
        help="Number of azimuth sub-apertures. Same default as "
             "shear_averaging.main(). Ignored when input is a "
             "*_subapertures.npz cache (the cache carries its own value).",
    )
    parser.add_argument(
        "--min-target-size-m", type=float, default=1.0,
        help="Used to derive Number_of_Range_Looks = "
             "int(min_target_size_m / range_spacing). Default: 1.0.",
    )
    parser.add_argument(
        "--range-sll-db", type=float, default=55.0,
        help="Taylor peak-sidelobe level (same default as shear_averaging).",
    )
    parser.add_argument(
        "--range-taylor-nbar", type=int, default=8,
        help="Taylor nbar (same default as shear_averaging).",
    )
    parser.add_argument(
        "--save", type=Path, default=None,
        help="Optional PNG output path for the 4-panel figure. A second "
             "'<stem>_clusters_on_mean<suffix>' is written next to it.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Pop figures via plt.show() (uses a GUI backend). "
             "Detected via sys.argv before pyplot import.",
    )
    args = parser.parse_args()

    # ---- Stage 1: get sub_mean / sub_var + grid metadata -----------------
    # Either from a save_subapertures.py cache (skip Taylor window +
    # sub-aperture recomputation) or from a raw SLC (full recompute,
    # same recipe as shear_averaging.main).
    if _looks_like_subaps_cache(args.path):
        print(f"Detected sub-aperture cache: {args.path}")
        cache = _load_subaps_cache(args.path)
        sub_mean = cache["sub_mean"]
        sub_var = cache["sub_var"]
        N_subaperture = cache["N_subaperture"]
        Number_of_Range_Looks = cache["Number_of_Range_Looks"]
        azimuth_spacing = cache["azimuth_spacing"]
        rg_m_per_px_degraded = cache["rg_m_per_px_degraded"]
        range_spacing = rg_m_per_px_degraded / max(Number_of_Range_Looks, 1)
        print(
            f"  loaded sub_mean / sub_var: shape={sub_mean.shape}, "
            f"N_subaperture={N_subaperture}, "
            f"Number_of_Range_Looks={Number_of_Range_Looks}, "
            f"az={azimuth_spacing} m/px, rg_deg={rg_m_per_px_degraded} m/px"
        )
    else:
        s, iceye_md = _load_slc_and_metadata(args.path, args.metadata_from)
        if iceye_md is None:
            raise SystemExit(
                "Missing pixel-spacing metadata: pass an ICEYE .tif "
                "(with sidecar .json), a .npz bundle, or --metadata-from "
                "<json> when using a bare .npy patch."
            )
        for k in _ICEYE_JSON_FIELDS:
            print(f"    {k} = {iceye_md[k]}")

        # Mirror shear_averaging.main lines 4467-4529
        range_spacing = iceye_md["sar_pixel_spacing_range"]
        azimuth_spacing = iceye_md["sar_pixel_spacing_azimuth"]

        Number_of_Range_Looks = max(
            1, int(args.min_target_size_m / range_spacing)
        )
        N_subaperture = int(args.n_subaperture)
        rg_m_per_px_degraded = range_spacing * Number_of_Range_Looks

        print(
            f"  range_spacing={range_spacing} m, "
            f"azimuth_spacing={azimuth_spacing} m"
        )
        print(
            f"  Number_of_Range_Looks={Number_of_Range_Looks}, "
            f"N_subaperture={N_subaperture}, "
            f"rg_m_per_px_degraded={rg_m_per_px_degraded:g} m"
        )

        s_windowed = apply_range_window(
            s,
            sll_db=args.range_sll_db,
            nbar=args.range_taylor_nbar,
        )
        print(
            f"  range Taylor window applied "
            f"(PSL ≤ -{args.range_sll_db:.0f} dB, nbar={args.range_taylor_nbar})"
        )

        s_degraded, _d_phase = degrade_range_resolution_range_sum(
            s_windowed, Number_of_Range_Looks
        )
        print(f"  s_degraded: shape={s_degraded.shape}, dtype={s_degraded.dtype}")
        del s, s_windowed

        # shear_averaging.main lines 4542-4545
        _subaps_unmasked = compute_subapertures(s_degraded, N_subaperture)
        sub_mean = _subaps_unmasked.mean(axis=0)
        sub_var = _subaps_unmasked.var(axis=0)
        del _subaps_unmasked, s_degraded
        print(
            f"  sub_mean / sub_var: shape={sub_mean.shape}, "
            f"dtype={sub_mean.dtype}"
        )

    # ---- Stage 2: cluster_targets --- shear_averaging.main lines 4547-4561
    cluster_result = cluster_targets(
        sub_mean,
        sub_var,
        az_m_per_px=azimuth_spacing * N_subaperture,
        rg_m_per_px=range_spacing * Number_of_Range_Looks,
        n_subaperture=N_subaperture,
        n_range_looks=Number_of_Range_Looks,
    )
    boxes = cluster_result.boxes
    labels = cluster_result.labels
    peaks_yx = cluster_result.peaks_yx
    seed_peaks_yx = cluster_result.seed_peaks_yx
    mask_filt = cluster_result.mask_filt
    sub_mean = cluster_result.sub_mean
    sub_var = cluster_result.sub_var
    th_applied = cluster_result.threshold
    az_m_dec = cluster_result.az_m_per_px
    rg_m_dec = cluster_result.rg_m_per_px
    n_clusters = int(labels.max()) + 1 if len(labels) else 0

    print(
        f"cluster_targets: {len(peaks_yx)} peaks → {n_clusters} clusters "
        f"→ {len(boxes)} boxes"
    )

    # ---- Stage 3: rendering (mirrors view_subaps_cov_baseline layout) ----
    eps = 1e-12
    cov_sq = sub_var / (sub_mean ** 2 + eps)

    log_mean = np.log10(np.maximum(sub_mean, 1e-6))
    log_cov = np.log10(np.maximum(cov_sq, 1e-6))

    vmin_m, vmax_m = _clip_range(log_mean)
    vmin_c, vmax_c = _clip_range(log_cov)

    n_kept = int(mask_filt.sum())
    n_total = mask_filt.size
    mean_p50 = float(np.median(sub_mean))
    cov_p50 = float(np.median(cov_sq))
    cov_p90 = float(np.percentile(cov_sq, 90.0))
    cov_p99 = float(np.percentile(cov_sq, 99.0))

    aspect = "auto"

    # 4-panel side-by-side ---------------------------------------------------
    fig, axes = plt.subplots(
        1, 4, figsize=(24, 8), constrained_layout=True, sharey=True,
    )
    ax_m, ax_c, ax_k, ax_cl = axes
    fig.suptitle(
        f"sub_mean vs cov² vs mask vs clusters  —  {args.path.name}  "
        f"—  {len(peaks_yx)} peaks → {n_clusters} clusters "
        f"→ {len(boxes)} boxes  —  "
        f"applied CoV² th = {th_applied:.3g}",
        fontsize=10,
    )

    im_m = ax_m.imshow(
        log_mean, cmap="gray", aspect=aspect,
        vmin=vmin_m, vmax=vmax_m, interpolation="nearest",
    )
    ax_m.set_title(f"log10(sub_mean)  [p50 = {mean_p50:.3g}]", fontsize=10)
    ax_m.set_xlabel("range pixel (s_degraded)")
    ax_m.set_ylabel(
        f"sub-aperture-index row  (× {N_subaperture} = azimuth pixel)"
    )
    fig.colorbar(im_m, ax=ax_m, shrink=0.75).set_label("log10(mean amplitude)")

    im_c = ax_c.imshow(
        log_cov, cmap="viridis", aspect=aspect,
        vmin=vmin_c, vmax=vmax_c, interpolation="nearest",
    )
    ax_c.contour(
        log_cov, levels=[np.log10(th_applied)],
        colors="red", linewidths=0.6,
    )
    ax_c.set_title(
        f"log10(cov²)  [p50/p90/p99 = "
        f"{cov_p50:.2g}/{cov_p90:.2g}/{cov_p99:.2g}]",
        fontsize=10,
    )
    ax_c.set_xlabel("range pixel (s_degraded)")
    fig.colorbar(im_c, ax=ax_c, shrink=0.75).set_label("log10(cov²)")

    ax_k.imshow(
        mask_filt, cmap="gray", aspect=aspect,
        vmin=0.0, vmax=1.0, interpolation="nearest",
    )
    ax_k.set_title(
        f"mask_filt (CoV gate + isolated-pixel filter)  "
        f"kept {n_kept}/{n_total} = "
        f"{100 * n_kept / n_total:.2f}%",
        fontsize=10,
    )
    ax_k.set_xlabel("range pixel (s_degraded)")

    _draw_boxes(
        ax_cl,
        background=mask_filt,
        bg_cmap="gray",
        bg_vmin=0.0,
        bg_vmax=1.0,
        aspect=aspect,
        boxes=boxes,
        labels=labels,
        peaks_yx=peaks_yx,
        title=(
            f"cluster_targets(): {len(peaks_yx)} peaks "
            f"→ {n_clusters} clusters → {len(boxes)} boxes"
        ),
        seed_peaks_yx=seed_peaks_yx,
    )

    # Standalone clusters-on-mean figure -------------------------------------
    fig_cl_mean, ax_cl_mean = plt.subplots(
        figsize=(8, 12), constrained_layout=True,
    )
    _draw_boxes(
        ax_cl_mean,
        background=log_mean,
        bg_cmap="gray",
        bg_vmin=vmin_m,
        bg_vmax=vmax_m,
        aspect=aspect,
        boxes=boxes,
        labels=labels,
        peaks_yx=peaks_yx,
        title=(
            f"cluster_targets() on log10(sub_mean): "
            f"{len(peaks_yx)} peaks → {n_clusters} clusters "
            f"→ {len(boxes)} boxes"
        ),
        seed_peaks_yx=seed_peaks_yx,
    )
    ax_cl_mean.set_ylabel(
        f"sub-aperture-index row  (× {N_subaperture} = azimuth pixel)"
    )
    fig_cl_mean.suptitle(f"{args.path.name}", fontsize=10)

    # ---- Save + show -------------------------------------------------------
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved 4-panel figure to {args.save}")
        clusters_mean_path = args.save.with_name(
            f"{args.save.stem}_clusters_on_mean{args.save.suffix}"
        )
        fig_cl_mean.savefig(clusters_mean_path, dpi=150)
        print(f"Saved clusters-on-mean figure to {clusters_mean_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    # `_iso_duration_seconds` is imported for parity with shear_averaging's
    # loader block but is currently unused here (we don't need
    # integration_time because we don't run the smear-max heuristic).
    # Reference it once so `ruff --select F401` stays quiet.
    _ = _iso_duration_seconds
    main()
