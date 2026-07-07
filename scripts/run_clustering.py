"""Run cluster_peaks() on the shear_debug.npz produced by shear_averaging.py
and save (and optionally display) a 3-panel figure showing the raw mask +
boundary peaks, the density-filtered mask, and the clusters + boxes.

Usage
-----
    # Save `<npz_stem>_clustering.png` next to the input npz (default):
    python scripts/run_clustering.py path/to/shear_WTW3YQ.npz

    # Force a specific output path:
    python scripts/run_clustering.py shear_WTW3YQ.npz --save /tmp/clust.png

    # Also open an interactive window:
    python scripts/run_clustering.py shear_WTW3YQ.npz --show

    # Skip the PNG write (must be combined with --show):
    python scripts/run_clustering.py shear_WTW3YQ.npz --no-save --show

Defaults to
    /home/odogan/Desktop/ship_focusing/6544394/shear_debug.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from clustering import cluster_peaks  # noqa: E402


DEFAULT_NPZ = Path(
    "/home/odogan/Desktop/ship_focusing/6544394/shear_debug.npz"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "npz", nargs="?", type=Path, default=DEFAULT_NPZ,
        help=f"Path to the shear_averaging.py `.npz` output. "
             f"Default: {DEFAULT_NPZ}",
    )
    p.add_argument(
        "--save", type=Path, default=None,
        help="PNG output path. Default: `<npz_stem>_clustering.png` next "
             "to the input npz.",
    )
    p.add_argument(
        "--no-save", dest="no_save", action="store_true",
        help="Do NOT write a PNG. Only meaningful with --show.",
    )
    p.add_argument(
        "--show", action="store_true",
        help="Open the interactive matplotlib window in addition to "
             "(or instead of) saving. Off by default because the 3-panel "
             "figure at s_degraded resolution can be slow to render on "
             "full ICEYE scenes.",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="PNG DPI (only used when saving). Default: 150.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    npz_path = args.npz.resolve()
    if not npz_path.exists():
        print(f"ERROR: {npz_path} does not exist", file=sys.stderr)
        sys.exit(1)

    # Decide whether to write a PNG. Default is yes; --no-save disables.
    if args.no_save:
        save_path: Path | None = None
    else:
        save_path = (
            args.save.resolve() if args.save is not None
            else npz_path.with_name(f"{npz_path.stem}_clustering.png")
        )

    # Force a headless backend when we are not going to show a window;
    # this must happen BEFORE `import matplotlib.pyplot` anywhere in
    # the process, so we do it here at CLI parse time.
    if not args.show:
        import matplotlib
        matplotlib.use("Agg")

    print(f"Loading {npz_path}")
    d = np.load(npz_path)
    mask = d["mask_full"]                         # int8, (H, W)
    peaks = d["boundary_box_peaks_yx"]            # int64, (K, 2)

    H, W = mask.shape
    K = len(peaks)
    print(f"  mask_full            : {mask.shape} {mask.dtype}, "
          f"density = {mask.mean():.4f}")
    print(f"  peaks                : K = {K}")

    # Pixel spacings on the s_degraded grid (as stored by shear_averaging).
    # Fall back to inferring from the array-shape ratio for older npz
    # files that pre-date the explicit save.
    if "az_m_per_px_s_degraded" in d.files:
        az_m_per_px = float(d["az_m_per_px_s_degraded"])
        rg_m_per_px = float(d["rg_m_per_px_s_degraded"])
    else:
        az_m_per_px = 0.043618
        rg_m_per_px = 0.242602 * 4
        print("  (spacings not in npz; using hard-coded fallback)")

    # Run clustering. enforce_max_size=False sidesteps a bug that
    # would otherwise crash on sparse label sets after re-splitting.
    print("\nClustering ...")
    t0 = time.time()
    labels, boxes = cluster_peaks(
        mask, peaks,
        az_m_per_px=az_m_per_px,
        rg_m_per_px=rg_m_per_px,
        enforce_max_size=False,
    )
    print(f"  done in {time.time() - t0:.2f} s")

    n_clusters = len(np.unique(labels))
    sizes = np.bincount(labels)
    h = boxes[:, 1] - boxes[:, 0]
    w = boxes[:, 3] - boxes[:, 2]
    print(f"\nClusters: {n_clusters}")
    print(f"  peaks/cluster    : min={sizes.min()}, "
          f"median={int(np.median(sizes))}, "
          f"max={sizes.max()}")
    print(f"  box height (az)  : min={h.min()}, median={int(np.median(h))}, "
          f"max={h.max()} px  "
          f"({h.min()*az_m_per_px:.0f} .. "
          f"{h.max()*az_m_per_px:.0f} m)")
    print(f"  box width  (rg)  : min={w.min()}, median={int(np.median(w))}, "
          f"max={w.max()} px  "
          f"({w.min()*rg_m_per_px:.0f} .. "
          f"{w.max()*rg_m_per_px:.0f} m)")

    # ---- apply the isolated-pixel filter (variant of the one baked
    # into shear_averaging.py) for the "filtered" panel. Window is a
    # physical target-scale box: `filter_target_size_m` metres in each
    # direction, converted to pixels via the current s_degraded-grid
    # spacings.
    #
    # DIFFERENCE from shear_averaging.py: the survival test here is a
    # DENSITY threshold (`neigh_density >= filter_min_density`) rather
    # than the absolute count (`neigh_count >= 3`) the detector uses.
    # Rationale: with the physical-metres refactor the pixel window
    # size varies across scenes (e.g. ICEYE spot at 0.5 m az / 1.44 m
    # rg gives a 6 az \u00d7 2 rg = 12-pixel window \u2014 with a fixed count
    # of 3 the effective density threshold is 25 %; on a coarser
    # scene where the window collapses to 3 az \u00d7 1 rg = 3 pixels the
    # same count of 3 becomes 100 %). Anchoring on density instead of
    # count keeps the filter's meaning scene-invariant, and setting
    # `filter_min_density` explicitly makes it easier to make the
    # test stricter without guessing what absolute count that maps to
    # on the current scene.
    from scipy.ndimage import uniform_filter
    filter_target_size_m = 3.0
    filter_min_density = 0.30  # stricter than shear_averaging's ~0.25 on ICEYE spot
    az_win = max(1, int(round(filter_target_size_m / az_m_per_px)))
    rg_win = max(1, int(round(filter_target_size_m / rg_m_per_px)))
    box_area = az_win * rg_win
    # uniform_filter of a 0/1 mask returns the local MEAN, i.e. the
    # density of mask==1 pixels in the window. mode='constant' treats
    # out-of-bounds as 0, so densities near the image edge are
    # slightly underestimated (a small extra strictness at borders).
    density = uniform_filter(
        mask.astype(np.float32), size=(az_win, rg_win), mode="constant",
    )
    mask_filt = (mask.astype(bool) & (density >= filter_min_density)).astype(np.float32)
    kept_before = int(mask.sum())
    kept_after = int(mask_filt.sum())
    # Also report what the density threshold corresponds to in
    # absolute count for THIS window, so the operator can compare
    # against shear_averaging.py's fixed "count >= 3" rule.
    min_count_equiv = filter_min_density * box_area
    print(
        f"\nisolated-pixel filter "
        f"({filter_target_size_m:g} m \u00d7 {filter_target_size_m:g} m "
        f"= {az_win} az \u00d7 {rg_win} rg px = {box_area} px window, "
        f"density \u2265 {filter_min_density:.2f} "
        f"\u2248 count \u2265 {min_count_equiv:.1f}): "
        f"kept {kept_after}/{kept_before} "
        f"({100 * kept_after / max(kept_before, 1):.2f}%)"
    )

    # ---- interactive figure --------------------------------------------
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(18, 12), constrained_layout=True, sharey=True,
    )

    # Left: raw mask + peaks
    ax1.imshow(mask, cmap="gray", vmin=0, vmax=1, aspect="auto",
               origin="upper", interpolation="nearest")
    ax1.scatter(peaks[:, 1], peaks[:, 0], s=8, c="red",
                marker="x", linewidths=0.7)
    ax1.set_title(f"mask_full (raw) + {K} peaks\n"
                  f"{kept_before} white pixels")
    ax1.set_xlabel("range (px, degraded)")
    ax1.set_ylabel("azimuth (px)")

    # Middle: filtered mask + peaks
    ax2.imshow(mask_filt, cmap="gray", vmin=0, vmax=1, aspect="auto",
               origin="upper", interpolation="nearest")
    ax2.scatter(peaks[:, 1], peaks[:, 0], s=8, c="red",
                marker="x", linewidths=0.7)
    ax2.set_title(
        f"after {filter_target_size_m:g} m \u00d7 "
        f"{filter_target_size_m:g} m filter "
        f"({az_win}\u00d7{rg_win} px = {box_area}), "
        f"density \u2265 {filter_min_density:.2f}\n"
        f"{kept_after} white pixels "
        f"({100 * kept_after / max(kept_before, 1):.1f}% kept)"
    )
    ax2.set_xlabel("range (px, degraded)")

    # Right: filtered mask + peaks coloured by cluster + cluster boxes
    ax3.imshow(mask_filt, cmap="gray", vmin=0, vmax=1, aspect="auto",
               origin="upper", interpolation="nearest")
    cmap = plt.get_cmap("tab20")
    for c in range(n_clusters):
        colour = cmap(c % 20)
        sel = labels == c
        ax3.scatter(peaks[sel, 1], peaks[sel, 0], s=8,
                    c=[colour], edgecolor="none")
        y_lo, y_hi, x_lo, x_hi = boxes[c]
        ax3.add_patch(Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            fill=False, edgecolor=colour, linewidth=0.7,
        ))
    ax3.set_title(f"cluster_peaks(): {K} peaks \u2192 {n_clusters} clusters")
    ax3.set_xlabel("range (px, degraded)")

    fig.suptitle(
        f"{npz_path.name}   |   mask {mask.shape}   |   "
        f"{K} peaks \u2192 {n_clusters} clusters",
        fontsize=11,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=args.dpi)
        print(f"Saved \u2192 {save_path}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
