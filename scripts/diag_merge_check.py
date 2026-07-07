"""Verify the --merge-same-corridor flag produces ONE box covering both
T1 and T2 arcs (rows 6100..6600, cols ~590..615).

Runs `detect_targets` twice — with and without the flag — and reports
which boxes overlap the merged (T1 ∪ T2) rectangle. Also saves a zoom
figure comparing the two.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from view_subaps_cov import detect_targets  # noqa: E402


T1 = (6100, 6350, 600, 615)   # (y_lo, y_hi, x_lo, x_hi) half-open
T2 = (6350, 6600, 590, 610)
COMBINED = (6100, 6600, 590, 615)

ZOOM = (5950, 6750, 560, 650)  # y_lo, y_hi, x_lo, x_hi


def overlapping(boxes, region):
    y_lo, y_hi, x_lo, x_hi = region
    out = []
    for c, (by_lo, by_hi, bx_lo, bx_hi) in enumerate(boxes):
        if not (by_hi < y_lo or by_lo >= y_hi
                or bx_hi < x_lo or bx_lo >= x_hi):
            out.append((c, int(by_lo), int(by_hi), int(bx_lo), int(bx_hi)))
    return out


def _draw(ax, boxes, cids, mask, extent, title):
    ax.imshow(mask, cmap="gray", aspect="auto", extent=extent,
              vmin=0.0, vmax=1.0, interpolation="nearest")
    cmap = plt.get_cmap("tab20")
    for c, y_lo, y_hi, x_lo, x_hi in cids:
        colour = cmap(c % 20)
        ax.add_patch(Rectangle(
            (x_lo, y_lo), x_hi - x_lo + 1, y_hi - y_lo + 1,
            fill=False, edgecolor=colour, linewidth=1.6,
        ))
        ax.text(x_hi + 0.5, y_lo + 2, f"c{c}", color=colour,
                fontsize=9, fontweight="bold")
    for name, (y_lo, y_hi, x_lo, x_hi), col in (
        ("T1", T1, "red"), ("T2", T2, "red"), ("T1+T2", COMBINED, "yellow"),
    ):
        ax.add_patch(Rectangle(
            (x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
            fill=False, edgecolor=col, linewidth=1.5,
            linestyle="--" if col == "red" else ":",
        ))
        ax.text(x_hi + 1, y_lo + 5, name, color=col, fontsize=9)
    ax.set_xlim(ZOOM[2], ZOOM[3])
    ax.set_ylim(ZOOM[1], ZOOM[0])
    ax.set_xlabel("range px")
    ax.set_title(title)


def main() -> None:
    print("=" * 70)
    print("WITHOUT merge_same_corridor")
    print("=" * 70)
    r_off = detect_targets(
        Path("/home/odogan/Desktop/ship_focusing/4439676/"
             "ICEYE_WTW3YQ_20250104T180444Z_4439676_X7_SLED_SLC_subapertures.npz"),
        merge_same_corridor=False,
        verbose=False,
    )
    hits_off = overlapping(r_off.boxes, COMBINED)
    print(f"  {len(hits_off)} boxes overlap T1∪T2 rectangle:")
    for c, y_lo, y_hi, x_lo, x_hi in hits_off:
        print(f"    c{c:>4d}  y=[{y_lo}..{y_hi}]  x=[{x_lo}..{x_hi}]  "
              f"h={y_hi-y_lo+1} × w={x_hi-x_lo+1} px")

    print()
    print("=" * 70)
    print("WITH merge_same_corridor (max_norm_dist=1.0)")
    print("=" * 70)
    r_on = detect_targets(
        Path("/home/odogan/Desktop/ship_focusing/4439676/"
             "ICEYE_WTW3YQ_20250104T180444Z_4439676_X7_SLED_SLC_subapertures.npz"),
        merge_same_corridor=True,
        merge_max_norm_dist=1.0,
        verbose=False,
    )
    hits_on = overlapping(r_on.boxes, COMBINED)
    print(f"  {len(hits_on)} boxes overlap T1∪T2 rectangle:")
    for c, y_lo, y_hi, x_lo, x_hi in hits_on:
        print(f"    c{c:>4d}  y=[{y_lo}..{y_hi}]  x=[{x_lo}..{x_hi}]  "
              f"h={y_hi-y_lo+1} × w={x_hi-x_lo+1} px")

    # ---- Zoom figure ----------------------------------------------
    mask_v = r_off.mask_filt[ZOOM[0]:ZOOM[1], ZOOM[2]:ZOOM[3]]
    extent = (ZOOM[2], ZOOM[3], ZOOM[1], ZOOM[0])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10),
                                    constrained_layout=True, sharey=True)
    _draw(ax1, r_off.boxes, hits_off, mask_v, extent,
          f"WITHOUT merge  ({len(hits_off)} boxes in T1∪T2)")
    _draw(ax2, r_on.boxes, hits_on, mask_v, extent,
          f"WITH merge  ({len(hits_on)} boxes in T1∪T2)")
    out = Path("/home/odogan/Desktop/ship_focus/qgis-iceye-plugin/diag_merge_check.png")
    fig.suptitle(
        "merge_same_corridor comparison: red dashed = user targets, "
        "yellow dotted = union",
        fontsize=10,
    )
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure to {out}")


if __name__ == "__main__":
    main()
