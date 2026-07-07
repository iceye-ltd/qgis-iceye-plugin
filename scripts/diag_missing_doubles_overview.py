"""Wide-ROI overview of the three "missing double" regions.

For each ROI we render:
  * |s_degraded| in the wider box (all four candidate boxes drawn on
    top, colour-coded by disposition).
  * Azimuth-integrated |I|² profile (sum over the ROI's range span) so
    every distinct target in the region shows up as a peak.
  * The final kept box's azimuth footprint on that profile — makes
    visually obvious how much of the target is being missed.

The three "missing" ROIs the user flagged:
  T1 :  y=72000..78000,  x=865..885
  T2 :  y=67400..72500,  x=930..950
  T3 :  y=62500..67500,  x=975..1000

Output goes to ``scripts/missing_doubles_out/overview_*.png``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shear_averaging import (  # noqa: E402
    apply_range_window,
    degrade_range_resolution_range_sum,
)
from diag_missing_doubles import load_azimuth_strip  # noqa: E402

TIF = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/"
    "ICEYE_WTW3YQ_20250104T180444Z_4439676_X7_SLED_SLC.tif"
)
CSV = Path("/home/odogan/Desktop/ship_focusing/4439676/WTW3YQ_box_stats.csv")
OUT_DIR = HERE / "missing_doubles_out"

RANGE_LOOKS = 4
RANGE_SLL_DB = 55.0
RANGE_TAYLOR_NBAR = 8

ROIS = {
    "T1": {"y": (72000, 78000), "x": (865, 885)},
    "T2": {"y": (67400, 72500), "x": (930, 950)},
    "T3": {"y": (62500, 67500), "x": (975, 1000)},
}

DISP_COLORS = {
    "kept": "#00e000",
    "dropped_by_nested_box_overlap": "#ff9900",
    "dropped_by_phase_fit_quality": "#e04040",
    "dropped_by_phase_residual": "#ffcc00",
    "dropped_by_phase_slope": "#c060ff",
    "dropped_by_motion_3tier": "#5090ff",
    "dropped_by_com_pp": "#a0a0a0",
    "dropped_by_refocus_gain": "#ff40b0",
    "dropped_by_subap_sharpness": "#804000",
}


def load_boxes_in_roi(y_lo: int, y_hi: int, x_lo: int, x_hi: int) -> list[dict]:
    boxes = []
    with CSV.open() as f:
        for row in csv.DictReader(f):
            yc = row["y_c"]; xc = row["x_c"]; h = row["h"]; w = row["w"]
            if not yc:
                continue
            yc = int(yc); xc = int(xc); h = int(h); w = int(w)
            yl_b = yc - h / 2; yh_b = yc + h / 2
            xl_b = xc - w / 2; xh_b = xc + w / 2
            if yh_b > y_lo and yl_b < y_hi and xh_b > x_lo and xl_b < x_hi:
                boxes.append({
                    "init_idx": int(row["init_idx"]),
                    "final_idx": row["final_idx"],
                    "disp": row["disposition"],
                    "yc": yc, "xc": xc, "h": h, "w": w,
                    "yl": yl_b, "yh": yh_b, "xl": xl_b, "xh": xh_b,
                })
    boxes.sort(key=lambda b: b["init_idx"])
    return boxes


def render_overview(
    *,
    tag: str,
    y_lo: int, y_hi: int, x_lo: int, x_hi: int,
    amp_full: np.ndarray, strip_y0: int,
    boxes: list[dict],
    out_path: Path,
) -> None:
    y_pad = 400
    x_pad = 30
    y_lo_v = max(y_lo - y_pad, strip_y0)
    y_hi_v = min(y_hi + y_pad, strip_y0 + amp_full.shape[0])
    x_lo_v = max(x_lo - x_pad, 0)
    x_hi_v = min(x_hi + x_pad, amp_full.shape[1])
    amp = amp_full[y_lo_v - strip_y0:y_hi_v - strip_y0, x_lo_v:x_hi_v]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 12),
        gridspec_kw={"width_ratios": [1.0, 2.2]},
        constrained_layout=True,
    )
    fig.suptitle(
        f"{tag}   ROI y=[{y_lo}..{y_hi})  x=[{x_lo}..{x_hi})   "
        f"— {sum(1 for b in boxes if b['disp']=='kept')}/{len(boxes)} "
        f"boxes kept",
        fontsize=11,
    )

    ax = axes[0]
    if amp.size:
        vmin = float(np.nanpercentile(amp, 1.0))
        vmax = float(np.nanpercentile(amp, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1.0
        ax.imshow(
            amp, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax,
            origin="upper",
            extent=[x_lo_v, x_hi_v, y_hi_v, y_lo_v],
        )
    ax.plot(
        [x_lo, x_hi, x_hi, x_lo, x_lo],
        [y_lo, y_lo, y_hi, y_hi, y_lo],
        color="cyan", lw=0.8, ls="--", alpha=0.7,
        label="user ROI",
    )
    for b in boxes:
        c = DISP_COLORS.get(b["disp"], "#ffffff")
        rect = Rectangle(
            (b["xl"], b["yl"]), b["w"], b["h"],
            fill=False, edgecolor=c, lw=1.3,
        )
        ax.add_patch(rect)
        ax.text(
            b["xh"] + 0.4, b["yc"], str(b["init_idx"]),
            color=c, fontsize=8, va="center",
        )
    ax.set_xlim(x_lo_v, x_hi_v)
    ax.set_ylim(y_hi_v, y_lo_v)
    ax.set_xlabel("range (s_degraded col)")
    ax.set_ylabel("azimuth (s_degraded row)")
    ax.set_title("|s_degraded|  (all candidate boxes)", fontsize=9)

    ax = axes[1]
    y_rows = np.arange(y_lo_v, y_hi_v)
    prof = (amp ** 2).sum(axis=1) if amp.size else np.array([])
    ax.plot(y_rows, prof, lw=0.6, color="k")
    mu = float(prof.mean()) if prof.size else 0.0
    ax.axhline(mu, color="0.6", ls="--", lw=0.6, label=f"mean = {mu:.2e}")
    for b in boxes:
        c = DISP_COLORS.get(b["disp"], "#888888")
        ymin, ymax = b["yl"], b["yh"]
        ax.axvspan(ymin, ymax, ymin=0.0, ymax=0.03,
                   color=c, alpha=0.9)
        if b["disp"] == "kept":
            ax.axvspan(ymin, ymax, ymin=0.03, ymax=1.0,
                       color=c, alpha=0.08)
    ax.set_xlim(y_lo_v, y_hi_v)
    ax.set_xlabel("absolute azimuth row")
    ax.set_ylabel(r"$\sum_{rg \in ROI}|I|^2$")
    ax.set_title(
        f"azimuth intensity profile — every candidate box's y-extent "
        f"as a coloured strip on the x-axis; kept boxes also shaded above",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)

    handles = [
        plt.Line2D([0], [0], color=DISP_COLORS[d], lw=2.0, label=d)
        for d in sorted({b["disp"] for b in boxes})
        if d in DISP_COLORS
    ]
    handles.append(
        plt.Line2D([0], [0], color="cyan", lw=1.0, ls="--", label="user ROI")
    )
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    y_lo_all = min(r["y"][0] for r in ROIS.values()) - 500
    y_hi_all = max(r["y"][1] for r in ROIS.values()) + 500
    print(f"Loading azimuth strip [{y_lo_all}, {y_hi_all}) from {TIF.name} ...")
    strip = load_azimuth_strip(TIF, y_lo_all, y_hi_all, left=True)
    print(f"  strip shape = {strip.shape}")
    print("Range Taylor window + range degradation ...")
    strip_win = apply_range_window(strip, sll_db=RANGE_SLL_DB, nbar=RANGE_TAYLOR_NBAR)
    del strip
    s_deg = degrade_range_resolution_range_sum(strip_win, RANGE_LOOKS)
    del strip_win
    amp_full = np.abs(s_deg).astype(np.float32)
    print(f"  s_degraded strip shape = {s_deg.shape}, amp dtype={amp_full.dtype}")
    del s_deg

    for tag, roi in ROIS.items():
        y_lo, y_hi = roi["y"]
        x_lo, x_hi = roi["x"]
        boxes = load_boxes_in_roi(y_lo, y_hi, x_lo, x_hi)
        print(f"\n=== {tag}  y=[{y_lo}..{y_hi}) x=[{x_lo}..{x_hi}) : "
              f"{len(boxes)} boxes ===")
        for b in boxes:
            print(f"  {b['init_idx']:4d}  {b['disp']:32s}  "
                  f"yc={b['yc']} xc={b['xc']} h={b['h']} w={b['w']}")
        out = OUT_DIR / f"overview_{tag}.png"
        render_overview(
            tag=tag,
            y_lo=y_lo, y_hi=y_hi, x_lo=x_lo, x_hi=x_hi,
            amp_full=amp_full, strip_y0=y_lo_all, boxes=boxes,
            out_path=out,
        )
        print(f"  → {out}")


if __name__ == "__main__":
    main()
