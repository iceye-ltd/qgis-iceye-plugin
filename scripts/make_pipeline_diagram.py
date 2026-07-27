"""Render the shear_averaging.py pipeline as a 16:9 PNG for slides.

Two-row zigzag flow chart: 5 stages on top (left → right) and 5 on
bottom (right → left), plus DROP sinks under the two decision boxes.
Sized 13.333 x 7.5 inches at 300 DPI (= 4000 x 2250 px) — a perfect
fit for a widescreen (16:9) PowerPoint slide.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)


COL_INPUT = "#dbe4f0"
COL_PREP = "#c7e9d1"
COL_DETECT = "#e5d4f5"
COL_FILTER = "#ffe0b3"
COL_DECISION = "#ffe873"
COL_AF = "#cfe9f2"
COL_OUT = "#f4d9b0"
COL_KEEP = "#d3f0d3"
COL_DROP = "#f8c9c9"

EDGE = "#333333"


def _add_box(
    ax, x, y, w, h, title, body="", *,
    face=COL_FILTER, edge=EDGE,
    title_fs=10.5, body_fs=8.0,
    title_weight="bold", zorder=2,
    decision_marker=False,
):
    """Rounded rectangle with a bold title and optional body text.

    ``decision_marker`` draws a small yellow tag in the top-left to
    signal a decision node — same visual role as a diamond but the
    rectangle keeps the full width available for text.
    """
    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        linewidth=1.2, facecolor=face, edgecolor=edge,
        zorder=zorder,
    )
    ax.add_patch(patch)
    if decision_marker:
        marker_w = 1.4
        marker_h = 0.55
        mx = x - w / 2 + 0.18
        my = y + h / 2 - 0.36
        ax.add_patch(FancyBboxPatch(
            (mx, my - marker_h / 2), marker_w, marker_h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=0.6, facecolor="#ffb703",
            edgecolor="#8a5a04", zorder=zorder + 1,
        ))
        ax.text(
            mx + marker_w / 2, my, "DECISION",
            ha="center", va="center",
            fontsize=6.6, fontweight="bold",
            color="#3d2a00", zorder=zorder + 2,
        )
    if body:
        ax.text(
            x, y + h / 2 - 0.55, title,
            ha="center", va="top",
            fontsize=title_fs, fontweight=title_weight,
            color="#111", zorder=zorder + 1,
        )
        ax.text(
            x, y + h / 2 - 1.20, body,
            ha="center", va="top",
            fontsize=body_fs, color="#111",
            zorder=zorder + 1,
        )
    else:
        ax.text(
            x, y, title,
            ha="center", va="center",
            fontsize=title_fs, fontweight=title_weight,
            color="#111", zorder=zorder + 1,
        )
    return (x, y, w, h)


def _arrow(
    ax, a, b, *,
    label=None, color=EDGE, lw=1.6,
    connectionstyle="arc3,rad=0.0",
    start_side=None, end_side=None,
    label_offset=(0.0, 0.0), label_fs=9.0,
):
    """Directed edge from box a → box b (snapped to a side)."""
    def _anchor(box, side):
        x, y, w, h = box
        if side == "top":
            return (x, y + h / 2)
        if side == "bottom":
            return (x, y - h / 2)
        if side == "left":
            return (x - w / 2, y)
        if side == "right":
            return (x + w / 2, y)
        return (x, y)

    ax_x, ax_y = a[0], a[1]
    bx_x, bx_y = b[0], b[1]
    dx = bx_x - ax_x
    dy = bx_y - ax_y
    if start_side is None:
        start_side = ("right" if dx > 0 else "left") if abs(dx) >= abs(dy) \
            else ("top" if dy > 0 else "bottom")
    if end_side is None:
        end_side = ("left" if dx > 0 else "right") if abs(dx) >= abs(dy) \
            else ("bottom" if dy > 0 else "top")

    start = _anchor(a, start_side)
    end = _anchor(b, end_side)
    arr = FancyArrowPatch(
        start, end, arrowstyle="-|>",
        mutation_scale=20, color=color, lw=lw,
        connectionstyle=connectionstyle, zorder=1.5,
    )
    ax.add_patch(arr)
    if label:
        mx = (start[0] + end[0]) / 2 + label_offset[0]
        my = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(
            mx, my, label,
            ha="center", va="center",
            fontsize=label_fs, fontweight="bold", color="#111",
            zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.22", facecolor="white",
                edgecolor="none", alpha=0.95,
            ),
        )


def build_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(13.333, 7.5), dpi=300)
    # Data range chosen so 56 / 31.5 ≈ 1.78 matches the 16:9 fig aspect;
    # with aspect="equal" this fills the whole slide without vertical
    # margins.
    ax.set_xlim(0, 56)
    ax.set_ylim(-1.0, 30.5)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.suptitle(
        "SAR Moving-Target Detection Pipeline  —  scripts/shear_averaging.py",
        fontsize=16, fontweight="bold", y=0.965,
    )
    ax.text(
        28, 28.0,
        "SLC → range-degraded SLC → sub-aperture stats → detection → "
        "phase / motion / quality / autofocus gates → focused output",
        ha="center", va="center",
        fontsize=11, style="italic", color="#444",
    )

    xs = [5.5, 15.7, 25.9, 36.1, 46.3]
    y_top = 22.0
    y_bot = 9.5
    W = 9.0
    H = 6.2
    title_fs = 10.0
    body_fs = 8.6

    # ============ Top row: 1 → 5, left to right =======================
    b1 = _add_box(
        ax, xs[0], y_top, W, H,
        "1. Load & configure",
        "• Read CLI + JSON config → cfg\n"
        "• Load SLC s from\n"
        "   .tif / .npz / .npy\n"
        "• Read ICEYE sidecar:\n"
        "   PRF, spacings, timestamps",
        face=COL_INPUT, title_fs=title_fs, body_fs=body_fs,
    )
    b2 = _add_box(
        ax, xs[1], y_top, W, H,
        "2. Preprocess",
        "• Range decimation\n"
        "  (coherent sum of range looks)\n"
        "  s → s_degraded\n"
        "• Split into N Doppler\n"
        "  sub-apertures\n"
        "  → sub_mean, sub_var",
        face=COL_PREP, title_fs=title_fs, body_fs=body_fs,
    )
    b3 = _add_box(
        ax, xs[2], y_top, W, H,
        "3. Detection",
        "cluster_targets:\n"
        "• CoV² gate on sub-var\n"
        "• Isolated-pixel filter (3×3 m)\n"
        "• Local-peak seeds → clusters\n"
        "• Grow + extend in azimuth\n"
        "• Optional CA-CFAR",
        face=COL_DETECT, title_fs=title_fs, body_fs=body_fs,
    )
    b4 = _add_box(
        ax, xs[3], y_top, W, H,
        "4. Baseline filters",
        "• Project boxes to\n"
        "  s_degraded grid\n"
        "• Nested-box cleanup\n"
        "• Phase-slope filter\n"
        "• Phase-residual filter\n"
        "• Sub-aperture COM PP (opt.)",
        face=COL_FILTER, title_fs=title_fs, body_fs=body_fs,
    )
    b5 = _add_box(
        ax, xs[4], y_top, W, H,
        "5. Motion gate",
        "on |slope · n_rows|:\n"
        "•  < lower  → drop (stationary)\n"
        "•  ≥ thresh → keep (clear mover)\n"
        "•  middle band → keep iff\n"
        "    az_pp ≥ threshold",
        face=COL_DECISION, title_fs=title_fs, body_fs=body_fs,
        decision_marker=True,
    )

    _arrow(ax, b1, b2)
    _arrow(ax, b2, b3)
    _arrow(ax, b3, b4)
    _arrow(ax, b4, b5)

    # ============ Bottom row: 6 → 10, right to left ===================
    b6 = _add_box(
        ax, xs[4], y_bot, W, H,
        "6. Quality gates",
        "• Phase-fit-quality\n"
        "  frac_within_1rad_coh ≥ min\n"
        "• Top-N cap on candidates\n"
        "  + score-floor rescue\n"
        "• Optional refocus-gain gate\n"
        "• Optional sharpness gate",
        face=COL_FILTER, title_fs=title_fs, body_fs=body_fs,
    )
    b7 = _add_box(
        ax, xs[3], y_bot, W, H,
        "7. Autofocus",
        "per-box, polynomial range-walk:\n"
        "• Coarse-to-fine grid search\n"
        "• Early-exit short-circuit\n"
        "• Optional PGA refinement\n"
        "  on the strong-target look\n"
        "→ best_deviation + gain (dB)",
        face=COL_AF, title_fs=title_fs, body_fs=body_fs,
    )
    b8 = _add_box(
        ax, xs[2], y_bot, W, H,
        "8. AF gate",
        "|best_deviation|  ≥\n"
        "af_min_abs_deviation ?\n\n"
        "→ keeps only boxes whose\n"
        "   corrected chip is worth\n"
        "   pasting back into the scene",
        face=COL_DECISION, title_fs=title_fs, body_fs=body_fs,
        decision_marker=True,
    )
    b9 = _add_box(
        ax, xs[1], y_bot, W, H,
        "9. Paste chips",
        "• Rebind s = |s|  (float32)\n"
        "  frees the complex SLC buffer\n"
        "  (rule: never copy s)\n"
        "• Paste amp-corrected chips\n"
        "  into s in place\n"
        "  at each surviving box",
        face=COL_AF, title_fs=title_fs, body_fs=body_fs,
    )
    b10 = _add_box(
        ax, xs[0], y_bot, W, H,
        "10. Save & finish",
        "Keeper PNGs (always):\n"
        "• {stem}_af_before.png\n"
        "• {stem}_af_after.png\n"
        "• {stem}_kept_vs_eliminated.png\n"
        "\nDebug mode also writes per-box\n"
        "PNGs / NPZ + box_stats.csv",
        face=COL_OUT, title_fs=title_fs, body_fs=body_fs,
    )

    # Top-row → bottom-row corner.
    _arrow(
        ax, b5, b6,
        start_side="bottom", end_side="top",
        connectionstyle="arc3,rad=0.0", lw=1.8,
    )
    _arrow(ax, b6, b7, label="keep",
           start_side="left", end_side="right",
           label_offset=(0.0, 0.4))
    _arrow(ax, b7, b8, start_side="left", end_side="right")
    _arrow(ax, b8, b9, label="yes",
           start_side="left", end_side="right",
           label_offset=(0.0, 0.4))
    _arrow(ax, b9, b10, start_side="left", end_side="right")

    # DROP sinks — placed OUTSIDE the flow columns and connected with
    # short horizontal arrows so they never cross a neighbouring box.
    b_drop_mot = _add_box(
        ax, 52.5, 15.7, 5.0, 2.6,
        "DROP\n(clear stationary /\nlow-az_pp ambiguous)",
        face=COL_DROP, title_fs=8.5, title_weight="bold",
    )
    _arrow(ax, b5, b_drop_mot,
           label="drop", label_offset=(0.0, 0.35),
           start_side="right", end_side="top",
           lw=1.6,
           connectionstyle="angle3,angleA=0,angleB=90")
    b_drop_af = _add_box(
        ax, 25.9, 3.2, 6.0, 1.6,
        "DROP  (no chip write)",
        face=COL_DROP, title_fs=9.0, title_weight="bold",
    )
    _arrow(ax, b8, b_drop_af,
           label="no", label_offset=(0.0, 0.15),
           start_side="bottom", end_side="top",
           lw=1.6)

    # ============ Legend =============================================
    legend_items = [
        ("Input", COL_INPUT),
        ("Preprocess", COL_PREP),
        ("Detection", COL_DETECT),
        ("Filter", COL_FILTER),
        ("Decision", COL_DECISION),
        ("Autofocus", COL_AF),
        ("Save output", COL_OUT),
        ("Drop", COL_DROP),
    ]
    lx0 = 0.6
    ly0 = -0.4
    swatch_w = 0.8
    swatch_h = 0.6
    step_x = 6.7
    ax.text(
        lx0, ly0 + 0.95, "Legend",
        fontsize=10.5, fontweight="bold",
        ha="left", va="center", color="#222",
    )
    for i, (label, face) in enumerate(legend_items):
        x0 = lx0 + i * step_x
        ax.add_patch(FancyBboxPatch(
            (x0, ly0 - swatch_h / 2), swatch_w, swatch_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            linewidth=0.9, facecolor=face, edgecolor=EDGE,
        ))
        ax.text(
            x0 + swatch_w + 0.2, ly0, label,
            ha="left", va="center", fontsize=9, color="#222",
        )

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(__file__).with_name("shear_averaging_pipeline.png")
    build_diagram(out)
    print(f"Saved → {out}")
