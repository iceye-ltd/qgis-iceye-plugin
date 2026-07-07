"""View `d_phase` with the detected bounding boxes overlaid.

Loads a `.npz` produced by scripts/shear_averaging.py (saved next to the
figure path; default location is `DEFAULT_SAVE_DIR/shear_<ts>_<rand>.npz`)
and writes one PNG per plot (no subplots) to ``DEFAULT_OUT_DIR``:

    <stem>_d_phase.png        d_phase image with bounding boxes
    <stem>_amp.png            |s_degraded| image with bounding boxes
    <stem>_box_<NNN>.png      φ(u) trace at the brightest amp_raw column,
                              one file per box.

Usage
-----
    python scripts/view_dphase_boxes.py PATH.npz
    python scripts/view_dphase_boxes.py                       # auto-pick latest
    python scripts/view_dphase_boxes.py PATH.npz --out-dir /tmp/figs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shear_averaging import DEFAULT_SAVE_DIR, _min_distance_line_fit

DEFAULT_OUT_DIR = Path("/home/odogan/Desktop/cop/ship_phase")


def _latest_npz(directory: Path) -> Path | None:
    files = sorted(directory.glob("shear_*.npz"))
    return files[-1] if files else None


def _overlay_boxes(
    ax: plt.Axes,
    boxes_yxhw: np.ndarray,
    color: str = "red",
    lw: float = 1.0,
) -> None:
    """Draw axis-aligned rectangles, one per row of boxes_yxhw (y_c, x_c, h, w)."""
    for y, x, h, w in np.atleast_2d(boxes_yxhw):
        ax.add_patch(plt.Rectangle(
            (x - w / 2 - 0.5, y - h / 2 - 0.5), w, h,
            fill=False, edgecolor=color, linewidth=lw,
        ))


def _make_d_phase_figure(
    d_phase: np.ndarray,
    boxes: np.ndarray,
    src_name: str,
    box_color: str,
    box_lw: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    im = ax.imshow(
        d_phase, cmap="twilight", aspect="auto", vmin=-np.pi, vmax=np.pi,
    )
    _overlay_boxes(ax, boxes, color=box_color, lw=box_lw)
    ax.set_title(
        f"d_phase = arg{{s_d[u+1]·s_d*[u]}}  ({len(boxes)} boxes)\n"
        f"source: {src_name}",
    )
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    fig.colorbar(im, ax=ax, label="rad")
    return fig


def _make_amp_figure(
    amp_raw: np.ndarray,
    boxes: np.ndarray,
    src_name: str,
    box_color: str,
    box_lw: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 10), constrained_layout=True)
    vmax = float(np.percentile(amp_raw, 99.5)) if amp_raw.size else 1.0
    im = ax.imshow(
        amp_raw, cmap="gray", aspect="auto", vmin=0.0, vmax=max(vmax, 1e-12),
    )
    _overlay_boxes(ax, boxes, color=box_color, lw=box_lw)
    ax.set_title(f"|s_degraded|  ({len(boxes)} boxes)\nsource: {src_name}")
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    fig.colorbar(im, ax=ax, label="amplitude")
    return fig


def _box_weighted_phase(
    box: np.ndarray, d_phase: np.ndarray, amp_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-azimuth-row circular weighted mean of d_phase across range.

    For each azimuth row u inside the box, returns
        phi(u) = arg( Σ_x w(u, x) · exp(i · d_phase[u, x]) ),
        coh(u) = |Σ_x w(u, x) · exp(...)| / Σ_x w(u, x)  ∈ [0, 1].
    The weight w(u, x) = amp_raw[u, x] · amp_raw[u+1, x] is the paired
    amplitude, matching the cross-product structure of d_phase. A simple
    linear average over phases would be wrong near ±π; using the complex
    weighted sum handles wrap-around correctly.
    """
    n_az, n_rg = d_phase.shape
    y_c, x_c, h, w = box
    y0 = max(0, int(round(y_c - h / 2)))
    y1 = min(n_az, int(round(y_c + h / 2)))
    x0 = max(0, int(round(x_c - w / 2)))
    x1 = min(n_rg, int(round(x_c + w / 2)))
    if y1 <= y0 or x1 <= x0:
        empty = np.empty(0)
        return empty.astype(int), empty, empty

    sub_phase = d_phase[y0:y1, x0:x1]
    weight = amp_raw[y0:y1, x0:x1] * amp_raw[y0 + 1: y1 + 1, x0:x1]
    z = (weight * np.exp(1j * sub_phase)).sum(axis=1)
    w_sum = weight.sum(axis=1) + 1e-12
    phi = np.angle(z)
    coh = np.abs(z) / w_sum
    return np.arange(y0, y1), coh, phi


def _fit_phase_line(
    u: np.ndarray, phi: np.ndarray, coh: np.ndarray,
) -> tuple[float, float, float]:
    """Wrap-aware line fit on the raw wrapped phase values.

    Delegates to ``shear_averaging._min_distance_line_fit`` — a brute-force
    grid search over 1001 candidate slopes in ``[-0.2, +0.2]`` rad/sample
    that picks the candidate whose wrapped line minimises the coh-weighted
    mean wrapped distance ``|arg(exp(j·(phi - slope·u - intercept)))|`` to
    the data. The intercept per candidate is the wrap-aware coh-weighted
    circular mean of ``phi - slope·u`` (closed-form, no inner search).

    Returns
    -------
    slope : float
        rad / azimuth row; NaN on degenerate input.
    intercept : float
        at ``u = 0``, in (-π, π]; NaN on degenerate input.
    residual : float
        Coh-weighted mean wrapped Euclidean distance from each sample to
        the chosen line (rad) — the actual cost the fit just minimised.
        Replaces the previous coh-weighted variance metric, which scored
        how scattered phi was around its weighted mean rather than how
        well a line described it.
    """
    if phi.size < 2:
        return float("nan"), float("nan"), float("nan")
    # min_row_coherence=0 → all rows participate (the helper falls back to
    # the universal mask when too few rows exceed the gate, which is the
    # right behaviour here since this viewer is asked to fit every box).
    slope, intercept, _n_in = _min_distance_line_fit(
        phi.astype(np.float64),
        u.astype(np.float64),
        np.clip(coh, 0.0, None).astype(np.float64),
        min_row_coherence=0.0,
        inlier_tol_rad=0.5,
    )
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return float("nan"), float("nan"), float("nan")
    # Coh-weighted mean wrapped distance to the chosen line — same metric
    # shear_averaging.compute_box_phase_estimates records per box.
    r = np.angle(np.exp(1j * (phi - (slope * u + intercept))))
    w = np.clip(coh, 0.0, None).astype(np.float64)
    denom = float(w.sum())
    residual = float((w * np.abs(r)).sum() / denom) if denom > 0 else float("nan")
    return float(slope), float(intercept), residual


def _make_box_figure(
    k_box: int,
    box: np.ndarray,
    d_phase: np.ndarray,
    amp_raw: np.ndarray,
    src_name: str,
) -> tuple[plt.Figure, dict]:
    """Single-axes figure: amp-weighted circular mean of d_phase along range.

    Marker size scales with the per-row coherence
    |Σ w·e^{iφ}| / Σ w ∈ [0, 1] — large markers mean range columns inside
    the box agree in phase, small markers mean they disagree. Also fits and
    overlays a wrap-aware linear trend φ(u) ≈ slope·u + intercept on the
    same φ values (no unwrapping) via :func:`_fit_phase_line`. The plot is
    drawn with the wrapped line ``arg(exp(j·(slope·u + intercept)))`` so it
    follows the data through the (-π, π] discontinuity exactly. Returns
    the figure and stats.
    """
    u, coh, phi = _box_weighted_phase(box, d_phase, amp_raw)
    slope, intercept, residual = _fit_phase_line(u, phi, coh)
    n_rows = int(phi.size)
    slope_total = slope * n_rows  # printed instead of slope: rad over the box
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    ax.axhline(0.0, color="0.7", lw=0.7)
    if phi.size > 0:
        sizes = 6 + 30 * np.clip(coh, 0.0, 1.0)
        ax.scatter(
            u, phi, s=sizes, c="tab:blue", edgecolors="none", alpha=0.85,
        )
    if np.isfinite(slope):
        u_dense = np.asarray(u, dtype=np.float64)
        line_wrapped = np.angle(np.exp(1j * (slope * u_dense + intercept)))
        ax.plot(
            u_dense, line_wrapped,
            color="tab:red", lw=1.2,
            label=(
                f"fit: slope·n={slope_total:+.3e} rad  "
                f"res={residual:.3f} rad"
            ),
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.6)
    y_c, x_c, h, w = box
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("azimuth row u")
    ax.set_ylabel(r"$\overline{\Delta\varphi}$  [rad]   (amp-weighted)")
    ax.set_title(
        f"box #{k_box}  y={y_c:.0f}  x={x_c:.0f}  h={int(h)}  w={int(w)}\n"
        f"slope·n={slope_total:+.3e} rad   res={residual:.3f} rad   "
        f"n={n_rows}\n"
        f"source: {src_name}",
    )
    stats = {
        "idx": int(k_box),
        "y_c": float(y_c),
        "x_c": float(x_c),
        "h": int(h),
        "w": int(w),
        "slope": slope,
        "slope_total": float(slope_total),
        "intercept": intercept,
        "residual": residual,
        "n_rows": n_rows,
    }
    return fig, stats


def _format_summary_table(
    label: str, items: list[dict], slope_thresh: float = 1.0,
) -> str:
    """Render a single in-/out-of-band table as a string.

    Also appends a one-liner counting how many boxes in the group satisfy
    ``|slope·n| > slope_thresh`` (rad), which is the easy heuristic for
    "this trace is too steep to be stationary noise".
    """
    lines = [f"[{label}]  ({len(items)} boxes)"]
    if not items:
        return "\n".join(lines)
    lines.append(
        f"{'idx':>4}  {'y_c':>5}  {'x_c':>5}  {'slope·n[rad]':>13}  "
        f"{'res[rad]':>10}  {'n':>4}"
    )
    for s in items:
        lines.append(
            f"{s['idx']:>4}  {s['y_c']:>5.0f}  {s['x_c']:>5.0f}  "
            f"{s['slope_total']:>+13.3e}  {s['residual']:>10.3f}  "
            f"{s['n_rows']:>4}"
        )
    n_above = sum(
        1 for s in items
        if np.isfinite(s["slope_total"]) and abs(s["slope_total"]) > slope_thresh
    )
    lines.append(
        f"  → boxes with |slope·n| > {slope_thresh:g} rad: "
        f"{n_above}/{len(items)}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, default=None,
        help="Path to a .npz produced by shear_averaging.py. "
             f"If omitted, the latest in {DEFAULT_SAVE_DIR} is used.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Directory to write PNGs into (default: {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Skip interactive plt.show().",
    )
    parser.add_argument(
        "--box-color", default="red", help="Bounding-box edge color.",
    )
    parser.add_argument(
        "--box-lw", type=float, default=1.0, help="Bounding-box line width.",
    )
    parser.add_argument(
        "--x-min", type=float, default=90.0,
        help="Lower x_c bound (inclusive) for the in-band group (default 90).",
    )
    parser.add_argument(
        "--x-max", type=float, default=140.0,
        help="Upper x_c bound (inclusive) for the in-band group (default 140).",
    )
    parser.add_argument(
        "--slope-thresh", type=float, default=1.0,
        help="Threshold (in rad) on |slope·n|. The summary tables append a "
             "count of boxes exceeding it. Default: 1.0.",
    )
    args = parser.parse_args()

    src = args.path or _latest_npz(DEFAULT_SAVE_DIR)
    if src is None or not src.exists():
        raise SystemExit(
            f"No .npz found at {args.path or DEFAULT_SAVE_DIR}. "
            "Run scripts/shear_averaging.py first (without --no-save)."
        )

    data = np.load(src)
    d_phase = data["d_phase"]
    boxes = data["boxes_yxhw"]
    amp_raw = data["amp_raw"] if "amp_raw" in data.files else None

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    fig_phase = _make_d_phase_figure(
        d_phase, boxes, src.name, args.box_color, args.box_lw,
    )
    fig_phase.savefig(out_dir / f"{stem}_d_phase.png", dpi=150)

    if amp_raw is not None:
        fig_amp = _make_amp_figure(
            amp_raw, boxes, src.name, args.box_color, args.box_lw,
        )
        fig_amp.savefig(out_dir / f"{stem}_amp.png", dpi=150)

    in_band: list[dict] = []
    out_band: list[dict] = []
    if amp_raw is not None and len(boxes) > 0:
        n_digits = max(3, len(str(len(boxes) - 1)))
        for k_box in range(len(boxes)):
            fig_box, stats = _make_box_figure(
                k_box, boxes[k_box], d_phase, amp_raw, src.name,
            )
            fig_box.savefig(
                out_dir / f"{stem}_box_{k_box:0{n_digits}d}.png", dpi=150,
            )
            plt.close(fig_box)
            x_c = stats["x_c"]
            (in_band if args.x_min <= x_c <= args.x_max else out_band).append(stats)

        csv_path = out_dir / f"{stem}_summary.csv"
        with csv_path.open("w") as f:
            f.write(
                "idx,y_c,x_c,h,w,in_band,slope_rad_per_row,"
                "intercept_rad,residual_rad,n_rows\n"
            )
            for group, flag in ((in_band, 1), (out_band, 0)):
                for s in group:
                    f.write(
                        f"{s['idx']},{s['y_c']:.3f},{s['x_c']:.3f},"
                        f"{s['h']},{s['w']},{flag},"
                        f"{s['slope']:.6e},{s['intercept']:.6e},"
                        f"{s['residual']:.6e},{s['n_rows']}\n"
                    )

        in_label = f"x_c ∈ [{args.x_min:g}, {args.x_max:g}]"
        out_label = f"x_c ∉ [{args.x_min:g}, {args.x_max:g}]"
        print(_format_summary_table(in_label, in_band, args.slope_thresh))
        print()
        print(_format_summary_table(out_label, out_band, args.slope_thresh))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
