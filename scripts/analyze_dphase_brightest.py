"""Per-box azimuth trace of d_phase sampled at the brightest range column.

For each detection box (filtered by range/x-centre to focus on the candidate
moving targets), and for each azimuth row inside the box, we:

    1. find the range column with the largest amplitude inside the box,
    2. read out d_phase at that (row, column) cell.

The resulting 1-D trace phi(u) is the phase derivative of the dominant
scatterer within the box at each azimuth slow-time. For stationary point
targets phi(u) sits near zero with random jitter; for movers it shows a
non-zero mean (constant Doppler offset) and/or an azimuth-dependent
trend (along-track acceleration / quadratic phase).

Loads the .npz produced by scripts/shear_averaging.py (must include
`d_phase`, `amp_raw`, `boxes_yxhw`).

Usage
-----
    python scripts/analyze_dphase_brightest.py [PATH.npz]
                          [--x-min 90] [--x-max 140]
                          [--save out.png] [--no-show] [--cols 4]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from shear_averaging import DEFAULT_SAVE_DIR


def _latest_npz(directory: Path) -> Path | None:
    files = sorted(directory.glob("shear_*.npz"))
    return files[-1] if files else None


def _box_bounds(
    box: np.ndarray, n_az_dphase: int, n_rg: int,
) -> tuple[int, int, int, int]:
    """Return (y0, y1, x0, x1) clipped so y0:y1 indexes valid d_phase rows."""
    y_c, x_c, h, w = box
    y0 = max(0, int(round(y_c - h / 2)))
    y1 = min(n_az_dphase, int(round(y_c + h / 2)))
    x0 = max(0, int(round(x_c - w / 2)))
    x1 = min(n_rg, int(round(x_c + w / 2)))
    return y0, y1, x0, x1


def _box_brightest_phase(
    box: np.ndarray, d_phase: np.ndarray, amp_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For each azimuth row u inside the box, return:
        u_idx[u]   : absolute d_phase row index,
        x_bright[u]: range column with largest paired amplitude,
        amp[u]     : the paired amplitude (amp_raw[u]*amp_raw[u+1]) at x_bright,
        phi[u]     : d_phase[u, x_bright].
    The pairing uses amp_raw[u]*amp_raw[u+1] because d_phase row u is the
    cross-product of s_degraded rows u and u+1.
    """
    n_az_dphase, n_rg = d_phase.shape
    y0, y1, x0, x1 = _box_bounds(box, n_az_dphase, n_rg)
    if y1 <= y0 or x1 <= x0:
        empty = np.empty(0)
        return empty.astype(int), empty.astype(int), empty, empty

    sub_phase = d_phase[y0:y1, x0:x1]
    amp_pair = amp_raw[y0:y1, x0:x1] * amp_raw[y0 + 1: y1 + 1, x0:x1]

    x_local = np.argmax(amp_pair, axis=1)
    rows = np.arange(y1 - y0)
    phi = sub_phase[rows, x_local]
    amp = amp_pair[rows, x_local]
    return (np.arange(y0, y1), x_local + x0, amp, phi)


def _circular_mean(phi: np.ndarray, w: np.ndarray | None = None) -> float:
    if phi.size == 0:
        return float("nan")
    w = np.ones_like(phi) if w is None else w
    z = (w * np.exp(1j * phi)).sum()
    return float(np.angle(z))


def _circular_coherence(phi: np.ndarray, w: np.ndarray | None = None) -> float:
    """|mean exp(i phi)| ∈ [0, 1]; 1 = perfectly aligned, 0 = uniform."""
    if phi.size == 0:
        return float("nan")
    w = np.ones_like(phi) if w is None else w
    num = abs((w * np.exp(1j * phi)).sum())
    den = float(w.sum()) + 1e-12
    return float(num / den)


def _compute_traces(
    sel_idx: np.ndarray,
    sel_boxes: np.ndarray,
    d_phase: np.ndarray,
    amp_raw: np.ndarray,
    label: str,
) -> list[dict]:
    """Build per-box phase traces and print a summary table."""
    traces: list[dict] = []
    print(f"\n[{label}]  {len(sel_boxes)} boxes")
    print(f"{'idx':>4}  {'y_c':>6}  {'x_c':>5}  {'h':>4}  {'w':>3}  "
          f"{'mean_phi':>9}  {'|mean|':>7}  {'coh':>5}  {'std':>5}")
    for k, box in zip(sel_idx, sel_boxes):
        u_idx, x_bright, amp, phi = _box_brightest_phase(box, d_phase, amp_raw)
        if phi.size == 0:
            continue
        mean_phi = _circular_mean(phi, w=amp)
        coherence = _circular_coherence(phi, w=amp)
        # Std around the circular mean, computed on phase residuals wrapped to (-π, π]
        resid = np.angle(np.exp(1j * (phi - mean_phi)))
        std = float(np.sqrt(np.average(resid ** 2, weights=amp)))
        traces.append({
            "idx": int(k), "box": box,
            "u": u_idx, "x": x_bright, "amp": amp, "phi": phi,
            "mean_phi": mean_phi, "coherence": coherence, "std": std,
        })
        y_c, x_c, h, w = box
        print(f"{int(k):>4}  {y_c:>6.0f}  {x_c:>5.0f}  {int(h):>4d}  "
              f"{int(w):>3d}  {mean_phi:>+9.3f}  {abs(mean_phi):>7.3f}  "
              f"{coherence:>5.2f}  {std:>5.2f}")
    return traces


def _draw_traces(
    parent,           # plt.Figure or matplotlib.figure.SubFigure
    traces: list[dict],
    cols: int,
    title: str,
) -> None:
    """Render a grid of per-box phi(u) scatter plots inside `parent`."""
    n = len(traces)
    cols = max(1, cols)
    rows = max(1, int(np.ceil(n / cols)))
    axes = parent.subplots(rows, cols, squeeze=False)
    for ax, tr in zip(axes.flat, traces):
        ax.axhline(0.0, color="0.7", lw=0.7)
        ax.axhline(tr["mean_phi"], color="tab:red", lw=0.8, linestyle="--",
                   label=f"mean={tr['mean_phi']:+.2f}")
        amp = tr["amp"]
        sizes = 6 + 30 * (amp / (amp.max() + 1e-12))
        ax.scatter(tr["u"], tr["phi"], s=sizes, c="tab:blue",
                   edgecolors="none", alpha=0.85)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_title(
            f"#{tr['idx']}  y={tr['box'][0]:.0f}  x={tr['box'][1]:.0f}  "
            f"coh={tr['coherence']:.2f}  std={tr['std']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("azimuth row")
        ax.set_ylabel(r"$\Delta\varphi$  [rad]")
        ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    parent.suptitle(title, fontsize=11)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=None)
    parser.add_argument("--x-min", type=float, default=90.0)
    parser.add_argument("--x-max", type=float, default=140.0)
    parser.add_argument("--invert", action="store_true",
                        help="Select boxes OUTSIDE [x_min, x_max] instead.")
    parser.add_argument("--sample", type=int, default=0,
                        help="If >0, pick this many boxes uniformly at random "
                             "from the selection.")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for --sample.")
    parser.add_argument("--with-others", type=int, default=0,
                        help="If >0, also open a second figure with N random "
                             "boxes from the OTHER side of [x_min, x_max].")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--cols", type=int, default=4,
                        help="Number of columns in the per-box grid.")
    args = parser.parse_args()

    src = args.path or _latest_npz(DEFAULT_SAVE_DIR)
    if src is None or not src.exists():
        raise SystemExit(
            f"No .npz found at {args.path or DEFAULT_SAVE_DIR}. "
            "Run scripts/shear_averaging.py first."
        )

    data = np.load(src)
    d_phase = data["d_phase"]
    amp_raw = data["amp_raw"]
    boxes = data["boxes_yxhw"]

    in_band = (boxes[:, 1] >= args.x_min) & (boxes[:, 1] <= args.x_max)
    rng = np.random.default_rng(args.seed)

    def _select(mask: np.ndarray, sample: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.flatnonzero(mask)
        if sample and sample < len(idx):
            pick = np.sort(rng.choice(len(idx), size=sample, replace=False))
            idx = idx[pick]
        return idx, boxes[idx]

    primary_mask = ~in_band if args.invert else in_band
    primary_idx, primary_boxes = _select(primary_mask, args.sample)
    primary_label = (
        f"x∉[{args.x_min:g}, {args.x_max:g}]" if args.invert
        else f"x∈[{args.x_min:g}, {args.x_max:g}]"
    )
    print(f"Loaded {src.name}:  total boxes={len(boxes)}")

    if len(primary_boxes) == 0:
        return

    figs_to_save: list[tuple[plt.Figure, str]] = []

    def _make_figure(traces: list[dict], title: str, suffix: str) -> None:
        n = len(traces)
        cols = max(1, args.cols)
        rows = max(1, int(np.ceil(n / cols)))
        fig = plt.figure(figsize=(4.0 * cols, 2.4 * rows),
                         constrained_layout=True)
        _draw_traces(fig, traces, args.cols, title)
        figs_to_save.append((fig, suffix))

    primary_traces = _compute_traces(
        primary_idx, primary_boxes, d_phase, amp_raw, primary_label,
    )
    _make_figure(
        primary_traces,
        f"d_phase at brightest range column per row, "
        f"boxes with {primary_label}  ({len(primary_traces)} boxes)\n"
        f"source: {src.name}",
        suffix="_primary",
    )

    # Optional second figure: random sample from the OTHER side of [x_min, x_max].
    if args.with_others > 0:
        other_idx, other_boxes = _select(~primary_mask, args.with_others)
        other_label = (
            f"x∈[{args.x_min:g}, {args.x_max:g}]" if args.invert
            else f"x∉[{args.x_min:g}, {args.x_max:g}]"
        )
        other_traces = _compute_traces(
            other_idx, other_boxes, d_phase, amp_raw,
            f"{other_label}  (random sample of {args.with_others}, seed={args.seed})",
        )
        _make_figure(
            other_traces,
            f"d_phase at brightest range column per row, "
            f"boxes with {other_label}  "
            f"(random sample of {len(other_traces)}, seed={args.seed})\n"
            f"source: {src.name}",
            suffix="_others",
        )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        for fig, suffix in figs_to_save:
            out = (
                args.save if len(figs_to_save) == 1
                else args.save.with_name(f"{args.save.stem}{suffix}{args.save.suffix}")
            )
            fig.savefig(out, dpi=150)
            print(f"Saved → {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
