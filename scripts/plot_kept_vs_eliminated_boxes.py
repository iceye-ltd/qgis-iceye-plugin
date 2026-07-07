"""Plot kept (cyan) + eliminated (red) boxes on a blank canvas.

Standalone reproduction of the box overlay from
``<stem>_kept_vs_eliminated.png`` without needing ``|s_degraded|``: since
``s_degraded`` is not persisted in the ``<stem>.npz`` payload, this
script only draws an empty ``imshow`` sized to ``s_degraded_shape`` (or
``(max(y_c+h/2), max(x_c+w/2))`` when the NPZ is absent) so you can
zoom / pan and read exact pixel coordinates off the axes.

Inputs
------
--csv   : path to ``<stem>_box_stats.csv`` (required). All boxes come
          from here — rows with ``disposition == "kept"`` are drawn
          cyan, everything else red.
--npz   : path to ``<stem>.npz`` (optional). Only read for
          ``s_degraded_shape`` to size the axes; if omitted the shape
          is inferred from the CSV.
--out   : output PNG path. Defaults to a sibling of ``--csv`` named
          ``<stem>_kept_vs_eliminated_boxes_only.png``.
--dpi   : output DPI (default 200).

The overlay uses the same ``plt.Rectangle`` geometry as
``shear_averaging._overlay_boxes`` — top-left corner at
``(x_c - w/2 - 0.5, y_c - h/2 - 0.5)`` — so positions match the
production figure pixel-for-pixel.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_boxes_from_csv(
    path_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (boxes_kept, init_idx_kept, boxes_elim, init_idx_elim).

    Each ``boxes_*`` array has shape ``(K, 4)`` with columns
    ``(y_c, x_c, h, w)`` matching ``boxes_yxhw``.
    """
    kept_rows: list[tuple[int, int, int, int, int]] = []
    elim_rows: list[tuple[int, int, int, int, int]] = []
    with path_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                y_c = int(row["y_c"])
                x_c = int(row["x_c"])
                h = int(row["h"])
                w = int(row["w"])
                init_idx = int(row["init_idx"])
            except (KeyError, ValueError):
                continue
            entry = (init_idx, y_c, x_c, h, w)
            if row["disposition"] == "kept":
                kept_rows.append(entry)
            else:
                elim_rows.append(entry)

    def _split(rows: list[tuple[int, int, int, int, int]]):
        if not rows:
            return (
                np.empty((0, 4), dtype=np.int64),
                np.empty((0,), dtype=np.int64),
            )
        arr = np.asarray(rows, dtype=np.int64)
        return arr[:, 1:5], arr[:, 0]

    boxes_kept, idx_kept = _split(kept_rows)
    boxes_elim, idx_elim = _split(elim_rows)
    return boxes_kept, idx_kept, boxes_elim, idx_elim


def _overlay_boxes(
    ax: plt.Axes,
    boxes_yxhw: np.ndarray,
    init_idx: np.ndarray | None,
    color: str,
    lw: float,
    label_boxes: bool,
) -> None:
    """Draw axis-aligned rectangles, one per row of ``boxes_yxhw``.

    Geometry mirrors ``shear_averaging._overlay_boxes`` so the
    resulting overlay matches ``<stem>_kept_vs_eliminated.png``
    pixel-for-pixel. When ``label_boxes`` is set, the CSV
    ``init_idx`` is printed at the box's top-left corner.
    """
    for k, (y, x, h, w) in enumerate(np.atleast_2d(boxes_yxhw)):
        ax.add_patch(plt.Rectangle(
            (x - w / 2 - 0.5, y - h / 2 - 0.5), w, h,
            fill=False, edgecolor=color, linewidth=lw,
        ))
        if label_boxes and init_idx is not None:
            ax.text(
                x - w / 2 - 0.5, y - h / 2 - 0.5,
                str(int(init_idx[k])),
                color=color, fontsize=5,
                ha="left", va="bottom",
                clip_on=True,
            )


def _infer_shape(
    boxes_all: np.ndarray, npz_path: Path | None,
) -> tuple[int, int]:
    """Prefer ``s_degraded_shape`` from the NPZ; otherwise bound-box the CSV."""
    if npz_path is not None and npz_path.exists():
        with np.load(npz_path) as data:
            if "s_degraded_shape" in data.files:
                shape = tuple(int(v) for v in data["s_degraded_shape"])
                return shape[0], shape[1]
    if boxes_all.size == 0:
        return 100, 100
    y_max = int(np.max(boxes_all[:, 0] + boxes_all[:, 2] / 2)) + 8
    x_max = int(np.max(boxes_all[:, 1] + boxes_all[:, 3] / 2)) + 8
    return max(y_max, 32), max(x_max, 32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Draw kept (cyan) + eliminated (red) boxes from "
            "<stem>_box_stats.csv on a blank imshow. Use "
            "--label to print init_idx next to each rectangle."
        ),
    )
    parser.add_argument(
        "--csv", type=Path, required=True,
        help="Path to <stem>_box_stats.csv.",
    )
    parser.add_argument(
        "--npz", type=Path, default=None,
        help=(
            "Optional path to <stem>.npz — read only for "
            "s_degraded_shape so the axes match the original figure. "
            "Auto-discovered next to --csv if not supplied."
        ),
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help=(
            "Output PNG path. Defaults to "
            "<stem>_kept_vs_eliminated_boxes_only.png next to --csv."
        ),
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Output PNG DPI (default 200).",
    )
    parser.add_argument(
        "--label", action="store_true",
        help=(
            "Print the CSV init_idx at each rectangle's top-left corner. "
            "Handy for locating a specific box after zooming."
        ),
    )
    parser.add_argument(
        "--kept-only", action="store_true",
        help="Skip the eliminated (red) overlay.",
    )
    parser.add_argument(
        "--eliminated-only", action="store_true",
        help="Skip the kept (cyan) overlay.",
    )
    args = parser.parse_args()

    csv_path: Path = args.csv.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    npz_path: Path | None = args.npz
    if npz_path is None:
        guess = csv_path.parent / (
            csv_path.name.replace("_box_stats.csv", ".npz")
        )
        npz_path = guess if guess.exists() else None

    boxes_kept, idx_kept, boxes_elim, idx_elim = _load_boxes_from_csv(
        csv_path,
    )
    boxes_all = np.concatenate([boxes_kept, boxes_elim], axis=0)
    n_az, n_rg = _infer_shape(boxes_all, npz_path)

    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    ax.imshow(
        np.zeros((n_az, n_rg), dtype=np.uint8),
        cmap="gray", vmin=0, vmax=1, aspect="auto",
        interpolation="nearest",
    )
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    fig.suptitle(
        f"boxes only — kept (cyan) vs eliminated (red) — {csv_path.name}",
        fontsize=11,
    )
    ax.set_title(
        f"{len(boxes_kept)} kept / {len(boxes_elim)} eliminated  "
        f"(axes on the {n_az} × {n_rg} s_degraded grid)"
    )

    if not args.kept_only and boxes_elim.size:
        _overlay_boxes(
            ax, boxes_elim, idx_elim,
            color="red", lw=0.9, label_boxes=args.label,
        )
    if not args.eliminated_only and boxes_kept.size:
        _overlay_boxes(
            ax, boxes_kept, idx_kept,
            color="cyan", lw=1.1, label_boxes=args.label,
        )

    handles = [
        plt.Line2D(
            [0], [0], color="cyan", lw=1.4,
            label=f"kept ({len(boxes_kept)})",
        ),
        plt.Line2D(
            [0], [0], color="red", lw=1.4,
            label=f"eliminated ({len(boxes_elim)})",
        ),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8)

    out_path: Path
    if args.out is not None:
        out_path = args.out.expanduser().resolve()
    else:
        out_path = csv_path.with_name(
            csv_path.name.replace(
                "_box_stats.csv", "_kept_vs_eliminated_boxes_only.png",
            )
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved → {out_path}  ({n_az} × {n_rg}, kept={len(boxes_kept)}, elim={len(boxes_elim)})")

    plt.show()


if __name__ == "__main__":
    main()
