"""Companion to ``diag_mosaic_top100.py`` — build the amplitude mosaic.

Same 10×10 grid, but each tile is the LEFT panel of the pipeline's
autofocus PNG (``|I| before``). Combined with the phase mosaic this
lets a reviewer scan the 100 kept boxes for real vs. spurious targets.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RUN_DIR = Path("/home/odogan/Desktop/ship_focusing/4439676")
PER_BOX = RUN_DIR / "WTW3YQ_per_box"
CSV_PATH = RUN_DIR / "WTW3YQ_box_stats.csv"
OUT_DIR = Path(__file__).resolve().parent / "missing_doubles_out"

N_ROWS = 10
N_COLS = 10


def load_kept_stats() -> dict[int, dict]:
    kept: dict[int, dict] = {}
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if r["disposition"] != "kept" or not r["final_idx"]:
                continue
            fi = int(r["final_idx"])
            kept[fi] = {
                "init_idx": int(r["init_idx"]),
                "y_c": int(r["y_c"]),
                "x_c": int(r["x_c"]),
                "h": int(r["h"]),
                "w": int(r["w"]),
                "slope_n": float(r["slope_total_rad"]) if r["slope_total_rad"] else float("nan"),
                "frac": float(r["frac_within_1rad_coh"]) if r["frac_within_1rad_coh"] else float("nan"),
                "gain": float(r["contrast_gain"]) if r["contrast_gain"] else float("nan"),
            }
    return kept


def crop_left_panel(img: np.ndarray) -> np.ndarray:
    """Extract just the |I| BEFORE panel (left half of the autofocus PNG). The pipeline writes autofocus PNGs as two side-by-side panels with a common title bar above; the |I| BEFORE panel is essentially the left half — we crop by columns only so it always works."""
    H, W = img.shape[:2]
    return img[:, : W // 2]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = load_kept_stats()
    print(f"Loaded {len(kept)} kept boxes from CSV")

    fig, axes = plt.subplots(
        N_ROWS, N_COLS, figsize=(N_COLS * 2.4, N_ROWS * 2.0),
        constrained_layout=True,
    )
    fig.suptitle(
        f"WTW3YQ  —  |I| before-AF for all 100 kept boxes "
        f"(sorted by final_idx)\n"
        f"caption: fi=<final_idx>  init=<init_idx>  y_c×x_c  h×w  "
        f"score=<frac_coh>",
        fontsize=11,
    )

    missing = 0
    for fi in range(N_ROWS * N_COLS):
        ax = axes[fi // N_COLS, fi % N_COLS]
        ax.axis("off")
        png = PER_BOX / f"box_{fi:03d}_autofocus.png"
        info = kept.get(fi, {})
        if not png.exists():
            missing += 1
            ax.set_title(f"fi={fi}  (no PNG)", fontsize=7)
            continue
        img = mpimg.imread(png)
        panel = crop_left_panel(img)
        ax.imshow(panel)
        if info:
            ax.set_title(
                f"fi={fi}  init={info['init_idx']}\n"
                f"y={info['y_c']} x={info['x_c']}  "
                f"h={info['h']}×w={info['w']}\n"
                f"score={info['frac']:.2f}  "
                f"|s·n|={abs(info['slope_n']):.1f}  gain={info['gain']:.1f}",
                fontsize=6,
            )
        else:
            ax.set_title(f"fi={fi}", fontsize=6)

    print(f"  {missing}/{N_ROWS * N_COLS} tiles missing")
    out_path = OUT_DIR / "mosaic_top100_amp.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
