"""Mosaic all 100 phase plots of the WTW3YQ run into one figure.

The user is skeptical that ``max_focus_targets = 100`` actually
corresponds to 100 real movers. This script:

  * Reads the per-box dphase PNGs the pipeline wrote for the 100 kept
    boxes (``box_000_dphase.png`` … ``box_099_dphase.png``).
  * Reads the corresponding scores from the pipeline's box-stats CSV
    (frac_within_1rad_coh, slope, coordinates) plus contrast_gain and
    velocity fields.
  * Renders one big 10×10 grid of thumbnails so we can scan the full
    top-100 in a single glance and see how many phase ramps look like
    real coherent movers vs. weak / noisy fits that could be sacrificed
    to make room for the 3 missing doubles.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

RUN_DIR = Path("/home/odogan/Desktop/ship_focusing/4439676")
PER_BOX = RUN_DIR / "WTW3YQ_per_box"
CSV_PATH = RUN_DIR / "WTW3YQ_box_stats.csv"
OUT_DIR = Path(__file__).resolve().parent / "missing_doubles_out"

N_ROWS = 10
N_COLS = 10


def load_kept_stats() -> dict[int, dict]:
    """Map final_idx (0..99) -> {init_idx, y_c, x_c, h, w, slope_n, frac, gain}."""
    kept: dict[int, dict] = {}
    with CSV_PATH.open() as f:
        for r in csv.DictReader(f):
            if r["disposition"] != "kept" or not r["final_idx"]:
                continue
            fi = int(r["final_idx"])
            slope_n = float(r["slope_total_rad"]) if r["slope_total_rad"] else float("nan")
            frac = float(r["frac_within_1rad_coh"]) if r["frac_within_1rad_coh"] else float("nan")
            gain = float(r["contrast_gain"]) if r["contrast_gain"] else float("nan")
            v_az = float(r["v_az_mps"]) if r["v_az_mps"] else float("nan")
            v_rg = float(r["v_rg_mps"]) if r["v_rg_mps"] else float("nan")
            kept[fi] = {
                "init_idx": int(r["init_idx"]),
                "y_c": int(r["y_c"]),
                "x_c": int(r["x_c"]),
                "h": int(r["h"]),
                "w": int(r["w"]),
                "slope_n": slope_n,
                "frac": frac,
                "gain": gain,
                "v_az": v_az, "v_rg": v_rg,
            }
    return kept


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = load_kept_stats()
    print(f"Loaded {len(kept)} kept boxes from CSV")

    fig, axes = plt.subplots(
        N_ROWS, N_COLS, figsize=(N_COLS * 2.6, N_ROWS * 1.9),
        constrained_layout=True,
    )
    fig.suptitle(
        f"WTW3YQ  —  all 100 kept boxes' phase plots (sorted by final_idx)\n"
        f"caption per tile: fi=<final_idx>  init=<init_idx>  score=<frac_coh>  "
        f"|slope·n|=<rad>  gain=<contrast_gain>",
        fontsize=11,
    )

    for fi in range(N_ROWS * N_COLS):
        ax = axes[fi // N_COLS, fi % N_COLS]
        ax.axis("off")
        png = PER_BOX / f"box_{fi:03d}_dphase.png"
        if not png.exists():
            ax.set_title(f"fi={fi}  (missing)", fontsize=7)
            continue
        img = mpimg.imread(png)
        ax.imshow(img)
        info = kept.get(fi, {})
        if info:
            ax.set_title(
                f"fi={fi}  init={info['init_idx']}  score={info['frac']:.2f}\n"
                f"|s·n|={abs(info['slope_n']):.1f} rad  "
                f"gain={info['gain']:.1f}",
                fontsize=6,
            )
        else:
            ax.set_title(f"fi={fi}", fontsize=6)

    out_path = OUT_DIR / "mosaic_top100_dphase.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
