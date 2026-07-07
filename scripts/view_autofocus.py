"""Visualize SAR data at three autofocus stages.

Reads three .npy files saved by `core/autofocus.py`:
    - data_before_rdc.npy
    - data_after_rdc.npy
    - data_after_pga.npy

and displays a 2 x 3 grid:
    top row    : spatial magnitude
    bottom row : |FFT axis=0| (azimuth-frequency spectrum)

Contrast stretch is `mean + 4*std` per panel.
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = Path("/home/odogan/Desktop/ship_focus")
DEFAULT_SAVE_DIR = Path("/home/odogan/Desktop/ship_focusing/4439676")


def _default_save_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return DEFAULT_SAVE_DIR / f"stages_{ts}_{suffix}.png"
STAGES: tuple[tuple[str, str], ...] = (
    ("before_rdc", "Before RDC"),
    ("after_rdc", "After RDC"),
    ("after_pga", "After PGA"),
)


def _show(ax: plt.Axes, data: np.ndarray, title: str) -> None:
    img = np.abs(data)
    vmax = float(img.mean() + 4 * img.std())
    im = ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Magnitude")


def _show_fft_axis0(ax: plt.Axes, data: np.ndarray, title: str) -> None:
    spectrum = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    img = np.abs(spectrum)
    vmax = float(img.mean() + 4 * img.std())
    im = ax.imshow(img, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Range bin")
    ax.set_ylabel("Azimuth freq bin (fftshifted)")
    plt.colorbar(im, ax=ax, label="|FFT|")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                        help="Directory containing the .npy stage files.")
    parser.add_argument(
        "--save", type=Path,
        default=_default_save_path(),
        help=f"PNG output path. Defaults to {DEFAULT_SAVE_DIR}/stages_<timestamp>_<random>.png. "
             "Pass --no-save to skip saving.",
    )
    parser.add_argument("--no-save", dest="save", action="store_const", const=None,
                        help="Disable PNG saving.")
    parser.add_argument("--no-show", action="store_true",
                        help="Skip interactive plt.show() (still saves the PNG).")
    args = parser.parse_args()

    arrays = []
    for stem, label in STAGES:
        path = args.dir / f"data_{stem}.npy"
        a = np.load(path)
        print(f"{label:>11s} ({path.name}): shape={a.shape} dtype={a.dtype}")
        arrays.append(a)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    for col, ((_, label), arr) in enumerate(zip(STAGES, arrays)):
        _show(axes[0, col], arr, f"{label}  (spatial)")
        _show_fft_axis0(axes[1, col], arr, f"{label}  |FFT axis=0|")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
