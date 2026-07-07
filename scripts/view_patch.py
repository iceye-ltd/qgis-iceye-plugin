"""Visualize a single SAR patch saved by `core/autofocus.py`.

Reads one .npy file (complex SLC) and displays a 1 x 2 grid:
    left  : spatial magnitude
    right : |FFT axis=0| (azimuth-frequency spectrum)

Contrast stretch is `mean + 4*std` per panel.
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_PATCH = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_141553_997630.npy"
)
DEFAULT_SAVE_DIR = Path("/home/odogan/Desktop/ship_focusing/4439676")


def _default_save_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return DEFAULT_SAVE_DIR / f"patch_{ts}_{suffix}.png"


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
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATCH,
        help=f"Path to a single .npy patch. Defaults to {DEFAULT_PATCH}.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=_default_save_path(),
        help=(
            f"PNG output path. Defaults to {DEFAULT_SAVE_DIR}/patch_<timestamp>_<random>.png. "
            "Pass --no-save to skip saving."
        ),
    )
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_const",
        const=None,
        help="Disable PNG saving.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Skip interactive plt.show() (still saves the PNG).",
    )
    args = parser.parse_args()

    arr = np.load(args.path)
    label = args.path.stem
    print(f"{label} ({args.path.name}): shape={arr.shape} dtype={arr.dtype}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    _show(axes[0], arr, f"{label}  (spatial)")
    _show_fft_axis0(axes[1], arr, f"{label}  |FFT axis=0|")

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
