"""Compare |FFT| along axis=0 vs axis=1 for before/after autofocus arrays.

Goal: identify which array axis is the genuinely band-limited direction
(typically range = chirp bandwidth) versus the full-band direction
(typically azimuth = PRF). If the autofocus is correcting along the
wrong axis, that mismatch will be obvious here.

Layout:
    Top row    : |FFT axis=0| (vertical-direction spectrum) for before / after
    Middle row : |FFT axis=1| (horizontal-direction spectrum) for before / after
    Bottom row : marginal log-power profiles for axis=0 and axis=1

All four imshow panels share a single vmin/vmax computed jointly across
all four magnitude images so contrast stretching cannot hide differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = Path("/home/odogan/Desktop/ship_focus")


def fft_axis(data: np.ndarray, axis: int) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(data, axis=axis), axes=axis)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before", type=Path, default=DEFAULT_DIR / "data_before_autofocus.npy"
    )
    parser.add_argument(
        "--after", type=Path, default=DEFAULT_DIR / "data_after_autofocus.npy"
    )
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    before = np.load(args.before)
    after = np.load(args.after)
    print(f"before: shape={before.shape} dtype={before.dtype}")
    print(f"after:  shape={after.shape} dtype={after.dtype}")

    Mb0 = np.abs(fft_axis(before, 0))
    Ma0 = np.abs(fft_axis(after, 0))
    Mb1 = np.abs(fft_axis(before, 1))
    Ma1 = np.abs(fft_axis(after, 1))

    joint = np.concatenate(
        [Mb0.ravel(), Ma0.ravel(), Mb1.ravel(), Ma1.ravel()]
    )
    vmax = float(joint.mean() + 4 * joint.std())
    print(f"shared display vmax = {vmax:.4g}")

    Pb0 = np.sum(Mb0 ** 2, axis=1)
    Pa0 = np.sum(Ma0 ** 2, axis=1)
    Pb1 = np.sum(Mb1 ** 2, axis=0)
    Pa1 = np.sum(Ma1 ** 2, axis=0)

    fig = plt.figure(figsize=(14, 16), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 0.9])

    def _panel(row: int, col: int, M: np.ndarray, title: str) -> None:
        ax = fig.add_subplot(gs[row, col])
        im = ax.imshow(M, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, label="|FFT|")

    _panel(0, 0, Mb0, "Before  |FFT axis=0|  (vertical FFT)")
    _panel(0, 1, Ma0, "After   |FFT axis=0|  (vertical FFT)")
    _panel(1, 0, Mb1, "Before  |FFT axis=1|  (horizontal FFT)")
    _panel(1, 1, Ma1, "After   |FFT axis=1|  (horizontal FFT)")

    ax_p0 = fig.add_subplot(gs[2, 0])
    k0 = np.arange(Pb0.size) - Pb0.size // 2
    ax_p0.semilogy(k0, Pb0 / Pb0.max(), label="before", color="C0")
    ax_p0.semilogy(k0, Pa0 / Pa0.max(), label="after", color="C1")
    ax_p0.set_title("Marginal power along axis=0 (rows)")
    ax_p0.set_xlabel("Vertical freq bin (fftshifted, 0=DC)")
    ax_p0.set_ylabel("normalized power")
    ax_p0.set_ylim(1e-6, 2)
    ax_p0.grid(True, which="both", lw=0.3)
    ax_p0.legend()

    ax_p1 = fig.add_subplot(gs[2, 1])
    k1 = np.arange(Pb1.size) - Pb1.size // 2
    ax_p1.semilogy(k1, Pb1 / Pb1.max(), label="before", color="C0")
    ax_p1.semilogy(k1, Pa1 / Pa1.max(), label="after", color="C1")
    ax_p1.set_title("Marginal power along axis=1 (cols)")
    ax_p1.set_xlabel("Horizontal freq bin (fftshifted, 0=DC)")
    ax_p1.set_ylim(1e-6, 2)
    ax_p1.grid(True, which="both", lw=0.3)
    ax_p1.legend()

    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
