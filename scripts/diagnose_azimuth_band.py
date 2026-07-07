"""Diagnose whether the azimuth-FFT band actually narrows after autofocus,
or whether it just looks narrower due to per-panel contrast stretching.

Tests, all using only the two saved .npy files:

1. Total energy: sum(|x|^2) before vs after  (Parseval check).
2. Marginal azimuth power spectrum:
       P(k) = sum_j |FFT(x, axis=0)[k, j]|^2
   plotted on log scale, before vs after, overlaid.
3. Integrated -3 dB / -10 dB / -20 dB widths of P(k).
4. Side-by-side |FFT axis=0| images using a SHARED vmin/vmax (computed
   from both arrays jointly) so the contrast stretch can no longer hide
   or invent a band-narrowing effect.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DIR = Path("/home/odogan/Desktop/ship_focus")


def fft_axis0(data: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)


def fft_axis1(data: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(data, axis=1), axes=1)


def width_at_db(power: np.ndarray, db: float) -> int:
    """Number of bins where power >= peak * 10**(db/10) (db is negative)."""
    thresh = power.max() * 10 ** (db / 10.0)
    return int(np.sum(power >= thresh))


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

    e_before = float(np.sum(np.abs(before) ** 2))
    e_after = float(np.sum(np.abs(after) ** 2))
    print("\n[1] Total energy (Parseval check)")
    print(f"    sum|before|^2 = {e_before:.6e}")
    print(f"    sum|after |^2 = {e_after:.6e}")
    print(f"    ratio after/before = {e_after / e_before:.6f}")

    Fb0 = fft_axis0(before)
    Fa0 = fft_axis0(after)
    Fb1 = fft_axis1(before)
    Fa1 = fft_axis1(after)

    Pb0 = np.sum(np.abs(Fb0) ** 2, axis=1)
    Pa0 = np.sum(np.abs(Fa0) ** 2, axis=1)
    Pb1 = np.sum(np.abs(Fb1) ** 2, axis=0)
    Pa1 = np.sum(np.abs(Fa1) ** 2, axis=0)

    def _print_widths(name: str, Pb_: np.ndarray, Pa_: np.ndarray) -> None:
        print(f"\n[2] {name} marginal power spectrum widths (#bins, total = "
              f"{Pb_.size})")
        for db in (-3.0, -10.0, -20.0):
            wb = width_at_db(Pb_, db)
            wa = width_at_db(Pa_, db)
            print(f"    {db:6.1f} dB :  before={wb:4d}  after={wa:4d}  "
                  f"ratio={wa / max(wb, 1):.3f}")

    _print_widths("axis=0 (rows / vertical)", Pb0, Pa0)
    _print_widths("axis=1 (cols / horizontal)", Pb1, Pa1)

    Fb = Fb0
    Fa = Fa0
    Pb = Pb0
    Pa = Pa0
    Pb_n = Pb / Pb.max()
    Pa_n = Pa / Pa.max()

    Mb = np.abs(Fb)
    Ma = np.abs(Fa)
    joint = np.concatenate([Mb.ravel(), Ma.ravel()])
    vmin = 0.0
    vmax = float(joint.mean() + 4 * joint.std())
    print(f"\n[4] Shared display stretch: vmin={vmin}, vmax={vmax:.4g}")
    print("    (per-image vmax would be:")
    print(f"      before: {Mb.mean() + 4 * Mb.std():.4g}")
    print(f"      after : {Ma.mean() + 4 * Ma.std():.4g})")

    fig = plt.figure(figsize=(14, 11), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])

    ax_b = fig.add_subplot(gs[0, 0])
    ax_a = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, :])

    im_b = ax_b.imshow(Mb, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax_b.set_title("Before  |FFT axis=0|  (shared stretch)")
    ax_b.set_xlabel("Range bin")
    ax_b.set_ylabel("Azimuth freq bin (fftshifted)")
    plt.colorbar(im_b, ax=ax_b, label="|FFT|")

    im_a = ax_a.imshow(Ma, cmap="gray", aspect="auto", vmin=vmin, vmax=vmax)
    ax_a.set_title("After  |FFT axis=0|  (shared stretch)")
    ax_a.set_xlabel("Range bin")
    plt.colorbar(im_a, ax=ax_a, label="|FFT|")

    k = np.arange(Pb.size) - Pb.size // 2
    ax_p.semilogy(k, Pb_n, label="before", color="C0")
    ax_p.semilogy(k, Pa_n, label="after", color="C1")
    for db in (-3.0, -10.0, -20.0):
        ax_p.axhline(10 ** (db / 10), color="gray", lw=0.5, ls="--")
        ax_p.text(k[0], 10 ** (db / 10), f" {db:.0f} dB",
                  va="bottom", ha="left", color="gray", fontsize=8)
    ax_p.set_xlabel("Azimuth frequency bin (fftshifted, 0 = DC)")
    ax_p.set_ylabel("Normalized power (per peak)")
    ax_p.set_title("Marginal azimuth power spectrum  P(k) = Σ_j |FFT|²")
    ax_p.grid(True, which="both", lw=0.3)
    ax_p.legend()
    ax_p.set_xlim(k[0], k[-1])
    ax_p.set_ylim(1e-6, 2.0)

    if args.save is not None:
        fig.savefig(args.save, dpi=150)
        print(f"\nSaved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
