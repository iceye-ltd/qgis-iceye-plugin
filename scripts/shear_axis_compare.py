"""Compare shear-product axes:

    P_v(u, v) = G(u, v) * conj(G(u, v - a))    (shift along RANGE pixel,    axis=1)
    P_u(u, v) = G(u, v) * conj(G(u - a, v))    (shift along AZIMUTH freq,   axis=0)

Both are computed from G = FFT_axis0(s). Side-by-side magnitude images make
clear which axis the shift is on.
"""

from __future__ import annotations

import argparse
import uuid
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_PATCH = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/patches/data_20260617_143542_433755.npy"
)
DEFAULT_SAVE_DIR = Path("/home/odogan/Desktop/ship_focusing/4439676")


def _default_save_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_SAVE_DIR / f"shear_axis_compare_{ts}_{uuid.uuid4().hex[:6]}.png"


def _show(ax: plt.Axes, p: np.ndarray, title: str) -> None:
    mag = np.log1p(np.abs(p))
    vmax = float(mag.mean() + 4 * mag.std())
    im = ax.imshow(mag, cmap="gray", aspect="auto", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("v (range pixel)")
    ax.set_ylabel("u (azimuth freq, fftshifted)")
    plt.colorbar(im, ax=ax, label="log(1+|P|)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATCH)
    parser.add_argument("-a", "--shift", type=int, default=1)
    parser.add_argument("--save", type=Path, default=_default_save_path())
    parser.add_argument("--no-save", dest="save", action="store_const", const=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    s = np.load(args.path)
    g = np.fft.fft(s, axis=0)

    p_range = g * np.conj(np.roll(g, args.shift, axis=1))
    p_azim = g * np.conj(np.roll(g, args.shift, axis=0))
    p_range = np.fft.fftshift(p_range, axes=0)
    p_azim = np.fft.fftshift(p_azim, axes=0)

    print(f"{args.path.name}: shape={s.shape}, a={args.shift}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    _show(
        axes[0],
        p_range,
        f"Shift in RANGE: |G(u,v) · G*(u, v-a)|   axis=1, a={args.shift}",
    )
    _show(
        axes[1],
        p_azim,
        f"Shift in AZIMUTH: |G(u,v) · G*(u-a, v)|   axis=0, a={args.shift}",
    )

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved figure to {args.save}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
