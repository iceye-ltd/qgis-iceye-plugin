#!/usr/bin/env python3
"""Compare two QLK PNGs by dividing them and displaying the difference.

Usage:
    python diff_qlk_pngs.py [png_a] [png_b]

Defaults to the two ICEYE QLK PNGs the user asked about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

DEFAULT_A = Path(
    "/home/odogan/Desktop/moving/reprocess_slccog/"
    "ICEYE_USRDN2_20260709T090353Z_10094570_X44_SLF_QLK.png"
)
DEFAULT_B = Path(
    "/home/odogan/Desktop/moving/"
    "ICEYE_USRDN2_20260709T090353Z_10094570_X44_SLF_QLK.png"
)


def _load_gray(path: Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ("L", "I", "I;16", "F"):
        img = img.convert("L")
    return np.asarray(img, dtype=np.float32)


def main(argv: list[str]) -> int:
    path_a = Path(argv[1]) if len(argv) > 1 else DEFAULT_A
    path_b = Path(argv[2]) if len(argv) > 2 else DEFAULT_B

    print(f"A: {path_a}")
    print(f"B: {path_b}")

    a = _load_gray(path_a)
    b = _load_gray(path_b)
    print(f"shape A={a.shape}, B={b.shape}, dtype A={a.dtype}, B={b.dtype}")

    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if a.shape != (h, w):
            print(f"cropping A from {a.shape} to ({h}, {w})")
            a = a[:h, :w]
        else:
            print(f"cropping B from {b.shape} to ({h}, {w})")
            b = b[:h, :w]

    eps = 1e-3
    ratio = a / np.maximum(b, eps)
    log_ratio = np.log2(np.maximum(ratio, eps))

    abs_diff = a - b

    finite = ratio[np.isfinite(ratio)]
    print(
        "ratio A/B  ->  "
        f"min={finite.min():.4f}  max={finite.max():.4f}  "
        f"mean={finite.mean():.4f}  median={np.median(finite):.4f}  "
        f"std={finite.std():.4f}"
    )
    print(
        "abs diff A-B  ->  "
        f"min={abs_diff.min():.2f}  max={abs_diff.max():.2f}  "
        f"mean={abs_diff.mean():.2f}  std={abs_diff.std():.2f}"
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), constrained_layout=True)

    vmax_ab = max(a.max(), b.max())
    axes[0, 0].imshow(a, cmap="gray", vmin=0, vmax=vmax_ab)
    axes[0, 0].set_title(f"A: {path_a.parent.name}/{path_a.name}", fontsize=8)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(b, cmap="gray", vmin=0, vmax=vmax_ab)
    axes[0, 1].set_title(f"B: {path_b.parent.name}/{path_b.name}", fontsize=8)
    axes[0, 1].axis("off")

    r_lim = float(np.nanpercentile(np.abs(log_ratio), 99))
    r_lim = max(r_lim, 0.1)
    im2 = axes[1, 0].imshow(log_ratio, cmap="RdBu_r", vmin=-r_lim, vmax=r_lim)
    axes[1, 0].set_title(f"log2(A / B)  (±{r_lim:.2f})")
    axes[1, 0].axis("off")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.02)

    d_lim = float(np.nanpercentile(np.abs(abs_diff), 99))
    d_lim = max(d_lim, 1.0)
    im3 = axes[1, 1].imshow(abs_diff, cmap="RdBu_r", vmin=-d_lim, vmax=d_lim)
    axes[1, 1].set_title(f"A - B  (±{d_lim:.1f})")
    axes[1, 1].axis("off")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.02)

    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
