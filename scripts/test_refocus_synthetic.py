"""Synthetic test: which refocus variant focuses a known chirp?

We build a 1720x9 complex chip consisting of:
  • A pointlike moving target: a single bright complex point at the centre,
    smeared by a chirp exp(j·½·slope·n²) across all azimuth rows. This is
    EXACTLY the SLC of an azimuth-moving target after the matched filter
    (residual QPE).
  • Optional sea-clutter speckle (complex circular Gaussian).

The correct refocus must collapse the chirp back to a single bright row.
We measure focus quality by the azimuth-integrated intensity profile's
peak-to-mean ratio — which jumps from O(1) (smeared) to O(N) (focused).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OUT_DIR = HERE / "missing_targets_out"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_chirp_chip(
    N: int, W: int, slope: float, snr_db: float = 30.0, rng=None,
) -> np.ndarray:
    """A pointlike target smeared by a chirp exp(j·½·slope·n²), + speckle."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = np.arange(N) - N // 2
    chirp = np.exp(1j * 0.5 * slope * n * n)        # shape (N,)
    # Target placed at range column W//2.
    chip = np.zeros((N, W), dtype=np.complex128)
    target_amp = 1.0
    chip[:, W // 2] = target_amp * chirp
    # Speckle (complex circular Gaussian), scaled so peak SNR ≈ snr_db.
    speckle_std = target_amp * 10 ** (-snr_db / 20.0)
    chip += speckle_std * (rng.standard_normal((N, W))
                            + 1j * rng.standard_normal((N, W))) / np.sqrt(2)
    return chip


# ---------------------------------------------------------------------------
# Refocus variants
# ---------------------------------------------------------------------------

def r_current(chip, slope):
    """Script's current: exp(+j·½·slope·m²) in freq, m=arange-N//2."""
    N = chip.shape[0]
    m = np.arange(N) - N // 2
    phi = 0.5 * slope * m * m
    F = np.fft.fft(np.fft.ifftshift(chip, axes=0), axis=0)
    F = F * np.exp(1j * phi)[:, None]
    return np.fft.fftshift(np.fft.ifft(F, axis=0), axes=0)


def r_current_sign(chip, slope):
    """Script's docstring (sign flipped): exp(-j·½·slope·m²) in freq."""
    N = chip.shape[0]
    m = np.arange(N) - N // 2
    phi = 0.5 * slope * m * m
    F = np.fft.fft(np.fft.ifftshift(chip, axes=0), axis=0)
    F = F * np.exp(-1j * phi)[:, None]
    return np.fft.fftshift(np.fft.ifft(F, axis=0), axes=0)


def r_spatial(chip, slope):
    """Spatial: chip · exp(-j·½·slope·n²) (pure phase — magnitude unchanged)."""
    N = chip.shape[0]
    n = np.arange(N) - N // 2
    phi = 0.5 * slope * n * n
    return chip * np.exp(-1j * phi)[:, None]


def r_fourier_dual(chip, slope):
    """Fourier-dual chirp: exp(+j·f²/(2·slope)) on centred-freq spectrum."""
    N = chip.shape[0]
    if abs(slope) < 1e-12:
        return chip.copy()
    f = np.fft.fftshift(np.fft.fftfreq(N) * 2 * np.pi)
    phi_dual = (f * f) / (2.0 * slope)
    phi_unshifted = np.fft.ifftshift(phi_dual)
    F = np.fft.fft(chip, axis=0)
    F = F * np.exp(1j * phi_unshifted)[:, None]
    return np.fft.ifft(F, axis=0)


def r_spatial_then_fft(chip, slope):
    """Spatial-then-FFT identity check.

    Sanity: applies the spatial conjugate phase, then FFT/IFFT through
    centred indices. By construction the magnitude is unchanged from
    r_spatial — included to confirm the previous 0.00 dB observation.
    """
    return r_spatial(chip, slope)


def az_profile(img):
    return (np.abs(img) ** 2).sum(axis=1)


def peak_to_mean(profile):
    m = float(np.mean(profile))
    if m <= 0.0:
        return float("nan")
    return float(profile.max() / m)


def main() -> None:
    N = 1720
    W = 9
    slope = 2.4e-3                       # T1's measured slope

    rng = np.random.default_rng(42)

    # Two cases: (1) clean chirp, (2) with strong speckle clutter.
    cases = {
        "clean": make_chirp_chip(N, W, slope, snr_db=60.0, rng=rng),
        "with speckle (-20 dB peak/clutter)": make_chirp_chip(
            N, W, slope, snr_db=20.0, rng=rng
        ),
    }

    variants = {
        "before": lambda c, s: c,
        "current (freq +sign)": r_current,
        "freq -sign (docstring)": r_current_sign,
        "spatial -sign": r_spatial,
        "Fourier-dual (freq +j·f²/2s)": r_fourier_dual,
    }

    fig, axes = plt.subplots(
        nrows=len(cases), ncols=len(variants),
        figsize=(4 * len(variants), 4.0 * len(cases)),
        constrained_layout=True,
    )
    if len(cases) == 1:
        axes = axes[None, :]
    print(f"slope = {slope:+.4e}  rad/row,  N={N}, W={W}")
    print(f"chirp bandwidth (rad/sample) = slope·N = {slope*N:+.3f} "
          f"({slope*N/(2*np.pi):+.3f} cycles)")
    print()
    for row, (case_name, chip) in enumerate(cases.items()):
        print(f"=== case: {case_name} ===")
        for col, (vname, fn) in enumerate(variants.items()):
            img = fn(chip, slope)
            prof = az_profile(img)
            ptm = peak_to_mean(prof)
            ax = axes[row, col]
            a = np.abs(img)
            vmin = float(np.percentile(a, 1.0))
            vmax = float(np.percentile(a, 99.0))
            if vmax <= vmin:
                vmax = vmin + 1.0
            ax.imshow(a, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xlabel("range"); ax.set_ylabel("azimuth")
            ax.set_title(f"{vname}\npeak/mean = {ptm:6.1f}", fontsize=9)
            print(f"  {vname:32s}  peak/mean = {ptm:8.2f}")
        print()
    fig.suptitle(
        f"Synthetic chirp refocus test — N={N}, W={W}, slope={slope:+.4e} rad/row",
        fontsize=12,
    )
    out = OUT_DIR / "synthetic_refocus_test.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
