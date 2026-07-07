"""Diagnose what `_refocus_box_chip` actually does to a known LFM chirp.

The function (scripts/shear_averaging.py:1793-1843) is supposed to be the
QPE refocus used in the pipeline. It does

    m   = arange(N) - N//2
    phi = 0.5 * slope * m**2
    F   = fft(ifftshift(chip))
    F  *= exp(+1j * phi)
    out = fftshift(ifft(F))

This script puts a *known* azimuth chirp into the function and compares to:
  - the spatial direct correction:  chip * exp(-j * 0.5 * slope * n**2)
  - the proper Fourier-dual:        F * exp(+j * f**2 / (2*slope)) in centred freq
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shear_averaging import _refocus_box_chip


def _norm_var(x):
    I = np.abs(x) ** 2
    mu = I.mean()
    return float(I.std() / mu) if mu > 0 else float("nan")


def _peak_mean(x):
    p = (np.abs(x) ** 2).sum(axis=1) if x.ndim == 2 else np.abs(x) ** 2
    return float(p.max() / p.mean()) if p.mean() > 0 else float("nan")


def spatial_direct(chip, slope, intercept=0.0, centred=True):
    """Exact undo by integrating dphi = slope·i + intercept."""
    N = chip.shape[0]
    i = (np.arange(N) - (N // 2)) if centred else np.arange(N, dtype=np.float64)
    phi = 0.5 * slope * i * i + intercept * i
    return chip * np.exp(-1j * phi)[:, None], phi


def fourier_dual(chip, slope, intercept=0.0):
    """Fourier-dual QPE: spectrum multiplied by exp(+j·f²/(2·slope))."""
    N = chip.shape[0]
    n = np.arange(N, dtype=np.float64) - (N // 2)
    # spatial pre-mult removes the Doppler-centroid intercept
    pre = chip * np.exp(-1j * intercept * n)[:, None]
    f = np.fft.fftshift(np.fft.fftfreq(N) * 2.0 * np.pi)   # rad/sample, centred
    phi_f = (f * f) / (2.0 * slope)
    phi_unshifted = np.fft.ifftshift(phi_f)
    F = np.fft.fft(pre, axis=0)
    F = F * np.exp(1j * phi_unshifted)[:, None]
    return np.fft.ifft(F, axis=0), phi_f


def make_chirp(N, slope, intercept=0.0, centred_origin=False, n_cols=7,
               weight="rect", noise_db=None, seed=0):
    """A single point target spread by chirp slope·i² + intercept·i."""
    rng = np.random.default_rng(seed)
    if centred_origin:
        i = np.arange(N, dtype=np.float64) - (N // 2)
    else:
        i = np.arange(N, dtype=np.float64)
    phi = 0.5 * slope * i * i + intercept * i
    s = np.exp(+1j * phi)
    if weight == "taylor":
        w = np.hanning(N)
    else:
        w = np.ones(N)
    s = s * w
    chip = np.tile(s[:, None], (1, n_cols))
    if noise_db is not None:
        sigma = 10 ** (-noise_db / 20.0)
        chip = chip + sigma * (rng.standard_normal(chip.shape)
                               + 1j * rng.standard_normal(chip.shape))
    return chip


def run_case(label, chip, slope, intercept=0.0):
    print(f"\n--- {label} ---")
    print(f"  N={chip.shape[0]}  slope={slope:+.3e} rad/row   "
          f"intercept={intercept:+.3e} rad/sample")
    print(f"  raw     C={_norm_var(chip):.3f}   peak/mean(az)={_peak_mean(chip):.2f}")

    out_script, _ = _refocus_box_chip(chip, slope)
    print(f"  script  C={_norm_var(out_script):.3f}   "
          f"peak/mean(az)={_peak_mean(out_script):.2f}   <-- _refocus_box_chip")

    out_spatial, _ = spatial_direct(chip, slope, intercept)
    print(f"  spatial C={_norm_var(out_spatial):.3f}   "
          f"peak/mean(az)={_peak_mean(out_spatial):.2f}   <-- integrate-and-multiply")

    out_dual, _ = fourier_dual(chip, slope, intercept)
    print(f"  dual    C={_norm_var(out_dual):.3f}   "
          f"peak/mean(az)={_peak_mean(out_dual):.2f}   <-- correct Fourier-dual")
    return dict(raw=chip, script=out_script, spatial=out_spatial, dual=out_dual)


def plot(label, results, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.4, 1.0]})
    titles = ["raw  (chirp signal)", "script  _refocus_box_chip",
              "spatial  integrate-and-multiply", "dual  Fourier-dual"]
    keys = ["raw", "script", "spatial", "dual"]
    for col, (k, ttl) in enumerate(zip(keys, titles)):
        img = np.abs(results[k])
        vmin = float(np.nanpercentile(img, 1))
        vmax = float(np.nanpercentile(img, 99.5))
        axes[0, col].imshow(img, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        axes[0, col].set_title(ttl, fontsize=10)
        axes[0, col].set_xlabel("range col"); axes[0, col].set_ylabel("azimuth row")
        prof = (img ** 2).sum(axis=1)
        axes[1, col].plot(prof, lw=1.0)
        axes[1, col].set_xlim(0, prof.size - 1)
        axes[1, col].set_title(f"|·|² summed across range   pk/mean={prof.max()/prof.mean():.2f}",
                               fontsize=10)
        axes[1, col].set_xlabel("azimuth row"); axes[1, col].set_ylabel("intensity")
    fig.suptitle(label, fontsize=12)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"  → {out_path}")


if __name__ == "__main__":
    OUT = HERE / "diag_refocus_out"
    OUT.mkdir(exist_ok=True)

    # Case A — pure chirp, T1-like parameters, intercept = 0 (centred).
    N, slope = 1720, 2.4e-3
    chip = make_chirp(N, slope, intercept=0.0, centred_origin=True, n_cols=7)
    r = run_case("A) pure chirp, slope=2.4e-3, intercept=0, centred",
                 chip, slope, intercept=0.0)
    plot("A) pure chirp, slope=2.4e-3, centred, intercept=0", r, OUT / "A_pure_chirp.png")

    # Case B — same chirp but with the actual T1 intercept ≈ -1.748 rad/sample
    # and uncentred origin (matches the fit's i=0..N-1 convention).
    intercept_T1 = -1.7484
    chip = make_chirp(N, slope, intercept=intercept_T1,
                      centred_origin=False, n_cols=7)
    r = run_case("B) chirp w/ intercept=-1.75 rad/sample (T1-like), uncentred origin",
                 chip, slope, intercept_T1)
    plot("B) chirp w/ T1 intercept", r, OUT / "B_T1_intercept.png")

    # Case C — same chirp but with hanning weighting + noise (simulating clutter).
    chip = make_chirp(N, slope, intercept=intercept_T1,
                      centred_origin=False, n_cols=7,
                      weight="taylor", noise_db=10.0, seed=42)
    r = run_case("C) windowed chirp + noise (SNR=10 dB)", chip, slope, intercept_T1)
    plot("C) windowed chirp + noise (SNR=10 dB)", r, OUT / "C_with_noise.png")
