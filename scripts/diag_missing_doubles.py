"""Diagnose why the three "double-box" targets are missing from
WTW3YQ.

Three ROIs the user flagged in ``WTW3YQ_box_stats.csv`` — each holds
two adjacent (tandem) movers, and each collapses to at most one kept
box. This script renders per-box amplitude / azimuth-profile / phase-
ramp figures for the CSV-known candidate boxes in every ROI so we can
see whether each long box actually spans two ships (i.e. the phase-fit
gate is being asked to fit a single line to two Doppler-different
targets).

Runs entirely from the CSV coordinates + the source SLC GeoTIFF; no
seed/grow re-run needed. Output goes to
``scripts/missing_doubles_out/`` so it never overwrites the pipeline's
``WTW3YQ_per_box/`` artefacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from shear_averaging import (  # noqa: E402
    apply_range_window,
    degrade_range_resolution_range_sum,
    compute_box_phase_estimates,
    _refocus_box_chip,
    _normalized_variance,
)

TIF = Path(
    "/home/odogan/Desktop/ship_focusing/4439676/"
    "ICEYE_WTW3YQ_20250104T180444Z_4439676_X7_SLED_SLC.tif"
)
OUT_DIR = HERE / "missing_doubles_out"

RANGE_LOOKS = 4
RANGE_SLL_DB = 55.0
RANGE_TAYLOR_NBAR = 8
MIN_ROW_COHERENCE = 0.5
INLIER_TOL_RAD = 0.5
PHASE_FIT_THRESH_RAD = 1.5

RANGE_SPACING = 0.24
AZIMUTH_SPACING = 0.04

# Candidate boxes per ROI (init_idx, disposition, y_c, x_c, h, w). Coordinates copied verbatim from WTW3YQ_box_stats.csv.
ROIS = {
    "T1_y72500-77500_x860-890": [
        (789, "dropped_by_nested_box_overlap", 74888, 877, 6384, 15),
        (790, "kept",                          76728, 868, 3088, 7),
        (851, "dropped_by_nested_box_overlap", 74344, 880, 5520, 12),
        (852, "dropped_by_phase_fit_quality",  74592, 879, 5472, 10),
    ],
    "T2_y67400-72500_x930-950": [
        (989, "dropped_by_phase_fit_quality",  69672, 944, 5712, 10),
        (990, "dropped_by_nested_box_overlap", 69848, 943, 5584, 11),
    ],
    "T3_y62500-67500_x975-1000": [
        (1188, "dropped_by_nested_box_overlap", 65424, 987, 5984, 17),
        (1189, "dropped_by_phase_fit_quality",  65552, 986, 6432, 15),
        (1190, "dropped_by_nested_box_overlap", 65536, 986, 6496, 15),
    ],
    "T4_y100000-105000_x640-665": [
        (196, "dropped_by_phase_fit_quality",  101336, 661, 2992, 10),
        (197, "dropped_by_phase_fit_quality",  102896, 653, 3328, 11),
        (201, "dropped_by_nested_box_overlap", 102744, 649, 3856, 20),
        (202, "kept",                          103912, 641, 2288,  5),
    ],
}


def load_azimuth_strip(tif_path: Path, y_lo: int, y_hi: int, left: bool = True) -> np.ndarray:
    """Load a complex64 azimuth strip [y_lo, y_hi) at full range."""
    from osgeo import gdal

    ds = gdal.Open(str(tif_path))
    if ds is None:
        raise FileNotFoundError(tif_path)
    W_file = ds.RasterXSize
    H_file = ds.RasterYSize
    n_rows = int(y_hi - y_lo)

    # After fliplr(cols)+transpose the output has shape (W_file, H_file). Output azimuth index y corresponds to file column (W_file-1-y) when left-look, or file column y otherwise. Reading a strip in output-azimuth coords therefore maps to a contiguous file-column window that we still have to fliplr locally so the strip's own column order is preserved.
    if left:
        col_lo = W_file - int(y_hi)
        col_hi = W_file - int(y_lo)
    else:
        col_lo = int(y_lo)
        col_hi = int(y_hi)
    col_lo = max(0, col_lo)
    col_hi = min(W_file, col_hi)
    xsize = col_hi - col_lo
    if xsize != n_rows:
        raise RuntimeError(
            f"y range {[y_lo, y_hi]} maps to {xsize} file cols, expected {n_rows}"
        )

    amp_band = ds.GetRasterBand(1)
    pha_band = ds.GetRasterBand(2)
    amp_scale = amp_band.GetScale() or 1.0
    amp_offset = amp_band.GetOffset() or 0.0
    pha_scale = pha_band.GetScale() or 1.0
    pha_offset = pha_band.GetOffset() or 0.0

    amp = amp_band.ReadAsArray(col_lo, 0, xsize, H_file)
    pha = pha_band.ReadAsArray(col_lo, 0, xsize, H_file)
    ds = None

    data = amp.astype(np.complex64) * amp_scale + amp_offset
    del amp
    data *= np.exp(-1j * (pha.astype(np.float32) * pha_scale + pha_offset))
    del pha

    if left:
        data = np.fliplr(data)
    return np.ascontiguousarray(data.T)


def refocus_freq_domain(chip: np.ndarray, slope_rad_per_row: float, intercept_rad: float):
    """Symmetric two-step centred-QPE refocus (spatial linear + spectral quadratic). Mirrors ``render_missing_box_figs._refocus_box_chip_fixed`` so we compare like for like."""
    N = chip.shape[0]
    if N < 2 or abs(slope_rad_per_row) < 1e-12:
        return chip.copy()
    n = np.arange(N, dtype=np.float64) - (N // 2)
    chip_shifted = chip * np.exp(-1j * float(intercept_rad) * n)[:, None]
    f = np.fft.fftshift(np.fft.fftfreq(N) * 2.0 * np.pi)
    phi_freq = (f * f) / (2.0 * float(slope_rad_per_row)) + f * n
    phi_unshifted = np.fft.ifftshift(phi_freq)
    F = np.fft.fft(chip_shifted, axis=0)
    F = F * np.exp(1j * phi_unshifted)[:, None]
    return np.fft.ifft(F, axis=0)


def plot_box_diagnostic(
    *,
    label: str,
    init_idx: int,
    disp: str,
    box: tuple[int, int, int, int],
    chip: np.ndarray,
    phi: np.ndarray,
    coh: np.ndarray,
    slope_pr: float,
    intercept: float,
    residual: float,
    frac_within: float,
    n_inliers: int,
    y0: int,
    n_rows: int,
    out_path: Path,
) -> None:
    y_c, x_c, h, w = box
    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 2.4, 2.4]},
    )
    fig.suptitle(
        f"[{label}] init_idx={init_idx}  disp={disp}\n"
        f"y_c={y_c} x_c={x_c} h={h} w={w}  |slope·n|="
        f"{abs(slope_pr * n_rows):.2f} rad  residual="
        f"{residual:.2f} rad  frac|r|≤{PHASE_FIT_THRESH_RAD}rad="
        f"{frac_within:.3f}",
        fontsize=10,
    )

    amp = np.abs(chip)
    if amp.size:
        vmin = float(np.nanpercentile(amp, 1.0))
        vmax = float(np.nanpercentile(amp, 99.5))
        if vmax <= vmin:
            vmax = vmin + 1.0
    else:
        vmin, vmax = 0.0, 1.0

    ax = axes[0, 0]
    ax.imshow(amp, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax,
              origin="upper",
              extent=[-w / 2, w / 2, y0 + n_rows, y0])
    ax.set_title("|chip|  (BEFORE refocus)", fontsize=9)
    ax.set_xlabel("range offset [px]")
    ax.set_ylabel("absolute az row")

    ax = axes[0, 1]
    prof = (amp ** 2).sum(axis=1)
    y_abs = y0 + np.arange(prof.size)
    ax.plot(y_abs, prof, lw=0.8)
    mu = float(prof.mean()) if prof.size else 0.0
    ax.axhline(mu, color="0.6", ls="--", lw=0.6, label=f"mean = {mu:.0f}")
    ax.set_xlabel("absolute az row")
    ax.set_ylabel(r"$\sum_{rg}|I|^2$")
    ax.set_title(
        f"azimuth intensity profile  —  peak/mean = "
        f"{(prof.max() / mu if mu > 0 else np.nan):.1f}",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(y_abs[0], y_abs[-1])

    ax = axes[0, 2]
    phi_n = phi[:n_rows]
    coh_n = coh[:n_rows]
    valid = np.isfinite(phi_n)
    i_loc = np.arange(n_rows, dtype=np.float64)
    y_abs_ph = y0 + i_loc
    if valid.any():
        sc = ax.scatter(
            y_abs_ph[valid], phi_n[valid],
            c=coh_n[valid], cmap="viridis",
            s=10, vmin=0.0, vmax=1.0,
            edgecolor="none", zorder=2,
        )
        fig.colorbar(sc, ax=ax, location="right", shrink=0.85, label="coh")
        if np.isfinite(slope_pr) and np.isfinite(intercept):
            line_w = np.angle(np.exp(1j * (slope_pr * i_loc + intercept)))
            if line_w.size > 1:
                jumps = np.where(np.abs(np.diff(line_w)) > np.pi)[0]
                if jumps.size:
                    line_w = line_w.copy()
                    line_w[jumps] = np.nan
            ax.plot(y_abs_ph, line_w, color="red", lw=1.1, zorder=3)
    ax.set_ylim(-np.pi, np.pi)
    ax.axhline(0.0, color="0.5", lw=0.5, ls="--", zorder=1)
    used = int((np.isfinite(phi_n) & np.isfinite(coh_n) & (coh_n > MIN_ROW_COHERENCE)).sum())
    coh_pct = (100.0 * used / n_rows) if n_rows else 0.0
    in_pct = (100.0 * n_inliers / n_rows) if n_rows else 0.0
    ax.set_title(
        f"d_phase  slope={np.degrees(slope_pr):+.3f}°/row  "
        f"coh>{MIN_ROW_COHERENCE}: {used}/{n_rows} ({coh_pct:.1f}%)  "
        f"inliers(|r|≤{INLIER_TOL_RAD}): {n_inliers} ({in_pct:.1f}%)",
        fontsize=9,
    )
    ax.set_xlabel("absolute az row")
    ax.set_ylabel(r"$\Delta\varphi$  [rad]")
    ax.grid(True, alpha=0.3)

    corrected = refocus_freq_domain(chip, slope_pr, intercept)
    amp_c = np.abs(corrected)
    vmin_c = float(np.nanpercentile(amp_c, 1.0)) if amp_c.size else 0.0
    vmax_c = float(np.nanpercentile(amp_c, 99.5)) if amp_c.size else 1.0
    if vmax_c <= vmin_c:
        vmax_c = vmin_c + 1.0

    ax = axes[1, 0]
    ax.imshow(amp_c, aspect="auto", cmap="gray", vmin=vmin_c, vmax=vmax_c,
              origin="upper",
              extent=[-w / 2, w / 2, y0 + n_rows, y0])
    ax.set_title("|chip|  (AFTER refocus)", fontsize=9)
    ax.set_xlabel("range offset [px]")
    ax.set_ylabel("absolute az row")

    ax = axes[1, 1]
    prof_a = (amp_c ** 2).sum(axis=1)
    ax.plot(y_abs, prof_a, lw=0.8)
    mu_a = float(prof_a.mean()) if prof_a.size else 0.0
    ax.axhline(mu_a, color="0.6", ls="--", lw=0.6, label=f"mean = {mu_a:.0f}")
    ax.set_xlabel("absolute az row")
    ax.set_ylabel(r"$\sum_{rg}|I_{ref}|^2$")
    c_b = _normalized_variance(chip)
    c_a = _normalized_variance(corrected)
    gain_db = (
        20.0 * np.log10(c_a / c_b) if (c_b > 0 and c_a > 0) else float("nan")
    )
    ax.set_title(
        f"AFTER refocus  peak/mean = "
        f"{(prof_a.max() / mu_a if mu_a > 0 else np.nan):.1f}"
        f"  C: {c_b:.3f}→{c_a:.3f}  ({gain_db:+.2f} dB)",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(y_abs[0], y_abs[-1])

    ax = axes[1, 2]
    ax.axis("off")
    txt = (
        f"init_idx = {init_idx}\n"
        f"disposition = {disp}\n"
        f"\n"
        f"box (y_c, x_c, h, w) = ({y_c}, {x_c}, {h}, {w})\n"
        f"az extent: rows {y0}..{y0 + n_rows} ({h * AZIMUTH_SPACING:.1f} m)\n"
        f"rg extent: {w} px ({w * RANGE_SPACING * RANGE_LOOKS:.1f} m)\n"
        f"\n"
        f"slope       = {slope_pr:+.4e} rad/row\n"
        f"slope·n     = {slope_pr * n_rows:+.2f} rad\n"
        f"intercept   = {intercept:+.3f} rad\n"
        f"residual    = {residual:.3f} rad\n"
        f"frac|r|≤{PHASE_FIT_THRESH_RAD}rad = {frac_within:.3f}\n"
        f"\n"
        f"gain (refocus)   = {gain_db:+.2f} dB\n"
        f"C before / after = {c_b:.3f} / {c_a:.3f}\n"
    )
    ax.text(0.02, 0.98, txt, family="monospace", fontsize=9,
            va="top", ha="left")

    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def compute_frac_within(phi: np.ndarray, coh: np.ndarray,
                        slope_pr: float, intercept: float,
                        n_rows: int, thresh: float = PHASE_FIT_THRESH_RAD) -> float:
    """Coherence-weighted fraction of az rows with |wrapped(phi − fit)| ≤ thresh. Matches ``compute_box_frac_within_thresh`` (all-valid-rows, coh clipped to [0,1] — NO ``min_row_coherence`` gate)."""
    if not (np.isfinite(slope_pr) and np.isfinite(intercept)):
        return float("nan")
    phi_row = np.asarray(phi[:n_rows], dtype=np.float64)
    coh_row = np.asarray(coh[:n_rows], dtype=np.float64)
    valid = np.isfinite(phi_row) & np.isfinite(coh_row)
    if not valid.any():
        return float("nan")
    idx = np.arange(n_rows, dtype=np.float64)[valid]
    phi_v = phi_row[valid]
    coh_v = np.clip(coh_row[valid], 0.0, 1.0)
    resid = np.angle(np.exp(1j * (phi_v - (slope_pr * idx + intercept))))
    close = (np.abs(resid) <= thresh).astype(np.float64)
    w_sum = float(coh_v.sum())
    if w_sum <= 0:
        return float("nan")
    return float((coh_v * close).sum() / w_sum)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    y_lo = min(b[2] - b[4] // 2 for boxes in ROIS.values() for b in boxes) - 200
    y_hi = max(b[2] + b[4] // 2 for boxes in ROIS.values() for b in boxes) + 200
    print(f"Loading azimuth strip [{y_lo}, {y_hi}) from {TIF.name} ...")
    strip = load_azimuth_strip(TIF, y_lo, y_hi, left=True)
    print(f"  strip shape (SLC, full range) = {strip.shape}, dtype={strip.dtype}")

    print("Range Taylor window + range degradation ...")
    strip_win = apply_range_window(strip, sll_db=RANGE_SLL_DB, nbar=RANGE_TAYLOR_NBAR)
    del strip
    s_deg = degrade_range_resolution_range_sum(strip_win, RANGE_LOOKS)
    del strip_win
    print(f"  s_degraded strip shape = {s_deg.shape}")

    summary_rows = []
    for label, boxes in ROIS.items():
        print(f"\n=== {label} ===")
        for init_idx, disp, y_c, x_c, h, w in boxes:
            y_c_loc = y_c - y_lo  # local azimuth center inside strip
            y0_full = int(y_c) - int(h) // 2
            y1_full = y0_full + int(h)
            y0_full = max(y_lo, y0_full)
            y1_full = min(y_hi, y1_full)
            box_full = np.array([[y_c, x_c, h, w]], dtype=np.int64)
            box_local = np.array(
                [[y_c_loc, x_c, y1_full - y0_full, w]], dtype=np.int64
            )
            phi_arr, coh_arr, sl_arr, ic_arr, y0_arr, n_arr, res_arr, in_arr = (
                compute_box_phase_estimates(
                    box_local,
                    s_deg,
                    min_row_coherence=MIN_ROW_COHERENCE,
                    inlier_tol_rad=INLIER_TOL_RAD,
                )
            )
            n_k = int(min(n_arr[0], phi_arr.shape[1]))
            y0_l = int(y0_arr[0])
            slope_pr = float(sl_arr[0])
            intercept = float(ic_arr[0])
            residual = float(res_arr[0])
            n_inliers = int(in_arr[0])
            frac_w = compute_frac_within(
                phi_arr[0], coh_arr[0], slope_pr, intercept, n_k
            )
            x0 = max(0, int(x_c) - int(w) // 2)
            x1 = min(s_deg.shape[1], x0 + int(w))
            chip = s_deg[y0_l:y0_l + n_k, x0:x1]
            y0_abs = y0_l + y_lo
            out_path = OUT_DIR / f"box_{init_idx:04d}_{label}.png"
            plot_box_diagnostic(
                label=label, init_idx=init_idx, disp=disp,
                box=(y_c, x_c, h, w),
                chip=chip, phi=phi_arr[0], coh=coh_arr[0],
                slope_pr=slope_pr, intercept=intercept,
                residual=residual, frac_within=frac_w, n_inliers=n_inliers,
                y0=y0_abs, n_rows=n_k, out_path=out_path,
            )
            summary_rows.append(
                (label, init_idx, disp, y_c, x_c, h, w,
                 slope_pr, slope_pr * n_k, residual, frac_w, n_inliers, n_k)
            )
            print(
                f"  box {init_idx:4d} disp={disp:32s} "
                f"slope·n={slope_pr * n_k:+.2f} rad  "
                f"res={residual:.2f}  frac={frac_w:.3f}  n={n_k}"
            )

    csv_path = OUT_DIR / "summary.csv"
    with csv_path.open("w") as f:
        f.write(
            "roi,init_idx,disposition,y_c,x_c,h,w,"
            "slope_rad_per_row,slope_total_rad,residual_rad,"
            f"frac_within_{PHASE_FIT_THRESH_RAD}rad_coh,n_inliers,n_rows\n"
        )
        for r in summary_rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"\nSummary CSV → {csv_path}")


if __name__ == "__main__":
    main()
