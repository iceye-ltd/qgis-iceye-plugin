"""Diagnostic plot of the linear azimuth-phase fit for one or more boxes.

Four panels per box:

  (a) phi(row) [wrapped to (-pi, pi]] with the linear fit m*i+b overlaid.
      The fit is drawn as a wrapped sawtooth so it stays inside the same
      +/-pi band as the data. Sample color = coherence.
  (b) residual r_i = wrap(phi_i - fit) vs row, coloured by coherence.
      Horizontal bands at +/-0.5, +/-1.0, +/-1.5 rad give context.
  (c) histogram of |r_i| with dashed markers at 0.5 / 1.0 / 1.5 rad.
  (d) coherence weight histogram.

For each box the console line reports fraction of rows with |r| <= T for
T in {0.5, 1.0, 1.5} rad, both unweighted and coherence-weighted.

Usage
-----
    python scripts/diag_box_fit.py <scene>.npz --box 2 3 --out-dir <dir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


THRESHOLDS = (0.5, 1.0, 1.5)


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def diagnose(npz_path: Path, box_idx: int, out_dir: Path) -> None:
    d = np.load(npz_path, allow_pickle=False)
    phi_all = d["box_phi_rad"]
    coh_all = d["box_coh"]
    slope = float(d["box_slope_rad_per_row"][box_idx])
    intercept = float(d["box_intercept_rad"][box_idx])
    n_k = int(d["box_n_rows"][box_idx])
    y0 = int(d["box_y0"][box_idx])
    y_c, x_c, h, w = [int(v) for v in d["boxes_yxhw"][box_idx]]
    slope_total = float(d["slope_totals_rad"][box_idx])
    n_inliers = int(d["box_n_inliers"][box_idx])
    stored_residual = float(d["box_residual_rad"][box_idx])
    inlier_tol = float(d["inlier_tol_rad"])
    az_m_per_px = float(d.get("az_m_per_px_s_degraded", np.array(np.nan)))

    if n_k <= 0:
        print(f"[box {box_idx}] empty (n_rows=0)")
        return

    idx = np.arange(n_k, dtype=np.float64)
    phi = phi_all[box_idx, :n_k].astype(np.float64)
    coh = coh_all[box_idx, :n_k].astype(np.float64)

    valid = np.isfinite(phi) & np.isfinite(coh)
    phi_v = phi[valid]
    coh_v = np.clip(coh[valid], 0.0, 1.0)
    idx_v = idx[valid]

    fit_v = slope * idx_v + intercept
    r = wrap_to_pi(phi_v - fit_v)
    ar = np.abs(r)

    w_sum = float(coh_v.sum())
    fracs = {t: float(np.mean(ar <= t)) for t in THRESHOLDS}
    fracs_w = {
        t: (float(np.sum(coh_v * (ar <= t)) / w_sum) if w_sum > 0 else np.nan)
        for t in THRESHOLDS
    }
    rms = float(np.sqrt(np.mean(r * r)))
    mad = float(1.4826 * np.median(np.abs(r - np.median(r))))

    print(
        f"[box {box_idx}] y_c={y_c} x_c={x_c} h={h} w={w}  "
        f"n_rows={n_k} (valid={len(phi_v)})"
    )
    print(
        f"  slope={slope:+.6f} rad/row  intercept={intercept:+.3f} rad  "
        f"|slope*n|={abs(slope_total):.2f} rad"
    )
    print(
        f"  stored_residual={stored_residual:.3f} rad (over {n_inliers} inliers, tol={inlier_tol})"
    )
    print(
        "  frac(|r|<=T):   "
        + "  ".join(f"T={t} -> {fracs[t]:.3f}" for t in THRESHOLDS)
    )
    print(
        "  frac(|r|<=T)_w: "
        + "  ".join(f"T={t} -> {fracs_w[t]:.3f}" for t in THRESHOLDS)
    )
    print(f"  RMS={rms:.3f} rad   MAD={mad:.3f} rad")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except Exception as exc:
        print(f"[warn] matplotlib unavailable: {exc}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # (a) phi + wrapped fit
    ax = axes[0, 0]
    sc = ax.scatter(idx_v, phi_v, c=coh_v, cmap="viridis", s=6, vmin=0.0, vmax=1.0)
    fit_line_full = wrap_to_pi(slope * idx + intercept)
    # break the line at wrap discontinuities so it renders as a sawtooth
    jump = np.where(np.abs(np.diff(fit_line_full)) > np.pi)[0]
    breaks = np.concatenate([[0], jump + 1, [len(idx)]])
    for a, b in zip(breaks[:-1], breaks[1:]):
        ax.plot(idx[a:b], fit_line_full[a:b], color="red", lw=1.2, alpha=0.9)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("local row index i")
    ax.set_ylabel("phi_i  [rad]  (wrapped)")
    ax.set_title(
        f"phi(row) with linear fit  |  slope={slope:+.4f} rad/row  b={intercept:+.2f}"
    )
    fig.colorbar(sc, ax=ax, label="coherence")

    # (b) residuals
    ax = axes[0, 1]
    sc = ax.scatter(idx_v, r, c=coh_v, cmap="viridis", s=6, vmin=0.0, vmax=1.0)
    for tval in THRESHOLDS:
        ax.axhline(+tval, color="k", linestyle="--", lw=0.8, alpha=0.5)
        ax.axhline(-tval, color="k", linestyle="--", lw=0.8, alpha=0.5)
    ax.axhline(0, color="0.4", lw=0.6)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("local row index i")
    ax.set_ylabel("residual  wrap(phi - fit)  [rad]")
    ax.set_title(
        f"residuals  |  RMS={rms:.2f} rad  MAD={mad:.2f} rad"
    )
    fig.colorbar(sc, ax=ax, label="coherence")

    # (c) |r| histogram
    ax = axes[1, 0]
    ax.hist(ar, bins=60, range=(0, np.pi), color="tab:blue", alpha=0.75)
    for tval in THRESHOLDS:
        ax.axvline(tval, color="k", linestyle="--", lw=0.9)
        ax.text(
            tval, ax.get_ylim()[1] * 0.90, f" {tval:g} rad", rotation=90,
            va="top", ha="left", fontsize=8
        )
    ax.set_xlim(0, np.pi)
    ax.set_xlabel("|residual|  [rad]")
    ax.set_ylabel("count")
    ax.set_title(
        f"frac|r|<=1 rad = {fracs[1.0]:.3f}  (coh-weighted {fracs_w[1.0]:.3f})"
    )

    # (d) coherence histogram
    ax = axes[1, 1]
    ax.hist(coh_v, bins=50, range=(0, 1), color="tab:orange", alpha=0.75)
    ax.set_xlim(0, 1)
    ax.set_xlabel("coherence weight")
    ax.set_ylabel("count")
    ax.set_title(f"per-row coherence  (mean={coh_v.mean():.3f})")

    az_len = n_k * az_m_per_px if np.isfinite(az_m_per_px) else np.nan
    fig.suptitle(
        f"{npz_path.name}  |  box {box_idx}  "
        f"y_c={y_c} x_c={x_c} h={h} w={w}px "
        f"(~{az_len:.0f} m az)  "
        f"|slope*n|={abs(slope_total):.2f} rad"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{npz_path.stem}_box_{box_idx:03d}_fit_diag.png"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  wrote {out_png}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("npz", type=str, help="scene NPZ from shear_averaging.py")
    p.add_argument("--box", type=int, nargs="+", required=True, help="box indices")
    p.add_argument("--out-dir", type=str, default=None,
                   help="directory for diagnostic PNGs (defaults to npz folder)")
    args = p.parse_args()

    npz_path = Path(args.npz).expanduser().resolve()
    if not npz_path.is_file():
        sys.exit(f"error: {npz_path} does not exist")
    out_dir = (Path(args.out_dir).expanduser().resolve()
               if args.out_dir else npz_path.parent)
    for k in args.box:
        diagnose(npz_path, int(k), out_dir)


if __name__ == "__main__":
    main()
