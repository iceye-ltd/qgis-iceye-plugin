"""Display `sub_mean` and `mask_dec * cov_sq` with cluster_targets overlays.

Input: a ``*_subapertures.npz`` produced by
``scripts/save_subapertures.py`` — carries
``sub_mean``, ``sub_var``, ``N_subaperture``, ``Number_of_Range_Looks``,
``az_m_per_px_s_degraded`` and ``rg_m_per_px_s_degraded``.

We reproduce the exact call made in the main pipeline
(``shear_averaging.py`` lines 4823–4833)::

    cluster_result = cluster_targets(
        sub_mean, sub_var,
        az_m_per_px=azimuth_spacing * N_subaperture,
        rg_m_per_px=range_spacing * Number_of_Range_Looks,
        n_subaperture=N_subaperture,
        n_range_looks=Number_of_Range_Looks,
        cov_th_mult=cfg.cov_th_mult,
        bright_mode=cfg.bright_mode,
        bright_cov_th_mult=cfg.bright_cov_th_mult,
        dark_amp_percentile=cfg.dark_amp_percentile,
    )

and render two diagnostic figures:

* ``<stem>_sub_mean_boxes.png`` — same layout as
  ``shear_averaging.py:4866–4901``: sub_mean in viridis with 4-sigma
  clipping, cyan cluster_targets boxes (halo'd) and red-× seed peaks.
* ``<stem>_cov_masked.png`` — ``mask_dec * cov_sq`` (CoV² gated to the
  pre-isolated-pixel-filter mask) with the same cyan boxes + red-×
  seed peaks overlaid, so the CoV gate can be inspected pixel-wise.
  ``cov_sq`` and ``mask_dec`` are recomputed locally with the exact
  CoV-gate code from ``shear_averaging.py:4323–4348`` — supports
  ``bright_mode`` ∈ {off, exclude, higher-th}.

No SLC is loaded and no shear/averaging is run; this script only exercises
the sub-aperture-domain detection stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# `shear_averaging` parses `sys.argv` at import time (its top-level `_split_cli(sys.argv)` call) and raises on any flag it doesn't own — so hide our CLI from it while importing.
_saved_argv = sys.argv
sys.argv = [_saved_argv[0]]
try:
    from shear_averaging import cluster_targets, _overlay_boxes  # noqa: E402
finally:
    sys.argv = _saved_argv


def _load_subapertures(path: Path) -> dict:
    """Return the fields written by ``save_subapertures.py``."""
    with np.load(path, allow_pickle=False) as arch:
        missing = [
            k for k in (
                "sub_mean", "sub_var",
                "N_subaperture", "Number_of_Range_Looks",
                "az_m_per_px_s_degraded", "rg_m_per_px_s_degraded",
            ) if k not in arch.files
        ]
        if missing:
            raise KeyError(
                f"{path} is missing expected keys: {missing}. "
                "Was it produced by scripts/save_subapertures.py?"
            )
        return dict(
            sub_mean=np.ascontiguousarray(arch["sub_mean"]),
            sub_var=np.ascontiguousarray(arch["sub_var"]),
            N_subaperture=int(arch["N_subaperture"]),
            Number_of_Range_Looks=int(arch["Number_of_Range_Looks"]),
            az_m_per_px_s_degraded=float(arch["az_m_per_px_s_degraded"]),
            rg_m_per_px_s_degraded=float(arch["rg_m_per_px_s_degraded"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Display sub_mean overlaid with cluster_targets boxes and "
            "seed peaks, reproducing shear_averaging.py:4823-4833."
        )
    )
    parser.add_argument(
        "npz", type=Path,
        help="Path to *_subapertures.npz written by save_subapertures.py.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="PNG destination for the sub_mean figure. Default: "
             "<input stem>_sub_mean_boxes.png in the current working "
             "directory. The CoV-masked figure is written next to it as "
             "<input stem>_cov_masked.png (override with --output-cov).",
    )
    parser.add_argument(
        "--output-cov", type=Path, default=None,
        help="PNG destination for the mask_dec * cov_sq figure. "
             "Default: <sub_mean output>_cov_masked.png sibling.",
    )
    parser.add_argument(
        "--cov-eps", type=float, default=1e-12,
        help="Epsilon added to sub_mean^2 in the CoV ratio "
             "(matches shear_averaging default: 1e-12).",
    )
    parser.add_argument(
        "--cov-th-mult", type=float, default=1.5,
        help="Multiplier on median(CoV^2) that sets the CoV gate "
             "(shear_averaging default: 1.5).",
    )
    parser.add_argument(
        "--bright-mode", choices=("off", "exclude", "higher-th"),
        default="off",
        help="How the CoV gate treats bright pixels (shear_averaging "
             "default: 'off').",
    )
    parser.add_argument(
        "--bright-cov-th-mult", type=float, default=3.0,
        help="Multiplier applied to median(CoV^2) among bright pixels "
             "when --bright-mode=higher-th (default: 3.0).",
    )
    parser.add_argument(
        "--dark-amp-percentile", type=float, default=50.0,
        help="Amplitude percentile that splits dark/bright for the CoV "
             "gate (default: 50.0).",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Skip plt.show(); only write the PNG.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Output PNG dpi (default: 150).",
    )
    args = parser.parse_args()

    if not args.npz.exists():
        raise FileNotFoundError(args.npz)

    bundle = _load_subapertures(args.npz)
    sub_mean = bundle["sub_mean"]
    sub_var = bundle["sub_var"]
    N_subaperture = bundle["N_subaperture"]
    Number_of_Range_Looks = bundle["Number_of_Range_Looks"]
    azimuth_spacing = bundle["az_m_per_px_s_degraded"]
    range_spacing_degraded = bundle["rg_m_per_px_s_degraded"]

    print(f"Loaded  : {args.npz}")
    print(
        f"sub_mean: shape={sub_mean.shape}, dtype={sub_mean.dtype}; "
        f"N_sub={N_subaperture}, N_RL={Number_of_Range_Looks}; "
        f"az={azimuth_spacing:g} m/row × rg_deg={range_spacing_degraded:g} m/px"
    )

    cluster_result = cluster_targets(
        sub_mean, sub_var,
        az_m_per_px=azimuth_spacing * N_subaperture,
        rg_m_per_px=range_spacing_degraded,
        n_subaperture=N_subaperture,
        n_range_looks=Number_of_Range_Looks,
        cov_th_mult=args.cov_th_mult,
        bright_mode=args.bright_mode,
        bright_cov_th_mult=args.bright_cov_th_mult,
        dark_amp_percentile=args.dark_amp_percentile,
    )

    boxes = cluster_result.boxes
    peaks_yx = cluster_result.peaks_yx
    sub_mean = cluster_result.sub_mean
    sub_var = cluster_result.sub_var

    if len(boxes):
        yl = boxes[:, 0].astype(np.int64)
        yh = boxes[:, 1].astype(np.int64)
        xl = boxes[:, 2].astype(np.int64)
        xh = boxes[:, 3].astype(np.int64)
        h = yh - yl + 1
        w = xh - xl + 1
        boxes_dec_yxhw = np.stack(
            [yl + h // 2, xl + w // 2, h, w], axis=1,
        )
    else:
        boxes_dec_yxhw = np.empty((0, 4), dtype=np.int64)

    # Recompute cov_sq / mask_dec locally with the same logic as `cluster_targets` (shear_averaging.py:4323-4348) — `cluster_result` only exposes `mask_filt` (post isolated-pixel filter), not the raw pre-filter mask.
    cov_sq = sub_var / (sub_mean ** 2 + args.cov_eps)
    if args.bright_mode == "off":
        cov_th = args.cov_th_mult * float(np.median(cov_sq))
        mask_dec = cov_sq > cov_th
        gate_desc = f"cov_sq > {cov_th:.3g} (mult={args.cov_th_mult:g}·median)"
    else:
        amp_dark_thresh = float(np.percentile(sub_mean, args.dark_amp_percentile))
        dark_dec = sub_mean <= amp_dark_thresh
        cov_sq_dark = cov_sq[dark_dec] if dark_dec.any() else cov_sq
        th_dark = args.cov_th_mult * float(np.median(cov_sq_dark))
        if args.bright_mode == "exclude":
            mask_dec = dark_dec & (cov_sq > th_dark)
            gate_desc = (
                f"dark & cov_sq > {th_dark:.3g} "
                f"(mult={args.cov_th_mult:g}·median_dark, "
                f"amp≤p{args.dark_amp_percentile:g})"
            )
        else:  # "higher-th"
            bright_dec = ~dark_dec
            median_bright = (
                float(np.median(cov_sq[bright_dec])) if bright_dec.any()
                else float(np.median(cov_sq_dark))
            )
            th_bright = args.bright_cov_th_mult * median_bright
            mask_dec = (
                (dark_dec & (cov_sq > th_dark))
                | (bright_dec & (cov_sq > th_bright))
            )
            gate_desc = (
                f"dark & cov_sq > {th_dark:.3g}  |  bright & cov_sq > "
                f"{th_bright:.3g} (dark mult={args.cov_th_mult:g}, "
                f"bright mult={args.bright_cov_th_mult:g}, "
                f"amp≤p{args.dark_amp_percentile:g})"
            )
    cov_masked = np.where(mask_dec, cov_sq, 0.0)
    n_kept = int(mask_dec.sum())
    n_total = int(mask_dec.size)
    print(
        f"CoV gate: kept {n_kept}/{n_total} = "
        f"{100.0 * n_kept / n_total:.2f}%  ({gate_desc})"
    )

    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        f"sub_mean + cluster_targets boxes — {args.npz.name}", fontsize=11,
    )
    mu = float(sub_mean.mean())
    sd = float(sub_mean.std())
    vmin = max(0.0, mu - 4.0 * sd)
    vmax = min(float(sub_mean.max()), mu + 4.0 * sd)
    im = ax.imshow(sub_mean, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    _overlay_boxes(ax, boxes_dec_yxhw, color="cyan", lw=1.0)
    if len(peaks_yx):
        ax.scatter(
            peaks_yx[:, 1], peaks_yx[:, 0],
            s=10, marker="x", c="red", linewidths=0.6,
            label=f"seed peaks ({len(peaks_yx)})",
        )
        ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"sub_mean (decimated grid: {sub_mean.shape[0]}×{sub_mean.shape[1]}), "
        f"{len(boxes_dec_yxhw)} cluster_targets boxes (cyan), "
        f"{len(peaks_yx)} seed peaks (red ×)"
    )
    ax.set_xlabel("range pixel (decimated)")
    ax.set_ylabel(
        f"azimuth pixel (decimated, ×{N_subaperture} = s_degraded row)"
    )
    plt.colorbar(im, ax=ax, label="mean |s| over N_sub")

    out_path = args.output or Path.cwd() / f"{args.npz.stem}_sub_mean_boxes.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi)
    print(f"Saved  → {out_path}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 9), constrained_layout=True)
    fig2.suptitle(
        f"mask_dec × cov_sq  (CoV² gated) — {args.npz.name}", fontsize=11,
    )
    cov_pos = cov_masked[mask_dec]
    if cov_pos.size:
        vmin2 = float(cov_pos.min())
        vmax2 = float(np.percentile(cov_pos, 99.0))
    else:
        vmin2, vmax2 = 0.0, 1.0
    im2 = ax2.imshow(
        cov_masked, cmap="magma", aspect="auto", vmin=vmin2, vmax=vmax2,
    )
    _overlay_boxes(ax2, boxes_dec_yxhw, color="cyan", lw=1.0)
    if len(peaks_yx):
        ax2.scatter(
            peaks_yx[:, 1], peaks_yx[:, 0],
            s=10, marker="x", c="red", linewidths=0.6,
            label=f"seed peaks ({len(peaks_yx)})",
        )
        ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title(
        f"mask_dec × cov_sq (decimated grid: {cov_masked.shape[0]}×"
        f"{cov_masked.shape[1]}); kept {n_kept}/{n_total} = "
        f"{100.0 * n_kept / n_total:.2f}%; "
        f"vmax=p99 of masked cov_sq={vmax2:.3g}"
    )
    ax2.set_xlabel("range pixel (decimated)")
    ax2.set_ylabel(
        f"azimuth pixel (decimated, ×{N_subaperture} = s_degraded row)"
    )
    plt.colorbar(im2, ax=ax2, label="mask_dec × cov_sq")

    out_cov = (
        args.output_cov
        or (out_path.with_name(f"{args.npz.stem}_cov_masked.png"))
    )
    out_cov.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(out_cov, dpi=args.dpi)
    print(f"Saved  → {out_cov}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
