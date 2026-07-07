"""Classify d_phase boxes as moving vs stationary using *only* the phase.

This is a companion to ``scripts/analyze_dphase_brightest.py``. For every
bounding box we build the 1-D amplitude-weighted phase trace

    phi(u) = d_phase[u, x_bright(u)]      u running over the rows of the box

where ``x_bright(u)`` is the range column with the strongest paired amplitude
in row ``u`` (same definition as in ``analyze_dphase_brightest``). ``phi(u)``
is the inter-pulse phase increment of the dominant scatterer inside the box.

Physical picture
================
After a perfectly tuned matched filter, a *stationary* scatterer focuses to
near a delta and adjacent azimuth samples are nearly identical, so their
conjugate product has phase ~ 0 with random jitter of std ~ 1/SNR.

A *constant cross-track velocity* moves the target through a Doppler bin,
which after focusing leaves a constant azimuth phase ramp on the SLC. In
``d_phase`` (the first difference of the SLC phase) that becomes a constant
offset:

    phi(u) ~= mu          mu proportional to the residual Doppler centroid.

An *along-track acceleration* (or any FM-rate mismatch) leaves a quadratic
azimuth phase, which becomes a linear ramp in ``d_phase``:

    phi(u) ~= mu + b*u    b proportional to the FM-rate mismatch.

So three things separate the two classes:
  * mu  != 0       -> constant-Doppler mover,
  * b   != 0       -> accelerating mover,
  * R   ~= 0       -> not a coherent point target at all (skip).

Features per box (amplitude weighted, w_u = paired amplitude at x_bright(u))
===========================================================================
  z_u   = exp(i phi_u)
  N_eff = (sum w)^2 / sum w^2                     effective sample count
  R0    = |sum w_u z_u| / sum w_u                 coherence at zero ramp
  mu    = arg(sum w_u z_u)                        circular mean
  Z0    = 2 N_eff R0^2                            Rayleigh statistic; under
                                                  H0 (uniform phase) Z0 ~ chi^2_2
                                                  -> Z0 > 6 means we are 95 %
                                                  sure the phase is non-random
  b_hat = argmax_b |sum w_u z_u exp(-i b u)|      best linear ramp via FFT
  Rb    = |sum w_u z_u exp(-i b_hat u)| / sum w_u coherence about the best ramp
  mu_b  = arg(sum w_u z_u exp(-i b_hat u))        mean about the best ramp
  Zb    = 2 N_eff Rb^2                            same test, ramp-compensated
  dphi  = b_hat * N                               total phase swing over the box

Decision rule
=============
A box is a mover iff it is coherent and not phase-flat:

    (Z0 > Z_min  OR  Zb > Z_min)
        AND ( |mu| > T_mu  OR  |dphi| > T_ramp )

Defaults that work well on this dataset (use ``--print-thresholds`` to let
the script pick the best ones from the labelled data):
    Z_min  = 6.0      ~ 95 % confidence the phase is non-random
    T_mu   = 0.6 rad
    T_ramp = 1.5 rad  (~ half a fringe over the box)

Usage
-----
    python scripts/classify_dphase_movers.py            # latest npz
    python scripts/classify_dphase_movers.py PATH.npz   # specific file

    # Override the truth band (default: x in [90, 140] are movers):
    python scripts/classify_dphase_movers.py --x-min 90 --x-max 140

    # Override the decision thresholds:
    python scripts/classify_dphase_movers.py --z-min 6 --t-mu 0.6 --t-ramp 1.5

    # Let the script grid-search the best (T_mu, T_ramp) for max F1:
    python scripts/classify_dphase_movers.py --tune
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_dphase_brightest import _box_brightest_phase, _box_bounds
from shear_averaging import DEFAULT_SAVE_DIR

DEFAULT_OUT_DIR = Path("/home/odogan/Desktop/cop/ship_phase")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class BoxFeatures:
    idx: int
    box: np.ndarray
    n: int
    n_eff: float
    R0: float
    mu: float
    Z0: float
    b_hat: float
    Rb: float
    mu_b: float
    Zb: float
    dphi: float
    # 2-D coherent-integration features (over all cells in the box, not just
    # the brightest column per row).  These are what actually separate the
    # classes on real data; the per-row "brightest column" features above are
    # kept for backwards-compatibility and as a baseline.
    R2d: float       # |sum amp^p z| / sum amp^p     (coherence over the box)
    mu2d: float      # arg(sum amp^p z)              (Doppler offset of the
                     #                                dominant scatterer)
    Z2d: float       # 2 N_eff_2d R2d^2              (Rayleigh statistic)
    b2d: float       # best linear ramp in u of the row-summed phasor T(u)
    R2d_b: float     # coherence about that ramp
    dphi2d: float    # b2d * H (total ramp over the box)
    truth: bool


def _features_for_box(
    box: np.ndarray,
    d_phase: np.ndarray,
    amp_raw: np.ndarray,
    fft_oversample: int = 8,
) -> tuple[int, float, float, float, float, float, float, float, float, float]:
    """Return ``(N, N_eff, R0, mu, Z0, b_hat, Rb, mu_b, Zb, dphi)`` for one box."""
    _u, _x, amp, phi = _box_brightest_phase(box, d_phase, amp_raw)
    n = int(phi.size)
    nan = float("nan")
    if n < 4:
        return n, 0.0, nan, nan, 0.0, nan, nan, nan, 0.0, nan

    w = amp.astype(np.float64)
    z = np.exp(1j * phi.astype(np.float64))
    sw = float(w.sum())
    if sw <= 0.0:
        return n, 0.0, nan, nan, 0.0, nan, nan, nan, 0.0, nan

    n_eff = (sw * sw) / float((w * w).sum() + 1e-30)

    S0 = (w * z).sum()
    R0 = abs(S0) / sw
    mu = float(np.angle(S0))
    Z0 = 2.0 * n_eff * R0 * R0

    M = max(256, fft_oversample * n)
    spec = np.fft.fftshift(np.fft.fft(w * z, n=M))
    b_grid = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(M, d=1.0))
    k_star = int(np.argmax(np.abs(spec)))
    b_hat = float(b_grid[k_star])
    Sb = spec[k_star]
    Rb = abs(Sb) / sw
    mu_b = float(np.angle(Sb))
    Zb = 2.0 * n_eff * Rb * Rb
    dphi = b_hat * n

    return n, n_eff, R0, mu, Z0, b_hat, Rb, mu_b, Zb, dphi


def _features_2d_for_box(
    box: np.ndarray,
    d_phase: np.ndarray,
    amp_raw: np.ndarray,
    amp_power: float = 2.0,
    fft_oversample: int = 8,
) -> tuple[float, float, float, float, float, float]:
    """Return ``(R2d, mu2d, Z2d, b2d, R2d_b, dphi2d)`` for one box.

    Sums coherently over the WHOLE box (every (u, x) cell) so the dominant
    scatterer reinforces and per-row range artefacts wash out.  Uses
    amp_pair = amp_raw[u, x] * amp_raw[u+1, x] as the weight, raised to
    ``amp_power`` so that the box's brightest scatterer dominates the
    integration (amp_power = 2 -> matched-filter-like weighting).
    """
    nan = float("nan")
    n_az_dphase, n_rg = d_phase.shape
    y0, y1, x0, x1 = _box_bounds(box, n_az_dphase, n_rg)
    h = y1 - y0
    w = x1 - x0
    if h < 4 or w < 1:
        return nan, nan, 0.0, nan, nan, nan

    phi = d_phase[y0:y1, x0:x1].astype(np.float64)
    amp_pair = (
        amp_raw[y0:y1, x0:x1].astype(np.float64)
        * amp_raw[y0 + 1:y1 + 1, x0:x1].astype(np.float64)
    )
    if amp_power != 1.0:
        amp_pair = np.power(amp_pair, amp_power)

    Z = amp_pair * np.exp(1j * phi)
    sw = float(amp_pair.sum())
    if sw <= 0.0:
        return nan, nan, 0.0, nan, nan, nan

    n_eff = (sw * sw) / float((amp_pair * amp_pair).sum() + 1e-30)
    S = Z.sum()
    R2d = abs(S) / sw
    mu2d = float(np.angle(S))
    Z2d = 2.0 * n_eff * R2d * R2d

    T = Z.sum(axis=1)
    M = max(256, fft_oversample * h)
    spec = np.fft.fftshift(np.fft.fft(T, n=M))
    b_grid = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(M, d=1.0))
    k_star = int(np.argmax(np.abs(spec)))
    b2d = float(b_grid[k_star])
    R2d_b = float(abs(spec[k_star]) / sw)
    dphi2d = b2d * h
    return R2d, mu2d, Z2d, b2d, R2d_b, dphi2d


def compute_all_features(
    boxes: np.ndarray,
    d_phase: np.ndarray,
    amp_raw: np.ndarray,
    truth: np.ndarray,
    amp_power: float = 2.0,
) -> list[BoxFeatures]:
    feats: list[BoxFeatures] = []
    for k, (box, t) in enumerate(zip(boxes, truth)):
        n, n_eff, R0, mu, Z0, b_hat, Rb, mu_b, Zb, dphi = _features_for_box(
            box, d_phase, amp_raw,
        )
        R2d, mu2d, Z2d, b2d, R2d_b, dphi2d = _features_2d_for_box(
            box, d_phase, amp_raw, amp_power=amp_power,
        )
        feats.append(BoxFeatures(
            idx=k, box=box, n=n, n_eff=n_eff,
            R0=R0, mu=mu, Z0=Z0,
            b_hat=b_hat, Rb=Rb, mu_b=mu_b, Zb=Zb, dphi=dphi,
            R2d=R2d, mu2d=mu2d, Z2d=Z2d,
            b2d=b2d, R2d_b=R2d_b, dphi2d=dphi2d,
            truth=bool(t),
        ))
    return feats


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def classify(
    feats: list[BoxFeatures],
    z_min: float = 70.0,
    t_mu: float = 0.0,
    t_ramp: float = 0.0,
    use_2d: bool = True,
) -> np.ndarray:
    """Phase-only mover decision per box.

    Primary feature is the box-wide coherence Rayleigh statistic ``Z2d``:
    a mover is a single dominant coherent scatterer, so its complex
    phasors all point the same way (Z2d large). Stationary clutter,
    multi-target boxes and distributed scenes average down (Z2d small).

    Optional refinements (set ``t_mu`` and/or ``t_ramp`` > 0 to require
    them):
      - ``|mu2d| > t_mu``   asks for a non-zero residual Doppler centroid
      - ``|b*H|  > t_ramp`` asks for an along-track ramp (acceleration)

    For the per-row brightest-column 1-D feature set, pass ``use_2d=False``.
    """
    out = np.zeros(len(feats), dtype=bool)
    for i, f in enumerate(feats):
        if use_2d:
            if not np.isfinite(f.R2d):
                continue
            coherent = f.Z2d > z_min
            non_flat = (
                (t_mu <= 0.0 or abs(f.mu2d) > t_mu)
                and (t_ramp <= 0.0 or abs(f.dphi2d) > t_ramp)
            )
        else:
            if not np.isfinite(f.R0):
                continue
            coherent = max(f.Z0, f.Zb) > z_min
            non_flat = (
                (t_mu <= 0.0 or abs(f.mu) > t_mu)
                and (t_ramp <= 0.0 or abs(f.dphi) > t_ramp)
            )
        out[i] = bool(coherent and non_flat)
    return out


def tune_thresholds(
    feats: list[BoxFeatures],
    z_grid: np.ndarray | None = None,
    use_2d: bool = True,
) -> tuple[float, dict]:
    """Grid-search ``z_min`` (primary feature) for the best F1 score."""
    truth = np.array([f.truth for f in feats], dtype=bool)
    if z_grid is None:
        z_grid = np.unique(np.r_[
            np.linspace(1.0, 30.0, 30),
            np.linspace(30.0, 300.0, 55),
        ])
    best = (-1.0, 70.0, {})
    for z_min in z_grid:
        pred = classify(feats, z_min=float(z_min), use_2d=use_2d)
        tp = int(((pred) & truth).sum())
        fp = int(((pred) & (~truth)).sum())
        fn = int(((~pred) & truth).sum())
        tn = int(((~pred) & (~truth)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best[0]:
            best = (f1, float(z_min),
                    {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
                     "precision": precision, "recall": recall, "f1": f1})
    _, z_min, stats = best
    return z_min, stats


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def _roc(score: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-score, kind="stable")
    t = truth[order]
    P = int(t.sum())
    N = int(len(t) - P)
    tp = np.cumsum(t.astype(int))
    fp = np.cumsum((~t).astype(int))
    tpr = tp / max(P, 1)
    fpr = fp / max(N, 1)
    fpr = np.concatenate(([0.0], fpr, [1.0]))
    tpr = np.concatenate(([0.0], tpr, [1.0]))
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def confusion(pred: np.ndarray, truth: np.ndarray) -> dict:
    tp = int(((pred) & truth).sum())
    fp = int(((pred) & (~truth)).sum())
    fn = int(((~pred) & truth).sum())
    tn = int(((~pred) & (~truth)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
# Plotting + reporting
# ---------------------------------------------------------------------------

def _print_table(feats: list[BoxFeatures], pred: np.ndarray) -> None:
    print(f"{'idx':>4} {'x_c':>5} {'y_c':>6} {'h':>5} "
          f"{'R2d':>5} {'Z2d':>7} {'mu2d':>7} {'b*H':>7} "
          f"{'truth':>5} {'pred':>4}")
    for f, p in zip(feats, pred):
        y, x, h, w = f.box
        truth_s = "M" if f.truth else "."
        pred_s = "M" if p else "."
        mark = " " if (p == f.truth) else "*"
        print(f"{f.idx:>4} {x:>5.0f} {y:>6.0f} {int(h):>5d} "
              f"{f.R2d:>5.2f} {f.Z2d:>7.1f} {f.mu2d:>+7.2f} {f.dphi2d:>+7.2f} "
              f"{truth_s:>5} {pred_s:>4}{mark}")


def _scatter_figure(
    feats: list[BoxFeatures],
    t_mu: float,
    t_ramp: float,
    z_min: float,
    src_name: str,
) -> plt.Figure:
    mu1d = np.array([abs(f.mu) for f in feats])
    dphi1d = np.array([abs(f.dphi) for f in feats])
    mu2d = np.array([abs(f.mu2d) for f in feats])
    dphi2d = np.array([abs(f.dphi2d) for f in feats])
    Z2d = np.array([f.Z2d for f in feats])
    R2d = np.array([f.R2d for f in feats])
    truth = np.array([f.truth for f in feats], dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax = axes[0]
    z_plot = np.clip(Z2d, 0.5, None)  # so log axis is happy
    ax.scatter(z_plot[~truth], mu2d[~truth], c="tab:blue", s=40, alpha=0.8,
               label="stationary")
    ax.scatter(z_plot[truth], mu2d[truth], c="tab:red", s=90, alpha=0.95,
               edgecolors="k", linewidths=0.7, label="mover (truth)")
    ax.axvline(z_min, color="tab:red", lw=0.9, ls="--",
               label=fr"$Z_{{2d}} > {z_min:.1f}$  (decision)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$Z_{2d} = 2\,N_{eff}\,R_{2d}^2$   (Rayleigh statistic, log)")
    ax.set_ylabel(r"$|\mu_{2d}|$  (residual Doppler centroid)  [rad]")
    ax.set_ylim(0.0, np.pi)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Primary feature: phase concentration over the box")

    ax = axes[1]
    feats_for_roc = [
        (r"$Z_{2d}$  (2D)", Z2d),
        (r"$R_{2d}$  (2D)", R2d),
        (r"$|\mu_{2d}|$  (2D)", mu2d),
        (r"$|b_{2d}\cdot H|$  (2D)", dphi2d),
        (r"$|\mu|$  (1D brightest)", mu1d),
        (r"$|b\cdot N|$  (1D brightest)", dphi1d),
    ]
    for label, s in feats_for_roc:
        fp, tp, a = _roc(s, truth)
        ls = "-" if "(2D)" in label else "--"
        lw = 2.0 if label.startswith("$Z_{2d}$") else 1.0
        ax.plot(fp, tp, lw=lw, ls=ls, label=f"{label},  AUC = {a:.3f}")
    ax.plot([0, 1], [0, 1], color="0.7", lw=0.7)
    ax.set_xlabel("false-positive rate")
    ax.set_ylabel("true-positive rate")
    ax.set_title("ROC per feature  (truth: x_c in mover band)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    fig.suptitle(f"d_phase mover classifier  --  source: {src_name}",
                 fontsize=11)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _latest_npz(directory: Path) -> Path | None:
    files = sorted(directory.glob("shear_*.npz"))
    return files[-1] if files else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", nargs="?", type=Path, default=None)
    p.add_argument("--x-min", type=float, default=90.0,
                   help="Boxes with x_c in [x_min, x_max] are TRUE movers.")
    p.add_argument("--x-max", type=float, default=140.0)
    p.add_argument("--z-min", type=float, default=70.0,
                   help="Rayleigh-test threshold (Z2d) above which a box is a "
                        "mover.  Z2d ~ chi^2_2 under H0; >70 means a strongly "
                        "concentrated phasor distribution.")
    p.add_argument("--t-mu", type=float, default=0.0,
                   help="Additional gate on |circular mean of phi|.  0 = off.")
    p.add_argument("--t-ramp", type=float, default=0.0,
                   help="Additional gate on |total ramp b*N| over the box.  "
                        "0 = off.")
    p.add_argument("--tune", action="store_true",
                   help="Grid-search z-min for best F1 on the labelled set.")
    p.add_argument("--amp-power", type=float, default=2.0,
                   help="Weighting exponent for the 2-D coherent integration "
                        "(2 = matched-filter-like, 1 = linear).")
    p.add_argument("--use-1d", action="store_true",
                   help="Use the per-row brightest-column features instead of "
                        "the (much more selective) 2-D box integration.")
    p.add_argument("--save", type=Path, default=None,
                   help=f"PNG output path (default: {DEFAULT_OUT_DIR}/<stem>_classifier.png).")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    src = args.path or _latest_npz(DEFAULT_SAVE_DIR)
    if src is None or not src.exists():
        raise SystemExit(
            f"No .npz found at {args.path or DEFAULT_SAVE_DIR}. "
            "Run scripts/shear_averaging.py first."
        )

    data = np.load(src)
    d_phase = data["d_phase"]
    amp_raw = data["amp_raw"]
    boxes = data["boxes_yxhw"]

    truth = (boxes[:, 1] >= args.x_min) & (boxes[:, 1] <= args.x_max)
    print(f"Loaded {src.name}:  total boxes={len(boxes)}, "
          f"movers (x in [{args.x_min:g}, {args.x_max:g}]) = {int(truth.sum())}")

    feats = compute_all_features(boxes, d_phase, amp_raw, truth,
                                 amp_power=args.amp_power)
    use_2d = not args.use_1d

    if args.tune:
        z_min, stats = tune_thresholds(feats, use_2d=use_2d)
        print(f"\n[tune] best F1 = {stats['f1']:.3f}  "
              f"at z_min={z_min:.2f}  "
              f"tp={stats['tp']}  fp={stats['fp']}  "
              f"fn={stats['fn']}  tn={stats['tn']}")
        args.z_min = z_min

    pred = classify(feats, z_min=args.z_min, t_mu=args.t_mu,
                    t_ramp=args.t_ramp, use_2d=use_2d)
    cm = confusion(pred, np.array([f.truth for f in feats], dtype=bool))
    z_name = "Z2d" if use_2d else "max(Z0,Zb)"
    mu_name = "|mu2d|" if use_2d else "|mu|"
    ramp_name = "|b*H|" if use_2d else "|b*N|"
    extra_gates = []
    if args.t_mu > 0.0:
        extra_gates.append(f"{mu_name}>{args.t_mu:.2f}")
    if args.t_ramp > 0.0:
        extra_gates.append(f"{ramp_name}>{args.t_ramp:.2f}")
    rule = f"{z_name} > {args.z_min:g}"
    if extra_gates:
        rule = f"({rule}) AND ({ ' AND '.join(extra_gates) })"
    print(f"\nDecision rule: {rule}")
    print(f"  TP={cm['tp']:3d}   FP={cm['fp']:3d}\n"
          f"  FN={cm['fn']:3d}   TN={cm['tn']:3d}\n"
          f"  precision={cm['precision']:.3f}  "
          f"recall={cm['recall']:.3f}  F1={cm['f1']:.3f}")
    print()
    _print_table(feats, pred)

    fig = _scatter_figure(feats, args.t_mu, args.t_ramp, args.z_min, src.name)
    out = args.save or (DEFAULT_OUT_DIR / f"{src.stem}_classifier.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nSaved figure -> {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
