#!/usr/bin/env python3
"""Compute + save the scene-wide azimuth phase derivative of a raw SLC.

    d_phase[u, v] = arg{ s[u+1, v] · conj(s[u, v]) }   ∈  (-π, π]

Runs stand-alone (does NOT sit inside `shear_averaging.py`) so the extra
buffers needed for the full-scene phase-derivative compute don't stack
on top of the main pipeline's already-heavy memory footprint.

Never applies range degradation — the phase derivative is computed
directly on the raw SLC. To match the per-box view (see
`compute_box_phase_estimates` in `shear_averaging.py`, which computes
`arg{ s_degraded[u+1,v] · s*_degraded[u,v] }` for **adjacent** azimuth
rows) the compute here also uses adjacent-row pairs. Only the display
grid is decimated: for each display row `i_out`, we pick the SLC row
pair `(i_out·step, i_out·step + 1)`. This keeps the wrap frequency
identical to per-box — stride-then-diff (`s[::step][1:] · s*[::step][:-1]`)
would multiply every phase change by `step` and cause ~step× extra
wrapping, which was the earlier bug. Range is decimated via a plain
stride (no phase difference is taken along range, so a stride is safe).

Memory: the fancy-indexed row pairs materialise only the decimated grid
(≈ 300 MB per copy for a full ICEYE scene at the default 60 M-px cap),
never the full-resolution complex intermediate `s[1:] · s*[:-1]` that
would sit around 15 GB.

Usage
-----
    python scripts/scene_phase_derivative.py <slc.tif|.npy|.npz> \
        [--save output.png] \
        [--max-display-pixels 60000000] \
        [--dpi 350]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _load_slc_from_tiff(tiff_path: Path, left: bool) -> np.ndarray:
    """Decode an ICEYE SLC GeoTIFF into a complex64 array (axis 0 = azimuth).

    Mirrors `_load_slc_from_tiff` in `shear_averaging.py`: band 1 = UInt16
    amplitude (scale/offset), band 2 = UInt16 wrapped phase (scale/offset).
    For left-look geometry we flip range so the output is always
    "shadows-down", then transpose so axis 0 = azimuth, axis 1 = range.
    """
    from osgeo import gdal

    ds = gdal.Open(str(tiff_path))
    if ds is None:
        raise FileNotFoundError(f"Could not open {tiff_path}")
    if ds.RasterCount < 2:
        raise ValueError(
            f"{tiff_path.name} has {ds.RasterCount} bands; need ≥ 2 (amp, phase)"
        )
    amp_band = ds.GetRasterBand(1)
    pha_band = ds.GetRasterBand(2)
    amp_scale = amp_band.GetScale() or 1.0
    amp_offset = amp_band.GetOffset() or 0.0
    pha_scale = pha_band.GetScale() or 1.0
    pha_offset = pha_band.GetOffset() or 0.0

    amp = amp_band.ReadAsArray()
    pha = pha_band.ReadAsArray()
    ds = None

    data = amp.astype(np.complex64) * amp_scale + amp_offset
    del amp
    data *= np.exp(
        -1j * (pha.astype(np.float32) * pha_scale + pha_offset)
    )
    del pha

    if left:
        data = np.fliplr(data)
    return np.ascontiguousarray(data.T)


def _load_slc(path: Path) -> tuple[np.ndarray, str]:
    """Load an SLC from .tif / .tiff / .npy / .npz. Returns (s, source_name)."""
    suffix = path.suffix.lower()

    if suffix in (".tif", ".tiff"):
        sidecar = path.with_suffix(".json")
        left = False
        if sidecar.exists():
            md = json.loads(sidecar.read_text())
            left = str(md.get("sar_observation_direction", "")).lower() == "left"
            print(
                f"  sidecar {sidecar.name}: "
                f"sar_observation_direction = "
                f"{md.get('sar_observation_direction', '?')} → left={left}"
            )
        s = _load_slc_from_tiff(path, left=left)
        return s, path.name

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as arch:
            if "data" not in arch.files:
                raise KeyError(
                    f".npz bundle {path} is missing the 'data' array."
                )
            s = np.ascontiguousarray(arch["data"])
        return s, path.name

    if suffix == ".npy":
        return np.load(path), path.name

    raise ValueError(
        f"Unsupported SLC suffix {suffix!r}. Expected .tif / .tiff / .npy / .npz."
    )


def scene_phase_derivative(
    slc_path: Path,
    save: Path,
    max_display_px: int = 60_000_000,
    dpi: int = 350,
    figsize: tuple[float, float] = (14.0, 16.0),
) -> None:
    """Compute + save the raw-SLC scene-wide azimuth phase derivative."""
    s, name = _load_slc(slc_path)
    if not np.iscomplexobj(s):
        raise TypeError(
            f"{name} is not complex (dtype={s.dtype}); the phase derivative "
            "needs the complex SLC, not an amplitude image."
        )
    h_full, w_full = int(s.shape[0]), int(s.shape[1])
    print(f"Loaded: {name}  shape=({h_full}, {w_full})  dtype={s.dtype}")

    step = max(1, int(np.ceil(np.sqrt(s.size / max_display_px))))

    # For each display row `i_out` we take the ADJACENT-row pair (i_out·step, i_out·step + 1) on the raw SLC so the phase-derivative wrap frequency matches the per-box view. Constraint: i_out·step + 1 ≤ h_full - 1 ⇒ h_out = (h_full - 2) // step + 1.
    h_out = max(0, (h_full - 2) // step + 1)
    row_indices = np.arange(h_out, dtype=np.int64) * step
    print(
        f"Display cap {max_display_px:,} px → decimation step ×{step} "
        f"→ display grid ({h_out}, {int(np.ceil(w_full / step))})  "
        f"[adjacent-row pairs at az rows i·{step}, i·{step}+1]"
    )

    # Fancy-index axis 0 with the adjacent-pair rows; stride axis 1 in range. Both `pair0` and `pair1` are fresh decimated-grid copies (~ hundreds of MB, not ~15 GB). Range is decimated via a plain stride: no phase difference is taken along range, so striding it does NOT affect wrap frequency.
    pair0 = s[row_indices, ::step]
    pair1 = s[row_indices + 1, ::step]
    del s
    d_phase = np.angle(pair1 * np.conj(pair0)).astype(np.float32, copy=False)
    del pair0, pair1
    print(f"d_phase (decimated) shape: {d_phase.shape}  dtype={d_phase.dtype}")

    d_phase_rot = d_phase.T[:, ::-1]

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    fig.suptitle(
        r"Scene-wide azimuth phase derivative  "
        r"$\partial\varphi/\partial u = "
        r"\arg\{s[u{+}1,v]\cdot s^\ast[u,v]\}$"
        f"  —  {name}",
        fontsize=11,
    )
    ax.imshow(
        d_phase_rot, cmap="twilight", aspect="auto",
        vmin=-np.pi, vmax=np.pi,
        extent=(0.0, float(h_full), float(w_full), 0.0),
    )
    ax.set_title(
        f"d_phase on raw SLC (no range degradation) — "
        f"native-spacing adjacent-row pairs, displayed decimated ×{step} "
        f"in azimuth and ×{step} in range, wrapped in (-π, π]"
    )
    ax.set_xlabel("azimuth pixel")
    ax.set_ylabel("range pixel")

    save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save, dpi=dpi)
    print(f"Saved → {save}")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compute + save the scene-wide azimuth phase derivative of an "
            "ICEYE SLC. Runs on the raw SLC (no range degradation)."
        ),
    )
    ap.add_argument(
        "input", type=Path,
        help="Path to the SLC file (.tif / .tiff / .npy / .npz).",
    )
    ap.add_argument(
        "--save", type=Path, default=None,
        help=(
            "Output PNG path. Default: <input_stem>_phase_derivative.png "
            "next to the input file."
        ),
    )
    ap.add_argument(
        "--max-display-pixels", type=int, default=60_000_000,
        help="Max decimated view size (default 60 M px, same cap as _input_slc.png).",
    )
    ap.add_argument("--dpi", type=int, default=350)
    args = ap.parse_args()

    save = args.save or args.input.with_name(
        args.input.stem + "_phase_derivative.png"
    )
    scene_phase_derivative(
        args.input, save,
        max_display_px=args.max_display_pixels,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
