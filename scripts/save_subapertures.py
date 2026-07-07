"""Save the sub-aperture mean/variance of `s_degraded` to disk.

Mirrors the head of `shear_averaging.main()` up to (and including) the
sub-aperture statistics on line 3877–3878 of `shear_averaging.py`,
then dumps only the two per-pixel moments a downstream consumer needs
to reconstruct the CoV mask (`cov_sq = sub_var / (sub_mean**2 + eps)`)
— the full `(N_subaperture, sub_size, N_rg)` amplitude stack itself
is *not* saved (it's a few GB for a scene-sized SLC and trivially
re-derivable if needed).

Contents of the output .npz:

  sub_mean : float32 (sub_size, N_rg)   subaps.mean(0)
  sub_var  : float32 (sub_size, N_rg)   subaps.var(0)
  N_subaperture, Number_of_Range_Looks
  az_m_per_px_s_degraded, rg_m_per_px_s_degraded  (pixel spacings)

`s_degraded` = range-Taylor-windowed SLC coherently summed by
`Number_of_Range_Looks` range bins (same recipe shear_averaging uses).

Reuses `shear_averaging`'s loaders (.tif / .npy / .npz) so any input
that script accepts works here unchanged. Nothing in `shear_averaging`
is modified.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shear_averaging import (  # noqa: E402
    _ICEYE_JSON_FIELDS,
    _load_iceye_sidecar_metadata,
    _load_slc_from_tiff,
    apply_range_window,
    compute_subapertures,
    degrade_range_resolution_range_sum,
)


def _resolve_input_path(user_path: Path) -> Path:
    """Accept the SLC path with or without a `.tif` extension.

    ICEYE deliveries are typically referred to by stem
    (`ICEYE_..._SLC`); `.tif` is the actual raster. If the user gives
    us a stem, look for the sibling `.tif` (or `.tiff`).
    """
    if user_path.exists():
        return user_path
    for suffix in (".tif", ".tiff", ".npy", ".npz"):
        candidate = user_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Neither {user_path} nor any .tif/.tiff/.npy/.npz sibling exists."
    )


def _load_slc_and_spacings(
    path: Path,
) -> tuple[np.ndarray, float, float]:
    """Return `(slc_complex, range_spacing_m, azimuth_spacing_m)`.

    Supports the three input types `shear_averaging` accepts:
      * `.tif` / `.tiff` (ICEYE GeoTIFF + sibling `.json` sidecar),
      * `.npz` bundle with `data` + metadata keys,
      * `.npy` — pixel spacings must be supplied on the CLI in that
        case (no sidecar is required upstream because the historical
        `.npy` patches carry no metadata).
    """
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        sidecar_json = path.with_suffix(".json")
        if not sidecar_json.exists():
            raise FileNotFoundError(
                f"GeoTIFF {path.name} requires a sidecar metadata JSON at "
                f"{sidecar_json}, but none was found."
            )
        iceye_md = _load_iceye_sidecar_metadata(sidecar_json)
        left = iceye_md["sar_observation_direction"].lower() == "left"
        s = _load_slc_from_tiff(path, left=left)
        return s, iceye_md["sar_pixel_spacing_range"], iceye_md["sar_pixel_spacing_azimuth"]

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as arch:
            if "data" not in arch.files:
                raise KeyError(f".npz bundle {path} is missing the 'data' array.")
            s = np.ascontiguousarray(arch["data"])
            missing = [k for k in _ICEYE_JSON_FIELDS if k not in arch.files]
            if missing:
                raise KeyError(
                    f".npz bundle {path} is missing metadata keys: {missing}."
                )
            return s, float(arch["sar_pixel_spacing_range"]), float(
                arch["sar_pixel_spacing_azimuth"]
            )

    if suffix == ".npy":
        raise ValueError(
            f"`.npy` input {path} has no bundled metadata. This helper "
            "needs the pixel spacings to reproduce shear_averaging's "
            "range-degradation step; please pass a `.tif` (with its "
            "sidecar `.json`) or a bundled `.npz` instead."
        )

    raise ValueError(
        f"Unsupported input extension {suffix!r} for {path}; "
        "expected .tif, .tiff, .npy or .npz."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Save the amplitude sub-apertures of the range-degraded SLC. "
            "Mirrors shear_averaging.py up to the sub-aperture statistics."
        )
    )
    parser.add_argument(
        "path", type=Path,
        help="ICEYE SLC input: .tif (+ sidecar .json), .npz bundle, or "
             "the ICEYE stem (extension auto-detected).",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Destination .npz path. Default: <input parent>/"
             "<stem>_subapertures.npz.",
    )
    parser.add_argument(
        "--n-subaperture", type=int, default=16,
        help="Number of Doppler sub-bands (same default as "
             "shear_averaging: 16).",
    )
    parser.add_argument(
        "--min-target-size-m", type=float, default=1.0,
        help="Same knob shear_averaging uses to derive "
             "Number_of_Range_Looks = int(min_target_size_m / range_spacing). "
             "Default: 1.0.",
    )
    parser.add_argument(
        "--range-sll-db", type=float, default=55.0,
        help="Taylor peak-sidelobe level (same default as shear_averaging).",
    )
    parser.add_argument(
        "--range-taylor-nbar", type=int, default=8,
        help="Taylor nbar (same default as shear_averaging).",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Write with np.savez instead of np.savez_compressed. Larger "
             "on disk but ~10x faster to save on a scene-sized stack.",
    )
    args = parser.parse_args()

    in_path = _resolve_input_path(args.path)
    out_path = args.output or in_path.with_name(f"{in_path.stem}_subapertures.npz")

    print(f"Input : {in_path}")
    print(f"Output: {out_path}")

    s, range_spacing, azimuth_spacing = _load_slc_and_spacings(in_path)
    print(
        f"Loaded SLC: shape={s.shape}, dtype={s.dtype}, "
        f"range_spacing={range_spacing} m, azimuth_spacing={azimuth_spacing} m"
    )

    Number_of_Range_Looks = int(args.min_target_size_m / range_spacing)
    Number_of_Range_Looks = max(Number_of_Range_Looks, 1)
    N_subaperture = int(args.n_subaperture)
    rg_m_per_px_degraded = range_spacing * Number_of_Range_Looks
    print(
        f"Number_of_Range_Looks={Number_of_Range_Looks}, "
        f"N_subaperture={N_subaperture}, "
        f"range spacing after degradation = {rg_m_per_px_degraded:g} m"
    )

    s_windowed = apply_range_window(
        s, sll_db=args.range_sll_db, nbar=args.range_taylor_nbar
    )
    print(
        f"Range Taylor window applied (PSL ≤ -{args.range_sll_db:.0f} dB, "
        f"nbar={args.range_taylor_nbar})."
    )

    s_degraded = degrade_range_resolution_range_sum(
        s_windowed, Number_of_Range_Looks
    )
    print(f"s_degraded: shape={s_degraded.shape}, dtype={s_degraded.dtype}")
    del s, s_windowed

    subaps = compute_subapertures(s_degraded, N_subaperture)
    print(
        f"Sub-apertures computed (transient): shape={subaps.shape}, "
        f"dtype={subaps.dtype} (≈ {subaps.nbytes / 1e9:.2f} GB in RAM, "
        "not saved to disk)"
    )

    sub_mean = subaps.mean(axis=0).astype(np.float32)
    sub_var = subaps.var(axis=0).astype(np.float32)
    del subaps
    print(
        f"sub_mean / sub_var: shape={sub_mean.shape}, dtype={sub_mean.dtype}"
    )

    payload = dict(
        sub_mean=sub_mean,
        sub_var=sub_var,
        N_subaperture=np.int32(N_subaperture),
        Number_of_Range_Looks=np.int32(Number_of_Range_Looks),
        az_m_per_px_s_degraded=np.float64(azimuth_spacing),
        rg_m_per_px_s_degraded=np.float64(rg_m_per_px_degraded),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    saver = np.savez if args.no_compress else np.savez_compressed
    saver(out_path, **payload)
    print(f"Wrote {out_path}  ({out_path.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
