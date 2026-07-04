"""SAR moving target detection via phase derivative analysis.

Given an SLC patch s with axes (azimuth=vertical, range=horizontal):

1. Range resolution is degraded by coherently summing range sub-bands.
   This suppresses range-dependent phase noise while preserving the
   azimuth phase structure of moving targets.

2. The azimuth phase derivative is computed on the degraded image:
       d_phase[i, v] = arg{ s_degraded[i+1, v] · conj(s_degraded[i, v]) }
   Result is real-valued, -pi to pi, axis 0 = azimuth pixels (spatial).

3. Moving target detection: each range column of d_phase is convolved
   with a sinusoidal kernel. A moving target produces an azimuth phase
   gradient with a sign inversion (ramps up then down, or vice versa),
   which matches the sinusoidal kernel and gives a strong response.
   Noise and stationary targets average to zero.
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib


# --------------------------------------------------------------------------- Config loader.  All CLI parameters + their help text now live in a JSON sidecar (`shear_averaging_config.json` next to this file by default). The loader walks the tree, treats every `{value, help, ...}` leaf as a single value, and flattens the result into a `SimpleNamespace` so downstream code writes `cfg.min_slope_deg` regardless of which group the key sits under. The loader must run before `matplotlib.pyplot` is imported so `cfg.show` can drive the backend selection.

_CONFIG_LEAF_KEYS: frozenset[str] = frozenset({"value", "help", "choices", "unit"})


def _split_cli(argv: list[str]) -> tuple[Path | None, Path | None, str | None]:
    """Extract CLI overrides from ``argv``.

    Positional args are dispatched by suffix so the historical
    ``python scripts/shear_averaging.py [config.json]`` form still
    works alongside the more ergonomic
    ``python scripts/shear_averaging.py <slc.tif> --save <png>``:

      * ``*.json``      → override the JSON config path (default is
                          the sidecar next to this script).
      * any other path  → override ``input.path`` from the config.

    Recognised flags (all optional):

      * ``--save <path>`` / ``--save=<path>`` — override ``output.save``.
        Accepts the same shortcuts as the JSON field: ``auto`` expands
        to a timestamped file inside ``output.save_dir`` and ``null``
        (or the ``--no-save`` alias) disables PNG saving.
      * ``--no-save`` — shortcut for ``--save null``.

    Raises ``ValueError`` on unrecognised flags or duplicate positional
    args of the same kind so silent typos fail loudly at import time.
    """
    config_path: Path | None = None
    input_path: Path | None = None
    save_override: str | None = None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "--no-save":
            save_override = "null"
            i += 1
            continue
        if arg == "--save":
            if i + 1 >= len(argv):
                raise ValueError("--save requires a value")
            save_override = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--save="):
            save_override = arg[len("--save="):]
            i += 1
            continue
        if arg.startswith("-"):
            raise ValueError(f"Unrecognised CLI flag: {arg!r}")
        candidate = Path(arg).expanduser()
        if candidate.suffix.lower() == ".json":
            if config_path is not None:
                raise ValueError(
                    "Multiple .json positional args on CLI; expected at "
                    "most one (the config-file override)."
                )
            config_path = candidate.resolve()
        else:
            if input_path is not None:
                raise ValueError(
                    "Multiple non-.json positional args on CLI; expected "
                    "at most one (the `input.path` override)."
                )
            input_path = candidate
        i += 1
    return config_path, input_path, save_override


def _resolve_save_field(save_field, save_dir) -> Path | None:
    """Turn a raw ``save`` value (from JSON or ``--save``) into a Path or None.

    Mirrors the shortcuts documented on the ``output.save`` help
    string: ``None`` disables saving, ``"auto"`` builds a timestamped
    path inside ``save_dir``, any other string is a literal path.
    Recognises ``"null"`` / ``"none"`` / ``""`` as synonyms for
    ``None`` so the same CLI value works for JSON- and shell-quoted
    overrides.
    """
    if save_field is None:
        return None
    if isinstance(save_field, str):
        s = save_field.strip()
        if s.lower() in ("null", "none", ""):
            return None
        if s == "auto":
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = uuid.uuid4().hex[:6]
            base = Path(save_dir) if save_dir is not None else Path.cwd()
            return base / f"shear_{ts}_{suffix}.png"
        return Path(s).expanduser()
    return Path(str(save_field)).expanduser()


def _flatten_config(node, sink: dict) -> None:
    """Recursively walk the nested config dict and populate `sink` with `{leaf_name: leaf_value}` pairs, treating `{value, help, ...}` dicts as leaves."""
    if not isinstance(node, dict):
        return
    for key, val in node.items():
        if key.startswith("_"):
            continue
        if isinstance(val, dict) and set(val.keys()) <= _CONFIG_LEAF_KEYS and "value" in val:
            if key in sink:
                raise ValueError(
                    f"Duplicate leaf key {key!r} in config — flatten needs unique names"
                )
            sink[key] = val["value"]
        elif isinstance(val, dict):
            _flatten_config(val, sink)


def _load_config(path: Path) -> SimpleNamespace:
    """Read `path`, flatten it, resolve save-path shortcuts and coerce path-like fields to `Path` objects."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}. Pass a path as the first CLI arg "
            "or drop shear_averaging_config.json next to shear_averaging.py."
        )
    with path.open() as f:
        raw = json.load(f)

    flat: dict = {}
    _flatten_config(raw, flat)

    for key in ("path", "metadata_from", "save_dir"):
        if flat.get(key) is not None:
            flat[key] = Path(str(flat[key])).expanduser()

    flat["save"] = _resolve_save_field(flat.get("save"), flat.get("save_dir"))

    return SimpleNamespace(**flat)


_CLI_CONFIG_PATH, _CLI_INPUT_PATH, _CLI_SAVE = _split_cli(sys.argv)
_CFG_PATH: Path = (
    _CLI_CONFIG_PATH
    if _CLI_CONFIG_PATH is not None
    else Path(__file__).with_name("shear_averaging_config.json")
)
_CFG: SimpleNamespace = _load_config(_CFG_PATH)
# CLI overrides applied on top of the JSON defaults so scene-specific runs don't need a bespoke config file. `input.path` accepts any non-.json positional arg; `output.save` accepts `--save <path>` (with `auto` / `null` / `--no-save` shortcuts).
if _CLI_INPUT_PATH is not None:
    _CFG.path = _CLI_INPUT_PATH
if _CLI_SAVE is not None:
    _CFG.save = _resolve_save_field(_CLI_SAVE, getattr(_CFG, "save_dir", None))
_SHOW_FIGURES: bool = bool(_CFG.show)
if not _SHOW_FIGURES:
    matplotlib.use("Agg")  # Headless: figures are saved to disk, never shown.
else:
    # `show=true` can open dozens to hundreds of figures (per-box PNGs). Silence the "More than 20 figures have been opened" warning so the log stays readable when the user wants everything on screen.
    matplotlib.rcParams["figure.max_open_warning"] = 0
import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import matplotlib.colors  # noqa: E402  (must follow matplotlib.use)
import matplotlib.patheffects  # noqa: E402  (must follow matplotlib.use)
import numpy as np
from scipy.signal import fftconvolve

import ctypes  # noqa: E402  (used only by `_release_glibc_arenas`)
import gc  # noqa: E402  (used to force cyclic-ref collection between AF calls)
import os  # noqa: E402  (used by `_rss_gb` to read /proc/self/status)


# `libc.malloc_trim(pad=0)` asks glibc's malloc to return unused arenas back to the OS. Without it, every ~40k chip-sized FFT / phase-ramp allocation performed by the AF+PGA pipeline over ~200 boxes sinks into the per-thread arena free-lists and never comes back — RSS climbs monotonically even though live Python objects stay flat, and on high-accept-rate scenes (SXK97E: ~90% of boxes trigger PGA) the accumulator crosses the 125 GB RAM ceiling before pre-compute finishes. Loading libc lazily-once at import time so the AF loop path is a plain function call, and swallowing OSError so non-glibc / non-Linux hosts (macOS, musl) silently no-op instead of raising.
try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=False)
    _LIBC.malloc_trim.argtypes = [ctypes.c_size_t]
    _LIBC.malloc_trim.restype = ctypes.c_int
    _HAS_MALLOC_TRIM = True
except (OSError, AttributeError):
    _LIBC = None
    _HAS_MALLOC_TRIM = False


def _release_glibc_arenas() -> None:
    """Run Python's cyclic GC then hand freed heap back to the OS.

    Called between AF pre-compute iterations to stop glibc's per-thread
    arena free-lists from monotonically growing across the loop. On
    non-glibc systems the ``malloc_trim`` call is a no-op.
    """
    gc.collect()
    if _HAS_MALLOC_TRIM:
        _LIBC.malloc_trim(0)


def _rss_gb() -> tuple[float, float]:
    """Return ``(current_rss_gb, peak_rss_gb)`` from ``/proc/self/status``.

    ``resource.getrusage(ru_maxrss)`` only exposes the high-water mark
    so it can't tell us whether ``malloc_trim`` is actually holding
    memory flat across iterations. Reading both ``VmRSS`` (current)
    and ``VmHWM`` (peak) directly from ``/proc/self/status`` gives the
    full picture at negligible cost (a single small ``read``). Returns
    ``(0.0, 0.0)`` on non-Linux hosts where the file is not present.
    """
    try:
        with open(f"/proc/{os.getpid()}/status", "r") as fh:
            rss_kb = 0
            hwm_kb = 0
            for line in fh:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                elif line.startswith("VmHWM:"):
                    hwm_kb = int(line.split()[1])
                if rss_kb and hwm_kb:
                    break
        return rss_kb / (1024.0 * 1024.0), hwm_kb / (1024.0 * 1024.0)
    except OSError:
        return 0.0, 0.0


def _maybe_close(fig) -> None:
    """Close `fig` unless the user asked for interactive display via `--show`.

    In interactive mode figures are held open so `plt.show()` at the
    end of `main()` can pop them up together. In headless mode (the
    default `Agg` backend) each figure is closed immediately after
    `savefig` to free memory — this matters for scenes that produce
    hundreds of per-box PNGs.
    """
    if not _SHOW_FIGURES:
        plt.close(fig)


class _Stopwatch:
    """Lightweight pipeline timer used by `main()` to report total wall
    time plus per-stage durations. `mark()` prints/records the time
    spent since the previous mark (or `__init__`); `report()` prints
    the consolidated summary at the end.
    """

    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.steps: list[tuple[str, float]] = []
        self._pause_start: float | None = None

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        dt = now - self.last
        self.steps.append((label, dt))
        # Trim glibc's malloc arena at every stage boundary. Pipeline stages like `degrade_range_resolution_range_sum` and `compute_subapertures` transiently allocate 15 GB+ complex64 workspace buffers that Python frees on stage exit — but glibc's per-thread arena keeps those blocks on its free-lists indefinitely (default `M_TRIM_THRESHOLD` is 128 KB, but that only triggers on ``top`` chunks; scattered ~15 GB blocks in the middle of the arena never come back to the OS on their own). Without this trim call, RSS ratchets up monotonically stage by stage — the 74 GB peak observed on SXK97E during the pre-AF stages persisted as reserved but idle memory through the entire AF pre-compute loop, so the subsequent 7.27 GB `np.abs(s)` amplitude alloc had to squeeze into a heap that appears "empty" (current 17 GB) but is actually reserved (peak 74 GB), risking SIGKILL. Cost: a few ms per stage; benefit: RSS floor tracks live objects, so downstream stages get to work on a clean heap.
        _release_glibc_arenas()
        _rss_cur, _rss_peak = _rss_gb()
        print(
            f"  [time] {label:60s} {dt:8.2f} s   "
            f"RSS={_rss_cur:5.2f}/{_rss_peak:5.2f} GB (cur/peak)"
        )
        self.last = now

    def pause(self) -> None:
        """Stop counting wall time. Spans bracketed by `pause` / `resume`
        are excluded from both the current step's duration and from the
        final total reported by `report()`.
        """
        if self._pause_start is None:
            self._pause_start = time.perf_counter()

    def resume(self) -> None:
        """Resume counting wall time. The duration between the matching
        `pause` and this call is discarded by advancing both `self.last`
        and `self.t0` by that delta.
        """
        if self._pause_start is None:
            return
        delta = time.perf_counter() - self._pause_start
        self.last += delta
        self.t0 += delta
        self._pause_start = None

    def report(self) -> None:
        total = time.perf_counter() - self.t0
        bar = "=" * 82
        print(bar)
        print("Processing-time summary")
        print(bar)
        for label, dt in self.steps:
            frac = 100.0 * dt / total if total > 0 else 0.0
            print(f"  {label:60s} {dt:8.2f} s  ({frac:5.1f}%)")
        print("-" * 82)
        print(f"  {'TOTAL':60s} {total:8.2f} s")
        print(bar)


# Subset of `core.autofocus.AutofocusTask.run` (lines 263-269) that is needed to drive the scalar pixel/resolution parameters of this script from a real ICEYE acquisition rather than the hard-coded patch defaults.
_ICEYE_JSON_FIELDS = (
    "sar_resolution_range",
    "sar_resolution_azimuth",
    "sar_pixel_spacing_range",
    "sar_pixel_spacing_azimuth",
    "iceye_acquisition_prf",
    "start_datetime",
    "end_datetime",
)


def _iso_duration_seconds(start_iso: str, end_iso: str) -> float:
    """Return ``end_iso − start_iso`` in seconds.

    Handles ICEYE STAC timestamps that end in ``Z`` and may have more
    than 6 fractional digits, both of which trip up ``datetime``'s
    parser before Python 3.11. We normalise to a trailing ``+00:00``
    and clamp the fractional second to microsecond precision.
    """
    from datetime import datetime

    def _parse(ts: str) -> datetime:
        ts = ts.strip()
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        if "." in ts:
            head, frac = ts.split(".", 1)
            for sep in ("+", "-"):
                if sep in frac:
                    digits, tz_sep, tz = frac.partition(sep)
                    frac = digits[:6] + tz_sep + tz
                    break
            else:
                frac = frac[:6]
            ts = f"{head}.{frac}"
        return datetime.fromisoformat(ts)

    return (_parse(end_iso) - _parse(start_iso)).total_seconds()


def _load_iceye_sidecar_metadata(json_path: Path) -> dict:
    """Read an ICEYE STAC-style sidecar JSON and return a flat metadata dict.

    The sidecar uses prefixed STAC keys (``sar:pixel_spacing_range``,
    ``iceye:acquisition_prf``, …). We rename them to the underscore form
    used by `IceyeMetadata` in the QGIS plugin so the rest of the script
    can pretend it was handed the same object the autofocus task uses.

    Optional keys (``iceye_processing_prf``, ``iceye_range``,
    ``sar_center_frequency``) are picked up when present so the
    Doppler-domain range-walk → v_a mapping in
    ``_va_from_range_walk_mps`` can use the actual scene geometry
    instead of hard-coded nominals; older sidecars that don't expose
    them still load fine (the fields are left absent from the dict and
    the AF path falls back to defaults).
    """
    import json

    with json_path.open() as f:
        doc = json.load(f)
    props = doc.get("properties", {}) or {}

    def _get(*keys):
        for k in keys:
            if k in props:
                return props[k]
        raise KeyError(
            f"Sidecar {json_path.name} is missing any of {keys!r}"
        )

    md = {
        "sar_resolution_range": float(_get("sar:resolution_range")),
        "sar_resolution_azimuth": float(_get("sar:resolution_azimuth")),
        "sar_pixel_spacing_range": float(_get("sar:pixel_spacing_range")),
        "sar_pixel_spacing_azimuth": float(_get("sar:pixel_spacing_azimuth")),
        "iceye_acquisition_prf": float(_get("iceye:acquisition_prf")),
        "start_datetime": str(_get("start_datetime")),
        "end_datetime": str(_get("end_datetime")),
        "sar_observation_direction": str(
            props.get("sar:observation_direction", "left")
        ),
    }
    # Optional geometry fields — driven by the Doppler-domain autofocus → v_a mapping. Stripmap sidecars typically lack ``iceye:processing_prf`` (acquisition PRF is what the compressed image is sampled at); spotlight sidecars expose a much higher value that reflects the beam-steering-inflated effective sampling. Left absent from the dict when the sidecar doesn't have them so callers can distinguish "known value" from "assumed default".
    for src_key, dst_key in (
        ("iceye:processing_prf", "iceye_processing_prf"),
        ("iceye:range", "iceye_range"),
        ("sar:center_frequency", "sar_center_frequency"),
    ):
        if src_key in props:
            md[dst_key] = float(props[src_key])
    return md


def _load_slc_from_tiff(tiff_path: Path, left: bool) -> np.ndarray:
    """Decode an ICEYE SLC GeoTIFF into a complex64 array.

    The GeoTIFF stores amplitude in band 1 and wrapped phase in band 2,
    each as UInt16 with per-band scale/offset (matches
    `core.raster.read_slc_layer`). The returned array follows the same
    "shadows-down" orientation that the QGIS plugin uses, so
    ``data.shape == (azimuth, range)`` for both `.npy` patches and
    `.tif` scenes.
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
    # Free the integer copy before we allocate the 8× larger complex phasor on top of `data`; on a full ICEYE spot scene `amp`/`pha` are >1.5 GB each.
    del amp
    data *= np.exp(
        -1j * (pha.astype(np.float32) * pha_scale + pha_offset)
    )
    del pha

    # Mirror `core.raster.toggle_shadows_down`: flip range for left-look geometry, then transpose so axis 0 = azimuth, axis 1 = range.
    if left:
        data = np.fliplr(data)
    return np.ascontiguousarray(data.T)


# --------------------------------------------------------------------------- Core processing ---------------------------------------------------------------------------

def apply_range_window(
    s: np.ndarray,
    sll_db: float = 55.0,
    nbar: int = 8,
) -> np.ndarray:
    """Aggressively suppress range sidelobes by tapering the SLC range
    spectrum with a Taylor window.

    The SLC range impulse response is sinc-like with first sidelobes at
    only ≈ −13 dB because the SAR matched filter leaves a
    near-rectangular spectrum across the chirp bandwidth. Multiplying
    that spectrum by a Taylor window collapses every sidelobe to
    ≤ −`sll_db`: the inner `nbar` lobes on each side are held flat at
    exactly that level, the rest decay as a sinc tail. The cost is a
    broader main lobe (≈ 1.5× rectangular at −40 dB, ≈ 1.85× at −55 dB,
    ≈ 2.0× at −65 dB).

    Taylor is the SAR canonical choice because the sidelobe level is a
    single dial, with the inner-lobe count `nbar` controlling how many
    sidelobes are forced flat (higher `nbar` → closer to ideal
    Dolph-Chebyshev, slightly more energy in the wings). Axis 0
    (azimuth) is untouched.

    Parameters
    ----------
    s : complex ndarray, shape (N_az, N_range)
        Input SLC; axis 1 = range (the axis being windowed).
    sll_db : float
        Peak sidelobe level in dB **below** the main lobe.
        ≈ 35 → Hamming-equivalent.
        45  → typical airborne-SAR setting.
        55+ → very low; for high dynamic range scenes (ship vs wake,
              ship vs small boat). 0 disables and returns `s`.
    nbar : int
        Number of equal-level sidelobes flattened on each side of the
        main lobe. Must be large enough that the design is well-posed
        for the requested `sll_db`; otherwise the realised PSL falls a
        few dB short of the dial. Empirically `nbar=8` saturates the
        target up to ≈ −55 dB; bump to 12–16 only if pushing past
        −60 dB.

    Returns
    -------
    s_w : complex ndarray, same shape as `s`.
        Range-windowed SLC. Window is mean-normalised so global
        amplitude is preserved (no DC gain change vs. `s`).
    """
    if sll_db <= 0.0:
        return s

    from scipy.signal.windows import taylor

    N_range = s.shape[1]
    w = taylor(N_range, nbar=nbar, sll=sll_db, norm=False).astype(np.float64)
    w /= w.mean()                                       # preserve mean amplitude
    S = np.fft.fftshift(np.fft.fft(s, axis=1), axes=1)  # DC at centre
    S *= w[np.newaxis, :]
    return np.fft.ifft(np.fft.ifftshift(S, axes=1), axis=1)


def degrade_range_resolution_range_sum(
    ship: np.ndarray,
    Number_of_Range_Looks: int,
    *,
    sll_db: float = 0.0,
    nbar: int = 8,
) -> np.ndarray:
    """Degrade range resolution by coherently summing range sub-bands.

    An optional Taylor range-sidelobe window can be folded into the
    sub-aperture loop by passing ``sll_db > 0``. This is mathematically
    identical to calling :func:`apply_range_window` on ``ship`` first
    and then feeding the windowed SLC to this function, but avoids
    materialising a full-size windowed copy of the input.

    Parameters
    ----------
    ship : complex ndarray, shape (N_az, N_range)
        Input SLC patch. Axis 0 = azimuth (spatial), axis 1 = range (spatial).
    Number_of_Range_Looks : int
        Number of coherent range sub-bands to sum.
    sll_db : float, keyword-only, default 0.0
        If > 0, multiply each range sub-band spectrum by the matching
        slice of a mean-normalised Taylor window designed for a peak
        sidelobe level of ``-sll_db`` dB. 0 disables windowing.
    nbar : int, keyword-only, default 8
        Number of equal-level Taylor sidelobes flattened on each side of
        the main lobe. Ignored when ``sll_db <= 0``.

    Returns
    -------
    s_degraded : complex ndarray, shape (N_az, N_range // Number_of_Range_Looks)
        Range-degraded SLC. Axis 0 is still azimuth spatial pixels.
    """
    N_range_full = ship.shape[1]
    N_range = N_range_full // Number_of_Range_Looks

    # Force complex64 for both the range spectrum and the accumulator; on a full ICEYE scene each buffer is ≈15 GB at complex128 vs. ≈7.5 GB at complex64, and the OOM killer will hit long before the loop finishes if either upcasts. `astype(..., copy=False)` is a no-op when numpy's FFT already preserves precision (numpy ≥ 1.17) and a one-time downcast otherwise.
    out_dtype = ship.dtype if ship.dtype == np.complex64 else np.complex64
    ship_fft_range = np.fft.fftshift(np.fft.fft(ship, axis=1), axes=1).astype(
        out_dtype, copy=False,
    )

    if sll_db > 0.0:
        from scipy.signal.windows import taylor
        w = taylor(N_range_full, nbar=nbar, sll=sll_db, norm=False).astype(np.float32)
        w /= w.mean()
    else:
        w = None

    s_degraded = np.zeros((ship.shape[0], N_range), dtype=out_dtype)
    for k in range(Number_of_Range_Looks):
        band = ship_fft_range[:, k * N_range:(k + 1) * N_range]
        if w is not None:
            band = band * w[k * N_range:(k + 1) * N_range][np.newaxis, :]
        # `astype(out_dtype, copy=False)` guards against a stray complex128 rhs upcasting the complex64 accumulator (which would otherwise materialise a scene-sized complex128 temp inside `+=`).
        s_degraded += np.fft.ifft(
            np.fft.ifftshift(band, axes=1), axis=1,
        ).astype(out_dtype, copy=False)
        del band

    del ship_fft_range
    return s_degraded

def _phasor_ls_slope(
    phi: np.ndarray,
    w: np.ndarray,
    n_iter_gn: int = 2,
) -> float:
    """Weighted phasor least-squares slope of a wrapped phase signal.

    Fits  phi[n] ≈ a + b·n  (mod 2π)  by minimising

        J(a, b) = Σ w[n] · |exp(j·phi[n]) − exp(j·(a + b·n))|²
                = 2 · Σ w[n] · (1 − cos(phi[n] − a − b·n))

    over (a, b) and returns the slope `b`. The cost depends on `phi`
    only through sin/cos, so a real ±2π wrap of `phi` inside the run is
    indistinguishable from no wrap — no `np.unwrap` is involved.

    Two-stage fit:

      (A) Initial slope from Kay's weighted lag-1 estimator,

              b₀ = arg( Σ_n v[n] · e^{j·(phi[n+1] − phi[n])} ),
              v[n] = sqrt(w[n] · w[n+1]).

          At high SNR this is the maximum-likelihood slope of a
          linear-phase signal in Gaussian noise; the bright (high-w)
          pairs dominate the sum.

      (B) Newton refinement of (a, b) on the full weighted cost, with a
          centred parameterisation that decouples slope and intercept
          at the optimum. Two iterations are enough from the Kay init.

    Parameters
    ----------
    phi : (L,) float ndarray
        Wrapped phase samples (rad), in (-π, π].
    w : (L,) float ndarray, non-negative
        Per-sample weight. For phi = arg(s[n+1]·conj(s[n])) the
        natural choice is w[n] = |s[n]|·|s[n+1]|, i.e. the modulus of
        the complex number whose argument is phi[n]. Set entries to 0
        to drop noise-dominated samples entirely.
    n_iter_gn : int
        Number of Newton iterations to run after the Kay init. 0
        returns Kay's estimate alone.

    Returns
    -------
    b : float
        Slope estimate (rad/sample). 0.0 if the run is degenerate
        (length < 2 or all weights zero).
    """
    L = phi.size
    if L < 2:
        return 0.0
    w_total = float(w.sum())
    if w_total <= 0.0:
        return 0.0

    # --- Step A: weighted Kay's lag-1 slope -------------------------------
    v = np.sqrt(w[:-1] * w[1:])
    if v.sum() <= 0.0:
        return 0.0
    diff_phasor = np.exp(1j * (phi[1:] - phi[:-1]))
    b = float(np.angle((v * diff_phasor).sum()))

    if n_iter_gn <= 0:
        return b

    # Centred sample index — at the weighted centroid, slope and intercept decouple in the Newton Hessian, which is numerically much better behaved than the raw (a, b) parameterisation.
    n = np.arange(L, dtype=np.float64)
    n_bar = float((w * n).sum() / w_total)
    n_c = n - n_bar

    # Initial intercept at the centred origin from the weighted phasor centroid evaluated at the current slope estimate.
    a_c = float(np.angle((w * np.exp(1j * (phi - b * n_c))).sum()))

    # --- Step B: Newton on J(a_c, b) = Σ w·(1 − cos(phi − a_c − b·n_c)) ---
    for _ in range(n_iter_gn):
        r = phi - a_c - b * n_c
        c = np.cos(r)
        s = np.sin(r)
        A  = float((w * c).sum())
        B  = float((w * n_c * c).sum())
        D  = float((w * n_c * n_c * c).sum())
        ga = float((w * s).sum())
        gb = float((w * n_c * s).sum())
        det = A * D - B * B
        # Bail out if the Hessian is not safely positive definite (can happen far from optimum; here we trust the Kay init enough that this is the only safety net we need).
        if A <= 0.0 or det <= 1e-12 * (abs(A) * abs(D) + 1e-30):
            break
        da = ( D * ga - B * gb) / det
        db = (-B * ga + A * gb) / det
        a_c += da
        b   += db

    return b


def _phasor_ls_slope_intercept(
    phi: np.ndarray,
    w: np.ndarray,
    n_iter_gn: int = 4,
) -> tuple[float, float]:
    """Wrap-immune weighted phasor LS that returns BOTH slope and intercept.

    Thin wrapper on :func:`_phasor_ls_slope` (which already does the
    wrap-immune optimisation but only returns the slope) that recovers
    the matching intercept at ``u = 0`` via the wrap-aware weighted
    circular mean

        intercept = arg( Σ_i w_i · exp(j · (phi_i - slope · i)) )    (∈ (-π, π])

    Rows can be excluded from the fit by setting their entry of ``w`` to
    zero — Kay's lag-1 estimator and the Newton step both ignore
    zero-weighted samples cleanly.

    Returns
    -------
    slope : float
        rad / sample; ``nan`` if degenerate (fewer than two non-zero
        weights, no consecutive non-zero pair for Kay's init, or the
        Newton step left a non-finite value).
    intercept : float
        rad, ∈ (-π, π]; ``nan`` if the fit was degenerate.
    """
    L = int(phi.size)
    if L < 2 or w.size != L:
        return float("nan"), float("nan")
    w_total = float(w.sum())
    if w_total <= 0.0:
        return float("nan"), float("nan")
    # Kay's init needs at least one consecutive pair with both weights > 0.
    if float(np.sqrt(w[:-1] * w[1:]).sum()) <= 0.0:
        return float("nan"), float("nan")
    slope = float(_phasor_ls_slope(phi, w, n_iter_gn=n_iter_gn))
    n = np.arange(L, dtype=np.float64)
    intercept = float(
        np.angle((w * np.exp(1j * (phi - slope * n))).sum())
    )
    if not (np.isfinite(slope) and np.isfinite(intercept)):
        return float("nan"), float("nan")
    return slope, intercept


def _min_distance_line_fit(
    phi: np.ndarray,
    u: np.ndarray,
    coh: np.ndarray,
    min_row_coherence: float = 0.5,
    inlier_tol_rad: float = 0.5,
    slope_max: float = 0.2,
    n_slope: int = 1001,
) -> tuple[float, float, int]:
    """Wrap-aware brute-force line fit by minimum wrapped-distance cost.

    For a wrapped phase trace ``phi(u)`` (sample indices ``u``), fits
    ``phi(u) ≈ slope·u + intercept`` by trying ``n_slope`` candidate
    slopes on an equispaced grid ``[-slope_max, +slope_max]`` and
    picking the candidate whose **wrapped** line is closest to the
    data — i.e. exactly the procedure asked for: "try 1000 different
    lines with incremental slopes, wrap if needed, find the Euclidean
    distance, and plot the lowest".

    Algorithm (fully vectorised, no Newton, no LS refit, no unwrap):

      1. For every candidate slope ``s``, the optimal intercept is the
         coh-weighted **circular** mean of ``phi - s·u`` evaluated at
         ``u = 0``::

             intercept(s) = arg( Σ_i coh[i] · exp(j·(phi[i] - s·u[i])) )

         This is the closed-form ``arg min_b Σ coh[i]·(1-cos(phi-s·u-b))``
         and is wrap-immune by construction.
      2. The per-sample wrapped distance to the candidate line is

             r[i] = arg( exp(j·(phi[i] - s·u[i] - intercept(s))) )
                  ∈ (-π, π]

         The candidate's **score** is the coh-weighted mean of
         ``|r[i]|`` over high-coh rows (rows with ``coh > min_row_coherence``;
         falls back to all rows if too few qualify)::

             score(s) = Σ_high coh[i]·|r[i]| / Σ_high coh[i]

         This is the "Euclidean wrapped distance" cost the user asked
         for, with the trivial improvement of weighting by coherence so
         pure-noise rows do not pull the optimum. It is robust to
         outliers (each row's contribution is bounded by π) and has no
         local-minimum pathology because we evaluate every candidate
         on the grid.
      3. The grid argmin is returned **directly** — no LS refit. The
         default grid step is ``2·slope_max / (n_slope-1) ≈ 4·10⁻⁴
         rad/sample`` which over a 1000-row patch resolves ``slope·n``
         to ~0.4 rad — well below ``2π``.

    Parameters
    ----------
    phi : (n,) float ndarray
        Wrapped phase samples in (-π, π].
    u : (n,) float ndarray
        Sample indices (e.g. ``np.arange(n)``).
    coh : (n,) float ndarray
        Non-negative per-sample weight in [0, 1] — the same per-row
        coherence ``compute_box_phase_estimates`` produces.
    min_row_coherence : float
        High-coh gate; only rows with ``coh > min_row_coherence``
        contribute to the score. If fewer than 2 rows qualify, the
        gate is lifted (all rows contribute). Default 0.5.
    inlier_tol_rad : float
        Diagnostic tolerance (rad) used **only** to report how many
        high-coh rows lie within ``±inlier_tol_rad`` of the chosen
        line. Does NOT affect slope selection. Default 0.5 (≈ ±29°).
    slope_max : float
        Half-width of the slope search range (rad / sample).
        Default 0.2.
    n_slope : int
        Number of equispaced slope candidates. Default 1001.

    Returns
    -------
    slope : float
        Grid-argmin slope (rad / sample); ``nan`` on degenerate input.
    intercept : float
        Optimal intercept at ``u = 0`` for that slope, in (-π, π];
        ``nan`` on degenerate input.
    n_inliers : int
        Diagnostic: number of high-coh rows within ``inlier_tol_rad``
        of the chosen line. 0 on degenerate input.
    """
    n = int(phi.size)
    if n < 2 or u.size != n or coh.size != n:
        return float("nan"), float("nan"), 0
    # Drop rows with no valid measurement (NaN phi / coh — e.g. mask-out rows introduced by the detection-mask factor in compute_box_phase_estimates). The line fit is then run on the finite subset; the returned slope / intercept are still in the original ``u`` frame because we index ``u`` alongside ``phi`` / ``coh``.
    finite = np.isfinite(phi) & np.isfinite(coh)
    if int(finite.sum()) < 2:
        return float("nan"), float("nan"), 0
    if not finite.all():
        phi = phi[finite]
        u = u[finite]
        coh = coh[finite]
        n = int(phi.size)
    if float(coh.sum()) <= 0.0:
        return float("nan"), float("nan"), 0

    high_coh_mask = coh > min_row_coherence
    # Fall back to all rows if the coh gate is too aggressive for this box; otherwise the score would be undefined or dominated by ≤1 row.
    if int(high_coh_mask.sum()) < 2:
        high_coh_mask = np.ones(n, dtype=bool)

    slopes = np.linspace(-slope_max, slope_max, n_slope)
    # phi - slope·u, broadcast to (n_slope, n)
    phi_minus_su = phi[None, :] - slopes[:, None] * u[None, :]
    # Optimal intercept per slope: wrap-aware coh-weighted circular mean.
    z = (coh[None, :] * np.exp(1j * phi_minus_su)).sum(axis=1)
    intercepts = np.angle(z)
    # Wrapped distance from every sample to every candidate line.
    r = np.angle(np.exp(1j * (phi_minus_su - intercepts[:, None])))
    abs_r = np.abs(r)

    # Cost: coh-weighted mean |wrapped r| over high-coh rows         = Σ_high coh[i]·|r[i]| / Σ_high coh[i]
    w_row = np.where(high_coh_mask, coh, 0.0)
    denom = float(w_row.sum())
    if denom <= 0.0:
        return float("nan"), float("nan"), 0
    scores = (w_row[None, :] * abs_r).sum(axis=1) / denom
    best = int(np.argmin(scores))
    s_best = float(slopes[best])
    b_best = float(intercepts[best])
    # Diagnostic inlier count for plot titles / CSV; not used for selection.
    n_in_best = int(((abs_r[best] <= inlier_tol_rad) & high_coh_mask).sum())
    return s_best, b_best, n_in_best


def estimate_phase_slope(
    d_phase: np.ndarray,
    window_length: int,
    weights: np.ndarray | None = None,
    n_iter_gn: int = 2,
) -> np.ndarray:
    """
    Per-range-column slope of `d_phase` over runs of non-zero samples in
    azimuth, using a weighted phasor LS fit that is wrap-immune by
    construction.

    For each range column `r`:
      1. Find maximal runs of indices where `d_phase[:, r] != 0`.
      2. Discard runs shorter than `0.5 * window_length` azimuth pixels.
      3. On every kept run, fit  d_phase[n] ≈ a + b·n  (mod 2π) by
         minimising  Σ w[n]·(1 − cos(d_phase[n] − a − b·n))  via
         `_phasor_ls_slope`. The slope `b` (rad/pixel) is written into
         the output map at the run's positions; pixels not in any kept
         run remain 0.

    The fit is initialised by Kay's weighted lag-1 estimator and then
    refined with a couple of Newton steps on the weighted cosine cost.
    Both stages depend on the wrapped phase only through sin/cos, so a
    real ±2π wrap inside the run produces the same contribution as no
    wrap — no `np.unwrap` is used and isolated noise excursions across
    ±π cannot inject spurious 2π/L slope bias.

    Parameters
    ----------
    d_phase : (N_az, N_rg) float ndarray
        Wrapped azimuth phase derivative (rad). Zeros mark masked-out
        pixels and define the run boundaries.
    window_length : int
        Minimum kept run length is ⌊0.5·window_length⌋.
    weights : (N_az, N_rg) float ndarray, optional
        Per-pixel non-negative weight for the fit. For
            d_phase = arg(s_degraded[1:] · conj(s_degraded[:-1]))
        the natural choice is
            weights = |s_degraded[:-1]| * |s_degraded[1:]|     (same shape)
        i.e. the modulus of the complex product whose argument was
        taken. Multiplying by an amplitude / coherence mask, or zeroing
        out samples below a threshold, lets the caller keep low-SNR
        pixels out of the fit without changing this routine. If None,
        all kept samples receive equal weight (still wrap-immune).
    n_iter_gn : int
        Number of Newton refinement iterations after Kay's init.
        Default 2. Set to 0 to use Kay's estimator alone.

    Returns
    -------
    slope_map : (N_az, N_rg) float ndarray
        Slope `b` in rad/pixel at every position belonging to a kept
        run; 0 elsewhere.

    Notes
    -----
    * "Non-zero" is interpreted strictly: the upstream CoV gate produces
      exact zeros at masked pixels, which propagate into d_phase via
      arg(0 · conj(·)) = 0. The run boundaries follow that mask.
    * The (unresolved) global 2π ambiguity in `b` is the same as in any
      single-lag Doppler estimator: if a target really has |b·L| > 2π
      the estimate folds. Use a shorter degradation factor / smaller
      kernel if that regime matters.
    """
    n_az, n_rg = d_phase.shape
    if weights is not None:
        assert weights.shape == d_phase.shape, (weights.shape, d_phase.shape)
        w_full = np.asarray(weights, dtype=np.float64)
    else:
        w_full = None
    min_len = max(2, int(0.5 * window_length))
    slope_map = np.zeros_like(d_phase, dtype=float)

    n_runs_total = 0
    n_runs_kept = 0
    for r in range(n_rg):
        col = d_phase[:, r]
        nz = col != 0
        if not nz.any():
            continue
        # Run boundaries: indices where nz transitions, plus the array ends.
        change = np.flatnonzero(np.diff(nz.astype(np.int8))) + 1
        edges = np.concatenate([[0], change, [n_az]])
        for s, e in zip(edges[:-1], edges[1:]):
            if not nz[s]:
                continue              # this segment is zeros
            n_runs_total += 1
            if (e - s) < min_len:
                continue
            phi = col[s:e].astype(np.float64, copy=False)
            if w_full is not None:
                w = w_full[s:e, r]
            else:
                w = np.ones(e - s, dtype=np.float64)
            slope = _phasor_ls_slope(phi, w, n_iter_gn=n_iter_gn)
            slope_map[s:e, r] = slope
            n_runs_kept += 1

    print(
        f"  estimate_phase_slope: d_phase {d_phase.shape}, "
        f"window_length={window_length}, min_len={min_len}, "
        f"runs kept {n_runs_kept}/{n_runs_total}, "
        f"weighted={'yes' if w_full is not None else 'no'}, "
        f"n_iter_gn={n_iter_gn}"
    )
    return slope_map
    

def compute_box_slope_col_total(
    boxes_yxhw: np.ndarray,
    slope_map: np.ndarray,
) -> np.ndarray:
    """Per-box aggregate of `slope_map` along each range column.

    For each box, for each range column the box covers, find the
    longest contiguous non-zero run of ``slope_map`` inside the box's
    azimuth row range, and compute ``|slope[run_start] · run_length|``.
    The per-box score is the **maximum** of that quantity over the
    box's columns.

    Rationale: a large moving target whose single-line wrap-aware fit
    (`_min_distance_line_fit`) reports a below-threshold
    ``|slope · n_rows|`` because the box straddles a phase wrap or
    contains heterogeneous scatterers can still have one or more
    columns with a clean, wrap-immune phasor-LS slope (`slope_map`)
    above threshold. This aggregate captures that.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int ndarray
        Box centres + sizes ``[y_c, x_c, h, w]`` in `s_degraded`
        coordinates (axis 0 = azimuth row, axis 1 = range column).
    slope_map : (N_az - 1, N_rg) float ndarray
        Per-pixel slope (rad/row) from :func:`estimate_phase_slope`;
        zero outside any kept column-run.

    Returns
    -------
    slope_col_total_rad : (K,) float ndarray
        ``max_c |slope_c · L_c|`` over the box's covered columns,
        where ``L_c`` is the length of the longest non-zero run in
        column ``c`` inside the box's azimuth range and ``slope_c``
        is the (constant) slope on that run.
    """
    K = len(boxes_yxhw)
    out = np.zeros(K, dtype=np.float64)
    if K == 0 or slope_map.size == 0:
        return out
    n_az, n_rg = slope_map.shape
    for i, (y_c, x_c, h, w) in enumerate(np.atleast_2d(boxes_yxhw)):
        y0 = max(0, int(y_c) - int(h) // 2)
        y1 = min(n_az, y0 + int(h))
        x0 = max(0, int(x_c) - int(w) // 2)
        x1 = min(n_rg, x0 + int(w))
        if y1 <= y0 or x1 <= x0:
            continue
        best = 0.0
        for c in range(x0, x1):
            col = slope_map[y0:y1, c]
            nz = col != 0
            if not nz.any():
                continue
            change = np.flatnonzero(np.diff(nz.astype(np.int8))) + 1
            edges = np.concatenate([[0], change, [len(col)]])
            best_col = 0.0
            for s, e in zip(edges[:-1], edges[1:]):
                if not nz[s]:
                    continue
                L = e - s
                score = abs(float(col[s])) * L
                if score > best_col:
                    best_col = score
            if best_col > best:
                best = best_col
        out[i] = best
    return out


# --------------------------------------------------------------------------- Display helpers ---------------------------------------------------------------------------

def _overlay_boxes(
    ax: plt.Axes,
    boxes_yxhw: np.ndarray,
    color: str = "cyan",
    lw: float = 1.4,
    slopes: np.ndarray | None = None,
    slope_vmin: float = -15.0,
    slope_vmax: float = 15.0,
    slope_cmap: str = "RdBu_r",
    signs: np.ndarray | None = None,
    sign_pos_color: str = "red",
    sign_neg_color: str = "yellow",
    halo: bool = True,
    halo_color: str = "black",
    halo_extra_lw: float | None = None,
    halo_alpha: float = 0.35,
) -> None:
    """Draw axis-aligned rectangles, one per row of `boxes_yxhw`.

    boxes_yxhw : (K, 4) array of [(y_centre, x_centre, h, w), …] in pixel
                 coords (axis 0 = azimuth row, axis 1 = range column).

    When ``slopes`` is given (length-K array, e.g. per-box slope · n_rows
    in rad), each rectangle is coloured by that value on ``slope_cmap``,
    clipped to ``[slope_vmin, slope_vmax]`` (values outside saturate).
    NaN slopes fall back to ``color``.

    When ``signs`` is given (length-K array, e.g. per-box
    ``best_deviation``), each rectangle is coloured ``sign_pos_color`` if
    the value is strictly positive and ``sign_neg_color`` if strictly
    negative. NaN / zero values fall back to ``color``. ``signs`` takes
    precedence over ``slopes`` when both are provided.

    When ``halo`` is true (default), each rectangle is stroked with a
    wider ``halo_color`` outline underneath so the box stays visible on
    both bright and dark image regions. ``halo_extra_lw`` defaults to
    ``max(0.3, lw * 0.4)``: half of it peeks outside the coloured line
    and half bleeds inside the box, so we keep both numbers small on top
    of the low default alpha to avoid a visible interior black band.
    """
    boxes = np.atleast_2d(boxes_yxhw)
    use_signs = signs is not None
    if use_signs:
        signs = np.asarray(signs, dtype=np.float64).ravel()
    elif slopes is not None:
        slopes = np.asarray(slopes, dtype=np.float64).ravel()
        cmap_obj = plt.get_cmap(slope_cmap)
        norm = matplotlib.colors.Normalize(
            vmin=slope_vmin, vmax=slope_vmax, clip=True,
        )
    effects = None
    if halo:
        extra = (
            halo_extra_lw if halo_extra_lw is not None
            else max(0.3, lw * 0.4)
        )
        effects = [
            matplotlib.patheffects.withStroke(
                linewidth=lw + extra,
                foreground=halo_color,
                alpha=halo_alpha,
            ),
        ]
    for i, (y, x, h, w) in enumerate(boxes):
        if use_signs and i < signs.size and np.isfinite(signs[i]):
            v = float(signs[i])
            if v > 0.0:
                edge = sign_pos_color
            elif v < 0.0:
                edge = sign_neg_color
            else:
                edge = color
        elif (
            not use_signs
            and slopes is not None
            and i < slopes.size
            and np.isfinite(slopes[i])
        ):
            edge = cmap_obj(norm(float(slopes[i])))
        else:
            edge = color
        rect = plt.Rectangle(
            (x - w / 2 - 0.5, y - h / 2 - 0.5), w, h,
            fill=False, edgecolor=edge, linewidth=lw,
            zorder=3,
        )
        if effects is not None:
            rect.set_path_effects(effects)
        ax.add_patch(rect)


def merge_overlapping_boxes(boxes_yxhw: np.ndarray) -> np.ndarray:
    """Merge any axis-aligned boxes whose rectangles intersect.

    Builds a graph where i and j are linked iff their boxes overlap by
    even one pixel (open-interval intersection > 0), then replaces each
    connected component with the union bounding box of its members.

    Useful as a final pass after IoU-NMS / distance-NMS / bridge-merge:
    earlier stages can leave touching-or-overlapping boxes (IoU > 0 but
    ≤ iou_thresh, or a bridge-merged union now overlapping a previously
    untouched neighbour).
    """
    n = len(boxes_yxhw)
    if n < 2:
        return boxes_yxhw.copy()

    yc = boxes_yxhw[:, 0].astype(np.int64)
    xc = boxes_yxhw[:, 1].astype(np.int64)
    h = boxes_yxhw[:, 2].astype(np.int64)
    w = boxes_yxhw[:, 3].astype(np.int64)
    y_lo = yc - h // 2
    y_hi = y_lo + h
    x_lo = xc - w // 2
    x_hi = x_lo + w

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            if (y_lo[i] < y_hi[j] and y_lo[j] < y_hi[i]
                    and x_lo[i] < x_hi[j] and x_lo[j] < x_hi[i]):
                union(i, j)

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    out = []
    for members in comps.values():
        if len(members) == 1:
            out.append(boxes_yxhw[members[0]])
            continue
        ys_lo = int(y_lo[members].min())
        ys_hi = int(y_hi[members].max())
        xs_lo = int(x_lo[members].min())
        xs_hi = int(x_hi[members].max())
        out.append(np.array([
            (ys_lo + ys_hi) // 2,
            (xs_lo + xs_hi) // 2,
            ys_hi - ys_lo,
            xs_hi - xs_lo,
        ], dtype=np.int64))
    return np.array(out, dtype=np.int64)


def merge_by_strip_amplitude(
    boxes_yxhw: np.ndarray,
    amp_raw: np.ndarray,
    az_spacing_m: float,
    rg_spacing_m: float,
    bridge_strength: float = 2.0,
    max_rg_offset_m: float = 20.0,
    max_az_gap_m: float = 2000.0,
) -> np.ndarray:
    """Bridge-merge boxes joined by a high-amplitude trail in `amp_raw`.

    Designed to merge multiple boxes that fall on the same physical
    moving target whose azimuth signature has been split by gaps in the
    CoV mask, but whose unthresholded amplitude is continuous.

    Two boxes (i, j) are linked iff:
      • |Δrg|·rg_spacing_m ≤ `max_rg_offset_m`
        (i.e. they share a range column to within one target's width),
      • |Δaz|·az_spacing_m ≤ `max_az_gap_m`,
      • median amplitude of the strip joining them along azimuth in
        `amp_raw` is at least
        `bridge_strength × max(amp_raw[centre_i], amp_raw[centre_j])`.

    Why max + median? `max` means the strip must be a comparable fraction
    of the *brighter* endpoint, so a faint box can't pull a bright box
    into a merge by anchoring on a dim trail. `median` means at least
    half the strip pixels clear the threshold — a single bright outlier
    can't carry the bridge the way a mean would.

    Connected components in this link graph are replaced with their
    union bounding box. Singletons pass through unchanged. The strip
    spans the *azimuth gap* between the two boxes' inner edges, with
    range columns set to the union of both boxes' range extents.
    """
    n = len(boxes_yxhw)
    if n < 2:
        return boxes_yxhw.copy()

    H, W = amp_raw.shape
    yc = boxes_yxhw[:, 0].astype(np.int64)
    xc = boxes_yxhw[:, 1].astype(np.int64)
    h = boxes_yxhw[:, 2].astype(np.int64)
    w = boxes_yxhw[:, 3].astype(np.int64)
    centre_amp = amp_raw[np.clip(yc, 0, H - 1), np.clip(xc, 0, W - 1)]

    y_lo_arr = np.clip(yc - h // 2, 0, H)
    y_hi_arr = np.clip(y_lo_arr + h, 0, H)
    x_lo_arr = np.clip(xc - w // 2, 0, W)
    x_hi_arr = np.clip(x_lo_arr + w, 0, W)

    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    n_links = 0
    for i in range(n):
        for j in range(i + 1, n):
            d_rg_m = abs(int(xc[i]) - int(xc[j])) * rg_spacing_m
            if d_rg_m > max_rg_offset_m:
                continue
            d_az_m = abs(int(yc[i]) - int(yc[j])) * az_spacing_m
            if d_az_m > max_az_gap_m:
                continue
            # Strip = azimuth gap between the two boxes' inner edges.
            strip_top = min(int(y_hi_arr[i]), int(y_hi_arr[j]))
            strip_bot = max(int(y_lo_arr[i]), int(y_lo_arr[j]))
            if strip_top >= strip_bot:
                # Boxes overlap in azimuth — IoU NMS already kept both, so they're not redundant by overlap. Skip; let other pairs link them transitively if they really belong together.
                continue
            sx_lo = min(int(x_lo_arr[i]), int(x_lo_arr[j]))
            sx_hi = max(int(x_hi_arr[i]), int(x_hi_arr[j]))
            strip = amp_raw[strip_top:strip_bot, sx_lo:sx_hi]
            if strip.size == 0:
                continue
            anchor = bridge_strength * max(float(centre_amp[i]),
                                           float(centre_amp[j]))
            if float(np.median(strip)) > anchor:
                union(i, j)
                n_links += 1

    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)

    out = []
    for members in comps.values():
        if len(members) == 1:
            out.append(boxes_yxhw[members[0]])
            continue
        ys_lo = int(y_lo_arr[members].min())
        ys_hi = int(y_hi_arr[members].max())
        xs_lo = int(x_lo_arr[members].min())
        xs_hi = int(x_hi_arr[members].max())
        out.append(np.array([
            (ys_lo + ys_hi) // 2,
            (xs_lo + xs_hi) // 2,
            ys_hi - ys_lo,
            xs_hi - xs_lo,
        ], dtype=np.int64))
    return np.array(out, dtype=np.int64)


def nms_by_centre_distance(
    boxes_yxhw: np.ndarray,
    scores: np.ndarray,
    az_spacing_m: float,
    rg_spacing_m: float,
    max_dist_az_m: float,
    max_dist_rg_m: float,
) -> np.ndarray:
    """Greedy centre-distance suppression in *metres*, anisotropic ellipse.

    Walk boxes from highest score down. Keep each one and suppress every
    later box whose centre lies inside the axis-aligned ellipse

        (Δaz_m / max_dist_az_m)^2 + (Δrg_m / max_dist_rg_m)^2 < 1

    around any kept centre. The ellipse is the right shape for this
    domain: a SAR moving target's image signature is long along azimuth
    (Doppler smear) but stays near-pointlike in range, so legitimate
    "same target → multiple boxes" duplicates have small Δrg even when
    Δaz is large, while genuinely-distinct targets are typically well
    separated in range. A single circular radius can't capture both.

    Use `max_dist_az_m` ≈ a few × max target length (e.g. 1000 m) and
    `max_dist_rg_m` ≈ max target width (e.g. 20 m).

    Returns indices into the original `boxes_yxhw` (score-descending).
    """
    if len(boxes_yxhw) == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(-scores)
    yxhw = boxes_yxhw[order]
    suppressed = np.zeros(len(yxhw), dtype=bool)
    keep = []
    inv_az_sq = 1.0 / float(max_dist_az_m) ** 2
    inv_rg_sq = 1.0 / float(max_dist_rg_m) ** 2
    for i in range(len(yxhw)):
        if suppressed[i]:
            continue
        keep.append(i)
        if i + 1 == len(yxhw):
            continue
        dy_m = (yxhw[i + 1:, 0] - yxhw[i, 0]) * az_spacing_m
        dx_m = (yxhw[i + 1:, 1] - yxhw[i, 1]) * rg_spacing_m
        norm = dy_m * dy_m * inv_az_sq + dx_m * dx_m * inv_rg_sq
        suppressed[i + 1:][norm < 1.0] = True
    return order[np.asarray(keep, dtype=np.int64)]


def nms_boxes(
    boxes_yxhw: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.3,
) -> np.ndarray:
    """Greedy non-maximum suppression of axis-aligned boxes.

    Walk boxes in descending `scores`. Keep each one and suppress every
    later box whose IoU with the kept one exceeds `iou_thresh`.

    Parameters
    ----------
    boxes_yxhw : (K, 4) array of (y_centre, x_centre, h, w).
    scores     : (K,)   per-box score (higher = better, kept first).
    iou_thresh : float  IoU above which a later box is dropped.

    Returns
    -------
    keep_idx : 1-D int array of indices into the *original* `boxes_yxhw`
               (in score-descending order).
    """
    if len(boxes_yxhw) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-scores)
    boxes = boxes_yxhw[order].astype(np.float64)
    y_c, x_c, h, w = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    y_lo, y_hi = y_c - h / 2, y_c + h / 2
    x_lo, x_hi = x_c - w / 2, x_c + w / 2
    areas = h * w

    suppressed = np.zeros(len(boxes), dtype=bool)
    keep = []
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        # Compute IoU of box i vs every still-alive box j > i.
        j = np.flatnonzero(~suppressed)
        j = j[j > i]
        if not j.size:
            continue
        oy = np.maximum(0.0, np.minimum(y_hi[i], y_hi[j]) - np.maximum(y_lo[i], y_lo[j]))
        ox = np.maximum(0.0, np.minimum(x_hi[i], x_hi[j]) - np.maximum(x_lo[i], x_lo[j]))
        inter = oy * ox
        iou = inter / np.maximum(areas[i] + areas[j] - inter, 1e-12)
        suppressed[j[iou > iou_thresh]] = True

    return order[np.asarray(keep, dtype=np.int64)]


def grow_and_recenter_boxes(
    peaks_yx: np.ndarray,
    mask: np.ndarray,
    amp: np.ndarray,
    initial_hw: tuple[int, int],
    az_step: int = 5,
    rg_step: int = 1,
    density_threshold: float = 0.25,
    az_tail: int = 20,
    rg_tail: int = 1,
    az_rescue_lookahead: int = 50,
    az_rescue_lookback: int = 50,
    az_rescue_threshold: float = 0.5,
    max_h: int | None = None,
    max_w: int | None = None,
    max_iter: int = 2000,
    boundary_box: np.ndarray | None = None,
    growth_sequence: str = "combined",
) -> np.ndarray:
    """Grow each peak's box per-edge — azimuth first, then range — and
    recenter on the amplitude centre-of-mass each step.

    Box state is tracked as edges (y_lo, y_hi, x_lo, x_hi); each of the
    four edges can expand independently. The box's centre is just the
    midpoint of its current bounds.

    Algorithm (per peak):
      1. Start with a box of size `initial_hw` centred on the peak.
      2. Each iteration (two-phase):
         (Phase 1 — azimuth)
         a. Try shifting the top edge:    y_lo ← y_lo - az_step.
         b. Try shifting the bottom edge: y_hi ← y_hi + az_step.
         (Phase 2 — range, only once both top AND bottom are dead)
         c. Try shifting the left edge:   x_lo ← x_lo - rg_step.
         d. Try shifting the right edge:  x_hi ← x_hi + rg_step.
            Each proposal is accepted iff the (image-clipped) box still
            has `mask` density ≥ `density_threshold`. A failed edge is
            "killed" — not retried in subsequent iterations.
         e. If no edge grew this iteration, stop.
         f. Recenter: shift bounds so the box (size h × w preserved) is
            centred on the amplitude-weighted CoM inside it.
         g. If post-recentre density falls below `density_threshold`, stop.

    Phase ordering (azimuth-first) keeps the box thin in range while
    azimuth grows, so the azimuth density check isn't diluted by a wide
    range strip that may contain partial-target columns.

    Returns
    -------
    boxes_yxhw : (K, 4) int array — final (y_c, x_c, h, w) per peak.
    """
    H, W = mask.shape
    h0, w0 = initial_hw
    out = np.zeros((len(peaks_yx), 4), dtype=np.int64)

    def _density_ok(yl, yh, xl, xh, threshold=None):
        """True iff `mask[yl:yh, xl:xh]` density ≥ threshold.

        Used for the per-edge GROWTH test (strip-only), the post-recentre
        FULL-BOX test, and the wider rescue look-ahead test (with a higher
        threshold). When `threshold` is None, falls back to the function's
        global `density_threshold` parameter.
        """
        sub = mask[yl:yh, xl:xh]
        if sub.size == 0:
            return False
        th = density_threshold if threshold is None else threshold
        return float(sub.sum()) >= th * sub.size

    def _strip_has_peak(yl, yh, xl, xh) -> bool:
        """True iff the region `boundary_box[yl:yh, xl:xh]` contains at
        least one seed peak (bit set to 1.0). Used as an OR partner
        to the density gate at every edge-growth test: if the newly-
        added region itself is a `boundary_box` seed, the strip is
        accepted unconditionally — a seed peak inside means the
        peak-picker independently identified a dense-detection cluster
        there, so the target's signature clearly continues into that
        strip regardless of the aggregate density.
        Always returns False when `boundary_box` is None (feature off).
        """
        if boundary_box is None or yh <= yl or xh <= xl:
            return False
        return bool(boundary_box[yl:yh, xl:xh].any())

    # Per-edge helpers (top / bottom / left / right). Each takes the current bounds, returns (new_bound_value, grew). The az helpers apply the primary density test with tail + boundary_box OR partner, and fall back to the rescue look-ahead. The rg helpers use the same OR partner but no rescue (rg dropouts are handled by az retries instead).
    def _try_top(y_lo, y_hi, x_lo, x_hi):
        ny_lo = y_lo - az_step
        new_h = y_hi - ny_lo
        if (max_h is not None and new_h > max_h) or ny_lo < 0:
            return y_lo, False
        tail_hi = min(y_lo + az_tail, y_hi)
        if (
            _density_ok(ny_lo, tail_hi, x_lo, x_hi)
            or _strip_has_peak(ny_lo, y_lo, x_lo, x_hi)
        ):
            return ny_lo, True
        look_lo = max(y_lo - az_rescue_lookahead, 0)
        tail_hi_resc = min(y_lo + az_rescue_lookback, y_hi)
        if _density_ok(look_lo, tail_hi_resc, x_lo, x_hi,
                       threshold=az_rescue_threshold):
            return ny_lo, True
        return y_lo, False

    def _try_bot(y_lo, y_hi, x_lo, x_hi):
        ny_hi = y_hi + az_step
        new_h = ny_hi - y_lo
        if (max_h is not None and new_h > max_h) or ny_hi > H:
            return y_hi, False
        tail_lo = max(y_hi - az_tail, y_lo)
        if (
            _density_ok(tail_lo, ny_hi, x_lo, x_hi)
            or _strip_has_peak(y_hi, ny_hi, x_lo, x_hi)
        ):
            return ny_hi, True
        look_hi = min(y_hi + az_rescue_lookahead, H)
        tail_lo_resc = max(y_hi - az_rescue_lookback, y_lo)
        if _density_ok(tail_lo_resc, look_hi, x_lo, x_hi,
                       threshold=az_rescue_threshold):
            return ny_hi, True
        return y_hi, False

    def _try_left(y_lo, y_hi, x_lo, x_hi):
        nx_lo = x_lo - rg_step
        new_w = x_hi - nx_lo
        if (max_w is not None and new_w > max_w) or nx_lo < 0:
            return x_lo, False
        tail_hi_x = min(x_lo + rg_tail, x_hi)
        if (
            _density_ok(y_lo, y_hi, nx_lo, tail_hi_x)
            or _strip_has_peak(y_lo, y_hi, nx_lo, x_lo)
        ):
            return nx_lo, True
        return x_lo, False

    def _try_right(y_lo, y_hi, x_lo, x_hi):
        nx_hi = x_hi + rg_step
        new_w = nx_hi - x_lo
        if (max_w is not None and new_w > max_w) or nx_hi > W:
            return x_hi, False
        tail_lo_x = max(x_hi - rg_tail, x_lo)
        if (
            _density_ok(y_lo, y_hi, tail_lo_x, nx_hi)
            or _strip_has_peak(y_lo, y_hi, x_hi, nx_hi)
        ):
            return nx_hi, True
        return x_hi, False

    def _recenter_box(y_lo, y_hi, x_lo, x_hi):
        """Recenter the box on the amplitude CoM, preserving current h × w."""
        h_cur = y_hi - y_lo
        w_cur = x_hi - x_lo
        sub_amp = amp[y_lo:y_hi, x_lo:x_hi]
        total = float(sub_amp.sum())
        if total > 0:
            row_marg = sub_amp.sum(axis=1)
            col_marg = sub_amp.sum(axis=0)
            yy = np.arange(sub_amp.shape[0])
            xx = np.arange(sub_amp.shape[1])
            dy = float((row_marg * yy).sum() / total)
            dx = float((col_marg * xx).sum() / total)
            y_com = y_lo + int(round(dy))
            x_com = x_lo + int(round(dx))
            y_lo = max(0, y_com - h_cur // 2)
            y_hi = min(H, y_lo + h_cur)
            y_lo = max(0, y_hi - h_cur)
            x_lo = max(0, x_com - w_cur // 2)
            x_hi = min(W, x_lo + w_cur)
            x_lo = max(0, x_hi - w_cur)
        return y_lo, y_hi, x_lo, x_hi

    def _run_axis_pass(y_lo, y_hi, x_lo, x_hi, axis):
        """Iterate one axis (`'az'` or `'rg'`) to convergence: at each
        iter, try both edges of that axis; if neither grew, stop.
        After every accepted grow-step recentre and re-check full-box
        density — the pass reports (bounds, dead) where dead=True
        means density fell below `density_threshold` at the recentre
        step and the whole growth should terminate.
        """
        for _ in range(max_iter):
            if axis == "az":
                y_lo, g_top = _try_top(y_lo, y_hi, x_lo, x_hi)
                y_hi, g_bot = _try_bot(y_lo, y_hi, x_lo, x_hi)
                grew = g_top or g_bot
            else:
                x_lo, g_lef = _try_left(y_lo, y_hi, x_lo, x_hi)
                x_hi, g_rig = _try_right(y_lo, y_hi, x_lo, x_hi)
                grew = g_lef or g_rig
            if not grew:
                return (y_lo, y_hi, x_lo, x_hi), False
            y_lo, y_hi, x_lo, x_hi = _recenter_box(y_lo, y_hi, x_lo, x_hi)
            if not _density_ok(y_lo, y_hi, x_lo, x_hi):
                return (y_lo, y_hi, x_lo, x_hi), True
        return (y_lo, y_hi, x_lo, x_hi), False

    for k, (y0, x0) in enumerate(peaks_yx):
        # Initial bounds centred on the seed peak, clipped to the image.
        y_lo = max(0, int(y0) - h0 // 2)
        y_hi = min(H, y_lo + h0)
        x_lo = max(0, int(x0) - w0 // 2)
        x_hi = min(W, x_lo + w0)

        # Sequential mode: five back-to-back per-axis convergence passes (az → rg → az → rg → az). Each pass iterates its own axis's two edges to convergence, recentring and re-checking full-box density after every accepted step. A density drop at the recentre step kills the whole growth (dead=True). This mode gives the target multiple chances to recover azimuth extent that only becomes reachable after the box has widened in range, and vice versa.
        if growth_sequence == "sequential":
            for axis in ("az", "rg", "az", "rg", "az"):
                (y_lo, y_hi, x_lo, x_hi), dead = _run_axis_pass(
                    y_lo, y_hi, x_lo, x_hi, axis,
                )
                if dead:
                    break

            h_final = y_hi - y_lo
            w_final = x_hi - x_lo
            y_c = (y_lo + y_hi) // 2
            x_c = (x_lo + x_hi) // 2
            out[k] = (y_c, x_c, h_final, w_final)
            continue

        # Edge proposals are tested fresh each iteration. Az edges are primary (tried every iter); range only advances when az is stuck on the current iter, after which the next iter retries az with the wider strip — so a brief az-density dip can be bridged by widening range.

        for _ in range(max_iter):
            # Each edge test runs on the *new strip plus a tail* of the box's already-grown interior on the same side (az_tail rows for top/bottom, rg_tail cols for left/right). The tail is clamped so it never crosses the opposite edge.

            # ---- Phase A: try both azimuth edges ---------------------- The primary strip test (next az_step + last az_tail rows) is permissive (density_threshold). If it fails on an edge, a rescue test peeks much further ahead (az_rescue_lookahead rows outside the box) plus a long tail (az_rescue_lookback rows of the box's interior on the same side), with a stricter threshold (az_rescue_threshold). The rescue is what bridges sparse az dropouts inside an otherwise dense target.
            grew_az = False

            # (a) Top edge: strip = rows [ny_lo, y_lo + az_tail], cols [x_lo, x_hi]
            ny_lo = y_lo - az_step
            new_h = y_hi - ny_lo
            over_cap = max_h is not None and new_h > max_h
            if not over_cap and ny_lo >= 0:
                tail_hi = min(y_lo + az_tail, y_hi)
                # OR partner: accept iff the newly-added region [ny_lo, y_lo) × [x_lo, x_hi) covers a boundary_box seed peak (see `_strip_has_peak`).
                if (
                    _density_ok(ny_lo, tail_hi, x_lo, x_hi)
                    or _strip_has_peak(ny_lo, y_lo, x_lo, x_hi)
                ):
                    y_lo = ny_lo
                    grew_az = True
                else:
                    # Rescue: wider look-ahead + longer tail, stricter th.
                    look_lo = max(y_lo - az_rescue_lookahead, 0)
                    tail_hi_resc = min(y_lo + az_rescue_lookback, y_hi)
                    if _density_ok(look_lo, tail_hi_resc, x_lo, x_hi,
                                   threshold=az_rescue_threshold):
                        y_lo = ny_lo
                        grew_az = True

            # (b) Bottom edge: strip = rows [y_hi - az_tail, ny_hi], cols [x_lo, x_hi]
            ny_hi = y_hi + az_step
            new_h = ny_hi - y_lo
            over_cap = max_h is not None and new_h > max_h
            if not over_cap and ny_hi <= H:
                tail_lo = max(y_hi - az_tail, y_lo)
                # OR partner: accept iff the newly-added region [y_hi, ny_hi) × [x_lo, x_hi) covers a boundary_box seed peak.
                if (
                    _density_ok(tail_lo, ny_hi, x_lo, x_hi)
                    or _strip_has_peak(y_hi, ny_hi, x_lo, x_hi)
                ):
                    y_hi = ny_hi
                    grew_az = True
                else:
                    # Rescue: wider look-ahead + longer tail, stricter th.
                    look_hi = min(y_hi + az_rescue_lookahead, H)
                    tail_lo_resc = max(y_hi - az_rescue_lookback, y_lo)
                    if _density_ok(tail_lo_resc, look_hi, x_lo, x_hi,
                                   threshold=az_rescue_threshold):
                        y_hi = ny_hi
                        grew_az = True

            # ---- Phase B: only if az did NOT advance, take ONE range step. This widens the box and lets next iter retry az with a wider strip (and thus more cells per row to satisfy the density check).
            grew_rg = False
            if not grew_az:
                # (c) Left edge: strip = rows [y_lo, y_hi], cols [nx_lo, x_lo + rg_tail]
                nx_lo = x_lo - rg_step
                new_w = x_hi - nx_lo
                over_cap = max_w is not None and new_w > max_w
                tail_hi_x = min(x_lo + rg_tail, x_hi)
                if (
                    not over_cap and nx_lo >= 0 and (
                        _density_ok(y_lo, y_hi, nx_lo, tail_hi_x)
                        or _strip_has_peak(y_lo, y_hi, nx_lo, x_lo)
                    )
                ):
                    x_lo = nx_lo
                    grew_rg = True

                # (d) Right edge: strip = rows [y_lo, y_hi], cols [x_hi - rg_tail, nx_hi]
                nx_hi = x_hi + rg_step
                new_w = nx_hi - x_lo
                over_cap = max_w is not None and new_w > max_w
                tail_lo_x = max(x_hi - rg_tail, x_lo)
                if (
                    not over_cap and nx_hi <= W and (
                        _density_ok(y_lo, y_hi, tail_lo_x, nx_hi)
                        or _strip_has_peak(y_lo, y_hi, x_hi, nx_hi)
                    )
                ):
                    x_hi = nx_hi
                    grew_rg = True

            # (e) Neither az nor rg can advance → done.
            if not (grew_az or grew_rg):
                break

            # (f) Recenter: shift bounds so they're centred on amp CoM, preserving the current h, w.
            h_cur = y_hi - y_lo
            w_cur = x_hi - x_lo
            sub_amp = amp[y_lo:y_hi, x_lo:x_hi]
            total = float(sub_amp.sum())
            if total > 0:
                row_marg = sub_amp.sum(axis=1)
                col_marg = sub_amp.sum(axis=0)
                yy = np.arange(sub_amp.shape[0])
                xx = np.arange(sub_amp.shape[1])
                dy = float((row_marg * yy).sum() / total)
                dx = float((col_marg * xx).sum() / total)
                y_com = y_lo + int(round(dy))
                x_com = x_lo + int(round(dx))
                # Re-derive bounds, clipping to image but keeping h_cur, w_cur if at all possible.
                y_lo = max(0, y_com - h_cur // 2)
                y_hi = min(H, y_lo + h_cur)
                y_lo = max(0, y_hi - h_cur)
                x_lo = max(0, x_com - w_cur // 2)
                x_hi = min(W, x_lo + w_cur)
                x_lo = max(0, x_hi - w_cur)

            # (g) Post-recentre density check.
            if not _density_ok(y_lo, y_hi, x_lo, x_hi):
                break

        h_final = y_hi - y_lo
        w_final = x_hi - x_lo
        y_c = (y_lo + y_hi) // 2
        x_c = (x_lo + x_hi) // 2
        out[k] = (y_c, x_c, h_final, w_final)

    return out


def extend_boxes_azimuth_strong_signal(
    boxes_yxhw: np.ndarray,
    mask: np.ndarray,
    *,
    step_frac: float = 0.05,
    max_steps: int = 2,
    density_threshold: float = 0.25,
    max_h: int | None = None,
) -> np.ndarray:
    """Post-growth azimuth-only extension of each box's top and bottom edges.

    After `grow_and_recenter_boxes` finalises a box, the density gate may
    have stopped a fraction below the loose-grow threshold even though the
    target's signature continues a little further in azimuth. For each box
    we therefore try, top and bottom independently, to expand the azimuth
    edge by ``step_frac`` of the **entry** box height (e.g. 5%). The trial
    is repeated up to ``max_steps`` times per side (e.g. 2 × 5% = 10%
    cap per side), and accepted iff the detection-mask density inside the
    candidate strip ``mask[strip_rows, x_lo:x_hi]`` is at least
    ``density_threshold`` (same loose 0.25 used during growth). The first
    failed step kills further extension on that side — we don't try to
    bridge gaps here; that's what the growth-phase rescue look-ahead is
    for.

    The box's range bounds and width are untouched. Range-direction
    extension is intentionally out of scope.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int array of (y_centre, x_centre, h, w).
    mask : 2-D detection mask (0/1) aligned with the boxes.
    step_frac : fractional step size, expressed as a fraction of the
        box's height **at entry** to this function (default 0.05 = 5%).
        Using the entry height keeps the total extension per side at
        exactly ``step_frac * max_steps`` of the original height,
        regardless of intermediate acceptances.
    max_steps : maximum number of steps per side (default 2).
    density_threshold : minimum fraction of mask==1 pixels in the
        candidate strip required to accept the step (default 0.25).
    max_h : optional global cap on box height in pixels; extension is
        skipped if it would push the box above this cap.

    Returns
    -------
    out : (K, 4) int array — boxes with possibly enlarged height and
        shifted centre (centre is recomputed from the new bounds).
    """
    if len(boxes_yxhw) == 0:
        return boxes_yxhw
    H, W = mask.shape
    out = np.array(boxes_yxhw, dtype=np.int64, copy=True)
    n_extended_top = 0
    n_extended_bot = 0
    total_extra_h = 0
    for k, (y_c, x_c, h, w) in enumerate(out):
        h_int, w_int = int(h), int(w)
        y_lo = max(0, int(y_c) - h_int // 2)
        y_hi = min(H, y_lo + h_int)
        x_lo = max(0, int(x_c) - w_int // 2)
        x_hi = min(W, x_lo + w_int)

        # Step is sized from the box's entry height so the total cap per side equals exactly step_frac * max_steps of that height.
        step = max(1, int(round(step_frac * h_int)))

        # Top edge (smaller y).
        for _ in range(max_steps):
            ny_lo = y_lo - step
            if ny_lo < 0:
                break
            if max_h is not None and (y_hi - ny_lo) > max_h:
                break
            strip = mask[ny_lo:y_lo, x_lo:x_hi]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_lo = ny_lo
                n_extended_top += 1
            else:
                break

        # Bottom edge (larger y).
        for _ in range(max_steps):
            ny_hi = y_hi + step
            if ny_hi > H:
                break
            if max_h is not None and (ny_hi - y_lo) > max_h:
                break
            strip = mask[y_hi:ny_hi, x_lo:x_hi]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_hi = ny_hi
                n_extended_bot += 1
            else:
                break

        h_new = y_hi - y_lo
        w_new = x_hi - x_lo
        y_c_new = (y_lo + y_hi) // 2
        x_c_new = (x_lo + x_hi) // 2
        total_extra_h += (h_new - h_int)
        out[k] = (y_c_new, x_c_new, h_new, w_new)

    print(
        f"  az-extend: step={step_frac:.0%}×h, max_steps={max_steps}, "
        f"density≥{density_threshold:g}: "
        f"{n_extended_top}/{len(out)} top-steps, "
        f"{n_extended_bot}/{len(out)} bottom-steps, "
        f"total +{total_extra_h} az rows across all boxes"
    )
    return out


def _show_slc(
    ax: plt.Axes,
    s: np.ndarray,
    title: str,
    sigma: float = 4.0,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    max_display_pixels: int = 60_000_000,
) -> tuple[float, float]:
    """Display complex SLC amplitude with mean+`sigma`·σ clip.

    Returns the ``(vmin, vmax)`` actually used so callers can pass
    them into a second panel for a fair side-by-side comparison.

    When the input has more than ``max_display_pixels`` samples, it is
    decimated by integer strides in both axes before being handed to
    ``imshow``. Empirically ``fig.savefig(dpi=350)`` on a
    ``(71784, 27206)`` float32 buffer (7.3 GB) transiently allocates
    ≈110 GB in matplotlib's `_ImageBase._make_image` — full-res norm +
    colormap + RGBA is a ~4×–15× multiplier on the input size before
    matplotlib resamples down to the ~5600×4900 output canvas. The
    decimation caps the imshow source at ≈60 M px (≈240 MB float32),
    which brings the savefig workspace back into the low-GB range and
    is still 2–10× the output canvas so the rendered PNG is visually
    indistinguishable. Overlay coordinates continue to work: we pass
    ``extent`` in full-resolution pixel units so callers can keep
    drawing bounding boxes in the un-decimated coordinate system.
    """
    # Skip the abs allocation for real input — the caller has already reduced `s` to amplitude and we must not double-buffer it (see rule: never copy s).
    img = np.abs(s) if np.iscomplexobj(s) else s
    H_full, W_full = img.shape
    n_pixels = H_full * W_full
    if n_pixels > max_display_pixels:
        # Choose integer strides so decimated size ≤ max_display_pixels while keeping the same aspect ratio. Rounded up so we always land under the cap. The strided slice is intentionally *not* made contiguous — matplotlib's `_make_image` copies the input into a fresh float32 workspace either way, and staying strided saves the ~240 MB contiguous-copy alloc that would otherwise co-exist with `img` for the duration of the imshow call.
        stride = int(np.ceil(np.sqrt(n_pixels / max_display_pixels)))
        step_y = stride
        step_x = stride
        img = img[::step_y, ::step_x]
    # ``extent`` uses matplotlib's [left, right, bottom, top] convention. Setting it to the FULL-resolution pixel bounds (rather than the decimated shape) means any bounding boxes / scatter overlays drawn by callers in full-resolution coordinates still land in the right place on top of the decimated image — imshow interpolates the decimated pixels onto that extent for us.
    extent = (0.0, float(W_full), float(H_full), 0.0)
    if vmin is None or vmax is None:
        mu, sd = float(img.mean()), float(img.std())
        auto_vmax = min(float(img.max()), mu + sigma * sd)
        auto_vmin = max(float(img.min()), mu - sigma * sd)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax
    im = ax.imshow(
        img, cmap=cmap, aspect="auto",
        vmin=vmin, vmax=vmax, extent=extent,
    )
    ax.set_title(title)
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    plt.colorbar(im, ax=ax, label="|s|")
    return float(vmin), float(vmax)


def _show_phase_derivative(ax: plt.Axes, d: np.ndarray, title: str) -> None:
    """Display a real phase array on the cyclic [-pi, pi] twilight colormap."""
    im = ax.imshow(d, cmap="twilight", aspect="auto", vmin=-np.pi, vmax=np.pi)
    ax.set_title(title)
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    plt.colorbar(im, ax=ax, label="rad")


def _show_detection_map(ax: plt.Axes, det: np.ndarray, title: str) -> None:
    """Display detection score map with robust 99th-percentile clip."""
    vmax = float(np.percentile(det, 99))
    im = ax.imshow(det, cmap="hot", aspect="auto", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("range pixel")
    ax.set_ylabel("azimuth pixel")
    plt.colorbar(im, ax=ax, label="detection score")


def _show_image_domain_phase_derivative(ax: plt.Axes, s: np.ndarray) -> None:
    """Compute and display the azimuth phase derivative directly on the input SLC."""
    d = np.angle(s[1:, :] * np.conj(s[:-1, :]))
    _show_phase_derivative(
        ax, d,
        r"$\partial\varphi/\partial u$ of input SLC" + "\narg{s[u+1]·s*[u]}  (image domain)"
    )

from scipy.ndimage import uniform_filter

def compute_subaperture_coherence(
    s: np.ndarray,
    overlap: float = 0.5,
    window: tuple[int, int] = (7, 7),
) -> np.ndarray:
    """Local CCD coherence between two azimuth sub-apertures."""
    N_az = s.shape[0]
    S = np.fft.fft(s, axis=0)

    half = N_az // 2
    keep = int(half * (1.0 + overlap))   # how many bins each look spans
    sub1 = np.zeros_like(S); sub1[:keep] = S[:keep]
    sub2 = np.zeros_like(S); sub2[-keep:] = S[-keep:]

    s1 = np.fft.ifft(sub1, axis=0)
    s2 = np.fft.ifft(sub2, axis=0)

    cross = uniform_filter((s1 * np.conj(s2)).real, size=window) + 1j * \
            uniform_filter((s1 * np.conj(s2)).imag, size=window)
    pow1 = uniform_filter(np.abs(s1) ** 2, size=window)
    pow2 = uniform_filter(np.abs(s2) ** 2, size=window)
    return np.abs(cross) / np.sqrt(pow1 * pow2 + 1e-12)


def compute_subaperture_com_per_box(
    boxes_yxhw: np.ndarray,
    subapertures: np.ndarray,
    N_subaperture: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-box, per-subaperture amplitude-weighted centre of mass.

    For each detection box (in ``s_degraded`` pixel coordinates), the box is
    projected onto each subaperture's grid by dividing its azimuth centre
    and height by ``N_subaperture`` (the range axis is unchanged because
    subapertures share the range axis with ``s_degraded``). Inside the
    projected window, the centre of mass is computed on ``|subapertures[j]|``
    — i.e. on the amplitude data — for every Doppler look ``j``.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int array
        ``[(y_centre, x_centre, h, w), …]`` in ``s_degraded`` pixels.
    subapertures : (N_sub, sub_size, N_range_d) ndarray
        Subaperture amplitudes returned by :func:`compute_subapertures`.
    N_subaperture : int
        Number of Doppler sub-bands (== ``subapertures.shape[0]``).

    Returns
    -------
    com_az_sub : (K, N_sub) float64 ndarray
        Azimuth centre of mass per box per subaperture, in subaperture
        pixel coordinates (``0 <= com_az_sub < sub_size``). NaN where the
        cropped window has zero amplitude.
    com_rg : (K, N_sub) float64 ndarray
        Range centre of mass per box per subaperture, in ``s_degraded``
        range pixels (range is not decimated by subaperturing). NaN where
        the cropped window has zero amplitude.
    """
    n_sub, sub_size, n_rg_d = subapertures.shape
    K = len(boxes_yxhw)
    com_az = np.full((K, n_sub), np.nan, dtype=np.float64)
    com_rg = np.full((K, n_sub), np.nan, dtype=np.float64)
    if K == 0:
        return com_az, com_rg

    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        # Project the box into the subaperture grid (azimuth ÷ N_subaperture).
        y0_sub = max(0, int(round((y_c - h / 2) / N_subaperture)))
        y1_sub = min(sub_size, int(round((y_c + h / 2) / N_subaperture)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg_d, int(round(x_c + w / 2)))
        if y1_sub - y0_sub < 1 or x1 - x0 < 1:
            continue

        az_idx = np.arange(y0_sub, y1_sub, dtype=np.float64)
        rg_idx = np.arange(x0, x1, dtype=np.float64)
        for j in range(n_sub):
            img = np.abs(subapertures[j, y0_sub:y1_sub, x0:x1]).astype(np.float64)
            total = float(img.sum())
            if total <= 0.0 or not np.isfinite(total):
                continue
            com_az[k, j] = float((img.sum(axis=1) * az_idx).sum() / total)
            com_rg[k, j] = float((img.sum(axis=0) * rg_idx).sum() / total)
    return com_az, com_rg


def compute_subaperture_contrast_gain_per_box(
    boxes_yxhw: np.ndarray,
    s_degraded: np.ndarray,
    subapertures: np.ndarray,
    N_subaperture: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-box contrast gain across the N Doppler sub-apertures.

    The contrast metric is the amplitude coefficient of variation

        C(img) = std(|img|) / mean(|img|).

    For each detection box ``k`` two crops are taken:

    * ``full_k``: the box cropped from ``|s_degraded|`` in s_degraded
      pixel coordinates → ``contrast_full[k] = C(full_k)``.
    * ``sub_{k,j}``: the box projected onto subaperture ``j``'s grid
      (azimuth coordinates divided by ``N_subaperture``, range
      unchanged — exactly the projection used by
      :func:`compute_subaperture_com_per_box`) →
      ``contrast_sub[k, j] = C(sub_{k,j})``.

    The per-box gain is

        contrast_gain[k] = sum_j contrast_sub[k, j] / contrast_full[k]

    which equals ``sum_j (contrast_sub[k, j] / contrast_full[k])`` —
    the total per-subaperture sharpness ratio relative to the full
    aperture. A stationary point target has roughly equal contrast in
    every Doppler sub-band as in the full image, so the gain ≈
    ``N_subaperture``. A defocused mover smears in ``s_degraded``
    (low ``contrast_full``) but refocuses in one or a few sub-bands
    (high individual ``contrast_sub``), so the gain rises above
    ``N_subaperture``.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int array
        ``[(y_centre, x_centre, h, w), …]`` in ``s_degraded`` pixels.
    s_degraded : complex ndarray, shape (N_az_d, N_range_d)
        Full-aperture degraded SLC; only its magnitude is used.
    subapertures : (N_sub, sub_size, N_range_d) ndarray
        Subaperture amplitudes returned by :func:`compute_subapertures`.
    N_subaperture : int
        Number of Doppler sub-bands (== ``subapertures.shape[0]``).

    Returns
    -------
    contrast_gain : (K,) float64 ndarray
        Per-box sum-ratio metric. NaN if ``contrast_full[k]`` is
        non-finite / non-positive, or no subaperture had a finite
        contrast.
    contrast_sub : (K, N_sub) float64 ndarray
        Per-box, per-subaperture contrast. NaN where the projected
        window is empty or the crop has zero mean.
    contrast_full : (K,) float64 ndarray
        Per-box contrast of the ``|s_degraded|`` crop. NaN if the crop
        is empty or has zero mean.
    """
    n_sub, sub_size, n_rg_d = subapertures.shape
    n_az_d = s_degraded.shape[0]
    K = len(boxes_yxhw)
    contrast_sub = np.full((K, n_sub), np.nan, dtype=np.float64)
    contrast_full = np.full(K, np.nan, dtype=np.float64)
    contrast_gain = np.full(K, np.nan, dtype=np.float64)
    if K == 0:
        return contrast_gain, contrast_sub, contrast_full

    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg_d, int(round(x_c + w / 2)))
        if x1 - x0 < 1:
            continue

        # Full-aperture crop on the s_degraded grid (azimuth NOT divided).
        y0_full = max(0, int(round(y_c - h / 2)))
        y1_full = min(n_az_d, int(round(y_c + h / 2)))
        if y1_full - y0_full >= 1:
            full = np.abs(s_degraded[y0_full:y1_full, x0:x1]).astype(np.float64)
            mu_full = float(full.mean())
            if np.isfinite(mu_full) and mu_full > 0.0:
                contrast_full[k] = float(full.std()) / mu_full

        # Same projection as compute_subaperture_com_per_box.
        y0_sub = max(0, int(round((y_c - h / 2) / N_subaperture)))
        y1_sub = min(sub_size, int(round((y_c + h / 2) / N_subaperture)))
        if y1_sub - y0_sub >= 1:
            for j in range(n_sub):
                # subapertures already holds amplitudes (real, non-negative).
                img = subapertures[j, y0_sub:y1_sub, x0:x1].astype(np.float64)
                mu = float(img.mean())
                if np.isfinite(mu) and mu > 0.0:
                    contrast_sub[k, j] = float(img.std()) / mu

        if (np.isfinite(contrast_full[k])
                and contrast_full[k] > 0.0
                and np.isfinite(contrast_sub[k]).any()):
            contrast_gain[k] = (
                float(np.nansum(contrast_sub[k])) / contrast_full[k]
            )

    return contrast_gain, contrast_sub, contrast_full


def compute_box_com_velocities(
    com_az_sub: np.ndarray,
    com_rg: np.ndarray,
    sub_size: int,
    prf: float,
    N_subaperture: int,
    azimuth_spacing_m: float,
    Number_of_Range_Looks: int,
    range_spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-box azimuth and range COM velocity (m/s) from a linear fit
    across the N subapertures.

    The k-th subaperture is an IFFT of the k-th of N_subaperture
    contiguous Doppler bands of the SLC azimuth spectrum. In slow time
    its centre lies at

        t_k = (k + 0.5) · (sub_size / PRF)      seconds

    where ``sub_size = N_az_s_degraded // N_subaperture`` is the number of
    raw azimuth samples per band (== ``subapertures.shape[1]``). The COM
    output of :func:`compute_subaperture_com_per_box` is in pixel
    coordinates; we convert each axis to ground metres before fitting:

        az_m[k, j] = com_az_sub[k, j] · N_subaperture · azimuth_spacing_m
        rg_m[k, j] = com_rg[k, j]    · Number_of_Range_Looks · range_spacing_m

    Then a 1-D linear fit ``y = v·t + b`` is performed across the
    subapertures with a finite COM (≥ 2 needed). The slope ``v`` is the
    apparent COM velocity along that axis in m/s. For along-track
    motion the target's amplitude centroid advances by v_az·(t_k+1 − t_k)
    between subaperture centres, so to first order the fit slope equals
    the target's azimuth ground velocity; the analogous logic holds in
    range.

    Returns
    -------
    v_az_mps : (K,) float64
        Azimuth COM velocity (m/s); NaN if fewer than 2 subapertures
        have a finite COM in this box.
    v_rg_mps : (K,) float64
        Range COM velocity (m/s); same NaN convention.
    """
    K, n_sub = com_az_sub.shape
    if K == 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    dt = float(sub_size) / float(prf)
    t = (np.arange(n_sub, dtype=np.float64) + 0.5) * dt
    az_per_pix_m = float(N_subaperture) * float(azimuth_spacing_m)
    rg_per_pix_m = float(Number_of_Range_Looks) * float(range_spacing_m)
    v_az = np.full(K, np.nan, dtype=np.float64)
    v_rg = np.full(K, np.nan, dtype=np.float64)
    for k in range(K):
        az_finite = np.isfinite(com_az_sub[k])
        if az_finite.sum() >= 2:
            try:
                slope, _ = np.polyfit(
                    t[az_finite],
                    com_az_sub[k, az_finite] * az_per_pix_m,
                    1,
                )
                v_az[k] = float(slope)
            except (np.linalg.LinAlgError, ValueError):
                pass
        rg_finite = np.isfinite(com_rg[k])
        if rg_finite.sum() >= 2:
            try:
                slope, _ = np.polyfit(
                    t[rg_finite],
                    com_rg[k, rg_finite] * rg_per_pix_m,
                    1,
                )
                v_rg[k] = float(slope)
            except (np.linalg.LinAlgError, ValueError):
                pass
    return v_az, v_rg


def compute_box_com_theilsen_fit(
    com_az_sub: np.ndarray,
    com_rg: np.ndarray,
    sub_size: int,
    prf: float,
    N_subaperture: int,
    azimuth_spacing_m: float,
    Number_of_Range_Looks: int,
    range_spacing_m: float,
    min_finite: int = 6,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
]:
    """Robust Theil-Sen linear fit of subaperture COM walk vs time.

    Parallels :func:`compute_box_com_velocities` (plain least squares)
    but uses the median of all pairwise slopes for robustness against
    one or two outlier subapertures (weak/defocused looks), and also
    returns the RMS fit residual in metres so downstream code can
    discriminate "clean linear drift = real mover" from "random walk =
    noise". Fit is NaN-aware; a box needs ≥ ``min_finite`` finite COM
    samples on the axis to get a non-NaN slope / residual back.

    Subaperture ``j`` is time-stamped at its centre
    ``t_j = (j + 0.5) · sub_size / PRF``. COM inputs are in pixels on
    the degraded/subaperture grid (1 sub-az px = ``N_subaperture ·
    azimuth_spacing_m``, 1 rg px = ``Number_of_Range_Looks ·
    range_spacing_m``); the fit is performed in ground metres.

    Returns
    -------
    v_az_mps, v_rg_mps : (K,) float64
        Theil-Sen slope (m/s); NaN if fewer than ``min_finite`` finite
        subapertures on that axis.
    res_az_m, res_rg_m : (K,) float64
        RMS residual of the fit (metres); NaN if the fit itself is NaN.
    n_finite_az, n_finite_rg : (K,) int32
        Number of finite COM samples used per axis (out of N_sub).
    """
    K, n_sub = com_az_sub.shape
    if K == 0:
        empty_f = np.empty(0, dtype=np.float64)
        empty_i = np.empty(0, dtype=np.int32)
        return empty_f, empty_f.copy(), empty_f.copy(), empty_f.copy(), empty_i, empty_i.copy()

    dt_per_sub = float(sub_size) / float(prf)
    t = (np.arange(n_sub, dtype=np.float64) + 0.5) * dt_per_sub
    az_per_pix_m = float(N_subaperture) * float(azimuth_spacing_m)
    rg_per_pix_m = float(Number_of_Range_Looks) * float(range_spacing_m)

    v_az = np.full(K, np.nan, dtype=np.float64)
    v_rg = np.full(K, np.nan, dtype=np.float64)
    res_az = np.full(K, np.nan, dtype=np.float64)
    res_rg = np.full(K, np.nan, dtype=np.float64)
    n_fin_az = np.zeros(K, dtype=np.int32)
    n_fin_rg = np.zeros(K, dtype=np.int32)

    def _theilsen(tt: np.ndarray, yy: np.ndarray) -> tuple[float, float]:
        # Median of pairwise slopes; intercept = median(y - slope·t) so
        # (slope, intercept) is the standard Theil-Sen point estimator.
        ii, jj = np.triu_indices(len(tt), k=1)
        dt_pair = tt[jj] - tt[ii]
        dy_pair = yy[jj] - yy[ii]
        # Guard against duplicate time stamps (shouldn't happen but be
        # safe: exclude zero-Δt pairs from the median).
        keep = dt_pair != 0.0
        if not keep.any():
            return float("nan"), float("nan")
        slope = float(np.median(dy_pair[keep] / dt_pair[keep]))
        intercept = float(np.median(yy - slope * tt))
        return slope, intercept

    for k in range(K):
        finite_az = np.isfinite(com_az_sub[k])
        n_fin_az[k] = int(finite_az.sum())
        if n_fin_az[k] >= min_finite:
            tt = t[finite_az]
            yy = com_az_sub[k, finite_az] * az_per_pix_m
            slope, intercept = _theilsen(tt, yy)
            if np.isfinite(slope) and np.isfinite(intercept):
                v_az[k] = slope
                res_az[k] = float(
                    np.sqrt(np.mean((yy - (slope * tt + intercept)) ** 2))
                )

        finite_rg = np.isfinite(com_rg[k])
        n_fin_rg[k] = int(finite_rg.sum())
        if n_fin_rg[k] >= min_finite:
            tt = t[finite_rg]
            yy = com_rg[k, finite_rg] * rg_per_pix_m
            slope, intercept = _theilsen(tt, yy)
            if np.isfinite(slope) and np.isfinite(intercept):
                v_rg[k] = slope
                res_rg[k] = float(
                    np.sqrt(np.mean((yy - (slope * tt + intercept)) ** 2))
                )

    return v_az, v_rg, res_az, res_rg, n_fin_az, n_fin_rg


def compute_box_phase_estimates(
    boxes_yxhw: np.ndarray,
    s_degraded: np.ndarray,
    min_row_coherence: float = 0.0,
    inlier_tol_rad: float = 0.5,
    mask_dec: np.ndarray | None = None,
    N_subaperture: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-box azimuth phase estimate, linear fit, and fit residual.

    d_phase and the amplitude weights are computed per-box from the
    complex SLC ``s_degraded`` rather than being pre-allocated scene-
    wide.

    For each box, the azimuth phase derivative inside the box is collapsed
    across range using an amplitude-weighted circular mean, giving one
    phase sample per azimuth row:

        d_phase[y0+i, x0+j] = arg( s_degraded[y0+i+1, x0+j]
                                 · conj(s_degraded[y0+i, x0+j]) )
        z[i]   = Σ_j w[i, j] · exp(j · d_phase[y0+i, x0+j])
        w[i,j] = |s_degraded[y0+i, x0+j]| · |s_degraded[y0+i+1, x0+j]|
                 · m[i, j] · m[i+1, j]        (mask factor if provided)
        phi[i] = arg z[i]                     (the estimated phase)
        coh[i] = |z[i]| / Σ_j w[i, j]         (per-row coherence ∈ [0, 1])

    When ``mask_dec`` is supplied, the per-pixel weight is multiplied by
    the detection-mask indicator ``m[i, j] = mask_full[y0+i, x0+j]``
    (where ``mask_full = np.repeat(mask_dec, N_subaperture, axis=0)``,
    zero-padded at the tail). This zeroes out speckle-only pixels so
    the per-row circular mean is not contaminated by clutter. Azimuth
    rows whose weight sum is exactly zero (every pixel in the row is
    below the detection mask, or the row is entirely outside the
    valid strip) are treated as **no measurement**: both ``phi[i]``
    and ``coh[i]`` are set to ``NaN`` rather than the spurious
    ``arg(0+0j) = 0`` value that the amplitude-weighted circular mean
    would otherwise yield. Downstream code (line fit, residual,
    ``compute_box_frac_within_thresh``, plotting) filters those rows
    via ``np.isfinite`` gates. The mask slice for each box is derived
    directly from ``mask_dec`` without materialising the scene-wide
    upsampled mask (same technique as
    :func:`filter_boxes_by_phase_residual`).

    The per-row phase ``phi(i)`` is the actual estimate we save per target.
    A coherence-weighted polyfit gives the linear model

        phi(i) ≈ slope · i + intercept            (i = 0 .. n_rows-1)

    which lets you reconstruct the fit at absolute azimuth row ``y0 + i``.
    The per-box residual score reported here is the unweighted mean of the
    wrapped distance from each sample to the fit,

        r[i] = arg{ exp(j · (phi[i] - (slope·i + intercept))) }   (∈ (-π, π])
        residual_mean = mean_i |r[i]|

    i.e. the mean Euclidean distance (in radians, with circular wrap) from
    each dot in Figure 5 to the red line. A coherent ramp has a small
    score; a noisy / clutter target has a large one. Same metric the
    ``--max-phase-residual-rad`` filter compares against.

    Identical numerics to :func:`filter_boxes_by_phase_slope` /
    :func:`filter_boxes_by_phase_residual`.

    Returns
    -------
    phi : (K, max_h) float64
        Per-row weighted-circular-mean azimuth phase (rad), NaN-padded
        beyond each box's actual row count.
    coh : (K, max_h) float64
        Per-row coherence in [0, 1], NaN-padded the same way.
    slope : (K,) float64
        Slope of the coherence-weighted linear fit (rad / azimuth row).
        NaN where the fit was not performed (< 2 valid rows, etc.).
    intercept : (K,) float64
        Intercept of the same fit at i = 0, i.e. at absolute row ``y0``
        (rad). NaN where the fit was not performed.
    y0 : (K,) int64
        Absolute azimuth row of the first sample of each box's strip in
        d_phase coordinates. Use this with ``i = 0 .. n_rows-1`` to
        reconstruct absolute azimuth rows.
    n_rows : (K,) int64
        Actual number of azimuth rows used per box (may be less than ``h``
        for boxes clipped at the image edge). 0 for boxes the fit could
        not run on.
    residual_mean : (K,) float64
        Coh-weighted mean wrapped Euclidean distance from each sample to
        the chosen line (rad), i.e. the actual cost the brute-force grid
        search minimised — the smaller the better. NaN where the fit
        was not performed.
    n_inliers : (K,) int64
        Diagnostic: number of high-coh rows that happen to land within
        ``inlier_tol_rad`` of the chosen line. 0 where the fit was not
        performed.
    """
    K = len(boxes_yxhw)
    if K == 0:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0, 0), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    # `int(...)` would truncate float box heights and can be one row short of the actual span `L = int(round(y_c + h/2)) - int(round(y_c - h/2))` for non-integer ``h`` (e.g. h=6391.7 → trunc=6391, but rounding can produce L=6392). Use ``ceil`` with one extra row of headroom so the (K, max_h) ``phi_arr`` / ``coh_arr`` buffers always fit `L`, and ``n_rows_arr[k] = L`` stays consistent with ``phi_arr.shape[1]`` for every downstream `box_phi[k, :n_k]` slice.
    max_h = max(
        int(np.ceil(float(boxes_yxhw[:, 2].max()))) + 1, 1,
    )
    phi_arr = np.full((K, max_h), np.nan, dtype=np.float64)
    coh_arr = np.full((K, max_h), np.nan, dtype=np.float64)
    slope_arr = np.full(K, np.nan, dtype=np.float64)
    intercept_arr = np.full(K, np.nan, dtype=np.float64)
    y0_arr = np.zeros(K, dtype=np.int64)
    n_rows_arr = np.zeros(K, dtype=np.int64)
    residual_arr = np.full(K, np.nan, dtype=np.float64)
    n_inliers_arr = np.zeros(K, dtype=np.int64)
    n_az_slc, n_rg = s_degraded.shape
    n_az = n_az_slc - 1                            # d_phase row count
    sub_size_mask = mask_dec.shape[0] if mask_dec is not None else 0
    N_sub = max(int(N_subaperture), 1)
    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        y0 = max(0, int(round(y_c - h / 2)))
        y1 = min(n_az, int(round(y_c + h / 2)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg, int(round(x_c + w / 2)))
        y0_arr[k] = y0
        L = y1 - y0
        if L < 2 or x1 - x0 < 1:
            continue
        # Per-box d_phase + amplitude weights, both derived from the complex SLC. y1 ≤ n_az = N_az_slc - 1 so ``y1 + 1 ≤ N_az_slc``; the extra row makes the conjugate product / weight product well-defined at the bottom edge. weight[i, j] = |s[y0+i, x0+j]|·|s[y0+i+1, x0+j]| — multiplied by the detection-mask indicator on both endpoints when a mask is provided (see docstring).
        s_box = s_degraded[y0:y1 + 1, x0:x1]
        amp_box = np.abs(s_box)
        sub_phase = np.angle(s_box[1:] * np.conj(s_box[:-1]))
        weight = amp_box[:-1] * amp_box[1:]
        if mask_dec is not None:
            r_lo = y0 // N_sub
            r_hi = min(sub_size_mask, -(-(y1 + 1) // N_sub))
            if r_hi > r_lo:
                mask_slice = np.repeat(
                    mask_dec[r_lo:r_hi, x0:x1].astype(np.float32),
                    N_sub, axis=0,
                )
                row_offset = y0 - r_lo * N_sub
                mask_slice = mask_slice[row_offset:row_offset + (y1 + 1 - y0)]
                if mask_slice.shape[0] < (y1 + 1 - y0):
                    pad_rows = (y1 + 1 - y0) - mask_slice.shape[0]
                    mask_slice = np.vstack([
                        mask_slice,
                        np.zeros((pad_rows, mask_slice.shape[1]),
                                 dtype=mask_slice.dtype),
                    ])
            else:
                mask_slice = np.zeros((y1 + 1 - y0, x1 - x0), dtype=np.float32)
            weight = weight * (mask_slice[:-1] * mask_slice[1:])
        z = (weight * np.exp(1j * sub_phase)).sum(axis=1)
        w_raw = weight.sum(axis=1)                     # no epsilon; used to detect rows with no measurement
        w_sum = w_raw + 1e-12
        phi = np.angle(z)
        coh = np.abs(z) / w_sum
        # Rows whose weight sum is exactly zero (all pixels below the detection mask, or a fully-clipped box row) have NO phase measurement — `np.angle(0+0j)` would spuriously return 0.0. Mark them NaN so downstream `np.isfinite`-gated logic (line fit, residual, frac_within_thresh, plots) drops them instead of treating a fabricated 0 rad as a real sample.
        _no_meas = w_raw <= 0.0
        if _no_meas.any():
            phi = np.where(_no_meas, np.nan, phi)
            coh = np.where(_no_meas, np.nan, coh)
        L_pad = min(L, max_h)
        phi_arr[k, :L_pad] = phi[:L_pad]
        coh_arr[k, :L_pad] = coh[:L_pad]
        n_rows_arr[k] = L
        # Centred index keeps slope and intercept numerically independent; we then shift the intercept back to absolute row y0 (i.e. i=0) for downstream reconstruction.
        u = np.arange(L, dtype=float)
        # Brute-force wrap-aware line fit on the wrapped phi values: try every slope on an equispaced grid, take the wrap-aware optimal intercept per candidate, and pick the candidate whose wrapped line minimises the coh-weighted mean wrapped distance to the data. No LS refit on top — the grid value is returned directly, so the smooth-cost local-min failure mode (phasor LS landing on a high-wrap-count line that does not actually pass through the data) cannot happen.
        slope, intercept, n_in = _min_distance_line_fit(
            phi.astype(np.float64), u, coh.astype(np.float64),
            min_row_coherence=min_row_coherence,
            inlier_tol_rad=inlier_tol_rad,
        )
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue
        slope_arr[k] = float(slope)
        intercept_arr[k] = float(intercept)
        n_inliers_arr[k] = int(n_in)
        # Residual = the actual cost minimised by the grid search = coh-weighted mean |wrapped(phi - line)| over high-coh rows, with fall-back to all finite rows when too few are high-coh. NaN phi / coh rows (mask-out rows: no measurement) always contribute 0 weight so they never poison the mean.
        r = np.angle(np.exp(1j * (phi - (slope * u + intercept))))
        finite_row = np.isfinite(phi) & np.isfinite(coh)
        high = finite_row & (coh > min_row_coherence)
        if int(high.sum()) < 2:
            high = finite_row
        w_row = np.where(high, coh, 0.0)
        denom = float(w_row.sum())
        if denom > 0.0:
            abs_r = np.where(finite_row, np.abs(r), 0.0)
            residual_arr[k] = float((w_row * abs_r).sum() / denom)
    return (
        phi_arr, coh_arr, slope_arr, intercept_arr,
        y0_arr, n_rows_arr, residual_arr, n_inliers_arr,
    )


def compute_box_frac_within_thresh(
    box_phi: np.ndarray,
    box_coh: np.ndarray,
    box_slope_rad_per_row: np.ndarray,
    box_intercept_rad: np.ndarray,
    box_n_rows: np.ndarray,
    thresh_rad: float = 1.0,
    min_row_coherence: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-box fraction of azimuth rows within ``+/- thresh_rad`` of the linear fit.

    Complements ``box_residual_rad`` (a *mean*-|r|-style metric restricted to
    high-coherence rows) with a scale-free "how much of the ramp is actually
    close to the line" fraction — computed on the SAME coherence-gated
    subset the line fit / residual use, so all three metrics agree on
    which rows are "real phase measurements".

    For each box ``k``, first restrict to rows with a valid measurement
    (finite phi, finite coh) whose coherence exceeds ``min_row_coherence``.
    If fewer than 2 rows pass that gate, fall back to every finite row so
    the score is at least defined (matches the ``_min_distance_line_fit``
    and ``compute_box_phase_estimates`` residual conventions). Then

        r_i = arg{ exp(j · (phi_i - (slope · i + intercept))) }
        frac[k]     = mean_i    1[|r_i| <= thresh_rad]              (unweighted)
        frac_coh[k] = Σ_i coh_i · 1[|r_i| <= thresh_rad] / Σ_i coh_i (coh-weighted)

    Both are in [0, 1]; NaN if no row passes the gate or the fit
    slope/intercept are NaN. Row set is identical between the two —
    ``frac_coh`` differs from ``frac`` only in the per-row weighting
    (coherence-weighted vs. uniform), not in which rows contribute.

    Empirically, on the WTW3YQ dev scene the AF-validated movers cluster
    at ``frac_coh >= ~0.7`` while boxes that AF gains ~0 dB on cluster
    around ~0.55 -- see ``fit_quality_out/WTW3YQ_fit_vs_af.png``.

    Parameters
    ----------
    min_row_coherence : float, keyword-only, default 0.5
        Per-row coherence threshold: only rows with ``coh > min_row_coherence``
        count towards the fraction. Set to 0.0 to disable the gate and
        recover the legacy "every finite row contributes, weighted by
        coh" behaviour.
    """
    K = int(box_phi.shape[0]) if box_phi.ndim >= 1 else 0
    frac_arr = np.full(K, np.nan, dtype=np.float64)
    frac_coh_arr = np.full(K, np.nan, dtype=np.float64)
    if K == 0:
        return frac_arr, frac_coh_arr
    thresh_val = float(thresh_rad)
    min_coh = float(min_row_coherence)
    for k in range(K):
        n_k = int(box_n_rows[k])
        if n_k <= 0:
            continue
        slope_k = float(box_slope_rad_per_row[k])
        b_k = float(box_intercept_rad[k])
        if not (np.isfinite(slope_k) and np.isfinite(b_k)):
            continue
        phi = np.asarray(box_phi[k, :n_k], dtype=np.float64)
        coh = np.asarray(box_coh[k, :n_k], dtype=np.float64)
        finite = np.isfinite(phi) & np.isfinite(coh)
        if not np.any(finite):
            continue
        # High-coh gate (mirrors the line fit / residual). Fall back to every finite row if too few rows pass, so the score is at least defined for degenerate boxes — identical convention to `_min_distance_line_fit` and the residual computation in `compute_box_phase_estimates`.
        gate = finite & (coh > min_coh)
        if int(gate.sum()) < 2:
            gate = finite
        phi_v = phi[gate]
        coh_v = np.clip(coh[gate], 0.0, 1.0)
        idx = np.arange(n_k, dtype=np.float64)[gate]
        r = np.angle(np.exp(1j * (phi_v - (slope_k * idx + b_k))))
        close = np.abs(r) <= thresh_val
        frac_arr[k] = float(close.mean())
        w_sum = float(coh_v.sum())
        if w_sum > 0.0:
            frac_coh_arr[k] = float(np.sum(coh_v * close) / w_sum)
    return frac_arr, frac_coh_arr


def filter_boxes_by_phase_residual(
    boxes_yxhw: np.ndarray,
    s_degraded: np.ndarray,
    max_residual_rad: float,
    mask_dec: np.ndarray | None = None,
    N_subaperture: int = 1,
    detection_skip_frac: float = 0.5,
    min_row_coherence: float = 0.0,
    inlier_tol_rad: float = 0.5,
) -> np.ndarray:
    """Drop boxes whose mean wrapped phase residual to the linear fit
    exceeds ``max_residual_rad`` (rad).

    For each box, the per-row weighted-circular-mean phase ``phi[i]`` and
    the brute-force wrap-aware fit ``phi_fit[i] = slope·i + intercept``
    are computed exactly as in :func:`compute_box_phase_estimates`. The
    score is the coh-weighted mean wrapped Euclidean distance over the
    high-coh rows (with a fall-back to all rows when too few qualify),

        r[i]          = arg{ exp(j · (phi[i] - phi_fit[i])) }   (∈ (-π, π])
        residual_mean = Σ_high coh[i]·|r[i]| / Σ_high coh[i]

    i.e. the average distance (rad) from each high-coh sample to the red
    fit line in Figure 5, with circular wrap so a real ±2π roll-over
    inside the box isn't counted as a 2π error. A coherent ramp scores
    low; a noisy / clutter target scores high. Identical to the cost
    the grid-search line fit just minimised, so the filter compares the
    box against the actual optimum.

    d_phase and the amplitude weights are computed per-box from the
    complex SLC ``s_degraded`` rather than being pre-allocated scene-
    wide.

    Bypass on detection count
    -------------------------
    If ``mask_dec`` is supplied, an upsampled binary mask
    ``mask_full = np.repeat(mask_dec, N_subaperture, axis=0)`` (trimmed
    to ``s_degraded.shape[0]`` rows, with any missing rows treated as
    zero) is defined **implicitly** — the actual (N_az, N_rg) array is
    never materialised. A box is kept regardless of its phase residual
    when

        mask_full[y0:y1, x0:x1].sum() > detection_skip_frac · (y1 - y0)

    (its CoV-gate detection count exceeds ``detection_skip_frac`` per
    azimuth row on average). The idea is that a target with that many
    CoV hits is already strongly supported by amplitude evidence alone,
    so a noisy phase trace shouldn't kill it. Set
    ``detection_skip_frac`` to a very large value to disable the bypass.

    Parameters
    ----------
    mask_dec : (sub_size, N_rg) real ndarray, optional
        Decimated CoV mask on the subaperture-statistics grid (as
        returned by :func:`cluster_targets` as ``mask_filt``). Each row
        stands for ``N_subaperture`` rows of the full-resolution grid
        where ``s_degraded`` lives.
    N_subaperture : int
        Number of azimuth rows in s_degraded per row of ``mask_dec``.
        Ignored when ``mask_dec is None``.

    Non-positive ``max_residual_rad`` disables the filter (passes
    everything through unchanged).
    """
    if max_residual_rad <= 0 or len(boxes_yxhw) == 0:
        return boxes_yxhw
    n_az_slc, n_rg = s_degraded.shape
    n_az = n_az_slc - 1                            # d_phase row count
    keep = np.zeros(len(boxes_yxhw), dtype=bool)
    n_bypass = 0
    sub_size = mask_dec.shape[0] if mask_dec is not None else 0
    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        y0 = max(0, int(round(y_c - h / 2)))
        y1 = min(n_az, int(round(y_c + h / 2)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg, int(round(x_c + w / 2)))
        L = y1 - y0
        if L < 2 or x1 - x0 < 1:
            continue
        # Detection-count bypass. mask_full = np.repeat(mask_dec, N_sub, axis=0) has each source row of mask_dec repeated N_sub times into the full grid; rows past sub_size*N_sub are zero-padded. So Σ mask_full[y0:y1, x0:x1] can be computed directly from mask_dec without materialising the (N_az, N_rg) upsampled array: for each source row r ∈ [r_lo, r_hi), the overlap `min((r+1)·N_sub, y1) − max(r·N_sub, y0)` counts how many full-grid rows it contributes to inside [y0, y1); multiply that by the column-sum of mask_dec[r, x0:x1] and sum. Any box with > detection_skip_frac · L mask hits passes regardless of residual — a bright target with a noisy phase trace shouldn't be culled by this filter.
        if mask_dec is not None:
            r_lo = y0 // N_subaperture
            r_hi = min(sub_size, -(-y1 // N_subaperture))
            if r_hi > r_lo:
                r_arr = np.arange(r_lo, r_hi)
                row_y_lo = r_arr * N_subaperture
                row_y_hi = row_y_lo + N_subaperture
                overlap = np.minimum(row_y_hi, y1) - np.maximum(row_y_lo, y0)
                per_row_sum = mask_dec[r_lo:r_hi, x0:x1].sum(axis=1)
                det_count = int((overlap * per_row_sum).sum())
            else:
                det_count = 0
            if det_count > detection_skip_frac * L:
                keep[k] = True
                n_bypass += 1
                continue
        # Per-box d_phase + amplitude weights from the complex SLC (see the slope filter).
        s_box = s_degraded[y0:y1 + 1, x0:x1]
        amp_box = np.abs(s_box)
        sub_phase = np.angle(s_box[1:] * np.conj(s_box[:-1]))
        weight = amp_box[:-1] * amp_box[1:]
        z = (weight * np.exp(1j * sub_phase)).sum(axis=1)
        w_sum = weight.sum(axis=1) + 1e-12
        phi = np.angle(z)
        coh = np.abs(z) / w_sum
        u = np.arange(L, dtype=float)
        slope, intercept, _n_in = _min_distance_line_fit(
            phi.astype(np.float64), u, coh.astype(np.float64),
            min_row_coherence=min_row_coherence,
            inlier_tol_rad=inlier_tol_rad,
        )
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            continue
        # Residual = the same coh-weighted mean wrapped distance the grid search minimised; nothing more, nothing less.
        r = np.angle(np.exp(1j * (phi - (slope * u + intercept))))
        high = coh > min_row_coherence
        if int(high.sum()) < 2:
            high = np.ones_like(coh, dtype=bool)
        w_row = np.where(high, coh, 0.0)
        denom = float(w_row.sum())
        if denom <= 0.0:
            continue
        residual_mean = float((w_row * np.abs(r)).sum() / denom)
        if np.isfinite(residual_mean) and residual_mean <= max_residual_rad:
            keep[k] = True
    if mask_dec is not None and n_bypass:
        print(
            f"  residual filter: bypassed {n_bypass}/{len(boxes_yxhw)} "
            f"boxes via detection count > "
            f"{detection_skip_frac:g} · n_azimuth_rows"
        )
    return boxes_yxhw[keep]

def compute_contrast(image: np.ndarray) -> float:
    """Image-intensity contrast ``std(|x|^2) / mean(|x|^2)``.

    Accepts a real or complex array of any shape. Higher values indicate
    sharper, more focused imagery. Computation is performed in ``float64``
    to avoid precision loss on large SLC magnitudes, and non-finite samples
    are dropped before the statistics are taken. Returns ``0.0`` if the
    input is empty, all non-finite, or has non-positive mean intensity.
    """
    arr = np.asarray(image)
    if arr.size == 0:
        return 0.0
    intensity = np.abs(arr).astype(np.float64, copy=False) ** 2
    if not np.all(np.isfinite(intensity)):
        intensity = intensity[np.isfinite(intensity)]
        if intensity.size == 0:
            return 0.0
    mean = intensity.mean()
    if not np.isfinite(mean) or mean <= 0.0:
        return 0.0
    return float(intensity.std() / mean)


def filter_boxes_by_phase_slope(
    boxes_yxhw: np.ndarray,
    s_degraded: np.ndarray,
    min_slope_deg: float,
    min_row_coherence: float = 0.0,
    inlier_tol_rad: float = 0.5,
) -> np.ndarray:
    """Drop boxes whose amp-weighted phase ramp is shallower than ``min_slope_deg``.

    For each box, computes the coherence-weighted circular mean of the
    per-box azimuth phase derivative
    ``d_phase = arg(s_degraded[1:] · conj(s_degraded[:-1]))`` along range
    for every azimuth row inside the box, then fits a wrap-aware line
    ``phi(u) = slope·u + b`` via brute-force grid search
    (see :func:`_min_distance_line_fit` — the slope minimising the
    coh-weighted mean wrapped distance over an equispaced grid of 1001
    candidates, with no LS refit). Boxes with
    ``abs(np.degrees(slope)) < min_slope_deg`` are removed, along with
    any box that cannot be fit (fewer than two valid azimuth rows or
    NaN slope).

    d_phase and the amplitude weights are both computed per-box from the
    complex SLC ``s_degraded`` rather than being pre-allocated scene-
    wide. Box centres/sizes are in ``s_degraded`` (azimuth-row)
    coordinates.

    Parameters
    ----------
    s_degraded : (N_az, N_rg) complex ndarray
        Range-degraded complex SLC. Only the slice
        ``s_degraded[y0:y1+1, x0:x1]`` is read per box; both d_phase and
        the amplitude weights ``|s[y0+i, x0+j]|·|s[y0+i+1, x0+j]|`` are
        derived from it inline.

    Returns the kept rows of ``boxes_yxhw`` in their original order. If
    ``min_slope_deg <= 0`` the input is returned unchanged.
    """
    if min_slope_deg <= 0 or len(boxes_yxhw) == 0:
        return boxes_yxhw
    n_az_slc, n_rg = s_degraded.shape
    n_az = n_az_slc - 1                            # d_phase row count
    min_slope_rad = float(np.deg2rad(min_slope_deg))
    keep = np.zeros(len(boxes_yxhw), dtype=bool)
    for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
        y0 = max(0, int(round(y_c - h / 2)))
        y1 = min(n_az, int(round(y_c + h / 2)))
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg, int(round(x_c + w / 2)))
        if y1 - y0 < 2 or x1 - x0 < 1:
            continue
        # Per-box d_phase + amplitude weights, both derived from the complex SLC. y1 ≤ n_az = N_az_slc - 1 so y1+1 ≤ N_az_slc; the extra row makes the conjugate product / weight product well-defined at the bottom edge.
        s_box = s_degraded[y0:y1 + 1, x0:x1]
        amp_box = np.abs(s_box)
        sub_phase = np.angle(s_box[1:] * np.conj(s_box[:-1]))
        weight = amp_box[:-1] * amp_box[1:]
        z = (weight * np.exp(1j * sub_phase)).sum(axis=1)
        w_sum = weight.sum(axis=1) + 1e-12
        phi = np.angle(z)
        coh = np.abs(z) / w_sum
        u_local = np.arange(y1 - y0, dtype=np.float64)
        slope, _, _n_in = _min_distance_line_fit(
            phi.astype(np.float64), u_local, coh.astype(np.float64),
            min_row_coherence=min_row_coherence,
            inlier_tol_rad=inlier_tol_rad,
        )
        if np.isfinite(slope) and abs(slope) >= min_slope_rad:
            keep[k] = True
    return boxes_yxhw[keep]


def filter_nested_boxes(
    boxes_yxhw: np.ndarray,
    coverage_thresh: float = 0.9,
) -> np.ndarray:
    """Drop the LARGER of any pair of boxes whose smaller member is
    covered ≥ ``coverage_thresh`` of its own area by the larger.

    For each ordered pair ``(i, j)`` in the input array the coverage
    of ``i`` by ``j`` is

        cov(i, j) = area(box_i ∩ box_j) / area(box_i)

    (i.e. what fraction of the smaller box lies inside the bigger).
    When ``cov(i, j) ≥ coverage_thresh`` and ``area(j) > area(i)``,
    box ``j`` is marked for deletion — the tighter ``i`` stays. Tied
    areas are broken by index: the higher-indexed member is treated
    as the bigger one, so mutual-overlap duplicates collapse to a
    single survivor (the lower-indexed one).

    Intended as a post-clustering cleanup: removes redundant loose
    bounding boxes that fully envelope a tighter detection. Boxes
    are on the same integer coordinate grid (``s_degraded`` rows /
    columns) with ``(y_c, x_c, h, w)`` as centre + inclusive size,
    so the intersection is a rectangle in those coordinates. Runs
    in ``O(K²)`` — fine for the ~1e3–1e4 boxes cluster_targets
    produces on an ICEYE tile.

    Parameters
    ----------
    boxes_yxhw : (K, 4) int ndarray
        Rows are ``(y_c, x_c, h, w)`` in the same coordinate system.
    coverage_thresh : float, default 0.9
        Fraction (of the smaller box's area) that must lie inside the
        bigger box to trigger deletion of the bigger one. Set
        ``> 1.0`` to disable the filter entirely.

    Returns
    -------
    ndarray
        Kept rows of ``boxes_yxhw`` in their original order.
    """
    K = len(boxes_yxhw)
    if K < 2 or coverage_thresh > 1.0:
        return boxes_yxhw
    y_c = boxes_yxhw[:, 0].astype(np.float64)
    x_c = boxes_yxhw[:, 1].astype(np.float64)
    h = boxes_yxhw[:, 2].astype(np.float64)
    w = boxes_yxhw[:, 3].astype(np.float64)
    y0 = y_c - h / 2.0
    y1 = y_c + h / 2.0
    x0 = x_c - w / 2.0
    x1 = x_c + w / 2.0
    area = h * w
    # Pairwise axis-aligned rectangle intersection area (K, K).
    ix0 = np.maximum(x0[:, None], x0[None, :])
    ix1 = np.minimum(x1[:, None], x1[None, :])
    iy0 = np.maximum(y0[:, None], y0[None, :])
    iy1 = np.minimum(y1[:, None], y1[None, :])
    iw = np.clip(ix1 - ix0, 0.0, None)
    ih = np.clip(iy1 - iy0, 0.0, None)
    inter = iw * ih
    # cov[i, j] = |box_i ∩ box_j| / |box_i| — how much of i is
    # covered by j. Self-pairs zeroed out so a box never marks
    # itself for deletion.
    cov = inter / area[:, None]
    idx = np.arange(K)
    cov[idx, idx] = 0.0
    # Strict area comparison + index tiebreaker: `j` is treated as
    # the "bigger" box relative to `i` when area[j] > area[i], OR
    # when the areas are equal and j > i. Guarantees exactly one
    # survivor from mutually-≥thresh duplicate pairs (the lower-
    # indexed one).
    bigger = area[None, :] > area[:, None]
    tied_higher = (area[None, :] == area[:, None]) & (
        idx[None, :] > idx[:, None]
    )
    # Column j is deleted iff some row i satisfies
    #   cov[i, j] ≥ thresh  AND  j is bigger-than-or-tied-higher-than i.
    delete_j = (
        (cov >= coverage_thresh) & (bigger | tied_higher)
    ).any(axis=0)
    return boxes_yxhw[~delete_j]


# --------------------------------------------------------------------------- Main ---------------------------------------------------------------------------
def compute_subapertures(s: np.ndarray, N: int = 8) -> np.ndarray:
    """Split SLC into N azimuth subapertures and return their amplitudes.

    Parameters
    ----------
    s : complex ndarray, shape (N_az, N_range)
        Input SLC. Axis 0 = azimuth (spatial), axis 1 = range.
    N : int
        Number of subapertures to split into.

    Returns
    -------
    amplitudes : real ndarray, shape (N, N_az // N, N_range)
        Amplitude of each subaperture image.
    """
    N_az = s.shape[0]
    sub_size = N_az // N

    # Azimuth FFT with DC at the centre so contiguous index slices map to contiguous Doppler sub-bands ordered from -Nyq to +Nyq. Force complex64 to keep the scene-sized spectrum in ≈7.5 GB rather than ≈15 GB (matches the `degrade_range_resolution_range_sum` accumulator dtype).
    fft_dtype = s.dtype if s.dtype == np.complex64 else np.complex64
    S = np.fft.fftshift(np.fft.fft(s, axis=0), axes=0).astype(
        fft_dtype, copy=False,
    )

    # float32 amplitudes: `|complex64|` naturally lands in float32, and storing 8×(sub_size × N_range) at float64 is a ≈3 GB persistent-alive allocation on a full ICEYE scene. Keep it at float32 so `amplitudes` is ≈1.5 GB and downstream `sub_mean` / `sub_var` stay float32 too.
    amplitudes = np.zeros((N, sub_size, s.shape[1]), dtype=np.float32)
    for k in range(N):
        band = S[k * sub_size:(k + 1) * sub_size]
        s_sub = np.fft.ifft(band, axis=0)
        amplitudes[k] = np.abs(s_sub).astype(np.float32, copy=False)

    # Free the ≈7.5 GB range/azimuth spectrum before the caller sees `amplitudes`; the rest of the pipeline never needs `S` again.
    del S
    return amplitudes


# --------------------------------------------------------------------------- Rough refocus helpers — line → quadratic phase → freq-domain correction ---------------------------------------------------------------------------
def _normalized_variance(chip: np.ndarray) -> float:
    """Image contrast = std(|I|²) / mean(|I|²) of a complex chip.

    This is the dimensionless intensity-contrast metric used in
    entropy-minimisation autofocus papers. Higher = sharper (more
    concentrated energy). NaN-safe; returns 0.0 if mean intensity
    is zero or non-finite.
    """
    intensity = np.abs(chip) ** 2
    m = float(np.mean(intensity))
    if not np.isfinite(m) or m <= 0.0:
        return 0.0
    return float(np.std(intensity)) / m


def _refocus_box_chip(
    chip: np.ndarray,
    slope_rad_per_row: float,
    intercept_rad: float = 0.0,  # kept for API symmetry; unused on purpose
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the centred quadratic phase of the fitted line as a refocus.

    Background. The per-box fit produced a line in d_phase space,

        d_phase_line(n) = slope · n + intercept,   n = 0..N-1.

    The intercept is the Doppler-centroid shift (= pure azimuth
    translation of the chip; irrelevant for contrast / focusing). The
    quadratic phase error itself comes from the *slope*: re-centring
    the line so its midpoint sits at zero gives

        d_phase_centered(m) = slope · m,   m = n − N/2,

    which integrates analytically to a centred, midpoint-zero
    quadratic phase

        φ(m) = ½ · slope · m²,    φ(0) = 0.

    Because the QPE is estimated in the azimuth-frequency domain in
    PGA, we apply the correction *there* following the reference
    `shear_focusing.apply_phase_correction` recipe:

        chip_F  = fft(ifftshift(chip, axis=0), axis=0)
        chip_F *= exp(-1j · φ)[:, None]
        out     = fftshift(ifft(chip_F, axis=0), axis=0)

    The purpose is NOT real autofocus (a much better one happens
    elsewhere) but a classification signal: if the slope we fitted
    really corresponds to a moving target's QPE, applying its
    conjugate on the spectrum will concentrate the chip's energy and
    raise time-domain image contrast; if the slope was noise, the
    contrast will not improve.

    Returns the corrected chip and the applied per-row phase φ(n).
    """
    N = chip.shape[0]
    if N < 2:
        return chip.copy(), np.zeros(N, dtype=np.float64)
    # Centred-line integral: φ(m) = ½ · slope · m²,  m = n − N//2. φ is zero at the midpoint by construction.
    m = np.arange(N, dtype=np.float64) - (N // 2)
    phi = 0.5 * float(slope_rad_per_row) * m * m
    chip_F = np.fft.fft(np.fft.ifftshift(chip, axes=0), axis=0)
    chip_F = chip_F * np.exp(1j * phi)[:, None]
    corrected = np.fft.fftshift(np.fft.ifft(chip_F, axis=0), axes=0)
    return corrected, phi


# ---------------------------------------------------------------------- Polynomial range-walk autofocus (mirrors core/autofocus.py) ---------------------------------------------------------------------- The QGIS plugin's `apply_global_range_deviation_correction` (core/autofocus.py:760-804) estimates a polynomial range walk by maximising the per-subaperture intensity-contrast sum across a coarse-to-fine grid of polynomial deviations, then applies the corresponding per-row range shift on the azimuth-FFT and IFFTs back. This script is standalone (no QGIS), so the same helpers are replicated locally without the QgsMessageLog logging.


def _af_compute_contrast_subaperture_sums(
    image: np.ndarray, N_subaperture: int = 10
) -> np.ndarray:
    """Per-subaperture focus metric over the centred 80 % of azimuth.

    Mirrors ``core.autofocus.compute_contrast_subaperture_sums``. Drops
    the outer 10 % of azimuth rows on each side, splits the remaining
    centred 80 % into ``N_subaperture`` equal chunks and returns
    ``std_range(mean_az(|x|²))`` per chunk. Higher = more focused.
    """
    H = image.shape[0]
    start = H // 10
    end = H - H // 10
    center = image[start:end]
    seg = (end - start) // N_subaperture
    intensity = np.zeros(N_subaperture, dtype=np.float32)
    if seg < 1:
        return intensity
    for i in range(N_subaperture):
        chunk = center[i * seg:(i + 1) * seg, :]
        if chunk.size == 0:
            continue
        intensity[i] = np.mean(np.abs(chunk) ** 2, axis=0).std()
    return intensity


def _af_shift_fitted(
    s: np.ndarray, fitted: np.ndarray
) -> np.ndarray:
    """Per-row range shift via Fourier-domain phase ramp.

    Mirrors ``core.autofocus.shift_fitted``. Each row ``k`` of ``s`` is
    shifted along the range axis by ``fitted[k]`` samples.
    """
    N = s.shape[1]
    # Force complex64 workspace throughout. The original ``np.exp(1j*…)`` call promoted to complex128 because ``1j`` is a Python complex (=complex128) and ``fitted``/``k_over_N`` were float64, and the subsequent ``fft(s_complex64) * phase_ramp_complex128`` product also upcast to complex128 — doubling the per-iteration transient on multi-MB chips. Because ``_af_find_best_deviation`` calls this ~40–50× per box across ~200 boxes, that allocator pressure is the dominant source of the AF-stage OOM ceiling on full ICEYE scenes. We also pre-fftshift the tiny 1-D ``k_over_N`` vector instead of fftshifting the chip-sized ``phase_ramp`` afterwards — ``np.fft.fftshift`` is implemented via ``np.roll`` which copies, so shifting the 1-D axis vector once eliminates a full chip-sized complex64 allocation per call.
    k_over_N = np.arange(N, dtype=np.float32) / np.float32(N) - np.float32(0.5)
    k_shifted = np.fft.fftshift(k_over_N)
    phase = (
        np.float32(2.0 * np.pi)
        * fitted.astype(np.float32, copy=False)[:, None]
        * k_shifted[None, :]
    )
    shift_term = np.empty(phase.shape, dtype=np.complex64)
    np.cos(phase, out=shift_term.real)
    np.sin(phase, out=shift_term.imag)
    del phase
    fft_s = np.fft.fft(s, axis=1)
    fft_s *= shift_term
    del shift_term
    return np.fft.ifft(fft_s, axis=1)


# Nominal geometry defaults for turning `best_deviation` (samples) into an azimuth-velocity estimate when a real ICEYE sidecar isn't available. `_AF_V_SAT_MPS` is the effective platform speed used in the RCMC reference (ICEYE LEO orbit at ~585 km gives ~7300 m/s); `_AF_R0_M` is a middle-of-swath nominal slant range (~700 km for typical ICEYE incidence angles); `_AF_LAMBDA_M` is X-band ICEYE wavelength (c / 9.65 GHz ≈ 0.031 m). Real runs override every one of these from ``iceye_md`` — see ``_af_scene_geometry`` — so the constants only bite for legacy .npy patches with no sidecar attached.
_AF_V_SAT_MPS = 7300.0
_AF_R0_M = 7.0e5
_AF_LAMBDA_M = 0.031


def _va_from_range_walk_mps(
    best_deviation_samples: float,
    *,
    range_spacing_m: float,
    processing_prf_hz: float,
    wavelength_m: float,
    slant_range_m: float,
    v_sat: float = _AF_V_SAT_MPS,
) -> float:
    """Azimuth velocity implied by the Doppler-domain residual-RCM fit (m/s).

    The polynomial ``fitted(x) = -best_deviation * (x/0.5)**2`` (samples,
    ``x`` normalised to ``[-0.5, 0.5]``) is applied as a *correction*
    on the Doppler-domain data — ``x`` therefore spans the FFT's
    Nyquist range in Doppler frequency, from ``-processing_prf/2`` to
    ``+processing_prf/2``. The target's actual residual range-cell
    migration in that domain is ``+best_deviation * (x/0.5)**2``
    samples, i.e. at the FFT edges

        |dR_peak| = best_deviation * range_spacing_m   [m].

    For a target moving at along-track velocity ``v_a`` the residual
    RCM at Doppler ``f_d`` (after the processor's stationary-target
    RCMC) is (to first order in ``v_a / v_sat``)

        dR_res(f_d)  =  f_d^2 * lambda^2 * R_0 * v_a / (4 * v_sat^3).

    Setting ``f_d = processing_prf / 2`` and inverting for ``v_a``:

        v_a  =  16 * v_sat^3 * dR_peak / (processing_prf^2 * lambda^2 * R_0).

    Sign convention: positive ``best_deviation`` → positive ``v_a``.
    Which platform-relative direction that corresponds to (co-moving
    vs counter-flying) depends on the SLC's range-axis orientation and
    needs one-off verification against a known-direction reference.

    Notes
    -----
    * Sensitivity for ICEYE spotlight-1m-class scenes (SXK97E-like:
      ``processing_prf`` ≈ 1e5 Hz, ``lambda`` ≈ 0.0306 m, ``R_0`` ≈ 680
      km, ``v_sat`` = 7300 m/s, ``range_spacing`` ≈ 0.098 m): about
      0.09 m/s per range sample of ``best_deviation``. So a
      ``best_deviation = -142.8`` sample fit maps to ``v_a`` ≈ −13
      m/s, which is a physically plausible ship velocity.
    * Stripmap scenes have ``processing_prf`` ≈ acquisition PRF (few
      kHz) and a coarser ``range_spacing``, so the same
      ``best_deviation`` in samples corresponds to a much larger
      ``v_a`` — the estimator's per-sample sensitivity is orders of
      magnitude worse in stripmap and this column should be treated as
      a coarse cross-check only there.
    """
    if (
        processing_prf_hz <= 0.0
        or wavelength_m <= 0.0
        or slant_range_m <= 0.0
        or v_sat == 0.0
    ):
        return float("nan")
    dR_peak_m = float(best_deviation_samples) * float(range_spacing_m)
    return (
        16.0 * float(v_sat) ** 3 * dR_peak_m
        / (
            float(processing_prf_hz) ** 2
            * float(wavelength_m) ** 2
            * float(slant_range_m)
        )
    )


def _format_v_a_str(
    best_deviation_samples: float,
    *,
    range_spacing_m: float,
    processing_prf_hz: float | None,
    wavelength_m: float | None,
    slant_range_m: float | None,
    v_sat_mps: float | None,
) -> str:
    """Return a ``"  v_a≈+X.XX m/s"`` suffix for the AF log line, or empty.

    Empty when any of the four geometry inputs is ``None`` (legacy call
    sites that don't thread the scene geometry through), or when the
    Doppler-domain formula returns a non-finite value.
    """
    if (
        processing_prf_hz is None
        or wavelength_m is None
        or slant_range_m is None
        or v_sat_mps is None
    ):
        return ""
    v_a_walk = _va_from_range_walk_mps(
        best_deviation_samples,
        range_spacing_m=range_spacing_m,
        processing_prf_hz=processing_prf_hz,
        wavelength_m=wavelength_m,
        slant_range_m=slant_range_m,
        v_sat=v_sat_mps,
    )
    if not np.isfinite(v_a_walk):
        return ""
    return f"  v_a≈{v_a_walk:+.2f} m/s"


def _af_scene_geometry(
    iceye_md: dict | None,
    *,
    fallback_prf_hz: float | None = None,
) -> tuple[float, float, float, float]:
    """Resolve scene-scoped geometry for the range-walk → v_a mapping.

    Returns ``(processing_prf_hz, wavelength_m, slant_range_m, v_sat_mps)``.
    Uses the loaded ICEYE STAC/bundle fields when present and falls back
    to the module-level nominals (``_AF_V_SAT_MPS``, ``_AF_R0_M``,
    ``_AF_LAMBDA_M``) for anything missing. ``fallback_prf_hz`` is used
    only when ``iceye_md`` is missing both ``iceye_processing_prf`` and
    was itself absent (legacy .npy patches with no sidecar): in that
    case we fall back to ``fallback_prf_hz`` (typically ``cfg.prf``,
    which for stripmap is a good approximation of the processing PRF).
    """
    v_sat = _AF_V_SAT_MPS
    if iceye_md is None:
        return (
            float(fallback_prf_hz) if fallback_prf_hz else float("nan"),
            _AF_LAMBDA_M,
            _AF_R0_M,
            v_sat,
        )
    processing_prf = iceye_md.get("iceye_processing_prf")
    if processing_prf is None:
        # Stripmap fallback: the compressed image is sampled at (roughly) the acquisition PRF, so use ``cfg.prf`` (which callers already threaded through). This is only exact for classic stripmap; anywhere the two disagree — spotlight, sliding spotlight — the sidecar's ``iceye:processing_prf`` should have been present and the fallback isn't hit.
        processing_prf = (
            float(iceye_md.get("iceye_acquisition_prf", fallback_prf_hz or 0.0))
        )
    center_freq = iceye_md.get("sar_center_frequency")
    if center_freq is not None and center_freq > 0.0:
        wavelength_m = 299_792_458.0 / float(center_freq)
    else:
        wavelength_m = _AF_LAMBDA_M
    slant_range_m = float(iceye_md.get("iceye_range", _AF_R0_M))
    return (
        float(processing_prf),
        float(wavelength_m),
        float(slant_range_m),
        float(v_sat),
    )


def _af_contrast_ratio_sum(
    C: np.ndarray, C_initial: np.ndarray
) -> float:
    """Sum of element-wise ``C / C_initial`` (1.0 where baseline is 0)."""
    ratio = np.divide(
        C, C_initial,
        out=np.ones_like(C, dtype=np.float64),
        where=C_initial > 0,
    )
    return float(np.sum(ratio))


def _af_find_best_deviation(
    spatch_fft: np.ndarray,
    x: np.ndarray,
    *,
    dev_min: float = -100.0,
    dev_max: float = 100.0,
    accuracy: float = 0.5,
    poly_degree: int = 2,
) -> tuple[float, int]:
    """Coarse-to-fine search for the polynomial range deviation that
    maximises ``sum(C / C_initial)`` over the 10 subapertures.

    Mirrors ``core.autofocus.find_best_deviation`` (no QGIS logging).
    Returns ``(best_deviation, n_sub_contrast_improved)``: the second
    value is the per-subaperture count of contrast improvements
    captured at the most recent baseline update — i.e. how many of the
    10 subapertures improved at the step that produced
    ``best_deviation``. The upstream implementation always returns 0
    for this field; here it is computed as documented so the per-box
    plot conveys a meaningful number.
    """
    coarse_step = max(accuracy * 5.0, (dev_max - dev_min) / 20.0)
    best_deviation = 0.0
    N_subaperture = 10
    max_contrast = float(N_subaperture)
    C_initial = _af_compute_contrast_subaperture_sums(
        spatch_fft, N_subaperture,
    )
    n_sub_contrast_improved = 0

    coarse = np.arange(dev_min, dev_max + coarse_step, coarse_step)
    for deviation in coarse:
        c = deviation / x[-1] ** poly_degree
        fitted = -c * x ** poly_degree
        shifted = np.abs(_af_shift_fitted(spatch_fft, fitted))
        C = _af_compute_contrast_subaperture_sums(shifted, N_subaperture)
        contrast_gain = _af_contrast_ratio_sum(C, C_initial)
        if contrast_gain > max_contrast:
            n_sub_contrast_improved = int(np.sum(C > C_initial))
            max_contrast = contrast_gain
            C_initial = np.copy(C)
            best_deviation = float(deviation)
    if abs(best_deviation) <= 1.0:
        return best_deviation, int(n_sub_contrast_improved)
    fine_min = max(dev_min, best_deviation - 2 * coarse_step)
    fine_max = min(dev_max, best_deviation + 2 * coarse_step)
    fine = np.arange(fine_min, fine_max + accuracy, accuracy)
    for deviation in fine:
        c = deviation / x[-1] ** poly_degree
        fitted = -c * x ** poly_degree
        shifted = np.abs(_af_shift_fitted(spatch_fft, fitted))
        C = _af_compute_contrast_subaperture_sums(shifted, N_subaperture)
        contrast_gain = _af_contrast_ratio_sum(C, C_initial)
        if contrast_gain > max_contrast:
            n_sub_contrast_improved = int(np.sum(C > C_initial))
            max_contrast = contrast_gain
            C_initial = np.copy(C)
            best_deviation = float(deviation)

    return best_deviation, int(n_sub_contrast_improved)


# --------------------------------------------------------------------------- Standalone phase-gradient-autofocus (PGA) pipeline --------------------------------------------------------------------------- Mirrors ``core.autofocus.focus_with_centered_looks_pga`` and the helpers it depends on (``phase_gradient_autofocus``, ``select_pulse_with_strong_target``, ``apply_phase_correction``, ``entropy``, ``center_on_strong_target``, ``calculate_window``, ``weigthed_estimator``) as well as ``core.looks.extract_centered_look``. The script-side copy drops all QGIS logging so this file remains importable without ``qgis.core``. Used by :func:`_af_apply_global_range_deviation_correction` to refine the polynomial range-walk correction with PGA on chips where the residual range walk is large (|best_deviation| > 3.6).

def _pga_ft(s: np.ndarray, axis: int = -1) -> np.ndarray:
    """Centered FFT along *axis*."""
    return np.fft.fftshift(np.fft.fft(s, axis=axis), axes=axis)


def _pga_ift(f: np.ndarray, axis: int = -1) -> np.ndarray:
    """Centered inverse FFT along *axis*."""
    return np.fft.ifft(np.fft.ifftshift(f, axes=axis), axis=axis)


def _pga_entropy(data: np.ndarray) -> float:
    """Power-normalised entropy of complex data."""
    pwr = np.abs(data) ** 2
    pwr = pwr[pwr > 0]
    if pwr.size == 0:
        return float("inf")
    p = pwr / pwr.sum()
    return float(-np.sum(p * np.log(p)))


def _pga_select_pulse_with_strong_target(
    s: np.ndarray, percentile: float = 95.0, axis: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the rows/columns whose max-amplitude profile is in the top
    ``100 - percentile`` percent.
    """
    if axis not in (0, 1):
        raise ValueError("Axis must be 0 or 1")
    line_max = np.amax(np.abs(s), axis=1 - axis)
    threshold = np.percentile(line_max, percentile)
    target_lines = np.where(line_max >= threshold)[0]
    if axis == 1:
        return s[:, target_lines], target_lines
    return s[target_lines], target_lines


def _pga_center_on_strong_target(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Roll each line so its strongest target sits at the centre of *axis*."""
    H, W = x.shape
    max_index = np.argmax(np.abs(x), axis=axis)
    if axis == 1:
        center = W // 2
        shifts = center - max_index
        return x[np.arange(H)[:, None], (np.arange(W) - shifts[:, None]) % W]
    center = H // 2
    shifts = center - max_index
    return x[
        (np.arange(H) - shifts[:, None]) % H,
        np.arange(W)[:, None],
    ]


def _pga_calculate_window(
    s: np.ndarray,
    threshold: float = -20.0,
    min_width: int = 50,
    axis: int = -1,
) -> np.ndarray:
    """Pick a centred target window from the power profile along *axis*.

    Sums ``|s|^2`` along *axis*, then returns a 1-D index array of length
    ``width`` covering a centred window on the OTHER axis (the axis that
    survives the reduction and is what the caller will index, e.g.
    ``p[:, window]`` after ``axis=0``). ``width`` is the count of bins
    within ``threshold`` dB of the per-axis power peak, floored at
    ``min_width`` and clamped to ``s.shape[other_axis] - 1`` so the
    window never overruns the indexed axis.

    Mirrors ``core.autofocus.calculate_window`` but fixes its axis
    confusion (the upstream version clamps / centres on ``s.shape[axis]``
    instead of the surviving axis, so ``rows > cols`` patches blow up
    when the window indices are used on a smaller axis).
    """
    p = np.sum(np.abs(s * np.conj(s)), axis=axis)
    p_max = p.max()
    if p_max > 0 and np.isfinite(p_max):
        p_db = 10.0 * np.log10(p / p_max)
        width = int(np.sum(p_db > threshold))
    else:
        width = min_width
    if width < min_width:
        width = min_width
    # The window must index the axis that survived the reduction, i.e. the OTHER axis from the one we summed over. This is the bug-fix over upstream ``core.autofocus.calculate_window``.
    other_axis = (axis + 1) % s.ndim
    n_other = s.shape[other_axis]
    width = min(width, n_other - 1)
    if width < 1:
        width = max(1, n_other - 1)
    center = (n_other - 1) // 2
    return np.arange(-width // 2, width // 2) + center


def _pga_weighted_estimator(x: np.ndarray) -> np.ndarray:
    """Magnitude-weighted phase-difference estimator (axis 0 = azimuth)."""
    s = np.conj(x[:-1, :]) * x[1:, :]
    return np.sum(np.angle(s) * np.abs(s), axis=1) / np.sum(np.abs(s), axis=1)


def _pga_phase_gradient_autofocus(
    data: np.ndarray,
    iter_num: int = 1,
    tolerance: float = 0.01,
) -> tuple[np.ndarray, list[float], list[float]]:
    """PGA estimate of the azimuth phase error.

    Returns ``(phase_corrections, rms_history, entropy_history)``.
    """
    entropies = [_pga_entropy(data)]
    phase_corrections = np.zeros(data.shape[0])
    rms: list[float] = []
    for _ in range(iter_num):
        data_centered = _pga_center_on_strong_target(data, axis=1)
        window = _pga_calculate_window(data_centered, axis=0)

        p = np.zeros_like(data_centered)
        p[:, window] = data_centered[:, window]
        P = _pga_ft(p, axis=0)

        phase_change = _pga_weighted_estimator(P)
        phase_change = np.unwrap([0, *np.cumsum(phase_change)])

        t = np.arange(0, phase_change.shape[0])
        trend = np.poly1d(np.polyfit(t, phase_change, 1))
        phase_change -= trend(t)
        rms.append(float(np.sqrt(np.mean(phase_change ** 2))))

        if rms[-1] < tolerance:
            break

        data = _pga_ft(data, axis=0)
        data *= np.exp(-1j * phase_change[:, None])
        data = _pga_ift(data, axis=0)

        entropies.append(_pga_entropy(data))
        phase_corrections += phase_change

    return phase_corrections[:, None], rms, entropies


def _pga_apply_phase_correction(
    data: np.ndarray, phase_error: np.ndarray,
) -> np.ndarray:
    """Apply (interpolated) azimuth phase correction to *data*."""
    x = np.linspace(0.0, 1.0, data.shape[0])
    xp = np.linspace(0.0, 1.0, phase_error.shape[0])
    phase_error_interp = np.interp(x, xp, phase_error.squeeze())
    data = _pga_ft(data, axis=0)
    data *= np.exp(-1j * phase_error_interp[:, None])
    return _pga_ift(data, axis=0)


def _pga_insert_center(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Insert *src* into the centre of *dst* (in place); returns *dst*."""
    rows, cols = dst.shape
    src_rows, src_cols = src.shape
    if src_rows > rows or src_cols > cols:
        raise ValueError("Source array is larger than destination array")
    start_row = rows // 2 - src_rows // 2
    start_col = cols // 2 - src_cols // 2
    dst[start_row:start_row + src_rows, start_col:start_col + src_cols] = src
    return dst


def _pga_extract_centered_look(
    spectrum: np.ndarray,
    center_row: int,
    center_col: int,
    look_rows: int,
    look_cols: int,
    *,
    apply_ifftshift: bool = True,
) -> np.ndarray:
    """Zero-pad a centred (look_rows, look_cols) sub-spectrum and inverse-FFT.

    Standalone mirror of ``core.looks.extract_centered_look``.
    """
    if look_rows <= 0 or look_cols <= 0:
        raise ValueError("Look size must be positive in both dimensions")

    rows, cols = spectrum.shape
    if not (0 <= center_row < rows and 0 <= center_col < cols):
        raise ValueError("Center index is out of bounds for the spectrum")
    if look_rows > rows or look_cols > cols:
        raise ValueError("Look size cannot exceed spectrum dimensions")

    window_row_start = center_row - look_rows // 2
    window_row_end = window_row_start + look_rows
    window_col_start = center_col - look_cols // 2
    window_col_end = window_col_start + look_cols

    src_row_start = max(0, window_row_start)
    src_row_end = min(rows, window_row_end)
    src_col_start = max(0, window_col_start)
    src_col_end = min(cols, window_col_end)

    look_window = np.zeros((look_rows, look_cols), dtype=spectrum.dtype)
    dst_row_start = src_row_start - window_row_start
    dst_row_end = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = src_col_start - window_col_start
    dst_col_end = dst_col_start + (src_col_end - src_col_start)

    look_window[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = (
        spectrum[src_row_start:src_row_end, src_col_start:src_col_end]
    )

    centered_spectrum = _pga_insert_center(np.zeros_like(spectrum), look_window)
    if apply_ifftshift:
        centered_spectrum = np.fft.ifftshift(centered_spectrum)
    return np.fft.ifft2(centered_spectrum)


def _pga_centered_look_row_counts_from_azimuth_fractions(
    spectrum_rows: int, azimuth_look_fractions: tuple[float, ...],
) -> list[int]:
    """Azimuth look heights as fractions of the Doppler spectrum row count."""
    if spectrum_rows < 1:
        return []
    heights: list[int] = []
    for frac in azimuth_look_fractions:
        h = max(1, min(spectrum_rows, int(round(spectrum_rows * frac))))
        if not heights or h > heights[-1]:
            heights.append(h)
    return heights


def _focus_with_centered_looks_pga(
    data: np.ndarray,
    *,
    azimuth_look_fractions: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25),
) -> tuple[np.ndarray, int]:
    """Standalone mirror of ``core.autofocus.focus_with_centered_looks_pga``.

    Builds centred azimuth looks at the requested fractions of the Doppler
    spectrum, runs PGA on each look's strong-pulse patch, and returns the
    PGA-corrected look with the lowest entropy. Falls back to *data* if no
    look yields a valid phase estimate.

    Returns ``(image, best_look_rows)``. The image is the PGA-corrected
    winning centred look, *already amplitude-rescaled* by ``rows /
    best_look_rows`` so its magnitudes are comparable to the full-bandwidth
    input — contrast comparisons against the caller's pre-PGA data are
    therefore meaningful. When no look produces a valid estimate, the
    function returns the input unchanged together with ``best_look_rows = 0``.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D SLC data, got shape {data.shape}")

    data = np.ascontiguousarray(data, dtype=np.complex64)
    spectrum = np.fft.fftshift(np.fft.fft2(data))
    rows, cols = spectrum.shape
    look_heights = _pga_centered_look_row_counts_from_azimuth_fractions(
        rows, azimuth_look_fractions,
    )
    if not look_heights:
        return data, 0

    best_entropy = float("inf")
    best: np.ndarray | None = None
    best_look_rows: int | None = None
    for azimuth_look_size in look_heights:
        try:
            look = _pga_extract_centered_look(
                spectrum,
                center_row=rows // 2,
                center_col=cols // 2,
                look_rows=azimuth_look_size,
                look_cols=cols,
                apply_ifftshift=True,
            )
        except ValueError:
            continue

        patch, _ = _pga_select_pulse_with_strong_target(look, axis=0)
        if patch.size == 0 or patch.shape[0] < 2:
            continue

        phase_error, _, _ = _pga_phase_gradient_autofocus(patch)
        corrected_look = _pga_apply_phase_correction(look, phase_error)
        score = _pga_entropy(corrected_look)
        if score < best_entropy:
            best_entropy = score
            best = corrected_look
            best_look_rows = azimuth_look_size

    if best is None or best_look_rows is None:
        return data, 0

    # Compensate for the azimuth bandwidth thrown away by the winning look: zeroing rows-best_look_rows of the Doppler spectrum reduces distributed-clutter *power* by best_look_rows/rows, so amplitudes scale by sqrt(best_look_rows/rows). Multiply by the inverse sqrt so speckle statistics match the surrounding scene when the chip is pasted back into |s|; point-target peaks are then slightly under-scaled (they would need the linear ratio) but that trade keeps distributed backgrounds visually seamless. The caller's contrast comparison reads off this scaled image.
    scale = np.sqrt(rows / float(best_look_rows))
    image = (best * scale).astype(data.dtype, copy=False)
    return image, int(best_look_rows)


def _af_apply_global_range_deviation_correction(
    data: np.ndarray,
    *,
    range_spacing: float,
    dev_min_meters: float = -20.0,
    dev_max_meters: float = 20.0,
    accuracy: float = 0.5,
    poly_degree: int = 2,
    pga_abs_deviation_threshold: float = 3.6,
    early_exit_threshold_m: float | None = 2.0,
    progress_index: int | None = None,
    progress_total: int | None = None,
    processing_prf_hz: float | None = None,
    wavelength_m: float | None = None,
    slant_range_m: float | None = None,
    v_sat_mps: float | None = None,
) -> tuple[np.ndarray | None, float, int, int, float, float, float]:
    """Standalone copy of ``core.autofocus.apply_global_range_deviation_correction``.

    Returns ``(corrected, best_deviation, n_sub_contrast_improved,
    best_look_rows, contrast_before, contrast_after, gain_db)``.

    When ``abs(best_deviation) > pga_abs_deviation_threshold`` (default
    3.6), the polynomial range-walk correction is refined with the local
    centred-look PGA pipeline (mirrors the QGIS-side two-stage autofocus
    in :class:`core.autofocus.AutofocusTask`).

    ``contrast_before`` is ``compute_contrast(data)`` on the raw input
    chip and ``contrast_after`` is ``compute_contrast(corrected)`` on
    the post-pipeline image (polynomial range-walk shift plus, when
    triggered, centred-look PGA refinement). Both numbers describe the
    images the caller plots as "before" and "after", so the dB gain on
    the figure title matches what the eye sees and is not inflated by
    any bandwidth-discard credit. ``best_look_rows`` is the winning
    centred-look height (``0`` when PGA was not triggered).

    False-alarm short-circuit: when ``early_exit_threshold_m`` is a
    non-negative float and the range-walk search returns
    ``|best_deviation| * range_spacing < early_exit_threshold_m``,
    the box is treated as doomed by the downstream
    ``af_min_abs_deviation`` gate and the correction is skipped: no
    final range-walk shift, no PGA, no ``contrast_after`` measurement.
    In that case ``corrected`` is ``None`` and ``contrast_after`` /
    ``gain_db`` are ``NaN``. Callers that already guard on
    ``corrected is None`` (e.g. the pre-compute loop populating
    ``corrected_chip_af``) get the "skip this box" signal for free.
    Passing ``early_exit_threshold_m=None`` disables the short-circuit.
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2D SLC data, got shape {data.shape}")
    # Convert the metres-side gate to slant-range samples for the actual comparison. Callers express the gate in ground units so it stays scene-independent (WTW3YQ ``range_spacing=0.24 m`` vs SXK97E ``range_spacing=0.098 m`` would otherwise pick different effective thresholds for the same "3.5 samples" hard-coded value). ``None`` disables the short-circuit — kept for parity with the previous samples-domain signature.
    early_exit_threshold = (
        None
        if early_exit_threshold_m is None
        else early_exit_threshold_m / range_spacing
    )
    dev_min = int(dev_min_meters / range_spacing)
    dev_max = int(dev_max_meters / range_spacing)
    rows = data.shape[0]
    spatch_fft = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    x = np.linspace(-0.5, 0.5, rows)
    best_deviation, n_sub_contrast_improved = _af_find_best_deviation(
        spatch_fft, x,
        dev_min=dev_min, dev_max=dev_max,
        accuracy=accuracy, poly_degree=poly_degree,
    )
    # Doomed-box short-circuit — skip everything downstream (final shift + inverse FFT + PGA + `contrast_after`) since the caller's `af_min_abs_deviation` gate will drop this box anyway. Returning ``corrected=None`` triggers the existing ``if bounds_k is None or corrected_k is None: continue`` guard in the AF PNG/NPZ writing loop, so no other plumbing has to change.
    if (
        early_exit_threshold is not None
        and abs(best_deviation) < early_exit_threshold
    ):
        if progress_index is not None and progress_total is not None:
            progress_str = f"[{progress_index}/{progress_total}] "
        else:
            progress_str = ""
        # Report the (very small) range-walk-derived v_a when the scene geometry is threaded through — useful for spotting boxes that were doomed only because they fell below the ``af_early_exit_threshold_m`` gate but whose walked-velocity was borderline. When any geometry piece is missing we fall back to the original message.
        v_a_str = _format_v_a_str(
            best_deviation,
            range_spacing_m=range_spacing,
            processing_prf_hz=processing_prf_hz,
            wavelength_m=wavelength_m,
            slant_range_m=slant_range_m,
            v_sat_mps=v_sat_mps,
        )
        print(
            f"[af] {progress_str}best_deviation={best_deviation:+.3f}  "
            f"(skipped: |dev| < {early_exit_threshold:g}){v_a_str}"
        )
        return (
            None,
            best_deviation,
            n_sub_contrast_improved,
            0,
            float("nan"),
            float("nan"),
            float("nan"),
        )
    fitted = (
        -best_deviation / x[-1] ** poly_degree * x ** poly_degree
    )
    spatch_fft = _af_shift_fitted(spatch_fft, fitted)
    corrected = np.fft.ifft(
        np.fft.ifftshift(spatch_fft, axes=0), axis=0,
    )
    best_look_rows = 0
    is_apply_focusing = True
    if is_apply_focusing and abs(best_deviation) > pga_abs_deviation_threshold:
        corrected, best_look_rows = _focus_with_centered_looks_pga(corrected)
    # Contrast pair measured directly on the images the caller plots: raw input chip vs final corrected chip. The dB gain is the straight ratio of those two contrasts — no bandwidth-discard credit, no compensation. Matches the eye.
    contrast_before = compute_contrast(data)
    contrast_after = compute_contrast(corrected)
    if contrast_before > 0.0 and contrast_after > 0.0:
        gain_db = 20.0 * np.log10(contrast_after / contrast_before)
    else:
        gain_db = float("nan")
    gain_str = (
        f"{gain_db:+.2f} dB" if np.isfinite(gain_db) else "n/a"
    )
    pga_tag = (
        f"range-walk + PGA @ look_rows={best_look_rows}"
        if best_look_rows > 0 else "range-walk only"
    )
    if progress_index is not None and progress_total is not None:
        progress_str = f"[{progress_index}/{progress_total}] "
    else:
        progress_str = ""
    # Range-walk-derived v_a in the "kept" log line, mirroring the CSV column. See ``_va_from_range_walk_mps`` for the derivation and sensitivity caveats — treat as a coarse cross-check to the phase-domain QPE estimator produced elsewhere in the pipeline.
    v_a_str = _format_v_a_str(
        best_deviation,
        range_spacing_m=range_spacing,
        processing_prf_hz=processing_prf_hz,
        wavelength_m=wavelength_m,
        slant_range_m=slant_range_m,
        v_sat_mps=v_sat_mps,
    )
    print(
        f"[af] {progress_str}best_deviation={best_deviation:+.3f}  "
        f"contrast {contrast_before:.4f} -> {contrast_after:.4f}  "
        f"({gain_str}; {pga_tag}){v_a_str}"
    )
    return (
        corrected.astype(data.dtype, copy=False),
        best_deviation,
        n_sub_contrast_improved,
        int(best_look_rows),
        float(contrast_before),
        float(contrast_after),
        float(gain_db),
    )


def _plot_refocus_box_1x2(
    chip_before: np.ndarray,
    chip_after: np.ndarray,
    *,
    box_idx: int,
    box_yxhw: tuple[int, int, int, int],
    slope_rad_per_row: float,
    slope_total_rad: float,
    contrast_before: float,
    contrast_after: float,
    gain_db: float,
    out_path: Path,
) -> None:
    """One PNG per box: |chip| before / |chip| after the centred-QPE refocus.

    chip is taken from s_degraded over the fit window. The title shows
    the per-box slope and the time-domain contrast gain (a moving-target
    classification signal, NOT a focusing claim).
    """
    y, x, h, w = box_yxhw
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Refocus box {box_idx:03d} @ (y={y}, x={x}, h={h}, w={w})  "
        f"slope = {slope_rad_per_row:+.4f} rad/row  "
        f"slope·n = {slope_total_rad:+.3f} rad  "
        f"contrast {contrast_before:.3f} → {contrast_after:.3f}  "
        f"({gain_db:+.2f} dB)",
        fontsize=11,
    )
    for ax, img, ttl in (
        (axes[0], chip_before, "|I|  before"),
        (axes[1], chip_after,  "|I|  after"),
    ):
        amp = np.abs(img)
        if amp.size == 0:
            ax.text(0.5, 0.5, "(empty)", ha="center", va="center")
            ax.set_title(ttl)
            continue
        vmin = float(np.nanpercentile(amp, 1.0))
        vmax = float(np.nanpercentile(amp, 99.0))
        if vmax <= vmin:
            vmax = vmin + 1.0
        ax.imshow(amp, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_xlabel("range")
        ax.set_ylabel("azimuth")
        ax.set_title(ttl)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    _maybe_close(fig)




def _plot_shear_averaging_box(
    chip: np.ndarray,
    *,
    box_idx: int,
    box_yxhw: tuple[int, int, int, int],
    out_path: Path,
) -> None:
    """Visualise the three inputs of the shear-averaging autofocus on one box chip.

    Given the complex chip ``image`` (axis 0 = azimuth, axis 1 = range),
    the function plots:

    * ``image``                                          — time-domain |chip|.
    * ``image_azfft = FFT_azimuth(image)``               — log-magnitude of
      the azimuth (Doppler) spectrum, fftshifted so DC sits at the centre.
    * ``image_azfft[:-1] * conj(image_azfft[1:])``       — the "shear"
      product of adjacent Doppler bins. Both the magnitude (where the
      shear product is reliable — bright bins dominate the average) and
      the wrapped phase (the quantity that the shear-averaging algorithm
      averages coherently across range to estimate the QPE) are shown.

    No averaging / weighting / unwrapping is performed here — the
    helper is purely diagnostic so the upstream steps of the
    shear-averaging method (Mancill & Swiger 1981; Wahl et al.) can be
    inspected per box.
    """
    y, x, h, w = box_yxhw

    if chip.size == 0 or chip.shape[0] < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "(empty chip)", ha="center", va="center")
        ax.set_title(
            f"Shear box {box_idx:03d} @ (y={y}, x={x}, h={h}, w={w})"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
        _maybe_close(fig)
        return

    image = chip
    image_azfft = np.fft.fftshift(np.fft.fft(chip, axis=0), axes=0)
    shear = image_azfft[:-1, :] * np.conj(image_azfft[1:, :])

    # --- Amplitude-weighted circular mean of the shear phase across range --- Mirrors the kernel of compute_box_phase_estimates: take the amplitude-weighted circular mean across the "range" axis. The natural per-cell weight for arg{S[k]·S*[k+1]} is the modulus of the same complex product, i.e. |shear| = |S[k]|·|S[k+1]|. With that choice the weighted sum collapses to a plain coherent sum:     z[k]   = Σ_r |shear[k, r]| · exp(j · arg{shear[k, r]})            = Σ_r shear[k, r]     phi[k] = arg z[k]                         (per-Doppler-bin phase)     coh[k] = |z[k]| / Σ_r |shear[k, r]|       (coherence ∈ [0, 1]) Low-|S| Doppler bins (noise floor) contribute weight ≈ 0 to z and to Σ_r |shear|, so the mask is encoded in the amplitudes — no explicit gate needed at this stage.
    shear_amp = np.abs(shear)
    z_k = shear.sum(axis=1)
    w_sum_k = shear_amp.sum(axis=1) + 1e-12
    phi_k = np.angle(z_k)
    coh_k = np.abs(z_k) / w_sum_k

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(
        f"Shear averaging box {box_idx:03d} @ (y={y}, x={x}, h={h}, w={w})  "
        f"shape={chip.shape}",
        fontsize=11,
    )

    amp = np.abs(image)
    vmin = float(np.nanpercentile(amp, 1.0))
    vmax = float(np.nanpercentile(amp, 99.0))
    if vmax <= vmin:
        vmax = vmin + 1.0
    im0 = axes[0, 0].imshow(amp, aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("image = |chip|")
    axes[0, 0].set_xlabel("range")
    axes[0, 0].set_ylabel("azimuth")
    plt.colorbar(im0, ax=axes[0, 0])

    amp_fft = np.abs(image_azfft)
    log_amp = 20.0 * np.log10(amp_fft + np.finfo(np.float64).eps)
    vmax_fft = float(np.nanpercentile(log_amp, 99.5))
    vmin_fft = vmax_fft - 40.0
    im1 = axes[0, 1].imshow(
        log_amp, aspect="auto", cmap="viridis",
        vmin=vmin_fft, vmax=vmax_fft,
    )
    axes[0, 1].set_title("image_azfft = |FFT_az(chip)|  (dB, fftshifted)")
    axes[0, 1].set_xlabel("range")
    axes[0, 1].set_ylabel("azimuth-freq bin")
    plt.colorbar(im1, ax=axes[0, 1], label="dB")

    log_shear = 20.0 * np.log10(shear_amp + np.finfo(np.float64).eps)
    vmax_sh = float(np.nanpercentile(log_shear, 99.5))
    vmin_sh = vmax_sh - 40.0
    im2 = axes[1, 0].imshow(
        log_shear, aspect="auto", cmap="viridis",
        vmin=vmin_sh, vmax=vmax_sh,
    )
    axes[1, 0].set_title(r"|S[k]·S*[k+1]|  (shear product magnitude, dB)")
    axes[1, 0].set_xlabel("range")
    axes[1, 0].set_ylabel("azimuth-freq bin (k)")
    plt.colorbar(im2, ax=axes[1, 0], label="dB")

    shear_phase = np.angle(shear)
    im3 = axes[1, 1].imshow(
        shear_phase, aspect="auto", cmap="twilight",
        vmin=-np.pi, vmax=np.pi,
    )
    axes[1, 1].set_title(r"arg{ S[k]·S*[k+1] }  (shear product phase)")
    axes[1, 1].set_xlabel("range")
    axes[1, 1].set_ylabel("azimuth-freq bin (k)")
    plt.colorbar(im3, ax=axes[1, 1], label="rad")

    # Bottom-left: per-Doppler-bin amplitude-weighted shear phase phi[k] = arg{ Σ_r |S[k]·S*[k+1]| · exp(j · arg{S[k]·S*[k+1]}) }, scatter coloured by per-bin coherence.
    k_axis = np.arange(phi_k.size, dtype=np.float64)
    sc = axes[2, 0].scatter(
        k_axis, phi_k, c=coh_k, cmap="viridis",
        s=12, vmin=0.0, vmax=1.0, edgecolor="none",
    )
    plt.colorbar(sc, ax=axes[2, 0], label="coh")
    axes[2, 0].axhline(0.0, color="0.5", lw=0.5, ls="--")
    axes[2, 0].set_ylim(-np.pi, np.pi)
    axes[2, 0].set_xlim(0, phi_k.size - 1)
    axes[2, 0].set_title(
        r"$\varphi[k] = \arg\!\left\{\sum_r S[k,r]\cdot S^*[k+1,r]\right\}$"
        "   (amp-weighted, range-collapsed)"
    )
    axes[2, 0].set_xlabel("azimuth-freq bin (k)")
    axes[2, 0].set_ylabel(r"$\varphi[k]$  [rad]")
    axes[2, 0].grid(True, alpha=0.3)

    # Bottom-right: per-bin coherence.
    axes[2, 1].plot(k_axis, coh_k, color="C0", lw=1.2)
    axes[2, 1].set_ylim(0.0, 1.05)
    axes[2, 1].set_xlim(0, phi_k.size - 1)
    axes[2, 1].set_title(
        r"$\mathrm{coh}[k] = |z[k]| / \sum_r |S[k,r]\cdot S^*[k+1,r]|$"
    )
    axes[2, 1].set_xlabel("azimuth-freq bin (k)")
    axes[2, 1].set_ylabel("coh")
    axes[2, 1].grid(True, alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    _maybe_close(fig)


# --------------------------------------------------------------------- # cluster_targets: sub-aperture-mean/variance target detection. --------------------------------------------------------------------- # Self-contained pipeline that turns the sub-aperture mean and variance arrays (as produced by ``compute_subapertures(...).mean(0)`` / ``.var(0)``, or read from a ``*_subapertures.npz`` written by ``scripts/save_subapertures.py``) into one axis-aligned bounding box per detected physical target. Runs entirely on the decimated sub-aperture grid; no SLC required. Pipeline:   1. CoV² gate           mask = (sub_var / (sub_mean² + eps)                                   > cov_th_mult · median(cov²))   2. Isolated-pixel      density in a 3 m × 3 m window ≥ 0.30   3. Seed peaks          local-amplitude-max + (N_TAW × N_TRL) window                          density > 0.5   4. Clustering          ``clustering.cluster_peaks`` with the                          mass-quantile tightening OFF   5. Grow-and-recenter   per cluster, seed = brightest peak; box is                          grown with a density + rescue gate, always                          constrained to contain the cluster's peak                          bounding rectangle so the amp-CoM recentre                          can't drift onto a brighter neighbour   6. Post-growth extend  +5% × h per side × N_steps at a looser                          density gate   7. (optional) CA-CFAR  on the grown boxes; disabled by default See ``scripts/view_subaps_cov.py`` for the same pipeline exposed as a standalone CLI.
from scipy.ndimage import maximum_filter  # noqa: E402  (uniform_filter already imported above)

_CT_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_CT_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_CT_SCRIPT_DIR))
from clustering import cluster_peaks  # noqa: E402


def _ct_grow_boxes(
    peaks_yx: np.ndarray,
    mask: np.ndarray,
    amp: np.ndarray,
    initial_hw: tuple[int, int],
    az_step: int = 1,
    rg_step: int = 1,
    density_threshold: float = 0.15,
    az_tail: int = 3,
    rg_tail: int = 1,
    az_rescue_lookahead: int = 10,
    az_rescue_lookback: int = 10,
    az_rescue_threshold: float = 0.35,
    max_h: int | None = None,
    max_w: int | None = None,
    max_iter: int = 2000,
    must_contain_yxyx: np.ndarray | None = None,
) -> np.ndarray:
    """Private helper for :func:`cluster_targets` — a self-contained
    variant of :func:`grow_and_recenter_boxes` with two differences:

    1. Takes a per-seed inclusive rectangle ``must_contain_yxyx[k]`` =
       (py_lo, py_hi, px_lo, px_hi) that the grown box is always
       shifted (or grown, if size preservation isn't enough) to
       contain. Used to pin the grown box around a cluster's full
       peak bounding rectangle so the amp-CoM recentre step cannot
       drift onto a neighbouring brighter target.
    2. When ``must_contain_yxyx`` is provided, the post-recentre
       FULL-BOX density check is bypassed — the strip-based edge
       tests still gate every growth step, but a large-cluster box
       that only happens to have low global density is no longer
       killed on the first iteration.

    Returns
    -------
    out : (K, 4) int64 array of (y_lo, y_hi, x_lo, x_hi) INCLUSIVE
        (i.e. edge-based, unlike :func:`grow_and_recenter_boxes` which
        returns centre-based ``(y_c, x_c, h, w)``).
    """
    H, W = mask.shape
    h0, w0 = initial_hw
    K = len(peaks_yx)
    out = np.zeros((K, 4), dtype=np.int64)
    if K == 0:
        return out

    def _density_ok(yl, yh, xl, xh, threshold=None):
        sub = mask[yl:yh, xl:xh]
        if sub.size == 0:
            return False
        th = density_threshold if threshold is None else threshold
        return float(sub.sum()) >= th * sub.size

    def _contain_rect(y_lo, y_hi, x_lo, x_hi, k):
        """Shift-or-grow (EXCLUSIVE y_hi/x_hi) to keep the k-th peak
        rectangle inside. Preserves box size when possible."""
        if must_contain_yxyx is None:
            return y_lo, y_hi, x_lo, x_hi
        py_lo, py_hi, px_lo, px_hi = (int(v) for v in must_contain_yxyx[k])
        h = y_hi - y_lo
        peak_h = py_hi - py_lo + 1
        if h >= peak_h:
            if y_lo > py_lo:
                shift = y_lo - py_lo
                y_lo -= shift
                y_hi -= shift
            if y_hi <= py_hi:
                shift = py_hi + 1 - y_hi
                y_lo += shift
                y_hi += shift
        else:
            if y_lo > py_lo:
                y_lo = py_lo
            if y_hi <= py_hi:
                y_hi = py_hi + 1
        w = x_hi - x_lo
        peak_w = px_hi - px_lo + 1
        if w >= peak_w:
            if x_lo > px_lo:
                shift = x_lo - px_lo
                x_lo -= shift
                x_hi -= shift
            if x_hi <= px_hi:
                shift = px_hi + 1 - x_hi
                x_lo += shift
                x_hi += shift
        else:
            if x_lo > px_lo:
                x_lo = px_lo
            if x_hi <= px_hi:
                x_hi = px_hi + 1
        y_lo = max(0, y_lo)
        y_hi = min(H, y_hi)
        x_lo = max(0, x_lo)
        x_hi = min(W, x_hi)
        return y_lo, y_hi, x_lo, x_hi

    for k, (y0, x0) in enumerate(peaks_yx):
        y_lo = max(0, int(y0) - h0 // 2)
        y_hi = min(H, y_lo + h0)
        x_lo = max(0, int(x0) - w0 // 2)
        x_hi = min(W, x_lo + w0)
        y_lo, y_hi, x_lo, x_hi = _contain_rect(y_lo, y_hi, x_lo, x_hi, k)

        for _ in range(max_iter):
            grew_az = False

            ny_lo = y_lo - az_step
            new_h = y_hi - ny_lo
            over_cap = max_h is not None and new_h > max_h
            if not over_cap and ny_lo >= 0:
                tail_hi = min(y_lo + az_tail, y_hi)
                if _density_ok(ny_lo, tail_hi, x_lo, x_hi):
                    y_lo = ny_lo
                    grew_az = True
                else:
                    look_lo = max(y_lo - az_rescue_lookahead, 0)
                    tail_hi_resc = min(y_lo + az_rescue_lookback, y_hi)
                    if _density_ok(look_lo, tail_hi_resc, x_lo, x_hi,
                                   threshold=az_rescue_threshold):
                        y_lo = ny_lo
                        grew_az = True

            ny_hi = y_hi + az_step
            new_h = ny_hi - y_lo
            over_cap = max_h is not None and new_h > max_h
            if not over_cap and ny_hi <= H:
                tail_lo = max(y_hi - az_tail, y_lo)
                if _density_ok(tail_lo, ny_hi, x_lo, x_hi):
                    y_hi = ny_hi
                    grew_az = True
                else:
                    look_hi = min(y_hi + az_rescue_lookahead, H)
                    tail_lo_resc = max(y_hi - az_rescue_lookback, y_lo)
                    if _density_ok(tail_lo_resc, look_hi, x_lo, x_hi,
                                   threshold=az_rescue_threshold):
                        y_hi = ny_hi
                        grew_az = True

            grew_rg = False
            if not grew_az:
                nx_lo = x_lo - rg_step
                new_w = x_hi - nx_lo
                over_cap = max_w is not None and new_w > max_w
                tail_hi_x = min(x_lo + rg_tail, x_hi)
                if (
                    not over_cap and nx_lo >= 0
                    and _density_ok(y_lo, y_hi, nx_lo, tail_hi_x)
                ):
                    x_lo = nx_lo
                    grew_rg = True

                nx_hi = x_hi + rg_step
                new_w = nx_hi - x_lo
                over_cap = max_w is not None and new_w > max_w
                tail_lo_x = max(x_hi - rg_tail, x_lo)
                if (
                    not over_cap and nx_hi <= W
                    and _density_ok(y_lo, y_hi, tail_lo_x, nx_hi)
                ):
                    x_hi = nx_hi
                    grew_rg = True

            if not (grew_az or grew_rg):
                break

            # Amp-CoM recentre (preserves current h × w).
            h_cur = y_hi - y_lo
            w_cur = x_hi - x_lo
            sub_amp = amp[y_lo:y_hi, x_lo:x_hi]
            total = float(sub_amp.sum())
            if total > 0:
                row_marg = sub_amp.sum(axis=1)
                col_marg = sub_amp.sum(axis=0)
                yy = np.arange(sub_amp.shape[0])
                xx = np.arange(sub_amp.shape[1])
                dy = float((row_marg * yy).sum() / total)
                dx = float((col_marg * xx).sum() / total)
                y_com = y_lo + int(round(dy))
                x_com = x_lo + int(round(dx))
                y_lo = max(0, y_com - h_cur // 2)
                y_hi = min(H, y_lo + h_cur)
                y_lo = max(0, y_hi - h_cur)
                x_lo = max(0, x_com - w_cur // 2)
                x_hi = min(W, x_lo + w_cur)
                x_lo = max(0, x_hi - w_cur)

            y_lo, y_hi, x_lo, x_hi = _contain_rect(y_lo, y_hi, x_lo, x_hi, k)

            # Full-box density safety net (bypassed when a must-contain rect is present — the strip gate decides).
            if must_contain_yxyx is None:
                if not _density_ok(y_lo, y_hi, x_lo, x_hi):
                    break

        out[k, 0] = y_lo
        out[k, 1] = y_hi - 1
        out[k, 2] = x_lo
        out[k, 3] = x_hi - 1
    return out


def _ct_extend_boxes_az(
    boxes: np.ndarray,
    mask: np.ndarray,
    *,
    step_frac: float = 0.05,
    max_steps: int = 4,
    density_threshold: float = 0.05,
    max_h: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Private helper for :func:`cluster_targets` — az-only edge
    extension. Same idea as :func:`extend_boxes_azimuth_strong_signal`
    but works on INCLUSIVE-edge boxes ``(y_lo, y_hi, x_lo, x_hi)``.
    Tries to push top/bottom outward by ``step_frac × h_entry`` rows,
    up to ``max_steps`` times per side, accepting each step iff the
    strip's mask density ≥ ``density_threshold``.

    Returns
    -------
    out : (K, 4) int64 array — same format as the input.
    """
    if len(boxes) == 0:
        return boxes
    H, _ = mask.shape
    out = np.array(boxes, dtype=np.int64, copy=True)
    n_top = n_bot = total_extra = 0
    for k in range(len(out)):
        y_lo_i, y_hi_i, x_lo_i, x_hi_i = (int(v) for v in out[k])
        y_lo = y_lo_i
        y_hi = y_hi_i + 1
        h_entry = y_hi - y_lo
        step = max(1, int(round(step_frac * h_entry)))

        for _ in range(max_steps):
            ny_lo = y_lo - step
            if ny_lo < 0:
                break
            if max_h is not None and (y_hi - ny_lo) > max_h:
                break
            strip = mask[ny_lo:y_lo, x_lo_i:x_hi_i + 1]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_lo = ny_lo
                n_top += 1
            else:
                break

        for _ in range(max_steps):
            ny_hi = y_hi + step
            if ny_hi > H:
                break
            if max_h is not None and (ny_hi - y_lo) > max_h:
                break
            strip = mask[y_hi:ny_hi, x_lo_i:x_hi_i + 1]
            if strip.size == 0:
                break
            if float(strip.sum()) >= density_threshold * strip.size:
                y_hi = ny_hi
                n_bot += 1
            else:
                break

        total_extra += (y_hi - y_lo) - h_entry
        out[k, 0] = y_lo
        out[k, 1] = y_hi - 1
    if verbose:
        print(
            f"  az-extend: step={step_frac:.0%}×h, max_steps={max_steps}, "
            f"density≥{density_threshold:g}: {n_top}/{len(out)} top-steps, "
            f"{n_bot}/{len(out)} bottom-steps, "
            f"+{total_extra} az rows total"
        )
    return out


@dataclass
class ClusterTargetsResult:
    """Return type of :func:`cluster_targets`. All arrays live on the
    decimated sub-aperture grid whose pixel spacings are
    ``az_m_per_px × rg_m_per_px`` metres.
    """
    boxes: np.ndarray            # (N, 4) inclusive (y_lo, y_hi, x_lo, x_hi)
    labels: np.ndarray           # (K,) peak → cluster id in [0, N)
    peaks_yx: np.ndarray         # (K, 2) all seed peaks
    seed_peaks_yx: np.ndarray    # (N, 2) brightest peak per cluster
    mask_filt: np.ndarray        # (H, W) isolated-pixel-filtered CoV mask
    sub_mean: np.ndarray         # (H, W)
    sub_var: np.ndarray          # (H, W)
    az_m_per_px: float
    rg_m_per_px: float
    n_subaperture: int | None
    n_range_looks: int | None
    threshold: float             # applied CoV² threshold (dark path for
    #                              exclude / higher-th modes)


def cluster_targets(
    sub_mean: np.ndarray,
    sub_var: np.ndarray,
    *,
    # ---- Grid metadata (decimated sub-aperture grid) ----
    az_m_per_px: float,           # metres per row (= az_m_per_px_s_degraded × N_sub)
    rg_m_per_px: float,           # metres per column (= rg_m_per_px_s_degraded)
    n_subaperture: int | None = None,
    n_range_looks: int | None = None,
    # ---- CoV gate ----
    cov_th_mult: float = 1.5,
    bright_mode: str = "off",
    bright_cov_th_mult: float = 3.0,
    dark_amp_percentile: float = 50.0,
    eps: float = 1e-12,
    # ---- Isolated-pixel filter ----
    filter_target_size_m: float = 3.0,
    filter_min_density: float = 0.30,
    # ---- Seed detection ----
    seed_density_th: float = 0.5,
    n_target_azimuth_width: int = 100,
    n_target_range_length: int = 3,
    # ---- Clustering ----
    az_search_m: float = 100.0,
    rg_search_m: float = 10.0,
    enable_arc_split: bool = False,
    max_peak_gap_az_m: float = float("inf"),
    # ---- Grow-and-recenter (relaxed-az defaults) ----
    grow_density_th: float = 0.15,
    grow_az_step_m: float = 10.0,
    grow_az_tail_m: float | None = None,   # None → same as step
    grow_rescue_lookahead_m: float = 200.0,
    grow_rescue_lookback_m: float = 200.0,
    grow_rescue_th: float = 0.35,
    grow_max_h_m: float = 300.0,
    grow_max_w_m: float = 130.0,
    # ---- Post-growth az extension (relaxed-az defaults) ----
    extend_density_th: float = 0.05,
    extend_max_steps: int = 4,
    # ---- Optional CA-CFAR ----
    cfar_snr_th: float = 0.0,
    cfar_range_bins: int = 2,
    verbose: bool = True,
) -> ClusterTargetsResult:
    """Detect targets on the sub-aperture mean/variance and return one
    grown axis-aligned bounding box per cluster.

    Parameters
    ----------
    sub_mean, sub_var : (H, W) float arrays
        Per-pixel mean and variance of the sub-aperture amplitude stack
        on the decimated grid (i.e. ``compute_subapertures(...).mean(0)``
        and ``.var(0)``). Same convention as ``save_subapertures.py``.
    az_m_per_px, rg_m_per_px : float
        Physical pixel spacings of the decimated grid, in metres.
        For a sub-aperture stack with ``N_subaperture`` looks over
        ``s_degraded``: ``az_m_per_px = az_m_per_px_s_degraded * N_sub``
        and ``rg_m_per_px = rg_m_per_px_s_degraded``.
    n_subaperture, n_range_looks : int, optional
        Metadata carried through into the returned bundle (for logging
        / downstream use only — not used by the algorithm).

    See the section header above for the pipeline overview. Every knob
    defaults to the tuned relaxed-azimuth values.
    """
    _log = print if verbose else (lambda *_a, **_k: None)

    # Promote to float64 once here so downstream arithmetic and the CoV ratio don't underflow on float16/32 inputs.
    sub_mean = np.asarray(sub_mean, dtype=np.float64)
    sub_var = np.asarray(sub_var, dtype=np.float64)
    if sub_mean.shape != sub_var.shape or sub_mean.ndim != 2:
        raise ValueError(
            f"sub_mean and sub_var must be 2-D and same shape; got "
            f"sub_mean.shape={sub_mean.shape}, sub_var.shape={sub_var.shape}"
        )
    az_m_dec = float(az_m_per_px)
    rg_m_dec = float(rg_m_per_px)
    n_sub = n_subaperture
    n_rl = n_range_looks

    _log(
        f"cluster_targets: shape={sub_mean.shape}, "
        f"az={az_m_dec:g} m/row × rg={rg_m_dec:g} m/px "
        f"(N_sub={n_sub}, N_RL={n_rl})"
    )

    # ---- CoV gate --------------------------------------------------
    cov_sq = sub_var / (sub_mean ** 2 + eps)
    th_applied: float
    if bright_mode == "off":
        th_applied = cov_th_mult * float(np.median(cov_sq))
        mask_dec = (cov_sq > th_applied).astype(np.float32)
    else:
        amp_dark_thresh = float(np.percentile(sub_mean, dark_amp_percentile))
        dark_dec = sub_mean <= amp_dark_thresh
        cov_sq_dark = cov_sq[dark_dec] if dark_dec.any() else cov_sq
        median_dark = float(np.median(cov_sq_dark))
        th_dark = cov_th_mult * median_dark
        th_applied = th_dark
        if bright_mode == "exclude":
            mask_dec = (dark_dec & (cov_sq > th_dark)).astype(np.float32)
        elif bright_mode == "higher-th":
            bright_dec = ~dark_dec
            median_bright = (
                float(np.median(cov_sq[bright_dec])) if bright_dec.any()
                else median_dark
            )
            th_bright = bright_cov_th_mult * median_bright
            mask_dec = (
                (dark_dec & (cov_sq > th_dark))
                | (bright_dec & (cov_sq > th_bright))
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown bright_mode: {bright_mode!r}")

    n_kept = int(mask_dec.sum())
    n_total = mask_dec.size
    _log(
        f"  CoV gate (mode='{bright_mode}', mult={cov_th_mult:g}): "
        f"kept {n_kept}/{n_total} = {100 * n_kept / n_total:.2f}%"
    )

    # ---- Isolated-pixel filter -------------------------------------
    az_filt_win = max(1, int(round(filter_target_size_m / az_m_dec)))
    rg_filt_win = max(1, int(round(filter_target_size_m / rg_m_dec)))
    density = uniform_filter(
        mask_dec, size=(az_filt_win, rg_filt_win), mode="constant",
    )
    mask_filt = (
        mask_dec.astype(bool) & (density >= filter_min_density)
    ).astype(np.float32)
    n_kept_after = int(mask_filt.sum())
    _log(
        f"  isolated-pixel filter ({filter_target_size_m:g} m × "
        f"{filter_target_size_m:g} m, density ≥ {filter_min_density:.2f}): "
        f"kept {n_kept_after}/{n_kept} = "
        f"{100 * n_kept_after / max(n_kept, 1):.2f}%"
    )

    # ---- Seed peaks -----------------------------------------------
    az_seed_win = max(
        1, int(round(n_target_azimuth_width / max(n_sub or 16, 1))),
    )
    rg_seed_win = int(n_target_range_length)
    box_size = (az_seed_win, rg_seed_win)
    box_area = az_seed_win * rg_seed_win

    amp_proxy = sub_mean.astype(np.float32)
    det_count = uniform_filter(
        mask_filt.astype(np.float32), size=box_size, mode="constant",
    ) * box_area
    amp_max = maximum_filter(amp_proxy, size=box_size, mode="constant")
    boundary = (
        (mask_filt == 1)
        & (det_count > seed_density_th * box_area)
        & (amp_proxy == amp_max)
    )
    peaks_yx = np.argwhere(boundary).astype(np.int64)
    _log(
        f"  seed peaks: window=({az_seed_win},{rg_seed_win}), "
        f"density > {seed_density_th:g} → {len(peaks_yx)} peaks"
    )

    # ---- Clustering (mass-quantile tightening OFF) ----------------
    labels = np.empty(0, dtype=np.int64)
    boxes = np.empty((0, 4), dtype=np.int64)
    if len(peaks_yx):
        arc_az_thr_m = 200.0 if enable_arc_split else float("inf")
        t0 = time.time()
        labels, boxes = cluster_peaks(
            mask_filt.astype(np.uint8), peaks_yx,
            az_m_per_px=az_m_dec, rg_m_per_px=rg_m_dec,
            az_search_m=az_search_m, rg_search_m=rg_search_m,
            arc_az_thr_m=arc_az_thr_m, arc_rg_max_m=20.0,
            max_peak_gap_az_m=max_peak_gap_az_m,
            tighten_boxes=False,
            enforce_max_size=False,
        )
        _log(
            f"  cluster_peaks: {len(peaks_yx)} peaks → "
            f"{int(labels.max()) + 1} clusters in {time.time() - t0:.2f} s"
        )

    # ---- Cluster → grow -----------------------------------------
    seed_peaks_yx = np.empty((0, 2), dtype=np.int64)
    if len(peaks_yx) and len(boxes):
        n_clusters_pre = int(labels.max()) + 1
        az_tail_m_eff = grow_az_step_m if grow_az_tail_m is None else grow_az_tail_m
        az_step_px = max(1, int(round(grow_az_step_m / az_m_dec)))
        az_tail_px = max(1, int(round(az_tail_m_eff / az_m_dec)))
        look_ahead_px = max(1, int(round(grow_rescue_lookahead_m / az_m_dec)))
        look_back_px = max(1, int(round(grow_rescue_lookback_m / az_m_dec)))
        max_h_px = max(1, int(round(grow_max_h_m / az_m_dec)))
        max_w_px = max(1, int(round(grow_max_w_m / rg_m_dec)))

        seed_list: list[np.ndarray] = []
        must_contain_list: list[tuple[int, int, int, int]] = []
        for c in range(n_clusters_pre):
            mem = np.where(labels == c)[0]
            if len(mem) == 0:
                continue
            mp = peaks_yx[mem]
            amps_at_peaks = amp_proxy[
                mp[:, 0].astype(np.int64), mp[:, 1].astype(np.int64),
            ]
            best_local = int(np.argmax(amps_at_peaks))
            seed_list.append(mp[best_local])
            must_contain_list.append((
                int(mp[:, 0].min()), int(mp[:, 0].max()),
                int(mp[:, 1].min()), int(mp[:, 1].max()),
            ))
        seed_peaks_yx = np.asarray(seed_list, dtype=np.int64)
        must_contain_yxyx = np.asarray(must_contain_list, dtype=np.int64)

        t_g = time.time()
        boxes = _ct_grow_boxes(
            seed_peaks_yx,
            mask=mask_filt.astype(np.float32),
            amp=amp_proxy,
            initial_hw=(az_seed_win, rg_seed_win),
            az_step=az_step_px,
            rg_step=1,
            density_threshold=grow_density_th,
            az_tail=az_tail_px,
            rg_tail=1,
            az_rescue_lookahead=look_ahead_px,
            az_rescue_lookback=look_back_px,
            az_rescue_threshold=grow_rescue_th,
            max_h=max_h_px,
            max_w=max_w_px,
            must_contain_yxyx=must_contain_yxyx,
        )
        boxes = _ct_extend_boxes_az(
            boxes,
            mask=mask_filt.astype(np.float32),
            step_frac=0.05,
            max_steps=extend_max_steps,
            density_threshold=extend_density_th,
            max_h=max_h_px,
            verbose=verbose,
        )
        gh = boxes[:, 1] - boxes[:, 0] + 1
        gw = boxes[:, 3] - boxes[:, 2] + 1
        _log(
            f"  grow-and-recenter: {n_clusters_pre} clusters → "
            f"{len(boxes)} grown boxes in {time.time() - t_g:.2f} s "
            f"(step={az_step_px}px [{grow_az_step_m:g} m], "
            f"tail={az_tail_px}px [{az_tail_m_eff:g} m"
            + (" ← =step" if grow_az_tail_m is None else "")
            + f"], density≥{grow_density_th:g}, "
            f"rescue lookahead={look_ahead_px}px "
            f"[{grow_rescue_lookahead_m:g} m], "
            f"cap h×w={max_h_px}×{max_w_px}px "
            f"[{grow_max_h_m:g}×{grow_max_w_m:g} m])"
        )
        _log(
            f"    grown az: min={gh.min()}, med={int(np.median(gh))}, "
            f"max={gh.max()} px  "
            f"({gh.min() * az_m_dec:.0f} .. {gh.max() * az_m_dec:.0f} m)"
        )
        _log(
            f"    grown rg: min={gw.min()}, med={int(np.median(gw))}, "
            f"max={gw.max()} px  "
            f"({gw.min() * rg_m_dec:.0f} .. {gw.max() * rg_m_dec:.0f} m)"
        )

    # ---- Optional CA-CFAR filter ---------------------------------
    if len(boxes) and cfar_snr_th > 0:
        _, W_mask = mask_filt.shape
        n_guard = int(max(1, cfar_range_bins))
        keep = np.ones(len(boxes), dtype=bool)
        for c in range(len(boxes)):
            y_lo, y_hi, x_lo, x_hi = (int(v) for v in boxes[c])
            box_amp = amp_proxy[y_lo:y_hi + 1, x_lo:x_hi + 1]
            box_msk = mask_filt[y_lo:y_hi + 1, x_lo:x_hi + 1]
            sig_pix = box_amp[box_msk == 1]
            if sig_pix.size == 0:
                sig_pix = box_amp
                if sig_pix.size == 0:
                    keep[c] = False
                    continue
            signal = float(sig_pix.mean())

            prev_lo = max(0, x_lo - n_guard)
            succ_hi = min(W_mask, x_hi + 1 + n_guard)
            noise_means: list[float] = []
            prev_strip = amp_proxy[y_lo:y_hi + 1, prev_lo:x_lo]
            succ_strip = amp_proxy[y_lo:y_hi + 1, x_hi + 1:succ_hi]
            if prev_strip.size:
                noise_means.append(float(prev_strip.mean()))
            if succ_strip.size:
                noise_means.append(float(succ_strip.mean()))
            if not noise_means:
                continue
            snr = signal / max(min(noise_means), 1e-12)
            if snr < cfar_snr_th:
                keep[c] = False

        n_before = len(boxes)
        boxes = boxes[keep]
        keep_pk = keep[labels]
        labels = labels[keep_pk]
        peaks_yx = peaks_yx[keep_pk]
        if len(labels):
            _, remap = np.unique(labels, return_inverse=True)
            labels = remap.astype(np.int64)
        seed_peaks_yx = seed_peaks_yx[keep]
        _log(
            f"  CA-CFAR (snr_th={cfar_snr_th:g}): "
            f"kept {len(boxes)}/{n_before} clusters"
        )

    return ClusterTargetsResult(
        boxes=boxes.astype(np.int64),
        labels=labels.astype(np.int64),
        peaks_yx=peaks_yx.astype(np.int64),
        seed_peaks_yx=seed_peaks_yx.astype(np.int64),
        mask_filt=mask_filt,
        sub_mean=sub_mean,
        sub_var=sub_var,
        az_m_per_px=az_m_dec,
        rg_m_per_px=rg_m_dec,
        n_subaperture=n_sub,
        n_range_looks=n_rl,
        threshold=th_applied,
    )


def main() -> None:
    """Run the shear-averaging pipeline using parameters from the JSON config.

    All tunable parameters live in ``shear_averaging_config.json`` next to
    this file (override the path by passing it as the first CLI arg). The
    module-level ``_CFG`` is loaded before ``matplotlib`` is imported so the
    ``show`` field can switch the backend; every other field is consumed via
    ``cfg = _CFG`` below.
    """
    cfg = _CFG
    print(f"[config] loaded {_CFG_PATH}")

    # Master save-verbosity switch (see `output.debug` help in the JSON config). When True every diagnostic artefact is written; when False only the three keeper files (`{stem}_kept_vs_eliminated.png`, `{stem}_af_before.png`, `{stem}_af_after.png`) land on disk. Compute the boolean once so the checks below stay cheap and readable.
    _save_all: bool = bool(cfg.debug)
    print(
        f"[mode] output.debug={_save_all} → "
        f"{'save-all diagnostics' if _save_all else 'minimal (kept_vs_eliminated + af_before + af_after)'}"
    )

    debug_rois: list[tuple[int, int, int, int]] = []
    if cfg.debug_roi:
        # Config supplies either a list of [y_lo, y_hi, x_lo, x_hi] lists (new form) or a semicolon-separated string in the legacy CLI form. Normalise both to the list-of-lists shape so the parsing loop below stays identical.
        raw_rois = cfg.debug_roi
        if isinstance(raw_rois, str):
            raw_rois = [
                [int(s) for s in chunk.strip().split(",")]
                for chunk in raw_rois.split(";")
                if chunk.strip()
            ]
        for rect in raw_rois:
            try:
                _y_lo, _y_hi, _x_lo, _x_hi = (int(v) for v in rect)
            except (ValueError, TypeError):
                print(
                    f"  (warning) debug_roi could not parse "
                    f"rectangle {rect!r}; skipped."
                )
                continue
            debug_rois.append((_y_lo, _y_hi, _x_lo, _x_hi))
        for i, (_y_lo, _y_hi, _x_lo, _x_hi) in enumerate(debug_rois):
            print(
                f"[roi #{i}] tracing boxes overlapping y={_y_lo}..{_y_hi}, "
                f"x={_x_lo}..{_x_hi}"
            )

    def _roi_hits(boxes_in: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
        """Indices of boxes overlapping the given ROI rectangle (or empty)."""
        if len(boxes_in) == 0:
            return np.empty(0, dtype=np.int64)
        y_lo, y_hi, x_lo, x_hi = roi
        y0 = boxes_in[:, 0] - boxes_in[:, 2] / 2
        y1 = boxes_in[:, 0] + boxes_in[:, 2] / 2
        x0 = boxes_in[:, 1] - boxes_in[:, 3] / 2
        x1 = boxes_in[:, 1] + boxes_in[:, 3] / 2
        ov = (y1 >= y_lo) & (y0 <= y_hi) & (x1 >= x_lo) & (x0 <= x_hi)
        return np.where(ov)[0]

    def _roi_print(label: str, boxes_in: np.ndarray) -> None:
        """Emit one log line per ROI showing how many boxes overlap it
        after the current pipeline stage. ROIs are numbered by their
        order on the CLI (`#0`, `#1`, …) so the stream is easy to
        follow when tracing several targets in the same run.
        """
        if not debug_rois:
            return
        for i, roi in enumerate(debug_rois):
            hit_idx = _roi_hits(boxes_in, roi)
            if hit_idx.size == 0:
                print(
                    f"[roi #{i}] after {label:30s}: 0 boxes overlap ROI"
                )
                continue
            bits = []
            for k in hit_idx[:6]:
                y_c, x_c, h, w = boxes_in[k]
                bits.append(
                    f"#{int(k)} y={int(y_c)} x={int(x_c)} "
                    f"h={int(h)} w={int(w)}"
                )
            extra = f" + {hit_idx.size - 6} more" if hit_idx.size > 6 else ""
            print(
                f"[roi #{i}] after {label:30s}: {hit_idx.size} boxes "
                f"overlap ROI → {', '.join(bits)}{extra}"
            )

    sw = _Stopwatch()

    suffix = cfg.path.suffix.lower()
    iceye_md: dict | None = None
    if suffix in (".tif", ".tiff"):
        sidecar_json = cfg.path.with_suffix(".json")
        if not sidecar_json.exists():
            raise FileNotFoundError(
                f"GeoTIFF input {cfg.path.name} requires a sidecar "
                f"metadata JSON at {sidecar_json}, but none was found."
            )
        iceye_md = _load_iceye_sidecar_metadata(sidecar_json)
        left = iceye_md["sar_observation_direction"].lower() == "left"
        s = _load_slc_from_tiff(cfg.path, left=left)
        print(
            f"Loaded: {cfg.path.name}  shape={s.shape}  dtype={s.dtype}  "
            f"(left-look={left}, sidecar={sidecar_json.name})"
        )
        print("  ICEYE sidecar metadata:")
        for k in _ICEYE_JSON_FIELDS:
            print(f"    {k} = {iceye_md[k]}")

        # PRF is read via `cfg.prf` by everything downstream (`compute_box_phase_stats`, COM velocity fit, etc.). A null config value ⇒ take from the sidecar; a numeric value ⇒ user override wins and we log the mismatch.
        if cfg.prf is None:
            cfg.prf = float(iceye_md["iceye_acquisition_prf"])
            print(f"  PRF from sidecar: cfg.prf = {cfg.prf:g} Hz")
        else:
            print(
                f"  PRF override in config ({cfg.prf:g} Hz) keeps precedence "
                f"over sidecar value ({iceye_md['iceye_acquisition_prf']:g} Hz)"
            )
    elif suffix == ".npz":
        # Self-contained ICEYE patch bundle: `data` holds the complex SLC (already shadows-down, axis 0 = azimuth) and the remaining archive keys mirror the flat dict `_load_iceye_sidecar_metadata` builds from a JSON sidecar, so the rest of the pipeline treats `.npz` exactly like `.npy + --metadata-from <json>`.
        with np.load(cfg.path, allow_pickle=False) as arch:
            if "data" not in arch.files:
                raise KeyError(
                    f".npz bundle {cfg.path} is missing the 'data' array."
                )
            s = np.ascontiguousarray(arch["data"])
            missing = [k for k in _ICEYE_JSON_FIELDS if k not in arch.files]
            if missing:
                raise KeyError(
                    f".npz bundle {cfg.path} is missing metadata keys: "
                    f"{missing}. Expected all of {_ICEYE_JSON_FIELDS}."
                )
            iceye_md = {
                "sar_resolution_range": float(arch["sar_resolution_range"]),
                "sar_resolution_azimuth": float(arch["sar_resolution_azimuth"]),
                "sar_pixel_spacing_range": float(arch["sar_pixel_spacing_range"]),
                "sar_pixel_spacing_azimuth": float(arch["sar_pixel_spacing_azimuth"]),
                "iceye_acquisition_prf": float(arch["iceye_acquisition_prf"]),
                "start_datetime": str(arch["start_datetime"]),
                "end_datetime": str(arch["end_datetime"]),
            }
            # Optional geometry fields for the Doppler-domain v_a-from-range-walk mapping. Bundles baked before the mapping was added won't have these keys — the AF path then falls back to sensible ICEYE defaults instead of failing, so old bundles keep loading unchanged.
            for opt_key in (
                "iceye_processing_prf",
                "iceye_range",
                "sar_center_frequency",
            ):
                if opt_key in arch.files:
                    iceye_md[opt_key] = float(arch[opt_key])
        print(
            f"Loaded: {cfg.path.name}  shape={s.shape}  dtype={s.dtype}  "
            "(bundled ICEYE metadata)"
        )
        print("  ICEYE bundled metadata:")
        for k in _ICEYE_JSON_FIELDS:
            print(f"    {k} = {iceye_md[k]}")
        if cfg.prf is None:
            cfg.prf = float(iceye_md["iceye_acquisition_prf"])
            print(f"  PRF from bundle: cfg.prf = {cfg.prf:g} Hz")
        else:
            print(
                f"  PRF override in config ({cfg.prf:g} Hz) keeps precedence "
                f"over bundled value ({iceye_md['iceye_acquisition_prf']:g} Hz)"
            )
    elif suffix == ".npy":
        s = np.load(cfg.path)
        print(f"Loaded: {cfg.path.name}  shape={s.shape}  dtype={s.dtype}")
        if cfg.metadata_from is not None:
            md_suffix = cfg.metadata_from.suffix.lower()
            if md_suffix in (".tif", ".tiff"):
                sidecar_json = cfg.metadata_from.with_suffix(".json")
            elif md_suffix == ".json":
                sidecar_json = cfg.metadata_from
            else:
                raise ValueError(
                    f"--metadata-from {cfg.metadata_from!r} must point at "
                    "a .tif / .tiff (its sidecar `<stem>.json` is loaded) "
                    "or directly at a .json file."
                )
            if not sidecar_json.exists():
                raise FileNotFoundError(
                    f"--metadata-from points at {cfg.metadata_from} but "
                    f"the sidecar JSON {sidecar_json} does not exist."
                )
            iceye_md = _load_iceye_sidecar_metadata(sidecar_json)
            print(
                f"  metadata sidecar: {sidecar_json} (loaded for .npy patch)"
            )
            print("  ICEYE sidecar metadata:")
            for k in _ICEYE_JSON_FIELDS:
                print(f"    {k} = {iceye_md[k]}")
            if cfg.prf is None:
                cfg.prf = float(iceye_md["iceye_acquisition_prf"])
                print(f"  PRF from sidecar: cfg.prf = {cfg.prf:g} Hz")
            else:
                print(
                    f"  PRF override in config ({cfg.prf:g} Hz) keeps "
                    "precedence over sidecar value "
                    f"({iceye_md['iceye_acquisition_prf']:g} Hz)"
                )
    else:
        raise ValueError(
            f"Unsupported input extension {suffix!r} for {cfg.path}; "
            "expected .npy, .npz, .tif or .tiff."
        )

    # Fallback: `.npy` without `metadata_from` doesn't provide a PRF; keep the historical 6000 Hz default from the old argparse block so downstream code (COM velocity fit, time-stamp math) can still run.
    if cfg.prf is None:
        cfg.prf = 6000.0
        print(f"  PRF fallback (no sidecar): cfg.prf = {cfg.prf:g} Hz")

    sw.mark(f"load SLC ({suffix}) + sidecar metadata")

    # --- Processing pipeline ---
    min_size_of_target = 1.0

    # Range / azimuth pixel spacing in metres. For the historical .npy patches we keep the hard-coded values that the patch was cropped at; for an ICEYE GeoTIFF we drive them from the sidecar JSON so the same script runs unchanged on other acquisitions.
    if iceye_md is not None:
        range_spacing = iceye_md["sar_pixel_spacing_range"]
        azimuth_spacing = iceye_md["sar_pixel_spacing_azimuth"]
        print(
            f"  spacings from sidecar: "
            f"range_spacing={range_spacing} m, "
            f"azimuth_spacing={azimuth_spacing} m"
        )

    # Aperture duration used by the `smear_max = v_max · T_int` heuristic that caps the max azimuth box length, and by the Theil-Sen COM-walk diagnostic to convert per-subaperture time steps to seconds. Real scenes drive it from the sidecar timestamps (capped at 30 s — past that, `v_max · T_int` grows beyond what is physically plausible for one synthetic aperture); when the sidecar is unavailable (e.g. .npy patches) we fall back to a fixed 20 s so downstream code stays well-defined.
    if iceye_md is not None:
        scene_duration = _iso_duration_seconds(
            iceye_md["start_datetime"], iceye_md["end_datetime"]
        )
        integration_time = float(min(scene_duration, 30.0))
        print(
            f"  integration_time from sidecar: "
            f"end − start = {scene_duration:.2f} s, "
            f"capped at 30 s → {integration_time:.2f} s"
        )
    else:
        integration_time = 20.0
        print(
            f"  integration_time: sidecar not available → "
            f"default {integration_time:.2f} s"
        )

    Number_of_Range_Looks = int(min_size_of_target / range_spacing) #
    N_subaperture = cfg.n_subaperture
   
    
    s_degraded = degrade_range_resolution_range_sum(
        s,
        Number_of_Range_Looks,
        sll_db=55.0,
        nbar=8,
    )
    
   
    subapertures = compute_subapertures(s_degraded, N_subaperture)  # (N, sub_size, N_rg)
    sub_mean = subapertures.mean(axis=0)                            # (sub_size, N_rg)
    sub_var = subapertures.var(axis=0)                              # (sub_size, N_rg)

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
   
    boxes = cluster_result.boxes
    peaks_yx = cluster_result.peaks_yx
    mask_filt = cluster_result.mask_filt
    sub_mean = cluster_result.sub_mean
    sub_var = cluster_result.sub_var

    # --- Intermediate diagnostic figure: sub_mean + cluster_targets ---
    # The mean amplitude across N_subaperture Doppler looks, on the
    # decimated subaperture-statistics grid (sub_size, N_rg), overlaid
    # with the raw cluster_targets boxes (cyan, before any phase
    # filtering) and the seed peaks that seeded each cluster (red ×).
    # Both boxes and peaks live in decimated coords so no rescaling is
    # needed. Saved further down alongside path_main under
    # `<stem>_sub_mean_boxes.png` when --save is set.
    # Skipped in minimal mode (`cfg.debug=False`) — this is a diagnostic-only artefact.
    if _save_all and cfg.save is not None and sub_mean.size:
        # Convert cluster boxes from (y_lo, y_hi, x_lo, x_hi) inclusive
        # to (y_c, x_c, h, w) for `_overlay_boxes`, still on the
        # decimated grid.
        if len(boxes):
            _yl_d = boxes[:, 0].astype(np.int64)
            _yh_d = boxes[:, 1].astype(np.int64)
            _xl_d = boxes[:, 2].astype(np.int64)
            _xh_d = boxes[:, 3].astype(np.int64)
            _h_d = _yh_d - _yl_d + 1
            _w_d = _xh_d - _xl_d + 1
            _boxes_dec_yxhw = np.stack(
                [_yl_d + _h_d // 2, _xl_d + _w_d // 2, _h_d, _w_d], axis=1,
            )
        else:
            _boxes_dec_yxhw = np.empty((0, 4), dtype=np.int64)
        fig_sub_mean_boxes, _ax_sub_mean = plt.subplots(
            1, 1, figsize=(14, 9), constrained_layout=True,
        )
        fig_sub_mean_boxes.suptitle(
            f"sub_mean + cluster_targets boxes — {cfg.path.name}",
            fontsize=11,
        )
        _mu = float(sub_mean.mean())
        _sd = float(sub_mean.std())
        _vmin = max(0.0, _mu - 4.0 * _sd)
        _vmax = min(float(sub_mean.max()), _mu + 4.0 * _sd)
        _im = _ax_sub_mean.imshow(
            sub_mean, cmap="viridis", aspect="auto",
            vmin=_vmin, vmax=_vmax,
        )
        _overlay_boxes(
            _ax_sub_mean, _boxes_dec_yxhw, color="cyan", lw=1.0,
        )
        if len(peaks_yx):
            _ax_sub_mean.scatter(
                peaks_yx[:, 1], peaks_yx[:, 0],
                s=10, marker="x", c="red", linewidths=0.6,
                label=f"seed peaks ({len(peaks_yx)})",
            )
            _ax_sub_mean.legend(loc="upper right", fontsize=8)
        _ax_sub_mean.set_title(
            f"sub_mean (decimated grid: {sub_mean.shape[0]}×"
            f"{sub_mean.shape[1]}), "
            f"{len(_boxes_dec_yxhw)} cluster_targets boxes (cyan), "
            f"{len(peaks_yx)} seed peaks (red ×)"
        )
        _ax_sub_mean.set_xlabel("range pixel (decimated)")
        _ax_sub_mean.set_ylabel(
            f"azimuth pixel (decimated, ×{N_subaperture} = s_degraded row)"
        )
        plt.colorbar(_im, ax=_ax_sub_mean, label="mean |s| over N_sub")
    else:
        fig_sub_mean_boxes = None

    # The upsampled binary mask `mask_full = np.repeat(mask_filt, N_subaperture, axis=0)` (trimmed to s_degraded.shape[0] with any missing rows treated as zero) is no longer materialised anywhere in the pipeline. `filter_boxes_by_phase_residual` runs the detection-count bypass directly on `mask_filt` + `N_subaperture`, and the former `--debug` `fig_mb` mask overlay figure has been retired to keep peak RSS off the ≈1.5 GB scene-wide float32 buffer that panel required.

    # Reconstruct the full-res seed-peak map (`boundary_box`) that the save / plot code expects: one 1-pixel per cluster_targets seed peak on the s_degraded grid. Peaks live on the decimated grid; multiply their y-coord by N_subaperture to project onto s_degraded rows.
    boundary_box = np.zeros(s_degraded.shape, dtype=np.int8)
    if len(peaks_yx):
        peaks_yx_full = np.stack(
            [peaks_yx[:, 0] * N_subaperture, peaks_yx[:, 1]], axis=1,
        ).astype(np.int64)
        _valid_full = (
            (peaks_yx_full[:, 0] >= 0)
            & (peaks_yx_full[:, 0] < s_degraded.shape[0])
            & (peaks_yx_full[:, 1] >= 0)
            & (peaks_yx_full[:, 1] < s_degraded.shape[1])
        )
        boundary_box[
            peaks_yx_full[_valid_full, 0],
            peaks_yx_full[_valid_full, 1],
        ] = 1
    else:
        peaks_yx_full = np.empty((0, 2), dtype=np.int64)

    # Convert cluster boxes from inclusive (y_lo, y_hi, x_lo, x_hi) on the decimated sub-aperture grid to (y_c, x_c, h, w) on the s_degraded grid — the format every downstream filter expects.
    if len(boxes):
        _y_lo_full = boxes[:, 0].astype(np.int64) * N_subaperture
        _y_hi_full = (boxes[:, 1].astype(np.int64) + 1) * N_subaperture - 1
        _x_lo_full = boxes[:, 2].astype(np.int64)
        _x_hi_full = boxes[:, 3].astype(np.int64)
        _h_full = _y_hi_full - _y_lo_full + 1
        _w_full = _x_hi_full - _x_lo_full + 1
        _y_c_full = _y_lo_full + _h_full // 2
        _x_c_full = _x_lo_full + _w_full // 2
        boxes_yxhw = np.stack(
            [_y_c_full, _x_c_full, _h_full, _w_full], axis=1,
        ).astype(np.int64)
    else:
        boxes_yxhw = np.empty((0, 4), dtype=np.int64)

    # Snapshot the pre-filter box set (on the s_degraded grid) so the
    # downstream `fig_kept_vs_eliminated` overlay and the CSV
    # `disposition` column can show which cluster_targets boxes made
    # it all the way through the phase-slope / residual / COM-PP /
    # motion 3-tier / refocus-gain / sub-aperture-sharpness filter
    # chain and which did not. All downstream filters only index-
    # select rows out of `boxes_yxhw`, never mutate them, so every
    # surviving row appears verbatim in this snapshot.
    boxes_yxhw_initial = boxes_yxhw.copy()

    # Per-initial-box detection count. For each box in
    # `boxes_yxhw_initial` we count how many `mask_filt` cells fall
    # inside its footprint. `mask_filt` lives on the decimated
    # sub-aperture grid (sub_size × N_rg), so the box's azimuth
    # range on s_degraded is projected down by N_subaperture; range
    # is not decimated. Result is stored per INITIAL box index so
    # the CSV can report it uniformly for surviving and dropped
    # boxes alike (the mask never changes after cluster_targets, so
    # the count is a stable property of each box footprint).
    def _n_det_in_box(y_c: int, x_c: int, h: int, w: int) -> int:
        sub_size_m, n_rg_m = mask_filt.shape
        y_lo = int(round(y_c - h / 2))
        y_hi = int(round(y_c + h / 2))
        y0_dec = max(0, y_lo // N_subaperture)
        y1_dec = min(
            sub_size_m,
            (y_hi + N_subaperture - 1) // N_subaperture,
        )
        x0 = max(0, int(round(x_c - w / 2)))
        x1 = min(n_rg_m, int(round(x_c + w / 2)))
        if y1_dec <= y0_dec or x1 <= x0:
            return 0
        return int(mask_filt[y0_dec:y1_dec, x0:x1].sum())

    n_det_per_initial_box = np.array(
        [
            _n_det_in_box(int(y_c), int(x_c), int(h), int(w))
            for y_c, x_c, h, w in boxes_yxhw_initial
        ],
        dtype=np.int64,
    )

    # Set of int (y_c, x_c, h, w) tuples for the current `boxes_yxhw`;
    # used to snapshot the survivors after each filter stage. The
    # sets are consumed in the CSV write to build the per-initial-box
    # `disposition` column (which stage a given cluster_targets box
    # was dropped by, or "kept" if it made it to the end).
    def _yxhw_tset(arr: np.ndarray) -> set:
        return {tuple(int(v) for v in row) for row in arr}

    sw.mark(
        f"cluster_targets \u2192 boxes_yxhw "
        f"({len(boxes_yxhw)} boxes; d_phase / amp / mask computed "
        f"per-box downstream)"
    )

    # --- Main figure canvas (populated further below) ------------------
    # Skipped in minimal mode (`cfg.debug=False`) — the overview panel is a diagnostic artefact. Downstream population + save calls guard on `fig is not None` so they no-op cleanly.
    if _save_all:
        fig, (ax_main, ax_bdy) = plt.subplots(
            1, 2, figsize=(20, 9), constrained_layout=True,
        )
        fig.suptitle(
            f"SAR moving target detection — {cfg.path.name}",
            fontsize=11,
        )
    else:
        fig = None
        ax_main = None
        ax_bdy = None

    # --- Peak-level ROI hit count ------------------------------------- Diagnostic print: for each --debug-roi rectangle report how many cluster_targets seed peaks fell inside. Peaks are projected onto the s_degraded grid (`peaks_yx_full`), same coordinates as `debug_rois` and `boxes_yxhw`.
    for i, (y_lo, y_hi, x_lo, x_hi) in enumerate(debug_rois):
        in_peaks = (
            (peaks_yx_full[:, 0] >= y_lo) & (peaks_yx_full[:, 0] <= y_hi)
            & (peaks_yx_full[:, 1] >= x_lo) & (peaks_yx_full[:, 1] <= x_hi)
        )
        n_peaks_roi = int(in_peaks.sum())
        print(
            f"[roi #{i}] {'peaks in boundary_box':34s}: "
            f"{n_peaks_roi}/{len(peaks_yx_full)} peaks inside ROI"
        )
        if n_peaks_roi == 0:
            print(
                f"[roi #{i}]   → the CoV gate did not seed any peak "
                "inside the ROI; the target never enters the box "
                "pipeline. Inspect mask_dec / boundary_box, or lower "
                "--cov-th-mult to recover it."
            )
    _roi_print("cluster_targets boxes", boxes_yxhw)

    # --- Baseline post-cluster filter chain --------------------------- `cluster_targets` already absorbed grow_and_recenter_boxes, extend_boxes_azimuth_strong_signal and the range-cap growth guard, and peak-level clustering (az_search_m × rg_search_m) subsumes the IoU / centre-distance / bridge / overlap merges. Only the phase- based filters and the COM / contrast_gain diagnostics remain from the baseline pipeline; they run in the same order they did there.

    # 0. Nested-box cleanup — first stage of the post-cluster filter chain. For every pair of boxes, if the SMALLER member is covered ≥ `--nested-box-coverage-thresh` of its own area by the bigger, the bigger box gets dropped (keep the tighter detection). Removes redundant loose envelopes that fully wrap a tight one. See `filter_nested_boxes` for the exact rule, tiebreakers and pathological cases (chains, duplicate triples, …).
    _n_pre_nested = len(boxes_yxhw)
    boxes_yxhw = filter_nested_boxes(
        boxes_yxhw,
        coverage_thresh=cfg.nested_box_coverage_thresh,
    )
    n_after_nested = len(boxes_yxhw)
    kept_after_nested = _yxhw_tset(boxes_yxhw)
    print(
        f"  nested-box cleanup: {_n_pre_nested} → {n_after_nested} "
        f"(drop bigger box when smaller is ≥ "
        f"{cfg.nested_box_coverage_thresh:g} covered)"
    )
    _roi_print(
        f"nested-box cleanup (coverage ≥ "
        f"{cfg.nested_box_coverage_thresh:g})",
        boxes_yxhw,
    )

    n0 = len(boxes_yxhw)

    # 1. Phase slope filter — drop boxes with a near-flat amp-weighted phase ramp. Stationary targets have per-row slope ~0; movers shear the phase along azimuth. --min-slope-deg 0 disables.
    boxes_yxhw = filter_boxes_by_phase_slope(
        boxes_yxhw, s_degraded, cfg.min_slope_deg,
        min_row_coherence=cfg.min_row_coherence,
        inlier_tol_rad=cfg.inlier_tol_rad,
    )
    n_after_slope = len(boxes_yxhw)
    kept_after_slope = _yxhw_tset(boxes_yxhw)
    _roi_print(
        f"|slope| ≥ {cfg.min_slope_deg:g}°/row", boxes_yxhw,
    )

    # 2. Phase residual filter — drop boxes whose phi trace doesn't actually look like a line (mean wrapped Euclidean distance from phi[i] to the fit slope·i + intercept). Boxes with enough CoV-gate hits inside them (> --residual-skip-det-frac · h_box) bypass this filter — a bright extended target is trusted on amplitude alone. --max-phase-residual-rad 0 disables.
    boxes_yxhw = filter_boxes_by_phase_residual(
        boxes_yxhw, s_degraded, cfg.max_phase_residual_rad,
        mask_dec=mask_filt,
        N_subaperture=N_subaperture,
        detection_skip_frac=cfg.residual_skip_det_frac,
        min_row_coherence=cfg.min_row_coherence,
        inlier_tol_rad=cfg.inlier_tol_rad,
    )
    n_after_residual = len(boxes_yxhw)
    kept_after_residual = _yxhw_tset(boxes_yxhw)
    _roi_print(
        f"mean |phi-fit| ≤ {cfg.max_phase_residual_rad:g} rad",
        boxes_yxhw,
    )

    # --- Intermediate diagnostic figure: |s_degraded| + post-residual boxes ---
    # Amplitude of the range-degraded SLC on the full s_degraded grid
    # overlaid with `boxes_yxhw` in the state right after the phase-
    # slope + phase-residual filters (i.e. the survivors of the two
    # phase-based gates, before COM / contrast / motion filtering).
    # Saved further down as `<stem>_slc_after_residual.png` when
    # --save is set.
    # Skipped in minimal mode (`cfg.debug=False`) — this is a diagnostic-only artefact.
    if _save_all and cfg.save is not None:
        fig_slc_boxes, _ax_slc_boxes = plt.subplots(
            1, 1, figsize=(14, 9), constrained_layout=True,
        )
        fig_slc_boxes.suptitle(
            f"|s_degraded| + boxes after phase-slope + residual filters "
            f"— {cfg.path.name}",
            fontsize=11,
        )
        _show_slc(
            _ax_slc_boxes, s_degraded,
            f"|s_degraded|: {n_after_residual} boxes surviving phase-slope "
            f"(min={cfg.min_slope_deg:g}°/row) + phase-residual "
            f"(max={cfg.max_phase_residual_rad:g} rad) filters",
        )
        _overlay_boxes(
            _ax_slc_boxes, boxes_yxhw, color="cyan", lw=1.0,
        )
    else:
        fig_slc_boxes = None

    # 3. Sub-aperture COM peak-to-peak filter — stationary targets keep their amp-weighted COM in the same place across all N Doppler sub-bands, so their PP shift is small. Kept optional (default --com-filter-mode='off'); the three-tier motion filter downstream is what actually gates most survival.
    com_az_sub, com_rg = compute_subaperture_com_per_box(
        boxes_yxhw, subapertures, N_subaperture,
    )
    if cfg.com_filter_mode != "off" and len(boxes_yxhw):
        az_finite_any = np.isfinite(com_az_sub).any(axis=1)
        rg_finite_any = np.isfinite(com_rg).any(axis=1)
        az_pp = np.where(
            az_finite_any,
            np.nanmax(com_az_sub, axis=1) - np.nanmin(com_az_sub, axis=1),
            0.0,
        )
        rg_pp = np.where(
            rg_finite_any,
            np.nanmax(com_rg, axis=1) - np.nanmin(com_rg, axis=1),
            0.0,
        )
        az_ok = az_pp >= cfg.com_az_thresh
        rg_ok = rg_pp >= cfg.com_rg_thresh
        if cfg.com_filter_mode == "any":
            com_passes = az_ok | rg_ok
        else:                          # "both"
            com_passes = az_ok & rg_ok
        boxes_yxhw = boxes_yxhw[com_passes]
        com_az_sub = com_az_sub[com_passes]
        com_rg = com_rg[com_passes]
    n_after_com = len(boxes_yxhw)
    kept_after_com = _yxhw_tset(boxes_yxhw)
    _roi_print(
        f"COM-PP filter ({cfg.com_filter_mode})", boxes_yxhw,
    )

    # 4. Per-box, per-subaperture contrast gain — diagnostic only, but travels alongside com_az_sub / com_rg through the rest of the pipeline so it lands in NPZ + CSV one-to-one with each survivor.
    contrast_gain, contrast_sub, contrast_full = (
        compute_subaperture_contrast_gain_per_box(
            boxes_yxhw, s_degraded, subapertures, N_subaperture,
        )
    )

    if n0:
        pct_suppressed = 100.0 * (1 - n_after_com / n0)
    else:
        pct_suppressed = 0.0
    print(
        f"  post-cluster filters: {n0} "
        f"→ {n_after_slope} (|slope| ≥ {cfg.min_slope_deg:g}°/row) "
        f"→ {n_after_residual} (mean |phi-fit| ≤ "
        f"{cfg.max_phase_residual_rad:g} rad or det > "
        f"{cfg.residual_skip_det_frac:g}·n_az) "
        f"→ {n_after_com} (COM motion: {cfg.com_filter_mode}, "
        f"Δaz≥{cfg.com_az_thresh:g} sub-az px, "
        f"Δrg≥{cfg.com_rg_thresh:g} rg px) "
        f"— {pct_suppressed:.1f}% total suppressed"
    )

    if len(boxes_yxhw):
        az_pp_print = np.nanmax(com_az_sub, axis=1) - np.nanmin(com_az_sub, axis=1)
        rg_pp_print = np.nanmax(com_rg, axis=1)     - np.nanmin(com_rg, axis=1)
        print(
            f"  per-box subaperture COM (after filter): shape={com_az_sub.shape},"
            f" median Δaz={np.nanmedian(az_pp_print):.2f} sub-az px,"
            f" median Δrg={np.nanmedian(rg_pp_print):.2f} rg px"
        )
        cg_finite = np.isfinite(contrast_gain)
        if cg_finite.any():
            cg_med = float(np.nanmedian(contrast_gain))
            cg_min = float(np.nanmin(contrast_gain))
            cg_max = float(np.nanmax(contrast_gain))
            print(
                f"  per-box contrast gain Σ_j C(sub_j)/C(s_degraded): "
                f"median={cg_med:.2f}, min={cg_min:.2f}, max={cg_max:.2f} "
                f"(reference N_subaperture={N_subaperture})"
            )
        else:
            print(
                f"  per-box contrast gain: no finite values "
                f"(reference N_subaperture={N_subaperture})"
            )

    sw.mark(
        "post-cluster filter chain "
        "(phase slope + residual + COM + contrast gain)"
    )

    # Per-box azimuth phase estimate + coherence-weighted linear fit. Returns the actual per-row phase phi[i] used by the fit (the "phase estimated for each target") plus its slope, intercept, and mean wrapped Euclidean residual to the fit. The "slope·n" total phase swing in radians is just slope · n_rows; box_residual_rad is the exact same metric the --max-phase-residual-rad filter uses, kept per box so it can be inspected in the CSV / NPZ. The isolated-pixel-filtered CoV mask (`mask_filt`) is passed in so speckle-only pixels don't feed the amplitude-weighted circular mean — long boxes whose tails extend past the actual ship footprint then collapse to coh=0 on those rows instead of producing a random-phase, non-zero circular mean that would otherwise pollute the linear-fit residual and the phase-fit-quality score.
    (box_phi, box_coh, box_slope_rad_per_row, box_intercept_rad,
     box_y0, box_n_rows, box_residual_rad, box_n_inliers) = (
        compute_box_phase_estimates(
            boxes_yxhw, s_degraded,
            min_row_coherence=cfg.min_row_coherence,
            inlier_tol_rad=cfg.inlier_tol_rad,
            mask_dec=mask_filt,
            N_subaperture=N_subaperture,
        )
    )
    slope_totals = box_slope_rad_per_row * box_n_rows.astype(np.float64)

    # Per-box fit-quality metric complementary to box_residual_rad: fraction of azimuth rows whose wrapped residual to the linear fit is within ±--phase-fit-thresh-rad. Two versions are stored — an unweighted count and a coherence-weighted one; the coh-weighted one is what the optional --phase-fit-frac1-min gate compares against below (AF-validated movers cluster at frac_coh ≳ 0.7 on WTW3YQ, see fit_quality_out/WTW3YQ_fit_vs_af.png).
    frac_within_1rad, frac_within_1rad_coh = compute_box_frac_within_thresh(
        box_phi, box_coh,
        box_slope_rad_per_row, box_intercept_rad, box_n_rows,
        thresh_rad=cfg.phase_fit_thresh_rad,
        min_row_coherence=cfg.min_row_coherence,
    )

    # ------------------------------------------------------------------ Diagnostic: Theil-Sen linear fit of the per-box COM walk vs subaperture time. For each box we fit `com_az_sub[k, :]` (converted to metres) as a function of subaperture-centre time; the slope is the target's apparent azimuth velocity (m/s), and the RMS residual around the fit (also in metres) tells us whether the walk is a coherent linear drift (real mover — small residual, high slope) or a random hop pattern (weak scatterer — residual comparable to walk). Same fit is run on `com_rg` to give the range velocity component. `v_com_ts_mps` is the L2 combined magnitude, `com_ts_res_m` is the L2 combined RMS residual. Fully separated from the phase-slope motion gate below — this is diagnostic-only (saved to NPZ / CSV, printed with hypothetical keep counts) so we can pick real thresholds from data before switching the gate over. Skipped in minimal mode (`cfg.debug=False`); NaN-filled placeholders keep the downstream `[is_moving]` / `[keep_mask]` slicing operations size-consistent, and the NPZ / CSV writes that consume them are also gated behind `_save_all`. ------------------------------------------------------------------
    _K_ts = len(boxes_yxhw)
    if _save_all:
        _sub_size_com = subapertures.shape[1]
        (v_az_com_ts_mps, v_rg_com_ts_mps,
         com_ts_res_az_m, com_ts_res_rg_m,
         com_ts_n_fin_az, com_ts_n_fin_rg) = compute_box_com_theilsen_fit(
            com_az_sub, com_rg,
            sub_size=_sub_size_com,
            prf=cfg.prf,
            N_subaperture=N_subaperture,
            azimuth_spacing_m=azimuth_spacing,
            Number_of_Range_Looks=Number_of_Range_Looks,
            range_spacing_m=range_spacing,
        )
        # L2 combined magnitude / residual. When one axis is NaN we fall
        # back to the other alone (still NaN when BOTH failed).
        _v_az_sq = np.where(np.isfinite(v_az_com_ts_mps), v_az_com_ts_mps ** 2, 0.0)
        _v_rg_sq = np.where(np.isfinite(v_rg_com_ts_mps), v_rg_com_ts_mps ** 2, 0.0)
        _v_any_finite = (
            np.isfinite(v_az_com_ts_mps) | np.isfinite(v_rg_com_ts_mps)
        )
        v_com_ts_mps = np.where(
            _v_any_finite, np.sqrt(_v_az_sq + _v_rg_sq), np.nan,
        )
        _r_az_sq = np.where(np.isfinite(com_ts_res_az_m), com_ts_res_az_m ** 2, 0.0)
        _r_rg_sq = np.where(np.isfinite(com_ts_res_rg_m), com_ts_res_rg_m ** 2, 0.0)
        _r_any_finite = (
            np.isfinite(com_ts_res_az_m) | np.isfinite(com_ts_res_rg_m)
        )
        com_ts_res_m = np.where(
            _r_any_finite, np.sqrt(_r_az_sq + _r_rg_sq), np.nan,
        )
    else:
        v_az_com_ts_mps = np.full(_K_ts, np.nan, dtype=np.float64)
        v_rg_com_ts_mps = np.full(_K_ts, np.nan, dtype=np.float64)
        com_ts_res_az_m = np.full(_K_ts, np.nan, dtype=np.float64)
        com_ts_res_rg_m = np.full(_K_ts, np.nan, dtype=np.float64)
        com_ts_n_fin_az = np.zeros(_K_ts, dtype=np.int32)
        com_ts_n_fin_rg = np.zeros(_K_ts, dtype=np.int32)
        v_com_ts_mps = np.full(_K_ts, np.nan, dtype=np.float64)
        com_ts_res_m = np.full(_K_ts, np.nan, dtype=np.float64)

    # Diagnostic print + hypothetical middle-band keep-count table. Skipped in minimal mode — expensive percentile + tier scan for the console table, with no gating effect.
    if _save_all and len(boxes_yxhw):
        _v_finite = np.isfinite(v_com_ts_mps)
        _r_finite = np.isfinite(com_ts_res_m)
        def _pct(vec, mask, q):
            return float(np.percentile(vec[mask], q)) if mask.any() else float("nan")
        _v_p50 = _pct(v_com_ts_mps, _v_finite, 50)
        _v_p90 = _pct(v_com_ts_mps, _v_finite, 90)
        _v_p99 = _pct(v_com_ts_mps, _v_finite, 99)
        _r_p50 = _pct(com_ts_res_m, _r_finite, 50)
        _r_p90 = _pct(com_ts_res_m, _r_finite, 90)
        _r_p99 = _pct(com_ts_res_m, _r_finite, 99)
        _n_fin_az_med = int(np.median(com_ts_n_fin_az))
        _n_fin_rg_med = int(np.median(com_ts_n_fin_rg))
        print("  COM Theil-Sen fit (diagnostic; does not gate):")
        print(
            f"    finite subs per box: median az={_n_fin_az_med}/"
            f"{N_subaperture}, rg={_n_fin_rg_med}/{N_subaperture}   "
            f"({int(_v_finite.sum())}/{len(boxes_yxhw)} boxes have a "
            f"finite |v|_fit, {int(_r_finite.sum())} a finite residual)"
        )
        print(
            f"    |v|_fit m/s : p50={_v_p50:.2f}, p90={_v_p90:.2f}, "
            f"p99={_v_p99:.2f}"
        )
        print(
            f"    residual m  : p50={_r_p50:.2f}, p90={_r_p90:.2f}, "
            f"p99={_r_p99:.2f}"
        )
        # Phase-slope tier labels — same thresholds as the motion filter
        # below, so the diagnostic and the actual gate speak the same
        # language.
        _abs_slope_diag = np.where(
            np.isfinite(slope_totals), np.abs(slope_totals), 0.0,
        )
        _tier_clear_no = _abs_slope_diag < cfg.slope_rad_lower
        _tier_clear_yes = _abs_slope_diag >= cfg.slope_rad_thresh
        _tier_mid = (~_tier_clear_no) & (~_tier_clear_yes)
        _az_pp_diag = np.where(
            np.isfinite(com_az_sub).any(axis=1),
            (np.nanmax(com_az_sub, axis=1)
             - np.nanmin(com_az_sub, axis=1)),
            0.0,
        )
        _cur_keep_mid = int(
            (_tier_mid & (_az_pp_diag >= cfg.az_pp_sub_thresh)).sum()
        )
        _v_grid = (1.0, 1.5, 2.0)
        _r_grid = (0.5, 1.0, 2.0, 5.0)
        print(
            f"    hypothetical middle-band keep counts (of "
            f"{int(_tier_mid.sum())} middle-band boxes; current "
            f"az_pp≥{cfg.az_pp_sub_thresh:g} rule keeps "
            f"{_cur_keep_mid}):"
        )
        _hdr_cells = "  ".join(f"{r:>5.1f} m" for r in _r_grid)
        print(f"      v_min\\res_max  {_hdr_cells}")
        for _v_min_try in _v_grid:
            _row_vals = []
            for _r_max_try in _r_grid:
                _pass = (
                    np.isfinite(v_com_ts_mps)
                    & np.isfinite(com_ts_res_m)
                    & (v_com_ts_mps >= _v_min_try)
                    & (com_ts_res_m <= _r_max_try)
                )
                _row_vals.append(int((_tier_mid & _pass).sum()))
            _row_cells = "  ".join(f"{v:>7d}" for v in _row_vals)
            print(f"      {_v_min_try:>4.1f} m/s     {_row_cells}")
        _n_cy_pass = int(
            (_tier_clear_yes
             & np.isfinite(v_com_ts_mps)
             & (v_com_ts_mps >= 2.0)).sum()
        )
        print(
            f"    sanity: of {int(_tier_clear_yes.sum())} clear-yes "
            f"(|slope·n|≥{cfg.slope_rad_thresh:g}) boxes, "
            f"{_n_cy_pass} also have |v|_fit ≥ 2.0 m/s"
        )

    # ------------------------------------------------------------------ Motion filter — three-tier rule on the box-wide phase slope.   |slope · n_rows| <  --slope-rad-lower    ⇒ DROP  (clear stationary)   |slope · n_rows| >= --slope-rad-thresh   ⇒ KEEP  (clear mover)   else (ambiguous middle band)             ⇒ KEEP iff                                              az_pp ≥ --az-pp-sub-thresh The COM-velocity branch (|v_az| or |v_rg| ≥ --min-velocity-mps) and the per-column wrap-immune slope_map aggregate are still computed and saved in NPZ / CSV as diagnostics; they do NOT gate the filter. Re-enable e.g. by OR-ing `is_moving_com` / `is_moving_slope_col` into `is_moving` below. ------------------------------------------------------------------
    sub_size = subapertures.shape[1]
    # Every downstream reader of `subapertures` needs just the second-axis size (captured above as `sub_size`); the last consumers of the actual (N, sub_size, N_rg) float32 buffer were `compute_subaperture_com_per_box` and `compute_subaperture_contrast_gain_per_box` upstream, and Figure 3 (built later only under `_save_all and cfg.display_all_mode`) reconstructs its `subapertures[k]` panels lazily. Free the ≈1.5 GB buffer here before the AF pre-compute + paste loop unless the display-all figure is going to be built.
    if not (_save_all and cfg.display_all_mode):
        del subapertures
    v_az_mps, v_rg_mps = compute_box_com_velocities(
        com_az_sub, com_rg,
        sub_size=sub_size,
        prf=cfg.prf,
        N_subaperture=N_subaperture,
        azimuth_spacing_m=azimuth_spacing,
        Number_of_Range_Looks=Number_of_Range_Looks,
        range_spacing_m=range_spacing,
    )

    # Diagnostic only — kept in CSV / NPZ but does NOT influence is_moving.
    is_moving_com = (
        (np.isfinite(v_az_mps) & (np.abs(v_az_mps) >= cfg.min_velocity_mps))
        | (np.isfinite(v_rg_mps) & (np.abs(v_rg_mps) >= cfg.min_velocity_mps))
    )
    # (The per-column wrap-immune `slope_map` aggregate diagnostic was dropped along with the scene-wide `d_phase`/`slope_map` allocation. `compute_box_slope_col_total` is still available as a library helper if you want to bring it back — it just needs a per-box slope_map to be computed first.)

    # Phase-free signal: peak-to-peak azimuth COM walk across the N Doppler subapertures (NaN-safe). For a moving target the bright spot shifts position from look to look; for stationary clutter the COM is roughly constant.
    if len(com_az_sub):
        _az_finite = np.isfinite(com_az_sub).any(axis=1)
        az_pp_per_box = np.where(
            _az_finite,
            np.nanmax(com_az_sub, axis=1) - np.nanmin(com_az_sub, axis=1),
            0.0,
        )
    else:
        az_pp_per_box = np.zeros(len(boxes_yxhw), dtype=np.float64)
    is_moving_az_pp = az_pp_per_box >= cfg.az_pp_sub_thresh

    # ------------------------------------------------------------------ Three-tier motion gate on the box-wide phase slope:   |slope·n| <  --slope-rad-lower    ⇒ DROP (clear stationary)   |slope·n| >= --slope-rad-thresh   ⇒ KEEP (clear mover)   else (ambiguous middle band)      ⇒ KEEP iff az_pp ≥ az-pp-sub-thresh NaN |slope·n| is treated as "below lower" and dropped — the phase fit failed entirely, we don't trust anything for that box. ------------------------------------------------------------------
    abs_slope = np.where(
        np.isfinite(slope_totals), np.abs(slope_totals), 0.0,
    )
    is_moving_slope_box = abs_slope >= cfg.slope_rad_thresh           # clear yes
    slope_clear_no      = abs_slope <  cfg.slope_rad_lower            # clear no → drop
    slope_ambiguous     = (~is_moving_slope_box) & (~slope_clear_no)   # middle band

    is_moving = is_moving_slope_box | (slope_ambiguous & is_moving_az_pp)

    # Diagnostic union flag (any slope-route would have admitted it). Used only for downstream save / NPZ; the actual is_moving above is what gates the survival.
    is_moving_slope = is_moving_slope_box | (slope_ambiguous & is_moving_az_pp)

    n_pre_motion = len(boxes_yxhw)
    n_post_motion = int(is_moving.sum())

    # Pre-filter counts in each tier of the three-tier rule.
    n_clear_no    = int(slope_clear_no.sum())                       # dropped: |slope·n| < lower
    n_clear_yes   = int(is_moving_slope_box.sum())                  # kept:    |slope·n| ≥ thresh
    n_mid_keep    = int((slope_ambiguous & is_moving_az_pp).sum())  # kept by az_pp tiebreaker
    n_mid_drop    = int((slope_ambiguous & ~is_moving_az_pp).sum()) # middle band, failed az_pp

    if n_pre_motion and not is_moving.all():
        boxes_yxhw           = boxes_yxhw[is_moving]
        com_az_sub           = com_az_sub[is_moving]
        com_rg               = com_rg[is_moving]
        v_az_mps             = v_az_mps[is_moving]
        v_rg_mps             = v_rg_mps[is_moving]
        box_phi              = box_phi[is_moving]
        box_coh              = box_coh[is_moving]
        box_slope_rad_per_row = box_slope_rad_per_row[is_moving]
        box_intercept_rad    = box_intercept_rad[is_moving]
        box_y0               = box_y0[is_moving]
        box_n_rows           = box_n_rows[is_moving]
        box_residual_rad     = box_residual_rad[is_moving]
        box_n_inliers        = box_n_inliers[is_moving]
        frac_within_1rad     = frac_within_1rad[is_moving]
        frac_within_1rad_coh = frac_within_1rad_coh[is_moving]
        slope_totals         = slope_totals[is_moving]
        contrast_gain        = contrast_gain[is_moving]
        contrast_sub         = contrast_sub[is_moving]
        contrast_full        = contrast_full[is_moving]
        az_pp_per_box        = az_pp_per_box[is_moving]
        # COM Theil-Sen fit diagnostics (velocity + RMS residual in metres, per axis and L2 combined). Diagnostic-only; travels alongside the box arrays so each row of the NPZ / CSV lines up with the survivor at the same index.
        v_az_com_ts_mps      = v_az_com_ts_mps[is_moving]
        v_rg_com_ts_mps      = v_rg_com_ts_mps[is_moving]
        v_com_ts_mps         = v_com_ts_mps[is_moving]
        com_ts_res_az_m      = com_ts_res_az_m[is_moving]
        com_ts_res_rg_m      = com_ts_res_rg_m[is_moving]
        com_ts_res_m         = com_ts_res_m[is_moving]
        com_ts_n_fin_az      = com_ts_n_fin_az[is_moving]
        com_ts_n_fin_rg      = com_ts_n_fin_rg[is_moving]
        # Per-box flags carried on the *post-filter* set as diagnostics. After the three-tier gate every survivor satisfies is_moving_slope == True (clear-yes OR ambiguous-and-az_pp). is_moving_com records whether the box would also have passed the COM-velocity route.
        is_moving_com        = is_moving_com[is_moving]
        is_moving_slope      = is_moving_slope[is_moving]
        is_moving_slope_box  = is_moving_slope_box[is_moving]
        is_moving_az_pp      = is_moving_az_pp[is_moving]

    kept_after_motion = _yxhw_tset(boxes_yxhw)

    n_com_only = int((is_moving_com & ~is_moving_slope).sum())
    print(
        f"  motion filter: {n_pre_motion} → {n_post_motion} "
        f"(3-tier on |slope·n|: <{cfg.slope_rad_lower:g} drop, "
        f"≥{cfg.slope_rad_thresh:g} keep, else az_pp ≥ "
        f"{cfg.az_pp_sub_thresh:g} sub-az px)"
    )
    print(
        f"    clear-no drop (|slope·n| < {cfg.slope_rad_lower:g}): "
        f"{n_clear_no}, "
        f"clear-yes keep (|slope·n| ≥ {cfg.slope_rad_thresh:g}): "
        f"{n_clear_yes}, "
        f"middle-band kept by az_pp: {n_mid_keep}, "
        f"middle-band dropped: {n_mid_drop}, "
        f"COM-only diag (would have passed via |v_az| or |v_rg| ≥ "
        f"{cfg.min_velocity_mps:g} m/s but dropped here): {n_com_only}"
    )
    _roi_print(
        f"motion 3-tier on |slope·n| (low={cfg.slope_rad_lower:g}, "
        f"high={cfg.slope_rad_thresh:g})",
        boxes_yxhw,
    )

    strong_mask = is_moving_slope.copy()

    # ------------------------------------------------------------------ Phase-fit-quality gate — two-stage: (1) THRESHOLD on the coherence-weighted score frac_within_1rad_coh. Finite scores < `phase_fit_frac1_min` and NaN scores are dropped ("coherent-only" cull). Setting `phase_fit_frac1_min = 0` disables just this stage. (2) TOP-N CAP with SCORE FLOOR safety-net: sort the surviving boxes by score descending and keep at most `max_focus_targets` of them (ties broken by original box index), UNIONed with any box whose score >= `phase_fit_score_floor` (the "safety net" — recovers clean marginal movers whose scores land between the top-N min and the floor). Setting `max_focus_targets <= 0` disables the top-N cap; setting `phase_fit_score_floor >= 1` disables the floor. Rationale: on the WTW3YQ dev scene the AF winners cluster at score ≳ 0.7 (median 0.85) while AF-rejected boxes sit at ~0.55 — see fit_quality_out/WTW3YQ_fit_vs_af.png. A 0.5 threshold discards the wrapped-uniform residual boxes; a top-100 cap feeds autofocus a bounded, ranked candidate list; a 0.85 floor pulls back the clean marginal movers with a small AF-runtime overhead. Autofocus is the pipeline bottleneck (~50-70 % of total time on WTW3YQ), so bounding the candidate count also bounds the runtime. ------------------------------------------------------------------
    if len(boxes_yxhw):
        n_pre = len(boxes_yxhw)
        thresh = float(cfg.phase_fit_frac1_min)
        max_targets = int(cfg.max_focus_targets)
        score_floor = float(getattr(cfg, "phase_fit_score_floor", 0.0))

        # Stage 1: threshold. NaN scores fail the check.
        score = frac_within_1rad_coh
        if thresh > 0.0:
            pass_thresh = (
                np.isfinite(score) & (score >= thresh)
            )
        else:
            pass_thresh = np.ones_like(score, dtype=bool)
        n_pass_thresh = int(pass_thresh.sum())

        # Stage 2: top-N cap by score (descending) UNION score >= floor.
        # NaN scores never win either branch. Top-N ties are broken
        # deterministically by original box index (stable sort).
        if max_targets > 0 and n_pass_thresh > max_targets:
            eligible_idx = np.where(pass_thresh)[0]
            sub_scores = np.where(
                np.isfinite(score[eligible_idx]),
                score[eligible_idx],
                -np.inf,
            )
            order = eligible_idx[np.argsort(-sub_scores, kind="stable")]
            top_idx = order[:max_targets]
            keep_mask = np.zeros_like(pass_thresh)
            keep_mask[top_idx] = True
        else:
            keep_mask = pass_thresh
        n_kept_topN = int(keep_mask.sum())

        # Safety-net floor: union in any threshold-passing box whose
        # score >= floor. Disabled when floor >= 1 or floor <= 0.
        if 0.0 < score_floor < 1.0:
            floor_mask = (
                pass_thresh
                & np.isfinite(score)
                & (score >= score_floor)
            )
            n_floor_rescued = int(
                (floor_mask & ~keep_mask).sum()
            )
            keep_mask = keep_mask | floor_mask
        else:
            n_floor_rescued = 0

        n_kept = int(keep_mask.sum())
        n_drop = n_pre - n_kept
        if n_drop > 0:
            boxes_yxhw           = boxes_yxhw[keep_mask]
            com_az_sub           = com_az_sub[keep_mask]
            com_rg               = com_rg[keep_mask]
            v_az_mps             = v_az_mps[keep_mask]
            v_rg_mps             = v_rg_mps[keep_mask]
            box_phi              = box_phi[keep_mask]
            box_coh              = box_coh[keep_mask]
            box_slope_rad_per_row = box_slope_rad_per_row[keep_mask]
            box_intercept_rad    = box_intercept_rad[keep_mask]
            box_y0               = box_y0[keep_mask]
            box_n_rows           = box_n_rows[keep_mask]
            box_residual_rad     = box_residual_rad[keep_mask]
            box_n_inliers        = box_n_inliers[keep_mask]
            frac_within_1rad     = frac_within_1rad[keep_mask]
            frac_within_1rad_coh = frac_within_1rad_coh[keep_mask]
            slope_totals         = slope_totals[keep_mask]
            contrast_gain        = contrast_gain[keep_mask]
            contrast_sub         = contrast_sub[keep_mask]
            contrast_full        = contrast_full[keep_mask]
            az_pp_per_box        = az_pp_per_box[keep_mask]
            v_az_com_ts_mps      = v_az_com_ts_mps[keep_mask]
            v_rg_com_ts_mps      = v_rg_com_ts_mps[keep_mask]
            v_com_ts_mps         = v_com_ts_mps[keep_mask]
            com_ts_res_az_m      = com_ts_res_az_m[keep_mask]
            com_ts_res_rg_m      = com_ts_res_rg_m[keep_mask]
            com_ts_res_m         = com_ts_res_m[keep_mask]
            com_ts_n_fin_az      = com_ts_n_fin_az[keep_mask]
            com_ts_n_fin_rg      = com_ts_n_fin_rg[keep_mask]
            is_moving_com        = is_moving_com[keep_mask]
            is_moving_slope      = is_moving_slope[keep_mask]
            is_moving_slope_box  = is_moving_slope_box[keep_mask]
            is_moving_az_pp      = is_moving_az_pp[keep_mask]
            strong_mask          = strong_mask[keep_mask]
        cap_txt = (
            f"top-{max_targets} by score"
            if (max_targets > 0 and n_pass_thresh > max_targets)
            else "no cap"
        )
        if 0.0 < score_floor < 1.0:
            floor_txt = (
                f"; floor≥{score_floor:.3g} rescued {n_floor_rescued}"
            )
        else:
            floor_txt = ""
        print(
            f"  phase-fit-quality filter: {n_pre} → {n_kept} "
            f"(threshold frac(|r|≤{cfg.phase_fit_thresh_rad:g} rad, "
            f"coh-weighted) ≥ {thresh:g}: {n_pass_thresh} kept; "
            f"{cap_txt}{floor_txt})"
        )
        if n_kept:
            _kept_scores = frac_within_1rad_coh[np.isfinite(frac_within_1rad_coh)]
            if _kept_scores.size:
                print(
                    f"    kept score range : min={_kept_scores.min():.3f} "
                    f"med={np.median(_kept_scores):.3f} "
                    f"max={_kept_scores.max():.3f} "
                    f"(NaN: {int((~np.isfinite(frac_within_1rad_coh)).sum())})"
                )
        _roi_print(
            f"phase-fit-quality gate frac_coh ≥ "
            f"{thresh:g} @ ±{cfg.phase_fit_thresh_rad:g} rad, "
            f"cap={max_targets if max_targets > 0 else 'off'}",
            boxes_yxhw,
        )
    kept_after_fit_quality = _yxhw_tset(boxes_yxhw)

    # ------------------------------------------------------------------ Optional refocus-gain gate (only when --refocus is set). Drop boxes whose centred-QPE refocus *decreases* time-domain contrast by more than |--refocus-min-gain-db| dB: the fitted slope is inconsistent with the chip's spectral structure and the detection is almost certainly spurious. NaN gains are kept (no evidence either way). ------------------------------------------------------------------
    if cfg.refocus and len(boxes_yxhw):
        H_full, W_full = s_degraded.shape
        gains_db_full = np.full(len(boxes_yxhw), np.nan, dtype=np.float64)
        for k in range(len(boxes_yxhw)):
            n_k = int(box_n_rows[k])
            y0_k = int(box_y0[k])
            y1_k = y0_k + n_k
            y_c_k = int(boxes_yxhw[k, 0])
            x_c_k = int(boxes_yxhw[k, 1])
            w_k = int(boxes_yxhw[k, 3])
            x0_k = max(0, x_c_k - w_k // 2)
            x1_k = min(W_full, x_c_k - w_k // 2 + w_k)
            s_k = float(box_slope_rad_per_row[k])
            if (
                n_k < 2 or x1_k <= x0_k or y1_k <= y0_k
                or y0_k < 0 or y1_k > H_full
                or not np.isfinite(s_k)
            ):
                continue
            chip_k = s_degraded[y0_k:y1_k, x0_k:x1_k]
            if chip_k.size == 0:
                continue
            corrected_k, _phi = _refocus_box_chip(chip_k, s_k)
            c_before = _normalized_variance(chip_k)
            c_after = _normalized_variance(corrected_k)
            if c_before > 0.0 and c_after > 0.0:
                gains_db_full[k] = 20.0 * np.log10(c_after / c_before)
        # keep iff gain >= threshold (or gain is NaN / no measurement)
        keep_mask = ~(
            np.isfinite(gains_db_full)
            & (gains_db_full < cfg.refocus_min_gain_db)
        )
        n_pre = len(boxes_yxhw)
        n_drop = int((~keep_mask).sum())
        if n_drop > 0:
            boxes_yxhw           = boxes_yxhw[keep_mask]
            com_az_sub           = com_az_sub[keep_mask]
            com_rg               = com_rg[keep_mask]
            v_az_mps             = v_az_mps[keep_mask]
            v_rg_mps             = v_rg_mps[keep_mask]
            box_phi              = box_phi[keep_mask]
            box_coh              = box_coh[keep_mask]
            box_slope_rad_per_row = box_slope_rad_per_row[keep_mask]
            box_intercept_rad    = box_intercept_rad[keep_mask]
            box_y0               = box_y0[keep_mask]
            box_n_rows           = box_n_rows[keep_mask]
            box_residual_rad     = box_residual_rad[keep_mask]
            box_n_inliers        = box_n_inliers[keep_mask]
            frac_within_1rad     = frac_within_1rad[keep_mask]
            frac_within_1rad_coh = frac_within_1rad_coh[keep_mask]
            slope_totals         = slope_totals[keep_mask]
            contrast_gain        = contrast_gain[keep_mask]
            contrast_sub         = contrast_sub[keep_mask]
            contrast_full        = contrast_full[keep_mask]
            az_pp_per_box        = az_pp_per_box[keep_mask]
            v_az_com_ts_mps      = v_az_com_ts_mps[keep_mask]
            v_rg_com_ts_mps      = v_rg_com_ts_mps[keep_mask]
            v_com_ts_mps         = v_com_ts_mps[keep_mask]
            com_ts_res_az_m      = com_ts_res_az_m[keep_mask]
            com_ts_res_rg_m      = com_ts_res_rg_m[keep_mask]
            com_ts_res_m         = com_ts_res_m[keep_mask]
            com_ts_n_fin_az      = com_ts_n_fin_az[keep_mask]
            com_ts_n_fin_rg      = com_ts_n_fin_rg[keep_mask]
            is_moving_com        = is_moving_com[keep_mask]
            is_moving_slope      = is_moving_slope[keep_mask]
            is_moving_slope_box  = is_moving_slope_box[keep_mask]
            is_moving_az_pp      = is_moving_az_pp[keep_mask]
            strong_mask          = strong_mask[keep_mask]
        print(
            f"  refocus-gain filter: {n_pre} → {n_pre - n_drop} "
            f"(drop boxes with gain < {cfg.refocus_min_gain_db:g} dB)"
        )
        _roi_print(
            f"refocus-gain gate ≥ {cfg.refocus_min_gain_db:g} dB",
            boxes_yxhw,
        )
    kept_after_refocus = _yxhw_tset(boxes_yxhw)

    # ------------------------------------------------------------------ Optional sub-aperture sharpness gate. Drop boxes whose per-box sub-aperture sharpness ratio Σ_j C(sub_j) / C(s_degraded) (the `contrast_gain` saved to CSV / NPZ; DIMENSIONLESS, not dB) is finite and below --subaperture-sharpness-min. NaN ratios (empty crop / zero mean) are kept (no evidence either way). Default threshold -inf ⇒ filter is a no-op. Distinct from the dB refocus gain drawn on `box_NNN_refocus.png`, which is gated by --refocus-min-gain-db above. ------------------------------------------------------------------
    if np.isfinite(cfg.subaperture_sharpness_min) and len(boxes_yxhw):
        keep_mask = ~(
            np.isfinite(contrast_gain)
            & (contrast_gain < cfg.subaperture_sharpness_min)
        )
        n_pre = len(boxes_yxhw)
        n_drop = int((~keep_mask).sum())
        if n_drop > 0:
            boxes_yxhw           = boxes_yxhw[keep_mask]
            com_az_sub           = com_az_sub[keep_mask]
            com_rg               = com_rg[keep_mask]
            v_az_mps             = v_az_mps[keep_mask]
            v_rg_mps             = v_rg_mps[keep_mask]
            box_phi              = box_phi[keep_mask]
            box_coh              = box_coh[keep_mask]
            box_slope_rad_per_row = box_slope_rad_per_row[keep_mask]
            box_intercept_rad    = box_intercept_rad[keep_mask]
            box_y0               = box_y0[keep_mask]
            box_n_rows           = box_n_rows[keep_mask]
            box_residual_rad     = box_residual_rad[keep_mask]
            box_n_inliers        = box_n_inliers[keep_mask]
            frac_within_1rad     = frac_within_1rad[keep_mask]
            frac_within_1rad_coh = frac_within_1rad_coh[keep_mask]
            slope_totals         = slope_totals[keep_mask]
            contrast_gain        = contrast_gain[keep_mask]
            contrast_sub         = contrast_sub[keep_mask]
            contrast_full        = contrast_full[keep_mask]
            az_pp_per_box        = az_pp_per_box[keep_mask]
            v_az_com_ts_mps      = v_az_com_ts_mps[keep_mask]
            v_rg_com_ts_mps      = v_rg_com_ts_mps[keep_mask]
            v_com_ts_mps         = v_com_ts_mps[keep_mask]
            com_ts_res_az_m      = com_ts_res_az_m[keep_mask]
            com_ts_res_rg_m      = com_ts_res_rg_m[keep_mask]
            com_ts_res_m         = com_ts_res_m[keep_mask]
            com_ts_n_fin_az      = com_ts_n_fin_az[keep_mask]
            com_ts_n_fin_rg      = com_ts_n_fin_rg[keep_mask]
            is_moving_com        = is_moving_com[keep_mask]
            is_moving_slope      = is_moving_slope[keep_mask]
            is_moving_slope_box  = is_moving_slope_box[keep_mask]
            is_moving_az_pp      = is_moving_az_pp[keep_mask]
            strong_mask          = strong_mask[keep_mask]
        print(
            f"  sub-aperture sharpness filter: {n_pre} → {n_pre - n_drop} "
            f"(drop boxes with Σ_j C(sub_j)/C(s_degraded) < "
            f"{cfg.subaperture_sharpness_min:g})"
        )
        _roi_print(
            f"sub-aperture sharpness gate ≥ "
            f"{cfg.subaperture_sharpness_min:g}",
            boxes_yxhw,
        )

    sw.mark("phase-fit motion filter + refocus/sharpness gates")

    strong_boxes = boxes_yxhw[strong_mask]
    weak_boxes = boxes_yxhw[~strong_mask]
    print(
        f"  → boxes with |slope·n| ≥ {cfg.slope_rad_thresh:g} rad "
        f"(red overlays): {int(strong_mask.sum())}/{len(boxes_yxhw)}"
    )

    # --- Intermediate diagnostic figure: |s_degraded| + final boxes ---
    # Final survivors of the entire filter chain (phase-slope +
    # phase-residual + optional COM filter + motion 3-tier + refocus-
    # gain + sub-aperture-sharpness) overlaid on `|s_degraded|`.
    # Strong movers (|slope·n| ≥ slope_rad_thresh, red in the main
    # figure) get a slightly thicker outline so they read as such at
    # a glance. Saved further down as `<stem>_slc_final_boxes.png`
    # when --save is set.
    # Skipped in minimal mode (`cfg.debug=False`) — this is a diagnostic-only artefact.
    if _save_all and cfg.save is not None:
        fig_final_boxes, _ax_final_boxes = plt.subplots(
            1, 1, figsize=(14, 9), constrained_layout=True,
        )
        fig_final_boxes.suptitle(
            f"|s_degraded| + final filtered boxes — {cfg.path.name}",
            fontsize=11,
        )
        _n_strong_final = int(strong_mask.sum()) if len(boxes_yxhw) else 0
        _n_weak_final = int((~strong_mask).sum()) if len(boxes_yxhw) else 0
        _show_slc(
            _ax_final_boxes, s_degraded,
            f"|s_degraded|: {len(boxes_yxhw)} final surviving boxes "
            f"(strong |slope·n| ≥ {cfg.slope_rad_thresh:g} rad: "
            f"{_n_strong_final} red, middle-band az_pp survivors: "
            f"{_n_weak_final} cyan)"
        )
        if len(weak_boxes):
            _overlay_boxes(
                _ax_final_boxes, weak_boxes, color="cyan", lw=1.0,
            )
        if len(strong_boxes):
            _overlay_boxes(
                _ax_final_boxes, strong_boxes, color="red", lw=1.4,
            )
    else:
        fig_final_boxes = None

    # --- Intermediate diagnostic figure: kept vs eliminated boxes ---
    # `|s_degraded|` overlaid with two colour-coded sets:
    #   cyan  — boxes that survived the full filter chain
    #           (== boxes_yxhw at this point).
    #   red   — cluster_targets boxes that were eliminated somewhere
    #           along the chain (phase-slope / residual / COM / motion
    #           3-tier / refocus-gain / sub-aperture-sharpness).
    # The set difference is computed on integer (y_c, x_c, h, w)
    # tuples: every filter only index-selects rows out of
    # `boxes_yxhw_initial`, never mutates them, so a survivor row
    # appears verbatim in `boxes_yxhw`. This gives a full-pipeline
    # audit view of which detections the phase / motion / contrast
    # gates threw away and where they lived. Saved further down as
    # `<stem>_kept_vs_eliminated.png` when --save is set.
    if cfg.save is not None:
        if len(boxes_yxhw_initial):
            _kept_set = {
                tuple(int(v) for v in row) for row in boxes_yxhw
            }
            _elim_mask = np.array([
                tuple(int(v) for v in row) not in _kept_set
                for row in boxes_yxhw_initial
            ], dtype=bool)
            boxes_eliminated = boxes_yxhw_initial[_elim_mask]
        else:
            boxes_eliminated = np.empty((0, 4), dtype=np.int64)

        fig_kept_vs_eliminated, _ax_kve = plt.subplots(
            1, 1, figsize=(14, 9), constrained_layout=True,
        )
        fig_kept_vs_eliminated.suptitle(
            f"|s_degraded| — kept (cyan) vs eliminated (red) — "
            f"{cfg.path.name}",
            fontsize=11,
        )
        _show_slc(
            _ax_kve, s_degraded,
            f"|s_degraded|: {len(boxes_yxhw)} kept (cyan) / "
            f"{len(boxes_eliminated)} eliminated (red) out of "
            f"{len(boxes_yxhw_initial)} cluster_targets boxes"
        )
        # Draw eliminated first so kept sits on top when they touch.
        if len(boxes_eliminated):
            _overlay_boxes(
                _ax_kve, boxes_eliminated, color="red", lw=0.9,
            )
        if len(boxes_yxhw):
            _overlay_boxes(
                _ax_kve, boxes_yxhw, color="cyan", lw=1.1,
            )
        # Manual legend proxies — `_overlay_boxes` uses raw
        # `plt.Rectangle` patches that don't feed matplotlib's
        # automatic legend picker.
        _legend_handles = [
            plt.Line2D(
                [0], [0], color="cyan", lw=1.4,
                label=f"kept ({len(boxes_yxhw)})",
            ),
            plt.Line2D(
                [0], [0], color="red", lw=1.4,
                label=f"eliminated ({len(boxes_eliminated)})",
            ),
        ]
        _ax_kve.legend(handles=_legend_handles, loc="upper right", fontsize=8)
    else:
        fig_kept_vs_eliminated = None
        boxes_eliminated = np.empty((0, 4), dtype=np.int64)

    # Persist per-box d_phase + final boxes for downstream analysis scripts (e.g. scripts/view_dphase_boxes.py). Skipped on --no-save.
    # Skipped in minimal mode (`cfg.debug=False`) — no external consumer of the archive in that profile.
    if _save_all and cfg.save is not None:
        data_path = cfg.save.with_suffix(".npz")

        # Per-box azimuth phase derivative strips, packed into a padded (K, max_L, max_W) float32 canvas with NaN outside each box's actual (L_k, W_k) footprint. d_phase is NOT allocated scene- wide anywhere else; this is the only place the per-box strips are materialised, and they are freed as soon as the NPZ has been written. `d_phase_box_coords[k] = (y0, L, x0, W)` gives the absolute pixel range in s_degraded that d_phase_boxes[k, :L, :W] describes:     d_phase_boxes[k, i, j] = arg(         s_degraded[y0+i+1, x0+j] · conj(s_degraded[y0+i, x0+j])     )   for i = 0..L-1, j = 0..W-1
        K = len(boxes_yxhw)
        if K:
            _h_arr = boxes_yxhw[:, 2].astype(np.int64)
            _w_arr = boxes_yxhw[:, 3].astype(np.int64)
            # +1 headroom so the (K, max_L, max_W) buffer always fits
            # the actual per-box (L_k, W_k) footprint. Rounding `y0 =
            # round(y_c - h/2)` / `y1 = round(y_c + h/2)` can push both
            # edges outward by 0.5 when y_c is a half-integer (banker's
            # rounding), giving L = h + 1 for that box; same for W.
            # `compute_box_phase_estimates` does the same headroom trick
            # for its (K, max_h) phi_arr / coh_arr — see the ceil+1
            # comment there.
            max_L = int(_h_arr.max()) + 1 if _h_arr.size else 0
            max_W = int(_w_arr.max()) + 1 if _w_arr.size else 0
            d_phase_boxes = np.full(
                (K, max_L, max_W), np.nan, dtype=np.float32,
            )
            d_phase_box_coords = np.zeros((K, 4), dtype=np.int64)
            # Per-box |s_degraded| strips paired with d_phase_boxes. amp_raw_boxes[k, :L+1, :W] is the raw amplitude (`|s_degraded|`) covering the same (y0..y1+1, x0..x1) window as d_phase_boxes[k], i.e. one extra azimuth row at the bottom edge so both amplitude factors in the phase weight `|s[i, j]|·|s[i+1, j]|` are available. That extra row is why the second axis is max_L + 1 (not max_L).
            amp_raw_boxes = np.full(
                (K, max_L + 1, max_W), np.nan, dtype=np.float32,
            )
            _n_az_slc = s_degraded.shape[0]
            _n_rg_full = s_degraded.shape[1]
            for _k, (_yc, _xc, _h, _w) in enumerate(boxes_yxhw):
                _y0 = max(0, int(round(_yc - _h / 2)))
                _y1 = min(_n_az_slc - 1, int(round(_yc + _h / 2)))
                _x0 = max(0, int(round(_xc - _w / 2)))
                _x1 = min(_n_rg_full, int(round(_xc + _w / 2)))
                _L = _y1 - _y0
                _W = _x1 - _x0
                d_phase_box_coords[_k] = (_y0, _L, _x0, _W)
                if _L < 1 or _W < 1:
                    continue
                _s_box = s_degraded[_y0:_y1 + 1, _x0:_x1]
                d_phase_boxes[_k, :_L, :_W] = np.angle(
                    _s_box[1:] * np.conj(_s_box[:-1])
                ).astype(np.float32)
                amp_raw_boxes[_k, :_L + 1, :_W] = np.abs(_s_box).astype(
                    np.float32,
                )
        else:
            d_phase_boxes = np.empty((0, 0, 0), dtype=np.float32)
            d_phase_box_coords = np.empty((0, 4), dtype=np.int64)
            amp_raw_boxes = np.empty((0, 0, 0), dtype=np.float32)

        # Per-box phase estimate. box_phi[k, i] is the amplitude-weighted circular-mean azimuth phase derivative for box k at azimuth row box_y0[k] + i, for i = 0 .. box_n_rows[k] - 1; NaN beyond that. box_coh is the matching per-row coherence weight. box_slope_rad_per_row[k] · i + box_intercept_rad[k] reconstructs the linear fit at row box_y0[k] + i.
        npz_payload = dict(
            d_phase_boxes=d_phase_boxes,
            d_phase_box_coords=d_phase_box_coords,
            amp_raw_boxes=amp_raw_boxes,
            boxes_yxhw=boxes_yxhw,
            slope_totals_rad=slope_totals,
            slope_rad_thresh=np.float64(cfg.slope_rad_thresh),
            slope_rad_lower=np.float64(cfg.slope_rad_lower),
            s_degraded_shape=np.asarray(s_degraded.shape, dtype=np.int64),
            com_az_sub=com_az_sub,
            com_rg=com_rg,
            contrast_gain=contrast_gain,
            contrast_sub=contrast_sub,
            contrast_full=contrast_full,
            N_subaperture=np.int64(N_subaperture),
            box_phi_rad=box_phi,
            box_coh=box_coh,
            box_slope_rad_per_row=box_slope_rad_per_row,
            box_intercept_rad=box_intercept_rad,
            box_y0=box_y0,
            box_n_rows=box_n_rows,
            box_residual_rad=box_residual_rad,
            box_n_inliers=box_n_inliers,
            # Fraction of azimuth rows within ±phase_fit_thresh_rad of the linear fit (unweighted and coherence-weighted). Complementary to box_residual_rad (a coh-weighted *mean* absolute residual); this is a scale-free fraction that maps cleanly onto the "most rows within ±T" intuition. See fit_quality_out/WTW3YQ_fit_vs_af.png for its empirical coupling with the AF outcome.
            frac_within_1rad=frac_within_1rad,
            frac_within_1rad_coh=frac_within_1rad_coh,
            phase_fit_thresh_rad=np.float64(cfg.phase_fit_thresh_rad),
            phase_fit_frac1_min=np.float64(cfg.phase_fit_frac1_min),
            max_focus_targets=np.int64(cfg.max_focus_targets),
            phase_fit_score_floor=np.float64(
                getattr(cfg, "phase_fit_score_floor", 0.0)
            ),
            max_phase_residual_rad=np.float64(cfg.max_phase_residual_rad),
            min_row_coherence=np.float64(cfg.min_row_coherence),
            inlier_tol_rad=np.float64(cfg.inlier_tol_rad),
            v_az_mps=v_az_mps,
            v_rg_mps=v_rg_mps,
            is_moving_com=is_moving_com.astype(np.int8),
            is_moving_slope=is_moving_slope.astype(np.int8),
            is_moving_slope_box=is_moving_slope_box.astype(np.int8),
            is_moving_az_pp=is_moving_az_pp.astype(np.int8),
            az_pp_per_box=az_pp_per_box,
            az_pp_sub_thresh=np.float64(cfg.az_pp_sub_thresh),
            prf_hz=np.float64(cfg.prf),
            min_velocity_mps=np.float64(cfg.min_velocity_mps),
            # COM Theil-Sen linear-fit diagnostics (Phase-1: diagnostic only, does not gate). Per-axis velocities in m/s + RMS residuals in metres from a robust Theil-Sen fit of com_az_sub / com_rg vs subaperture-centre time. *_ts_mps = slope (m/s), com_ts_res_*_m = RMS residual (m). L2-combined magnitudes stored separately (v_com_ts_mps, com_ts_res_m) — L2 because a target moving in either axis contributes to |v|. n_fin_* counts how many subapertures had a finite COM going into each fit (out of N_subaperture). integration_time_s is included for downstream consumers that want to recompute the per-subaperture time step (dt = sub_size / prf_hz) or the total expected walk (v · integration_time_s).
            v_az_com_ts_mps=v_az_com_ts_mps,
            v_rg_com_ts_mps=v_rg_com_ts_mps,
            v_com_ts_mps=v_com_ts_mps,
            com_ts_res_az_m=com_ts_res_az_m,
            com_ts_res_rg_m=com_ts_res_rg_m,
            com_ts_res_m=com_ts_res_m,
            com_ts_n_fin_az=com_ts_n_fin_az,
            com_ts_n_fin_rg=com_ts_n_fin_rg,
            integration_time_s=np.float64(integration_time),
            # Decimated 0/1 mask on the subaperture-statistics grid (shape (sub_size, N_rg) with sub_size = N_az // N_sub- aperture). This is the isolated-pixel-filtered CoV mask returned by `cluster_targets` (`mask_filt`) — the same array the boundary_box seed peaks were picked from. Downstream consumers that want the RAW pre-filter mask should recompute it from `sub_var / (sub_mean**2 + eps)`. Also saved: the seed-peak `boundary_box` (one non-zero pixel per dense detection cluster on the full-res grid) and `boundary_box_peaks_yx`, a (K, 2) coordinate list of red-× seeds in FULL-RESOLUTION (N_az, N_rg) coordinates — divide y by N_subaperture to index into mask_dec. Note: the redundant scene-wide `s_degraded_abs = |s_degraded|` field was retired along with the scene-wide `amp_raw` allocation — per-box strips live under `amp_raw_boxes` instead, paired with `d_phase_box_coords` for their absolute pixel locations.
            mask_dec=mask_filt.astype(np.int8),
            boundary_box=boundary_box.astype(np.int8),
            boundary_box_peaks_yx=peaks_yx_full.astype(np.int64),
            # Pixel spacings.   *_s_degraded : the full-resolution grid where s_degraded,                  boundary_box, peaks, and the per-box                  d_phase / amp_raw strips live (azimuth                  unchanged from raw SLC; range decimated                  by Number_of_Range_Looks).   *_mask_dec   : the subaperture-statistics grid where                  mask_dec lives (azimuth =                  s_degraded_az × N_subaperture; range =                  same as s_degraded). Storing both is redundant with N_subaperture + range-looks but makes the npz self-describing for consumers that don't want to derive it.
            az_m_per_px_s_degraded=np.float64(azimuth_spacing),
            rg_m_per_px_s_degraded=np.float64(
                range_spacing * Number_of_Range_Looks
            ),
            az_m_per_px_mask_dec=np.float64(
                azimuth_spacing * N_subaperture
            ),
            rg_m_per_px_mask_dec=np.float64(
                range_spacing * Number_of_Range_Looks
            ),
            number_of_range_looks=np.int32(Number_of_Range_Looks),
        )
        try:
            data_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(data_path, **npz_payload)
            print(
                f"Saved per-box d_phase + per-box amp_raw + boxes + "
                f"mask_dec + boundary_box_peaks → {data_path}"
            )
        except PermissionError as exc:
            fallback = Path("/tmp") / data_path.name
            print(
                f"  (warning) cannot write {data_path} ({exc.strerror});"
                f" falling back to {fallback}"
            )
            np.savez_compressed(fallback, **npz_payload)
            print(
                f"Saved per-box d_phase + per-box amp_raw + boxes + "
                f"mask + boundary_box_peaks → {fallback}"
            )
        # Free the per-box d_phase / amp_raw strips as soon as the NPZ has been written; the rest of main() (display figures, per- box PNGs, etc.) computes any d_phase / |s_degraded| it needs on demand.
        del d_phase_boxes, d_phase_box_coords, amp_raw_boxes

    n_strong = int(strong_mask.sum())

    # Main figure panels — both in s_degraded pixel coords, so the strong-box overlay uses the box arrays as-is (no Number_of_Range_- Looks rescaling needed). Left: range-degraded SLC amplitude. Right: boundary_box peak map (one peak per dense detection cluster — the inputs to grow_and_recenter_boxes).
    # In minimal mode (`cfg.debug=False`) the main figure was never built (`ax_main` / `ax_bdy` are None), so skip the population.
    if fig is not None:
        _show_slc(
            ax_main, s_degraded,
            f"|s_degraded| (range-degraded SLC): "
            f"{n_strong}/{len(boxes_yxhw)} boxes "
            f"|slope·n| > {cfg.slope_rad_thresh:g} rad",
        )
        _overlay_boxes(ax_main, strong_boxes, color="red", lw=1.4)

        _show_slc(
            ax_bdy, boundary_box,
            f"boundary_box (peak per dense cluster): "
            f"{int(boundary_box.sum())} peaks",
        )
        _overlay_boxes(ax_bdy, strong_boxes, color="red", lw=1.4)

    # The d_phase panel of the old 2×3 layout is disabled to keep the main figure light. Re-enable by restoring `plt.subplots(2, 3, …)` above and uncommenting this block. _show_phase_derivative(     axes[1, 1], d_phase,     f"d_phase (degraded): {n_strong}/{len(boxes_yxhw)} boxes "     f"|slope·n| > {cfg.slope_rad_thresh:g} rad", ) _overlay_boxes(axes[1, 1], strong_boxes, color="red", lw=1.4)


    # n_rows = int(np.floor(np.sqrt(N_subaperture))) n_cols = int(np.ceil(N_subaperture / n_rows)) fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14), constrained_layout=True) axes_flat = np.atleast_1d(axes).ravel() for i in range(N_subaperture):     _show_slc(axes_flat[i], subapertures[i], f"subaperture {i}") for j in range(N_subaperture, axes_flat.size):     axes_flat[j].set_visible(False)

    # Per-pixel statistics across the N=8 sub-aperture amplitudes. Stationary scatterers → consistent |s_k| across looks → low variance. Moving targets / non-stationary scenes → |s_k| varies with Doppler band → high variance (and high variance / mean^2). sub_mean = subapertures.mean(axis=0) sub_var = subapertures.var(axis=0)

    # Plot a handful of range columns of d_phase as 1-D azimuth traces. plt.plot(2D) draws one line per column, with row index on the x-axis, so axis 0 (azimuth) ends up on the horizontal axis as requested. range_cols = slice(10, 15) plt.figure() plt.plot(d_phase[:, range_cols],'o') plt.xlabel("azimuth pixel") plt.ylabel(r"$\Delta\varphi$  [rad]") plt.title(f"d_phase along azimuth, range cols {range_cols.start}:{range_cols.stop}")

    # range_cols = slice(124, 134) plt.figure() plt.plot(d_phase[:, range_cols],'o') plt.xlabel("azimuth pixel") plt.ylabel(r"$\Delta\varphi$  [rad]") plt.title(f"d_phase along azimuth, range cols {range_cols.start}:{range_cols.stop}")

    # d = np.angle(s[1:, :] * np.conj(s[:-1, :])) range_cols = slice(640, 645) plt.figure() plt.plot(d[:, range_cols],'o') plt.xlabel("azimuth pixel") plt.ylabel(r"$\Delta\varphi$  [rad]") plt.title(f"d_phase along azimuth, range cols {range_cols.start}:{range_cols.stop}")


    # fig_cmp, axes_cmp = plt.subplots(1, 3, figsize=(20, 8), constrained_layout=True) _show_slc(axes_cmp[0], s_degraded, "|s_degraded|  (coherent)") _show_slc(     axes_cmp[1],     sub_var/sub_mean**2,     r"sub_var/sub_mean$_k\,|s_k|$  (incoherent average)",     sigma=1.0, ) _show_slc(     axes_cmp[2],     sub_var,     r"var$_k\,|s_k|$  (per-pixel variance over $N=8$ looks)",     sigma=1.0, )


    

    # Number of grown boxes — used by Figure 4 (--display-all-mode) and by the per-box save / NPZ blocks further down. Defined once, outside any gate, so the rest of the function can rely on it regardless of `--debug` / `--display-all-mode`.
    K = len(boxes_yxhw)

    # (The `--debug` `mask_full + grown boxes` overlay figure was retired: it materialised the upsampled binary mask scene-wide (`np.repeat(mask_filt, N_subaperture, axis=0)` at ≈(N_az, N_rg) float32 = ~1.5 GB on a full ICEYE scene), matplotlib held a reference to that array until the figure was saved, and it stacked on top of `s`, `s_degraded`, `subapertures` and the AF chip buffers. The decimated `mask_filt` is still saved in the NPZ via `mask_dec`, so the same overlay can be rebuilt offline from `mask_dec + N_subaperture + boxes_yxhw` without paying the RSS cost inside the pipeline. The Figure cv `_cov_map.png` panel and its dark/bright companion `_dark_bright.png` were retired earlier when the in-line CoV gate was absorbed into `cluster_targets`; recompute them from `sub_var / (sub_mean**2 + eps)` if you need to bring the panel back.)

    # --- Rich display figures (subapertures, d_phase fit, per-box COM) --- The expensive grid figures: per-Doppler-band subaperture amplitudes, the d_phase + per-box slope·n overview, and the per-box subaperture-COM trajectory grid. They scale poorly with K and N_subaperture, so they are gated on --display-all-mode and skipped entirely when it is unset (default).
    # Additionally suppressed in minimal mode (`cfg.debug=False`) so the fast-path run stays predictable.
    if _save_all and cfg.display_all_mode:
        # (Figure 2 — the scene-wide d_phase overview + per-box slope·n canvas — was retired: both arrays are ~(N_az, N_rg) at float64 (≈3 GB each on a full ICEYE scene), and the diagnostic never justified the RSS spike. Per-box slope·n is already stored in `<stem>_box_stats.csv` and the NPZ, so a lightweight per-box replacement can be reconstructed offline if needed.)

        # --- Figure 3: per-Doppler-band subaperture amplitudes --- Lay them out on a near-square grid. Subapertures are shape (N_subaperture, sub_size, N_range_degraded), already amplitudes.
        n_sub = subapertures.shape[0]
        n_cols_sub = int(np.ceil(np.sqrt(n_sub)))
        n_rows_sub = int(np.ceil(n_sub / n_cols_sub))
        fig3, axes3 = plt.subplots(
            n_rows_sub, n_cols_sub,
            figsize=(4 * n_cols_sub, 3 * n_rows_sub),
            constrained_layout=True,
        )
        fig3.suptitle(
            f"Subaperture amplitudes — {cfg.path.name}\n"
            f"N_subaperture = {n_sub} Doppler sub-bands, "
            f"sub_size = {subapertures.shape[1]} az px each",
            fontsize=11,
        )
        axes3_flat = np.atleast_1d(axes3).ravel()
        # Common amplitude clip so brightness is comparable across panels: mean ± 2σ over the whole stack, with vmin floored at 0 (amplitudes are non-negative).
        sub_mu = float(subapertures.mean())
        sub_sd = float(subapertures.std())
        sub_vmin = max(0.0, sub_mu - 2.0 * sub_sd)
        sub_vmax = max(sub_mu + 2.0 * sub_sd, sub_vmin + 1e-12)
        for k in range(n_sub):
            ax_k = axes3_flat[k]
            im_k = ax_k.imshow(
                subapertures[k], cmap="viridis", aspect="auto",
                vmin=sub_vmin, vmax=sub_vmax,
            )
            ax_k.set_title(f"subaperture {k}", fontsize=9)
            ax_k.set_xlabel("range pixel", fontsize=8)
            ax_k.set_ylabel("azimuth pixel", fontsize=8)
            ax_k.tick_params(labelsize=7)
        for j in range(n_sub, axes3_flat.size):
            axes3_flat[j].set_visible(False)
        fig3.colorbar(
            im_k, ax=axes3_flat.tolist(), location="right",
            label="|s_sub|", shrink=0.6,
        )

        # --- Figure 4: per-box COM trajectory across the N subapertures --- One small panel per box (whole population, not just strong-slope).
        if K:
            n_cols_box = int(np.ceil(np.sqrt(K)))
            n_rows_box = int(np.ceil(K / n_cols_box))
            fig4, axes4 = plt.subplots(
                n_rows_box, n_cols_box,
                figsize=(2.4 * n_cols_box, 2.2 * n_rows_box),
                constrained_layout=True,
            )
            fig4.suptitle(
                f"Per-box subaperture COM trajectory — {cfg.path.name}\n"
                f"x = range pixel (s_degraded),  y = azimuth pixel (subaperture grid),"
                f"  colour = subaperture index 0..{n_sub - 1}",
                fontsize=11,
            )
            axes4_flat = np.atleast_1d(axes4).ravel()
            sub_idx = np.arange(n_sub, dtype=np.float64)
            for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                ax_k = axes4_flat[k]
                cx = com_rg[k]
                cy = com_az_sub[k]
                valid = np.isfinite(cx) & np.isfinite(cy)
                if valid.any():
                    ax_k.plot(cx[valid], cy[valid], color="0.6", lw=0.7, zorder=1)
                    ax_k.scatter(
                        cx[valid], cy[valid],
                        c=sub_idx[valid], cmap="viridis",
                        s=22, edgecolor="k", linewidth=0.3, zorder=2,
                        vmin=0, vmax=n_sub - 1,
                    )
                slope_k = slope_totals[k]
                star = "*" if strong_mask[k] else " "
                ax_k.set_title(
                    f"{star}box {k:02d}  y={int(y_c)} x={int(x_c)}\n"
                    f"slope·n={slope_k:+.2f} rad" if np.isfinite(slope_k)
                    else f"{star}box {k:02d}  y={int(y_c)} x={int(x_c)}\nslope·n=NaN",
                    fontsize=8,
                )
                ax_k.set_xlabel("range pix", fontsize=7)
                ax_k.set_ylabel("az pix (sub)", fontsize=7)
                ax_k.tick_params(labelsize=6)
                ax_k.grid(True, alpha=0.3)
            for j in range(K, axes4_flat.size):
                axes4_flat[j].set_visible(False)

    # The per-box azimuth phase-derivative trace + wrapped linear fit used to live in a single Figure 5 grid (n_rows × n_cols of subplots). That grid became unreadable on K > 50 boxes and slow to render / open interactively, so it was split into one PNG per box. Those PNGs are written inside the cfg.save block below, together with the other per-box per_box_dir/box_NNN_*.png files.

    if cfg.save is not None:
        # Resolve the actual save directory once. Probe by trying to create cfg.save.parent; on PermissionError fall back to /tmp. This way main figures, per-box figures and data files all land together.
        try:
            cfg.save.parent.mkdir(parents=True, exist_ok=True)
            (cfg.save.parent / ".write_probe").write_text("")
            (cfg.save.parent / ".write_probe").unlink(missing_ok=True)
            save_dir = cfg.save.parent
        except PermissionError as exc:
            print(
                f"  (warning) cannot write into {cfg.save.parent} "
                f"({exc.strerror}); falling back to /tmp"
            )
            save_dir = Path("/tmp")

        stem = cfg.save.stem
        path_main = save_dir / cfg.save.name
        path_csv = save_dir / f"{stem}_box_stats.csv"
        per_box_dir = save_dir / f"{stem}_per_box"

        # Save the main overview figure + the other intermediate diagnostic figures. Written only when `cfg.debug=True`; in minimal mode only `{stem}_kept_vs_eliminated.png` (below) survives.
        if _save_all:
            if fig is not None:
                fig.savefig(path_main, dpi=150);          print(f"Saved → {path_main}")
            if fig_sub_mean_boxes is not None:
                path_sub_mean_boxes = save_dir / f"{stem}_sub_mean_boxes.png"
                fig_sub_mean_boxes.savefig(path_sub_mean_boxes, dpi=150)
                print(f"Saved → {path_sub_mean_boxes}")
            if fig_slc_boxes is not None:
                path_slc_boxes = save_dir / f"{stem}_slc_after_residual.png"
                fig_slc_boxes.savefig(path_slc_boxes, dpi=150)
                print(f"Saved → {path_slc_boxes}")
            if fig_final_boxes is not None:
                path_final_boxes = save_dir / f"{stem}_slc_final_boxes.png"
                fig_final_boxes.savefig(path_final_boxes, dpi=150)
                print(f"Saved → {path_final_boxes}")

        # `{stem}_kept_vs_eliminated.png` is a keeper — always written when `--save` is set, regardless of `cfg.debug`. Close the figure immediately after savefig in headless mode so matplotlib can release the ``np.abs(s_degraded)`` float32 amplitude buffer (~780 MB on full ICEYE scenes) that imshow holds by reference. Without this close, that buffer stays alive until ``plt.close("all")`` at the very end of ``main()`` — i.e. throughout the entire AF pre-compute loop, which is exactly where the OOM ceiling gets hit. In ``--show`` mode ``_maybe_close`` is a no-op.
        if fig_kept_vs_eliminated is not None:
            path_kve = save_dir / f"{stem}_kept_vs_eliminated.png"
            fig_kept_vs_eliminated.savefig(path_kve, dpi=150)
            print(f"Saved → {path_kve}")
            _maybe_close(fig_kept_vs_eliminated)

        # Debug-only artefact. Only the raw slope-per-box `.npy` survives under `--debug`; the former `mask_boxes.png` overlay and the scene-wide `_dphase.npy` dump were retired because both required scene-sized float allocations (see the "mask_full + grown boxes" and "fig2" retirement notes upstream).
        if cfg.debug:
            path_slopes = save_dir / f"{stem}_slopes.npy"
            np.save(path_slopes, slope_totals);       print(f"Saved → {path_slopes}")

            # (The CoV² map PNG / NPZ and dark/bright partition PNG were retired together with their driving arrays — see the fig_cv / fig_db construction site upstream.)

        # Rich display figures. Written only when `--display-all-mode` is set; they mirror the `if cfg.display_all_mode:` block that constructs `fig3` and `fig4` further up. Also gated by `_save_all` so minimal mode never emits them. Figure 2 (scene-wide d_phase + per-box slope·n) was retired to avoid the ~6 GB float64 materialisation on full ICEYE scenes.
        if _save_all and cfg.display_all_mode:
            path_subaps = save_dir / f"{stem}_subapertures.png"
            path_box_com = save_dir / f"{stem}_box_com.png"

            fig3.savefig(path_subaps, dpi=150);       print(f"Saved → {path_subaps}")
            _maybe_close(fig3)
            if K:
                fig4.savefig(path_box_com, dpi=150);  print(f"Saved → {path_box_com}")
                _maybe_close(fig4)
            # `fig3` had 16 imshow panels backed by views into `subapertures`; closing it in headless mode drops those references so the ≈1.5 GB float32 buffer can finally be released before the AF loop starts. In `--show` mode `_maybe_close` is a no-op and the buffer intentionally stays alive until the interactive window is closed.
            if not _SHOW_FIGURES and "subapertures" in locals():
                del subapertures

        # --- Pre-compute per-box "target score" fields --- Runs the polynomial range-walk / PGA autofocus once per survivor (`boxes_yxhw`) and caches the corrected chip + summary scalars in per-box arrays so both the CSV write below and the per-box artefact loop further down consume the same result without recomputing. Also derives `rg_pp_sub_px_arr` (COM range peak-to-peak across sub-apertures, analogous to the existing az_pp_sub_px) and `com_pp_total_m_arr` (combined |Δ COM| in metres) so the six per-box "target score" fields — COM movement, phase slope, n_detections, best_deviation, af gain, frac_within_1rad_coh — all land in `<stem>_box_stats.csv`.
        K_save = len(boxes_yxhw)

        # COM peak-to-peak per axis (native pixels on each grid) and combined magnitude in metres. `com_az_sub[k, j]` lives on the sub-aperture azimuth grid (compressed by N_subaperture vs the SLC), `com_rg[k, j]` on the s_degraded range grid (compressed by Number_of_Range_Looks). Convert both to metres before combining. Pure CSV-fill diagnostic — skipped in minimal mode.
        if _save_all and K_save:
            _az_finite_pp = np.isfinite(com_az_sub)
            _rg_finite_pp = np.isfinite(com_rg)
            _az_any_pp = _az_finite_pp.any(axis=1)
            _rg_any_pp = _rg_finite_pp.any(axis=1)
            az_pp_sub_px_arr = np.where(
                _az_any_pp,
                np.nanmax(com_az_sub, axis=1)
                - np.nanmin(com_az_sub, axis=1),
                np.nan,
            )
            rg_pp_sub_px_arr = np.where(
                _rg_any_pp,
                np.nanmax(com_rg, axis=1) - np.nanmin(com_rg, axis=1),
                np.nan,
            )
            _az_pp_m = np.where(
                np.isfinite(az_pp_sub_px_arr),
                az_pp_sub_px_arr * N_subaperture * azimuth_spacing,
                0.0,
            )
            _rg_pp_m = np.where(
                np.isfinite(rg_pp_sub_px_arr),
                rg_pp_sub_px_arr * Number_of_Range_Looks * range_spacing,
                0.0,
            )
            _any_finite_pp = (
                np.isfinite(az_pp_sub_px_arr)
                | np.isfinite(rg_pp_sub_px_arr)
            )
            com_pp_total_m_arr = np.where(
                _any_finite_pp,
                np.sqrt(_az_pp_m ** 2 + _rg_pp_m ** 2),
                np.nan,
            )
        else:
            az_pp_sub_px_arr = np.empty(0, dtype=np.float64)
            rg_pp_sub_px_arr = np.empty(0, dtype=np.float64)
            com_pp_total_m_arr = np.empty(0, dtype=np.float64)

        # Fraction of azimuth rows in the box whose per-row coherence exceeds the line-fit gate (`cfg.min_row_coherence`, default 0.5) — i.e. the share of rows the wrap-aware linear fit was allowed to use, expressed as (# high-coh rows) / (box azimuth size = box_n_rows). Complements `frac_within_1rad_coh`: `frac_within_1rad_coh` says "how much of the fit is tight", `frac_high_coh_rows` says "how much of the box actually voted". Kept rows only; empty for dropped rows because `box_coh` is only carried on the post-motion survivor set. Pure CSV-fill diagnostic — skipped in minimal mode.
        if _save_all and K_save:
            frac_high_coh_rows_arr = np.empty(K_save, dtype=np.float64)
            for _k_hc in range(K_save):
                _n_k_hc = int(box_n_rows[_k_hc])
                if _n_k_hc > 0:
                    _phi_hc = box_phi[_k_hc, :_n_k_hc]
                    _coh_hc = box_coh[_k_hc, :_n_k_hc]
                    _high_hc = (
                        np.isfinite(_phi_hc)
                        & np.isfinite(_coh_hc)
                        & (_coh_hc > cfg.min_row_coherence)
                    )
                    frac_high_coh_rows_arr[_k_hc] = (
                        float(_high_hc.sum()) / float(_n_k_hc)
                    )
                else:
                    frac_high_coh_rows_arr[_k_hc] = np.nan
        else:
            frac_high_coh_rows_arr = np.empty(0, dtype=np.float64)

        # Autofocus constants (shared with the AF PNG/NPZ writing loop further down). Kept at module-level defaults here so the loop below only reads them.
        af_dev_min_meters = -20.0
        af_dev_max_meters = 20.0
        af_accuracy = 0.5
        af_poly_degree = 2
        af_min_abs_deviation = 3.6
        # False-alarm short-circuit for the polynomial range-walk search, expressed in ground metres so it stays scene-independent (different ICEYE modes span ``range_spacing`` from ~0.1 m to ~0.3 m; a hard-coded samples value would silently move the gate around). When the coarse pass returns ``|best_deviation| * range_spacing < af_early_exit_threshold_m`` the fine pass and every downstream AF stage (final shift, inverse FFT, PGA, ``contrast_after``) are skipped — the box will be dropped by the ``af_min_abs_deviation`` gate below anyway, and skipping the ~40-eval fine pass + PGA on that (majority) population is where this stage claws back most of its wall time in false-alarm-heavy runs. Kept strictly below the metres-equivalent of ``af_min_abs_deviation`` so any coarse-pass verdict that could plausibly refine to above the delete gate still gets the fine pass.
        af_early_exit_threshold_m = 1.0
        af_chip_az_pad_frac = 0.10

        # Scene-scoped geometry for the range-walk → v_a mapping. Resolved once here so every AF call (log line) and every CSV / NPZ row (below) sees the same values. When the sidecar exposes ``iceye:processing_prf`` (spotlight scenes like SXK97E) the mapping uses the real, beam-steered value; stripmap scenes without it fall back to ``cfg.prf`` inside ``_af_scene_geometry`` — which is the correct approximation for that mode.
        (
            af_processing_prf_hz,
            af_wavelength_m,
            af_slant_range_m,
            af_v_sat_mps,
        ) = _af_scene_geometry(iceye_md, fallback_prf_hz=cfg.prf)
        print(
            f"[af] geometry: processing_prf={af_processing_prf_hz:.1f} Hz, "
            f"lambda={af_wavelength_m:.4f} m, "
            f"R0={af_slant_range_m/1e3:.1f} km, "
            f"v_sat={af_v_sat_mps:.1f} m/s"
        )

        # Per-box AF results (aligned with boxes_yxhw). NaN / -1 marks boxes whose chip cropping degenerated (empty or < 2 rows). `corrected_chip_af[k]` is the post-correction chip we need later to (a) paste into `s_with_corrected` for the `_af_corrected.png` overview and (b) write `box_{k:03d}_autofocus.npz`. `chip_bounds_af[k] = (y0_full, y1_full, x0_full, x1_full)` records the padded chip window used by the autofocus.
        best_dev_af = np.full(K_save, np.nan, dtype=np.float64)
        gain_db_af = np.full(K_save, np.nan, dtype=np.float64)
        n_sub_imp_af = np.full(K_save, -1, dtype=np.int32)
        best_look_af = np.full(K_save, -1, dtype=np.int32)
        c_before_af = np.full(K_save, np.nan, dtype=np.float64)
        c_after_af = np.full(K_save, np.nan, dtype=np.float64)
        corrected_chip_af: list[np.ndarray | None] = [None] * K_save
        chip_bounds_af: list[tuple[int, int, int, int] | None] = (
            [None] * K_save
        )
        if K_save:
            H_az_full_save, W_rg_full_save = s.shape
            # RSS logging cadence: print current + peak RSS every ``_af_rss_report_every`` AF iterations so the log carries a memory trajectory (useful when a scene later OOMs — we can see whether resident stayed flat or climbed). Fixed cadence rather than a percentage of ``K_save`` so short and long runs both get ~5–20 log lines. Also print at box 1 and the final box so start/end watermarks are always captured. ``current`` should stay roughly flat across the loop if ``_release_glibc_arenas`` is holding; a rising ``peak`` means an alloc/free burst that briefly went above steady state.
            _af_rss_report_every = max(1, K_save // 12)
            _rss_cur_start, _rss_peak_start = _rss_gb()
            print(
                f"[af] pre-compute start: current RSS = "
                f"{_rss_cur_start:.2f} GB  (peak = {_rss_peak_start:.2f} GB)"
            )
            for _k_af, (_yc_af, _xc_af, _h_af, _w_af) in enumerate(
                boxes_yxhw
            ):
                _y_int = int(_yc_af)
                _h_int = int(_h_af)
                _az_pad = int(round(_h_int * af_chip_az_pad_frac))
                _y0_box = _y_int - _h_int // 2
                _y1_box = _y0_box + _h_int
                _y0_full = max(0, _y0_box - _az_pad)
                _y1_full = min(H_az_full_save, _y1_box + _az_pad)
                _x_full = int(_xc_af) * Number_of_Range_Looks
                _w_full = int(_w_af) * Number_of_Range_Looks
                _x0_full = max(0, _x_full - _w_full // 2)
                _x1_full = min(W_rg_full_save, _x0_full + _w_full)
                if _x1_full <= _x0_full or _y1_full - _y0_full < 2:
                    continue
                _chip_full = s[_y0_full:_y1_full, _x0_full:_x1_full]
                if _chip_full.size == 0 or _chip_full.shape[0] < 2:
                    continue
                (
                    _corr_k, _dev_k, _n_sub_imp_k, _best_look_k,
                    _c_before_k, _c_after_k, _gain_db_k,
                ) = _af_apply_global_range_deviation_correction(
                    _chip_full,
                    range_spacing=range_spacing,
                    dev_min_meters=af_dev_min_meters,
                    dev_max_meters=af_dev_max_meters,
                    accuracy=af_accuracy, poly_degree=af_poly_degree,
                    early_exit_threshold_m=af_early_exit_threshold_m,
                    progress_index=_k_af + 1,
                    progress_total=K_save,
                    processing_prf_hz=af_processing_prf_hz,
                    wavelength_m=af_wavelength_m,
                    slant_range_m=af_slant_range_m,
                    v_sat_mps=af_v_sat_mps,
                )
                best_dev_af[_k_af] = float(_dev_k)
                gain_db_af[_k_af] = float(_gain_db_k)
                n_sub_imp_af[_k_af] = int(_n_sub_imp_k)
                best_look_af[_k_af] = int(_best_look_k)
                c_before_af[_k_af] = float(_c_before_k)
                c_after_af[_k_af] = float(_c_after_k)
                chip_bounds_af[_k_af] = (
                    _y0_full, _y1_full, _x0_full, _x1_full,
                )
                # Only retain the corrected chip when the box will actually survive the `af_min_abs_deviation` gate. Boxes that will be dropped (early-exit with `_corr_k is None` — already free — plus the narrow `[af_early_exit_threshold_m / range_spacing, af_min_abs_deviation)` samples-domain window where AF produced a chip we're about to throw away) leave `corrected_chip_af[_k_af]` as its `None` list default. The downstream loop still counts them via `chip_bounds_af[_k_af]` + `best_dev_af[_k_af]`, so the CSV and `n_af_filtered` bookkeeping is unchanged; the only difference is that a busy scene with hundreds of large-vessel boxes no longer stacks their corrected chips (each up to tens of MB of complex64) in RAM through the rest of the loop. Explicit `_corr_k = None` on the drop path releases the local reference before the next iteration overwrites it, so RSS actually shrinks between iterations for dropped boxes instead of holding one extra chip alive.
                if _corr_k is not None and abs(float(_dev_k)) >= af_min_abs_deviation:
                    # Minimal mode never touches the complex chip again — the only downstream consumer is the amplitude paste into `s_amp` — so collapse to float32 amplitude immediately. Halves the retained bytes per box (complex64 → float32) and, importantly, `np.abs(complex64)` releases the complex64 buffer as soon as the amplitude is written, so the peak is one-chip transient rather than two-chip. Debug mode still needs the complex chip for `box_NNN_autofocus.png` and `box_NNN_autofocus.npz`, so keep it as-is there.
                    if _save_all:
                        corrected_chip_af[_k_af] = _corr_k
                    else:
                        corrected_chip_af[_k_af] = np.abs(_corr_k).astype(
                            np.float32, copy=False,
                        )
                _corr_k = None
                # Release Python refs and hand freed heap arenas back to the OS. Each AF call performs ~10 chip-sized complex64 allocations (spatch_fft, phase_ramp workspace, PGA look_window / centered_spectrum / correction FFT temps, entropy `|x|^2` copies, …) — over 200 boxes that is ~40k large mallocs. glibc's per-thread arena keeps freed blocks on internal free-lists and, on default settings, never returns them to the OS. On a high-accept-rate scene like SXK97E where ~90% of boxes trigger the full PGA path, the arena grows monotonically and the process crosses the 125 GB physical-RAM ceiling long before the loop finishes even though live Python objects stay flat. ``_release_glibc_arenas`` runs Python's cyclic GC and then ``malloc_trim(0)`` so the arena hands its idle blocks back to the kernel each iteration — a couple of ms per call, orders of magnitude cheaper than a full AF+PGA pass on a big chip.
                _release_glibc_arenas()
                # Log current + peak RSS at a fixed cadence + at both endpoints of the loop so the log carries a memory trajectory even on runs that eventually get SIGKILLed. A flat ``current`` trace means the trimmed-arena strategy is holding; a rising ``current`` trace means something upstream of the arena is still leaking (real live objects, a growing Python container, matplotlib figure held open, …). ``peak`` is monotonic — it only tells us the transient high-water mark reached anywhere in the run.
                if (
                    _k_af == 0
                    or _k_af == K_save - 1
                    or (_k_af + 1) % _af_rss_report_every == 0
                ):
                    _rss_cur, _rss_peak = _rss_gb()
                    print(
                        f"[af] [{_k_af + 1}/{K_save}] current RSS = "
                        f"{_rss_cur:.2f} GB  (peak = {_rss_peak:.2f} GB)"
                    )

        # Per-box CSV — one row per *cluster_targets* box (i.e. per
        # row of `boxes_yxhw_initial`), NOT one row per final
        # survivor. The `disposition` column records where in the
        # filter chain each box left the pipeline.
        # Fields:
        #   init_idx            — row index in boxes_yxhw_initial
        #                         (i.e. cluster_targets output order).
        #   final_idx           — row index in the final `boxes_yxhw`
        #                         (matches box_<idx>.png / NPZ row
        #                         order). Empty for dropped boxes.
        #   disposition         — "kept" or one of the
        #                         "dropped_by_<stage>" values below.
        #                         Stages are checked from latest to
        #                         earliest, so each box is attributed
        #                         to the first filter that removed it:
        #                            dropped_by_nested_box_overlap
        #                            dropped_by_phase_slope
        #                            dropped_by_phase_residual
        #                            dropped_by_com_pp
        #                            dropped_by_motion_3tier
        #                            dropped_by_phase_fit_quality
        #                            dropped_by_refocus_gain
        #                            dropped_by_subap_sharpness
        #   y_c, x_c, h, w      — box centre + size on s_degraded grid
        #                         (from boxes_yxhw_initial).
        #   is_strong           — 1 if |slope·n_rows| ≥ slope_rad_thresh
        #                         (the "clear yes" tier of the three-
        #                         tier motion filter; = strong_mask,
        #                         which drives the red overlays in
        #                         Figure 1). 0 for boxes that only
        #                         passed via the middle-band az_pp
        #                         tiebreaker. Empty for dropped boxes
        #                         (is_strong is only computed for the
        #                         final survivors).
        #   slope_total_rad     — slope · n_rows (rad).
        #   slope_rad_per_row   — phase-slope filter output.
        #   phase_residual_rad  — phase-residual filter output.
        #   frac_within_1rad    — fraction of azimuth rows within
        #                         ±--phase-fit-thresh-rad of the linear
        #                         fit (unweighted). Complements
        #                         phase_residual_rad (mean-|r|) with a
        #                         "how much of the ramp actually fits"
        #                         fraction.
        #   frac_within_1rad_coh — same fraction, coherence-weighted.
        #                         Optionally used as the gate driven by
        #                         --phase-fit-frac1-min (before the
        #                         refocus-gain / autofocus stages).
        #   frac_high_coh_rows  — # azimuth rows with per-row coherence
        #                         above the line-fit gate
        #                         (`cfg.min_row_coherence`, default 0.5)
        #                         divided by the box azimuth size
        #                         (`box_n_rows`). I.e. the share of rows
        #                         the wrap-aware linear fit was allowed
        #                         to use. Complements
        #                         `frac_within_1rad_coh` ("how tight the
        #                         fit is") with "how much of the box
        #                         actually voted". Empty for dropped
        #                         boxes (`box_coh` only exists on the
        #                         post-motion survivor set).
        #   az_pp_sub_px        — COM azimuth peak-to-peak across N sub-
        #                         apertures (middle-band tiebreaker).
        #   rg_pp_sub_px        — COM range peak-to-peak across N sub-
        #                         apertures (analogous to az_pp_sub_px,
        #                         on the s_degraded range grid).
        #   com_pp_total_m      — L2 combined COM peak-to-peak in metres:
        #                         sqrt((az_pp·N_sub·az_spacing)² +
        #                              (rg_pp·N_rng_looks·rg_spacing)²).
        #                         Primary "COM movement" magnitude used
        #                         for the target score.
        #   n_detections        — # of mask_filt cells covered by the box
        #                         footprint (isolated-pixel-filtered CoV
        #                         mask from cluster_targets, on the
        #                         decimated sub-aperture grid). Set for
        #                         EVERY row — kept + dropped — because
        #                         the mask never changes after clustering.
        #   best_deviation      — winning polynomial range-walk deviation
        #                         from the per-box autofocus (mirrors
        #                         `core.autofocus.apply_global_range_
        #                         deviation_correction`). Empty for
        #                         dropped boxes; also empty if the chip
        #                         cropping degenerated (< 2 rows).
        #   af_gain_db          — post-autofocus contrast gain in dB
        #                         (same-BW, computed inside
        #                         `_af_apply_global_range_deviation_
        #                         correction`). Empty for dropped boxes.
        #   v_az_mps, v_rg_mps  — COM linear-fit velocities (m/s).
        #   contrast_gain       — --refocus-min-gain-db diagnostic.
        #   v_az_com_ts_mps,    — Robust Theil-Sen COM linear-fit
        #   v_rg_com_ts_mps,      slope (m/s) and RMS residual (m) per
        #   com_ts_res_az_m,      axis, plus L2 combined magnitude
        #   com_ts_res_rg_m,      (v_com_ts_mps) and combined residual
        #   v_com_ts_mps,         (com_ts_res_m). Diagnostic-only:
        #   com_ts_res_m,         used to sanity-check the middle-band
        #   com_ts_n_fin_az,      tie-breaker rule but not gating yet.
        #   com_ts_n_fin_rg       n_fin_* = # finite subapertures used
        #                         in the fit per axis (out of N_sub).
        # The five stat / velocity columns and `is_strong` are only
        # populated for `disposition == "kept"` rows — the earlier
        # filters return only their survivors, so the per-box arrays
        # (slope_totals, com_az_sub, v_az_mps, …) are indexed on the
        # post-final set. Full linear-fit reconstruction
        # (intercept_rad, y0, n_rows, n_inliers) and the per-sub-
        # aperture arrays live in the NPZ.
        # Disposition machinery — pure CSV-fill diagnostic. Built only in debug mode; the disposition logic is a linear scan of the eight stage-snapshot sets so on large scenes the initial dict build + O(K_initial × 8) membership checks below dominates the "save outputs" wall time.
        if _save_all:
            _final_idx_by_tuple = {
                tuple(int(v) for v in row): k
                for k, row in enumerate(boxes_yxhw)
            }

            # Stage snapshots gathered along the pipeline (all sets of
            # (y_c, x_c, h, w) int tuples). Each set contains the boxes
            # still alive at the end of that stage. Ordered latest-first
            # so the `disposition` logic below returns on the first
            # membership hit.
            _stage_snapshots = (
                ("kept",                        _final_idx_by_tuple.keys()),
                ("dropped_by_subap_sharpness",  kept_after_refocus),
                ("dropped_by_refocus_gain",     kept_after_fit_quality),
                ("dropped_by_phase_fit_quality", kept_after_motion),
                ("dropped_by_motion_3tier",     kept_after_com),
                ("dropped_by_com_pp",           kept_after_residual),
                ("dropped_by_phase_residual",   kept_after_slope),
                ("dropped_by_phase_slope",      kept_after_nested),
            )

            def _disposition(t: tuple) -> str:
                for label, snapshot in _stage_snapshots:
                    if t in snapshot:
                        return label
                # Not in any post-nested snapshot: dropped by the
                # nested-box cleanup (the earliest stage of the chain).
                return "dropped_by_nested_box_overlap"

        # `{stem}_box_stats.csv` — one row per cluster_targets box, disposition attribution + kept-box statistics. Written only in debug mode (`cfg.debug=True`); a bulk diagnostic dump with no consumer in minimal mode.
        if _save_all:
            with path_csv.open("w") as f:
                f.write(
                    "init_idx,final_idx,disposition,y_c,x_c,h,w,"
                    "n_detections,"
                    "is_strong,slope_total_rad,slope_rad_per_row,"
                    "phase_residual_rad,frac_within_1rad,frac_within_1rad_coh,"
                    "frac_high_coh_rows,"
                    "az_pp_sub_px,rg_pp_sub_px,com_pp_total_m,"
                    "best_deviation,af_gain_db,"
                    "v_az_mps,v_rg_mps,"
                    "contrast_gain,"
                    "v_az_com_ts_mps,v_rg_com_ts_mps,v_com_ts_mps,"
                    "com_ts_res_az_m,com_ts_res_rg_m,com_ts_res_m,"
                    "com_ts_n_fin_az,com_ts_n_fin_rg\n"
                )
                n_csv_rows = 0
                n_csv_kept = 0
                n_csv_strong = 0
                _disposition_counts: dict[str, int] = {}
                for init_idx, (y_c, x_c, h, w) in enumerate(boxes_yxhw_initial):
                    t = (int(y_c), int(x_c), int(h), int(w))
                    disp = _disposition(t)
                    _disposition_counts[disp] = _disposition_counts.get(disp, 0) + 1
                    _n_det_str = str(int(n_det_per_initial_box[init_idx]))
                    if disp == "kept":
                        k = _final_idx_by_tuple[t]
                        _az_pp_k = float(az_pp_sub_px_arr[k])
                        _rg_pp_k = float(rg_pp_sub_px_arr[k])
                        _com_pp_m_k = float(com_pp_total_m_arr[k])
                        _best_dev_k = float(best_dev_af[k])
                        _gain_db_k = float(gain_db_af[k])
                        _frac_hc_k = float(frac_high_coh_rows_arr[k])
                        _strong_k = bool(strong_mask[k])
                        fields: list[str] = [
                            str(init_idx), str(k), disp,
                            str(int(y_c)), str(int(x_c)),
                            str(int(h)), str(int(w)),
                            _n_det_str,
                            "1" if _strong_k else "0",
                            (f"{slope_totals[k]:+.6e}"
                             if np.isfinite(slope_totals[k]) else ""),
                            (f"{box_slope_rad_per_row[k]:+.6e}"
                             if np.isfinite(box_slope_rad_per_row[k]) else ""),
                            (f"{box_residual_rad[k]:.6f}"
                             if np.isfinite(box_residual_rad[k]) else ""),
                            (f"{frac_within_1rad[k]:.6f}"
                             if np.isfinite(frac_within_1rad[k]) else ""),
                            (f"{frac_within_1rad_coh[k]:.6f}"
                             if np.isfinite(frac_within_1rad_coh[k]) else ""),
                            (f"{_frac_hc_k:.6f}"
                             if np.isfinite(_frac_hc_k) else ""),
                            f"{_az_pp_k:.6f}" if np.isfinite(_az_pp_k) else "",
                            f"{_rg_pp_k:.6f}" if np.isfinite(_rg_pp_k) else "",
                            (f"{_com_pp_m_k:.6f}"
                             if np.isfinite(_com_pp_m_k) else ""),
                            (f"{_best_dev_k:+.6f}"
                             if np.isfinite(_best_dev_k) else ""),
                            (f"{_gain_db_k:+.6f}"
                             if np.isfinite(_gain_db_k) else ""),
                            (f"{v_az_mps[k]:+.6e}"
                             if np.isfinite(v_az_mps[k]) else ""),
                            (f"{v_rg_mps[k]:+.6e}"
                             if np.isfinite(v_rg_mps[k]) else ""),
                            (f"{contrast_gain[k]:.6f}"
                             if np.isfinite(contrast_gain[k]) else ""),
                            (f"{v_az_com_ts_mps[k]:+.6e}"
                             if np.isfinite(v_az_com_ts_mps[k]) else ""),
                            (f"{v_rg_com_ts_mps[k]:+.6e}"
                             if np.isfinite(v_rg_com_ts_mps[k]) else ""),
                            (f"{v_com_ts_mps[k]:.6e}"
                             if np.isfinite(v_com_ts_mps[k]) else ""),
                            (f"{com_ts_res_az_m[k]:.6e}"
                             if np.isfinite(com_ts_res_az_m[k]) else ""),
                            (f"{com_ts_res_rg_m[k]:.6e}"
                             if np.isfinite(com_ts_res_rg_m[k]) else ""),
                            (f"{com_ts_res_m[k]:.6e}"
                             if np.isfinite(com_ts_res_m[k]) else ""),
                            str(int(com_ts_n_fin_az[k])),
                            str(int(com_ts_n_fin_rg[k])),
                        ]
                        n_csv_kept += 1
                        if _strong_k:
                            n_csv_strong += 1
                    else:
                        # Box was dropped somewhere along the chain — we
                        # do not carry per-box stats for pre-motion drop
                        # stages (the corresponding arrays only exist for
                        # the post-COM survivor set), so leave them empty.
                        # `n_detections` is the exception: it depends only
                        # on the box footprint + mask_filt, both of which
                        # exist for every initial box, so we fill it in
                        # for dropped rows too. 22 numeric columns after
                        # (init_idx, final_idx, disposition, y_c, x_c, h,
                        # w, n_detections) — keep this list aligned with
                        # the header when adding columns.
                        fields = [
                            str(init_idx), "", disp,
                            str(int(y_c)), str(int(x_c)),
                            str(int(h)), str(int(w)),
                            _n_det_str,
                            "", "", "", "", "", "", "",
                            "", "", "",
                            "", "",
                            "", "", "",
                            "", "", "", "", "", "", "", "",
                        ]
                    f.write(",".join(fields) + "\n")
                    n_csv_rows += 1
            _disp_summary = ", ".join(
                f"{name}={_disposition_counts[name]}"
                for name in (
                    "kept",
                    "dropped_by_nested_box_overlap",
                    "dropped_by_phase_slope",
                    "dropped_by_phase_residual",
                    "dropped_by_com_pp",
                    "dropped_by_motion_3tier",
                    "dropped_by_phase_fit_quality",
                    "dropped_by_refocus_gain",
                    "dropped_by_subap_sharpness",
                )
                if name in _disposition_counts
            )
            print(
                f"Saved → {path_csv} ({n_csv_rows} rows, "
                f"{n_csv_kept} kept [{n_csv_strong} strong / "
                f"{n_csv_kept - n_csv_strong} middle-band], "
                f"{n_csv_rows - n_csv_kept} dropped) — {_disp_summary}"
            )

        # Per-box artefacts. The 4×4 subaperture-crops figure (box_NNN.png), the per-box azimuth-FFT figure (box_NNN_fft.png) and the per-subaperture FFT figure (box_NNN_sub_fft.png) were dropped — none of them are inspected any more. Only the per-box d_phase phase trace + wrapped linear fit (box_NNN_dphase.png) is still written. In minimal mode (`cfg.debug=False`) we still need to iterate the surviving boxes below to run the polynomial-range-walk / PGA autofocus and paste the corrected chips back into `s` for the `af_before` / `af_after` keeper PNGs, but every disk write (per-box PNG / NPZ, `autofocus_summary.csv`, refocus PNGs) is skipped.
        if K:
            if _save_all:
                per_box_dir.mkdir(parents=True, exist_ok=True)

            # --- Per-box d_phase phase trace + wrapped linear fit --- One PNG per box, replacing the old dense Figure 5 grid. Same content as before (scatter of phi vs absolute az row, coloured by per-row coherence, plus the wrapped grid-fit red line and the diagnostics in the title) but at a readable size and individually openable. Diagnostic only — the loop iterates over an empty sequence when `cfg.debug=False`, so no PNGs are written and the "Saved N per-box phase PNGs" summary reports 0.
            for k, (y_c, x_c, h, w) in (enumerate(boxes_yxhw) if _save_all else ()):
                n_k = int(box_n_rows[k])
                slope_k_pr = float(box_slope_rad_per_row[k])
                b_k = float(box_intercept_rad[k])
                star = "*" if strong_mask[k] else " "
                fig_dphi, ax_dphi = plt.subplots(
                    figsize=(10, 4.5), constrained_layout=True,
                )
                n_used_k = 0
                if n_k >= 1:
                    i_loc = np.arange(n_k, dtype=np.float64)
                    y_abs = int(box_y0[k]) + i_loc
                    phi_k = box_phi[k, :n_k]
                    coh_k = box_coh[k, :n_k]
                    valid = np.isfinite(phi_k)
                    if valid.any():
                        sc = ax_dphi.scatter(
                            y_abs[valid], phi_k[valid],
                            c=coh_k[valid], cmap="viridis",
                            s=10, vmin=0.0, vmax=1.0,
                            edgecolor="none", zorder=2,
                        )
                        fig_dphi.colorbar(
                            sc, ax=ax_dphi, location="right",
                            shrink=0.85, label="coh",
                        )
                    used_mask = (
                        np.isfinite(phi_k) & np.isfinite(coh_k)
                        & (coh_k > cfg.min_row_coherence)
                    )
                    n_used_k = int(used_mask.sum())
                    if np.isfinite(slope_k_pr) and np.isfinite(b_k):
                        line_wrapped = np.angle(
                            np.exp(1j * (slope_k_pr * i_loc + b_k))
                        )
                        if line_wrapped.size > 1:
                            d_line = np.abs(np.diff(line_wrapped))
                            jumps = np.where(d_line > np.pi)[0]
                            if jumps.size:
                                line_wrapped = line_wrapped.copy()
                                line_wrapped[jumps] = np.nan
                        ax_dphi.plot(
                            y_abs, line_wrapped,
                            color="red", lw=1.1, zorder=3,
                        )
                ax_dphi.set_ylim(-np.pi, np.pi)
                ax_dphi.axhline(0.0, color="0.5", lw=0.5, ls="--", zorder=1)
                slope_tot_k = slope_totals[k]
                res_k = float(box_residual_rad[k])
                vaz_k = float(v_az_mps[k])
                vrg_k = float(v_rg_mps[k])
                v_str = (
                    f"  v=({vaz_k:+.2f},{vrg_k:+.2f}) m/s"
                    if (np.isfinite(vaz_k) or np.isfinite(vrg_k))
                    else ""
                )
                coh_pct_k = (100.0 * n_used_k / n_k) if n_k > 0 else 0.0
                coh_str = (
                    f"  coh>{cfg.min_row_coherence:g}: "
                    f"{n_used_k}/{n_k} ({coh_pct_k:.1f}%)"
                )
                n_in_k = int(box_n_inliers[k])
                in_pct_k = (100.0 * n_in_k / n_k) if n_k > 0 else 0.0
                inlier_str = (
                    f"  inliers(d≤{cfg.inlier_tol_rad:g}): "
                    f"{n_in_k}/{n_k} ({in_pct_k:.1f}%)"
                )
                if np.isfinite(slope_tot_k) and np.isfinite(slope_k_pr):
                    res_str = (
                        f"  res={res_k:.2f} rad"
                        if np.isfinite(res_k) else ""
                    )
                    ax_dphi.set_title(
                        f"{star}box {k:02d}  y={int(y_c)} x={int(x_c)}"
                        f"{v_str}\n"
                        f"slope={np.degrees(slope_k_pr):+.2f}°/row, "
                        f"slope·n={slope_tot_k:+.2f} rad"
                        f"{res_str}{coh_str}{inlier_str}",
                        fontsize=9,
                    )
                else:
                    ax_dphi.set_title(
                        f"{star}box {k:02d}  y={int(y_c)} x={int(x_c)}"
                        f"{v_str}\nfit failed{coh_str}{inlier_str}",
                        fontsize=9,
                    )
                ax_dphi.set_xlabel("absolute az row")
                ax_dphi.set_ylabel(r"$\Delta\varphi$  [rad]")
                ax_dphi.grid(True, alpha=0.3)

                path_box_dphi_k = per_box_dir / f"box_{k:03d}_dphase.png"
                fig_dphi.savefig(path_box_dphi_k, dpi=130)
                _maybe_close(fig_dphi)
            if _save_all:
                print(
                    f"Saved {K} per-box phase PNGs → "
                    f"{per_box_dir}/box_000_dphase.png … "
                    f"box_{K-1:03d}_dphase.png"
                )

            # --- Per-box azimuth FFT on the FULL-RESOLUTION SLC --- The detections live on the `s_degraded` range grid (axis 1 decimated by `Number_of_Range_Looks`). To crop the same boxes from the full-resolution SLC `s`, rescale the range coordinates (x_centre and w) by `Number_of_Range_Looks`; y_centre and h are unchanged because azimuth is untouched by range degradation. The cropped chip's azimuth FFT `|FFT_az|` is written to `box_NNN_fft.png` next to `box_NNN_dphase.png`. Diagnostic only — loop body iterates over an empty sequence when `cfg.debug=False`.
            H_az_full, W_rg_full = s.shape
            n_fft_written = 0
            for k, (y_c, x_c, h, w) in (enumerate(boxes_yxhw) if _save_all else ()):
                y_int = int(y_c)
                h_int = int(h)
                y0_full = max(0, y_int - h_int // 2)
                y1_full = min(H_az_full, y0_full + h_int)
                x_full = int(x_c) * Number_of_Range_Looks
                w_full = int(w) * Number_of_Range_Looks
                x0_full = max(0, x_full - w_full // 2)
                x1_full = min(W_rg_full, x0_full + w_full)
                if x1_full <= x0_full or y1_full - y0_full < 2:
                    continue
                chip_full = s[y0_full:y1_full, x0_full:x1_full]
                if chip_full.size == 0:
                    continue
                fft_az = np.fft.fftshift(
                    np.fft.fft(chip_full, axis=0), axes=0,
                )
                amp_fft = np.abs(fft_az).astype(np.float32)
                n_az_chip = chip_full.shape[0]
                # imshow extent: row 0 of the fftshifted FFT is the most negative Doppler bin, so map it to the top of the axes (top < bottom inverts the y-axis as desired).
                f_top = -0.5
                f_bot = 0.5 - 1.0 / n_az_chip

                fig_fft, ax_fft = plt.subplots(
                    figsize=(7, 5), constrained_layout=True,
                )
                vmin = float(np.nanpercentile(amp_fft, 1.0))
                vmax = float(np.nanpercentile(amp_fft, 99.0))
                if vmax <= vmin:
                    vmax = vmin + 1.0
                im_fft = ax_fft.imshow(
                    amp_fft, aspect="auto", cmap="viridis",
                    vmin=vmin, vmax=vmax,
                    extent=(0, chip_full.shape[1], f_bot, f_top),
                )
                fig_fft.colorbar(im_fft, ax=ax_fft, label="|FFT|")
                star_fft = "*" if bool(strong_mask[k]) else " "
                ax_fft.set_title(
                    f"{star_fft}box {k:03d}  y={y_int} x={int(x_c)} "
                    f"h={h_int} w={int(w)}  "
                    f"|FFT_az(full-res chip)|",
                    fontsize=10,
                )
                ax_fft.set_xlabel("range column (full-res, in box)")
                ax_fft.set_ylabel("Doppler bin  [cycles / az pixel]")

                out_path_fft = per_box_dir / f"box_{k:03d}_fft.png"
                fig_fft.savefig(out_path_fft, dpi=110)
                _maybe_close(fig_fft)
                n_fft_written += 1
            if _save_all:
                print(
                    f"Saved {n_fft_written} per-box azimuth FFT PNGs → "
                    f"{per_box_dir}/box_000_fft.png … "
                    f"box_{K-1:03d}_fft.png"
                )

            # --- Per-box polynomial range-walk autofocus (PNG + NPZ) --- Consumes the AF results pre-computed above (`best_dev_af`, `gain_db_af`, `corrected_chip_af`, `chip_bounds_af`, …) so we never call `_af_apply_global_range_deviation_correction` twice on the same chip. For each surviving box whose chip was AF-focused and passed the |best_deviation| ≥ af_min_abs_deviation gate, this loop writes `box_NNN_autofocus.png` (1×2: |chip| before / |chip| after), `box_NNN_autofocus.npz`, and a row in `autofocus_summary.csv`. Boxes below the deviation threshold are skipped here; their AF results (`best_deviation`, `af_gain_db`) still land in `<stem>_box_stats.csv` so the full deviation histogram remains available to downstream consumers.
            af_csv_lines = [
                "box_idx,y,x,h,w,h_chip,w_chip,"
                "best_deviation,n_sub_contrast_improved,"
                "best_look_rows,"
                "contrast_before,contrast_after,gain_db,"
                "v_a_from_walk_mps"
            ]
            n_af_written = 0
            n_af_filtered = 0
            # Collect [y_centre, x_centre_full, h, w_full] for every box that passes the autofocus gate so the surviving set can be overlaid on `|s|` at the end of the loop.
            af_kept_boxes_full: list[tuple[float, float, float, float]] = []
            # Parallel list of (y0, y1, x0, x1, corrected_chip) for every surviving box so a second full-image figure can paste the PGA-refined chips back into `|s|`.
            af_kept_chip_data: list[
                tuple[int, int, int, int, np.ndarray]
            ] = []
            # Parallel list of the coherence-weighted linear-fit slope · n_rows (rad) for every surviving box — used to colour the overlaid rectangles on the AF full-image figures.
            af_kept_slope_totals: list[float] = []
            # Parallel list of `best_deviation` (signed samples of range-walk correction) for every surviving box — drives the red (positive) / yellow (negative) two-tone overlay on the AF full-image figures.
            af_kept_best_dev_totals: list[float] = []
            for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                bounds_k = chip_bounds_af[k]
                corrected_k = corrected_chip_af[k]
                # Two failure modes land here: (a) chip cropping degenerated before AF ran, so ``best_dev_af[k]`` is still NaN — a silent skip, not counted as "filtered by |dev|" since the box never got a verdict; (b) the AF early-exit short-circuit returned ``corrected=None`` because the coarse pass already saw ``|dev| * range_spacing < af_early_exit_threshold_m``, so ``best_dev_af[k]`` is a finite small number — that's a real "filtered by |dev|" event and belongs in ``n_af_filtered`` alongside the boxes filtered further below.
                if bounds_k is None or corrected_k is None:
                    if (
                        bounds_k is not None
                        and np.isfinite(best_dev_af[k])
                        and abs(float(best_dev_af[k])) < af_min_abs_deviation
                    ):
                        n_af_filtered += 1
                    continue
                y0_full, y1_full, x0_full, x1_full = bounds_k
                chip_full = s[y0_full:y1_full, x0_full:x1_full]
                y_int = int(y_c)
                h_int = int(h)
                best_dev_k = float(best_dev_af[k])
                n_sub_imp_k = int(n_sub_imp_af[k])
                best_look_rows_k = int(best_look_af[k])
                c_before_k = float(c_before_af[k])
                c_after_k = float(c_after_af[k])
                gain_db_k = float(gain_db_af[k])

                if not np.isfinite(best_dev_k):
                    continue

                # Skip boxes where the correction is negligible. The full deviation histogram is still available per-box in `<stem>_box_stats.csv` (columns `best_deviation`, `af_gain_db`); `autofocus_summary.csv` only lists boxes that actually got a PNG / NPZ written. In the new flow this branch only fires for boxes whose ``|dev|`` fell in the narrow ``[af_early_exit_threshold_m / range_spacing, af_min_abs_deviation)`` samples-domain window — the vast majority of below-gate boxes short-circuit up in ``_af_apply_global_range_deviation_correction`` and are counted at the ``corrected_k is None`` guard above.
                if abs(best_dev_k) < af_min_abs_deviation:
                    n_af_filtered += 1
                    continue

                # Minimal-mode fast path: skip the disk writes (fig_af PNG, autofocus NPZ, `autofocus_summary.csv` row) but keep the AF bookkeeping so the downstream `af_before` / `af_after` keeper PNGs still paint the correct rectangles and paste the correct chips into `s`.
                if not _save_all:
                    y_center_padded = 0.5 * (y0_full + y1_full)
                    x_center_padded = 0.5 * (x0_full + x1_full)
                    h_padded = float(y1_full - y0_full)
                    w_padded = float(x1_full - x0_full)
                    af_kept_boxes_full.append(
                        (y_center_padded, x_center_padded,
                         h_padded, w_padded)
                    )
                    af_kept_chip_data.append(
                        (y0_full, y1_full, x0_full, x1_full,
                         np.ascontiguousarray(corrected_k)),
                    )
                    af_kept_slope_totals.append(float(slope_totals[k]))
                    af_kept_best_dev_totals.append(best_dev_k)
                    # Drop the parallel reference in `corrected_chip_af` so only `af_kept_chip_data` retains the chip; without this every kept chip lives in both lists until the very end of main().
                    corrected_chip_af[k] = None
                    corrected_k = None
                    n_af_written += 1
                    continue

                gain_db_str = (
                    f"{gain_db_k:+.2f} dB"
                    if np.isfinite(gain_db_k) else "nan dB"
                )
                if best_look_rows_k > 0:
                    pga_tag = (
                        f"same-BW @ look_rows={best_look_rows_k}, "
                        f"range-walk + PGA"
                    )
                else:
                    pga_tag = "full-BW, PGA not triggered"

                star_af = "*" if bool(strong_mask[k]) else " "
                fig_af, axes_af = plt.subplots(
                    1, 2, figsize=(11, 5), constrained_layout=True,
                )
                # Panels show the raw input chip and the post-correction chip directly. The contrast gain in the title is still measured at *matched* Doppler bandwidth inside `_af_apply_global_range_deviation_correction` — i.e. the dB number compares same-BW images even though the panels display the raw full-BW chip on the left and the sub-band PGA output on the right.
                fig_af.suptitle(
                    f"{star_af}box {k:03d}  y={y_int} x={int(x_c)} "
                    f"h={h_int} w={int(w)}\n"
                    f"contrast gain: {gain_db_str}   "
                    f"({c_before_k:.3f} → {c_after_k:.3f};  "
                    f"{pga_tag})\n"
                    f"best_deviation={best_dev_k:+.3f}   "
                    f"n_sub_contrast_improved={n_sub_imp_k}/10",
                    fontsize=11,
                )
                for ax_af, img_af, ttl_af in (
                    (axes_af[0], chip_full,   "|I| before"),
                    (axes_af[1], corrected_k, "|I| after"),
                ):
                    amp_af = np.abs(img_af)
                    if amp_af.size == 0:
                        ax_af.text(
                            0.5, 0.5, "(empty)",
                            ha="center", va="center",
                        )
                        ax_af.set_title(ttl_af)
                        continue
                    vmin_af = float(np.nanpercentile(amp_af, 1.0))
                    vmax_af = float(np.nanpercentile(amp_af, 99.0))
                    if vmax_af <= vmin_af:
                        vmax_af = vmin_af + 1.0
                    ax_af.imshow(
                        amp_af, aspect="auto", cmap="gray",
                        vmin=vmin_af, vmax=vmax_af,
                    )
                    ax_af.set_xlabel(
                        "range column (full-res, in box)"
                    )
                    ax_af.set_ylabel("azimuth")
                    ax_af.set_title(ttl_af)
                out_path_af = (
                    per_box_dir / f"box_{k:03d}_autofocus.png"
                )
                fig_af.savefig(out_path_af, dpi=110)
                _maybe_close(fig_af)

                # Range-walk-derived azimuth velocity — Doppler-domain residual-RCM inversion (see `_va_from_range_walk_mps`). Uses the scene-scoped geometry resolved once at the top of the AF stage (``af_processing_prf_hz`` / ``af_wavelength_m`` / ``af_slant_range_m`` / ``af_v_sat_mps``) so the value written here matches the AF-loop stdout log line exactly. Computed before the NPZ save so both the NPZ archive and the ``autofocus_summary.csv`` row see the same value.
                v_a_walk_k = _va_from_range_walk_mps(
                    best_dev_k,
                    range_spacing_m=range_spacing,
                    processing_prf_hz=af_processing_prf_hz,
                    wavelength_m=af_wavelength_m,
                    slant_range_m=af_slant_range_m,
                    v_sat=af_v_sat_mps,
                )

                # Persist the per-box autofocus result as an NPZ so the complex `chip_before` / `chip_after` arrays (and the scalar diagnostics drawn on the PNG title) can be reloaded later for re-analysis without re-running the coarse-to-fine search.
                out_path_af_npz = (
                    per_box_dir / f"box_{k:03d}_autofocus.npz"
                )
                np.savez_compressed(
                    out_path_af_npz,
                    chip_before=chip_full.astype(
                        np.complex64, copy=False,
                    ),
                    chip_after=corrected_k.astype(
                        np.complex64, copy=False,
                    ),
                    best_deviation=np.float64(best_dev_k),
                    n_sub_contrast_improved=np.int32(n_sub_imp_k),
                    best_look_rows=np.int32(best_look_rows_k),
                    contrast_before=np.float64(c_before_k),
                    contrast_after=np.float64(c_after_k),
                    gain_db=np.float64(gain_db_k),
                    box_idx=np.int32(k),
                    y_center=np.int32(y_int),
                    x_center=np.int32(int(x_c)),
                    h=np.int32(h_int),
                    w=np.int32(int(w)),
                    y0_full=np.int32(y0_full),
                    y1_full=np.int32(y1_full),
                    x0_full=np.int32(x0_full),
                    x1_full=np.int32(x1_full),
                    Number_of_Range_Looks=np.int32(
                        Number_of_Range_Looks
                    ),
                    af_dev_min_meters=np.float64(af_dev_min_meters),
                    af_dev_max_meters=np.float64(af_dev_max_meters),
                    af_accuracy=np.float64(af_accuracy),
                    af_poly_degree=np.int32(af_poly_degree),
                    strong=np.bool_(bool(strong_mask[k])),
                    v_a_from_walk_mps=np.float64(v_a_walk_k),
                    v_a_walk_v_sat_mps=np.float64(af_v_sat_mps),
                    v_a_walk_R0_m=np.float64(af_slant_range_m),
                    v_a_walk_processing_prf_hz=np.float64(
                        af_processing_prf_hz
                    ),
                    v_a_walk_wavelength_m=np.float64(af_wavelength_m),
                )

                c_before_csv = (
                    f"{c_before_k:.6f}"
                    if np.isfinite(c_before_k) else "nan"
                )
                c_after_csv = (
                    f"{c_after_k:.6f}"
                    if np.isfinite(c_after_k) else "nan"
                )
                gain_db_csv = (
                    f"{gain_db_k:.6f}"
                    if np.isfinite(gain_db_k) else "nan"
                )
                v_a_walk_csv = (
                    f"{v_a_walk_k:.6f}"
                    if np.isfinite(v_a_walk_k) else "nan"
                )
                af_csv_lines.append(
                    f"{k},{y_int},{int(x_c)},{h_int},{int(w)},"
                    f"{chip_full.shape[0]},{chip_full.shape[1]},"
                    f"{best_dev_k:.6f},{n_sub_imp_k},"
                    f"{best_look_rows_k},"
                    f"{c_before_csv},{c_after_csv},{gain_db_csv},"
                    f"{v_a_walk_csv}"
                )
                y_center_padded = 0.5 * (y0_full + y1_full)
                x_center_padded = 0.5 * (x0_full + x1_full)
                h_padded = float(y1_full - y0_full)
                w_padded = float(x1_full - x0_full)
                af_kept_boxes_full.append(
                    (y_center_padded, x_center_padded,
                     h_padded, w_padded)
                )
                af_kept_chip_data.append(
                    (y0_full, y1_full, x0_full, x1_full,
                     np.ascontiguousarray(corrected_k)),
                )
                af_kept_slope_totals.append(float(slope_totals[k]))
                af_kept_best_dev_totals.append(best_dev_k)
                # Drop the parallel reference in `corrected_chip_af` so only `af_kept_chip_data` retains the chip; without this every kept chip lives in both lists until the very end of main().
                corrected_chip_af[k] = None
                corrected_k = None
                n_af_written += 1
            if _save_all:
                (per_box_dir / "autofocus_summary.csv").write_text(
                    "\n".join(af_csv_lines) + "\n"
                )
                print(
                    f"Saved {n_af_written} per-box autofocus PNGs + NPZs → "
                    f"{per_box_dir}/box_000_autofocus.{{png,npz}} … "
                    f"box_{K-1:03d}_autofocus.{{png,npz}}  "
                    f"(+ autofocus_summary.csv; "
                    f"{n_af_filtered} boxes dropped with "
                    f"|best_deviation| < {af_min_abs_deviation:g})"
                )
            else:
                print(
                    f"[minimal] AF ran on {n_af_written}/{K} boxes "
                    f"({n_af_filtered} filtered by |dev| < "
                    f"{af_min_abs_deviation:g}); per-box PNG/NPZ writes skipped."
                )

            # Building + saving the four full-image overview PNGs below ({stem}_af_boxes.png, {stem}_af_before.png, {stem}_af_corrected.png, {stem}_af_after.png) is excluded from the timing report — they are large-canvas plot/save operations dominated by matplotlib + PNG encoding cost, not pipeline work we care about benchmarking.
            sw.pause()

            af_boxes_arr = (
                np.asarray(af_kept_boxes_full, dtype=np.float64)
                if af_kept_boxes_full else np.empty((0, 4), dtype=np.float64)
            )
            af_kept_slope_arr = (
                np.asarray(af_kept_slope_totals, dtype=np.float64)
                if af_kept_slope_totals else np.empty((0,), dtype=np.float64)
            )
            af_kept_best_dev_arr = (
                np.asarray(af_kept_best_dev_totals, dtype=np.float64)
                if af_kept_best_dev_totals
                else np.empty((0,), dtype=np.float64)
            )

            # Rule "never copy s": reduce the full-scene complex SLC to real amplitude, freeing the 14.55 GB complex buffer as soon as the 7.27 GB float32 amplitude is ready. The explicit two-step (`np.empty` → `np.abs(s, out=…)` → rebind → `del`) is functionally identical to a naive `s = np.abs(s)` but wraps the alloc in a pair of `_release_glibc_arenas()` calls: the pre-abs trim returns the tens of GB of arena holes left behind by `degrade_range_resolution_range_sum` / `compute_subapertures` (peak 74 GB, current 17 GB just before this line) so the 7.27 GB `np.empty` lands in unfragmented address space rather than fighting with them; the post-abs trim hands the 14.55 GB complex buffer back to the OS the moment it's dropped, keeping the RSS floor at ~7.3 GB for the subsequent matplotlib savefigs and paste loop. Without the pre-abs trim, `np.empty(7.27 GB)` on a heap already 74 GB high and 17 GB live has been observed to SIGKILL the process — the OOM-killer counts committed pages, not live objects, and glibc never voluntarily returns the arena to the kernel. The RSS log lines flanking the alloc make the trim's effect visible in the log.
            _release_glibc_arenas()
            _rss_cur_pre_abs, _rss_peak_pre_abs = _rss_gb()
            print(
                f"[af] pre |s|: current RSS = {_rss_cur_pre_abs:.2f} GB  "
                f"(peak = {_rss_peak_pre_abs:.2f} GB)"
            )
            _s_amp = np.empty(s.shape, dtype=np.float32)
            np.abs(s, out=_s_amp)
            s = _s_amp
            del _s_amp
            _release_glibc_arenas()
            _rss_cur_post_abs, _rss_peak_post_abs = _rss_gb()
            print(
                f"[af] post |s|: current RSS = {_rss_cur_post_abs:.2f} GB  "
                f"(peak = {_rss_peak_post_abs:.2f} GB)"
            )

            # --- Full image with boundary boxes (before AF paste) --- `{stem}_af_boxes.png`: `|s|` (grayscale) with the bounding rectangles of every box that survived the `|best_deviation| ≥ af_min_abs_deviation` autofocus gate, coloured red for positive `best_deviation` and yellow for negative. Diagnostic — gated behind `_save_all`.
            if _save_all:
                path_af_boxes = save_dir / f"{stem}_af_boxes.png"
                fig_af_boxes, ax_af_boxes = plt.subplots(
                    figsize=(11, 9), constrained_layout=True,
                )
                _show_slc(
                    ax_af_boxes, s,
                    "Input SAR image and MTI",
                    cmap="gray", sigma=3.0,
                )
                if len(af_boxes_arr):
                    _overlay_boxes(
                        ax_af_boxes, af_boxes_arr,
                        color="red", lw=2.8,
                        signs=af_kept_best_dev_arr,
                    )
                fig_af_boxes.savefig(path_af_boxes, dpi=150)
                _maybe_close(fig_af_boxes)
                print(f"Saved → {path_af_boxes}")

            # --- High-DPI "before" panel --- Captures vmin/vmax so the "after" panel reuses the same amplitude clip and the visible jump inside each red/yellow rectangle reflects real focusing gain, not per-figure rescaling. Keeper: always saved (one of the three files retained in minimal mode).
            path_af_before = save_dir / f"{stem}_af_before.png"
            fig_af_before, ax_af_before = plt.subplots(
                figsize=(16, 14), constrained_layout=True,
            )
            vmin_shared, vmax_shared = _show_slc(
                ax_af_before, s,
                "Input SAR image and MTI (unfocused)",
                cmap="gray", sigma=3.0,
            )
            if len(af_boxes_arr):
                _overlay_boxes(
                    ax_af_before, af_boxes_arr,
                    color="red", lw=0.7,
                    signs=af_kept_best_dev_arr,
                )
            fig_af_before.savefig(path_af_before, dpi=350)
            _maybe_close(fig_af_before)
            print(f"Saved → {path_af_before}")

            # Paste `|chip_corr|` into `s` in place — rule "never copy s". After this loop, `s` represents the AF-corrected amplitude scene; the "before" figures were already written above and closed, so their AxesImages no longer reference `s`. We rebind each entry to `None` in-place right after it has been consumed so the retained chip's last reference is released the moment its amplitude has landed in `s`; keeping the tuples alive would hold every AF chip in RAM until `af_after` renders. `chip_corr` is float32 amplitude in minimal mode (collapsed in the AF pre-compute loop above) and complex64 in debug mode; the `dtype` branch keeps the paste allocation-free for the minimal-mode case (no `np.abs` on real input, no cast to the matching float32 dtype).
            for _i_paste in range(len(af_kept_chip_data)):
                y0c, y1c, x0c, x1c, chip_corr = af_kept_chip_data[_i_paste]
                target = s[y0c:y1c, x0c:x1c]
                if chip_corr.shape == target.shape:
                    if np.iscomplexobj(chip_corr):
                        s[y0c:y1c, x0c:x1c] = np.abs(chip_corr).astype(
                            s.dtype, copy=False,
                        )
                    else:
                        s[y0c:y1c, x0c:x1c] = chip_corr
                af_kept_chip_data[_i_paste] = None
                del chip_corr

            # Every kept chip has been amplitude-pasted into `s` above; the entries we cleared in the pre-compute loop are already gone but the outer list still costs a slot per box, so drop it entirely. `corrected_chip_af` is now a list of `None`s (Fixes 1 + 2 cleared every real chip), but release the container too so nothing lingers into the refocus / show blocks below.
            del af_kept_chip_data
            del corrected_chip_af

            # --- Full image with corrected chips pasted into the boxes --- Companion to `{stem}_af_boxes.png`; renders the mutated `s` with the surviving boxes outlined red (positive `best_deviation`) / yellow (negative) so the patched regions are easy to locate. Uses its own auto-scaled vmin/vmax (from the post-paste amplitude stats). Diagnostic — gated behind `_save_all`.
            if _save_all:
                path_af_corrected = save_dir / f"{stem}_af_corrected.png"
                fig_af_corr, ax_af_corr = plt.subplots(
                    figsize=(11, 9), constrained_layout=True,
                )
                _show_slc(
                    ax_af_corr, s,
                    "Input SAR image and Focused Moving Targets",
                    cmap="gray", sigma=3.0,
                )
                if len(af_boxes_arr):
                    _overlay_boxes(
                        ax_af_corr, af_boxes_arr,
                        color="red", lw=2.8,
                        signs=af_kept_best_dev_arr,
                    )
                fig_af_corr.savefig(path_af_corrected, dpi=150)
                _maybe_close(fig_af_corr)
                print(f"Saved → {path_af_corrected}")

            # --- High-DPI "after" panel --- Reuses `vmin_shared` / `vmax_shared` captured on the "before" panel. Keeper: always saved (one of the three files retained in minimal mode).
            path_af_after = save_dir / f"{stem}_af_after.png"
            fig_af_after, ax_af_after = plt.subplots(
                figsize=(16, 14), constrained_layout=True,
            )
            _show_slc(
                ax_af_after, s,
                "Input SAR image with focused moving targets",
                vmin=vmin_shared, vmax=vmax_shared,
                cmap="gray",
            )
            if len(af_boxes_arr):
                _overlay_boxes(
                    ax_af_after, af_boxes_arr,
                    color="red", lw=0.7,
                    signs=af_kept_best_dev_arr,
                )
            fig_af_after.savefig(path_af_after, dpi=350)
            _maybe_close(fig_af_after)
            print(f"Saved → {path_af_after}")

            sw.resume()

            # --- Per-box rough refocus (classification signal) --- For each surviving box write a `box_NNN_refocus.png` (|chip| before / |chip| after the centred-QPE correction) alongside the existing `box_NNN_dphase.png` in per_box_dir, plus a `refocus_summary.csv` with per-box contrast & gain. Diagnostic only — skipped entirely in minimal mode.
            if _save_all and cfg.refocus:
                csv_lines = [
                    "box_idx,y,x,h,w,y0_fit,n_rows_fit,"
                    "slope_rad_per_row,slope_total_rad,"
                    "contrast_before,contrast_after,gain_db"
                ]
                gains_db = []
                n_written = 0
                H_full, W_full = s_degraded.shape
                for k, (y_c, x_c, h, w) in enumerate(boxes_yxhw):
                    n_k = int(box_n_rows[k])
                    y0 = int(box_y0[k])
                    y1 = y0 + n_k
                    x_int = int(x_c)
                    w_int = int(w)
                    x0 = max(0, x_int - w_int // 2)
                    x1 = min(W_full, x_int - w_int // 2 + w_int)
                    s_k = float(box_slope_rad_per_row[k])
                    if (
                        n_k < 2 or x1 <= x0 or y1 <= y0
                        or y0 < 0 or y1 > H_full
                        or not np.isfinite(s_k)
                    ):
                        continue
                    chip = s_degraded[y0:y1, x0:x1]
                    if chip.size == 0:
                        continue
                    corrected, _phi = _refocus_box_chip(chip, s_k)
                    c_before = _normalized_variance(chip)
                    c_after = _normalized_variance(corrected)
                    if c_before > 0.0 and c_after > 0.0:
                        db = 20.0 * np.log10(c_after / c_before)
                    else:
                        db = np.nan
                    gains_db.append(db)
                    db_str = f"{db:.6f}" if np.isfinite(db) else "nan"
                    csv_lines.append(
                        f"{k},{int(y_c)},{int(x_c)},{int(h)},{w_int},"
                        f"{y0},{n_k},"
                        f"{s_k:.6f},{float(slope_totals[k]):.6f},"
                        f"{c_before:.6f},{c_after:.6f},{db_str}"
                    )
                    _plot_refocus_box_1x2(
                        chip, corrected,
                        box_idx=k,
                        box_yxhw=(int(y_c), int(x_c), int(h), w_int),
                        slope_rad_per_row=s_k,
                        slope_total_rad=float(slope_totals[k]),
                        contrast_before=c_before,
                        contrast_after=c_after,
                        gain_db=db,
                        out_path=per_box_dir / f"box_{k:03d}_refocus.png",
                    )
                    n_written += 1
                (per_box_dir / "refocus_summary.csv").write_text(
                    "\n".join(csv_lines) + "\n"
                )
                gains_arr = np.asarray(gains_db, dtype=np.float64)
                ok = np.isfinite(gains_arr)
                if ok.any():
                    med_db = float(np.nanmedian(gains_arr[ok]))
                    max_db = float(np.nanmax(gains_arr[ok]))
                    print(
                        f"Saved {n_written} per-box refocus PNGs → "
                        f"{per_box_dir}/box_NNN_refocus.png "
                        f"(contrast gain: median {med_db:+.2f} dB, "
                        f"max {max_db:+.2f} dB)"
                    )
                else:
                    print(
                        f"Saved {n_written} per-box refocus PNGs → "
                        f"{per_box_dir}/box_NNN_refocus.png "
                        f"(no valid gains)"
                    )
    sw.mark("save outputs + per-box PNGs (main fig, d_phase, FFT, autofocus, refocus)")

    # Interactive display: if `--show` was set at import time we switched off `Agg` and skipped every intermediate `plt.close(fig)` via `_maybe_close`, so all figures are still live in memory. Pop them up now via a blocking `plt.show()`; this returns once the user closes every window. Timing this outside the pipeline so it is not accounted for in the stopwatch summary.
    if _SHOW_FIGURES:
        # `--show-mask-boxes-only` used to isolate the retired `fig_mb` window; the mask overlay is no longer built (see the "mask_full + grown boxes" retirement note upstream), so the flag now falls through and shows every live figure.
        if getattr(cfg, "show_mask_boxes_only", False):
            print(
                "  [show] --show-mask-boxes-only requested but fig_mb "
                "was retired (no scene-wide mask allocation); showing "
                "all live figures instead."
            )
        n_figs = len(plt.get_fignums())
        print(
            f"  [show] popping up {n_figs} figures; close the windows "
            "to exit."
        )
        sw.pause()
        plt.show()
        sw.resume()

    # Always close every figure we built so the interpreter can free the matplotlib state cleanly. Figures intended for review have already been written to disk by the cfg.save branch above; when `--show` is unset the Agg backend has kept the process headless throughout.
    plt.close("all")

    sw.report()


if __name__ == "__main__":
    main()
