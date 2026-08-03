# Shear averaging — SAR moving target detection

Standalone baseline of the shear-averaging / phase-derivative MTI pipeline.
Detects moving targets in ICEYE SLC scenes, estimates per-target phase slopes,
and optionally applies polynomial range-walk + PGA autofocus per box.

## Requirements

- Python 3.10+
- `numpy`, `scipy`, `matplotlib` (see `requirements.txt`)
- GDAL Python bindings (`osgeo`) only if you load `.tif` / `.tiff` SLC inputs

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Files

| File | Role |
|------|------|
| `shear_averaging.py` | End-to-end pipeline |
| `clustering.py` | Peak clustering helper (`cluster_peaks`) |
| `shear_averaging_config.json` | Tunable parameters (JSON sidecar) |

## Run

Example: python shear_averaging.py ~/shear_no_debug.json /home/odogan/Desktop/ship_focusing/4439676/ICEYE_WTW3YQ_20250104T180444Z_4439676_X7_SLED_SLC.tif
Discussion is in here:
https://docs.google.com/presentation/d/199rUxs6vvOqdETQqlZJmnVElgn3BV8fi0BOcEet1y7c/edit?slide=id.g39fd0c834c0_0_9766#slide=id.g39fd0c834c0_0_9766


```bash
# GeoTIFF + ICEYE sidecar JSON next to it (<stem>.json)
python shear_averaging.py /path/to/scene.tif --save auto

# .npy patch (optionally with metadata)
python shear_averaging.py /path/to/patch.npy --save auto

# .npz bundle (complex `data` + ICEYE metadata keys)
python shear_averaging.py /path/to/patch.npz --save auto

# Custom config / disable PNG saving
python shear_averaging.py my_config.json /path/to/scene.tif --no-save
```

With default config (`output.debug=false`), the run writes the two AF keeper
figures next to the scene (or under `output.save_dir` when not using a CLI path):

- `{stem}_af_before.png`
- `{stem}_af_after.png`

Set `"debug": {"value": true}` in the config for the full diagnostic suite.

## Input formats

1. **ICEYE SLC GeoTIFF** (`.tif`/`.tiff`) — requires sidecar `<stem>.json`
2. **`.npz` bundle** — complex `data` plus ICEYE metadata fields
3. **`.npy` patch** — optional `metadata_from` pointing at a `.tif` or `.json`

## Notes

- Full-scene SLC arrays are large (`complex64`, often 10+ GB). Do not copy the
  scene buffer; the script mutates in place where needed.
- PRF / pixel spacings are read from ICEYE metadata when available; override
  via `input.prf` in the JSON config if needed.
