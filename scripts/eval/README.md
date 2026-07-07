# Ground-truth database + evaluator for `shear_averaging.py`

Two scripts, one workflow:

1. **`label_ground_truth.py`** – interactive per-image labeller. Given a
   `.tif`/`.tiff` (ICEYE SLC with sidecar `.json`) or `.npy` patch,
   pops up a decimated `|s|` view; you drag rectangles around moving
   targets; boxes are saved to a sidecar `<image_stem>.gt.json`.

2. **`eval_ground_truth.py`** – runs `scripts/shear_averaging.py` on
   the same image, reads the resulting `_box_stats.csv`, matches
   detections against the ground-truth sidecar, and writes per-image +
   aggregate TP / FA / FN statistics plus an overlay figure.

## Coordinate convention (READ THIS FIRST)

Everything is stored in the same frame that
`scripts/shear_averaging.py::_load_slc_from_tiff` produces:

* Axis 0 = azimuth, top row = azimuth 0.
* Axis 1 = range, at **FULL** range resolution (i.e. every original
  range pixel of the TIFF, AFTER any left-look `fliplr`).
* Shape: `(N_az, N_range)`.

The detector reports boxes in `s_degraded` coordinates, where the range
axis is decimated by
`Number_of_Range_Looks = int(1.5 / sar_pixel_spacing_range)`
(1.5 m = `min_size_of_target` hard-coded in `shear_averaging.py`).
The evaluator multiplies `x_c` and `w` by `Number_of_Range_Looks`
before matching, so labels + detections live in the same frame.

## `<image_stem>.gt.json` schema

```json
{
  "schema": 1,
  "image_path": "/abs/path/to/scene.tif",
  "image_stem": "scene",
  "sha256": "…",
  "shape_az_rg": [N_az, N_range_full],
  "left_look": true,
  "range_pixel_spacing_m": 0.5,
  "azimuth_pixel_spacing_m": 0.6,
  "created_at": "2026-…",
  "updated_at": "2026-…",
  "coord_frame": "shear_averaging (azimuth, range) full-resolution…",
  "boxes": [
    {
      "id": 0,
      "y_c": 4123.0,
      "x_c": 8721.0,
      "h": 180,
      "w": 260,
      "class": "1",
      "notes": "",
      "created_at": "2026-…"
    }
  ]
}
```

`y_c, x_c, h, w` are floats/ints in the full-resolution
`(azimuth, range)` frame described above. `class` is whatever
single-digit key you pressed before the drag (or `null`).

## Labelling workflow

```bash
python scripts/eval/label_ground_truth.py path/to/scene.tif
# optional overrides
python scripts/eval/label_ground_truth.py scene.tif \
    --decimate-az 8 --decimate-rg 2 \
    --gt-json custom_labels.gt.json
```

Inside the window:

| Key           | Action                                      |
|---------------|---------------------------------------------|
| drag L-click  | add a rectangle                             |
| `z`           | undo the last box                           |
| `x`           | delete the box under the cursor             |
| `s`           | save to disk                                |
| `q`           | save AND quit                               |
| `Q`           | quit WITHOUT saving                         |
| `r`           | reload the sidecar from disk                |
| `l`           | toggle id/class text overlays               |
| `1`..`9`      | set the class for the NEXT drag             |
| `c`           | clear the pending class                     |

The sidecar is re-loaded automatically on startup if it already exists,
so labelling is incremental across sessions.

## Evaluation workflow

Single image:

```bash
python scripts/eval/eval_ground_truth.py path/to/scene.tif
# forward extra flags to shear_averaging after '--':
python scripts/eval/eval_ground_truth.py path/to/scene.tif -- \
    --slope-rad-thresh 1.5 --min-velocity-mps 3.0
```

Whole folder:

```bash
python scripts/eval/eval_ground_truth.py /data/scenes/ --out-dir ./eval_out
```

Re-score without re-running `shear_averaging` (fast; useful when you
only tweak the matching or the overlay):

```bash
python scripts/eval/eval_ground_truth.py path/to/scene.tif --skip-existing-csv
```

## Matching rule (centre-in-GT, 1-to-1)

For every detection with centre `(y_c, x_c)`:

* If it falls inside **exactly one** GT rectangle → candidate `TP` for
  that GT.
* If it falls inside **several** GT rectangles → assigned to the GT
  whose CENTRE is closest.
* If it falls inside **no** GT → `FA`.

If multiple detections match the same GT, only the nearest-centre one
becomes `TP`; the rest are `TP_dup` (they do not lower precision).
GT boxes with zero matches are `FN`.

Precision = `TP / (TP + FA)`,
recall = `TP / (TP + FN)`,
F1 = harmonic mean.

## Evaluator output layout

```
eval_out/
├── eval_summary.csv            # appended per image, per invocation
├── scene1/
│   ├── shear.png               # main shear_averaging figure
│   ├── shear_box_stats.csv     # raw detector output
│   ├── scene1_eval.csv         # per-detection status
│   ├── scene1_eval_gt.csv      # per-GT match info
│   ├── scene1_eval.json        # per-image summary
│   └── scene1_eval_overlay.png # decimated |s| with GT/TP/FA/FN
├── scene2/…
```

Overlay colour code:

| Colour        | Meaning        |
|---------------|----------------|
| lime, solid   | GT             |
| gold, solid   | TP             |
| red, solid    | FA             |
| blue, dashed  | FN             |

## Notes

* `label_ground_truth.py` imports `_load_slc_from_tiff` and
  `_load_iceye_sidecar_metadata` from `scripts/shear_averaging.py` so
  the labelling frame is always identical to the detection frame.
* Both scripts assume `matplotlib` is installed. The labeller needs an
  interactive backend; use `--backend TkAgg` or `Qt5Agg` if the default
  doesn't open a window.
* `Number_of_Range_Looks` is recomputed from
  `sar_pixel_spacing_range` (stored in the GT sidecar) — no need to
  pass it on the CLI. It must therefore be present in the sidecar,
  which is automatic for `.tif` inputs. For a bare `.npy` input the
  spacing is unknown and `Number_of_Range_Looks = 1`, i.e. the
  detector CSV is assumed already at full range resolution (which is
  only true if you ran `shear_averaging.py` without any range
  degradation).
