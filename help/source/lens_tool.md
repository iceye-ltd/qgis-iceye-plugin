# Lens Tool

## What it does
- Shows a movable lens overlay that renders a zoomed view of the active layer.
- Draws a dashed rectangle on the map showing the lens footprint.
- Supports multiple render modes for SLC inspection and spectral views.

## How to use
1. Open the `ICEYE Lens` toolbar and click the lens icon to enable the tool.
2. Move the cursor on the map to position the lens.
3. Scroll the mouse wheel to zoom the lens view.
4. Click to pin the lens in place; click again to unpin.
5. Use the toolbar buttons to switch render modes.

## Render modes
- **Normal**: renders the map image inside the lens.
- **Focus**: shows a focused SLC view (requires 2-band SLC layer).
- **2D Spectrum**: shows the 2D spectrum magnitude (requires 2-band SLC layer).
- **Color**: shows an RGB aperture-weighted spectrum (requires 2-band SLC layer).

## Notes
- The lens operates on the current active layer; non-raster layers are ignored.
- SLC-based modes require a valid 2-band SLC raster layer and metadata.

[demo](lens_tool_demo.webm)