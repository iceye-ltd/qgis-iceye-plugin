# ICEYE QGIS plugins

The ICEYE Toolbox plugin regroup a collection of tools for explotation of ICEYE's Single Look Complex (SLC) Data as well as other image products.

## Features

#### STAC Browser

Toggle to show or hide the STAC catalog dock widget. Browse the ICEYE STAC catalog, search collections, and load items as raster layers in the map. The catalog points to the [ICEYE SAR Open Dataset on the AWS Registry of Open Data](https://registry.opendata.aws/iceye-opendata/), which provides publicly available SAR imagery for research and education.

> [!TIP]
> Click the catalog icon in the ICEYE Toolbox toolbar to show or hide the dock. In the catalog, browse collections, use the search bar to filter, and double-click an item (or use the load action) to add it as a raster layer to the map.

[catalog-demo.webm](https://github.com/user-attachments/assets/63b12d10-1baa-4202-a796-58a516835cbb)

#### Canvas Screenshot

Exports the current map canvas view as PNG or JPEG. Opens a save dialog to choose the output path and format.

> [!TIP]
> Click the screenshot icon. In the save dialog, choose the output path and format (PNG or JPEG), then save. The current map view (including all visible layers) is exported as-is.

#### Layer Export

Exports the active raster layer to various formats: **TIFF** (direct copy), **PNG** (rendered image), **GIF** (multi-band animation, requires ImageMagick), **MP4** (multi-band animation, requires ffmpeg).

> [!TIP]
> Select the raster layer to export in the Layers panel, then click the export icon. Choose the output path and format in the dialog. For PNG, you can set a downscale factor. For GIF/MP4, ensure ImageMagick (GIF) or ffmpeg (MP4) is installed.

#### Lens Tool

Interactive magnifying lens over the map that follows the cursor or can be pinned with a click. Use the mouse wheel to zoom. Modes:

- **Normal**: Standard map view of the active layer
- **Focus**: 1D PGA focusing
- **2D Spectrum**: 2D FFT magnitude visualization
- **Range Power Spectrum**: Range-axis FFT power visualization (selectable from the 2D Spectrum dropdown)
- **Color**: RGB aperture-weighted spectrum visualization
- **Sub-aperture Viewer**: Filters to a sub-aperture window in either the azimuth or range and renders the result. Scroll the mouse wheel or use Shift+↑ / Shift+↓ to slide the window through the apertures. Starts at the center aperture. Select azimuth or range from the dropdown.

> [!WARNING]
> The active layer must be selected in the Layers panel before activating the lens. Focus, 2D Spectrum, Range Power Spectrum, Color and Sub-aperture viewer modes read directly from the selected SLC layer — if no layer is selected or the wrong layer is active, the lens will show no output.

> [!TIP]
> Select a raster layer, then click the lens icon to activate. Move the cursor over the map — the lens follows. Click to pin the lens in place; click again to unpin. Use the mouse wheel to zoom in or out. Switch modes with the toolbar buttons (Normal, Focus, Color). The **2D Spectrum** button is a dropdown — click the arrow to choose between **2D Spectrum** and **Range Power Spectrum**; clicking the button face re-applies the last selected spectrum mode. The **Sub-aperture Viewer** button is a dropdown — click the arrow to choose between **Azimuth** and **Range**; in either mode the mouse wheel or Shift+↑ / Shift+↓ slides the look window through the band instead of zooming. Focus, 2D Spectrum, Range Power Spectrum, and Color require ICEYE SLC data with metadata.

[lens-demo.webm](https://github.com/user-attachments/assets/23365b5d-8b6d-4289-84d9-738606436f86)

#### Crop Tool

Draw an extent on the map to process the active layer. Modes:

- **Normal**: Clip/crop the raster layer to the drawn extent
- **Focus**: Target focus workflow — PGA focusing on the selected area (SAR)
- **Video**: Create a video from the cropped SLC area (frame-by-frame imaging)
- **Color**: RGB spectrum visualization (Range/Linear or Slow Time sub-modes)

> [!TIP]
> Select the raster layer, choose a mode (Normal, Focus, Video, or Color), then click the crop icon. Draw a rectangle on the map by clicking and dragging. The tool processes the extent and adds the result as a new layer (or opens the video dialog for Video mode).

[crop-mode-demo.webm](https://github.com/user-attachments/assets/14aefad5-8766-455f-8b52-f9cd0e87157c)

#### Canvas Rotation and SAR Mandala

Canvas rotation tools align the map view with SAR geometry. Requires an active ICEYE layer with metadata.

- **Rotate Shadows Down**: Rotate the canvas so that shadows point downward
- **Reset to North**: Reset canvas rotation to North up
- **Rotate Layover Up**: Rotate the canvas so that layover points upward
- **Toggle SAR Mandala**: Show or hide a compass overlay indicating North (N), Shadows (S), Track (T), and Layover (L) directions. The mandala updates automatically when the active layer changes.
- **Place / Move SAR Mandala**: Click a location on the map to drop a mandala pinned to that geographic point. The shadow and layover line lengths are computed from the local incidence angle at that specific position in the swath. The tool deactivates automatically after each placement. Drag any placed mandala at any time to reposition it and the geometry and incidence angle update automatically at the new location. Right-click a mandala to remove it individually. Use the dropdown arrow on the button to  **Clear placed mandalas**.

> [!TIP]
> Activate an ICEYE layer with metadata. Click **Rotate Shadows Down**, **Rotate Layover Up**, or **Reset to North** to align the view. Click **Toggle SAR Mandala** to show the compass overlay; click again to hide it. Click **Place / Move SAR Mandala** and then click anywhere on the map to drop a mandala — the tool returns to pan automatically after placement. Move the cursor near any placed mandala; it will glow gold to indicate it can be interacted with. **Drag** it (left-click near its centre and drag) to reposition it and geometry updates on drop. **Right-click** a mandala to remove it. Use the dropdown arrow to clear all placed mandalas at once.

[canvas-rotation-demo.webm](https://github.com/user-attachments/assets/dab72e49-55a2-4ff3-b318-55bdfc089230)

#### Measuring tools

**Height Ruler**

Places a SAR-geometry-aware height ruler on the map anchored to a geographic point. The ruler displays a shadow line and a layover line that reflect the actual SAR shadow and layover directions at the clicked location. The displayed height label (H: X.X m) is derived from the layover ground distance and the local incidence angle.

> [!WARNING]
> Requires an active ICEYE layer with metadata. If no ICEYE layer is selected, placing a ruler will have no effect.

> [!TIP]
> Select an ICEYE layer, then click the Place Height Ruler button and click a location on the map to drop a ruler. The tool deactivates automatically after placement. Once placed, move the cursor near a ruler's origin. It will glow gold to indicate it can be interacted with. Then:
> - **Placement:** Put the **shadow/layover junction** on the **base** of the feature you want the height for.
> - **Drag** the ruler (left-click near its origin and drag) to reposition it. The geometry updates at the new location.
> - **Shift + mouse wheel** over a ruler to adjust the ruler length in large steps, which updates the derived height in real time.
> - **Control + mouse wheel** over a ruler to adjust the ruler length in small steps.
> - **Right-click** a ruler to remove it individually.
> - Use the **dropdown arrow** on the button to **Clear height rulers** and it removes all at once. The ruler scales with canvas zoom and stays correctly oriented when the canvas is rotated.

**IRF analysis tool**

Impulse Response Function (IRF) analysis tool for a bright point target on SLC data. The tool reads a patch around your click, recentres on the strongest scatterer, oversamples it in the frequency domain and then shows range and azimuth amplitude profiles (dB, with a -3dB reference line) and summary metrics: **half-power beamwidth (HPBW)** in metres, **Peak sidelobe ratio (PSLR)** and **Integrated sidelobe ratio (ISLR)** in dB.

>[!WARNING]
> Requires an active ICEYE layer with metadata. Works on SLC products (amplitude + phase). Clicks that cannot be mapped to the raster or invalid reads will show a message and do nothing.

>[!TIP]
>Select the SLC layer in the Layers panel, click the IRF Analysis button on the ICEYE Measuring toolbar, then click on or near a point-like target (e.g. corner reflector). A dialog opens with the two profiles and the metrics table; the map tool turns off after one analysis. Click the button again for another point.

[measuring-tools-demo.webm](https://github.com/user-attachments/assets/6094f5be-073e-4bcd-b0e0-198798c3700f)

#### Batch processing

The **ICEYE Batch** toolbar collects several areas on the map, then runs **Crop**, **Color**, **Focus**, or **Video** on each area **one after another** (sequential background tasks).

- **Batch masks**: Toggle **Batch masks** and choose a **base ICEYE raster** in the Layers panel. Each **left-click** on the map adds a square mask extent. The size of the square mask is roughly 200m per side.
- **Run Batch Process**: The main button runs **Crop** on every collected area. Use the **dropdown** on the button to pick **Crop**, **Color**, **Focus**, or **Video** instead.
- **Batch video**: A **single dialog** asks for the number of frames; that choice applies to **all** areas in the batch.

> [!WARNING]
> If more than one raster layer is selected, batch will show an error, use Ctrl+click to build a selection with one raster and your mask layers only. Run **one** batch at a time until it finishes; starting another batch workflow (or overlapping heavy tasks) while one is still running can stress GDAL/QGIS and may destabilize the session.

> [!TIP]
> Make the base ICEYE raster the active layer (single-click it). Then multi-select your mask polygon layers: Ctrl+click each mask (and the base layer in the tree if you want it in the selection), or Shift+click to select a range of rows from one layer to another. Do not multi-select two different image (raster) layers, only one raster should be in the selection. Then run Batch Process from the menu.

[batch-processing-demo.webm](https://github.com/user-attachments/assets/4300138e-7361-44a6-9895-b608405514bc)


#### Temporal Properties

ICEYE layers are automatically configured with temporal properties when added to the map. Standard layers (SLC, CSI, QLK, crop, focus) use a fixed temporal range spanning the acquisition start and end time. Multi-frame layers (VID, short) use per-band temporal ranges so each band corresponds to a time step within the acquisition. This enables QGIS's Temporal Controller to scrub through video layers and to filter or animate layers by acquisition time.

> [!TIP]
> No manual setup required. When you add ICEYE layers, temporal properties are applied automatically. Open the Temporal Controller (View → Temporal Controller) to scrub through multi-frame layers or animate by acquisition time.

[temporal-properties-demo.webm](https://github.com/user-attachments/assets/1fdd630a-7e37-4135-bb4e-34a1ce5d2e64)

#### Auto Style and Colormaps

ICEYE SLC layers receive a default grayscale style automatically when added to the map (QLK layers use their own styling). The plugin also registers five colormaps in QGIS: four grayscale colormaps optimized for SAR's high dynamic range (**Grey Log**, **Grey Square-Root**, **Grey Asinh**, **Grey LogLog**) plus an **RGB Teal** gradient (black → teal → light mint → white). Use them via the layer's Symbology panel. Log compresses the brightest values; Square-Root is gentler; Asinh offers soft logarithmic compression; LogLog provides strong compression for very high dynamic range data.

> [!TIP]
> Styling is applied automatically when SLC layers are loaded. To change the colormap, open the layer's Properties → Symbology, then select one of the registered colormaps (Grey Log, Grey Square-Root, Grey Asinh, Grey LogLog, Teal) from the color ramp dropdown.

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style, and how to submit changes.
