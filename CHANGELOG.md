# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2](https://github.com/iceye-ltd/qgis-iceye-plugin/compare/v0.1.1...v0.1.2) (2026-05-26)


### ### Fixed

* address security scanner blockers and clean up code quality ([#11](https://github.com/iceye-ltd/qgis-iceye-plugin/issues/11)) ([96658a6](https://github.com/iceye-ltd/qgis-iceye-plugin/commit/96658a6e7dbca3d5fa52a75099b39f1e00c06abb))

## [0.1.1](https://github.com/iceye-ltd/qgis-iceye-plugin/compare/v0.1.0...v0.1.1) (2026-05-11)


### ### Fixed

* handle QCloseEvent when closing metadata dockwidget ([#10](https://github.com/iceye-ltd/qgis-iceye-plugin/issues/10)) ([d244c9c](https://github.com/iceye-ltd/qgis-iceye-plugin/commit/d244c9c660451e8b0d1d2a73322f4d6da20ebdfe))
* use gitattritubes to exclude fixtures and tests ([#9](https://github.com/iceye-ltd/qgis-iceye-plugin/issues/9)) ([bae471a](https://github.com/iceye-ltd/qgis-iceye-plugin/commit/bae471a99c492201b3ca2688fc36040574eb049b))


### ### Changed

* use iceye_tooblox name consistently ([#5](https://github.com/iceye-ltd/qgis-iceye-plugin/issues/5)) ([df55201](https://github.com/iceye-ltd/qgis-iceye-plugin/commit/df55201dadf91031d6093c20df1f0db9c9a510c6))

## [Unreleased]

## [0.1.0] - 2025-02-25

### Added

- **STAC Browser** — Browse the ICEYE STAC catalog on AWS Open Data, search collections, and load items as raster layers
- **Canvas Screenshot** — Export the current map canvas view as PNG or JPEG
- **Layer Export** — Export raster layers to TIFF, PNG, GIF (ImageMagick), and MP4 (ffmpeg)
- **Lens Tool** — Interactive magnifying lens with Normal, Focus (1D PGA), 2D Spectrum, and Color (RGB aperture-weighted) modes
- **Crop Tool** — Draw extent to crop, focus (PGA), create video, or generate RGB spectrum visualization
- **Canvas Rotation & SAR Mandala** — Rotate shadows down, layover up, reset to north; toggle compass overlay (N/S/T/L)
- **Temporal Properties** — Auto-configure temporal ranges for ICEYE layers; multi-frame support for QGIS Temporal Controller
- **Auto Style & Colormaps** — Default grayscale styling for SLC layers; Grey Log, Square-Root, Asinh, and LogLog colormaps for SAR

[0.1.0]: https://github.com/iceye-ltd/qgis-iceye-plugin/releases/tag/v0.1.0
