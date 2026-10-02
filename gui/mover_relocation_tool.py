"""Mover Relocation: two-click relocation of moving targets in Spotlight / Dwell SLCs.

1. Click the imaged target (snapped to its hull). The
   possible-location band is drawn along the target's range line with ``|v_r|`` ticks,
   and the cursor readout shows the radial velocity a band position implies.
2. Click two points on the road / rail / bridge deck / wake axis, one on each side of
   the band. The true position, displacement and velocity are computed with a
   plausibility indicator.

Clicks are mapped onto the surface the GCP-warped display lies on (mean GCP height),
so they land on the pixels the user sees. For bridges click the deck's direct return,
not its water-level line. All maths lives in ``core.mover_relocation``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFillSymbol,
    QgsGeometry,
    QgsLineSymbol,
    QgsMarkerLineSymbolLayer,
    QgsMarkerSymbol,
    QgsMessageLog,
    QgsPalLayerSettings,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsSimpleLineSymbolLayer,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtCore import QCoreApplication, QMetaType, Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor
from qgis.PyQt.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..core.cropper import get_extend_image_coords
from ..core.metadata import MetadataProvider
from ..core.mover_relocation import (
    MODE_SINGLE_CLICK_AUTO,
    MPS_TO_KMH,
    MPS_TO_KNOTS,
    TARGET_CLASSES,
    Band,
    ConstraintError,
    ImagedTarget,
    Relocation,
    RelocationSettings,
    band_for_target,
    cursor_readout,
    estimate_constraint_axis,
    image_time_limits,
    locate_imaged_target,
    relocate,
    relocate_from_axis,
    relocate_single_click,
)
from ..core.target_finder import (
    ProductGeometry,
    RelocationParameters,
    SlcChip,
    gcp_lonlat_to_pixel,
    gcp_mean_height,
    gcp_pixel_to_lonlat,
    lonlat_to_ecef,
    patch_to_file_layout,
    read_iceye_properties,
)
from .lens_tool import read_slc_data

WGS84 = "EPSG:4326"
# Largest SLC chip read around a click (complex samples).
MAX_CHIP_PIXELS = 20_000_000
_METRES_PER_DEG = 111_320.0

# Output layers: key -> (geometry type, layer name).
LAYERS = {
    "band": ("Polygon", "Mover band"),
    "ticks": ("Point", "Mover band ticks"),
    "imaged": ("Point", "Mover imaged position"),
    "true": ("Point", "Mover true position"),
    "displacement": ("LineString", "Mover displacement"),
    "track": ("LineString", "Mover track"),
}

_INDICATOR_COLORS = {"green": "#27ae60", "amber": "#e67e22", "red": "#c0392b"}

# Canvas colours: saturated hues that read on dark sea and bright clutter alike,
# each drawn with a black HALO casing.
COLORS = {
    "band": (255, 221, 0),  # yellow
    "ticks": (255, 221, 0),
    "imaged": (255, 0, 204),  # magenta
    "displacement": (255, 0, 204),
    "true": (57, 255, 20),  # lime
    "track": (57, 255, 20),
    "constraint": (0, 229, 255),  # cyan
}
HALO = (0, 0, 0)

STEP_IDLE = "idle"
STEP_TARGET = "target"
STEP_CONSTRAINT = "constraint"
STEP_DONE = "done"


def _tr(message: str) -> str:
    """Translate message for ICEYE Toolbox context."""
    return QCoreApplication.translate("ICEYE Toolbox", message)


# ----------------------------------------------------------------------------------
# Scene: one SLC layer's geometry and display surface
# ----------------------------------------------------------------------------------


@dataclass
class MoverScene:
    """Geometry of the SLC layer the tool works on."""

    layer: QgsRasterLayer
    geometry: ProductGeometry
    display_height: float
    pixel_to_lonlat: Any
    lonlat_to_pixel: Any
    layer_id: str = ""

    @classmethod
    def from_layer(cls, layer: QgsRasterLayer) -> MoverScene:
        """Parse metadata and GCP geolocation of an ICEYE SLC layer."""
        source = layer.dataProvider().dataSourceUri()
        geometry = ProductGeometry.from_properties(read_iceye_properties(source))
        return cls(
            layer=layer,
            geometry=geometry,
            display_height=gcp_mean_height(source, geometry.scene_height),
            pixel_to_lonlat=gcp_pixel_to_lonlat(source),
            lonlat_to_pixel=gcp_lonlat_to_pixel(source),
            layer_id=layer.id(),
        )

    def ecef(self, lon: float, lat: float):
        """ECEF of a displayed point (on the display surface)."""
        return lonlat_to_ecef(lon, lat, self.display_height)

    def time_limits(self, target: ImagedTarget) -> tuple[float, float]:
        """Zero-Doppler span of the image along the target's range line."""
        _, row = self.lonlat_to_pixel(*target.lonlat)
        return image_time_limits(
            self.geometry,
            self.pixel_to_lonlat,
            self.layer.width(),
            row,
            self.display_height,
        )


def target_from_click(
    scene: MoverScene,
    lon: float,
    lat: float,
    metadata_provider: MetadataProvider,
    detect_hull: bool = True,
    params: RelocationParameters | None = None,
) -> ImagedTarget:
    """Imaged target at a click: the hull centroid nearby, or the click itself."""
    geometry, h = scene.geometry, scene.display_height
    if not detect_hull:
        return ImagedTarget.from_ecef(geometry, scene.ecef(lon, lat), h)
    params = params or RelocationParameters()
    radius = (
        params.corridor_half_width_m + params.ring_gap_m + params.ring_width_m + 5.0
    )
    chip, row, col = chip_around(scene, lon, lat, radius, metadata_provider)
    rows, cols = chip.shape
    fraction = (
        min(max(col / max(cols - 1, 1), 0.0), 1.0),
        min(max(row / max(rows - 1, 1), 0.0), 1.0),
    )
    return locate_imaged_target(chip, [fraction] * 4, h, params)


def read_slc_chip(
    layer: QgsRasterLayer,
    extent: QgsRectangle,
    metadata_provider: MetadataProvider,
) -> SlcChip:
    """Complex SLC chip of ``layer`` under ``extent`` (layer CRS), in file layout.

    Raises
    ------
    ValueError
        With a user-facing message when the layer or extent cannot be read.
    """
    if not isinstance(layer, QgsRasterLayer) or layer.bandCount() < 2:
        raise ValueError(_tr("Select an ICEYE SLC layer first."))
    metadata = metadata_provider.get(layer)
    if metadata is None:
        raise ValueError(_tr("The active layer has no ICEYE metadata."))
    bounds = get_extend_image_coords(layer, extent)
    if bounds is None:
        raise ValueError(_tr("Could not map the click to SLC pixels."))
    if (
        bounds.xMinimum() < 0
        or bounds.yMinimum() < 0
        or bounds.xMaximum() > layer.width()
        or bounds.yMaximum() > layer.height()
    ):
        raise ValueError(_tr("The click is too close to the edge of the SLC."))
    if bounds.width() * bounds.height() > MAX_CHIP_PIXELS:
        raise ValueError(_tr("The area around the click is too large to read."))
    slc = read_slc_data(layer, extent, metadata_provider)
    if slc is None:
        raise ValueError(_tr("Failed to read the SLC samples."))
    source = layer.dataProvider().dataSourceUri()
    left = (metadata.sar_observation_direction or "").lower() == "left"
    return SlcChip(
        data=patch_to_file_layout(slc.data_patch, left),
        col0=int(bounds.xMinimum()),
        row0=int(bounds.yMinimum()),
        geometry=ProductGeometry.from_properties(read_iceye_properties(source)),
        pixel_to_lonlat=gcp_pixel_to_lonlat(source),
    )


def chip_around(
    scene: MoverScene,
    lon: float,
    lat: float,
    radius_m: float,
    metadata_provider: MetadataProvider,
):
    """SLC chip within ``radius_m`` of a point, and the point's chip (row, col)."""
    dlat = radius_m / _METRES_PER_DEG
    dlon = dlat / max(math.cos(math.radians(lat)), 1e-6)
    extent = QgsRectangle(lon - dlon, lat - dlat, lon + dlon, lat + dlat)
    layer_crs = scene.layer.crs()
    if layer_crs.authid() != WGS84:
        extent = QgsCoordinateTransform(
            QgsCoordinateReferenceSystem(WGS84), layer_crs, QgsProject.instance()
        ).transformBoundingBox(extent)
    chip = read_slc_chip(scene.layer, extent, metadata_provider)
    col, row = scene.lonlat_to_pixel(lon, lat)
    return chip, row - chip.row0 - 0.5, col - chip.col0 - 0.5


# ----------------------------------------------------------------------------------
# Map tool
# ----------------------------------------------------------------------------------


class MoverRelocationMapTool(QgsMapTool):
    """Emits clicks and cursor moves in EPSG:4326; right click or Esc cancels."""

    clicked = pyqtSignal(object)  # QgsPointXY, lon/lat
    moved = pyqtSignal(object)  # QgsPointXY, lon/lat
    cancelled = pyqtSignal()

    def __init__(self, canvas) -> None:
        """Create the tool on ``canvas``."""
        super().__init__(canvas)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def to_wgs84(self, point: QgsPointXY) -> QgsPointXY:
        """Canvas CRS point to lon/lat."""
        crs = self.canvas().mapSettings().destinationCrs()
        if crs.authid() == WGS84:
            return QgsPointXY(point)
        transform = QgsCoordinateTransform(
            crs, QgsCoordinateReferenceSystem(WGS84), QgsProject.instance()
        )
        return transform.transform(point)

    def canvasMoveEvent(self, event) -> None:
        """Report the cursor position for the live readout."""
        self.moved.emit(self.to_wgs84(event.mapPoint()))

    def canvasReleaseEvent(self, event) -> None:
        """Left click: next input; right click: cancel."""
        if event.button() == Qt.MouseButton.RightButton:
            self.cancelled.emit()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.to_wgs84(event.mapPoint()))

    def keyPressEvent(self, event) -> None:
        """Esc cancels the current relocation."""
        if event.key() == Qt.Key.Key_Escape:
            self.cancelled.emit()


# ----------------------------------------------------------------------------------
# Output layers
# ----------------------------------------------------------------------------------


def _field_type(value: Any) -> QMetaType.Type:
    if isinstance(value, bool):
        return QMetaType.Type.Bool
    if isinstance(value, (int, float)) or value is None:
        return QMetaType.Type.Double
    return QMetaType.Type.QString


def _rgb(color: tuple[int, int, int], alpha: int = 255) -> str:
    return ",".join(str(c) for c in (*color, alpha))


def _haloed_line(
    color: tuple[int, int, int], width: float, dashed: bool = False
) -> list[QgsSimpleLineSymbolLayer]:
    """Return a black casing under a coloured line, visible on any pixel."""
    casing = QgsSimpleLineSymbolLayer(QColor(*HALO), width + 0.6)
    line = QgsSimpleLineSymbolLayer(QColor(*color), width)
    if dashed:
        line.setPenStyle(Qt.PenStyle.DashLine)
    return [casing, line]


def _arrowhead(color: tuple[int, int, int]) -> QgsMarkerLineSymbolLayer:
    """Return a filled arrowhead on a line's last vertex, pointing along the line."""
    head = QgsMarkerSymbol.createSimple(
        {
            "name": "filled_arrowhead",
            "color": _rgb(color),
            "size": "7",
            "outline_color": _rgb(HALO),
            "outline_width": "0.5",
        }
    )
    arrow = QgsMarkerLineSymbolLayer(True)
    arrow.setPlacements(Qgis.MarkerLinePlacement.LastVertex)
    arrow.setSubSymbol(head)
    return arrow


def _style(layer: QgsVectorLayer, key: str) -> None:
    """Default symbology of each output layer, made to stand out on grayscale SAR."""
    if key == "band":
        symbol = QgsFillSymbol.createSimple(
            {
                "color": _rgb(COLORS["band"], 45),
                "outline_style": "no",
            }
        )
        for symbol_layer in _haloed_line(COLORS["band"], 0.6):
            symbol.appendSymbolLayer(symbol_layer)
    elif key in ("displacement", "track"):
        symbol = QgsLineSymbol.createSimple({"color": _rgb(HALO), "width": "1.4"})
        symbol.deleteSymbolLayer(0)
        for symbol_layer in _haloed_line(COLORS[key], 0.8, dashed=key == "track"):
            symbol.appendSymbolLayer(symbol_layer)
        if key == "track":
            # The track runs from collection start to end: an arrowhead on its
            # last vertex shows the direction of travel.
            symbol.appendSymbolLayer(_arrowhead(COLORS[key]))
    else:
        shapes = {
            "ticks": ("circle", "3"),
            "imaged": ("circle", "5"),
            "true": ("star", "7"),
        }
        name, size = shapes[key]
        symbol = QgsMarkerSymbol.createSimple(
            {
                "name": name,
                "color": _rgb(COLORS[key]),
                "size": size,
                "outline_color": _rgb(HALO),
                "outline_width": "0.5",
            }
        )
    layer.renderer().setSymbol(symbol)
    if key == "ticks":
        text = QgsTextFormat()
        text.setColor(QColor(255, 255, 255))
        text.setSize(9)
        buffer = QgsTextBufferSettings()
        buffer.setEnabled(True)
        buffer.setSize(1.0)
        buffer.setColor(QColor(*HALO))
        text.setBuffer(buffer)
        labels = QgsPalLayerSettings()
        labels.fieldName = "label"
        labels.enabled = True
        labels.setFormat(text)
        layer.setLabeling(QgsVectorLayerSimpleLabeling(labels))
        layer.setLabelsEnabled(True)


# Custom property marking a layer as Mover Relocation output (value: layer key).
# Stored with the layer, so outputs are recognised after a plugin reload too.
OUTPUT_PROPERTY = "iceye_toolbox/mover_output"


class OutputLayers:
    """EPSG:4326 memory layers, created on first use and reused afterwards.

    Layers are found by ``OUTPUT_PROPERTY`` rather than remembered ids, so layers
    from an earlier panel instance are reused and cleaned up as well. QGIS makes
    every newly added layer the active one; with ``iface`` the previously active
    layer (the SLC) is restored so tools reading ``activeLayer()`` keep working.
    """

    def __init__(self, iface=None) -> None:
        """Start; existing tagged layers in the project are picked up on use."""
        self.iface = iface

    @staticmethod
    def layers(key: str | None = None) -> list[QgsVectorLayer]:
        """All Mover output layers in the project (only those of ``key`` if given)."""
        found = []
        for layer in QgsProject.instance().mapLayers().values():
            tag = layer.customProperty(OUTPUT_PROPERTY)
            if isinstance(layer, QgsVectorLayer) and tag and key in (None, tag):
                found.append(layer)
        return found

    def layer(self, key: str) -> QgsVectorLayer | None:
        """Return the output layer for ``key`` if there is one."""
        found = [lyr for lyr in self.layers(key) if lyr.isValid()]
        return found[0] if found else None

    def add(self, key: str, geometry: QgsGeometry, attributes: dict[str, Any]) -> None:
        """Append one feature, creating the layer with fields from ``attributes``."""
        layer = self.layer(key)
        if layer is None:
            kind, name = LAYERS[key]
            layer = QgsVectorLayer(f"{kind}?crs={WGS84}", _tr(name), "memory")
            layer.dataProvider().addAttributes(
                [QgsField(n, _field_type(v)) for n, v in attributes.items()]
            )
            layer.updateFields()
            layer.setCustomProperty(OUTPUT_PROPERTY, key)
            _style(layer, key)
            active = self.iface.activeLayer() if self.iface is not None else None
            QgsProject.instance().addMapLayer(layer)
            if active is not None and self.iface.activeLayer() is not active:
                self.iface.setActiveLayer(active)
        feature = QgsFeature(layer.fields())
        for name, value in attributes.items():
            if layer.fields().indexOf(name) >= 0:
                feature.setAttribute(name, value)
        feature.setGeometry(geometry)
        layer.dataProvider().addFeature(feature)
        layer.updateExtents()
        layer.triggerRepaint()

    def drop(self, *keys: str) -> None:
        """Remove the output layers of ``keys`` from the project (all if none given)."""
        doomed = [lyr.id() for key in (keys or (None,)) for lyr in self.layers(key)]
        if doomed:
            QgsProject.instance().removeMapLayers(doomed)


def _points(lonlats) -> list[QgsPointXY]:
    return [QgsPointXY(lon, lat) for lon, lat in lonlats]


def format_relocation(result: Relocation) -> str:
    """Rich-text summary with the traffic-light indicator."""
    color = _INDICATOR_COLORS.get(result.indicator, "#7f8c8d")
    lon, lat = result.true_lonlat
    lines = [
        f"<b style='color:{color}'>&#9679; {result.indicator.upper()}</b>"
        + _tr(" flags: {flags}").format(flags=", ".join(result.flags) or "-"),
        _tr("True position {lon:.6f}, {lat:.6f} at {utc}").format(
            lon=lon, lat=lat, utc=result.t_true_utc
        ),
        _speed_line(result),
        _tr(
            "v_r {vr:+.2f} &plusmn; {s:.2f} m/s (towards radar +), v_gr {vgr:+.2f} m/s"
        ).format(vr=result.v_r, s=result.sigma_v_r, vgr=result.v_gr),
        _tr("dx {dx:+.0f} &plusmn; {s:.0f} m").format(
            dx=result.dx_m, s=result.sigma_dx_m
        )
        + (
            ""
            if result.constraint_track_angle_deg is None
            else _tr(", road angle to track {a:.0f} deg").format(
                a=result.constraint_track_angle_deg
            )
        ),
    ]
    return "<br>".join(lines)


def _speed_line(result: Relocation) -> str:
    """Ground speed and heading, or the minimum speed in single-click mode."""
    if result.v_t is None:
        v = result.v_t_min
        return _tr(
            "v_t &ge; {v:.1f} m/s ({kmh:.0f} km/h, {kn:.1f} kn), heading unknown "
            "(single click)"
        ).format(v=v, kmh=v * MPS_TO_KMH, kn=v * MPS_TO_KNOTS)
    return _tr(
        "v_t {v:.1f} &plusmn; {s:.1f} m/s ({kmh:.0f} km/h, {kn:.1f} kn), "
        "heading {hdg:.0f} deg"
    ).format(
        v=result.v_t,
        s=result.sigma_v_t,
        kmh=result.v_t * MPS_TO_KMH,
        kn=result.v_t * MPS_TO_KNOTS,
        hdg=result.heading_deg,
    )


# ----------------------------------------------------------------------------------
# Panel
# ----------------------------------------------------------------------------------


class MoverRelocationDialog(QDialog):
    """Non-modal panel driving the two-click relocation on the map canvas."""

    def __init__(
        self,
        iface,
        metadata_provider: MetadataProvider | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the panel; the map tool is created on the iface canvas."""
        super().__init__(parent)
        self.iface = iface
        self.metadata_provider = metadata_provider or MetadataProvider()
        self.scene: MoverScene | None = None
        self.target: ImagedTarget | None = None
        self.band: Band | None = None
        self.clicks: list[QgsPointXY] = []
        self.last_result: Relocation | None = None
        self.step = STEP_IDLE
        self.outputs = OutputLayers(iface)

        self.setWindowTitle(_tr("Mover Relocation"))
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Window)

        canvas = iface.mapCanvas()
        self.map_tool = MoverRelocationMapTool(canvas)
        self.map_tool.clicked.connect(self.handle_click)
        self.map_tool.moved.connect(self.handle_move)
        self.map_tool.cancelled.connect(self.reset)
        # Saturated colours with a black casing stay visible on grayscale SAR.
        self._band_rb = QgsRubberBand(canvas, Qgis.GeometryType.Polygon)
        self._band_rb.setColor(QColor(*COLORS["band"]))
        self._band_rb.setFillColor(QColor(*COLORS["band"], 50))
        self._band_rb.setSecondaryStrokeColor(QColor(*HALO))
        self._band_rb.setWidth(2)
        self._constraint_rb = QgsRubberBand(canvas, Qgis.GeometryType.Line)
        self._constraint_rb.setColor(QColor(*COLORS["constraint"]))
        self._constraint_rb.setSecondaryStrokeColor(QColor(*HALO))
        self._constraint_rb.setWidth(3)
        self._points_rb = QgsRubberBand(canvas, Qgis.GeometryType.Point)
        self._points_rb.setColor(QColor(*COLORS["constraint"]))
        self._points_rb.setSecondaryStrokeColor(QColor(*HALO))
        self._points_rb.setIconSize(12)
        self._points_rb.setWidth(3)

        self.class_combo = QComboBox()
        for name, target_class in TARGET_CLASSES.items():
            self.class_combo.addItem(
                _tr("{name} (up to {v:.0f} m/s)").format(
                    name=name, v=target_class.v_max_mps
                ),
                name,
            )
        self.class_combo.setToolTip(
            _tr(
                "Type of moving target. Its maximum speed sets how long the band is "
                "(the band covers every position a target this fast could truly be "
                "at) and the speed limit of the plausibility check. Ships are "
                "placed on the sea surface; cars and trains on the image surface."
            )
        )
        self.margin_spin = QDoubleSpinBox()
        self.margin_spin.setRange(0.0, 500.0)
        self.margin_spin.setSuffix(" m")
        self.margin_spin.setValue(RelocationSettings.band_margin_m)
        self.margin_spin.setToolTip(
            _tr(
                "Extra half-width of the band beyond half the target's size, in "
                "ground metres. Wider makes the band easier to see and lets the "
                "cursor readout work slightly off the exact range line."
            )
        )
        self.tick_spin = QDoubleSpinBox()
        self.tick_spin.setRange(0.5, 50.0)
        self.tick_spin.setSuffix(" m/s")
        self.tick_spin.setValue(RelocationSettings.tick_step_mps)
        self.tick_spin.setToolTip(
            _tr(
                "Spacing of the speed ticks along the band, in radial velocity "
                "|v_r| (towards / away from the radar). Each tick also shows the "
                "minimum ground speed that radial velocity implies."
            )
        )
        self.detect_check = QCheckBox(_tr("Snap the click to the target hull"))
        self.detect_check.setChecked(True)
        self.detect_check.setToolTip(
            _tr(
                "On: the target click searches about 15 m around itself for the "
                "bright target and uses its intensity-weighted centre. Off: the "
                "clicked point itself is the imaged position."
            )
        )
        self.residual_check = QCheckBox(_tr("Apply range residual (fast movers)"))
        self.residual_check.setToolTip(
            _tr(
                "A mover's true slant range is slightly longer than its imaged one "
                "(about 6 mm at 1 m/s, 0.9 m at 12.5 m/s). Turn on for fast "
                "targets; negligible for ships and slow traffic."
            )
        )
        self.extrapolate_check = QCheckBox(
            _tr("Allow constraint points on one side of the band")
        )
        self.extrapolate_check.setToolTip(
            _tr(
                "Normally the two constraint clicks must lie on opposite sides of "
                "the band. Turn on to extend the line through both clicks to the "
                "band instead (less accurate)."
            )
        )

        self.single_check = QCheckBox(
            _tr("Single click on the constraint (rough heading from the image)")
        )
        self.single_check.setToolTip(
            _tr(
                "On: after the target, click once where the road / rail / bridge "
                "deck / wake crosses the yellow band. The true position, "
                "displacement and radial velocity are as with two clicks. The road "
                "direction is estimated from the image about 50 m around the click "
                "(dominant linear feature); two cyan points are placed along it to "
                "give a rough speed and heading (about +-4 deg on clean features), "
                "so check that they follow the road. Where the area is not clearly "
                "linear only a minimum ground speed is given and the heading stays "
                "unknown. Off: click two points along the constraint."
            )
        )
        self.single_check.toggled.connect(self._on_mode_toggled)

        form = QFormLayout()
        form.addRow(_tr("Target class"), self.class_combo)
        form.addRow(_tr("Band margin"), self.margin_spin)
        form.addRow(_tr("Tick step |v_r|"), self.tick_spin)
        form.addRow(self.detect_check)
        form.addRow(self.residual_check)
        form.addRow(self.extrapolate_check)
        form.addRow(self.single_check)
        legend = QLabel(
            _tr(
                "<span style='color:#b8a000'>&#9632;</span> band of possible true "
                "positions &nbsp; <span style='color:#ff00cc'>&#9679;</span> imaged "
                "&nbsp; <span style='color:#2fbf10'>&#9733;</span> true &nbsp; "
                "<span style='color:#00b8cc'>&#9679;</span> constraint clicks. "
                "Hover over a setting for what it does."
            )
        )
        legend.setWordWrap(True)
        legend.setTextFormat(Qt.TextFormat.RichText)
        form.addRow(legend)

        self.start_button = QPushButton(_tr("Pick target"))
        self.start_button.setToolTip(_tr("Click the imaged target on the map"))
        self.start_button.setDefault(True)
        self.start_button.clicked.connect(self.start)
        reset_btn = QPushButton(_tr("Reset"))
        reset_btn.setToolTip(
            _tr(
                "Remove everything Mover Relocation added: all Mover layers (band, "
                "ticks, imaged and true positions, displacements, tracks) and the "
                "canvas overlays. Esc or right-click on the map only cancels the "
                "current target."
            )
        )
        reset_btn.clicked.connect(self.reset_all)
        close_btn = QPushButton(_tr("Close"))
        close_btn.clicked.connect(self.close)

        self._step_label = QLabel()
        self._step_label.setWordWrap(True)
        self._readout = QLabel()
        self._readout.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._result = QLabel()
        self._result.setWordWrap(True)
        self._result.setTextFormat(Qt.TextFormat.RichText)
        self._result.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addStretch(1)
        buttons.addWidget(reset_btn)
        buttons.addWidget(close_btn)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.addLayout(form)
        root.addWidget(self._step_label)
        root.addWidget(self._readout)
        root.addWidget(self._result, 1)
        root.addLayout(buttons)
        self._set_step(STEP_IDLE)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def settings(self) -> RelocationSettings:
        """Relocation settings from the panel controls."""
        return RelocationSettings(
            band_margin_m=self.margin_spin.value(),
            tick_step_mps=self.tick_spin.value(),
            range_residual=self.residual_check.isChecked(),
            allow_extrapolation=self.extrapolate_check.isChecked(),
        )

    def target_class(self):
        """Return the selected target class."""
        return TARGET_CLASSES[self.class_combo.currentData()]

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def _set_step(self, step: str) -> None:
        self.step = step
        prompts = {
            STEP_IDLE: _tr("Select an ICEYE SLC layer, then Pick target."),
            STEP_TARGET: _tr("Step 1: click the imaged (displaced) target."),
            STEP_CONSTRAINT: (
                _tr(
                    "Step 2: click once where the road / rail / bridge deck / wake "
                    "crosses the band."
                )
                if self.single_check.isChecked()
                else _tr(
                    "Step 2: click two points on the road / rail / bridge deck / "
                    "wake, one on each side of the band ({n}/2)."
                ).format(n=len(self.clicks))
            )
            + _tr(
                " On bridges click the deck's bright line, not its reflection on "
                "the water."
            ),
            STEP_DONE: _tr("Done. Click another target, or Reset."),
        }
        self._step_label.setText(prompts[step])

    def _activate_tool(self) -> None:
        canvas = self.iface.mapCanvas()
        if canvas.mapTool() is not self.map_tool:
            canvas.setMapTool(self.map_tool)

    def _slc_layer(self, layer) -> QgsRasterLayer | None:
        """``layer`` if it is an SLC, else the SLC already in use if still loaded."""
        if isinstance(layer, QgsRasterLayer) and layer.bandCount() >= 2:
            return layer
        if self.scene is not None:
            current = QgsProject.instance().mapLayer(self.scene.layer_id)
            if isinstance(current, QgsRasterLayer):
                return current
        return None

    def _ensure_scene(self, layer) -> bool:
        layer = self._slc_layer(layer)
        if layer is None:
            self._step_label.setText(_tr("Select an ICEYE SLC layer first."))
            return False
        if self.metadata_provider.get(layer) is None:
            self._step_label.setText(_tr("The active layer has no ICEYE metadata."))
            return False
        if self.scene is None or self.scene.layer_id != layer.id():
            try:
                self.scene = MoverScene.from_layer(layer)
            except Exception as e:
                self._step_label.setText(_tr("Cannot read the SLC: {e}").format(e=e))
                return False
        # Keep the SLC active so the tool and the toolbar
        # policy all keep seeing the image layer.
        if self.iface.activeLayer() is not layer:
            self.iface.setActiveLayer(layer)
        return True

    def start(self) -> bool:
        """Activate the map tool and wait for the target click."""
        if not self._ensure_scene(self.iface.activeLayer()):
            return False
        self.reset()
        self._activate_tool()
        self._set_step(STEP_TARGET)
        return True

    def reset(self) -> None:
        """Cancel the current target: clicks and its band / ticks (results kept)."""
        self.target = self.band = None
        self.clicks = []
        self._clear_band()
        self._constraint_rb.reset(Qgis.GeometryType.Line)
        self._points_rb.reset(Qgis.GeometryType.Point)
        self._readout.clear()
        if self.step != STEP_IDLE:
            self._set_step(STEP_TARGET)

    def reset_all(self) -> None:
        """Remove every Mover output layer and overlay, then await a new target."""
        self.reset()
        self.outputs.drop()
        self.last_result = None
        self._result.clear()

    def set_target(
        self, target: ImagedTarget, layer: QgsRasterLayer | None = None
    ) -> bool:
        """Use an already located ``target`` and wait for the constraint."""
        if not self._ensure_scene(layer or self.iface.activeLayer()):
            return False
        self.reset()
        self._activate_tool()
        self._use_target(target)
        return True

    def handle_click(self, point: QgsPointXY) -> None:
        """Next input of the two-click workflow (``point`` in lon/lat)."""
        if self.step in (STEP_TARGET, STEP_DONE):
            if self.scene is None:
                return
            self.reset()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                target = target_from_click(
                    self.scene,
                    point.x(),
                    point.y(),
                    self.metadata_provider,
                    detect_hull=self.detect_check.isChecked(),
                )
            except Exception as e:
                self._step_label.setText(
                    _tr("Could not read the target: {e}").format(e=e)
                )
                return
            finally:
                QApplication.restoreOverrideCursor()
            self._use_target(target)
        elif self.step == STEP_CONSTRAINT:
            self.clicks.append(QgsPointXY(point))
            self._draw_clicks()
            self._set_step(STEP_CONSTRAINT)
            if len(self.clicks) >= self._clicks_needed():
                self._finish()

    def _clicks_needed(self) -> int:
        return 1 if self.single_check.isChecked() else 2

    def _on_mode_toggled(self, _checked: bool) -> None:
        """Restart the constraint step when switching between one and two clicks."""
        if self.step == STEP_CONSTRAINT:
            self.clicks = []
            self._points_rb.reset(Qgis.GeometryType.Point)
            self._constraint_rb.reset(Qgis.GeometryType.Line)
            self._set_step(STEP_CONSTRAINT)

    def handle_move(self, point: QgsPointXY) -> None:
        """Live ``|v_r|`` readout while the cursor is inside the band."""
        if self.band is None or self.scene is None or self.step != STEP_CONSTRAINT:
            return
        readout = cursor_readout(
            self.scene.geometry,
            self.target,
            self.band,
            self.scene.ecef(point.x(), point.y()),
        )
        if readout is None:
            self._readout.setText(_tr("Cursor outside the band"))
            return
        self._readout.setText(
            _tr(
                "|v_r| {v:.1f} m/s ({kmh:.0f} km/h, {kn:.1f} kn), |dx| {dx:.0f} m"
            ).format(
                v=readout.v_r_abs,
                kmh=readout.kmh,
                kn=readout.knots,
                dx=readout.dx_abs_m,
            )
        )

    # ------------------------------------------------------------------
    # Band and relocation
    # ------------------------------------------------------------------

    def _use_target(self, target: ImagedTarget) -> None:
        scene = self.scene
        target_class = self.target_class()
        try:
            limits = scene.time_limits(target)
        except ValueError:
            limits = None
        band = band_for_target(
            scene.geometry, target, target_class, self.settings(), limits
        )
        self._clear_band()
        self.target, self.band = target, band
        crs = QgsCoordinateReferenceSystem(WGS84)
        self._band_rb.setToGeometry(
            QgsGeometry.fromPolygonXY([_points(band.polygon_lonlat)]), crs
        )
        self._write_band(band)
        # Adding layers can make other tools hand the canvas back to Pan (via the
        # toolbar policy's layer-change hooks); the constraint clicks come next.
        self._activate_tool()
        self._set_step(STEP_CONSTRAINT)
        if band.clipped:
            self._result.setText(_tr("The band is clipped by the image extent."))
        else:
            self._result.clear()

    def _draw_clicks(self) -> None:
        crs = QgsCoordinateReferenceSystem(WGS84)
        self._points_rb.setToGeometry(QgsGeometry.fromMultiPointXY(self.clicks), crs)
        if len(self.clicks) == 2:
            self._constraint_rb.setToGeometry(
                QgsGeometry.fromPolylineXY(self.clicks), crs
            )

    def _finish(self) -> None:
        scene, target = self.scene, self.target
        points = [scene.ecef(p.x(), p.y()) for p in self.clicks]
        target_class = self.target_class()
        # Ships sit on the sea surface; other targets stay on the display surface.
        height = scene.geometry.scene_height if target_class.name == "ship" else None
        try:
            note = ""
            if len(points) == 1:
                result = relocate_single_click(
                    scene.geometry,
                    target,
                    points[0],
                    target_class,
                    self.settings(),
                    band=self.band,
                    target_height=height,
                )
                result, note = self._with_image_axis(result, points[0], height)
            else:
                result = relocate(
                    scene.geometry,
                    target,
                    points[0],
                    points[1],
                    target_class,
                    self.settings(),
                    band=self.band,
                    target_height=height,
                )
        except ConstraintError as e:
            self.clicks = []
            self._points_rb.reset(Qgis.GeometryType.Point)
            self._constraint_rb.reset(Qgis.GeometryType.Line)
            self._result.setText(str(e))
            self._set_step(STEP_CONSTRAINT)
            return
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Mover relocation failed: {e}",
                "ICEYE Toolbox",
                Qgis.MessageLevel.Warning,
            )
            self._result.setText(_tr("Relocation failed: {e}").format(e=e))
            return
        self.last_result = result
        self._result.setText(format_relocation(result) + note)
        if result.mode == MODE_SINGLE_CLICK_AUTO:
            self._show_spawned_points(result)
        self._write_result(result)
        self._clear_band()
        self._activate_tool()
        self._set_step(STEP_DONE)

    def _with_image_axis(
        self, basic: Relocation, point, height: float | None
    ) -> tuple[Relocation, str]:
        """Upgrade a single-click result with the image's road axis if it is clear."""
        scene, settings = self.scene, self.settings()
        click = self.clicks[0]
        try:
            chip, row, col = chip_around(
                scene,
                click.x(),
                click.y(),
                settings.axis_radius_m + 10.0,
                self.metadata_provider,
            )
            axis = estimate_constraint_axis(chip, row, col, settings)
        except Exception as e:
            return basic, "<br>" + _tr("Road direction not estimated: {e}").format(e=e)
        if axis.coherence < settings.min_axis_coherence:
            return basic, "<br>" + _tr(
                "No clear linear feature at the click (coherence {c:.2f}); heading "
                "unknown. Use two clicks instead."
            ).format(c=axis.coherence)
        try:
            result = relocate_from_axis(
                scene.geometry,
                self.target,
                point,
                axis,
                self.target_class(),
                settings,
                band=self.band,
                target_height=height,
            )
        except ConstraintError:
            return basic, "<br>" + _tr(
                "The estimated road direction runs along the band; heading unknown. "
                "Use two clicks instead."
            )
        return result, "<br>" + _tr(
            "Road direction estimated from the image (coherence {c:.2f}); check the "
            "cyan points."
        ).format(c=axis.coherence)

    def _show_spawned_points(self, result: Relocation) -> None:
        crs = QgsCoordinateReferenceSystem(WGS84)
        spawned = _points(result.constraint_lonlat)
        self._points_rb.setToGeometry(QgsGeometry.fromMultiPointXY(spawned), crs)
        self._constraint_rb.setToGeometry(QgsGeometry.fromPolylineXY(spawned), crs)

    # ------------------------------------------------------------------
    # Output layers
    # ------------------------------------------------------------------

    def _clear_band(self) -> None:
        """Remove the band and ticks: they only ever show the current target."""
        self.outputs.drop("band", "ticks")
        self._band_rb.reset(Qgis.GeometryType.Polygon)

    def _write_band(self, band: Band) -> None:
        target = band.target
        common = {"target_class": band.target_class.name}
        self.outputs.add(
            "band",
            QgsGeometry.fromPolygonXY([_points(band.polygon_lonlat)]),
            {
                **common,
                "v_max": band.target_class.v_max_mps,
                "dt_max_s": band.dt_max,
                "dx_max_m": band.dx_max_m,
                "half_width_m": band.half_width_m,
                "clipped": band.clipped,
            },
        )
        for tick in band.ticks:
            self.outputs.add(
                "ticks",
                QgsGeometry.fromPointXY(QgsPointXY(*tick.lonlat)),
                {
                    "v_r": tick.v_r,
                    "v_r_abs": abs(tick.v_r),
                    "v_ground_min": tick.v_ground_min,
                    "label": tick.label,
                },
            )
        self.outputs.add(
            "imaged",
            QgsGeometry.fromPointXY(QgsPointXY(*target.lonlat)),
            {
                **common,
                "r_img_m": target.slant_range,
                "t_img_s": target.time,
                "half_extent_m": target.half_extent_m,
            },
        )

    def _write_result(self, result: Relocation) -> None:
        attributes = result.attributes()
        true_point = QgsPointXY(*result.true_lonlat)
        self.outputs.add("true", QgsGeometry.fromPointXY(true_point), attributes)
        self.outputs.add(
            "displacement",
            QgsGeometry.fromPolylineXY([QgsPointXY(*result.target.lonlat), true_point]),
            attributes,
        )
        if result.track_lonlat:
            self.outputs.add(
                "track",
                QgsGeometry.fromPolylineXY(_points(result.track_lonlat)),
                attributes,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        """Release the map tool and clear the canvas overlays."""
        self.reset()
        self._set_step(STEP_IDLE)
        canvas = self.iface.mapCanvas()
        if canvas.mapTool() is self.map_tool:
            canvas.unsetMapTool(self.map_tool)
        super().closeEvent(event)
