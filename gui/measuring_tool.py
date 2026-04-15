"""Measuring toolbar: SAR-geometry-aware height ruler and IRF analysis."""

from __future__ import annotations

import math

import numpy as np
from qgis.core import (
    Qgis,
    QgsDistanceArea,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
)
from qgis.PyQt.QtCore import QCoreApplication, QEvent, QPointF, Qt
from qgis.PyQt.QtGui import QBrush, QColor, QFont, QIcon, QPainter, QPainterPath, QPen
from qgis.PyQt.QtWidgets import (
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsTextItem,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..core.geometry import SARViewGeometry, get_geometry_from_metadata
from ..core.irf import analyze_point, map_point_to_pixel, read_slc_chip
from ..core.metadata import MetadataProvider
from .irf_dialog import IRFResultDialog
from .sar_canvas_overlay import (
    BaseInteractionFilter,
    SARItemPlacementTool,
    SAROverlayToolbarBase,
    _map_point_to_scene,
)
from .toolbar_button_policy import ToolbarButtonPolicy


def _tr(message: str) -> str:
    """Translate message for ICEYE Toolbox context."""
    return QCoreApplication.translate("ICEYE Toolbox", message)


class HeightRulerItem(QGraphicsItemGroup):
    """SAR height ruler overlay: shadow line + layover line anchored to a map point.

    Line lengths represent real ground distances and scale with canvas zoom.
    Scroll the mouse wheel (via RulerInteractionFilter) to adjust the layover
    length; the corresponding building height is derived and displayed as a label.
    """

    _DEFAULT_LAYOVER_M = 50.0
    _MIN_LAYOVER_M = 1.0

    def __init__(self, canvas, pos: QPointF = QPointF(0, 0)) -> None:
        """Create a height ruler group at *pos* in scene coordinates.

        Args:
            canvas: The QGIS map canvas, used for zoom-aware line scaling.
            pos: Initial scene position (origin = building base).
        """
        super().__init__()
        self.setPos(pos)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self._canvas = canvas
        self._layover_ground_m: float = self._DEFAULT_LAYOVER_M
        self._height_m: float = 0.0
        self._graze: float = 45.0
        self._incidence: float = 45.0
        self._range_spacing: float = 1.0
        self._is_slc: bool = False
        self._rot_shadow: float = 90.0
        self._rot_layover: float = 180.0
        self._geo_point = None
        self._layer_id: str | None = None

        _halo_pen = QPen(QColor(255, 215, 0, 140), 14, Qt.SolidLine)
        _halo_pen.setCapStyle(Qt.RoundCap)
        _halo_pen.setJoinStyle(Qt.RoundJoin)
        self._halo = QGraphicsPathItem(self)
        self._halo.setPen(_halo_pen)
        self._halo.setVisible(False)

        self.shadow_line = QGraphicsLineItem(0, 0, 50, 0, self)
        self.shadow_line.setPen(QPen(QColor("sandybrown"), 4, Qt.SolidLine))
        self.shadow_line.setRotation(self._rot_shadow)

        self.layover_line = QGraphicsLineItem(0, 0, 50, 0, self)
        self.layover_line.setPen(QPen(QColor("plum"), 4, Qt.SolidLine))
        self.layover_line.setRotation(self._rot_layover)

        label_font = QFont("Arial", 10, QFont.Bold)

        self._label_bg = QGraphicsPathItem(self)
        self._label_bg.setPen(QPen(Qt.NoPen))
        self._label_bg.setBrush(QBrush(QColor(0, 0, 0, 180)))

        self._height_label = QGraphicsTextItem(self)
        self._height_label.setDefaultTextColor(QColor(255, 230, 0, 255))
        self._height_label.setFont(label_font)

        self.addToGroup(self._halo)
        self.addToGroup(self.shadow_line)
        self.addToGroup(self.layover_line)
        self.addToGroup(self._label_bg)
        self.addToGroup(self._height_label)

        self._update_lines()

    def updateGeometry(self, geometry: SARViewGeometry, incidence_angle: float) -> None:
        """Re-orient lines to match shadow/layover directions from geometry."""
        self._rot_shadow = geometry.shadow - geometry.north - 90
        self._rot_layover = geometry.layover - geometry.north - 90
        self._graze = geometry.graze
        self._incidence = incidence_angle
        self.shadow_line.setRotation(self._rot_shadow)
        self.layover_line.setRotation(self._rot_layover)
        self._update_lines()

    def setRotation(self, angle: float) -> None:
        """Negate the canvas rotation to keep the height label horizontal."""
        super().setRotation(angle)
        self._height_label.setRotation(-angle)
        self._label_bg.setRotation(-angle)

    def set_highlighted(self, active: bool) -> None:
        """Show or hide the halo path to indicate this ruler is under the cursor."""
        self._halo.setVisible(active)

    def set_layover(self, layover_m: float) -> None:
        """Set the layover ground length, derive height, and refresh lines and label."""
        self._layover_ground_m = max(self._MIN_LAYOVER_M, layover_m)
        self._update_lines()

    def hoverEnterEvent(self, event) -> None:
        """Highlight the ruler when the cursor enters its area."""
        self.set_highlighted(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        """Remove highlight when the cursor leaves the ruler area."""
        self.set_highlighted(False)
        super().hoverLeaveEvent(event)

    def _update_lines(self) -> None:
        """Recompute line lengths in scene pixels from current layover length and zoom.

        Height is derived from layover ground length and incidence angle:
            layover_ground = H * cos(θ) / sin(θ)  →  H = layover_ground * tan(θ)
        Shadow ground length:
            shadow_ground = H / tan(graze)
        Both converted to screen pixels via mapUnitsPerPixel().
        """
        mpp = self._canvas.mapUnitsPerPixel()
        if mpp <= 0:
            return

        da = QgsDistanceArea()
        da.setSourceCrs(
            self._canvas.mapSettings().destinationCrs(),
            QgsProject.instance().transformContext(),
        )
        da.setEllipsoid(QgsProject.instance().ellipsoid())
        center = self._canvas.center()
        p1 = QgsPointXY(center.x() - mpp / 2, center.y())
        p2 = QgsPointXY(center.x() + mpp / 2, center.y())
        metres_per_pixel = da.measureLine(p1, p2)

        graze_r = math.radians(self._graze)
        theta_r = math.radians(self._incidence)

        tan_graze = math.tan(graze_r) if math.tan(graze_r) > 1e-6 else 1e-6
        tan_theta = math.tan(theta_r) if abs(math.tan(theta_r)) > 1e-6 else 1e-6

        self._height_m = self._layover_ground_m * tan_theta

        shadow_ground_m = self._height_m / tan_graze
        shadow_px = shadow_ground_m / metres_per_pixel

        layover_px = self._layover_ground_m / metres_per_pixel

        shadow_rad = math.radians(self._rot_shadow)
        layover_rad = math.radians(self._rot_layover)
        halo_path = QPainterPath()
        halo_path.moveTo(
            shadow_px * math.cos(shadow_rad),
            shadow_px * math.sin(shadow_rad),
        )
        halo_path.lineTo(0, 0)
        halo_path.lineTo(
            layover_px * math.cos(layover_rad),
            layover_px * math.sin(layover_rad),
        )
        self._halo.setPath(halo_path)

        self.shadow_line.setLine(0, 0, shadow_px, 0)
        self.layover_line.setLine(0, 0, layover_px, 0)

        self._height_label.setPlainText(f"H: {self._height_m:.1f} m")
        self._height_label.setPos(QPointF(12, 12))
        br = self._height_label.boundingRect().adjusted(-4, -2, 4, 2)
        bg_path = QPainterPath()
        bg_path.addRoundedRect(br, 4, 4)
        self._label_bg.setPath(bg_path)
        self._label_bg.setPos(self._height_label.pos())


class RulerInteractionFilter(BaseInteractionFilter):
    """Install on the canvas viewport once at setup; remove on unload.

    Canvas viewport event filter that enables drag-to-move and Shift+scroll resize
    for HeightRulerItems regardless of which map tool is currently active.
    """

    _HIT_RADIUS = 50.0
    _remove_label = "Remove height ruler"

    def _handle_wheel(self, event) -> bool:
        modifiers = event.modifiers()
        if modifiers & Qt.ShiftModifier:
            step_multiplier = 10.0
        elif modifiers & Qt.ControlModifier:
            step_multiplier = 1.0
        else:
            return False
        map_point = self._viewport_to_map(event.pos())
        hit = self._hit_item(map_point)
        if hit is None:
            return False
        delta = event.angleDelta().y() / 120
        sin_theta = math.sin(math.radians(hit._incidence))
        if hit._is_slc:
            step = hit._range_spacing / sin_theta if sin_theta > 1e-6 else 1.0
        else:
            step = hit._range_spacing
        step *= step_multiplier
        hit.set_layover(hit._layover_ground_m + delta * step)
        event.accept()
        return True


class HeightRulerTool(SARItemPlacementTool):
    """Placement tool for height rulers — one-shot: deactivates after a single placement."""

    _remove_label = "Remove height ruler"


class IRFPointTool(SARItemPlacementTool):
    """One-shot placement tool that triggers IRF analysis on click.

    No overlay items are placed on the canvas; the click simply fires the
    ``placed`` signal which the toolbar action handles to run the analysis.
    """

    _remove_label = "Cancel IRF analysis"


class MeasuringToolbarAction(SAROverlayToolbarBase):
    """ICEYE Measuring Tools toolbar — height ruler and IRF analysis."""

    _IRF_HALF_WINDOW = 64

    def __init__(
        self,
        iface,
        metadata_provider: MetadataProvider | None = None,
        toolbar_button_policy: ToolbarButtonPolicy | None = None,
    ) -> None:
        """Initialise the measuring toolbar action.

        Args:
            iface: QGIS interface handle.
            metadata_provider: Provider for ICEYE layer metadata; used to
                read SAR geometry and product type at placement time.
            toolbar_button_policy: Optional policy that enables/disables controls
                from the active layer.
        """
        super().__init__(iface, metadata_provider)
        self._toolbar_button_policy = toolbar_button_policy
        self._irf_placed_items: list = []
        self._irf_btn: QToolButton | None = None
        self._irf_tool: IRFPointTool | None = None
        self._irf_dialog: IRFResultDialog | None = None

    def setup(self) -> None:
        """Create the toolbar, wire signals."""
        self.toolbar = self.iface.addToolBar("ICEYE Measuring")
        self.toolbar.setObjectName("ICEYE Measuring")

        # --- Height ruler button ---
        place_menu = QMenu(self.toolbar)
        clear_action = place_menu.addAction(_tr("Clear height rulers"))
        clear_action.triggered.connect(self._clear_items)
        place_menu.aboutToShow.connect(
            lambda: clear_action.setEnabled(len(self._placed_items) > 0)
        )

        self._place_btn = QToolButton(self.toolbar)
        self._place_btn.setIcon(QIcon(":/plugins/ICEYEToolbox/ruler.svg"))
        self._place_btn.setToolTip(_tr("Place Height Ruler"))
        self._place_btn.setCheckable(True)
        self._place_btn.setMenu(place_menu)
        self._place_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._place_btn.clicked.connect(self._activate_tool)
        self.toolbar.addWidget(self._place_btn)

        canvas = self.iface.mapCanvas()
        self._placement_tool = HeightRulerTool(canvas, self._placed_items)
        self._placement_tool.setButton(self._place_btn)
        self._placement_tool.placed.connect(self._on_placement_click)
        self._placement_tool.removed.connect(self._remove_item)
        self._placement_tool.moved.connect(self._on_item_moved)
        self._placement_tool.deactivated.connect(self._on_placement_deactivated)

        self._interaction_filter = RulerInteractionFilter(
            canvas,
            self._placed_items,
            on_moved=self._on_item_moved,
            on_removed=self._remove_item,
        )
        canvas.viewport().installEventFilter(self._interaction_filter)

        canvas.extentsChanged.connect(self._reposition_items)
        canvas.extentsChanged.connect(self._recompute_ruler_lengths)
        self.iface.currentLayerChanged.connect(self._on_layer_changed)

        # --- IRF analysis button ---

        self._irf_btn = QToolButton(self.toolbar)
        self._irf_btn.setIcon(QIcon(":/plugins/ICEYEToolbox/irf.svg"))
        self._irf_btn.setToolTip(
            _tr("IRF Analysis — click a point target on an SLC layer")
        )
        self._irf_btn.setCheckable(True)
        self._irf_btn.clicked.connect(self._activate_irf)
        self.toolbar.addWidget(self._irf_btn)

        self._irf_tool = IRFPointTool(canvas, self._irf_placed_items)
        self._irf_tool.setButton(self._irf_btn)
        self._irf_tool.placed.connect(self._on_irf_click)
        self._irf_tool.deactivated.connect(self._on_irf_deactivated)

        if self._toolbar_button_policy is not None:
            self._toolbar_button_policy.register(
                self._irf_btn,
                self._toolbar_button_policy.enabled_if_slc_cog,
                on_disable=self._on_irf_policy_disabled,
            )
            self._toolbar_button_policy.register(
                self._place_btn,
                self._toolbar_button_policy.enabled_if_iceye_layer,
                on_disable=self._on_height_ruler_policy_disabled,
            )

    # ------------------------------------------------------------------
    # Height ruler slots
    # ------------------------------------------------------------------

    def _on_placement_click(self, map_point) -> None:
        """Spawn a HeightRulerItem at the clicked map location, then deactivate tool."""
        meta = self.metadata_provider.get(self.iface.activeLayer())
        geometry, incidence_angle = self._local_sar_geometry(map_point)
        if geometry is None:
            QgsMessageLog.logMessage(
                "No active ICEYE layer — activate an ICEYE layer first",
                "ICEYE Toolbox",
                Qgis.Warning,
            )
            return

        canvas = self.iface.mapCanvas()
        scene_pt = _map_point_to_scene(canvas, map_point)

        ruler = HeightRulerItem(canvas=canvas, pos=scene_pt)
        ruler._geo_point = map_point
        ruler._range_spacing = (
            meta.sar_pixel_spacing_range
            if meta and meta.sar_pixel_spacing_range
            else 1.0
        )
        layer = self.iface.activeLayer()
        layer_name = layer.name() if layer else ""
        ruler._is_slc = "SLC" in layer_name.upper()
        ruler._layer_id = layer.id() if layer else None
        ruler.updateGeometry(geometry, incidence_angle)
        ruler.setRotation(canvas.rotation())
        canvas.rotationChanged.connect(ruler.setRotation)
        canvas.scene().addItem(ruler)
        self._placed_items.append(ruler)

        self._activate_tool(False)

    def _recompute_ruler_lengths(self) -> None:
        """Refresh line lengths for all rulers after a zoom change."""
        for ruler in self._placed_items:
            ruler._update_lines()

    def _on_layer_changed(self, layer) -> None:
        """Re-orient rulers that were placed on *layer* when it becomes active."""
        if layer is None:
            return
        meta = self.metadata_provider.get(layer)
        if meta is None:
            return
        geometry = get_geometry_from_metadata(meta)
        if geometry is None:
            return
        geometry.radian2deg()
        incidence_angle = meta.view_incidence_angle or 45.0
        layer_id = layer.id()
        for ruler in self._placed_items:
            if ruler._layer_id == layer_id:
                ruler.updateGeometry(geometry, incidence_angle)

    def _on_height_ruler_policy_disabled(self) -> None:
        """Turn off Height Ruler when the active layer no longer matches policy."""
        self._activate_tool(False)

    # ------------------------------------------------------------------
    # IRF analysis slots
    # ------------------------------------------------------------------

    def _activate_irf(self, checked: bool = True) -> None:
        """Activate or deactivate the IRF point-click tool."""
        if self._irf_tool is None:
            return
        canvas = self.iface.mapCanvas()
        if checked:
            canvas.setMapTool(self._irf_tool)
        else:
            canvas.unsetMapTool(self._irf_tool)
            self.iface.actionPan().trigger()

    def _on_irf_deactivated(self) -> None:
        """Uncheck the IRF button when the map tool is deactivated."""
        if self._irf_btn is not None:
            self._irf_btn.setChecked(False)

    def _on_irf_policy_disabled(self) -> None:
        """Turn off IRF map tool when the active layer no longer matches policy."""
        self._activate_irf(False)

    def _on_irf_click(self, map_point) -> None:
        """Run IRF analysis at the clicked map point and display results.

        Converts the map click to raster pixel coordinates, reads a chip of
        SLC data centred on the click, finds the brightest pixel inside the
        chip as the target, and runs analyze_point() from core.irf.
        """
        layer = self.iface.activeLayer()
        meta = self.metadata_provider.get(layer)

        if meta is None:
            self.iface.messageBar().pushMessage(
                _tr("IRF Analysis"),
                _tr("No active ICEYE layer — activate an SLC layer first."),
                level=Qgis.Warning,
                duration=4,
            )
            self._activate_irf(False)
            return

        pixel = map_point_to_pixel(layer, map_point)
        if pixel is None:
            self.iface.messageBar().pushMessage(
                _tr("IRF Analysis"),
                _tr("Could not map click position to raster pixel coordinates."),
                level=Qgis.Warning,
                duration=4,
            )
            self._activate_irf(False)
            return

        col, row = pixel
        chip = read_slc_chip(layer, row, col, self._IRF_HALF_WINDOW, meta)
        if chip is None:
            self.iface.messageBar().pushMessage(
                _tr("IRF Analysis"),
                _tr("Failed to read SLC chip from layer."),
                level=Qgis.Critical,
                duration=4,
            )
            self._activate_irf(False)
            return

        peak_row, peak_col = np.unravel_index(np.argmax(np.abs(chip) ** 2), chip.shape)

        spacing_rng = meta.sar_pixel_spacing_range or 1.0
        spacing_az = meta.sar_pixel_spacing_azimuth or 1.0

        try:
            result = analyze_point(
                chip,
                int(peak_row),
                int(peak_col),
                spacing_rng,
                spacing_az,
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"IRF analysis failed: {e}",
                "ICEYE Toolbox",
                Qgis.Critical,
            )
            self.iface.messageBar().pushMessage(
                _tr("IRF Analysis"),
                _tr(f"Analysis failed: {e}"),
                level=Qgis.Critical,
                duration=5,
            )
            self._activate_irf(False)
            return

        if self._irf_dialog is not None:
            try:
                self._irf_dialog.close()
            except Exception:
                pass

        self._irf_dialog = IRFResultDialog(result, parent=self.iface.mainWindow())
        self._irf_dialog.show()
        self._irf_dialog.raise_()

        self._activate_irf(False)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def unload(self) -> None:
        """Disconnect signals and clean up."""
        canvas = self.iface.mapCanvas()
        try:
            canvas.extentsChanged.disconnect(self._recompute_ruler_lengths)
        except Exception:
            pass
        try:
            self.iface.currentLayerChanged.disconnect(self._on_layer_changed)
        except Exception:
            pass
        if self._irf_tool is not None:
            try:
                canvas.unsetMapTool(self._irf_tool)
            except Exception:
                pass
            self._irf_tool = None
        if self._irf_dialog is not None:
            try:
                self._irf_dialog.close()
            except Exception:
                pass
            self._irf_dialog = None
        super().unload()
