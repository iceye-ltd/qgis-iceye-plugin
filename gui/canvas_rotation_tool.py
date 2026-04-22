"""Canvas rotation and SAR mandala overlay tools."""

from __future__ import annotations

import math

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsMapLayer,
    QgsMessageLog,
    QgsProject,
    QgsRasterLayer,
)
from qgis.PyQt.QtCore import QObject, QPointF, Qt, pyqtSignal
from qgis.PyQt.QtGui import (
    QBrush,
    QColor,
    QFont,
    QIcon,
    QPainterPath,
    QPen,
    QPolygonF,
)
from qgis.PyQt.QtWidgets import (
    QAction,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsLineItem,
    QMenu,
    QToolButton,
)

from ..core.geometry import SARViewGeometry, get_geometry_from_metadata
from ..core.metadata import IceyeMetadata, MetadataProvider
from .sar_canvas_overlay import (
    MandalaInteractionFilter,
    MandalaPlacementTool,
    SAROverlayToolbarBase,
    _compute_sar_geometry_at_point,
    _map_point_to_scene,
    _to_wgs84,
    _tr,
)
from .toolbar_button_policy import ToolbarButtonPolicy


class CanvasRotationTool(QObject):
    """Canvas rotation tool for aligning SAR view with shadows/layover."""

    geometry_updated = pyqtSignal(SARViewGeometry, object)

    def __init__(self, iface, metadata_provider) -> None:
        """Initialise with a QGIS iface reference and a MetadataProvider."""
        super().__init__()
        self.iface = iface
        self.canvas = self.iface.mapCanvas()
        self.metadata_provider = metadata_provider
        self.geometry = None
        self._current_meta = None
        self._current_layer = None

    def toShadowsDown(self) -> None:
        """Rotate canvas so shadows point downward."""
        if not self.geometry:
            QgsMessageLog.logMessage(
                "No geometry found",
                "ICEYE Toolbox",
                Qgis.Warning,
            )
            return
        angle = self.geometry.shadow - 180 + self.geometry.north
        self.setRotation(angle)
        QgsMessageLog.logMessage(
            f"Canvas rotated to align shadows downward to {angle}°",
            "ICEYE Toolbox",
            Qgis.Info,
        )

    def toLayoverUp(self) -> None:
        """Rotate canvas so layover points upward."""
        if not self.geometry:
            QgsMessageLog.logMessage(
                "No geometry found",
                "ICEYE Toolbox",
                Qgis.Warning,
            )
            return
        angle = self.geometry.north - self.geometry.layover
        self.setRotation(angle)
        QgsMessageLog.logMessage(
            f"Canvas rotated to align layover upward to {angle}°",
            "ICEYE Toolbox",
            Qgis.Info,
        )

    def toNorthUp(self) -> None:
        """Reset canvas rotation to North up."""
        self.setRotation(0)
        QgsMessageLog.logMessage("Canvas reset to North up", "ICEYE Toolbox", Qgis.Info)

    def setRotation(self, angle: float) -> None:
        """Set canvas rotation angle in degrees."""
        self.canvas.setRotation(angle % 360)
        self.canvas.refresh()

    def layerChanged(self, layer: QgsRasterLayer) -> None:
        """Update the mandala when the layer changes."""
        meta = self.metadata_provider.get(layer)
        if meta is None:
            self._current_meta = None
            return

        self._current_layer = layer
        self._current_meta = meta
        self.geometry = get_geometry_from_metadata(meta)
        if self.geometry is not None:
            self.geometry.radian2deg()
            incidence_angle = self._incidence_angle_at_canvas_center(meta)
            self.geometry_updated.emit(self.geometry, incidence_angle)

    def _incidence_angle_at_canvas_center(self, meta: IceyeMetadata) -> float | None:
        """Compute incidence angle at canvas center, or fall back to metadata value."""
        try:
            center = self.canvas.center()
            map_crs = self.canvas.mapSettings().destinationCrs()

            if self._current_layer is not None:
                layer_crs = self._current_layer.crs()
                to_layer = QgsCoordinateTransform(
                    map_crs, layer_crs, QgsProject.instance()
                )
                center_in_layer = to_layer.transform(center)
                if not self._current_layer.extent().contains(center_in_layer):
                    return meta.view_incidence_angle

            wgs84_pt = _to_wgs84(center, map_crs)
            geom = _compute_sar_geometry_at_point(wgs84_pt.y(), wgs84_pt.x(), meta)
            return 90.0 - math.degrees(geom.graze)
        except Exception:
            return meta.view_incidence_angle

    def _on_extents_changed(self) -> None:
        """Re-emit geometry with updated incidence angle when canvas is panned/zoomed."""
        if self.geometry is None or self._current_meta is None:
            return
        incidence_angle = self._incidence_angle_at_canvas_center(self._current_meta)
        self.geometry_updated.emit(self.geometry, incidence_angle)


class MandalaItem(QGraphicsItemGroup):
    """Mandala graphics item group for SAR view overlay."""

    def __init__(
        self,
        parent=None,
        pos: QPointF = QPointF(100, 100),
        movable: bool = True,
    ):
        """Create a mandala graphics item at pos. Set movable=True for placed instances."""
        super().__init__(parent)
        self.setPos(pos)
        self.setAcceptHoverEvents(True)

        if movable:
            self.setFlag(QGraphicsItem.ItemIsMovable, True)
            self.setFlag(QGraphicsItem.ItemIsSelectable, True)

        self.diameter = 100
        self.ring_width = 30
        self.ring = QGraphicsEllipseItem(
            -self.diameter / 2, -self.diameter / 2, self.diameter, self.diameter, self
        )
        self.ring.setPen(QPen(QColor("whitesmoke"), self.ring_width, Qt.SolidLine))
        self.ring.setOpacity(0.3)
        self.addToGroup(self.ring)

        inner_disc_d = self.diameter - self.ring_width
        self.inner_disc = QGraphicsEllipseItem(
            -inner_disc_d / 2, -inner_disc_d / 2, inner_disc_d, inner_disc_d, self
        )
        self.inner_disc.setPen(QPen(Qt.NoPen))
        self.inner_disc.setBrush(QBrush(QColor("black")))
        self.inner_disc.setOpacity(0.3)
        self.addToGroup(self.inner_disc)

        # Direction lines — fixed length, always readable regardless of scale
        line_tip = self.diameter / 2
        self.layover_line = QGraphicsLineItem(0, 0, line_tip, 0, self.ring)
        self.layover_line.setPen(QPen(QColor("plum"), 3, Qt.SolidLine))
        self.shadow_line = QGraphicsLineItem(0, 0, line_tip, 0, self.ring)
        self.shadow_line.setPen(QPen(QColor("sandybrown"), 3, Qt.SolidLine))
        self.addToGroup(self.layover_line)
        self.addToGroup(self.shadow_line)

        # Outer arrows
        arrow_height = self.ring_width
        arrow_tip = (self.diameter / 2) + self.ring_width / 2
        self.north_arrow = ArrowItem(
            length=arrow_tip,
            color=QColor("steelblue"),
            height=arrow_height,
            parent=self.ring,
            label="N",
        )
        self.shadows_arrow = ArrowItem(
            length=arrow_tip,
            color=QColor("sandybrown"),
            height=arrow_height,
            parent=self.ring,
            label="S",
        )
        self.track_arrow = ArrowItem(
            length=arrow_tip,
            color=QColor("olivedrab"),
            height=arrow_height,
            parent=self.ring,
            label="T",
        )
        self.layover_arrow = ArrowItem(
            length=arrow_tip,
            color=QColor("plum"),
            height=arrow_height,
            parent=self.ring,
            label="L",
        )
        self.addToGroup(self.shadows_arrow)
        self.addToGroup(self.north_arrow)
        self.addToGroup(self.track_arrow)
        self.addToGroup(self.layover_arrow)

        self.layover_arrow.setVisible(True)
        self.shadows_arrow.setVisible(True)

        halo_d = self.diameter + self.ring_width + 10
        _halo_pen = QPen(QColor(255, 215, 0, 140), 6, Qt.SolidLine)
        self._halo = QGraphicsEllipseItem(
            -halo_d / 2, -halo_d / 2, halo_d, halo_d, self
        )
        self._halo.setPen(_halo_pen)
        self._halo.setBrush(QBrush(Qt.NoBrush))
        self._halo.setVisible(False)
        self.addToGroup(self._halo)

        self.resetGeometry()

    def onGeometryUpdated(
        self, geometry: SARViewGeometry, incidence_angle: float
    ) -> None:
        """Update mandala arrows when geometry changes."""
        if not geometry:
            return
        self.updateGeometry(geometry, incidence_angle)

    def updateGeometry(self, geometry: SARViewGeometry, incidence_angle: float) -> None:
        """Apply geometry angles to mandala arrows."""
        # Outer arrows
        rot_layover = geometry.layover - geometry.north - 90
        rot_shadow = geometry.shadow - geometry.north - 90

        self.north_arrow.setRotation(-90.0)
        self.track_arrow.setRotation((180 + geometry.track + geometry.north - 90) % 360)
        self.layover_arrow.setRotation(rot_layover)
        self.shadows_arrow.setRotation(rot_shadow)
        self.layover_line.setRotation(rot_layover)
        self.shadow_line.setRotation(rot_shadow)

        min_len = 8.0
        max_len = (self.diameter - self.ring_width) / 2

        theta = math.radians(incidence_angle)
        layover_frac = math.cos(theta) ** 2
        shadow_frac = math.sin(theta) ** 2

        available = max_len - min_len
        layover_len = min_len + layover_frac * available
        shadow_len = min_len + shadow_frac * available

        self.layover_line.setLine(0, 0, layover_len, 0)
        self.shadow_line.setLine(0, 0, shadow_len, 0)

    def resetGeometry(self) -> None:
        """Reset mandala arrows to default positions."""
        self.north_arrow.setRotation(-90.0)
        self.track_arrow.setRotation(0.0)
        self.layover_arrow.setRotation(180)
        self.shadows_arrow.setRotation(90)
        self.layover_line.setRotation(180)
        self.shadow_line.setRotation(90)
        self.layover_line.setLine(0, 0, self.diameter / 2, 0)
        self.shadow_line.setLine(0, 0, self.diameter / 2, 0)

    def toggle(self) -> None:
        """Show or hide the mandala overlay."""
        if self.isVisible():
            self.hide()
        else:
            self.show()

    def set_highlighted(self, active: bool) -> None:
        """Show or hide the halo to indicate this mandala is under the cursor."""
        self._halo.setVisible(active)

    def hoverEnterEvent(self, event) -> None:
        """Highlight the mandala when the cursor enters its area."""
        self.set_highlighted(True)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:
        """Remove highlight when the cursor leaves the mandala area."""
        self.set_highlighted(False)
        super().hoverLeaveEvent(event)


class ArrowItem(QGraphicsItem):
    """Graphics item for a single arrow (N, S, T, or L) in the mandala."""

    def __init__(
        self,
        length: float = 50,
        color: QColor | None = None,
        height: float = 10,
        label: str | None = None,
        parent: QGraphicsItem | None = None,
    ) -> None:
        """Create an arrow of given length and color, with an optional text label."""
        super().__init__(parent)
        self.length = length
        self.color = color if color is not None else QColor(0, 0, 0)
        self.height = height
        self.label = label

    def boundingRect(self):
        """Return bounding rect for the arrow."""
        return self.shape().boundingRect()

    def setLength(self, length: float) -> None:
        """Set the length of the arrow."""
        self.prepareGeometryChange()
        self.length = length
        self.update()

    def shape(self):
        """Return arrow shape as polygon."""
        return QPolygonF([QPointF(0, 0), QPointF(self.length, 0)])

    def paint(self, painter, option, widget):
        """Paint the arrow."""
        half_side = self.height / 0.866 / 2 / 1.5
        path = QPainterPath()
        path.moveTo(self.length, 0)
        path.lineTo(self.length - self.height, half_side)
        path.lineTo(self.length - self.height, -half_side)
        path.closeSubpath()
        painter.fillPath(path, self.color)
        if self.label:
            painter.setPen(QPen(Qt.white, 1, Qt.SolidLine))
            painter.setFont(QFont("Arial", 8, QFont.Bold))

            # Save the current transform
            painter.save()

            # Calculate the center of the arrow head
            arrow_head_center_x = self.length - self.height / 2
            arrow_head_center_y = 0

            # Move to the center of the arrow head
            painter.translate(arrow_head_center_x, arrow_head_center_y)

            # Rotate the text to point towards the tip (90 degrees counterclockwise)
            painter.rotate(90)

            # Draw the text centered at the origin (which is now at the arrow head center)
            font_metrics = painter.fontMetrics()
            text_width = font_metrics.horizontalAdvance(self.label)
            text_height = font_metrics.height()
            painter.drawText(QPointF(-text_width / 2, text_height / 2), self.label)

            # Restore the transform
            painter.restore()


class MandalaToolbarAction(SAROverlayToolbarBase):
    """Encapsulates the SAR view toolbar, canvas rotation tool, and mandala."""

    def __init__(
        self,
        iface,
        metadata_provider: MetadataProvider | None = None,
        toolbar_button_policy: ToolbarButtonPolicy | None = None,
    ) -> None:
        """Initialise with a QGIS iface reference and an optional MetadataProvider."""
        super().__init__(iface, metadata_provider)
        self._toolbar_button_policy = toolbar_button_policy
        self.canvas_rotation_tool = CanvasRotationTool(
            self.iface, metadata_provider=self.metadata_provider
        )
        self.mandala = MandalaItem()

    def setup(self) -> None:
        """Create the toolbar, actions, add mandala to scene, connect signals."""
        self.toolbar = self.iface.addToolBar("ICEYE SAR View")
        self.toolbar.setObjectName("ICEYE SAR View")

        shadows_action = QAction(
            QIcon(":/plugins/iceye_toolbox/shadows_down_label.svg"),
            _tr("Rotate Shadows Down"),
            self.iface.mainWindow(),
        )
        shadows_action.setStatusTip("Rotate Shadows Down")
        shadows_action.triggered.connect(self.canvas_rotation_tool.toShadowsDown)
        self.toolbar.addAction(shadows_action)

        north_action = QAction(
            QIcon(":/plugins/iceye_toolbox/north_up_label.svg"),
            _tr("Reset to North"),
            self.iface.mainWindow(),
        )
        north_action.setStatusTip("Reset to North")
        north_action.triggered.connect(self.canvas_rotation_tool.toNorthUp)
        self.toolbar.addAction(north_action)

        layover_action = QAction(
            QIcon(":/plugins/iceye_toolbox/layover_up_label.svg"),
            _tr("Rotate Layover Up"),
            self.iface.mainWindow(),
        )
        layover_action.setStatusTip("Rotate Layover Up")
        layover_action.triggered.connect(self.canvas_rotation_tool.toLayoverUp)
        self.toolbar.addAction(layover_action)

        mandala_action = QAction(
            QIcon(":/plugins/iceye_toolbox/sar_angles.png"),
            _tr("Toggle SAR Mandala"),
            self.iface.mainWindow(),
        )
        mandala_action.setStatusTip("Toggle SAR Mandala")
        mandala_action.triggered.connect(self.mandala.toggle)
        self.toolbar.addAction(mandala_action)

        place_menu = QMenu(self.toolbar)
        clear_action = place_menu.addAction(_tr("Clear placed mandalas"))
        clear_action.triggered.connect(self._clear_items)
        place_menu.aboutToShow.connect(
            lambda: clear_action.setEnabled(len(self._placed_items) > 0)
        )

        self._place_btn = QToolButton(self.toolbar)
        self._place_btn.setIcon(QIcon(":/plugins/iceye_toolbox/sar_angles.png"))
        self._place_btn.setToolTip(_tr("Place / Move SAR Mandala"))
        self._place_btn.setCheckable(True)
        self._place_btn.setMenu(place_menu)
        self._place_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._place_btn.clicked.connect(self._activate_tool)
        self.toolbar.addWidget(self._place_btn)

        self.canvas_rotation_tool.geometry_updated.connect(
            self.mandala.onGeometryUpdated
        )
        self.iface.mapCanvas().rotationChanged.connect(self.mandala.setRotation)
        self.iface.mapCanvas().scene().addItem(self.mandala)
        self.mandala.hide()

        self._placement_tool = MandalaPlacementTool(
            self.iface.mapCanvas(), self._placed_items
        )
        self._placement_tool.setButton(self._place_btn)
        self._placement_tool.placed.connect(self._on_placement_click)
        self._placement_tool.removed.connect(self._remove_item)
        self._placement_tool.moved.connect(self._on_item_moved)
        self._placement_tool.deactivated.connect(self._on_placement_deactivated)

        canvas = self.iface.mapCanvas()
        self._interaction_filter = MandalaInteractionFilter(
            canvas,
            self._placed_items,
            on_moved=self._on_item_moved,
            on_removed=self._remove_item,
        )
        canvas.viewport().installEventFilter(self._interaction_filter)

        canvas.extentsChanged.connect(self._reposition_items)

        self.iface.currentLayerChanged.connect(self.canvas_rotation_tool.layerChanged)
        self.iface.currentLayerChanged.connect(self._on_layer_changed)

        canvas.extentsChanged.connect(self.canvas_rotation_tool._on_extents_changed)

        if self._toolbar_button_policy is not None:
            for control in (
                shadows_action,
                north_action,
                layover_action,
                mandala_action,
                self._place_btn,
            ):
                on_disable = None
                if control is self._place_btn:
                    on_disable = self._on_place_tool_policy_disabled
                self._toolbar_button_policy.register(
                    control,
                    self._policy_sar_view_iceye_layer,
                    on_disable=on_disable,
                )
            self._toolbar_button_policy.refresh()

    def _policy_sar_view_iceye_layer(self, layer: QgsMapLayer | None) -> bool:
        """SAR view controls: any ICEYE raster with metadata."""
        if self._toolbar_button_policy is None:
            return True
        return self._toolbar_button_policy.enabled_if_iceye_layer(layer)

    def _on_place_tool_policy_disabled(self) -> None:
        if self._place_btn is not None and self._place_btn.isChecked():
            self._place_btn.setChecked(False)
            self._activate_tool(False)

    def _on_placement_click(self, map_point) -> None:
        """Create a movable MandalaItem at the clicked map location.

        Recomputes SAR geometry for the clicked coordinate so that the
        shadow/layover line lengths reflect the local incidence angle at
        that position rather than the scene-average value.
        """
        meta = self.metadata_provider.get(self.iface.activeLayer())
        if meta is None:
            QgsMessageLog.logMessage(
                "No active ICEYE layer — activate an ICEYE layer first",
                "ICEYE Toolbox",
                Qgis.Warning,
            )
            return

        canvas = self.iface.mapCanvas()
        local_geometry, local_incidence_angle = self._local_sar_geometry(map_point)

        if local_geometry is None:
            QgsMessageLog.logMessage(
                "Could not compute geometry for the clicked location",
                "ICEYE Toolbox",
                Qgis.Warning,
            )
            return

        scene_pt = _map_point_to_scene(canvas, map_point)
        layer = self.iface.activeLayer()

        mandala = MandalaItem(pos=scene_pt, movable=True)
        mandala._geo_point = map_point
        mandala._layer_id = layer.id() if layer else None
        mandala.updateGeometry(local_geometry, local_incidence_angle)
        mandala.setRotation(canvas.rotation())
        canvas.rotationChanged.connect(mandala.setRotation)
        canvas.scene().addItem(mandala)
        self._placed_items.append(mandala)

        self._activate_tool(False)

    def _on_layer_changed(self, layer) -> None:
        """Re-orient placed mandalas for *layer* when it becomes active."""
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
        for mandala in self._placed_items:
            if mandala._layer_id == layer_id:
                mandala.updateGeometry(geometry, incidence_angle)

    def unload(self) -> None:
        """Disconnect signals, remove mandala from scene, delete toolbar."""
        canvas = self.iface.mapCanvas()
        try:
            self.iface.currentLayerChanged.disconnect(self._on_layer_changed)
        except Exception:
            pass
        try:
            self.iface.currentLayerChanged.disconnect(
                self.canvas_rotation_tool.layerChanged
            )
        except Exception:
            pass

        try:
            self.canvas_rotation_tool.geometry_updated.disconnect(
                self.mandala.onGeometryUpdated
            )
        except Exception:
            pass

        try:
            canvas.rotationChanged.disconnect(self.mandala.setRotation)
        except Exception:
            pass

        try:
            canvas.extentsChanged.disconnect(
                self.canvas_rotation_tool._on_extents_changed
            )
        except Exception:
            pass

        if self.mandala is not None:
            try:
                canvas.scene().removeItem(self.mandala)
            except Exception:
                pass

        super().unload()
