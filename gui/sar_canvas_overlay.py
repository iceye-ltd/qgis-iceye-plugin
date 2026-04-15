"""SAR canvas overlay: placement tools, viewport interaction filters, toolbar base, and helpers."""

from __future__ import annotations

import numpy as np
from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
)
from qgis.gui import QgsMapTool
from qgis.PyQt.QtCore import QCoreApplication, QEvent, QObject, QPointF, Qt, pyqtSignal
from qgis.PyQt.QtWidgets import QMenu

from ..core.geometry import (
    SARViewGeometry,
    geodetic_to_ecef,
    get_geometry_from_metadata,
    sar_vectors_to_geometry,
)
from ..core.metadata import IceyeMetadata, MetadataProvider


def _tr(message: str) -> str:
    """Translate message for ICEYE Toolbox context."""
    return QCoreApplication.translate("ICEYE Toolbox", message)


def _compute_sar_geometry_at_point(
    lat: float, lon: float, meta: IceyeMetadata
) -> SARViewGeometry:
    """Compute SAR geometry (in radians) at a WGS84 lat/lon using metadata orbit state."""
    aim_ecef = np.array(
        geodetic_to_ecef(lat, lon, meta.iceye_average_scene_height or 0.0)
    )
    return sar_vectors_to_geometry(
        AIM=aim_ecef,
        P=meta.center_aperture_position,
        VEL=meta.center_aperture_velocity,
    )


def sar_geometry_and_incidence(
    lat: float, lon: float, meta: IceyeMetadata
) -> tuple[SARViewGeometry, float] | tuple[None, None]:
    """Return (SARViewGeometry in degrees, incidence_angle) for a WGS84 point.

    Uses per-pixel orbit computation when aperture state vectors are available,
    otherwise falls back to scene-average geometry from metadata.
    Returns (None, None) if geometry cannot be determined.
    """
    if (
        meta.center_aperture_position is not None
        and meta.center_aperture_velocity is not None
    ):
        geometry = _compute_sar_geometry_at_point(lat, lon, meta)
        geometry.radian2deg()
        return geometry, 90.0 - geometry.graze
    geometry = get_geometry_from_metadata(meta)
    if geometry is not None:
        geometry.radian2deg()
    return geometry, meta.view_incidence_angle or 45.0


def _to_wgs84(point, src_crs) -> object:
    """Transform a map point to WGS84 (EPSG:4326), returning a QgsPointXY."""
    wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    if src_crs == wgs84:
        return point
    return QgsCoordinateTransform(src_crs, wgs84, QgsProject.instance()).transform(
        point
    )


def _map_point_to_scene(canvas, map_point) -> QPointF:
    """Convert a map CRS point to QGraphicsScene coordinates."""
    canvas_pt = canvas.getCoordinateTransform().transform(map_point)
    view = canvas.scene().views()[0]
    return view.mapToScene(int(canvas_pt.x()), int(canvas_pt.y()))


class SARItemPlacementTool(QgsMapTool):
    """Persistent map tool: drag, place, and right-click-remove any SAR overlay item.

    Subclass and override _remove_label to customise the context menu text.
    """

    placed = pyqtSignal(object)  # QgsPointXY
    removed = pyqtSignal(object)  # item
    moved = pyqtSignal(object, object)  # item, QgsPointXY

    _HIT_RADIUS = 60.0
    _remove_label = "Remove item"  # override in subclass

    def __init__(self, canvas, placed_items: list) -> None:
        """Initialise the tool with a shared list of already-placed items.

        Args:
            canvas: The QGIS map canvas this tool operates on.
            placed_items: Mutable list shared with the toolbar action; items
                are appended on placement and removed on right-click removal.
        """
        super().__init__(canvas)
        self._placed_items = placed_items
        self.dragging = None
        self._press_on_item: bool = False

    def _hit_item(self, map_point):
        """Return the first item within hit radius of map_point, or None."""
        scene_pt = _map_point_to_scene(self.canvas(), map_point)
        r2 = self._HIT_RADIUS**2
        for item in self._placed_items:
            dx = scene_pt.x() - item.pos().x()
            dy = scene_pt.y() - item.pos().y()
            if dx * dx + dy * dy <= r2:
                return item
        return None

    def canvasPressEvent(self, event) -> None:
        """Begin dragging when the left button is pressed on a placed item."""
        if event.button() != Qt.LeftButton:
            return
        hit = self._hit_item(event.mapPoint())
        if hit is not None:
            self.dragging = hit
            self._press_on_item = True
        else:
            self._press_on_item = False

    def canvasMoveEvent(self, event) -> None:
        """Track cursor position and reposition the item being dragged."""
        if self.dragging is None:
            return
        self.dragging.setPos(_map_point_to_scene(self.canvas(), event.mapPoint()))
        self.dragging._geo_point = event.mapPoint()

    def canvasReleaseEvent(self, event) -> None:
        """Finish drag, emit placed/moved, or show removal context menu on right-click."""
        if event.button() == Qt.RightButton:
            hit = self._hit_item(event.mapPoint())
            if hit is not None:
                menu = QMenu(self.canvas())
                remove_action = menu.addAction(_tr(self._remove_label))
                if menu.exec_(self.canvas().mapToGlobal(event.pos())) is remove_action:
                    self.removed.emit(hit)
            return
        if event.button() != Qt.LeftButton:
            return
        if self.dragging is not None:
            dropped = self.dragging
            self.dragging = None
            self.moved.emit(dropped, dropped._geo_point)
            return
        if not self._press_on_item:
            self.placed.emit(event.mapPoint())

    def deactivate(self) -> None:
        """Cancel any in-progress drag and reset press state."""
        self.dragging = None
        self._press_on_item = False
        super().deactivate()


class MandalaPlacementTool(SARItemPlacementTool):
    """Placement tool for SAR mandalas."""

    _remove_label = "Remove mandala"


class BaseInteractionFilter(QObject):
    """Base class for interaction filters."""

    _HIT_RADIUS = 50.0
    _remove_label = "Remove item"

    def __init__(self, canvas, placed_items: list, on_moved, on_removed) -> None:
        """Initialise filter."""
        super().__init__()
        self._canvas = canvas
        self._placed_items = placed_items
        self._on_moved = on_moved
        self._on_removed = on_removed
        self._dragging = None
        self._highlighted: object | None = None
        self._ignored_tools: list = []

    def _viewport_to_map(self, pos):
        """Convert a  QPoint in viewport coords to a map CRS QgsPointXY."""
        return self._canvas.getCoordinateTransform().toMapCoordinates(pos.x(), pos.y())

    def _hit_item(self, map_point):
        """Return the first Item within hit radius of map_point or None."""
        scene_pt = _map_point_to_scene(self._canvas, map_point)
        r2 = self._HIT_RADIUS**2
        for item in self._placed_items:
            dx = scene_pt.x() - item.pos().x()
            dy = scene_pt.y() - item.pos().y()
            if dx * dx + dy * dy <= r2:
                return item
        return None

    def eventFilter(self, obj, event) -> bool:
        """Intercept mouse events to handle item dragging."""
        if any(self._canvas.mapTool() is t for t in self._ignored_tools):
            return False

        etype = event.type()

        if etype == QEvent.MouseButtonPress:
            return self._handle_press(event)
        if etype == QEvent.MouseMove:
            return self._handle_move(event)
        if etype == QEvent.MouseButtonRelease:
            return self._handle_release(event)
        if etype == QEvent.Wheel:
            return self._handle_wheel(event)

        return False

    def _handle_press(self, event):
        if event.button() == Qt.LeftButton:
            map_point = self._viewport_to_map(event.pos())
            hit = self._hit_item(map_point)
            if hit is not None:
                self._dragging = hit
                return True

        elif event.button() == Qt.RightButton:
            map_point = self._viewport_to_map(event.pos())
            hit = self._hit_item(map_point)
            if hit is not None:
                return True
        return False

    def _handle_move(self, event) -> bool:
        map_point = self._viewport_to_map(event.pos())

        hit = self._hit_item(map_point)
        if hit is not self._highlighted:
            if self._highlighted is not None:
                self._highlighted.set_highlighted(False)
            self._highlighted = hit
            if hit is not None:
                hit.set_highlighted(True)

        if self._dragging is None:
            return False

        self._dragging.setPos(_map_point_to_scene(self._canvas, map_point))
        self._dragging._geo_point = map_point
        return True

    def _handle_release(self, event):
        if event.button() == Qt.RightButton:
            map_point = self._viewport_to_map(event.pos())
            hit = self._hit_item(map_point)
            if hit is not None:
                menu = QMenu()
                remove_action = menu.addAction(_tr(self._remove_label))
                if (
                    menu.exec_(self._canvas.viewport().mapToGlobal(event.pos()))
                    is remove_action
                ):
                    self._on_removed(hit)
                return True
            return False

        if event.button() == Qt.LeftButton and self._dragging is not None:
            dropped = self._dragging
            self._dragging = None
            self._on_moved(dropped, dropped._geo_point)
            return True

        return False

    def _handle_wheel(self, event) -> bool:
        """No-op wheel handler; override in subclasses for scroll interaction."""
        return False


class MandalaInteractionFilter(BaseInteractionFilter):
    """Canvas viewport event filter for always-on drag and remove of MandalaItems."""

    _HIT_RADIUS = 60.0
    _remove_label = "Remove mandala"


class SAROverlayToolbarBase:
    """Base class for SAR overlay toolbar actions managing placed graphics items.

    Provides shared item lifecycle: placement activation, drag-to-move,
    right-click-remove, clear-all, geo-anchor repositioning on pan/zoom,
    and SAR geometry lookup via MetadataProvider.

    Subclasses must implement _on_placement_click(map_point) to create and
    append the concrete item type to self._placed_items, and must call
    super().__init__ and super().unload().
    """

    def __init__(
        self, iface, metadata_provider: MetadataProvider | None = None
    ) -> None:
        """Initialise shared state for placed items and toolbar references.

        Args:
            iface: QGIS interface handle.
            metadata_provider: Provider for ICEYE layer metadata; a default
                instance is created when not supplied.
        """
        self.iface = iface
        self.metadata_provider = metadata_provider or MetadataProvider()
        self.toolbar = None
        self._place_btn = None
        self._placement_tool: SARItemPlacementTool | None = None
        self._placed_items: list = []
        self._interaction_filter: BaseInteractionFilter | None = None

    def _local_sar_geometry(self, map_point):
        """Return (SARViewGeometry in degrees, incidence_angle) for map_point.

        Returns (None, None) if no ICEYE metadata is available for the active layer.
        """
        meta = self.metadata_provider.get(self.iface.activeLayer())
        if meta is None:
            return None, None
        wgs84_pt = _to_wgs84(
            map_point, self.iface.mapCanvas().mapSettings().destinationCrs()
        )
        return sar_geometry_and_incidence(wgs84_pt.y(), wgs84_pt.x(), meta)

    def _activate_tool(self, checked: bool = True) -> None:
        """Activate or deactivate the placement tool."""
        if self._placement_tool is None:
            return
        canvas = self.iface.mapCanvas()
        if checked:
            canvas.setMapTool(self._placement_tool)
        else:
            canvas.unsetMapTool(self._placement_tool)
            self.iface.actionPan().trigger()

    def _on_placement_deactivated(self) -> None:
        """Uncheck the place button when the tool is deactivated."""
        if self._place_btn is not None:
            self._place_btn.setChecked(False)

    def _on_item_moved(self, item, map_point) -> None:
        """Recompute SAR geometry for an item after it has been dragged."""
        geometry, incidence_angle = self._local_sar_geometry(map_point)
        if geometry is None:
            return
        item.updateGeometry(geometry, incidence_angle)

    def _remove_item(self, item) -> None:
        """Remove a single placed item from the canvas scene."""
        if item not in self._placed_items:
            return
        canvas = self.iface.mapCanvas()
        try:
            canvas.rotationChanged.disconnect(item.setRotation)
        except Exception:
            pass
        canvas.scene().removeItem(item)
        self._placed_items.remove(item)

    def _clear_items(self) -> None:
        """Remove all placed items from the canvas scene."""
        canvas = self.iface.mapCanvas()
        for item in self._placed_items:
            try:
                canvas.rotationChanged.disconnect(item.setRotation)
            except Exception:
                pass
            canvas.scene().removeItem(item)
        self._placed_items.clear()

    def register_exclusive_tools(self, tools: list) -> None:
        """Suppress the interaction filter while any of *tools* is the active map tool."""
        if self._interaction_filter is not None:
            self._interaction_filter._ignored_tools = tools

    def _reposition_items(self) -> None:
        """Snap all placed items back to their geo anchor after pan/zoom."""
        canvas = self.iface.mapCanvas()
        dragging = self._placement_tool.dragging if self._placement_tool else None
        for item in self._placed_items:
            if item is dragging:
                continue
            item.setPos(_map_point_to_scene(canvas, item._geo_point))

    def unload(self) -> None:
        """Disconnect shared signals, clear all placed items, and delete toolbar."""
        canvas = self.iface.mapCanvas()
        if self._interaction_filter is not None:
            canvas.viewport().removeEventFilter(self._interaction_filter)
            self._interaction_filter = None
        try:
            canvas.extentsChanged.disconnect(self._reposition_items)
        except Exception:
            pass
        self._clear_items()
        if self.toolbar is not None:
            try:
                del self.toolbar
            except Exception:
                pass
            self.toolbar = None
