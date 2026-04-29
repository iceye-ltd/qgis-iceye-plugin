"""Batch toolbar: place ~300 m ground square mask areas, then run batch jobs on selected masks."""

from __future__ import annotations

from qgis.core import (
    Qgis,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsGeometry,
    QgsLayerTreeGroup,
    QgsLayerTreeLayer,
    QgsLayerTreeNode,
    QgsMapLayer,
    QgsMessageLog,
    QgsPointXY,
    QgsProject,
    QgsRasterLayer,
    QgsRectangle,
    QgsVectorLayer,
)
from qgis.gui import QgsMapTool
from qgis.PyQt.QtCore import QCoreApplication, Qt
from qgis.PyQt.QtGui import QCursor, QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu, QToolButton

from ..core.autofocus import AutofocusTool
from ..core.color import ColorTool
from ..core.cropper import CropTool, MaskLayerFactory
from ..core.metadata import MetadataProvider
from ..core.video import VideoTool

_TARGET_GROUND_M = 300.0


def _tr(message: str) -> str:
    return QCoreApplication.translate("ICEYE Toolbox", message)


def _is_derivative_processing_layer(layer: QgsRasterLayer) -> bool:
    """Return True if the layer name looks like a CROP/SHORT/FOCUS/COLOR output, not base imagery."""
    name = layer.name()
    return any(token in name for token in ("SHORT", "CROP", "FOCUS", "COLOR"))


def _layers_from_selected_tree_nodes(
    nodes: list[QgsLayerTreeNode],
) -> list[QgsMapLayer]:
    """Map layers implied by the layer-tree selection; selected groups include all descendant layers."""
    seen: set[str] = set()
    out: list[QgsMapLayer] = []

    def visit(node: QgsLayerTreeNode) -> None:
        if isinstance(node, QgsLayerTreeLayer):
            layer = node.layer()
            if layer is not None and layer.id() not in seen:
                seen.add(layer.id())
                out.append(layer)
        elif isinstance(node, QgsLayerTreeGroup):
            for child in node.children():
                visit(child)

    for n in nodes:
        visit(n)
    return out


def batch_mask_square_extent(
    map_point: QgsPointXY,
    destination_crs: QgsCoordinateReferenceSystem,
) -> QgsRectangle | None:
    """Return an axis-aligned square in *destination_crs* with fixed _TARGET_GROUND_M side (meters on ground).

    The square is built in a local UTM frame then transformed to *destination_crs*.
    """
    half = _TARGET_GROUND_M / 2.0

    wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
    to_wgs = QgsCoordinateTransform(destination_crs, wgs84, QgsProject.instance())
    try:
        ll = to_wgs.transform(map_point)
    except Exception:
        QgsMessageLog.logMessage(
            "Batch mask: could not transform click to WGS84",
            "ICEYE Toolbox",
            Qgis.Warning,
        )
        return None

    lon = ll.x()
    lat = ll.y()
    zone = int((lon + 180) / 6) + 1
    zone = min(max(zone, 1), 60)
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    utm = QgsCoordinateReferenceSystem(f"EPSG:{epsg}")

    to_utm = QgsCoordinateTransform(destination_crs, utm, QgsProject.instance())
    try:
        utm_pt = to_utm.transform(map_point)
    except Exception:
        QgsMessageLog.logMessage(
            "Batch mask: could not transform click to UTM",
            "ICEYE Toolbox",
            Qgis.Warning,
        )
        return None

    rect = QgsRectangle(
        utm_pt.x() - half,
        utm_pt.y() - half,
        utm_pt.x() + half,
        utm_pt.y() + half,
    )
    geom = QgsGeometry.fromRect(rect)
    to_canvas = QgsCoordinateTransform(utm, destination_crs, QgsProject.instance())
    try:
        geom.transform(to_canvas)
    except Exception:
        QgsMessageLog.logMessage(
            "Batch mask: could not transform square to map CRS",
            "ICEYE Toolbox",
            Qgis.Warning,
        )
        return None

    return geom.boundingBox()


class BatchMaskMapTool(QgsMapTool):
    """Persistent tool: each left click adds a mask polygon at the clicked location."""

    def __init__(self, canvas, batch_action: "BatchToolbarAction") -> None:
        super().__init__(canvas)
        self._batch = batch_action
        self.setCursor(QCursor(Qt.CrossCursor))

    def canvasReleaseEvent(self, event) -> None:
        """On left click, add a batch mask at the map location."""
        if event.button() != Qt.LeftButton:
            return
        self._batch.place_mask_at(event.mapPoint())


def _polygon_feature_extents_map_crs(
    vlayer: QgsVectorLayer, destination_crs: QgsCoordinateReferenceSystem
) -> list[QgsRectangle]:
    """Bounding boxes of each polygon feature in *destination_crs* (map CRS)."""
    if vlayer.geometryType() != Qgis.GeometryType.PolygonGeometry:
        return []
    project = QgsProject.instance()
    xform = QgsCoordinateTransform(vlayer.crs(), destination_crs, project)
    out: list[QgsRectangle] = []
    for feat in vlayer.getFeatures():
        geom = feat.geometry()
        if geom is None or geom.isEmpty():
            continue
        g = QgsGeometry(geom)
        try:
            g.transform(xform)
        except Exception:
            continue
        box = g.boundingBox()
        if box.isEmpty():
            continue
        out.append(box)
    return out


class BatchToolbarAction:
    """Toolbar for batch mask placement and batch runs on selected mask layers."""

    def __init__(
        self,
        iface,
        metadata_provider: MetadataProvider,
        crop_tool: CropTool,
        target_focus_tool: AutofocusTool | None = None,
        color_tool: ColorTool | None = None,
        video_tool: VideoTool | None = None,
    ) -> None:
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.metadata_provider = metadata_provider
        self._crop_tool = crop_tool
        self._target_focus_tool = target_focus_tool
        self._color_tool = color_tool
        self._video_tool = video_tool

        self.toolbar = None
        self.toggle_action: QAction | None = None
        self._process_tool_btn: QToolButton | None = None

        self.map_tool = BatchMaskMapTool(self.canvas, self)
        self.map_tool.deactivated.connect(self._on_map_tool_deactivated)

        self._mask_serial = 0
        self._mask_group_session = 0
        self._mask_group: QgsLayerTreeGroup | None = None
        self._suppress_toggle = False

    def setup(self) -> None:
        """Create the toolbar and actions, connect signals."""
        self.toolbar = self.iface.addToolBar("ICEYE Batch")
        self.toolbar.setObjectName("ICEYE Batch")

        self.toggle_action = QAction(
            QIcon(":/plugins/iceye_toolbox/batch_masks.svg"),
            _tr("Batch masks"),
            self.iface.mainWindow(),
        )
        self.toggle_action.setCheckable(True)
        self.toggle_action.setStatusTip(
            _tr("Click map to add ~300 m (ground) square mask areas.")
        )
        self.toggle_action.toggled.connect(self._toggle_tool)
        self.toolbar.addAction(self.toggle_action)

        self._process_tool_btn = QToolButton(self.toolbar)
        self._process_tool_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self._process_tool_btn.setToolTip(
            _tr(
                "Run on selected polygon mask layers. ICEYE source: active base raster, or "
                "exactly one base ICEYE raster included in the selection."
            )
        )

        process_menu = QMenu(self._process_tool_btn)
        tip_crop_each = _tr(
            "Crop each selected polygon mask sequentially (ICEYE source: active base layer "
            "or exactly one base ICEYE raster in the selection)."
        )
        icon_batch_process = QIcon(":/plugins/iceye_toolbox/batch_process.svg")
        icon_crop = QIcon(":/plugins/iceye_toolbox/crop-simple-svgrepo-com.svg")
        icon_color = QIcon(":/plugins/iceye_toolbox/rgb-svgrepo-com.svg")
        icon_focus = QIcon(
            ":/plugins/iceye_toolbox/focus-horizontal-round-round-840-svgrepo-com.svg"
        )
        icon_video = QIcon(":/plugins/iceye_toolbox/video-1-svgrepo-com.svg")

        action_run_batch = QAction(
            icon_batch_process, _tr("Run Batch Process"), self.iface.mainWindow()
        )
        action_run_batch.setStatusTip(tip_crop_each)
        action_crop = QAction(icon_crop, _tr("Crop"), self.iface.mainWindow())
        action_crop.setStatusTip(tip_crop_each)
        tip_color = _tr(
            "Color composite for each selected mask (ICEYE source: active base layer or "
            "exactly one base ICEYE raster in the selection)."
        )
        action_color = QAction(icon_color, _tr("Color"), self.iface.mainWindow())
        action_color.setStatusTip(tip_color)
        tip_focus = _tr(
            "Crop and focus each selected mask (ICEYE source: active base layer or exactly "
            "one base ICEYE raster in the selection)."
        )
        action_focus = QAction(icon_focus, _tr("Focus"), self.iface.mainWindow())
        action_focus.setStatusTip(tip_focus)
        tip_video = _tr(
            "Video for each selected mask (ICEYE source: active base layer or exactly one "
            "base ICEYE raster in the selection)."
        )
        action_video = QAction(icon_video, _tr("Video"), self.iface.mainWindow())
        action_video.setStatusTip(tip_video)

        process_menu.addAction(action_crop)
        process_menu.addAction(action_color)
        process_menu.addAction(action_focus)
        process_menu.addAction(action_video)

        self._process_tool_btn.setMenu(process_menu)
        self._process_tool_btn.setDefaultAction(action_run_batch)

        action_run_batch.triggered.connect(self._run_batch_crop)
        action_crop.triggered.connect(self._run_batch_crop)
        action_color.triggered.connect(self._run_batch_color)
        action_focus.triggered.connect(self._run_batch_focus)
        action_video.triggered.connect(self._run_batch_video)

        self.toolbar.addWidget(self._process_tool_btn)

    def _jobs_from_selected_masks(
        self,
    ) -> tuple[list[tuple[QgsRectangle, str]] | None, str | None]:
        """Build (extent, source_id) jobs from layer-tree selection.

        Returns (jobs, None) on success, or (None, translated error message).
        Source raster is the **active** base ICEYE layer if that is valid; otherwise the
        **single** base ICEYE raster included in the selection.
        Masks are **selected** polygon layers or layers inside **selected**
        groups (whole group contents are used).
        """
        tree = self.iface.layerTreeView()
        if tree is None:
            return None, _tr("Layer tree is not available.")

        selected = _layers_from_selected_tree_nodes(tree.selectedNodes())
        if not selected:
            selected = tree.selectedLayers()
        if not selected:
            return None, _tr(
                "Select one or more polygon mask layers (or a group of masks) in the "
                "Layers panel, then run batch."
            )

        n_rasters = sum(
            1
            for layer in selected
            if isinstance(layer, QgsRasterLayer) and layer.isValid()
        )
        if n_rasters > 1:
            return None, _tr(
                "Select only one image layer in the Layers panel, then run batch."
            )

        source = self._resolve_batch_source_for_jobs(selected)
        if source is None:
            return None, _tr(
                "Select a base ICEYE raster as the active layer, or include exactly one "
                "base ICEYE raster in the selection with your mask layers, then run batch "
                "again."
            )

        dest = self.canvas.mapSettings().destinationCrs()
        source_id = source.id()
        jobs: list[tuple[QgsRectangle, str]] = []

        for layer in selected:
            if layer.id() == source_id:
                continue
            if not isinstance(layer, QgsVectorLayer) or not layer.isValid():
                continue
            for rect in _polygon_feature_extents_map_crs(layer, dest):
                jobs.append((rect, source_id))

        if not jobs:
            return None, _tr(
                "No polygon masks in the selection. Select vector polygon layer(s) or a "
                "group containing them (mask areas), with the base ICEYE raster as the active "
                "layer or included once in the selection, and run again."
            )

        return jobs, None

    def _is_batch_source_raster_layer(self, layer: QgsMapLayer | None) -> bool:
        """Return True if *layer* is a valid base (non-derived) ICEYE raster for batch processing."""
        return (
            layer is not None
            and isinstance(layer, QgsRasterLayer)
            and self.metadata_provider.get(layer)
            and not _is_derivative_processing_layer(layer)
        )

    def _resolve_batch_source_for_jobs(
        self, selected: list[QgsMapLayer]
    ) -> QgsRasterLayer | None:
        """Prefer active layer if it is a valid base ICEYE raster; else the single such raster in *selected*."""
        active = self.iface.activeLayer()
        if self._is_batch_source_raster_layer(active):
            return active
        candidates = [
            layer for layer in selected if self._is_batch_source_raster_layer(layer)
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None

    def place_mask_at(self, map_point: QgsPointXY) -> None:
        """Add a ~300 m mask extent at *map_point* and record it for batch processing."""
        dest = self.canvas.mapSettings().destinationCrs()
        extent = batch_mask_square_extent(map_point, dest)
        if extent is None:
            self.iface.messageBar().pushMessage(
                _tr("ICEYE Toolbox"),
                _tr("Could not build mask extent at this location."),
                level=Qgis.Warning,
                duration=4,
            )
            return

        self._mask_serial += 1
        name = _tr("Batch mask {n}").format(n=self._mask_serial)
        MaskLayerFactory(self.canvas).create(
            extent, layer_name=name, parent_group=self._mask_group
        )

        self.canvas.setMapTool(self.map_tool)

        self.iface.messageBar().pushMessage(
            _tr("ICEYE Toolbox"),
            _tr("Added {name}. Select mask layer(s) and run batch when ready.").format(
                name=name
            ),
            level=Qgis.Info,
            duration=3,
        )

    def _toggle_tool(self, enabled: bool) -> None:
        if self._suppress_toggle:
            return
        if enabled:
            self._mask_group_session += 1
            root = QgsProject.instance().layerTreeRoot()
            self._mask_group = root.insertGroup(
                0, _tr("Mask group {n}").format(n=self._mask_group_session)
            )
            self.canvas.setMapTool(self.map_tool)
            return

        self._mask_group = None

        if self.canvas.mapTool() is self.map_tool:
            self.canvas.unsetMapTool(self.map_tool)
            self.iface.actionPan().trigger()

    def _on_map_tool_deactivated(self) -> None:
        if self.toggle_action is not None and self.toggle_action.isChecked():
            self._suppress_toggle = True
            self.toggle_action.blockSignals(True)
            self.toggle_action.setChecked(False)
            self.toggle_action.blockSignals(False)
            self._suppress_toggle = False

    def _switch_to_pan_after_batch_start(self) -> None:
        if self.canvas.mapTool() is self.map_tool:
            self.canvas.unsetMapTool(self.map_tool)
        self.iface.actionPan().trigger()

    def _run_batch_crop(self) -> None:
        jobs, err = self._jobs_from_selected_masks()
        if jobs is None:
            self.iface.messageBar().pushMessage(
                _tr("ICEYE Toolbox"),
                err or _tr("Cannot run batch."),
                level=Qgis.Warning,
                duration=6,
            )
            return
        self._crop_tool.process_extents_batch(jobs)
        self._switch_to_pan_after_batch_start()

    def _run_batch_color(self) -> None:
        if self._color_tool is None:
            self.iface.messageBar().pushMessage(
                _tr("ICEYE Toolbox"),
                _tr("Batch color is not available."),
                level=Qgis.Warning,
                duration=4,
            )
            return
        jobs, err = self._jobs_from_selected_masks()
        if jobs is None:
            self.iface.messageBar().pushMessage(
                _tr("ICEYE Toolbox"),
                err or _tr("Cannot run batch."),
                level=Qgis.Warning,
                duration=6,
            )
            return
        self._color_tool.process_extents_batch(jobs)
        self._switch_to_pan_after_batch_start()

    def _run_batch_focus(self) -> None:
        if self._target_focus_tool is None:
            self.iface.messageBar().pushMessage(
                _tr("ICEYE Toolbox"),
                _tr("Batch focus is not available."),
                level=Qgis.Warning,
                duration=4,
            )
            return
        jobs, err = self._jobs_from_selected_masks()
        if jobs is None:
            self.iface.messageBar().pushMessage(
                _tr("ICEYE Toolbox"),
                err or _tr("Cannot run batch."),
                level=Qgis.Warning,
                duration=6,
            )
            return
        self._target_focus_tool.process_extents_batch(jobs)
        self._switch_to_pan_after_batch_start()

    def _run_batch_video(self) -> None:
        if self._video_tool is None:
            self.iface.messageBar().pushMessage(
                _tr("ICEYE Toolbox"),
                _tr("Batch video is not available."),
                level=Qgis.Warning,
                duration=4,
            )
            return
        jobs, err = self._jobs_from_selected_masks()
        if jobs is None:
            self.iface.messageBar().pushMessage(
                _tr("ICEYE Toolbox"),
                err or _tr("Cannot run batch."),
                level=Qgis.Warning,
                duration=6,
            )
            return
        self._video_tool.process_extents_batch(jobs)
        self._switch_to_pan_after_batch_start()

    def deactivate(self) -> None:
        """Uncheck Batch masks (turn batch placement off)."""
        if self.toggle_action is not None and self.toggle_action.isChecked():
            self.toggle_action.setChecked(False)

    def unload(self) -> None:
        """Remove toolbar actions, unset the map tool, and drop the toolbar."""
        self._mask_group = None
        if self.toolbar is not None:
            if self.toggle_action is not None:
                try:
                    self.toolbar.removeAction(self.toggle_action)
                except Exception:
                    pass
            if self._process_tool_btn is not None:
                try:
                    self.toolbar.removeWidget(self._process_tool_btn)
                except Exception:
                    pass
        self.toggle_action = None
        self._process_tool_btn = None

        if self.canvas.mapTool() is self.map_tool:
            try:
                self.canvas.unsetMapTool(self.map_tool)
            except Exception:
                pass

        if self.toolbar is not None:
            try:
                del self.toolbar
            except Exception:
                pass
            self.toolbar = None
