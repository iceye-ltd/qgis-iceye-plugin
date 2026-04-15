"""Crop toolbar with toggle and crop mode menu (normal, focus, video, color)."""

from __future__ import annotations

from qgis.core import QgsMapLayer
from qgis.gui import QgsMapToolExtent
from qgis.PyQt.QtCore import QCoreApplication, QSize, Qt
from qgis.PyQt.QtGui import QColor, QIcon, QPainter, QPixmap
from qgis.PyQt.QtWidgets import QAction, QActionGroup, QMenu, QToolButton

from .toolbar_button_policy import ToolbarButtonPolicy

_CROP_ICON = ":/plugins/ICEYEToolbox/crop-simple-svgrepo-com.svg"
_FOCUS_ICON = ":/plugins/ICEYEToolbox/focus-horizontal-round-round-840-svgrepo-com.svg"
_VIDEO_ICON = ":/plugins/ICEYEToolbox/video-1-svgrepo-com.svg"
_COLOR_ICON = ":/plugins/ICEYEToolbox/rgb-svgrepo-com.svg"

# Mode key -> overlay resource (None = base crop only)
_MODE_OVERLAY = {
    "normal": None,
    "focus": _FOCUS_ICON,
    "video": _VIDEO_ICON,
    "color": _COLOR_ICON,
}

_ICON_PX = 24
_BADGE_PX = 10


def _tr(message: str) -> str:
    """Translate message for ICEYE Toolbox context."""
    return QCoreApplication.translate("ICEYE Toolbox", message)


class CropToolbarAction:
    """Encapsulates the crop toolbar, toggle button, and crop mode menu."""

    def __init__(
        self,
        iface,
        crop_tool,
        target_focus_tool,
        video_tool,
        color_tool,
        lens_toolbar_action,
        toolbar_button_policy: ToolbarButtonPolicy | None = None,
    ) -> None:
        self.iface = iface
        self.canvas = iface.mapCanvas()
        self.crop_tool = crop_tool
        self.target_focus_tool = target_focus_tool
        self.video_tool = video_tool
        self.color_tool = color_tool
        self.lens_toolbar_action = lens_toolbar_action
        self._toolbar_button_policy = toolbar_button_policy

        self.map_tool = QgsMapToolExtent(self.canvas)
        self.map_tool.extentChanged.connect(self._on_extent_drawn)

        self.toolbar = None
        self.crop_btn: QToolButton | None = None
        self._mode_menu_actions: dict[str, QAction] = {}
        self._suppress_toggle = False
        self._crop_mode = "normal"
        self._color_sub_mode = "fast_time"

    def setup(self) -> None:
        """Create the toolbar and actions, connect signals."""
        self.toolbar = self.iface.addToolBar("ICEYE Crop")
        self.toolbar.setObjectName("ICEYE Crop")

        self.crop_btn = QToolButton(self.iface.mainWindow())
        self.crop_btn.setObjectName("ICEYECropToolButton")
        self.crop_btn.setCheckable(True)
        self.crop_btn.setPopupMode(QToolButton.MenuButtonPopup)
        self.crop_btn.setStatusTip(_tr("Toggle crop tool"))
        self.crop_btn.clicked.connect(self._toggle_crop)

        crop_mode_menu = QMenu(self.crop_btn)

        normal_action = crop_mode_menu.addAction(
            QIcon(_CROP_ICON),
            _tr("Crop Mode: Normal"),
        )
        normal_action.setStatusTip(_tr("Crop mode: normal"))
        normal_action.setCheckable(True)
        normal_action.triggered.connect(lambda: self._set_crop_mode("normal"))
        self._mode_menu_actions["normal"] = normal_action

        focus_action = crop_mode_menu.addAction(
            QIcon(_FOCUS_ICON),
            _tr("Crop Mode: Focus"),
        )
        focus_action.setStatusTip(_tr("Crop mode: focus"))
        focus_action.setCheckable(True)
        focus_action.triggered.connect(lambda: self._set_crop_mode("focus"))
        self._mode_menu_actions["focus"] = focus_action

        video_action = crop_mode_menu.addAction(
            QIcon(_VIDEO_ICON),
            _tr("Crop Mode: Video"),
        )
        video_action.setStatusTip(_tr("Crop mode: video"))
        video_action.setCheckable(True)
        video_action.triggered.connect(lambda: self._set_crop_mode("video"))
        self._mode_menu_actions["video"] = video_action

        crop_mode_menu.addSeparator()

        color_fast = crop_mode_menu.addAction(
            QIcon(_COLOR_ICON),
            _tr("Crop Mode: Color — Range subaperture"),
        )
        color_fast.setStatusTip(_tr("Crop mode: color (range subaperture)"))
        color_fast.setCheckable(True)
        color_fast.triggered.connect(lambda: self._pick_color_mode("fast_time"))
        self._mode_menu_actions["color_fast"] = color_fast

        color_slow = crop_mode_menu.addAction(
            QIcon(_COLOR_ICON),
            _tr("Crop Mode: Color — Azimuth subaperture"),
        )
        color_slow.setStatusTip(_tr("Crop mode: color (azimuth subaperture)"))
        color_slow.setCheckable(True)
        color_slow.triggered.connect(lambda: self._pick_color_mode("slow_time"))
        self._mode_menu_actions["color_slow"] = color_slow

        self.crop_mode_group = QActionGroup(self.iface.mainWindow())
        self.crop_mode_group.setExclusive(True)
        for action in self._mode_menu_actions.values():
            self.crop_mode_group.addAction(action)

        self.crop_btn.setMenu(crop_mode_menu)
        self.toolbar.addWidget(self.crop_btn)

        self._set_crop_mode("normal")

        for action in self._mode_menu_actions.values():
            action.setEnabled(False)

        if self._toolbar_button_policy is not None:
            self._toolbar_button_policy.register(
                self.crop_btn,
                self._policy_crop_action,
                on_disable=self.deactivate,
            )
            self._toolbar_button_policy.register(
                self._mode_menu_actions["normal"],
                self._policy_crop_normal,
            )
            for key in ("focus", "video", "color_fast", "color_slow"):
                self._toolbar_button_policy.register(
                    self._mode_menu_actions[key],
                    self._policy_crop_focus_video_color,
                    on_disable=self._on_slc_crop_mode_policy_disabled,
                )
            self._toolbar_button_policy.refresh()

        self.map_tool.deactivated.connect(self._on_map_tool_deactivated)

    def _compose_crop_toolbar_icon(self) -> QIcon:
        """Return the crop toolbar icon, with a small badge when mode is focus, video, or color."""
        size = QSize(_ICON_PX, _ICON_PX)
        pm = QPixmap(size)
        pm.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pm)
        base = QIcon(_CROP_ICON).pixmap(size)
        painter.drawPixmap(0, 0, base)
        overlay_path = _MODE_OVERLAY.get(self._crop_mode)
        if overlay_path:
            badge = QIcon(overlay_path).pixmap(QSize(_BADGE_PX, _BADGE_PX))
            x = _ICON_PX - _BADGE_PX - 1
            y = _ICON_PX - _BADGE_PX - 1
            painter.setBrush(QColor(255, 255, 255, 230))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(x - 1, y - 1, _BADGE_PX + 2, _BADGE_PX + 2)
            painter.drawPixmap(x, y, badge)
        painter.end()
        return QIcon(pm)

    def _update_crop_btn_tooltip(self) -> None:
        if self.crop_btn is None:
            return
        mode = self._crop_mode
        if mode == "color":
            if self._color_sub_mode == "fast_time":
                tip = _tr("Crop Mode: Color — Range subaperture")
            else:
                tip = _tr("Crop Mode: Color — Azimuth subaperture")
        else:
            tip = {
                "normal": _tr("Crop Mode: Normal"),
                "focus": _tr("Crop Mode: Focus"),
                "video": _tr("Crop Mode: Video"),
            }[mode]
        self.crop_btn.setToolTip(f"{_tr('Crop')} — {tip}")

    def _policy_crop_action(self, layer: QgsMapLayer | None) -> bool:
        """Crop toggle: any ICEYE raster with metadata."""
        if self._toolbar_button_policy is None:
            return True
        return self._toolbar_button_policy.enabled_if_iceye_layer(layer)

    def _policy_crop_normal(self, layer: QgsMapLayer | None) -> bool:
        """Return True if the crop session is active and the layer has ICEYE metadata."""
        if self.crop_btn is None or not self.crop_btn.isChecked():
            return False
        if self._toolbar_button_policy is None:
            return True
        return self._toolbar_button_policy.enabled_if_iceye_layer(layer)

    def _policy_crop_focus_video_color(self, layer: QgsMapLayer | None) -> bool:
        """Focus / video / color: SLC-COG product while crop session is active."""
        if self.crop_btn is None or not self.crop_btn.isChecked():
            return False
        if self._toolbar_button_policy is None:
            return False
        return self._toolbar_button_policy.enabled_if_slc_cog(layer)

    def _on_slc_crop_mode_policy_disabled(self) -> None:
        m = self._crop_mode
        if m in ("focus", "video", "color"):
            self._set_crop_mode(m)

    def unload(self) -> None:
        """Remove toolbar and actions, deactivate map tool."""
        if self.crop_btn is not None:
            try:
                if self.toolbar is not None:
                    self.toolbar.removeWidget(self.crop_btn)
            except Exception:
                pass
            self.crop_btn.deleteLater()
            self.crop_btn = None

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

    def deactivate(self) -> None:
        """Deactivate crop toolbar."""
        if self.crop_btn is not None and self.crop_btn.isChecked():
            self.crop_btn.setChecked(False)

    def _toggle_crop(self) -> None:
        if self._suppress_toggle:
            return

        enabled = self.crop_btn is not None and self.crop_btn.isChecked()

        if enabled:
            if (
                self.lens_toolbar_action is not None
                and self.canvas.mapTool() is self.lens_toolbar_action.lens_tool
            ):
                self.lens_toolbar_action.deactivate()
            self.canvas.setMapTool(self.map_tool)
            if self._toolbar_button_policy is not None:
                self._toolbar_button_policy.refresh()
            else:
                for action in self._mode_menu_actions.values():
                    action.setEnabled(True)
            self._set_crop_mode("normal")
            return

        for action in self._mode_menu_actions.values():
            action.setEnabled(False)

        if self._toolbar_button_policy is not None:
            self._toolbar_button_policy.refresh()

        if self.canvas.mapTool() is self.map_tool:
            self.canvas.unsetMapTool(self.map_tool)
            self.iface.actionPan().trigger()

    def _on_extent_drawn(self, extent):
        """Handle extent drawn: unset map tool, switch to pan, dispatch to handler."""
        self.canvas.unsetMapTool(self.map_tool)
        self.iface.actionPan().trigger()

        if self._crop_mode == "normal":
            self.crop_tool.process_extent(extent)
        elif self._crop_mode == "focus":
            self.target_focus_tool.process_extent(extent)
        elif self._crop_mode == "video":
            self.video_tool.process_extent(extent)
        elif self._crop_mode == "color":
            self.color_tool.process_extent(extent, color_mode=self._color_sub_mode)

        self._suppress_toggle = True
        if self.crop_btn is not None:
            self.crop_btn.blockSignals(True)
            self.crop_btn.setChecked(False)
            self.crop_btn.blockSignals(False)
        self._suppress_toggle = False

        for action in self._mode_menu_actions.values():
            action.setEnabled(False)

        if self._toolbar_button_policy is not None:
            self._toolbar_button_policy.refresh()

    def _pick_color_mode(self, sub_mode: str) -> None:
        self._color_sub_mode = sub_mode
        self._set_crop_mode("color")

    def _set_crop_mode(self, mode: str) -> None:
        self._crop_mode = mode

        for action in self._mode_menu_actions.values():
            action.blockSignals(True)
            action.setChecked(False)
            action.blockSignals(False)

        if mode == "normal":
            self._mode_menu_actions["normal"].setChecked(True)
        elif mode == "focus":
            self._mode_menu_actions["focus"].setChecked(True)
        elif mode == "video":
            self._mode_menu_actions["video"].setChecked(True)
        elif mode == "color":
            if self._color_sub_mode == "fast_time":
                self._mode_menu_actions["color_fast"].setChecked(True)
            else:
                self._mode_menu_actions["color_slow"].setChecked(True)

        if self.crop_btn is not None:
            self.crop_btn.setIcon(self._compose_crop_toolbar_icon())
            self._update_crop_btn_tooltip()

    def _on_map_tool_deactivated(self) -> None:
        """Handle map tool deactivated: unset crop tool."""
        if self.crop_btn is not None and self.crop_btn.isChecked():
            self._suppress_toggle = True
            self.crop_btn.blockSignals(True)
            self.crop_btn.setChecked(False)
            self.crop_btn.blockSignals(False)
            self._suppress_toggle = False
            for action in self._mode_menu_actions.values():
                action.setEnabled(False)

            if self._toolbar_button_policy is not None:
                self._toolbar_button_policy.refresh()
