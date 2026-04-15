"""Enable or disable toolbar controls from the active map layer.

Call ``enabled_if_iceye_layer`` for any ICEYE raster (via ``MetadataProvider``).
Call ``enabled_if_slc_cog`` when the product must be ``SLC-COG``.
"""

from __future__ import annotations

from collections.abc import Callable

from qgis.core import QgsMapLayer
from qgis.PyQt.QtCore import QObject

from ..core.metadata import MetadataProvider


def layer_is_slc_cog(
    layer: QgsMapLayer | None, metadata_provider: MetadataProvider | None
) -> bool:
    """Return True if ICEYE metadata lists ``sar_product_type`` as ``SLC-COG``."""
    if metadata_provider is None or layer is None:
        return False
    meta = metadata_provider.get(layer)
    if meta is None:
        return False
    return meta.sar_product_type == "SLC-COG"


class ToolbarButtonPolicy:
    """Apply enable/disable rules for registered controls when the layer changes."""

    def __init__(
        self,
        iface: object,
        metadata_provider: MetadataProvider | None = None,
    ) -> None:
        self._iface = iface
        self._metadata_provider = metadata_provider
        self._entries: list[
            tuple[
                QObject, Callable[[QgsMapLayer | None], bool], Callable[[], None] | None
            ]
        ] = []
        self._last_enabled: dict[int, bool | None] = {}

    def enabled_if_iceye_layer(self, layer: QgsMapLayer | None) -> bool:
        """Return True if the active layer has ICEYE metadata (any product type).

        Returns False when the layer is missing, not ICEYE, or the provider is unset.
        """
        if self._metadata_provider is None or layer is None:
            return False
        return self._metadata_provider.get(layer) is not None

    def enabled_if_slc_cog(self, layer: QgsMapLayer | None) -> bool:
        """Return True if metadata lists product type ``SLC-COG``."""
        return layer_is_slc_cog(layer, self._metadata_provider)

    def register(
        self,
        control: QObject,
        enabled_if: Callable[[QgsMapLayer | None], bool],
        on_disable: Callable[[], None] | None = None,
    ) -> None:
        """Register a control; ``enabled_if(active_layer)`` sets whether it is enabled.

        Parameters
        ----------
        control
            Widget or action to enable or disable.
        enabled_if
            Predicate on the active map layer.
        on_disable
            Called when the policy turns the control off after it was enabled
            (e.g. deactivate a map tool). Not used on the first ``refresh``.
        """
        self._entries.append((control, enabled_if, on_disable))
        oid = id(control)
        if oid not in self._last_enabled:
            self._last_enabled[oid] = None

    def setup(self) -> None:
        """Connect to ``currentLayerChanged`` and run an initial ``refresh``."""
        self._iface.currentLayerChanged.connect(self._on_layer_changed)
        self.refresh()

    def refresh(self, layer: QgsMapLayer | None = None) -> None:
        """Re-evaluate all registered controls (default layer: active layer)."""
        if layer is None:
            layer = self._iface.activeLayer()
        for control, enabled_if, on_disable in self._entries:
            want = enabled_if(layer)
            oid = id(control)
            prev = self._last_enabled.get(oid)
            if prev is True and not want and on_disable is not None:
                on_disable()
            control.setEnabled(want)
            self._last_enabled[oid] = want

    def _on_layer_changed(self, layer: QgsMapLayer | None) -> None:
        self.refresh(layer)

    def unload(self) -> None:
        """Disconnect from the interface and clear all registrations."""
        try:
            self._iface.currentLayerChanged.disconnect(self._on_layer_changed)
        except (TypeError, RuntimeError):
            pass
        self._entries.clear()
        self._last_enabled.clear()
