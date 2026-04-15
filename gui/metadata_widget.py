"""Metadata dock widget for displaying ICEYE layer metadata."""

from __future__ import annotations

import json
from pathlib import Path

from qgis.core import Qgis, QgsMessageLog
from qgis.PyQt import uic

from ..core.metadata import MetadataProvider, is_iceye_layer

WIDGET, BASE = uic.loadUiType(
    str(Path(__file__).resolve().parent.parent / "ui" / "metadata_widget.ui")
)

ESSENTIAL_FIELDS = [
    "datetime",
    "created",
    "start_datetime",
    "end_datetime",
    "platform",
    "constellation",
    "proj_centroid",
    "iceye_squint_angle",
]


class MetadataWidget(BASE, WIDGET):
    """Widget for displaying metadata."""

    def __init__(
        self,
        metadata_provider: MetadataProvider,
        parent=None,
        fields_to_display: list[str] | None = None,
    ):
        super().__init__(parent)
        self.setupUi(self)
        self.metadata_provider = metadata_provider
        self.fields_to_display = fields_to_display or ESSENTIAL_FIELDS

        # Signals
        self.mapLayerComboBox.currentIndexChanged.connect(self.on_combox_changed)
        self.on_combox_changed()

    def on_combox_changed(self) -> None:
        """Update metadata display when layer selection changes."""
        layer = self.mapLayerComboBox.currentLayer()
        if layer:
            if is_iceye_layer(layer):
                self.textBrowser.clear()
                m = self.metadata_provider.get(layer)
                if m:
                    self.textBrowser.setText(
                        json.dumps(
                            m.view(
                                self.fields_to_display,
                            ),
                            indent=4,
                            sort_keys=False,
                        )
                    )
                else:
                    QgsMessageLog.logMessage(
                        f"Could not load metadata for layer {layer.name()}",
                        "ICEYE Toolbox",
                        level=Qgis.Warning,
                    )
