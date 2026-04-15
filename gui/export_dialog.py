"""Export dialog for layer export (PNG, TIFF, GIF, MP4) settings."""

from __future__ import annotations

from pathlib import Path

from PyQt5 import uic
from PyQt5.QtWidgets import QFileDialog

WIDGET, BASE = uic.loadUiType(
    str(Path(__file__).resolve().parent.parent / "ui" / "export_dialog.ui")
)


class ExportDialog(BASE, WIDGET):
    """Dialog for configuring layer export (path, format, downscale)."""

    def __init__(
        self,
        parent,
        layer,
        default_path: str = "",
        file_filter: str = "",
    ) -> None:
        super().__init__(parent)
        self.setupUi(self)

        # Store settings
        self.layer = layer
        self.default_path = default_path
        self.file_filter = file_filter
        self._full_path = default_path

        # Get original layer dimensions
        self.original_width = layer.width()
        self.original_height = layer.height()

        # Connect signals
        self.browseButton.clicked.connect(self.browse_file)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Connect downscale controls
        self.downscaleCheckBox.toggled.connect(self._on_downscale_toggled)
        self.downscaleSpinBox.valueChanged.connect(self._update_downscaled_size)

        # Connect path changes to update UI based on file type
        self.pathLineEdit.textChanged.connect(self._on_path_changed)

        # Initialize downscale controls
        self.downscaleSpinBox.setEnabled(False)
        self._update_downscaled_size()

        # Set initial path if provided
        if default_path:
            self._full_path = default_path
            self.pathLineEdit.setText(default_path)
            self._update_ui_for_file_type(default_path)

    def _on_path_changed(self, path):
        """Handle path text changes to update UI based on file type."""
        self._update_ui_for_file_type(path)

    def _update_ui_for_file_type(self, path):
        """Show/hide downscale controls based on file extension."""
        if not path:
            return

        # Get file extension
        ext = Path(path).suffix.lower()

        # Hide downscale for TIFF files (they're just copied, not re-rendered)
        is_tiff = ext in [".tif", ".tiff"]

        # Show/hide downscale controls
        self.downscaleLabel.setVisible(not is_tiff)
        self.downscaleCheckBox.setVisible(not is_tiff)
        self.downscaleSpinBox.setVisible(not is_tiff)
        self.downscaleSizeLabel.setVisible(not is_tiff)

        # If hiding, ensure checkbox is unchecked
        if is_tiff:
            self.downscaleCheckBox.setChecked(False)

    def _on_downscale_toggled(self, checked):
        """Enable/disable downscale spinbox based on checkbox."""
        self.downscaleSpinBox.setEnabled(checked)
        self._update_downscaled_size()

    def _update_downscaled_size(self):
        """Update the displayed downscaled size."""
        if self.downscaleCheckBox.isChecked():
            factor = self.downscaleSpinBox.value() / 100.0
            new_width = int(self.original_width * factor)
            new_height = int(self.original_height * factor)

            self.downscaleSizeLabel.setText(
                f"Downscaled size: {new_width} × {new_height} px "
                f"({self.downscaleSpinBox.value()}% of {self.original_width} × {self.original_height} px)"
            )
        else:
            self.downscaleSizeLabel.setText(
                f"Output size: {self.original_width} × {self.original_height} px (full resolution)"
            )

    def browse_file(self):
        """Open file dialog to select output path."""
        # Use current path if set, otherwise use default
        start_path = self._full_path if self._full_path else self.default_path

        output_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Select Output File", start_path, self.file_filter
        )

        if output_path:
            self._full_path = output_path
            self.pathLineEdit.setText(output_path)

    def get_output_path(self):
        """Get the full output path."""
        # Get the current path from the line edit
        current_path = self.pathLineEdit.text()

        if current_path and current_path != "...":
            return current_path

        return self._full_path if self._full_path else ""

    def get_downscale_enabled(self):
        """Check if downscaling is enabled."""
        return self.downscaleCheckBox.isChecked()

    def get_downscale_factor(self):
        """Get the downscale factor from spinbox."""
        if self.downscaleCheckBox.isChecked():
            return self.downscaleSpinBox.value() / 100.0
        return 1.0

    def get_export_settings(self):
        """Get all export settings as a dictionary."""
        return {
            "output_path": self.get_output_path(),
            "downscale_enabled": self.get_downscale_enabled(),
            "downscale_factor": self.get_downscale_factor(),
        }
