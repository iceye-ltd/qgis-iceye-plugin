"""Video settings dialog for frame count selection."""

from __future__ import annotations

from pathlib import Path

from qgis.PyQt import uic

WIDGET, BASE = uic.loadUiType(
    str(Path(__file__).resolve().parent.parent / "ui" / "video_dialog.ui")
)


class VideoDialog(BASE, WIDGET):
    """Dialog for selecting number of video frames."""

    def __init__(self, parent) -> None:
        super().__init__(parent)
        self.setupUi(self)

        # Set window title
        self.setWindowTitle("Video Settings")
        self.setModal(True)

        self.videoButtonBox.accepted.connect(self.accept)
        self.videoButtonBox.rejected.connect(self.reject)

    def get_num_frames(self) -> int:
        """Get the number of frames from the spinbox."""
        return self.frameSpinBox.value()
