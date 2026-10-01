"""Plugin code must import Qt through qgis.PyQt so it loads on QGIS 3 (Qt5) and 4 (Qt6)."""

from __future__ import annotations

import re
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
DIRECT_QT = re.compile(r"^\s*(from|import)\s+PyQt[56]\b", re.MULTILINE)


def test_no_direct_pyqt_imports():
    """Generated files (e.g. resources.py from pyrcc5) must be fixed up too."""
    offenders = [
        str(path.relative_to(PLUGIN_DIR))
        for path in PLUGIN_DIR.rglob("*.py")
        if "test" not in path.relative_to(PLUGIN_DIR).parts[:1]
        and ".claude" not in path.parts
        and DIRECT_QT.search(path.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert offenders == [], f"Import Qt via qgis.PyQt instead: {offenders}"
