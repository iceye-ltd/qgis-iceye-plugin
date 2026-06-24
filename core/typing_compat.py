"""NumPy typing compatibility (< 1.21)."""

from __future__ import annotations

from typing import Any

try:
    from numpy.typing import NDArray
except ImportError:
    NDArray = Any

__all__ = ["NDArray"]
