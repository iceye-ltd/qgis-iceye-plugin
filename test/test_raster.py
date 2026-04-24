"""Tests for raster module."""

import numpy as np

from iceye_toolbox.core.raster import toggle_shadows_down


class TestToggleShadowsDown:
    """Tests for ToggleShadowsDown class."""

    def test_leftside_flip_and_transposes(self):
        """Left-side data should be flipped and transposed."""
        data = np.array([[1, 2, 3], [4, 5, 6]])

        result = toggle_shadows_down(data, left=True)

        expected = np.array([[3, 6], [2, 5], [1, 4]])
        np.testing.assert_array_equal(result, expected)

    def test_rightside_transposes(self):
        """Right-side data should be transposed only."""
        data = np.array([[1, 2, 3], [4, 5, 6]])

        result = toggle_shadows_down(data, left=False)

        expected = np.array([[1, 4], [2, 5], [3, 6]])
        np.testing.assert_array_equal(result, expected)

    def test_preserves_complex_dtype(self):
        """Complex data type should be preserved."""
        data = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)
        result = toggle_shadows_down(data, left=False)

        assert result.dtype == np.complex64
