"""SAR look extraction utilities for spectrum processing."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray


def _insert_center(
    dst: NDArray[np.complex64], src: NDArray[np.complex64]
) -> NDArray[np.complex64]:
    """Insert source array centered into destination array.

    Parameters
    ----------
    dst : ndarray of complex64
        Destination array (modified in place).
    src : ndarray of complex64
        Source array to insert.

    Returns
    -------
    ndarray of complex64
        The destination array with source inserted at center.

    Raises
    ------
    ValueError
        If source array is larger than destination array.
    """
    rows, cols = dst.shape
    src_rows, src_cols = src.shape

    if src_rows > rows or src_cols > cols:
        raise ValueError("Source array is larger than destination array")

    start_row = rows // 2 - src_rows // 2
    start_col = cols // 2 - src_cols // 2

    dst[
        start_row : start_row + src_rows,
        start_col : start_col + src_cols,
    ] = src
    return dst


def extract_centered_look(
    spectrum: NDArray[np.complex64],
    center_row: int,
    center_col: int,
    look_rows: int,
    look_cols: int,
    *,
    apply_ifftshift: bool = True,
) -> NDArray[np.complex64]:
    """Extract a look from spectrum, center it, then return its inverse FFT.

    Parameters
    ----------
    spectrum : ndarray of complex64
        2D complex spectrum.
    center_row : int
        Row index of look center.
    center_col : int
        Column index of look center.
    look_rows : int
        Number of rows in the look.
    look_cols : int
        Number of columns in the look.
    apply_ifftshift : bool, optional
        Whether to apply ifftshift before ifft2. Default is True.

    Returns
    -------
    ndarray of complex64
        Focused look in spatial domain.

    Raises
    ------
    ValueError
        If look size is non-positive, center is out of bounds, or look
        exceeds spectrum dimensions.
    """
    if look_rows <= 0 or look_cols <= 0:
        raise ValueError("Look size must be positive in both dimensions")

    rows, cols = spectrum.shape
    if not (0 <= center_row < rows and 0 <= center_col < cols):
        raise ValueError("Center index is out of bounds for the spectrum")

    if look_rows > rows or look_cols > cols:
        raise ValueError("Look size cannot exceed spectrum dimensions")

    window_row_start = center_row - look_rows // 2
    window_row_end = window_row_start + look_rows
    window_col_start = center_col - look_cols // 2
    window_col_end = window_col_start + look_cols

    src_row_start = max(0, window_row_start)
    src_row_end = min(rows, window_row_end)
    src_col_start = max(0, window_col_start)
    src_col_end = min(cols, window_col_end)

    look_window = np.zeros((look_rows, look_cols), dtype=spectrum.dtype)
    dst_row_start = src_row_start - window_row_start
    dst_row_end = dst_row_start + (src_row_end - src_row_start)
    dst_col_start = src_col_start - window_col_start
    dst_col_end = dst_col_start + (src_col_end - src_col_start)

    look_window[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = spectrum[
        src_row_start:src_row_end, src_col_start:src_col_end
    ]

    centered_spectrum = _insert_center(np.zeros_like(spectrum), look_window)
    if apply_ifftshift:
        centered_spectrum = np.fft.ifftshift(centered_spectrum)

    return np.fft.ifft2(centered_spectrum)
