from __future__ import annotations

import numpy as np


def inner_valid_mask(shape: tuple[int, int], margin: int) -> np.ndarray:
    """Return a mask that excludes a fixed margin from image borders."""

    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    height, width = map(int, shape)
    margin = int(margin)
    if height <= 0 or width <= 0:
        raise ValueError("shape values must be positive")
    if margin < 0:
        raise ValueError("margin must be non-negative")
    mask = np.zeros((height, width), dtype=bool)
    if 2 * margin >= height or 2 * margin >= width:
        return mask
    mask[margin : height - margin, margin : width - margin] = True
    return mask


def apply_valid_mask(array: np.ndarray, valid_mask: np.ndarray, fill_value: float = np.nan) -> np.ndarray:
    """Return a copy of ``array`` with invalid pixels replaced by ``fill_value``."""

    values = np.asarray(array, dtype=float)
    mask = np.asarray(valid_mask, dtype=bool)
    if values.shape != mask.shape:
        raise ValueError("array and valid_mask must have the same shape")
    return np.where(mask, values, fill_value)


def mask_bounds(valid_mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(y0, x0, y1, x1)`` bounds for true pixels in a mask."""

    mask = np.asarray(valid_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("valid_mask must be a 2D array")
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        raise ValueError("valid_mask has no true pixels")
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def crop_to_mask(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Crop an array to the bounding box of ``valid_mask``."""

    values = np.asarray(array)
    mask = np.asarray(valid_mask, dtype=bool)
    if values.shape[:2] != mask.shape:
        raise ValueError("array first two dimensions and valid_mask must match")
    y0, x0, y1, x1 = mask_bounds(mask)
    return values[y0:y1, x0:x1]


def robust_limits(
    array: np.ndarray,
    *,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[float, float]:
    """Return percentile color limits while ignoring NaNs."""

    values = np.asarray(array, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("array has no finite values")
    lo, hi = np.percentile(finite, percentiles)
    if np.isclose(lo, hi):
        delta = 1.0 if np.isclose(lo, 0.0) else abs(lo) * 0.05
        return float(lo - delta), float(hi + delta)
    return float(lo), float(hi)
