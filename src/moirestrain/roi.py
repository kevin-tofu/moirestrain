from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import crop_roi


@dataclass(frozen=True)
class GratingROI:
    """Detected grating region in a larger image.

    ``bounds`` and ``mask`` are the primary detection results. ``image_points``
    are optional rectification hints estimated from the mask; they are not
    required when the downstream analysis works on an axis-aligned crop.
    """

    image_points: np.ndarray
    bounds: tuple[int, int, int, int]
    mask: np.ndarray
    energy: np.ndarray


def _box_filter(image: np.ndarray, window: int) -> np.ndarray:
    if window < 1:
        raise ValueError("window must be positive")
    if window == 1:
        return image
    kernel = np.ones(window, dtype=float) / window
    pad = (window // 2, window - 1 - window // 2)
    padded_y = np.pad(image, (pad, (0, 0)), mode="reflect")
    smoothed = np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"),
        axis=0,
        arr=padded_y,
    )
    padded_x = np.pad(smoothed, ((0, 0), pad), mode="reflect")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"),
        axis=1,
        arr=padded_x,
    )


def grating_energy(image: np.ndarray, *, period: int, window: int | None = None) -> np.ndarray:
    """Calculate local high-frequency energy for grating ROI detection."""

    source = np.asarray(image, dtype=float)
    if source.ndim != 2:
        raise ValueError("image must be a 2D array")
    if period < 3:
        raise ValueError("period must be greater than or equal to 3")
    low_pass = _box_filter(source, max(3, int(period)))
    high_pass = source - low_pass
    energy_window = int(window if window is not None else max(5, 2 * period + 1))
    return _box_filter(high_pass * high_pass, energy_window)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    visited = np.zeros(mask.shape, dtype=bool)
    best: list[tuple[int, int]] = []
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        component = []
        while stack:
            y, x = stack.pop()
            component.append((y, x))
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        stack.append((ny, nx))
        if len(component) > len(best):
            best = component

    component_mask = np.zeros(mask.shape, dtype=bool)
    if best:
        yy, xx = np.asarray(best, dtype=int).T
        component_mask[yy, xx] = True
    return component_mask


def _corner_points_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size < 4:
        raise ValueError("not enough grating pixels were detected")
    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    sums = coords[:, 0] + coords[:, 1]
    diffs = coords[:, 0] - coords[:, 1]
    return np.array(
        [
            coords[np.argmin(sums)],
            coords[np.argmax(diffs)],
            coords[np.argmax(sums)],
            coords[np.argmin(diffs)],
        ],
        dtype=float,
    )


def _oriented_box_from_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if xs.size < 4:
        raise ValueError("not enough grating pixels were detected")

    coords = np.column_stack([xs.astype(float), ys.astype(float)])
    center = np.mean(coords, axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh
    projected = centered @ axes.T
    min_u, min_v = np.min(projected, axis=0)
    max_u, max_v = np.max(projected, axis=0)
    corners_local = np.array(
        [
            [min_u, min_v],
            [max_u, min_v],
            [max_u, max_v],
            [min_u, max_v],
        ]
    )
    corners = corners_local @ axes + center
    return _order_points_clockwise(corners)


def _order_points_clockwise(points: np.ndarray) -> np.ndarray:
    coords = np.asarray(points, dtype=float)
    if coords.shape != (4, 2):
        raise ValueError("points must have shape (4, 2)")
    sums = coords[:, 0] + coords[:, 1]
    diffs = coords[:, 0] - coords[:, 1]
    return np.array(
        [
            coords[np.argmin(sums)],
            coords[np.argmax(diffs)],
            coords[np.argmax(sums)],
            coords[np.argmin(diffs)],
        ],
        dtype=float,
    )


def detect_grating_roi(
    image: np.ndarray,
    *,
    period: int,
    threshold: float | None = None,
    min_area: int | None = None,
    corner_method: str = "oriented_box",
) -> GratingROI:
    """Detect the dominant grating patch by thresholding grating energy."""

    source = np.asarray(image, dtype=float)
    if source.ndim != 2:
        raise ValueError("image must be a 2D array")
    energy = grating_energy(source, period=period)
    if threshold is None:
        median = float(np.median(energy))
        q95 = float(np.quantile(energy, 0.95))
        threshold = median + 0.35 * (q95 - median)

    component = _largest_component(energy > threshold)
    area = int(np.count_nonzero(component))
    required = int(min_area if min_area is not None else max(64, period * period))
    if area < required:
        raise ValueError("no grating ROI large enough was detected")

    ys, xs = np.nonzero(component)
    bounds = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
    if corner_method == "oriented_box":
        image_points = _oriented_box_from_mask(component)
    elif corner_method == "extreme":
        image_points = _corner_points_from_mask(component)
    else:
        raise ValueError("corner_method must be 'oriented_box' or 'extreme'")
    return GratingROI(
        image_points=image_points,
        bounds=bounds,
        mask=component,
        energy=energy,
    )


def crop_grating_roi(
    image: np.ndarray,
    roi: GratingROI,
    *,
    margin: int = 0,
) -> np.ndarray:
    """Crop the axis-aligned bounding box of a detected grating ROI."""

    source = np.asarray(image)
    y0, x0, y1, x1 = roi.bounds
    margin = int(margin)
    if margin < 0:
        raise ValueError("margin must be non-negative")
    bounds = (
        max(0, y0 - margin),
        max(0, x0 - margin),
        min(source.shape[0], y1 + margin),
        min(source.shape[1], x1 + margin),
    )
    return crop_roi(source, bounds)
