from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PointArray = np.ndarray


@dataclass(frozen=True)
class PerspectiveCalibration:
    """Planar perspective calibration for a grating region.

    ``image_points`` and ``world_points`` are four ``(x, y)`` corner points in
    corresponding order. The homography maps image coordinates to rectified
    pixel coordinates.
    """

    image_points: np.ndarray
    world_points: np.ndarray
    output_shape: tuple[int, int]
    homography: np.ndarray
    pixel_spacing: tuple[float, float]

    @classmethod
    def from_points(
        cls,
        image_points: PointArray,
        world_points: PointArray,
        *,
        output_shape: tuple[int, int],
    ) -> "PerspectiveCalibration":
        image_points = _as_points(image_points, "image_points")
        world_points = _as_points(world_points, "world_points")
        height, width = _as_shape(output_shape)
        destination = rectangle_points(output_shape)
        homography = homography_from_points(image_points, destination)
        return cls(
            image_points=image_points,
            world_points=world_points,
            output_shape=(height, width),
            homography=homography,
            pixel_spacing=pixel_spacing_from_world_points(world_points, output_shape),
        )


def _as_points(points: PointArray, name: str) -> np.ndarray:
    array = np.asarray(points, dtype=float)
    if array.shape != (4, 2):
        raise ValueError(f"{name} must have shape (4, 2)")
    return array


def _as_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError("output_shape must be (height, width)")
    height, width = map(int, shape)
    if height < 2 or width < 2:
        raise ValueError("output_shape values must be greater than one")
    return height, width


def rectangle_points(shape: tuple[int, int]) -> np.ndarray:
    """Return rectangle corner points for ``shape`` in ``(x, y)`` order."""

    height, width = _as_shape(shape)
    return np.array(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ]
    )


def homography_from_points(source_points: PointArray, destination_points: PointArray) -> np.ndarray:
    """Estimate a projective transform from four point correspondences.

    Points are ``(x, y)`` pairs. The returned matrix maps homogeneous source
    coordinates to destination coordinates.
    """

    source = _as_points(source_points, "source_points")
    destination = _as_points(destination_points, "destination_points")
    rows = []
    for (x, y), (u, v) in zip(source, destination):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    homography = vh[-1].reshape(3, 3)
    if np.isclose(homography[2, 2], 0.0):
        raise ValueError("degenerate point configuration")
    return homography / homography[2, 2]


def apply_homography(points: PointArray, homography: np.ndarray) -> np.ndarray:
    """Apply a homography to ``(..., 2)`` points."""

    pts = np.asarray(points, dtype=float)
    if pts.shape[-1] != 2:
        raise ValueError("points must have last dimension 2")
    h = np.asarray(homography, dtype=float)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")

    flat = pts.reshape(-1, 2)
    homogeneous = np.column_stack([flat, np.ones(flat.shape[0])])
    mapped = homogeneous @ h.T
    denom = mapped[:, 2]
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("homography maps points to infinity")
    xy = mapped[:, :2] / denom[:, None]
    return xy.reshape(pts.shape)


def warp_perspective(
    image: np.ndarray,
    homography: np.ndarray,
    output_shape: tuple[int, int],
    *,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Warp an image with bilinear interpolation.

    ``homography`` maps input image coordinates to output image coordinates.
    The implementation samples the input image by applying the inverse
    homography to each output pixel.
    """

    source = np.asarray(image, dtype=float)
    if source.ndim != 2:
        raise ValueError("image must be a 2D array")
    height, width = _as_shape(output_shape)
    h = np.asarray(homography, dtype=float)
    if h.shape != (3, 3):
        raise ValueError("homography must have shape (3, 3)")

    yy, xx = np.mgrid[:height, :width]
    output_points = np.stack([xx, yy], axis=-1)
    input_points = apply_homography(output_points, np.linalg.inv(h))
    return sample_bilinear(source, input_points[..., 0], input_points[..., 1], fill_value)


def sample_bilinear(
    image: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Sample a 2D image at floating-point ``x``/``y`` coordinates."""

    source = np.asarray(image, dtype=float)
    if source.ndim != 2:
        raise ValueError("image must be a 2D array")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    height, width = source.shape
    valid = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0)
    x_clipped = np.clip(x, 0.0, width - 1.0)
    y_clipped = np.clip(y, 0.0, height - 1.0)
    x0 = np.floor(x_clipped).astype(int)
    y0 = np.floor(y_clipped).astype(int)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    wx = x_clipped - x0
    wy = y_clipped - y0

    top = (1.0 - wx) * source[y0, x0] + wx * source[y0, x1]
    bottom = (1.0 - wx) * source[y1, x0] + wx * source[y1, x1]
    sampled = (1.0 - wy) * top + wy * bottom
    return np.where(valid, sampled, fill_value)


def rectify_image(
    image: np.ndarray,
    calibration_or_points: PerspectiveCalibration | PointArray,
    *,
    output_shape: tuple[int, int] | None = None,
    world_points: PointArray | None = None,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Rectify a planar grating ROI to a fronto-parallel image.

    Pass either a :class:`PerspectiveCalibration` or four image corner points.
    When passing points directly, ``output_shape`` is required. The point order
    should be top-left, top-right, bottom-right, bottom-left.
    """

    if isinstance(calibration_or_points, PerspectiveCalibration):
        calibration = calibration_or_points
    else:
        if output_shape is None:
            raise ValueError("output_shape is required when passing image points")
        image_points = _as_points(calibration_or_points, "image_points")
        if world_points is None:
            world_points = rectangle_points(output_shape)
        calibration = PerspectiveCalibration.from_points(
            image_points,
            world_points,
            output_shape=output_shape,
        )
    return warp_perspective(
        image,
        calibration.homography,
        calibration.output_shape,
        fill_value=fill_value,
    )


def rectify_image_pair(
    reference_image: np.ndarray,
    deformed_image: np.ndarray,
    calibration_or_points: PerspectiveCalibration | PointArray,
    *,
    output_shape: tuple[int, int] | None = None,
    world_points: PointArray | None = None,
    fill_value: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Rectify reference/deformed images with the same planar transform."""

    reference = rectify_image(
        reference_image,
        calibration_or_points,
        output_shape=output_shape,
        world_points=world_points,
        fill_value=fill_value,
    )
    deformed = rectify_image(
        deformed_image,
        calibration_or_points,
        output_shape=output_shape,
        world_points=world_points,
        fill_value=fill_value,
    )
    return reference, deformed


def resample_oblique_grid(
    image: np.ndarray,
    *,
    origin: tuple[float, float],
    x_vector: tuple[float, float],
    y_vector: tuple[float, float],
    output_shape: tuple[int, int],
    fill_value: float = np.nan,
) -> np.ndarray:
    """Sample an image on an oblique grid defined by two image-space vectors.

    This is useful when the grating is rotated or sheared but a full
    perspective model is unnecessary. ``origin``, ``x_vector``, and
    ``y_vector`` are image-space ``(x, y)`` coordinates. The output x/y axes
    follow those vectors.
    """

    height, width = _as_shape(output_shape)
    yy, xx = np.mgrid[:height, :width]
    origin_array = np.asarray(origin, dtype=float)
    x_vec = np.asarray(x_vector, dtype=float)
    y_vec = np.asarray(y_vector, dtype=float)
    if origin_array.shape != (2,) or x_vec.shape != (2,) or y_vec.shape != (2,):
        raise ValueError("origin, x_vector, and y_vector must be length-2 tuples")
    x_coords = origin_array[0] + xx * x_vec[0] + yy * y_vec[0]
    y_coords = origin_array[1] + xx * x_vec[1] + yy * y_vec[1]
    return sample_bilinear(image, x_coords, y_coords, fill_value=fill_value)


def crop_roi(image: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a rectangular ROI as ``(y0, x0, y1, x1)``."""

    source = np.asarray(image)
    if source.ndim != 2:
        raise ValueError("image must be a 2D array")
    y0, x0, y1, x1 = map(int, bounds)
    if y0 < 0 or x0 < 0 or y1 > source.shape[0] or x1 > source.shape[1] or y1 <= y0 or x1 <= x0:
        raise ValueError("bounds must be inside the image and non-empty")
    return source[y0:y1, x0:x1]


def pixel_spacing_from_world_points(
    world_points: PointArray,
    output_shape: tuple[int, int],
) -> tuple[float, float]:
    """Estimate ``(dy, dx)`` spacing from rectangular world corner points."""

    world = _as_points(world_points, "world_points")
    height, width = _as_shape(output_shape)
    top = np.linalg.norm(world[1] - world[0])
    bottom = np.linalg.norm(world[2] - world[3])
    right = np.linalg.norm(world[2] - world[1])
    left = np.linalg.norm(world[3] - world[0])
    dx = 0.5 * (top + bottom) / (width - 1)
    dy = 0.5 * (left + right) / (height - 1)
    return float(dy), float(dx)
