import numpy as np
import pytest

from moirestrain import crop_grating_roi, detect_grating_roi


def test_detect_grating_roi_on_synthetic_partial_patch():
    period = 8
    patch_shape = (80, 100)
    y, x = np.mgrid[: patch_shape[0], : patch_shape[1]]
    patch = 0.5 + 0.42 * np.cos(2.0 * np.pi * x / period)
    image = np.full((256, 320), 0.25)
    image[120:200, 180:280] = patch

    roi = detect_grating_roi(image, period=period, min_area=1_000)
    y0, x0, y1, x1 = roi.bounds

    assert y0 <= 130
    assert x0 <= 190
    assert y1 >= 190
    assert x1 >= 270
    cropped = crop_grating_roi(image, roi)
    assert cropped.ndim == 2
    assert cropped.shape[0] == y1 - y0
    assert cropped.shape[1] == x1 - x0


def test_detect_grating_roi_oriented_box_is_close_to_rotated_patch():
    transform = pytest.importorskip("skimage.transform")

    period = 8
    patch_shape = (80, 100)
    y, x = np.mgrid[: patch_shape[0], : patch_shape[1]]
    patch = 0.5 + 0.42 * np.cos(2.0 * np.pi * x / period)
    image = np.full((150, 180), 0.25)
    source = np.array([[0, 0], [99, 0], [99, 79], [0, 79]], dtype=float)
    expected = np.array([[40, 25], [140, 34], [132, 116], [32, 106]], dtype=float)
    tform = transform.ProjectiveTransform()
    assert tform.estimate(expected, source)
    warped = transform.warp(
        patch,
        inverse_map=tform,
        output_shape=image.shape,
        mode="constant",
        cval=np.nan,
        preserve_range=True,
    )
    valid = np.isfinite(warped)
    image[valid] = warped[valid]

    roi = detect_grating_roi(image, period=period, min_area=1_000)

    assert roi.image_points.shape == (4, 2)
