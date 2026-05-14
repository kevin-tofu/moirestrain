import numpy as np
import pytest

from moirestrain import (
    PerspectiveCalibration,
    analyze,
    apply_homography,
    crop_roi,
    homography_from_points,
    rectangle_points,
    rectify_image,
    resample_oblique_grid,
    warp_perspective,
)


def test_homography_maps_source_points_to_destination_points():
    source = np.array([[2, 3], [12, 2], [11, 8], [1, 9]], dtype=float)
    destination = np.array([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=float)

    homography = homography_from_points(source, destination)
    mapped = apply_homography(source, homography)

    np.testing.assert_allclose(mapped, destination, atol=1e-10)


def test_perspective_calibration_reports_pixel_spacing():
    image_points = np.array([[10, 12], [50, 10], [54, 42], [8, 45]], dtype=float)
    world_points = np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float)

    calibration = PerspectiveCalibration.from_points(
        image_points,
        world_points,
        output_shape=(21, 41),
    )

    assert calibration.output_shape == (21, 41)
    np.testing.assert_allclose(calibration.pixel_spacing, (0.1, 0.1))


def test_rectify_partial_grating_roi_then_analyze_displacement():
    output_shape = (96, 128)
    period = 8
    shift = 0.75
    _y, x = np.mgrid[: output_shape[0], : output_shape[1]]
    reference_rect = 0.5 + 0.45 * np.cos(2.0 * np.pi * x / period)
    deformed_rect = 0.5 + 0.45 * np.cos(2.0 * np.pi * (x + shift) / period)

    canvas_shape = (150, 190)
    image_points = np.array(
        [
            [34.0, 18.0],
            [164.0, 30.0],
            [150.0, 128.0],
            [20.0, 118.0],
        ]
    )
    rect_to_canvas = homography_from_points(rectangle_points(output_shape), image_points)
    reference_raw = warp_perspective(
        reference_rect,
        rect_to_canvas,
        canvas_shape,
        fill_value=0.5,
    )
    deformed_raw = warp_perspective(
        deformed_rect,
        rect_to_canvas,
        canvas_shape,
        fill_value=0.5,
    )

    reference = rectify_image(reference_raw, image_points, output_shape=output_shape)
    deformed = rectify_image(deformed_raw, image_points, output_shape=output_shape)
    result = analyze(reference, deformed, period=period, axis="x", unwrap_axis="x")

    margin = 2 * period
    central = result.displacement[margin:-margin, margin:-margin]
    assert np.mean(central) == pytest.approx(shift, abs=0.08)


def test_crop_roi_uses_yx_bounds():
    image = np.arange(6 * 8).reshape(6, 8)

    cropped = crop_roi(image, (1, 2, 4, 6))

    np.testing.assert_array_equal(cropped, image[1:4, 2:6])


def test_resample_oblique_grid_samples_along_basis_vectors():
    y, x = np.mgrid[:20, :30]
    image = x + 10.0 * y

    sampled = resample_oblique_grid(
        image,
        origin=(3.0, 4.0),
        x_vector=(1.0, 0.0),
        y_vector=(0.0, 1.0),
        output_shape=(5, 6),
    )

    expected_y, expected_x = np.mgrid[:5, :6]
    expected = (expected_x + 3.0) + 10.0 * (expected_y + 4.0)
    np.testing.assert_allclose(sampled, expected)
