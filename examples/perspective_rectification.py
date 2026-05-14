from pathlib import Path

import numpy as np

from moirestrain import (
    analyze,
    homography_from_points,
    rectangle_points,
    rectify_image,
    warp_perspective,
)


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    rectified_shape = (96, 128)
    canvas_shape = (150, 190)
    period = 8
    shift = 0.75

    _y, x = np.mgrid[: rectified_shape[0], : rectified_shape[1]]
    reference_rectified = 0.5 + 0.45 * np.cos(2.0 * np.pi * x / period)
    deformed_rectified = 0.5 + 0.45 * np.cos(2.0 * np.pi * (x + shift) / period)

    # Four corners of the grating region in the full camera image.
    image_points = np.array(
        [
            [34.0, 18.0],
            [164.0, 30.0],
            [150.0, 128.0],
            [20.0, 118.0],
        ]
    )
    rect_to_camera = homography_from_points(rectangle_points(rectified_shape), image_points)
    reference_camera = warp_perspective(
        reference_rectified,
        rect_to_camera,
        canvas_shape,
        fill_value=0.5,
    )
    deformed_camera = warp_perspective(
        deformed_rectified,
        rect_to_camera,
        canvas_shape,
        fill_value=0.5,
    )

    reference = rectify_image(reference_camera, image_points, output_shape=rectified_shape)
    deformed = rectify_image(deformed_camera, image_points, output_shape=rectified_shape)
    result = analyze(reference, deformed, period=period, axis="x", unwrap_axis="x")

    margin = 2 * period
    roi = np.s_[margin:-margin, margin:-margin]
    np.savez_compressed(
        output_dir / "perspective_rectification_result.npz",
        reference_camera=reference_camera,
        deformed_camera=deformed_camera,
        reference_rectified=reference,
        deformed_rectified=deformed,
        displacement=result.displacement,
        image_points=image_points,
    )
    print("wrote: data/perspective_rectification_result.npz")
    print(f"mean displacement in ROI: {np.mean(result.displacement[roi]):.4f} px")
    print(f"true displacement:        {shift:.4f} px")


if __name__ == "__main__":
    main()
