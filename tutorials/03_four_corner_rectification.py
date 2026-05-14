import json
from pathlib import Path

import numpy as np

from moirestrain import (
    analyze_grid,
    homography_from_points,
    make_microstrain_square_grid,
    recommended_strain_smoothing_window,
    rectangle_points,
    rectify_image_pair,
    save_analysis_npz,
    warp_perspective,
)


def main() -> None:
    output_dir = Path("data/tutorials")
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = make_microstrain_square_grid(
        shape=(120, 160),
        period=16,
        strain_xx=1200e-6,
        strain_yy=-500e-6,
        shear_xy=400e-6,
        supersample=32,
        noise_std=0.0,
    )
    image_points = np.array(
        [
            [72.0, 44.0],
            [256.0, 58.0],
            [238.0, 190.0],
            [54.0, 176.0],
        ]
    )
    rect_to_camera = homography_from_points(rectangle_points(experiment.reference.shape), image_points)
    reference_camera = warp_perspective(
        experiment.reference,
        rect_to_camera,
        output_shape=(240, 320),
        fill_value=1.0,
    )
    deformed_camera = warp_perspective(
        experiment.deformed,
        rect_to_camera,
        output_shape=(240, 320),
        fill_value=1.0,
    )
    reference_rect, deformed_rect = rectify_image_pair(
        reference_camera,
        deformed_camera,
        image_points,
        output_shape=experiment.reference.shape,
    )

    calibration = {
        "image_points": image_points.tolist(),
        "output_shape": list(experiment.reference.shape),
        "period": experiment.period,
    }
    with (output_dir / "03_calibration.json").open("w", encoding="utf-8") as handle:
        json.dump(calibration, handle, indent=2)
        handle.write("\n")

    np.save(output_dir / "03_reference_camera.npy", reference_camera)
    np.save(output_dir / "03_reference_rectified.npy", reference_rect)

    strain_window = recommended_strain_smoothing_window(experiment.period, cycles=3)
    result = analyze_grid(
        reference_rect,
        deformed_rect,
        period=experiment.period,
        strain_smooth_window=strain_window,
    )
    valid_mask = np.ones(experiment.reference.shape, dtype=bool)
    save_analysis_npz(
        output_dir / "03_rectified_analysis.npz",
        u=result.x.displacement,
        v=result.y.displacement,
        exx=result.strain.exx,
        eyy=result.strain.eyy,
        gamma_xy=result.strain.gamma_xy,
        valid_mask=valid_mask,
        reference=reference_rect,
        deformed=deformed_rect,
        extra_arrays={"image_points": image_points},
    )

    print("wrote: data/tutorials/03_calibration.json")
    print("wrote: data/tutorials/03_reference_camera.npy")
    print("wrote: data/tutorials/03_reference_rectified.npy")
    print("wrote: data/tutorials/03_rectified_analysis.npz")


if __name__ == "__main__":
    main()
