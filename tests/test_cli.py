import json

import numpy as np

from moirestrain import (
    homography_from_points,
    make_microstrain_square_grid,
    rectangle_points,
    warp_perspective,
)
from moirestrain.cli import main


def test_cli_analyze_grid_with_npz_inputs(tmp_path):
    experiment = make_microstrain_square_grid(
        shape=(96, 112),
        period=8,
        supersample=16,
        noise_std=0.0,
    )
    reference_path = tmp_path / "ref.npz"
    deformed_path = tmp_path / "def.npz"
    result_path = tmp_path / "result.npz"
    figure_path = tmp_path / "result.png"
    np.savez_compressed(reference_path, image=experiment.reference)
    np.savez_compressed(deformed_path, image=experiment.deformed)

    exit_code = main(
        [
            "analyze-grid",
            str(reference_path),
            str(deformed_path),
            "--reference-key",
            "image",
            "--deformed-key",
            "image",
            "--period",
            "8",
            "--strain-cycles",
            "4",
            "--valid-margin",
            "16",
            "--out",
            str(result_path),
            "--figure",
            str(figure_path),
        ]
    )

    assert exit_code == 0
    assert result_path.exists()
    assert figure_path.exists()
    result = np.load(result_path)
    assert result["u"].shape == experiment.reference.shape
    assert result["valid_mask"].dtype == np.bool_
    assert int(result["period"]) == 8
    assert int(result["strain_window"]) > 0


def test_cli_analyze_grid_with_auto_roi_and_no_figure(tmp_path):
    experiment = make_microstrain_square_grid(
        shape=(72, 80),
        period=8,
        supersample=16,
        noise_std=0.0,
    )
    reference = np.ones((128, 144), dtype=float)
    deformed = np.ones_like(reference)
    reference[24:96, 32:112] = experiment.reference
    deformed[24:96, 32:112] = experiment.deformed

    reference_path = tmp_path / "ref.npy"
    deformed_path = tmp_path / "def.npy"
    result_path = tmp_path / "result.npz"
    figure_path = tmp_path / "result.png"
    np.save(reference_path, reference)
    np.save(deformed_path, deformed)

    exit_code = main(
        [
            "analyze-grid",
            str(reference_path),
            str(deformed_path),
            "--period",
            "8",
            "--auto-roi",
            "--roi-min-area",
            "1000",
            "--valid-margin",
            "12",
            "--out",
            str(result_path),
            "--figure",
            str(figure_path),
            "--no-figure",
        ]
    )

    assert exit_code == 0
    assert result_path.exists()
    assert not figure_path.exists()
    result = np.load(result_path)
    assert result["u"].shape[0] < reference.shape[0]
    assert result["u"].shape[1] < reference.shape[1]
    assert result["roi_bounds"].shape == (4,)


def test_cli_analyze_grid_with_rectification_corners(tmp_path):
    experiment = make_microstrain_square_grid(
        shape=(80, 96),
        period=8,
        supersample=16,
        noise_std=0.0,
    )
    image_points = np.array(
        [
            [22.0, 14.0],
            [124.0, 22.0],
            [116.0, 104.0],
            [14.0, 96.0],
        ]
    )
    rect_to_camera = homography_from_points(rectangle_points(experiment.reference.shape), image_points)
    reference_camera = warp_perspective(
        experiment.reference,
        rect_to_camera,
        (124, 144),
        fill_value=1.0,
    )
    deformed_camera = warp_perspective(
        experiment.deformed,
        rect_to_camera,
        (124, 144),
        fill_value=1.0,
    )

    reference_path = tmp_path / "ref.npy"
    deformed_path = tmp_path / "def.npy"
    result_path = tmp_path / "result.npz"
    np.save(reference_path, reference_camera)
    np.save(deformed_path, deformed_camera)
    corners = ",".join(str(v) for v in image_points.ravel())

    exit_code = main(
        [
            "analyze-grid",
            str(reference_path),
            str(deformed_path),
            "--period",
            "8",
            "--rectify-corners",
            corners,
            "--rectify-shape",
            "80,96",
            "--valid-margin",
            "16",
            "--no-figure",
            "--out",
            str(result_path),
        ]
    )

    assert exit_code == 0
    result = np.load(result_path)
    assert result["u"].shape == experiment.reference.shape
    np.testing.assert_allclose(result["rectify_corners"], image_points)
    np.testing.assert_array_equal(result["rectify_shape"], np.array([80, 96]))


def test_cli_make_calibration_rectify_pair_and_analyze_with_json(tmp_path):
    experiment = make_microstrain_square_grid(
        shape=(72, 88),
        period=8,
        supersample=16,
        noise_std=0.0,
    )
    image_points = np.array(
        [
            [18.0, 12.0],
            [112.0, 18.0],
            [106.0, 92.0],
            [12.0, 86.0],
        ]
    )
    rect_to_camera = homography_from_points(rectangle_points(experiment.reference.shape), image_points)
    reference_camera = warp_perspective(
        experiment.reference,
        rect_to_camera,
        (112, 128),
        fill_value=1.0,
    )
    deformed_camera = warp_perspective(
        experiment.deformed,
        rect_to_camera,
        (112, 128),
        fill_value=1.0,
    )

    reference_path = tmp_path / "ref.npy"
    deformed_path = tmp_path / "def.npy"
    calibration_path = tmp_path / "calibration.json"
    rectified_reference_path = tmp_path / "rect_ref.npy"
    rectified_deformed_path = tmp_path / "rect_def.npy"
    metadata_path = tmp_path / "metadata.json"
    result_path = tmp_path / "result.npz"
    np.save(reference_path, reference_camera)
    np.save(deformed_path, deformed_camera)

    exit_code = main(
        [
            "make-calibration",
            "--image-points",
            ",".join(str(v) for v in image_points.ravel()),
            "--output-shape",
            "72,88",
            "--period",
            "8",
            "--world-points",
            "0,0,8.7,0,8.7,7.1,0,7.1",
            "--unit",
            "mm",
            "--out",
            str(calibration_path),
        ]
    )
    assert exit_code == 0
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert calibration["period"] == 8
    assert calibration["unit"] == "mm"
    assert len(calibration["pixel_spacing"]) == 2

    exit_code = main(
        [
            "rectify-pair",
            str(reference_path),
            str(deformed_path),
            "--calibration",
            str(calibration_path),
            "--reference-out",
            str(rectified_reference_path),
            "--deformed-out",
            str(rectified_deformed_path),
            "--metadata-out",
            str(metadata_path),
        ]
    )
    assert exit_code == 0
    assert np.load(rectified_reference_path).shape == experiment.reference.shape
    assert np.load(rectified_deformed_path).shape == experiment.deformed.shape
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["reference_output"] == str(rectified_reference_path)
    assert metadata["unit"] == "mm"

    exit_code = main(
        [
            "analyze-grid",
            str(reference_path),
            str(deformed_path),
            "--period",
            "8",
            "--calibration",
            str(calibration_path),
            "--valid-margin",
            "16",
            "--no-figure",
            "--out",
            str(result_path),
        ]
    )
    assert exit_code == 0
    result = np.load(result_path)
    assert result["u"].shape == experiment.reference.shape
    np.testing.assert_array_equal(result["rectify_shape"], np.array([72, 88]))
    np.testing.assert_allclose(result["rectify_world_points"][1], np.array([8.7, 0.0]))
    np.testing.assert_allclose(result["pixel_spacing"], np.array([7.1 / 71.0, 8.7 / 87.0]))
    np.testing.assert_allclose(result["u_physical"], result["u_px"] * result["pixel_spacing"][1])
    np.testing.assert_allclose(result["v_physical"], result["v_px"] * result["pixel_spacing"][0])
    assert str(result["unit"]) == "mm"
