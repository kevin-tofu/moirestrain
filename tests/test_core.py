import numpy as np
import pytest

from moirestrain import (
    analyze,
    analyze_grid,
    displacement,
    make_microstrain_experiment,
    make_microstrain_square_grid,
    make_strain_distribution_experiment,
    phase_shifted_sampling_moire,
    recommended_strain_smoothing_window,
    separate_grid_components,
    strain_field,
    unwrap_phase,
    wrapped_phase,
)


def test_wrapped_phase_recovers_synthetic_phase():
    y, x = np.mgrid[:12, :16]
    phase = 0.2 * x + 0.1 * y - 0.7
    shifts = 2.0 * np.pi * np.arange(8)[:, None, None] / 8
    images = 10.0 + 2.5 * np.cos(phase[None, :, :] + shifts)

    estimated = wrapped_phase(images)
    error = np.angle(np.exp(1j * (estimated - phase)))

    assert np.max(np.abs(error)) < 1e-12


def test_unwrap_phase_unwraps_both_axes_by_default():
    phase = np.array(
        [
            [0.0, 0.75 * np.pi, 1.5 * np.pi],
            [0.25 * np.pi, np.pi, 1.75 * np.pi],
        ]
    )
    wrapped = np.angle(np.exp(1j * phase))

    np.testing.assert_allclose(unwrap_phase(wrapped), phase)


def test_displacement_uses_phase_difference_and_pitch():
    reference = np.zeros((4, 5))
    expected_u = np.full((4, 5), 1.25)
    deformed = 2.0 * np.pi * expected_u / 10.0

    actual = displacement(reference, deformed, grating_pitch=10.0)

    np.testing.assert_allclose(actual, expected_u)


def test_phase_shifted_sampling_moire_matches_analyze():
    height, width = 64, 80
    period = 8
    shift = 0.5
    _y, x = np.mgrid[:height, :width]
    reference = 0.5 + 0.4 * np.cos(2.0 * np.pi * x / period)
    deformed = 0.5 + 0.4 * np.cos(2.0 * np.pi * (x + shift) / period)

    explicit = phase_shifted_sampling_moire(
        reference,
        deformed,
        period,
        axis="x",
        unwrap_axis="x",
    )
    legacy = analyze(reference, deformed, period, axis="x", unwrap_axis="x")

    np.testing.assert_allclose(explicit.displacement, legacy.displacement)


def test_analyze_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        analyze(np.zeros((8, 8)), np.zeros((8, 9)), period=4)


def test_strain_field_returns_constant_small_strain():
    _y, x = np.mgrid[:8, :10]
    u = 0.2 + 750e-6 * x

    strain = strain_field(u)

    np.testing.assert_allclose(strain.exx, 750e-6, atol=1e-14)
    assert strain.eyy is None
    assert strain.gamma_xy is None


def test_separate_grid_components_reduces_cross_component():
    y, x = np.mgrid[:40, :48]
    period = 8
    square_grid = np.where((x % period < 3) & (y % period < 3), 0.0, 1.0)

    x_component, y_component = separate_grid_components(square_grid, period=period)

    center = np.s_[period:-period, period:-period]
    x_center = x_component[center]
    y_center = y_component[center]
    assert np.std(np.mean(x_center, axis=0)) > 0.08
    assert np.std(np.mean(x_center, axis=1)) < 0.02
    assert np.std(np.mean(y_center, axis=1)) > 0.08
    assert np.std(np.mean(y_center, axis=0)) < 0.02


def test_recommended_strain_smoothing_window_is_odd_and_period_scaled():
    assert recommended_strain_smoothing_window(8, cycles=8) == 65
    assert recommended_strain_smoothing_window(7, cycles=4) == 29
    with pytest.raises(ValueError, match="period"):
        recommended_strain_smoothing_window(0)


def test_synthetic_microstrain_experiment_is_analyzable():
    experiment = make_microstrain_experiment(
        shape=(96, 128),
        period=8,
        strain_xx=500e-6,
        rigid_shift=0.7,
        noise_std=0.0,
    )

    result = analyze(
        experiment.reference_image,
        experiment.deformed_image,
        period=experiment.period,
        axis="x",
        unwrap_axis="x",
    )

    central = np.s_[:, experiment.period : -experiment.period]
    np.testing.assert_allclose(
        result.displacement[central],
        experiment.true_u[central],
        atol=0.03,
    )


def test_two_direction_synthetic_experiment_recovers_strain_components():
    experiment = make_strain_distribution_experiment(
        shape=(96, 128),
        period=8,
        strain_xx=700e-6,
        strain_yy=-300e-6,
        shear_xy=400e-6,
        rigid_shift=(0.5, -0.25),
        noise_std=0.0,
    )

    result_x = analyze(
        experiment.reference_x,
        experiment.deformed_x,
        period=experiment.period,
        axis="x",
        unwrap_axis="x",
    )
    result_y = analyze(
        experiment.reference_y,
        experiment.deformed_y,
        period=experiment.period,
        axis="y",
        unwrap_axis="y",
    )
    strain = strain_field(result_x.displacement, result_y.displacement)

    central = np.s_[experiment.period : -experiment.period, experiment.period : -experiment.period]
    np.testing.assert_allclose(strain.exx[central], experiment.true_exx[central], atol=4e-5)
    np.testing.assert_allclose(strain.eyy[central], experiment.true_eyy[central], atol=4e-5)
    np.testing.assert_allclose(
        strain.gamma_xy[central],
        experiment.true_gamma_xy[central],
        atol=8e-5,
    )


def test_analyze_grid_returns_two_displacement_components():
    experiment = make_strain_distribution_experiment(
        shape=(96, 128),
        period=8,
        strain_xx=500e-6,
        strain_yy=-200e-6,
        shear_xy=300e-6,
        rigid_shift=(0.4, -0.2),
        noise_std=0.0,
    )
    reference_grid = 0.5 * (experiment.reference_x + experiment.reference_y)
    deformed_grid = 0.5 * (experiment.deformed_x + experiment.deformed_y)

    result = analyze_grid(
        reference_grid,
        deformed_grid,
        period=experiment.period,
        strain_smooth_window=1,
    )

    assert result.x.displacement.shape == reference_grid.shape
    assert result.y.displacement.shape == reference_grid.shape
    assert result.strain.exx.shape == reference_grid.shape


def test_microstrain_square_grid_is_analyzable_in_valid_roi():
    experiment = make_microstrain_square_grid(
        shape=(128, 160),
        period=8,
        strain_xx=300e-6,
        strain_yy=-150e-6,
        shear_xy=100e-6,
        rigid_shift=(0.5, -0.25),
        supersample=32,
        noise_std=0.0,
    )

    result = analyze_grid(
        experiment.reference,
        experiment.deformed,
        period=experiment.period,
        strain_smooth_window=49,
    )

    roi = np.s_[3 * experiment.period : -3 * experiment.period, 3 * experiment.period : -3 * experiment.period]
    assert np.mean(np.abs(result.x.displacement[roi] - experiment.true_u[roi])) < 0.08
    assert np.mean(np.abs(result.y.displacement[roi] - experiment.true_v[roi])) < 0.08
    assert np.mean(np.abs(result.strain.exx[roi] - experiment.true_exx[roi])) < 1e-3
    assert np.mean(np.abs(result.strain.eyy[roi] - experiment.true_eyy[roi])) < 1e-3


def test_analyze_with_skimage_warped_noisy_grating():
    util = pytest.importorskip("skimage.util")

    height, width = 96, 128
    period = 8
    shift = 0.75
    _y, x = np.mgrid[:height, :width]
    reference = 0.5 + 0.45 * np.cos(2.0 * np.pi * x / period)
    deformed = 0.5 + 0.45 * np.cos(2.0 * np.pi * (x + shift) / period)
    deformed = util.random_noise(deformed, mode="gaussian", var=1e-5, rng=0)

    result = analyze(reference, deformed, period=period, axis="x", unwrap_axis="x")

    central = result.displacement[:, period:-period]
    assert np.mean(central) == pytest.approx(shift, abs=0.03)
