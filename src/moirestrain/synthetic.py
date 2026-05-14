from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SyntheticExperiment:
    """Synthetic grating image pair with ground-truth displacement."""

    reference_image: np.ndarray
    deformed_image: np.ndarray
    true_u: np.ndarray
    true_exx: np.ndarray
    period: int
    strain_xx: float
    rigid_shift: float
    noise_std: float


@dataclass(frozen=True)
class SyntheticStrainExperiment:
    """Synthetic two-direction grating data with ground-truth strain."""

    reference_x: np.ndarray
    deformed_x: np.ndarray
    reference_y: np.ndarray
    deformed_y: np.ndarray
    true_u: np.ndarray
    true_v: np.ndarray
    true_exx: np.ndarray
    true_eyy: np.ndarray
    true_gamma_xy: np.ndarray
    period: int
    strain_xx: float
    strain_yy: float
    shear_xy: float
    rigid_shift: tuple[float, float]
    noise_std: float


@dataclass(frozen=True)
class SyntheticSquareGridExperiment:
    """Synthetic square-marker grid pair with ground-truth strain."""

    reference: np.ndarray
    deformed: np.ndarray
    true_u: np.ndarray
    true_v: np.ndarray
    true_exx: np.ndarray
    true_eyy: np.ndarray
    true_gamma_xy: np.ndarray
    period: int
    marker_size: int
    strain_xx: float
    strain_yy: float
    shear_xy: float
    rigid_shift: tuple[float, float]
    noise_std: float


def _smooth_axis_local(image: np.ndarray, window: int, axis: int) -> np.ndarray:
    if window <= 1:
        return image
    kernel = np.ones(window, dtype=float) / window
    pad = (window // 2, window - 1 - window // 2)
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = pad
    padded = np.pad(image, pad_width, mode="reflect")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"),
        axis=axis,
        arr=padded,
    )


def _box_blur_local(image: np.ndarray, window: int) -> np.ndarray:
    return _smooth_axis_local(_smooth_axis_local(image, window, axis=0), window, axis=1)


def make_microstrain_square_grid(
    *,
    shape: tuple[int, int] = (160, 192),
    period: int = 8,
    marker_size: int | None = None,
    strain_xx: float = 500e-6,
    strain_yy: float = -200e-6,
    shear_xy: float = 150e-6,
    rigid_shift: tuple[float, float] = (0.6, -0.35),
    supersample: int = 8,
    blur_window: int = 3,
    noise_std: float = 0.0,
    seed: int | None = 0,
) -> SyntheticSquareGridExperiment:
    """Create a fronto-parallel square-marker grid with known microstrain.

    The target geometry is white background with black square markers on a
    regular grid. By default, ``marker_size = period / 2``, so the black square
    width equals the white gap width. The captured image is simulated by
    supersampling each pixel, then applying an optional box blur.
    """

    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    height, width = map(int, shape)
    if height < 8 or width < 8:
        raise ValueError("shape must be at least (8, 8)")
    if period < 4:
        raise ValueError("period must be greater than or equal to 4")
    if marker_size is None:
        if period % 2 != 0:
            raise ValueError("period must be even when marker_size is omitted")
        marker_size = period // 2
    marker_size = int(marker_size)
    if marker_size <= 0 or marker_size >= period:
        raise ValueError("marker_size must be between 1 and period - 1")
    if supersample < 1:
        raise ValueError("supersample must be positive")
    if blur_window < 1:
        raise ValueError("blur_window must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if len(rigid_shift) != 2:
        raise ValueError("rigid_shift must be (u0, v0)")

    y, x = np.mgrid[:height, :width]
    xc = x - 0.5 * (width - 1)
    yc = y - 0.5 * (height - 1)
    u0, v0 = map(float, rigid_shift)
    true_u = u0 + strain_xx * xc + 0.5 * shear_xy * yc
    true_v = v0 + strain_yy * yc + 0.5 * shear_xy * xc

    def target(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        x_mod = np.mod(xx, period)
        y_mod = np.mod(yy, period)
        black = (x_mod < marker_size) & (y_mod < marker_size)
        return np.where(black, 0.05, 0.95)

    offsets = (np.arange(supersample, dtype=float) + 0.5) / supersample - 0.5

    def camera_sample(u: np.ndarray | float, v: np.ndarray | float) -> np.ndarray:
        image = np.zeros(shape, dtype=float)
        for oy in offsets:
            for ox in offsets:
                image += target(x + ox + u, y + oy + v)
        image = image / float(supersample * supersample)
        return _box_blur_local(image, blur_window)

    reference = camera_sample(0.0, 0.0)
    deformed = camera_sample(true_u, true_v)
    if noise_std:
        rng = np.random.default_rng(seed)
        reference = reference + rng.normal(scale=noise_std, size=shape)
        deformed = deformed + rng.normal(scale=noise_std, size=shape)

    return SyntheticSquareGridExperiment(
        reference=np.clip(reference, 0.0, 1.0),
        deformed=np.clip(deformed, 0.0, 1.0),
        true_u=true_u,
        true_v=true_v,
        true_exx=np.full(shape, strain_xx, dtype=float),
        true_eyy=np.full(shape, strain_yy, dtype=float),
        true_gamma_xy=np.full(shape, shear_xy, dtype=float),
        period=period,
        marker_size=marker_size,
        strain_xx=float(strain_xx),
        strain_yy=float(strain_yy),
        shear_xy=float(shear_xy),
        rigid_shift=(u0, v0),
        noise_std=float(noise_std),
    )


def save_square_grid_experiment_npz(
    path: str | Path, experiment: SyntheticSquareGridExperiment
) -> None:
    """Save a square-marker grid experiment to ``.npz``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        reference=experiment.reference,
        deformed=experiment.deformed,
        true_u=experiment.true_u,
        true_v=experiment.true_v,
        true_exx=experiment.true_exx,
        true_eyy=experiment.true_eyy,
        true_gamma_xy=experiment.true_gamma_xy,
        period=np.array(experiment.period, dtype=np.int64),
        marker_size=np.array(experiment.marker_size, dtype=np.int64),
        strain_xx=np.array(experiment.strain_xx, dtype=float),
        strain_yy=np.array(experiment.strain_yy, dtype=float),
        shear_xy=np.array(experiment.shear_xy, dtype=float),
        rigid_shift=np.asarray(experiment.rigid_shift, dtype=float),
        noise_std=np.array(experiment.noise_std, dtype=float),
    )


def make_microstrain_experiment(
    *,
    shape: tuple[int, int] = (128, 160),
    period: int = 8,
    strain_xx: float = 800e-6,
    rigid_shift: float = 0.6,
    noise_std: float = 0.01,
    contrast: float = 0.42,
    background: float = 0.5,
    illumination_gradient: float = 0.15,
    seed: int | None = 0,
) -> SyntheticExperiment:
    """Create an experimental-looking grating image pair.

    The deformed image contains a known x-direction displacement field
    ``u = rigid_shift + strain_xx * (x - x_center)``. Images are normalized to
    roughly ``[0, 1]`` and include smooth illumination variation plus Gaussian
    sensor noise.
    """

    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    height, width = shape
    if height < 8 or width < 8:
        raise ValueError("shape must be at least (8, 8)")
    if period < 3:
        raise ValueError("period must be greater than or equal to 3")
    if contrast <= 0:
        raise ValueError("contrast must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width]
    x_center = 0.5 * (width - 1)
    x_unit = (x - x_center) / max(x_center, 1.0)
    y_unit = (y - 0.5 * (height - 1)) / max(0.5 * (height - 1), 1.0)

    illumination = 1.0 + illumination_gradient * (0.65 * x_unit - 0.35 * y_unit)
    phase_reference = 2.0 * np.pi * x / period
    true_u = rigid_shift + strain_xx * (x - x_center)
    true_exx = np.full(shape, strain_xx, dtype=float)
    phase_deformed = 2.0 * np.pi * (x + true_u) / period

    reference = background + contrast * illumination * np.cos(phase_reference)
    deformed = background + contrast * illumination * np.cos(phase_deformed)
    if noise_std:
        reference = reference + rng.normal(scale=noise_std, size=shape)
        deformed = deformed + rng.normal(scale=noise_std, size=shape)

    reference = np.clip(reference, 0.0, 1.0)
    deformed = np.clip(deformed, 0.0, 1.0)
    return SyntheticExperiment(
        reference_image=reference,
        deformed_image=deformed,
        true_u=true_u,
        true_exx=true_exx,
        period=period,
        strain_xx=float(strain_xx),
        rigid_shift=float(rigid_shift),
        noise_std=float(noise_std),
    )


def save_experiment_npz(path: str | Path, experiment: SyntheticExperiment) -> None:
    """Save a synthetic experiment to a compressed ``.npz`` file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        reference_image=experiment.reference_image,
        deformed_image=experiment.deformed_image,
        true_u=experiment.true_u,
        true_exx=experiment.true_exx,
        period=np.array(experiment.period, dtype=np.int64),
        strain_xx=np.array(experiment.strain_xx, dtype=float),
        rigid_shift=np.array(experiment.rigid_shift, dtype=float),
        noise_std=np.array(experiment.noise_std, dtype=float),
    )


def make_strain_distribution_experiment(
    *,
    shape: tuple[int, int] = (160, 192),
    period: int = 8,
    strain_xx: float = 800e-6,
    strain_yy: float = -250e-6,
    shear_xy: float = 350e-6,
    rigid_shift: tuple[float, float] = (0.6, -0.35),
    noise_std: float = 0.008,
    contrast: float = 0.42,
    background: float = 0.5,
    illumination_gradient: float = 0.12,
    seed: int | None = 1,
) -> SyntheticStrainExperiment:
    """Create x/y grating images for full small-strain recovery.

    The displacement field is linear:

    ``u = u0 + exx * x + 0.5 * gamma_xy * y``
    ``v = v0 + eyy * y + 0.5 * gamma_xy * x``

    Coordinates are centered, so the rigid shift is easy to interpret.
    """

    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    height, width = shape
    if height < 8 or width < 8:
        raise ValueError("shape must be at least (8, 8)")
    if period < 3:
        raise ValueError("period must be greater than or equal to 3")
    if len(rigid_shift) != 2:
        raise ValueError("rigid_shift must be (u0, v0)")
    if contrast <= 0:
        raise ValueError("contrast must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")

    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width]
    xc = x - 0.5 * (width - 1)
    yc = y - 0.5 * (height - 1)
    u0, v0 = map(float, rigid_shift)

    true_u = u0 + strain_xx * xc + 0.5 * shear_xy * yc
    true_v = v0 + strain_yy * yc + 0.5 * shear_xy * xc
    true_exx = np.full(shape, strain_xx, dtype=float)
    true_eyy = np.full(shape, strain_yy, dtype=float)
    true_gamma_xy = np.full(shape, shear_xy, dtype=float)

    x_unit = xc / max(0.5 * (width - 1), 1.0)
    y_unit = yc / max(0.5 * (height - 1), 1.0)
    illumination = 1.0 + illumination_gradient * (0.65 * x_unit - 0.35 * y_unit)

    reference_x = background + contrast * illumination * np.cos(2.0 * np.pi * x / period)
    deformed_x = background + contrast * illumination * np.cos(
        2.0 * np.pi * (x + true_u) / period
    )
    reference_y = background + contrast * illumination * np.cos(2.0 * np.pi * y / period)
    deformed_y = background + contrast * illumination * np.cos(
        2.0 * np.pi * (y + true_v) / period
    )
    if noise_std:
        reference_x = reference_x + rng.normal(scale=noise_std, size=shape)
        deformed_x = deformed_x + rng.normal(scale=noise_std, size=shape)
        reference_y = reference_y + rng.normal(scale=noise_std, size=shape)
        deformed_y = deformed_y + rng.normal(scale=noise_std, size=shape)

    return SyntheticStrainExperiment(
        reference_x=np.clip(reference_x, 0.0, 1.0),
        deformed_x=np.clip(deformed_x, 0.0, 1.0),
        reference_y=np.clip(reference_y, 0.0, 1.0),
        deformed_y=np.clip(deformed_y, 0.0, 1.0),
        true_u=true_u,
        true_v=true_v,
        true_exx=true_exx,
        true_eyy=true_eyy,
        true_gamma_xy=true_gamma_xy,
        period=period,
        strain_xx=float(strain_xx),
        strain_yy=float(strain_yy),
        shear_xy=float(shear_xy),
        rigid_shift=(u0, v0),
        noise_std=float(noise_std),
    )


def save_strain_experiment_npz(
    path: str | Path, experiment: SyntheticStrainExperiment
) -> None:
    """Save a two-direction synthetic strain experiment to ``.npz``."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        reference_x=experiment.reference_x,
        deformed_x=experiment.deformed_x,
        reference_y=experiment.reference_y,
        deformed_y=experiment.deformed_y,
        true_u=experiment.true_u,
        true_v=experiment.true_v,
        true_exx=experiment.true_exx,
        true_eyy=experiment.true_eyy,
        true_gamma_xy=experiment.true_gamma_xy,
        period=np.array(experiment.period, dtype=np.int64),
        strain_xx=np.array(experiment.strain_xx, dtype=float),
        strain_yy=np.array(experiment.strain_yy, dtype=float),
        shear_xy=np.array(experiment.shear_xy, dtype=float),
        rigid_shift=np.asarray(experiment.rigid_shift, dtype=float),
        noise_std=np.array(experiment.noise_std, dtype=float),
    )
