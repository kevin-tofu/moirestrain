from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Axis = Literal["x", "y", 0, 1]


@dataclass(frozen=True)
class AnalysisResult:
    """Result returned by :func:`analyze`."""

    reference_phase: np.ndarray
    deformed_phase: np.ndarray
    phase_difference: np.ndarray
    displacement: np.ndarray


@dataclass(frozen=True)
class StrainResult:
    """Small-strain components calculated from displacement fields."""

    exx: np.ndarray
    eyy: np.ndarray | None
    gamma_xy: np.ndarray | None


@dataclass(frozen=True)
class GridAnalysisResult:
    """Two-direction grid analysis result."""

    x: AnalysisResult
    y: AnalysisResult
    strain: StrainResult
    reference_x_component: np.ndarray
    reference_y_component: np.ndarray
    deformed_x_component: np.ndarray
    deformed_y_component: np.ndarray


def _axis_index(axis: Axis) -> int:
    if axis in ("y", 0):
        return 0
    if axis in ("x", 1):
        return 1
    raise ValueError("axis must be 'x', 'y', 0, or 1")


def _as_2d_image(image: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if min(array.shape) < 2:
        raise ValueError(f"{name} must have at least two pixels in each direction")
    return array


def _smooth_box(image: np.ndarray, window: int | tuple[int, int]) -> np.ndarray:
    if isinstance(window, tuple):
        if len(window) != 2:
            raise ValueError("smooth_window tuple must be (height, width)")
        wy, wx = window
    else:
        wy = wx = window
    wy = int(wy)
    wx = int(wx)
    if wy < 1 or wx < 1:
        raise ValueError("smooth_window values must be positive")
    if wy == 1 and wx == 1:
        return image

    smoothed = image
    if wy > 1:
        kernel_y = np.ones(wy, dtype=float) / wy
        pad_y = (wy // 2, wy - 1 - wy // 2)
        padded = np.pad(smoothed, (pad_y, (0, 0)), mode="reflect")
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel_y, mode="valid"),
            axis=0,
            arr=padded,
        )
    if wx > 1:
        kernel_x = np.ones(wx, dtype=float) / wx
        pad_x = (wx // 2, wx - 1 - wx // 2)
        padded = np.pad(smoothed, ((0, 0), pad_x), mode="reflect")
        smoothed = np.apply_along_axis(
            lambda row: np.convolve(row, kernel_x, mode="valid"),
            axis=1,
            arr=padded,
        )
    return smoothed


def smooth_axis(image: np.ndarray, window: int, axis: Axis) -> np.ndarray:
    """Smooth a 2D image along one axis using a box filter."""

    source = _as_2d_image(image, "image")
    axis_index = _axis_index(axis)
    window = int(window)
    if window < 1:
        raise ValueError("window must be positive")
    if window == 1:
        return source.copy()
    kernel = np.ones(window, dtype=float) / window
    pad = (window // 2, window - 1 - window // 2)
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis_index] = pad
    padded = np.pad(source, pad_width, mode="reflect")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"),
        axis=axis_index,
        arr=padded,
    )


def separate_grid_components(
    image: np.ndarray,
    *,
    period: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Separate a square 2D grid target into x/y grating components.

    Smoothing along y suppresses row-wise markers and leaves the x-periodic
    component. Smoothing along x suppresses column-wise markers and leaves the
    y-periodic component.
    """

    if period < 1:
        raise ValueError("period must be positive")
    return smooth_axis(image, period, axis="y"), smooth_axis(image, period, axis="x")


def recommended_strain_smoothing_window(
    period: int,
    *,
    cycles: float = 8.0,
    minimum: int = 9,
) -> int:
    """Return an odd smoothing window for strain estimation.

    Square-marker targets contain strong harmonics, and displacement fields can
    retain pixel-periodic ripple after phase analysis. Strain calculation
    differentiates displacement, so this ripple should be smoothed before
    gradients are taken. ``cycles`` controls the smoothing length in grating
    periods. Larger values reduce ripple but lower spatial resolution.
    """

    if period < 1:
        raise ValueError("period must be positive")
    if cycles <= 0:
        raise ValueError("cycles must be positive")
    if minimum < 1:
        raise ValueError("minimum must be positive")
    window = max(int(round(period * cycles)), int(minimum))
    if window % 2 == 0:
        window += 1
    return window


def phase_shifted_stack(image: np.ndarray, period: int, axis: Axis = "x") -> np.ndarray:
    """Generate phase-shifted moire images from one grating image.

    The image is thinned every ``period`` pixels at each possible offset, then
    linearly interpolated back to the original size. The returned array has
    shape ``(period, height, width)``.
    """

    image = _as_2d_image(image, "image")
    if not isinstance(period, (int, np.integer)) or period < 3:
        raise ValueError("period must be an integer greater than or equal to 3")

    axis_index = _axis_index(axis)
    length = image.shape[axis_index]
    if period > length:
        raise ValueError("period must not exceed the image length along axis")

    coordinates = np.arange(length, dtype=float)
    stack = []
    for offset in range(period):
        sampled = np.arange(offset, length, period, dtype=float)
        if sampled.size < 2:
            raise ValueError("period is too large for interpolation along axis")
        values = np.take(image, sampled.astype(int), axis=axis_index)
        interpolated = np.apply_along_axis(
            lambda row: np.interp(coordinates, sampled, row),
            axis_index,
            values,
        )
        stack.append(interpolated)

    return np.stack(stack, axis=0)


def wrapped_phase(images: np.ndarray) -> np.ndarray:
    """Estimate wrapped phase from equally phase-shifted images.

    ``images`` must have shape ``(n_shifts, height, width)``. For synthetic
    data ``I_k = a + b * cos(phi + 2*pi*k/n_shifts)``, this returns ``phi``
    wrapped to ``[-pi, pi]``.
    """

    stack = np.asarray(images, dtype=float)
    if stack.ndim != 3:
        raise ValueError("images must have shape (n_shifts, height, width)")

    n_shifts = stack.shape[0]
    if n_shifts < 3:
        raise ValueError("at least three phase shifts are required")

    shifts = 2.0 * np.pi * np.arange(n_shifts, dtype=float) / n_shifts
    cos_terms = np.cos(shifts)[:, None, None]
    sin_terms = np.sin(shifts)[:, None, None]
    real = np.sum(stack * cos_terms, axis=0)
    imag = -np.sum(stack * sin_terms, axis=0)
    return np.arctan2(imag, real)


def unwrap_phase(phase: np.ndarray, axis: Axis | None = None) -> np.ndarray:
    """Unwrap phase along one axis or both image axes."""

    unwrapped = np.asarray(phase, dtype=float)
    if unwrapped.ndim != 2:
        raise ValueError("phase must be a 2D array")

    if axis is None:
        return np.unwrap(np.unwrap(unwrapped, axis=0), axis=1)
    return np.unwrap(unwrapped, axis=_axis_index(axis))


def displacement(
    reference_phase: np.ndarray,
    deformed_phase: np.ndarray,
    grating_pitch: float,
    *,
    unwrap: bool = True,
    unwrap_axis: Axis | None = None,
) -> np.ndarray:
    """Calculate displacement from phase difference.

    The sign convention is ``u = (phase_deformed - phase_reference) * pitch / 2pi``.
    Use a negative ``grating_pitch`` if your imaging setup uses the opposite
    phase-to-displacement convention.
    """

    ref = np.asarray(reference_phase, dtype=float)
    deformed = np.asarray(deformed_phase, dtype=float)
    if ref.shape != deformed.shape:
        raise ValueError("reference_phase and deformed_phase must have the same shape")
    if grating_pitch == 0:
        raise ValueError("grating_pitch must be non-zero")

    diff = deformed - ref
    if unwrap:
        diff = unwrap_phase(diff, axis=unwrap_axis)
    return diff * float(grating_pitch) / (2.0 * np.pi)


def strain_field(
    u: np.ndarray,
    v: np.ndarray | None = None,
    *,
    spacing: float | tuple[float, float] = 1.0,
    smooth_window: int | tuple[int, int] = 1,
) -> StrainResult:
    """Calculate small-strain components from displacement fields.

    Parameters
    ----------
    u:
        x-direction displacement field.
    v:
        Optional y-direction displacement field. When omitted, only ``exx`` is
        returned and ``eyy`` / ``gamma_xy`` are ``None``.
    spacing:
        Pixel spacing as a scalar or ``(dy, dx)`` tuple. The returned strain is
        displacement unit divided by spacing unit.
    smooth_window:
        Optional box smoothing window applied before differentiation. This is
        useful because strain estimation differentiates displacement noise.
    """

    u_array = _as_2d_image(u, "u")
    if isinstance(spacing, tuple):
        if len(spacing) != 2:
            raise ValueError("spacing tuple must be (dy, dx)")
        dy, dx = map(float, spacing)
    else:
        dy = dx = float(spacing)
    if dy == 0 or dx == 0:
        raise ValueError("spacing values must be non-zero")

    u_smooth = _smooth_box(u_array, smooth_window)
    edge_order = 2 if min(u_smooth.shape) > 2 else 1
    du_dy, du_dx = np.gradient(u_smooth, dy, dx, edge_order=edge_order)
    if v is None:
        return StrainResult(exx=du_dx, eyy=None, gamma_xy=None)

    v_array = _as_2d_image(v, "v")
    if v_array.shape != u_array.shape:
        raise ValueError("u and v must have the same shape")

    v_smooth = _smooth_box(v_array, smooth_window)
    dv_dy, dv_dx = np.gradient(v_smooth, dy, dx, edge_order=edge_order)
    return StrainResult(exx=du_dx, eyy=dv_dy, gamma_xy=du_dy + dv_dx)


def analyze(
    reference_image: np.ndarray,
    deformed_image: np.ndarray,
    period: int,
    *,
    axis: Axis = "x",
    grating_pitch: float | None = None,
    unwrap_axis: Axis | None = None,
) -> AnalysisResult:
    """Run the sampling moire pipeline for a reference/deformed image pair."""

    reference = _as_2d_image(reference_image, "reference_image")
    deformed = _as_2d_image(deformed_image, "deformed_image")
    if reference.shape != deformed.shape:
        raise ValueError("reference_image and deformed_image must have the same shape")

    pitch = float(period if grating_pitch is None else grating_pitch)
    phase_ref = unwrap_phase(
        wrapped_phase(phase_shifted_stack(reference, period, axis=axis)),
        axis=unwrap_axis,
    )
    phase_def = unwrap_phase(
        wrapped_phase(phase_shifted_stack(deformed, period, axis=axis)),
        axis=unwrap_axis,
    )
    phase_diff = unwrap_phase(phase_def - phase_ref, axis=unwrap_axis)
    disp = phase_diff * pitch / (2.0 * np.pi)

    return AnalysisResult(
        reference_phase=phase_ref,
        deformed_phase=phase_def,
        phase_difference=phase_diff,
        displacement=disp,
    )


def phase_shifted_sampling_moire(
    reference_image: np.ndarray,
    deformed_image: np.ndarray,
    period: int,
    *,
    axis: Axis = "x",
    grating_pitch: float | None = None,
    unwrap_axis: Axis | None = None,
) -> AnalysisResult:
    """Analyze displacement with the phase-shifted sampling moire method.

    This is an explicit alias for :func:`analyze`. Internally it generates
    ``period`` phase-shifted moire images by shifting the sampling offset and
    estimates the wrapped phase using the phase-shifting/DFT formula in
    :func:`wrapped_phase`.
    """

    return analyze(
        reference_image,
        deformed_image,
        period,
        axis=axis,
        grating_pitch=grating_pitch,
        unwrap_axis=unwrap_axis,
    )


def analyze_grid(
    reference_image: np.ndarray,
    deformed_image: np.ndarray,
    period: int,
    *,
    grating_pitch: float | None = None,
    strain_spacing: float | tuple[float, float] = 1.0,
    strain_smooth_window: int | tuple[int, int] = 1,
) -> GridAnalysisResult:
    """Analyze a two-direction square-marker grid image pair.

    The x/y grating components are first separated by directional smoothing.
    Each component is then analyzed with the phase-shifted sampling moire
    method, and small-strain components are calculated from the resulting
    displacement fields.
    """

    reference_x, reference_y = separate_grid_components(reference_image, period=period)
    deformed_x, deformed_y = separate_grid_components(deformed_image, period=period)
    x_result = phase_shifted_sampling_moire(
        reference_x,
        deformed_x,
        period,
        axis="x",
        grating_pitch=grating_pitch,
        unwrap_axis="x",
    )
    y_result = phase_shifted_sampling_moire(
        reference_y,
        deformed_y,
        period,
        axis="y",
        grating_pitch=grating_pitch,
        unwrap_axis="y",
    )
    strain = strain_field(
        x_result.displacement,
        y_result.displacement,
        spacing=strain_spacing,
        smooth_window=strain_smooth_window,
    )
    return GridAnalysisResult(
        x=x_result,
        y=y_result,
        strain=strain,
        reference_x_component=reference_x,
        reference_y_component=reference_y,
        deformed_x_component=deformed_x,
        deformed_y_component=deformed_y,
    )
