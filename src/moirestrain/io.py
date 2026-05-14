from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_grayscale_image(path: str | Path, *, npz_key: str | None = None) -> np.ndarray:
    """Load a grayscale image from ``.npy``, ``.npz``, or common image files."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".npy":
        image = np.load(source)
    elif suffix == ".npz":
        data = np.load(source)
        key = npz_key
        if key is None:
            if len(data.files) != 1:
                raise ValueError(f"{source} contains multiple arrays; specify npz_key")
            key = data.files[0]
        image = data[key]
    else:
        try:
            import imageio.v3 as iio
        except ImportError as exc:
            raise ImportError("imageio is required to read image files") from exc
        image = iio.imread(source)

    array = np.asarray(image, dtype=float)
    if array.ndim == 3:
        if array.shape[2] < 3:
            array = array[..., 0]
        else:
            array = (
                0.2126 * array[..., 0]
                + 0.7152 * array[..., 1]
                + 0.0722 * array[..., 2]
            )
    if array.ndim != 2:
        raise ValueError("loaded image must be 2D grayscale or RGB")
    if array.size == 0:
        raise ValueError("loaded image is empty")
    if np.nanmax(array) > 1.0:
        array = array / 255.0 if np.nanmax(array) <= 255.0 else array / np.nanmax(array)
    return array


def save_grayscale_image(path: str | Path, image: np.ndarray) -> None:
    """Save a grayscale array to ``.npy``, ``.npz``, or a common image file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(image, dtype=float)
    suffix = destination.suffix.lower()
    if suffix == ".npy":
        np.save(destination, array)
        return
    if suffix == ".npz":
        np.savez_compressed(destination, image=array)
        return

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError("imageio is required to write image files") from exc
    finite = np.isfinite(array)
    output = np.zeros(array.shape, dtype=float)
    if np.any(finite):
        finite_values = array[finite]
        low = float(np.nanmin(finite_values))
        high = float(np.nanmax(finite_values))
        if high > low:
            output[finite] = (array[finite] - low) / (high - low)
        else:
            output[finite] = np.clip(finite_values[0], 0.0, 1.0)
    iio.imwrite(destination, np.clip(output * 255.0, 0, 255).astype(np.uint8))


def save_analysis_npz(
    path: str | Path,
    *,
    u: np.ndarray,
    v: np.ndarray,
    exx: np.ndarray,
    eyy: np.ndarray,
    gamma_xy: np.ndarray,
    valid_mask: np.ndarray,
    reference: np.ndarray | None = None,
    deformed: np.ndarray | None = None,
    extra_arrays: Mapping[str, Any] | None = None,
) -> None:
    """Save grid-analysis arrays to a compressed ``.npz`` file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "u": u,
        "v": v,
        "exx": exx,
        "eyy": eyy,
        "gamma_xy": gamma_xy,
        "valid_mask": valid_mask,
    }
    if reference is not None:
        payload["reference"] = reference
    if deformed is not None:
        payload["deformed"] = deformed
    if extra_arrays is not None:
        for key, value in extra_arrays.items():
            payload[key] = np.asarray(value)
    np.savez_compressed(destination, **payload)
