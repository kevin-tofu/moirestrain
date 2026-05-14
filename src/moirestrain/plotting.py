from __future__ import annotations

from pathlib import Path

import numpy as np

from .masking import apply_valid_mask, crop_to_mask, robust_limits


def save_grid_analysis_figure(
    path: str | Path,
    *,
    reference: np.ndarray,
    deformed: np.ndarray,
    u: np.ndarray,
    exx: np.ndarray,
    eyy: np.ndarray,
    gamma_xy: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    """Save a compact PNG summary for grid analysis."""

    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/moirestrain-matplotlib")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save analysis figures") from exc

    def valid_view(array: np.ndarray) -> np.ndarray:
        return crop_to_mask(apply_valid_mask(array, valid_mask), valid_mask)

    fields = {
        "reference": reference,
        "deformed": deformed,
        "u valid ROI": valid_view(u),
        "exx valid ROI": valid_view(exx),
        "eyy valid ROI": valid_view(eyy),
        "gamma_xy valid ROI": valid_view(gamma_xy),
    }
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, (title, field) in zip(axes.ravel(), fields.items()):
        cmap = "gray" if title in {"reference", "deformed"} else "coolwarm"
        kwargs = {}
        if cmap == "coolwarm":
            kwargs["vmin"], kwargs["vmax"] = robust_limits(field)
        image = ax.imshow(field, cmap=cmap, **kwargs)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, shrink=0.72)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def save_grid_truth_comparison_figure(
    path: str | Path,
    *,
    measured: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> None:
    """Save measured/truth/error comparison panels for grid analysis."""

    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/moirestrain-matplotlib")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to save comparison figures") from exc

    names = [name for name in ("u", "v", "exx", "eyy", "gamma_xy") if name in measured and name in truth]
    if not names:
        raise ValueError("measured and truth must share at least one field")

    def valid_view(array: np.ndarray) -> np.ndarray:
        return crop_to_mask(apply_valid_mask(array, valid_mask), valid_mask)

    fig, axes = plt.subplots(len(names), 3, figsize=(10.5, 2.6 * len(names)), constrained_layout=True)
    if len(names) == 1:
        axes = axes[None, :]
    for row, name in enumerate(names):
        measured_view = valid_view(measured[name])
        truth_view = valid_view(truth[name])
        error_view = measured_view - truth_view
        field_limits = robust_limits(np.stack([measured_view, truth_view]))
        error_limits = robust_limits(error_view)
        panels = [
            (f"{name} measured", measured_view, field_limits),
            (f"{name} true", truth_view, field_limits),
            (f"{name} error", error_view, error_limits),
        ]
        for ax, (title, field, limits) in zip(axes[row], panels):
            kwargs = {"vmin": limits[0], "vmax": limits[1]}
            image = ax.imshow(field, cmap="coolwarm", **kwargs)
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, shrink=0.72)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
