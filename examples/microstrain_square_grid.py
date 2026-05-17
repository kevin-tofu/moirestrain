from pathlib import Path

import numpy as np

from moirestrain import (
    analyze_grid,
    apply_valid_mask,
    inner_valid_mask,
    make_microstrain_square_grid,
    recommended_strain_smoothing_window,
    robust_limits,
    save_square_grid_experiment_npz,
)


def _save_figure(path: Path, fields: dict[str, np.ndarray]) -> None:
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/moirestrain-matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped PNG visualization")
        return

    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, (title, field) in zip(axes.ravel(), fields.items()):
        cmap = "gray" if "grid" in title else "coolwarm"
        kwargs = {}
        if cmap == "coolwarm":
            kwargs["vmin"], kwargs["vmax"] = robust_limits(field)
        image = ax.imshow(field, cmap=cmap, **kwargs)
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, shrink=0.72)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    experiment = make_microstrain_square_grid(
        shape=(160, 192),
        period=8,
        strain_xx=500e-6,
        strain_yy=-200e-6,
        shear_xy=150e-6,
        rigid_shift=(0.6, -0.35),
        supersample=64,
        noise_std=0.0,
        seed=4,
    )
    strain_window = recommended_strain_smoothing_window(experiment.period, cycles=8)
    result = analyze_grid(
        experiment.reference,
        experiment.deformed,
        period=experiment.period,
        strain_smooth_window=strain_window,
    )
    valid_mask = inner_valid_mask(experiment.reference.shape, margin=5 * experiment.period)
    save_square_grid_experiment_npz(output_dir / "microstrain_square_grid_input.npz", experiment)
    np.savez_compressed(
        output_dir / "microstrain_square_grid_result.npz",
        u=result.x.displacement,
        v=result.y.displacement,
        exx=result.strain.exx,
        eyy=result.strain.eyy,
        gamma_xy=result.strain.gamma_xy,
        valid_mask=valid_mask,
        true_u=experiment.true_u,
        true_v=experiment.true_v,
        true_exx=experiment.true_exx,
        true_eyy=experiment.true_eyy,
        true_gamma_xy=experiment.true_gamma_xy,
    )

    def valid_view(array: np.ndarray) -> np.ndarray:
        return apply_valid_mask(array, valid_mask)

    _save_figure(
        output_dir / "microstrain_square_grid.png",
        {
            "reference grid": experiment.reference,
            "deformed grid": experiment.deformed,
            "u valid ROI": valid_view(result.x.displacement),
            "exx error valid ROI": valid_view(result.strain.exx - experiment.true_exx),
            "eyy error valid ROI": valid_view(result.strain.eyy - experiment.true_eyy),
            "gamma_xy error valid ROI": valid_view(
                result.strain.gamma_xy - experiment.true_gamma_xy
            ),
        },
    )

    print("wrote: data/microstrain_square_grid_input.npz")
    print("wrote: data/microstrain_square_grid_result.npz")
    print("wrote: data/microstrain_square_grid.png")
    print(f"strain smoothing window: {strain_window} px")
    print(
        "mean |u error|:        "
        f"{np.nanmean(np.abs(apply_valid_mask(result.x.displacement - experiment.true_u, valid_mask))):.6e} px"
    )
    print(
        "mean |v error|:        "
        f"{np.nanmean(np.abs(apply_valid_mask(result.y.displacement - experiment.true_v, valid_mask))):.6e} px"
    )
    print(
        "mean |exx error|:      "
        f"{np.nanmean(np.abs(apply_valid_mask(result.strain.exx - experiment.true_exx, valid_mask))):.6e}"
    )
    print(
        "mean |eyy error|:      "
        f"{np.nanmean(np.abs(apply_valid_mask(result.strain.eyy - experiment.true_eyy, valid_mask))):.6e}"
    )
    print(
        "mean |gamma_xy error|: "
        f"{np.nanmean(np.abs(apply_valid_mask(result.strain.gamma_xy - experiment.true_gamma_xy, valid_mask))):.6e}"
    )


if __name__ == "__main__":
    main()
