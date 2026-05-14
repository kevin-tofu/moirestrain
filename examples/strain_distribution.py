import os
from pathlib import Path

import numpy as np

from moirestrain import (
    analyze,
    make_strain_distribution_experiment,
    save_strain_experiment_npz,
    strain_field,
)


def _save_figure(path: Path, fields: dict[str, np.ndarray]) -> None:
    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/moirestrain-matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped PNG visualization")
        return

    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    for ax, (title, field) in zip(axes.ravel(), fields.items()):
        image = ax.imshow(field, cmap="coolwarm")
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, shrink=0.75)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    experiment = make_strain_distribution_experiment(
        shape=(160, 192),
        period=8,
        strain_xx=800e-6,
        strain_yy=-250e-6,
        shear_xy=350e-6,
        rigid_shift=(0.6, -0.35),
        noise_std=0.008,
        seed=4,
    )
    save_strain_experiment_npz(output_dir / "synthetic_strain_distribution_input.npz", experiment)

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
    strain = strain_field(result_x.displacement, result_y.displacement, smooth_window=17)

    np.savez_compressed(
        output_dir / "synthetic_strain_distribution_result.npz",
        u=result_x.displacement,
        v=result_y.displacement,
        exx=strain.exx,
        eyy=strain.eyy,
        gamma_xy=strain.gamma_xy,
        true_u=experiment.true_u,
        true_v=experiment.true_v,
        true_exx=experiment.true_exx,
        true_eyy=experiment.true_eyy,
        true_gamma_xy=experiment.true_gamma_xy,
    )

    margin = experiment.period
    roi = np.s_[margin:-margin, margin:-margin]
    print("strain distribution metrics in central ROI")
    print(f"mean |u error|:        {np.mean(np.abs(result_x.displacement[roi] - experiment.true_u[roi])):.5f} px")
    print(f"mean |v error|:        {np.mean(np.abs(result_y.displacement[roi] - experiment.true_v[roi])):.5f} px")
    print(f"mean |exx error|:      {np.mean(np.abs(strain.exx[roi] - experiment.true_exx[roi])):.6e}")
    print(f"mean |eyy error|:      {np.mean(np.abs(strain.eyy[roi] - experiment.true_eyy[roi])):.6e}")
    print(
        "mean |gamma_xy error|: "
        f"{np.mean(np.abs(strain.gamma_xy[roi] - experiment.true_gamma_xy[roi])):.6e}"
    )

    _save_figure(
        output_dir / "synthetic_strain_distribution.png",
        {
            "u [px]": result_x.displacement,
            "v [px]": result_y.displacement,
            "exx": strain.exx,
            "eyy": strain.eyy,
            "gamma_xy": strain.gamma_xy,
            "true gamma_xy": experiment.true_gamma_xy,
        },
    )


if __name__ == "__main__":
    main()
