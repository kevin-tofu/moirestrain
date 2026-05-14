from pathlib import Path

import numpy as np

from moirestrain import analyze, make_microstrain_experiment, save_experiment_npz, strain_field


def main() -> None:
    experiment = make_microstrain_experiment(
        shape=(128, 160),
        period=8,
        strain_xx=800e-6,
        rigid_shift=0.6,
        noise_std=0.01,
        seed=2,
    )
    output = Path("data/synthetic_microstrain_x.npz")
    save_experiment_npz(output, experiment)

    result = analyze(
        experiment.reference_image,
        experiment.deformed_image,
        period=experiment.period,
        axis="x",
        unwrap_axis="x",
    )
    strain = strain_field(result.displacement, smooth_window=17)
    margin = experiment.period
    roi = np.s_[:, margin:-margin]
    u_mae = np.mean(np.abs(result.displacement[roi] - experiment.true_u[roi]))
    exx_mae = np.mean(np.abs(strain.exx[roi] - experiment.true_exx[roi]))

    print(f"wrote: {output}")
    print(f"mean |u error|:   {u_mae:.5f} px")
    print(f"mean |exx error|: {exx_mae:.6e}")


if __name__ == "__main__":
    main()
