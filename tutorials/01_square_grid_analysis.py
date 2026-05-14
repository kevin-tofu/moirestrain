from pathlib import Path

import numpy as np

from moirestrain import (
    analyze_grid,
    apply_valid_mask,
    inner_valid_mask,
    make_microstrain_square_grid,
    recommended_strain_smoothing_window,
    save_analysis_npz,
    save_grid_analysis_figure,
)


def main() -> None:
    output_dir = Path("data/tutorials")
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = make_microstrain_square_grid(
        shape=(160, 192),
        period=16,
        strain_xx=1500e-6,
        strain_yy=-600e-6,
        shear_xy=450e-6,
        supersample=32,
        noise_std=0.0,
    )
    strain_window = recommended_strain_smoothing_window(experiment.period, cycles=3)
    result = analyze_grid(
        experiment.reference,
        experiment.deformed,
        period=experiment.period,
        strain_smooth_window=strain_window,
    )
    valid_mask = inner_valid_mask(experiment.reference.shape, margin=3 * experiment.period)

    save_analysis_npz(
        output_dir / "01_square_grid_result.npz",
        u=result.x.displacement,
        v=result.y.displacement,
        exx=result.strain.exx,
        eyy=result.strain.eyy,
        gamma_xy=result.strain.gamma_xy,
        valid_mask=valid_mask,
        reference=experiment.reference,
        deformed=experiment.deformed,
        extra_arrays={
            "true_exx": experiment.true_exx,
            "true_eyy": experiment.true_eyy,
            "true_gamma_xy": experiment.true_gamma_xy,
        },
    )
    save_grid_analysis_figure(
        output_dir / "01_square_grid_summary.png",
        reference=experiment.reference,
        deformed=experiment.deformed,
        u=result.x.displacement,
        exx=result.strain.exx,
        eyy=result.strain.eyy,
        gamma_xy=result.strain.gamma_xy,
        valid_mask=valid_mask,
    )

    def mae(array: np.ndarray) -> float:
        return float(np.nanmean(np.abs(apply_valid_mask(array, valid_mask))))

    print("wrote: data/tutorials/01_square_grid_result.npz")
    print("wrote: data/tutorials/01_square_grid_summary.png")
    print(f"mean |exx error|: {mae(result.strain.exx - experiment.true_exx):.6e}")
    print(f"mean |eyy error|: {mae(result.strain.eyy - experiment.true_eyy):.6e}")
    print(f"mean |gamma_xy error|: {mae(result.strain.gamma_xy - experiment.true_gamma_xy):.6e}")


if __name__ == "__main__":
    main()

