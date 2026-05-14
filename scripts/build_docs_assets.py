from __future__ import annotations

import shutil
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))
    from examples import (
        benchmark_microstrain,
        benchmark_sweep,
        partial_grid_detection_analysis,
        skimage_natural_grating_strain,
    )

    docs_static = root / "docs" / "_static"
    data_dir = root / "data"
    docs_static.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    benchmark_status = benchmark_microstrain.main(
        [
            "--output-dir",
            str(data_dir),
        ]
    )
    if benchmark_status != 0:
        return benchmark_status
    shutil.copyfile(
        data_dir / "benchmark_microstrain_summary.png",
        docs_static / "benchmark_microstrain_summary.png",
    )
    shutil.copyfile(
        data_dir / "benchmark_microstrain_comparison.png",
        docs_static / "benchmark_microstrain_comparison.png",
    )
    benchmark_sweep.main(
        [
            "--output-dir",
            str(data_dir),
            "--shape",
            "160,192",
            "--periods",
            "8,12,16",
            "--blur-windows",
            "3",
            "--noise-stds",
            "0",
            "--strain-presets",
            "micro,demo,stress",
            "--supersample",
            "16",
        ]
    )
    shutil.copyfile(
        data_dir / "benchmark_sweep_period_plot.png",
        docs_static / "benchmark_sweep_period_plot.png",
    )
    partial_grid_detection_analysis.main(
        [
            "--output-dir",
            str(data_dir),
        ]
    )
    shutil.copyfile(
        data_dir / "partial_grid_detection_analysis.png",
        docs_static / "partial_grid_detection_analysis.png",
    )
    shutil.copyfile(
        data_dir / "partial_grid_strain_comparison.png",
        docs_static / "partial_grid_strain_comparison.png",
    )
    shutil.copyfile(
        data_dir / "partial_grid_strain_measured_true.png",
        docs_static / "partial_grid_strain_measured_true.png",
    )
    for name in (
        "partial_grid_exx_comparison.png",
        "partial_grid_eyy_comparison.png",
        "partial_grid_gamma_xy_comparison.png",
    ):
        shutil.copyfile(data_dir / name, docs_static / name)

    skimage_natural_grating_strain.main()
    shutil.copyfile(
        data_dir / "skimage_natural_grating_strain.png",
        docs_static / "natural_grating_strain.png",
    )
    print("wrote: docs/_static/benchmark_microstrain_summary.png")
    print("wrote: docs/_static/benchmark_microstrain_comparison.png")
    print("wrote: docs/_static/benchmark_sweep_period_plot.png")
    print("wrote: docs/_static/partial_grid_detection_analysis.png")
    print("wrote: docs/_static/partial_grid_strain_comparison.png")
    print("wrote: docs/_static/partial_grid_strain_measured_true.png")
    print("wrote: docs/_static/partial_grid_exx_comparison.png")
    print("wrote: docs/_static/partial_grid_eyy_comparison.png")
    print("wrote: docs/_static/partial_grid_gamma_xy_comparison.png")
    print("wrote: docs/_static/natural_grating_strain.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
