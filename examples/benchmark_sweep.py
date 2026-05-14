from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from moirestrain import (
    analyze_grid,
    apply_valid_mask,
    inner_valid_mask,
    make_microstrain_square_grid,
    recommended_strain_smoothing_window,
)


STRAIN_BASE = {
    "micro": (500e-6, -200e-6, 150e-6),
    "demo": (2000e-6, -800e-6, 600e-6),
    "stress": (5000e-6, -2000e-6, 1500e-6),
}


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_shape(value: str) -> tuple[int, int]:
    height, width = [int(part.strip()) for part in value.split(",")]
    if height < 32 or width < 32:
        raise argparse.ArgumentTypeError("shape values must be at least 32")
    return height, width


def _mae(array: np.ndarray, valid_mask: np.ndarray) -> float:
    return float(np.nanmean(np.abs(apply_valid_mask(array, valid_mask))))


def _valid_margin(shape: tuple[int, int], period: int, strain_window: int) -> int:
    requested = max(5 * period, strain_window // 2)
    largest_nonempty = max(1, min(shape) // 2 - 1)
    return min(requested, largest_nonempty)


def _run_case(
    *,
    shape: tuple[int, int],
    period: int,
    blur_window: int,
    noise_std: float,
    strain_preset: str,
    supersample: int,
    strain_cycles: float,
    seed: int,
) -> dict[str, float | int | str]:
    strain_xx, strain_yy, shear_xy = STRAIN_BASE[strain_preset]
    experiment = make_microstrain_square_grid(
        shape=shape,
        period=period,
        strain_xx=strain_xx,
        strain_yy=strain_yy,
        shear_xy=shear_xy,
        rigid_shift=(0.6, -0.35),
        supersample=supersample,
        blur_window=blur_window,
        noise_std=noise_std,
        seed=seed,
    )
    strain_window = recommended_strain_smoothing_window(period, cycles=strain_cycles)
    valid_margin = _valid_margin(experiment.reference.shape, period, strain_window)
    result = analyze_grid(
        experiment.reference,
        experiment.deformed,
        period=period,
        strain_smooth_window=strain_window,
    )
    valid_mask = inner_valid_mask(experiment.reference.shape, margin=valid_margin)
    return {
        "period": period,
        "blur_window": blur_window,
        "noise_std": noise_std,
        "strain_preset": strain_preset,
        "strain_xx": strain_xx,
        "strain_yy": strain_yy,
        "shear_xy": shear_xy,
        "supersample": supersample,
        "strain_cycles": strain_cycles,
        "strain_window": strain_window,
        "valid_margin": valid_margin,
        "u_mae_px": _mae(result.x.displacement - experiment.true_u, valid_mask),
        "v_mae_px": _mae(result.y.displacement - experiment.true_v, valid_mask),
        "exx_mae": _mae(result.strain.exx - experiment.true_exx, valid_mask),
        "eyy_mae": _mae(result.strain.eyy - experiment.true_eyy, valid_mask),
        "gamma_xy_mae": _mae(result.strain.gamma_xy - experiment.true_gamma_xy, valid_mask),
    }


def _score(row: dict[str, float | int | str]) -> float:
    return float(row["exx_mae"]) + float(row["eyy_mae"]) + float(row["gamma_xy_mae"])


def _write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_markdown(path: Path, rows: list[dict[str, float | int | str]], top_n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    best = sorted(rows, key=_score)[:top_n]
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# Microstrain Benchmark Sweep\n\n")
        handle.write("| rank | preset | period | blur | noise | u MAE px | v MAE px | exx MAE | eyy MAE | gamma MAE |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(best, start=1):
            handle.write(
                "| "
                f"{rank} | {row['strain_preset']} | {row['period']} | {row['blur_window']} | "
                f"{float(row['noise_std']):.3g} | {float(row['u_mae_px']):.3e} | "
                f"{float(row['v_mae_px']):.3e} | {float(row['exx_mae']):.3e} | "
                f"{float(row['eyy_mae']):.3e} | {float(row['gamma_xy_mae']):.3e} |\n"
            )


def _save_plot(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/moirestrain-matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    periods = sorted({int(row["period"]) for row in rows})
    presets = sorted({str(row["strain_preset"]) for row in rows}, key=list(STRAIN_BASE).index)
    fig, axes = plt.subplots(1, len(presets), figsize=(4.2 * len(presets), 3.4), constrained_layout=True)
    if len(presets) == 1:
        axes = [axes]
    for ax, preset in zip(axes, presets):
        subset = [row for row in rows if row["strain_preset"] == preset and float(row["noise_std"]) == 0.0]
        values = []
        for period in periods:
            candidates = [row for row in subset if int(row["period"]) == period]
            values.append(min((_score(row) for row in candidates), default=np.nan))
        ax.plot(periods, np.asarray(values) * 1e6, marker="o")
        ax.set_title(preset)
        ax.set_xlabel("period [px]")
        ax.set_ylabel("strain MAE sum [microstrain]")
        ax.grid(True, alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_sweep(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    rows = []
    for preset in args.strain_presets:
        if preset not in STRAIN_BASE:
            raise ValueError(f"unknown strain preset: {preset}")
        for period in args.periods:
            for blur_window in args.blur_windows:
                for noise_std in args.noise_stds:
                    rows.append(
                        _run_case(
                            shape=args.shape,
                            period=period,
                            blur_window=blur_window,
                            noise_std=noise_std,
                            strain_preset=preset,
                            supersample=args.supersample,
                            strain_cycles=args.strain_cycles,
                            seed=args.seed,
                        )
                    )
                    row = rows[-1]
                    print(
                        "case "
                        f"preset={preset} period={period} blur={blur_window} noise={noise_std:g} "
                        f"u={float(row['u_mae_px']):.3e} exx={float(row['exx_mae']):.3e}"
                    )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep square-grid microstrain benchmark conditions.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--shape", type=_parse_shape, default=(160, 192))
    parser.add_argument("--periods", type=_parse_int_list, default=[8, 12, 16])
    parser.add_argument("--blur-windows", type=_parse_int_list, default=[3, 5])
    parser.add_argument("--noise-stds", type=_parse_float_list, default=[0.0, 0.004])
    parser.add_argument("--strain-presets", type=_parse_str_list, default=["micro", "demo", "stress"])
    parser.add_argument("--supersample", type=int, default=16)
    parser.add_argument("--strain-cycles", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir)
    rows = run_sweep(args)
    _write_csv(output_dir / "benchmark_sweep.csv", rows)
    with (output_dir / "benchmark_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    _write_summary_markdown(output_dir / "benchmark_sweep_summary.md", rows, args.top_n)
    if not args.no_plot:
        _save_plot(output_dir / "benchmark_sweep_period_plot.png", rows)
    print(f"wrote: {output_dir / 'benchmark_sweep.csv'}")
    print(f"wrote: {output_dir / 'benchmark_sweep.json'}")
    print(f"wrote: {output_dir / 'benchmark_sweep_summary.md'}")
    if not args.no_plot:
        print(f"wrote: {output_dir / 'benchmark_sweep_period_plot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
