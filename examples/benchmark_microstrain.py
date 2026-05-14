from __future__ import annotations

import argparse
import json
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
    save_grid_truth_comparison_figure,
    save_square_grid_experiment_npz,
)


DEFAULT_LIMITS = {
    "u_mae_px": 2.0e-2,
    "v_mae_px": 2.0e-2,
    "exx_mae": 1.0e-4,
    "eyy_mae": 1.2e-4,
    "gamma_xy_mae": 8.0e-5,
}


def _parse_shape(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be height,width")
    height, width = parts
    if height < 32 or width < 32:
        raise argparse.ArgumentTypeError("shape values must be at least 32")
    return height, width


def _mae(array: np.ndarray, valid_mask: np.ndarray) -> float:
    return float(np.nanmean(np.abs(apply_valid_mask(array, valid_mask))))


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, float], dict[str, bool]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment = make_microstrain_square_grid(
        shape=args.shape,
        period=args.period,
        strain_xx=args.strain_xx,
        strain_yy=args.strain_yy,
        shear_xy=args.shear_xy,
        rigid_shift=(args.shift_u, args.shift_v),
        supersample=args.supersample,
        blur_window=args.blur_window,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    strain_window = recommended_strain_smoothing_window(args.period, cycles=args.strain_cycles)
    valid_margin = args.valid_margin
    if valid_margin is None:
        valid_margin = max(5 * args.period, strain_window // 2)

    result = analyze_grid(
        experiment.reference,
        experiment.deformed,
        period=experiment.period,
        strain_smooth_window=strain_window,
    )
    valid_mask = inner_valid_mask(experiment.reference.shape, margin=valid_margin)

    metrics = {
        "u_mae_px": _mae(result.x.displacement - experiment.true_u, valid_mask),
        "v_mae_px": _mae(result.y.displacement - experiment.true_v, valid_mask),
        "exx_mae": _mae(result.strain.exx - experiment.true_exx, valid_mask),
        "eyy_mae": _mae(result.strain.eyy - experiment.true_eyy, valid_mask),
        "gamma_xy_mae": _mae(result.strain.gamma_xy - experiment.true_gamma_xy, valid_mask),
        "period_px": float(experiment.period),
        "strain_window_px": float(strain_window),
        "valid_margin_px": float(valid_margin),
    }
    limits = {
        "u_mae_px": args.max_u_mae,
        "v_mae_px": args.max_v_mae,
        "exx_mae": args.max_exx_mae,
        "eyy_mae": args.max_eyy_mae,
        "gamma_xy_mae": args.max_gamma_xy_mae,
    }
    passed = {name: metrics[name] <= limit for name, limit in limits.items()}

    save_square_grid_experiment_npz(output_dir / "benchmark_microstrain_input.npz", experiment)
    save_analysis_npz(
        output_dir / "benchmark_microstrain_result.npz",
        u=result.x.displacement,
        v=result.y.displacement,
        exx=result.strain.exx,
        eyy=result.strain.eyy,
        gamma_xy=result.strain.gamma_xy,
        valid_mask=valid_mask,
        reference=experiment.reference,
        deformed=experiment.deformed,
        extra_arrays={
            "true_u": experiment.true_u,
            "true_v": experiment.true_v,
            "true_exx": experiment.true_exx,
            "true_eyy": experiment.true_eyy,
            "true_gamma_xy": experiment.true_gamma_xy,
            "period": experiment.period,
            "strain_window": strain_window,
            "valid_margin": valid_margin,
        },
    )
    with (output_dir / "benchmark_microstrain_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "metrics": metrics,
                "limits": limits,
                "passed": passed,
                "all_passed": all(passed.values()),
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    if not args.no_figure:
        save_grid_analysis_figure(
            output_dir / "benchmark_microstrain_summary.png",
            reference=experiment.reference,
            deformed=experiment.deformed,
            u=result.x.displacement,
            exx=result.strain.exx,
            eyy=result.strain.eyy,
            gamma_xy=result.strain.gamma_xy,
            valid_mask=valid_mask,
        )
        save_grid_truth_comparison_figure(
            output_dir / "benchmark_microstrain_comparison.png",
            measured={
                "u": result.x.displacement,
                "v": result.y.displacement,
                "exx": result.strain.exx,
                "eyy": result.strain.eyy,
                "gamma_xy": result.strain.gamma_xy,
            },
            truth={
                "u": experiment.true_u,
                "v": experiment.true_v,
                "exx": experiment.true_exx,
                "eyy": experiment.true_eyy,
                "gamma_xy": experiment.true_gamma_xy,
            },
            valid_mask=valid_mask,
        )
    return metrics, passed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible square-grid microstrain benchmark.",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--shape", type=_parse_shape, default=(160, 192))
    parser.add_argument("--period", type=int, default=8)
    parser.add_argument("--strain-xx", type=float, default=500e-6)
    parser.add_argument("--strain-yy", type=float, default=-200e-6)
    parser.add_argument("--shear-xy", type=float, default=150e-6)
    parser.add_argument("--shift-u", type=float, default=0.6)
    parser.add_argument("--shift-v", type=float, default=-0.35)
    parser.add_argument("--supersample", type=int, default=64)
    parser.add_argument("--blur-window", type=int, default=3)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--strain-cycles", type=float, default=8.0)
    parser.add_argument("--valid-margin", type=int)
    parser.add_argument("--max-u-mae", type=float, default=DEFAULT_LIMITS["u_mae_px"])
    parser.add_argument("--max-v-mae", type=float, default=DEFAULT_LIMITS["v_mae_px"])
    parser.add_argument("--max-exx-mae", type=float, default=DEFAULT_LIMITS["exx_mae"])
    parser.add_argument("--max-eyy-mae", type=float, default=DEFAULT_LIMITS["eyy_mae"])
    parser.add_argument(
        "--max-gamma-xy-mae",
        type=float,
        default=DEFAULT_LIMITS["gamma_xy_mae"],
    )
    parser.add_argument("--no-figure", action="store_true")
    parser.add_argument("--no-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metrics, passed = run_benchmark(args)
    for name, value in metrics.items():
        print(f"{name}: {value:.6e}")
    if args.no_check:
        return 0
    failed = [name for name, ok in passed.items() if not ok]
    if failed:
        print("benchmark failed: " + ", ".join(failed))
        return 1
    print("benchmark passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
