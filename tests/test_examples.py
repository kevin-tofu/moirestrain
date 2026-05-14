import importlib.util
import json
from pathlib import Path


def _load_example_module(name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_microstrain_benchmark_writes_metrics_and_checks_limits(tmp_path):
    benchmark = _load_example_module("benchmark_microstrain")
    output_dir = tmp_path / "benchmark"

    exit_code = benchmark.main(
        [
            "--output-dir",
            str(output_dir),
            "--no-figure",
        ]
    )

    assert exit_code == 0
    metrics_path = output_dir / "benchmark_microstrain_metrics.json"
    result_path = output_dir / "benchmark_microstrain_result.npz"
    input_path = output_dir / "benchmark_microstrain_input.npz"
    assert metrics_path.exists()
    assert result_path.exists()
    assert input_path.exists()

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["all_passed"] is True
    assert metrics["metrics"]["exx_mae"] < metrics["limits"]["exx_mae"]


def test_microstrain_benchmark_sweep_writes_table(tmp_path):
    sweep = _load_example_module("benchmark_sweep")
    output_dir = tmp_path / "sweep"

    exit_code = sweep.main(
        [
            "--output-dir",
            str(output_dir),
            "--shape",
            "96,112",
            "--periods",
            "8,12",
            "--blur-windows",
            "3",
            "--noise-stds",
            "0",
            "--strain-presets",
            "micro,demo",
            "--supersample",
            "8",
            "--no-plot",
        ]
    )

    assert exit_code == 0
    csv_path = output_dir / "benchmark_sweep.csv"
    json_path = output_dir / "benchmark_sweep.json"
    summary_path = output_dir / "benchmark_sweep_summary.md"
    assert csv_path.exists()
    assert json_path.exists()
    assert summary_path.exists()
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(rows) == 4
    assert {"micro", "demo"} == {row["strain_preset"] for row in rows}
    assert "period" in csv_path.read_text(encoding="utf-8").splitlines()[0]


def test_partial_grid_detection_analysis_writes_result(tmp_path):
    example = _load_example_module("partial_grid_detection_analysis")
    output_dir = tmp_path / "partial"

    exit_code = example.main(
        [
            "--output-dir",
            str(output_dir),
            "--no-figure",
        ]
    )

    assert exit_code == 0
    result_path = output_dir / "partial_grid_detection_result.npz"
    assert result_path.exists()
    result = __import__("numpy").load(result_path)
    assert result["roi_bounds"].shape == (4,)
    assert result["valid_mask"].any()
    assert result["exx_error"].shape == result["exx"].shape
