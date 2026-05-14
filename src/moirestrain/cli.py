from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .core import analyze_grid, recommended_strain_smoothing_window, strain_field
from .geometry import crop_roi, pixel_spacing_from_world_points, rectify_image_pair
from .io import load_grayscale_image, save_analysis_npz, save_grayscale_image
from .masking import inner_valid_mask
from .plotting import save_grid_analysis_figure
from .roi import crop_grating_roi, detect_grating_roi


def _parse_roi(value: str) -> tuple[int, int, int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be y0,x0,y1,x1")
    y0, x0, y1, x1 = parts
    if y1 <= y0 or x1 <= x0:
        raise argparse.ArgumentTypeError("ROI must be non-empty")
    return y0, x0, y1, x1


def _parse_shape(value: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be height,width")
    height, width = parts
    if height < 2 or width < 2:
        raise argparse.ArgumentTypeError("shape values must be greater than one")
    return height, width


def _parse_corner_values(value: str) -> np.ndarray:
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 8:
        raise argparse.ArgumentTypeError(
            "rectification corners must be x0,y0,x1,y1,x2,y2,x3,y3"
        )
    return np.asarray(parts, dtype=float).reshape(4, 2)


def _parse_optional_point_values(value: str) -> np.ndarray:
    return _parse_corner_values(value)


def _load_rectification_config(
    value: str,
    output_shape: tuple[int, int] | None,
) -> tuple[np.ndarray, tuple[int, int] | None, np.ndarray | None, dict]:
    path = Path(value)
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        if isinstance(config, dict):
            points = config.get("image_points", config.get("corners"))
            shape = config.get("output_shape", output_shape)
            world_points = config.get("world_points")
            metadata = dict(config)
        else:
            points = config
            shape = output_shape
            world_points = None
            metadata = {}
        corners = np.asarray(points, dtype=float)
        if corners.shape != (4, 2):
            raise ValueError("rectification corners must have shape (4, 2)")
        if shape is not None:
            shape = tuple(int(v) for v in shape)
        world = None if world_points is None else np.asarray(world_points, dtype=float)
        if world is not None and world.shape != (4, 2):
            raise ValueError("world_points must have shape (4, 2)")
        return corners, shape, world, metadata
    return _parse_corner_values(value), output_shape, None, {}


def _json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    return value


def _write_calibration_json(
    path: str | Path,
    *,
    image_points: np.ndarray,
    output_shape: tuple[int, int],
    period: int | None = None,
    world_points: np.ndarray | None = None,
    unit: str | None = None,
) -> None:
    payload = {
        "image_points": np.asarray(image_points, dtype=float).tolist(),
        "output_shape": list(output_shape),
    }
    if period is not None:
        payload["period"] = int(period)
    if world_points is not None:
        payload["world_points"] = np.asarray(world_points, dtype=float).tolist()
        payload["pixel_spacing"] = list(pixel_spacing_from_world_points(world_points, output_shape))
    if unit is not None:
        payload["unit"] = unit
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moirestrain")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze-grid", help="Analyze a square-grid image pair")
    analyze.add_argument("reference", help="Reference image path")
    analyze.add_argument("deformed", help="Deformed image path")
    analyze.add_argument("--period", type=int, required=True, help="Grid pitch in pixels")
    analyze.add_argument("--roi", type=_parse_roi, help="Manual ROI as y0,x0,y1,x1")
    analyze.add_argument(
        "--auto-roi",
        action="store_true",
        help="Detect and crop the dominant grating ROI from the reference image",
    )
    analyze.add_argument(
        "--roi-margin",
        type=int,
        default=0,
        help="Extra pixels around the manual or detected ROI crop",
    )
    analyze.add_argument(
        "--roi-min-area",
        type=int,
        help="Minimum detected ROI area in pixels for --auto-roi",
    )
    analyze.add_argument("--reference-key", help="Array key for reference .npz input")
    analyze.add_argument("--deformed-key", help="Array key for deformed .npz input")
    analyze.add_argument(
        "--rectify-corners",
        help=(
            "Four image-space corners as x0,y0,x1,y1,x2,y2,x3,y3 or a JSON file "
            "containing image_points/corners"
        ),
    )
    analyze.add_argument("--calibration", help="Calibration JSON with image_points/corners and output_shape")
    analyze.add_argument(
        "--rectify-shape",
        type=_parse_shape,
        help="Rectified output shape as height,width; required unless JSON provides output_shape",
    )
    analyze.add_argument("--strain-cycles", type=float, default=8.0)
    analyze.add_argument("--strain-window", type=int, help="Override strain smoothing window")
    analyze.add_argument("--valid-margin", type=int, help="Valid ROI margin in pixels")
    analyze.add_argument("--out", default="moirestrain_result.npz", help="Output .npz path")
    analyze.add_argument("--figure", default="moirestrain_result.png", help="Output summary PNG path")
    analyze.add_argument("--no-figure", action="store_true", help="Skip summary PNG output")
    analyze.set_defaults(func=run_analyze_grid)

    make_cal = subparsers.add_parser("make-calibration", help="Write a four-corner rectification JSON")
    make_cal.add_argument("--image-points", required=True, type=_parse_corner_values)
    make_cal.add_argument("--output-shape", required=True, type=_parse_shape)
    make_cal.add_argument("--period", type=int)
    make_cal.add_argument("--world-points", type=_parse_optional_point_values)
    make_cal.add_argument("--unit")
    make_cal.add_argument("--out", required=True)
    make_cal.set_defaults(func=run_make_calibration)

    rectify = subparsers.add_parser("rectify-pair", help="Rectify an image pair using four planar corners")
    rectify.add_argument("reference", help="Reference image path")
    rectify.add_argument("deformed", help="Deformed image path")
    rectify.add_argument("--reference-key", help="Array key for reference .npz input")
    rectify.add_argument("--deformed-key", help="Array key for deformed .npz input")
    rectify.add_argument("--calibration", help="Calibration JSON with image_points/corners and output_shape")
    rectify.add_argument("--corners", type=_parse_corner_values, help="Four corners as x0,y0,x1,y1,x2,y2,x3,y3")
    rectify.add_argument("--output-shape", type=_parse_shape, help="Rectified output shape as height,width")
    rectify.add_argument("--reference-out", default="rectified_reference.npy")
    rectify.add_argument("--deformed-out", default="rectified_deformed.npy")
    rectify.add_argument("--metadata-out", default="rectification_metadata.json")
    rectify.set_defaults(func=run_rectify_pair)
    return parser


def run_analyze_grid(args: argparse.Namespace) -> int:
    reference = load_grayscale_image(args.reference, npz_key=args.reference_key)
    deformed = load_grayscale_image(args.deformed, npz_key=args.deformed_key)
    if reference.shape != deformed.shape:
        raise ValueError("reference and deformed images must have the same shape")

    rectified_shape = None
    rectified_corners = None
    rectified_world_points = None
    rectified_metadata = {}
    if args.rectify_corners is not None and args.calibration is not None:
        raise ValueError("use either --rectify-corners or --calibration, not both")
    rectification_source = args.calibration if args.calibration is not None else args.rectify_corners
    if rectification_source is not None:
        rectified_corners, rectified_shape, rectified_world_points, rectified_metadata = _load_rectification_config(
            rectification_source,
            args.rectify_shape,
        )
        if rectified_shape is None:
            raise ValueError("--rectify-shape is required unless the corner JSON provides output_shape")
        reference, deformed = rectify_image_pair(
            reference,
            deformed,
            rectified_corners,
            output_shape=rectified_shape,
        )

    roi_bounds = None
    if args.roi is not None and args.auto_roi:
        raise ValueError("use either --roi or --auto-roi, not both")
    if args.roi is not None:
        roi_bounds = args.roi
        if args.roi_margin:
            y0, x0, y1, x1 = args.roi
            roi_bounds = (
                max(0, y0 - args.roi_margin),
                max(0, x0 - args.roi_margin),
                min(reference.shape[0], y1 + args.roi_margin),
                min(reference.shape[1], x1 + args.roi_margin),
            )
        reference = crop_roi(reference, roi_bounds)
        deformed = crop_roi(deformed, roi_bounds)
    elif args.auto_roi:
        roi = detect_grating_roi(
            reference,
            period=args.period,
            min_area=args.roi_min_area,
        )
        roi_bounds = roi.bounds
        if args.roi_margin:
            y0, x0, y1, x1 = roi_bounds
            roi_bounds = (
                max(0, y0 - args.roi_margin),
                max(0, x0 - args.roi_margin),
                min(reference.shape[0], y1 + args.roi_margin),
                min(reference.shape[1], x1 + args.roi_margin),
            )
            reference = crop_roi(reference, roi_bounds)
            deformed = crop_roi(deformed, roi_bounds)
        else:
            reference = crop_grating_roi(reference, roi)
            deformed = crop_grating_roi(deformed, roi)

    strain_window = args.strain_window
    if strain_window is None:
        strain_window = recommended_strain_smoothing_window(
            args.period,
            cycles=args.strain_cycles,
        )
    valid_margin = args.valid_margin
    if valid_margin is None:
        valid_margin = max(3 * args.period, strain_window // 2)

    result = analyze_grid(
        reference,
        deformed,
        period=args.period,
        strain_smooth_window=strain_window,
    )
    valid_mask = inner_valid_mask(reference.shape, margin=valid_margin)
    physical_arrays = {}
    pixel_spacing = None
    unit = None
    if rectified_world_points is not None and rectified_shape is not None:
        pixel_spacing = pixel_spacing_from_world_points(rectified_world_points, rectified_shape)
        unit = rectified_metadata.get("unit")
        dy, dx = pixel_spacing
        u_physical = result.x.displacement * dx
        v_physical = result.y.displacement * dy
        physical_strain = strain_field(
            u_physical,
            v_physical,
            spacing=(dy, dx),
            smooth_window=strain_window,
        )
        physical_arrays = {
            "u_px": result.x.displacement,
            "v_px": result.y.displacement,
            "u_physical": u_physical,
            "v_physical": v_physical,
            "exx_physical": physical_strain.exx,
            "eyy_physical": physical_strain.eyy,
            "gamma_xy_physical": physical_strain.gamma_xy,
            "pixel_spacing": np.asarray(pixel_spacing, dtype=float),
            **({"unit": unit} if unit is not None else {}),
        }
    save_analysis_npz(
        args.out,
        u=result.x.displacement,
        v=result.y.displacement,
        exx=result.strain.exx,
        eyy=result.strain.eyy,
        gamma_xy=result.strain.gamma_xy,
        valid_mask=valid_mask,
        reference=reference,
        deformed=deformed,
        extra_arrays={
            "period": args.period,
            "strain_window": strain_window,
            "valid_margin": valid_margin,
            **({"roi_bounds": roi_bounds} if roi_bounds is not None else {}),
            **({"rectify_corners": rectified_corners} if rectified_corners is not None else {}),
            **({"rectify_shape": rectified_shape} if rectified_shape is not None else {}),
            **(
                {"rectify_world_points": rectified_world_points}
                if rectified_world_points is not None
                else {}
            ),
            **physical_arrays,
        },
    )
    if not args.no_figure:
        save_grid_analysis_figure(
            args.figure,
            reference=reference,
            deformed=deformed,
            u=result.x.displacement,
            exx=result.strain.exx,
            eyy=result.strain.eyy,
            gamma_xy=result.strain.gamma_xy,
            valid_mask=valid_mask,
        )
    print(f"wrote: {Path(args.out)}")
    if not args.no_figure:
        print(f"wrote: {Path(args.figure)}")
    if roi_bounds is not None:
        print(f"roi bounds: {roi_bounds}")
    if rectified_shape is not None:
        print(f"rectified shape: {rectified_shape}")
    if pixel_spacing is not None:
        suffix = f" {unit}/px" if unit is not None else " physical-unit/px"
        print(f"pixel spacing: dy={pixel_spacing[0]:.6g}, dx={pixel_spacing[1]:.6g}{suffix}")
    print(f"strain smoothing window: {strain_window} px")
    print(f"valid margin: {valid_margin} px")
    return 0


def run_make_calibration(args: argparse.Namespace) -> int:
    _write_calibration_json(
        args.out,
        image_points=args.image_points,
        output_shape=args.output_shape,
        period=args.period,
        world_points=args.world_points,
        unit=args.unit,
    )
    print(f"wrote: {Path(args.out)}")
    return 0


def run_rectify_pair(args: argparse.Namespace) -> int:
    if args.calibration is not None and args.corners is not None:
        raise ValueError("use either --calibration or --corners, not both")
    if args.calibration is None and args.corners is None:
        raise ValueError("either --calibration or --corners is required")

    reference = load_grayscale_image(args.reference, npz_key=args.reference_key)
    deformed = load_grayscale_image(args.deformed, npz_key=args.deformed_key)
    if reference.shape != deformed.shape:
        raise ValueError("reference and deformed images must have the same shape")

    if args.calibration is not None:
        corners, output_shape, world_points, metadata = _load_rectification_config(
            args.calibration,
            args.output_shape,
        )
    else:
        corners = args.corners
        output_shape = args.output_shape
        world_points = None
        metadata = {}
    if output_shape is None:
        raise ValueError("--output-shape is required unless calibration JSON provides output_shape")

    reference_rect, deformed_rect = rectify_image_pair(
        reference,
        deformed,
        corners,
        output_shape=output_shape,
    )
    save_grayscale_image(args.reference_out, reference_rect)
    save_grayscale_image(args.deformed_out, deformed_rect)

    metadata_payload = {
        **metadata,
        "image_points": _json_ready(corners),
        "output_shape": _json_ready(output_shape),
        "reference_input": str(args.reference),
        "deformed_input": str(args.deformed),
        "reference_output": str(args.reference_out),
        "deformed_output": str(args.deformed_out),
    }
    if world_points is not None:
        metadata_payload["world_points"] = _json_ready(world_points)
        metadata_payload["pixel_spacing"] = _json_ready(
            pixel_spacing_from_world_points(world_points, output_shape)
        )
    metadata_path = Path(args.metadata_out)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"wrote: {Path(args.reference_out)}")
    print(f"wrote: {Path(args.deformed_out)}")
    print(f"wrote: {metadata_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
