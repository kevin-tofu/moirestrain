from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from moirestrain import (
    analyze_grid,
    apply_valid_mask,
    crop_grating_roi,
    crop_to_mask,
    detect_grating_roi,
    grating_energy,
    inner_valid_mask,
    recommended_strain_smoothing_window,
    robust_limits,
    save_analysis_npz,
    save_grid_truth_comparison_figure,
)


def _background(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    image = (
        0.42
        + 0.14 * np.sin(2.0 * np.pi * x / 170.0)
        + 0.10 * np.cos(2.0 * np.pi * y / 120.0)
        + 0.06 * np.sin(2.0 * np.pi * (x + 0.6 * y) / 90.0)
    )
    rng = np.random.default_rng(12)
    image += rng.normal(scale=0.018, size=shape)
    return np.clip(image, 0.0, 1.0)


def _paste_axis_aligned(
    background: np.ndarray,
    patch: np.ndarray,
    *,
    top: int,
    left: int,
) -> np.ndarray:
    output = background.copy()
    height, width = patch.shape
    output[top : top + height, left : left + width] = patch
    return output


def _blur_axis(image: np.ndarray, window: int, axis: int) -> np.ndarray:
    if window <= 1:
        return image
    kernel = np.ones(window, dtype=float) / window
    pad = (window // 2, window - 1 - window // 2)
    pad_width = [(0, 0), (0, 0)]
    pad_width[axis] = pad
    padded = np.pad(image, pad_width, mode="reflect")
    return np.apply_along_axis(
        lambda row: np.convolve(row, kernel, mode="valid"),
        axis=axis,
        arr=padded,
    )


def _box_blur(image: np.ndarray, window: int) -> np.ndarray:
    return _blur_axis(_blur_axis(image, window, axis=0), window, axis=1)


def _distributed_square_grid_pair(
    shape: tuple[int, int],
    *,
    period: int,
    supersample: int = 32,
    blur_window: int = 3,
) -> dict[str, np.ndarray]:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    xc = x - 0.5 * (width - 1)
    yc = y - 0.5 * (height - 1)
    lx = float(width)
    ly = float(height)

    u = (
        0.55
        + 0.540 * np.sin(2.0 * np.pi * x / lx)
        + 0.150 * np.sin(2.0 * np.pi * y / ly)
        + 0.210 * np.sin(2.0 * np.pi * x / lx) * np.cos(2.0 * np.pi * y / ly)
    )
    v = (
        -0.30
        + 0.360 * np.cos(2.0 * np.pi * y / ly)
        + 0.105 * np.cos(2.0 * np.pi * x / lx)
        + 0.150 * np.cos(2.0 * np.pi * x / lx) * np.sin(2.0 * np.pi * y / ly)
    )
    du_dy, du_dx = np.gradient(u, 1.0, 1.0, edge_order=2)
    dv_dy, dv_dx = np.gradient(v, 1.0, 1.0, edge_order=2)

    if period % 2 != 0:
        raise ValueError("period must be even so black square width equals white gap width")
    marker_size = period // 2

    def target(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        black = (np.mod(xx, period) < marker_size) & (np.mod(yy, period) < marker_size)
        return np.where(black, 0.05, 0.95)

    offsets = (np.arange(supersample, dtype=float) + 0.5) / supersample - 0.5

    def camera_sample(uu: np.ndarray | float, vv: np.ndarray | float) -> np.ndarray:
        image = np.zeros(shape, dtype=float)
        for oy in offsets:
            for ox in offsets:
                image += target(x + ox + uu, y + oy + vv)
        return _box_blur(image / float(supersample * supersample), blur_window)

    return {
        "reference": camera_sample(0.0, 0.0),
        "deformed": camera_sample(u, v),
        "true_u": u,
        "true_v": v,
        "true_exx": du_dx,
        "true_eyy": dv_dy,
        "true_gamma_xy": du_dy + dv_dx,
    }


def _crop_truth(array: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    y0, x0, y1, x1 = bounds
    return array[y0:y1, x0:x1]


def _valid_margin(shape: tuple[int, int], period: int, strain_window: int) -> int:
    requested = max(3 * period, strain_window // 2)
    largest_nonempty = max(1, min(shape) // 2 - 1)
    return min(requested, largest_nonempty)


def _save_figure(
    path: Path,
    *,
    reference: np.ndarray,
    energy: np.ndarray,
    detected_mask: np.ndarray,
    cropped_reference: np.ndarray,
    exx: np.ndarray,
    true_exx: np.ndarray,
    exx_error: np.ndarray,
    valid_mask: np.ndarray,
    bounds: tuple[int, int, int, int],
) -> None:
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/moirestrain-matplotlib")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib is not installed; skipped PNG visualization")
        return

    y0, x0, y1, x1 = bounds

    def valid_view(array: np.ndarray) -> np.ndarray:
        return crop_to_mask(apply_valid_mask(array, valid_mask), valid_mask)

    fields = [
        ("full image + detected ROI", reference, "gray", None),
        ("grating energy", energy, "magma", None),
        ("detected mask", detected_mask.astype(float), "gray", None),
        ("cropped grating ROI", cropped_reference, "gray", None),
        ("exx measured valid ROI", valid_view(exx), "coolwarm", "shared"),
        ("exx true valid ROI", valid_view(true_exx), "coolwarm", "shared"),
        ("exx error valid ROI", valid_view(exx_error), "coolwarm", "robust"),
    ]
    exx_limits = robust_limits(np.stack([valid_view(exx), valid_view(true_exx)]))
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, (title, field, cmap, limit_mode) in zip(axes.ravel(), fields):
        kwargs = {}
        if limit_mode == "robust":
            kwargs["vmin"], kwargs["vmax"] = robust_limits(field)
        elif limit_mode == "shared":
            kwargs["vmin"], kwargs["vmax"] = exx_limits
        image = ax.imshow(field, cmap=cmap, **kwargs)
        if title == "full image + detected ROI":
            ax.add_patch(
                Rectangle(
                    (x0, y0),
                    x1 - x0,
                    y1 - y0,
                    fill=False,
                    edgecolor="tab:red",
                    linewidth=2.0,
                )
            )
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, shrink=0.72)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_measured_true_figure(
    path: Path,
    *,
    grid: np.ndarray,
    measured: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    valid_mask: np.ndarray,
) -> None:
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/moirestrain-matplotlib")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped measured/true PNG visualization")
        return

    names = ["exx", "eyy", "gamma_xy"]

    def valid_view(array: np.ndarray) -> np.ndarray:
        return crop_to_mask(apply_valid_mask(array, valid_mask), valid_mask)

    fig = plt.figure(figsize=(10.4, 8.4), constrained_layout=True)
    spec = fig.add_gridspec(3, 3, width_ratios=(1.15, 1.0, 1.0))
    for row in range(len(names)):
        grid_ax = fig.add_subplot(spec[row, 0])
        grid_ax.imshow(grid, cmap="gray", vmin=0.0, vmax=1.0)
        if row == 0:
            grid_ax.set_title("detected grid ROI")
        grid_ax.set_axis_off()

    axes = np.array(
        [
            [fig.add_subplot(spec[row, col]) for col in (1, 2)]
            for row in range(len(names))
        ],
        dtype=object,
    )
    for row, name in enumerate(names):
        measured_view = valid_view(measured[name])
        truth_view = valid_view(truth[name])
        limits = robust_limits(np.stack([measured_view, truth_view]))
        for ax, title, field in (
            (axes[row, 0], f"{name} measured", measured_view),
            (axes[row, 1], f"{name} true", truth_view),
        ):
            image = ax.imshow(field, cmap="coolwarm", vmin=limits[0], vmax=limits[1])
            ax.set_title(title)
            ax.set_axis_off()
            fig.colorbar(image, ax=ax, shrink=0.72)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect a partial square-grid patch in a full image and analyze its strain.",
    )
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--period", type=int, default=16)
    parser.add_argument("--strain-cycles", type=float, default=3.0)
    parser.add_argument("--no-figure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    canvas_shape = (384, 512)
    patch_top = 96
    patch_left = 168
    experiment = _distributed_square_grid_pair(
        (144, 176),
        period=args.period,
        supersample=32,
        blur_window=3,
    )
    reference = _paste_axis_aligned(
        _background(canvas_shape),
        experiment["reference"],
        top=patch_top,
        left=patch_left,
    )
    deformed = _paste_axis_aligned(
        _background(canvas_shape),
        experiment["deformed"],
        top=patch_top,
        left=patch_left,
    )

    true_exx_full = np.full(canvas_shape, np.nan)
    true_eyy_full = np.full(canvas_shape, np.nan)
    true_gamma_full = np.full(canvas_shape, np.nan)
    true_exx_full[
        patch_top : patch_top + experiment["reference"].shape[0],
        patch_left : patch_left + experiment["reference"].shape[1],
    ] = experiment["true_exx"]
    true_eyy_full[
        patch_top : patch_top + experiment["reference"].shape[0],
        patch_left : patch_left + experiment["reference"].shape[1],
    ] = experiment["true_eyy"]
    true_gamma_full[
        patch_top : patch_top + experiment["reference"].shape[0],
        patch_left : patch_left + experiment["reference"].shape[1],
    ] = experiment["true_gamma_xy"]

    energy = grating_energy(reference, period=args.period)
    roi = detect_grating_roi(
        reference,
        period=args.period,
        threshold=float(np.quantile(energy, 0.90)),
        min_area=6_000,
        corner_method="oriented_box",
    )
    cropped_reference = crop_grating_roi(reference, roi)
    cropped_deformed = crop_grating_roi(deformed, roi)
    true_exx = _crop_truth(true_exx_full, roi.bounds)
    true_eyy = _crop_truth(true_eyy_full, roi.bounds)
    true_gamma = _crop_truth(true_gamma_full, roi.bounds)

    strain_window = recommended_strain_smoothing_window(args.period, cycles=args.strain_cycles)
    result = analyze_grid(
        cropped_reference,
        cropped_deformed,
        period=args.period,
        strain_smooth_window=strain_window,
    )
    valid_margin = _valid_margin(cropped_reference.shape, args.period, strain_window)
    valid_mask = inner_valid_mask(cropped_reference.shape, margin=valid_margin)
    truth_mask = np.isfinite(true_exx) & np.isfinite(true_eyy) & np.isfinite(true_gamma)
    valid_mask = valid_mask & truth_mask

    exx_error = result.strain.exx - true_exx
    eyy_error = result.strain.eyy - true_eyy
    gamma_error = result.strain.gamma_xy - true_gamma
    save_analysis_npz(
        output_dir / "partial_grid_detection_result.npz",
        u=result.x.displacement,
        v=result.y.displacement,
        exx=result.strain.exx,
        eyy=result.strain.eyy,
        gamma_xy=result.strain.gamma_xy,
        valid_mask=valid_mask,
        reference=cropped_reference,
        deformed=cropped_deformed,
        extra_arrays={
            "full_reference": reference,
            "full_deformed": deformed,
            "grating_energy": energy,
            "detected_mask": roi.mask,
            "roi_bounds": roi.bounds,
            "roi_image_points": roi.image_points,
            "true_exx": true_exx,
            "true_eyy": true_eyy,
            "true_gamma_xy": true_gamma,
            "exx_error": exx_error,
            "eyy_error": eyy_error,
            "gamma_xy_error": gamma_error,
            "period": args.period,
            "strain_window": strain_window,
            "strain_cycles": args.strain_cycles,
            "valid_margin": valid_margin,
        },
    )
    if not args.no_figure:
        _save_figure(
            output_dir / "partial_grid_detection_analysis.png",
            reference=reference,
            energy=energy,
            detected_mask=roi.mask,
            cropped_reference=cropped_reference,
            exx=result.strain.exx,
            true_exx=true_exx,
            exx_error=exx_error,
            valid_mask=valid_mask,
            bounds=roi.bounds,
        )
        save_grid_truth_comparison_figure(
            output_dir / "partial_grid_strain_comparison.png",
            measured={
                "exx": result.strain.exx,
                "eyy": result.strain.eyy,
                "gamma_xy": result.strain.gamma_xy,
            },
            truth={
                "exx": true_exx,
                "eyy": true_eyy,
                "gamma_xy": true_gamma,
            },
            valid_mask=valid_mask,
        )
        _save_measured_true_figure(
            output_dir / "partial_grid_strain_measured_true.png",
            grid=cropped_reference,
            measured={
                "exx": result.strain.exx,
                "eyy": result.strain.eyy,
                "gamma_xy": result.strain.gamma_xy,
            },
            truth={
                "exx": true_exx,
                "eyy": true_eyy,
                "gamma_xy": true_gamma,
            },
            valid_mask=valid_mask,
        )
        save_grid_truth_comparison_figure(
            output_dir / "partial_grid_exx_comparison.png",
            measured={"exx": result.strain.exx},
            truth={"exx": true_exx},
            valid_mask=valid_mask,
        )
        save_grid_truth_comparison_figure(
            output_dir / "partial_grid_eyy_comparison.png",
            measured={"eyy": result.strain.eyy},
            truth={"eyy": true_eyy},
            valid_mask=valid_mask,
        )
        save_grid_truth_comparison_figure(
            output_dir / "partial_grid_gamma_xy_comparison.png",
            measured={"gamma_xy": result.strain.gamma_xy},
            truth={"gamma_xy": true_gamma},
            valid_mask=valid_mask,
        )

    def mae(array: np.ndarray) -> float:
        return float(np.nanmean(np.abs(apply_valid_mask(array, valid_mask))))

    print(f"wrote: {output_dir / 'partial_grid_detection_result.npz'}")
    if not args.no_figure:
        print(f"wrote: {output_dir / 'partial_grid_detection_analysis.png'}")
        print(f"wrote: {output_dir / 'partial_grid_strain_comparison.png'}")
        print(f"wrote: {output_dir / 'partial_grid_strain_measured_true.png'}")
        print(f"wrote: {output_dir / 'partial_grid_exx_comparison.png'}")
        print(f"wrote: {output_dir / 'partial_grid_eyy_comparison.png'}")
        print(f"wrote: {output_dir / 'partial_grid_gamma_xy_comparison.png'}")
    print(f"detected bounds: {roi.bounds}")
    print(f"strain smoothing window: {strain_window} px")
    print(f"valid margin: {valid_margin} px")
    print(f"mean |exx error|:      {mae(exx_error):.6e}")
    print(f"mean |eyy error|:      {mae(eyy_error):.6e}")
    print(f"mean |gamma_xy error|: {mae(gamma_error):.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
