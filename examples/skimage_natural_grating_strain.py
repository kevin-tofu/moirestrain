from pathlib import Path

import numpy as np

from moirestrain import (
    analyze_grid,
    apply_valid_mask,
    detect_grating_roi,
    grating_energy,
    homography_from_points,
    inner_valid_mask,
    rectangle_points,
    rectify_image_pair,
    robust_limits,
    warp_perspective,
)


def _camera_image() -> np.ndarray:
    try:
        from skimage import color, data, img_as_float
    except ImportError:
        y, x = np.mgrid[:512, :512]
        image = (
            0.45
            + 0.18 * np.sin(2.0 * np.pi * x / 180.0)
            + 0.12 * np.cos(2.0 * np.pi * y / 130.0)
            + 0.08 * np.sin(2.0 * np.pi * (x + 0.7 * y) / 95.0)
        )
        rng = np.random.default_rng(8)
        image += rng.normal(scale=0.025, size=image.shape)
        return np.clip(image, 0.0, 1.0)

    image = data.camera()
    if image.ndim == 3:
        image = color.rgb2gray(image)
    return img_as_float(image)


def _save_figure(
    path: Path,
    *,
    reference: np.ndarray,
    image_points: np.ndarray,
    detected_mask: np.ndarray,
    rectified_reference: np.ndarray,
    rectified_deformed: np.ndarray,
    reference_x_component: np.ndarray,
    reference_y_component: np.ndarray,
) -> None:
    try:
        import os

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/moirestrain-matplotlib")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        print("matplotlib is not installed; skipped PNG visualization")
        return

    fields = [
        ("oblique camera image", reference, "gray", None),
        ("detected ROI mask", detected_mask.astype(float), "gray", (0.0, 1.0)),
        ("rectified reference", rectified_reference, "gray", None),
        ("rectified deformed", rectified_deformed, "gray", None),
        ("x-periodic component", reference_x_component, "gray", None),
        ("y-periodic component", reference_y_component, "gray", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7), constrained_layout=True)
    for ax, (title, field, cmap, limits) in zip(axes.ravel(), fields):
        kwargs = {}
        if limits is not None:
            kwargs["vmin"], kwargs["vmax"] = limits
        elif cmap == "coolwarm":
            kwargs["vmin"], kwargs["vmax"] = robust_limits(field)
        image = ax.imshow(field, cmap=cmap, **kwargs)
        if title == "oblique camera image":
            ax.add_patch(
                Polygon(
                    image_points,
                    closed=True,
                    fill=False,
                    edgecolor="tab:red",
                    linewidth=2.0,
                )
            )
        ax.set_title(title)
        ax.set_axis_off()
        fig.colorbar(image, ax=ax, shrink=0.72)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _blur_axis(image: np.ndarray, window: int, axis: int) -> np.ndarray:
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


def _optical_blur(image: np.ndarray, window: int = 3) -> np.ndarray:
    return _blur_axis(_blur_axis(image, window, axis=0), window, axis=1)


def _square_marker_grid_pair(shape: tuple[int, int], period: int) -> dict[str, np.ndarray]:
    height, width = shape
    y, x = np.mgrid[:height, :width]
    xc = x - 0.5 * (width - 1)
    yc = y - 0.5 * (height - 1)
    exx = 900e-6
    eyy = -300e-6
    gamma_xy = 450e-6
    u = 0.7 + exx * xc + 0.5 * gamma_xy * yc
    v = -0.4 + eyy * yc + 0.5 * gamma_xy * xc
    if period % 2 != 0:
        raise ValueError("period must be even so black square width equals white gap width")
    marker_size = period // 2

    def target(xx: np.ndarray, yy: np.ndarray) -> np.ndarray:
        x_mod = np.mod(xx, period)
        y_mod = np.mod(yy, period)
        black_square = (x_mod < marker_size) & (y_mod < marker_size)
        return np.where(black_square, 0.05, 0.95)

    def camera_sample(uu: np.ndarray, vv: np.ndarray) -> np.ndarray:
        scale = 8
        offsets = (np.arange(scale, dtype=float) + 0.5) / scale - 0.5
        accum = np.zeros(shape, dtype=float)
        for oy in offsets:
            for ox in offsets:
                accum += target(x + ox + uu, y + oy + vv)
        return accum / float(scale * scale)

    reference = _optical_blur(camera_sample(0.0, 0.0), window=3)
    deformed = _optical_blur(camera_sample(u, v), window=3)
    return {
        "reference": reference,
        "deformed": deformed,
        "true_u": u,
        "true_v": v,
        "true_exx": np.full(shape, exx),
        "true_eyy": np.full(shape, eyy),
        "true_gamma_xy": np.full(shape, gamma_xy),
    }


def _paste_patch(background: np.ndarray, patch: np.ndarray, image_points: np.ndarray) -> np.ndarray:
    rect_to_camera = homography_from_points(rectangle_points(patch.shape), image_points)
    warped = warp_perspective(patch, rect_to_camera, background.shape, fill_value=np.nan)
    output = background.copy()
    mask = np.isfinite(warped)
    # An opaque grating sheet placed in the natural image.
    output[mask] = warped[mask]
    return np.clip(output, 0.0, 1.0)


def main() -> None:
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    period = 8
    rectified_shape = (112, 144)
    image_points = np.array(
        [
            [155.0, 118.0],
            [338.0, 132.0],
            [318.0, 286.0],
            [136.0, 270.0],
        ]
    )
    background = _camera_image()
    grating = _square_marker_grid_pair(rectified_shape, period)
    rng = np.random.default_rng(4)

    reference = _paste_patch(background, grating["reference"], image_points)
    deformed = _paste_patch(background, grating["deformed"], image_points)
    reference = np.clip(reference + rng.normal(scale=0.004, size=background.shape), 0.0, 1.0)
    deformed = np.clip(deformed + rng.normal(scale=0.004, size=background.shape), 0.0, 1.0)

    energy = grating_energy(reference, period=period)
    detected = detect_grating_roi(
        reference,
        period=period,
        threshold=float(np.quantile(energy, 0.90)),
        min_area=4_000,
    )
    ref_rect, def_rect = rectify_image_pair(reference, deformed, image_points, output_shape=rectified_shape)

    grid_result = analyze_grid(
        ref_rect,
        def_rect,
        period=period,
        strain_smooth_window=17,
    )

    valid_mask = inner_valid_mask(rectified_shape, margin=3 * period)
    np.savez_compressed(
        output_dir / "skimage_natural_grating_strain_result.npz",
        reference=reference,
        deformed=deformed,
        rectified_reference=ref_rect,
        rectified_deformed=def_rect,
        reference_x_component=grid_result.reference_x_component,
        reference_y_component=grid_result.reference_y_component,
        detected_points=detected.image_points,
        rectification_points=image_points,
        detected_mask=detected.mask,
        u=grid_result.x.displacement,
        v=grid_result.y.displacement,
        exx=grid_result.strain.exx,
        eyy=grid_result.strain.eyy,
        gamma_xy=grid_result.strain.gamma_xy,
        valid_mask=valid_mask,
    )
    _save_figure(
        output_dir / "skimage_natural_grating_strain.png",
        reference=reference,
        image_points=image_points,
        detected_mask=detected.mask,
        rectified_reference=ref_rect,
        rectified_deformed=def_rect,
        reference_x_component=grid_result.reference_x_component,
        reference_y_component=grid_result.reference_y_component,
    )

    print("wrote: data/skimage_natural_grating_strain_result.npz")
    print("wrote: data/skimage_natural_grating_strain.png")
    print(f"detected corners:\n{detected.image_points}")
    print(
        "mean |exx error|:      "
        f"{np.nanmean(np.abs(apply_valid_mask(grid_result.strain.exx - grating['true_exx'], valid_mask))):.6e}"
    )
    print(
        "mean |eyy error|:      "
        f"{np.nanmean(np.abs(apply_valid_mask(grid_result.strain.eyy - grating['true_eyy'], valid_mask))):.6e}"
    )
    print(
        "mean |gamma_xy error|: "
        f"{np.nanmean(np.abs(apply_valid_mask(grid_result.strain.gamma_xy - grating['true_gamma_xy'], valid_mask))):.6e}"
    )


if __name__ == "__main__":
    main()
