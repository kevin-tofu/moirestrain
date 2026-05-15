# moirestrain

Current version: `0.1.1`

NumPy-first tools for phase-shifted sampling moire analysis of periodic
grating images. `moirestrain` targets full-field micro-displacement and
micro-strain measurement from reference/deformed grid images.

`moirestrain` estimates phase fields from periodic grating images and converts
reference/deformed phase differences into displacement and small-strain fields.
The current high-level workflow focuses on two-dimensional square-marker grid
targets, such as regularly spaced black squares on a white background.

## Highlights

- Generate phase-shifted sampling moire images from a single grating image.
- Estimate wrapped phase with a phase-shifting formula.
- Calculate displacement from reference/deformed phase difference.
- Analyze square-grid targets in both x/y directions.
- Output `u`, `v`, `exx`, `eyy`, and `gamma_xy` fields.
- Detect partial grating ROIs inside larger images.
- Rectify tilted planar grating regions from four corner points.
- Save analysis arrays, valid ROI masks, and PNG summaries.

The numerical core is NumPy-based. Runtime dependencies also include
`imageio` and `matplotlib` for CLI image I/O and PNG reporting. `scikit-image`
is used only by optional examples and development tests.

## Results

### Partial-Grid Strain Recovery

This example detects a square-grid patch inside a larger image, crops the
valid ROI, and compares measured strain against the known synthetic truth.

![partial grid measured true strain comparison](https://media.githubusercontent.com/media/kevin-tofu/moirestrain/main/docs/_static/partial_grid_strain_measured_true.png)

### ROI Detection In A Full Image

The same workflow also reports the full image, grating-energy map, detected
mask, cropped ROI, and strain field.

![partial grid detection analysis](https://media.githubusercontent.com/media/kevin-tofu/moirestrain/main/docs/_static/partial_grid_detection_analysis.png)

### Natural-Image Background

This example places a square-grid strain target on a natural-image background,
detects the grid ROI, rectifies the patch, and outputs the measured strain
fields.

![natural image grid strain analysis](https://media.githubusercontent.com/media/kevin-tofu/moirestrain/main/docs/_static/natural_grating_strain.png)

### Benchmark

The benchmark compares measured fields with ground truth and sweeps target
period and strain presets.

![microstrain benchmark sweep](https://media.githubusercontent.com/media/kevin-tofu/moirestrain/main/docs/_static/benchmark_sweep_period_plot.png)

## Installation

Install the released package from PyPI with pip:

```bash
pip install moirestrain
```

Or add it to a Poetry project:

```bash
poetry add moirestrain
```

For local development:

```bash
poetry install
```

Run tests:

```bash
poetry run pytest
```

Build distributions:

```bash
poetry build
```

## Quick Start

```python
from moirestrain import analyze_grid, recommended_strain_smoothing_window

period = 16
strain_window = recommended_strain_smoothing_window(period, cycles=3)

result = analyze_grid(
    reference_image,
    deformed_image,
    period=period,
    strain_smooth_window=strain_window,
)

u = result.x.displacement
v = result.y.displacement
exx = result.strain.exx
eyy = result.strain.eyy
gamma_xy = result.strain.gamma_xy
```

For one-direction grating analysis:

```python
from moirestrain import phase_shifted_sampling_moire

result = phase_shifted_sampling_moire(
    reference_image,
    deformed_image,
    period=8,
    axis="x",
    grating_pitch=8.0,
)

u = result.displacement
```

## CLI

Analyze a manually cropped square-grid ROI:

```bash
moirestrain analyze-grid ref.png def.png \
  --period 16 \
  --roi y0,x0,y1,x1 \
  --out result.npz \
  --figure result.png
```

Detect the dominant grating ROI automatically:

```bash
moirestrain analyze-grid ref.png def.png \
  --period 16 \
  --auto-roi \
  --out result.npz \
  --figure result.png
```

Create a four-corner calibration, rectify an image pair, then analyze:

```bash
moirestrain make-calibration \
  --image-points x0,y0,x1,y1,x2,y2,x3,y3 \
  --output-shape 240,320 \
  --period 16 \
  --out calibration.json

moirestrain rectify-pair ref.png def.png \
  --calibration calibration.json \
  --reference-out rectified_ref.npy \
  --deformed-out rectified_def.npy \
  --metadata-out rectification_metadata.json

moirestrain analyze-grid ref.png def.png \
  --period 16 \
  --calibration calibration.json \
  --out result.npz \
  --figure result.png
```

If physical dimensions are known, pass `--world-points` and `--unit` to
`make-calibration`. The analysis output then includes `u_physical`,
`v_physical`, `pixel_spacing`, and physical-coordinate strain fields.

Use `--no-figure` for batch jobs that only need `.npz` arrays.

## Examples

Generate synthetic data, run benchmarks, and build figures:

```bash
PYTHONPATH=src python examples/benchmark_microstrain.py
PYTHONPATH=src python examples/benchmark_sweep.py
PYTHONPATH=src python examples/partial_grid_detection_analysis.py
PYTHONPATH=src python examples/skimage_natural_grating_strain.py
```

Runnable learning-oriented workflows are under `tutorials/`:

```bash
PYTHONPATH=src python tutorials/01_square_grid_analysis.py
PYTHONPATH=src python tutorials/02_partial_grid_detection.py
PYTHONPATH=src python tutorials/03_four_corner_rectification.py
```

Other examples:

```bash
PYTHONPATH=src python examples/generate_experiment.py
PYTHONPATH=src python examples/microstrain_square_grid.py
PYTHONPATH=src python examples/strain_distribution.py
PYTHONPATH=src python examples/perspective_rectification.py
```

Build documentation assets and Sphinx HTML:

```bash
PYTHONPATH=src python scripts/build_docs_assets.py
poetry run sphinx-build -b html docs docs/_build/html
```

## Notes For Experiments

- Use a camera-space grid pitch around `12-16 px` or larger when possible.
- Square-marker targets contain strong harmonics; strain estimation needs
  smoothing before differentiation.
- Evaluate and display only a valid inner ROI near image boundaries.
- For tilted planar targets, rectify from four corner points before analysis.
- For real images, check illumination, blur, noise, grid pitch calibration, and
  registration between reference/deformed frames.

## Citation


## License

MIT License. See [LICENSE](LICENSE).
