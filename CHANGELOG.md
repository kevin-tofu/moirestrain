# Changelog

All notable changes to `moirestrain` will be documented in this file.

The format follows the spirit of Keep a Changelog, and this project uses
semantic versioning before publication where practical.

## [0.1.0] - 2026-05-14

### Added

- NumPy implementation of phase-shifted sampling moire analysis.
- Wrapped phase estimation from phase-shifted sampling stacks.
- Phase unwrapping, displacement calculation, and small-strain field utilities.
- Two-direction square-grid analysis via `analyze_grid`.
- Synthetic microstrain square-grid data generator with ground truth.
- Valid ROI masking and robust plotting utilities.
- ROI detection for grating patches embedded in larger images.
- Planar perspective rectification and oblique-grid resampling helpers.
- `moirestrain analyze-grid` CLI with manual ROI, automatic ROI, and four-corner rectification.
- Reproducible microstrain benchmark example with error metrics.
- Sphinx documentation, docs image asset generation, and GitHub Actions workflows.
