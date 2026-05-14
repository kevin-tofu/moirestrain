import numpy as np
import pytest

from moirestrain import apply_valid_mask, crop_to_mask, inner_valid_mask, mask_bounds, robust_limits


def test_inner_valid_mask_and_crop_to_mask():
    mask = inner_valid_mask((10, 12), margin=2)
    assert mask.shape == (10, 12)
    assert mask.sum() == 6 * 8
    assert mask_bounds(mask) == (2, 2, 8, 10)

    values = np.arange(10 * 12).reshape(10, 12)
    masked = apply_valid_mask(values, mask)
    cropped = crop_to_mask(masked, mask)

    assert cropped.shape == (6, 8)
    assert np.isnan(masked[0, 0])
    assert cropped[0, 0] == values[2, 2]


def test_robust_limits_ignore_nan_and_outliers():
    values = np.array([np.nan, 0.0, 1.0, 2.0, 100.0])

    lo, hi = robust_limits(values, percentiles=(0, 75))

    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(26.5)
