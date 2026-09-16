"""Small reusable metric helpers, validated by the phantom suite (test_phantom.py).

Full segmentation evaluation (Hausdorff95, AVD, lesion recall/F1, and label-2
handling) is delegated to the vendored official scorer, checks/evaluation.py,
so that W7 numbers are directly comparable to the published leaderboard. This
module holds only the primitives needed to validate the pipeline's geometry
(distance transforms, connectivity, Dice) against the synthetic phantom now,
in Week 1, before any of it depends on real data.
"""

import numpy as np


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice similarity coefficient between two boolean masks of identical shape."""
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    assert mask_a.shape == mask_b.shape, "Dice requires matching shapes"

    total = mask_a.sum() + mask_b.sum()
    if total == 0:
        return 1.0  # both empty: defined as perfect agreement
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(2.0 * intersection / total)
