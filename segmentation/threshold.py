"""Week 3, Route A — the intensity-threshold baseline (R4).

The simplest thing that could possibly work, and the reason it is worth building
is not that it will win. It is that "our network reaches Dice 0.71" means very
little on its own, whereas "our network improves Dice from 0.52 to 0.71 over an
intensity-threshold baseline" is an actual result. ROADMAP 6.1 calls this
baseline essential for exactly that reason.

**Why a single fixed threshold is even defensible here.** On raw FLAIR it would
be meaningless — the same tissue reads about 4x brighter at Utrecht than at
Singapore. Week 2's normalisation removed that: normal-appearing white matter is
now exactly 1.0 on every scan from every scanner, and reference lesions sit at
**1.39-1.61x** that, per-subject range 1.316-1.882. So one number genuinely
means approximately the same thing at all three hospitals. Route A is only
viable because Stage 5 happened.

**What this deliberately does NOT do — and why.**

Week 2 measured that the white-matter mask covers a median 70% and worst case
11% of true lesions, because WMH are dark on T1 and the tissue segmenter files
about half of them under grey matter. ROADMAP 6.3.1 proposes using that mask as
a hard constraint to remove false positives. Doing so would delete a median 30%
and worst case 89% of *true* lesions, capping recall at a level no later stage
could recover — and it would look like a segmentation failure rather than a
masking bug.

So the only anatomical constraint offered here is `exclude_csf`, which drops the
CSF tissue class. That is safe: WMH are white-matter lesions and are never
cerebrospinal fluid, and the CSF class does not swallow them the way the
white-matter class fails to contain them. `restrict_to_white_matter` exists so
the cost of the ROADMAP proposal can be *measured and reported* rather than
argued about, and defaults to off.
"""

from __future__ import annotations

import numpy as np

from preprocessing.morphology import CONNECTIVITY_26, remove_small_components
from preprocessing.tissue_seg import LABEL_CSF, LABEL_WM


def threshold_segment(
    normalised: np.ndarray,
    brain_mask: np.ndarray,
    *,
    threshold: float,
    tissue_labels: np.ndarray | None = None,
    exclude_csf: bool = True,
    restrict_to_white_matter: bool = False,
    min_size_voxels: int = 0,
) -> tuple[np.ndarray, dict]:
    """Segment lesions as 'brighter than `threshold` on the normalised scale'.

    `normalised` must be the Stage 5 output, where 1.0 is normal-appearing white
    matter. Returns the binary prediction and a diagnostics dict recording what
    each step removed, so the contribution of every constraint is reportable
    rather than buried.
    """
    if normalised.shape != brain_mask.shape:
        raise AssertionError(
            f"image shape {normalised.shape} != brain mask shape {brain_mask.shape}"
        )
    if not brain_mask.any():
        raise ValueError("Brain mask is empty")
    if threshold <= 0:
        raise ValueError(f"Threshold must be positive on the normalised scale, got {threshold}")
    if restrict_to_white_matter and tissue_labels is None:
        raise ValueError("restrict_to_white_matter needs tissue_labels")
    if exclude_csf and tissue_labels is None:
        raise ValueError("exclude_csf needs tissue_labels")

    prediction = brain_mask & (normalised > threshold)
    diagnostics = {
        "threshold": threshold,
        "voxels_after_threshold": int(prediction.sum()),
    }

    if exclude_csf:
        before = int(prediction.sum())
        prediction = prediction & (tissue_labels != LABEL_CSF)
        diagnostics["voxels_removed_by_csf_exclusion"] = before - int(prediction.sum())

    if restrict_to_white_matter:
        # Off by default. Present so its cost can be measured, not applied.
        before = int(prediction.sum())
        prediction = prediction & (tissue_labels == LABEL_WM)
        diagnostics["voxels_removed_by_wm_constraint"] = before - int(prediction.sum())

    if min_size_voxels > 1:
        prediction, info = remove_small_components(
            prediction, min_size_voxels, connectivity=CONNECTIVITY_26
        )
        diagnostics.update({
            "min_size_voxels": min_size_voxels,
            "components_removed_by_size": info["components_removed"],
            "voxels_removed_by_size": info["voxels_removed"],
        })

    diagnostics["voxels_predicted"] = int(prediction.sum())
    return prediction, diagnostics


def predicted_volume_ratio(prediction: np.ndarray, reference: np.ndarray) -> float:
    """How many times too much (or too little) volume we predicted.

    Reported alongside Dice because it says *which direction* a bad score comes
    from. A threshold that is too low produces a poor Dice and a ratio of 20;
    one that is too high produces a poor Dice and a ratio of 0.3. Dice alone
    cannot tell those apart, and they call for opposite fixes.
    """
    reference_total = float(reference.sum())
    if reference_total == 0:
        raise ValueError("Empty reference mask")
    return float(prediction.sum()) / reference_total
