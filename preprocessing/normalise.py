"""Stage 5 — intensity normalisation (R2).

The step that makes one Week 3 model work across three scanners. Cross-site
intensity scales differ by construction — inversion times span 900-2800 ms
across the sites (ROADMAP 1.4) — and measured on the raw data the p99 of a
Utrecht FLAIR is roughly 4x that of a Singapore FLAIR for the same tissue.
Stage 2 removed the gradient *within* each brain; nothing before this stage made
different brains comparable to each other.

**Three methods are implemented, and ROADMAP's recommendation loses.** ROADMAP
5.4 proposed z-score within the brain mask as primary. Measured on the real
pipeline output across all 60 subjects, it is the worst of the three on every
criterion below.

    metric (all measured on the final pipeline)   z-score  percentile   WM-ref
    cross-site CoV of normalised lesion level      0.130     0.029       0.074
    correlation with disease burden (want ~0)     +0.421    -0.457      +0.306
    within-site subject-to-subject spread          0.138     0.064       0.064
    margin above brain tissue / separation           51%       38%         49%
    margin growth, mild -> severe subjects          +23%      +12%        +33%

**The choice between percentile and WM-referenced is genuinely close, and
percentile wins the metric this project pre-registered** (cross-site CoV). That
is recorded here rather than buried, because the pre-registered rule turned out
to be an incomplete proxy for what Week 3 actually needs. Cross-site agreement
of the *mean* level says nothing about how much headroom a threshold has, nor
about whether severe subjects drift toward it.

WM-referenced is used because:

- it has the larger relative margin (49% vs 38% of the lesion-to-tissue
  separation), which is the quantity a fixed threshold spends;
- its margin grows fastest with disease severity (+33% vs +12%), so the subjects
  that matter most get progressively easier rather than harder;
- percentile's scale is set by the brain's own p99, which lesions themselves
  inflate — hence its strong NEGATIVE burden correlation (-0.457). The practical
  harm is muted (its margin still grows with severity), but the mechanism is a
  known defect rather than a neutral trade;
- it is the only one of the three with a physical meaning. "1.0 = normal white
  matter, lesions at 1.4-1.6x" is interpretable, citable, and directly reusable
  in Week 4. Percentile's "0.95-1.03 on a p1-p99 scale" means nothing
  anatomically.

Switching is one line in metadata/dataset.yaml plus a 30-second re-run, and the
full per-subject comparison for all three methods is written to
stage5_method_comparison.csv regardless of which is selected.

**Why WM-referenced works here, and why it needed Stage 4.** It divides by the
median FLAIR intensity of normal-appearing white matter, so NAWM maps to exactly
1.0 and lesions land above it on every scanner. That requires knowing where
normal-appearing white matter is — which is precisely what Stage 4's tissue
segmentation provides, and why Stage 4 had to exist even though ROADMAP never
scheduled it.

The reference is the *raw* WM class from the tissue label map, not the
hole-filled `wm_mask`. WMH are T1-hypointense so they fall outside that class
automatically, which is the definition of "normal-appearing" — measured lesion
contamination is under 5% on every one of the 60 subjects.

**Normalisation statistics are per subject**, computed from that subject's own
brain voxels. Fitting them on the training set and applying to test would be
both leakage and unnecessary (ROADMAP 5.4).
"""

from __future__ import annotations

import numpy as np

WM_REFERENCED = "wm_referenced"
Z_SCORE = "z_score"
PERCENTILE = "percentile"
METHODS = (WM_REFERENCED, Z_SCORE, PERCENTILE)

# Below this many reference voxels the median is not stable enough to divide by.
# The smallest NAWM reference measured across the 60 is 88,737 voxels, so this
# is a guard against a broken upstream stage, not a routine constraint.
MINIMUM_REFERENCE_VOXELS = 1000


def normalise(
    image: np.ndarray,
    brain_mask: np.ndarray,
    *,
    method: str = WM_REFERENCED,
    reference_mask: np.ndarray | None = None,
    percentile_range: tuple[float, float] = (1.0, 99.0),
    zero_outside_brain: bool = True,
    minimum_reference_voxels: int = MINIMUM_REFERENCE_VOXELS,
) -> tuple[np.ndarray, dict]:
    """Rescale a FLAIR volume onto a scanner-independent intensity scale.

    Returns the normalised image and the statistics used, so every subject's
    scaling is recoverable from the QC table rather than being an unrecorded
    side effect.

    `reference_mask` is required for WM_REFERENCED and must be normal-appearing
    white matter (Stage 4's raw WM class).
    """
    if image.shape != brain_mask.shape:
        raise AssertionError(f"image shape {image.shape} != brain mask shape {brain_mask.shape}")
    if not brain_mask.any():
        raise ValueError("Brain mask is empty — cannot normalise")
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf")

    brain_values = image[brain_mask].astype(np.float64)

    if method == WM_REFERENCED:
        if reference_mask is None:
            raise ValueError("WM-referenced normalisation requires a reference_mask")
        reference = reference_mask & brain_mask
        if reference.sum() < minimum_reference_voxels:
            raise ValueError(
                f"Normal-appearing WM reference has only {int(reference.sum())} voxels "
                f"(minimum {minimum_reference_voxels}) — Stage 4 likely failed for this subject"
            )
        # Median, not mean: robust to the residual lesion and partial-volume
        # voxels that inevitably survive inside any tissue class.
        scale = float(np.median(image[reference]))
        if scale <= 0:
            raise ValueError(f"Non-positive white-matter reference value: {scale}")
        normalised = image / scale
        statistics = {"reference_value": scale, "reference_voxels": int(reference.sum())}

    elif method == Z_SCORE:
        mean = float(brain_values.mean())
        sd = float(brain_values.std())
        if sd <= 0:
            raise ValueError("Zero standard deviation inside the brain mask")
        normalised = (image - mean) / sd
        statistics = {"mean": mean, "sd": sd}

    elif method == PERCENTILE:
        low, high = np.percentile(brain_values, percentile_range)
        if high <= low:
            raise ValueError(f"Degenerate percentile range: p{percentile_range[0]}={low}, "
                             f"p{percentile_range[1]}={high}")
        normalised = (image - low) / (high - low)
        statistics = {"p_low": float(low), "p_high": float(high),
                      "percentile_range": list(percentile_range)}
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

    if zero_outside_brain:
        normalised = np.where(brain_mask, normalised, 0.0)

    statistics["method"] = method
    return normalised, statistics


def normalisation_quality(
    normalised: np.ndarray,
    brain_mask: np.ndarray,
    *,
    reference_mask: np.ndarray | None = None,
    lesion_mask: np.ndarray | None = None,
) -> dict:
    """Where do tissue and lesion land on the normalised scale?

    `lesion_level` is the quantity that decides the method: if normalisation has
    worked, it should be the same number on every scanner, so its spread across
    sites is the selection metric.
    """
    quality = {
        "brain_median": float(np.median(normalised[brain_mask])),
        "brain_p99": float(np.percentile(normalised[brain_mask], 99)),
    }
    if reference_mask is not None and reference_mask.any():
        quality["nawm_level"] = float(np.median(normalised[reference_mask & brain_mask]))
    if lesion_mask is not None and lesion_mask.any():
        quality["lesion_level"] = float(np.median(normalised[lesion_mask]))
        quality["lesion_p10"] = float(np.percentile(normalised[lesion_mask], 10))
        if "nawm_level" in quality:
            quality["lesion_over_nawm"] = quality["lesion_level"] / quality["nawm_level"]
    return quality


def fixed_threshold_transfers(
    normalised: np.ndarray,
    brain_mask: np.ndarray,
    lesion_mask: np.ndarray,
    threshold: float,
) -> dict:
    """Sanity check: does ONE cut-off mean the same thing on every scanner?

    This is deliberately NOT a segmentation result and is not tuned — Week 3
    owns segmentation (ROADMAP 6.1). It exists because it is the most direct
    possible test of what normalisation claims to achieve: if the intensity
    scale really did transfer across sites, then a single fixed threshold should
    behave comparably at all three, and if it did not, it will not. Reporting
    per-site Dice from an untuned threshold answers that question and nothing
    more.
    """
    predicted = brain_mask & (normalised > threshold)
    intersection = float((predicted & lesion_mask).sum())
    total = float(predicted.sum() + lesion_mask.sum())
    return {
        "threshold": threshold,
        "dice": (2.0 * intersection / total) if total > 0 else 1.0,
        "recall": float((predicted & lesion_mask).sum() / lesion_mask.sum()),
        "predicted_volume_ratio": float(predicted.sum() / max(lesion_mask.sum(), 1)),
    }
