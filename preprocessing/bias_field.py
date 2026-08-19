"""Stage 2 — N4 bias field correction (R3).

The dataset already ships SPM12-corrected images in `pre/`, so using those
would satisfy nothing: R3 asks us to implement correction with justification.
We run N4ITK on `orig/FLAIR.nii.gz` ourselves and keep `pre/FLAIR.nii.gz`
strictly as an independent comparator (ROADMAP 5.3).

**Why the fitting-level count is the parameter that matters.**

N4 models the bias field as a smooth multiplicative B-spline field. Each
multi-resolution level doubles the control point mesh, so the default four
levels give a field flexible enough to follow lesion-scale structure — and a
large confluent WMH looks exactly like a broad bright patch. N4 then absorbs
the disease into the "correction" and flattens it away. Nothing crashes; the
image simply comes out with less lesion contrast than it went in with, and
Week 3 inherits the damage.

Swept over the 48 training subjects (`sweep_n4.py`). Mean / median change in
lesion-to-WM contrast-to-noise, by site:

    levels   Amsterdam      Singapore      Utrecht        worst site
      2      +4.0 / +3.1    +0.7 / +0.7   +27.8 / +20.4    +0.7  PASS
      3      +6.3 / +2.9    -1.8 / -1.3   +47.7 / +28.3    -1.8  REJECTED
      4     +13.3 / +12.4   -0.0 / -0.9   +47.1 / +20.4    -0.9  REJECTED

**Two levels is what the project uses.** It is the only setting that keeps
every site positive under both aggregations, the only one where no subject
loses more than 5% lesion contrast, and the one where the estimated field is
essentially flat across lesions (bias inside lesions / inside NAWM peaks at
1.032, against 1.194 at four levels).

The sites genuinely disagree — Utrecht gains a great deal from aggressive
correction while Singapore degrades — which is why this is measured per site
and never pooled. See `metadata/dataset.yaml` for the full record, including
the accepted cost: two levels removes ~14% of field span where SPM12 removes
28-38%, so we deliberately under-correct in order not to damage the signal
Week 3 has to detect.

**Order of operations.** Correction runs before skull stripping, on the rough
head mask from Stage 1: N4 is degraded by large air regions, but a brain mask
computed on an uncorrected image is itself degraded by the bias field
(ROADMAP 5.1). The field is *fitted* on a shrunk image for speed and *applied*
at full resolution, which is the standard N4 usage and not an approximation of
the output.
"""

from __future__ import annotations

import time

import numpy as np
import SimpleITK as sitk

from preprocessing.sitk_interop import from_sitk, to_sitk


def correct_bias_field(
    array: np.ndarray,
    spacing: tuple[float, float, float],
    head_mask: np.ndarray,
    *,
    fitting_levels: int,
    iterations_per_level: int = 50,
    shrink_factor: tuple[int, int, int] = (2, 2, 1),
    convergence_threshold: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate and remove a multiplicative bias field.

    Returns `(corrected, bias_field, diagnostics)`, all in the input's
    (x, y, z) order. `corrected == array / bias_field` wherever the field is
    defined.

    `shrink_factor` defaults to (2, 2, 1): no shrinking along z, because slice
    thickness is already 3.00 mm against 0.56-1.30 mm in-plane, and shrinking
    an already-coarse axis discards the little through-plane information there
    is.
    """
    if array.shape != head_mask.shape:
        raise AssertionError(
            f"FLAIR shape {array.shape} != head mask shape {head_mask.shape}"
        )
    if not head_mask.any():
        raise ValueError("Head mask is empty — cannot fit a bias field")
    if not np.isfinite(array).all():
        raise ValueError("FLAIR volume contains NaN or Inf")
    if fitting_levels < 1:
        raise ValueError(f"fitting_levels must be >= 1, got {fitting_levels}")

    image = to_sitk(array, spacing, dtype=sitk.sitkFloat32)
    mask = to_sitk(head_mask.astype(np.uint8), spacing, dtype=sitk.sitkUInt8)

    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([iterations_per_level] * fitting_levels)
    corrector.SetConvergenceThreshold(convergence_threshold)

    started = time.time()
    # Fit on the shrunk image, then evaluate the field at full resolution.
    corrector.Execute(
        sitk.Shrink(image, list(shrink_factor)),
        sitk.Shrink(mask, list(shrink_factor)),
    )
    log_bias = corrector.GetLogBiasFieldAsImage(image)
    elapsed = time.time() - started

    bias = np.exp(from_sitk(log_bias))
    if not np.isfinite(bias).all():
        raise ValueError("Estimated bias field contains NaN or Inf")

    corrected = np.divide(array, bias, out=np.zeros_like(array, dtype=np.float64), where=bias > 0)

    inside = bias[head_mask]
    p5, p95 = np.percentile(inside, [5, 95])
    diagnostics = {
        "fitting_levels": fitting_levels,
        "iterations_per_level": iterations_per_level,
        "shrink_factor": list(shrink_factor),
        "runtime_s": elapsed,
        "bias_p5": float(p5),
        "bias_p95": float(p95),
        # Peak-to-peak inhomogeneity removed, as a percentage. SPM12's own
        # correction on this dataset spans 28-38%, so this is the number our
        # correction is compared against.
        "bias_span_percent": float(100.0 * (p95 / p5 - 1.0)) if p5 > 0 else float("nan"),
        "bias_mean_in_head": float(inside.mean()),
    }
    return corrected, bias, diagnostics


def intensity_uniformity_cv(array: np.ndarray, roi_mask: np.ndarray) -> float:
    """Coefficient of variation within an ROI — lower is more uniform.

    ROADMAP 5.3 asks for a before/after intensity-uniformity metric within a
    white-matter ROI. Computed on a *fixed* ROI defined once on the uncorrected
    image, so before and after are measured over identical voxels.
    """
    values = array[roi_mask]
    if values.size == 0:
        raise ValueError("Empty ROI for uniformity measurement")
    mean = float(values.mean())
    if mean == 0:
        return float("nan")
    return float(values.std() / mean)


def lesion_contrast_to_noise(
    array: np.ndarray, lesion_mask: np.ndarray, reference_mask: np.ndarray
) -> float:
    """(median lesion - median reference) / robust noise SD of the reference tissue.

    The quantity Stage 2 must not damage. Both ROIs are supplied by the caller
    and must be defined ONCE on the uncorrected image, then reused unchanged for
    the corrected image — deriving the ROI from each image separately would let
    the correction move the ROI and confound the comparison it is being judged by.

    Robust (MAD-based) SD rather than the plain standard deviation, because the
    reference tissue region inevitably contains some partial-volume and vessel
    voxels that would inflate a non-robust estimate.
    """
    if not lesion_mask.any():
        raise ValueError("Empty lesion mask for CNR")
    if not reference_mask.any():
        raise ValueError("Empty reference tissue mask for CNR")

    reference = array[reference_mask]
    reference_median = float(np.median(reference))
    noise = 1.4826 * float(np.median(np.abs(reference - reference_median)))
    if noise == 0:
        return float("nan")
    return (float(np.median(array[lesion_mask])) - reference_median) / noise


def normal_appearing_tissue_mask(
    array: np.ndarray, head_mask: np.ndarray, lesion_mask: np.ndarray
) -> np.ndarray:
    """Proxy ROI for normal-appearing brain tissue, for CNR and uniformity.

    A genuine white-matter mask does not exist until Stage 4, and Stage 4 runs
    on the *output* of Stage 2 — so Stage 2 cannot depend on it without a
    circular dependency. This proxy takes head voxels above the median head
    intensity, excluding reference lesions.

    It is explicitly a proxy and is only ever used to compare N4 settings
    against each other on identical voxels, never as an anatomical statement.
    Stage 5's normalisation uses the real Stage 4 white-matter mask.
    """
    if not head_mask.any():
        raise ValueError("Empty head mask")
    threshold = np.percentile(array[head_mask], 50)
    return head_mask & ~lesion_mask & (array > threshold)
