"""Stage 6 — noise reduction and contrast enhancement, as evaluated branches.

Both appear in the Week 2 schedule line ("contrast enhancement, noise
reduction") but neither is in the problem statement's R1-R3. ROADMAP 5.5 is
explicit that they must be *optional branches evaluated against the un-enhanced
path, never the default*. This module implements them so that the decision is a
measurement rather than an omission.

**The expectation, recorded before measuring.** Denoising is expected to be
REJECTED. Two reasons, both already quantified on this dataset:

- 49.6% of the 3,680 reference lesions in the training set are 5 voxels or
  smaller and 69.2% are 10 or smaller, yet those carry only 1.5% and 3.1% of
  total lesion volume. Any smoothing that erases them devastates lesion recall
  and lesion F1 while barely moving Dice, which is volume-weighted and therefore
  structurally blind to exactly this failure.
- There is little noise to win back: measured lesion-to-WM contrast-to-noise is
  already 7.0 at Utrecht, 14.9 at Singapore and 21.7 at Amsterdam.

Recording the expectation up front is what makes the result a finding either
way, rather than a rationalisation after the fact.

**How the damage is measured.** Not by Dice, and not by SSIM. Dice is
volume-weighted and would barely move; SSIM measures perceptual similarity of
the whole image and says nothing about lesions specifically. Instead, every
reference lesion is treated as a connected component and its *contrast above
normal-appearing white matter* is compared before and after filtering:

    contrast_retention = (intensity_after - 1.0) / (intensity_before - 1.0)

on the WM-referenced scale from Stage 5, where 1.0 is normal white matter. A
retention of 1.0 means the lesion is untouched; 0.0 means it has been smoothed
into the surrounding tissue and no longer exists as a signal. Reported per
lesion-size bin, because the whole question is whether small lesions survive.

SSIM is still computed, because ROADMAP 9.3 and CLAUDE.md Decision #5 assign it
to exactly this comparison — denoised versus original pre-processing quality —
and it is the one place in this project where SSIM is meaningful. It is reported
alongside, never used to decide.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from preprocessing.sitk_interop import from_sitk, to_sitk

CURVATURE_DIFFUSION = "curvature_anisotropic_diffusion"
PATCH_BASED = "patch_based"
GAUSSIAN = "gaussian"
DENOISE_METHODS = (CURVATURE_DIFFUSION, PATCH_BASED, GAUSSIAN)


def denoise(
    image: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    method: str = CURVATURE_DIFFUSION,
    iterations: int = 5,
    time_step: float = 0.0625,
    conductance: float = 1.0,
    gaussian_sigma_mm: float = 1.0,
) -> np.ndarray:
    """Apply one denoising filter. Returns the filtered volume.

    Anisotropic diffusion is the primary candidate because it is edge-preserving
    by construction — it smooths along intensity gradients but not across them,
    which is the property that might let it reduce noise without dissolving
    small lesions. Gaussian is included as the naive baseline that demonstrably
    should not be used, so the comparison has a floor.
    """
    if not np.isfinite(image).all():
        raise ValueError("Image contains NaN or Inf")

    if method == CURVATURE_DIFFUSION:
        sitk_image = to_sitk(image, spacing, dtype=sitk.sitkFloat32)
        filt = sitk.CurvatureAnisotropicDiffusionImageFilter()
        filt.SetNumberOfIterations(iterations)
        filt.SetTimeStep(time_step)
        filt.SetConductanceParameter(conductance)
        return from_sitk(filt.Execute(sitk_image)).astype(np.float64)

    if method == PATCH_BASED:
        sitk_image = to_sitk(image, spacing, dtype=sitk.sitkFloat32)
        filt = sitk.PatchBasedDenoisingImageFilter()
        filt.SetNumberOfIterations(1)
        return from_sitk(filt.Execute(sitk_image)).astype(np.float64)

    if method == GAUSSIAN:
        # Sigma given in mm and converted per axis, so the kernel is the same
        # physical size everywhere despite 3 mm slices against ~1 mm in-plane.
        sigma_voxels = tuple(gaussian_sigma_mm / s for s in spacing)
        return ndi.gaussian_filter(image, sigma=sigma_voxels)

    raise ValueError(f"Unknown denoise method {method!r}; expected one of {DENOISE_METHODS}")


def clahe(
    image: np.ndarray,
    brain_mask: np.ndarray,
    *,
    clip_limit: float = 0.01,
    kernel_size_fraction: float = 0.125,
) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation, applied slice by slice.

    2D and in-plane only: a 3D tile on this grid would span ~1 mm in-plane and
    3 mm through-plane, so it would mean a different physical neighbourhood on
    each axis.

    Note for the report: CLAHE is a *non-linear, spatially varying* intensity
    transform. It therefore destroys the property Stage 5 just established —
    that 1.0 means normal-appearing white matter everywhere — which is why it
    cannot be the default path for a threshold-based Week 3 method. One
    challenge team (uned_enhanced) did use non-linear FLAIR enhancement, so it
    is not disqualifying, merely something that has to be justified against the
    un-enhanced path.
    """
    from skimage.exposure import equalize_adapthist

    output = np.zeros_like(image, dtype=np.float64)
    finite = image[brain_mask]
    if finite.size == 0:
        raise ValueError("Empty brain mask")
    low, high = float(finite.min()), float(finite.max())
    if high <= low:
        raise ValueError("Degenerate intensity range inside the brain mask")

    for z in range(image.shape[2]):
        plane = image[:, :, z]
        if not brain_mask[:, :, z].any():
            continue
        scaled = np.clip((plane - low) / (high - low), 0.0, 1.0)
        kernel = max(8, int(min(plane.shape) * kernel_size_fraction))
        equalised = equalize_adapthist(scaled, kernel_size=kernel, clip_limit=clip_limit)
        output[:, :, z] = equalised * (high - low) + low

    return np.where(brain_mask, output, 0.0)


def lesion_contrast_retention(
    before: np.ndarray,
    after: np.ndarray,
    lesion_mask: np.ndarray,
    *,
    baseline: float = 1.0,
    size_bins: tuple[int, ...] = (1, 3, 5, 10, 20, 50),
    small_lesion_threshold: int = 10,
) -> dict:
    """Per-lesion contrast retention, binned by lesion size.

    Both volumes must be on the Stage 5 WM-referenced scale, where `baseline`
    (1.0) is normal-appearing white matter. Returns overall and per-bin
    retention, plus the count of lesions effectively destroyed.

    26-connectivity, matching checks/evaluation.py.
    """
    if before.shape != after.shape != lesion_mask.shape:
        raise AssertionError("before, after and lesion_mask must share a shape")
    if not lesion_mask.any():
        raise ValueError("Empty lesion mask")

    labelled, n_components = ndi.label(lesion_mask, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labelled.ravel())[1:]

    retentions, kept_sizes = [], []
    for index in range(1, n_components + 1):
        component = labelled == index
        contrast_before = float(before[component].mean()) - baseline
        if contrast_before <= 0:
            continue  # lesion was not above white matter to begin with
        contrast_after = float(after[component].mean()) - baseline
        retentions.append(contrast_after / contrast_before)
        kept_sizes.append(int(sizes[index - 1]))

    if not retentions:
        raise ValueError("No lesion component had positive contrast before filtering")

    retentions = np.asarray(retentions)
    kept_sizes = np.asarray(kept_sizes)

    result = {
        "n_lesions": int(len(retentions)),
        "retention_mean": float(retentions.mean()),
        "retention_median": float(np.median(retentions)),
        # A lesion retaining under half its contrast has effectively been
        # smoothed into the background as far as any threshold is concerned.
        "lesions_below_half_retention": int((retentions < 0.5).sum()),
        "fraction_below_half_retention": float((retentions < 0.5).mean()),
    }

    # A single explicit small-lesion figure, so the decision rule reads one
    # unambiguous column rather than reassembling it from bin ranges. 69.2% of
    # reference lesions fall at or below this threshold.
    small = kept_sizes <= small_lesion_threshold
    result["small_lesion_threshold"] = small_lesion_threshold
    result["retention_small"] = float(retentions[small].mean()) if small.any() else float("nan")
    result["n_small_lesions"] = int(small.sum())
    result["small_lesions_below_half"] = int((retentions[small] < 0.5).sum()) if small.any() else 0
    lower = 0
    for upper in size_bins:
        selected = (kept_sizes > lower) & (kept_sizes <= upper)
        if selected.any():
            result[f"retention_{lower + 1}_to_{upper}vox"] = float(retentions[selected].mean())
        lower = upper
    selected = kept_sizes > size_bins[-1]
    if selected.any():
        result[f"retention_over_{size_bins[-1]}vox"] = float(retentions[selected].mean())
    return result


def structural_similarity_in_mask(
    before: np.ndarray, after: np.ndarray, mask: np.ndarray
) -> float:
    """SSIM between two volumes, evaluated over the masked bounding box.

    The one legitimate use of SSIM in this project (CLAUDE.md Decision #5):
    comparing a filtered image against its original. Applying SSIM to a binary
    segmentation mask, as the Week 7 schedule line invites, would be
    meaningless — that is stated in the report rather than quietly ignored.

    Computed slice by slice over the mask's bounding box so that the large empty
    background, which is identical in both volumes and would inflate the score
    towards 1.0, does not dominate.
    """
    from skimage.metrics import structural_similarity

    if not mask.any():
        raise ValueError("Empty mask for SSIM")
    coords = np.argwhere(mask)
    (x0, y0, z0), (x1, y1, z1) = coords.min(axis=0), coords.max(axis=0) + 1
    a = before[x0:x1, y0:y1, z0:z1]
    b = after[x0:x1, y0:y1, z0:z1]

    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range <= 0:
        return 1.0

    scores = [
        structural_similarity(a[:, :, z], b[:, :, z], data_range=data_range)
        for z in range(a.shape[2])
    ]
    return float(np.mean(scores))
