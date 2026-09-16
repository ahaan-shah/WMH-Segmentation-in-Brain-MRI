"""Stage 4 — WM/GM/CSF tissue segmentation, and the white-matter mask.

**Why this stage exists at all.** It is absent from ROADMAP's Week 2 definition
of done, yet three separate later stages depend on it:

- Stage 5 needs normal-appearing white matter as the reference that makes
  intensity normalisation comparable across scanners (R2).
- Week 3 applies a white-matter mask to remove false positives (ROADMAP 6.3.1).
- Week 4 cannot define "deep white matter" (R5) without one.

Leaving it to Week 3 would have blocked all three.

**Why T1 and not FLAIR.** FLAIR nulls the CSF signal by design — that is the
whole point of the sequence — so CSF and white matter are not separable on it by
intensity. T1 gives the classic three-way contrast (CSF dark, grey matter mid,
white matter bright). `orig/T1.nii.gz` is already resampled onto the FLAIR grid
by the challenge organisers, sharing shape and affine exactly, so no
registration is involved.

**The trap this stage must be tested against.** White-matter hyperintensities
are bright on FLAIR but iso- to *hypo*-intense on T1 — they look darker than
healthy white matter. An intensity-driven tissue segmenter will therefore tend
to assign lesions to the grey-matter class. If that happens and Week 3 then uses
the white-matter mask to constrain its search, the mask will exclude precisely
the voxels the whole project is trying to find, and recall will collapse for
reasons that look like a segmentation failure.

`wm_lesion_coverage` in `tissue_quality()` measures this directly, and the
`include_lesion_prone_wm` option is the fix: holes strictly interior to the
white matter are reclaimed into it, since a pocket of "grey matter" fully
surrounded by white matter is anatomically implausible and is far more likely to
be a lesion. This mirrors the "WM + lesion" masks used by BIANCA and similar
tools.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

from preprocessing.sitk_interop import from_sitk, to_sitk

KMEANS = "sitk_kmeans"
GMM = "sklearn_gmm"
METHODS = (KMEANS, GMM)

# Tissue labels, ordered by T1 intensity: CSF darkest, WM brightest.
LABEL_BACKGROUND = 0
LABEL_CSF = 1
LABEL_GM = 2
LABEL_WM = 3
TISSUE_NAMES = {LABEL_CSF: "CSF", LABEL_GM: "GM", LABEL_WM: "WM"}


def segment_tissues(
    t1: np.ndarray,
    brain_mask: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    method: str = KMEANS,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Segment brain voxels into CSF / GM / WM by intensity.

    Returns a label map (0 outside the brain, then 1/2/3 in increasing intensity
    order) and diagnostics. Classes are relabelled by their mean intensity, not
    by whatever arbitrary order the clusterer produced — otherwise "class 3"
    means a different tissue on different subjects and every downstream use is
    silently wrong.
    """
    if t1.shape != brain_mask.shape:
        raise AssertionError(f"T1 shape {t1.shape} != brain mask shape {brain_mask.shape}")
    if not brain_mask.any():
        raise ValueError("Brain mask is empty — cannot segment tissue")
    if not np.isfinite(t1).all():
        raise ValueError("T1 volume contains NaN or Inf")

    values = t1[brain_mask].astype(np.float64)
    if values.max() <= values.min():
        raise ValueError("Degenerate T1 intensity range inside the brain mask")

    if method == KMEANS:
        # SimpleITK's k-means operates on the image, so restrict it to the brain
        # by zeroing outside and letting the class assignment be remapped below.
        masked = np.where(brain_mask, t1, 0.0)
        image = to_sitk(masked, spacing, dtype=sitk.sitkFloat32)
        filt = sitk.ScalarImageKmeansImageFilter()
        # Initial class means spread across the observed range; the filter
        # refines them. Four classes because the zeroed background forms one.
        lo, hi = float(values.min()), float(values.max())
        filt.SetClassWithInitialMean([lo * 0.0, lo + 0.25 * (hi - lo),
                                      lo + 0.5 * (hi - lo), lo + 0.8 * (hi - lo)])
        raw_labels = from_sitk(filt.Execute(image)).astype(np.int16)
    elif method == GMM:
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=3, random_state=seed, covariance_type="full")
        raw = model.fit_predict(values.reshape(-1, 1))
        raw_labels = np.zeros(t1.shape, dtype=np.int16)
        raw_labels[brain_mask] = raw + 1
    else:
        raise ValueError(f"Unknown method {method!r}; expected one of {METHODS}")

    # --- relabel by mean intensity so class order is anatomically meaningful ---
    present = [c for c in np.unique(raw_labels[brain_mask]) if c != 0]
    if len(present) < 3:
        raise ValueError(
            f"Tissue segmentation found only {len(present)} classes inside the brain; "
            f"expected 3 (CSF/GM/WM)"
        )
    means = {c: float(t1[brain_mask & (raw_labels == c)].mean()) for c in present}
    ordered = sorted(means, key=means.get)[-3:]  # the three brightest classes

    labels = np.zeros(t1.shape, dtype=np.uint8)
    for new_label, old_label in zip((LABEL_CSF, LABEL_GM, LABEL_WM), ordered):
        labels[brain_mask & (raw_labels == old_label)] = new_label

    unassigned = int((brain_mask & (labels == 0)).sum())
    voxel_volume_mm3 = float(np.prod(spacing))
    diagnostics = {
        "method": method,
        "class_means_t1": {TISSUE_NAMES[n]: means[o]
                           for n, o in zip((LABEL_CSF, LABEL_GM, LABEL_WM), ordered)},
        "unassigned_brain_voxels": unassigned,
        **{f"{TISSUE_NAMES[n].lower()}_volume_ml": float((labels == n).sum() * voxel_volume_mm3 / 1000.0)
           for n in (LABEL_CSF, LABEL_GM, LABEL_WM)},
        **{f"{TISSUE_NAMES[n].lower()}_fraction": float((labels == n).sum() / brain_mask.sum())
           for n in (LABEL_CSF, LABEL_GM, LABEL_WM)},
    }
    return labels, diagnostics


def white_matter_mask(
    labels: np.ndarray, *, include_lesion_prone_wm: bool = True
) -> np.ndarray:
    """The white-matter mask, optionally reclaiming lesion-shaped holes.

    WMH are hypointense on T1, so an intensity-driven segmenter assigns many of
    them to grey matter — leaving holes inside the white matter. A pocket of
    "grey matter" strictly enclosed by white matter is anatomically implausible
    (grey matter is cortex at the brain surface, plus the deep nuclei), so
    filling enclosed holes reclaims lesions without pulling in cortex, which is
    never enclosed.

    Done in-plane, per slice: a 3D fill would leak through the natural openings
    that exist between slices on a 3 mm grid.
    """
    wm = labels == LABEL_WM
    if not include_lesion_prone_wm:
        return wm

    filled = wm.copy()
    for z in range(wm.shape[2]):
        filled[:, :, z] = ndi.binary_fill_holes(wm[:, :, z])
    return filled


def tissue_quality(
    labels: np.ndarray,
    brain_mask: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    wm_mask: np.ndarray | None = None,
    lesion_mask: np.ndarray | None = None,
) -> dict:
    """Score a tissue segmentation, including the lesion-coverage trap."""
    voxel_volume_mm3 = float(np.prod(spacing))
    quality = {
        "brain_volume_ml": float(brain_mask.sum() * voxel_volume_mm3 / 1000.0),
    }
    for label in (LABEL_CSF, LABEL_GM, LABEL_WM):
        name = TISSUE_NAMES[label].lower()
        quality[f"{name}_fraction"] = float((labels == label).sum() / brain_mask.sum())

    if wm_mask is not None:
        quality["wm_mask_volume_ml"] = float(wm_mask.sum() * voxel_volume_mm3 / 1000.0)
        quality["wm_mask_fraction"] = float(wm_mask.sum() / brain_mask.sum())

    if lesion_mask is not None and lesion_mask.any():
        # THE metric for this stage. WMH are white-matter lesions by definition,
        # so a white-matter mask that excludes them is wrong in the one way that
        # matters — Week 3 would use it to delete the very targets it is hunting.
        raw_wm = labels == LABEL_WM
        quality["wm_lesion_coverage_raw"] = float(raw_wm[lesion_mask].mean())
        if wm_mask is not None:
            quality["wm_lesion_coverage"] = float(wm_mask[lesion_mask].mean())
        # Where do the lesions land if not in WM? Expected answer: grey matter,
        # because WMH are T1-hypointense.
        quality["lesion_in_gm_fraction"] = float((labels[lesion_mask] == LABEL_GM).mean())
        quality["lesion_in_csf_fraction"] = float((labels[lesion_mask] == LABEL_CSF).mean())
    return quality
