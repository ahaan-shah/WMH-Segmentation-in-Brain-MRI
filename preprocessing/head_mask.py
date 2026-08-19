"""Stage 1 — rough head mask (input to N4 only).

This is NOT skull stripping. It separates head tissue from air so that N4 has a
region to fit its bias field in; the brain mask proper is Stage 3. Keeping the
two apart is what ROADMAP Section 5.1's ordering requires: N4 is degraded by
large air regions, but a brain mask computed on an uncorrected image is itself
degraded by the bias field, so the standard resolution is coarse mask -> N4 ->
proper skull strip on the corrected image.

Three properties of this dataset shape the implementation:

**Slices are 3.00 mm everywhere against 0.56-1.30 mm in-plane** — an anisotropy
of 2.5x to 5.4x. Every structuring element here is therefore in-plane only. A
naive 3x3x3 ball spans ~3 mm in-plane but 9 mm through-plane, so it would close
across two entire slices while barely moving in-plane. This is the same reason
the official scorer erodes in 2D (ROADMAP Section 1.3).

**Field of view varies enormously.** Amsterdam covers 249-309 mm through-plane
(83-103 slices, including the neck); Utrecht and Singapore cover 144 mm (48
slices, brain only). Nothing here may assume the head is centred or that the
volume contains only brain.

**Backgrounds are not clean.** Singapore volumes carry visible aliasing/ghosting
streaks outside the head, and 19-37% of voxels are already exactly zero from
upstream masking. Largest-connected-component selection is what removes the
ghosting; without it the mask leaks into streaks and N4 fits a field across
empty space.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu


def in_plane_structure(radius_voxels: int) -> np.ndarray:
    """A (2r+1, 2r+1, 1) structuring element — in-plane only, never through-plane.

    Through-plane morphology is deliberately unavailable from this module: at
    3 mm slices a single voxel step in z is a 3 mm anatomical jump, and treating
    it as equivalent to an in-plane step is how masks silently bleed across
    slices.
    """
    if radius_voxels < 1:
        raise ValueError(f"radius_voxels must be >= 1, got {radius_voxels}")
    size = 2 * radius_voxels + 1
    return np.ones((size, size, 1), dtype=bool)


def _radius_in_voxels(radius_mm: float, spacing: tuple[float, float, float]) -> int:
    """Convert an in-plane radius in mm to voxels, using the finer in-plane axis.

    Spacing is read from the header, never assumed (Decision #6). In-plane
    spacing ranges 0.56-1.30 mm across this dataset, so a fixed voxel radius
    would mean a 2.3x different physical size at Amsterdam/Philips than at
    Utrecht.
    """
    in_plane = min(spacing[0], spacing[1])
    return max(1, int(round(radius_mm / in_plane)))


def largest_connected_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep only the largest 26-connected component. Returns (mask, n_components).

    26-connectivity matches checks/evaluation.py (SetFullyConnected(True)), so
    connectivity conventions stay consistent across the whole project.
    """
    if not mask.any():
        raise ValueError("Cannot take the largest component of an empty mask")
    labelled, n_components = ndi.label(mask, structure=np.ones((3, 3, 3)))
    if n_components == 1:
        return mask, 1
    counts = np.bincount(labelled.ravel())
    counts[0] = 0  # background is not a candidate
    return labelled == counts.argmax(), int(n_components)


def compute_head_mask(
    array: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    closing_radius_mm: float,
    opening_radius_mm: float,
    n_classes: int = 3,
) -> tuple[np.ndarray, dict]:
    """Segment head tissue from air on a FLAIR volume.

    Returns the boolean mask and a diagnostics dict. The diagnostics are logged
    per subject and aggregated into the Stage 1 QC table — a mask that silently
    succeeds records nothing, and outlier detection across 170 subjects is what
    replaces inspecting 170 masks by eye.

    Fails loud (Section 5.2) on a degenerate image rather than returning an
    empty or whole-volume mask that would quietly wreck the N4 fit downstream.
    """
    if not np.isfinite(array).all():
        raise ValueError("FLAIR volume contains NaN or Inf")
    if array.max() <= array.min():
        raise ValueError(f"Degenerate FLAIR intensity range: [{array.min()}, {array.max()}]")

    # Multi-Otsu with three classes, keeping everything above the LOWEST
    # threshold (i.e. everything that is not air).
    #
    # Plain two-class Otsu was tried first and rejected on measurement: it
    # assumes a bimodal histogram, but these volumes are trimodal — air near
    # zero, brain tissue, and very bright scalp fat. On Utrecht subject 11,
    # whose scalp is exceptionally bright, two-class Otsu placed the threshold
    # at 635 (between brain and scalp) and returned a mask containing only the
    # scalp ring: it retained 25.6% of that subject's reference WMH voxels,
    # meaning the brain itself was largely outside the "head" mask. Across the
    # 60 training subjects, two-class Otsu failed 1 of 60 that way; three-class
    # multi-Otsu retains >= 0.9997 of reference WMH on every subject.
    thresholds = threshold_multiotsu(array, classes=n_classes)
    threshold = float(thresholds[0])
    raw = array > threshold
    if not raw.any():
        raise ValueError(f"Multi-Otsu threshold {threshold} selected no voxels")

    open_r = _radius_in_voxels(opening_radius_mm, spacing)
    close_r = _radius_in_voxels(closing_radius_mm, spacing)

    # Opening first: removes isolated speckle and thin ghosting streaks before
    # they can bridge into the head and survive component selection.
    opened = ndi.binary_opening(raw, structure=in_plane_structure(open_r))
    if not opened.any():  # pathological, but never silently continue
        opened = raw

    head, n_components = largest_connected_component(opened)

    # Closing after component selection: seals the skull/scalp boundary without
    # having first bridged the head to a neighbouring streak.
    closed = ndi.binary_closing(head, structure=in_plane_structure(close_r))

    # Fill enclosed cavities: sinuses, ventricles, orbits. N4 must fit across
    # the whole head interior, and a mask riddled with internal holes biases the
    # field estimate towards whatever remains.
    filled = ndi.binary_fill_holes(closed)
    for z in range(filled.shape[2]):  # cavities open in 3D but closed in-plane
        filled[:, :, z] = ndi.binary_fill_holes(filled[:, :, z])

    voxel_volume_mm3 = float(np.prod(spacing))
    diagnostics = {
        "threshold": threshold,
        "threshold_method": f"multiotsu_{n_classes}class_lowest",
        "all_thresholds": [float(t) for t in thresholds],
        "opening_radius_voxels": open_r,
        "closing_radius_voxels": close_r,
        "n_components_before_selection": n_components,
        "n_voxels_raw_threshold": int(raw.sum()),
        "n_voxels_head": int(filled.sum()),
        "head_volume_ml": float(filled.sum() * voxel_volume_mm3 / 1000.0),
        "head_fraction_of_volume": float(filled.mean()),
        "filled_voxels_added": int(filled.sum() - head.sum()),
    }
    return filled, diagnostics
