"""Stage 7 — morphological operators, implemented here and applied in Week 3.

The Week 2 schedule lists "morphological processing", but in this pipeline its
real use is false-positive removal *after* segmentation (ROADMAP 5.5). So the
operators are built and characterised here, and consumed in Week 3. Both reports
say so, rather than one of them appearing to skip the step.

Several of these already existed, scattered across the stages that needed them
first — `in_plane_structure` and `largest_connected_component` in
`head_mask.py`, `dilate_in_plane` in `skull_strip.py`. They are re-exported here
so Week 3 has one place to import from, rather than reaching into Stage 1's
module for a general-purpose tool.

**Everything is in-plane.** Slices are 3.00 mm against 0.56-1.30 mm in-plane, so
a 3x3x3 structuring element reaches roughly three times further through the
brain than across it. A "3x3x3 opening" is not a small operation on this grid.

**The measurement that matters for Week 3.** Removing small connected components
is the standard false-positive filter, and ROADMAP 6.3 flags it as trading
directly between two scored metrics — it raises lesion F1 by deleting noise
specks but costs lesion recall by deleting real small lesions. On this dataset
that trade is unusually harsh, because 49.6% of reference lesions are 5 voxels
or fewer. `minimum_size_cost()` quantifies exactly what each threshold would
delete from the *reference* masks, which is the ceiling any Week 3 filter
operates under: a size filter cannot do better on true lesions than this table
says. Week 3 still has to tune the threshold on predictions, on the validation
split only, and report the trade-off curve.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from preprocessing.head_mask import in_plane_structure, largest_connected_component
from preprocessing.skull_strip import dilate_in_plane

__all__ = [
    "in_plane_structure",
    "largest_connected_component",
    "dilate_in_plane",
    "erode_in_plane",
    "open_in_plane",
    "close_in_plane",
    "fill_holes_in_plane",
    "remove_small_components",
    "minimum_size_cost",
    "CONNECTIVITY_6",
    "CONNECTIVITY_26",
]

# 26-connectivity is the project default, matching checks/evaluation.py
# (SetFullyConnected(True)). 6-connectivity is kept because Week 4 reports lesion
# counts under both (ROADMAP 7.2) — with 3 mm slices the choice materially
# changes the count, and showing awareness of that is worth more than one number.
CONNECTIVITY_6 = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ],
    dtype=bool,
)
CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)


def _radius_in_voxels(radius_mm: float, spacing: tuple[float, float, float]) -> int:
    """In-plane radius in mm -> voxels, using the finer in-plane axis.

    Radii are specified in mm throughout, because in-plane spacing ranges
    0.56-1.30 mm across this dataset and a fixed voxel radius would mean a 2.3x
    different physical operation at Amsterdam/Philips than at Utrecht.
    """
    return max(1, int(round(radius_mm / min(spacing[0], spacing[1]))))


def erode_in_plane(mask: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    """Shrink a mask in-plane. Removes thin protrusions and single-voxel spurs."""
    return ndi.binary_erosion(
        mask, structure=in_plane_structure(_radius_in_voxels(radius_mm, spacing))
    )


def open_in_plane(mask: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    """Erode then dilate: deletes structures thinner than the element, keeps the rest.

    The classic speck remover — and on this dataset the classic lesion remover
    too. See `minimum_size_cost` before applying it to anything.
    """
    return ndi.binary_opening(
        mask, structure=in_plane_structure(_radius_in_voxels(radius_mm, spacing))
    )


def close_in_plane(mask: np.ndarray, radius_mm: float, spacing) -> np.ndarray:
    """Dilate then erode: seals small gaps without inflating the overall shape."""
    return ndi.binary_closing(
        mask, structure=in_plane_structure(_radius_in_voxels(radius_mm, spacing))
    )


def fill_holes_in_plane(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes slice by slice.

    Per-slice rather than 3D: a 3D fill leaks through the natural openings that
    exist between slices on a 3 mm grid, and would swallow regions that are not
    enclosed at all.
    """
    filled = mask.copy()
    for z in range(mask.shape[2]):
        filled[:, :, z] = ndi.binary_fill_holes(mask[:, :, z])
    return filled


def remove_small_components(
    mask: np.ndarray, minimum_voxels: int, *, connectivity: np.ndarray = CONNECTIVITY_26
) -> tuple[np.ndarray, dict]:
    """Delete connected components smaller than `minimum_voxels`.

    Returns the filtered mask and what it cost, so the caller can report the
    trade rather than silently absorbing it.
    """
    if minimum_voxels <= 1:
        return mask.copy(), {"minimum_voxels": minimum_voxels, "components_before": 0,
                             "components_removed": 0, "voxels_removed": 0}

    labelled, n_components = ndi.label(mask, structure=connectivity)
    if n_components == 0:
        return mask.copy(), {"minimum_voxels": minimum_voxels, "components_before": 0,
                             "components_removed": 0, "voxels_removed": 0}

    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    too_small = np.flatnonzero((sizes > 0) & (sizes < minimum_voxels))
    filtered = mask & ~np.isin(labelled, too_small)

    return filtered, {
        "minimum_voxels": minimum_voxels,
        "components_before": int(n_components),
        "components_removed": int(len(too_small)),
        "voxels_removed": int(sizes[too_small].sum()),
    }


def minimum_size_cost(
    lesion_mask: np.ndarray,
    voxel_volume_mm3: float,
    thresholds: tuple[int, ...] = (2, 3, 5, 10, 20),
    *,
    connectivity: np.ndarray = CONNECTIVITY_26,
) -> list[dict]:
    """What each minimum-size threshold would delete from a REFERENCE mask.

    This is the ceiling on any Week 3 size filter: it cannot lose fewer true
    lesions than this. Reported as both a lesion count fraction and a volume
    fraction, because the two diverge enormously here — small lesions are half
    the population but a rounding error of the volume, so a filter that looks
    harmless on Dice can be devastating on lesion F1.
    """
    labelled, n_components = ndi.label(lesion_mask, structure=connectivity)
    if n_components == 0:
        raise ValueError("Empty lesion mask")
    sizes = np.bincount(labelled.ravel())[1:]
    total_voxels = int(sizes.sum())

    rows = []
    for threshold in thresholds:
        removed = sizes[sizes < threshold]
        rows.append({
            "minimum_voxels": threshold,
            "minimum_volume_mm3": threshold * voxel_volume_mm3,
            "lesions_total": int(n_components),
            "lesions_removed": int(len(removed)),
            "lesion_count_fraction_removed": float(len(removed) / n_components),
            "volume_fraction_removed": float(removed.sum() / total_voxels),
        })
    return rows
