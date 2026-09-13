"""Week 4 — quantitative lesion features (R6, R7, R8, R9).

Everything here is computed in **millimetres and millilitres via the affine**,
never in voxel counts. Slices are 3.00 mm against 0.56-1.30 mm in-plane, so any
length or volume derived from voxel counts is wrong by up to 5.4x depending on
which way it runs. The phantom suite (`checks/test_phantom.py`) pins this.

Requirements covered:

- **R6 lesion count** — connected components at 26-connectivity, matching the
  official scorer. 6-connectivity is reported alongside because on a 3 mm slice
  grid the choice changes the count by about 30%, and showing awareness of that
  is worth more than a single number.
- **R7 total lesion volume** — in mL.
- **R8 largest lesion, volume AND diameter** — see the note below on which
  diameter and why.
- **R9 spatial distribution** — hemispheres now, from world coordinates.

**The diameter decision (R8).** Two definitions give different answers:

- *Equivalent-sphere diameter*: the width of a sphere with the same volume.
  Simple, always defined, cheap.
- *Maximum Feret diameter*: the greatest distance between any two voxels of the
  lesion — its true longest span.

`dataset.yaml` locked equivalent-sphere in Week 1. **We now report maximum Feret
as primary and equivalent-sphere alongside**, and the reason is evidence that
did not exist in Week 1: having now seen the lesions, the large ones in this
dataset are long thin caps hugging the ventricles, nothing like spheres. A
60 mm x 5 mm lesion measures ~60 mm by Feret and ~20 mm by equivalent-sphere —
and equivalent-sphere *always* under-reports elongation, so the error is a
systematic bias that grows with disease severity rather than random noise.
Reporting both costs almost nothing and makes the choice defensible instead of
arbitrary.

Feret is computed on the **convex hull** of the lesion's world coordinates,
because the two most distant points of any set are always hull vertices — this
turns an O(n^2) search over every voxel pair into one over a handful.

**Hemisphere assignment (R9) never uses array indices.** Left and right come
from world coordinates through the affine, with the midline taken from the brain
mask's own centroid rather than assumed to be x = 0. Deriving side from a voxel
index would silently invert on any subject stored in a different orientation —
the class of bug CLAUDE.md Section 11.3 exists to make structurally impossible.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from metadata.geometry import world_coordinates
from preprocessing.morphology import CONNECTIVITY_6, CONNECTIVITY_26


def maximum_feret_diameter_mm(voxel_coords: np.ndarray, affine: np.ndarray) -> float:
    """Greatest distance in mm between any two voxels of one lesion.

    Operates on world coordinates, so anisotropic spacing is handled by the
    affine rather than by hand. A single-voxel lesion has a Feret diameter of
    0 mm by this definition — it has no extent — which is reported honestly
    rather than substituted with a voxel width.
    """
    if len(voxel_coords) == 0:
        raise ValueError("Empty lesion component")
    world = world_coordinates(voxel_coords.astype(float), affine)
    if len(world) == 1:
        return 0.0
    if len(world) <= 4:
        differences = world[:, None, :] - world[None, :, :]
        return float(np.sqrt((differences ** 2).sum(axis=-1)).max())

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(world)
        vertices = world[hull.vertices]
    except Exception:
        # Degenerate (coplanar/collinear) lesions make a 3D hull impossible.
        # Falling back to all points is correct, just slower.
        vertices = world

    differences = vertices[:, None, :] - vertices[None, :, :]
    return float(np.sqrt((differences ** 2).sum(axis=-1)).max())


def equivalent_sphere_diameter_mm(volume_mm3: float) -> float:
    """Diameter of a sphere with the same volume: d = (6V/pi)^(1/3)."""
    if volume_mm3 < 0:
        raise ValueError("Negative volume")
    return float((6.0 * volume_mm3 / np.pi) ** (1.0 / 3.0))


def hemisphere_midline_x(brain_mask: np.ndarray, affine: np.ndarray) -> float:
    """World x of the brain's own midline.

    Estimated from the brain mask centroid rather than assuming x = 0, because
    subjects are not centred in the scanner and the affine's origin is arbitrary
    (CLAUDE.md Section 11.3).
    """
    if not brain_mask.any():
        raise ValueError("Empty brain mask")
    centroid_voxel = np.array(ndi.center_of_mass(brain_mask))[None, :]
    return float(world_coordinates(centroid_voxel, affine)[0, 0])


def extract_features(
    lesion_mask: np.ndarray,
    affine: np.ndarray,
    voxel_volume_mm3: float,
    *,
    brain_mask: np.ndarray | None = None,
    source: str = "",
) -> dict:
    """All of R6-R9 for one subject's lesion mask.

    `source` labels where the mask came from ("reference" or "prediction"), which
    matters because Weeks 5-6 must take their target from the reference and their
    features from the prediction — using reference-derived features to predict a
    reference-derived label would let the model invert its own target and report
    a meaningless near-perfect accuracy.
    """
    features = {"source": source}

    # --- R7: total burden ---
    n_voxels = int(lesion_mask.sum())
    features["total_lesion_voxels"] = n_voxels
    features["total_lesion_volume_ml"] = float(n_voxels * voxel_volume_mm3 / 1000.0)

    if n_voxels == 0:
        # Possible for a prediction, never for a reference in this cohort
        # (minimum burden 0.78 mL). Recorded rather than crashed on.
        features.update({
            "lesion_count_26conn": 0, "lesion_count_6conn": 0,
            "largest_lesion_volume_ml": 0.0, "largest_lesion_feret_mm": 0.0,
            "largest_lesion_equiv_sphere_mm": 0.0,
            "left_lesion_volume_ml": 0.0, "right_lesion_volume_ml": 0.0,
            "laterality_index": float("nan"), "mean_lesion_volume_ml": 0.0,
            "small_lesion_count_le5vox": 0,
        })
        return features

    # --- R6: lesion count, both connectivities ---
    labelled, n_26 = ndi.label(lesion_mask, structure=CONNECTIVITY_26)
    _, n_6 = ndi.label(lesion_mask, structure=CONNECTIVITY_6)
    features["lesion_count_26conn"] = int(n_26)
    features["lesion_count_6conn"] = int(n_6)

    sizes = np.bincount(labelled.ravel())[1:]
    features["mean_lesion_volume_ml"] = float(sizes.mean() * voxel_volume_mm3 / 1000.0)
    # Half this cohort's reference lesions are <=5 voxels; carrying the count
    # makes that visible in the feature table rather than only in Week 2's notes.
    features["small_lesion_count_le5vox"] = int((sizes <= 5).sum())

    # --- R8: largest lesion, volume and BOTH diameters ---
    largest_label = int(np.argmax(sizes)) + 1
    largest_voxels = np.argwhere(labelled == largest_label)
    largest_volume_mm3 = float(len(largest_voxels) * voxel_volume_mm3)
    features["largest_lesion_volume_ml"] = largest_volume_mm3 / 1000.0
    features["largest_lesion_feret_mm"] = maximum_feret_diameter_mm(largest_voxels, affine)
    features["largest_lesion_equiv_sphere_mm"] = equivalent_sphere_diameter_mm(largest_volume_mm3)

    # --- R9: hemispheres, via world coordinates ---
    if brain_mask is not None:
        midline = hemisphere_midline_x(brain_mask, affine)
        lesion_voxels = np.argwhere(lesion_mask)
        world_x = world_coordinates(lesion_voxels.astype(float), affine)[:, 0]
        # RAS convention after canonicalisation: +x is the patient's RIGHT.
        right = int((world_x > midline).sum())
        left = n_voxels - right
        features["left_lesion_volume_ml"] = float(left * voxel_volume_mm3 / 1000.0)
        features["right_lesion_volume_ml"] = float(right * voxel_volume_mm3 / 1000.0)
        total = left + right
        features["laterality_index"] = float((left - right) / total) if total else float("nan")
        features["midline_world_x_mm"] = midline

    return features
