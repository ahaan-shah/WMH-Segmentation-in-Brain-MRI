"""Geometry assertions and orientation handling for images crossing function
boundaries (CLAUDE.md Section 5.2, Section 11.3).

Two jobs, both of which exist to make a whole class of silent bug impossible
rather than merely tested-for.

**1. "Same space?" checks that don't false-alarm.**

CLAUDE.md Section 5.2 requires asserting shape and affine at every boundary
where images cross. Doing that with exact equality does not work on this
dataset: `orig/T1.nii.gz`, `pre/FLAIR.nii.gz` and `wmh.nii.gz` are stored with
float32 affines while nibabel promotes to float64, so 6 of the 170 subjects
disagree with their own FLAIR by up to 6.3e-4 mm — pure storage rounding, not
misalignment. A bare `np.array_equal(a.affine, b.affine)` therefore crashes on
correctly-aligned data, and the natural "fix" (dropping the assertion) removes
the protection entirely.

So the check here is geometric rather than numerical: map the eight corners of
the voxel grid into world coordinates through each affine and take the largest
displacement, in millimetres. That number means something physical — "these two
grids describe the same volume of space to within X mm" — and it can be
reported. The tolerance lives in dataset.yaml, set well above the observed
6.3e-4 mm rounding and far below any real misregistration.

**2. Orientation, and the trap it sets for Week 7.**

`loader.load_nifti()` applies `nibabel.as_closest_canonical()`, so every array
this project computes on is RAS. Every file in `data/raw/` is stored LPS. The
vendored official scorer (`checks/evaluation.py`) reads the raw files directly
with SimpleITK and never canonicalises.

That means a prediction written to disk in RAS would be scored against an LPS
reference, silently, with no shape mismatch to catch it — the volumes are the
same shape, just flipped. Dice would collapse and the cause would look like a
segmentation failure rather than an I/O bug.

`restore_orientation()` is the resolution: compute in canonical RAS, write to
disk in the dataset's native orientation. `metadata/derived.py` routes every
saved artefact through it, so the rule is enforced by the only sanctioned write
path rather than by remembering. `test_geometry.py` validates the round-trip
the way CLAUDE.md Section 11.3 demands — by confirming a marker's *world*
coordinate is unchanged, since round-tripping the array alone proves nothing.
"""

from __future__ import annotations

import nibabel as nib
import numpy as np
from nibabel.orientations import (
    apply_orientation,
    io_orientation,
    ornt_transform,
)

from metadata.config import GEOMETRY_TOLERANCE_MM

# nibabel's canonical target: RAS+, i.e. axis 0 -> R, axis 1 -> A, axis 2 -> S,
# each in positive direction. This is what as_closest_canonical() reorients to.
RAS_ORIENTATION = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])


def voxel_spacing_mm(img: nib.spatialimages.SpatialImage) -> tuple[float, float, float]:
    """Voxel spacing in mm, read from the header — never inferred or assumed.

    CLAUDE.md Decision #6. Every FLAIR in this dataset is anisotropic (z = 3.00 mm
    against 0.56-1.30 mm in-plane, a ratio of up to 5.4x), so any distance,
    volume or diameter computed without this is wrong by that factor.
    """
    zooms = img.header.get_zooms()[:3]
    return (float(zooms[0]), float(zooms[1]), float(zooms[2]))


def voxel_volume_mm3(img: nib.spatialimages.SpatialImage) -> float:
    """Volume of a single voxel in mm^3."""
    return float(np.prod(voxel_spacing_mm(img)))


def volume_corners_world(shape: tuple, affine: np.ndarray) -> np.ndarray:
    """World (mm) coordinates of the eight corners of a voxel grid.

    Corners rather than the affine entries themselves: two affines can differ
    numerically while describing the same physical grid, and the quantity we
    actually care about is where the data sits in space.
    """
    i, j, k = (shape[0] - 1, shape[1] - 1, shape[2] - 1)
    corners = np.array(
        [
            [0, 0, 0],
            [i, 0, 0],
            [0, j, 0],
            [0, 0, k],
            [i, j, 0],
            [i, 0, k],
            [0, j, k],
            [i, j, k],
        ],
        dtype=float,
    )
    return nib.affines.apply_affine(affine, corners)


def max_corner_displacement_mm(
    shape_a: tuple, affine_a: np.ndarray, shape_b: tuple, affine_b: np.ndarray
) -> float:
    """Largest distance in mm between corresponding corners of two voxel grids.

    Requires matching shapes; a shape mismatch is not a tolerance question and
    is reported separately by assert_same_geometry().
    """
    if tuple(shape_a[:3]) != tuple(shape_b[:3]):
        raise ValueError(
            f"Cannot compare corner positions across different shapes: {shape_a} vs {shape_b}"
        )
    corners_a = volume_corners_world(shape_a, affine_a)
    corners_b = volume_corners_world(shape_b, affine_b)
    return float(np.linalg.norm(corners_a - corners_b, axis=1).max())


def assert_same_geometry(
    image_a: nib.spatialimages.SpatialImage,
    image_b: nib.spatialimages.SpatialImage,
    *,
    atol_mm: float | None = None,
    context: str = "",
) -> float:
    """Assert two images occupy the same voxel grid and physical space.

    Call this at every boundary where images cross (CLAUDE.md Section 5.2).
    Returns the measured corner displacement in mm so callers can log it —
    a silently-passing assertion records nothing, and Week 2's report needs
    the number.

    Raises rather than warns: CLAUDE.md Section 5.2 prefers a crash over a
    plausible-looking wrong number.
    """
    atol_mm = GEOMETRY_TOLERANCE_MM if atol_mm is None else atol_mm
    where = f" [{context}]" if context else ""

    shape_a = tuple(image_a.shape[:3])
    shape_b = tuple(image_b.shape[:3])
    if shape_a != shape_b:
        raise AssertionError(f"Shape mismatch{where}: {shape_a} vs {shape_b}")

    displacement = max_corner_displacement_mm(
        shape_a, image_a.affine, shape_b, image_b.affine
    )
    if displacement > atol_mm:
        raise AssertionError(
            f"Geometry mismatch{where}: corners differ by {displacement:.6g} mm "
            f"(tolerance {atol_mm:g} mm). The images do not describe the same "
            f"physical volume — this is a real misalignment, not storage rounding."
        )
    return displacement


def world_coordinates(voxel_coords: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Map (N, 3) voxel indices to (N, 3) world coordinates in mm.

    CLAUDE.md Section 11.3 Layer 2: anatomy — left vs right above all — is
    derived from world coordinates through the affine, never from array
    indices. Routing it through the affine makes the L/R flip bug structurally
    impossible rather than something a test has to catch.
    """
    voxel_coords = np.asarray(voxel_coords, dtype=float)
    if voxel_coords.ndim != 2 or voxel_coords.shape[1] != 3:
        raise ValueError(f"Expected (N, 3) voxel coordinates, got {voxel_coords.shape}")
    return nib.affines.apply_affine(affine, voxel_coords)


def orientation_code(img: nib.spatialimages.SpatialImage) -> str:
    """Axis orientation code, e.g. 'LPS' or 'RAS'."""
    return "".join(nib.aff2axcodes(img.affine))


def to_canonical(img: nib.spatialimages.SpatialImage) -> nib.Nifti1Image:
    """Reorient to closest canonical RAS. Axis reorder/flip only — never a resample.

    Identical to what loader.load_nifti() applies; exposed here so code that
    loads an image raw (to inspect its native orientation) can canonicalise
    explicitly rather than reimplementing it.
    """
    return nib.as_closest_canonical(img)


def restore_orientation(
    array_canonical: np.ndarray, original_img: nib.spatialimages.SpatialImage
) -> nib.Nifti1Image:
    """Convert an array computed in canonical RAS back to `original_img`'s orientation.

    This is what keeps files written by this project readable by the vendored
    official scorer, which reads raw and never canonicalises (see module
    docstring). The returned image carries the original affine exactly, so a
    downstream `assert_same_geometry` against the raw FLAIR passes at 0.0 mm.
    """
    array_canonical = np.asarray(array_canonical)
    if array_canonical.dtype == bool:
        raise TypeError(
            "NIfTI has no boolean data type. Cast the mask before reorienting — "
            "metadata/derived.py resolves boolean masks to uint8 for you, so "
            "prefer save_derived() over calling this directly."
        )

    original_ornt = io_orientation(original_img.affine)
    canonical_to_original = ornt_transform(RAS_ORIENTATION, original_ornt)
    array_original = apply_orientation(np.asarray(array_canonical), canonical_to_original)

    if tuple(array_original.shape[:3]) != tuple(original_img.shape[:3]):
        raise AssertionError(
            f"Orientation restore produced shape {array_original.shape}, expected "
            f"{original_img.shape} — the input array was not in canonical space "
            f"for this image."
        )
    return nib.Nifti1Image(array_original, original_img.affine)
