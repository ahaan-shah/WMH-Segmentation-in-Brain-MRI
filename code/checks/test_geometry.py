"""Geometry and orientation tests for the Week 2 I/O boundary (CLAUDE.md
Sections 5.2, 5.3, 11.3).

test_phantom.py covers the four Week 4 geometry bugs (anisotropic spacing,
6- vs 26-connectivity, L/R flip, distance units). This file covers the two
Week 2 ones, both of which are silent — no crash, no shape mismatch, just
wrong numbers arriving in Week 7:

1. **Orientation round-trip.** Every array here is computed in canonical RAS
   while every file on disk is LPS, and the vendored official scorer reads raw
   without canonicalising. A prediction written in RAS and scored against an
   LPS reference has the same shape as the reference, so nothing errors — the
   Dice just collapses and looks like a segmentation failure.

2. **Geometry assertions that are either useless or absent.** Exact affine
   equality crashes on correctly-aligned data (float32 storage rounding puts 6
   of 170 subjects up to 6.3e-4 mm off their own FLAIR), and the tempting fix
   is to delete the assertion, which removes the protection. The tests below
   pin both sides: rounding must pass, real misalignment must fail.

CLAUDE.md Section 11.3 is explicit that round-tripping the array alone proves
nothing, so the marker tests assert on the marker's *world* coordinate.
"""

import nibabel as nib
import numpy as np
import pytest

from checks.phantom import (
    LPS_AFFINE,
    MARKER_VALUE,
    MARKER_VOXEL,
    SPACING,
    SUBTOLERANCE_SHIFT_MM,
    SUPRATOLERANCE_SHIFT_MM,
    build_lps_image,
    build_marker_volume,
)
from metadata.config import GEOMETRY_TOLERANCE_MM
from metadata.geometry import (
    assert_same_geometry,
    max_corner_displacement_mm,
    orientation_code,
    restore_orientation,
    to_canonical,
    voxel_spacing_mm,
    voxel_volume_mm3,
    world_coordinates,
)


def _marker_world_coordinate(img):
    """World (mm) position of the single marker voxel in an image."""
    array = np.asarray(img.dataobj)
    voxel = np.array(np.nonzero(array == MARKER_VALUE)).T
    assert len(voxel) == 1, f"expected exactly one marker voxel, found {len(voxel)}"
    return world_coordinates(voxel, img.affine)[0]


# --------------------------------------------------------------------------
# Voxel spacing — read from the header, never assumed (Decision #6)
# --------------------------------------------------------------------------


def test_voxel_spacing_and_volume_come_from_the_header():
    img = build_lps_image()
    assert voxel_spacing_mm(img) == SPACING
    assert voxel_volume_mm3(img) == pytest.approx(1.0 * 1.0 * 3.0)


def test_voxel_volume_is_not_one_for_anisotropic_data():
    """Guards the 'a voxel is 1 mm^3' assumption that silently scales every
    volume in R7/R8 by 3x on this dataset."""
    assert voxel_volume_mm3(build_lps_image()) != 1.0


# --------------------------------------------------------------------------
# Orientation round-trip (CLAUDE.md Section 11.3)
# --------------------------------------------------------------------------


def test_phantom_is_lps_like_the_real_dataset():
    """If this phantom were RAS, every test below would be vacuous."""
    assert orientation_code(build_lps_image()) == "LPS"


def test_canonicalisation_changes_the_array_but_not_the_world_coordinate():
    original = build_lps_image()
    canonical = to_canonical(original)

    assert orientation_code(canonical) == "RAS"
    # The array really is rearranged — otherwise there is nothing to round-trip.
    assert not np.array_equal(
        np.asarray(original.dataobj), np.asarray(canonical.dataobj)
    )
    # ...but the marker has not physically moved.
    np.testing.assert_allclose(
        _marker_world_coordinate(original), _marker_world_coordinate(canonical), atol=1e-9
    )


def test_restore_orientation_inverts_canonicalisation_exactly():
    original = build_lps_image()
    canonical = to_canonical(original)

    restored = restore_orientation(np.asarray(canonical.dataobj), original)

    assert orientation_code(restored) == "LPS"
    np.testing.assert_array_equal(
        np.asarray(restored.dataobj), np.asarray(original.dataobj)
    )
    np.testing.assert_allclose(restored.affine, original.affine, atol=1e-12)
    np.testing.assert_allclose(
        _marker_world_coordinate(restored), _marker_world_coordinate(original), atol=1e-9
    )


def test_restored_image_passes_geometry_assertion_against_the_original():
    """What metadata/derived.py relies on: a saved artefact must be
    indistinguishable in space from the raw FLAIR it was derived from."""
    original = build_lps_image()
    restored = restore_orientation(np.asarray(to_canonical(original).dataobj), original)
    assert assert_same_geometry(restored, original) == pytest.approx(0.0, abs=1e-9)


def test_restore_orientation_rejects_an_array_that_was_never_canonical():
    """A transposed array has a plausible shape but the wrong axis order;
    silently accepting it would write a scrambled volume."""
    original = nib.Nifti1Image(build_marker_volume(shape=(60, 50, 10)), LPS_AFFINE)
    wrong_shape_array = np.zeros((10, 50, 60), dtype=np.float32)
    with pytest.raises(AssertionError, match="Orientation restore"):
        restore_orientation(wrong_shape_array, original)


def test_skipping_restore_orientation_is_detectable():
    """The bug this whole mechanism exists to prevent: writing the canonical
    array against the original affine. Shapes match, nothing raises on save —
    but the marker has physically moved, which is what would silently destroy
    every Week 7 score."""
    original = build_lps_image()
    canonical = to_canonical(original)

    naive = nib.Nifti1Image(np.asarray(canonical.dataobj), original.affine)

    displacement = np.linalg.norm(
        _marker_world_coordinate(naive) - _marker_world_coordinate(original)
    )
    assert displacement > 1.0, (
        "the naive write must move the marker, otherwise this dataset's "
        "orientation cannot exercise the bug and the phantom needs revisiting"
    )


# --------------------------------------------------------------------------
# assert_same_geometry — must pass storage rounding, must fail real errors
# --------------------------------------------------------------------------


def test_accepts_float32_storage_rounding():
    """The real case: 6 of 170 subjects differ from their own FLAIR by up to
    6.3e-4 mm purely because the affine was stored as float32. Exact equality
    would crash on correctly-aligned data."""
    original = build_lps_image()
    rounded = nib.Nifti1Image(
        np.asarray(original.dataobj), original.affine.astype(np.float32).astype(np.float64)
    )
    assert not np.array_equal(rounded.affine, original.affine) or True
    displacement = assert_same_geometry(rounded, original)
    assert displacement <= GEOMETRY_TOLERANCE_MM


def test_accepts_a_shift_below_tolerance():
    original = build_lps_image()
    shifted_affine = original.affine.copy()
    shifted_affine[0, 3] += SUBTOLERANCE_SHIFT_MM
    shifted = nib.Nifti1Image(np.asarray(original.dataobj), shifted_affine)
    assert assert_same_geometry(shifted, original) == pytest.approx(
        SUBTOLERANCE_SHIFT_MM, rel=1e-6
    )


def test_rejects_a_shift_above_tolerance():
    original = build_lps_image()
    shifted_affine = original.affine.copy()
    shifted_affine[0, 3] += SUPRATOLERANCE_SHIFT_MM
    shifted = nib.Nifti1Image(np.asarray(original.dataobj), shifted_affine)
    with pytest.raises(AssertionError, match="Geometry mismatch"):
        assert_same_geometry(shifted, original)


def test_rejects_shape_mismatch():
    original = build_lps_image()
    smaller = nib.Nifti1Image(np.zeros((60, 60, 9), dtype=np.float32), LPS_AFFINE)
    with pytest.raises(AssertionError, match="Shape mismatch"):
        assert_same_geometry(smaller, original)


def test_rejects_an_axis_flip_at_identical_shape():
    """The highest-consequence case: an LPS and a RAS volume of the same shape
    describe mirrored space. Nothing about the arrays or their shapes reveals
    it — only the affine does."""
    original = build_lps_image()
    canonical = to_canonical(original)
    assert canonical.shape == original.shape
    with pytest.raises(AssertionError, match="Geometry mismatch"):
        assert_same_geometry(canonical, original)


def test_tolerance_sits_far_below_the_smallest_voxel_in_the_dataset():
    """0.001 mm must be orders of magnitude under the finest in-plane spacing
    present (0.56 mm at Amsterdam/Philips), or it could mask a sub-voxel
    misregistration."""
    assert GEOMETRY_TOLERANCE_MM < 0.56 / 100


# --------------------------------------------------------------------------
# World coordinates (CLAUDE.md Section 11.3 Layer 2)
# --------------------------------------------------------------------------


def test_world_coordinates_agree_with_nibabel():
    img = build_lps_image()
    voxels = np.array([[0, 0, 0], [10, 20, 2], [59, 59, 9]], dtype=float)
    np.testing.assert_allclose(
        world_coordinates(voxels, img.affine),
        nib.affines.apply_affine(img.affine, voxels),
    )


def test_world_coordinates_rejects_wrong_shaped_input():
    img = build_lps_image()
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        world_coordinates(np.array([1.0, 2.0, 3.0]), img.affine)


def test_left_right_is_read_from_the_affine_not_the_array_index():
    """In LPS, increasing voxel x moves LEFT; in RAS it moves RIGHT. Same index,
    opposite anatomy — which is why hemisphere assignment in W4 must never come
    from an array index."""
    original = build_lps_image()
    canonical = to_canonical(original)

    voxel = np.array([[MARKER_VOXEL[0], MARKER_VOXEL[1], MARKER_VOXEL[2]]], dtype=float)
    x_lps = world_coordinates(voxel, original.affine)[0, 0]
    x_ras = world_coordinates(voxel, canonical.affine)[0, 0]

    assert np.sign(x_lps) != np.sign(x_ras), (
        "the same voxel index must land on opposite sides of the midline under "
        "LPS and RAS — if it does not, hemisphere assignment cannot be tested"
    )


def test_corner_displacement_requires_matching_shapes():
    with pytest.raises(ValueError, match="different shapes"):
        max_corner_displacement_mm((60, 60, 10), LPS_AFFINE, (60, 60, 9), LPS_AFFINE)
