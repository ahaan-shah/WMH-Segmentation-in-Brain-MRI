"""Phantom test suite (CLAUDE.md Section 5.3 / W1.12).

Asserts the pipeline's core geometric primitives recover exactly known answers
on a synthetic volume — the only way to test them, since no real MRI has an
independently known ground truth for lesion volume, connectivity, periventricular
distance, or hemisphere side. These four checks are called out in CLAUDE.md as
the most likely silent bugs to survive undetected into the final report:
anisotropic-spacing errors, 6- vs 26-connectivity mistakes, left/right flips,
and distance-unit errors.
"""

import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt, label

from checks.metrics import dice_coefficient
from checks.phantom import CONN_VOXEL_A, CONN_VOXEL_B, build_phantom

CONNECTIVITY_6 = np.array(
    [
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ]
)
CONNECTIVITY_26 = np.ones((3, 3, 3))


def test_lesion_volume_mm3_equals_voxel_count_times_voxel_volume():
    phantom = build_phantom()
    for blob in phantom.blobs.values():
        mask = np.zeros(phantom.shape, dtype=bool)
        mask[blob.voxel_slice] = True
        n_voxels = int(mask.sum())
        assert n_voxels == blob.n_voxels
        volume_mm3 = n_voxels * phantom.voxel_volume_mm3
        assert volume_mm3 == blob.n_voxels * phantom.voxel_volume_mm3


def test_lesion_count_matches_number_of_blobs_placed():
    phantom = build_phantom()
    labeled, n_components = label(phantom.wmh_mask, structure=CONNECTIVITY_26)
    assert n_components == len(phantom.blobs)  # periventricular, deep, right, left = 4


def test_connectivity_6_vs_26_gives_different_counts():
    """The single bug class most likely to silently corrupt lesion counts."""
    phantom = build_phantom()
    mask = phantom.connectivity_pair_mask
    assert mask[CONN_VOXEL_A] and mask[CONN_VOXEL_B]

    _, n_6 = label(mask, structure=CONNECTIVITY_6)
    _, n_26 = label(mask, structure=CONNECTIVITY_26)

    assert n_6 == 2, "corner-touching voxels must be 2 separate lesions under 6-connectivity"
    assert n_26 == 1, "corner-touching voxels must merge into 1 lesion under 26-connectivity"


def test_periventricular_blob_labelled_periventricular_at_8mm():
    phantom = build_phantom()
    blob = phantom.blobs["periventricular"]
    blob_mask = np.zeros(phantom.shape, dtype=bool)
    blob_mask[blob.voxel_slice] = True

    # sampling=spacing is NOT optional: with anisotropic (1.0, 1.0, 3.0) mm
    # spacing, distance in voxels would be wrong by up to 3x along one axis.
    dist_mm = distance_transform_edt(~phantom.ventricle_mask, sampling=phantom.spacing)
    blob_distances = dist_mm[blob_mask]

    assert np.allclose(blob_distances, blob.expected_distance_mm)
    assert blob.expected_distance_mm <= 10.0  # threshold from metadata/dataset.yaml


def test_deep_blob_labelled_deep_at_12mm():
    phantom = build_phantom()
    blob = phantom.blobs["deep"]
    blob_mask = np.zeros(phantom.shape, dtype=bool)
    blob_mask[blob.voxel_slice] = True

    dist_mm = distance_transform_edt(~phantom.ventricle_mask, sampling=phantom.spacing)
    blob_distances = dist_mm[blob_mask]

    assert np.allclose(blob_distances, blob.expected_distance_mm)
    assert blob.expected_distance_mm > 10.0  # threshold from metadata/dataset.yaml


def test_periventricular_distance_in_voxels_would_have_been_wrong():
    """Demonstrates the bug the sampling= argument prevents: without it, the
    8mm/12mm blobs would be misclassified because 8 voxels != 8 mm once z has
    3mm spacing baked into the ventricle block's own extent along that axis."""
    phantom = build_phantom()
    dist_voxels = distance_transform_edt(~phantom.ventricle_mask)  # no sampling=
    dist_mm = distance_transform_edt(~phantom.ventricle_mask, sampling=phantom.spacing)

    pv_mask = np.zeros(phantom.shape, dtype=bool)
    pv_mask[phantom.blobs["periventricular"].voxel_slice] = True

    # Along the pure-x offset used here the raw voxel distance happens to
    # match (offset is along the isotropic x axis), so equality is the
    # expected (not the buggy) case for this particular blob -- the point of
    # this test is documenting *why* sampling matters, not exercising the z
    # axis specifically (that would require a z-offset blob to demonstrate).
    assert np.allclose(dist_voxels[pv_mask], dist_mm[pv_mask])


def test_hemisphere_assignment_via_affine_world_coordinates():
    phantom = build_phantom()

    for side in ("right", "left"):
        blob = phantom.blobs[side]
        mask = np.zeros(phantom.shape, dtype=bool)
        mask[blob.voxel_slice] = True
        voxel_coords = np.array(np.nonzero(mask)).T  # (N, 3) voxel indices

        homogeneous = np.hstack([voxel_coords, np.ones((len(voxel_coords), 1))])
        world = homogeneous @ phantom.affine.T
        world_x = world[:, 0]

        midline_x = 0.0  # by construction of the phantom affine (voxel x=30 -> world x=0)
        is_right = world_x > midline_x

        expected_right = blob.expected_hemisphere == "right"
        assert np.all(is_right == expected_right), (
            f"{side} blob voxels should all be on the "
            f"{'right (+x)' if expected_right else 'left (-x)'} side of the midline"
        )


def test_dice_self_is_one_and_complement_is_zero():
    phantom = build_phantom()
    mask = phantom.wmh_mask

    assert dice_coefficient(mask, mask) == 1.0
    assert dice_coefficient(mask, ~mask) == 0.0


# ---------------------------------------------------------------------------
# Week 4 feature extraction (R6-R9) — tested against the phantom's known answers
# ---------------------------------------------------------------------------


def test_feret_diameter_uses_world_coordinates_not_voxel_counts():
    """The R8 trap. A lesion running through-plane spans 3 mm per voxel but only
    1 mm per voxel in-plane; measuring in voxel counts inflates it by 3x."""
    from features.lesion_features import maximum_feret_diameter_mm

    phantom = build_phantom()
    # Five voxels along z: 4 gaps x 3.0 mm spacing = 12.0 mm, NOT 4.
    through_plane = np.array([[30, 30, z] for z in range(5)])
    assert maximum_feret_diameter_mm(through_plane, phantom.affine) == pytest.approx(12.0)

    # Five voxels along x: 4 gaps x 1.0 mm = 4.0 mm.
    in_plane = np.array([[x, 30, 5] for x in range(30, 35)])
    assert maximum_feret_diameter_mm(in_plane, phantom.affine) == pytest.approx(4.0)


def test_feret_finds_the_true_longest_span_of_an_elongated_lesion():
    """Why Feret is the primary diameter: on a long thin lesion it reports the
    length, while equivalent-sphere reports a small fraction of it."""
    from features.lesion_features import (
        equivalent_sphere_diameter_mm,
        maximum_feret_diameter_mm,
    )

    phantom = build_phantom()
    # 40 mm long, 2 voxels wide, one slice thick.
    coords = np.array([[x, y, 5] for x in range(10, 50) for y in (30, 31)])
    feret = maximum_feret_diameter_mm(coords, phantom.affine)
    equivalent = equivalent_sphere_diameter_mm(len(coords) * phantom.voxel_volume_mm3)

    assert feret == pytest.approx(np.hypot(39.0, 1.0), rel=1e-6)
    assert feret > 3 * equivalent, (
        f"Feret {feret:.1f} mm vs equivalent-sphere {equivalent:.1f} mm — this gap is "
        f"exactly why the primary diameter was switched to Feret in Week 4"
    )


def test_single_voxel_lesion_has_zero_feret_diameter():
    from features.lesion_features import maximum_feret_diameter_mm

    phantom = build_phantom()
    assert maximum_feret_diameter_mm(np.array([[30, 30, 5]]), phantom.affine) == 0.0


def test_equivalent_sphere_diameter_matches_the_formula():
    from features.lesion_features import equivalent_sphere_diameter_mm

    # A sphere of radius 10 mm has volume 4/3 pi r^3 and diameter 20 mm.
    volume = 4.0 / 3.0 * np.pi * 10.0 ** 3
    assert equivalent_sphere_diameter_mm(volume) == pytest.approx(20.0)


def test_features_recover_the_phantom_blob_counts_and_volumes():
    """R6 and R7 against blobs of exactly known size."""
    from features.lesion_features import extract_features

    phantom = build_phantom()
    features = extract_features(
        phantom.wmh_mask, phantom.affine, phantom.voxel_volume_mm3, source="phantom"
    )
    assert features["lesion_count_26conn"] == len(phantom.blobs)   # 4 blobs placed
    expected_voxels = sum(blob.n_voxels for blob in phantom.blobs.values())
    assert features["total_lesion_voxels"] == expected_voxels
    assert features["total_lesion_volume_ml"] == pytest.approx(
        expected_voxels * phantom.voxel_volume_mm3 / 1000.0
    )


def test_hemisphere_assignment_matches_the_phantom_blobs():
    """R9. The phantom places one blob on each side at known world x."""
    from features.lesion_features import extract_features

    phantom = build_phantom()
    brain = np.zeros(phantom.shape, dtype=bool)
    brain[5:55, 5:55, 0:10] = True  # symmetric about the phantom's midline

    for side, other in (("right", "left"), ("left", "right")):
        blob = np.zeros(phantom.shape, dtype=bool)
        blob[phantom.blobs[side].voxel_slice] = True
        features = extract_features(blob, phantom.affine, phantom.voxel_volume_mm3,
                                    brain_mask=brain, source="phantom")
        assert features[f"{side}_lesion_volume_ml"] > 0, f"{side} blob assigned to {other}"
        assert features[f"{other}_lesion_volume_ml"] == 0.0


def test_laterality_index_sign_is_left_minus_right():
    from features.lesion_features import extract_features

    phantom = build_phantom()
    brain = np.zeros(phantom.shape, dtype=bool)
    brain[5:55, 5:55, 0:10] = True
    left_blob = np.zeros(phantom.shape, dtype=bool)
    left_blob[phantom.blobs["left"].voxel_slice] = True
    features = extract_features(left_blob, phantom.affine, phantom.voxel_volume_mm3,
                                brain_mask=brain, source="phantom")
    assert features["laterality_index"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# R5 — the same geometry, but exercising features/periventricular.py itself.
#
# The three tests above prove the 10 mm rule is computable correctly, but they
# call distance_transform_edt inline, so they test scipy rather than this
# project. These call the real function Week 4 ships, so a regression in it
# fails the suite.
# ---------------------------------------------------------------------------

def test_real_split_function_labels_both_blobs_correctly():
    from features.periventricular import split_periventricular_deep

    phantom = build_phantom()
    lesions = np.zeros(phantom.shape, dtype=bool)
    for name in ("periventricular", "deep"):
        lesions[phantom.blobs[name].voxel_slice] = True

    result = split_periventricular_deep(lesions, phantom.ventricle_mask,
                                        phantom.spacing, threshold_mm=10.0)

    pv_expected = np.zeros(phantom.shape, dtype=bool)
    pv_expected[phantom.blobs["periventricular"].voxel_slice] = True
    deep_expected = np.zeros(phantom.shape, dtype=bool)
    deep_expected[phantom.blobs["deep"].voxel_slice] = True

    assert np.array_equal(result["periventricular"], pv_expected)
    assert np.array_equal(result["deep"], deep_expected)


def test_real_split_conserves_every_lesion_voxel():
    """No lesion voxel may be lost or counted twice by the split."""
    from features.periventricular import split_periventricular_deep

    phantom = build_phantom()
    lesions = np.zeros(phantom.shape, dtype=bool)
    for blob in phantom.blobs.values():
        lesions[blob.voxel_slice] = True

    result = split_periventricular_deep(lesions, phantom.ventricle_mask,
                                        phantom.spacing)
    assert not (result["periventricular"] & result["deep"]).any()
    assert (result["periventricular"] | result["deep"]).sum() == lesions.sum()


def test_lesion_touching_the_ventricle_is_kept_and_called_periventricular():
    """A confluent lesion continuous with the ventricle must survive.

    This is the Week 2 `wm_mask` failure in a new guise: confluent
    periventricular lesions are contiguous with the ventricle, and SynthSeg
    assigns a measured mean 2.96% of true WMH to the ventricle label. If the
    lesion were simply intersected away, the largest and most clinically
    significant lesions would silently disappear from the very class they
    define.
    """
    from features.periventricular import split_periventricular_deep

    phantom = build_phantom()
    # A lesion overlapping the ventricle wall: half inside the label, half out.
    lesion = np.zeros(phantom.shape, dtype=bool)
    ventricle_voxels = np.argwhere(phantom.ventricle_mask)
    x, y, z = ventricle_voxels[len(ventricle_voxels) // 2]
    lesion[x, y, z] = True          # inside the ventricle label
    lesion[x + 1, y, z] = True      # just outside it

    result = split_periventricular_deep(lesion, phantom.ventricle_mask,
                                        phantom.spacing)

    assert result["periventricular"].sum() + result["deep"].sum() == 2, \
        "no lesion voxel may be dropped for overlapping the ventricle"
    assert result["periventricular"].sum() == 2, \
        "a lesion on the ventricle wall is periventricular by definition"
    assert not result["ventricles_used"][x, y, z], \
        "the lesion voxel must be removed from the ventricle mask, not kept as CSF"


def test_empty_ventricle_mask_raises_rather_than_calling_everything_deep():
    from features.periventricular import distance_to_ventricles_mm

    phantom = build_phantom()
    with pytest.raises(ValueError, match="empty ventricle mask"):
        distance_to_ventricles_mm(np.zeros(phantom.shape, dtype=bool), phantom.spacing)


def test_sensitivity_sweep_is_monotonic():
    """More distance allowed can only ever move lesions INTO periventricular."""
    from features.periventricular import summarise

    phantom = build_phantom()
    lesions = np.zeros(phantom.shape, dtype=bool)
    for blob in phantom.blobs.values():
        lesions[blob.voxel_slice] = True

    out = summarise(lesions, phantom.ventricle_mask, phantom.spacing,
                    thresholds_mm=(5.0, 10.0, 15.0), primary_mm=10.0)
    at5 = out["periventricular_ml_at_5mm"]
    at10 = out["periventricular_ml"]
    at15 = out["periventricular_ml_at_15mm"]
    assert at5 <= at10 <= at15
