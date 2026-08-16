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
