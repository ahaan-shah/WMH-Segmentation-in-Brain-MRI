"""Synthetic phantom generator (CLAUDE.md Section 5.3).

Builds a synthetic NIfTI-shaped volume with deliberately anisotropic spacing,
a "ventricle" block, and WMH-like blobs placed at *exactly known* distances
from it and of *exactly known* voxel counts. Real MRI cannot test any of this,
because there is no independently known ground truth to compare against — the
phantom is what lets test_phantom.py catch the four highest-risk silent bugs
(anisotropic-spacing errors, 6- vs 26-connectivity mistakes, left/right
flips, and distance-unit errors) before they can survive undetected into the
final report.

All blobs are placed so that their nearest ventricle voxel is aligned in y/z
(only the x-offset differs), which is what makes their distance-to-ventricle
exact and unambiguous rather than an off-axis Euclidean approximation.
"""

from dataclasses import dataclass, field

import numpy as np

SHAPE = (60, 60, 10)
SPACING = (1.0, 1.0, 3.0)  # mm; anisotropic like real FLAIR (thick slices)
VOXEL_VOLUME_MM3 = float(np.prod(SPACING))

# World-coordinate midline for the hemisphere test: affine maps voxel (i, j, k)
# to world (i - 30, j - 30, 3k - 15), so voxel x = 30 is world x = 0.
AFFINE = np.array(
    [
        [1.0, 0.0, 0.0, -30.0],
        [0.0, 1.0, 0.0, -30.0],
        [0.0, 0.0, 3.0, -15.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Ventricle block: x in [25,35), y in [25,35), z in [3,7) -> 10*10*4 = 400 voxels
VENTRICLE_SLICE = (slice(25, 35), slice(25, 35), slice(3, 7))
VENTRICLE_LAST_X = 34  # last ventricle voxel index along x

# Periventricular blob: 1 voxel thick in x at index 42 -> distance to x=34 is
# exactly 8 voxels * 1.0 mm/voxel = 8.0 mm. y/z ranges are subsets of the
# ventricle's own y/z ranges so the nearest ventricle voxel is directly
# aligned in x (no diagonal shortcut).
BLOB_PV_SLICE = (slice(42, 43), slice(27, 32), slice(4, 6))  # 1*5*2 = 10 voxels
BLOB_PV_DISTANCE_MM = 8.0

# Deep blob: 1 voxel thick in x at index 46 -> distance to x=34 is exactly
# 12 voxels * 1.0 mm/voxel = 12.0 mm.
BLOB_DEEP_SLICE = (slice(46, 47), slice(27, 32), slice(4, 6))  # 1*5*2 = 10 voxels
BLOB_DEEP_DISTANCE_MM = 12.0

# Hemisphere blobs, well clear of the ventricle/pv/deep region.
BLOB_RIGHT_SLICE = (slice(45, 50), slice(5, 10), slice(0, 2))  # 5*5*2 = 50 voxels
BLOB_LEFT_SLICE = (slice(10, 15), slice(5, 10), slice(0, 2))  # 5*5*2 = 50 voxels

# Two single-voxel lesions touching only at a corner (differ by 1 voxel in
# both x and y, same z): face-adjacent (6-connectivity) treats them as 2
# separate components; full 26-connectivity merges them into 1.
CONN_VOXEL_A = (5, 45, 8)
CONN_VOXEL_B = (6, 46, 8)


@dataclass
class Blob:
    name: str
    voxel_slice: tuple
    n_voxels: int
    expected_distance_mm: float | None = None
    expected_zone: str | None = None
    expected_hemisphere: str | None = None


@dataclass
class Phantom:
    shape: tuple
    spacing: tuple
    voxel_volume_mm3: float
    affine: np.ndarray
    ventricle_mask: np.ndarray
    wmh_mask: np.ndarray
    connectivity_pair_mask: np.ndarray
    blobs: dict = field(default_factory=dict)


def _blob_mask(shape, voxel_slice):
    mask = np.zeros(shape, dtype=bool)
    mask[voxel_slice] = True
    return mask


def build_phantom() -> Phantom:
    """Construct the phantom described in the module docstring."""
    ventricle_mask = _blob_mask(SHAPE, VENTRICLE_SLICE)

    blob_pv = _blob_mask(SHAPE, BLOB_PV_SLICE)
    blob_deep = _blob_mask(SHAPE, BLOB_DEEP_SLICE)
    blob_right = _blob_mask(SHAPE, BLOB_RIGHT_SLICE)
    blob_left = _blob_mask(SHAPE, BLOB_LEFT_SLICE)

    wmh_mask = blob_pv | blob_deep | blob_right | blob_left

    connectivity_pair_mask = np.zeros(SHAPE, dtype=bool)
    connectivity_pair_mask[CONN_VOXEL_A] = True
    connectivity_pair_mask[CONN_VOXEL_B] = True

    blobs = {
        "periventricular": Blob(
            "periventricular",
            BLOB_PV_SLICE,
            n_voxels=int(blob_pv.sum()),
            expected_distance_mm=BLOB_PV_DISTANCE_MM,
            expected_zone="periventricular",
        ),
        "deep": Blob(
            "deep",
            BLOB_DEEP_SLICE,
            n_voxels=int(blob_deep.sum()),
            expected_distance_mm=BLOB_DEEP_DISTANCE_MM,
            expected_zone="deep",
        ),
        "right": Blob(
            "right",
            BLOB_RIGHT_SLICE,
            n_voxels=int(blob_right.sum()),
            expected_hemisphere="right",
        ),
        "left": Blob(
            "left",
            BLOB_LEFT_SLICE,
            n_voxels=int(blob_left.sum()),
            expected_hemisphere="left",
        ),
    }

    return Phantom(
        shape=SHAPE,
        spacing=SPACING,
        voxel_volume_mm3=VOXEL_VOLUME_MM3,
        affine=AFFINE,
        ventricle_mask=ventricle_mask,
        wmh_mask=wmh_mask,
        connectivity_pair_mask=connectivity_pair_mask,
        blobs=blobs,
    )


# ---------------------------------------------------------------------------
# Week 2 additions: orientation and geometry phantoms
# ---------------------------------------------------------------------------
# The blobs above test W4 geometry (distance, connectivity, hemisphere). The
# helpers below test the W2 I/O boundary instead — specifically the trap that
# every array in this project is computed in canonical RAS while every file on
# disk, ours included, is stored LPS, and the vendored official scorer reads
# raw without canonicalising.
#
# A RAS phantom cannot test that: as_closest_canonical() would be a no-op and
# restore_orientation() would be the identity, so the tests would pass without
# exercising anything. LPS_AFFINE below therefore mirrors the real dataset,
# where all 170 subjects report LPS.

# LPS: +x -> Left, +y -> Posterior, +z -> Superior, with the same anisotropic
# 1.0 x 1.0 x 3.0 mm spacing as the RAS phantom above and as real FLAIR.
LPS_AFFINE = np.array(
    [
        [-1.0, 0.0, 0.0, 30.0],
        [0.0, -1.0, 0.0, 30.0],
        [0.0, 0.0, 3.0, -15.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

# Deliberately asymmetric in all three axes: a marker at the centre, or on any
# plane of symmetry, would survive an axis flip undetected and the round-trip
# test would prove nothing.
MARKER_VOXEL = (10, 20, 2)
MARKER_VALUE = 1000.0

# Corner displacement used to check the geometry tolerance from both sides:
# well under the 0.001 mm tolerance, and well over it.
SUBTOLERANCE_SHIFT_MM = 1e-5
SUPRATOLERANCE_SHIFT_MM = 1.0


def build_marker_volume(shape=SHAPE, marker_voxel=MARKER_VOXEL):
    """Volume that is zero everywhere except a single marker voxel."""
    array = np.zeros(shape, dtype=np.float32)
    array[marker_voxel] = MARKER_VALUE
    return array


def build_lps_image():
    """A marker volume wrapped in an LPS-oriented NIfTI image, like the real data.

    Returned as a nibabel image (not a Phantom) because the orientation tests
    operate on images and affines rather than on masks.
    """
    import nibabel as nib

    return nib.Nifti1Image(build_marker_volume(), LPS_AFFINE)
