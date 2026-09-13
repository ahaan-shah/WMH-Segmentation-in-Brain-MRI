"""Tests for the Week 2 pre-processing primitives (Stages 1 and 2).

These test the properties that real MRI cannot: on a real scan there is no
independently known head boundary, no known bias field, and no known answer to
compare against (CLAUDE.md Section 5.3). The synthetic volumes here have all
three by construction.

Two failure modes get most of the attention, because both are silent:

- **Axis order across the SimpleITK boundary.** nibabel is (x, y, z),
  SimpleITK is (z, y, x). Getting it wrong attaches 3.00 mm slice spacing to an
  in-plane axis, and every physical measurement downstream is wrong by up to
  5.4x while the image still looks fine. On Utrecht and Singapore the in-plane
  dimensions are square, so a transposed volume can even keep a plausible shape.

- **Through-plane morphology.** Slices are 3.00 mm against 0.56-1.30 mm
  in-plane, so a 3x3x3 structuring element reaches three times further through
  the brain than across it.
"""

import numpy as np
import pytest
import SimpleITK as sitk
from scipy import ndimage as ndi

from preprocessing.bias_field import (
    correct_bias_field,
    intensity_uniformity_cv,
    lesion_contrast_to_noise,
    normal_appearing_tissue_mask,
)
from preprocessing.head_mask import (
    compute_head_mask,
    in_plane_structure,
    largest_connected_component,
)
from preprocessing.sitk_interop import from_sitk, to_sitk
from preprocessing.skull_strip import brain_mask_quality, dilate_in_plane

ANISOTROPIC_SPACING = (1.0, 1.0, 3.0)
# Deliberately non-cubic and all-different, so any axis permutation changes the
# shape and cannot pass silently.
ASYMMETRIC_SHAPE = (30, 24, 10)


# ---------------------------------------------------------------------------
# SimpleITK interop — the silent axis-order trap
# ---------------------------------------------------------------------------


def test_to_sitk_preserves_shape_as_size():
    array = np.random.default_rng(0).random(ASYMMETRIC_SHAPE).astype(np.float32)
    image = to_sitk(array, ANISOTROPIC_SPACING)
    assert tuple(image.GetSize()) == ASYMMETRIC_SHAPE


def test_sitk_round_trip_is_exact():
    array = np.random.default_rng(1).random(ASYMMETRIC_SHAPE).astype(np.float32)
    np.testing.assert_array_equal(from_sitk(to_sitk(array, ANISOTROPIC_SPACING)), array)


def test_spacing_lands_on_the_correct_axis():
    """The whole point: 3.00 mm must attach to z, not to x."""
    array = np.zeros(ASYMMETRIC_SHAPE, dtype=np.float32)
    image = to_sitk(array, ANISOTROPIC_SPACING)
    assert tuple(image.GetSpacing()) == ANISOTROPIC_SPACING


def test_a_transposed_volume_would_be_caught():
    """If conversion silently transposed, distances along z would be measured
    with in-plane spacing. Asserting size == shape is what rules that out."""
    array = np.zeros(ASYMMETRIC_SHAPE, dtype=np.float32)
    image = to_sitk(array, ANISOTROPIC_SPACING)
    assert tuple(image.GetSize()) != tuple(reversed(ASYMMETRIC_SHAPE))


def test_marker_position_survives_the_sitk_round_trip():
    array = np.zeros(ASYMMETRIC_SHAPE, dtype=np.float32)
    array[3, 17, 8] = 1.0
    back = from_sitk(to_sitk(array, ANISOTROPIC_SPACING))
    assert tuple(np.argwhere(back == 1.0)[0]) == (3, 17, 8)


def test_to_sitk_rejects_non_3d():
    with pytest.raises(ValueError, match="3D"):
        to_sitk(np.zeros((5, 5)), ANISOTROPIC_SPACING)


# ---------------------------------------------------------------------------
# Morphology — in-plane only
# ---------------------------------------------------------------------------


def test_structuring_element_is_flat_in_z():
    """A (2r+1, 2r+1, 1) element cannot reach across 3 mm slices."""
    for radius in (1, 2, 5):
        element = in_plane_structure(radius)
        assert element.shape == (2 * radius + 1, 2 * radius + 1, 1)
        assert element.shape[2] == 1


def test_structuring_element_rejects_zero_radius():
    with pytest.raises(ValueError, match="radius_voxels"):
        in_plane_structure(0)


def test_largest_component_discards_ghosting():
    """Singapore volumes carry aliasing streaks outside the head; without
    component selection the mask leaks into them."""
    mask = np.zeros((20, 20, 4), dtype=bool)
    mask[2:12, 2:12, 1:3] = True  # the "head"
    mask[17, 17, 0] = True  # a speckle
    mask[15:17, 1, 3] = True  # a "streak"
    kept, n_components = largest_connected_component(mask)
    assert n_components == 3
    assert kept.sum() == 10 * 10 * 2
    assert not kept[17, 17, 0]


def test_largest_component_rejects_empty_mask():
    with pytest.raises(ValueError, match="empty mask"):
        largest_connected_component(np.zeros((4, 4, 2), dtype=bool))


# ---------------------------------------------------------------------------
# Head mask
# ---------------------------------------------------------------------------


def _synthetic_head(shape=(40, 40, 8), spacing=ANISOTROPIC_SPACING, bright_rim=False):
    """An ellipsoidal 'head' with a dim interior, optional bright rim, and noise."""
    rng = np.random.default_rng(7)
    x, y, z = np.indices(shape)
    cx, cy, cz = (shape[0] - 1) / 2, (shape[1] - 1) / 2, (shape[2] - 1) / 2
    radial = ((x - cx) / 15.0) ** 2 + ((y - cy) / 15.0) ** 2 + ((z - cz) / 3.5) ** 2
    head = radial <= 1.0
    volume = rng.normal(3.0, 1.0, shape)  # air noise floor
    volume[head] = rng.normal(300.0, 20.0, head.sum())
    if bright_rim:
        rim = head & ~(radial <= 0.75)
        volume[rim] = rng.normal(2500.0, 100.0, rim.sum())
    return np.clip(volume, 0, None), head


def test_head_mask_recovers_a_synthetic_head():
    volume, truth = _synthetic_head()
    mask, diagnostics = compute_head_mask(
        volume, ANISOTROPIC_SPACING, closing_radius_mm=3.0, opening_radius_mm=1.0
    )
    overlap = 2 * (mask & truth).sum() / (mask.sum() + truth.sum())
    assert overlap > 0.95, f"Dice against the known head was {overlap:.3f}"
    assert diagnostics["threshold_method"] == "multiotsu_3class_lowest"


def test_head_mask_survives_a_bright_scalp_rim():
    """The Utrecht subject 11 failure, reproduced synthetically. Two-class Otsu
    thresholds between the dim interior and the bright rim and returns only the
    rim; three-class multi-Otsu must keep the whole head."""
    volume, truth = _synthetic_head(bright_rim=True)
    mask, _ = compute_head_mask(
        volume, ANISOTROPIC_SPACING, closing_radius_mm=3.0, opening_radius_mm=1.0
    )
    interior = truth & (volume < 1000)
    retained = mask[interior].mean()
    assert retained > 0.99, (
        f"only {retained:.3f} of the dim brain interior survived — the threshold "
        f"landed between brain and scalp, which is exactly the two-class Otsu bug"
    )


def test_head_mask_fills_interior_cavities():
    volume, _ = _synthetic_head()
    volume[18:22, 18:22, 3:5] = 0.0  # a "ventricle" of pure background
    mask, _ = compute_head_mask(
        volume, ANISOTROPIC_SPACING, closing_radius_mm=3.0, opening_radius_mm=1.0
    )
    assert mask[18:22, 18:22, 3:5].all(), "enclosed cavities must be filled for N4"


def test_head_mask_rejects_nan_and_degenerate_volumes():
    volume, _ = _synthetic_head()
    volume[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        compute_head_mask(volume, ANISOTROPIC_SPACING, closing_radius_mm=3.0, opening_radius_mm=1.0)

    with pytest.raises(ValueError, match="Degenerate"):
        compute_head_mask(
            np.zeros((10, 10, 4)), ANISOTROPIC_SPACING, closing_radius_mm=3.0, opening_radius_mm=1.0
        )


def test_head_mask_radius_scales_with_in_plane_spacing():
    """Radii are given in mm, so a coarser grid must use fewer voxels for the
    same physical size — otherwise the operation means something different at
    Amsterdam/Philips (0.56 mm) than at Utrecht (0.96 mm)."""
    volume, _ = _synthetic_head()
    _, fine = compute_head_mask(
        volume, (0.5, 0.5, 3.0), closing_radius_mm=3.0, opening_radius_mm=1.0
    )
    _, coarse = compute_head_mask(
        volume, (1.5, 1.5, 3.0), closing_radius_mm=3.0, opening_radius_mm=1.0
    )
    assert fine["closing_radius_voxels"] > coarse["closing_radius_voxels"]


# ---------------------------------------------------------------------------
# N4 bias correction
# ---------------------------------------------------------------------------


def test_n4_recovers_a_known_synthetic_bias_field():
    """The only way to test a bias corrector: impose a field we chose ourselves
    and check the correction flattens it."""
    volume, head = _synthetic_head()
    x = np.linspace(0.7, 1.4, volume.shape[0])[:, None, None]
    imposed = np.broadcast_to(x, volume.shape).copy()
    corrupted = volume * imposed

    before = intensity_uniformity_cv(corrupted, head)
    corrected, estimated, diagnostics = correct_bias_field(
        corrupted, ANISOTROPIC_SPACING, head, fitting_levels=3
    )
    after = intensity_uniformity_cv(corrected, head)

    assert after < before, f"uniformity worsened: CV {before:.4f} -> {after:.4f}"
    assert diagnostics["bias_span_percent"] > 0
    assert np.isfinite(estimated).all()
    assert estimated.shape == volume.shape


def test_n4_output_keeps_shape_and_is_finite():
    volume, head = _synthetic_head()
    corrected, bias, _ = correct_bias_field(
        volume, ANISOTROPIC_SPACING, head, fitting_levels=2
    )
    assert corrected.shape == volume.shape == bias.shape
    assert np.isfinite(corrected).all()


def test_n4_rejects_mismatched_mask_and_empty_mask():
    volume, head = _synthetic_head()
    with pytest.raises(AssertionError, match="head mask shape"):
        correct_bias_field(volume, ANISOTROPIC_SPACING, head[:-1], fitting_levels=2)
    with pytest.raises(ValueError, match="Head mask is empty"):
        correct_bias_field(
            volume, ANISOTROPIC_SPACING, np.zeros_like(head), fitting_levels=2
        )


def test_more_fitting_levels_give_a_more_flexible_field():
    """The mechanism behind the whole Stage 2 sweep: each level doubles the
    control point mesh, so higher level counts can follow finer structure — and
    at some point that structure is the lesion rather than the scanner."""
    volume, head = _synthetic_head()
    _, _, coarse = correct_bias_field(volume, ANISOTROPIC_SPACING, head, fitting_levels=2)
    _, _, fine = correct_bias_field(volume, ANISOTROPIC_SPACING, head, fitting_levels=4)
    assert fine["bias_span_percent"] > coarse["bias_span_percent"]


# ---------------------------------------------------------------------------
# CNR / uniformity metrics used to score the sweep
# ---------------------------------------------------------------------------


def test_cnr_is_positive_when_lesions_are_brighter():
    rng = np.random.default_rng(3)
    volume = rng.normal(100.0, 5.0, (20, 20, 4))
    lesion = np.zeros((20, 20, 4), dtype=bool)
    lesion[5:9, 5:9, 1:3] = True
    volume[lesion] = 200.0
    tissue = np.ones((20, 20, 4), dtype=bool) & ~lesion
    assert lesion_contrast_to_noise(volume, lesion, tissue) > 0


def test_cnr_is_scale_invariant():
    """Multiplying an image by a constant must not change contrast-to-noise,
    or the sweep would reward N4 settings purely for rescaling."""
    rng = np.random.default_rng(4)
    volume = rng.normal(100.0, 5.0, (20, 20, 4))
    lesion = np.zeros((20, 20, 4), dtype=bool)
    lesion[5:9, 5:9, 1:3] = True
    volume[lesion] = 200.0
    tissue = np.ones((20, 20, 4), dtype=bool) & ~lesion
    assert lesion_contrast_to_noise(volume, lesion, tissue) == pytest.approx(
        lesion_contrast_to_noise(volume * 7.5, lesion, tissue), rel=1e-9
    )


def test_cnr_and_cv_reject_empty_regions():
    volume = np.ones((10, 10, 4))
    empty = np.zeros((10, 10, 4), dtype=bool)
    full = np.ones((10, 10, 4), dtype=bool)
    with pytest.raises(ValueError, match="Empty lesion mask"):
        lesion_contrast_to_noise(volume, empty, full)
    with pytest.raises(ValueError, match="Empty reference tissue mask"):
        lesion_contrast_to_noise(volume, full, empty)
    with pytest.raises(ValueError, match="Empty ROI"):
        intensity_uniformity_cv(volume, empty)


def test_uniformity_cv_falls_when_a_gradient_is_removed():
    rng = np.random.default_rng(5)
    flat = rng.normal(100.0, 2.0, (20, 20, 4))
    gradient = np.linspace(0.6, 1.5, 20)[:, None, None]
    roi = np.ones((20, 20, 4), dtype=bool)
    assert intensity_uniformity_cv(flat * gradient, roi) > intensity_uniformity_cv(flat, roi)


def test_normal_appearing_mask_excludes_lesions_and_air():
    volume, head = _synthetic_head()
    lesion = np.zeros_like(head)
    lesion[18:22, 18:22, 3:5] = True
    volume[lesion] = 900.0
    nawm = normal_appearing_tissue_mask(volume, head, lesion)
    assert not (nawm & lesion).any(), "reference lesions must be excluded"
    assert not (nawm & ~head).any(), "air must be excluded"
    assert nawm.sum() > 0


# ---------------------------------------------------------------------------
# Stage 3 — skull stripping acceptance metrics
# ---------------------------------------------------------------------------
# The two criteria are deliberately asymmetric detectors: WMH retention sees
# only over-stripping, shallow_fraction sees only under-stripping. Each test
# below builds a mask that fails ONE of them, and asserts the other stays
# clean — otherwise a single metric could be mistaken for sufficient.


def _head_and_brain(shape=(40, 40, 16), spacing=(1.0, 1.0, 3.0), shell_mm=9.0):
    """A 'head' with a skull/scalp shell of known thickness around a 'brain'.

    The brain is defined by DEPTH below the head surface rather than by a scaled
    ellipsoid, because with 3 mm slices a geometrically-scaled inner ellipsoid
    leaves a much thinner shell along z than in-plane — the brain would sit
    legitimately within 5 mm of the surface at the vertex, and the phantom would
    fail its own test for reasons that have nothing to do with the code.
    Defining it in millimetres makes the shell thickness exactly what it claims.
    """
    x, y, z = np.indices(shape)
    cx, cy, cz = (shape[0] - 1) / 2, (shape[1] - 1) / 2, (shape[2] - 1) / 2
    head = ((x - cx) / 17.0) ** 2 + ((y - cy) / 17.0) ** 2 + ((z - cz) / 7.0) ** 2 <= 1.0
    depth_mm = ndi.distance_transform_edt(head, sampling=spacing)
    brain = depth_mm > shell_mm
    return head, brain


def test_shallow_fraction_is_near_zero_for_a_correct_brain_mask():
    head, brain = _head_and_brain()
    quality = brain_mask_quality(brain, head, (1.0, 1.0, 3.0))
    assert quality["shallow_fraction"] < 0.02, (
        f"a brain mask well inside the skull should have almost nothing within "
        f"5 mm of the head surface, got {quality['shallow_fraction']:.4f}"
    )


def test_shallow_fraction_catches_a_mask_that_kept_the_scalp():
    """Under-stripping. WMH retention is blind to this, which is the whole
    reason a second criterion exists."""
    head, brain = _head_and_brain()
    lesion = np.zeros_like(head)
    lesion[19:21, 19:21, 7:9] = True  # deep, so it survives either way

    good = brain_mask_quality(brain, head, (1.0, 1.0, 3.0), lesion_mask=lesion)
    leaky = brain_mask_quality(head, head, (1.0, 1.0, 3.0), lesion_mask=lesion)

    assert leaky["shallow_fraction"] > good["shallow_fraction"] * 5
    # ...and criterion 1 cannot tell the difference:
    assert good["wmh_retained_fraction"] == leaky["wmh_retained_fraction"] == 1.0


def test_wmh_retention_catches_over_stripping():
    """Over-stripping. shallow_fraction is blind to this — the mirror case."""
    head, brain = _head_and_brain()
    lesion = np.zeros_like(head)
    lesion = brain & ~ndi.binary_erosion(brain, structure=np.ones((5, 5, 1)))  # brain-edge lesion

    eroded = ndi.binary_erosion(brain, structure=np.ones((5, 5, 1)))
    full = brain_mask_quality(brain, head, (1.0, 1.0, 3.0), lesion_mask=lesion)
    cut = brain_mask_quality(eroded, head, (1.0, 1.0, 3.0), lesion_mask=lesion)

    assert full["wmh_retained_fraction"] > cut["wmh_retained_fraction"]
    assert cut["wmh_lost_voxels"] > 0
    # ...and the over-stripped mask looks BETTER on criterion 2, which is
    # exactly why criterion 2 alone would be dangerous.
    assert cut["shallow_fraction"] <= full["shallow_fraction"]


def test_depth_is_measured_in_millimetres_not_voxels():
    """With 3 mm slices, a 5-voxel threshold would mean 15 mm along z. The
    shallow test must change when only the spacing changes."""
    head, brain = _head_and_brain()
    isotropic = brain_mask_quality(brain, head, (1.0, 1.0, 1.0))["shallow_fraction"]
    anisotropic = brain_mask_quality(brain, head, (1.0, 1.0, 3.0))["shallow_fraction"]
    assert isotropic != anisotropic, "spacing must reach the distance transform"


def test_quality_rejects_empty_and_mismatched_masks():
    head, brain = _head_and_brain()
    with pytest.raises(ValueError, match="Brain mask is empty"):
        brain_mask_quality(np.zeros_like(head), head, (1.0, 1.0, 3.0))
    with pytest.raises(AssertionError, match="head mask shape"):
        brain_mask_quality(brain, head[:-1], (1.0, 1.0, 3.0))
    with pytest.raises(ValueError, match="lesion mask is empty"):
        brain_mask_quality(brain, head, (1.0, 1.0, 3.0), lesion_mask=np.zeros_like(head))


def test_t1_transfer_dilation_is_in_plane_only():
    """The T1->FLAIR transfer dilates to absorb boundary differences. Doing it
    through-plane would grow the mask 3 mm per step instead of ~1 mm."""
    mask = np.zeros((20, 20, 6), dtype=bool)
    mask[10, 10, 3] = True
    grown = dilate_in_plane(mask, 1)
    assert grown[:, :, 2].sum() == 0 and grown[:, :, 4].sum() == 0, "must not grow across slices"
    assert grown[:, :, 3].sum() == 9  # a 3x3 in-plane square


def test_dilation_of_zero_is_a_no_op():
    mask = np.zeros((10, 10, 4), dtype=bool)
    mask[5, 5, 2] = True
    np.testing.assert_array_equal(dilate_in_plane(mask, 0), mask)


# ---------------------------------------------------------------------------
# Artefact routing — deliverables vs scaffolding
# ---------------------------------------------------------------------------


def test_only_deliverables_live_in_processed():
    """`data/processed/` must contain exactly the files a later week opens —
    not everything a stage happened to produce.

    This list is pinned deliberately. Adding an artefact here should be a
    conscious decision about whether it is a deliverable or scaffolding, and
    this test failing is the prompt to make that decision rather than an
    inconvenience to be loosened. Week 3 added `pred_wmh` — the predicted lesion
    mask, which Week 4 measures features from and Week 7 scores."""
    from metadata.config import DATA_INTERIM, DATA_PROCESSED
    from metadata.derived import (
        ARTEFACT_ROOTS, BIAS_FIELD, BRAIN_MASK, FLAIR_N4, FLAIR_NORM, HEAD_MASK,
        PRED_WMH, WM_MASK,
    )

    deliverables = {name for name, root in ARTEFACT_ROOTS.items() if root == DATA_PROCESSED}
    assert deliverables == {BRAIN_MASK, WM_MASK, FLAIR_NORM, PRED_WMH}

    for scaffolding in (HEAD_MASK, FLAIR_N4, BIAS_FIELD):
        assert ARTEFACT_ROOTS[scaffolding] == DATA_INTERIM


def test_unknown_artefact_names_are_rejected():
    """A new artefact must make a deliberate interim/processed decision rather
    than silently defaulting into one of them."""
    from metadata.derived import artefact_root

    with pytest.raises(KeyError, match="ARTEFACT_ROOTS"):
        artefact_root("some_new_thing")


def test_artefact_paths_are_rooted_correctly():
    from metadata.derived import BRAIN_MASK, FLAIR_N4, derived_path

    assert derived_path("S", BRAIN_MASK).parent.parent.name == "processed"
    assert derived_path("S", FLAIR_N4).parent.parent.name == "interim"


def test_bright_tissue_retention_flags_a_mask_that_kept_the_scalp():
    """The corrected under-stripping detector: scalp fat is the brightest tissue
    in a FLAIR head, so a mask that kept the scalp keeps the bright voxels."""
    from preprocessing.skull_strip import bright_tissue_retention

    head, brain = _head_and_brain()
    image = np.zeros(head.shape, dtype=float)
    image[head] = 100.0
    rim = head & ~ndi.binary_dilation(brain, structure=np.ones((7, 7, 3)))
    image[rim] = 2000.0  # bright scalp

    correct = bright_tissue_retention(image, brain, head)
    leaky = bright_tissue_retention(image, head, head)

    assert correct["bright_retained_fraction"] < 0.05
    assert leaky["bright_retained_fraction"] == pytest.approx(1.0)


def test_bright_tissue_retention_reports_how_much_bright_is_lesion():
    """Without this, a correct mask scores as broken wherever lesions — not
    scalp — are the brightest thing, which is the case at two of three sites."""
    from preprocessing.skull_strip import bright_tissue_retention

    head, brain = _head_and_brain()
    image = np.zeros(head.shape, dtype=float)
    image[head] = 100.0
    lesion = np.zeros_like(head)
    # Must be larger than the top 1% of head voxels, or the 99th-percentile
    # threshold lands in flat tissue and "bright" stops meaning anything.
    lesion[14:26, 14:26, 5:10] = True
    lesion &= brain
    image[lesion] = 3000.0  # lesions are the brightest thing here
    assert lesion.sum() > 0.01 * head.sum()

    result = bright_tissue_retention(image, brain, head, lesion_mask=lesion)
    assert result["bright_that_is_wmh"] > 0.5, (
        "when lesions are the brightest tissue, high retention is CORRECT and "
        "the metric must say so"
    )


def test_bright_tissue_retention_rejects_empty_masks():
    from preprocessing.skull_strip import bright_tissue_retention

    head, brain = _head_and_brain()
    image = np.ones(head.shape, dtype=float)
    with pytest.raises(ValueError, match="Empty brain or head mask"):
        bright_tissue_retention(image, np.zeros_like(head), head)


# ---------------------------------------------------------------------------
# Stage 5 — intensity normalisation
# ---------------------------------------------------------------------------


def _normalisation_phantom():
    """A 'brain' with three tissue levels and lesions brighter than white matter."""
    head, brain = _head_and_brain()
    image = np.zeros(head.shape, dtype=float)
    image[brain] = 300.0                       # white matter
    gm = brain & ~ndi.binary_erosion(brain, structure=np.ones((5, 5, 1)))
    image[gm] = 200.0                          # grey matter, dimmer
    lesion = np.zeros_like(brain)
    lesion[16:24, 16:24, 6:10] = True
    lesion &= ndi.binary_erosion(brain, structure=np.ones((7, 7, 1)))
    image[lesion] = 480.0                      # lesions: 1.6x white matter
    nawm = brain & ~gm & ~lesion
    return image, brain, nawm, lesion


def test_wm_referenced_places_normal_white_matter_at_exactly_one():
    from preprocessing.normalise import WM_REFERENCED, normalise

    image, brain, nawm, _ = _normalisation_phantom()
    out, stats = normalise(image, brain, method=WM_REFERENCED, reference_mask=nawm, minimum_reference_voxels=100)
    assert np.median(out[nawm]) == pytest.approx(1.0)
    assert stats["reference_value"] == pytest.approx(300.0)


def test_wm_referenced_is_invariant_to_scanner_gain():
    """The entire point: two scanners imaging the same brain with different
    arbitrary gains must produce the same normalised image."""
    from preprocessing.normalise import WM_REFERENCED, normalise

    image, brain, nawm, _ = _normalisation_phantom()
    a, _ = normalise(image, brain, method=WM_REFERENCED, reference_mask=nawm, minimum_reference_voxels=100)
    b, _ = normalise(image * 4.7, brain, method=WM_REFERENCED, reference_mask=nawm, minimum_reference_voxels=100)
    np.testing.assert_allclose(a, b, rtol=1e-9)


def test_lesions_land_above_one_after_wm_referencing():
    from preprocessing.normalise import WM_REFERENCED, normalise

    image, brain, nawm, lesion = _normalisation_phantom()
    out, _ = normalise(image, brain, method=WM_REFERENCED, reference_mask=nawm, minimum_reference_voxels=100)
    assert np.median(out[lesion]) == pytest.approx(480.0 / 300.0, rel=1e-6)
    assert np.median(out[lesion]) > 1.0


def test_all_methods_are_gain_invariant():
    """Gain invariance is the minimum bar; every candidate must clear it, or it
    is not a normalisation at all."""
    from preprocessing.normalise import METHODS, WM_REFERENCED, normalise

    image, brain, nawm, _ = _normalisation_phantom()
    for method in METHODS:
        a, _ = normalise(image, brain, method=method, reference_mask=nawm, minimum_reference_voxels=100)
        b, _ = normalise(image * 3.3, brain, method=method, reference_mask=nawm, minimum_reference_voxels=100)
        np.testing.assert_allclose(a, b, atol=1e-9, err_msg=f"{method} is not gain-invariant")


def test_normalise_zeroes_outside_the_brain():
    from preprocessing.normalise import WM_REFERENCED, normalise

    image, brain, nawm, _ = _normalisation_phantom()
    out, _ = normalise(image, brain, method=WM_REFERENCED, reference_mask=nawm, minimum_reference_voxels=100)
    assert not out[~brain].any()


def test_normalise_rejects_a_too_small_reference():
    """A tiny reference makes the median unstable, and dividing by it would
    scale the whole subject wrongly with nothing to signal it."""
    from preprocessing.normalise import WM_REFERENCED, normalise

    image, brain, _, _ = _normalisation_phantom()
    tiny = np.zeros_like(brain)
    tiny[20, 20, 8] = True
    with pytest.raises(ValueError, match="reference has only"):
        normalise(image, brain, method=WM_REFERENCED, reference_mask=tiny)


def test_wm_referenced_requires_a_reference_mask():
    from preprocessing.normalise import WM_REFERENCED, normalise

    image, brain, _, _ = _normalisation_phantom()
    with pytest.raises(ValueError, match="requires a reference_mask"):
        normalise(image, brain, method=WM_REFERENCED)


def test_normalise_rejects_unknown_method_and_bad_input():
    from preprocessing.normalise import normalise

    image, brain, nawm, _ = _normalisation_phantom()
    with pytest.raises(ValueError, match="Unknown method"):
        normalise(image, brain, method="nonsense", reference_mask=nawm)
    broken = image.copy()
    broken[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        normalise(broken, brain, method="z_score")


# ---------------------------------------------------------------------------
# Stages 6 and 7 — filtering branches and morphological operators
# ---------------------------------------------------------------------------


def _contrast_phantom(lesion_value=2.0):
    """Normalised-scale phantom: white matter at 1.0, lesions above it.

    Lesions are placed relative to the brain's own centroid and the placement is
    asserted, rather than hard-coded at indices that happen to fall outside the
    brain — which is exactly the mistake that made an earlier version of this
    phantom silently test something other than what it claimed.
    """
    head, brain = _head_and_brain()
    image = np.zeros(head.shape, dtype=float)
    image[brain] = 1.0

    cx, cy, cz = (int(round(c)) for c in ndi.center_of_mass(brain))
    lesion = np.zeros_like(brain)
    lesion[cx - 6, cy - 6, cz] = True                          # 1 voxel
    lesion[cx - 3:cx - 1, cy - 3:cy - 1, cz] = True            # 4 voxels
    lesion[cx + 1:cx + 6, cy + 1:cy + 6, cz - 1:cz + 1] = True  # 50 voxels

    assert (lesion & ~brain).sum() == 0, "phantom lesions must lie inside the brain"
    sizes = sorted(np.bincount(ndi.label(lesion, structure=np.ones((3, 3, 3)))[0].ravel())[1:])
    assert sizes == [1, 4, 50], f"phantom lesion sizes drifted: {sizes}"

    image[lesion] = lesion_value
    return image, brain, lesion


def test_contrast_retention_is_one_when_nothing_changes():
    from preprocessing.denoise import lesion_contrast_retention

    image, _, lesion = _contrast_phantom()
    result = lesion_contrast_retention(image, image, lesion)
    assert result["retention_mean"] == pytest.approx(1.0)
    assert result["lesions_below_half_retention"] == 0


def test_contrast_retention_is_zero_when_lesions_are_erased():
    from preprocessing.denoise import lesion_contrast_retention

    image, brain, lesion = _contrast_phantom()
    flattened = np.where(brain, 1.0, 0.0)  # every lesion smoothed into white matter
    result = lesion_contrast_retention(image, flattened, lesion)
    assert result["retention_mean"] == pytest.approx(0.0)
    assert result["fraction_below_half_retention"] == pytest.approx(1.0)


def test_small_lesions_are_reported_separately_from_large_ones():
    """The entire Stage 6 question is whether SMALL lesions survive, so the
    metric must not let large ones mask their loss."""
    from preprocessing.denoise import lesion_contrast_retention

    image, brain, lesion = _contrast_phantom()
    damaged = image.copy()
    labelled, n = ndi.label(lesion, structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labelled.ravel())[1:]
    for index in range(1, n + 1):
        if sizes[index - 1] <= 10:            # erase only the small ones
            damaged[labelled == index] = 1.0

    result = lesion_contrast_retention(damaged * 0 + damaged, image, lesion) if False else \
        lesion_contrast_retention(image, damaged, lesion)
    assert result["retention_small"] == pytest.approx(0.0)
    assert result["retention_mean"] > 0.0, "large lesions should still be intact"


def test_ssim_can_be_high_while_lesions_are_destroyed():
    """Encodes the Stage 6 finding: SSIM is not a safe proxy for lesion damage.
    Measured on real data, anisotropic diffusion scores SSIM 0.981 while small
    lesions keep only 66% of their contrast."""
    from preprocessing.denoise import lesion_contrast_retention, structural_similarity_in_mask

    image, brain, lesion = _contrast_phantom()
    damaged = image.copy()
    damaged[lesion] = 1.0  # every lesion gone

    ssim = structural_similarity_in_mask(image, damaged, brain)
    retention = lesion_contrast_retention(image, damaged, lesion)["retention_mean"]

    # The bound is 0.85 rather than 0.95 only because this phantom's lesions are
    # ~4.8% of brain volume, roughly ten times the real ~0.5%. On the actual data
    # the effect is far stronger, not weaker: anisotropic diffusion scores
    # SSIM 0.981 while small lesions keep only 66% of their contrast.
    assert ssim > 0.85, "SSIM stays high because lesions are a small fraction of the volume"
    assert retention == pytest.approx(0.0), "...while every lesion has in fact been erased"


def test_minimum_size_cost_separates_count_loss_from_volume_loss():
    """The divergence between these two columns is the reason a size filter can
    look harmless on Dice while gutting lesion F1."""
    from preprocessing.morphology import minimum_size_cost

    _, _, lesion = _contrast_phantom()
    rows = {r["minimum_voxels"]: r for r in minimum_size_cost(lesion, 3.0, (2, 5))}
    at_five = rows[5]
    # 2 of 3 lesions deleted (66.7%) but only 5 of 55 voxels (9.1%) — the same
    # order-of-magnitude divergence measured on the real data.
    assert at_five["lesion_count_fraction_removed"] == pytest.approx(2 / 3)
    assert at_five["volume_fraction_removed"] == pytest.approx(5 / 55)
    assert at_five["lesion_count_fraction_removed"] > at_five["volume_fraction_removed"] * 5


def test_remove_small_components_reports_what_it_deleted():
    from preprocessing.morphology import remove_small_components

    _, _, lesion = _contrast_phantom()
    filtered, info = remove_small_components(lesion, 5)
    assert info["components_before"] == 3
    assert info["components_removed"] == 2      # the 1-voxel and 4-voxel lesions
    assert info["voxels_removed"] == 5
    assert filtered.sum() == lesion.sum() - 5


def test_remove_small_components_is_a_no_op_below_two():
    from preprocessing.morphology import remove_small_components

    _, _, lesion = _contrast_phantom()
    filtered, _ = remove_small_components(lesion, 1)
    np.testing.assert_array_equal(filtered, lesion)


def test_six_and_twentysix_connectivity_disagree_on_lesion_counts():
    """Week 4 reports both; 26 is the project default matching evaluation.py."""
    from preprocessing.morphology import CONNECTIVITY_6, CONNECTIVITY_26

    mask = np.zeros((12, 12, 4), dtype=bool)
    mask[3, 3, 1] = True
    mask[4, 4, 1] = True  # corner-touching only
    assert ndi.label(mask, structure=CONNECTIVITY_6)[1] == 2
    assert ndi.label(mask, structure=CONNECTIVITY_26)[1] == 1


def test_morphology_operators_are_all_in_plane():
    from preprocessing.morphology import close_in_plane, erode_in_plane, open_in_plane

    spacing = (1.0, 1.0, 3.0)
    mask = np.zeros((20, 20, 5), dtype=bool)
    mask[8:13, 8:13, 2] = True  # a single-slice block
    for operator in (erode_in_plane, open_in_plane, close_in_plane):
        result = operator(mask, 1.0, spacing)
        assert not result[:, :, 1].any() and not result[:, :, 3].any(), (
            f"{operator.__name__} reached across a 3 mm slice boundary"
        )
