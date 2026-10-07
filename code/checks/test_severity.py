"""Tests for classification/severity.py — known inputs, known answers.

The boundary tests matter more than they look: with 8 subjects in the smallest
class, one subject moved across a cut-off by an off-by-one convention is 12% of
that class.
"""

import numpy as np
import pandas as pd
import pytest

from classification.severity import FEATURE_NAMES, assign_class, derive_features
from metadata.config import SEVERITY_CONFIG

CUTOFFS = SEVERITY_CONFIG["cutoffs_ml"]


def test_config_matches_the_verified_citation():
    # Joo et al. 2022, checked against the paper. If someone edits these, the
    # class meanings and every report quoting them change with it.
    assert CUTOFFS == [3.4, 9.6, 17.1]
    assert SEVERITY_CONFIG["classes"] == ["normal", "mild", "moderate", "severe"]
    assert SEVERITY_CONFIG["sensitivity_cutoffs_ml"] == [9.6, 17.1]


def test_each_band_lands_in_its_class():
    volumes = [0.0, 1.0, 5.0, 12.0, 40.0]
    assert assign_class(volumes, CUTOFFS).tolist() == [0, 0, 1, 2, 3]


def test_a_volume_on_a_cutoff_goes_to_the_higher_class():
    assert assign_class([3.4, 9.6, 17.1], CUTOFFS).tolist() == [1, 2, 3]
    # and the value just under each stays below
    below = np.nextafter(np.array(CUTOFFS), -np.inf)
    assert assign_class(below, CUTOFFS).tolist() == [0, 1, 2]


def test_three_class_scheme_merges_normal_and_mild_rather_than_relabelling():
    sens = SEVERITY_CONFIG["sensitivity_cutoffs_ml"]
    volumes = np.array([1.0, 5.0, 12.0, 40.0])        # normal, mild, moderate, severe
    four = assign_class(volumes, CUTOFFS)
    three = assign_class(volumes, sens)
    assert three.tolist() == [0, 0, 1, 2]
    # Merging must be a pure function of the 4-class label: normal and mild
    # collapse together, nothing else moves.
    assert np.array_equal(three, np.maximum(four - 1, 0))


@pytest.mark.parametrize("bad", [[], [9.6, 3.4], [3.4, 3.4, 17.1]])
def test_cutoffs_must_be_strictly_increasing(bad):
    with pytest.raises(ValueError):
        assign_class([1.0], bad)


def test_negative_or_nan_volume_raises():
    with pytest.raises(ValueError):
        assign_class([-1.0], CUTOFFS)
    with pytest.raises(ValueError):
        assign_class([np.nan], CUTOFFS)


def _row(**overrides):
    row = {"total_lesion_volume_ml": 10.0, "periventricular_fraction": 0.75,
           "laterality_index": -0.2, "lesion_count_26conn": 40,
           "small_lesion_count_le5vox": 10, "largest_lesion_volume_ml": 4.0}
    row.update(overrides)
    return row


def test_features_are_computed_exactly():
    out = derive_features(pd.DataFrame([_row()]))
    assert tuple(out.columns) == FEATURE_NAMES
    r = out.iloc[0]
    assert r.log_total_volume == pytest.approx(np.log1p(10.0))
    assert r.periventricular_fraction == pytest.approx(0.75)
    assert r.abs_laterality_index == pytest.approx(0.2)   # sign dropped
    assert r.log_lesion_count == pytest.approx(np.log1p(40))
    assert r.small_lesion_fraction == pytest.approx(10 / 40)
    assert r.largest_lesion_share == pytest.approx(4.0 / 10.0)


def test_a_single_confluent_lesion_has_share_one():
    # One lesion holding all the burden is the "large confluent areas" extreme.
    out = derive_features(pd.DataFrame([_row(lesion_count_26conn=1,
                                             small_lesion_count_le5vox=0,
                                             largest_lesion_volume_ml=10.0)]))
    assert out.iloc[0].largest_lesion_share == pytest.approx(1.0)
    assert out.iloc[0].small_lesion_fraction == 0.0


def test_empty_segmentation_raises_rather_than_inventing_values():
    with pytest.raises(ValueError, match="no lesions"):
        derive_features(pd.DataFrame([_row(lesion_count_26conn=0)]))


def test_missing_week4_column_raises():
    with pytest.raises(KeyError):
        derive_features(pd.DataFrame([_row()]).drop(columns="laterality_index"))


def test_impossible_proportion_is_caught():
    # largest lesion bigger than the total cannot happen; if it does, a Week 4
    # column is wrong and the feature must not be silently built from it.
    with pytest.raises(AssertionError, match="largest_lesion_share"):
        derive_features(pd.DataFrame([_row(largest_lesion_volume_ml=12.0)]))
