"""Weeks 5-6 — severity classes and classifier features (R10).

Tables in, tables out: no paths, no I/O, so everything here is testable against
inputs with a known answer (`checks/test_severity.py`). The decisions these
functions implement — which cut-offs, which features, and why — are recorded in
`metadata/dataset.yaml` under `severity_classification`, and were fixed before
any model was fitted or any exploratory plot was drawn.

**The target.** There is no severity ground truth in this dataset, so the class
is derived from the expert reference WMH volume, using the Fazekas-anchored
cut-offs of Joo et al., PLOS ONE 2022 (3.4 / 9.6 / 17.1 mL). Those were derived
against radiologists' Fazekas grades, so a class here means what a radiologist
would call it — not "worse than two thirds of these 60 people", which is all a
tertile can mean.

**The features.** Six, reduced from the fourteen Week 4 produced *by definition*
rather than by checking which ones predict the class:

- one feature for HOW MUCH disease there is (log total volume), and
- five for its PATTERN — where it sits, whether it is one-sided, how scattered,
  how much is punctate, how confluent.

The pattern features are proportions on purpose. Most raw Week 4 measurements
correlate 0.85-0.98 with total volume, so they would hand a model the same
number several times over. Proportions are near-independent of it, which is the
only way they can add anything a classifier does not already have.

The last two follow the Fazekas scale's own wording — deep WMH progress from
"punctate foci" to "beginning confluence" to "large confluent areas" — so they
measure the progression a radiologist is actually grading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Week 4 column names, in one place, so a rename upstream fails here loudly
# instead of silently producing NaN features.
TOTAL_ML = "total_lesion_volume_ml"
PV_FRACTION = "periventricular_fraction"
LATERALITY = "laterality_index"
COUNT = "lesion_count_26conn"
SMALL_COUNT = "small_lesion_count_le5vox"
LARGEST_ML = "largest_lesion_volume_ml"

FEATURE_NAMES = (
    "log_total_volume",
    "periventricular_fraction",
    "abs_laterality_index",
    "log_lesion_count",
    "small_lesion_fraction",
    "largest_lesion_share",
)


def assign_class(volume_ml, cutoffs_ml) -> np.ndarray:
    """Class index for each volume: 0 below the first cut-off, len(cutoffs) above the last.

    A volume exactly equal to a cut-off goes to the HIGHER class
    (normal < 3.4 <= mild), the convention recorded in dataset.yaml. Stated
    explicitly because the opposite convention silently moves a subject sitting
    on a boundary, and with 8 subjects in the smallest class one subject is
    12% of it.
    """
    volume_ml = np.asarray(volume_ml, dtype=float)
    cutoffs = np.asarray(cutoffs_ml, dtype=float)
    if cutoffs.ndim != 1 or cutoffs.size == 0 or np.any(np.diff(cutoffs) <= 0):
        raise ValueError(f"cut-offs must be strictly increasing, got {cutoffs_ml}")
    if np.any(~np.isfinite(volume_ml)) or np.any(volume_ml < 0):
        raise ValueError("volumes must be finite and non-negative")
    # right=False: bins[i-1] <= x < bins[i], so x == cut-off lands in the upper class.
    return np.digitize(volume_ml, cutoffs, right=False)


def derive_features(table: pd.DataFrame) -> pd.DataFrame:
    """The six classifier inputs, from rows of Week 4's features.csv.

    Returns a new frame indexed like `table`, holding exactly FEATURE_NAMES.

    A row with no lesions at all raises rather than inventing values: every
    proportion is undefined there, and choosing a stand-in (0? the cohort
    median?) is a modelling decision, not an arithmetic one. No reference mask
    in this cohort is empty (minimum 0.79 mL). A prediction could be — most
    plausibly on Week 7's unseen scanners — and that decision should be made
    deliberately when it happens, not defaulted here.
    """
    required = {TOTAL_ML, PV_FRACTION, LATERALITY, COUNT, SMALL_COUNT, LARGEST_ML}
    missing = required - set(table.columns)
    if missing:
        raise KeyError(f"features.csv is missing {sorted(missing)} — has Week 4 been re-run?")

    empty = table[COUNT] <= 0
    if empty.any():
        raise ValueError(
            f"{int(empty.sum())} row(s) have no lesions; proportions are undefined. "
            f"Decide how to treat an empty segmentation explicitly. "
            f"First: {list(table.index[empty][:3])}")
    if table[list(required)].isna().any().any():
        raise ValueError("NaN in Week 4 features — R5 columns are blank where no "
                         "ventricle mask existed; re-run features.run_synthseg")

    out = pd.DataFrame(index=table.index)
    # log1p, not log: lesion volume is roughly log-normal across a population,
    # and log1p stays defined at zero should a later caller relax the guard above.
    out["log_total_volume"] = np.log1p(table[TOTAL_ML].astype(float))
    out["periventricular_fraction"] = table[PV_FRACTION].astype(float)
    # Magnitude only: severity has no left/right direction, and the signed index
    # averages to ~0 across subjects (Week 4: -0.006), which would hide it.
    out["abs_laterality_index"] = table[LATERALITY].astype(float).abs()
    out["log_lesion_count"] = np.log1p(table[COUNT].astype(float))
    out["small_lesion_fraction"] = table[SMALL_COUNT] / table[COUNT]
    out["largest_lesion_share"] = table[LARGEST_ML] / table[TOTAL_ML]

    for name in ("periventricular_fraction", "abs_laterality_index",
                 "small_lesion_fraction", "largest_lesion_share"):
        values = out[name]
        if ((values < 0) | (values > 1)).any():
            raise AssertionError(f"{name} outside [0, 1] — a proportion cannot be; "
                                 f"check the Week 4 columns it is built from")
    return out[list(FEATURE_NAMES)]


def merge_to_three_class(class4) -> np.ndarray:
    """The 3-class sensitivity scheme, as a pure function of the 4-class label.

    normal and mild collapse into one class; moderate and severe keep their
    meaning and shift down one index. MERGING, never relabelling: no patient's
    clinical meaning changes, which is the point of the sensitivity scheme
    (dataset.yaml `sensitivity_scheme`). The experiment driver asserts this
    equals `assign_class(volume, sensitivity_cutoffs_ml)` for every patient, so
    the config and this function cannot drift apart silently.
    """
    class4 = np.asarray(class4)
    if class4.size and (class4.min() < 0 or class4.max() > 3):
        raise ValueError("4-class labels must be in 0..3")
    return np.maximum(class4 - 1, 0)


def near_cutoff(volume_ml, cutoffs_ml, band: float) -> np.ndarray:
    """True where a volume sits within a relative `band` of any cut-off.

    The same definition the EDA used for its 14 "boundary patients": within one
    median held-out measurement error (band = 0.129) of a cut-off c means
    c / (1 + band) <= v <= c * (1 + band). An ordinary measurement error can
    move such a patient across the line, so no classifier can be expected to
    place them reliably; they are scored separately.
    """
    volume_ml = np.asarray(volume_ml, dtype=float)
    if band < 0:
        raise ValueError("band must be non-negative")
    return np.array([any(c / (1 + band) <= v <= c * (1 + band) for c in cutoffs_ml)
                     for v in volume_ml], dtype=bool)
