"""Score a prediction with the OFFICIAL challenge metrics, unmodified.

`checks/evaluation.py` is the challenge organisers' own scorer, vendored
verbatim in Week 1. This module is a thin wrapper around it — it does not
reimplement anything, because the whole point of vendoring was that our Week 7
numbers be directly comparable to the published leaderboard (ROADMAP 9.1).

**Why this matters more than it looks.** `evaluation.py::getImages` performs two
steps that are easy to get wrong by hand and that silently change every score:

1. The reference is thresholded to keep **label 1 only**. Binarising with
   `mask > 0` instead would fold in label 2 ("other pathology") and produce
   numbers that match no published result.
2. The *prediction* is masked wherever the reference is **label 2**, so a false
   positive on other pathology is neither punished nor rewarded — label 2 is
   "don't care", not background.

Both happen inside the vendored code. Calling it with file paths is what
guarantees we inherit them rather than approximating them.

It reads files from disk with SimpleITK and never canonicalises, which is
exactly why `metadata/derived.py` writes every prediction back in the dataset's
native LPS orientation. A prediction saved in our internal RAS would be scored
against a mirrored reference with no error raised.

Week 7 additionally reimplements these metrics independently and asserts the two
agree (ROADMAP 9.1). This module is the source of truth side of that check.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from checks import evaluation

# The five metrics the challenge leaderboard is built from.
METRIC_NAMES = ("dice", "h95_mm", "avd_percent", "lesion_recall", "lesion_f1")


def score_prediction(reference_path: Path | str, prediction_path: Path | str) -> dict:
    """Official Dice, H95, AVD, lesion recall and lesion F1 for one subject.

    `reference_path` is the subject's raw `wmh.nii.gz`; `prediction_path` is our
    binary prediction, written through `metadata.derived.save_derived`.

    H95 comes back as NaN when the prediction is empty — that is the vendored
    behaviour and is preserved rather than papered over, because an empty
    prediction is a real failure that aggregation must handle explicitly rather
    than silently score as zero distance.
    """
    reference_path, prediction_path = Path(reference_path), Path(prediction_path)
    for path in (reference_path, prediction_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # This call is where label-2 handling happens. Do not inline or replace it.
    test_image, result_image = evaluation.getImages(str(reference_path), str(prediction_path))

    dice = evaluation.getDSC(test_image, result_image)
    avd = evaluation.getAVD(test_image, result_image)
    recall, f1 = evaluation.getLesionDetection(test_image, result_image)
    try:
        h95 = evaluation.getHausdorff(test_image, result_image)
    except Exception:
        # The vendored scorer raises on an empty prediction rather than
        # returning NaN. Record it as NaN so a whole run is not lost to one
        # degenerate subject, and so aggregation can see and report it.
        h95 = float("nan")

    return {
        "dice": float(dice),
        "h95_mm": float(h95),
        "avd_percent": float(avd),
        "lesion_recall": float(recall),
        "lesion_f1": float(f1),
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean of each metric across subjects, NaN-aware and honest about it.

    H95 is NaN wherever a prediction was empty. Averaging with `np.mean` would
    poison the whole column; averaging with `np.nanmean` while saying nothing
    would hide that subjects failed. So the count of NaNs is reported alongside.
    """
    if not rows:
        raise ValueError("No rows to aggregate")
    summary = {}
    for name in METRIC_NAMES:
        values = np.array([row[name] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        summary[name] = float(finite.mean()) if finite.size else float("nan")
        if name == "h95_mm":
            summary["h95_undefined_subjects"] = int((~np.isfinite(values)).sum())
    summary["n_subjects"] = len(rows)
    return summary


def score_in_memory(prediction: np.ndarray, raw_reference: np.ndarray) -> dict:
    """Dice and volume ratio from arrays, mirroring the official label-2 rules.

    The vendored scorer only takes file paths, which makes it far too slow for a
    parameter sweep — writing every candidate prediction of every subject to
    disk to score it would dominate the runtime. This computes the same Dice
    from arrays.

    It must reproduce `evaluation.py::getImages` exactly or the sweep optimises
    a slightly different quantity than Week 7 reports:

      reference  = (raw == 1)          label 2 is NOT part of the target
      prediction = prediction & (raw != 2)   label 2 is "don't care", so a
                                             prediction there is neither
                                             rewarded nor penalised

    `raw_reference` is the UNFILTERED mask array (values 0/1/2) — use
    `loader.load_raw_mask_array`, not `load_wmh_mask`, or the label-2 masking
    cannot be applied.

    Whichever configuration a sweep selects is re-scored with the real vendored
    scorer afterwards, and the two Dice values are asserted to agree.
    """
    reference = raw_reference == 1
    masked_prediction = prediction & (raw_reference != 2)

    total = int(reference.sum()) + int(masked_prediction.sum())
    intersection = int(np.logical_and(masked_prediction, reference).sum())
    dice = 1.0 if total == 0 else 2.0 * intersection / total

    reference_total = int(reference.sum())
    return {
        "dice": float(dice),
        "predicted_voxels": int(masked_prediction.sum()),
        "reference_voxels": reference_total,
        "volume_ratio": (float(masked_prediction.sum()) / reference_total
                         if reference_total else float("nan")),
        "recall_voxelwise": (float(intersection / reference_total)
                             if reference_total else float("nan")),
    }
