"""Week 4 driver — extract lesion features for every subject (R6-R9).

Computes every feature **twice per subject**: once from the expert reference
mask and once from our prediction. That is not redundancy, it is what makes
Weeks 5-7 possible:

- **Weeks 5-6 (severity classification).** The severity label has to be derived
  from the *reference* lesion volume, because no severity ground truth ships
  with this dataset. If the classifier's input features also came from the
  reference, the model would be predicting a number computed from its own
  inputs — it would score near-perfectly and mean nothing. Target from
  reference, features from prediction. This file produces both halves.
- **Week 7.** "How good is our segmentation?" and "how good are our
  *measurements*?" are different questions. Comparing the two halves answers
  the second.

**R5 (periventricular vs deep) is not here yet** — it needs a ventricle mask,
which is blocked on FreeSurfer/SynthSeg. The columns slot in without disturbing
anything else once that lands.

**A third source, `loso_prediction` (Weeks 5-6).** Where a leave-one-site-out
prediction exists (`segmentation.run_loso_predictions`, the 60 labelled subjects
only), it is measured too. Those are the severity classifier's inputs: the
production prediction is in-sample for 48 of the 60, the leave-one-site-out one
is honest for all 60. Its rows are written AFTER all reference/prediction rows,
so the Week 4 rows keep their exact bytes and order in the CSV; they are only
read if that script's per-site Dice check passed, and an empty one stops the
run rather than writing a row of NaNs.

**Prediction-derived features must be regenerated whenever the model changes.**
They are currently computed from the single seed-42 U-Net; after the ensemble
lands they need re-running. Cheap (seconds), but it has to happen in that order.

    code/.venv/bin/python -m features.run_features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classification.severity import assign_class
from metadata.config import (CODE_ROOT, PERIVENTRICULAR_SENSITIVITY_THRESHOLDS_MM,
                             PERIVENTRICULAR_THRESHOLD_MM, SEVERITY_CONFIG)
from metadata.derived import (BRAIN_MASK, PRED_WMH, PRED_WMH_LOSO, VENTRICLES,
                             derived_exists, load_derived_mask)
from metadata.geometry import voxel_volume_mm3
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from metadata.split_outputs import split_output
from features.lesion_features import extract_features
from features.periventricular import summarise as periventricular_summary

SCRIPT_NAME = "run_features"
# Written by segmentation.run_loso_predictions; read here as a file rather than
# imported, so this script does not pull in torch.
LOSO_VERDICT_JSON = CODE_ROOT / "segmentation" / "outputs" / "loso_prediction_check.json"
OUTPUTS_DIR = CODE_ROOT / "features" / "outputs"
FEATURES_CSV = OUTPUTS_DIR / "features.csv"
TEST_FEATURES_CSV = OUTPUTS_DIR / "features_test.csv"


def output_csv_for(splits) -> Path:
    """Where a run's table goes: the 110 sealed subjects never share a file with the 60.

    Found 2026-10-08 while preparing the Week 7 handoff: with a single output
    path, Week 7's `--splits test` would have overwritten the 60 subjects'
    table — and with it every classifier input and the Week 4 results — with
    the 110 test rows. The rule now lives in `metadata.split_outputs`, shared
    with every other driver that writes a per-run table.
    """
    path = split_output(FEATURES_CSV, splits)
    assert path in (FEATURES_CSV, TEST_FEATURES_CSV)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    output_csv = output_csv_for(args.splits)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    subjects = subjects_by_key()
    keys = [key for split in args.splits for key in load_split(split)]
    thresholds = tuple(PERIVENTRICULAR_SENSITIVITY_THRESHOLDS_MM)
    primary = float(PERIVENTRICULAR_THRESHOLD_MM)
    logger.info("extracting R5-R9 features for %d subjects, from BOTH the reference "
                "and the prediction", len(keys))
    logger.info("R5 split at %.0f mm (DeCarli 2005), sensitivity also at %s mm",
                primary, list(thresholds))
    missing_ventricles = []
    loso_keys = check_loso_predictions(keys, subjects, logger)

    rows = []
    loso_rows = []   # appended after every Week 4 row — see the module docstring
    missing_predictions = []
    for index, key in enumerate(keys, start=1):
        subject = subjects[key]
        image = load_nifti(subject.flair_path)
        affine, voxel_volume = image.affine, voxel_volume_mm3(image)
        brain = load_derived_mask(key, BRAIN_MASK)

        spacing = np.abs(np.diag(affine))[:3]
        ventricles = (load_derived_mask(key, VENTRICLES)
                      if derived_exists(key, VENTRICLES) else None)
        if ventricles is None:
            missing_ventricles.append(key)

        def with_r5(mask, source):
            """R6-R9 plus, where a ventricle mask exists, R5."""
            row = {"subject_key": key, "split": subject.split, "site": subject.site,
                   **extract_features(mask, affine, voxel_volume,
                                      brain_mask=brain, source=source)}
            if ventricles is not None and mask.any():
                row.update(periventricular_summary(
                    mask, ventricles, spacing,
                    thresholds_mm=thresholds, primary_mm=primary))
            return row

        reference = load_wmh_mask(subject.mask_path)
        rows.append(with_r5(reference, "reference"))

        if derived_exists(key, PRED_WMH):
            rows.append(with_r5(load_derived_mask(key, PRED_WMH), "prediction"))
        else:
            missing_predictions.append(key)

        if key in loso_keys:
            loso_mask = load_derived_mask(key, PRED_WMH_LOSO)
            if not loso_mask.any():
                raise SystemExit(
                    f"{key}: the leave-one-site-out prediction is EMPTY. Every proportion "
                    f"the classifier uses is undefined for it; decide how to treat it "
                    f"explicitly rather than writing a row of NaNs (CLAUDE.md 15, A3).")
            loso_rows.append(with_r5(loso_mask, "loso_prediction"))

        if index % 15 == 0 or index == len(keys):
            logger.info("  %d/%d", index, len(keys))

    if missing_ventricles:
        logger.warning("no ventricle mask for %d subject(s) — R5 columns blank for them. "
                       "Run: python -m features.run_synthseg",
                       len(missing_ventricles))
    if missing_predictions:
        logger.warning("no prediction found for %d subject(s) — reference features only: %s",
                       len(missing_predictions), missing_predictions[:5])

    table = pd.DataFrame(rows + loso_rows)
    table.to_csv(output_csv, index=False)
    write_manifest(output_csv, generating_script=f"code/features/{SCRIPT_NAME}.py",
                   extra={"note": "R5-R9. Prediction-side features must be regenerated "
                                  "if the segmentation model changes.",
                          "r5_primary_mm": primary,
                          "r5_sensitivity_mm": list(thresholds),
                          "r5_source": "SynthSeg on raw T1 (selected by sweep)",
                          "sources": sorted(table.source.unique().tolist()),
                          "loso_prediction_rows_appended_last": len(loso_rows)})
    logger.info("wrote %s (%d rows)", output_csv, len(table))

    pd.set_option("display.width", 250)
    reference_rows = table[table.source == "reference"]
    prediction_rows = table[table.source == "prediction"]

    logger.info("REFERENCE features by site:\n%s", reference_rows.groupby("site")[
        ["lesion_count_26conn", "total_lesion_volume_ml", "largest_lesion_volume_ml",
         "largest_lesion_feret_mm", "largest_lesion_equiv_sphere_mm"]
    ].mean().round(2).to_string())

    # R8: the two diameter definitions, side by side. This is the evidence for
    # having switched the primary definition in Week 4.
    ratio = (reference_rows["largest_lesion_feret_mm"]
             / reference_rows["largest_lesion_equiv_sphere_mm"].replace(0, np.nan))
    logger.info("R8 — max Feret is %.2fx the equivalent-sphere diameter on average "
                "(median %.2fx, max %.2fx). Equivalent-sphere under-reports every "
                "elongated lesion, which is most of the large ones here.",
                ratio.mean(), ratio.median(), ratio.max())

    logger.info("R6 — connectivity: 26-conn finds %.1f lesions per subject, 6-conn %.1f "
                "(%.2fx). Both reported; 26 matches the official scorer.",
                reference_rows["lesion_count_26conn"].mean(),
                reference_rows["lesion_count_6conn"].mean(),
                (reference_rows["lesion_count_6conn"]
                 / reference_rows["lesion_count_26conn"]).mean())

    for source in ("prediction", "loso_prediction"):
        source_rows = table[table.source == source]
        if source_rows.empty:
            continue
        log_agreement(reference_rows, source_rows, source, logger)


def check_loso_predictions(keys, subjects, logger) -> set:
    """Which subjects get a `loso_prediction` row — all 60 labelled ones, or none.

    Refuses a partial set (a crashed run would otherwise hand the classifier
    fewer than 60 patients without saying so) and refuses any set whose
    per-site Dice check did not pass.
    """
    present = {key for key in keys if derived_exists(key, PRED_WMH_LOSO)}
    if not present:
        return set()
    labelled = {key for key in keys if subjects[key].split == "training"}
    if present != labelled:
        raise SystemExit(
            f"leave-one-site-out predictions exist for {len(present)} of the "
            f"{len(labelled)} labelled subjects in this run — partial. Re-run "
            f"segmentation.run_loso_predictions to completion.")
    if not LOSO_VERDICT_JSON.exists():
        raise SystemExit(f"{LOSO_VERDICT_JSON.name} is missing: the run that wrote these "
                         f"predictions did not finish its per-site Dice check")
    verdict = json.loads(LOSO_VERDICT_JSON.read_text())
    if not verdict.get("passed"):
        raise SystemExit("leave-one-site-out predictions FAILED their per-site Dice check "
                         f"({LOSO_VERDICT_JSON}); refusing to measure them")
    logger.info("leave-one-site-out predictions: %d subjects, per-site Dice check passed",
                len(present))
    return present


def log_agreement(reference_rows, source_rows, source, logger) -> None:
    """How well one prediction source's MEASUREMENTS match the expert's (Phase A4)."""
    merged = reference_rows.merge(source_rows, on="subject_key", suffixes=("_ref", "_pred"))
    logger.info("how well do the %s MEASUREMENTS match the expert's? (Spearman, n=%d)",
                source, len(merged))
    for column in ("total_lesion_volume_ml", "lesion_count_26conn",
                   "largest_lesion_volume_ml", "largest_lesion_feret_mm"):
        rho = merged[f"{column}_ref"].corr(merged[f"{column}_pred"], method="spearman")
        logger.info("    %-28s rho = %+.3f", column, rho)
    logger.info("  (a high volume correlation with a lower COUNT correlation would mean "
                "we measure burden well but miscount individual lesions — exactly the "
                "small-lesion blindness Week 2 predicted)")

    # The threshold rule, applied to this source's volumes: the simplest possible
    # severity classifier, and the incumbent every Weeks 5-6 model must beat.
    cutoffs = SEVERITY_CONFIG["cutoffs_ml"]
    truth = assign_class(merged["total_lesion_volume_ml_ref"], cutoffs)
    given = assign_class(merged["total_lesion_volume_ml_pred"], cutoffs)
    error = (merged["total_lesion_volume_ml_pred"]
             / merged["total_lesion_volume_ml_ref"] - 1).abs()
    for label, mask in (("all", np.ones(len(merged), bool)),
                        ("validation split", (merged["split_ref"] == "training")
                         & merged["subject_key"].isin(load_split("val")))):
        if mask.any():
            logger.info("    threshold rule on %s volumes, %s (n=%d): 4-class accuracy "
                        "%.1f%%, median |volume error| %.1f%%", source, label,
                        int(mask.sum()), 100 * np.mean(truth[mask] == given[mask]),
                        100 * np.median(error[mask]))


if __name__ == "__main__":
    main()
