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

**Prediction-derived features must be regenerated whenever the model changes.**
They are currently computed from the single seed-42 U-Net; after the ensemble
lands they need re-running. Cheap (seconds), but it has to happen in that order.

    code/.venv/bin/python -m features.run_features
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from metadata.config import (PERIVENTRICULAR_SENSITIVITY_THRESHOLDS_MM,
                             PERIVENTRICULAR_THRESHOLD_MM, PROJECT_ROOT)
from metadata.derived import (BRAIN_MASK, PRED_WMH, VENTRICLES, derived_exists,
                             load_derived_mask)
from metadata.geometry import voxel_volume_mm3
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from features.lesion_features import extract_features
from features.periventricular import summarise as periventricular_summary

SCRIPT_NAME = "run_features"
OUTPUTS_DIR = PROJECT_ROOT / "features" / "outputs"
FEATURES_CSV = OUTPUTS_DIR / "features.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    args = parser.parse_args()

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

    rows = []
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

        if index % 15 == 0 or index == len(keys):
            logger.info("  %d/%d", index, len(keys))

    if missing_ventricles:
        logger.warning("no ventricle mask for %d subject(s) — R5 columns blank for them. "
                       "Run: python -m features.run_synthseg",
                       len(missing_ventricles))
    if missing_predictions:
        logger.warning("no prediction found for %d subject(s) — reference features only: %s",
                       len(missing_predictions), missing_predictions[:5])

    table = pd.DataFrame(rows)
    table.to_csv(FEATURES_CSV, index=False)
    write_manifest(FEATURES_CSV, generating_script=f"features/{SCRIPT_NAME}.py",
                   extra={"note": "R5-R9. Prediction-side features must be regenerated "
                                  "if the segmentation model changes.",
                          "r5_primary_mm": primary,
                          "r5_sensitivity_mm": list(thresholds),
                          "r5_source": "SynthSeg on raw T1 (selected by sweep)"})
    logger.info("wrote %s (%d rows)", FEATURES_CSV, len(table))

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

    if not prediction_rows.empty:
        merged = reference_rows.merge(prediction_rows, on="subject_key",
                                      suffixes=("_ref", "_pred"))
        logger.info("how well do our MEASUREMENTS match the expert's? (Spearman)")
        for column in ("total_lesion_volume_ml", "lesion_count_26conn",
                       "largest_lesion_volume_ml", "largest_lesion_feret_mm"):
            rho = merged[f"{column}_ref"].corr(merged[f"{column}_pred"], method="spearman")
            logger.info("    %-28s rho = %+.3f", column, rho)
        logger.info("  (a high volume correlation with a lower COUNT correlation would mean "
                    "we measure burden well but miscount individual lesions — exactly the "
                    "small-lesion blindness Week 2 predicted)")


if __name__ == "__main__":
    main()
