"""Phase E (Week 7) — apply the frozen grader ONCE to the 110 sealed subjects.

Prepared in Week 6 and deliberately NOT run then: the 110 are opened in Week 7,
after the classifier is frozen, so nothing about them can have shaped it.

Prerequisites, in order (CLAUDE.md Section 15, Phase E):

    1. the frozen Week 2 pipeline on the 110          (each driver, --splits test)
    2. python -m segmentation.run_ensemble --splits test
    3. python -m features.run_synthseg  (test subjects)
       python -m features.run_features --splits test   -> features_test.csv
    4. python -m classification.apply_frozen            <- this script

Target from the 110 reference masks with the same cut-offs; features from the
PRODUCTION ensemble's predictions (`prediction` rows) — the real pipeline, and
honest here because none of the 110 was ever trained on.

**A mismatch, stated rather than hidden.** The classifier was trained on
features from single leave-one-site-out networks and is applied to features from
the 4-network ensemble. Deliberate — the ensemble is what the pipeline actually
runs — but the two are not the same measuring instrument.

Scores overall and per scanner, including the two scanners never seen in any
training (Amsterdam GE1T5 and Philips_VU).

    code/.venv/bin/python -m classification.apply_frozen
"""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from classification.freeze import MODEL_PATH, versions
from classification.metrics import score_all
from classification.protocol import build_dataset
from metadata.config import CODE_ROOT, SEVERITY_CONFIG
from metadata.loader import subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "apply_frozen"
GENERATING = f"code/classification/{SCRIPT_NAME}.py"
OUTPUTS_DIR = CODE_ROOT / "classification" / "outputs"
TEST_FEATURES_CSV = CODE_ROOT / "features" / "outputs" / "features_test.csv"
PREDICTIONS_CSV = OUTPUTS_DIR / "week7_severity_predictions.csv"
SCORES_JSON = OUTPUTS_DIR / "week7_severity_scores.json"
MISMATCH_NOTE = ("trained on single leave-one-site-out network features, applied to "
                 "4-network production ensemble features — deliberate, stated")


def apply_bundle(bundle: dict, features_table: pd.DataFrame, scanner_of: dict) -> tuple:
    """Predict every subject's class and score it, overall and per scanner.

    Pure (no I/O), so it is tested on a synthetic table without opening the
    sealed set. Refuses a bundle whose cut-offs no longer match dataset.yaml —
    the frozen model's classes would then mean something else than the target's.
    """
    if bundle["cutoffs_ml"] != SEVERITY_CONFIG["cutoffs_ml"]:
        raise ValueError("the frozen model's cut-offs differ from dataset.yaml's")
    data = build_dataset(features_table, SEVERITY_CONFIG, feature_source="prediction",
                         expected_subjects=None)
    predicted = np.asarray(bundle["model"].predict(data.frame), dtype=int)
    n_classes = len(bundle["classes"])
    scanners = np.array([scanner_of[key] for key in data.keys])
    table = pd.DataFrame({"subject_key": data.keys, "scanner": scanners,
                          "reference_volume_ml": data.reference_volume_ml,
                          "predicted_volume_ml": data.frame["predicted_volume_ml"].to_numpy(),
                          "true_class": [bundle["classes"][c] for c in data.class4],
                          "given_class": [bundle["classes"][c] for c in predicted],
                          "y_true": data.class4, "y_pred": predicted})
    scores = {"overall": {"n": len(table), **score_all(data.class4, predicted, n_classes)}}
    for scanner, part in table.groupby("scanner"):
        scores[scanner] = {"n": len(part), **score_all(part.y_true.to_numpy(),
                                                       part.y_pred.to_numpy(), n_classes)}
    return table, scores


def scanner_label(subject) -> str:
    return f"{subject.site}/{subject.scanner_dir}" if subject.scanner_dir else subject.site


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features-csv", default=str(TEST_FEATURES_CSV))
    args = parser.parse_args()
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    bundle = joblib.load(MODEL_PATH)
    if bundle["versions"]["scikit-learn"] != versions()["scikit-learn"]:
        raise SystemExit(f"model frozen under scikit-learn {bundle['versions']['scikit-learn']}, "
                         f"running {versions()['scikit-learn']} — not guaranteed to load "
                         f"identically")
    table_in = pd.read_csv(args.features_csv)
    subjects = subjects_by_key()
    scanner_of = {key: scanner_label(subjects[key]) for key in table_in.subject_key.unique()}
    logger.info("applying frozen %s ONCE to %d subjects from %s", bundle["candidate_id"],
                len(scanner_of), args.features_csv)
    logger.info("NOTE: %s", MISMATCH_NOTE)

    table, scores = apply_bundle(bundle, table_in, scanner_of)
    table.to_csv(PREDICTIONS_CSV, index=False)
    SCORES_JSON.write_text(json.dumps(scores, indent=2))
    for path in (PREDICTIONS_CSV, SCORES_JSON):
        write_manifest(path, generating_script=GENERATING,
                       extra={"model": bundle["candidate_id"], "mismatch": MISMATCH_NOTE,
                              "cv_mean_qwk_at_freeze": bundle["cv_mean_qwk"]})
    for group, s in scores.items():
        logger.info("  %-32s n=%3d  QWK %.3f  MACE %.3f  bal.acc %.3f  signed %+.3f", group,
                    s["n"], s["qwk"], s["mean_absolute_class_error"],
                    s["balanced_accuracy"], s["mean_signed_class_error"])


if __name__ == "__main__":
    main()
