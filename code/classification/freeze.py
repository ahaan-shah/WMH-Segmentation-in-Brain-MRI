"""Weeks 5-6, Phase D5 — refit the selected grader on all 60 and freeze it.

Cross-validation answered "how well does this KIND of model do?". The model
Week 7 applies is one more fit of that kind, on all 60 leave-one-site-out
feature rows, saved once and never refit. A tuned candidate re-runs its inner
cross-validation on the 60 to choose its setting — the same procedure each CV
fold used, so the frozen model is made exactly the way the scored ones were.

Its cross-validated score is carried in the bundle and labelled as mildly
OPTIMISTIC: selection looked at that score to choose this model, which biases it
upward. The number that counts is Week 7's, on the 110 sealed subjects.

Saved with joblib plus a sidecar recording the library versions, because a
pickled scikit-learn / statsmodels object is only guaranteed to load under the
versions that wrote it.

    code/.venv/bin/python -m classification.freeze
"""

from __future__ import annotations

import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import statsmodels

from classification.interpret import features_of, split_id
from classification.models import build
from classification.protocol import build_dataset
from classification.run_experiments import FEATURES_CSV, SCHEMES
from classification.select import SELECTION_JSON
from metadata.config import CODE_ROOT, SEVERITY_CONFIG
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "freeze"
GENERATING = f"code/classification/{SCRIPT_NAME}.py"
OUTPUTS_DIR = CODE_ROOT / "classification" / "outputs"
MODEL_PATH = OUTPUTS_DIR / "final_model.joblib"


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__, "joblib": joblib.__version__}


def make_bundle(candidate_id: str, data, cv_mean_qwk: float) -> dict:
    """The frozen model plus everything needed to apply it correctly later."""
    _, name = split_id(candidate_id)
    features = features_of(candidate_id)
    cutoffs, classes = SCHEMES["4class"]
    model = build(name, features, len(classes), cutoffs).fit(
        data.frame, data.class4, data.log_reference_volume)
    return {
        "model": model,
        "candidate_id": candidate_id,
        "features": list(features),
        "scheme": "4class",
        "cutoffs_ml": list(cutoffs),
        "classes": list(classes),
        "trained_on": {"n_patients": len(data.keys), "subject_keys": data.keys,
                       "feature_source": "loso_prediction", "target_source": "reference"},
        "tuned_value": model.chosen_,
        "cv_mean_qwk": cv_mean_qwk,
        "cv_score_note": "mildly optimistic: the selection rule looked at this score",
        "versions": versions(),
    }


def main() -> None:
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    if not SELECTION_JSON.exists():
        raise SystemExit("selection.json missing — run classification.select first")
    selection = json.loads(SELECTION_JSON.read_text())
    data = build_dataset(pd.read_csv(FEATURES_CSV), SEVERITY_CONFIG)
    bundle = make_bundle(selection["selected"], data, selection["selected_mean_qwk"])

    # The refit must reproduce the training-set behaviour it was selected for;
    # a model that cannot even predict its own training rows sensibly is a bug.
    predicted = bundle["model"].predict(data.frame)
    agreement = float(np.mean(predicted == data.class4))
    logger.info("frozen %s on %d patients (tuned value %s); training-set agreement %.2f "
                "(descriptive only — not a performance estimate)", bundle["candidate_id"],
                len(data.keys), bundle["tuned_value"], agreement)

    joblib.dump(bundle, MODEL_PATH)
    write_manifest(MODEL_PATH, generating_script=GENERATING,
                   extra={k: v for k, v in bundle.items() if k != "model"}
                   | {"training_set_agreement": agreement})
    logger.info("wrote %s with versions %s", MODEL_PATH.name, bundle["versions"])

    # Load it straight back and check it predicts identically — a frozen model
    # that does not round-trip is not frozen.
    reloaded = joblib.load(MODEL_PATH)
    if not np.array_equal(reloaded["model"].predict(data.frame), predicted):
        raise AssertionError("the reloaded model predicts differently from the one saved")
    logger.info("round-trip check passed")


if __name__ == "__main__":
    main()
