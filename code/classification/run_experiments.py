"""Weeks 5-6, Phases B5 / C1 / D1 — run every candidate through the same folds.

    code/.venv/bin/python -m classification.run_experiments --experiments volume_only     # Week 5
    code/.venv/bin/python -m classification.run_experiments --experiments full_features   # Week 6

**What runs.** The two baselines (majority class, threshold rule) always run —
they are cheap and every table needs them beside it. Then the four learned
candidates on each requested feature set:

- `volume_only` (Experiment 2) — log total volume alone. Can a model that LEARNS
  do better with the same single number than the fixed clinical cut-offs?
- `full_features` (Experiment 3) — volume plus the five pattern features. Does
  the PATTERN of disease add anything beyond its AMOUNT? The EDA predicted not
  (same-volume rho 0.04-0.16); this is the test under cross-validation.

Both class schemes run every time: the 4-class primary, and the merged 3-class
sensitivity scheme — reported beside it, never used for selection.

**Same folds for everyone, across runs too.** The fold assignment is written to
`experiments/folds.csv` on first use. Every later run (Week 6 after Week 5)
rebuilds it from SEED and checks it is identical to the saved one before
fitting anything, so the two experiments are guaranteed to have been dealt the
same patients — the condition `select.py`'s paired comparison depends on.

**What is written** (`classification/outputs/experiments/`), each with a sidecar:

- `oof_<experiment>.csv` — every patient's out-of-fold prediction, per
  candidate, scheme and repeat. The raw evidence; everything else is computed
  from it.
- `summary_<experiment>.csv` — per candidate and scheme: every score as mean and
  SD over repeats, per-hospital QWK, boundary-patient scores, convergence count,
  chosen tuning values.
- `confusion_<experiment>.json` — confusion matrices pooled over repeats.

Features come from the leave-one-site-out predictions (`loso_prediction` rows of
features.csv); the target from the expert reference. See dataset.yaml
`feature_source` for why.
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial

import numpy as np
import pandas as pd

from classification.models import BASELINES, LEARNED, build
from classification.protocol import (build_dataset, cross_validate, folds_table,
                                     make_folds, pooled_confusion, summarise)
from classification.severity import near_cutoff
from metadata.config import CODE_ROOT, SEED, SEVERITY_CONFIG
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "run_experiments"
GENERATING = f"code/classification/{SCRIPT_NAME}.py"
OUTPUTS_DIR = CODE_ROOT / "classification" / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
FEATURES_CSV = CODE_ROOT / "features" / "outputs" / "features.csv"
EDA_SUMMARY = OUTPUTS_DIR / "eda_summary.json"
FOLDS_CSV = EXPERIMENTS_DIR / "folds.csv"

FEATURE_SETS = {"volume_only": SEVERITY_CONFIG["feature_set_volume_only"],
                "full_features": SEVERITY_CONFIG["feature_set_full"]}
SCHEMES = {
    "4class": (SEVERITY_CONFIG["cutoffs_ml"], SEVERITY_CONFIG["classes"]),
    "3class": (SEVERITY_CONFIG["sensitivity_cutoffs_ml"],
               SEVERITY_CONFIG["sensitivity_classes"]),
}


def boundary_patients(data, scheme: str, eda: dict) -> set:
    """Patients within one median held-out volume error of a cut-off (EDA definition).

    For the 4-class scheme this must reproduce the EDA's count exactly (14);
    a mismatch means the definition or the data has drifted, and the run stops.
    """
    cutoffs, _ = SCHEMES[scheme]
    band = eda["heldout_median_abs_volume_error"]
    near = near_cutoff(data.reference_volume_ml, cutoffs, band)
    if scheme == "4class" and int(near.sum()) != eda["n_within_median_error_of_cutoff"]:
        raise AssertionError(f"{int(near.sum())} boundary patients, but the EDA found "
                             f"{eda['n_within_median_error_of_cutoff']}")
    return {key for key, flag in zip(data.keys, near) if flag}


def shared_folds(data, logger):
    """The one fold assignment, checked against any earlier run's before use."""
    folds = make_folds(data.class4, SEVERITY_CONFIG["cv_folds"],
                       SEVERITY_CONFIG["cv_repeats"], SEED)
    table = folds_table(folds, data.keys)
    if FOLDS_CSV.exists():
        saved = pd.read_csv(FOLDS_CSV)
        if not saved.equals(table):
            raise SystemExit(f"{FOLDS_CSV.name} differs from the folds rebuilt now — the "
                             f"patients, their classes or the seed changed since the "
                             f"earlier experiment. The experiments would not be paired. "
                             f"Investigate before deleting it.")
        logger.info("fold assignment identical to the saved %s", FOLDS_CSV.name)
    else:
        table.to_csv(FOLDS_CSV, index=False)
        write_manifest(FOLDS_CSV, generating_script=GENERATING,
                       extra={"splitter": "RepeatedStratifiedKFold",
                              "n_splits": SEVERITY_CONFIG["cv_folds"],
                              "n_repeats": SEVERITY_CONFIG["cv_repeats"],
                              "stratified_on": "4-class reference target", "seed": SEED})
        logger.info("fold assignment written to %s", FOLDS_CSV.name)
    return folds


def run_block(experiment, candidates, features, data, folds, eda, logger):
    """Every (candidate, scheme) for one experiment. Returns (oof, summary, confusions)."""
    oofs, summaries, confusions = [], [], {}
    for scheme, (cutoffs, classes) in SCHEMES.items():
        boundary = boundary_patients(data, scheme, eda)
        for name in candidates:
            started = time.time()
            factory = partial(build, name, features, len(classes), cutoffs)
            oof = cross_validate(factory, data, scheme, folds)
            oof.insert(0, "experiment", experiment)
            oof.insert(1, "scheme", scheme)
            oof.insert(2, "candidate", name)
            summary = summarise(oof, len(classes), boundary)
            summaries.append({"experiment": experiment, "scheme": scheme,
                              "candidate": name, "features": "+".join(features),
                              **summary})
            confusions[f"{scheme}/{name}"] = pooled_confusion(oof, len(classes)).tolist()
            oofs.append(oof)
            logger.info("  %-7s %-27s QWK %.3f +- %.3f  MACE %.3f  bal.acc %.3f  "
                        "signed %+.3f  (%.0fs%s)", scheme, name, summary["qwk_mean"],
                        summary["qwk_sd"], summary["mean_absolute_class_error_mean"],
                        summary["balanced_accuracy_mean"],
                        summary["mean_signed_class_error_mean"], time.time() - started,
                        f", {summary['fits_not_converged']} fits NOT converged"
                        if summary["fits_not_converged"] else "")
            if abs(summary["mean_signed_class_error_mean"]) > 0.25:
                logger.warning("    %s leans %s by %.2f classes on average — flagged for "
                               "the report (dataset.yaml: |value| > 0.25)", name,
                               "UP" if summary["mean_signed_class_error_mean"] > 0 else "DOWN",
                               abs(summary["mean_signed_class_error_mean"]))
    return pd.concat(oofs, ignore_index=True), pd.DataFrame(summaries), confusions


def write_block(tag, oof, summary, confusions, data, extra):
    oof_path = EXPERIMENTS_DIR / f"oof_{tag}.csv"
    summary_path = EXPERIMENTS_DIR / f"summary_{tag}.csv"
    confusion_path = EXPERIMENTS_DIR / f"confusion_{tag}.json"
    oof.to_csv(oof_path, index=False)
    summary.to_csv(summary_path, index=False)
    confusion_path.write_text(json.dumps(confusions, indent=2))
    meta = {"n_patients": len(data.keys), "feature_source": "loso_prediction",
            "target_source": "reference", "seed": SEED, **extra}
    for path in (oof_path, summary_path, confusion_path):
        write_manifest(path, generating_script=GENERATING, extra=meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=list(FEATURE_SETS),
                        choices=list(FEATURE_SETS))
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    if not EDA_SUMMARY.exists():
        raise SystemExit("eda_summary.json missing — run classification.explore first; the "
                         "boundary-patient definition comes from it")
    eda = json.loads(EDA_SUMMARY.read_text())
    data = build_dataset(pd.read_csv(FEATURES_CSV), SEVERITY_CONFIG)
    logger.info("%d patients; features from leave-one-site-out predictions, target from "
                "the expert reference", len(data.keys))
    logger.info("4-class counts %s; 3-class counts %s",
                np.bincount(data.class4, minlength=4).tolist(),
                np.bincount(data.class3, minlength=3).tolist())
    folds = shared_folds(data, logger)
    logger.info("%d-fold x %d repeats, stratified on the 4-class target, seed %d",
                SEVERITY_CONFIG["cv_folds"], SEVERITY_CONFIG["cv_repeats"], SEED)

    logger.info("BASELINES (no features beyond volume; never selectable / tier-1 incumbent)")
    oof, summary, confusions = run_block("baseline", BASELINES,
                                         SEVERITY_CONFIG["feature_set_volume_only"],
                                         data, folds, eda, logger)
    write_block("baselines", oof, summary, confusions, data, {})
    baseline_summary = summary

    for experiment in args.experiments:
        features = FEATURE_SETS[experiment]
        logger.info("EXPERIMENT %s — features: %s", experiment, features)
        oof, summary, confusions = run_block(experiment, LEARNED, features, data, folds,
                                             eda, logger)
        write_block(experiment, oof, pd.concat([baseline_summary, summary],
                                               ignore_index=True), confusions, data,
                    {"features": features})

    logger.info("done. Next: %s", "python -m classification.select (needs both "
                "experiments)" if set(args.experiments) == set(FEATURE_SETS)
                else "run the remaining experiment, then classification.select")


if __name__ == "__main__":
    main()
