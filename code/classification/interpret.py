"""Weeks 5-6, Phases D3 / D4 — what the selected grader gets right, wrong, and why.

Run after `classification.select`. For the selected option and its closest
rivals (the threshold rule, and each tier's best and put-forward candidates):

- **where it errs** — confusion matrix pooled over repeats, per-class precision /
  recall / F1, and the signed error: does it call people sicker (over-grade) or
  healthier (under-grade) than they are?
- **for whom** — every score per hospital, and the 14 boundary patients (within
  one median measurement error of a cut-off) scored apart from the rest. A model
  that only fails on boundary patients is failing where no model can be sure.
- **on what** — permutation importance on held-out patients for every learned
  candidate, and standardised coefficients for the linear ones refit on all 60.
  Read clinically: does periventricular vs deep matter? Do the two features the
  EDA flagged as possible hospital proxies (`largest_lesion_share`,
  `small_lesion_fraction`) carry weight — and if so, is it severity or scanner?

And D4, the **3-class sensitivity table**: every candidate under the merged
scheme beside its 4-class result, the way Week 4 reported 5 / 15 mm beside
10 mm. Reported, never used for selection.

Everything is computed from the saved out-of-fold predictions except importance
and coefficients, which refit (deterministically, same folds, same seed).

    code/.venv/bin/python -m classification.interpret
"""

from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd

from classification.metrics import (per_class_precision_recall_f1, per_group,
                                    score_all)
from classification.models import LEARNED, build
from classification.protocol import (build_dataset, make_folds, permutation_importance_cv,
                                     pooled_confusion)
from classification.run_experiments import (EDA_SUMMARY, EXPERIMENTS_DIR, FEATURE_SETS,
                                            FEATURES_CSV, SCHEMES, boundary_patients)
from classification.select import SELECTION_JSON
from metadata.config import CODE_ROOT, SEED, SEVERITY_CONFIG
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "interpret"
GENERATING = f"code/classification/{SCRIPT_NAME}.py"
OUTPUTS_DIR = CODE_ROOT / "classification" / "outputs"
INTERPRETATION_DIR = OUTPUTS_DIR / "interpretation"
SETTINGS = SEVERITY_CONFIG["selection"]["interpretation"]


def split_id(candidate_id: str) -> tuple[str, str]:
    """'volume_only/ordinal_logistic' -> ('volume_only', 'ordinal_logistic');
    'threshold_rule' -> ('baseline', 'threshold_rule')."""
    if "/" in candidate_id:
        experiment, name = candidate_id.split("/")
        return experiment, name
    return "baseline", candidate_id


def features_of(candidate_id: str) -> list:
    experiment, _ = split_id(candidate_id)
    return FEATURE_SETS.get(experiment, SEVERITY_CONFIG["feature_set_volume_only"])


def candidates_to_interpret(selection: dict) -> list:
    """Selected, threshold rule, and each tier's best and put-forward candidate.

    A tier's best is taken from the experiment summaries, not only from the
    selection steps: when selection stops early, later tiers never appear in
    `steps`, yet "does the PATTERN matter, and is it the hospital?" can only be
    read from a tier-3 model. Found 2026-10-08 when selection stopped at tier 2
    and no pattern-feature model was interpreted.
    """
    ids = [selection["selected"], "threshold_rule"]
    for step in selection["steps"]:
        ids += [step["best"], step["put_forward"]]
    for experiment in ("volume_only", "full_features"):
        path = EXPERIMENTS_DIR / f"summary_{experiment}.csv"
        if path.exists():
            summary = pd.read_csv(path)
            learned = summary[(summary.scheme == "4class") & summary.candidate.isin(LEARNED)]
            ids.append(f"{experiment}/{learned.sort_values('qwk_mean').candidate.iloc[-1]}")
    return list(dict.fromkeys(ids))   # de-duplicated, order kept


def load_oof(candidate_id: str, scheme: str) -> pd.DataFrame:
    experiment, name = split_id(candidate_id)
    tag = "baselines" if experiment == "baseline" else experiment
    oof = pd.read_csv(EXPERIMENTS_DIR / f"oof_{tag}.csv")
    part = oof[(oof.candidate == name) & (oof.scheme == scheme)]
    if part.empty:
        raise ValueError(f"no out-of-fold predictions for {candidate_id} / {scheme}")
    return part


def error_profile(oof: pd.DataFrame, n_classes: int, classes: list, boundary: set) -> dict:
    """Where, for whom, and which way one candidate errs — from its OOF predictions."""
    y_true, y_pred = oof.y_true.to_numpy(), oof.y_pred.to_numpy()
    prf = per_class_precision_recall_f1(y_true, y_pred, n_classes)
    per_site = {}
    for site, part in oof.groupby("site"):
        per_site[site] = {name: float(np.mean([fn(p.y_true.to_numpy(), p.y_pred.to_numpy(),
                                                  n_classes)
                                               for _, p in part.groupby("repeat")]))
                          for name, fn in _per_site_scores().items()}
    in_boundary = oof.subject_key.isin(boundary)
    return {
        "confusion_pooled_over_repeats": pooled_confusion(oof, n_classes).tolist(),
        "classes": classes,
        "per_class": {cls: {k: _clean(prf[k][i]) for k in prf} for i, cls in enumerate(classes)},
        "pooled_scores": score_all(y_true, y_pred, n_classes),
        "per_site": per_site,
        "boundary": {"n_patients": int(oof[in_boundary].subject_key.nunique()),
                     "accuracy": float(np.mean(y_true[in_boundary] == y_pred[in_boundary]))
                     if in_boundary.any() else None,
                     "non_boundary_accuracy": float(np.mean(y_true[~in_boundary]
                                                            == y_pred[~in_boundary]))},
        "per_patient_error_rate": (oof.assign(wrong=y_true != y_pred)
                                   .groupby("subject_key").wrong.mean()
                                   .sort_values(ascending=False).head(15).to_dict()),
    }


def _per_site_scores():
    from classification.metrics import SCORES
    return {k: SCORES[k] for k in ("qwk", "mean_absolute_class_error", "accuracy",
                                   "mean_signed_class_error")}


def _clean(value):
    value = float(value)
    return None if not np.isfinite(value) else value


def coefficients(candidate_id: str, data) -> pd.DataFrame:
    """Standardised coefficients of a linear candidate refit on all 60 (descriptive only)."""
    experiment, name = split_id(candidate_id)
    if name not in ("regression_then_threshold", "ordinal_logistic", "multinomial_logistic"):
        return pd.DataFrame()
    features = features_of(candidate_id)
    cutoffs, classes = SCHEMES["4class"]
    model = build(name, features, len(classes), cutoffs).fit(
        data.frame, data.class4, data.log_reference_volume)
    estimator = model.pipeline_.named_steps["model"]
    coef = np.atleast_2d(estimator.coef_)
    rows = []
    for r, row in enumerate(coef):
        target = ("log volume" if name == "regression_then_threshold" else
                  "latent severity" if name == "ordinal_logistic" else
                  classes[int(estimator.classes_[r])])
        for feature, value in zip(features, row):
            rows.append({"candidate": candidate_id, "equation": target, "feature": feature,
                         "standardised_coefficient": float(value),
                         "tuned_value": model.chosen_})
    return pd.DataFrame(rows)


def main() -> None:
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    INTERPRETATION_DIR.mkdir(parents=True, exist_ok=True)
    if not SELECTION_JSON.exists():
        raise SystemExit("selection.json missing — run classification.select first")
    selection = json.loads(SELECTION_JSON.read_text())
    eda = json.loads(EDA_SUMMARY.read_text())
    data = build_dataset(pd.read_csv(FEATURES_CSV), SEVERITY_CONFIG)
    folds = make_folds(data.class4, SEVERITY_CONFIG["cv_folds"],
                       SEVERITY_CONFIG["cv_repeats"], SEED)
    ids = candidates_to_interpret(selection)
    logger.info("selected: %s; interpreting %s", selection["selected"], ids)

    # --- D3: error profiles -------------------------------------------------
    profiles = {}
    for scheme in ("4class", "3class"):
        _, classes = SCHEMES[scheme]
        boundary = boundary_patients(data, scheme, eda)
        for cid in ids:
            profile = error_profile(load_oof(cid, scheme), len(classes), classes, boundary)
            profiles[f"{scheme}/{cid}"] = profile
            if scheme == "4class":
                logger.info("%s: pooled QWK %.3f, signed error %+.3f, boundary accuracy "
                            "%.2f vs %.2f elsewhere", cid, profile["pooled_scores"]["qwk"],
                            profile["pooled_scores"]["mean_signed_class_error"],
                            profile["boundary"]["accuracy"],
                            profile["boundary"]["non_boundary_accuracy"])
                logger.info("    recall by class: %s", {c: (round(v["recall"], 2)
                                                            if v["recall"] is not None else None)
                                                        for c, v in profile["per_class"].items()})
    path = INTERPRETATION_DIR / "error_profiles.json"
    path.write_text(json.dumps(profiles, indent=2, default=float))
    write_manifest(path, generating_script=GENERATING)

    # --- D3: what drives the decisions --------------------------------------
    coefficient_tables, importance_tables = [], []
    cutoffs, classes = SCHEMES["4class"]
    for cid in ids:
        experiment, name = split_id(cid)
        if name not in LEARNED:
            continue
        coefficient_tables.append(coefficients(cid, data))
        features = features_of(cid)
        factory = partial(build, name, features, len(classes), cutoffs)
        table = permutation_importance_cv(factory, data, "4class", folds, features,
                                          SETTINGS["permutation_repeats"], SEED)
        table.insert(0, "candidate", cid)
        importance_tables.append(table)
        summary = table.groupby("feature").qwk_drop.agg(["mean", "std"])
        logger.info("%s permutation importance (QWK drop on held-out patients):\n%s",
                    cid, summary.sort_values("mean", ascending=False).round(4).to_string())

    for name, tables in (("coefficients.csv", coefficient_tables),
                         ("permutation_importance.csv", importance_tables)):
        path = INTERPRETATION_DIR / name
        frame = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
        frame.to_csv(path, index=False)
        write_manifest(path, generating_script=GENERATING,
                       extra={"permutation_repeats": SETTINGS["permutation_repeats"],
                              "seed": SEED})

    # --- sensitivity of the CHOICE: every learned candidate vs the threshold rule
    # The rule puts forward the most interpretable of a tier's tied candidates,
    # not its top scorer. This asks whether that tie-break decided the outcome:
    # each candidate is put through both promotion conditions directly against
    # the threshold rule. Reported only; selection.json is unchanged by it.
    from classification.select import load_tier_predictions, promotion
    conditions = SEVERITY_CONFIG["selection"]["promotion_conditions"]
    threshold, tiers = load_tier_predictions(scheme="4class")
    rows = []
    for tier, members in tiers.items():
        for cid, p in members.items():
            r = promotion(p, threshold, len(SCHEMES["4class"][1]),
                          conditions["bootstrap_resamples"], SEED,
                          float(conditions["per_site_tie_tolerance"]))
            rows.append({"candidate": cid, "tier": tier,
                         "qwk_difference": r["qwk_difference"],
                         "bootstrap_se": r["bootstrap_se"],
                         "gain_in_se": r["qwk_difference"] / r["bootstrap_se"],
                         "overall_condition": r["overall_condition"],
                         "sites_won_or_tied": sum(r["wins_or_ties"].values()),
                         "would_be_promoted": r["promoted"]})
    head_to_head = pd.DataFrame(rows).sort_values("qwk_difference", ascending=False)
    path = INTERPRETATION_DIR / "every_candidate_vs_threshold.csv"
    head_to_head.to_csv(path, index=False)
    write_manifest(path, generating_script=GENERATING,
                   extra={"note": "sensitivity of the selection to its within-tier "
                                  "tie-break; does not change selection.json"})
    logger.info("every candidate head-to-head with the threshold rule:\n%s",
                head_to_head.round(4).to_string(index=False))

    # --- D4: the 3-class sensitivity table -----------------------------------
    summaries = []
    for tag in ("volume_only", "full_features"):
        path = EXPERIMENTS_DIR / f"summary_{tag}.csv"
        if path.exists():
            summaries.append(pd.read_csv(path))
    combined = pd.concat(summaries).drop_duplicates(["experiment", "scheme", "candidate"])
    wide = combined.pivot_table(index=["experiment", "candidate"], columns="scheme",
                                values=["qwk_mean", "mean_absolute_class_error_mean",
                                        "balanced_accuracy_mean"])
    wide.columns = [f"{metric}_{scheme}" for metric, scheme in wide.columns]
    path = INTERPRETATION_DIR / "sensitivity_3class.csv"
    wide.to_csv(path)
    write_manifest(path, generating_script=GENERATING,
                   extra={"note": "3-class = normal+mild merged. Reported beside the "
                                  "4-class primary; never used for selection."})
    logger.info("3-class sensitivity (QWK):\n%s",
                wide[["qwk_mean_4class", "qwk_mean_3class"]].round(3).to_string())


if __name__ == "__main__":
    main()
