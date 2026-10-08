"""Weeks 5-6 — the cross-validation protocol every candidate goes through.

Tables in, tables out; no I/O. The rules it implements are recorded in
dataset.yaml (`severity_classification`) and were fixed before any model ran:

- **One fold assignment, shared by everyone.** Repeated stratified 5-fold, 10
  repeats, seeded from SEED, stratified on the 4-class target. It is built ONCE
  and handed to every candidate, both feature sets and both class schemes, so
  two candidates' scores differ only because the candidates differ — never
  because one was dealt easier folds. That is also what makes a PAIRED
  comparison between them legitimate.
- **Out-of-fold predictions are the raw evidence.** Within each repeat every
  patient is predicted exactly once, by a model that never saw them. Every
  score in the report is computed from these, so any number can be recomputed
  from the saved predictions without refitting anything.
- **Scores are pooled per repeat.** Per fold, QWK would rest on 12 patients;
  instead each repeat's 60 out-of-fold predictions are pooled and scored once.
  Ten repeats give ten values per candidate: mean +- SD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold

from classification.metrics import (confusion, per_group, quadratic_weighted_kappa,
                                    score_all)
from classification.models import PREDICTED_VOLUME
from classification.severity import assign_class, derive_features, merge_to_three_class

FEATURE_SOURCE = "loso_prediction"
TARGET_SOURCE = "reference"


@dataclass(frozen=True)
class SeverityData:
    """The 60 patients, aligned: features from our segmentation, target from the expert."""

    frame: pd.DataFrame            # FEATURE_NAMES + predicted_volume_ml, index subject_key
    site: np.ndarray
    reference_volume_ml: np.ndarray
    log_reference_volume: np.ndarray
    class4: np.ndarray
    class3: np.ndarray

    @property
    def keys(self) -> list:
        return self.frame.index.tolist()

    def target(self, scheme: str) -> np.ndarray:
        return {"4class": self.class4, "3class": self.class3}[scheme]


def build_dataset(features_table: pd.DataFrame, severity_config: dict, *,
                  feature_source: str = FEATURE_SOURCE,
                  expected_subjects: int | None = 60) -> SeverityData:
    """Join Week 4's table into one aligned dataset, refusing anything inconsistent.

    Target from the REFERENCE rows, features from the `feature_source` rows,
    matched by subject. The two must cover exactly the same patients; a patient
    with a target and no features (or the reverse) is an error, not a row to
    drop quietly (CLAUDE.md 4, no silent exclusions).
    """
    if severity_config["target_source"] != TARGET_SOURCE:
        raise ValueError("the target must come from the reference — see dataset.yaml")
    reference = features_table[features_table.source == TARGET_SOURCE].set_index("subject_key")
    source = features_table[features_table.source == feature_source].set_index("subject_key")
    if source.empty:
        raise ValueError(f"no {feature_source!r} rows in the feature table — run "
                         f"segmentation.run_loso_predictions then features.run_features")
    if reference.index.has_duplicates or source.index.has_duplicates:
        raise ValueError("a subject appears twice for one source")
    if set(reference.index) != set(source.index):
        only_ref = sorted(set(reference.index) - set(source.index))
        only_src = sorted(set(source.index) - set(reference.index))
        raise ValueError(f"reference and {feature_source} cover different patients: "
                         f"{len(only_ref)} only in reference {only_ref[:3]}, "
                         f"{len(only_src)} only in {feature_source} {only_src[:3]}")
    if expected_subjects is not None and len(reference) != expected_subjects:
        raise ValueError(f"expected {expected_subjects} patients, found {len(reference)}")

    order = sorted(reference.index)
    reference, source = reference.loc[order], source.loc[order]
    if not (reference["site"] == source["site"]).all():
        raise ValueError("site disagrees between the reference and feature rows")

    frame = derive_features(source)
    frame[PREDICTED_VOLUME] = source["total_lesion_volume_ml"].astype(float)
    if not np.allclose(np.log1p(frame[PREDICTED_VOLUME]), frame["log_total_volume"]):
        raise AssertionError("log_total_volume is not log1p of the predicted volume")

    volume = reference["total_lesion_volume_ml"].to_numpy(dtype=float)
    class4 = assign_class(volume, severity_config["cutoffs_ml"])
    class3 = assign_class(volume, severity_config["sensitivity_cutoffs_ml"])
    if not np.array_equal(class3, merge_to_three_class(class4)):
        raise AssertionError("the 3-class cut-offs in dataset.yaml no longer merge "
                             "normal+mild — the sensitivity scheme has drifted")
    return SeverityData(frame=frame, site=reference["site"].to_numpy(),
                        reference_volume_ml=volume, log_reference_volume=np.log1p(volume),
                        class4=class4, class3=class3)


def make_folds(class4, n_splits: int, n_repeats: int, seed: int) -> list:
    """[(repeat, fold, train_index, test_index), ...] — built once, shared by all."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                       random_state=seed)
    folds = []
    for i, (train, test) in enumerate(splitter.split(np.zeros(len(class4)), class4)):
        folds.append((i // n_splits, i % n_splits, train, test))
    # Each repeat must predict every patient exactly once.
    for repeat in range(n_repeats):
        tests = np.concatenate([t for r, _, _, t in folds if r == repeat])
        if not np.array_equal(np.sort(tests), np.arange(len(class4))):
            raise AssertionError(f"repeat {repeat} does not cover every patient once")
    return folds


def folds_table(folds, keys) -> pd.DataFrame:
    """The fold assignment as a saveable table: which fold each patient was tested in."""
    rows = [{"subject_key": keys[i], "repeat": repeat, "fold": fold}
            for repeat, fold, _, test in folds for i in test]
    return pd.DataFrame(rows).sort_values(["repeat", "subject_key"]).reset_index(drop=True)


def cross_validate(factory, data: SeverityData, scheme: str, folds) -> pd.DataFrame:
    """Out-of-fold predictions for one candidate: one row per patient per repeat.

    `factory()` returns a fresh, unfitted model each fold — nothing carries over
    between folds. The model is fitted on the training rows only; the test rows
    are passed to `predict` and nothing else.
    """
    y = data.target(scheme)
    rows = []
    for repeat, fold, train, test in folds:
        model = factory()
        model.fit(data.frame.iloc[train], y[train], data.log_reference_volume[train])
        predicted = np.asarray(model.predict(data.frame.iloc[test]))
        if predicted.shape != (len(test),):
            raise AssertionError(f"{model.name}: {predicted.shape} predictions for "
                                 f"{len(test)} patients")
        for position, prediction in zip(test, predicted):
            rows.append({"subject_key": data.keys[position], "site": data.site[position],
                         "repeat": repeat, "fold": fold, "y_true": int(y[position]),
                         "y_pred": int(prediction), "tuned_value": model.chosen_,
                         "converged": bool(model.converged_)})
    return pd.DataFrame(rows)


def repeat_scores(oof: pd.DataFrame, n_classes: int, subset=None) -> pd.DataFrame:
    """Every scalar score, pooled within each repeat. One row per repeat.

    `subset` (a set of subject keys) restricts scoring to those patients — used
    for the boundary patients.
    """
    if subset is not None:
        oof = oof[oof.subject_key.isin(subset)]
    rows = []
    for repeat, part in oof.groupby("repeat", sort=True):
        rows.append({"repeat": repeat,
                     **score_all(part.y_true.to_numpy(), part.y_pred.to_numpy(), n_classes)})
    return pd.DataFrame(rows).set_index("repeat")


def site_scores(oof: pd.DataFrame, n_classes: int) -> pd.DataFrame:
    """Per-site QWK within each repeat. Rows = repeat, columns = site."""
    rows = {}
    for repeat, part in oof.groupby("repeat", sort=True):
        rows[repeat] = per_group(quadratic_weighted_kappa, part.y_true.to_numpy(),
                                 part.y_pred.to_numpy(), part.site.to_numpy(), n_classes)
    return pd.DataFrame.from_dict(rows, orient="index")


def summarise(oof: pd.DataFrame, n_classes: int, boundary_keys=frozenset()) -> dict:
    """One candidate's headline row: mean and SD over repeats of every score."""
    scores = repeat_scores(oof, n_classes)
    summary = {}
    for column in scores.columns:
        summary[f"{column}_mean"] = float(scores[column].mean())
        summary[f"{column}_sd"] = float(scores[column].std(ddof=1)) if len(scores) > 1 else 0.0
    for site, values in site_scores(oof, n_classes).items():
        summary[f"qwk_{site}_mean"] = float(values.mean())
    if boundary_keys:
        inside = repeat_scores(oof, n_classes, subset=set(boundary_keys))
        outside = repeat_scores(oof, n_classes,
                                subset=set(oof.subject_key) - set(boundary_keys))
        summary["boundary_n"] = len(set(boundary_keys) & set(oof.subject_key))
        summary["boundary_accuracy_mean"] = float(inside["accuracy"].mean())
        summary["boundary_mace_mean"] = float(inside["mean_absolute_class_error"].mean())
        summary["non_boundary_accuracy_mean"] = float(outside["accuracy"].mean())
    summary["n_repeats"] = int(oof.repeat.nunique())
    summary["fits_not_converged"] = int((~oof.groupby(["repeat", "fold"]).converged.all()).sum())
    tuned = oof.drop_duplicates(["repeat", "fold"]).tuned_value.dropna()
    summary["tuned_values"] = (tuned.value_counts().sort_index().to_dict()
                               if not tuned.empty else {})
    return summary


def pooled_confusion(oof: pd.DataFrame, n_classes: int) -> np.ndarray:
    """Confusion counts summed over every repeat (each patient counted once per repeat)."""
    return confusion(oof.y_true.to_numpy(), oof.y_pred.to_numpy(), n_classes)


def permutation_importance_cv(factory, data: SeverityData, scheme: str, folds,
                              features, n_permutations: int, seed: int) -> pd.DataFrame:
    """Drop in pooled out-of-fold QWK when one feature is shuffled — on HELD-OUT patients.

    Model-agnostic, so the forest and the linear models are read the same way.
    Computed inside the cross-validation (each fold's model is scored on its own
    test patients with one column permuted among them) because importance
    measured on the training data rewards whatever the model memorised.

    Returns one row per (repeat, feature, permutation) with the QWK drop.
    """
    rng = np.random.default_rng(seed)
    y = data.target(scheme)
    n_classes = {"4class": 4, "3class": 3}[scheme]
    baseline = {}
    permuted = {}
    for repeat, fold, train, test in folds:
        model = factory()
        model.fit(data.frame.iloc[train], y[train], data.log_reference_volume[train])
        test_frame = data.frame.iloc[test]
        baseline.setdefault(repeat, {})
        for position, prediction in zip(test, model.predict(test_frame)):
            baseline[repeat][position] = int(prediction)
        for feature in features:
            for k in range(n_permutations):
                shuffled = test_frame.copy()
                shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
                if feature == "log_total_volume":
                    # Keep the threshold rule's view consistent with the feature.
                    shuffled[PREDICTED_VOLUME] = np.expm1(shuffled[feature])
                store = permuted.setdefault((repeat, feature, k), {})
                for position, prediction in zip(test, model.predict(shuffled)):
                    store[position] = int(prediction)

    rows = []
    for (repeat, feature, k), predictions in permuted.items():
        order = sorted(predictions)
        base = quadratic_weighted_kappa(y[order], np.array([baseline[repeat][i] for i in order]),
                                        n_classes)
        shuffled = quadratic_weighted_kappa(y[order],
                                            np.array([predictions[i] for i in order]),
                                            n_classes)
        rows.append({"repeat": repeat, "feature": feature, "permutation": k,
                     "qwk_drop": base - shuffled})
    return pd.DataFrame(rows)

