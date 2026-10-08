"""Weeks 5-6 — the candidate severity graders, behind one interface.

Every candidate does the same two things:

    model.fit(frame, y, log_reference_volume)    # y: class index per patient
    model.predict(frame) -> class index per patient

`frame` holds the six classifier features (`severity.FEATURE_NAMES`) plus
`predicted_volume_ml`, all measured from OUR leave-one-site-out segmentation.
`log_reference_volume` is log1p of the EXPERT's volume — the quantity the class
was derived from — and only `regression_then_threshold` reads it, because it is
the one candidate that learns the continuous value rather than the class.

The candidates, simplest first (settings from dataset.yaml
`severity_classification.model_settings`, never literals here):

- **majority_class** — always says the commonest class. The floor; any method
  that cannot beat it has learned nothing. Never selectable.
- **threshold_rule** — Joo et al.'s cut-offs applied straight to our predicted
  volume. No learning at all: the "rule-based thresholds" route the problem
  statement names. The incumbent every model must beat.
- **regression_then_threshold** — ridge regression predicts log volume, then the
  same cut-offs are applied to the prediction. Learns from the continuous
  number, so the thin classes (8 mild, 9 moderate) cannot starve it.
- **ordinal_logistic** — proportional-odds model: one direction through feature
  space along which normal < mild < moderate < severe, with three cut-points.
  Built for exactly this kind of ordered target.
- **multinomial_logistic** — treats the four classes as unordered. The control:
  if it matches the ordinal model, the ordering is not adding anything.
- **random_forest** — the only non-linear candidate. Tests whether combinations
  of features matter. Deliberately untuned (see dataset.yaml).

**Leakage.** Every learned candidate is a scikit-learn Pipeline with the scaler
inside it, so standardisation is fitted on the training fold alone. Tuning is
nested: a tuned candidate runs its own inner cross-validation on the training
fold it was given and never sees the outer test fold. Both are tested.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from classification.metrics import quadratic_weighted_kappa
from classification.severity import FEATURE_NAMES, assign_class
from metadata.config import SEED, SEVERITY_CONFIG

PREDICTED_VOLUME = "predicted_volume_ml"
SETTINGS = SEVERITY_CONFIG["model_settings"]

BASELINES = ("majority_class", "threshold_rule")
LEARNED = ("regression_then_threshold", "ordinal_logistic",
           "multinomial_logistic", "random_forest")
# The order dataset.yaml's within-tier tie-break puts candidates forward in.
INTERPRETABILITY_ORDER = LEARNED

# Which direction of each tuned parameter is "more regularised", for the
# tie-break recorded in dataset.yaml (`tuning_tiebreak`).
MORE_REGULARISED_IS_LARGER = {"alpha": True, "C": False}


# ---------------------------------------------------------------------------
# the ordinal model, as a scikit-learn estimator so it can sit in a Pipeline
# ---------------------------------------------------------------------------
class OrderedLogit(BaseEstimator, ClassifierMixin):
    """statsmodels' OrderedModel (proportional odds, logit link) in sklearn clothing.

    Records whether the optimiser converged rather than assuming it did.
    statsmodels warns on non-convergence and carries on; here the flag is kept
    on the fitted object and counted by the experiment driver, and non-finite
    coefficients raise.
    """

    def __init__(self, distribution="logit", method="bfgs", max_iterations=1000):
        self.distribution = distribution
        self.method = method
        self.max_iterations = max_iterations

    def fit(self, X, y):
        from statsmodels.miscmodels.ordinal_model import OrderedModel

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size < 2:
            raise ValueError("ordinal model needs at least two classes in the training fold")
        # Re-index to 0..k-1 so a class absent from this fold cannot leave a gap
        # statsmodels would treat as an empty category.
        codes = np.searchsorted(self.classes_, y)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = OrderedModel(codes, X, distr=self.distribution).fit(
                method=self.method, maxiter=self.max_iterations, disp=False)
        if not np.all(np.isfinite(result.params)):
            raise FloatingPointError("ordinal model produced non-finite coefficients "
                                     "(complete separation?)")
        self.result_ = result
        self.converged_ = bool(result.mle_retvals.get("converged", False))
        self.fit_warnings_ = sorted({type(w.message).__name__ for w in caught})
        self.n_features_in_ = X.shape[1]
        return self

    @property
    def coef_(self) -> np.ndarray:
        """One coefficient per feature (the cut-points follow them in params)."""
        return np.asarray(self.result_.params[: self.n_features_in_])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        return np.asarray(self.result_.model.predict(self.result_.params, exog=X))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# the candidates
# ---------------------------------------------------------------------------
class SeverityModel:
    """Base: a name, the columns it reads, and the fitted-state bookkeeping."""

    name = ""
    tuned_parameter: str | None = None

    def __init__(self, features, n_classes, cutoffs):
        self.features = list(features)
        self.n_classes = int(n_classes)
        self.cutoffs = list(cutoffs)
        self.chosen_ = None          # tuned value, if any, after fit
        self.converged_ = True       # only the ordinal model can make this False

    def fit(self, frame: pd.DataFrame, y, log_reference_volume):
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def _X(self, frame):
        missing = set(self.features) - set(frame.columns)
        if missing:
            raise KeyError(f"{self.name}: frame is missing {sorted(missing)}")
        return frame[self.features].to_numpy(dtype=float)


class MajorityClass(SeverityModel):
    name = "majority_class"

    def fit(self, frame, y, log_reference_volume=None):
        # bincount().argmax() takes the LOWER class on a tie — deterministic.
        self.majority_ = int(np.bincount(np.asarray(y), minlength=self.n_classes).argmax())
        return self

    def predict(self, frame):
        return np.full(len(frame), self.majority_, dtype=int)


class ThresholdRule(SeverityModel):
    """The cut-offs on our measured volume. Nothing is fitted.

    Reads the volume in mL directly rather than undoing the log feature: a
    volume sitting exactly on a cut-off must land in the same class it does in
    `assign_class`, and expm1(log1p(v)) is not always exactly v.
    """

    name = "threshold_rule"

    def fit(self, frame, y=None, log_reference_volume=None):
        return self

    def predict(self, frame):
        return assign_class(frame[PREDICTED_VOLUME].to_numpy(dtype=float), self.cutoffs)


class RegressionThenThreshold(SeverityModel):
    name = "regression_then_threshold"
    tuned_parameter = "alpha"

    def fit(self, frame, y, log_reference_volume, *, alpha=None):
        alpha = SETTINGS[self.name]["tune"]["alpha"][0] if alpha is None else alpha
        self.pipeline_ = Pipeline([("scale", StandardScaler()),
                                   ("model", Ridge(alpha=alpha))])
        self.pipeline_.fit(self._X(frame), np.asarray(log_reference_volume, dtype=float))
        self.chosen_ = alpha
        return self

    def predict(self, frame):
        log_volume = self.pipeline_.predict(self._X(frame))
        # A predicted log volume below 0 means a volume below 0 mL, which is
        # "less than nothing" — the lowest class. Clipped at 0 rather than
        # passed to assign_class, which rightly refuses negative volumes.
        return assign_class(np.clip(np.expm1(log_volume), 0.0, None), self.cutoffs)


class OrdinalLogistic(SeverityModel):
    name = "ordinal_logistic"

    def fit(self, frame, y, log_reference_volume=None):
        settings = SETTINGS[self.name]
        self.pipeline_ = Pipeline([
            ("scale", StandardScaler()),
            ("model", OrderedLogit(distribution=settings["distribution"],
                                   method=settings["fit_method"],
                                   max_iterations=settings["max_iterations"]))])
        self.pipeline_.fit(self._X(frame), np.asarray(y))
        self.converged_ = self.pipeline_.named_steps["model"].converged_
        return self

    def predict(self, frame):
        return self.pipeline_.predict(self._X(frame)).astype(int)


class MultinomialLogistic(SeverityModel):
    name = "multinomial_logistic"
    tuned_parameter = "C"

    def fit(self, frame, y, log_reference_volume=None, *, C=None):
        settings = SETTINGS[self.name]
        C = settings["tune"]["C"][0] if C is None else C
        self.pipeline_ = Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=C, class_weight=settings["class_weight"],
                                         max_iter=10_000))])
        self.pipeline_.fit(self._X(frame), np.asarray(y))
        self.chosen_ = C
        return self

    def predict(self, frame):
        return self.pipeline_.predict(self._X(frame)).astype(int)


class RandomForest(SeverityModel):
    name = "random_forest"

    def fit(self, frame, y, log_reference_volume=None):
        settings = SETTINGS[self.name]
        # The scaler is a no-op for a forest (splits are rank-based); it is kept
        # so every learned candidate has the same Pipeline shape.
        self.pipeline_ = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=settings["n_estimators"],
                min_samples_leaf=settings["min_samples_leaf"],
                max_features=settings["max_features"],
                class_weight=settings["class_weight"],
                # n_jobs=1, deliberately: measured 2026-10-08, the same seed gave
                # DIFFERENT probabilities at n_jobs=-1 (parallel summation order),
                # and one thread was also faster at n=48 (1.1 s vs 1.7 s).
                random_state=SEED, n_jobs=1))])
        self.pipeline_.fit(self._X(frame), np.asarray(y))
        return self

    def predict(self, frame):
        return self.pipeline_.predict(self._X(frame)).astype(int)


# ---------------------------------------------------------------------------
# nested tuning
# ---------------------------------------------------------------------------
class Tuned(SeverityModel):
    """Wraps a candidate with a `tune:` grid in its own inner cross-validation.

    Given only the outer TRAINING fold, it splits that again (stratified, SEED),
    predicts every inner held-out patient once per grid value, pools those
    predictions and scores one QWK per value — the same pooled unit the outer
    selection uses. The winning value is refit on the whole training fold.
    Ties go to the more regularised value (dataset.yaml `tuning_tiebreak`).
    """

    def __init__(self, inner: SeverityModel, grid, inner_folds: int):
        super().__init__(inner.features, inner.n_classes, inner.cutoffs)
        self.inner = inner
        self.name = inner.name
        self.tuned_parameter = inner.tuned_parameter
        self.grid = list(grid)
        self.inner_folds = int(inner_folds)

    def _fresh(self):
        return type(self.inner)(self.features, self.n_classes, self.cutoffs)

    def fit(self, frame, y, log_reference_volume):
        y = np.asarray(y)
        log_reference_volume = np.asarray(log_reference_volume, dtype=float)
        splitter = StratifiedKFold(self.inner_folds, shuffle=True, random_state=SEED)
        with warnings.catch_warnings():
            # "least populated class has only N members" — expected with 6-7
            # mild patients in a training fold; the split is still valid.
            warnings.filterwarnings("ignore", message=".*least populated class.*")
            splits = list(splitter.split(np.zeros(len(y)), y))

        self.inner_scores_ = {}
        for value in self.grid:
            pooled = np.empty(len(y), dtype=int)
            for train, test in splits:
                model = self._fresh().fit(frame.iloc[train], y[train],
                                          log_reference_volume[train],
                                          **{self.tuned_parameter: value})
                pooled[test] = model.predict(frame.iloc[test])
            self.inner_scores_[value] = quadratic_weighted_kappa(y, pooled, self.n_classes)

        self.chosen_ = choose_tuned_value(self.inner_scores_, self.tuned_parameter)
        self.model_ = self._fresh().fit(frame, y, log_reference_volume,
                                        **{self.tuned_parameter: self.chosen_})
        return self

    def predict(self, frame):
        return self.model_.predict(frame)

    @property
    def pipeline_(self):
        return self.model_.pipeline_


def choose_tuned_value(scores: dict, parameter: str):
    """Best inner QWK; ties to the most regularised value. NaN never wins."""
    finite = {value: score for value, score in scores.items() if np.isfinite(score)}
    if not finite:
        raise ValueError(f"every {parameter} value scored NaN in the inner loop")
    best = max(finite.values())
    tied = [value for value, score in finite.items() if score == best]
    return max(tied) if MORE_REGULARISED_IS_LARGER[parameter] else min(tied)


# ---------------------------------------------------------------------------
CLASSES = {cls.name: cls for cls in (MajorityClass, ThresholdRule, RegressionThenThreshold,
                                     OrdinalLogistic, MultinomialLogistic, RandomForest)}


def build(name: str, features, n_classes: int, cutoffs) -> SeverityModel:
    """A fresh, unfitted candidate — wrapped for nested tuning if it has a grid."""
    if name not in CLASSES:
        raise KeyError(f"unknown candidate {name!r}; known: {sorted(CLASSES)}")
    unknown = set(features) - set(FEATURE_NAMES)
    if unknown:
        raise KeyError(f"not classifier features: {sorted(unknown)}")
    model = CLASSES[name](features, n_classes, cutoffs)
    tune = SETTINGS.get(name, {}).get("tune", "none")
    if isinstance(tune, dict):
        (parameter, grid), = tune.items()
        if parameter != model.tuned_parameter:
            raise ValueError(f"dataset.yaml tunes {parameter!r} for {name}, but the "
                             f"model exposes {model.tuned_parameter!r}")
        return Tuned(model, grid, SEVERITY_CONFIG["inner_cv_folds"])
    return model
