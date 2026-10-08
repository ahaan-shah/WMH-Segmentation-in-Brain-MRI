"""Weeks 5-6 — how a severity grading is scored.

Arrays in, numbers out; no I/O. Checked against scikit-learn and hand-worked
cases in `checks/test_classification.py`.

**Why order-aware scores come first.** The classes are ordered — normal < mild
< moderate < severe — and the errors are not equal: calling a severe patient
normal is clinically far worse than calling them moderate. Plain accuracy scores
both mistakes identically. So the primary scores are:

- **Quadratic weighted kappa (QWK)** — agreement beyond chance, with an error
  penalised by the SQUARE of how many classes it is off. 1 is perfect, 0 is what
  guessing from the class frequencies achieves. The standard agreement measure
  for ordinal clinical grades, Fazekas included.
- **Mean absolute class error** — on average, how many classes off a prediction
  is. 0 is perfect; directly readable ("a third of a class").
- **Balanced accuracy** — the mean of per-class recall, so the 8 mild and 9
  moderate patients count as much as the 25 severe ones.

**Mean signed class error** says which way the errors lean: positive means the
model over-grades, negative under-grades. A model can have a fine kappa and
still systematically call people sicker than they are, which matters
clinically, so it is reported for every candidate.

Every function takes `n_classes` explicitly rather than inferring it from the
labels present. Twelve patients in a fold, or one hospital's twenty, will often
be missing a class, and an inferred class count would silently change the
weighting between subsets that are supposed to be compared.
"""

from __future__ import annotations

import numpy as np


def _validate(y_true, y_pred, n_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.ndim != 1:
        raise ValueError(f"y_true {y_true.shape} and y_pred {y_pred.shape} must be "
                         f"matching 1-D arrays")
    if y_true.size == 0:
        raise ValueError("no predictions to score")
    for name, y in (("y_true", y_true), ("y_pred", y_pred)):
        if not np.issubdtype(y.dtype, np.integer):
            raise TypeError(f"{name} must hold integer class indices, got {y.dtype}")
        if y.min() < 0 or y.max() >= n_classes:
            raise ValueError(f"{name} has a class outside 0..{n_classes - 1}")
    return y_true, y_pred


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Counts, rows = true class, columns = predicted class."""
    y_true, y_pred = _validate(y_true, y_pred, n_classes)
    return np.bincount(y_true * n_classes + y_pred,
                       minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def quadratic_weighted_kappa(y_true, y_pred, n_classes: int) -> float:
    """Cohen's kappa with quadratic disagreement weights (i - j)^2.

    NaN when chance disagreement is zero — both raters put everyone in one and
    the same class — where kappa is undefined. That is returned rather than
    papered over as 0 or 1; callers decide what an undefined score means.
    """
    observed = confusion(y_true, y_pred, n_classes).astype(float)
    total = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    index = np.arange(n_classes)
    weights = (index[:, None] - index[None, :]) ** 2
    denominator = (weights * expected).sum()
    if denominator == 0:
        return float("nan")
    return float(1.0 - (weights * observed).sum() / denominator)


def mean_absolute_class_error(y_true, y_pred, n_classes: int) -> float:
    y_true, y_pred = _validate(y_true, y_pred, n_classes)
    return float(np.mean(np.abs(y_pred - y_true)))


def mean_signed_class_error(y_true, y_pred, n_classes: int) -> float:
    """> 0 over-grades (calls patients sicker than they are), < 0 under-grades."""
    y_true, y_pred = _validate(y_true, y_pred, n_classes)
    return float(np.mean(y_pred - y_true))


def accuracy(y_true, y_pred, n_classes: int) -> float:
    y_true, y_pred = _validate(y_true, y_pred, n_classes)
    return float(np.mean(y_true == y_pred))


def balanced_accuracy(y_true, y_pred, n_classes: int) -> float:
    """Mean recall over the classes PRESENT in y_true.

    A class with no true patients has no recall to average, so it is left out
    rather than counted as 0 — matching scikit-learn's balanced_accuracy_score.
    """
    matrix = confusion(y_true, y_pred, n_classes)
    support = matrix.sum(axis=1)
    present = support > 0
    return float(np.mean(np.diag(matrix)[present] / support[present]))


def per_class_precision_recall_f1(y_true, y_pred, n_classes: int) -> dict:
    """Per class: precision, recall, F1 and support. NaN where undefined (0/0)."""
    matrix = confusion(y_true, y_pred, n_classes).astype(float)
    true_positive = np.diag(matrix)
    predicted = matrix.sum(axis=0)
    support = matrix.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, true_positive / predicted, np.nan)
        recall = np.where(support > 0, true_positive / support, np.nan)
        f1 = np.where(precision + recall > 0,
                      2 * precision * recall / (precision + recall), np.nan)
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": support.astype(int)}


SCORES = {
    "qwk": quadratic_weighted_kappa,
    "mean_absolute_class_error": mean_absolute_class_error,
    "balanced_accuracy": balanced_accuracy,
    "accuracy": accuracy,
    "mean_signed_class_error": mean_signed_class_error,
}


def score_all(y_true, y_pred, n_classes: int) -> dict:
    """Every scalar score at once, in a fixed order."""
    return {name: fn(y_true, y_pred, n_classes) for name, fn in SCORES.items()}


def per_group(score, y_true, y_pred, groups, n_classes: int) -> dict:
    """`score` computed separately within each group (e.g. each hospital)."""
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    return {group: score(y_true[groups == group], y_pred[groups == group], n_classes)
            for group in sorted(set(groups.tolist()))}
