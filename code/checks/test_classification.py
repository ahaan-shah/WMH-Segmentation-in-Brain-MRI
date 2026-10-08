"""Tests for the Weeks 5-6 classification code — known inputs, known answers.

Nothing here opens real patient data or the sealed test set. Every test builds
a synthetic table, a synthetic volume or a synthetic set of predictions whose
right answer is known in advance. What they pin down:

- the metrics agree with scikit-learn and with hand-worked cases;
- the threshold rule IS assign_class, including a volume exactly on a cut-off;
- nothing a model is fitted on ever includes the patients it is tested on —
  not the scaler, not the inner tuning loop;
- every candidate is dealt the same folds;
- the selection rule promotes, refuses and stops exactly as dataset.yaml says;
- the leave-one-site-out driver refuses a network that saw the hospital, and
  its inference path puts the prediction back on the right grid;
- Week 7's test-set features cannot overwrite the training table.
"""

from functools import partial

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score,
                             precision_recall_fscore_support)

from classification import metrics
from classification.models import (LEARNED, PREDICTED_VOLUME, MajorityClass,
                                   OrderedLogit, ThresholdRule, Tuned, build,
                                   choose_tuned_value)
from classification.protocol import (build_dataset, cross_validate, folds_table,
                                     make_folds, permutation_importance_cv, summarise)
from classification.select import Predictions, apply_rule, paired_bootstrap_se, put_forward
from classification.severity import FEATURE_NAMES, assign_class, merge_to_three_class, near_cutoff
from metadata.config import SEED, SEVERITY_CONFIG

CUTOFFS = SEVERITY_CONFIG["cutoffs_ml"]
SITES = ("Amsterdam", "Singapore", "Utrecht")


# ---------------------------------------------------------------------------
# synthetic Week 4 table
# ---------------------------------------------------------------------------
def synthetic_features(n=60, seed=0, noise=0.2, feature_source="loso_prediction"):
    """A features.csv lookalike: reference rows + one prediction source.

    Volumes log-normal around the cohort's; the prediction is the reference
    times a log-normal error, so volume carries the signal and the pattern
    features are pure noise — which is the situation the EDA found.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        site = SITES[i % 3]
        volume = float(np.exp(rng.normal(2.2, 1.2)))
        for source, v in (("reference", volume),
                          (feature_source, volume * float(np.exp(rng.normal(0, noise))))):
            count = int(rng.integers(5, 80))
            rows.append({"subject_key": f"training_{site}_{site}_{i:03d}", "split": "training",
                         "site": site, "source": source, "total_lesion_volume_ml": v,
                         "periventricular_fraction": rng.uniform(0.3, 1.0),
                         "laterality_index": rng.uniform(-0.3, 0.3),
                         "lesion_count_26conn": count,
                         "small_lesion_count_le5vox": int(count * rng.uniform(0.2, 0.6)),
                         "largest_lesion_volume_ml": v * rng.uniform(0.1, 0.9)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def data():
    return build_dataset(synthetic_features(), SEVERITY_CONFIG)


@pytest.fixture(scope="module")
def folds(data):
    return make_folds(data.class4, 5, 2, SEED)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", range(5))
def test_qwk_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 4, 60)
    y_pred = np.clip(y_true + rng.integers(-2, 3, 60), 0, 3)
    expected = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=range(4))
    assert metrics.quadratic_weighted_kappa(y_true, y_pred, 4) == pytest.approx(expected)


def test_qwk_hand_worked():
    # Two patients, true (0, 1), given (0, 0). Observed weighted disagreement:
    # one patient off by one -> sum w*O = 1. Marginals: true (1,1), given (2,0);
    # expected E = outer / 2 = [[1,0],[1,0]] -> sum w*E = 1. kappa = 1 - 1/1 = 0.
    assert metrics.quadratic_weighted_kappa([0, 1], [0, 0], 2) == pytest.approx(0.0)
    assert metrics.quadratic_weighted_kappa([0, 1, 2, 3], [0, 1, 2, 3], 4) == pytest.approx(1.0)
    # Fully reversed on a symmetric spread is maximal disagreement: -1.
    assert metrics.quadratic_weighted_kappa([0, 3], [3, 0], 4) == pytest.approx(-1.0)


def test_qwk_penalises_a_two_class_error_more_than_two_one_class_errors():
    truth = np.array([0, 1, 2, 3] * 5)
    one_off = truth.copy(); one_off[[1, 2]] += 1          # two errors of one class
    two_off = truth.copy(); two_off[1] += 2               # one error of two classes
    assert (metrics.quadratic_weighted_kappa(truth, one_off, 4)
            > metrics.quadratic_weighted_kappa(truth, two_off, 4))


def test_qwk_undefined_when_everyone_is_in_one_class():
    assert np.isnan(metrics.quadratic_weighted_kappa([2, 2, 2], [2, 2, 2], 4))


def test_class_errors_and_accuracies():
    y_true = np.array([0, 1, 2, 3, 3, 3])
    y_pred = np.array([0, 2, 2, 1, 3, 3])
    assert metrics.mean_absolute_class_error(y_true, y_pred, 4) == pytest.approx(3 / 6)
    assert metrics.mean_signed_class_error(y_true, y_pred, 4) == pytest.approx((1 - 2) / 6)
    assert metrics.accuracy(y_true, y_pred, 4) == pytest.approx(4 / 6)
    assert metrics.balanced_accuracy(y_true, y_pred, 4) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred))


def test_balanced_accuracy_skips_absent_classes_like_sklearn():
    y_true, y_pred = np.array([0, 0, 3, 3]), np.array([0, 1, 3, 2])
    assert metrics.balanced_accuracy(y_true, y_pred, 4) == pytest.approx(
        balanced_accuracy_score(y_true, y_pred))


def test_per_class_prf_matches_sklearn():
    rng = np.random.default_rng(3)
    y_true, y_pred = rng.integers(0, 4, 50), rng.integers(0, 4, 50)
    ours = metrics.per_class_precision_recall_f1(y_true, y_pred, 4)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=range(4),
                                                 zero_division=np.nan)
    np.testing.assert_allclose(ours["precision"], p)
    np.testing.assert_allclose(ours["recall"], r)
    np.testing.assert_allclose(ours["f1"], f)
    np.testing.assert_array_equal(ours["support"], s)


@pytest.mark.parametrize("bad", [([0, 4], [0, 1]), ([0, 1], [0, -1]), ([0.0, 1.0], [0, 1]),
                                 ([0, 1], [0])])
def test_metrics_refuse_malformed_labels(bad):
    with pytest.raises((ValueError, TypeError)):
        metrics.quadratic_weighted_kappa(np.array(bad[0]), np.array(bad[1]), 4)


def test_per_group_scores_each_site_separately():
    out = metrics.per_group(metrics.accuracy, [0, 1, 2, 3], [0, 1, 0, 0],
                            ["A", "A", "B", "B"], 4)
    assert out == {"A": 1.0, "B": 0.0}


# ---------------------------------------------------------------------------
# targets, boundary patients
# ---------------------------------------------------------------------------
def test_merge_to_three_class_matches_the_sensitivity_cutoffs():
    volumes = np.array([0.5, 3.39, 3.4, 9.59, 9.6, 17.09, 17.1, 80.0])
    four = assign_class(volumes, CUTOFFS)
    assert np.array_equal(merge_to_three_class(four),
                          assign_class(volumes, SEVERITY_CONFIG["sensitivity_cutoffs_ml"]))


def test_near_cutoff_matches_the_eda_definition():
    band = 0.1
    volumes = [3.4 / 1.1, 3.4 / 1.1 - 0.01, 9.6 * 1.1, 9.6 * 1.1 + 0.01, 30.0]
    assert near_cutoff(volumes, CUTOFFS, band).tolist() == [True, False, True, False, False]


# ---------------------------------------------------------------------------
# the dataset
# ---------------------------------------------------------------------------
def test_dataset_aligns_target_and_features(data):
    assert len(data.keys) == 60 and data.keys == sorted(data.keys)
    assert set(data.frame.columns) == set(FEATURE_NAMES) | {PREDICTED_VOLUME}
    assert np.array_equal(data.class4, assign_class(data.reference_volume_ml, CUTOFFS))
    assert np.allclose(data.log_reference_volume, np.log1p(data.reference_volume_ml))


def test_dataset_takes_features_from_the_prediction_not_the_reference():
    table = synthetic_features()
    data = build_dataset(table, SEVERITY_CONFIG)
    loso = table[table.source == "loso_prediction"].set_index("subject_key")
    np.testing.assert_allclose(data.frame[PREDICTED_VOLUME],
                               loso.loc[data.keys, "total_lesion_volume_ml"])
    assert not np.allclose(data.frame[PREDICTED_VOLUME], data.reference_volume_ml)


def test_dataset_refuses_a_patient_without_features():
    table = synthetic_features()
    drop = table[(table.source == "loso_prediction")].index[0]
    with pytest.raises(ValueError, match="different patients"):
        build_dataset(table.drop(index=drop), SEVERITY_CONFIG)


def test_dataset_refuses_a_missing_source_and_a_wrong_count():
    table = synthetic_features()
    with pytest.raises(ValueError, match="no 'loso_prediction' rows"):
        build_dataset(table[table.source == "reference"], SEVERITY_CONFIG)
    with pytest.raises(ValueError, match="expected 60"):
        build_dataset(synthetic_features(n=30), SEVERITY_CONFIG)


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------
def test_threshold_rule_is_assign_class_including_exact_cutoffs():
    volumes = np.array([1.0, 3.4, 9.6, 17.1, np.nextafter(17.1, 0), 50.0])
    frame = pd.DataFrame({PREDICTED_VOLUME: volumes})
    rule = ThresholdRule([], 4, CUTOFFS).fit(frame)
    assert np.array_equal(rule.predict(frame), assign_class(volumes, CUTOFFS))
    assert rule.predict(frame).tolist() == [0, 1, 2, 3, 2, 3]


def test_majority_class_predicts_the_commonest_and_breaks_ties_low():
    frame = pd.DataFrame(index=range(5))
    assert MajorityClass([], 4, CUTOFFS).fit(frame, [3, 3, 1, 0, 3]).predict(frame).tolist() == [3] * 5
    assert MajorityClass([], 4, CUTOFFS).fit(frame.iloc[:4], [2, 2, 1, 1]).predict(frame)[0] == 1


@pytest.mark.parametrize("name", LEARNED)
def test_scaler_is_fitted_on_the_training_fold_only(name, data):
    features = list(FEATURE_NAMES)
    train = np.arange(0, 48)
    model = build(name, features, 4, CUTOFFS)
    model.fit(data.frame.iloc[train], data.class4[train], data.log_reference_volume[train])
    scaler = model.pipeline_.named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_,
                               data.frame.iloc[train][features].to_numpy().mean(axis=0))


class Spy:
    """Records exactly which rows it was fitted and asked to predict."""

    calls = []

    def __init__(self, *args, **kwargs):
        self.name, self.chosen_, self.converged_ = "spy", None, True

    def fit(self, frame, y, log_reference_volume, **kwargs):
        Spy.calls.append(("fit", set(frame.index), kwargs))
        return self

    def predict(self, frame):
        Spy.calls.append(("predict", set(frame.index), {}))
        return np.zeros(len(frame), dtype=int)


def test_cross_validation_never_fits_on_the_patients_it_tests(data, folds):
    Spy.calls = []
    oof = cross_validate(Spy, data, "4class", folds)
    fits = [c[1] for c in Spy.calls if c[0] == "fit"]
    predicts = [c[1] for c in Spy.calls if c[0] == "predict"]
    assert len(fits) == len(predicts) == len(folds)
    for fitted, tested in zip(fits, predicts):
        assert fitted.isdisjoint(tested)
        assert len(fitted | tested) == 60
    # every repeat predicts each patient exactly once
    assert (oof.groupby("repeat").subject_key.nunique() == 60).all()
    assert (oof.groupby(["repeat", "subject_key"]).size() == 1).all()


def test_inner_tuning_never_sees_the_outer_test_fold(data, folds):
    Spy.calls = []
    _, _, train, test = folds[0]
    spy = Spy()
    spy.features, spy.n_classes, spy.cutoffs, spy.tuned_parameter = [], 4, CUTOFFS, "C"
    tuned = Tuned(spy, [0.1, 1.0], inner_folds=3)
    tuned._fresh = Spy
    tuned.fit(data.frame.iloc[train], data.class4[train], data.log_reference_volume[train])
    outer_test = set(data.frame.index[test])
    seen = set().union(*(c[1] for c in Spy.calls))
    assert seen.isdisjoint(outer_test)
    assert seen == set(data.frame.index[train])


def test_tuning_ties_go_to_the_more_regularised_value():
    assert choose_tuned_value({0.1: 0.8, 1.0: 0.8, 10.0: 0.7}, "alpha") == 1.0
    assert choose_tuned_value({0.1: 0.8, 1.0: 0.8, 10.0: 0.7}, "C") == 0.1
    assert choose_tuned_value({0.1: np.nan, 1.0: 0.5}, "C") == 1.0
    with pytest.raises(ValueError):
        choose_tuned_value({0.1: np.nan}, "C")


def test_every_candidate_is_dealt_the_same_folds(data):
    a = folds_table(make_folds(data.class4, 5, 3, SEED), data.keys)
    b = folds_table(make_folds(data.class4, 5, 3, SEED), data.keys)
    assert a.equals(b)
    c = folds_table(make_folds(data.class4, 5, 3, SEED + 1), data.keys)
    assert not a.equals(c)


def test_folds_are_stratified_on_the_four_class_target(data):
    for _, _, _, test in make_folds(data.class4, 5, 2, SEED):
        counts = np.bincount(data.class4[test], minlength=4)
        overall = np.bincount(data.class4, minlength=4) / 5
        assert np.all(np.abs(counts - overall) <= 1)


def test_ordinal_model_recovers_an_ordered_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 1))
    y = np.digitize(x[:, 0] + 0.3 * rng.normal(size=200), [-0.8, 0, 0.8])
    model = OrderedLogit().fit(x, y)
    assert model.converged_ and model.coef_[0] > 0
    assert np.mean(model.predict(x) == y) > 0.7
    assert np.allclose(model.predict_proba(x[:5]).sum(axis=1), 1.0)


def test_ordinal_model_handles_a_class_missing_from_the_fold():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(60, 1))
    y = np.where(x[:, 0] > 0, 3, 0)                     # classes 1 and 2 absent
    y[np.argsort(np.abs(x[:, 0]))[:10]] = 1
    predicted = OrderedLogit().fit(x, y).predict(x)
    assert set(predicted) <= {0, 1, 3}


@pytest.mark.parametrize("name", ("majority_class", "threshold_rule") + LEARNED)
def test_every_candidate_runs_end_to_end_and_is_deterministic(name, data, folds):
    factory = partial(build, name, list(FEATURE_NAMES), 4, CUTOFFS)
    a = cross_validate(factory, data, "4class", folds)
    b = cross_validate(factory, data, "4class", folds)
    assert a.equals(b)
    summary = summarise(a, 4, set(data.keys[:5]))
    assert summary["n_repeats"] == 2 and summary["boundary_n"] == 5


def test_volume_signal_is_found(data, folds):
    # Volume carries the class here by construction; the threshold rule and the
    # learned volume models must find it, and the majority floor must not.
    def qwk(name):
        oof = cross_validate(partial(build, name, ["log_total_volume"], 4, CUTOFFS),
                             data, "4class", folds)
        return summarise(oof, 4)["qwk_mean"]
    assert qwk("majority_class") == pytest.approx(0.0)
    for name in ("threshold_rule", "regression_then_threshold", "ordinal_logistic"):
        assert qwk(name) > 0.8


def test_three_class_scheme_runs_on_merged_labels(data, folds):
    oof = cross_validate(partial(build, "threshold_rule", [], 3,
                                 SEVERITY_CONFIG["sensitivity_cutoffs_ml"]),
                         data, "3class", folds)
    assert set(oof.y_true) <= {0, 1, 2} and set(oof.y_pred) <= {0, 1, 2}


def test_permutation_importance_finds_the_informative_feature(data, folds):
    factory = partial(build, "regression_then_threshold", list(FEATURE_NAMES), 4, CUTOFFS)
    table = permutation_importance_cv(factory, data, "4class", folds[:5],
                                      list(FEATURE_NAMES), 3, SEED)
    means = table.groupby("feature").qwk_drop.mean()
    assert means.idxmax() == "log_total_volume"
    assert means["log_total_volume"] > 0.3


# ---------------------------------------------------------------------------
# the selection rule
# ---------------------------------------------------------------------------
def _predictions(cid, y, sites, rows):
    return Predictions(cid, np.array([f"s{i:02d}" for i in range(len(y))]), np.asarray(y),
                       np.asarray(sites), np.asarray(rows))


def _scenario(seed=0, n=60):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    sites = np.array([SITES[i % 3] for i in range(n)])

    def noisy(flip):
        rows = []
        for _ in range(4):
            p = y.copy()
            hit = rng.random(n) < flip
            p[hit] = np.clip(p[hit] + rng.choice([-1, 1], hit.sum()), 0, 3)
            rows.append(p)
        return np.array(rows)
    return y, sites, noisy


def test_bootstrap_se_is_zero_for_identical_candidates():
    y, sites, noisy = _scenario()
    rows = noisy(0.3)
    a, b = _predictions("x/a", y, sites, rows), _predictions("x/b", y, sites, rows)
    se, dropped = paired_bootstrap_se(a, b, 4, 200, SEED)
    assert se == 0.0 and dropped == 0


def test_a_clearly_better_tier_is_promoted_and_selection_continues():
    y, sites, noisy = _scenario()
    threshold = _predictions("threshold_rule", y, sites, noisy(0.6))
    tier2 = {"volume_only/regression_then_threshold":
             _predictions("volume_only/regression_then_threshold", y, sites, noisy(0.1))}
    tier3 = {"full_features/regression_then_threshold":
             _predictions("full_features/regression_then_threshold", y, sites, noisy(0.1))}
    result = apply_rule(threshold, {2: tier2, 3: tier3}, 4, n_resamples=300, seed=SEED,
                        tie_tolerance=0.0)
    assert result["steps"][0]["promoted"]
    assert len(result["steps"]) == 2                   # tier 3 was considered
    assert not result["steps"][1]["promoted"]          # ...but adds nothing
    assert result["selected"] == "volume_only/regression_then_threshold"


def test_a_tier_that_fails_stops_selection_even_if_the_next_is_better():
    y, sites, noisy = _scenario()
    base = noisy(0.2)
    threshold = _predictions("threshold_rule", y, sites, base)
    tier2 = {"volume_only/ordinal_logistic":
             _predictions("volume_only/ordinal_logistic", y, sites, base)}   # identical
    tier3 = {"full_features/ordinal_logistic":
             _predictions("full_features/ordinal_logistic", y, sites, noisy(0.0))}
    result = apply_rule(threshold, {2: tier2, 3: tier3}, 4, n_resamples=200, seed=SEED,
                        tie_tolerance=0.0)
    assert len(result["steps"]) == 1
    assert result["selected"] == "threshold_rule"


def test_winning_overall_on_one_hospital_alone_is_not_enough():
    # Checked on seeds 0-3 while writing: the overall gain clears one SE on all
    # four, so this isolates the 2-of-3-hospitals condition.
    y, sites, noisy = _scenario(seed=0, n=90)
    incumbent_rows = noisy(0.5)
    challenger_rows = incumbent_rows.copy()
    amsterdam = sites == "Amsterdam"
    challenger_rows[:, amsterdam] = y[amsterdam]                 # perfect on one site
    for site in ("Singapore", "Utrecht"):                         # one class worse on two
        idx = np.flatnonzero(sites == site)[:2]                   # patients elsewhere
        challenger_rows[:, idx] = np.where(y[idx] < 3, y[idx] + 1, 2)
    threshold = _predictions("threshold_rule", y, sites, incumbent_rows)
    challenger = _predictions("volume_only/random_forest", y, sites, challenger_rows)
    result = apply_rule(threshold, {2: {challenger.candidate_id: challenger}}, 4,
                        n_resamples=300, seed=SEED, tie_tolerance=0.0)
    step = result["steps"][0]
    assert step["overall_condition"] and not step["site_condition"]
    assert not step["promoted"] and result["selected"] == "threshold_rule"


def test_within_a_tier_a_tie_goes_to_the_more_interpretable_candidate():
    y, sites, noisy = _scenario()
    rows = noisy(0.2)
    slightly_better = rows.copy()
    slightly_better[0, 0] = y[0]                        # a hair better, well within one SE
    tier = {"volume_only/random_forest": _predictions("volume_only/random_forest", y, sites,
                                                      slightly_better),
            "volume_only/regression_then_threshold":
                _predictions("volume_only/regression_then_threshold", y, sites, rows)}
    out = put_forward(tier, 4, 200, SEED)
    assert out["best"] in tier
    assert out["put_forward"] == "volume_only/regression_then_threshold"


def test_predictions_refuse_an_incomplete_repeat():
    oof = pd.DataFrame({"subject_key": ["a", "b", "a"], "repeat": [0, 0, 1],
                        "y_pred": [0, 1, 0], "y_true": [0, 1, 0], "site": ["A", "A", "A"]})
    with pytest.raises(ValueError, match="missing a prediction"):
        Predictions.from_oof("x", oof)


# ---------------------------------------------------------------------------
# freezing and the Week 7 application
# ---------------------------------------------------------------------------
def test_frozen_bundle_round_trips_and_applies(tmp_path, data):
    from classification.apply_frozen import apply_bundle
    from classification.freeze import make_bundle

    bundle = make_bundle("volume_only/ordinal_logistic", data, cv_mean_qwk=0.5)
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    reloaded = joblib.load(path)
    assert np.array_equal(reloaded["model"].predict(data.frame),
                          bundle["model"].predict(data.frame))
    assert bundle["versions"]["scikit-learn"] and bundle["trained_on"]["n_patients"] == 60

    # Week 7 shape: `prediction` rows, any number of patients, two scanners.
    test_table = synthetic_features(n=21, seed=9, feature_source="prediction")
    keys = test_table.subject_key.unique()
    scanners = {k: ("Amsterdam/GE1T5" if i % 2 else "Utrecht") for i, k in enumerate(keys)}
    table, scores = apply_bundle(reloaded, test_table, scanners)
    assert len(table) == 21 and set(scores) == {"overall", "Amsterdam/GE1T5", "Utrecht"}
    assert scores["overall"]["n"] == 21


def test_bundle_with_stale_cutoffs_is_refused(data):
    from classification.apply_frozen import apply_bundle
    from classification.freeze import make_bundle

    bundle = make_bundle("threshold_rule", data, cv_mean_qwk=0.5)
    bundle["cutoffs_ml"] = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="cut-offs"):
        apply_bundle(bundle, synthetic_features(n=6, feature_source="prediction"), {})


def test_week7_features_never_overwrite_the_training_table():
    from features.run_features import FEATURES_CSV, TEST_FEATURES_CSV, output_csv_for

    assert output_csv_for(["train", "val"]) == FEATURES_CSV
    assert output_csv_for(["test"]) == TEST_FEATURES_CSV
    with pytest.raises(SystemExit):
        output_csv_for(["train", "test"])


# ---------------------------------------------------------------------------
# leave-one-site-out predictions (Phase A1)
# ---------------------------------------------------------------------------
def test_loso_network_must_have_held_out_the_site_it_is_applied_to(tmp_path):
    torch = pytest.importorskip("torch")
    from segmentation.run_loso_predictions import LOSO, load_loso_network
    from segmentation.unet import UNet

    state = UNet(in_channels=2).state_dict()
    name = LOSO["checkpoints"]["Utrecht"]
    torch.save({"model": state, "epoch": 70, "val_dice": 0.76,
                "held_out_site": "Singapore"}, tmp_path / name)
    with pytest.raises(ValueError, match="in-sample"):
        load_loso_network("Utrecht", "cpu", checkpoint_dir=tmp_path)

    torch.save({"model": state, "epoch": 70, "val_dice": 0.76,
                "held_out_site": "Utrecht"}, tmp_path / name)
    _, info = load_loso_network("Utrecht", "cpu", checkpoint_dir=tmp_path)
    assert info["held_out_site"] == "Utrecht" and info["epoch"] == 70

    torch.save({"model": state, "epoch": 70}, tmp_path / name)      # a resume file, say
    with pytest.raises(ValueError, match="missing"):
        load_loso_network("Utrecht", "cpu", checkpoint_dir=tmp_path)


def test_loso_config_maps_every_site_to_its_own_held_out_network():
    from segmentation.run_loso_predictions import LOSO

    for site, filename in LOSO["checkpoints"].items():
        assert filename == f"unet_loso_{site.lower()}_aug.pt"
    assert set(LOSO["checkpoints"]) == set(LOSO["expected_dice_by_site"]) == set(SITES)


def test_site_agreement_passes_fails_and_refuses_a_partial_run():
    from segmentation.run_loso_predictions import site_agreement

    expected = {"A": 0.78, "B": 0.80}
    close = pd.DataFrame({"site": ["A", "A", "B"], "dice": [0.77, 0.79, 0.81]})
    assert site_agreement(close, expected, 0.03)["passed"]
    far = pd.DataFrame({"site": ["A", "B"], "dice": [0.70, 0.80]})
    assert not site_agreement(far, expected, 0.03)["passed"]
    partial_run = pd.DataFrame({"site": ["A"], "dice": [0.78]})
    verdict = site_agreement(partial_run, expected, 0.03)
    assert not verdict["passed"] and verdict["per_site"]["B"]["n"] == 0


def test_inference_path_puts_tta_probabilities_back_on_the_grid(monkeypatch):
    """run_ensemble.predict with one network: TTA averaging, un-padding, brain clip.

    A stub network outputs a fixed left-heavy logit map, so its mirrored pass
    differs from the direct one; the averaged probability is then exactly
    computable by hand.
    """
    torch = pytest.importorskip("torch")
    from segmentation import run_ensemble
    from segmentation.dataset import pad_or_crop

    shape = (6, 8, 3)
    brain = np.zeros(shape, bool); brain[1:5, 1:7, :] = True
    centre = (3, 4)
    size = 8

    def fake_slices(key, with_labels):
        slices = [np.stack([pad_or_crop(np.zeros(shape[:2], np.float32), size, centre=centre)] * 2)
                  for _ in range(shape[2])]
        brains = [pad_or_crop(brain[:, :, z].astype(np.float32), size, centre=centre)[None]
                  for z in range(shape[2])]
        return {"image": np.stack(slices), "brain": np.stack(brains),
                "z_indices": np.arange(shape[2]), "volume_shape": shape, "centre": centre}

    class LeftHeavy(torch.nn.Module):
        def forward(self, x):
            logits = torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), -10.0)
            logits[..., : x.shape[3] // 2] = 10.0     # confident on the left half only
            return logits

    # Only the disk read is stubbed. The real undo_pad_or_crop runs, so the test
    # also checks that padding round-trips onto the original grid.
    monkeypatch.setattr(run_ensemble, "load_subject_slices", fake_slices)
    without, passes1 = run_ensemble.predict([LeftHeavy()], "k", "cpu", use_tta=False,
                                            threshold=0.5)
    with_tta, passes2 = run_ensemble.predict([LeftHeavy()], "k", "cpu", use_tta=True,
                                             threshold=0.5)
    assert (passes1, passes2) == (1, 2)
    assert without.shape == shape and without.dtype == bool
    assert not (without & ~brain).any()                # clipped to the brain
    # Mirrored, the left-half-only map averages to 0.5 everywhere -> nothing
    # strictly above 0.5 survives: TTA genuinely changed the answer.
    assert without.any() and not with_tta.any()


# ---------------------------------------------------------------------------
# figures render on synthetic results (and are looked at by a human)
# ---------------------------------------------------------------------------
def test_figures_render_on_synthetic_results(tmp_path, data, folds):
    from classification import make_outputs as mo
    from classification.protocol import pooled_confusion

    summaries, confusions, oofs = {}, {}, {}
    for experiment, features in (("volume_only", ["log_total_volume"]),
                                 ("full_features", list(FEATURE_NAMES))):
        rows = []
        for name in ("majority_class", "threshold_rule") + LEARNED:
            oof = cross_validate(partial(build, name, features, 4, CUTOFFS), data, "4class",
                                 folds)
            oofs[f"{experiment}/{name}"] = oof
            rows.append({"scheme": "4class", "candidate": name, **summarise(oof, 4)})
            confusions[f"4class/{name}"] = pooled_confusion(oof, 4).tolist()
        summaries[experiment] = pd.DataFrame(rows)

    figs = {"scores.png": mo.figure_scores(summaries["volume_only"], "title", "subtitle",
                                           mo.VOLUME_C),
            "confusion.png": mo.figure_confusions(
                confusions, ["4class/threshold_rule", "4class/ordinal_logistic"],
                SEVERITY_CONFIG["classes"], "title", "subtitle"),
            "pattern.png": mo.figure_pattern(summaries["volume_only"],
                                             summaries["full_features"])}

    threshold = Predictions.from_oof("threshold_rule", oofs["volume_only/threshold_rule"])
    tiers = {t: {f"{e}/{n}": Predictions.from_oof(f"{e}/{n}", oofs[f"{e}/{n}"])
                 for n in LEARNED}
             for t, e in ((2, "volume_only"), (3, "full_features"))}
    selection = apply_rule(threshold, tiers, 4, n_resamples=50, seed=SEED, tie_tolerance=0.0)
    figs["choice.png"] = mo.figure_choice(selection)

    factory = partial(build, "ordinal_logistic", list(FEATURE_NAMES), 4, CUTOFFS)
    importance = permutation_importance_cv(factory, data, "4class", folds[:5],
                                           list(FEATURE_NAMES), 2, SEED)
    importance.insert(0, "candidate", "full_features/ordinal_logistic")
    figs["drivers.png"] = mo.figure_drivers(importance)

    rng = np.random.default_rng(0)
    panels = []
    for site, true, given in (("Amsterdam", "mild", "mild"), ("Singapore", "moderate", "severe"),
                              ("Utrecht", "severe", "severe")):
        flair = rng.random((64, 64))
        ref = np.zeros((64, 64), bool); ref[20:30, 20:28] = True
        pred = np.zeros((64, 64), bool); pred[22:32, 21:29] = True
        panels.append({"site": site, "key": f"training_{site}_{site}_7", "flair": flair,
                       "reference": ref, "prediction": pred, "true_class": true,
                       "given_class": given, "reference_ml": 6.0, "predicted_ml": 7.1,
                       "agreement": 0.9})
    figs["patients.png"] = mo.figure_patients(panels, "ordinal logistic")

    for name, fig in figs.items():
        path = mo._save(fig, name, "synthetic", None, root=tmp_path)
        assert path.stat().st_size > 10_000


# ---------------------------------------------------------------------------
# the one-vs-rest and cut-off scores (added with the metrics write-up)
# ---------------------------------------------------------------------------
def test_one_vs_rest_matches_sklearn_and_hand_counts():
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.default_rng(7)
    y_true, y_pred = rng.integers(0, 4, 80), rng.integers(0, 4, 80)
    table = metrics.one_vs_rest(y_true, y_pred, 4)
    np.testing.assert_allclose(table["precision"],
                               precision_score(y_true, y_pred, average=None, labels=range(4)))
    np.testing.assert_allclose(table["recall"],
                               recall_score(y_true, y_pred, average=None, labels=range(4)))
    np.testing.assert_allclose(table["f1"], f1_score(y_true, y_pred, average=None, labels=range(4)))
    assert table["macro"]["f1"] == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    # specificity by hand for class 0: true negatives / all actual negatives
    negatives = y_true != 0
    expected = np.mean(y_pred[negatives] != 0)
    assert table["specificity"][0] == pytest.approx(expected)
    np.testing.assert_allclose(table["false_positive_rate"], 1 - table["specificity"])


def test_at_or_above_reads_the_grade_as_a_cutoff_question():
    y_true = np.array([0, 1, 2, 3, 2, 1])
    y_pred = np.array([0, 2, 2, 3, 1, 1])
    out = metrics.at_or_above(y_true, y_pred, 4, level=2)   # "moderate or worse?"
    # truth >=2: patients 2,3,4 ; called >=2: patients 1,2,3
    assert out["sensitivity"] == pytest.approx(2 / 3)
    assert out["specificity"] == pytest.approx(2 / 3)
    assert out["false_positive_rate"] == pytest.approx(1 / 3)
    assert out["accuracy"] == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        metrics.at_or_above(y_true, y_pred, 4, level=0)
