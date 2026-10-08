"""Weeks 5-6, Phase D2 — apply the pre-registered selection rule, mechanically.

The rule was fixed in dataset.yaml (`severity_classification.selection`) on
2026-10-07, before any model was fitted. This file applies it to the saved
out-of-fold predictions and logs every step with its numbers, so the choice can
be re-derived by anyone rather than narrated after the fact.

**Why a rule at all.** Ten candidates on 60 patients: the gaps between them may
be smaller than fold-to-fold noise, and simply picking the top scorer selects
partly for luck — and then reports a lucky score. The rule makes each step up in
complexity EARN its place:

1. The threshold rule (no learning) starts as the incumbent.
2. The next tier's best candidate is found. Anything within one bootstrap SE of
   it counts as tied, and the most interpretable tied candidate is put forward
   (regression -> ordinal -> multinomial -> forest).
3. It replaces the incumbent only if BOTH hold:
   - its mean QWK beats the incumbent's by more than one standard error of the
     PAIRED difference (patient-level bootstrap, 2000 resamples), and
   - it wins or ties on at least 2 of the 3 hospitals — a model that wins
     overall by excelling at one hospital has partly learned the hospital.
4. If promoted, repeat with the tier above. If not, STOP: a tier that cannot
   beat the simpler option is not skipped over by the tier above it.

**Why a patient-level bootstrap for the SE**, not the spread across the ten
repeats: every repeat re-deals the SAME 60 people into folds, so the repeat
spread only measures fold-dealing luck and understates how much the result
depends on which 60 people we happened to have. Resampling patients answers the
second question. The same resamples (same seed) are used for every comparison,
and both candidates are scored on the same resample — that is what "paired"
means, and it cancels the shared difficulty of the patients drawn.

The simple threshold rule winning is a legitimate outcome, and reported as one.

    code/.venv/bin/python -m classification.select
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from classification.metrics import per_group, quadratic_weighted_kappa
from classification.models import INTERPRETABILITY_ORDER
from metadata.config import CODE_ROOT, SEED, SEVERITY_CONFIG
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "select"
GENERATING = f"code/classification/{SCRIPT_NAME}.py"
OUTPUTS_DIR = CODE_ROOT / "classification" / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
SELECTION_JSON = OUTPUTS_DIR / "selection.json"

RULE = SEVERITY_CONFIG["selection"]
EXPERIMENT_OF_TIER = {2: "volume_only", 3: "full_features"}


@dataclass(frozen=True)
class Predictions:
    """One candidate's out-of-fold predictions as arrays, patients in a fixed order."""

    candidate_id: str
    keys: np.ndarray        # (n,)
    y_true: np.ndarray      # (n,)
    sites: np.ndarray       # (n,)
    by_repeat: np.ndarray   # (repeats, n)

    @classmethod
    def from_oof(cls, candidate_id: str, oof: pd.DataFrame) -> "Predictions":
        wide = oof.pivot(index="subject_key", columns="repeat", values="y_pred").sort_index()
        if wide.isna().any().any():
            raise ValueError(f"{candidate_id}: some patient is missing a prediction in "
                             f"some repeat — every repeat must predict all of them")
        first = oof.drop_duplicates("subject_key").set_index("subject_key").loc[wide.index]
        if (oof.groupby("subject_key").y_true.nunique() > 1).any():
            raise ValueError(f"{candidate_id}: a patient's true class differs between repeats")
        return cls(candidate_id, wide.index.to_numpy(), first.y_true.to_numpy(dtype=int),
                   first.site.to_numpy(), wide.to_numpy(dtype=int).T)


def mean_qwk(p: Predictions, n_classes: int, index=None) -> float:
    """Mean over repeats of the pooled QWK, optionally on a resample of patients."""
    index = slice(None) if index is None else index
    return float(np.mean([quadratic_weighted_kappa(p.y_true[index], row[index], n_classes)
                          for row in p.by_repeat]))


def site_qwk(p: Predictions, n_classes: int) -> dict:
    """Mean over repeats of each hospital's QWK."""
    per_repeat = [per_group(quadratic_weighted_kappa, p.y_true, row, p.sites, n_classes)
                  for row in p.by_repeat]
    return {site: float(np.mean([r[site] for r in per_repeat])) for site in per_repeat[0]}


def paired_bootstrap_se(a: Predictions, b: Predictions, n_classes: int,
                        n_resamples: int, seed: int) -> tuple[float, int]:
    """SE of mean_qwk(a) - mean_qwk(b) by resampling PATIENTS with replacement.

    Both candidates are scored on the same resample. The generator is re-seeded
    on every call, so every comparison in a selection run uses the identical
    sequence of resamples. Returns (SE, number of resamples dropped as NaN) —
    a resample can in principle draw a single class, where kappa is undefined;
    those are counted, not hidden.
    """
    if not (np.array_equal(a.keys, b.keys) and np.array_equal(a.y_true, b.y_true)):
        raise ValueError(f"{a.candidate_id} and {b.candidate_id} are not scored on the "
                         f"same patients with the same targets — not a paired comparison")
    rng = np.random.default_rng(seed)
    n = len(a.keys)
    differences = []
    for _ in range(n_resamples):
        index = rng.integers(0, n, n)
        differences.append(mean_qwk(a, n_classes, index) - mean_qwk(b, n_classes, index))
    differences = np.asarray(differences)
    finite = differences[np.isfinite(differences)]
    if finite.size < 0.95 * n_resamples:
        raise ValueError(f"over 5% of bootstrap resamples gave an undefined kappa "
                         f"({a.candidate_id} vs {b.candidate_id})")
    return float(np.std(finite, ddof=1)), int(n_resamples - finite.size)


def put_forward(tier_members: dict, n_classes: int, n_resamples: int, seed: int) -> dict:
    """The tier's representative: best mean QWK, then the most interpretable tie."""
    scores = {cid: mean_qwk(p, n_classes) for cid, p in tier_members.items()}
    if any(not np.isfinite(s) for s in scores.values()):
        raise ValueError(f"undefined mean QWK in tier: {scores}")
    best_id = max(scores, key=lambda cid: (scores[cid], -_rank(tier_members[cid])))
    tied = {best_id: {"difference": 0.0, "se": 0.0}}
    for cid, p in tier_members.items():
        if cid == best_id:
            continue
        se, dropped = paired_bootstrap_se(tier_members[best_id], p, n_classes,
                                          n_resamples, seed)
        difference = scores[best_id] - scores[cid]
        if difference <= se:
            tied[cid] = {"difference": difference, "se": se, "nan_resamples": dropped}
    chosen = min(tied, key=lambda cid: _rank(tier_members[cid]))
    return {"scores": scores, "best": best_id, "tied_with_best": tied, "put_forward": chosen}


def _rank(p: Predictions) -> int:
    name = p.candidate_id.split("/")[-1]
    return INTERPRETABILITY_ORDER.index(name)


def promotion(challenger: Predictions, incumbent: Predictions, n_classes: int,
              n_resamples: int, seed: int, tie_tolerance: float) -> dict:
    """Both promotion conditions, with every number that decided them."""
    difference = mean_qwk(challenger, n_classes) - mean_qwk(incumbent, n_classes)
    se, dropped = paired_bootstrap_se(challenger, incumbent, n_classes, n_resamples, seed)
    ch_sites, inc_sites = site_qwk(challenger, n_classes), site_qwk(incumbent, n_classes)
    if set(ch_sites) != set(inc_sites):
        raise ValueError("challenger and incumbent are scored on different hospitals")
    for site, value in {**ch_sites, **{f"inc:{k}": v for k, v in inc_sites.items()}}.items():
        if not np.isfinite(value):
            raise ValueError(f"per-site QWK undefined ({site}); the 2-of-3 rule cannot "
                             f"be applied")
    wins_or_ties = {site: ch_sites[site] >= inc_sites[site] - tie_tolerance
                    for site in ch_sites}
    overall_ok = difference > se
    sites_ok = sum(wins_or_ties.values()) >= 2
    return {"challenger": challenger.candidate_id, "incumbent": incumbent.candidate_id,
            "qwk_difference": difference, "bootstrap_se": se, "nan_resamples": dropped,
            "overall_condition": overall_ok,
            "site_qwk_challenger": ch_sites, "site_qwk_incumbent": inc_sites,
            "wins_or_ties": wins_or_ties, "site_condition": sites_ok,
            "promoted": bool(overall_ok and sites_ok)}


def apply_rule(threshold_rule: Predictions, tiers: dict, n_classes: int, *,
               n_resamples: int, seed: int, tie_tolerance: float) -> dict:
    """The whole procedure. `tiers` maps tier number -> {candidate_id: Predictions}."""
    incumbent = threshold_rule
    steps = []
    for tier in sorted(tiers):
        forward = put_forward(tiers[tier], n_classes, n_resamples, seed)
        challenger = tiers[tier][forward["put_forward"]]
        decision = promotion(challenger, incumbent, n_classes, n_resamples, seed,
                             tie_tolerance)
        steps.append({"tier": tier, **forward, **decision})
        if not decision["promoted"]:
            break
        incumbent = challenger
    return {"selected": incumbent.candidate_id,
            "selected_mean_qwk": mean_qwk(incumbent, n_classes),
            "steps": steps}


# ---------------------------------------------------------------------------
def load_tier_predictions(experiments_dir=None, scheme: str = None) -> tuple:
    # Resolved at call time, not import time, so the directory has one source
    # of truth (the module constant) rather than a frozen copy in a default.
    experiments_dir = EXPERIMENTS_DIR if experiments_dir is None else experiments_dir
    scheme = scheme or RULE["scheme"]
    baselines = pd.read_csv(experiments_dir / "oof_baselines.csv")
    baselines = baselines[baselines.scheme == scheme]
    threshold = Predictions.from_oof(
        "threshold_rule", baselines[baselines.candidate == "threshold_rule"])
    tiers = {}
    for tier, experiment in EXPERIMENT_OF_TIER.items():
        path = experiments_dir / f"oof_{experiment}.csv"
        if not path.exists():
            raise SystemExit(f"{path.name} missing — run classification.run_experiments "
                             f"--experiments {experiment} first. The rule needs every tier.")
        oof = pd.read_csv(path)
        oof = oof[oof.scheme == scheme]
        tiers[tier] = {f"{experiment}/{name}": Predictions.from_oof(f"{experiment}/{name}", part)
                       for name, part in oof.groupby("candidate")}
        listed = set(RULE["complexity_tiers"][tier])
        if {cid.split("/")[-1] for cid in tiers[tier]} != listed:
            raise SystemExit(f"tier {tier} candidates on disk do not match dataset.yaml: "
                             f"{sorted(tiers[tier])} vs {sorted(listed)}")
    return threshold, tiers


def main() -> None:
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    scheme = RULE["scheme"]
    n_classes = len(SEVERITY_CONFIG["classes"])
    if RULE["score"] != "quadratic_weighted_kappa":
        raise SystemExit(f"this script implements QWK selection; dataset.yaml says "
                         f"{RULE['score']!r}")
    conditions = RULE["promotion_conditions"]
    threshold, tiers = load_tier_predictions(scheme=scheme)
    result = apply_rule(threshold, tiers, n_classes,
                        n_resamples=conditions["bootstrap_resamples"], seed=SEED,
                        tie_tolerance=float(conditions["per_site_tie_tolerance"]))

    logger.info("selection on the %s scheme, QWK, %d bootstrap resamples, seed %d",
                scheme, conditions["bootstrap_resamples"], SEED)
    logger.info("tier 1 incumbent: threshold_rule, mean QWK %.4f", mean_qwk(threshold, n_classes))
    for step in result["steps"]:
        logger.info("tier %d — mean QWK per candidate:", step["tier"])
        for cid, score in sorted(step["scores"].items(), key=lambda kv: -kv[1]):
            tie = step["tied_with_best"].get(cid)
            note = (" (best)" if cid == step["best"] else
                    f" (tied: {tie['difference']:.4f} <= SE {tie['se']:.4f})" if tie else "")
            logger.info("    %-45s %.4f%s", cid, score, note)
        logger.info("  put forward: %s", step["put_forward"])
        logger.info("  vs incumbent %s: difference %+.4f, paired bootstrap SE %.4f -> %s",
                    step["incumbent"], step["qwk_difference"], step["bootstrap_se"],
                    "PASS" if step["overall_condition"] else "FAIL")
        for site, ok in step["wins_or_ties"].items():
            logger.info("    %-10s challenger %.4f vs incumbent %.4f  %s", site,
                        step["site_qwk_challenger"][site], step["site_qwk_incumbent"][site],
                        "win/tie" if ok else "loss")
        logger.info("  2-of-3 hospitals -> %s;  PROMOTED: %s",
                    "PASS" if step["site_condition"] else "FAIL", step["promoted"])
        if not step["promoted"]:
            logger.info("  selection stops here; higher tiers are not considered")
    logger.info("SELECTED: %s (mean QWK %.4f — optimistic, because selection itself "
                "looked at it; Week 7's number is the one that counts)",
                result["selected"], result["selected_mean_qwk"])

    SELECTION_JSON.write_text(json.dumps(result, indent=2, default=_json_default))
    write_manifest(SELECTION_JSON, generating_script=GENERATING,
                   extra={"rule": RULE, "seed": SEED})


def _json_default(value):
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(type(value))


if __name__ == "__main__":
    main()
