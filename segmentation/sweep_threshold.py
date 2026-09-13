"""Week 3, Route A sweep — pick the threshold and minimum lesion size (R4).

Tunes on the **training** split (48 subjects) and reports the winner on the
**validation** split (12), which has seen none of the tuning. The official 110
stay sealed until Week 7.

That direction is deliberate. ROADMAP 6.3 suggests tuning on validation, but
with 48 training subjects available there is no reason to spend the only clean
held-out set on a two-parameter search — doing so would leave no honest estimate
of what the baseline actually achieves. Tuning on train and reporting on val
costs nothing and keeps one number trustworthy.

**What is swept**

- `threshold` on the Stage 5 normalised scale, where 1.0 is normal-appearing
  white matter. Week 2 measured reference lesions at 1.39-1.61x that (per-subject
  range 1.316-1.882), so the search brackets that range.
- `min_size_voxels`, the minimum connected-component size. Week 2 measured what
  this costs on the *reference* masks — deleting components under 5 voxels
  removes 43.5% of true lesions while touching only 3.4% of the volume — so the
  values here stay deliberately small and the trade-off is reported as a curve
  rather than collapsed to one number.

**Selection metric: mean Dice.** With the honest caveat that Dice is
volume-weighted and therefore nearly blind to losing small lesions, which is
half the population here. Voxel-wise recall and the predicted-volume ratio are
reported alongside so a Dice chosen by deleting everything small is visible as
such. Week 7 adds the official lesion-F1 and lesion-recall metrics, which is
where that blindness gets properly measured.

    code/.venv/bin/python -m segmentation.sweep_threshold
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd

from checks.official_score import score_in_memory
from metadata.config import PROJECT_ROOT
from metadata.derived import BRAIN_MASK, FLAIR_NORM, TISSUE_SEG, load_derived, load_derived_mask
from metadata.loader import load_raw_mask_array, load_split, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from segmentation.threshold import threshold_segment

SCRIPT_NAME = "sweep_threshold"
OUTPUTS_DIR = PROJECT_ROOT / "segmentation" / "outputs"
SWEEP_CSV = OUTPUTS_DIR / "route_a_threshold_sweep.csv"
SUMMARY_CSV = OUTPUTS_DIR / "route_a_threshold_summary.csv"

# Brackets the measured lesion range (1.32-1.88x normal white matter), extended
# downward because a lower threshold trades precision for recall and the optimum
# need not sit at the lesion median.
THRESHOLDS = (1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50, 1.60)
MIN_SIZES = (0, 2, 3, 5)


def load_subject(key: str):
    normalised = np.asarray(load_derived(key, FLAIR_NORM).dataobj, dtype=np.float64)
    brain = load_derived_mask(key, BRAIN_MASK)
    tissue = np.asarray(load_derived(key, TISSUE_SEG).dataobj)
    raw_reference, _ = load_raw_mask_array(subjects_by_key()[key].mask_path)
    return normalised, brain, tissue, raw_reference


def evaluate_split(keys: list[str], subjects: dict, logger, label: str) -> pd.DataFrame:
    rows = []
    for index, key in enumerate(keys, start=1):
        normalised, brain, tissue, raw_reference = load_subject(key)
        for threshold, min_size in itertools.product(THRESHOLDS, MIN_SIZES):
            prediction, _ = threshold_segment(
                normalised, brain, threshold=threshold,
                tissue_labels=tissue, exclude_csf=True,
                restrict_to_white_matter=False, min_size_voxels=min_size,
            )
            score = score_in_memory(prediction, raw_reference)
            rows.append({"split": label, "subject_key": key,
                         "site": subjects[key].site, "threshold": threshold,
                         "min_size_voxels": min_size, **score})
        if index % 8 == 0 or index == len(keys):
            logger.info("  %s %d/%d", label, index, len(keys))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    subjects = subjects_by_key()
    train_keys, val_keys = load_split("train"), load_split("val")
    logger.info("Route A sweep: %d thresholds x %d min-sizes = %d settings",
                len(THRESHOLDS), len(MIN_SIZES), len(THRESHOLDS) * len(MIN_SIZES))
    logger.info("tuning on train (%d subjects), reporting on val (%d) — val sees no tuning",
                len(train_keys), len(val_keys))

    train = evaluate_split(train_keys, subjects, logger, "train")
    val = evaluate_split(val_keys, subjects, logger, "val")
    table = pd.concat([train, val], ignore_index=True)
    table.to_csv(SWEEP_CSV, index=False)
    write_manifest(SWEEP_CSV, generating_script=f"segmentation/{SCRIPT_NAME}.py")

    pd.set_option("display.width", 250)

    # Volume ratio is reported as a MEDIAN, not a mean. It is a ratio with the
    # reference volume in the denominator, so subjects with small lesion loads
    # produce enormous values — measured correlation between log reference
    # volume and ratio is -0.651, and the mean came out at 5.6 against a median
    # of 1.2, which reads as catastrophic over-prediction when the typical
    # subject is close to correct.
    grid = train.groupby(["threshold", "min_size_voxels"]).agg(
        dice=("dice", "mean"),
        recall=("recall_voxelwise", "mean"),
        volume_ratio=("volume_ratio", "median"),
    ).reset_index()

    logger.info("TRAIN — mean Dice by threshold (rows) and minimum size (columns):\n%s",
                grid.pivot(index="threshold", columns="min_size_voxels",
                           values="dice").round(4).to_string())
    logger.info("TRAIN — MEDIAN predicted-volume ratio (1.0 = correct total volume):\n%s",
                grid.pivot(index="threshold", columns="min_size_voxels",
                           values="volume_ratio").round(2).to_string())

    best = grid.loc[grid["dice"].idxmax()]
    threshold, min_size = float(best["threshold"]), int(best["min_size_voxels"])
    logger.info("SELECTED on train: threshold=%.2f, min_size=%d voxels "
                "(train Dice %.4f, recall %.3f, volume ratio %.2f)",
                threshold, min_size, best["dice"], best["recall"], best["volume_ratio"])

    chosen = val[(val.threshold == threshold) & (val.min_size_voxels == min_size)]
    logger.info("VALIDATION (held out, no tuning) — Dice %.4f +- %.4f, recall %.3f, "
                "median volume ratio %.2f",
                chosen["dice"].mean(), chosen["dice"].std(),
                chosen["recall_voxelwise"].mean(), chosen["volume_ratio"].median())
    logger.info("VALIDATION per site:\n%s",
                chosen.groupby("site").agg(
                    dice=("dice", "mean"),
                    recall=("recall_voxelwise", "mean"),
                    volume_ratio_median=("volume_ratio", "median"),
                ).round(4).to_string())

    # A global threshold is a compromise. Quantify what it costs per site — but
    # do NOT switch to per-site thresholds: the sealed test set contains two
    # scanners never seen in training, so a per-site rule has nothing to select
    # on there and would be exactly the domain-shift trap the challenge exists
    # to expose (ROADMAP 6.4).
    per_site = train[train.min_size_voxels == min_size].groupby(
        ["site", "threshold"]).dice.mean().reset_index()
    for site, group in per_site.groupby("site"):
        row = group.loc[group.dice.idxmax()]
        at_global = float(group[group.threshold == threshold].dice.iloc[0])
        logger.info("  %-10s would prefer threshold %.2f (Dice %.4f); the global %.2f "
                    "gives %.4f — cost of using one rule everywhere: %.4f",
                    site, row.threshold, row.dice, threshold, at_global, row.dice - at_global)

    summary = grid.copy()
    summary["selected"] = ((summary.threshold == threshold)
                           & (summary.min_size_voxels == min_size))
    summary.to_csv(SUMMARY_CSV, index=False)
    write_manifest(SUMMARY_CSV, generating_script=f"segmentation/{SCRIPT_NAME}.py")

    # Dice is volume-weighted, so a setting can win it by deleting small lesions.
    # Surface that rather than let the single number speak for itself.
    no_filter = grid[(grid.threshold == threshold) & (grid.min_size_voxels == 0)]
    if min_size > 0 and not no_filter.empty:
        logger.info("size filter contribution at the same threshold: Dice %.4f (no filter) "
                    "-> %.4f (min_size=%d). Week 7's lesion-F1 is where the cost of that "
                    "filter actually shows up — Dice is nearly blind to it.",
                    float(no_filter["dice"].iloc[0]), best["dice"], min_size)

    logger.info("Record in metadata/dataset.yaml: segmentation.route_a."
                "{threshold: %.2f, min_size_voxels: %d}", threshold, min_size)


if __name__ == "__main__":
    main()
