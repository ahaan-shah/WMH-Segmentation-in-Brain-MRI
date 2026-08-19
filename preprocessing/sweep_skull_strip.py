"""Stage 3 sweep — choose the skull-stripping configuration by measurement.

Runs all four configurations (HD-BET and deepbet, each on FLAIR and on
T1-with-transfer) over a **site-stratified subset of the training split** and
scores each on both failure directions.

**Why a subset rather than all 48.** HD-BET is an nnU-Net running on CPU here,
so a full 48 x 4 sweep is hours of compute for a decision that a stratified
sample settles. Selection uses `--n-per-site` subjects from each site (12 total
by default); the winning configuration is then applied to all 60 by
`run_skull_strip.py`, and its acceptance criteria are checked on every one of
them there. Nothing is accepted on the strength of the subset alone.

Selection rule, fixed before the results are seen:

    1. Reject any configuration that loses reference WMH voxels on any subject
       (over-stripping — unrecoverable recall loss in Week 3).
    2. Among survivors, take the lowest mean `shallow_fraction`
       (under-stripping — retained scalp becomes false lesion in Week 3).
    3. Tie-break on the tighter brain mask, then on runtime.

Criterion 1 is a hard gate rather than a scored quantity because the two
failures are not symmetric: retained scalp can still be cleaned up by the
white-matter constraint in Week 3, but a lesion removed at Stage 3 is gone from
the pipeline permanently.

    code/.venv/bin/python -m preprocessing.sweep_skull_strip
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metadata.config import PROJECT_ROOT, SKULL_STRIP_CONFIG
from metadata.derived import FLAIR_N4, HEAD_MASK, derived_path, load_derived_mask
from metadata.geometry import voxel_spacing_mm
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from preprocessing.skull_strip import (
    CONFIGURATIONS,
    DEEPBET_FLAIR,
    DEEPBET_T1,
    HDBET_FLAIR,
    HDBET_T1,
    brain_mask_quality,
    dilate_in_plane,
    run_deepbet,
    run_hdbet,
)

SCRIPT_NAME = "sweep_skull_strip"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
SWEEP_CSV = OUTPUTS_DIR / "stage3_skull_strip_sweep.csv"
SUMMARY_CSV = OUTPUTS_DIR / "stage3_skull_strip_summary.csv"


def stratified_subset(keys: list[str], subjects: dict, n_per_site: int, seed: int) -> list[str]:
    """Deterministically sample n subjects per site, so all scanners are covered."""
    rng = np.random.default_rng(seed)
    by_site: dict[str, list[str]] = {}
    for key in keys:
        by_site.setdefault(subjects[key].site, []).append(key)
    chosen: list[str] = []
    for site in sorted(by_site):
        pool = sorted(by_site[site])
        take = min(n_per_site, len(pool))
        chosen.extend(rng.choice(pool, size=take, replace=False).tolist())
    return sorted(chosen)


def compute_mask(config: str, subject, reference_img, predictor, dilation: int) -> np.ndarray:
    """Run one configuration on one subject, returning a boolean brain mask."""
    if config == DEEPBET_FLAIR:
        return run_deepbet(derived_path(subject.subject_key, FLAIR_N4), reference_img)
    if config == DEEPBET_T1:
        # deepbet dilates internally, so no separate step is needed.
        return run_deepbet(Path(subject.t1_path), reference_img, n_dilate=dilation)
    if config == HDBET_FLAIR:
        return run_hdbet(derived_path(subject.subject_key, FLAIR_N4), reference_img, predictor)
    if config == HDBET_T1:
        mask = run_hdbet(Path(subject.t1_path), reference_img, predictor)
        return dilate_in_plane(mask, dilation)
    raise ValueError(f"Unknown configuration {config!r}")


def select_configuration(summary: pd.DataFrame, logger) -> str | None:
    """Apply the fixed selection rule."""
    minimum = SKULL_STRIP_CONFIG["min_wmh_retained_fraction"]
    logger.info("gate 1 — WMH retention (over-stripping), must be >= %.4f on every subject:", minimum)

    survivors = []
    for config, row in summary.iterrows():
        passed = row["wmh_retained_worst"] >= minimum
        logger.info("    %-15s worst=%.6f  lost on %d subject(s)  %s", config,
                    row["wmh_retained_worst"], int(row["subjects_losing_wmh"]),
                    "PASS" if passed else "REJECTED (cuts into brain)")
        if passed:
            survivors.append(config)

    if not survivors:
        logger.error("No configuration preserved the reference WMH on every subject.")
        return None

    logger.info("gate 2 — shallow_fraction (under-stripping), lower is better:")
    ranked = summary.loc[survivors].sort_values(
        ["shallow_fraction", "brain_volume_ml", "runtime_s"]
    )
    for config, row in ranked.iterrows():
        logger.info("    %-15s shallow=%.4f  brain=%.0f mL  %.1fs/subject",
                    config, row["shallow_fraction"], row["brain_volume_ml"], row["runtime_s"])
    return str(ranked.index[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-site", type=int, default=4)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGURATIONS), choices=list(CONFIGURATIONS))
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    from metadata.config import SEED

    subjects = subjects_by_key()
    keys = stratified_subset(load_split("train"), subjects, args.n_per_site, SEED)
    dilation = SKULL_STRIP_CONFIG["t1_transfer_dilation_voxels"]
    logger.info("Stage 3 sweep: %d subjects x %d configurations", len(keys), len(args.configs))
    logger.info("subjects: %s", keys)

    rows = []
    for config in args.configs:
        predictor = None
        if config.startswith("hdbet"):
            import torch
            from HD_BET.checkpoint_download import maybe_download_parameters
            from HD_BET.hd_bet_prediction import get_hdbet_predictor

            maybe_download_parameters()
            started = time.time()
            predictor = get_hdbet_predictor(device=torch.device("cpu"), verbose=False)
            logger.info("%s: predictor initialised in %.1fs", config, time.time() - started)

        for i, key in enumerate(keys, start=1):
            subject = subjects[key]
            reference_img = load_nifti(subject.flair_path)
            spacing = voxel_spacing_mm(reference_img)
            head = load_derived_mask(key, HEAD_MASK)
            lesion = load_wmh_mask(subject.mask_path)

            started = time.time()
            mask = compute_mask(config, subject, reference_img, predictor, dilation)
            elapsed = time.time() - started

            quality = brain_mask_quality(mask, head, spacing, lesion_mask=lesion)
            rows.append({"config": config, "subject_key": key, "site": subject.site,
                         "runtime_s": elapsed, **quality})
            logger.info(
                "%-15s [%2d/%d] %-34s %-10s brain=%6.0f mL  retained=%.6f (lost %4d)  "
                "shallow=%.4f  %5.1fs",
                config, i, len(keys), key, subject.site, quality["brain_volume_ml"],
                quality["wmh_retained_fraction"], quality["wmh_lost_voxels"],
                quality["shallow_fraction"], elapsed,
            )

    table = pd.DataFrame(rows)
    table.to_csv(SWEEP_CSV, index=False)
    write_manifest(SWEEP_CSV, generating_script=f"preprocessing/{SCRIPT_NAME}.py")

    summary = table.groupby("config").agg(
        n=("subject_key", "size"),
        wmh_retained_mean=("wmh_retained_fraction", "mean"),
        wmh_retained_worst=("wmh_retained_fraction", "min"),
        subjects_losing_wmh=("wmh_lost_voxels", lambda s: int((s > 0).sum())),
        shallow_fraction=("shallow_fraction", "mean"),
        shallow_worst=("shallow_fraction", "max"),
        brain_volume_ml=("brain_volume_ml", "mean"),
        brain_fraction_of_head=("brain_fraction_of_head", "mean"),
        runtime_s=("runtime_s", "mean"),
    )
    summary.to_csv(SUMMARY_CSV)
    write_manifest(SUMMARY_CSV, generating_script=f"preprocessing/{SCRIPT_NAME}.py")

    pd.set_option("display.width", 250)
    logger.info("summary:\n%s", summary.round(4).to_string())
    logger.info("per-site brain volume (mL):\n%s",
                table.pivot_table(index="config", columns="site",
                                  values="brain_volume_ml", aggfunc="mean").round(0).to_string())

    chosen = select_configuration(summary, logger)
    if chosen is None:
        raise SystemExit("Stage 3 sweep found no acceptable configuration — investigate.")
    logger.info("SELECTED %s", chosen)
    logger.info("Record it: set preprocessing.skull_strip.selected: %s in metadata/dataset.yaml", chosen)


if __name__ == "__main__":
    main()
