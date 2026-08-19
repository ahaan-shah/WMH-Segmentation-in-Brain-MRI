"""Stage 1 driver — compute and save the rough head mask for every subject.

Writes `head_mask.nii.gz` per subject into data/interim/ (through
metadata/derived.py, so geometry, orientation and provenance are enforced) and
a QC table to preprocessing/outputs/stage1_head_mask_qc.csv.

Run from the project root:

    code/.venv/bin/python -m preprocessing.run_head_mask
    code/.venv/bin/python -m preprocessing.run_head_mask --splits train val test

**Default is train+val (the 60), not all 170.** The official 110 are sealed
until Week 7 (Decision #7), and there is no reason to pay for them now: the
downstream stages still have parameters to select, so anything computed for the
test set today would be recomputed once the pipeline is frozen. Week 7 reruns
this with `--splits test`.

The acceptance criterion — every reference WMH voxel must fall inside the head
mask — is evaluated on train and val only, for the same reason. The head mask
is a superset of the brain by construction, so a retention below 1.0 means it
has cut into brain tissue and the run has failed.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from metadata.config import HEAD_MASK_CONFIG, PROJECT_ROOT
from metadata.derived import HEAD_MASK, save_derived
from metadata.geometry import voxel_spacing_mm, voxel_volume_mm3
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from preprocessing.head_mask import compute_head_mask

SCRIPT_NAME = "run_head_mask"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
QC_CSV = OUTPUTS_DIR / "stage1_head_mask_qc.csv"


def process_subject(subject, *, evaluate_retention: bool) -> dict:
    """Compute, save and QC one subject's head mask."""
    flair_img = load_nifti(subject.flair_path)
    flair = np.asarray(flair_img.dataobj, dtype=np.float64)
    spacing = voxel_spacing_mm(flair_img)

    mask, diagnostics = compute_head_mask(
        flair,
        spacing,
        closing_radius_mm=HEAD_MASK_CONFIG["closing_radius_mm"],
        opening_radius_mm=HEAD_MASK_CONFIG["opening_radius_mm"],
        n_classes=HEAD_MASK_CONFIG["threshold_classes"],
    )

    save_derived(
        mask,
        subject,
        HEAD_MASK,
        generating_script=f"preprocessing/{SCRIPT_NAME}.py",
        extra={"stage": 1, **diagnostics},
    )

    row = {
        "subject_key": subject.subject_key,
        "split": subject.split,
        "site": subject.site,
        "voxel_volume_mm3": voxel_volume_mm3(flair_img),
        "fov_z_mm": flair.shape[2] * spacing[2],
        **{k: v for k, v in diagnostics.items() if k != "all_thresholds"},
    }

    if evaluate_retention:
        wmh = load_wmh_mask(subject.mask_path)
        # Guard against a vacuous pass: retention of an empty mask is undefined,
        # and this cohort has no zero-lesion subjects (min 0.78 mL).
        if not wmh.any():
            raise ValueError(f"{subject.subject_key}: reference WMH mask is empty")
        row["wmh_retained_fraction"] = float(mask[wmh].mean())
    else:
        row["wmh_retained_fraction"] = np.nan  # sealed split: labels not inspected

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
        help="Which frozen splits to process (default: train val).",
    )
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    subjects = subjects_by_key()
    keys = [key for split in args.splits for key in load_split(split)]
    logger.info("Stage 1: head mask for %d subjects across splits %s", len(keys), args.splits)
    logger.info("config: %s", HEAD_MASK_CONFIG)

    rows = []
    for i, key in enumerate(keys, start=1):
        subject = subjects[key]
        evaluate = subject.split == "training"
        row = process_subject(subject, evaluate_retention=evaluate)
        rows.append(row)
        logger.info(
            "[%3d/%d] %-34s %-10s thr=%7.1f  head=%7.1f mL  ncomp=%4d  retained=%s",
            i,
            len(keys),
            key,
            row["site"],
            row["threshold"],
            row["head_volume_ml"],
            row["n_components_before_selection"],
            "n/a (sealed)" if np.isnan(row["wmh_retained_fraction"])
            else f"{row['wmh_retained_fraction']:.6f}",
        )

    table = pd.DataFrame(rows).sort_values(["site", "subject_key"])
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=f"preprocessing/{SCRIPT_NAME}.py")
    logger.info("wrote %s", QC_CSV)

    # --- acceptance ---------------------------------------------------------
    evaluated = table.dropna(subset=["wmh_retained_fraction"])
    logger.info("head volume (mL) by site:\n%s",
                table.groupby("site")["head_volume_ml"].agg(["min", "median", "max"]).round(1))

    if not evaluated.empty:
        minimum_required = HEAD_MASK_CONFIG["min_wmh_retained_fraction"]
        worst = evaluated["wmh_retained_fraction"].min()
        failures = evaluated[evaluated["wmh_retained_fraction"] < minimum_required]
        logger.info(
            "WMH retention across %d labelled subjects: min=%.6f mean=%.6f",
            len(evaluated), worst, evaluated["wmh_retained_fraction"].mean(),
        )
        if not failures.empty:
            for _, bad in failures.iterrows():
                logger.error(
                    "FAILED %s (%s): retained %.6f < %.6f — the head mask cut into brain",
                    bad["subject_key"], bad["site"],
                    bad["wmh_retained_fraction"], minimum_required,
                )
            raise SystemExit(
                f"Stage 1 acceptance failed for {len(failures)} subject(s). "
                f"No subject is dropped — investigate before proceeding (CLAUDE.md Section 4)."
            )
        logger.info("acceptance PASSED: all %d labelled subjects retain >= %.4f",
                    len(evaluated), minimum_required)

    # Outlier report: flags a bad mask without inspecting all 170 by eye.
    for site, group in table.groupby("site"):
        median = group["head_volume_ml"].median()
        mad = 1.4826 * np.median(np.abs(group["head_volume_ml"] - median))
        outliers = group[np.abs(group["head_volume_ml"] - median) > 3 * mad] if mad > 0 else group.iloc[:0]
        logger.info("%s: median %.0f mL, %d volume outlier(s)%s", site, median, len(outliers),
                    "" if outliers.empty else f" -> {list(outliers['subject_key'])}")


if __name__ == "__main__":
    main()
