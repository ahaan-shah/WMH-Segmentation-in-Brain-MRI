"""Stage 2 driver — apply the selected N4 setting and save the corrected FLAIR.

Run `sweep_n4.py` first: it measures which fitting-level count to use and
prints the value to record in `metadata/dataset.yaml` as
`preprocessing.n4.selected_fitting_levels`. This script refuses to run until
that value is set, rather than silently falling back to a default — the default
is precisely what damages high-burden subjects (see `bias_field.py`).

Writes `flair_n4.nii.gz` and `bias_field.nii.gz` per subject into data/interim/,
plus a QC table to preprocessing/outputs/stage2_bias_field_qc.csv carrying the
before/after uniformity and lesion-contrast numbers the Week 2 report needs.

    code/.venv/bin/python -m preprocessing.run_bias_field
    code/.venv/bin/python -m preprocessing.run_bias_field --splits train val test
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from metadata.config import N4_CONFIG, PROJECT_ROOT
from metadata.derived import BIAS_FIELD, FLAIR_N4, HEAD_MASK, load_derived_mask, save_derived
from metadata.geometry import assert_same_geometry, voxel_spacing_mm
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from preprocessing.bias_field import (
    correct_bias_field,
    intensity_uniformity_cv,
    lesion_contrast_to_noise,
    normal_appearing_tissue_mask,
)

SCRIPT_NAME = "run_bias_field"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
QC_CSV = OUTPUTS_DIR / "stage2_bias_field_qc.csv"


def process_subject(subject, levels: int, *, evaluate_contrast: bool) -> dict:
    flair_img = load_nifti(subject.flair_path)
    flair = np.asarray(flair_img.dataobj, dtype=np.float64)
    spacing = voxel_spacing_mm(flair_img)
    head = load_derived_mask(subject.subject_key, HEAD_MASK)

    corrected, bias, diagnostics = correct_bias_field(
        flair, spacing, head,
        fitting_levels=levels,
        iterations_per_level=N4_CONFIG["iterations_per_level"],
        shrink_factor=tuple(N4_CONFIG["shrink_factor"]),
        convergence_threshold=N4_CONFIG["convergence_threshold"],
    )

    save_derived(corrected, subject, FLAIR_N4,
                 generating_script=f"preprocessing/{SCRIPT_NAME}.py",
                 extra={"stage": 2, **diagnostics})
    save_derived(bias, subject, BIAS_FIELD,
                 generating_script=f"preprocessing/{SCRIPT_NAME}.py",
                 extra={"stage": 2, "artefact_role": "estimated multiplicative bias field",
                        **diagnostics})

    row = {
        "subject_key": subject.subject_key,
        "split": subject.split,
        "site": subject.site,
        **diagnostics,
    }
    row["shrink_factor"] = str(row["shrink_factor"])

    # Independent comparison against the organisers' SPM12 correction.
    pre_img = load_nifti(subject.flair_pre_path)
    assert_same_geometry(pre_img, flair_img,
                         context=f"{subject.subject_key}: pre/FLAIR vs orig/FLAIR")
    pre = np.asarray(pre_img.dataobj, dtype=np.float64)
    comparable = head & (pre > 0) & (flair > 0)
    spm_field = flair[comparable] / pre[comparable]
    row["spm12_field_correlation"] = float(np.corrcoef(bias[comparable], spm_field)[0, 1])

    if evaluate_contrast:
        lesion = load_wmh_mask(subject.mask_path)
        nawm = normal_appearing_tissue_mask(flair, head, lesion)  # fixed on the uncorrected image
        row["cnr_before"] = lesion_contrast_to_noise(flair, lesion, nawm)
        row["cnr_after"] = lesion_contrast_to_noise(corrected, lesion, nawm)
        row["cnr_change_percent"] = 100.0 * (row["cnr_after"] / row["cnr_before"] - 1.0)
        row["cv_before"] = intensity_uniformity_cv(flair, nawm)
        row["cv_after"] = intensity_uniformity_cv(corrected, nawm)
        row["cv_change_percent"] = 100.0 * (row["cv_after"] / row["cv_before"] - 1.0)
    else:
        for key in ("cnr_before", "cnr_after", "cnr_change_percent",
                    "cv_before", "cv_after", "cv_change_percent"):
            row[key] = np.nan  # sealed split: labels not inspected
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    levels = N4_CONFIG["selected_fitting_levels"]
    if levels is None:
        raise SystemExit(
            "preprocessing.n4.selected_fitting_levels is null in metadata/dataset.yaml.\n"
            "Run `python -m preprocessing.sweep_n4` first and record the value it selects. "
            "Falling back to a default here is exactly the failure mode Stage 2 exists to "
            "prevent — SimpleITK's default of 4 levels costs up to 32% of lesion contrast "
            "on high-burden subjects."
        )

    subjects = subjects_by_key()
    keys = [key for split in args.splits for key in load_split(split)]
    logger.info("Stage 2: N4 at %d fitting levels for %d subjects (splits %s)",
                levels, len(keys), args.splits)

    rows = []
    for i, key in enumerate(keys, start=1):
        subject = subjects[key]
        row = process_subject(subject, levels, evaluate_contrast=subject.split == "training")
        rows.append(row)
        logger.info("[%3d/%d] %-34s %-10s span=%5.1f%%  CNR %+6.1f%%  CV %+6.1f%%  r(SPM12)=%.3f  %4.1fs",
                    i, len(keys), key, row["site"], row["bias_span_percent"],
                    row["cnr_change_percent"], row["cv_change_percent"],
                    row["spm12_field_correlation"], row["runtime_s"])

    table = pd.DataFrame(rows).sort_values(["site", "subject_key"])
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=f"preprocessing/{SCRIPT_NAME}.py")
    logger.info("wrote %s", QC_CSV)

    pd.set_option("display.width", 250)
    logger.info("per-site summary:\n%s", table.groupby("site")[
        ["bias_span_percent", "cnr_change_percent", "cv_change_percent",
         "spm12_field_correlation"]].mean().round(3).to_string())

    labelled = table.dropna(subset=["cnr_change_percent"])
    if not labelled.empty:
        damaged = labelled[labelled["cnr_change_percent"] < -1.0]
        if not damaged.empty:
            for _, bad in damaged.iterrows():
                logger.warning("%s (%s): lesion contrast fell %.1f%%",
                               bad["subject_key"], bad["site"], bad["cnr_change_percent"])
            logger.warning("%d of %d subjects lost >1%% lesion contrast — reported, not hidden "
                           "(CLAUDE.md Section 4)", len(damaged), len(labelled))
        else:
            logger.info("no subject lost more than 1%% lesion contrast")


if __name__ == "__main__":
    main()
