"""Stage 2 sweep — choose the N4 fitting-level count by measurement.

Runs N4 at each candidate level count over the **training split only** (48
subjects; val is held for confirmation, test is sealed until Week 7) and scores
each setting on the two quantities that matter:

1. **Lesion-to-WM contrast-to-noise change.** Must not go down. This is the
   guard against N4 absorbing confluent WMH into the bias field, which is
   silent and costs Week 3 real Dice on exactly the high-burden subjects that
   matter most.
2. **Bias field span removed.** Among settings that pass (1), more correction
   is better — that is the point of doing R3 at all.

Selection rule, fixed before the results are seen:

    reject any level count whose mean CNR change is negative on ANY site;
    among survivors, take the one removing the largest bias field span.

Reporting a per-site breakdown rather than a pooled mean is essential here:
the sites genuinely disagree (Utrecht gains contrast from aggressive
correction, Amsterdam loses it), so a pooled average would hide the failure it
exists to catch.

All ROIs are defined ONCE on the uncorrected image and reused for the corrected
image, so before/after are measured over identical voxels.

Run from the project root:

    code/.venv/bin/python -m preprocessing.sweep_n4
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from metadata.config import N4_CONFIG, PROJECT_ROOT
from metadata.derived import HEAD_MASK, load_derived_mask
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

SCRIPT_NAME = "sweep_n4"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
SWEEP_CSV = OUTPUTS_DIR / "stage2_n4_sweep.csv"
SUMMARY_CSV = OUTPUTS_DIR / "stage2_n4_sweep_summary.csv"


def evaluate_subject(subject, levels: int) -> dict:
    """Run N4 at one setting on one subject and score it."""
    flair_img = load_nifti(subject.flair_path)
    flair = np.asarray(flair_img.dataobj, dtype=np.float64)
    spacing = voxel_spacing_mm(flair_img)

    head = load_derived_mask(subject.subject_key, HEAD_MASK)
    lesion = load_wmh_mask(subject.mask_path)

    # ROIs fixed on the uncorrected image; reused unchanged after correction.
    nawm = normal_appearing_tissue_mask(flair, head, lesion)

    cnr_before = lesion_contrast_to_noise(flair, lesion, nawm)
    cv_before = intensity_uniformity_cv(flair, nawm)

    corrected, bias, diagnostics = correct_bias_field(
        flair, spacing, head,
        fitting_levels=levels,
        iterations_per_level=N4_CONFIG["iterations_per_level"],
        shrink_factor=tuple(N4_CONFIG["shrink_factor"]),
        convergence_threshold=N4_CONFIG["convergence_threshold"],
    )

    cnr_after = lesion_contrast_to_noise(corrected, lesion, nawm)
    cv_after = intensity_uniformity_cv(corrected, nawm)

    # Independent validation against the organisers' SPM12 correction
    # (ROADMAP 5.3). pre/FLAIR is a comparator only and never enters the
    # pipeline. The SPM12-implied field is orig/pre.
    pre_img = load_nifti(subject.flair_pre_path)
    assert_same_geometry(pre_img, flair_img, context=f"{subject.subject_key}: pre/FLAIR vs orig/FLAIR")
    pre = np.asarray(pre_img.dataobj, dtype=np.float64)
    comparable = head & (pre > 0) & (flair > 0)
    spm_field = flair[comparable] / pre[comparable]
    ours = bias[comparable]
    field_correlation = (
        float(np.corrcoef(ours, spm_field)[0, 1]) if ours.size > 2 and ours.std() > 0 else np.nan
    )

    return {
        "subject_key": subject.subject_key,
        "site": subject.site,
        "fitting_levels": levels,
        "lesion_volume_ml": float(lesion.sum() * np.prod(spacing) / 1000.0),
        "cnr_before": cnr_before,
        "cnr_after": cnr_after,
        "cnr_change_percent": 100.0 * (cnr_after / cnr_before - 1.0),
        "cv_before": cv_before,
        "cv_after": cv_after,
        "cv_change_percent": 100.0 * (cv_after / cv_before - 1.0),
        "bias_in_lesion_over_nawm": float(bias[lesion].mean() / bias[nawm].mean()),
        "spm12_field_correlation": field_correlation,
        **{k: diagnostics[k] for k in ("bias_span_percent", "runtime_s")},
    }


def select_fitting_levels(summary: pd.DataFrame, logger) -> int | None:
    """Apply the fixed selection rule to the per-site summary."""
    worst_per_level = summary.groupby("fitting_levels")["cnr_change_percent"].min()
    survivors = worst_per_level[worst_per_level >= 0].index.tolist()

    logger.info("worst per-site mean CNR change, by level count:")
    for levels, worst in worst_per_level.items():
        logger.info("    %d levels: %+.2f%%  %s", levels, worst,
                    "PASS" if worst >= 0 else "REJECTED (damages lesion contrast)")

    if not survivors:
        logger.error("No candidate preserves lesion contrast on every site.")
        return None

    spans = summary[summary["fitting_levels"].isin(survivors)].groupby("fitting_levels")[
        "bias_span_percent"
    ].mean()
    chosen = int(spans.idxmax())
    logger.info("survivors %s; mean bias span removed: %s",
                survivors, {int(k): round(v, 2) for k, v in spans.items()})
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Split to sweep over (default: train; test is sealed until W7).")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    subjects = subjects_by_key()
    keys = load_split(args.split)
    candidates = N4_CONFIG["candidate_fitting_levels"]
    logger.info("Stage 2 sweep: %d subjects x %s fitting levels = %d N4 runs",
                len(keys), candidates, len(keys) * len(candidates))

    rows = []
    for levels in candidates:
        for i, key in enumerate(keys, start=1):
            row = evaluate_subject(subjects[key], levels)
            rows.append(row)
            logger.info(
                "levels=%d [%2d/%d] %-34s %-10s CNR %6.2f -> %6.2f (%+6.1f%%)  "
                "span=%5.1f%%  bias(les/nawm)=%.3f  %4.1fs",
                levels, i, len(keys), key, row["site"],
                row["cnr_before"], row["cnr_after"], row["cnr_change_percent"],
                row["bias_span_percent"], row["bias_in_lesion_over_nawm"], row["runtime_s"],
            )

    table = pd.DataFrame(rows)
    table.to_csv(SWEEP_CSV, index=False)
    write_manifest(SWEEP_CSV, generating_script=f"preprocessing/{SCRIPT_NAME}.py")

    summary = (
        table.groupby(["fitting_levels", "site"])
        .agg(
            n=("subject_key", "size"),
            cnr_change_percent=("cnr_change_percent", "mean"),
            cnr_change_worst=("cnr_change_percent", "min"),
            cv_change_percent=("cv_change_percent", "mean"),
            bias_span_percent=("bias_span_percent", "mean"),
            bias_in_lesion_over_nawm=("bias_in_lesion_over_nawm", "mean"),
            spm12_field_correlation=("spm12_field_correlation", "mean"),
            runtime_s=("runtime_s", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    write_manifest(SUMMARY_CSV, generating_script=f"preprocessing/{SCRIPT_NAME}.py")

    pd.set_option("display.width", 250)
    logger.info("per-site summary:\n%s", summary.round(3).to_string(index=False))
    logger.info("wrote %s and %s", SWEEP_CSV, SUMMARY_CSV)

    chosen = select_fitting_levels(summary, logger)
    if chosen is None:
        raise SystemExit("Stage 2 sweep found no acceptable setting — investigate before proceeding.")

    logger.info("SELECTED fitting_levels = %d", chosen)
    logger.info(
        "Record it: set preprocessing.n4.selected_fitting_levels: %d in metadata/dataset.yaml", chosen
    )


if __name__ == "__main__":
    main()
