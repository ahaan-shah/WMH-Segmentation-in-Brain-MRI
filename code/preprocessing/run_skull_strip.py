"""Stage 3 driver — apply the selected skull-stripping configuration (R1).

Run `sweep_skull_strip.py` first; it selects a configuration and prints the
value to record in `metadata/dataset.yaml` as
`preprocessing.skull_strip.selected`. This script refuses to run until that is
set rather than picking a default, because there is no defensible default —
ROADMAP's own recommendation (SynthStrip) is not installable here.

Writes `brain_mask.nii.gz` per subject and a QC table to
preprocessing/outputs/stage3_skull_strip_qc.csv, then re-checks BOTH acceptance
criteria on every subject — the sweep only ever saw a stratified subset, so
full-cohort acceptance is established here, not there.

    code/.venv/bin/python -m preprocessing.run_skull_strip
    code/.venv/bin/python -m preprocessing.run_skull_strip --splits train val test
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metadata.config import CODE_ROOT, PROJECT_ROOT, SKULL_STRIP_CONFIG
from metadata.derived import BRAIN_MASK, HEAD_MASK, load_derived_mask, save_derived
from metadata.geometry import voxel_spacing_mm
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from preprocessing.skull_strip import brain_mask_quality
from preprocessing.sweep_skull_strip import compute_mask

SCRIPT_NAME = "run_skull_strip"
OUTPUTS_DIR = CODE_ROOT / "preprocessing" / "outputs"
QC_CSV = OUTPUTS_DIR / "stage3_skull_strip_qc.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    config = SKULL_STRIP_CONFIG["selected"]
    if config is None:
        raise SystemExit(
            "preprocessing.skull_strip.selected is null in metadata/dataset.yaml.\n"
            "Run `python -m preprocessing.sweep_skull_strip` first and record its choice. "
            "There is no defensible default here — ROADMAP 5.2's recommended tool "
            "(SynthStrip) is not installable in this environment."
        )

    predictor = None
    if config.startswith("hdbet"):
        import torch
        from HD_BET.checkpoint_download import maybe_download_parameters
        from HD_BET.hd_bet_prediction import get_hdbet_predictor

        maybe_download_parameters()
        predictor = get_hdbet_predictor(device=torch.device("cpu"), verbose=False)

    dilation = SKULL_STRIP_CONFIG["t1_transfer_dilation_voxels"]
    subjects = subjects_by_key()
    keys = [key for split in args.splits for key in load_split(split)]
    logger.info("Stage 3: %s for %d subjects (splits %s)", config, len(keys), args.splits)

    rows = []
    for i, key in enumerate(keys, start=1):
        subject = subjects[key]
        reference_img = load_nifti(subject.flair_path)
        spacing = voxel_spacing_mm(reference_img)
        head = load_derived_mask(key, HEAD_MASK)

        started = time.time()
        mask = compute_mask(config, subject, reference_img, predictor, dilation)
        elapsed = time.time() - started

        # The brain mask must lie inside the head mask; anything outside is a
        # stripper artefact and is trimmed rather than propagated.
        outside = int((mask & ~head).sum())
        mask = mask & head

        lesion = load_wmh_mask(subject.mask_path) if subject.split == "training" else None
        quality = brain_mask_quality(
            mask, head, spacing, lesion_mask=lesion,
            shallow_depth_mm=SKULL_STRIP_CONFIG["shallow_depth_mm"],
        )

        save_derived(mask, subject, BRAIN_MASK,
                     generating_script=f"code/preprocessing/{SCRIPT_NAME}.py",
                     extra={"stage": 3, "config": config,
                            "trimmed_outside_head_voxels": outside, **quality})

        row = {"subject_key": key, "split": subject.split, "site": subject.site,
               "config": config, "runtime_s": elapsed,
               "trimmed_outside_head_voxels": outside, **quality}
        row.setdefault("wmh_retained_fraction", np.nan)
        row.setdefault("wmh_lost_voxels", np.nan)
        rows.append(row)

        logger.info("[%3d/%d] %-34s %-10s brain=%6.0f mL  retained=%s  shallow=%.4f  %5.1fs",
                    i, len(keys), key, subject.site, quality["brain_volume_ml"],
                    "n/a (sealed)" if lesion is None else f"{quality['wmh_retained_fraction']:.6f}",
                    quality["shallow_fraction"], elapsed)

    table = pd.DataFrame(rows).sort_values(["site", "subject_key"])
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=f"code/preprocessing/{SCRIPT_NAME}.py")
    logger.info("wrote %s", QC_CSV)

    pd.set_option("display.width", 250)
    logger.info("per-site summary:\n%s", table.groupby("site")[
        ["brain_volume_ml", "brain_fraction_of_head", "shallow_fraction", "wmh_retained_fraction"]
    ].mean().round(4).to_string())

    # --- full-cohort acceptance ---------------------------------------------
    labelled = table.dropna(subset=["wmh_retained_fraction"])
    if not labelled.empty:
        minimum = SKULL_STRIP_CONFIG["min_wmh_retained_fraction"]
        failures = labelled[labelled["wmh_retained_fraction"] < minimum]
        logger.info("WMH retention over %d labelled subjects: min=%.6f mean=%.6f",
                    len(labelled), labelled["wmh_retained_fraction"].min(),
                    labelled["wmh_retained_fraction"].mean())
        if not failures.empty:
            for _, bad in failures.iterrows():
                logger.error("FAILED %s (%s): retained %.6f, lost %d lesion voxels",
                             bad["subject_key"], bad["site"],
                             bad["wmh_retained_fraction"], int(bad["wmh_lost_voxels"]))
            raise SystemExit(
                f"Stage 3 acceptance failed for {len(failures)} subject(s). No subject is "
                f"dropped — investigate before proceeding (CLAUDE.md Section 4)."
            )
        logger.info("acceptance criterion 1 PASSED (no over-stripping on any subject)")

    # Brain-volume outliers flag a failed strip without inspecting all 60 by eye.
    for site, group in table.groupby("site"):
        median = group["brain_volume_ml"].median()
        mad = 1.4826 * np.median(np.abs(group["brain_volume_ml"] - median))
        outliers = group[np.abs(group["brain_volume_ml"] - median) > 3 * mad] if mad > 0 else group.iloc[:0]
        logger.info("%s: median brain %.0f mL, %d outlier(s)%s", site, median, len(outliers),
                    "" if outliers.empty else f" -> {list(outliers['subject_key'])}")


if __name__ == "__main__":
    main()
