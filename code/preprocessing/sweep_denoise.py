"""Stage 6 driver — evaluate the denoising and contrast-enhancement branches.

Runs each candidate filter on the normalised FLAIR and measures what it does to
lesions, then decides whether any of them earns a place in the pipeline.

The decision rule, fixed before the results are seen:

    Adopt a filter only if it leaves >= 95% of small-lesion contrast intact
    AND improves lesion-to-WM contrast-to-noise. Otherwise reject it and report
    the numbers.

`enabled_in_pipeline` in metadata/dataset.yaml is deliberately null until this
runs, so the outcome is recorded as a measurement either way rather than as an
assumption. The expectation on record is rejection — see `denoise.py` for why —
and a measured confirmation of an expectation is still a result.

Outputs are persisted for whichever filters run, because ROADMAP 9.3 asks Week 7
to report SSIM for pre-processing quality and that comparison needs the
filtered/original pair still on disk. Discovering in Week 7 that the inputs were
never kept is the failure this avoids.

    code/.venv/bin/python -m preprocessing.sweep_denoise
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from metadata.config import (
    CODE_ROOT,
    CONTRAST_ENHANCEMENT_CONFIG,
    DENOISING_CONFIG,
)
from metadata.derived import (
    BRAIN_MASK,
    FLAIR_CLAHE,
    FLAIR_DENOISED,
    FLAIR_NORM,
    load_derived,
    load_derived_mask,
    save_derived,
)
from metadata.geometry import voxel_spacing_mm
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from preprocessing.denoise import (
    CURVATURE_DIFFUSION,
    GAUSSIAN,
    clahe,
    denoise,
    lesion_contrast_retention,
    structural_similarity_in_mask,
)

SCRIPT_NAME = "sweep_denoise"
OUTPUTS_DIR = CODE_ROOT / "preprocessing" / "outputs"
SWEEP_CSV = OUTPUTS_DIR / "stage6_denoise_sweep.csv"
SUMMARY_CSV = OUTPUTS_DIR / "stage6_denoise_summary.csv"

CLAHE = "clahe"
# patch_based is implemented in denoise.py but excluded from the sweep: the
# SimpleITK filter takes many minutes per subject on CPU, which is hours for the
# 60 and more again for the 110 in Week 7, for a branch already expected to be
# rejected. Excluded on cost, with the reason recorded rather than the method
# quietly omitted.
CANDIDATES = (CURVATURE_DIFFUSION, GAUSSIAN, CLAHE)

# Small lesions are the whole question: 49.6% of reference lesions are <= 5
# voxels and 69.2% are <= 10, carrying 1.5% and 3.1% of volume respectively.
SMALL_LESION_BIN = "retention_small"


def evaluate_subject(subject, method: str, *, persist: bool) -> dict:
    key = subject.subject_key
    spacing = voxel_spacing_mm(load_nifti(subject.flair_path))
    original = np.asarray(load_derived(key, FLAIR_NORM).dataobj, dtype=np.float64)
    brain = load_derived_mask(key, BRAIN_MASK)
    lesion = load_wmh_mask(subject.mask_path)

    started = time.time()
    if method == CLAHE:
        filtered = clahe(
            original, brain,
            clip_limit=CONTRAST_ENHANCEMENT_CONFIG["clip_limit"],
            kernel_size_fraction=CONTRAST_ENHANCEMENT_CONFIG["kernel_size_fraction"],
        )
    else:
        filtered = denoise(original, spacing, method=method)
        filtered = np.where(brain, filtered, 0.0)
    elapsed = time.time() - started

    retention = lesion_contrast_retention(original, filtered, lesion)
    ssim = structural_similarity_in_mask(original, filtered, brain)

    # Contrast-to-noise on identical ROIs, defined once on the original.
    nawm = brain & ~lesion & (original > 0.9) & (original < 1.1)
    def cnr(volume):
        reference = volume[nawm]
        median = float(np.median(reference))
        noise = 1.4826 * float(np.median(np.abs(reference - median)))
        return (float(np.median(volume[lesion])) - median) / noise if noise > 0 else np.nan

    if persist:
        save_derived(filtered, subject, FLAIR_CLAHE if method == CLAHE else FLAIR_DENOISED,
                     generating_script=f"code/preprocessing/{SCRIPT_NAME}.py",
                     extra={"stage": 6, "method": method, "ssim_vs_original": ssim,
                            **retention})

    return {"subject_key": key, "site": subject.site, "method": method,
            "runtime_s": elapsed, "ssim": ssim,
            "cnr_before": cnr(original), "cnr_after": cnr(filtered),
            **retention}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--methods", nargs="+", default=list(CANDIDATES))
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    subjects = subjects_by_key()
    keys = load_split(args.split)
    logger.info("Stage 6: %d subjects x %s", len(keys), args.methods)

    rows = []
    for method in args.methods:
        for i, key in enumerate(keys, start=1):
            # Persist one representative subject per method for the Week 7 SSIM
            # comparison; persisting all 60 for a branch we expect to reject
            # would cost ~1 GB for nothing.
            row = evaluate_subject(subjects[key], method, persist=(i == 1))
            rows.append(row)
            logger.info("%-32s [%2d/%d] %-34s retention=%.3f small(<=10vox)=%s "
                        "below-half=%.3f  CNR %+.1f%%  ssim=%.4f  %4.1fs",
                        method, i, len(keys), key, row["retention_mean"],
                        f"{row.get(SMALL_LESION_BIN, float('nan')):.3f}",
                        row["fraction_below_half_retention"],
                        100 * (row["cnr_after"] / row["cnr_before"] - 1),
                        row["ssim"], row["runtime_s"])

    table = pd.DataFrame(rows)
    table.to_csv(SWEEP_CSV, index=False)
    write_manifest(SWEEP_CSV, generating_script=f"code/preprocessing/{SCRIPT_NAME}.py")

    table["cnr_change_percent"] = 100 * (table["cnr_after"] / table["cnr_before"] - 1)
    summary = table.groupby("method").agg(
        n=("subject_key", "size"),
        retention_mean=("retention_mean", "mean"),
        small_lesion_retention=(SMALL_LESION_BIN, "mean"),
        fraction_below_half=("fraction_below_half_retention", "mean"),
        lesions_below_half=("lesions_below_half_retention", "sum"),
        cnr_change_percent=("cnr_change_percent", "mean"),
        ssim=("ssim", "mean"),
        runtime_s=("runtime_s", "mean"),
    )
    summary.to_csv(SUMMARY_CSV)
    write_manifest(SUMMARY_CSV, generating_script=f"code/preprocessing/{SCRIPT_NAME}.py")

    pd.set_option("display.width", 250)
    logger.info("summary:\n%s", summary.round(4).to_string())

    # --- the fixed decision rule --------------------------------------------
    minimum_retention = DENOISING_CONFIG.get("minimum_small_lesion_retention", 0.95)
    logger.info("decision rule: adopt only if small-lesion (<=10 vox) contrast retention "
                ">= %.2f AND lesion CNR improves.", minimum_retention)
    adopted = []
    for method, row in summary.iterrows():
        keeps_lesions = row["small_lesion_retention"] >= minimum_retention
        improves_cnr = row["cnr_change_percent"] > 0
        verdict = "ADOPT" if (keeps_lesions and improves_cnr) else "REJECT"
        logger.info("    %-32s small-lesion retention %.3f (%s), CNR %+.1f%% (%s) -> %s",
                    method, row["small_lesion_retention"], "pass" if keeps_lesions else "FAIL",
                    row["cnr_change_percent"], "pass" if improves_cnr else "FAIL", verdict)
        if verdict == "ADOPT":
            adopted.append(method)

    if adopted:
        logger.info("ADOPTED: %s — record in metadata/dataset.yaml", adopted)
    else:
        logger.info("NONE ADOPTED. The un-filtered normalised FLAIR from Stage 5 remains "
                    "the pipeline input for Week 3. This confirms the expectation recorded "
                    "in denoise.py before measuring, and is reported as a finding.")


if __name__ == "__main__":
    main()
