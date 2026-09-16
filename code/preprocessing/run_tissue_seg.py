"""Stage 4 driver — tissue segmentation and the white-matter mask.

Produces two artefacts per subject:

- `tissue_seg` (interim) — the 3-class CSF/GM/WM label map, so any later stage
  can derive whatever variant it needs rather than being stuck with ours.
- `wm_mask` (processed) — the white-matter class with in-plane enclosed holes
  reclaimed.

**Read the measured limitation before consuming `wm_mask` anywhere.** WMH are
bright on FLAIR but iso- to hypo-intense on T1, so an intensity-driven segmenter
assigns roughly half of them to grey matter. Measured across 15 training
subjects (5 per site):

    variant            lesion coverage (mean / worst)   share of brain
    raw WM class              0.478 / 0.245                  0.32
    + hole filling            0.785 / 0.298                  0.36
    + 1 mm dilation           0.897 / 0.565                  0.50
    + 2 mm dilation           0.951 / 0.676                  0.62
    + 3 mm dilation           0.971 / 0.710                  0.71
    + 4 mm dilation           0.979 / 0.733                  0.78
    non-CSF (WM + GM)         0.961 / 0.847                  0.71

No variant reaches 95% worst-case coverage. Dilation inflates the mask without
rescuing the hard subjects, because on those the lesions are extensive enough to
be classified solidly as grey matter rather than sitting at the WM boundary.

**Two consequences, both recorded rather than discovered later:**

1. `wm_mask` must NOT be used as a hard constraint in Week 3. ROADMAP 6.3.1
   proposes exactly that for false-positive removal; doing so here would delete
   a median 21% and a worst-case 70% of true lesions, capping recall at a level
   no later stage could recover. If a constraint is wanted, the non-CSF mask is
   the safer choice, and its cost should still be reported.
2. For Stage 5 the same behaviour is an *advantage*. Normalisation needs
   *normal-appearing* white matter, and the raw WM class naturally excludes
   lesions precisely because they are T1-hypointense. Stage 5 therefore reads
   the label map's WM class directly rather than the dilated `wm_mask`.

    code/.venv/bin/python -m preprocessing.run_tissue_seg
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from metadata.config import CODE_ROOT, N4_CONFIG, PROJECT_ROOT, SEED, TISSUE_SEGMENTATION_CONFIG
from metadata.derived import BRAIN_MASK, TISSUE_SEG, WM_MASK, load_derived_mask, save_derived
from metadata.geometry import assert_same_geometry, voxel_spacing_mm
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from preprocessing.bias_field import correct_bias_field
from preprocessing.tissue_seg import (
    LABEL_WM,
    segment_tissues,
    tissue_quality,
    white_matter_mask,
)

SCRIPT_NAME = "run_tissue_seg"
OUTPUTS_DIR = CODE_ROOT / "preprocessing" / "outputs"
QC_CSV = OUTPUTS_DIR / "stage4_tissue_seg_qc.csv"


def process_subject(subject, method: str, *, evaluate_coverage: bool) -> dict:
    t1_img = load_nifti(subject.t1_path)
    flair_img = load_nifti(subject.flair_path)
    # orig/T1 is already resampled onto the FLAIR grid by the organisers; assert
    # it rather than trusting the documentation.
    assert_same_geometry(t1_img, flair_img, context=f"{subject.subject_key}: T1 vs FLAIR")

    t1 = np.asarray(t1_img.dataobj, dtype=np.float64)
    spacing = voxel_spacing_mm(t1_img)
    brain = load_derived_mask(subject.subject_key, BRAIN_MASK)

    # Bias-correct the T1 first. Intensity-based tissue segmentation is highly
    # sensitive to the bias field — an uncorrected gradient makes the same
    # tissue cluster into two different classes on opposite sides of the brain.
    started = time.time()
    t1_corrected, _, _ = correct_bias_field(
        t1, spacing, brain,
        fitting_levels=N4_CONFIG["selected_fitting_levels"],
        iterations_per_level=N4_CONFIG["iterations_per_level"],
        shrink_factor=tuple(N4_CONFIG["shrink_factor"]),
        convergence_threshold=N4_CONFIG["convergence_threshold"],
    )
    labels, diagnostics = segment_tissues(
        t1_corrected, brain, spacing, method=method, seed=SEED
    )
    wm = white_matter_mask(labels, include_lesion_prone_wm=True)
    elapsed = time.time() - started

    lesion = load_wmh_mask(subject.mask_path) if evaluate_coverage else None
    quality = tissue_quality(labels, brain, spacing, wm_mask=wm, lesion_mask=lesion)

    save_derived(labels, subject, TISSUE_SEG,
                 generating_script=f"code/preprocessing/{SCRIPT_NAME}.py",
                 extra={"stage": 4, **diagnostics, **quality})
    save_derived(wm, subject, WM_MASK,
                 generating_script=f"code/preprocessing/{SCRIPT_NAME}.py",
                 extra={"stage": 4, "method": method,
                        "derivation": "WM class with in-plane enclosed holes filled",
                        **quality})

    row = {"subject_key": subject.subject_key, "split": subject.split,
           "site": subject.site, "method": method, "runtime_s": elapsed, **quality}
    row["class_means_t1"] = str(diagnostics["class_means_t1"])
    for key in ("wm_lesion_coverage", "wm_lesion_coverage_raw",
                "lesion_in_gm_fraction", "lesion_in_csf_fraction"):
        row.setdefault(key, np.nan)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--method", default=None, help="Override the configured method.")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    method = args.method or TISSUE_SEGMENTATION_CONFIG["selected"]
    if method is None:
        raise SystemExit(
            "preprocessing.tissue_segmentation.selected is null in metadata/dataset.yaml."
        )

    subjects = subjects_by_key()
    keys = [key for split in args.splits for key in load_split(split)]
    logger.info("Stage 4: %s tissue segmentation for %d subjects", method, len(keys))

    rows = []
    for i, key in enumerate(keys, start=1):
        subject = subjects[key]
        row = process_subject(subject, method, evaluate_coverage=subject.split == "training")
        rows.append(row)
        logger.info(
            "[%3d/%d] %-34s %-10s CSF/GM/WM %.2f/%.2f/%.2f  wm_mask=%.2f of brain  "
            "lesion-in-WM raw=%s filled=%s  %4.1fs",
            i, len(keys), key, subject.site,
            row["csf_fraction"], row["gm_fraction"], row["wm_fraction"],
            row["wm_mask_fraction"],
            "n/a" if np.isnan(row["wm_lesion_coverage_raw"]) else f"{row['wm_lesion_coverage_raw']:.3f}",
            "n/a" if np.isnan(row["wm_lesion_coverage"]) else f"{row['wm_lesion_coverage']:.3f}",
            row["runtime_s"],
        )

    table = pd.DataFrame(rows).sort_values(["site", "subject_key"])
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=f"code/preprocessing/{SCRIPT_NAME}.py")
    logger.info("wrote %s", QC_CSV)

    pd.set_option("display.width", 250)
    logger.info("per-site tissue fractions:\n%s", table.groupby("site")[
        ["csf_fraction", "gm_fraction", "wm_fraction", "wm_mask_fraction"]
    ].mean().round(3).to_string())

    labelled = table.dropna(subset=["wm_lesion_coverage"])
    if not labelled.empty:
        logger.info("lesion coverage by the WM mask (the Stage 4 limitation):\n%s",
                    labelled.groupby("site")[
                        ["wm_lesion_coverage_raw", "wm_lesion_coverage",
                         "lesion_in_gm_fraction"]
                    ].agg(["mean", "min"]).round(3).to_string())
        logger.warning(
            "wm_mask covers a mean %.3f and worst-case %.3f of reference WMH. "
            "It must NOT be used as a hard constraint in Week 3 — see the module "
            "docstring. Stage 5 uses the raw WM class instead, where excluding "
            "lesions is the desired behaviour.",
            labelled["wm_lesion_coverage"].mean(), labelled["wm_lesion_coverage"].min(),
        )

    # Sanity band on tissue proportions: a segmentation that collapses two
    # classes together shows up here rather than in a figure nobody opens.
    for name, low, high in (("csf_fraction", 0.05, 0.40),
                            ("gm_fraction", 0.25, 0.60),
                            ("wm_fraction", 0.20, 0.55)):
        bad = table[(table[name] < low) | (table[name] > high)]
        if not bad.empty:
            for _, row in bad.iterrows():
                logger.error("%s (%s): %s = %.3f outside the plausible band [%.2f, %.2f]",
                             row["subject_key"], row["site"], name, row[name], low, high)
            raise SystemExit(
                f"Stage 4: {len(bad)} subject(s) have implausible {name}. Investigate; "
                f"no subject is dropped (CLAUDE.md Section 4)."
            )
    logger.info("tissue proportions plausible for all %d subjects", len(table))


if __name__ == "__main__":
    main()
