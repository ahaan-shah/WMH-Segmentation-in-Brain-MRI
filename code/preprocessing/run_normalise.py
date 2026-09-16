"""Stage 5 driver — normalise the FLAIR and compare the three methods (R2).

Computes all three normalisation methods for every subject so the report can
quantify the rejected alternatives, but saves only the selected one as
`flair_norm` in data/processed/ — the image Week 3 actually segments.

Also runs the fixed-threshold transfer check (see `normalise.py`), which is the
most direct test of whether the intensity scale genuinely transferred across
scanners. It is a normalisation diagnostic, not a segmentation result.

    code/.venv/bin/python -m preprocessing.run_normalise
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from metadata.config import CODE_ROOT, NORMALISATION_CONFIG, PROJECT_ROOT
from metadata.derived import (
    BRAIN_MASK,
    FLAIR_N4,
    FLAIR_NORM,
    TISSUE_SEG,
    load_derived,
    load_derived_mask,
    save_derived,
)
from metadata.loader import load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from preprocessing.normalise import (
    METHODS,
    fixed_threshold_transfers,
    normalisation_quality,
    normalise,
)
from preprocessing.tissue_seg import LABEL_WM

SCRIPT_NAME = "run_normalise"
OUTPUTS_DIR = CODE_ROOT / "preprocessing" / "outputs"
QC_CSV = OUTPUTS_DIR / "stage5_normalisation_qc.csv"
COMPARISON_CSV = OUTPUTS_DIR / "stage5_method_comparison.csv"

# Threshold for the transfer check only. Chosen a priori from the Stage 4
# measurement that lesions sit at 1.39-1.61x normal-appearing WM on every site;
# a cut-off just below that range is the natural probe. Not tuned, not a
# segmentation parameter — Week 3 owns that decision.
TRANSFER_THRESHOLD = 1.3


def process_subject(subject, selected: str, *, evaluate: bool) -> tuple[dict, list[dict]]:
    key = subject.subject_key
    flair = np.asarray(load_derived(key, FLAIR_N4).dataobj, dtype=np.float64)
    brain = load_derived_mask(key, BRAIN_MASK)
    labels = np.asarray(load_derived(key, TISSUE_SEG).dataobj)
    nawm = labels == LABEL_WM  # raw WM class: excludes lesions by construction

    lesion = load_wmh_mask(subject.mask_path) if evaluate else None

    comparison_rows = []
    saved_row = None
    for method in METHODS:
        normalised, statistics = normalise(
            flair, brain, method=method, reference_mask=nawm,
            percentile_range=tuple(NORMALISATION_CONFIG["percentile_range"]),
        )
        quality = normalisation_quality(
            normalised, brain, reference_mask=nawm, lesion_mask=lesion
        )
        row = {"subject_key": key, "site": subject.site, "method": method,
               **{k: v for k, v in statistics.items() if k != "method"}, **quality}

        if method == selected:
            if lesion is not None:
                row.update(fixed_threshold_transfers(
                    normalised, brain, lesion, TRANSFER_THRESHOLD))
            save_derived(normalised, subject, FLAIR_NORM,
                         generating_script=f"code/preprocessing/{SCRIPT_NAME}.py",
                         extra={"stage": 5, **statistics, **quality})
            saved_row = {"split": subject.split, **row}
        comparison_rows.append(row)

    return saved_row, comparison_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    selected = NORMALISATION_CONFIG["primary"]
    subjects = subjects_by_key()
    keys = [key for split in args.splits for key in load_split(split)]
    logger.info("Stage 5: %s normalisation for %d subjects (all %d methods measured)",
                selected, len(keys), len(METHODS))

    saved_rows, comparison_rows = [], []
    for i, key in enumerate(keys, start=1):
        subject = subjects[key]
        saved, comparison = process_subject(
            subject, selected, evaluate=subject.split == "training")
        saved_rows.append(saved)
        comparison_rows.extend(comparison)
        logger.info("[%3d/%d] %-34s %-10s ref=%8.1f  NAWM=%.3f  lesion=%s  dice@%.1f=%s",
                    i, len(keys), key, subject.site,
                    saved.get("reference_value", float("nan")), saved["nawm_level"],
                    "n/a" if "lesion_level" not in saved else f"{saved['lesion_level']:.3f}",
                    TRANSFER_THRESHOLD,
                    "n/a" if "dice" not in saved else f"{saved['dice']:.3f}")

    table = pd.DataFrame(saved_rows).sort_values(["site", "subject_key"])
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=f"code/preprocessing/{SCRIPT_NAME}.py")

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(COMPARISON_CSV, index=False)
    write_manifest(COMPARISON_CSV, generating_script=f"code/preprocessing/{SCRIPT_NAME}.py")

    pd.set_option("display.width", 250)

    # --- the selection evidence, recomputed on the full cohort --------------
    labelled = comparison.dropna(subset=["lesion_level"]) if "lesion_level" in comparison else comparison.iloc[:0]
    if not labelled.empty:
        per_site = labelled.pivot_table(index="method", columns="site",
                                        values="lesion_level", aggfunc="mean")
        cov = (per_site.std(axis=1) / per_site.mean(axis=1)).sort_values()
        logger.info("normalised WMH level by site:\n%s", per_site.round(3).to_string())
        logger.info("cross-site coefficient of variation (LOWER IS BETTER):\n%s",
                    cov.round(4).to_string())
        best = cov.index[0]
        if best != selected:
            logger.warning("configured primary is %s but %s scores better (%.4f vs %.4f)",
                           selected, best, cov[best], cov[selected])
        else:
            logger.info("configured primary %s is confirmed best on the full cohort", selected)

    # --- acceptance ----------------------------------------------------------
    nawm = table["nawm_level"]
    logger.info("NAWM level on the normalised scale: min=%.6f max=%.6f (target exactly 1.0)",
                nawm.min(), nawm.max())
    if not np.allclose(nawm, 1.0, atol=1e-6):
        raise SystemExit(
            "WM-referenced normalisation must place normal-appearing WM at exactly 1.0; "
            "it does not. Investigate before proceeding."
        )
    logger.info("acceptance PASSED: NAWM sits at 1.0 for all %d subjects", len(table))

    if "dice" in table:
        transfer = table.dropna(subset=["dice"])
        if not transfer.empty:
            logger.info("fixed-threshold transfer check at %.1f (NOT a segmentation result, "
                        "not tuned — Week 3 owns that):\n%s", TRANSFER_THRESHOLD,
                        transfer.groupby("site")[["dice", "recall", "predicted_volume_ratio"]]
                        .mean().round(3).to_string())


if __name__ == "__main__":
    main()
