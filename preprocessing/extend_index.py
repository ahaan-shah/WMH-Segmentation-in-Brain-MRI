"""Stage 8 — join every Week 2 per-subject measurement onto the subject index.

ROADMAP 5.6 asks for the per-subject brain volume "in the index". This does that
and more: it merges the QC table from every stage onto Week 1's index, so one
file answers any per-subject question about the pre-processing.

**Week 1's `subject_index.csv` is not modified.** It is a committed Week 1
artefact with its own provenance manifest, and rewriting it in place would make
that manifest a lie and lose the ability to re-derive Week 1 independently. The
join is written alongside it as `subject_index_week2.csv`, which carries every
Week 1 column plus the Week 2 ones.

    code/.venv/bin/python -m preprocessing.extend_index
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metadata.config import METADATA_OUTPUTS, PROJECT_ROOT
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "extend_index"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
WEEK1_INDEX = METADATA_OUTPUTS / "subject_index.csv"
WEEK2_INDEX = METADATA_OUTPUTS / "subject_index_week2.csv"

# (file, prefix, columns to keep). Prefixes keep the provenance of each column
# obvious in the merged table — `stage3_brain_volume_ml` says where it came from
# in a way that a bare `brain_volume_ml` does not.
SOURCES = [
    ("stage1_head_mask_qc.csv", "stage1", [
        "threshold", "n_components_before_selection", "head_volume_ml",
        "head_fraction_of_volume", "wmh_retained_fraction",
    ]),
    ("stage2_bias_field_qc.csv", "stage2", [
        "bias_span_percent", "bias_p5", "bias_p95", "spm12_field_correlation",
        "cnr_before", "cnr_after", "cnr_change_percent",
        "cv_before", "cv_after", "cv_change_percent",
    ]),
    ("stage3_skull_strip_qc.csv", "stage3", [
        "config", "brain_volume_ml", "brain_fraction_of_head", "shallow_fraction",
        "mean_depth_mm", "wmh_retained_fraction", "wmh_lost_voxels",
    ]),
    ("stage4_tissue_seg_qc.csv", "stage4", [
        "method", "csf_fraction", "gm_fraction", "wm_fraction",
        "wm_mask_volume_ml", "wm_mask_fraction",
        "wm_lesion_coverage", "wm_lesion_coverage_raw", "lesion_in_gm_fraction",
    ]),
    ("stage5_normalisation_qc.csv", "stage5", [
        "method", "reference_value", "reference_voxels", "nawm_level",
        "lesion_level", "lesion_over_nawm", "brain_median", "brain_p99",
    ]),
]


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    if not WEEK1_INDEX.exists():
        raise SystemExit(f"{WEEK1_INDEX} not found — run metadata/build_index.py first.")
    index = pd.read_csv(WEEK1_INDEX)
    logger.info("Week 1 index: %d subjects, %d columns", len(index), index.shape[1])

    merged = index.copy()
    for filename, prefix, columns in SOURCES:
        path = OUTPUTS_DIR / filename
        if not path.exists():
            logger.warning("%s missing — skipping %s columns", filename, prefix)
            continue
        table = pd.read_csv(path)
        available = [c for c in columns if c in table.columns]
        missing = set(columns) - set(available)
        if missing:
            logger.warning("%s: columns not found and skipped: %s", filename, sorted(missing))
        subset = table[["subject_key"] + available].rename(
            columns={c: f"{prefix}_{c}" for c in available}
        )
        merged = merged.merge(subset, on="subject_key", how="left")
        logger.info("merged %-34s +%d columns for %d subjects",
                    filename, len(available), subset["subject_key"].nunique())

    processed = merged["stage5_nawm_level"].notna().sum() if "stage5_nawm_level" in merged else 0
    merged["week2_processed"] = merged.get("stage5_nawm_level", pd.Series(index=merged.index)).notna()

    merged.to_csv(WEEK2_INDEX, index=False)
    write_manifest(WEEK2_INDEX, generating_script=f"preprocessing/{SCRIPT_NAME}.py",
                   extra={"week1_index": str(WEEK1_INDEX),
                          "subjects_processed": int(processed),
                          "note": "Week 1 subject_index.csv is not modified."})

    pd.set_option("display.width", 250)
    logger.info("wrote %s — %d subjects, %d columns (%d Week 2 columns added)",
                WEEK2_INDEX, len(merged), merged.shape[1], merged.shape[1] - index.shape[1])
    logger.info("subjects with the full Week 2 pipeline: %d (the sealed 110 are expected "
                "to be blank until Week 7)", processed)

    done = merged[merged["week2_processed"]]
    if not done.empty:
        logger.info("per-site Week 2 summary:\n%s", done.groupby("site")[[
            "stage1_head_volume_ml", "stage3_brain_volume_ml", "stage4_wm_fraction",
            "stage5_reference_value", "stage5_lesion_over_nawm",
        ]].mean().round(3).to_string())

        # Cross-stage consistency: each mask must be a subset of the previous.
        bad_volume = done[done["stage3_brain_volume_ml"] > done["stage1_head_volume_ml"]]
        if not bad_volume.empty:
            raise SystemExit(f"{len(bad_volume)} subject(s) have a brain mask larger than "
                             f"their head mask — the stages are inconsistent.")
        bad_wm = done[done["stage4_wm_mask_volume_ml"] > done["stage3_brain_volume_ml"]]
        if not bad_wm.empty:
            raise SystemExit(f"{len(bad_wm)} subject(s) have a WM mask larger than their "
                             f"brain mask — the stages are inconsistent.")
        if not np.allclose(done["stage5_nawm_level"], 1.0, atol=1e-6):
            raise SystemExit("Normalised NAWM is not at 1.0 for every subject.")
        logger.info("cross-stage consistency PASSED: head >= brain >= WM for all %d subjects, "
                    "and normalised NAWM is 1.0 everywhere", len(done))


if __name__ == "__main__":
    main()
