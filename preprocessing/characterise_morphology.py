"""Stage 7 driver — characterise the morphological operators for Week 3.

The operators themselves live in `preprocessing/morphology.py`. This script
measures what each one would cost if applied, using the *reference* lesion masks
— which gives Week 3 the ceiling it operates under, because a filter cannot
preserve more true lesions than this table says.

Two things get quantified:

1. **Minimum-component-size filtering.** The standard false-positive filter, and
   ROADMAP 6.3 flags it as trading directly between two scored metrics. On this
   dataset the trade is unusually harsh because half the reference lesions are
   5 voxels or fewer. Reported as both a lesion-count loss and a volume loss,
   because the two diverge enormously and a filter that looks harmless on Dice
   can be devastating on lesion F1.

2. **Opening.** The classic speck remover. Reported separately because it is the
   operation most likely to be reached for by default and the one most likely to
   be catastrophic here.

Nothing is applied to any pipeline artefact. Week 3 tunes the actual threshold
on predictions, on the validation split only, and reports the trade-off curve
(ROADMAP 6.3).

    code/.venv/bin/python -m preprocessing.characterise_morphology
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from metadata.config import MORPHOLOGY_CONFIG, PROJECT_ROOT
from metadata.geometry import voxel_volume_mm3
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from preprocessing.morphology import (
    CONNECTIVITY_6,
    CONNECTIVITY_26,
    minimum_size_cost,
    open_in_plane,
)

SCRIPT_NAME = "characterise_morphology"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
SIZE_CSV = OUTPUTS_DIR / "stage7_minimum_size_cost.csv"
OPENING_CSV = OUTPUTS_DIR / "stage7_opening_cost.csv"

SIZE_THRESHOLDS = (2, 3, 5, 10, 20)
OPENING_RADII_MM = (1.0, 2.0, 3.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    subjects = subjects_by_key()
    keys = load_split(args.split)
    logger.info("Stage 7: characterising morphological operators on %d reference masks",
                len(keys))

    size_rows, opening_rows = [], []
    for i, key in enumerate(keys, start=1):
        subject = subjects[key]
        image = load_nifti(subject.flair_path)
        spacing = image.header.get_zooms()[:3]
        voxel_volume = voxel_volume_mm3(image)
        lesion = load_wmh_mask(subject.mask_path)

        for row in minimum_size_cost(lesion, voxel_volume, SIZE_THRESHOLDS):
            size_rows.append({"subject_key": key, "site": subject.site, **row})

        for radius_mm in OPENING_RADII_MM:
            opened = open_in_plane(lesion, radius_mm, spacing)
            opening_rows.append({
                "subject_key": key, "site": subject.site, "radius_mm": radius_mm,
                "volume_fraction_retained": float(opened.sum() / lesion.sum()),
                "volume_fraction_removed": float(1 - opened.sum() / lesion.sum()),
            })

        if i % 12 == 0 or i == len(keys):
            logger.info("  ...%d/%d", i, len(keys))

    size_table = pd.DataFrame(size_rows)
    size_table.to_csv(SIZE_CSV, index=False)
    write_manifest(SIZE_CSV, generating_script=f"preprocessing/{SCRIPT_NAME}.py")

    opening_table = pd.DataFrame(opening_rows)
    opening_table.to_csv(OPENING_CSV, index=False)
    write_manifest(OPENING_CSV, generating_script=f"preprocessing/{SCRIPT_NAME}.py")

    pd.set_option("display.width", 250)

    summary = size_table.groupby("minimum_voxels").agg(
        minimum_volume_mm3=("minimum_volume_mm3", "mean"),
        lesion_count_fraction_removed=("lesion_count_fraction_removed", "mean"),
        volume_fraction_removed=("volume_fraction_removed", "mean"),
    )
    logger.info("MINIMUM-SIZE FILTER — what it deletes from the REFERENCE masks:\n%s",
                summary.round(4).to_string())
    logger.info("  read this as the ceiling for Week 3: a size filter cannot lose fewer "
                "true lesions than this. The count and volume columns diverge by roughly "
                "an order of magnitude, which is exactly why Dice would not reveal it.")

    opening_summary = opening_table.groupby("radius_mm")["volume_fraction_removed"].agg(
        ["mean", "max"])
    logger.info("OPENING — fraction of reference lesion VOLUME destroyed:\n%s",
                opening_summary.round(4).to_string())

    # Connectivity: the same masks counted two ways (ROADMAP 7.2).
    from scipy import ndimage as ndi

    counts = []
    for key in keys:
        lesion = load_wmh_mask(subjects[key].mask_path)
        _, n26 = ndi.label(lesion, structure=CONNECTIVITY_26)
        _, n6 = ndi.label(lesion, structure=CONNECTIVITY_6)
        counts.append({"subject_key": key, "n_26": n26, "n_6": n6})
    connectivity = pd.DataFrame(counts)
    ratio = (connectivity["n_6"] / connectivity["n_26"]).mean()
    logger.info("CONNECTIVITY — 6- vs 26-connectivity lesion counts: mean %.1f vs %.1f "
                "(6-conn finds %.2fx as many components). Week 4 reports both (ROADMAP 7.2); "
                "26 is the project default, matching checks/evaluation.py.",
                connectivity["n_6"].mean(), connectivity["n_26"].mean(), ratio)

    logger.info("Operators implemented in preprocessing/morphology.py and APPLIED IN WEEK 3 "
                "(config: implemented_in_week=%s, applied_in_week=%s). Nothing here modifies "
                "a pipeline artefact.",
                MORPHOLOGY_CONFIG["implemented_in_week"], MORPHOLOGY_CONFIG["applied_in_week"])


if __name__ == "__main__":
    main()
