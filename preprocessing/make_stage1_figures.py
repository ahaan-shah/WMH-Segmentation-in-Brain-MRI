"""Stage 1 report figure — head mask overlays across all three sites.

The Week 2 checklist calls for a visual check of the head mask. CLAUDE.md
Section 5.7 requires every report figure to come from committed code rather
than a screenshot, so the check lives here.

Includes the subject that broke two-class Otsu (Utrecht 11, whose exceptionally
bright scalp pulled the threshold above the brain), because a regression there
is the specific failure this stage was rebuilt to prevent.

    code/.venv/bin/python -m preprocessing.make_stage1_figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metadata.config import PROJECT_ROOT
from metadata.derived import HEAD_MASK, load_derived_mask
from metadata.geometry import voxel_spacing_mm
from metadata.loader import load_nifti, subjects_by_key
from metadata.provenance import get_git_commit_hash, write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "make_stage1_figures"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
QC_CSV = OUTPUTS_DIR / "stage1_head_mask_qc.csv"

# Utrecht 11 is included deliberately: it is the two-class Otsu failure case.
FEATURED = [
    "training_Utrecht_Utrecht_11",
    "training_Utrecht_Utrecht_0",
    "training_Singapore_Singapore_50",
    "training_Singapore_Singapore_51",
    "training_Amsterdam_GE3T_100",
    "training_Amsterdam_GE3T_101",
]


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    commit = get_git_commit_hash()

    subjects = subjects_by_key()
    keys = [k for k in FEATURED if k in subjects]

    fig, axes = plt.subplots(2, len(keys), figsize=(3.1 * len(keys), 7.2))
    for column, key in enumerate(keys):
        subject = subjects[key]
        image = load_nifti(subject.flair_path)
        flair = np.asarray(image.dataobj, dtype=float)
        spacing = voxel_spacing_mm(image)
        mask = load_derived_mask(key, HEAD_MASK)

        vmax = np.percentile(flair[flair > 0], 99.5)
        z = flair.shape[2] // 2
        axes[0, column].imshow(flair[:, :, z].T[::-1], cmap="gray", vmax=vmax)
        axes[0, column].contour(mask[:, :, z].T[::-1], colors="lime", linewidths=0.8)
        axes[0, column].set_title(f"{subject.site} {subject.subject_id}\naxial", fontsize=8)

        y = flair.shape[0] // 2
        aspect = spacing[2] / spacing[1]
        axes[1, column].imshow(flair[y, :, :].T[::-1], cmap="gray", vmax=vmax, aspect=aspect)
        axes[1, column].contour(mask[y, :, :].T[::-1], colors="lime", linewidths=0.8)
        axes[1, column].set_title("sagittal", fontsize=8)

        for row in (0, 1):
            axes[row, column].axis("off")

    fig.suptitle(
        "Stage 1 head mask (green) — multi-Otsu 3-class + in-plane morphology.  "
        f"Leftmost is the subject two-class Otsu failed on.  (commit {commit[:8]})",
        fontsize=10,
    )
    fig.tight_layout()
    path = FIGURES_DIR / "stage1_head_masks.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    write_manifest(path, generating_script=f"preprocessing/{SCRIPT_NAME}.py")

    if QC_CSV.exists():
        table = pd.read_csv(QC_CSV)
        fig, axis = plt.subplots(figsize=(6.5, 4))
        for site, group in table.groupby("site"):
            axis.scatter(group["fov_z_mm"], group["head_volume_ml"], s=26, label=site, alpha=0.85)
        axis.set(xlabel="through-plane field of view (mm)", ylabel="head mask volume (mL)",
                 title="Head volume vs field of view\nAmsterdam is larger because its FOV includes the neck")
        axis.legend(fontsize=8)
        fig.tight_layout()
        path = FIGURES_DIR / "stage1_head_volumes.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        write_manifest(path, generating_script=f"preprocessing/{SCRIPT_NAME}.py")

    logger.info("wrote figures to %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
