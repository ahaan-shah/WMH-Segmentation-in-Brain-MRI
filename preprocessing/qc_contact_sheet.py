"""Whole-cohort visual QC — every subject on one page, worst cases called out.

A stage is not finished because its acceptance numbers passed. Numbers catch
the failures you thought to measure; a contact sheet catches the ones you
didn't. Inspecting three representative subjects proves almost nothing about
60, and median subjects are the *least* likely to have failed — so this renders
every subject, then repeats the extremes at a readable size.

Two outputs per stage:

- `<stage>_contact_sheet.png` — all subjects, one slice each, mask outlined.
  Meant to be scanned for anything that looks unlike its neighbours.
- `<stage>_worst_cases.png` — the subjects ranked worst by the stage's own QC
  metric, large enough to actually judge. If a failure exists, it is far more
  likely to be here than in a random sample.

    code/.venv/bin/python -m preprocessing.qc_contact_sheet --stage 3
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metadata.config import PROJECT_ROOT
from metadata.derived import BRAIN_MASK, FLAIR_N4, HEAD_MASK, load_derived, load_derived_mask
from metadata.loader import load_nifti, load_wmh_mask, subjects_by_key
from metadata.provenance import get_git_commit_hash, write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "qc_contact_sheet"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

STAGES = {
    1: {
        "qc_csv": "stage1_head_mask_qc.csv",
        "mask": HEAD_MASK,
        "colour": "deepskyblue",
        "rank_by": "head_volume_ml",
        "label": "Stage 1 head mask",
    },
    3: {
        "qc_csv": "stage3_skull_strip_qc.csv",
        "mask": BRAIN_MASK,
        "colour": "red",
        "rank_by": "shallow_fraction",
        "label": "Stage 3 brain mask",
    },
}


def _slice_with_most_lesion(subject) -> int:
    lesion = load_wmh_mask(subject.mask_path)
    return int(np.argmax(lesion.sum(axis=(0, 1))))


def contact_sheet(table: pd.DataFrame, spec: dict, commit: str, logger) -> None:
    subjects = subjects_by_key()
    keys = list(table.sort_values(["site", "subject_key"])["subject_key"])
    columns = 10
    rows = int(np.ceil(len(keys) / columns))

    fig, axes = plt.subplots(rows, columns, figsize=(1.7 * columns, 1.9 * rows))
    axes = np.atleast_2d(axes).ravel()

    for index, key in enumerate(keys):
        subject = subjects[key]
        image = np.asarray(load_derived(key, FLAIR_N4).dataobj, dtype=float)
        mask = load_derived_mask(key, spec["mask"])
        head = load_derived_mask(key, HEAD_MASK)
        z = _slice_with_most_lesion(subject)
        vmax = float(np.percentile(image[head], 99.5))

        axis = axes[index]
        axis.imshow(image[:, :, z].T[::-1], cmap="gray", vmin=0, vmax=vmax)
        axis.contour(mask[:, :, z].T[::-1], colors=spec["colour"], linewidths=0.5)
        axis.set_title(f"{subject.site[:4]} {subject.subject_id}", fontsize=5.5, pad=1.5)
        axis.axis("off")

    for axis in axes[len(keys):]:
        axis.axis("off")

    fig.suptitle(
        f"{spec['label']} — all {len(keys)} subjects (commit {commit[:8]}). "
        f"Scan for anything unlike its neighbours.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = FIGURES_DIR / f"stage{spec['stage']}_contact_sheet.png"
    fig.savefig(path, dpi=135)
    plt.close(fig)
    write_manifest(path, generating_script=f"preprocessing/{SCRIPT_NAME}.py")
    logger.info("wrote %s", path)


def worst_cases(table: pd.DataFrame, spec: dict, commit: str, logger, n: int = 6) -> None:
    """The n subjects ranked worst by this stage's QC metric, shown large."""
    subjects = subjects_by_key()
    ranked = table.nlargest(n, spec["rank_by"])
    logger.info("worst %d by %s:\n%s", n, spec["rank_by"],
                ranked[["subject_key", "site", spec["rank_by"]]].to_string(index=False))

    fig, axes = plt.subplots(2, n, figsize=(2.9 * n, 6.4))
    axes = np.atleast_2d(axes)
    for column, (_, row) in enumerate(ranked.iterrows()):
        key = row["subject_key"]
        subject = subjects[key]
        image = np.asarray(load_derived(key, FLAIR_N4).dataobj, dtype=float)
        mask = load_derived_mask(key, spec["mask"])
        head = load_derived_mask(key, HEAD_MASK)
        z = _slice_with_most_lesion(subject)
        vmax = float(np.percentile(image[head], 99.5))

        axes[0, column].imshow(image[:, :, z].T[::-1], cmap="gray", vmin=0, vmax=vmax)
        axes[0, column].contour(mask[:, :, z].T[::-1], colors=spec["colour"], linewidths=0.9)
        axes[0, column].set_title(
            f"{subject.site} {subject.subject_id}\n{spec['rank_by']}={row[spec['rank_by']]:.4f}",
            fontsize=8,
        )

        y = image.shape[0] // 2
        aspect = 3.0 / 1.0
        axes[1, column].imshow(image[y, :, :].T[::-1], cmap="gray", vmin=0, vmax=vmax, aspect=aspect)
        axes[1, column].contour(mask[y, :, :].T[::-1], colors=spec["colour"], linewidths=0.9)
        axes[1, column].set_title("sagittal", fontsize=8)

        for r in (0, 1):
            axes[r, column].axis("off")

    fig.suptitle(
        f"{spec['label']} — the {n} WORST subjects by {spec['rank_by']} "
        f"(commit {commit[:8]}). If a failure exists it is most likely here.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIGURES_DIR / f"stage{spec['stage']}_worst_cases.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    write_manifest(path, generating_script=f"preprocessing/{SCRIPT_NAME}.py")
    logger.info("wrote %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=int, required=True, choices=sorted(STAGES))
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(f"{SCRIPT_NAME}_stage{args.stage}", OUTPUTS_DIR)

    spec = dict(STAGES[args.stage], stage=args.stage)
    qc_path = OUTPUTS_DIR / spec["qc_csv"]
    if not qc_path.exists():
        raise SystemExit(f"{qc_path} not found — run the stage first.")

    table = pd.read_csv(qc_path)
    commit = get_git_commit_hash()
    contact_sheet(table, spec, commit, logger)
    worst_cases(table, spec, commit, logger)


if __name__ == "__main__":
    main()
