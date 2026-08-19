"""Stage 5 figures — the direct evidence that R2 was done properly.

ROADMAP 5.4 asks for per-site histograms before and after normalisation. That
pair of plots is the evidence: before, the three sites occupy disjoint intensity
ranges; after, they should overlap. Anything less and a single Week 3 model
cannot work across scanners.

    code/.venv/bin/python -m preprocessing.make_stage5_figures
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metadata.config import NORMALISATION_CONFIG, PROJECT_ROOT
from metadata.derived import (
    BRAIN_MASK,
    FLAIR_NORM,
    TISSUE_SEG,
    load_derived,
    load_derived_mask,
)
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import get_git_commit_hash, write_manifest
from metadata.runlog import setup_logging
from preprocessing.tissue_seg import LABEL_WM

SCRIPT_NAME = "make_stage5_figures"
OUTPUTS_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
COLOURS = {"Amsterdam": "#4C72B0", "Singapore": "#DD8452", "Utrecht": "#55A868"}


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    commit = get_git_commit_hash()
    subjects = subjects_by_key()
    keys = load_split("train")

    raw_by_site: dict[str, list] = {}
    norm_by_site: dict[str, list] = {}
    lesion_norm_by_site: dict[str, list] = {}
    for key in keys:
        subject = subjects[key]
        brain = load_derived_mask(key, BRAIN_MASK)
        raw = np.asarray(load_nifti(subject.flair_path).dataobj, dtype=float)
        norm = np.asarray(load_derived(key, FLAIR_NORM).dataobj, dtype=float)
        lesion = load_wmh_mask(subject.mask_path)
        rng = np.random.default_rng(0)
        inside = np.flatnonzero(brain.ravel())
        sample = rng.choice(inside, size=min(20000, inside.size), replace=False)
        raw_by_site.setdefault(subject.site, []).append(raw.ravel()[sample])
        norm_by_site.setdefault(subject.site, []).append(norm.ravel()[sample])
        if lesion.any():
            lesion_norm_by_site.setdefault(subject.site, []).append(norm[lesion])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    for site, chunks in raw_by_site.items():
        values = np.concatenate(chunks)
        axes[0].hist(values, bins=200, range=(0, np.percentile(values, 99.5)),
                     histtype="step", density=True, label=site, color=COLOURS.get(site), lw=1.4)
    axes[0].set(xlabel="raw FLAIR intensity (arbitrary units)", ylabel="density",
                title="BEFORE — raw FLAIR inside the brain\nthe three sites barely overlap")
    axes[0].legend(fontsize=8)

    for site, chunks in norm_by_site.items():
        values = np.concatenate(chunks)
        axes[1].hist(values, bins=200, range=(0, 2.5), histtype="step", density=True,
                     label=site, color=COLOURS.get(site), lw=1.4)
    axes[1].axvline(1.0, color="k", ls="--", lw=1, label="normal-appearing WM = 1.0")
    axes[1].set(xlabel="normalised intensity", ylabel="density",
                title=f"AFTER — {NORMALISATION_CONFIG['primary']}\nthe distributions should coincide")
    axes[1].legend(fontsize=8)

    for site, chunks in lesion_norm_by_site.items():
        values = np.concatenate(chunks)
        axes[2].hist(values, bins=150, range=(0, 3.0), histtype="step", density=True,
                     label=site, color=COLOURS.get(site), lw=1.4)
    axes[2].axvline(1.0, color="k", ls="--", lw=1)
    axes[2].set(xlabel="normalised intensity", ylabel="density",
                title="AFTER — reference WMH voxels only\nlesions must land at the same place per site")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"Stage 5 — intensity normalisation (R2), {len(keys)} training subjects "
                 f"(commit {commit[:8]})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = FIGURES_DIR / "stage5_normalisation_histograms.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    write_manifest(path, generating_script=f"preprocessing/{SCRIPT_NAME}.py")
    logger.info("wrote %s", path)

    # --- images on a shared scale: the visual form of the same claim --------
    chosen = []
    for site in sorted(COLOURS):
        for key in keys:
            if subjects[key].site == site:
                chosen.append(key)
                break
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.4 * len(chosen), 7.2))
    for column, key in enumerate(chosen):
        subject = subjects[key]
        raw = np.asarray(load_nifti(subject.flair_path).dataobj, dtype=float)
        norm = np.asarray(load_derived(key, FLAIR_NORM).dataobj, dtype=float)
        brain = load_derived_mask(key, BRAIN_MASK)
        lesion = load_wmh_mask(subject.mask_path)
        z = int(np.argmax(lesion.sum(axis=(0, 1))))

        # Raw: a SHARED window across sites, which is the point — it cannot work.
        axes[0, column].imshow(np.where(brain, raw, 0)[:, :, z].T[::-1],
                               cmap="gray", vmin=0, vmax=1200)
        axes[0, column].set_title(f"{subject.site} {subject.subject_id}\nraw, shared window 0-1200",
                                  fontsize=9)
        axes[1, column].imshow(norm[:, :, z].T[::-1], cmap="gray", vmin=0, vmax=2.0)
        axes[1, column].contour(lesion[:, :, z].T[::-1], colors="lime", linewidths=0.7)
        axes[1, column].set_title("normalised, shared window 0-2.0", fontsize=9)
        for row in (0, 1):
            axes[row, column].axis("off")

    fig.suptitle("Same display window applied to every site. Before: unusable. "
                 f"After: comparable.  (commit {commit[:8]})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = FIGURES_DIR / "stage5_shared_window.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    write_manifest(path, generating_script=f"preprocessing/{SCRIPT_NAME}.py")
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
