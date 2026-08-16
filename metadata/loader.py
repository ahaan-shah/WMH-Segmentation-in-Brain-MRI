"""Single entry point for discovering and reading subjects (CLAUDE.md Section 7).

Nothing else in the codebase should walk data/raw/ directly or index the raw
mask array with anything other than load_wmh_mask(). Centralising both here
is what makes CLAUDE.md Decision #1 (label 2 is excluded — WMH mask is
`mask == 1`, never `mask > 0`) structurally impossible to get wrong rather
than merely remembered.

Subject discovery is glob-based (search for every wmh.nii.gz under
data/raw/{training,test}) rather than assuming a fixed directory depth,
because Amsterdam has an extra scanner-name directory level that Utrecht and
Singapore don't, and the test split has two more scanner subdirectories than
training does (CLAUDE.md Section 13 log). A fixed-depth walk would silently
miss or misparse those.
"""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np

from metadata.config import DATA_RAW, LABEL_WMH, SCANNER_METADATA


@dataclass
class Subject:
    subject_key: str  # globally unique: f"{split}_{site}_{scanner_tag}_{subject_id}"
    split: str  # "training" or "test"
    site: str  # "Utrecht", "Singapore", "Amsterdam"
    scanner_dir: str | None  # intermediate directory name, or None if site is flat
    subject_id: str  # raw folder name — NOT globally unique on its own
    subject_dir: Path
    flair_path: Path  # orig/FLAIR.nii.gz — untouched, this project's working file
    t1_path: Path  # orig/T1.nii.gz — 2D T1, pre-registered to FLAIR
    mask_path: Path  # wmh.nii.gz — reference standard, in FLAIR space
    flair_pre_path: Path  # pre/FLAIR.nii.gz — organisers' SPM12 bias-corrected version
    t1_pre_path: Path
    reg_t1_to_flair_path: Path | None
    has_flair_mask_extra: bool  # Amsterdam/GE3T-only undocumented extra file
    has_t1_mask_extra: bool


def discover_subjects() -> list[Subject]:
    """Find every subject under data/raw/{training,test} by locating wmh.nii.gz files."""
    subjects = []
    for wmh_path in sorted(DATA_RAW.rglob("wmh.nii.gz")):
        rel_parts = wmh_path.relative_to(DATA_RAW).parts  # e.g. ('training','Utrecht','4','wmh.nii.gz')
        split = rel_parts[0]
        site = rel_parts[1]

        if len(rel_parts) == 4:
            scanner_dir = None
            subject_id = rel_parts[2]
        elif len(rel_parts) == 5:
            scanner_dir = rel_parts[2]
            subject_id = rel_parts[3]
        else:
            raise ValueError(f"Unexpected subject path depth for {wmh_path}: {rel_parts}")

        subject_dir = wmh_path.parent
        scanner_tag = scanner_dir if scanner_dir is not None else site
        subject_key = f"{split}_{site}_{scanner_tag}_{subject_id}".replace(" ", "_")

        orig = subject_dir / "orig"
        pre = subject_dir / "pre"
        reg_path = orig / "reg_3DT1_to_FLAIR.txt"

        subjects.append(
            Subject(
                subject_key=subject_key,
                split=split,
                site=site,
                scanner_dir=scanner_dir,
                subject_id=subject_id,
                subject_dir=subject_dir,
                flair_path=orig / "FLAIR.nii.gz",
                t1_path=orig / "T1.nii.gz",
                mask_path=wmh_path,
                flair_pre_path=pre / "FLAIR.nii.gz",
                t1_pre_path=pre / "T1.nii.gz",
                reg_t1_to_flair_path=reg_path if reg_path.exists() else None,
                has_flair_mask_extra=(orig / "FLAIR_mask.nii.gz").exists(),
                has_t1_mask_extra=(orig / "T1_mask.nii.gz").exists(),
            )
        )
    return subjects


def get_scanner_metadata(site: str, scanner_dir: str | None) -> dict:
    """Look up best-effort scanner metadata; never guesses beyond what's recorded."""
    key = (site, scanner_dir)
    if key in SCANNER_METADATA:
        return SCANNER_METADATA[key]
    return {"scanner": "unknown", "field_strength_t": None, "vendor": "unknown", "source": "not recorded"}


def load_nifti(path: Path) -> nib.Nifti1Image:
    """Load a NIfTI file. Fails loud: raises if the file is missing or unreadable."""
    if not path.exists():
        raise FileNotFoundError(f"Expected NIfTI file not found: {path}")
    img = nib.load(path)
    img = nib.as_closest_canonical(img)  # reorientation only (axis reorder/flip), never a resample
    return img


def load_wmh_mask(mask_path: Path) -> np.ndarray:
    """Return the WMH mask as a boolean array: True only where the raw label == 1.

    This is the ONLY sanctioned way to read a mask array in this codebase.
    Label 2 ("other pathology") is never included — CLAUDE.md Decision #1.
    """
    img = load_nifti(mask_path)
    raw = np.asarray(img.dataobj)
    return raw == LABEL_WMH


def load_raw_mask_array(mask_path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Return the raw (unfiltered) mask array and its image, for validation/metadata
    code that legitimately needs to inspect label 2 rather than exclude it."""
    img = load_nifti(mask_path)
    raw = np.asarray(img.dataobj)
    return raw, img
