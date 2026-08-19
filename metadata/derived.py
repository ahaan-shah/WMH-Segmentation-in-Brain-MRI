"""The single sanctioned write path for derived image artefacts (CLAUDE.md
Sections 4, 5.1, 5.4).

Everything Week 2 onwards produces per subject — brain masks, bias-corrected
FLAIR, tissue masks, normalised FLAIR, and in later weeks predicted lesion
masks — is written through `save_derived()`. Centralising it is what enforces
four separate rules at once, none of which survive being left to memory:

- **Raw data stays immutable.** Output is rooted at `data/interim/`; nothing
  here can write into `data/raw/`.
- **Everything stays in FLAIR space.** Every save asserts the array's geometry
  against that subject's own FLAIR before writing (ROADMAP Section 3's
  invariant), so a resampled or mis-shaped array fails at the boundary that
  produced it instead of surfacing as a bad number in Week 7.
- **Files land in the dataset's native orientation.** Arrays are computed in
  canonical RAS but written back through `geometry.restore_orientation()`, so
  the vendored official scorer — which reads raw and never canonicalises — can
  consume our predictions directly. See `metadata/geometry.py` for why this is
  the highest-consequence trap in the pipeline.
- **Provenance is automatic.** Each artefact gets its `.json` sidecar recording
  the generating script, commit hash, timestamp and config, so the Section 5.1
  test ("could a stranger regenerate every file?") is satisfied by construction.

Idempotence (Section 5.4): saves overwrite in place. No appending, no
timestamped filenames accumulating across runs.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from metadata.config import DATA_INTERIM
from metadata.geometry import assert_same_geometry, restore_orientation
from metadata.loader import Subject, load_nifti
from metadata.provenance import write_manifest

# Canonical artefact names, so a typo is an ImportError rather than a silently
# orphaned file. Stage numbering matches the Week 2 checklist.
BRAIN_MASK = "brain_mask"  # Stage 3 — skull stripping (R1)
HEAD_MASK = "head_mask"  # Stage 1 — rough mask, input to N4 only
FLAIR_N4 = "flair_n4"  # Stage 2 — our own bias field correction (R3)
BIAS_FIELD = "bias_field"  # Stage 2 — the estimated field, a report figure
TISSUE_SEG = "tissue_seg"  # Stage 4 — WM/GM/CSF label map
WM_MASK = "wm_mask"  # Stage 4 — white matter, needed by W3 and W4
FLAIR_NORM = "flair_norm"  # Stage 5 — normalised FLAIR (R2)
FLAIR_DENOISED = "flair_denoised"  # Stage 6 — evaluated branch, kept for W7 SSIM
FLAIR_CLAHE = "flair_clahe"  # Stage 6 — evaluated branch


def derived_dir(subject_key: str) -> Path:
    """Directory holding one subject's derived artefacts."""
    return DATA_INTERIM / subject_key


def derived_path(subject_key: str, name: str) -> Path:
    """Resolved path of a single derived artefact."""
    return derived_dir(subject_key) / f"{name}.nii.gz"


def derived_exists(subject_key: str, name: str) -> bool:
    return derived_path(subject_key, name).exists()


def _resolve_dtype(array: np.ndarray, dtype: np.dtype | str | None) -> np.dtype:
    """Masks store as uint8, continuous images as float32.

    Explicit rather than inherited: a boolean mask saved as float64 quadruples
    the size of ~170 volumes for no benefit, and an integer label map saved as
    float invites `==` comparisons against floats downstream.
    """
    if dtype is not None:
        return np.dtype(dtype)
    if array.dtype == bool:
        return np.dtype(np.uint8)
    if np.issubdtype(array.dtype, np.integer):
        return np.dtype(np.uint8) if array.max() <= 255 and array.min() >= 0 else np.dtype(np.int16)
    return np.dtype(np.float32)


def save_derived(
    array_canonical: np.ndarray,
    subject: Subject,
    name: str,
    *,
    generating_script: str,
    dtype: np.dtype | str | None = None,
    extra: dict | None = None,
) -> Path:
    """Write a derived volume for one subject, in FLAIR space, with provenance.

    `array_canonical` must be in canonical RAS orientation — i.e. computed from
    arrays obtained through `loader.load_nifti()`. It is written to disk in the
    raw FLAIR's native orientation (LPS for every subject in this dataset).

    Returns the path written. Raises if the array does not share the subject's
    FLAIR geometry — failing loud at the boundary that produced the array,
    rather than letting a mis-shaped volume propagate (Section 5.2).
    """
    array_canonical = np.asarray(array_canonical)

    flair_canonical = load_nifti(subject.flair_path)  # canonicalised
    if tuple(array_canonical.shape[:3]) != tuple(flair_canonical.shape[:3]):
        raise AssertionError(
            f"Derived array shape {array_canonical.shape} does not match canonical "
            f"FLAIR shape {flair_canonical.shape} for {subject.subject_key} "
            f"while saving '{name}'. Everything stays on the FLAIR grid — "
            f"no resampling (ROADMAP Section 3)."
        )

    flair_raw = nib.load(subject.flair_path)  # NOT canonicalised: the on-disk orientation

    # Cast before reorienting, not after. NIfTI has no boolean data type, so a
    # bool mask must become uint8 before it is ever wrapped in a Nifti1Image.
    target_dtype = _resolve_dtype(array_canonical, dtype)
    out_img = restore_orientation(array_canonical.astype(target_dtype), flair_raw)
    out_img.header.set_zooms(flair_raw.header.get_zooms()[:3])

    # The written file must be indistinguishable from the raw FLAIR in space.
    displacement = assert_same_geometry(
        out_img, flair_raw, context=f"{subject.subject_key}:{name} vs raw FLAIR"
    )

    out_path = derived_path(subject.subject_key, name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, out_path)

    write_manifest(
        out_path,
        generating_script=generating_script,
        extra={
            "subject_key": subject.subject_key,
            "artefact": name,
            "site": subject.site,
            "split": subject.split,
            "source_flair": str(subject.flair_path),
            "shape": list(map(int, out_img.shape[:3])),
            "dtype": str(target_dtype),
            "written_orientation": "".join(nib.aff2axcodes(flair_raw.affine)),
            "computed_orientation": "RAS (canonical)",
            "corner_displacement_mm_vs_raw_flair": displacement,
            **(extra or {}),
        },
    )
    return out_path


def load_derived(subject_key: str, name: str) -> nib.Nifti1Image:
    """Load a derived artefact, canonicalised to RAS to match the compute space."""
    path = derived_path(subject_key, name)
    if not path.exists():
        raise FileNotFoundError(
            f"Derived artefact '{name}' not found for {subject_key}: {path}. "
            f"Has the stage that produces it been run?"
        )
    return load_nifti(path)


def load_derived_mask(subject_key: str, name: str) -> np.ndarray:
    """Load a derived binary mask as a boolean array, canonicalised."""
    return np.asarray(load_derived(subject_key, name).dataobj) > 0
