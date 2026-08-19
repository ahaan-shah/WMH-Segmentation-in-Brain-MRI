"""Stage 3 — skull stripping (R1), and the two-sided test that judges it.

**Why the tool choice was reopened.** ROADMAP 5.2 named SynthStrip as primary
with FSL FAST as fallback. Neither is installable in this environment — no
FreeSurfer, no FSL, no Docker/Singularity — so that section has no working
implementation. The two candidates here are pip-installable, run on CPU, and
are competed against each other on measured criteria rather than chosen by
reputation:

- **HD-BET** — nnU-Net based, trained on multi-sequence clinical MRI.
- **deepbet** — a fast lightweight CNN.

Each is run two ways, giving four configurations:

- **on FLAIR** (our N4-corrected FLAIR), the direct route.
- **on T1, transferred** — `orig/T1.nii.gz` is already resampled onto the FLAIR
  grid by the challenge organisers, sharing its shape and affine exactly, so a
  mask computed on T1 applies to FLAIR with no registration at all. T1 is the
  contrast these tools are strongest on. The cost is that T1 and FLAIR brain
  boundaries differ slightly, hence a small dilation.

**Why two acceptance criteria, not one.** Skull stripping fails in two opposite
directions and a single metric only sees one of them:

1. **Over-stripping** — the mask eats into brain and removes real lesions,
   especially juxtacortical ones at the brain edge. Caught by WMH retention:
   the reference standard is only ever drawn inside the brain, so any reference
   lesion voxel outside our brain mask is proof the mask cut too deep. This
   costs recall in Week 3 that no later stage can recover.

2. **Under-stripping** — skull and scalp survive. Completely invisible to
   criterion 1, and the more dangerous failure here, because scalp fat is
   extremely bright on FLAIR. Measured earlier on this data: at Utrecht, zero
   percent of the brightest 1% of head voxels are lesion, and thresholding at
   the true WMH median admits a median of 19x (worst case 583x) as many
   non-lesion voxels as lesion voxels. Anything left behind gets segmented as
   disease in Week 3.

`shallow_fraction` is the under-stripping detector. Skull plus scalp is several
millimetres thick everywhere, so a correct brain mask should contain almost no
voxels within a few mm of the outer head surface. Depth is measured with a
distance transform carrying `sampling=spacing`, in millimetres — never in
voxels, which on a 3 mm slice grid would be wrong by up to 5.4x.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

# Configuration names, used as dict keys, CSV values and artefact suffixes.
HDBET_FLAIR = "hdbet_flair"
HDBET_T1 = "hdbet_t1"
DEEPBET_FLAIR = "deepbet_flair"
DEEPBET_T1 = "deepbet_t1"
CONFIGURATIONS = (HDBET_FLAIR, HDBET_T1, DEEPBET_FLAIR, DEEPBET_T1)


def _load_mask_like(mask_path: Path, reference_img: nib.Nifti1Image) -> np.ndarray:
    """Load a tool's output mask, canonicalised, and check it matches the grid."""
    from metadata.geometry import assert_same_geometry
    from metadata.loader import load_nifti

    mask_img = load_nifti(Path(mask_path))
    assert_same_geometry(mask_img, reference_img, context=f"stripper output {mask_path}")
    return np.asarray(mask_img.dataobj) > 0


def run_deepbet(input_path: Path, reference_img: nib.Nifti1Image, *, n_dilate: int = 0) -> np.ndarray:
    """Brain mask from deepbet. `n_dilate` is applied by the tool itself."""
    from deepbet import run_bet

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "mask.nii.gz"
        run_bet([str(input_path)], mask_paths=[str(out)], n_dilate=n_dilate, no_gpu=True)
        if not out.exists():
            raise RuntimeError(f"deepbet produced no mask for {input_path}")
        return _load_mask_like(out, reference_img)


def run_hdbet(input_path: Path, reference_img: nib.Nifti1Image, predictor) -> np.ndarray:
    """Brain mask from HD-BET, reusing an already-initialised predictor.

    The predictor holds the loaded network; building it costs far more than a
    single inference, so callers construct it once and pass it in.
    """
    from HD_BET.hd_bet_prediction import hdbet_predict

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "brain.nii.gz"
        hdbet_predict(
            str(input_path), str(out), predictor,
            keep_brain_mask=True, compute_brain_extracted_image=False,
        )
        candidates = sorted(Path(tmp).glob("*bet*.nii.gz")) or sorted(Path(tmp).glob("*.nii.gz"))
        if not candidates:
            raise RuntimeError(f"HD-BET produced no mask for {input_path}")
        return _load_mask_like(candidates[0], reference_img)


def dilate_in_plane(mask: np.ndarray, radius_voxels: int) -> np.ndarray:
    """In-plane dilation, for transferring a T1-derived mask onto FLAIR.

    In-plane only, for the same reason as everywhere else in this pipeline: one
    voxel through-plane is a 3 mm anatomical step against ~1 mm in-plane.
    """
    if radius_voxels <= 0:
        return mask
    from preprocessing.head_mask import in_plane_structure

    return ndi.binary_dilation(mask, structure=in_plane_structure(radius_voxels))


def bright_tissue_retention(
    image: np.ndarray,
    brain_mask: np.ndarray,
    head_mask: np.ndarray,
    *,
    percentile: float = 99.0,
    lesion_mask: np.ndarray | None = None,
) -> dict:
    """Direct under-stripping test: how much of the brightest head tissue survived?

    Added after `shallow_fraction` was found to be confounded (see its note in
    `brain_mask_quality`). This measures the thing that actually matters rather
    than a geometric proxy for it: scalp fat is the brightest tissue in a FLAIR
    head, so if a brain mask has retained scalp, the brightest head voxels will
    be inside it.

    **It must be read together with `bright_that_is_wmh`.** "Bright" is only a
    synonym for "scalp" where lesions are not themselves the brightest thing.
    Measured on this data, at Utrecht 0% of the brightest 1% of head voxels are
    reference WMH — so retaining them there is unambiguously a failure. At
    Amsterdam and Singapore a large share of that same top 1% IS reference
    lesion, so retaining it is correct. A metric that ignored this would score a
    correct mask as broken on two of the three sites.
    """
    if not brain_mask.any() or not head_mask.any():
        raise ValueError("Empty brain or head mask")

    threshold = float(np.percentile(image[head_mask], percentile))
    bright = head_mask & (image >= threshold)
    if not bright.any():
        raise ValueError(f"No voxels above the {percentile}th percentile of head intensity")

    result = {
        "bright_percentile": percentile,
        "bright_retained_fraction": float(bright[brain_mask].sum() / bright.sum()),
        "bright_fraction_of_brain": float((bright & brain_mask).sum() / brain_mask.sum()),
    }
    if lesion_mask is not None and lesion_mask.any():
        # The disambiguator: where this is high, bright means lesion, not scalp.
        result["bright_that_is_wmh"] = float(lesion_mask[bright].mean())
    return result


def brain_mask_quality(
    brain_mask: np.ndarray,
    head_mask: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    lesion_mask: np.ndarray | None = None,
    shallow_depth_mm: float = 5.0,
) -> dict:
    """Score a candidate brain mask against both failure directions.

    Returns `wmh_retained_fraction` (over-stripping; 1.0 is perfect) and
    `shallow_fraction` (under-stripping; lower is better), plus volumes for
    plausibility.
    """
    if brain_mask.shape != head_mask.shape:
        raise AssertionError(
            f"brain mask shape {brain_mask.shape} != head mask shape {head_mask.shape}"
        )
    if not brain_mask.any():
        raise ValueError("Brain mask is empty — the stripper failed on this subject")

    voxel_volume_mm3 = float(np.prod(spacing))

    # Depth of every head voxel below the outer head surface, in MILLIMETRES.
    # sampling= is not optional: without it this is measured in voxels and the
    # 5 mm criterion means 5 voxels, i.e. 15 mm along z.
    depth_mm = ndi.distance_transform_edt(head_mask, sampling=spacing)
    shallow = brain_mask & (depth_mm < shallow_depth_mm)

    quality = {
        "brain_volume_ml": float(brain_mask.sum() * voxel_volume_mm3 / 1000.0),
        "head_volume_ml": float(head_mask.sum() * voxel_volume_mm3 / 1000.0),
        "brain_fraction_of_head": float(brain_mask.sum() / head_mask.sum()),
        # Under-stripping proxy — CONFOUNDED, read the caveat before using it.
        #
        # Intended as "brain voxels suspiciously close to the outer head
        # surface", i.e. retained skull/scalp. Verified against the images
        # afterwards and it does NOT measure only that. At the skull base and
        # posterior fossa the FLAIR signal outside the brain is very dark
        # (cortical bone, mastoid air), so Stage 1's multi-Otsu head mask never
        # captures a scalp shell there and its boundary hugs the brain —
        # measured brain-surface depth reaches 0.0 mm at the 5th percentile on
        # the worst subject. A large share of `shallow_fraction` is therefore
        # Stage 1 head-mask thinness, not Stage 3 brain-mask error, plus
        # genuinely thin skull in temporal regions.
        #
        # It is kept because it was applied identically to all four candidate
        # configurations, so the Stage 3 ranking it produced is still a fair
        # comparison. It should NOT be read as an absolute statement that
        # N% of the brain mask is retained scalp. Use
        # bright_tissue_retention() for that.
        "shallow_fraction": float(shallow.sum() / brain_mask.sum()),
        "shallow_volume_ml": float(shallow.sum() * voxel_volume_mm3 / 1000.0),
        "mean_depth_mm": float(depth_mm[brain_mask].mean()),
        "outside_head_voxels": int((brain_mask & ~head_mask).sum()),
    }

    if lesion_mask is not None:
        if not lesion_mask.any():
            raise ValueError("Reference lesion mask is empty — retention is undefined")
        # Over-stripping detector: the reference standard exists only inside the
        # brain, so a lesion voxel outside our mask means we cut into brain.
        quality["wmh_retained_fraction"] = float(brain_mask[lesion_mask].mean())
        quality["wmh_lost_voxels"] = int((~brain_mask & lesion_mask).sum())
    return quality


def is_tool_available(name: str) -> bool:
    """Whether a stripper can actually run, without importing torch eagerly."""
    if name.startswith("deepbet"):
        return shutil.which("deepbet-cli") is not None or _importable("deepbet")
    if name.startswith("hdbet"):
        return _importable("HD_BET")
    raise ValueError(f"Unknown stripper {name!r}")


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None
