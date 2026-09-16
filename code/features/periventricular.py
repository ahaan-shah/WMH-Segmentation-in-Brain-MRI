"""R5 — split lesions into periventricular and deep white matter.

The clinical distinction the problem statement asks for. Periventricular WMH
(hugging the ventricle wall) and deep WMH (out in the white matter) have
different associations and different prognostic weight, which is why they are
reported separately rather than as one lesion burden.

**The rule.** A lesion voxel is periventricular if it lies within 10 mm of the
lateral ventricles, deep otherwise. 10 mm follows DeCarli et al. 2005, and is
the same cut-off UK Biobank and FSL BIANCA use. The threshold and its
sensitivity range live in `metadata/dataset.yaml`, not here.

**Two ways this goes silently wrong, both guarded:**

1. **Distance in voxels instead of millimetres.** FLAIR here is 3.00 mm through
   plane against 0.56-1.30 mm in plane — up to 5.4x anisotropy — so an unscaled
   distance transform is wrong by that factor along one axis, and "10 mm" ends
   up meaning three different things in three directions. `sampling=spacing_mm`
   is mandatory, and `checks/test_phantom.py` asserts the failure explicitly.

2. **Lesions absorbed into the ventricle label.** SynthSeg assigns a mean 2.96%
   of true WMH voxels to the ventricle (measured, `sweep_synthseg.py`).
   Confluent periventricular lesions are continuous with the ventricle and have
   CSF-like T1 intensity, so this is expected rather than a tool failure. Left
   alone it biases the split the wrong way twice over: those voxels vanish from
   the lesion set, *and* they inflate the structure that distance is measured
   from, pushing the 10 mm envelope outward. A voxel cannot be both lesion and
   cerebrospinal fluid, so the lesion mask wins and is removed from the
   ventricles before the transform.

Everything here takes arrays and returns arrays — no paths, no I/O — so it is
testable against a phantom with a known answer (CLAUDE.md Section 5.3).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def clean_ventricles(ventricle_mask: np.ndarray, lesion_mask: np.ndarray) -> np.ndarray:
    """Remove lesion voxels from the ventricle mask. See point 2 above."""
    return np.asarray(ventricle_mask, dtype=bool) & ~np.asarray(lesion_mask, dtype=bool)


def distance_to_ventricles_mm(ventricle_mask: np.ndarray,
                              spacing_mm) -> np.ndarray:
    """Euclidean distance from every voxel to the nearest ventricle voxel, in mm.

    `sampling` is what makes the result millimetres rather than voxel counts.
    Omitting it is the single most likely silent bug in this project.
    """
    ventricle_mask = np.asarray(ventricle_mask, dtype=bool)
    if not ventricle_mask.any():
        raise ValueError(
            "empty ventricle mask — every lesion would be labelled deep by "
            "default. Refusing to produce that silently.")
    spacing_mm = np.asarray(spacing_mm, dtype=float)
    if spacing_mm.shape != (3,) or not np.all(spacing_mm > 0):
        raise ValueError(f"spacing_mm must be three positive values, got {spacing_mm}")
    return distance_transform_edt(~ventricle_mask, sampling=spacing_mm)


def split_periventricular_deep(lesion_mask: np.ndarray,
                               ventricle_mask: np.ndarray,
                               spacing_mm,
                               threshold_mm: float = 10.0) -> dict:
    """Split `lesion_mask` at `threshold_mm` from the ventricles.

    Returns the two boolean masks plus the distance map, so callers can run
    several thresholds without recomputing the transform.
    """
    lesion_mask = np.asarray(lesion_mask, dtype=bool)
    cleaned = clean_ventricles(ventricle_mask, lesion_mask)
    distance = distance_to_ventricles_mm(cleaned, spacing_mm)
    periventricular = lesion_mask & (distance <= threshold_mm)
    deep = lesion_mask & (distance > threshold_mm)
    # Every lesion voxel lands in exactly one class — no voxel lost, none double
    # counted. Cheap to assert, and it would catch any future edit that changed
    # the comparison operators to overlap or leave a gap.
    assert int(periventricular.sum()) + int(deep.sum()) == int(lesion_mask.sum())
    return {"periventricular": periventricular, "deep": deep,
            "distance_mm": distance, "ventricles_used": cleaned}


def summarise(lesion_mask: np.ndarray,
              ventricle_mask: np.ndarray,
              spacing_mm,
              thresholds_mm=(5.0, 10.0, 15.0),
              primary_mm: float = 10.0) -> dict:
    """Volumes and counts at the primary threshold, plus the sensitivity sweep.

    Reporting 5 and 15 mm alongside 10 mm turns a hard-coded constant into a
    stated analysis — the split is only worth reporting if we know how much it
    moves when the cut-off does.
    """
    lesion_mask = np.asarray(lesion_mask, dtype=bool)
    voxel_mm3 = float(np.prod(np.asarray(spacing_mm, dtype=float)))
    total = int(lesion_mask.sum())

    cleaned = clean_ventricles(ventricle_mask, lesion_mask)
    distance = distance_to_ventricles_mm(cleaned, spacing_mm)

    out = {"lesion_voxels": total,
           "lesion_volume_ml": total * voxel_mm3 / 1000.0}
    for threshold in sorted({*thresholds_mm, primary_mm}):
        pv = int((lesion_mask & (distance <= threshold)).sum())
        dp = total - pv
        tag = "" if threshold == primary_mm else f"_at_{threshold:g}mm"
        out[f"periventricular_ml{tag}"] = pv * voxel_mm3 / 1000.0
        out[f"deep_ml{tag}"] = dp * voxel_mm3 / 1000.0
        out[f"periventricular_fraction{tag}"] = (pv / total) if total else float("nan")
    return out
