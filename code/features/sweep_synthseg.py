"""Which image should SynthSeg segment — T1, FLAIR, or our preprocessed FLAIR?

R5 (periventricular vs deep) needs a ventricle mask, and Week 1 recorded
SynthSeg as the preferred route the moment FreeSurfer became available. It now
is. What was never decided is **which input to feed it**, and that choice is not
obvious:

- **raw T1** — SynthSeg's native modality, and W1.8 confirmed T1 already shares
  FLAIR's exact shape and affine in this dataset, so no registration is needed.
- **raw FLAIR** — SynthSeg is advertised as contrast-agnostic, trained on
  synthetic data precisely so it runs on any sequence. If true, this removes the
  T1 dependency entirely.
- **flair_norm** — our Week 2 output: skull-stripped, bias-corrected,
  WM-referenced. Cleaner, but SynthSeg was trained on whole heads, so brain
  extraction may be out of distribution for it.

**The rule, fixed before any result is seen:**

1. **Geometry is a hard gate.** The output must land on the subject's FLAIR grid
   exactly, because that is where the WMH masks live and where the 10 mm
   distance transform must be computed. `--keepgeom` is passed for this; it is
   verified rather than trusted.
2. **Anatomical plausibility.** Lateral ventricle volume must be physiologically
   sensible. The band is deliberately wide — 5-150 mL — because this cohort is
   70.1 +- 9.3 years old *with* cerebrovascular disease, and ventricles enlarge
   substantially with both age and WMH burden. A tighter "normal adult" band of
   10-40 mL would reject genuinely correct segmentations on the very subjects
   this project is about. Narrowness here would be false rigour.
3. **Cross-site consistency.** The winner must work on all three scanners. An
   input that succeeds at Utrecht and fails at Amsterdam is useless to us.
4. **Agreement.** Where inputs disagree wildly on the same brain, at most one can
   be right; that disagreement is itself the diagnostic.

Ventricle labels are FreeSurfer's standard ones: 4/43 lateral (left/right),
5/44 inferior lateral. The lateral pair is what DeCarli's 10 mm rule refers to.

    code/.venv/bin/python -m features.sweep_synthseg
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from metadata.config import CODE_ROOT, PROJECT_ROOT
from metadata.derived import FLAIR_NORM, derived_path
from metadata.geometry import assert_same_geometry
from metadata.loader import load_nifti, load_split, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "sweep_synthseg"
OUTPUTS_DIR = CODE_ROOT / "features" / "outputs"
WORK_DIR = PROJECT_ROOT / "data" / "interim" / "synthseg_sweep"
RESULTS_CSV = OUTPUTS_DIR / "synthseg_input_sweep.csv"

SYNTHSEG = Path("/opt/freesurfer/bin/mri_synthseg")

LATERAL = (4, 43)
INFERIOR_LATERAL = (5, 44)
THIRD_FOURTH = (14, 15)

PLAUSIBLE_ML = (5.0, 150.0)   # see rule 2 in the module docstring


def run_synthseg(input_path: Path, output_path: Path, *, threads: int) -> float:
    """Run SynthSeg on one image. Returns wall-clock seconds."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    result = subprocess.run(
        [str(SYNTHSEG), "--i", str(input_path), "--o", str(output_path),
         "--keepgeom", "--cpu", "--threads", str(threads)],
        capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"SynthSeg failed on {input_path.name} (exit {result.returncode})\n"
            f"{result.stdout[-1500:]}\n{result.stderr[-1500:]}")
    return time.time() - started


def volume_ml(seg: np.ndarray, labels, voxel_mm3: float) -> float:
    return float(np.isin(seg, labels).sum() * voxel_mm3 / 1000.0)


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    if not SYNTHSEG.exists():
        raise SystemExit(f"mri_synthseg not found at {SYNTHSEG}")

    subjects = subjects_by_key()
    # One subject per site, from the TRAINING split — the method choice must not
    # be made by looking at held-out data.
    train = load_split("train")
    chosen, seen = [], set()
    for key in train:
        site = subjects[key].site
        if site not in seen:
            seen.add(site)
            chosen.append(key)
    logger.info("input sweep on %d subjects, one per site: %s", len(chosen), chosen)
    logger.info("acceptance: geometry must match FLAIR exactly; lateral ventricle "
                "volume within %.0f-%.0f mL on every site", *PLAUSIBLE_ML)

    rows = []
    for key in chosen:
        subject = subjects[key]
        flair = load_nifti(subject.flair_path)
        candidates = {
            "raw_T1": subject.t1_path,
            "raw_FLAIR": subject.flair_path,
            "flair_norm": derived_path(key, FLAIR_NORM),
        }
        for name, path in candidates.items():
            if not path.exists():
                logger.warning("  %s / %s: input missing, skipped", key, name)
                continue
            out = WORK_DIR / key / f"{name}_seg.nii.gz"
            try:
                seconds = run_synthseg(path, out, threads=4)
            except RuntimeError as exc:
                logger.error("  %s / %-10s FAILED: %s", key, name, str(exc)[:300])
                rows.append({"subject_key": key, "site": subject.site, "input": name,
                             "failed": True})
                continue

            # Through load_nifti, NOT nib.load: it applies as_closest_canonical,
            # and `flair` above was loaded the same way. Comparing a canonical
            # FLAIR against a non-canonical segmentation reports a mismatch of a
            # whole field of view on a pair that is in fact bit-identical on disk.
            seg_img = load_nifti(out)
            seg = np.asarray(seg_img.dataobj).astype(np.int32)
            spacing = np.abs(np.diag(seg_img.affine))[:3]
            voxel_mm3 = float(np.prod(spacing))

            # Catch ONLY the geometry failure. A broad `except Exception` here
            # previously swallowed a TypeError from calling this with the wrong
            # keyword names and reported it as a geometry mismatch on all nine
            # configurations — a programming error dressed up as a data finding,
            # which is the exact failure CLAUDE.md 5.2 exists to prevent.
            try:
                assert_same_geometry(seg_img, flair,
                                     context=f"{name} segmentation vs FLAIR")
                geometry_ok = True
            except AssertionError as exc:
                geometry_ok = False
                logger.warning("  %s / %-10s geometry MISMATCH: %s",
                               key, name, str(exc)[:200])

            lateral = volume_ml(seg, LATERAL, voxel_mm3)
            inferior = volume_ml(seg, INFERIOR_LATERAL, voxel_mm3)
            third_fourth = volume_ml(seg, THIRD_FOURTH, voxel_mm3)
            plausible = PLAUSIBLE_ML[0] <= lateral <= PLAUSIBLE_ML[1]

            rows.append({
                "subject_key": key, "site": subject.site, "input": name,
                "failed": False, "geometry_ok": geometry_ok,
                "lateral_ml": round(lateral, 2),
                "inferior_lateral_ml": round(inferior, 2),
                "third_fourth_ml": round(third_fourth, 2),
                "plausible": plausible, "seconds": round(seconds, 1),
                "shape_match": seg.shape == flair.shape,
            })
            logger.info("  %-34s %-10s  lateral %6.2f mL  geom_ok=%-5s plausible=%-5s "
                        "%5.0fs", key, name, lateral, geometry_ok, plausible, seconds)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_CSV, index=False)
    write_manifest(RESULTS_CSV, generating_script=f"code/features/{SCRIPT_NAME}.py")

    logger.info("=" * 78)
    ok = table[(~table.failed)]
    if not ok.empty:
        summary = ok.groupby("input").agg(
            sites=("site", "nunique"),
            geometry_ok=("geometry_ok", "all"),
            all_plausible=("plausible", "all"),
            mean_lateral_ml=("lateral_ml", "mean"),
            mean_seconds=("seconds", "mean"),
        ).round(2)
        logger.info("PER-INPUT SUMMARY\n%s", summary.to_string())
        eligible = summary[(summary.geometry_ok) & (summary.all_plausible)
                           & (summary.sites == ok.site.nunique())]
        logger.info("")
        if eligible.empty:
            logger.warning("NO input passed every gate. R5 must fall back to the "
                           "Week 1 FLAIR-darkness heuristic, and the report must "
                           "say so plainly.")
        else:
            logger.info("ELIGIBLE INPUTS: %s", list(eligible.index))
            logger.info("Extrapolated cost for 60 subjects, per input (minutes):")
            for name, row in summary.iterrows():
                logger.info("    %-12s %.0f min", name, row.mean_seconds * 60 / 60)
    logger.info("written: %s", RESULTS_CSV.name)


if __name__ == "__main__":
    main()
