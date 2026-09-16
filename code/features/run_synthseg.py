"""R5 step 1 — segment the ventricles with SynthSeg, for every subject.

**Input: the raw T1.** Chosen by measurement in `features/sweep_synthseg.py`,
not by preference. All three candidates (raw T1, raw FLAIR, our normalised
FLAIR) produced geometry-exact output on all three sites and agreed on ventricle
volume to within 6%, so the tiebreaker was the thing that actually matters for
R5: **how much true WMH the ventricle label swallows.** Periventricular lesions
sit directly against the ventricle wall, so contamination there corrupts the
10 mm split precisely where most of the lesion burden is.

    mean WMH absorbed into the ventricle mask, 3 subjects, one per site:
        raw T1       2.96%      <- selected
                                (4.10% mean / 16.61% worst once measured on all
                                 60 — see dataset.yaml; n=3 understated it, which
                                 changes the number reported, not the choice)
        flair_norm   4.79%
        raw FLAIR    4.95%      (11.64% on Amsterdam alone — disqualifying)

T1 also costs nothing extra: W1.8 established that `orig/T1.nii.gz` already
shares FLAIR's exact shape and affine in this dataset, so no registration is
involved, and `--keepgeom` writes the segmentation straight onto the FLAIR grid.

**The residual 2.96% is not ignored.** Any voxel the reference (or, at predict
time, our model) calls a lesion is removed from the ventricle mask before the
distance transform is computed — a lesion cannot also be cerebrospinal fluid.
See `features/periventricular.py` for where that is applied.

**Memory.** SynthSeg peaks near 12.5 GB on these volumes. On a 16 GB machine
that is tight enough that TensorFlow intermittently fails to allocate, which
surfaces as an opaque Keras error rather than a clean out-of-memory message —
three of nine sweep runs failed this way and all three succeeded on retry with
the machine otherwise idle. Hence `--threads 2` and the retry below.

    code/.venv/bin/python -m features.run_synthseg
    code/.venv/bin/python -m features.run_synthseg --splits train val
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from metadata.config import CODE_ROOT, PROJECT_ROOT
from metadata.derived import (VENTRICLES, derived_exists, derived_path,
                             save_derived)
from metadata.geometry import assert_same_geometry
from metadata.loader import load_nifti, load_split, load_wmh_mask, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging

SCRIPT_NAME = "run_synthseg"
OUTPUTS_DIR = CODE_ROOT / "features" / "outputs"
QC_CSV = OUTPUTS_DIR / "ventricle_qc.csv"

SYNTHSEG = Path("/opt/freesurfer/bin/mri_synthseg")

# FreeSurfer's standard labels. The lateral pair is what DeCarli's 10 mm rule
# refers to; the inferior-lateral horns are included because they are continuous
# with them and excluding them would put a false "deep" gap at the temporal horn.
LATERAL = (4, 43)
INFERIOR_LATERAL = (5, 44)
VENTRICLE_LABELS = LATERAL + INFERIOR_LATERAL

PLAUSIBLE_ML = (5.0, 150.0)


def segment_one(input_path: Path, output_path: Path, *, threads: int = 8) -> tuple[float, str]:
    """Run SynthSeg, escalating to more conservative settings if it runs out of memory.

    The failure this handles is real and was hit on 2026-09-14: SynthSeg tried to
    allocate a [1,256,288,160,72] float tensor — about 3.4 GB for a single layer —
    and TensorFlow raised OOM. Retrying the IDENTICAL command, which is what this
    function did before, cannot help a deterministic out-of-memory: the ladder has
    to actually give something up each time.

    Threads are the first thing surrendered because MKL allocates per-thread
    workspace, so thread count trades directly against peak memory. That is the
    whole cause here — Amsterdam's subjects have a LARGER field of view (9.86 L vs
    8.41 L) and completed fine at 8 threads, while Singapore 52 died at 14.

    Returns (seconds, which rung of the ladder succeeded).
    """
    ladder = [
        (f"{threads} threads", ["--threads", str(threads)]),
        ("4 threads", ["--threads", "4"]),
        ("2 threads + --fast", ["--threads", "2", "--fast"]),
        ("1 thread + --fast", ["--threads", "1", "--fast"]),
    ]
    last = ""
    for label, extra in ladder:
        started = time.time()
        result = subprocess.run(
            [str(SYNTHSEG), "--i", str(input_path), "--o", str(output_path),
             "--keepgeom", "--cpu", *extra],
            capture_output=True, text=True)
        if result.returncode == 0 and output_path.exists():
            return time.time() - started, label
        last = (result.stdout[-600:] + result.stderr[-600:])
    raise RuntimeError(f"SynthSeg failed at every setting on {input_path}:\n{last}")


def measure(ventricles, subject, affine, seconds, logger) -> dict:
    """The QC row. One function, so a resumed subject is measured exactly like a
    freshly-segmented one — the two paths cannot drift apart."""
    voxel_mm3 = float(np.prod(np.abs(np.diag(affine))[:3]))
    lateral_ml = float(ventricles["lateral"].sum() * voxel_mm3 / 1000.0)
    total_ml = float(ventricles["all"].sum() * voxel_mm3 / 1000.0)
    reference = load_wmh_mask(subject.mask_path)
    overlap = int((reference & ventricles["all"]).sum())
    plausible = PLAUSIBLE_ML[0] <= lateral_ml <= PLAUSIBLE_ML[1]
    if not plausible:
        logger.warning("  %s lateral ventricle %.1f mL is OUTSIDE %s — flagged, "
                       "not dropped", subject.subject_key, lateral_ml, PLAUSIBLE_ML)
    return {"subject_key": subject.subject_key, "site": subject.site,
            "lateral_ml": round(lateral_ml, 2),
            "total_ventricle_ml": round(total_ml, 2),
            "plausible": plausible,
            "reference_wmh_in_ventricle_vox": overlap,
            "reference_wmh_in_ventricle_pct":
                round(100 * overlap / max(int(reference.sum()), 1), 2),
            "seconds": round(seconds, 1) if seconds is not None else None}


def verify_existing(key: str, subject, logger):
    """Is the mask already on disk trustworthy enough to skip re-segmenting?

    EXISTENCE IS NOT ENOUGH. `save_derived` writes with `nib.save` straight to
    the final path, so a process killed mid-write leaves a truncated file that
    still satisfies `derived_exists`. A resume would then skip it and carry a
    corrupt ventricle mask into R5 — silently, because a broken mask still
    produces a distance transform and still produces numbers.

    So the file is opened and checked: readable, same geometry as the FLAIR,
    binary, and non-empty. Anything that fails is re-segmented rather than
    trusted. Returns the ventricle masks, or None to redo the subject.
    """
    try:
        img = load_nifti(derived_path(key, VENTRICLES))
        array = np.asarray(img.dataobj)
        assert_same_geometry(img, load_nifti(subject.flair_path),
                             context=f"{key} existing ventricles vs FLAIR")
        if not set(np.unique(array)) <= {0, 1}:
            raise ValueError(f"not binary: {np.unique(array)[:6]}")
        if array.sum() == 0:
            raise ValueError("empty mask")
        return {"all": array.astype(bool), "img": img}
    except Exception as exc:
        logger.warning("  %s has a ventricle file but it FAILED verification "
                       "(%s) — re-segmenting rather than trusting it",
                       key, str(exc)[:120])
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--threads", type=int, default=8,
                        help="SynthSeg runs one subject at a time (it peaks near 12.5 GB, "
                             "so parallel subjects do not fit); threads is where the "
                             "machine's cores actually get used.")
    parser.add_argument("--redo", action="store_true",
                        help="re-segment subjects that already have a ventricle mask")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    if not SYNTHSEG.exists():
        raise SystemExit(f"mri_synthseg not found at {SYNTHSEG}")

    subjects = subjects_by_key()
    keys = [k for split in args.splits for k in load_split(split)]
    logger.info("SynthSeg on the raw T1 for %d subjects (selected by sweep)", len(keys))

    rows, done_earlier, failed = [], [], []
    with tempfile.TemporaryDirectory(prefix="synthseg_") as tmp:
        for index, key in enumerate(keys, start=1):
            subject = subjects[key]
            # Resume by default. This machine hard-powered-off once already today
            # and a full pass is over an hour; re-segmenting what is already on
            # disk would turn any interruption into starting over.
            if derived_exists(key, VENTRICLES) and not args.redo:
                existing = verify_existing(key, subject, logger)
                if existing is not None:
                    # Re-measured from the mask on disk, NOT carried over from a
                    # previous CSV — no completed run may ever have written one,
                    # and a QC table covering 41 of 60 while claiming to cover 60
                    # is exactly the plausible-looking wrong number to avoid.
                    lateral = np.asarray(load_nifti(
                        derived_path(key, VENTRICLES)).dataobj).astype(bool)
                    rows.append(measure({"all": existing["all"], "lateral": lateral},
                                        subject, existing["img"].affine, None, logger))
                    done_earlier.append(key)
                    logger.info("  %d/%d  %s already done (verified), skipping",
                                index, len(keys), key)
                    continue
            raw_out = Path(tmp) / f"{key}_seg.nii.gz"
            try:
                seconds, rung = segment_one(subject.t1_path, raw_out,
                                            threads=args.threads)
            except RuntimeError as exc:
                # Recorded, never silently dropped (CLAUDE.md Section 4). The run
                # continues so one bad subject does not cost the other 59, and
                # main() refuses to declare success at the end.
                logger.error("  %s FAILED at every setting: %s", key, str(exc)[:200])
                failed.append(key)
                continue
            if rung != f"{args.threads} threads":
                logger.warning("  %s needed a reduced setting (%s) — memory pressure",
                               key, rung)

            seg_img = load_nifti(raw_out)          # canonical RAS, like everything else
            flair = load_nifti(subject.flair_path)
            assert_same_geometry(seg_img, flair, context=f"{key} SynthSeg vs FLAIR")

            seg = np.asarray(seg_img.dataobj).astype(np.int32)
            ventricles = np.isin(seg, VENTRICLE_LABELS)

            voxel_mm3 = float(np.prod(np.abs(np.diag(seg_img.affine))[:3]))
            lateral_ml = float(np.isin(seg, LATERAL).sum() * voxel_mm3 / 1000.0)
            total_ml = float(ventricles.sum() * voxel_mm3 / 1000.0)

            reference = load_wmh_mask(subject.mask_path)
            overlap = int((reference & ventricles).sum())

            save_derived(ventricles, subject, VENTRICLES,
                         generating_script=f"code/features/{SCRIPT_NAME}.py",
                         dtype=np.uint8,
                         extra={"source": "raw_T1", "labels": list(VENTRICLE_LABELS),
                                "tool": "mri_synthseg --keepgeom",
                                "lateral_ml": round(lateral_ml, 2)})

            rows.append(measure({"all": ventricles, "lateral": np.isin(seg, LATERAL)},
                                subject, seg_img.affine, seconds, logger))
            if index % 5 == 0 or index == len(keys):
                logger.info("  %d/%d  (last: %s, %.1f mL, %.0fs)",
                            index, len(keys), key, lateral_ml, seconds)

    table = pd.DataFrame(rows).sort_values("subject_key").reset_index(drop=True)
    logger.info("QC rows: %d (%d segmented now, %d verified from an earlier run)",
                len(table), len(table) - len(done_earlier), len(done_earlier))
    covered = set(table.subject_key)
    uncovered = [k for k in keys if k not in covered and k not in failed]
    if uncovered:
        raise SystemExit(f"QC table is missing {len(uncovered)} subject(s): "
                         f"{uncovered[:5]}. Refusing to report summary statistics "
                         f"that claim to cover the cohort but do not.")
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=f"code/features/{SCRIPT_NAME}.py")

    logger.info("=" * 72)
    logger.info("lateral ventricle volume by site (mL):\n%s",
                table.groupby("site")["lateral_ml"].describe()[
                    ["mean", "min", "max"]].round(1).to_string())
    logger.info("outside the plausible band: %d of %d",
                int((~table.plausible).sum()), len(table))
    logger.info("reference WMH absorbed into the ventricle label: mean %.2f%%, "
                "worst %.2f%% (%s)",
                table.reference_wmh_in_ventricle_pct.mean(),
                table.reference_wmh_in_ventricle_pct.max(),
                table.loc[table.reference_wmh_in_ventricle_pct.idxmax(), "subject_key"])
    logger.info("this is removed before the distance transform — see "
                "code/features/periventricular.py")
    logger.info("total runtime %.0f min", table.seconds.sum() / 60)
    logger.info("written: %s", QC_CSV.name)

    if failed:
        raise SystemExit(
            f"{len(failed)} subject(s) produced no ventricle mask: {failed}\n"
            f"R5 must not be computed on a partial cohort. Re-run to retry only "
            f"these (finished subjects are skipped automatically).")


if __name__ == "__main__":
    main()
