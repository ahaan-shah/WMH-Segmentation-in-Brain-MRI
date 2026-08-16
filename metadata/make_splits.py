"""Freeze data splits (CLAUDE.md W1.5).

Section 11.6 was resolved in Week 1: all 110 official test subjects ship
real, non-empty wmh.nii.gz masks. This means the strong evaluation path
applies (metadata/dataset.yaml: splits.test_set = official_110) — the 110 are
the sealed test set, evaluated directly against the published leaderboard in
W7, and are NOT opened again until then. Only the 60 official training
subjects are split here, into train/val, stratified by site and seeded from
config so the split is identical on every re-run (CLAUDE.md Section 5.4).

Split is by SUBJECT, never by slice — slices from one brain appearing in both
train and val would be leakage and would inflate every downstream number.

Usage: python -m metadata.make_splits
Output: metadata/outputs/split_{train,val,test}.txt (+ .json provenance sidecars)
"""

import logging
import random
from datetime import date

from metadata.config import METADATA_OUTPUTS, SEED, TRAIN_FRACTION
from metadata.loader import discover_subjects
from metadata.provenance import write_manifest


def _setup_logging() -> logging.Logger:
    METADATA_OUTPUTS.mkdir(parents=True, exist_ok=True)
    log_path = METADATA_OUTPUTS / f"make_splits_{date.today().isoformat()}.log"

    logger = logging.getLogger("make_splits")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(fmt)
    console_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def make_splits() -> dict[str, list[str]]:
    logger = _setup_logging()
    subjects = discover_subjects()

    training = [s for s in subjects if s.split == "training"]
    test = [s for s in subjects if s.split == "test"]
    logger.info(f"{len(training)} training subjects (split into train/val), {len(test)} official test subjects (sealed)")

    rng = random.Random(SEED)

    train_keys, val_keys = [], []
    for site in sorted({s.site for s in training}):
        site_subjects = sorted([s.subject_key for s in training if s.site == site])
        rng.shuffle(site_subjects)
        n_train = round(len(site_subjects) * TRAIN_FRACTION)
        train_keys.extend(site_subjects[:n_train])
        val_keys.extend(site_subjects[n_train:])
        logger.info(f"site={site}: {len(site_subjects)} total -> {n_train} train, {len(site_subjects) - n_train} val")

    train_keys.sort()
    val_keys.sort()
    test_keys = sorted(s.subject_key for s in test)

    assert set(train_keys).isdisjoint(val_keys), "train/val overlap — subject leakage"
    assert set(train_keys).isdisjoint(test_keys), "train/test overlap — subject leakage"
    assert set(val_keys).isdisjoint(test_keys), "val/test overlap — subject leakage"
    assert len(train_keys) + len(val_keys) == len(training)

    splits = {"train": train_keys, "val": val_keys, "test": test_keys}

    for name, keys in splits.items():
        out_path = METADATA_OUTPUTS / f"split_{name}.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(keys) + "\n")
        write_manifest(
            out_path,
            generating_script="metadata/make_splits.py",
            extra={"n_subjects": len(keys), "seed": SEED},
        )
        logger.info(f"Wrote {out_path} ({len(keys)} subjects)")

    logger.info(f"train={len(train_keys)} val={len(val_keys)} test={len(test_keys)} (SEED={SEED})")
    return splits


if __name__ == "__main__":
    make_splits()
