"""Where a driver's tables go, so the sealed test set never overwrites the 60.

Every per-run table (stage QC, ensemble scores, ventricle QC, features) used to
be written to one fixed filename. Run with `--splits test` in Week 7, a driver
would have replaced the committed 60-subject table — the evidence behind every
Week 2-6 number — with the 110 test rows. Found 2026-10-08: first in
`features/run_features.py` (Phase A), then in seven more drivers while writing
the Week 7 checklist.

The rule, in one place:

- train / val runs keep the original filename, so nothing already committed moves;
- a test-only run writes beside it with a `_test` suffix
  (`stage1_head_mask_qc.csv` -> `stage1_head_mask_qc_test.csv`);
- a run mixing test with train/val is REFUSED. One table must never hold both
  the subjects a method was developed on and the subjects it is judged on.
"""

from __future__ import annotations

from pathlib import Path

TEST_SUFFIX = "_test"


def split_output(path: Path, splits) -> Path:
    """The file a run over `splits` writes `path` to. See the module docstring."""
    path = Path(path)
    splits = set(splits)
    if not splits:
        raise ValueError("no splits given")
    if "test" in splits and splits - {"test"}:
        raise SystemExit(
            f"--splits {' '.join(sorted(splits))}: the sealed test set cannot share a "
            f"run with train/val. Its tables are written to their own files and never "
            f"mixed into the ones the methods were developed on — run test on its own.")
    if splits == {"test"}:
        return path.with_name(f"{path.stem}{TEST_SUFFIX}{path.suffix}")
    return path
