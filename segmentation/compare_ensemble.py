"""Did ensembling actually help? Score every member alone, then together.

**Why this exists.** `run_ensemble.py` printed a comparison against a hardcoded
`best_single = 0.8056`. That figure is the Week 3 **pre-augmentation** single
model. Holding the augmented ensemble against it conflates two independent
changes — scanner-simulating augmentation, and ensembling — so the resulting
"-0.0018, the ensemble did not help" answers neither question. Worse, it reads
as a clean negative result, which is exactly the kind of plausible-looking wrong
number CLAUDE.md 5.2 is written about.

This separates the three effects by scoring every configuration the same way,
on the same held-out subjects, with the official scorer's semantics:

    augmentation cost   = augmented single      - un-augmented single (0.8056)
    ensembling gain     = ensemble of 4         - best augmented single
    test-time flip gain = with TTA              - without TTA

**Scored in memory, not from disk, deliberately.** Writing each configuration's
prediction to `data/processed/` would overwrite the production ensemble output
that Week 4's features depend on. `score_in_memory` reproduces the vendored
scorer's label-2 rules exactly (see `checks/official_score.py`), and both it and
the prediction path read through `load_nifti`, so they share one canonical
orientation.

    code/.venv/bin/python -m segmentation.compare_ensemble
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from checks.official_score import score_in_memory
from metadata.config import PROJECT_ROOT
from metadata.loader import load_raw_mask_array, load_split, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from segmentation.run_ensemble import CHECKPOINT_DIR, predict
from segmentation.unet import UNet

SCRIPT_NAME = "compare_ensemble"
OUTPUTS_DIR = PROJECT_ROOT / "segmentation" / "outputs"
RESULTS_CSV = OUTPUTS_DIR / "ensemble_contribution.csv"

# The Week 3 single U-Net, trained WITHOUT scanner augmentation, scored by the
# official scorer on these same 12 held-out subjects. Recorded in Section 13.
WEEK3_SINGLE_NO_AUG = 0.8056


def load_one(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = UNet(in_channels=2).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def score_config(members, keys, subjects, device, *, use_tta):
    """Mean official Dice over `keys` for this set of members."""
    dice = []
    for key in keys:
        prediction, _ = predict(members, key, device, use_tta=use_tta, threshold=0.5)
        raw_reference, _ = load_raw_mask_array(subjects[key].mask_path)
        dice.append(score_in_memory(prediction, raw_reference)["dice"])
    return float(np.mean(dice)), dice


def main() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = sorted(p for p in CHECKPOINT_DIR.glob("unet_seed*_aug.pt")
                   if not p.name.endswith(".resume.pt"))
    if not paths:
        raise SystemExit("no augmented members found")

    subjects = subjects_by_key()
    keys = load_split("val")
    logger.info("scoring on the %d held-out subjects, official scorer semantics", len(keys))

    loaded = [load_one(p, device) for p in paths]
    rows = []

    for path, (model, checkpoint) in zip(paths, loaded):
        for use_tta in (False, True):
            mean, per_subject = score_config([model], keys, subjects, device, use_tta=use_tta)
            rows.append({"config": path.stem, "members": 1, "tta": use_tta, "dice": mean})
            logger.info("  %-20s tta=%-5s  Dice %.4f", path.stem, use_tta, mean)

    members = [m for m, _ in loaded]
    for use_tta in (False, True):
        mean, _ = score_config(members, keys, subjects, device, use_tta=use_tta)
        rows.append({"config": f"ensemble_{len(members)}", "members": len(members),
                     "tta": use_tta, "dice": mean})
        logger.info("  %-20s tta=%-5s  Dice %.4f", f"ensemble of {len(members)}",
                    use_tta, mean)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_CSV, index=False)
    write_manifest(RESULTS_CSV, generating_script=f"segmentation/{SCRIPT_NAME}.py")

    singles = table[(table.members == 1) & (table.tta)]
    best_single = singles.dice.max()
    best_name = singles.loc[singles.dice.idxmax(), "config"]
    ens_tta = float(table[(table.members > 1) & (table.tta)].dice.iloc[0])
    ens_notta = float(table[(table.members > 1) & (~table.tta)].dice.iloc[0])
    single_notta = float(singles.dice.mean())
    singles_notta_mean = float(table[(table.members == 1) & (~table.tta)].dice.mean())

    logger.info("=" * 70)
    logger.info("THE THREE EFFECTS, SEPARATED")
    logger.info("  augmentation (in-domain cost): %.4f -> %.4f  = %+.4f",
                WEEK3_SINGLE_NO_AUG, best_single, best_single - WEEK3_SINGLE_NO_AUG)
    logger.info("     (un-augmented Week 3 single vs best augmented single, both +TTA)")
    logger.info("  ensembling 4 nets            : %.4f -> %.4f  = %+.4f",
                best_single, ens_tta, ens_tta - best_single)
    logger.info("     (best augmented single %s vs 4 together, both +TTA)", best_name)
    logger.info("  test-time flips              : %.4f -> %.4f  = %+.4f",
                ens_notta, ens_tta, ens_tta - ens_notta)
    logger.info("     (the ensemble, without vs with mirrored passes)")
    logger.info("  mean single net, no TTA      : %.4f", singles_notta_mean)
    logger.info("=" * 70)

    summary = {
        "week3_single_no_augmentation": WEEK3_SINGLE_NO_AUG,
        "best_augmented_single": best_single,
        "best_augmented_single_name": best_name,
        "ensemble_with_tta": ens_tta,
        "ensemble_without_tta": ens_notta,
        "augmentation_effect": best_single - WEEK3_SINGLE_NO_AUG,
        "ensembling_effect": ens_tta - best_single,
        "tta_effect": ens_tta - ens_notta,
        "n_held_out": len(keys),
    }
    (OUTPUTS_DIR / "ensemble_contribution.json").write_text(json.dumps(summary, indent=2))
    logger.info("written: %s", RESULTS_CSV.name)


if __name__ == "__main__":
    main()
