"""Weeks 5-6, Phase A1 — honest segmentations for the classifier's features.

The severity classifier needs features measured from OUR segmentation, not the
expert's, or it would be predicting a number computed from its own inputs. But
the production ensemble is no good for this either: 48 of the 60 labelled
subjects are patients those four networks *trained on*, so their measurements
are in-sample. A plain volume threshold on them scores 92% on those 48 and 75% on
the 12 validation patients — the classifier would learn from measurements far
cleaner than anything it will meet in Week 7.

So each hospital's 20 patients are segmented here by the Week 3
**leave-one-site-out** network that never saw that hospital:

    Amsterdam patients  <-  unet_loso_amsterdam_aug.pt  (trained on Singapore + Utrecht)
    Singapore patients  <-  unet_loso_singapore_aug.pt  (trained on Amsterdam + Utrecht)
    Utrecht patients    <-  unet_loso_utrecht_aug.pt    (trained on Amsterdam + Singapore)

Every one of the 60 is then measured by a network that has never seen that
patient OR that scanner — harder than the production model's in-domain
condition, which is the honest direction to err in.

**Inference is the production path, unchanged.** `run_ensemble.predict` is
called with a one-network "ensemble": same soft probability, same left-right
test-time flip, same threshold, same brain-mask clip. The network is the only
difference, so any change in the features is attributable to it.

**A known, accepted caveat** (dataset.yaml `loso_epoch_selection_caveat`): each
checkpoint is the epoch that scored best on its own held-out site, so these
predictions are mildly optimistic. The networks never trained on these patients,
but the saved version was chosen with a look at them.

**The check that stops the pipeline.** Per-site Dice must land within
`dice_agreement_tolerance` of Week 3's leave-one-site-out results. Close, not
equal: Week 3 scored in memory without TTA, this uses the official scorer with
TTA. A larger gap means the wrong network met the wrong patients, or the path
differs — and then the verdict file says FAILED and `features.run_features`
refuses to read these predictions.

Only the 60 labelled training subjects. There is no leave-one-site-out
prediction for the 110 test subjects and there must never be one — Week 7 uses
the production ensemble.

    code/.venv/bin/python -m segmentation.run_loso_predictions
"""

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch

from checks.official_score import aggregate, score_prediction
from metadata.config import CODE_ROOT, SEVERITY_CONFIG, UNET_CONFIG
from metadata.derived import PRED_WMH_LOSO, derived_path, save_derived
from metadata.loader import load_split, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from segmentation.run_ensemble import predict
from segmentation.unet import UNet

SCRIPT_NAME = "run_loso_predictions"
GENERATING = f"code/segmentation/{SCRIPT_NAME}.py"
OUTPUTS_DIR = CODE_ROOT / "segmentation" / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
QC_CSV = OUTPUTS_DIR / "loso_prediction_scores.csv"
VERDICT_JSON = OUTPUTS_DIR / "loso_prediction_check.json"

LOSO = SEVERITY_CONFIG["loso_predictions"]
REQUIRED_KEYS = {"model", "epoch", "val_dice", "held_out_site"}


def load_loso_network(site: str, device, checkpoint_dir=CHECKPOINT_DIR):
    """The network that never saw `site`, refusing anything that is not provably it.

    The checkpoint records which site it held out (train_loso.py writes it).
    That record is checked against the site we are about to segment, rather
    than trusting the filename: a network applied to patients it trained on
    would produce features that look perfectly normal and are quietly
    in-sample — the exact failure this whole script exists to prevent.
    """
    path = checkpoint_dir / LOSO["checkpoints"][site]
    if not path.exists():
        raise FileNotFoundError(f"{path} — the leave-one-site-out network for {site}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    missing = REQUIRED_KEYS - set(checkpoint)
    if missing:
        raise ValueError(f"{path.name} is missing {sorted(missing)}; not a finished "
                         f"leave-one-site-out checkpoint")
    if checkpoint["held_out_site"] != site:
        raise ValueError(f"{path.name} held out {checkpoint['held_out_site']!r}, not "
                         f"{site!r} — applying it would put in-sample measurements "
                         f"into the classifier")
    model = UNet(in_channels=UNET_CONFIG["in_channels"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    info = {"checkpoint": path.name, "epoch": int(checkpoint["epoch"]),
            "week3_val_dice": float(checkpoint["val_dice"]),
            "held_out_site": checkpoint["held_out_site"]}
    return model, info


def site_agreement(scores: pd.DataFrame, expected: dict, tolerance: float) -> dict:
    """Per-site mean Dice against Week 3's, and whether every site is within tolerance.

    A site absent from `scores` FAILS rather than being skipped: a partial run
    must not be able to report itself as a pass.
    """
    per_site = {}
    for site, reference in expected.items():
        part = scores[scores.site == site]
        if part.empty:
            per_site[site] = {"n": 0, "dice": None, "expected": reference,
                              "difference": None, "within_tolerance": False}
            continue
        dice = float(part["dice"].mean())
        per_site[site] = {"n": int(len(part)), "dice": dice, "expected": reference,
                          "difference": dice - reference,
                          "within_tolerance": abs(dice - reference) <= tolerance}
    return {"tolerance": tolerance, "per_site": per_site,
            "passed": all(v["within_tolerance"] for v in per_site.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_tta = bool(LOSO["test_time_augmentation"])
    threshold = float(UNET_CONFIG["ensemble_probability_threshold"])

    # Any verdict from an earlier run is withdrawn before anything is written,
    # so a crash part-way through leaves NO verdict — and run_features refuses
    # to read predictions without one — rather than a stale PASS describing
    # files that have since been half-overwritten.
    VERDICT_JSON.unlink(missing_ok=True)

    subjects = subjects_by_key()
    keys = load_split("train") + load_split("val")
    logger.info("leave-one-site-out predictions for %d labelled subjects "
                "(TTA %s, threshold %.2f) on %s", len(keys),
                "ON" if use_tta else "OFF", threshold, device)

    networks = {}
    for site in LOSO["checkpoints"]:
        networks[site] = load_loso_network(site, device)
        logger.info("  %-10s <- %s (epoch %d, Week 3 unseen-site Dice %.4f)", site,
                    networks[site][1]["checkpoint"], networks[site][1]["epoch"],
                    networks[site][1]["week3_val_dice"])

    rows = []
    for index, key in enumerate(keys, start=1):
        subject = subjects[key]
        if subject.site not in networks:
            raise KeyError(f"{key}: no leave-one-site-out network for site {subject.site!r}")
        model, info = networks[subject.site]
        assert info["held_out_site"] == subject.site  # load_loso_network checked; belt and braces
        prediction, passes = predict([model], key, device, use_tta=use_tta,
                                     threshold=threshold)
        save_derived(prediction, subject, PRED_WMH_LOSO, generating_script=GENERATING,
                     extra={"route": "C_unet_leave_one_site_out", **info,
                            "tta": use_tta, "forward_passes": passes,
                            "threshold": threshold})
        scores = score_prediction(subject.mask_path, derived_path(key, PRED_WMH_LOSO))
        rows.append({"subject_key": key, "split": subject.split, "site": subject.site,
                     "network": info["checkpoint"],
                     "predicted_voxels": int(prediction.sum()), **scores})
        logger.info("  %2d/%d %-34s Dice %.4f  lesion F1 %.4f", index, len(keys), key,
                    scores["dice"], scores["lesion_f1"])

    table = pd.DataFrame(rows)
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=GENERATING,
                   extra={"note": "each subject scored against the network that never "
                                  "saw its hospital; official scorer"})

    empty = table[table.predicted_voxels == 0]
    if not empty.empty:
        logger.warning("%d EMPTY prediction(s): %s — derive_features will refuse these; "
                       "decide how to treat them explicitly", len(empty),
                       empty.subject_key.tolist())

    logger.info("all %d: %s", len(table), {k: (round(v, 4) if isinstance(v, float) else v)
                                           for k, v in aggregate(rows).items()})
    verdict = site_agreement(table, LOSO["expected_dice_by_site"],
                             float(LOSO["dice_agreement_tolerance"]))
    for site, v in verdict["per_site"].items():
        if v["n"] == 0:
            logger.error("  %-10s NO SUBJECTS SCORED", site)
            continue
        logger.info("  %-10s Dice %.4f vs Week 3 %.4f  (%+.4f)  %s", site, v["dice"],
                    v["expected"], v["difference"],
                    "ok" if v["within_tolerance"] else "OUTSIDE TOLERANCE")

    VERDICT_JSON.write_text(json.dumps(verdict, indent=2))
    write_manifest(VERDICT_JSON, generating_script=GENERATING)
    if not verdict["passed"]:
        logger.error("per-site Dice disagrees with Week 3 by more than %.2f — the wrong "
                     "network may have met the wrong patients, or the inference path "
                     "differs. STOP: features.run_features will refuse these predictions "
                     "until this is investigated.", verdict["tolerance"])
        raise SystemExit(1)
    logger.info("PASSED: every site within %.2f of Week 3. Next: python -m "
                "features.run_features", verdict["tolerance"])


if __name__ == "__main__":
    main()
