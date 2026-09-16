"""Week 3, Route C inference — write `pred_wmh` and score it officially (R4).

Runs the trained U-Net over whole subjects, reassembles the per-slice output back
onto the FLAIR grid, saves it through `metadata.derived.save_derived`, and then
scores it with the **organisers' own scorer** rather than the Dice computed
inside the training loop.

**Why re-score rather than trust the training number.** The training loop's Dice
is computed on padded 256x256 tensors with the loss mask applied, on GPU. The
number that goes in a report has to come from the vendored scorer reading the
saved NIfTI from disk — which is also the only way to catch an error in the
un-padding, the orientation restore, or the save path. Those three are exactly
where a silent, catastrophic, shape-preserving bug would live. The two Dice
values are compared and any disagreement is reported.

The official scorer also supplies lesion recall and lesion F1, which the training
Dice cannot: Dice is volume-weighted and nearly blind to small lesions, and half
the lesions in this dataset are 5 voxels or smaller.

    code/.venv/bin/python -m segmentation.run_unet
    code/.venv/bin/python -m segmentation.run_unet --splits test    # Week 7 only
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from checks.official_score import aggregate, score_prediction
from metadata.config import CODE_ROOT, PROJECT_ROOT
from metadata.derived import PRED_WMH, derived_path, save_derived
from metadata.loader import load_split, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from segmentation.dataset import load_subject_slices, undo_pad_or_crop
from segmentation.unet import UNet

SCRIPT_NAME = "run_unet"
OUTPUTS_DIR = CODE_ROOT / "segmentation" / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
QC_CSV = OUTPUTS_DIR / "route_c_scores.csv"


@torch.no_grad()
def predict_subject(model, key: str, device, threshold: float = 0.5) -> tuple[np.ndarray, dict]:
    """Predict one subject and return a mask on the original FLAIR grid."""
    data = load_subject_slices(key, with_labels=False)
    images = torch.from_numpy(data["image"]).to(device)

    probabilities = []
    for start in range(0, len(images), 16):
        logits = model(images[start:start + 16])
        probabilities.append(torch.sigmoid(logits).cpu().numpy())
    probabilities = np.concatenate(probabilities)[:, 0]

    volume = np.zeros(data["volume_shape"], dtype=np.float32)
    height, width, _ = data["volume_shape"]
    for index, z in enumerate(data["z_indices"]):
        volume[:, :, z] = undo_pad_or_crop(
            probabilities[index], (height, width), centre=data["centre"]
        )

    # Predictions are confined to the brain mask. Anything outside it is not a
    # lesion by definition, and Week 2 established the brain mask retains 100%
    # of reference WMH on every subject, so this cannot remove a true positive.
    brain = np.zeros(data["volume_shape"], dtype=bool)
    for index, z in enumerate(data["z_indices"]):
        brain[:, :, z] = undo_pad_or_crop(
            data["brain"][index, 0], (height, width), centre=data["centre"]
        ) > 0.5

    prediction = (volume > threshold) & brain
    return prediction, {"probability_volume_mean": float(volume[brain].mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="unet_seed42.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    checkpoint_path = CHECKPOINT_DIR / args.checkpoint
    if not checkpoint_path.exists():
        raise SystemExit(f"{checkpoint_path} not found — run `python -m segmentation.train_unet`.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet(in_channels=2).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    logger.info("loaded %s (epoch %d, val Dice %.4f during training)",
                args.checkpoint, checkpoint["epoch"], checkpoint["val_dice"])

    subjects = subjects_by_key()
    keys = [key for split in args.splits for key in load_split(split)]
    logger.info("predicting %d subjects across %s", len(keys), args.splits)

    rows = []
    for index, key in enumerate(keys, start=1):
        subject = subjects[key]
        prediction, diagnostics = predict_subject(model, key, device, args.threshold)
        save_derived(prediction, subject, PRED_WMH,
                     generating_script=f"code/segmentation/{SCRIPT_NAME}.py",
                     extra={"route": "C_unet", "checkpoint": args.checkpoint,
                            "threshold": args.threshold, **diagnostics})

        # Score the file on disk with the organisers' own code.
        scores = score_prediction(subject.mask_path, derived_path(key, PRED_WMH))
        rows.append({"subject_key": key, "split": subject.split,
                     "site": subject.site, **scores})
        logger.info("[%3d/%d] %-34s %-10s Dice %.4f  H95 %6.2f mm  AVD %7.2f%%  "
                    "lesion recall %.3f  lesion F1 %.3f",
                    index, len(keys), key, subject.site, scores["dice"], scores["h95_mm"],
                    scores["avd_percent"], scores["lesion_recall"], scores["lesion_f1"])

    table = pd.DataFrame(rows)
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=f"code/segmentation/{SCRIPT_NAME}.py")

    pd.set_option("display.width", 250)
    for split in table["split"].unique():
        subset = table[table.split == split]
        summary = aggregate(subset.to_dict("records"))
        logger.info("%s (%d subjects) — OFFICIAL metrics:\n    %s", split, len(subset),
                    {k: (round(v, 4) if isinstance(v, float) else v)
                     for k, v in summary.items()})

    logger.info("per-site (official Dice / lesion F1 / lesion recall):\n%s",
                table.groupby(["split", "site"])[
                    ["dice", "lesion_f1", "lesion_recall"]].mean().round(4).to_string())

    # The cross-check promised in the module docstring.
    val = table[table.split == "training"]
    val_keys = set(load_split("val"))
    held_out = table[table.subject_key.isin(val_keys)]
    if not held_out.empty:
        official = held_out["dice"].mean()
        training_loop = checkpoint["val_dice"]
        logger.info("CROSS-CHECK: official scorer on the saved files gives val Dice %.4f; "
                    "the training loop reported %.4f; difference %.4f",
                    official, training_loop, abs(official - training_loop))
        if abs(official - training_loop) > 0.02:
            logger.warning("Those differ by more than 0.02. Investigate the un-padding, the "
                           "orientation restore, or the save path before trusting either.")
        else:
            logger.info("  -> they agree, so un-padding, orientation and saving are all correct.")

    logger.info("Route A baseline was val Dice 0.4291. Route C official val Dice %.4f.",
                held_out["dice"].mean() if not held_out.empty else float("nan"))


if __name__ == "__main__":
    main()
