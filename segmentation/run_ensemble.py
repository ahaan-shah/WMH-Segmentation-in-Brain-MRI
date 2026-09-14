"""Week 3 — combine the trained U-Nets into one prediction (R4).

Each member saw the same data and the same architecture; only the random seed
differed. They therefore make *different* mistakes while agreeing on the real
signal, so averaging their probability maps cancels the noise and keeps the
lesions. ROADMAP 6.2 identifies this as the cheapest accuracy gain available,
and both leaderboard leaders used it — sysu_media three networks, pgs five plus
test-time flips.

**Soft voting, not hard voting.** The members' *probabilities* are averaged and
the threshold is applied once at the end, rather than each member casting a
binary vote. Thresholding first throws away exactly the confidence information
that makes averaging worth doing — a voxel four members call 0.49 is very
different from one they call 0.02, and hard voting cannot tell them apart.

**Test-time augmentation.** Each member also predicts on the left-right mirrored
image; the result is flipped back and averaged in. Free accuracy, no retraining.
Safe here for the same reason the training augmentation was: lesions are not
lateralised in a way the network should be learning. It must never be used for
Week 4's hemisphere features, where flipping would invert the answer.

    code/.venv/bin/python -m segmentation.run_ensemble
    code/.venv/bin/python -m segmentation.run_ensemble --no-tta
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from checks.official_score import aggregate, score_prediction
from metadata.config import PROJECT_ROOT
from metadata.derived import PRED_WMH, derived_path, save_derived
from metadata.loader import load_split, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from segmentation.dataset import load_subject_slices, undo_pad_or_crop
from segmentation.unet import UNet

SCRIPT_NAME = "run_ensemble"
OUTPUTS_DIR = PROJECT_ROOT / "segmentation" / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
QC_CSV = OUTPUTS_DIR / "ensemble_scores.csv"


def load_members(device, logger) -> list:
    """Load every FINISHED ensemble member.

    `*.resume.pt` files are excluded deliberately. They are mid-training state
    written once per epoch for crash recovery, and their names end in `.pt`, so
    a plain `unet_seed*.pt` glob sweeps them up as though they were trained
    models — quietly adding a half-trained duplicate of a seed to the vote while
    training is still running. That is exactly the failure CLAUDE.md 5.2 is
    about: it would not crash, it would just return a slightly wrong number.
    """
    paths = sorted(p for p in CHECKPOINT_DIR.glob("unet_seed*.pt")
                   if not p.name.endswith(".resume.pt"))
    if not paths:
        raise SystemExit(f"no finished checkpoints in {CHECKPOINT_DIR}")

    members = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        missing = {"model", "epoch", "val_dice"} - set(checkpoint)
        if missing:
            raise SystemExit(
                f"{path.name} is missing {sorted(missing)} — this is not a finished "
                f"member. Refusing to ensemble it rather than guessing.")
        model = UNet(in_channels=2).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        members.append(model)
        logger.info("  %-24s epoch %3d, individual val Dice %.4f",
                    path.name, checkpoint["epoch"], checkpoint["val_dice"])

    logger.info("ensembling %d member(s) — every one a completed run", len(members))
    return members


@torch.no_grad()
def predict(members, key: str, device, *, use_tta: bool, threshold: float):
    data = load_subject_slices(key, with_labels=False)
    images = torch.from_numpy(data["image"]).to(device)

    accumulated = torch.zeros((len(images), 1, images.shape[-2], images.shape[-1]),
                              device=device)
    passes = 0
    for model in members:
        for start in range(0, len(images), 16):
            batch = images[start:start + 16]
            accumulated[start:start + 16] += torch.sigmoid(model(batch))
        passes += 1
        if use_tta:
            for start in range(0, len(images), 16):
                batch = torch.flip(images[start:start + 16], dims=[-1])
                flipped = torch.sigmoid(model(batch))
                accumulated[start:start + 16] += torch.flip(flipped, dims=[-1])
            passes += 1

    probabilities = (accumulated / passes).cpu().numpy()[:, 0]

    height, width, _ = data["volume_shape"]
    volume = np.zeros(data["volume_shape"], dtype=np.float32)
    brain = np.zeros(data["volume_shape"], dtype=bool)
    for index, z in enumerate(data["z_indices"]):
        volume[:, :, z] = undo_pad_or_crop(probabilities[index], (height, width),
                                           centre=data["centre"])
        brain[:, :, z] = undo_pad_or_crop(data["brain"][index, 0], (height, width),
                                          centre=data["centre"]) > 0.5
    return (volume > threshold) & brain, passes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("ensemble members:")
    members = load_members(device, logger)
    use_tta = not args.no_tta
    logger.info("%d members, test-time flip augmentation %s",
                len(members), "ON" if use_tta else "OFF")

    subjects = subjects_by_key()
    keys = [key for split in args.splits for key in load_split(split)]
    val_keys = set(load_split("val"))

    rows = []
    for index, key in enumerate(keys, start=1):
        subject = subjects[key]
        prediction, passes = predict(members, key, device,
                                     use_tta=use_tta, threshold=args.threshold)
        save_derived(prediction, subject, PRED_WMH,
                     generating_script=f"segmentation/{SCRIPT_NAME}.py",
                     extra={"route": "C_unet_ensemble", "members": len(members),
                            "tta": use_tta, "forward_passes": passes,
                            "threshold": args.threshold})
        scores = score_prediction(subject.mask_path, derived_path(key, PRED_WMH))
        rows.append({"subject_key": key, "split": subject.split, "site": subject.site,
                     "held_out": key in val_keys, **scores})
        if index % 10 == 0 or index == len(keys):
            logger.info("  %d/%d", index, len(keys))

    table = pd.DataFrame(rows)
    table.to_csv(QC_CSV, index=False)
    write_manifest(QC_CSV, generating_script=f"segmentation/{SCRIPT_NAME}.py")

    pd.set_option("display.width", 250)
    held_out = table[table.held_out]
    logger.info("HELD-OUT (%d subjects) — the number that counts:\n    %s",
                len(held_out), {k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in aggregate(held_out.to_dict("records")).items()})
    logger.info("held-out per site:\n%s", held_out.groupby("site")[
        ["dice", "lesion_f1", "lesion_recall"]].mean().round(4).to_string())
    logger.info("all %d subjects:\n    %s", len(table),
                {k: (round(v, 4) if isinstance(v, float) else v)
                 for k, v in aggregate(table.to_dict("records")).items()})

    # NOTE: 0.8056 is the WEEK 3 single U-Net trained WITHOUT scanner
    # augmentation. Comparing the augmented ensemble against it mixes two
    # separate changes and answers neither question — it previously printed a
    # "the ensemble did not help" warning that was an artefact of that mix-up.
    # `segmentation/compare_ensemble.py` separates the effects properly by
    # scoring every configuration the same way; read that, not this line.
    week3_single_no_aug = 0.8056
    logger.info("COMPARISON on held-out subjects (reference points, NOT a controlled test):")
    logger.info("    simple threshold baseline (Route A)      : 0.4291")
    logger.info("    Week 3 single U-Net, NO augmentation     : %.4f", week3_single_no_aug)
    logger.info("    this ensemble of %d + TTA                 : %.4f",
                len(members), held_out["dice"].mean())
    logger.info("    For the augmentation / ensembling / TTA contributions measured "
                "separately, run: python -m segmentation.compare_ensemble")
    if gain <= 0:
        logger.warning("The ensemble did not beat the best single model. That is a real "
                       "result, not a bug — report it rather than quietly keeping the "
                       "single model. Note the single model's 0.8056 was itself selected "
                       "as the best epoch ON this same validation set, so it is mildly "
                       "optimistic; the ensemble number is not.")


if __name__ == "__main__":
    main()
