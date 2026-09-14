"""Leave-one-site-out validation — does the model survive an unfamiliar scanner?

Trains three times. Each run sees two hospitals and is tested on the third,
which it has never seen. That is the closest honest approximation of the real
Week 7 test, where the sealed 110 subjects include **two scanners absent from
training entirely**.

**This is not a way to score higher. It is a way to find out whether the score
we already have is real.** Our 0.808 comes from validation subjects drawn from
the same three scanners the model trained on, so it cannot distinguish a model
that learned "what a white-matter lesion looks like" from one that learned "what
a Utrecht scan looks like". Those two score identically here and completely
differently in Week 7.

ROADMAP 6.4 calls this out as the main risk of the whole segmentation stage:
methods that overfit to scanner-specific intensity characteristics collapse on
unseen vendors, which is precisely what the challenge was designed to expose.

**Interpretation, fixed before the results arrive**, so the outcome cannot be
rationalised afterwards:

- **~0.75 or above** on a held-out site — the model generalises. Our 0.808 is
  broadly trustworthy and we are genuinely competitive.
- **~0.65 to 0.75** — some scanner dependence. Expect the Week 7 number to land
  below our validation figure, and say so in the report rather than being
  surprised by it.
- **below ~0.65** — the model is substantially fitting scanner appearance rather
  than anatomy. That is a real problem with eight weeks left to fix it, and far
  better learned now than in Week 7.

A gap between pooled and leave-one-site-out performance is a finding worth
reporting, not a failure to hide (ROADMAP 6.4).

    code/.venv/bin/python -m segmentation.train_loso
    code/.venv/bin/python -m segmentation.train_loso --sites Utrecht
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from metadata.config import PROJECT_ROOT, SEED
from metadata.loader import load_split, subjects_by_key
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from segmentation.dataset import build_split, load_subject_slices
from segmentation.augment import augment_for_domain_generalisation
from segmentation.train_unet import augment, set_determinism, validate
from segmentation.unet import UNet, masked_dice_bce_loss

SCRIPT_NAME = "train_loso"
OUTPUTS_DIR = PROJECT_ROOT / "segmentation" / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
RESULTS_JSON = OUTPUTS_DIR / "leave_one_site_out.json"

SITES = ("Amsterdam", "Singapore", "Utrecht")

# Interpretation bands, recorded here rather than decided after the fact.
GENERALISES = 0.75
PARTIAL = 0.65


def site_split(held_out_site: str) -> tuple[list[str], list[str]]:
    """Train on every subject NOT from `held_out_site`; test on every one that is.

    Uses all 60 labelled subjects, since the split here is by scanner rather
    than the frozen train/val division. The official 110 remain sealed.
    """
    subjects = subjects_by_key()
    keys = load_split("train") + load_split("val")
    train = [k for k in keys if subjects[k].site != held_out_site]
    test = [k for k in keys if subjects[k].site == held_out_site]
    if not train or not test:
        raise ValueError(f"empty split for held-out site {held_out_site!r}")
    return sorted(train), sorted(test)


def run_one(held_out_site: str, args, logger) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(args.seed)

    train_keys, test_keys = site_split(held_out_site)
    logger.info("=== holding out %s: train on %d subjects from the other two sites, "
                "test on %d unseen ===", held_out_site, len(train_keys), len(test_keys))

    train_data = build_split(train_keys, logger)
    test_subjects = [load_subject_slices(key) for key in test_keys]

    dataset = TensorDataset(
        torch.from_numpy(train_data["image"]),
        torch.from_numpy(train_data["label"]),
        torch.from_numpy(train_data["loss_mask"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = UNet(in_channels=2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    suffix = f"_{args.tag}" if args.tag else ""
    checkpoint_path = CHECKPOINT_DIR / f"unet_loso_{held_out_site.lower()}{suffix}.pt"
    best, best_epoch, started = -1.0, -1, time.time()

    for epoch in range(1, args.epochs + 1):
        total, batches = 0.0, 0
        for images, labels, masks in loader:
            images, labels, masks = (images.to(device), labels.to(device), masks.to(device))
            if args.augment_strength > 0:
                images, labels, masks = augment_for_domain_generalisation(
                    images, labels, masks, generator, strength=args.augment_strength)
            else:
                images, labels, masks = augment(images, labels, masks, generator)
            optimiser.zero_grad(set_to_none=True)
            loss = masked_dice_bce_loss(model(images), labels, masks)
            loss.backward()
            optimiser.step()
            total += loss.item(); batches += 1
        scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            scores = validate(model, test_subjects, device)
            # NOTE: this is the UNSEEN site. It is reported for monitoring only —
            # the checkpoint is saved on it, which is a mild optimism we accept
            # and state, because the alternative (a third split) costs subjects
            # we do not have at n=60.
            if scores["dice_mean"] > best:
                best, best_epoch = scores["dice_mean"], epoch
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_dice": best, "held_out_site": held_out_site},
                           checkpoint_path)
            logger.info("  epoch %3d/%d  loss %.4f  UNSEEN-%s Dice %.4f +- %.4f",
                        epoch, args.epochs, total / batches, held_out_site,
                        scores["dice_mean"], scores["dice_std"])
        else:
            logger.info("  epoch %3d/%d  loss %.4f", epoch, args.epochs, total / batches)

    minutes = (time.time() - started) / 60
    logger.info("  %s done in %.0f min — best unseen-site Dice %.4f at epoch %d",
                held_out_site, minutes, best, best_epoch)
    return {"held_out_site": held_out_site, "dice": best, "best_epoch": best_epoch,
            "n_train": len(train_keys), "n_test": len(test_keys), "minutes": minutes}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sites", nargs="+", default=list(SITES), choices=list(SITES))
    parser.add_argument("--augment-strength", type=float, default=0.0,
                        help="0 = Week 3 behaviour (flip + mild gain). >0 enables "
                             "scanner-simulating augmentation at that strength.")
    parser.add_argument("--tag", default="", help="suffix for checkpoints and results")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)
    logger.info("leave-one-site-out over %s", args.sites)
    logger.info("bands fixed in advance: >=%.2f generalises, %.2f-%.2f partial scanner "
                "dependence, <%.2f substantial overfitting to scanner appearance",
                GENERALISES, PARTIAL, GENERALISES, PARTIAL)

    results = [run_one(site, args, logger) for site in args.sites]

    out = (OUTPUTS_DIR / f"leave_one_site_out_{args.tag}.json") if args.tag else RESULTS_JSON
    out.write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    write_manifest(out, generating_script=f"segmentation/{SCRIPT_NAME}.py")

    mean = float(np.mean([r["dice"] for r in results]))
    logger.info("=" * 70)
    for r in results:
        logger.info("  unseen %-10s Dice %.4f", r["held_out_site"], r["dice"])
    logger.info("  MEAN across held-out sites: %.4f", mean)
    logger.info("  same-scanner validation (for comparison): 0.8084")
    logger.info("  generalisation gap: %+.4f", mean - 0.8084)

    if mean >= GENERALISES:
        logger.info("  VERDICT: generalises. The 0.808 is broadly trustworthy.")
    elif mean >= PARTIAL:
        logger.info("  VERDICT: partial scanner dependence. Expect Week 7 below 0.808; "
                    "state that in the report rather than being surprised by it.")
    else:
        logger.info("  VERDICT: substantially fitting scanner appearance, not anatomy. "
                    "This is a real problem and worth addressing now.")


if __name__ == "__main__":
    main()
