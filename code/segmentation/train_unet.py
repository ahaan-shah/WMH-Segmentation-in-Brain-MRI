"""Week 3, Route C — train the 2D U-Net (R4).

Trains on the 48-subject training split and validates on the 12-subject
validation split. The official 110 stay sealed until Week 7.

**Validation is per subject, not per slice.** Slice-level Dice averages are not
the number anyone reports, and they are dominated by the many slices containing
no lesion at all. Intersection and denominator sums are accumulated per subject
and only then turned into a Dice, which is exactly what the official scorer
computes on a whole volume.

**Splits come from the frozen Week 1 files.** Slices from one brain appearing in
both training and validation would inflate every number substantially and
invisibly — the single most common way a medical segmentation result turns out
to be fiction. Nothing here re-derives a split.

    code/.venv/bin/python -m segmentation.train_unet
    code/.venv/bin/python -m segmentation.train_unet --epochs 120 --seed 1
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from metadata.config import CODE_ROOT, PROJECT_ROOT, SEED
from metadata.loader import load_split
from metadata.provenance import write_manifest
from metadata.runlog import setup_logging
from segmentation.augment import augment_for_domain_generalisation
from segmentation.dataset import build_split, load_subject_slices
from segmentation.unet import UNet, masked_dice_bce_loss

SCRIPT_NAME = "train_unet"
OUTPUTS_DIR = CODE_ROOT / "segmentation" / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"


def set_determinism(seed: int) -> None:
    """One seed, everywhere (CLAUDE.md Section 5.4)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def augment(images: torch.Tensor, labels: torch.Tensor, masks: torch.Tensor, generator):
    """Left-right flip plus a mild intensity gain.

    Flipping left-right is safe for *segmentation* — WMH are not lateralised in a
    way the network should learn — but it must never be used when generating
    data for Week 4's hemisphere features, where it would invert the answer.
    That distinction is why augmentation lives here and not in the shared
    dataset module.

    The intensity gain is deliberately small (+-5%). Week 2 spent the entire
    normalisation stage putting every scanner on one scale; large intensity
    jitter would throw that away and teach the network to ignore the very
    property that makes a single model work across sites.
    """
    flip = torch.rand(images.shape[0], generator=generator, device=images.device) < 0.5
    if flip.any():
        images[flip] = torch.flip(images[flip], dims=[-1])
        labels[flip] = torch.flip(labels[flip], dims=[-1])
        masks[flip] = torch.flip(masks[flip], dims=[-1])

    gain = 1.0 + 0.05 * (2 * torch.rand(images.shape[0], 1, 1, 1,
                                        generator=generator, device=images.device) - 1)
    return images * gain, labels, masks


def run_signature(args) -> dict:
    """The arguments a resumed run must match, or the resume is meaningless.

    Continuing an 80-epoch cosine schedule as if it were a 120-epoch one, or
    swapping the augmentation strength halfway, would produce a model that
    matches no recorded configuration. Fail loud instead (CLAUDE.md 5.2).
    """
    return {k: getattr(args, k) for k in
            ("epochs", "batch_size", "lr", "seed", "tag", "augment_strength")}


def save_resume_state(path, *, model, optimiser, scheduler, generator, epoch,
                      best_dice, best_epoch, history, minutes, args) -> None:
    """Write the full training state atomically, once per epoch.

    Atomic because the failure this exists for is a power cut: writing in place
    would eventually catch a crash mid-write and leave a corrupt file that is
    worse than no file. Write to a sibling, then rename — rename is atomic on
    any POSIX filesystem, so the file on disk is always a complete state.

    Everything needed to continue *identically* is here, not just the weights:
    the optimiser's momentum, the learning-rate schedule's position, and all
    three random-number streams. Restoring weights alone would silently restart
    Adam's momentum and re-roll the augmentation, which is a different run.
    """
    state = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "scheduler": scheduler.state_dict(),
        "generator": generator.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "best_dice": best_dice,
        "best_epoch": best_epoch,
        "history": history,
        "minutes": minutes,
        "signature": run_signature(args),
    }
    staging = path.with_suffix(".partial")
    torch.save(state, staging)
    staging.replace(path)


@torch.no_grad()
def validate(model, subjects: list[dict], device, threshold: float = 0.5) -> dict:
    """Per-subject Dice on the validation split."""
    model.eval()
    dice_per_subject, per_site = [], {}
    for data in subjects:
        images = torch.from_numpy(data["image"]).to(device)
        labels = torch.from_numpy(data["label"]).to(device)
        valid = torch.from_numpy(data["loss_mask"] * data["brain"]).to(device)

        intersection = denominator = 0.0
        for start in range(0, len(images), 16):
            logits = model(images[start:start + 16])
            prediction = (torch.sigmoid(logits) > threshold).float() * valid[start:start + 16]
            reference = labels[start:start + 16] * valid[start:start + 16]
            intersection += 2.0 * (prediction * reference).sum().item()
            denominator += prediction.sum().item() + reference.sum().item()

        dice = 1.0 if denominator == 0 else intersection / denominator
        dice_per_subject.append(dice)
        per_site.setdefault(data["site"], []).append(dice)

    model.train()
    return {
        "dice_mean": float(np.mean(dice_per_subject)),
        "dice_std": float(np.std(dice_per_subject)),
        "per_site": {site: float(np.mean(values)) for site, values in per_site.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--tag", default="", help="suffix for the checkpoint, for ensembling")
    parser.add_argument("--augment-strength", type=float, default=0.0,
                        help="0 = Week 3 behaviour. 1.0 = scanner-simulating augmentation, "
                             "which improved leave-one-site-out from 0.7382 to 0.7795.")
    parser.add_argument("--resume", action="store_true",
                        help="continue from the per-epoch resume state if one exists")
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(SCRIPT_NAME, OUTPUTS_DIR)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available — this will run on CPU and be far slower.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_determinism(args.seed)

    train_keys, val_keys = load_split("train"), load_split("val")
    logger.info("loading %d training subjects", len(train_keys))
    train_data = build_split(train_keys, logger)
    logger.info("loading %d validation subjects", len(val_keys))
    val_subjects = [load_subject_slices(key) for key in val_keys]

    lesion_fraction = train_data["label"].sum() / train_data["label"].size
    logger.info("training slices: %d  |  lesion voxels: %.3f%% of all pixels",
                len(train_data["image"]), 100 * lesion_fraction)
    logger.info("this imbalance is why the loss is Dice+BCE and not BCE alone — "
                "plain BCE converges to predicting all-background here")

    dataset = TensorDataset(
        torch.from_numpy(train_data["image"]),
        torch.from_numpy(train_data["label"]),
        torch.from_numpy(train_data["loss_mask"]),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    model = UNet(in_channels=2).to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    tag = f"_{args.tag}" if args.tag else ""
    checkpoint_path = CHECKPOINT_DIR / f"unet_seed{args.seed}{tag}.pt"
    resume_path = CHECKPOINT_DIR / f"unet_seed{args.seed}{tag}.resume.pt"
    history, best_dice, best_epoch = [], -1.0, -1
    start_epoch, prior_minutes = 1, 0.0

    if args.resume and resume_path.exists():
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state["signature"] != run_signature(args):
            raise SystemExit(
                f"{resume_path.name} was written by a different configuration "
                f"({state['signature']} vs {run_signature(args)}). Delete it to "
                f"start fresh rather than resuming into a run that matches nothing.")
        model.load_state_dict(state["model"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        # RNG states must go back as CPU byte tensors regardless of where the
        # rest of the checkpoint was mapped, or set_state raises TypeError.
        generator.set_state(state["generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"])
        history = state["history"]
        best_dice, best_epoch = state["best_dice"], state["best_epoch"]
        start_epoch, prior_minutes = state["epoch"] + 1, state["minutes"]
        logger.info("resuming seed %d from epoch %d (best val Dice %.4f so far, "
                    "%.0f min already spent)", args.seed, start_epoch, best_dice,
                    prior_minutes)
    elif args.resume:
        logger.info("--resume given but no resume state at %s — starting fresh",
                    resume_path.name)

    started = time.time()

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_loss, n_batches = 0.0, 0
        for images, labels, loss_masks in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            loss_masks = loss_masks.to(device, non_blocking=True)
            if args.augment_strength > 0:
                images, labels, loss_masks = augment_for_domain_generalisation(
                    images, labels, loss_masks, generator, strength=args.augment_strength)
            else:
                images, labels, loss_masks = augment(images, labels, loss_masks, generator)

            optimiser.zero_grad(set_to_none=True)
            loss = masked_dice_bce_loss(model(images), labels, loss_masks)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        if epoch % 5 == 0 or epoch == args.epochs or epoch == 1:
            scores = validate(model, val_subjects, device)
            history.append({"epoch": epoch, "loss": epoch_loss / n_batches, **scores})
            marker = ""
            if scores["dice_mean"] > best_dice:
                best_dice, best_epoch = scores["dice_mean"], epoch
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_dice": best_dice, "seed": args.seed}, checkpoint_path)
                marker = "  <- best, saved"
            logger.info("epoch %3d/%d  loss %.4f  val Dice %.4f +- %.4f  %s%s",
                        epoch, args.epochs, epoch_loss / n_batches,
                        scores["dice_mean"], scores["dice_std"],
                        {k: round(v, 3) for k, v in scores["per_site"].items()}, marker)
        else:
            logger.info("epoch %3d/%d  loss %.4f", epoch, args.epochs, epoch_loss / n_batches)

        # Costs ~0.2 s against a ~70 s epoch. The machine hard-powered-off mid-run
        # on 2026-09-14 and cost 71 epochs; this caps that loss at one.
        save_resume_state(resume_path, model=model, optimiser=optimiser,
                          scheduler=scheduler, generator=generator, epoch=epoch,
                          best_dice=best_dice, best_epoch=best_epoch, history=history,
                          minutes=prior_minutes + (time.time() - started) / 60,
                          args=args)

    elapsed = prior_minutes + (time.time() - started) / 60
    resume_path.unlink(missing_ok=True)  # completed cleanly; nothing left to resume
    logger.info("done in %.1f min — best val Dice %.4f at epoch %d, saved to %s",
                elapsed, best_dice, best_epoch, checkpoint_path.name)

    history_path = OUTPUTS_DIR / f"route_c_history_seed{args.seed}{tag}.json"
    history_path.write_text(json.dumps(
        {"args": vars(args), "best_dice": best_dice, "best_epoch": best_epoch,
         "minutes": elapsed, "history": history}, indent=2))
    write_manifest(history_path, generating_script=f"code/segmentation/{SCRIPT_NAME}.py")

    logger.info("Route A baseline for comparison: val Dice 0.4291 "
                "(threshold 1.40, min size 5). Improvement: %+.4f", best_dice - 0.4291)


if __name__ == "__main__":
    main()
