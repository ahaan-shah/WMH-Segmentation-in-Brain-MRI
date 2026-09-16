"""Week 3, Route C — the 2D U-Net and its loss.

Architecture follows ROADMAP 6.2 and the challenge winners: a standard 2D U-Net
on axial slices with two input channels (normalised FLAIR + normalised T1). No
architectural novelty is attempted — the marks here are for a justified,
correctly-validated pipeline, not a new network, and a standard U-Net is the
thing the leaderboard was built on.

**The loss is the part that actually matters.**

WMH occupy well under 1% of brain voxels. Plain binary cross-entropy on that
balance converges to predicting all-background, which scores ~99.5% pixel
accuracy and is completely useless — this is the single most common way a lesion
segmentation network fails. So the loss is **Dice + BCE combined**: the Dice term
is scale-invariant with respect to class balance and supplies gradient even when
the prediction is nearly empty, while BCE keeps per-pixel calibration sane.

**Every loss term is masked.** Two masks, for two different reasons:

- Label-2 voxels ("other pathology") are excluded, because the official scorer
  treats them as don't-care. Training them as background teaches the network to
  avoid regions the metric ignores.
- Everything outside the brain is excluded, because it is zero by construction
  after Week 2 and contributes nothing but an enormous, trivially-solved
  background class.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with batch norm — the standard U-Net block."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """2D U-Net. Returns raw logits — apply a sigmoid to get probabilities.

    `base_filters=32` with four down/up levels fits comfortably in the 6 GB of an
    RTX 3050 at 256x256 with a batch of 16, leaving room for the ensembling that
    ROADMAP 6.2 identifies as the cheapest available accuracy gain.
    """

    def __init__(self, in_channels: int = 2, base_filters: int = 32, depth: int = 4):
        super().__init__()
        self.depth = depth

        self.encoders = nn.ModuleList()
        channels = in_channels
        for level in range(depth):
            out_channels = base_filters * (2 ** level)
            self.encoders.append(DoubleConv(channels, out_channels))
            channels = out_channels

        self.bottleneck = DoubleConv(channels, channels * 2)

        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = channels * 2
        for level in reversed(range(depth)):
            skip_channels = base_filters * (2 ** level)
            self.upsamples.append(
                nn.ConvTranspose2d(channels, skip_channels, kernel_size=2, stride=2)
            )
            self.decoders.append(DoubleConv(skip_channels * 2, skip_channels))
            channels = skip_channels

        self.head = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)

        x = self.bottleneck(x)

        for upsample, decoder, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = upsample(x)
            x = decoder(torch.cat([skip, x], dim=1))

        return self.head(x)


def masked_dice_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Dice + BCE, with both terms restricted to valid voxels.

    `loss_mask` is 1 where the voxel counts and 0 where it must be ignored —
    label-2 regions and anything outside the brain. Multiplying rather than
    indexing keeps the operation batched and shape-stable.

    Dice is computed over the whole batch rather than per-slice. Per-slice Dice
    is undefined on the many slices containing no lesion at all, and averaging
    those in swamps the signal from the slices that do.
    """
    probabilities = torch.sigmoid(logits)

    valid = loss_mask
    intersection = (probabilities * target * valid).sum()
    denominator = (probabilities * valid).sum() + (target * valid).sum()
    dice_loss = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)

    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    valid_total = valid.sum().clamp(min=1.0)
    bce_loss = (bce * valid).sum() / valid_total

    return dice_weight * dice_loss + bce_weight * bce_loss


@torch.no_grad()
def dice_on_batch(logits, target, loss_mask, threshold: float = 0.5) -> tuple[float, float]:
    """Hard-thresholded Dice over a batch. Returns (intersection, denominator).

    Returned as the two sums rather than a ratio so an epoch-level Dice can be
    accumulated correctly — averaging per-batch Dice values is not the same
    number and is wrong whenever batches differ in lesion content.
    """
    prediction = (torch.sigmoid(logits) > threshold).float() * loss_mask
    reference = target * loss_mask
    intersection = 2.0 * (prediction * reference).sum().item()
    denominator = prediction.sum().item() + reference.sum().item()
    return intersection, denominator
