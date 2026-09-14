"""Scanner-simulating augmentation — teaching the network to ignore appearance.

**Why this exists, and why it is a revision of a Week 3 decision.**

Week 3 deliberately used weak augmentation: left-right flips and a +-5% gain,
nothing more. The stated reasoning was that Week 2's normalisation had already
put every scanner on one intensity scale, so strong jitter would throw that
alignment away.

Leave-one-site-out partly falsified that. Hiding a scanner costs roughly 0.06-0.09
Dice, so the network is still keying off scanner-specific appearance that
normalisation did not remove. That is not a failure of normalisation — Week 2
measured cross-site agreement of lesion intensity at 0.074, so the *scale* really
is aligned. What remains is everything a scale factor cannot touch: noise
texture, edge sharpness, the contrast *relationships* between tissues, and
partial-volume behaviour at boundaries. Those differ per scanner and survive any
amount of rescaling.

**The mechanism.** Show the network the same brain many times with different
appearances while the correct answer stays identical. Appearance then stops
predicting the label reliably, and the only features that survive across every
variant are shape, location and how a region relates to its neighbours. It is a
pressure, not a guarantee — networks take the path of least resistance, and this
makes the appearance shortcut a worse path rather than closing it.

**The failure mode to respect.** Push the jitter too hard and lesions stop being
distinguishable from normal tissue at all, destroying the signal. `strength`
exists so that trade-off can be swept rather than guessed.

**How to judge it: leave-one-site-out, NOT ordinary validation.** The expected
outcome is in-domain Dice flat or slightly down, out-of-domain Dice up. Measured
the usual way, a successful change would look like a failure. Same trap as Week
2's denoising test, where SSIM reported "nothing changed" while 399 lesions were
being erased — the metric has to match the question being asked.

Every transform is applied **per channel independently**, because FLAIR and T1
are different sequences acquired with different parameters and a scanner does
not distort them identically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _random(shape, generator, device, low=0.0, high=1.0):
    return low + (high - low) * torch.rand(shape, generator=generator, device=device)


def simulated_bias_field(images: torch.Tensor, generator, magnitude: float) -> torch.Tensor:
    """A smooth, low-frequency multiplicative field, like residual scanner shading.

    Built by upsampling a tiny random grid, which guarantees the field is smooth
    and cannot follow anatomy — the same property real bias fields have, and the
    reason N4 can model them in the first place.
    """
    if magnitude <= 0:
        return images
    batch, channels, height, width = images.shape
    coarse = _random((batch, channels, 4, 4), generator, images.device,
                     1.0 - magnitude, 1.0 + magnitude)
    field = F.interpolate(coarse, size=(height, width), mode="bicubic", align_corners=False)
    return images * field


def random_gamma(images: torch.Tensor, generator, spread: float) -> torch.Tensor:
    """Bend the intensity curve, changing contrast BETWEEN tissues.

    This is the most important transform here. A gain change moves every tissue
    together and normalisation already handles it; gamma changes how grey matter
    sits relative to white matter relative to lesion — exactly the scanner-
    dependent relationship that survives normalisation.

    Applied only to positive values: the background is exactly 0 outside the
    brain after Week 2, and 0**gamma is 0, so it is left untouched either way.
    """
    if spread <= 0:
        return images
    batch, channels = images.shape[:2]
    gamma = _random((batch, channels, 1, 1), generator, images.device,
                    1.0 - spread, 1.0 + spread)
    return torch.where(images > 0, images.clamp(min=1e-6) ** gamma, images)


def random_noise(images: torch.Tensor, generator, sigma_max: float) -> torch.Tensor:
    """Additive Gaussian noise at a random level per sample.

    Scanners differ in signal-to-noise; Week 2 measured lesion-to-WM CNR at 7.0
    (Utrecht) against 21.7 (Amsterdam), a 3x spread. Varying noise during
    training stops the network from relying on one site's noise floor.
    """
    if sigma_max <= 0:
        return images
    batch, channels = images.shape[:2]
    sigma = _random((batch, channels, 1, 1), generator, images.device, 0.0, sigma_max)
    noise = torch.randn(images.shape, generator=generator, device=images.device)
    return images + noise * sigma


def random_blur(images: torch.Tensor, generator, probability: float) -> torch.Tensor:
    """Occasional mild smoothing, standing in for differing effective resolution.

    Deliberately mild and occasional. Week 2 established that aggressive
    smoothing destroys small lesions — half of them are 5 voxels or smaller —
    so this is a light touch applied to a minority of samples, not a default.
    """
    if probability <= 0:
        return images
    batch = images.shape[0]
    apply = torch.rand(batch, generator=generator, device=images.device) < probability
    if not apply.any():
        return images
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
                          device=images.device) / 16.0
    channels = images.shape[1]
    kernel = kernel.expand(channels, 1, 3, 3)
    blurred = F.conv2d(images, kernel, padding=1, groups=channels)
    mask = apply.view(-1, 1, 1, 1).float()
    return images * (1 - mask) + blurred * mask


def augment_for_domain_generalisation(
    images: torch.Tensor,
    labels: torch.Tensor,
    masks: torch.Tensor,
    generator,
    *,
    strength: float = 1.0,
):
    """Geometry plus scanner simulation. `strength` scales every intensity effect.

    `strength=0` reduces to Week 3's original behaviour (flip + mild gain), so
    the two can be compared without changing anything else.

    Only the flip touches geometry, so labels and masks move with it; every
    other transform alters appearance alone and leaves the answer unchanged —
    which is precisely the point.
    """
    # --- geometry: flip. Safe for segmentation, NEVER for Week 4 hemisphere
    # features, where it would invert left and right.
    flip = torch.rand(images.shape[0], generator=generator, device=images.device) < 0.5
    if flip.any():
        images[flip] = torch.flip(images[flip], dims=[-1])
        labels[flip] = torch.flip(labels[flip], dims=[-1])
        masks[flip] = torch.flip(masks[flip], dims=[-1])

    if strength <= 0:
        gain = 1.0 + 0.05 * (2 * _random((images.shape[0], 1, 1, 1),
                                         generator, images.device) - 1)
        return images * gain, labels, masks

    # --- appearance: the answer is unchanged, so only the image is touched ---
    gain = _random((images.shape[0], images.shape[1], 1, 1), generator, images.device,
                   1.0 - 0.15 * strength, 1.0 + 0.15 * strength)
    images = images * gain
    images = random_gamma(images, generator, spread=0.25 * strength)
    images = simulated_bias_field(images, generator, magnitude=0.15 * strength)
    images = random_noise(images, generator, sigma_max=0.05 * strength)
    images = random_blur(images, generator, probability=0.15 * strength)

    return images, labels, masks
