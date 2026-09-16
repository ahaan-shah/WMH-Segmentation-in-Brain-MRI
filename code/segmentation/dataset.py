"""Week 3, Route C — turning subjects into trainable 2D slices.

**Why 2D and not 3D.** Slices are 3.00 mm thick against 0.56-1.30 mm in-plane.
A 3x3x3 convolution therefore spans roughly three times further through the
brain than across it, so a 3D kernel is not looking at a cube of anatomy — it is
looking at a slab. The challenge winners (sysu_media, pgs) were all 2D or 2.5D
for this reason, and ROADMAP 6.2 follows them.

**Two input channels: normalised FLAIR and normalised T1.** FLAIR is where
lesions are visible; T1 is what distinguishes a genuine lesion from something
merely bright, because WMH are hypointense on T1 while many false positives are
not. The T1 is already on the FLAIR grid, so the two channels align voxel for
voxel with no registration.

T1 is normalised the same way the FLAIR was in Week 2 — divided by the median
intensity of its own normal-appearing white matter — so both channels are on a
scanner-independent scale rather than one being raw counts.

**Label 2 is excluded from the loss, not treated as background.** The official
scorer masks predictions wherever the reference is label 2, so a prediction
there is neither rewarded nor punished. Training with those voxels as background
would actively teach the network to avoid regions the metric does not care
about, which is a real and avoidable mistake. Every sample therefore carries a
`loss_mask` that is zero on label-2 voxels.

**Padding, never resizing.** Slices differ in size across sites (240x240,
232x256, 132x256, ...). They are zero-padded or centre-cropped to a fixed square,
which preserves every voxel exactly. Resizing would interpolate, and the whole
pipeline has avoided resampling since Week 1 precisely because the reference
standard lives on this grid.
"""

from __future__ import annotations

import numpy as np

from metadata.derived import BRAIN_MASK, FLAIR_NORM, TISSUE_SEG, load_derived, load_derived_mask
from metadata.loader import load_nifti, load_raw_mask_array, subjects_by_key
from preprocessing.normalise import normalise
from preprocessing.tissue_seg import LABEL_WM

# Every in-plane dimension in the training set is <= 256, so this pads rather
# than crops for all 60. The test split contains one 321-wide acquisition, which
# centre-crops around the brain — handled, and flagged in Week 7's report.
PATCH_SIZE = 256


def pad_or_crop(array: np.ndarray, size: int = PATCH_SIZE, *, centre=None) -> np.ndarray:
    """Centre a 2D slice in a `size` x `size` field without interpolating.

    Returns a new array; the original voxels are copied verbatim. `centre` lets
    the caller centre on the brain rather than the image when cropping is needed.
    """
    height, width = array.shape
    output = np.zeros((size, size), dtype=array.dtype)

    cy, cx = (height // 2, width // 2) if centre is None else centre
    top = int(cy - size // 2)
    left = int(cx - size // 2)

    source_top, source_left = max(0, top), max(0, left)
    source_bottom, source_right = min(height, top + size), min(width, left + size)
    target_top, target_left = source_top - top, source_left - left

    output[target_top:target_top + (source_bottom - source_top),
           target_left:target_left + (source_right - source_left)] = \
        array[source_top:source_bottom, source_left:source_right]
    return output


def undo_pad_or_crop(padded: np.ndarray, shape: tuple[int, int], *, centre=None) -> np.ndarray:
    """Inverse of `pad_or_crop`, so a prediction lands back on the FLAIR grid."""
    height, width = shape
    size = padded.shape[0]
    output = np.zeros(shape, dtype=padded.dtype)

    cy, cx = (height // 2, width // 2) if centre is None else centre
    top, left = int(cy - size // 2), int(cx - size // 2)

    source_top, source_left = max(0, top), max(0, left)
    source_bottom, source_right = min(height, top + size), min(width, left + size)
    target_top, target_left = source_top - top, source_left - left

    output[source_top:source_bottom, source_left:source_right] = \
        padded[target_top:target_top + (source_bottom - source_top),
               target_left:target_left + (source_right - source_left)]
    return output


def normalised_t1(key: str, subject, brain_mask: np.ndarray) -> np.ndarray:
    """T1 on the same white-matter-referenced scale the FLAIR uses.

    Reuses Week 2's normalisation with the tissue map's white-matter class as the
    reference, so both channels mean the same kind of thing: 1.0 is that
    subject's own normal-appearing white matter.
    """
    t1 = np.asarray(load_nifti(subject.t1_path).dataobj, dtype=np.float64)
    tissue = np.asarray(load_derived(key, TISSUE_SEG).dataobj)
    normalised_volume, _ = normalise(
        t1, brain_mask, method="wm_referenced", reference_mask=(tissue == LABEL_WM)
    )
    return normalised_volume


def load_subject_slices(key: str, *, with_labels: bool = True) -> dict:
    """All brain-containing axial slices of one subject, ready for the network.

    Returns arrays shaped (n_slices, 2, PATCH_SIZE, PATCH_SIZE) for the image and
    (n_slices, 1, PATCH_SIZE, PATCH_SIZE) for the label, loss mask and brain mask,
    plus the metadata needed to put a prediction back on the original grid.
    """
    subject = subjects_by_key()[key]
    flair = np.asarray(load_derived(key, FLAIR_NORM).dataobj, dtype=np.float32)
    brain = load_derived_mask(key, BRAIN_MASK)
    t1 = normalised_t1(key, subject, brain).astype(np.float32)

    if with_labels:
        raw_reference, _ = load_raw_mask_array(subject.mask_path)
        label = (raw_reference == 1)
        # Label 2 is "don't care" to the official scorer, so it must be
        # "don't care" to the loss as well.
        loss_mask = (raw_reference != 2)
    else:
        label = np.zeros_like(brain)
        loss_mask = np.ones_like(brain)

    # Only slices containing brain. Empty slices teach nothing and would skew
    # the already-extreme class balance further towards background.
    z_indices = [z for z in range(brain.shape[2]) if brain[:, :, z].any()]
    if not z_indices:
        raise ValueError(f"{key}: brain mask is empty on every slice")

    # Crop/pad centred on the brain, not the image, so nothing is lost when a
    # slice is wider than PATCH_SIZE.
    coords = np.argwhere(brain.any(axis=2))
    centre = tuple(coords.mean(axis=0).round().astype(int))

    images, labels, loss_masks, brains = [], [], [], []
    for z in z_indices:
        images.append(np.stack([pad_or_crop(flair[:, :, z], centre=centre),
                                pad_or_crop(t1[:, :, z], centre=centre)]))
        labels.append(pad_or_crop(label[:, :, z].astype(np.float32), centre=centre)[None])
        loss_masks.append(pad_or_crop(loss_mask[:, :, z].astype(np.float32), centre=centre)[None])
        brains.append(pad_or_crop(brain[:, :, z].astype(np.float32), centre=centre)[None])

    return {
        "key": key,
        "site": subject.site,
        "image": np.stack(images).astype(np.float32),
        "label": np.stack(labels).astype(np.float32),
        "loss_mask": np.stack(loss_masks).astype(np.float32),
        "brain": np.stack(brains).astype(np.float32),
        "z_indices": np.array(z_indices),
        "volume_shape": brain.shape,
        "centre": centre,
    }


def build_split(keys: list[str], logger=None) -> dict:
    """Concatenate every subject's slices into one training array."""
    images, labels, loss_masks, sources = [], [], [], []
    for index, key in enumerate(keys, start=1):
        data = load_subject_slices(key)
        images.append(data["image"])
        labels.append(data["label"])
        loss_masks.append(data["loss_mask"])
        sources.extend([key] * len(data["image"]))
        if logger and (index % 12 == 0 or index == len(keys)):
            logger.info("  loaded %d/%d subjects", index, len(keys))

    return {
        "image": np.concatenate(images),
        "label": np.concatenate(labels),
        "loss_mask": np.concatenate(loss_masks),
        "subject_of_slice": np.array(sources),
    }
