"""numpy <-> SimpleITK conversion, with the axis order made explicit.

SimpleITK and nibabel disagree about array layout, and the disagreement is
silent: nibabel gives `(x, y, z)`, while `sitk.GetArrayFromImage` gives
`(z, y, x)`. Every volume in this dataset is anisotropic (z = 3.00 mm against
0.56-1.30 mm in-plane), so getting this wrong does not throw — it produces an
image whose spacing is attached to the wrong axes, and every downstream
physical measurement is wrong by up to 5.4x while looking entirely plausible.

Worse, on Utrecht and Singapore the in-plane dimensions happen to be square
(240x240x48, 232x256x48), so a transposed volume can even keep a valid-looking
shape. Routing all conversion through these two functions is what stops that.

Spacing is always attached explicitly, never inherited: N4's B-spline control
point mesh, the anisotropic diffusion filters and the distance transforms all
work in physical units, so an image carrying SimpleITK's default (1, 1, 1)
spacing silently changes what those algorithms compute.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

# nibabel (x, y, z) <-> SimpleITK (z, y, x). Self-inverse, but named in both
# directions so call sites read unambiguously.
_TO_SITK = (2, 1, 0)
_FROM_SITK = (2, 1, 0)


def to_sitk(
    array: np.ndarray,
    spacing: tuple[float, float, float],
    *,
    dtype=sitk.sitkFloat32,
) -> sitk.Image:
    """Wrap a nibabel-ordered (x, y, z) array as a SimpleITK image with spacing.

    The returned image's `GetSize()` is `(x, y, z)` — i.e. it matches the input
    array's shape, not the transposed buffer SimpleITK stores internally.
    """
    array = np.asarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {array.shape}")

    image = sitk.GetImageFromArray(np.ascontiguousarray(np.transpose(array, _TO_SITK)))
    image.SetSpacing(tuple(float(s) for s in spacing))

    if tuple(image.GetSize()) != tuple(array.shape):
        raise AssertionError(
            f"Axis order lost in conversion: SimpleITK size {image.GetSize()} "
            f"!= numpy shape {array.shape}"
        )
    return sitk.Cast(image, dtype) if dtype is not None else image


def from_sitk(image: sitk.Image) -> np.ndarray:
    """Unwrap a SimpleITK image back to a nibabel-ordered (x, y, z) array."""
    array = np.transpose(sitk.GetArrayFromImage(image), _FROM_SITK)
    if tuple(array.shape) != tuple(image.GetSize()):
        raise AssertionError(
            f"Axis order lost in conversion: numpy shape {array.shape} "
            f"!= SimpleITK size {image.GetSize()}"
        )
    return array
