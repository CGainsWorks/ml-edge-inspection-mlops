"""Augmentation.

Two rules govern what is in here.

**Geometry is applied to the image and the mask together.**  A pipeline that
flips the image and forgets the mask trains the model to predict defects in the
wrong place, and the loss still goes down, so it is not obvious from a training
curve.  :func:`apply` takes both and returns both.

**Photometric augmentation is bounded by the drift domains.**  If training
augmentation covers every exposure and blur the drifted domain produces, the
model is invariant to the drift and the drift detector has nothing to detect --
the whole experiment becomes untestable.  ``configs/train.yaml`` keeps the
photometric ranges deliberately narrower than the shift in
``configs/data.yaml``, and ``tests/test_augment.py`` asserts that relationship
so it cannot be widened by accident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class AugmentConfig:
    """Photometric ranges are half-widths around the identity."""

    hflip: bool = True
    vflip: bool = True
    rot90: bool = True
    brightness: float = 0.06   # additive, in units of the [0, 1] image range
    contrast: float = 0.08     # multiplicative, 1 +/- this
    noise: float = 0.010       # extra Gaussian noise, standard deviation
    blur: float = 0.25         # extra Gaussian blur, sigma, uniform in [0, this]
    enabled: bool = True

    @classmethod
    def from_dict(cls, data: Mapping) -> "AugmentConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                "unknown augmentation field(s): {0}".format(sorted(unknown))
            )
        return cls(**dict(data))


def apply(image: np.ndarray, mask: np.ndarray, config: AugmentConfig,
          rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Augment one ``(1, H, W)`` image and its ``(1, H, W)`` mask."""
    if not config.enabled:
        return image, mask

    out = np.array(image, dtype=np.float32, copy=True)
    out_mask = np.array(mask, dtype=np.float32, copy=True)

    # Geometry: image and mask, always together.
    if config.hflip and rng.random() < 0.5:
        out = out[:, :, ::-1]
        out_mask = out_mask[:, :, ::-1]
    if config.vflip and rng.random() < 0.5:
        out = out[:, ::-1, :]
        out_mask = out_mask[:, ::-1, :]
    if config.rot90:
        turns = int(rng.integers(0, 4))
        if turns:
            out = np.rot90(out, turns, axes=(1, 2))
            out_mask = np.rot90(out_mask, turns, axes=(1, 2))
    out = np.ascontiguousarray(out)
    out_mask = np.ascontiguousarray(out_mask)

    # Photometry: image only.  The mask is a label, not a picture.
    if config.contrast > 0:
        scale = 1.0 + float(rng.uniform(-config.contrast, config.contrast))
        out = (out - 0.5) * scale + 0.5
    if config.brightness > 0:
        out = out + float(rng.uniform(-config.brightness, config.brightness))
    if config.blur > 0:
        sigma = float(rng.uniform(0.0, config.blur))
        if sigma > 0.05:
            out = ndimage.gaussian_filter(out, sigma=(0.0, sigma, sigma), mode="reflect")
    if config.noise > 0:
        out = out + rng.standard_normal(out.shape).astype(np.float32) * config.noise

    return np.clip(out, 0.0, 1.0).astype(np.float32), out_mask
