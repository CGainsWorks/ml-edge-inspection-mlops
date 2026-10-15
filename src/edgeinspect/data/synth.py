"""A deterministic synthetic surface-inspection dataset.

Why synthetic rather than MVTec AD or DAGM by default
-----------------------------------------------------
Two reasons, and only the second is about convenience.

1.  **Licence.** MVTec AD is CC BY-NC-SA 4.0 and DAGM 2007 is research-use
    only.  Neither may be redistributed inside a repository that is intended to
    be usable commercially, so neither can ship here.  A real loader for both
    lives in :mod:`edgeinspect.data.mvtec` and works against a local copy the
    user downloads and accepts the licence for.

2.  **Drift has to be causal to be testable.** This project's claim is that a
    drift detector fires on a real distribution shift and that retraining
    recovers the loss.  To test that, the shift has to be something that is
    *applied*, with a known cause and a known magnitude, not something found by
    chance in a fixed public dataset.  A generator parameterised by imaging
    conditions -- exposure, focus, sensor noise, surface finish -- gives that.
    ``configs/data.yaml`` describes the drifted domains as what went wrong on
    the line, and the pixels follow.

The trade is stated plainly in ``docs/limitations.md``: absolute accuracy
numbers on synthetic textures do not transfer to a real production line.  The
*operational* numbers -- detection latency of the drift monitor, recovery after
retraining, INT8 accuracy delta -- are the ones this repository is making
claims about.

Determinism
-----------
Every frame is a pure function of ``(dataset_seed, split, index, domain)``.
Nothing depends on generation order, so a single frame can be regenerated
without regenerating the split, and two machines produce byte-identical PNGs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from ..config import DomainSpec

# The defect vocabulary.  Names are the ones an inspection engineer would use,
# and each has a different failure signature under blur and INT8 quantisation,
# which is the point of having more than one.
DEFECT_KINDS: Tuple[str, ...] = ("scratch", "pit", "dent", "inclusion")

IMAGE_SIZE = 256


def frame_seed(dataset_seed: int, split: str, index: int, domain: str) -> int:
    """A stable 63-bit seed for one frame.

    Hashing rather than ``dataset_seed + index`` because the latter makes
    ``train[5]`` and ``val[4]`` correlated for adjacent seeds, and correlated
    train and validation images are the quietest way to invent a good number.
    """
    payload = "{0}|{1}|{2}|{3}".format(dataset_seed, split, index, domain).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


@dataclass(frozen=True)
class Defect:
    """One defect instance, in pixel coordinates of the generated image."""

    kind: str
    x0: int
    y0: int
    x1: int
    y1: int
    area_px: int

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "box": [self.x0, self.y0, self.x1, self.y1],
            "area_px": self.area_px,
        }


@dataclass(frozen=True)
class Frame:
    """A generated frame: the image, its defect mask and its boxes."""

    image: np.ndarray  # uint8, (H, W)
    mask: np.ndarray   # uint8, (H, W), 255 where defective
    defects: List[Defect]
    domain: str

    @property
    def label(self) -> int:
        return 1 if self.defects else 0


def _correlated_noise(rng: np.random.Generator, size: int, scale: float) -> np.ndarray:
    """Band-limited noise: white noise through a Gaussian of the given scale.

    ``scale`` is the correlation length in pixels, so a scale of 7 produces
    blotches roughly seven pixels across, which is what a cast or rolled metal
    surface looks like at this magnification.
    """
    white = rng.standard_normal((size, size)).astype(np.float32)
    field = ndimage.gaussian_filter(white, sigma=max(scale, 1e-3) / 2.0, mode="reflect")
    std = float(field.std())
    if std < 1e-8:
        return np.zeros_like(field)
    return field / std


def _surface(rng: np.random.Generator, size: int, domain: DomainSpec) -> np.ndarray:
    """The defect-free surface: correlated grain plus directional machining marks."""
    grain = _correlated_noise(rng, size, domain.grain_scale) * domain.grain_strength

    # Machining marks: a sinusoid at a random orientation, with the phase and
    # angle drawn per frame so the model cannot memorise one texture.
    angle = float(rng.uniform(0.0, np.pi))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    projected = xx * np.cos(angle) + yy * np.sin(angle)
    period = max(float(domain.ripple_period), 2.0)
    ripple = np.sin(2.0 * np.pi * projected / period + phase) * domain.ripple_strength

    # A slow low-frequency undulation stands in for sheet flatness variation.
    undulation = _correlated_noise(rng, size, size / 4.0) * 0.04

    return 0.5 + grain + ripple + undulation


def _ellipse_mask(size: int, cy: float, cx: float, ry: float, rx: float,
                  angle: float) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dy = yy - cy
    dx = xx - cx
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a
    return ((u / max(rx, 0.5)) ** 2 + (v / max(ry, 0.5)) ** 2) <= 1.0


def _draw_scratch(rng: np.random.Generator, size: int, field: np.ndarray,
                  contrast: float) -> np.ndarray:
    """A thin, long, low-contrast line.  The hardest kind to keep through INT8."""
    length = float(rng.uniform(size * 0.12, size * 0.34))
    width = float(rng.uniform(0.8, 2.0))
    angle = float(rng.uniform(0.0, np.pi))
    cy = float(rng.uniform(size * 0.15, size * 0.85))
    cx = float(rng.uniform(size * 0.15, size * 0.85))
    footprint = _ellipse_mask(size, cy, cx, width, length / 2.0, angle)
    # A scratch is a groove: dark core with a bright burr on one side.
    sign = -1.0 if rng.random() < 0.75 else 1.0
    field[footprint] += sign * contrast * float(rng.uniform(0.8, 1.2))
    return footprint


def _draw_pit(rng: np.random.Generator, size: int, field: np.ndarray,
              contrast: float) -> np.ndarray:
    """A small, round, high-contrast hole."""
    radius = float(rng.uniform(2.5, 5.5))
    cy = float(rng.uniform(radius + 4, size - radius - 4))
    cx = float(rng.uniform(radius + 4, size - radius - 4))
    footprint = _ellipse_mask(size, cy, cx, radius, radius * float(rng.uniform(0.8, 1.25)), 0.0)
    field[footprint] -= contrast * float(rng.uniform(1.1, 1.6))
    return footprint


def _draw_dent(rng: np.random.Generator, size: int, field: np.ndarray,
               contrast: float) -> np.ndarray:
    """A broad, smooth, very low-contrast depression.

    Nearly invisible frame by frame and the first thing lost when the lens goes
    soft, which is exactly why it is in the vocabulary.
    """
    radius = float(rng.uniform(9.0, 20.0))
    cy = float(rng.uniform(radius + 4, size - radius - 4))
    cx = float(rng.uniform(radius + 4, size - radius - 4))
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (radius ** 2)
    bump = np.exp(-2.0 * r2).astype(np.float32)
    field -= bump * contrast * float(rng.uniform(0.45, 0.7))
    return r2 <= 1.0


def _draw_inclusion(rng: np.random.Generator, size: int, field: np.ndarray,
                    contrast: float) -> np.ndarray:
    """A blotchy foreign body: irregular outline, texture of its own."""
    radius = float(rng.uniform(5.0, 11.0))
    cy = float(rng.uniform(radius + 4, size - radius - 4))
    cx = float(rng.uniform(radius + 4, size - radius - 4))
    angle = float(rng.uniform(0.0, np.pi))
    base = _ellipse_mask(size, cy, cx, radius, radius * float(rng.uniform(0.6, 1.4)), angle)
    ragged = _correlated_noise(rng, size, 3.0) > -0.2
    footprint = base & ragged
    if not footprint.any():
        footprint = base
    sign = 1.0 if rng.random() < 0.5 else -1.0
    texture = _correlated_noise(rng, size, 2.0) * 0.25
    field[footprint] += (sign * contrast * float(rng.uniform(0.9, 1.3))) + texture[footprint]
    return footprint


_DRAW = {
    "scratch": _draw_scratch,
    "pit": _draw_pit,
    "dent": _draw_dent,
    "inclusion": _draw_inclusion,
}


def _bounding_box(footprint: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.nonzero(footprint)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def generate_frame(seed: int, domain: DomainSpec, size: int = IMAGE_SIZE,
                   force_label: int = -1,
                   kinds: Sequence[str] = DEFECT_KINDS) -> Frame:
    """Generate one frame under the given imaging conditions.

    ``force_label`` of 0 or 1 overrides ``domain.defect_rate`` for that frame,
    which is how the evaluation splits get an exactly balanced class mix while
    a production stream keeps a realistic, mostly-good one.

    The order of operations matters and mirrors the physics: the defect is part
    of the surface, so it is added *before* the lens blur and the sensor noise.
    Blurring the clean surface and pasting a sharp defect on top produces a
    dataset where defects are trivially detectable by high-frequency energy
    alone, and a model trained on it collapses the moment a real lens is
    slightly out of focus.
    """
    rng = np.random.default_rng(seed)
    field = _surface(rng, size, domain)

    if force_label < 0:
        has_defect = bool(rng.random() < domain.defect_rate)
    else:
        has_defect = bool(force_label)

    mask = np.zeros((size, size), dtype=bool)
    defects: List[Defect] = []
    if has_defect:
        count = 1 if rng.random() < 0.75 else 2
        for _ in range(count):
            kind = str(kinds[int(rng.integers(0, len(kinds)))])
            footprint = _DRAW[kind](rng, size, field, domain.defect_contrast)
            footprint = np.asarray(footprint, dtype=bool)
            if not footprint.any():
                continue
            x0, y0, x1, y1 = _bounding_box(footprint)
            defects.append(Defect(kind, x0, y0, x1, y1, int(footprint.sum())))
            mask |= footprint

    # Optics: the defect goes through the same lens as the surface.
    if domain.blur_sigma > 0:
        field = ndimage.gaussian_filter(field, sigma=domain.blur_sigma, mode="reflect")

    # Illumination: radial falloff, then exposure gain and black level.
    if domain.vignette > 0:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        cy = cx = (size - 1) / 2.0
        r2 = ((yy - cy) ** 2 + (xx - cx) ** 2) / (cy ** 2 + cx ** 2)
        field = field * (1.0 - domain.vignette * r2)
    field = field * domain.gain + domain.offset

    # Sensor: additive read noise, then 8-bit quantisation.
    if domain.noise_sigma > 0:
        field = field + rng.standard_normal((size, size)).astype(np.float32) * domain.noise_sigma

    image = np.clip(field * 255.0, 0, 255).astype(np.uint8)
    return Frame(image=image, mask=(mask.astype(np.uint8) * 255), defects=defects,
                 domain=domain.name)


def generate_split(dataset_seed: int, split: str, count: int, domain: DomainSpec,
                   size: int = IMAGE_SIZE, balanced: bool = True):
    """Yield ``count`` frames for a named split.

    With ``balanced`` the defect and good frames alternate deterministically,
    so the class ratio of an evaluation split does not wobble with the seed.
    """
    for index in range(count):
        seed = frame_seed(dataset_seed, split, index, domain.name)
        force = (index % 2) if balanced else -1
        yield index, generate_frame(seed, domain, size=size, force_label=force)
