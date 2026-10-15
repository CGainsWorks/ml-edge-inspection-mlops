"""The features the edge device reports for every frame.

These are what the drift detector actually watches, so the constraint is that
they must be computable on the device inside the inference budget.  Everything
here is a few NumPy passes over a 256x256 array: about 1.5 ms per frame on the
development machine, measured by scripts/benchmark_features.py, against 2.4 ms
for the model itself.  The edge runtime reports the two separately rather than
folding telemetry cost into inference cost.

Why image statistics and not embeddings
---------------------------------------
An embedding-based drift detector is more sensitive and needs the model's
intermediate activations, a reference embedding set on the device, and a
distance metric whose scale nobody can interpret when it fires at 03:00.  These
eight numbers are interpretable: ``brightness`` fell, ``sharpness`` fell and
``noise_level`` rose means the lamp is dying and the camera is compensating with
gain, and that is an instruction rather than an alert.  The model's own ``score``
and flagged rate travel in the same telemetry record and cover what image
statistics cannot.

The trade is stated in ``docs/drift.md``: this set is blind to a shift that
preserves first- and second-order statistics while changing structure.  The
``material_batch`` domain is exactly that case, which is why ``texture_scale``
is in the set.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

# The names every downstream component uses.  Adding one here means the drift
# rules in configs/drift.yaml can reference it; renaming one breaks a saved
# reference window, which is why the reference stores the names it was built on.
FEATURE_NAMES: Sequence[str] = (
    "brightness",
    "contrast",
    "sharpness",
    "noise_level",
    "edge_density",
    "entropy",
    "highlight_fraction",
    "texture_scale",
)

_LAPLACIAN = np.asarray([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
_BOX = np.full((3, 3), 1.0 / 9.0, dtype=np.float32)

# Immerkaer's noise mask: a second difference in both directions, so any locally
# linear patch of image convolves to zero and only noise survives.
_IMMERKAER = np.asarray([[1.0, -2.0, 1.0],
                         [-2.0, 4.0, -2.0],
                         [1.0, -2.0, 1.0]], dtype=np.float32)
# Its squared coefficients sum to 36, so iid noise of standard deviation s
# convolves to standard deviation 6s.  For a zero-mean normal, E|X| is
# s * sqrt(2/pi), hence the sqrt(pi/2) on the way back.
_IMMERKAER_GAIN = 6.0
_ABS_TO_SIGMA = float(np.sqrt(np.pi / 2.0))

# Smoothing alone does not make the focus measure noise-free, it only makes it
# less noisy: convolving the 3x3 box with the 4-neighbour Laplacian gives a 5x5
# kernel whose squared coefficients sum to 32/81, so iid noise of standard
# deviation s still contributes exactly (32/81) * s^2 to the measured variance.
# Since noise_level already estimates s, that contribution is subtracted rather
# than tolerated, and the remainder is image structure.  Verified numerically in
# tests/test_features.py against synthetic noise of a known standard deviation.
_SHARPNESS_NOISE_GAIN = 32.0 / 81.0


def _convolve3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """A 3x3 valid convolution as nine shifted adds.

    Faster than ``scipy.signal.convolve2d`` at this size, and it keeps the
    per-frame feature cost well under the inference cost, which is the point.
    """
    out = np.zeros((image.shape[0] - 2, image.shape[1] - 2), dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            weight = float(kernel[dy, dx])
            if weight != 0.0:
                out += weight * image[dy:dy + out.shape[0], dx:dx + out.shape[1]]
    return out


def extract(image: np.ndarray) -> Dict[str, float]:
    """Eight scalars describing one frame.

    ``image`` is float32 in ``[0, 1]``, shape ``(h, w)`` or ``(1, h, w)``.
    """
    frame = np.asarray(image, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[0]
    if frame.ndim != 2:
        raise ValueError("expected a 2D frame, got shape {0}".format(np.shape(image)))

    mean = float(frame.mean())
    std = float(frame.std())

    # Sharpness and noise, separated on purpose.
    #
    # The obvious focus measure -- the variance of the Laplacian of the raw
    # frame -- does not work here, and finding that out is worth writing down.
    # The lamp-ageing domain softens the lens AND raises sensor read noise,
    # because the camera compensates for the dimmer scene with gain.  Read noise
    # is broadband, so it dominates the Laplacian: the raw measure RISES by a
    # factor of four under a lens that is visibly softer.  An alert saying
    # "sharpness increased" while the image is blurring is worse than no alert.
    #
    # So: smooth first, then measure structure, and estimate the noise
    # separately with Immerkaer's estimator, whose kernel is a second difference
    # in both directions and therefore annihilates locally linear image content
    # while passing noise.  The two then move the way their names say --
    # sharpness down, noise up -- and the alert text is an instruction rather
    # than a puzzle.  docs/drift.md keeps the numbers.
    noise_level = float(
        np.abs(_convolve3(frame, _IMMERKAER)).mean() * _ABS_TO_SIGMA / _IMMERKAER_GAIN
    )
    smoothed = _convolve3(frame, _BOX)
    sharpness = max(
        float(_convolve3(smoothed, _LAPLACIAN).var())
        - _SHARPNESS_NOISE_GAIN * noise_level * noise_level,
        0.0,
    )

    gy = np.abs(np.diff(frame, axis=0))
    gx = np.abs(np.diff(frame, axis=1))
    gradient = float(gy.mean() + gx.mean()) * 0.5
    # A fixed absolute threshold rather than a percentile: a percentile-based
    # edge density is scale-invariant and would report the same value for a
    # sharp frame and a blurred one, which is the shift being watched for.
    edge_density = float((gx > 0.02).mean() + (gy > 0.02).mean()) * 0.5

    # Histogram entropy over 32 bins, via bincount on the 8-bit quantisation.
    # np.histogram is the obvious call and costs a millisecond a frame, which at
    # 30 fps is a third of the inference budget spent on telemetry.  The frames
    # are 8-bit to begin with, so this is exact rather than an approximation.
    quantised = np.clip(frame * 255.0, 0, 255).astype(np.uint8) >> 3
    histogram = np.bincount(quantised.reshape(-1), minlength=32).astype(np.float64)
    total = histogram.sum()
    if total > 0:
        histogram /= total
    nonzero = histogram[histogram > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum())

    highlight_fraction = float((frame > 0.85).mean())

    # Texture scale: the ratio of fine-grain energy to total variation.  Coarser
    # texture pushes energy out of the one-pixel difference and this falls,
    # which is what separates a material change from an illumination change.
    texture_scale = float(gradient / (std + 1e-6))

    return {
        "brightness": round(mean, 6),
        "contrast": round(std, 6),
        "sharpness": round(sharpness, 8),
        "noise_level": round(noise_level, 8),
        "edge_density": round(edge_density, 6),
        "entropy": round(entropy, 6),
        "highlight_fraction": round(highlight_fraction, 6),
        "texture_scale": round(texture_scale, 6),
    }


def extract_batch(images: np.ndarray) -> Dict[str, np.ndarray]:
    """Feature arrays for a stack of frames, keyed by feature name."""
    records = [extract(image) for image in images]
    return {name: np.asarray([r[name] for r in records], dtype=np.float64)
            for name in FEATURE_NAMES}
