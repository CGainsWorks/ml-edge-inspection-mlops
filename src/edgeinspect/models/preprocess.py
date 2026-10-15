"""The input contract, in one place.

The model does not take an image in ``[0, 1]``.  It takes ``(image - 0.5) * 2``,
in ``[-1, 1]``, and that is not a stylistic choice -- it is the difference
between INT8 costing nothing and INT8 costing a sixth of the model's AUROC.

The measurement
---------------
The first version of this project fed the graph a ``[0, 1]`` image.  Static INT8
quantisation with **symmetric** activations then maps the input tensor onto
``[-absmax, absmax]``, and for an image that is never negative that spends half
the 256 available levels on values that cannot occur.  The frames are 8-bit to
begin with, so the stem convolution was seeing a 7-bit image -- and a defect
whose contrast against the surface is about 0.10 has roughly twelve levels to
live in rather than twenty-five.

``scripts/ablate_quantisation.py`` trains three centred and three uncentred
models and quantises each four ways.  Deltas are paired against each model's own
FP32 AUROC on the 600-frame test split:

=============================================  ==================  ==============
INT8 configuration                             ``[0, 1]`` input    ``[-1, 1]``
=============================================  ==================  ==============
symmetric activations (the deployed one)       **-0.156**          -0.002
asymmetric activations                         +0.009              +0.002
symmetric, stem convolution excluded           +0.005              -0.002
symmetric, per-tensor weight scales            +0.005              -0.002
=============================================  ==================  ==============

The bottom three rows are what identify the cause rather than merely working
around it.  Asymmetric activations recover the loss because a zero point lets
the quantiser use the whole range on a one-sided distribution.  Excluding the
stem recovers it because the stem is the only layer whose input is the image.
Per-tensor weight scales are a control and change nothing, so the loss is not in
the weights.

**The failure is unstable, which is why it survived a first look.**  The three
uncentred seeds lost 0.148, 0.043 and 0.277 AUROC.  An early single-seed
measurement of this happened to land near the mild end, and the effect was
nearly written off as noise.  It is not noise -- every seed is negative and the
centred model's three seeds are -0.004, +0.001 and -0.003 -- but a one-run
comparison could not have told the difference.  That is the argument for
``--seeds`` being three by default rather than one.

Asymmetric activations and stem exclusion are both real fixes, and both are
worse than this one:

* asymmetric activations put a zero point on every activation tensor, and
  TensorRT's INT8 path has per-tensor **scales only**.  A graph like that either
  gets rejected or silently falls back to FP16 on the affected layers, which is
  a latency regression discovered on the device;
* excluding the stem leaves about a tenth of the model's multiply-accumulates in
  floating point, on the target where INT8 throughput is the reason for
  quantising at all.

Centring the input costs one subtraction and one multiply per pixel, keeps every
layer in INT8, and keeps every activation symmetric.  ``tests/test_preprocess.py``
asserts that the value here, the value in each ONNX sidecar and the value the
training loop applies are the same number, because the failure mode when they
drift apart is a model that is merely slightly worse rather than obviously
broken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np

# The contract.  image in [0, 1] -> (image - MEAN) * SCALE, in [-1, 1].
MEAN = 0.5
SCALE = 2.0


@dataclass(frozen=True)
class InputSpec:
    """How raw frames become graph input."""

    mean: float = MEAN
    scale: float = SCALE
    channels: int = 1
    size: int = 256
    dtype: str = "float32"
    layout: str = "NCHW"

    def apply(self, images):
        """``[0, 1]`` frames to graph input.  Accepts NumPy arrays or torch tensors."""
        return (images - self.mean) * self.scale

    def invert(self, values):
        """Graph input back to ``[0, 1]``, for anything that needs to look at it."""
        return values / self.scale + self.mean

    def to_dict(self) -> dict:
        return {
            "name": "input",
            "layout": self.layout,
            "dtype": self.dtype,
            "size": self.size,
            "channels": self.channels,
            "mean": self.mean,
            "scale": self.scale,
            "note": (
                "uint8 greyscale / 255, then (x - {0}) * {1}, giving [-1, 1].  "
                "Centring the input is what keeps symmetric INT8 lossless here; "
                "see edgeinspect.models.preprocess."
            ).format(MEAN, SCALE),
        }

    @classmethod
    def from_sidecar(cls, sidecar: Optional[Mapping]) -> "InputSpec":
        """Read the contract out of a model sidecar, defaulting to the current one.

        A sidecar written before the contract existed has no ``mean``; those
        graphs were trained on ``[0, 1]`` and must keep being fed ``[0, 1]``, so
        the default when the key is absent is the identity, not the current
        contract.  Silently re-centring an old graph's input would be a quiet
        accuracy regression with no error anywhere.
        """
        block = dict((sidecar or {}).get("input", {}))
        if "mean" not in block:
            return cls(mean=0.0, scale=1.0,
                       channels=int(block.get("channels", 1)),
                       size=int(block.get("size", 256)))
        return cls(
            mean=float(block["mean"]),
            scale=float(block.get("scale", 1.0)),
            channels=int(block.get("channels", 1)),
            size=int(block.get("size", 256)),
            dtype=str(block.get("dtype", "float32")),
            layout=str(block.get("layout", "NCHW")),
        )


DEFAULT = InputSpec()


def to_model_input(images, spec: Optional[InputSpec] = None) -> np.ndarray:
    """``[0, 1]`` frames to graph input, as a contiguous float32 array."""
    spec = spec or DEFAULT
    return np.ascontiguousarray(spec.apply(np.asarray(images, dtype=np.float32)),
                                dtype=np.float32)
