"""The detector and its post-processing.

``postprocess`` is NumPy only and is imported eagerly, because the edge runtime
needs it.  ``net`` needs torch and is imported lazily through :func:`build`, so
that ``import edgeinspect.models`` works inside the runtime image where torch is
deliberately absent.
"""

from .postprocess import Box, heatmap_to_boxes, image_score, iou, sigmoid
from .preprocess import DEFAULT as DEFAULT_INPUT
from .preprocess import InputSpec, to_model_input

__all__ = [
    "Box", "heatmap_to_boxes", "image_score", "iou", "sigmoid",
    "InputSpec", "DEFAULT_INPUT", "to_model_input", "build", "STRIDE",
]

STRIDE = 8


def build(width: int = 16, in_channels: int = 1):
    """Construct a :class:`edgeinspect.models.net.DefectNet`.  Needs torch."""
    from .net import build as _build

    return _build(width=width, in_channels=in_channels)
