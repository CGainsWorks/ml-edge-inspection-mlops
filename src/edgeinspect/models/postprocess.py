"""Turning a defect heatmap into boxes.

Pure NumPy and SciPy on purpose: this runs on the edge device inside the
inference loop, after the ONNX or TensorRT graph, and the runtime image has no
torch in it.  ``tests/test_runtime_boundary.py`` enforces that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Box:
    """A detected defect region in input-image pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    area_px: int

    def to_dict(self) -> dict:
        return {
            "box": [self.x0, self.y0, self.x1, self.y1],
            "score": round(float(self.score), 6),
            "area_px": self.area_px,
        }

    def as_array(self) -> np.ndarray:
        return np.asarray([self.x0, self.y0, self.x1, self.y1], dtype=np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic.

    ``1 / (1 + exp(-x))`` overflows for the large negative logits a confident
    good-part prediction produces, and NumPy turns that into a RuntimeWarning
    that CI is configured to treat as an error.
    """
    x = np.asarray(x, dtype=np.float32)
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def image_score(heatmap: np.ndarray, top_k: int = 8) -> float:
    """One number per frame: the mean of the ``top_k`` hottest cells.

    A plain maximum is what an inspection system wants in principle and is too
    noisy in practice -- a single hot cell fires on sensor noise.  A plain mean
    is the opposite failure: a five-pixel pit is diluted by a thousand cells of
    clean surface and never crosses a threshold.  The mean of the top few cells
    sits between the two and is what the reported AUROC is computed from.
    """
    flat = np.asarray(heatmap, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return 0.0
    k = int(min(max(top_k, 1), flat.size))
    top = np.partition(flat, flat.size - k)[flat.size - k:]
    return float(top.mean())


def heatmap_to_boxes(heatmap: np.ndarray, threshold: float = 0.5,
                     input_size: int = 256, min_area_cells: int = 1,
                     max_boxes: int = 16) -> List[Box]:
    """Threshold, label connected components, and scale each box to input pixels.

    The heatmap is at the network's stride, typically 32x32 for a 256x256
    input, so a box is only ever accurate to one cell.  That is the honest
    resolution of this head and ``docs/MODEL_CARD.md`` reports localisation at
    an IoU threshold that reflects it.
    """
    grid = np.asarray(heatmap, dtype=np.float32)
    if grid.ndim == 3 and grid.shape[0] == 1:
        grid = grid[0]
    if grid.ndim != 2:
        raise ValueError("heatmap must be 2D or (1, H, W), got shape {0}".format(heatmap.shape))

    binary = grid >= threshold
    if not binary.any():
        return []
    labelled, count = ndimage.label(binary)
    scale_y = input_size / float(grid.shape[0])
    scale_x = input_size / float(grid.shape[1])

    boxes: List[Box] = []
    for index in range(1, count + 1):
        component = labelled == index
        cells = int(component.sum())
        if cells < min_area_cells:
            continue
        ys, xs = np.nonzero(component)
        x0 = int(np.floor(xs.min() * scale_x))
        y0 = int(np.floor(ys.min() * scale_y))
        x1 = int(np.ceil((xs.max() + 1) * scale_x))
        y1 = int(np.ceil((ys.max() + 1) * scale_y))
        boxes.append(Box(
            x0=x0, y0=y0,
            x1=min(x1, input_size), y1=min(y1, input_size),
            score=float(grid[component].max()),
            area_px=int(cells * scale_x * scale_y),
        ))
    boxes.sort(key=lambda b: b.score, reverse=True)
    return boxes[:max_boxes]


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two ``x0, y0, x1, y1`` boxes."""
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(ix1 - ix0, 0.0), max(iy1 - iy0, 0.0)
    intersection = iw * ih
    if intersection <= 0:
        return 0.0
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection
    return float(intersection / union) if union > 0 else 0.0
