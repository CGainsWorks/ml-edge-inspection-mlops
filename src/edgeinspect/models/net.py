"""The detector.

A small fully-convolutional network that maps a 1x256x256 greyscale frame to a
1x32x32 defect logit map.  The image-level score is derived from the map in
:func:`edgeinspect.models.postprocess.image_score`, so classification and
localisation come out of one head rather than two, and there is no way for the
two to disagree.

Design notes, all of them driven by what has to happen downstream:

* **Fully convolutional, no global pooling and no fully connected layer.**  A
  linear layer would fix the input resolution into the graph.  A line camera
  that changes resolution is a Tuesday, and rebuilding a TensorRT engine
  because the frame grew by 32 pixels is not an acceptable failure mode.
* **61k parameters.**  Small enough to train on CPU in CI, which is what makes
  the end-to-end retraining loop testable on a runner with no GPU.  The point of
  this project is the loop, and a loop that can only be tested on a GPU is a
  loop that does not get tested.
* **ReLU, not SiLU or GELU.**  ReLU fuses into the preceding convolution in
  every TensorRT precision including INT8.  SiLU does not fuse in INT8 without a
  plugin, and the resulting reformat layers cost more than the activation
  function is worth at this size.
* **Batch norm, folded at export.**  Kept for training stability, folded into
  the convolution weights by ``torch.onnx.export`` so it costs nothing at
  inference and does not become a per-channel quantisation hazard.
* **Stride-2 convolutions, not max pooling.**  Max pooling in INT8 is fine, but
  a strided convolution keeps the whole downsampling path in one quantised
  operator type and gives ONNX Runtime's QDQ pass fewer seams to work with.
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

STRIDE = 8


class ConvBlock(nn.Module):
    """Convolution, batch norm, ReLU.  The whole vocabulary of this network."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class DefectNet(nn.Module):
    """Greyscale frame in, defect logit map out.

    ``width`` scales every channel count.  The published model is ``width=16``;
    ``width=8`` is what the tiny CI configuration trains, and the two share the
    architecture so a CI run genuinely exercises the export and quantisation
    path rather than a simplified stand-in.
    """

    def __init__(self, width: int = 16, in_channels: int = 1):
        super().__init__()
        w = int(width)
        self.stem = ConvBlock(in_channels, w, stride=2)        # 256 -> 128
        self.down1 = ConvBlock(w, w * 2, stride=2)             # 128 -> 64
        self.down2 = ConvBlock(w * 2, w * 4, stride=2)         # 64  -> 32
        self.refine1 = ConvBlock(w * 4, w * 4)
        self.refine2 = ConvBlock(w * 4, w * 4)
        self.head = nn.Conv2d(w * 4, 1, kernel_size=1)

        # A defect is rare in a frame and rarer still per cell.  Starting the
        # head at a strongly negative bias means the first epoch does not have
        # to spend itself unlearning a uniform 0.5 prediction, which on a run
        # this short is the difference between converging and not.
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.refine1(x)
        x = self.refine2(x)
        return self.head(x)

    @property
    def stride(self) -> int:
        return STRIDE


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build(width: int = 16, in_channels: int = 1) -> DefectNet:
    return DefectNet(width=width, in_channels=in_channels)


def image_logit(logits: torch.Tensor, top_k: int = 8) -> torch.Tensor:
    """Image-level logit: the mean of the ``top_k`` hottest cells.

    The differentiable twin of :func:`edgeinspect.models.postprocess.image_score`.
    They have to agree, because the training objective is defined here and the
    reported metric is computed there; ``tests/test_postprocess.py`` asserts
    they match to within float tolerance on random inputs.
    """
    flat = logits.flatten(1)
    k = int(min(max(top_k, 1), flat.shape[1]))
    top = torch.topk(flat, k=k, dim=1).values
    return top.mean(dim=1)


def target_maps(masks: torch.Tensor, stride: int = STRIDE) -> Tuple[torch.Tensor, torch.Tensor]:
    """Downsample pixel masks to the head's resolution.

    Average pooling then a small threshold, rather than nearest-neighbour
    sampling: a two-pixel-wide scratch survives average pooling and disappears
    entirely under nearest-neighbour at stride 8, taking the only training
    signal for the hardest defect class with it.
    """
    pooled = torch.nn.functional.avg_pool2d(masks, kernel_size=stride, stride=stride)
    cell_target = (pooled > (0.5 / (stride * stride))).float()
    image_target = (cell_target.flatten(1).amax(dim=1) > 0).float()
    return cell_target, image_target
