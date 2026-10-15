"""Quantisation-aware training.

PTQ reads activation ranges off a calibration set and hopes the weights tolerate
being rounded.  QAT puts the rounding inside the training loop, so the optimiser
sees the quantisation error and moves the weights somewhere it hurts less.

How this is wired, and why it is not ``torch.ao.quantization``
--------------------------------------------------------------
The eager-mode ``torch.ao`` QAT flow produces a quantised torch module, and
getting that into a QDQ ONNX graph that TensorRT will ingest means going through
a converter that changes between torch releases.  This project needs one
artefact format -- QDQ ONNX -- consumed by both ONNX Runtime and TensorRT, and
it needs the QAT and PTQ paths to differ in exactly one thing so the comparison
between them means something.

So QAT here is fake quantisation with a straight-through estimator, applied to
the inputs and weights of every convolution, trained for a short fine-tune from
the FP32 checkpoint.  The output is an ordinary FP32 checkpoint whose weights
have been shaped by quantisation noise.  It is then exported and quantised
through the *same* :mod:`edgeinspect.quantise.ptq` path as the baseline, so the
INT8 accuracy difference between the two is attributable to the weights and to
nothing else.

The observers match the export settings deliberately: symmetric signed INT8,
per-channel for weights, per-tensor for activations.  A QAT run that simulates a
different quantisation grid from the one the exporter uses is a QAT run that
optimises for a model nobody deploys.
"""

from __future__ import annotations

from typing import Dict, Iterator, Tuple

import torch
from torch import nn

QMIN = -128
QMAX = 127


def _ste_quantise(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Fake-quantise with a straight-through gradient.

    ``x + (q - x).detach()`` is the whole trick: the forward value is the
    quantised one, the backward gradient is the identity.
    """
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(x / scale), QMIN, QMAX) * scale
    return x + (q - x).detach()


class ActivationFakeQuant(nn.Module):
    """Per-tensor symmetric INT8 with an exponential moving average of absmax.

    EMA rather than running max: a single outlier batch otherwise pins the scale
    for the rest of training and every subsequent activation quantises into a
    handful of levels.
    """

    def __init__(self, momentum: float = 0.95, warmup: int = 20):
        super().__init__()
        self.momentum = float(momentum)
        self.warmup = int(warmup)
        self.register_buffer("absmax", torch.zeros(1))
        self.register_buffer("steps", torch.zeros(1, dtype=torch.long))
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_absmax = x.detach().abs().amax()
            if int(self.steps.item()) == 0:
                self.absmax.fill_(float(batch_absmax))
            else:
                self.absmax.mul_(self.momentum).add_(batch_absmax * (1.0 - self.momentum))
            self.steps += 1
        if not self.enabled or int(self.steps.item()) < self.warmup:
            # Observe first, quantise later.  Fake-quantising against an
            # unconverged scale in the first few steps destabilises the
            # fine-tune for no benefit.
            return x
        return _ste_quantise(x, self.absmax / QMAX)


class WeightFakeQuant(nn.Module):
    """Per-output-channel symmetric INT8, recomputed every forward pass.

    Weights change every step, so there is nothing to observe: the scale is a
    function of the current weight tensor.
    """

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        absmax = weight.detach().abs().amax(dim=(1, 2, 3), keepdim=True)
        return _ste_quantise(weight, absmax / QMAX)


class QuantConv2d(nn.Module):
    """A ``Conv2d`` with fake quantisation on its input and its weights."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        self.conv = conv
        self.input_quant = ActivationFakeQuant()
        self.weight_quant = WeightFakeQuant()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_quant(x)
        weight = self.weight_quant(self.conv.weight)
        return nn.functional.conv2d(
            x, weight, self.conv.bias, self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


def _named_convs(module: nn.Module) -> Iterator[Tuple[nn.Module, str, nn.Conv2d]]:
    for child_name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            yield module, child_name, child
        else:
            yield from _named_convs(child)


def prepare(model: nn.Module) -> Tuple[nn.Module, int]:
    """Replace every ``Conv2d`` in place with a fake-quantised one.

    Returns ``(model, converted_count)``.  Converting zero convolutions means
    the model shape changed and QAT is silently a plain fine-tune, so callers
    check the count.
    """
    targets = list(_named_convs(model))
    for parent, name, conv in targets:
        setattr(parent, name, QuantConv2d(conv))
    return model, len(targets)


def unwrap(model: nn.Module) -> nn.Module:
    """Undo :func:`prepare`, giving back a plain FP32 model with the tuned weights.

    The exported artefact is an ordinary graph.  The quantisation nodes come
    from the ONNX quantiser, not from here, which is what keeps the QAT and PTQ
    artefacts comparable.
    """
    for parent, name, child in list(_named_quant_convs(model)):
        setattr(parent, name, child.conv)
    return model


def _named_quant_convs(module: nn.Module):
    for child_name, child in module.named_children():
        if isinstance(child, QuantConv2d):
            yield module, child_name, child
        else:
            yield from _named_quant_convs(child)


def observed_ranges(model: nn.Module) -> Dict[str, float]:
    """The activation absmax each observer settled on, for the run summary."""
    out: Dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, ActivationFakeQuant):
            out[name] = round(float(module.absmax.item()), 6)
    return out


def finetune(model: nn.Module, data_root, config: dict, device,
             epochs: int = 6, lr: float = 3e-4, tracker=None):
    """Fine-tune ``model`` with fake quantisation active.  Returns a summary dict.

    Deliberately short and at a low learning rate.  QAT here is a repair pass on
    weights that are already good, not a second training run: a long QAT run at
    the original learning rate discards the FP32 model's solution and the
    comparison against PTQ stops being about quantisation.
    """
    import numpy as np
    import torch

    from ..data import AugmentConfig, load_arrays
    from ..data import augment as augment_module
    from ..metrics import classification
    from ..models import postprocess
    from ..models.net import image_logit, target_maps
    from ..models.preprocess import DEFAULT as DEFAULT_INPUT
    from ..models.preprocess import InputSpec

    loss_config = dict(config.get("loss", {}))
    model_config = dict(config.get("model", {}))
    input_spec = InputSpec(
        mean=float(model_config.get("input_mean", DEFAULT_INPUT.mean)),
        scale=float(model_config.get("input_scale", DEFAULT_INPUT.scale)),
    )
    top_k = int(loss_config.get("top_k", 8))
    batch_size = int(config.get("data", {}).get("batch_size", 32))
    augment_config = AugmentConfig.from_dict(config.get("augment", {}))

    train_images, train_masks, _train_labels, _ = load_arrays(data_root, "train")
    val_images, _val_masks, val_labels, _ = load_arrays(data_root, "val")

    model, converted = prepare(model)
    if converted == 0:
        raise RuntimeError(
            "QAT converted zero convolutions.  The model shape changed and this run "
            "would be a plain fine-tune reported as QAT."
        )
    model = model.to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    pos_weight = torch.tensor([float(loss_config.get("pos_weight", 24.0))], device=device)
    cell_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    image_criterion = nn.BCEWithLogitsLoss()
    cell_weight = float(loss_config.get("cell_weight", 1.0))
    image_weight = float(loss_config.get("image_weight", 0.5))

    rng = np.random.default_rng(int(config.get("seed", 7)) + 1)
    history = []
    best_auroc = -1.0
    best_state = None
    for epoch in range(int(epochs)):
        model.train()
        order = rng.permutation(len(train_images))
        total, batches = 0.0, 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            if len(indices) < 2:
                continue
            batch_images = np.empty((len(indices), *train_images.shape[1:]), dtype=np.float32)
            batch_masks = np.empty((len(indices), *train_masks.shape[1:]), dtype=np.float32)
            for position, index in enumerate(indices):
                image, mask = augment_module.apply(train_images[index], train_masks[index],
                                                   augment_config, rng)
                batch_images[position] = image
                batch_masks[position] = mask
            x = torch.from_numpy(input_spec.apply(batch_images)).to(device)
            m = torch.from_numpy(batch_masks).to(device)
            cell_target, image_target = target_maps(m, stride=int(getattr(model, "stride", 8)))
            logits = model(x)
            loss = (cell_weight * cell_criterion(logits, cell_target)
                    + image_weight * image_criterion(image_logit(logits, top_k=top_k),
                                                     image_target))
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            total += float(loss.detach())
            batches += 1

        model.eval()
        with torch.no_grad():
            scores = []
            for start in range(0, len(val_images), 64):
                chunk = torch.from_numpy(
                    input_spec.apply(val_images[start:start + 64])).to(device)
                for logit in model(chunk).float().cpu().numpy():
                    scores.append(postprocess.image_score(logit, top_k=top_k))
            auroc = float(classification.auroc(np.asarray(scores), val_labels))
        history.append({"epoch": epoch, "loss": total / max(batches, 1), "val_auroc": auroc})
        if tracker is not None:
            tracker.log_metrics({"qat_loss": history[-1]["loss"], "qat_val_auroc": auroc},
                                step=epoch)
        print("qat epoch {0:3d}  loss {1:.4f}  val_auroc {2:.4f}".format(
            epoch, history[-1]["loss"], auroc), flush=True)
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    ranges = observed_ranges(model)
    unwrap(model)
    return {"history": history, "best_val_auroc": best_auroc,
            "converted_convolutions": converted, "activation_ranges": ranges}
