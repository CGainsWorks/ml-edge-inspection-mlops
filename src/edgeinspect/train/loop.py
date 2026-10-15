"""The training loop.

Supervised on both heads at once: per-cell segmentation against the downsampled
mask, and image-level classification against the frame label.  One head, two
losses, so the score a threshold is applied to at deployment is the score that
was trained.

Model selection is on validation AUROC rather than validation loss.  Loss on
this task is dominated by the 300:1 clean-to-defective cell ratio and moves
long after the ranking has stopped improving.

Selection is on an *exponentially smoothed* validation AUROC, and that detail
came from being wrong first.  With 400 validation frames the standard error on
AUROC is around 0.02, so raw per-epoch selection reliably picks a lucky epoch.
On the reference run it picked epoch 1 -- training loss 0.47 against 0.38 at the
end -- and shipped a visibly undertrained checkpoint whose validation number
looked like the best of the run.  Smoothing costs nothing and picks a converged
epoch; the raw value is still logged and reported, so the smoothing is a
selection rule rather than a nicer-looking metric.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn

from .. import paths
from ..data import AugmentConfig, load_arrays, load_meta
from ..data import augment as augment_module
from ..metrics import classification, localisation
from ..models import postprocess
from ..models.preprocess import DEFAULT as DEFAULT_INPUT
from ..models.preprocess import InputSpec
from ..models.net import DefectNet, count_parameters, image_logit, target_maps


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    train_cell_loss: float
    train_image_loss: float
    val_auroc: float
    val_auroc_smoothed: float
    val_loss: float
    lr: float
    seconds: float


@dataclass
class TrainOutcome:
    checkpoint: Path
    best_epoch: int
    best_val_auroc: float
    history: List[EpochRecord] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)
    run_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "checkpoint": paths.relative(self.checkpoint),
            "best_epoch": self.best_epoch,
            "best_val_auroc": self.best_val_auroc,
            "history": [vars(record) for record in self.history],
            "metrics": self.metrics,
            "mlflow_run_id": self.run_id,
        }


def pick_device(requested: str = "auto") -> torch.device:
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda, got " + repr(requested))
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed % (2 ** 32))


def _lr_at(epoch: int, total: int, base_lr: float, warmup: int, scheduler: str) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * (epoch + 1) / float(warmup)
    if scheduler == "cosine":
        progress = (epoch - warmup) / max(total - warmup, 1)
        return base_lr * 0.5 * (1.0 + float(np.cos(np.pi * min(progress, 1.0))))
    return base_lr


def _augmented_batch(images: np.ndarray, masks: np.ndarray, indices: np.ndarray,
                     config: AugmentConfig, rng: np.random.Generator):
    batch_images = np.empty((len(indices), *images.shape[1:]), dtype=np.float32)
    batch_masks = np.empty((len(indices), *masks.shape[1:]), dtype=np.float32)
    for position, index in enumerate(indices):
        image, mask = augment_module.apply(images[index], masks[index], config, rng)
        batch_images[position] = image
        batch_masks[position] = mask
    return batch_images, batch_masks


@torch.no_grad()
def predict_logits(model: nn.Module, images: np.ndarray, device: torch.device,
                   batch_size: int = 64, spec=None) -> np.ndarray:
    """Run the model over frames in [0, 1] and return raw logit maps."""
    spec = spec or DEFAULT_INPUT
    model.eval()
    out = []
    for start in range(0, len(images), batch_size):
        chunk = torch.from_numpy(
            spec.apply(images[start:start + batch_size])).to(device)
        out.append(model(chunk).float().cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 1, 1, 1), dtype=np.float32)


def evaluate_arrays(model: nn.Module, images: np.ndarray, masks: np.ndarray,
                    labels: np.ndarray, records, device: torch.device,
                    top_k: int = 8, threshold: Optional[float] = None,
                    target_fpr: float = 0.02, min_iou: float = 0.1,
                    cell_threshold: float = 0.5, spec=None) -> dict:
    """Image-level and box-level metrics for a split already in memory."""
    logits = predict_logits(model, images, device, spec=spec)
    scores = np.asarray([postprocess.image_score(logit, top_k=top_k) for logit in logits])
    image_metrics = classification.summarise(scores, labels, threshold=threshold,
                                             target_fpr=target_fpr)
    input_size = images.shape[-1]
    predicted_boxes = []
    truth_boxes = []
    for logit, record in zip(logits, records):
        probability = postprocess.sigmoid(logit[0])
        boxes = postprocess.heatmap_to_boxes(probability, threshold=cell_threshold,
                                             input_size=input_size)
        predicted_boxes.append(np.stack([b.as_array() for b in boxes])
                               if boxes else np.zeros((0, 4), dtype=np.float32))
        truth_boxes.append(record.boxes())
    box_metrics = localisation.summarise(predicted_boxes, truth_boxes, min_iou=min_iou)
    return {"image": image_metrics, "boxes": box_metrics,
            "scores": scores.astype(float).tolist(),
            "labels": np.asarray(labels).astype(int).tolist()}


def train(config: dict, data_root: Path, out_dir: Path, name: str,
          device_name: str = "auto", tracker=None,
          extra_tags: Optional[Dict] = None,
          initial_state: Optional[Dict] = None) -> TrainOutcome:
    """Train one model and write a checkpoint.  Returns the outcome summary.

    ``initial_state`` warm-starts from an existing state dict, which is what the
    retraining pipeline passes.  Retraining from scratch on every drift event
    throws away everything the model knows about defect appearance -- which has
    not changed -- to relearn it from a training set that is now mostly replay.
    """
    seed = int(config.get("seed", 7))
    seed_everything(seed)
    device = pick_device(device_name)

    model_config = dict(config.get("model", {}))
    optim_config = dict(config.get("optim", {}))
    loss_config = dict(config.get("loss", {}))
    eval_config = dict(config.get("eval", {}))
    augment_config = AugmentConfig.from_dict(config.get("augment", {}))

    input_spec = InputSpec(
        mean=float(model_config.get("input_mean", DEFAULT_INPUT.mean)),
        scale=float(model_config.get("input_scale", DEFAULT_INPUT.scale)),
        channels=int(model_config.get("in_channels", 1)),
    )
    top_k = int(loss_config.get("top_k", 8))
    epochs = int(optim_config.get("epochs", 24))
    batch_size = int(config.get("data", {}).get("batch_size", 32))

    train_images, train_masks, train_labels, train_records = load_arrays(data_root, "train")
    val_images, val_masks, val_labels, val_records = load_arrays(data_root, "val")

    model = DefectNet(width=int(model_config.get("width", 16)),
                      in_channels=int(model_config.get("in_channels", 1))).to(device)
    if initial_state is not None:
        # strict=True on purpose: a warm start from a differently shaped
        # checkpoint should fail here rather than silently load a subset of the
        # weights and produce a model nobody can account for.
        model.load_state_dict({k: v.to(device) for k, v in initial_state.items()})
    parameters = count_parameters(model)

    optimiser = torch.optim.AdamW(model.parameters(),
                                  lr=float(optim_config.get("lr", 3e-3)),
                                  weight_decay=float(optim_config.get("weight_decay", 1e-4)))
    pos_weight = torch.tensor([float(loss_config.get("pos_weight", 24.0))], device=device)
    cell_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    image_criterion = nn.BCEWithLogitsLoss()
    cell_weight = float(loss_config.get("cell_weight", 1.0))
    image_weight = float(loss_config.get("image_weight", 0.5))
    grad_clip = float(optim_config.get("grad_clip", 1.0))

    if tracker is not None:
        tracker.log_params({
            "model": model_config, "optim": optim_config, "loss": loss_config,
            "augment": vars(augment_config), "batch_size": batch_size, "seed": seed,
            "parameters": parameters, "device": device.type,
            "train_frames": len(train_images), "val_frames": len(val_images),
        })
        if extra_tags:
            tracker.log_params(extra_tags)

    rng = np.random.default_rng(seed)
    history: List[EpochRecord] = []
    best_auroc = -1.0
    best_smoothed = -1.0
    best_epoch = -1
    best_state = None
    smoothed = None
    smoothing = float(optim_config.get("selection_smoothing", 0.4))
    scheduler_name = str(optim_config.get("scheduler", "cosine"))
    warmup = int(optim_config.get("warmup_epochs", 0))
    base_lr = float(optim_config.get("lr", 3e-3))

    for epoch in range(epochs):
        started = time.perf_counter()
        lr = _lr_at(epoch, epochs, base_lr, warmup, scheduler_name)
        for group in optimiser.param_groups:
            group["lr"] = lr

        model.train()
        order = rng.permutation(len(train_images))
        totals = {"loss": 0.0, "cell": 0.0, "image": 0.0}
        batches = 0
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            if len(indices) < 2:
                continue  # batch norm needs more than one sample
            batch_images, batch_masks = _augmented_batch(train_images, train_masks,
                                                         indices, augment_config, rng)
            x = torch.from_numpy(input_spec.apply(batch_images)).to(device)
            m = torch.from_numpy(batch_masks).to(device)
            cell_target, image_target = target_maps(m, stride=model.stride)

            logits = model(x)
            cell_loss = cell_criterion(logits, cell_target)
            image_loss = image_criterion(image_logit(logits, top_k=top_k), image_target)
            loss = cell_weight * cell_loss + image_weight * image_loss

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()

            totals["loss"] += float(loss.detach())
            totals["cell"] += float(cell_loss.detach())
            totals["image"] += float(image_loss.detach())
            batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = predict_logits(model, val_images, device, spec=input_spec)
            val_scores = np.asarray(
                [postprocess.image_score(logit, top_k=top_k) for logit in val_logits])
            val_auroc = float(classification.auroc(val_scores, val_labels))
            val_tensor = torch.from_numpy(val_logits).to(device)
            val_mask_tensor = torch.from_numpy(val_masks).to(device)
            val_cell_target, val_image_target = target_maps(val_mask_tensor, stride=model.stride)
            val_loss = float(
                cell_weight * cell_criterion(val_tensor, val_cell_target)
                + image_weight * image_criterion(image_logit(val_tensor, top_k=top_k),
                                                 val_image_target)
            )

        record = EpochRecord(
            epoch=epoch,
            train_loss=totals["loss"] / max(batches, 1),
            train_cell_loss=totals["cell"] / max(batches, 1),
            train_image_loss=totals["image"] / max(batches, 1),
            val_auroc=val_auroc,
            val_auroc_smoothed=0.0,
            val_loss=val_loss,
            lr=lr,
            seconds=time.perf_counter() - started,
        )
        smoothed = val_auroc if smoothed is None else (
            (1.0 - smoothing) * smoothed + smoothing * val_auroc)
        record.val_auroc_smoothed = float(smoothed)
        history.append(record)
        if tracker is not None:
            tracker.log_metrics({
                "train_loss": record.train_loss,
                "train_cell_loss": record.train_cell_loss,
                "train_image_loss": record.train_image_loss,
                "val_loss": record.val_loss,
                "val_auroc": record.val_auroc,
                "val_auroc_smoothed": record.val_auroc_smoothed,
                "lr": record.lr,
                "epoch_seconds": record.seconds,
            }, step=epoch)
        print("epoch {0:3d}  loss {1:.4f}  val_loss {2:.4f}  val_auroc {3:.4f}  "
              "smoothed {4:.4f}  lr {5:.5f}  {6:.1f}s".format(
                  epoch, record.train_loss, record.val_loss, record.val_auroc,
                  record.val_auroc_smoothed, lr, record.seconds), flush=True)

        # Selection on the smoothed value; the raw value is what gets reported.
        if record.val_auroc_smoothed > best_smoothed:
            best_smoothed = record.val_auroc_smoothed
            best_auroc = val_auroc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint; epochs must be at least 1")
    model.load_state_dict(best_state)

    # Final metrics at the operating point chosen on validation, not on test.
    target_fpr = float(eval_config.get("target_fpr", 0.02))
    min_iou = float(eval_config.get("min_iou", 0.1))
    val_metrics = evaluate_arrays(model, val_images, val_masks, val_labels, val_records,
                                  device, top_k=top_k, target_fpr=target_fpr,
                                  min_iou=min_iou, spec=input_spec)
    threshold = float(val_metrics["image"]["threshold"])

    dataset_meta = load_meta(data_root)
    paths.ensure(Path(out_dir))
    checkpoint_path = Path(out_dir) / (name + ".pt")
    torch.save({
        "state_dict": model.state_dict(),
        "model": {"width": int(model_config.get("width", 16)),
                  "in_channels": int(model_config.get("in_channels", 1))},
        "meta": {
            "name": name,
            "parameters": parameters,
            "top_k": top_k,
            "stride": model.stride,
            "input": input_spec.to_dict(),
            "threshold": threshold,
            "target_fpr": target_fpr,
            "best_epoch": best_epoch,
            "best_val_auroc": best_auroc,
            "seed": seed,
            "device": device.type,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "dataset": {
                "root": paths.relative(Path(data_root)),
                "preset": dataset_meta.get("preset"),
                "seed": dataset_meta.get("seed"),
                "splits": {k: v.get("index_sha256")
                           for k, v in dataset_meta.get("splits", {}).items()},
                "domains": {k: v.get("domain", {}).get("name")
                            for k, v in dataset_meta.get("splits", {}).items()},
            },
            "mlflow_run_id": getattr(tracker, "run_id", None),
        },
    }, checkpoint_path)

    outcome = TrainOutcome(
        checkpoint=checkpoint_path,
        best_epoch=best_epoch,
        best_val_auroc=best_auroc,
        history=history,
        metrics={"val": {"image": val_metrics["image"], "boxes": val_metrics["boxes"]}},
        run_id=getattr(tracker, "run_id", None),
    )

    summary_path = Path(out_dir) / (name + "_train.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(outcome.to_dict(), handle, indent=2)
    if tracker is not None:
        final = {"best_val_auroc": best_auroc, "best_epoch": best_epoch}
        for key, value in val_metrics["image"].items():
            if isinstance(value, (int, float)):
                final["val_" + key] = value
        for key, value in val_metrics["boxes"].items():
            if isinstance(value, (int, float)):
                final["val_box_" + key] = value
        tracker.log_metrics(final)
        tracker.log_artifact(summary_path)
        tracker.log_artifact(checkpoint_path)
    return outcome


def load_checkpoint(path: Path, device: Optional[torch.device] = None):
    """Rebuild a :class:`DefectNet` from a checkpoint.  Returns ``(model, meta)``."""
    device = device or torch.device("cpu")
    payload = torch.load(paths.resolve(path), map_location=device, weights_only=False)
    model = DefectNet(**payload["model"]).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("meta", {})
