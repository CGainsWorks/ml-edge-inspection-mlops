#!/usr/bin/env python
"""Accuracy for every precision variant, on every split.

    python scripts/evaluate.py --models models/surface_defect_fp32.onnx \\
        models/surface_defect_int8.onnx --splits test test_drift

Two things happen here that are easy to get wrong and that change the answer.

**Each variant gets its own threshold, chosen on validation.**  Quantisation
shifts the score distribution, so applying the FP32 threshold to the INT8 graph
measures the shift rather than the model.  Every variant is given the threshold
that hits the configured false positive rate on the *validation* split, and that
threshold is then written back into the variant's sidecar so a deployment uses
the number this evaluation measured.

**The drifted split is evaluated at the same threshold as the reference split.**
Re-tuning the threshold on the drifted data would measure how recoverable the
drift is with a knob nobody in production can turn, because production has no
labels.  The reported drop is the drop an operator would actually see.

``--torch`` additionally evaluates the checkpoint directly, in FP32 and in FP16
on CUDA.  FP16 is measured that way because IEEE half arithmetic is the same on
an RTX 4060 as on a Jetson; the *latency* is not, and none is claimed.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

import numpy as np

from edgeinspect import paths
from edgeinspect.config import load_yaml
from edgeinspect.data import load_arrays
from edgeinspect.edge.backends import load as load_backend
from edgeinspect.artefacts import read_sidecar, sidecar_path
from edgeinspect.metrics import classification, localisation
from edgeinspect.models import postprocess
from edgeinspect.models.preprocess import DEFAULT as INPUT_SPEC


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--torch", default=None, help="also evaluate this checkpoint directly")
    parser.add_argument("--torch-precisions", nargs="+", default=["fp32", "fp16"])
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--splits", nargs="+", default=["test", "test_drift"])
    parser.add_argument("--threshold-split", default="val")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--out", default="results/accuracy.json")
    parser.add_argument("--no-write-threshold", action="store_true",
                        help="do not update the sidecars with the chosen threshold")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def score_with_backend(backend, images, batch_size=32):
    top_k = backend.top_k
    scores, heatmaps = [], []
    for start in range(0, len(images), batch_size):
        chunk = np.ascontiguousarray(images[start:start + batch_size], dtype=np.float32)
        logits = backend.infer(chunk)
        for logit in logits:
            scores.append(postprocess.image_score(logit, top_k=top_k))
            heatmaps.append(postprocess.sigmoid(logit[0]))
    return np.asarray(scores, dtype=np.float64), heatmaps


def score_with_torch(model, images, device, top_k, half=False, batch_size=32):
    import torch

    model = model.to(device)
    model = model.half() if half else model.float()
    scores, heatmaps = [], []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            chunk = torch.from_numpy(np.ascontiguousarray(
                INPUT_SPEC.apply(images[start:start + batch_size]))).to(device)
            chunk = chunk.half() if half else chunk.float()
            logits = model(chunk).float().cpu().numpy()
            for logit in logits:
                scores.append(postprocess.image_score(logit, top_k=top_k))
                heatmaps.append(postprocess.sigmoid(logit[0]))
    return np.asarray(scores, dtype=np.float64), heatmaps


def box_metrics(heatmaps, records, cell_threshold, input_size, min_iou):
    predicted, truth = [], []
    for heatmap, record in zip(heatmaps, records):
        boxes = postprocess.heatmap_to_boxes(heatmap, threshold=cell_threshold,
                                             input_size=input_size)
        predicted.append(np.stack([b.as_array() for b in boxes]) if boxes
                         else np.zeros((0, 4), dtype=np.float32))
        truth.append(record.boxes())
    return localisation.summarise(predicted, truth, min_iou=min_iou)


def main(argv=None) -> int:
    args = parse_args(argv)
    eval_config = load_yaml(args.config).get("eval", {})
    target_fpr = float(eval_config.get("target_fpr", 0.02))
    min_iou = float(eval_config.get("min_iou", 0.1))
    data_root = paths.resolve(args.data_root)

    splits = {}
    for split in [args.threshold_split] + list(args.splits):
        if split in splits:
            continue
        images, _masks, labels, records = load_arrays(data_root, split)
        splits[split] = (images, labels, records)

    variants = []
    for model in args.models:
        path = paths.resolve(model)
        if not path.exists():
            if args.skip_missing:
                print("skipping missing " + paths.relative(path))
                continue
            raise SystemExit("no model at " + paths.relative(path))
        variants.append(("onnx", path, None))
    if args.torch:
        for precision in args.torch_precisions:
            variants.append(("torch", paths.resolve(args.torch), precision))
    if not variants:
        raise SystemExit("nothing to evaluate")

    results = {}
    for kind, path, precision in variants:
        if kind == "onnx":
            sidecar = read_sidecar(path)
            name = sidecar.get("name", path.stem)
            cell_threshold = float(sidecar.get("decision", {}).get("cell_threshold", 0.5))
            backend = load_backend(path, backend="onnx", providers=args.providers)
            try:
                scorer = lambda images: score_with_backend(  # noqa: E731
                    backend, images, batch_size=args.batch_size)
                entry = evaluate_variant(scorer, splits, args, target_fpr, min_iou,
                                         cell_threshold)
                entry["precision"] = sidecar.get("precision", "fp32")
                entry["model"] = paths.relative(path)
                entry["bytes"] = path.stat().st_size
                entry["quantisation"] = sidecar.get("quantisation", {})
            finally:
                backend.close()
            if not args.no_write_threshold:
                sidecar["decision"]["threshold"] = entry["threshold"]
                sidecar["decision"]["threshold_source"] = (
                    "scripts/evaluate.py on split '{0}' at target_fpr {1}".format(
                        args.threshold_split, target_fpr))
                with sidecar_path(path).open("w", encoding="utf-8") as handle:
                    json.dump(sidecar, handle, indent=2)
        else:
            import torch

            from edgeinspect.train.loop import load_checkpoint

            half = precision == "fp16"
            device = torch.device("cuda" if (half or torch.cuda.is_available()) else "cpu")
            if half and not torch.cuda.is_available():
                print("skipping torch fp16: no CUDA device")
                continue
            model, meta = load_checkpoint(path, device=device)
            top_k = int(meta.get("top_k", 8))
            name = "{0}_torch_{1}".format(meta.get("name", path.stem), precision)
            scorer = lambda images: score_with_torch(  # noqa: E731
                model, images, device, top_k, half=half, batch_size=args.batch_size)
            entry = evaluate_variant(scorer, splits, args, target_fpr, min_iou, 0.5)
            entry["precision"] = precision
            entry["model"] = paths.relative(path)
            entry["device"] = device.type
            entry["backend"] = "torch"
        results[name] = entry

    payload = {
        "data_root": paths.relative(data_root),
        "threshold_split": args.threshold_split,
        "target_fpr": target_fpr,
        "min_iou": min_iou,
        "providers": args.providers,
        "splits": {k: {"frames": len(v[1]), "defective": int(v[1].sum())}
                   for k, v in splits.items()},
        "variants": results,
    }
    out_path = paths.resolve(args.out)
    paths.ensure(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print_table(payload, args.splits)
    print("\nwrote " + paths.relative(out_path))
    return 0


def evaluate_variant(scorer, splits, args, target_fpr, min_iou, cell_threshold):
    """Choose a threshold on the threshold split, then apply it everywhere."""
    images, labels, _records = splits[args.threshold_split]
    scores, _ = scorer(images)
    threshold = float(classification.threshold_at_fpr(scores, labels, target_fpr))

    entry = {"threshold": threshold, "splits": {}}
    entry["splits"][args.threshold_split] = {
        "image": classification.summarise(scores, labels, threshold=threshold,
                                          target_fpr=target_fpr),
    }
    for split in args.splits:
        images, labels, records = splits[split]
        scores, heatmaps = scorer(images)
        entry["splits"][split] = {
            "image": classification.summarise(scores, labels, threshold=threshold,
                                              target_fpr=target_fpr),
            "boxes": box_metrics(heatmaps, records, cell_threshold,
                                 images.shape[-1], min_iou),
        }
    return entry


def print_table(payload, splits) -> None:
    header = "{0:<28} {1:<7}".format("variant", "prec")
    for split in splits:
        header += " {0:>8} {1:>7} {2:>7}".format(split[:8] + ":auroc", "recall", "fpr")
    print("\n" + header)
    print("-" * len(header))
    for name, entry in payload["variants"].items():
        row = "{0:<28} {1:<7}".format(name[:28], entry.get("precision", "?"))
        for split in splits:
            image = entry["splits"].get(split, {}).get("image", {})
            row += " {0:>8.4f} {1:>7.4f} {2:>7.4f}".format(
                image.get("auroc", float("nan")),
                image.get("recall", float("nan")),
                image.get("false_positive_rate", float("nan")))
        print(row)


if __name__ == "__main__":
    raise SystemExit(main())
