#!/usr/bin/env python
"""Quantisation-aware fine-tuning, then export and quantise the result.

    python scripts/qat.py --weights models/surface_defect.pt

Produces ``models/<name>_qat.pt``, ``models/<name>_qat_fp32.onnx`` and the INT8
graphs built from it.  The INT8 graphs go through exactly the same PTQ path as
the baseline, on exactly the same calibration frames, so the accuracy difference
between ``<name>_int8`` and ``<name>_qat_int8`` is attributable to the weights
and to nothing else.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import time

import torch

from edgeinspect import paths
from edgeinspect.config import load_yaml
from edgeinspect.export import onnx_export
from edgeinspect.quantise import calibration, ptq
from edgeinspect.quantise import qat as qat_module
from edgeinspect.train import track
from edgeinspect.train.loop import load_checkpoint, pick_device


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="models/surface_defect.pt")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--export-config", default="configs/export.yaml")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--calibration", default="models/calibration.npy",
                        help="reuse the PTQ calibration frames; built if absent")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--out", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    export_config = load_yaml(args.export_config)
    quantise_config = export_config.get("quantise", {})
    data_root = paths.resolve(args.data_root)
    checkpoint = paths.resolve(args.weights)
    device = pick_device(args.device)

    model, meta = load_checkpoint(checkpoint, device=device)
    out_dir = paths.resolve(args.out) if args.out else checkpoint.parent
    name = meta.get("name", checkpoint.stem) + "_qat"

    started = time.perf_counter()
    with track(experiment=config.get("mlflow", {}).get("experiment", "edge-inspection"),
               run_name=name + "-" + time.strftime("%Y%m%d-%H%M%S"),
               tracking_uri=config.get("mlflow", {}).get("tracking_uri"),
               enabled=not args.no_mlflow,
               tags={"stage": "qat", "parent_checkpoint": paths.relative(checkpoint)}
               ) as tracker:
        summary = qat_module.finetune(model, data_root, config, device,
                                      epochs=args.epochs, lr=args.lr, tracker=tracker)

    qat_checkpoint = out_dir / (name + ".pt")
    torch.save({
        "state_dict": model.state_dict(),
        "model": {"width": model.stem.conv.out_channels,
                  "in_channels": model.stem.conv.in_channels},
        "meta": dict(meta, name=name, qat=summary, parent=paths.relative(checkpoint)),
    }, qat_checkpoint)

    onnx_path = out_dir / (name + "_fp32.onnx")
    export_meta = {
        "name": name,
        "precision": "fp32",
        "in_channels": model.stem.conv.in_channels,
        "stride": meta.get("stride", 8),
        "input": meta.get("input", {}),
        "threshold": meta.get("threshold", 0.0),
        "top_k": meta.get("top_k", 8),
        "target_fpr": meta.get("target_fpr", 0.02),
        "cell_threshold": 0.5,
        "training": {"qat": True, "parent": paths.relative(checkpoint),
                     "qat_best_val_auroc": summary["best_val_auroc"]},
    }
    onnx_export.export(model, onnx_path, export_meta,
                       input_size=int(export_config.get("onnx", {}).get("input_size", 256)),
                       opset=int(export_config.get("onnx", {}).get("opset", 17)),
                       checkpoint_path=qat_checkpoint)
    verification = onnx_export.verify(onnx_path, model, tolerance=float(
        export_config.get("onnx", {}).get("verify_tolerance", 1e-4)))

    calibration_path = paths.resolve(args.calibration)
    if calibration_path.exists():
        frames, manifest = calibration.load(calibration_path.parent,
                                            name=calibration_path.stem)
        declared = manifest.get("input_space", {})
        wanted = onnx_export.read_sidecar(onnx_path).get("input", {})
        if (declared.get("mean"), declared.get("scale")) != (
                wanted.get("mean"), wanted.get("scale")):
            raise SystemExit(
                "the saved calibration set was built for input space {0} and this "
                "graph expects {1}.  Rebuild it, or the INT8 activation ranges will "
                "be calibrated against data the model never sees.".format(
                    declared, wanted))
    else:
        frames, manifest = calibration.build(
            data_root, split=quantise_config.get("calibration_split", "train"),
            count=int(quantise_config.get("calibration_frames", 192)),
            defect_fraction=float(quantise_config.get("calibration_defect_fraction", 0.5)))
        calibration.save(frames, manifest, out_dir)

    built = ptq.build_variants(onnx_path, out_dir, frames, manifest,
                               methods=list(quantise_config.get("methods",
                                                                ["minmax", "percentile"])),
                               per_channel=bool(quantise_config.get("per_channel", True)))

    payload = {
        "checkpoint": paths.relative(qat_checkpoint),
        "onnx": paths.relative(onnx_path),
        "variants": {k: paths.relative(v) for k, v in built.items()},
        "qat": summary,
        "verification": verification,
        "calibration_frames": manifest["count"],
        "seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
