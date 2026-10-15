#!/usr/bin/env python
"""Export a checkpoint to ONNX and verify it against the torch model.

    python scripts/export_onnx.py --weights models/surface_defect.pt

Writes ``models/<name>_fp32.onnx`` and its sidecar JSON.  The verification step
is not optional and not a formality: batch-norm folding and constant folding
change numerics quietly, and the first place a changed graph shows up is a
mis-calibrated INT8 model that nobody thinks to blame on the export.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

from edgeinspect import paths
from edgeinspect.config import load_yaml
from edgeinspect.export import onnx_export
from edgeinspect.train.loop import load_checkpoint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default="models/surface_defect.pt")
    parser.add_argument("--config", default="configs/export.yaml")
    parser.add_argument("--out", default=None,
                        help="output path; defaults to <weights stem>_fp32.onnx")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--opset", type=int, default=None)
    parser.add_argument("--dynamic-size", action="store_true")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config).get("onnx", {})
    checkpoint = paths.resolve(args.weights)
    model, meta = load_checkpoint(checkpoint)

    input_size = int(args.input_size or config.get("input_size", 256))
    opset = int(args.opset or config.get("opset", onnx_export.DEFAULT_OPSET))
    dynamic_size = bool(args.dynamic_size or config.get("dynamic_size", False))

    out_path = paths.resolve(args.out) if args.out else (
        checkpoint.parent / (checkpoint.stem + "_fp32.onnx"))

    export_meta = {
        "name": meta.get("name", checkpoint.stem),
        "precision": "fp32",
        "in_channels": model.stem.conv.in_channels,
        "stride": meta.get("stride", 8),
        "input": meta.get("input", {}),
        "threshold": meta.get("threshold", 0.0),
        "top_k": meta.get("top_k", 8),
        "target_fpr": meta.get("target_fpr", 0.02),
        "cell_threshold": 0.5,
        "training": {
            "best_epoch": meta.get("best_epoch"),
            "best_val_auroc": meta.get("best_val_auroc"),
            "seed": meta.get("seed"),
            "dataset": meta.get("dataset", {}),
            "mlflow_run_id": meta.get("mlflow_run_id"),
        },
    }
    onnx_export.export(model, out_path, export_meta, input_size=input_size,
                       opset=opset, dynamic_size=dynamic_size,
                       checkpoint_path=checkpoint)

    summary = {"onnx": paths.relative(out_path),
               "sidecar": paths.relative(onnx_export.sidecar_path(out_path)),
               "bytes": out_path.stat().st_size,
               "opset": opset,
               "dynamic_size": dynamic_size}

    verify = (not args.no_verify) and bool(config.get("verify", True))
    if verify:
        tolerance = float(config.get("verify_tolerance", 1e-4))
        summary["verification"] = onnx_export.verify(
            out_path, model, input_size=input_size, tolerance=tolerance)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
