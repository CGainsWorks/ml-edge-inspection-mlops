#!/usr/bin/env python
"""Build TensorRT engines from the exported ONNX graphs.

    python scripts/export_tensorrt.py --dry-run          # works anywhere
    python scripts/export_tensorrt.py --precision fp16   # Jetson only

**This has never been executed.**  No Jetson was available for this project and
TensorRT does not install on the development machine, so no number anywhere in
this repository came from a TensorRT engine.  ``docs/jetson.md`` says what would
have to be run to change that.

``--dry-run`` is the part that works without TensorRT: it prints the exact
builds that would run, checks that the ONNX graphs and the shared calibration
set are present, and exits.  That makes the gap inspectable rather than
theoretical, and ``tests/test_tensorrt_plan.py`` asserts the plan is coherent
without needing a device.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import sys

from edgeinspect import paths
from edgeinspect.config import load_yaml
from edgeinspect.export import tensorrt_export
from edgeinspect.quantise import calibration


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", default="models/surface_defect_fp32.onnx")
    parser.add_argument("--config", default="configs/export.yaml")
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--precision", nargs="+", default=None,
                        choices=list(tensorrt_export.PRECISIONS))
    parser.add_argument("--calibration", default="models/calibration.npy")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config).get("tensorrt", {})
    precisions = list(args.precision or config.get("precisions",
                                                   tensorrt_export.PRECISIONS))
    onnx_path = paths.resolve(args.onnx)
    out_dir = paths.resolve(args.out_dir)
    plan = list(tensorrt_export.plan_builds(onnx_path, out_dir, precisions))

    available = tensorrt_export.available()
    print("tensorrt importable: {0}".format(available))
    print("onnx: {0} ({1})".format(paths.relative(onnx_path),
                                   "present" if onnx_path.exists() else "MISSING"))
    calibration_path = paths.resolve(args.calibration)
    needs_calibration = any(item["needs_calibration"] for item in plan)
    if needs_calibration:
        print("calibration: {0} ({1})".format(
            paths.relative(calibration_path),
            "present" if calibration_path.exists() else "MISSING; run scripts/quantise.py"))
    print()
    for item in plan:
        print("  {0:<6} -> {1}".format(item["precision"], item["engine"]))

    if args.dry_run:
        print("\ndry run; nothing built")
        print(json.dumps({"plan": plan, "tensorrt_available": available}, indent=2))
        return 0

    if not available:
        print("\nTensorRT is not importable on this machine.  It ships with JetPack and "
              "is not installable from PyPI on a desktop.\n"
              "Run this on the Jetson, or use --dry-run.  See docs/jetson.md.",
              file=sys.stderr)
        return 2
    if not onnx_path.exists():
        print("no ONNX graph at " + paths.relative(onnx_path), file=sys.stderr)
        return 2

    frames = None
    if needs_calibration:
        if not calibration_path.exists():
            print("INT8 needs the calibration set; run scripts/quantise.py first.",
                  file=sys.stderr)
            return 2
        frames, _manifest = calibration.load(calibration_path.parent,
                                             name=calibration_path.stem)

    batch = config.get("batch", {})
    built = []
    for item in plan:
        engine = tensorrt_export.build_engine(
            onnx_path, paths.resolve(item["engine"]), precision=item["precision"],
            workspace_gb=float(config.get("workspace_gb", 2.0)),
            calibration=frames if item["needs_calibration"] else None,
            min_batch=int(batch.get("min", 1)), opt_batch=int(batch.get("opt", 1)),
            max_batch=int(batch.get("max", 8)), verbose=args.verbose)
        built.append(paths.relative(engine))
        print("built " + paths.relative(engine))
    print(json.dumps({"engines": built}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
