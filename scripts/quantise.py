#!/usr/bin/env python
"""Post-training quantisation: build the calibration set and the INT8 graphs.

    python scripts/quantise.py --model models/surface_defect_fp32.onnx

Builds one INT8 QDQ graph per calibration method, saves the calibration frames
to ``models/calibration.npy`` so the TensorRT INT8 build on the Jetson can use
identical frames, and prints the size comparison.

``--defect-fraction 0.0`` reproduces the failure this project documents: a
calibration set with no defects in it clips exactly the activations a defect
produces, and INT8 recall falls off a cliff while overall accuracy barely moves.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

from edgeinspect import paths
from edgeinspect.config import load_yaml
from edgeinspect.quantise import calibration, ptq


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/surface_defect_fp32.onnx")
    parser.add_argument("--config", default="configs/export.yaml")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--defect-fraction", type=float, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--name", default="calibration")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config).get("quantise", {})
    model_path = paths.resolve(args.model)
    out_dir = paths.resolve(args.out_dir) if args.out_dir else model_path.parent
    data_root = paths.resolve(args.data_root)

    frames_wanted = int(args.frames or config.get("calibration_frames", 192))
    split = args.split or config.get("calibration_split", "train")
    defect_fraction = (args.defect_fraction if args.defect_fraction is not None
                       else float(config.get("calibration_defect_fraction", 0.5)))
    methods = list(args.methods or config.get("methods", ["minmax", "entropy"]))
    per_channel = bool(config.get("per_channel", True))

    # The calibration set has to live in the graph's own input space, which the
    # sidecar declares.  Calibrating in a different space measures activation
    # ranges the model will never see in production.
    from edgeinspect.artefacts import read_sidecar
    from edgeinspect.models.preprocess import InputSpec

    spec = InputSpec.from_sidecar(read_sidecar(model_path))
    frames, manifest = calibration.build(
        data_root, split=split, count=frames_wanted,
        defect_fraction=defect_fraction, seed=args.seed, spec=spec)
    npy_path, json_path = calibration.save(frames, manifest, out_dir, name=args.name)
    print("calibration: {0} frames, {1:.0%} defective, saved to {2}".format(
        manifest["count"], manifest["defect_fraction_actual"], paths.relative(npy_path)))

    built = ptq.build_variants(model_path, out_dir, frames, manifest,
                               methods=methods, per_channel=per_channel)
    sizes = ptq.size_report({"fp32": model_path, **built})

    summary = {
        "source": paths.relative(model_path),
        "calibration": {
            "frames": paths.relative(npy_path),
            "manifest": paths.relative(json_path),
            "count": manifest["count"],
            "defect_fraction": manifest["defect_fraction_actual"],
            "brightness_coverage": {
                "calibration": manifest["brightness"],
                "split": manifest["split_brightness"],
            },
        },
        "variants": {name: paths.relative(path) for name, path in built.items()},
        "sizes": sizes,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
