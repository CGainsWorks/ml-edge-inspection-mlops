#!/usr/bin/env python
"""Train the detector, tracked in MLflow.

    python scripts/train.py --config configs/train.yaml
    python scripts/train.py --preset tiny --epochs 6 --name surface_defect_tiny

Writes ``models/<name>.pt`` and ``models/<name>_train.json``.  The checkpoint
carries the operating threshold chosen on the validation split, the dataset
hashes it was trained against and the MLflow run id, so a deployed model can be
traced back to its data without consulting anything else.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import subprocess
import time

from edgeinspect import paths
from edgeinspect.config import deep_merge, load_yaml
from edgeinspect.train import track
from edgeinspect.train.loop import train


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=str(paths.ROOT), capture_output=True, text=True,
                             timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 - provenance is best-effort, never fatal
        return "unknown"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default="models")
    parser.add_argument("--name", default="surface_defect")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--tag", action="append", default=[],
                        metavar="KEY=VALUE", help="extra MLflow tag; repeatable")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)

    overrides = {}
    if args.epochs is not None:
        overrides.setdefault("optim", {})["epochs"] = args.epochs
    if args.width is not None:
        overrides.setdefault("model", {})["width"] = args.width
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = deep_merge(config, overrides)

    data_root = paths.resolve(args.data_root or config.get("data", {}).get(
        "root", "data/processed"))
    out_dir = paths.resolve(args.out)
    mlflow_config = config.get("mlflow", {})

    tags = {"git_sha": git_sha(), "dataset": paths.relative(data_root)}
    for item in args.tag:
        if "=" not in item:
            raise SystemExit("--tag expects KEY=VALUE, got " + repr(item))
        key, value = item.split("=", 1)
        tags[key] = value

    run_name = args.run_name or "{0}-{1}".format(args.name, time.strftime("%Y%m%d-%H%M%S"))
    started = time.perf_counter()
    with track(experiment=mlflow_config.get("experiment", "edge-inspection"),
               run_name=run_name,
               tracking_uri=mlflow_config.get("tracking_uri"),
               enabled=not args.no_mlflow,
               tags=tags) as tracker:
        outcome = train(config, data_root, out_dir, args.name,
                        device_name=args.device, tracker=tracker, extra_tags=None)

    elapsed = time.perf_counter() - started
    summary = {
        "checkpoint": paths.relative(outcome.checkpoint),
        "best_epoch": outcome.best_epoch,
        "best_val_auroc": round(outcome.best_val_auroc, 6),
        "val": outcome.metrics.get("val", {}),
        "seconds": round(elapsed, 1),
        "mlflow_run_id": outcome.run_id,
    }
    print(json.dumps(summary, indent=2))
    print("\ncheckpoint: {0}  ({1:.1f}s)".format(paths.relative(outcome.checkpoint), elapsed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
