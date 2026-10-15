#!/usr/bin/env python
"""How reproducible is the retraining step, really?

    python scripts/retrain_seed_sweep.py --seeds 7 17 27 37

Runs the retraining pipeline several times against the *same* captured frames,
changing only the training seed, and reports what the candidate reached on each
holdout and what the gate decided.

Why this exists
---------------
The loop's recovery number comes from one retraining run.  One run cannot tell
the difference between "retraining recovers the drift" and "that retraining run
happened to recover the drift", and the distinction decides whether CI can
assert the outcome or only the mechanism.

It earned its keep immediately.  CI blocked a candidate that had passed the
identical check locally, and the first guess -- a torch version difference --
was wrong: the same torch and onnxruntime versions reproduced the pass locally.
This sweep showed what was actually happening.  On the tiny dataset the
retrained model gains 0.17 to 0.23 AUROC on the distribution that triggered the
alert and gives up 0.04 to 0.08 on the one it already handled, on **every**
seed.  The variance was not the story; the trade was, and at a tenth of the data
it is not avoidable.  ``configs/deploy_tiny.yaml`` records the consequence.

Writes ``results/retrain_seed_sweep.json``.  Needs telemetry and captured
frames to exist, so run a drifted stream first -- ``make drift`` or any
``run_loop.py``.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import subprocess
import sys
import time

from edgeinspect import paths
from edgeinspect.registry import Registry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 37])
    parser.add_argument("--telemetry", default="artifacts/telemetry/drifted.jsonl")
    parser.add_argument("--capture", default="artifacts/capture/drifted")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--deploy-config", default="configs/deploy.yaml")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--captured-fraction", type=float, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out", default="results/retrain_seed_sweep.json")
    parser.add_argument("--label", default="sweep")
    return parser.parse_args(argv)


def incumbent_metrics() -> dict:
    """What the deployed model scores, for the deltas to be against something."""
    production = Registry().production()
    if production is None:
        return {}
    metrics = production.metrics or {}
    return {
        "version": production.version,
        "reference_auroc": metrics.get("reference", {}).get("image", {}).get("auroc"),
        "drift_auroc": metrics.get("drift", {}).get("image", {}).get("auroc"),
    }


def run_one(args, seed: int) -> dict:
    run_id = "{0}-{1}".format(args.label, seed)
    command = [
        sys.executable, str(paths.ROOT / "pipelines" / "retrain.py"),
        "--telemetry", args.telemetry,
        "--capture", args.capture,
        "--data-root", args.data_root,
        "--deploy-config", args.deploy_config,
        "--epochs", str(args.epochs),
        "--seed", str(seed),
        "--device", args.device,
        "--run-id", run_id,
        "--no-deploy", "--no-mlflow",
    ]
    if args.captured_fraction is not None:
        command += ["--captured-fraction", str(args.captured_fraction)]

    started = time.perf_counter()
    completed = subprocess.run(command, cwd=str(paths.ROOT), capture_output=True,
                               text=True)
    seconds = time.perf_counter() - started

    manifest_path = paths.RETRAIN / run_id / "manifest.json"
    entry = {"seed": seed, "returncode": completed.returncode,
             "seconds": round(seconds, 1), "run_id": run_id}
    if not manifest_path.exists():
        entry["error"] = (completed.stderr or completed.stdout)[-600:]
        return entry

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    stages = {stage["name"]: stage for stage in manifest.get("stages", [])}
    evaluate = stages.get("evaluate", {}).get("outputs", {})
    gate = stages.get("gate", {}).get("outputs", {})
    dataset = stages.get("dataset", {}).get("outputs", {})
    entry.update({
        "reference_auroc": evaluate.get("reference_auroc"),
        "drift_auroc": evaluate.get("drift_auroc"),
        "decision": gate.get("decision"),
        "failed_checks": [c["metric"] for c in gate.get("checks", [])
                          if not c.get("passed")],
        "train_frames": dataset.get("train"),
        "captured_share": dataset.get("train_captured_share"),
        "status": manifest.get("status"),
    })
    return entry


def main(argv=None) -> int:
    args = parse_args(argv)
    if not paths.resolve(args.telemetry).exists():
        raise SystemExit(
            "no telemetry at {0}.  Run a drifted stream first: `make drift`."
            .format(args.telemetry))

    incumbent = incumbent_metrics()
    runs = []
    for seed in args.seeds:
        print("\n== seed {0} ==".format(seed), flush=True)
        entry = run_one(args, seed)
        runs.append(entry)
        if entry.get("reference_auroc") is None:
            print("   crashed: " + str(entry.get("error", ""))[:200])
            continue
        print("   reference {0:.4f}  drift {1:.4f}  {2}".format(
            entry["reference_auroc"], entry["drift_auroc"],
            (entry.get("decision") or "?").upper()), flush=True)

    scored = [r for r in runs if r.get("reference_auroc") is not None]
    payload = {
        "seeds": [int(s) for s in args.seeds],
        "epochs": args.epochs,
        "data_root": args.data_root,
        "deploy_config": args.deploy_config,
        "captured_fraction": args.captured_fraction,
        "incumbent": incumbent,
        "runs": runs,
        "summary": _summarise(scored, incumbent),
    }
    out_path = paths.resolve(args.out)
    paths.ensure(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    print("\n{0:<6} {1:>10} {2:>10} {3:>9}".format("seed", "reference", "drifted",
                                                   "decision"))
    for run in runs:
        if run.get("reference_auroc") is None:
            print("{0:<6} {1:>10} {2:>10} {3:>9}".format(run["seed"], "--", "--",
                                                         "crash"))
            continue
        print("{0:<6} {1:>10.4f} {2:>10.4f} {3:>9}".format(
            run["seed"], run["reference_auroc"], run["drift_auroc"],
            (run.get("decision") or "?")))
    print("\n" + payload["summary"]["sentence"])
    print("wrote " + paths.relative(out_path))
    return 0 if scored else 1


def _summarise(scored, incumbent) -> dict:
    if not scored:
        return {"sentence": "no run produced a score"}
    reference = [r["reference_auroc"] for r in scored]
    drift = [r["drift_auroc"] for r in scored]
    promoted = [r for r in scored if r.get("decision") == "promote"]
    base_reference = incumbent.get("reference_auroc")
    base_drift = incumbent.get("drift_auroc")

    out = {
        "runs": len(scored),
        "promoted": len(promoted),
        "reference_min": round(min(reference), 6),
        "reference_max": round(max(reference), 6),
        "drift_min": round(min(drift), 6),
        "drift_max": round(max(drift), 6),
    }
    if base_reference is not None and base_drift is not None:
        out["reference_drop_min"] = round(base_reference - max(reference), 6)
        out["reference_drop_max"] = round(base_reference - min(reference), 6)
        out["drift_gain_min"] = round(min(drift) - base_drift, 6)
        out["drift_gain_max"] = round(max(drift) - base_drift, 6)
        out["sentence"] = (
            "Over {0} seeds the candidate gains {1:.3f} to {2:.3f} AUROC on the "
            "drifted holdout and gives up {3:.3f} to {4:.3f} on the reference one; "
            "{5} of {0} promoted.".format(
                len(scored), out["drift_gain_min"], out["drift_gain_max"],
                out["reference_drop_min"], out["reference_drop_max"], len(promoted)))
    else:
        out["sentence"] = (
            "Over {0} seeds the reference holdout spans {1:.3f} to {2:.3f} and the "
            "drifted one {3:.3f} to {4:.3f}; {5} promoted.  No incumbent metrics "
            "were registered, so no deltas.".format(
                len(scored), out["reference_min"], out["reference_max"],
                out["drift_min"], out["drift_max"], len(promoted)))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
