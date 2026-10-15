#!/usr/bin/env python
"""The automated retraining pipeline: drift alert in, deployed model out.

    python pipelines/retrain.py --telemetry artifacts/telemetry/drifted.jsonl \\
        --capture artifacts/capture/drifted --reason "lamp_ageing drift"

Stages, in dependency order:

    collect   what the device kept, and which of it came after the drift began
    label     turn kept frames into labelled training data
    dataset   build the retraining set: captured frames plus a replay buffer
    train     fine-tune from the incumbent checkpoint
    export    ONNX
    quantise  INT8, on the same calibration policy as the baseline
    evaluate  on BOTH holdouts: the original one and the drifted one
    gate      promote or block, on thresholds from configs/deploy.yaml
    register  a new version, parented to the incumbent
    deploy    promote, capture a fresh drift reference, run the canary

Three decisions in here are the ones that matter.

**The replay buffer, balanced in both directions.**  Training only on the frames
captured during the drift produces a model that is excellent under the new
conditions and has forgotten the old ones -- and because the alert came from the
new conditions, nothing in the pipeline would notice.  So the retraining set is
captured frames plus the original training set, mixed to a target share.

The mixing has to work both ways, and both failures were observed here.  On the
full dataset the captured frames were under ten percent of each epoch and the
retrained model barely moved; on the tiny one they were sixty percent and it
forgot the reference domain.  The gate blocked both candidates, correctly, which
is the argument for having a gate rather than a checklist.

**Labels come from an oracle, and that is a simulation.**  On a real line these
frames go to a human, and the pipeline blocks for hours or days waiting.  Here
the generator's ground truth stands in.  ``docs/limitations.md`` says so, and
``--label-latency-frames`` records what the wait would have cost so the runbook
can quote a realistic cycle time rather than the one this script measures.

**The gate is asymmetric.**  Must improve materially on the drifted holdout;
must not regress more than a little on the original one; must clear absolute
floors on both.  A gate with only the improvement condition is not a gate.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

from edgeinspect import paths  # noqa: E402
from edgeinspect.config import deep_merge, load_yaml  # noqa: E402
from edgeinspect.data import dataset as ds  # noqa: E402
from edgeinspect.edge.telemetry import CaptureStore  # noqa: E402
from edgeinspect.edge.telemetry import read as read_telemetry  # noqa: E402
from edgeinspect.pipeline import Pipeline, Stage  # noqa: E402
from edgeinspect.registry import Registry  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--telemetry", default="artifacts/telemetry/drifted.jsonl")
    parser.add_argument("--capture", default="artifacts/capture/drifted")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--export-config", default="configs/export.yaml")
    parser.add_argument("--deploy-config", default="configs/deploy.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--drift-config", default="configs/drift.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--reason", default="drift-triggered retraining")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--replay-fraction", type=float, default=1.0,
                        help="fraction of the original training set to replay")
    parser.add_argument("--captured-fraction", type=float, default=0.4,
                        help="target share of the retraining set drawn from captured frames")
    parser.add_argument("--since-frame", type=int, default=None,
                        help="discard frames captured before this index; off by default")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--label-latency-frames", type=int, default=0,
                        help="recorded, not waited for; see the module docstring")
    parser.add_argument("--no-deploy", action="store_true",
                        help="run everything up to and including the gate, then stop")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--post-drift-stream", default="post_drift",
                        help="the stream describing the NEW normal; the drift "
                             "reference and the canary are captured under it")
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--skip", nargs="+", default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_collect(context: dict) -> dict:
    args = context["args"]
    telemetry_path = paths.resolve(args.telemetry)
    capture_root = paths.resolve(args.capture)

    records, dropped = read_telemetry(telemetry_path)
    index = CaptureStore.load(capture_root)
    frames = list(index.get("frames", []))

    # Everything the device kept, unless a cutoff is asked for explicitly.
    #
    # An earlier version filtered to frames at or after the drift onset, using
    # the onset from configs/data.yaml.  Two things were wrong with that.  The
    # small one: a shortened run scales its onset, so the configured value was
    # the wrong index and threw away frames that had already drifted.  The
    # large one: production does not know when the drift began.  A pipeline
    # that depends on knowing is a pipeline that only works in a simulation,
    # and the device's own capture policy is already the sampling decision.
    cutoff = args.since_frame
    if cutoff:
        kept = [f for f in frames if int(f["frame_id"]) >= int(cutoff)]
    else:
        kept = frames

    by_reason: dict = {}
    for frame in kept:
        by_reason[frame["reason"]] = by_reason.get(frame["reason"], 0) + 1

    context["captured"] = kept
    return {
        "telemetry": paths.relative(telemetry_path),
        "telemetry_records": len(records),
        "telemetry_dropped": dropped,
        "capture_index": paths.relative(capture_root / "index.json"),
        "captured_total": len(frames),
        "captured_used": len(kept),
        "since_frame": cutoff,
        "known_onset": context.get("onset"),
        "by_reason": by_reason,
        "capture_policy": index.get("policy", {}),
    }


def stage_label(context: dict) -> dict:
    """Attach labels and boxes to the captured frames.

    The oracle stands in for a human adjudicating flagged parts.  What is real
    here is the shape of the step: a frame is usable for retraining only if it
    has a mask, and frames the device kept without one are dropped rather than
    silently treated as defect-free.  Treating unlabelled frames as good is the
    single fastest way to teach a retrained model to miss the defect that
    triggered the retraining.
    """
    args = context["args"]
    labelled = []
    unlabelled = 0
    for frame in context["captured"]:
        mask_path = frame.get("mask")
        if not mask_path:
            unlabelled += 1
            continue
        mask = ds.read_png(paths.resolve(mask_path))
        from edgeinspect.data.mvtec import _boxes_from_mask

        defects = _boxes_from_mask(mask)
        labelled.append({
            "id": frame["id"],
            "image": frame["image"],
            "mask": mask_path,
            "label": 1 if defects else 0,
            "domain": "production/" + frame.get("stream", "unknown"),
            "defects": defects,
        })

    context["labelled"] = labelled
    positives = sum(item["label"] for item in labelled)
    return {
        "labelled": len(labelled),
        "dropped_unlabelled": unlabelled,
        "defective": positives,
        "defective_fraction": round(positives / len(labelled), 4) if labelled else 0.0,
        "labelling": "oracle (simulation); a real line adjudicates these by hand",
        "label_latency_frames_recorded": args.label_latency_frames,
    }


def stage_dataset(context: dict) -> dict:
    """Build the retraining dataset: captured frames plus a replay buffer.

    Nothing is copied.  The index references the original files by
    repository-relative path, so a retraining run costs a few kilobytes rather
    than a duplicate of the dataset, and the frames a run trained on can still
    be found afterwards.
    """
    args = context["args"]
    run_dir = context["run_dir"]
    original_root = paths.resolve(args.data_root)
    new_root = run_dir / "dataset"

    original_train = ds.load_index(original_root, "train")
    original_val = ds.load_index(original_root, "val")

    rng = np.random.default_rng(1234)
    replay_count = int(round(len(original_train) * float(args.replay_fraction)))
    replay_indices = rng.permutation(len(original_train))[:replay_count]
    replay = [original_train[i].to_dict() for i in sorted(replay_indices)]

    labelled = list(context["labelled"])
    if not labelled:
        raise RuntimeError(
            "no labelled frames were captured, so retraining would reproduce the "
            "incumbent exactly.  Check the capture policy in scripts/simulate_edge.py."
        )
    # Hold a fifth of the captured frames out for validation, stratified by
    # label so the validation split is not accidentally all good parts.
    positives = [f for f in labelled if f["label"] == 1]
    negatives = [f for f in labelled if f["label"] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    val_captured = positives[:max(len(positives) // 5, 1)] + \
        negatives[:max(len(negatives) // 5, 1)]
    val_ids = {f["id"] for f in val_captured}
    train_captured = [f for f in labelled if f["id"] not in val_ids]

    # Balance the two halves of the retraining set, in both directions.
    #
    # The failure runs one way on a big dataset and the other way on a small
    # one, and both were observed here.
    #
    # Too little new data: a production stream yields a few hundred kept frames
    # against a training set of thousands, so a naive concatenation gives the
    # new domain under ten percent of each epoch and the retrained model barely
    # moves.  On the first run that was 177 captured frames against 1,600
    # replayed, and the gate blocked a candidate that had learnt nothing.
    #
    # Too much new data: on the tiny preset the same stream yields 236 captured
    # frames against 160 replayed, so the new domain is sixty percent of every
    # epoch, the model forgets the original one, and the gate blocks it for
    # regressing on the reference holdout.  The gate was right both times, which
    # is the argument for having it.
    #
    # So the minority half is repeated until the captured share reaches the
    # target.  Repetition rather than subsampling, in either direction: throwing
    # away production frames discards the only evidence of the new conditions,
    # and cutting replay is precisely how the model forgets.  Repeating is safe
    # because augmentation is applied per sample, so a repeated frame is not an
    # identical example.
    target = float(args.captured_fraction)
    captured_repeats = 1
    replay_repeats = 1
    if train_captured and replay and 0.0 < target < 1.0:
        wanted_captured = target / (1.0 - target) * len(replay)
        if wanted_captured >= len(train_captured):
            captured_repeats = max(1, int(round(wanted_captured / len(train_captured))))
        else:
            wanted_replay = (1.0 - target) / target * len(train_captured)
            replay_repeats = max(1, int(round(wanted_replay / len(replay))))
    new_train = replay * replay_repeats + train_captured * captured_repeats
    new_val = [record.to_dict() for record in original_val] + val_captured

    for split, items in (("train", new_train), ("val", new_val)):
        target = new_root / split
        paths.ensure(target)
        with (target / "index.json").open("w", encoding="utf-8") as handle:
            json.dump(items, handle, indent=2)

    original_meta = ds.load_meta(original_root)
    ds.write_meta(new_root, {
        "source": "retraining",
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preset": "retrain",
        "seed": original_meta.get("seed"),
        "image_size": original_meta.get("image_size", 256),
        "licence": original_meta.get("licence"),
        "parent_dataset": paths.relative(original_root),
        "splits": {
            "train": {"count": len(new_train),
                      "replayed_unique": len(replay),
                      "replay_repeats": replay_repeats,
                      "captured_unique": len(train_captured),
                      "captured_repeats": captured_repeats,
                      "defective": sum(int(i["label"]) for i in new_train)},
            "val": {"count": len(new_val), "replayed": len(original_val),
                    "captured": len(val_captured),
                    "defective": sum(int(i["label"]) for i in new_val)},
        },
    })

    context["retrain_root"] = new_root
    return {
        "root": paths.relative(new_root),
        "train": len(new_train),
        "train_replayed_unique": len(replay),
        "train_replay_repeats": replay_repeats,
        "train_captured_unique": len(train_captured),
        "train_captured_repeats": captured_repeats,
        "train_captured_share": round(
            len(train_captured) * captured_repeats / max(len(new_train), 1), 4),
        "train_defective": sum(int(i["label"]) for i in new_train),
        "val": len(new_val),
        "val_captured": len(val_captured),
        "replay_fraction": args.replay_fraction,
        "captured_fraction_target": target,
    }


def stage_train(context: dict) -> dict:
    args = context["args"]
    run_dir = context["run_dir"]
    config = load_yaml(args.train_config)
    overrides = {"optim": {"epochs": int(args.epochs)}}
    if args.lr is not None:
        overrides["optim"]["lr"] = float(args.lr)
    config = deep_merge(config, overrides)

    from edgeinspect.train import track
    from edgeinspect.train.loop import load_checkpoint, pick_device, train

    incumbent = context.get("incumbent_checkpoint")
    models_dir = run_dir / "models"
    paths.ensure(models_dir)
    name = "surface_defect_v{0}".format(context["next_version"])

    device = pick_device(args.device)
    warm_start = None
    if incumbent and paths.resolve(incumbent).exists():
        # Fine-tune from the incumbent rather than starting over.  Retraining
        # from scratch on every drift event throws away everything the model
        # learnt about defect appearance, which has not changed; what changed is
        # the imaging conditions.
        model, _meta = load_checkpoint(paths.resolve(incumbent), device=device)
        warm_start = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    mlflow_config = config.get("mlflow", {})
    with track(experiment=mlflow_config.get("experiment", "edge-inspection"),
               run_name=name + "-retrain",
               tracking_uri=mlflow_config.get("tracking_uri"),
               enabled=not args.no_mlflow,
               tags={"stage": "retrain", "trigger": args.reason,
                     "parent_version": context.get("incumbent_version")}) as tracker:
        outcome = train(config, context["retrain_root"], models_dir, name,
                        device_name=args.device, tracker=tracker,
                        initial_state=warm_start)

    context["checkpoint"] = outcome.checkpoint
    context["mlflow_run_id"] = outcome.run_id
    return {
        "checkpoint": paths.relative(outcome.checkpoint),
        "best_epoch": outcome.best_epoch,
        "best_val_auroc": round(outcome.best_val_auroc, 6),
        "warm_started_from": paths.relative(paths.resolve(incumbent)) if warm_start else None,
        "epochs": int(args.epochs),
        "mlflow_run_id": outcome.run_id,
    }


def stage_export(context: dict) -> dict:
    args = context["args"]
    export_config = load_yaml(args.export_config).get("onnx", {})
    from edgeinspect.export import onnx_export
    from edgeinspect.train.loop import load_checkpoint

    checkpoint = context["checkpoint"]
    model, meta = load_checkpoint(checkpoint)
    onnx_path = checkpoint.parent / (checkpoint.stem + "_fp32.onnx")
    onnx_export.export(model, onnx_path, {
        "name": meta.get("name", checkpoint.stem),
        "precision": "fp32",
        "in_channels": model.stem.conv.in_channels,
        "stride": meta.get("stride", 8),
        "input": meta.get("input", {}),
        "threshold": meta.get("threshold", 0.0),
        "top_k": meta.get("top_k", 8),
        "target_fpr": meta.get("target_fpr", 0.02),
        "cell_threshold": 0.5,
        "training": {"retrain": True, "parent_version": context.get("incumbent_version")},
    }, input_size=int(export_config.get("input_size", 256)),
        opset=int(export_config.get("opset", 17)), checkpoint_path=checkpoint)
    verification = onnx_export.verify(
        onnx_path, model, tolerance=float(export_config.get("verify_tolerance", 1e-4)))
    context["onnx"] = onnx_path
    return {"onnx": paths.relative(onnx_path), "verification": verification}


def stage_quantise(context: dict) -> dict:
    args = context["args"]
    quantise_config = load_yaml(args.export_config).get("quantise", {})
    from edgeinspect.quantise import calibration, ptq

    onnx_path = context["onnx"]
    # Calibrate on the retraining set, not the original one.  The activation
    # ranges that matter are the ones production now produces.
    from edgeinspect.artefacts import read_sidecar
    from edgeinspect.models.preprocess import InputSpec

    spec = InputSpec.from_sidecar(read_sidecar(onnx_path))
    frames, manifest = calibration.build(
        context["retrain_root"], split="train",
        count=int(quantise_config.get("calibration_frames", 192)),
        defect_fraction=float(quantise_config.get("calibration_defect_fraction", 0.5)),
        spec=spec)
    calibration.save(frames, manifest, onnx_path.parent)
    built = ptq.build_variants(onnx_path, onnx_path.parent, frames, manifest,
                               methods=list(quantise_config.get("methods",
                                                                ["minmax", "percentile"])),
                               per_channel=bool(quantise_config.get("per_channel", True)))
    context["variants"] = built
    return {
        "variants": {k: paths.relative(v) for k, v in built.items()},
        "calibration_frames": manifest["count"],
        "calibration_defect_fraction": manifest["defect_fraction_actual"],
        "sizes": ptq.size_report({"fp32": onnx_path, **built}),
    }


def stage_evaluate(context: dict) -> dict:
    """Score the candidate on both holdouts, using the ORIGINAL data root.

    The holdouts must not come from the retraining set.  The captured frames are
    production data the model has now trained on, and evaluating on any of them
    would be measuring memorisation.
    """
    args = context["args"]
    accuracy_path = context["run_dir"] / "accuracy.json"
    models = [str(path) for path in context["variants"].values()]
    models.insert(0, str(context["onnx"]))
    command = [
        sys.executable, str(paths.ROOT / "scripts" / "evaluate.py"),
        "--models", *models,
        "--data-root", str(paths.resolve(args.data_root)),
        "--splits", "test", "test_drift",
        "--out", str(accuracy_path),
    ]
    subprocess.run(command, check=True, cwd=str(paths.ROOT))
    with accuracy_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    context["accuracy"] = payload
    context["accuracy_path"] = accuracy_path

    deploy_config = load_yaml(args.deploy_config)
    key = deploy_config.get("artefact", "int8")
    chosen = None
    for name, entry in payload.get("variants", {}).items():
        if name.endswith(key):
            chosen = (name, entry)
            break
    if chosen is None:
        chosen = sorted(payload.get("variants", {}).items())[0]
    context["candidate_variant"] = chosen[0]
    context["candidate_metrics"] = chosen[1]
    return {
        "accuracy": paths.relative(accuracy_path),
        "candidate": chosen[0],
        "reference_auroc": chosen[1]["splits"]["test"]["image"]["auroc"],
        "drift_auroc": chosen[1]["splits"]["test_drift"]["image"]["auroc"],
        "reference_recall": chosen[1]["splits"]["test"]["image"]["recall"],
        "drift_recall": chosen[1]["splits"]["test_drift"]["image"]["recall"],
    }


def _metric(entry: dict, dotted: str):
    """``drift.image.auroc`` -> the number, from an evaluate.py variant block."""
    split_key, rest = dotted.split(".", 1)
    split = "test_drift" if split_key == "drift" else (
        "test" if split_key == "reference" else split_key)
    node = entry["splits"].get(split, {})
    for part in rest.split("."):
        node = node.get(part, {})
    return None if isinstance(node, dict) else float(node)


def stage_gate(context: dict) -> dict:
    """Promote or block.  This is the step that has to be able to say no."""
    args = context["args"]
    gate_config = load_yaml(args.deploy_config).get("gate", {})
    if not gate_config:
        raise RuntimeError(
            "no gate configured.  A retraining pipeline that deploys whatever it "
            "produced is not a pipeline with a gate, it is a pipeline that looks "
            "like it has one."
        )
    candidate = context["candidate_metrics"]
    incumbent = context.get("incumbent_metrics") or {}

    checks = []

    improvement = gate_config.get("improvement", {})
    if improvement:
        metric = improvement["metric"]
        value = _metric(candidate, metric)
        baseline = None
        split_key = metric.split(".", 1)[0]
        baseline_block = incumbent.get(split_key, {})
        for part in metric.split(".", 1)[1].split("."):
            baseline_block = baseline_block.get(part, {}) if isinstance(
                baseline_block, dict) else baseline_block
        if isinstance(baseline_block, (int, float)):
            baseline = float(baseline_block)
        delta = None if baseline is None else value - baseline
        passed = baseline is None or delta >= float(improvement["min_delta"])
        checks.append({
            "kind": "improvement", "metric": metric, "candidate": value,
            "incumbent": baseline, "delta": None if delta is None else round(delta, 6),
            "required": improvement["min_delta"], "passed": bool(passed),
            "note": ("no incumbent metric recorded; improvement check skipped"
                     if baseline is None else ""),
        })

    regression = gate_config.get("regression", {})
    if regression:
        metric = regression["metric"]
        value = _metric(candidate, metric)
        split_key = metric.split(".", 1)[0]
        baseline_block = incumbent.get(split_key, {})
        for part in metric.split(".", 1)[1].split("."):
            baseline_block = baseline_block.get(part, {}) if isinstance(
                baseline_block, dict) else baseline_block
        baseline = float(baseline_block) if isinstance(baseline_block, (int, float)) else None
        drop = None if baseline is None else baseline - value
        passed = baseline is None or drop <= float(regression["max_drop"])
        checks.append({
            "kind": "regression", "metric": metric, "candidate": value,
            "incumbent": baseline, "drop": None if drop is None else round(drop, 6),
            "allowed": regression["max_drop"], "passed": bool(passed),
        })

    for floor in gate_config.get("floors", []):
        metric = floor["metric"]
        value = _metric(candidate, metric)
        if "min" in floor:
            passed = value is not None and value >= float(floor["min"])
            checks.append({"kind": "floor", "metric": metric, "candidate": value,
                           "min": floor["min"], "passed": bool(passed)})
        if "max" in floor:
            passed = value is not None and value <= float(floor["max"])
            checks.append({"kind": "ceiling", "metric": metric, "candidate": value,
                           "max": floor["max"], "passed": bool(passed)})

    passed = all(check["passed"] for check in checks)
    context["gate_passed"] = passed
    context["gate_checks"] = checks
    for check in checks:
        print("  [{0}] {1:<12} {2:<38} {3}".format(
            "PASS" if check["passed"] else "FAIL", check["kind"], check["metric"],
            check.get("candidate")))
    print("  gate: " + ("PROMOTE" if passed else "BLOCK"))
    return {"passed": passed, "checks": checks,
            "decision": "promote" if passed else "block"}


def stage_register(context: dict) -> dict:
    args = context["args"]
    registry: Registry = context["registry"]
    artefacts = {"fp32": context["onnx"], "checkpoint": context["checkpoint"]}
    precisions = {"fp32": "fp32", "checkpoint": "fp32"}
    for key, path in context["variants"].items():
        artefacts[key] = path
        precisions[key] = "int8"

    metrics = {"variant": context["candidate_variant"],
               "threshold": context["candidate_metrics"].get("threshold")}
    for split, block in context["candidate_metrics"].get("splits", {}).items():
        key = "drift" if "drift" in split else ("reference" if split == "test" else split)
        metrics[key] = {"image": block.get("image", {}), "boxes": block.get("boxes", {})}

    dataset_info = {}
    meta_path = context["retrain_root"] / "dataset.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as handle:
            dataset_info = json.load(handle)

    version = registry.register(
        artefacts, metrics, dataset_info,
        parent=context.get("incumbent_version"),
        reason=args.reason,
        mlflow_run_id=context.get("mlflow_run_id"),
        notes={"gate": context["gate_checks"], "retrain_run": paths.relative(
            context["run_dir"])},
        precisions=precisions,
    )
    context["version"] = version.version

    if not context["gate_passed"]:
        registry.block(version.version, reason="gate failed: " + json.dumps(
            [c for c in context["gate_checks"] if not c["passed"]]))
    return {"version": version.version, "stage": registry.get(version.version).stage}


def stage_deploy(context: dict) -> dict:
    args = context["args"]
    if not context["gate_passed"]:
        return {"deployed": False,
                "reason": "gate blocked version {0}".format(context["version"])}
    if args.no_deploy:
        return {"deployed": False, "reason": "--no-deploy"}

    # The new reference and the canary are captured under the conditions the
    # line is now running in, not the ones it used to run in.  Capturing them
    # under the old normal leaves the monitor firing on a model that has
    # already adapted, which is the fastest way to have it switched off.
    command = [
        sys.executable, str(paths.ROOT / "scripts" / "deploy.py"),
        "--config", str(args.deploy_config),
        "promote",
        "--version", str(context["version"]),
        "--reason", args.reason,
        "--stream", args.post_drift_stream,
    ]
    completed = subprocess.run(command, cwd=str(paths.ROOT))
    deployed = completed.returncode == 0
    return {
        "deployed": deployed,
        "version": context["version"],
        "returncode": completed.returncode,
        "note": ("promoted, reference captured, canary passed" if deployed
                 else "promotion failed its canary and was rolled back"),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    run_dir = paths.RETRAIN / run_id
    paths.ensure(run_dir)

    deploy_config = load_yaml(args.deploy_config)
    registry = Registry(model_name=deploy_config.get("model_name", "surface-defect"))
    incumbent = registry.production()

    data_config = load_yaml(args.data_config)
    stream = Path(args.telemetry).stem
    onset = None
    spec = data_config.get("streams", {}).get(stream, {})
    if spec.get("to") is not None:
        onset = int(spec.get("ramp_start", 0))

    context = {
        "args": args,
        "run_dir": run_dir,
        "registry": registry,
        "onset": onset,
        "incumbent_version": incumbent.version if incumbent else None,
        "incumbent_metrics": incumbent.metrics if incumbent else {},
        "incumbent_checkpoint": (
            incumbent.artefacts["checkpoint"].path
            if incumbent and "checkpoint" in incumbent.artefacts else None),
        "next_version": registry.latest_version_number() + 1,
    }

    stages = [
        Stage("collect", stage_collect, description="what the device kept"),
        Stage("label", stage_label, needs=["collect"], description="label the kept frames"),
        Stage("dataset", stage_dataset, needs=["label"],
              description="captured frames plus a replay buffer"),
        Stage("train", stage_train, needs=["dataset"],
              description="fine-tune from the incumbent"),
        Stage("export", stage_export, needs=["train"], description="ONNX"),
        Stage("quantise", stage_quantise, needs=["export"], description="INT8"),
        Stage("evaluate", stage_evaluate, needs=["quantise"],
              description="both holdouts, original data root"),
        Stage("gate", stage_gate, needs=["evaluate"], description="promote or block"),
        Stage("register", stage_register, needs=["gate"], description="new version"),
        Stage("deploy", stage_deploy, needs=["register"],
              description="promote, reference, canary"),
    ]

    pipeline = Pipeline("retrain", run_dir, stages, context=context)
    try:
        pipeline.run(only=args.only, skip=args.skip)
    finally:
        summary = pipeline.summary()
        summary["seconds"] = round(time.perf_counter() - started, 2)
        summary["run_id"] = run_id
        with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    deploy_result = pipeline.results.get("deploy")
    print("\n" + json.dumps(pipeline.summary(), indent=2))
    print("run directory: " + paths.relative(run_dir))

    # Read the outcome off the pipeline, which is what the stages wrote to.
    final = pipeline.context
    if not final.get("gate_passed", False):
        return 1
    if deploy_result and deploy_result.outputs.get("deployed") is False and not args.no_deploy:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
