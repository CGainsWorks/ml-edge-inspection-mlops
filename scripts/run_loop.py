#!/usr/bin/env python
"""The whole operational loop, end to end, in one command.

    python scripts/run_loop.py                 # the published run
    python scripts/run_loop.py --quick         # short streams; what CI runs

    1  register and deploy the baseline, capture the drift reference, canary it
    2  run the CLEAN control stream  -> the monitor must NOT fire
    3  run the DRIFTED ramp          -> the monitor MUST fire, and when
    4  run the POST-DRIFT steady state on the incumbent -> the "before" number
    5  retrain from what the device captured, gate it, deploy it, canary it
    6  run the POST-DRIFT steady state on the new model -> the "after" number,
       and the monitor must now be QUIET, because the reference was recaptured
       under the conditions the line is actually in
    7  roll back and prove the device is serving the previous version

Every step shells out to the same script a person would run, and the command is
printed before it runs, so this file doubles as an executable version of
``docs/RUNBOOK.md``.  If a step here works, the documented command works.

Step 2 is the one that is easy to leave out and is the reason to trust the rest.
A drift detector that has only ever been shown drifting data has never been
shown not to fire, and a monitor that fires on everything is indistinguishable
from a working one until the day it costs somebody a retraining run.

Steps 4 and 6 are the same stream on two different models, which is what makes
the recovery a comparison rather than an anecdote.  They use the *post-drift*
stream -- the steady state after the lamp has aged -- rather than replaying the
ramp, because the question after a retrain is "how does the line run now", and
the ramp is a transient that has already happened.

Step 6 also checks that the monitor has gone quiet.  It should have: the new
reference was captured under the post-drift conditions, so the line's current
state is now the reference.  Capturing it under the old conditions instead would
leave the monitor firing forever on a model that has already adapted.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import shutil
import subprocess
import sys
import time

from edgeinspect import paths
from edgeinspect.config import load_yaml
from edgeinspect.registry import Registry

PYTHON = sys.executable


class Step:
    """One numbered step of the loop, timed and recorded."""

    def __init__(self, number: int, title: str):
        self.number = number
        self.title = title
        self.started = time.perf_counter()
        print("\n" + "=" * 78)
        print("STEP {0}  {1}".format(number, title))
        print("=" * 78, flush=True)

    def done(self, **outputs) -> dict:
        seconds = time.perf_counter() - self.started
        print("-- step {0} finished in {1:.1f}s".format(self.number, seconds), flush=True)
        return {"step": self.number, "title": self.title,
                "seconds": round(seconds, 1), **outputs}


def run(command, check: bool = True, capture: bool = False):
    """Run a command, printing it first.  Returns the CompletedProcess."""
    printable = " ".join(
        str(part) if " " not in str(part) else '"{0}"'.format(part) for part in command)
    printable = printable.replace(str(PYTHON), "python")
    print("\n$ " + printable, flush=True)
    completed = subprocess.run(command, cwd=str(paths.ROOT), check=False,
                               capture_output=capture, text=True)
    if capture and completed.stdout:
        print(completed.stdout, end="")
    if check and completed.returncode != 0:
        if capture and completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise SystemExit("command failed with exit code {0}: {1}".format(
            completed.returncode, printable))
    return completed


def script(name: str, *args):
    return [PYTHON, str(paths.ROOT / "scripts" / name), *[str(a) for a in args]]


def pipeline(name: str, *args):
    return [PYTHON, str(paths.ROOT / "pipelines" / name), *[str(a) for a in args]]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--model-prefix", default="models/surface_defect")
    parser.add_argument("--quick", action="store_true",
                        help="short streams and a short retrain; what CI runs")
    parser.add_argument("--clean-frames", type=int, default=None)
    parser.add_argument("--drift-frames", type=int, default=None)
    parser.add_argument("--steady-frames", type=int, default=None)
    parser.add_argument("--retrain-epochs", type=int, default=None)
    parser.add_argument("--accuracy", default="results/accuracy.json",
                        help="the evaluate.py report to register as the baseline's metrics")
    parser.add_argument("--deploy-config", default=None,
                        help="gate and canary thresholds; --quick defaults to "
                             "configs/deploy_tiny.yaml, whose floors are "
                             "calibrated for the tiny dataset")
    parser.add_argument("--out", default="results/loop.json")
    parser.add_argument("--fresh", action="store_true",
                        help="delete artifacts/ first: registry, deployments, telemetry")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--skip-rollback", action="store_true")
    return parser.parse_args(argv)


def read_json(path) -> dict:
    with paths.resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    clean_frames = args.clean_frames or (320 if args.quick else 900)
    drift_frames = args.drift_frames or (420 if args.quick else 1200)
    steady_frames = args.steady_frames or (300 if args.quick else 600)
    retrain_epochs = args.retrain_epochs or (10 if args.quick else 14)

    prefix = args.model_prefix
    required = [prefix + "_fp32.onnx", prefix + "_int8.onnx", prefix + ".pt"]
    missing = [p for p in required if not paths.resolve(p).exists()]
    if missing:
        raise SystemExit(
            "missing artefact(s): {0}\nRun `make train export quantise` first."
            .format(", ".join(missing)))

    if args.fresh:
        for directory in (paths.REGISTRY, paths.DEPLOY, paths.TELEMETRY,
                          paths.CAPTURE, paths.RETRAIN):
            if directory.exists():
                shutil.rmtree(directory)
        print("cleared artifacts/: registry, deployments, telemetry, capture, retrain runs")

    timeline = []
    # An absolute accuracy floor is a property of the dataset, and --quick
    # trains on a tenth of the data.  The gate's shape is the same either way;
    # configs/deploy_tiny.yaml explains which numbers differ and why.
    deploy_config_path = args.deploy_config or (
        "configs/deploy_tiny.yaml" if args.quick else "configs/deploy.yaml")
    deploy_config = load_yaml(deploy_config_path)
    print("gate and canary thresholds: " + deploy_config_path)
    registry = Registry(model_name=deploy_config.get("model_name", "surface-defect"))

    # -- 1 -------------------------------------------------------------------
    step = Step(1, "Register and deploy the baseline")
    if registry.production() is None:
        variants = [p for p in (prefix + "_fp32.onnx", prefix + "_int8.onnx",
                                prefix + "_int8_percentile.onnx",
                                prefix + "_int8_keephead.onnx")
                    if paths.resolve(p).exists()]
        run(script("deploy.py", "register", "--models", *variants,
                   "--checkpoint", prefix + ".pt",
                   "--accuracy", args.accuracy,
                   "--dataset", args.data_root,
                   "--reason", "baseline"), capture=True)
        # Reload: the registration happened in a subprocess, so this object's
        # view of the registry is one write out of date.
        registry = Registry(model_name=registry.model_name)
        baseline_version = registry.latest_version_number()
        run(script("deploy.py", "--config", deploy_config_path, "promote",
                   "--version", baseline_version,
                   "--reason", "baseline deployment"), capture=True)
    else:
        baseline_version = registry.production().version
        print("version {0} is already in production; reusing it".format(baseline_version))
    registry = Registry(model_name=registry.model_name)
    timeline.append(step.done(version=baseline_version,
                              deployed=registry.production().version))

    # -- 2 -------------------------------------------------------------------
    step = Step(2, "Clean stream: the monitor must NOT fire")
    run(script("simulate_edge.py", "--stream", "clean", "--frames", clean_frames,
               "--out", "results/edge_clean.json"), capture=True)
    clean_drift = run(script("detect_drift.py", "--telemetry",
                             "artifacts/telemetry/clean.jsonl",
                             "--expect-no-trigger",
                             "--out", "results/drift_clean.json", "--quiet"),
                      capture=True)
    clean_report = read_json("results/drift_clean.json")
    timeline.append(step.done(
        frames=clean_report["frames"],
        triggered=clean_report["triggered"],
        windows=len(clean_report["windows"]),
        false_alarms=sum(1 for w in clean_report["windows"] if w["triggered"]),
        exit_code=clean_drift.returncode))

    # -- 3 -------------------------------------------------------------------
    step = Step(3, "Drifted stream: the monitor MUST fire")
    run(script("simulate_edge.py", "--stream", "drifted", "--frames", drift_frames,
               "--capture", "--out", "results/edge_drifted.json"), capture=True)
    run(script("detect_drift.py", "--telemetry", "artifacts/telemetry/drifted.jsonl",
               "--expect-trigger", "--out", "results/drift_drifted.json"), capture=True)
    drift_report = read_json("results/drift_drifted.json")
    edge_drifted = read_json("results/edge_drifted.json")
    timeline.append(step.done(
        frames=drift_report["frames"],
        triggered=drift_report["triggered"],
        onset_frame=drift_report["onset_frame"],
        first_trigger_frame=drift_report["first_trigger_frame"],
        detection_latency_frames=drift_report["detection_latency_frames"],
        sequential_earliest_frame=drift_report["sequential"].get("earliest_frame"),
        flagged_rate=edge_drifted["flagged_rate"],
        captured=edge_drifted["captured"]))

    # -- 4 -------------------------------------------------------------------
    step = Step(4, "Post-drift steady state on the incumbent: the BEFORE number")
    run(script("simulate_edge.py", "--stream", "post_drift", "--frames", steady_frames,
               "--telemetry", "artifacts/telemetry/post_drift_before.jsonl",
               "--out", "results/edge_post_drift_before.json"), capture=True)
    before = read_json("results/edge_post_drift_before.json")
    timeline.append(step.done(
        model_version=before.get("model_version"),
        frames=before["frames"],
        flagged_rate=before["flagged_rate"],
        latency_p95_ms=before["latency_ms"]["p95"]))

    # -- 5 -------------------------------------------------------------------
    step = Step(5, "Retrain from what the device captured, gate it, deploy it")
    retrain_command = pipeline("retrain.py",
                               "--telemetry", "artifacts/telemetry/drifted.jsonl",
                               "--capture", "artifacts/capture/drifted",
                               "--data-root", args.data_root,
                               "--epochs", retrain_epochs,
                               "--deploy-config", deploy_config_path,
                               "--reason", "lamp_ageing drift detected at frame {0}".format(
                                   drift_report["first_trigger_frame"]))
    if args.no_mlflow:
        retrain_command.append("--no-mlflow")
    retrain = run(retrain_command, check=False, capture=True)
    registry = Registry(model_name=registry.model_name)
    run_dirs = sorted(paths.RETRAIN.glob("*/manifest.json"))
    manifest = read_json(run_dirs[-1]) if run_dirs else {}
    stages = {s["name"]: s for s in manifest.get("stages", [])}
    gate = stages.get("gate", {})
    deploy_stage = stages.get("deploy", {})
    evaluate_stage = stages.get("evaluate", {})
    new_version = registry.production().version if registry.production() else None
    failed = [name for name, stage in stages.items() if stage.get("status") == "failed"]
    timeline.append(step.done(
        returncode=retrain.returncode,
        gate=gate.get("outputs", {}).get("decision"),
        gate_checks=gate.get("outputs", {}).get("checks", []),
        deployed=deploy_stage.get("outputs", {}).get("deployed"),
        failed_stages=failed,
        candidate_reference_auroc=evaluate_stage.get("outputs", {}).get("reference_auroc"),
        candidate_drift_auroc=evaluate_stage.get("outputs", {}).get("drift_auroc"),
        production_version=new_version,
        run_dir=manifest.get("run_dir")))

    if retrain.returncode != 0:
        # A pipeline that crashed and a gate that said no are different
        # outcomes, and reporting a crash as a gate decision would turn a bug
        # into a feature every time.
        if failed:
            outcome = "retraining failed in stage(s): " + ", ".join(failed)
            print("\nThe retraining pipeline FAILED in: " + ", ".join(failed))
            print("See " + str(manifest.get("run_dir")) + "/manifest.json.")
        else:
            outcome = "gate blocked"
            print("\nThe retraining pipeline did not deploy.  That is a legitimate "
                  "outcome: the gate is allowed to say no.  The loop stops here.")
        payload = _write(args, timeline, started, registry, outcome=outcome)
        _print_timeline(payload)
        return 1

    # -- 6 -------------------------------------------------------------------
    step = Step(6, "Post-drift steady state on the new model: the AFTER number")
    run(script("simulate_edge.py", "--stream", "post_drift", "--frames", steady_frames,
               "--telemetry", "artifacts/telemetry/post_drift_after.jsonl",
               "--out", "results/edge_post_drift_after.json"), capture=True)
    after = read_json("results/edge_post_drift_after.json")
    # The monitor must now be quiet: the reference was recaptured under these
    # very conditions when the new version was promoted.
    quiet = run(script("detect_drift.py", "--telemetry",
                       "artifacts/telemetry/post_drift_after.jsonl",
                       "--stream", "post_drift", "--expect-no-trigger",
                       "--out", "results/drift_post_drift_after.json", "--quiet"),
                check=False, capture=True)
    after_report = read_json("results/drift_post_drift_after.json")
    timeline.append(step.done(
        model_version=after.get("model_version"),
        flagged_rate_before=before["flagged_rate"],
        flagged_rate_after=after["flagged_rate"],
        latency_p95_ms=after["latency_ms"]["p95"],
        monitor_quiet=not after_report["triggered"],
        monitor_exit_code=quiet.returncode))

    # -- 7 -------------------------------------------------------------------
    if not args.skip_rollback:
        step = Step(7, "Roll back and prove the device is serving the previous version")
        run(script("deploy.py", "--config", deploy_config_path, "rollback",
                   "--reason", "run_loop.py rollback demonstration"),
            capture=True)
        manifest_after = read_json(paths.DEPLOY / "current.json")
        run(script("simulate_edge.py", "--stream", "clean", "--frames", 60,
                   "--telemetry", "artifacts/telemetry/rollback_check.jsonl",
                   "--out", "results/edge_rollback_check.json"), capture=True)
        rollback_check = read_json("results/edge_rollback_check.json")
        timeline.append(step.done(
            rolled_back_to=manifest_after["version"],
            device_is_serving=rollback_check["model_version"],
            matches=bool(manifest_after["version"] == rollback_check["model_version"])))

    payload = _write(args, timeline, started, Registry(model_name=registry.model_name),
                     outcome="ok")
    _print_timeline(payload)
    return 0


def _write(args, timeline, started, registry, outcome: str) -> dict:
    payload = {
        "outcome": outcome,
        "out": args.out,
        "quick": bool(args.quick),
        "seconds": round(time.perf_counter() - started, 1),
        "production_version": registry.production().version if registry.production() else None,
        "versions": [{"version": v.version, "stage": v.stage, "reason": v.reason}
                     for v in registry.versions()],
        "events": registry.events(),
        "steps": timeline,
    }
    out_path = paths.resolve(args.out)
    paths.ensure(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def _print_timeline(payload: dict) -> None:
    print("\n" + "=" * 78)
    print("LOOP SUMMARY  ({0:.0f}s, outcome: {1})".format(
        payload["seconds"], payload["outcome"]))
    print("=" * 78)
    for entry in payload["steps"]:
        print("\n{0}. {1}  [{2:.1f}s]".format(entry["step"], entry["title"],
                                              entry["seconds"]))
        for key, value in entry.items():
            if key in ("step", "title", "seconds", "gate_checks"):
                continue
            print("     {0:<28} {1}".format(key, value))
    print("\nversions:")
    for version in payload["versions"]:
        print("  v{0:<3} {1:<11} {2}".format(version["version"], version["stage"],
                                             version["reason"][:52]))
    print("\nfull record: " + paths.relative(paths.resolve(payload["out"])))


if __name__ == "__main__":
    raise SystemExit(main())
