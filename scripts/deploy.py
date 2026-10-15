#!/usr/bin/env python
"""Registry and deployment: register, promote, roll back, inspect.

    python scripts/deploy.py register --models models/surface_defect_int8.onnx ... \\
        --accuracy results/accuracy.json --reason "baseline"
    python scripts/deploy.py promote --version 1
    python scripts/deploy.py rollback
    python scripts/deploy.py status
    python scripts/deploy.py history

``promote`` writes the deployment manifest, captures a fresh drift reference
from the newly deployed model's own telemetry, and runs the canary.  If the
canary fails it rolls back on its own and exits non-zero, because a deployment
that has to be undone by hand at 03:00 is a deployment that stays broken until
morning.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

from edgeinspect import paths
from edgeinspect.config import DriftPolicy, load_yaml
from edgeinspect.edge import runtime
from edgeinspect.edge.agent import EdgeAgent
from edgeinspect.edge.source import StreamSpec, SyntheticCamera
from edgeinspect.edge.telemetry import read as read_telemetry
from edgeinspect.monitoring.drift import build_reference
from edgeinspect.registry import Registry


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--drift-config", default="configs/drift.yaml")
    parser.add_argument("--registry-root", default=None)
    parser.add_argument("--deploy-root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    register = sub.add_parser("register", help="register a new version in staging")
    register.add_argument("--models", nargs="+", required=True)
    register.add_argument("--checkpoint", default=None)
    register.add_argument("--accuracy", default=None,
                          help="results/accuracy.json to attach as this version's metrics")
    register.add_argument("--dataset", default="data/processed")
    register.add_argument("--parent", type=int, default=None)
    register.add_argument("--reason", default="")
    register.add_argument("--mlflow-run-id", default=None)

    promote = sub.add_parser("promote", help="promote a version to production")
    promote.add_argument("--version", type=int, required=True)
    promote.add_argument("--artefact", default=None)
    promote.add_argument("--reason", default="")
    promote.add_argument("--no-canary", action="store_true")
    promote.add_argument("--no-reference", action="store_true")
    promote.add_argument("--stream", default=None,
                         help="conditions to capture the new reference and run the "
                              "canary under; defaults to configs/deploy.yaml.  After "
                              "a drift-triggered redeployment this must be the NEW "
                              "normal, not the old one")
    promote.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])

    rollback = sub.add_parser("rollback", help="return to the previous deployed version")
    rollback.add_argument("--reason", default="manual rollback")

    block = sub.add_parser("block", help="mark a version as never deployable")
    block.add_argument("--version", type=int, required=True)
    block.add_argument("--reason", required=True)

    sub.add_parser("status", help="what is deployed and what is registered")
    sub.add_parser("history", help="the deployment log")
    return parser.parse_args(argv)


def _metrics_from_accuracy(path, artefact_key: str) -> dict:
    """Pull the deployed variant's metrics out of an evaluate.py report."""
    with paths.resolve(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    variants = payload.get("variants", {})
    chosen = None
    for name, entry in variants.items():
        if name.endswith(artefact_key) or entry.get("precision") == artefact_key:
            chosen = (name, entry)
            break
    if chosen is None and variants:
        chosen = sorted(variants.items())[0]
    if chosen is None:
        return {}
    name, entry = chosen
    out = {"variant": name, "threshold": entry.get("threshold")}
    for split, block in entry.get("splits", {}).items():
        key = "drift" if "drift" in split else ("reference" if split == "test" else split)
        out[key] = {"image": block.get("image", {}), "boxes": block.get("boxes", {})}
    return out


def capture_reference(config: dict, data_config: dict, policy: DriftPolicy,
                      version: int, providers, deploy_root=None,
                      stream: str = None) -> dict:
    """Run the freshly deployed model over current conditions and freeze it.

    Two things have to be true of a new reference, and each was learned by
    getting it wrong.

    **It must come from the NEW model.**  Score distributions are not
    comparable across versions, so a reference inherited from the previous
    model makes the monitor fire on the deployment itself, every time.

    **It must come from the CURRENT conditions.**  After a drift-triggered
    redeployment the line is running under the drifted conditions -- that is
    what drift means -- and a reference captured under the conditions that no
    longer exist keeps the monitor firing on a model that has already adapted.
    ``stream`` is how the caller says what the new normal is; the retraining
    pipeline passes the post-drift stream.
    """
    frames = int(config.get("reference_frames", 400))
    stream_name = stream or config.get("canary", {}).get("stream", "clean")
    spec_config = dict(data_config.get("streams", {}).get(stream_name, {}))
    spec_config["frames"] = frames
    spec_config["seed"] = int(spec_config.get("seed", 991)) + 10_000 + version
    spec_config.pop("to", None)
    from edgeinspect.config import load_domains

    spec = StreamSpec.from_config("reference", spec_config, load_domains(data_config))
    camera = SyntheticCamera(spec, image_size=int(data_config.get("image_size", 256)))

    telemetry_path = paths.TELEMETRY / "reference_v{0}.jsonl".format(version)
    with EdgeAgent(deploy_root=deploy_root, providers=providers) as agent:
        outcome = agent.activate()
        if not outcome["activated"]:
            raise SystemExit("cannot capture a reference: " + outcome["detail"])
        runtime.run(agent.backend, camera, telemetry_path, model_version=version,
                    model_name=agent.manifest.get("model", "unregistered"),
                    stream_name="reference", record_ground_truth=False)

    records, _dropped = read_telemetry(telemetry_path)
    reference = build_reference(records, policy, meta={
        "model_version": version,
        "frames": len(records),
        "stream": "reference",
        "telemetry": paths.relative(telemetry_path),
    })
    reference_path = (paths.resolve(deploy_root) if deploy_root else paths.DEPLOY) / \
        "reference_v{0}.json".format(version)
    reference.save(reference_path)
    current = (paths.resolve(deploy_root) if deploy_root else paths.DEPLOY) / "reference.json"
    reference.save(current)
    return {"reference": paths.relative(reference_path), "frames": len(records)}


def run_canary(config: dict, data_config: dict, version: int, providers,
               deploy_root=None, stream: str = None) -> dict:
    """Short live run on the device, with pass or fail conditions that bite.

    Runs under the same conditions the reference was captured under, for the
    same reason: a canary against conditions the line is no longer in tests
    the wrong thing.
    """
    canary = dict(config.get("canary", {}))
    stream_name = stream or canary.get("stream", "clean")
    spec_config = dict(data_config.get("streams", {}).get(stream_name, {}))
    spec_config["frames"] = int(canary.get("frames", 300))
    spec_config["seed"] = int(spec_config.get("seed", 991)) + 20_000 + version
    spec_config.pop("to", None)
    from edgeinspect.config import load_domains

    spec = StreamSpec.from_config("canary", spec_config, load_domains(data_config))
    camera = SyntheticCamera(spec, image_size=int(data_config.get("image_size", 256)))

    telemetry_path = paths.TELEMETRY / "canary_v{0}.jsonl".format(version)
    with EdgeAgent(deploy_root=deploy_root, providers=providers) as agent:
        outcome = agent.activate()
        if not outcome["activated"]:
            return {"passed": False, "reason": outcome["detail"], "smoke": outcome["smoke"]}
        stats = runtime.run(agent.backend, camera, telemetry_path, model_version=version,
                            model_name=agent.manifest.get("model", "unregistered"),
                            stream_name="canary", record_ground_truth=False)

    summary = stats.to_dict()
    failures = []
    flagged_rate = summary["flagged_rate"]
    if flagged_rate > float(canary.get("max_flagged_rate", 1.0)):
        failures.append("flagged rate {0:.3f} exceeds {1}".format(
            flagged_rate, canary["max_flagged_rate"]))
    if flagged_rate < float(canary.get("min_flagged_rate", 0.0)):
        failures.append("flagged rate {0:.4f} is below {1}: the model has failed open"
                        .format(flagged_rate, canary["min_flagged_rate"]))
    p95 = summary["latency_ms"].get("p95", 0.0)
    if p95 > float(canary.get("max_latency_p95_ms", 1e9)):
        failures.append("p95 latency {0:.1f} ms exceeds {1} ms".format(
            p95, canary["max_latency_p95_ms"]))
    return {"passed": not failures, "reason": "; ".join(failures) or "ok",
            "stats": summary, "telemetry": paths.relative(telemetry_path)}


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    data_config = load_yaml(args.data_config)
    registry = Registry(root=paths.resolve(args.registry_root) if args.registry_root else None,
                        model_name=config.get("model_name", "surface-defect"))

    if args.command == "register":
        artefacts = {}
        precisions = {}
        for model in args.models:
            path = paths.resolve(model)
            if not path.exists():
                raise SystemExit("no artefact at " + paths.relative(path))
            stem = path.stem
            key = "fp32"
            for candidate in ("int8_percentile", "int8_entropy", "int8", "fp16", "fp32"):
                if stem.endswith(candidate):
                    key = candidate
                    break
            artefacts[key] = path
            precisions[key] = "int8" if key.startswith("int8") else key
        if args.checkpoint:
            artefacts["checkpoint"] = paths.resolve(args.checkpoint)
            precisions["checkpoint"] = "fp32"

        default_key = config.get("artefact", "int8")
        metrics = _metrics_from_accuracy(args.accuracy, default_key) if args.accuracy else {}
        dataset_meta = {}
        dataset_path = paths.resolve(args.dataset) / "dataset.json"
        if dataset_path.exists():
            with dataset_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            dataset_meta = {
                "root": paths.relative(paths.resolve(args.dataset)),
                "preset": raw.get("preset"),
                "seed": raw.get("seed"),
                "splits": {k: v.get("index_sha256") for k, v in raw.get("splits", {}).items()},
            }
        version = registry.register(artefacts, metrics, dataset_meta, parent=args.parent,
                                    reason=args.reason, mlflow_run_id=args.mlflow_run_id,
                                    precisions=precisions)
        print(json.dumps(version.to_dict(), indent=2))
        return 0

    if args.command == "promote":
        artefact_key = args.artefact or config.get("artefact", "int8")
        manifest = registry.deploy(args.version, artefact_key=artefact_key,
                                   deploy_root=args.deploy_root, reason=args.reason)
        print("deployed version {0} ({1}) from {2}".format(
            manifest["version"], manifest["precision"], manifest["artefact"]))

        result = {"manifest": manifest}
        if not args.no_reference:
            policy = DriftPolicy.from_dict(load_yaml(args.drift_config))
            result["reference"] = capture_reference(
                config, data_config, policy, args.version, args.providers,
                args.deploy_root, stream=args.stream)
            print("drift reference captured: {0} frames".format(
                result["reference"]["frames"]))

        if not args.no_canary:
            canary = run_canary(config, data_config, args.version, args.providers,
                                args.deploy_root, stream=args.stream)
            result["canary"] = canary
            if canary["passed"]:
                print("canary passed: flagged rate {0:.3f}, p95 {1:.1f} ms".format(
                    canary["stats"]["flagged_rate"], canary["stats"]["latency_ms"]["p95"]))
            else:
                print("CANARY FAILED: " + canary["reason"])
                rolled = registry.rollback(deploy_root=args.deploy_root,
                                           reason="canary failed: " + canary["reason"])
                registry.block(args.version, reason="canary failed: " + canary["reason"])
                result["rolled_back_to"] = rolled["version"]
                print("rolled back to version {0}".format(rolled["version"]))
                print(json.dumps(result, indent=2))
                return 1
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "rollback":
        manifest = registry.rollback(deploy_root=args.deploy_root, reason=args.reason)
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "block":
        version = registry.block(args.version, reason=args.reason)
        print(json.dumps(version.to_dict(), indent=2))
        return 0

    if args.command == "status":
        production = registry.production()
        print("model:      " + registry.model_name)
        print("production: " + (str(production.version) if production else "none"))
        print("\n{0:<9} {1:<11} {2:<21} {3}".format("version", "stage", "created", "reason"))
        for version in registry.versions():
            print("{0:<9} {1:<11} {2:<21} {3}".format(
                version.version, version.stage, version.created, version.reason[:48]))
        return 0

    if args.command == "history":
        for event in registry.events():
            print("{0}  {1:<9} v{2}{3}  {4}".format(
                event["ts"], event["action"], event.get("version"),
                (" <- v{0}".format(event["previous_version"])
                 if event.get("previous_version") is not None else ""),
                event.get("reason", "")))
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
