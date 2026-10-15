#!/usr/bin/env python
"""Run the edge runtime over a production stream and collect telemetry.

    python scripts/simulate_edge.py --stream clean
    python scripts/simulate_edge.py --stream drifted --capture

Uses whatever the deployment manifest says is in production unless ``--model``
overrides it.  Writes one JSONL telemetry file per run and, with ``--capture``,
keeps a sampled subset of frames on the device for a future retraining run.

This is a simulation of a camera, not of the model: the inference, the
thresholding, the box extraction, the feature computation and the latency
measurement are the real code paths that would run on the device.  What is
simulated is the optics: the frames come from the generator rather than from a
sensor.  ``docs/limitations.md`` is explicit about which claims that supports.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

from edgeinspect import paths
from edgeinspect.config import load_domains, load_yaml
from edgeinspect.edge import runtime
from edgeinspect.edge.agent import EdgeAgent
from edgeinspect.edge.backends import load as load_backend
from edgeinspect.edge.source import StreamSpec, SyntheticCamera
from edgeinspect.edge.telemetry import CaptureStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stream", default="clean")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--model", default=None,
                        help="bypass the deployment manifest and run this artefact")
    parser.add_argument("--deploy-root", default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--providers", nargs="+", default=["CPUExecutionProvider"])
    parser.add_argument("--telemetry", default=None)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--capture-root", default=None)
    parser.add_argument("--capture-every", type=int, default=25)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def build_stream(name: str, config: dict, frames=None, seed=None) -> StreamSpec:
    streams = config.get("streams", {})
    if name not in streams:
        raise SystemExit("unknown stream {0!r}.  Known: {1}".format(name, sorted(streams)))
    spec = dict(streams[name])
    if frames is not None:
        spec["frames"] = int(frames)
        # A shortened stream must keep its onset inside the run, or the drifted
        # stream silently becomes a clean one and the demo proves nothing.
        if spec.get("to") is not None:
            original = int(streams[name]["frames"])
            scale = spec["frames"] / max(original, 1)
            spec["ramp_start"] = int(round(int(spec.get("ramp_start", 0)) * scale))
            spec["ramp_end"] = max(int(round(int(spec.get("ramp_end", 0)) * scale)),
                                   spec["ramp_start"] + 1)
    if seed is not None:
        spec["seed"] = int(seed)
    return StreamSpec.from_config(name, spec, load_domains(config))


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    spec = build_stream(args.stream, config, frames=args.frames, seed=args.seed)
    camera = SyntheticCamera(spec, image_size=int(config.get("image_size", 256)))

    telemetry_path = paths.resolve(
        args.telemetry or (paths.TELEMETRY / (args.stream + ".jsonl")))
    capture = None
    if args.capture:
        capture_root = paths.resolve(args.capture_root or (paths.CAPTURE / args.stream))
        capture = CaptureStore(capture_root, every=args.capture_every)

    agent = None
    backend = None
    version = 0
    model_name = "unregistered"
    try:
        if args.model:
            backend = load_backend(paths.resolve(args.model), backend="auto",
                                   providers=args.providers)
            model_name = backend.sidecar.get("name", "unregistered")
        else:
            agent = EdgeAgent(deploy_root=args.deploy_root, providers=args.providers)
            outcome = agent.activate()
            if not outcome["activated"]:
                print(json.dumps(outcome, indent=2))
                raise SystemExit("the deployed model failed its smoke check; nothing to run")
            backend = agent.backend
            version = int(agent.manifest.get("version", 0))
            model_name = agent.manifest.get("model", "unregistered")

        stats = runtime.run(
            backend, camera, telemetry_path, model_version=version,
            model_name=model_name, stream_name=args.stream, fps=args.fps,
            capture=capture, progress_every=0 if args.quiet else 300,
        )
    finally:
        if agent is not None:
            agent.close()
        elif backend is not None:
            backend.close()

    extra = {
        "stream": args.stream,
        "model_version": version,
        "model": model_name,
        "precision": backend.precision if backend else None,
        "onset_frame": spec.onset,
        "domain": spec.domain.name,
        "to_domain": spec.to_domain.name if spec.to_domain else None,
        "ramp": [spec.ramp_start, spec.ramp_end] if spec.to_domain else None,
        "defect_rate": spec.defect_rate,
    }
    out_path = paths.resolve(args.out or (paths.RESULTS / ("edge_" + args.stream + ".json")))
    runtime.write_stats(stats, out_path, extra=extra)

    summary = stats.to_dict()
    summary.update(extra)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
