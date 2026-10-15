#!/usr/bin/env python
"""Run the drift monitor over collected telemetry.

    python scripts/detect_drift.py --telemetry artifacts/telemetry/drifted.jsonl
    python scripts/detect_drift.py --telemetry artifacts/telemetry/clean.jsonl \\
        --expect-no-trigger

``--expect-no-trigger`` is what makes the thresholds falsifiable.  The clean
control stream has the same imaging conditions as the reference and different
random draws, so a trigger on it is a false alarm and the command exits
non-zero.  CI runs both directions; a detector that has only ever been shown
drifting data is untested code.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json

from edgeinspect import paths
from edgeinspect.config import DriftPolicy, load_yaml
from edgeinspect.edge.telemetry import read as read_telemetry
from edgeinspect.monitoring import alerting
from edgeinspect.monitoring.drift import DriftMonitor, Reference


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--telemetry", required=True)
    parser.add_argument("--reference", default=None,
                        help="defaults to the reference captured at deployment")
    parser.add_argument("--config", default="configs/drift.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--stream", default=None)
    parser.add_argument("--onset", type=int, default=None,
                        help="known onset frame; inferred from configs/data.yaml if omitted")
    parser.add_argument("--out", default=None)
    parser.add_argument("--alerts", default=None)
    parser.add_argument("--action", default="retrain")
    parser.add_argument("--expect-no-trigger", action="store_true")
    parser.add_argument("--expect-trigger", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def infer_onset(stream: str, data_config: dict, frames: int):
    """The frame at which the configured stream starts to drift, scaled to the run.

    Telemetry from a shortened run has fewer frames than the config declares, so
    the onset is scaled by the same ratio ``simulate_edge.py`` used.
    """
    spec = data_config.get("streams", {}).get(stream)
    if not spec or spec.get("to") is None:
        return None
    declared = int(spec.get("frames", frames))
    scale = frames / max(declared, 1)
    return int(round(int(spec.get("ramp_start", 0)) * scale))


def main(argv=None) -> int:
    args = parse_args(argv)
    policy = DriftPolicy.from_dict(load_yaml(args.config))
    data_config = load_yaml(args.data_config)

    telemetry_path = paths.resolve(args.telemetry)
    records, dropped = read_telemetry(telemetry_path)
    if dropped:
        print("warning: dropped {0} unparseable telemetry line(s)".format(dropped))
    if not records:
        raise SystemExit("no telemetry in " + paths.relative(telemetry_path))

    stream = args.stream or records[0].get("stream", telemetry_path.stem)
    model_version = records[0].get("model_version")
    onset = args.onset if args.onset is not None else infer_onset(
        stream, data_config, len(records))

    reference_path = paths.resolve(args.reference) if args.reference else (
        paths.DEPLOY / "reference.json")
    reference = Reference.load(reference_path)

    monitor = DriftMonitor(policy, reference)
    report = monitor.scan(records, stream=stream, onset=onset, model_version=model_version)

    alerts_path = paths.resolve(args.alerts) if args.alerts else (
        paths.TELEMETRY / "alerts.jsonl")
    sink = alerting.AlertSink(alerts_path, echo=not args.quiet)
    emitted = sink.from_report(report, action=args.action)

    out_path = paths.resolve(args.out) if args.out else (
        paths.RESULTS / ("drift_" + stream + ".json"))
    report.save(out_path)

    summary = alerting.summarise(report)
    summary["alerts"] = len(emitted)
    summary["reference"] = paths.relative(reference_path)
    summary["reference_frames"] = reference.size
    summary["telemetry"] = paths.relative(telemetry_path)
    summary["report"] = paths.relative(out_path)
    print("\n" + json.dumps(summary, indent=2))

    if args.expect_no_trigger and report.triggered:
        print("\nFAILED: the monitor triggered on a stream that should not drift.  "
              "This is a false alarm and the thresholds in {0} are too tight."
              .format(args.config))
        return 1
    if args.expect_trigger and not report.triggered:
        print("\nFAILED: the monitor did not trigger on a stream that does drift.  "
              "The thresholds in {0} are too loose.".format(args.config))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
