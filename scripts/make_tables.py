#!/usr/bin/env python
"""Generate the Markdown tables that the documentation embeds.

    python scripts/make_tables.py            # write docs/tables/*.md
    python scripts/make_tables.py --stdout   # print them instead

Every table is built from a file in ``results/published/``.  Nothing here
invents, rounds differently from, or reformats a number that is not in one of
those files, which is what lets ``scripts/check_docs.py`` verify the whole
documentation set by regenerating it.

Publishing a result is a deliberate act: ``make publish`` copies ``results/*.json``
into ``results/published/`` and re-runs this.  Until that happens the documented
numbers stay pinned to the last published run, and a local experiment cannot
quietly rewrite the README.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Optional

from edgeinspect import paths

TABLES: Dict[str, Callable] = {}


def table(name: str, source: str):
    def register(function):
        function.source = source
        TABLES[name] = function
        return function
    return register


def load(source: str) -> Optional[dict]:
    path = paths.PUBLISHED / source
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _rows(header, rows, align: str = "") -> str:
    """A Markdown table.

    ``align`` is one character per column: ``l`` for left, ``r`` for right.
    Columns beyond the string default to right.  Worth the parameter rather
    than right-aligning everything: a right-aligned column of words reads as a
    mistake, and these tables are the first thing anybody sees in the README.
    """
    rules = []
    for index in range(len(header)):
        kind = align[index] if index < len(align) else "r"
        rules.append("---" if kind == "l" else "--:")
    out = ["|" + "|".join(str(cell) for cell in header) + "|",
           "|" + "|".join(rules) + "|"]
    for row in rows:
        out.append("|" + "|".join(str(cell) for cell in row) + "|")
    return "\n".join(out)


def _f(value, digits=4, dash="--"):
    if value is None:
        return dash
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number != number:
        return dash
    return "{0:.{1}f}".format(number, digits)


@table("accuracy", "accuracy.json")
def accuracy_table(payload: dict) -> str:
    header = ["variant", "precision", "test AUROC", "test recall", "test FPR",
              "drift AUROC", "drift recall", "bytes"]
    rows = []
    for name, entry in payload["variants"].items():
        test = entry["splits"].get("test", {}).get("image", {})
        drift = entry["splits"].get("test_drift", {}).get("image", {})
        rows.append([
            "`" + name + "`",
            entry.get("precision", "?"),
            _f(test.get("auroc")),
            _f(test.get("recall")),
            _f(test.get("false_positive_rate"), 4),
            _f(drift.get("auroc")),
            _f(drift.get("recall")),
            "{0:,}".format(entry["bytes"]) if entry.get("bytes") else "--",
        ])
    note = ("\nOperating point: threshold chosen on the validation split at a "
            "{0:.0%} target false positive rate, then applied unchanged to both "
            "test splits.  `drift` is the `lamp_ageing` holdout.".format(
                payload.get("target_fpr", 0.02)))
    return _rows(header, rows, align="llrrrrrr") + "\n" + note


@table("latency", "latency.json")
def latency_table(payload: dict) -> str:
    header = ["variant", "precision", "batch", "mean ms", "p95 ms", "fps", "bytes"]
    rows = []
    for row in payload["rows"]:
        rows.append([
            "`" + row["variant"] + "`",
            row["precision"],
            row["batch_size"],
            _f(row["latency_ms"]["mean"], 3),
            _f(row["latency_ms"]["p95"], 3),
            _f(row["throughput_fps"], 1),
            "{0:,}".format(row["bytes"]),
        ])
    host = payload.get("host", {})
    note = ("\nMeasured on {0}, {1} threads pinned, {2} iterations after {3} warm-up "
            "calls.  **Not Jetson numbers** -- no Jetson was available; see "
            "[jetson.md](../jetson.md).".format(
                host.get("platform", "unknown"), payload.get("intra_op_threads"),
                payload.get("iterations"), payload.get("warmup")))
    if payload.get("features_ms"):
        note += ("\n\nTelemetry feature extraction costs {0} ms per frame (p95 {1} ms) "
                 "on the same host, reported separately from inference rather than "
                 "folded into it.".format(_f(payload["features_ms"]["mean"], 3),
                                          _f(payload["features_ms"]["p95"], 3)))
    return _rows(header, rows, align="llrrrrr") + "\n" + note


@table("quantisation", "quantisation_ablation.json")
def quantisation_table(payload: dict) -> str:
    header = ["model input", "quantisation", "AUROC", "vs FP32"]
    rows = []
    labels = {
        "centred": "`[-1, 1]` (deployed)",
        "uncentred": "`[0, 1]`",
    }
    for model_name, entry in payload["models"].items():
        label = labels.get(model_name, model_name)
        rows.append([label, "FP32", _f(entry["fp32_auroc"]), "--"])
        for configuration, variant in entry["variants"].items():
            rows.append([label, "INT8, " + configuration.replace("_", " "),
                         _f(variant["auroc"]),
                         "{0:+.4f}".format(variant["delta_vs_fp32"])])
    return _rows(header, rows, align="llrr") + "\n\n" + payload.get("conclusion", "")


@table("seed_sweep", "retrain_seed_sweep.json")
def seed_sweep_table(payload: dict) -> str:
    """How reproducible the retraining step is, across seeds."""
    header = ["seed", "reference holdout", "drifted holdout", "gate",
              "failed check"]
    rows = []
    for run in payload["runs"]:
        if run.get("reference_auroc") is None:
            rows.append([run["seed"], "--", "--", "crash",
                         str(run.get("error", ""))[:40]])
            continue
        rows.append([
            run["seed"],
            _f(run["reference_auroc"]),
            _f(run["drift_auroc"]),
            (run.get("decision") or "?"),
            ", ".join("`" + m + "`" for m in run.get("failed_checks", []))
            or "--",
        ])
    incumbent = payload.get("incumbent", {})
    note = "Against an incumbent at {0} reference and {1} drifted.  {2}".format(
        _f(incumbent.get("reference_auroc")), _f(incumbent.get("drift_auroc")),
        payload.get("summary", {}).get("sentence", ""))
    note += (
        "  Same captured frames every time; only the training seed changes.  "
        "Reproduce with `python scripts/retrain_seed_sweep.py` after any "
        "drifted stream.")
    return _rows(header, rows, align="lrrll") + chr(10) + note


@table("drift", "loop.json")
def drift_table(payload: dict) -> str:
    steps = {entry["step"]: entry for entry in payload["steps"]}
    clean = steps.get(2, {})
    drifted = steps.get(3, {})
    header = ["stream", "frames", "windows", "triggered", "false alarms",
              "onset", "first trigger", "detection latency"]
    rows = [
        ["`clean` (control)", clean.get("frames", "--"), clean.get("windows", "--"),
         str(clean.get("triggered")), clean.get("false_alarms", "--"), "--", "--", "--"],
        ["`drifted`", drifted.get("frames", "--"), "--", str(drifted.get("triggered")),
         "--", drifted.get("onset_frame", "--"),
         drifted.get("first_trigger_frame", "--"),
         "{0} frames".format(drifted.get("detection_latency_frames"))
         if drifted.get("detection_latency_frames") is not None else "--"],
    ]
    note = ("\nThe control row is the important one: same imaging conditions as the "
            "reference window, different random draws.  A monitor that fires there is "
            "a monitor nobody will leave switched on.")
    sequential = drifted.get("sequential_earliest_frame")
    if sequential is not None:
        note += ("\n\nThe sequential Page-Hinkley detector first accumulated past its "
                 "threshold at frame {0}, against a windowed first trigger at frame {1}."
                 .format(sequential, drifted.get("first_trigger_frame")))
    return _rows(header, rows, align="lrrlrrrr") + "\n" + note


@table("recovery", "loop.json")
def recovery_table(payload: dict) -> str:
    steps = {entry["step"]: entry for entry in payload["steps"]}
    before = steps.get(4, {})
    retrain = steps.get(5, {})
    after = steps.get(6, {})
    rollback = steps.get(7, {})
    header = ["stage", "outcome"]
    rows = [
        ["baseline deployed", "v{0}".format(steps.get(1, {}).get("version", "?"))],
        ["drift detected at frame", drifted_frame(steps)],
        ["frames captured on the device", steps.get(3, {}).get("captured", "--")],
        ["**flagged rate, post-drift, incumbent**", _f(before.get("flagged_rate"), 4)],
        ["retrained candidate: reference-holdout AUROC",
         _f(retrain.get("candidate_reference_auroc"))],
        ["retrained candidate: drifted-holdout AUROC",
         _f(retrain.get("candidate_drift_auroc"))],
        ["retraining gate", "**{0}**".format(retrain.get("gate", "--"))],
        ["deployed after retraining", "v{0}".format(retrain.get("production_version", "?"))],
        ["**flagged rate, post-drift, retrained**", _f(after.get("flagged_rate_after"), 4)],
        ["monitor quiet under the new normal", str(after.get("monitor_quiet"))],
    ]
    if rollback:
        rows.append(["rolled back to", "v{0}".format(rollback.get("rolled_back_to", "?"))])
        rows.append(["device confirmed serving",
                     "v{0}".format(rollback.get("device_is_serving", "?"))])
    rows.append(["whole loop", "{0} s".format(_f(payload.get("seconds"), 1))])
    note = ("\nThe two flagged rates are the same 600-frame post-drift stream through "
            "two models, so they are a comparison rather than an anecdote.  The true "
            "defect rate on that stream is 8%.  The monitor is quiet afterwards because "
            "the reference window was recaptured under the post-drift conditions when "
            "the new version was promoted; capturing it under the old conditions would "
            "leave it firing forever on a model that has already adapted.")
    return _rows(header, rows, align="ll") + "\n" + note


def drifted_frame(steps: dict) -> str:
    value = steps.get(3, {}).get("first_trigger_frame")
    return str(value) if value is not None else "--"


@table("gate", "loop.json")
def gate_table(payload: dict) -> str:
    steps = {entry["step"]: entry for entry in payload["steps"]}
    checks = steps.get(5, {}).get("gate_checks", [])
    if not checks:
        return "_No gate record in the published loop run._"
    header = ["check", "metric", "candidate", "incumbent", "requirement", "result"]
    rows = []
    for check in checks:
        requirement = check.get("required", check.get("allowed",
                                                      check.get("min", check.get("max"))))
        rows.append([
            check["kind"],
            "`" + check["metric"] + "`",
            _f(check.get("candidate")),
            _f(check.get("incumbent")),
            _f(requirement, 3),
            "PASS" if check["passed"] else "**FAIL**",
        ])
    return _rows(header, rows, align="llrrrl")


def build(name: str) -> Optional[str]:
    function = TABLES[name]
    payload = load(function.source)
    if payload is None:
        return None
    return function(payload)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="docs/tables")
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--only", nargs="+", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = paths.resolve(args.out_dir)
    paths.ensure(out_dir)
    names = args.only or sorted(TABLES)
    written, skipped = [], []
    for name in names:
        content = build(name)
        if content is None:
            skipped.append((name, TABLES[name].source))
            continue
        if args.stdout:
            print("\n### " + name + "\n")
            print(content)
        else:
            target = out_dir / (name + ".md")
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content.rstrip() + "\n")
            written.append(paths.relative(target))
    if not args.quiet:
        for path in written:
            print("wrote " + path)
        for name, source in skipped:
            print("skipped {0}: results/published/{1} does not exist".format(name, source))
    return 0


def table_names():
    return sorted(TABLES)


def sources() -> Dict[str, str]:
    return {name: function.source for name, function in TABLES.items()}


def published_files() -> Dict[str, Path]:
    return {source: paths.PUBLISHED / source for source in set(sources().values())}


if __name__ == "__main__":
    raise SystemExit(main())
