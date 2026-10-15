"""The pieces meeting each other: runtime, backend, CLI and documentation checks.

These need the exported artefacts and skip without them.  They are the tests
that would catch a change which is individually correct in every module and
wrong where two of them join.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from edgeinspect import cli, paths
from edgeinspect.config import DriftPolicy, load_domains, load_drift_policy, load_yaml
from edgeinspect.edge import runtime
from edgeinspect.edge.backends import load as load_backend
from edgeinspect.edge.source import StreamSpec, SyntheticCamera
from edgeinspect.edge.telemetry import CaptureStore
from edgeinspect.edge.telemetry import read as read_telemetry
from edgeinspect.monitoring.drift import DriftMonitor, build_reference


# ---------------------------------------------------------------------------
# Streams
# ---------------------------------------------------------------------------

def test_a_ramped_stream_moves_between_its_domains(domains):
    spec = StreamSpec(name="s", domain=domains["reference"], frames=100,
                      to_domain=domains["lamp_ageing"], ramp_start=20, ramp_end=80)
    assert spec.domain_at(0).gain == pytest.approx(domains["reference"].gain)
    assert spec.domain_at(99).gain == pytest.approx(domains["lamp_ageing"].gain)
    middle = spec.domain_at(50).gain
    assert domains["lamp_ageing"].gain < middle < domains["reference"].gain
    assert spec.onset == 20


def test_a_fixed_stream_never_moves(domains):
    spec = StreamSpec(name="s", domain=domains["reference"], frames=50)
    assert spec.domain_at(0) is spec.domain_at(49)
    assert spec.onset is None


def test_stream_config_rejects_an_unknown_domain(domains):
    with pytest.raises(ValueError, match="unknown domain"):
        StreamSpec.from_config("s", {"domain": "moonlight", "frames": 10}, domains)


def test_the_shipped_streams_all_resolve(data_config, domains):
    for name, spec in data_config["streams"].items():
        stream = StreamSpec.from_config(name, spec, domains)
        assert stream.frames > 0
        if stream.to_domain is not None:
            assert stream.ramp_start < stream.ramp_end <= stream.frames


def test_clean_and_drifted_streams_contain_different_parts(data_config, domains):
    """Relighting the same parts would make the drift trivially separable."""
    clean = SyntheticCamera(StreamSpec.from_config("clean", data_config["streams"]["clean"],
                                                   domains))
    drifted = SyntheticCamera(StreamSpec.from_config(
        "drifted", data_config["streams"]["drifted"], domains))
    _id, clean_image, _m, _l, _d = clean.frame(0)
    _id, drift_image, _m, _l, _d = drifted.frame(0)
    assert not np.allclose(clean_image, drift_image)


def test_a_shortened_stream_keeps_its_onset_inside_the_run(data_config):
    """A truncated drifted stream that never drifts would make the demo prove nothing."""
    sys.path.insert(0, str(paths.ROOT / "scripts"))
    import simulate_edge

    spec = simulate_edge.build_stream("drifted", data_config, frames=420)
    assert spec.frames == 420
    assert spec.ramp_start < spec.ramp_end < spec.frames


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------

@pytest.mark.needs_model
@pytest.mark.slow
def test_the_runtime_produces_telemetry_the_monitor_can_consume(int8_model, tmp_path,
                                                                data_config, domains):
    spec = StreamSpec.from_config("clean", dict(data_config["streams"]["clean"],
                                                frames=60), domains)
    camera = SyntheticCamera(spec)
    backend = load_backend(int8_model, backend="onnx", providers=["CPUExecutionProvider"])
    capture = CaptureStore(tmp_path / "capture", every=20)
    try:
        stats = runtime.run(backend, camera, tmp_path / "t.jsonl", model_version=1,
                            model_name="test", stream_name="clean", capture=capture,
                            warmup=1)
    finally:
        backend.close()

    assert stats.frames == 60
    assert stats.captured >= 3
    records, dropped = read_telemetry(tmp_path / "t.jsonl")
    assert dropped == 0 and len(records) == 60

    first = records[0]
    assert first["model_version"] == 1
    assert first["precision"] == "int8"
    assert set(first["features"]) >= {"brightness", "sharpness", "score"}
    assert first["latency_ms"] > 0

    summary = stats.to_dict()
    assert summary["latency_ms"]["p95"] >= summary["latency_ms"]["p50"]
    assert summary["feature_ms"]["mean"] > 0, "telemetry cost is reported, not hidden"


@pytest.mark.needs_model
@pytest.mark.slow
def test_the_monitor_separates_the_control_stream_from_the_drifted_one(
        int8_model, tmp_path, data_config, domains):
    """The end-to-end version of the property the whole project rests on.

    Real model, real inference, real telemetry, real detectors.  The control
    stream must not trigger and the drifted stream must.
    """
    policy = DriftPolicy.from_dict(dict(load_yaml(paths.CONFIGS / "drift.yaml"),
                                        window=120, stride=60))
    backend = load_backend(int8_model, backend="onnx", providers=["CPUExecutionProvider"])

    def telemetry_for(name, frames, seed_offset=0):
        spec_config = dict(data_config["streams"][name], frames=frames)
        spec_config["seed"] = int(spec_config.get("seed", 991)) + seed_offset
        if spec_config.get("to") is not None:
            spec_config["ramp_start"] = frames // 6
            spec_config["ramp_end"] = frames // 2
        spec = StreamSpec.from_config(name, spec_config, domains)
        path = tmp_path / (name + str(seed_offset) + ".jsonl")
        runtime.run(backend, SyntheticCamera(spec), path, model_version=1,
                    model_name="test", stream_name=name, warmup=1)
        return read_telemetry(path)[0], spec

    try:
        reference_records, _ = telemetry_for("clean", 300, seed_offset=7000)
        control_records, _ = telemetry_for("clean", 360, seed_offset=8000)
        drifted_records, drifted_spec = telemetry_for("drifted", 480)
    finally:
        backend.close()

    reference = build_reference(reference_records, policy)
    monitor = DriftMonitor(policy, reference)

    control = monitor.scan(control_records, stream="clean")
    assert not control.triggered, (
        "false alarm on the control stream: " +
        "; ".join(w.reason for w in control.windows if w.triggered))

    drifted = monitor.scan(drifted_records, stream="drifted",
                           onset=drifted_spec.ramp_start)
    assert drifted.triggered, "the monitor slept through a real distribution shift"
    assert drifted.detection_latency() is not None


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

def test_cli_help_lists_the_commands(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for command in ("train", "quantise", "drift", "deploy", "retrain", "loop"):
        assert command in out


def test_cli_version(capsys):
    assert cli.main(["--version"]) == 0
    from edgeinspect import __version__

    assert capsys.readouterr().out.strip() == __version__


def test_cli_rejects_an_unknown_command(capsys):
    assert cli.main(["teleport"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_every_cli_command_points_at_a_file_that_exists():
    for command in list(cli.COMMANDS) + list(cli.PIPELINES):
        target = cli._resolve(command)
        assert target is not None and target.exists(), (
            "edgeinspect {0} points at {1}, which does not exist".format(command, target))


@pytest.mark.slow
def test_every_script_has_a_working_help():
    """A script whose argument parser is broken fails here, not in the runbook."""
    scripts = sorted(paths.ROOT.glob("scripts/*.py")) + \
        sorted(paths.ROOT.glob("pipelines/*.py"))
    for script in scripts:
        if script.name.startswith("_"):
            continue
        completed = subprocess.run([sys.executable, str(script), "--help"],
                                   capture_output=True, text=True, cwd=str(paths.ROOT),
                                   env=dict(__import__("os").environ, PYTHONPATH="src"))
        assert completed.returncode == 0, (
            "{0} --help exited {1}:\n{2}".format(script.name, completed.returncode,
                                                 completed.stderr[-1200:]))


# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

def test_the_shipped_drift_policy_and_the_deploy_gate_agree():
    """The gate metrics have to exist in what evaluate.py writes."""
    gate = load_yaml(paths.CONFIGS / "deploy.yaml")["gate"]
    known_splits = {"reference", "drift"}
    metrics = [gate["improvement"]["metric"], gate["regression"]["metric"]]
    metrics += [floor["metric"] for floor in gate["floors"]]
    for metric in metrics:
        split, rest = metric.split(".", 1)
        assert split in known_splits, "gate metric {0} names an unknown split".format(metric)
        assert rest.startswith("image.") or rest.startswith("boxes."), metric


def test_domains_used_by_streams_are_all_defined(data_config):
    domains = load_domains(data_config)
    for name, spec in data_config["streams"].items():
        assert spec["domain"] in domains, name
        if spec.get("to"):
            assert spec["to"] in domains, name


def test_the_control_stream_is_not_the_training_domain(data_config):
    """The control must be a different draw, or the comparison is circular."""
    assert data_config["streams"]["clean"]["domain"] == "reference_holdout"
    assert data_config["streams"]["clean"]["domain"] != "reference"


def test_published_results_referenced_by_the_makefile_are_the_ones_docs_use():
    sys.path.insert(0, str(paths.ROOT / "scripts"))
    import make_tables

    makefile = (paths.ROOT / "Makefile").read_text(encoding="utf-8")
    for source in set(make_tables.sources().values()):
        assert source in makefile, (
            "make_tables reads results/published/{0}, which `make publish` never copies"
            .format(source))


def test_drift_policy_is_loadable_through_its_helper():
    assert isinstance(load_drift_policy(), DriftPolicy)


@pytest.mark.needs_model
def test_published_accuracy_covers_every_variant_the_benchmark_expects():
    """A benchmark row with no accuracy row is a latency number nobody can price.

    Joined on the artefact path rather than on the variant name.  The name is a
    label and the path is the identity: this test originally joined on the name
    and failed, because the FP32 sidecar was named after its source checkpoint
    while its file was named after its precision.  Joining on the path would
    have caught that as the inconsistency it was rather than as a missing row.
    """
    accuracy_path = paths.PUBLISHED / "accuracy.json"
    if not accuracy_path.exists():
        pytest.skip("nothing published yet")
    with accuracy_path.open("r", encoding="utf-8") as handle:
        variants = json.load(handle)["variants"]
    evaluated = {entry.get("model") for entry in variants.values()}

    benchmark = load_yaml(paths.CONFIGS / "benchmark.yaml")
    for row in benchmark["variants"]:
        if not paths.resolve(row["model"]).exists():
            continue
        assert row["model"] in evaluated, (
            "{0} is benchmarked for latency but has no accuracy row".format(row["model"]))

    # And the name should still identify the artefact, because that is what the
    # generated tables show and what a reader joins the two tables on by eye.
    for name, entry in variants.items():
        if entry.get("backend") == "torch":
            continue
        assert entry["model"].endswith(name + ".onnx"), (
            "accuracy variant {0!r} points at {1}".format(name, entry["model"]))
