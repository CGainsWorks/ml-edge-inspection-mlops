"""Telemetry, the capture store, the DAG runner and the runtime boundary."""

from __future__ import annotations

import json

import numpy as np
import pytest

from edgeinspect import paths
from edgeinspect.edge import telemetry
from edgeinspect.pipeline import Pipeline, PipelineError, Stage


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

def a_record(frame_id=0, **overrides):
    payload = dict(
        frame_id=frame_id, stream="test", ts=frame_id / 30.0, model_version=1,
        model_name="surface-defect", precision="int8", backend="onnx",
        latency_ms=3.1, score=0.4, decision=0, n_boxes=0,
        features={"brightness": 0.5, "contrast": 0.16},
    )
    payload.update(overrides)
    return telemetry.TelemetryRecord(**payload)


def test_records_round_trip(tmp_path):
    path = tmp_path / "t.jsonl"
    with telemetry.TelemetryWriter(path) as writer:
        for index in range(10):
            writer.write(a_record(index, score=index / 10.0))
    records, dropped = telemetry.read(path)
    assert dropped == 0 and len(records) == 10
    assert records[3]["score"] == pytest.approx(0.3)
    assert records[0]["schema"] == telemetry.SCHEMA_VERSION


def test_a_truncated_last_line_is_dropped_not_fatal(tmp_path):
    """A device losing power mid-write must not take the monitoring stack down."""
    path = tmp_path / "t.jsonl"
    with telemetry.TelemetryWriter(path) as writer:
        for index in range(5):
            writer.write(a_record(index))
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"frame_id": 5, "sco')
    records, dropped = telemetry.read(path)
    assert len(records) == 5 and dropped == 1


def test_strict_reading_can_be_asked_for(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text('{"a": 1}\n{"b":\n', encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        telemetry.read(path, tolerate_partial=False)


def test_iter_records_streams(tmp_path):
    path = tmp_path / "t.jsonl"
    with telemetry.TelemetryWriter(path) as writer:
        for index in range(4):
            writer.write(a_record(index))
    assert [r["frame_id"] for r in telemetry.iter_records(path)] == [0, 1, 2, 3]


def test_to_arrays_looks_in_features_as_well_as_the_top_level(tmp_path):
    records = [a_record(i, score=i / 10.0).__dict__ for i in range(5)]
    arrays = telemetry.to_arrays(records, ["score", "brightness", "decision"])
    assert arrays["score"].tolist() == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert arrays["brightness"].tolist() == [0.5] * 5
    assert arrays["decision"].tolist() == [0.0] * 5


def test_to_arrays_returns_nan_for_a_signal_nobody_recorded():
    arrays = telemetry.to_arrays([a_record(0).__dict__], ["nonexistent"])
    assert np.isnan(arrays["nonexistent"]).all()


def test_missing_telemetry_says_where(tmp_path):
    with pytest.raises(FileNotFoundError, match="no telemetry"):
        telemetry.read(tmp_path / "absent.jsonl")


# ---------------------------------------------------------------------------
# Capture policy
# ---------------------------------------------------------------------------

def test_capture_policy_keeps_periodic_uncertain_and_flagged(tmp_path):
    store = telemetry.CaptureStore(tmp_path / "capture", every=10,
                                   uncertainty_band=0.1)
    assert store.should_keep(20, score=-5.0, threshold=0.0, decision=0) == "periodic"
    assert store.should_keep(21, score=0.05, threshold=0.0, decision=0) == "uncertain"
    assert store.should_keep(23, score=5.0, threshold=0.0, decision=1) == "flagged"
    assert store.should_keep(27, score=-5.0, threshold=0.0, decision=0) is None


def test_capture_respects_its_cap(tmp_path):
    store = telemetry.CaptureStore(tmp_path / "capture", every=1, max_frames=3)
    frame = np.zeros((1, 8, 8), dtype=np.float32)
    for index in range(10):
        reason = store.should_keep(index, 0.0, 0.0, 0)
        if reason:
            store.keep(index, frame, reason, "s", 0.0)
    assert store.saved == 3


def test_capture_writes_an_index_with_its_policy(tmp_path):
    store = telemetry.CaptureStore(tmp_path / "capture", every=2)
    frame = np.full((1, 8, 8), 0.5, dtype=np.float32)
    mask = np.zeros((1, 8, 8), dtype=np.float32)
    mask[0, 2:4, 2:4] = 1.0
    store.keep(0, frame, "periodic", "drifted", 0.3, ground_truth=1, mask=mask)
    index_path = store.flush()
    loaded = telemetry.CaptureStore.load(index_path.parent)
    assert loaded["total"] == 1
    assert loaded["policy"]["every"] == 2
    entry = loaded["frames"][0]
    assert entry["reason"] == "periodic" and entry["ground_truth"] == 1
    assert paths.resolve(entry["image"]).exists()
    assert paths.resolve(entry["mask"]).exists()


def test_capture_paths_are_repository_relative_and_posix(tmp_path):
    """Absolute Windows paths in a committed index break on Linux, silently."""
    store = telemetry.CaptureStore(tmp_path / "capture", every=1)
    stored = store.keep(0, np.zeros((1, 8, 8), dtype=np.float32), "periodic", "s", 0.0)
    assert "\\" not in stored


def test_missing_capture_index_says_where(tmp_path):
    with pytest.raises(FileNotFoundError, match="no capture index"):
        telemetry.CaptureStore.load(tmp_path / "nothing")


# ---------------------------------------------------------------------------
# The DAG runner
# ---------------------------------------------------------------------------

def test_stages_run_in_dependency_order(tmp_path):
    order = []
    stages = [
        Stage("c", lambda ctx: order.append("c"), needs=["b"]),
        Stage("a", lambda ctx: order.append("a")),
        Stage("b", lambda ctx: order.append("b"), needs=["a"]),
    ]
    Pipeline("p", tmp_path, stages, echo=False).run()
    assert order == ["a", "b", "c"]


def test_a_cycle_is_rejected(tmp_path):
    stages = [Stage("a", lambda ctx: None, needs=["b"]),
              Stage("b", lambda ctx: None, needs=["a"])]
    with pytest.raises(ValueError, match="cycle"):
        Pipeline("p", tmp_path, stages, echo=False).run()


def test_an_unknown_dependency_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown stage"):
        Pipeline("p", tmp_path, [Stage("a", lambda ctx: None, needs=["ghost"])],
                 echo=False)


def test_duplicate_stage_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="unique"):
        Pipeline("p", tmp_path, [Stage("a", lambda ctx: None),
                                 Stage("a", lambda ctx: None)], echo=False)


def test_outputs_are_visible_to_later_stages(tmp_path):
    def first(context):
        return {"value": 42}

    def second(context):
        assert context["stages"]["first"]["value"] == 42
        return {"seen": True}

    pipeline = Pipeline("p", tmp_path, [Stage("first", first),
                                        Stage("second", second, needs=["first"])],
                        echo=False)
    results = pipeline.run()
    assert results["second"].outputs == {"seen": True}


def test_a_failure_stops_the_run_and_is_recorded(tmp_path):
    reached = []

    def boom(context):
        raise RuntimeError("the GPU fell over")

    stages = [Stage("ok", lambda ctx: {"fine": True}),
              Stage("boom", boom, needs=["ok"]),
              Stage("after", lambda ctx: reached.append("after"), needs=["boom"])]
    pipeline = Pipeline("p", tmp_path, stages, echo=False)
    with pytest.raises(PipelineError) as error:
        pipeline.run()
    assert error.value.stage == "boom"
    assert reached == [], "a stage after the failure still ran"

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    statuses = {stage["name"]: stage["status"] for stage in manifest["stages"]}
    assert statuses["ok"] == "ok" and statuses["boom"] == "failed"
    assert "the GPU fell over" in dict(
        (s["name"], s["error"] or "") for s in manifest["stages"])["boom"]


def test_stages_can_be_skipped(tmp_path):
    ran = []
    stages = [Stage("a", lambda ctx: ran.append("a")),
              Stage("b", lambda ctx: ran.append("b"), needs=["a"])]
    pipeline = Pipeline("p", tmp_path, stages, echo=False)
    pipeline.run(skip=["b"])
    assert ran == ["a"]
    assert pipeline.results["b"].status == "skipped"


def test_manifest_is_written_even_for_a_clean_run(tmp_path):
    Pipeline("p", tmp_path, [Stage("a", lambda ctx: {"x": 1})], echo=False).run()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "ok"
    assert manifest["stages"][0]["outputs"] == {"x": 1}


# ---------------------------------------------------------------------------
# The runtime boundary
# ---------------------------------------------------------------------------

def test_the_edge_runtime_does_not_import_torch():
    """The runtime image is built without torch; a stray import breaks it silently.

    Run in a subprocess so that another test having imported torch cannot make
    this one pass.
    """
    import subprocess
    import sys
    import textwrap

    program = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, {src!r})
        import edgeinspect.edge.runtime          # noqa: F401
        import edgeinspect.edge.agent            # noqa: F401
        import edgeinspect.edge.telemetry        # noqa: F401
        import edgeinspect.monitoring            # noqa: F401
        import edgeinspect.registry              # noqa: F401
        import edgeinspect.models.postprocess    # noqa: F401
        import edgeinspect.models.preprocess     # noqa: F401
        leaked = sorted(m for m in sys.modules if m.split('.')[0] in
                        ('torch', 'torchvision', 'mlflow'))
        print(','.join(leaked))
        """
    ).format(src=str(paths.ROOT / "src"))
    completed = subprocess.run([sys.executable, "-c", program], capture_output=True,
                               text=True, check=True)
    leaked = completed.stdout.strip()
    assert leaked == "", "the edge runtime pulled in: " + leaked
