"""The run manifest stays small enough to be read.

The retraining pipeline carries the frames the device kept in its context, and
the manifest is rewritten after every one of its ten stages.  Serialising the
context wholesale produced a 571 KB file, written nine times, for about two
kilobytes of useful record.
"""

from __future__ import annotations

import json

from edgeinspect.pipeline import Pipeline, Stage


def test_large_context_values_are_summarised_not_dumped(tmp_path):
    context = {
        "run_id": "20260822-200612",
        "frames": [{"id": "f{0}".format(i), "image": "x" * 80} for i in range(400)],
        "small": [1, 2, 3],
    }
    pipeline = Pipeline("p", tmp_path, [Stage("a", lambda ctx: {"ok": True})],
                        echo=False, context=context)
    pipeline.run()

    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recorded = manifest["context"]

    assert recorded["run_id"] == "20260822-200612"
    assert recorded["small"] == [1, 2, 3], "a small value should survive intact"
    assert isinstance(recorded["frames"], str), "the big list should be summarised"
    assert "omitted" in recorded["frames"]
    assert "400" in recorded["frames"], "the summary should say how much was there"
    assert manifest_path.stat().st_size < 4096, (
        "the manifest is {0} bytes".format(manifest_path.stat().st_size))


def test_stage_outputs_are_still_recorded_in_full(tmp_path):
    """The summarising must not reach the thing the manifest is for."""
    pipeline = Pipeline("p", tmp_path, [Stage("a", lambda ctx: {"count": 7, "ok": True})],
                        echo=False)
    pipeline.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stages"][0]["outputs"] == {"count": 7, "ok": True}


def test_the_stages_key_is_not_duplicated_into_the_context(tmp_path):
    """Stage outputs live in `stages`, with timings; a second copy is noise."""
    pipeline = Pipeline("p", tmp_path, [Stage("a", lambda ctx: {"x": 1}),
                                        Stage("b", lambda ctx: {"y": 2}, needs=["a"])],
                        echo=False)
    pipeline.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "stages" not in manifest["context"]
    assert [s["name"] for s in manifest["stages"]] == ["a", "b"]


def test_an_unserialisable_value_does_not_break_the_manifest(tmp_path):
    """A manifest that cannot be written loses the record of a failed run."""
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    pipeline = Pipeline("p", tmp_path, [Stage("a", lambda ctx: None)],
                        echo=False, context={"thing": Opaque()})
    pipeline.run()
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["context"]["thing"] == "<opaque>"


def test_a_stage_writing_to_the_context_is_visible_to_the_caller(tmp_path):
    """The bug this guards against reported success as failure.

    The gate stage records its decision in the context and the caller reads it
    to choose an exit code.  When the runner copied the caller's dict, the
    stages still shared their own copy -- so the pipeline ran correctly, the
    model was promoted, and the caller exited non-zero because it was reading a
    dict nothing had written to.
    """
    context = {"gate_passed": None}

    def gate(ctx):
        ctx["gate_passed"] = True
        return {"decision": "promote"}

    pipeline = Pipeline("p", tmp_path, [Stage("gate", gate)], echo=False,
                        context=context)
    pipeline.run()

    assert pipeline.context["gate_passed"] is True
    assert context["gate_passed"] is True, (
        "the caller cannot see what the stage recorded")
    assert pipeline.context is context
