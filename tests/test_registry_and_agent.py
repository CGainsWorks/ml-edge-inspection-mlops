"""The registry, the deployment record and the edge agent.

These are the parts that decide what is running on a device, so the tests are
about the ways that goes wrong: a version being overwritten, two versions
claiming production at once, a rollback landing on the wrong model, a corrupted
artefact loading anyway, and a bad model being activated before it is checked.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from edgeinspect.edge import agent as agent_module
from edgeinspect.edge.backends.base import Backend
from edgeinspect.registry import Registry, load_manifest


@pytest.fixture
def artefact(tmp_path):
    """A file that stands in for an ONNX graph, with a sidecar."""
    def make(name: str, content: bytes = b"graph", threshold: float = -1.5):
        path = tmp_path / (name + ".onnx")
        path.write_bytes(content)
        with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump({"name": name, "precision": "int8",
                       "decision": {"threshold": threshold, "top_k": 8,
                                    "cell_threshold": 0.5},
                       "input": {"mean": 0.5, "scale": 2.0}}, handle)
        return path
    return make


@pytest.fixture
def registry(tmp_path):
    return Registry(root=tmp_path / "registry", model_name="test-model")


def register(registry, artefact, name="v", drift=0.8, reference=0.9, **kwargs):
    return registry.register(
        {"int8": artefact(name)},
        metrics={"reference": {"image": {"auroc": reference, "false_positive_rate": 0.01}},
                 "drift": {"image": {"auroc": drift}}},
        dataset={"preset": "tiny"}, **kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_versions_are_monotonic_and_start_in_staging(registry, artefact):
    first = register(registry, artefact, "a")
    second = register(registry, artefact, "b")
    assert (first.version, second.version) == (1, 2)
    assert first.stage == "staging"


def test_registering_never_overwrites(registry, artefact):
    register(registry, artefact, "a")
    register(registry, artefact, "b")
    assert [v.version for v in registry.versions()] == [1, 2]
    assert registry.get(1).artefacts["int8"].path.endswith("a.onnx")


def test_registry_survives_a_reload(registry, artefact, tmp_path):
    register(registry, artefact, "a")
    reloaded = Registry(root=tmp_path / "registry", model_name="test-model")
    assert reloaded.latest_version_number() == 1


def test_promotion_demotes_the_incumbent(registry, artefact, tmp_path):
    register(registry, artefact, "a")
    register(registry, artefact, "b")
    registry.deploy(1, deploy_root=tmp_path / "deploy", reason="first")
    registry.deploy(2, deploy_root=tmp_path / "deploy", reason="second")
    stages = {v.version: v.stage for v in registry.versions()}
    assert stages == {1: "archived", 2: "production"}
    assert sum(1 for v in registry.versions() if v.stage == "production") == 1


def test_deploy_writes_a_manifest_the_agent_can_read(registry, artefact, tmp_path):
    register(registry, artefact, "a")
    manifest = registry.deploy(1, deploy_root=tmp_path / "deploy", reason="first")
    on_disk = load_manifest(tmp_path / "deploy")
    assert on_disk == manifest
    assert on_disk["version"] == 1
    assert on_disk["threshold"] == -1.5
    assert len(on_disk["sha256"]) == 64


def test_deploy_appends_to_the_history(registry, artefact, tmp_path):
    register(registry, artefact, "a")
    register(registry, artefact, "b")
    registry.deploy(1, deploy_root=tmp_path / "deploy")
    registry.deploy(2, deploy_root=tmp_path / "deploy")
    lines = (tmp_path / "deploy" / "history.jsonl").read_text().strip().splitlines()
    assert [json.loads(line)["version"] for line in lines] == [1, 2]


def test_deploy_refuses_a_modified_artefact(registry, artefact, tmp_path):
    path = artefact("a")
    registry.register({"int8": path}, metrics={}, dataset={})
    path.write_bytes(b"a different graph entirely")
    with pytest.raises(ValueError, match="Refusing to deploy"):
        registry.deploy(1, deploy_root=tmp_path / "deploy")


def test_deploy_refuses_a_blocked_version(registry, artefact, tmp_path):
    register(registry, artefact, "a")
    registry.block(1, reason="failed the gate")
    with pytest.raises(ValueError, match="blocked"):
        registry.deploy(1, deploy_root=tmp_path / "deploy")


def test_blocked_versions_stay_in_the_registry(registry, artefact):
    register(registry, artefact, "a")
    registry.block(1, reason="regressed on the reference holdout")
    version = registry.get(1)
    assert version.stage == "blocked"
    assert "regressed" in version.stage_reason
    assert version.reason == "", "the registration reason must not be overwritten"


def test_rollback_returns_to_the_previous_deployment(registry, artefact, tmp_path):
    deploy_root = tmp_path / "deploy"
    register(registry, artefact, "a")
    register(registry, artefact, "b")
    registry.deploy(1, deploy_root=deploy_root, reason="first")
    registry.deploy(2, deploy_root=deploy_root, reason="second")
    manifest = registry.rollback(deploy_root=deploy_root)
    assert manifest["version"] == 1
    assert registry.production().version == 1


def test_rollback_skips_a_version_that_was_never_deployed(registry, artefact, tmp_path):
    """Rollback follows the deployment log, not version minus one."""
    deploy_root = tmp_path / "deploy"
    for name in ("a", "b", "c"):
        register(registry, artefact, name)
    registry.deploy(1, deploy_root=deploy_root, reason="first")
    registry.block(2, reason="gate failed, never deployed")
    registry.deploy(3, deploy_root=deploy_root, reason="third")
    assert registry.rollback(deploy_root=deploy_root)["version"] == 1


def test_rollback_with_no_history_says_so(registry, artefact, tmp_path):
    register(registry, artefact, "a")
    registry.deploy(1, deploy_root=tmp_path / "deploy")
    with pytest.raises(RuntimeError, match="nothing to roll back"):
        registry.rollback(deploy_root=tmp_path / "deploy")


def test_missing_artefact_key_lists_what_is_available(registry, artefact, tmp_path):
    register(registry, artefact, "a")
    with pytest.raises(KeyError, match="int8"):
        registry.deploy(1, artefact_key="fp16", deploy_root=tmp_path / "deploy")


def test_unknown_stage_is_rejected(registry, artefact):
    register(registry, artefact, "a")
    with pytest.raises(ValueError, match="stage must be"):
        registry.set_stage(1, "sort-of-live")


def test_load_manifest_without_a_deployment(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing deployed"):
        load_manifest(tmp_path / "empty")


# ---------------------------------------------------------------------------
# The agent's smoke check
# ---------------------------------------------------------------------------

class FakeBackend(Backend):
    """A backend whose behaviour the test chooses."""

    name = "fake"

    def __init__(self, behaviour="good"):
        super().__init__("memory://fake", {"decision": {"top_k": 8, "threshold": 0.0},
                                           "input": {"mean": 0.5, "scale": 2.0}})
        self.behaviour = behaviour
        self.closed = False

    def infer_raw(self, batch):
        batch = np.asarray(batch, dtype=np.float32)
        if self.behaviour == "raises":
            raise RuntimeError("device fell off the bus")
        if self.behaviour == "nan":
            return np.full((len(batch), 1, 32, 32), np.nan, dtype=np.float32)
        if self.behaviour == "wrong_shape":
            return np.zeros((len(batch), 3, 32, 32), dtype=np.float32)
        if self.behaviour == "constant":
            return np.zeros((len(batch), 1, 32, 32), dtype=np.float32)
        # A dark patch means a defect; a working model scores it higher.
        darkness = (0.5 - batch.mean(axis=(1, 2, 3))) * 40.0
        if self.behaviour == "inverted":
            darkness = -darkness
        out = np.zeros((len(batch), 1, 32, 32), dtype=np.float32)
        out += darkness[:, None, None, None]
        return out

    def close(self):
        self.closed = True


def test_smoke_check_passes_a_working_model():
    result = agent_module.smoke_check(FakeBackend("good"))
    assert result.passed and result.defect_score > result.clean_score


@pytest.mark.parametrize("behaviour,fragment", [
    ("inverted", "separation"),
    ("constant", "separation"),
    ("nan", "non-finite"),
    ("wrong_shape", "unexpected output shape"),
    ("raises", "inference raised"),
])
def test_smoke_check_catches_the_ways_a_model_loads_but_does_not_work(behaviour, fragment):
    """Loading proves the file is a graph.  It does not prove it is the model."""
    result = agent_module.smoke_check(FakeBackend(behaviour))
    assert not result.passed
    assert fragment in result.reason


def test_probe_frames_are_constructible_without_a_dataset():
    clean, defective = agent_module.probe_frames(size=64)
    assert clean.shape == defective.shape == (1, 1, 64, 64)
    assert defective.min() < clean.min(), "the defect probe must actually be darker"


# ---------------------------------------------------------------------------
# The agent
# ---------------------------------------------------------------------------

class FakeAgent(agent_module.EdgeAgent):
    """An agent whose backends are fakes, so activation can be tested offline."""

    def __init__(self, deploy_root, behaviours):
        super().__init__(deploy_root=deploy_root)
        self.behaviours = list(behaviours)
        self.opened = []

    def verify(self, manifest):
        return None                      # hashes are covered by the registry tests

    def _open(self, manifest):
        backend = FakeBackend(self.behaviours.pop(0))
        self.opened.append(backend)
        return backend


def write_manifest(path, version):
    path.mkdir(parents=True, exist_ok=True)
    with (path / "current.json").open("w", encoding="utf-8") as handle:
        json.dump({"version": version, "artefact": "models/x.onnx", "sha256": "0" * 64,
                   "precision": "int8", "model": "test"}, handle)


def test_activation_swaps_the_backend(tmp_path):
    write_manifest(tmp_path, 1)
    agent = FakeAgent(tmp_path, ["good"])
    outcome = agent.activate()
    assert outcome["activated"] and agent.version == 1


def test_a_failing_model_does_not_displace_the_running_one(tmp_path):
    """The rule the whole agent exists for."""
    write_manifest(tmp_path, 1)
    agent = FakeAgent(tmp_path, ["good", "inverted"])
    agent.activate()
    running = agent.backend

    write_manifest(tmp_path, 2)
    outcome = agent.activate()
    assert outcome["activated"] is False
    assert outcome["kept_version"] == 1
    assert agent.version == 1
    assert agent.backend is running, "the working backend was replaced by a failing one"
    assert agent.opened[-1].closed, "the rejected backend was left open"


def test_poll_is_a_no_op_when_the_version_has_not_changed(tmp_path):
    write_manifest(tmp_path, 1)
    agent = FakeAgent(tmp_path, ["good"])
    agent.activate()
    assert agent.poll() is None


def test_poll_activates_a_new_version(tmp_path):
    write_manifest(tmp_path, 1)
    agent = FakeAgent(tmp_path, ["good", "good"])
    agent.activate()
    write_manifest(tmp_path, 2)
    outcome = agent.poll()
    assert outcome is not None and outcome["activated"] and agent.version == 2


def test_missing_manifest_raises_rather_than_silently_doing_nothing(tmp_path):
    agent = FakeAgent(tmp_path / "empty", ["good"])
    with pytest.raises(FileNotFoundError, match="nothing deployed"):
        agent.activate()
