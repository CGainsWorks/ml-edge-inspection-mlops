"""The model registry and the deployment record.

A JSON file, not a service.  The whole registry is one file that a person can
read, diff and restore from a backup, which for a fleet of edge devices running
one model is the right size of tool.  The API here is what a real registry would
expose, so swapping the storage for MLflow's Model Registry or S3 is a change to
this module and to nothing above it.

The rules this enforces, and why each exists
--------------------------------------------
**Versions are immutable and monotonic.**  Re-registering never overwrites.  A
metric attached to "version 3" has to keep meaning the same thing a month later.

**Exactly one version is in ``production`` at a time.**  Promotion demotes the
incumbent to ``archived`` in the same write, so there is no window in which two
versions both claim to be live.

**Every artefact is hashed at registration and verified at deployment.**  The
failure this prevents is real and quiet: a partially copied ONNX file loads,
runs, and produces plausible nonsense.

**Deployment events are append-only.**  Rollback adds an event; it does not
delete the one that made the mistake.  The record of what was live at 14:32 is
the only thing that makes a production incident investigable.

**Rollback targets the previous production version, not "version minus one".**
Those differ as soon as a promotion has ever been skipped or a version blocked.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .. import paths
from ..artefacts import sha256_file

STAGES = ("staging", "production", "archived", "blocked")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Artefact:
    """One deployable file and the hash it must still have when loaded."""

    kind: str            # "onnx" | "engine" | "checkpoint"
    precision: str
    path: str
    sha256: str
    bytes: int

    @classmethod
    def of(cls, path: Path, kind: str, precision: str) -> "Artefact":
        resolved = paths.resolve(path)
        return cls(kind=kind, precision=precision, path=paths.relative(resolved),
                   sha256=sha256_file(resolved), bytes=int(resolved.stat().st_size))

    def verify(self) -> None:
        resolved = paths.resolve(self.path)
        if not resolved.exists():
            raise FileNotFoundError("registered artefact is missing: " + self.path)
        actual = sha256_file(resolved)
        if actual != self.sha256:
            raise ValueError(
                "artefact {0} has hash {1}, registry says {2}.  Refusing to deploy a "
                "file that is not the one that was evaluated.".format(
                    self.path, actual[:12], self.sha256[:12])
            )


@dataclass
class Version:
    """One registered model version."""

    version: int
    name: str
    created: str
    stage: str
    artefacts: Dict[str, Artefact] = field(default_factory=dict)
    metrics: Dict = field(default_factory=dict)
    dataset: Dict = field(default_factory=dict)
    parent: Optional[int] = None
    reason: str = ""          # why this version was created
    stage_reason: str = ""    # why it is in the stage it is in
    mlflow_run_id: Optional[str] = None
    notes: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["artefacts"] = {k: asdict(v) for k, v in self.artefacts.items()}
        return payload

    @classmethod
    def from_dict(cls, data: dict) -> "Version":
        artefacts = {k: Artefact(**v) for k, v in data.get("artefacts", {}).items()}
        return cls(
            version=int(data["version"]),
            name=data["name"],
            created=data["created"],
            stage=data["stage"],
            artefacts=artefacts,
            metrics=data.get("metrics", {}),
            dataset=data.get("dataset", {}),
            parent=data.get("parent"),
            reason=data.get("reason", ""),
            stage_reason=data.get("stage_reason", ""),
            mlflow_run_id=data.get("mlflow_run_id"),
            notes=data.get("notes", {}),
        )


class Registry:
    """Versions, stages and the deployment log, backed by one JSON file."""

    def __init__(self, root: Optional[Path] = None, model_name: str = "surface-defect"):
        self.root = Path(root) if root else paths.REGISTRY
        self.model_name = model_name
        self.path = self.root / "registry.json"
        self._state = self._load()

    # -- storage ---------------------------------------------------------

    def _load(self) -> dict:
        if not self.path.exists():
            return {"model": self.model_name, "versions": [], "events": [],
                    "schema": 1}
        with self.path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _save(self) -> None:
        """Atomic write.  A half-written registry is a fleet that cannot deploy."""
        paths.ensure(self.root)
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                             dir=str(self.root), suffix=".tmp")
        try:
            json.dump(self._state, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, self.path)

    # -- queries ---------------------------------------------------------

    def versions(self) -> List[Version]:
        return [Version.from_dict(v) for v in self._state.get("versions", [])]

    def get(self, version: int) -> Version:
        for item in self._state.get("versions", []):
            if int(item["version"]) == int(version):
                return Version.from_dict(item)
        raise KeyError("no version {0} in the registry".format(version))

    def latest_version_number(self) -> int:
        numbers = [int(v["version"]) for v in self._state.get("versions", [])]
        return max(numbers) if numbers else 0

    def production(self) -> Optional[Version]:
        for item in self._state.get("versions", []):
            if item["stage"] == "production":
                return Version.from_dict(item)
        return None

    def events(self) -> List[dict]:
        return list(self._state.get("events", []))

    def previous_production(self) -> Optional[Version]:
        """The version that was live before the current one.

        Read off the deployment log rather than inferred from version numbers,
        because a blocked or skipped version breaks the arithmetic and a
        rollback that lands on the wrong model is worse than no rollback.
        """
        current = self.production()
        for event in reversed(self._state.get("events", [])):
            if event.get("action") not in ("deploy", "rollback"):
                continue
            previous = event.get("previous_version")
            if previous is None:
                continue
            if current is not None and int(event.get("version", -1)) != current.version:
                continue
            try:
                candidate = self.get(int(previous))
            except KeyError:
                continue
            if candidate.stage != "blocked":
                return candidate
        return None

    # -- mutations -------------------------------------------------------

    def register(self, artefacts: Dict[str, Path], metrics: Dict, dataset: Dict,
                 parent: Optional[int] = None, reason: str = "",
                 mlflow_run_id: Optional[str] = None,
                 notes: Optional[Dict] = None,
                 precisions: Optional[Dict[str, str]] = None) -> Version:
        """Register a new version in ``staging``.  Returns it."""
        number = self.latest_version_number() + 1
        precisions = precisions or {}
        registered: Dict[str, Artefact] = {}
        for key, path in artefacts.items():
            kind = "engine" if str(path).endswith(".engine") else (
                "checkpoint" if str(path).endswith(".pt") else "onnx")
            registered[key] = Artefact.of(Path(path), kind=kind,
                                          precision=precisions.get(key, key))
        version = Version(
            version=number,
            name=self.model_name,
            created=_now(),
            stage="staging",
            artefacts=registered,
            metrics=metrics,
            dataset=dataset,
            parent=parent,
            reason=reason,
            mlflow_run_id=mlflow_run_id,
            notes=notes or {},
        )
        self._state.setdefault("versions", []).append(version.to_dict())
        self._append_event("register", number, None, reason)
        self._save()
        return version

    def set_stage(self, version: int, stage: str, reason: str = "") -> Version:
        if stage not in STAGES:
            raise ValueError("stage must be one of {0}, got {1!r}".format(STAGES, stage))
        target = self.get(version)
        previous = self.production()
        for item in self._state["versions"]:
            if stage == "production" and item["stage"] == "production":
                item["stage"] = "archived"
            if int(item["version"]) == int(version):
                item["stage"] = stage
                # The reason has to live on the version, not only in the event
                # log: it is what `deploy` quotes when it refuses a blocked
                # version, and an empty refusal message is a support ticket.
                item["stage_reason"] = reason
        action = "promote" if stage == "production" else "stage"
        self._append_event(action, version,
                           previous.version if previous else None, reason, stage=stage)
        self._save()
        target.stage = stage
        target.stage_reason = reason
        return target

    def block(self, version: int, reason: str) -> Version:
        """Mark a version as never deployable.

        Used by the retraining gate.  A model that failed its gate stays in the
        registry -- deleting it loses the record that it was tried and why it
        was rejected, which is exactly what somebody re-running the same
        experiment in six months needs.
        """
        return self.set_stage(version, "blocked", reason=reason)

    def _append_event(self, action: str, version: Optional[int],
                      previous_version: Optional[int], reason: str, **extra) -> None:
        event = {
            "ts": _now(),
            "action": action,
            "version": version,
            "previous_version": previous_version,
            "reason": reason,
        }
        event.update(extra)
        self._state.setdefault("events", []).append(event)

    # -- deployment ------------------------------------------------------

    def deploy(self, version: int, artefact_key: str = "int8",
               deploy_root: Optional[Path] = None, reason: str = "",
               action: str = "deploy") -> dict:
        """Promote a version to production and write the deployment manifest.

        The manifest is what the edge agent reads.  It is written atomically and
        it carries the artefact hash, so a device that picks up a half-copied
        file refuses it rather than serving it.
        """
        target = self.get(version)
        if target.stage == "blocked":
            raise ValueError(
                "version {0} is blocked ({1}); it failed its gate and cannot be deployed."
                .format(version, target.stage_reason or target.reason or "no reason recorded")
            )
        if artefact_key not in target.artefacts:
            raise KeyError(
                "version {0} has no artefact {1!r}.  Available: {2}".format(
                    version, artefact_key, sorted(target.artefacts))
            )
        artefact = target.artefacts[artefact_key]
        artefact.verify()

        previous = self.production()
        self.set_stage(version, "production", reason=reason)

        sidecar_path = paths.resolve(artefact.path).with_suffix(".json")
        threshold = None
        if sidecar_path.exists():
            with sidecar_path.open("r", encoding="utf-8") as handle:
                threshold = json.load(handle).get("decision", {}).get("threshold")

        manifest = {
            "model": self.model_name,
            "version": int(version),
            "artefact_key": artefact_key,
            "artefact": artefact.path,
            "sha256": artefact.sha256,
            "precision": artefact.precision,
            "threshold": threshold,
            "deployed_at": _now(),
            "previous_version": previous.version if previous else None,
            "reason": reason,
            "metrics": target.metrics,
        }
        root = Path(deploy_root) if deploy_root else paths.DEPLOY
        paths.ensure(root)
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                             dir=str(root), suffix=".tmp")
        try:
            json.dump(manifest, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, root / "current.json")

        with (root / "history.jsonl").open("a", encoding="utf-8") as history:
            history.write(json.dumps(manifest, separators=(",", ":")) + "\n")

        self._append_event(action, version, previous.version if previous else None,
                           reason, artefact=artefact.path)
        self._save()
        return manifest

    def rollback(self, deploy_root: Optional[Path] = None,
                 reason: str = "rollback") -> dict:
        """Return production to the previously deployed version."""
        target = self.previous_production()
        if target is None:
            raise RuntimeError(
                "nothing to roll back to: no earlier deployment is recorded.  "
                "Check artifacts/deploy/history.jsonl."
            )
        current = self.production()
        if current is not None:
            self.set_stage(current.version, "archived", reason=reason)
        key = "int8" if "int8" in target.artefacts else sorted(target.artefacts)[0]
        return self.deploy(target.version, artefact_key=key, deploy_root=deploy_root,
                           reason=reason, action="rollback")


def load_manifest(deploy_root: Optional[Path] = None) -> dict:
    """Read the current deployment manifest.  Raises if nothing is deployed."""
    root = Path(deploy_root) if deploy_root else paths.DEPLOY
    manifest_path = root / "current.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "nothing deployed: no manifest at {0}.  Run `make deploy`.".format(
                paths.relative(manifest_path))
        )
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
