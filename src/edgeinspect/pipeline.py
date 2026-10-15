"""A very small DAG runner.

Airflow, Prefect, Dagster and Kubeflow all solve this, and all of them are the
wrong dependency for a repository whose point is that the loop can be run and
inspected on one machine.  What is actually needed from an orchestrator here is
four things:

* stages run in dependency order and a stage cannot start before its inputs
  exist;
* every stage is timed and its outputs recorded, so a run manifest exists
  afterwards without anybody having to write one;
* a failure stops the run and says which stage failed, with the state up to that
  point still on disk;
* the run can be resumed from a completed stage, because retraining includes a
  training step and re-running it to get past a typo in the deployment step is
  how people learn to run pipeline steps by hand instead.

That is about a hundred lines.  When this needs to run across machines, the
stage functions lift into Airflow tasks unchanged, because none of them knows
about this runner.
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import paths


@dataclass
class StageResult:
    name: str
    status: str                    # "ok" | "failed" | "skipped" | "cached"
    seconds: float
    outputs: Dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "seconds": round(self.seconds, 3),
            "outputs": self.outputs,
            "error": self.error,
        }


@dataclass
class Stage:
    name: str
    run: Callable
    needs: Sequence[str] = ()
    description: str = ""


class PipelineError(RuntimeError):
    """A stage failed.  Carries the stage name and the run directory."""

    def __init__(self, stage: str, run_dir: Path, original: BaseException):
        super().__init__("stage {0!r} failed: {1}".format(stage, original))
        self.stage = stage
        self.run_dir = run_dir
        self.original = original


class Pipeline:
    """Runs stages in dependency order and writes a manifest as it goes."""

    def __init__(self, name: str, run_dir: Path, stages: Sequence[Stage],
                 echo: bool = True, context: Optional[Dict] = None):
        self.name = name
        self.run_dir = Path(run_dir)
        self.stages = list(stages)
        self.echo = echo
        self.results: Dict[str, StageResult] = {}
        # The caller's dict, not a copy of it.
        #
        # The stages both read from this and write to it -- `gate` records its
        # decision here and `deploy` acts on it -- so it is shared state by
        # design.  Copying it was worse than it looks: the stages still shared
        # `self.context` with each other, so the pipeline ran correctly and only
        # the caller's own view was stale.  pipelines/retrain.py read
        # `gate_passed` from its original dict, never saw it, and exited 1 after
        # a run whose gate had passed and whose model was already deployed.  A
        # bug that reports success as failure is worse than one that crashes.
        self.context: Dict = context if context is not None else {}
        self._validate()

    def _validate(self) -> None:
        known = {stage.name for stage in self.stages}
        if len(known) != len(self.stages):
            raise ValueError("stage names must be unique")
        for stage in self.stages:
            missing = [n for n in stage.needs if n not in known]
            if missing:
                raise ValueError(
                    "stage {0!r} needs unknown stage(s) {1}".format(stage.name, missing))

    def order(self) -> List[Stage]:
        """Topological order, stable with respect to the declared order."""
        resolved: List[Stage] = []
        done: set = set()
        remaining = list(self.stages)
        while remaining:
            progressed = False
            for stage in list(remaining):
                if all(need in done for need in stage.needs):
                    resolved.append(stage)
                    done.add(stage.name)
                    remaining.remove(stage)
                    progressed = True
            if not progressed:
                raise ValueError(
                    "cycle in the pipeline among: {0}".format(
                        sorted(s.name for s in remaining))
                )
        return resolved

    # The manifest is read by people and by scripts/make_tables.py, and it is
    # rewritten after every stage.  Serialising the whole context put the
    # retraining run's frame lists in it and produced a 571 KB file -- rewritten
    # nine times -- for maybe two kilobytes of useful record.  Large values are
    # summarised instead of dropped silently, so the manifest still says what
    # was there.
    CONTEXT_VALUE_LIMIT = 2000

    def _serialisable_context(self) -> dict:
        out: Dict = {}
        for key, value in self.context.items():
            if key == "stages":
                continue          # already recorded, stage by stage, with timings
            if isinstance(value, (str, int, float, bool, type(None))):
                out[key] = value
                continue
            if isinstance(value, (list, tuple, dict)):
                try:
                    encoded = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    out[key] = "<unserialisable {0}>".format(type(value).__name__)
                    continue
                if len(encoded) <= self.CONTEXT_VALUE_LIMIT:
                    out[key] = value
                else:
                    out[key] = "<{0} of {1}, {2} bytes, omitted>".format(
                        type(value).__name__, len(value), len(encoded))
            else:
                out[key] = str(value)
        return out

    def _write_manifest(self, status: str) -> Path:
        paths.ensure(self.run_dir)
        manifest = {
            "pipeline": self.name,
            "run_dir": paths.relative(self.run_dir),
            "status": status,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "stages": [self.results[stage.name].to_dict()
                       for stage in self.stages if stage.name in self.results],
            "context": self._serialisable_context(),
        }
        target = self.run_dir / "manifest.json"
        with target.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, default=str)
        return target

    def run(self, only: Optional[Sequence[str]] = None,
            skip: Optional[Sequence[str]] = None) -> Dict[str, StageResult]:
        """Execute the pipeline.  Raises :class:`PipelineError` on the first failure."""
        skip = set(skip or ())
        only = set(only) if only else None
        for stage in self.order():
            if stage.name in skip or (only is not None and stage.name not in only):
                self.results[stage.name] = StageResult(stage.name, "skipped", 0.0)
                self._write_manifest("running")
                continue
            if self.echo:
                print("\n== {0} == {1}".format(stage.name, stage.description), flush=True)
            started = time.perf_counter()
            try:
                outputs = stage.run(self.context) or {}
            except BaseException as error:  # noqa: BLE001 - recorded, then re-raised
                self.results[stage.name] = StageResult(
                    stage.name, "failed", time.perf_counter() - started,
                    error=traceback.format_exc(limit=6))
                self._write_manifest("failed")
                if self.echo:
                    print("stage {0!r} FAILED: {1}".format(stage.name, error), flush=True)
                raise PipelineError(stage.name, self.run_dir, error) from error
            self.results[stage.name] = StageResult(
                stage.name, "ok", time.perf_counter() - started, outputs=outputs)
            self.context.setdefault("stages", {})[stage.name] = outputs
            self._write_manifest("running")
            if self.echo:
                print("   {0}: ok in {1:.1f}s".format(
                    stage.name, self.results[stage.name].seconds), flush=True)
        self._write_manifest("ok")
        return self.results

    def summary(self) -> dict:
        return {
            "pipeline": self.name,
            "run_dir": paths.relative(self.run_dir),
            "seconds": round(sum(r.seconds for r in self.results.values()), 2),
            "stages": {name: result.status for name, result in self.results.items()},
        }
