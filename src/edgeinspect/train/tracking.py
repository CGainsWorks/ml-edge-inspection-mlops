"""MLflow, wrapped thinly.

Two things this wrapper is for.

**A serverless local backend by default.**  ``make train`` must work on a laptop
with nothing running, so the tracking URI defaults to a SQLite database under
``mlruns/``.  Setting ``MLFLOW_TRACKING_URI`` to a real server changes nothing
else.

SQLite rather than the older ``file:./mlruns`` store, and the reason is worth
recording: as of MLflow 3.x the filesystem store is in maintenance mode and
``mlflow.set_experiment`` **raises** against it unless ``MLFLOW_ALLOW_FILE_STORE``
is set.  On a wrapper that swallows tracking errors, that failure is invisible --
training runs perfectly and logs nothing, which is the worst of both.  The
default here is the backend MLflow actually supports, and a ``file:`` URI passed
explicitly gets the opt-out environment variable set for it rather than failing.

    mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

**Not failing the training run when tracking fails.**  A tracking server going
down should not destroy a GPU-hour of work.  Every call here is best-effort and
logs a warning on failure; the checkpoint and the metrics JSON on disk are the
source of truth, and MLflow is the index over them.  ``run_id`` is written into
the checkpoint so the two can always be reconciled.
"""

from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional

from .. import paths


def _flatten(prefix: str, data: Mapping[str, Any]) -> dict:
    out: dict = {}
    for key, value in data.items():
        name = "{0}.{1}".format(prefix, key) if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(name, value))
        elif isinstance(value, (list, tuple)):
            out[name] = ",".join(str(v) for v in value)
        else:
            out[name] = value
    return out


class Tracker:
    """A no-op-safe MLflow run."""

    def __init__(self, experiment: str, tracking_uri: Optional[str] = None,
                 enabled: bool = True):
        self.experiment = experiment
        self.enabled = enabled
        self.run_id: Optional[str] = None
        self._mlflow = None
        self._active = False
        if not enabled:
            return
        try:
            import mlflow
        except ImportError:
            warnings.warn("mlflow is not installed; tracking disabled", RuntimeWarning)
            self.enabled = False
            return
        self._mlflow = mlflow
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if not uri:
            paths.ensure(paths.MLRUNS)
            uri = "sqlite:///" + (paths.MLRUNS / "mlflow.db").as_posix()
        if uri.startswith("file:") or uri.startswith("./") or uri == "mlruns":
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        try:
            mlflow.set_tracking_uri(uri)
            artifact_root = (paths.MLRUNS / "artifacts").as_uri() if uri.startswith(
                "sqlite:") else None
            if artifact_root:
                paths.ensure(paths.MLRUNS / "artifacts")
                if mlflow.get_experiment_by_name(experiment) is None:
                    mlflow.create_experiment(experiment, artifact_location=artifact_root)
            mlflow.set_experiment(experiment)
        except Exception as error:  # noqa: BLE001 - tracking must never be fatal
            warnings.warn("mlflow setup failed ({0}); tracking disabled".format(error),
                          RuntimeWarning)
            self.enabled = False
        self.tracking_uri = uri

    def start(self, run_name: str, tags: Optional[Mapping[str, Any]] = None) -> None:
        if not self.enabled:
            return
        try:
            run = self._mlflow.start_run(run_name=run_name)
            self.run_id = run.info.run_id
            self._active = True
            if tags:
                self._mlflow.set_tags({k: str(v) for k, v in tags.items()})
        except Exception as error:  # noqa: BLE001
            warnings.warn("mlflow start_run failed: {0}".format(error), RuntimeWarning)
            self.enabled = False

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not (self.enabled and self._active):
            return
        try:
            flat = _flatten("", params)
            self._mlflow.log_params({k: str(v) for k, v in flat.items()})
        except Exception as error:  # noqa: BLE001
            warnings.warn("mlflow log_params failed: {0}".format(error), RuntimeWarning)

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if not (self.enabled and self._active):
            return
        numeric = {}
        for key, value in _flatten("", metrics).items():
            try:
                as_float = float(value)
            except (TypeError, ValueError):
                continue
            if as_float == as_float:  # skip NaN, which MLflow stores as a string
                numeric[key] = as_float
        if not numeric:
            return
        try:
            self._mlflow.log_metrics(numeric, step=step)
        except Exception as error:  # noqa: BLE001
            warnings.warn("mlflow log_metrics failed: {0}".format(error), RuntimeWarning)

    def log_artifact(self, path: Path, artifact_path: Optional[str] = None) -> None:
        if not (self.enabled and self._active):
            return
        try:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)
        except Exception as error:  # noqa: BLE001
            warnings.warn("mlflow log_artifact failed: {0}".format(error), RuntimeWarning)

    def end(self, status: str = "FINISHED") -> None:
        if not (self.enabled and self._active):
            return
        try:
            self._mlflow.end_run(status=status)
        except Exception as error:  # noqa: BLE001
            warnings.warn("mlflow end_run failed: {0}".format(error), RuntimeWarning)
        finally:
            self._active = False


@contextmanager
def track(experiment: str, run_name: str, tracking_uri: Optional[str] = None,
          enabled: bool = True, tags: Optional[Mapping[str, Any]] = None):
    """Context manager around :class:`Tracker`.

    Exits with status ``FAILED`` if the body raises, so a crashed training run
    does not sit in the UI looking like a finished one.
    """
    tracker = Tracker(experiment, tracking_uri=tracking_uri, enabled=enabled)
    tracker.start(run_name, tags=tags)
    try:
        yield tracker
    except BaseException:
        tracker.end(status="FAILED")
        raise
    else:
        tracker.end(status="FINISHED")
