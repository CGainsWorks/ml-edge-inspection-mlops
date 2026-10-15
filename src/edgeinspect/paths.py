"""Where things live.

Every path in this project is derived from the repository root so that scripts
behave the same whether they are run from the root, from ``scripts/`` or from
inside a container.  Nothing here creates a directory as a side effect of being
imported; call :func:`ensure` when you actually intend to write.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """The repository root.

    ``EDGEINSPECT_ROOT`` overrides the search, which is how the Docker images
    point at ``/app`` without relying on the source tree layout.
    """
    override = os.environ.get("EDGEINSPECT_ROOT")
    if override:
        return Path(override).resolve()
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "pyproject.toml").exists() or (candidate / "Makefile").exists():
            return candidate
    # src/edgeinspect/paths.py -> src/edgeinspect -> src -> root
    return here.parents[2]


ROOT = repo_root()

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
PUBLISHED = RESULTS / "published"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
BENCHMARKS = ROOT / "benchmarks"

# Operational state.  These are the directories the running system reads and
# writes: what is deployed, what the edge device has seen, and what the
# retraining pipeline produced.
REGISTRY = ARTIFACTS / "registry"
DEPLOY = ARTIFACTS / "deploy"
TELEMETRY = ARTIFACTS / "telemetry"
CAPTURE = ARTIFACTS / "capture"
RETRAIN = ARTIFACTS / "retrain"

MLRUNS = ROOT / "mlruns"


def ensure(*targets: Path) -> None:
    """Create directories, parents included.  Idempotent."""
    for target in targets:
        Path(target).mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    """A repository-relative POSIX path, for anything that gets serialised.

    Absolute Windows paths in a committed JSON file break on Linux CI and the
    breakage is silent, so serialised paths go through here.
    """
    path = Path(path)
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def resolve(path) -> Path:
    """Inverse of :func:`relative`: repository-relative strings become absolute."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()
