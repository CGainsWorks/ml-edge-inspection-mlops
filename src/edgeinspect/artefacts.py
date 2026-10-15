"""Artefact identity: checksums and sidecars.

Deliberately free of torch, ONNX Runtime and everything else heavy, because of
who imports it.  The registry hashes artefacts, the edge agent verifies them
before loading, and the ONNX backend reads their sidecars -- all three run
inside the edge runtime container, which is built without a training framework.

This module exists because those helpers originally lived in
:mod:`edgeinspect.export.onnx_export`, which imports torch at module scope.
Nothing looked wrong: the code ran everywhere it was tested, because torch was
installed everywhere it was tested.  The runtime image would have failed on
first start with an ImportError three layers down from anything a reader would
associate with the cause.  ``tests/test_telemetry_and_pipeline.py`` now imports
the runtime in a subprocess and asserts that torch never appears in
``sys.modules``, which is what found this.

The sidecar
-----------
A ``.json`` next to every ``.onnx`` carrying what the graph itself cannot: the
decision threshold, the input normalisation, the top-k, the source checkpoint
and its hash.  A deployed model whose threshold lives only in a training log is
a model nobody can safely redeploy, and a graph that has been separated from its
sidecar will run and quietly use the wrong operating point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import paths


def sha256_file(path) -> str:
    """The SHA-256 of a file, read in one-megabyte chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sidecar_path(model_path) -> Path:
    """Where a model's sidecar lives: the same name with a ``.json`` suffix."""
    return Path(model_path).with_suffix(".json")


def write_sidecar(model_path, meta: dict) -> Path:
    """Write the sidecar and stamp it with the graph's own hash and size."""
    payload = dict(meta)
    payload["onnx"] = paths.relative(Path(model_path))
    payload["onnx_sha256"] = sha256_file(model_path)
    payload["onnx_bytes"] = int(Path(model_path).stat().st_size)
    target = sidecar_path(model_path)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def read_sidecar(model_path) -> dict:
    """Read a model's sidecar, or say clearly why the model is not deployable."""
    target = sidecar_path(model_path)
    if not target.exists():
        raise FileNotFoundError(
            "no sidecar at {0}.  The decision threshold and the input normalisation "
            "live there; a graph without one cannot be deployed safely.".format(
                paths.relative(target))
        )
    with target.open("r", encoding="utf-8") as handle:
        return json.load(handle)
