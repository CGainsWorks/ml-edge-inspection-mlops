"""The inference backend interface.

One method that matters: :meth:`Backend.infer` takes an ``(n, 1, H, W)`` float32
batch of frames **in [0, 1]** and returns the raw logit map.  Everything above
this line -- scoring, box extraction, telemetry, drift -- is backend-agnostic,
which is what lets the ONNX Runtime path in CI and the TensorRT path on the
Jetson be compared without a second implementation of the pipeline.

The backend, not the caller, applies the input normalisation recorded in the
model's sidecar.  That normalisation is part of the deployment contract and is
worth 0.156 AUROC under INT8 (see :mod:`edgeinspect.models.preprocess`), and
a contract that every call site has to remember to honour is a contract that one
call site will eventually get wrong.  :meth:`Backend.infer_raw` bypasses it for
the two places that genuinely want the bare graph: export verification and the
quantisation calibration reader.
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ...models.preprocess import InputSpec


class Backend(abc.ABC):
    """A loaded model that can be run on frames."""

    name = "base"

    def __init__(self, model_path: Path, sidecar: Optional[Dict] = None):
        self.model_path = Path(model_path)
        self.sidecar = dict(sidecar or {})
        self._input_spec: Optional[InputSpec] = None

    @abc.abstractmethod
    def infer_raw(self, batch: np.ndarray) -> np.ndarray:
        """Run the graph on already-normalised input.  No preprocessing applied."""

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """Run the model on frames in ``[0, 1]``.

        ``(n, c, h, w)`` float32 in, ``(n, 1, gh, gw)`` float32 logits out.
        """
        return self.infer_raw(self.input_spec.apply(
            np.asarray(batch, dtype=np.float32)))

    @property
    def input_spec(self) -> InputSpec:
        if self._input_spec is None:
            self._input_spec = InputSpec.from_sidecar(self.sidecar)
        return self._input_spec

    @property
    def precision(self) -> str:
        return str(self.sidecar.get("precision", "unknown"))

    @property
    def threshold(self) -> float:
        return float(self.sidecar.get("decision", {}).get("threshold", 0.0))

    @property
    def top_k(self) -> int:
        return int(self.sidecar.get("decision", {}).get("top_k", 8))

    @property
    def cell_threshold(self) -> float:
        return float(self.sidecar.get("decision", {}).get("cell_threshold", 0.5))

    def describe(self) -> Dict:
        return {
            "backend": self.name,
            "model": str(self.model_path),
            "precision": self.precision,
            "threshold": self.threshold,
            "top_k": self.top_k,
            "input": {"mean": self.input_spec.mean, "scale": self.input_spec.scale},
        }

    def close(self) -> None:
        """Release device resources.  Safe to call more than once."""

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def load(model_path: Path, backend: str = "auto", **kwargs) -> Backend:
    """Construct a backend for ``model_path``.

    ``auto`` picks TensorRT for a ``.engine`` and ONNX Runtime for a ``.onnx``,
    which is the only difference between the Jetson deployment and the CI one.
    """
    model_path = Path(model_path)
    if backend == "auto":
        backend = "tensorrt" if model_path.suffix == ".engine" else "onnx"
    if backend == "onnx":
        from .onnx_backend import OnnxBackend

        return OnnxBackend(model_path, **kwargs)
    if backend == "tensorrt":
        from .trt_backend import TensorRTBackend

        return TensorRTBackend(model_path, **kwargs)
    raise ValueError("unknown backend " + repr(backend))
