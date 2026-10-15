"""ONNX Runtime backend.

The portable path: it runs on the development machine, in CI, in the runtime
container and on a Jetson that has not had an engine built for it yet.  Every
published accuracy number in this repository came from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ...artefacts import read_sidecar
from .base import Backend

DEFAULT_PROVIDERS = ("CPUExecutionProvider",)


class OnnxBackend(Backend):
    name = "onnx"

    def __init__(self, model_path: Path, providers: Optional[Sequence[str]] = None,
                 intra_op_threads: Optional[int] = None,
                 sidecar: Optional[Dict] = None):
        import onnxruntime as ort

        super().__init__(model_path, sidecar or self._read_sidecar(model_path))
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if intra_op_threads is not None:
            # Pinned for benchmarking.  Letting ONNX Runtime choose a thread
            # count makes latency a function of what else the machine is doing,
            # and a latency table that moves with the background load is not a
            # measurement.
            options.intra_op_num_threads = int(intra_op_threads)
        requested = list(providers or DEFAULT_PROVIDERS)
        available = set(ort.get_available_providers())
        missing = [p for p in requested if p not in available]
        if missing:
            raise RuntimeError(
                "execution provider(s) {0} are not available in this onnxruntime build. "
                "Available: {1}".format(missing, sorted(available))
            )
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options,
                                            providers=requested)
        self.providers: List[str] = list(self.session.get_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @staticmethod
    def _read_sidecar(model_path: Path) -> Dict:
        try:
            return read_sidecar(model_path)
        except FileNotFoundError:
            return {}

    def infer_raw(self, batch: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(batch, dtype=np.float32)
        return self.session.run([self.output_name], {self.input_name: array})[0]

    def describe(self) -> Dict:
        out = super().describe()
        out["providers"] = self.providers
        return out

    def close(self) -> None:
        self.session = None
