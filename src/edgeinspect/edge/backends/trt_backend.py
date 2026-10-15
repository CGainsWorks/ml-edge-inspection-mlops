"""TensorRT backend.

**Never executed.**  No Jetson was available, so no number in this repository
came from this file.  ``docs/jetson.md`` states what would need to be run to
produce one.

The code is real and is written against the TensorRT 10 API, whose execution
model differs from TensorRT 8 in a way that matters: bindings are addressed by
tensor *name* rather than by index, and shapes are set on the execution context
rather than baked into the engine.  Code copied from a TensorRT 8 example does
not run on JetPack 6, which is the most common reason a Jetson bring-up stalls
on day one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .base import Backend


class TensorRTBackend(Backend):
    """A deserialised engine with device buffers and one CUDA stream."""

    name = "tensorrt"

    def __init__(self, model_path: Path, sidecar: Optional[Dict] = None,
                 max_batch: int = 8):
        super().__init__(model_path, sidecar or self._read_sidecar(model_path))
        try:
            import tensorrt as trt
            from cuda import cudart
        except Exception as error:  # noqa: BLE001
            raise RuntimeError(
                "TensorRT backend requested but TensorRT is not importable ({0}).\n"
                "This path runs on a Jetson with JetPack; see docs/jetson.md."
                .format(error)
            ) from error

        self._trt = trt
        self._cudart = cudart
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(model_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(
                "failed to deserialise {0}.  A plan is tied to the exact GPU, driver and "
                "TensorRT version that built it; rebuild it on this device."
                .format(model_path)
            )
        self.context = self.engine.create_execution_context()
        self.max_batch = int(max_batch)

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.input_name = next(
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.output_name = next(
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT)

        status, self.stream = cudart.cudaStreamCreate()
        if status != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("cudaStreamCreate failed")
        self._device_buffers: Dict[str, int] = {}
        self._buffer_bytes: Dict[str, int] = {}

    @staticmethod
    def _read_sidecar(model_path: Path) -> Dict:
        candidate = Path(model_path).with_suffix(".json")
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {}

    def _ensure_buffer(self, name: str, nbytes: int) -> int:
        if self._buffer_bytes.get(name, 0) >= nbytes:
            return self._device_buffers[name]
        if name in self._device_buffers:
            self._cudart.cudaFree(self._device_buffers[name])
        status, pointer = self._cudart.cudaMalloc(nbytes)
        if status != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("cudaMalloc failed for {0} ({1} bytes)".format(name, nbytes))
        self._device_buffers[name] = pointer
        self._buffer_bytes[name] = nbytes
        return pointer

    def infer_raw(self, batch: np.ndarray) -> np.ndarray:
        import ctypes

        cudart = self._cudart
        array = np.ascontiguousarray(batch, dtype=np.float32)
        if len(array) > self.max_batch:
            raise ValueError(
                "batch of {0} exceeds the engine's optimisation profile maximum of {1}. "
                "Rebuild the engine with a larger max_batch.".format(len(array), self.max_batch)
            )
        self.context.set_input_shape(self.input_name, array.shape)
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        output = np.empty(output_shape, dtype=np.float32)

        input_ptr = self._ensure_buffer(self.input_name, array.nbytes)
        output_ptr = self._ensure_buffer(self.output_name, output.nbytes)

        cudart.cudaMemcpyAsync(input_ptr, ctypes.c_void_p(array.ctypes.data), array.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream)
        self.context.set_tensor_address(self.input_name, int(input_ptr))
        self.context.set_tensor_address(self.output_name, int(output_ptr))
        if not self.context.execute_async_v3(stream_handle=int(self.stream)):
            raise RuntimeError("execute_async_v3 returned False")
        cudart.cudaMemcpyAsync(ctypes.c_void_p(output.ctypes.data), output_ptr, output.nbytes,
                               cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream)
        cudart.cudaStreamSynchronize(self.stream)
        return output

    def describe(self) -> Dict:
        out = super().describe()
        out["tensorrt"] = self._trt.__version__
        out["max_batch"] = self.max_batch
        return out

    def close(self) -> None:
        for pointer in self._device_buffers.values():
            self._cudart.cudaFree(pointer)
        self._device_buffers.clear()
        self._buffer_bytes.clear()
        if getattr(self, "stream", None) is not None:
            self._cudart.cudaStreamDestroy(self.stream)
            self.stream = None
        self.context = None
        self.engine = None
