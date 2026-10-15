"""ONNX to a TensorRT engine, at FP32, FP16 or INT8.

**This module has never been executed.**  No Jetson was available for this
project and TensorRT does not run on the development machine, so every number in
this repository that involves TensorRT is absent rather than estimated.
``docs/jetson.md`` says exactly what that means and what would have to be run to
fill the gap.

What is here is the real build code, written against the TensorRT 10 Python API,
kept in the repository for three reasons: it is the artefact a Jetson bring-up
would actually run; it documents the INT8 calibration contract that
``edgeinspect.quantise`` already satisfies for ONNX Runtime; and it makes the
gap concrete rather than a sentence in a README.

Import safety
-------------
``import tensorrt`` happens inside the functions, never at module scope, so
``import edgeinspect.export`` works on a machine with no TensorRT and the test
suite can import and inspect this module without skipping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .. import paths

PRECISIONS = ("fp32", "fp16", "int8")


class NotAvailable(RuntimeError):
    """Raised when TensorRT is not importable.  Carries the reason, not a stack."""


def available() -> bool:
    try:
        import tensorrt  # noqa: F401
    except Exception:  # noqa: BLE001 - a broken CUDA install raises more than ImportError
        return False
    return True


def _require():
    try:
        import tensorrt as trt
    except Exception as error:  # noqa: BLE001
        raise NotAvailable(
            "TensorRT is not importable ({0}).\n"
            "It ships with JetPack and is not installable from PyPI on a desktop.\n"
            "See docs/jetson.md; nothing in results/published/ came from this path."
            .format(error)
        ) from error
    return trt


def build_calibrator(trt, calibration: np.ndarray, cache_path: Path, batch_size: int = 8):
    """An ``IInt8EntropyCalibrator2`` over a fixed array of calibration frames.

    The calibration set is the same one ``edgeinspect.quantise.calibration``
    selects for the ONNX Runtime PTQ path, so the two INT8 variants are
    calibrated on identical data and any accuracy difference between them is
    attributable to the quantisation implementation rather than to the sample.
    """
    import ctypes

    class ArrayCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.data = np.ascontiguousarray(calibration, dtype=np.float32)
            self.batch_size = int(batch_size)
            self.cursor = 0
            self.cache_path = Path(cache_path)
            self.device_input = None
            self._allocate()

        def _allocate(self):
            from cuda import cudart

            nbytes = int(self.batch_size * self.data[0].nbytes)
            status, pointer = cudart.cudaMalloc(nbytes)
            if status != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("cudaMalloc failed for the calibration buffer")
            self.device_input = pointer
            self.nbytes = nbytes

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            from cuda import cudart

            if self.cursor + self.batch_size > len(self.data):
                return None
            batch = np.ascontiguousarray(self.data[self.cursor:self.cursor + self.batch_size])
            self.cursor += self.batch_size
            cudart.cudaMemcpy(
                self.device_input,
                ctypes.c_void_p(batch.ctypes.data),
                batch.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            )
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if self.cache_path.exists():
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            paths.ensure(self.cache_path.parent)
            self.cache_path.write_bytes(cache)

    return ArrayCalibrator()


def build_engine(onnx_path: Path, engine_path: Path, precision: str = "fp16",
                 workspace_gb: float = 2.0,
                 calibration: Optional[np.ndarray] = None,
                 calibration_cache: Optional[Path] = None,
                 min_batch: int = 1, opt_batch: int = 1, max_batch: int = 8,
                 verbose: bool = False) -> Path:
    """Build a serialised TensorRT engine.  Jetson only; never run in this repository.

    A plan file is tied to the exact GPU architecture, driver and TensorRT
    version that produced it, which is why ``.engine`` is in ``.gitignore`` and
    why this runs on the device rather than in CI.
    """
    if precision not in PRECISIONS:
        raise ValueError("precision must be one of {0}, got {1!r}".format(PRECISIONS, precision))
    trt = _require()

    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    onnx_bytes = Path(onnx_path).read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [parser.get_error(i) for i in range(parser.num_errors)]
        raise RuntimeError("failed to parse {0}:\n{1}".format(
            onnx_path, "\n".join(str(e) for e in errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))

    profile = builder.create_optimization_profile()
    input_tensor = network.get_input(0)
    shape = list(input_tensor.shape)
    spatial = tuple(int(v) for v in shape[1:])
    profile.set_shape(input_tensor.name,
                      min=(min_batch,) + spatial,
                      opt=(opt_batch,) + spatial,
                      max=(max_batch,) + spatial)
    config.add_optimization_profile(profile)

    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("this platform has no fast FP16; refusing to build a slower engine")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("this platform has no fast INT8")
        if calibration is None:
            raise ValueError(
                "INT8 needs a calibration set.  Build one with "
                "edgeinspect.quantise.calibration.select() so the TensorRT and "
                "ONNX Runtime INT8 variants share the same frames."
            )
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # mixed: layers INT8 cannot serve stay FP16
        cache = Path(calibration_cache or Path(engine_path).with_suffix(".calib"))
        config.int8_calibrator = build_calibrator(trt, calibration, cache)

    serialised = builder.build_serialized_network(network, config)
    if serialised is None:
        raise RuntimeError("TensorRT returned no engine; run with verbose=True for the log")

    engine_path = Path(engine_path)
    paths.ensure(engine_path.parent)
    engine_path.write_bytes(serialised)

    manifest = {
        "engine": paths.relative(engine_path),
        "onnx": paths.relative(Path(onnx_path)),
        "precision": precision,
        "tensorrt": trt.__version__,
        "batch_profile": {"min": min_batch, "opt": opt_batch, "max": max_batch},
        "input_shape": [int(v) for v in shape],
        "bytes": engine_path.stat().st_size,
        "warning": (
            "A TensorRT plan is specific to the GPU architecture, driver and TensorRT "
            "version that built it.  Rebuild on the target device; never commit."
        ),
    }
    with engine_path.with_suffix(".engine.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return engine_path


def plan_builds(onnx_path: Path, out_dir: Path,
                precisions: Sequence[str] = PRECISIONS) -> Iterable[dict]:
    """What ``make engine`` would build, as data.

    Callable without TensorRT, so ``scripts/export_tensorrt.py --dry-run`` can
    print the exact plan on a machine that cannot execute it, and
    ``tests/test_tensorrt_plan.py`` can assert the plan is coherent.
    """
    stem = Path(onnx_path).stem.replace("_fp32", "")
    for precision in precisions:
        if precision not in PRECISIONS:
            raise ValueError("unknown precision " + repr(precision))
        yield {
            "precision": precision,
            "onnx": paths.relative(Path(onnx_path)),
            "engine": paths.relative(Path(out_dir) / "{0}_{1}.engine".format(stem, precision)),
            "needs_calibration": precision == "int8",
        }
