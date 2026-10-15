"""Post-training quantisation of the ONNX graph.

Produces QDQ INT8 graphs that ONNX Runtime executes on CPU today and that
TensorRT ingests on the Jetson.  Two calibration methods are built rather than
one -- MinMax and Percentile -- because which of the two wins is a property of
the activation distribution and not something to assume.  The answer for this
model is in ``docs/results.md``.

A finding worth knowing before reaching for Entropy
---------------------------------------------------
``CalibrationMethod.Entropy`` produces a **byte-identical graph to MinMax**
when ``ActivationSymmetric`` is true, on onnxruntime 1.29.  The entropy
calibrator runs, prints its progress, computes optimal clipping thresholds --
and the symmetric path then discards them in favour of the raw absolute
maximum.  Every activation scale comes out equal to the MinMax scale to the last
bit, and increasing ``CalibNumBins`` changes nothing.

The failure mode this creates is the dangerous kind: a pipeline that builds two
INT8 variants, evaluates both, reports that "entropy calibration made no
difference", and is wrong about why.  Percentile does *not* degenerate -- it
clips, symmetric or not -- so it is what this project builds as the second
variant.  ``tests/test_quantise.py`` pins both behaviours so an onnxruntime
upgrade that fixes the entropy path fails the test rather than passing quietly.

The reason a QDQ graph is the right artefact rather than the older
``QOperator`` form: TensorRT reads QuantizeLinear/DequantizeLinear nodes
natively and derives its own per-tensor scales from them, so the same file
serves the CPU fallback and the device engine build.  A QOperator graph would
need a separate TensorRT path.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .. import paths
from ..artefacts import read_sidecar, write_sidecar

METHODS = ("minmax", "entropy", "percentile")


class ArrayCalibrationReader:
    """A ``CalibrationDataReader`` over an in-memory array of frames."""

    def __init__(self, frames: np.ndarray, input_name: str = "input", batch_size: int = 8):
        self.frames = np.ascontiguousarray(frames, dtype=np.float32)
        self.input_name = input_name
        self.batch_size = int(batch_size)
        self.cursor = 0

    def get_next(self):
        if self.cursor >= len(self.frames):
            return None
        batch = self.frames[self.cursor:self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return {self.input_name: np.ascontiguousarray(batch)}

    def rewind(self):
        self.cursor = 0


def _calibration_method(name: str):
    from onnxruntime.quantization import CalibrationMethod

    return {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }[name]


def preprocess(onnx_path: Path, out_path: Path) -> Path:
    """Shape inference and graph cleanup before quantisation.

    Skipping this is the single most common way to get a QDQ graph that is
    slower than its FP32 source: without inferred shapes the quantiser leaves
    conversions in place that the runtime then cannot fuse.
    """
    from onnxruntime.quantization.shape_inference import quant_pre_process

    out_path = Path(out_path)
    paths.ensure(out_path.parent)
    quant_pre_process(str(onnx_path), str(out_path), skip_symbolic_shape=False)
    return out_path


def output_conv_nodes(onnx_path: Path, count: int = 1) -> list:
    """The names of the last ``count`` Conv nodes in topological order.

    Used to keep the detection head out of INT8.  The head's output is a logit
    map whose *ranking* is the product -- the image score is the mean of its
    hottest cells -- and ranking is exactly what a coarse quantisation grid
    destroys.  Everything before the head produces features whose absolute
    values matter less than their relative pattern, and those survive INT8
    comfortably.  Measured effect on this model is in ``docs/results.md``.
    """
    import onnx

    model = onnx.load(str(onnx_path))
    convs = [node.name for node in model.graph.node if node.op_type == "Conv"]
    return convs[-count:] if convs else []


def quantise(onnx_path: Path, out_path: Path, frames: np.ndarray,
             method: str = "minmax", per_channel: bool = True,
             batch_size: int = 8, input_name: str = "input",
             calibration_manifest: Optional[Dict] = None,
             extra_options: Optional[Dict] = None,
             exclude_nodes: Optional[Sequence[str]] = None) -> Path:
    """Build one INT8 QDQ graph from ``onnx_path``.  Returns the output path."""
    if method not in METHODS:
        raise ValueError("method must be one of {0}, got {1!r}".format(METHODS, method))
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

    onnx_path = Path(onnx_path)
    out_path = Path(out_path)
    paths.ensure(out_path.parent)

    prepared = out_path.with_name(out_path.stem + "_prepared.onnx")
    preprocess(onnx_path, prepared)

    reader = ArrayCalibrationReader(frames, input_name=input_name, batch_size=batch_size)
    options = {
        # Symmetric activations map cleanly onto TensorRT's INT8 tensor scales,
        # which have no zero point.  Leaving activations asymmetric produces a
        # graph ONNX Runtime is happy with and TensorRT has to work around.
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    }
    if extra_options:
        options.update(extra_options)

    try:
        quantize_static(
            model_input=str(prepared),
            model_output=str(out_path),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            calibrate_method=_calibration_method(method),
            nodes_to_exclude=list(exclude_nodes) if exclude_nodes else None,
            extra_options=options,
        )
    finally:
        prepared.unlink(missing_ok=True)
        external = prepared.with_suffix(".onnx.data")
        external.unlink(missing_ok=True)

    # The sidecar travels with the graph.  A quantised model that has lost its
    # decision threshold is not deployable, and rediscovering the threshold
    # means re-running evaluation.
    source_sidecar = read_sidecar(onnx_path)
    sidecar = dict(source_sidecar)
    sidecar["precision"] = "int8"
    sidecar["name"] = out_path.stem
    sidecar["quantisation"] = {
        "format": "QDQ",
        "method": method,
        "per_channel": per_channel,
        "activation_type": "QInt8",
        "weight_type": "QInt8",
        "activation_symmetric": bool(options.get("ActivationSymmetric", True)),
        "excluded_nodes": list(exclude_nodes) if exclude_nodes else [],
        "source": paths.relative(onnx_path),
        "calibration": calibration_manifest or {},
    }
    # The INT8 graph's own decision threshold is not the FP32 one: quantisation
    # shifts the score distribution.  It is recalibrated by scripts/evaluate.py
    # and written back, so the value inherited here is marked stale until then.
    sidecar["decision"] = dict(source_sidecar.get("decision", {}))
    sidecar["decision"]["threshold_source"] = "inherited-from-fp32-pending-recalibration"
    write_sidecar(out_path, sidecar)
    return out_path


def build_variants(onnx_path: Path, out_dir: Path, frames: np.ndarray,
                   manifest: Dict, methods: Sequence[str] = ("minmax", "percentile"),
                   per_channel: bool = True, keep_head_fp32: bool = True) -> Dict[str, Path]:
    """Build the INT8 variant set.  Returns ``{name: path}``.

    One graph per calibration method, plus -- with ``keep_head_fp32`` -- a
    variant that leaves the detection head in floating point.  That variant
    exists because on this model the naive all-layers INT8 graph loses far more
    accuracy than INT8 should cost, and having both in the table is what turns
    "INT8 hurt" into "INT8 hurt *here*, for this reason".
    """
    onnx_path = Path(onnx_path)
    out_dir = Path(out_dir)
    stem = onnx_path.stem.replace("_fp32", "")
    built: Dict[str, Path] = {}
    for method in methods:
        suffix = "int8" if method == "minmax" else "int8_" + method
        target = out_dir / "{0}_{1}.onnx".format(stem, suffix)
        quantise(onnx_path, target, frames, method=method, per_channel=per_channel,
                 calibration_manifest=manifest)
        built[suffix] = target
    if keep_head_fp32:
        head = output_conv_nodes(onnx_path, count=1)
        target = out_dir / "{0}_int8_keephead.onnx".format(stem)
        quantise(onnx_path, target, frames, method="minmax", per_channel=per_channel,
                 calibration_manifest=manifest, exclude_nodes=head)
        built["int8_keephead"] = target
    return built


def size_report(paths_by_name: Dict[str, Path]) -> Dict[str, dict]:
    """Bytes on disk per variant, with the ratio against FP32.

    Reported because it is the one INT8 benefit that is real on every platform.
    Latency depends on whether the runtime has INT8 kernels for these operators;
    the file is four times smaller regardless, and on a device that ships a
    model over the air that matters on its own.
    """
    out: Dict[str, dict] = {}
    reference = None
    for name, path in paths_by_name.items():
        size = int(Path(path).stat().st_size)
        out[name] = {"bytes": size}
        if name in ("fp32", "float32"):
            reference = size
    if reference:
        for name in out:
            out[name]["ratio_vs_fp32"] = round(out[name]["bytes"] / reference, 4)
    return out


def copy_with_sidecar(source: Path, target: Path) -> Path:
    """Copy an ONNX graph and its sidecar together.

    Used by the deployment step.  A graph copied without its sidecar is a
    deployment that will look fine and use the wrong threshold, so this is the
    only supported way to move one.
    """
    source, target = Path(source), Path(target)
    paths.ensure(target.parent)
    shutil.copy2(source, target)
    sidecar = read_sidecar(source)
    sidecar["onnx"] = paths.relative(target)
    with target.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2)
    return target
