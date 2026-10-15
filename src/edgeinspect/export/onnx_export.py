"""Torch checkpoint to ONNX.

The exported graph is the deployment artefact.  Everything after this point --
PTQ calibration, the accuracy comparison, the latency benchmark, the TensorRT
build on the Jetson -- consumes ONNX and never touches the checkpoint again.

Three choices worth stating:

**Dynamic batch, static spatial size.**  Batch is dynamic because the benchmark
sweeps it and the Jetson runtime batches frames from the camera.  Height and
width are static because TensorRT builds a separate optimisation profile for
every dynamic dimension, and a profile per spatial size costs build time and
engine size for a resolution that does not change once a camera is bolted down.
``--dynamic-size`` overrides this when it genuinely does change.

**Opset 17, and the TorchScript exporter rather than the dynamo one.**  The
opset floor for the operators used here is far lower, but 17 is what TensorRT 10
and ONNX Runtime 1.18 both parse without a shim.  The exporter is pinned for the
same reason the opset is: from torch 2.9 the default became the dynamo exporter,
which needs `onnxscript` and emits a structurally different graph, so leaving it
to the default would mean the deployment artefact changing shape because a
dependency moved.  Pinning both means the graph the Jetson receives is the graph
CI tested.

**A sidecar JSON.**  Threshold, top-k, stride, input normalisation and the
source checkpoint's hash live next to the ``.onnx`` rather than inside it.  A
deployed model whose decision threshold is only recorded in a training log is a
model nobody can safely redeploy.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .. import paths
# Checksums and sidecars live in a torch-free module because the edge runtime
# needs them and is built without torch; see edgeinspect/artefacts.py.
from ..artefacts import read_sidecar, sha256_file, sidecar_path, write_sidecar
from ..models.net import DefectNet, count_parameters
from ..models.preprocess import InputSpec

__all__ = ["export", "verify", "read_sidecar", "write_sidecar", "sidecar_path",
           "sha256_file", "DEFAULT_OPSET"]

DEFAULT_OPSET = 17

# Whether this torch lets the exporter be chosen.  Resolved once, at import,
# against the real function: doing it per call means inspecting whatever
# `torch.onnx.export` happens to be at that moment, and anything that wraps it
# -- a profiler, a test spy -- hides the parameter and silently turns the pin
# off.  A test monkeypatching the export found exactly that.
_EXPORTER_IS_SELECTABLE = "dynamo" in inspect.signature(torch.onnx.export).parameters


def _input_spec(meta: dict, input_size: int) -> InputSpec:
    """The input contract for this checkpoint.

    Taken from the checkpoint's own metadata rather than from the module
    default, so a model trained under a different contract -- which is what
    scripts/ablate_quantisation.py builds -- exports a sidecar describing the
    contract it was actually trained with.
    """
    declared = dict(meta.get("input", {}))
    return InputSpec(
        mean=float(declared.get("mean", InputSpec().mean)),
        scale=float(declared.get("scale", InputSpec().scale)),
        size=input_size,
        channels=int(meta.get("in_channels", 1)),
    )


def export(model: DefectNet, out_path: Path, meta: dict,
           input_size: int = 256, opset: int = DEFAULT_OPSET,
           dynamic_size: bool = False,
           checkpoint_path: Optional[Path] = None) -> Path:
    """Export ``model`` to ONNX and write its sidecar.  Returns the graph path."""
    out_path = Path(out_path)
    paths.ensure(out_path.parent)
    model = model.eval().to("cpu")

    dummy = torch.zeros((1, int(meta.get("in_channels", 1)), input_size, input_size),
                        dtype=torch.float32)
    dynamic_axes = {"input": {0: "batch"}, "heatmap": {0: "batch"}}
    if dynamic_size:
        dynamic_axes["input"].update({2: "height", 3: "width"})
        dynamic_axes["heatmap"].update({2: "cells_y", 3: "cells_x"})

    # Pin the exporter, not the framework.
    #
    # From torch 2.9 `torch.onnx.export` defaults to the dynamo exporter, which
    # needs `onnxscript` installed and produces a structurally different graph.
    # This repository's requirements allow any torch 2.x, so CI installed a
    # newer one than the development machine and the export died with
    # ModuleNotFoundError: No module named 'onnxscript' -- on code that had
    # worked locally all along.
    #
    # Adding onnxscript would have made it run; it would also have meant the
    # deployment artefact silently changing shape because a dependency moved,
    # and the artefact is what the INT8 calibration, the sidecar and the
    # TensorRT build all consume.  So the exporter is chosen explicitly and the
    # graph is the same on every torch that offers the choice.  `dynamo` does
    # not exist before 2.5, hence the capability check rather than a version
    # comparison.
    options = {"dynamo": False} if _EXPORTER_IS_SELECTABLE else {}

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["heatmap"],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        **options,
    )

    sidecar = {
        # The sidecar describes the graph, so its name is the graph's filename.
        # Naming it after the source checkpoint instead left the FP32 variant
        # called "surface_defect" while its file was surface_defect_fp32.onnx --
        # and since the quantised variants are named after their own files, the
        # accuracy report and the latency report could not be joined on the
        # variant at all.  The source model's name is kept below.
        "name": out_path.stem,
        "source_name": meta.get("name", out_path.stem),
        "precision": meta.get("precision", "fp32"),
        "input": _input_spec(meta, input_size).to_dict(),
        "output": {"name": "heatmap", "kind": "logits", "stride": int(meta.get("stride", 8))},
        "decision": {
            "threshold": float(meta.get("threshold", 0.0)),
            "top_k": int(meta.get("top_k", 8)),
            "cell_threshold": float(meta.get("cell_threshold", 0.5)),
            "target_fpr": float(meta.get("target_fpr", 0.02)),
        },
        "opset": opset,
        "dynamic_size": bool(dynamic_size),
        "parameters": count_parameters(model),
        "torch": torch.__version__,
        "source_checkpoint": paths.relative(Path(checkpoint_path)) if checkpoint_path else None,
        "source_checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path else None
        ),
        "training": meta.get("training", {}),
    }
    write_sidecar(out_path, sidecar)
    return out_path


def verify(onnx_path: Path, model: DefectNet, input_size: int = 256,
           samples: int = 4, tolerance: float = 1e-4, seed: int = 0) -> dict:
    """Check the exported graph against the torch model it came from.

    Not a formality.  Batch-norm folding, constant folding and opset lowering
    are all places where an export silently changes numerics, and the first
    place that shows up is a quantisation calibration built on the wrong
    activation ranges.  ``scripts/export_onnx.py`` runs this and fails on it.
    """
    import onnxruntime as ort

    rng = np.random.default_rng(seed)
    channels = int(model.stem.conv.in_channels)
    batch = rng.random((samples, channels, input_size, input_size)).astype(np.float32)

    with torch.no_grad():
        reference = model.eval().to("cpu")(torch.from_numpy(batch)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    produced = session.run(["heatmap"], {"input": batch})[0]

    difference = np.abs(reference - produced)
    result = {
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "tolerance": tolerance,
        "passed": bool(difference.max() <= tolerance),
        "shape": list(produced.shape),
    }
    if not result["passed"]:
        raise AssertionError(
            "ONNX export does not match the torch model: max |diff| = {0:.3e} > {1:.3e}. "
            "Do not quantise or deploy this graph.".format(result["max_abs_diff"], tolerance)
        )
    return result
