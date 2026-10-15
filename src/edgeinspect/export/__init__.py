"""Export: torch to ONNX, ONNX to TensorRT.

``tensorrt_export`` is importable without TensorRT installed; it raises
:class:`~edgeinspect.export.tensorrt_export.NotAvailable` only when a build is
actually attempted.
"""

from . import tensorrt_export
# From the torch-free module, so `import edgeinspect.export` works inside the
# runtime container.  The torch-dependent entry points below are lazy.
from ..artefacts import read_sidecar, sha256_file, sidecar_path, write_sidecar

__all__ = [
    "tensorrt_export",
    "read_sidecar",
    "sha256_file",
    "sidecar_path",
    "write_sidecar",
    "export",
    "verify",
]


def export(*args, **kwargs):
    """Export a torch model to ONNX.  Needs torch."""
    from .onnx_export import export as _export

    return _export(*args, **kwargs)


def verify(*args, **kwargs):
    """Check an exported graph against its torch source.  Needs torch."""
    from .onnx_export import verify as _verify

    return _verify(*args, **kwargs)
