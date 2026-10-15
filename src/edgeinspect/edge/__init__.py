"""The edge side: inference runtime, telemetry, frame capture and the deploy agent.

Nothing in this subpackage imports torch.  ``tests/test_runtime_boundary.py``
asserts that, because the runtime container is built without it and a stray
import would only surface as a broken image.
"""

from .agent import EdgeAgent, smoke_check
from .backends import Backend, load
from .source import DatasetSource, StreamSpec, SyntheticCamera
from .telemetry import CaptureStore, TelemetryRecord, TelemetryWriter, read, to_arrays

__all__ = [
    "EdgeAgent",
    "smoke_check",
    "Backend",
    "load",
    "DatasetSource",
    "StreamSpec",
    "SyntheticCamera",
    "CaptureStore",
    "TelemetryRecord",
    "TelemetryWriter",
    "read",
    "to_arrays",
]
