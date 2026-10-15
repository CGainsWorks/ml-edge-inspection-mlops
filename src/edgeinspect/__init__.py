"""Edge defect inspection with a closed MLOps loop.

Train, quantise, export, deploy to an edge device, watch production telemetry
for drift, and retrain automatically when the drift detector fires.

The interesting part of this package is not :mod:`edgeinspect.models` -- it is
:mod:`edgeinspect.monitoring` and :mod:`edgeinspect.registry`, which are what
turn a checkpoint into something that can be operated.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
