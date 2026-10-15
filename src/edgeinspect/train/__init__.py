"""Training, tracking and quantisation-aware fine-tuning.

Everything in this subpackage needs torch and none of it is imported by the
edge runtime.
"""

from .tracking import Tracker, track

__all__ = ["Tracker", "track", "train", "load_checkpoint", "evaluate_arrays"]


def train(*args, **kwargs):
    from .loop import train as _train

    return _train(*args, **kwargs)


def load_checkpoint(*args, **kwargs):
    from .loop import load_checkpoint as _load

    return _load(*args, **kwargs)


def evaluate_arrays(*args, **kwargs):
    from .loop import evaluate_arrays as _evaluate

    return _evaluate(*args, **kwargs)
