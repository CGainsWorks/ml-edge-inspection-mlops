"""Quantisation: calibration set selection, post-training quantisation, QAT.

``calibration`` and ``ptq`` need only NumPy and ONNX Runtime.  ``qat`` needs
torch and is imported lazily.
"""

from . import calibration, ptq

__all__ = ["calibration", "ptq", "qat_module"]


def qat_module():
    """Import and return :mod:`edgeinspect.quantise.qat`.  Needs torch."""
    from . import qat

    return qat
