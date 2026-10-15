"""Drift detectors.

Four tests, each with a different blind spot, which is why there are four:

* **PSI** -- population stability index.  Bins the reference distribution and
  measures how much probability mass moved.  Interpretable, bounded in practice,
  and the industry convention for thresholds (0.1 warn, 0.25 alarm) is worth
  inheriting because operations teams already know what those numbers mean.
  Blind to a shift that redistributes within bins.
* **Kolmogorov-Smirnov** -- the largest gap between the two empirical CDFs.
  Sensitive to shape changes PSI's binning smooths over.  Blind to nothing much,
  but its p-value is useless at these window sizes: with 200 samples any real
  shift gives p < 1e-6, so the *statistic* is what the rules threshold, not the
  p-value.
* **Standardised mean shift** -- the plainest thing that can be said about a
  distribution moving, in units of the reference standard deviation.  Reported
  because it is the number that goes in the alert text: "brightness fell by 2.4
  reference standard deviations" is actionable in a way that "PSI 0.42" is not.
* **Page-Hinkley** -- sequential rather than windowed.  The other three compare
  a window to a reference and therefore cannot answer "when did it start"; a
  ramp is fully inside a window long before it is fully inside the data.  Page-
  Hinkley accumulates deviation frame by frame and fires at a specific index,
  which is how ``docs/results.md`` reports detection latency in frames after
  onset rather than in windows.

All of them are two-sample against a *frozen reference window*, captured at
deployment and stored with the deployment.  Comparing against a rolling recent
window instead is the classic self-defeating design: a slow ramp never looks
different from ten minutes ago, and the detector sleeps through it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np


@dataclass
class DetectorResult:
    """One test applied to one signal."""

    detector: str
    signal: str
    statistic: float
    detail: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "signal": self.signal,
            "statistic": round(float(self.statistic), 6),
            "detail": self.detail,
        }


def _clean(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def psi_null_expectation(bins: int, n_reference: int, n_current: int) -> float:
    """What PSI reads on two samples from the *same* distribution.

    PSI is a plug-in estimate of a symmetrised divergence between two
    histograms, and like every plug-in divergence estimate it is biased upward
    by sampling noise.  The bias is ``(bins - 1) * (1/n_current + 1/n_reference)``
    -- the usual chi-square degrees-of-freedom argument -- and
    ``tests/test_drift.py`` checks that formula against simulation at four
    sample sizes, where it is accurate to three decimal places.

    The number matters.  At this project's window of 200 frames against a
    400-frame reference with 10 bins, two samples of identical data read
    **0.067**, and the ninety-fifth percentile is **0.128**.  The conventional
    thresholds everybody quotes -- 0.1 warn, 0.25 alarm -- are for large
    samples; applied to a 200-frame window they fire on nothing at all roughly a
    third of the time.  That is not a threshold that needs tuning, it is a
    statistic that needs correcting.
    """
    return (bins - 1) * (1.0 / max(n_current, 1) + 1.0 / max(n_reference, 1))


def psi(reference: Sequence[float], current: Sequence[float], bins: int = 10,
        epsilon: float = 1e-4, bias_correct: bool = True) -> DetectorResult:
    """Population stability index, corrected for sample-size bias.

    Bin edges are the reference quantiles, so each reference bin holds the same
    mass and the index is not dominated by an empty tail bin.  ``epsilon``
    floors the proportions; without it a bin that is empty in the current window
    sends the index to infinity, and an infinite alarm carries no information
    about how bad the shift is.

    With ``bias_correct`` the statistic is the raw index minus
    :func:`psi_null_expectation`, floored at zero, so that zero means "no
    detectable shift" at any window size and the conventional 0.1 / 0.25
    thresholds keep the meaning they have in the literature.  The raw value is
    kept in ``detail`` so nothing is hidden.
    """
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size < bins or cur.size == 0:
        return DetectorResult("psi", "", float("nan"),
                              {"reason": "not enough samples",
                               "n_reference": int(ref.size), "n_current": int(cur.size)})
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if edges.size < 3:
        # A constant reference signal cannot drift in distribution; treat any
        # departure from the constant as a shift of the whole mass.
        moved = float(np.mean(cur != ref[0]))
        return DetectorResult("psi", "", moved,
                              {"reason": "degenerate reference", "moved_fraction": moved})
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_p = np.maximum(ref_counts / ref.size, epsilon)
    cur_p = np.maximum(cur_counts / cur.size, epsilon)
    raw = float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    null = psi_null_expectation(len(ref_counts), int(ref.size), int(cur.size))
    index = max(raw - null, 0.0) if bias_correct else raw
    return DetectorResult("psi", "", index, {
        "raw": round(raw, 6),
        "null_expectation": round(null, 6),
        "bias_corrected": bool(bias_correct),
        "bins": int(len(ref_counts)),
        "n_reference": int(ref.size),
        "n_current": int(cur.size),
    })


def ks(reference: Sequence[float], current: Sequence[float]) -> DetectorResult:
    """Two-sample Kolmogorov-Smirnov.  Statistic is D, the maximum CDF gap."""
    from scipy import stats

    ref = _clean(reference)
    cur = _clean(current)
    if ref.size < 2 or cur.size < 2:
        return DetectorResult("ks", "", float("nan"),
                              {"reason": "not enough samples"})
    outcome = stats.ks_2samp(ref, cur, alternative="two-sided", method="asymp")
    return DetectorResult("ks", "", float(outcome.statistic), {
        "p_value": float(outcome.pvalue),
        "n_reference": int(ref.size),
        "n_current": int(cur.size),
        "note": "p-value is reported but not thresholded; see the module docstring",
    })


def mean_shift(reference: Sequence[float], current: Sequence[float]) -> DetectorResult:
    """Absolute mean shift in units of the reference standard deviation."""
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size < 2 or cur.size < 1:
        return DetectorResult("mean_shift", "", float("nan"),
                              {"reason": "not enough samples"})
    ref_std = float(ref.std(ddof=1))
    difference = float(cur.mean() - ref.mean())
    if ref_std < 1e-12:
        statistic = 0.0 if abs(difference) < 1e-12 else float("inf")
    else:
        statistic = abs(difference) / ref_std
    return DetectorResult("mean_shift", "", statistic, {
        "reference_mean": round(float(ref.mean()), 6),
        "current_mean": round(float(cur.mean()), 6),
        "reference_std": round(ref_std, 8),
        "signed_shift": round(difference, 6),
        "direction": "up" if difference > 0 else "down",
    })


def rate_shift(reference: Sequence[float], current: Sequence[float]) -> DetectorResult:
    """Two-proportion z-test, for binary signals such as the flagged-defect rate.

    The model's own output rate is the one signal that needs no image features
    at all, and on a line whose true defect rate is stable it is the most direct
    evidence that something changed.  It is also the one an operations team
    notices first, so a monitor that does not watch it is behind the humans.
    """
    ref = _clean(reference)
    cur = _clean(current)
    if ref.size < 2 or cur.size < 2:
        return DetectorResult("rate_shift", "", float("nan"),
                              {"reason": "not enough samples"})
    p1, p2 = float(ref.mean()), float(cur.mean())
    n1, n2 = int(ref.size), int(cur.size)
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    denominator = (pooled * (1 - pooled) * (1 / n1 + 1 / n2)) ** 0.5
    statistic = 0.0 if denominator < 1e-12 else abs(p2 - p1) / denominator
    return DetectorResult("rate_shift", "", float(statistic), {
        "reference_rate": round(p1, 6),
        "current_rate": round(p2, 6),
        "n_reference": n1,
        "n_current": n2,
    })


DETECTORS: Dict[str, Callable[..., DetectorResult]] = {
    "psi": psi,
    "ks": ks,
    "mean_shift": mean_shift,
    "rate_shift": rate_shift,
}


class PageHinkley:
    """Sequential change detection on a stream of scalars.

    Accumulates the running deviation from the reference mean and fires when the
    accumulated deviation exceeds ``threshold`` above its own minimum.
    ``delta`` is the magnitude of change considered unremarkable, which is what
    keeps ordinary noise from accumulating into an alarm over a long enough run.

    One-sided by default and instantiated twice by
    :class:`~edgeinspect.monitoring.drift.DriftMonitor`, once per direction: a
    single two-sided accumulator cancels a signal that rises and then falls, and
    "brightness fell" is a different work order from "brightness rose".
    """

    def __init__(self, reference_mean: float, delta: float = 0.005,
                 threshold: float = 0.5, direction: str = "down"):
        if direction not in ("up", "down"):
            raise ValueError("direction must be 'up' or 'down'")
        self.reference_mean = float(reference_mean)
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.direction = direction
        self.cumulative = 0.0
        self.extreme = 0.0
        self.index = -1
        self.fired_at: Optional[int] = None

    def update(self, value: float) -> bool:
        """Feed one observation.  Returns True on the frame the alarm fires."""
        self.index += 1
        if not np.isfinite(value):
            return False
        deviation = (self.reference_mean - value) if self.direction == "down" else (
            value - self.reference_mean)
        self.cumulative += deviation - self.delta
        if self.fired_at is not None:
            return False
        self.extreme = min(self.extreme, self.cumulative)
        if self.cumulative - self.extreme > self.threshold:
            self.fired_at = self.index
            return True
        return False

    def run(self, values: Sequence[float]) -> Optional[int]:
        """Feed a whole series.  Returns the firing index, or ``None``."""
        for value in values:
            if self.update(float(value)):
                return self.fired_at
        return self.fired_at

    def state(self) -> dict:
        return {
            "direction": self.direction,
            "reference_mean": round(self.reference_mean, 6),
            "delta": self.delta,
            "threshold": self.threshold,
            "cumulative": round(self.cumulative, 6),
            "fired_at": self.fired_at,
            "frames_seen": self.index + 1,
        }
