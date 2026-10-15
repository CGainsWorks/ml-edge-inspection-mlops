"""The drift monitor.

Takes a frozen reference window, a policy, and a stream of telemetry, and
answers two questions: is the production distribution still the one the model
was validated on, and if not, when did that stop being true.

The reference window
--------------------
Captured at deployment from the deployed model's own telemetry on known-good
frames, and stored *with the deployment*.  Two consequences that are the whole
reason for doing it this way:

* A new model version gets a new reference.  Score distributions are not
  comparable across versions, and a monitor that keeps comparing v2's scores to
  v1's reference fires on the deployment itself, every time, until somebody
  turns it off -- which is how monitoring gets turned off.
* The reference is frozen.  Comparing the current window to a rolling recent
  window is the mistake that makes a slow ramp invisible: nothing is ever
  different from ten minutes ago.

Windowed and sequential together
--------------------------------
Windowed tests answer "is it different now"; :class:`~edgeinspect.monitoring
.detectors.PageHinkley` answers "when did it start".  Both are run because
detection latency is a headline number for this project and a windowed test can
only ever locate an onset to within one stride.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import paths
from ..config import DriftPolicy, DriftRule
from .detectors import DETECTORS, PageHinkley
from .features import FEATURE_NAMES

SEVERITY_ORDER = {"ok": 0, "warn": 1, "alarm": 2}

# Signals the sequential detector is allowed to watch: the per-frame image
# statistics, which are unimodal and stationary under fixed imaging conditions.
# The model's own score and decision are excluded -- see DriftMonitor._sequential.
SEQUENTIAL_SIGNALS = frozenset(FEATURE_NAMES)


@dataclass
class RuleOutcome:
    rule: DriftRule
    statistic: float
    severity: str
    detail: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "detector": self.rule.detector,
            "signal": self.rule.signal,
            "statistic": round(float(self.statistic), 6),
            "warn": self.rule.warn,
            "alarm": self.rule.alarm,
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class WindowReport:
    start: int
    end: int
    outcomes: List[RuleOutcome]
    triggered: bool
    reason: str

    @property
    def alarms(self) -> List[RuleOutcome]:
        return [o for o in self.outcomes if o.severity == "alarm"]

    @property
    def warns(self) -> List[RuleOutcome]:
        return [o for o in self.outcomes if o.severity == "warn"]

    @property
    def severity(self) -> str:
        if self.alarms:
            return "alarm"
        if self.warns:
            return "warn"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "severity": self.severity,
            "triggered": self.triggered,
            "reason": self.reason,
            "n_alarm": len(self.alarms),
            "n_warn": len(self.warns),
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


@dataclass
class DriftReport:
    """Everything one scan found."""

    stream: str
    model_version: Optional[int]
    windows: List[WindowReport]
    sequential: Dict
    first_trigger: Optional[int]
    onset: Optional[int]
    frames: int

    @property
    def triggered(self) -> bool:
        return self.first_trigger is not None

    def detection_latency(self) -> Optional[int]:
        """Frames between the onset of the shift and the first trigger.

        Only meaningful in simulation, where the onset is known because it was
        applied.  On a real line nothing knows the onset, which is the argument
        for reporting the sequential detector's index as well: it is the
        closest thing to an onset estimate that production has.
        """
        if self.first_trigger is None or self.onset is None:
            return None
        return int(self.first_trigger - self.onset)

    def to_dict(self) -> dict:
        return {
            "stream": self.stream,
            "model_version": self.model_version,
            "frames": self.frames,
            "triggered": self.triggered,
            "first_trigger_frame": self.first_trigger,
            "onset_frame": self.onset,
            "detection_latency_frames": self.detection_latency(),
            "sequential": self.sequential,
            "windows": [w.to_dict() for w in self.windows],
        }

    def save(self, path: Path) -> Path:
        paths.ensure(Path(path).parent)
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        return Path(path)


class Reference:
    """The frozen distribution a deployment is compared against."""

    def __init__(self, signals: Dict[str, np.ndarray], meta: Optional[Dict] = None):
        self.signals = {k: np.asarray(v, dtype=np.float64) for k, v in signals.items()}
        self.meta = dict(meta or {})

    @property
    def size(self) -> int:
        return int(min((len(v) for v in self.signals.values()), default=0))

    def get(self, name: str) -> np.ndarray:
        if name not in self.signals:
            raise KeyError(
                "reference has no signal {0!r}.  It was built with: {1}.  A rule that "
                "watches a signal the reference does not carry cannot be evaluated, and "
                "silently skipping it would leave a monitor that looks armed."
                .format(name, sorted(self.signals))
            )
        return self.signals[name]

    @classmethod
    def from_records(cls, records: Sequence[dict], names: Sequence[str],
                     meta: Optional[Dict] = None) -> "Reference":
        from ..edge.telemetry import to_arrays

        return cls(to_arrays(records, names), meta=meta)

    def save(self, path: Path) -> Path:
        paths.ensure(Path(path).parent)
        payload = {
            "meta": self.meta,
            "signals": {k: [float(x) for x in v] for k, v in self.signals.items()},
        }
        with Path(path).open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> "Reference":
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(
                "no drift reference at {0}.  A reference is captured at deployment; "
                "run `make deploy` or scripts/capture_reference.py."
                .format(paths.relative(target))
            )
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        signals = {k: np.asarray(v, dtype=np.float64)
                   for k, v in payload.get("signals", {}).items()}
        return cls(signals, meta=payload.get("meta", {}))


class DriftMonitor:
    """Applies a :class:`~edgeinspect.config.DriftPolicy` to telemetry."""

    def __init__(self, policy: DriftPolicy, reference: Reference):
        self.policy = policy
        self.reference = reference
        missing = sorted({r.signal for r in policy.rules if r.enabled}
                         - set(reference.signals))
        if missing:
            raise ValueError(
                "the drift policy watches signal(s) {0} that the reference does not "
                "contain ({1}).  Rebuild the reference or fix configs/drift.yaml."
                .format(missing, sorted(reference.signals))
            )

    def evaluate(self, window: Dict[str, np.ndarray], start: int, end: int) -> WindowReport:
        """Run every enabled rule against one window of signals."""
        outcomes: List[RuleOutcome] = []
        for rule in self.policy.rules:
            if not rule.enabled:
                continue
            detector = DETECTORS.get(rule.detector)
            if detector is None:
                raise ValueError(
                    "unknown detector {0!r} in configs/drift.yaml.  Known: {1}"
                    .format(rule.detector, sorted(DETECTORS))
                )
            result = detector(self.reference.get(rule.signal), window[rule.signal])
            statistic = float(result.statistic)
            severity = "ok" if not np.isfinite(statistic) else rule.severity(statistic)
            detail = dict(result.detail)
            if not np.isfinite(statistic):
                detail["skipped"] = "statistic is not finite"
            outcomes.append(RuleOutcome(rule, statistic, severity, detail))

        alarms = sum(1 for o in outcomes if o.severity == "alarm")
        warns = sum(1 for o in outcomes if o.severity == "warn")
        triggered = alarms >= self.policy.min_alarms or warns >= self.policy.min_warns
        if triggered:
            names = ", ".join(
                "{0}/{1}={2:.3f}".format(o.rule.detector, o.rule.signal, o.statistic)
                for o in outcomes if o.severity in ("alarm", "warn")
            )
            reason = "{0} alarm(s), {1} warn(s): {2}".format(alarms, warns, names)
        else:
            reason = "{0} alarm(s), {1} warn(s): below trigger".format(alarms, warns)
        return WindowReport(start=start, end=end, outcomes=outcomes,
                            triggered=triggered, reason=reason)

    def scan(self, records: Sequence[dict], stream: str = "stream",
             onset: Optional[int] = None,
             model_version: Optional[int] = None) -> DriftReport:
        """Slide the policy window over a telemetry sequence."""
        from ..edge.telemetry import to_arrays

        names = sorted({r.signal for r in self.policy.rules if r.enabled})
        series = to_arrays(records, names)
        total = len(records)
        window = int(self.policy.window)
        stride = int(self.policy.stride)

        windows: List[WindowReport] = []
        first_trigger: Optional[int] = None
        cooldown = 0
        start = 0
        while start + window <= total:
            end = start + window
            chunk = {name: series[name][start:end] for name in names}
            report = self.evaluate(chunk, start, end)
            if cooldown > 0 and report.triggered:
                report.triggered = False
                report.reason += " (suppressed: cooldown)"
                cooldown -= 1
            elif report.triggered:
                if first_trigger is None:
                    first_trigger = end
                cooldown = int(self.policy.cooldown_windows)
            elif cooldown > 0:
                cooldown -= 1
            windows.append(report)
            start += stride

        sequential = self._sequential(series, names)
        return DriftReport(stream=stream, model_version=model_version, windows=windows,
                           sequential=sequential, first_trigger=first_trigger,
                           onset=onset, frames=total)

    def _sequential(self, series: Dict[str, np.ndarray], names: Sequence[str]) -> Dict:
        """Page-Hinkley per signal, both directions, from the reference mean.

        Both parameters are scaled by the reference standard deviation, so one
        configured sensitivity means the same thing for ``brightness`` (order
        0.5) and ``sharpness`` (order 1e-4).  Hard-coding an absolute threshold
        per signal is the alternative and it goes stale the first time the
        feature set changes.

        The specific values come from the statistic's false-alarm behaviour
        rather than from taste.  Under no drift the accumulator is a random walk
        with drift ``-delta`` and step deviation ``sigma``, and the largest
        excursion above its running minimum after ``n`` steps grows like
        ``sigma^2 / (2 * delta) * log(n)``.  With ``delta = 0.5 * sigma`` that is
        ``sigma * log(n)``, which reaches ``8 * sigma`` at about 3,000 frames --
        under two minutes at 30 fps.  A sequential detector that is guaranteed
        to fire eventually on a stationary signal is a clock, not a detector.
        ``delta = sigma`` halves the growth rate and ``threshold = 10 * sigma``
        raises the bar, putting the expected false alarm beyond 10^4 frames
        while still firing within a few frames of the multi-sigma shifts this
        project's domains produce.  ``tests/test_drift.py`` pins both ends.

        Only the image statistics are fed to it, never ``score`` or ``decision``.
        Page-Hinkley assumes a roughly symmetric noise process around a stable
        mean; the model's score distribution is a *mixture* -- a dense mass of
        good parts and a sparse high tail of defective ones -- so a run of
        defective parts arriving together looks exactly like a positive step.
        On the first end-to-end run this fired at frame 244 of a control stream
        that had not drifted at all, while every windowed rule sat at zero.  The
        windowed tests handle ``score`` correctly because they compare whole
        distributions rather than a running mean.
        """
        out: Dict[str, Dict] = {}
        for name in names:
            if name not in SEQUENTIAL_SIGNALS:
                continue
            reference = self.reference.get(name)
            reference = reference[np.isfinite(reference)]
            if reference.size < 2:
                continue
            mean = float(reference.mean())
            std = float(reference.std(ddof=1))
            if std < 1e-12:
                continue
            entry: Dict[str, Optional[int]] = {}
            for direction in ("down", "up"):
                detector = PageHinkley(
                    reference_mean=mean,
                    delta=1.0 * std,        # a one-sigma move is unremarkable
                    threshold=10.0 * std,   # sustained shift, not a spike
                    direction=direction,
                )
                entry[direction] = detector.run(series[name])
            fired = [v for v in entry.values() if v is not None]
            entry["first"] = min(fired) if fired else None
            out[name] = entry
        earliest = [v["first"] for v in out.values() if v.get("first") is not None]
        return {"per_signal": out, "earliest_frame": min(earliest) if earliest else None}


def build_reference(records: Sequence[dict], policy: DriftPolicy,
                    meta: Optional[Dict] = None, limit: Optional[int] = None) -> Reference:
    """Build a reference from telemetry, carrying exactly the signals the policy needs."""
    names = sorted({r.signal for r in policy.rules if r.enabled})
    subset = list(records)[:limit] if limit else list(records)
    if len(subset) < policy.window:
        raise ValueError(
            "reference needs at least {0} frames (the policy window) and has {1}.  "
            "A reference shorter than the comparison window makes every statistic "
            "unstable and the monitor unreliable in the direction that matters."
            .format(policy.window, len(subset))
        )
    reference = Reference.from_records(subset, names, meta=meta)
    return reference
