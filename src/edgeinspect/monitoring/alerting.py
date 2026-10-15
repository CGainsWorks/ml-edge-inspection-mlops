"""Alerts: what a drift trigger turns into.

An alert has to be readable by whoever is on call, which means it says what
changed, by how much, in what direction, and what happens next.  "PSI 0.42 on
feature 3" is a number; "brightness fell 2.4 reference standard deviations and
sharpness fell 1.9; retraining triggered" is a work order.

Sinks are deliberately dull: a JSONL file and stdout.  Anything that posts to a
webhook or pages someone is a two-line subclass, and nothing in this repository
should be able to send a message to a real person by accident.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .. import paths
from .drift import DriftReport, WindowReport


@dataclass
class Alert:
    ts: str
    severity: str
    stream: str
    model_version: Optional[int]
    frame: int
    summary: str
    signals: List[dict] = field(default_factory=list)
    action: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        lines = [
            "[{0}] {1} on {2} at frame {3}".format(
                self.severity.upper(), self.summary, self.stream, self.frame),
        ]
        for signal in self.signals:
            lines.append(
                "    {0:<14} {1:<11} {2:>9.4f}  (warn {3}, alarm {4})  {5}".format(
                    signal["signal"], signal["detector"], signal["statistic"],
                    signal["warn"], signal["alarm"], signal.get("note", ""))
            )
        lines.append("    action: " + self.action)
        return "\n".join(lines)


def _describe(outcome: dict) -> str:
    detail = outcome.get("detail", {})
    if "signed_shift" in detail:
        return "{0} {1} by {2:+.4f} (reference {3:.4f})".format(
            outcome["signal"], detail.get("direction", "moved"),
            detail["signed_shift"], detail.get("reference_mean", float("nan")))
    if "current_rate" in detail:
        return "{0} {1:.3f} against reference {2:.3f}".format(
            outcome["signal"], detail["current_rate"], detail["reference_rate"])
    return ""


def from_window(window: WindowReport, stream: str, model_version: Optional[int],
                action: str) -> Alert:
    firing = [o.to_dict() for o in window.outcomes if o.severity in ("warn", "alarm")]
    for item in firing:
        item["note"] = _describe(item)
    worst = ", ".join(
        "{0}/{1}".format(o["detector"], o["signal"]) for o in firing[:3]
    ) or "no rule"
    summary = "drift detected ({0})".format(worst)
    return Alert(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        severity=window.severity,
        stream=stream,
        model_version=model_version,
        frame=window.end,
        summary=summary,
        signals=firing,
        action=action,
    )


class AlertSink:
    """Appends alerts to a JSONL file and optionally prints them."""

    def __init__(self, path: Optional[Path] = None, echo: bool = True):
        self.path = Path(path) if path else (paths.TELEMETRY / "alerts.jsonl")
        self.echo = bool(echo)
        self.alerts: List[Alert] = []

    def emit(self, alert: Alert) -> Alert:
        self.alerts.append(alert)
        paths.ensure(self.path.parent)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(alert.to_dict(), separators=(",", ":")) + "\n")
        if self.echo:
            print(alert.render(), flush=True)
        return alert

    def from_report(self, report: DriftReport, action: str = "retrain") -> List[Alert]:
        """Emit one alert per triggering window.

        Only triggering windows, not every window whose severity is above ok.
        A warn that the policy decided was not worth acting on is a line in the
        report, not a page; emitting it too is how an alert channel becomes
        something people mute.
        """
        emitted = []
        for window in report.windows:
            if window.triggered:
                emitted.append(self.emit(
                    from_window(window, report.stream, report.model_version, action)))
        return emitted


def summarise(report: DriftReport) -> Dict:
    """A compact operational summary of a scan, for the loop timeline and the docs."""
    firing_signals: Dict[str, str] = {}
    for window in report.windows:
        for outcome in window.outcomes:
            if outcome.severity == "ok":
                continue
            key = "{0}/{1}".format(outcome.rule.detector, outcome.rule.signal)
            current = firing_signals.get(key, "ok")
            if outcome.severity == "alarm" or current == "ok":
                firing_signals[key] = outcome.severity
    return {
        "stream": report.stream,
        "frames": report.frames,
        "triggered": report.triggered,
        "first_trigger_frame": report.first_trigger,
        "onset_frame": report.onset,
        "detection_latency_frames": report.detection_latency(),
        "sequential_earliest_frame": report.sequential.get("earliest_frame"),
        "windows": len(report.windows),
        "windows_triggered": sum(1 for w in report.windows if w.triggered),
        "firing_rules": firing_signals,
    }
