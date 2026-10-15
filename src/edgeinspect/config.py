"""YAML configuration loading.

Configs are plain dictionaries with a small amount of structure enforced where
it matters: the dataset domain, because a typo there silently changes what the
model is trained on, and the drift policy, because a policy that parses but
means nothing is worse than one that fails loudly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import paths


def load_yaml(path) -> dict:
    """Read a YAML file into a dictionary."""
    resolved = paths.resolve(path)
    if not resolved.exists():
        raise FileNotFoundError("config not found: " + paths.relative(resolved))
    with resolved.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            paths.relative(resolved) + " must contain a mapping at the top level"
        )
    return data


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` onto ``base`` without mutating either."""
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], Mapping) and isinstance(value, Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class DomainSpec:
    """The imaging conditions a batch of frames was captured under.

    This is the knob the whole drift story turns on.  The training set is
    captured under the ``reference`` domain; a production stream captured under
    a different domain is exactly what a dirty lens, an ageing lamp or a
    changed material batch looks like to a camera.

    Every field is a physical quantity rather than a pixel operation, so the
    drifted domains in ``configs/data.yaml`` read as a description of what went
    wrong on the line.
    """

    name: str = "reference"
    gain: float = 1.0              # illumination gain, 1.0 is nominal exposure
    offset: float = 0.0            # additive black level shift
    blur_sigma: float = 0.6        # optical blur in pixels: focus and lens dirt
    noise_sigma: float = 0.020     # sensor read noise, standard deviation
    grain_scale: float = 7.0       # surface texture correlation length in pixels
    grain_strength: float = 0.16   # surface texture contrast
    ripple_strength: float = 0.05  # directional machining marks
    ripple_period: float = 11.0    # machining mark spacing in pixels
    vignette: float = 0.06         # radial illumination falloff, 0 is flat
    defect_contrast: float = 0.30  # how far a defect departs from the surface
    defect_rate: float = 0.5       # fraction of frames containing a defect

    def to_dict(self) -> dict:
        out = {}
        for key in self.__dataclass_fields__:
            out[key] = getattr(self, key)
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DomainSpec":
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                "unknown domain field(s): {0}.  Known fields: {1}".format(
                    sorted(unknown), sorted(known)
                )
            )
        return cls(**dict(data))

    def blend(self, other: "DomainSpec", alpha: float) -> "DomainSpec":
        """Linear interpolation towards ``other``.

        Used to simulate gradual drift, which is the case a threshold detector
        finds hardest: nothing is ever suddenly wrong.
        """
        alpha = float(min(max(alpha, 0.0), 1.0))
        numeric = {}
        for key in self.__dataclass_fields__:
            if key == "name":
                continue
            a = getattr(self, key)
            b = getattr(other, key)
            numeric[key] = a + (b - a) * alpha
        name = "{0}->{1}@{2:.2f}".format(self.name, other.name, alpha)
        return replace(self, name=name, **numeric)


@dataclass(frozen=True)
class DriftRule:
    """One statistical test, the signal it watches, and where its alarm sits."""

    detector: str
    signal: str
    warn: float
    alarm: float
    direction: str = "above"  # "above": a larger statistic is worse
    enabled: bool = True

    def severity(self, statistic: float) -> str:
        if self.direction == "above":
            worse = statistic >= self.alarm, statistic >= self.warn
        else:
            worse = statistic <= self.alarm, statistic <= self.warn
        if worse[0]:
            return "alarm"
        if worse[1]:
            return "warn"
        return "ok"


@dataclass(frozen=True)
class DriftPolicy:
    """When a collection of firing rules becomes a retraining trigger."""

    rules: tuple = field(default_factory=tuple)
    window: int = 200
    stride: int = 50
    min_alarms: int = 1
    min_warns: int = 3
    cooldown_windows: int = 2

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DriftPolicy":
        raw_rules = data.get("rules")
        if not raw_rules:
            raise ValueError(
                "drift policy has no rules.  A monitor with no rules never fires, "
                "which looks like a working monitor and is not one."
            )
        rules = tuple(DriftRule(**rule) for rule in raw_rules)
        for rule in rules:
            if rule.direction not in ("above", "below"):
                raise ValueError(
                    "rule {0}/{1}: direction must be 'above' or 'below', got {2!r}".format(
                        rule.detector, rule.signal, rule.direction
                    )
                )
        trigger = dict(data.get("trigger", {}))
        return cls(
            rules=rules,
            window=int(data.get("window", 200)),
            stride=int(data.get("stride", 50)),
            min_alarms=int(trigger.get("min_alarms", 1)),
            min_warns=int(trigger.get("min_warns", 3)),
            cooldown_windows=int(trigger.get("cooldown_windows", 2)),
        )


def load_domains(config: Mapping[str, Any]) -> dict:
    """Read the ``domains`` block of ``configs/data.yaml``.

    Domains inherit from ``reference`` so a drifted domain only has to state
    what actually changed.
    """
    raw = dict(config.get("domains", {}))
    if "reference" not in raw:
        raise ValueError("configs/data.yaml must define a 'reference' domain")
    base_fields = dict(raw["reference"])
    base_fields["name"] = "reference"
    base = DomainSpec.from_dict(base_fields)
    domains = {"reference": base}
    for name, spec in raw.items():
        if name == "reference":
            continue
        merged = base.to_dict()
        merged.update(dict(spec))
        merged["name"] = name
        domains[name] = DomainSpec.from_dict(merged)
    return domains


def load_drift_policy(path=None) -> DriftPolicy:
    """Load ``configs/drift.yaml`` (or an override) into a :class:`DriftPolicy`."""
    target = Path(path) if path else (paths.CONFIGS / "drift.yaml")
    return DriftPolicy.from_dict(load_yaml(target))
