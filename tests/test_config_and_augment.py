"""Configuration loading, domain inheritance, and the augmentation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinspect import paths
from edgeinspect.config import (
    DomainSpec,
    DriftPolicy,
    deep_merge,
    load_domains,
    load_drift_policy,
    load_yaml,
)
from edgeinspect.data import AugmentConfig
from edgeinspect.data import augment as augment_module


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_deep_merge_does_not_mutate_its_inputs():
    base = {"a": {"b": 1}, "c": 2}
    override = {"a": {"d": 3}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"b": 1, "d": 3}, "c": 2}
    assert base == {"a": {"b": 1}, "c": 2}


def test_missing_config_says_which_one():
    with pytest.raises(FileNotFoundError, match="config not found"):
        load_yaml("configs/does_not_exist.yaml")


def test_domains_inherit_from_the_reference(domains):
    """A drifted domain states what changed; everything else comes from reference."""
    reference = domains["reference"]
    drifted = domains["lamp_ageing"]
    assert drifted.gain != reference.gain
    assert drifted.grain_scale == reference.grain_scale


def test_an_unknown_domain_field_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="unknown domain field"):
        DomainSpec.from_dict({"name": "typo", "briteness": 0.5})


def test_a_config_without_a_reference_domain_is_rejected():
    with pytest.raises(ValueError, match="must define a 'reference' domain"):
        load_domains({"domains": {"drifted": {"gain": 0.5}}})


def test_blend_interpolates_and_records_where_it_is():
    a = DomainSpec(name="a", gain=1.0, blur_sigma=0.5)
    b = DomainSpec(name="b", gain=0.5, blur_sigma=1.5)
    half = a.blend(b, 0.5)
    assert half.gain == pytest.approx(0.75)
    assert half.blur_sigma == pytest.approx(1.0)
    assert "a->b@0.50" == half.name


def test_blend_clamps_outside_the_unit_interval():
    a = DomainSpec(name="a", gain=1.0)
    b = DomainSpec(name="b", gain=0.5)
    assert a.blend(b, -3).gain == pytest.approx(1.0)
    assert a.blend(b, 7).gain == pytest.approx(0.5)


def test_the_shipped_drift_policy_parses_and_has_teeth():
    policy = load_drift_policy()
    assert policy.rules, "the shipped policy has no rules"
    assert policy.window > 0 and policy.stride > 0
    assert policy.min_alarms >= 1
    for rule in policy.rules:
        assert rule.warn <= rule.alarm, (
            "{0}/{1}: the warn threshold is above the alarm threshold".format(
                rule.detector, rule.signal))


def test_every_watched_signal_is_something_telemetry_records():
    """A rule watching a signal nobody emits is a rule that never fires."""
    from edgeinspect.monitoring.features import FEATURE_NAMES

    top_level = {"score", "decision", "latency_ms", "n_boxes"}
    available = set(FEATURE_NAMES) | top_level
    for rule in load_drift_policy().rules:
        assert rule.signal in available, (
            "configs/drift.yaml watches {0!r}, which telemetry never records"
            .format(rule.signal))


def test_severity_ladder():
    from edgeinspect.config import DriftRule

    rule = DriftRule("psi", "brightness", warn=0.1, alarm=0.25)
    assert rule.severity(0.05) == "ok"
    assert rule.severity(0.15) == "warn"
    assert rule.severity(0.30) == "alarm"


def test_severity_ladder_downwards():
    from edgeinspect.config import DriftRule

    rule = DriftRule("psi", "x", warn=0.9, alarm=0.5, direction="below")
    assert rule.severity(0.95) == "ok"
    assert rule.severity(0.7) == "warn"
    assert rule.severity(0.4) == "alarm"


def test_drift_policy_trigger_defaults_are_explicit():
    policy = DriftPolicy.from_dict({
        "rules": [{"detector": "psi", "signal": "brightness", "warn": 0.1, "alarm": 0.2}]})
    assert policy.min_alarms == 1 and policy.min_warns == 3


def test_benchmark_config_points_at_artefacts_the_makefile_builds():
    """A benchmark variant naming a file nothing produces is silently skipped."""
    config = load_yaml(paths.CONFIGS / "benchmark.yaml")
    makefile = (paths.ROOT / "Makefile").read_text(encoding="utf-8")
    for variant in config["variants"]:
        stem = variant["model"].split("/")[-1].replace(".onnx", "")
        suffix = stem.replace("surface_defect", "").lstrip("_") or "fp32"
        assert suffix in makefile or suffix in ("fp32",), (
            "configs/benchmark.yaml expects {0}, which no Makefile target builds"
            .format(variant["model"]))


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def test_geometry_is_applied_to_the_image_and_the_mask_together(rng):
    """Flipping the image and not the mask trains the wrong thing, quietly."""
    image = rng.random((1, 16, 16)).astype(np.float32)
    mask = np.zeros((1, 16, 16), dtype=np.float32)
    mask[0, 2:5, 3:6] = 1.0
    config = AugmentConfig(hflip=True, vflip=True, rot90=True,
                           brightness=0.0, contrast=0.0, noise=0.0, blur=0.0)
    for seed in range(30):
        out_image, out_mask = augment_module.apply(
            image, mask, config, np.random.default_rng(seed))
        # Geometry only: the pixel values are a permutation of the originals,
        # and the mask must have moved with them.
        assert sorted(out_image.reshape(-1).tolist()) == pytest.approx(
            sorted(image.reshape(-1).tolist()))
        hot = out_image[out_mask > 0.5]
        original_hot = image[mask > 0.5]
        assert sorted(hot.tolist()) == pytest.approx(sorted(original_hot.tolist())), (
            "the mask no longer covers the same pixels after augmentation")


def test_photometric_augmentation_leaves_the_mask_alone(rng):
    image = rng.random((1, 16, 16)).astype(np.float32)
    mask = np.zeros((1, 16, 16), dtype=np.float32)
    mask[0, 4:8, 4:8] = 1.0
    config = AugmentConfig(hflip=False, vflip=False, rot90=False,
                           brightness=0.2, contrast=0.2, noise=0.05, blur=0.5)
    _out_image, out_mask = augment_module.apply(image, mask, config, rng)
    assert np.array_equal(out_mask, mask)


def test_augmentation_stays_inside_the_unit_interval(rng):
    image = rng.random((1, 16, 16)).astype(np.float32)
    mask = np.zeros((1, 16, 16), dtype=np.float32)
    config = AugmentConfig(brightness=0.5, contrast=0.5, noise=0.3, blur=1.0)
    for seed in range(20):
        out, _ = augment_module.apply(image, mask, config, np.random.default_rng(seed))
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_disabled_augmentation_is_the_identity(rng):
    image = rng.random((1, 8, 8)).astype(np.float32)
    mask = rng.random((1, 8, 8)).astype(np.float32)
    out_image, out_mask = augment_module.apply(
        image, mask, AugmentConfig(enabled=False), rng)
    assert out_image is image and out_mask is mask


def test_unknown_augmentation_field_is_rejected():
    with pytest.raises(ValueError, match="unknown augmentation field"):
        AugmentConfig.from_dict({"hfilp": True})


def test_augmentation_is_narrower_than_the_drift_it_must_not_hide(domains):
    """The relationship the whole drift experiment depends on.

    If training augmentation spans the lamp_ageing shift, the model is invariant
    to it, the drift detector has nothing to detect that matters, and the
    recovery this project demonstrates cannot be demonstrated.  Widening the
    augmentation ranges past the drift has to fail a test rather than quietly
    make the headline result unmeasurable.
    """
    train_config = load_yaml(paths.CONFIGS / "train.yaml")
    augment = AugmentConfig.from_dict(train_config["augment"])
    reference = domains["reference"]
    drifted = domains["lamp_ageing"]

    brightness_shift = abs(reference.gain - drifted.gain) * 0.5 + abs(
        reference.offset - drifted.offset)
    assert augment.brightness < brightness_shift, (
        "augmentation brightness {0} covers the {1:.3f} illumination shift"
        .format(augment.brightness, brightness_shift))

    extra_blur = max(drifted.blur_sigma ** 2 - reference.blur_sigma ** 2, 0.0) ** 0.5
    assert augment.blur < extra_blur, (
        "augmentation blur {0} covers the {1:.3f} of extra blur the drift adds"
        .format(augment.blur, extra_blur))

    extra_noise = max(drifted.noise_sigma ** 2 - reference.noise_sigma ** 2, 0.0) ** 0.5
    assert augment.noise < extra_noise, (
        "augmentation noise {0} covers the {1:.3f} of extra sensor noise"
        .format(augment.noise, extra_noise))


def test_the_ci_gate_has_the_same_shape_as_the_published_one():
    """Different numbers are legitimate; a different gate is not.

    ``configs/deploy_tiny.yaml`` exists because an absolute accuracy floor is a
    property of the dataset and CI trains on a tenth of the data.  That is a
    reason to move the thresholds, not to drop a check -- so the two configs
    must agree on which checks exist and on which metric each one watches.
    """
    full = load_yaml(paths.CONFIGS / "deploy.yaml")
    tiny = load_yaml(paths.CONFIGS / "deploy_tiny.yaml")

    assert set(full["gate"]) == set(tiny["gate"]), (
        "the two gates do not have the same kinds of check")
    for kind in ("improvement", "regression"):
        assert full["gate"][kind]["metric"] == tiny["gate"][kind]["metric"], kind
    assert ([floor["metric"] for floor in full["gate"]["floors"]]
            == [floor["metric"] for floor in tiny["gate"]["floors"]]), (
        "the two gates do not apply floors to the same metrics")
    assert set(full["canary"]) == set(tiny["canary"]), (
        "the two canaries do not check the same conditions")
    assert full["model_name"] == tiny["model_name"]
    assert full["artefact"] == tiny["artefact"]


def test_the_ci_gate_is_looser_but_not_absent():
    """A relaxed threshold is a judgement; a disabled one is a hole."""
    full = load_yaml(paths.CONFIGS / "deploy.yaml")["gate"]
    tiny = load_yaml(paths.CONFIGS / "deploy_tiny.yaml")["gate"]

    # The improvement requirement is the one that proves recovery happened, so
    # it must not be weakened at all.
    assert tiny["improvement"]["min_delta"] >= full["improvement"]["min_delta"]

    # The others may be looser, and must still bite.
    assert full["regression"]["max_drop"] <= tiny["regression"]["max_drop"] < 0.2
    for full_floor, tiny_floor in zip(full["floors"], tiny["floors"]):
        if "min" in full_floor:
            assert 0.5 < tiny_floor["min"] <= full_floor["min"]
        if "max" in full_floor:
            assert full_floor["max"] <= tiny_floor["max"] < 0.5


def test_the_ci_reference_window_is_at_least_the_policy_window():
    """A reference shorter than the window it is compared against is unstable."""
    policy = load_drift_policy()
    for name in ("deploy.yaml", "deploy_tiny.yaml"):
        config = load_yaml(paths.CONFIGS / name)
        assert config["reference_frames"] >= policy.window, (
            "{0} captures {1} reference frames for a {2}-frame window"
            .format(name, config["reference_frames"], policy.window))
