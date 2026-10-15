"""The drift detectors and the monitor.

The important test in this file is
:func:`test_monitor_does_not_fire_on_the_control_stream`.  Every other test here
checks that the monitor can detect something; that one checks that it can decline
to, which is the property that decides whether anybody leaves it switched on.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgeinspect.config import DriftPolicy, DriftRule
from edgeinspect.monitoring import detectors
from edgeinspect.monitoring.drift import DriftMonitor, Reference, build_reference


# ---------------------------------------------------------------------------
# Individual detectors
# ---------------------------------------------------------------------------

def test_psi_is_near_zero_for_the_same_distribution(rng):
    reference = rng.normal(0, 1, 4000)
    current = rng.normal(0, 1, 4000)
    assert detectors.psi(reference, current).statistic < 0.05


def test_psi_null_expectation_matches_simulation():
    """The bias correction is only worth having if the formula is right."""
    for n_reference, n_current, bins in ((400, 100, 10), (400, 200, 10),
                                         (2000, 200, 10), (400, 200, 5)):
        raw = []
        for seed in range(160):
            local = np.random.default_rng(seed)
            result = detectors.psi(local.normal(0, 1, n_reference),
                                   local.normal(0, 1, n_current),
                                   bins=bins, bias_correct=False)
            raw.append(result.statistic)
        predicted = detectors.psi_null_expectation(bins, n_reference, n_current)
        assert float(np.mean(raw)) == pytest.approx(predicted, rel=0.15), (
            'bins={0} ref={1} cur={2}'.format(bins, n_reference, n_current))


def test_bias_correction_makes_the_conventional_thresholds_usable():
    """At a 200-frame window the uncorrected index clears its own warn line.

    This is the bug the correction exists for: with a 400-frame reference and
    a 200-frame window, two samples of identical data read about 0.067 raw,
    with a ninety-fifth percentile above the conventional 0.1 warn
    threshold.  The monitor then fires on nothing at all.
    """
    raw, corrected = [], []
    for seed in range(200):
        local = np.random.default_rng(seed)
        reference = local.normal(0.49, 0.02, 400)
        current = local.normal(0.49, 0.02, 200)
        raw.append(detectors.psi(reference, current, bias_correct=False).statistic)
        corrected.append(detectors.psi(reference, current).statistic)
    assert float(np.percentile(raw, 95)) > 0.10, (
        'the uncorrected index no longer reproduces the problem')
    assert float(np.percentile(corrected, 95)) < 0.10
    assert float(np.mean(corrected)) < 0.02


def test_bias_correction_keeps_a_real_shift_visible(rng):
    reference = rng.normal(0, 1, 400)
    shifted = rng.normal(1.5, 1, 200)
    assert detectors.psi(reference, shifted).statistic > 0.25


def test_psi_grows_with_the_size_of_the_shift(rng):
    reference = rng.normal(0, 1, 4000)
    small = detectors.psi(reference, rng.normal(0.2, 1, 4000)).statistic
    large = detectors.psi(reference, rng.normal(1.5, 1, 4000)).statistic
    assert small < large
    assert large > 0.25, "a 1.5-sigma shift should clear the conventional alarm line"


def test_psi_is_finite_when_a_bin_empties(rng):
    """An empty bin sends the textbook formula to infinity; epsilon floors it."""
    reference = rng.normal(0, 1, 2000)
    current = rng.normal(20, 0.01, 2000)
    value = detectors.psi(reference, current).statistic
    assert np.isfinite(value) and value > 1.0


def test_psi_handles_a_constant_reference():
    result = detectors.psi(np.zeros(500), np.ones(500))
    assert np.isfinite(result.statistic)
    assert result.detail.get("reason") == "degenerate reference"


def test_ks_separates_shifted_distributions(rng):
    reference = rng.normal(0, 1, 500)
    assert detectors.ks(reference, rng.normal(0, 1, 500)).statistic < 0.12
    assert detectors.ks(reference, rng.normal(1.0, 1, 500)).statistic > 0.3


def test_mean_shift_is_in_reference_sigmas(rng):
    reference = rng.normal(10.0, 2.0, 5000)
    result = detectors.mean_shift(reference, rng.normal(14.0, 2.0, 5000))
    assert result.statistic == pytest.approx(2.0, rel=0.1)
    assert result.detail["direction"] == "up"


def test_mean_shift_reports_direction_downwards(rng):
    reference = rng.normal(10.0, 2.0, 2000)
    assert detectors.mean_shift(reference, rng.normal(6.0, 2.0, 2000)
                                ).detail["direction"] == "down"


def test_rate_shift_on_binary_signals():
    reference = np.zeros(500)
    reference[:40] = 1              # 8%
    unchanged = np.zeros(500)
    unchanged[:42] = 1              # 8.4%
    changed = np.zeros(500)
    changed[:150] = 1               # 30%
    assert detectors.rate_shift(reference, unchanged).statistic < 2.0
    assert detectors.rate_shift(reference, changed).statistic > 5.0


def test_detectors_return_nan_rather_than_raising_on_short_input():
    for detector in (detectors.psi, detectors.ks, detectors.mean_shift,
                     detectors.rate_shift):
        assert not np.isfinite(detector([1.0], [2.0]).statistic)


# ---------------------------------------------------------------------------
# Page-Hinkley
# ---------------------------------------------------------------------------

def test_page_hinkley_fires_after_a_step_and_reports_where(rng):
    stream = np.concatenate([rng.normal(1.0, 0.05, 300), rng.normal(0.6, 0.05, 300)])
    detector = detectors.PageHinkley(reference_mean=1.0, delta=0.05, threshold=0.5,
                                     direction="down")
    fired = detector.run(stream)
    assert fired is not None
    assert 300 <= fired < 380, "fired at {0}, expected shortly after the step".format(fired)


def test_page_hinkley_does_not_fire_on_a_stationary_stream(rng):
    """The parameters have to beat the log-growth of the null excursion.

    Under no drift the accumulator's largest excursion above its running
    minimum grows like sigma^2 / (2 delta) * log(n).  With delta = sigma the
    growth is sigma/2 * log(n), so 3,000 steps of sigma 0.05 reach about
    0.2 against a threshold of 0.5.  With delta = sigma/2 -- which is what
    this project used first -- the same run reaches the threshold and the
    detector fires on stationary data.
    """
    stream = rng.normal(1.0, 0.05, 3000)
    detector = detectors.PageHinkley(reference_mean=1.0, delta=0.05, threshold=0.5,
                                     direction="down")
    assert detector.run(stream) is None


def test_page_hinkley_false_alarm_time_grows_with_delta(rng):
    """The trade the parameter choice is making, measured."""
    stream = rng.normal(1.0, 0.05, 20000)
    loose = detectors.PageHinkley(1.0, delta=0.02, threshold=0.5, direction='down')
    tight = detectors.PageHinkley(1.0, delta=0.05, threshold=0.5, direction='down')
    loose_fired = loose.run(stream)
    tight_fired = tight.run(stream)
    assert loose_fired is not None, 'a loose delta should eventually false alarm'
    assert tight_fired is None or tight_fired > loose_fired


def test_page_hinkley_is_one_sided():
    """Two accumulators, one per direction: a rise and a fall are different faults."""
    rising = np.concatenate([np.full(200, 1.0), np.full(200, 1.5)])
    down = detectors.PageHinkley(1.0, delta=0.02, threshold=0.3, direction="down")
    up = detectors.PageHinkley(1.0, delta=0.02, threshold=0.3, direction="up")
    assert down.run(rising) is None
    assert up.run(rising) is not None


def test_page_hinkley_rejects_a_bad_direction():
    with pytest.raises(ValueError):
        detectors.PageHinkley(0.0, direction="sideways")


# ---------------------------------------------------------------------------
# The monitor
# ---------------------------------------------------------------------------

def synthetic_records(count, brightness, score, seed=0, decision_rate=0.08):
    rng = np.random.default_rng(seed)
    records = []
    for index in range(count):
        records.append({
            "frame_id": index,
            "stream": "synthetic",
            "score": float(rng.normal(score, 0.2)),
            "decision": int(rng.random() < decision_rate),
            "features": {
                "brightness": float(rng.normal(brightness, 0.02)),
                "contrast": float(rng.normal(0.16, 0.002)),
            },
        })
    return records


def simple_policy(window=100, stride=50, min_alarms=1, min_warns=2):
    return DriftPolicy(
        rules=(
            DriftRule("psi", "brightness", warn=0.1, alarm=0.25),
            DriftRule("mean_shift", "brightness", warn=1.0, alarm=2.0),
            DriftRule("ks", "score", warn=0.15, alarm=0.25),
        ),
        window=window, stride=stride, min_alarms=min_alarms, min_warns=min_warns,
        cooldown_windows=1,
    )


def test_monitor_does_not_fire_on_the_control_stream():
    """Same conditions, different draws.  A trigger here is a false alarm."""
    policy = simple_policy()
    reference = build_reference(synthetic_records(400, 0.49, 0.1, seed=1), policy)
    monitor = DriftMonitor(policy, reference)
    report = monitor.scan(synthetic_records(600, 0.49, 0.1, seed=2), stream="control")
    assert not report.triggered, report.windows[0].reason
    assert report.detection_latency() is None


def test_monitor_fires_on_a_shifted_stream():
    policy = simple_policy()
    reference = build_reference(synthetic_records(400, 0.49, 0.1, seed=1), policy)
    monitor = DriftMonitor(policy, reference)
    report = monitor.scan(synthetic_records(600, 0.39, 0.6, seed=3), stream="drifted",
                          onset=0)
    assert report.triggered
    assert report.first_trigger is not None
    assert report.detection_latency() == report.first_trigger


def test_cooldown_suppresses_repeat_triggers():
    policy = simple_policy(window=100, stride=50)
    reference = build_reference(synthetic_records(400, 0.49, 0.1, seed=1), policy)
    monitor = DriftMonitor(policy, reference)
    report = monitor.scan(synthetic_records(800, 0.39, 0.6, seed=4), stream="drifted")
    triggered = [w for w in report.windows if w.triggered]
    suppressed = [w for w in report.windows if "cooldown" in w.reason]
    assert triggered, "nothing fired at all"
    assert suppressed, "every window fired; the cooldown did nothing"


def test_monitor_refuses_a_reference_missing_a_watched_signal():
    policy = simple_policy()
    reference = Reference({"brightness": np.zeros(300)})
    with pytest.raises(ValueError, match="score"):
        DriftMonitor(policy, reference)


def test_reference_shorter_than_the_window_is_rejected():
    policy = simple_policy(window=200)
    with pytest.raises(ValueError, match="at least 200"):
        build_reference(synthetic_records(50, 0.49, 0.1), policy)


def test_policy_without_rules_is_rejected():
    with pytest.raises(ValueError, match="no rules"):
        DriftPolicy.from_dict({"window": 100})


def test_policy_rejects_an_unknown_direction():
    with pytest.raises(ValueError, match="direction"):
        DriftPolicy.from_dict({"rules": [
            {"detector": "psi", "signal": "brightness", "warn": 0.1, "alarm": 0.2,
             "direction": "sideways"}]})


def test_monitor_rejects_an_unknown_detector():
    policy = DriftPolicy(rules=(DriftRule("telepathy", "brightness", 0.1, 0.2),),
                         window=50, stride=25)
    reference = Reference({"brightness": np.zeros(100) + np.arange(100) * 1e-3})
    monitor = DriftMonitor(policy, reference)
    with pytest.raises(ValueError, match="unknown detector"):
        monitor.scan(synthetic_records(100, 0.49, 0.1), stream="x")


def test_reference_round_trips_through_disk(tmp_path):
    policy = simple_policy()
    reference = build_reference(synthetic_records(400, 0.49, 0.1, seed=1), policy,
                                meta={"model_version": 3})
    path = reference.save(tmp_path / "reference.json")
    loaded = Reference.load(path)
    assert loaded.meta["model_version"] == 3
    for name in reference.signals:
        assert np.allclose(loaded.get(name), reference.get(name))


def test_missing_reference_says_how_to_make_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="captured at deployment"):
        Reference.load(tmp_path / "nothing.json")
