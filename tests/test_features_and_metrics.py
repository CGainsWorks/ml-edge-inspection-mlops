"""Telemetry features and the metric implementations.

Both are hand-rolled here rather than imported, so both get checked against
something independent: the features against synthetic signals whose answer is
known analytically, and the metrics against brute-force reference
implementations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from edgeinspect.metrics import classification, localisation
from edgeinspect.monitoring import features


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.01, 0.02, 0.05, 0.1])
def test_noise_estimator_recovers_a_known_standard_deviation(sigma):
    """Immerkaer's estimator, against noise whose sigma we chose."""
    rng = np.random.default_rng(int(sigma * 10000))
    frame = np.clip(0.5 + rng.standard_normal((256, 256)).astype(np.float32) * sigma, 0, 1)
    estimated = features.extract(frame)["noise_level"]
    assert estimated == pytest.approx(sigma, rel=0.06), (
        "estimated {0:.5f} against a true {1:.5f}".format(estimated, sigma))


def test_noise_estimator_is_not_fooled_by_structure():
    """A smooth gradient is not noise, however large its dynamic range."""
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    ramp = ((xx + yy) / 512.0).astype(np.float32)
    assert features.extract(ramp)["noise_level"] < 0.002


def test_sharpness_subtracts_the_noise_floor():
    """Pure noise must not read as structure, whatever its amplitude."""
    rng = np.random.default_rng(3)
    for sigma in (0.01, 0.05, 0.12):
        frame = np.clip(0.5 + rng.standard_normal((256, 256)).astype(np.float32) * sigma, 0, 1)
        assert features.extract(frame)["sharpness"] < 1e-4


def test_sharpness_falls_under_blur_even_when_noise_rises():
    """The failure that motivated the noise correction, pinned as a test.

    A dying lamp both softens the image and raises sensor gain.  A naive
    Laplacian-variance focus measure rises in that situation, and an alert
    saying sharpness increased while the image blurs is worse than no alert.
    """
    rng = np.random.default_rng(11)
    structure = ndimage.gaussian_filter(
        rng.standard_normal((256, 256)).astype(np.float32), 3.0)
    structure = (structure / structure.std()) * 0.15 + 0.5

    crisp = np.clip(structure + rng.standard_normal(structure.shape).astype(np.float32) * 0.02,
                    0, 1)
    blurred_structure = ndimage.gaussian_filter(structure, 1.6)
    noisy_blurred = np.clip(
        blurred_structure + rng.standard_normal(structure.shape).astype(np.float32) * 0.045,
        0, 1)

    crisp_features = features.extract(crisp)
    blurred_features = features.extract(noisy_blurred)
    assert blurred_features["sharpness"] < crisp_features["sharpness"]
    assert blurred_features["noise_level"] > crisp_features["noise_level"]


def test_sharpness_noise_gain_constant_is_the_right_number():
    """32/81 is the squared-coefficient sum of the box-then-Laplacian kernel."""
    from scipy.signal import convolve2d

    combined = convolve2d(features._LAPLACIAN, features._BOX)
    assert float((combined ** 2).sum()) == pytest.approx(features._SHARPNESS_NOISE_GAIN)


def test_brightness_and_contrast_are_mean_and_standard_deviation(rng):
    frame = rng.random((64, 64)).astype(np.float32)
    extracted = features.extract(frame)
    assert extracted["brightness"] == pytest.approx(float(frame.mean()), abs=1e-6)
    assert extracted["contrast"] == pytest.approx(float(frame.std()), abs=1e-6)


def test_every_declared_feature_is_produced(rng):
    extracted = features.extract(rng.random((64, 64)).astype(np.float32))
    assert set(extracted) == set(features.FEATURE_NAMES)
    assert all(np.isfinite(value) for value in extracted.values())


def test_extract_accepts_a_channel_dimension(rng):
    frame = rng.random((1, 64, 64)).astype(np.float32)
    assert features.extract(frame) == features.extract(frame[0])


def test_extract_rejects_the_wrong_rank(rng):
    with pytest.raises(ValueError):
        features.extract(rng.random((3, 4, 5, 6)))


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def brute_force_auroc(scores, labels):
    """The definition, pair by pair, ties counted as a half."""
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(positives) * len(negatives))


def test_auroc_matches_brute_force(rng):
    for _ in range(20):
        n = int(rng.integers(10, 60))
        scores = rng.normal(size=n)
        labels = rng.integers(0, 2, size=n)
        if labels.sum() in (0, n):
            continue
        assert classification.auroc(scores, labels) == pytest.approx(
            brute_force_auroc(scores, labels))


def test_auroc_handles_ties_as_half_credit():
    """INT8 output has 256 distinct values, so ties are the normal case."""
    scores = [1.0, 1.0, 1.0, 1.0]
    labels = [1, 0, 1, 0]
    assert classification.auroc(scores, labels) == pytest.approx(0.5)


def test_auroc_of_a_perfect_and_an_inverted_ranking():
    assert classification.auroc([3, 2, 1, 0], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert classification.auroc([0, 1, 2, 3], [1, 1, 0, 0]) == pytest.approx(0.0)


def test_auroc_is_nan_without_both_classes():
    assert np.isnan(classification.auroc([1, 2, 3], [1, 1, 1]))


def test_average_precision_on_a_worked_example():
    # Ranked: 1, 0, 1, 1 -> precisions at the hits are 1, 2/3, 3/4.
    value = classification.average_precision([4, 3, 2, 1], [1, 0, 1, 1])
    assert value == pytest.approx((1.0 + 2 / 3 + 3 / 4) / 3, abs=1e-9)


def test_confusion_matrix_arithmetic():
    matrix = classification.confusion([0.9, 0.8, 0.2, 0.1], [1, 0, 1, 0], threshold=0.5)
    assert (matrix.tp, matrix.fp, matrix.tn, matrix.fn) == (1, 1, 1, 1)
    assert matrix.precision == pytest.approx(0.5)
    assert matrix.recall == pytest.approx(0.5)
    assert matrix.f1 == pytest.approx(0.5)
    assert matrix.false_positive_rate == pytest.approx(0.5)


def test_threshold_at_fpr_respects_the_target(rng):
    scores = np.concatenate([rng.normal(0, 1, 2000), rng.normal(3, 1, 2000)])
    labels = np.concatenate([np.zeros(2000), np.ones(2000)])
    threshold = classification.threshold_at_fpr(scores, labels, 0.02, safety_sigma=0.0)
    achieved = float((scores[labels == 0] >= threshold).mean())
    assert achieved <= 0.02 + 1e-9


def test_safety_margin_makes_the_threshold_stricter(rng):
    scores = np.concatenate([rng.normal(0, 1, 400), rng.normal(3, 1, 400)])
    labels = np.concatenate([np.zeros(400), np.ones(400)])
    plain = classification.threshold_at_fpr(scores, labels, 0.02, safety_sigma=0.0)
    safe = classification.threshold_at_fpr(scores, labels, 0.02, safety_sigma=1.0)
    assert safe >= plain


def test_safety_margin_reduces_threshold_transfer_overshoot(rng):
    """The reason the margin exists, measured rather than asserted.

    Choose the threshold on one sample and apply it to another, many times.
    The unadjusted rule should overshoot the target more often than not; the
    adjusted one should not.
    """
    overshoot_plain = 0
    overshoot_safe = 0
    trials = 60
    for trial in range(trials):
        local = np.random.default_rng(trial)
        validation = local.normal(0, 1, 200)
        test = local.normal(0, 1, 600)
        labels = np.concatenate([np.zeros(200), np.ones(200)])
        scores = np.concatenate([validation, local.normal(3, 1, 200)])
        plain = classification.threshold_at_fpr(scores, labels, 0.02, safety_sigma=0.0)
        safe = classification.threshold_at_fpr(scores, labels, 0.02, safety_sigma=1.0)
        overshoot_plain += float((test >= plain).mean()) > 0.02
        overshoot_safe += float((test >= safe).mean()) > 0.02
    assert overshoot_safe < overshoot_plain


def test_summarise_reports_the_operating_point(rng):
    scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(3, 1, 200)])
    labels = np.concatenate([np.zeros(200), np.ones(200)])
    summary = classification.summarise(scores, labels, target_fpr=0.02)
    assert summary["n"] == 400
    assert summary["n_defective"] == 200
    assert summary["target_fpr"] == 0.02
    assert 0.0 <= summary["auroc"] <= 1.0
    assert summary["tp"] + summary["fn"] == 200


# ---------------------------------------------------------------------------
# Localisation
# ---------------------------------------------------------------------------

def test_iou_of_identical_and_disjoint_boxes():
    from edgeinspect.models.postprocess import iou

    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == pytest.approx(0.0)
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(5 / 15)


def test_matching_is_one_to_one():
    predicted = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    truth = np.array([[0, 0, 10, 10]], dtype=np.float32)
    pairs, unmatched = localisation.match_boxes(predicted, truth)
    assert len(pairs) == 1
    assert unmatched == []


def test_false_boxes_are_counted_only_on_good_frames():
    predictions = [np.array([[0, 0, 5, 5]], dtype=np.float32),
                   np.zeros((0, 4), dtype=np.float32)]
    truths = [np.zeros((0, 4), dtype=np.float32),
              np.array([[0, 0, 5, 5]], dtype=np.float32)]
    summary = localisation.summarise(predictions, truths)
    assert summary["n_good_frames"] == 1
    assert summary["false_boxes_per_good_frame"] == pytest.approx(1.0)
    assert summary["hit_rate"] == pytest.approx(0.0)
