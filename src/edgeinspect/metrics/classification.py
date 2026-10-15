"""Image-level metrics, implemented rather than imported.

scikit-learn is not a dependency of this project and adding a 40 MB wheel to an
edge runtime image to compute an ROC curve is not a trade worth making.  These
are the standard definitions; ``tests/test_metrics.py`` checks them against
hand-worked examples and against a brute-force reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve, via the rank (Mann-Whitney U) identity.

    Ties are handled with average ranks, which is the difference between this
    and the naive sort-and-count version.  A quantised model produces ties
    constantly -- INT8 output has 256 distinct values -- so a tie-blind AUROC
    would systematically flatter or punish the INT8 variant depending on sort
    order.  That is precisely the comparison this repository publishes.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    index = 0
    while index < len(sorted_scores):
        end = index
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[index]:
            end += 1
        average = (index + end) / 2.0 + 1.0
        ranks[order[index:end + 1]] = average
        index = end + 1
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Average precision, the step-wise integral of the precision-recall curve."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if int((labels == 1).sum()) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    labels = labels[order]
    true_positives = np.cumsum(labels == 1)
    false_positives = np.cumsum(labels == 0)
    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / max(int((labels == 1).sum()), 1)
    previous_recall = 0.0
    total = 0.0
    for p, r in zip(precision, recall):
        total += float(p) * (float(r) - previous_recall)
        previous_recall = float(r)
    return total


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    @property
    def false_positive_rate(self) -> float:
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) else 0.0

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "accuracy": round(self.accuracy, 6),
            "false_positive_rate": round(self.false_positive_rate, 6),
        }


def confusion(scores: Sequence[float], labels: Sequence[int],
              threshold: float) -> ConfusionMatrix:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predicted = scores >= threshold
    return ConfusionMatrix(
        tp=int(((predicted == 1) & (labels == 1)).sum()),
        fp=int(((predicted == 1) & (labels == 0)).sum()),
        tn=int(((predicted == 0) & (labels == 0)).sum()),
        fn=int(((predicted == 0) & (labels == 1)).sum()),
    )


def threshold_at_fpr(scores: Sequence[float], labels: Sequence[int],
                     target_fpr: float, safety_sigma: float = 1.0) -> float:
    """A threshold whose false positive rate is at or below the target.

    Inspection is a false-positive-constrained problem: every false alarm stops
    the line and costs an operator's attention, so the operating point is chosen
    on the *good* parts and the recall that follows is reported rather than
    chosen.  Selecting a threshold to maximise F1 and then reporting that F1 is
    the same number twice.

    ``safety_sigma`` is the part that came from measurement rather than theory.
    The obvious rule -- allow exactly ``target_fpr * n`` of the validation
    negatives above the threshold -- is an unbiased estimate of the right
    threshold and a badly biased estimate of the *achieved* false positive rate
    on new data.  At a 2% target with 200 validation negatives the rule allows
    four exceedances, and the number of exceedances is binomial with a standard
    deviation of about two.  Half the time the threshold lands too low, and the
    line sees three or four percent false alarms against a two percent
    specification.  Measured on the reference model, the point-estimate rule
    gave 2.0% on validation and 3.7% on test.

    Subtracting one binomial standard deviation from the allowance moves the
    threshold up by roughly one order statistic and costs a little recall.  That
    is the right direction to be wrong in for this application: a missed defect
    is found at the next station, an operator who stops trusting the alarm is
    not recoverable.  Set ``safety_sigma=0`` for the unadjusted rule.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    negatives = np.sort(scores[labels == 0])
    n = negatives.size
    if n == 0:
        return 0.5
    expected = target_fpr * n
    sigma = float(np.sqrt(max(n * target_fpr * (1.0 - target_fpr), 0.0)))
    allowed = int(np.floor(max(expected - safety_sigma * sigma, 0.0)))
    if allowed >= n:
        return float(negatives[0])
    index = n - 1 - allowed
    # Nudge above the highest allowed negative so the comparison is strict.
    return float(np.nextafter(negatives[index], np.inf))


def summarise(scores: Sequence[float], labels: Sequence[int],
              threshold: Optional[float] = None,
              target_fpr: float = 0.02) -> Dict[str, float]:
    """The standard block of image-level numbers, at a stated operating point."""
    if threshold is None:
        threshold = threshold_at_fpr(scores, labels, target_fpr)
    matrix = confusion(scores, labels, threshold)
    out = {
        "auroc": round(float(auroc(scores, labels)), 6),
        "average_precision": round(float(average_precision(scores, labels)), 6),
        "threshold": float(threshold),
        "target_fpr": float(target_fpr),
        "n": int(len(labels)),
        "n_defective": int(np.asarray(labels).sum()),
    }
    out.update(matrix.to_dict())
    return out
