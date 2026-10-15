"""Box-level metrics.

The head predicts at stride 8, so a predicted box is accurate to eight pixels
at best.  Reporting mAP at IoU 0.5 against a four-pixel-wide scratch would
report the stride, not the model.  What is reported instead:

* **hit rate** -- the fraction of ground-truth defects that a predicted box
  overlaps at all.  For inspection this is the number that matters: an operator
  is being told where to look, and eight pixels of slop on a 256-pixel frame is
  not what makes that useful or useless.
* **mean IoU of matched pairs** -- how tight the boxes are, reported separately
  rather than folded into the hit rate.
* **false boxes per good frame** -- localisation's own false alarm rate,
  measured only on frames with no defect, so it cannot be traded against recall.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from ..models.postprocess import iou


def match_boxes(predicted: np.ndarray, truth: np.ndarray, min_iou: float = 0.1):
    """Greedy highest-IoU-first matching.  Returns ``(pairs, unmatched_truth)``.

    Greedy rather than Hungarian: with at most two defects per frame the two
    agree, and greedy is inspectable.
    """
    predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 4)
    truth = np.asarray(truth, dtype=np.float32).reshape(-1, 4)
    candidates = []
    for p_index in range(len(predicted)):
        for t_index in range(len(truth)):
            value = iou(predicted[p_index], truth[t_index])
            if value >= min_iou:
                candidates.append((value, p_index, t_index))
    candidates.sort(reverse=True)
    used_p, used_t = set(), set()
    pairs = []
    for value, p_index, t_index in candidates:
        if p_index in used_p or t_index in used_t:
            continue
        used_p.add(p_index)
        used_t.add(t_index)
        pairs.append((p_index, t_index, value))
    unmatched_truth = [i for i in range(len(truth)) if i not in used_t]
    return pairs, unmatched_truth


def summarise(predictions: Sequence[np.ndarray], truths: Sequence[np.ndarray],
              min_iou: float = 0.1) -> Dict[str, float]:
    """Aggregate box metrics over a split."""
    total_truth = 0
    total_hits = 0
    ious = []
    false_boxes_on_good = 0
    good_frames = 0
    for predicted, truth in zip(predictions, truths):
        truth = np.asarray(truth, dtype=np.float32).reshape(-1, 4)
        predicted = np.asarray(predicted, dtype=np.float32).reshape(-1, 4)
        if len(truth) == 0:
            good_frames += 1
            false_boxes_on_good += len(predicted)
            continue
        pairs, _ = match_boxes(predicted, truth, min_iou=min_iou)
        total_truth += len(truth)
        total_hits += len(pairs)
        ious.extend(value for _, _, value in pairs)
    return {
        "hit_rate": round(total_hits / total_truth, 6) if total_truth else float("nan"),
        "mean_iou_matched": round(float(np.mean(ious)), 6) if ious else float("nan"),
        "false_boxes_per_good_frame": (
            round(false_boxes_on_good / good_frames, 6) if good_frames else float("nan")
        ),
        "n_truth_boxes": int(total_truth),
        "n_good_frames": int(good_frames),
        "min_iou": float(min_iou),
    }
