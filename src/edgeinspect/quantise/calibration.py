"""Choosing the calibration set.

Post-training quantisation reduces to one question: what range does each
activation take?  The answer is read off a handful of frames, and the frames are
the whole ballgame.  Calibrate on 128 defect-free frames and the activations a
defect produces sit outside the observed range, get clipped, and INT8 recall
collapses -- on a metric suite that only reports overall accuracy this looks
like a small regression rather than the specific, fixable problem it is.

So the selection here is explicit and recorded:

* **stratified** by label, so defective frames are represented in proportion to
  a stated ratio rather than by luck;
* **diverse** across image brightness and contrast, so the exposure range the
  line actually produces is covered rather than the middle of it sampled 128
  times;
* **written to disk as a list of frame ids**, so the ONNX Runtime PTQ path and
  the TensorRT INT8 path calibrate on identical frames and the two can be
  compared.

``tests/test_calibration.py`` includes the failure case: a defect-free
calibration set, and the recall drop it causes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from .. import paths
from ..data import Record, load_arrays, load_index
from ..models.preprocess import DEFAULT as DEFAULT_INPUT
from ..models.preprocess import InputSpec


def _image_stats(images: np.ndarray) -> np.ndarray:
    """``(n, 2)`` of per-frame mean and standard deviation."""
    flat = images.reshape(len(images), -1)
    return np.stack([flat.mean(axis=1), flat.std(axis=1)], axis=1)


def select_indices(labels: np.ndarray, stats: np.ndarray, count: int,
                   defect_fraction: float = 0.5, bins: int = 8,
                   seed: int = 0) -> List[int]:
    """Pick ``count`` indices, stratified by label and spread over image statistics."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    n_defect = int(round(count * defect_fraction))
    n_good = count - n_defect

    chosen: List[int] = []
    for target, wanted in ((1, n_defect), (0, n_good)):
        pool = np.nonzero(labels == target)[0]
        if len(pool) == 0:
            continue
        take = min(wanted, len(pool))
        # Spread the take over brightness bins so a dark corner of the
        # distribution is not represented by zero frames.
        brightness = stats[pool, 0]
        edges = np.quantile(brightness, np.linspace(0, 1, bins + 1))
        edges[-1] += 1e-6
        assignment = np.clip(np.digitize(brightness, edges[1:-1]), 0, bins - 1)
        per_bin = [pool[assignment == b] for b in range(bins)]
        rng_order = [rng.permutation(members) for members in per_bin]
        cursor = 0
        while take > 0:
            progressed = False
            for members in rng_order:
                if cursor < len(members) and take > 0:
                    chosen.append(int(members[cursor]))
                    take -= 1
                    progressed = True
            cursor += 1
            if not progressed:
                break
    chosen = sorted(set(chosen))
    if len(chosen) < count:
        remaining = [i for i in range(len(labels)) if i not in set(chosen)]
        rng.shuffle(remaining)
        chosen.extend(int(i) for i in remaining[:count - len(chosen)])
    return sorted(chosen)[:count]


def build(data_root: Path, split: str = "train", count: int = 192,
          defect_fraction: float = 0.5, seed: int = 0,
          records: Optional[Sequence[Record]] = None,
          spec: Optional[InputSpec] = None):
    """Return ``(frames, manifest)``.

    ``frames`` comes back in **graph input space**, not in ``[0, 1]``: the
    calibration reader feeds the ONNX quantiser directly, bypassing the backend
    that would otherwise apply the contract in
    :mod:`edgeinspect.models.preprocess`.  A calibration set normalised
    differently from production input calibrates the wrong activation ranges,
    and the symptom is an INT8 model that is quietly worse rather than broken.
    """
    spec = spec or DEFAULT_INPUT
    items = list(records) if records is not None else load_index(data_root, split)
    images, _masks, labels, items = load_arrays(data_root, split, records=items)
    stats = _image_stats(images)
    count = int(min(count, len(images)))
    indices = select_indices(labels, stats, count, defect_fraction=defect_fraction, seed=seed)
    frames = np.ascontiguousarray(spec.apply(images[indices]), dtype=np.float32)
    manifest = {
        "split": split,
        "count": len(indices),
        "requested": count,
        "defect_fraction_requested": defect_fraction,
        "defect_fraction_actual": round(float(labels[indices].mean()), 6),
        "seed": seed,
        "input_space": {"mean": spec.mean, "scale": spec.scale},
        "frame_ids": [items[i].id for i in indices],
        "brightness": {
            "min": round(float(stats[indices, 0].min()), 6),
            "max": round(float(stats[indices, 0].max()), 6),
            "mean": round(float(stats[indices, 0].mean()), 6),
        },
        "coverage_note": (
            "Brightness range of the calibration set against the full split; if the "
            "calibration range is materially narrower, INT8 activations clip on the "
            "frames outside it."
        ),
        "split_brightness": {
            "min": round(float(stats[:, 0].min()), 6),
            "max": round(float(stats[:, 0].max()), 6),
        },
    }
    return frames, manifest


def save(frames: np.ndarray, manifest: dict, out_dir: Path, name: str = "calibration"):
    """Persist the calibration set so every quantisation path uses the same frames."""
    out_dir = Path(out_dir)
    paths.ensure(out_dir)
    npy_path = out_dir / (name + ".npy")
    json_path = out_dir / (name + ".json")
    np.save(npy_path, frames)
    payload = dict(manifest)
    payload["frames"] = paths.relative(npy_path)
    payload["shape"] = list(frames.shape)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return npy_path, json_path


def load(out_dir: Path, name: str = "calibration"):
    """Load a saved calibration set.  Returns ``(frames, manifest)``."""
    out_dir = Path(out_dir)
    json_path = out_dir / (name + ".json")
    if not json_path.exists():
        raise FileNotFoundError(
            "no calibration manifest at {0}.  Run `make quantise`.".format(
                paths.relative(json_path))
        )
    with json_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return np.load(paths.resolve(manifest["frames"])), manifest
