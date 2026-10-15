"""Inference telemetry: what the edge device sends back.

One JSON object per frame, appended to a JSONL file.  JSONL and not a database
because the device is expected to lose power mid-write and a truncated last line
is recoverable; a truncated SQLite page is not.  :func:`read` drops a trailing
partial line and says how many it dropped rather than raising, because a
monitoring stack that crashes on a power cut is a monitoring stack that is
offline exactly when something interesting happened.

What is in a record, and what deliberately is not
------------------------------------------------
In: the model version and precision, so a metric can be attributed to a
deployment; the frame's feature vector, because that is what the drift detector
consumes; the score and the decision; and the measured latency.

Not in: the frame itself.  Shipping every frame off the device is neither
affordable nor, on most lines, permitted.  :class:`CaptureStore` keeps a sampled
subset locally under a stated policy, and that sample is what a retraining run
pulls.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .. import paths
from ..data.dataset import write_png

SCHEMA_VERSION = 1


@dataclass
class TelemetryRecord:
    """One inferred frame."""

    frame_id: int
    stream: str
    ts: float                     # seconds since stream start, monotonic within a stream
    model_version: int
    model_name: str
    precision: str
    backend: str
    latency_ms: float
    score: float
    decision: int                 # 1 = flagged as defective
    n_boxes: int
    features: Dict[str, float] = field(default_factory=dict)
    boxes: List[dict] = field(default_factory=list)
    ground_truth: Optional[int] = None   # simulation only; never present on a real line
    captured: Optional[str] = None       # relative path if the frame was retained
    schema: int = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


class TelemetryWriter:
    """Appends records to a JSONL file, flushing on a fixed interval.

    ``flush_every`` trades durability for throughput.  At the default of 32 a
    power cut loses at most 32 frames of telemetry, which for drift detection
    over windows of 200 frames is not material.
    """

    def __init__(self, path: Path, flush_every: int = 32, append: bool = False):
        self.path = Path(path)
        paths.ensure(self.path.parent)
        self.handle = self.path.open("a" if append else "w", encoding="utf-8")
        self.flush_every = int(flush_every)
        self.written = 0

    def write(self, record: TelemetryRecord) -> None:
        self.handle.write(record.to_json() + "\n")
        self.written += 1
        if self.flush_every and self.written % self.flush_every == 0:
            self.handle.flush()
            os.fsync(self.handle.fileno())

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.handle.close()

    def __enter__(self) -> "TelemetryWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


def read(path: Path, tolerate_partial: bool = True) -> Tuple[List[dict], int]:
    """Read a telemetry file.  Returns ``(records, dropped)``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("no telemetry at " + paths.relative(path))
    records: List[dict] = []
    dropped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                dropped += 1
                if not tolerate_partial:
                    raise
    return records, dropped


def iter_records(path: Path) -> Iterator[dict]:
    """Stream a telemetry file without holding it in memory."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def to_arrays(records: Iterable[dict], names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Pull named signals out of telemetry records into arrays.

    A name is looked up first as a top-level field and then inside ``features``,
    so a drift rule can watch ``score`` and ``brightness`` with the same syntax.
    """
    materialised = list(records)
    out: Dict[str, np.ndarray] = {}
    for name in names:
        values = []
        for record in materialised:
            if name in record:
                values.append(record[name])
            else:
                values.append(record.get("features", {}).get(name, np.nan))
        out[name] = np.asarray(values, dtype=np.float64)
    return out


class CaptureStore:
    """The sampled subset of frames the device keeps for retraining.

    The policy is the interesting part and it is not "every Nth frame".  Three
    reasons a frame is retained:

    * **periodic** -- one frame in ``every``, so the retained set tracks the
      production distribution rather than only its tails;
    * **uncertain** -- the score sits in a band around the decision threshold,
      where the model is least useful and a label is worth most;
    * **flagged** -- the frame was called defective, because those are the ones
      an operator will adjudicate anyway.

    Keeping only uncertain frames is the classic mistake: the retraining set
    ends up entirely composed of hard cases, the retrained model is calibrated
    for a distribution that does not exist, and its false alarm rate on ordinary
    good parts goes up.  ``docs/RUNBOOK.md`` records the ratio actually used.
    """

    def __init__(self, root: Path, every: int = 25, uncertainty_band: float = 0.15,
                 keep_flagged: bool = True, max_frames: int = 4000):
        self.root = Path(root)
        paths.ensure(self.root / "images")
        self.every = int(every)
        self.uncertainty_band = float(uncertainty_band)
        self.keep_flagged = bool(keep_flagged)
        self.max_frames = int(max_frames)
        self.saved = 0
        self.reasons: Dict[str, int] = {"periodic": 0, "uncertain": 0, "flagged": 0}
        self._index: List[dict] = []

    def should_keep(self, frame_id: int, score: float, threshold: float,
                    decision: int) -> Optional[str]:
        if self.saved >= self.max_frames:
            return None
        if self.every and frame_id % self.every == 0:
            return "periodic"
        if abs(score - threshold) <= self.uncertainty_band:
            return "uncertain"
        if self.keep_flagged and decision == 1:
            return "flagged"
        return None

    def keep(self, frame_id: int, image: np.ndarray, reason: str, stream: str,
             score: float, ground_truth: Optional[int] = None,
             mask: Optional[np.ndarray] = None) -> str:
        """Persist one frame.  Returns its repository-relative path."""
        name = "{0}_{1:06d}".format(stream, frame_id)
        image_path = self.root / "images" / (name + ".png")
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 3:
            array = array[0]
        write_png(image_path, np.clip(array * 255.0, 0, 255).astype(np.uint8))
        entry = {
            "id": name,
            "frame_id": frame_id,
            "stream": stream,
            "image": paths.relative(image_path),
            "reason": reason,
            "score": round(float(score), 6),
            "ground_truth": ground_truth,
        }
        if mask is not None:
            mask_path = self.root / "masks" / (name + ".png")
            mask_array = np.asarray(mask, dtype=np.float32)
            if mask_array.ndim == 3:
                mask_array = mask_array[0]
            write_png(mask_path, (mask_array > 0.5).astype(np.uint8) * 255)
            entry["mask"] = paths.relative(mask_path)
        self._index.append(entry)
        self.reasons[reason] = self.reasons.get(reason, 0) + 1
        self.saved += 1
        return entry["image"]

    def flush(self) -> Path:
        """Write the capture index atomically.  Returns its path."""
        index_path = self.root / "index.json"
        payload = {
            "policy": {
                "every": self.every,
                "uncertainty_band": self.uncertainty_band,
                "keep_flagged": self.keep_flagged,
                "max_frames": self.max_frames,
            },
            "counts": dict(self.reasons),
            "total": self.saved,
            "frames": self._index,
        }
        paths.ensure(self.root)
        handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                             dir=str(self.root), suffix=".tmp")
        try:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, index_path)
        return index_path

    @staticmethod
    def load(root: Path) -> dict:
        index_path = Path(root) / "index.json"
        if not index_path.exists():
            raise FileNotFoundError("no capture index at " + paths.relative(index_path))
        with index_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
