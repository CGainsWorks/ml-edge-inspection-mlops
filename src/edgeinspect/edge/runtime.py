"""The edge inference runtime.

Load whatever is deployed, run frames through it, decide, extract features,
write telemetry, and keep a sampled subset of frames for retraining.  This is
the process that would run on the Jetson.

Latency accounting
------------------
Three numbers are recorded separately and never added together into one
"inference time", because they scale differently and have different fixes:

* ``latency_ms`` -- the backend call alone.  This is what precision changes
  move and what the benchmark table reports.
* ``feature_ms`` -- the telemetry feature extraction.  This is the cost of being
  able to detect drift at all, and it is reported so that cost is visible rather
  than buried.
* ``post_ms`` -- thresholding and box extraction.

A pipeline that reports only the first is reporting the part that looks best.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from .. import paths
from ..models import postprocess
from ..monitoring import features as feature_module
from .backends.base import Backend
from .telemetry import CaptureStore, TelemetryRecord, TelemetryWriter


@dataclass
class RuntimeStats:
    """What one run of the runtime produced."""

    frames: int = 0
    flagged: int = 0
    latency_ms: List[float] = field(default_factory=list)
    feature_ms: List[float] = field(default_factory=list)
    post_ms: List[float] = field(default_factory=list)
    captured: int = 0
    telemetry_path: Optional[str] = None
    capture_index: Optional[str] = None

    def percentiles(self, values: Iterable[float]) -> Dict[str, float]:
        array = np.asarray(list(values), dtype=np.float64)
        if array.size == 0:
            return {}
        return {
            "mean": round(float(array.mean()), 4),
            "p50": round(float(np.percentile(array, 50)), 4),
            "p95": round(float(np.percentile(array, 95)), 4),
            "p99": round(float(np.percentile(array, 99)), 4),
            "max": round(float(array.max()), 4),
        }

    def to_dict(self) -> dict:
        latency = self.percentiles(self.latency_ms)
        return {
            "frames": self.frames,
            "flagged": self.flagged,
            "flagged_rate": round(self.flagged / self.frames, 6) if self.frames else 0.0,
            "captured": self.captured,
            "latency_ms": latency,
            "feature_ms": self.percentiles(self.feature_ms),
            "post_ms": self.percentiles(self.post_ms),
            "fps_single_stream": (
                round(1000.0 / latency["mean"], 2) if latency.get("mean") else None
            ),
            "telemetry": self.telemetry_path,
            "capture_index": self.capture_index,
        }


def run(backend: Backend, source, telemetry_path: Path, model_version: int,
        model_name: str, stream_name: Optional[str] = None,
        threshold: Optional[float] = None, fps: float = 30.0,
        capture: Optional[CaptureStore] = None,
        warmup: int = 5, progress_every: int = 0,
        record_ground_truth: bool = True) -> RuntimeStats:
    """Run ``source`` through ``backend``, writing telemetry.  Returns the stats.

    ``threshold`` defaults to the one in the model's sidecar.  Overriding it is
    for evaluation sweeps only; a deployment that runs at a threshold other than
    the one recorded with the artefact is a deployment nobody can reproduce.
    """
    threshold = backend.threshold if threshold is None else float(threshold)
    top_k = backend.top_k
    cell_threshold = backend.cell_threshold
    stream_name = stream_name or getattr(source, "name", None) or getattr(
        getattr(source, "spec", None), "name", "stream")
    stats = RuntimeStats()

    # Warm up before measuring.  The first call through ONNX Runtime allocates
    # arenas and picks kernels, and it is several times slower than the steady
    # state.  Including it makes p99 a measurement of process start-up.
    if warmup > 0:
        dummy = np.zeros((1, 1, 256, 256), dtype=np.float32)
        for _ in range(warmup):
            backend.infer(dummy)

    writer = TelemetryWriter(telemetry_path)
    try:
        for frame_id, image, mask, ground_truth, domain_name in source:
            batch = np.ascontiguousarray(image[None, ...], dtype=np.float32)

            started = time.perf_counter()
            logits = backend.infer(batch)
            latency_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            probability = postprocess.sigmoid(logits[0][0])
            score = postprocess.image_score(logits[0], top_k=top_k)
            decision = int(score >= threshold)
            boxes = postprocess.heatmap_to_boxes(
                probability, threshold=cell_threshold, input_size=image.shape[-1]
            ) if decision else []
            post_ms = (time.perf_counter() - started) * 1000.0

            started = time.perf_counter()
            frame_features = feature_module.extract(image)
            frame_features["score"] = round(float(score), 6)
            feature_ms = (time.perf_counter() - started) * 1000.0

            captured_path = None
            if capture is not None:
                reason = capture.should_keep(frame_id, float(score), threshold, decision)
                if reason:
                    captured_path = capture.keep(
                        frame_id, image, reason, stream_name, float(score),
                        ground_truth=int(ground_truth) if record_ground_truth else None,
                        mask=mask,
                    )
                    stats.captured += 1

            writer.write(TelemetryRecord(
                frame_id=frame_id,
                stream=str(stream_name),
                ts=round(frame_id / max(fps, 1e-6), 6),
                model_version=int(model_version),
                model_name=str(model_name),
                precision=backend.precision,
                backend=backend.name,
                latency_ms=round(latency_ms, 4),
                score=round(float(score), 6),
                decision=decision,
                n_boxes=len(boxes),
                features=frame_features,
                boxes=[b.to_dict() for b in boxes],
                ground_truth=int(ground_truth) if record_ground_truth else None,
                captured=captured_path,
            ))

            stats.frames += 1
            stats.flagged += decision
            stats.latency_ms.append(latency_ms)
            stats.feature_ms.append(feature_ms)
            stats.post_ms.append(post_ms)
            if progress_every and stats.frames % progress_every == 0:
                print("  {0}: {1} frames".format(stream_name, stats.frames), flush=True)
    finally:
        writer.close()
        if capture is not None:
            stats.capture_index = paths.relative(capture.flush())

    stats.telemetry_path = paths.relative(Path(telemetry_path))
    return stats


def write_stats(stats: RuntimeStats, path: Path, extra: Optional[Dict] = None) -> Path:
    payload = stats.to_dict()
    if extra:
        payload.update(extra)
    paths.ensure(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return Path(path)
