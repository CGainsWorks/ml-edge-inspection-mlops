#!/usr/bin/env python
"""Latency and throughput for every precision variant, on this host.

    python scripts/benchmark_latency.py --skip-missing
    python scripts/benchmark_latency.py --features      # telemetry cost only

Every row carries the host it was measured on.  **None of these are Jetson
numbers.**  No Jetson was available for this project; ``docs/jetson.md`` states
what would have to be run to produce one and why the CPU numbers here do not
predict it.

Two measurement choices that change the answer:

**Threads are pinned.**  ONNX Runtime otherwise picks a thread count from the
machine's load, and a latency table that moves with the background load is not a
measurement.  ``intra_op_threads`` in ``configs/benchmark.yaml`` fixes it and the
value is recorded in the output.

**Warm-up iterations are discarded.**  The first calls allocate arenas and
select kernels and are several times slower than the steady state.  Leaving them
in makes p99 a measurement of process start-up.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import platform
import time

import numpy as np

from edgeinspect import paths
from edgeinspect.config import load_yaml
from edgeinspect.data import load_arrays
from edgeinspect.edge.backends import load as load_backend
from edgeinspect.monitoring import features as feature_module


def host_info() -> dict:
    import onnxruntime as ort

    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "cpu_count": None,
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    return info


def percentiles(values) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": round(float(array.mean()), 4),
        "std": round(float(array.std()), 4),
        "p50": round(float(np.percentile(array, 50)), 4),
        "p95": round(float(np.percentile(array, 95)), 4),
        "p99": round(float(np.percentile(array, 99)), 4),
        "min": round(float(array.min()), 4),
        "max": round(float(array.max()), 4),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/benchmark.yaml")
    parser.add_argument("--data-root", default="data/processed")
    parser.add_argument("--split", default="test")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--out", default="results/latency.json")
    parser.add_argument("--csv", default="benchmarks/results.csv")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--features", action="store_true",
                        help="also benchmark the telemetry feature extraction")
    return parser.parse_args(argv)


def benchmark_features(images, iterations: int, warmup: int) -> dict:
    for index in range(warmup):
        feature_module.extract(images[index % len(images)])
    timings = []
    for index in range(iterations):
        frame = images[index % len(images)]
        started = time.perf_counter()
        feature_module.extract(frame)
        timings.append((time.perf_counter() - started) * 1000.0)
    return percentiles(timings)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    iterations = int(args.iterations or config.get("iterations", 300))
    warmup = int(args.warmup or config.get("warmup", 30))
    batch_sizes = list(args.batch_sizes or config.get("batch_sizes", [1]))
    threads = config.get("intra_op_threads")

    images, _masks, _labels, _records = load_arrays(paths.resolve(args.data_root), args.split)
    rows = []

    for variant in config.get("variants", []):
        model_path = paths.resolve(variant["model"])
        if not model_path.exists():
            if args.skip_missing:
                print("skipping missing " + paths.relative(model_path))
                continue
            raise SystemExit("no model at " + paths.relative(model_path))
        backend = load_backend(model_path, backend="onnx",
                               providers=variant.get("providers"),
                               intra_op_threads=threads)
        try:
            for batch_size in batch_sizes:
                batch = np.ascontiguousarray(images[:batch_size], dtype=np.float32)
                if len(batch) < batch_size:
                    reps = int(np.ceil(batch_size / max(len(batch), 1)))
                    batch = np.ascontiguousarray(
                        np.tile(batch, (reps, 1, 1, 1))[:batch_size], dtype=np.float32)
                for _ in range(warmup):
                    backend.infer(batch)
                timings = []
                for _ in range(iterations):
                    started = time.perf_counter()
                    backend.infer(batch)
                    timings.append((time.perf_counter() - started) * 1000.0)
                stats = percentiles(timings)
                rows.append({
                    "variant": variant["name"],
                    "precision": backend.precision,
                    "model": paths.relative(model_path),
                    "bytes": int(model_path.stat().st_size),
                    "providers": list(backend.providers),
                    "batch_size": batch_size,
                    "iterations": iterations,
                    "latency_ms": stats,
                    "throughput_fps": round(batch_size * 1000.0 / stats["mean"], 2),
                })
                print("{0:<16} batch {1:<3} {2:>8.3f} ms mean  {3:>8.3f} p95  "
                      "{4:>8.1f} fps".format(variant["name"], batch_size, stats["mean"],
                                             stats["p95"], rows[-1]["throughput_fps"]),
                      flush=True)
        finally:
            backend.close()

    payload = {
        "host": host_info(),
        "intra_op_threads": threads,
        "iterations": iterations,
        "warmup": warmup,
        "split": args.split,
        "jetson": False,
        "note": (
            "Measured on the host described above, not on a Jetson.  No Jetson was "
            "available for this project; see docs/jetson.md."
        ),
        "rows": rows,
    }
    if args.features:
        payload["features_ms"] = benchmark_features(images, iterations, warmup)
        print("\nfeature extraction: {0:.3f} ms mean, {1:.3f} p95".format(
            payload["features_ms"]["mean"], payload["features_ms"]["p95"]))

    out_path = paths.resolve(args.out)
    paths.ensure(out_path.parent)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    csv_path = paths.resolve(args.csv)
    paths.ensure(csv_path.parent)
    with csv_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("variant,precision,batch_size,providers,mean_ms,p50_ms,p95_ms,"
                     "p99_ms,throughput_fps,bytes,host\n")
        host = payload["host"]["platform"].replace(",", " ")
        for row in rows:
            handle.write("{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}\n".format(
                row["variant"], row["precision"], row["batch_size"],
                "+".join(row["providers"]), row["latency_ms"]["mean"],
                row["latency_ms"]["p50"], row["latency_ms"]["p95"],
                row["latency_ms"]["p99"], row["throughput_fps"], row["bytes"], host))

    print("\nwrote {0} and {1}".format(paths.relative(out_path), paths.relative(csv_path)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
