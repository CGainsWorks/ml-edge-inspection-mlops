#!/bin/sh
# Benchmark on a Jetson.
#
# ***THIS SCRIPT HAS NEVER BEEN RUN.***
#
# No Jetson was available for this project.  Nothing in results/published/, in
# benchmarks/results.csv, or anywhere in the documentation came from this file
# or from any TensorRT engine.  docs/jetson.md is the long version.
#
# It is committed because it is the script a bring-up would actually run, and
# because writing down the procedure -- clocks, power mode, thermal state, what
# to record alongside the latency -- is most of the value of having done the
# thinking.  A Jetson benchmark taken without pinning the power mode is not a
# benchmark, it is a sample of what the DVFS governor felt like doing.
#
#   sudo sh scripts/bench_jetson.sh

set -eu

MODEL=${MODEL:-surface_defect}
ITERATIONS=${ITERATIONS:-500}
POWER_MODE=${POWER_MODE:-0}          # 0 is MAXN on Orin
OUT=${OUT:-results/latency_jetson.json}

if ! command -v tegrastats >/dev/null 2>&1; then
    echo "tegrastats not found: this is not a Jetson." >&2
    echo "See docs/jetson.md.  Nothing in this repository depends on this script." >&2
    exit 2
fi

echo "==> pinning the power mode and clocks"
# Without these two lines the numbers move by tens of percent between runs, and
# the first run after boot is the slowest one.
nvpmodel -m "$POWER_MODE"
jetson_clocks

echo "==> recording the device state that the numbers depend on"
mkdir -p results
{
    echo "date: $(date -Is)"
    echo "model: $(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo unknown)"
    echo "jetpack: $(dpkg-query --show nvidia-l4t-core 2>/dev/null || echo unknown)"
    echo "power_mode: $(nvpmodel -q | tr '\n' ' ')"
    echo "tensorrt: $(python3 -c 'import tensorrt; print(tensorrt.__version__)' 2>/dev/null \
        || echo unknown)"
} > results/jetson_device.txt
cat results/jetson_device.txt

echo "==> letting the board settle thermally"
# A cold board flatters the first minute of any benchmark and then throttles.
sleep 60

echo "==> building engines"
python3 scripts/export_tensorrt.py --onnx "models/${MODEL}_fp32.onnx" \
    --precision fp32 fp16 int8

echo "==> starting tegrastats"
tegrastats --interval 100 --logfile results/tegrastats.log &
TEGRA_PID=$!
# Stop the logger whatever happens next, including a failed benchmark: leaving
# tegrastats running after the script exits is how a device ends up with a
# hundred stale loggers and a full disk.
trap 'kill "$TEGRA_PID" 2>/dev/null || true' EXIT INT TERM

echo "==> benchmarking each engine"
for PRECISION in fp32 fp16 int8; do
    python3 scripts/benchmark_latency.py \
        --config configs/benchmark_jetson.yaml \
        --iterations "$ITERATIONS" \
        --out "results/latency_jetson_${PRECISION}.json" || true
done

kill "$TEGRA_PID" 2>/dev/null || true
trap - EXIT INT TERM

echo "==> merging"
python3 - "$OUT" <<'PYTHON'
import glob
import json
import sys

merged = {"rows": [], "device": open("results/jetson_device.txt").read()}
for path in sorted(glob.glob("results/latency_jetson_*.json")):
    with open(path) as handle:
        merged["rows"].extend(json.load(handle).get("rows", []))
with open(sys.argv[1], "w") as handle:
    json.dump(merged, handle, indent=2)
print("wrote", sys.argv[1], "with", len(merged["rows"]), "rows")
PYTHON

echo
echo "Record alongside these numbers, or they are not reproducible:"
echo "  the power mode, the clock state, the ambient temperature,"
echo "  the tegrastats log, and whether anything else was on the GPU."
