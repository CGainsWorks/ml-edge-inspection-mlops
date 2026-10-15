#!/bin/sh
# Jetson container entrypoint.  NEVER RUN: no Jetson was available.
#
# Builds the TensorRT engine on the device if it is missing, then starts the
# inference runtime.  The engine is built here rather than baked into the image
# because a TensorRT plan is specific to the GPU architecture, driver version
# and TensorRT version that produced it.

set -eu

MODEL=${MODEL:-surface_defect}
PRECISION=${PRECISION:-int8}
ENGINE="models/${MODEL}_${PRECISION}.engine"
STREAM=${STREAM:-clean}

if [ ! -f "$ENGINE" ]; then
    echo "==> no engine at $ENGINE; building it on this device"
    if [ "$PRECISION" = "int8" ] && [ ! -f models/calibration.npy ]; then
        echo "INT8 needs models/calibration.npy, produced by scripts/quantise.py" >&2
        echo "on the training host.  Copy it onto the device with the ONNX graph." >&2
        exit 2
    fi
    python3 scripts/export_tensorrt.py \
        --onnx "models/${MODEL}_fp32.onnx" \
        --precision "$PRECISION"
fi

echo "==> engine ready: $ENGINE"
exec python3 scripts/simulate_edge.py --stream "$STREAM" --model "$ENGINE" "$@"
