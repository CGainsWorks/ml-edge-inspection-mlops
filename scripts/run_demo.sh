#!/bin/sh
# The whole thing from a clean checkout, in one command.
#
#   sh scripts/run_demo.sh              # full run, about thirty minutes
#   sh scripts/run_demo.sh --quick      # tiny dataset and short streams, about four
#
# Builds whatever is missing and then runs the operational loop.  Safe to re-run:
# each stage is skipped when its output already exists.

set -eu

PRESET=full
DATA=data/processed
LOOP_FLAGS=""

for argument in "$@"; do
    case "$argument" in
        --quick) PRESET=tiny; DATA=data/tiny; LOOP_FLAGS="--quick" ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

if [ -x .venv/bin/python ]; then
    PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
    PY=.venv/Scripts/python.exe
else
    PY=${PYTHON:-python}
fi
export PYTHONPATH=src

step() { printf '\n\033[36m==> %s\033[0m\n' "$1"; }

if [ ! -f "$DATA/dataset.json" ]; then
    step "generating the $PRESET dataset"
    "$PY" scripts/make_dataset.py --preset "$PRESET" --root "$DATA" --verify 3
else
    step "dataset present at $DATA"
fi

if [ ! -f models/surface_defect.pt ]; then
    step "training the detector"
    "$PY" scripts/train.py --data-root "$DATA" --name surface_defect
else
    step "checkpoint present"
fi

if [ ! -f models/surface_defect_fp32.onnx ]; then
    step "exporting to ONNX and verifying against the checkpoint"
    "$PY" scripts/export_onnx.py --weights models/surface_defect.pt
fi

if [ ! -f models/surface_defect_int8.onnx ]; then
    step "post-training quantisation"
    "$PY" scripts/quantise.py --model models/surface_defect_fp32.onnx --data-root "$DATA"
fi

if [ ! -f results/accuracy.json ]; then
    step "evaluating every precision variant"
    "$PY" scripts/evaluate.py --skip-missing --data-root "$DATA" \
        --models models/surface_defect_fp32.onnx models/surface_defect_int8.onnx \
                 models/surface_defect_int8_percentile.onnx \
                 models/surface_defect_int8_keephead.onnx \
        --out results/accuracy.json
fi

step "running the operational loop"
# shellcheck disable=SC2086
"$PY" scripts/run_loop.py --data-root "$DATA" --fresh $LOOP_FLAGS

step "done"
echo "  results/loop.json      the whole run, step by step"
echo "  results/accuracy.json  accuracy per precision variant"
echo "  make status            what is registered and what is deployed"
