#!/bin/sh
# Create the virtualenv and install everything.
#
#   sh scripts/setup.sh            # runtime, dev and CPU training wheels
#   TORCH_INDEX=cu124 sh scripts/setup.sh    # CUDA training wheels instead
#   sh scripts/setup.sh --no-train # runtime and dev only, no torch
#
# CPU torch by default and on purpose: the model is 61k parameters and the whole
# loop -- including the retraining step -- runs on CPU in a couple of minutes.
# A CUDA install is a 2.5 GB download for a training run that takes 130 seconds
# instead of 400.  Pass TORCH_INDEX when you want it.

set -eu

PYTHON=${PYTHON:-python3}
VENV=${VENV:-.venv}
TORCH_INDEX=${TORCH_INDEX:-cpu}
WITH_TRAIN=1

for argument in "$@"; do
    case "$argument" in
        --no-train) WITH_TRAIN=0 ;;
        *) echo "unknown option: $argument" >&2; exit 2 ;;
    esac
done

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    PYTHON=python
fi

echo "==> creating $VENV with $PYTHON"
"$PYTHON" -m venv "$VENV"

if [ -x "$VENV/bin/python" ]; then
    VENV_PY="$VENV/bin/python"
else
    VENV_PY="$VENV/Scripts/python.exe"     # Windows
fi

echo "==> upgrading pip"
"$VENV_PY" -m pip install --upgrade pip >/dev/null

echo "==> installing runtime and development requirements"
"$VENV_PY" -m pip install -r requirements.txt

if [ "$WITH_TRAIN" -eq 1 ]; then
    echo "==> installing torch from the $TORCH_INDEX index"
    "$VENV_PY" -m pip install -r requirements-train.txt \
        --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
fi

echo
"$VENV_PY" - <<'PYTHON'
import importlib
import sys

print("python      ", sys.version.split()[0])
for module in ("numpy", "scipy", "onnx", "onnxruntime", "mlflow", "torch"):
    try:
        loaded = importlib.import_module(module)
        version = getattr(loaded, "__version__", "?")
    except ImportError:
        version = "not installed"
    print("{0:<12} {1}".format(module, version))
try:
    import torch
    print("cuda        ", torch.cuda.is_available(),
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except ImportError:
    pass
PYTHON

echo
echo "done.  Next:"
echo "  make data          # generate the dataset (about 130 s)"
echo "  make train export quantise"
echo "  make demo          # the whole operational loop"
