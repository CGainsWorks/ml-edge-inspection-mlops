# Edge inspection with a closed MLOps loop.
#
# `make demo` is the one command.  `make check` is exactly what CI runs.
#
# Four classes of target, and the distinction is the honest part of this file:
#
#   FAST     seconds to a couple of minutes.  No dataset, no GPU, no weights.
#            Everything CI runs on every push.
#              make check  make test  make lint  make docs-check
#
#   DATA     regenerates the dataset from its seeds.  About 130 seconds for the
#            published preset, byte-for-byte deterministic.
#              make data  make data-tiny
#
#   LOOP     the operational loop.  Needs the dataset and the artefacts; runs on
#            CPU.  `make demo` is about six minutes, most of it the retraining
#            step; `make demo-quick` is about two and is what CI runs.
#              make demo  make demo-quick  make drift  make retrain  make rollback
#
#   HEAVY    needs the training toolchain, and wants a CUDA GPU.
#            `make experiments` regenerates every published number: about
#            twenty five minutes on an RTX 4060.
#              make train  make qat  make export  make quantise  make evaluate
#              make bench  make ablate  make experiments  make publish
#
#   JETSON   needs a Jetson with TensorRT.  NEVER RUN: no board was available.
#            No number in this repository came from these targets, and
#            docs/jetson.md says so at length.
#              make engine  make bench-jetson

SHELL := /bin/sh
PYTHON ?= python
VENV ?= .venv
VENV_PY := $(VENV)/bin/python
ifeq ($(wildcard $(VENV_PY)),)
VENV_PY := $(VENV)/Scripts/python.exe
endif
ifeq ($(wildcard $(VENV_PY)),)
VENV_PY := $(PYTHON)
endif

DATA ?= data/processed
MODELS ?= models
RESULTS ?= results
PUBLISHED := $(RESULTS)/published
NAME ?= surface_defect
PRESET ?= full
DEVICE ?= auto

CHECKPOINT := $(MODELS)/$(NAME).pt
FP32 := $(MODELS)/$(NAME)_fp32.onnx
INT8 := $(MODELS)/$(NAME)_int8.onnx
INT8_PCT := $(MODELS)/$(NAME)_int8_percentile.onnx
INT8_HEAD := $(MODELS)/$(NAME)_int8_keephead.onnx
QAT_INT8 := $(MODELS)/$(NAME)_qat_int8.onnx
QAT_INT8_PCT := $(MODELS)/$(NAME)_qat_int8_percentile.onnx

export PYTHONPATH := src

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "  python: $(VENV_PY)"

# ---------------------------------------------------------------------------
# Fast
# ---------------------------------------------------------------------------

.PHONY: check
check: lint test docs-check  ## Everything CI runs on every push

.PHONY: lint
lint:  ## flake8 over src, tests, scripts and pipelines
	$(VENV_PY) -m flake8 src/ tests/ scripts/ pipelines/

.PHONY: test
test:  ## The whole test suite
	$(VENV_PY) -m pytest tests/ -q --durations=15

.PHONY: test-fast
test-fast:  ## Tests that need neither a dataset nor weights
	$(VENV_PY) -m pytest tests/ -q -m "not needs_model and not needs_dataset and not slow"

.PHONY: docs-check
docs-check:  ## Every number in the docs traces to results/published/
	$(VENV_PY) scripts/check_docs.py

.PHONY: docs-fix
docs-fix:  ## Regenerate the tables embedded in the docs
	$(VENV_PY) scripts/make_tables.py --quiet
	$(VENV_PY) scripts/check_docs.py --fix

.PHONY: setup
setup:  ## Create the virtualenv and install everything
	sh scripts/setup.sh

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

.PHONY: data
data:  ## Generate the dataset ($(PRESET) preset), verifying determinism
	$(VENV_PY) scripts/make_dataset.py --preset $(PRESET) --root $(DATA) --verify 5

.PHONY: data-tiny
data-tiny:  ## The smallest dataset that exercises every stage
	$(VENV_PY) scripts/make_dataset.py --preset tiny --root $(DATA) --verify 3

$(DATA)/dataset.json:
	$(MAKE) data

# ---------------------------------------------------------------------------
# Heavy: training, export, quantisation, measurement
# ---------------------------------------------------------------------------

.PHONY: train
train: $(DATA)/dataset.json  ## Train the detector, tracked in MLflow
	$(VENV_PY) scripts/train.py --data-root $(DATA) --name $(NAME) --device $(DEVICE)

.PHONY: export
export:  ## Export FP32 ONNX and verify it against the checkpoint
	$(VENV_PY) scripts/export_onnx.py --weights $(CHECKPOINT)

.PHONY: quantise
quantise:  ## Build the calibration set and the INT8 graphs
	$(VENV_PY) scripts/quantise.py --model $(FP32) --data-root $(DATA)

.PHONY: qat
qat:  ## Quantisation-aware fine-tune, then export and quantise it
	$(VENV_PY) scripts/qat.py --weights $(CHECKPOINT) --data-root $(DATA) --device $(DEVICE)

.PHONY: evaluate
evaluate:  ## Accuracy for every precision variant, on both holdouts
	$(VENV_PY) scripts/evaluate.py --skip-missing --data-root $(DATA) \
	  --models $(FP32) $(INT8) $(INT8_PCT) $(INT8_HEAD) $(QAT_INT8) $(QAT_INT8_PCT) \
	  --torch $(CHECKPOINT) --out $(RESULTS)/accuracy.json

.PHONY: bench
bench:  ## Latency and throughput on this host (not a Jetson)
	$(VENV_PY) scripts/benchmark_latency.py --skip-missing --data-root $(DATA) --features

.PHONY: ablate
ablate:  ## Reproduce the INT8 input-range investigation (three seeds)
	$(VENV_PY) scripts/ablate_quantisation.py --data-root $(DATA) --device $(DEVICE)

.PHONY: experiments
experiments: train export quantise qat evaluate bench ablate demo  ## Every published number
	@echo
	@echo "Results are in $(RESULTS)/.  To publish them: make publish"

.PHONY: publish
publish:  ## Copy this run's results into results/published and refresh the docs
	cp $(RESULTS)/accuracy.json $(RESULTS)/latency.json $(RESULTS)/loop.json \
	   $(RESULTS)/quantisation_ablation.json $(PUBLISHED)/
	$(MAKE) docs-fix
	$(MAKE) docs-check

# ---------------------------------------------------------------------------
# The operational loop
# ---------------------------------------------------------------------------

.PHONY: demo
demo:  ## The whole loop: deploy, drift, retrain, redeploy, roll back
	$(VENV_PY) scripts/run_loop.py --data-root $(DATA) --fresh

.PHONY: demo-quick
demo-quick:  ## The same loop on short streams; what CI runs
	$(VENV_PY) scripts/run_loop.py --data-root $(DATA) --quick --fresh --no-mlflow

.PHONY: deploy
deploy:  ## Register and promote the current artefacts as a new version
	$(VENV_PY) scripts/deploy.py register --models $(FP32) $(INT8) $(INT8_PCT) \
	  --checkpoint $(CHECKPOINT) --accuracy $(RESULTS)/accuracy.json --dataset $(DATA) \
	  --reason "make deploy"
	$(VENV_PY) scripts/deploy.py promote --version $$($(VENV_PY) -c \
	  "import sys; sys.path.insert(0,'src'); from edgeinspect.registry import Registry; \
	   print(Registry().latest_version_number())") --reason "make deploy"

.PHONY: status
status:  ## What is registered and what is deployed
	$(VENV_PY) scripts/deploy.py status
	@echo
	$(VENV_PY) scripts/deploy.py history

.PHONY: edge
edge:  ## Run the edge runtime over the clean production stream
	$(VENV_PY) scripts/simulate_edge.py --stream clean

.PHONY: drift
drift:  ## Run both streams through the monitor: control first, then drifted
	$(VENV_PY) scripts/simulate_edge.py --stream clean
	$(VENV_PY) scripts/detect_drift.py --telemetry artifacts/telemetry/clean.jsonl \
	  --expect-no-trigger
	$(VENV_PY) scripts/simulate_edge.py --stream drifted --capture
	$(VENV_PY) scripts/detect_drift.py --telemetry artifacts/telemetry/drifted.jsonl \
	  --expect-trigger

.PHONY: retrain
retrain:  ## Run the retraining pipeline against the captured drifted frames
	$(VENV_PY) pipelines/retrain.py --telemetry artifacts/telemetry/drifted.jsonl \
	  --capture artifacts/capture/drifted --data-root $(DATA) --device $(DEVICE)

.PHONY: rollback
rollback:  ## Return production to the previously deployed version
	$(VENV_PY) scripts/deploy.py rollback --reason "make rollback"

.PHONY: mlflow
mlflow:  ## Open the MLflow UI against the local SQLite store
	$(VENV_PY) -m mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# ---------------------------------------------------------------------------
# Jetson.  Never executed; see docs/jetson.md.
# ---------------------------------------------------------------------------

.PHONY: engine-plan
engine-plan:  ## Print the TensorRT builds that would run (works anywhere)
	$(VENV_PY) scripts/export_tensorrt.py --onnx $(FP32) --dry-run

.PHONY: engine
engine:  ## Build TensorRT engines (Jetson only, never run)
	$(VENV_PY) scripts/export_tensorrt.py --onnx $(FP32)

.PHONY: bench-jetson
bench-jetson:  ## Benchmark on a Jetson (never run)
	sh scripts/bench_jetson.sh

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

.PHONY: up
up:  ## docker compose up: the loop, end to end, in containers
	docker compose -f docker/docker-compose.yml up --build

.PHONY: docker-runtime
docker-runtime:  ## Build the edge runtime image and run the demo inside it
	docker build -f docker/Dockerfile.runtime -t edgeinspect:runtime .
	docker run --rm -v "$(PWD)/artifacts:/app/artifacts" -v "$(PWD)/models:/app/models" \
	  -v "$(PWD)/results:/app/results" edgeinspect:runtime \
	  python scripts/simulate_edge.py --stream clean --frames 120 \
	    --model models/$(NAME)_int8.onnx --out results/docker_edge.json

.PHONY: docker-train
docker-train:  ## Build the training image
	docker build -f docker/Dockerfile.train -t edgeinspect:train .

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: clean
clean:  ## Remove run outputs, keeping results/published and the dataset
	rm -rf $(RESULTS)/*.json $(RESULTS)/*.png .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

.PHONY: clean-state
clean-state:  ## Also clear the registry, deployments, telemetry and retrain runs
	rm -rf artifacts/registry artifacts/deploy artifacts/telemetry \
	       artifacts/capture artifacts/retrain

.PHONY: clean-all
clean-all: clean clean-state  ## Also remove the dataset, the models and the MLflow store
	rm -rf $(DATA) $(MODELS)/*.onnx $(MODELS)/*.pt $(MODELS)/*.npy $(MODELS)/*.json \
	       $(MODELS)/ablation mlruns
