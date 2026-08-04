# ml-edge-inspection-mlops

> **Status: planned. This repository is a brief, not an implementation.**
> Nothing here is built yet. The sections below define what goes in it.

The emphasis here is the operational loop rather than the model. A defect detector is trained, quantised, compiled to TensorRT and deployed to a Jetson, then monitored for drift in production, with drift triggering an automated retraining and redeployment pipeline. Uses public defect datasets such as MVTec AD or DAGM.

---

## What this covers

CV entries this repository is the evidence for:

- Real-Time Visual Inspection Model, edge AI
- MLOps and CI/CD for edge model delivery (Convex Technology)

---

## Requirements

### Hardware

- NVIDIA Jetson
- Industrial or USB camera
- Controlled lighting
- Sample parts (or public dataset imagery)
- Training GPU

### Software

- PyTorch
- Labelling and augmentation pipeline
- Post-training and quantisation-aware quantisation
- TensorRT compilation
- Drift monitoring (Evidently or custom)
- MLflow
- Docker
- CI/CD for edge delivery

### Report and documentation deliverables

- Dataset card
- Accuracy against latency trade-off across precision levels
- FP32 against INT8 comparison
- Drift detection methodology
- Retraining runbook
- Edge deployment and rollback procedure

---

## Inputs

- Labelled defect images
- Augmentation config
- Calibration dataset for quantisation
- Live camera stream
- Production inference logs

## Outputs

- Trained checkpoint
- TensorRT engine file
- Per-frame defect classification with bounding boxes
- Latency and throughput metrics
- Drift alerts
- Retraining triggers
- Versioned edge deployments

---

## Intended structure

```
configs/
src/
  data/                   loaders, augmentation, labelling helpers
  train/                  training loop, MLflow logging
  quantise/               PTQ, QAT, calibration
  export/                 TensorRT engine build
  edge/                   Jetson inference runtime, telemetry
  monitoring/             drift detectors, alerting
pipelines/                retraining DAG, deployment pipeline
benchmarks/               precision vs accuracy vs latency
.github/workflows/        train, export, deploy-edge
docs/                     RUNBOOK.md, MODEL_CARD.md
```

Plus the standard baseline: `README.md`, `LICENSE`, `.gitignore`, `.gitattributes`
(Git LFS rules for weights, bags and media), `.github/workflows/ci.yml`, `docs/`,
`scripts/setup.sh`, `scripts/run_demo.sh` and `tests/`.

---

## Build checklist

- [ ] Public defect dataset ingested with a dataset card
- [ ] Baseline PyTorch detector trained with MLflow tracking
- [ ] Augmentation pipeline
- [ ] Post-training quantisation with a calibration set
- [ ] TensorRT export at FP32, FP16 and INT8
- [ ] Jetson deployment with latency benchmarks
- [ ] Accuracy vs latency trade-off table
- [ ] Inference telemetry collection from the edge
- [ ] Drift detector on production logs
- [ ] Automated retraining pipeline triggered by drift
- [ ] Versioned deployment and rollback
- [ ] Runbook

---

## Definition of done

This repository is finished when:

- [ ] The whole thing runs from one command on a clean machine.
- [ ] The README opens with a diagram or a GIF, not a wall of text.
- [ ] Every number quoted in the README is reproducible from a script in the repo.
- [ ] Limitations are stated honestly in their own section.
- [ ] A licence is declared.
- [ ] Large binaries live in Releases or LFS, not in the Git history.
- [ ] The repository URL is added to the matching CV bullet point.

---

## Starter prompt

Paste this into your coding agent of choice to begin.

<details>
<summary>Expand</summary>

```
You are helping me build a portfolio project from scratch. Read this brief and then
propose a concrete implementation plan before writing any code.

PROJECT: ml-edge-inspection-mlops
The emphasis here is the operational loop rather than the model. A defect detector is trained, quantised, compiled to TensorRT and deployed to a Jetson, then monitored for drift in production, with drift triggering an automated retraining and redeployment pipeline. Uses public defect datasets such as MVTec AD or DAGM.

TARGET STACK
- PyTorch
- Labelling and augmentation pipeline
- Post-training and quantisation-aware quantisation
- TensorRT compilation
- Drift monitoring (Evidently or custom)
- MLflow
- Docker
- CI/CD for edge delivery

HARDWARE CONTEXT
- NVIDIA Jetson
- Industrial or USB camera
- Controlled lighting
- Sample parts (or public dataset imagery)
- Training GPU

THE SYSTEM MUST TAKE THESE INPUTS
- Labelled defect images
- Augmentation config
- Calibration dataset for quantisation
- Live camera stream
- Production inference logs

AND PRODUCE THESE OUTPUTS
- Trained checkpoint
- TensorRT engine file
- Per-frame defect classification with bounding boxes
- Latency and throughput metrics
- Drift alerts
- Retraining triggers
- Versioned edge deployments

INTENDED REPOSITORY LAYOUT
configs/
src/
  data/                   loaders, augmentation, labelling helpers
  train/                  training loop, MLflow logging
  quantise/               PTQ, QAT, calibration
  export/                 TensorRT engine build
  edge/                   Jetson inference runtime, telemetry
  monitoring/             drift detectors, alerting
pipelines/                retraining DAG, deployment pipeline
benchmarks/               precision vs accuracy vs latency
.github/workflows/        train, export, deploy-edge
docs/                     RUNBOOK.md, MODEL_CARD.md

CONSTRAINTS
- Everything must run from a single command: `docker compose up` or `make demo`.
- No placeholder or stub code presented as working. If something is not implemented, say so.
- Every quantitative claim in the README must be reproducible from a script in this repo.
- Use only public or synthetic data. Record the licence of every dataset used.


START BY
1. Confirming or challenging the proposed repository layout.
2. Listing the specific libraries and versions you intend to pin.
3. Identifying the single riskiest technical unknown and how to de-risk it first.
Then work through the milestones in the README build checklist, one at a time.

```

</details>

---

## Notes

- Do not backdate commits to match CV dates. Publishing older work today is normal and expected.
- Record dataset licences before training on anything. Several common research datasets are
  non-commercial only.
- Flip this repository to public once there is something in it worth showing.
