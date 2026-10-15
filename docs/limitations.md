# Limitations

What this repository does not show, cannot show, and would need in order to
show it. Ordered by how much it should change your reading of the results.

---

## 1. There is no Jetson, and no TensorRT number anywhere

The headline hardware claim of an edge project is latency on the target, and it
is absent. No engine was built, no engine was benchmarked, and the latency
numbers here are ONNX Runtime on a desktop CPU with the host recorded in every
row. [`jetson.md`](jetson.md) is the full accounting: what ran, what did not,
the code that would close the gap, and the design decisions the target shaped
anyway.

**What this invalidates:** every Jetson-shaped inference you might draw. The
CPU INT8 result in particular does not predict the device, because ONNX
Runtime's CPU INT8 kernels and TensorRT's tensor-core INT8 path are different
products.

**What it does not invalidate:** the operational results. Drift detection
latency, retraining recovery, gate behaviour and rollback are properties of the
loop, and the loop ran.

## 2. There is no camera, and no real defect

The dataset is generated. `edgeinspect.data.synth` builds textured surfaces with
four kinds of defect under parameterised imaging conditions, and the "production
streams" are the same generator under different conditions.

This was a choice, not only a constraint, and the reasoning is in
[`synth.py`](../src/edgeinspect/data/synth.py): a drift experiment needs a shift
with a known cause and a known magnitude, applied deliberately. A fixed public
dataset cannot provide that, and inventing one by relabelling a different MVTec
category as "drift" would be worse than synthetic honesty.

**What this invalidates:** the absolute accuracy numbers. An AUROC on synthetic
textures is a statement about this generator. Do not read it as a statement
about a production line.

**What it does not invalidate:** the *relative* results, which are what the
repository publishes — the INT8 delta, the drop under drift, the recovery after
retraining, the false-alarm behaviour of the monitor. Those are comparisons
under one fixed data distribution.

**How to close it:** `scripts/make_dataset.py --source mvtec` converts a local
copy of MVTec AD into the same layout, after which every stage runs unchanged.
The licence prevents shipping it; see [`licences.md`](licences.md).

## 3. Labels for retraining come from an oracle

The retraining pipeline collects frames the device kept, labels them, and
retrains. On a real line those frames go to a human, and the pipeline blocks for
hours or days. Here `stage_label` reads the generator's ground-truth mask.

**What this hides:** the dominant cost and the dominant latency of a real
retraining cycle, and the label noise a human introduces. A drift-to-redeploy
time measured here is a measurement of compute, not of a process.

**What is still real:** the shape of the step. A frame without a mask is
dropped rather than assumed defect-free — treating unlabelled frames as good is
the fastest way to teach a retrained model to miss the defect that triggered the
retraining — and `--label-latency-frames` exists so a runbook can quote a
realistic cycle time rather than this script's.

## 4. The production streams are simulated, but the runtime is not

`scripts/simulate_edge.py` replaces the camera and nothing else. The inference,
the thresholding, the box extraction, the feature computation, the latency
measurement, the telemetry format, the capture policy and the deploy agent are
the code that would run on the device, in the container built without a training
framework.

**What is simulated:** the optics and the sensor.
**What is not:** everything downstream of them.

## 5. One device, not a fleet

The registry, the deployment manifest and the agent are written for one model
and one device. A fleet needs staged rollout, per-device version reporting,
partial rollback and a way to answer "which devices are on v3". None of that is
here. The registry API is the shape a real one would have, so the storage could
be swapped for MLflow's Model Registry without changing anything above it, but
that is a claim about design rather than a working feature.

## 6. The drift thresholds are calibrated on one control stream

`configs/drift.yaml` was tuned by requiring zero triggers on the `clean` control
stream and at least one on `drifted`. That is a real calibration and it is
reproducible, but it is one control stream from one generator.

Two specific consequences:

- **The false alarm rate is measured, not bounded.** Zero triggers in a few
  hundred windows is consistent with a false alarm rate of a few percent per
  window. A production deployment would want far more control data than this.
- **The detectors have a shared blind spot.** All of them watch first- and
  second-order image statistics plus the model's own output. A shift that
  preserves those while changing structure would pass. `texture_scale` and the
  `material_batch` domain exist because of exactly that gap, but the gap is
  narrowed rather than closed.

## 7. The model is deliberately small, and that shows

61k parameters, chosen so the retraining half of the loop runs on a CI runner
with no GPU. On the reference dataset it plateaus well short of what a larger
model would reach: validation AUROC settles around 0.87 within a few epochs and
does not move much afterwards.

That is a real ceiling and it is the right trade for this repository — a loop
that can only be exercised on a GPU is a loop that does not get exercised — but
the accuracy numbers should be read as "a small model on a hard synthetic task",
not as an attempt at the state of the art.

The localisation head is at stride 8, so a predicted box is accurate to eight
pixels at best. `docs/MODEL_CARD.md` reports box metrics at an IoU threshold
that reflects the stride rather than one that flatters it.

## 8. What has never been run

- `scripts/export_tensorrt.py` without `--dry-run`
- `src/edgeinspect/edge/backends/trt_backend.py`
- `scripts/bench_jetson.sh`
- `docker/Dockerfile.jetson` (arm64 and L4T only; it cannot be built here)
- `scripts/make_dataset.py --source mvtec` against a real MVTec AD copy —
  the adapter is written and unit-tested against its own layout, but no
  licensed copy was downloaded

Everything else in this repository has been executed, and the numbers it
publishes came from those executions.
