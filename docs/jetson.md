# The Jetson gap

**No Jetson was available for this project. No number in this repository came
from a Jetson, from TensorRT, or from any device other than the development
machine.**

That is the whole of the honest summary, and the rest of this page exists so
that the gap is specific rather than a disclaimer. Everything below is either
"here is exactly what is missing" or "here is the code that would close it, and
why it is written the way it is".

---

## What was actually run, and where

| Stage | Ran? | On what |
|---|---|---|
| Dataset generation | yes | development machine, CPU |
| Training | yes | development machine, RTX 4060 (also runs on CPU) |
| ONNX export and numerical verification | yes | CPU |
| Post-training INT8 quantisation | yes | CPU, ONNX Runtime |
| Quantisation-aware fine-tuning | yes | RTX 4060 |
| Accuracy across precisions | yes | CPU, ONNX Runtime; FP16 via torch on the 4060 |
| Latency benchmark | yes | **CPU, ONNX Runtime** |
| Edge runtime, telemetry, capture | yes | CPU, in the runtime container |
| Drift detection, retraining, deploy, rollback | yes | CPU |
| **TensorRT engine build** | **no** | needs a device |
| **TensorRT inference** | **no** | needs a device |
| **Jetson latency and power** | **no** | needs a device |
| **Camera capture** | **no** | needs a camera |

The two numbers people will want from an edge project are the ones that are
missing: how fast it is on the target, and what INT8 buys there. Those are the
Jetson rows, and they are absent rather than estimated.

## Why the CPU latency numbers do not predict the Jetson

`benchmarks/results.csv` carries a real latency measurement, and every row names
the host it came from. It is a genuine measurement of the deployed graph. It is
not a proxy for the device, for four reasons that all point the same way:

1. **INT8 on a CPU and INT8 on a Jetson are different products.** ONNX Runtime's
   CPU INT8 path is a set of quantised kernels; TensorRT on an Orin dispatches
   to tensor cores with a fundamentally different arithmetic throughput.
   Observing that INT8 is roughly the same speed as FP32 on this CPU says
   nothing at all about the device.
2. **The bottleneck moves.** At this model size on a CPU the run is dominated by
   memory traffic. On a Jetson the same graph is small enough that kernel launch
   overhead and the host-to-device copy dominate, and the fix for that is
   batching and pinned memory rather than a faster model.
3. **The Jetson throttles and the desktop does not.** A Jetson benchmark taken
   without pinning the power mode and the clocks, and without letting the board
   settle thermally, measures the DVFS governor. `scripts/bench_jetson.sh` does
   all three and records the device state next to the numbers.
4. **A TensorRT plan is not portable.** It is tied to the exact GPU
   architecture, driver and TensorRT version that built it, which is why
   `.engine` is in `.gitignore`, why the engine is built on the device at first
   start, and why there is no shortcut that produces a device number from here.

## The code that would close the gap

All of it is written, none of it has run:

| File | What it does |
|---|---|
| [`src/edgeinspect/export/tensorrt_export.py`](../src/edgeinspect/export/tensorrt_export.py) | Builds FP32, FP16 and INT8 engines, with an `IInt8EntropyCalibrator2` over the same calibration frames the ONNX Runtime path uses |
| [`src/edgeinspect/edge/backends/trt_backend.py`](../src/edgeinspect/edge/backends/trt_backend.py) | The runtime backend: engine deserialisation, device buffers, one CUDA stream |
| [`scripts/export_tensorrt.py`](../scripts/export_tensorrt.py) | The build command, with a `--dry-run` that works anywhere |
| [`scripts/bench_jetson.sh`](../scripts/bench_jetson.sh) | The benchmark procedure, including the parts people skip |
| [`configs/benchmark_jetson.yaml`](../configs/benchmark_jetson.yaml) | Engine variants plus an ONNX Runtime GPU control |
| [`docker/Dockerfile.jetson`](../docker/Dockerfile.jetson) | L4T base, JetPack-matched TensorRT, engine built on first start |
| [`requirements-jetson.txt`](../requirements-jetson.txt) | The NVIDIA-index packages, never installed |

`make engine-plan` runs the planner in dry-run mode and prints exactly what a
device would build. It works on any machine and is part of CI, which is the most
that can honestly be checked without hardware.

## Decisions that were made *for* the device, and can be checked here

The absence of a board did not stop the deployment target from shaping the
design. These are all verifiable in this repository:

**Symmetric INT8 activations.** TensorRT's INT8 activation quantisation has
per-tensor scales and no zero points. A QDQ graph with asymmetric activations
either gets rejected or silently falls back to FP16 on the affected layers,
which is a latency regression discovered on the device. This project therefore
quantises symmetrically -- and that constraint is what surfaced the input-range
problem documented in [`preprocess.py`](../src/edgeinspect/models/preprocess.py),
because asymmetric activations would have hidden it.

**ReLU rather than SiLU.** ReLU fuses into the preceding convolution in every
TensorRT precision including INT8. SiLU does not fuse in INT8 without a plugin,
and the reformat layers cost more than the activation function is worth at this
size.

**Fully convolutional, no linear layer.** A linear layer bakes the input
resolution into the graph. Rebuilding an engine because a camera changed
resolution by 32 pixels is not an acceptable failure mode.

**Static spatial size, dynamic batch.** TensorRT builds an optimisation profile
per dynamic dimension. Batch is dynamic because the runtime batches frames;
height and width are static because a bolted-down camera does not change
resolution. `--dynamic-size` exists for when it does.

**QDQ rather than QOperator.** TensorRT reads QuantizeLinear/DequantizeLinear
nodes natively and derives its own scales from them, so one file serves both the
CPU fallback and the device engine build.

**TensorRT 10 API, not TensorRT 8.** Bindings are addressed by tensor name
rather than index, and shapes are set on the execution context. Code copied from
a TensorRT 8 example does not run on JetPack 6, and that is the most common way
a bring-up loses its first day.

## What a bring-up would actually involve

Roughly a day, in this order:

1. Flash JetPack 6.x. Record the L4T version: the container tag has to match.
2. `docker build -f docker/Dockerfile.jetson`. Expect the first failure here, in
   the `onnxruntime-gpu` install, because the PyPI wheel is x86-only and has to
   be uninstalled first.
3. Copy `models/surface_defect_fp32.onnx`, its sidecar and
   `models/calibration.npy` onto the device. The calibration set travels with
   the model so the device's INT8 engine is calibrated on the same frames the
   CPU INT8 graph was.
4. `python3 scripts/export_tensorrt.py --precision fp32 fp16 int8`. Expect the
   INT8 build to take several minutes; it runs the calibrator over all 192
   frames.
5. `sudo sh scripts/bench_jetson.sh`. Do not skip `nvpmodel`, `jetson_clocks` or
   the thermal settle, and keep the tegrastats log.
6. Re-run `scripts/evaluate.py` against the engines. **Expect the INT8 accuracy
   to differ from the ONNX Runtime INT8 number.** They are different quantisers
   on the same calibration data, and treating the CPU number as a prediction of
   the device number is the mistake this whole page exists to avoid.
7. Fill in the Jetson rows of `benchmarks/results.csv` and delete this section.

## What would probably go wrong

Written down before the fact, so it can be checked afterwards:

- **The INT8 engine will not be four times faster than FP16.** At 61k parameters
  the model is small enough that fixed overheads dominate. The honest expected
  win is on model size and memory bandwidth, not on arithmetic.
- **Latency at batch 1 will be dominated by the host-to-device copy.** A
  256x256 float32 frame is 256 KB each way. Pinned memory and a persistent
  stream matter more than the precision does.
- **The first INT8 calibration will produce a worse model than expected**, and
  the cause will be the calibration set rather than TensorRT. That is the same
  failure the ONNX Runtime path has, and it is why the calibration frames are a
  committed, shared artefact rather than something each path picks for itself.
