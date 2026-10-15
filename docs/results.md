# Results

Every number on this page was produced by a script in this repository and lives
in `results/published/`. `make docs-check` regenerates each table from its
source and fails if any of them, or any decimal in the surrounding prose, has
drifted from the run that produced it.

**No number here came from a Jetson or from TensorRT.** See
[jetson.md](jetson.md).

---

## Accuracy across precisions

<!-- table:accuracy -->

|variant|precision|test AUROC|test recall|test FPR|drift AUROC|drift recall|bytes|
|---|---|--:|--:|--:|--:|--:|--:|
|`surface_defect_fp32`|fp32|0.8796|0.7800|0.0167|0.7266|1.0000|390,903|
|`surface_defect_int8`|int8|0.8817|0.7800|0.0167|0.7190|1.0000|113,424|
|`surface_defect_int8_percentile`|int8|0.8814|0.7800|0.0167|0.7225|1.0000|113,424|
|`surface_defect_int8_keephead`|int8|0.8816|0.7800|0.0100|0.7191|1.0000|112,792|
|`surface_defect_qat_int8`|int8|0.8857|0.7833|0.0200|0.7568|1.0000|113,424|
|`surface_defect_qat_int8_percentile`|int8|0.8877|0.7800|0.0100|0.7610|1.0000|113,424|
|`surface_defect_torch_fp32`|fp32|0.8796|0.7800|0.0167|0.7266|1.0000|--|
|`surface_defect_torch_fp16`|fp16|0.8796|0.7800|0.0167|0.7265|1.0000|--|

Operating point: threshold chosen on the validation split at a 2% target false positive rate, then applied unchanged to both test splits.  `drift` is the `lamp_ageing` holdout.

<!-- /table:accuracy -->

Reproduce with `make train export quantise evaluate`.

The operating point is chosen on the validation split at a 2 percent target
false positive rate and then applied unchanged to both test splits. The drifted
split is *not* re-thresholded: production has no labels, so re-tuning there
would measure a recovery with a knob nobody can turn. The drop shown is the drop
an operator would see.

## What the input contract is worth under INT8

Three centred and three uncentred models, each quantised four ways, each delta
paired against that model's own FP32 AUROC:

<!-- table:quantisation -->

|model input|quantisation|AUROC|vs FP32|
|---|---|--:|--:|
|`[-1, 1]` (deployed)|FP32|0.8884|--|
|`[-1, 1]` (deployed)|INT8, symmetric|0.8861|-0.0022|
|`[-1, 1]` (deployed)|INT8, asymmetric|0.8904|+0.0020|
|`[-1, 1]` (deployed)|INT8, exclude stem|0.8863|-0.0021|
|`[-1, 1]` (deployed)|INT8, per tensor weights|0.8861|-0.0023|
|`[0, 1]`|FP32|0.8853|--|
|`[0, 1]`|INT8, symmetric|0.7295|-0.1558|
|`[0, 1]`|INT8, asymmetric|0.8947|+0.0093|
|`[0, 1]`|INT8, exclude stem|0.8899|+0.0046|
|`[0, 1]`|INT8, per tensor weights|0.8900|+0.0047|

Over 3 seed(s), symmetric INT8 costs -0.1558 AUROC (spread 0.1171) on the [0, 1] model and -0.0022 (spread 0.0023) on the centred one -- a difference of 0.1536.  On the [0, 1] model, allowing zero points gives +0.0093 and excluding the stem convolution gives +0.0046; both recover most of the loss, which localises it to the input tensor rather than to the weights or the depth of the network.

<!-- /table:quantisation -->

Reproduce with `make ablate` (about eleven minutes on an RTX 4060; it trains six
models).

**The reading.** Symmetric INT8 on a `[0, 1]` input is catastrophic and
*unstable*: the three uncentred seeds lost 0.148, 0.043 and 0.277 AUROC. Every
seed is negative, but a single-seed measurement lands anywhere in that range,
which is why the ablation defaults to three seeds. On the centred model the
three seeds are −0.004, +0.001 and −0.003: the quantisation is, for practical
purposes, free.

The other two rows identify the cause rather than working around it. Allowing
asymmetric activations recovers the uncentred model, because a zero point lets
the quantiser use the whole range on a one-sided distribution. Excluding the stem
convolution recovers it, because the stem is the only layer whose input is the
image. Per-tensor weight scales change nothing, so the loss is not in the
weights.

Both recoveries are worse fixes than centring the input. Asymmetric activations
put a zero point on every activation tensor and TensorRT's INT8 path has
per-tensor scales only; excluding the stem leaves about a tenth of the
multiply-accumulates in floating point on the device where INT8 throughput is
the reason for quantising. Centring costs one subtraction per pixel and keeps
every layer in INT8.

**A note on how this was nearly missed.** The first measurement of this effect
was made on a single checkpoint and gave a much larger number than the ablation
reproduces; a second single-seed measurement gave a much smaller one. Neither
was a reliable estimate. The claim in this repository is the three-seed paired
one, and the earlier figures are not quoted anywhere.

## A finding about the entropy calibrator

`CalibrationMethod.Entropy` produces a **byte-identical graph to MinMax** when
`ActivationSymmetric` is true, on onnxruntime 1.29. The calibrator runs, prints
its progress and computes clipping thresholds; the symmetric path then discards
them in favour of the raw absolute maximum. Every activation scale matches the
MinMax scale to the last bit, and raising `CalibNumBins` changes nothing.

That is the dangerous kind of no-op: a pipeline that builds both variants,
evaluates both, and reports that "entropy calibration made no difference" is
right about the outcome and wrong about the reason. Percentile does not
degenerate — it clips, symmetric or not — so it is the second variant this
project builds. `tests/test_quantise_and_export.py` pins both behaviours, so an
onnxruntime release that fixes the entropy path fails the test rather than
passing quietly.

## Latency

<!-- table:latency -->

|variant|precision|batch|mean ms|p95 ms|fps|bytes|
|---|---|--:|--:|--:|--:|--:|
|`fp32`|fp32|1|1.141|1.849|876.6|390,903|
|`fp32`|fp32|4|4.794|6.871|834.4|390,903|
|`fp32`|fp32|8|9.250|11.946|864.9|390,903|
|`int8`|int8|1|1.404|1.813|712.2|113,424|
|`int8`|int8|4|5.107|6.260|783.2|113,424|
|`int8`|int8|8|9.999|12.423|800.1|113,424|
|`int8_percentile`|int8|1|1.290|1.773|775.5|113,424|
|`int8_percentile`|int8|4|5.844|8.567|684.5|113,424|
|`int8_percentile`|int8|8|11.487|14.692|696.4|113,424|
|`int8_keephead`|int8|1|1.243|1.643|804.6|112,792|
|`int8_keephead`|int8|4|5.383|7.288|743.0|112,792|
|`int8_keephead`|int8|8|11.627|17.084|688.0|112,792|
|`qat_int8`|int8|1|1.417|2.125|705.9|113,424|
|`qat_int8`|int8|4|5.957|8.492|671.5|113,424|
|`qat_int8`|int8|8|11.577|15.521|691.0|113,424|

Measured on Windows-11-10.0.26200-SP0, 4 threads pinned, 300 iterations after 30 warm-up calls.  **Not Jetson numbers** -- no Jetson was available; see [jetson.md](../jetson.md).

Telemetry feature extraction costs 2.185 ms per frame (p95 2.810 ms) on the same host, reported separately from inference rather than folded into it.

<!-- /table:latency -->

Reproduce with `make bench`.

**These are not device numbers.** ONNX Runtime's CPU INT8 kernels and TensorRT's
tensor-core INT8 path are different products, so the INT8-versus-FP32 result
here says nothing about a Jetson. What does transfer is the file size: INT8 is
about a third of FP32 on disk, and for a device that receives models over the
air that matters on its own.

**Telemetry costs more than inference.** The feature extraction that makes drift
detection possible runs at roughly twice the cost of the INT8 forward pass at
this model size. That is a real operational trade and the runtime reports the
two separately rather than folding one into the other. It is also a consequence
of how small the model is: the features are a fixed per-frame cost, so their
share falls as the model grows.

## The drift monitor

<!-- table:drift -->

|stream|frames|windows|triggered|false alarms|onset|first trigger|detection latency|
|---|--:|--:|---|--:|--:|--:|--:|
|`clean` (control)|900|15|False|0|--|--|--|
|`drifted`|1200|--|True|--|200|300|100 frames|

The control row is the important one: same imaging conditions as the reference window, different random draws.  A monitor that fires there is a monitor nobody will leave switched on.

The sequential Page-Hinkley detector first accumulated past its threshold at frame 210, against a windowed first trigger at frame 300.

<!-- /table:drift -->

Reproduce with `make demo`.

The control row is the one that matters. The `clean` stream runs under the same
imaging conditions as the reference window with different random draws — a
trigger there is a false alarm, and CI fails on one.

The gap between the windowed and sequential detectors is the reason both exist.
The windowed test compares whole distributions and is what triggers retraining;
it can only locate an onset to within one stride. Page-Hinkley accumulates frame
by frame and puts the onset within a few frames, which is what makes "detected N
frames after onset" a measurement rather than an estimate.

### Two corrections the control stream forced

**PSI is biased upward at small windows.** The conventional 0.1 warn and 0.25
alarm thresholds are large-sample values. At a 200-frame window against a
400-frame reference with ten bins, two samples of *identical* data read about
0.067 on average with a ninety-fifth percentile of 0.128 — so the warn line
fires on nothing at all about a third of the time, and the first end-to-end run
duly false-alarmed. The detector now subtracts the analytic null expectation,
`(bins − 1)(1/n_current + 1/n_reference)`, which
`tests/test_drift.py` checks against simulation at four sample sizes.

**Page-Hinkley must not watch the model's score.** It assumes roughly symmetric
noise around a stable mean; the score distribution is a mixture of a dense
good-part mass and a sparse defective tail, so a run of defective parts arriving
together looks exactly like a step change. It fired at frame 244 of a clean
control stream while every windowed rule sat at zero. It now watches only the
image statistics.

Its parameters also changed for a reason worth recording: under no drift the
accumulator's largest excursion above its running minimum grows like
`sigma^2 / (2 * delta) * log(n)`, so with `delta = sigma/2` an eight-sigma
threshold is reached at about 3,000 frames — under two minutes at 30 fps. A
sequential detector guaranteed to fire eventually on a stationary signal is a
clock. `delta = sigma` with a ten-sigma threshold pushes the expected false
alarm past 10^4 frames.

## The loop, end to end

<!-- table:recovery -->

|stage|outcome|
|---|---|
|baseline deployed|v1|
|drift detected at frame|300|
|frames captured on the device|862|
|**flagged rate, post-drift, incumbent**|1.0000|
|retrained candidate: reference-holdout AUROC|0.8861|
|retrained candidate: drifted-holdout AUROC|0.9041|
|retraining gate|**promote**|
|deployed after retraining|v2|
|**flagged rate, post-drift, retrained**|0.0533|
|monitor quiet under the new normal|True|
|rolled back to|v1|
|device confirmed serving|v1|
|whole loop|373.6 s|

The two flagged rates are the same 600-frame post-drift stream through two models, so they are a comparison rather than an anecdote.  The true defect rate on that stream is 8%.  The monitor is quiet afterwards because the reference window was recaptured under the post-drift conditions when the new version was promoted; capturing it under the old conditions would leave it firing forever on a model that has already adapted.

<!-- /table:recovery -->

### The retraining gate

<!-- table:gate -->

|check|metric|candidate|incumbent|requirement|result|
|---|---|--:|--:|--:|---|
|improvement|`drift.image.auroc`|0.9041|0.7190|0.030|PASS|
|regression|`reference.image.auroc`|0.8861|0.8817|0.020|PASS|
|floor|`reference.image.auroc`|0.8861|--|0.850|PASS|
|floor|`drift.image.auroc`|0.9041|--|0.750|PASS|
|ceiling|`reference.image.false_positive_rate`|0.0333|--|0.050|PASS|

<!-- /table:gate -->

The gate is asymmetric on purpose: the candidate must improve materially on the
distribution that triggered the alert, and must not have paid for that by
regressing on the distribution it already handled. A gate with only the first
condition promotes a model that has forgotten the original domain, and because
the trigger came from the drifted domain nothing else in the pipeline would
notice.

### How the drift was calibrated to be recoverable

The floors in `configs/deploy.yaml` are set against a measured reference point
rather than chosen by taste: what a model trained *directly* on 1,200 frames of
the drifted domain reaches.

It is a reference point and not an upper bound, and the distinction turned out
to matter — the retrained model clears it, because it trains on the reference
domain as well, which is both more data and a domain that shares most of what
there is to learn about defect appearance.

The first version of the `lamp_ageing` domain reached only **0.709** AUROC even
when trained on nothing but that domain,
and the reason turned out to be a modelling error rather than a hard physical
limit: the domain lowered `defect_contrast` explicitly *and* lowered `gain`, and
since the generator applies gain to the whole frame the defect was being dimmed
twice. Combined with the blur and the extra noise that put the defects under the
noise floor, so no retraining could have recovered from it and the demonstration
would have been of an unrecoverable fault rather than a drift.

Removing the double-counting leaves gain to do the dimming, which is the
physics: an effective defect contrast of 0.30 × 0.72. The domain is still a
clear fault — dimmer, softer, noisier — and it is now a fault a model can adapt
to.

**Measuring that reference point before setting the gate is the general
point.** A gate threshold above what the data permits blocks every candidate
forever and looks like a broken pipeline; one below what a trivial model reaches
passes everything and looks like a working one.

## Reproducing all of it

```bash
make setup
make data          # about 130 s, deterministic
make experiments   # train, qat, export, quantise, evaluate, bench, ablate, demo
make publish       # copy into results/published and regenerate the docs
make docs-check    # verify every number on this page
```

About thirty minutes end to end on an RTX 4060, most of it the six models the
quantisation ablation trains. Everything except the ablation runs on CPU; the
whole loop runs on a CI runner in a couple of minutes at the `tiny` preset.
