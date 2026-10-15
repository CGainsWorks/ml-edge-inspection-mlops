# Model card: `surface-defect`

## What it is

A fully convolutional defect detector for surface inspection. One greyscale
frame in, one defect logit map out; the image-level score and the bounding boxes
both come from that single map.

| | |
|---|---|
| Architecture | `DefectNet` — five 3x3 convolutions with batch norm and ReLU, one 1x1 head |
| Parameters | 61,057 at the published width of 16 |
| Input | 256 x 256 greyscale, `(image/255 − 0.5) × 2`, in `[−1, 1]` |
| Output | 1 x 32 x 32 logit map, stride 8 |
| Image score | Mean of the 8 hottest cells |
| Deployed artefact | INT8 QDQ ONNX |
| Licence | Apache-2.0 |
| Training data | `edgeinspect-synth`, generated — see [DATASET_CARD](DATASET_CARD.md) |

## Intended use

Demonstrating an operational MLOps loop for edge inspection: quantisation,
deployment, telemetry, drift detection, automated retraining and rollback.

**Not intended for** any real inspection task. It is trained on synthetic
textures, it has never seen a camera, and it has never run on the device class
it was designed for. See [limitations.md](limitations.md).

## Design decisions, and what each one is for

**Fully convolutional, no linear layer.** A linear layer bakes the input
resolution into the graph, and rebuilding a TensorRT engine because a camera
changed resolution by 32 pixels is not an acceptable failure mode.

**61k parameters.** Small enough to train on CPU in CI, which is what makes the
*retraining* half of the loop testable on a runner with no GPU. The claim this
repository makes is about the loop, and a loop that can only be exercised on a
GPU is a loop that does not get exercised. The cost is a real accuracy ceiling
— see below.

**ReLU, not SiLU.** ReLU fuses into the preceding convolution in every TensorRT
precision including INT8. SiLU does not fuse in INT8 without a plugin, and the
resulting reformat layers cost more than the activation function is worth at
this size.

**Stride-2 convolutions, not max pooling.** Keeps the whole downsampling path in
one quantised operator type and gives the QDQ pass fewer seams.

**Batch norm, folded at export.** Kept for training stability, folded into the
convolution weights by the exporter so it costs nothing at inference and does
not become a per-channel quantisation hazard.

**A head bias of −4.** A defect is rare per frame and rarer per cell. Starting
the head strongly negative means the first epoch is not spent unlearning a
uniform prediction, which on a run this short is the difference between
converging and not.

**One head, two losses.** Per-cell segmentation against the downsampled mask and
image-level classification against the frame label. The score a threshold is
applied to at deployment is the score that was trained, and a test asserts the
NumPy and torch definitions of that score agree.

**Average pooling to build the cell targets, not nearest-neighbour.** A
two-pixel-wide scratch survives average pooling and disappears entirely under
nearest-neighbour sampling at stride 8 — taking the only training signal for the
hardest defect class with it.

## The input contract

`model_input = (image_in_[0,1] − 0.5) × 2`

This is not cosmetic. Static INT8 quantisation with **symmetric** activations
maps the input tensor onto `[−absmax, absmax]`, and for an image that is never
negative that spends half the available levels on values that cannot occur. The
frames are 8-bit to begin with, so the stem convolution sees a 7-bit image.

Measured over three seeds, paired against each model's own FP32 AUROC:

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

Symmetric activations are not negotiable — TensorRT's INT8 path has per-tensor
scales and no zero points — so centring the input is the fix that keeps every
layer in INT8 and costs one subtraction per pixel. The full reasoning, including
why the two alternative fixes are worse, is in
[`preprocess.py`](../src/edgeinspect/models/preprocess.py).

The contract lives in one module, is written into every ONNX sidecar, and is
applied by the backend rather than by each call site. A test asserts the module,
the training config and the sidecar all agree, because when they drift apart the
model is merely slightly worse rather than obviously broken.

## Training

| | |
|---|---|
| Optimiser | AdamW, lr 3e-3, weight decay 1e-4 |
| Schedule | 2 warm-up epochs, then cosine |
| Epochs | 24 |
| Batch size | 32 |
| Loss | BCE on cells (`pos_weight` 24) + 0.5 x BCE on the image logit |
| Augmentation | Flips, 90-degree rotations, bounded brightness/contrast/noise/blur |
| Tracking | MLflow, SQLite backend |

**Model selection is on an exponentially smoothed validation AUROC**, and that
detail came from being wrong first. With 400 validation frames the standard
error on AUROC is around 0.02, so raw per-epoch selection reliably picks a lucky
epoch. On the first reference run it picked epoch 1 — training loss 0.47 against
0.38 at the end — and shipped a visibly undertrained checkpoint whose validation
number looked like the best of the run. Smoothing costs nothing; the raw value
is still logged and reported.

**Augmentation is deliberately narrower than the drift.** If training
augmentation spans the `lamp_ageing` shift, the model is invariant to it, the
drift detector has nothing consequential to detect, and the recovery this
project demonstrates cannot be demonstrated.
`tests/test_config_and_augment.py` asserts that relationship against the domain
definitions, so widening the ranges fails a test rather than quietly making the
headline result unmeasurable.

## The operating point

Inspection is a false-positive-constrained problem: every false alarm stops the
line and costs an operator's attention. So the threshold is chosen on the
**good** parts at a target false positive rate, on the **validation** split, and
the recall that follows is reported rather than chosen. Selecting a threshold to
maximise F1 and then reporting that F1 is the same number twice.

Each precision variant gets its **own** threshold, because quantisation shifts
the score distribution — applying the FP32 threshold to the INT8 graph measures
the shift rather than the model. The chosen threshold is written back into the
variant's sidecar, so a deployment uses the number the evaluation measured.

**The rule includes a one-sigma safety margin, and that also came from
measurement.** Allowing exactly `target_fpr × n` exceedances is an unbiased
estimate of the threshold and a badly biased estimate of the achieved false
positive rate on new data: at a 2 percent target with 200 validation negatives
the allowance is four frames and its standard deviation is two, so half the time
the threshold lands too low. On the reference model the unadjusted rule gave 2.0
percent on validation and 3.7 percent on test. Subtracting one binomial standard
deviation moves the threshold up by roughly one order statistic and costs a
little recall — the right direction to be wrong in, because a missed defect is
caught at the next station and an operator who stops trusting the alarm is not
recoverable.

## Metrics

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

Box-level metrics are reported at IoU 0.1 rather than the customary 0.5. The
head is at stride 8, so a predicted box is accurate to eight pixels at best;
reporting mAP@0.5 against a four-pixel-wide scratch would report the stride
rather than the model. What is reported instead is the fraction of ground-truth
defects a predicted box overlaps at all — which for inspection is the number
that matters, because an operator is being told where to look — with the mean
IoU of matched pairs reported separately so the looseness is visible rather than
folded in.

## Limitations

**It plateaus.** Validation AUROC settles within a few epochs and does not move
much afterwards. That is the 61k-parameter ceiling on a deliberately hard
synthetic task, and it is the trade described above.

**It fails under the `lamp_ageing` drift**, which is the point of the
experiment, not a defect in the model. What the drift does to it, and what
retraining recovers, is in [results.md](results.md).

**Localisation is coarse.** Stride 8, one cell of resolution.

**No calibration.** The score is a ranking, not a probability. Nothing here
produces calibrated confidences, and the threshold is an operating point rather
than a probability cut.

**Synthetic data throughout.** Absolute numbers do not transfer.
[limitations.md](limitations.md) is the full accounting.

**No Jetson.** The deployment target this model was designed around was never
available. [jetson.md](jetson.md) is explicit about which claims that removes.
