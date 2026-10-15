# Drift detection

How the monitor works, why each detector is in it, and what it is blind to.

The one-line version: a frozen reference window captured at deployment, ten
statistical rules over a sliding window, plus a sequential detector that answers
"when did it start". A trigger needs two alarms or four warnings, and the
thresholds were set by requiring **zero** triggers on a control stream that does
not drift.

---

## What the device reports

Eight image statistics per frame, plus the model's own score and decision. All
of them are computed on the device inside the inference budget; the runtime
reports that cost separately rather than folding it into inference time,
because it is the price of being able to detect drift at all and it should be
visible.

| Signal | What it measures | Moves when |
|---|---|---|
| `brightness` | Frame mean | Illumination changes, exposure changes |
| `contrast` | Frame standard deviation | Illumination, or surface finish |
| `sharpness` | Variance of the Laplacian of the smoothed frame, **minus the noise floor** | Focus drifts, lens gets dirty |
| `noise_level` | Immerkaer estimator | Sensor gain rises, temperature rises |
| `edge_density` | Fraction of pixels above a fixed gradient threshold | Focus, texture |
| `entropy` | Histogram entropy over 32 bins | Dynamic range collapses or expands |
| `highlight_fraction` | Fraction above 0.85 | Specular behaviour, over-exposure |
| `texture_scale` | Fine-grain energy over total variation | Surface finish changes without illumination changing |
| `score` | The model's image-level score | Anything the model reacts to |
| `decision` | Flagged or not | The operating point, or the parts |

### The sharpness measure took two attempts

The obvious focus measure — the variance of the Laplacian of the raw frame —
**goes the wrong way** on the scenario this project is built around.

A dying lamp does two things at once: the scene gets dimmer and softer, and the
camera compensates for the dimmer scene with gain, which raises read noise. Read
noise is broadband, so it dominates the Laplacian. The naive measure *rose* by a
factor of about four while the image was visibly blurring.

An alert that says "sharpness increased" while the lens is going soft is worse
than no alert, so the measure is now: estimate the noise separately with
Immerkaer's mask, smooth the frame, take the Laplacian variance of that, and
subtract the noise contribution analytically. The 3x3 box convolved with the
4-neighbour Laplacian gives a 5x5 kernel whose squared coefficients sum to
32/81, so iid noise of standard deviation *s* contributes exactly (32/81)·*s*²
and that term comes off. `tests/test_features_and_metrics.py` verifies the
constant against simulation and pins the failure case.

Both signals now move the way their names say, and the pair is diagnostic:
sharpness down *and* noise up is "the lamp is dying and the camera is
compensating", which is a work order.

## The reference window

Captured at deployment, from the newly deployed model's own telemetry on
known-good frames, and stored with the deployment as
`artifacts/deploy/reference_v<n>.json`.

Two properties, both load-bearing:

**It is frozen.** Comparing the current window against a rolling recent window
is the design that cannot see a slow ramp — nothing is ever different from ten
minutes ago, and the lamp-ageing scenario is precisely a slow ramp.

**It is per version.** Score distributions are not comparable across model
versions. A monitor that keeps comparing v2's scores against v1's reference
fires on the deployment itself, every time, until somebody switches it off —
which is how monitoring actually dies.

## The detectors

Four kinds, chosen because they have different blind spots.

### PSI, with a bias correction that matters

Population stability index over ten reference-quantile bins. Thresholds 0.1 warn
and 0.25 alarm, which is the convention operations teams already know.

**Those conventional thresholds do not work at a 200-frame window, and finding
that out was the single largest correction in this project.** PSI is a plug-in
divergence estimate and is biased upward by sampling noise. The bias is

    E[PSI] ≈ (bins − 1) × (1/n_current + 1/n_reference)

which at ten bins, a 200-frame window and a 400-frame reference is **0.067**,
with a ninety-fifth percentile of **0.128**. Two samples of *identical* data
clear the 0.1 warn line about a third of the time.

The first end-to-end run duly fired on a control stream that had not drifted at
all. That is not a threshold that needs tuning; it is a statistic that needs
correcting. `edgeinspect.monitoring.detectors.psi` now subtracts the analytic
null expectation and floors at zero, so zero means "no detectable shift" at any
window size and the conventional thresholds keep the meaning they have in the
literature. The raw value is kept in the result's `detail` so nothing is hidden,
and the formula is checked against simulation at four sample sizes in
`tests/test_drift.py`.

### Kolmogorov-Smirnov, on the statistic rather than the p-value

Two-sample KS on the model's score distribution. The **statistic** is
thresholded, not the p-value: with 200 samples any real shift gives p < 1e-6, so
a p-value threshold is a step function with no useful resolution. The p-value is
reported anyway, for anybody who wants it.

### Standardised mean shift, for the alert text

Absolute mean shift in reference standard deviations. Redundant with PSI as a
detector and not redundant as an explanation: PSI says the distribution moved,
this says by how much and in which direction. "Brightness fell by 4.9 reference
standard deviations" is the sentence that goes in the alert.

### Page-Hinkley, for the onset

Windowed tests can only locate an onset to within one stride. Page-Hinkley
accumulates deviation frame by frame and fires at a specific index, which is what
makes "detected N frames after onset" a number rather than an estimate.

Two things about it were learned the hard way.

**Its false-alarm time is finite and depends on the run length.** Under no drift
the accumulator is a random walk with drift −δ, and its largest excursion above
its running minimum grows like (σ²/2δ)·log(n). With δ = σ/2 — the first choice
here — an 8σ threshold is reached at about 3,000 frames, which at 30 fps is
under two minutes. A sequential detector guaranteed to fire eventually on a
stationary signal is a clock. The parameters are now δ = σ and threshold = 10σ,
which pushes the expected false alarm past 10⁴ frames.

**It must not watch the model's score.** Page-Hinkley assumes roughly symmetric
noise around a stable mean. The score distribution is a *mixture*: a dense mass
of good parts and a sparse high tail of defective ones. A run of defective parts
arriving together looks exactly like a positive step, and on the first
end-to-end run it fired at frame 244 of a clean control stream while every
windowed rule sat at zero. The sequential detector now watches only the image
statistics; the windowed tests handle `score` correctly because they compare
whole distributions rather than a running mean.

### Prediction-rate shift

Two-proportion z-test on the flagged rate. The one signal that needs no image
features at all, and the one an operations team notices first. A monitor that
does not watch it is behind the humans.

## The trigger policy

```yaml
trigger:
  min_alarms: 2        # two independent rules, not one
  min_warns: 4
  cooldown_windows: 2  # the drift is still there while retraining runs
```

One alarm is not enough. With ten rules over a 200-frame window, a single
rule crossing its alarm line once in a long run is ordinary, and a monitor that
retrains on it retrains constantly and gets switched off.

The cooldown exists because the drift does not stop while the retraining runs.
Without it the same event fires every stride until the new model is deployed,
and the alert channel becomes noise at exactly the moment somebody is reading
it.

## How the thresholds were set

Not by taste. `configs/data.yaml` defines a `reference_holdout` domain with the
same physics as `reference` and different random draws, and the `clean` stream
runs under it. The requirement is:

- **zero triggers on `clean`** — a trigger there is a false alarm;
- **at least one trigger on `drifted`** — and as early as possible.

Both directions are asserted, in CI, on every push:

```bash
python scripts/detect_drift.py --telemetry artifacts/telemetry/clean.jsonl \
    --expect-no-trigger      # exits non-zero if it fires
python scripts/detect_drift.py --telemetry artifacts/telemetry/drifted.jsonl \
    --expect-trigger         # exits non-zero if it does not
```

A drift detector that has only ever been shown drifting data has never been
shown not to fire.

## Measured behaviour

<!-- table:drift -->

|stream|frames|windows|triggered|false alarms|onset|first trigger|detection latency|
|---|--:|--:|---|--:|--:|--:|--:|
|`clean` (control)|900|15|False|0|--|--|--|
|`drifted`|1200|--|True|--|200|300|100 frames|

The control row is the important one: same imaging conditions as the reference window, different random draws.  A monitor that fires there is a monitor nobody will leave switched on.

The sequential Page-Hinkley detector first accumulated past its threshold at frame 210, against a windowed first trigger at frame 300.

<!-- /table:drift -->

## What this monitor is blind to

Stated rather than discovered later:

**A shift that preserves first- and second-order statistics.** Every rule here
watches a summary statistic or the model's own output. A change in the spatial
*arrangement* of texture that leaves brightness, contrast, gradient energy and
the score distribution alone would pass. `texture_scale` and the
`material_batch` domain exist because of exactly that gap, and they narrow it
rather than close it. Closing it properly needs an embedding-based detector,
which needs a reference embedding set on the device and a distance whose scale
nobody can interpret at 03:00 — a real trade, made deliberately.

**A slow shift that stays inside the reference spread.** The reference window is
400 frames. A drift far slower than that, or smaller than the reference's own
variability, is not detectable by any windowed test against it.

**Label drift.** There is no ground truth in production, so a change in the
*true* defect rate is indistinguishable from a change in the model's behaviour.
`rate_shift/decision` will fire; the alert cannot say which happened, and the
runbook sends that case to a human.

**A false alarm rate that is measured rather than bounded.** Zero triggers over
a few hundred control windows is consistent with a per-window false alarm rate
of a few percent. A production deployment would want far more control data
before quoting a number.
