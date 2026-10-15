# ml-edge-inspection-mlops

**A defect detector for edge inspection, and the operational loop that keeps it
alive.** Train → quantise → export → deploy with a hash check and a smoke test →
collect telemetry from the device → detect drift → retrain from what the device
captured → pass a gate that can say no → redeploy behind a canary → roll back.

The model is the least interesting part. What this repository is about is the
five places the automated path is allowed to stop.

```
   train ──▶ export ──▶ quantise ──▶ evaluate ──▶ ┌──────────┐
              │            │                      │ REGISTRY │
        verify vs torch  shared calibration       └────┬─────┘
        (fails the build)  set, ONNX + TensorRT        │ manifest: version, sha256, threshold
                                                       ▼
                                            ┌─────────────────────┐
                                            │  EDGE AGENT         │ refuses a hash mismatch
                                            │  verify→smoke→swap  │ refuses a model that loads
                                            └──────────┬──────────┘ but does not discriminate
                                                       │
                                    telemetry ◀────────┴───────▶ sampled frames kept
                                        │                             on the device
                                        ▼
                              ┌───────────────────┐
                              │  DRIFT MONITOR    │  frozen reference, per version
                              │ windowed + Page-  │  10 rules, 2 alarms to trigger
                              │ Hinkley for onset │
                              └─────────┬─────────┘
                                        │ trigger
                              ┌─────────▼─────────┐
                              │ RETRAIN PIPELINE  │  captured frames + replay buffer
                              └─────────┬─────────┘
                              ┌─────────▼─────────┐
                              │  GATE             │  must improve on the drifted holdout
                              │  promote / BLOCK  │  must not regress on the original one
                              └────┬─────────┬────┘
                                   ▼         ▼
                            deploy+canary   blocked, kept in the registry
                                   │
                                   ▼ canary fails
                              ROLLBACK to the previous *deployed* version
```

Full diagram and rationale: **[docs/architecture.md](docs/architecture.md)**.

---

## The one honest caveat, first

**No Jetson was available, so there is no TensorRT number anywhere in this
repository.** The engine builder, the TensorRT runtime backend, the benchmark
script and the L4T image are all written and none of them has ever been
executed. The latency numbers here are ONNX Runtime on a desktop CPU, and every
row names the host it came from.

[docs/jetson.md](docs/jetson.md) is the full accounting: what ran, what did not,
the code that would close the gap, what a bring-up would involve, and the design
decisions the target shaped anyway. [docs/limitations.md](docs/limitations.md)
lists everything else this does not show.

What *did* run is the loop, end to end, on every push.

## Results

### The loop

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

### The drift monitor

<!-- table:drift -->

|stream|frames|windows|triggered|false alarms|onset|first trigger|detection latency|
|---|--:|--:|---|--:|--:|--:|--:|
|`clean` (control)|900|15|False|0|--|--|--|
|`drifted`|1200|--|True|--|200|300|100 frames|

The control row is the important one: same imaging conditions as the reference window, different random draws.  A monitor that fires there is a monitor nobody will leave switched on.

The sequential Page-Hinkley detector first accumulated past its threshold at frame 210, against a windowed first trigger at frame 300.

<!-- /table:drift -->

The control row is the point. The `clean` stream runs under the same imaging
conditions as the reference window with different random draws, so a trigger
there is a false alarm — and CI fails on one. A drift detector that has only
ever been shown drifting data has never been shown not to fire.

### Accuracy across precisions

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

Full detail, including two statistical corrections that a control stream forced
and a quantisation finding that nearly got away: **[docs/results.md](docs/results.md)**.

## Quickstart

```bash
make setup         # venv, runtime deps, CPU torch
make data          # generate the dataset, verify it is deterministic
make train export quantise evaluate
make demo          # the whole loop: deploy, drift, retrain, redeploy, roll back
```

No GPU needed for any of that. The whole loop is about six minutes, most of it
the retraining step; `make demo-quick` is the two-minute version and is what CI
runs on every push.

In containers instead:

```bash
docker compose -f docker/docker-compose.yml up --build
```

## What makes the loop trustworthy

This is the part worth reading.

**The drift detector is tested in both directions.** Firing on drift is easy.
The test that matters is the control stream — same conditions as the reference,
different draws — where a trigger is a false alarm. Both assertions run in CI:

```bash
python scripts/detect_drift.py --telemetry artifacts/telemetry/clean.jsonl   --expect-no-trigger
python scripts/detect_drift.py --telemetry artifacts/telemetry/drifted.jsonl --expect-trigger
```

**The control stream found two real statistical bugs.** PSI is biased upward at
small windows: at this project's 200-frame window against a 400-frame reference,
two samples of *identical* data read about 0.067 with a ninety-fifth percentile
of 0.128 — so the conventional 0.1 warn threshold fires on nothing at all about
a third of the time, and it did. The detector now subtracts the analytic null
expectation. Separately, Page-Hinkley fired at frame 244 of a clean stream
because the model's score is a mixture distribution and a run of defective parts
looks like a step change; it now watches only the image statistics.

**The gate is asymmetric and it blocks things.** A retrained model must improve
materially on the distribution that triggered the alert *and* must not have
regressed on the distribution it already handled. A gate with only the first
condition promotes a model that has forgotten the original domain — and because
the alert came from the drifted domain, nothing else would notice. Blocked
versions stay in the registry with their reason.

**The gate thresholds are set against a measured reference point**, not chosen
by taste: what a model trained *directly* on the drifted domain reaches.
Measuring it first caught a modelling error — the drift domain was dimming
defects twice, which put them under the noise floor and held even a
drift-trained model to 0.709 AUROC. That would have been a demonstration of an
unrecoverable fault dressed up as a drift.

**A new model is proved on the device before the old one is dropped.** The agent
verifies the artefact hash, loads the candidate *alongside* the running model,
runs a probe pair, and only then swaps. If the candidate fails, the old model
keeps serving and the failure is reported. Loading proves the file is a graph;
it does not prove it is the model.

**Rollback follows the deployment log, not version arithmetic.** "Version minus
one" is the wrong model as soon as anything has been skipped or blocked.

**Quantisation is measured, not assumed.** The model input is centred on zero
because symmetric INT8 on a `[0, 1]` image spends half its levels on values that
cannot occur. Over three seeds that is worth 0.156 AUROC — and the effect is
*unstable*, so the ablation runs three seeds by default because one cannot tell
the difference between a catastrophe and noise.

**The edge runtime cannot import torch.** A test imports it in a subprocess and
fails if torch appears in `sys.modules`; the runtime Dockerfile runs the same
check at build time. That test found a real leak — the sidecar helpers lived in a
module importing torch at top level — which would have made the runtime image
fail on first start with an `ImportError` three layers from the cause.

**CI runs the whole loop on every push.** Not the unit tests around it: the
loop itself, including the control-stream assertion, the gate decision and the
rollback. A weekly job re-samples the false-alarm rate across six independent
control streams, because zero triggers in one run is not a rate.
[docs/ci.md](docs/ci.md) covers all three workflows and what each is for.

**Every number in this documentation traces to a file in `results/published/`.**
`make docs-check` regenerates each table from its source and scans the prose for
decimals that appear in no published result. Tables get regenerated; sentences do
not, and that is how documentation actually rots.

## What is in here

| | |
|---|---|
| [architecture.md](docs/architecture.md) | The loop, the five refusals, the module map |
| [RUNBOOK.md](docs/RUNBOOK.md) | On-call procedures: an alert fired, the gate blocked, roll back |
| [drift.md](docs/drift.md) | Every detector, why it is there, what the monitor is blind to |
| [ci.md](docs/ci.md) | The three workflows, what each asserts, and how they are kept inside a free account's budget |
| [results.md](docs/results.md) | All the numbers, and how each was produced |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | The model, the input contract, the operating point |
| [DATASET_CARD.md](docs/DATASET_CARD.md) | The generator, the domains, the streams |
| [jetson.md](docs/jetson.md) | What was not run, and what it would take |
| [limitations.md](docs/limitations.md) | Everything this does not show |
| [licences.md](docs/licences.md) | Data provenance and dependency licences |

## Commands

`make help` lists them all. The ones that matter:

| | |
|---|---|
| `make demo` | The whole loop, about seven minutes |
| `make demo-quick` | The same, two minutes; what CI runs |
| `make check` | Lint, tests, docs — exactly what CI runs |
| `make experiments` | Regenerate every published number |
| `make status` | What is registered and what is deployed |
| `make drift` | Both streams through the monitor, control first |
| `make retrain` | The retraining pipeline against captured frames |
| `make rollback` | Return production to the previous deployed version |
| `make engine-plan` | What TensorRT would build (works without a Jetson) |
| `make mlflow` | The tracking UI |

Every command is a thin wrapper over a script in `scripts/`, and
`scripts/run_loop.py` prints each command before running it — so it doubles as
an executable version of the runbook.

## Data and licensing

- Code: **Apache-2.0**.
- Data: **generated by this repository**, Apache-2.0, commercially usable. Not
  committed; `scripts/make_dataset.py` rebuilds it byte for byte and `--verify`
  proves it.
- MVTec AD (CC BY-NC-SA 4.0) and DAGM 2007 (research use only) are **supported
  and not shipped**. `scripts/make_dataset.py --source mvtec` converts a local
  copy you have licensed yourself, and refuses to run without an explicit
  acknowledgement flag.
- Ultralytics is deliberately absent: AGPL-3.0 in a deployed edge image is a
  licence conversation nobody wants at integration time.

[docs/licences.md](docs/licences.md)

## Limitations

No Jetson, no camera, synthetic data, oracle labels for retraining, one device
rather than a fleet, and a 61k-parameter model with a real accuracy ceiling
chosen so the retraining loop runs on a CI runner with no GPU. Each of those is
written out properly, with what it does and does not invalidate, in
[docs/limitations.md](docs/limitations.md).

## Licence

Apache-2.0 — see [LICENSE](LICENSE).
