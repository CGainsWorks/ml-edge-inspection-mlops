# Architecture

The model is the least interesting thing in this repository. What is
interesting is the loop it sits inside, and specifically the five places where
that loop is allowed to say no.

---

## The loop

```
                        configs/data.yaml
                               │
                   ┌───────────▼────────────┐
                   │  synthetic generator   │  deterministic; imaging
                   │  edgeinspect.data      │  conditions are parameters
                   └───────────┬────────────┘
                               │  train / val / test / test_drift
                   ┌───────────▼────────────┐
                   │  train                 │──▶ MLflow (SQLite, serverless)
                   │  DefectNet, 61k params │
                   └───────────┬────────────┘
                               │  checkpoint + threshold + dataset hashes
                   ┌───────────▼────────────┐
                   │  export  ──▶ verify    │  ◀── FAILS the build if the
                   │  torch → ONNX          │      graph does not match torch
                   └───────────┬────────────┘
                               │
                   ┌───────────▼────────────┐
                   │  quantise              │  one shared calibration set,
                   │  PTQ → INT8 QDQ        │  used by ONNX Runtime AND by
                   └───────────┬────────────┘  the TensorRT build
                               │
                   ┌───────────▼────────────┐
                   │  evaluate              │  per-variant threshold, chosen
                   │  both holdouts         │  on validation only
                   └───────────┬────────────┘
                               │
   ╔═══════════════════════════▼═══════════════════════════════════════════╗
   ║                        REGISTRY                                       ║
   ║  versions are immutable · exactly one production · artefacts hashed   ║
   ╚═══════════════════════════╤═══════════════════════════════════════════╝
                               │  deployment manifest (version, sha256, threshold)
                   ┌───────────▼────────────┐
                   │  EDGE AGENT            │  ◀── REFUSES a hash mismatch
                   │  verify → smoke → swap │  ◀── REFUSES a model that loads
                   └───────────┬────────────┘      but does not separate
                               │
                   ┌───────────▼────────────┐
                   │  edge runtime          │──▶ telemetry (JSONL, per frame)
                   │  infer · decide · box  │──▶ capture (sampled frames)
                   └───────────┬────────────┘
                               │
                   ┌───────────▼────────────┐      reference window, frozen
                   │  DRIFT MONITOR         │◀─────  at deployment, from THIS
                   │  windowed + sequential │        model's own telemetry
                   └───────────┬────────────┘
                               │  trigger
                   ┌───────────▼────────────┐
                   │  RETRAIN PIPELINE      │
                   │  collect → label →     │
                   │  dataset(+replay) →    │
                   │  train → export →      │
                   │  quantise → evaluate   │
                   └───────────┬────────────┘
                               │
                   ┌───────────▼────────────┐
                   │  GATE                  │  ◀── REFUSES a model that
                   │  improve · not regress │      improved on the drift and
                   │  absolute floors       │      regressed on the original
                   └──────┬─────────┬───────┘
                   pass   │         │  fail
                          ▼         ▼
                   ┌──────────┐  ┌──────────────────┐
                   │ deploy   │  │ registered,      │
                   │ + canary │  │ stage=blocked,   │
                   └────┬─────┘  │ NOT deployed     │
                        │        └──────────────────┘
                 ┌──────▼──────┐
                 │ CANARY      │  ◀── REFUSES a deployment whose live
                 │ live check  │      behaviour is wrong, and rolls back
                 └──────┬──────┘      by itself
                        │ fail
                        ▼
                 ┌─────────────┐
                 │  ROLLBACK   │  previous *deployed* version, from the
                 └─────────────┘  event log, not version minus one
```

## The five refusals

Every one of these is a place where the automated path stops, and each exists
because the alternative failure is quiet.

**1. Export verification.** `scripts/export_onnx.py` runs the exported graph and
the torch model on the same random input and fails if they differ by more than
1e-4. Batch-norm folding, constant folding and opset lowering all change
numerics silently, and the first place a changed graph shows up is a
mis-calibrated INT8 model that nobody thinks to blame on the export.

**2. Artefact hashing.** The registry hashes every artefact at registration; the
agent re-hashes before loading. A partially transferred ONNX file loads, runs,
and produces plausible nonsense.

**3. The smoke check.** Loading proves the file is a graph. It does not prove
it is the model. The agent runs a fixed probe pair — one clean frame, one with
an unmistakable defect — and requires finite, correctly shaped output with the
defective probe scoring higher. That catches an inverted output, a saturated
quantisation and a graph exported from an untrained network, all of which load
perfectly. The new backend is loaded *alongside* the running one and only
swapped in after it passes.

**4. The gate.** Asymmetric on purpose: must improve materially on the drifted
holdout, must not regress more than a little on the original one, must clear
absolute floors on both. A gate with only the improvement condition promotes a
model that has forgotten the original domain — and because the alert came from
the drifted domain, nothing else in the pipeline would notice.

**5. The canary.** A short live run after promotion, with conditions that bite:
a flagged rate too high means runaway false alarms, a flagged rate near zero
means the model has failed open, and a p95 latency over budget means it will not
keep up with the line. Failing any of them rolls the deployment back
automatically and blocks the version. A deployment that has to be undone by hand
at 03:00 is a deployment that stays broken until morning.

The canary runs under the same conditions the drift reference was captured
under, and after a drift-triggered redeployment that is the *new* normal rather
than the old one — the post-drift stream, not the clean one. Checking a
retrained model against conditions the line is no longer in tests the wrong
thing, and capturing the reference there would leave the monitor firing forever
on a model that has already adapted.

## Module map

| Module | Responsibility | Needs torch? |
|---|---|---|
| `edgeinspect.data` | Deterministic generator, on-disk layout, augmentation, MVTec adapter | no |
| `edgeinspect.models.net` | `DefectNet`, the loss targets | yes |
| `edgeinspect.models.preprocess` | The input contract, in one place | no |
| `edgeinspect.models.postprocess` | Heatmap → score and boxes | no |
| `edgeinspect.metrics` | AUROC, AP, confusion, box matching | no |
| `edgeinspect.train` | Training loop, MLflow wrapper | yes |
| `edgeinspect.quantise` | Calibration selection, PTQ, QAT | PTQ no, QAT yes |
| `edgeinspect.export` | ONNX export and verification, TensorRT build | export yes |
| `edgeinspect.artefacts` | Checksums and sidecars | **no, deliberately** |
| `edgeinspect.edge` | Backends, runtime, telemetry, capture, deploy agent | **no, enforced** |
| `edgeinspect.monitoring` | Features, detectors, drift monitor, alerts | no |
| `edgeinspect.registry` | Versions, stages, deployment record | no |
| `edgeinspect.pipeline` | The DAG runner | no |

The "needs torch" column is enforced, not documented:
`tests/test_telemetry_and_pipeline.py` imports the edge runtime in a subprocess
and fails if `torch` appears in `sys.modules`, and `docker/Dockerfile.runtime`
runs the same check at build time. That test found a real leak — the sidecar
helpers lived in a module that imports torch at top level, which would have made
the runtime image fail on first start with an `ImportError` three layers away
from the cause. `edgeinspect.artefacts` exists because of it.

## Design decisions and what they cost

**One head, two losses.** The model emits a single logit map. The image score is
the mean of its hottest cells, defined once in torch for the loss and once in
NumPy for the metric — and a test asserts the two agree, because a classifier
and a localiser trained separately can disagree about the same frame and nothing
will tell you.

**The reference window is frozen at deployment, per version.** Comparing the
current window against a rolling recent window is the design that cannot see a
slow ramp: nothing is ever different from ten minutes ago. And a *new model
version* gets a *new reference*, because score distributions are not comparable
across versions — a monitor that keeps comparing v2's scores to v1's reference
fires on the deployment itself, every time, until somebody turns it off.

**Windowed and sequential detectors together.** Windowed tests answer "is it
different now"; Page-Hinkley answers "when did it start". Both run because
detection latency is a headline number and a windowed test can only locate an
onset to within one stride. Page-Hinkley watches only the image statistics: the
model's score is a mixture distribution, and a run of defective parts arriving
together looks exactly like a step change to a running-mean detector. That was
found by a false alarm on a control stream.

**The retraining set is captured frames plus a replay buffer, mixed to a target
share.** Training only on what the device kept during the drift produces a model
that is excellent under the new conditions and has forgotten the old ones.

The mix has to be balanced in both directions, which was learned by getting it
wrong twice in the same way. On the full dataset the captured frames were under
ten percent of each epoch and the retrained candidate barely moved; on the tiny
dataset the same production stream gave them sixty percent and the candidate
forgot the reference domain. Whichever half is in the minority is now repeated
until the share is reached — repeated rather than the other half being cut,
because discarding production frames throws away the only evidence of the new
conditions and cutting replay is precisely how a model forgets.

Both candidates were blocked by the gate, which is the clearest argument for
having one.

**A registry that is one JSON file.** Readable, diffable, restorable from a
backup. For one model on one device that is the right size of tool, and the API
is the shape a real registry has, so the storage is the only thing that would
change.

**Scripts are the interface.** `edgeinspect train` and `python scripts/train.py`
are the same code; the Makefile, the README, the runbook and `run_loop.py` all
use the script form, because a command someone can read and modify beats a
subcommand whose implementation they then have to find. `run_loop.py` prints
every command before running it, which makes it an executable version of the
runbook.
