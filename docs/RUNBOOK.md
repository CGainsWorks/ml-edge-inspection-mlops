# Runbook

For whoever is on call. Written as procedures, in the order they get needed.

Every command here is run by `scripts/run_loop.py`, which prints each one before
executing it — so if the loop passes in CI, the commands on this page work.

---

## Contents

1. [A drift alert has fired](#1-a-drift-alert-has-fired)
2. [Retraining has been triggered](#2-retraining-has-been-triggered)
3. [The gate blocked a model](#3-the-gate-blocked-a-model)
4. [A deployment has gone wrong](#4-a-deployment-has-gone-wrong)
5. [Rolling back](#5-rolling-back)
6. [The device is flagging everything](#6-the-device-is-flagging-everything)
7. [The device is flagging nothing](#7-the-device-is-flagging-nothing)
8. [Deploying a new model deliberately](#8-deploying-a-new-model-deliberately)
9. [After changing the feature set](#9-after-changing-the-feature-set)
10. [Routine checks](#10-routine-checks)

---

## 1. A drift alert has fired

**Symptom.** A line in `artifacts/telemetry/alerts.jsonl` with severity `alarm`.

**First: read the alert, not the dashboard.** The alert text says which signals
moved, in which direction, and by how much in reference standard deviations.
That is a diagnosis, and the diagnosis determines whether retraining is even the
right response.

```bash
python scripts/detect_drift.py --telemetry artifacts/telemetry/drifted.jsonl
```

**Read the pattern:**

| Signals that moved | Likely cause | Right response |
|---|---|---|
| `brightness` down, `sharpness` down, `noise_level` up | Lamp ageing, dirty lens; the camera is compensating with gain | **Clean the lens and check the illuminator first.** Retraining around a fixable optical fault teaches the model to live with it |
| `texture_scale` and `edge_density` moved, `brightness` flat | Material or process change upstream | Retrain. The parts really are different |
| `score` and `decision` moved, image statistics flat | The model's behaviour changed without its input changing | **Suspect the deployment, not the data.** Check `scripts/deploy.py history` for a recent promotion |
| Everything moved at once, abruptly | Camera reseated, exposure reconfigured, wrong lens fitted | Fix the camera. Then capture a new reference |

**Then decide.** A drift alert is a request for a decision, not an instruction.
The two wrong answers are retraining around a mechanical fault, and ignoring a
real distribution shift because the line is still running.

## 2. Retraining has been triggered

```bash
python pipelines/retrain.py \
  --telemetry artifacts/telemetry/drifted.jsonl \
  --capture   artifacts/capture/drifted \
  --reason    "lamp ageing, alert 2026-08-22T14:03Z"
```

Ten stages. Watch three of them:

- **`label`** — `dropped_unlabelled` should be zero or near it. A large number
  means the device kept frames without masks, and those are useless for
  retraining.
- **`dataset`** — check `train_captured` against `train_replayed`. If the
  captured share is tiny, the retrained model will barely move; run the device
  longer under the drifted conditions before retraining.
- **`gate`** — the decision, with every check and its numbers.

The run directory is `artifacts/retrain/<timestamp>/`, and `manifest.json` in it
records every stage, its timing and its outputs — including for a failed run.

**On a real line this stage blocks for hours.** The labelling step here uses the
generator's ground truth. In production those frames go to a human, and the
runbook entry is "raise the labelling request, then come back". Quote a cycle
time that includes it.

## 3. The gate blocked a model

**This is a normal outcome, not an incident.** The blocked version stays in the
registry with its reason so the attempt is on the record.

```bash
python scripts/deploy.py status
python -c "import json,sys; sys.path.insert(0,'src'); from edgeinspect.registry import Registry; \
  print(json.dumps(Registry().get(2).notes['gate'], indent=2))"
```

| Check that failed | What it means | What to do |
|---|---|---|
| `improvement` on `drift.image.auroc` | The retrained model is not materially better under the new conditions | Not enough drifted data, or the drift is a genuine loss of information rather than a domain shift. Check whether defect contrast actually survived |
| `regression` on `reference.image.auroc` | It learnt the new conditions by forgetting the old ones | Raise `--replay-fraction`, or retrain from the incumbent rather than further from it |
| `floor` on `reference.image.auroc` | Both models are bad | Do not deploy either. Something upstream is wrong |
| `ceiling` on `reference.image.false_positive_rate` | It would stop the line too often | Check the operating point; the threshold is recalibrated per variant by `scripts/evaluate.py` |

**Do not lower the gate to get a model out.** If the thresholds are genuinely
wrong they are wrong for every future run, so change them deliberately in
`configs/deploy.yaml`, in a commit, with the reason.

## 4. A deployment has gone wrong

The canary runs automatically after every promotion and rolls back by itself on
failure. If you are reading this, either the canary caught it (and the system has
already recovered) or something got past it.

**Check what is actually running on the device**, which is not necessarily what
the registry says:

```bash
cat artifacts/deploy/current.json
python scripts/deploy.py history | tail -20
```

**If the agent refused to activate**, the previous model is still serving and
the line is still running. The refusal reason is in the agent's outcome record:

- `artefact hash mismatch` — the file on the device is not the file that was
  evaluated. Re-copy it. Do not "just try loading it".
- `smoke check failed ... separation` — the model loads but does not separate
  the probe pair. It is the wrong model, an inverted output, or a broken
  quantisation. Do not deploy it anywhere else either.

## 5. Rolling back

```bash
python scripts/deploy.py rollback --reason "flagged rate spike after v3"
```

Rollback targets the previous **deployed** version, read from the event log —
not version minus one, which is a different version as soon as anything has been
skipped or blocked.

**Then prove the device is actually serving it.** A manifest is an instruction,
not a fact:

```bash
python scripts/simulate_edge.py --stream clean --frames 60 \
  --out results/rollback_check.json
python -c "import json; d=json.load(open('results/rollback_check.json')); \
  print('device is serving version', d['model_version'])"
```

## 6. The device is flagging everything

**Do not start by retraining.** Work down this list:

1. **Was a new model deployed recently?** `scripts/deploy.py history`. If yes,
   roll back first and diagnose afterwards. The line is more important than the
   root cause.
2. **Did the drift monitor fire?** If the image statistics moved, the input
   changed and section 1 applies.
3. **Did the image statistics *not* move?** Then the model changed without its
   input changing, which means the deployment. Check that the artefact hash in
   `current.json` matches the version the registry has in production.
4. **Is the threshold the one that was evaluated?** Read the sidecar:
   `decision.threshold_source` should name `scripts/evaluate.py` and a split. If
   it says `inherited-from-fp32-pending-recalibration`, the graph was deployed
   without being evaluated, and its threshold belongs to a different model.

## 7. The device is flagging nothing

More dangerous than section 6, because the line keeps running and nobody
notices.

The canary checks for this — `min_flagged_rate` in `configs/deploy.yaml` — so a
model that fails open at deployment is caught. What the canary cannot catch is a
model that fails open later.

1. Check the flagged rate against the reference: `rate_shift/decision` in the
   drift report. A rate that has fallen to zero is a drift alarm in the "below"
   direction.
2. Run the probe pair by hand:

   ```bash
   python -c "import sys; sys.path.insert(0,'src'); \
     from edgeinspect.edge.agent import EdgeAgent, smoke_check; \
     a=EdgeAgent(); a.activate(); print(smoke_check(a.backend).to_dict()); a.close()"
   ```

   A small `separation` means the model has stopped discriminating.
3. If the separation is fine, the threshold is wrong for the current conditions
   — which is drift, and section 1 applies.

## 8. Deploying a new model deliberately

```bash
make train export quantise evaluate      # or the individual scripts
python scripts/deploy.py register \
  --models models/surface_defect_fp32.onnx models/surface_defect_int8.onnx \
  --checkpoint models/surface_defect.pt \
  --accuracy results/accuracy.json \
  --reason "quarterly refresh"
python scripts/deploy.py promote --version <n> --reason "quarterly refresh"
```

`promote` does three things and any of them can stop it: it writes the manifest,
captures a fresh drift reference from the *new* model's own telemetry, and runs
the canary — rolling back on its own if the canary fails.

**The fresh reference is not optional.** Score distributions are not comparable
across versions. Deploying a new model against the old model's reference makes
the monitor fire on the deployment itself, every time, and that is how
monitoring gets switched off.

## 9. After changing the feature set

Adding or renaming anything in `edgeinspect.monitoring.features.FEATURE_NAMES`
invalidates every saved reference, because a reference stores the signals it was
built with.

The monitor fails loudly rather than skipping the missing signal — a rule that
is silently not evaluated leaves a monitor that looks armed. Recapture:

```bash
python scripts/deploy.py promote --version <current> --reason "feature set changed" \
  --no-canary
```

Then re-run the control stream and confirm it still does not trigger, before
trusting anything the monitor says.

## 10. Routine checks

**Weekly.**

```bash
python scripts/deploy.py status          # one production version, no surprises
python scripts/detect_drift.py --telemetry artifacts/telemetry/clean.jsonl \
  --expect-no-trigger                    # exits non-zero on a false alarm
```

**After any change to the monitor, the model or the features.**

```bash
make check                               # lint, tests, docs
make demo-quick                          # the whole loop, about two minutes
```

`make demo-quick` asserts, in one command: the monitor does not fire on the
control stream, does fire on the drifted one, retraining runs, the gate decides,
and rollback puts the previous version back on the device. If that passes, the
loop works.

**Before quoting any number from the documentation.**

```bash
make docs-check
```

Regenerates every embedded table from `results/published/` and fails if a table
or a prose number has drifted from its source.
