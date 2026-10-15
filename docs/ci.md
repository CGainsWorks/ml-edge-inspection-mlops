# CI and edge delivery

Three workflows. One runs on every push, one runs weekly, one runs when somebody
asks for a release.

| Workflow | Trigger | Cost | What it proves |
|---|---|---|---|
| [`ci.yml`](../.github/workflows/ci.yml) | push, PR | ~11 min | Lint, the whole test suite, a real training run, the export verification, quantisation, and **the entire operational loop** |
| [`drift-watch.yml`](../.github/workflows/drift-watch.yml) | Mondays 04:17, manual | ~8 min/week | The monitor stays quiet across several independent control streams, and still fires on a drifted one |
| [`edge-delivery.yml`](../.github/workflows/edge-delivery.yml) | manual only | 0 until asked | Builds, gates and packages the bundle that would be copied to a device |

---

## What `ci.yml` actually asserts

The interesting step is `The operational loop`, which runs
`scripts/run_loop.py --quick --fresh`. A non-zero exit means one of these
failed:

1. The baseline registers, promotes, captures a drift reference and passes its
   canary.
2. **The monitor does not fire on the control stream.** Same imaging conditions
   as the reference, different random draws. A trigger is a false alarm.
3. **The monitor does fire on the drifted stream**, and the detection latency is
   recorded.
4. The retraining pipeline runs all ten of its stages against the frames the
   device kept.
5. The gate reaches a decision and records every check.
6. The monitor goes quiet under the post-drift steady state, because the
   reference was recaptured under the conditions the line is now in.
7. Rollback puts the previous version back, and the device confirms it is
   serving it.

Assertion 2 is the one that is easy to leave out. A drift detector that has only
ever been shown drifting data has never been shown *not* to fire, and a monitor
that fires on everything is indistinguishable from a working one until the day
it costs somebody a retraining run. It has already paid for itself twice: it
caught a PSI small-sample bias and a Page-Hinkley false alarm, both documented
in [drift.md](drift.md).

## The gate thresholds CI runs against

CI uses `configs/deploy_tiny.yaml`, not `configs/deploy.yaml`, and that deserves
stating plainly rather than being left to be discovered.

The gate's **shape** is identical: improve materially on the distribution that
triggered the alert, do not regress on the one already handled, clear absolute
floors on both. Only the absolute numbers differ, because an absolute accuracy
floor is a property of the dataset and CI trains on a tenth of the data — 160
training frames against 1,600, 100 test frames against 600. The baseline itself
reaches 0.856 AUROC on the tiny test split against 0.882 on the full one, so the
full dataset's 0.85 floor would block every candidate CI could ever produce,
including ones working perfectly. A gate that can only say no is as useless as
one that can only say yes.

Two of the tiny thresholds are set by measurement resolution rather than taste:

- `max_drop` is 0.04 rather than 0.02, because the standard error on AUROC over
  100 test frames is about 0.04. A 0.02 allowance sits below the noise floor and
  would reject half of all *identical* models.
- the false-positive ceiling is 0.12 rather than 0.05, because with 50 test
  negatives the measured rate moves in steps of 0.02.

The published results were gated on the full-dataset thresholds. Finding this
out was itself useful: the first attempt ran CI against the full gate, and it
blocked a candidate that had recovered the drift perfectly and lost 0.026 AUROC
on a holdout too small to resolve 0.026.

## Why this is possible at all on a free runner

Because the model is 61k parameters and trains on CPU in about a minute. That
was a deliberate trade, taken at the cost of real accuracy: this project's claim
is about the loop, and a loop that can only be exercised on a GPU is a loop that
does not get exercised. The ceiling it buys is written up in
[limitations.md](limitations.md).

## How the workflows are kept cheap

The account running this has a monthly minute budget, so the structure is part
of the design rather than an afterthought.

**One job does the whole pipeline.** Splitting lint, tests, the loop and the
docs check across four jobs re-does checkout, Python setup and a torch install
four times — roughly three minutes of pure overhead per push for no extra
coverage. They are one job on one install.

**The Docker build is not on every push.** Building an image with buildx is the
most expensive thing in the file, and the image can only break when the
Dockerfile, the requirements or the runtime source change. The job checks the
diff and skips itself otherwise — a skip rather than a path filter on the
workflow, so a docs-only push shows a green skip instead of leaving a required
check pending forever. Its layer cache is `mode=min`: caching every intermediate
layer of a pip install costs more to upload and restore than it saves, and the
10 GB cache would evict the layers worth keeping.

**The second Python version runs lint and the dataset-free tests only.** The
question it answers is "does the package still work on the oldest Python it
claims to support", and a second training run does not help answer it.

**`timeout-minutes` on every job.** This is the single most important line in
the file. A hung step otherwise runs to GitHub's six-hour default, and six hours
is a fifth of a free month's budget spent on one mistake.

`tests/test_workflows.py` enforces that and seven other things about these
files, because workflow YAML is otherwise covered by nothing and fails late and
expensively: every job has a timeout under 45 minutes, permissions are declared
and read-only, every script a workflow runs exists, torch comes from the CPU
index, pip caching is keyed on the requirement files, uploaded artifacts have a
short retention, embedded Python compiles, and no token is hardcoded.

**Artifacts only on failure, with short retention.** A green run needs no
evidence, and artifact storage is a quota as well.

**`concurrency` with `cancel-in-progress`.** A new push to the same branch
cancels the previous run rather than paying for both. The scheduled workflow
deliberately does *not* cancel in progress, because two of its runs will never
overlap and cancelling a weekly job loses a week of evidence.

**CPU torch from the PyTorch index.** The default PyPI wheel pulls the whole
CUDA runtime, about 2.5 GB that no runner here can use.

## What the weekly run is for

`ci.yml` asserts the monitor does not fire on *one* control stream. That is
necessary and it is not a false-alarm *rate*: zero triggers across three windows
is consistent with a per-window rate of several percent, which
[limitations.md](limitations.md) says out loud.

`drift-watch.yml` runs the control stream under six different seeds and requires
zero triggers across all of them, writing the table to the run summary so the
evidence accumulates rather than being re-asserted. It runs the drifted stream
in the same job, because thresholds loose enough never to false-alarm are also
loose enough never to fire, and that failure is silent.

It is weekly because thresholds do not change between pushes. After changing one
deliberately, run it by hand.

## What `edge-delivery.yml` can and cannot do

**There is no device.** Nothing in any workflow pushes a model to a Jetson, and
[jetson.md](jetson.md) is the full accounting. What this workflow does is
everything up to the transfer: build the artefacts, put them through the
registry, prove the deployed artefact passes the *edge agent's own* smoke check,
build the runtime container, and package the bundle.

The bundle is the useful part, and its contents are not arbitrary:

| In the bundle | Why it has to be |
|---|---|
| The INT8 graph | What the device serves |
| Its sidecar | The decision threshold and the input normalisation. A graph without them runs and is quietly wrong |
| The FP32 graph | What the device's TensorRT build consumes |
| `calibration.npy` and its manifest | So the device's INT8 engine calibrates on the *same* frames the CPU INT8 graph did, making the two comparable |
| The deployment manifest | Version, artefact hash and threshold; the agent refuses a hash mismatch |
| The drift reference window | The monitor is useless without it, and it is per version |
| `Dockerfile.jetson`, `entrypoint-jetson.sh` | So the bundle is self-describing |
| `SHA256SUMS` | So the device can verify the transfer itself |

No TensorRT engine is included, and that is correct rather than a gap: a plan is
tied to the exact GPU, driver and TensorRT version that built it, so it is built
on the device at first start by the entrypoint.

## What is deliberately absent

**A training workflow on a GPU runner.** GitHub's hosted runners have no NVIDIA
GPU. A `train.yml` here would either need a self-hosted runner that does not
exist or would be a workflow that can never run, and a workflow that can never
run is worse than no workflow — it looks like coverage. The training that *can*
run on CPU runs in `ci.yml`, on every push, for real.

**A TensorRT job.** Same reason. `ci.yml` runs the planner in `--dry-run`, which
checks that the plan is coherent and the inputs exist, and that is the most that
can honestly be checked without hardware.

**Publishing to a container registry.** The runtime image is built and checked
and not pushed. Pushing it would need write permissions this workflow
deliberately does not have, and an image nobody is pulling is storage quota
spent on a gesture.

## If Actions is not running at all

A brand-new **private** repository gets no runner assigned and the run fails in
seconds with no steps executed — which looks exactly like a broken workflow and
is not one. Check the repository's visibility and its Actions settings before
debugging the YAML. Actions minutes are unlimited on public repositories.
