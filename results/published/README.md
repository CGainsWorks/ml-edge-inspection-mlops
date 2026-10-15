# Published results

The numbers the documentation is allowed to quote.

`results/` is where a run writes its output and is not tracked. This directory
is the tracked exception: `make publish` copies a run's results here and
regenerates every table in the documentation from them, and `make docs-check`
fails if a table or a prose decimal has drifted from what is here.

Publishing is therefore a deliberate act. Until it happens the documented
numbers stay pinned to the last published run, and a local experiment cannot
quietly rewrite the README.

| File | Produced by | Feeds |
|---|---|---|
| `accuracy.json` | `make evaluate` | the accuracy table in the README, MODEL_CARD and results.md |
| `latency.json` | `make bench` | the latency table in results.md |
| `loop.json` | `make demo` | the drift, recovery and gate tables |
| `quantisation_ablation.json` | `make ablate` | the quantisation table |
| `retrain_seed_sweep.json` | `make seed-sweep` | the reproducibility table in ci.md |

Every file records the host, the configuration and the inputs it came from, so a
number here can be traced back to the run that produced it without consulting
anything else.
