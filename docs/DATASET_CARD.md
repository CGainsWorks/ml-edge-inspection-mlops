# Dataset card: `edgeinspect-synth`

## Summary

A deterministic synthetic surface-inspection dataset: greyscale 256x256 frames
of a machined or rolled metal surface, with four kinds of defect, generated
under parameterised imaging conditions.

Not shipped. `scripts/make_dataset.py` rebuilds it byte for byte from the seeds
in `configs/data.yaml`, and `--verify` regenerates a sample and compares the PNG
bytes so a result can be traced to its data without storing the data.

| | |
|---|---|
| Generator | [`src/edgeinspect/data/synth.py`](../src/edgeinspect/data/synth.py) |
| Licence | Apache-2.0 |
| Commercial use | yes |
| Frame size | 256 x 256, greyscale, 8-bit PNG |
| Seed | 20260822 |
| Determinism | verified on every CI run |

## Why synthetic

Two reasons, and only the second is convenience.

**Licence.** MVTec AD is CC BY-NC-SA 4.0 and DAGM 2007 has no redistribution
grant. Neither can ship inside an Apache-2.0 repository, and MVTec's share-alike
clause would infect any derived dataset. See [`licences.md`](licences.md).

**Drift has to be causal to be testable.** This project's claim is that a drift
detector fires on a real distribution shift and that retraining recovers the
loss. To test that, the shift must be *applied*, with a known cause and a known
magnitude. A fixed public dataset cannot provide one, and relabelling a
different MVTec category as "drift" would be worse than synthetic honesty.

The cost is stated in [`limitations.md`](limitations.md): absolute accuracy on
synthetic textures says nothing about a production line. The *relative* results
— the INT8 delta, the drop under drift, the recovery after retraining, the
monitor's false-alarm behaviour — are what this repository publishes.

## How a frame is built

In this order, which mirrors the physics:

1. **Surface.** Band-limited noise at a stated correlation length, plus a
   sinusoid at a random orientation and phase standing in for machining marks,
   plus a slow undulation for sheet flatness.
2. **Defects.** With probability `defect_rate`, one or two of:
   - `scratch` — thin, long, low contrast, with a bright burr on one side
   - `pit` — small, round, high contrast
   - `dent` — broad, smooth, very low contrast
   - `inclusion` — irregular outline with a texture of its own
3. **Optics.** Gaussian blur, applied to the surface *and* the defect together.
4. **Illumination.** Radial vignette, then exposure gain and black level.
5. **Sensor.** Additive Gaussian read noise, then 8-bit quantisation.

**Step 3 is the one that matters.** Blurring the clean surface and pasting a
sharp defect on top produces a dataset where defects are separable by
high-frequency energy alone, and a model trained on it collapses the moment a
real lens is slightly out of focus. `tests/test_synth.py` asserts that blur
attenuates defect contrast.

Defects cover under two percent of a defective frame and typically around 0.3
percent. This is a hard task on purpose: a defect that is trivially separable
makes every downstream number meaningless.

## Determinism

Every frame is a pure function of `(dataset_seed, split, index, domain)`, hashed
with blake2b rather than added. Hashing rather than `dataset_seed + index`
because the latter makes `train[5]` and `val[4]` correlated for adjacent seeds,
and correlated train and validation images are the quietest way to invent a good
number. Nothing depends on generation order, so a single frame can be
regenerated without regenerating its split, and two machines produce identical
PNGs.

## Splits

The `full` preset, which produced every published number:

| Split | Frames | Domain | Purpose |
|---|---:|---|---|
| `train` | 1600 | `reference` | Training |
| `val` | 400 | `reference` | Model selection and the operating point |
| `test` | 600 | `reference` | The reference holdout |
| `test_drift` | 600 | `lamp_ageing` | The drifted holdout |

Evaluation splits are balanced exactly 50/50 by alternating forced labels, so
the class ratio does not wobble with the seed. Production *streams* are not
balanced — they run at an 8 percent defect rate, because a real line is mostly
good parts.

The `tiny` preset (160/60/100/100) is what CI runs. It exercises every stage
with the same code.

## Domains

A domain is a set of imaging conditions. Each drifted one is named after what
goes wrong on a real cell.

| Domain | Change from `reference` | Stands for |
|---|---|---|
| `reference` | — | Nominal: correct exposure, focused lens, clean optics |
| `reference_holdout` | none; different draws | **The control.** The monitor must not fire on it |
| `lamp_ageing` | gain down, black level up, blur up, noise up, defect contrast down | An 18-month-old LED illuminator and an uncleaned lens. The camera compensates with gain, which raises read noise |
| `lamp_ageing_mild` | the same, less of it | The endpoint of a gradual ramp |
| `material_batch` | coarser, more contrasty grain; wider machining marks | The supplier changed the rolling process. Optics untouched |

`reference_holdout` is the one that makes the experiment falsifiable. Without a
control stream the detector's alarm threshold cannot be shown to be anything
other than "fires on everything".

`material_batch` exists because a monitor that only watches brightness would
miss it entirely; `tests/test_synth.py` asserts that it moves the texture
statistic without moving brightness.

## Production streams

| Stream | Frames | Conditions | Defect rate |
|---|---:|---|---:|
| `clean` | 900 | `reference_holdout`, fixed | 8% |
| `drifted` | 1200 | ramps from `reference_holdout` to `lamp_ageing` between frames 200 and 600, then holds | 8% |
| `material` | 800 | ramps to `material_batch` between frames 100 and 300 | 8% |

The ramp matters. A step change is caught by anything; a gradual onset is what
actually happens when a lamp ages, and it is the case a threshold detector finds
hardest.

Frames in the clean and drifted streams are *different parts*, not the same
parts relit — the frame seed depends on the stream name, not only on the domain.
Relighting the same parts would make the drift trivially separable and the
experiment worthless.

## Labelling

Ground truth comes from the generator: a pixel mask, and boxes derived from the
connected components of that mask. There is no annotation noise, no ambiguity
and no inter-annotator disagreement, which is a difference from any real dataset
and is why the retraining pipeline's labelling step is documented as an oracle
standing in for a human.

## Using MVTec AD or DAGM instead

```bash
python scripts/make_dataset.py --source mvtec \
    --mvtec-root /data/mvtec_anomaly_detection --category metal_nut \
    --i-accept-dataset-licence
```

The adapter converts a category into this layout, after which every downstream
stage runs unchanged. Two honest caveats:

- MVTec's train split is defect-free by design, which suits an anomaly detector
  and not the supervised head here, so the annotated test split is re-split
  deterministically and stratified by defect type. The mapping is recorded in
  `dataset.json`.
- MVTec has no drifted split. `--drift-transform` applies the *difference*
  between two domains to real frames — extra blur, an exposure ratio, extra read
  noise — which is a weaker stand-in than the synthetic path, and is documented
  as such.

The flag is required. A build that silently ingests CC BY-NC-SA data into an
Apache-2.0 project is a licence incident waiting to be found by somebody else.
