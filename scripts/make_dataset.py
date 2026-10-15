#!/usr/bin/env python
"""Generate the dataset.

    python scripts/make_dataset.py --preset full
    python scripts/make_dataset.py --preset tiny --root data/tiny
    python scripts/make_dataset.py --source mvtec --mvtec-root /data/mvtec_ad \\
        --category metal_nut --i-accept-dataset-licence

Deterministic: the same preset and seed produce byte-identical PNGs on any
machine.  ``--verify`` regenerates a sample of frames and compares them to what
is on disk, which is how CI checks that determinism has not been broken by a
NumPy or SciPy upgrade.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from edgeinspect import paths
from edgeinspect.config import load_domains, load_yaml
from edgeinspect.data import dataset as ds
from edgeinspect.data import mvtec, synth


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--preset", default="full")
    parser.add_argument("--root", default="data/processed")
    parser.add_argument("--source", choices=("synthetic", "mvtec"), default="synthetic")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--drift-domain", default="lamp_ageing",
                        help="domain for the extra drifted evaluation split")
    parser.add_argument("--no-drift-split", action="store_true",
                        help="skip the drifted holdout (it is what proves recovery)")
    parser.add_argument("--mvtec-root", default=None)
    parser.add_argument("--category", default="metal_nut")
    parser.add_argument("--i-accept-dataset-licence", action="store_true",
                        dest="accept_licence")
    parser.add_argument("--verify", type=int, default=0,
                        help="regenerate N frames per split and compare to disk")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def verify_determinism(root: Path, dataset_seed: int, domains, samples: int) -> dict:
    """Regenerate a sample of frames and compare the PNG bytes."""
    checked = 0
    mismatches = []
    for split in ds.SPLITS:
        try:
            records = ds.load_index(root, split)
        except FileNotFoundError:
            continue
        domain_name = records[0].domain
        domain = domains.get(domain_name)
        if domain is None:
            continue
        step = max(len(records) // max(samples, 1), 1)
        for index in range(0, len(records), step):
            if checked >= samples * len(ds.SPLITS):
                break
            record = records[index]
            frame_index = int(record.id.rsplit("_", 1)[1])
            seed = synth.frame_seed(dataset_seed, split, frame_index, domain_name)
            force = frame_index % 2
            frame = synth.generate_frame(seed, domain, force_label=force)
            on_disk = ds.read_png(paths.resolve(record.image))
            if not np.array_equal(frame.image, on_disk):
                mismatches.append(record.id)
            checked += 1
    return {"checked": checked, "mismatches": mismatches,
            "deterministic": not mismatches}


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_yaml(args.config)
    domains = load_domains(config)
    root = paths.resolve(args.root)
    started = time.perf_counter()

    if args.source == "mvtec":
        mvtec.check_licence_acknowledged("mvtec", args.accept_licence)
        if not args.mvtec_root:
            print("--source mvtec needs --mvtec-root", file=sys.stderr)
            return 2
        drift = None if args.no_drift_split else domains.get(args.drift_domain)
        meta = mvtec.convert_mvtec(
            Path(args.mvtec_root), args.category, root,
            size=int(config.get("image_size", 256)),
            drift_domain=drift, reference_domain=domains["reference"],
        )
        meta["seed"] = int(args.seed or config.get("seed", 0))
        meta["preset"] = "mvtec/" + args.category
        ds.write_meta(root, meta)
        print(json.dumps(meta, indent=2))
        return 0

    presets = config.get("presets", {})
    if args.preset not in presets:
        print("unknown preset {0!r}.  Known: {1}".format(args.preset, sorted(presets)),
              file=sys.stderr)
        return 2
    preset = presets[args.preset]
    dataset_seed = int(args.seed if args.seed is not None else config.get("seed", 0))
    image_size = int(config.get("image_size", 256))

    splits_meta = {}
    plan = [(split, int(count), domains["reference"]) for split, count in preset.items()]
    if not args.no_drift_split:
        drift_domain = domains.get(args.drift_domain)
        if drift_domain is None:
            print("unknown drift domain {0!r}".format(args.drift_domain), file=sys.stderr)
            return 2
        # The drifted holdout mirrors the test split so the two are directly
        # comparable: same count, same balance, one changed variable.
        plan.append(("test_drift", int(preset.get("test", 200)), drift_domain))

    for split, count, domain in plan:
        if not args.quiet:
            print("generating {0}: {1} frames under {2}".format(split, count, domain.name),
                  flush=True)
        records = ds.build_split(root, split, count, domain, dataset_seed,
                                 size=image_size, balanced=True,
                                 progress=not args.quiet)
        splits_meta[split] = ds.split_meta(records, domain)

    meta = {
        "source": "synthetic",
        "generator": "edgeinspect.data.synth",
        "preset": args.preset,
        "seed": dataset_seed,
        "image_size": image_size,
        "licence": "Apache-2.0 (generated by this repository; no third-party data)",
        "commercial_use": True,
        "splits": splits_meta,
        "config_sha256": hashlib.sha256(
            paths.resolve(args.config).read_bytes()).hexdigest(),
        "seconds": round(time.perf_counter() - started, 2),
    }
    ds.write_meta(root, meta)

    if args.verify:
        result = verify_determinism(root, dataset_seed, domains, args.verify)
        meta["determinism_check"] = result
        ds.write_meta(root, meta)
        if not result["deterministic"]:
            print("DETERMINISM FAILURE: {0}".format(result["mismatches"]), file=sys.stderr)
            return 1
        print("determinism check: {0} frames regenerated, all identical".format(
            result["checked"]))

    total = sum(v["count"] for v in splits_meta.values())
    print("wrote {0} frames to {1} in {2:.1f}s".format(
        total, paths.relative(root), meta["seconds"]))
    for split, info in splits_meta.items():
        print("  {0:<11} {1:>5} frames  {2:>4} defective  domain={3}".format(
            split, info["count"], info["defective"], info["domain"]["name"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
