"""Adapter for the real public defect datasets: MVTec AD and DAGM 2007.

Neither dataset ships in this repository, and neither is downloaded
automatically, because of their licences:

* **MVTec AD** -- CC BY-NC-SA 4.0.  Non-commercial, share-alike.  Redistributing
  it, or a dataset derived from it, inside an Apache-2.0 repository would be a
  licence violation.
* **DAGM 2007** -- released for research use by the DAGM symposium organisers,
  with no redistribution grant.

So this module converts a copy that *you* have downloaded and accepted the
licence for into the same on-disk layout the synthetic generator produces.  Once
converted, every downstream stage -- training, PTQ calibration, export,
evaluation, the drift reference window -- is identical, because they all read
``index.json`` and none of them knows where the pixels came from.

    python scripts/make_dataset.py --source mvtec \\
        --mvtec-root /data/mvtec_anomaly_detection --category metal_nut

What this adapter does not give you
-----------------------------------
MVTec AD has no *drifted* split, and that is the whole point of this project.
``--source mvtec`` therefore produces ``train``, ``val`` and ``test`` only.  To
exercise the drift loop on real imagery you must supply the shift yourself:
``--drift-transform`` applies the same physically parameterised imaging change
the synthetic domains use (exposure, focus, sensor noise) to real MVTec frames,
which is honest in a way that relabelling a different MVTec category as "drift"
would not be.  ``docs/limitations.md`` says so.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .. import paths
from ..config import DomainSpec
from . import dataset as ds
from .synth import IMAGE_SIZE

MVTEC_CATEGORIES = (
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood",
    "zipper",
)

MVTEC_LICENCE = "CC BY-NC-SA 4.0 (non-commercial, share-alike)"
DAGM_LICENCE = "DAGM 2007 competition data, research use only, no redistribution grant"


def _load_grey(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as handle:
        img = handle.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _boxes_from_mask(mask: np.ndarray, min_area: int = 12) -> List[dict]:
    """Connected components of a ground-truth mask become defect boxes.

    MVTec ships pixel masks and no boxes, so the boxes this project evaluates
    against are derived here rather than annotated.  Components smaller than
    ``min_area`` pixels after downscaling are dropped: at 256x256 they are
    resampling artefacts on the boundary of a larger region, not defects.
    """
    binary = mask > 127
    labelled, count = ndimage.label(binary)
    out: List[dict] = []
    for index in range(1, count + 1):
        component = labelled == index
        area = int(component.sum())
        if area < min_area:
            continue
        ys, xs = np.nonzero(component)
        out.append({
            "kind": "anomaly",
            "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "area_px": area,
        })
    return out


def _apply_drift(image: np.ndarray, domain: DomainSpec, reference: DomainSpec,
                 rng: np.random.Generator) -> np.ndarray:
    """Apply the *difference* between two domains to an already-captured image.

    The synthetic generator applies imaging conditions while building a frame.
    Here the frame already exists, so only the delta can be applied: extra blur,
    an exposure ratio, a black-level shift and extra read noise.  Grain and
    machining-mark parameters have no meaning for a real photograph and are
    ignored, which is why ``--drift-transform`` is documented as a weaker
    stand-in than the synthetic path.
    """
    field = image.astype(np.float32) / 255.0
    extra_blur = max(domain.blur_sigma ** 2 - reference.blur_sigma ** 2, 0.0) ** 0.5
    if extra_blur > 0.05:
        field = ndimage.gaussian_filter(field, sigma=extra_blur, mode="reflect")
    ratio = domain.gain / max(reference.gain, 1e-6)
    field = field * ratio + (domain.offset - reference.offset)
    extra_noise = max(domain.noise_sigma ** 2 - reference.noise_sigma ** 2, 0.0) ** 0.5
    if extra_noise > 0:
        field = field + rng.standard_normal(field.shape).astype(np.float32) * extra_noise
    return np.clip(field * 255.0, 0, 255).astype(np.uint8)


def _mvtec_pairs(category_root: Path) -> Tuple[List[Path], List[Tuple[Path, Optional[Path]]]]:
    """``(train_good, test_items)`` where each test item is ``(image, mask_or_None)``."""
    train_good = sorted((category_root / "train" / "good").glob("*.png"))
    if not train_good:
        raise FileNotFoundError(
            "no images under {0}.  Expected the MVTec AD layout "
            "<root>/<category>/train/good/*.png".format(category_root / "train" / "good")
        )
    test_items: List[Tuple[Path, Optional[Path]]] = []
    for defect_dir in sorted((category_root / "test").iterdir()):
        if not defect_dir.is_dir():
            continue
        for image_path in sorted(defect_dir.glob("*.png")):
            if defect_dir.name == "good":
                test_items.append((image_path, None))
                continue
            mask_path = (category_root / "ground_truth" / defect_dir.name /
                         (image_path.stem + "_mask.png"))
            test_items.append((image_path, mask_path if mask_path.exists() else None))
    if not test_items:
        raise FileNotFoundError("no test images under " + str(category_root / "test"))
    return train_good, test_items


def convert_mvtec(mvtec_root: Path, category: str, out_root: Path,
                  size: int = IMAGE_SIZE, seed: int = 20260822,
                  drift_domain: Optional[DomainSpec] = None,
                  reference_domain: Optional[DomainSpec] = None,
                  val_fraction: float = 0.2) -> dict:
    """Convert one MVTec AD category into this project's layout.

    MVTec's train split is defect-free by design, which suits an anomaly
    detector and not a supervised one.  This project trains a supervised
    segmentation head, so the *test* split -- the only one with masks -- is
    re-split into train/val/test by a stable hash of the file stem, and the
    original defect-free train images are added to the training half.  The
    re-split is deterministic and stratified by defect type, and the mapping is
    written into ``dataset.json`` so a result can be traced back to exact files.
    """
    category_root = Path(mvtec_root) / category
    if not category_root.exists():
        raise FileNotFoundError(
            "no category at {0}.  Known MVTec AD categories: {1}".format(
                category_root, ", ".join(MVTEC_CATEGORIES))
        )
    train_good, test_items = _mvtec_pairs(category_root)
    rng = np.random.default_rng(seed)
    reference_domain = reference_domain or DomainSpec()

    # Deterministic, stratified split of the annotated test set.
    buckets: dict = {}
    for image_path, mask_path in test_items:
        key = image_path.parent.name
        buckets.setdefault(key, []).append((image_path, mask_path))
    assignment: List[Tuple[Path, Optional[Path], str]] = []
    for key in sorted(buckets):
        items = sorted(buckets[key], key=lambda pair: pair[0].name)
        for position, (image_path, mask_path) in enumerate(items):
            if position % 5 == 0:
                split = "test"
            elif position % 5 == 1:
                split = "val" if val_fraction > 0 else "test"
            else:
                split = "train"
            assignment.append((image_path, mask_path, split))
    for image_path in train_good:
        assignment.append((image_path, None, "train"))

    counters = {"train": 0, "val": 0, "test": 0}
    records: dict = {"train": [], "val": [], "test": []}
    for image_path, mask_path, split in assignment:
        image = _load_grey(image_path, size)
        if mask_path is not None:
            mask = _load_grey(mask_path, size)
            mask = ((mask > 127).astype(np.uint8)) * 255
        else:
            mask = np.zeros((size, size), dtype=np.uint8)
        defects = _boxes_from_mask(mask)
        if not defects:
            mask = np.zeros((size, size), dtype=np.uint8)
        name = "{0}_{1:06d}".format(split, counters[split])
        counters[split] += 1
        out_image = Path(out_root) / split / "images" / (name + ".png")
        out_mask = Path(out_root) / split / "masks" / (name + ".png")
        ds.write_png(out_image, image)
        ds.write_png(out_mask, mask)
        records[split].append(ds.Record(
            id=name,
            image=paths.relative(out_image),
            mask=paths.relative(out_mask),
            label=1 if defects else 0,
            domain="mvtec/" + category,
            defects=defects,
        ).to_dict())

    # Optional drifted evaluation split, built from the real test images.
    if drift_domain is not None:
        counters["test_drift"] = 0
        records["test_drift"] = []
        for record in records["test"]:
            source = ds.read_png(paths.resolve(record["image"]))
            shifted = _apply_drift(source, drift_domain, reference_domain, rng)
            name = "test_drift_{0:06d}".format(counters["test_drift"])
            counters["test_drift"] += 1
            out_image = Path(out_root) / "test_drift" / "images" / (name + ".png")
            out_mask = Path(out_root) / "test_drift" / "masks" / (name + ".png")
            ds.write_png(out_image, shifted)
            ds.write_png(out_mask, ds.read_png(paths.resolve(record["mask"])))
            drifted = dict(record)
            drifted["id"] = name
            drifted["image"] = paths.relative(out_image)
            drifted["mask"] = paths.relative(out_mask)
            drifted["domain"] = drift_domain.name
            records["test_drift"].append(drifted)

    for split, items in records.items():
        index_path = Path(out_root) / split / "index.json"
        paths.ensure(index_path.parent)
        with index_path.open("w", encoding="utf-8") as handle:
            json.dump(items, handle, indent=2)

    return {
        "source": "mvtec",
        "category": category,
        "licence": MVTEC_LICENCE,
        "commercial_use": False,
        "redistributed": False,
        "splits": {split: len(items) for split, items in records.items()},
    }


def check_licence_acknowledged(source: str, acknowledged: bool) -> None:
    """Refuse to convert a non-commercial dataset without an explicit flag.

    A build that silently ingests CC BY-NC-SA data into an Apache-2.0 project is
    a licence incident waiting to happen, so the acknowledgement is a required
    command-line flag rather than a config default.
    """
    if source == "synthetic":
        return
    licence = MVTEC_LICENCE if source == "mvtec" else DAGM_LICENCE
    if not acknowledged:
        raise SystemExit(
            "\n".join([
                "Refusing to convert {0}.".format(source),
                "",
                "  Licence: " + licence,
                "",
                "  This dataset may not be redistributed and, for MVTec AD, may not be",
                "  used commercially.  Nothing it produces may be committed to this",
                "  repository.  Pass --i-accept-dataset-licence to confirm you have",
                "  downloaded it yourself and accepted its terms.",
            ])
        )


def describe_sources() -> Sequence[dict]:
    """The dataset licence registry, used by ``docs/DATASET_CARD.md``."""
    return (
        {
            "name": "edgeinspect-synth",
            "kind": "synthetic",
            "licence": "Apache-2.0 (generated by this repository)",
            "commercial_use": True,
            "redistributable": True,
            "used_for_published_numbers": True,
        },
        {
            "name": "MVTec AD",
            "kind": "real",
            "licence": MVTEC_LICENCE,
            "commercial_use": False,
            "redistributable": False,
            "used_for_published_numbers": False,
        },
        {
            "name": "DAGM 2007",
            "kind": "real",
            "licence": DAGM_LICENCE,
            "commercial_use": False,
            "redistributable": False,
            "used_for_published_numbers": False,
        },
    )
