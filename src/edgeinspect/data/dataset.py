"""On-disk dataset layout, writing and reading.

::

    data/processed/
      dataset.json          preset, seed, domain, per-split counts and hashes
      train/index.json      one record per frame
      train/images/*.png
      train/masks/*.png
      val/... test/... test_drift/...

The index is the contract.  Everything downstream -- training, calibration,
evaluation, the drift reference window -- reads ``index.json`` and never globs
a directory, so a partially written split is a hard error rather than a
quietly smaller dataset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image

from .. import paths
from ..config import DomainSpec
from . import synth

SPLITS = ("train", "val", "test")


@dataclass
class Record:
    """One frame in an index."""

    id: str
    image: str
    mask: str
    label: int
    domain: str
    defects: List[dict]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "image": self.image,
            "mask": self.mask,
            "label": self.label,
            "domain": self.domain,
            "defects": self.defects,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Record":
        return cls(
            id=data["id"],
            image=data["image"],
            mask=data["mask"],
            label=int(data["label"]),
            domain=data.get("domain", "reference"),
            defects=list(data.get("defects", [])),
        )

    def boxes(self) -> np.ndarray:
        """Defect boxes as an ``(n, 4)`` float array of ``x0, y0, x1, y1``."""
        if not self.defects:
            return np.zeros((0, 4), dtype=np.float32)
        return np.asarray([d["box"] for d in self.defects], dtype=np.float32)


def write_png(path: Path, array: np.ndarray) -> None:
    paths.ensure(path.parent)
    Image.fromarray(array).save(path, format="PNG", optimize=True)


def read_png(path: Path) -> np.ndarray:
    with Image.open(path) as handle:
        return np.asarray(handle.convert("L"), dtype=np.uint8)


def _hash_index(records: Sequence[Record]) -> str:
    payload = json.dumps([r.to_dict() for r in records], sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_split(root: Path, split: str, count: int, domain: DomainSpec,
                dataset_seed: int, size: int = synth.IMAGE_SIZE,
                balanced: bool = True, progress: bool = False) -> List[Record]:
    """Generate and write one split.  Returns the index."""
    split_dir = Path(root) / split
    paths.ensure(split_dir / "images", split_dir / "masks")
    records: List[Record] = []
    for index, frame in synth.generate_split(dataset_seed, split, count, domain,
                                             size=size, balanced=balanced):
        name = "{0}_{1:06d}".format(split, index)
        image_path = split_dir / "images" / (name + ".png")
        mask_path = split_dir / "masks" / (name + ".png")
        write_png(image_path, frame.image)
        write_png(mask_path, frame.mask)
        records.append(Record(
            id=name,
            image=paths.relative(image_path),
            mask=paths.relative(mask_path),
            label=frame.label,
            domain=frame.domain,
            defects=[d.to_dict() for d in frame.defects],
        ))
        if progress and (index + 1) % 200 == 0:
            print("  {0}: {1}/{2}".format(split, index + 1, count), flush=True)
    with (split_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump([r.to_dict() for r in records], handle, indent=2)
    return records


def load_index(root: Path, split: str) -> List[Record]:
    """Read one split's index.  Missing or empty is an error, not an empty list."""
    index_path = Path(root) / split / "index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            "no index at {0}.  Run `make data` first.".format(paths.relative(index_path))
        )
    with index_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not raw:
        raise ValueError("empty index at " + paths.relative(index_path))
    return [Record.from_dict(item) for item in raw]


def load_meta(root: Path) -> dict:
    meta_path = Path(root) / "dataset.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            "no dataset at {0}.  Run `make data` first.".format(paths.relative(meta_path))
        )
    with meta_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_meta(root: Path, meta: dict) -> None:
    paths.ensure(Path(root))
    with (Path(root) / "dataset.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def split_meta(records: Sequence[Record], domain: DomainSpec) -> dict:
    labels = [r.label for r in records]
    kinds: Dict[str, int] = {}
    for record in records:
        for defect in record.defects:
            kinds[defect["kind"]] = kinds.get(defect["kind"], 0) + 1
    return {
        "count": len(records),
        "defective": int(sum(labels)),
        "good": int(len(labels) - sum(labels)),
        "defect_instances": kinds,
        "domain": domain.to_dict(),
        "index_sha256": _hash_index(records),
    }


def load_arrays(root: Path, split: str, limit: Optional[int] = None,
                records: Optional[Iterable[Record]] = None):
    """Load a split into memory as ``(images, masks, labels, records)``.

    Images come back as float32 in ``[0, 1]`` with shape ``(n, 1, H, W)``, which
    is what both the torch model and the ONNX graph expect.  These datasets are
    small on purpose; a split that does not fit in memory belongs in a
    ``torch.utils.data.Dataset``, and :class:`InspectionDataset` is that.
    """
    items = list(records) if records is not None else load_index(root, split)
    if limit is not None:
        items = items[:limit]
    images = np.stack([read_png(paths.resolve(r.image)) for r in items]).astype(np.float32)
    masks = np.stack([read_png(paths.resolve(r.mask)) for r in items]).astype(np.float32)
    labels = np.asarray([r.label for r in items], dtype=np.int64)
    images = (images / 255.0)[:, None, :, :]
    masks = (masks > 127).astype(np.float32)[:, None, :, :]
    return images, masks, labels, items
