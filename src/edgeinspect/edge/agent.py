"""The edge agent: what actually swaps a model on the device.

The registry decides what *should* be running.  The agent is the thing on the
device that notices, checks, and either switches or refuses.  Splitting the two
matters because they fail differently: the registry fails at the office, the
agent fails at 03:00 on a line that is still running.

The rule the agent enforces
---------------------------
**A new model is proved on the device before the old one is dropped.**  The
sequence is: read the manifest, verify the artefact hash, load the new backend
*alongside* the running one, run a smoke check, and only then swap the
reference.  If any step fails the old backend keeps serving and the failure is
reported.  A device that unloads the working model and then discovers the new
one does not load has taken the line down over a bad copy.

The smoke check is not "does it load"
-------------------------------------
Loading proves the file is a valid graph.  It does not prove the graph is the
model.  The check here runs a fixed probe set -- a defect-free frame and a
frame with a large synthetic defect -- and requires that the outputs are finite,
correctly shaped, and that the defective probe scores higher than the clean one.
That catches an inverted output, a mis-quantised graph that saturates, and a
graph exported from a randomly initialised network, all of which load
perfectly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .. import paths
from ..artefacts import sha256_file
from ..models import postprocess
from .backends.base import Backend, load as load_backend


@dataclass
class SmokeResult:
    passed: bool
    reason: str
    clean_score: float
    defect_score: float
    finite: bool
    shape: Tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "clean_score": round(self.clean_score, 6),
            "defect_score": round(self.defect_score, 6),
            "separation": round(self.defect_score - self.clean_score, 6),
            "finite": self.finite,
            "shape": list(self.shape),
        }


def probe_frames(size: int = 256, seed: int = 12345) -> Tuple[np.ndarray, np.ndarray]:
    """A clean probe and an unmistakably defective one.

    Deliberately not drawn from the dataset: the probe has to be constructible
    on a device that has no dataset, and it has to be crude enough that any
    working model separates the two.  A subtle probe would make the smoke check
    a second accuracy evaluation, which is not its job.
    """
    rng = np.random.default_rng(seed)
    clean = (0.5 + rng.standard_normal((1, 1, size, size)).astype(np.float32) * 0.03)
    defective = clean.copy()
    half = size // 2
    defective[0, 0, half - 12:half + 12, half - 4:half + 4] -= 0.45
    return np.clip(clean, 0, 1), np.clip(defective, 0, 1)


def smoke_check(backend: Backend, size: int = 256,
                min_separation: float = 0.05) -> SmokeResult:
    """Run the probe pair through a backend and judge the result."""
    clean, defective = probe_frames(size=size)
    try:
        clean_out = backend.infer(clean)
        defect_out = backend.infer(defective)
    except Exception as error:  # noqa: BLE001 - any failure here is a failed check
        return SmokeResult(False, "inference raised: {0}".format(error), 0.0, 0.0,
                           False, ())

    finite = bool(np.isfinite(clean_out).all() and np.isfinite(defect_out).all())
    shape = tuple(int(v) for v in clean_out.shape)
    if not finite:
        return SmokeResult(False, "non-finite output", 0.0, 0.0, False, shape)
    if len(shape) != 4 or shape[1] != 1:
        return SmokeResult(False, "unexpected output shape {0}".format(shape), 0.0, 0.0,
                           True, shape)

    top_k = backend.top_k
    clean_score = postprocess.image_score(clean_out[0], top_k=top_k)
    defect_score = postprocess.image_score(defect_out[0], top_k=top_k)
    separation = defect_score - clean_score
    if separation < min_separation:
        return SmokeResult(
            False,
            "defect probe scored {0:.4f} against clean {1:.4f}; separation {2:.4f} is "
            "below the required {3:.4f}".format(defect_score, clean_score, separation,
                                                min_separation),
            clean_score, defect_score, True, shape,
        )
    return SmokeResult(True, "ok", clean_score, defect_score, True, shape)


class EdgeAgent:
    """Holds the running backend and applies deployments to the device."""

    def __init__(self, deploy_root: Optional[Path] = None, backend: str = "auto",
                 providers=None, min_separation: float = 0.05):
        self.deploy_root = Path(deploy_root) if deploy_root else paths.DEPLOY
        self.backend_name = backend
        self.providers = providers
        self.min_separation = float(min_separation)
        self.backend: Optional[Backend] = None
        self.manifest: Optional[Dict] = None
        self.history: list = []

    # -- manifest --------------------------------------------------------

    def read_manifest(self) -> Dict:
        manifest_path = self.deploy_root / "current.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                "nothing deployed: no manifest at " + paths.relative(manifest_path)
            )
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def verify(self, manifest: Dict) -> None:
        artefact = paths.resolve(manifest["artefact"])
        if not artefact.exists():
            raise FileNotFoundError("deployed artefact is missing: " + manifest["artefact"])
        actual = sha256_file(artefact)
        if actual != manifest["sha256"]:
            raise ValueError(
                "artefact hash mismatch for {0}: on disk {1}, manifest {2}.  Refusing to "
                "load.  A partially transferred model loads and produces plausible "
                "nonsense.".format(manifest["artefact"], actual[:12], manifest["sha256"][:12])
            )

    # -- activation ------------------------------------------------------

    def _open(self, manifest: Dict) -> Backend:
        kwargs = {}
        if self.providers is not None:
            kwargs["providers"] = self.providers
        return load_backend(paths.resolve(manifest["artefact"]),
                            backend=self.backend_name, **kwargs)

    def activate(self, manifest: Optional[Dict] = None) -> Dict:
        """Load, check and switch.  Returns an outcome record.

        Never raises on a bad deployment: it reports ``activated: False`` and
        leaves the previous backend serving.  Missing or corrupt manifests do
        raise, because there is nothing sensible to fall back to when the
        instruction itself cannot be read.
        """
        manifest = manifest or self.read_manifest()
        self.verify(manifest)

        candidate = self._open(manifest)
        result = smoke_check(candidate, min_separation=self.min_separation)
        outcome = {
            "version": manifest.get("version"),
            "artefact": manifest.get("artefact"),
            "precision": manifest.get("precision"),
            "smoke": result.to_dict(),
        }
        if not result.passed:
            candidate.close()
            outcome["activated"] = False
            outcome["kept_version"] = self.manifest.get("version") if self.manifest else None
            outcome["detail"] = (
                "smoke check failed ({0}); the previously loaded model is still serving"
                .format(result.reason)
            )
            self.history.append(outcome)
            return outcome

        previous = self.backend
        previous_version = self.manifest.get("version") if self.manifest else None
        self.backend = candidate
        self.manifest = manifest
        if previous is not None:
            previous.close()
        outcome["activated"] = True
        outcome["replaced_version"] = previous_version
        outcome["detail"] = "activated version {0}".format(manifest.get("version"))
        self.history.append(outcome)
        return outcome

    def poll(self) -> Optional[Dict]:
        """Activate if the manifest names a version other than the running one."""
        manifest = self.read_manifest()
        if self.manifest is not None and manifest.get("version") == self.manifest.get("version"):
            return None
        return self.activate(manifest)

    @property
    def version(self) -> Optional[int]:
        return self.manifest.get("version") if self.manifest else None

    def close(self) -> None:
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def __enter__(self) -> "EdgeAgent":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
