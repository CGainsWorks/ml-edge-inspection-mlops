"""Shared fixtures.

The suite is designed so that most of it runs with no dataset, no weights and no
GPU: that is what makes it worth running on every push.  Anything that needs
more says so with a marker, and ``pytest -m "not needs_dataset and not
needs_model"`` is a complete, meaningful run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from edgeinspect import paths  # noqa: E402
from edgeinspect.config import DomainSpec, load_domains, load_yaml  # noqa: E402


@pytest.fixture(scope="session")
def data_config() -> dict:
    return load_yaml(paths.CONFIGS / "data.yaml")


@pytest.fixture(scope="session")
def domains(data_config) -> dict:
    return load_domains(data_config)


@pytest.fixture(scope="session")
def reference_domain(domains) -> DomainSpec:
    return domains["reference"]


@pytest.fixture(scope="session")
def drift_domain(domains) -> DomainSpec:
    return domains["lamp_ageing"]


@pytest.fixture(scope="session")
def dataset_root() -> Path:
    """The generated dataset, or skip.

    Tests that need real frames are marked ``needs_dataset``; this makes the
    skip say what to run rather than failing with a missing file.
    """
    for candidate in (paths.DATA / "processed", paths.DATA / "tiny"):
        if (candidate / "dataset.json").exists():
            return candidate
    pytest.skip("no dataset: run `make data` or `make data-tiny`")


@pytest.fixture(scope="session")
def fp32_model() -> Path:
    candidate = paths.MODELS / "surface_defect_fp32.onnx"
    if not candidate.exists():
        pytest.skip("no exported model: run `make train export`")
    return candidate


@pytest.fixture(scope="session")
def int8_model() -> Path:
    candidate = paths.MODELS / "surface_defect_int8.onnx"
    if not candidate.exists():
        pytest.skip("no INT8 model: run `make quantise`")
    return candidate


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(20260822)


def pytest_collection_modifyitems(config, items):
    """Mark anything in a torch-importing module as needing torch."""
    for item in items:
        if "needs_torch" in item.keywords:
            try:
                import torch  # noqa: F401
            except ImportError:
                item.add_marker(pytest.mark.skip(reason="torch is not installed"))
