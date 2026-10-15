"""Quantisation, calibration selection, and the TensorRT plan.

The quantisation tests need the exported graphs, so they skip without them.  The
calibration-selection and TensorRT-plan tests need neither and run everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgeinspect import paths
from edgeinspect.export import tensorrt_export
from edgeinspect.quantise import calibration, ptq


# ---------------------------------------------------------------------------
# Calibration set selection
# ---------------------------------------------------------------------------

def test_selection_hits_the_requested_defect_ratio(rng):
    labels = np.array([0] * 500 + [1] * 500)
    stats = np.stack([rng.normal(0.5, 0.05, 1000), rng.normal(0.16, 0.01, 1000)], axis=1)
    chosen = calibration.select_indices(labels, stats, count=100, defect_fraction=0.5)
    assert len(chosen) == 100
    assert labels[chosen].mean() == pytest.approx(0.5, abs=0.02)


def test_selection_covers_the_brightness_range(rng):
    """Sampling the middle of the exposure range 128 times calibrates the wrong ranges."""
    labels = np.array([0, 1] * 500)
    brightness = np.linspace(0.3, 0.7, 1000)
    stats = np.stack([brightness, np.full(1000, 0.16)], axis=1)
    chosen = calibration.select_indices(labels, stats, count=64, defect_fraction=0.5)
    covered = brightness[chosen]
    assert covered.min() < 0.35 and covered.max() > 0.65, (
        "the calibration set only sampled the middle of the exposure range")


def test_selection_is_deterministic(rng):
    labels = np.array([0, 1] * 200)
    stats = rng.random((400, 2))
    first = calibration.select_indices(labels, stats, count=50, seed=3)
    second = calibration.select_indices(labels, stats, count=50, seed=3)
    assert first == second


def test_selection_asks_for_more_than_exists(rng):
    labels = np.array([0, 1] * 10)
    stats = rng.random((20, 2))
    assert len(calibration.select_indices(labels, stats, count=200)) == 20


def test_a_defect_free_calibration_set_is_requestable_and_recorded(dataset_root):
    """The failure this project documents has to be constructible on demand."""
    _frames, manifest = calibration.build(dataset_root, split="train", count=32,
                                          defect_fraction=0.0)
    assert manifest["defect_fraction_actual"] == 0.0
    assert manifest["defect_fraction_requested"] == 0.0


def test_calibration_frames_are_in_graph_input_space(dataset_root):
    frames, manifest = calibration.build(dataset_root, split="train", count=16)
    assert manifest["input_space"] == {"mean": 0.5, "scale": 2.0}
    assert frames.min() >= -1.001 and frames.max() <= 1.001
    assert frames.min() < 0.0, "centred frames must actually use the negative half"


def test_calibration_round_trips(dataset_root, tmp_path):
    frames, manifest = calibration.build(dataset_root, split="train", count=16)
    calibration.save(frames, manifest, tmp_path, name="cal")
    loaded, loaded_manifest = calibration.load(tmp_path, name="cal")
    assert np.array_equal(frames, loaded)
    assert loaded_manifest["count"] == manifest["count"]


def test_missing_calibration_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="make quantise"):
        calibration.load(tmp_path)


# ---------------------------------------------------------------------------
# Quantisation
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.needs_model
def test_entropy_calibration_degenerates_under_symmetric_activations(fp32_model, tmp_path,
                                                                     dataset_root):
    """Pinned so an onnxruntime upgrade that fixes this fails loudly.

    On onnxruntime 1.29, ``CalibrationMethod.Entropy`` with symmetric
    activations produces a byte-identical graph to MinMax: the calibrator runs,
    computes clipping thresholds, and the symmetric path discards them for the
    raw absolute maximum.  This project therefore builds Percentile as its
    second variant.  If this test starts failing, entropy has begun to do
    something, and ``configs/export.yaml`` and the module docstring in
    ``edgeinspect/quantise/ptq.py`` both need revisiting.
    """
    frames, manifest = calibration.build(dataset_root, split="train", count=32)
    minmax = ptq.quantise(fp32_model, tmp_path / "m.onnx", frames, method="minmax",
                          calibration_manifest=manifest)
    entropy = ptq.quantise(fp32_model, tmp_path / "e.onnx", frames, method="entropy",
                           calibration_manifest=manifest)
    assert minmax.read_bytes() == entropy.read_bytes(), (
        "entropy calibration now differs from minmax under symmetric activations; "
        "the documented finding is out of date")


@pytest.mark.slow
@pytest.mark.needs_model
def test_percentile_calibration_does_not_degenerate(fp32_model, tmp_path, dataset_root):
    frames, manifest = calibration.build(dataset_root, split="train", count=32)
    minmax = ptq.quantise(fp32_model, tmp_path / "m.onnx", frames, method="minmax",
                          calibration_manifest=manifest)
    percentile = ptq.quantise(fp32_model, tmp_path / "p.onnx", frames,
                              method="percentile", calibration_manifest=manifest)
    assert minmax.read_bytes() != percentile.read_bytes()


@pytest.mark.needs_model
def test_quantised_graph_keeps_its_sidecar(int8_model):
    from edgeinspect.artefacts import read_sidecar

    sidecar = read_sidecar(int8_model)
    assert sidecar["precision"] == "int8"
    assert "quantisation" in sidecar
    assert sidecar["input"]["mean"] == 0.5
    assert "threshold" in sidecar["decision"]


@pytest.mark.needs_model
def test_int8_is_materially_smaller_than_fp32(fp32_model, int8_model):
    report = ptq.size_report({"fp32": fp32_model, "int8": int8_model})
    assert report["int8"]["ratio_vs_fp32"] < 0.5


def test_unknown_quantisation_method_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="method must be one of"):
        ptq.quantise(tmp_path / "x.onnx", tmp_path / "y.onnx",
                     np.zeros((1, 1, 8, 8), dtype=np.float32), method="vibes")


def test_calibration_reader_yields_batches_and_rewinds():
    frames = np.arange(40, dtype=np.float32).reshape(10, 1, 2, 2)
    reader = ptq.ArrayCalibrationReader(frames, batch_size=4)
    batches = []
    while True:
        batch = reader.get_next()
        if batch is None:
            break
        batches.append(batch["input"].shape[0])
    assert batches == [4, 4, 2]
    reader.rewind()
    assert reader.get_next()["input"].shape[0] == 4


@pytest.mark.needs_model
def test_output_conv_nodes_finds_the_head(fp32_model):
    convs = ptq.output_conv_nodes(fp32_model, count=99)
    assert convs, "no Conv nodes found in the exported graph"
    assert "head" in convs[-1].lower()


# ---------------------------------------------------------------------------
# The TensorRT plan
# ---------------------------------------------------------------------------

def test_the_plan_is_inspectable_without_tensorrt():
    plan = list(tensorrt_export.plan_builds("models/surface_defect_fp32.onnx", "models"))
    assert [item["precision"] for item in plan] == list(tensorrt_export.PRECISIONS)
    assert all(item["engine"].endswith(".engine") for item in plan)
    assert [item["needs_calibration"] for item in plan] == [False, False, True]


def test_the_plan_names_engines_after_the_model_not_the_source_precision():
    plan = list(tensorrt_export.plan_builds("models/surface_defect_fp32.onnx", "models"))
    assert plan[0]["engine"].endswith("surface_defect_fp32.engine")
    assert plan[1]["engine"].endswith("surface_defect_fp16.engine")


def test_an_unknown_precision_is_rejected():
    with pytest.raises(ValueError, match="unknown precision"):
        list(tensorrt_export.plan_builds("a.onnx", "b", precisions=["fp8"]))


def test_building_without_tensorrt_says_where_to_run_it():
    if tensorrt_export.available():
        pytest.skip("TensorRT is importable here, so the guidance path cannot be tested")
    with pytest.raises(tensorrt_export.NotAvailable, match="docs/jetson.md"):
        tensorrt_export.build_engine("a.onnx", "b.engine", precision="fp16")


def test_int8_without_a_calibration_set_is_rejected_before_anything_else():
    with pytest.raises((ValueError, tensorrt_export.NotAvailable)):
        tensorrt_export.build_engine("a.onnx", "b.engine", precision="int8",
                                     calibration=None)


def test_the_jetson_paths_are_marked_as_never_run():
    """Two files make Jetson claims.  Both have to say they were never executed."""
    for relative in ("scripts/bench_jetson.sh", "configs/benchmark_jetson.yaml",
                     "src/edgeinspect/export/tensorrt_export.py",
                     "src/edgeinspect/edge/backends/trt_backend.py"):
        text = (paths.ROOT / relative).read_text(encoding="utf-8").lower()
        assert "never" in text and "jetson" in text, (
            "{0} makes a Jetson claim without saying it was never run".format(relative))
