"""Pre-processing, post-processing, and the contract between torch and NumPy.

The recurring hazard in this project is two implementations of the same idea
drifting apart: the image score is defined in torch for the loss and in NumPy
for the metric, and the input normalisation is applied in the training loop and
declared in the ONNX sidecar.  Each of those pairs gets a test that fails when
they disagree.
"""

from __future__ import annotations

import numpy as np
import pytest

from edgeinspect.models import postprocess, preprocess


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def test_default_contract_maps_the_unit_interval_onto_minus_one_to_one():
    spec = preprocess.DEFAULT
    assert spec.apply(np.array([0.0, 0.5, 1.0])).tolist() == [-1.0, 0.0, 1.0]


def test_apply_and_invert_round_trip(rng):
    spec = preprocess.DEFAULT
    values = rng.random((4, 1, 8, 8)).astype(np.float32)
    assert np.allclose(spec.invert(spec.apply(values)), values, atol=1e-6)


def test_sidecar_without_a_mean_is_treated_as_the_identity():
    """An older graph was trained on [0, 1] and must keep being fed [0, 1]."""
    spec = preprocess.InputSpec.from_sidecar({"input": {"scale": 0.00392, "size": 256}})
    assert spec.mean == 0.0 and spec.scale == 1.0
    assert spec.apply(np.array([0.25])).tolist() == [0.25]


def test_sidecar_round_trips_the_contract():
    spec = preprocess.InputSpec(mean=0.25, scale=4.0, size=128, channels=1)
    restored = preprocess.InputSpec.from_sidecar({"input": spec.to_dict()})
    assert (restored.mean, restored.scale, restored.size) == (0.25, 4.0, 128)


def test_to_model_input_is_contiguous_float32(rng):
    values = rng.random((2, 1, 8, 8))
    out = preprocess.to_model_input(values)
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]


def test_training_config_declares_the_module_default():
    """A config that silently disagrees with the module is a quiet accuracy bug."""
    from edgeinspect import paths
    from edgeinspect.config import load_yaml

    model_config = load_yaml(paths.CONFIGS / "train.yaml").get("model", {})
    assert float(model_config["input_mean"]) == preprocess.MEAN
    assert float(model_config["input_scale"]) == preprocess.SCALE


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def test_sigmoid_matches_the_definition_where_it_is_safe(rng):
    values = rng.normal(0, 3, 500).astype(np.float32)
    expected = 1.0 / (1.0 + np.exp(-values.astype(np.float64)))
    assert np.allclose(postprocess.sigmoid(values), expected, atol=1e-6)


def test_sigmoid_does_not_overflow_on_confident_logits():
    """A confident good-part prediction produces logits around -40.

    Overflow and invalid are errors; underflow is not.  ``exp(-800)`` flushing to
    zero is the correct answer here, and treating it as a failure would mean
    rejecting the very inputs the stable form exists to handle.
    """
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = postprocess.sigmoid(np.array([-800.0, -40.0, 0.0, 40.0, 800.0],
                                           dtype=np.float32))
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)


def test_image_score_is_the_mean_of_the_top_k():
    grid = np.zeros((1, 8, 8), dtype=np.float32)
    grid[0, 0, :4] = [4.0, 3.0, 2.0, 1.0]
    assert postprocess.image_score(grid, top_k=4) == pytest.approx(2.5)
    assert postprocess.image_score(grid, top_k=1) == pytest.approx(4.0)


def test_image_score_clamps_k_to_the_grid(rng):
    grid = rng.normal(size=(1, 2, 2)).astype(np.float32)
    assert postprocess.image_score(grid, top_k=999) == pytest.approx(float(grid.mean()))


def test_empty_heatmap_scores_zero():
    assert postprocess.image_score(np.zeros((1, 0, 0), dtype=np.float32)) == 0.0


@pytest.mark.needs_torch
def test_numpy_and_torch_image_scores_agree(rng):
    """The metric and the training objective must be the same function."""
    torch = pytest.importorskip("torch")
    from edgeinspect.models.net import image_logit

    logits = rng.normal(0, 4, size=(6, 1, 32, 32)).astype(np.float32)
    torch_scores = image_logit(torch.from_numpy(logits), top_k=8).numpy()
    numpy_scores = np.asarray([postprocess.image_score(logit, top_k=8) for logit in logits])
    assert np.allclose(torch_scores, numpy_scores, atol=1e-5)


def test_boxes_come_back_empty_below_the_threshold():
    assert postprocess.heatmap_to_boxes(np.zeros((32, 32), dtype=np.float32),
                                        threshold=0.5) == []


def test_a_single_hot_cell_becomes_one_box_at_the_right_place():
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[4, 6] = 0.9
    boxes = postprocess.heatmap_to_boxes(heatmap, threshold=0.5, input_size=256)
    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x0, box.y0, box.x1, box.y1) == (48, 32, 56, 40)
    assert box.score == pytest.approx(0.9)


def test_separate_blobs_become_separate_boxes():
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[2:4, 2:4] = 0.8
    heatmap[20:22, 20:22] = 0.7
    boxes = postprocess.heatmap_to_boxes(heatmap, threshold=0.5)
    assert len(boxes) == 2
    assert boxes[0].score > boxes[1].score, "boxes must come back score-ordered"


def test_boxes_are_capped(rng):
    heatmap = np.zeros((32, 32), dtype=np.float32)
    heatmap[::2, ::2] = 0.9        # 256 disconnected components
    boxes = postprocess.heatmap_to_boxes(heatmap, threshold=0.5, max_boxes=5)
    assert len(boxes) == 5


def test_boxes_never_leave_the_frame():
    heatmap = np.ones((32, 32), dtype=np.float32)
    for box in postprocess.heatmap_to_boxes(heatmap, threshold=0.5, input_size=256):
        assert 0 <= box.x0 < box.x1 <= 256
        assert 0 <= box.y0 < box.y1 <= 256


def test_heatmap_rank_is_rejected():
    with pytest.raises(ValueError, match="2D"):
        postprocess.heatmap_to_boxes(np.zeros((2, 3, 4, 5), dtype=np.float32))
