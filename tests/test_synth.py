"""The dataset generator: determinism, and that the domains mean what they say."""

from __future__ import annotations

import numpy as np
import pytest

from edgeinspect.data import synth
from edgeinspect.monitoring import features


def test_frame_is_a_pure_function_of_its_seed(reference_domain):
    a = synth.generate_frame(1234, reference_domain)
    b = synth.generate_frame(1234, reference_domain)
    assert np.array_equal(a.image, b.image)
    assert np.array_equal(a.mask, b.mask)
    assert [d.to_dict() for d in a.defects] == [d.to_dict() for d in b.defects]


def test_different_seeds_give_different_frames(reference_domain):
    a = synth.generate_frame(1, reference_domain)
    b = synth.generate_frame(2, reference_domain)
    assert not np.array_equal(a.image, b.image)


def test_seeds_do_not_correlate_across_splits():
    """train[5] and val[5] must not be the same frame under a different name."""
    train = synth.frame_seed(99, "train", 5, "reference")
    val = synth.frame_seed(99, "val", 5, "reference")
    assert train != val
    # Adjacent indices must not be adjacent seeds either, or a hash-free
    # dataset_seed + index scheme has crept back in.
    a = synth.frame_seed(99, "train", 5, "reference")
    b = synth.frame_seed(99, "train", 6, "reference")
    assert abs(a - b) > 1000


def test_generation_order_does_not_matter(reference_domain):
    forward = [synth.generate_frame(synth.frame_seed(7, "test", i, "reference"),
                                    reference_domain).image for i in range(4)]
    backward = [synth.generate_frame(synth.frame_seed(7, "test", i, "reference"),
                                     reference_domain).image for i in reversed(range(4))]
    for index, image in enumerate(forward):
        assert np.array_equal(image, backward[3 - index])


def test_forced_labels_are_honoured(reference_domain):
    assert synth.generate_frame(11, reference_domain, force_label=1).label == 1
    assert synth.generate_frame(11, reference_domain, force_label=0).label == 0
    assert synth.generate_frame(11, reference_domain, force_label=0).defects == []


def test_defective_frames_have_a_mask_and_boxes(reference_domain):
    frame = synth.generate_frame(21, reference_domain, force_label=1)
    assert frame.mask.max() == 255
    assert frame.defects
    for defect in frame.defects:
        assert defect.x1 > defect.x0 and defect.y1 > defect.y0
        assert defect.area_px > 0
        assert defect.kind in synth.DEFECT_KINDS


def test_boxes_contain_the_masked_pixels(reference_domain):
    frame = synth.generate_frame(31, reference_domain, force_label=1)
    ys, xs = np.nonzero(frame.mask)
    union = [min(d.x0 for d in frame.defects), min(d.y0 for d in frame.defects),
             max(d.x1 for d in frame.defects), max(d.y1 for d in frame.defects)]
    assert union[0] <= xs.min() and union[1] <= ys.min()
    assert union[2] >= xs.max() and union[3] >= ys.max()


def test_defects_are_subtle(reference_domain):
    """A defect that is trivially separable makes every downstream number a lie."""
    coverage = []
    for index in range(24):
        frame = synth.generate_frame(1000 + index, reference_domain, force_label=1)
        coverage.append((frame.mask > 0).mean())
    assert np.mean(coverage) < 0.02, "defects cover too much of the frame to be subtle"


def test_defects_go_through_the_lens(reference_domain, rng):
    """Blur must affect the defect, not only the surface.

    If the defect were pasted on after the blur it would carry high-frequency
    energy the surface does not, and a model could find it by sharpness alone.
    """
    from dataclasses import replace

    sharp = replace(reference_domain, blur_sigma=0.2, name="sharp")
    soft = replace(reference_domain, blur_sigma=2.5, name="soft")
    sharp_frame = synth.generate_frame(55, sharp, force_label=1)
    soft_frame = synth.generate_frame(55, soft, force_label=1)
    mask = sharp_frame.mask > 0
    sharp_contrast = abs(float(sharp_frame.image[mask].mean())
                         - float(sharp_frame.image[~mask].mean()))
    soft_contrast = abs(float(soft_frame.image[mask].mean())
                        - float(soft_frame.image[~mask].mean()))
    assert soft_contrast < sharp_contrast, "blur did not attenuate the defect"


@pytest.mark.parametrize("field,direction", [
    ("brightness", "down"),
    ("sharpness", "down"),
    ("noise_level", "up"),
    ("contrast", "down"),
])
def test_lamp_ageing_moves_features_the_way_its_name_says(reference_domain, drift_domain,
                                                          field, direction):
    """The drift scenario has to be readable as a physical fault, not just a shift."""
    def measure(domain):
        values = []
        for index in range(12):
            frame = synth.generate_frame(2000 + index, domain, force_label=0)
            values.append(features.extract(frame.image.astype(np.float32) / 255.0)[field])
        return float(np.mean(values))

    reference = measure(reference_domain)
    drifted = measure(drift_domain)
    if direction == "down":
        assert drifted < reference, "{0} did not fall under lamp_ageing".format(field)
    else:
        assert drifted > reference, "{0} did not rise under lamp_ageing".format(field)


def test_material_batch_changes_texture_without_changing_brightness(domains):
    """The second scenario exists to catch a monitor that only watches brightness."""
    reference = domains["reference"]
    material = domains["material_batch"]

    def measure(domain, field):
        return float(np.mean([
            features.extract(synth.generate_frame(3000 + i, domain,
                                                  force_label=0).image.astype(np.float32) / 255)
            [field] for i in range(12)]))

    brightness_shift = abs(measure(material, "brightness") - measure(reference, "brightness"))
    texture_shift = abs(measure(material, "texture_scale") - measure(reference, "texture_scale"))
    assert brightness_shift < 0.02, "material_batch moved brightness; it should not"
    assert texture_shift > 0.01, "material_batch did not move the texture statistic"


def test_balanced_split_alternates_labels(reference_domain):
    labels = [frame.label for _index, frame
              in synth.generate_split(5, "test", 10, reference_domain, balanced=True)]
    assert labels == [0, 1] * 5
