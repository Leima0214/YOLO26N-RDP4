"""Focused invariants for the Japan4 S1, G1, and GS1 candidates."""

from __future__ import annotations

import copy

import pytest
import torch

from ultralytics.nn.modules.head import StripAwareResidual
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.region_loss import gaussian_region_targets


@pytest.fixture(scope="module")
def candidates():
    paths = {
        "b0": "ultralytics/cfg/models/26/yolo26.yaml",
        "s1": "ultralytics/cfg/models/26/yolo26n-japan4-s1-strip-regression.yaml",
        "g1": "ultralytics/cfg/models/26/yolo26n-japan4-g1-region-guidance.yaml",
        "gs1": "ultralytics/cfg/models/26/yolo26n-japan4-gs1-region-strip.yaml",
    }
    models = {name: DetectionModel(path, nc=4, ch=3, verbose=False).float() for name, path in paths.items()}
    for name in ("s1", "g1", "gs1"):
        models[name].load(models["b0"], verbose=False)
    return models


def deployed(model, image):
    model.eval()
    with torch.inference_mode():
        return model(image)[0]


def test_s1_initial_equivalence(candidates):
    image = torch.randn(1, 3, 128, 128)
    torch.testing.assert_close(deployed(candidates["s1"], image), deployed(candidates["b0"], image), atol=0, rtol=0)


def test_g1_detection_equivalence(candidates):
    image = torch.randn(1, 3, 128, 128)
    torch.testing.assert_close(deployed(candidates["g1"], image), deployed(candidates["b0"], image), atol=0, rtol=0)


def test_g1_region_target_shape():
    target = gaussian_region_targets(torch.tensor([0]), torch.tensor([[0.5, 0.5, 0.2, 0.1]]), 2, 16, 20)
    assert target.shape == (2, 1, 16, 20)
    assert 0 < target[0].max() <= 1
    assert target[1].count_nonzero() == 0


def test_g1_empty_gt():
    target = gaussian_region_targets(torch.empty(0), torch.empty(0, 4), 2, 8, 8)
    assert target.shape == (2, 1, 8, 8)
    assert target.count_nonzero() == 0


def test_g1_multi_gt():
    boxes = torch.tensor([[0.25, 0.5, 0.2, 0.1], [0.75, 0.5, 0.1, 0.2]])
    combined = gaussian_region_targets(torch.tensor([0, 0]), boxes, 1, 16, 16)
    separate = [gaussian_region_targets(torch.tensor([0]), box[None], 1, 16, 16) for box in boxes]
    torch.testing.assert_close(combined, torch.maximum(*separate))


def test_gs1_initial_equivalence(candidates):
    image = torch.randn(1, 3, 128, 128)
    torch.testing.assert_close(deployed(candidates["gs1"], image), deployed(candidates["b0"], image), atol=0, rtol=0)


def test_gs1_inference_removes_region_head(candidates):
    fused = copy.deepcopy(candidates["gs1"]).eval().fuse(verbose=False)
    assert fused.model[-1].region_heads is None
    assert fused.model[-1].strip_cv2 is None


def test_o2m_o2o_regression_symmetry(candidates):
    for name in ("s1", "gs1"):
        head = candidates[name].model[-1]
        many = [module for module in head.strip_cv2 if isinstance(module, StripAwareResidual)]
        one = [module for module in head.one2one_strip_cv2 if isinstance(module, StripAwareResidual)]
        assert [module.horizontal.kernel_size for module in many] == [(1, 7), (1, 5)]
        assert [module.horizontal.kernel_size for module in one] == [(1, 7), (1, 5)]
        assert [module.state_dict().keys() for module in many] == [module.state_dict().keys() for module in one]
