"""Focused invariants for the Japan4 S2 shape-supervised strip head."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.head import ShapeSupervisedStripResidual
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.shape_gate_loss import shape_gate_targets


B0 = "ultralytics/cfg/models/26/yolo26.yaml"
S2 = "ultralytics/cfg/models/26/yolo26n-japan4-s2-shape-strip.yaml"


def deployed(model, image):
    model.eval()
    with torch.inference_mode():
        return model(image)[0]


def test_shape_targets_encode_orientation():
    boxes = torch.tensor([[0.0, 0.0, 6.0, 1.0], [0.0, 0.0, 1.0, 6.0], [0.0, 0.0, 2.0, 2.0]])
    targets = shape_gate_targets(boxes)
    assert targets[0, 0] > 0.9 and targets[0, 1] < 0.1
    assert targets[1, 1] > 0.9 and targets[1, 0] < 0.1
    assert torch.all(targets[2] < 0.1)


def test_adapter_starts_identity_and_gate_loss_is_trainable():
    adapter = ShapeSupervisedStripResidual(8, 7)
    x = torch.randn(2, 8, 12, 12, requires_grad=True)
    output, logits = adapter.forward_with_gates(x)
    torch.testing.assert_close(output, x, atol=0, rtol=0)
    assert 0.05 < logits.sigmoid().min() and logits.sigmoid().max() < 0.95
    F.binary_cross_entropy_with_logits(logits, torch.rand_like(logits)).backward()
    assert adapter.gate.weight.grad is not None and adapter.gate.weight.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_s2_initial_detection_is_exact_b0():
    torch.manual_seed(42)
    baseline = DetectionModel(B0, nc=4, ch=3, verbose=False).float()
    candidate = DetectionModel(S2, nc=4, ch=3, verbose=False).float()
    candidate.load(baseline, verbose=False)
    image = torch.randn(1, 3, 128, 128)
    torch.testing.assert_close(deployed(candidate, image), deployed(baseline, image), atol=0, rtol=0)


def test_s2_o2m_o2o_symmetry_and_finite_loss():
    model = DetectionModel(S2, nc=4, ch=3, verbose=False).float()
    head = model.model[-1]
    many = [module for module in head.shape_strip_cv2 if isinstance(module, ShapeSupervisedStripResidual)]
    one = [module for module in head.one2one_shape_strip_cv2 if isinstance(module, ShapeSupervisedStripResidual)]
    assert [module.horizontal.kernel_size for module in many] == [(1, 7), (1, 5)]
    assert [module.horizontal.kernel_size for module in one] == [(1, 7), (1, 5)]
    assert [module.state_dict().keys() for module in many] == [module.state_dict().keys() for module in one]

    model.args = type("Args", (), {"box": 7.5, "cls": 0.5, "dfl": 1.5, "epochs": 5})()
    model.train()
    batch = {
        "img": torch.rand(2, 3, 128, 128),
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.1], [0.5, 0.5, 0.1, 0.5]]),
    }
    loss, items = model.loss(batch)
    assert loss.shape == (4,) and items.shape == (4,)
    assert torch.isfinite(loss).all() and torch.isfinite(items).all()
    loss.sum().backward()
    assert all(module.gate.weight.grad is not None for module in many + one)
    assert any(abs(float(module.gamma_h.grad)) > 0 or abs(float(module.gamma_v.grad)) > 0 for module in many + one)


def test_square_threshold_matches_protocol():
    assert math.isclose(math.log(1.5), 0.4054651081081644)
