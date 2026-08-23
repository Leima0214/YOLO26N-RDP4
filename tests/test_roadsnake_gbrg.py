"""Focused unit tests for RoadSnake-GBRG targets and inference boundary."""

import torch

from ultralytics.nn.roadsnake import RoadSnakeGBRGDetect
from ultralytics.utils.roadsnake_gbrg_loss import gbrg_anneal_scale, gbrg_region_targets


def test_gbrg_anneal_is_one_then_cosine_to_zero() -> None:
    assert gbrg_anneal_scale(1, 50, 100) == 1.0
    assert gbrg_anneal_scale(50, 50, 100) == 1.0
    assert abs(gbrg_anneal_scale(75, 50, 100) - 0.5) < 1e-12
    assert gbrg_anneal_scale(100, 50, 100) == 0.0
    assert gbrg_anneal_scale(101, 50, 100) == 0.0


def test_gbrg_masks_protect_gt_border() -> None:
    target, inside, clear = gbrg_region_targets(
        torch.tensor([0]), torch.tensor([[0.5, 0.5, 0.25, 0.125]]), 1, 32, 32, ignore_dilation=1.25
    )
    assert target.isfinite().all()
    assert inside.any() and clear.any()
    assert not (inside & clear).any()
    assert (target[~inside] == 0).all()


def test_gbrg_head_is_training_only() -> None:
    head = RoadSnakeGBRGDetect(nc=4, kernel_size=5, expansion=0.25, reg_max=1, end2end=True, ch=(32, 64, 128))
    features = [torch.randn(2, 32, 20, 20), torch.randn(2, 64, 10, 10), torch.randn(2, 128, 5, 5)]
    head.train()
    output = head([feature.clone() for feature in features])
    assert output["gbrg_region_logits"].shape == (2, 1, 20, 20)
    assert output["gbrg_p3_feature"].data_ptr() != 0
    head.eval()
    with torch.no_grad():
        output = head([feature.clone() for feature in features])
    if isinstance(output, tuple):
        output = output[1]
    assert "gbrg_region_logits" not in output
