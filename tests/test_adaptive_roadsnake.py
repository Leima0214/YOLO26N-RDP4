"""Synthetic regression tests for SA-RS and locked MG-SA-RS experiment engineering."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from scripts.prune_adaptive_roadsnake import output_error, prune_model
from ultralytics.nn.modules.head import Detect
from ultralytics.nn.roadsnake import RoadSnakeAdapter
from ultralytics.nn.roadsnake_adaptive import (
    MetricGuidedScaleAdaptiveRoadSnakeAdapter,
    MetricGuidedScaleAdaptiveRoadSnakeDetect,
    ScaleAdaptiveRoadSnakeAdapter,
    ScaleAdaptiveRoadSnakeDetect,
)
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "ultralytics/cfg/models/26/yolo26.yaml"
R1 = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-roadsnake-r1.yaml"
SA = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-sa-roadsnake.yaml"
MG = ROOT / "ultralytics/cfg/models/26/yolo26n-japan4-mg-sa-roadsnake.yaml"


@pytest.fixture
def sa() -> ScaleAdaptiveRoadSnakeAdapter:
    torch.manual_seed(42)
    return ScaleAdaptiveRoadSnakeAdapter(128, kernel_size=5, expansion=0.25, scale_max=2.5)


@pytest.fixture
def mg() -> MetricGuidedScaleAdaptiveRoadSnakeAdapter:
    torch.manual_seed(42)
    return MetricGuidedScaleAdaptiveRoadSnakeAdapter(128, kernel_size=5, expansion=0.25, scale_max=2.5)


def test_scale_head_starts_at_exact_unit_scale(sa: ScaleAdaptiveRoadSnakeAdapter) -> None:
    reduced = torch.randn(2, sa.hidden, 9, 11)
    scale_h, scale_v, _ = sa.predict_scales(reduced)
    assert torch.count_nonzero(sa.scale_head.weight) == 0
    assert torch.count_nonzero(sa.scale_head.bias) == 0
    torch.testing.assert_close(scale_h, torch.ones_like(scale_h), atol=0, rtol=0)
    torch.testing.assert_close(scale_v, torch.ones_like(scale_v), atol=0, rtol=0)


def test_gamma_zero_is_bit_exact_identity(sa: ScaleAdaptiveRoadSnakeAdapter) -> None:
    x = torch.randn(2, 128, 8, 8)
    with torch.inference_mode():
        output = sa(x)
    torch.testing.assert_close(output, x, atol=0, rtol=0)


def test_scale_one_sampler_is_bit_exact_r1(sa: ScaleAdaptiveRoadSnakeAdapter) -> None:
    r1 = RoadSnakeAdapter(128, kernel_size=5, expansion=0.25)
    r1.load_state_dict(sa.state_dict(), strict=False)
    feature = torch.randn(2, sa.hidden, 8, 8)
    offset = torch.randn(2, sa.kernel_size, 8, 8).tanh()
    ones = torch.ones(2, 1, 8, 8)
    with torch.inference_mode():
        for horizontal in (True, False):
            expected = r1._sample_curve(feature, offset, horizontal)
            actual = sa._sample_curve(feature, offset, horizontal, ones)
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(("scale", "span"), ((0.5, 2.0), (1.0, 4.0), (2.0, 8.0)))
def test_forced_scale_controls_only_longitudinal_span(
    sa: ScaleAdaptiveRoadSnakeAdapter, scale: float, span: float
) -> None:
    feature = torch.randn(1, sa.hidden, 9, 9)
    offset = torch.randn(1, sa.kernel_size, 9, 9).tanh()
    value = torch.full((1, 1, 9, 9), scale)
    h_x, h_y, h_curve = sa.sampling_coordinates(feature, offset, value, horizontal=True)
    v_x, v_y, v_curve = sa.sampling_coordinates(feature, offset, value, horizontal=False)
    assert float((h_x[:, -1] - h_x[:, 0]).mean()) == span
    assert float((v_y[:, -1] - v_y[:, 0]).mean()) == span
    torch.testing.assert_close(h_y - torch.arange(9).view(1, 1, 9, 1), h_curve, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(v_x - torch.arange(9).view(1, 1, 1, 9), v_curve, atol=1e-6, rtol=1e-6)


def test_scale_does_not_change_orthogonal_curve(sa: ScaleAdaptiveRoadSnakeAdapter) -> None:
    feature = torch.randn(1, sa.hidden, 7, 7)
    offset = torch.randn(1, sa.kernel_size, 7, 7).tanh()
    half = torch.full((1, 1, 7, 7), 0.5)
    double = torch.full((1, 1, 7, 7), 2.0)
    _, _, h_half = sa.sampling_coordinates(feature, offset, half, horizontal=True)
    _, _, h_double = sa.sampling_coordinates(feature, offset, double, horizontal=True)
    _, _, v_half = sa.sampling_coordinates(feature, offset, half, horizontal=False)
    _, _, v_double = sa.sampling_coordinates(feature, offset, double, horizontal=False)
    torch.testing.assert_close(h_half, h_double, atol=0, rtol=0)
    torch.testing.assert_close(v_half, v_double, atol=0, rtol=0)


def test_batch_isolation(sa: ScaleAdaptiveRoadSnakeAdapter) -> None:
    x = torch.randn(2, 128, 8, 8)
    sa.eval()
    with torch.no_grad():
        sa.gamma.fill_(0.05)
        isolated = sa(x[:1])
        batched = sa(x)[:1]
    torch.testing.assert_close(isolated, batched, atol=1e-6, rtol=1e-6)


def test_scale_and_r1_branches_receive_gradients(sa: ScaleAdaptiveRoadSnakeAdapter) -> None:
    x = torch.randn(2, 128, 8, 8, requires_grad=True)
    with torch.no_grad():
        sa.gamma.fill_(0.05)
    sa(x).square().mean().backward()
    for parameter in (sa.scale_head.weight, sa.offset.weight, sa.fuse.conv.weight, x):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_diagnostics_are_opt_in_and_detached(sa: ScaleAdaptiveRoadSnakeAdapter) -> None:
    expected = {
        "scale_h",
        "scale_v",
        "offset_h",
        "offset_v",
        "cumulative_offset_h",
        "cumulative_offset_v",
        "horizontal_grid_x",
        "horizontal_grid_y",
        "vertical_grid_x",
        "vertical_grid_y",
        "residual",
        "gamma",
    }
    assert sa.diagnostics() == {}
    sa.set_diagnostics(True)
    sa(torch.randn(1, 128, 6, 6))
    diagnostics = sa.diagnostics()
    assert expected <= diagnostics.keys()
    assert all(not value.requires_grad for value in diagnostics.values())


def test_mg_cues_are_finite_detached_and_per_image(mg: MetricGuidedScaleAdaptiveRoadSnakeAdapter) -> None:
    reduced = torch.randn(2, mg.hidden, 8, 8, requires_grad=True)
    cues, _ = mg.metric_cues(reduced)
    assert cues.shape == (2, 3, 8, 8)
    assert not cues.requires_grad
    assert torch.isfinite(cues).all()
    single, _ = mg.metric_cues(reduced[:1])
    torch.testing.assert_close(cues[:1], single, atol=1e-6, rtol=1e-6)


def test_zero_metric_branch_is_exact_sa_special_case(mg: MetricGuidedScaleAdaptiveRoadSnakeAdapter) -> None:
    reduced = torch.randn(2, mg.hidden, 8, 8)
    base = mg.scale_head(reduced)
    combined, _ = mg._scale_logits(reduced)
    torch.testing.assert_close(combined, base, atol=0, rtol=0)


def test_forced_metric_branch_changes_spatial_scale(mg: MetricGuidedScaleAdaptiveRoadSnakeAdapter) -> None:
    reduced = torch.randn(2, mg.hidden, 8, 8)
    with torch.no_grad():
        mg.scale_metric.weight.fill_(0.01)
        scale_h, scale_v, _ = mg.predict_scales(reduced)
    assert scale_h.std() > 0
    assert scale_v.std() > 0
    assert scale_h.min() >= 1 / mg.scale_max
    assert scale_h.max() <= mg.scale_max


def test_metric_branch_receives_gradient_but_cues_do_not_backprop_extra_path(
    mg: MetricGuidedScaleAdaptiveRoadSnakeAdapter,
) -> None:
    reduced = torch.randn(2, mg.hidden, 8, 8, requires_grad=True)
    scale_h, scale_v, extras = mg.predict_scales(reduced)
    (scale_h.mean() + scale_v.mean()).backward()
    assert mg.scale_metric.weight.grad is not None
    assert mg.scale_metric.weight.grad.abs().sum() > 0
    assert not extras["metric_cues"].requires_grad


def test_yaml_changes_only_detect_class_from_r1() -> None:
    r1 = yaml.safe_load(R1.read_text(encoding="utf-8"))
    sa = yaml.safe_load(SA.read_text(encoding="utf-8"))
    mg = yaml.safe_load(MG.read_text(encoding="utf-8"))
    assert sa["backbone"] == r1["backbone"] == mg["backbone"]
    assert sa["head"][:-1] == r1["head"][:-1] == mg["head"][:-1]
    assert sa["head"][-1][2] == "ScaleAdaptiveRoadSnakeDetect"
    assert mg["head"][-1][2] == "MetricGuidedScaleAdaptiveRoadSnakeDetect"


@pytest.mark.parametrize(
    ("path", "head_type"),
    ((SA, ScaleAdaptiveRoadSnakeDetect), (MG, MetricGuidedScaleAdaptiveRoadSnakeDetect)),
)
def test_yaml_builds_registered_detect_head(path: Path, head_type: type[Detect]) -> None:
    model = DetectionModel(str(path), nc=4, ch=3, verbose=False)
    assert type(model.model[-1]) is head_type
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
    assert model.model[-1].reg_max == 1
    assert model.model[-1].end2end


def test_step0_detector_matches_loaded_b0_exactly() -> None:
    torch.manual_seed(42)
    baseline = DetectionModel(str(BASELINE), nc=4, ch=3, verbose=False).eval()
    candidate = DetectionModel(str(SA), nc=4, ch=3, verbose=False).eval()
    candidate.load(baseline, verbose=False)
    image = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        torch.testing.assert_close(candidate(image)[0], baseline(image)[0], atol=0, rtol=0)


def test_physical_pruning_restores_native_detect_bit_exact() -> None:
    model = DetectionModel(str(SA), nc=4, ch=3, verbose=False).eval()
    with torch.no_grad():
        model.model[-1].road_snake.gamma.zero_()
    reference = copy.deepcopy(model).eval()
    pruned = copy.deepcopy(model).eval()
    prune_model(pruned, "sa")
    assert type(pruned.model[-1]) is Detect
    image = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        error = output_error(reference(image), pruned(image))
    assert error["max_abs"] == 0.0
    assert not any("road_snake" in key for key in pruned.state_dict())


def test_invalid_scale_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="greater than one"):
        ScaleAdaptiveRoadSnakeAdapter(128, scale_max=1.0)
