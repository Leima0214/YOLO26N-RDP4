"""Unit checks for the fixed RoadSnake-Anneal schedule and native bypass."""

import torch

from scripts.train_roadsnake_anneal_30e import schedule
from ultralytics.nn.roadsnake import RoadSnakeAdapter


def test_schedule_boundaries():
    assert schedule(0) == ("full", 1.0, False)
    assert schedule(17) == ("full", 1.0, False)
    assert schedule(18) == ("withdraw", 1.0, True)
    assert 0.0 < schedule(23)[1] < 1.0
    assert schedule(24) == ("native", 0.0, True)
    assert schedule(29) == ("native", 0.0, True)


def test_native_scale_is_exact_bypass_without_adapter_gradients():
    adapter = RoadSnakeAdapter(16, kernel_size=5, expansion=0.5)
    adapter.gamma.data.fill_(0.5)
    adapter.set_anneal_scale(0.0)
    x = torch.randn(2, 16, 8, 8, requires_grad=True)
    output = adapter(x)
    torch.testing.assert_close(output, x, rtol=0, atol=0)
    output.sum().backward()
    assert x.grad is not None
    assert all(parameter.grad is None for parameter in adapter.parameters())


def test_historical_adapter_without_anneal_attribute_remains_fully_active():
    adapter = RoadSnakeAdapter(16, kernel_size=5, expansion=0.5)
    del adapter.anneal_scale
    x = torch.randn(1, 16, 8, 8)
    output = adapter(x)
    assert output.shape == x.shape
