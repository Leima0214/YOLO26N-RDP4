"""Dual-view detection loss for training-only RoadSnake network augmentation."""

from __future__ import annotations

from typing import Any

import torch

from ultralytics.utils.loss import E2ELoss


class RoadSnakeDualPathE2ELoss:
    """Average native and RoadSnake E2E losses while sharing one criterion and schedule."""

    def __init__(self, model) -> None:
        self.base = E2ELoss(model)
        self.last_native_total = None
        self.last_snake_total = None

    @property
    def one2many(self):
        return self.base.one2many

    @property
    def one2one(self):
        return self.base.one2one

    @property
    def o2m(self):
        return self.base.o2m

    @property
    def o2o(self):
        return self.base.o2o

    def __call__(self, preds: Any, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        parsed = preds[1] if isinstance(preds, tuple) else preds
        if isinstance(parsed, dict) and set(parsed) == {"native", "snake"}:
            native_total, native_items = self.base(parsed["native"], batch)
            snake_total, snake_items = self.base(parsed["snake"], batch)
            self.last_native_total = native_total.detach()
            self.last_snake_total = snake_total.detach()
            return 0.5 * (native_total + snake_total), 0.5 * (native_items + snake_items)
        if isinstance(parsed, dict) and set(parsed) == {"one2many", "one2one"}:
            # Validation/deployment deliberately executes only the native path.
            native_total, native_items = self.base(parsed, batch)
            self.last_native_total = native_total.detach()
            self.last_snake_total = None
            return native_total, native_items
        raise TypeError("RoadSnake DP expects dual training predictions or native E2E validation predictions")

    def update(self) -> None:
        """Advance the original O2M/O2O schedule exactly once per epoch."""
        self.base.update()
