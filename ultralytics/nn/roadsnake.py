"""Pretrained-preserving curved sampling adapter for Japan4 road-damage detection."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect

__all__ = ("RoadSnakeAdapter", "RoadSnakeDetect", "RoadSnakeO2MDetect")


class RoadSnakeAdapter(nn.Module):
    """Add a zero-gated local/curved residual while preserving the incoming feature exactly at initialization."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        expansion: float = 0.25,
        max_offset: float = 1.0,
        gamma_init: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")
        if not 0 < expansion <= 1:
            raise ValueError(f"expansion must be in (0, 1], got {expansion}")
        if max_offset <= 0:
            raise ValueError(f"max_offset must be positive, got {max_offset}")

        hidden = max(8, int(round(channels * expansion / 8)) * 8)
        self.channels = channels
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.max_offset = float(max_offset)

        self.reduce = Conv(channels, hidden, 1)
        self.local = Conv(hidden, hidden, 3, g=hidden)
        self.offset = nn.Conv2d(hidden, 2 * kernel_size, 3, padding=1)
        self.horizontal_weight = nn.Parameter(
            torch.full((hidden, kernel_size), 1.0 / kernel_size)
        )
        self.vertical_weight = nn.Parameter(
            torch.full((hidden, kernel_size), 1.0 / kernel_size)
        )
        groups = max(1, min(8, hidden // 4))
        while hidden % groups:
            groups -= 1
        self.horizontal_norm = nn.GroupNorm(groups, hidden)
        self.vertical_norm = nn.GroupNorm(groups, hidden)
        self.act = nn.SiLU(inplace=True)
        self.fuse = Conv(3 * hidden, channels, 1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        positions = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        self.register_buffer("positions", positions, persistent=False)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    @staticmethod
    def _cumulative_offsets(offset: torch.Tensor) -> torch.Tensor:
        """Accumulate offsets outwards from the fixed kernel center without Python loops."""
        center = offset.shape[1] // 2
        accumulated = torch.zeros_like(offset)
        if center:
            left = torch.flip(
                torch.cumsum(torch.flip(offset[:, :center], dims=(1,)), dim=1),
                dims=(1,),
            )
            right = torch.cumsum(offset[:, center + 1 :], dim=1)
            accumulated[:, :center] = left
            accumulated[:, center + 1 :] = right
        return accumulated

    @staticmethod
    def _normalize_grid(coordinate: torch.Tensor, size: int) -> torch.Tensor:
        """Convert pixel coordinates to the align_corners=True grid_sample domain."""
        if size <= 1:
            return torch.zeros_like(coordinate)
        return coordinate.mul(2.0 / (size - 1)).sub(1.0)

    def _sample_curve(
        self,
        feature: torch.Tensor,
        orthogonal_offset: torch.Tensor,
        horizontal: bool,
    ) -> torch.Tensor:
        """Sample a continuous horizontal or vertical snake and collapse its K points channel-wise."""
        batch, channels, height, width = feature.shape
        dtype, device = feature.dtype, feature.device
        base_y, base_x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        base_y = base_y.view(1, 1, height, width)
        base_x = base_x.view(1, 1, height, width)
        positions = self.positions.to(device=device, dtype=dtype).view(
            1, self.kernel_size, 1, 1
        )
        curved = self._cumulative_offsets(orthogonal_offset).mul(self.max_offset)

        if horizontal:
            grid_y = base_y + curved
            grid_x = (base_x + positions).expand(batch, -1, -1, -1)
            weight = self.horizontal_weight
        else:
            grid_y = (base_y + positions).expand(batch, -1, -1, -1)
            grid_x = base_x + curved
            weight = self.vertical_weight

        grid = torch.stack(
            (self._normalize_grid(grid_x, width), self._normalize_grid(grid_y, height)),
            dim=-1,
        )
        grid = grid.permute(0, 2, 3, 1, 4).reshape(
            batch, height, width * self.kernel_size, 2
        )
        sampled = F.grid_sample(
            feature, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        sampled = sampled.reshape(
            batch, channels, height, width, self.kernel_size
        ).permute(0, 1, 4, 2, 3)
        return (
            sampled * weight.to(dtype=dtype).view(1, channels, self.kernel_size, 1, 1)
        ).sum(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the unmodified input at step zero, then learn a bounded curved residual."""
        reduced = self.reduce(x)
        offset_h, offset_v = self.offset(reduced).tanh().chunk(2, dim=1)
        horizontal = self.act(
            self.horizontal_norm(self._sample_curve(reduced, offset_h, horizontal=True))
        )
        vertical = self.act(
            self.vertical_norm(self._sample_curve(reduced, offset_v, horizontal=False))
        )
        local = self.local(reduced)
        residual = self.fuse(torch.cat((local, horizontal, vertical), dim=1))
        return x + self.gamma.to(dtype=x.dtype) * residual


class RoadSnakeDetect(Detect):
    """YOLO26 Detect head with one shared RoadSnake adapter on the P4 input only."""

    def __init__(
        self,
        nc: int = 80,
        kernel_size: int = 5,
        expansion: float = 0.25,
        max_offset: float = 1.0,
        gamma_init: float = 0.0,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if len(ch) != 3:
            raise ValueError(
                f"RoadSnakeDetect expects P3/P4/P5 inputs, got {len(ch)} levels"
            )
        self.road_snake = RoadSnakeAdapter(
            channels=ch[1],
            kernel_size=kernel_size,
            expansion=expansion,
            max_offset=max_offset,
            gamma_init=gamma_init,
        )

    def forward(self, x: list[torch.Tensor]):
        """Refine only P4 and leave the P3/P5 tensors and all Detect semantics unchanged."""
        x = list(x)
        x[1] = self.road_snake(x[1])
        return super().forward(x)


class RoadSnakeO2MDetect(Detect):
    """Use RoadSnake only for the training-time O2M proposer while keeping O2O and inference native."""

    schm_enabled = True
    rs_schm = True

    def __init__(
        self,
        nc: int = 80,
        kernel_size: int = 5,
        expansion: float = 0.25,
        max_offset: float = 1.0,
        gamma_init: float = 0.0,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ) -> None:
        super().__init__(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)
        if len(ch) != 3:
            raise ValueError(
                f"RoadSnakeO2MDetect expects P3/P4/P5 inputs, got {len(ch)} levels"
            )
        if not end2end:
            raise ValueError(
                "RoadSnakeO2MDetect requires the YOLO26 end-to-end O2M/O2O head"
            )
        self.road_snake = RoadSnakeAdapter(
            channels=ch[1],
            kernel_size=kernel_size,
            expansion=expansion,
            max_offset=max_offset,
            gamma_init=gamma_init,
        )

    def forward(self, x: list[torch.Tensor]):
        """Route native features to O2O and RoadSnake-refined P4 only to the training-time O2M proposer."""
        native = list(x)
        if self.training:
            proposer = list(native)
            proposer[1] = self.road_snake(proposer[1])
        else:
            # RoadSnake and its compute are absent from the deployed inference path even before physical pruning.
            proposer = native

        preds = self.forward_head(proposer, **self.one2many)
        native_detached = [feature.detach() for feature in native]
        one2one = self.forward_head(native_detached, **self.one2one)
        preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)
