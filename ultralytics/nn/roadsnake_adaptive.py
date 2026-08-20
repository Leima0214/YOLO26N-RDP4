"""Scale-adaptive RoadSnake variants for pretrained-preserving Japan4 experiments.

The frozen RoadSnake-R1 implementation remains in :mod:`ultralytics.nn.roadsnake`.
This module subclasses it and changes only the longitudinal sample spacing.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.roadsnake import RoadSnakeAdapter, RoadSnakeDetect

__all__ = (
    "ScaleAdaptiveRoadSnakeAdapter",
    "ScaleAdaptiveRoadSnakeDetect",
    "MetricGuidedScaleAdaptiveRoadSnakeAdapter",
    "MetricGuidedScaleAdaptiveRoadSnakeDetect",
)


class ScaleAdaptiveRoadSnakeAdapter(RoadSnakeAdapter):
    """Learn independent horizontal and vertical main-axis sampling spans.

    Scale changes only how far the snake samples along its main axis. The
    inherited cumulative orthogonal displacement continues to determine how
    the snake bends. Zero-initialized scale logits produce an exact scale of
    one, so this geometry is identical to RoadSnake-R1 at initialization.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        expansion: float = 0.25,
        max_offset: float = 1.0,
        gamma_init: float = 0.0,
        scale_max: float = 2.5,
    ) -> None:
        super().__init__(
            channels=channels,
            kernel_size=kernel_size,
            expansion=expansion,
            max_offset=max_offset,
            gamma_init=gamma_init,
        )
        if scale_max <= 1.0:
            raise ValueError(f"scale_max must be greater than one, got {scale_max}")
        self.scale_max = float(scale_max)
        self.log_max_scale = math.log(self.scale_max)
        self.scale_head = nn.Conv2d(self.hidden, 2, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.scale_head.weight)
        nn.init.zeros_(self.scale_head.bias)

        self._diagnostics_enabled = False
        self._last_diagnostics: dict[str, torch.Tensor] = {}

    def set_diagnostics(self, enabled: bool = True) -> None:
        """Enable or disable detached read-only tensors from the most recent forward."""
        self._diagnostics_enabled = bool(enabled)
        if not enabled:
            self._last_diagnostics.clear()

    def diagnostics(self, *, clone: bool = True) -> dict[str, torch.Tensor]:
        """Return detached diagnostics without changing the normal forward graph."""
        if clone:
            return {name: value.clone() for name, value in self._last_diagnostics.items()}
        return dict(self._last_diagnostics)

    def _scale_logits(
        self, reduced: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return two scale logits and optional diagnostic-only auxiliary tensors."""
        return self.scale_head(reduced), {}

    def predict_scales(
        self, reduced: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Map unconstrained logits to the bounded reciprocal range [1/M, M]."""
        logits, extras = self._scale_logits(reduced)
        z_h, z_v = logits.chunk(2, dim=1)
        coefficient = torch.as_tensor(
            self.log_max_scale, device=logits.device, dtype=logits.dtype
        )
        scale_h = torch.exp(coefficient * z_h.tanh())
        scale_v = torch.exp(coefficient * z_v.tanh())
        return scale_h, scale_v, extras

    def sampling_coordinates(
        self,
        feature: torch.Tensor,
        orthogonal_offset: torch.Tensor,
        longitudinal_scale: torch.Tensor,
        *,
        horizontal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return pixel-space x/y coordinates and inherited cumulative offsets."""
        batch, _, height, width = feature.shape
        dtype, device = feature.dtype, feature.device
        expected = (batch, 1, height, width)
        if longitudinal_scale.shape != expected:
            raise ValueError(
                f"longitudinal_scale must have shape {expected}, got {tuple(longitudinal_scale.shape)}"
            )
        if orthogonal_offset.shape != (batch, self.kernel_size, height, width):
            raise ValueError(
                "orthogonal_offset must have shape "
                f"{(batch, self.kernel_size, height, width)}, got {tuple(orthogonal_offset.shape)}"
            )

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
        scaled_positions = positions * longitudinal_scale

        if horizontal:
            grid_x = base_x + scaled_positions
            grid_y = base_y + curved
        else:
            grid_x = base_x + curved
            grid_y = base_y + scaled_positions
        return grid_x, grid_y, curved

    def _sample_curve_with_scale(
        self,
        feature: torch.Tensor,
        orthogonal_offset: torch.Tensor,
        longitudinal_scale: torch.Tensor,
        *,
        horizontal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample and collapse one adaptive snake while retaining its coordinates."""
        batch, channels, height, width = feature.shape
        grid_x, grid_y, curved = self.sampling_coordinates(
            feature,
            orthogonal_offset,
            longitudinal_scale,
            horizontal=horizontal,
        )
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
        weight = self.horizontal_weight if horizontal else self.vertical_weight
        collapsed = (
            sampled
            * weight.to(device=feature.device, dtype=feature.dtype).view(
                1, channels, self.kernel_size, 1, 1
            )
        ).sum(dim=2)
        return collapsed, grid_x, grid_y, curved

    def _sample_curve(
        self,
        feature: torch.Tensor,
        orthogonal_offset: torch.Tensor,
        horizontal: bool,
        longitudinal_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Keep the R1 call contract while accepting an optional adaptive scale."""
        if longitudinal_scale is None:
            longitudinal_scale = feature.new_ones(
                (feature.shape[0], 1, feature.shape[2], feature.shape[3])
            )
        sampled, _, _, _ = self._sample_curve_with_scale(
            feature,
            orthogonal_offset,
            longitudinal_scale,
            horizontal=horizontal,
        )
        return sampled

    def _record_diagnostics(self, values: dict[str, Any]) -> None:
        """Retain detached tensors only when explicitly requested."""
        if not self._diagnostics_enabled:
            return
        self._last_diagnostics = {
            name: value.detach()
            for name, value in values.items()
            if isinstance(value, torch.Tensor)
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply R1 residual processing with adaptive main-axis spacing only."""
        reduced = self.reduce(x)
        offset_h, offset_v = self.offset(reduced).tanh().chunk(2, dim=1)
        scale_h, scale_v, scale_extras = self.predict_scales(reduced)
        horizontal_raw, horizontal_x, horizontal_y, cumulative_h = (
            self._sample_curve_with_scale(
                reduced, offset_h, scale_h, horizontal=True
            )
        )
        vertical_raw, vertical_x, vertical_y, cumulative_v = (
            self._sample_curve_with_scale(
                reduced, offset_v, scale_v, horizontal=False
            )
        )
        horizontal = self.act(self.horizontal_norm(horizontal_raw))
        vertical = self.act(self.vertical_norm(vertical_raw))
        local = self.local(reduced)
        residual = self.fuse(torch.cat((local, horizontal, vertical), dim=1))
        output = x + self.gamma.to(dtype=x.dtype) * residual
        self._record_diagnostics(
            {
                "scale_h": scale_h,
                "scale_v": scale_v,
                "offset_h": offset_h,
                "offset_v": offset_v,
                "cumulative_offset_h": cumulative_h,
                "cumulative_offset_v": cumulative_v,
                "horizontal_grid_x": horizontal_x,
                "horizontal_grid_y": horizontal_y,
                "vertical_grid_x": vertical_x,
                "vertical_grid_y": vertical_y,
                "residual": residual,
                "gamma": self.gamma,
                **scale_extras,
            }
        )
        return output


class MetricGuidedScaleAdaptiveRoadSnakeAdapter(ScaleAdaptiveRoadSnakeAdapter):
    """Condition SA-RS scale logits on detached local structure/edge proxy cues."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 5,
        expansion: float = 0.25,
        max_offset: float = 1.0,
        gamma_init: float = 0.0,
        scale_max: float = 2.5,
        cue_eps: float = 1e-6,
    ) -> None:
        super().__init__(
            channels=channels,
            kernel_size=kernel_size,
            expansion=expansion,
            max_offset=max_offset,
            gamma_init=gamma_init,
            scale_max=scale_max,
        )
        if cue_eps <= 0:
            raise ValueError(f"cue_eps must be positive, got {cue_eps}")
        self.cue_eps = float(cue_eps)
        self.scale_metric = nn.Conv2d(3, 2, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.scale_metric.weight)
        nn.init.zeros_(self.scale_metric.bias)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-1, -2).contiguous()
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def metric_cues(
        self, reduced: torch.Tensor, *, detach: bool = True
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Build normalized deviation and Sobel proxy cues without labels or boxes."""
        gray = reduced.mean(dim=1, keepdim=True)
        average = F.avg_pool2d(gray, kernel_size=3, stride=1, padding=1)
        deviation = (gray - average).abs()
        sobel_x = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sobel_y = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        edge_x = F.conv2d(gray, sobel_x, padding=1).abs()
        edge_y = F.conv2d(gray, sobel_y, padding=1).abs()

        normalized = []
        for cue in (deviation, edge_x, edge_y):
            denominator = cue.mean(dim=(-2, -1), keepdim=True).add(self.cue_eps)
            normalized.append(torch.tanh(cue / denominator))
        cues = torch.cat(normalized, dim=1)
        if detach:
            cues = cues.detach()
        return cues, {
            "metric_deviation": deviation,
            "metric_edge_x": edge_x,
            "metric_edge_y": edge_y,
            "metric_cues": cues,
        }

    def _scale_logits(
        self, reduced: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Add a zero-initialized metric residual to the inherited SA logits."""
        base_logits = self.scale_head(reduced)
        cues, extras = self.metric_cues(reduced, detach=True)
        metric_logits = self.scale_metric(cues)
        extras.update(
            {"scale_base_logits": base_logits, "scale_metric_logits": metric_logits}
        )
        return base_logits + metric_logits, extras


class ScaleAdaptiveRoadSnakeDetect(RoadSnakeDetect):
    """RoadSnake-R1 Detect head with scale-adaptive P4 longitudinal spacing."""

    def __init__(
        self,
        nc: int = 80,
        kernel_size: int = 5,
        expansion: float = 0.25,
        max_offset: float = 1.0,
        gamma_init: float = 0.0,
        scale_max: float = 2.5,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ) -> None:
        super().__init__(
            nc=nc,
            kernel_size=kernel_size,
            expansion=expansion,
            max_offset=max_offset,
            gamma_init=gamma_init,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
        )
        self.road_snake = ScaleAdaptiveRoadSnakeAdapter(
            channels=ch[1],
            kernel_size=kernel_size,
            expansion=expansion,
            max_offset=max_offset,
            gamma_init=gamma_init,
            scale_max=scale_max,
        )


class MetricGuidedScaleAdaptiveRoadSnakeDetect(ScaleAdaptiveRoadSnakeDetect):
    """SA-RS Detect head whose scale predictor also consumes detached metric cues."""

    def __init__(
        self,
        nc: int = 80,
        kernel_size: int = 5,
        expansion: float = 0.25,
        max_offset: float = 1.0,
        gamma_init: float = 0.0,
        scale_max: float = 2.5,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ) -> None:
        super().__init__(
            nc=nc,
            kernel_size=kernel_size,
            expansion=expansion,
            max_offset=max_offset,
            gamma_init=gamma_init,
            scale_max=scale_max,
            reg_max=reg_max,
            end2end=end2end,
            ch=ch,
        )
        self.road_snake = MetricGuidedScaleAdaptiveRoadSnakeAdapter(
            channels=ch[1],
            kernel_size=kernel_size,
            expansion=expansion,
            max_offset=max_offset,
            gamma_init=gamma_init,
            scale_max=scale_max,
        )
