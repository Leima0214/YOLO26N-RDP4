"""Japan4-specific YOLO26 adapters with pretrained-compatible residual paths."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.Dysample import DySample
from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.head import Detect


class QualityAwareDetect(Detect):
    """O2O Detect head with a detached, class-agnostic localization-quality scorer."""

    quality_aware = True

    def __init__(
        self,
        nc: int = 80,
        quality_loss_gain: float = 0.25,
        quality_init: float = 0.99,
        reg_max: int = 16,
        end2end: bool = False,
        ch: tuple = (),
    ):
        if not end2end:
            raise ValueError("QualityAwareDetect requires end2end=True")
        if quality_loss_gain < 0:
            raise ValueError(f"quality_loss_gain must be non-negative, got {quality_loss_gain}")
        if not 0 < quality_init < 1:
            raise ValueError(f"quality_init must be in (0, 1), got {quality_init}")
        super().__init__(nc, reg_max, end2end, ch)
        self.quality_loss_gain = float(quality_loss_gain)
        self.quality_init = float(quality_init)
        self.one2one_quality = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in ch)

    @property
    def one2one(self):
        """Return O2O box, class, and quality heads."""
        heads = super().one2one
        heads["quality_head"] = self.one2one_quality
        return heads

    def forward_head(self, x, box_head=None, cls_head=None, quality_head=None):
        """Run the unchanged Detect heads and optionally append one quality logit per location."""
        preds = super().forward_head(x, box_head, cls_head)
        if quality_head is not None:
            bs = x[0].shape[0]
            preds["quality"] = torch.cat(
                [quality_head[i](x[i]).view(bs, 1, -1) for i in range(self.nl)],
                dim=-1,
            )
        return preds

    def _inference(self, x):
        """Calibrate O2O class scores by predicted localization quality before top-k selection."""
        predictions = super()._inference(x)
        quality = x.get("quality")
        if quality is not None:
            predictions[:, 4:] *= quality.sigmoid()
        return predictions

    def bias_init(self):
        """Keep initial ranking near B0 while preserving first-batch quality gradients."""
        super().bias_init()
        bias = torch.logit(torch.tensor(self.quality_init)).item()
        for quality_head in self.one2one_quality:
            nn.init.zeros_(quality_head.weight)
            nn.init.constant_(quality_head.bias, bias)


class OCEC3k2(C3k2):
    """Orientation-aware crack enhancement after a pretrained C3k2 stage.

    The horizontal, vertical and local branches target D10, D00 and
    D20/D40-like structures respectively. Spatial soft routing lets every
    location choose a mixture instead of imposing one orientation globally.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        kernel: int = 7,
        residual_scale: float = 1e-3,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError(f"OCE kernel must be odd and >= 3, got {kernel}")
        padding = kernel // 2
        self.oce_local = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.oce_horizontal = nn.Sequential(
            nn.Conv2d(c2, c2, (1, kernel), padding=(0, padding), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.oce_vertical = nn.Sequential(
            nn.Conv2d(c2, c2, (kernel, 1), padding=(padding, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.oce_router = nn.Conv2d(c2, 3, 1, bias=True)
        self.oce_fuse = nn.Sequential(
            nn.Conv2d(c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.oce_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        weights = self.oce_router(y).softmax(1)
        branches = torch.stack(
            (self.oce_local(y), self.oce_horizontal(y), self.oce_vertical(y)),
            dim=1,
        )
        mixed = (branches * weights.unsqueeze(2)).sum(1)
        return y + self.oce_scale * self.oce_fuse(mixed)


class MogaC3k2(C3k2):
    """Pretrained-compatible C3k2 with a small MogaNet spatial-gating residual."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        g: int = 1,
        shortcut: bool = True,
        residual_scale: float = 1e-3,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        if c2 % 8:
            raise ValueError(f"MogaC3k2 requires output channels divisible by 8, got {c2}")
        # Lazy import keeps unrelated YOLO variants free of MogaNet's optional dependencies.
        from ultralytics.nn.moganet import MultiOrderGatedAggregation

        self.moga_aggregation = MultiOrderGatedAggregation(c2)
        self.moga_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        enhanced = self.moga_aggregation(y)
        return y + self.moga_scale * (enhanced - y)


class GatedDySample(nn.Module):
    """DySample upsampling introduced as a small residual over nearest."""

    def __init__(
        self,
        channels: int,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = True,
        residual_scale: float = 1e-3,
    ):
        super().__init__()
        self.scale = scale
        self.dysample = DySample(channels, scale=scale, style=style, groups=groups, dyscope=dyscope)
        self.dysample_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nearest = F.interpolate(x, scale_factor=self.scale, mode="nearest")
        adaptive = self.dysample(x)
        return nearest + self.dysample_scale * (adaptive - nearest)


class P4GuidedDySampleConcat(nn.Module):
    """P4-guided DySample residual fusion with relative spatial modulation."""

    def __init__(
        self,
        channels: list[int],
        dimension: int = 1,
        hidden_channels: int = 32,
        scale: int = 2,
        style: str = "lp",
        groups: int = 4,
        dyscope: bool = True,
        residual_scale: float = 1e-3,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"P4GuidedDySampleConcat expects [high_res, low_res], got {channels}")
        high_channels, low_channels = channels
        hidden = max(min(int(hidden_channels), high_channels, low_channels), 8)
        self.dimension = dimension
        self.scale = scale
        self.dysample = DySample(low_channels, scale=scale, style=style, groups=groups, dyscope=dyscope)
        self.guide_high = nn.Conv2d(high_channels, hidden, 1, bias=False)
        self.guide_low = nn.Conv2d(low_channels, hidden, 1, bias=False)
        self.guide_delta = nn.Conv2d(low_channels, hidden, 1, bias=False)
        self.compatibility_gate = nn.Sequential(
            nn.Conv2d(hidden * 4 + 2, hidden, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
        )
        nn.init.normal_(self.compatibility_gate[-1].weight, std=1e-3)
        nn.init.zeros_(self.compatibility_gate[-1].bias)
        self.dysample_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError(f"P4GuidedDySampleConcat expects two inputs, got {len(inputs)}")
        high_res, low_res = inputs
        nearest = F.interpolate(low_res, size=high_res.shape[-2:], mode="nearest")
        adaptive = self.dysample(low_res)
        if adaptive.shape[-2:] != high_res.shape[-2:]:
            raise ValueError(
                "P4GuidedDySampleConcat requires an exact 2x pyramid, "
                f"got high={tuple(high_res.shape[-2:])}, low={tuple(low_res.shape[-2:])}"
            )
        delta = adaptive - nearest
        high_guide = self.guide_high(high_res)
        low_guide = self.guide_low(nearest)
        delta_guide = self.guide_delta(delta)
        similarity = F.cosine_similarity(high_guide, low_guide, dim=1, eps=1e-6).unsqueeze(1)
        residual_strength = delta.abs().mean(1, keepdim=True)
        modulation = 2.0 * self.compatibility_gate(
            torch.cat(
                (
                    high_guide,
                    low_guide,
                    (high_guide - low_guide).abs(),
                    delta_guide,
                    similarity,
                    residual_strength,
                ),
                dim=1,
            )
        ).sigmoid()
        low_output = nearest + self.dysample_scale.to(dtype=nearest.dtype) * modulation * delta
        return torch.cat((low_output, high_res), self.dimension)


class GatedSPDDown(Conv):
    """Stride-2 Conv with a gated space-to-depth alternative path."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 2,
        residual_scale: float = 1e-3,
    ):
        if s != 2:
            raise ValueError(f"GatedSPDDown requires stride 2, got {s}")
        super().__init__(c1, c2, k, s)
        self.spd_path = Conv(c1 * 4, c2, 1, 1)
        self.spd_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    @staticmethod
    def _space_to_depth(x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] % 2 or x.shape[-1] % 2:
            raise ValueError(f"GatedSPDDown requires even spatial dimensions, got {tuple(x.shape[-2:])}")
        return F.pixel_unshuffle(x, 2)

    def _blend(self, baseline: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        adaptive = self.spd_path(self._space_to_depth(x))
        return baseline + self.spd_scale * (adaptive - baseline)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._blend(super().forward(x), x)

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self._blend(super().forward_fuse(x), x)


class FreqFusionConcat(nn.Module):
    """Single-node lightweight FreqFusion adaptation for YOLO26.

    The module combines content-guided low-pass filtering, high-pass skip
    enhancement and learned spatial resampling. A small residual gate preserves
    the pretrained nearest-plus-concat path at initialization.
    """

    def __init__(
        self,
        channels: list[int],
        dimension: int = 1,
        compressed_channels: int = 32,
        residual_scale: float = 1e-3,
        max_offset: float = 1.0,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"FreqFusionConcat expects [high_res, low_res], got {channels}")
        high_channels, low_channels = channels
        hidden = max(min(int(compressed_channels), high_channels, low_channels), 8)
        self.dimension = dimension
        self.max_offset = float(max_offset)
        self.freq_hr_compress = nn.Conv2d(high_channels, hidden, 1, bias=False)
        self.freq_lr_compress = nn.Conv2d(low_channels, hidden, 1, bias=False)
        self.freq_encoder = nn.Sequential(
            Conv(hidden * 2, hidden, 3, 1),
            nn.Conv2d(hidden, 4, 1, bias=True),
        )
        self.freq_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    @staticmethod
    def _resample(x: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        y = (torch.arange(height, device=x.device, dtype=x.dtype) + 0.5) * (2.0 / height) - 1.0
        x_coord = (torch.arange(width, device=x.device, dtype=x.dtype) + 0.5) * (2.0 / width) - 1.0
        grid_y, grid_x = torch.meshgrid(y, x_coord, indexing="ij")
        base_grid = torch.stack((grid_x, grid_y), -1).unsqueeze(0).expand(batch, -1, -1, -1)
        normalized_offset = torch.stack(
            (offset[:, 0] * (2.0 / width), offset[:, 1] * (2.0 / height)),
            dim=-1,
        )
        return F.grid_sample(
            x,
            base_grid + normalized_offset,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != 2:
            raise ValueError(f"FreqFusionConcat expects two inputs, got {len(inputs)}")
        high_res, low_res = inputs
        low_nearest = F.interpolate(low_res, size=high_res.shape[-2:], mode="nearest")
        context = torch.cat(
            (self.freq_hr_compress(high_res), self.freq_lr_compress(low_nearest)),
            dim=1,
        )
        low_logit, high_logit, offset_x, offset_y = self.freq_encoder(context).split(1, 1)
        offset = torch.cat((offset_x, offset_y), 1).tanh() * self.max_offset
        aligned_low = self._resample(low_nearest, offset)
        low_pass = F.avg_pool2d(aligned_low, 5, stride=1, padding=2)
        low_candidate = aligned_low + low_logit.sigmoid() * (low_pass - aligned_low)
        high_detail = high_res - F.avg_pool2d(high_res, 3, stride=1, padding=1)
        high_candidate = high_res + high_logit.sigmoid() * high_detail
        scale = self.freq_scale.to(dtype=high_res.dtype)
        low_output = low_nearest + scale * (low_candidate - low_nearest)
        high_output = high_res + scale * (high_candidate - high_res)
        return torch.cat((low_output, high_output), self.dimension)
