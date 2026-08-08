"""Official StarNet-S1 backbone wrapper for YOLO26 detection."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.conv import Conv


class OfficialStarConvBN(nn.Sequential):
    """Exact ConvBN parameterization used by the official StarNet repository."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        with_bn: bool = True,
    ) -> None:
        super().__init__()
        self.add_module(
            "conv",
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, groups),
        )
        if with_bn:
            self.add_module("bn", nn.BatchNorm2d(out_planes))
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)


class OfficialStarBlock(nn.Module):
    """Exact StarNet block graph; S1 uses zero stochastic-depth probability."""

    def __init__(self, dim: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        hidden = mlp_ratio * dim
        self.dwconv = OfficialStarConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=True)
        self.f1 = OfficialStarConvBN(dim, hidden, 1, with_bn=False)
        self.f2 = OfficialStarConvBN(dim, hidden, 1, with_bn=False)
        self.g = OfficialStarConvBN(hidden, dim, 1, with_bn=True)
        self.dwconv2 = OfficialStarConvBN(dim, dim, 7, 1, 3, groups=dim, with_bn=False)
        self.act = nn.ReLU6()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.act(self.f1(x)) * self.f2(x)
        x = self.dwconv2(self.g(x))
        return residual + x


class OfficialStarNetS1BackboneYOLO(nn.Module):
    """Official StarNet-S1 stem/stages with post-backbone YOLO channel adapters.

    The official classification norm and head are intentionally absent. Stage
    1/2/3 outputs are stride 8/16/32 and adapters are the only new randomly
    initialized backbone-side parameters.
    """

    official_variant = "starnet_s1"
    base_dim = 24
    depths = (2, 2, 8, 3)
    mlp_ratio = 4

    def __init__(
        self,
        variant: str = "s1",
        out_channels: tuple[int, int, int] | list[int] = (256, 512, 1024),
        in_chans: int = 3,
    ) -> None:
        super().__init__()
        if str(variant).lower() not in {"s1", "starnet_s1"}:
            raise ValueError(f"Only the official pretrained StarNet-S1 graph is supported, got {variant}")
        if int(in_chans) != 3:
            raise ValueError("The official S1 checkpoint requires a three-channel input stem")
        self.out_channels = [int(channel) for channel in out_channels]
        self.in_channel = 32
        self.stem = nn.Sequential(
            OfficialStarConvBN(3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6(),
        )
        self.stages = nn.ModuleList()
        for level, depth in enumerate(self.depths):
            embed_dim = self.base_dim * 2**level
            downsample = OfficialStarConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [OfficialStarBlock(embed_dim, self.mlp_ratio) for _ in range(depth)]
            self.stages.append(nn.Sequential(downsample, *blocks))
        raw_channels = (self.base_dim * 2, self.base_dim * 4, self.base_dim * 8)
        self.adapters = nn.ModuleList(
            Conv(source, target, 1, 1) for source, target in zip(raw_channels, self.out_channels)
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        features = []
        for index, stage in enumerate(self.stages):
            x = stage(x)
            if index >= 1:
                features.append(x)
        return [adapter(feature) for adapter, feature in zip(self.adapters, features)]
