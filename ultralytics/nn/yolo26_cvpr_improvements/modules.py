"""Focused YOLO26 modules inspired by HVI, MSHC, StarNet and sMLP.

The classes keep Ultralytics YAML compatibility:
    module(c1, c2, *args) -> feature map with c2 channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv, DWConv


class HVIEnhanceStem(nn.Module):
    """Low-light enhancement stem inspired by HVI-CIDNet's HVI color decomposition."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2, expand: int = 2):
        super().__init__()
        hidden = max(c2 * expand, 16)
        self.rgb_stem = Conv(c1, c2, k, s)
        self.hvi_stem = Conv(3, c2, k, s)
        self.fuse = nn.Sequential(
            Conv(c2 * 2, hidden, 1, 1),
            DWConv(hidden, hidden, 3, 1),
            Conv(hidden, c2, 1, 1),
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, max(c2 // 4, 8), 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(max(c2 // 4, 8), c2, 1, bias=True),
            nn.Sigmoid(),
        )

    @staticmethod
    def _rgb_to_hvi(x: torch.Tensor) -> torch.Tensor:
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        maxc = x.max(1, keepdim=True).values
        minc = x.min(1, keepdim=True).values
        intensity = x.mean(1, keepdim=True)
        value = maxc
        saturation = (maxc - minc) / (maxc + 1e-6)
        warm_cool = (r - b) / (r + b + 1e-6)
        hue_vector = torch.cat((warm_cool, saturation, value), 1)
        return hue_vector * (0.5 + intensity)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rgb = self.rgb_stem(x)
        hvi = self.hvi_stem(self._rgb_to_hvi(x))
        y = self.fuse(torch.cat((rgb, hvi), 1))
        return y * (1.0 + self.gate(y))


class MSHCBlock(nn.Module):
    """Lightweight multi-scale spatial heterogeneous convolution block."""

    def __init__(self, c1: int, c2: int, kernels: tuple[int, ...] = (3, 5, 7), expansion: float = 0.5):
        super().__init__()
        hidden = max(int(c2 * expansion), 16)
        self.proj = Conv(c1, c2, 1, 1)
        self.reduce = Conv(c2, hidden, 1, 1)
        self.square = nn.ModuleList(DWConv(hidden, hidden, k, 1) for k in kernels)
        self.horizontal = nn.Conv2d(hidden, hidden, (1, 7), padding=(0, 3), groups=hidden, bias=False)
        self.vertical = nn.Conv2d(hidden, hidden, (7, 1), padding=(3, 0), groups=hidden, bias=False)
        self.fuse = Conv(hidden * (len(kernels) + 2), c2, 1, 1)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c2, c2, 1, bias=True), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        h = self.reduce(x)
        feats = [branch(h) for branch in self.square]
        feats.extend((self.horizontal(h), self.vertical(h)))
        y = self.fuse(torch.cat(feats, 1))
        return x + y * self.gate(y)


class StarStem(nn.Module):
    """StarNet-style P1/2 stem.

    The first version downsampled twice inside the stem, which is convenient for
    classification but too aggressive for detection. Keeping a P1/2 feature
    gives the later P3/P4/P5 maps a better low-level signal.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class StarDown(nn.Module):
    """StarNet-style downsampling projection."""

    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, k // 2, bias=False),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class StarBlock(nn.Module):
    """Detection-stable StarNet block.

    This follows the official StarNet block more closely: depthwise context,
    ReLU6-gated star multiplication, point projection, a second depthwise
    filter, then residual addition. A small residual scale keeps scratch
    detection training stable.
    """

    def __init__(self, c1: int, c2: int, mlp_ratio: float = 4.0, shortcut: bool = True, residual_scale: float = 0.1):
        super().__init__()
        hidden = max(int(c2 * mlp_ratio), 16)
        self.proj = Conv(c1, c2, 1, 1) if c1 != c2 else nn.Identity()
        self.dwconv = nn.Sequential(
            nn.Conv2d(c2, c2, 7, padding=3, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.f1 = nn.Conv2d(c2, hidden, 1, bias=False)
        self.f2 = nn.Conv2d(c2, hidden, 1, bias=False)
        self.act = nn.ReLU6(inplace=True)
        self.g = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.dwconv2 = nn.Conv2d(c2, c2, 7, padding=3, groups=c2, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(residual_scale)))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        y = self.dwconv(x)
        y = self.act(self.f1(y)) * self.f2(y)
        y = self.dwconv2(self.g(y))
        return x + self.scale * y if self.add else y


class SOMC3k2(C3k2):
    """C3k2 with a StarNet-style SOM residual used at backbone P3/P4.

    Inheriting C3k2 preserves every original parameter name so yolo26n.pt can
    initialize the complete baseline path before the new residual learns.
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
        star_ratio: float = 3.0,
        residual_scale: float = 1e-3,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        hidden = max(int(c2 * star_ratio), 16)
        self.som_dw1 = nn.Sequential(
            nn.Conv2d(c2, c2, 7, padding=3, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.som_f1 = nn.Conv2d(c2, hidden, 1, bias=False)
        self.som_f2 = nn.Conv2d(c2, hidden, 1, bias=False)
        self.som_act = nn.ReLU6(inplace=True)
        self.som_reduce = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.Conv2d(c2, c2, 7, padding=3, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.som_scale = nn.Parameter(torch.tensor(float(residual_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        z = self.som_dw1(y)
        z = self.som_act(self.som_f1(z)) * self.som_f2(z)
        return y + self.som_scale * self.som_reduce(z)


class MAFConcat(nn.Module):
    """Multi-scale attentive fusion with identity-initialized ARM alignment.

    Each input receives a depthwise deformable alignment and a spatial gate.
    Softmax source gains are multiplied by the input count, making the initial
    output equivalent to ordinary concatenation while all new paths get
    gradients from the first batch.
    """

    def __init__(self, channels: list[int], dimension: int = 1, align: bool = True):
        super().__init__()
        if len(channels) < 2:
            raise ValueError("MAFConcat requires at least two inputs")
        self.dimension = dimension
        self.align = align
        self.source_logits = nn.Parameter(torch.zeros(len(channels)))
        if align:
            self.offsets = nn.ModuleList(nn.Conv2d(c, 18, 3, padding=1) for c in channels)
            self.deforms = nn.ModuleList(
                DeformConv2d(c, c, 3, padding=1, groups=c, bias=False) for c in channels
            )
            self.spatial_gates = nn.ModuleList(nn.Conv2d(c, 1, 1) for c in channels)
            self._initialize_identity()

    def _initialize_identity(self) -> None:
        for offset, deform, gate in zip(self.offsets, self.deforms, self.spatial_gates):
            nn.init.zeros_(offset.weight)
            nn.init.zeros_(offset.bias)
            nn.init.zeros_(deform.weight)
            with torch.no_grad():
                deform.weight[:, 0, 1, 1] = 1.0
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.source_logits.numel():
            raise ValueError(f"expected {self.source_logits.numel()} inputs, got {len(inputs)}")
        gains = self.source_logits.softmax(0) * len(inputs)
        outputs = []
        for index, feature in enumerate(inputs):
            if self.align:
                feature = self.deforms[index](feature, self.offsets[index](feature))
                feature = feature * (2.0 * self.spatial_gates[index](feature).sigmoid())
            outputs.append(feature * gains[index].to(dtype=feature.dtype))
        return torch.cat(outputs, self.dimension)


class WTCC3k2(C3k2):
    """C3k2 followed by one-level Haar wavelet convolution refinement.

    All four depthwise sub-band residual filters start at zero. The complete
    module therefore reproduces the pretrained C3k2 output at initialization,
    while its filters receive gradients immediately through a non-zero scale.
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
        residual_scale: float = 0.1,
    ):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self.wtc_filters = nn.ModuleList(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False) for _ in range(4)
        )
        self.wtc_band_logits = nn.Parameter(torch.zeros(4))
        self.wtc_scale = nn.Parameter(torch.tensor(float(residual_scale)))
        for layer in self.wtc_filters:
            nn.init.zeros_(layer.weight)

    @staticmethod
    def _haar(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x00, x01 = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
        x10, x11 = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
        return (
            (x00 + x01 + x10 + x11) * 0.5,
            (x00 - x01 + x10 - x11) * 0.5,
            (x00 + x01 - x10 - x11) * 0.5,
            (x00 - x01 - x10 + x11) * 0.5,
        )

    @staticmethod
    def _inverse_haar(bands: tuple[torch.Tensor, ...]) -> torch.Tensor:
        ll, lh, hl, hh = bands
        output = torch.empty(
            ll.shape[0], ll.shape[1], ll.shape[2] * 2, ll.shape[3] * 2,
            device=ll.device, dtype=ll.dtype,
        )
        output[..., 0::2, 0::2] = (ll + lh + hl + hh) * 0.5
        output[..., 0::2, 1::2] = (ll - lh + hl - hh) * 0.5
        output[..., 1::2, 0::2] = (ll + lh - hl - hh) * 0.5
        output[..., 1::2, 1::2] = (ll - lh - hl + hh) * 0.5
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        height, width = y.shape[-2:]
        padded = F.pad(y, (0, width % 2, 0, height % 2), mode="replicate")
        bands = self._haar(padded)
        gains = self.wtc_band_logits.softmax(0) * 4.0
        filtered = tuple(
            (band + layer(band)) * gains[index].to(dtype=band.dtype)
            for index, (layer, band) in enumerate(zip(self.wtc_filters, bands))
        )
        reconstructed = self._inverse_haar(filtered)[..., :height, :width]
        baseline_reconstruction = self._inverse_haar(bands)[..., :height, :width]
        return y + self.wtc_scale * (reconstructed - baseline_reconstruction)


class SMLPBlock(nn.Module):
    """Sparse/spatial MLP block using axial token mixing."""

    def __init__(self, c1: int, c2: int, expansion: float = 2.0):
        super().__init__()
        hidden = max(int(c2 * expansion), 16)
        self.proj = Conv(c1, c2, 1, 1)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(c2, hidden, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(hidden, c2, 1, bias=False),
        )
        self.h_mix = nn.Conv1d(c2, c2, 7, padding=3, groups=c2, bias=False)
        self.w_mix = nn.Conv1d(c2, c2, 7, padding=3, groups=c2, bias=False)
        self.norm = nn.BatchNorm2d(c2)
        self.out = Conv(c2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        b, c, h, w = x.shape
        h_token = x.mean(3)
        w_token = x.mean(2)
        h_gate = self.h_mix(h_token).sigmoid().view(b, c, h, 1)
        w_gate = self.w_mix(w_token).sigmoid().view(b, c, 1, w)
        y = x * (1.0 + h_gate + w_gate)
        y = self.channel_mlp(self.norm(y))
        return self.out(x + y)
