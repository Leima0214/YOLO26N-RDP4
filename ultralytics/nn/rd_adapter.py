"""Retriever-Dictionary adapter for the pretrained YOLO26 P3 stage."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv


class NormalizedDictionary(nn.Module):
    """A 1x1 dictionary whose atom columns stay unit-normalized."""

    def __init__(self, channels: int, atoms: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(channels, atoms, 1, 1))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def normalized_weight(self) -> torch.Tensor:
        return F.normalize(self.weight, dim=0, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.normalized_weight())


class RDAdapter(nn.Module):
    """Compact per-image feature retriever with a near-identity residual path."""

    def __init__(
        self,
        channels: int,
        atoms: int = 64,
        kernel_size: int = 5,
        eps: float = 1e-5,
        max_mix: float = 0.8,
    ):
        super().__init__()
        if channels < 1 or atoms < 2:
            raise ValueError(
                f"channels and atoms must be positive with atoms >= 2, got {channels=}, {atoms=}"
            )
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        if not 0.0 < max_mix <= 1.0:
            raise ValueError(f"max_mix must be in (0, 1], got {max_mix}")
        self.eps = float(eps)
        self.max_mix = float(max_mix)
        self.coefficient = Conv(channels, atoms, 1)
        self.exchange = Conv(atoms, atoms, kernel_size, g=atoms, act=False)
        self.pono_scale = nn.Parameter(torch.ones(1, atoms, 1, 1))
        self.pono_bias = nn.Parameter(torch.zeros(1, atoms, 1, 1))
        self.dictionary = NormalizedDictionary(channels, atoms)
        self.gamma = nn.Parameter(torch.tensor(1e-3))

    def pono(self, x: torch.Tensor) -> torch.Tensor:
        """Position-normalize across dictionary atoms in float32 for AMP stability."""
        stats = x.float()
        mean = stats.mean(dim=1, keepdim=True)
        variance = (stats - mean).square().mean(dim=1, keepdim=True)
        normalized = ((stats - mean) * torch.rsqrt(variance + self.eps)).to(
            dtype=x.dtype
        )
        return normalized * self.pono_scale + self.pono_bias

    def effective_mix(self) -> torch.Tensor:
        """Bound the learned retrieval mix by the 0.8 weight used in YOLO-RD."""
        return self.max_mix * self.gamma.tanh()

    @torch.no_grad()
    def initialize_atoms(self, atoms: torch.Tensor) -> None:
        """Initialize the retriever and dictionary from [atoms, channels] dataset centroids."""
        expected = (
            self.coefficient.conv.out_channels,
            self.coefficient.conv.in_channels,
        )
        if atoms.ndim != 2 or tuple(atoms.shape) != expected:
            raise ValueError(
                f"expected atom matrix {expected}, got {tuple(atoms.shape)}"
            )
        atoms = F.normalize(
            atoms.to(device=self.gamma.device, dtype=self.gamma.dtype), dim=1, eps=1e-6
        )
        self.coefficient.conv.weight.copy_(atoms[:, :, None, None])
        self.coefficient.bn.reset_running_stats()
        self.coefficient.bn.weight.fill_(1)
        self.coefficient.bn.bias.zero_()

        self.exchange.conv.weight.zero_()
        center = self.exchange.conv.kernel_size[0] // 2
        self.exchange.conv.weight[:, 0, center, center] = 1
        self.exchange.bn.reset_running_stats()
        self.exchange.bn.weight.fill_(1)
        self.exchange.bn.bias.zero_()

        self.dictionary.weight.copy_(atoms.t()[:, :, None, None])
        self.pono_scale.fill_(1)
        self.pono_bias.zero_()
        self.gamma.fill_(1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"RDAdapter expects BCHW features, got shape {tuple(x.shape)}"
            )
        retrieved = self.dictionary(self.pono(self.exchange(self.coefficient(x))))
        # A 1e-3 gate keeps the pretrained path dominant while enabling first-batch RD gradients.
        return x + self.effective_mix() * (retrieved - x)


class RDStage(C3k2):
    """Original C3k2 followed by one RD adapter; inherited weights keep their names."""

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
        atoms: int = 64,
    ):
        super().__init__(c1, c2, n, c3k=c3k, e=e, attn=attn, g=g, shortcut=shortcut)
        self.rd = RDAdapter(c2, atoms=atoms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rd(super().forward(x))


class RDP3Stage(RDStage):
    """YOLO-RD's original B3/P3 placement adapted to YOLO26."""
