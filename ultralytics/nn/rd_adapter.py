"""Retriever-Dictionary adapter for a single pretrained YOLO26 P4 stage."""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C3k2
from ultralytics.nn.modules.conv import Conv


class RDAdapter(nn.Module):
    """Compact per-image feature retriever with an identity-initialized residual path."""

    def __init__(self, channels: int, atoms: int = 64, kernel_size: int = 5, eps: float = 1e-5):
        super().__init__()
        if channels < 1 or atoms < 2:
            raise ValueError(f"channels and atoms must be positive with atoms >= 2, got {channels=}, {atoms=}")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")
        self.eps = float(eps)
        self.coefficient = Conv(channels, atoms, 1)
        self.exchange = Conv(atoms, atoms, kernel_size, g=atoms, act=False)
        self.dictionary = Conv(atoms, channels, 1, act=False)
        self.gamma = nn.Parameter(torch.zeros(()))

    def pono(self, x: torch.Tensor) -> torch.Tensor:
        """Position-normalize across dictionary atoms in float32 for AMP stability."""
        stats = x.float()
        mean = stats.mean(dim=1, keepdim=True)
        variance = (stats - mean).square().mean(dim=1, keepdim=True)
        return ((stats - mean) * torch.rsqrt(variance + self.eps)).to(dtype=x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"RDAdapter expects BCHW features, got shape {tuple(x.shape)}")
        retrieved = self.dictionary(self.pono(self.exchange(self.coefficient(x))))
        # gamma=0 exactly preserves the pretrained P4 output at initialization.
        return x + self.gamma * (retrieved - x)


class RDP4Stage(C3k2):
    """Original C3k2 followed by one RD adapter; inherited weights keep their original names."""

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
