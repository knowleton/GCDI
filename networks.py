from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import Dist, distribution_features


def _group_count(channels: int) -> int:
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


class ConvBlock3D(nn.Module):
    def __init__(
        self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity(),
        )
        self.skip = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)
            if stride != 1 or in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class Tiny3DEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 24,
        feature_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            ConvBlock3D(in_channels, c, stride=1, dropout=0.0),
            ConvBlock3D(c, c * 2, stride=2, dropout=dropout * 0.5),
            ConvBlock3D(c * 2, c * 4, stride=2, dropout=dropout * 0.5),
            ConvBlock3D(c * 4, c * 8, stride=2, dropout=dropout),
            ConvBlock3D(c * 8, c * 8, stride=2, dropout=dropout),
            nn.AdaptiveAvgPool3d(1),
        )
        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 8, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_dim = int(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.net(x))


class GaussianHead(nn.Module):
    def __init__(
        self, feature_dim: int, latent_dim: int, sigma_min: float = 1e-3
    ) -> None:
        super().__init__()
        self.mu = nn.Linear(feature_dim, latent_dim)
        self.raw_sigma = nn.Linear(feature_dim, latent_dim)
        self.sigma_min = float(sigma_min)

    def forward(self, feat: torch.Tensor) -> Dist:
        mu = self.mu(feat)
        sigma = F.softplus(self.raw_sigma(feat)) + self.sigma_min
        return {"mu": mu, "sigma": sigma}


class Transporter(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        sigma_min: float = 1e-3,
        depth: int = 2,
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim or latent_dim * 2)
        self.sigma_min = float(sigma_min)
        blocks = [
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        ]
        for layer_idx in range(max(2, int(depth)) - 2):
            blocks.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            )
        blocks.append(nn.Linear(hidden_dim, latent_dim * 2))
        self.net = nn.Sequential(*blocks)

    def forward(self, dist: Dist) -> Dist:
        x = distribution_features(dist, sigma_min=self.sigma_min)
        delta_mu, delta_log_sigma = self.net(x).chunk(2, dim=-1)
        log_sigma = torch.log(torch.clamp(dist["sigma"], min=self.sigma_min))
        return {
            "mu": dist["mu"] + delta_mu,
            "sigma": torch.exp(log_sigma + 0.5 * torch.tanh(delta_log_sigma)).clamp(
                min=self.sigma_min
            ),
        }


class ReliabilityFusionHead(nn.Module):
    def __init__(
        self,
        modalities: Sequence[str],
        latent_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.modalities = list(modalities)
        self.proj = nn.ModuleDict(
            {
                m: nn.Sequential(
                    nn.Linear(latent_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for m in self.modalities
            }
        )
        in_dim = hidden_dim * len(self.modalities) + 2 * len(self.modalities)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(
        self,
        dists: Dict[str, Dist],
        reliability: Dict[str, torch.Tensor],
        observed: Dict[str, torch.Tensor],
        use_reliability: bool = True,
    ) -> torch.Tensor:
        parts = []
        aux = []
        for m in self.modalities:
            z = self.proj[m](distribution_features(dists[m]))
            r = reliability[m].to(dtype=z.dtype).unsqueeze(-1)
            if not use_reliability:
                r = torch.ones_like(r)
            parts.append(z * r)
            aux.append(r)
            aux.append(observed[m].to(dtype=z.dtype).unsqueeze(-1))
        return self.classifier(torch.cat(parts + aux, dim=-1))
