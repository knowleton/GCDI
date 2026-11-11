from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .geometry import Dist
from .inference import infer_missing_distribution
from .networks import GaussianHead, ReliabilityFusionHead, Tiny3DEncoder, Transporter


class GCDIModel(nn.Module):

    def __init__(
        self,
        modalities: Sequence[str],
        *,
        in_channels: int = 1,
        base_channels: int = 24,
        feature_dim: int = 256,
        latent_dim: int = 128,
        shared_encoder: bool = True,
        dropout: float = 0.25,
        transporter_hidden_dim: Optional[int] = None,
        transporter_depth: int = 2,
        fusion_hidden_dim: int = 256,
        sigma_min: float = 1e-3,
        num_orbit_candidates: int = 6,
        orbit_scale_std: float = 0.04,
        orbit_shift_std: float = 0.04,
        gamma: float = 1.0,
        tau: float = 4.0,
        barycenter_max_steps: int = 64,
        barycenter_tol: float = 1e-6,
    ) -> None:
        super().__init__()
        if isinstance(modalities, str):
            raise ValueError("23")
        names = tuple(modalities)
        if len(names) < 2 or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("24")
        if len(set(names)) != len(names):
            raise ValueError("25")
        if any(
            "." in name or hasattr(nn.ModuleDict, name) or name == "training"
            for name in names
        ):
            raise ValueError("26")
        for name, value in {
            "in_channels": in_channels,
            "base_channels": base_channels,
            "feature_dim": feature_dim,
            "latent_dim": latent_dim,
            "fusion_hidden_dim": fusion_hidden_dim,
            "num_orbit_candidates": num_orbit_candidates,
            "barycenter_max_steps": barycenter_max_steps,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("27")
        if fusion_hidden_dim < 2:
            raise ValueError("28")
        for name, value in {
            "sigma_min": sigma_min,
            "barycenter_tol": barycenter_tol,
        }.items():
            if not math.isfinite(float(value)) or value <= 0:
                raise ValueError("29")
        for name, value in {
            "gamma": gamma,
            "tau": tau,
            "orbit_scale_std": orbit_scale_std,
            "orbit_shift_std": orbit_shift_std,
        }.items():
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError("30")
        self.modalities = names
        self.mod_to_idx = {name: i for i, name in enumerate(names)}
        self.latent_dim = latent_dim
        self.sigma_min = float(sigma_min)
        self.num_orbit_candidates = num_orbit_candidates
        self.orbit_scale_std = float(orbit_scale_std)
        self.orbit_shift_std = float(orbit_shift_std)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.barycenter_max_steps = barycenter_max_steps
        self.barycenter_tol = float(barycenter_tol)
        encoder_args = dict(
            in_channels=in_channels,
            base_channels=base_channels,
            feature_dim=feature_dim,
            dropout=dropout,
        )
        self.shared_encoder = Tiny3DEncoder(**encoder_args) if shared_encoder else None
        self.encoders = nn.ModuleDict(
            {name: Tiny3DEncoder(**encoder_args) for name in names}
            if not shared_encoder
            else {}
        )
        self.gaussian_heads = nn.ModuleDict(
            {name: GaussianHead(feature_dim, latent_dim, sigma_min) for name in names}
        )
        self.transporters = nn.ModuleDict(
            {
                self._pair_key(src, tgt): Transporter(
                    latent_dim,
                    hidden_dim=transporter_hidden_dim,
                    dropout=dropout,
                    sigma_min=sigma_min,
                    depth=transporter_depth,
                )
                for src in names
                for tgt in names
                if src != tgt
            }
        )
        self.fusion = ReliabilityFusionHead(
            names, latent_dim, hidden_dim=fusion_hidden_dim, dropout=dropout
        )

    def _pair_key(self, source: str, target: str) -> str:
        return f"{self.mod_to_idx[source]}_to_{self.mod_to_idx[target]}"

    def _validate_available(self, available: torch.Tensor) -> None:
        if not isinstance(available, torch.Tensor) or available.dtype != torch.bool:
            raise TypeError("31")
        if (
            available.ndim != 2
            or available.shape[1] != len(self.modalities)
            or available.shape[0] == 0
        ):
            raise ValueError("32")
        if not bool(available.any(dim=1).all()):
            raise ValueError("33")

    def transport(self, source: str, target: str, dist: Dist) -> Dist:
        if (
            source == target
            or source not in self.mod_to_idx
            or target not in self.mod_to_idx
        ):
            raise ValueError(
                "34"
            )
        return self.transporters[self._pair_key(source, target)](dist)

    def encode_available(
        self, images: Mapping[str, torch.Tensor], available: torch.Tensor
    ) -> Dict[str, Dist]:
        self._validate_available(available)
        batch_size = available.shape[0]
        encoded: Dict[str, Dist] = {}
        for name in self.modalities:
            j = self.mod_to_idx[name]
            if not bool(available[:, j].any()):
                continue
            if name not in images:
                raise KeyError("35")
            image = images[name]
            if image.ndim != 5 or image.shape[0] != batch_size:
                raise ValueError("36")
            visible = available[:, j].to(device=image.device)
            index = visible.nonzero(as_tuple=False).squeeze(1)
            encoder = (
                self.shared_encoder
                if self.shared_encoder is not None
                else self.encoders[name]
            )
            observed_dist = self.gaussian_heads[name](
                encoder(image.index_select(0, index))
            )
            encoded[name] = {
                "mu": observed_dist["mu"]
                .new_zeros((batch_size, self.latent_dim))
                .index_copy(0, index, observed_dist["mu"]),
                "sigma": observed_dist["sigma"]
                .new_ones((batch_size, self.latent_dim))
                .index_copy(0, index, observed_dist["sigma"]),
            }
        template = next(iter(encoded.values()))["mu"]
        for name in self.modalities:
            if name not in encoded:
                encoded[name] = {
                    "mu": template.new_zeros((batch_size, self.latent_dim)),
                    "sigma": template.new_ones((batch_size, self.latent_dim)),
                }
        return {name: encoded[name] for name in self.modalities}

    def _prepare_distributions(
        self, dists: Mapping[str, Dist], available: torch.Tensor
    ) -> Tuple[Dict[str, Dist], torch.Tensor]:
        self._validate_available(available)
        batch_size = available.shape[0]
        template = None
        prepared: Dict[str, Dist] = {}
        for name in self.modalities:
            mask = available[:, self.mod_to_idx[name]]
            if not bool(mask.any()):
                continue
            if name not in dists:
                raise KeyError("37")
            dist = dists[name]
            if set(dist) != {"mu", "sigma"}:
                raise ValueError("38")
            mu, sigma = dist["mu"], dist["sigma"]
            if mu.shape != (batch_size, self.latent_dim) or sigma.shape != mu.shape:
                raise ValueError("39")
            if (
                not mu.is_floating_point()
                or sigma.dtype != mu.dtype
                or sigma.device != mu.device
            ):
                raise ValueError(
                    "40"
                )
            if template is not None and (
                mu.dtype != template.dtype or mu.device != template.device
            ):
                raise ValueError(
                    "41"
                )
            visible = mask.to(device=mu.device)
            observed_mu, observed_sigma = mu[visible], sigma[visible]
            if not bool(
                torch.isfinite(observed_mu).all()
                and torch.isfinite(observed_sigma).all()
            ):
                raise ValueError("42")
            if not bool((observed_sigma > 0).all()):
                raise ValueError("43")
            template = mu if template is None else template
            prepared[name] = {"mu": mu, "sigma": sigma}
        assert template is not None
        for name in self.modalities:
            if name not in prepared:
                prepared[name] = {
                    "mu": template.new_zeros((batch_size, self.latent_dim)),
                    "sigma": template.new_ones((batch_size, self.latent_dim)),
                }
        return {name: prepared[name] for name in self.modalities}, available.to(
            device=template.device
        )

    def infer_distributions(
        self,
        dists: Mapping[str, Dist],
        available: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Dict[str, Dist], dict]:
        sources, mask = self._prepare_distributions(dists, available)
        batch_size = mask.shape[0]
        template = next(iter(sources.values()))["mu"]
        completed: Dict[str, Dist] = {}
        reliability: Dict[str, torch.Tensor] = {}
        observed = {name: mask[:, j] for j, name in enumerate(self.modalities)}
        inferred: Dict[str, Dist] = {}
        candidate_records = {name: [] for name in self.modalities}
        for target in self.modalities:
            visible = observed[target]
            observed_index = visible.nonzero(as_tuple=False).squeeze(1)
            target_mu = template.new_zeros((batch_size, self.latent_dim))
            target_sigma = template.new_ones((batch_size, self.latent_dim))
            if observed_index.numel():
                target_mu = target_mu.index_copy(
                    0,
                    observed_index,
                    sources[target]["mu"].index_select(0, observed_index),
                )
                target_sigma = target_sigma.index_copy(
                    0,
                    observed_index,
                    sources[target]["sigma"].index_select(0, observed_index),
                )
            inferred_mu, inferred_sigma = template.new_zeros(
                target_mu.shape
            ), template.new_ones(target_sigma.shape)
            target_reliability = visible.to(dtype=template.dtype)
            missing_index = (~visible).nonzero(as_tuple=False).squeeze(1)
            if missing_index.numel():
                patterns, membership = torch.unique(
                    mask.index_select(0, missing_index), dim=0, return_inverse=True
                )
                for group, pattern in enumerate(patterns):
                    index = missing_index[membership == group]
                    source_names = tuple(
                        name
                        for j, name in enumerate(self.modalities)
                        if bool(pattern[j])
                    )
                    group_sources = {
                        name: {
                            key: sources[name][key].index_select(0, index)
                            for key in ("mu", "sigma")
                        }
                        for name in source_names
                    }
                    estimate, rel, detail = infer_missing_distribution(
                        self.transport,
                        group_sources,
                        source_names,
                        target,
                        num_orbit_candidates=self.num_orbit_candidates,
                        orbit_scale_std=self.orbit_scale_std,
                        orbit_shift_std=self.orbit_shift_std,
                        gamma=self.gamma,
                        tau=self.tau,
                        sigma_min=self.sigma_min,
                        barycenter_max_steps=self.barycenter_max_steps,
                        barycenter_tol=self.barycenter_tol,
                        generator=generator,
                    )
                    target_mu = target_mu.index_copy(0, index, estimate["mu"])
                    target_sigma = target_sigma.index_copy(0, index, estimate["sigma"])
                    inferred_mu = inferred_mu.index_copy(0, index, estimate["mu"])
                    inferred_sigma = inferred_sigma.index_copy(
                        0, index, estimate["sigma"]
                    )
                    target_reliability = target_reliability.index_copy(
                        0, index, rel.to(dtype=target_reliability.dtype)
                    )
                    candidate_records[target].append(
                        {"patient_indices": index, **detail}
                    )
            completed[target] = {"mu": target_mu, "sigma": target_sigma}
            inferred[target] = {"mu": inferred_mu, "sigma": inferred_sigma}
            reliability[target] = target_reliability
        diagnostics = {
            "source_dists": sources,
            "dists": completed,
            "inferred": inferred,
            "reliability": reliability,
            "observed": observed,
            "available": mask,
            "missing_masks": {name: ~observed[name] for name in self.modalities},
            "candidates": candidate_records,
            "candidate_records": candidate_records,
        }
        return completed, diagnostics

    def infer(
        self,
        dists: Mapping[str, Dist],
        available: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[Dict[str, Dist], dict]:
        return self.infer_distributions(dists, available, generator=generator)

    def forward_from_distributions(
        self,
        dists: Mapping[str, Dist],
        available: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, dict]:
        completed, diagnostics = self.infer_distributions(
            dists, available, generator=generator
        )
        logits = self.fusion(
            completed, diagnostics["reliability"], diagnostics["observed"]
        )
        return logits, diagnostics

    def classify(
        self,
        dists: Mapping[str, Dist],
        available: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, dict]:
        return self.forward_from_distributions(dists, available, generator=generator)

    def forward(
        self,
        images: Mapping[str, torch.Tensor],
        available: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, dict]:
        dists = self.encode_available(images, available)
        return self.forward_from_distributions(dists, available, generator=generator)
