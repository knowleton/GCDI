from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .geometry import (
    Dist,
    apply_group,
    barycenter_diag,
    fr_distance_squared,
    inverse_group,
    sample_group_params,
)


def build_orbit_candidates(
    transporter: Callable[[Dist], Dist],
    source: Dist,
    num_candidates: int = 6,
    scale_std: float = 0.04,
    shift_std: float = 0.04,
    sigma_min: float = 1e-3,
    *,
    generator: Optional[torch.Generator] = None,
) -> Dist:
    if (
        not isinstance(num_candidates, int)
        or isinstance(num_candidates, bool)
        or num_candidates < 1
    ):
        raise ValueError("15")
    if source["mu"].ndim != 2 or source["sigma"].shape != source["mu"].shape:
        raise ValueError("16")
    b, d = source["mu"].shape
    k = num_candidates
    scale, shift = sample_group_params(
        k,
        b,
        d,
        device=source["mu"].device,
        dtype=source["mu"].dtype,
        scale_std=scale_std,
        shift_std=shift_std,
        generator=generator,
    )
    expanded = {
        key: source[key].unsqueeze(0).expand(k, b, d).reshape(k * b, d)
        for key in ("mu", "sigma")
    }
    flat_scale = scale.reshape(k * b, d)
    flat_shift = shift.reshape(k * b, d)
    transformed = apply_group(expanded, flat_scale, flat_shift, sigma_min=sigma_min)
    transported = transporter(transformed)
    aligned = inverse_group(transported, flat_scale, flat_shift, sigma_min=sigma_min)
    return {key: aligned[key].reshape(k, b, d) for key in ("mu", "sigma")}


def candidate_scores(
    squared_deviation: torch.Tensor,
    gamma: float = 1.0,
    tau: float = 4.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if squared_deviation.ndim != 2 or 0 in squared_deviation.shape:
        raise ValueError("17")
    if not math.isfinite(float(gamma)) or gamma < 0:
        raise ValueError("18")
    if not math.isfinite(float(tau)) or tau < 0:
        raise ValueError("19")
    delta = squared_deviation.mean(dim=0)
    reliability = torch.exp(-float(gamma) * delta).clamp(0.0, 1.0)
    weights = F.softmax(-float(tau) * squared_deviation, dim=0)
    return reliability, weights, delta


def infer_missing_distribution(
    transport: Callable[[str, str, Dist], Dist],
    source_dists: Mapping[str, Dist],
    source_names: Sequence[str],
    target: str,
    *,
    num_orbit_candidates: int = 6,
    orbit_scale_std: float = 0.04,
    orbit_shift_std: float = 0.04,
    gamma: float = 1.0,
    tau: float = 4.0,
    sigma_min: float = 1e-3,
    barycenter_max_steps: int = 64,
    barycenter_tol: float = 1e-6,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dist, torch.Tensor, dict]:
    sources = tuple(source_names)
    if not sources or len(set(sources)) != len(sources) or target in sources:
        raise ValueError(
            "20"
        )
    by_source = []
    shape = None
    for source_name in sources:
        if source_name not in source_dists:
            raise KeyError("21")
        current = source_dists[source_name]
        if shape is None:
            shape = current["mu"].shape
        elif current["mu"].shape != shape:
            raise ValueError("22")
        by_source.append(
            build_orbit_candidates(
                lambda dist, src=source_name: transport(src, target, dist),
                current,
                num_candidates=num_orbit_candidates,
                scale_std=orbit_scale_std,
                shift_std=orbit_shift_std,
                sigma_min=sigma_min,
                generator=generator,
            )
        )
    candidates = {
        key: torch.cat([group[key] for group in by_source], dim=0)
        for key in ("mu", "sigma")
    }
    reference = barycenter_diag(
        candidates,
        weights=None,
        sigma_min=sigma_min,
        max_steps=barycenter_max_steps,
        tol=barycenter_tol,
    )
    squared_deviation = fr_distance_squared(candidates, reference, sigma_min=sigma_min)
    reliability, weights, delta = candidate_scores(
        squared_deviation, gamma=gamma, tau=tau
    )
    estimate = barycenter_diag(
        candidates,
        weights=weights,
        sigma_min=sigma_min,
        max_steps=barycenter_max_steps,
        tol=barycenter_tol,
    )
    diagnostics = {
        "sources": sources,
        "candidate_sources": tuple(
            src for src in sources for _ in range(num_orbit_candidates)
        ),
        "candidate_action_indices": tuple(
            i for _ in sources for i in range(num_orbit_candidates)
        ),
        "candidates": candidates,
        "reference_center": reference,
        "squared_deviation": squared_deviation,
        "dispersion": delta,
        "weights": weights,
        "reliability": reliability,
        "estimate": estimate,
    }
    return estimate, reliability, diagnostics
