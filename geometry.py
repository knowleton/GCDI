from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch

Dist = Dict[str, torch.Tensor]


def clamp_sigma(sigma: torch.Tensor, sigma_min: float = 1e-3) -> torch.Tensor:
    if not math.isfinite(float(sigma_min)) or sigma_min <= 0:
        raise ValueError("1")
    return torch.clamp(sigma, min=float(sigma_min))


def distribution_features(dist: Dist, sigma_min: float = 1e-3) -> torch.Tensor:
    return torch.cat(
        [dist["mu"], torch.log(clamp_sigma(dist["sigma"], sigma_min))], dim=-1
    )


def _acosh1p_squared(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp_min(0.0)
    small = t < 1e-4
    local = t.clamp_max(1e-4)
    series = local * (2.0 + local * (-1.0 / 3.0 + local * (4.0 / 45.0 - local / 35.0)))
    safe = t.clamp_min(1e-4)
    exact = 4.0 * torch.asinh(torch.sqrt(safe / 2.0)).square()
    return torch.where(small, series, exact)


def _logmap_factor(t: torch.Tensor) -> torch.Tensor:
    t = t.clamp_min(0.0)
    local = t.clamp_max(1e-4)
    series = 1.0 + local * (-1.0 / 3.0 + local * (2.0 / 15.0 - 2.0 * local / 35.0))
    safe = t.clamp_min(1e-4)
    exact = 2.0 * torch.asinh(torch.sqrt(safe / 2.0)) / torch.sqrt(safe * (safe + 2.0))
    return torch.where(t < 1e-4, series, exact)


def fr_distance_squared(p: Dist, q: Dist, sigma_min: float = 1e-3) -> torch.Tensor:
    values = [p["mu"], p["sigma"], q["mu"], q["sigma"]]
    if any(not value.is_floating_point() for value in values):
        raise TypeError("2")
    dtype = (
        torch.float64
        if any(value.dtype == torch.float64 for value in values)
        else torch.float32
    )
    mu1, s1, mu2, s2 = torch.broadcast_tensors(
        *(value.to(dtype=dtype) for value in values)
    )
    s1, s2 = clamp_sigma(s1, sigma_min), clamp_sigma(s2, sigma_min)
    t = ((mu1 - mu2).square() + 2.0 * (s1 - s2).square()) / (4.0 * s1 * s2)
    return (2.0 * _acosh1p_squared(t)).sum(dim=-1)


def sample_group_params(
    k: int,
    batch_size: int,
    dim: int,
    device,
    dtype: torch.dtype,
    scale_std: float,
    shift_std: float,
    *,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k < 1 or batch_size < 1 or dim < 1:
        raise ValueError("3")
    if (
        not (math.isfinite(scale_std) and math.isfinite(shift_std))
        or min(scale_std, shift_std) < 0
    ):
        raise ValueError(
            "4"
        )
    shape = (int(k), int(batch_size), int(dim))
    scale = torch.exp(
        torch.randn(shape, device=device, dtype=dtype, generator=generator)
        * float(scale_std)
    )
    shift = torch.randn(shape, device=device, dtype=dtype, generator=generator) * float(
        shift_std
    )
    scale[0].fill_(1.0)
    shift[0].zero_()
    return scale, shift


def _validate_action(dist: Dist, scale: torch.Tensor, sigma_min: float) -> None:
    if not math.isfinite(float(sigma_min)) or sigma_min <= 0:
        raise ValueError("5")
    if bool((~torch.isfinite(scale) | (scale <= 0)).any()):
        raise ValueError("6")
    if bool((~torch.isfinite(dist["sigma"]) | (dist["sigma"] <= 0)).any()):
        raise ValueError("7")


def apply_group(
    dist: Dist, scale: torch.Tensor, shift: torch.Tensor, sigma_min: float = 1e-3
) -> Dist:
    _validate_action(dist, scale, sigma_min)
    return {"mu": dist["mu"] * scale + shift, "sigma": dist["sigma"] * scale}


def inverse_group(
    dist: Dist, scale: torch.Tensor, shift: torch.Tensor, sigma_min: float = 1e-3
) -> Dist:
    _validate_action(dist, scale, sigma_min)
    return {"mu": (dist["mu"] - shift) / scale, "sigma": dist["sigma"] / scale}


def _component_objective(x, y, candidate_x, candidate_y, weights):
    t = (candidate_x - x.unsqueeze(0)).square() + (
        candidate_y - y.unsqueeze(0)
    ).square()
    t = t / (2.0 * candidate_y * y.unsqueeze(0))
    return (weights.unsqueeze(-1) * (2.0 * _acosh1p_squared(t))).sum(dim=0)


def _mean_log_at_point(x, y, candidate_x, candidate_y, weights):
    u = (candidate_x - x.unsqueeze(0)) / y.unsqueeze(0)
    v = candidate_y / y.unsqueeze(0)
    t = (u.square() + (v - 1.0).square()) / (2.0 * v)
    factor = _logmap_factor(t)
    log_x = factor * u / v
    log_y = factor * (u.square() + v.square() - 1.0) / (2.0 * v)
    w = weights.unsqueeze(-1)
    return (w * log_x).sum(dim=0), (w * log_y).sum(dim=0)


def _exp_at_point(x, y, tangent_x, tangent_y):
    norm2 = tangent_x.square() + tangent_y.square()
    safe_r = torch.sqrt(norm2.clamp_min(1e-12))
    sinhc = torch.where(
        norm2 < 1e-8,
        1.0 + norm2 / 6.0 + norm2.square() / 120.0,
        torch.sinh(safe_r) / safe_r,
    )
    cosh = torch.where(
        norm2 < 1e-8, 1.0 + norm2 / 2.0 + norm2.square() / 24.0, torch.cosh(safe_r)
    )
    denominator = cosh - sinhc * tangent_y
    return x + y * (sinhc * tangent_x / denominator), y / denominator


def barycenter_diag(
    candidates: Dist,
    weights: Optional[torch.Tensor] = None,
    sigma_min: float = 1e-3,
    max_steps: int = 64,
    tol: float = 1e-6,
) -> Dist:
    mu, sigma = candidates["mu"], candidates["sigma"]
    if mu.ndim != 3 or sigma.shape != mu.shape or min(mu.shape) < 1:
        raise ValueError("8")
    if not (mu.is_floating_point() and sigma.is_floating_point()):
        raise TypeError("9")
    if mu.device != sigma.device:
        raise ValueError("10")
    if bool((~torch.isfinite(mu) | ~torch.isfinite(sigma)).any()):
        raise ValueError("11")
    if (
        int(max_steps) != max_steps
        or max_steps < 1
        or not math.isfinite(float(tol))
        or tol <= 0
    ):
        raise ValueError("12")
    k, batch_size, _ = mu.shape
    original_dtype = mu.dtype
    m = mu.to(dtype=torch.float64)
    s = clamp_sigma(sigma.to(dtype=torch.float64), sigma_min)
    if weights is None:
        w = torch.full((k, batch_size), 1.0 / k, device=mu.device, dtype=torch.float64)
    else:
        if weights.shape != (k, batch_size) or weights.device != mu.device:
            raise ValueError("13")
        w = weights.to(dtype=torch.float64)
        if bool((~torch.isfinite(w) | (w < 0)).any()) or bool(
            (w.sum(dim=0) <= 0).any()
        ):
            raise ValueError(
                "14"
            )
        w = w / w.sum(dim=0, keepdim=True)
    candidate_x = m / math.sqrt(2.0)
    candidate_y = s
    x = (w.unsqueeze(-1) * candidate_x).sum(dim=0)
    y = torch.exp((w.unsqueeze(-1) * torch.log(candidate_y)).sum(dim=0))
    eps = torch.finfo(torch.float64).eps
    for _ in range(int(max_steps)):
        tangent_x, tangent_y = _mean_log_at_point(x, y, candidate_x, candidate_y, w)
        norm2 = tangent_x.square() + tangent_y.square()
        active = norm2 > float(tol) ** 2
        if not bool(active.any()):
            break
        objective = _component_objective(x, y, candidate_x, candidate_y, w)
        step = torch.minimum(
            torch.ones_like(norm2), 4.0 / torch.sqrt(norm2.clamp_min(1e-30))
        )
        step = torch.where(active, step, torch.zeros_like(step))
        next_x, next_y = x, y
        pending, moved = active, torch.zeros_like(active)
        for _backtrack in range(32):
            trial_x, trial_y = _exp_at_point(x, y, step * tangent_x, step * tangent_y)
            trial_objective = _component_objective(
                trial_x, trial_y, candidate_x, candidate_y, w
            )
            armijo = objective - 4e-4 * step * norm2
            numerical_slack = (
                16.0 * eps * torch.maximum(torch.ones_like(objective), objective.abs())
            )
            acceptable = (
                torch.isfinite(trial_objective)
                & (trial_y >= float(sigma_min))
                & (trial_objective <= armijo + numerical_slack)
            )
            take = pending & acceptable
            next_x = torch.where(take, trial_x, next_x)
            next_y = torch.where(take, trial_y, next_y)
            moved = moved | take
            pending = pending & ~take
            if not bool(pending.any()):
                break
            step = torch.where(pending, step * 0.5, step)
        if not bool(moved.any()):
            break
        x, y = next_x, next_y
    return {
        "mu": (math.sqrt(2.0) * x).to(dtype=original_dtype),
        "sigma": clamp_sigma(y, sigma_min).to(dtype=original_dtype),
    }
