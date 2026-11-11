from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .geometry import Dist, apply_group, fr_distance_squared, sample_group_params


@dataclass
class LossOutput:
    total: Tensor
    task: Tensor
    geo: Tensor
    equi: Tensor
    bar: Tensor
    warmup_factor: float
    diagnostics: dict[str, Any]


def binary_bce(logits: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    if logits.ndim == 2 and logits.shape[1] == 2:
        binary_logits = logits[:, 1] - logits[:, 0]
    elif logits.ndim == 2 and logits.shape[1] == 1:
        binary_logits = logits[:, 0]
    elif logits.ndim == 1:
        binary_logits = logits
    else:
        raise ValueError("44")
    target = labels.reshape(-1).to(device=logits.device, dtype=logits.dtype)
    if target.shape != binary_logits.shape:
        raise ValueError("45")
    return F.binary_cross_entropy_with_logits(
        binary_logits, target, reduction=reduction
    )


def sample_pseudo_missing(
    n_patients: int,
    n_modalities: int,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    if n_patients < 0 or n_modalities < 2:
        raise ValueError("46")
    hidden = torch.empty((n_patients, n_modalities), device=device, dtype=torch.bool)
    pending = torch.arange(n_patients, device=hidden.device)
    while pending.numel():
        proposed = torch.randint(
            2,
            (pending.numel(), n_modalities),
            device=hidden.device,
            generator=generator,
            dtype=torch.long,
        ).bool()
        valid = proposed.any(dim=1) & ~proposed.all(dim=1)
        hidden[pending[valid]] = proposed[valid]
        pending = pending[~valid]
    return hidden


def _select(dist: Mapping[str, Tensor], indices: Tensor) -> Dist:
    return {"mu": dist["mu"][indices], "sigma": dist["sigma"][indices]}


def _paired_transports(
    model: nn.Module, source: str, target: str, original: Dist, acted: Dist
) -> tuple[Dist, Dist]:
    cpu_before = torch.get_rng_state()
    device = original["mu"].device
    cuda_before = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    base = model.transport(source, target, original)
    cpu_after = torch.get_rng_state()
    cuda_after = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    try:
        torch.set_rng_state(cpu_before)
        if cuda_before is not None:
            torch.cuda.set_rng_state(cuda_before, device)
        perturbed = model.transport(source, target, acted)
    finally:
        torch.set_rng_state(cpu_after)
        if cuda_after is not None:
            torch.cuda.set_rng_state(cuda_after, device)
    return base, perturbed


class GCDIObjective(nn.Module):

    def __init__(
        self,
        *,
        eta_pm: float = 0.5,
        lambda_task: float = 1.0,
        lambda_geo: float = 0.03,
        lambda_equi: float = 0.005,
        lambda_bar: float = 0.03,
        warmup_epochs: int = 5,
    ) -> None:
        super().__init__()
        if warmup_epochs < 0:
            raise ValueError("47")
        self.eta_pm = float(eta_pm)
        self.lambda_task = float(lambda_task)
        self.lambda_geo = float(lambda_geo)
        self.lambda_equi = float(lambda_equi)
        self.lambda_bar = float(lambda_bar)
        self.warmup_epochs = int(warmup_epochs)

    def forward(
        self,
        model: nn.Module,
        images: Mapping[str, Tensor],
        available: Tensor,
        labels: Tensor,
        *,
        epoch: int = 0,
        generator: torch.Generator | None = None,
        pseudo_hidden: Tensor | None = None,
    ) -> LossOutput:
        dists = model.encode_available(images, available)
        return self.forward_from_distributions(
            model,
            dists,
            available,
            labels,
            epoch=epoch,
            generator=generator,
            pseudo_hidden=pseudo_hidden,
        )

    def forward_from_distributions(
        self,
        model: nn.Module,
        dists: Mapping[str, Dist],
        available: Tensor,
        labels: Tensor,
        *,
        epoch: int = 0,
        generator: torch.Generator | None = None,
        pseudo_hidden: Tensor | None = None,
    ) -> LossOutput:
        modalities = tuple(model.modalities)
        sigma_min = float(getattr(model, "sigma_min", 1e-3))
        if epoch < 0:
            raise ValueError("48")
        if available.dtype != torch.bool or available.ndim != 2:
            raise ValueError("49")
        batch, n_modalities = available.shape
        if batch == 0 or n_modalities != len(modalities) or n_modalities < 2:
            raise ValueError("50")
        if not bool(available.any(dim=1).all()):
            raise ValueError("51")
        labels = labels.reshape(-1).to(device=available.device)
        if labels.numel() != batch:
            raise ValueError("52")
        complete = available.all(dim=1)
        complete_indices = complete.nonzero(as_tuple=True)[0]
        n_complete = complete_indices.numel()
        actual_logits, actual_diag = model.forward_from_distributions(
            dists,
            available,
            generator=generator,
        )
        per_patient_bce = binary_bce(actual_logits, labels, reduction="none")
        accumulation_dtype = (
            torch.float64 if actual_logits.dtype == torch.float64 else torch.float32
        )
        zero = actual_logits.to(dtype=accumulation_dtype).sum() * 0.0
        full_task = per_patient_bce[complete].mean() if n_complete else zero
        incomplete_task = (
            per_patient_bce[~complete].mean() if bool((~complete).any()) else zero
        )
        pseudo_task, geo, bar = zero, zero, zero
        pseudo_diag = None
        if pseudo_hidden is not None:
            if (
                pseudo_hidden.dtype != torch.bool
                or pseudo_hidden.shape != (n_complete, n_modalities)
                or pseudo_hidden.device != available.device
            ):
                raise ValueError(
                    "53"
                )
            if not bool((pseudo_hidden.any(dim=1) & ~pseudo_hidden.all(dim=1)).all()):
                raise ValueError(
                    "54"
                )
        elif n_complete:
            pseudo_hidden = sample_pseudo_missing(
                n_complete,
                n_modalities,
                device=available.device,
                generator=generator,
            )
        else:
            pseudo_hidden = torch.empty(
                (0, n_modalities), dtype=torch.bool, device=available.device
            )

        if n_complete:
            complete_dists = {
                m: _select(dists[m], complete_indices) for m in modalities
            }
            pseudo_logits, pseudo_diag = model.forward_from_distributions(
                complete_dists,
                ~pseudo_hidden,
                generator=generator,
            )
            pseudo_task = binary_bce(pseudo_logits, labels[complete])
            geo_per_direction = []
            for source in modalities:
                for target in modalities:
                    if source != target:
                        transported = model.transport(
                            source, target, complete_dists[source]
                        )
                        geo_per_direction.append(
                            fr_distance_squared(
                                transported,
                                complete_dists[target],
                                sigma_min=sigma_min,
                            )
                        )
            geo = torch.stack(geo_per_direction, dim=1).mean()
            bar_per_patient = zero.expand(n_complete).clone()
            for target_index, target in enumerate(modalities):
                hidden_indices = pseudo_hidden[:, target_index].nonzero(as_tuple=True)[
                    0
                ]
                if hidden_indices.numel():
                    inferred = _select(pseudo_diag["inferred"][target], hidden_indices)
                    observed_target = _select(complete_dists[target], hidden_indices)
                    error = fr_distance_squared(
                        inferred, observed_target, sigma_min=sigma_min
                    )
                    bar_per_patient = bar_per_patient.index_add(
                        0, hidden_indices, error
                    )
            bar = (bar_per_patient / pseudo_hidden.sum(dim=1)).mean()

        task = full_task + self.eta_pm * pseudo_task + incomplete_task

        directions = torch.empty((batch, 2), dtype=torch.long, device=available.device)
        for patient in range(batch):
            observed = available[patient].nonzero(as_tuple=True)[0]
            source = observed[
                torch.randint(
                    observed.numel(), (), device=available.device, generator=generator
                )
            ]
            target = torch.randint(
                n_modalities - 1, (), device=available.device, generator=generator
            )
            target = target + (target >= source)
            directions[patient, 0], directions[patient, 1] = source, target
        equi_per_patient = zero.expand(batch).clone()
        for source_index, source in enumerate(modalities):
            for target_index, target in enumerate(modalities):
                if source == target:
                    continue
                indices = (
                    (directions[:, 0] == source_index)
                    & (directions[:, 1] == target_index)
                ).nonzero(as_tuple=True)[0]
                if not indices.numel():
                    continue
                source_dist = _select(dists[source], indices)
                scale, shift = sample_group_params(
                    2,
                    indices.numel(),
                    source_dist["mu"].shape[-1],
                    device=source_dist["mu"].device,
                    dtype=source_dist["mu"].dtype,
                    scale_std=model.orbit_scale_std,
                    shift_std=model.orbit_shift_std,
                    generator=generator,
                )
                acted = apply_group(
                    source_dist, scale[1], shift[1], sigma_min=sigma_min
                )
                base, perturbed = _paired_transports(
                    model, source, target, source_dist, acted
                )
                identity_error = fr_distance_squared(base, base, sigma_min=sigma_min)
                nonidentity_error = fr_distance_squared(
                    perturbed,
                    apply_group(base, scale[1], shift[1], sigma_min=sigma_min),
                    sigma_min=sigma_min,
                )
                equi_per_patient = equi_per_patient.index_add(
                    0,
                    indices,
                    0.5 * (identity_error + nonidentity_error),
                )
        equi = equi_per_patient.mean()
        warmup = (
            min(1.0, (epoch + 1) / self.warmup_epochs) if self.warmup_epochs else 1.0
        )
        total = self.lambda_task * task + warmup * (
            self.lambda_geo * geo + self.lambda_equi * equi + self.lambda_bar * bar
        )
        return LossOutput(
            total,
            task,
            geo,
            equi,
            bar,
            warmup,
            {
                "complete_indices": complete_indices,
                "pseudo_hidden": pseudo_hidden,
                "equivariance_directions": directions,
                "full_task": full_task,
                "pseudo_task": pseudo_task,
                "incomplete_task": incomplete_task,
                "actual": actual_diag,
                "pseudo": pseudo_diag,
            },
        )
