"""Load and validate PISM cloud configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import yaml

from .aws import instance_spec
from .cost import estimate_total_cost

DEFAULT_CPU_INSTANCE_TYPES = ["c7i.4xlarge", "hpc6a.48xlarge"]
DEFAULT_GPU_INSTANCE_TYPES = ["g5.xlarge"]
INSTANCE_ALIASES = {"c7i": "c7i.4xlarge", "hpc6a": "hpc6a.48xlarge", "g5": "g5.xlarge"}


@dataclass(frozen=True)
class NormalizedConfig:
    run_id: str
    ensemble_members: int
    compute: Dict[str, Any]
    cost: Dict[str, Any]
    io: Dict[str, Any]
    pism: Dict[str, Any]
    cost_estimate: Dict[str, Any]


def _require(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value


def _ensure_s3_uri(value: str, field_name: str) -> str:
    if not value.startswith("s3://"):
        raise ValueError(f"{field_name} must start with s3://")
    return value


def _resolve_instance(instance: str | None) -> str:
    if instance is None:
        raise ValueError("Instance type not specified")
    return INSTANCE_ALIASES.get(instance, instance)


def _select_instance_for_budget(
    candidates: list[str],
    use_spot: bool,
    max_usd: float,
    hours: float,
    ensemble_members: int,
    hourly_override: float | None,
) -> str:
    affordable = []
    for candidate in candidates:
        try:
            instance_spec(candidate)
        except Exception:
            continue
        estimate = estimate_total_cost(
            candidate,
            hours,
            ensemble_members,
            use_spot,
            hourly_override=hourly_override,
        )
        if estimate.total_usd <= max_usd:
            affordable.append((estimate.total_usd, candidate))

    if not affordable:
        raise ValueError("No instance type fits within cost.max_usd")

    affordable.sort(key=lambda item: item[0])
    return affordable[0][1]


def load_config(path: str) -> NormalizedConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    run_id = _require(data.get("run_id"), "run_id is required")
    ensemble_members = int(_require(data.get("ensemble_members"), "ensemble_members is required"))

    compute = data.get("compute", {})
    if compute.get("backend") != "aws-batch":
        raise ValueError("compute.backend must be aws-batch")

    cost = data.get("cost", {})
    max_usd = float(_require(cost.get("max_usd"), "cost.max_usd is required"))
    hours = float(cost.get("estimated_hours_per_member", 1.0))
    hourly_override = cost.get("estimated_usd_per_hour")
    if hourly_override is not None:
        hourly_override = float(hourly_override)

    use_spot = bool(compute.get("use_spot", True))
    requested_gpus = int(compute.get("gpus", 0) or 0)

    instance_value = compute.get("instance")
    instance_types = compute.get("instance_types")
    if instance_value is None:
        if hourly_override is not None:
            raise ValueError("cost.estimated_usd_per_hour requires compute.instance")
        if instance_types is None:
            instance_types = (
                DEFAULT_GPU_INSTANCE_TYPES if requested_gpus > 0 else DEFAULT_CPU_INSTANCE_TYPES
            )
        if isinstance(instance_types, str):
            candidates = [value.strip() for value in instance_types.split(",") if value.strip()]
        else:
            candidates = list(instance_types)
        instance = _select_instance_for_budget(
            candidates, use_spot, max_usd, hours, ensemble_members, hourly_override
        )
    else:
        instance = _resolve_instance(instance_value)

    spec = instance_spec(instance)

    if spec.gpus > 0 and not use_spot:
        raise ValueError("GPU on-demand is not configured; set compute.use_spot to true")

    gpus = int(compute.get("gpus", spec.gpus))
    mpi_ranks = int(compute.get("mpi_ranks", gpus if gpus > 0 else 1))

    if spec.gpus > 0 and gpus == 0:
        raise ValueError("GPU instance selected but gpus is set to 0")
    if spec.gpus == 0 and gpus > 0:
        raise ValueError("CPU instance selected but gpus > 0")

    if gpus > 0 and mpi_ranks != gpus:
        raise ValueError("GPU jobs must use exactly 1 MPI rank per GPU")

    if mpi_ranks < 1:
        raise ValueError("mpi_ranks must be >= 1")
    if mpi_ranks > spec.vcpus:
        raise ValueError("mpi_ranks exceeds instance vCPU count")

    estimate = estimate_total_cost(
        instance,
        hours,
        ensemble_members,
        use_spot,
        hourly_override=hourly_override,
    )

    if estimate.total_usd > max_usd:
        raise ValueError(
            f"Estimated cost {estimate.total_usd:.2f} exceeds max_usd {max_usd:.2f}"
        )

    io_cfg = data.get("io", {})
    input_s3 = _ensure_s3_uri(_require(io_cfg.get("input_s3"), "io.input_s3 is required"), "io.input_s3")
    output_s3 = _ensure_s3_uri(
        _require(io_cfg.get("output_s3"), "io.output_s3 is required"),
        "io.output_s3",
    )

    pism_cfg = data.get("pism", {})
    args = _require(pism_cfg.get("args"), "pism.args is required")
    executable = pism_cfg.get("executable", "pismr")

    normalized_compute = {
        "backend": "aws-batch",
        "instance": instance,
        "vcpus": spec.vcpus,
        "memory_mib": spec.memory_mib,
        "gpus": gpus,
        "mpi_ranks": mpi_ranks,
        "use_spot": use_spot,
    }

    normalized_cost = {
        "max_usd": max_usd,
        "estimated_hours_per_member": hours,
        "estimated_usd_per_hour": hourly_override,
    }

    normalized_io = {"input_s3": input_s3, "output_s3": output_s3}
    normalized_pism = {"args": args, "executable": executable}

    cost_estimate = {
        "hourly_rate_usd": estimate.hourly_rate_usd,
        "total_usd": estimate.total_usd,
        "use_spot": estimate.use_spot,
    }

    return NormalizedConfig(
        run_id=run_id,
        ensemble_members=ensemble_members,
        compute=normalized_compute,
        cost=normalized_cost,
        io=normalized_io,
        pism=normalized_pism,
        cost_estimate=cost_estimate,
    )
