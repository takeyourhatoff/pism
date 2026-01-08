"""Load and validate PISM cloud configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable

import yaml

from .aws import instance_spec

INSTANCE_ALIASES = {"c7i": "c7i.4xlarge", "hpc6a": "hpc6a.48xlarge", "g5": "g5.xlarge"}


@dataclass(frozen=True)
class NormalizedConfig:
    run_id: str
    ensemble_members: int
    compute: Dict[str, Any]
    io: Dict[str, Any]
    pism: Dict[str, Any]


def _require(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value


def _ensure_s3_uri(value: str, field_name: str) -> str:
    if not value.startswith("s3://"):
        raise ValueError(f"{field_name} must start with s3://")
    return value


def _normalize_prefix(value: str) -> str:
    return value.strip("/")


def _resolve_instance(instance: str | None) -> str:
    if instance is None:
        raise ValueError("Instance type not specified")
    return INSTANCE_ALIASES.get(instance, instance)


def _select_instance(
    candidates: Iterable[str],
    requested_gpus: int | None,
) -> str:
    for candidate in candidates:
        try:
            spec = instance_spec(candidate)
        except Exception:
            continue
        if requested_gpus is None:
            return candidate
        if requested_gpus == 0 and spec.gpus == 0:
            return candidate
        if requested_gpus > 0 and spec.gpus >= requested_gpus:
            return candidate
    raise ValueError("No instance type matches requested GPUs")


def load_config(path: str) -> NormalizedConfig:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    run_id = _require(data.get("run_id"), "run_id is required")
    ensemble_members = int(_require(data.get("ensemble_members"), "ensemble_members is required"))

    compute = data.get("compute", {})
    if compute.get("backend") != "aws-batch":
        raise ValueError("compute.backend must be aws-batch")

    use_spot = bool(compute.get("use_spot", True))
    requested_gpus_raw = compute.get("gpus")
    requested_gpus = int(requested_gpus_raw) if requested_gpus_raw is not None else None

    instance_value = compute.get("instance")
    instance_types = compute.get("instance_types")
    if instance_value is None:
        if instance_types is None:
            raise ValueError("compute.instance or compute.instance_types is required")
        if isinstance(instance_types, str):
            candidates = [value.strip() for value in instance_types.split(",") if value.strip()]
        else:
            candidates = list(instance_types)
        instance = _select_instance(candidates, requested_gpus)
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

    io_cfg = data.get("io", {})
    input_s3 = io_cfg.get("input_s3")
    output_s3 = io_cfg.get("output_s3")
    input_prefix = io_cfg.get("input_prefix")
    output_prefix = io_cfg.get("output_prefix")

    if input_s3 is not None:
        input_s3 = _ensure_s3_uri(input_s3, "io.input_s3")
        input_prefix = None
    else:
        input_prefix = _normalize_prefix(_require(input_prefix, "io.input_prefix is required"))

    if output_s3 is not None:
        output_s3 = _ensure_s3_uri(output_s3, "io.output_s3")
        output_prefix = None
    else:
        output_prefix = _normalize_prefix(output_prefix or run_id)

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

    normalized_io = {
        "input_s3": input_s3,
        "output_s3": output_s3,
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
    }
    normalized_pism = {"args": args, "executable": executable}

    return NormalizedConfig(
        run_id=run_id,
        ensemble_members=ensemble_members,
        compute=normalized_compute,
        io=normalized_io,
        pism=normalized_pism,
    )
