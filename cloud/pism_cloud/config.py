"""Load and validate PISM cloud configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from .aws import instance_spec

INSTANCE_ALIASES = {"c7i": "c7i.4xlarge", "hpc6a": "hpc6a.48xlarge", "g5": "g5.xlarge"}


@dataclass(frozen=True)
class NormalizedConfig:
    run_id: str
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


def _select_instance(candidates: Iterable[str]) -> str:
    resolved = []
    has_cpu = False
    has_gpu = False
    for candidate in candidates:
        try:
            spec = instance_spec(candidate)
        except Exception:
            continue
        resolved.append((candidate, spec))
        if spec.gpus > 0:
            has_gpu = True
        else:
            has_cpu = True
    if not resolved:
        raise ValueError("No instance types resolved from compute.instance_types")
    if has_cpu and has_gpu:
        raise ValueError("compute.instance_types must not mix CPU and GPU instance types")
    return resolved[0][0]


def _load_documents(path: str) -> List[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as handle:
        docs = list(yaml.safe_load_all(handle))
    results: List[Tuple[int, Dict[str, Any]]] = []
    for index, doc in enumerate(docs, start=1):
        if doc is None:
            continue
        if not isinstance(doc, dict):
            raise ValueError(f"{path}: document {index} must be a mapping")
        results.append((index, doc))
    return results


def _normalize_config(data: Dict[str, Any], source: str) -> NormalizedConfig:
    run_id = _require(data.get("run_id"), f"{source}: run_id is required")
    compute = data.get("compute", {})
    if compute.get("backend") != "aws-batch":
        raise ValueError(f"{source}: compute.backend must be aws-batch")

    use_spot = bool(compute.get("use_spot", True))
    instance_value = compute.get("instance")
    instance_types = compute.get("instance_types")
    if instance_value is None:
        if instance_types is None:
            raise ValueError(f"{source}: compute.instance or compute.instance_types is required")
        if isinstance(instance_types, str):
            candidates = [value.strip() for value in instance_types.split(",") if value.strip()]
        else:
            candidates = list(instance_types)
        instance = _select_instance(candidates)
    else:
        instance = _resolve_instance(instance_value)

    spec = instance_spec(instance)

    gpus = int(spec.gpus)
    mpi_ranks = int(gpus if gpus > 0 else spec.vcpus)

    io_cfg = data.get("io", {})
    input_s3 = io_cfg.get("input_s3")
    output_s3 = io_cfg.get("output_s3")
    input_prefix = io_cfg.get("input_prefix")
    output_prefix = io_cfg.get("output_prefix")

    if input_s3 is not None:
        input_s3 = _ensure_s3_uri(input_s3, "io.input_s3")
        input_prefix = None
    else:
        input_prefix = _normalize_prefix(
            _require(input_prefix, f"{source}: io.input_prefix is required")
        )

    if output_s3 is not None:
        output_s3 = _ensure_s3_uri(output_s3, "io.output_s3")
        output_prefix = None
    else:
        output_prefix = _normalize_prefix(output_prefix or run_id)

    pism_cfg = data.get("pism", {})
    args = _require(pism_cfg.get("args"), f"{source}: pism.args is required")
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
        compute=normalized_compute,
        io=normalized_io,
        pism=normalized_pism,
    )


def load_configs(paths: List[str]) -> List[NormalizedConfig]:
    configs: List[NormalizedConfig] = []
    for path in paths:
        documents = _load_documents(path)
        if not documents:
            raise ValueError(f"{path}: no config documents found")
        for index, data in documents:
            source = f"{path} (doc {index})"
            configs.append(_normalize_config(data, source))
    return configs
