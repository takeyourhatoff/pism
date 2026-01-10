"""Load and validate PISM cloud configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class NormalizedConfig:
    job_name: str
    accelerator: str
    use_spot: bool
    inputs: Dict[str, str]
    pism: Dict[str, Any]
    budget_usd: float | None


def _require(value: Any, message: str) -> Any:
    if value is None:
        raise ValueError(message)
    return value


def _normalize_prefix(value: str) -> str:
    return value.strip("/")


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
    job_name = _require(data.get("job_name"), f"{source}: job_name is required")
    budget_raw = data.get("budget_usd")
    budget_usd = None
    if budget_raw is not None:
        try:
            budget_usd = float(budget_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{source}: budget_usd must be a number") from exc
        if budget_usd <= 0:
            raise ValueError(f"{source}: budget_usd must be greater than zero")

    compute = data.get("compute", {}) or {}
    if not isinstance(compute, dict):
        raise ValueError(f"{source}: compute must be a mapping")

    use_spot = bool(compute.get("use_spot", True))
    accelerator = str(compute.get("accelerator") or "cpu").lower()
    if accelerator not in {"cpu", "gpu"}:
        raise ValueError(f"{source}: compute.accelerator must be cpu or gpu")

    inputs_cfg = data.get("inputs")
    if inputs_cfg is None:
        raise ValueError(f"{source}: inputs is required")
    if not isinstance(inputs_cfg, dict) or not inputs_cfg:
        raise ValueError(f"{source}: inputs must be a non-empty map")

    normalized_inputs: Dict[str, str] = {}
    for name, value in inputs_cfg.items():
        if name is None or str(name).strip() == "":
            raise ValueError(f"{source}: input name must not be empty")
        raw = str(value).strip()
        if not raw:
            raise ValueError(f"{source}: inputs.{name} must not be empty")
        if raw.startswith("s3://"):
            raise ValueError(f"{source}: inputs.{name} must be a prefix, not a full S3 URI")
        normalized_inputs[str(name)] = _normalize_prefix(raw)

    pism_cfg = data.get("pism", {})
    args = _require(pism_cfg.get("args"), f"{source}: pism.args is required")
    executable = pism_cfg.get("executable", "pismr")

    normalized_pism = {"args": args, "executable": executable}

    return NormalizedConfig(
        job_name=job_name,
        accelerator=accelerator,
        use_spot=use_spot,
        inputs=normalized_inputs,
        pism=normalized_pism,
        budget_usd=budget_usd,
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
