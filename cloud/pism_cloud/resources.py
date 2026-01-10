"""Resolve AWS Batch resource requirements from job definitions."""

from __future__ import annotations

from typing import Dict

import boto3

from .aws import aws_region


def resolve_resources(job_definition: str) -> Dict[str, int]:
    region = aws_region()
    batch = boto3.client("batch", region_name=region)
    response = batch.describe_job_definitions(jobDefinitionName=job_definition, status="ACTIVE")
    definitions = response.get("jobDefinitions", [])
    if not definitions:
        raise ValueError(f"Job definition {job_definition} not found")
    definition = max(definitions, key=lambda item: int(item.get("revision", 0)))
    container = definition.get("containerProperties", {})
    requirements = container.get("resourceRequirements", [])
    resources: Dict[str, int] = {"vcpus": 0, "memory_mib": 0, "gpus": 0}
    for entry in requirements:
        if not isinstance(entry, dict):
            continue
        rtype = str(entry.get("Type") or entry.get("type") or "").upper()
        value = entry.get("Value") if "Value" in entry else entry.get("value")
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if rtype == "VCPU":
            resources["vcpus"] = parsed
        elif rtype == "MEMORY":
            resources["memory_mib"] = parsed
        elif rtype == "GPU":
            resources["gpus"] = parsed

    if resources["vcpus"] <= 0 or resources["memory_mib"] <= 0:
        raise ValueError(f"Job definition {job_definition} is missing CPU or memory requirements")

    mpi_ranks = resources["gpus"] if resources["gpus"] > 0 else resources["vcpus"]
    return {
        "vcpus": resources["vcpus"],
        "memory_mib": resources["memory_mib"],
        "gpus": resources["gpus"],
        "mpi_ranks": mpi_ranks,
    }
