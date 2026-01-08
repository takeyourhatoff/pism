"""AWS Batch submission and status helpers."""

from __future__ import annotations

import os
from typing import Dict, List

import boto3

CPU_SPOT_QUEUE = os.environ.get("PISM_CPU_SPOT_QUEUE", "pism-cpu-spot")
GPU_SPOT_QUEUE = os.environ.get("PISM_GPU_SPOT_QUEUE", "pism-gpu-spot")
CPU_ONDEMAND_QUEUE = os.environ.get("PISM_CPU_ONDEMAND_QUEUE", "pism-cpu-ondemand")
GPU_ONDEMAND_QUEUE = os.environ.get("PISM_GPU_ONDEMAND_QUEUE", "pism-gpu-ondemand")

CPU_JOB_DEFINITION = os.environ.get("PISM_CPU_JOB_DEFINITION", "pism-cpu")
GPU_JOB_DEFINITION = os.environ.get("PISM_GPU_JOB_DEFINITION", "pism-gpu")


def batch_client():
    return boto3.client("batch")


def select_queue(gpus: int, use_spot: bool) -> str:
    if gpus > 0:
        return GPU_SPOT_QUEUE if use_spot else GPU_ONDEMAND_QUEUE
    return CPU_SPOT_QUEUE if use_spot else CPU_ONDEMAND_QUEUE


def select_job_definition(gpus: int) -> str:
    return GPU_JOB_DEFINITION if gpus > 0 else CPU_JOB_DEFINITION


def build_job_command() -> List[str]:
    script = """
set -euo pipefail
INPUT_DIR=/workspace/input
OUTPUT_DIR=/workspace/output
mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"
aws s3 sync "$INPUT_S3" "$INPUT_DIR"
cd "$OUTPUT_DIR"
mpirun -n "$MPI_RANKS" "$PISM_EXECUTABLE" $PISM_ARGS
aws s3 sync "$OUTPUT_DIR" "$OUTPUT_S3/$MEMBER_ID"
""".strip()
    return ["bash", "-lc", script]


def build_overrides(
    vcpus: int,
    memory_mib: int,
    mpi_ranks: int,
    gpus: int,
    input_s3: str,
    output_s3: str,
    pism_args: str,
    pism_executable: str,
    run_id: str,
    member_id: str,
) -> Dict[str, object]:
    resource_requirements = [
        {"type": "VCPU", "value": str(vcpus)},
        {"type": "MEMORY", "value": str(memory_mib)},
    ]
    if gpus > 0:
        resource_requirements.append({"type": "GPU", "value": str(gpus)})

    return {
        "command": build_job_command(),
        "resourceRequirements": resource_requirements,
        "environment": [
            {"name": "RUN_ID", "value": run_id},
            {"name": "MEMBER_ID", "value": member_id},
            {"name": "INPUT_S3", "value": input_s3},
            {"name": "OUTPUT_S3", "value": output_s3},
            {"name": "MPI_RANKS", "value": str(mpi_ranks)},
            {"name": "PISM_ARGS", "value": pism_args},
            {"name": "PISM_EXECUTABLE", "value": pism_executable},
        ],
    }


def submit_jobs(
    run_id: str,
    member_ids: List[str],
    job_queue: str,
    job_definition: str,
    overrides_builder,
) -> Dict[str, str]:
    client = batch_client()
    job_ids: Dict[str, str] = {}
    for member_id in member_ids:
        job_name = f"pism-{run_id}-{member_id}"
        response = client.submit_job(
            jobName=job_name,
            jobQueue=job_queue,
            jobDefinition=job_definition,
            containerOverrides=overrides_builder(member_id),
        )
        job_ids[member_id] = response["jobId"]
    return job_ids


def describe_jobs(job_ids: List[str]) -> List[Dict[str, object]]:
    client = batch_client()
    results: List[Dict[str, object]] = []
    for i in range(0, len(job_ids), 100):
        chunk = job_ids[i : i + 100]
        if not chunk:
            continue
        response = client.describe_jobs(jobs=chunk)
        results.extend(response.get("jobs", []))
    return results


def summarize_status(jobs: List[Dict[str, object]]) -> Dict[str, int]:
    summary = {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}
    for job in jobs:
        status = job.get("status", "")
        if status in {"SUBMITTED", "PENDING", "RUNNABLE"}:
            summary["queued"] += 1
        elif status in {"STARTING", "RUNNING"}:
            summary["running"] += 1
        elif status == "SUCCEEDED":
            summary["succeeded"] += 1
        elif status == "FAILED":
            summary["failed"] += 1
    return summary
