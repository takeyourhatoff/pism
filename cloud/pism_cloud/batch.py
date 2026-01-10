"""AWS Batch submission and status helpers."""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import boto3

from .aws import aws_region
CPU_SPOT_QUEUE = os.environ.get("PISM_CPU_SPOT_QUEUE", "pism-cpu-spot")
GPU_SPOT_QUEUE = os.environ.get("PISM_GPU_SPOT_QUEUE", "pism-gpu-spot")
CPU_ONDEMAND_QUEUE = os.environ.get("PISM_CPU_ONDEMAND_QUEUE", "pism-cpu-ondemand")
GPU_ONDEMAND_QUEUE = os.environ.get("PISM_GPU_ONDEMAND_QUEUE", "pism-gpu-ondemand")

CPU_JOB_DEFINITION = os.environ.get("PISM_CPU_JOB_DEFINITION", "pism-cpu")
GPU_JOB_DEFINITION = os.environ.get("PISM_GPU_JOB_DEFINITION", "pism-gpu")


def batch_client():
    return boto3.client("batch", region_name=aws_region())


def select_queue(gpus: int, use_spot: bool) -> str:
    if gpus > 0:
        return GPU_SPOT_QUEUE if use_spot else GPU_ONDEMAND_QUEUE
    return CPU_SPOT_QUEUE if use_spot else CPU_ONDEMAND_QUEUE


def select_job_definition(gpus: int) -> str:
    return GPU_JOB_DEFINITION if gpus > 0 else CPU_JOB_DEFINITION


def build_job_command() -> List[str]:
    return ["python3", "/opt/pism-cloud/run_job.py"]


def build_overrides(
    vcpus: int,
    memory_mib: int,
    mpi_ranks: int,
    gpus: int,
    inputs_json: str,
    output_s3: str,
    pism_args_s3: str,
    pism_executable: str,
    job_name: str,
) -> Dict[str, object]:
    region = aws_region()
    resource_requirements = [
        {"Type": "VCPU", "Value": str(vcpus)},
        {"Type": "MEMORY", "Value": str(memory_mib)},
    ]
    if gpus > 0:
        resource_requirements.append({"Type": "GPU", "Value": str(gpus)})

    environment = [
        {"Name": "JOB_NAME", "Value": job_name},
        {"Name": "INPUTS_JSON", "Value": inputs_json},
        {"Name": "OUTPUT_S3", "Value": output_s3},
        {"Name": "MPI_RANKS", "Value": str(mpi_ranks)},
        {"Name": "PISM_EXECUTABLE", "Value": pism_executable},
        {"Name": "PISM_ARGS_S3", "Value": pism_args_s3},
        {"Name": "AWS_REGION", "Value": region},
        {"Name": "AWS_DEFAULT_REGION", "Value": region},
    ]

    return {
        "Command": build_job_command(),
        "ResourceRequirements": resource_requirements,
        "Environment": environment,
    }


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


def _cluster_name_from_container_instance_arn(arn: str) -> Optional[str]:
    marker = "container-instance/"
    if marker not in arn:
        return None
    cluster = arn.split(marker, 1)[1].split("/", 1)[0]
    return cluster or None


def resolve_instance_metadata(job: Dict[str, object]) -> Optional[Dict[str, str]]:
    container = job.get("container", {}) if isinstance(job.get("container"), dict) else {}
    container_instance_arn = container.get("containerInstanceArn")
    if not container_instance_arn:
        return None
    cluster_name = _cluster_name_from_container_instance_arn(str(container_instance_arn))
    if not cluster_name:
        return None

    region = aws_region()
    ecs = boto3.client("ecs", region_name=region)
    response = ecs.describe_container_instances(
        cluster=cluster_name,
        containerInstances=[str(container_instance_arn)],
    )
    container_instances = response.get("containerInstances", [])
    if not container_instances:
        return None
    instance_id = container_instances[0].get("ec2InstanceId")
    if not instance_id:
        return None

    ec2 = boto3.client("ec2", region_name=region)
    response = ec2.describe_instances(InstanceIds=[str(instance_id)])
    reservations = response.get("Reservations", [])
    if not reservations:
        return None
    instances = reservations[0].get("Instances", [])
    if not instances:
        return None
    payload = instances[0]
    return {
        "instance_type": str(payload.get("InstanceType", "")),
        "availability_zone": str(payload.get("Placement", {}).get("AvailabilityZone", "")),
    }
