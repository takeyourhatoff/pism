"""Minimal dashboard for PISM Batch jobs."""

from __future__ import annotations

import os
import urllib.parse
from typing import Dict, List

import boto3
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pism_cloud.aws import aws_region
from pism_cloud.batch import describe_jobs, summarize_status
from pism_cloud.dynamo import get_job, get_table, list_jobs
from pism_cloud.logs import fetch_log_events
from pism_cloud.progress import estimate_progress
from pism_cloud.runtime import hourly_rate, summarize_costs, summarize_runtime_seconds

LOG_GROUP = os.environ.get("PISM_LOG_GROUP", "/aws/batch/pism")
TEMPLATES = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "..", "pism_cloud", "templates")
)

app = FastAPI()


def log_url(log_stream: str) -> str:
    region = aws_region()
    log_group = LOG_GROUP.replace("/", "$252F")
    stream = log_stream.replace("/", "$252F")
    return (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#logsV2:log-groups/log-group/{log_group}/log-events/{stream}"
    )


def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    if not uri or not uri.startswith("s3://"):
        return None
    path = uri[len("s3://") :]
    if not path:
        return None
    if "/" in path:
        bucket, prefix = path.split("/", 1)
    else:
        bucket, prefix = path, ""
    return bucket, prefix.rstrip("/")


def _s3_console_url(bucket: str, prefix: str) -> str:
    region = aws_region()
    quoted = urllib.parse.quote(prefix + "/" if prefix else "")
    return (
        f"https://s3.console.aws.amazon.com/s3/buckets/{bucket}"
        f"?region={region}&prefix={quoted}&showversions=false"
    )


def _list_output_objects(uri: str, max_keys: int = 50) -> tuple[Dict[str, object], List[Dict[str, object]]]:
    parsed = _parse_s3_uri(uri)
    if not parsed:
        return {}, []
    bucket, prefix = parsed
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(
        Bucket=bucket,
        Prefix=f"{prefix}/" if prefix else "",
        MaxKeys=max_keys + 1,
    )
    contents = response.get("Contents", [])
    objects = []
    for item in contents[:max_keys]:
        key = item.get("Key", "")
        if not key or key == prefix or key == f"{prefix}/":
            continue
        objects.append(
            {
                "key": key,
                "size_bytes": item.get("Size", 0),
                "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else "",
            }
        )
    summary = {
        "bucket": bucket,
        "prefix": prefix,
        "truncated": response.get("IsTruncated", False),
        "key_count": response.get("KeyCount", len(contents)),
    }
    return summary, objects


def _container_instance_arn(job: Dict[str, object]) -> str | None:
    container = job.get("container", {}) if isinstance(job.get("container"), dict) else {}
    arn = container.get("containerInstanceArn")
    if arn:
        return arn
    attempts = job.get("attempts", [])
    if isinstance(attempts, list):
        for attempt in reversed(attempts):
            container = attempt.get("container", {}) if isinstance(attempt, dict) else {}
            arn = container.get("containerInstanceArn")
            if arn:
                return arn
    return None


def _actual_instance(batch_jobs: List[Dict[str, object]]) -> Dict[str, str] | None:
    arn = None
    for job in batch_jobs:
        arn = _container_instance_arn(job)
        if arn:
            break
    if not arn:
        return None
    try:
        resource = arn.split("container-instance/", 1)[-1]
        cluster = resource.split("/", 1)[0]
        ecs = boto3.client("ecs")
        resp = ecs.describe_container_instances(cluster=cluster, containerInstances=[arn])
        if not resp.get("containerInstances"):
            return None
        ec2_instance_id = resp["containerInstances"][0].get("ec2InstanceId")
        if not ec2_instance_id:
            return None
        ec2 = boto3.client("ec2")
        res = ec2.describe_instances(InstanceIds=[ec2_instance_id])
        instance = res["Reservations"][0]["Instances"][0]
        return {
            "instance_id": ec2_instance_id,
            "instance_type": instance.get("InstanceType", ""),
            "availability_zone": instance.get("Placement", {}).get("AvailabilityZone", ""),
        }
    except Exception:
        return None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/jobs")
async def jobs() -> List[Dict[str, object]]:
    table = get_table()
    items = list_jobs(table)
    results = []
    for item in items:
        results.append(
            {
                "job_name": item.get("job_name"),
                "created_at": item.get("created_at"),
                "instance_type": item.get("instance_type"),
                "use_spot": item.get("use_spot"),
            }
        )
    return sorted(results, key=lambda item: item.get("created_at", ""))


@app.get("/api/jobs/{job_name}")
async def job_detail(job_name: str) -> Dict[str, object]:
    table = get_table()
    job_item = get_job(table, job_name)
    if not job_item:
        raise HTTPException(status_code=404, detail="Job not found")

    job_id = job_item.get("job_id")
    batch_jobs = describe_jobs([str(job_id)]) if job_id else []
    summary = summarize_status(batch_jobs) if batch_jobs else {"queued": 0, "running": 0, "succeeded": 0, "failed": 0}

    hourly_rate_usd = None
    cost_to_date_usd = None
    running_rate_usd = None
    progress_percent = None
    eta_seconds = None
    estimated_total_cost = None
    runtime_seconds = None
    actual_instance = None
    instance_type = job_item.get("instance_type")
    use_spot = job_item.get("use_spot")
    budget_usd = job_item.get("budget_usd")
    timeout_seconds = job_item.get("timeout_seconds")
    if instance_type and use_spot is not None:
        try:
            rate = hourly_rate(str(instance_type), bool(use_spot))
            snapshot = summarize_costs(batch_jobs, rate)
            hourly_rate_usd = snapshot.hourly_rate_usd
            cost_to_date_usd = snapshot.cost_to_date_usd
            running_rate_usd = snapshot.running_rate_usd
        except Exception:
            pass

    runtime_seconds = summarize_runtime_seconds(batch_jobs)
    actual_instance = _actual_instance(batch_jobs)

    running_job = next(
        (
            job
            for job in batch_jobs
            if job.get("status") in {"STARTING", "RUNNING"}
            and job.get("container", {}).get("logStreamName")
        ),
        None,
    )
    if running_job:
        log_stream = running_job.get("container", {}).get("logStreamName")
        if log_stream:
            try:
                events = fetch_log_events(LOG_GROUP, log_stream, include_head=True)
                progress = estimate_progress(events)
                if progress:
                    progress_percent = progress.progress_fraction * 100.0
                    eta_seconds = progress.eta_seconds
                    if hourly_rate_usd is not None and cost_to_date_usd is not None:
                        eta_hours = progress.eta_seconds / 3600.0
                        estimated_total_cost = cost_to_date_usd + eta_hours * hourly_rate_usd
            except Exception:
                pass

    if estimated_total_cost is None and cost_to_date_usd is not None and summary["running"] == 0:
        estimated_total_cost = cost_to_date_usd

    job_details = []
    for job in batch_jobs:
        log_stream = job.get("container", {}).get("logStreamName")
        job_details.append(
            {
                "job_name": job.get("jobName"),
                "status": job.get("status"),
                "status_reason": job.get("statusReason", ""),
                "job_id": job.get("jobId"),
                "log_url": log_url(log_stream) if log_stream else "",
            }
        )

    job_details.sort(key=lambda item: item.get("job_name", ""))
    output_s3 = job_item.get("output_s3")
    output_console_url = None
    output_summary = None
    output_objects: List[Dict[str, object]] = []
    if output_s3:
        parsed = _parse_s3_uri(output_s3)
        if parsed:
            output_console_url = _s3_console_url(*parsed)
        if summary["queued"] == 0 and summary["running"] == 0:
            try:
                output_summary, output_objects = _list_output_objects(output_s3)
            except Exception:
                output_summary = {"status": "error"}
        else:
            output_summary = {"status": "pending"}
    return {
        "job_name": job_name,
        "job_id": job_item.get("job_id"),
        "created_at": job_item.get("created_at"),
        "instance_type": job_item.get("instance_type"),
        "use_spot": job_item.get("use_spot"),
        "job_queue": job_item.get("job_queue"),
        "job_definition": job_item.get("job_definition"),
        "inputs": job_item.get("inputs", {}),
        "output_s3": output_s3,
        "output_console_url": output_console_url,
        "output_summary": output_summary,
        "output_objects": output_objects,
        "pism_args": job_item.get("pism_args"),
        "pism_executable": job_item.get("pism_executable"),
        "mpi_ranks": job_item.get("mpi_ranks"),
        "gpus": job_item.get("gpus"),
        "actual_instance_type": actual_instance.get("instance_type") if actual_instance else None,
        "actual_instance_id": actual_instance.get("instance_id") if actual_instance else None,
        "actual_availability_zone": actual_instance.get("availability_zone") if actual_instance else None,
        "summary": summary,
        "jobs": job_details,
        "hourly_rate_usd": hourly_rate_usd,
        "cost_to_date_usd": cost_to_date_usd,
        "running_rate_usd": running_rate_usd,
        "budget_usd": budget_usd,
        "timeout_seconds": timeout_seconds,
        "runtime_seconds": runtime_seconds,
        "progress_percent": progress_percent,
        "eta_seconds": eta_seconds,
        "estimated_total_cost_usd": estimated_total_cost,
    }
