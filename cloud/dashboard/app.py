"""Minimal dashboard for PISM Batch jobs."""

from __future__ import annotations

import os
import urllib.parse
from typing import Dict, List
from pathlib import Path

import boto3
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from pism_cloud.aws import aws_region
from pism_cloud.batch import describe_jobs, resolve_instance_metadata
from pism_cloud.dynamo import get_job, get_table, list_jobs
from pism_cloud.jobs import attempt_ids_for_item, current_job
from pism_cloud.logs import extract_log_streams_for_jobs, fetch_log_events_for_streams
from pism_cloud.progress import estimate_progress
from pism_cloud.runtime import hourly_rate, summarize_costs, summarize_runtime_seconds
from pism_cloud.s3 import parse_s3_uri
from pism_cloud.timeline import build_timeline_for_attempts
from pism_cloud.preview import (
    PreviewDependencyError,
    PreviewUnavailable,
    preview_meta,
    render_preview_png,
)

LOG_GROUP = os.environ.get("PISM_LOG_GROUP", "/aws/batch/pism")
STATIC_ROOT = (Path(__file__).resolve().parent / ".." / "pism_cloud" / "static").resolve()
ASSETS_ROOT = STATIC_ROOT / "assets"
TEMPLATES = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "..", "pism_cloud", "templates")
)
ACTIVE_STATUSES = {"SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING"}

app = FastAPI()

if ASSETS_ROOT.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_ROOT)), name="assets")


def log_url(log_stream: str) -> str:
    region = aws_region()
    log_group = LOG_GROUP.replace("/", "$252F")
    stream = log_stream.replace("/", "$252F")
    return (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#logsV2:log-groups/log-group/{log_group}/log-events/{stream}"
    )


def _parse_s3_uri(uri: str) -> tuple[str, str] | None:
    if not uri:
        return None
    try:
        return parse_s3_uri(uri)
    except ValueError:
        return None


def _s3_console_url(bucket: str, prefix: str) -> str:
    region = aws_region()
    quoted = urllib.parse.quote(prefix + "/" if prefix else "")
    return (
        f"https://s3.console.aws.amazon.com/s3/buckets/{bucket}"
        f"?region={region}&prefix={quoted}&showversions=false"
    )


def _list_output_objects(uri: str, max_keys: int = 50) -> List[Dict[str, object]]:
    parsed = _parse_s3_uri(uri)
    if not parsed:
        return []
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
        if key.endswith("pism-cloud-resume.json"):
            continue
        objects.append(
            {
                "key": key,
                "size_bytes": item.get("Size", 0),
                "last_modified": item.get("LastModified").isoformat() if item.get("LastModified") else "",
            }
        )
    return objects


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    index_path = STATIC_ROOT / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return TEMPLATES.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/jobs")
async def jobs() -> List[Dict[str, object]]:
    table = get_table()
    items = list_jobs(table)
    attempts_by_job = {}
    all_attempts: List[str] = []
    for item in items:
        job_name = item.get("job_name")
        attempts = attempt_ids_for_item(item)
        attempts_by_job[job_name] = attempts
        all_attempts.extend(attempts)
    batch_jobs = describe_jobs(all_attempts) if all_attempts else []
    results = []
    for item in items:
        job_name = item.get("job_name")
        attempts = attempts_by_job.get(job_name, [])
        current = current_job(batch_jobs, attempts) if attempts else None
        status = current.get("status") if current else None
        eta_seconds = None
        cost_to_date_usd = None
        if current:
            log_stream = current.get("container", {}).get("logStreamName")
            if status in ACTIVE_STATUSES and log_stream:
                try:
                    events = fetch_log_events_for_streams(
                        LOG_GROUP,
                        [str(log_stream)],
                        include_head=True,
                        head_limit=50,
                    )
                    progress = estimate_progress(events, monotonic=True)
                    if progress:
                        eta_seconds = progress.eta_seconds
                except Exception:
                    pass
            instance_meta = None
            try:
                instance_meta = resolve_instance_metadata(current)
            except Exception:
                instance_meta = None
            use_spot = item.get("use_spot")
            if instance_meta and instance_meta.get("instance_type") and use_spot is not None:
                rate = hourly_rate(str(instance_meta["instance_type"]), bool(use_spot))
                job_attempts = [job for job in batch_jobs if str(job.get("jobId")) in attempts]
                snapshot = summarize_costs(job_attempts, rate)
                cost_to_date_usd = snapshot.cost_to_date_usd
        results.append(
            {
                "job_name": job_name,
                "created_at": item.get("created_at"),
                "vcpus": item.get("vcpus"),
                "memory_mib": item.get("memory_mib"),
                "gpus": item.get("gpus"),
                "use_spot": item.get("use_spot"),
                "status": status,
                "eta_seconds": eta_seconds,
                "cost_to_date_usd": cost_to_date_usd,
            }
        )
    return sorted(results, key=lambda item: item.get("created_at", ""))


@app.get("/api/jobs/{job_name}")
async def job_detail(job_name: str) -> Dict[str, object]:
    table = get_table()
    job_item = get_job(table, job_name)
    if not job_item:
        raise HTTPException(status_code=404, detail="Job not found")

    attempt_ids = attempt_ids_for_item(job_item)

    batch_jobs = describe_jobs(attempt_ids) if attempt_ids else []
    active_job = current_job(batch_jobs, attempt_ids)
    status = str(active_job.get("status") or "") if active_job else ""
    status_reason = str(active_job.get("statusReason") or "") if active_job else ""

    hourly_rate_usd = None
    cost_to_date_usd = None
    progress_percent = None
    eta_seconds = None
    estimated_total_cost = None
    runtime_seconds = None
    actual_instance = None
    use_spot = job_item.get("use_spot")
    budget_usd = job_item.get("budget_usd")

    runtime_seconds = summarize_runtime_seconds(batch_jobs)

    output_s3 = job_item.get("output_s3")
    is_active = status in ACTIVE_STATUSES
    log_events = []
    log_stream = active_job.get("container", {}).get("logStreamName") if active_job else None
    log_streams = extract_log_streams_for_jobs(batch_jobs)
    if log_streams and is_active:
        try:
            log_events = fetch_log_events_for_streams(LOG_GROUP, log_streams, include_head=True)
            progress = estimate_progress(log_events, monotonic=True)
            if progress:
                progress_percent = progress.progress_fraction * 100.0
                eta_seconds = progress.eta_seconds
        except Exception:
            pass
    instance_meta = None
    if active_job:
        try:
            instance_meta = resolve_instance_metadata(active_job)
        except Exception:
            instance_meta = None
    if instance_meta and instance_meta.get("instance_type"):
        actual_instance = {
            "instance_type": instance_meta.get("instance_type", ""),
            "availability_zone": instance_meta.get("availability_zone", ""),
        }

    if actual_instance and actual_instance.get("instance_type") and use_spot is not None:
        try:
            rate = hourly_rate(str(actual_instance["instance_type"]), bool(use_spot))
            snapshot = summarize_costs(batch_jobs, rate)
            hourly_rate_usd = snapshot.hourly_rate_usd
            cost_to_date_usd = snapshot.cost_to_date_usd
        except Exception:
            pass

    if progress_percent is not None and eta_seconds is not None:
        if hourly_rate_usd is not None and cost_to_date_usd is not None:
            eta_hours = eta_seconds / 3600.0
            estimated_total_cost = cost_to_date_usd + eta_hours * hourly_rate_usd

    if estimated_total_cost is None and cost_to_date_usd is not None and not is_active:
        estimated_total_cost = cost_to_date_usd
    output_console_url = None
    output_objects: List[Dict[str, object]] = []
    inputs_console: Dict[str, str] = {}
    inputs = job_item.get("inputs", {}) if isinstance(job_item.get("inputs"), dict) else {}
    for name, uri in inputs.items():
        parsed = _parse_s3_uri(str(uri))
        if parsed:
            inputs_console[str(name)] = _s3_console_url(*parsed)
    if output_s3:
        parsed = _parse_s3_uri(output_s3)
        if parsed:
            output_console_url = _s3_console_url(*parsed)
        if not is_active:
            try:
                output_objects = _list_output_objects(output_s3)
            except Exception:
                output_objects = []
    timeline = build_timeline_for_attempts(job_item, batch_jobs, attempt_ids, log_events)

    return {
        "job_name": job_name,
        "inputs": inputs,
        "inputs_console": inputs_console,
        "output_s3": output_s3,
        "output_console_url": output_console_url,
        "output_objects": output_objects,
        "pism_args": job_item.get("pism_args"),
        "gpus": job_item.get("gpus"),
        "vcpus": job_item.get("vcpus"),
        "memory_mib": job_item.get("memory_mib"),
        "actual_instance_type": actual_instance.get("instance_type") if actual_instance else None,
        "actual_availability_zone": actual_instance.get("availability_zone") if actual_instance else None,
        "status": status,
        "status_reason": status_reason,
        "log_url": log_url(log_stream) if log_stream else "",
        "hourly_rate_usd": hourly_rate_usd,
        "cost_to_date_usd": cost_to_date_usd,
        "budget_usd": budget_usd,
        "runtime_seconds": runtime_seconds,
        "progress_percent": progress_percent,
        "eta_seconds": eta_seconds,
        "estimated_total_cost_usd": estimated_total_cost,
        "timeline": [
            {
                "timestamp": event.iso_timestamp(),
                "label": event.label,
                "detail": event.detail,
            }
            for event in timeline
        ],
    }


@app.get("/api/jobs/{job_name}/preview/meta")
async def job_preview_meta(job_name: str) -> Dict[str, object]:
    table = get_table()
    job_item = get_job(table, job_name)
    if not job_item:
        raise HTTPException(status_code=404, detail="Job not found")
    output_s3 = job_item.get("output_s3")
    if not output_s3:
        return {"available": False, "message": "No output S3 configured."}
    try:
        return preview_meta(str(output_s3))
    except PreviewDependencyError as exc:
        return {"available": False, "message": str(exc)}
    except Exception:
        return {"available": False, "message": "Preview unavailable."}


@app.get("/api/jobs/{job_name}/preview.png")
async def job_preview_png(job_name: str, var: str | None = None) -> Response:
    table = get_table()
    job_item = get_job(table, job_name)
    if not job_item:
        raise HTTPException(status_code=404, detail="Job not found")
    output_s3 = job_item.get("output_s3")
    if not output_s3:
        raise HTTPException(status_code=404, detail="No output S3 configured.")
    try:
        payload = render_preview_png(str(output_s3), var_name=var)
    except PreviewDependencyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PreviewUnavailable as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Preview failed") from exc
    return Response(content=payload, media_type="image/png")
