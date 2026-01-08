"""Minimal dashboard for PISM Batch runs."""

from __future__ import annotations

import os
from typing import Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pism_cloud.aws import aws_region
from pism_cloud.batch import describe_jobs, summarize_status
from pism_cloud.dynamo import get_run, get_table, list_jobs, list_runs
from pism_cloud.logs import fetch_log_events
from pism_cloud.progress import estimate_progress
from pism_cloud.runtime import hourly_rate, summarize_costs

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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse("dashboard.html", {"request": request})


@app.get("/api/runs")
async def runs() -> List[Dict[str, object]]:
    table = get_table()
    items = list_runs(table)
    results = []
    for item in items:
        results.append(
            {
                "run_id": item.get("run_id"),
                "created_at": item.get("created_at"),
                "instance_type": item.get("instance_type"),
                "use_spot": item.get("use_spot"),
            }
        )
    return sorted(results, key=lambda item: item.get("created_at", ""))


@app.get("/api/runs/{run_id}")
async def run_detail(run_id: str) -> Dict[str, object]:
    table = get_table()
    jobs = list_jobs(table, run_id)
    if not jobs:
        raise HTTPException(status_code=404, detail="Run not found")

    run_item = get_run(table, run_id) or {}
    job_ids = [job["job_id"] for job in jobs]
    batch_jobs = describe_jobs(job_ids)
    summary = summarize_status(batch_jobs)

    hourly_rate_usd = None
    cost_to_date_usd = None
    running_rate_usd = None
    progress_percent = None
    eta_seconds = None
    estimated_total_cost = None
    instance_type = run_item.get("instance_type")
    use_spot = run_item.get("use_spot")
    budget_usd = run_item.get("budget_usd")
    timeout_seconds = run_item.get("timeout_seconds")
    if instance_type and use_spot is not None:
        try:
            rate = hourly_rate(str(instance_type), bool(use_spot))
            snapshot = summarize_costs(batch_jobs, rate)
            hourly_rate_usd = snapshot.hourly_rate_usd
            cost_to_date_usd = snapshot.cost_to_date_usd
            running_rate_usd = snapshot.running_rate_usd
        except Exception:
            pass

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
                events = fetch_log_events(LOG_GROUP, log_stream)
                progress = estimate_progress(events)
                if progress:
                    progress_percent = progress.progress_fraction * 100.0
                    eta_seconds = progress.eta_seconds
                    if hourly_rate_usd is not None and cost_to_date_usd is not None:
                        eta_hours = progress.eta_seconds / 3600.0
                        estimated_total_cost = cost_to_date_usd + eta_hours * hourly_rate_usd
            except Exception:
                pass

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
    return {
        "run_id": run_id,
        "summary": summary,
        "jobs": job_details,
        "hourly_rate_usd": hourly_rate_usd,
        "cost_to_date_usd": cost_to_date_usd,
        "running_rate_usd": running_rate_usd,
        "budget_usd": budget_usd,
        "timeout_seconds": timeout_seconds,
        "progress_percent": progress_percent,
        "eta_seconds": eta_seconds,
        "estimated_total_cost_usd": estimated_total_cost,
    }
