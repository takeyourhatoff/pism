"""Helpers for job metadata and attempt ordering."""

from __future__ import annotations

from typing import Dict, Iterable, List


def attempt_ids_for_item(job_item: Dict[str, object]) -> List[str]:
    attempt_ids: List[str] = []
    raw_attempts = job_item.get("attempt_job_ids")
    if isinstance(raw_attempts, list):
        attempt_ids = [str(item) for item in raw_attempts if item]
    return attempt_ids


def ordered_jobs(
    batch_jobs: Iterable[Dict[str, object]], attempt_ids: Iterable[str]
) -> List[Dict[str, object]]:
    job_by_id = {str(job.get("jobId") or ""): job for job in batch_jobs}
    return [job_by_id[job_id] for job_id in attempt_ids if job_id in job_by_id]


def current_job(
    batch_jobs: Iterable[Dict[str, object]], attempt_ids: Iterable[str]
) -> Dict[str, object] | None:
    ordered = ordered_jobs(batch_jobs, attempt_ids)
    if ordered:
        return ordered[-1]
    for job in batch_jobs:
        return job
    return None
