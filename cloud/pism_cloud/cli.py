"""CLI for submitting PISM ensembles to AWS Batch."""

from __future__ import annotations

import argparse
import sys
from typing import Dict

from .batch import (
    build_overrides,
    describe_jobs,
    select_job_definition,
    select_queue,
    submit_jobs,
    summarize_status,
)
from .config import NormalizedConfig, load_config
from .dynamo import get_table, list_jobs, list_runs, store_job, store_run, utc_now

INPUT_DIR = "/workspace/input"
OUTPUT_DIR = "/workspace/output"


def render_args(template: str, run_id: str, member_id: str) -> str:
    return (
        template.replace("{{RUN_ID}}", run_id)
        .replace("{{MEMBER_ID}}", member_id)
        .replace("{{INPUT_DIR}}", INPUT_DIR)
        .replace("{{OUTPUT_DIR}}", OUTPUT_DIR)
    )


def build_run_item(
    config: NormalizedConfig,
    job_queue: str,
    job_definition: str,
) -> Dict[str, object]:
    return {
        "run_id": config.run_id,
        "sort_key": "RUN",
        "created_at": utc_now(),
        "ensemble_members": config.ensemble_members,
        "instance_type": config.compute["instance"],
        "use_spot": config.compute["use_spot"],
        "job_queue": job_queue,
        "job_definition": job_definition,
        "input_s3": config.io["input_s3"],
        "output_s3": config.io["output_s3"],
        "pism_args": config.pism["args"],
        "mpi_ranks": config.compute["mpi_ranks"],
        "gpus": config.compute["gpus"],
        "cost_max_usd": config.cost["max_usd"],
        "estimated_hours_per_member": config.cost["estimated_hours_per_member"],
        "estimated_total_cost": config.cost_estimate["total_usd"],
        "estimated_hourly_rate": config.cost_estimate["hourly_rate_usd"],
    }


def build_job_item(run_id: str, member_id: str, job_id: str, job_name: str) -> Dict[str, object]:
    return {
        "run_id": run_id,
        "sort_key": f"JOB#{member_id}",
        "member_id": member_id,
        "job_id": job_id,
        "job_name": job_name,
        "status": "SUBMITTED",
    }


def submit(config_path: str) -> int:
    try:
        config = load_config(config_path)
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    try:
        job_queue = select_queue(config.compute["instance"], config.compute["use_spot"])
    except ValueError as exc:
        print(f"Queue selection error: {exc}", file=sys.stderr)
        return 2

    job_definition = select_job_definition(config.compute["instance"])
    member_ids = [f"{i + 1:04d}" for i in range(config.ensemble_members)]

    def overrides_builder(member_id: str) -> Dict[str, object]:
        rendered_args = render_args(config.pism["args"], config.run_id, member_id)
        return build_overrides(
            instance_type=config.compute["instance"],
            mpi_ranks=config.compute["mpi_ranks"],
            gpus=config.compute["gpus"],
            input_s3=config.io["input_s3"],
            output_s3=config.io["output_s3"],
            pism_args=rendered_args,
            pism_executable=config.pism["executable"],
            run_id=config.run_id,
            member_id=member_id,
        )

    job_ids = submit_jobs(
        run_id=config.run_id,
        member_ids=member_ids,
        job_queue=job_queue,
        job_definition=job_definition,
        overrides_builder=overrides_builder,
    )

    table = get_table()
    store_run(table, build_run_item(config, job_queue, job_definition))
    for member_id, job_id in job_ids.items():
        job_name = f"pism-{config.run_id}-{member_id}"
        store_job(table, build_job_item(config.run_id, member_id, job_id, job_name))

    print(f"Submitted run {config.run_id} ({config.ensemble_members} jobs)")
    print(f"Queue: {job_queue}")
    print(f"Estimated total cost: ${config.cost_estimate['total_usd']:.2f}")
    return 0


def status(run_id: str) -> int:
    table = get_table()
    jobs = list_jobs(table, run_id)
    if not jobs:
        print(f"Run {run_id} not found")
        return 1

    job_ids = [job["job_id"] for job in jobs]
    batch_jobs = describe_jobs(job_ids)
    summary = summarize_status(batch_jobs)

    print(f"Run {run_id}")
    print(
        "Jobs: queued={queued} running={running} succeeded={succeeded} failed={failed}".format(
            **summary
        )
    )
    for job in batch_jobs:
        print(
            "{name} {status} {status_reason}".format(
                name=job.get("jobName"),
                status=job.get("status"),
                status_reason=job.get("statusReason", ""),
            )
        )
    return 0


def cancel(run_id: str) -> int:
    from .batch import batch_client

    table = get_table()
    jobs = list_jobs(table, run_id)
    if not jobs:
        print(f"Run {run_id} not found")
        return 1

    client = batch_client()
    for job in jobs:
        client.terminate_job(jobId=job["job_id"], reason="Cancelled by user")

    print(f"Cancelled run {run_id} ({len(jobs)} jobs)")
    return 0


def list_runs_cmd() -> int:
    table = get_table()
    runs = list_runs(table)
    if not runs:
        print("No runs found")
        return 0

    for run in sorted(runs, key=lambda item: item.get("created_at", "")):
        print(
            "{run_id} members={members} est_cost=${cost:.2f} spot={spot}".format(
                run_id=run.get("run_id"),
                members=run.get("ensemble_members"),
                cost=float(run.get("estimated_total_cost", 0.0)),
                spot=run.get("use_spot"),
            )
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pism-cloud", description="PISM AWS Batch runner")
    subcommands = parser.add_subparsers(dest="command")

    submit_parser = subcommands.add_parser("submit", help="Submit an ensemble run")
    submit_parser.add_argument("config", help="Path to config.yml")

    status_parser = subcommands.add_parser("status", help="Show run status")
    status_parser.add_argument("run_id")

    cancel_parser = subcommands.add_parser("cancel", help="Cancel a run")
    cancel_parser.add_argument("run_id")

    subcommands.add_parser("runs", help="List submitted runs")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        return submit(args.config)
    if args.command == "status":
        return status(args.run_id)
    if args.command == "cancel":
        return cancel(args.run_id)
    if args.command == "runs":
        return list_runs_cmd()

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
