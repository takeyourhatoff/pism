"""CLI for submitting PISM ensembles to AWS Batch."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

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
from .images import build_and_push, repo_root
from .infra import deploy_stack

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
        "estimated_usd_per_hour": config.cost.get("estimated_usd_per_hour"),
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
        job_queue = select_queue(config.compute["gpus"], config.compute["use_spot"])
    except ValueError as exc:
        print(f"Queue selection error: {exc}", file=sys.stderr)
        return 2

    job_definition = select_job_definition(config.compute["gpus"])
    member_ids = [f"{i + 1:04d}" for i in range(config.ensemble_members)]

    def overrides_builder(member_id: str) -> Dict[str, object]:
        rendered_args = render_args(config.pism["args"], config.run_id, member_id)
        return build_overrides(
            vcpus=config.compute["vcpus"],
            memory_mib=config.compute["memory_mib"],
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


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_images_internal(args: argparse.Namespace) -> tuple[str, str]:
    root = repo_root()
    context = Path(args.context or root)
    cpu_dockerfile = Path(args.cpu_dockerfile)
    gpu_dockerfile = Path(args.gpu_dockerfile)

    cpu_image = build_and_push(
        repo_name=args.cpu_repo,
        dockerfile=cpu_dockerfile,
        tag=args.tag,
        context=context,
        region=args.region,
    )
    gpu_image = build_and_push(
        repo_name=args.gpu_repo,
        dockerfile=gpu_dockerfile,
        tag=args.tag,
        context=context,
        region=args.region,
        build_args={"CUDA_ARCH": args.cuda_arch},
    )

    return cpu_image, gpu_image


def build_images(args: argparse.Namespace) -> int:
    cpu_image, gpu_image = _build_images_internal(args)
    print(f"CPU image: {cpu_image}")
    print(f"GPU image: {gpu_image}")
    return 0


def deploy(args: argparse.Namespace) -> int:
    template_path = Path(args.template)
    cpu_image = args.cpu_image
    gpu_image = args.gpu_image

    if args.build_images:
        cpu_image, gpu_image = _build_images_internal(args)

    if not cpu_image or not gpu_image:
        print("--cpu-image and --gpu-image are required when --no-build-images is set", file=sys.stderr)
        return 2

    parameters = {
        "SubnetIds": ",".join(args.subnet_ids),
        "SecurityGroupIds": ",".join(args.security_group_ids),
        "InputBucketArn": args.input_bucket_arn,
        "OutputBucketArn": args.output_bucket_arn,
        "CpuImage": cpu_image,
        "GpuImage": gpu_image,
    }

    outputs = deploy_stack(args.stack_name, template_path, parameters)
    print("Stack deployed")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


def dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=False,
        app_dir=str(repo_root() / "cloud"),
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

    images_parser = subcommands.add_parser("build-images", help="Build and push CPU/GPU images to ECR")
    images_parser.add_argument("--cpu-repo", default="pism-cpu")
    images_parser.add_argument("--gpu-repo", default="pism-gpu")
    images_parser.add_argument("--cpu-dockerfile", default=str(repo_root() / "cloud/images/Dockerfile.cpu"))
    images_parser.add_argument("--gpu-dockerfile", default=str(repo_root() / "cloud/images/Dockerfile.gpu"))
    images_parser.add_argument("--tag", default="latest")
    images_parser.add_argument("--context", default=str(repo_root()))
    images_parser.add_argument("--region", default=None)
    images_parser.add_argument("--cuda-arch", default="sm_86")

    deploy_parser = subcommands.add_parser("deploy", help="Build images and deploy CloudFormation stack")
    deploy_parser.add_argument("--stack-name", default="pism-batch")
    deploy_parser.add_argument("--template", default=str(repo_root() / "cloud/infra/cloudformation/pism-batch.yml"))
    deploy_parser.add_argument("--subnet-ids", type=_parse_csv, required=True)
    deploy_parser.add_argument("--security-group-ids", type=_parse_csv, required=True)
    deploy_parser.add_argument("--input-bucket-arn", required=True)
    deploy_parser.add_argument("--output-bucket-arn", required=True)
    deploy_parser.add_argument("--cpu-repo", default="pism-cpu")
    deploy_parser.add_argument("--gpu-repo", default="pism-gpu")
    deploy_parser.add_argument("--cpu-dockerfile", default=str(repo_root() / "cloud/images/Dockerfile.cpu"))
    deploy_parser.add_argument("--gpu-dockerfile", default=str(repo_root() / "cloud/images/Dockerfile.gpu"))
    deploy_parser.add_argument("--tag", default="latest")
    deploy_parser.add_argument("--context", default=str(repo_root()))
    deploy_parser.add_argument("--region", default=None)
    deploy_parser.add_argument("--cuda-arch", default="sm_86")
    deploy_parser.add_argument("--no-build-images", dest="build_images", action="store_false")
    deploy_parser.add_argument("--cpu-image", default=None)
    deploy_parser.add_argument("--gpu-image", default=None)
    deploy_parser.set_defaults(build_images=True)

    dashboard_parser = subcommands.add_parser("dashboard", help="Run the local dashboard")
    dashboard_parser.add_argument("--host", default="0.0.0.0")
    dashboard_parser.add_argument("--port", type=int, default=8080)

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
    if args.command == "build-images":
        return build_images(args)
    if args.command == "deploy":
        return deploy(args)
    if args.command == "dashboard":
        return dashboard(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
