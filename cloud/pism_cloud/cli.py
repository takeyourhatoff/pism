"""CLI for submitting PISM jobs to AWS Batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

from .batch import (
    build_overrides,
    describe_jobs,
    select_job_definition,
    select_queue,
    submit_job,
)
from .config import NormalizedConfig, load_configs
from .dynamo import get_job, get_table, list_jobs, store_job, utc_now
from .images import build_and_push, repo_root
from .infra import deploy_stack, stack_outputs
from .logs import fetch_log_events
from .progress import estimate_progress
from .runtime import hourly_rate, job_costs, summarize_costs
from .s3 import upload_path, upload_text

INPUT_DIR = "/workspace/input"
OUTPUT_DIR = "/workspace/output"


def render_args_with_inputs(template: str, job_name: str, input_names: list[str]) -> str:
    rendered = (
        template.replace("{{JOB_NAME}}", job_name)
        .replace("{{INPUT_DIR}}", INPUT_DIR)
        .replace("{{OUTPUT_DIR}}", OUTPUT_DIR)
    )
    for name in input_names:
        rendered = rendered.replace(f"{{{{INPUTS.{name}}}}}", f"{INPUT_DIR}/{name}")
    return rendered


def build_job_item(
    config: NormalizedConfig,
    job_queue: str,
    job_definition: str,
    inputs_s3: Dict[str, str],
    output_s3: str,
    timeout_seconds: int | None,
    job_id: str,
    job_name: str,
) -> Dict[str, object]:
    item: Dict[str, object] = {
        "job_name": job_name,
        "sort_key": "JOB",
        "created_at": utc_now(),
        "job_id": job_id,
        "instance_type": config.compute["instance"],
        "use_spot": config.compute["use_spot"],
        "job_queue": job_queue,
        "job_definition": job_definition,
        "inputs": inputs_s3,
        "output_s3": output_s3,
        "pism_args": config.pism["args"],
        "pism_executable": config.pism["executable"],
        "mpi_ranks": config.compute["mpi_ranks"],
        "gpus": config.compute["gpus"],
        "status": "SUBMITTED",
    }
    if config.budget_usd is not None:
        item["budget_usd"] = config.budget_usd
    if timeout_seconds is not None:
        item["timeout_seconds"] = timeout_seconds
    return item


def submit(config_paths: list[str], stack_name: str) -> int:
    try:
        configs = load_configs(config_paths)
    except ValueError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 2

    table = get_table()

    seen_job_names = set()
    try:
        outputs = stack_outputs(stack_name)
    except Exception as exc:
        print(f"Stack lookup failed: {exc}", file=sys.stderr)
        return 2

    input_bucket = outputs.get("InputBucketName")
    output_bucket = outputs.get("OutputBucketName")
    if not input_bucket or not output_bucket:
        print("InputBucketName/OutputBucketName not found in stack outputs", file=sys.stderr)
        return 2

    for cfg in configs:
        if cfg.job_name in seen_job_names:
            print(f"Config error: duplicate job_name {cfg.job_name}", file=sys.stderr)
            return 2
        seen_job_names.add(cfg.job_name)

    for cfg in configs:
        try:
            existing = get_job(table, cfg.job_name)
        except Exception as exc:
            print(f"Job lookup failed: {exc}", file=sys.stderr)
            return 2
        if existing:
            print(f"Config error: job_name {cfg.job_name} already exists", file=sys.stderr)
            return 2

    for cfg in configs:
        try:
            job_queue = select_queue(cfg.compute["gpus"], cfg.compute["use_spot"])
        except ValueError as exc:
            print(f"Queue selection error: {exc}", file=sys.stderr)
            return 2

        job_definition = select_job_definition(cfg.compute["gpus"])
        resolved_inputs: Dict[str, str] = {}
        for name, value in cfg.inputs.items():
            if value.startswith("s3://"):
                resolved_inputs[name] = value
            else:
                resolved_inputs[name] = f"s3://{input_bucket}/{value}"

        output_s3 = f"s3://{output_bucket}/{cfg.job_name}"

        rendered_args = render_args_with_inputs(
            cfg.pism["args"],
            cfg.job_name,
            list(cfg.inputs.keys()),
        )

        timeout_seconds = None
        if cfg.budget_usd is not None:
            try:
                rate = hourly_rate(str(cfg.compute["instance"]), bool(cfg.compute["use_spot"]))
            except Exception as exc:
                print(f"Budget error: unable to fetch pricing for {cfg.compute['instance']}: {exc}", file=sys.stderr)
                return 2
            if rate <= 0:
                print(f"Budget error: invalid hourly rate for {cfg.compute['instance']}", file=sys.stderr)
                return 2
            budget_hours = cfg.budget_usd / rate
            timeout_seconds = max(60, int(budget_hours * 3600))
        args_s3 = upload_text(rendered_args, output_s3, "pism_args.txt")
        overrides = build_overrides(
            vcpus=cfg.compute["vcpus"],
            memory_mib=cfg.compute["memory_mib"],
            mpi_ranks=cfg.compute["mpi_ranks"],
            gpus=cfg.compute["gpus"],
            inputs_json=json.dumps(resolved_inputs),
            output_s3=output_s3,
            pism_args="",
            pism_args_s3=args_s3,
            pism_executable=cfg.pism["executable"],
            job_name=cfg.job_name,
        )

        job_id, job_name = submit_job(
            job_name=cfg.job_name,
            job_queue=job_queue,
            job_definition=job_definition,
            overrides=overrides,
            timeout_seconds=timeout_seconds,
            tags={
                "PismJobName": cfg.job_name,
                "PismInstanceType": cfg.compute["instance"],
                "PismSpot": "true" if cfg.compute["use_spot"] else "false",
            },
        )

        store_job(
            table,
            build_job_item(
                cfg,
                job_queue,
                job_definition,
                resolved_inputs,
                output_s3,
                timeout_seconds,
                job_id,
                job_name,
            ),
        )

        print(f"Submitted job {cfg.job_name}")
        print(f"Queue: {job_queue}")
        if cfg.budget_usd is not None:
            if timeout_seconds is not None:
                hours = timeout_seconds / 3600.0
                print(f"Budget guard: ${cfg.budget_usd:.2f} (~{hours:.2f}h timeout)")
            else:
                print(f"Budget guard: ${cfg.budget_usd:.2f}")

    return 0


def status(job_name: str) -> int:
    table = get_table()
    job_item = get_job(table, job_name)
    if not job_item:
        print(f"Job {job_name} not found")
        return 1

    job_id = job_item.get("job_id")
    if not job_id:
        print(f"Job {job_name} missing AWS Batch job id")
        return 1

    batch_jobs = describe_jobs([str(job_id)])
    if not batch_jobs:
        print(f"Job {job_name} not found in AWS Batch")
        return 1

    job = batch_jobs[0]

    print(f"Job {job_name}")
    status_value = job.get("status") or "UNKNOWN"
    status_reason = job.get("statusReason", "")
    if status_reason:
        print(f"Status: {status_value} ({status_reason})")
    else:
        print(f"Status: {status_value}")
    rate = None
    snapshot = None
    cost_by_job = {}
    instance_type = job_item.get("instance_type")
    use_spot = job_item.get("use_spot")
    budget_usd = job_item.get("budget_usd")

    if instance_type and use_spot is not None:
        try:
            rate = hourly_rate(str(instance_type), bool(use_spot))
            snapshot = summarize_costs(batch_jobs, rate)
            print(
                "Cost per hour: ${rate:.2f}/hr per job | Running cost per hour: ${running:.2f}/hr | Cost so far: ${total:.2f}".format(
                    rate=snapshot.hourly_rate_usd,
                    running=snapshot.running_rate_usd,
                    total=snapshot.cost_to_date_usd,
                )
            )
            cost_by_job = job_costs(batch_jobs, rate)
        except Exception as exc:
            print(f"Cost per hour unavailable: {exc}", file=sys.stderr)

    progress = None
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
            log_group = os.environ.get("PISM_LOG_GROUP", "/aws/batch/pism")
            try:
                events = fetch_log_events(log_group, log_stream, include_head=True)
                progress = estimate_progress(events)
            except Exception as exc:
                print(f"ETA unavailable: {exc}", file=sys.stderr)

    if progress:
        bar = _format_progress_bar(progress.progress_fraction)
        eta = _format_eta(progress.eta_seconds)
        print(
            "Progress: {bar} | model time {current:.2f}/{end:.2f} | ETA {eta}".format(
                bar=bar,
                current=progress.current_time,
                end=progress.end_time,
                eta=eta,
            )
        )
        if rate is not None and snapshot is not None:
            eta_hours = progress.eta_seconds / 3600.0
            estimated_total = snapshot.cost_to_date_usd + eta_hours * rate
            print(f"Estimated total cost: ${estimated_total:.2f}")
            if budget_usd is not None:
                buffer_multiplier = 1.2
                min_wall_seconds = 900
                if (
                    estimated_total > float(budget_usd) * buffer_multiplier
                    and progress.wall_seconds >= min_wall_seconds
                    and progress.sample_count >= 3
                ):
                    print(
                        f"Budget guard: estimate ${estimated_total:.2f} exceeds budget ${float(budget_usd):.2f}; cancelling job"
                    )
                    from .batch import batch_client

                    client = batch_client()
                    client.terminate_job(jobId=str(job_id), reason="Budget estimate exceeded")

    timeout_seconds = job_item.get("timeout_seconds")
    if budget_usd is not None:
        if timeout_seconds:
            hours = float(timeout_seconds) / 3600.0
            print(f"Budget guard: ${float(budget_usd):.2f} (~{hours:.2f}h timeout)")
        else:
            print(f"Budget guard: ${float(budget_usd):.2f}")

    for job in batch_jobs:
        job_id_value = str(job.get("jobId") or "")
        cost_value = cost_by_job.get(job_id_value)
        cost_suffix = ""
        if cost_value is not None and cost_value > 0:
            cost_suffix = f" cost=${cost_value:.2f}"
        print(
            "{name} {status} {status_reason}{cost}".format(
                name=job.get("jobName"),
                status=job.get("status"),
                status_reason=job.get("statusReason", ""),
                cost=cost_suffix,
            )
        )
    return 0


def _format_progress_bar(progress: float, width: int = 24) -> str:
    clamped = max(0.0, min(1.0, progress))
    filled = int(round(clamped * width))
    return "[{fill}{rest}] {pct:5.1f}%".format(
        fill="#" * filled,
        rest="-" * (width - filled),
        pct=clamped * 100.0,
    )


def _format_eta(seconds: float) -> str:
    if seconds <= 0:
        return "0m"
    minutes = int(round(seconds / 60.0))
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"


def cancel(job_name: str) -> int:
    from .batch import batch_client

    table = get_table()
    job_item = get_job(table, job_name)
    if not job_item:
        print(f"Job {job_name} not found")
        return 1
    job_id = job_item.get("job_id")
    if not job_id:
        print(f"Job {job_name} missing AWS Batch job id")
        return 1

    client = batch_client()
    client.terminate_job(jobId=str(job_id), reason="Cancelled by user")

    print(f"Cancelled job {job_name}")
    return 0


def list_jobs_cmd() -> int:
    table = get_table()
    jobs = list_jobs(table)
    if not jobs:
        print("No jobs found")
        return 0

    for job in sorted(jobs, key=lambda item: item.get("created_at", "")):
        print(
            "{job_name} instance={instance} spot={spot}".format(
                job_name=job.get("job_name"),
                instance=job.get("instance_type"),
                spot=job.get("use_spot"),
            )
        )
    return 0


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
        "CpuImage": cpu_image,
        "GpuImage": gpu_image,
    }
    if args.vpc_cidr:
        parameters["VpcCidr"] = args.vpc_cidr
    if args.public_subnet_cidr_a:
        parameters["PublicSubnetCidrA"] = args.public_subnet_cidr_a
    if args.public_subnet_cidr_b:
        parameters["PublicSubnetCidrB"] = args.public_subnet_cidr_b
    if args.spot_allocation_strategy:
        parameters["SpotAllocationStrategy"] = args.spot_allocation_strategy

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


def upload_inputs(args: argparse.Namespace) -> int:
    if not args.source:
        print("--source is required", file=sys.stderr)
        return 2

    source = Path(args.source)

    if args.prefix is None:
        print("--prefix is required", file=sys.stderr)
        return 2

    try:
        outputs = stack_outputs(args.stack_name)
    except Exception as exc:
        print(f"Stack lookup failed: {exc}", file=sys.stderr)
        return 2

    bucket = outputs.get("InputBucketName")
    if not bucket:
        print("InputBucketName not found in stack outputs", file=sys.stderr)
        return 2

    dest = f"s3://{bucket}/{args.prefix}"

    try:
        upload_path(source, dest)
    except ValueError as exc:
        print(f"Upload error: {exc}", file=sys.stderr)
        return 2

    print(f"Uploaded {source} to {dest}")
    return 0




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pism-cloud", description="PISM AWS Batch job runner")
    subcommands = parser.add_subparsers(dest="command")

    submit_parser = subcommands.add_parser("submit", help="Submit one or more jobs")
    submit_parser.add_argument("config", nargs="+", help="Path(s) to config.yml")
    submit_parser.add_argument("--stack-name", default="pism-batch")

    status_parser = subcommands.add_parser("status", help="Show job status")
    status_parser.add_argument("job_name")

    cancel_parser = subcommands.add_parser("cancel", help="Cancel a job")
    cancel_parser.add_argument("job_name")

    subcommands.add_parser("jobs", help="List submitted jobs")

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
    deploy_parser.add_argument("--cpu-repo", default="pism-cpu")
    deploy_parser.add_argument("--gpu-repo", default="pism-gpu")
    deploy_parser.add_argument("--cpu-dockerfile", default=str(repo_root() / "cloud/images/Dockerfile.cpu"))
    deploy_parser.add_argument("--gpu-dockerfile", default=str(repo_root() / "cloud/images/Dockerfile.gpu"))
    deploy_parser.add_argument("--tag", default="latest")
    deploy_parser.add_argument("--context", default=str(repo_root()))
    deploy_parser.add_argument("--region", default=None)
    deploy_parser.add_argument("--cuda-arch", default="sm_86")
    deploy_parser.add_argument("--vpc-cidr", default=None)
    deploy_parser.add_argument("--public-subnet-cidr-a", default=None)
    deploy_parser.add_argument("--public-subnet-cidr-b", default=None)
    deploy_parser.add_argument(
        "--spot-allocation-strategy",
        default=None,
        choices=["SPOT_PRICE_CAPACITY_OPTIMIZED", "SPOT_CAPACITY_OPTIMIZED"],
    )
    deploy_parser.add_argument("--no-build-images", dest="build_images", action="store_false")
    deploy_parser.add_argument("--cpu-image", default=None)
    deploy_parser.add_argument("--gpu-image", default=None)
    deploy_parser.set_defaults(build_images=True)

    dashboard_parser = subcommands.add_parser("dashboard", help="Run the local dashboard")
    dashboard_parser.add_argument("--host", default="0.0.0.0")
    dashboard_parser.add_argument("--port", type=int, default=8080)

    upload_parser = subcommands.add_parser("upload", help="Upload input data to the stack input bucket")
    upload_parser.add_argument("--stack-name", default="pism-batch")
    upload_parser.add_argument("--source", default=None)
    upload_parser.add_argument("--prefix", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        return submit(args.config, args.stack_name)
    if args.command == "status":
        return status(args.job_name)
    if args.command == "cancel":
        return cancel(args.job_name)
    if args.command == "jobs":
        return list_jobs_cmd()
    if args.command == "build-images":
        return build_images(args)
    if args.command == "deploy":
        return deploy(args)
    if args.command == "dashboard":
        return dashboard(args)
    if args.command == "upload":
        return upload_inputs(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
