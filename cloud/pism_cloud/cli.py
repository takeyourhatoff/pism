"""CLI for submitting PISM jobs to AWS Batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

if sys.version_info < (3, 13):
    raise SystemExit("pism-cloud requires Python 3.13+.")

from .aws import aws_region
from .batch import (
    build_overrides,
    describe_jobs,
    resolve_instance_metadata,
    select_job_definition,
    select_queue,
)
from .config import NormalizedConfig, load_configs
from .dynamo import get_job, get_table, list_jobs, store_job, update_job_execution, utc_now
from .images import build_and_push, repo_root
from .infra import deploy_stack, stack_outputs
from .jobs import attempt_ids_for_item, current_job
from .logs import extract_log_streams_for_jobs, fetch_log_events_for_streams
from .orchestrator import start_execution, stop_execution
from .progress import estimate_progress
from .resources import resolve_resources
from .runtime import hourly_rate, summarize_costs, summarize_runtime_seconds
from .timeline import build_timeline_for_attempts
from .s3 import parse_s3_uri, upload_path, upload_text

INPUT_DIR = "/workspace/input"
OUTPUT_DIR = "/workspace/output"
RESUME_MARKER_NAME = "pism-cloud-resume.json"


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
    inputs_s3: Dict[str, str],
    output_s3: str,
    resources: Dict[str, int],
) -> Dict[str, object]:
    item: Dict[str, object] = {
        "job_name": config.job_name,
        "sort_key": "JOB",
        "created_at": utc_now(),
        "use_spot": config.use_spot,
        "inputs": inputs_s3,
        "output_s3": output_s3,
        "pism_args": config.pism["args"],
        "gpus": resources["gpus"],
        "vcpus": resources["vcpus"],
        "memory_mib": resources["memory_mib"],
    }
    if config.budget_usd is not None:
        item["budget_usd"] = config.budget_usd
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
    state_machine_arn = outputs.get("StateMachineArn")
    if not input_bucket or not output_bucket or not state_machine_arn:
        print(
            "InputBucketName/OutputBucketName/StateMachineArn not found in stack outputs",
            file=sys.stderr,
        )
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
        gpus_hint = 1 if cfg.accelerator == "gpu" else 0
        try:
            job_definition = select_job_definition(gpus_hint)
            resources = resolve_resources(job_definition)
        except ValueError as exc:
            print(f"Resource selection error: {exc}", file=sys.stderr)
            return 2
        if cfg.accelerator == "gpu" and resources["gpus"] <= 0:
            print(
                f"Resource selection error: job definition {job_definition} has no GPU requirement",
                file=sys.stderr,
            )
            return 2
        try:
            job_queue = select_queue(resources["gpus"], cfg.use_spot)
        except ValueError as exc:
            print(f"Queue selection error: {exc}", file=sys.stderr)
            return 2
        resolved_inputs: Dict[str, str] = {}
        for name, value in cfg.inputs.items():
            resolved_inputs[name] = f"s3://{input_bucket}/{value}"

        output_s3 = f"s3://{output_bucket}/{cfg.job_name}"
        output_bucket_name, output_prefix = parse_s3_uri(output_s3)
        resume_key = (
            f"{output_prefix}/{RESUME_MARKER_NAME}" if output_prefix else RESUME_MARKER_NAME
        )

        rendered_args = render_args_with_inputs(
            cfg.pism["args"],
            cfg.job_name,
            list(cfg.inputs.keys()),
        )

        args_s3 = upload_text(rendered_args, output_s3, "pism_args.txt")
        overrides = build_overrides(
            vcpus=resources["vcpus"],
            memory_mib=resources["memory_mib"],
            mpi_ranks=resources["mpi_ranks"],
            gpus=resources["gpus"],
            inputs_json=json.dumps(resolved_inputs),
            output_s3=output_s3,
            pism_args_s3=args_s3,
            pism_executable=cfg.pism["executable"],
            job_name=cfg.job_name,
        )

        store_job(
            table,
            build_job_item(
                cfg,
                resolved_inputs,
                output_s3,
                resources,
            ),
        )
        try:
            execution_arn = start_execution(
                state_machine_arn,
                {
                    "job_name": cfg.job_name,
                    "job_queue": job_queue,
                    "job_definition": job_definition,
                    "overrides": overrides,
                    "tags": {
                        "PismJobName": cfg.job_name,
                        "PismSpot": "true" if cfg.use_spot else "false",
                    },
                    "resume_bucket": output_bucket_name,
                    "resume_key": resume_key,
                },
            )
        except Exception as exc:
            print(f"Orchestration start failed: {exc}", file=sys.stderr)
            return 2

        try:
            update_job_execution(table, cfg.job_name, execution_arn)
        except Exception as exc:
            print(f"Warning: unable to store execution ARN: {exc}", file=sys.stderr)

        print(f"Submitted job {cfg.job_name}")
        print(f"Queue: {job_queue}")
        print(f"Execution: {execution_arn}")
        mem_gib = resources["memory_mib"] / 1024.0
        print(
            "Resources: vcpus={vcpus} mem={mem:.1f}GiB gpus={gpus} mpi_ranks={mpi}".format(
                vcpus=resources["vcpus"],
                mem=mem_gib,
                gpus=resources["gpus"],
                mpi=resources["mpi_ranks"],
            )
        )
        if cfg.budget_usd is not None:
            print(f"Budget guard: ${cfg.budget_usd:.2f}")

    return 0


def status(job_name: str) -> int:
    table = get_table()
    job_item = get_job(table, job_name)
    if not job_item:
        print(f"Job {job_name} not found")
        return 1

    attempt_ids = attempt_ids_for_item(job_item)

    if not attempt_ids:
        print(f"Job {job_name} has no attempts yet")
        return 1

    batch_jobs = describe_jobs(attempt_ids)
    if not batch_jobs:
        print(f"Job {job_name} not found in AWS Batch")
        return 1

    job = current_job(batch_jobs, attempt_ids)
    if not job:
        print(f"Job {job_name} not found in AWS Batch")
        return 1
    current_job_id = job.get("jobId")

    print(f"Job {job_name}")
    status_value = job.get("status") or "UNKNOWN"
    status_reason = job.get("statusReason", "")
    status_line = f"{status_value} ({status_reason})" if status_reason else status_value
    print(f"Status: {status_line}")
    rate = None
    snapshot = None
    use_spot = job_item.get("use_spot")
    budget_usd = job_item.get("budget_usd")
    actual_instance_type = None
    availability_zone = None
    execution_arn = job_item.get("execution_arn")
    runtime_seconds = summarize_runtime_seconds(batch_jobs)

    if use_spot is not None:
        print(f"Spot: {'yes' if use_spot else 'no'}")

    resources_line = _format_resources(job_item)
    if resources_line:
        print(f"Resources: {resources_line}")

    progress = None
    log_events = []
    log_group = os.environ.get("PISM_LOG_GROUP", "/aws/batch/pism")
    log_streams = extract_log_streams_for_jobs(batch_jobs)
    log_stream = job.get("container", {}).get("logStreamName")
    if log_streams:
        try:
            log_events = fetch_log_events_for_streams(log_group, log_streams, include_head=True)
            progress = estimate_progress(log_events, monotonic=True)
        except Exception as exc:
            print(f"ETA unavailable: {exc}", file=sys.stderr)

    instance_meta = None
    try:
        instance_meta = resolve_instance_metadata(job)
    except Exception as exc:
        print(f"Instance lookup unavailable: {exc}", file=sys.stderr)

    if instance_meta:
        actual_instance_type = instance_meta.get("instance_type")
        availability_zone = instance_meta.get("availability_zone")

    if actual_instance_type:
        instance_line = actual_instance_type
        if availability_zone:
            instance_line = f"{instance_line} ({availability_zone})"
        print(f"Instance: {instance_line}")

    if runtime_seconds is not None:
        print(f"Runtime: {_format_duration(runtime_seconds)}")

    if log_stream:
        print(f"Logs: {_log_url(log_group, log_stream)}")

    if actual_instance_type and use_spot is not None:
        try:
            rate = hourly_rate(str(actual_instance_type), bool(use_spot))
            snapshot = summarize_costs(batch_jobs, rate)
            print(
                "Cost per hour: ${rate:.2f}/hr | Cost so far: ${total:.2f}".format(
                    rate=snapshot.hourly_rate_usd, total=snapshot.cost_to_date_usd
                )
            )
        except Exception as exc:
            print(f"Cost per hour unavailable: {exc}", file=sys.stderr)

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
                    if current_job_id:
                        client.terminate_job(jobId=str(current_job_id), reason="Budget estimate exceeded")
                    if execution_arn:
                        try:
                            stop_execution(str(execution_arn), reason="Budget estimate exceeded")
                        except Exception as exc:
                            print(f"Warning: unable to stop orchestrator: {exc}", file=sys.stderr)

    if budget_usd is not None:
        print(f"Budget: ${float(budget_usd):.2f}")
    timeline = build_timeline_for_attempts(job_item, batch_jobs, attempt_ids, log_events)
    if timeline:
        print("Timeline:")
        for event in timeline:
            stamp = event.iso_timestamp()
            detail = f" - {event.detail}" if event.detail else ""
            print(f"{stamp} {event.label}{detail}")
    return 0


def _format_progress_bar(progress: float, width: int = 24) -> str:
    clamped = max(0.0, min(1.0, progress))
    filled = int(round(clamped * width))
    return "[{fill}{rest}] {pct:5.1f}%".format(
        fill="#" * filled,
        rest="-" * (width - filled),
        pct=clamped * 100.0,
    )


def _format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _format_memory_gib(memory_mib: object) -> str | None:
    if not isinstance(memory_mib, (int, float)):
        return None
    return f"{float(memory_mib) / 1024.0:.1f} GiB"


def _format_resources(job_item: Dict[str, object]) -> str:
    if not job_item:
        return ""
    parts = []
    vcpus = job_item.get("vcpus")
    if vcpus is not None:
        parts.append(f"{vcpus} vCPU")
    mem = _format_memory_gib(job_item.get("memory_mib"))
    if mem:
        parts.append(mem)
    gpus = job_item.get("gpus")
    if gpus is not None:
        parts.append(f"{gpus} GPU")
    return " | ".join(parts)


def _log_url(log_group: str, log_stream: str) -> str:
    region = aws_region()
    group = log_group.replace("/", "$252F")
    stream = log_stream.replace("/", "$252F")
    return (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#logsV2:log-groups/log-group/{group}/log-events/{stream}"
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
    execution_arn = job_item.get("execution_arn")
    if execution_arn:
        try:
            stop_execution(str(execution_arn), reason="Cancelled by user")
        except Exception as exc:
            print(f"Warning: unable to stop orchestrator: {exc}", file=sys.stderr)

    attempt_ids = attempt_ids_for_item(job_item)

    if attempt_ids:
        client = batch_client()
        client.terminate_job(jobId=str(attempt_ids[-1]), reason="Cancelled by user")
    print(f"Cancelled job {job_name}")
    return 0


def list_jobs_cmd() -> int:
    table = get_table()
    jobs = list_jobs(table)
    if not jobs:
        print("No jobs found")
        return 0

    for job in sorted(jobs, key=lambda item: item.get("created_at", "")):
        mem_mib = job.get("memory_mib")
        mem_gib = None
        if isinstance(mem_mib, (int, float)):
            mem_gib = float(mem_mib) / 1024.0
        mem_display = f"{mem_gib:.1f}GiB" if mem_gib is not None else "?"
        print(
            "{job_name} vcpus={vcpus} mem={mem} gpus={gpus} spot={spot}".format(
                job_name=job.get("job_name"),
                vcpus=job.get("vcpus", "?"),
                mem=mem_display,
                gpus=job.get("gpus", "?"),
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
