"""AWS Batch submission and status helpers."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

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
python3 - <<'PY'
import json
import os
import shlex
import signal
import subprocess
import sys
import threading

INPUT_DIR = os.environ.get("INPUT_DIR", "/workspace/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/output")
OUTPUT_S3 = os.environ.get("OUTPUT_S3", "")
MPI_RANKS = os.environ.get("MPI_RANKS", "1")
PISM_EXECUTABLE = os.environ.get("PISM_EXECUTABLE", "pismr")
PISM_ARGS_RAW = os.environ.get("PISM_ARGS", "")
CHECKPOINT_ENABLED = True
CHECKPOINT_RESUME = True
CHECKPOINT_SYNC_INTERVAL = 600


def sync_outputs() -> None:
    if not OUTPUT_S3:
        return
    subprocess.run(["aws", "s3", "sync", OUTPUT_DIR, OUTPUT_S3], check=True)


def sync_inputs() -> None:
    inputs = json.loads(os.environ.get("INPUTS_JSON", "{}"))
    for name, uri in inputs.items():
        dest = os.path.join(INPUT_DIR, name)
        os.makedirs(dest, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", uri, dest], check=True)


def find_output_file(tokens: list[str]) -> str | None:
    for idx, token in enumerate(tokens):
        if token in {"-o", "--o", "-output"} and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith("-o="):
            return token.split("=", 1)[1]
    return None


def resume_if_possible(tokens: list[str]) -> list[str]:
    if not (CHECKPOINT_ENABLED and CHECKPOINT_RESUME):
        return tokens
    output_file = find_output_file(tokens)
    if not output_file:
        return tokens
    if not output_file.startswith(OUTPUT_DIR + os.sep):
        return tokens
    relpath = output_file[len(OUTPUT_DIR) + 1 :]
    if not relpath or not OUTPUT_S3:
        return tokens
    s3_uri = OUTPUT_S3.rstrip("/") + "/" + relpath
    result = subprocess.run(
        ["aws", "s3", "ls", s3_uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if result.returncode != 0:
        return tokens
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    subprocess.run(["aws", "s3", "cp", s3_uri, output_file], check=True)
    print(f"Resuming from {s3_uri}", flush=True)
    new_tokens: list[str] = []
    replaced_i = False
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "-bootstrap":
            idx += 1
            continue
        if token == "-i" and idx + 1 < len(tokens):
            new_tokens.extend(["-i", output_file])
            idx += 2
            replaced_i = True
            continue
        new_tokens.append(token)
        idx += 1
    if not replaced_i:
        new_tokens.extend(["-i", output_file])
    return new_tokens


os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
sync_inputs()

args_tokens = shlex.split(PISM_ARGS_RAW)
args_tokens = resume_if_possible(args_tokens)

stop_event = threading.Event()


def sync_loop() -> None:
    while not stop_event.wait(CHECKPOINT_SYNC_INTERVAL):
        try:
            sync_outputs()
        except Exception:
            pass


if CHECKPOINT_ENABLED and CHECKPOINT_SYNC_INTERVAL > 0 and OUTPUT_S3:
    threading.Thread(target=sync_loop, daemon=True).start()

proc: subprocess.Popen[str] | None = None
terminated = False


def handle_term(signum, frame) -> None:
    global terminated
    terminated = True
    if proc and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


signal.signal(signal.SIGTERM, handle_term)
signal.signal(signal.SIGINT, handle_term)

cmd = ["mpirun", "-n", str(MPI_RANKS), PISM_EXECUTABLE] + args_tokens
proc = subprocess.Popen(cmd, preexec_fn=os.setsid)
exit_code = proc.wait()

stop_event.set()
try:
    sync_outputs()
except Exception:
    pass

if terminated:
    sys.exit(143)
sys.exit(exit_code)
PY
""".strip()
    return ["bash", "-lc", script]


def build_overrides(
    vcpus: int,
    memory_mib: int,
    mpi_ranks: int,
    gpus: int,
    inputs_json: str,
    output_s3: str,
    pism_args: str,
    pism_executable: str,
    run_id: str,
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
            {"name": "INPUTS_JSON", "value": inputs_json},
            {"name": "OUTPUT_S3", "value": output_s3},
            {"name": "MPI_RANKS", "value": str(mpi_ranks)},
            {"name": "PISM_ARGS", "value": pism_args},
            {"name": "PISM_EXECUTABLE", "value": pism_executable},
        ],
    }


def submit_job(
    run_id: str,
    job_queue: str,
    job_definition: str,
    overrides: Dict[str, object],
    timeout_seconds: int | None = None,
) -> Tuple[str, str]:
    client = batch_client()
    job_name = f"pism-{run_id}"
    payload: Dict[str, object] = {
        "jobName": job_name,
        "jobQueue": job_queue,
        "jobDefinition": job_definition,
        "containerOverrides": overrides,
    }
    if timeout_seconds is not None:
        payload["timeout"] = {"attemptDurationSeconds": int(timeout_seconds)}
    response = client.submit_job(
        **payload,
    )
    return response["jobId"], job_name


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
