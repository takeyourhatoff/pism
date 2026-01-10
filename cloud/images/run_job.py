#!/usr/bin/env python3
"""Run a PISM job inside AWS Batch with checkpoint sync and resume."""

from __future__ import annotations

import datetime
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
PISM_ARGS_S3 = os.environ.get("PISM_ARGS_S3", "")
CHECKPOINT_SYNC_INTERVAL = 600
CHECKPOINT_INTERVAL_HOURS = 0.1667
RESUME_MARKER_NAME = "pism-cloud-resume.json"


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


def load_args_from_s3() -> str:
    if not PISM_ARGS_S3:
        raise RuntimeError("PISM_ARGS_S3 is required")
    result = subprocess.run(
        ["aws", "s3", "cp", PISM_ARGS_S3, "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if message:
            raise RuntimeError(f"Args download failed for {PISM_ARGS_S3}: {message}")
        raise RuntimeError(f"Args download failed for {PISM_ARGS_S3}")
    args_text = result.stdout.strip()
    if not args_text:
        raise RuntimeError(f"Args file {PISM_ARGS_S3} is empty")
    return args_text


def resume_marker_uri() -> str | None:
    if not OUTPUT_S3:
        return None
    return OUTPUT_S3.rstrip("/") + "/" + RESUME_MARKER_NAME


def clear_resume_marker() -> None:
    uri = resume_marker_uri()
    if not uri:
        return
    subprocess.run(["aws", "s3", "rm", uri], check=False)


def write_resume_marker(reason: str) -> None:
    uri = resume_marker_uri()
    if not uri:
        return
    payload = {
        "reason": reason,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "job_name": os.environ.get("JOB_NAME", ""),
    }
    subprocess.run(
        ["aws", "s3", "cp", "-", uri],
        input=json.dumps(payload),
        text=True,
        check=False,
    )


def find_output_file(tokens: list[str]) -> str | None:
    for idx, token in enumerate(tokens):
        if token in {"-o", "--o", "-output"} and idx + 1 < len(tokens):
            return tokens[idx + 1]
        if token.startswith("-o="):
            return token.split("=", 1)[1]
    return None


def _s3_object_mtime(uri: str) -> float | None:
    if not uri or not uri.startswith("s3://"):
        return None
    path = uri[len("s3://") :]
    if "/" not in path:
        return None
    bucket, key = path.split("/", 1)
    result = subprocess.run(
        [
            "aws",
            "s3api",
            "list-objects-v2",
            "--bucket",
            bucket,
            "--prefix",
            key,
            "--max-items",
            "10",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if message:
            print(f"Resume lookup failed for {uri}: {message}", flush=True)
        return None
    try:
        payload = json.loads(result.stdout)
        contents = payload.get("Contents", [])
        for entry in contents:
            if entry.get("Key") == key:
                last_modified = entry.get("LastModified", "")
                if not last_modified:
                    return None
                return datetime.datetime.fromisoformat(last_modified.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None
    return None


def _checkpoint_path(output_file: str) -> str:
    root, ext = os.path.splitext(output_file)
    if ext:
        return root + "_checkpoint" + ext
    return output_file + "_checkpoint"


def resume_if_possible(tokens: list[str]) -> list[str]:
    output_file = find_output_file(tokens)
    if not output_file:
        return tokens
    if not output_file.startswith(OUTPUT_DIR + os.sep):
        output_file = os.path.join(OUTPUT_DIR, output_file)
    relpath = output_file[len(OUTPUT_DIR) + 1 :]
    if not relpath or not OUTPUT_S3:
        return tokens
    output_uri = OUTPUT_S3.rstrip("/") + "/" + relpath
    checkpoint_file = _checkpoint_path(output_file)
    checkpoint_rel = checkpoint_file[len(OUTPUT_DIR) + 1 :]
    checkpoint_uri = OUTPUT_S3.rstrip("/") + "/" + checkpoint_rel

    output_mtime = _s3_object_mtime(output_uri)
    checkpoint_mtime = _s3_object_mtime(checkpoint_uri)
    resume_uri = None
    local_file = None
    if output_mtime is None and checkpoint_mtime is None:
        return tokens
    if checkpoint_mtime is None or (output_mtime is not None and output_mtime >= checkpoint_mtime):
        resume_uri = output_uri
        local_file = output_file
    else:
        resume_uri = checkpoint_uri
        local_file = checkpoint_file
    os.makedirs(os.path.dirname(local_file), exist_ok=True)
    result = subprocess.run(["aws", "s3", "cp", resume_uri, local_file], check=False)
    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        if message:
            print(f"Resume download failed for {resume_uri}: {message}", flush=True)
        return tokens
    print(f"Resuming from {resume_uri}", flush=True)
    new_tokens: list[str] = []
    replaced_i = False
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token == "-bootstrap":
            idx += 1
            continue
        if token == "-i" and idx + 1 < len(tokens):
            new_tokens.extend(["-i", local_file])
            idx += 2
            replaced_i = True
            continue
        new_tokens.append(token)
        idx += 1
    if not replaced_i:
        new_tokens.extend(["-i", local_file])
    return new_tokens


def main() -> int:
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clear_resume_marker()
    sync_inputs()
    try:
        args_text = load_args_from_s3()
    except RuntimeError as exc:
        print(str(exc), flush=True)
        return 2

    args_tokens = shlex.split(args_text)
    if "-checkpoint_interval" not in args_tokens and "--checkpoint_interval" not in args_tokens:
        args_tokens.extend(["-checkpoint_interval", str(CHECKPOINT_INTERVAL_HOURS)])
    args_tokens = resume_if_possible(args_tokens)

    stop_event = threading.Event()

    def sync_loop() -> None:
        while not stop_event.wait(CHECKPOINT_SYNC_INTERVAL):
            try:
                sync_outputs()
            except Exception:
                pass

    if CHECKPOINT_SYNC_INTERVAL > 0 and OUTPUT_S3:
        threading.Thread(target=sync_loop, daemon=True).start()

    proc: subprocess.Popen[str] | None = None
    terminated = False
    term_reason = ""

    def handle_term(signum, frame) -> None:
        nonlocal terminated, term_reason
        terminated = True
        term_reason = "sigterm"
        if proc and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    mpi_env = os.environ.copy()
    mpi_env.setdefault("OMPI_ALLOW_RUN_AS_ROOT", "1")
    mpi_env.setdefault("OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", "1")
    mpi_host = f"localhost:{MPI_RANKS}"
    cmd = ["mpirun", "--host", mpi_host, "-n", str(MPI_RANKS), PISM_EXECUTABLE] + args_tokens
    proc = subprocess.Popen(cmd, env=mpi_env, preexec_fn=os.setsid)
    exit_code = proc.wait()

    stop_event.set()
    if terminated:
        write_resume_marker(term_reason or "terminated")
        try:
            sync_outputs()
        except Exception:
            pass
        print("SIGTERM received; resume requested.", flush=True)
        return 0
    try:
        sync_outputs()
    except Exception:
        pass
    clear_resume_marker()
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
