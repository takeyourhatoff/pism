"""Build and publish PISM container images to ECR."""

from __future__ import annotations

import base64
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import boto3


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def ensure_docker() -> None:
    if not shutil.which("docker"):
        raise RuntimeError("docker is required but was not found in PATH")


def buildx_available() -> bool:
    try:
        subprocess.run(
            ["docker", "buildx", "version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    return True


def buildx_driver() -> Optional[str]:
    try:
        result = subprocess.run(
            ["docker", "buildx", "inspect", "--bootstrap"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Driver:"):
            return line.split(":", 1)[1].strip()
    return None


def ensure_buildx_builder() -> bool:
    driver = buildx_driver()
    if driver == "docker-container":
        return True
    if not driver:
        return False
    name = "pism-cloud"
    result = subprocess.run(
        ["docker", "buildx", "create", "--name", name, "--driver", "docker-container", "--use"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode == 0:
        return True
    if "already exists" in result.stderr.lower():
        try:
            subprocess.run(["docker", "buildx", "use", name], check=True)
            subprocess.run(["docker", "buildx", "inspect", "--bootstrap"], check=True)
        except subprocess.CalledProcessError:
            return False
        return True
    return False


def build_cache_dir(dockerfile: Path) -> Path:
    safe_name = dockerfile.name.replace(".", "-")
    return Path.home() / ".cache" / "pism-cloud" / "docker" / safe_name


def ecr_client(region: Optional[str] = None):
    return boto3.client("ecr", region_name=region)


def ensure_ecr_repo(repo_name: str, region: Optional[str] = None) -> str:
    client = ecr_client(region)
    try:
        response = client.describe_repositories(repositoryNames=[repo_name])
        repo = response["repositories"][0]
    except client.exceptions.RepositoryNotFoundException:
        response = client.create_repository(repositoryName=repo_name)
        repo = response["repository"]
    return repo["repositoryUri"]


def ecr_login(registry: str, region: Optional[str] = None) -> None:
    client = ecr_client(region)
    response = client.get_authorization_token()
    auth_data = response["authorizationData"][0]
    token = base64.b64decode(auth_data["authorizationToken"]).decode("utf-8")
    _, password = token.split(":", 1)
    subprocess.run(
        ["docker", "login", "-u", "AWS", "--password-stdin", registry],
        check=True,
        input=password.encode("utf-8"),
    )


def build_image(
    dockerfile: Path,
    image_uri: str,
    context: Path,
    build_args: Optional[dict[str, str]] = None,
) -> None:
    ensure_docker()
    use_buildx = buildx_available() and ensure_buildx_builder()
    if use_buildx:
        cache_dir = build_cache_dir(dockerfile)
        cache_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "buildx",
            "build",
            "--load",
            "-f",
            str(dockerfile),
            "-t",
            image_uri,
            "--cache-from",
            f"type=local,src={cache_dir}",
            "--cache-to",
            f"type=local,dest={cache_dir},mode=max",
        ]
    else:
        command = [
            "docker",
            "build",
            "-f",
            str(dockerfile),
            "-t",
            image_uri,
        ]
    for key, value in (build_args or {}).items():
        command.extend(["--build-arg", f"{key}={value}"])
    command.append(str(context))
    subprocess.run(command, check=True)


def push_image(image_uri: str) -> None:
    ensure_docker()
    subprocess.run(["docker", "push", image_uri], check=True)


def build_and_push(
    repo_name: str,
    dockerfile: Path,
    tag: str,
    context: Path,
    region: Optional[str] = None,
    build_args: Optional[dict[str, str]] = None,
) -> str:
    repo_uri = ensure_ecr_repo(repo_name, region)
    image_uri = f"{repo_uri}:{tag}"
    registry = repo_uri.split("/", 1)[0]

    ecr_login(registry, region)
    build_image(dockerfile, image_uri, context, build_args=build_args)
    push_image(image_uri)
    return image_uri
