"""S3 upload helpers for PISM cloud workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
from urllib.parse import urlparse

import boto3

from .aws import aws_region

def parse_s3_uri(uri: str) -> Tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"Invalid S3 URI: {uri}")
    prefix = parsed.path.lstrip("/").rstrip("/")
    return parsed.netloc, prefix


def upload_path(source: Path, dest_uri: str) -> None:
    if not source.exists():
        raise ValueError(f"Source path not found: {source}")

    bucket, prefix = parse_s3_uri(dest_uri)
    client = boto3.client("s3", region_name=aws_region())

    if source.is_dir():
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source).as_posix()
            key = f"{prefix}/{relative}" if prefix else relative
            client.upload_file(str(path), bucket, key)
    else:
        key = f"{prefix}/{source.name}" if prefix else source.name
        client.upload_file(str(source), bucket, key)


def upload_text(text: str, dest_uri: str, name: str) -> str:
    bucket, prefix = parse_s3_uri(dest_uri)
    key = f"{prefix}/{name}" if prefix else name
    client = boto3.client("s3", region_name=aws_region())
    client.put_object(Bucket=bucket, Key=key, Body=text.encode("utf-8"))
    return f"s3://{bucket}/{key}"
