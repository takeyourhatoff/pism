"""NetCDF preview helpers for the dashboard."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import boto3

from .aws import aws_region
from .s3 import parse_s3_uri

DEFAULT_VARIABLES = (
    "thk",
    "usurf",
    "topg",
    "velsurf_mag",
    "mask",
    "ice_mask",
)
MAX_PREVIEW_DIM = 512


class PreviewUnavailable(RuntimeError):
    """Raised when no previewable NetCDF output exists yet."""


class PreviewDependencyError(RuntimeError):
    """Raised when preview dependencies are missing."""


@dataclass(frozen=True)
class NetCDFSource:
    bucket: str
    key: str
    etag: str | None
    last_modified: datetime | None
    size_bytes: int | None


def _cache_root() -> Path:
    root = os.environ.get("PISM_CLOUD_PREVIEW_CACHE", "")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "pism-cloud" / "dashboard"


def _cache_dir(bucket: str, key: str) -> Path:
    digest = hashlib.sha256(f"{bucket}/{key}".encode("utf-8")).hexdigest()
    return _cache_root() / digest


def _load_cache_meta(path: Path) -> Dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_cache_meta(path: Path, meta: Dict[str, object]) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def _meta_matches(meta: Dict[str, object] | None, source: NetCDFSource) -> bool:
    if not meta:
        return False
    if meta.get("etag") and source.etag and meta["etag"] != source.etag:
        return False
    if meta.get("size_bytes") and source.size_bytes is not None:
        if int(meta["size_bytes"]) != int(source.size_bytes):
            return False
    if meta.get("last_modified") and source.last_modified is not None:
        if meta["last_modified"] != source.last_modified.isoformat():
            return False
    return True


def _require_preview_deps() -> tuple[object, object, object, object]:
    try:
        import numpy as np
        from netCDF4 import Dataset
        from PIL import Image, ImageOps
    except ImportError as exc:
        raise PreviewDependencyError(
            "NetCDF preview requires numpy, netCDF4, and pillow."
        ) from exc
    return np, Dataset, Image, ImageOps


def _list_netcdf_objects(bucket: str, prefix: str) -> List[Dict[str, object]]:
    s3 = boto3.client("s3", region_name=aws_region())
    objects: List[Dict[str, object]] = []
    kwargs = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
    while True:
        response = s3.list_objects_v2(**kwargs)
        for entry in response.get("Contents", []):
            key = str(entry.get("Key", ""))
            if key.endswith(".nc"):
                objects.append(entry)
        if not response.get("IsTruncated"):
            break
        kwargs["ContinuationToken"] = response.get("NextContinuationToken")
    return objects


def _select_netcdf_source(output_s3: str) -> NetCDFSource | None:
    parsed = parse_s3_uri(output_s3)
    bucket, prefix = parsed
    objects = _list_netcdf_objects(bucket, f"{prefix}/" if prefix else "")
    if not objects:
        return None
    def last_modified_seconds(entry: Dict[str, object]) -> float:
        value = entry.get("LastModified")
        if isinstance(value, datetime):
            return value.timestamp()
        return 0.0

    chosen = max(objects, key=last_modified_seconds)
    return NetCDFSource(
        bucket=bucket,
        key=str(chosen.get("Key", "")),
        etag=str(chosen.get("ETag", "")).strip('"') or None,
        last_modified=chosen.get("LastModified"),
        size_bytes=int(chosen.get("Size")) if chosen.get("Size") is not None else None,
    )


def _ensure_local_source(source: NetCDFSource) -> tuple[Path, Dict[str, object]]:
    cache_dir = _cache_dir(source.bucket, source.key)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "data.nc"
    meta_path = cache_dir / "meta.json"
    cached = _load_cache_meta(meta_path)
    if _meta_matches(cached, source) and data_path.exists():
        return data_path, cached or {}

    for entry in cache_dir.iterdir():
        if entry.is_file():
            entry.unlink()

    s3 = boto3.client("s3", region_name=aws_region())
    s3.download_file(source.bucket, source.key, str(data_path))
    meta = {
        "etag": source.etag,
        "last_modified": source.last_modified.isoformat() if source.last_modified else "",
        "size_bytes": source.size_bytes,
    }
    _write_cache_meta(meta_path, meta)
    return data_path, meta


def _list_preview_variables(path: Path) -> List[str]:
    _np, Dataset, _Image, _ImageOps = _require_preview_deps()
    variables: List[str] = []
    with Dataset(path, "r") as dataset:
        for name, var in dataset.variables.items():
            if getattr(var, "ndim", 0) < 2:
                continue
            variables.append(str(name))
    variables.sort()
    return variables


def _default_variable(variables: Iterable[str]) -> str | None:
    for name in DEFAULT_VARIABLES:
        if name in variables:
            return name
    for name in variables:
        return name
    return None


def _sanitize_var(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def preview_meta(output_s3: str) -> Dict[str, object]:
    _require_preview_deps()
    source = _select_netcdf_source(output_s3)
    if not source:
        return {"available": False, "message": "No NetCDF outputs yet."}
    local_path, _ = _ensure_local_source(source)
    variables = _list_preview_variables(local_path)
    if not variables:
        return {"available": False, "message": "No 2D fields found in NetCDF output."}
    default_var = _default_variable(variables)
    return {
        "available": True,
        "variables": variables,
        "default_var": default_var,
        "source_key": source.key,
    }


def _extract_slice(dataset, var_name: str):
    var = dataset.variables[var_name]
    if var.ndim < 2:
        raise PreviewUnavailable(f"{var_name} has fewer than 2 dimensions")
    index = []
    for idx, size in enumerate(var.shape):
        if idx < var.ndim - 2:
            index.append(max(0, size - 1))
        else:
            index.append(slice(None))
    return var[tuple(index)]


def _downsample(data, max_dim: int):
    height, width = data.shape
    stride_y = max(1, int(math.ceil(height / max_dim)))
    stride_x = max(1, int(math.ceil(width / max_dim)))
    return data[::stride_y, ::stride_x]


def _render_image(array, Image, ImageOps):
    import numpy as np

    if array.ndim != 2:
        raise PreviewUnavailable("Selected field is not 2D after slicing")
    data = np.array(array, dtype=float)
    if isinstance(array, np.ma.MaskedArray):
        data = np.ma.filled(array, np.nan)

    data = _downsample(data, MAX_PREVIEW_DIM)
    finite = np.isfinite(data)
    if not finite.any():
        data = np.zeros_like(data)
        finite = np.isfinite(data)

    lower = float(np.nanpercentile(data, 2)) if finite.any() else 0.0
    upper = float(np.nanpercentile(data, 98)) if finite.any() else 1.0
    if upper <= lower:
        upper = lower + 1.0

    scaled = (data - lower) / (upper - lower)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[~finite] = 0.0
    img = Image.fromarray((scaled * 255).astype("uint8"), mode="L")
    img = ImageOps.colorize(img, black="#0b1d3a", white="#e7f0ff")
    return img


def render_preview_png(output_s3: str, var_name: str | None = None) -> bytes:
    _np, Dataset, Image, ImageOps = _require_preview_deps()
    source = _select_netcdf_source(output_s3)
    if not source:
        raise PreviewUnavailable("No NetCDF outputs yet.")
    local_path, meta = _ensure_local_source(source)
    variables = _list_preview_variables(local_path)
    if not variables:
        raise PreviewUnavailable("No 2D fields found in NetCDF output.")

    if var_name not in variables:
        var_name = _default_variable(variables)
    if not var_name:
        raise PreviewUnavailable("No previewable variables found.")

    cache_dir = _cache_dir(source.bucket, source.key)
    png_name = f"preview_{_sanitize_var(var_name)}.png"
    png_path = cache_dir / png_name
    if _meta_matches(meta, source) and png_path.exists():
        return png_path.read_bytes()

    with Dataset(local_path, "r") as dataset:
        array = _extract_slice(dataset, var_name)
    img = _render_image(array, Image, ImageOps)
    img.save(png_path, format="PNG")
    return png_path.read_bytes()
