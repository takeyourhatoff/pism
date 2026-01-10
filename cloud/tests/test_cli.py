import sys
import types
import unittest
from pathlib import Path


if sys.version_info < (3, 13):
    raise unittest.SkipTest("pism-cloud tests require Python 3.13+")


def _install_boto3_stub() -> None:
    if "boto3" in sys.modules:
        return
    boto3 = types.ModuleType("boto3")
    boto3.client = lambda *args, **kwargs: None
    boto3.resource = lambda *args, **kwargs: None

    class _Session:
        region_name = "us-east-1"

    boto3.session = types.SimpleNamespace(Session=lambda: _Session())
    sys.modules["boto3"] = boto3

    conditions = types.ModuleType("boto3.dynamodb.conditions")

    class _Cond:
        def eq(self, *args, **kwargs):
            return self

    conditions.Attr = lambda *args, **kwargs: _Cond()
    conditions.Key = lambda *args, **kwargs: _Cond()
    sys.modules["boto3.dynamodb"] = types.ModuleType("boto3.dynamodb")
    sys.modules["boto3.dynamodb.conditions"] = conditions


_install_boto3_stub()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pism_cloud.cli import INPUT_DIR, OUTPUT_DIR, render_args_with_inputs


class CliTests(unittest.TestCase):
    def test_render_args_with_inputs(self) -> None:
        rendered = render_args_with_inputs(
            "--in {{INPUTS.base}}/a.nc --o {{OUTPUT_DIR}}/out-{{JOB_NAME}}.nc",
            "job-123",
            ["base"],
        )
        self.assertEqual(
            rendered,
            f"--in {INPUT_DIR}/base/a.nc --o {OUTPUT_DIR}/out-job-123.nc",
        )
