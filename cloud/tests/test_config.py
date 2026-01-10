import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path


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

from pism_cloud import config


class ConfigTests(unittest.TestCase):
    def _write_config(self, content: str) -> str:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        handle.write(textwrap.dedent(content))
        handle.flush()
        handle.close()
        return handle.name

    def test_load_minimal_config(self) -> None:
        path = self._write_config(
            """
            job_name: job-001
            inputs:
              base: my-prefix/
            pism:
              args: "--config {{INPUTS.base}}/config.cfg"
            """
        )
        configs = config.load_configs([path])
        self.assertEqual(len(configs), 1)
        cfg = configs[0]
        self.assertEqual(cfg.job_name, "job-001")
        self.assertEqual(cfg.inputs["base"], "my-prefix")
        self.assertTrue(cfg.use_spot)
        self.assertEqual(cfg.accelerator, "cpu")

    def test_gpu_accelerator_sets_mode(self) -> None:
        path = self._write_config(
            """
            job_name: gpu-001
            compute:
              accelerator: gpu
            inputs:
              base: bucket/path
            pism:
              args: "--config {{INPUTS.base}}/config.cfg"
            """
        )
        cfg = config.load_configs([path])[0]
        self.assertEqual(cfg.accelerator, "gpu")
        self.assertEqual(cfg.inputs["base"], "bucket/path")

    def test_compute_extra_keys_ignored(self) -> None:
        path = self._write_config(
            """
            job_name: overrides
            compute:
              vcpus: 8
            inputs:
              base: bucket/path
            pism:
              args: "--config {{INPUTS.base}}/config.cfg"
            """
        )
        cfg = config.load_configs([path])[0]
        self.assertEqual(cfg.accelerator, "cpu")

    def test_budget_usd_validation(self) -> None:
        path = self._write_config(
            """
            job_name: budget-001
            budget_usd: -1
            inputs:
              base: my-prefix
            pism:
              args: "--config {{INPUTS.base}}/config.cfg"
            """
        )
        with self.assertRaises(ValueError):
            config.load_configs([path])

    def test_missing_inputs(self) -> None:
        path = self._write_config(
            """
            job_name: missing-inputs
            pism:
              args: "--config foo"
            """
        )
        with self.assertRaises(ValueError):
            config.load_configs([path])
