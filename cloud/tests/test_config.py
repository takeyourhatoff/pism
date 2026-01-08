import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch


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
from pism_cloud.aws import InstanceSpec


class ConfigTests(unittest.TestCase):
    def _write_config(self, content: str) -> str:
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False)
        handle.write(textwrap.dedent(content))
        handle.flush()
        handle.close()
        return handle.name

    @patch("pism_cloud.config.instance_spec")
    def test_load_minimal_config(self, mock_spec) -> None:
        mock_spec.return_value = InstanceSpec(
            instance_type="c7i.4xlarge",
            vcpus=16,
            memory_mib=32768,
            gpus=0,
        )
        path = self._write_config(
            """
            run_id: run-001
            compute:
              backend: aws-batch
              instance: c7i.4xlarge
            inputs:
              base: my-prefix/
            pism:
              args: "--config {{INPUTS.base}}/config.cfg"
            """
        )
        configs = config.load_configs([path])
        self.assertEqual(len(configs), 1)
        cfg = configs[0]
        self.assertEqual(cfg.run_id, "run-001")
        self.assertEqual(cfg.inputs["base"], "my-prefix")
        self.assertTrue(cfg.compute["use_spot"])
        self.assertEqual(cfg.compute["mpi_ranks"], 16)

    @patch("pism_cloud.config.instance_spec")
    def test_gpu_defaults_mpi_ranks_to_gpus(self, mock_spec) -> None:
        mock_spec.return_value = InstanceSpec(
            instance_type="g5.xlarge",
            vcpus=4,
            memory_mib=16384,
            gpus=1,
        )
        path = self._write_config(
            """
            run_id: gpu-001
            compute:
              backend: aws-batch
              instance: g5.xlarge
            inputs:
              base: s3://bucket/path
            pism:
              args: "--config {{INPUTS.base}}/config.cfg"
            """
        )
        cfg = config.load_configs([path])[0]
        self.assertEqual(cfg.compute["gpus"], 1)
        self.assertEqual(cfg.compute["mpi_ranks"], 1)
        self.assertEqual(cfg.inputs["base"], "s3://bucket/path")

    @patch("pism_cloud.config.instance_spec")
    def test_budget_usd_validation(self, mock_spec) -> None:
        mock_spec.return_value = InstanceSpec(
            instance_type="c7i.4xlarge",
            vcpus=16,
            memory_mib=32768,
            gpus=0,
        )
        path = self._write_config(
            """
            run_id: budget-001
            budget_usd: -1
            compute:
              backend: aws-batch
              instance: c7i.4xlarge
            inputs:
              base: my-prefix
            pism:
              args: "--config {{INPUTS.base}}/config.cfg"
            """
        )
        with self.assertRaises(ValueError):
            config.load_configs([path])

    @patch("pism_cloud.config.instance_spec")
    def test_missing_inputs(self, mock_spec) -> None:
        mock_spec.return_value = InstanceSpec(
            instance_type="c7i.4xlarge",
            vcpus=16,
            memory_mib=32768,
            gpus=0,
        )
        path = self._write_config(
            """
            run_id: missing-inputs
            compute:
              backend: aws-batch
              instance: c7i.4xlarge
            pism:
              args: "--config foo"
            """
        )
        with self.assertRaises(ValueError):
            config.load_configs([path])

    @patch("pism_cloud.config.instance_spec")
    def test_instance_types_rejected(self, mock_spec) -> None:
        mock_spec.return_value = InstanceSpec(
            instance_type="c7i.4xlarge",
            vcpus=16,
            memory_mib=32768,
            gpus=0,
        )
        path = self._write_config(
            """
            run_id: bad-instance-types
            compute:
              backend: aws-batch
              instance_types: [c7i.4xlarge]
            inputs:
              base: my-prefix
            pism:
              args: "--config {{INPUTS.base}}/config.cfg"
            """
        )
        with self.assertRaises(ValueError):
            config.load_configs([path])
