"""CloudFormation deployment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import boto3


def deploy_stack(
    stack_name: str,
    template_path: Path,
    parameters: Dict[str, str],
) -> Dict[str, str]:
    client = boto3.client("cloudformation")
    template_body = template_path.read_text(encoding="utf-8")

    params = [
        {"ParameterKey": key, "ParameterValue": value}
        for key, value in parameters.items()
    ]

    try:
        client.describe_stacks(StackName=stack_name)
        action = "update"
    except client.exceptions.ClientError as exc:
        if "does not exist" in str(exc):
            action = "create"
        else:
            raise

    if action == "create":
        client.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Parameters=params,
            Capabilities=["CAPABILITY_NAMED_IAM"],
        )
        waiter = client.get_waiter("stack_create_complete")
        waiter.wait(StackName=stack_name)
    else:
        try:
            client.update_stack(
                StackName=stack_name,
                TemplateBody=template_body,
                Parameters=params,
                Capabilities=["CAPABILITY_NAMED_IAM"],
            )
            waiter = client.get_waiter("stack_update_complete")
            waiter.wait(StackName=stack_name)
        except client.exceptions.ClientError as exc:
            if "No updates are to be performed" not in str(exc):
                raise

    response = client.describe_stacks(StackName=stack_name)
    outputs = response["Stacks"][0].get("Outputs", [])
    return {item["OutputKey"]: item["OutputValue"] for item in outputs}


def stack_outputs(stack_name: str) -> Dict[str, str]:
    client = boto3.client("cloudformation")
    response = client.describe_stacks(StackName=stack_name)
    outputs = response["Stacks"][0].get("Outputs", [])
    return {item["OutputKey"]: item["OutputValue"] for item in outputs}
