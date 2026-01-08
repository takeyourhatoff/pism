#!/usr/bin/env bash
set -euo pipefail

STACK_NAME=${STACK_NAME:-pism-batch}
SOURCE_PREFIX=${SOURCE_PREFIX:-std-greenland-2}
DEST_PREFIX=${DEST_PREFIX:-std-greenland}
MEMBER_ID=${MEMBER_ID:-0001}
FILENAME=${FILENAME:-g20km_10ka_hy.nc}

python3 - <<'PY'
import os
import boto3

stack_name = os.environ["STACK_NAME"]
source_prefix = os.environ["SOURCE_PREFIX"]
dest_prefix = os.environ["DEST_PREFIX"]
member_id = os.environ["MEMBER_ID"]
filename = os.environ["FILENAME"]

cf = boto3.client("cloudformation")
response = cf.describe_stacks(StackName=stack_name)
outputs = {item["OutputKey"]: item["OutputValue"] for item in response["Stacks"][0].get("Outputs", [])}
input_bucket = outputs.get("InputBucketName")
output_bucket = outputs.get("OutputBucketName")

if not input_bucket or not output_bucket:
    raise SystemExit("InputBucketName/OutputBucketName not found in stack outputs")

source_key = f"{source_prefix}/{member_id}/{filename}"
dest_key = f"{dest_prefix}/{filename}"

s3 = boto3.client("s3")
s3.copy_object(
    Bucket=input_bucket,
    Key=dest_key,
    CopySource={"Bucket": output_bucket, "Key": source_key},
)

print(f"Copied s3://{output_bucket}/{source_key} to s3://{input_bucket}/{dest_key}")
PY
