# PISM Cloud (AWS Batch)

Minimal, production-shaped tooling to run PISM ensembles as single-node AWS Batch jobs.

## Repository layout

- `cloud/pism_cloud/` - CLI and shared helpers.
- `cloud/infra/cloudformation/` - AWS Batch, IAM, DynamoDB, CloudWatch infra.
- `cloud/dashboard/` - FastAPI dashboard.
- `cloud/examples/` - Example `config.yml`.

## Run this in 30 minutes

### 1) Prerequisites

- AWS account + configured credentials (`aws sts get-caller-identity` should work).
- Python 3.9+.
- Docker (for building images).

### 2) Install the CLI

```
python -m venv .venv
source .venv/bin/activate
pip install -e cloud
```

### 3) Build images and deploy infrastructure

```
pism-cloud deploy --stack-name pism-batch
```

This builds and pushes CPU/GPU images to ECR, then deploys the CloudFormation stack.
The stack creates a VPC, subnets, security group, S3 buckets, Batch queues, job definitions,
a log group, and the DynamoDB table.
ECR repositories (`pism-cpu`, `pism-gpu`) are created automatically if missing.
Use `--vpc-cidr` and `--public-subnet-cidr-a/b` to override the default network ranges.

### 4) Upload inputs

Use the `InputBucketName` output from `pism-cloud deploy` and upload inputs to:

```
s3://<input-bucket>/<run-id>/
```

To prepare and upload the standard Greenland inputs (requires Docker and an activated venv):

```
cloud/examples/std-greenland/prepare_inputs.sh
```

Use `STACK_NAME=<name>` and `PREFIX=<name>` to override the defaults.
This script runs the upstream `preprocess.sh` in a temporary Docker container.

To upload the Haseloff smoke-test input:

```
pism-cloud upload --source test/cases/haseloff/startSMALLablate.nc --prefix haseloff
```

To upload your own inputs:

```
pism-cloud upload --source /path/to/inputs --prefix <run-id>
```

If your stack name differs from `pism-batch`, add `--stack-name <name>`.

### 5) Submit a run

```
pism-cloud submit cloud/examples/config.yml
# or
pism-cloud submit cloud/examples/haseloff.yml
```

If your stack name differs from `pism-batch`, add `--stack-name <name>`.

### 6) Watch status

```
pism-cloud status biis-test-001
```

### 7) Launch the dashboard

```
pism-cloud dashboard --port 8080
```

Open `http://localhost:8080`.

## Configuration

Set `io.input_prefix` and `io.output_prefix` to control the paths inside the buckets
created by `pism-cloud deploy`. If you want to bypass stack outputs entirely, set
`io.input_s3` and `io.output_s3` to full S3 URIs instead.
Set either `compute.instance` or `compute.instance_types` to control instance selection.

Placeholders inside `pism.args` are expanded per job:

- `{{INPUT_DIR}}` -> `/workspace/input`
- `{{OUTPUT_DIR}}` -> `/workspace/output`
- `{{RUN_ID}}`
- `{{MEMBER_ID}}`

Jobs download inputs with `aws s3 sync`, run PISM, then sync outputs to
`s3://<output-bucket>/<output_prefix>/<member-id>/`.

If your container uses `pism` instead of `pismr`, set `pism.executable: pism`.

## Container images

Default Dockerfiles live in `cloud/images/`:

- `cloud/images/Dockerfile.cpu`
- `cloud/images/Dockerfile.gpu`

Build and push both to ECR with:

```
pism-cloud build-images
```

These Dockerfiles build PISM from source; expect longer build times and customize as needed.
For other GPU types, set `--cuda-arch` when building images.

## Example smoke test

The `cloud/examples/haseloff.yml` config runs a small CPU job using the bundled Haseloff
input file. Use it to validate the end-to-end path before running larger ensembles.

## Standard Greenland runs (1-4)

The following configs mirror the manual's standard Greenland runs:

- `cloud/examples/std-greenland-1.yml`
- `cloud/examples/std-greenland-2.yml`
- `cloud/examples/std-greenland-3.yml`
- `cloud/examples/std-greenland-4.yml`

Workflow:

1. Prepare inputs once:

```
cloud/examples/std-greenland/prepare_inputs.sh
```

2. Submit runs 1-3:

```
pism-cloud submit cloud/examples/std-greenland-1.yml
pism-cloud submit cloud/examples/std-greenland-2.yml
pism-cloud submit cloud/examples/std-greenland-3.yml
```

3. After run 2 completes, stage its output for run 4:

```
SOURCE_PREFIX=std-greenland-2 DEST_PREFIX=std-greenland \
  cloud/examples/std-greenland/stage_output.sh
```

Use `STACK_NAME=<name>` or `MEMBER_ID=<id>` to override defaults.

4. Submit run 4:

```
pism-cloud submit cloud/examples/std-greenland-4.yml
```

## Batch queues and job definitions

Queues (required by the CLI):

- `pism-cpu-spot`
- `pism-gpu-spot`
- `pism-cpu-ondemand`

Job definitions:

- `pism-cpu`
- `pism-gpu`

Override these via environment variables:

- `PISM_CPU_SPOT_QUEUE`, `PISM_GPU_SPOT_QUEUE`, `PISM_CPU_ONDEMAND_QUEUE`
- `PISM_CPU_JOB_DEFINITION`, `PISM_GPU_JOB_DEFINITION`
- `PISM_RUNS_TABLE` for DynamoDB
- `PISM_LOG_GROUP` for the dashboard log links

## Notes

- GPU on-demand is not configured; GPU runs are Spot-only for now.
- Ensure the container images include `aws` CLI for S3 sync.
- One job is forced per instance by requesting full instance vCPU/memory.
