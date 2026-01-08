# PISM Cloud (AWS Batch)

Minimal, production-shaped tooling to run PISM batch jobs as single-node AWS Batch jobs.

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
a log group, and the DynamoDB table. ECR repositories (`pism-cpu`, `pism-gpu`) are created
automatically if missing. Use `--vpc-cidr` and `--public-subnet-cidr-a/b` to override
the default network ranges. Spot defaults to `SPOT_PRICE_CAPACITY_OPTIMIZED`; override
with `--spot-allocation-strategy SPOT_CAPACITY_OPTIMIZED` for fewer interruptions.

### 4) Upload inputs

The stack creates an input bucket (printed as `InputBucketName` after deploy).
You do not need to type the bucket name; use `pism-cloud upload` and choose a prefix.
Inputs live at:

```
s3://<input-bucket>/<prefix>/
```

Use the same `<prefix>` value in your config under `inputs` (see examples below).

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
pism-cloud upload --source /path/to/inputs --prefix <inputs-prefix>
```

If your stack name differs from `pism-batch`, add `--stack-name <name>`.

### 5) Submit a run

```
pism-cloud submit cloud/examples/config.yml
# or
pism-cloud submit cloud/examples/haseloff.yml
```

If your stack name differs from `pism-batch`, add `--stack-name <name>`.

To submit multiple jobs in one command, pass multiple config files or use a
multi-document YAML (each document is one job). Each document should use a unique
`run_id`.

```
pism-cloud submit configs/member-1.yml configs/member-2.yml
```

```
pism-cloud submit configs/run.yml
```

```yaml
run_id: biis-run-0001
compute:
  backend: aws-batch
  instance: c7i.4xlarge
inputs:
  base: biis
pism:
  args: >
    --config {{INPUTS.base}}/member-0001.cfg
    --o {{OUTPUT_DIR}}/out-0001.nc
---
run_id: biis-run-0002
compute:
  backend: aws-batch
  instance: c7i.4xlarge
inputs:
  base: biis
pism:
  args: >
    --config {{INPUTS.base}}/member-0002.cfg
    --o {{OUTPUT_DIR}}/out-0002.nc
```

### 6) Watch status

```
pism-cloud status biis-test-001
```

Status prints a progress bar, ETA, and estimated total cost once logs are available.
If `budget_usd` is set and the estimate exceeds the budget by ~20% after at least
15 minutes, the run is cancelled.
ETA is only available when PISM logs numeric model years (not calendar dates).

### 7) Launch the dashboard

```
pism-cloud dashboard --port 8080
```

Open `http://localhost:8080`.

## Configuration

- `run_id` identifies a run in the dashboard. Each YAML document is one job, so every
  document must use a unique `run_id`; submissions fail if the run already exists.
- `inputs` is a map of input names to S3 prefixes or full S3 URIs. Prefixes are resolved
  under the input bucket created by `pism-cloud deploy`.
- Outputs always land under `s3://<output-bucket>/<run_id>/`.
- `compute.instance` controls instance selection.

Set `budget_usd` to enable an automatic safety timeout. The CLI converts the budget
into a job timeout using the current on-demand or spot hourly rate for the instance
type and passes that timeout to AWS Batch. This guards against runaway jobs, but
does not include queue time or data transfer costs. The ETA-based budget guard only
acts when ETA is available.

Checkpointing is always enabled. The wrapper injects `-checkpoint_interval 0.1667`
(10 minutes wall clock) unless you already set it, syncs outputs every 10 minutes, and
handles SIGTERM so PISM writes its last model state to the `-o` output file before the
container exits. On retry, the wrapper picks the newest of the checkpoint file
(`<output>_checkpoint`) or the `-o` file in S3, replaces any `-i` argument with that file,
and drops `-bootstrap`. The output file must live under `{{OUTPUT_DIR}}`.

Placeholders inside `pism.args` are expanded per job:

- `{{INPUT_DIR}}` -> `/workspace/input`
- `{{OUTPUT_DIR}}` -> `/workspace/output`
- `{{RUN_ID}}`
- `{{INPUTS.<name>}}` -> `/workspace/input/<name>`

Jobs download inputs with `aws s3 sync`, run PISM, then sync outputs to
`s3://<output-bucket>/<run_id>/`.

If your container uses `pism` instead of `pismr`, set `pism.executable: pism`.

Jobs are tagged with `PismRunId`, `PismInstanceType`, and `PismSpot` for cost allocation.

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

## Minimal config schema

```yaml
run_id: <unique-run-id>
budget_usd: 50

compute:
  backend: aws-batch
  instance: c7i.4xlarge | hpc6a.48xlarge | g5.xlarge | <any AWS instance type>
  # use_spot: true | false

inputs:
  <name>: <prefix-or-s3-uri>

pism:
  # executable: pismr | pism
  args: >
    --config {{INPUTS.<name>}}/config.cfg
    # Optional: override checkpoint interval
    # --checkpoint_interval 0.25
    --o {{OUTPUT_DIR}}/out.nc
```

## Example smoke test

The `cloud/examples/haseloff.yml` config runs a small CPU job using the bundled Haseloff
input file. Use it to validate the end-to-end path before running larger batches.

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

Use `STACK_NAME=<name>` to override the defaults.

4. Submit run 4:

```
pism-cloud submit cloud/examples/std-greenland-4.yml
```

## Batch queues and job definitions

Queues (required by the CLI):

- `pism-cpu-spot`
- `pism-gpu-spot`
- `pism-cpu-ondemand`
- `pism-gpu-ondemand`

Job definitions:

- `pism-cpu`
- `pism-gpu`

Override these via environment variables:

- `PISM_CPU_SPOT_QUEUE`, `PISM_GPU_SPOT_QUEUE`, `PISM_CPU_ONDEMAND_QUEUE`, `PISM_GPU_ONDEMAND_QUEUE`
- `PISM_CPU_JOB_DEFINITION`, `PISM_GPU_JOB_DEFINITION`
- `PISM_RUNS_TABLE` for DynamoDB
- `PISM_LOG_GROUP` for the dashboard log links
