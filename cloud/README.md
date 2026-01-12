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
- Python 3.13+.
- Docker (for building images).

### 2) Install the CLI

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e cloud
```

### 3) Build images and deploy infrastructure

```
pism-cloud deploy --stack-name pism-batch
```

This builds and pushes CPU/GPU images to ECR, then deploys the CloudFormation stack.
The stack creates a VPC, subnets, security group, S3 buckets, Batch queues, job definitions,
a log group, the DynamoDB table, and a Step Functions state machine for spot-safe retries.
ECR repositories (`pism-cpu`, `pism-gpu`) are created
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

### 5) Submit a job

```
pism-cloud submit cloud/examples/config.yml
# or
pism-cloud submit cloud/examples/haseloff.yml
```

If your stack name differs from `pism-batch`, add `--stack-name <name>`.

To submit multiple jobs in one command, pass multiple config files or use a
multi-document YAML (each document is one job). Each document should use a unique
`job_name`.

```
pism-cloud submit configs/job-1.yml configs/job-2.yml
```

```
pism-cloud submit configs/jobs.yml
```

```yaml
job_name: biis-job-0001
inputs:
  base: biis
pism:
  args: >
    --config {{INPUTS.base}}/job-0001.cfg
    --o {{OUTPUT_DIR}}/out-0001.nc
---
job_name: biis-job-0002
inputs:
  base: biis
pism:
  args: >
    --config {{INPUTS.base}}/job-0002.cfg
    --o {{OUTPUT_DIR}}/out-0002.nc
```

### 6) Watch status

```
pism-cloud status biis-test-001
```

Status prints a progress bar, ETA, and estimated total cost once logs are available.
If `budget_usd` is set and the estimate exceeds the budget by ~20% after at least
15 minutes, the job is cancelled.
ETA is available when PISM logs a `* Run time:` line plus `S ...` progress lines
(numeric years or calendar dates).

### 7) Launch the dashboard

Build the React UI once (re-run after frontend changes):

```
cd cloud/dashboard-ui
npm install
npm run build
```

Then start the FastAPI dashboard:

```
pism-cloud dashboard --port 8080
```

Open `http://localhost:8080`.

## Configuration

- `job_name` identifies a job in the dashboard. Each YAML document is one job, so every
  document must use a unique `job_name`; submissions fail if the job already exists.
- `inputs` is a map of input names to S3 prefixes. Prefixes are resolved under the input
  bucket created by `pism-cloud deploy`; do not include `s3://` or the bucket name.
- Outputs always land under `s3://<output-bucket>/<job_name>/`.
- `compute` is optional; omit it for a default CPU Spot job.
- `compute.accelerator` picks `cpu` (default) or `gpu`.
- Set `compute.use_spot: false` to force on-demand.
- Resources are taken from the Batch job definitions (`pism-cpu`, `pism-gpu`). If you
  change instance types in the compute environments, update the job definition
  resource requirements in the CloudFormation template so they match.
- MPI ranks default to the selected GPU count (1 rank per GPU), or the selected vCPU
  count for CPU jobs.

Set `budget_usd` to enable the budget guard. Once the job is running, the CLI and
dashboard estimate cost using the actual instance type resolved from Batch/ECS/EC2
metadata and cancel the job if the ETA-based estimate exceeds the budget by ~20%
after at least 15 minutes. Queue time and data transfer are not included.

Checkpointing is always enabled. The wrapper injects `-checkpoint_interval 0.1667`
(10 minutes wall clock) unless you already set it, syncs outputs every 10 minutes, and
handles SIGTERM so PISM writes its last model state to the `-o` output file before the
container exits. When SIGTERM arrives, the wrapper uploads a
`pism-cloud-resume.json` marker to the output prefix and exits 0 so the Batch attempt
is marked as succeeded. The Step Functions orchestrator checks for this marker after
each attempt and resubmits if it exists. On resume, the wrapper picks the newest of the
checkpoint file (`<output>_checkpoint`) or the `-o` file in S3, replaces any `-i`
argument with that file, and drops `-bootstrap`. The output file must live under
`{{OUTPUT_DIR}}`.

Placeholders inside `pism.args` are expanded per job:

- `{{INPUT_DIR}}` -> `/workspace/input`
- `{{OUTPUT_DIR}}` -> `/workspace/output`
- `{{JOB_NAME}}`
- `{{INPUTS.<name>}}` -> `/workspace/input/<name>`

Jobs download inputs with `aws s3 sync`, run PISM, then sync outputs to
`s3://<output-bucket>/<job_name>/`.

If your container uses `pism` instead of `pismr`, set `pism.executable: pism`.

Jobs are tagged with `PismJobName` and `PismSpot` for cost allocation.
Spot interruptions are handled by a Step Functions orchestrator that re-submits attempts
when the `pism-cloud-resume.json` marker is present.

## Container images

Default Dockerfiles live in `cloud/images/`:

- `cloud/images/Dockerfile.cpu`
- `cloud/images/Dockerfile.gpu`

Build and push both to ECR with:

```
pism-cloud build-images
```

These Dockerfiles build PISM from source; expect longer build times and customize as needed.
If Docker Buildx is available, `pism-cloud` uses a local cache under
`~/.cache/pism-cloud/docker/` to speed up rebuilds.
For other GPU types, set `--cuda-arch` when building images.

## Minimal config schema

```yaml
job_name: <unique-job-name>
budget_usd: 50

# compute:
#   accelerator: cpu | gpu
#   use_spot: true | false

inputs:
  <name>: <prefix>

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

## Standard Greenland jobs (1-4)

The following configs mirror the manual's standard Greenland jobs:

- `cloud/examples/std-greenland-1.yml`
- `cloud/examples/std-greenland-2.yml`
- `cloud/examples/std-greenland-3.yml`
- `cloud/examples/std-greenland-4.yml`

Workflow:

1. Prepare inputs once:

```
cloud/examples/std-greenland/prepare_inputs.sh
```

2. Submit jobs 1-3:

```
pism-cloud submit cloud/examples/std-greenland-1.yml
pism-cloud submit cloud/examples/std-greenland-2.yml
pism-cloud submit cloud/examples/std-greenland-3.yml
```

3. After job 2 completes, stage its output for job 4:

```
SOURCE_JOB=std-greenland-2 DEST_PREFIX=std-greenland \
  cloud/examples/std-greenland/stage_output.sh
```

Use `STACK_NAME=<name>` to override the defaults.

4. Submit job 4:

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

Retries are managed by the orchestrator, so the Batch job definitions use a single attempt.

Override these via environment variables:

- `PISM_CPU_SPOT_QUEUE`, `PISM_GPU_SPOT_QUEUE`, `PISM_CPU_ONDEMAND_QUEUE`, `PISM_GPU_ONDEMAND_QUEUE`
- `PISM_CPU_JOB_DEFINITION`, `PISM_GPU_JOB_DEFINITION`
- `PISM_JOBS_TABLE` for DynamoDB
- `PISM_LOG_GROUP` for the dashboard log links
